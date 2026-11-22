#include "ephemeris/orbit_curves.hpp"
#include "ephemeris/jacobi.hpp"
#include "ephemeris/spice_ephemeris.hpp"
#include "core/math_utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace sf::ephemeris {
namespace {

double parameter_angle(double e, double theta) {
    return e < 0.999 ? std::atan2(std::sqrt(1.0 - e * e) * std::sin(theta), std::cos(theta) + e) : theta;
}

std::optional<OrbitCurve> extract(glm::dvec3 r, glm::dvec3 v, double mu, double scale,
                                 glm::dvec3 focus, glm::vec3 color, int body) {
    double radius = glm::length(r);
    auto h = glm::cross(r, v);
    double h2 = glm::dot(h, h);
    if (!(mu > 0.0 && radius > 0.0 && h2 > 0.0 && scale > 0.0)) return {};
    auto ev = glm::cross(v, h) / mu - r / radius;
    double e = glm::length(ev);
    double p = h2 / mu * scale;
    if (!std::isfinite(e) || !std::isfinite(p) || p <= 0.0) return {};
    auto normal = h / std::sqrt(h2);
    auto ref = std::abs(normal.z) < 0.9 ? glm::dvec3(0.0, 0.0, 1.0) : glm::dvec3(1.0, 0.0, 0.0);
    auto periapsis = e > 1e-10 ? ev / e : glm::normalize(glm::cross(normal, ref));
    OrbitCurve curve;
    curve.periapsis = periapsis;
    curve.transverse = glm::cross(normal, periapsis);
    curve.focus = EclipticPosition{focus};
    curve.semilatus_au = p;
    curve.eccentricity = e;
    curve.color = color;
    curve.body_index = body;
    return curve;
}

double angle_at(const OrbitCurve& curve, glm::dvec3 r) {
    return std::atan2(glm::dot(r, curve.transverse), glm::dot(r, curve.periapsis));
}

}

OrbitArc compute_orbit_arc(double e, double theta, double p, double radius, double hill, double scale) {
    OrbitArc arc;
    arc.outside = std::isfinite(hill) && radius >= hill;
    arc.escape = arc.outside;
    arc.exit_angle = theta;
    if (!arc.outside && std::isfinite(hill) && e > 1e-12) {
        double c = (p / hill - 1.0) / e;
        if (c > -1.0 && c < 1.0) {
            arc.escape = true;
            arc.exit_angle = std::acos(c);
        }
    }
    arc.start = parameter_angle(e, theta);
    if (arc.outside) arc.span = 0.0;
    else if (arc.escape) {
        double end = parameter_angle(e, arc.exit_angle);
        arc.start = -end;
        arc.span = 2.0 * end;
        arc.phase = arc.span > 0.0 ? (parameter_angle(e, theta) + end) / arc.span : 0.0;
    } else if (e >= 1.0) {
        double extent = std::max(300.0 / scale, p / (1.0 + e) * 10.0);
        double end = std::min(std::acos(-1.0 / e) * 0.999,
                             std::acos(std::clamp((p / extent - 1.0) / e, -1.0, 1.0)));
        arc.start = -end;
        arc.span = 2.0 * end;
        arc.phase = arc.span > 0.0 ? (theta + end) / arc.span : 0.0;
    }
    arc.phase = std::clamp(arc.phase, 0.0, 1.0);
    return arc;
}

EclipticPosition OrbitCurve::point(double fraction) const {
    double angle = arc.start + fraction * arc.span;
    if (fraction == 1.0 && arc.span == TWO_PI && eccentricity < 1.0) angle = arc.start;
    double c = std::cos(angle), s = std::sin(angle), e = eccentricity;
    if (e < 0.999) {
        double a = semilatus_au / (1.0 - e * e);
        return EclipticPosition{focus.value + a * (c - e) * periapsis + a * std::sqrt(1.0 - e * e) * s * transverse};
    }
    double r = semilatus_au / std::max(1.0 + e * c, 1e-15);
    return EclipticPosition{focus.value + r * (c * periapsis + s * transverse)};
}

std::vector<OrbitCurve> compute_orbit_curves(const SystemSnapshot& sys) {
    if (sys.empty()) return {};
    const auto& hierarchy = sys.orbit_hierarchy();
    auto coordinates = extract_jacobi(sys.definition(), sys.motions(), hierarchy);
    const auto& subsystems = coordinates.subsystems;
    const auto& children = hierarchy.children;
    std::vector<int> dominant(sys.size(), -1);
    std::vector<OrbitCurve> curves;
    curves.reserve(sys.size() * 3);
    for (size_t i = 0; i < sys.size(); ++i) {
        for (int child : children[i]) {
            if (subsystems[child].mass > 0.0 && (dominant[i] < 0 || subsystems[child].mass > subsystems[dominant[i]].mass)) dominant[i] = child;
        }
    }
    auto hill_radius = [&](int parent) {
        auto b = sys.body(parent);
        int gp = hierarchy.parents[parent];
        if (gp < 0 || sys.body(gp).definition.mass_msun <= 0.0) return std::numeric_limits<double>::infinity();
        return glm::length(b.motion.position.value - sys.body(gp).motion.position.value) *
               std::cbrt(b.definition.mass_msun / (3.0 * sys.body(gp).definition.mass_msun));
    };
    for (size_t i = 0; i < sys.size(); ++i) {
        auto body = sys.body(i);
        int parent = hierarchy.parents[i];
        if (parent < 0) continue;
        const auto& frame = coordinates.frames[i];
        auto effective = frame.inner;
        if (effective.mass <= 0.0) continue;
        double mass = subsystems[i].mass, total = effective.mass + mass;
        double scale = effective.mass / total, mu = G_CONST * total;
        auto r = frame.relative_position;
        auto v = frame.relative_velocity;
        auto bary = frame.barycenter;
        auto extracted = extract(r, v, mu, scale, bary, body.definition.color_rgb, static_cast<int>(i));
        if (!extracted) continue;
        auto curve = *extracted;
        double theta = angle_at(curve, r), p = curve.semilatus_au / scale, e = curve.eccentricity;
        curve.arc = compute_orbit_arc(e, theta, p, glm::length(r), hill_radius(parent), scale);
        if (curve.arc.span > 0.0) curves.push_back(curve);
        int gp = hierarchy.parents[parent];
        if (!curve.arc.escape || gp < 0 || sys.body(gp).definition.mass_msun <= 0.0) continue;
        glm::dvec3 exit_r = r, exit_v = v;
        double dt = 0.0;
        if (!curve.arc.outside) {
            double c = std::cos(curve.arc.exit_angle), s = std::sin(curve.arc.exit_angle);
            exit_r = p / (1.0 + e * c) * (c * curve.periapsis + s * curve.transverse);
            exit_v = std::sqrt(mu / p) * (-s * curve.periapsis + (e + c) * curve.transverse);
            dt = math::conic_time_at_angle(p, e, curve.arc.exit_angle, mu) - math::conic_time_at_angle(p, e, theta, mu);
            if (dt < 0.0 && e < 1.0) {
                double a = p / ((1.0 - e) * (1.0 + e));
                dt += TWO_PI * std::sqrt(a * a * a / mu);
            }
            dt = std::max(0.0, dt);
        }
        auto grandparent = sys.body(gp);
        double gm = grandparent.definition.mass_msun;
        auto propagated = math::propagate_conic_state(effective.position - grandparent.motion.position.value,
                         effective.velocity - grandparent.motion.velocity.value, G_CONST * (gm + effective.mass), dt);
        auto gr = propagated.pos + exit_r, gv = propagated.vel + exit_v;
        auto focus = grandparent.motion.position.value + mass / (gm + mass) * gr;
        auto outer = extract(gr, gv, G_CONST * (gm + mass), gm / (gm + mass), focus,
                             body.definition.color_rgb * (curve.arc.outside ? 1.0f : 0.5f), static_cast<int>(i));
        if (!outer) continue;
        double gt = angle_at(*outer, gr), ge = outer->eccentricity;
        outer->arc.start = parameter_angle(ge, gt);
        if (ge >= 1.0) {
            double extent = std::max(300.0, outer->semilatus_au / (1.0 + ge) * 10.0);
            double end = std::min(std::acos(-1.0 / ge) * 0.999,
                                 std::acos(std::clamp((outer->semilatus_au / extent - 1.0) / ge, -1.0, 1.0)));
            outer->arc.span = std::max(0.0, end - gt);
        }
        outer->patched = !curve.arc.outside;
        if (outer->arc.span > 0.0) curves.push_back(*outer);
    }
    for (size_t parent = 0; parent < sys.size(); ++parent) {
        int child = dominant[parent];
        if (child < 0) continue;
        auto primary = sys.body(parent);
        double pm = primary.definition.mass_msun, cm = subsystems[child].mass, total = pm + cm;
        if (pm <= 0.0 || cm <= 0.0) continue;
        auto r = subsystems[child].position - primary.motion.position.value;
        auto v = subsystems[child].velocity - primary.motion.velocity.value;
        auto bary = (pm * primary.motion.position.value + cm * subsystems[child].position) / total;
        auto extracted = extract(r, v, G_CONST * total, cm / total, bary, primary.definition.color_rgb, static_cast<int>(parent));
        if (!extracted) continue;
        auto curve = *extracted;
        if (curve.eccentricity > 1e-10) {
            curve.periapsis = -curve.periapsis;
            curve.transverse = -curve.transverse;
        }
        curve.arc = compute_orbit_arc(curve.eccentricity, angle_at(curve, primary.motion.position.value - bary),
                                      curve.semilatus_au / (cm / total), glm::length(r),
                                      hill_radius(static_cast<int>(parent)), pm / total);
        if (curve.arc.outside) continue;
        curve.reflex = true;
        curves.push_back(curve);
    }
    return curves;
}

namespace {

bool finite_vector(const glm::dvec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool compute_synodic_basis(const BodyMotion& p1, const BodyMotion& p2,
                           glm::dvec3& e_r, glm::dvec3& e_t, glm::dvec3& e_n, double& dist_au) {
    glm::dvec3 d = p2.position.value - p1.position.value;
    dist_au = glm::length(d);
    if (!std::isfinite(dist_au) || dist_au <= 1e-12) return false;
    e_r = d / dist_au;
    glm::dvec3 dv = p2.velocity.value - p1.velocity.value;
    glm::dvec3 h = glm::cross(d, dv);
    double h_len = glm::length(h);
    if (!std::isfinite(h_len) || h_len <= 1e-15) {
        glm::dvec3 ref = std::abs(e_r.z) < 0.9 ? glm::dvec3(0.0, 0.0, 1.0) : glm::dvec3(0.0, 1.0, 0.0);
        e_n = glm::normalize(glm::cross(e_r, ref));
    } else {
        e_n = h / h_len;
    }
    e_t = glm::normalize(glm::cross(e_n, e_r));
    return true;
}

int find_body_by_spice_id(const SystemDefinition& def, int target_id) {
    for (size_t i = 0; i < def.bodies.size(); ++i) {
        int id = def.bodies[i].spice_id;
        if (id == target_id) return static_cast<int>(i);
        if (target_id > 100 && target_id < 1000 && target_id % 100 == 99 && id == target_id / 100) return static_cast<int>(i);
        if (target_id > 0 && target_id < 10 && id == target_id * 100 + 99) return static_cast<int>(i);
    }
    return -1;
}

} // namespace

SpiceTrajectoryCache::~SpiceTrajectoryCache() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void SpiceTrajectoryCache::ensure_worker() {
    if (!worker_.joinable()) {
        worker_ = std::thread(&SpiceTrajectoryCache::worker_loop, this);
    }
}

void SpiceTrajectoryCache::worker_loop() {
    for (;;) {
        SampleJob job{};
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || !pending_jobs_.empty(); });
            if (stopping_) return;
            job = std::move(pending_jobs_.front());
            pending_jobs_.erase(pending_jobs_.begin());
            if (job.epoch != request_epoch_) continue;
        }

        SampleResult res = run_sample_job(job);

        {
            std::lock_guard lock(mutex_);
            if (stopping_) return;
            if (res.epoch == request_epoch_) {
                completed_results_.push_back(std::move(res));
            }
        }
    }
}

SpiceTrajectoryCache::SampleResult SpiceTrajectoryCache::run_sample_job(const SampleJob& job) {
    SampleResult res{};
    res.epoch = job.epoch;
    res.entry_index = job.entry_index;

    double win_start = job.cov_start;
    double win_end = job.cov_end;
    if (job.rotating) {
        const double half_w = 0.5 * std::max(5.0, static_cast<double>(job.config.lagrange_days)) * 86400.0;
        win_start = std::max(job.cov_start, job.ref_et - half_w);
        win_end = std::min(job.cov_end, job.ref_et + half_w);
    } else if ((job.cov_end - job.cov_start) > 400.0 * 86400.0 && job.config.trail_fade) {
        const double back_s = (static_cast<double>(job.config.trail_days) + 45.0) * 86400.0;
        const double fwd_s = (static_cast<double>(job.config.lead_days) + 45.0) * 86400.0;
        win_start = std::max(job.cov_start, job.ref_et - back_s);
        win_end = std::min(job.cov_end, job.ref_et + fwd_s);
    }

    const double span = std::max(1.0, win_end - win_start);
    const double eps = std::min(1.0, span * 1e-4);
    const double t0 = win_start + eps;
    const double t1 = win_end - eps;
    if (t1 <= t0) return res;

    auto eval_coord = [&](double t, glm::dvec3& out) -> bool {
        BodyMotion bm{};
        if (job.rotating) {
            if (!job.spice->state(job.spice_id, t, bm)) return false;
            BodyMotion m1{}, m2{};
            if (!job.spice->state(job.rot_p1_spice, t, m1) || !job.spice->state(job.rot_p2_spice, t, m2)) return false;
            glm::dvec3 e_r{}, e_t{}, e_n{};
            double dist = 0.0;
            if (!compute_synodic_basis(m1, m2, e_r, e_t, e_n, dist)) return false;
            const glm::dvec3 anchor_pos = (job.rot_anchor_spice == job.rot_p1_spice) ? m1.position.value : m2.position.value;
            const glm::dvec3 rel = bm.position.value - anchor_pos;
            out = glm::dvec3(glm::dot(rel, e_r), glm::dot(rel, e_t), glm::dot(rel, e_n)) / dist;
            return finite_vector(out);
        } else {
            if (!job.spice->state_relative(job.spice_id, job.parent_spice_id, t, bm)) return false;
            out = bm.position.value;
            return finite_vector(out);
        }
    };

    const size_t max_pts = std::clamp<size_t>(job.config.sample_points, 256, 25000);
    const size_t full_coarse = std::clamp<size_t>(max_pts / 3, 96, 1600);

    auto sample_range = [&](double ra, double rb, size_t slice_max_pts,
                            std::vector<double>& dst_ets, std::vector<glm::dvec3>& dst_coords) {
        if (rb <= ra || slice_max_pts < 2) return;
        const double frac_span = std::clamp((rb - ra) / span, 0.0, 1.0);
        const size_t slice_coarse = std::clamp<size_t>(
            static_cast<size_t>(std::ceil(static_cast<double>(full_coarse) * frac_span)), 8, full_coarse);

        std::vector<double> coarse_t(slice_coarse);
        std::vector<glm::dvec3> coarse_p(slice_coarse);
        std::vector<bool> coarse_ok(slice_coarse, false);
        for (size_t k = 0; k < slice_coarse; ++k) {
            const double frac = static_cast<double>(k) / static_cast<double>(slice_coarse - 1);
            coarse_t[k] = ra + frac * (rb - ra);
            coarse_ok[k] = eval_coord(coarse_t[k], coarse_p[k]);
        }

        const size_t start_size = dst_ets.size();
        auto subdivide = [&](auto& self, double ta, const glm::dvec3& pa, double tb, const glm::dvec3& pb, int depth) -> void {
            if (depth >= 5 || (dst_ets.size() - start_size) + 2 >= slice_max_pts) {
                dst_ets.push_back(tb);
                dst_coords.push_back(pb);
                return;
            }
            const double tm = 0.5 * (ta + tb);
            glm::dvec3 pm{};
            if (!eval_coord(tm, pm)) {
                dst_ets.push_back(tb);
                dst_coords.push_back(pb);
                return;
            }
            const glm::dvec3 v1 = pm - pa;
            const glm::dvec3 v2 = pb - pm;
            const double l1 = glm::length(v1);
            const double l2 = glm::length(v2);
            bool need_split = false;
            if (l1 > 1e-12 && l2 > 1e-12) {
                const double cos_ang = glm::dot(v1, v2) / (l1 * l2);
                if (cos_ang < 0.9992) need_split = true;
            }
            if (need_split) {
                self(self, ta, pa, tm, pm, depth + 1);
                self(self, tm, pm, tb, pb, depth + 1);
            } else {
                dst_ets.push_back(tb);
                dst_coords.push_back(pb);
            }
        };

        bool started = false;
        for (size_t k = 0; k < slice_coarse; ++k) {
            if (!coarse_ok[k]) continue;
            if (!started) {
                if (dst_ets.empty() || coarse_t[k] > dst_ets.back() + 1e-6) {
                    dst_ets.push_back(coarse_t[k]);
                    dst_coords.push_back(coarse_p[k]);
                }
                started = true;
            } else {
                subdivide(subdivide, dst_ets.back(), dst_coords.back(), coarse_t[k], coarse_p[k], 0);
            }
        }
    };

    const bool has_overlap = !job.prev_ets.empty() &&
                             win_end > job.prev_start_et &&
                             win_start < job.prev_end_et;

    if (has_overlap) {
        auto it_beg = std::lower_bound(job.prev_ets.begin(), job.prev_ets.end(), t0);
        auto it_end = std::upper_bound(job.prev_ets.begin(), job.prev_ets.end(), t1);
        const size_t keep_beg = static_cast<size_t>(it_beg - job.prev_ets.begin());
        const size_t keep_end = static_cast<size_t>(it_end - job.prev_ets.begin());

        if (keep_beg < keep_end) {
            const size_t kept_count = keep_end - keep_beg;
            const size_t remaining_budget = (max_pts > kept_count) ? (max_pts - kept_count) : 64;
            res.ets.reserve(kept_count + remaining_budget);
            res.coords.reserve(kept_count + remaining_budget);

            if (t0 < job.prev_ets[keep_beg] - 1.0 && win_start < job.prev_start_et) {
                sample_range(t0, job.prev_ets[keep_beg], std::max<size_t>(remaining_budget / 2, 32), res.ets, res.coords);
            }

            for (size_t k = keep_beg; k < keep_end; ++k) {
                if (res.ets.empty() || job.prev_ets[k] > res.ets.back() + 1e-6) {
                    res.ets.push_back(job.prev_ets[k]);
                    res.coords.push_back(job.prev_coords[k]);
                }
            }

            if (t1 > res.ets.back() + 1.0 && win_end > job.prev_end_et) {
                const size_t tail_budget = (max_pts > res.ets.size()) ? (max_pts - res.ets.size()) : 32;
                sample_range(res.ets.back(), t1, std::max<size_t>(tail_budget, 32), res.ets, res.coords);
            }

            res.start_et = win_start;
            res.end_et = win_end;
            return res;
        }
    }

    res.ets.reserve(max_pts);
    res.coords.reserve(max_pts);
    sample_range(t0, t1, max_pts, res.ets, res.coords);
    res.start_et = win_start;
    res.end_et = win_end;
    return res;
}

void SpiceTrajectoryCache::evaluate(const SystemSnapshot& sys, const EphemerisOrbitConfig& config,
                                    std::vector<OrbitPolyline>& out_polylines,
                                    std::vector<bool>& out_covered_bodies) {
    out_polylines.clear();
    out_covered_bodies.assign(sys.size(), false);
    if (sys.empty() || sys.definition().motion_model != MotionModel::Spice) {
        {
            std::lock_guard lock(mutex_);
            ++request_epoch_;
            pending_jobs_.clear();
            completed_results_.clear();
        }
        entries_.clear();
        return;
    }

    auto spice = SpiceEphemeris::active();
    if (!spice) {
        std::string err;
        spice = SpiceEphemeris::open(err);
    }
    if (!spice) return;

    const auto& def = sys.definition();
    double epoch_et = 0.0;
    if (def.spice_epoch_et.has_value() && std::isfinite(*def.spice_epoch_et)) {
        epoch_et = *def.spice_epoch_et;
    } else if (!spice->epoch(def.epoch_utc, epoch_et)) {
        return;
    }
    const double cur_et = epoch_et + units::to_seconds(sys.time()).value;

    const bool structure_changed = (generation_ != sys.generation() ||
                                    definition_revision_ != sys.definition_revision() ||
                                    config_.frame_mode != config.frame_mode ||
                                    config_.rotating_pair != config.rotating_pair ||
                                    std::abs(config_.lagrange_days - config.lagrange_days) > 0.1f ||
                                    config_.sample_points != config.sample_points ||
                                    config_.trail_fade != config.trail_fade);

    constexpr double kFiniteMissionMaxSec = 50.0 * 365.25 * 86400.0;
    const bool is_custom_bsp = !def.custom_bsp_path.empty();

    if (structure_changed) {
        {
            std::lock_guard lock(mutex_);
            ++request_epoch_;
            pending_jobs_.clear();
            completed_results_.clear();
        }
        entries_.clear();
        generation_ = sys.generation();
        definition_revision_ = sys.definition_revision();

        struct PairSpec { int p1; int p2; int anchor; };
        static constexpr PairSpec kPairs[] = {
            {10, 399, 399},
            {399, 301, 399},
            {10, 599, 599},
            {10, 699, 699},
            {699, 606, 606}
        };
        const PairSpec manual_pair = kPairs[std::min<uint32_t>(config.rotating_pair, 4u)];
        const int manual_p1_idx = find_body_by_spice_id(def, manual_pair.p1);
        const int manual_p2_idx = find_body_by_spice_id(def, manual_pair.p2);
        const int manual_anchor_idx = find_body_by_spice_id(def, manual_pair.anchor);

        for (size_t i = 0; i < sys.size(); ++i) {
            const auto& b = def.bodies[i];
            if (b.is_star || b.spice_id == 0) continue;

            const bool finite_mission = b.has_finite_spice_coverage() && (b.spice_end_et - b.spice_start_et) < kFiniteMissionMaxSec;
            const bool use_polyline = b.is_spacecraft || finite_mission || is_custom_bsp || (config.frame_mode == 2);
            if (!use_polyline) continue;

            int parent_idx = sys.reference_parent(i);
            if (parent_idx < 0 || static_cast<size_t>(parent_idx) >= sys.size()) {
                parent_idx = b.parent_idx;
            }
            if (parent_idx < 0 || static_cast<size_t>(parent_idx) >= sys.size() || parent_idx == static_cast<int>(i)) {
                parent_idx = def.star_idx;
            }
            if (parent_idx < 0 || static_cast<size_t>(parent_idx) >= sys.size() || parent_idx == static_cast<int>(i)) {
                continue;
            }

            CachedBody entry{};
            entry.body_index = static_cast<int>(i);
            entry.spice_id = b.spice_id;
            entry.parent_index = parent_idx;
            entry.parent_spice_id = def.bodies[parent_idx].spice_id;

            if (config.frame_mode == 2) {
                if (manual_p1_idx >= 0 && manual_p2_idx >= 0 && manual_anchor_idx >= 0 &&
                    static_cast<int>(i) != manual_p1_idx && static_cast<int>(i) != manual_p2_idx) {
                    entry.rotating = true;
                    entry.rot_p1_idx = manual_p1_idx;
                    entry.rot_p2_idx = manual_p2_idx;
                    entry.rot_anchor_idx = manual_anchor_idx;
                }
            } else if (config.frame_mode == 0 && b.is_spacecraft) {
                if (b.spice_id == -170) {
                    int sun_i = find_body_by_spice_id(def, 10);
                    int earth_i = find_body_by_spice_id(def, 399);
                    if (sun_i >= 0 && earth_i >= 0) {
                        entry.rotating = true;
                        entry.rot_p1_idx = sun_i;
                        entry.rot_p2_idx = earth_i;
                        entry.rot_anchor_idx = earth_i;
                    }
                } else {
                    int gp_idx = sys.reference_parent(parent_idx);
                    if (gp_idx >= 0 && static_cast<size_t>(gp_idx) < sys.size() && gp_idx != parent_idx) {
                        const double m1 = def.bodies[gp_idx].mass_msun;
                        const double m2 = def.bodies[parent_idx].mass_msun;
                        if (m1 > 0.0 && m2 > 0.0) {
                            const double d_p = glm::length(sys.body(parent_idx).motion.position.value - sys.body(gp_idx).motion.position.value);
                            const double r_hill = d_p * std::cbrt(m2 / (3.0 * m1));
                            const double d_sc = glm::length(sys.body(i).motion.position.value - sys.body(parent_idx).motion.position.value);
                            if (r_hill > 0.0 && d_sc >= 0.35 * r_hill && d_sc <= 1.85 * r_hill) {
                                entry.rotating = true;
                                entry.rot_p1_idx = gp_idx;
                                entry.rot_p2_idx = parent_idx;
                                entry.rot_anchor_idx = parent_idx;
                            }
                        }
                    }
                }
            }

            if (entry.rotating) {
                entry.rot_p1_spice = def.bodies[entry.rot_p1_idx].spice_id;
                entry.rot_p2_spice = def.bodies[entry.rot_p2_idx].spice_id;
                entry.rot_anchor_spice = def.bodies[entry.rot_anchor_idx].spice_id;
            }
            entries_.push_back(std::move(entry));
        }
    }
    config_ = config;

    std::vector<SampleResult> completed;
    uint64_t current_epoch = 0;
    {
        std::lock_guard lock(mutex_);
        current_epoch = request_epoch_;
        completed.swap(completed_results_);
    }
    for (auto& res : completed) {
        if (res.epoch == current_epoch && res.entry_index < entries_.size()) {
            auto& entry = entries_[res.entry_index];
            entry.ets = std::move(res.ets);
            entry.coords = std::move(res.coords);
            entry.start_et = res.start_et;
            entry.end_et = res.end_et;
            entry.sampling_in_flight = false;
        }
    }

    std::vector<SampleJob> new_jobs;
    for (size_t idx = 0; idx < entries_.size(); ++idx) {
        auto& entry = entries_[idx];
        if (entry.body_index < 0 || static_cast<size_t>(entry.body_index) >= sys.size()) continue;
        out_covered_bodies[entry.body_index] = true;

        const auto& bview = sys.body(entry.body_index);
        if (config.hide_outside && !bview.render.spice_valid) {
            continue;
        }

        const auto& bdef = bview.definition;
        const double cov_start = bdef.has_finite_spice_coverage() ? bdef.spice_start_et : (cur_et - 180.0 * 86400.0);
        double cov_end = bdef.has_finite_spice_coverage() ? bdef.spice_end_et : (cur_et + 180.0 * 86400.0);
        if (cov_end <= cov_start) cov_end = cov_start + 86400.0;
        const double clamped_et = std::clamp(cur_et, cov_start, cov_end);

        bool need_resample = entry.ets.empty();
        if (!need_resample) {
            const double cached_span = std::max(1.0, entry.end_et - entry.start_et);
            const double margin = std::min(5.0 * 86400.0, 0.25 * cached_span);
            if (clamped_et < entry.start_et + margin && entry.start_et > cov_start + 60.0) need_resample = true;
            if (clamped_et > entry.end_et - margin && entry.end_et < cov_end - 60.0) need_resample = true;
            if (clamped_et < entry.start_et || clamped_et > entry.end_et) need_resample = true;
        }
        if (need_resample && !entry.sampling_in_flight) {
            entry.sampling_in_flight = true;
            SampleJob job{};
            job.epoch = current_epoch;
            job.entry_index = idx;
            job.spice = spice;
            job.spice_id = entry.spice_id;
            job.parent_spice_id = entry.parent_spice_id;
            job.rotating = entry.rotating;
            job.rot_p1_spice = entry.rot_p1_spice;
            job.rot_p2_spice = entry.rot_p2_spice;
            job.rot_anchor_spice = entry.rot_anchor_spice;
            job.cov_start = cov_start;
            job.cov_end = cov_end;
            job.ref_et = clamped_et;
            job.config = config;
            job.prev_start_et = entry.start_et;
            job.prev_end_et = entry.end_et;
            job.prev_ets = entry.ets;
            job.prev_coords = entry.coords;
            new_jobs.push_back(std::move(job));
        }
        if (entry.coords.size() < 2) continue;

        glm::dvec3 rot_er{}, rot_et{}, rot_en{}, anchor_pos{};
        double rot_dist = 1.0;
        if (entry.rotating) {
            const auto& m1 = sys.body(entry.rot_p1_idx).motion;
            const auto& m2 = sys.body(entry.rot_p2_idx).motion;
            if (!compute_synodic_basis(m1, m2, rot_er, rot_et, rot_en, rot_dist)) continue;
            anchor_pos = sys.body(entry.rot_anchor_idx).motion.position.value;
        } else {
            anchor_pos = sys.body(entry.parent_index).motion.position.value;
        }

        auto to_world = [&](const glm::dvec3& c) -> glm::dvec3 {
            if (entry.rotating) {
                return anchor_pos + rot_dist * (c.x * rot_er + c.y * rot_et + c.z * rot_en);
            }
            return anchor_pos + c;
        };

        auto compute_alpha = [&](double t_sample) -> float {
            if (!config.trail_fade) {
                return (t_sample <= cur_et) ? 0.90f : 0.35f;
            }
            const double dt_days = (t_sample - cur_et) / 86400.0;
            if (dt_days <= 0.0) {
                const double trail_d = std::max(0.05, static_cast<double>(config.trail_days));
                if (-dt_days > trail_d) return 0.06f;
                const double u = std::clamp(1.0 + dt_days / trail_d, 0.0, 1.0);
                const double s = u * u * (3.0 - 2.0 * u);
                return static_cast<float>(0.06 + 0.94 * s);
            } else {
                const double lead_d = static_cast<double>(config.lead_days);
                if (lead_d <= 1e-4 || dt_days > lead_d) return 0.06f;
                const double u = std::clamp(1.0 - dt_days / lead_d, 0.0, 1.0);
                const double s = u * u * (3.0 - 2.0 * u);
                return static_cast<float>(0.06 + 0.54 * s);
            }
        };

        OrbitPolyline poly{};
        poly.body_index = entry.body_index;
        poly.color = bdef.color_rgb;
        poly.center = EclipticPosition{anchor_pos};
        const size_t n = entry.coords.size();
        poly.positions.reserve(n + 1);
        poly.alphas.reserve(n + 1);

        bool inserted_now = !bview.render.spice_valid;
        double max_r2 = 0.0;

        for (size_t k = 0; k < n; ++k) {
            const double tk = entry.ets[k];
            if (!inserted_now && tk >= cur_et) {
                const glm::dvec3 cur_pos = bview.motion.position.value;
                poly.positions.push_back(EclipticPosition{cur_pos});
                poly.alphas.push_back(1.0f);
                const glm::dvec3 d = cur_pos - anchor_pos;
                max_r2 = std::max(max_r2, glm::dot(d, d));
                inserted_now = true;
            }
            const glm::dvec3 wp = to_world(entry.coords[k]);
            poly.positions.push_back(EclipticPosition{wp});
            poly.alphas.push_back(compute_alpha(tk));
            const glm::dvec3 d = wp - anchor_pos;
            max_r2 = std::max(max_r2, glm::dot(d, d));
        }

        if (!inserted_now) {
            const glm::dvec3 cur_pos = bview.motion.position.value;
            poly.positions.push_back(EclipticPosition{cur_pos});
            poly.alphas.push_back(1.0f);
            const glm::dvec3 d = cur_pos - anchor_pos;
            max_r2 = std::max(max_r2, glm::dot(d, d));
        }

        poly.extent_au = std::sqrt(max_r2);
        if (poly.positions.size() >= 2) {
            out_polylines.push_back(std::move(poly));
        }
    }

    if (!new_jobs.empty()) {
        ensure_worker();
        {
            std::lock_guard lock(mutex_);
            for (auto& j : new_jobs) {
                pending_jobs_.push_back(std::move(j));
            }
        }
        cv_.notify_one();
    }
}

}
