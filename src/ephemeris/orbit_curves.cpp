#include "ephemeris/orbit_curves.hpp"
#include "ephemeris/jacobi.hpp"
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
    auto coordinates = extract_jacobi(sys.definition(), sys.motions(), sys.jacobi_hierarchy());
    const auto& subsystems = coordinates.subsystems;
    const auto& children = sys.jacobi_hierarchy().children;
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
        int gp = b.definition.parent_idx;
        if (gp < 0 || sys.body(gp).definition.mass_msun <= 0.0) return std::numeric_limits<double>::infinity();
        return glm::length(b.motion.position.value - sys.body(gp).motion.position.value) *
               std::cbrt(b.definition.mass_msun / (3.0 * sys.body(gp).definition.mass_msun));
    };
    for (size_t i = 0; i < sys.size(); ++i) {
        auto body = sys.body(i);
        int parent = body.definition.parent_idx;
        if (parent < 0) continue;
        auto primary = sys.body(parent);
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
        int gp = primary.definition.parent_idx;
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
                             body.definition.color_rgb * 0.5f, static_cast<int>(i));
        if (!outer) continue;
        double gt = angle_at(*outer, gr), ge = outer->eccentricity;
        outer->arc.start = parameter_angle(ge, gt);
        if (ge >= 1.0) {
            double extent = std::max(300.0, outer->semilatus_au / (1.0 + ge) * 10.0);
            double end = std::min(std::acos(-1.0 / ge) * 0.999,
                                 std::acos(std::clamp((outer->semilatus_au / extent - 1.0) / ge, -1.0, 1.0)));
            outer->arc.span = std::max(0.0, end - gt);
        }
        outer->patched = true;
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

}
