#include "ephemeris/simulation.hpp"
#include "core/math_utils.hpp"
#include "core/log.hpp"
#include "ephemeris/spice_ephemeris.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace sf::ephemeris {

namespace {

bool finite_vector(glm::dvec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool valid_physical(const BodyPhysicalProperties& values) {
    bool ok = std::isfinite(values.mass.value) && values.mass.value >= 0.0 &&
              std::isfinite(values.radius.value) && values.radius.value > 0.0 &&
              std::isfinite(values.oblateness) && values.oblateness >= 0.0 && values.oblateness < 1.0 &&
              std::isfinite(values.j2) && std::isfinite(values.j4) &&
              std::isfinite(values.rotation_period_hr);
    if (values.update_orbit) {
        ok = ok && std::isfinite(values.a_au) && values.a_au > 0.0 &&
             std::isfinite(values.e) && values.e >= 0.0 && values.e < 1.0 &&
             std::isfinite(values.inc_deg) && std::isfinite(values.omega_node_deg) &&
             std::isfinite(values.arg_peri_deg) && std::isfinite(values.mean_anom_deg);
    }
    return ok;
}

}

CommandResult Simulation::replace_system(LoadedSystem loaded) {
    if (loaded.definition.bodies.empty() || loaded.definition.bodies.size() != loaded.motions.size()) {
        return {false, false, "System definitions and initial states must have matching, nonzero sizes."};
    }
    auto& definition = loaded.definition;
    definition.name_to_idx.clear();
    for (size_t i = 0; i < definition.bodies.size(); ++i) {
        const auto& body = definition.bodies[i];
        const auto& motion = loaded.motions[i];
        if (body.name.empty() || !definition.name_to_idx.emplace(body.name, static_cast<int>(i)).second ||
            body.parent_idx < -1 || body.parent_idx >= static_cast<int>(i) ||
            !valid_physical(BodyPhysicalProperties::from(body)) ||
            !finite_vector(motion.position.value) || !finite_vector(motion.velocity.value) ||
            !std::isfinite(body.radius_au()) || body.radius_au() <= 0.0 ||
            !finite_vector(body.pole_ecl) || !std::isfinite(glm::length(body.pole_ecl)) || glm::length(body.pole_ecl) < 1e-12 ||
            !std::isfinite(body.rotation_period_hr) || !std::isfinite(body.w0_deg) ||
            !std::isfinite(body.lum_eq) || !std::isfinite(body.lum_pole) ||
            !finite_vector(glm::dvec3(body.color_rgb)) || !finite_vector(glm::dvec3(body.color_pole_rgb))) {
            return {false, false, "System contains invalid properties, states, duplicate names, or an unordered hierarchy."};
        }
        if (body.parent_idx >= 0 && body.parent_id != definition.bodies[body.parent_idx].name) {
            return {false, false, "Body parent name and index disagree."};
        }
    }
    if (definition.star_idx < 0 || definition.star_idx >= static_cast<int>(definition.bodies.size())) {
        return {false, false, "System reference body is out of range."};
    }

    Simulation candidate;
    if (definition.motion_model == MotionModel::Spice) {
        std::string error;
        candidate.spice_ = SpiceEphemeris::open(error);
        if (!candidate.spice_) return {false, false, error};
        auto status = candidate.spice_->epoch(definition.epoch_utc, candidate.epoch_et_);
        if (!status) return status;
        std::vector<BodyDefinition> available;
        std::vector<BodyMotion> states;
        for (const auto& body : definition.bodies) {
            if (body.spice_id == 0) return {false, false, "Missing SPICE ID for " + body.name};
            BodyMotion state;
            status = candidate.spice_->state(body.spice_id, candidate.epoch_et_, state);
            if (!status) {
                if (!body.is_spacecraft) return {false, false, body.name + ": " + status.error};
                log::warn("Omitting spacecraft '{}' without SPICE coverage at {}", body.name, definition.epoch_utc);
                continue;
            }
            available.push_back(body);
            states.push_back(state);
        }
        if (available.empty()) return {false, false, "No ephemeris bodies are available at this epoch."};
        definition.bodies = std::move(available);
        loaded.motions = std::move(states);
        definition.name_to_idx.clear();
        for (size_t i = 0; i < definition.bodies.size(); ++i) {
            auto& body = definition.bodies[i];
            body.parent_idx = -1;
            if (!body.parent_id.empty()) {
                auto parent = definition.name_to_idx.find(body.parent_id);
                if (parent == definition.name_to_idx.end()) return {false, false, "Ephemeris body has an unavailable parent: " + body.name};
                body.parent_idx = parent->second;
            }
            definition.name_to_idx[body.name] = static_cast<int>(i);
        }
        definition.star_idx = 0;
        for (size_t i = 0; i < definition.bodies.size(); ++i) if (definition.bodies[i].is_star) {
            definition.star_idx = static_cast<int>(i);
            break;
        }
    }
    auto owned_definition = std::make_shared<const SystemDefinition>(std::move(definition));
    candidate.snapshot_.definition_ = std::move(owned_definition);
    candidate.snapshot_.motions_ = std::move(loaded.motions);
    candidate.snapshot_.render_.resize(candidate.snapshot_.size());
    candidate.snapshot_.revisions_.assign(candidate.snapshot_.size(), BodyRevisions{});
    candidate.snapshot_.generation_ = snapshot_.generation_ + 1;
    candidate.snapshot_.definition_revision_ = snapshot_.definition_revision_ + 1;
    candidate.snapshot_.state_revision_ = snapshot_.state_revision_ + 1;
    candidate.recenter();
    candidate.reset_jacobi();
    candidate.refresh_reference_hierarchy(true);
    if (candidate.snapshot_.definition().motion_model == MotionModel::NBody) {
        candidate.ias15_.reset(candidate.snapshot_.definition(), candidate.snapshot_.motions(), candidate.clock_.time);
    }
    candidate.prepare_render_state(true);
    for (size_t i = 0; i < candidate.snapshot_.size(); ++i) {
        auto body = candidate.snapshot_.body(i);
        if (!finite_vector(body.motion.position.value) || !finite_vector(body.motion.velocity.value) ||
            !std::isfinite(body.render.spin_angle_rad)) return {false, false, "System preparation produced a nonfinite state."};
    }
    snapshot_ = std::move(candidate.snapshot_);
    spice_ = std::move(candidate.spice_);
    epoch_et_ = candidate.epoch_et_;
    kepler_ = std::move(candidate.kepler_);
    ias15_ = std::move(candidate.ias15_);
    hierarchy_tracker_ = std::move(candidate.hierarchy_tracker_);
    hierarchy_elapsed_seconds_ = 0.0;
    clock_.time = units::Years{};
    clock_.direction = 1.0;
    previous_motions_.resize(snapshot_.size());
    return {true, true, {}};
}

CommandResult Simulation::apply(SetPaused command) {
    bool changed = command.value != clock_.paused;
    if (changed && command.value && hierarchy_elapsed_seconds_ > 0.0) refresh_reference_hierarchy();
    clock_.paused = command.value;
    return {true, changed, {}};
}

CommandResult Simulation::apply(SetTimeWarp command) {
    if (!std::isfinite(command.value) || command.value < 1e-6 || command.value > 1e16) {
        return {false, false, "Time warp must be finite and between 1e-6 and 1e16."};
    }
    bool changed = command.value != clock_.warp;
    clock_.warp = command.value;
    return {true, changed, {}};
}

CommandResult Simulation::apply(SetTimeDirection command) {
    if (command.value != 1.0 && command.value != -1.0) {
        return {false, false, "Time direction must be +1.0 or -1.0."};
    }
    bool changed = command.value != clock_.direction;
    clock_.direction = command.value;
    return {true, changed, {}};
}

CommandResult Simulation::apply(ReverseTimeDirection) {
    clock_.direction = -clock_.direction;
    return {true, true, {}};
}

CommandResult Simulation::apply(SetMotionModel command) {
    if (snapshot_.empty()) return {false, false, "No active system."};
    if (command.model == snapshot_.definition().motion_model) return {true, false, {}};

    if (command.model == MotionModel::Spice) {
        if (!spice_) {
            std::string error;
            spice_ = SpiceEphemeris::open(error);
            if (!spice_) return {false, false, "SPICE runtime unavailable: " + error};
        }
        double test_et = epoch_et_ + units::to_seconds(clock_.time).value;
        for (const auto& b : snapshot_.definition().bodies) {
            if (b.spice_id == 0) return {false, false, "Body '" + b.name + "' has no SPICE ID."};
            BodyMotion test_motion;
            auto status = spice_->state(b.spice_id, test_et, test_motion);
            if (!status) return {false, false, b.name + ": " + status.error};
        }
    }

    auto updated = std::make_shared<SystemDefinition>(*snapshot_.definition_);
    updated->motion_model = command.model;
    snapshot_.definition_ = std::move(updated);
    ++snapshot_.definition_revision_;

    if (command.model == MotionModel::Keplerian) {
        reset_jacobi(true);
    } else if (command.model == MotionModel::NBody) {
        ias15_.reset(snapshot_.definition(), snapshot_.motions(), clock_.time);
    } else if (command.model == MotionModel::Spice) {
        double et = epoch_et_ + units::to_seconds(clock_.time).value;
        for (size_t i = 0; i < snapshot_.size(); ++i) {
            spice_->state(snapshot_.definition().bodies[i].spice_id, et, snapshot_.motions_[i]);
        }
        recenter();
    }

    refresh_reference_hierarchy(true);
    prepare_render_state(true);
    return {true, true, {}};
}

CommandResult Simulation::apply(const ApplyBodyPhysicalProperties& command) {
    if (command.system_generation != snapshot_.generation_ || command.body_index < 0 ||
        static_cast<size_t>(command.body_index) >= snapshot_.size()) {
        return {false, false, "The edited body belongs to a different system or no longer exists."};
    }
    auto index = static_cast<size_t>(command.body_index);
    if (command.expected_body_revision != snapshot_.revisions_[index].definition) {
        return {false, false, "The body changed after this draft was opened. Revert to reload its current values."};
    }
    if (!valid_physical(command.values)) {
        return {false, false, "Mass must be nonnegative, radius positive, oblateness in [0, 1), and all values finite."};
    }
    if (command.values.update_orbit && command.expected_reference_parent != -2 &&
        command.expected_reference_parent != snapshot_.reference_parent(index)) {
        return {false, false, "The reference parent changed after this orbit draft was opened. Revert to reload the current orbit."};
    }
    const auto& old = snapshot_.definition_->bodies[index];
    auto values = command.values;
    if (old.is_black_hole) {
        if (values.mass.value <= 0.0) return {false, false, "Black hole mass must be positive."};
        values.radius = units::Kilometres{2.95325008 * values.mass.value};
    }
    double radius_rsun = values.radius.value / SOLAR_RADIUS_KM;
    double radius_au = radius_rsun * SOLAR_RADII_TO_AU * (old.is_black_hole ? 2.5980762 : 1.0);
    if (!valid_physical(values) || !std::isfinite(radius_au) || radius_au <= 0.0) {
        return {false, false, "Physical values exceed the supported numerical range."};
    }
    auto before = BodyPhysicalProperties::from(old);
    if (before == values) return {};

    auto updated = std::make_shared<SystemDefinition>(*snapshot_.definition_);
    auto& body = updated->bodies[index];
    body.mass_msun = values.mass.value;
    body.radius_rsun = radius_rsun;
    body.oblateness = values.oblateness;
    body.j2 = values.j2;
    body.j4 = values.j4;
    body.type = values.type;
    body.is_star = (values.type == "Star");
    body.rotation_period_hr = values.rotation_period_hr;
    body.tidally_locked = values.tidally_locked;

    bool dynamics_changed = (before.mass != values.mass || before.oblateness != values.oblateness ||
                             before.j2 != values.j2 || before.j4 != values.j4 ||
                             (before.radius != values.radius && (before.j2 != 0.0 || before.j4 != 0.0)));

    int orbit_parent = snapshot_.reference_parent(index);
    if (values.update_orbit && orbit_parent >= 0) {
        int parent_idx = orbit_parent;
        double parent_m = snapshot_.definition_->bodies[parent_idx].mass_msun;
        auto cart = math::get_cartesian_from_keplerian(parent_m, values.mass.value,
                                                        values.a_au, values.e,
                                                        values.inc_deg, values.omega_node_deg,
                                                        values.arg_peri_deg, values.mean_anom_deg);
        if (values.equatorial_frame) {
            glm::dvec3 parent_pole = snapshot_.definition_->bodies[parent_idx].pole_ecl;
            auto rotated = math::rotate_equatorial_to_ecliptic(cart.pos, cart.vel, parent_pole);
            cart.pos = rotated.first;
            cart.vel = rotated.second;
        }
        glm::dvec3 parent_pos = snapshot_.motions_[parent_idx].position.value;
        glm::dvec3 parent_vel = snapshot_.motions_[parent_idx].velocity.value;
        glm::dvec3 old_pos = snapshot_.motions_[index].position.value;
        glm::dvec3 old_vel = snapshot_.motions_[index].velocity.value;
        glm::dvec3 new_pos = parent_pos + cart.pos;
        glm::dvec3 new_vel = parent_vel + cart.vel;
        glm::dvec3 delta_pos = new_pos - old_pos;
        glm::dvec3 delta_vel = new_vel - old_vel;

        snapshot_.motions_[index].position = EclipticPosition{new_pos};
        snapshot_.motions_[index].velocity = EclipticVelocity{new_vel};
        body.semi_major_au = values.a_au;
        body.eccentricity = values.e;
        body.inc_deg = values.inc_deg;
        body.omega_node_deg = values.omega_node_deg;
        body.arg_peri_deg = values.arg_peri_deg;
        body.mean_anom_deg = values.mean_anom_deg;
        dynamics_changed = true;

        if (values.bring_moons) {
            double body_m = values.mass.value > 0.0 ? values.mass.value : old.mass_msun;
            double hill_radius = std::numeric_limits<double>::infinity();
            double parent_dist = glm::length(old_pos - parent_pos);
            if (parent_m > 0.0 && body_m > 0.0 && parent_dist > 0.0) {
                hill_radius = parent_dist * std::cbrt(body_m / (3.0 * parent_m));
            }

            std::vector<bool> is_descendant(snapshot_.size(), false);
            std::vector<int> descendants{static_cast<int>(index)};
            for (size_t k = 0; k < descendants.size(); ++k) {
                for (int child : snapshot_.reference_hierarchy().children[descendants[k]]) {
                    is_descendant[child] = true;
                    descendants.push_back(child);
                }
            }

            for (size_t k = 0; k < snapshot_.size(); ++k) {
                if (k == index) continue;
                double dist = glm::length(snapshot_.motions_[k].position.value - old_pos);
                bool inside_hill = (dist <= hill_radius);
                bool is_moon = (is_descendant[k] || updated->bodies[k].type == "Moon") && inside_hill &&
                               !updated->bodies[k].is_star && static_cast<int>(k) != parent_idx;
                if (is_moon) {
                    snapshot_.motions_[k].position.value += delta_pos;
                    snapshot_.motions_[k].velocity.value += delta_vel;
                    ++snapshot_.revisions_[k].dynamics;
                }
            }
        }
    }

    snapshot_.definition_ = std::move(updated);

    auto& revisions = snapshot_.revisions_[index];
    ++revisions.definition;
    if (dynamics_changed) ++revisions.dynamics;
    if (before.radius != values.radius || before.oblateness != values.oblateness || before.type != values.type) {
        ++revisions.geometry;
        ++revisions.lighting;
    }
    ++snapshot_.definition_revision_;
    if (dynamics_changed) {
        reset_jacobi(true);
        if (snapshot_.definition_->motion_model == MotionModel::NBody) {
            ias15_.reset(snapshot_.definition(), snapshot_.motions(), clock_.time);
        } else {
            ias15_.update_parameters(snapshot_.definition());
        }
    }
    refresh_reference_hierarchy(dynamics_changed);
    prepare_render_state(true);
    return {true, true, {}};
}

CommandResult Simulation::apply(const AddBody& command) {
    if (command.system_generation != snapshot_.generation_) {
        return {false, false, "The target system changed before adding the body."};
    }
    if (snapshot_.definition().motion_model == MotionModel::Spice) {
        return {false, false, "Adding bodies is not supported in SPICE playback mode. Switch to Keplerian or N-Body mode first."};
    }
    const auto& def = command.definition;
    if (def.name.empty()) {
        return {false, false, "Body name cannot be empty."};
    }
    if (snapshot_.find_body(def.name) != -1) {
        return {false, false, "A body with that name already exists in this system."};
    }
    if (def.parent_idx < 0 || def.parent_idx >= static_cast<int>(snapshot_.size())) {
        return {false, false, "Invalid parent index."};
    }
    if (!std::isfinite(def.mass_msun) || def.mass_msun < 0.0 ||
        !std::isfinite(def.radius_rsun) || def.radius_rsun <= 0.0 ||
        !std::isfinite(def.semi_major_au) || def.semi_major_au <= 0.0 ||
        !std::isfinite(def.eccentricity) || def.eccentricity < 0.0 || def.eccentricity >= 1.0 ||
        !std::isfinite(def.inc_deg) || !std::isfinite(def.omega_node_deg) ||
        !std::isfinite(def.arg_peri_deg) || !std::isfinite(def.mean_anom_deg) ||
        !std::isfinite(def.rotation_period_hr) ||
        !std::isfinite(def.oblateness) || def.oblateness < 0.0 || def.oblateness >= 1.0 ||
        !std::isfinite(def.j2) || !std::isfinite(def.j4) ||
        !std::isfinite(command.axial_tilt_deg)) {
        return {false, false, "Body parameters contain invalid or nonfinite values."};
    }

    int parent_idx = def.parent_idx;
    double parent_m = snapshot_.definition_->bodies[parent_idx].mass_msun;
    auto cart = math::get_cartesian_from_keplerian(parent_m, def.mass_msun,
                                                   def.semi_major_au, def.eccentricity,
                                                   def.inc_deg, def.omega_node_deg,
                                                   def.arg_peri_deg, def.mean_anom_deg);
    if (command.equatorial_frame) {
        glm::dvec3 parent_pole = snapshot_.definition_->bodies[parent_idx].pole_ecl;
        auto rotated = math::rotate_equatorial_to_ecliptic(cart.pos, cart.vel, parent_pole);
        cart.pos = rotated.first;
        cart.vel = rotated.second;
    }

    if (!finite_vector(cart.pos) || !finite_vector(cart.vel)) {
        return {false, false, "Failed to compute initial orbital state."};
    }

    glm::dvec3 new_pos = snapshot_.motions_[parent_idx].position.value + cart.pos;
    glm::dvec3 new_vel = snapshot_.motions_[parent_idx].velocity.value + cart.vel;

    glm::dvec3 pole_ecl = def.pole_ecl;
    if (glm::length(pole_ecl) < 1e-6 || !finite_vector(pole_ecl)) {
        glm::dvec3 orb_h = glm::cross(cart.pos, cart.vel);
        double h_norm = glm::length(orb_h);
        glm::dvec3 orb_n = h_norm > 1e-12 ? (orb_h / h_norm) : glm::dvec3{0.0, 0.0, 1.0};
        double tilt_deg = command.axial_tilt_deg;
        if (std::abs(tilt_deg) < 1e-6) {
            pole_ecl = orb_n;
        } else {
            glm::dvec3 ref{0.0, 0.0, 1.0};
            if (std::abs(glm::dot(orb_n, ref)) > 0.99) ref = glm::dvec3{1.0, 0.0, 0.0};
            glm::dvec3 perp = glm::cross(orb_n, ref);
            double perp_n = glm::length(perp);
            perp = perp_n > 1e-10 ? (perp / perp_n) : glm::dvec3{1.0, 0.0, 0.0};
            double tilt_rad = tilt_deg * DEG_TO_RAD;
            pole_ecl = glm::normalize(orb_n * std::cos(tilt_rad) + perp * std::sin(tilt_rad));
        }
    } else {
        pole_ecl = glm::normalize(pole_ecl);
    }

    auto [pole_ra, pole_dec] = math::ecliptic_to_pole(pole_ecl);

    auto updated = std::make_shared<SystemDefinition>(*snapshot_.definition_);
    BodyDefinition new_body = def;
    new_body.parent_idx = parent_idx;
    new_body.parent_id = snapshot_.definition_->bodies[parent_idx].name;
    new_body.pole_ecl = pole_ecl;
    new_body.pole_ra_deg = pole_ra;
    new_body.pole_dec_deg = pole_dec;
    if (new_body.color_pole_rgb == glm::vec3(0.0f)) new_body.color_pole_rgb = new_body.color_rgb;

    int new_idx = static_cast<int>(updated->bodies.size());
    updated->bodies.push_back(new_body);
    updated->name_to_idx[new_body.name] = new_idx;

    snapshot_.definition_ = std::move(updated);
    snapshot_.motions_.push_back(BodyMotion{EclipticPosition{new_pos}, EclipticVelocity{new_vel}});
    snapshot_.render_.resize(snapshot_.motions_.size());
    snapshot_.revisions_.push_back(BodyRevisions{});
    ++snapshot_.definition_revision_;
    ++snapshot_.state_revision_;

    recenter();
    reset_jacobi(false);
    refresh_reference_hierarchy(true);
    if (snapshot_.definition_->motion_model == MotionModel::NBody) {
        ias15_.reset(snapshot_.definition(), snapshot_.motions(), clock_.time);
    }
    prepare_render_state(true);
    previous_motions_.resize(snapshot_.size());
    return {true, true, {}};
}

CommandResult Simulation::apply(const DeleteBody& command) {
    if (command.system_generation != snapshot_.generation_) {
        return {false, false, "The target system changed before deleting the body."};
    }
    if (snapshot_.definition().motion_model == MotionModel::Spice) {
        return {false, false, "Deleting bodies is not supported in SPICE playback mode. Switch to Keplerian or N-Body mode first."};
    }
    if (snapshot_.size() <= 1) {
        return {false, false, "Cannot delete the last remaining body in a system."};
    }
    int del_idx = command.body_index;
    if (del_idx < 0 || del_idx >= static_cast<int>(snapshot_.size())) {
        return {false, false, "Body index out of range."};
    }
    if (del_idx == snapshot_.definition_->star_idx || snapshot_.definition_->bodies[del_idx].is_star) {
        return {false, false, "Cannot delete the system's primary star."};
    }

    auto updated = std::make_shared<SystemDefinition>(*snapshot_.definition_);
    int fallback_parent_idx = updated->bodies[del_idx].parent_idx;
    if (fallback_parent_idx < 0 || fallback_parent_idx == del_idx || fallback_parent_idx >= static_cast<int>(updated->bodies.size())) {
        fallback_parent_idx = updated->star_idx;
    }
    int adjusted_fallback = (fallback_parent_idx > del_idx) ? (fallback_parent_idx - 1) : fallback_parent_idx;
    std::string fallback_parent_id = updated->bodies[fallback_parent_idx].name;

    for (size_t i = 0; i < updated->bodies.size(); ++i) {
        if (static_cast<int>(i) == del_idx) continue;
        auto& b = updated->bodies[i];
        if (b.parent_idx == del_idx) {
            glm::dvec3 rel_pos = snapshot_.motions_[i].position.value - snapshot_.motions_[fallback_parent_idx].position.value;
            glm::dvec3 rel_vel = snapshot_.motions_[i].velocity.value - snapshot_.motions_[fallback_parent_idx].velocity.value;
            double parent_m = snapshot_.definition_->bodies[fallback_parent_idx].mass_msun;
            double mu = G_CONST * (parent_m + b.mass_msun);
            auto oe = math::compute_keplerian_elements(rel_pos, rel_vel, mu);
            if (oe.a > 0.0 && oe.e >= 0.0 && oe.e < 1.0) {
                b.semi_major_au = oe.a;
                b.eccentricity = oe.e;
                b.inc_deg = oe.inc_deg;
                b.omega_node_deg = oe.omega_node_deg;
                b.arg_peri_deg = oe.arg_peri_deg;
                b.mean_anom_deg = oe.mean_anomaly_deg;
            }
            b.parent_idx = adjusted_fallback;
            b.parent_id = fallback_parent_id;
        } else if (b.parent_idx > del_idx) {
            b.parent_idx -= 1;
        }
    }

    updated->bodies.erase(updated->bodies.begin() + del_idx);
    updated->name_to_idx.clear();
    for (size_t i = 0; i < updated->bodies.size(); ++i) {
        updated->name_to_idx[updated->bodies[i].name] = static_cast<int>(i);
    }
    if (updated->star_idx == del_idx) {
        updated->star_idx = 0;
    } else if (updated->star_idx > del_idx) {
        updated->star_idx -= 1;
    }

    snapshot_.definition_ = std::move(updated);
    snapshot_.motions_.erase(snapshot_.motions_.begin() + del_idx);
    snapshot_.render_.erase(snapshot_.render_.begin() + del_idx);
    snapshot_.revisions_.erase(snapshot_.revisions_.begin() + del_idx);
    ++snapshot_.definition_revision_;
    ++snapshot_.state_revision_;

    recenter();
    reset_jacobi(false);
    refresh_reference_hierarchy(true);
    if (snapshot_.definition_->motion_model == MotionModel::NBody) {
        ias15_.reset(snapshot_.definition(), snapshot_.motions(), clock_.time);
    }
    prepare_render_state(true);
    previous_motions_.resize(snapshot_.size());
    return {true, true, {}};
}

CommandResult Simulation::advance(units::Seconds elapsed) {
    if (!std::isfinite(elapsed.value) || elapsed.value < 0.0) {
        return {false, false, "Frame time must be finite and nonnegative."};
    }
    if (clock_.paused || elapsed.value == 0.0 || snapshot_.empty()) return {};
    auto dt = units::to_years(units::Seconds{elapsed.value * clock_.warp * clock_.direction});
    double next_time = clock_.time.value + dt.value;
    if (!std::isfinite(dt.value) || !std::isfinite(next_time) || !std::isfinite(units::to_seconds(units::Years{next_time}).value)) {
        return {false, false, "Simulation time overflow."};
    }

    previous_motions_ = snapshot_.motions_;
    bool is_spice = (snapshot_.definition().motion_model == MotionModel::Spice);
    bool is_nbody = (snapshot_.definition().motion_model == MotionModel::NBody);

    if (is_nbody) {
        auto status = ias15_.integrate(snapshot_.definition(), snapshot_.motions_, units::Years{next_time});
        if (!status) {
            snapshot_.motions_.swap(previous_motions_);
            return status;
        }
    } else if (!is_spice) {
        kepler_.evaluate(snapshot_.definition(), snapshot_.jacobi_hierarchy(), units::Years{next_time}, snapshot_.motions_);
    }

    const auto& definitions = snapshot_.definition_->bodies;
    for (size_t i = 0; i < snapshot_.size(); ++i) {
        auto& motion = snapshot_.motions_[i];
        const auto& body = definitions[i];
        if (is_spice) {
            if (!spice_) {
                snapshot_.motions_.swap(previous_motions_);
                return {false, false, "SPICE runtime uninitialized."};
            }
            auto status = spice_->state(body.spice_id, epoch_et_ + units::to_seconds(units::Years{next_time}).value, motion);
            if (!status) {
                snapshot_.motions_.swap(previous_motions_);
                return {false, false, body.name + ": " + status.error};
            }
        }
        if (!finite_vector(motion.position.value) || !finite_vector(motion.velocity.value)) {
            snapshot_.motions_.swap(previous_motions_);
            return {false, false, "Propagation produced a nonfinite state; the step was rolled back."};
        }
    }
    recenter();
    for (const auto& motion : snapshot_.motions_) {
        if (!finite_vector(motion.position.value) || !finite_vector(motion.velocity.value)) {
            snapshot_.motions_.swap(previous_motions_);
            return {false, false, "Recentering produced a nonfinite state; the step was rolled back."};
        }
    }
    clock_.time = units::Years{next_time};
    snapshot_.time_ = clock_.time;
    ++snapshot_.state_revision_;
    constexpr double hierarchy_interval = 1.0 / 15.0;
    hierarchy_elapsed_seconds_ += elapsed.value;
    if (hierarchy_elapsed_seconds_ >= hierarchy_interval) {
        double remainder = std::fmod(hierarchy_elapsed_seconds_, hierarchy_interval);
        refresh_reference_hierarchy();
        hierarchy_elapsed_seconds_ = remainder;
    }
    prepare_render_state();
    return {true, true, {}};
}

void Simulation::reset_jacobi(bool preserve_order) {
    auto hierarchy = build_jacobi_hierarchy(snapshot_.definition(), snapshot_.motions(), preserve_order ? snapshot_.jacobi_.get() : nullptr);
    snapshot_.jacobi_ = std::make_shared<const JacobiHierarchy>(std::move(hierarchy));
    if (snapshot_.definition().motion_model == MotionModel::Keplerian) {
        kepler_.reset(snapshot_.definition(), snapshot_.motions(), snapshot_.jacobi_hierarchy(), clock_.time);
    }
}

void Simulation::refresh_reference_hierarchy(bool dynamics_changed) {
    auto next = hierarchy_tracker_.evaluate(snapshot_.definition(), snapshot_.motions(), snapshot_.reference_hierarchy());
    bool parents_changed = next.parents != snapshot_.reference_->parents;
    if (snapshot_.definition().motion_model != MotionModel::NBody || next.parents == snapshot_.jacobi_->parents) {
        snapshot_.reference_jacobi_.reset();
    } else if (parents_changed || dynamics_changed || !snapshot_.reference_jacobi_) {
        auto hierarchy = build_jacobi_hierarchy(snapshot_.definition(), snapshot_.motions(), snapshot_.reference_jacobi_.get(), next.parents);
        snapshot_.reference_jacobi_ = std::make_shared<const JacobiHierarchy>(std::move(hierarchy));
    }
    if (parents_changed || next.children != snapshot_.reference_->children) {
        snapshot_.reference_ = std::make_shared<const ReferenceHierarchy>(std::move(next));
        ++snapshot_.hierarchy_revision_;
    }
    hierarchy_elapsed_seconds_ = 0.0;
}

void Simulation::recenter() {
    double total_mass = 0.0;
    glm::dvec3 position{0.0}, velocity{0.0};
    for (size_t i = 0; i < snapshot_.size(); ++i) {
        double mass = snapshot_.definition_->bodies[i].mass_msun;
        total_mass += mass;
        position += mass * snapshot_.motions_[i].position.value;
        velocity += mass * snapshot_.motions_[i].velocity.value;
    }
    if (total_mass <= 0.0) return;
    position /= total_mass;
    velocity /= total_mass;
    for (auto& motion : snapshot_.motions_) {
        motion.position.value -= position;
        motion.velocity.value -= velocity;
    }
    if (snapshot_.definition().motion_model == MotionModel::NBody) {
        ias15_.shift_com(position, velocity);
    }
}

void Simulation::prepare_render_state(bool definitions_changed) {
    double seconds = units::to_seconds(clock_.time).value;
    for (size_t i = 0; i < snapshot_.size(); ++i) {
        auto& render = snapshot_.render_[i];
        const auto& body = snapshot_.definition_->bodies[i];
        render.position = ecliptic_to_render(snapshot_.motions_[i].position);
        render.velocity = ecliptic_to_render(snapshot_.motions_[i].velocity);
        if (definitions_changed) render.pole = glm::normalize(ecliptic_to_render(body.pole_ecl));
        double w0 = body.w0_deg * DEG_TO_RAD;
        if (body.tidally_locked && body.parent_idx >= 0) {
            auto relative = render.position.value - snapshot_.render_[body.parent_idx].position.value;
            render.spin_angle_rad = std::atan2(relative.z, relative.x) + PI + w0;
        } else if (std::abs(body.rotation_period_hr) > 1e-9) {
            double omega = TWO_PI / (body.rotation_period_hr * 3600.0);
            render.spin_angle_rad = std::fmod(w0 + omega * seconds, TWO_PI);
            if (render.spin_angle_rad < 0.0) render.spin_angle_rad += TWO_PI;
        } else {
            render.spin_angle_rad = w0;
        }
    }
}

}
