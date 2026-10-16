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
    return std::isfinite(values.mass.value) && values.mass.value >= 0.0 &&
           std::isfinite(values.radius.value) && values.radius.value > 0.0 &&
           std::isfinite(values.oblateness) && values.oblateness >= 0.0 && values.oblateness < 1.0 &&
           std::isfinite(values.j2) && std::isfinite(values.j4);
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
    clock_.time = units::Years{};
    previous_motions_.resize(snapshot_.size());
    return {true, true, {}};
}

CommandResult Simulation::apply(SetPaused command) {
    bool changed = command.value != clock_.paused;
    clock_.paused = command.value;
    return {true, changed, {}};
}

CommandResult Simulation::apply(SetTimeWarp command) {
    if (!std::isfinite(command.value) || command.value < 1.0 || command.value > 1e12) {
        return {false, false, "Time warp must be finite and between 1 and 1e12."};
    }
    bool changed = command.value != clock_.warp;
    clock_.warp = command.value;
    return {true, changed, {}};
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
    snapshot_.definition_ = std::move(updated);

    auto& revisions = snapshot_.revisions_[index];
    ++revisions.definition;
    bool dynamics_changed = (before.mass != values.mass || before.oblateness != values.oblateness ||
                             before.j2 != values.j2 || before.j4 != values.j4 ||
                             (before.radius != values.radius && (before.j2 != 0.0 || before.j4 != 0.0)));
    if (dynamics_changed) ++revisions.dynamics;
    if (before.radius != values.radius || before.oblateness != values.oblateness) {
        ++revisions.geometry;
        ++revisions.lighting;
    }
    ++snapshot_.definition_revision_;
    if (dynamics_changed) reset_jacobi(true);
    if (dynamics_changed) ias15_.update_parameters(snapshot_.definition());
    prepare_render_state(true);
    return {true, true, {}};
}

CommandResult Simulation::advance(units::Seconds elapsed) {
    if (!std::isfinite(elapsed.value) || elapsed.value < 0.0) {
        return {false, false, "Frame time must be finite and nonnegative."};
    }
    if (clock_.paused || elapsed.value == 0.0 || snapshot_.empty()) return {};
    auto dt = units::to_years(units::Seconds{elapsed.value * clock_.warp});
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
        } else {
            render.spin_angle_rad = w0;
        }
    }
}

}
