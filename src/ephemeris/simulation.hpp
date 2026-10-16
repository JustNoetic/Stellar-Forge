#pragma once

#include "core/command_result.hpp"
#include "ephemeris/system_manager.hpp"
#include "ephemeris/ias15.hpp"
#include "ephemeris/jacobi.hpp"
#include "ephemeris/reference_hierarchy.hpp"

namespace sf::ephemeris {

class SpiceEphemeris;

struct SimulationClock {
    units::Years time{};
    double warp = 1.0;
    double direction = 1.0;
    bool paused = false;

    double effective_warp() const noexcept { return warp * direction; }
};

struct SetPaused { bool value; };
struct SetTimeWarp { double value; };
struct SetTimeDirection { double value; };
struct ReverseTimeDirection {};
struct SetMotionModel { MotionModel model; };

struct BodyPhysicalProperties {
    units::SolarMasses mass{};
    units::Kilometres radius{};
    double oblateness = 0.0;
    double j2 = 0.0;
    double j4 = 0.0;
    std::string type = "Terrestrial";
    double rotation_period_hr = 0.0;
    bool tidally_locked = false;

    bool update_orbit = false;
    double a_au = 0.0;
    double e = 0.0;
    double inc_deg = 0.0;
    double omega_node_deg = 0.0;
    double arg_peri_deg = 0.0;
    double mean_anom_deg = 0.0;
    bool equatorial_frame = false;
    bool bring_moons = true;

    bool operator==(const BodyPhysicalProperties&) const = default;

    static BodyPhysicalProperties from(const BodyDefinition& body) {
        return {units::SolarMasses{body.mass_msun}, units::Kilometres{body.radius_rsun * SOLAR_RADIUS_KM},
                body.oblateness, body.j2, body.j4, body.type, body.rotation_period_hr, body.tidally_locked,
                false, body.semi_major_au, body.eccentricity, body.inc_deg, body.omega_node_deg,
                body.arg_peri_deg, body.mean_anom_deg, false, true};
    }
};

struct ApplyBodyPhysicalProperties {
    uint64_t system_generation;
    int body_index;
    uint64_t expected_body_revision;
    BodyPhysicalProperties values;
    int expected_reference_parent = -2;
};

struct AddBody {
    uint64_t system_generation = 0;
    BodyDefinition definition;
    bool equatorial_frame = false;
    double axial_tilt_deg = 0.0;
};

struct DeleteBody {
    uint64_t system_generation = 0;
    int body_index = -1;
};

class Simulation {
public:
    const SystemSnapshot& snapshot() const noexcept { return snapshot_; }
    const SimulationClock& clock() const noexcept { return clock_; }
    const JacobiPropagator& kepler() const noexcept { return kepler_; }
    double nbody_epsilon() const noexcept { return ias15_.epsilon(); }
    CommandResult replace_system(LoadedSystem loaded);
    CommandResult apply(SetPaused command);
    CommandResult apply(SetTimeWarp command);
    CommandResult apply(SetTimeDirection command);
    CommandResult apply(ReverseTimeDirection command);
    CommandResult apply(SetMotionModel command);
    CommandResult apply(const ApplyBodyPhysicalProperties& command);
    CommandResult apply(const AddBody& command);
    CommandResult apply(const DeleteBody& command);
    CommandResult advance(units::Seconds elapsed);

private:
    void recenter();
    void prepare_render_state(bool definitions_changed = false);
    void reset_jacobi(bool preserve_order = false);
    void refresh_reference_hierarchy(bool dynamics_changed = false);

    SystemSnapshot snapshot_{};
    SimulationClock clock_{};
    std::vector<BodyMotion> previous_motions_;
    std::shared_ptr<SpiceEphemeris> spice_;
    double epoch_et_ = 0.0;
    JacobiPropagator kepler_;
    Ias15Integrator ias15_;
    ReferenceHierarchyTracker hierarchy_tracker_;
    double hierarchy_elapsed_seconds_ = 0.0;
};

}
