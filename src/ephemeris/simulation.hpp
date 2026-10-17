#pragma once

#include "core/command_result.hpp"
#include "ephemeris/system_manager.hpp"
#include "ephemeris/ias15.hpp"
#include "ephemeris/jacobi.hpp"
#include "ephemeris/reference_hierarchy.hpp"

namespace sf::ephemeris {

class SpiceEphemeris;
struct TimelineRecording;

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

    double temp_k = 5778.0;
    double lum_eq = 1.0;
    double lum_pole = 1.0;
    glm::vec3 color_rgb{1.0f};
    glm::vec3 color_pole_rgb{1.0f};
    std::string spectral_class;
    std::string stage;
    nlohmann::json star_props;

    bool operator==(const BodyPhysicalProperties&) const = default;

    static BodyPhysicalProperties from(const BodyDefinition& body) {
        return {units::SolarMasses{body.mass_msun}, units::Kilometres{body.radius_rsun * SOLAR_RADIUS_KM},
                body.oblateness, body.j2, body.j4, body.type, body.rotation_period_hr, body.tidally_locked,
                false, body.semi_major_au, body.eccentricity, body.inc_deg, body.omega_node_deg,
                body.arg_peri_deg, body.mean_anom_deg, false, true,
                body.temp_k, body.lum_eq, body.lum_pole, body.color_rgb, body.color_pole_rgb,
                body.spectral_class, body.stage,
                body.source_json.contains("star_props") ? body.source_json["star_props"] : nlohmann::json{}};
    }
};

struct ApplyBodyPhysicalProperties {
    uint64_t system_generation;
    int body_index;
    uint64_t expected_body_revision;
    BodyPhysicalProperties values;
    int expected_reference_parent = -2;
};

struct ApplyBodyRings {
    uint64_t system_generation = 0;
    int body_index = -1;
    uint64_t expected_body_revision = 0;
    std::vector<RingLayerDefinition> rings;
    std::optional<double> ring_texture_inner;
    std::optional<double> ring_texture_outer;
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

struct JumpToTime {
    units::Years target_time;
};

struct SyncTimelineState {
    units::Years time;
    std::vector<BodyMotion> motions;
};

class Simulation {
public:
    const SystemSnapshot& snapshot() const noexcept { return snapshot_; }
    const SimulationClock& clock() const noexcept { return clock_; }
    const JacobiPropagator& kepler() const noexcept { return kepler_; }
    double nbody_epsilon() const noexcept { return ias15_.epsilon(); }
    std::shared_ptr<SpiceEphemeris> spice() const noexcept { return spice_; }
    double epoch_et() const noexcept { return epoch_et_; }
    const Ias15Integrator& ias15() const noexcept { return ias15_; }

    CommandResult replace_system(LoadedSystem loaded);
    CommandResult apply(SetPaused command);
    CommandResult apply(SetTimeWarp command);
    CommandResult apply(SetTimeDirection command);
    CommandResult apply(ReverseTimeDirection command);
    CommandResult apply(SetMotionModel command);
    CommandResult apply(const ApplyBodyPhysicalProperties& command);
    CommandResult apply(const ApplyBodyRings& command);
    CommandResult apply(const AddBody& command);
    CommandResult apply(const DeleteBody& command);
    CommandResult apply(JumpToTime command);
    CommandResult apply(const SyncTimelineState& command);
    void apply_timeline_frame(const TimelineRecording& rec, size_t frame_index);
    double estimate_min_orbital_period() const;
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
