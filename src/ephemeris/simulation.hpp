#pragma once

#include "core/command_result.hpp"
#include "ephemeris/system_manager.hpp"

namespace sf::ephemeris {

struct SimulationClock {
    units::Years time{};
    double warp = 1.0;
    bool paused = false;
};

struct SetPaused { bool value; };
struct SetTimeWarp { double value; };

struct BodyPhysicalProperties {
    units::SolarMasses mass{};
    units::Kilometres radius{};
    double oblateness = 0.0;
    double j2 = 0.0;
    double j4 = 0.0;
    bool operator==(const BodyPhysicalProperties&) const = default;

    static BodyPhysicalProperties from(const BodyDefinition& body) {
        return {units::SolarMasses{body.mass_msun}, units::Kilometres{body.radius_rsun * SOLAR_RADIUS_KM},
                body.oblateness, body.j2, body.j4};
    }
};

struct ApplyBodyPhysicalProperties {
    uint64_t system_generation;
    int body_index;
    uint64_t expected_body_revision;
    BodyPhysicalProperties values;
};

class Simulation {
public:
    const SystemSnapshot& snapshot() const noexcept { return snapshot_; }
    const SimulationClock& clock() const noexcept { return clock_; }
    CommandResult replace_system(LoadedSystem loaded);
    CommandResult apply(SetPaused command);
    CommandResult apply(SetTimeWarp command);
    CommandResult apply(const ApplyBodyPhysicalProperties& command);
    CommandResult advance(units::Seconds elapsed);

private:
    void recenter();
    void prepare_render_state(bool definitions_changed = false);

    SystemSnapshot snapshot_{};
    SimulationClock clock_{};
    std::vector<BodyMotion> previous_motions_;
};

}
