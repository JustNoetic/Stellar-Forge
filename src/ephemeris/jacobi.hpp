#pragma once

#include "ephemeris/system_manager.hpp"
#include <span>

namespace sf::ephemeris {

struct JacobiHierarchy {
    std::vector<std::vector<int>> children;
    std::vector<double> subsystem_mass;
    std::vector<double> inner_mass;
};

struct JacobiSubsystem {
    double mass = 0.0;
    glm::dvec3 position{0.0}, velocity{0.0};
};

struct JacobiFrame {
    JacobiSubsystem inner;
    glm::dvec3 relative_position{0.0}, relative_velocity{0.0}, barycenter{0.0};
};

struct JacobiCoordinates {
    std::vector<JacobiSubsystem> subsystems;
    std::vector<JacobiFrame> frames;
};

JacobiHierarchy build_jacobi_hierarchy(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                                      const JacobiHierarchy* previous = nullptr);
JacobiCoordinates extract_jacobi(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                                 const JacobiHierarchy& hierarchy);

class JacobiPropagator {
public:
    void reset(const SystemDefinition& definition, std::span<const BodyMotion> motions,
               const JacobiHierarchy& hierarchy, units::Years time);
    void evaluate(const SystemDefinition& definition, const JacobiHierarchy& hierarchy,
                  units::Years time, std::vector<BodyMotion>& output);

private:
    units::Years epoch_{};
    std::vector<BodyMotion> seeds_, initial_centers_, relative_, centers_;
};

}
