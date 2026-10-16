#pragma once

#include "ephemeris/system_manager.hpp"
#include <span>

namespace sf::ephemeris {

struct JacobiHierarchy {
    std::vector<int> parents;
    std::vector<int> traversal;
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

struct KeplerOrbit {
    double a = 0.0;
    double e = 0.0;
    double M0 = 0.0;
    double mu = 0.0;
    glm::dvec3 n{0.0, 0.0, 1.0};
    glm::dvec3 u_e{1.0, 0.0, 0.0};
    glm::dvec3 prec_vec{0.0};
    double domega_peri = 0.0;
    bool is_elliptic = false;
};

JacobiHierarchy build_jacobi_hierarchy(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                                      const JacobiHierarchy* previous = nullptr, std::span<const int> parents = {});
JacobiCoordinates extract_jacobi(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                                 const JacobiHierarchy& hierarchy);

class JacobiPropagator {
public:
    void reset(const SystemDefinition& definition, std::span<const BodyMotion> motions,
               const JacobiHierarchy& hierarchy, units::Years time);
    void evaluate(const SystemDefinition& definition, const JacobiHierarchy& hierarchy,
                  units::Years time, std::vector<BodyMotion>& output);

    const std::vector<KeplerOrbit>& orbits() const noexcept { return orbits_; }

private:
    units::Years epoch_{};
    std::vector<BodyMotion> seeds_, initial_centers_, relative_, centers_;
    std::vector<KeplerOrbit> orbits_;
};

}
