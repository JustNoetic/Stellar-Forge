#pragma once

#include "ephemeris/system_manager.hpp"

namespace sf::ephemeris {

struct OrbitArc {
    double start = 0.0;
    double span = TWO_PI;
    double phase = 0.0;
    double exit_angle = 0.0;
    bool escape = false;
    bool outside = false;
};

struct OrbitCurve {
    glm::dvec3 periapsis{1.0, 0.0, 0.0};
    glm::dvec3 transverse{0.0, 1.0, 0.0};
    EclipticPosition focus{};
    double semilatus_au = 0.0;
    double eccentricity = 0.0;
    OrbitArc arc{};
    glm::vec3 color{1.0f};
    int body_index = -1;
    bool patched = false;
    bool reflex = false;

    EclipticPosition point(double fraction) const;
};

OrbitArc compute_orbit_arc(double eccentricity, double current_angle, double relative_p,
                           double relative_radius, double hill_radius, double primary_scale);
std::vector<OrbitCurve> compute_orbit_curves(const SystemSnapshot& snapshot);

}
