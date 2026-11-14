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

struct EphemerisOrbitConfig {
    uint32_t frame_mode = 0;
    uint32_t rotating_pair = 0;
    float lagrange_days = 240.0f;
    uint32_t sample_points = 5000;
    bool trail_fade = true;
    float trail_days = 30.0f;
    float lead_days = 10.0f;
    bool hide_outside = true;
    bool operator==(const EphemerisOrbitConfig&) const = default;
};

struct OrbitPolyline {
    int body_index = -1;
    glm::vec3 color{1.0f};
    EclipticPosition center{};
    double extent_au = 0.0;
    std::vector<EclipticPosition> positions;
    std::vector<float> alphas;
};

class SpiceTrajectoryCache {
public:
    void evaluate(const SystemSnapshot& snapshot, const EphemerisOrbitConfig& config,
                  std::vector<OrbitPolyline>& out_polylines,
                  std::vector<bool>& out_covered_bodies);

private:
    struct CachedBody {
        int body_index = -1;
        int spice_id = 0;
        int parent_index = -1;
        int parent_spice_id = 0;
        bool rotating = false;
        int rot_p1_idx = -1;
        int rot_p2_idx = -1;
        int rot_anchor_idx = -1;
        int rot_p1_spice = 0;
        int rot_p2_spice = 0;
        int rot_anchor_spice = 0;
        double start_et = 0.0;
        double end_et = 0.0;
        std::vector<double> ets;
        std::vector<glm::dvec3> coords;
    };

    uint64_t generation_ = 0;
    uint64_t definition_revision_ = 0;
    EphemerisOrbitConfig config_{};
    std::vector<CachedBody> entries_;
};

OrbitArc compute_orbit_arc(double eccentricity, double current_angle, double relative_p,
                           double relative_radius, double hill_radius, double primary_scale);
std::vector<OrbitCurve> compute_orbit_curves(const SystemSnapshot& snapshot);

}
