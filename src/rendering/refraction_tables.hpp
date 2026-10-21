#pragma once

#include "physics/atmosphere_refraction.hpp"
#include <glm/glm.hpp>
#include <cmath>
#include <numbers>
#include <vector>

namespace sf::rendering {

inline constexpr uint32_t REFRACTION_PROFILE_TEXELS_X = 1024;
inline constexpr uint32_t REFRACTION_OBSERVER_WIDTH = 512;
inline constexpr uint32_t REFRACTION_OBSERVER_HEIGHT = 128;

struct RefractionHost {
    int body = -1;
    double camera_radius_km = 0.0;
    double ground_radius_km = 0.0;
    double max_displacement = 0.0;
    double lowest_visible_chord_km = 0.0;
};

struct alignas(16) RefractionGPU {
    glm::vec4 host{0.0f};
    glm::vec4 pole{0.0f, 1.0f, 0.0f, 0.0f};
    glm::vec4 frame{0.0f};
    glm::vec4 mapping{0.0f};
    glm::vec4 model{0.0f};
    glm::vec4 profile{0.0f};
    glm::vec4 escape{0.0f};
};
static_assert(sizeof(RefractionGPU) == 112);

struct RefractionTables {
    std::vector<float> travel;
    std::vector<glm::vec2> rows;
    std::vector<glm::vec2> profile;
    uint32_t profile_rows = 0;
};

inline RefractionTables pack_refraction_tables(const physics::PreparedRefraction& r) {
    RefractionTables tables;
    double ground = r.ground_radius_km;
    tables.travel.assign(r.travel.begin(), r.travel.end());
    tables.rows.resize(physics::REFRACTION_IMPACT_SAMPLES);
    for (size_t i = 0; i < tables.rows.size(); ++i)
        tables.rows[i] = {float(r.impact[i]-ground), float(r.impact[i]-r.turn_radius[i])};
    size_t samples = r.fine_refractivity.size();
    tables.profile_rows = uint32_t((samples+REFRACTION_PROFILE_TEXELS_X-1)/REFRACTION_PROFILE_TEXELS_X);
    tables.profile.assign(size_t(tables.profile_rows)*REFRACTION_PROFILE_TEXELS_X, glm::vec2(0.0f));
    for (size_t i = 0; i < samples; ++i)
        tables.profile[i] = {float(r.fine_refractivity[i].y), float(r.fine_gradient[i].y)};
    return tables;
}

inline RefractionGPU refraction_parameters(const physics::PreparedRefraction& r, double camera_radius_km) {
    RefractionGPU g;
    double ground = r.ground_radius_km;
    double camera = camera_radius_km-ground;
    double top = r.top_radius_km-ground;
    bool outside = camera_radius_km > r.top_radius_km;
    g.frame = {float(camera), float(ground), float(top), outside ? 1.0f : 0.0f};
    double horizon = std::numbers::pi/2;
    if (camera > 0) horizon += std::atan2(std::sqrt(camera*(2*ground+camera)), ground);
    double scale = std::max(r.scale_km, 1e-3);
    double lowest = std::min(r.lowest_visible_chord(camera_radius_km)-ground-1.0, -1.0);
    lowest = std::max(lowest, -ground);
    double distance = std::sqrt(std::max(camera_radius_km*camera_radius_km-ground*ground, 0.0))+10.0;
    if (outside) g.mapping = {float(horizon), float(std::asinh(lowest/scale)), float(std::asinh(top/scale)), float(distance)};
    else g.mapping = {float(horizon), float(std::min(1.5*r.max_deflection()+1e-3, .5*(std::numbers::pi-horizon))), 0.0f, float(distance)};
    double span = std::asinh((r.top_radius_km-r.impact_min_km)/r.impact_scale_km);
    g.model = {float(r.impact_min_km-ground), float(r.impact_scale_km), float(span), float(r.bottom_radius_km-ground)};
    double step = (r.top_radius_km-r.bottom_radius_km)/double(r.fine_refractivity.size()-1);
    g.profile = {float(step), float(r.fine_refractivity.size()), float(r.index(camera_radius_km)-1.0), float(r.max_deflection())};
    double escape = 1e30, escape_height = 1e30;
    if (!r.monotone.y && camera_radius_km < r.top_radius_km) {
        double minimum = r.invariant(camera_radius_km), at = camera_radius_km;
        for (size_t i = 0; i < r.node_radii.size(); ++i) {
            double radius = r.node_radii[i];
            if (radius > camera_radius_km && radius < r.top_radius_km && r.node_invariant[i].y < minimum) {
                minimum = r.node_invariant[i].y;
                at = radius;
            }
        }
        if (at > camera_radius_km) {
            escape = minimum-ground;
            escape_height = at-ground;
        }
    }
    g.escape = {float(escape), float(escape_height), float(149597870.7), float(scale)};
    return g;
}

}
