#pragma once

#include "benchmark/earth_closeup.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/ring_generator.hpp"
#include <algorithm>
#include <string_view>

namespace sf::benchmark {

inline ephemeris::RingLayerDefinition audit_earth_ring(const rendering::ProceduralRingResult& profile) {
    ephemeris::RingLayerDefinition ring;
    rendering::apply_procedural_ring_layer(profile, ring);
    return ring;
}

inline void apply_audit_variant(rendering::RenderSettings& s, std::string_view variant) {
    if (variant == "terrain-detail") s.terrain_error_pixels = 2;
    if (variant == "no-bloom") s.bloom_mode = 0;
    if (variant == "fast-bloom") s.bloom_mode = 1;
    if (variant == "no-clouds") s.clouds_enabled = false;
    if (variant == "legacy-clouds") s.transparency_reference_legacy = true;
    if (variant == "cached-shadow-weights") s.atmosphere_shadow_cached_weights = true;
    if (variant == "native-shadow-weights") s.atmosphere_shadow_cached_weights = false;
    if (variant == "fast-ring-shadows") s.atmosphere_shadow_fast_rings = true;
    if (variant == "analytic-shadows") s.atmosphere_shadow_analytic = true;
    if (variant == "jittered-shadows") s.atmosphere_shadow_analytic = false;
    if (variant == "no-atmosphere") s.show_atmospheres = false;
    if (variant == "no-aerial") s.aerial_perspective_enabled = false;
    if (variant == "no-shadows") s.atmosphere_shadows_enabled = false;
    if (variant == "no-refraction") s.refraction_enabled = false;
    if (variant == "no-secondary") s.planetshine_enabled = s.ringshine_enabled = false;
    if (variant == "no-planetshine") s.planetshine_enabled = false;
    if (variant == "no-ringshine") s.ringshine_enabled = false;
    if (variant == "high-exposure") s.exposure = 1e4f;
    if (variant == "no-orbits") s.show_orbits = false;
    if (variant == "no-gaia") s.show_gaia_stars = false;
    if (variant == "smooth") s.terrain_imported_enabled = s.terrain_enabled = s.terrain_normal_maps_enabled = false;
    if (variant == "no-textures") s.surface_textures_enabled = false;
    if (variant == "no-normals") s.terrain_normal_maps_enabled = false;
    if (variant == "no-rings") s.show_rings = false;
    if (variant == "no-ring-shadows") s.ring_shadows_enabled = false;
}

inline rendering::RenderSettings audit_settings(std::string_view variant) {
    rendering::RenderSettings s;
    for (size_t start = 0; start <= variant.size();) {
        size_t plus = variant.find('+', start);
        apply_audit_variant(s, variant.substr(start, plus == std::string_view::npos ? std::string_view::npos : plus - start));
        if (plus == std::string_view::npos) break;
        start = plus + 1;
    }
    return s;
}

inline void audit_look(Camera& camera, glm::dvec3 forward, glm::dvec3 up) {
    forward = glm::normalize(forward);
    camera.cam_look_aim = false;
    camera.up = up;
    camera.yaw_deg = camera.yaw_actual_deg = std::atan2(forward.z, forward.x)*RAD_TO_DEG;
    camera.pitch_deg = camera.pitch_actual_deg = std::asin(std::clamp(forward.y, -1.0, 1.0))*RAD_TO_DEG;
}

inline glm::dvec3 audit_shadow_direction(const ephemeris::SystemSnapshot& system, int earth) {
    const auto& body = system.body(earth);
    auto sun = glm::normalize(system.body(system.star_index()).render.position.value-body.render.position.value);
    auto pole = body.render.pole;
    double elevation = glm::dot(sun, pole);
    auto equatorial = glm::normalize(sun-pole*elevation);
    glm::dvec3 best = sun;
    double error = 1e9;
    for (int degrees = 1; degrees < 85; ++degrees) {
        double angle = degrees*DEG_TO_RAD;
        auto direction = equatorial*std::cos(angle)-pole*std::copysign(std::sin(angle), elevation);
        double distance = -glm::dot(direction, pole)/elevation;
        double ring_radius = glm::length(direction+sun*distance);
        if (distance <= 0 || glm::dot(direction, sun) <= 0) continue;
        if (std::abs(ring_radius-1.7) < error) {
            best = direction;
            error = std::abs(ring_radius-1.7);
        }
    }
    if (error > 0.1) throw std::runtime_error("Earth audit ring shadow geometry is unavailable at this epoch.");
    return best;
}

inline Camera audit_camera(const ephemeris::SystemSnapshot& system, std::string_view scene) {
    if (scene == "earth-full" || scene == "earth-full-live" || scene == "earth-eclipse") return earth_closeup_camera(system, scene == "earth-eclipse");
    Camera camera;
    camera.horizon_align = false;
    camera.tracking_idx = system.find_body(scene.starts_with("saturn") ? "Saturn" : "Earth");
    if (camera.tracking_idx < 0) throw std::runtime_error("Audit camera target missing.");
    const auto& body = system.body(camera.tracking_idx);
    auto sun = glm::normalize(system.body(system.star_index()).render.position.value-body.render.position.value);
    if (scene == "saturn-midnight") {
        camera.fov_deg = 17.0;
        camera.cam_pos_rel = -sun*(247000.0*KM_TO_AU);
        camera.yaw_deg = camera.yaw_actual_deg = std::atan2(sun.z, sun.x)*RAD_TO_DEG;
        camera.pitch_deg = camera.pitch_actual_deg = std::asin(std::clamp(sun.y, -1.0, 1.0))*RAD_TO_DEG;
        camera.cam_look_aim = true;
        return camera;
    }
    auto normal = scene.starts_with("earth-ground") || scene == "earth-full-rings"
        ? audit_shadow_direction(system, camera.tracking_idx) : sun;
    double radius = body.definition.radius_au();
    if (scene.starts_with("earth-distant")) {
        auto side = glm::normalize(glm::cross(body.render.pole, sun));
        double range = scene.size() > 14 ? std::stod(std::string(scene.substr(14))) : 7.0;
        camera.fov_deg = std::max(0.001, 2*std::atan(body.definition.radius_au()/range)*RAD_TO_DEG/0.7);
        camera.cam_pos_rel = (sun*std::cos(75*DEG_TO_RAD)+side*std::sin(75*DEG_TO_RAD))*range;
        return camera;
    }
    double distance = scene == "earth-disc" ? radius*2.5 : scene == "saturn" ? radius*5 : radius*1.1;
    if (scene.starts_with("earth-ground")) {
        distance = ellipsoid_surface_radius(radius, body.definition.oblateness, body.render.pole, normal)+2*KM_TO_AU;
        auto up = oblate_surface_normal(normal, body.definition.oblateness, body.render.pole);
        auto tangent = glm::normalize(sun-up*glm::dot(sun, up));
        audit_look(camera, tangent*std::cos(3*DEG_TO_RAD)+up*std::sin(3*DEG_TO_RAD), up);
    } else if (scene == "earth-limb") {
        auto tangent = glm::normalize(body.render.pole-normal*glm::dot(body.render.pole, normal));
        double angle = std::asin(radius/distance);
        audit_look(camera, -normal*std::cos(angle)+tangent*std::sin(angle), normal);
    }
    camera.cam_pos_rel = normal*distance;
    return camera;
}

}
