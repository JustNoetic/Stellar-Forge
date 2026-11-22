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
    if (variant == "layered-clouds") s.cloud_sheet_inline = false;
    if (variant == "reference-shadows") s.atmosphere_shadow_reference = true;
    if (variant == "progressive-shadows") {
        s.atmosphere_shadow_temporal = true;
        s.atmosphere_shadow_samples = 16;
    }
    if (variant == "no-shadow-vrs") s.atmosphere_shadow_vrs = false;
    if (variant == "always-shadow-vrs") s.atmosphere_shadow_vrs_budget = 0.0f;
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
    if (variant == "smooth") s.terrain_imported_enabled = s.terrain_enabled = s.terrain_normal_maps_enabled = s.terrain_self_shadows_enabled = false;
    if (variant == "no-textures") s.surface_textures_enabled = false;
    if (variant == "no-normals") s.terrain_normal_maps_enabled = false;
    if (variant == "no-terrain-shadows") s.terrain_self_shadows_enabled = false;
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

// Oblique view of the Moon's shadow: the camera sits 3000 km (along the ground) to the side of where the shadow
// axis meets Earth, about 1000 km up, looking at that point near the limb. Long grazing rays cross the penumbral
// gradient, so shadow integration errors cover a large part of the frame.
inline Camera eclipse_oblique_camera(const ephemeris::SystemSnapshot& system) {
    Camera camera = earth_closeup_camera(system, true);
    const auto& body = system.body(camera.tracking_idx);
    auto hit = glm::normalize(camera.cam_pos_rel);
    auto sun = glm::normalize(system.body(system.star_index()).render.position.value-body.render.position.value);
    auto sunward = sun-hit*glm::dot(sun, hit);
    if (glm::length(sunward) < 1e-6) sunward = glm::cross(body.render.pole, hit);
    auto side = glm::normalize(glm::cross(hit, glm::normalize(sunward)));
    double radius = body.definition.radius_au();
    auto target = hit*ellipsoid_surface_radius(radius, body.definition.oblateness, body.render.pole, hit);
    double range = 3000.0*KM_TO_AU, elevation = 10.0*DEG_TO_RAD;
    auto position = target+(side*std::cos(elevation)+hit*std::sin(elevation))*range;
    auto aim = target+hit*(30.0*KM_TO_AU);
    camera.fov_deg = 30.0;
    auto up = glm::normalize(position);
    audit_look(camera, aim-position, glm::normalize(up-glm::normalize(aim-position)*glm::dot(up, glm::normalize(aim-position))));
    camera.cam_pos_rel = position;
    return camera;
}

inline glm::dvec3 audit_shadow_direction(const ephemeris::SystemSnapshot& system, int earth, double target_ratio = 1.7) {
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
        if (std::abs(ring_radius-target_ratio) < error) {
            best = direction;
            error = std::abs(ring_radius-target_ratio);
        }
    }
    if (error > 0.1) throw std::runtime_error("Audit ring shadow geometry is unavailable at this epoch.");
    return best;
}

// Grazing view along a latitude circle of Saturn's dayside, 15000 km from a point under the Cassini division's
// shadow and 2 degrees above its horizon. The equator lies to the right, so the ring shadow bands run left to right
// and converge toward the hazy limb; long sight lines cross several bands, which stresses shadow integration.
inline Camera saturn_limb_camera(const ephemeris::SystemSnapshot& system) {
    Camera camera;
    camera.horizon_align = false;
    camera.tracking_idx = system.find_body("Saturn");
    if (camera.tracking_idx < 0) throw std::runtime_error("Audit camera target missing.");
    const auto& body = system.body(camera.tracking_idx);
    auto pole = body.render.pole;
    auto normal = audit_shadow_direction(system, camera.tracking_idx, 1.98);
    double radius = body.definition.radius_au();
    auto target = normal*ellipsoid_surface_radius(radius, body.definition.oblateness, pole, normal);
    auto up = oblate_surface_normal(normal, body.definition.oblateness, pole);
    auto north = glm::normalize(pole-up*glm::dot(pole, up));
    auto equatorward = glm::dot(normal, pole) > 0 ? -north : north;
    auto forward = glm::normalize(glm::cross(up, equatorward));
    double range = 15000.0*KM_TO_AU, elevation = 2.0*DEG_TO_RAD;
    auto position = target-forward*(range*std::cos(elevation))+up*(range*std::sin(elevation));
    camera.fov_deg = 7.6;
    auto look = glm::normalize(target-position);
    auto camera_up = glm::normalize(position);
    audit_look(camera, look, glm::normalize(camera_up-look*glm::dot(camera_up, look)));
    camera.cam_pos_rel = position;
    return camera;
}

// Near-equinox ring shadow fixtures: the simulation advances until the Sun is this many degrees from Earth's
// equator. 0.15 degrees is inside the Sun's own angular radius, so its disk straddles the ring plane.
inline double equinox_fixture_elevation(std::string_view scene) {
    return scene.ends_with("-1deg") ? 1.0 : 0.15;
}

inline double sun_equator_elevation(const ephemeris::SystemSnapshot& system) {
    const auto& body = system.body(system.find_body("Earth"));
    auto sun = glm::normalize(system.body(system.star_index()).render.position.value-body.render.position.value);
    return std::asin(std::clamp(glm::dot(sun, body.render.pole), -1.0, 1.0))*RAD_TO_DEG;
}

// 40 km above the point 35 degrees of longitude before local noon where the ring shadow falls at 1.7 radii, looking
// along the equator toward noon and 8 degrees down, so the thin equinox shadow band runs to the horizon and the
// edge-on ring rises from it.
inline Camera equinox_camera(const ephemeris::SystemSnapshot& system) {
    Camera camera;
    camera.horizon_align = false;
    camera.tracking_idx = system.find_body("Earth");
    const auto& body = system.body(camera.tracking_idx);
    auto pole = body.render.pole;
    auto sun = glm::normalize(system.body(system.star_index()).render.position.value-body.render.position.value);
    double elevation = glm::dot(sun, pole);
    auto equatorial = glm::normalize(sun-pole*elevation);
    auto east = glm::normalize(glm::cross(pole, equatorial));
    double longitude = -35.0*DEG_TO_RAD, side = elevation < 0.0 ? 1.0 : -1.0;
    auto meridian = equatorial*std::cos(longitude)+east*std::sin(longitude);
    double radius = body.definition.radius_au(), error = 1e9;
    glm::dvec3 best = meridian;
    for (int i = 1; i <= 40000; ++i) {
        double latitude = i*0.00025*DEG_TO_RAD;
        auto normal = meridian*std::cos(latitude)+pole*(side*std::sin(latitude));
        auto p = normal*(ellipsoid_surface_radius(radius, body.definition.oblateness, pole, normal)/radius);
        double distance = -glm::dot(p, pole)/elevation;
        if (distance <= 0.0) continue;
        double ring = glm::length(p+sun*distance);
        if (std::abs(ring-1.7) < error) {
            error = std::abs(ring-1.7);
            best = normal;
        }
    }
    if (error > 0.05) throw std::runtime_error("Equinox ring shadow geometry is unavailable.");
    auto up = oblate_surface_normal(best, body.definition.oblateness, pole);
    auto along = glm::normalize(-equatorial*std::sin(longitude)+east*std::cos(longitude));
    along = glm::normalize(along-up*glm::dot(along, up));
    camera.cam_pos_rel = best*(ellipsoid_surface_radius(radius, body.definition.oblateness, pole, best)+40.0*KM_TO_AU);
    camera.fov_deg = 60.0;
    audit_look(camera, along*std::cos(8*DEG_TO_RAD)-up*std::sin(8*DEG_TO_RAD), up);
    return camera;
}

inline Camera audit_camera(const ephemeris::SystemSnapshot& system, std::string_view scene) {
    if (scene.starts_with("earth-equinox")) return equinox_camera(system);
    if (scene == "earth-full" || scene == "earth-full-live" || scene == "earth-eclipse") return earth_closeup_camera(system, scene == "earth-eclipse");
    if (scene == "earth-eclipse-oblique") return eclipse_oblique_camera(system);
    if (scene == "saturn-limb") return saturn_limb_camera(system);
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
    if (scene.starts_with("earth-backlit")) {
        double radius = body.definition.radius_au(), angular = std::asin(1.0/60.0);
        auto side = glm::normalize(glm::cross(body.render.pole, sun));
        auto forward = glm::normalize(sun*std::cos(0.2*angular)+side*std::sin(0.2*angular));
        camera.cam_pos_rel = -forward*(60.0*radius);
        camera.fov_deg = scene == "earth-backlit-zoom" ? 3.6 : 34.5;
        audit_look(camera, forward, glm::normalize(body.render.pole-forward*glm::dot(body.render.pole, forward)));
        return camera;
    }
    if (scene == "earth-sky-rings-south") {
        // Noon at 34 degrees latitude in the Sun's hemisphere, thousands of km from the ring shadow on the other one.
        // Looks equatorward 25 degrees up, so the lit rings fill the sky; no shadowed haze is anywhere in view.
        auto pole = body.render.pole;
        double side = glm::dot(sun, pole) >= 0 ? 1.0 : -1.0;
        auto equatorial = glm::normalize(sun-pole*glm::dot(sun, pole));
        auto normal = glm::normalize(equatorial*std::cos(34*DEG_TO_RAD)+pole*(side*std::sin(34*DEG_TO_RAD)));
        double radius = body.definition.radius_au();
        auto up = oblate_surface_normal(normal, body.definition.oblateness, pole);
        camera.cam_pos_rel = normal*(ellipsoid_surface_radius(radius, body.definition.oblateness, pole, normal)+2*KM_TO_AU);
        camera.fov_deg = 45.0;
        auto equatorward = glm::normalize(-pole*side-up*glm::dot(-pole*side, up));
        audit_look(camera, equatorward*std::cos(25*DEG_TO_RAD)+up*std::sin(25*DEG_TO_RAD), up);
        return camera;
    }
    if (scene.starts_with("earth-far-rings")) {
        // The ringed Earth seen whole from N radii (default 10) on the day side, 25 degrees above the ring plane on the
        // shadowed hemisphere's side, so the ring shadow bands cross a disc a few hundred pixels across.
        double range = scene.size() > 16 ? std::stod(std::string(scene.substr(16))) : 10.0;
        auto pole = body.render.pole;
        auto equatorial = glm::normalize(sun-pole*glm::dot(sun, pole));
        double side = glm::dot(sun, pole) >= 0 ? -1.0 : 1.0;
        auto normal = glm::normalize(equatorial*std::cos(25*DEG_TO_RAD)+pole*(side*std::sin(25*DEG_TO_RAD)));
        camera.cam_pos_rel = normal*(range*body.definition.radius_au());
        camera.fov_deg = 45.0;
        audit_look(camera, -normal, glm::normalize(pole-normal*glm::dot(pole, normal)));
        return camera;
    }
    if (scene.starts_with("earth-sky-rings")) {
        auto normal = audit_shadow_direction(system, camera.tracking_idx, 2.45);
        double radius = body.definition.radius_au();
        auto up = oblate_surface_normal(normal, body.definition.oblateness, body.render.pole);
        camera.cam_pos_rel = normal*(ellipsoid_surface_radius(radius, body.definition.oblateness, body.render.pole, normal)+2*KM_TO_AU);
        camera.fov_deg = 70.0;
        if (scene == "earth-sky-rings-away") {
            auto poleward = glm::normalize(-body.render.pole-up*glm::dot(-body.render.pole, up));
            if (glm::dot(normal, body.render.pole) > 0) poleward = glm::normalize(body.render.pole-up*glm::dot(body.render.pole, up));
            audit_look(camera, poleward*std::cos(35*DEG_TO_RAD)+up*std::sin(35*DEG_TO_RAD), up);
        } else {
            auto look = glm::normalize(sun);
            audit_look(camera, look, glm::normalize(up-look*glm::dot(up, look)));
        }
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
