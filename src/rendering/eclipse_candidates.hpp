#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <glm/glm.hpp>

namespace sf::rendering {

inline bool ring_could_shadow_surface(glm::vec4 receiver, glm::vec4 star,
                                     glm::vec4 center_outer, glm::vec4 pole_inner, float focal_pixels, float softness = 1.25f) {
    auto magnitude = glm::max(glm::max(glm::abs(glm::vec3(receiver)), glm::abs(glm::vec3(star))), glm::abs(glm::vec3(center_outer)));
    double padding = 32.0 * std::numeric_limits<float>::epsilon()
                   * std::max({double(magnitude.x), double(magnitude.y), double(magnitude.z), 1e-12});
    double radius = receiver.w + padding;
    glm::dvec3 origin(receiver), light(star), center(center_outer), pole(pole_inner);
    glm::dvec3 to_light = light - origin;
    double distance = glm::length(to_light);
    double denominator = glm::dot(to_light, pole);
    double minimum_denominator = std::abs(denominator) - radius - star.w;
    if (distance <= radius || minimum_denominator <= 1e-6 * (distance + radius)) return true;
    double plane_offset = glm::dot(center - origin, pole);
    if ((denominator > 0.0 ? plane_offset : -plane_offset) + radius < 0.0) return false;
    double fraction = plane_offset / denominator;
    glm::dvec3 hit = origin + to_light * fraction;
    double projected_radius = (std::abs(1.0 - fraction) * radius + std::abs(fraction) * star.w)
                            * (distance + radius + star.w) / minimum_denominator;
    double minimum_mu = minimum_denominator / std::max(distance + radius + star.w, 1e-5);
    double maximum_hit_distance = (std::abs(plane_offset) + radius) / minimum_mu;
    double stretch = 1.0 / minimum_mu;
    double stellar_blur = 2.0 * std::max(double(softness), 1.0) * star.w / std::max(distance - radius, 1e-5) * maximum_hit_distance * stretch;
    double pixel_blur = 0.75 * (glm::length(origin) + radius) / std::max(double(focal_pixels), 1.0) * stretch;
    double profile_blur = 0.75 * std::max(double(center_outer.w - pole_inner.w), 1e-9) / 4096.0;
    double reach = projected_radius + std::max({stellar_blur, pixel_blur, profile_blur}) + padding;
    double radial_distance = glm::length(hit - center);
    return radial_distance + reach >= pole_inner.w && radial_distance - reach <= center_outer.w;
}

inline bool eclipse_could_shadow(glm::vec4 receiver, glm::vec4 star, glm::vec4 caster) {
    auto magnitude = glm::max(glm::max(glm::abs(glm::vec3(receiver)), glm::abs(glm::vec3(star))), glm::abs(glm::vec3(caster)));
    double padding = 16.0 * std::numeric_limits<float>::epsilon()
                   * std::max({double(magnitude.x), double(magnitude.y), double(magnitude.z), 1e-12});
    double radius = receiver.w + padding;
    glm::dvec3 to_receiver = glm::dvec3(receiver) - glm::dvec3(star);
    glm::dvec3 to_caster = glm::dvec3(caster) - glm::dvec3(star);
    double distance = glm::length(to_receiver);
    if (distance <= radius) return true;
    double variation = glm::length(to_caster) * radius;
    double axial = glm::dot(to_caster, to_receiver);
    double maximum_distance = distance + radius;
    double maximum_distance2 = maximum_distance * maximum_distance;
    if (axial + variation < 0.0 || axial - variation > maximum_distance2) return false;
    double minimum_perpendicular = std::max(0.0, (glm::length(glm::cross(to_caster, to_receiver)) - variation) / maximum_distance);
    double maximum_projection = std::clamp(1.0 - (axial - variation) / maximum_distance2, 0.0, 1.0);
    return minimum_perpendicular <= caster.w + padding + star.w * maximum_projection;
}

}
