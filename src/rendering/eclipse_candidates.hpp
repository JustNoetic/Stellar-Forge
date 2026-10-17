#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <glm/glm.hpp>

namespace sf::rendering {

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
