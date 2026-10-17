#pragma once

#include "core/camera.hpp"
#include <cmath>
#include <stdexcept>

namespace sf::benchmark {

inline constexpr double ECLIPSE_FIXTURE_YEARS = 1.565;

inline Camera earth_closeup_camera(const ephemeris::SystemSnapshot& system, bool eclipse) {
    int earth = system.find_body("Earth");
    int moon = system.find_body("Moon");
    if (earth < 0 || moon < 0) throw std::runtime_error("Earth close-up requires Earth and Moon.");
    auto center = system.body(earth).render.position.value;
    auto sun = system.body(system.star_index()).render.position.value;
    double radius = system.body(earth).definition.radius_au();
    glm::dvec3 direction = glm::normalize(sun - center);
    if (eclipse) {
        auto caster = system.body(moon).render.position.value;
        auto axis = glm::normalize(caster - sun);
        auto offset = caster - center;
        double along = glm::dot(offset, axis);
        double perpendicular2 = glm::dot(offset, offset) - along * along;
        double discriminant = radius * radius - perpendicular2;
        if (discriminant <= 0.0 || along >= 0.0)
            throw std::runtime_error("Moon shadow axis misses Earth at the eclipse fixture epoch.");
        direction = glm::normalize(offset + axis * (-along - std::sqrt(discriminant)));

    }
    Camera camera;
    camera.tracking_idx = earth;
    camera.horizon_align = false;
    camera.cam_pos_rel = direction * (radius * 1.1);
    return camera;
}

}
