#pragma once

#include "core/constants.hpp"
#include "rendering/surface_materials.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace sf::rendering {

inline constexpr size_t PLANETSHINE_LIGHTS = 4;

struct alignas(16) PlanetshineGPU {
    std::array<glm::vec4, PLANETSHINE_LIGHTS> direction{};
    std::array<glm::vec4, PLANETSHINE_LIGHTS> irradiance{};
};

inline double planetshine_phase(double cosine, double radius_distance) {
    double cutoff = -std::sqrt(std::max(0.0, 1.0-radius_distance*radius_distance));
    if (cosine <= cutoff) return 0.0;
    double mu = std::clamp(2.0*(cosine-cutoff)/(1.0-cutoff)-1.0, -1.0, 1.0);
    double beta = std::acos(-mu), beta2 = beta*beta;
    return beta < 0.1 ? beta*beta2*(1.0/3.0-beta2/30.0+beta2*beta2/840.0)/PI
        : (std::sin(beta)-beta*std::cos(beta))/PI;
}

inline std::vector<PlanetshineGPU> prepare_planetshine(const ephemeris::SystemSnapshot& system,
                                                      bool enabled, bool hdr, uint32_t limit = 1024,
                                                      SurfaceMaterialCache* prepared_materials = nullptr) {
    uint32_t count = std::min(uint32_t(system.size()), limit);
    std::vector<PlanetshineGPU> result(count);
    if (!enabled) return result;
    SurfaceMaterialCache fallback;
    auto& materials = prepared_materials ? *prepared_materials : fallback;
    if (!prepared_materials) materials.update(system, limit);
    struct SourceLight { glm::dvec3 direction{}, color{}; double angular_radius = 0; };
    std::vector<std::array<SourceLight, 16>> lighting(count);
    uint32_t star_count = 0;
    for (uint32_t j = 0; j < count; ++j) {
        auto source = system.body(j);
        if (source.definition.is_star || source.definition.is_black_hole || source.definition.is_spacecraft) continue;
        uint32_t star_slot = 0;
        for (uint32_t s = 0; s < count && star_slot < 16; ++s) {
            auto star = system.body(s);
            if (!star.definition.is_star) continue;
            auto& light = lighting[j][star_slot++];
            glm::dvec3 delta = star.render.position.value-source.render.position.value;
            double distance = glm::length(delta);
            if (distance <= 1e-12) continue;
            light.direction = delta/distance;
            light.angular_radius = std::asin(std::min(0.84, star.opaque_radius_au()/distance));
            double visibility = 1.0;
            for (uint32_t k = 0; k < count; ++k) {
                if (k == j || k == s) continue;
                auto caster = system.body(k);
                glm::dvec3 v = caster.render.position.value-source.render.position.value;
                double along = glm::dot(v, light.direction);
                if (along <= 0.0 || along >= distance) continue;
                double perpendicular = glm::length(glm::cross(v, light.direction));
                double source_radius = source.opaque_radius_au();
                double beta = caster.opaque_radius_au()/along, alpha = star.opaque_radius_au()/distance;
                double theta = source_radius/along, gamma = perpendicular/along;
                double outer = alpha+beta+theta, inner = std::max(0.0, std::abs(beta-alpha)-theta);
                double t = std::clamp((outer-gamma)/std::max(1e-15, outer-inner), 0.0, 1.0);
                double area = std::min(1.0, beta*beta/std::max(1e-30, alpha*alpha))
                    *std::min(1.0, std::pow(caster.opaque_radius_au()/std::max(source_radius, 1e-15), 2));
                visibility *= std::max(0.0, 1.0-area*t*t*(3-2*t));
            }
            double scale = hdr ? std::max(star.definition.lum_eq, 0.0)/(distance*distance) : 1.0;
            light.color = glm::dvec3(star.definition.color_rgb)*(scale*visibility);
        }
        star_count = std::max(star_count, star_slot);
    }
    if (star_count == 0) return result;
    for (uint32_t i = 0; i < count; ++i) {
        auto receiver = system.body(i);
        if (receiver.definition.is_star || receiver.definition.is_black_hole) continue;
        std::array<double, PLANETSHINE_LIGHTS> weights{};
        for (uint32_t j = 0; j < count; ++j) {
            if (i == j) continue;
            auto source = system.body(j);
            if (source.definition.is_star || source.definition.is_black_hole || source.definition.is_spacecraft) continue;
            glm::dvec3 delta = source.render.position.value-receiver.render.position.value;
            double distance2 = glm::dot(delta, delta), radius = source.opaque_radius_au();
            if (distance2 <= radius*radius*1.1025 || radius*radius < distance2*1e-8) continue;
            double distance = std::sqrt(distance2);
            glm::dvec3 direction = delta/distance, color(0.0), direction_moment(0.0);
            glm::dvec3 albedo = glm::clamp(glm::dvec3(source.definition.color_rgb), glm::dvec3(0.0), glm::dvec3(1.0));
            for (uint32_t s = 0; s < star_count; ++s) {
                const auto& light = lighting[j][s];
                if (glm::dot(light.color, light.color) == 0.0) continue;
                glm::dvec3 shifted = glm::normalize(delta+light.direction*(0.7*radius));
                double cosine = glm::dot(light.direction, -shifted);
                double cutoff = -std::sqrt(std::max(0.0, 1.0-radius*radius/distance2));
                if (cosine <= cutoff) continue;
                double mu = std::clamp(2*(cosine-cutoff)/(1-cutoff)-1, -1.0, 1.0);
                const auto& material = materials.material(j);
                double phase = material.grain.x > 0.5f ? physics::disk_response(materials.phase(j), mu, light.angular_radius)
                    : (2.0/3.0)*material.surface.w*planetshine_phase(cosine, radius/distance);
                glm::dvec3 contribution = light.color*phase;
                color += contribution;
                direction_moment += shifted*glm::dot(contribution*albedo, glm::dvec3(0.2126, 0.7152, 0.0722));
            }
            color *= albedo*(radius*radius/distance2);
            double weight = glm::dot(color, glm::dvec3(0.2126, 0.7152, 0.0722));
            auto weakest = std::min_element(weights.begin(), weights.end());
            if (weight <= *weakest) continue;
            size_t slot = size_t(weakest-weights.begin());
            *weakest = weight;
            result[i].direction[slot] = {glm::vec3(glm::dot(direction_moment, direction_moment) > 1e-30 ? glm::normalize(direction_moment) : direction), float(std::asin(std::min(radius/distance, 1.0)))};
            result[i].irradiance[slot] = {glm::vec3(color), 0.0f};
        }
    }
    return result;
}

}
