#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

namespace sf::ephemeris {

struct AtmosphereDefinition {
    bool enabled = false;
    float height_km = 100.0f;
    float rayleigh_scale_height_km = 8.0f;
    float mie_scale_height_km = 1.2f;
    glm::vec3 rayleigh_scattering{0.005802f, 0.013558f, 0.033100f};
    glm::vec3 mie_scattering{0.003996f};
    glm::vec3 mie_extinction{0.004440f};
    float mie_asymmetry = 0.8f;
    glm::vec3 ground_albedo{0.3f};
    bool operator==(const AtmosphereDefinition&) const = default;
};

inline bool valid_atmosphere(const AtmosphereDefinition& a) {
    if (!std::isfinite(a.height_km) || a.height_km < 0.01f || a.height_km > 10000.0f ||
        !std::isfinite(a.rayleigh_scale_height_km) || a.rayleigh_scale_height_km < 0.001f || a.rayleigh_scale_height_km > a.height_km ||
        !std::isfinite(a.mie_scale_height_km) || a.mie_scale_height_km < 0.001f || a.mie_scale_height_km > a.height_km ||
        !std::isfinite(a.mie_asymmetry) || std::abs(a.mie_asymmetry) > 0.99f) return false;
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(a.rayleigh_scattering[i]) || a.rayleigh_scattering[i] < 0.0f || a.rayleigh_scattering[i] > 100.0f ||
            !std::isfinite(a.mie_scattering[i]) || a.mie_scattering[i] < 0.0f ||
            !std::isfinite(a.mie_extinction[i]) || a.mie_extinction[i] < a.mie_scattering[i] || a.mie_extinction[i] > 100.0f ||
            !std::isfinite(a.ground_albedo[i]) || a.ground_albedo[i] < 0.0f || a.ground_albedo[i] > 1.0f) return false;
    }
    return true;
}

inline void to_json(nlohmann::json& j, const AtmosphereDefinition& a) {
    j = {{"enabled", a.enabled}, {"height_km", a.height_km},
         {"rayleigh_scale_height_km", a.rayleigh_scale_height_km}, {"mie_scale_height_km", a.mie_scale_height_km},
         {"rayleigh_scattering", {a.rayleigh_scattering.x, a.rayleigh_scattering.y, a.rayleigh_scattering.z}},
         {"mie_scattering", {a.mie_scattering.x, a.mie_scattering.y, a.mie_scattering.z}},
         {"mie_extinction", {a.mie_extinction.x, a.mie_extinction.y, a.mie_extinction.z}},
         {"mie_asymmetry", a.mie_asymmetry}, {"ground_albedo", {a.ground_albedo.x, a.ground_albedo.y, a.ground_albedo.z}}};
}

inline void from_json(const nlohmann::json& j, AtmosphereDefinition& a) {
    a.enabled = j.value("enabled", false);
    a.height_km = j.value("height_km", a.height_km);
    a.rayleigh_scale_height_km = j.value("rayleigh_scale_height_km", a.rayleigh_scale_height_km);
    a.mie_scale_height_km = j.value("mie_scale_height_km", a.mie_scale_height_km);
    a.mie_asymmetry = j.value("mie_asymmetry", a.mie_asymmetry);
    auto read = [&](const char* key, glm::vec3& v) {
        if (!j.contains(key)) return;
        const auto& c = j.at(key);
        v = {c.at(0).get<float>(), c.at(1).get<float>(), c.at(2).get<float>()};
    };
    read("rayleigh_scattering", a.rayleigh_scattering);
    read("mie_scattering", a.mie_scattering);
    read("mie_extinction", a.mie_extinction);
    read("ground_albedo", a.ground_albedo);
}

}
