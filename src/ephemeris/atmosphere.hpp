#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>

namespace sf::ephemeris {

inline constexpr size_t MAX_AEROSOL_LAYERS = 4;

struct AerosolLayer {
    std::string material = "mineral_dust";
    double optical_depth_550 = 0.0;
    double effective_radius_um = 1.5;
    double effective_variance = 0.2;
    double scale_height_km = 12.0;
    double peak_km = 0.0;
    double width_km = 0.0;
    bool operator==(const AerosolLayer&) const = default;
};

inline bool valid_aerosol(const AerosolLayer& p) {
    return (p.material == "mineral_dust" || p.material == "organic_haze" || p.material == "water" ||
        p.material == "ice" || p.material == "sulfate" || p.material == "ammonia_ice" ||
        p.material == "ammonium_hydrosulfide" || p.material == "methane_ice" || p.material == "hydrogen_sulfide_ice") &&
        std::isfinite(p.optical_depth_550) && p.optical_depth_550 >= 0 && p.optical_depth_550 <= 1000 &&
        std::isfinite(p.effective_radius_um) && p.effective_radius_um >= .005 && p.effective_radius_um <= 10 &&
        std::isfinite(p.effective_variance) && p.effective_variance >= .01 && p.effective_variance <= .5 &&
        std::isfinite(p.scale_height_km) && p.scale_height_km >= .1 && p.scale_height_km <= 1000 &&
        std::isfinite(p.peak_km) && p.peak_km >= -1000 && p.peak_km <= 9000 &&
        std::isfinite(p.width_km) && p.width_km >= 0 && p.width_km <= 1000 &&
        (p.width_km == 0 ? p.peak_km == 0 : p.width_km >= .1 && p.peak_km+8*p.width_km <= 10000);
}

inline void to_json(nlohmann::json& j, const AerosolLayer& p) {
    j = {{"material", p.material}, {"optical_depth_550", p.optical_depth_550},
        {"effective_radius_um", p.effective_radius_um}, {"effective_variance", p.effective_variance}};
    if (p.width_km > 0) { j["peak_km"] = p.peak_km; j["width_km"] = p.width_km; }
    else j["scale_height_km"] = p.scale_height_km;
}

inline void from_json(const nlohmann::json& j, AerosolLayer& p) {
    p = AerosolLayer{};
    p.material = j.at("material").get<std::string>();
    p.optical_depth_550 = j.at("optical_depth_550").get<double>();
    p.effective_radius_um = j.value("effective_radius_um", p.effective_radius_um);
    p.effective_variance = j.value("effective_variance", p.effective_variance);
    p.scale_height_km = j.value("scale_height_km", p.scale_height_km);
    p.peak_km = j.value("peak_km", 0.0);
    p.width_km = j.value("width_km", 0.0);
    if ((j.contains("peak_km") != j.contains("width_km")) ||
        (j.contains("width_km") && (p.width_km <= 0 || j.contains("scale_height_km"))) || !valid_aerosol(p))
        throw std::invalid_argument("Invalid aerosol layer");
}

struct AtmosphereDefinition {
    bool enabled = false;
    bool physical = false;
    bool automatic_height = true;
    bool automatic_aerosols = true;
    double pressure_pa = 101325.0;
    double temperature_k = 288.15;
    bool pressure_reference = false;
    double bottom_altitude_km = 0;
    std::map<std::string, double> composition{{"N2", 0.78}, {"O2", 0.21}, {"Ar", 0.01}};
    std::vector<glm::dvec2> temperature_layers;
    std::vector<AerosolLayer> aerosols;
    double ozone_column_du = 0.0;
    double ozone_peak_km = 25.0;
    double ozone_width_km = 7.0;
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
    if (!std::isfinite(a.bottom_altitude_km) || a.bottom_altitude_km > 0 || a.bottom_altitude_km < -1000 ||
        (a.pressure_reference && !a.physical) || (!a.pressure_reference && a.bottom_altitude_km != 0)) return false;
    if (a.aerosols.size() > MAX_AEROSOL_LAYERS || (!a.aerosols.empty() && !(a.physical && a.automatic_aerosols))) return false;
    for (const auto& p : a.aerosols) if (!valid_aerosol(p)) return false;
    if (!std::isfinite(a.pressure_pa) || a.pressure_pa < 0.0 || a.pressure_pa > 1e9 ||
        !std::isfinite(a.temperature_k) || a.temperature_k < 1.0 || a.temperature_k > 10000.0 ||
        !std::isfinite(a.ozone_column_du) || a.ozone_column_du < 0.0 || a.ozone_column_du > 100000.0 ||
        !std::isfinite(a.ozone_peak_km) || a.ozone_peak_km < 0.0 || a.ozone_peak_km > 10000.0 ||
        (a.ozone_column_du > 0 && !(a.physical && a.automatic_height) && a.ozone_peak_km > a.height_km) ||
        !std::isfinite(a.ozone_width_km) || a.ozone_width_km <= 0.0 || a.ozone_width_km > 1000.0 ||
        a.temperature_layers.size() > 64 || a.composition.empty() || a.composition.size() > 11) return false;
    double total = 0.0;
    for (const auto& [gas, weight] : a.composition) {
        if (gas != "N2" && gas != "O2" && gas != "Ar" && gas != "CO2" && gas != "CH4" &&
            gas != "H2" && gas != "He" && gas != "H2O" && gas != "NH3" && gas != "O3" && gas != "SO2") return false;
        if (!std::isfinite(weight) || weight < 0.0 || weight > 1e6) return false;
        total += weight;
    }
    if (total <= 0.0) return false;
    double previous = -10001.0;
    bool reference_knot = a.temperature_layers.empty();
    for (const auto& layer : a.temperature_layers) {
        if (!std::isfinite(layer.x) || !std::isfinite(layer.y) || layer.x <= previous || layer.x < -1000 || layer.x > 10000.0 ||
            layer.y < 1.0 || layer.y > 10000.0) return false;
        if (layer.x == 0 && std::abs(layer.y-a.temperature_k) <= 1e-6) reference_knot = true;
        previous = layer.x;
    }
    if (!reference_knot) return false;
    for (const auto& p : a.aerosols) if (p.peak_km < a.bottom_altitude_km) return false;
    if (!std::isfinite(a.height_km) || a.height_km < 0.01f || a.height_km > 10000.0f ||
        !std::isfinite(a.rayleigh_scale_height_km) || a.rayleigh_scale_height_km < 0.001f || (!a.physical && a.rayleigh_scale_height_km > a.height_km) ||
        !std::isfinite(a.mie_scale_height_km) || a.mie_scale_height_km < 0.001f || (!a.physical && a.mie_scale_height_km > a.height_km) ||
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
    j = {{"schema_version", 2}, {"enabled", a.enabled}};
    j["model"] = a.physical ? "physical" : "optical";
    if (!a.automatic_height || !a.physical) j["height_km"] = a.height_km;
    if (!a.automatic_height) j["automatic_height"] = false;
    if (!a.physical) {
        j["rayleigh_scale_height_km"] = a.rayleigh_scale_height_km;
        j["rayleigh_scattering"] = {a.rayleigh_scattering.x, a.rayleigh_scattering.y, a.rayleigh_scattering.z};
    }
    if (!a.automatic_aerosols || !a.physical) {
        j["automatic_aerosols"] = a.automatic_aerosols;
        j["mie_scale_height_km"] = a.mie_scale_height_km;
        j["mie_scattering"] = {a.mie_scattering.x, a.mie_scattering.y, a.mie_scattering.z};
        j["mie_extinction"] = {a.mie_extinction.x, a.mie_extinction.y, a.mie_extinction.z};
        j["mie_asymmetry"] = a.mie_asymmetry;
    } else j["aerosols"] = a.aerosols;
    if (a.ground_albedo != glm::vec3(.3f)) j["ground_albedo"] = {a.ground_albedo.x, a.ground_albedo.y, a.ground_albedo.z};
    {
        if (a.pressure_reference) {
            j["reference_pressure_pa"] = a.pressure_pa;
            j["bottom_altitude_km"] = a.bottom_altitude_km;
        } else j["pressure_pa"] = a.pressure_pa;
        j["temperature_k"] = a.temperature_k;
        j["composition"] = a.composition;
        if (!a.temperature_layers.empty()) {
            j["temperature_layers"] = nlohmann::json::array();
            for (auto layer : a.temperature_layers) j["temperature_layers"].push_back({layer.x, layer.y});
        }
        if (a.ozone_column_du != 0 || a.ozone_peak_km != 25 || a.ozone_width_km != 7)
            j["ozone"] = {{"column_du", a.ozone_column_du}, {"peak_km", a.ozone_peak_km}, {"width_km", a.ozone_width_km}};
    }
}

inline void from_json(const nlohmann::json& j, AtmosphereDefinition& a) {
    a = AtmosphereDefinition{};
    int version = j.value("schema_version", 1);
    if (version < 1 || version > 2) throw std::invalid_argument("Unsupported atmosphere schema version");
    bool physical_data = (j.contains("pressure_pa") || j.contains("surface_pressure") || j.contains("reference_pressure_pa")) &&
        (j.contains("temperature_k") || j.contains("temperature")) &&
        j.contains("composition") && j.at("composition").is_object() && !j.at("composition").empty();
    a.enabled = j.value("enabled", physical_data);
    auto model = j.value("model", std::string(physical_data ? "physical" : "optical"));
    if (model != "optical" && model != "physical") throw std::invalid_argument("Unknown atmosphere model");
    a.physical = model == "physical";
    a.automatic_height = j.value("automatic_height", true);
    if (!j.contains("automatic_height") && j.contains("height_mode")) a.automatic_height = j.at("height_mode") == "auto";
    bool manual_aerosols = j.contains("mie_scattering") || j.contains("mie_extinction") || j.contains("beta_mie");
    a.automatic_aerosols = j.value("automatic_aerosols", j.value("aerosol_mode", std::string(manual_aerosols ? "manual" : "auto")) == "auto");
    a.pressure_pa = j.value("pressure_pa", j.value("surface_pressure", 1.0)*101325.0);
    if (j.contains("reference_pressure_pa")) {
        if (j.contains("pressure_pa") || j.contains("surface_pressure")) throw std::invalid_argument("Ambiguous pressure reference");
        a.pressure_reference = true;
        a.pressure_pa = j.at("reference_pressure_pa").get<double>();
    }
    a.bottom_altitude_km = j.value("bottom_altitude_km", 0.0);
    a.temperature_k = j.value("temperature_k", j.value("temperature", 288.15));
    a.composition = j.value("composition", a.composition);
    if (j.contains("aerosols")) a.aerosols = j.at("aerosols").get<std::vector<AerosolLayer>>();
    a.temperature_layers.clear();
    const char* layers_key = j.contains("temperature_layers") ? "temperature_layers" : "temperature_profile";
    if (j.contains(layers_key)) for (const auto& layer : j.at(layers_key))
        a.temperature_layers.emplace_back(layer.at(0).get<double>(), layer.at(1).get<double>());
    if (j.contains("ozone")) {
        const auto& o = j.at("ozone");
        a.ozone_column_du = o.value("column_du", 0.0);
        a.ozone_peak_km = o.value("peak_km", 25.0);
        a.ozone_width_km = o.value("width_km", 7.0);
    }
    a.height_km = j.value("height_km", a.height_km);
    a.rayleigh_scale_height_km = j.value("rayleigh_scale_height_km", a.rayleigh_scale_height_km);
    a.mie_scale_height_km = j.value("mie_scale_height_km", j.value("h_mie", a.mie_scale_height_km));
    a.mie_asymmetry = j.value("mie_asymmetry", j.value("mie_g", a.mie_asymmetry));
    auto read = [&](const char* key, glm::vec3& v) {
        if (!j.contains(key)) return;
        const auto& c = j.at(key);
        v = {c.at(0).get<float>(), c.at(1).get<float>(), c.at(2).get<float>()};
    };
    read("rayleigh_scattering", a.rayleigh_scattering);
    read("mie_scattering", a.mie_scattering);
    read("mie_extinction", a.mie_extinction);
    read("ground_albedo", a.ground_albedo);
    if (j.contains("beta_mie") && !j.contains("mie_extinction")) {
        double beta = j.at("beta_mie").get<double>()*1000.0;
        double alpha = j.contains("mie_angstrom") && !j.at("mie_angstrom").is_null() ? j.at("mie_angstrom").get<double>() : 1.2;
        constexpr double wavelengths[] = {680.0, 550.0, 440.0};
        glm::vec3 albedo(1.0f);
        if (j.contains("mie_albedo")) {
            const auto& value = j.at("mie_albedo");
            if (value.is_number()) albedo = glm::vec3(value.get<float>());
            else albedo = {value.at(0).get<float>(), value.at(1).get<float>(), value.at(2).get<float>()};
        }
        for (int c = 0; c < 3; ++c) a.mie_extinction[c] = float(beta*std::pow(wavelengths[c]/550.0, -alpha));
        a.mie_scattering = a.mie_extinction*albedo;
    }
    if (version == 1 && a.physical && a.automatic_aerosols && !j.contains("aerosols")) {
        double total = 0;
        for (auto [_, w] : a.composition) total += w;
        auto fraction = [&](const char* gas) { auto i = a.composition.find(gas); return i == a.composition.end() || total <= 0 ? 0.0 : i->second/total; };
        double pressure = a.pressure_pa/101325.0;
        if (fraction("CH4") > .02 && a.temperature_k < 140 && pressure > .5)
            a.aerosols.push_back({"organic_haze", 6*pressure/1.45, .2, .2, 50});
        else if (fraction("SO2") > .00005)
            a.aerosols.push_back({"sulfate", 10*pressure/92, 1, .2, 50});
        else if (fraction("CO2") > .8 && pressure < .05)
            a.aerosols.push_back({"mineral_dust", .48*pressure/.006, 1.5, .3, 12});
        else if (pressure > 0)
            a.aerosols.push_back({"water", .0024*pressure, .1, .2, 1.2});
    }
    if (!valid_atmosphere(a)) throw std::invalid_argument("Invalid atmosphere definition");
}

}
