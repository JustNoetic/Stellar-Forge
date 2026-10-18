#pragma once

#include "ephemeris/atmosphere.hpp"
#include "physics/particle_optics.hpp"
#include <algorithm>
#include <array>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace sf::physics {

inline constexpr double GAS_CONSTANT = 8.31446261815324;
inline constexpr double BOLTZMANN = 1.380649e-23;
inline constexpr size_t ATMOSPHERE_PROFILE_SAMPLES = 256;

struct GasOptics {
    const char* name;
    double molar_mass, refractive_index, king_factor;
    glm::dvec3 absorption;
};

inline const std::array<GasOptics, 11> ATMOSPHERIC_GASES{{
    {"N2", .0280134, 1.000298, 1.034, {}}, {"O2", .0319988, 1.000271, 1.096, {}},
    {"Ar", .039948, 1.000281, 1.0, {}}, {"CO2", .0440095, 1.000450, 1.15, {}},
    {"CH4", .0160425, 1.000444, 1.0, {3.50986e-30, 6.70107e-31, 8.25387234042553e-32}},
    {"H2", .00201588, 1.000132, 1.02, {}}, {"He", .004002602, 1.000036, 1.0, {}},
    {"H2O", .01801528, 1.000252, 1.001, {}}, {"NH3", .01703052, 1.000376, 1.0, {}},
    {"O3", .0479982, 1.000520, 1.03, {1.209e-25, 3.5e-25, 1.582e-26}},
    {"SO2", .0640638, 1.000686, 1.075, {1e-29, 5e-29, 3e-28}}
}};

struct GasState {
    double pressure_pa = 0, temperature_k = 0, gravity_m_s2 = 0;
    double number_density_m3 = 0, density_kg_m3 = 0, scale_height_km = 0;
};

inline double mixture_molar_mass(const ephemeris::AtmosphereDefinition& a) {
    double total = 0, mass = 0;
    for (const auto& [name, weight] : a.composition) {
        auto gas = std::find_if(ATMOSPHERIC_GASES.begin(), ATMOSPHERIC_GASES.end(), [&](const auto& g) { return name == g.name; });
        if (gas == ATMOSPHERIC_GASES.end() || !std::isfinite(weight) || weight < 0) throw std::invalid_argument("Invalid gas mixture");
        total += weight;
        mass += weight*gas->molar_mass;
    }
    if (!(total > 0)) throw std::invalid_argument("Empty gas mixture");
    return mass/total;
}

inline double atmosphere_temperature(const ephemeris::AtmosphereDefinition& a, double height) {
    if (a.temperature_layers.empty()) return a.temperature_k;
    if (height <= a.temperature_layers.front().x) return a.temperature_layers.front().y;
    for (size_t i = 1; i < a.temperature_layers.size(); ++i) {
        auto low = a.temperature_layers[i-1], high = a.temperature_layers[i];
        if (height < high.x) return std::lerp(low.y, high.y, (height-low.x)/(high.x-low.x));
    }
    return a.temperature_layers.back().y;
}

inline double hydrostatic_log_pressure(const ephemeris::AtmosphereDefinition& a, double gravity, double radius_km, double height) {
    double coefficient = mixture_molar_mass(a)*gravity*1000/GAS_CONSTANT;
    if (a.temperature_layers.empty())
        return -coefficient*radius_km*height/((radius_km+height)*a.temperature_k);
    auto derivative = [&](double z) {
        double factor = radius_km/(radius_km+z);
        return -coefficient*factor*factor/atmosphere_temperature(a, z);
    };
    double result = 0, lower = 0;
    auto interval = [&](double upper) {
        int count = std::max(1, int(std::ceil(std::abs(upper-lower)/2.0)));
        double step = (upper-lower)/count;
        for (int i = 0; i < count; ++i) {
            double z = lower+i*step;
            result += step*(derivative(z)+4*derivative(z+step*.5)+derivative(z+step))/6;
        }
        lower = upper;
    };
    if (height >= 0) {
        for (auto knot : a.temperature_layers) if (knot.x > 0 && knot.x < height) interval(knot.x);
    } else {
        for (auto i = a.temperature_layers.rbegin(); i != a.temperature_layers.rend(); ++i)
            if (i->x < 0 && i->x > height) interval(i->x);
    }
    interval(height);
    return result;
}

inline GasState sample_atmosphere(const ephemeris::AtmosphereDefinition& a, double mass_solar, double radius_km, double height_km) {
    if (!ephemeris::valid_atmosphere(a) || !std::isfinite(mass_solar) || mass_solar <= 0 ||
        !std::isfinite(radius_km) || radius_km <= 0 || !std::isfinite(height_km) || height_km <= -radius_km)
        throw std::invalid_argument("Hydrostatic atmosphere requires valid inputs and positive body mass/radius");
    double g0 = 6.67430e-11*1.98847e30*mass_solar/(radius_km*radius_km*1e6);
    double t = atmosphere_temperature(a, height_km);
    double g = g0*std::pow(radius_km/(radius_km+height_km), 2);
    double p = a.pressure_pa*std::exp(std::clamp(hydrostatic_log_pressure(a, g0, radius_km, height_km), -700.0, 80.0));
    double molar_mass = mixture_molar_mass(a);
    return {p, t, g, p/(BOLTZMANN*t), p*molar_mass/(GAS_CONSTANT*t), GAS_CONSTANT*t/(molar_mass*g*1000)};
}

struct PreparedAtmosphere {
    ephemeris::AtmosphereDefinition inputs;
    double mass_solar = 0, radius_km = 0;
    float height_km = 100;
    GasState surface;
    glm::vec3 rayleigh{}, mixed_absorption{}, ozone_absorption{};
    glm::vec4 profile_domain{};
    std::array<glm::vec4, ATMOSPHERE_PROFILE_SAMPLES> log_density{};
    glm::vec3 mie_scattering{}, mie_extinction{};
    float mie_scale_height_km = 1.2f, mie_asymmetry = .8f;
    std::vector<PreparedAerosol> aerosols;
};

inline std::shared_ptr<const PreparedAtmosphere> prepare_atmosphere(const ephemeris::AtmosphereDefinition& a,
                                                                   double mass_solar, double radius_km) {
    auto result = std::make_shared<PreparedAtmosphere>();
    result->inputs = a;
    result->mass_solar = mass_solar;
    result->radius_km = radius_km;
    result->surface = sample_atmosphere(a, mass_solar, radius_km, 0);
    result->mie_scattering = a.mie_scattering;
    result->mie_extinction = a.mie_extinction;
    result->mie_scale_height_km = a.mie_scale_height_km;
    result->mie_asymmetry = a.mie_asymmetry;
    double total = 0;
    for (const auto& [_, weight] : a.composition) total += weight;
    if (a.automatic_aerosols) {
        result->mie_scattering = result->mie_extinction = glm::vec3(0);
        if (a.pressure_pa > 0) for (const auto& layer : a.aerosols)
            result->aerosols.push_back(prepare_aerosol(layer, a.bottom_altitude_km));
    }
    constexpr double wavelengths[] = {680e-9, 550e-9, 440e-9};
    double reference_density = 101325.0/(BOLTZMANN*273.15);
    glm::dvec3 rayleigh(0), absorption(0);
    for (const auto& gas : ATMOSPHERIC_GASES) {
        auto entry = a.composition.find(gas.name);
        if (entry == a.composition.end()) continue;
        double fraction = entry->second/total;
        double ll = (gas.refractive_index*gas.refractive_index-1)/(gas.refractive_index*gas.refractive_index+2);
        for (int c = 0; c < 3; ++c)
            rayleigh[c] += fraction*24*std::pow(std::numbers::pi, 3)*ll*ll*gas.king_factor/
                (std::pow(wavelengths[c], 4)*reference_density*reference_density);
        absorption += fraction*gas.absorption;
    }
    result->rayleigh = rayleigh*result->surface.number_density_m3*1000.0;
    result->mixed_absorption = absorption*result->surface.number_density_m3*1000.0;
    double integral_m = a.ozone_width_km*1000*std::sqrt(std::numbers::pi/2)*
        std::erfc(-a.ozone_peak_km/(std::sqrt(2.0)*a.ozone_width_km));
    double ozone_peak_density = a.pressure_pa > 0 ? a.ozone_column_du*2.687e20/integral_m : 0;
    result->ozone_absorption = glm::dvec3(1.209e-25, 3.5e-25, 1.582e-26)*ozone_peak_density*1000.0;
    double top = a.height_km;
    if (a.automatic_height) {
        top = std::max(1.0, a.ozone_column_du > 0 ? a.ozone_peak_km+6*a.ozone_width_km : 1.0);
        if (!a.temperature_layers.empty()) top = std::max(top, a.temperature_layers.back().x);
        for (const auto& layer : a.aerosols) if (layer.width_km > 0) top = std::max(top, layer.peak_km+8*layer.width_km);
        double molecular = glm::max(result->rayleigh.r+result->mixed_absorption.r,
            glm::max(result->rayleigh.g+result->mixed_absorption.g, result->rayleigh.b+result->mixed_absorption.b));
        double aerosol = a.pressure_pa > 0 ? glm::max(result->mie_extinction.r, glm::max(result->mie_extinction.g, result->mie_extinction.b)) : 0;
        for (int i = 0; i < 128 && top < 10000; ++i) {
            auto state = sample_atmosphere(a, mass_solar, radius_km, top);
            double relative = std::exp(hydrostatic_log_pressure(a, result->surface.gravity_m_s2, radius_km, top))*a.temperature_k/state.temperature_k;
            double residual = molecular*relative*std::sqrt(2*std::numbers::pi*(radius_km+top)*state.scale_height_km)+
                aerosol*std::exp(-top/result->mie_scale_height_km)*std::sqrt(2*std::numbers::pi*(radius_km+top)*result->mie_scale_height_km);
            for (const auto& layer : result->aerosols) {
                double scale_height = layer.shape.z > 0 ? layer.shape.z : layer.shape.x;
                residual += glm::max(layer.extinction.r, glm::max(layer.extinction.g, layer.extinction.b))*
                    aerosol_density(layer.shape, top)*std::sqrt(2*std::numbers::pi*(radius_km+top)*scale_height);
            }
            if (residual < 1e-5) break;
            top = std::min(10000.0, top*1.1+1);
        }
    }
    result->height_km = float(std::min(top, 10000.0));
    double scale = std::max(.01, std::min(10.0, result->surface.scale_height_km));
    double minimum = std::max(-radius_km*.25, std::min(-50.0, a.bottom_altitude_km));
    double start = std::asinh(minimum/scale), span = std::asinh(result->height_km/scale)-start;
    result->profile_domain = {start, 1/span, scale, 1.0};
    for (size_t i = 0; i < ATMOSPHERE_PROFILE_SAMPLES; ++i) {
        double height = scale*std::sinh(start+span*double(i)/double(ATMOSPHERE_PROFILE_SAMPLES-1));
        double t = atmosphere_temperature(a, height);
        result->log_density[i].x = float(hydrostatic_log_pressure(a, result->surface.gravity_m_s2, radius_km, height)+std::log(a.temperature_k/t));
    }
    return result;
}

inline ephemeris::AtmosphereDefinition earth_physical_atmosphere() {
    ephemeris::AtmosphereDefinition a;
    a.enabled = true;
    a.physical = true;
    a.automatic_aerosols = false;
    a.ozone_column_du = 300;
    a.mie_scattering = glm::vec3(.002);
    a.mie_extinction = glm::vec3(.002);
    a.mie_scale_height_km = 1.2f;
    constexpr double heights[] = {0, 11, 20, 32, 47, 51, 71, 84.852};
    constexpr double temperatures[] = {288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946};
    for (int i = 0; i < 8; ++i) a.temperature_layers.emplace_back(6356.766*heights[i]/(6356.766-heights[i]), temperatures[i]);
    return a;
}

}
