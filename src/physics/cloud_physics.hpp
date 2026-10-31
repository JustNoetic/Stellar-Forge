#pragma once

#include "physics/atmosphere_physics.hpp"
#include <optional>

namespace sf::physics {

inline double water_log_saturation_pressure(double temperature) {
    double log_t = std::log(temperature);
    return 54.842763-6763.22/temperature-4.21*log_t+0.000367*temperature+
        std::tanh(0.0415*(temperature-218.8))*(53.878-1331.22/temperature-9.44523*log_t+0.014025*temperature);
}

inline std::optional<double> cloud_base_altitude(const PreparedAtmosphere& atmosphere, double humidity, double pressure_pa) {
    const auto& a = atmosphere.inputs;
    if (!a.enabled || !a.physical || !(atmosphere.surface.gravity_m_s2 > 0) || !(a.pressure_pa > 0) ||
        !std::isfinite(humidity) || humidity < 0 || humidity > 1 || !std::isfinite(pressure_pa) || pressure_pa < 0) return {};
    double minimum = std::max(0.0, atmosphere.opaque_altitude_km);
    double maximum = double(atmosphere.height_km);
    if (!(maximum > minimum)) return {};
    auto log_pressure = [&](double height) {
        return std::log(a.pressure_pa)+hydrostatic_log_pressure(a, atmosphere.surface.gravity_m_s2, atmosphere.radius_km, height);
    };
    double target;
    if (pressure_pa > 0) target = std::log(pressure_pa);
    else {
        if (a.pressure_reference || a.temperature_k < 180 || a.temperature_k > 330) return {};
        double total = 0, heat_capacity = 0;
        for (const auto& [gas, weight] : a.composition) {
            double cp = gas == "He" || gas == "Ar" ? 2.5 : gas == "CO2" || gas == "SO2" ? 4.5 :
                gas == "H2O" || gas == "CH4" || gas == "NH3" ? 4.0 : 3.5;
            total += weight;
            heat_capacity += weight*cp;
        }
        double vapor = humidity*std::exp(water_log_saturation_pressure(a.temperature_k));
        if (auto water = a.composition.find("H2O"); water != a.composition.end()) vapor = a.pressure_pa*water->second/total;
        if (!(vapor > 0) || vapor/a.pressure_pa > 0.1) return {};
        double exponent = total/heat_capacity;
        auto saturation = [&](double ratio) {
            double t = a.temperature_k*std::pow(ratio, exponent);
            return std::log(vapor)+std::log(ratio)-water_log_saturation_pressure(t);
        };
        double lower = std::pow(123.0/a.temperature_k, 1.0/exponent), upper = 1.0;
        if (saturation(lower) < 0) return {};
        if (saturation(upper) < 0) {
            for (int i = 0; i < 48; ++i) {
                double middle = (lower+upper)*0.5;
                if (saturation(middle) > 0) lower = middle;
                else upper = middle;
            }
        }
        target = std::log(a.pressure_pa)+std::log(upper);
    }
    if (target > log_pressure(minimum)+1e-9 || target < log_pressure(maximum)) return {};
    for (int i = 0; i < 48; ++i) {
        double middle = (minimum+maximum)*0.5;
        if (log_pressure(middle) > target) minimum = middle;
        else maximum = middle;
    }
    return (minimum+maximum)*0.5;
}

}
