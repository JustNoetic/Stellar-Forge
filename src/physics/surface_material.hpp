#pragma once

#include "core/constants.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <string>
#include <string_view>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

namespace sf::physics {

struct SurfaceMaterialDefinition {
    std::string preset = "auto";
    bool hapke = false;
    double single_scattering_albedo = 0.23;
    double phase_width = 0.35;
    double backscatter_fraction = 0.9;
    double opposition_strength = 1.0;
    double opposition_width = 0.06;
    double roughness_deg = 20.0;
    double brightness = 1.0;
    double coherent_strength = 0.15;
    double coherent_width = 0.006;
    bool operator==(const SurfaceMaterialDefinition&) const = default;
};

struct alignas(16) SurfaceMaterialGPU {
    glm::vec4 grain{0.0f, 0.23f, 0.35f, 0.9f};
    glm::vec4 surface{1.0f, 0.06f, 20.0f, 1.0f};
    glm::vec4 opposition{0.15f, 0.006f, 1.0f, 0.0f};
};

inline SurfaceMaterialDefinition material_preset(std::string_view name) {
    SurfaceMaterialDefinition m;
    m.preset = name;
    m.hapke = name == "lunar" || name == "icy" || name == "custom";
    if (name == "icy") {
        m.single_scattering_albedo = 0.85;
        m.phase_width = 0.3;
        m.backscatter_fraction = 0.75;
        m.opposition_strength = 0.5;
        m.opposition_width = 0.035;
        m.roughness_deg = 12.0;
        m.coherent_strength = 0.4;
        m.coherent_width = 0.008;
    }
    return m;
}

inline SurfaceMaterialDefinition resolve_material(const SurfaceMaterialDefinition& input, std::string_view name, bool stellar = false) {
    auto m = input;
    if (m.preset == "auto") {
        std::string lower(name);
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        m = material_preset(lower == "moon" || lower == "luna" ? "lunar" : lower == "europa" ? "icy" : "lambert");
    }
    if (stellar) m = material_preset("lambert");
    return m;
}

inline constexpr std::array<double SurfaceMaterialDefinition::*, 9> MATERIAL_FIELDS{
    &SurfaceMaterialDefinition::single_scattering_albedo, &SurfaceMaterialDefinition::phase_width,
    &SurfaceMaterialDefinition::backscatter_fraction, &SurfaceMaterialDefinition::opposition_strength,
    &SurfaceMaterialDefinition::opposition_width, &SurfaceMaterialDefinition::roughness_deg,
    &SurfaceMaterialDefinition::brightness, &SurfaceMaterialDefinition::coherent_strength,
    &SurfaceMaterialDefinition::coherent_width};
inline constexpr std::array<const char*, 9> MATERIAL_NAMES{"single_scattering_albedo", "phase_width", "backscatter_fraction",
    "opposition_strength", "opposition_width", "roughness_deg", "brightness", "coherent_strength", "coherent_width"};
inline constexpr std::array<double, 9> MATERIAL_MIN{0.001, 0.0, 0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 0.003};
inline constexpr std::array<double, 9> MATERIAL_MAX{0.999, 0.85, 1.0, 2.0, 0.5, 45.0, 2.0, 1.0, 0.1};

inline bool valid_material(const SurfaceMaterialDefinition& m) {
    if (m.preset != "auto" && m.preset != "lambert" && m.preset != "lunar" && m.preset != "icy" && m.preset != "custom") return false;
    for (size_t i = 0; i < MATERIAL_FIELDS.size(); ++i) {
        double v = m.*MATERIAL_FIELDS[i];
        if (!std::isfinite(v) || v < MATERIAL_MIN[i] || v > MATERIAL_MAX[i]) return false;
    }
    return true;
}

inline SurfaceMaterialDefinition parse_material(const nlohmann::json& value, std::string_view name) {
    auto raw = value.is_string() ? nlohmann::json{{"preset", value}} : value;
    if (!raw.is_object()) return {};
    auto preset = raw.value("preset", std::string("auto"));
    if (preset == "auto" && raw.size() <= (raw.contains("preset") ? 1u : 0u)) return {};
    auto m = resolve_material(material_preset(preset), name);
    m.preset = preset;
    if (raw.contains("model") && raw["model"].is_string()) {
        m.hapke = raw["model"] == "hapke";
        if (preset == "auto") m.preset = "custom";
    }
    for (size_t i = 0; i < MATERIAL_FIELDS.size(); ++i) {
        auto it = raw.find(MATERIAL_NAMES[i]);
        if (it == raw.end() || !it->is_number()) continue;
        double v = it->get<double>();
        if (std::isfinite(v)) m.*MATERIAL_FIELDS[i] = std::clamp(v, MATERIAL_MIN[i], MATERIAL_MAX[i]);
        if (preset == "auto") m.preset = "custom";
    }
    if (!valid_material(m)) return {};
    return m;
}

inline nlohmann::json serialize_material(const SurfaceMaterialDefinition& m) {
    if (m.preset == "auto") return {{"preset", "auto"}};
    nlohmann::json j{{"preset", m.preset}, {"model", m.hapke ? "hapke" : "lambert"}};
    for (size_t i = 0; i < MATERIAL_FIELDS.size(); ++i) j[MATERIAL_NAMES[i]] = m.*MATERIAL_FIELDS[i];
    return j;
}

inline double hapke_raw(double mu0, double mu, double cosine, const SurfaceMaterialGPU& m) {
    if (mu0 <= 0 || mu <= 0) return 0;
    mu0 = std::min(mu0, 1.0); mu = std::min(mu, 1.0);
    double c = std::clamp(cosine, -1.0, 1.0), b = m.grain.z, back = m.grain.w;
    double pf = (1-b*b)*((1-back)/std::pow(std::max(1e-9, 1+b*b+2*b*c), 1.5)
                        + back/std::pow(std::max(1e-9, 1+b*b-2*b*c), 1.5));
    double tangent = std::sqrt(std::max(0.0, (1-c)/std::max(1e-12, 1+c)));
    double sh = m.surface.x/(1+tangent/m.surface.y), x = tangent/m.opposition.y;
    double cb = m.opposition.x*(1+(x > 1e-5 ? -std::expm1(-x)/x : 1.0))/(2*(1+x)*(1+x));
    double u0 = mu0, u = mu, shadow = 1;
    if (m.surface.z > 0.001) {
        double t = std::tan(m.surface.z*PI/180), chi = 1/std::sqrt(1+PI*t*t);
        double si = std::sqrt(std::max(0.0, 1-mu0*mu0)), se = std::sqrt(std::max(0.0, 1-mu*mu));
        double ci = mu0/std::max(1e-10, si)/t, ce = mu/std::max(1e-10, se)/t;
        double e1i = std::exp(-2*ci/PI), e1e = std::exp(-2*ce/PI);
        double e2i = std::exp(-ci*ci/PI), e2e = std::exp(-ce*ce/PI);
        double u00 = chi*(mu0+si*t*e2i/(2-e1i)), u10 = chi*(mu+se*t*e2e/(2-e1e));
        double cp = si*se > 1e-10 ? std::clamp((c-mu0*mu)/(si*se), -1.0, 1.0) : 1;
        double psi = std::acos(cp), hs = 0.5*(1-cp);
        double f = std::exp(-2*std::sqrt(std::max(0.0, (1-cp)/std::max(1e-12, 1+cp)))), q;
        if (mu0 >= mu) {
            double den = std::max(1e-10, 2-e1e-psi/PI*e1i);
            u0 = chi*(mu0+si*t*(cp*e2e+hs*e2i)/den); u = chi*(mu+se*t*(e2e-hs*e2i)/den);
            q = chi*mu0/std::max(1e-10, u00);
        } else {
            double den = std::max(1e-10, 2-e1i-psi/PI*e1e);
            u0 = chi*(mu0+si*t*(e2i-hs*e2e)/den); u = chi*(mu+se*t*(cp*e2i+hs*e2e)/den);
            q = chi*mu/std::max(1e-10, u10);
        }
        shadow = u*mu0*chi/std::max(1e-10, u10*u00*(1-f+f*q));
    }
    double gamma = std::sqrt(1-m.grain.y);
    double h0 = (1+2*u0)/(1+2*u0*gamma), h = (1+2*u)/(1+2*u*gamma);
    return std::max(0.0, 0.25*m.grain.y*u0/std::max(1e-10, u0+u)*((1+sh)*pf+h0*h-1)*shadow*(1+cb));
}

inline SurfaceMaterialGPU pack_material(const SurfaceMaterialDefinition& m) {
    SurfaceMaterialGPU gpu;
    gpu.grain = {m.hapke ? 1.0f : 0.0f, float(m.single_scattering_albedo), float(m.phase_width), float(m.backscatter_fraction)};
    gpu.surface = {float(m.opposition_strength), float(m.opposition_width), float(m.roughness_deg), float(m.brightness)};
    gpu.opposition = {float(m.coherent_strength), float(m.coherent_width), 1.0f, 0.0f};
    if (m.hapke) gpu.opposition.z = float(std::cos(PI/6)/std::max(1e-8, hapke_raw(std::cos(PI/6), 1, std::cos(PI/6), gpu)));
    return gpu;
}

inline double surface_response(double mu0, double mu, double cosine, const SurfaceMaterialGPU& m) {
    return (m.grain.x > 0.5f ? hapke_raw(mu0, mu, cosine, m)*m.opposition.z : std::max(0.0, mu0))*m.surface.w;
}

inline constexpr uint32_t MATERIAL_PHASE_SAMPLES = 721;
using MaterialPhaseTable = std::array<float, MATERIAL_PHASE_SAMPLES>;

inline MaterialPhaseTable build_phase_table(const SurfaceMaterialGPU& m) {
    std::array<double, 24> nodes{}, weights{};
    for (size_t i = 0; i < nodes.size(); ++i) {
        double z = std::cos(PI*(double(i)+0.75)/(double(nodes.size())+0.5)), derivative = 0;
        for (int step = 0; step < 16; ++step) {
            double p0 = 1, p1 = z;
            for (size_t j = 2; j <= nodes.size(); ++j) { double p = ((2*j-1)*z*p1-(j-1)*p0)/j; p0 = p1; p1 = p; }
            derivative = nodes.size()*(z*p1-p0)/(z*z-1);
            double change = p1/derivative; z -= change;
            if (std::abs(change) < 1e-14) break;
        }
        nodes[i] = (z+1)*0.5; weights[i] = 1/((1-z*z)*derivative*derivative);
    }
    MaterialPhaseTable table{};
    for (size_t k = 0; k+1 < table.size(); ++k) {
        double angle = double(k)*PI/(table.size()-1), c = std::cos(angle), s = std::sin(angle);
        if (m.grain.x < 0.5f) { table[k] = float((2.0/3.0)*m.surface.w*(s+(PI-angle)*c)/PI); continue; }
        double total = 0, zmax = c < 0 ? std::max(0.0, s) : 1;
        for (size_t j = 0; j < nodes.size(); ++j) {
            double z = nodes[j]*zmax, radial = std::sqrt(1-z*z);
            double arc = s > 1e-12 ? std::acos(std::clamp(-z*c/std::max(1e-12, radial*s), -1.0, 1.0)) : PI;
            for (int a = 0; a < 48; ++a) {
                double mu0 = z*c+radial*s*std::cos(arc*(a+0.5)/48);
                total += surface_response(mu0, z, c, m)*z*weights[j]*zmax*arc;
            }
        }
        table[k] = float(total*2/(PI*48));
    }
    return table;
}

inline double lookup_phase(const MaterialPhaseTable& table, double angle) {
    double pos = std::clamp(angle/PI, 0.0, 1.0)*(table.size()-1);
    size_t lo = std::min(size_t(pos), table.size()-2);
    return table[lo]*(1-(pos-lo))+table[lo+1]*(pos-lo);
}

inline double disk_response(const MaterialPhaseTable& table, double cosine, double angular_radius = 0) {
    double c = std::clamp(cosine, -1.0, 1.0), angle = std::acos(c);
    if (angular_radius < 1e-5) return lookup_phase(table, angle);
    double beta = std::min(1.0, angular_radius)*std::sqrt(0.625), cb = std::cos(beta), sb = std::sin(beta);
    double s = std::sqrt(std::max(0.0, 1-c*c));
    return 0.2*(lookup_phase(table, angle)+lookup_phase(table, std::acos(std::clamp(c*cb+s*sb, -1.0, 1.0)))
        +lookup_phase(table, std::acos(std::clamp(c*cb-s*sb, -1.0, 1.0)))
        +2*lookup_phase(table, std::acos(std::clamp(c*cb, -1.0, 1.0))));
}

}
