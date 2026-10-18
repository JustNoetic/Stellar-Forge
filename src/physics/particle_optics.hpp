#pragma once

#include "ephemeris/atmosphere.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <numbers>

namespace sf::physics {

inline constexpr size_t AEROSOL_PHASE_SAMPLES = 128;

struct MieSphere {
    double extinction = 0, scattering = 0;
    double size_parameter = 0;
    std::vector<std::complex<double>> electric, magnetic;

    double intensity(double mu) const {
        std::complex<double> s1{}, s2{};
        double previous = 0, current = 1;
        for (size_t i = 0; i < electric.size(); ++i) {
            double n = double(i+1), tau = n*mu*current-(n+1)*previous;
            double factor = (2*n+1)/(n*(n+1));
            s1 += factor*(electric[i]*current+magnetic[i]*tau);
            s2 += factor*(electric[i]*tau+magnetic[i]*current);
            double next = ((2*n+1)*mu*current-(n+1)*previous)/n;
            previous = current;
            current = next;
        }
        return (std::norm(s1)+std::norm(s2))/(2*std::numbers::pi*size_parameter*size_parameter);
    }
};

inline MieSphere mie_sphere(std::complex<double> index, double x) {
    if (!(x > 0) || x > 2000 || index.real() <= 0 || index.imag() > 0)
        throw std::invalid_argument("Invalid Mie sphere");
    MieSphere sphere;
    sphere.size_parameter = x;
    int count = int(std::ceil(x+4*std::cbrt(x)+2));
    int depth = std::max(count, int(std::ceil(std::abs(index*x))))+40;
    std::vector<std::complex<double>> derivative(depth+1);
    for (int n = depth; n > 0; --n) {
        auto term = double(n)/(index*x);
        derivative[n-1] = term-1.0/(derivative[n]+term);
    }
    double previous = std::sin(x), current = previous/x-std::cos(x);
    std::complex<double> xi_previous(previous, std::cos(x));
    std::complex<double> xi_current(current, std::cos(x)/x+std::sin(x));
    for (int n = 1; n <= count; ++n) {
        auto da = derivative[n]/index+double(n)/x;
        auto db = derivative[n]*index+double(n)/x;
        auto a = (da*current-previous)/(da*xi_current-xi_previous);
        auto b = (db*current-previous)/(db*xi_current-xi_previous);
        sphere.electric.push_back(a);
        sphere.magnetic.push_back(b);
        sphere.extinction += (2*n+1)*std::real(a+b)*2/(x*x);
        sphere.scattering += (2*n+1)*(std::norm(a)+std::norm(b))*2/(x*x);
        double next = (2*n+1)*current/x-previous;
        auto xi_next = double(2*n+1)*xi_current/x-xi_previous;
        previous = current; current = next;
        xi_previous = xi_current; xi_current = xi_next;
    }
    return sphere;
}

inline std::complex<double> particle_refractive_index(const std::string& material, int channel) {
    if (material == "mineral_dust") {
        constexpr double absorption[] = {.001, .003, .012};
        return {1.5, -absorption[channel]};
    }
    if (material == "organic_haze") {
        constexpr double absorption[] = {.004, .02, .06};
        return {1.65, -absorption[channel]};
    }
    if (material == "water") return {1.333, -1e-8};
    if (material == "ice") return {1.31, -1e-8};
    if (material == "sulfate") return {1.44, -1e-8};
    if (material == "ammonia_ice") return {1.44, -1e-8};
    if (material == "methane_ice") return {1.30, -1e-8};
    if (material == "ammonium_hydrosulfide") return {1.70, -1e-5};
    if (material == "hydrogen_sulfide_ice") return {1.55, -1e-8};
    throw std::invalid_argument("Unknown particle material");
}

struct PreparedAerosol {
    glm::vec4 shape{};
    glm::vec4 scattering{}, extinction{};
    std::array<glm::vec4, AEROSOL_PHASE_SAMPLES> phase{};
};

inline double aerosol_density(const glm::vec4& shape, double height) {
    return shape.z > 0 ? std::exp(-.5*std::pow((height-shape.y)/shape.z, 2)) : std::exp(-std::max(height, 0.0)/shape.x);
}

inline PreparedAerosol prepare_aerosol(const ephemeris::AerosolLayer& layer, double bottom_altitude_km = 0) {
    if (!ephemeris::valid_aerosol(layer)) throw std::invalid_argument("Invalid aerosol layer");
    PreparedAerosol result;
    result.shape = {layer.scale_height_km, layer.peak_km, layer.width_km, 0};
    constexpr double wavelengths[] = {.680, .550, .440};
    glm::dvec3 extinction(0), scattering(0);
    std::array<glm::dvec3, AEROSOL_PHASE_SAMPLES> intensity{};
    double variance = std::log1p(layer.effective_variance), sigma = std::sqrt(variance);
    double center = std::log(layer.effective_radius_um)-2.5*variance;
    for (int bin = 0; bin < 48; ++bin) {
        double z = -4.5+9*(bin+.5)/48;
        double radius = std::exp(center+z*sigma);
        double weight = std::exp(-.5*z*z)*radius*radius;
        for (int c = 0; c < 3; ++c) {
            auto sphere = mie_sphere(particle_refractive_index(layer.material, c), 2*std::numbers::pi*radius/wavelengths[c]);
            extinction[c] += weight*sphere.extinction;
            scattering[c] += weight*sphere.scattering;
            for (size_t i = 0; i < AEROSOL_PHASE_SAMPLES; ++i) {
                double t = double(i)/(AEROSOL_PHASE_SAMPLES-1);
                intensity[i][c] += weight*sphere.intensity(std::cos(std::numbers::pi*t*t));
            }
        }
    }
    double column = layer.width_km > 0 ? layer.width_km*std::sqrt(std::numbers::pi/2)*
        std::erfc((bottom_altitude_km-layer.peak_km)/(std::sqrt(2.0)*layer.width_km)) : layer.scale_height_km-std::min(bottom_altitude_km, 0.0);
    double normalization = layer.optical_depth_550/(column*extinction.g);
    result.extinction = glm::vec4(extinction*normalization, 0);
    result.scattering = glm::vec4(scattering*normalization, 0);
    glm::dvec3 phase_integral(0);
    glm::dvec3 phase_moment(0);
    for (size_t i = 1; i < AEROSOL_PHASE_SAMPLES; ++i) {
        double a = double(i-1)/(AEROSOL_PHASE_SAMPLES-1), b = double(i)/(AEROSOL_PHASE_SAMPLES-1);
        double delta = std::cos(std::numbers::pi*a*a)-std::cos(std::numbers::pi*b*b);
        phase_integral += std::numbers::pi*delta*(intensity[i-1]+intensity[i]);
        double mu_a = std::cos(std::numbers::pi*a*a), mu_b = std::cos(std::numbers::pi*b*b);
        phase_moment += std::numbers::pi*delta/3.0*((2*mu_a+mu_b)*intensity[i-1]+(mu_a+2*mu_b)*intensity[i]);
    }
    if (layer.material == "ice" || layer.material.ends_with("_ice") || layer.material == "ammonium_hydrosulfide") {
        auto g = glm::clamp(phase_moment/phase_integral, glm::dvec3(-.98), glm::dvec3(.98));
        for (size_t i = 0; i < AEROSOL_PHASE_SAMPLES; ++i) {
            double t = double(i)/(AEROSOL_PHASE_SAMPLES-1), mu = std::cos(std::numbers::pi*t*t);
            intensity[i] = (glm::dvec3(1)-g*g)/(4*std::numbers::pi*glm::pow(glm::dvec3(1)+g*g-2.0*g*mu, glm::dvec3(1.5)));
        }
        phase_integral = glm::dvec3(0);
        for (size_t i = 1; i < AEROSOL_PHASE_SAMPLES; ++i) {
            double a = double(i-1)/(AEROSOL_PHASE_SAMPLES-1), b = double(i)/(AEROSOL_PHASE_SAMPLES-1);
            double delta = std::cos(std::numbers::pi*a*a)-std::cos(std::numbers::pi*b*b);
            phase_integral += std::numbers::pi*delta*(intensity[i-1]+intensity[i]);
        }
    }
    for (size_t i = 0; i < AEROSOL_PHASE_SAMPLES; ++i) result.phase[i] = glm::vec4(intensity[i]/phase_integral, 0);
    return result;
}

}
