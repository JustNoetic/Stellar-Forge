#pragma once

#include "physics/atmosphere_physics.hpp"

namespace sf::physics {

inline constexpr uint32_t ECLIPSE_SHADOW_WIDTH = 256;
inline constexpr uint32_t ECLIPSE_SHADOW_HEIGHT = 64;
inline constexpr double ECLIPSE_SOURCE_MIN = .002;
inline constexpr double ECLIPSE_SOURCE_MAX = 4;

struct EclipseRay {
    double impact = 0, bend = 0;
    glm::dvec3 transmission{1};
};

struct EclipseProfile {
    double radius_km = 0, top = 1, max_bend = 0;
    std::vector<EclipseRay> rays;
};

struct EclipseShadowTable {
    float support = 1;
    float max_bend = 0;
    std::vector<glm::vec4> pixels;
};

inline glm::dvec3 eclipse_medium(const PreparedAtmosphere& a, double radius) {
    double h = radius-a.radius_km;
    double x = std::clamp((std::asinh(h/a.profile_domain.z)-a.profile_domain.x)*a.profile_domain.y, 0.0, 1.0)
             *double(ATMOSPHERE_PROFILE_SAMPLES-1);
    size_t i = std::min(size_t(x), ATMOSPHERE_PROFILE_SAMPLES-2);
    double density = std::exp(std::lerp(double(a.log_density[i].x), double(a.log_density[i+1].x), x-double(i)));
    glm::dvec3 extinction = glm::dvec3(a.rayleigh+a.mixed_absorption)*density
        +glm::dvec3(a.mie_extinction)*std::exp(-std::max(h, 0.0)/a.mie_scale_height_km);
    for (const auto& layer : a.aerosols) extinction += glm::dvec3(layer.extinction)*aerosol_density(layer.shape, h);
    double z = (h-a.inputs.ozone_peak_km)/std::max(a.inputs.ozone_width_km, .001);
    return extinction+glm::dvec3(a.ozone_absorption)*std::exp(-.5*z*z);
}

inline EclipseProfile prepare_eclipse_profile(const PreparedAtmosphere& a, uint32_t samples = 128, uint32_t columns = 64) {
    EclipseProfile result;
    if (!a.refraction || samples < 2 || columns == 0) return result;
    const auto& r = *a.refraction;
    result.radius_km = r.ground_radius_km;
    result.top = r.top_radius_km/result.radius_km;
    double bottom = std::max(r.ground_radius_km, r.critical_radius_km > 0 ? r.critical_radius_km+.001 : 0);
    if (bottom >= r.top_radius_km) return result;
    double scale = std::max(r.scale_km, .01);
    double span = std::asinh((r.top_radius_km-bottom)/scale);
    for (uint32_t i = 0; i < samples; ++i) {
        double radius = bottom+scale*std::sinh(span*double(i)/double(samples-1));
        if (i+1 == samples) { result.rays.push_back({result.top, 0, glm::dvec3(1)}); break; }
        auto ng = r.index_and_gradient(radius);
        double impact = ng.x*radius;
        if (!result.rays.empty() && impact/result.radius_km <= result.rays.back().impact) continue;
        double turn = r.turning_radius(impact);
        if (!std::isfinite(turn) || std::abs(turn-radius) > .01) continue;
        double height = ng.y < 0 && ng.x > 1 ? -(ng.x-1)/ng.y : a.surface.scale_height_km;
        height = std::max(height, .01);
        double bend = std::clamp((ng.x-1)*std::sqrt(2*std::numbers::pi*radius/height), 0.0, .1);
        glm::dvec3 tau(0);
        double extent = std::sqrt(std::max((r.top_radius_km-radius)*(r.top_radius_km+radius), 0.0));
        for (uint32_t j = 0; j < columns; ++j) {
            double t = (double(j)+.5)/columns, s = extent*t*t;
            tau += eclipse_medium(a, std::hypot(radius, s))*(4*extent*t/columns);
        }
        result.rays.push_back({impact/result.radius_km, bend, glm::exp(-glm::max(tau, glm::dvec3(0)))});
        result.max_bend = std::max(result.max_bend, bend);
    }
    return result;
}

inline double eclipse_circle_light(double radius, double source, double separation) {
    if (radius <= 0 || separation >= radius+source) return 1;
    if (source <= 0) return 0;
    if (separation <= std::abs(radius-source)) return std::max(0.0, 1-std::min(radius*radius/(source*source), 1.0));
    double a = std::acos(std::clamp((separation*separation+radius*radius-source*source)/(2*separation*radius), -1.0, 1.0));
    double b = std::acos(std::clamp((separation*separation+source*source-radius*radius)/(2*separation*source), -1.0, 1.0));
    double product = (-separation+radius+source)*(separation+radius-source)*(separation-radius+source)*(separation+radius+source);
    return std::clamp(1-(radius*radius*a+source*source*b-.5*std::sqrt(std::max(product, 0.0)))/(std::numbers::pi*source*source), 0.0, 1.0);
}

inline double eclipse_source_fraction(double y, double separation, double source) {
    if (y+separation <= source) return 1;
    if (std::abs(y-separation) >= source || y <= 0 || separation <= 0) return 0;
    return std::acos(std::clamp(((y-source)*(y+source)+separation*separation)/(2*y*separation), -1.0, 1.0))/std::numbers::pi;
}

inline glm::dvec3 eclipse_shadow_correction(const EclipseProfile& profile, double alpha, double source, double separation) {
    if (profile.rays.size() < 2 || alpha <= 0 || source <= 0) return glm::dvec3(0);
    constexpr std::array<glm::dvec2, 4> quadrature{{
        {-.8611363115940526, .3478548451374538}, {-.3399810435848563, .6521451548625461},
        {.3399810435848563, .6521451548625461}, {.8611363115940526, .3478548451374538}
    }};
    double distance = source/alpha;
    glm::dvec3 sum(eclipse_circle_light(profile.top, source, separation)-eclipse_circle_light(1, source, separation));
    for (size_t i = 1; i < profile.rays.size(); ++i) {
        const auto& a = profile.rays[i-1]; const auto& b = profile.rays[i];
        double y0 = a.impact-distance*a.bend, y1 = b.impact-distance*b.bend;
        double dy = y1-y0;
        for (int side : {-1, 1}) {
            double lo = std::max(separation-source, 0.0), hi = separation+source;
            if (side < 0) { double temporary = lo; lo = -hi; hi = -temporary; }
            double first, last;
            if (std::abs(dy) < 1e-14) {
                if (y0 < lo || y0 >= hi) continue;
                first = 0; last = 1;
            } else {
                double t0 = (lo-y0)/dy, t1 = (hi-y0)/dy;
                first = std::max(0.0, std::min(t0, t1)); last = std::min(1.0, std::max(t0, t1));
                if (first >= last) continue;
            }
            for (const auto& q : quadrature) {
                double angle = .5*std::numbers::pi*q.x;
                double t = .5*(first+last)+.5*(last-first)*std::sin(angle);
                double p = std::lerp(a.impact, b.impact, t), y = std::abs(std::lerp(y0, y1, t));
                double weight = q.y*.25*std::numbers::pi*(last-first)*std::cos(angle);
                double flux = 2*p*(b.impact-a.impact)*weight*eclipse_source_fraction(y, separation, source)/(source*source);
                sum += glm::mix(a.transmission, b.transmission, t)*flux;
            }
        }
    }
    return glm::clamp(sum, glm::dvec3(-1), glm::dvec3(8));
}

inline double eclipse_contact_curve(double x) {
    return x*x/(x*x+(1-x)*(1-x));
}

inline double eclipse_contact_inverse(double x) {
    double a = std::sqrt(std::clamp(x, 0.0, 1.0)), b = std::sqrt(std::clamp(1-x, 0.0, 1.0));
    return a/(a+b);
}

inline double eclipse_shadow_v(double x, double q, double support) {
    double inner = std::abs(1-q)/(1+q);
    if (x <= .375) return inner*eclipse_contact_curve(x/.375);
    if (x <= .75) return inner+(1-inner)*eclipse_contact_curve((x-.375)/.375);
    return 1+(support-1)*std::pow((x-.75)/.25, 2);
}

inline glm::dvec2 eclipse_shadow_uv(double q, double v, double support) {
    double x, inner = std::abs(1-q)/(1+q);
    if (v < inner) x = .375*eclipse_contact_inverse(v/inner);
    else if (v <= 1) x = .375+.375*eclipse_contact_inverse((v-inner)/(1-inner));
    else x = .75+.25*std::sqrt((v-1)/(support-1));
    constexpr double pivot = ECLIPSE_SHADOW_HEIGHT*3/4;
    double row = q <= 1 ? pivot*std::log(q/ECLIPSE_SOURCE_MIN)/std::log(1/ECLIPSE_SOURCE_MIN)
                        : pivot+(ECLIPSE_SHADOW_HEIGHT-1-pivot)*std::log(q)/std::log(ECLIPSE_SOURCE_MAX);
    return {x, row/(ECLIPSE_SHADOW_HEIGHT-1)};
}

inline EclipseShadowTable bake_eclipse_shadow(const EclipseProfile& profile, double alpha) {
    EclipseShadowTable result;
    result.max_bend = float(profile.max_bend);
    result.support = float(std::max(profile.top, profile.max_bend/alpha)*1.001+.01);
    result.pixels.resize(ECLIPSE_SHADOW_WIDTH*ECLIPSE_SHADOW_HEIGHT);
    for (uint32_t y = 0; y < ECLIPSE_SHADOW_HEIGHT; ++y) {
        constexpr uint32_t pivot = ECLIPSE_SHADOW_HEIGHT*3/4;
        double q = y <= pivot ? ECLIPSE_SOURCE_MIN*std::pow(1/ECLIPSE_SOURCE_MIN, double(y)/pivot)
                              : std::pow(ECLIPSE_SOURCE_MAX, double(y-pivot)/(ECLIPSE_SHADOW_HEIGHT-1-pivot));
        for (uint32_t x = 0; x < ECLIPSE_SHADOW_WIDTH; ++x) {
            double v = eclipse_shadow_v(double(x)/double(ECLIPSE_SHADOW_WIDTH-1), q, result.support);
            auto correction = eclipse_shadow_correction(profile, alpha, q, v*(1+q));
            auto visibility = glm::dvec3(eclipse_circle_light(1, q, v*(1+q)))+correction;
            result.pixels[size_t(y)*ECLIPSE_SHADOW_WIDTH+x] = glm::vec4(glm::vec3(glm::log2(glm::max(visibility, glm::dvec3(1e-12)))), 0);
        }
    }
    return result;
}

}
