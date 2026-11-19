#pragma once

#include "physics/atmosphere_refraction.hpp"
#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace sf::rendering {

inline constexpr uint32_t REFRACTION_PROFILE_TEXELS_X = 1024;
inline constexpr uint32_t REFRACTION_OBSERVER_WIDTH = 512;
inline constexpr uint32_t REFRACTION_OBSERVER_HEIGHT = 128;
inline constexpr uint32_t REFRACTION_EXIT_CELLS = 4096;
inline constexpr uint32_t ATMOSPHERE_REFRACTION_SAMPLES = 64;

struct RefractionHost {
    int body = -1;
    double camera_radius_km = 0.0;
    double ground_radius_km = 0.0;
    double max_displacement = 0.0;
    double lowest_visible_chord_km = 0.0;
    glm::dvec3 center{0.0};
    glm::dvec3 pole{0.0, 1.0, 0.0};
    double stretch = 0.0;
    const physics::PreparedRefraction* refraction = nullptr;
};

struct AtmosphereRefraction {
    glm::dmat3 apparent{1.0};
    double pad = 0.0;
    bool differential = false;
    glm::dvec3 host_pole{0.0, 1.0, 0.0};
    double host_stretch = 0.0;
    std::array<glm::dvec3, 3> host_apparent_basis{};
    std::array<glm::dvec3, 3> host_true_basis{};
    glm::vec4 trig{};
    glm::vec4 domain{};
    std::array<glm::vec4, 16> bend{};
};

inline glm::dmat3 refraction_rotation_matrix(glm::dvec3 original, glm::dvec3 apparent) {
    glm::dvec3 axis = glm::cross(original, apparent);
    double s = glm::length(axis), c = glm::dot(original, apparent);
    if (s < 1e-15) return glm::dmat3(1.0);
    axis /= s;
    glm::dmat3 k(0.0, axis.z, -axis.y, -axis.z, 0.0, axis.x, axis.y, -axis.x, 0.0);
    return glm::dmat3(1.0)+s*k+(1.0-c)*k*k;
}

inline glm::dmat3 apparent_rotation(const RefractionHost& host, glm::dvec3 position_au) {
    glm::dmat3 identity(1.0);
    if (host.body < 0 || !host.refraction) return identity;
    const auto& r = *host.refraction;
    auto stretched = [&](glm::dvec3 v) { return v+host.pole*(glm::dot(v, host.pole)*host.stretch); };
    glm::dvec3 a = stretched(position_au)*149597870.7;
    double distance = glm::length(a);
    if (!(distance > 0.0)) return identity;
    glm::dvec3 v = a/distance;
    glm::dvec3 up = glm::normalize(stretched(-host.center));
    double mu = std::clamp(glm::dot(v, up), -1.0, 1.0);
    glm::dvec3 side = up-v*mu;
    double sine = glm::length(side);
    double camera = host.camera_radius_km;
    if (sine < 1e-12 || (camera > r.top_radius_km && (mu >= 0.0 || camera*sine >= r.top_radius_km))) return identity;
    double b = camera*sine, along = -camera*mu;
    double target = std::sqrt(b*b+(distance-along)*(distance-along));
    double polar = std::atan2(along, b)+std::atan2(distance-along, b);
    auto solution = r.solve(camera, target, polar, 1, distance > .05*r.ground_radius_km);
    double rotation = std::atan2(sine, mu)-solution.apparent_zenith;
    if (!std::isfinite(rotation)) return identity;
    rotation = std::clamp(rotation, -host.max_displacement, host.max_displacement);
    glm::dvec3 lifted = v*std::cos(rotation)+side/sine*std::sin(rotation);
    glm::dvec3 apparent = glm::normalize(lifted-host.pole*(glm::dot(lifted, host.pole)*host.stretch/(1.0+host.stretch)));
    return refraction_rotation_matrix(glm::normalize(position_au), apparent);
}

inline double refraction_horizon_zenith(const physics::PreparedRefraction& r, double camera_radius_km) {
    bool outside = camera_radius_km > r.top_radius_km;
    double n_c = outside ? 1.0 : r.index(camera_radius_km, 1);
    double p_cam = n_c*camera_radius_km;
    if (!outside && !r.monotone.y) {
        double p_barrier = p_cam;
        for (size_t i = 0; i < r.node_radii.size(); ++i) {
            double rad = r.node_radii[i];
            if (rad > camera_radius_km && rad < r.top_radius_km)
                p_barrier = std::min(p_barrier, r.node_invariant[i].y);
        }
        if (p_barrier < p_cam) return std::asin(std::clamp(p_barrier/p_cam, 0.0, 1.0));
    }
    if (camera_radius_km > r.ground_radius_km) {
        double p_h = r.horizon_impact(camera_radius_km, 1);
        return std::numbers::pi-std::asin(std::clamp(p_h/p_cam, 0.0, 1.0));
    }
    return 0.5*std::numbers::pi;
}

inline double forward_refraction_bend(const physics::PreparedRefraction& r, double camera_radius_km,
                                      double theta_app, double distance_km, bool& ok) {
    ok = true;
    double s = std::sin(theta_app);
    if (s < 1e-12) return 0.0;
    bool outside = camera_radius_km > r.top_radius_km;
    double n_c = outside ? 1.0 : r.index(camera_radius_km, 1);
    double p = n_c*camera_radius_km*s;
    if (outside) {
        if (theta_app <= 0.5*std::numbers::pi || p >= r.top_radius_km) return 0.0;
        double entry = -camera_radius_km*std::cos(theta_app)-std::sqrt(std::max(0.0, r.top_radius_km*r.top_radius_km-p*p));
        if (distance_km <= entry) return 0.0;
    }
    double phi_exit = r.reach_travel(camera_radius_km, theta_app, r.top_radius_km, ok, 1, true, true);
    if (!ok || !std::isfinite(phi_exit)) {
        ok = false;
        return 0.0;
    }
    double psi_exit = std::asin(std::clamp(p/r.top_radius_km, 0.0, 1.0));
    double theta_inf = phi_exit+psi_exit;
    double delta_inf = theta_inf-theta_app;
    if (outside) {
        delta_inf = 2.0*(r.table_travel(p, r.top_radius_km)-std::acos(std::clamp(p/r.top_radius_km, 0.0, 1.0)));
        theta_inf = theta_app+delta_inf;
    }
    if (!std::isfinite(distance_km)) return delta_inf;
    double d_perp = camera_radius_km*std::sin(theta_inf)-p;
    double cos_psi = std::sqrt(std::max(0.0, 1.0-(p/r.top_radius_km)*(p/r.top_radius_km)));
    double b_along = r.top_radius_km*cos_psi-camera_radius_km*std::cos(theta_inf);
    double exit_dist_sq = d_perp*d_perp+b_along*b_along;
    if (distance_km*distance_km <= exit_dist_sq || std::abs(d_perp) >= distance_km)
        return std::max(0.0, delta_inf*std::clamp((distance_km-std::max(0.0, -camera_radius_km*std::cos(theta_app)))/std::max(distance_km, 1e-6), 0.0, 1.0));
    return delta_inf+std::asin(std::clamp(d_perp/distance_km, -1.0, 1.0));
}

inline AtmosphereRefraction prepare_atmosphere_refraction(const RefractionHost& host, glm::dvec3 position_au,
                                                        double outer_km, double focal) {
    AtmosphereRefraction out;
    if (host.body < 0 || !host.refraction) return out;
    const auto& r = *host.refraction;
    auto to_host = [&](glm::dvec3 v) { return v+host.pole*(glm::dot(v, host.pole)*host.stretch); };
    auto from_host = [&](glm::dvec3 v) { return v-host.pole*(glm::dot(v, host.pole)*host.stretch/(1.0+host.stretch)); };
    glm::dvec3 a_B = to_host(position_au)*149597870.7;
    double D_H = glm::length(a_B);
    double outer_H = outer_km*(1.0+host.stretch);
    if (!(D_H > 0.0) || D_H <= std::max(outer_H*1.05, 0.05*r.ground_radius_km)) {
        out.apparent = apparent_rotation(host, position_au);
        return out;
    }
    glm::dvec3 v_B = a_B/D_H;
    glm::dvec3 up_H = glm::normalize(to_host(-host.center));
    double mu_B = std::clamp(glm::dot(v_B, up_H), -1.0, 1.0);
    glm::dvec3 side_B = up_H-v_B*mu_B;
    double sine_B = glm::length(side_B);
    if (sine_B < 1e-6) {
        out.apparent = apparent_rotation(host, position_au);
        return out;
    }
    double r_c = host.camera_radius_km;
    bool outside = r_c > r.top_radius_km;
    double theta_top = outside ? (std::numbers::pi-std::asin(std::clamp(r.top_radius_km/r_c, 0.0, 1.0))) : 0.0;
    double alpha_B = std::asin(std::clamp(outer_H/D_H, 0.0, 1.0));
    double alpha_pad = alpha_B+std::max(0.02*alpha_B, 2.0/std::max(focal, 1.0));
    double theta_true_c = std::atan2(sine_B, mu_B);
    double theta_true_lo = std::max(0.0, theta_true_c-alpha_pad);
    double theta_true_hi = std::min(std::numbers::pi, theta_true_c+alpha_pad);
    if (outside) {
        if (theta_true_hi <= theta_top || D_H+outer_H <= r_c-r.top_radius_km) return out;
        if (theta_true_hi < std::numbers::pi) {
            double sin_hi = r_c*std::sin(theta_true_hi);
            if (sin_hi < r.top_radius_km) {
                double entry_hi = -r_c*std::cos(theta_true_hi)-std::sqrt(std::max(0.0, r.top_radius_km*r.top_radius_km-sin_hi*sin_hi));
                if (D_H+outer_H <= entry_hi) return out;
            }
        }
    }
    double theta_horiz = refraction_horizon_zenith(r, r_c);
    double theta_horiz_safe = std::max(theta_top, theta_horiz-1e-7);
    bool ok_h = false;
    double delta_horiz = forward_refraction_bend(r, r_c, theta_horiz_safe, D_H, ok_h);
    if (!ok_h) {
        double lo = theta_top, hi = theta_horiz_safe;
        if (!forward_refraction_bend(r, r_c, lo, D_H, ok_h) || !ok_h) {
            out.apparent = apparent_rotation(host, position_au);
            return out;
        }
        for (int k = 0; k < 20; ++k) {
            double mid = 0.5*(lo+hi);
            bool ok_m = false;
            forward_refraction_bend(r, r_c, mid, D_H, ok_m);
            if (ok_m) lo = mid;
            else hi = mid;
        }
        theta_horiz_safe = lo;
        delta_horiz = forward_refraction_bend(r, r_c, theta_horiz_safe, D_H, ok_h);
        if (!ok_h) {
            out.apparent = apparent_rotation(host, position_au);
            return out;
        }
    }
    double theta_true_max_visible = theta_horiz_safe+delta_horiz;
    if (theta_true_lo >= theta_true_max_visible && (!outside || D_H >= -r_c*std::cos(theta_true_lo))) {
        out.apparent = apparent_rotation(host, position_au);
        return out;
    }
    auto solve_apparent = [&](double theta_true) {
        if (outside && theta_true <= theta_top) return theta_true;
        double b = r_c*std::sin(theta_true), along = -r_c*std::cos(theta_true);
        double target_r = std::hypot(b, D_H-along);
        double polar = std::atan2(along, b)+std::atan2(D_H-along, b);
        auto sol = r.solve(r_c, target_r, polar, 1, true);
        return std::isfinite(sol.apparent_zenith) ? sol.apparent_zenith : theta_true;
    };
    double theta_app_lo = std::min(solve_apparent(theta_true_lo), theta_horiz_safe);
    double theta_app_hi = theta_true_hi >= theta_true_max_visible ? theta_horiz_safe : std::min(solve_apparent(theta_true_hi), theta_horiz_safe);
    double theta_min = std::max(theta_app_lo, theta_top);
    double theta_max = std::min(theta_app_hi, theta_horiz_safe);
    if (theta_max <= theta_min+1e-12) {
        out.apparent = apparent_rotation(host, position_au);
        return out;
    }
    bool ok_lo = false, ok_hi = false;
    double delta_min = (outside && theta_min <= theta_top+1e-12) ? 0.0 : forward_refraction_bend(r, r_c, theta_min, D_H, ok_lo);
    double delta_max = (outside && theta_max <= theta_top+1e-12) ? 0.0 : forward_refraction_bend(r, r_c, theta_max, D_H, ok_hi);
    if (outside && theta_min <= theta_top+1e-12) ok_lo = true;
    if (outside && theta_max <= theta_top+1e-12) ok_hi = true;
    double spread = (ok_lo && ok_hi) ? std::abs(delta_max-delta_min) : 0.0;
    bool center_hidden = theta_true_c > theta_true_max_visible && (!outside || D_H >= -r_c*std::cos(theta_true_c));
    if (spread*focal < 0.05 && !center_hidden) {
        out.apparent = apparent_rotation(host, position_au);
        out.pad = spread;
        return out;
    }
    double theta_0 = center_hidden ? 0.5*(theta_app_lo+theta_max) : std::min(solve_apparent(theta_true_c), theta_horiz_safe);
    bool ok_ref = false;
    double delta_ref = (outside && theta_0 <= theta_top) ? 0.0 : forward_refraction_bend(r, r_c, theta_0, D_H, ok_ref);
    if (!ok_ref && !(outside && theta_0 <= theta_top)) delta_ref = 0.5*(delta_min+delta_max);
    double theta_t0 = theta_0+delta_ref;
    glm::dvec3 m_H = glm::normalize(v_B-up_H*mu_B);
    glm::dvec3 e3 = up_H*std::cos(theta_0)+m_H*std::sin(theta_0);
    glm::dvec3 e2 = up_H*std::sin(theta_0)-m_H*std::cos(theta_0);
    glm::dvec3 e1 = glm::cross(e2, e3);
    glm::dvec3 f3 = up_H*std::cos(theta_t0)+m_H*std::sin(theta_t0);
    glm::dvec3 f2 = up_H*std::sin(theta_t0)-m_H*std::cos(theta_t0);
    glm::dvec3 f1 = e1;
    out.apparent = refraction_rotation_matrix(glm::normalize(from_host(f3)), glm::normalize(from_host(e3)));
    out.pad = spread;
    out.differential = true;
    out.host_pole = host.pole;
    out.host_stretch = host.stretch;
    out.host_apparent_basis = {e1, e2, e3};
    out.host_true_basis = {f1, f2, f3};
    out.trig = {float(std::sin(theta_0)), float(std::cos(theta_0)), float(std::sin(theta_t0)), float(std::cos(theta_t0))};
    out.domain = {float(theta_max-theta_0), float(1.0/(theta_max-theta_min)), float(theta_0), 1.0f};
    double last_valid = ok_lo ? delta_min : delta_ref;
    for (uint32_t i = 0; i < ATMOSPHERE_REFRACTION_SAMPLES; ++i) {
        double t = double(i)/double(ATMOSPHERE_REFRACTION_SAMPLES-1);
        double u_q = 1.0-t;
        double theta_i = theta_max-(theta_max-theta_min)*u_q*u_q;
        bool ok_i = false;
        double delta_i = (outside && theta_i <= theta_top) ? 0.0 : forward_refraction_bend(r, r_c, theta_i, D_H, ok_i);
        if (outside && theta_i <= theta_top) ok_i = true;
        if (!ok_i) delta_i = last_valid;
        else last_valid = delta_i;
        out.bend[i >> 2][i & 3] = float(delta_i-delta_ref);
    }
    return out;
}

struct alignas(16) RefractionGPU {
    glm::vec4 host{0.0f};
    glm::vec4 pole{0.0f, 1.0f, 0.0f, 0.0f};
    glm::vec4 frame{0.0f};
    glm::vec4 mapping{0.0f};
    glm::vec4 model{0.0f};
    glm::vec4 profile{0.0f};
    glm::vec4 escape{0.0f};
};
static_assert(sizeof(RefractionGPU) == 112);

struct RefractionTables {
    std::vector<float> travel;
    std::vector<glm::vec2> rows;
    std::vector<glm::vec2> profile;
    uint32_t profile_rows = 0;
};

inline RefractionTables pack_refraction_tables(const physics::PreparedRefraction& r) {
    RefractionTables tables;
    double ground = r.ground_radius_km;
    tables.travel.assign(r.travel.begin(), r.travel.end());
    tables.rows.resize(physics::REFRACTION_IMPACT_SAMPLES);
    for (size_t i = 0; i < tables.rows.size(); ++i)
        tables.rows[i] = {float(r.impact[i]-ground), float(r.impact[i]-r.turn_radius[i])};
    size_t samples = r.fine_refractivity.size();
    tables.profile_rows = uint32_t((samples+REFRACTION_PROFILE_TEXELS_X-1)/REFRACTION_PROFILE_TEXELS_X);
    tables.profile.assign(size_t(tables.profile_rows)*REFRACTION_PROFILE_TEXELS_X, glm::vec2(0.0f));
    for (size_t i = 0; i < samples; ++i)
        tables.profile[i] = {float(r.fine_refractivity[i].y), float(r.fine_gradient[i].y)};
    return tables;
}

inline RefractionGPU refraction_parameters(const physics::PreparedRefraction& r, double camera_radius_km) {
    RefractionGPU g;
    double ground = r.ground_radius_km;
    double camera = camera_radius_km-ground;
    double top = r.top_radius_km-ground;
    bool outside = camera_radius_km > r.top_radius_km;
    g.frame = {float(camera), float(ground), float(top), outside ? 1.0f : 0.0f};
    double horizon = std::numbers::pi/2;
    if (camera > 0) horizon += std::atan2(std::sqrt(camera*(2*ground+camera)), ground);
    double scale = std::max(r.scale_km, 1e-3);
    double lowest = std::min(r.lowest_visible_chord(camera_radius_km)-ground-1.0, -1.0);
    lowest = std::max(lowest, -ground);
    double distance = std::sqrt(std::max(camera_radius_km*camera_radius_km-ground*ground, 0.0))+10.0;
    if (outside) g.mapping = {float(horizon), float(std::asinh(lowest/scale)), float(std::asinh(top/scale)), float(distance)};
    else g.mapping = {float(horizon), float(std::min(1.5*r.max_deflection()+1e-3, .5*(std::numbers::pi-horizon))), 0.0f, float(distance)};
    double span = std::asinh((r.top_radius_km-r.impact_min_km)/r.impact_scale_km);
    g.model = {float(r.impact_min_km-ground), float(r.impact_scale_km), float(span), float(r.bottom_radius_km-ground)};
    double step = (r.top_radius_km-r.bottom_radius_km)/double(r.fine_refractivity.size()-1);
    g.profile = {float(step), float(r.fine_refractivity.size()), float(r.index(camera_radius_km)-1.0), float(r.max_deflection())};
    double escape = 1e30, escape_height = 1e30;
    if (!r.monotone.y && camera_radius_km < r.top_radius_km) {
        double minimum = r.invariant(camera_radius_km), at = camera_radius_km;
        for (size_t i = 0; i < r.node_radii.size(); ++i) {
            double radius = r.node_radii[i];
            if (radius > camera_radius_km && radius < r.top_radius_km && r.node_invariant[i].y < minimum) {
                minimum = r.node_invariant[i].y;
                at = radius;
            }
        }
        if (at > camera_radius_km) {
            escape = minimum-ground;
            escape_height = at-ground;
        }
    }
    g.escape = {float(escape), float(escape_height), float(149597870.7), float(scale)};
    return g;
}

}
