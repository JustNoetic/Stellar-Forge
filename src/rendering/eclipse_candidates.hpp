#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <glm/glm.hpp>

namespace sf::rendering {

inline bool ring_could_shadow_surface(glm::vec4 receiver, glm::vec4 star,
                                     glm::vec4 center_outer, glm::vec4 pole_inner, float focal_pixels, float softness = 1.25f) {
    auto magnitude = glm::max(glm::max(glm::abs(glm::vec3(receiver)), glm::abs(glm::vec3(star))), glm::abs(glm::vec3(center_outer)));
    double padding = 32.0 * std::numeric_limits<float>::epsilon()
                   * std::max({double(magnitude.x), double(magnitude.y), double(magnitude.z), 1e-12});
    double radius = receiver.w + padding;
    glm::dvec3 origin(receiver), light(star), center(center_outer), pole(pole_inner);
    glm::dvec3 to_light = light - origin;
    double distance = glm::length(to_light);
    double denominator = glm::dot(to_light, pole);
    double minimum_denominator = std::abs(denominator) - radius - star.w;
    if (distance <= radius || minimum_denominator <= 1e-6 * (distance + radius)) return true;
    double plane_offset = glm::dot(center - origin, pole);
    if ((denominator > 0.0 ? plane_offset : -plane_offset) + radius < 0.0) return false;
    double fraction = plane_offset / denominator;
    glm::dvec3 hit = origin + to_light * fraction;
    double projected_radius = (std::abs(1.0 - fraction) * radius + std::abs(fraction) * star.w)
                            * (distance + radius + star.w) / minimum_denominator;
    double minimum_mu = minimum_denominator / std::max(distance + radius + star.w, 1e-5);
    double maximum_hit_distance = (std::abs(plane_offset) + radius) / minimum_mu;
    double stretch = 1.0 / minimum_mu;
    double stellar_blur = 2.0 * std::max(double(softness), 1.0) * star.w / std::max(distance - radius, 1e-5) * maximum_hit_distance * stretch;
    double pixel_blur = 0.75 * (glm::length(origin) + radius) / std::max(double(focal_pixels), 1.0) * stretch;
    double profile_blur = 0.75 * std::max(double(center_outer.w - pole_inner.w), 1e-9) / 4096.0;
    double reach = projected_radius + std::max({stellar_blur, pixel_blur, profile_blur}) + padding;
    double radial_distance = glm::length(hit - center);
    return radial_distance + reach >= pole_inner.w && radial_distance - reach <= center_outer.w;
}

inline bool eclipse_could_shadow(glm::vec4 receiver, glm::vec4 star, glm::vec4 caster) {
    auto magnitude = glm::max(glm::max(glm::abs(glm::vec3(receiver)), glm::abs(glm::vec3(star))), glm::abs(glm::vec3(caster)));
    double padding = 16.0 * std::numeric_limits<float>::epsilon()
                   * std::max({double(magnitude.x), double(magnitude.y), double(magnitude.z), 1e-12});
    double radius = receiver.w + padding;
    glm::dvec3 to_receiver = glm::dvec3(receiver) - glm::dvec3(star);
    glm::dvec3 to_caster = glm::dvec3(caster) - glm::dvec3(star);
    double distance = glm::length(to_receiver);
    if (distance <= radius) return true;
    double variation = glm::length(to_caster) * radius;
    double axial = glm::dot(to_caster, to_receiver);
    double maximum_distance = distance + radius;
    double maximum_distance2 = maximum_distance * maximum_distance;
    if (axial + variation < 0.0 || axial - variation > maximum_distance2) return false;
    double minimum_perpendicular = std::max(0.0, (glm::length(glm::cross(to_caster, to_receiver)) - variation) / maximum_distance);
    double maximum_projection = std::clamp(1.0 - (axial - variation) / maximum_distance2, 0.0, 1.0);
    return minimum_perpendicular <= caster.w + padding + star.w * maximum_projection;
}

struct RingShadowFootprint {
    bool bounded = false;
    bool has_sphere = false;
    glm::dvec3 sphere_center{0.0};
    double sphere_radius = 0.0;
    bool has_cone = false;
    glm::dvec3 cone_center{0.0};
    double cone_radius = 0.0;
};

inline bool eclipse_ring_shadow_footprint(glm::dvec3 center, glm::dvec3 pole_raw, double inner_r, double outer_r,
                                          glm::vec4 star, glm::vec4 caster, RingShadowFootprint& out) {
    out = {};
    double r_pad = 0.03 * std::max(outer_r, 1e-12);
    double r_min = std::max(0.0, inner_r - r_pad);
    double r_max = outer_r + r_pad;
    if (!eclipse_could_shadow(glm::vec4(glm::vec3(center), float(r_max)), star, caster)) return false;

    double pole_len = glm::length(pole_raw);
    if (pole_len <= 1e-12) return true;
    glm::dvec3 pole = pole_raw / pole_len;

    glm::dvec3 S_rel = glm::dvec3(star) - center;
    glm::dvec3 C_rel = glm::dvec3(caster) - center;
    auto magnitude = glm::max(glm::max(glm::abs(S_rel), glm::abs(C_rel)), glm::dvec3(r_max));
    double padding = 32.0 * std::numeric_limits<double>::epsilon()
                   * std::max({magnitude.x, magnitude.y, magnitude.z, 1e-12});

    double R_s = double(star.w);
    double R_c = double(caster.w) + padding;

    double star_h = glm::dot(S_rel, pole);
    double caster_h = glm::dot(C_rel, pole);
    if (std::abs(star_h) > R_s + padding) {
        double sign_s = star_h >= 0.0 ? 1.0 : -1.0;
        double c_h = caster_h * sign_s;
        if (c_h + R_c < 0.0) return false;
        if (c_h - R_c > std::abs(star_h) + R_s + padding) return false;
    }

    bool hits_annulus = false;
    if (std::abs(caster_h) <= R_c) {
        glm::dvec3 c_proj = C_rel - pole * caster_h;
        double d_proj = glm::length(c_proj);
        out.has_sphere = true;
        out.sphere_center = c_proj;
        out.sphere_radius = R_c;
        if (d_proj + R_c >= r_min && d_proj - R_c <= r_max) hits_annulus = true;
    }

    glm::dvec3 V = C_rel - S_rel;
    double d_sc = glm::length(V);
    if (d_sc <= R_c + R_s + 1e-12) return true;
    glm::dvec3 A = V / d_sc;

    double sin_a = std::min((R_c + R_s) / d_sc, 0.999999);
    double cos_a = std::sqrt(std::max(0.0, 1.0 - sin_a * sin_a));
    double A_n = glm::dot(A, pole);
    double mu = std::abs(A_n);
    glm::dvec3 A_horiz = A - pole * A_n;
    double sin_plane = glm::length(A_horiz);
    double sin_minus = mu * cos_a - sin_plane * sin_a;
    if (sin_minus <= 1e-5) return true;

    double t_apex = d_sc * R_s / std::max(R_s + R_c, 1e-12);
    glm::dvec3 V_apex = S_rel + A * t_apex;
    double h_apex = glm::dot(V_apex, pole);
    if (h_apex * A_n >= 0.0) {
        out.bounded = out.has_sphere;
        return hits_annulus;
    }

    double H_axis = -h_apex / A_n;
    glm::dvec3 P_axis = V_apex + A * H_axis;
    double sin_plus = mu * cos_a + sin_plane * sin_a;
    double d_far = H_axis * sin_a / sin_minus;
    double d_near = sin_plus > 1e-6 ? H_axis * sin_a / sin_plus : d_far;
    double radius_circle = 0.5 * (d_far + d_near) + padding;
    glm::dvec3 p_center = P_axis;
    if (sin_plane > 1e-12) p_center += (A_horiz / sin_plane) * (0.5 * (d_far - d_near));
    double dist_c = glm::length(p_center);

    out.has_cone = true;
    out.cone_center = p_center;
    out.cone_radius = radius_circle;
    out.bounded = true;
    if (dist_c + radius_circle >= r_min && dist_c - radius_circle <= r_max) hits_annulus = true;
    return hits_annulus;
}

inline bool eclipse_could_shadow_ring(glm::dvec3 center, glm::dvec3 pole_raw, double inner_r, double outer_r,
                                      glm::vec4 star, glm::vec4 caster) {
    RingShadowFootprint fp;
    return eclipse_ring_shadow_footprint(center, pole_raw, inner_r, outer_r, star, caster, fp);
}

}
