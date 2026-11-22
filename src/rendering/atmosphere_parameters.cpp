#include "rendering/atmosphere_internal.hpp"
#include "rendering/eclipse_candidates.hpp"
#include "rendering/ring_renderer.hpp"
#include "rendering/refraction_tables.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace sf::rendering {

static_assert(sizeof(AtmosphereParametersGPU) == 15328);
static_assert(sizeof(AtmosphereParametersGPU) <= 16384);
static_assert(sizeof(AtmosphereShadowHeaderGPU) == 592);
static_assert(sizeof(AtmosphereShadowCasterGPU) == 48);

namespace {

// tan_x/tan_y are the real projection tangents (the shaders build view rays from the
// unpadded projection), so NDC coordinates map straight to pixels. Differential
// refraction can move a pixel's true ray by up to `pad` radians, so the culling planes
// use padded tangents and `limit` is the NDC extent those planes reach. Callers clamp
// to +-limit instead of +-1 and widen the final rect by `margin`.
struct ShadowFrustum {
    glm::dvec3 right{}, up{}, forward{};
    double tan_x = 1.0, tan_y = 1.0;
    glm::dvec2 limit{1.0}, margin{0.0};
    std::array<glm::dvec3, 5> planes{};

    glm::dvec3 to_camera(glm::dvec3 v) const noexcept {
        return {glm::dot(v, right), glm::dot(v, up), glm::dot(v, forward)};
    }
};

ShadowFrustum make_shadow_frustum(const Camera& camera, const glm::dmat3& apparent, VkExtent2D extent, double pad) {
    glm::dmat3 to_true = glm::transpose(apparent);
    double ty = std::tan(camera.fov_deg * DEG_TO_RAD * 0.5);
    double tx = ty * double(extent.width) / std::max(1u, extent.height);
    double plane_x = tx, plane_y = ty;
    if (pad > 0.0) {
        plane_x = std::tan(std::min(std::atan(tx) + pad, 1.5));
        plane_y = std::tan(std::min(std::atan(ty) + pad, 1.5));
    }
    ShadowFrustum f{
        glm::normalize(to_true * camera.right_dir()),
        glm::normalize(to_true * camera.up_dir()),
        glm::normalize(to_true * camera.forward_dir()),
        tx, ty, glm::dvec2(plane_x / tx, plane_y / ty), {}, {}
    };
    // tan is convex, so the angular pad spans the most NDC at the screen edge.
    f.margin = f.limit - 1.0;
    f.planes = {
        glm::normalize(glm::dvec3(1.0, 0.0, plane_x)),
        glm::normalize(glm::dvec3(-1.0, 0.0, plane_x)),
        glm::normalize(glm::dvec3(0.0, 1.0, plane_y)),
        glm::normalize(glm::dvec3(0.0, -1.0, plane_y)),
        glm::dvec3(0.0, 0.0, 1.0)
    };
    return f;
}

bool project_sphere_ndc(glm::dvec3 world_center, double radius, const ShadowFrustum& f,
                        glm::dvec2& out_min, glm::dvec2& out_max) {
    glm::dvec3 c = f.to_camera(world_center);
    for (const auto& n : f.planes)
        if (glm::dot(n, c) < -radius) return false;
    auto axis_bounds = [](double u, double z, double r, double tan_half, double limit) {
        double d2 = u * u + z * z;
        if (d2 <= r * r) return std::pair{-limit, limit};
        double t = std::sqrt(d2 - r * r), inv = 1.0 / d2;
        double lx = (u * t - z * r) * inv, lz = (z * t + u * r) * inv;
        double rx = (u * t + z * r) * inv, rz = (z * t - u * r) * inv;
        double lo = lz > 1e-9 ? std::clamp(lx / (lz * tan_half), -limit, limit) : -limit;
        double hi = rz > 1e-9 ? std::clamp(rx / (rz * tan_half), -limit, limit) : limit;
        return std::pair{lo, hi};
    };
    auto [xmin, xmax] = axis_bounds(c.x, c.z, radius, f.tan_x, f.limit.x);
    auto [ymin, ymax] = axis_bounds(c.y, c.z, radius, f.tan_y, f.limit.y);
    out_min = {xmin, -ymax};
    out_max = {xmax, -ymin};
    return out_min.x <= out_max.x && out_min.y <= out_max.y;
}

bool project_cylinder_ndc(glm::dvec3 a_world, glm::dvec3 b_world, double radius, const ShadowFrustum& f,
                          glm::dvec2& out_min, glm::dvec2& out_max) {
    glm::dvec3 a = f.to_camera(a_world), b = f.to_camera(b_world), ab = b - a;
    double ab2 = glm::dot(ab, ab);
    double u = ab2 > 1e-12 ? std::clamp(-glm::dot(a, ab) / ab2, 0.0, 1.0) : 0.0;
    glm::dvec3 closest = a + u * ab;
    if (glm::dot(closest, closest) <= radius * radius) {
        out_min = -f.limit;
        out_max = f.limit;
        return true;
    }
    double t0 = 0.0, t1 = 1.0;
    for (const auto& n : f.planes) {
        double da = glm::dot(n, a) + radius, db = glm::dot(n, b) + radius;
        if (da < 0.0 && db < 0.0) return false;
        double slope = db - da;
        if (da < 0.0) t0 = std::max(t0, -da / slope);
        else if (db < 0.0) t1 = std::min(t1, -da / slope);
        if (t0 > t1) return false;
    }
    glm::dvec3 axis = ab2 > 1e-12 ? ab / std::sqrt(ab2) : glm::dvec3(0.0, 0.0, 1.0);
    glm::dvec3 ref = std::abs(axis.z) < 0.9 ? glm::dvec3(0.0, 0.0, 1.0) : glm::dvec3(1.0, 0.0, 0.0);
    glm::dvec3 u_ax = glm::normalize(glm::cross(ref, axis));
    glm::dvec3 v_ax = glm::cross(axis, u_ax);
    constexpr int sides = 12;
    double r_prism = radius / std::cos(PI / double(sides));
    std::array<glm::dvec3, sides> ring_a{}, ring_b{};
    for (int i = 0; i < sides; ++i) {
        double angle = 2.0 * PI * double(i) / double(sides);
        glm::dvec3 offset = (u_ax * std::cos(angle) + v_ax * std::sin(angle)) * r_prism;
        ring_a[i] = a + offset;
        ring_b[i] = b + offset;
    }
    constexpr double z_near = 1e-4;
    glm::dvec2 ndc_min(2.0), ndc_max(-2.0);
    auto push_point = [&](glm::dvec3 p) {
        glm::dvec2 ndc(p.x / (p.z * f.tan_x), -p.y / (p.z * f.tan_y));
        ndc_min = glm::min(ndc_min, ndc);
        ndc_max = glm::max(ndc_max, ndc);
    };
    auto slice_edge = [&](glm::dvec3 p0, glm::dvec3 p1) {
        if ((p0.z >= z_near) != (p1.z >= z_near))
            push_point(glm::mix(p0, p1, (z_near - p0.z) / (p1.z - p0.z)));
    };
    for (int i = 0; i < sides; ++i) {
        int next = (i + 1) % sides;
        if (ring_a[i].z >= z_near) push_point(ring_a[i]);
        if (ring_b[i].z >= z_near) push_point(ring_b[i]);
        slice_edge(ring_a[i], ring_a[next]);
        slice_edge(ring_b[i], ring_b[next]);
        slice_edge(ring_a[i], ring_b[i]);
    }
    out_min = glm::max(ndc_min, -f.limit);
    out_max = glm::min(ndc_max, f.limit);
    return out_min.x <= out_max.x && out_min.y <= out_max.y;
}

bool project_ring_ndc(glm::dvec3 atm_center, double outer, double ground, double reach,
                      glm::dvec3 ring_center, glm::dvec3 pole, double inner, double radius, float support,
                      glm::dvec3 star, double r_star, const ShadowFrustum& f, double refraction_pad,
                      glm::dvec2 atm_min, glm::dvec2 atm_max, glm::dvec2& out_min, glm::dvec2& out_max) {
    glm::dvec3 to_star_rc = star - ring_center;
    double star_plane = glm::dot(to_star_rc, pole);
    double star_side = star_plane >= 0.0 ? 1.0 : -1.0;
    bool has_halfspace = std::abs(star_plane) > r_star &&
                         glm::dot(star - atm_center, pole) * star_side > r_star + outer;
    glm::dvec3 n_shadow = -star_side * pole;
    double s_dist = glm::length(to_star_rc);
    glm::dvec3 l_hat = s_dist > 1e-6 ? to_star_rc / s_dist : pole;
    double denom = glm::dot(l_hat, pole);
    bool has_annulus = has_halfspace && support >= 0.0f &&
                       std::abs(denom) > (r_star + outer) / std::max(s_dist, 1.0) + 1e-4;
    double r_outer_max = double(support);
    double r_inner_min = 0.0;
    glm::dvec3 ring_u(1.0, 0.0, 0.0), ring_v(0.0, 1.0, 0.0);
    if (has_annulus) {
        double pad_ring = std::max(double(support) - radius, 0.0);
        double max_h = outer + glm::length(atm_center - ring_center);
        double max_d_hit = max_h / std::abs(denom);
        double parallax = s_dist > outer ? max_d_hit * outer / (s_dist - outer) : outer;
        r_outer_max = double(support) + parallax;
        r_inner_min = inner - pad_ring - parallax;
        glm::dvec3 ref = std::abs(pole.z) < 0.9 ? glm::dvec3(0.0, 0.0, 1.0) : glm::dvec3(1.0, 0.0, 0.0);
        ring_u = glm::normalize(glm::cross(ref, pole));
        ring_v = glm::cross(pole, ring_u);
    }

    double atm_dist = glm::length(atm_center);
    glm::dvec3 atm_hat = atm_dist > 1e-9 ? atm_center / atm_dist : f.forward;
    double sph_ang = atm_dist > outer ? std::asin(std::clamp(outer / atm_dist, 0.0, 1.0)) : PI;
    double ground_r = ground * 0.999;
    double ground_ang = (ground_r > 0.0 && ground_r < outer && atm_dist > ground_r)
        ? std::asin(std::clamp(ground_r / atm_dist, 0.0, 1.0)) : 0.0;
    double reach_cap = std::min(reach, atm_dist + outer);
    double global_d_lo = std::max(1e-4, atm_dist - outer);

    constexpr int grid = 8;
    constexpr int edges[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0},
        {4, 5}, {5, 7}, {7, 6}, {6, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };
    glm::dvec2 res_min(2.0), res_max(-2.0);
    for (int gy = 0; gy < grid; ++gy) {
        double ny0 = glm::mix(atm_min.y, atm_max.y, double(gy) / double(grid));
        double ny1 = glm::mix(atm_min.y, atm_max.y, double(gy + 1) / double(grid));
        for (int gx = 0; gx < grid; ++gx) {
            double nx0 = glm::mix(atm_min.x, atm_max.x, double(gx) / double(grid));
            double nx1 = glm::mix(atm_min.x, atm_max.x, double(gx + 1) / double(grid));
            auto ray_dir = [&](double nx, double ny) {
                return glm::normalize(f.forward + f.right * (nx * f.tan_x) - f.up * (ny * f.tan_y));
            };
            std::array<glm::dvec3, 4> corners = {
                ray_dir(nx0, ny0), ray_dir(nx1, ny0), ray_dir(nx0, ny1), ray_dir(nx1, ny1)
            };
            glm::dvec3 mid = glm::normalize(corners[0] + corners[1] + corners[2] + corners[3]);
            double cos_cone = 1.0;
            double cos_fwd_min = 1.0;
            for (const auto& c : corners) {
                cos_cone = std::min(cos_cone, glm::dot(c, mid));
                cos_fwd_min = std::min(cos_fwd_min, glm::dot(c, f.forward));
            }
            double cx_near = std::clamp(0.0, nx0, nx1) * f.tan_x;
            double cy_near = std::clamp(0.0, ny0, ny1) * f.tan_y;
            double cos_fwd_max = 1.0 / std::sqrt(1.0 + cx_near * cx_near + cy_near * cy_near);
            double cone_ang = std::acos(std::clamp(cos_cone, -1.0, 1.0));
            double eff_cone = cone_ang + refraction_pad;
            double theta = std::acos(std::clamp(glm::dot(atm_hat, mid), -1.0, 1.0));
            double cell_d_lo = global_d_lo;
            double d_hi = reach_cap;
            if (atm_dist > outer) {
                if (theta > eff_cone + sph_ang) continue;
                double phi_min = std::max(0.0, theta - eff_cone);
                double sin_lo = std::sin(phi_min);
                double d_ent = atm_dist * std::cos(phi_min) -
                    std::sqrt(std::max(outer * outer - atm_dist * atm_dist * sin_lo * sin_lo, 0.0));
                cell_d_lo = std::max(cell_d_lo, d_ent);
                if (ground_ang > 0.0 && theta + eff_cone < ground_ang) {
                    double phi_hi = theta + eff_cone;
                    double sin_hi = std::sin(phi_hi);
                    double d_gnd = atm_dist * std::cos(phi_hi) -
                        std::sqrt(std::max(ground_r * ground_r - atm_dist * atm_dist * sin_hi * sin_hi, 0.0));
                    d_hi = std::min(d_hi, d_gnd + 1.0);
                } else {
                    double phi_ex = std::max(phi_min, ground_ang);
                    if (phi_ex < sph_ang) {
                        double sin_ex = std::sin(phi_ex);
                        double d_ex = atm_dist * std::cos(phi_ex) +
                            std::sqrt(std::max(outer * outer - atm_dist * atm_dist * sin_ex * sin_ex, 0.0));
                        d_hi = std::min(d_hi, d_ex + 1.0);
                    }
                }
            } else {
                double phi_min = std::max(0.0, theta - eff_cone);
                if (ground_ang > 0.0 && theta + eff_cone < ground_ang) {
                    double phi_hi = theta + eff_cone;
                    double sin_hi = std::sin(phi_hi);
                    double d_gnd = atm_dist * std::cos(phi_hi) -
                        std::sqrt(std::max(ground_r * ground_r - atm_dist * atm_dist * sin_hi * sin_hi, 0.0));
                    d_hi = std::min(d_hi, d_gnd + 1.0);
                } else {
                    double phi_ex = std::max(phi_min, ground_ang);
                    double sin_ex = std::sin(phi_ex);
                    double d_ex = atm_dist * std::cos(phi_ex) +
                        std::sqrt(std::max(outer * outer - atm_dist * atm_dist * sin_ex * sin_ex, 0.0));
                    d_hi = std::min(d_hi, d_ex + 1.0);
                }
            }
            if (d_hi <= cell_d_lo) continue;
            double cz_lo = std::max(1e-4, cell_d_lo * std::max(cos_fwd_min, 0.0));
            double cz_hi = std::min(d_hi * cos_fwd_max, reach_cap);
            if (cz_hi <= cz_lo) continue;

            std::array<glm::dvec3, 8> box{};
            for (int k = 0; k < 2; ++k) {
                double cz = k == 0 ? cz_lo : cz_hi;
                double x0 = nx0 * f.tan_x * cz, x1 = nx1 * f.tan_x * cz;
                double y0 = -ny1 * f.tan_y * cz, y1 = -ny0 * f.tan_y * cz;
                box[k * 4 + 0] = f.right * x0 + f.up * y0 + f.forward * cz;
                box[k * 4 + 1] = f.right * x1 + f.up * y0 + f.forward * cz;
                box[k * 4 + 2] = f.right * x0 + f.up * y1 + f.forward * cz;
                box[k * 4 + 3] = f.right * x1 + f.up * y1 + f.forward * cz;
            }
            std::array<glm::dvec3, 20> clipped{};
            size_t count = 0;
            if (has_halfspace) {
                std::array<double, 8> sd{};
                for (int i = 0; i < 8; ++i) {
                    sd[i] = glm::dot(box[i] - ring_center, n_shadow) + 0.1;
                    if (sd[i] >= 0.0) clipped[count++] = box[i];
                }
                for (const auto& e : edges) {
                    if ((sd[e[0]] >= 0.0) != (sd[e[1]] >= 0.0))
                        clipped[count++] = glm::mix(box[e[0]], box[e[1]], sd[e[0]] / (sd[e[0]] - sd[e[1]]));
                }
                if (count == 0) continue;
            } else {
                for (int i = 0; i < 8; ++i) clipped[count++] = box[i];
            }

            if (has_annulus) {
                bool all_inside_inner = r_inner_min > 0.0;
                glm::dvec2 q_min(1e30), q_max(-1e30);
                glm::dvec3 q_sum(0.0);
                std::array<glm::dvec3, 20> q_pts{};
                for (size_t i = 0; i < count; ++i) {
                    glm::dvec3 rel = clipped[i] - ring_center;
                    glm::dvec3 q = rel - l_hat * (glm::dot(rel, pole) / denom);
                    q_pts[i] = q;
                    if (all_inside_inner && glm::dot(q, q) >= r_inner_min * r_inner_min)
                        all_inside_inner = false;
                    glm::dvec2 q2(glm::dot(q, ring_u), glm::dot(q, ring_v));
                    q_min = glm::min(q_min, q2);
                    q_max = glm::max(q_max, q2);
                    q_sum += q;
                }
                if (all_inside_inner) continue;
                if (q_max.x < -r_outer_max || q_min.x > r_outer_max ||
                    q_max.y < -r_outer_max || q_min.y > r_outer_max) continue;
                glm::dvec3 q_avg = q_sum / double(count);
                double r_q2 = 0.0;
                for (size_t i = 0; i < count; ++i) {
                    glm::dvec3 d = q_pts[i] - q_avg;
                    r_q2 = std::max(r_q2, glm::dot(d, d));
                }
                if (glm::length(q_avg) > r_outer_max + std::sqrt(r_q2)) continue;
            }

            for (size_t i = 0; i < count; ++i) {
                glm::dvec3 c = f.to_camera(clipped[i]);
                if (c.z <= 1e-4) {
                    res_min = glm::min(res_min, glm::dvec2(nx0, ny0));
                    res_max = glm::max(res_max, glm::dvec2(nx1, ny1));
                } else {
                    glm::dvec2 p(c.x / (c.z * f.tan_x), -c.y / (c.z * f.tan_y));
                    res_min = glm::min(res_min, p);
                    res_max = glm::max(res_max, p);
                }
            }
        }
    }
    out_min = glm::max(res_min, atm_min);
    out_max = glm::min(res_max, atm_max);
    return out_min.x <= out_max.x && out_min.y <= out_max.y;
}

} // namespace

bool stellar_source_resolved(const AtmosphereParametersGPU& p) {
    double bottom = p.radii_heights.x, top = p.radii_heights.y;
    double step = std::sqrt((top-bottom)*(top+bottom))/63.0;
    auto resolved = [&](glm::vec3 extinction, double peak, double width) {
        double optical_depth = std::max({extinction.x, extinction.y, extinction.z})*width*std::sqrt(2.0*PI);
        if (optical_depth <= 0.005 || width <= 0.0) return true;
        double radius = std::clamp(double(p.reference_shape.x)+peak, bottom, top);
        double rho = std::sqrt((radius-bottom)*(radius+bottom));
        double increment = step*(2.0*rho+step);
        double spacing = increment/(std::sqrt(radius*radius+increment)+radius);
        return width >= 2.0*spacing;
    };
    if (p.profile_domain.w > 0.5f && !resolved(glm::vec3(p.ozone_absorption), p.ozone_params.x, p.ozone_params.y)) return false;
    for (int i = 0; i < int(p.aerosol_count.x); ++i)
        if (p.aerosol_shape[i].z > 0.0f && !resolved(glm::vec3(p.aerosol_extinction[i]), p.aerosol_shape[i].y, p.aerosol_shape[i].z)) return false;
    return true;
}

bool prepare_shadows(rhi::VkContext& ctx, rhi::Buffer& buffer, const ephemeris::SystemSnapshot& system,
                     uint32_t receiver, glm::dvec3 eye, float outer, float ground, float focal,
                     const Camera& camera, const glm::dmat3& apparent, VkExtent2D extent, double refraction_pad,
                     const RenderSettings& settings, std::span<const RingHostGPU> rings,
                     AtmosphereShadowHeaderGPU& previous, std::vector<AtmosphereShadowCasterGPU>& previous_casters,
                     uint32_t sequence, uint32_t& changes, const RingPenumbraSlots* penumbra_slots) {
    AtmosphereShadowHeaderGPU header;
    const auto origin = system.body(receiver).render.position.value;
    header.eye_focal = {glm::vec3((eye-origin)*AU_TO_KM), focal};
    header.flags = {settings.atmosphere_shadows_enabled ? 1u : 0u,
                    (settings.aerial_perspective_enabled ? 4u : 0u) |
                    (!settings.atmosphere_shadow_temporal ? ((settings.atmosphere_shadow_analytic ? 16u : 0u) | (settings.atmosphere_shadow_closed_form ? 32u : 0u)) : 0u) |
                    (uint32_t(std::bit_width(std::clamp(settings.atmosphere_shadow_vrs_scale, 2u, 16u))-1) << 8u),
                    settings.eclipse_reference_smoothstep ? 1u : 0u, settings.atmosphere_shadow_reference ? 1u : 0u};
    uint32_t effective_sequence = !settings.atmosphere_shadow_reference &&
        (settings.atmosphere_shadow_temporal || settings.atmosphere_shadow_animate_jitter) ? sequence : 0u;
    header.temporal = {effective_sequence, settings.atmosphere_shadow_noise_type,
                       settings.atmosphere_shadow_samples, 1u};
    std::vector<AtmosphereShadowCasterGPU> casters;
    uint32_t star_index = 0;
    double eye_radius = glm::length(eye-origin)*AU_TO_KM;
    double reach = ground > 0.0f && ground < outer
        ? std::sqrt(std::max(eye_radius*eye_radius-double(ground)*ground, 0.0))+std::sqrt(double(outer)*outer-double(ground)*ground)
        : std::numeric_limits<double>::infinity();
    glm::vec4 visible = reach < outer ? glm::vec4(0.0f, 0.0f, 0.0f, float(reach*KM_TO_AU))
                                      : glm::vec4(glm::vec3(origin-eye), outer*float(KM_TO_AU));
    bool view_cull = !settings.atmosphere_shadow_reference && !settings.atmosphere_shadow_temporal &&
                     extent.width > 0 && extent.height > 0;
    ShadowFrustum frustum = view_cull ? make_shadow_frustum(camera, apparent, extent, refraction_pad) : ShadowFrustum{};
    glm::dvec3 atm_center = (origin - eye) * AU_TO_KM;
    glm::dvec2 atm_min(-1.0), atm_max(1.0);
    bool atm_visible = !view_cull || project_sphere_ndc(atm_center, double(outer), frustum, atm_min, atm_max);
    if (view_cull && atm_visible && reach < outer) {
        glm::dvec2 reach_min, reach_max;
        if (project_sphere_ndc(glm::dvec3(0.0), reach, frustum, reach_min, reach_max)) {
            atm_min = glm::max(atm_min, reach_min);
            atm_max = glm::min(atm_max, reach_max);
            atm_visible = atm_min.x <= atm_max.x && atm_min.y <= atm_max.y;
        } else {
            atm_visible = false;
        }
    }
    glm::dvec2 shadow_ndc_min = view_cull ? glm::dvec2(2.0) : glm::dvec2(-1.0);
    glm::dvec2 shadow_ndc_max = view_cull ? glm::dvec2(-2.0) : glm::dvec2(1.0);
    // Scene star index (every star body in system order, as the scene UBO lists them), for the ring
    // penumbra slots; star_index skips a star at the receiver's own centre.
    uint32_t scene_star = 0;
    for (uint32_t s = 0; s < system.size() && star_index < header.stars.size(); ++s) {
        const auto star = system.body(s);
        if (!star.definition.is_star) continue;
        uint32_t this_scene_star = scene_star++;
        if (glm::length(star.render.position.value-origin) < 1e-12) continue;
        auto& light = header.stars[star_index];
        light = {glm::vec3((star.render.position.value-origin)*AU_TO_KM), float(star.opaque_radius_au()*AU_TO_KM)};
        auto& range = header.ranges[star_index++];
        if (!settings.atmosphere_shadows_enabled || !system.illuminates(s, receiver) || !atm_visible) continue;
        range.x = uint32_t(casters.size());
        for (uint32_t j = 0; j < system.size(); ++j) {
            const auto caster = system.body(j);
            if (j == receiver || caster.definition.is_star || caster.opaque_radius_au() <= 0.0) continue;
            glm::vec4 position(glm::vec3((caster.render.position.value-origin)*AU_TO_KM),
                        float(caster.opaque_radius_au()*AU_TO_KM));
            if (!eclipse_could_shadow({0, 0, 0, outer}, light, position)) continue;
            if (view_cull) {
                glm::dvec2 c_min = atm_min, c_max = atm_max;
                glm::dvec3 star_eye = atm_center + glm::dvec3(light);
                glm::dvec3 caster_eye = atm_center + glm::dvec3(position);
                glm::dvec3 cs = caster_eye - star_eye;
                double d_cs = glm::length(cs);
                if (d_cs > 1e-6) {
                    glm::dvec3 axis = cs / d_cs;
                    // Outer penumbra cone radius R_c + (R_c + R_s) x / D_sc, at the farthest
                    // receiver point behind the caster (x <= |caster| + outer).
                    double behind = glm::length(glm::dvec3(position)) + double(outer);
                    double r_cyl = (double(position.w) + (double(position.w) + double(light.w)) * behind / d_cs) * 1.0005 + 0.01;
                    if (r_cyl < double(outer)) {
                        auto sphere_span = [&](glm::dvec3 center_s, double radius_s, double& t_lo, double& t_hi) {
                            glm::dvec3 delta = center_s - caster_eye;
                            double proj = glm::dot(delta, axis);
                            double d2 = glm::dot(delta, delta) - proj * proj;
                            double lim = radius_s + r_cyl;
                            if (d2 >= lim * lim) return false;
                            double half = std::sqrt(lim * lim - d2);
                            t_lo = proj - half;
                            t_hi = proj + half;
                            return true;
                        };
                        double t0 = 0.0, t1 = 0.0;
                        if (!sphere_span(atm_center, double(outer), t0, t1)) continue;
                        t0 = std::max(t0, -r_cyl);
                        if (reach < outer) {
                            double e0 = 0.0, e1 = 0.0;
                            if (!sphere_span(glm::dvec3(0.0), reach, e0, e1)) continue;
                            t0 = std::max(t0, e0);
                            t1 = std::min(t1, e1);
                        }
                        if (t1 <= t0) continue;
                        if (!project_cylinder_ndc(caster_eye + t0 * axis, caster_eye + t1 * axis, r_cyl, frustum, c_min, c_max))
                            continue;
                        c_min = glm::max(c_min, atm_min);
                        c_max = glm::min(c_max, atm_max);
                        if (c_min.x > c_max.x || c_min.y > c_max.y) continue;
                    }
                }
                shadow_ndc_min = glm::min(shadow_ndc_min, c_min);
                shadow_ndc_max = glm::max(shadow_ndc_max, c_max);
            }
            casters.push_back({position, {glm::vec3(caster.render.pole),
                position.w*float(std::clamp(1.0-caster.definition.oblateness, 0.05, 1.0))}, {}});
        }
        range.y = uint32_t(casters.size())-range.x;
        range.z = uint32_t(casters.size());
        if (settings.show_rings && settings.ring_shadows_enabled) {
            for (const auto& ring : rings) {
                uint32_t host = ring.counts_indices.x;
                if (host >= system.size() || ring.pole_max_r.w <= ring.center_min_r.w) continue;
                glm::vec3 center((system.body(host).render.position.value-origin)*AU_TO_KM);
                glm::vec3 pole(ring.pole_max_r);
                float inner = float(ring.center_min_r.w*AU_TO_KM), radius = float(ring.pole_max_r.w*AU_TO_KM);
                glm::vec4 light_eye(glm::vec3(star.render.position.value-eye), light.w*float(KM_TO_AU));
                if (!ring_could_shadow_surface(visible, light_eye,
                    {glm::vec3(ring.center_min_r), ring.pole_max_r.w}, {pole, ring.center_min_r.w}, focal, settings.ring_shadow_softness)) continue;
                double light_distance = glm::length(glm::dvec3(light));
                double denominator = std::abs(glm::dot(glm::dvec3(light), glm::dvec3(pole)))-outer-light.w;
                float support = -1.0f;
                if (light_distance > outer && denominator > 1e-6*(light_distance+outer)) {
                    double minimum_mu = denominator/(light_distance+outer+light.w);
                    double hit_distance = (std::abs(glm::dot(glm::dvec3(center), glm::dvec3(pole)))+outer)/minimum_mu;
                    double stretch = 1.0/minimum_mu;
                    double stellar_blur = 2.0*std::max(settings.ring_shadow_softness, 1.0f)*light.w/(light_distance-outer)*hit_distance*stretch;
                    double pixel_blur = 0.75*(glm::length(glm::dvec3(header.eye_focal))+outer)/focal*stretch;
                    double profile_blur = 0.75*(radius-inner)/4096.0;
                    support = float(radius+std::max({stellar_blur, pixel_blur, profile_blur})+0.01);
                }
                if (view_cull) {
                    glm::dvec2 c_min, c_max;
                    if (!project_ring_ndc(atm_center, double(outer), double(ground), reach, atm_center + glm::dvec3(center), glm::dvec3(pole),
                                          double(inner), double(radius), support, atm_center + glm::dvec3(light), double(light.w),
                                          frustum, refraction_pad, atm_min, atm_max, c_min, c_max))
                        continue;
                    shadow_ndc_min = glm::min(shadow_ndc_min, c_min);
                    shadow_ndc_max = glm::max(shadow_ndc_max, c_max);
                }
                // Penumbra table slot the ring renderer assigned to this ring and scene star (keyed by the
                // star's incidence at the ring centre). support.w = slot + 1 + the slot's mu coordinate.
                float penumbra = 0.0f;
                if (penumbra_slots && this_scene_star < RING_PENUMBRA_STARS && !settings.atmosphere_shadow_reference) {
                    uint32_t slot = (ring.counts_indices.w >> (4u*this_scene_star)) & 15u;
                    if (slot > 0u && slot <= RING_PENUMBRA_SLOTS && penumbra_slots->slots[slot-1u].key.x == float(ring.counts_indices.y))
                        penumbra = float(slot)+penumbra_slots->slots[slot-1u].key.y;
                }
                casters.push_back({{center, radius}, {pole, inner}, {support, float(ring.counts_indices.y), settings.ring_shadow_softness, penumbra}});
            }
        }
        range.w = uint32_t(casters.size())-range.z;
    }
    if (casters.empty() || shadow_ndc_min.x > shadow_ndc_max.x || shadow_ndc_min.y > shadow_ndc_max.y) {
        casters.clear();
        header.flags.x = 0;
    }
    if (!casters.empty() && extent.width > 0 && extent.height > 0) {
        int w = int(extent.width), h = int(extent.height);
        header.shadow_bounds = {0, 0, w, h};
        if (view_cull) {
            shadow_ndc_min -= frustum.margin;
            shadow_ndc_max += frustum.margin;
            int px0 =std::clamp(int(std::floor((shadow_ndc_min.x * 0.5 + 0.5) * double(w))) - 1, 0, w - 1);
            int py0 = std::clamp(int(std::floor((shadow_ndc_min.y * 0.5 + 0.5) * double(h))) - 1, 0, h - 1);
            int px1 = std::clamp(int(std::ceil((shadow_ndc_max.x * 0.5 + 0.5) * double(w))) + 1, 1, w);
            int py1 = std::clamp(int(std::ceil((shadow_ndc_max.y * 0.5 + 0.5) * double(h))) + 1, 1, h);
            header.shadow_bounds = {
                (px0 / 8) * 8,
                (py0 / 8) * 8,
                std::min(((px1 + 7) / 8) * 8, w),
                std::min(((py1 + 7) / 8) * 8, h)
            };
        }
        // Fixed ray budget: the finest rate whose integrated rays fit (native = log2 rate 0), up to the VRS rate.
        const auto& bounds = header.shadow_bounds;
        double area = double(bounds.z - bounds.x) * double(bounds.w - bounds.y);
        double budget = double(settings.atmosphere_shadow_vrs_budget) * 0.01 * double(w) * double(h);
        int max_rate = std::bit_width(std::clamp(settings.atmosphere_shadow_vrs_scale, 2u, 16u)) - 1;
        int rate = 0;
        while (rate < max_rate && area > budget * double(1 << (2 * rate))) ++rate;
        header.flags.y = (header.flags.y & ~(15u << 8u)) | (uint32_t(rate) << 8u);
        int scale = 1 << rate;
        header.shadow_coarse_bounds = {
            bounds.x / scale,
            bounds.y / scale,
            std::min((bounds.z - 1) / scale + 2, (w + scale - 1) / scale),
            std::min((bounds.w - 1) / scale + 2, (h + scale - 1) / scale)
        };
    }
    VkDeviceSize bytes = sizeof(header)+casters.size()*sizeof(AtmosphereShadowCasterGPU);
    if (buffer.size < bytes) {
        VkDeviceSize capacity = std::max(bytes, buffer.size*2);
        ctx.destroy_buffer(buffer);
        buffer = ctx.create_host_buffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    std::memcpy(buffer.mapped, &header, sizeof(header));
    if (!casters.empty()) std::memcpy(static_cast<std::byte*>(buffer.mapped)+sizeof(header), casters.data(), bytes-sizeof(header));
    ctx.flush_buffer(buffer, 0, bytes);
    changes = 0;
    if (glm::vec3(previous.eye_focal) != glm::vec3(header.eye_focal)) changes |= ShadowHistoryCamera;
    if (previous.eye_focal.w != header.eye_focal.w) changes |= ShadowHistoryProjection;
    if (previous.flags != header.flags || glm::uvec3(previous.temporal.y, previous.temporal.z, previous.temporal.w) !=
        glm::uvec3(header.temporal.y, header.temporal.z, header.temporal.w)) changes |= ShadowHistorySampling;
    if (previous.stars != header.stars) changes |= ShadowHistoryLighting;
    if (previous.ranges != header.ranges || previous_casters.size() != casters.size() || (!casters.empty() &&
        std::memcmp(previous_casters.data(), casters.data(), casters.size()*sizeof(AtmosphereShadowCasterGPU)) != 0))
        changes |= ShadowHistoryCasters;
    auto previous_key = previous;
    previous_key.temporal.x = header.temporal.x;
    if (changes == 0 && std::memcmp(&previous_key, &header, sizeof(header)) != 0) changes |= ShadowHistorySampling;
    previous = header;
    previous_casters = std::move(casters);
    return !previous_casters.empty();
}

AtmosphereParametersGPU make_atmosphere_parameters(const ephemeris::BodyDefinition& body,
    const std::shared_ptr<const physics::PreparedAtmosphere>& prepared, glm::dvec3 pole, double minimum) {
    const auto& a = body.atmosphere;
    float height = prepared ? prepared->height_km : a.height_km;
    double polar_ratio = std::clamp(1.0-body.oblateness, 0.05, 1.0);
    float radius = static_cast<float>(body.radius_au()*AU_TO_KM);
    AtmosphereParametersGPU p{};
    float bottom = std::max(radius*0.01f, radius+static_cast<float>(minimum/polar_ratio));
    if (a.pressure_reference) bottom = radius+float(prepared ? prepared->opaque_altitude_km : a.bottom_altitude_km);
    p.radii_heights = {bottom, radius+height, a.rayleigh_scale_height_km, a.mie_scale_height_km};
    p.reference_shape = {radius, static_cast<float>(polar_ratio), a.pressure_reference ? bottom : radius, 0.0f};
    p.world_pole = {glm::vec3(pole), static_cast<float>(1.0/polar_ratio-1.0)};
    p.rayleigh = {a.rayleigh_scattering, 0.0f};
    p.mie_scattering = {a.mie_scattering, a.mie_asymmetry};
    p.mie_extinction = {a.mie_extinction, 0.0f};
    p.ground_albedo = {a.ground_albedo, 0.0f};
    if (const auto& physical = prepared) {
        p.radii_heights.z = float(physical->surface.scale_height_km);
        p.radii_heights.w = physical->mie_scale_height_km;
        p.mie_scattering = {physical->mie_scattering, physical->mie_asymmetry};
        p.mie_extinction = {physical->mie_extinction, 0.0f};
        p.rayleigh = {physical->rayleigh, 0.0f};
        p.mixed_absorption = {physical->mixed_absorption, 0.0f};
        p.ozone_absorption = {physical->ozone_absorption, 0.0f};
        p.ozone_params = {float(a.ozone_peak_km), float(a.ozone_width_km), 0.0f, 0.0f};
        p.profile_domain = physical->profile_domain;
        p.log_density = physical->log_density;
        p.aerosol_count.x = float(physical->aerosols.size());
        for (size_t i = 0; i < physical->aerosols.size(); ++i) {
            const auto& layer = physical->aerosols[i];
            p.aerosol_shape[i] = layer.shape;
            p.aerosol_scattering[i] = layer.scattering;
            p.aerosol_extinction[i] = layer.extinction;
            std::copy(layer.phase.begin(), layer.phase.end(), p.aerosol_phase.begin()+i*physics::AEROSOL_PHASE_SAMPLES);
        }
        if (a.pressure_pa == 0) p.mie_scattering = p.mie_extinction = glm::vec4(0.0f);
    }
    return p;
}

AtmosphereFrame make_atmosphere_frame(const ephemeris::SystemSnapshot& system, uint32_t index, const Camera& camera,
                                      const RenderSettings& settings, VkExtent2D extent, const AtmosphereRefraction& refraction_warp,
                                      double candidate_distance, double focal, glm::dvec3 eye, std::span<const float> minimum_heights_km) {
    bool limb_active = false;
    const auto b = system.body(index);
    const auto& a = b.definition.atmosphere;
    float height = b.render.atmosphere ? b.render.atmosphere->height_km : a.height_km;
    double polar_ratio = std::clamp(1.0-b.definition.oblateness, 0.05, 1.0);
    glm::dvec3 pole = glm::normalize(b.render.pole);
    auto atmosphere_space = [&](glm::dvec3 v) { return v+pole*glm::dot(v, pole)*(1.0/polar_ratio-1.0); };
    glm::dvec3 relative = atmosphere_space((eye-b.render.position.value)*AU_TO_KM);
    double distance = glm::length(relative);
    glm::dvec3 z = distance > 1e-12 ? relative/distance : glm::dvec3(0.0, 1.0, 0.0);
    glm::dvec3 x = glm::normalize(glm::cross(std::abs(z.y) < 0.99 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0), z));
    glm::dvec3 y = glm::cross(z, x);
    double minimum = std::min(0.0, b.definition.height_range_km.x);
    if (index < minimum_heights_km.size()) minimum = std::min(minimum, double(minimum_heights_km[index]));
    auto p = make_atmosphere_parameters(b.definition, b.render.atmosphere, pole, minimum);
    float bottom = p.radii_heights.x;
    float radius = p.reference_shape.x;
    p.camera = {0.0f, 0.0f, std::max(static_cast<float>(distance), bottom+0.001f), 0.0f};
    p.axis_x = {glm::vec3(x), 0.0f};
    p.axis_y = {glm::vec3(y), 0.0f};
    p.axis_z = {glm::vec3(z), 0.0f};
    for (uint32_t s = 0; s < system.size() && p.camera.w < 16.0f; ++s) {
        const auto star = system.body(s);
        if (!star.definition.is_star) continue;
        glm::dvec3 delta = star.render.position.value-b.render.position.value;
        double d = glm::length(delta);
        if (d < 1e-12) continue;
        glm::dvec3 direction = glm::normalize(atmosphere_space(delta/d));
        uint32_t star_slot = static_cast<uint32_t>(p.camera.w++);
        p.sun_direction[star_slot] = {glm::vec3(glm::dot(direction, x), glm::dot(direction, y), glm::dot(direction, z)),
            float(std::asin(std::clamp(star.opaque_radius_au()/d, 1e-7, 1.0)))};
        float irradiance = settings.hdr_enabled ? static_cast<float>(std::max(star.definition.lum_eq, 0.0)/(d*d)) : 1.0f;
        if (!system.illuminates(s, index)) irradiance = 0.0f;
        p.sun_irradiance[star_slot] = {star.definition.color_rgb*(irradiance*static_cast<float>(PI)), 0.0f};
        double star_radius = std::max(star.definition.radius_au(), 1e-12);
        float radiance = !system.illuminates(s, index) ? 0.0f :
            settings.hdr_enabled ? static_cast<float>(std::max(star.definition.lum_eq, 0.0)/(star_radius*star_radius)) : 1.0f;
        p.star_radiance[star_slot] = {star.definition.color_rgb*radiance, 0.0f};
        glm::dvec3 sight = star.render.position.value-eye;
        double range = glm::length(sight);
        glm::dvec3 seen = range > 1e-12 ? sight/range : glm::normalize(delta);
        p.star_disc[star_slot] = {glm::vec3(glm::dot(seen, x), glm::dot(seen, y), glm::dot(seen, z)),
            float(std::asin(std::clamp(star.opaque_radius_au()/std::max(range, 1e-12), 1e-7, 1.0)))};
    }
    auto projection = camera.proj_matrix(float(extent.width)/std::max(1u, extent.height), camera.compute_near_plane_au());
    auto view = camera.view_matrix_eye_relative();
    glm::dmat3 apparent_to_true = glm::transpose(refraction_warp.apparent);
    p.inverse_view_projection = glm::mat4(glm::dmat4(apparent_to_true)*glm::dmat4(glm::inverse(projection*view)));
    glm::dmat3 world_from_view = glm::inverse(glm::dmat3(glm::mat3(view)));
    glm::dmat3 view_to_world = apparent_to_true*world_from_view;
    glm::dvec3 ray_scale{1.0/projection[0][0], 1.0/projection[1][1], -1.0};
    for (int k = 0; k < 3; ++k) {
        glm::dvec3 axis = k == 0 ? x : k == 1 ? y : z;
        glm::dvec3 row;
        for (int j = 0; j < 3; ++j) row[j] = glm::dot(axis, atmosphere_space(view_to_world[j]))*ray_scale[j];
        p.view_ray[k] = {glm::vec3(row), 1.0f};
    }
    if (refraction_warp.differential) {
        auto to_host = [&](glm::dvec3 v) {
            return v+refraction_warp.host_pole*(glm::dot(v, refraction_warp.host_pole)*refraction_warp.host_stretch);
        };
        auto from_host = [&](glm::dvec3 v) {
            return v-refraction_warp.host_pole*(glm::dot(v, refraction_warp.host_pole)*refraction_warp.host_stretch/(1.0+refraction_warp.host_stretch));
        };
        for (int k = 0; k < 3; ++k) {
            glm::dvec3 row;
            for (int j = 0; j < 3; ++j)
                row[j] = glm::dot(refraction_warp.host_apparent_basis[k], to_host(world_from_view[j]))*ray_scale[j];
            p.refract_view[k] = {glm::vec3(row), 0.0f};
        }
        for (int j = 0; j < 3; ++j) {
            glm::dvec3 atm_f = atmosphere_space(from_host(refraction_warp.host_true_basis[j]));
            p.refract_to_atm[j] = {float(glm::dot(atm_f, x)), float(glm::dot(atm_f, y)), float(glm::dot(atm_f, z)), 0.0f};
        }
        p.refract_trig = refraction_warp.trig;
        p.refract_domain = refraction_warp.domain;
        p.refract_bend = refraction_warp.bend;
    }
    p.depth_params.x = settings.hdr_enabled ? settings.exposure : 1.0f;
    p.depth_params.y = settings.aerial_perspective_enabled ? 1.0f : 0.0f;
    p.depth_params.z = settings.direct_sunlight_only ? 1.0f : 0.0f;
    double point_radius = b.opaque_radius_au()*AU_TO_KM/std::max(candidate_distance, 1e-12)*focal;
    float blend = float(std::clamp(point_radius-1.5, 0.0, 1.0));
    p.depth_params.w = blend*blend*(3.0f-2.0f*blend);
    limb_active = false;
    p.limb_image.z = 2.0f;
    if (distance > p.radii_heights.y && settings.aerial_perspective_enabled) {
        double limb = std::asin(std::clamp(double(p.radii_heights.x)/distance, 0.0, 1.0));
        double edge = std::asin(std::clamp(double(p.radii_heights.y)/distance, 0.0, 1.0));
        bool star_image = false;
        if (settings.refraction_enabled && b.render.atmosphere && b.render.atmosphere->refraction)
            for (uint32_t star = 0; star < uint32_t(p.camera.w); ++star) {
                glm::dvec3 seen{p.star_disc[star]};
                glm::dvec3 sun = atmosphere_space(seen.x*x+seen.y*y+seen.z*z);
                double separation = std::atan2(std::hypot(glm::dot(sun, x), glm::dot(sun, y)), -glm::dot(sun, z));
                star_image = star_image || (glm::any(glm::greaterThan(glm::vec3(p.star_radiance[star]), glm::vec3(0.0f))) &&
                                            separation-p.star_disc[star].w/polar_ratio <= edge);
            }
        if (star_image) {
            const auto& refraction = *b.render.atmosphere->refraction;
            double horizon = refraction.horizon_impact(distance);
            for (size_t row = 0; row < p.limb_bend.size(); ++row) {
                double t_row = double(row)/double(p.limb_bend.size()-1);
                double impact = distance*std::sin(limb+(edge-limb)*t_row*t_row);
                glm::dvec3 bend = refraction.total_deflection(impact);
                for (int channel = 0; channel < 3; ++channel) if (!std::isfinite(bend[channel])) bend[channel] = bend.y;
                bool visible = impact >= horizon && std::isfinite(bend.y);
                p.limb_bend[row] = {glm::vec3(bend), visible ? 1.0f : 0.0f};
                if (visible) p.limb_image.y = std::max(p.limb_image.y, float(std::max({bend.x, bend.y, bend.z})));
            }
            p.limb_image.x = 1.0f;
        }
        limb_active = star_image || (edge-limb)*focal < LIMB_RESOLVED_PIXELS;
        if (limb_active) {
            double margin = 1.5/std::max(focal, 1e-9);
            p.limb_image.z = float(std::sin(std::max(limb-margin, 0.0)));
            p.limb_image.w = float(std::sin(std::min(edge+margin, PI/2)));
        }
    }
    bool particle_scattering = std::any_of(p.aerosol_scattering.begin(), p.aerosol_scattering.end(),
        [](glm::vec4 coefficient) { return glm::any(glm::greaterThan(glm::vec3(coefficient), glm::vec3(0.0f))); });
    p.aerosol_count.z = particle_scattering ? 1.0f : 0.0f;
    p.aerosol_count.y = particle_scattering || glm::any(glm::greaterThan(glm::vec3(p.mie_scattering), glm::vec3(0.0f))) ? 1.0f : 0.0f;
    return {p, pole, x, y, z, polar_ratio, distance, point_radius, height, bottom, radius, projection, view, limb_active};
}

}
