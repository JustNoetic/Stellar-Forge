#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define MAX_RING_PLANES 16
#define PI 3.14159265358979

uniform sampler3D u_stbn_tex;
uniform int u_atmo_noise_type; // 0 = Interleaved Gradient Noise (IGN), 1 = Spatiotemporal Blue Noise (STBN)

float get_ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float get_stbn(vec2 screen_pos, float frame_idx) {
    ivec3 sz = textureSize(u_stbn_tex, 0);
    if (sz.x >= 64) {
        ivec3 coord = ivec3(
            int(screen_pos.x) & 127,
            int(screen_pos.y) & 127,
            int(frame_idx) & 63
        );
        float base_noise = texelFetch(u_stbn_tex, coord, 0).r;
        float cycle = floor(frame_idx / 64.0);
        return fract(base_noise + cycle * 0.61803398875);
    }
    float base_ign = get_ign(screen_pos);
    return fract(base_ign + float(int(frame_idx) % 16) * 0.61803398875);
}

float get_stochastic_jitter(vec2 screen_pos, float frame_idx) {
    if (u_atmo_noise_type == 1) {
        return get_stbn(screen_pos, frame_idx);
    }
    // Default (0): Interleaved Gradient Noise with Golden Ratio Weyl sequence
    float base_ign = get_ign(screen_pos);
    return fract(base_ign + float(int(frame_idx) % 16) * 0.61803398875);
}

in vec3 f_local_pos;
in float f_clip_z;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
    vec4 u_caster_ozone[MAX_CASTERS];
};

layout(std430, binding = 2) buffer AllInstances {
    vec4 instances[];
};
uint u_caster_mask_lo;
uint u_caster_mask_hi;
uint u_ring_mask;
vec2 u_cur_solstice;

uniform vec3 u_camera_pos;

layout(std430, binding = 8) buffer AtmoData {
    vec3  u_body_offset;
    float u_atmo_radius_au;
    vec3  u_beta_rayleigh;
    float u_h_rayleigh;
    vec3  u_beta_mie;
    float u_h_mie;
    vec3  u_beta_abs_mixed;
    float u_mie_g;
    vec3  u_beta_abs_layered;
    float u_sun_intensity;
    float u_planet_radius_km;
    float u_atmo_radius_km;
    float u_au_to_km;
    int   u_num_samples;
    vec4  u_pole_obl;
    int   u_num_active_casters;
    bool  u_atmo_adaptive_steps;
    int   u_body_idx;
    float u_frame_counter;
    vec3  u_mie_albedo;          // per-channel single-scattering albedo (omega_0)
    float u_refractivity;        // surface (n_mix - 1), drives eclipse refraction
    vec4  u_active_casters[8];
    vec4  u_active_caster_poles_obl[8];
    float u_active_caster_R_minor[8];
    vec4  u_active_caster_atmos[8];
    float u_active_max_bend[8];
    vec4  u_active_caster_ozone[8];
    float u_ozone_peak_km;
    float u_ozone_width_km;
    float u_planet_clip_km;
    float u_max_adaptive_steps;
    vec4  u_precomp_opt;         // x: inv_h_rayleigh, y: inv_h_mie, z: inv_ozone_width, w: max_bend
    vec4  u_precomp_mie;         // x: c1, y: c2, z: c3, w: has_methane (1.0 or 0.0)
    vec4  u_star_color_irrad[4]; // rgb = star_color * irradiance * atmo_sun_intensity, w = sin_star
    vec4  u_star_dir_sph_eff[4]; // xyz = sun_dir_sph_const, w = cos_sun_eff
    vec4  u_star_pos_local[4];   // xyz = sun_pos_local_km, w = effective_star_rad
    vec4  u_star_solstice[4];    // x = solstice_factor, y = sun_pole_dot, z = dist_star_au, w = star_radius_au
    float u_active_scale_height[8]; // km, appended at byte offset 1024
};

uniform sampler2D u_ring_gradients;
uniform sampler2D u_ring_shadow_tex;
uniform int u_num_ring_planes;
#ifdef ATMO_QUALITY
const int u_atmo_quality = ATMO_QUALITY;
#else
uniform int u_atmo_quality;
#endif
// Scene clipping follows terrain elevations; table coordinates stay at the datum.
uniform float u_scattering_terrain_bottom_km;
uniform bool u_stochastic_noise;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec4 u_ring_params[MAX_RING_PLANES];
uniform int u_atmo_clip_mode;
uniform vec4 u_ring_station_cells[9];   // 33 station-cell boundaries in u space (Kmax = 32), packed 4 per vec4
uniform int u_ring_station_count;      // number of cells (boundaries = count + 1)
uniform int u_atmo_shadow_method;      // 0 = Station-Locked Slicing, 1 = Uniform Stochastic Raymarching, 2 = Bounded Subtraction (Blackrack)
#ifdef ATMO_BOUNDED_SHADOWS
const bool u_bounded_shadows = ATMO_BOUNDED_SHADOWS != 0;
#else
uniform bool u_bounded_shadows;
#endif

uniform uint u_ring_coplanar_mask[16];
uniform sampler2D u_eclipse_lut; // Kept to avoid uniform bound errors
uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform bool u_planetshine_enabled;

uniform sampler2D u_ringshine_map;
#include "common/ringshine_lookup.glsl"
uniform bool u_ringshine_enabled;
uniform int u_ringshine_band_count;
uniform sampler2D u_depth_texture;
uniform vec2 u_screen_res;

uniform sampler2D u_sky_view_lut;
uniform sampler2D u_sky_view_trans_lut;
// Per-star in-scatter slices of the sky-view LUT (bake locations 2-5): the
// Mode 3 ring-shadow slicing normalizes each star's deficit against its own
// light, so one star's shadow never darkens another star's illumination.
uniform sampler2D u_sky_view_star_lut[4];
uniform float u_density;
uniform vec3 u_atmo_tint;

uniform bool u_vrs_highres_pass;
uniform sampler2D u_lowres_trans;
uniform vec2 u_lowres_size;
uniform mat4 u_inv_proj;
uniform mat4 u_inv_view;

uniform bool u_atmo_optical_enabled; // High mode: extinction independent of ray-step budget
uniform bool u_temporal_accum;
uniform bool u_history_valid;
uniform float u_temporal_alpha;
uniform float u_prev_exposure;
uniform mat4 u_prev_view_proj;
uniform vec3 u_prev_body_offset;
uniform sampler2D u_history_scatter;
uniform sampler2D u_history_trans;

layout(location = 0, index = 0) out vec4 out_scattered;
layout(location = 0, index = 1) out vec4 out_transmittance;


vec3 toSphericalSpace(vec3 p, vec4 pole_scale) {
    float f_scale = pole_scale.w;
    if (f_scale <= 1.00001) return p;
    vec3 pole = pole_scale.xyz;
    float h = dot(p, pole);
    return p + (h * (f_scale - 1.0)) * pole;
}

vec3 fromSphericalSpace(vec3 p_sph, vec4 pole_scale) {
    float f_scale = pole_scale.w;
    if (f_scale <= 1.00001) return p_sph;
    vec3 pole = pole_scale.xyz;
    float h_sph = dot(p_sph, pole);
    return p_sph + (h_sph * (1.0 / f_scale - 1.0)) * pole;
}

#include "common/polar_winter.glsl"

#define ATMO_DATA_HAS_AU_TO_KM 1
#include "common/refraction.glsl"

#include "common/sun_terminator.glsl"
#include "common/scattering_segment.glsl"
#include "common/scattering_shine.glsl"
#include "common/aerial_lookup.glsl"
// get_oblate_radius / casterShadowTerm / compute_caster_shadow live in the
// shared include so the Mode 3 Sky-View LUT bake uses identical eclipse math.
#include "common/caster_shadow.glsl"
#include "common/ring_shadow_filter.glsl"
vec3 compute_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, vec3 planet_center_render, float star_radius, float star_obl, vec3 star_pole) {
    // Spherical-body (moon/planet) eclipse casters: shared with the Mode 3
    // Sky-View LUT bake via common/caster_shadow.glsl.
    vec3 shadow = compute_caster_shadow(eval_render_pos, L_dir, dist_to_star, star_radius, star_obl, star_pole);

    uint processed_mask = 0u;
    for (int k = 0; k < u_num_ring_planes; k++) {
        if ((u_ring_mask & (1u << k)) == 0u) continue;
        if ((processed_mask & (1u << k)) != 0u) continue;

        vec3 plane_center = u_ring_center[k];
        vec3 plane_normal = u_ring_normal[k];

        uint coplanar_mask = u_ring_coplanar_mask[k];
        processed_mask |= coplanar_mask;

        float denom = dot(L_dir, plane_normal);
        if (abs(denom) < 1e-8) continue;

        float t_ring = dot(plane_center - eval_render_pos, plane_normal) / denom;
        if (t_ring <= 0.0 || t_ring >= dist_to_star) continue;

        vec3 hit = eval_render_pos + t_ring * L_dir;
        vec3 vec_radial = hit - plane_center;
        float d = length(vec_radial);

        float r_star_proj = star_radius * t_ring / max(dist_to_star, 1e-6);
        vec3 L_plane = L_dir - denom * plane_normal;
        float L_plane_len = length(L_plane);

        float R_eff = r_star_proj;
        if (L_plane_len > 1e-5 && d > 1e-5) {
            vec3 L_proj = L_plane / L_plane_len;
            vec3 T = normalize(cross(plane_normal, L_dir));
            vec3 dir_radial = vec_radial / d;
            float cos_theta = dot(dir_radial, L_proj);
            float sin_theta = dot(dir_radial, T);
            R_eff = r_star_proj * sqrt( pow(cos_theta / max(1e-6, abs(denom)), 2.0) + pow(sin_theta, 2.0) );
        }

        float plane_occlusion = 0.0;

        for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
            if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
            float inner_r = u_ring_params[ring_idx].x;
            float outer_r = u_ring_params[ring_idx].y;

            if (u_atmo_quality >= 2) {
                float inv_mu = 1.0 / max(1e-4, abs(denom));
                float filtered_footprint = min(R_eff, (outer_r - inner_r) * 0.5);
                plane_occlusion += surface_ring_profile_occlusion(ring_idx, d * u_au_to_km,
                    inner_r * u_au_to_km, outer_r * u_au_to_km,
                    u_ring_params[ring_idx].z, inv_mu, filtered_footprint * u_au_to_km);
                continue;
            }

            float overlap_min = max(inner_r, d - R_eff);
            float overlap_max = min(outer_r, d + R_eff);

            if (overlap_min >= overlap_max) continue;

            float v_min = clamp((overlap_min - d) / max(1e-9, R_eff), -1.0, 1.0);
            float v_max = clamp((overlap_max - d) / max(1e-9, R_eff), -1.0, 1.0);

            float f_max = (v_max * sqrt(max(0.0, 1.0 - v_max*v_max)) + asin(v_max)) / PI + 0.5;
            float f_min = (v_min * sqrt(max(0.0, 1.0 - v_min*v_min)) + asin(v_min)) / PI + 0.5;
            float fraction = max(0.0, f_max - f_min);

            float p_mid = ((overlap_min + overlap_max) * 0.5 - inner_r) / max(1e-6, outer_r - inner_r);
            float alpha_mult = textureLod(u_ring_gradients, vec2(p_mid, (float(ring_idx) + 0.5) / float(textureSize(u_ring_gradients, 0).y)), 0.0).a;
            float opacity = u_ring_params[ring_idx].z;

            float tau = -log(max(1e-6, 1.0 - opacity * alpha_mult));
            float light_mu = max(1e-4, abs(dot(normalize(u_ring_normal[ring_idx]), L_dir)));
            plane_occlusion += fraction * (1.0 - exp(-tau / light_mu));
        }

        shadow *= vec3(1.0 - min(plane_occlusion, 1.0));
    }
    return shadow;
}

vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float a = dot(dir, dir);
    float b = dot(origin, dir);

    // Use cross product for perpendicular distance — avoids catastrophic cancellation
    // at large distances where origin and (b/a)*dir are both huge but nearly equal.
    // |origin × dir|² / |dir|² = perpendicular distance² from ray to sphere center.
    vec3 cross_vec = cross(origin, dir);
    float p2 = dot(cross_vec, cross_vec) / a;
    float r2 = radius * radius;

    if (p2 > r2) return vec2(1e10, -1e10);

    float d = sqrt((r2 - p2) / a);
    float t_closest = -b / a;

    return vec2(t_closest - d, t_closest + d);
}

vec3 get_transmittance(float r, float cos_theta) {
    float h_norm = clamp((r - u_planet_radius_km) / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float v = sqrt(h_norm);
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

vec3 get_transmittance_precomputed(float v, float cos_theta) {
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}



float map_t_to_s(float t, float s_start, float s_end, float s_min, float p) {
    float L = s_end - s_start;
    if (L < 1e-4) return s_start;
    if (abs(p - 1.0) < 0.01) return s_start + t * L;
    float t_min = clamp((s_min - s_start) / L, 0.0, 1.0);
    if (t <= t_min) {
        if (t_min < 1e-5) return s_min;
        float u = t / t_min;
        return s_min - (s_min - s_start) * pow(1.0 - u, p);
    } else {
        if (t_min > 0.99999) return s_min;
        float u = (t - t_min) / (1.0 - t_min);
        return s_min + (s_end - s_min) * pow(u, p);
    }
}

// Conservative hull of a projected ray's overlap with a padded annulus. Keep
// both lobes if it crosses the hole, but reject/trim rays wholly on one side.
// Closest-approach form avoids subtracting large quadratic discriminants.
bool ring_shadow_interval(vec3 A, vec3 B, float inner_r, float outer_r,
                          inout float lo, inout float hi) {
    float speed_sq = dot(B, B);
    if (speed_sq < 1e-12) {
        float d = length(A);
        return lo < hi && d >= inner_r && d <= outer_r;
    }
    float center = -dot(A, B) / speed_sq;
    vec3 closest = A + center * B;
    float closest_sq = dot(closest, closest);
    float outer_sq = outer_r * outer_r - closest_sq;
    if (outer_sq < 0.0) return false;
    float half_span = sqrt(outer_sq / speed_sq);
    lo = max(lo, center - half_span);
    hi = min(hi, center + half_span);
    if (lo >= hi) return false;
    float hole_sq = inner_r * inner_r - closest_sq;
    if (inner_r > 0.0 && hole_sq > 0.0) {
        float hole_span = sqrt(hole_sq / speed_sq);
        float hole_lo = center - hole_span, hole_hi = center + hole_span;
        if (lo >= hole_lo && hi <= hole_hi) return false;
        if (lo > hole_lo && lo < hole_hi) lo = hole_hi;
        if (hi > hole_lo && hi < hole_hi) hi = hole_lo;
    }
    return lo < hi;
}

bool check_ray_intersects_shadow(float s_start, float s_end, vec3 ray_dir, vec3 frag_local, vec3 planet_center_render, out float out_sh_min, out float out_sh_max) {
    out_sh_min = 1e9;
    out_sh_max = -1e9;
    if (s_start >= s_end) return false;
    bool has_hit = false;
    vec3 V = ray_dir;
    for (int s = 0; s < min(u_num_stars, 4); s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = (s < 4) ? u_star_solstice[s].w : u_stars_pos_radius[s].w;
        float dist_mid_star = (s < 4) ? u_star_solstice[s].z : length(star_pos - planet_center_render);
        vec3 sun_pos_local = (s < 4) ? u_star_pos_local[s].xyz : ((star_pos - planet_center_render) * u_au_to_km);
        vec3 L_mid = (length(sun_pos_local) > 1e-6) ? normalize(sun_pos_local) : vec3(0.0, 1.0, 0.0);

        if (u_num_ring_planes > 0 && u_ring_mask != 0u) {
            uint processed_mask = 0u;
            for (int k = 0; k < u_num_ring_planes; k++) {
                if ((u_ring_mask & (1u << k)) == 0u) continue;
                if ((processed_mask & (1u << k)) != 0u) continue;

                vec3 N = u_ring_normal[k];
                float denom = dot(L_mid, N);
                if (abs(denom) < 1e-8) continue;

                vec3 ring_center_local_km = (u_ring_center[k] - planet_center_render) * u_au_to_km;

                float ra_km = dot(ring_center_local_km - frag_local, N) / denom;
                float rb_km = -dot(ray_dir, N) / denom;

                float ring_s_valid_min = -1e9;
                float ring_s_valid_max = 1e9;
                if (abs(rb_km) < 1e-8) {
                    if (ra_km <= 0.0) { ring_s_valid_min = 1e9; ring_s_valid_max = -1e9; }
                } else {
                    float s_cross = -ra_km / rb_km;
                    if (rb_km < 0.0) {
                        ring_s_valid_max = s_cross;
                    } else {
                        ring_s_valid_min = s_cross;
                    }
                }
                ring_s_valid_min = max(s_start, ring_s_valid_min);
                ring_s_valid_max = min(s_end, ring_s_valid_max);
                if (ring_s_valid_min > ring_s_valid_max) continue;

                vec3 A_prime_km = (frag_local - ring_center_local_km) + ra_km * L_mid;
                vec3 B_km = ray_dir + rb_km * L_mid;

                // The midpoint footprint can under-bound the far penumbra.
                float t_max_km = max(0.0, max(ra_km + ring_s_valid_min * rb_km,
                                             ra_km + ring_s_valid_max * rb_km));
                float alpha_star = star_radius / max(dist_mid_star, 1e-6);
                float r_star_proj_km = alpha_star * t_max_km;
                float inv_denom_clamped = min(1.0 / max(1e-4, abs(denom)), 20.0);
                float R_eff_max_km = r_star_proj_km * inv_denom_clamped;

                uint coplanar_mask = u_ring_coplanar_mask[k];
                processed_mask |= coplanar_mask;
                for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
                    if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
                    float inner_r_km = u_ring_params[ring_idx].x * u_au_to_km;
                    float outer_r_km = u_ring_params[ring_idx].y * u_au_to_km;
                    float opacity = u_ring_params[ring_idx].z;
                    if (outer_r_km <= inner_r_km || opacity <= 0.0) continue;

                    float R_eff_clamped = min(R_eff_max_km, (outer_r_km - inner_r_km) * 0.5);
                    float inter_min = ring_s_valid_min, inter_max = ring_s_valid_max;
                    if (ring_shadow_interval(A_prime_km, B_km,
                            max(0.0, inner_r_km - R_eff_clamped), outer_r_km + R_eff_clamped,
                            inter_min, inter_max)) {
                        out_sh_min = min(out_sh_min, inter_min);
                        out_sh_max = max(out_sh_max, inter_max);
                        has_hit = true;
                    }
                }
            }
        }

        for (int c = 0; c < min(u_num_active_casters, 8); ++c) {
            vec3 D = (u_active_casters[c].xyz - planet_center_render) * u_au_to_km - frag_local;
            float first = s_start, last = s_end;
            // Equatorial radii conservatively contain every oblate projection.
            if (caster_shadow_interval(D, V, L_mid, u_active_casters[c].w * u_au_to_km,
                    star_radius / max(dist_mid_star, 1e-6), first, last)) {
                out_sh_min = min(out_sh_min, first);
                out_sh_max = max(out_sh_max, last);
                has_hit = true;
            }
        }
    }
    if (has_hit) {
        out_sh_min = max(s_start, out_sh_min);
        out_sh_max = min(s_end, out_sh_max);
        return out_sh_min <= out_sh_max;
    }
    return false;
}

// A segment entirely above the table's reference radius shares one view-ray
// family. Cache its initial optical depth instead of querying it in every cell.
bool atmo_regular_segment(vec3 a, vec3 b) {
    vec3 v = normalize(b - a);
    float closest = clamp(-dot(a, v), 0.0, length(b - a));
    return length(a + closest * v) >= u_scattering_bottom_km - 0.001;
}

vec3 atmo_view_transmittance(vec3 a, vec3 p, vec3 v, vec3 initial_tau,
                            bool ground, bool regular) {
    if (!regular) return endpoint_transmittance(a, p);
    vec3 tau = max(vec3(0.0), initial_tau - endpoint_tau(p, v, ground));
    float distance = length(p - a);
    if (distance < max(0.001, min(u_h_rayleigh, u_h_mie)) * 0.5) {
        // Avoid cancellation in the optical-depth table for meter-scale paths.
        vec3 R, M, extinction;
        endpoint_medium(0.5 * (a + p), R, M, extinction);
        tau = mix(extinction * distance, tau, endpoint_long_weight(distance));
    }
    return exp(-tau);
}

// Estimate the fraction of each star's light removed inside the shadow hull.
// Endpoint tables supply the hull's unshadowed energy and view extinction.
// Normalizing the sampled weights preserves both zero-shadow and full-shadow
// limits at any step count. Secondary light is not shadowed by the solar caster.
vec3 march_shadow_deficit(float first, float last, float ray_start, float ray_end,
        vec3 cam_sph, vec3 dir_sph, vec3 dir_world, int steps, float jitter,
        vec3 star_baseline[4]) {
    vec3 a = cam_sph + first * dir_sph;
    vec3 b = cam_sph + last * dir_sph;
    vec3 ray_a = cam_sph + ray_start * dir_sph;
    bool full_hull = first == ray_start && last == ray_end;
    vec3 hull_T = full_hull ? vec3(1.0) : endpoint_transmittance(a, b);
    vec3 before_T = full_hull ? vec3(1.0) : endpoint_transmittance(ray_a, a);
    vec3 view_direction = normalize(dir_sph);
    bool regular = atmo_regular_segment(a, b);
    bool ground = dot(a, view_direction) / length(a)
        < scattering_horizon(max(length(a), u_scattering_bottom_km));
    vec3 initial_tau = regular ? endpoint_tau(a, view_direction, ground) : vec3(0.0);
    float closest = clamp(-dot(cam_sph, dir_sph) / dot(dir_sph, dir_sph), first, last);
    vec3 deficit = vec3(0.0);
    for (int st = 0; st < min(u_num_stars, 4); ++st) {
        vec3 sun = u_star_dir_sph_eff[st].xyz;
        if (dot(sun, sun) < 1e-8) continue;
        sun = normalize(sun);
        vec3 hull_light = full_hull ? star_baseline[st]
            : before_T * endpoint_radiance_star(a, b, sun, hull_T,
                st, u_star_pos_local[st].w) * u_star_color_irrad[st].rgb;
        if (all(lessThan(hull_light, vec3(1e-20)))) continue;
        vec3 L = normalize(u_star_pos_local[st].xyz);
        float nu = clamp(dot(dir_world, L), -1.0, 1.0);
        float phase_R = 3.0 / (16.0 * PI) * (1.0 + nu * nu);
        float phase_M = u_precomp_mie.x * (1.0 + nu * nu)
            / pow(max(1e-4, u_precomp_mie.y - u_precomp_mie.z * nu), 1.5);
        vec3 energy = vec3(0.0), loss = vec3(0.0);
        float haze;
        vec3 boost;
        eval_polar_winter(0.5 * (a + b), u_star_solstice[st].xy, u_ring_mask != 0u ? 1.0 : 0.0,
            u_precomp_mie.w, haze, boost);
        for (int i = 0; i < steps; ++i) {
            float s0 = map_t_to_s(float(i) / float(steps), first, last, closest, 2.0);
            float s1 = map_t_to_s(float(i + 1) / float(steps), first, last, closest, 2.0);
            float station = mix(s0, s1, jitter);
            vec3 p = cam_sph + station * dir_sph;
            float r = max(length(p), 1e-6), mus = dot(p, sun) / r;
            vec3 R, M, extinction;
            endpoint_medium(p, R, M, extinction);
            float sin_p = u_planet_radius_km / max(r, u_planet_radius_km + 0.01);
            vec2 term = sun_terminator(mus, sin_p, sqrt(max(0.0, 1.0 - sin_p * sin_p)),
                u_star_pos_local[st].w, u_star_dir_sph_eff[st].w);
            float h = sqrt(clamp((r - u_planet_radius_km)
                / (u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0));
            vec3 solar_T = term.x > 1e-4 ? get_transmittance_precomputed(h, term.y) * term.x : vec3(0.0);
            vec3 psi = textureLod(u_multi_scatter_lut, vec2(scattering_sun_coord(mus), h), 0.0).rgb;
            vec3 weight = atmo_view_transmittance(a, p, view_direction, initial_tau, ground, regular)
                * max(0.0, s1 - s0)
                * ((R * boost * phase_R + M * haze * phase_M) * solar_T + (R + M) * psi);
            vec3 world_p = u_body_offset + fromSphericalSpace(p, u_pole_obl) / u_au_to_km;
            vec3 shadow = compute_shadow(world_p, L, u_star_solstice[st].z,
                u_body_offset, u_star_solstice[st].w,
                u_stars_poles_obl[st].w, u_stars_poles_obl[st].xyz);
            energy += weight;
            loss += weight * (1.0 - shadow);
        }
        vec3 fraction = clamp(loss / max(energy, vec3(1e-20)), 0.0, 1.0);
        deficit += min(star_baseline[st], hull_light * fraction);
    }
    return deficit;
}

void main() {
    if (f_clip_z < 0.0) discard;
    
    if (u_vrs_highres_pass && u_screen_res.x > 0.0 && u_screen_res.y > 0.0) {
        vec2 depth_uv = gl_FragCoord.xy / u_screen_res;
        
        vec2 low_res_size = (u_lowres_size.x > 0.0) ? u_lowres_size : vec2(textureSize(u_lowres_trans, 0));
        vec2 texel_pos = depth_uv * low_res_size - 0.5;
        vec2 p_low = floor(texel_pos);
        
        vec2 uv00 = (p_low + vec2(0.5, 0.5)) / low_res_size;
        vec2 uv10 = (p_low + vec2(1.5, 0.5)) / low_res_size;
        vec2 uv01 = (p_low + vec2(0.5, 1.5)) / low_res_size;
        vec2 uv11 = (p_low + vec2(1.5, 1.5)) / low_res_size;
        
        float t00_a = texture(u_lowres_trans, uv00).a;
        float t10_a = texture(u_lowres_trans, uv10).a;
        float t01_a = texture(u_lowres_trans, uv01).a;
        float t11_a = texture(u_lowres_trans, uv11).a;
        
        bool is_atmo_edge = (t00_a > 0.0 || t10_a > 0.0 || t01_a > 0.0 || t11_a > 0.0) &&
                            (t00_a == 0.0 || t10_a == 0.0 || t01_a == 0.0 || t11_a == 0.0);
        
        float d00 = texture(u_depth_texture, uv00).r;
        float d10 = texture(u_depth_texture, uv10).r;
        float d01 = texture(u_depth_texture, uv01).r;
        float d11 = texture(u_depth_texture, uv11).r;
        
        float high_depth_raw = texture(u_depth_texture, depth_uv).r;
        
        float log_far_denom = log2(u_depth_C * u_far + 1.0);
        float d00_lin = (d00 >= 0.99999) ? 1e12 : (exp2(d00 * log_far_denom) - 1.0) / u_depth_C;
        float d10_lin = (d10 >= 0.99999) ? 1e12 : (exp2(d10 * log_far_denom) - 1.0) / u_depth_C;
        float d01_lin = (d01 >= 0.99999) ? 1e12 : (exp2(d01 * log_far_denom) - 1.0) / u_depth_C;
        float d11_lin = (d11 >= 0.99999) ? 1e12 : (exp2(d11 * log_far_denom) - 1.0) / u_depth_C;
        float high_depth = (high_depth_raw >= 0.99999) ? 1e12 : (exp2(high_depth_raw * log_far_denom) - 1.0) / u_depth_C;
        
        float max_d_lin = max(max(d00_lin, d10_lin), max(d01_lin, d11_lin));
        float min_d_lin = min(min(d00_lin, d10_lin), min(d01_lin, d11_lin));
        
        float depth_tol_edge = max(high_depth * 0.05, 0.0001);
        bool is_depth_edge = (max_d_lin - min_d_lin) > depth_tol_edge;
        
        if (!is_atmo_edge && !is_depth_edge) {
            discard; // Inner pixel, rendered efficiently in low-res pass
        }
    }

    u_ring_mask = floatBitsToUint(instances[u_body_idx * 7 + 3].w);
    vec3 body_planetshine_dir = instances[u_body_idx * 7 + 4].xyz;
    vec3 body_planetshine_color = instances[u_body_idx * 7 + 5].xyz;

    vec3 planet_center_render = u_body_offset;
    vec3 f_pos_local_au = f_local_pos * u_atmo_radius_au;

    vec3 ray_origin_au = u_camera_pos;
    vec3 cam_to_body = planet_center_render - u_camera_pos;

    // Compute exact mathematical screen ray to avoid quantization noise
    // and moire patterns caused by interpolating f_pos_local_au across the proxy mesh triangles.
    vec2 ndc = (gl_FragCoord.xy / u_screen_res) * 2.0 - 1.0;
    vec4 clip_ray = vec4(ndc, -1.0, 1.0);
    vec4 eye_ray = u_inv_proj * clip_ray;
    eye_ray = vec4(eye_ray.xy, -1.0, 0.0);
    vec3 view_ray = normalize((u_inv_view * eye_ray).xyz);

    vec3 ray_dir = view_ray;
    bool is_refract_host = length(u_body_offset - u_refract_center) < 1e-7;


    vec3 ring_ray_dir = view_ray;
    vec3 ring_ray_origin_au = u_camera_pos;

    float s_min_au = 0.0;
    float ring_s_min_au = 0.0;

    if (u_refract_max_bend > 1e-6) {
        vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
        float d_km = length(cam_to_body * u_au_to_km);

        vec3 pole_n_approx = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
        float d_n_approx = dot(view_ray, pole_n_approx);
        if (abs(d_n_approx) > 1e-6) {
            float t_approx = dot(cam_to_body, pole_n_approx) / d_n_approx;
            if (t_approx > 0.0) {
                d_km = t_approx * u_au_to_km;
            }
        }

        // Total (un-parallaxed) bend: the anchored rotation below applies the
        // (1 - s_min/d) parallax geometrically; using the parallaxed alpha here
        // would double-count it.
        float alpha = compute_refraction_total(C_km, view_ray, d_km);
        if (alpha > 1e-7) {
            vec3 u_dir = C_km - view_ray * dot(C_km, view_ray);
            float u_len = length(u_dir);
            if (u_len > 1e-5) {
                u_dir /= u_len;
                ring_ray_dir = normalize(view_ray * cos(alpha) - u_dir * sin(alpha));

                float local_s_min = -dot(u_camera_pos - u_refract_center, view_ray);
                if (local_s_min > 0.0) {
                    ring_s_min_au = local_s_min;
                    ring_ray_origin_au = u_camera_pos + ring_s_min_au * (view_ray - ring_ray_dir);
                }
            }
        }

        if (!is_refract_host) {
            ray_dir = ring_ray_dir;
            ray_origin_au = ring_ray_origin_au;
            s_min_au = ring_s_min_au;
        } else {
            // Terrestrial Refraction for host planet atmosphere ray marching (horizon extension)
            vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
            float r_cam = length(C_km);

            vec3 pole_n = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
            float f_obl = u_pole_obl.w;
            float f_scale = (f_obl > 0.0 && f_obl < 0.99) ? (1.0 / (1.0 - f_obl)) : 1.0;
            vec3 C_scaled = C_km + pole_n * (dot(C_km, pole_n) * (f_scale * f_scale - 1.0));
            vec3 local_up = normalize(C_scaled);

            float local_refract_radius = u_refract_radius;
            if (f_obl > 0.001 && f_obl < 0.99) {
                vec3 P_dir = r_cam > 1e-6 ? (C_km / r_cam) : vec3(0.0, 1.0, 0.0);
                float cos_t = abs(dot(P_dir, pole_n));
                float k = 1.0 / (1.0 - f_obl);
                float denom = sqrt(max(1e-6, 1.0 + (k * k - 1.0) * cos_t * cos_t));
                local_refract_radius = u_refract_radius / denom;
            }

            float h = r_cam - local_refract_radius;
            if (h < u_refract_scale_height * 15.0) {
                float density = exp(-max(h, 0.0) / max(1e-4, u_refract_scale_height));

                // Physical geodetic coefficient of terrestrial refraction: k = (R / H) * (n_0 - 1)
                // Derived algebraically from u_refract_max_bend = 2 * (n_0 - 1) * sqrt(pi * R / (2H)):
                float k_refr = u_refract_max_bend * 0.5 * sqrt((2.0 * local_refract_radius) / max(1e-4, 3.141592653589793 * u_refract_scale_height));

                // Horizon dip angle theta_dip = sqrt(2 * h_eff / R)
                float h_eff = max(h, 0.002); // Minimum 2 meters eye level for landed observer
                float theta_dip_eff = sqrt(2.0 * h_eff / local_refract_radius);

                // Physical terrestrial horizon refraction angle: Delta_theta = 0.5 * k * density * theta_dip
                float max_terr_alpha = clamp(0.5 * k_refr * density * theta_dip_eff, 0.0, 0.05);

                if (max_terr_alpha > 1e-7) {
                    float mu = dot(view_ray, local_up);
                    float mu_horiz = -sqrt(max(0.0, 2.0 * max(h, 0.0) / local_refract_radius));
                    float delta_mu = sqrt(2.0 * u_refract_scale_height / local_refract_radius);

                    float x = (mu - mu_horiz) / max(1e-6, delta_mu);
                    float alpha = max_terr_alpha * exp(-x * x);

                    if (alpha > 1e-7) {
                        vec3 u_dir = local_up - view_ray * mu;
                        float u_len = length(u_dir);
                        if (u_len > 1e-5) {
                            u_dir /= u_len;
                            ray_dir = normalize(view_ray * cos(alpha) - u_dir * sin(alpha));
                        }
                    }
                }
            }
        }
    }

    // Gravitational Lensing Deflection (atmospheres seen past the lens).
    // Composed after atmospheric refraction: rotates both the sky ray and the
    // ring ray by the gravitational deflection angle.
    bool is_grav_lens_host = u_grav_lens_enabled && u_grav_lens_rs > 1e-6
        && length(u_body_offset - u_grav_lens_center) < 1e-6;
    if (u_grav_lens_enabled && u_grav_lens_rs > 1e-6 && !is_grav_lens_host) {
        vec3 C_km = (u_camera_pos - u_grav_lens_center) * u_au_to_km;
        float d_km = length(cam_to_body * u_au_to_km);
        bool is_sh = false;
        float alpha_gr = compute_gravitational_deflection(C_km, view_ray, d_km, is_sh);
        if (is_sh) {
            discard; // Blocked by the black hole shadow
        }
        if (alpha_gr > 1e-7) {
            vec3 u_dir = C_km - view_ray * dot(C_km, view_ray);
            float u_len = length(u_dir);
            if (u_len > 1e-5) {
                u_dir /= u_len;
                vec3 bent = normalize(ray_dir * cos(alpha_gr) - u_dir * sin(alpha_gr));
                float local_s_min = -dot(u_camera_pos - u_grav_lens_center, view_ray);
                if (local_s_min > 0.0) {
                    s_min_au = max(s_min_au, local_s_min);
                    ring_s_min_au = max(ring_s_min_au, local_s_min);
                    ray_origin_au = u_camera_pos + s_min_au * (view_ray - bent);
                    ring_ray_origin_au = u_camera_pos + ring_s_min_au * (view_ray - bent);
                }
                ray_dir = bent;
                ring_ray_dir = bent;
            }
        }
    }

    // Precise ray origin in body-local coordinates, avoiding catastrophic cancellation.
    // We use the closest approach on the unrefracted ray to find a stable anchor point.
    float dist_to_center = length(cam_to_body);
    float bounding_radius_au = length(f_pos_local_au);
    float ray_shift_au = 0.0;

    vec3 cam_local_au;
    vec3 ring_cam_local_au;

    if (dist_to_center > bounding_radius_au * 4.0) {
        // Find closest approach of the unrefracted ray relative to the planet center
        float t_ca = -dot(f_pos_local_au, view_ray);
        vec3 closest_approach_au = f_pos_local_au + t_ca * view_ray;

        // Place precise origin 2.0 bounding radii in front of the closest approach
        float dist_from_f_pos = t_ca - 2.0 * bounding_radius_au;
        vec3 precise_origin_au = f_pos_local_au + dist_from_f_pos * view_ray;

        // Compute distance from the camera to this new origin
        float dist_to_f_pos = length(f_pos_local_au + cam_to_body);
        ray_shift_au = dist_to_f_pos + dist_from_f_pos;

        // Apply refraction shift laterally at this distance
        cam_local_au = precise_origin_au + (ray_shift_au - s_min_au) * (ray_dir - view_ray);
        ring_cam_local_au = precise_origin_au + (ray_shift_au - ring_s_min_au) * (ring_ray_dir - view_ray);
    } else {
        cam_local_au = ray_origin_au - planet_center_render;
        ring_cam_local_au = ring_ray_origin_au - planet_center_render;
    }

    vec3 cam_local = cam_local_au * u_au_to_km;
    vec3 ring_cam_local = ring_cam_local_au * u_au_to_km;

    vec3 f_pos_local = f_pos_local_au * u_au_to_km;

    vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
    vec3 cam_local_sph = toSphericalSpace(cam_local, u_pole_obl);

    vec2 s_atmo = raySphereIntersect(cam_local_sph, ray_dir_sph, u_atmo_radius_km);
    if (s_atmo.x > s_atmo.y) discard;

    // Clamped camera distance for analytical planet intersection.
    // Clamping to slightly outside the sphere (1e-4 km = 10 cm) ensures landed origins
    // do not produce negative entry roots (t1 <= 0) when intersecting the planet.
    float dist_to_planet_center_km = length(cam_local_sph);
    bool cam_inside = dist_to_planet_center_km <= u_atmo_radius_km;
    vec3 cam_pos_eff = cam_local_sph;
    if (dist_to_planet_center_km < u_planet_radius_km + 1e-4) {
        cam_pos_eff = (dist_to_planet_center_km > 1e-4)
            ? (cam_local_sph * ((u_planet_radius_km + 1e-4) / dist_to_planet_center_km))
            : vec3(0.0, u_planet_radius_km + 1e-4, 0.0);
    }

    float clip_radius = (u_atmo_quality == 3 && u_scattering_enabled && u_scattering_terrain_bottom_km > 0.0 && cam_inside)
        ? u_scattering_terrain_bottom_km : u_planet_clip_km;
    vec2 s_planet = raySphereIntersect(cam_pos_eff, ray_dir_sph, clip_radius);

    float s_start = max(0.0, s_atmo.x);
    float s_end = s_atmo.y;
    float scene_limit = s_atmo.y;
    bool has_scene_surface = false;

    bool hits_surface = false;
    if (s_planet.x > 0.0 && s_planet.x < s_end) {
        s_end = s_planet.x;
        hits_surface = true;
    }

    vec3 frag_local = cam_local;

    float closest_s_ring = 1e10;

    if (u_atmo_clip_mode != 0 && u_num_ring_planes > 0) {
        for (int k = 0; k < u_num_ring_planes; k++) {
            vec3 ring_center_world_rel = u_ring_center[k];
            // Allow clipping against any ring plane in the system to support moon atmospheres correctly layering with host planet rings
            // if (length(ring_center_world_rel - u_body_offset) > 1e-4) continue;

            vec3 ring_center_local = (ring_center_world_rel - u_body_offset) * u_au_to_km;
            vec3 ring_normal = u_ring_normal[k];

            float denom = dot(ring_ray_dir, ring_normal);
            if (abs(denom) > 1e-8) {
                float s_ring = dot(ring_center_local - ring_cam_local, ring_normal) / denom;
                // ray_shift_au artificially pushes the ray origin forward to avoid precision loss.
                // If it pushes the origin past the ring plane, s_ring becomes negative.
                // We must accept it if it is still physically in front of the TRUE camera.
                if (s_ring + ray_shift_au * u_au_to_km > 0.0) {
                    float ring_opacity = u_ring_params[k].z;
                    if (ring_opacity > 1e-6) {
                        vec3 hit_pt = ring_cam_local + s_ring * ring_ray_dir;
                        float hit_r = length(hit_pt - ring_center_local);
                        float inner_r_km = u_ring_params[k].x * u_au_to_km;
                        float outer_r_km = u_ring_params[k].y * u_au_to_km;
                        if (hit_r >= inner_r_km && hit_r <= outer_r_km) {
                            if (s_ring < closest_s_ring) {
                                closest_s_ring = s_ring;
                            }
                        }
                    }
                }
            }
        }
    }

    if (u_atmo_clip_mode == 1) {
        s_start = max(s_start, closest_s_ring);
    } else if (u_atmo_clip_mode == 2) {
        s_end = min(s_end, closest_s_ring);
        scene_limit = min(scene_limit, closest_s_ring);
    }

    if (u_num_active_casters > 0) {
        for (int i = 0; i < u_num_active_casters; i++) {
            vec3 caster_pos = u_active_casters[i].xyz;
            float caster_r = u_active_casters[i].w * u_au_to_km;
            vec3 caster_local = (caster_pos - planet_center_render) * u_au_to_km;

            if (length(caster_local) < 1.0) continue;

            vec4 pole_obl = u_active_caster_poles_obl[i];
            vec3 origin_c = toSphericalSpace(frag_local - caster_local, pole_obl);
            vec3 dir_c = toSphericalSpace(ray_dir, pole_obl);

            vec2 s_c = raySphereIntersect(origin_c, dir_c, caster_r);
            if (s_c.x > 0.0 && s_c.x < s_end) {
                s_end = s_c.x;
                scene_limit = min(scene_limit, s_c.x);
            }
        }
    }

    if (u_screen_res.x > 0.0 && u_screen_res.y > 0.0) {
        vec2 depth_uv = gl_FragCoord.xy / u_screen_res;
        
        // Manually bilinear filter the high-res depth texture to prevent perfectly vertical Moiré bands
        // caused by sub-pixel snapping when sampling a GL_NEAREST depth buffer at a lower resolution.
        vec2 high_res_size = vec2(textureSize(u_depth_texture, 0));
        float log_depth;
        if (abs(u_screen_res.x - high_res_size.x) > 1.0) {
            vec2 texel_pos = depth_uv * high_res_size - 0.5;
            vec2 p = floor(texel_pos);
            vec2 f = fract(texel_pos);
            
            float d00 = texture(u_depth_texture, (p + vec2(0.5, 0.5)) / high_res_size).r;
            float d10 = texture(u_depth_texture, (p + vec2(1.5, 0.5)) / high_res_size).r;
            float d01 = texture(u_depth_texture, (p + vec2(0.5, 1.5)) / high_res_size).r;
            float d11 = texture(u_depth_texture, (p + vec2(1.5, 1.5)) / high_res_size).r;
            
            log_depth = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
        } else {
            log_depth = texture(u_depth_texture, depth_uv).r;
        }
        if (log_depth < 0.99999) {
            float log_far_denom = log2(u_depth_C * u_far + 1.0);
            float scene_clip_z = (exp2(log_depth * log_far_denom) - 1.0) / u_depth_C;
            vec3 cam_fw = -vec3(view[0][2], view[1][2], view[2][2]);
            // Terrain depth is written after apparent-position refraction.
            // Its projection cosine belongs to the screen ray; recover the
            // distance before mapping it onto the atmospheric ray.
            float cos_angle = dot((u_atmo_quality == 3 && u_scattering_enabled) ? view_ray : ray_dir, cam_fw);
            if (cos_angle > 1e-4) {
                float s_depth = (scene_clip_z * u_au_to_km) / cos_angle;
                s_depth -= ray_shift_au * u_au_to_km; // Convert from camera-relative to O_local_km relative

                // Only clamp and mark surface hit if the depth belongs to a local object within the planet's atmospheric envelope.
                // Celestial objects (Sun at 1 AU, distant stars, other bodies) are far beyond max_local_ground_s
                // and must NOT clamp s_end or set hits_surface, allowing atmospheric sunset extinction to apply.
                float max_local_ground_s = length(cam_local_sph) + u_atmo_radius_km + 100.0;
                if (s_depth <= max_local_ground_s) {
                    // For space observers (!cam_inside), planetary terrain forms the macro ground boundary
                    // and is evaluated by the smooth analytical Sky-View LUT without faceted mesh sagitta.
                    // Only explicit foreground objects in space/air (satellites, moons, spacecraft > 50 km)
                    // clamp s_end and set has_scene_surface.
                    // For observers inside the atmosphere (cam_inside), terrain in front of the horizon
                    // acts as an authoritative scene surface for aerial perspective.
                    float scene_margin = cam_inside ? (dist_to_center * 2500.0) : max(50.0, dist_to_center * 2500.0);
                    if (s_depth < s_end - scene_margin) {
                        s_end = s_depth;
                        has_scene_surface = true;
                        hits_surface = true;
                    }
                }
            }
        }
    }

    if (s_start >= s_end) discard;

    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;       // Mie extinction
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;            // Mie scattering = omega_0 * extinction
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M); // Mie absorption
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;

    float ray_len = s_end - s_start;
    float s_closest = clamp(-dot(cam_local_sph, ray_dir_sph) / dot(ray_dir_sph, ray_dir_sph), s_start, s_end);
    float min_altitude = max(0.0, length(cam_local_sph + s_closest * ray_dir_sph) - u_planet_radius_km);

    vec3 pole_dir_norm = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    float max_bend = u_precomp_opt.w;
    float inv_h_rayleigh = u_precomp_opt.x;
    float inv_h_mie = u_precomp_opt.y;
    float inv_ozone_width = u_precomp_opt.z;

    // Grade cell boundaries towards dense gas, then sample uniformly inside
    // each physical cell. Jitter must not pass through the grading function.
    float grade_p = 2.0;
    if (u_atmo_quality == 1) {
        // Keep Low's original cell distribution for its cheaper cell integral.
        float peak_rho_R = exp(-min_altitude * inv_h_rayleigh);
        float peak_rho_M = exp(-min_altitude * inv_h_mie);
        float peak_ext = dot(beta_R, vec3(0.333333)) * peak_rho_R
                       + dot(beta_M_ext, vec3(0.333333)) * peak_rho_M;
        float tau_ray = peak_ext * min(ray_len, 2.0 * sqrt(max(0.0,
            2.0 * u_planet_radius_km * u_h_rayleigh + u_h_rayleigh * u_h_rayleigh)));
        grade_p = hits_surface && tau_ray > 1.0
            ? clamp(1.0 + 0.5 * log(tau_ray), 1.0, 2.0) : 1.0;
    }

    float sh_min = s_start;
    float sh_max = s_end;
    bool has_shadow = false;
    if (u_atmo_quality == 3 && ((u_num_ring_planes > 0 && u_ring_mask != 0u)
                               || u_num_active_casters > 0)) {
        has_shadow = check_ray_intersects_shadow(s_start, s_end, ray_dir,
            frag_local, planet_center_render, sh_min, sh_max);
    }

    bool use_bounded = (u_atmo_quality == 3 && u_bounded_shadows && u_scattering_enabled && has_shadow);

    float march_start = s_start;
    float march_end = s_end;
    if (use_bounded) {
        // Dilate by 2 km to smoothly capture penumbral fringes into full sunlight
        march_start = max(s_start, sh_min - 2.0);
        march_end = min(s_end, sh_max + 2.0);
        if (march_start >= march_end - 0.1) {
            use_bounded = false;
        }
    }

    int integration_quality = u_atmo_quality;

    float jitter = 0.5;
    if ((integration_quality != 3 || has_shadow) && u_stochastic_noise) {
        jitter = get_stochastic_jitter(gl_FragCoord.xy, u_frame_counter);
    }

    int steps = u_num_samples;
    if ((integration_quality != 3 || has_shadow) && u_atmo_adaptive_steps) {
        float base_steps = float(u_num_samples);
        float max_adaptive = max(base_steps, u_max_adaptive_steps > 0.0 ? u_max_adaptive_steps : 128.0);
        float min_steps = clamp(base_steps * 0.15, 3.0, 8.0);

        // Reference distance where ray transitions from "short ray" to "full baseline atmospheric ray"
        // Scale height H (~8 km for Earth); 3*H covers ~95% of atmospheric density profile vertically.
        float d_base = max(u_h_rayleigh * 3.0, 25.0);

        // Density / optical thickness factor: 1.0 at sea level/dense atmosphere, decaying towards 0 at vacuum
        float alt_factor = clamp(exp(-min_altitude / max(1e-3, u_h_rayleigh * 3.0)), 0.0, 1.0);

        float continuous_steps;
        if (ray_len <= d_base) {
            // Short ray to ground / near surface: steps scale down proportionally to distance traveled
            float t_short = clamp(ray_len / d_base, 0.0, 1.0);
            continuous_steps = mix(min_steps, base_steps, t_short);
        } else {
            // Long ray through thick atmosphere into infinity / horizon: scales up using extra steps
            float d_max = 2.0 * sqrt(max(0.0, u_atmo_radius_km * u_atmo_radius_km - u_planet_radius_km * u_planet_radius_km));
            float t_long = clamp((ray_len - d_base) / max(1e-3, d_max - d_base), 0.0, 1.0);

            // Sub-linear response (sqrt) ensures upward rays to infinity (zenith ~100 km) receive meaningful extra steps,
            // while horizon/limb chords (~1000-2000 km) scale up to max_adaptive.
            float length_boost = sqrt(t_long);
            float extra_importance = length_boost * alt_factor;

            continuous_steps = mix(base_steps, max_adaptive, extra_importance);
        }

        // Softened stochastic dithering across fractional step count boundaries:
        // Converts sharp integer step rings into imperceptible smooth transitions without high-frequency flicker
        float step_dither = u_stochastic_noise ? (jitter - 0.5) * 0.5 : 0.0;
        steps = int(clamp(continuous_steps + step_dither + 0.5, min_steps, max_adaptive));
    }

    if (use_bounded) {
        float march_len = max(0.0, march_end - march_start);
        float base_bounded = clamp(march_len / 12.0 + 8.0, 8.0, 32.0);
        float step_dither = u_stochastic_noise ? (jitter - 0.5) * 0.5 : 0.0;
        // Bounding may reduce the configured/adaptive budget, never raise it.
        // In particular, a 9-step non-adaptive march must not become 32 steps.
        steps = max(1, min(steps, int(base_bounded + step_dither + 0.5)));
    }

    float march_s_closest = clamp(-dot(cam_local_sph, ray_dir_sph) / dot(ray_dir_sph, ray_dir_sph), march_start, march_end);
    float march_grade_p = (use_bounded && march_end < s_end - 0.1) ? 1.0 : grade_p;

    vec3 scattered = vec3(0.0);
    vec3 final_transmittance = vec3(1.0);
    vec3 star_baseline[4];
    for (int st = 0; st < 4; ++st) star_baseline[st] = vec3(0.0);
    vec3 view_a = cam_local_sph + s_start * ray_dir_sph;
    vec3 view_b = cam_local_sph + s_end * ray_dir_sph;
    vec3 view_direction = normalize(ray_dir_sph);
    bool view_regular = false, view_ground = false;
    vec3 view_initial_tau = vec3(0.0);
    vec3 high_transmittance = vec3(1.0);
    if (integration_quality == 2 && u_atmo_optical_enabled) {
        view_regular = atmo_regular_segment(view_a, view_b);
        view_ground = dot(view_a, view_direction) / length(view_a)
            < scattering_horizon(max(length(view_a), u_scattering_bottom_km));
        if (view_regular) view_initial_tau = endpoint_tau(view_a, view_direction, view_ground);
        high_transmittance = atmo_view_transmittance(view_a, view_b, view_direction,
            view_initial_tau, view_ground, view_regular);
    }

    if (integration_quality == 3 && u_scattering_enabled
        && (has_scene_surface || u_atmo_clip_mode != 0 || scene_limit < s_atmo.y)) {
        // The sky-view LUT ends at the smooth ground/top boundary. For actual
        // scene hits and split atmosphere passes, query only the visible segment.
        // Keep the unnormalized spherical-space direction: s_start/s_end are
        // physical ray distances, including the oblate transform and origin shift.
        vec3 a = cam_local_sph + s_start * ray_dir_sph;
        vec3 b = cam_local_sph + s_end * ray_dir_sph;
        float r_b = length(b);
        if (r_b < u_scattering_bottom_km && r_b > 1e-4) {
            b *= (u_scattering_bottom_km / r_b);
        }
        float volume_weight = 0.0;
        bool used_volume = u_atmo_clip_mode == 0
            && aerial_lookup(a, b, ray_dir, scattered, final_transmittance, volume_weight);
        if (!used_volume || volume_weight < 1.0) {
            vec3 volume_L = scattered, volume_T = final_transmittance;
            scattered = vec3(0.0);
            final_transmittance = endpoint_transmittance(a, b);
            for (int st = 0; st < min(u_num_stars, 4); ++st) {
                if (dot(u_star_dir_sph_eff[st].xyz, u_star_dir_sph_eff[st].xyz) > 1e-8) {
                    star_baseline[st] = endpoint_radiance_star(a, b,
                        normalize(u_star_dir_sph_eff[st].xyz), final_transmittance,
                        st, u_star_pos_local[st].w) * u_star_color_irrad[st].rgb;
                    scattered += star_baseline[st];
                }
            }
            scattered += endpoint_secondary_light(a, b, final_transmittance,
                body_planetshine_dir, body_planetshine_color, u_ring_mask);
            if (used_volume) {
                scattered = mix(scattered, volume_L, volume_weight);
                final_transmittance = mix(final_transmittance, volume_T, volume_weight);
            }
        }
        if (has_shadow && used_volume && volume_weight >= 1.0) {
            vec3 direct_T = endpoint_transmittance(a, b);
            for (int st = 0; st < min(u_num_stars, 4); ++st) {
                if (dot(u_star_dir_sph_eff[st].xyz, u_star_dir_sph_eff[st].xyz) > 1e-8)
                    star_baseline[st] = endpoint_radiance_star(a, b,
                        normalize(u_star_dir_sph_eff[st].xyz), direct_T,
                        st, u_star_pos_local[st].w) * u_star_color_irrad[st].rgb;
            }
        }
    } else if (integration_quality == 3) {


        // Camera position and Zenith in planet local spherical frame (km)
        vec3 true_cam_local_km = (u_camera_pos - u_body_offset) * u_au_to_km;
        vec3 true_cam_sph_km = toSphericalSpace(true_cam_local_km, u_pole_obl);
        float D = length(true_cam_sph_km);
        vec3 u_zenith = true_cam_sph_km / max(D, 1e-6);
        float D_eff = max(D, u_planet_radius_km);

        // Primary sunlight direction in spherical frame
        vec3 L_sun = (u_num_stars > 0) ? normalize(u_star_dir_sph_eff[0].xyz) : vec3(0.0, 1.0, 0.0);
        if (length(L_sun) < 1e-4) {
            vec3 sun_pos_local = (u_num_stars > 0) ? ((u_stars_pos_radius[0].xyz - planet_center_render) * u_au_to_km) : vec3(0.0, 1.0, 0.0);
            L_sun = normalize(toSphericalSpace(normalize(sun_pos_local), u_pole_obl));
        }

        // Build continuous orthonormal basis aligned with local Zenith and Sun Azimuth (matching sky_view_lut.frag)
        vec3 L_proj = L_sun - dot(L_sun, u_zenith) * u_zenith;
        vec3 x_basis;
        float len_L_proj = length(L_proj);
        if (len_L_proj > 1e-4) {
            x_basis = L_proj / len_L_proj;
        } else {
            vec3 arb = (abs(u_zenith.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            x_basis = normalize(cross(u_zenith, arb));
        }
        vec3 y_basis = cross(u_zenith, x_basis);

        vec3 V = normalize(ray_dir_sph);
        float u_lut = 0.0;
        float v_lut = 0.0;
        float half_tex_y = 0.5 / float(textureSize(u_sky_view_lut, 0).y);
        cam_inside = (D <= u_atmo_radius_km);
        bool is_ground = false;
        float t_limb = 0.0;

        if (cam_inside) {
            float cos_theta = clamp(dot(V, u_zenith), -1.0, 1.0);
            float theta = acos(cos_theta);
            float x_proj = dot(V, x_basis);
            float y_proj = dot(V, y_basis);
            float phi = atan(y_proj, x_proj);
            if (phi < 0.0) phi += 2.0 * PI;
            u_lut = phi / (2.0 * PI);

            float sin_horizon = clamp(u_planet_radius_km / D_eff, 0.0, 1.0);
            float theta_horizon = PI - asin(sin_horizon);

            if (hits_surface) {
                // Ground disk: [theta_horizon, PI]
                float theta_clamped = max(theta, theta_horizon);
                float t = sqrt(clamp((theta_clamped - theta_horizon) / max(1e-5, PI - theta_horizon), 0.0, 1.0));
                v_lut = 0.5 * (1.0 - t);
                v_lut = clamp(v_lut, half_tex_y, 0.5 - half_tex_y);
                is_ground = true;
            } else {
                // Sky dome: [0, theta_horizon]
                // For rays near or below the geometric horizon looking at the sky / setting sun,
                // clamp to theta_horizon to sample the tangent atmospheric path with maximum sunset extinction.
                float theta_clamped = min(theta, theta_horizon);
                float t = sqrt(clamp((theta_horizon - theta_clamped) / max(1e-5, theta_horizon), 0.0, 1.0));
                v_lut = 0.5 + 0.5 * t;
                v_lut = clamp(v_lut, 0.5 + half_tex_y, 1.0 - half_tex_y);
                is_ground = false;
            }
        } else {
            // Space observer: compute stable impact parameter from local sphere-centered ray
            // cam_local_sph is already anchored near the planet (~2 bounding radii away)
            float t_ca = -dot(cam_local_sph, V);
            vec3 P_ca = cam_local_sph + t_ca * V;
            float r_ca = length(P_ca);

            // Project P_ca onto view plane basis for azimuth phi (in kilometers, immune to float32 cancellation)
            float x_proj_ca = dot(P_ca, x_basis);
            float y_proj_ca = dot(P_ca, y_basis);
            float phi_ca = (r_ca > 1e-4) ? atan(y_proj_ca, x_proj_ca) : 0.0;
            if (phi_ca < 0.0) phi_ca += 2.0 * PI;
            u_lut = phi_ca / (2.0 * PI);

            is_ground = hits_surface || (u_screen_res.x <= 0.0 && r_ca <= u_planet_radius_km);
            if (is_ground) {
                // Ground disk: r_ca in [0, R_planet]
                float r_clamped = min(r_ca, u_planet_radius_km);
                float t = sqrt(clamp(1.0 - r_clamped / max(1e-3, u_planet_radius_km), 0.0, 1.0));
                v_lut = 0.5 * (1.0 - t);
                v_lut = clamp(v_lut, half_tex_y, 0.5 - half_tex_y);
            } else {
                // Atmosphere limb: r_ca in [R_planet, R_atmo]
                float r_clamped = clamp(r_ca, u_planet_radius_km, u_atmo_radius_km);
                float t = sqrt(clamp((r_clamped - u_planet_radius_km) / max(1e-3, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0));
                t_limb = t;
                v_lut = 0.5 + 0.5 * t;
                v_lut = clamp(v_lut, 0.5 + half_tex_y, 1.0 - half_tex_y);
            }
        }

        vec4 atmo_sample = texture(u_sky_view_lut, vec2(u_lut, v_lut));
        vec3 L_scatter = atmo_sample.rgb;
        vec3 T_lut = texture(u_sky_view_trans_lut, vec2(u_lut, v_lut)).rgb;
        if (!cam_inside && !is_ground) {
            float limb_fade = clamp((1.0 - t_limb) * 128.0, 0.0, 1.0);
            L_scatter *= limb_fade;
            T_lut = mix(vec3(1.0), T_lut, limb_fade);
        }

        scattered = L_scatter;
        final_transmittance = T_lut;
        if (has_shadow && u_scattering_enabled) {
            for (int st = 0; st < min(u_num_stars, 4); ++st) {
                star_baseline[st] = texture(u_sky_view_star_lut[st], vec2(u_lut, v_lut)).rgb;
                if (!cam_inside && !is_ground)
                    star_baseline[st] *= clamp((1.0 - t_limb) * 128.0, 0.0, 1.0);
            }
        }
    } else {
        for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = (s < 4) ? u_star_solstice[s].w : u_stars_pos_radius[s].w;
        float dist_mid_star = (s < 4) ? u_star_solstice[s].z : length(star_pos - planet_center_render);
        float sin_star = (s < 4) ? u_star_color_irrad[s].w : (star_radius / max(dist_mid_star, star_radius + 1e-6));
        vec3 sun_pos_local = (s < 4) ? u_star_pos_local[s].xyz : ((star_pos - planet_center_render) * u_au_to_km);
        float effective_star_rad = (s < 4) ? u_star_pos_local[s].w : (sin_star + max_bend);
        vec3 sun_dir_sph_const = (s < 4) ? u_star_dir_sph_eff[s].xyz : normalize(toSphericalSpace(normalize(sun_pos_local), u_pole_obl));
        float cos_sun_eff = (s < 4) ? u_star_dir_sph_eff[s].w : sqrt(max(0.0, 1.0 - effective_star_rad * effective_star_rad));
        float solstice_factor = (s < 4) ? u_star_solstice[s].x : 0.0;
        float sun_pole_dot = (s < 4) ? u_star_solstice[s].y : 0.0;

        vec3 L = (length(sun_pos_local) > 1e-6) ? normalize(sun_pos_local) : vec3(0.0, 1.0, 0.0);
        vec3 L_mid = L;

        vec3 global_eclipse_shadow = vec3(1.0);
        vec3 end_eclipse_shadow = vec3(1.0);
        bool skip_volumetric_shadow = false;

        vec4 ring_s1_out = vec4(1e9);
        vec4 ring_s2_out = vec4(-1e9);
        vec4 ring_inner = vec4(0.0);
        vec4 ring_outer = vec4(1.0);
        vec4 ring_inv_mu = vec4(0.0);
        vec4 ring_opacity = vec4(0.0);
        ivec4 ring_rows = ivec4(0);
        vec3 ring_A_prime[4];
        vec3 ring_B[4];
        vec3 ring_L_proj[4];
        vec3 ring_T_vec[4];
        vec4 ring_r_star_proj = vec4(0.0);
        vec4 ring_inv_denom = vec4(1.0);

        bool has_body_shadow = false;
        int ring_count = 0;


        float s_mid = (march_start + march_end) * 0.5;
        vec3 mid_pos = frag_local + s_mid * ray_dir;
        vec3 mid_render = mid_pos / u_au_to_km + planet_center_render;
        vec3 mid_to_star = star_pos - mid_render;
        float sr_start = star_radius / max(dist_mid_star, 1e-6);
        vec3 O = frag_local / u_au_to_km + planet_center_render;
        vec3 V = ray_dir; // Fix dimensional error: ray_dir is unit vector in km space
        if (integration_quality > 0) {
            if (integration_quality == 1) {
                global_eclipse_shadow = compute_shadow(mid_render, L_mid, dist_mid_star, planet_center_render, star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz);
                end_eclipse_shadow = global_eclipse_shadow;
            }

            if (integration_quality == 2) {
                ring_count = 0;
                uint processed_mask = 0u;
                if (u_num_ring_planes > 0) {
                    for (int k = 0; k < u_num_ring_planes; k++) {
                        if ((u_ring_mask & (1u << k)) == 0u) continue;
                        if ((processed_mask & (1u << k)) != 0u) continue;

                        vec3 N = u_ring_normal[k];
                        vec3 ring_center_local_km = (u_ring_center[k] - planet_center_render) * u_au_to_km;

                        float denom = dot(L_mid, N);
                        if (abs(denom) < 1e-8) continue;

                        float ra_km = dot(ring_center_local_km - frag_local, N) / denom;
                        float rb_km = -dot(ray_dir, N) / denom;

                        vec3 A_prime_km = (frag_local - ring_center_local_km) + ra_km * L_mid;
                        vec3 B_km = ray_dir + rb_km * L_mid;

                        float t_mid_km = max(0.0, ra_km + s_mid * rb_km);
                        float alpha_star = star_radius / max(dist_mid_star, 1e-6);
                        float r_star_proj_km = alpha_star * t_mid_km;

                        vec3 L_plane = L_mid - denom * N;
                        float L_plane_len = length(L_plane);
                        vec3 L_proj = (L_plane_len > 1e-5) ? (L_plane / L_plane_len) : vec3(1.0, 0.0, 0.0);
                        vec3 T_vec = (L_plane_len > 1e-5) ? normalize(cross(N, L_mid)) : vec3(0.0, 1.0, 0.0);
                        float inv_denom_clamped = min(1.0 / max(1e-4, abs(denom)), 20.0);

                        uint coplanar_mask = u_ring_coplanar_mask[k];
                        processed_mask |= coplanar_mask;
                        for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
                            if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
                            if (ring_count >= 4) continue;

                            float inner_r_km = u_ring_params[ring_idx].x * u_au_to_km;
                            float outer_r_km = u_ring_params[ring_idx].y * u_au_to_km;
                            float opacity = u_ring_params[ring_idx].z;
                            if (outer_r_km <= inner_r_km || opacity <= 0.0) continue;
                            float inv_mu = 1.0 / max(1e-4, abs(denom));

                            float ring_s_valid_min = -1e9;
                            float ring_s_valid_max = 1e9;
                            if (abs(rb_km) < 1e-8) {
                                if (ra_km <= 0.0) { ring_s_valid_min = 1e9; ring_s_valid_max = -1e9; }
                            } else {
                                float s_cross = -ra_km / rb_km;
                                if (rb_km < 0.0) {
                                    ring_s_valid_max = s_cross;
                                } else {
                                    ring_s_valid_min = s_cross;
                                }
                            }
                            if (ring_s_valid_min > ring_s_valid_max) continue;

                            ring_s1_out[ring_count] = ring_s_valid_min;
                            ring_s2_out[ring_count] = ring_s_valid_max;
                            ring_inner[ring_count] = inner_r_km;
                            ring_outer[ring_count] = outer_r_km;
                            ring_inv_mu[ring_count] = inv_mu;
                            ring_opacity[ring_count] = opacity;
                            ring_rows[ring_count] = ring_idx;
                            ring_A_prime[ring_count] = A_prime_km;
                            ring_B[ring_count] = B_km;
                            ring_L_proj[ring_count] = L_proj;
                            ring_T_vec[ring_count] = T_vec;
                            ring_r_star_proj[ring_count] = r_star_proj_km;
                            ring_inv_denom[ring_count] = inv_denom_clamped;

                            ring_count++;
                        }
                    }
                }

                for (int c = 0; c < min(u_num_active_casters, 8); ++c) {
                    vec3 D = (u_active_casters[c].xyz - planet_center_render) * u_au_to_km - frag_local;
                    float first = march_start, last = march_end;
                    if (caster_shadow_interval(D, V, L_mid, u_active_casters[c].w * u_au_to_km,
                            star_radius / max(dist_mid_star, 1e-6), first, last)) {
                        has_body_shadow = true;
                        break;
                    }
                }
                if (!has_body_shadow && ring_count == 0) skip_volumetric_shadow = true;
            }
        }

        vec3 ring_normal_vec = pole_dir_norm;
        for (int k = 0; k < u_num_ring_planes; k++) {
            if ((u_ring_mask & (1u << k)) != 0u) {
                ring_normal_vec = u_ring_normal[k];
                break;
            }
        }

        // SpaceEngine Solstice Winter Model (atmosphere thickness is negligible compared to radius)
        vec3 mid_pos_sph = toSphericalSpace(mid_pos, u_pole_obl);
        float has_rings = (u_ring_mask != 0u) ? 1.0 : 0.0;
        float polar_haze_factor;
        vec3 polar_rayleigh_inscatter_boost;
        eval_polar_winter(mid_pos_sph, vec2(solstice_factor, sun_pole_dot), has_rings, u_precomp_mie.w, polar_haze_factor, polar_rayleigh_inscatter_boost);


        vec3 total_rayleigh = vec3(0.0);
        vec3 total_mie = vec3(0.0);
        vec3 total_ms = vec3(0.0);
        vec3 current_transmittance = vec3(1.0);

        vec3 total_rayleigh_ps = vec3(0.0);
        vec3 total_mie_ps = vec3(0.0);
        vec3 total_ms_ps = vec3(0.0);

        vec3 total_rayleigh_rs = vec3(0.0);
        vec3 total_mie_rs = vec3(0.0);
        vec3 total_ms_rs = vec3(0.0);

        float inv_atmo_thickness = 1.0 / max(1e-4, u_atmo_radius_km - u_planet_radius_km);

        for (int i = 0; i < steps; i++) {
            float t0 = float(i) / float(steps);
            float t1 = float(i + 1) / float(steps);

            float s0 = map_t_to_s(t0, march_start, march_end, march_s_closest, march_grade_p);
            float s1 = map_t_to_s(t1, march_start, march_end, march_s_closest, march_grade_p);
            float current_s = mix(s0, s1, jitter);
            if (integration_quality == 1)
                current_s = map_t_to_s((float(i) + jitter) / float(steps),
                    march_start, march_end, march_s_closest, march_grade_p);
            float step_size = max(0.0, s1 - s0);

            vec3 current_pos_sph = cam_local_sph + current_s * ray_dir_sph;
            float sample_len = length(current_pos_sph);
            float altitude = max(0.0, sample_len - u_planet_radius_km);

            float rho_R = exp(-altitude * inv_h_rayleigh);
            float rho_M = exp(-altitude * inv_h_mie) * polar_haze_factor;
            float t_ozone = (altitude - u_ozone_peak_km) * inv_ozone_width;
            float rho_O = exp(-(t_ozone * t_ozone));

            // Extinction uses physical beta_R to prevent artificial limb color fringing
            vec3 step_extinction = beta_R * rho_R + beta_M * rho_M + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;
            vec3 step_transmittance = vec3(1.0);
            vec3 int_factor = vec3(step_size);
            if (integration_quality == 2 && u_atmo_optical_enabled) {
                // This is a stratified integral of the actual attenuated source.
                // Extinction is deterministic; it is not exponentiated from the
                // same random density sample that estimates the light source.
                current_transmittance = atmo_view_transmittance(view_a, current_pos_sph,
                    view_direction, view_initial_tau, view_ground, view_regular);
            } else {
                step_transmittance = exp(-step_extinction * step_size);
                int_factor = (vec3(1.0) - step_transmittance) / max(step_extinction, 1e-6);
            }

            vec3 sun_dir_sph = sun_dir_sph_const;
            float light_cos_theta = dot(current_pos_sph, sun_dir_sph) / sample_len;
            float h_norm = clamp(altitude * inv_atmo_thickness, 0.0, 1.0);

            float sin_planet = u_planet_radius_km / max(sample_len, u_planet_radius_km + 0.01);
            float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));

            // Terminator: stellar-disc rise/set with refraction-extended penumbra
            // (effective_star_rad = sin_star + max_bend), shared with Mode 3.
            vec2 term = sun_terminator(light_cos_theta, sin_planet, cos_planet,
                                       effective_star_rad, cos_sun_eff);
            float vis_fraction = term.x;
            float effective_cos = term.y;

            float v_norm = sqrt(h_norm);
            vec3 transmittance_to_sun = (vis_fraction > 1e-4) ? get_transmittance_precomputed(v_norm, effective_cos) : vec3(0.0);

            vec3 sample_shadow = global_eclipse_shadow;
            if (integration_quality == 2 && !skip_volumetric_shadow) {
                sample_shadow = vec3(1.0);

                // === Mode 2 shared ring transmission ===
                if (ring_count > 0) {
                    vec4 s_vec = vec4(current_s);
                    vec4 mask = step(ring_s1_out, s_vec) * step(s_vec, ring_s2_out);
                    for (int r = 0; r < ring_count; r++) {
                        if (mask[r] <= 0.0) continue;
                        vec3 P_ring = ring_A_prime[r] + current_s * ring_B[r];
                        float d = length(P_ring);
                        vec3 dir_radial = (d > 1e-4) ? (P_ring / d) : ring_L_proj[r];
                        float cos_theta = dot(dir_radial, ring_L_proj[r]);
                        float sin_theta = dot(dir_radial, ring_T_vec[r]);
                        float R_eff = ring_r_star_proj[r] * sqrt(pow(cos_theta * ring_inv_denom[r], 2.0) + sin_theta * sin_theta);
                        R_eff = min(R_eff, (ring_outer[r] - ring_inner[r]) * 0.5);
                        float occlusion = surface_ring_profile_occlusion(ring_rows[r], d,
                            ring_inner[r], ring_outer[r], ring_opacity[r], ring_inv_mu[r], R_eff);
                        sample_shadow *= 1.0 - clamp(occlusion, 0.0, 1.0);
                    }
                }
                // === End Mode 2 shared ring transmission ===

                if (has_body_shadow) {
                    vec3 world_p = planet_center_render + fromSphericalSpace(current_pos_sph, u_pole_obl) / u_au_to_km;
                    sample_shadow *= compute_caster_shadow(world_p, L_mid, dist_mid_star,
                        star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz);
                }
            }

            vec3 sample_attenuation = current_transmittance * transmittance_to_sun * sample_shadow * vis_fraction;

            total_rayleigh += (rho_R * polar_rayleigh_inscatter_boost) * sample_attenuation * int_factor;
            total_mie      += rho_M * sample_attenuation * int_factor;

            float sun_cos_zenith = light_cos_theta;
            float ms_u = 0.5 + 0.5 * sign(sun_cos_zenith) * sqrt(abs(sun_cos_zenith));
            float ms_v = v_norm;
            vec2 ms_uv = vec2(ms_u, ms_v);
            vec3 psi = textureLod(u_multi_scatter_lut, ms_uv, 0.0).rgb;

            vec3 ms_shadow = sample_shadow;
            total_ms += (beta_R * rho_R + beta_M * rho_M) * psi * current_transmittance * ms_shadow * int_factor;

            if (u_planetshine_enabled || u_ringshine_enabled) {
                float light_cos_theta_ps = dot(current_pos_sph, body_planetshine_dir) / sample_len;
                float vis_fraction_ps = smoothstep(-cos_planet - 0.05, -cos_planet + 0.05, light_cos_theta_ps);
                vec3 transmittance_to_ps = get_transmittance_precomputed(v_norm, light_cos_theta_ps);
                vec3 ps_attenuation = current_transmittance * transmittance_to_ps * vis_fraction_ps * sample_shadow;
                total_rayleigh_ps += rho_R * ps_attenuation * int_factor;
                total_mie_ps      += rho_M * ps_attenuation * int_factor;

                float ms_u_ps = 0.5 + 0.5 * sign(light_cos_theta_ps) * sqrt(abs(light_cos_theta_ps));
                vec2 ms_uv_ps = vec2(ms_u_ps, v_norm);
                vec3 psi_ps = textureLod(u_multi_scatter_lut, ms_uv_ps, 0.0).rgb;
                total_ms_ps += (beta_R * rho_R + beta_M * rho_M) * psi_ps * ps_attenuation * int_factor;
            }

            if (u_ringshine_enabled) {
                vec3 rs_attenuation = current_transmittance;
                total_rayleigh_rs += rho_R * rs_attenuation * int_factor;
                total_mie_rs      += rho_M * rs_attenuation * int_factor;

                vec3 rs_pos_sph = normalize(current_pos_sph);
                float rs_elev = dot(rs_pos_sph, ring_normal_vec);
                float ring_cos_zenith = clamp(sqrt(max(0.0, 1.0 - rs_elev * rs_elev)), 0.0, 1.0);
                float ms_u_rs = 0.5 + 0.5 * sqrt(ring_cos_zenith);
                vec2 ms_uv_rs = vec2(ms_u_rs, v_norm);
                vec3 psi_rs = textureLod(u_multi_scatter_lut, ms_uv_rs, 0.0).rgb;
                total_ms_rs += (beta_R * rho_R + beta_M * rho_M) * psi_rs * rs_attenuation * int_factor;
            }

            if (!(integration_quality == 2 && u_atmo_optical_enabled))
                current_transmittance *= step_transmittance;
            if (!(integration_quality == 2 && u_atmo_optical_enabled)
                && all(lessThan(current_transmittance, vec3(1e-6)))) {
                current_transmittance = vec3(0.0);
                break;
            }
        }

        float cos_theta = dot(ray_dir, L_mid);
        float phase_R = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);

        float phase_M_scalar = u_precomp_mie.x * (1.0 + cos_theta * cos_theta)
                               / pow(max(1e-4, u_precomp_mie.y - u_precomp_mie.z * cos_theta), 1.5);
        vec3 phase_M = vec3(phase_M_scalar);

        vec3 star_combined_intensity = (s < 4) ? u_star_color_irrad[s].rgb : vec3(1.0);
        scattered += star_combined_intensity * (
            phase_R * beta_R * total_rayleigh +
            phase_M * beta_M * total_mie +
            total_ms
        );

        if (s == 0 && (u_planetshine_enabled || u_ringshine_enabled) && dot(body_planetshine_color, body_planetshine_color) > 1e-12) {
            float cos_theta_ps = dot(ray_dir, body_planetshine_dir);
            float phase_R_ps = (3.0 / (16.0 * PI)) * (1.0 + cos_theta_ps * cos_theta_ps);
            float phase_M_ps_scalar = u_precomp_mie.x * (1.0 + cos_theta_ps * cos_theta_ps)
                                      / pow(max(1e-4, u_precomp_mie.y - u_precomp_mie.z * cos_theta_ps), 1.5);
            vec3 phase_M_ps = vec3(phase_M_ps_scalar);
            float atmo_sun_intensity = u_sun_intensity * PI;
            scattered += body_planetshine_color * atmo_sun_intensity * (
                phase_R_ps * beta_R * total_rayleigh_ps +
                phase_M_ps * beta_M * total_mie_ps +
                total_ms_ps
            );
        }

        if (u_ringshine_enabled && u_num_ring_planes > 0 && u_ring_mask != 0u) {
            vec3 ringshine_irradiance = vec3(0.0);
            vec3 P_dir = normalize(mid_pos);
            vec3 L_dir = L_mid;

            for (int j = 0; j < u_num_ring_planes; j++) {
                if ((u_ring_mask & (1u << j)) == 0u) continue;
                vec3 ring_normal = u_ring_normal[j];

                vec3 antiL = -L_dir;
                vec3 antiL_eq_raw = antiL - ring_normal * dot(antiL, ring_normal);
                float len_antiL_eq = length(antiL_eq_raw);
                vec3 antiL_eq = len_antiL_eq > 1e-5 ? antiL_eq_raw / len_antiL_eq : vec3(-1.0, 0.0, 0.0);

                vec3 P_eq_raw = P_dir - ring_normal * dot(P_dir, ring_normal);
                float len_P_eq = length(P_eq_raw);
                vec3 P_eq = len_P_eq > 1e-5 ? P_eq_raw / len_P_eq : vec3(1.0, 0.0, 0.0);

                vec3 cross_rel = cross(antiL_eq, P_eq);
                float sin_rel = dot(cross_rel, ring_normal);
                float cos_rel = clamp(dot(P_eq, antiL_eq), -1.0, 1.0);
                float phi_center = atan(sin_rel, cos_rel);

                float frag_elevation = dot(P_dir, ring_normal);

                float x_prime = phi_center / PI;
                float phi_uv = sign(x_prime) * pow(abs(x_prime), 0.666666667) * 0.5 + 0.5;

                float y_prime = frag_elevation;
                float elev_uv = sign(y_prime) * pow(abs(y_prime), 0.666666667) * 0.5 + 0.5;

                vec2 map_uv = vec2(phi_uv, (float(j) + elev_uv) / 16.0);
                ringshine_irradiance += ringshine_sample(map_uv, j, s);
            }
            // Apply 1/PI (~0.318309886) factor to convert incoming irradiance map to ambient field.
            // Note: Solar elevation (mu_0) is already fully resolved inside ringshine_map.frag radiative transfer.
            ringshine_irradiance *= 0.318309886;

            float ambient_phase = 1.0 / (4.0 * PI);
            float ambient_phase_M = ambient_phase * (1.0 / max(0.15, 1.0 - u_precomp_mie.z * 0.5));
            scattered += star_combined_intensity * ringshine_irradiance * (
                ambient_phase * beta_R * total_rayleigh_rs +
                ambient_phase_M * beta_M * total_mie_rs +
                total_ms_rs
            );
        }

        final_transmittance = (integration_quality == 2 && u_atmo_optical_enabled)
            ? high_transmittance
            : current_transmittance;
    }
    }

    if (u_atmo_quality == 3 && u_scattering_enabled && has_shadow) {
        vec3 deficit = march_shadow_deficit(march_start, march_end, s_start, s_end,
            cam_local_sph, ray_dir_sph, ray_dir, max(1, steps), jitter, star_baseline);
        scattered = max(vec3(0.0), scattered - deficit);
    }

    vec3 transmittance = final_transmittance;

    if (u_hdr_enabled) {
        scattered *= u_exposure;
    }

    if (integration_quality != 3 && u_temporal_accum && u_history_valid) {
        float d_repr_km;
        if (hits_surface) {
            d_repr_km = max(s_end, 0.001);
        } else {
            float cam_r = length(cam_local_sph);
            float cos_zenith = dot(cam_local_sph, ray_dir_sph) / max(cam_r, 1e-3);
            float sin_elev = max(0.05, cos_zenith);
            float eff_h = u_h_rayleigh / sin_elev;
            float chord = max(0.0, s_end - s_start);
            float offset = clamp(max(s_closest - s_start, eff_h), chord * 0.05, chord * 0.75);
            d_repr_km = s_start + offset;
        }
        float d_repr_au = d_repr_km / u_au_to_km;

        vec3 p_repr_world = u_camera_pos + d_repr_au * ray_dir;
        vec3 p_body_local = p_repr_world - u_body_offset;
        vec3 p_prev_world = p_body_local + u_prev_body_offset;

        vec4 clip_prev = u_prev_view_proj * vec4(p_prev_world, 1.0);
        if (clip_prev.w > 1e-13) {
            vec2 uv_prev = (clip_prev.xy / clip_prev.w) * 0.5 + 0.5;
            if (uv_prev.x >= 0.0 && uv_prev.x <= 1.0 && uv_prev.y >= 0.0 && uv_prev.y <= 1.0) {
                vec4 hist_s = texture(u_history_scatter, uv_prev);
                vec4 hist_t = texture(u_history_trans, uv_prev);

                if (hist_t.a > 0.001) {
                    float exp_ratio = (u_prev_exposure > 1e-6) ? (u_exposure / u_prev_exposure) : 1.0;
                    vec3 hist_s_curr = hist_s.rgb * exp_ratio;

                    // Range clamping against current frame raymarch result to prevent ghosting
                    vec3 s_min = max(vec3(0.0), scattered * 0.35 - 0.02);
                    vec3 s_max = scattered * 2.5 + 0.05;
                    vec3 clamped_hist_s = clamp(hist_s_curr, s_min, s_max);

                    vec3 t_min = max(vec3(0.0), transmittance * 0.6 - 0.02);
                    vec3 t_max = min(vec3(1.0), transmittance * 1.4 + 0.02);
                    vec3 clamped_hist_t = clamp(hist_t.rgb, t_min, t_max);

                    scattered = mix(clamped_hist_s, scattered, u_temporal_alpha);
                    transmittance = mix(clamped_hist_t, transmittance, u_temporal_alpha);
                }
            }
        }
    }

    out_scattered = vec4(scattered, 1.0);
    out_transmittance = vec4(clamp(transmittance, vec3(0.0), vec3(1.0)), 1.0);
}
