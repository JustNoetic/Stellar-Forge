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
    vec4  u_precomp_mie;         // x: c1, y: c2, z: c3, w: unused
    vec4  u_star_color_irrad[4]; // rgb = star_color * irradiance * atmo_sun_intensity, w = sin_star
    vec4  u_star_dir_sph_eff[4]; // xyz = sun_dir_sph_const, w = cos_sun_eff
    vec4  u_star_pos_local[4];   // xyz = sun_pos_local_km, w = effective_star_rad
    vec4  u_star_solstice[4];    // x = solstice_factor, y = sun_pole_dot, z = dist_star_au, w = star_radius_au
};

uniform sampler2D u_ring_gradients;
uniform sampler2D u_ring_shadow_tex;
uniform int u_num_ring_planes;
uniform int u_atmo_quality;
uniform bool u_stochastic_noise;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec4 u_ring_params[MAX_RING_PLANES];
uniform int u_atmo_clip_mode;
uniform vec4 u_ring_station_cells[9];   // 33 station-cell boundaries in u space (Kmax = 32), packed 4 per vec4
uniform int u_ring_station_count;      // number of cells (boundaries = count + 1)
uniform int u_atmo_shadow_method;      // 0 = Station-Locked Slicing, 1 = Uniform Stochastic Raymarching, 2 = Bounded Subtraction (Blackrack)

uniform uint u_ring_coplanar_mask[16];
uniform sampler2D u_eclipse_lut; // Kept to avoid uniform bound errors
uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform bool u_planetshine_enabled;

uniform sampler3D u_ringshine_lut;
uniform sampler3D u_ringshine_cdf_lut;
uniform sampler2D u_ringshine_map;
uniform bool u_ringshine_enabled;
uniform int u_ringshine_band_count;
uniform sampler2D u_depth_texture;
uniform vec2 u_screen_res;
uniform bool u_terrain_depth_enabled;
uniform bool u_aerial_volume_enabled;
uniform sampler3D u_aerial_scatter_lut;
uniform sampler3D u_aerial_trans_lut;
uniform vec3 u_aerial_cam_sph;
uniform vec2 u_aerial_radius_range;
#include "common/aerial_volume_mapping.glsl"


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

float eval_ringshine_cdf(float angle, float v_tex, float sin_lat) {
    float TWO_PI = 6.28318530717958647692;
    float a_mod = angle - TWO_PI * floor((angle + PI) / TWO_PI);
    float k = floor((angle + PI) / TWO_PI);
    float u = clamp(abs(a_mod) / PI, 0.0, 1.0);

    float u_tex = 0.5 / 128.0 + u * (127.0 / 128.0);
    float v_tex_mapped = 0.5 / 128.0 + v_tex * (127.0 / 128.0);
    float sin_lat_mapped = 0.5 / 64.0 + sin_lat * (63.0 / 64.0);

    float base_cdf = texture(u_ringshine_cdf_lut, vec3(u_tex, v_tex_mapped, sin_lat_mapped)).r;
    float signed_cdf = (a_mod < 0.0) ? -base_cdf : base_cdf;
    return 2.0 * k + signed_cdf;
}
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

// SpaceEngine Solstice Winter Model: suppressed aerosol Mie haze and azure Rayleigh in-scatter boost
void eval_polar_winter(vec3 pos_sph, vec2 solstice_params, float has_rings,
                       out float polar_haze_factor, out vec3 polar_rayleigh_boost) {
    if (solstice_params.x < 1e-4 || has_rings < 0.5) {
        polar_haze_factor = 1.0;
        polar_rayleigh_boost = vec3(1.0);
        return;
    }
    vec3 pole_dir_norm = (length(u_pole_obl.xyz) > 1e-4) ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    vec3 dir_norm = normalize(pos_sph);
    float frag_pole_dot = dot(dir_norm, pole_dir_norm);
    float mid_sin_lat = abs(frag_pole_dot);
    float lat_factor = smoothstep(0.1, 0.7, mid_sin_lat);
    float is_winter = step(solstice_params.y * frag_pole_dot, 0.0);
    float winter_solstice_effect = lat_factor * is_winter * solstice_params.x * has_rings;
    polar_haze_factor = mix(1.0, 0.05, winter_solstice_effect);
    polar_rayleigh_boost = mix(vec3(1.0), vec3(0.65, 0.95, 2.5), winter_solstice_effect);
}

#define ATMO_DATA_HAS_AU_TO_KM 1
#include "common/refraction.glsl"

#include "common/sun_terminator.glsl"
// get_oblate_radius / casterShadowTerm / compute_caster_shadow live in the
// shared include so the Mode 3 Sky-View LUT bake uses identical eclipse math.
#include "common/caster_shadow.glsl"
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

            float overlap_min = max(inner_r, d - R_eff);
            float overlap_max = min(outer_r, d + R_eff);

            if (overlap_min >= overlap_max) continue;

            float v_min = clamp((overlap_min - d) / max(1e-9, R_eff), -1.0, 1.0);
            float v_max = clamp((overlap_max - d) / max(1e-9, R_eff), -1.0, 1.0);

            float f_max = (v_max * sqrt(max(0.0, 1.0 - v_max*v_max)) + asin(v_max)) / PI + 0.5;
            float f_min = (v_min * sqrt(max(0.0, 1.0 - v_min*v_min)) + asin(v_min)) / PI + 0.5;
            float fraction = max(0.0, f_max - f_min);

            float p_mid = ((overlap_min + overlap_max) * 0.5 - inner_r) / max(1e-6, outer_r - inner_r);
            float alpha_mult = textureLod(u_ring_gradients, vec2(p_mid, (float(ring_idx) + 0.5) / 16.0), 0.0).a;
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

// --- Mode 3 station-locked ring-shadow quadrature helpers ---
// Cell boundaries are baked on the CPU in ring coordinate u (see bake_station_cells in
// render_utils.py) and packed 4 floats per vec4: index i -> element (i/4, i%4).
float station_bound(int i) {
    vec4 v = u_ring_station_cells[i >> 2];
    int r = i & 3;
    return (r == 0) ? v.x : ((r == 1) ? v.y : ((r == 2) ? v.z : v.w));
}

// Inverse of the chord radius parabola r^2(s) = qa*s^2 + qb*s + qc on one monotone branch.
// sig = -1 selects the s < s_apex branch, +1 the s > s_apex branch.
float station_s_of_u(float u, int sig, float qa, float qb, float qc, float r_inner, float ring_span) {
    float r_t = r_inner + u * ring_span;
    float disc_t = qb * qb - 4.0 * qa * (qc - r_t * r_t);
    return (-qb + float(sig) * sqrt(max(disc_t, 0.0))) / (2.0 * qa);
}

// Accumulate one station cell of the Mode 3 volumetric ring shadow: Simpson-integrated
// in-scatter over the cell's s-extent with one footprint-filtered ring opacity sample over
// the cell's u-window, weighted by the running transmittance T_run.
void accumulate_shadow_cell(
    float s0, float s1, float u_mid, float du,
    float inv_sin_sun,
    vec3 cam_local_sph, vec3 ray_dir_sph, vec3 L_sun,
    float phase_R, float phase_M,
    float eff_star_rad, float cos_sun_eff, vec3 star_int,
    vec3 beta_R, vec3 beta_M, vec3 beta_M_ext, vec3 beta_A_mixed, vec3 beta_A_layered,
    float inv_h_rayleigh, float inv_h_mie, float inv_ozone_width,
    inout vec3 T_run, inout vec3 delta_L, inout vec3 slice_total,
    inout float total_blocked, inout float step_count)
{
    float dsc = s1 - s0;
    float sm = 0.5 * (s0 + s1);

    // Simpson 3-point density sampling over the cell's s-extent
    vec3 P0 = cam_local_sph + s0 * ray_dir_sph;
    vec3 Pm = cam_local_sph + sm * ray_dir_sph;
    vec3 P1 = cam_local_sph + s1 * ray_dir_sph;
    float h0 = max(0.0, length(P0) - u_planet_radius_km);
    float hm = max(0.0, length(Pm) - u_planet_radius_km);
    float h1 = max(0.0, length(P1) - u_planet_radius_km);
    float tO3_0 = (h0 - u_ozone_peak_km) * inv_ozone_width;
    float tO3_m = (hm - u_ozone_peak_km) * inv_ozone_width;
    float polar_haze_factor;
    vec3 polar_rayleigh_boost;
    float has_rings = (u_ring_mask != 0u) ? 1.0 : 0.0;
    eval_polar_winter(Pm, u_cur_solstice, has_rings, polar_haze_factor, polar_rayleigh_boost);

    float tO3_1 = (h1 - u_ozone_peak_km) * inv_ozone_width;
    float rho_R_q  = (exp(-h0 * inv_h_rayleigh) + 4.0 * exp(-hm * inv_h_rayleigh) + exp(-h1 * inv_h_rayleigh)) / 6.0;
    float rho_M_q  = ((exp(-h0 * inv_h_mie) + 4.0 * exp(-hm * inv_h_mie) + exp(-h1 * inv_h_mie)) / 6.0) * polar_haze_factor;
    float rho_O3_q = (exp(-(tO3_0 * tO3_0)) + 4.0 * exp(-(tO3_m * tO3_m)) + exp(-(tO3_1 * tO3_1))) / 6.0;

    vec3 ext_q = beta_R * rho_R_q + beta_M_ext * rho_M_q + beta_A_mixed * rho_R_q + beta_A_layered * rho_O3_q;
    vec3 T_cell = exp(-ext_q * dsc);
    vec3 int_factor = (vec3(1.0) - T_cell) / max(ext_q, vec3(1e-6));

    // Footprint-filtered ring opacity over the cell's u-window.
    // LOD is clamped to [0.0, 3.0] to prevent downsampled box-filtering from bleeding
    // the 100% opaque main rings into faint outer rings (like Saturn's E-ring).
    float ring_blocked = 0.0;
    if (u_mid >= -0.05 && u_mid <= 1.05) {
        float lod = clamp(log2(max(1.0, du * 4096.0)), 0.0, 3.0);
        float raw_a = textureLod(u_ring_shadow_tex, vec2(clamp(u_mid, 0.0, 1.0), 0.5), lod).a;
        float tau_ring = -log(max(1e-4, 1.0 - raw_a));
        ring_blocked = 1.0 - exp(-tau_ring * inv_sin_sun);
    }
    // Note: Do not zero ring_blocked on the night side; if the parcel is in the shadow cone,
    // multi-scattered light accumulated by the LUT must still be subtracted.

    // Direct in-scatter at the cell midpoint (matching sky_view_lut.frag single scattering)
    float rq = max(length(Pm), 1e-6);
    float light_cos_theta = dot(Pm, L_sun) / rq;
    float sin_planet = u_planet_radius_km / max(rq, u_planet_radius_km + 0.01);
    float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));
    // Terminator matching sky_view_lut.frag / Mode 1-2: stellar-disc rise/set with
    // refraction-extended penumbra (eff_star_rad = sin_star + max_bend).
    vec2 term_q = sun_terminator(light_cos_theta, sin_planet, cos_planet,
                                 eff_star_rad, cos_sun_eff);
    float vis_fraction = term_q.x;
    vec3 trans_to_sun = (vis_fraction > 1e-4) ? get_transmittance(rq, term_q.y) : vec3(0.0);
    vec3 inscatter_direct = (beta_R * (rho_R_q * polar_rayleigh_boost) * phase_R + beta_M * rho_M_q * phase_M)
                             * trans_to_sun * vis_fraction * star_int;

    // Multi-scatter component matching sky_view_lut.frag
    float h_norm_q = clamp(hm / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float ms_u_q = 0.5 + 0.5 * sign(light_cos_theta) * sqrt(abs(light_cos_theta));
    float ms_v_q = sqrt(h_norm_q);
    vec3 psi_q = textureLod(u_multi_scatter_lut, vec2(ms_u_q, ms_v_q), 0.0).rgb;
    vec3 inscatter_ms = star_int * (beta_R * rho_R_q + beta_M * rho_M_q) * psi_q;

    vec3 step_inscatter = (inscatter_direct + inscatter_ms) * T_run * int_factor;
    delta_L += step_inscatter * ring_blocked;
    slice_total += step_inscatter;
    total_blocked += ring_blocked;
    step_count += 1.0;
    T_run *= T_cell;
}

// March uniform ray steps with optional stochastic jitter across the shadow interval [cs, ce].
// Evaluates ring optical depth directly across the steps without aggressive MIP LOD blurring,
// and evaluates atmospheric in-scattering at the jittered parcel to eliminate onion banding.
void march_uniform_shadow(
    float cs, float ce, float qa, float qb, float qc, float r_inner, float ring_span, float inv_sin_sun,
    vec3 cam_local_sph, vec3 ray_dir_sph, vec3 L_sun,
    float phase_R, float phase_M,
    float eff_star_rad, float cos_sun_eff, vec3 star_int,
    vec3 beta_R, vec3 beta_M, vec3 beta_M_ext, vec3 beta_A_mixed, vec3 beta_A_layered,
    float inv_h_rayleigh, float inv_h_mie, float inv_ozone_width,
    float jitter, int steps,
    inout vec3 T_run, inout vec3 delta_L, inout vec3 slice_total,
    inout float total_blocked, inout float step_count)
{
    float ds = (ce - cs) / float(steps);
    for (int q = 0; q < steps; q++) {
        float sq = cs + (float(q) + jitter) * ds;

        vec3 Pq = cam_local_sph + sq * ray_dir_sph;
        float rq = max(length(Pq), 1e-6);
        float hq = max(0.0, rq - u_planet_radius_km);

        float polar_haze_factor;
        vec3 polar_rayleigh_boost;
        float has_rings = (u_ring_mask != 0u) ? 1.0 : 0.0;
        eval_polar_winter(Pq, u_cur_solstice, has_rings, polar_haze_factor, polar_rayleigh_boost);

        float rho_R = exp(-hq * inv_h_rayleigh);
        float rho_M = exp(-hq * inv_h_mie) * polar_haze_factor;
        float tO3 = (hq - u_ozone_peak_km) * inv_ozone_width;
        float rho_O3 = exp(-(tO3 * tO3));

        vec3 ext_q = beta_R * rho_R + beta_M_ext * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O3;
        vec3 T_cell = exp(-ext_q * ds);
        vec3 int_factor = (vec3(1.0) - T_cell) / max(ext_q, vec3(1e-6));

        // Direct in-scatter at jittered parcel Pq
        float light_cos_theta = dot(Pq, L_sun) / rq;
        float sin_planet = u_planet_radius_km / max(rq, u_planet_radius_km + 0.01);
        float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));
        vec2 term_q = sun_terminator(light_cos_theta, sin_planet, cos_planet,
                                     eff_star_rad, cos_sun_eff);
        float vis_fraction = term_q.x;
        vec3 trans_to_sun = (vis_fraction > 1e-4) ? get_transmittance(rq, term_q.y) : vec3(0.0);
        vec3 inscatter_direct = (beta_R * (rho_R * polar_rayleigh_boost) * phase_R + beta_M * rho_M * phase_M)
                                 * trans_to_sun * vis_fraction * star_int;

        // Multiple scattering component matching sky_view_lut.frag
        float h_norm_q = clamp(hq / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
        float ms_u_q = 0.5 + 0.5 * sign(light_cos_theta) * sqrt(abs(light_cos_theta));
        float ms_v_q = sqrt(h_norm_q);
        vec3 psi_q = textureLod(u_multi_scatter_lut, vec2(ms_u_q, ms_v_q), 0.0).rgb;
        vec3 inscatter_ms = star_int * (beta_R * rho_R + beta_M * rho_M) * psi_q;

        vec3 step_inscatter = (inscatter_direct + inscatter_ms) * T_run * int_factor;

        // Footprint-filtered ring opacity at jittered ring coordinate u_q
        float r_sq = (qa > 1e-6) ? sqrt(max(0.0, qa * sq * sq + qb * sq + qc)) : sqrt(max(0.0, qc));
        float u_q = (r_sq - r_inner) / ring_span;

        float ring_blocked = 0.0;
        if (u_q >= -0.05 && u_q <= 1.05) {
            float s0 = cs + float(q) * ds;
            float s1 = cs + float(q + 1) * ds;
            float r_s0 = (qa > 1e-6) ? sqrt(max(0.0, qa * s0 * s0 + qb * s0 + qc)) : sqrt(max(0.0, qc));
            float r_s1 = (qa > 1e-6) ? sqrt(max(0.0, qa * s1 * s1 + qb * s1 + qc)) : sqrt(max(0.0, qc));
            float du_q = u_stochastic_noise ? 0.0001 : (abs(r_s1 - r_s0) / ring_span);
            float lod = clamp(log2(max(1.0, du_q * 4096.0)), 0.0, 3.0);
            float raw_a = textureLod(u_ring_shadow_tex, vec2(clamp(u_q, 0.0, 1.0), 0.5), lod).a;
            float tau_ring = -log(max(1e-4, 1.0 - raw_a));
            ring_blocked = 1.0 - exp(-tau_ring * inv_sin_sun);
        }

        delta_L += step_inscatter * ring_blocked;
        slice_total += step_inscatter;
        total_blocked += ring_blocked;
        step_count += 1.0;
        T_run *= T_cell;
    }
}

// Walk station cells across one monotone-u branch of the shadow interval [cs, ce].
// Cells are clipped to the interval's u-range and converted back to s via the branch
// inverse; T_run is marched in ascending s (ascending u on the sig = +1 branch,
// descending u on the sig = -1 branch).
void walk_station_branch(
    float cs, float ce, int sig,
    float qa, float qb, float qc, float r_inner, float ring_span, float inv_sin_sun,
    vec3 cam_local_sph, vec3 ray_dir_sph, vec3 L_sun,
    float phase_R, float phase_M,
    float eff_star_rad, float cos_sun_eff, vec3 star_int,
    vec3 beta_R, vec3 beta_M, vec3 beta_M_ext, vec3 beta_A_mixed, vec3 beta_A_layered,
    float inv_h_rayleigh, float inv_h_mie, float inv_ozone_width,
    inout vec3 T_run, inout vec3 delta_L, inout vec3 slice_total,
    inout float total_blocked, inout float step_count)
{
    float u_cs = (sqrt(max(qa * cs * cs + qb * cs + qc, 0.0)) - r_inner) / ring_span;
    float u_ce = (sqrt(max(qa * ce * ce + qb * ce + qc, 0.0)) - r_inner) / ring_span;
    float u_lo = clamp(min(u_cs, u_ce), 0.0, 1.0);
    float u_hi = clamp(max(u_cs, u_ce), 0.0, 1.0);
    if (u_hi <= u_lo) return;

    int K = clamp(u_ring_station_count, 2, 32);

    // First cell whose upper bound exceeds u_lo.
    int j0 = K - 1;
    for (int j = 0; j < K; j++) {
        if (station_bound(j + 1) > u_lo) { j0 = j; break; }
    }
    // Last cell whose lower bound is below u_hi.
    int j1 = 0;
    for (int j = K - 1; j >= 0; j--) {
        if (station_bound(j) < u_hi) { j1 = j; break; }
    }

    for (int t = 0; t <= j1 - j0; t++) {
        int j = (sig > 0) ? (j0 + t) : (j1 - t);
        float u_a = max(station_bound(j), u_lo);
        float u_b = min(station_bound(j + 1), u_hi);
        if (u_b <= u_a) continue;

        float s_a = station_s_of_u(u_a, sig, qa, qb, qc, r_inner, ring_span);
        float s_b = station_s_of_u(u_b, sig, qa, qb, qc, r_inner, ring_span);
        float s0 = clamp(min(s_a, s_b), cs, ce);
        float s1 = clamp(max(s_a, s_b), cs, ce);
        if (s1 <= s0 + 1e-7) continue;

        accumulate_shadow_cell(s0, s1, 0.5 * (u_a + u_b), u_b - u_a, inv_sin_sun,
                               cam_local_sph, ray_dir_sph, L_sun, phase_R, phase_M,
                               eff_star_rad, cos_sun_eff, star_int,
                               beta_R, beta_M, beta_M_ext, beta_A_mixed, beta_A_layered,
                               inv_h_rayleigh, inv_h_mie, inv_ozone_width,
                               T_run, delta_L, slice_total, total_blocked, step_count);
    }
}

// Path transmittance along an atmospheric ray segment [s0, s1] (km).
// Evaluated with an 8-step deterministic midpoint integration along the line of sight.
// Does not query the 2D Transmittance LUT (avoiding ground-clip horizon zero-boundaries,
// division-by-zero artifacts, and chromatic fringes), while taking sufficiently small steps
// (ds << scale height H) to eliminate Simpson quadrature error oscillations.
vec3 get_transmittance_segment(
    float s0, float s1,
    vec3 cam_local_sph, vec3 ray_dir_sph,
    vec3 beta_R, vec3 beta_M_ext, vec3 beta_A_mixed, vec3 beta_A_layered,
    float inv_h_rayleigh, float inv_h_mie, float inv_ozone_width)
{
    if (s1 <= s0 + 1e-4) return vec3(1.0);

    float polar_haze_factor = 1.0;
    vec3 dummy_boost;
    float has_rings = (u_ring_mask != 0u) ? 1.0 : 0.0;
    eval_polar_winter(cam_local_sph + 0.5 * (s0 + s1) * ray_dir_sph, u_cur_solstice, has_rings, polar_haze_factor, dummy_boost);

    const int K = 24;
    float ds = (s1 - s0) / float(K);
    vec3 tau = vec3(0.0);

    for (int i = 0; i < K; i++) {
        float sq = s0 + (float(i) + 0.5) * ds;
        vec3 Pq = cam_local_sph + sq * ray_dir_sph;
        float rq = max(length(Pq), 1e-6);
        float hq = max(0.0, rq - u_planet_radius_km);

        float rho_R = exp(-hq * inv_h_rayleigh);
        float rho_M = exp(-hq * inv_h_mie) * polar_haze_factor;
        float tO3 = (hq - u_ozone_peak_km) * inv_ozone_width;
        float rho_O3 = exp(-(tO3 * tO3));

        vec3 ext_q = (beta_R + beta_A_mixed) * rho_R + beta_M_ext * rho_M + beta_A_layered * rho_O3;
        tau += ext_q * ds;
    }

    return exp(-tau);
}

// Method 2 (Blackrack): Deterministic midpoint raymarching across bounded ring shadow interval [cs, ce].
// Uses cross-step derivative + screen-space LOD for ring anti-aliasing, Mie phase anti-halo scaling (0.95),
// smooth interpolated LOD along the chord (matching KSA), and relative deficit normalization.
void march_bounded_shadow_blackrack(
    float cs, float ce, float du_screen,
    float qa, float qb, float qc, float r_inner, float ring_span, float inv_sin_sun,
    vec3 cam_local_sph, vec3 ray_dir_sph, vec3 L_sun,
    float phase_R, float phase_M,
    float eff_star_rad, float cos_sun_eff, vec3 star_int,
    vec3 beta_R, vec3 beta_M, vec3 beta_M_ext, vec3 beta_A_mixed, vec3 beta_A_layered,
    float inv_h_rayleigh, float inv_h_mie, float inv_ozone_width,
    int steps,
    inout vec3 T_run, inout vec3 delta_L, inout vec3 slice_total)
{
    float ds = (ce - cs) / float(steps);
    float phase_M_eff = phase_M;

    // Precompute analytical radial step derivatives at chord entry and exit (matching KSA).
    // The radial change rate dr/ds = (qa * s + 0.5 * qb) / r(s).
    float r_cs = (qa > 1e-6) ? sqrt(max(0.0, qa * cs * cs + qb * cs + qc)) : sqrt(max(0.0, qc));
    float r_ce = (qa > 1e-6) ? sqrt(max(0.0, qa * ce * ce + qb * ce + qc)) : sqrt(max(0.0, qc));
    float dr_ds_cs = (qa > 1e-6 && r_cs > 1e-4) ? abs(qa * cs + 0.5 * qb) / r_cs : 0.0;
    float dr_ds_ce = (qa > 1e-6 && r_ce > 1e-4) ? abs(qa * ce + 0.5 * qb) / r_ce : 0.0;

    float du_step_cs = 2.0 * ds * dr_ds_cs / max(ring_span, 1e-4);
    float du_step_ce = 2.0 * ds * dr_ds_ce / max(ring_span, 1e-4);
    float lod_start = clamp(log2(max(1.0, max(du_step_cs, du_screen) * 4096.0)), 0.0, 8.0);
    float lod_end = clamp(log2(max(1.0, max(du_step_ce, du_screen) * 4096.0)), 0.0, 8.0);
    float eff_inv_sin_sun = min(inv_sin_sun, 5.0);

    for (int q = 0; q < steps; q++) {
        // Deterministic midpoint quadrature: zero jitter, zero temporal noise
        float sq = cs + (float(q) + 0.5) * ds;

        vec3 Pq = cam_local_sph + sq * ray_dir_sph;
        float rq = max(length(Pq), 1e-6);
        float hq = max(0.0, rq - u_planet_radius_km);

        float polar_haze_factor;
        vec3 polar_rayleigh_boost;
        float has_rings = (u_ring_mask != 0u) ? 1.0 : 0.0;
        eval_polar_winter(Pq, u_cur_solstice, has_rings, polar_haze_factor, polar_rayleigh_boost);

        float rho_R = exp(-hq * inv_h_rayleigh);
        float rho_M = exp(-hq * inv_h_mie) * polar_haze_factor;
        float tO3 = (hq - u_ozone_peak_km) * inv_ozone_width;
        float rho_O3 = exp(-(tO3 * tO3));

        vec3 ext_q = beta_R * rho_R + beta_M_ext * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O3;
        vec3 T_cell = exp(-ext_q * ds);
        vec3 int_factor = (vec3(1.0) - T_cell) / max(ext_q, vec3(1e-6));

        // Direct in-scatter at parcel Pq
        float light_cos_theta = dot(Pq, L_sun) / rq;
        float sin_planet = u_planet_radius_km / max(rq, u_planet_radius_km + 0.01);
        float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));
        vec2 term_q = sun_terminator(light_cos_theta, sin_planet, cos_planet,
                                     eff_star_rad, cos_sun_eff);
        float vis_fraction = term_q.x;
        vec3 trans_to_sun = (vis_fraction > 1e-4) ? get_transmittance(rq, term_q.y) : vec3(0.0);
        vec3 inscatter_direct = (beta_R * (rho_R * polar_rayleigh_boost) * phase_R + beta_M * rho_M * phase_M_eff)
                                 * trans_to_sun * vis_fraction * star_int;

        // Multiple scattering component matching sky_view_lut.frag
        float h_norm_q = clamp(hq / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
        float ms_u_q = 0.5 + 0.5 * sign(light_cos_theta) * sqrt(abs(light_cos_theta));
        float ms_v_q = sqrt(h_norm_q);
        vec3 psi_q = textureLod(u_multi_scatter_lut, vec2(ms_u_q, ms_v_q), 0.0).rgb;
        vec3 inscatter_ms = star_int * (beta_R * rho_R + beta_M * rho_M) * psi_q;

        // Smoothly interpolate LOD along the chord (matching KSA)
        float t_seg = (float(q) + 0.5) / float(steps);
        float lod = mix(lod_start, lod_end, t_seg);

        // Ring opacity at radial coordinate u_q
        float r_sq = (qa > 1e-6) ? sqrt(max(0.0, qa * sq * sq + qb * sq + qc)) : sqrt(max(0.0, qc));
        float u_q = (r_sq - r_inner) / ring_span;

        float ring_blocked = 0.0;
        if (u_q >= 0.0 && u_q <= 1.0) {
            float raw_a = textureLod(u_ring_shadow_tex, vec2(u_q, 0.5), lod).a;
            float tau_ring = -log(max(1e-4, 1.0 - raw_a));
            ring_blocked = 1.0 - exp(-tau_ring * eff_inv_sin_sun);
        }

        vec3 step_unshadowed = (inscatter_direct + inscatter_ms) * T_run * int_factor;
        slice_total += step_unshadowed;
        delta_L += step_unshadowed * ring_blocked;
        T_run *= T_cell;
    }
}

// Analytical cylinder intersection bounding celestial eclipse casters (Blackrack approach)
vec2 get_eclipse_cylinder_bounds(vec3 ray_origin, vec3 ray_dir, vec3 caster_pos, vec3 cone_axis, float cylinder_radius) {
    vec3 origin_to_base = ray_origin - caster_pos;
    float axis_dot_dir = dot(cone_axis, ray_dir);
    float axis_dot_orig = dot(cone_axis, origin_to_base);

    vec3 d_perp = ray_dir - axis_dot_dir * cone_axis;
    vec3 o_perp = origin_to_base - axis_dot_orig * cone_axis;

    float a = dot(d_perp, d_perp);
    float b = dot(o_perp, d_perp);
    float c = dot(o_perp, o_perp) - cylinder_radius * cylinder_radius;

    if (a < 1e-8) {
        if (c <= 0.0) return vec2(-1e9, 1e9);
        return vec2(-1.0);
    }

    float disc = b * b - a * c;
    if (disc < 0.0) return vec2(-1.0);

    float sq = sqrt(disc);
    float t1 = (-b - sq) / a;
    float t2 = (-b + sq) / a;
    return vec2(min(t1, t2), max(t1, t2));
}

// Method 2 (Blackrack): Raymarch bounded celestial shadow cone segment [cs, ce] for an eclipse caster
void march_bounded_eclipse_blackrack(
    float cs, float ce, float s_start,
    vec3 cam_local, vec3 ray_dir, vec3 cam_local_sph, vec3 ray_dir_sph,
    vec3 planet_center_render, vec3 L_cart, vec3 L_sun,
    float phase_R, float phase_M,
    float eff_star_rad, float cos_sun_eff, vec3 star_int,
    float dist_to_star_au, float star_radius_au, float star_obl, vec3 star_pole,
    int caster_idx,
    vec3 beta_R, vec3 beta_M, vec3 beta_M_ext, vec3 beta_A_mixed, vec3 beta_A_layered,
    float inv_h_rayleigh, float inv_h_mie, float inv_ozone_width,
    int steps,
    inout vec3 delta_L, inout vec3 slice_total)
{
    float ds = (ce - cs) / float(steps);
    float phase_M_eff = phase_M;

    // Initialize transmittance from atmosphere entry (s_start) to shadow entry (cs)
    vec3 T_run = get_transmittance_segment(s_start, cs, cam_local_sph, ray_dir_sph,
                                           beta_R, beta_M_ext, beta_A_mixed, beta_A_layered,
                                           inv_h_rayleigh, inv_h_mie, inv_ozone_width);

    uint caster_mask = (1u << caster_idx);

    for (int q = 0; q < steps; q++) {
        float sq = cs + (float(q) + 0.5) * ds;

        vec3 Pq = cam_local_sph + sq * ray_dir_sph;
        float rq = max(length(Pq), 1e-6);
        float hq = max(0.0, rq - u_planet_radius_km);

        float polar_haze_factor;
        vec3 polar_rayleigh_boost;
        float has_rings = (u_ring_mask != 0u) ? 1.0 : 0.0;
        eval_polar_winter(Pq, u_cur_solstice, has_rings, polar_haze_factor, polar_rayleigh_boost);

        float rho_R = exp(-hq * inv_h_rayleigh);
        float rho_M = exp(-hq * inv_h_mie) * polar_haze_factor;
        float tO3 = (hq - u_ozone_peak_km) * inv_ozone_width;
        float rho_O3 = exp(-(tO3 * tO3));

        vec3 ext_q = beta_R * rho_R + beta_M_ext * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O3;
        vec3 T_cell = exp(-ext_q * ds);
        vec3 int_factor = (vec3(1.0) - T_cell) / max(ext_q, vec3(1e-6));

        // Direct in-scatter at parcel Pq
        float light_cos_theta = dot(Pq, L_sun) / rq;
        float sin_planet = u_planet_radius_km / max(rq, u_planet_radius_km + 0.01);
        float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));
        vec2 term_q = sun_terminator(light_cos_theta, sin_planet, cos_planet,
                                     eff_star_rad, cos_sun_eff);
        float vis_fraction = term_q.x;
        vec3 trans_to_sun = (vis_fraction > 1e-4) ? get_transmittance(rq, term_q.y) : vec3(0.0);
        vec3 inscatter_direct = (beta_R * (rho_R * polar_rayleigh_boost) * phase_R + beta_M * rho_M * phase_M_eff)
                                 * trans_to_sun * vis_fraction * star_int;

        // Multiple scattering component matching sky_view_lut.frag
        float h_norm_q = clamp(hq / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
        float ms_u_q = 0.5 + 0.5 * sign(light_cos_theta) * sqrt(abs(light_cos_theta));
        float ms_v_q = sqrt(h_norm_q);
        vec3 psi_q = textureLod(u_multi_scatter_lut, vec2(ms_u_q, ms_v_q), 0.0).rgb;
        vec3 inscatter_ms = star_int * (beta_R * rho_R + beta_M * rho_M) * psi_q;

        // Evaluate caster shadow at this parcel in render frame (AU)
        vec3 P_render = planet_center_render + (cam_local + sq * ray_dir) / u_au_to_km;
        vec3 shadow_val = compute_caster_shadow(P_render, L_cart, dist_to_star_au, star_radius_au, star_obl, star_pole, caster_mask);
        vec3 caster_blocked = clamp(vec3(1.0) - shadow_val, vec3(0.0), vec3(1.0));

        vec3 step_unshadowed = (inscatter_direct + inscatter_ms) * T_run * int_factor;
        slice_total += step_unshadowed;
        delta_L += step_unshadowed * caster_blocked;
        T_run *= T_cell;
    }
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

// TERRAIN_DEPTH_ENDPOINT_BEGIN
void resolveTerrainDepthEndpoint(
    float s_depth, float s_start, vec2 s_atmo,
    vec3 cam_local_sph, vec3 ray_dir_sph,
    float atmo_radius_km, float analytical_surface_s,
    bool terrain_depth_enabled,
    inout float s_end, inout bool hits_surface,
    out bool terrain_depth_override)
{
    terrain_depth_override = false;
    vec3 depth_pos_sph = cam_local_sph + s_depth * ray_dir_sph;
    float depth_radius_sph = length(depth_pos_sph);
    float depth_shell_margin = max(0.01, atmo_radius_km * 1e-5);
    bool depth_in_local_atmo =
        s_depth >= s_start - depth_shell_margin &&
        s_depth <= s_atmo.y + depth_shell_margin &&
        depth_radius_sph <= atmo_radius_km + depth_shell_margin;
    if (!depth_in_local_atmo) return;

    float depth_surface_s = clamp(s_depth, s_start, s_atmo.y);
    float terrain_delta = abs(depth_surface_s - analytical_surface_s);
    bool displaced_terrain_hit = terrain_depth_enabled && terrain_delta > 0.001;

    // Raised terrain naturally shortens the current endpoint. For terrain
    // below the reference ellipsoid, replace the analytical endpoint only
    // when no nearer ring/caster already clipped it.
    bool no_nearer_occluder = s_end >= analytical_surface_s - depth_shell_margin;
    if (depth_surface_s < s_end || (displaced_terrain_hit && no_nearer_occluder)) {
        s_end = depth_surface_s;
        hits_surface = true;
        terrain_depth_override = displaced_terrain_hit;
    }
}
// TERRAIN_DEPTH_ENDPOINT_END

#include "common/aerial_volume_shadows.glsl"

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
    vec3 cam_pos_eff = cam_local_sph;
    if (dist_to_planet_center_km < u_planet_radius_km + 1e-4) {
        cam_pos_eff = (dist_to_planet_center_km > 1e-4)
            ? (cam_local_sph * ((u_planet_radius_km + 1e-4) / dist_to_planet_center_km))
            : vec3(0.0, u_planet_radius_km + 1e-4, 0.0);
    }

    vec2 s_planet = raySphereIntersect(cam_pos_eff, ray_dir_sph, u_planet_clip_km);

    float s_start = max(0.0, s_atmo.x);
    float s_end = s_atmo.y;

    bool hits_surface = false;
    if (s_planet.x > 0.0 && s_planet.x < s_end) {
        s_end = s_planet.x;
        hits_surface = true;
    }
    float analytical_surface_s = s_end;
    bool terrain_depth_override = false;
    bool has_local_opaque_depth = false;

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
            // Log depth belongs to the rendered screen ray. Refraction/lensing
            // can rotate the physical integration ray, but preserves distance.
            float cos_angle = dot(view_ray, cam_fw);
            if (cos_angle > 1e-4) {
                float s_depth = (scene_clip_z * u_au_to_km) / cos_angle;
                s_depth -= ray_shift_au * u_au_to_km; // Convert from camera-relative to O_local_km relative

                // An opaque foreground surface blocks this atmosphere too,
                // even when it lies outside this body's atmospheric ellipsoid.
                if (s_depth < s_start) discard;
                float depth_margin = max(0.01, u_atmo_radius_km * 1e-5);
                has_local_opaque_depth = s_depth <= s_atmo.y + depth_margin
                    && length(cam_local_sph + s_depth * ray_dir_sph) <= u_atmo_radius_km + depth_margin;

                // The analytical ellipsoid is only a fallback for pixels without
                // local opaque depth. Terrain can lie on either side of it.
                resolveTerrainDepthEndpoint(
                    s_depth, s_start, s_atmo, cam_local_sph, ray_dir_sph,
                    u_atmo_radius_km, analytical_surface_s,
                    u_terrain_depth_enabled, s_end, hits_surface,
                    terrain_depth_override);
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
    float s_closest = clamp(-dot(cam_local_sph, ray_dir_sph), s_start, s_end);
    float min_altitude = max(0.0, length(cam_local_sph + s_closest * ray_dir_sph) - u_planet_radius_km);

    vec3 pole_dir_norm = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    float max_bend = u_precomp_opt.w;
    float inv_h_rayleigh = u_precomp_opt.x;
    float inv_h_mie = u_precomp_opt.y;
    float inv_ozone_width = u_precomp_opt.z;

    // Optical depth awareness: compute peak extinction along this ray to set non-linear grading
    float peak_rho_R = exp(-min_altitude * inv_h_rayleigh);
    float peak_rho_M = exp(-min_altitude * inv_h_mie);
    float peak_ext = dot(beta_R, vec3(0.333333)) * peak_rho_R + dot(beta_M_ext, vec3(0.333333)) * peak_rho_M;
    float tau_ray_approx = peak_ext * min(ray_len, 2.0 * sqrt(max(0.0, 2.0 * u_planet_radius_km * u_h_rayleigh + u_h_rayleigh * u_h_rayleigh)));

    // Spherical limb geometry already concentrates path length quadratically near closest approach (h ~ h_min + s^2 / 2R).
    // Applying power grading (p > 1) on limb rays distorts altitude as u^(2p) (e.g. u^5 for p=2.5), which over-concentrates
    // steps in the core and stretches outer steps to >1,000 km, ruining numerical convergence and color stability.
    // Therefore, limb rays use linear distance spacing (p = 1.0).
    // Only surface-intersecting rays (where altitude varies towards the planetary surface) benefit from
    // exponential grading towards the planetary surface.
    bool is_surface_ray = hits_surface;

    float grade_p = 1.0;
    if (is_surface_ray && tau_ray_approx > 1.0) {
        grade_p = clamp(1.0 + 0.5 * log(tau_ray_approx), 1.0, 2.0);
    }

    float jitter = 0.5;
    if (u_stochastic_noise) {
        jitter = get_stochastic_jitter(gl_FragCoord.xy, u_frame_counter);
    }

    int steps = u_num_samples;
    if (u_atmo_adaptive_steps) {
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
    vec3 scattered = vec3(0.0);
    vec3 final_transmittance = vec3(1.0);

    bool use_sky_view_lut = (u_atmo_quality == 3) && !terrain_depth_override;
    int raymarch_quality = (u_atmo_quality == 3) ? 2 : u_atmo_quality;

    // Host refraction can rotate the view ray: sample the volume using the
    // projected bent direction. Rays with an anchored lens/refraction origin
    // still require the terrain-aware marcher because they leave this camera.
    vec4 aerial_clip = projection * view * vec4(ray_dir, 0.0);
    vec2 aerial_uv = aerial_clip.xy / max(aerial_clip.w, 1e-12) * 0.5 + 0.5;
    bool use_aerial_volume = u_atmo_quality == 3 && u_aerial_volume_enabled
        && u_terrain_depth_enabled && has_local_opaque_depth && hits_surface && s_min_au == 0.0
        && aerial_clip.w > 0.0
        && all(greaterThanEqual(aerial_uv, vec2(0.0)))
        && all(lessThanEqual(aerial_uv, vec2(1.0)));
    vec3 aerial_S, aerial_T;
    bool aerial_sample_valid = false;
    float aerial_refinement = 0.0;
    float aerial_shadow_refinement = 0.0;
    vec2 aerial_shadow_span = vec2(s_end, s_start);
    float minimum_shadow_width = s_end - s_start;
    ivec3 aerial_segment_steps = ivec3(0);
    if (use_aerial_volume) {
        vec3 V = ray_dir_sph;
        bool clipped_start = s_start > max(0.0, s_atmo.x) + 1e-5;
        float a = clipped_start
            ? dot(cam_local_sph + s_start * V - u_aerial_cam_sph, V) / dot(V, V)
            : 0.0;
        float b = dot(cam_local_sph + s_end * V - u_aerial_cam_sph, V) / dot(V, V);
        {
            if (u_num_active_casters > 0 || (u_num_ring_planes > 0 && u_ring_mask != 0u)) {
                // Expand selection by one coarse column footprint. The shadow
                // itself is evaluated only along the actual finite pixel ray.
                vec2 grid = vec2(textureSize(u_aerial_scatter_lut, 0).xy);
                float footprint = 0.0;
                for (int axis = 0; axis < 2; axis++) {
                    vec2 offset = axis == 0 ? vec2(1.0 / grid.x, 0.0) : vec2(0.0, 1.0 / grid.y);
                    for (int side = -1; side <= 1; side += 2) {
                        vec3 neighbor_sph = aerialColumnRay(clamp(aerial_uv + float(side) * offset, 0.0, 1.0),
                            u_inv_proj, u_inv_view, u_pole_obl);
                        footprint = max(footprint, s_end * length(fromSphericalSpace(neighbor_sph - V, u_pole_obl)));
                    }
                }
                aerial_shadow_refinement = aerialShadowRefinement(cam_local, ray_dir, s_start, s_end, footprint,
                                                                  aerial_shadow_span, minimum_shadow_width);
            }
            aerial_refinement = aerial_shadow_refinement;
            if (aerial_refinement < 1.0) {
                float angular_footprint = aerialGroundFootprint(aerial_uv, b, u_aerial_cam_sph, V,
                    textureSize(u_aerial_scatter_lut, 0).xy, u_inv_proj, u_inv_view, u_pole_obl);
                for (int st = 0; st < clamp(u_num_stars, 1, 4); st++) {
                    vec3 L = u_star_dir_sph_eff[st].xyz;
                    if (length(L) < 1e-4) L = normalize(toSphericalSpace(
                        u_stars_pos_radius[st].xyz - u_body_offset, u_pole_obl));
                    aerial_refinement = max(aerial_refinement, aerialTerminatorRefinement(
                        b, u_aerial_cam_sph, V, vec4(L, u_star_pos_local[st].w),
                        vec3(u_planet_radius_km, u_atmo_radius_km, max(u_h_rayleigh, u_h_mie)),
                        angular_footprint));
                }
            }
        }
        // Fully refined pixels need column compatibility, but no cached
        // radiance. Decide this before paying for 16 (or 32) voxel fetches.
        bool read_voxels = aerial_refinement < 1.0;
        bool valid_b = sampleAerialGround(u_aerial_scatter_lut, u_aerial_trans_lut,
            aerial_uv, b, u_aerial_cam_sph, V, u_atmo_radius_km, u_aerial_radius_range,
            u_inv_proj, u_inv_view, u_pole_obl, read_voxels, aerial_S, aerial_T);
        // Recover clipped segments: T_ab=T_b/T_a, S_ab=(S_b-S_a)/T_a.
        if (clipped_start && valid_b) {
            vec3 Sa, Ta;
            valid_b = sampleAerialGround(u_aerial_scatter_lut, u_aerial_trans_lut,
                aerial_uv, a, u_aerial_cam_sph, V, u_atmo_radius_km, u_aerial_radius_range,
                u_inv_proj, u_inv_view, u_pole_obl, read_voxels, Sa, Ta);
            aerial_S = max(vec3(0.0), (aerial_S - Sa) / max(Ta, vec3(1e-6)));
            aerial_T = clamp(aerial_T / max(Ta, vec3(1e-6)), vec3(0.0), vec3(1.0));
        }
        aerial_sample_valid = valid_b;
        if (!valid_b) {
            aerial_refinement = 0.0;
            aerial_shadow_refinement = 0.0;
        }
        use_aerial_volume = valid_b && aerial_refinement == 0.0;
        if (!use_aerial_volume) use_sky_view_lut = false;
        if (aerial_shadow_refinement > 0.0) {
            // Resolve thin shadows inside their finite bounds instead of
            // spending 128 samples uniformly across mostly unshadowed air.
            // A requested 128-step reference retains its original quadrature.
            if (u_num_samples >= 128 || steps >= 128) {
                steps = max(steps, 128);
            } else if (aerial_shadow_span.y > aerial_shadow_span.x) {
                int shadow_steps = max(steps, clamp(u_aerial_shadow_steps, 16, 128));
                // Several separated thin shadows can have a wide union. Raise
                // the budget there so concentrating the march cannot miss them.
                int feature_steps = int(clamp(ceil(8.0 * (aerial_shadow_span.y - aerial_shadow_span.x)
                    / max(minimum_shadow_width, 1e-5)), 16.0, 128.0));
                shadow_steps = max(shadow_steps, feature_steps);
                float inv_length = 1.0 / max(s_end - s_start, 1e-6);
                float outer_budget = float(max(u_num_samples, 8));
                int before = aerial_shadow_span.x > s_start + 1e-5
                    ? max(1, int(ceil(outer_budget * (aerial_shadow_span.x - s_start) * inv_length))) : 0;
                int after = aerial_shadow_span.y < s_end - 1e-5
                    ? max(1, int(ceil(outer_budget * (s_end - aerial_shadow_span.y) * inv_length))) : 0;
                aerial_segment_steps = ivec3(before, shadow_steps, after);
                steps = before + shadow_steps + after;
            }
        }
    }
    if (use_aerial_volume) {
        scattered = aerial_S;
        final_transmittance = aerial_T;
    } else if (use_sky_view_lut) {
        // Multi-pass ring clipping: Pass 1 (clip_mode == 1) renders behind the rings.
        // If clip_mode == 2, discard for pixels where rings exist in front to avoid duplicate draw blowout.
        if (u_atmo_clip_mode == 2 && closest_s_ring < s_end) {
            discard;
        }

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
        bool cam_inside = (D <= u_atmo_radius_km);
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

        // Volumetric Ring Shadow via Analytical Depth Slicing (per star).
        // Each star's shadow interval is solved and marched separately, and the
        // deficit is normalized against THAT star's baked in-scatter slice
        // (u_sky_view_star_lut) — so a ring shadow from star A never darkens
        // regions that are only lit by star B.
        vec3 p_up = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
        vec3 p_ref = (abs(p_up.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.1, 1.0);
        vec3 p_right = normalize(cross(p_up, p_ref));
        vec3 p_fwd = cross(p_right, p_up);

        vec3 C_ring = vec3(dot(cam_local, p_right), dot(cam_local, p_up), dot(cam_local, p_fwd));
        vec3 V_ring = vec3(dot(ray_dir, p_right), dot(ray_dir, p_up), dot(ray_dir, p_fwd));

        float g = clamp(u_mie_g, 0.0, 0.88);
        float c1 = (u_precomp_mie.x > 1e-6) ? u_precomp_mie.x : ((3.0 / (8.0 * PI)) * ((1.0 - g * g) / (2.0 + g * g)));
        float c2 = (u_precomp_mie.y > 1e-6) ? u_precomp_mie.y : (1.0 + g * g);
        float c3 = (u_precomp_mie.z > 1e-6) ? u_precomp_mie.z : (2.0 * g);

        vec3 L_atmo = L_scatter;
        int n_stars_m3 = clamp(u_num_stars, 1, 4);
        bool limb_faded = (!cam_inside && !is_ground);
        float limb_fade = limb_faded ? clamp((1.0 - t_limb) * 128.0, 0.0, 1.0) : 1.0;

        for (int st = 0; st < n_stars_m3; st++) {
            u_cur_solstice = (st < 4) ? u_star_solstice[st].xy : vec2(0.0);
            vec3 sun_pos_km = u_star_pos_local[st].xyz;
            if (dot(sun_pos_km, sun_pos_km) < 1e-8) continue;
            vec3 L_cart = normalize(sun_pos_km);
            vec3 L_ring = normalize(vec3(dot(L_cart, p_right), dot(L_cart, p_up), dot(L_cart, p_fwd)));

            if (s_start >= s_end) continue;
            bool can_do_rings = (u_num_ring_planes > 0 && abs(L_ring.y) > 1e-5);

            vec3 L_sun_s = normalize(u_star_dir_sph_eff[st].xyz);
            if (length(L_sun_s) < 1e-4) L_sun_s = L_sun; // fallback: primary direction
            vec3 star_int_s = (length(u_star_color_irrad[st].rgb) > 1e-6) ? u_star_color_irrad[st].rgb : ((st == 0) ? vec3(1.0) : vec3(0.0));
            float eff_star_rad_s = u_star_pos_local[st].w;
            float cos_sun_eff_s = u_star_dir_sph_eff[st].w;

            // Phase functions are constant along the view ray: hoist out of the ring loop.
            float cos_theta_sun_s = dot(normalize(ray_dir_sph), L_sun_s);
            float phase_R_s = (3.0 / (16.0 * PI)) * (1.0 + cos_theta_sun_s * cos_theta_sun_s);
            float phase_M_s = c1 * (1.0 + cos_theta_sun_s * cos_theta_sun_s) / pow(max(1e-4, c2 - c3 * cos_theta_sun_s), 1.5);

            vec3 delta_L = vec3(0.0);
            vec3 slice_total = vec3(0.0);
            bool has_shadow_interval = false;
            float total_shadow_len = 0.0;
            float total_ring_blocked_accum = 0.0;
            float shadow_step_count = 0.0;
            if (can_do_rings) {
            // Project ray P(s) = C_ring + s*V_ring onto equatorial ring plane Y = 0 along sunlight vector L_ring:
            vec3 A = C_ring - (C_ring.y / L_ring.y) * L_ring;
            vec3 B = V_ring - (V_ring.y / L_ring.y) * L_ring;

            float qa = B.x * B.x + B.z * B.z;
            float qb = 2.0 * (A.x * B.x + A.z * B.z);
            float qc = A.x * A.x + A.z * A.z;

            for (int k = 0; k < u_num_ring_planes && k < 4; k++) {
                if ((u_ring_mask & (1u << k)) == 0u) continue;
                // Circumplanetary (own) rings only: external rings (parent/neighbor planets) are
                // baked directly into the Sky-View LUT (sky_view_lut.frag) along with other eclipse casters.
                if (distance(u_ring_center[k], planet_center_render) > 1e-4) continue;
                if (!aerialRingMayOcclude(k, 0.0, 1.0)) continue;

                float r_inner = u_ring_params[k].x * u_au_to_km;
                float r_outer = u_ring_params[k].y * u_au_to_km;
                float ring_v_coord = (float(k) + 0.5) / 16.0;
                float ring_span = r_outer - r_inner;
                float inv_sin_sun = 1.0 / max(abs(L_ring.y), 0.05);

                // Safe screen footprint derivative evaluated unconditionally across screen pixels
                float s_closest = clamp(-qb / (2.0 * max(qa, 1e-6)), s_start, s_end);
                float r_closest = sqrt(max(0.0, qa * s_closest * s_closest + qb * s_closest + qc));
                float u_closest = (r_closest - r_inner) / max(ring_span, 1e-4);
                float du_screen = clamp(fwidth(u_closest), 0.0, 0.05);

                // Shadow validity window: lambda = -(C_ring.y + s*V_ring.y)/L_ring.y > 0
                float w_min = s_start;
                float w_max = s_end;

                if (abs(V_ring.y) > 1e-6) {
                    float s_crit = -C_ring.y / V_ring.y;
                    if ((V_ring.y / L_ring.y) > 0.0) {
                        w_max = min(w_max, s_crit);
                    } else {
                        w_min = max(w_min, s_crit);
                    }
                } else {
                    if (-C_ring.y / L_ring.y <= 0.0) {
                        w_min = 1e9;
                    }
                }

                if (w_min < w_max) {
                    if (u_atmo_shadow_method == 2) {
                        // Method 2 (Blackrack / KSA): Single continuous chord segment [cs, ce].
                        // Never split the segment into two when crossing the inner hole, preventing
                        // abrupt step size discontinuities and concentric onion banding.
                        float cs = 0.0;
                        float ce = -1.0;

                        if (qa > 1e-6) {
                            float disc_out = qb * qb - 4.0 * qa * (qc - r_outer * r_outer);
                            if (disc_out >= 0.0) {
                                float sq_out = sqrt(disc_out);
                                float s_out_min = (-qb - sq_out) / (2.0 * qa);
                                float s_out_max = (-qb + sq_out) / (2.0 * qa);

                                cs = max(s_out_min, w_min);
                                ce = min(s_out_max, w_max);

                                float disc_in = qb * qb - 4.0 * qa * (qc - r_inner * r_inner);
                                if (disc_in >= 0.0) {
                                    float sq_in = sqrt(disc_in);
                                    float s_in_min = (-qb - sq_in) / (2.0 * qa);
                                    float s_in_max = (-qb + sq_in) / (2.0 * qa);

                                    // Trim inner hole only from segment ends (matching KSA Atmosphere.comp L365-380).
                                    // If inner hole is strictly inside the segment, march straight through; ring_blocked = 0 there.
                                    bool startsInsideHole = (cs >= s_in_min && cs <= s_in_max);
                                    bool endsInsideHole = (ce >= s_in_min && ce <= s_in_max);

                                    if (startsInsideHole && endsInsideHole) {
                                        ce = cs - 1.0; // Segment lies entirely within the hole
                                    } else if (startsInsideHole) {
                                        cs = s_in_max;
                                    } else if (endsInsideHole) {
                                        ce = s_in_min;
                                    }
                                }
                            }
                        } else {
                            // Degenerate case: view ray is parallel or anti-parallel to sunlight (V_ring || L_ring)
                            float r_A = length(A.xz);
                            if (r_A >= r_inner && r_A <= r_outer) {
                                cs = w_min;
                                ce = w_max;
                            }
                        }

                        if (ce > cs + 1e-5) {
                            has_shadow_interval = true;
                            total_shadow_len += (ce - cs);

                            vec3 T_run = vec3(1.0);
                            if (cs > s_start + 1e-3) {
                                vec3 T_gap = get_transmittance_segment(s_start, cs, cam_local_sph, ray_dir_sph,
                                                                       beta_R, beta_M_ext, beta_A_mixed, beta_A_layered,
                                                                       inv_h_rayleigh, inv_h_mie, inv_ozone_width);
                                T_run *= T_gap;
                            }

                            int N_steps = clamp(u_ring_station_count, 8, 48);
                            march_bounded_shadow_blackrack(cs, ce, du_screen, qa, qb, qc, r_inner, ring_span, inv_sin_sun,
                                                           cam_local_sph, ray_dir_sph, L_sun_s, phase_R_s, phase_M_s,
                                                           eff_star_rad_s, cos_sun_eff_s, star_int_s,
                                                           beta_R, beta_M, beta_M_ext, beta_A_mixed, beta_A_layered,
                                                           inv_h_rayleigh, inv_h_mie, inv_ozone_width,
                                                           N_steps,
                                                           T_run, delta_L, slice_total);
                        }
                    } else {
                        // Methods 0 and 1: Interval splitting across inner hole
                        vec2 intervals[2];
                        int num_intervals = 0;

                        if (qa > 1e-6) {
                            float disc_out = qb * qb - 4.0 * qa * (qc - r_outer * r_outer);
                            if (disc_out >= 0.0) {
                                float sq_out = sqrt(disc_out);
                                float s_out_min = (-qb - sq_out) / (2.0 * qa);
                                float s_out_max = (-qb + sq_out) / (2.0 * qa);

                                float disc_in = qb * qb - 4.0 * qa * (qc - r_inner * r_inner);

                                if (disc_in <= 0.0) {
                                    intervals[0] = vec2(s_out_min, s_out_max);
                                    num_intervals = 1;
                                } else {
                                    float sq_in = sqrt(disc_in);
                                    float s_in_min = (-qb - sq_in) / (2.0 * qa);
                                    float s_in_max = (-qb + sq_in) / (2.0 * qa);
                                    intervals[0] = vec2(s_out_min, s_in_min);
                                    intervals[1] = vec2(s_in_max, s_out_max);
                                    num_intervals = 2;
                                }
                            }
                        } else {
                            // Degenerate case: view ray is parallel or anti-parallel to sunlight (V_ring || L_ring).
                            // All points along the chord project onto the equatorial ring plane at point A.
                            float r_A = length(A.xz);
                            if (r_A >= r_inner && r_A <= r_outer) {
                                intervals[0] = vec2(w_min, w_max);
                                num_intervals = 1;
                            }
                        }

                        if (num_intervals > 0) {
                            vec3 T_run = vec3(1.0);
                            float prev_s = s_start;

                            for (int inv = 0; inv < 2; inv++) {
                                if (inv >= num_intervals) break;

                                float cs = max(intervals[inv].x, w_min);
                                float ce = min(intervals[inv].y, w_max);

                                if (ce > cs + 1e-5) {
                                    has_shadow_interval = true;
                                    total_shadow_len += (ce - cs);

                                    // Account for unshadowed segment before this interval (from prev_s to cs)
                                    if (cs > prev_s + 1e-3) {
                                        vec3 T_gap = get_transmittance_segment(prev_s, cs, cam_local_sph, ray_dir_sph,
                                                                               beta_R, beta_M_ext, beta_A_mixed, beta_A_layered,
                                                                               inv_h_rayleigh, inv_h_mie, inv_ozone_width);
                                        T_run *= T_gap;
                                    }

                                    if (u_atmo_shadow_method == 1) {
                                        // Method 1: Uniform Stochastic Raymarching across [cs, ce]
                                        int N_steps = clamp(u_ring_station_count, 4, 64);
                                        float inv_jitter = (inv == 0 && st == 0) ? jitter : fract(jitter + float(inv) * 0.381966 + float(st) * 0.618034);
                                        march_uniform_shadow(cs, ce, qa, qb, qc, r_inner, ring_span, inv_sin_sun,
                                                             cam_local_sph, ray_dir_sph, L_sun_s, phase_R_s, phase_M_s,
                                                             eff_star_rad_s, cos_sun_eff_s, star_int_s,
                                                             beta_R, beta_M, beta_M_ext, beta_A_mixed, beta_A_layered,
                                                             inv_h_rayleigh, inv_h_mie, inv_ozone_width,
                                                             inv_jitter, N_steps,
                                                             T_run, delta_L, slice_total, total_ring_blocked_accum, shadow_step_count);
                                    } else {
                                        // Method 0: Station-locked slicing quadrature across the shadow interval [cs, ce].
                                        // Cell boundaries are fixed in ring coordinate u (CPU-baked by
                                        // bake_station_cells, edges snapped to ring profile discontinuities).
                                        if (qa > 1e-6) {
                                            if (num_intervals == 2) {
                                                // Chord crosses the inner hole: interval 0 is the incoming
                                                // (sig = -1) branch, interval 1 the outgoing (sig = +1) branch.
                                                int sig = (inv == 0) ? -1 : 1;
                                                walk_station_branch(cs, ce, sig, qa, qb, qc, r_inner, ring_span, inv_sin_sun,
                                                                    cam_local_sph, ray_dir_sph, L_sun_s, phase_R_s, phase_M_s,
                                                                    eff_star_rad_s, cos_sun_eff_s, star_int_s,
                                                                    beta_R, beta_M, beta_M_ext, beta_A_mixed, beta_A_layered,
                                                                    inv_h_rayleigh, inv_h_mie, inv_ozone_width,
                                                                    T_run, delta_L, slice_total, total_ring_blocked_accum, shadow_step_count);
                                            } else {
                                                // Apex inside the annulus: split the single interval at the apex
                                                // into incoming (sig = -1) and outgoing (sig = +1) branches.
                                                float s_apex = clamp(-qb / (2.0 * qa), cs, ce);
                                                walk_station_branch(cs, s_apex, -1, qa, qb, qc, r_inner, ring_span, inv_sin_sun,
                                                                    cam_local_sph, ray_dir_sph, L_sun_s, phase_R_s, phase_M_s,
                                                                    eff_star_rad_s, cos_sun_eff_s, star_int_s,
                                                                    beta_R, beta_M, beta_M_ext, beta_A_mixed, beta_A_layered,
                                                                    inv_h_rayleigh, inv_h_mie, inv_ozone_width,
                                                                    T_run, delta_L, slice_total, total_ring_blocked_accum, shadow_step_count);
                                                walk_station_branch(s_apex, ce, 1, qa, qb, qc, r_inner, ring_span, inv_sin_sun,
                                                                    cam_local_sph, ray_dir_sph, L_sun_s, phase_R_s, phase_M_s,
                                                                    eff_star_rad_s, cos_sun_eff_s, star_int_s,
                                                                    beta_R, beta_M, beta_M_ext, beta_A_mixed, beta_A_layered,
                                                                    inv_h_rayleigh, inv_h_mie, inv_ozone_width,
                                                                    T_run, delta_L, slice_total, total_ring_blocked_accum, shadow_step_count);
                                            }
                                        } else {
                                            // Degenerate chord (view ray parallel to sunlight in the ring frame):
                                            // u is constant along the interval — integrate with uniform steps at fixed u.
                                            int K = clamp(u_ring_station_count, 2, 32);
                                            float u_A = (length(A.xz) - r_inner) / ring_span;
                                            float ds = (ce - cs) / float(K);
                                            for (int q = 0; q < K; q++) {
                                                accumulate_shadow_cell(cs + float(q) * ds, cs + float(q + 1) * ds, u_A, 1e-5, inv_sin_sun,
                                                                       cam_local_sph, ray_dir_sph, L_sun_s, phase_R_s, phase_M_s,
                                                                       eff_star_rad_s, cos_sun_eff_s, star_int_s,
                                                                       beta_R, beta_M, beta_M_ext, beta_A_mixed, beta_A_layered,
                                                                       inv_h_rayleigh, inv_h_mie, inv_ozone_width,
                                                                       T_run, delta_L, slice_total, total_ring_blocked_accum, shadow_step_count);
                                            }
                                        }
                                    }

                                    prev_s = ce;
                                }
                            }
                        }
                    }
                }
            }
        }

            if (has_shadow_interval) {
                // Deficit evaluation against this star's baked in-scatter slice:
                // the subtraction can never eat another star's light or shine.
                vec3 L_scatter_s = texture(u_sky_view_star_lut[st], vec2(u_lut, v_lut)).rgb;
                if (limb_faded) L_scatter_s *= limb_fade;

                if (u_atmo_shadow_method == 2) {
                    // Method 2 (Blackrack / KSA): Direct physical deficit subtraction.
                    // Clamped to L_scatter_s so a ring shadow cannot subtract more than
                    // this star's total scattered light. Completely eliminates chord_coverage
                    // blending and ratio normalization, removing concentric onion-shell banding.
                    vec3 sub_val = min(delta_L, L_scatter_s);
                    L_atmo = max(vec3(0.0), L_atmo - sub_val);
                } else {
                    // Methods 0 and 1: Station-Locked Slicing and Uniform Stochastic Raymarching
                    float chord_len = max(s_end - s_start, 1e-4);
                    float chord_coverage = clamp(total_shadow_len / chord_len, 0.0, 1.0);

                    vec3 shadow_factor = vec3(0.0);
                    if (dot(slice_total, vec3(1.0)) > 1e-6) {
                        shadow_factor = clamp(delta_L / max(slice_total, vec3(1e-6)), vec3(0.0), vec3(1.0));
                    } else if (shadow_step_count > 0.0) {
                        float avg_blocked = total_ring_blocked_accum / shadow_step_count;
                        shadow_factor = vec3(clamp(avg_blocked, 0.0, 1.0));
                    }

                    float blend = smoothstep(0.85, 0.98, chord_coverage);
                    vec3 eff_slice = mix(slice_total, L_scatter_s, blend);
                    L_atmo = max(vec3(0.0), L_atmo - eff_slice * shadow_factor);
                }
            }

            // Method 2 (Blackrack): Analytical bounding cylinder culling + raymarching for celestial eclipse casters (moons/planets)
            if (u_atmo_shadow_method == 2 && u_num_active_casters > 0) {
                vec3 delta_L_eclipse = vec3(0.0);
                vec3 slice_total_eclipse = vec3(0.0);
                vec3 cone_axis = -L_cart;

                float dist_to_star_au = max(u_star_solstice[st].z, 1e-6);
                float star_radius_au = u_star_solstice[st].w;
                float star_obl = u_stars_poles_obl[st].w;
                vec3 star_pole = u_stars_poles_obl[st].xyz;

                for (int c = 0; c < u_num_active_casters; c++) {
                    vec3 caster_pos_au = u_active_casters[c].xyz;
                    float caster_r_au = u_active_casters[c].w;
                    float atmo_h_au = u_active_caster_atmos[c].w;

                    vec3 caster_local_km = (caster_pos_au - planet_center_render) * u_au_to_km;

                    // Skip self-shadowing (host planet itself is not an external caster on its own atmosphere)
                    if (length(caster_local_km) < 1.0) continue;

                    // Check if caster lies between star and planet
                    float t_caster = dot(caster_local_km, L_cart);
                    if (t_caster <= 0.0) continue;

                    float dist_to_caster_au = length(caster_pos_au - planet_center_render);
                    float eff_r_au = caster_r_au + (atmo_h_au > 0.0 ? atmo_h_au * 4.0 : 0.0);
                    float r_penumbra_au = eff_r_au + dist_to_caster_au * (star_radius_au / dist_to_star_au);
                    float r_penumbra_km = r_penumbra_au * u_au_to_km * 1.15; // 15% safety dilation

                    vec2 cyl_bounds = get_eclipse_cylinder_bounds(cam_local, ray_dir, caster_local_km, cone_axis, r_penumbra_km);
                    if (cyl_bounds.y < 0.0) continue;

                    float cs_c = max(cyl_bounds.x, s_start);
                    float ce_c = min(cyl_bounds.y, s_end);
                    if (ce_c <= cs_c + 1e-4) continue;

                    int N_steps_eclipse = clamp(u_ring_station_count, 8, 32);
                    march_bounded_eclipse_blackrack(
                        cs_c, ce_c, s_start,
                        cam_local, ray_dir, cam_local_sph, ray_dir_sph,
                        planet_center_render, L_cart, L_sun_s,
                        phase_R_s, phase_M_s,
                        eff_star_rad_s, cos_sun_eff_s, star_int_s,
                        dist_to_star_au, star_radius_au, star_obl, star_pole,
                        c,
                        beta_R, beta_M, beta_M_ext, beta_A_mixed, beta_A_layered,
                        inv_h_rayleigh, inv_h_mie, inv_ozone_width,
                        N_steps_eclipse,
                        delta_L_eclipse, slice_total_eclipse);
                }

                if (dot(delta_L_eclipse, vec3(1.0)) > 1e-6) {
                    vec3 L_scatter_s = texture(u_sky_view_star_lut[st], vec2(u_lut, v_lut)).rgb;
                    if (limb_faded) L_scatter_s *= limb_fade;
                    vec3 max_sub = max(vec3(0.0), L_scatter_s - delta_L);
                    vec3 eclipse_sub = min(delta_L_eclipse, max_sub);
                    L_atmo = max(vec3(0.0), L_atmo - eclipse_sub);
                }
            }
        } // per-star ring-shadow loop

        scattered = L_atmo;
        final_transmittance = T_lut;
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
        vec4 ring_opac = vec4(0.0);
        vec4 ring_v_coord = vec4(0.0);
        vec4 ring_R_eff = vec4(0.0);
        vec3 ring_A_prime = vec3(0.0);
        vec3 ring_B = vec3(0.0);

        int opt_caster_count = 0;
        int ring_count = 0;
        vec4 c_p0[4];
        vec4 c_p1[4];
        vec4 c_p2[4];
        vec4 c_p3[4];
        vec2 c_p4[4];
        vec4 c_p5[4];

        float s_mid = (s_start + s_end) * 0.5;
        vec3 mid_pos = frag_local + s_mid * ray_dir;
        vec3 mid_render = mid_pos / u_au_to_km + planet_center_render;
        vec3 mid_to_star = star_pos - mid_render;
        float sr_start = star_radius / max(dist_mid_star, 1e-6);
        vec3 O = frag_local / u_au_to_km + planet_center_render;
        vec3 V = ray_dir; // Fix dimensional error: ray_dir is unit vector in km space
        if (raymarch_quality > 0 && !skip_volumetric_shadow) {
            if (raymarch_quality == 1) {
                global_eclipse_shadow = compute_shadow(mid_render, L_mid, dist_mid_star, planet_center_render, star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz);
                end_eclipse_shadow = global_eclipse_shadow;
            }

            if (raymarch_quality == 2) {
                ring_count = 0;
                uint processed_mask = 0u;
                if (u_num_ring_planes > 0) {
                    for (int k = 0; k < u_num_ring_planes; k++) {
                        if ((u_ring_mask & (1u << k)) == 0u) continue;
                        if ((processed_mask & (1u << k)) != 0u) continue;

                        vec3 N = u_ring_normal[k];
                        vec3 C_km = u_ring_center[k] * u_au_to_km;
                        vec3 O_km = frag_local + planet_center_render * u_au_to_km;

                        float denom = dot(L_mid, N);
                        if (abs(denom) < 1e-8) continue;

                        float ra_km = dot(C_km - O_km, N) / denom;
                        float rb_km = -dot(ray_dir, N) / denom;

                        vec3 A_km = O_km + ra_km * L_mid;
                        vec3 B_km = ray_dir + rb_km * L_mid;
                        vec3 A_prime_km = A_km - C_km;

                        float qa = dot(B_km, B_km);
                        float qb = 2.0 * dot(A_prime_km, B_km);
                        float qc = dot(A_prime_km, A_prime_km);

                        float r_star_proj_km = abs((star_radius * u_au_to_km) * ra_km / max(dist_mid_star * u_au_to_km, 1e-6));
                        float R_eff_km = r_star_proj_km;
                        vec3 L_plane = L_mid - denom * N;
                        float L_plane_len = length(L_plane);
                        if (L_plane_len > 1e-5 && qa > 1e-8) {
                            vec3 L_proj = L_plane / L_plane_len;
                            vec3 T_vec = normalize(cross(N, L_mid));
                            float s_mid_ring = (-qb) / (2.0 * qa);
                            vec3 dir_radial = normalize(A_prime_km + s_mid_ring * B_km);
                            float cos_theta = dot(dir_radial, L_proj);
                            float sin_theta = dot(dir_radial, T_vec);
                            R_eff_km = r_star_proj_km * sqrt( pow(cos_theta / max(1e-6, abs(denom)), 2.0) + pow(sin_theta, 2.0) );
                        }

                        uint coplanar_mask = u_ring_coplanar_mask[k];
                        processed_mask |= coplanar_mask;
                        for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
                            if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
                            if (ring_count >= 4) continue;

                            float inner_r_km = u_ring_params[ring_idx].x * u_au_to_km;
                            float outer_r_km = u_ring_params[ring_idx].y * u_au_to_km;
                            float opacity = u_ring_params[ring_idx].z;
                            float scaled_tau = 1.0 / max(1e-4, abs(denom));

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

                            if (ring_count == 0) {
                                ring_s1_out.x = ring_s_valid_min; ring_s2_out.x = ring_s_valid_max;
                                ring_inner.x = inner_r_km; ring_outer.x = outer_r_km; ring_opac.x = scaled_tau; ring_v_coord.x = (float(ring_idx) + 0.5) / 16.0;
                                ring_A_prime = A_prime_km; ring_B = B_km; ring_R_eff.x = R_eff_km;
                            }
                            else if (ring_count == 1) {
                                ring_s1_out.y = ring_s_valid_min; ring_s2_out.y = ring_s_valid_max;
                                ring_inner.y = inner_r_km; ring_outer.y = outer_r_km; ring_opac.y = scaled_tau; ring_v_coord.y = (float(ring_idx) + 0.5) / 16.0;
                                ring_R_eff.y = R_eff_km;
                            }
                            else if (ring_count == 2) {
                                ring_s1_out.z = ring_s_valid_min; ring_s2_out.z = ring_s_valid_max;
                                ring_inner.z = inner_r_km; ring_outer.z = outer_r_km; ring_opac.z = scaled_tau; ring_v_coord.z = (float(ring_idx) + 0.5) / 16.0;
                                ring_R_eff.z = R_eff_km;
                            }
                            else if (ring_count == 3) {
                                ring_s1_out.w = ring_s_valid_min; ring_s2_out.w = ring_s_valid_max;
                                ring_inner.w = inner_r_km; ring_outer.w = outer_r_km; ring_opac.w = scaled_tau; ring_v_coord.w = (float(ring_idx) + 0.5) / 16.0;
                                ring_R_eff.w = R_eff_km;
                            }

                            ring_count++;
                        }
                    }
                }

                opt_caster_count = 0;
                if (u_num_active_casters > 0) {
                    for (int c = 0; c < u_num_active_casters && opt_caster_count < 4; c++) {
                        vec3 pos = u_active_casters[c].xyz;
                        float base_rad_au = u_active_casters[c].w;
                        float atmo_au = u_active_caster_atmos[c].w;
                        float caster_r_minor_au = u_active_caster_R_minor[c];
                        vec3 pole = u_active_caster_poles_obl[c].xyz;

                        vec3 D = (pos - planet_center_render) * u_au_to_km - frag_local;
                        float t0 = dot(D, L_mid);
                        float t1 = -dot(V, L_mid);

                        vec3 A = D - t0 * L_mid;
                        vec3 B = -V - t1 * L_mid;

                        float qa = dot(B, B);
                        float qb = 2.0 * dot(A, B);
                        float qc = dot(A, A);

                        float s_mid_local = (s_start + s_end) * 0.5;
                        vec3 perp_mid_km = A + s_mid_local * B;

                        float r_au = base_rad_au;
                        if (caster_r_minor_au < base_rad_au - 1e-5) {
                            r_au = get_oblate_radius(base_rad_au, caster_r_minor_au, pole, L_mid, perp_mid_km);
                        }
                        float r_km = r_au * u_au_to_km;
                        float atmo_km = atmo_au * u_au_to_km;
                        float eff_r_km = r_km + (atmo_km > 0.0 ? atmo_km * 4.0 : 0.0);

                        float directional_star_r_au = star_radius;
                        if (u_stars_poles_obl[s].w < star_radius - 1e-5) {
                            directional_star_r_au = get_oblate_radius(star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz, L_mid, perp_mid_km);
                        }
                        float alpha_star = directional_star_r_au / max(dist_mid_star, 1e-6);

                        float dist_approx_min_km = max(0.0, t0 + s_start * t1);
                        float dist_approx_max_km = max(0.0, t0 + s_end * t1);
                        float r_penumbra_max_km = eff_r_km + max(dist_approx_min_km, dist_approx_max_km) * alpha_star;

                        float s_valid_min = s_start;
                        float s_valid_max = s_end;

                        if (qa > 1e-8) {
                            float det = qb * qb - 4.0 * qa * (qc - r_penumbra_max_km * r_penumbra_max_km);
                            if (det < 0.0) continue;

                            float sqrt_det = sqrt(det);
                            float s_c1 = (-qb - sqrt_det) / (2.0 * qa);
                            float s_c2 = (-qb + sqrt_det) / (2.0 * qa);

                            s_valid_min = max(s_start, min(s_c1, s_c2));
                            s_valid_max = min(s_end, max(s_c1, s_c2));

                            if (s_valid_min > s_valid_max) continue;
                        } else {
                            if (qc > r_penumbra_max_km * r_penumbra_max_km) continue;
                        }

                        float s_mid_valid = (s_valid_min + s_valid_max) * 0.5;
                        float t_proj_mid = t0 + s_mid_valid * t1;
                        if (t_proj_mid <= 0.0) continue;

                        float perp_sq_mid = qa * s_mid_valid * s_mid_valid + qb * s_mid_valid + qc;
                        float dist_to_caster_mid = sqrt(max(0.0, perp_sq_mid) + t_proj_mid * t_proj_mid);
                        float inv_dist = 1.0 / max(dist_to_caster_mid, 1e-6);

                        float beta = eff_r_km * inv_dist;
                        float po = alpha_star + beta;
                        float pi = abs(beta - alpha_star);

                        float r_penumbra_sq = po * po * dist_to_caster_mid * dist_to_caster_mid;
                        float occ_mult = min(1.0, (beta * beta) / max(1e-9, alpha_star * alpha_star));

                        c_p0[opt_caster_count] = vec4(qa, qb, qc, r_penumbra_sq);
                        c_p1[opt_caster_count] = vec4(inv_dist, po, pi, occ_mult);
                        c_p2[opt_caster_count] = vec4(alpha_star, beta, u_active_max_bend[c], dist_to_caster_mid / u_au_to_km);
                        c_p3[opt_caster_count] = u_active_caster_atmos[c];
                        c_p4[opt_caster_count] = vec2(s_valid_min, s_valid_max);
                        c_p5[opt_caster_count] = u_active_caster_ozone[c];

                        opt_caster_count++;
                    }
                }

                if (opt_caster_count == 0 && ring_count == 0) skip_volumetric_shadow = true;
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
        eval_polar_winter(mid_pos_sph, vec2(solstice_factor, sun_pole_dot), has_rings, polar_haze_factor, polar_rayleigh_inscatter_boost);


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
            float tj = (float(i) + jitter) / float(steps);

            float s0 = map_t_to_s(t0, s_start, s_end, s_closest, grade_p);
            float s1 = map_t_to_s(t1, s_start, s_end, s_closest, grade_p);
            float current_s = map_t_to_s(tj, s_start, s_end, s_closest, grade_p);
            if (aerial_segment_steps.y > 0) {
                int offset = 0;
                int count = aerial_segment_steps.x;
                float begin = s_start, end = aerial_shadow_span.x;
                if (i >= count) {
                    offset = count;
                    count = aerial_segment_steps.y;
                    begin = aerial_shadow_span.x; end = aerial_shadow_span.y;
                    if (i >= offset + count) {
                        offset += count;
                        count = aerial_segment_steps.z;
                        begin = aerial_shadow_span.y; end = s_end;
                    }
                }
                float local_t = float(i - offset) / float(count);
                s0 = map_t_to_s(local_t, begin, end, clamp(s_closest, begin, end), grade_p);
                s1 = map_t_to_s(local_t + 1.0 / float(count), begin, end, clamp(s_closest, begin, end), grade_p);
                current_s = map_t_to_s(local_t + jitter / float(count), begin, end, clamp(s_closest, begin, end), grade_p);
            }
            float step_size = max(1e-4, s1 - s0);

            vec3 current_pos_sph = cam_local_sph + current_s * ray_dir_sph;
            float sample_len = length(current_pos_sph);
            float altitude = max(0.0, sample_len - u_planet_radius_km);

            float rho_R = exp(-altitude * inv_h_rayleigh);
            float rho_M = exp(-altitude * inv_h_mie) * polar_haze_factor;
            float t_ozone = (altitude - u_ozone_peak_km) * inv_ozone_width;
            float rho_O = exp(-(t_ozone * t_ozone));

            // Extinction uses physical beta_R to prevent artificial limb color fringing
            vec3 step_extinction = beta_R * rho_R + beta_M * rho_M + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;
            vec3 step_transmittance = exp(-step_extinction * step_size);
            vec3 int_factor = (vec3(1.0) - step_transmittance) / max(step_extinction, 1e-6);

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
            if (raymarch_quality == 2 && !skip_volumetric_shadow) {
                sample_shadow = vec3(1.0);

                if (ring_count > 0) {
                    float d = length(ring_A_prime + current_s * ring_B);
                    vec4 s_vec = vec4(current_s);
                    vec4 mask = step(ring_s1_out, s_vec) * step(s_vec, ring_s2_out);

                    if (any(greaterThan(mask, vec4(0.0)))) {
                        vec4 o_min = max(ring_inner, vec4(d) - ring_R_eff);
                        vec4 o_max = min(ring_outer, vec4(d) + ring_R_eff);
                        vec4 valid = step(o_min, o_max) * mask;

                        if (any(greaterThan(valid, vec4(0.0)))) {
                            vec4 r_eff_inv = 1.0 / max(vec4(1e-9), ring_R_eff);
                            vec4 v_min = clamp((o_min - vec4(d)) * r_eff_inv, -1.0, 1.0);
                            vec4 v_max = clamp((o_max - vec4(d)) * r_eff_inv, -1.0, 1.0);
                            vec4 frac = max(vec4(0.0), smoothstep(-1.0, 1.0, v_max) - smoothstep(-1.0, 1.0, v_min));
                            vec4 p_mid = clamp(((o_min + o_max) * 0.5 - ring_inner) / max(vec4(1e-6), ring_outer - ring_inner), 0.0, 1.0);

                            vec4 sh_mult = vec4(1.0);
                            if (valid.x > 0.0) {
                                float a_x = textureLod(u_ring_gradients, vec2(p_mid.x, ring_v_coord.x), 0.0).a;
                                float tau_x = -log(max(1e-6, 1.0 - a_x));
                                sh_mult.x = 1.0 - frac.x * (1.0 - exp(-ring_opac.x * tau_x));
                            }
                            if (valid.y > 0.0) {
                                float a_y = textureLod(u_ring_gradients, vec2(p_mid.y, ring_v_coord.y), 0.0).a;
                                float tau_y = -log(max(1e-6, 1.0 - a_y));
                                sh_mult.y = 1.0 - frac.y * (1.0 - exp(-ring_opac.y * tau_y));
                            }
                            if (valid.z > 0.0) {
                                float a_z = textureLod(u_ring_gradients, vec2(p_mid.z, ring_v_coord.z), 0.0).a;
                                float tau_z = -log(max(1e-6, 1.0 - a_z));
                                sh_mult.z = 1.0 - frac.z * (1.0 - exp(-ring_opac.z * tau_z));
                            }
                            if (valid.w > 0.0) {
                                float a_w = textureLod(u_ring_gradients, vec2(p_mid.w, ring_v_coord.w), 0.0).a;
                                float tau_w = -log(max(1e-6, 1.0 - a_w));
                                sh_mult.w = 1.0 - frac.w * (1.0 - exp(-ring_opac.w * tau_w));
                            }

                            sample_shadow *= sh_mult.x * sh_mult.y * sh_mult.z * sh_mult.w;
                        }
                    }
                }

                if (opt_caster_count > 0) {
                    for (int c = 0; c < 4; c++) {
                        if (c >= opt_caster_count) break;

                        vec2 s_bounds = c_p4[c];
                        if (current_s >= s_bounds.x && current_s <= s_bounds.y) {
                            vec4 p0 = c_p0[c];
                            float perp_sq = p0.x * current_s * current_s + p0.y * current_s + p0.z;
                            if (perp_sq < p0.w) {
                                vec4 p1 = c_p1[c];
                                float gamma = sqrt(max(0.0, perp_sq)) * p1.x;
                                vec4 p2 = c_p2[c];

                                vec3 sh = casterShadowTerm(p2.x, p2.y, gamma, p1.y, p1.z, p2.z, c_p3[c], c_p5[c], c_p3[c].w, p2.w);
                                sample_shadow *= sh;
                            }
                        }
                    }
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

            current_transmittance *= step_transmittance;
            if (all(lessThan(current_transmittance, vec3(1e-6)))) {
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

            vec3 sun_pos_local_0 = (0 < 4) ? u_star_pos_local[0].xyz : ((u_stars_pos_radius[0].xyz - planet_center_render) * u_au_to_km);
            vec3 L0 = (length(sun_pos_local_0) > 1e-6) ? normalize(sun_pos_local_0) : vec3(0.0, 1.0, 0.0);

            for (int j = 0; j < u_num_ring_planes; j++) {
                if ((u_ring_mask & (1u << j)) == 0u) continue;
                vec3 ring_normal = u_ring_normal[j];
                float sun_elev_0 = dot(L0, ring_normal);
                float sun_elev_s = dot(L_dir, ring_normal);
                float rel_hemi = (sun_elev_s * sun_elev_0 >= 0.0) ? 1.0 : -1.0;

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

                float y_prime = frag_elevation * rel_hemi;
                float elev_uv = sign(y_prime) * pow(abs(y_prime), 0.666666667) * 0.5 + 0.5;

                vec2 map_uv = vec2(phi_uv, (float(j) + elev_uv) / 16.0);
                ringshine_irradiance += texture(u_ringshine_map, map_uv).rgb;
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

        final_transmittance = current_transmittance;
    }
    }

    if (aerial_sample_valid && aerial_refinement > 0.0 && aerial_refinement < 1.0) {
        scattered = mix(aerial_S, scattered, aerial_refinement);
        final_transmittance = mix(aerial_T, final_transmittance, aerial_refinement);
    }
    vec3 transmittance = final_transmittance;

    if (u_hdr_enabled) {
        scattered *= u_exposure;
    }

    if (!aerial_sample_valid && !use_sky_view_lut && u_temporal_accum && u_history_valid) {
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
