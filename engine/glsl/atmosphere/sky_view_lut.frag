#version 460 core
#define PI 3.14159265358979323846

in vec2 f_uv;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_transmittance;
// Per-star in-scatter slices (locations 2-5). atmo.frag's ring-shadow slicing
// normalizes each star's shadow deficit against that star's OWN baked light,
// so one star's ring shadow can never darken another star's illumination.
layout(location = 2) out vec4 out_star0;
layout(location = 3) out vec4 out_star1;
layout(location = 4) out vec4 out_star2;
layout(location = 5) out vec4 out_star3;

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
    vec3  u_mie_albedo;
    float u_refractivity;
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

// Scene UBO (bound globally at binding 1 by app.py): supplies the star count
// and per-star pole/oblateness for multi-star eclipse evaluation. Layout must
// match the SceneData block in atmo.frag exactly.
#define MAX_STARS 16
#define MAX_CASTERS 64
layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = directional r_minor
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

// Instance SSBO (globally bound at binding 2 by app.py): per-body planetshine
// direction/color and ring mask bitfield — same layout as atmo.frag reads.
layout(std430, binding = 2) buffer AllInstances {
    vec4 instances[];
};

uniform vec3 u_cam_pos; // Camera position in planet local frame (km)
uniform vec3 u_sun_dir; // Sunlight direction in planet local frame (normalized)

uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;

// Stage-2 shine baking: planetshine/moonshine (second light from the instance
// SSBO) and host-planet ringshine (irradiance map), matching Mode 1/2 exactly.
#define MAX_RING_PLANES 16
uniform sampler2D u_ringshine_map;
uniform int  u_num_ring_planes;
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform bool u_planetshine_enabled;
uniform bool u_ringshine_enabled;

// Ring uniforms for external ring shadow evaluation (e.g. Saturn's rings shadowing Titan)
uniform sampler2D u_ring_gradients;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec4 u_ring_params[MAX_RING_PLANES];
uniform uint u_ring_coplanar_mask[16];
uniform int  u_num_steps;
uniform int  u_atmo_shadow_method;

#include "common/sun_terminator.glsl"
// Analytical caster eclipses (moons/planets), byte-identical to the Mode 1/2
// raymarchers: get_oblate_radius / casterShadowTerm / compute_caster_shadow.
#include "common/caster_shadow.glsl"

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

// Evaluates analytical penumbra/umbra shadow occlusion for external rings (belonging
// to parent/neighbor bodies, e.g. Saturn's rings casting shadow onto Titan).
// Circumplanetary (own) rings are skipped here because Mode 3 slices own-ring shadows
// per-pixel along the screen ray in atmo.frag.
vec3 compute_external_ring_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, float star_radius, uint r_mask) {
    vec3 shadow = vec3(1.0);
    if (u_num_ring_planes == 0 || r_mask == 0u) return shadow;

    uint processed_mask = 0u;
    for (int k = 0; k < u_num_ring_planes; k++) {
        if ((r_mask & (1u << k)) == 0u) continue;
        if ((processed_mask & (1u << k)) != 0u) continue;

        // Skip circumplanetary (own) rings: distance to current body center < 1e-4 AU
        if (distance(u_ring_center[k], u_body_offset) < 1e-4) continue;

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

            float f_max = (v_max * sqrt(max(0.0, 1.0 - v_max * v_max)) + asin(v_max)) / PI + 0.5;
            float f_min = (v_min * sqrt(max(0.0, 1.0 - v_min * v_min)) + asin(v_min)) / PI + 0.5;
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
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float delta = b * b - c;
    if (delta < 0.0) return vec2(1e10, -1e10);
    float sq = sqrt(delta);
    return vec2(-b - sq, -b + sq);
}

vec3 get_transmittance(float r, float cos_theta) {
    float h_norm = clamp((r - u_planet_radius_km) / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float v = sqrt(h_norm);
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

void main() {
    float D = length(u_cam_pos);
    vec3 u_zenith = u_cam_pos / max(D, 1e-6);
    float D_clamped = max(D, u_planet_radius_km);
    bool cam_inside = (D <= u_atmo_radius_km);
    vec3 L_sun = normalize(u_sun_dir);

    // Build continuous orthonormal basis aligned with local Zenith and Sun Azimuth
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

    // Map f_uv.x in [0, 1] to azimuth phi in [0, 2*PI]
    float phi = f_uv.x * 2.0 * PI;

    vec3 ray_origin;
    vec3 V;
    float s_start = 0.0;
    float s_end = 0.0;
    bool hits_ground = false;

    if (cam_inside) {
        // Piecewise non-linear elevation mapping centered at horizon (Hillaire 2020)
        float sin_horizon = clamp(u_planet_radius_km / D_clamped, 0.0, 1.0);
        float theta_horizon = PI - asin(sin_horizon);

        float theta = 0.0;
        if (f_uv.y <= 0.5) {
            // Ground disk: [theta_horizon, PI]
            // Row 0 is Nadir (theta = PI), row 127 is Horizon (theta = theta_horizon)
            float t = (0.5 - f_uv.y) / 0.5;
            theta = theta_horizon + (t * t) * (PI - theta_horizon);
            hits_ground = true;
        } else {
            // Sky dome: [0, theta_horizon]
            // Row 128 is Horizon (theta = theta_horizon), row 255 is Zenith (theta = 0)
            float t = (f_uv.y - 0.5) / 0.5;
            theta = theta_horizon * (1.0 - t * t);
            hits_ground = false;
        }

        V = normalize(sin(theta) * (cos(phi) * x_basis + sin(phi) * y_basis) + cos(theta) * u_zenith);
        ray_origin = u_zenith * D_clamped;

        vec2 t_atmo = raySphereIntersect(ray_origin, V, u_atmo_radius_km);
        if (t_atmo.y < 0.0) {
            out_color = vec4(0.0, 0.0, 0.0, 1.0);
            out_transmittance = vec4(1.0);
            out_star0 = vec4(0.0); out_star1 = vec4(0.0); out_star2 = vec4(0.0); out_star3 = vec4(0.0);
            return;
        }

        s_start = max(0.0, t_atmo.x);
        s_end = t_atmo.y;

        if (hits_ground) {
            vec2 t_planet = raySphereIntersect(ray_origin, V, u_planet_radius_km);
            if (t_planet.x > 0.0) {
                s_end = min(s_end, t_planet.x);
            } else {
                s_end = 0.0;
            }
        }
    } else {
        // Space observer: looking towards planet
        // Exact mapping from impact parameter r_ca and azimuth phi
        float r_ca;
        if (f_uv.y <= 0.5) {
            // Ground disk: impact parameter r_ca in [0, R_planet]
            float t = (0.5 - f_uv.y) / 0.5;
            r_ca = u_planet_radius_km * max(0.0, 1.0 - t * t);
            hits_ground = true;
        } else {
            // Atmosphere limb: impact parameter r_ca in [R_planet, R_atmo]
            float t = (f_uv.y - 0.5) / 0.5;
            r_ca = u_planet_radius_km + (t * t) * (u_atmo_radius_km - u_planet_radius_km);
            hits_ground = false;
        }

        vec3 e_phi = cos(phi) * x_basis + sin(phi) * y_basis;
        float sin_alpha = clamp(r_ca / max(D, u_atmo_radius_km), 0.0, 1.0);
        float cos_alpha = sqrt(max(0.0, 1.0 - sin_alpha * sin_alpha));

        V = normalize(sin_alpha * e_phi - cos_alpha * u_zenith);
        ray_origin = r_ca * cos_alpha * e_phi + r_ca * sin_alpha * u_zenith;

        float r_ca_sq = r_ca * r_ca;
        s_start = -sqrt(max(0.0, u_atmo_radius_km * u_atmo_radius_km - r_ca_sq));

        if (hits_ground) {
            s_end = -sqrt(max(0.0, u_planet_radius_km * u_planet_radius_km - r_ca_sq));
        } else {
            s_end = +sqrt(max(0.0, u_atmo_radius_km * u_atmo_radius_km - r_ca_sq));
        }
    }

    if (s_start >= s_end) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        out_transmittance = vec4(1.0);
        out_star0 = vec4(0.0); out_star1 = vec4(0.0); out_star2 = vec4(0.0); out_star3 = vec4(0.0);
        return;
    }

    // Physical scattering parameters
    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M);
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;

    float inv_h_rayleigh = (u_precomp_opt.x > 1e-6) ? u_precomp_opt.x : (1.0 / max(u_h_rayleigh, 1e-3));
    float inv_h_mie = (u_precomp_opt.y > 1e-6) ? u_precomp_opt.y : (1.0 / max(u_h_mie, 1e-3));
    float inv_ozone_width = (u_precomp_opt.z > 1e-6) ? u_precomp_opt.z : (1.0 / max(u_ozone_width_km, 1e-3));

    int num_steps = (u_num_steps > 0) ? clamp(u_num_steps, 4, 64) : 32;
    float ds = (s_end - s_start) / float(num_steps);

    // --- Per-star precompute (constant per texel) ---
    // The Sky-View LUT is view-direction parameterized, so any number of light
    // sources can be baked in here with zero extra cost in the per-pixel
    // atmo.frag pass. Only the 4 precomputed SSBO star slots are baked
    // (matches app.py's staging); more distant companions are negligible.
    int n_stars = clamp(u_num_stars, 1, 4);

    float g = clamp(u_mie_g, 0.0, 0.88);
    float c1 = (u_precomp_mie.x > 1e-6) ? u_precomp_mie.x : ((3.0 / (8.0 * PI)) * ((1.0 - g * g) / (2.0 + g * g)));
    float c2 = (u_precomp_mie.y > 1e-6) ? u_precomp_mie.y : (1.0 + g * g);
    float c3 = (u_precomp_mie.z > 1e-6) ? u_precomp_mie.z : (2.0 * g);

    vec3  star_L_sph[4];    // sunlight dir, planet-local spherical frame
    vec3  star_L_render[4]; // sunlight dir, render frame (for eclipse cones)
    vec3  star_int[4];      // rgb = star_color * irradiance * atmo_sun_intensity
    float star_phase_R[4];  // Rayleigh phase for this view direction
    float star_phase_M[4];  // Cornette-Shanks phase for this view direction
    vec4  star_geo[4];      // x: eff_star_rad, y: cos_sun_eff, z: dist_star_au, w: star_radius_au
    vec4  star_pole_obl[4]; // xyz: pole, w: directional r_minor

    for (int s = 0; s < n_stars; s++) {
        vec3 L_sph = u_star_dir_sph_eff[s].xyz;
        float l_len = length(L_sph);
        star_L_sph[s] = (l_len > 1e-4) ? L_sph / l_len : ((s == 0) ? L_sun : vec3(0.0, 1.0, 0.0));
        star_L_render[s] = normalize(u_star_pos_local[s].xyz);
        vec3 irrad = u_star_color_irrad[s].rgb;
        star_int[s] = (length(irrad) > 1e-6) ? irrad : ((s == 0) ? vec3(1.0) : vec3(0.0));
        star_geo[s] = vec4(u_star_pos_local[s].w, u_star_dir_sph_eff[s].w,
                           u_star_solstice[s].z, u_star_solstice[s].w);
        star_pole_obl[s] = u_stars_poles_obl[s];

        float cos_vs = dot(V, star_L_sph[s]);
        star_phase_R[s] = (3.0 / (16.0 * PI)) * (1.0 + cos_vs * cos_vs);
        star_phase_M[s] = c1 * (1.0 + cos_vs * cos_vs) / pow(max(1e-4, c2 - c3 * cos_vs), 1.5);
    }

    vec3 star_scatter[4]; // per-star in-scatter totals (single + multi), for out_starN
    for (int s = 0; s < 4; s++) star_scatter[s] = vec3(0.0);

    // --- Planetshine / moonshine (secondary bounce light from the instance
    // SSBO): one extra directional light with a terminator-free visibility
    // smoothstep, matching the Mode 1/2 per-step accumulation (which is gated
    // to the star-0 iteration, i.e. applied once). ---
    vec3 ps_dir = instances[u_body_idx * 7 + 4].xyz;
    vec3 ps_color = instances[u_body_idx * 7 + 5].xyz;
    uint ring_mask_bits = floatBitsToUint(instances[u_body_idx * 7 + 3].w);
    bool ps_active = (u_planetshine_enabled || u_ringshine_enabled)
                     && dot(ps_color, ps_color) > 1e-12 && length(ps_dir) > 1e-6;
    vec3 ps_dir_n = ps_active ? normalize(ps_dir) : vec3(0.0, 1.0, 0.0);
    vec3 total_rayleigh_ps = vec3(0.0);
    vec3 total_mie_ps = vec3(0.0);
    vec3 total_ms_ps = vec3(0.0);

    // --- Host-planet ringshine: star-independent optical accumulators; the
    // per-star irradiance map lookup happens once after the march. ---
    bool rs_active = u_ringshine_enabled && u_num_ring_planes > 0 && ring_mask_bits != 0u;
    vec3 ring_normal_vec = (length(u_pole_obl.xyz) > 1e-4) ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    if (rs_active) {
        for (int k = 0; k < u_num_ring_planes; k++) {
            if ((ring_mask_bits & (1u << k)) != 0u) { ring_normal_vec = u_ring_normal[k]; break; }
        }
    }
    vec3 total_rayleigh_rs = vec3(0.0);
    vec3 total_mie_rs = vec3(0.0);
    vec3 total_ms_rs = vec3(0.0);

    // --- Pre-loop ray-cone caster culling ---
    uint star_caster_mask[4];
    for (int s = 0; s < 4; s++) star_caster_mask[s] = 0u;

    vec3 O_ren = fromSphericalSpace(ray_origin, u_pole_obl) / u_au_to_km + u_body_offset;
    vec3 V_ren = fromSphericalSpace(V, u_pole_obl) / u_au_to_km;

    if (u_num_active_casters > 0) {
        for (int st = 0; st < n_stars; st++) {
            vec3 L_dir = star_L_render[st];
            float dist_to_star = star_geo[st].z;
            float star_radius = star_geo[st].w;
            float inv_dist_star = 1.0 / max(dist_to_star, 1e-6);

            for (int c = 0; c < u_num_active_casters; c++) {
                vec3 caster_pos = u_active_casters[c].xyz;
                float caster_r = u_active_casters[c].w;
                float atmo_h = u_active_caster_atmos[c].w;
                float eff_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);

                vec3 D0 = caster_pos - O_ren;
                float t0 = dot(D0, L_dir);
                float tv = dot(V_ren, L_dir);

                // Check if caster is entirely behind chord along L_dir
                float t_max_chord = max(t0 - s_start * tv, t0 - s_end * tv);
                if (t_max_chord < 0.0) continue;

                vec3 A = D0 - t0 * L_dir;
                vec3 B = -(V_ren - tv * L_dir);
                float qa = dot(B, B);
                float qb = 2.0 * dot(A, B);
                float qc = dot(A, A);

                float s_closest = (qa > 1e-12) ? clamp(-qb / (2.0 * qa), s_start, s_end) : s_start;
                float min_perp_sq = max(0.0, qa * s_closest * s_closest + qb * s_closest + qc);

                // Conservative penumbra bound: maximum distance to caster along chord
                float d0_sq = dot(D0 - s_start * V_ren, D0 - s_start * V_ren);
                float d1_sq = dot(D0 - s_end * V_ren, D0 - s_end * V_ren);
                float max_dist_to_caster = sqrt(max(d0_sq, d1_sq));
                float r_pen_max = (eff_r + max_dist_to_caster * (star_radius * inv_dist_star)) * 1.15 + 1e-6;

                if (min_perp_sq <= r_pen_max * r_pen_max) {
                    star_caster_mask[st] |= (1u << c);
                }
            }
        }
    }

    // --- Pre-loop external ring shadow culling ---
    uint star_ring_mask[4];
    for (int s = 0; s < 4; s++) star_ring_mask[s] = 0u;

    if (u_num_ring_planes > 0 && ring_mask_bits != 0u) {
        for (int st = 0; st < n_stars; st++) {
            vec3 L_dir = star_L_render[st];
            float dist_to_star = star_geo[st].z;
            float star_radius = star_geo[st].w;

            for (int k = 0; k < u_num_ring_planes; k++) {
                if ((ring_mask_bits & (1u << k)) == 0u) continue;
                if (distance(u_ring_center[k], u_body_offset) < 1e-4) continue; // circumplanetary handled by slicing

                vec3 plane_normal = u_ring_normal[k];
                vec3 plane_center = u_ring_center[k];
                float denom = dot(L_dir, plane_normal);
                if (abs(denom) < 1e-8) continue;

                float h0 = dot(plane_center - O_ren, plane_normal);
                float hv = -dot(V_ren, plane_normal);

                float t0 = (h0 + s_start * hv) / denom;
                float t1 = (h0 + s_end * hv) / denom;
                if (max(t0, t1) <= 0.0 || min(t0, t1) >= dist_to_star) continue;

                vec3 H0 = (O_ren - plane_center) + (h0 / denom) * L_dir;
                vec3 Hv = V_ren + (hv / denom) * L_dir;
                float qa = dot(Hv, Hv);
                float qb = 2.0 * dot(H0, Hv);
                float s_c = (qa > 1e-12) ? clamp(-qb / (2.0 * qa), s_start, s_end) : s_start;
                float min_d_sq = max(0.0, qa * s_c * s_c + qb * s_c + dot(H0, H0));
                float max_d_sq = max(dot(H0 + s_start * Hv, H0 + s_start * Hv),
                                     dot(H0 + s_end * Hv, H0 + s_end * Hv));

                uint coplanar_mask = u_ring_coplanar_mask[k];
                float min_inner_r = u_ring_params[k].x;
                float max_outer_r = u_ring_params[k].y;
                for (int r_i = 0; r_i < u_num_ring_planes; r_i++) {
                    if ((coplanar_mask & (1u << r_i)) != 0u) {
                        min_inner_r = min(min_inner_r, u_ring_params[r_i].x);
                        max_outer_r = max(max_outer_r, u_ring_params[r_i].y);
                    }
                }

                float max_t = max(t0, t1);
                float max_Reff = star_radius * max_t / max(dist_to_star, 1e-6) * 4.0; // conservative dilation

                float outer_bound = max_outer_r + max_Reff;
                float inner_bound = max(0.0, min_inner_r - max_Reff);

                if (min_d_sq <= outer_bound * outer_bound && max_d_sq >= inner_bound * inner_bound) {
                    star_ring_mask[st] |= coplanar_mask;
                }
            }
        }
    }

    vec3 total_scatter = vec3(0.0); // single scattering, all stars, phases applied
    vec3 total_ms = vec3(0.0);      // multi scattering, all stars
    vec3 current_transmittance = vec3(1.0);
    float has_rings = (ring_mask_bits != 0u) ? 1.0 : 0.0;

    for (int i = 0; i < num_steps; i++) {
        float s = s_start + (float(i) + 0.5) * ds;
        vec3 P = ray_origin + s * V;
        float r = length(P);
        float altitude = max(0.0, r - u_planet_radius_km);

        float polar_haze_factor0;
        vec3 polar_rayleigh_boost0;
        eval_polar_winter(P, u_star_solstice[0].xy, has_rings, polar_haze_factor0, polar_rayleigh_boost0);

        float rho_R = exp(-altitude * inv_h_rayleigh);
        float rho_M = exp(-altitude * inv_h_mie) * polar_haze_factor0;
        float t_ozone = (altitude - u_ozone_peak_km) * inv_ozone_width;
        float rho_O = exp(-(t_ozone * t_ozone));

        vec3 step_extinction = beta_R * rho_R + beta_M * rho_M + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;
        vec3 step_transmittance = exp(-step_extinction * ds);
        vec3 int_factor = (vec3(1.0) - step_transmittance) / max(step_extinction, vec3(1e-6));

        vec3 Pn = P / max(r, 1e-6);
        float sin_planet = u_planet_radius_km / max(r, u_planet_radius_km + 0.01);
        float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));

        // Parcel position in render frame (AU) for eclipse cone evaluation.
        // Convert from planet-local spherical space back to true Cartesian coordinates.
        vec3 P_cart = fromSphericalSpace(P, u_pole_obl);
        vec3 P_render = P_cart / u_au_to_km + u_body_offset;

        float h_norm = clamp(altitude / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
        float ms_v = sqrt(h_norm);

        vec3 caster_shadow_0 = vec3(1.0);
        for (int st = 0; st < n_stars; st++) {
            vec3 polar_rayleigh_boost_st = polar_rayleigh_boost0;
            if (st > 0) {
                float dummy_haze;
                eval_polar_winter(P, u_star_solstice[st].xy, has_rings, dummy_haze, polar_rayleigh_boost_st);
            }

            float light_cos_theta = dot(Pn, star_L_sph[st]);

            // Terminator: stellar-disc rise/set with refraction-extended penumbra
            // (eff_star_rad = sin_star + max_bend), shared with the Mode 1/2 raymarchers.
            vec2 term = sun_terminator(light_cos_theta, sin_planet, cos_planet,
                                       star_geo[st].x, star_geo[st].y);
            float vis_fraction = term.x;

            // Analytical moon/planet caster eclipses with atmospheric refraction
            // ring + Danjon tinting, identical to the Mode 1/2 compute_shadow.
            vec3 shadow = vec3(1.0);
            if (u_atmo_shadow_method != 2 && star_caster_mask[st] != 0u) {
                shadow = compute_caster_shadow(P_render, star_L_render[st],
                                               star_geo[st].z, star_geo[st].w,
                                               star_pole_obl[st].w, star_pole_obl[st].xyz,
                                               star_caster_mask[st]);
            }
            if (st == 0) {
                caster_shadow_0 = shadow;
            }
            if (star_ring_mask[st] != 0u) {
                shadow *= compute_external_ring_shadow(P_render, star_L_render[st],
                                                       star_geo[st].z, star_geo[st].w,
                                                       star_ring_mask[st]);
            }

            vec3 s_single = vec3(0.0);
            if (vis_fraction > 1e-4) {
                vec3 trans_to_sun = get_transmittance(r, term.y);
                vec3 sample_attenuation = current_transmittance * trans_to_sun * vis_fraction * shadow * int_factor;
                s_single = star_int[st] * (star_phase_R[st] * beta_R * (rho_R * polar_rayleigh_boost_st)
                                         + star_phase_M[st] * beta_M * rho_M) * sample_attenuation;
                total_scatter += s_single;
            }

            // Multi-scatter, attenuated by the same eclipse shadow (ms_shadow in Mode 1/2).
            float ms_u = 0.5 + 0.5 * sign(light_cos_theta) * sqrt(abs(light_cos_theta));
            vec3 psi = textureLod(u_multi_scatter_lut, vec2(ms_u, ms_v), 0.0).rgb;
            vec3 s_ms = star_int[st] * (beta_R * rho_R + beta_M * rho_M) * psi * current_transmittance * shadow * int_factor;
            total_ms += s_ms;
            star_scatter[st] += s_single + s_ms;
        }

        // Planetshine/moonshine second-light accumulation (once per step, like
        // the Mode 1/2 star-0-gated block).
        if (ps_active) {
            float light_cos_ps = dot(Pn, ps_dir_n);
            float vis_ps = smoothstep(-cos_planet - 0.05, -cos_planet + 0.05, light_cos_ps);
            vec3 trans_to_ps = get_transmittance(r, light_cos_ps);
            vec3 ps_attenuation = current_transmittance * trans_to_ps * vis_ps * caster_shadow_0;
            total_rayleigh_ps += rho_R * ps_attenuation * int_factor;
            total_mie_ps      += rho_M * ps_attenuation * int_factor;
            float ms_u_ps = 0.5 + 0.5 * sign(light_cos_ps) * sqrt(abs(light_cos_ps));
            vec3 psi_ps = textureLod(u_multi_scatter_lut, vec2(ms_u_ps, ms_v), 0.0).rgb;
            total_ms_ps += (beta_R * rho_R + beta_M * rho_M) * psi_ps * ps_attenuation * int_factor;
        }

        // Ringshine optical accumulators (star-independent, Mode 1/2 parity).
        if (rs_active) {
            vec3 rs_attenuation = current_transmittance;
            total_rayleigh_rs += rho_R * rs_attenuation * int_factor;
            total_mie_rs      += rho_M * rs_attenuation * int_factor;
            float rs_elev = dot(Pn, ring_normal_vec);
            float ring_cos_zenith = clamp(sqrt(max(0.0, 1.0 - rs_elev * rs_elev)), 0.0, 1.0);
            float ms_u_rs = 0.5 + 0.5 * sqrt(ring_cos_zenith);
            vec3 psi_rs = textureLod(u_multi_scatter_lut, vec2(ms_u_rs, ms_v), 0.0).rgb;
            total_ms_rs += (beta_R * rho_R + beta_M * rho_M) * psi_rs * rs_attenuation * int_factor;
        }

        current_transmittance *= step_transmittance;
        if (all(lessThan(current_transmittance, vec3(1e-5)))) break;
    }

    vec3 scattered = total_scatter + total_ms;

    // Planetshine combine (Mode 1/2 parity: atmo_sun_intensity = u_sun_intensity * PI).
    if (ps_active) {
        float cos_theta_ps = dot(V, ps_dir_n);
        float phase_R_ps = (3.0 / (16.0 * PI)) * (1.0 + cos_theta_ps * cos_theta_ps);
        float phase_M_ps = c1 * (1.0 + cos_theta_ps * cos_theta_ps) / pow(max(1e-4, c2 - c3 * cos_theta_ps), 1.5);
        float atmo_sun_intensity = u_sun_intensity * PI;
        scattered += ps_color * atmo_sun_intensity * (
            phase_R_ps * beta_R * total_rayleigh_ps +
            phase_M_ps * beta_M * total_mie_ps +
            total_ms_ps
        );
    }

    // Ringshine: per-star irradiance map lookup at the ray midpoint, with the
    // shared star-independent optical accumulators (Mode 1/2 parity).
    if (rs_active) {
        vec3 P_mid = ray_origin + 0.5 * (s_start + s_end) * V;
        vec3 P_dir = normalize(P_mid);
        float ambient_phase = 1.0 / (4.0 * PI);
        float ambient_phase_M = ambient_phase * (1.0 / max(0.15, 1.0 - c3 * 0.5));
        for (int st = 0; st < n_stars; st++) {
            vec3 L_dir = star_L_render[st];
            vec3 ringshine_irradiance = vec3(0.0);
            for (int j = 0; j < u_num_ring_planes; j++) {
                if ((ring_mask_bits & (1u << j)) == 0u) continue;
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
                ringshine_irradiance += textureLod(u_ringshine_map, map_uv, 0.0).rgb;
            }
            // 1/PI converts incoming irradiance map to ambient field; solar
            // elevation is already resolved inside ringshine_map.frag.
            ringshine_irradiance *= 0.318309886;
            scattered += star_int[st] * ringshine_irradiance * (
                ambient_phase * beta_R * total_rayleigh_rs +
                ambient_phase_M * beta_M * total_mie_rs +
                total_ms_rs
            );
        }
    }

    float mean_trans = dot(current_transmittance, vec3(0.333333));
    out_color = vec4(scattered, clamp(mean_trans, 0.0, 1.0));
    out_transmittance = vec4(clamp(current_transmittance, vec3(0.0), vec3(1.0)), 1.0);
    // Per-star slices contain star-owned light only (no planetshine/ringshine):
    // the ring-shadow slicing normalizes each star's shadow against these.
    out_star0 = vec4(star_scatter[0], 1.0);
    out_star1 = vec4(star_scatter[1], 1.0);
    out_star2 = vec4(star_scatter[2], 1.0);
    out_star3 = vec4(star_scatter[3], 1.0);
}
