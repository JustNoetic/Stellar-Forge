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

vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float delta = b * b - c;
    if (delta < 0.0) return vec2(1e10, -1e10);
    float sq = sqrt(delta);
    return vec2(-b - sq, -b + sq);
}

#include "common/sun_terminator.glsl"
#include "common/scattering_segment.glsl"
#include "common/scattering_shine.glsl"
// Camera-specific projection of cached fields; no marching or polar winter.
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

    vec3 a = ray_origin + s_start * V, b = ray_origin + s_end * V;
    vec3 T = endpoint_transmittance(a, b), star_scatter[4], total = vec3(0.0);
    for (int st = 0; st < 4; ++st) {
        star_scatter[st] = vec3(0.0);
        if (st < u_num_stars && dot(u_star_dir_sph_eff[st].xyz, u_star_dir_sph_eff[st].xyz) > 1e-8) {
            star_scatter[st] = endpoint_radiance_star(a, b, normalize(u_star_dir_sph_eff[st].xyz), T,
                st, u_star_pos_local[st].w) * u_star_color_irrad[st].rgb;
            total += star_scatter[st];
        }
    }
    total += endpoint_secondary_light(a, b, T, instances[u_body_idx*7+4].xyz,
        instances[u_body_idx*7+5].xyz, floatBitsToUint(instances[u_body_idx*7+3].w));
    out_color = vec4(total, dot(T, vec3(1.0/3.0)));
    out_transmittance = vec4(T, 1.0);
    out_star0 = vec4(star_scatter[0], 1.0); out_star1 = vec4(star_scatter[1], 1.0);
    out_star2 = vec4(star_scatter[2], 1.0); out_star3 = vec4(star_scatter[3], 1.0);
}
