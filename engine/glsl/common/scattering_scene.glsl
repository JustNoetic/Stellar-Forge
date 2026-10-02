// Shared scene inputs and endpoint transport for camera-dependent LUT bakes.
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
