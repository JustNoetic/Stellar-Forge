#version 460 core
in vec3 in_position;
in vec3 in_normal;

#define MAX_CASTERS 64
#define MAX_STARS 16
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
uniform vec3 u_body_offset;
uniform vec3 u_camera_pos;
uniform float u_refract_max_bend;
uniform vec3 u_refract_center;
uniform float u_refract_radius;
uniform float u_refract_scale_height;
uniform float u_au_to_km;

// Gravitational Lensing expansion
uniform bool u_grav_lens_enabled;
uniform float u_grav_lens_rs;
uniform vec3 u_grav_lens_center;
uniform float u_grav_lens_radius;
uniform int u_grav_lens_type;

struct RingPlane {
    vec3 color;
    float inner_r;
    float outer_r;
    float opacity;
    float scatter;
    float asymmetry;
    float backscatter;
    int row_idx;
    float is_textured;
    float unlit_factor;
    float saturation;
    float hue_shift;
    float brightness;
    float alpha_boost;
};

#define MAX_RING_PLANES 16
uniform RingPlane u_ring_planes[MAX_RING_PLANES];
uniform int u_num_ring_planes;

out vec3 f_world_pos;
out vec3 f_normal;
out vec3 f_local_pos;
out float f_clip_z;

void main() {
    float r = length(in_position);

    float ring_min_r = 1e9;
    float ring_max_r = 0.0;
    for (int i = 0; i < u_num_ring_planes; i++) {
        ring_min_r = min(ring_min_r, u_ring_planes[i].inner_r);
        ring_max_r = max(ring_max_r, u_ring_planes[i].outer_r);
    }
    if (ring_max_r <= ring_min_r) {
        ring_min_r = r * 0.5;
        ring_max_r = r;
    }

    float dist = length(u_body_offset - u_camera_pos);

    // Expand bounding mesh to cover refracted rays and gravitational lensing.
    // Base expansion: covers ray bending through the host planet atmosphere.
    // Rays passing through the limb bend by up to u_refract_max_bend, shifting
    // the apparent hit point on the ring plane by ~ring_max_r * tan(max_bend).
    float bend_expand = ring_max_r * tan(max(0.0, u_refract_max_bend)) * 2.0;
    float base_expand = (u_refract_max_bend > 0.0) ? max(ring_max_r * 0.25, bend_expand) : (ring_max_r * 0.01);

    // Background body deflection: distant rings seen through a foreground atmosphere
    // can appear shifted across the sky by up to tan(max_bend).
    float bg_expand = 0.0;
    if (u_refract_max_bend > 0.0 && dist > ring_max_r * 2.0) {
        float max_bg_expand = max(0.0, dist * 0.40 - ring_max_r);
        bg_expand = min(dist * tan(u_refract_max_bend) * 2.0, max_bg_expand);
    }

    float atmo_expand = base_expand + bg_expand;

    // Gravitational Lensing expansion for Einstein rings and lensed background arcs
    bool is_lens_bh_host = u_grav_lens_enabled && u_grav_lens_type == 3 && u_grav_lens_rs > 1e-6
        && distance(u_body_offset, u_grav_lens_center) < 1e-6;
    float grav_expand = 0.0;
    if (u_grav_lens_enabled && u_grav_lens_rs > 1e-6 && !is_lens_bh_host) {
        float d_lens_km = length(u_grav_lens_center - u_camera_pos) * u_au_to_km;
        float d_body_km = dist * u_au_to_km;
        if (d_body_km > d_lens_km * 0.7) {
            float d_ls_km = max(1.0, d_body_km - d_lens_km);
            float theta_E = sqrt(max(0.0, 2.0 * u_grav_lens_rs * d_ls_km / (max(1.0, d_lens_km) * max(1.0, d_body_km))));
            float max_ang = min(0.6, theta_E * 2.5 + (2.0 * u_grav_lens_rs / max(1.0, u_grav_lens_radius)));
            grav_expand = dist * tan(max_ang) * 1.25;
        }
    }

    float total_expand = max(atmo_expand, grav_expand);

    float span = max(1e-6, ring_max_r - ring_min_r);
    float p = clamp((r - ring_min_r) / span, 0.0, 1.0);
    float r_prime_min = max(0.0, ring_min_r - total_expand);
    float r_prime_max = ring_max_r + total_expand;
    float r_prime = mix(r_prime_min, r_prime_max, p);
    vec3 expanded_pos = (r > 1e-7) ? (in_position * (r_prime / r)) : in_position;

    vec3 world_pos = expanded_pos + u_body_offset;
    f_world_pos = world_pos;
    f_normal = in_normal;
    f_local_pos = expanded_pos;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    f_clip_z = gl_Position.w;
}
