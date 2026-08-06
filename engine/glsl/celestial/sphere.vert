#version 460 core
in vec3 in_position;
in vec3 in_normal;

layout(std430, binding = 2) buffer AllInstances {
    vec4 instances[];
};

layout(std430, binding = 3) buffer VisibleIndices {
    uint vis_indices[];
};

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
};
uniform float screen_height;
uniform float fov_factor;
uniform float u_refract_max_bend;

out vec3 f_color;
out vec3 f_world_pos;
out vec3 f_normal;
out float f_is_star;
out float f_clip_z;
out float f_brightness_scale;
out float f_subpixel_factor;
flat out vec2 f_center_px;
flat out float f_clamped_min_px;
flat out uvec2 f_caster_mask;
flat out uint f_ring_mask;
flat out vec3 f_planetshine_dir;
flat out vec3 f_planetshine_color;
flat out float f_tex_idx;
flat out float f_rotation_angle;
out vec3 f_local_pos;
flat out vec3 f_pole;
flat out vec3 f_center_pos;
flat out float f_radius;
flat out float f_final_radius;
flat out float f_oblateness;
flat out float f_bounding_radius;

vec3 rotate_about_axis(vec3 v, vec3 axis, float angle) {
    if (abs(angle) < 1e-7) return v;
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

void main() {
    uint inst_idx = vis_indices[gl_InstanceID];
    vec4 f0 = instances[inst_idx * 7 + 0];
    vec4 f1 = instances[inst_idx * 7 + 1];
    vec4 f2 = instances[inst_idx * 7 + 2];
    vec4 f3 = instances[inst_idx * 7 + 3];
    vec4 f4 = instances[inst_idx * 7 + 4];
    vec4 f5 = instances[inst_idx * 7 + 5];
    vec4 f6 = instances[inst_idx * 7 + 6];

    vec3 in_offset = f0.xyz;
    vec3 in_color = vec3(f0.w, f1.x, f1.y);
    float in_radius = f1.z;
    float in_min_size = f1.w;
    float in_is_star = f2.x;
    vec3 in_pole = f2.yzw;
    float in_oblateness = f3.x;
    uvec2 in_caster_mask = uvec2(floatBitsToUint(f3.y), floatBitsToUint(f3.z));
    uint in_ring_mask = floatBitsToUint(f3.w);

    f_planetshine_dir = f4.xyz;
    f_planetshine_color = f5.xyz;
    f_tex_idx = f6.x;
    f_rotation_angle = f6.y;
    f_pole = in_pole;

    f_center_pos = in_offset;
    f_radius = in_radius;
    f_oblateness = in_oblateness;

    f_color = in_color;
    f_caster_mask = in_caster_mask;
    f_ring_mask = in_ring_mask;
    f_is_star = in_is_star;
    float aspect = projection[1][1] / max(1e-6, projection[0][0]);
    float screen_width = screen_height * aspect;

    vec4 center_clip = projection * view * vec4(in_offset, 1.0);
    vec2 center_ndc = center_clip.xy / max(1e-6, center_clip.w);
    f_center_px = (center_ndc * 0.5 + 0.5) * vec2(screen_width, screen_height);

    float dist = length((view * vec4(in_offset, 1.0)).xyz);
    float apparent_px = (in_radius / dist) * screen_height * fov_factor;
    float final_radius = in_radius;
    float brightness_scale = 1.0;

    float clamped_min_px = max(in_min_size, 2.0);
    f_clamped_min_px = clamped_min_px;

    if (apparent_px < clamped_min_px) {
        final_radius = (clamped_min_px * dist) / (screen_height * fov_factor);
        float ratio = apparent_px / clamped_min_px;
        brightness_scale = ratio * ratio;
    }
    f_final_radius = final_radius;
    f_brightness_scale = brightness_scale;
    f_subpixel_factor = smoothstep(2.0, 1.0, apparent_px);

    vec3 pole_n = length(in_pole) > 1e-4 ? normalize(in_pole) : vec3(0.0, 1.0, 0.0);
    vec3 mesh_pos = rotate_about_axis(in_position, pole_n, f_rotation_angle);
    vec3 mesh_norm = rotate_about_axis(in_normal, pole_n, f_rotation_angle);

    f_local_pos = mesh_pos;
    vec3 scaled_pos = mesh_pos;
    vec3 adj_normal = mesh_norm;
    if (in_oblateness > 0.0) {
        float pole_proj = dot(mesh_pos, pole_n);
        scaled_pos -= pole_n * (pole_proj * in_oblateness);
        float f_inv = in_oblateness / (1.0 - in_oblateness);
        adj_normal = normalize(mesh_norm + pole_n * (dot(mesh_norm, pole_n) * f_inv));
    }

    // Expand bounding mesh radius to cover the refracted/ray-traced shape
    float atmo_expand = (u_refract_max_bend > 0.0) ? (dist * tan(u_refract_max_bend) * 1.5 + final_radius * 0.08) : (final_radius * 0.01);
    float bounding_radius = final_radius + atmo_expand;
    f_bounding_radius = bounding_radius;

    vec3 bounding_world_pos = (scaled_pos * bounding_radius) + in_offset;
    f_world_pos = bounding_world_pos;
    f_normal = adj_normal;

    gl_Position = projection * view * vec4(bounding_world_pos, 1.0);
    f_clip_z = gl_Position.w;
}
