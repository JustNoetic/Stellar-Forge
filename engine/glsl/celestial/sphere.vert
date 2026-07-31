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

out vec3 f_color;
out vec3 f_world_pos;
out vec3 f_normal;
out float f_is_star;
out float f_clip_z;
out float f_brightness_scale;
flat out uvec2 f_caster_mask;
flat out uint f_ring_mask;
flat out vec3 f_planetshine_dir;
flat out vec3 f_planetshine_color;
flat out float f_tex_idx;
flat out float f_rotation_angle;
out vec3 f_local_pos;
flat out vec3 f_pole;
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

    f_color = in_color;
    f_caster_mask = in_caster_mask;
    f_ring_mask = in_ring_mask;
    f_is_star = in_is_star;
    float dist = length((view * vec4(in_offset, 1.0)).xyz);
    float apparent_px = (in_radius / dist) * screen_height * fov_factor;
    float final_radius = in_radius;
    float brightness_scale = 1.0;

    if (apparent_px < in_min_size) {
        final_radius = (in_min_size * dist) / (screen_height * fov_factor);
        float ratio = apparent_px / in_min_size;
        brightness_scale = ratio;
    }
    f_brightness_scale = brightness_scale;

    f_local_pos = in_position;
    vec3 scaled_pos = in_position;
    vec3 adj_normal = in_normal;
    if (in_oblateness > 0.0) {
        float pole_proj = dot(in_position, in_pole);
        scaled_pos -= in_pole * (pole_proj * in_oblateness);
        float f_inv = in_oblateness / (1.0 - in_oblateness);
        adj_normal = normalize(in_normal + in_pole * (dot(in_normal, in_pole) * f_inv));
    }

    vec3 world_pos = (scaled_pos * final_radius) + in_offset;
    f_world_pos = world_pos;
    f_normal = adj_normal;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    gl_Position.z = (log2(max(1e-6, u_depth_C * gl_Position.w + 1.0)) / log2(u_depth_C * u_far + 1.0) * 2.0 - 1.0) * gl_Position.w;
    f_clip_z = gl_Position.w;
}
