#version 460

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

#include "common/celestial_scene.glsl"

layout(location = 0) out vec3 f_world_pos;
layout(location = 1) out vec3 f_local_pos;
layout(location = 2) out vec3 f_normal;
layout(location = 3) out float f_subpixel_factor;
layout(location = 4) flat out uint f_body_idx;

vec3 rotate_about_axis(vec3 v, vec3 axis, float angle) {
    if (abs(angle) < 1e-7) return v;
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

void main() {
    uint idx = gl_InstanceIndex;
    f_body_idx = idx;
    BodyInstanceGPU b = bodies[idx];

    vec3 center = b.pos_radius.xyz;
    float radius = b.pos_radius.w;
    vec3 pole = length(b.pole_oblateness.xyz) > 1e-4 ? normalize(b.pole_oblateness.xyz) : vec3(0.0, 1.0, 0.0);
    float obl = b.pole_oblateness.w;
    float spin = b.spin_flags.x;
    bool is_bh = b.spin_flags.z > 0.5;

    float weight = is_bh ? 1.0 : mesh_weight(apparent_radius_px(b.pos_radius));
    f_subpixel_factor = 1.0 - weight;
    if (weight <= 0.0 || b.surface_flags.x != 0u) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        f_world_pos = vec3(0.0);
        f_local_pos = vec3(0.0);
        f_normal = vec3(0.0, 1.0, 0.0);
        return;
    }

    vec3 mesh_pos = rotate_about_axis(in_position, pole, spin);
    vec3 mesh_norm = rotate_about_axis(in_normal, pole, spin);
    f_local_pos = in_position;

    vec3 scaled_pos = mesh_pos;
    vec3 adj_norm = mesh_norm;
    if (obl > 0.0) {
        float pole_proj = dot(mesh_pos, pole);
        scaled_pos -= pole * (pole_proj * obl);
        float f_inv = obl / max(1.0 - obl, 1e-4);
        adj_norm = normalize(mesh_norm + pole * (dot(mesh_norm, pole) * f_inv));
    }

    vec3 world_pos = center + scaled_pos * radius;
    f_world_pos = world_pos;
    f_normal = adj_norm;
    gl_Position = u_scene.view_proj * vec4(world_pos, 1.0);
}
