#version 460

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

#include "common/celestial_scene.glsl"
#include "common/body_surface.glsl"
#include "common/refraction.glsl"

layout(location = 0) out vec3 f_world_pos;
layout(location = 1) out vec3 f_local_pos;
layout(location = 2) out vec3 f_normal;
layout(location = 3) out float f_subpixel_factor;
layout(location = 4) flat out uint f_body_idx;
layout(location = 7) out vec3 f_base_chord;
layout(location = 8) out vec3 f_apparent_pos;

void main() {
    uint idx = gl_InstanceIndex;
    f_body_idx = idx;
    f_base_chord = in_position;
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
        f_apparent_pos = vec3(0.0);
        return;
    }

    mat3 rotation = body_surface_rotation(pole, spin);
    f_local_pos = in_position;

    vec3 scale = vec3(1.0, max(1.0-obl, 0.05), 1.0);
    vec3 scaled_pos = rotation*(in_position*scale);
    vec3 adj_norm = rotation*normalize(in_normal/scale);

    vec3 world_pos = center + scaled_pos * radius;
    f_world_pos = world_pos;
    f_normal = adj_norm;
    vec3 apparent = refract_position(world_pos);
    f_apparent_pos = apparent;
    gl_Position = u_scene.view_proj * vec4(apparent, 1.0);
}
