#version 460

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

#include "common/celestial_scene.glsl"
#include "common/body_surface.glsl"
#include "common/fluid_surface.glsl"

layout(location = 0) out vec3 f_world_pos;
layout(location = 1) flat out uint f_body_idx;
layout(location = 2) out vec3 f_body_offset;

void main() {
    uint idx = gl_InstanceIndex;
    f_body_idx = idx;
    BodyInstanceGPU b = bodies[idx];
    SurfaceFluidGPU f = fluids[idx];
    mat3 rotation = body_surface_rotation(body_pole(b), b.spin_flags.x);
    float radius = (f.cloud_motion.w+f.cloud_shape.y+f.cloud_shape.z)/FLUID_AU_KM*1.006;
    vec3 scale = vec3(1.0, max(1.0-b.pole_oblateness.w, 0.05), 1.0);
    vec3 offset = rotation*(in_position*scale)*radius;
    f_world_pos = b.pos_radius.xyz+offset;
    f_body_offset = offset;
    gl_Position = u_scene.projection*vec4(b.view_center.xyz+mat3(u_scene.view)*offset, 1.0);
}
