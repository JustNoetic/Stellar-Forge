#version 460

layout(location = 0) in vec3 in_polar;

layout(location = 0) out vec3 f_local_pos;
layout(location = 1) out vec3 f_normal;
layout(location = 2) flat out uint f_host_idx;

#include "common/celestial_scene.glsl"
#include "common/surface_ring_shadow.glsl"
#include "common/ring_push.glsl"

void main() {
    uint h = u_push.host_index;
    float r = mix(ring_hosts[h].center_min_r.w, ring_hosts[h].pole_max_r.w, in_polar.x);
    vec3 local_pos = r * (in_polar.y * ring_hosts[h].tangent_radius.xyz + in_polar.z * ring_hosts[h].bitangent_oblateness.xyz);
    vec3 world_pos = ring_hosts[h].center_min_r.xyz + local_pos;

    f_local_pos = local_pos;
    f_normal = ring_hosts[h].pole_max_r.xyz;
    f_host_idx = h;
    gl_Position = u_scene.view_proj * vec4(world_pos, 1.0);
}
