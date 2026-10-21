#version 460

layout(location = 0) in vec3 in_polar;

layout(location = 0) out vec3 f_local_pos;
layout(location = 1) out vec3 f_normal;
layout(location = 2) flat out uint f_host_idx;

#include "common/celestial_scene.glsl"
#include "common/ring_hosts.glsl"
#include "common/ring_push.glsl"
#include "common/refraction.glsl"

void main() {
    uint h = u_push.host_index;
    float inner = ring_hosts[h].center_min_r.w, outer = ring_hosts[h].pole_max_r.w;
    if (refraction_active()) {
        float distance = length(ring_hosts[h].center_min_r.xyz);
        float bend = tan(min(u_refraction.profile.w, 0.6));
        float expansion = max(outer*0.25, outer*bend*2.0);
        if (distance > 2.0*outer) expansion += min(distance*bend*2.0, max(0.0, distance*0.4-outer));
        inner = max(0.0, inner-expansion);
        outer += expansion;
    }
    float r = mix(inner, outer, in_polar.x);
    vec3 local_pos = r * (in_polar.y * ring_hosts[h].tangent_radius.xyz + in_polar.z * ring_hosts[h].bitangent_oblateness.xyz);
    vec3 world_pos = ring_hosts[h].center_min_r.xyz + local_pos;

    f_local_pos = local_pos;
    f_normal = ring_hosts[h].pole_max_r.xyz;
    f_host_idx = h;
    gl_Position = u_scene.view_proj * vec4(world_pos, 1.0);
    if (refraction_active()) gl_Position.z = gl_Position.w;
}
