#include "ring_shadow_filter.glsl"

struct RingSegmentGPU {
    vec4 bounds_opacity_scatter;
    vec4 phase_row;
    vec4 color_flags;
};

struct RingHostGPU {
    vec4 center_min_r;
    vec4 pole_max_r;
    vec4 tangent_radius;
    vec4 bitangent_oblateness;
    uvec4 counts_indices;
    RingSegmentGPU segments[8];
};

layout(std430, set = 0, binding = 7) readonly buffer RingHostBuffer {
    RingHostGPU ring_hosts[];
};

layout(set = 0, binding = 9) uniform sampler2DArray u_ring_shadow_lut;

float compute_surface_ring_shadow(vec3 frag_pos, vec3 L, float sun_ang_rad) {
    float ring_atten = 1.0;
    float focal_px = max(u_scene.screen_params.y * u_scene.screen_params.z, 1.0);
    float px_world_size = length(frag_pos) / focal_px;

    for (uint r = 0u; r < u_scene.num_ring_hosts; ++r) {
        vec3 r_center = ring_hosts[r].center_min_r.xyz;
        vec3 r_pole = ring_hosts[r].pole_max_r.xyz;
        vec2 r_bounds = vec2(ring_hosts[r].center_min_r.w, ring_hosts[r].pole_max_r.w);
        int lut_layer = int(ring_hosts[r].counts_indices.y);

        float d = dot(L, r_pole);
        if (abs(d) > 1e-6) {
            float t = dot(r_center - frag_pos, r_pole) / d;
            if (t > 0.0) {
                vec3 hit_rel = (frag_pos + L * t) - r_center;
                float dist = length(hit_rel);
                float pad = sun_ang_rad * t * 3.0 + px_world_size * 2.0;
                if (dist >= r_bounds.x - pad && dist <= r_bounds.y + pad) {
                    float ext = ring_shadow_extinction(u_ring_shadow_lut, lut_layer, hit_rel, r_pole,
                                                       L, abs(d), t, r_bounds, sun_ang_rad, px_world_size);
                    ring_atten *= (1.0 - ext);
                    if (ring_atten <= 1e-4) return 0.0;
                }
            }
        }
    }
    return ring_atten;
}
