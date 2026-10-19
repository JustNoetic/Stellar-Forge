#include "ring_shadow_filter.glsl"

#include "ring_hosts.glsl"
layout(set = 0, binding = 9) uniform sampler2DArray u_ring_shadow_lut;

float compute_surface_ring_shadow(vec3 frag_pos, vec3 L, float sun_ang_rad, uint candidates) {
    float ring_atten = 1.0;
    float focal_px = max(u_scene.screen_params.y * u_scene.screen_params.z, 1.0);
    float px_world_size = length(frag_pos) / focal_px;

    while (candidates != 0u) {
        uint r = uint(findLSB(candidates));
        candidates &= candidates - 1u;
        vec3 r_center = ring_hosts[r].center_min_r.xyz;
        vec3 r_pole = ring_hosts[r].pole_max_r.xyz;
        vec2 r_bounds = vec2(ring_hosts[r].center_min_r.w, ring_hosts[r].pole_max_r.w);
        int lut_layer = int(ring_hosts[r].counts_indices.y);

        float d = dot(L, r_pole);
        if (abs(d) > 1e-6) {
            float t = dot(r_center - frag_pos, r_pole) / d;
            if (t > 0.0) {
                vec3 hit_rel = (frag_pos + L * t) - r_center;
                float ext = ring_shadow_extinction(u_ring_shadow_lut, lut_layer, hit_rel, r_pole,
                                                   L, abs(d), t, r_bounds, sun_ang_rad, px_world_size, u_scene.ring_shadow_params.z != 0u);
                ring_atten *= (1.0 - ext);
                if (ring_atten <= 1e-4) return 0.0;
            }
        }
    }
    return ring_atten;
}
