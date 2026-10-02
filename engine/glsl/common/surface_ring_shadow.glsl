// Surface/cloud ring shadows. Requires the ring uniforms declared by the caller.
// The radial footprint retains the small-disc, locally straight-edge model.
#include "ring_shadow_filter.glsl"

float surface_ring_shadow(vec3 body_center, vec3 surface_rel, vec3 L,
                          float dist_to_star, float star_ang_radius, uint ring_mask) {
    float transmission = 1.0;
    uint processed_mask = 0u;
    for (int k = 0; k < u_num_ring_planes; k++) {
        if ((ring_mask & (1u << k)) == 0u || (processed_mask & (1u << k)) != 0u) continue;
        vec3 plane_normal = u_ring_normal[k];
        uint coplanar_mask = u_ring_coplanar_mask[k];
        processed_mask |= coplanar_mask;
        float denom = dot(L, plane_normal);
        if (abs(denom) < 1e-8) continue;
        vec3 center_rel = u_ring_center[k] - body_center;
        float t = dot(center_rel - surface_rel, plane_normal) / denom;
        if (t <= 0.0 || t >= dist_to_star) continue;

        vec3 vec_radial = surface_rel + t * L - center_rel;
        float d = length(vec_radial);
        float r_star_proj = star_ang_radius * t;
        vec3 L_plane = L - denom * plane_normal;
        float L_plane_len = length(L_plane);
        float R_eff = r_star_proj;
        if (L_plane_len > 1e-5 && d > 1e-5) {
            vec3 dir_radial = vec_radial / d;
            float cos_theta = dot(dir_radial, L_plane / L_plane_len);
            float sin_theta = dot(dir_radial, normalize(cross(plane_normal, L)));
            R_eff *= sqrt(pow(cos_theta / max(1e-6, abs(denom)), 2.0) + sin_theta * sin_theta);
        }
        R_eff = max(R_eff, fwidth(d) * 0.75);
        float plane_occlusion = 0.0;
        for (int j = k; j < u_num_ring_planes; j++) {
            if ((coplanar_mask & (1u << j)) == 0u) continue;
            float inner_r = u_ring_params[j].x;
            float outer_r = u_ring_params[j].y;
            float opacity = u_ring_params[j].z;
            if (outer_r <= inner_r || opacity <= 0.0) continue;
            float inv_mu = 1.0 / max(1e-4, abs(dot(normalize(u_ring_normal[j]), L)));
            plane_occlusion += surface_ring_profile_occlusion(j, d, inner_r, outer_r,
                opacity, inv_mu, R_eff);
        }
        transmission *= 1.0 - clamp(plane_occlusion, 0.0, 1.0);
    }
    return transmission;
}
