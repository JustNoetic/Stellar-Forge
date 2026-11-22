#include "terrain_horizon_math.glsl"

layout(set = 0, binding = 47) uniform sampler2DArray u_horizon_tiles;
layout(std430, set = 0, binding = 48) readonly buffer HorizonBounds {
    uint horizon_slot_max[];
};

vec4 sample_horizon_slot_bins(uint slot, vec2 uv, uint km1) {
    uint base_layer = (slot - 1u) * 4u;
    uint rem = km1 & 3u;
    uint la = km1 >> 2u;
    uint lb = (la + 1u) & 3u;
    vec4 va = textureLod(u_horizon_tiles, vec3(uv, float(base_layer + la)), 0.0);
    vec4 vb = (rem == 0u) ? va : textureLod(u_horizon_tiles, vec3(uv, float(base_layer + lb)), 0.0);
    vec4 u_bins = (rem == 0u) ? va
                : (rem == 1u) ? vec4(va.yzw, vb.x)
                : (rem == 2u) ? vec4(va.zw, vb.xy)
                              : vec4(va.w, vb.xyz);
    return decode_horizon_sin4(u_bins);
}

vec2 terrain_self_shadow(SurfaceNormalLookup lookup, vec3 local_dir, float polar_ratio,
                          vec3 N_local, vec3 L_local, float star_ang_rad, bool need_ocean) {
    uint coarse_slot = normal_lookup_coarse_slot(lookup);
    if ((normal_lookup_flags(lookup) & 2u) == 0u || coarse_slot == 0u) return vec2(1.0);
    vec3 U_local = normalize(vec3(local_dir.x, local_dir.y / max(polar_ratio, 0.05), local_dir.z));
    float sin_alpha = clamp(dot(L_local, U_local), -1.0, 1.0);
    uint fine_slot = normal_lookup_fine_slot(lookup);
    uint max_u_byte = horizon_slot_max[coarse_slot - 1u];
    if (fine_slot != 0u && fine_slot != coarse_slot) {
        max_u_byte = max(max_u_byte, horizon_slot_max[fine_slot - 1u]);
    }
    float max_sin_h = decode_horizon_sin(float(max_u_byte) * (1.0 / 255.0));
    float r_ang = max(star_ang_rad, 1e-5);
    float max_theta_eff = asin(clamp(max_sin_h, -1.0, 1.0)) - HORIZON_SLOPE_BIAS + 1e-4;
    if (max_theta_eff <= 0.0 && sin_alpha >= r_ang) return vec2(1.0);
    float alpha = asin(sin_alpha);
    if (max_theta_eff <= alpha - r_ang) return vec2(1.0);
    vec3 L_h = L_local - U_local * sin_alpha;
    float len_h = length(L_h);
    if (len_h < 1e-5) return vec2(1.0);
    vec3 dir_h = L_h / len_h;

    vec3 T, B;
    horizon_tangent_basis(normal_lookup_face(lookup), U_local, T, B);
    float phi = atan(dot(dir_h, B), dot(dir_h, T));
    if (phi < 0.0) phi += HORIZON_TWO_PI;
    float fbin = phi * (16.0 / HORIZON_TWO_PI);
    uint k0 = uint(floor(fbin)) & 15u;
    float frac = fract(fbin);
    uint km1 = (k0 + 15u) & 15u;

    vec4 hc = sample_horizon_slot_bins(coarse_slot, lookup.coarse_uv, km1);
    vec4 h = (fine_slot != 0u && fine_slot != coarse_slot)
        ? mix(hc, sample_horizon_slot_bins(fine_slot, lookup.fine_uv, km1), lookup.blend)
        : hc;
    float sin_h = horizon_catmull_rom(h.x, h.y, h.z, h.w, frac);
    float theta_horizon = asin(clamp(sin_h, -1.0, 1.0));
    if (theta_horizon - HORIZON_SLOPE_BIAS <= alpha - r_ang) return vec2(1.0);

    float nu = dot(N_local, U_local);
    float theta_tangent = (nu > 1e-4) ? atan(-dot(N_local, dir_h) / nu) : 1.5707963;
    float slope_bias = HORIZON_SLOPE_BIAS + 0.006 * sqrt(max(sin(max(theta_tangent, 0.0)), 0.0));
    float eff_horizon = theta_horizon - slope_bias;
    float terrain_vis = 1.0;
    if (eff_horizon > theta_tangent && eff_horizon > alpha - r_ang) {
        float f_base = horizon_disk_above(alpha, theta_tangent, star_ang_rad);
        float f_eff = horizon_disk_above(alpha, eff_horizon, star_ang_rad);
        terrain_vis = (f_base > 1e-5) ? clamp(f_eff / f_base, 0.0, 1.0) : 0.0;
    }

    float ocean_vis = 1.0;
    float ocean_eff = theta_horizon - HORIZON_SLOPE_BIAS;
    if (need_ocean && ocean_eff > 0.0 && ocean_eff > alpha - r_ang) {
        float f_ocean_base = horizon_disk_above(alpha, 0.0, star_ang_rad);
        float f_ocean_eff = horizon_disk_above(alpha, ocean_eff, star_ang_rad);
        ocean_vis = (f_ocean_base > 1e-5) ? clamp(f_ocean_eff / f_ocean_base, 0.0, 1.0) : 0.0;
    }
    return vec2(terrain_vis, ocean_vis);
}

vec2 terrain_self_shadow(SurfaceNormalLookup lookup, vec3 local_dir, float polar_ratio,
                          vec3 N_local, vec3 L_local, float star_ang_rad) {
    return terrain_self_shadow(lookup, local_dir, polar_ratio, N_local, L_local, star_ang_rad, true);
}
