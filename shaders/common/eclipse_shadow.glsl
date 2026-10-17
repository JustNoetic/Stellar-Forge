layout(std430, set = 0, binding = 2) readonly buffer EclipseCandidateBuffer {
    uint eclipse_words[];
};

float oblate_caster_radius(float r_eq, float r_polar, vec3 pole, vec3 L, vec3 perp_vec) {
    if (r_polar < 1e-9 || r_eq < 1e-9) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-9) return r_eq;
    vec3 p_proj = pole - L * dot(pole, L);
    float p_proj_len = length(p_proj);
    if (p_proj_len < 1e-6) return r_eq;
    vec3 p_dir = p_proj / p_proj_len;
    float y = dot(perp_vec, p_dir);
    float x = length(perp_vec - y * p_dir);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / r_polar) * (y / r_polar));
    return denom > 1e-9 ? (perp_len / denom) : r_eq;
}

float evaluate_eclipse_shadow(vec3 frag_pos, vec3 star_pos, float star_radius,
                              uint receiver_idx, uint star_idx, bool oblate_casters) {
    uint range = (receiver_idx * u_scene.num_stars + star_idx) * 2u;
    uint start = eclipse_words[range];
    uint count = eclipse_words[range + 1u];
    if (count == 0u) return 1.0;
    vec3 to_star = star_pos - frag_pos;
    float dist_star = length(to_star);
    if (dist_star <= 1e-9) return 1.0;
    vec3 L = to_star / dist_star;

    float vis = 1.0;
    for (uint candidate = 0u; candidate < count; ++candidate) {
        uint i = eclipse_words[start + candidate];
        vec4 caster = bodies[i].pos_radius;
        vec3 oc = caster.xyz - frag_pos;
        float proj = dot(oc, L);
        if (proj <= 0.0 || proj >= dist_star) continue;

        float r_eq = caster.w;
        vec3 perp_vec = oc - L * proj;
        float r_sun_proj = star_radius * (proj / dist_star);
        float max_penumbra = r_eq + r_sun_proj;
        if (dot(perp_vec, perp_vec) > max_penumbra * max_penumbra * 1.00002) continue;
        float d_perp = length(perp_vec);
        float r_eff = r_eq;
        if (oblate_casters) {
            float obl = clamp(bodies[i].pole_oblateness.w, 0.0, 0.95);
            if (obl > 1e-5)
                r_eff = oblate_caster_radius(r_eq, r_eq * (1.0 - obl), bodies[i].pole_oblateness.xyz, L, perp_vec);
        }

        float r_umbra = max(0.0, r_eff - r_sun_proj);
        float r_penumbra = r_eff + r_sun_proj;
        if (d_perp < r_penumbra) {
            float t = clamp((d_perp - r_umbra) / max(r_penumbra - r_umbra, 1e-9), 0.0, 1.0);
            vis *= pow(smoothstep(0.0, 1.0, t), 0.625);
            if (vis <= 1e-4) return 0.0;
        }
    }
    return vis;
}
