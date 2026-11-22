#include "eclipse_optics.glsl"

#ifndef ECLIPSE_CANDIDATE_BUFFER_DEFINED
#define ECLIPSE_CANDIDATE_BUFFER_DEFINED
layout(std430, set = 0, binding = 2) readonly buffer EclipseCandidateBuffer {
    uint eclipse_words[];
};
#endif

#include "eclipse_lookup.glsl"

vec3 eclipse_shadow_caster(uint i, uint star_index, vec3 frag_pos, vec3 L, float dist_star, float star_radius) {
    BodyInstanceGPU body = bodies[i];
    vec4 caster = body.pos_radius;
    vec3 oc = caster.xyz-frag_pos;
    float proj = dot(oc, L);
    if (proj <= 0.0 || proj >= dist_star) return vec3(1.0);
    vec3 perp = oc-L*proj;
    float source = star_radius*(proj/dist_star);
    vec4 record = eclipse_lookup_record(i, star_index);
    float bound = (caster.w+source)*max(record.z, 1.0);
    if (dot(perp, perp) > bound*bound*1.00002) return vec3(1.0);
    float obl = clamp(body.pole_oblateness.w, 0.0, .95);
    vec3 pole = body_pole(body);
    vec4 optical = vec4(0.0);
    if (record.x >= .5) {
        float directional = oblate_caster_radius(caster.w, caster.w*(1.0-obl), pole, L, perp);
        optical = eclipse_lookup_visibility(record, directional, source, length(perp));
        if (optical.a >= 1.0) return optical.rgb;
    }
    float geometric = eclipse_caster_visibility(caster.w, caster.w*(1.0-obl), pole, L, perp, source, u_scene.eclipse_params.x != 0u);
    return mix(vec3(geometric), optical.rgb, optical.a);
}

vec3 evaluate_eclipse_shadow_ray(vec3 frag_pos, vec3 L, float dist_star, float star_radius,
                                   uint receiver_idx, uint star_idx) {
    uint range = (receiver_idx * u_scene.num_stars + star_idx) * 2u;
    uint start = eclipse_words[range];
    uint count = eclipse_words[range + 1u];
    if (count == 0u) return vec3(1.0);

    vec3 vis = vec3(1.0);
    float vis_cutoff = (u_scene.eclipse_params.y == 0u ? 1e-4 : 0.0);
    for (uint candidate = 0u; candidate < count; ++candidate) {
        uint i = eclipse_words[start + candidate];
        vis *= eclipse_shadow_caster(i, star_idx, frag_pos, L, dist_star, star_radius);
        if (max(vis.r, max(vis.g, vis.b)) <= vis_cutoff) return vec3(0.0);
    }
    return vis;
}

vec3 evaluate_ring_eclipse_shadow(vec3 frag_pos, vec3 hit_local, vec3 ring_u, vec3 ring_v, float outer_r,
                                  vec3 L, float dist_star, float star_radius,
                                  uint receiver_idx, uint star_idx) {
    uint range = (receiver_idx * u_scene.num_stars + star_idx) * 2u;
    uint start = eclipse_words[range];
    uint count = eclipse_words[range + 1u];
    if (count == 0u) return vec3(1.0);

    float grid_r = max(outer_r * 1.04, 1e-12);
    vec2 norm_uv = vec2(dot(hit_local, ring_u), dot(hit_local, ring_v)) / grid_r * 0.5 + 0.5;
    ivec2 cell = clamp(ivec2(floor(norm_uv * 32.0)), ivec2(0), ivec2(31));
    uint mask = eclipse_words[start + count + uint(cell.y * 32 + cell.x)];
    if (mask == 0u) return vec3(1.0);

    vec3 vis = vec3(1.0);
    float vis_cutoff = (u_scene.eclipse_params.y == 0u ? 1e-4 : 0.0);
    uint active_bits = mask & 0x7FFFFFFFu;
    while (active_bits != 0u) {
        uint slot = uint(findLSB(active_bits));
        active_bits &= active_bits - 1u;
        uint i = eclipse_words[start + slot];
        vis *= eclipse_shadow_caster(i, star_idx, frag_pos, L, dist_star, star_radius);
        if (max(vis.r, max(vis.g, vis.b)) <= vis_cutoff) return vec3(0.0);
    }
    if ((mask & 0x80000000u) != 0u) {
        for (uint slot = 31u; slot < count; ++slot) {
            uint i = eclipse_words[start + slot];
            vis *= eclipse_shadow_caster(i, star_idx, frag_pos, L, dist_star, star_radius);
            if (max(vis.r, max(vis.g, vis.b)) <= vis_cutoff) return vec3(0.0);
        }
    }
    return vis;
}

vec3 evaluate_eclipse_shadow(vec3 frag_pos, vec3 star_pos, float star_radius,
                              uint receiver_idx, uint star_idx) {
    uint range = (receiver_idx * u_scene.num_stars + star_idx) * 2u;
    uint count = eclipse_words[range + 1u];
    if (count == 0u) return vec3(1.0);
    vec3 to_star = star_pos - frag_pos;
    float dist_star = length(to_star);
    if (dist_star <= 1e-9) return vec3(1.0);
    return evaluate_eclipse_shadow_ray(frag_pos, to_star / dist_star, dist_star, star_radius, receiver_idx, star_idx);
}
