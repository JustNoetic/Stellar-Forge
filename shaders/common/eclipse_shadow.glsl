#include "eclipse_optics.glsl"

layout(std430, set = 0, binding = 2) readonly buffer EclipseCandidateBuffer {
    uint eclipse_words[];
};

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

vec3 evaluate_eclipse_shadow(vec3 frag_pos, vec3 star_pos, float star_radius,
                              uint receiver_idx, uint star_idx) {
    uint range = (receiver_idx * u_scene.num_stars + star_idx) * 2u;
    uint start = eclipse_words[range];
    uint count = eclipse_words[range + 1u];
    if (count == 0u) return vec3(1.0);
    vec3 to_star = star_pos - frag_pos;
    float dist_star = length(to_star);
    if (dist_star <= 1e-9) return vec3(1.0);
    vec3 L = to_star / dist_star;

    vec3 vis = vec3(1.0);
    for (uint candidate = 0u; candidate < count; ++candidate) {
        uint i = eclipse_words[start + candidate];
        vis *= eclipse_shadow_caster(i, star_idx, frag_pos, L, dist_star, star_radius);
        if (max(vis.r, max(vis.g, vis.b)) <= (u_scene.eclipse_params.y == 0u ? 1e-4 : 0.0)) return vec3(0.0);
    }
    return vis;
}
