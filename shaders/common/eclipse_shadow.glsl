#include "eclipse_optics.glsl"

layout(std430, set = 0, binding = 2) readonly buffer EclipseCandidateBuffer {
    uint eclipse_words[];
};

float evaluate_eclipse_shadow(vec3 frag_pos, vec3 star_pos, float star_radius,
                              uint receiver_idx, uint star_idx) {
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
        vec4 pole_oblateness = bodies[i].pole_oblateness;
        float obl = clamp(pole_oblateness.w, 0.0, 0.95);
        vec3 pole = length(pole_oblateness.xyz) > 1e-4 ? normalize(pole_oblateness.xyz) : vec3(0, 1, 0);
        vis *= eclipse_caster_visibility(r_eq, r_eq * (1.0 - obl), pole, L, perp_vec, r_sun_proj, u_scene.eclipse_params.x != 0u);
        if (vis <= 1e-4) return 0.0;
    }
    return vis;
}
