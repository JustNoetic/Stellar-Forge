// Shared analytical eclipse shadowing for atmosphere shaders.
//
// Contents:
//   get_oblate_radius      - projected radius of an oblate spheroid along a
//                            light direction, used for oblate casters/stars.
//   casterShadowTerm       - per-caster penumbra/umbra cone attenuation with
//                            physical atmospheric lens optics (refraction ring
//                            brightening + Danjon spectral tinting) for casters
//                            that own an atmosphere.
//   compute_caster_shadow  - product of casterShadowTerm over all active
//                            eclipse casters (the spherical-body half of
//                            atmo.frag's compute_shadow; ring-plane occlusion
//                            stays in atmo.frag because it needs the ring
//                            uniforms, and Mode 3 handles own-ring shadows via
//                            the per-pixel station slicing instead).
//
// Required globals (declared by the including shader BEFORE this include):
//   float u_au_to_km;                  (AtmoData SSBO, binding = 8)
//   int   u_num_active_casters;        (AtmoData SSBO)
//   vec4  u_active_casters[8];         (xyz = render-frame pos AU, w = eq. radius AU)
//   vec4  u_active_caster_poles_obl[8];
//   float u_active_caster_R_minor[8];
//   vec4  u_active_caster_atmos[8];    (xyz = grazing tau, w = atmo thickness km)
//   float u_active_scale_height[8];  (km; independent of thickness)
//   float u_active_max_bend[8];
//   vec4  u_active_caster_ozone[8];
//
// Consumed by atmo.frag (Modes 1/2 raymarchers, via compute_shadow) and
// sky_view_lut.frag (Mode 3 Sky-View LUT bake), so every quality mode shares
// byte-identical eclipse math.

float get_oblate_radius(float r_eq, float r_minor, vec3 pole, vec3 L, vec3 perp_vec) {
    if (r_minor < 1e-6 || r_eq < 1e-6) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / r_minor) * (y / r_minor));
    if (denom < 1e-6) return r_eq;
    return perp_len / denom;
}

// Shared analytical eclipse penumbra + physical atmospheric lens optics & Danjon refraction tinting.
// Inputs are per-caster quantities already resolved by the caller.
#include "eclipse_shadow.glsl"

// Product of per-caster eclipse attenuation for a parcel at eval_render_pos
// (render frame, AU) lit by a star along L_dir. Oblate casters and oblate
// stars are projected along the light direction. Ring-plane occlusion is
// intentionally NOT handled here (see header comment).
vec3 compute_caster_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, float star_radius, float star_obl, vec3 star_pole, uint caster_mask) {
    if (caster_mask == 0u) return vec3(1.0);
    vec3 shadow = vec3(1.0);

    for (int i = 0; i < u_num_active_casters; i++) {
        if ((caster_mask & (1u << i)) == 0u) continue;
        vec3 caster_pos = u_active_casters[i].xyz;
        float caster_r = u_active_casters[i].w;
        float atmo_h = u_active_caster_atmos[i].w;

        vec3 s_to_c = caster_pos - eval_render_pos;
        float t_proj = dot(s_to_c, L_dir);
        if (t_proj < 0.0) continue;

        float dist_sq = dot(s_to_c, s_to_c);
        if (dist_sq < caster_r * caster_r * 1.0404) continue;

        float dist_to_caster = sqrt(dist_sq);
        vec3 cross_vec = cross(s_to_c, L_dir);
        float perp_sq = dot(cross_vec, cross_vec);

        vec3 perp_vec = s_to_c - t_proj * L_dir;
        float caster_r_minor = u_active_caster_R_minor[i];
        if (caster_r_minor < caster_r - 1e-5) {
            vec3 pole = u_active_caster_poles_obl[i].xyz;
            caster_r = get_oblate_radius(caster_r, caster_r_minor, pole, L_dir, perp_vec);
        }

        float directional_star_r = star_radius;
        if (star_obl < star_radius - 1e-5) {
            directional_star_r = get_oblate_radius(star_radius, star_obl, star_pole, L_dir, perp_vec);
        }
        float local_star_radius_over_dist = directional_star_r / max(dist_to_star, 1e-6);

        float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * (4.0 / u_au_to_km) : 0.0);
        float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
        if (perp_sq > r_penumbra * r_penumbra) continue;

        vec4 grazing = vec4(u_active_caster_atmos[i].xyz,
            1.0 / max(u_active_casters[i].w * u_au_to_km, 100.0));
        vec3 caster_shadow = casterShadowTerm(local_star_radius_over_dist * dist_to_caster,
            caster_r, sqrt(perp_sq), u_active_max_bend[i], grazing,
            u_active_caster_ozone[i], u_active_scale_height[i], dist_to_caster, u_au_to_km);
        shadow *= caster_shadow;
    }

    return shadow;
}

vec3 compute_caster_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, float star_radius, float star_obl, vec3 star_pole) {
    return compute_caster_shadow(eval_render_pos, L_dir, dist_to_star, star_radius, star_obl, star_pole, 0xFFFFFFFFu);
}

