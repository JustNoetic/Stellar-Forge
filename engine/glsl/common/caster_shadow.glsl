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

// Conservative support of casterShadowTerm along a view segment, in km.
// Its optics are confined to the solid radius + stellar footprint; atmosphere
// thickness does not expand that support. Use endpoint distances to bound the
// changing footprint, and clip to the half-space behind the caster first.
bool caster_shadow_interval(vec3 D, vec3 V, vec3 L, float radius_km,
                            float star_ratio, inout float first, inout float last) {
    float axial = dot(D, L), slope = -dot(V, L);
    if (abs(slope) < 1e-8) {
        if (axial <= 0.0) return false;
    } else {
        float crossing = -axial / slope;
        if (slope > 0.0) first = max(first, crossing);
        else last = min(last, crossing);
    }
    if (first >= last) return false;
    float distance_max = max(length(D - first * V), length(D - last * V));
    float outer = radius_km + distance_max * star_ratio;
    vec3 A = D - axial * L;
    vec3 B = -V - slope * L;
    float qa = dot(B, B);
    if (qa < 1e-12) return dot(A, A) <= outer * outer;
    float closest = -dot(A, B) / qa;
    vec3 perpendicular = A + closest * B;
    float delta = outer * outer - dot(perpendicular, perpendicular);
    if (delta < 0.0) return false;
    float half_width = sqrt(delta / qa);
    first = max(first, closest - half_width);
    last = min(last, closest + half_width);
    return first < last;
}

// One conservative mask per view ray and star, reused by every march sample.
// The interval covers the solid radius and the full stellar footprint. Include
// the AU round-trip precision of compute_caster_shadow's world-space parcels,
// so rejecting a caster never clips a grazing penumbra or its refraction tail.
uint caster_shadow_ray_mask(vec3 origin_render, vec3 direction,
                           float first_km, float last_km, vec3 light,
                           float star_ratio) {
    uint mask = 0u;
    float ray_extent_au = max(abs(first_km), abs(last_km)) / u_au_to_km;
    for (int i = 0; i < min(u_num_active_casters, 8); ++i) {
        vec3 caster = u_active_casters[i].xyz;
        vec3 D = (caster - origin_render) * u_au_to_km;
        vec3 magnitude = max(abs(caster), abs(origin_render)) + vec3(ray_extent_au);
        float rounding_km = max(0.01, 4.0 * 1.192092896e-7 * u_au_to_km
            * max(magnitude.x, max(magnitude.y, magnitude.z)));
        float axial_first = dot(D - first_km * direction, light);
        float axial_last = dot(D - last_km * direction, light);
        if (max(axial_first, axial_last) < -rounding_km) continue;
        // A ray straddling the caster's plane is ambiguous at float precision;
        // retain it instead of relying on the interval's strict half-space clip.
        if (min(axial_first, axial_last) <= rounding_km) {
            mask |= 1u << i;
            continue;
        }
        float first = first_km, last = last_km;
        if (caster_shadow_interval(D, direction, light,
                u_active_casters[i].w * u_au_to_km + rounding_km,
                star_ratio, first, last)) {
            mask |= 1u << i;
        }
    }
    return mask;
}

// Product of per-caster eclipse attenuation for a parcel at eval_render_pos
// (render frame, AU) lit by a star along L_dir. Oblate casters and oblate
// stars are projected along the light direction. Ring-plane occlusion is
// intentionally NOT handled here (see header comment).
vec3 caster_shadow_at(int i, vec3 eval_render_pos, vec3 L_dir, float dist_to_star, float star_radius, float star_obl, vec3 star_pole) {
    vec3 caster_pos = u_active_casters[i].xyz;
    float caster_r = u_active_casters[i].w;

    vec3 s_to_c = caster_pos - eval_render_pos;
    float t_proj = dot(s_to_c, L_dir);
    if (t_proj < 0.0) return vec3(1.0);

    float dist_sq = dot(s_to_c, s_to_c);
    if (dist_sq < caster_r * caster_r * 1.0404) return vec3(1.0);

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

    float r_penumbra = caster_r + dist_to_caster * local_star_radius_over_dist;
    if (perp_sq > r_penumbra * r_penumbra) return vec3(1.0);

    vec4 grazing = vec4(u_active_caster_atmos[i].xyz,
        1.0 / max(u_active_casters[i].w * u_au_to_km, 100.0));
    vec3 caster_shadow = casterShadowTerm(local_star_radius_over_dist * dist_to_caster,
        caster_r, sqrt(perp_sq), u_active_max_bend[i], grazing,
        u_active_caster_ozone[i], u_active_scale_height[i], dist_to_caster, u_au_to_km);
    return caster_shadow;
}

vec3 compute_caster_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, float star_radius, float star_obl, vec3 star_pole, uint caster_mask) {
    uint remaining = caster_mask & ((1u << uint(min(u_num_active_casters, 8))) - 1u);
    if (remaining == 0u) return vec3(1.0);
    // The common one-caster case needs no bit scan or loop bookkeeping.
    if (remaining == 1u) return caster_shadow_at(0, eval_render_pos, L_dir, dist_to_star, star_radius, star_obl, star_pole);
    vec3 shadow = vec3(1.0);
    // Visit selected bits in ascending order, preserving multiplication order.
    while (remaining != 0u) {
        int i = findLSB(remaining);
        remaining &= remaining - 1u;
        shadow *= caster_shadow_at(i, eval_render_pos, L_dir, dist_to_star, star_radius, star_obl, star_pole);
    }
    return shadow;
}

vec3 compute_caster_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, float star_radius, float star_obl, vec3 star_pole) {
    return compute_caster_shadow(eval_render_pos, L_dir, dist_to_star, star_radius, star_obl, star_pole, 0xFFFFFFFFu);
}

