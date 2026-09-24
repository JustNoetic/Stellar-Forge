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
//   vec4  u_active_caster_atmos[8];    (xyz = grazing tau, w = atmo thickness AU)
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
vec3 casterShadowTerm(float alpha, float beta, float gamma,
                      float penumbra_outer, float penumbra_inner,
                      float max_bend, vec4 atmo_param, vec4 ozone_param,
                      float scale_height_km, float dist_to_caster) {
    float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
    float occ = clamp(max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma), 0.0, 1.0);
    // Correct geometric shadow curve: (1 - occ)^GAMMA (Space Engine gamma correction: 1 / 1.6)
    float geom_sh = pow(clamp(1.0 - occ, 0.0, 1.0), 1.0 / 1.6);
    vec3 sh = vec3(geom_sh);

    if (max_bend > 1e-6 && gamma < penumbra_outer) {
        float H_scale = max(scale_height_km, 0.1);
        float sigma_z = max(0.707 * H_scale, 0.1);
        float z_peak = ozone_param.w;
        float dist_km = max(dist_to_caster * u_au_to_km, 1e-6);

        float caster_r_km = max(beta * dist_km, 100.0);

        // For an extended light source (like the Sun), the transmitted light is dominated
        // by the rays passing through the highest possible altitude (least required bend).
        // We approximate the integral over the Sun's disk by evaluating the ray from the
        // upper limb (offset by ~80% of the Sun's radius).
        float req_bend = beta - gamma - alpha * 0.8;

        // Rays requiring bend > max_bend hit the solid planet body (100% blocked)
        if (req_bend <= max_bend) {
            // Normalized penetration depth in atmosphere (0 = top of atmosphere, 1 = surface level)
            float atmo_depth = clamp(max(0.0, req_bend) / max_bend, 0.0, 1.0);

            // Grazing altitude z corresponding to this refraction bend angle:
            // theta(z) = max_bend * exp(-z / H) -> atmo_depth = exp(-z / H)
            float z_km = -H_scale * log(max(atmo_depth, 1e-5));

            // 1. Rayleigh grazing optical depth at altitude z
            vec3 tau_R = atmo_param.xyz * atmo_depth;

            // 2. Stratospheric ozone layer absorption along grazing ray (Chappuis band)
            float z_diff = (z_km - z_peak) / sigma_z;
            vec3 tau_O3 = ozone_param.xyz * exp(-0.5 * z_diff * z_diff);

            // Total optical depth and physical spectral transmittance
            vec3 tau_total = tau_R + tau_O3;
            vec3 atmo_transmittance = exp(-tau_total);

            // Physical Atmospheric Ring Geometric Dilution with Radial Astigmatic Defocusing:
            // Tangential divergence scales as 1/D around the ring. Radial divergence across the
            // exponential atmosphere density gradient (d_theta/dz = -theta/H) dilutes flux by
            // 1 / (1 + D * theta / H), giving true 3D 1/D^2 energy conservation at large distances.
            float radial_defocus = 1.0 / (1.0 + (dist_km * max(req_bend, 1e-6)) / H_scale);

            // Annular lens focusing requires rays from opposite limbs to reach the focal line (D >= f_focal = R / max_bend).
            // In the near field (D < f_focal, e.g. a planet's own rings), rays have not converged.
            float f_focal_km = caster_r_km / max(max_bend, 1e-6);
            float focal_ratio = clamp(dist_km / f_focal_km, 0.0, 1.0);

            // Full annular Einstein ring is only visible on-axis (gamma <= alpha). Off-axis observers
            // (gamma > alpha, such as planetary rings) only see a local limb arc subtending fraction alpha / gamma.
            float off_axis_factor = clamp(alpha / max(1e-6, gamma), 0.0, 1.0);

            float annular_factor = (2.0 * H_scale) / max(alpha * dist_km, 1e-9);
            float ring_intensity = clamp(annular_factor * focal_ratio * off_axis_factor, 0.0, 1.0) * radial_defocus;

            // Smooth surface grazing fade to zero at solid body boundary (h = 0, atmo_depth = 1)
            float body_surface_fade = smoothstep(1.0, 0.75, atmo_depth);

            float refraction_intensity = ring_intensity * (1.0 - atmo_depth * 0.7) * body_surface_fade;

            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            sh += atmo_transmittance * refraction_intensity * atmo_blend;
        }
    }
    return clamp(sh, vec3(0.0), vec3(1.0));
}

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

        float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
        float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
        if (perp_sq > r_penumbra * r_penumbra) continue;

        float inv_dist = 1.0 / dist_to_caster;
        float alpha = local_star_radius_over_dist;
        float beta = caster_r * inv_dist;
        float gamma = sqrt(perp_sq) * inv_dist;

        float penumbra_outer = alpha + beta;
        float penumbra_inner = abs(beta - alpha);

        float max_bend = u_active_max_bend[i];
        vec3 caster_shadow = casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, max_bend, u_active_caster_atmos[i], u_active_caster_ozone[i], u_active_caster_atmos[i].w, dist_to_caster);
        shadow *= caster_shadow;
    }

    return shadow;
}

vec3 compute_caster_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, float star_radius, float star_obl, vec3 star_pole) {
    return compute_caster_shadow(eval_render_pos, L_dir, dist_to_star, star_radius, star_obl, star_pole, 0xFFFFFFFFu);
}

