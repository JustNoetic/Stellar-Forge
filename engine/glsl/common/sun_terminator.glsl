// Shared planetary terminator model (stellar-disc rise/set at an atmosphere parcel).
//
// Determines how much of the stellar disc clears the planet's local horizon for an
// atmosphere parcel, and which cosine to use for the sun-transmittance lookup. The
// stellar disc has effective angular half-width eff_star_rad = sin_star + max_bend:
// the star's true angular radius widened by the maximum atmospheric refraction bend
// (precomputed on the CPU, app.py -> u_star_pos_local[s].w), exactly like the caster
// penumbra in casterShadowTerm: refraction lets grazing sunlight wrap past the
// geometric horizon, extending the penumbra into the night side.
//
// Inputs:
//   light_cos_theta  - cos(sun zenith angle) at the parcel (dot(P, L_sun) / |P|)
//   sin/cos_planet   - sin/cos of the horizon angle at the parcel (R_planet / |P|)
//   eff_star_rad     - effective stellar angular radius (sin_star + max_bend)
//   cos_sun_eff      - cos(eff_star_rad), precomputed (u_star_dir_sph_eff[s].w)
// Returns vec2(vis_fraction, effective_cos):
//   vis_fraction  - fraction of the stellar disc visible above the local horizon
//   effective_cos - cosine for the sun-transmittance lookup: the mean of the visible
//                   disc segment, clamped to the tangent direction on the night side
//
// Consumed by the Mode 1/2 raymarchers (atmo.frag main loop), the Mode 3 ring-shadow
// slicing cells (accumulate_shadow_cell), and the Mode 3 Sky-View LUT bake
// (sky_view_lut.frag), so every quality mode shares one terminator: penumbra width
// from the star's angular size, extended by atmospheric refraction.
vec2 sun_terminator(float light_cos_theta, float sin_planet, float cos_planet,
                    float eff_star_rad, float cos_sun_eff) {
    // Horizon-grazing annulus in -cos(sun zenith): the disc is fully clear below
    // cos_outer, fully set above cos_inner, partially visible in between.
    float cos_outer = cos_planet * cos_sun_eff - sin_planet * eff_star_rad;
    float cos_inner = cos_planet * cos_sun_eff + sin_planet * eff_star_rad;

    float neg_light_cos = -light_cos_theta;

    if (neg_light_cos <= cos_outer) {
        // Daylight: Sun disc is 100% visible above the local horizon
        return vec2(1.0, light_cos_theta);
    } else if (neg_light_cos >= cos_inner) {
        // Night side: Planet fully blocks the direct solar disc (parcels high above
        // the surface may still see the un-occulted rim of a large stellar disc).
        float max_occ = min(1.0, (sin_planet * sin_planet) / max(1e-9, eff_star_rad * eff_star_rad));
        return vec2(1.0 - max_occ, max(-cos_planet + 1e-5, light_cos_theta));
    } else {
        // Twilight / penumbral transition zone: Sun disc partially sets below horizon
        float max_occ = min(1.0, (sin_planet * sin_planet) / max(1e-9, eff_star_rad * eff_star_rad));
        float min_vis = 1.0 - max_occ;
        float x_vis = (cos_planet * cos_sun_eff - neg_light_cos) / max(1e-7, sin_planet * eff_star_rad);
        float vis_fraction = mix(min_vis, 1.0, smoothstep(-1.0, 1.0, x_vis));

        float sin_Z = sqrt(max(0.0, 1.0 - light_cos_theta * light_cos_theta));
        float exact_disc_top_cos = light_cos_theta * cos_sun_eff + sin_Z * eff_star_rad;
        float exact_disc_bot_cos = light_cos_theta * cos_sun_eff - sin_Z * eff_star_rad;
        float disc_top_cos = (light_cos_theta > cos_sun_eff) ? 1.0 : exact_disc_top_cos;
        float disc_bot_cos = max(exact_disc_bot_cos, -cos_planet + 1e-5);
        float effective_cos = max(-cos_planet + 1e-5, (disc_top_cos + disc_bot_cos) * 0.5);
        return vec2(vis_fraction, effective_cos);
    }
}
