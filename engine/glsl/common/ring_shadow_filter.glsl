// Shared radial ring-shadow filter for surfaces, clouds and Mode 2 atmosphere.
// Radius, ring bounds and footprint must use the same units. No derivatives:
// callers supply the physical solar footprint and any pixel filtering they need.
uniform sampler2DArray u_ring_shadow_integral;

float surface_ring_box_occlusion(int row, float p, float half_width, float angle) {
    vec2 bounds = clamp(vec2(p - half_width, p + half_width), 0.0, 1.0);
    vec3 size = vec3(textureSize(u_ring_shadow_integral, 0));
    vec2 uv = (bounds * (size.x - 1.0) + 0.5) / size.x;
    float y = (angle * (size.y - 1.0) + 0.5) / size.y;
    float low = textureLod(u_ring_shadow_integral, vec3(uv.x, y, float(row)), 0.0).r;
    float high = textureLod(u_ring_shadow_integral, vec3(uv.y, y, float(row)), 0.0).r;
    // Divide by the FULL footprint: regions outside the ring transmit freely.
    return max(0.0, high - low) / (2.0 * half_width);
}

float surface_ring_filtered_occlusion(int row, float p, float half_width, float inv_mu) {
    float angle = clamp(log(inv_mu) / log(1e4), 0.0, 1.0);
    // Positive nested boxes approximate the uniform stellar-disc kernel.
    // Their fitted CDF differs by <= 0.0074 from the analytic disc CDF.
    // Six integral reads include ALL ringlets, independent of footprint size.
    return 0.51872425 * surface_ring_box_occlusion(row, p, half_width, angle)
         + 0.40202055 * surface_ring_box_occlusion(row, p, half_width * 0.75, angle)
         + 0.07925520 * surface_ring_box_occlusion(row, p, half_width * 0.40, angle);
}

float surface_ring_disc_cdf(float v) {
    v = clamp(v, -1.0, 1.0);
    return (v * sqrt(max(0.0, 1.0 - v * v)) + asin(v)) / PI + 0.5;
}

float surface_ring_occlusion(int row, float radius, float inner_r, float outer_r,
                             float opacity, float inv_light_mu) {
    float p = (radius - inner_r) / (outer_r - inner_r);
    float alpha = textureLod(u_ring_gradients, vec2(p, (float(row) + 0.5) / 16.0), 0.0).a;
    float normal_transmission = clamp(1.0 - opacity * alpha, 1e-6, 1.0);
    // Apply oblique extinction BEFORE filtering, so opaque bands and clear gaps
    // retain their separate transmissions within the stellar footprint.
    return 1.0 - pow(normal_transmission, inv_light_mu);
}

float surface_ring_profile_occlusion(int row, float d, float inner_r, float outer_r,
                                     float opacity, float inv_mu, float R_eff) {
    if (outer_r <= inner_r || opacity <= 0.0) return 0.0;
    float occlusion = 0.0;
    if (R_eff <= 1e-12) {
        if (d >= inner_r && d <= outer_r)
            occlusion += surface_ring_occlusion(row, d, inner_r, outer_r, opacity, inv_mu);
        return occlusion;
    }
    float overlap_min = max(inner_r, d - R_eff);
    float overlap_max = min(outer_r, d + R_eff);
    if (overlap_min >= overlap_max) return 0.0;
    float overlap_width = overlap_max - overlap_min;
    float radial_texel = (outer_r - inner_r) / float(textureSize(u_ring_gradients, 0).x);
    // The runtime unified atlas includes opacity (params.z is 0 or 1).
    // Retain direct filtering for tiny footprints and external callers
    // with a separate opacity multiplier or an unbaked atlas row.
    if (opacity == 1.0 && row < textureSize(u_ring_shadow_integral, 0).z
        && R_eff >= radial_texel) {
        float span = outer_r - inner_r;
        occlusion += surface_ring_filtered_occlusion(row,
            (d - inner_r) / span, R_eff / span, inv_mu);
        return occlusion;
    }
    float v_min = clamp((overlap_min - d) / R_eff, -1.0, 1.0);
    float v_max = clamp((overlap_max - d) / R_eff, -1.0, 1.0);
    float cdf_min = surface_ring_disc_cdf(v_min);
    float cdf_max = surface_ring_disc_cdf(v_max);
    float fraction = max(0.0, cdf_max - cdf_min);
    // A single tap is safe only for a sub-texel interval. A fixed
    // fraction of the entire ring span can otherwise straddle a gap.
    if (overlap_width <= radial_texel * 0.25) {
        occlusion += fraction * surface_ring_occlusion(row,
            0.5 * (overlap_min + overlap_max), inner_r, outer_r, opacity, inv_mu);
    } else {
        float previous_cdf = cdf_min;
        for (int sample_idx = 0; sample_idx < 8; sample_idx++) {
            float u = (float(sample_idx) + 0.5) / 8.0;
            float radius = mix(overlap_min, overlap_max, u);
            float edge = mix(v_min, v_max, float(sample_idx + 1) / 8.0);
            float cdf = surface_ring_disc_cdf(edge);
            float weight = max(0.0, cdf - previous_cdf);
            previous_cdf = cdf;
            occlusion += weight * surface_ring_occlusion(row, radius,
                inner_r, outer_r, opacity, inv_mu);
        }
    }
    return occlusion;
}
