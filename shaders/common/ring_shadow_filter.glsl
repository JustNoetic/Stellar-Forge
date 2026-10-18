float ring_integral_sample(sampler2DArray lut, int ring_idx, float u, float mu_coord) {
    float clamped = clamp(u, 0.0, 1.0);
    float texel = (clamped * 4096.0 + 0.5) / 4097.0;
    return textureLod(lut, vec3(texel, mu_coord, float(ring_idx)), 0.0).r;
}

float ring_box_extinction(sampler2DArray lut, int ring_idx, float u_center, float half_width, float mu_coord) {
    float a = clamp(u_center - half_width, 0.0, 1.0);
    float b = clamp(u_center + half_width, 0.0, 1.0);
    float lo = ring_integral_sample(lut, ring_idx, a, mu_coord);
    float hi = ring_integral_sample(lut, ring_idx, b, mu_coord);
    return max(0.0, hi - lo) / max(2.0 * half_width, 1e-6);
}

float ring_shadow_extinction(sampler2DArray lut, int ring_idx, vec3 hit_rel, vec3 pole, vec3 L,
                             float mu_s, float t_hit, vec2 bounds, float sun_ang_rad, float px_world_size) {
    float span = max(bounds.y - bounds.x, 1e-9);
    float r_hit = length(hit_rel);
    float u0 = (r_hit - bounds.x) / span;
    float mu = clamp(mu_s, 1.0 / 128.0, 1.0);
    float mu_coord = clamp(mu, 0.5 / 64.0, 63.5 / 64.0);

    vec3 r_hat = r_hit > 1e-9 ? hit_rel / r_hit : vec3(1.0, 0.0, 0.0);
    vec3 l_plane = L - pole * dot(L, pole);
    float l_plane_len = length(l_plane);
    vec3 l_hat = l_plane_len > 1e-6 ? l_plane / l_plane_len : r_hat;
    float cos_phi = dot(r_hat, l_hat);
    float sin_phi = sqrt(max(0.0, 1.0 - cos_phi * cos_phi));
    float stretch = sqrt((cos_phi * cos_phi) / (mu * mu) + sin_phi * sin_phi);
    float footprint_u = sun_ang_rad * t_hit * stretch / span;
    float px_u = px_world_size * stretch / span;
    float min_u = max(0.75 * max(px_u, 1.0 / 4096.0), 0.75 / 4096.0);
    float h = max(footprint_u, min_u);

    if (u0 + h <= 0.0 || u0 - h >= 1.0) {
        return 0.0;
    }

    if (u_scene.ring_shadow_params.z != 0u)
        return clamp(ring_box_extinction(lut, ring_idx, u0, h, mu_coord), 0.0, 1.0);
    float b1 = ring_box_extinction(lut, ring_idx, u0, 0.36 * h, mu_coord);
    float b2 = ring_box_extinction(lut, ring_idx, u0, 0.72 * h, mu_coord);
    float b3 = ring_box_extinction(lut, ring_idx, u0, 1.00 * h, mu_coord);
    return clamp(0.24 * b1 + 0.46 * b2 + 0.30 * b3, 0.0, 1.0);
}
