const float POINT_PI = 3.141592653589793;
const vec2 POINT_QUAD_VERTS[6] = vec2[](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

float mesh_weight(float radius_px) {
    return smoothstep(1.5, 2.5, radius_px);
}

float point_profile(vec2 pixel_offset, float radius_px) {
    float r2 = dot(pixel_offset, pixel_offset) / (radius_px * radius_px);
    if (r2 >= 1.0) return 0.0;
    return exp(-3.5 * r2) * (1.0 - smoothstep(0.6, 1.0, sqrt(r2)));
}

float point_profile_sum(vec2 center_px, float radius_px) {
    vec2 first = ceil(center_px - radius_px - 0.5) + 0.5;
    vec2 end = center_px + radius_px;
    float sum = 0.0;
    for (float y = first.y; y < end.y; y += 1.0) {
        for (float x = first.x; x < end.x; x += 1.0) {
            sum += point_profile(vec2(x, y) - center_px, radius_px);
        }
    }
    return max(sum, 1e-12);
}

float point_signal_fade(vec3 flux, float radius_px) {
    float peak = max(flux.r, max(flux.g, flux.b)) / max(0.7928351 * radius_px * radius_px, 1.0);
    return smoothstep(1.0e-6, 2.5e-6, peak);
}
