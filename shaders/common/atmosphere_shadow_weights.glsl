layout(set = 0, binding = 5) uniform sampler3D shadow_aerial_radiance;
layout(set = 0, binding = 9) uniform sampler2D shadow_aerial_range;

vec3 shadow_cached_weight(vec2 uv, float range, float path, float distance_weight) {
    if (path >= range) return vec3(0.0);
    float fraction = clamp(path/max(range, 1e-8), 0.0, 1.0);
    float coordinate = atm.camera.z < top_radius() ? sqrt(fraction) : fraction;
    float slice = min(floor(coordinate*float(AERIAL_SIZE.z-1)), float(AERIAL_SIZE.z-2));
    float t0 = aerial_distance(slice/float(AERIAL_SIZE.z-1), range);
    float t1 = aerial_distance((slice+1.0)/float(AERIAL_SIZE.z-1), range);
    vec3 near_light = textureLod(shadow_aerial_radiance, vec3(uv, (0.5+slice)/float(AERIAL_SIZE.z)), 0.0).rgb;
    vec3 far_light = textureLod(shadow_aerial_radiance, vec3(uv, (1.5+slice)/float(AERIAL_SIZE.z)), 0.0).rgb;
    return max(far_light-near_light, vec3(0.0))*(distance_weight/max(t1-t0, 1e-8));
}
