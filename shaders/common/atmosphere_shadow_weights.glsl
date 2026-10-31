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

vec3 shadow_cached_light(vec2 uv, float range, float path) {
    float fraction = clamp(path/max(range, 1e-8), 0.0, 1.0);
    float coordinate = atm.camera.z < top_radius() ? sqrt(fraction) : fraction;
    float z = (0.5+coordinate*float(AERIAL_SIZE.z-1))/float(AERIAL_SIZE.z);
    return textureLod(shadow_aerial_radiance, vec3(uv, z), 0.0).rgb;
}

vec3 shadow_cached_upper_light(vec2 uv, float range, float path) {
    float fraction = clamp(path/max(range, 1e-8), 0.0, 1.0);
    float coordinate = atm.camera.z < top_radius() ? sqrt(fraction) : fraction;
    float slice = min(floor(coordinate*float(AERIAL_SIZE.z-1))+1.0, float(AERIAL_SIZE.z-1));
    return textureLod(shadow_aerial_radiance, vec3(uv, (0.5+slice)/float(AERIAL_SIZE.z)), 0.0).rgb;
}

bool shadow_cached_column_unlit(vec3 direction, vec2 segment, float path) {
    if ((shadow_flags.y & 4u) == 0u || atm.camera.w != 1.0 || path < 2.0 || atm.camera.z-bottom_radius() < 0.01)
        return false;
    vec3 entry = atmosphere_entry(direction, segment.x);
    if (transport_subdivisions(entry, direction, 0.0, path) != 1) return false;
    vec2 uv = sky_uv(direction);
    float half_texel = 0.5/float(AERIAL_SIZE.y);
    float middle = uv.y < 0.5 ? 0.5-half_texel : 0.5+half_texel;
    uv.y = uv.y < 0.5 ? min(uv.y, middle) : max(uv.y, middle);
    float range = atmosphere_span(direction);
    if (abs(uv.y-0.5) < 1.0/float(AERIAL_SIZE.y))
        range = textureLod(shadow_aerial_range, uv, 0.0).r;
    return !any(greaterThan(shadow_cached_upper_light(uv, range, path), vec3(0.0)));
}
