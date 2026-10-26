int shadow_vrs_scale() {
    return 1 << clamp(int((shadow_flags.y >> 8u) & 15u), 1, 4);
}

bool shadow_vrs_spatial_safe(vec3 direction, vec2 segment, float path) {
    vec3 origin = atmosphere_to_physical(atm.camera.xyz+direction*segment.x);
    vec3 end = origin+atmosphere_to_physical(direction)*path;
    float footprint = 1.5*float(shadow_vrs_scale())*max(length(origin-shadow_eye_focal.xyz), length(end-shadow_eye_focal.xyz))/shadow_eye_focal.w;
    vec4 star = shadow_stars[0];
    vec3 light0 = normalize(star.xyz-origin), light1 = normalize(star.xyz-end);
    float angular_radius = star.w/max(length(star.xyz-origin), length(star.xyz-end));
    uvec4 range = shadow_ranges[0];
    for (uint i = 0u; i < range.y; ++i) {
        vec3 center = shadow_casters[range.x+i].position_radius.xyz;
        float a = dot(center-origin, light0), b = dot(center-end, light1);
        if (max(a, b) <= 0.0) continue;
        if (min(a, b)*angular_radius < footprint) return false;
    }
    for (uint i = 0u; i < range.w; ++i) {
        AtmosphereShadowCaster caster = shadow_casters[range.z+i];
        vec3 pole = caster.pole_inner.xyz;
        float a = dot(origin-caster.position_radius.xyz, pole), b = dot(end-caster.position_radius.xyz, pole);
        float d0 = dot(light0, pole), d1 = dot(light1, pole);
        if (a*d0 >= 0.0 && b*d1 >= 0.0) continue;
        float denominator = max(abs(d0), abs(d1));
        if (a*b <= 0.0 || d0*d1 <= 0.0 || denominator < 1e-5) return false;
        float blur = min(abs(a), abs(b))/denominator*angular_radius;
        if (blur < footprint) return false;
    }
    return true;
}

#ifdef ATMOSPHERE_SHADOW_VRS
layout(set = 0, binding = 21) uniform sampler2D coarse_shadow_fraction;
layout(set = 0, binding = 22) uniform sampler2D coarse_shadow_guide;

bool reconstruct_shadow_vrs(ivec2 pixel, float depth, vec3 direction, vec2 segment, float path,
                            out vec3 fraction, out bool supported) {
    int scale = shadow_vrs_scale();
    ivec2 base = pixel/scale;
    vec2 f = vec2(pixel%scale)/float(scale);
    if (all(equal(pixel%scale, ivec2(0)))) {
        vec4 guide = texelFetch(coarse_shadow_guide, base, 0);
        if (guide.w < 0.5) return false;
        vec4 value = texelFetch(coarse_shadow_fraction, base, 0);
        fraction = value.rgb;
        supported = value.a > 0.5;
        return true;
    }
    vec3 minimum = vec3(1.0), maximum = vec3(0.0);
    vec3 sum = vec3(0.0);
    float log_path = log2(max(path, 1e-5));
    ivec2 size = textureSize(coarse_shadow_fraction, 0);
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
        float weight = (x == 0 ? 1.0-f.x : f.x)*(y == 0 ? 1.0-f.y : f.y);
        if (weight <= 0.0) continue;
        ivec2 tap = base+ivec2(x, y);
        if (any(greaterThanEqual(tap, size))) return false;
        vec4 guide = texelFetch(coarse_shadow_guide, tap, 0);
        if (guide.w < 0.5 || (guide.z > 0.5) != (depth > 0.0) || abs(guide.x-log_path) > 0.04 ||
            abs(guide.y-segment.x) > 0.02*max(path, 1.0)) return false;
        vec4 value = texelFetch(coarse_shadow_fraction, tap, 0);
        if (value.a < 0.5) return false;
        minimum = min(minimum, value.rgb);
        maximum = max(maximum, value.rgb);
        sum += weight*value.rgb;
    }
    if (any(greaterThan(maximum-minimum, vec3(0.02)))) return false;
    if (!shadow_vrs_spatial_safe(direction, segment, path)) return false;
    fraction = sum;
    supported = true;
    return true;
}
#endif
