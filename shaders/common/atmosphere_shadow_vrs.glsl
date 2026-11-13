int shadow_vrs_scale() {
    return 1 << clamp(int((shadow_flags.y >> 8u) & 15u), 1, 4);
}

bool shadow_vrs_spatial_safe(vec3 direction, vec2 segment, float path) {
    vec3 origin = atmosphere_to_physical(atmosphere_entry(direction, segment.x));
    vec3 end = origin+atmosphere_to_physical(direction)*path;
    float footprint = 1.5*float(shadow_vrs_scale())*max(length(origin-shadow_eye_focal.xyz), length(end-shadow_eye_focal.xyz))/shadow_eye_focal.w;
    vec4 star = shadow_stars[0];
    vec3 light0 = normalize(star.xyz-origin), light1 = normalize(star.xyz-end);
    float angular_radius = star.w/max(length(star.xyz-origin), length(star.xyz-end));
    uvec4 range = shadow_ranges[0];
    for (uint i = 0u; i < range.y; ++i) {
        vec4 caster = shadow_casters[range.x+i].position_radius;
        vec3 center = caster.xyz;
        float a = dot(center-origin, light0), b = dot(center-end, light1);
        if (max(a, b) <= 0.0) continue;
        vec3 axis = normalize(star.xyz-center);
        vec3 perp0 = cross(center-origin, axis), perp1 = cross(center-end, axis);
        vec3 span = perp1-perp0;
        float u = dot(span, span) > 1e-12 ? clamp(-dot(perp0, span)/dot(span, span), 0.0, 1.0) : 0.0;
        float star_dist = max(length(star.xyz), 1e-6);
        float cone = caster.w*(star_dist+top_radius())/max(star_dist-length(center)-top_radius(), 1.0);
        if (length(perp0+span*u) > cone+max(a, b)*angular_radius+2.0*footprint) continue;
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
layout(set = 0, binding = 21) uniform sampler2DArray coarse_shadow_fraction;
layout(set = 0, binding = 22) uniform sampler2DArray coarse_shadow_guide;

// Fixed-rate guided upsample: every pixel blends its 2x2 coarse neighbourhood with bilinear weights scaled by how
// well each tap's visible segment (path length, entry distance, finite/infinite endpoint) matches its own. There is
// no fallback, so the cost is the same for every pixel whatever the shadow geometry.
vec3 reconstruct_shadow_vrs(ivec2 pixel, int layer, bool finite, vec2 segment, float path) {
    int scale = shadow_vrs_scale();
    ivec2 base = pixel/scale;
    vec2 f = vec2(pixel%scale)/float(scale);
    float log_path = log2(max(path, 1e-5));
    float segment_tolerance = max(0.02*max(path, 1.0), 4.0*float(scale)*segment.x/max(shadow_eye_focal.w, 1.0));
    ivec2 lo = max(shadow_coarse_bounds.xy, ivec2(0));
    ivec2 hi = min(shadow_coarse_bounds.zw, textureSize(coarse_shadow_fraction, 0).xy)-1;
    vec3 sum = vec3(0.0);
    float total = 0.0;
    // A thin cloud or transparent fragment the coarse anchors all missed borrows the opaque layer's rays.
    for (int source = layer; source >= 0 && total <= 0.0; source = source == 0 ? -1 : 0)
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
        ivec2 tap = clamp(base+ivec2(x, y), lo, hi);
        vec4 guide = texelFetch(coarse_shadow_guide, ivec3(tap, source), 0);
        if (guide.w < 0.5) continue;
        // A small floor keeps the off-centre taps available when the nearest one belongs to another surface.
        float weight = (x == 0 ? 1.0-f.x : f.x)*(y == 0 ? 1.0-f.y : f.y)+1e-3;
        float mismatch = max(abs(guide.x-log_path)/0.04, abs(guide.y-segment.x)/segment_tolerance);
        weight /= 1.0+mismatch*mismatch*mismatch*mismatch;
        if ((guide.z > 0.5) != finite) weight *= 1e-4;
        sum += weight*texelFetch(coarse_shadow_fraction, ivec3(tap, source), 0).rgb;
        total += weight;
    }
    return total > 0.0 ? sum/total : vec3(0.0);
}
#endif
