// Ground endpoints lie on a thin curved surface. Match their radius in every
// neighboring column, rather than mixing air above and below the ground at one
// camera-distance plane. Reject sky columns that cannot reach that radius.
float aerialSliceRadius(int slice, int slices, vec2 radius_range) {
    return mix(radius_range.x, radius_range.y, float(slice) / float(slices - 1));
}

float aerialRadiusDistance(vec3 O, vec3 V, float radius) {
    float closest = max(0.0, -dot(O, V) / dot(V, V));
    vec2 hit = aerialSphereIntersect(O, V, radius);
    return hit.x <= hit.y ? clamp(hit.x, 0.0, closest) : closest;
}

vec3 aerialColumnRay(vec2 uv, mat4 inv_proj, mat4 inv_view, vec4 pole_scale) {
    vec4 eye = inv_proj * vec4(uv * 2.0 - 1.0, -1.0, 1.0);
    vec3 V = normalize((inv_view * vec4(eye.xy, -1.0, 0.0)).xyz);
    return V + dot(V, pole_scale.xyz) * (pole_scale.w - 1.0) * pole_scale.xyz;
}

vec3 aerialOneMinusExp(vec3 x) {
    return mix(vec3(1.0) - exp(x), -x * (vec3(1.0) + 0.5 * x + x * x / 6.0),
        lessThan(abs(x), vec3(0.001)));
}

bool sampleAerialGround(
    sampler3D scatter_lut, sampler3D trans_lut, vec2 uv, float distance_km,
    vec3 O, vec3 V, float outer_radius, vec2 radius_range,
    mat4 inv_proj, mat4 inv_view, vec4 pole_scale,
    bool read_voxels,
    out vec3 S, out vec3 T)
{
    S = vec3(0.0); T = vec3(1.0);
    vec2 air = aerialRaySpan(O, V, outer_radius);
    float actual_length = max(0.0, distance_km - air.x);
    if (actual_length <= 1e-6) return true;
    vec3 endpoint = O + distance_km * V;
    float radius = length(endpoint);
    float incidence = -dot(endpoint, V) / max(1e-6, radius * length(V));
    // Very long grazing paths have strongly curved density profiles that the
    // slant-length correction cannot recover from a coarse neighbor. Refine
    // this thin limb region with the existing terrain-aware marcher.
    if (actual_length > 25.0 && incidence < 0.15) return false;
    if (dot(endpoint, V) > 0.0 || radius > radius_range.x + 0.001
        || radius < radius_range.y - 0.001) return false;
    radius = clamp(radius, radius_range.y, radius_range.x);
    ivec3 size = textureSize(scatter_lut, 0);
    float z = (radius_range.x - radius) / max(1e-6, radius_range.x - radius_range.y) * float(size.z - 1);
    int lo = clamp(int(floor(z)), 0, size.z - 2);
    float r0 = aerialSliceRadius(lo, size.z, radius_range);
    float r1 = aerialSliceRadius(lo + 1, size.z, radius_range);
    vec2 texel = uv * vec2(size.xy) - 0.5;
    vec2 base = floor(texel), fraction = fract(texel);
    vec3 sum_S = vec3(0.0), sum_T = vec3(0.0);
    float sum_weight = 0.0;
    for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) {
        vec2 column = clamp(base + vec2(x, y), vec2(0.0), vec2(size.xy - 1));
        vec2 column_uv = (column + 0.5) / vec2(size.xy);
        vec3 column_V = aerialColumnRay(column_uv, inv_proj, inv_view, pole_scale);
        if (dot(O, column_V) >= 0.0) continue;
        vec2 hit = aerialSphereIntersect(O, column_V, radius);
        if (hit.x > hit.y || hit.x < -0.001) continue;
        float column_end = max(0.0, hit.x);
        float column_start = aerialRaySpan(O, column_V, outer_radius).x;
        float column_length = column_end - column_start;
        if (column_length <= 1e-6) continue;
        vec2 weight = mix(vec2(1.0) - fraction, fraction, vec2(x, y));
        float xy_weight = weight.x * weight.y;
        if (!read_voxels) {
            sum_weight += xy_weight;
            continue;
        }
        float d0 = aerialRadiusDistance(O, column_V, r0);
        float d1 = aerialRadiusDistance(O, column_V, r1);
        float w = clamp((column_end - d0) / max(1e-6, d1 - d0), 0.0, 1.0);
        // These are exact voxel centers. Hardware trilinear filtering wastes
        // bandwidth and can mix neighboring columns through rounding.
        ivec3 voxel = ivec3(ivec2(column), lo);
        vec3 S0 = texelFetch(scatter_lut, voxel, 0).rgb;
        vec3 S1 = texelFetch(scatter_lut, voxel + ivec3(0, 0, 1), 0).rgb;
        vec3 T0 = texelFetch(trans_lut, voxel, 0).rgb;
        vec3 T1 = texelFetch(trans_lut, voxel + ivec3(0, 0, 1), 0).rgb;
        vec3 log_T0 = log(max(T0, vec3(1e-12)));
        vec3 column_log_T = mix(log_T0, log(max(T1, vec3(1e-12))), w);
        vec3 ratio = T0 * aerialOneMinusExp(column_log_T - log_T0) / max(T0 - T1, vec3(1e-7));
        ratio = mix(vec3(w), ratio, step(vec3(1e-7), T0 - T1));
        vec3 column_S = S0 + (S1 - S0) * ratio;
        // Match the actual finite slant length. Without this correction, coarse
        // downward directions over/under-fog meter-scale horizon ground rays.
        float length_ratio = actual_length / column_length;
        vec3 corrected_log_T = column_log_T * length_ratio;
        vec3 corrected_T = exp(corrected_log_T);
        vec3 attenuation = aerialOneMinusExp(column_log_T);
        vec3 scattering_ratio = aerialOneMinusExp(corrected_log_T) / max(attenuation, vec3(1e-7));
        scattering_ratio = mix(vec3(length_ratio), scattering_ratio, step(vec3(1e-7), attenuation));
        vec3 corrected_S = column_S * scattering_ratio;
        sum_S += corrected_S * xy_weight;
        sum_T += corrected_T * xy_weight;
        sum_weight += xy_weight;
    }
    if (sum_weight < 1e-4) return false;
    if (!read_voxels) return true;
    S = max(vec3(0.0), sum_S / sum_weight);
    T = clamp(sum_T / sum_weight, vec3(0.0), vec3(1.0));
    return true;
}

bool sampleAerialGround(
    sampler3D scatter_lut, sampler3D trans_lut, vec2 uv, float distance_km,
    vec3 O, vec3 V, float outer_radius, vec2 radius_range,
    mat4 inv_proj, mat4 inv_view, vec4 pole_scale, out vec3 S, out vec3 T)
{
    return sampleAerialGround(scatter_lut, trans_lut, uv, distance_km,
        O, V, outer_radius, radius_range, inv_proj, inv_view, pole_scale, true, S, T);
}
