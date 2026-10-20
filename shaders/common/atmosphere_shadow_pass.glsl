#include "common/atmosphere.glsl"
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 3) uniform sampler2D scene_depth;
layout(set = 0, binding = 7) uniform sampler2D transmittance_lut;
layout(set = 0, binding = 8) uniform sampler2D multiscatter_lut;
#include "common/atmosphere_shadow.glsl"
#if defined(ATMOSPHERE_SHADOW_COARSE) || defined(ATMOSPHERE_SHADOW_VRS)
#include "common/atmosphere_shadow_vrs.glsl"
#endif
#ifdef ATMOSPHERE_SHADOW_COARSE
layout(rgba16f, set = 0, binding = 18) uniform writeonly image2D current_deficit;
layout(rgba32f, set = 0, binding = 19) uniform writeonly image2D current_guide;
#else
layout(rgba16f, set = 0, binding = 18) uniform writeonly image2DArray current_deficit;
layout(r32f, set = 0, binding = 19) uniform writeonly image2DArray current_depth;
#endif
#ifdef ATMOSPHERE_SHADOW_TRANSPARENCY
#include "common/transparency.glsl"
#endif
#ifdef ATMOSPHERE_SHADOW_ACCUMULATE
layout(rgba32f, set = 0, binding = 23) uniform image2DArray history_energy;
layout(rgba32f, set = 0, binding = 24) uniform image2DArray history_loss;
layout(push_constant) uniform ShadowHistory { uint reusable; } history;

vec3 accumulate_shadow(ivec2 pixel, int layer, float depth, vec3 energy, vec3 loss, float count) {
    ivec3 p = ivec3(pixel, layer);
    if (count > 0.0) {
        energy += imageLoad(history_energy, p).rgb;
        loss += imageLoad(history_loss, p).rgb;
    }
    imageStore(history_energy, p, vec4(energy, count+1.0));
    imageStore(history_loss, p, vec4(loss, depth));
    return normalized_shadow_loss(loss, energy);
}
#endif

float shadow_endpoint_path(vec2 ndc, float depth, vec2 segment) {
    if (depth <= 0.0) return segment.y-segment.x;
    if (atm.depth_params.y == 0.0) return 0.0;
    vec4 surface = atm.inverse_view_projection*vec4(ndc, depth, 1.0);
    float distance = length(world_to_atmosphere(surface.xyz/surface.w))*149597870.7;
    return max(0.0, min(distance, segment.y)-segment.x);
}

#ifndef ATMOSPHERE_SHADOW_COARSE
void store_shadow(ivec2 pixel, int layer, float depth, vec3 fraction, bool supported) {
    imageStore(current_deficit, ivec3(pixel, layer), vec4(fraction, supported ? 1.0 : 0.0));
    imageStore(current_depth, ivec3(pixel, layer), vec4(depth));
}
#endif

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(current_deficit).xy;
    if (any(greaterThanEqual(pixel, size))) return;
#ifdef ATMOSPHERE_SHADOW_COARSE
    ivec2 output_pixel = pixel;
    pixel *= 2;
    size = textureSize(scene_depth, 0);
#endif
    vec2 ndc = (vec2(pixel)+0.5)/vec2(size)*2.0-1.0;
    vec3 direction = view_direction(ndc);
    vec2 segment = atmosphere_segment(atm.camera.xyz, direction);
    float depth = texelFetch(scene_depth, pixel, 0).r;
    float transparent_depth = -1.0;
#ifdef ATMOSPHERE_SHADOW_TRANSPARENCY
    uint current = imageLoad(transparency_heads, pixel).x;
    while (current != TRANSPARENCY_END) {
        transparent_depth = max(transparent_depth, uintBitsToFloat(fragments[current].z));
        current = fragments[current].w;
    }
#endif
    float path = segment.y > segment.x ? shadow_endpoint_path(ndc, depth, segment) : 0.0;
#ifdef ATMOSPHERE_SHADOW_COARSE
    vec3 unused, fraction = vec3(0.0);
    bvec2 supported = bvec2(false);
    bool valid = path > 0.0 && shadow_vrs_spatial_safe(direction, segment, path);
    if (valid) fraction = atmosphere_shadow_deficit(direction, segment, path, pixel, shadow_temporal.x, 0.0, unused, supported);
    imageStore(current_deficit, output_pixel, vec4(fraction, supported.x ? 1.0 : 0.0));
    imageStore(current_guide, output_pixel, vec4(log2(max(path, 1e-5)), segment.x, depth > 0.0 ? 1.0 : 0.0, valid ? 1.0 : 0.0));
#else
    float prefix_path = transparent_depth >= 0.0 && path > 0.0 ? shadow_endpoint_path(ndc, transparent_depth, segment) : 0.0;
    vec3 prefix_fraction = vec3(0.0), fraction = vec3(0.0);
    bvec2 supported = bvec2(false);
#ifdef ATMOSPHERE_SHADOW_ACCUMULATE
    float count = 0.0;
    if (history.reusable != 0u && imageLoad(history_loss, ivec3(pixel, 0)).a == depth &&
        imageLoad(history_loss, ivec3(pixel, 1)).a == transparent_depth)
        count = imageLoad(history_energy, ivec3(pixel, 0)).a;
    if (count >= 64.0) return;
    AtmosphereShadowEstimate estimate = AtmosphereShadowEstimate(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    if (path > 0.0)
        fraction = atmosphere_shadow_estimate(direction, segment, path, pixel, uint(count),
            prefix_path, prefix_fraction, supported, estimate);
    fraction = accumulate_shadow(pixel, 0, depth, estimate.energy, estimate.loss, count);
    prefix_fraction = accumulate_shadow(pixel, 1, transparent_depth, estimate.prefix_energy, estimate.prefix_loss, count);
#else
    if (path > 0.0) {
        bool reconstructed = false;
#ifdef ATMOSPHERE_SHADOW_VRS
        if (transparent_depth < 0.0)
            reconstructed = reconstruct_shadow_vrs(pixel, depth, direction, segment, path, fraction, supported.x);
#endif
        if (!reconstructed)
            fraction = atmosphere_shadow_deficit(direction, segment, path, pixel, shadow_temporal.x,
                prefix_path, prefix_fraction, supported);
    }
#endif
    store_shadow(pixel, 0, depth, fraction, supported.x);
    store_shadow(pixel, 1, transparent_depth, prefix_fraction, supported.y);
#endif
}

