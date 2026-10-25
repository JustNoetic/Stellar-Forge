#include "common/atmosphere.glsl"
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 3) uniform sampler2D scene_depth;
layout(set = 0, binding = 7) uniform sampler2D transmittance_lut;
layout(set = 0, binding = 8) uniform sampler2D multiscatter_lut;
#define ATMOSPHERE_SHADOW_CACHED_WEIGHTS
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
layout(push_constant) uniform ShadowHistory { uint reusable; uint diagnostics; } history;
layout(std430, set = 0, binding = 25) buffer ShadowDiagnostics { uint history_counters[8]; };
shared uvec4 history_group_counts;
shared uint history_group_minimum, history_group_maximum;
shared uvec2 history_group_rotation;
uvec4 history_pixel_counts = uvec4(0u);
uvec2 history_pixel_rotation = uvec2(0u);
uint history_pixel_age = 0u;
#ifdef ATMOSPHERE_SHADOW_ROTATION
layout(rgba32f, set = 0, binding = 26) uniform readonly image2D reprojected_energy;
layout(rgba32f, set = 0, binding = 27) uniform readonly image2D reprojected_loss;
#endif

vec3 accumulate_shadow(ivec2 pixel, int layer, float depth, vec3 energy, vec3 loss, float count) {
    ivec3 p = ivec3(pixel, layer);
    if (count > 0.0) {
#ifdef ATMOSPHERE_SHADOW_ROTATION
        if (layer == 0) {
            energy += imageLoad(reprojected_energy, pixel).rgb;
            loss += imageLoad(reprojected_loss, pixel).rgb;
        }
#else
        energy += imageLoad(history_energy, p).rgb;
        loss += imageLoad(history_loss, p).rgb;
#endif
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

#ifdef ATMOSPHERE_SHADOW_ACCUMULATE
void evaluate_shadow_pixel() {
#else
void main() {
#endif
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
    bool endpoints_changed = false;
#ifdef ATMOSPHERE_SHADOW_ROTATION
    if (depth == 0.0 && transparent_depth < 0.0 && path > 0.0)
        count = imageLoad(reprojected_energy, pixel).a;
#else
    if (history.reusable != 0u) {
        endpoints_changed = imageLoad(history_loss, ivec3(pixel, 0)).a != depth ||
            imageLoad(history_loss, ivec3(pixel, 1)).a != transparent_depth;
        if (!endpoints_changed) count = imageLoad(history_energy, ivec3(pixel, 0)).a;
    }
#endif
    if (count >= 64.0) {
        if (history.diagnostics != 0u && path > 0.0) {
            history_pixel_age = 64u;
            history_pixel_counts = uvec4(1u, 0u, 0u, 1u);
        }
        return;
    }
    AtmosphereShadowEstimate estimate = AtmosphereShadowEstimate(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    uint sequence = uint(count);
#ifdef ATMOSPHERE_SHADOW_ROTATION
    sequence = count > 0.0 ? shadow_temporal.x : 0u;
#endif
    if (path > 0.0)
        fraction = atmosphere_shadow_estimate(direction, segment, path, pixel, sequence,
            prefix_path, prefix_fraction, supported, estimate);
#ifdef ATMOSPHERE_SHADOW_ROTATION
    if (count > 0.0) {
        vec3 previous_energy = imageLoad(reprojected_energy, pixel).rgb;
        vec3 previous_loss = imageLoad(reprojected_loss, pixel).rgb;
        vec3 previous_mean = previous_energy/count;
        vec3 energy_scale = max(max(previous_mean, estimate.energy), vec3(1e-20));
        if (any(greaterThan(abs(normalized_shadow_loss(previous_loss, previous_energy)-fraction), vec3(0.05))) ||
            any(greaterThan(abs(previous_mean-estimate.energy), 0.75*energy_scale))) count = 0.0;
    }
    if (history.diagnostics != 0u && path > 0.0)
        history_pixel_rotation = count > 0.0 ? uvec2(1u, 0u) : uvec2(0u, 1u);
#endif
    if (history.diagnostics != 0u && path > 0.0) {
        history_pixel_age = uint(min(count+1.0, 64.0));
        history_pixel_counts = uvec4(1u, endpoints_changed ? 1u : 0u,
            history_pixel_age < 64u ? 1u : 0u, history_pixel_age == 64u ? 1u : 0u);
    }
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

#ifdef ATMOSPHERE_SHADOW_ACCUMULATE
void main() {
    if (history.diagnostics != 0u) {
        if (gl_LocalInvocationIndex == 0u) {
            history_group_counts = uvec4(0u);
            history_group_rotation = uvec2(0u);
            history_group_minimum = 64u;
            history_group_maximum = 0u;
        }
        barrier();
    }
    evaluate_shadow_pixel();
    if (history.diagnostics != 0u) {
        if (history_pixel_counts.x != 0u) {
            atomicAdd(history_group_counts.x, history_pixel_counts.x);
            atomicAdd(history_group_counts.y, history_pixel_counts.y);
            atomicAdd(history_group_counts.z, history_pixel_counts.z);
            atomicAdd(history_group_counts.w, history_pixel_counts.w);
            atomicMin(history_group_minimum, history_pixel_age);
            atomicMax(history_group_maximum, history_pixel_age);
            atomicAdd(history_group_rotation.x, history_pixel_rotation.x);
            atomicAdd(history_group_rotation.y, history_pixel_rotation.y);
        }
        barrier();
        if (gl_LocalInvocationIndex == 0u && history_group_counts.x != 0u) {
            atomicAdd(history_counters[0], history_group_counts.x);
            atomicAdd(history_counters[1], history_group_counts.y);
            atomicAdd(history_counters[2], history_group_counts.z);
            atomicAdd(history_counters[3], history_group_counts.w);
            atomicMin(history_counters[4], history_group_minimum);
            atomicMax(history_counters[5], history_group_maximum);
            atomicAdd(history_counters[6], history_group_rotation.x);
            atomicAdd(history_counters[7], history_group_rotation.y);
        }
    }
}
#endif

