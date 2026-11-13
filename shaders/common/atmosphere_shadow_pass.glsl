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
layout(rgba16f, set = 0, binding = 18) uniform writeonly image2DArray current_deficit;
layout(rgba32f, set = 0, binding = 19) uniform writeonly image2DArray current_guide;
#else
layout(rgba16f, set = 0, binding = 18) uniform writeonly image2DArray current_deficit;
layout(r32f, set = 0, binding = 19) uniform writeonly image2DArray current_depth;
#endif
#ifdef ATMOSPHERE_SHADOW_TRANSPARENCY
#include "common/transparency.glsl"
#endif
#include "common/atmosphere_tiles.glsl"
layout(set = 0, binding = 34) uniform sampler2D cloud_layer_depth;

// Reverse-Z depth of the 2D cloud sheet composited inline by the atmosphere pass, when it lies in front of the
// opaque scene; -1 otherwise. Its shadowed prefix is stored as layer 2.
float shadow_cloud_depth(ivec2 pixel, float depth) {
    if (atm.cloud_layer.y < 0.5) return -1.0;
    float cloud = texelFetch(cloud_layer_depth, pixel, 0).r;
    return cloud > depth ? cloud : -1.0;
}
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

float shadow_endpoint_path(vec2 ndc, vec3 direction, float depth, vec2 segment, float span) {
    if (depth <= 0.0) return span;
    if (atm.depth_params.y == 0.0) return 0.0;
    vec4 surface = atm.inverse_view_projection*vec4(ndc, depth, 1.0);
    float distance = length(world_to_atmosphere(surface.xyz/surface.w))*149597870.7;
    bool finite_endpoint;
    return max(depth_endpoint_path(distance, direction, segment, span, finite_endpoint), 0.0);
}

#ifndef ATMOSPHERE_SHADOW_COARSE
void store_shadow(ivec2 pixel, int layer, float depth, vec3 fraction, bool supported) {
    imageStore(current_deficit, ivec3(pixel, layer), vec4(fraction, supported ? 1.0 : 0.0));
    imageStore(current_depth, ivec3(pixel, layer), vec4(depth));
}
#endif

// Returns true when a lean VRS resolve leaves this pixel to the deferred full pass.
bool evaluate_shadow_pixel(ivec2 pixel) {
    ivec2 size = imageSize(current_deficit).xy;
#ifdef ATMOSPHERE_SHADOW_COARSE
    if (any(lessThan(pixel, shadow_coarse_bounds.xy)) || any(greaterThanEqual(pixel, min(size, shadow_coarse_bounds.zw)))) return false;
    ivec2 output_pixel = pixel;
    pixel *= shadow_vrs_scale();
    size = textureSize(scene_depth, 0);
#else
    if (any(lessThan(pixel, shadow_bounds.xy)) || any(greaterThanEqual(pixel, min(size, shadow_bounds.zw)))) return false;
#endif
    vec2 ndc = (vec2(pixel)+0.5)/vec2(size)*2.0-1.0;
    vec3 direction = view_direction(ndc);
    float span;
    vec2 segment = atmosphere_segment(atm.camera.xyz, direction, span);
    float depth = texelFetch(scene_depth, pixel, 0).r;
    float path = span > 0.0 ? shadow_endpoint_path(ndc, direction, depth, segment, span) : 0.0;
    float transparent_depth = -1.0;
#ifdef ATMOSPHERE_SHADOW_TRANSPARENCY
    if (path > 0.0) {
        uint current = imageLoad(transparency_heads, pixel).x;
        while (current != TRANSPARENCY_END) {
            transparent_depth = max(transparent_depth, uintBitsToFloat(fragments[current].z));
            current = fragments[current].w;
        }
    }
#endif
    float prefix_path = transparent_depth >= 0.0 && path > 0.0 ? shadow_endpoint_path(ndc, direction, transparent_depth, segment, span) : 0.0;
    float cloud_depth = path > 0.0 ? shadow_cloud_depth(pixel, depth) : -1.0;
    float cloud_path = cloud_depth >= 0.0 ? shadow_endpoint_path(ndc, direction, cloud_depth, segment, span) : 0.0;
#ifdef ATMOSPHERE_SHADOW_COARSE
    vec3 prefix_fraction = vec3(0.0), fraction = vec3(0.0);
    bvec2 supported = bvec2(false);
    bool spatial_safe = path > 0.0 && shadow_vrs_spatial_safe(direction, segment, path);
    bool unlit_safe = !spatial_safe && path > 0.0 && shadow_cached_column_unlit(direction, segment, path);
    bool prefix_safe = prefix_path > 0.0 && shadow_vrs_spatial_safe(direction, segment, prefix_path);
    bool prefix_unlit = prefix_path > 0.0 && !prefix_safe && shadow_cached_column_unlit(direction, segment, prefix_path);
    bool cloud_safe = cloud_path > 0.0 && shadow_vrs_spatial_safe(direction, segment, cloud_path);
    bool cloud_unlit = cloud_path > 0.0 && !cloud_safe && shadow_cached_column_unlit(direction, segment, cloud_path);
    vec3 cloud_fraction = vec3(0.0);
    bool cloud_supported = false;
#ifdef ATMOSPHERE_SHADOW_CLOSED_FORM
    if (spatial_safe || prefix_safe || cloud_safe) {
        bvec3 endpoint_supported;
        closed_form_shadow_endpoints(direction, segment, path, prefix_path, cloud_safe ? cloud_path : 0.0,
            fraction, prefix_fraction, cloud_fraction, endpoint_supported);
        supported = endpoint_supported.xy;
        cloud_supported = endpoint_supported.z;
    }
#else
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 0 ? !(spatial_safe || prefix_safe) : !cloud_safe) continue;
        vec3 second;
        bvec2 flags;
        vec3 first = atmosphere_shadow_deficit(direction, segment, pass == 0 ? path : cloud_path,
            pixel, shadow_temporal.x, pass == 0 ? prefix_path : 0.0, second, flags);
        if (pass == 0) {
            fraction = first;
            prefix_fraction = second;
            supported = flags;
        } else {
            cloud_fraction = first;
            cloud_supported = flags.x;
        }
    }
#endif
    imageStore(current_deficit, ivec3(output_pixel, 0), vec4(spatial_safe ? fraction : vec3(0.0), spatial_safe && supported.x ? 1.0 : 0.0));
    imageStore(current_guide, ivec3(output_pixel, 0), vec4(log2(max(path, 1e-5)), segment.x, depth > 0.0 ? 1.0 : 0.0,
        unlit_safe ? 2.0 : spatial_safe ? 1.0 : 0.0));
    imageStore(current_deficit, ivec3(output_pixel, 1), vec4(prefix_safe ? prefix_fraction : vec3(0.0), prefix_safe && supported.y ? 1.0 : 0.0));
    imageStore(current_guide, ivec3(output_pixel, 1), vec4(log2(max(prefix_path, 1e-5)), segment.x, prefix_path > 0.0 ? 1.0 : 0.0,
        prefix_unlit ? 2.0 : prefix_safe ? 1.0 : 0.0));
    imageStore(current_deficit, ivec3(output_pixel, 2), vec4(cloud_safe ? cloud_fraction : vec3(0.0), cloud_safe && cloud_supported ? 1.0 : 0.0));
    imageStore(current_guide, ivec3(output_pixel, 2), vec4(log2(max(cloud_path, 1e-5)), segment.x, cloud_path > 0.0 ? 1.0 : 0.0,
        cloud_unlit ? 2.0 : cloud_safe ? 1.0 : 0.0));
#else
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
        return false;
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
    store_shadow(pixel, 0, depth, fraction, supported.x);
    store_shadow(pixel, 1, transparent_depth, prefix_fraction, supported.y);
    imageStore(current_depth, ivec3(pixel, 2), vec4(-1.0));
#else
    if (path > 0.0) {
        bool reconstructed = false;
        vec3 cloud_fraction = vec3(0.0);
        bool cloud_supported = false;
#ifdef ATMOSPHERE_SHADOW_VRS
        reconstructed = reconstruct_shadow_vrs(pixel, 0, depth > 0.0, direction, segment, path, fraction, supported.x) &&
            (prefix_path <= 0.0 || reconstruct_shadow_vrs(pixel, 1, true, direction, segment, prefix_path, prefix_fraction, supported.y)) &&
            (cloud_path <= 0.0 || reconstruct_shadow_vrs(pixel, 2, true, direction, segment, cloud_path, cloud_fraction, cloud_supported));
        if (!reconstructed) {
            fraction = prefix_fraction = cloud_fraction = vec3(0.0);
            supported = bvec2(false);
            cloud_supported = false;
        }
#endif
#ifdef ATMOSPHERE_TILE_LEAN
        if (!reconstructed && !shadow_cached_column_unlit(direction, segment, path)) return true;
#else
        if (!reconstructed && !shadow_cached_column_unlit(direction, segment, path)) {
#ifdef ATMOSPHERE_SHADOW_CLOSED_FORM
            bvec3 endpoint_supported;
            closed_form_shadow_endpoints(direction, segment, path, prefix_path, cloud_path,
                fraction, prefix_fraction, cloud_fraction, endpoint_supported);
            supported = endpoint_supported.xy;
            cloud_supported = endpoint_supported.z;
#else
            for (int pass = 0; pass < (cloud_path > 0.0 ? 2 : 1); ++pass) {
                vec3 second;
                bvec2 flags;
                vec3 first = atmosphere_shadow_deficit(direction, segment, pass == 0 ? path : cloud_path,
                    pixel, shadow_temporal.x, pass == 0 ? prefix_path : 0.0, second, flags);
                if (pass == 0) {
                    fraction = first;
                    prefix_fraction = second;
                    supported = flags;
                } else {
                    cloud_fraction = first;
                    cloud_supported = flags.x;
                }
            }
#endif
        }
#endif
        store_shadow(pixel, 0, depth, fraction, supported.x);
        if (cloud_path > 0.0)
            store_shadow(pixel, 2, cloud_depth, cloud_fraction, cloud_supported);
        else
            imageStore(current_depth, ivec3(pixel, 2), vec4(-1.0));
#ifdef ATMOSPHERE_SHADOW_TRANSPARENCY
        if (prefix_path > 0.0)
            store_shadow(pixel, 1, transparent_depth, prefix_fraction, supported.y);
        else
            imageStore(current_depth, ivec3(pixel, 1), vec4(-1.0));
#endif
    }
#endif
#endif
    return false;
}

#ifndef ATMOSPHERE_SHADOW_ACCUMULATE
void main() {
#if defined(ATMOSPHERE_TILE_DEFERRED)
    evaluate_shadow_pixel(shadow_bounds.xy+atmosphere_tile_pixel());
#elif defined(ATMOSPHERE_TILE_LEAN)
    atmosphere_tile_begin();
    atmosphere_tile_end(evaluate_shadow_pixel(shadow_bounds.xy+ivec2(gl_GlobalInvocationID.xy)));
#elif defined(ATMOSPHERE_SHADOW_COARSE)
    evaluate_shadow_pixel(shadow_coarse_bounds.xy+ivec2(gl_GlobalInvocationID.xy));
#else
    evaluate_shadow_pixel(shadow_bounds.xy+ivec2(gl_GlobalInvocationID.xy));
#endif
}
#else
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
    evaluate_shadow_pixel(shadow_bounds.xy+ivec2(gl_GlobalInvocationID.xy));
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

