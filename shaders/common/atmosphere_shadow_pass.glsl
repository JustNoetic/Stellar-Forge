#include "common/atmosphere.glsl"
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 3) uniform sampler2D scene_depth;
layout(set = 0, binding = 7) uniform sampler2D transmittance_lut;
layout(set = 0, binding = 8) uniform sampler2D multiscatter_lut;
#include "common/atmosphere_shadow.glsl"
layout(rgba16f, set = 0, binding = 18) uniform writeonly image2DArray current_deficit;
layout(r32f, set = 0, binding = 19) uniform writeonly image2DArray current_depth;
#ifdef ATMOSPHERE_SHADOW_TRANSPARENCY
#include "common/transparency.glsl"
#endif

float shadow_endpoint_path(vec2 ndc, float depth, vec2 segment) {
    if (depth <= 0.0) return segment.y-segment.x;
    if (atm.depth_params.y == 0.0) return 0.0;
    vec4 surface = atm.inverse_view_projection*vec4(ndc, depth, 1.0);
    float distance = length(world_to_atmosphere(surface.xyz/surface.w))*149597870.7;
    return max(0.0, min(distance, segment.y)-segment.x);
}

void store_shadow(ivec2 pixel, int layer, float depth, vec3 fraction, bool supported) {
    imageStore(current_deficit, ivec3(pixel, layer), vec4(fraction, supported ? 1.0 : 0.0));
    imageStore(current_depth, ivec3(pixel, layer), vec4(depth));
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(current_deficit).xy;
    if (any(greaterThanEqual(pixel, size))) return;
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
    float prefix_path = transparent_depth >= 0.0 && path > 0.0 ? shadow_endpoint_path(ndc, transparent_depth, segment) : 0.0;
    vec3 prefix_fraction = vec3(0.0), fraction = vec3(0.0);
    bvec2 supported = bvec2(false);
    if (path > 0.0)
        fraction = atmosphere_shadow_deficit(direction, segment, path, pixel, shadow_temporal.x,
            prefix_path, prefix_fraction, supported);
    store_shadow(pixel, 0, depth, fraction, supported.x);
    store_shadow(pixel, 1, transparent_depth, prefix_fraction, supported.y);
}

