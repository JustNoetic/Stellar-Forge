#include "common/atmosphere.glsl"
#ifndef ATMOSPHERE_FRAGMENT
layout(local_size_x = 8, local_size_y = 8) in;
#endif
layout(set = 0, binding = 1) uniform sampler2D radiance_lut;
layout(set = 0, binding = 2) uniform sampler2D throughput_lut;
layout(set = 0, binding = 3) uniform sampler2D scene_depth;
#if !defined(ATMOSPHERE_FRAGMENT) && !defined(ATMOSPHERE_TRANSPARENT_ONLY)
layout(rgba16f, set = 0, binding = 4) uniform image2D hdr_color;
#elif defined(ATMOSPHERE_FRAGMENT)
layout(location = 0, index = 0) out vec4 out_scattered;
layout(location = 0, index = 1) out vec4 out_transmittance;
#endif
layout(set = 0, binding = 5) uniform sampler3D aerial_radiance_lut;
layout(set = 0, binding = 6) uniform sampler3D aerial_optical_depth_lut;
layout(set = 0, binding = 7) uniform sampler2D transmittance_lut;
layout(set = 0, binding = 8) uniform sampler2D multiscatter_lut;
layout(set = 0, binding = 9) uniform sampler2D aerial_range_lut;
#ifndef ATMOSPHERE_TRANSPARENCY
#define ATMOSPHERE_SHADOW_DATA_ONLY
#endif
#include "atmosphere_shadow.glsl"
layout(set = 0, binding = 16) uniform sampler2DArray shadow_deficit_lut;
layout(set = 0, binding = 17) uniform sampler2DArray shadow_endpoint_lut;
#ifdef ATMOSPHERE_TRANSPARENCY
#include "common/transparency.glsl"
#endif
#ifdef ATMOSPHERE_SECONDARY
layout(set = 0, binding = 28) uniform sampler2D secondary_sky_lut;
layout(set = 0, binding = 29) uniform sampler3D secondary_aerial_lut;
layout(set = 0, binding = 30) uniform sampler2D secondary_range_lut;
layout(push_constant) uniform SecondaryLighting { vec4 irradiance; } secondary;
#endif

vec4 cubic_weights(float f) {
    float g = 1.0-f;
    return vec4(g*g*g, 3.0*f*f*f-6.0*f*f+4.0, -3.0*f*f*f+3.0*f*f+3.0*f+1.0, f*f*f)/6.0;
}

struct AerialFilter { vec2 x, y, sx, sy; float z; };

AerialFilter aerial_filter(vec3 direction, vec2 segment, float path, ivec3 size, sampler2D ranges) {
    vec2 uv = sky_uv(direction, size.xy);
    float half_texel = 0.5/float(size.y);
    vec2 y_bounds = uv.y < 0.5 ? vec2(half_texel, 0.5-half_texel) : vec2(0.5+half_texel, 1.0-half_texel);
    float range = segment.y-segment.x;
    if (abs(uv.y-0.5) < 1.0/float(size.y) || atm.camera.z-bottom_radius() < 0.01)
        range = textureLod(ranges, uv, 0.0).r;
    float fraction = clamp(path/max(range, 1e-8), 0.0, 1.0);
    float coordinate = atm.camera.z < top_radius() ? sqrt(fraction) : fraction;
    float slice = min(floor(coordinate*float(size.z-1)), float(size.z-2));
    float t0 = aerial_distance(slice/float(size.z-1), range);
    float t1 = aerial_distance((slice+1.0)/float(size.z-1), range);
    float depth_blend = clamp((path-t0)/max(t1-t0, 1e-8), 0.0, 1.0);
    float z = (0.5+slice+depth_blend)/float(size.z);
    vec2 grid = uv*vec2(size.xy)-0.5;
    vec2 base = floor(grid), f = fract(grid);
    vec4 wx = cubic_weights(f.x), wy = cubic_weights(f.y);
    vec2 sx = vec2(wx.x+wx.y, wx.z+wx.w), sy = vec2(wy.x+wy.y, wy.z+wy.w);
    vec2 x = (0.5+mod(base.x+vec2(-1.0, 1.0)+vec2(wx.y, wx.w)/sx, float(size.x-1)))/float(size.x);
    vec2 y = (base.y+vec2(-0.5, 1.5)+vec2(wy.y, wy.w)/sy)/float(size.y);
    y = clamp(y, y_bounds.x, y_bounds.y);
    return AerialFilter(x, y, sx, sy, z);
}

void aerial_transport(vec3 direction, vec2 segment, float path, out vec3 L, out vec3 T) {
    AerialFilter f = aerial_filter(direction, segment, path, AERIAL_SIZE, aerial_range_lut);
    L = vec3(0.0);
    vec3 tau = vec3(0.0);
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
        vec3 lookup = vec3(f.x[i], f.y[j], f.z);
        float weight = f.sx[i]*f.sy[j];
        L += weight*textureLod(aerial_radiance_lut, lookup, 0.0).rgb;
        tau += weight*textureLod(aerial_optical_depth_lut, lookup, 0.0).rgb;
    }
    T = exp(-max(tau, vec3(0.0)));
}

#ifdef ATMOSPHERE_SECONDARY
vec3 secondary_radiance(vec3 direction, vec2 segment, float path, bool finite_endpoint) {
    if (!finite_endpoint)
        return textureLod(secondary_sky_lut, sky_uv(direction, textureSize(secondary_sky_lut, 0)), 0.0).rgb;
    AerialFilter f = aerial_filter(direction, segment, path, textureSize(secondary_aerial_lut, 0), secondary_range_lut);
    vec3 L = vec3(0.0);
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i)
        L += f.sx[i]*f.sy[j]*textureLod(secondary_aerial_lut, vec3(f.x[i], f.y[j], f.z), 0.0).rgb;
    return L;
}
#endif

vec3 finite_transmittance(vec3 x, vec3 p, vec3 direction, vec3 extinction, float path) {
    float rx = clamp(length(x), bottom_radius(), top_radius());
    float rp = clamp(length(p), bottom_radius(), top_radius());
    float mux = clamp(dot(x, direction) / rx, -1.0, 1.0);
    float mup = clamp(dot(p, direction) / rp, -1.0, 1.0);

    bool hits_ground = (mux < 0.0 && rx * rx * (mux * mux - 1.0) + bottom_radius() * bottom_radius() >= 0.0);

    vec3 T_lut = segment_transmittance(transmittance_lut, rx, mux, rp, mup, hits_ground);

    return mix(exp(-extinction * path), T_lut, smoothstep(0.5, 2.0, path));
}

bool atmosphere_transport(ivec2 pixel, ivec2 size, float depth, out vec3 L, out vec3 T) {
    if (any(greaterThanEqual(pixel, size))) return false;
    vec2 ndc = (vec2(pixel)+0.5)/vec2(size)*2.0-1.0;
    vec3 d = view_direction(ndc);
    vec2 segment = atmosphere_segment(atm.camera.xyz, d);
    if (segment.y <= segment.x) return false;
    vec2 uv = sky_uv(d);
    L = textureLod(radiance_lut, uv, 0.0).rgb;
    T = clamp(textureLod(throughput_lut, uv, 0.0).rgb, 0.0, 1.0);
    float path = segment.y-segment.x;
    bool finite_endpoint = false;
    if (depth > 0.0) {
        vec4 surface = atm.inverse_view_projection*vec4(ndc, depth, 1.0);
        float distance = length(world_to_atmosphere(surface.xyz/surface.w))*149597870.7;
        path = min(distance, segment.y)-segment.x;
        if (path <= 0.0) return false;
        if (atm.depth_params.y == 0.0) {
            vec3 x = atm.camera.xyz+d*segment.x;
            vec3 p = atm.camera.xyz+d*(segment.x+path);
            vec3 r, m, e;
            medium((x+p)*0.5, r, m, e);
            T = finite_transmittance(x, p, d, e, path);
            L = vec3(0.0);
            return true;
        }
        if (distance < segment.y) {
            aerial_transport(d, segment, path, L, T);
            finite_endpoint = true;
        }
    }
    if (shadow_flags.x != 0u) {
        vec3 fraction = vec3(0.0);
        bool cached = false;
        for (int layer = 0; layer < 2; ++layer) {
            if (texelFetch(shadow_endpoint_lut, ivec3(pixel, layer), 0).r == depth) {
                fraction = texelFetch(shadow_deficit_lut, ivec3(pixel, layer), 0).rgb;
                cached = true;
                break;
            }
        }
#ifndef ATMOSPHERE_SHADOW_DATA_ONLY
        if (!cached) {
            vec3 unused;
            bvec2 supported;
            fraction = atmosphere_shadow_deficit(d, segment, path, pixel, 0u, 0.0, unused, supported);
        }
#endif
        vec3 deficit = L * fraction;
        L = max(L - deficit, vec3(0.0));
    }
#ifdef ATMOSPHERE_SECONDARY
    L += secondary.irradiance.rgb*secondary_radiance(d, segment, path, finite_endpoint);
#endif
    L *= atm.depth_params.w;
    return true;
}

void main() {
    vec3 L, T;
#ifdef ATMOSPHERE_TRANSPARENCY
    ivec2 transparent_pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 transparent_size = imageSize(transparency_heads);
    if (any(greaterThanEqual(transparent_pixel, transparent_size))) return;
    uint current = imageLoad(transparency_heads, transparent_pixel).x;
    while (current != TRANSPARENCY_END) {
        uvec4 fragment = fragments[current];
        if (atmosphere_transport(transparent_pixel, transparent_size, uintBitsToFloat(fragment.z), L, T)) {
            vec4 color = transparent_color(fragment);
            color.rgb = color.rgb*T+color.a*L*atm.depth_params.x;
            fragments[current].xy = pack_transparent_color(color);
        }
        current = fragment.w;
    }
#endif
#ifdef ATMOSPHERE_FRAGMENT
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (!atmosphere_transport(pixel, textureSize(scene_depth, 0), texelFetch(scene_depth, pixel, 0).r, L, T)) discard;
    out_scattered = vec4(min(L*atm.depth_params.x, vec3(65000.0)), 0.0);
    out_transmittance = vec4(T, 1.0);
#elif !defined(ATMOSPHERE_TRANSPARENT_ONLY)
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, imageSize(hdr_color)))) return;
    if (!atmosphere_transport(pixel, imageSize(hdr_color), texelFetch(scene_depth, pixel, 0).r, L, T)) return;
    vec4 background = imageLoad(hdr_color, pixel);
    imageStore(hdr_color, pixel, vec4(min(background.rgb*T+L*atm.depth_params.x, vec3(65000.0)), background.a));
#endif
}
