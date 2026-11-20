#include "common/atmosphere.glsl"
#ifndef ATMOSPHERE_FRAGMENT
layout(local_size_x = 8, local_size_y = 8) in;
#endif
layout(set = 0, binding = 1) uniform sampler2D radiance_lut;
layout(set = 0, binding = 2) uniform sampler2D throughput_lut;
layout(set = 0, binding = 3) uniform sampler2D scene_depth;
#if !defined(ATMOSPHERE_FRAGMENT) && !defined(ATMOSPHERE_TRANSPARENT_ONLY)
#ifdef HDR_RGBA32F
layout(rgba32f, set = 0, binding = 4) uniform image2D hdr_color;
#else
layout(rgba16f, set = 0, binding = 4) uniform image2D hdr_color;
#endif
#elif defined(ATMOSPHERE_FRAGMENT)
layout(location = 0, index = 0) out vec4 out_scattered;
layout(location = 0, index = 1) out vec4 out_transmittance;
#endif
layout(set = 0, binding = 5) uniform sampler3D aerial_radiance_lut;
layout(set = 0, binding = 6) uniform sampler3D aerial_optical_depth_lut;
layout(set = 0, binding = 7) uniform sampler2D transmittance_lut;
layout(set = 0, binding = 8) uniform sampler2D multiscatter_lut;
layout(set = 0, binding = 9) uniform sampler2D aerial_range_lut;
#if !defined(ATMOSPHERE_TRANSPARENCY) || defined(ATMOSPHERE_UNSHADOWED) || defined(ATMOSPHERE_TILE_LEAN)
#define ATMOSPHERE_SHADOW_DATA_ONLY
#endif
#include "atmosphere_shadow.glsl"
layout(set = 0, binding = 16) uniform sampler2DArray shadow_deficit_lut;
layout(set = 0, binding = 17) uniform sampler2DArray shadow_endpoint_lut;
layout(set = 0, binding = 33) uniform sampler2D cloud_layer_color;
layout(set = 0, binding = 34) uniform sampler2D cloud_layer_depth;
#if defined(ATMOSPHERE_RING_SLOT) || defined(ATMOSPHERE_TRANSPARENCY)
#define TRANSPARENCY_RING_SLOT
#endif
#if defined(ATMOSPHERE_TRANSPARENCY) || defined(ATMOSPHERE_RING_SLOT)
#include "common/transparency.glsl"
#endif
#if defined(ATMOSPHERE_TILE_LEAN) || defined(ATMOSPHERE_TILE_DEFERRED)
#include "common/atmosphere_tiles.glsl"
const uint TRANSPARENT_DEFERRED_BIT = 0x80000000u;
#endif
bool shadow_cache_missed = false;
#ifdef ATMOSPHERE_SECONDARY
layout(set = 0, binding = 28) uniform sampler2D secondary_sky_lut;
layout(set = 0, binding = 29) uniform sampler3D secondary_aerial_lut;
layout(set = 0, binding = 30) uniform sampler2D secondary_range_lut;
#endif
#if defined(ATMOSPHERE_SECONDARY) || defined(ATMOSPHERE_RING_SLOT)
// ring.x: 0 ignores the ring slot, 1 transports it in place, 2 also blends it where no list fragment exists.
layout(push_constant) uniform CompositePush {
    vec4 irradiance;
#ifdef ATMOSPHERE_RING_SLOT
    uvec4 ring;
#endif
} secondary;
#endif

vec4 cubic_weights(float f) {
    float g = 1.0-f;
    return vec4(g*g*g, 3.0*f*f*f-6.0*f*f+4.0, -3.0*f*f*f+3.0*f*f+3.0*f+1.0, f*f*f)/6.0;
}

struct AerialFilterXY {
    vec2 x, y, sx, sy;
    float range;
    bool valid;
};

struct TransportCache {
    AerialFilterXY aerial;
#ifdef ATMOSPHERE_SECONDARY
    AerialFilterXY secondary;
#endif
};

TransportCache init_transport_cache() {
    TransportCache c;
    c.aerial.valid = false;
#ifdef ATMOSPHERE_SECONDARY
    c.secondary.valid = false;
#endif
    return c;
}

AerialFilterXY sky_filter_xy(vec3 direction, ivec2 size, out vec2 uv) {
    AerialFilterXY f;
    uv = sky_uv(direction, size);
    f.range = 0.0;
    vec2 grid = uv*vec2(size)-0.5;
    vec2 base = floor(grid), t = fract(grid);
    vec4 wx = cubic_weights(t.x), wy = cubic_weights(t.y);
    f.sx = vec2(wx.x+wx.y, wx.z+wx.w);
    f.sy = vec2(wy.x+wy.y, wy.z+wy.w);
    f.x = (0.5+mod(base.x+vec2(-1.0, 1.0)+vec2(wx.y, wx.w)/f.sx, float(size.x-1)))/float(size.x);
    float half_texel = 0.5/float(size.y);
    vec2 y_bounds = uv.y < 0.5 ? vec2(half_texel, 0.5-half_texel) : vec2(0.5+half_texel, 1.0-half_texel);
    f.y = clamp((base.y+vec2(-0.5, 1.5)+vec2(wy.y, wy.w)/f.sy)/float(size.y), y_bounds.x, y_bounds.y);
    f.valid = true;
    return f;
}

AerialFilterXY aerial_filter_xy(vec3 direction, float range, ivec3 size, sampler2D ranges) {
    vec2 uv;
    AerialFilterXY f = sky_filter_xy(direction, size.xy, uv);
    f.range = (abs(uv.y-0.5) < 1.0/float(size.y) || atm.camera.z-bottom_radius() < 0.01)
        ? textureLod(ranges, uv, 0.0).r : range;
    return f;
}

void sky_transport(vec3 direction, out vec3 L, out vec3 T) {
    vec2 uv;
    AerialFilterXY f = sky_filter_xy(direction, SKY_SIZE, uv);
    L = vec3(0.0);
    T = vec3(0.0);
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
        vec2 lookup = vec2(f.x[i], f.y[j]);
        float weight = f.sx[i]*f.sy[j];
        L += weight*textureLod(radiance_lut, lookup, 0.0).rgb;
        T += weight*clamp(textureLod(throughput_lut, lookup, 0.0).rgb, 0.0, 1.0);
    }
    T = clamp(T, 0.0, 1.0);
}

float aerial_filter_z(float range, float path, int depth_slices) {
    float fraction = clamp(path/max(range, 1e-8), 0.0, 1.0);
    float coordinate = atm.camera.z < top_radius() ? sqrt(fraction) : fraction;
    float slice = min(floor(coordinate*float(depth_slices-1)), float(depth_slices-2));
    float t0 = aerial_distance(slice/float(depth_slices-1), range);
    float t1 = aerial_distance((slice+1.0)/float(depth_slices-1), range);
    float depth_blend = clamp((path-t0)/max(t1-t0, 1e-8), 0.0, 1.0);
    return (0.5+slice+depth_blend)/float(depth_slices);
}

void aerial_transport(vec3 direction, float span, float path, inout AerialFilterXY f, out vec3 L, out vec3 T) {
    if (!f.valid) f = aerial_filter_xy(direction, span, AERIAL_SIZE, aerial_range_lut);
    float z = aerial_filter_z(f.range, path, AERIAL_SIZE.z);
    L = vec3(0.0);
    vec3 tau = vec3(0.0);
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
        vec3 lookup = vec3(f.x[i], f.y[j], z);
        float weight = f.sx[i]*f.sy[j];
        L += weight*textureLod(aerial_radiance_lut, lookup, 0.0).rgb;
        tau += weight*textureLod(aerial_optical_depth_lut, lookup, 0.0).rgb;
    }
    T = exp(-max(tau, vec3(0.0)));
}

#ifdef ATMOSPHERE_SECONDARY
vec3 secondary_radiance(vec3 direction, float span, float path, bool finite_endpoint, inout AerialFilterXY f) {
    if (!finite_endpoint) {
        vec2 uv;
        AerialFilterXY sf = sky_filter_xy(direction, textureSize(secondary_sky_lut, 0), uv);
        vec3 L = vec3(0.0);
        for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i)
            L += sf.sx[i]*sf.sy[j]*textureLod(secondary_sky_lut, vec2(sf.x[i], sf.y[j]), 0.0).rgb;
        return L;
    }
    ivec3 size = textureSize(secondary_aerial_lut, 0);
    if (!f.valid) f = aerial_filter_xy(direction, span, size, secondary_range_lut);
    float z = aerial_filter_z(f.range, path, size.z);
    vec3 L = vec3(0.0);
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i)
        L += f.sx[i]*f.sy[j]*textureLod(secondary_aerial_lut, vec3(f.x[i], f.y[j], z), 0.0).rgb;
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

void atmosphere_shadow_loss(ivec2 pixel, float depth, int preferred_layer, inout vec3 L) {
#ifndef ATMOSPHERE_UNSHADOWED
    if (shadow_flags.x != 0u) {
        if (any(lessThan(pixel, shadow_bounds.xy)) || any(greaterThanEqual(pixel, shadow_bounds.zw))) return;
        if (any(greaterThan(L, vec3(0.0)))) {
            vec3 fraction = vec3(0.0);
            bool cached = false;
            for (int k = 0; k < 3; ++k) {
                int layer = preferred_layer + k;
                if (layer >= 3) layer -= 3;
                if (texelFetch(shadow_endpoint_lut, ivec3(pixel, layer), 0).r == depth) {
                    fraction = texelFetch(shadow_deficit_lut, ivec3(pixel, layer), 0).rgb;
                    cached = true;
                    break;
                }
            }
            shadow_cache_missed = shadow_cache_missed || !cached;
            vec3 deficit = L * fraction;
            L = max(L - deficit, vec3(0.0));
        } else {
            L = max(L, vec3(0.0));
        }
    }
#endif
}

bool atmosphere_transport_ray_base(ivec2 pixel, vec2 ndc, vec3 d, vec2 segment, float span, float depth,
                                   int shadow_layer, inout TransportCache cache,
                                   out vec3 L, out vec3 T, out float path, out bool finite_endpoint) {
    path = span;
    finite_endpoint = false;
    if (depth > 0.0) {
        float distance = depth_atmosphere_distance(ndc, depth, d);
        path = depth_endpoint_path(distance, d, segment, span, finite_endpoint);
        if (path <= 0.0) return false;
        if (atm.depth_params.y == 0.0) {
            vec3 x = atmosphere_entry(d, segment.x);
            vec3 p = x+d*path;
            vec3 r, m, e;
            medium((x+p)*0.5, r, m, e);
            T = finite_transmittance(x, p, d, e, path);
            L = vec3(0.0);
            return true;
        }
    }
    if (finite_endpoint) {
        aerial_transport(d, span, path, cache.aerial, L, T);
    } else {
        sky_transport(d, L, T);
    }
    atmosphere_shadow_loss(pixel, finite_endpoint ? depth : 0.0, shadow_layer, L);
    return true;
}

void atmosphere_finish_radiance(vec3 d, float span, float path, bool finite_endpoint, float depth,
                                inout TransportCache cache, inout vec3 L) {
#ifdef ATMOSPHERE_SECONDARY
    // Matches the base early return: surface pixels without aerial perspective carry no in-scatter.
    if (depth <= 0.0 || atm.depth_params.y != 0.0)
        L += secondary.irradiance.rgb*secondary_radiance(d, span, path, finite_endpoint, cache.secondary);
#endif
    L *= atm.depth_params.w;
}

bool atmosphere_transport_ray(ivec2 pixel, vec2 ndc, vec3 d, vec2 segment, float span, float depth,
                              int shadow_layer, inout TransportCache cache, out vec3 L, out vec3 T) {
    float path;
    bool finite_endpoint;
    if (!atmosphere_transport_ray_base(pixel, ndc, d, segment, span, depth, shadow_layer, cache, L, T, path, finite_endpoint))
        return false;
    atmosphere_finish_radiance(d, span, path, finite_endpoint, depth, cache, L);
    return true;
}

#ifdef ATMOSPHERE_LIMB
const float LIMB_RESOLVED_PIXELS = 32.0;
const int LIMB_MAX_TAPS = 16;
const int LIMB_BEND_ROWS = 64;

struct LimbFootprint { vec2 across; float center, wide, narrow, limb, edge, azimuth; bool resolved; };

bool limb_footprint(vec2 ndc, vec3 d, float refract_slope, ivec2 size, out LimbFootprint f) {
    float sine = length(d.xy);
    if (sine < atm.limb_image.z || sine > atm.limb_image.w || d.z >= 0.0) return false;
    float r = atm.camera.z;
    f.limb = precise_asin(clamp(bottom_radius()/r, 0.0, 1.0));
    f.edge = precise_asin(clamp(top_radius()/r, 0.0, 1.0));
    f.center = exterior_sky_angle(d);
    vec3 row_x = vec3(atm.view_ray[0].x, atm.view_ray[1].x, atm.view_ray[2].x);
    vec3 row_y = vec3(atm.view_ray[0].y, atm.view_ray[1].y, atm.view_ray[2].y);
    vec3 row_z = vec3(atm.view_ray[0].z, atm.view_ray[1].z, atm.view_ray[2].z);
    f.across = d.xy/max(sine, 1e-30);
    vec3 radial = vec3(f.across*-d.z, sine);
    vec2 steps = refract_limb_steps(ndc, size, radial, refract_slope,
        vec2(dot(radial, row_x), dot(radial, row_y)), length(row_x*ndc.x+row_y*ndc.y+row_z));
    f.wide = max(steps.x, steps.y);
    f.narrow = max(min(steps.x, steps.y), 1e-3*f.wide);
    float reach = 0.5*(f.wide+f.narrow);
    if (f.center+reach <= f.limb || f.center-reach >= f.edge) return false;
    f.resolved = f.edge-f.limb > LIMB_RESOLVED_PIXELS*f.wide;
    if (f.resolved && atm.limb_image.x < 0.5) return false;
    f.azimuth = (atan(d.y, d.x)+ATM_PI)/(2.0*ATM_PI);
    return true;
}

float limb_coverage(LimbFootprint f, float angle) {
    float offset = angle-f.center, x = abs(offset);
    float outside = max(0.5*(f.wide+f.narrow)-x, 0.0);
    float area = x <= 0.5*(f.wide-f.narrow) ? x/f.wide : 0.5-0.5*outside*outside/(f.wide*f.narrow);
    return 0.5+sign(offset)*area;
}

float limb_row(LimbFootprint f, bool lower, float angle) {
    return lower ? sqrt(clamp((angle-f.limb)/(f.edge-f.limb), 0.0, 1.0)) : 1.0-sqrt(clamp(1.0-angle/f.limb, 0.0, 1.0));
}

float limb_row_angle(LimbFootprint f, bool lower, float t) {
    return lower ? f.limb+(f.edge-f.limb)*t*t : f.limb*t*(2.0-t);
}

void limb_integrate(LimbFootprint f, bool lower, inout vec3 L, out vec3 T, out float weight) {
    float reach = 0.5*(f.wide+f.narrow);
    float first = limb_row(f, lower, lower ? max(f.center-reach, f.limb) : f.center-reach);
    float last = limb_row(f, lower, lower ? f.center+reach : min(f.center+reach, f.limb));
    T = vec3(0.0);
    weight = 0.0;
    if (last <= first) return;
    int taps = clamp(int(ceil((last-first)*float(SKY_SIZE.y/2-1))), 1, LIMB_MAX_TAPS);
    float below = limb_coverage(f, limb_row_angle(f, lower, first));
    for (int i = 0; i < taps; ++i) {
        float above = limb_coverage(f, limb_row_angle(f, lower, mix(first, last, float(i+1)/float(taps))));
        float t = mix(first, last, (float(i)+0.5)/float(taps));
        float share = above-below;
        vec2 uv = sky_grid_uv(f.azimuth, t, lower, SKY_SIZE);
        L += share*textureLod(radiance_lut, uv, 0.0).rgb;
        T += share*clamp(textureLod(throughput_lut, uv, 0.0).rgb, 0.0, 1.0);
#ifdef ATMOSPHERE_SECONDARY
        L += share*secondary.irradiance.rgb*textureLod(secondary_sky_lut, sky_grid_uv(f.azimuth, t, lower, textureSize(secondary_sky_lut, 0)), 0.0).rgb;
#endif
        weight += share;
        below = above;
    }
}

vec3 limb_pole() {
    return vec3(dot(atm.world_pole.xyz, atm.axis_x.xyz), dot(atm.world_pole.xyz, atm.axis_y.xyz), dot(atm.world_pole.xyz, atm.axis_z.xyz));
}

vec3 limb_bend_at(float t) {
    float row = t*float(LIMB_BEND_ROWS-1);
    int index = min(int(row), LIMB_BEND_ROWS-2);
    return mix(atm.limb_bend[index].rgb, atm.limb_bend[index+1].rgb, row-float(index));
}

const int LIMB_SPECTRAL_SAMPLES = 16;

const vec2 LIMB_SPEC_DISP[16] = vec2[16](
    vec2(+0.0177838, 0.0018827),
    vec2(+0.0143002, 0.0016149),
    vec2(+0.0112996, 0.0013964),
    vec2(+0.0086954, 0.0012162),
    vec2(+0.0064198, 0.0010661),
    vec2(+0.0044190, 0.0009400),
    vec2(+0.0026501, 0.0008333),
    vec2(+0.0010781, 0.0007423),
    vec2(-0.0003254, 0.0006642),
    vec2(-0.0015839, 0.0005967),
    vec2(-0.0027169, 0.0005382),
    vec2(-0.0037406, 0.0004872),
    vec2(-0.0046687, 0.0004424),
    vec2(-0.0055129, 0.0004030),
    vec2(-0.0062831, 0.0003682),
    vec2(-0.0069877, 0.0003373));

const vec3 LIMB_SPEC_TAU[16] = vec3[16](
    vec3(+0.941909, -2.109551, +2.167642),
    vec3(+0.368136, -0.878957, +1.510821),
    vec3(+0.020376, -0.052737, +1.032361),
    vec3(-0.168256, +0.483643, +0.684613),
    vec3(-0.244744, +0.811282, +0.433462),
    vec3(-0.242404, +0.988205, +0.254198),
    vec3(-0.185039, +1.056306, +0.128733),
    vec3(-0.089769, +1.046062, +0.043706),
    vec3(+0.031042, +0.979795, -0.010837),
    vec3(+0.168432, +0.873934, -0.042366),
    vec3(+0.315901, +0.740624, -0.056525),
    vec3(+0.468737, +0.588863, -0.057600),
    vec3(+0.623537, +0.425328, -0.048865),
    vec3(+0.777859, +0.254971, -0.032829),
    vec3(+0.929968, +0.081454, -0.011423),
    vec3(+1.078657, -0.092524, +0.013867));

const vec3 LIMB_SPEC_RGB[16] = vec3[16](
    vec3(+0.000719, -0.001140, +0.012524),
    vec3(+0.017616, -0.017522, +0.127470),
    vec3(+0.039457, -0.045465, +0.367013),
    vec3(+0.005446, -0.023604, +0.364184),
    vec3(-0.037944, +0.028553, +0.195334),
    vec3(-0.086282, +0.102445, +0.057638),
    vec3(-0.131352, +0.216156, -0.001767),
    vec3(-0.107294, +0.288862, -0.029038),
    vec3(+0.018233, +0.262936, -0.033942),
    vec3(+0.194881, +0.175771, -0.028275),
    vec3(+0.341658, +0.065212, -0.017838),
    vec3(+0.364740, -0.012882, -0.008185),
    vec3(+0.242227, -0.026248, -0.003104),
    vec3(+0.103899, -0.011861, -0.001252),
    vec3(+0.028916, -0.001744, -0.000554),
    vec3(+0.005081, +0.000531, -0.000208));

float limb_bin_integral(float z) {
    if (z <= 0.0) return 0.0;
    if (z >= 1.0) return z - 0.5;
    return 0.5*z*z;
}

float limb_bin_t(float target, float start, float slope, float hw) {
    if (hw <= 1e-4*slope) return clamp((target-start)/slope, 0.0, 1.0);
    if (slope <= 1e-4*hw) return clamp((target-(start+0.5*slope)+hw)/(2.0*hw), 0.0, 1.0);
    return clamp((slope/(2.0*hw))*(limb_bin_integral((target+hw-start)/slope)-limb_bin_integral((target-hw-start)/slope)), 0.0, 1.0);
}

float limb_spectral_transmit(vec3 tau, int k) {
    float tau_k = max(dot(LIMB_SPEC_TAU[k], tau), 0.0);
    if (k < 3 && tau.b >= tau.g) tau_k = max(tau_k, tau.b);
    return exp(-tau_k);
}

vec3 limb_star_image(LimbFootprint f) {
    if (atm.limb_image.x < 0.5) return vec3(0.0);
    float reach = 0.5*(f.wide+f.narrow);
    float first = limb_row(f, true, f.center-reach), last = limb_row(f, true, f.center+reach);
    if (last <= first) return vec3(0.0);
    float rows = float(LIMB_BEND_ROWS-1);
    int begin = int(first*rows), end = min(int(ceil(last*rows)), LIMB_BEND_ROWS-1);
    float lowest = limb_row_angle(f, true, first), highest = limb_row_angle(f, true, last);
    vec3 high_bend = limb_bend_at(last);
    float exit_low = lowest-1.02*atm.limb_image.y;
    float exit_high = highest-0.99*min(high_bend.r, min(high_bend.g, high_bend.b));
    vec3 image = vec3(0.0);
    vec3 pole = limb_pole();
    vec3 plane = vec3(-f.across.y, f.across.x, 0.0);
    plane = normalize(plane+pole*(dot(plane, pole)*atm.world_pole.w));
    for (uint star = 0u; star < uint(atm.camera.w); ++star) {
        vec3 radiance = atm.star_radiance[star].rgb;
        vec4 sun = atm.star_disc[star];
        float side = dot(plane, sun.xyz);
        if (!any(greaterThan(radiance, vec3(0.0))) || abs(side) >= sun.w) continue;
        float true_chord = sqrt(sun.w*sun.w-side*side);
        vec3 nearest = normalize(sun.xyz-plane*side), along = cross(plane, nearest);
        vec3 near_end = nearest*cos(true_chord)-along*sin(true_chord), far_end = nearest*cos(true_chord)+along*sin(true_chord);
        near_end += pole*(dot(near_end, pole)*atm.world_pole.w);
        far_end += pole*(dot(far_end, pole)*atm.world_pole.w);
        float s0 = atan(dot(f.across, near_end.xy), -near_end.z), s1 = atan(dot(f.across, far_end.xy), -far_end.z);
        float center = 0.5*(s0+s1), chord = 0.5*abs(s1-s0);
        float to_true = true_chord/max(chord, 1e-30);
        if (center+chord < exit_low || center-chord > exit_high) continue;
        vec3 star_image = vec3(0.0);
        for (int row = begin; row < end; ++row) {
            if (atm.limb_bend[row].w < 0.5 || atm.limb_bend[row+1].w < 0.5) continue;
            float t0 = max(float(row)/rows, first), t1 = min(float(row+1)/rows, last);
            if (t1 <= t0) continue;
            float a0 = limb_row_angle(f, true, t0), a1 = limb_row_angle(f, true, t1);
            vec3 b0 = limb_bend_at(t0), b1 = limb_bend_at(t1);
            float s0_r = max((b0.g-b0.r)*151.0493, 0.0), s0_b = b0.b > b0.g ? (b0.b-b0.g)*90.2860 : s0_r;
            float s1_r = max((b1.g-b1.r)*151.0493, 0.0), s1_b = b1.b > b1.g ? (b1.b-b1.g)*90.2860 : s1_r;
            float b_hi0 = b0.g+0.0197407*s0_b, b_hi1 = b1.g+0.0197407*s1_b;
            float b_lo0 = b0.g-0.0073177*s0_r, b_lo1 = b1.g-0.0073177*s1_r;
            if (center+chord < min(a0-b_hi0, a1-b_hi1) || center-chord > max(a0-b_lo0, a1-b_lo1)) continue;
            if (center-chord <= min(a0-b_hi0, a1-b_hi1) && center+chord >= max(a0-b_lo0, a1-b_lo1)) {
                float share = limb_coverage(f, a1)-limb_coverage(f, a0);
                vec2 uv = sky_grid_uv(f.azimuth, 0.5*(t0+t1), true, SKY_SIZE);
                vec3 transmitted_rgb = clamp(textureLod(throughput_lut, uv, 0.0).rgb, 0.0, 1.0);
                if (!any(greaterThan(transmitted_rgb, vec3(1e-7)))) continue;
                vec3 tau = -log(max(transmitted_rgb, vec3(1e-10)));
                float mid_a = 0.5*(a0+a1)-0.5*(b0.g+b1.g);
                float slope_r = 0.5*(s0_r+s1_r), slope_b = 0.5*(s0_b+s1_b);
                for (int k = 0; k < LIMB_SPECTRAL_SAMPLES; ++k) {
                    float dd = LIMB_SPEC_DISP[k].x;
                    float offset = (mid_a-dd*(dd >= 0.0 ? slope_b : slope_r)-center)*to_true;
                    float rim = 1.0-sqrt(max(1.0-(side*side+offset*offset)/(sun.w*sun.w), 0.0));
                    star_image += share*limb_spectral_transmit(tau, k)*(1.0-0.4*rim-0.2*rim*rim)*LIMB_SPEC_RGB[k];
                }
                continue;
            }
            for (int k = 0; k < LIMB_SPECTRAL_SAMPLES; ++k) {
                vec2 disp = LIMB_SPEC_DISP[k];
                float start = a0-(b0.g+disp.x*(disp.x >= 0.0 ? s0_b : s0_r));
                float end_s = a1-(b1.g+disp.x*(disp.x >= 0.0 ? s1_b : s1_r));
                float slope = max(end_s-start, 1e-12);
                float hw = disp.y*(disp.x >= 0.0 ? 0.5*(s0_b+s1_b) : 0.5*(s0_r+s1_r));
                float enter = limb_bin_t(center-chord, start, slope, hw);
                float leave = limb_bin_t(center+chord, start, slope, hw);
                if (leave <= enter) continue;
                float cov_share = limb_coverage(f, mix(a0, a1, leave))-limb_coverage(f, mix(a0, a1, enter));
                float row_share = (limb_coverage(f, a1)-limb_coverage(f, a0))*(leave-enter);
                float share = mix(cov_share, row_share, smoothstep(0.1*slope, 0.5*slope, hw));
                float middle = 0.5*(enter+leave);
                float offset = (start+slope*middle-center)*to_true;
                float rim = 1.0-sqrt(max(1.0-(side*side+offset*offset)/(sun.w*sun.w), 0.0));
                vec2 uv = sky_grid_uv(f.azimuth, mix(t0, t1, middle), true, SKY_SIZE);
                vec3 transmitted_rgb = clamp(textureLod(throughput_lut, uv, 0.0).rgb, 0.0, 1.0);
                if (!any(greaterThan(transmitted_rgb, vec3(1e-7)))) continue;
                vec3 tau = -log(max(transmitted_rgb, vec3(1e-10)));
                star_image += share*limb_spectral_transmit(tau, k)*(1.0-0.4*rim-0.2*rim*rim)*LIMB_SPEC_RGB[k];
            }
        }
        image += max(star_image, vec3(0.0))*radiance;
    }
    return image;
}

bool limb_star_behind(vec3 d) {
    if (atm.limb_image.x < 0.5) return false;
    vec3 pole = limb_pole();
    vec3 seen = normalize(d-pole*(dot(d, pole)*atm.world_pole.w/(1.0+atm.world_pole.w)));
    for (uint star = 0u; star < uint(atm.camera.w); ++star) {
        vec4 sun = atm.star_disc[star];
        if (dot(seen, sun.xyz) > 0.0 && length(cross(seen, sun.xyz)) < sun.w) return true;
    }
    return false;
}

float limb_depth_distance(vec2 ndc, float depth, vec3 d) {
    return depth_atmosphere_distance(ndc, depth, d);
}

bool limb_transport(ivec2 pixel, LimbFootprint f, vec2 ndc, vec3 d, vec2 segment, float span, float depth,
                    int shadow_layer, inout TransportCache cache, out vec3 L, out vec3 T) {
    float path = span;
    bool surface_seen = false, finite_endpoint = false;
    if (depth > 0.0) {
        float distance = limb_depth_distance(ndc, depth, d);
        float closest = -dot(atm.camera.xyz, d);
        if (distance < closest-top_radius()) return false;
        surface_seen = distance < closest+top_radius();
        if (span > 0.0) path = depth_endpoint_path(distance, d, segment, span, finite_endpoint);
    }
    if (f.resolved) {
        L = vec3(0.0);
        T = vec3(1.0);
        if (span > 0.0 && !atmosphere_transport_ray(pixel, ndc, d, segment, span, depth, shadow_layer, cache, L, T))
            return false;
    } else {
        L = vec3(0.0);
        vec3 ground, sky;
        float ground_weight, sky_weight;
        limb_integrate(f, false, L, ground, ground_weight);
        limb_integrate(f, true, L, sky, sky_weight);
        float space = max(1.0-ground_weight-sky_weight, 0.0);
        T = surface_seen && ground_weight > 0.0 ? ground/ground_weight
          : sky_weight+space > 0.0 ? (sky+space)/(sky_weight+space) : ground/max(ground_weight, 1e-20);
        if (span > 0.0) atmosphere_shadow_loss(pixel, finite_endpoint ? depth : 0.0, shadow_layer, L);
        L *= atm.depth_params.w;
    }
    if (!surface_seen || !f.resolved) L += limb_star_image(f)*atm.depth_params.w;
    if (!surface_seen && limb_star_behind(d)) T = vec3(1.0-limb_coverage(f, f.edge));
    return true;
}
#endif

bool atmosphere_transport_endpoint(ivec2 pixel, vec2 ndc, vec3 d, vec2 segment, float span,
#ifdef ATMOSPHERE_LIMB
                                   bool has_limb, LimbFootprint limb,
#endif
                                   float depth, int shadow_layer, inout TransportCache cache, out vec3 L, out vec3 T) {
#ifdef ATMOSPHERE_LIMB
    if (has_limb) return limb_transport(pixel, limb, ndc, d, segment, span, depth, shadow_layer, cache, L, T);
#endif
    if (span <= 0.0) return false;
    return atmosphere_transport_ray(pixel, ndc, d, segment, span, depth, shadow_layer, cache, L, T);
}

// The 2D cloud sheet of this atmosphere's body at this pixel, as the cloud pass wrote it (premultiplied colour with
// exposure applied; alpha carries 2*slot+opacity), when it lies in front of the opaque scene.
bool cloud_layer_fetch(ivec2 pixel, float scene_depth, out vec4 color, out float depth) {
    color = vec4(0.0);
    depth = 0.0;
    if (atm.cloud_layer.y < 0.5) return false;
    depth = texelFetch(cloud_layer_depth, pixel, 0).r;
    if (depth <= scene_depth) return false;
    color = texelFetch(cloud_layer_color, pixel, 0);
    float slot = floor(color.a*0.5);
    if (slot != atm.cloud_layer.z) return false;
    color.a -= 2.0*slot;
    return true;
}

bool composite_atmosphere_pixel(ivec2 pixel, ivec2 size, float depth, out vec3 scattered, out vec3 transmitted) {
    scattered = vec3(0.0);
    transmitted = vec3(1.0);
    if (any(greaterThanEqual(pixel, size))) return false;
    vec4 cloud;
    float cloud_depth;
    bool clouded = cloud_layer_fetch(pixel, depth, cloud, cloud_depth);
    bool opaque_cloud = clouded && cloud.a >= 0.99999;
#ifdef ATMOSPHERE_RING_SLOT
    // The ring slot is transported like a transparent fragment: P -> T(d)P + aS(d). The last composite blends it
    // over this atmosphere's result unless list fragments need it sorted in the resolve; earlier ones store it.
    vec4 ring = vec4(0.0);
    float ring_depth = 0.0;
    bool ringed = false, blend_ring = false;
    if (secondary.ring.x != 0u) {
        uint slot = imageLoad(ring_slot_depth, pixel).x;
        ring_depth = uintBitsToFloat(slot & RING_SLOT_DEPTH_MASK);
        ringed = (slot & RING_SLOT_CONTENDED) == 0u && ring_depth > 0.0 && ring_depth >= depth;
        if (ringed) {
            ring = load_ring_slot(pixel);
            blend_ring = secondary.ring.x == 2u && imageLoad(transparency_heads, pixel).x == TRANSPARENCY_END;
        }
    }
#endif

    vec2 ndc = (vec2(pixel)+0.5)/vec2(size)*2.0-1.0;
    float refract_slope;
    vec3 d = view_direction(ndc, refract_slope);
    float span;
    vec2 segment = atmosphere_segment(atm.camera.xyz, d, span);
#ifdef ATMOSPHERE_LIMB
    LimbFootprint limb;
    bool has_limb = limb_footprint(ndc, d, refract_slope, size, limb);
#endif
    TransportCache cache = init_transport_cache();
    vec3 L, T;
    bool background = false;
    if (!opaque_cloud) {
        background = atmosphere_transport_endpoint(pixel, ndc, d, segment, span,
#ifdef ATMOSPHERE_LIMB
            has_limb, limb,
#endif
            depth, 0, cache, L, T);
        if (background) {
            scattered = L*atm.depth_params.x;
            transmitted = T;
        }
    }
#ifdef ATMOSPHERE_RING_SLOT
    bool ring_changed = false;
    if (ringed && span > 0.0) {
        float path;
        bool finite_endpoint;
        vec3 ring_L, ring_T;
        if (atmosphere_transport_ray_base(pixel, ndc, d, segment, span, ring_depth, 1, cache, ring_L, ring_T, path, finite_endpoint)) {
            atmosphere_finish_radiance(d, span, path, finite_endpoint, ring_depth, cache, ring_L);
            ring.rgb = ring.rgb*ring_T+ring.a*ring_L*atm.depth_params.x;
#ifdef ATMOSPHERE_LIMB
            if (has_limb && !limb.resolved && length(atm.camera.xyz+d*limb_depth_distance(ndc, ring_depth, d)) < top_radius()) {
                ring *= limb_coverage(limb, limb.limb);
                ring.rgb += ring.a*limb_star_image(limb)*atm.depth_params.w*atm.depth_params.x;
            }
#endif
            ring_changed = true;
        }
    }
    if (!background && !clouded && !blend_ring) {
        if (ring_changed) store_ring_slot(pixel, ring);
        return false;
    }
#else
    if (!background && !clouded) return false;
#endif
    if (clouded) {
        if (atmosphere_transport_endpoint(pixel, ndc, d, segment, span,
#ifdef ATMOSPHERE_LIMB
            has_limb, limb,
#endif
            cloud_depth, 2, cache, L, T))
            cloud.rgb = cloud.rgb*T+cloud.a*L*atm.depth_params.x;
        scattered = scattered*(1.0-cloud.a)+cloud.rgb;
        transmitted *= 1.0-cloud.a;
#ifdef ATMOSPHERE_RING_SLOT
        // A ring behind the cloud becomes (1-c)P + aC, which blended over the clouded background hides it behind the cloud.
        if (ringed && ring_depth < cloud_depth) {
            ring.rgb = ring.rgb*(1.0-cloud.a)+ring.a*cloud.rgb;
            ring_changed = true;
        }
#endif
    }
#ifdef ATMOSPHERE_RING_SLOT
    if (blend_ring) {
        scattered = ring.rgb+(1.0-ring.a)*scattered;
        transmitted *= 1.0-ring.a;
    } else if (ring_changed) {
        store_ring_slot(pixel, ring);
    }
#endif
    return true;
}

#ifdef ATMOSPHERE_TRANSPARENCY
// A lean pass reads shadow deficits only from the shadow pass's endpoint cache and marks fragments that miss it (and
// fragments behind a cloud whose own endpoint misses) for the deferred full pass; returns true when it marked any.
bool transport_transparent_fragments(ivec2 transparent_pixel) {
    vec3 L, T;
    bool deferred_any = false;
    ivec2 transparent_size = imageSize(transparency_heads);
    if (any(greaterThanEqual(transparent_pixel, transparent_size))) return false;
    uint current = imageLoad(transparency_heads, transparent_pixel).x;
    if (current != TRANSPARENCY_END) {
        vec2 transparent_ndc = (vec2(transparent_pixel)+0.5)/vec2(transparent_size)*2.0-1.0;
        float transparent_slope;
        vec3 transparent_dir = view_direction(transparent_ndc, transparent_slope);
        float transparent_span;
        vec2 transparent_segment = atmosphere_segment(atm.camera.xyz, transparent_dir, transparent_span);
#ifdef ATMOSPHERE_LIMB
        LimbFootprint transparent_limb;
        bool thin_limb = limb_footprint(transparent_ndc, transparent_dir, transparent_slope, transparent_size, transparent_limb) && !transparent_limb.resolved;
        float limb_cover = thin_limb ? limb_coverage(transparent_limb, transparent_limb.limb) : 1.0;
        vec3 limb_light = thin_limb ? limb_star_image(transparent_limb)*atm.depth_params.w*atm.depth_params.x : vec3(0.0);
#endif
        // Transparent fragments resolve over the background after the cloud sheet is already blended into it. One
        // behind the cloud (premultiplied F, alpha f) is rewritten to (1-a)F + f*C, so blending it over the clouded
        // background gives C + (1-a)(F + (1-f)X): the fragment hidden behind the cloud as if sorted with it.
        vec4 cloud;
        float cloud_depth;
        bool clouded = cloud_layer_fetch(transparent_pixel, texelFetch(scene_depth, transparent_pixel, 0).r, cloud, cloud_depth);
        bool cloud_transported = false, cloud_missed = false;
        bool single = (current & TRANSPARENCY_MULTI_BIT) == 0u;
        current &= TRANSPARENCY_INDEX_MASK;
        // A contended ring slot (the loser of its claim is in this list) is transported last, as one more fragment.
        uint slot = ring_slot_bits(transparent_pixel);
        bool slot_pending = (slot & RING_SLOT_CONTENDED) != 0u;
        TransportCache cache = init_transport_cache();
        if (transparent_span > 0.0) {
            while (current != TRANSPARENCY_END || slot_pending) {
                bool is_slot = current == TRANSPARENCY_END;
                uint frag_depth_bits, frag_next;
                if (is_slot) {
                    slot_pending = false;
                    frag_depth_bits = slot & RING_SLOT_DEPTH_MASK;
                    if ((slot & RING_SLOT_DEFERRED) != 0u) frag_depth_bits |= 0x80000000u;
                    frag_next = TRANSPARENCY_END;
                } else {
                    frag_depth_bits = transparent_depth_bits(current);
                    frag_next = single ? TRANSPARENCY_END : transparent_next(current);
                }
#ifdef ATMOSPHERE_TILE_DEFERRED
                if ((frag_depth_bits & TRANSPARENT_DEFERRED_BIT) == 0u) {
                    current = frag_next;
                    continue;
                }
                frag_depth_bits &= ~TRANSPARENT_DEFERRED_BIT;
                if (is_slot) imageStore(ring_slot_depth, transparent_pixel, uvec4(slot & ~RING_SLOT_DEFERRED));
                else set_transparent_depth_bits(current, frag_depth_bits);
#endif
                float fragment_depth = uintBitsToFloat(frag_depth_bits);
                bool need_cloud = clouded && fragment_depth < cloud_depth && !cloud_transported;
                bool fragment_valid = false, fragment_missed = false;
                for (int pass = need_cloud ? 0 : 1; pass < 2; ++pass) {
                    float endpoint_depth = pass == 0 ? cloud_depth : fragment_depth;
                    float path;
                    bool finite_endpoint;
                    shadow_cache_missed = false;
                    vec3 endpoint_L, endpoint_T;
                    bool valid = atmosphere_transport_ray_base(transparent_pixel, transparent_ndc, transparent_dir,
                        transparent_segment, transparent_span, endpoint_depth, pass == 0 ? 2 : 1, cache,
                        endpoint_L, endpoint_T, path, finite_endpoint);
                    if (valid) {
#ifndef ATMOSPHERE_SHADOW_DATA_ONLY
                        if (shadow_cache_missed && any(greaterThan(endpoint_L, vec3(0.0)))) {
                            vec3 unused;
                            vec3 fraction = atmosphere_shadow_deficit(transparent_dir, transparent_segment, path, transparent_pixel, 0u, 0.0, unused);
                            endpoint_L = max(endpoint_L - endpoint_L * fraction, vec3(0.0));
                        }
#endif
                        atmosphere_finish_radiance(transparent_dir, transparent_span, path, finite_endpoint, endpoint_depth, cache, endpoint_L);
                    }
                    if (pass == 0) {
                        if (valid) {
                            cloud.rgb = cloud.rgb*endpoint_T+cloud.a*endpoint_L*atm.depth_params.x;
#ifdef ATMOSPHERE_LIMB
                            if (thin_limb && length(atm.camera.xyz+transparent_dir*limb_depth_distance(transparent_ndc, cloud_depth, transparent_dir)) < top_radius()) {
                                cloud *= limb_cover;
                                cloud.rgb += cloud.a*limb_light;
                            }
#endif
                        }
                        cloud_missed = shadow_cache_missed;
                        cloud_transported = true;
                    } else {
                        fragment_valid = valid;
                        fragment_missed = shadow_cache_missed;
                        L = endpoint_L;
                        T = endpoint_T;
                    }
                }
                if (fragment_valid) {
                    bool missed = fragment_missed;
                    uvec2 frag_packed = is_slot ? uvec2(0u) : transparent_color_packed(current);
                    vec4 color = is_slot ? load_ring_slot(transparent_pixel) : transparent_color(frag_packed);
                    color.rgb = color.rgb*T+color.a*L*atm.depth_params.x;
#ifdef ATMOSPHERE_LIMB
                    if (thin_limb && length(atm.camera.xyz+transparent_dir*limb_depth_distance(transparent_ndc, fragment_depth, transparent_dir)) < top_radius()) {
                        color *= limb_cover;
                        color.rgb += color.a*limb_light;
                    }
#endif
                    if (clouded && fragment_depth < cloud_depth) {
                        missed = missed || cloud_missed;
                        color.rgb = color.rgb*(1.0-cloud.a)+color.a*cloud.rgb;
                    }
#ifdef ATMOSPHERE_TILE_LEAN
                    if (missed) {
                        if (is_slot) imageStore(ring_slot_depth, transparent_pixel, uvec4(slot | RING_SLOT_DEFERRED));
                        else set_transparent_depth_bits(current, frag_depth_bits | TRANSPARENT_DEFERRED_BIT);
                        deferred_any = true;
                    } else
#endif
                    if (is_slot) store_ring_slot(transparent_pixel, color);
                    else set_transparent_color_packed(current, pack_transparent_color(color, transparent_excludes_bloom(frag_packed)));
                }
                current = frag_next;
            }
        }
    }
    return deferred_any;
}
#endif

void main() {
#if defined(ATMOSPHERE_TILE_DEFERRED)
    transport_transparent_fragments(atmosphere_tile_pixel());
#elif defined(ATMOSPHERE_TILE_LEAN)
    atmosphere_tile_begin();
    atmosphere_tile_end(transport_transparent_fragments(ivec2(gl_GlobalInvocationID.xy)));
#elif defined(ATMOSPHERE_TRANSPARENCY)
    if (any(greaterThanEqual(ivec2(gl_GlobalInvocationID.xy), imageSize(transparency_heads)))) return;
    transport_transparent_fragments(ivec2(gl_GlobalInvocationID.xy));
#endif
#ifdef ATMOSPHERE_FRAGMENT
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(scene_depth, 0);
    float depth = texelFetch(scene_depth, pixel, 0).r;
    vec3 scattered, transmitted;
    if (!composite_atmosphere_pixel(pixel, size, depth, scattered, transmitted)) discard;
#ifdef HDR_RGBA32F
    out_scattered = vec4(scattered, 0.0);
#else
    out_scattered = vec4(min(scattered, vec3(65000.0)), 0.0);
#endif
    out_transmittance = vec4(transmitted, 1.0);
#elif !defined(ATMOSPHERE_TRANSPARENT_ONLY)
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(hdr_color);
    if (any(greaterThanEqual(pixel, size))) return;
    float depth = texelFetch(scene_depth, pixel, 0).r;
    vec3 scattered, transmitted;
    if (!composite_atmosphere_pixel(pixel, size, depth, scattered, transmitted)) return;
    vec4 color = imageLoad(hdr_color, pixel);
#ifdef HDR_RGBA32F
    imageStore(hdr_color, pixel, vec4(color.rgb*transmitted+scattered, color.a));
#else
    imageStore(hdr_color, pixel, vec4(min(color.rgb*transmitted+scattered, vec3(65000.0)), color.a));
#endif
#endif
}
