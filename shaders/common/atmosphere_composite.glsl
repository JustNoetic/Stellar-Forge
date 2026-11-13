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
#if !defined(ATMOSPHERE_TRANSPARENCY) || defined(ATMOSPHERE_UNSHADOWED) || defined(ATMOSPHERE_TILE_LEAN)
#define ATMOSPHERE_SHADOW_DATA_ONLY
#endif
#include "atmosphere_shadow.glsl"
layout(set = 0, binding = 16) uniform sampler2DArray shadow_deficit_lut;
layout(set = 0, binding = 17) uniform sampler2DArray shadow_endpoint_lut;
layout(set = 0, binding = 33) uniform sampler2D cloud_layer_color;
layout(set = 0, binding = 34) uniform sampler2D cloud_layer_depth;
#ifdef ATMOSPHERE_TRANSPARENCY
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
layout(push_constant) uniform SecondaryLighting { vec4 irradiance; } secondary;
#endif

vec4 cubic_weights(float f) {
    float g = 1.0-f;
    return vec4(g*g*g, 3.0*f*f*f-6.0*f*f+4.0, -3.0*f*f*f+3.0*f*f+3.0*f+1.0, f*f*f)/6.0;
}

struct AerialFilter { vec2 x, y, sx, sy; float z; };

AerialFilter aerial_filter(vec3 direction, float range, float path, ivec3 size, sampler2D ranges) {
    vec2 uv = sky_uv(direction, size.xy);
    float half_texel = 0.5/float(size.y);
    vec2 y_bounds = uv.y < 0.5 ? vec2(half_texel, 0.5-half_texel) : vec2(0.5+half_texel, 1.0-half_texel);
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

void aerial_transport(vec3 direction, float span, float path, out vec3 L, out vec3 T) {
    AerialFilter f = aerial_filter(direction, span, path, AERIAL_SIZE, aerial_range_lut);
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
vec3 secondary_radiance(vec3 direction, float span, float path, bool finite_endpoint) {
    if (!finite_endpoint)
        return textureLod(secondary_sky_lut, sky_uv(direction, textureSize(secondary_sky_lut, 0)), 0.0).rgb;
    AerialFilter f = aerial_filter(direction, span, path, textureSize(secondary_aerial_lut, 0), secondary_range_lut);
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

void atmosphere_shadow_loss(ivec2 pixel, float depth, inout vec3 L) {
#ifndef ATMOSPHERE_UNSHADOWED
    if (shadow_flags.x != 0u) {
        if (any(lessThan(pixel, shadow_bounds.xy)) || any(greaterThanEqual(pixel, shadow_bounds.zw))) return;
        if (any(greaterThan(L, vec3(0.0)))) {
            vec3 fraction = vec3(0.0);
            bool cached = false;
            for (int layer = 0; layer < 3; ++layer) {
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
                                   out vec3 L, out vec3 T, out float path, out bool finite_endpoint) {
    path = span;
    finite_endpoint = false;
    if (depth > 0.0) {
        vec4 surface = atm.inverse_view_projection*vec4(ndc, depth, 1.0);
        float distance = length(world_to_atmosphere(surface.xyz/surface.w))*149597870.7;
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
        aerial_transport(d, span, path, L, T);
    } else {
        vec2 uv = sky_uv(d);
        L = textureLod(radiance_lut, uv, 0.0).rgb;
        T = clamp(textureLod(throughput_lut, uv, 0.0).rgb, 0.0, 1.0);
    }
    atmosphere_shadow_loss(pixel, depth, L);
    return true;
}

void atmosphere_finish_radiance(vec3 d, float span, float path, bool finite_endpoint, float depth, inout vec3 L) {
#ifdef ATMOSPHERE_SECONDARY
    // Matches the base early return: surface pixels without aerial perspective carry no in-scatter.
    if (depth <= 0.0 || atm.depth_params.y != 0.0)
        L += secondary.irradiance.rgb*secondary_radiance(d, span, path, finite_endpoint);
#endif
    L *= atm.depth_params.w;
}

bool atmosphere_transport_ray(ivec2 pixel, vec2 ndc, vec3 d, vec2 segment, float span, float depth, out vec3 L, out vec3 T) {
    float path;
    bool finite_endpoint;
    if (!atmosphere_transport_ray_base(pixel, ndc, d, segment, span, depth, L, T, path, finite_endpoint)) return false;
    atmosphere_finish_radiance(d, span, path, finite_endpoint, depth, L);
    return true;
}

#ifdef ATMOSPHERE_LIMB
const float LIMB_RESOLVED_PIXELS = 32.0;
const int LIMB_MAX_TAPS = 16;
const int LIMB_BEND_ROWS = 64;

struct LimbFootprint { vec2 across; float center, wide, narrow, limb, edge, azimuth; bool resolved; };

bool limb_footprint(vec2 ndc, vec3 d, ivec2 size, out LimbFootprint f) {
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
    vec2 steps = abs(vec2(dot(radial, row_x), dot(radial, row_y)))*2.0/(vec2(size)*length(row_x*ndc.x+row_y*ndc.y+row_z));
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

vec3 limb_star_image(LimbFootprint f) {
    if (atm.limb_image.x < 0.5) return vec3(0.0);
    float reach = 0.5*(f.wide+f.narrow);
    float first = limb_row(f, true, f.center-reach), last = limb_row(f, true, f.center+reach);
    if (last <= first) return vec3(0.0);
    float rows = float(LIMB_BEND_ROWS-1);
    int begin = int(first*rows), end = min(int(ceil(last*rows)), LIMB_BEND_ROWS-1);
    float lowest = limb_row_angle(f, true, first), highest = limb_row_angle(f, true, last);
    vec3 high_bend = limb_bend_at(last);
    float exit_low = lowest-atm.limb_image.y;
    float exit_high = highest-min(high_bend.r, min(high_bend.g, high_bend.b));
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
        for (int row = begin; row < end; ++row) {
            if (atm.limb_bend[row].w < 0.5 || atm.limb_bend[row+1].w < 0.5) continue;
            float t0 = max(float(row)/rows, first), t1 = min(float(row+1)/rows, last);
            if (t1 <= t0) continue;
            float a0 = limb_row_angle(f, true, t0), a1 = limb_row_angle(f, true, t1);
            vec3 start = a0-limb_bend_at(t0), slope = max(a1-limb_bend_at(t1)-start, vec3(1e-12));
            vec3 enter = clamp((center-chord-start)/slope, 0.0, 1.0), leave = clamp((center+chord-start)/slope, 0.0, 1.0);
            if (all(lessThanEqual(leave, enter))) continue;
            for (int channel = 0; channel < 3; ++channel) {
                if (leave[channel] <= enter[channel]) continue;
                float share = limb_coverage(f, mix(a0, a1, leave[channel]))-limb_coverage(f, mix(a0, a1, enter[channel]));
                float middle = 0.5*(enter[channel]+leave[channel]);
                float offset = (start[channel]+slope[channel]*middle-center)*to_true;
                float rim = 1.0-sqrt(max(1.0-(side*side+offset*offset)/(sun.w*sun.w), 0.0));
                vec2 uv = sky_grid_uv(f.azimuth, mix(t0, t1, middle), true, SKY_SIZE);
                float transmitted = clamp(textureLod(throughput_lut, uv, 0.0)[channel], 0.0, 1.0);
                image[channel] += share*transmitted*radiance[channel]*(1.0-0.4*rim-0.2*rim*rim);
            }
        }
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

float limb_depth_distance(vec2 ndc, float depth) {
    vec4 surface = atm.inverse_view_projection*vec4(ndc, depth, 1.0);
    return length(world_to_atmosphere(surface.xyz/surface.w))*149597870.7;
}

bool limb_transport(ivec2 pixel, LimbFootprint f, vec2 ndc, vec3 d, vec2 segment, float span, float depth, out vec3 L, out vec3 T) {
    float path = span;
    bool surface_seen = false;
    if (depth > 0.0) {
        float distance = limb_depth_distance(ndc, depth);
        float closest = -dot(atm.camera.xyz, d);
        if (distance < closest-top_radius()) return false;
        surface_seen = distance < closest+top_radius();
        bool finite_endpoint;
        if (span > 0.0) path = depth_endpoint_path(distance, d, segment, span, finite_endpoint);
    }
    if (f.resolved) {
        L = vec3(0.0);
        T = vec3(1.0);
        if (span > 0.0 && !atmosphere_transport_ray(pixel, ndc, d, segment, span, depth, L, T)) return false;
    } else {
        L = vec3(0.0);
        vec3 ground, sky;
        float ground_weight, sky_weight;
        limb_integrate(f, false, L, ground, ground_weight);
        limb_integrate(f, true, L, sky, sky_weight);
        float space = max(1.0-ground_weight-sky_weight, 0.0);
        T = surface_seen && ground_weight > 0.0 ? ground/ground_weight
          : sky_weight+space > 0.0 ? (sky+space)/(sky_weight+space) : ground/max(ground_weight, 1e-20);
        if (span > 0.0) atmosphere_shadow_loss(pixel, depth, L);
        L *= atm.depth_params.w;
    }
    if (!surface_seen || !f.resolved) L += limb_star_image(f)*atm.depth_params.w;
    if (!surface_seen && limb_star_behind(d)) T = vec3(1.0-limb_coverage(f, f.edge));
    return true;
}
#endif

bool atmosphere_transport(ivec2 pixel, ivec2 size, float depth, out vec3 L, out vec3 T) {
    if (any(greaterThanEqual(pixel, size))) return false;
    vec2 ndc = (vec2(pixel)+0.5)/vec2(size)*2.0-1.0;
    vec3 d = view_direction(ndc);
    float span;
    vec2 segment = atmosphere_segment(atm.camera.xyz, d, span);
#ifdef ATMOSPHERE_LIMB
    LimbFootprint limb;
    if (limb_footprint(ndc, d, size, limb)) return limb_transport(pixel, limb, ndc, d, segment, span, depth, L, T);
#endif
    if (span <= 0.0) return false;
    return atmosphere_transport_ray(pixel, ndc, d, segment, span, depth, L, T);
}

// The 2D cloud sheet of this atmosphere's body at this pixel, as the cloud pass wrote it (premultiplied colour with
// exposure applied; alpha carries 2*body+opacity), when it lies in front of the opaque scene.
bool cloud_layer_fetch(ivec2 pixel, float scene_depth, out vec4 color, out float depth) {
    color = vec4(0.0);
    depth = 0.0;
    if (atm.cloud_layer.y < 0.5) return false;
    depth = texelFetch(cloud_layer_depth, pixel, 0).r;
    if (depth <= scene_depth) return false;
    color = texelFetch(cloud_layer_color, pixel, 0);
    float body = floor(color.a*0.5);
    if (body != atm.cloud_layer.x) return false;
    color.a -= 2.0*body;
    return true;
}

// The cloud's premultiplied colour as seen from the camera: it receives the atmosphere in front of it.
vec4 cloud_layer_transported(ivec2 pixel, ivec2 size, vec4 cloud, float cloud_depth) {
    vec3 L, T;
    if (atmosphere_transport(pixel, size, cloud_depth, L, T)) cloud.rgb = cloud.rgb*T+cloud.a*L*atm.depth_params.x;
    return cloud;
}

// Places the cloud sheet between the camera and the background transport, hiding (1-alpha) of the background and of
// the air behind it.
void apply_cloud_layer(ivec2 pixel, ivec2 size, vec4 cloud, float cloud_depth, inout vec3 scattered, inout vec3 transmitted) {
    cloud = cloud_layer_transported(pixel, size, cloud, cloud_depth);
    scattered = scattered*(1.0-cloud.a)+cloud.rgb;
    transmitted *= 1.0-cloud.a;
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
        vec3 transparent_dir = view_direction(transparent_ndc);
        float transparent_span;
        vec2 transparent_segment = atmosphere_segment(atm.camera.xyz, transparent_dir, transparent_span);
#ifdef ATMOSPHERE_LIMB
        LimbFootprint transparent_limb;
        bool thin_limb = limb_footprint(transparent_ndc, transparent_dir, transparent_size, transparent_limb) && !transparent_limb.resolved;
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
        if (transparent_span > 0.0) {
            while (current != TRANSPARENCY_END) {
                uvec4 fragment = fragments[current];
#ifdef ATMOSPHERE_TILE_DEFERRED
                if ((fragment.z & TRANSPARENT_DEFERRED_BIT) == 0u) {
                    current = fragment.w;
                    continue;
                }
                fragment.z &= ~TRANSPARENT_DEFERRED_BIT;
                fragments[current].z = fragment.z;
#endif
                float fragment_depth = uintBitsToFloat(fragment.z);
                bool need_cloud = clouded && fragment_depth < cloud_depth && !cloud_transported;
                bool fragment_valid = false, fragment_missed = false;
                for (int pass = need_cloud ? 0 : 1; pass < 2; ++pass) {
                    float endpoint_depth = pass == 0 ? cloud_depth : fragment_depth;
                    float path;
                    bool finite_endpoint;
                    shadow_cache_missed = false;
                    vec3 endpoint_L, endpoint_T;
                    bool valid = atmosphere_transport_ray_base(transparent_pixel, transparent_ndc, transparent_dir,
                        transparent_segment, transparent_span, endpoint_depth, endpoint_L, endpoint_T, path, finite_endpoint);
                    if (valid) {
#ifndef ATMOSPHERE_SHADOW_DATA_ONLY
                        if (shadow_cache_missed && any(greaterThan(endpoint_L, vec3(0.0)))) {
                            vec3 unused;
                            vec3 fraction = atmosphere_shadow_deficit(transparent_dir, transparent_segment, path, transparent_pixel, 0u, 0.0, unused);
                            endpoint_L = max(endpoint_L - endpoint_L * fraction, vec3(0.0));
                        }
#endif
                        atmosphere_finish_radiance(transparent_dir, transparent_span, path, finite_endpoint, endpoint_depth, endpoint_L);
                    }
                    if (pass == 0) {
                        if (valid) {
                            cloud.rgb = cloud.rgb*endpoint_T+cloud.a*endpoint_L*atm.depth_params.x;
#ifdef ATMOSPHERE_LIMB
                            if (thin_limb && length(atm.camera.xyz+transparent_dir*limb_depth_distance(transparent_ndc, cloud_depth)) < top_radius()) {
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
                    vec4 color = transparent_color(fragment);
                    color.rgb = color.rgb*T+color.a*L*atm.depth_params.x;
#ifdef ATMOSPHERE_LIMB
                    if (thin_limb && length(atm.camera.xyz+transparent_dir*limb_depth_distance(transparent_ndc, fragment_depth)) < top_radius()) {
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
                        fragments[current].z = fragment.z | TRANSPARENT_DEFERRED_BIT;
                        deferred_any = true;
                    } else
#endif
                    fragments[current].xy = pack_transparent_color(color, transparent_excludes_bloom(fragment));
                }
                current = fragment.w;
            }
        }
    }
    return deferred_any;
}
#endif

void main() {
    vec3 L, T;
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
    bool background = atmosphere_transport(pixel, size, depth, L, T);
    vec4 cloud;
    float cloud_depth;
    bool clouded = cloud_layer_fetch(pixel, depth, cloud, cloud_depth);
    if (!background && !clouded) discard;
    vec3 scattered = background ? L*atm.depth_params.x : vec3(0.0), transmitted = background ? T : vec3(1.0);
    if (clouded) apply_cloud_layer(pixel, size, cloud, cloud_depth, scattered, transmitted);
    out_scattered = vec4(min(scattered, vec3(65000.0)), 0.0);
    out_transmittance = vec4(transmitted, 1.0);
#elif !defined(ATMOSPHERE_TRANSPARENT_ONLY)
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(hdr_color);
    if (any(greaterThanEqual(pixel, size))) return;
    float depth = texelFetch(scene_depth, pixel, 0).r;
    bool background = atmosphere_transport(pixel, size, depth, L, T);
    vec4 cloud;
    float cloud_depth;
    bool clouded = cloud_layer_fetch(pixel, depth, cloud, cloud_depth);
    if (!background && !clouded) return;
    vec3 scattered = background ? L*atm.depth_params.x : vec3(0.0), transmitted = background ? T : vec3(1.0);
    if (clouded) apply_cloud_layer(pixel, size, cloud, cloud_depth, scattered, transmitted);
    vec4 color = imageLoad(hdr_color, pixel);
    imageStore(hdr_color, pixel, vec4(min(color.rgb*transmitted+scattered, vec3(65000.0)), color.a));
#endif
}
