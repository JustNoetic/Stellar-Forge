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
layout(set = 0, binding = 9) uniform sampler3D bruneton_rayleigh_lut;
layout(set = 0, binding = 10) uniform sampler3D bruneton_mie_lut;
#ifdef ATMOSPHERE_TRANSPARENCY
#include "common/transparency.glsl"
#endif

float shell_length(vec3 d, vec2 segment, float end, float radius) {
    vec2 roots = sphere_roots(atm.camera.xyz, d, radius);
    return max(0.0, min(end, roots.y)-max(segment.x, roots.x));
}

bool aerial_column(ivec2 pixel, float path, float radius, bool far_side, float alignment,
                   float full_shell, float shell_radius, float slant_weight, out vec3 L, out vec3 tau) {
    L = vec3(0.0); tau = vec3(0.0);
    vec3 d = view_direction(vec2(pixel)/vec2(AERIAL_SIZE.xy-1)*2.0-1.0);
    vec2 segment = atmosphere_segment(atm.camera.xyz, d);
    vec2 roots = sphere_roots(atm.camera.xyz, d, radius);
    bool supported = alignment <= 0.0 || (roots.y >= 0.0 && segment.y > segment.x);
    if (!supported && alignment >= 1.0) return false;
    float column_path = path;
    if (roots.y >= 0.0 && segment.y > segment.x)
        column_path = mix(path, max(0.0, (far_side ? roots.y : roots.x)-segment.x), alignment);
    float focus = aerial_focus(d, segment);
    float z = aerial_coordinate(column_path, focus);
    float slice = min(floor(z*float(AERIAL_SIZE.z-1)), float(AERIAL_SIZE.z-2));
    float t0 = min(aerial_distance(slice/float(AERIAL_SIZE.z-1), focus), segment.y-segment.x);
    float t1 = min(aerial_distance((slice+1.0)/float(AERIAL_SIZE.z-1), focus), segment.y-segment.x);
    float fraction = clamp((column_path-t0)/max(t1-t0, 1e-8), 0.0, 1.0);
    z = (slice+fraction)/float(AERIAL_SIZE.z-1);
    vec3 uv = vec3((vec2(pixel)+0.5)/vec2(AERIAL_SIZE.xy), (0.5+z*float(AERIAL_SIZE.z-1))/float(AERIAL_SIZE.z));
    L = textureLod(aerial_radiance_lut, uv, 0.0).rgb;
    tau = textureLod(aerial_optical_depth_lut, uv, 0.0).rgb;
    if (slant_weight > 0.0) {
        float column_shell = shell_length(d, segment, segment.x+min(column_path, segment.y-segment.x), shell_radius);
        float ratio = mix(1.0, clamp(full_shell/max(column_shell, 1e-5), 0.125, 8.0), slant_weight);
        vec3 corrected = tau*ratio;
        L *= mix((1.0-exp(-corrected))/max(1.0-exp(-tau), vec3(1e-8)), vec3(ratio), lessThan(tau, vec3(1e-4)));
        tau = corrected;
    }
    return supported;
}

void aerial_transport(vec2 screen_uv, vec3 direction, vec2 full_segment, float path, inout vec3 L, inout vec3 T) {
    vec2 grid = screen_uv*vec2(AERIAL_SIZE.xy-1);
    ivec2 base = min(ivec2(grid), AERIAL_SIZE.xy-2);
    vec2 blend = grid-vec2(base);
    vec3 endpoint = atm.camera.xyz+direction*(full_segment.x+path);
    float radius = length(endpoint);
    bool far_side = dot(endpoint, direction) > 0.0;
    float alignment = smoothstep(0.25, 1.0, (atm.camera.z-atm.reference_shape.x)/(top_radius()-atm.reference_shape.x));
    alignment *= smoothstep(0.0, 1.0, path);
    float h = radius-atm.reference_shape.x;
    float r_weight = dot(atm.rayleigh.rgb, vec3(1.0))*atm.radii_heights.z*exp(clamp(-h/atm.radii_heights.z, -80.0, 40.0));
    float m_weight = dot(atm.mie_extinction.rgb, vec3(1.0))*(atm.radii_heights.w-min(h, 0.0))*exp(-max(h, 0.0)/atm.radii_heights.w);
    float m_height = m_weight*atm.radii_heights.w;
    for (int i = 0; i < int(atm.aerosol_count.x); ++i) {
        vec4 shape = atm.aerosol_shape[i];
        float scale = shape.z > 0.0 ? shape.z : shape.x;
        float weight = dot(atm.aerosol_extinction[i].rgb, vec3(1.0))*scale*particle_density(i, h);
        m_weight += weight;
        m_height += weight*scale;
    }
    float effective_height = (r_weight*atm.radii_heights.z+m_height)/max(r_weight+m_weight, 1e-20);
    float shell_radius = min(top_radius(), radius+4.0/ATM_PI*effective_height);
    float full_shell = shell_length(direction, full_segment, full_segment.x+path, shell_radius);
    float slant_weight = far_side ? 0.0 : alignment*(1.0-smoothstep(0.1, 0.35, abs(dot(endpoint/radius, direction))));
    vec3 columns_L[4], columns_tau[4];
    bool valid[4];
    int nearest = -1;
    float nearest_distance = 1e20;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int index = y*2+x;
            ivec2 pixel = base+ivec2(x, y);
            valid[index] = aerial_column(pixel, path, radius, far_side, alignment, full_shell, shell_radius, slant_weight, columns_L[index], columns_tau[index]);
            float distance = dot(vec2(pixel)-grid, vec2(pixel)-grid);
            if (valid[index] && distance < nearest_distance) { nearest = index; nearest_distance = distance; }
        }
    }
    vec3 padded_L, padded_tau;
    bool supported = nearest >= 0;
    if (supported) { padded_L = columns_L[nearest]; padded_tau = columns_tau[nearest]; }
    else {
        for (int y = -1; y <= 2; ++y) {
            for (int x = -1; x <= 2; ++x) {
                ivec2 pixel = clamp(base+ivec2(x, y), ivec2(0), AERIAL_SIZE.xy-1);
                float distance = dot(vec2(pixel)-grid, vec2(pixel)-grid);
                vec3 candidate_L, candidate_tau;
                if (distance < nearest_distance && aerial_column(pixel, path, radius, far_side, alignment, full_shell, shell_radius, slant_weight, candidate_L, candidate_tau)) {
                    supported = true; nearest_distance = distance;
                    padded_L = candidate_L; padded_tau = candidate_tau;
                }
            }
        }
    }
    if (!supported) {
        float fraction = clamp(path/max(full_segment.y-full_segment.x, 1e-8), 0.0, 1.0);
        L *= fraction;
        T = pow(max(T, vec3(1e-20)), vec3(fraction));
        return;
    }
    vec3 tau = vec3(0.0);
    L = vec3(0.0);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int index = y*2+x;
            float weight = (x == 0 ? 1.0-blend.x : blend.x)*(y == 0 ? 1.0-blend.y : blend.y);
            L += (valid[index] ? columns_L[index] : mix(columns_L[index], padded_L, alignment))*weight;
            tau += (valid[index] ? columns_tau[index] : mix(columns_tau[index], padded_tau, alignment))*weight;
        }
    }
    T = exp(-max(tau, vec3(0.0)));
}

void bruneton_aerial_transport(vec3 direction, vec2 segment, float path, inout vec3 L, inout vec3 T) {
    vec3 x = atm.camera.xyz + direction * segment.x;
    vec3 p = atm.camera.xyz + direction * (segment.x + path);
    float rx = clamp(length(x), bottom_radius(), top_radius());
    float rp = clamp(length(p), bottom_radius(), top_radius());
    float mux = clamp(dot(x, direction) / rx, -1.0, 1.0);
    float mup = clamp(dot(p, direction) / rp, -1.0, 1.0);

    bool hits_ground = (mux < 0.0 && rx * rx * (mux * mux - 1.0) + bottom_radius() * bottom_radius() >= 0.0);

    vec3 T_lut = bruneton_transmittance(transmittance_lut, rx, mux, rp, mup, hits_ground);

    vec3 r_mid, m_mid, e_mid;
    medium((x + p) * 0.5, r_mid, m_mid, e_mid);
    vec3 tau_direct = e_mid * path;
    vec3 T_direct = exp(-tau_direct);

    float blend = smoothstep(0.5, 2.0, path);
    T = mix(T_direct, T_lut, blend);

    vec3 L_direct = vec3(0.0);
    vec3 q = (x + p) * 0.5;
    for (uint s = 0u; s < uint(atm.camera.w); ++s) {
        vec3 sun = atm.sun_direction[s].xyz;
        float mu = dot(direction, sun);
        vec3 direct = sun_transmittance(transmittance_lut, q, sun) * (r_mid * rayleigh_phase(mu) + aerosol_source(q, mu));
        vec3 multiple = textureLod(multiscatter_lut, multiscatter_uv(q, sun), 0.0).rgb * (r_mid + m_mid);
        L_direct += atm.sun_irradiance[s].rgb * (direct + multiple);
    }
    L_direct *= segment_integral(e_mid, path);

    vec3 L_scatter = vec3(0.0);
    for (uint s = 0u; s < uint(atm.camera.w); ++s) {
        vec3 sun = atm.sun_direction[s].xyz;
        float mu_s_x = clamp(dot(x, sun) / rx, -1.0, 1.0);
        float mu_s_p = clamp(dot(p, sun) / rp, -1.0, 1.0);
        float nu = clamp(dot(direction, sun), -1.0, 1.0);

        vec3 L_R_x = sample_bruneton_scattering(bruneton_rayleigh_lut, rx, mux, mu_s_x, nu, hits_ground);
        vec3 L_M_x = sample_bruneton_scattering(bruneton_mie_lut, rx, mux, mu_s_x, nu, hits_ground);
        vec3 L_R_p = sample_bruneton_scattering(bruneton_rayleigh_lut, rp, mup, mu_s_p, nu, hits_ground);
        vec3 L_M_p = sample_bruneton_scattering(bruneton_mie_lut, rp, mup, mu_s_p, nu, hits_ground);

        vec3 L_R = max(L_R_x - T * L_R_p, vec3(0.0));
        vec3 L_M = max(L_M_x - T * L_M_p, vec3(0.0));

        L_scatter += atm.sun_irradiance[s].rgb * (L_R + aerosol_reference_phase(nu) * L_M);
    }

    L = mix(L_direct, L_scatter, blend);
}

bool atmosphere_transport(ivec2 pixel, ivec2 size, float depth, out vec3 L, out vec3 T) {
    if (any(greaterThanEqual(pixel, size))) return false;
    if (depth > 0.0 && atm.depth_params.y == 0.0) return false;
    vec2 ndc = (vec2(pixel)+0.5)/vec2(size)*2.0-1.0;
    vec3 d = view_direction(ndc);
    vec2 segment = atmosphere_segment(atm.camera.xyz, d);
    if (segment.y <= segment.x) return false;
    vec2 uv = sky_uv(d);
    L = textureLod(radiance_lut, uv, 0.0).rgb;
    T = clamp(textureLod(throughput_lut, uv, 0.0).rgb, 0.0, 1.0);
    if (depth > 0.0) {
        vec4 surface = atm.inverse_view_projection*vec4(ndc, depth, 1.0);
        float distance = length(world_to_atmosphere(surface.xyz/surface.w))*149597870.7;
        float path = min(distance, segment.y)-segment.x;
        if (path <= 0.0) return false;
        if (atm.depth_params.z > 0.5) {
            bruneton_aerial_transport(d, segment, path, L, T);
        } else {
            aerial_transport(ndc*0.5+0.5, d, segment, path, L, T);
        }
    }
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
