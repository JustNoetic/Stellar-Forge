#version 460
#define RING_REFRACTION_TABLE

#ifdef TRANSPARENCY_CAPTURE
layout(early_fragment_tests) in;
#endif

layout(location = 0) in vec3 f_local_pos;
layout(location = 1) in vec3 f_normal;
layout(location = 2) flat in uint f_host_idx;

layout(location = 0) out vec4 out_color;
#ifdef TRANSPARENCY_CAPTURE
#include "common/transparency.glsl"
#endif

#include "common/celestial_scene.glsl"
#include "common/eclipse_shadow.glsl"
#include "common/surface_ring_shadow.glsl"
#ifndef RING_PLANETSHINE_NUMERICAL
#define RING_PLANETSHINE_NUMERICAL 0
#endif

#include "common/ring_optics.glsl"
#if RING_PLANETSHINE_NUMERICAL
#include "common/ring_planetshine_disk.glsl"
#else
#include "common/ring_planetshine_lookup.glsl"
#endif
#include "common/ring_push.glsl"
#include "common/ring_refraction.glsl"

layout(set = 0, binding = 8) uniform sampler2DArray u_ring_gradients;

struct RingSample {
    vec3 linear_color;
    float alpha;
    float tau;
};

RingSample sample_ring_segment(RingSegmentGPU seg, float r, float footprint) {
    RingSample s;
    s.linear_color = vec3(0.0);
    s.alpha = 0.0;
    s.tau = 0.0;

    float inner_r = seg.bounds_opacity_scatter.x;
    float outer_r = seg.bounds_opacity_scatter.y;
    float span = max(outer_r - inner_r, 1e-9);
    float dr = min(footprint * 0.75, span * 0.05);
    if (r < inner_r - dr || r > outer_r + dr) return s;

    float t = clamp((r - inner_r) / span, 0.0, 1.0);
    float texels = footprint / span * float(textureSize(u_ring_gradients, 0).x);
    float lod = u_push.filter_profiles != 0u ? log2(max(texels, 1.0)) : 0.0;
    vec4 tex = textureLod(u_ring_gradients, vec3(t, 0.5, seg.phase_row.w), lod);

    s.linear_color = max(tex.rgb / max(tex.a, 1e-20), vec3(0.0));
    s.alpha = clamp(tex.a * seg.bounds_opacity_scatter.z, 0.0, RING_MAX_ALPHA);
    s.tau = ring_tau(s.alpha);
    return s;
}


void main() {
    vec3 host_center = ring_hosts[f_host_idx].center_min_r.xyz;
    vec3 pole_n = normalize(f_normal);

    vec3 ray_dir, hit_local, apparent_pos;
    float pixel_angle = 1.0/max(u_scene.screen_params.y*u_scene.screen_params.z, 1.0);
    if (!ring_refracted_hit(f_local_pos, host_center, pole_n, pixel_angle, hit_local, ray_dir, apparent_pos)) discard;
    float d_n = dot(ray_dir, pole_n);
    vec3 hit_pos = host_center + hit_local;
    vec4 clip = u_scene.view_proj*vec4(apparent_pos, 1.0);
    if (clip.w <= 0.0) discard;
    float fragment_depth = refraction_active() ? clip.z/clip.w : gl_FragCoord.z;
    if (fragment_depth < 0.0 || fragment_depth > 1.0) discard;
    gl_FragDepth = fragment_depth;

    float r = length(hit_local);
    float footprint = fwidth(r);

    if (abs(d_n) < 1e-7) {
        discard;
    }

    uint seg_count = min(u_push.segment_count, uint(RING_DRAW_CAPACITY));
    vec4 samples[RING_DRAW_CAPACITY];
    uint active_mask = 0u;
    float total_tau = 0.0;
    float total_faded_tau = 0.0;

    for (uint slot = 0u; slot < seg_count; ++slot) {
        uint i = (u_push.packed_segments >> (slot * 3u)) & 7u;
        RingSegmentGPU seg = ring_hosts[f_host_idx].segments[i];
        RingSample layer_sample = sample_ring_segment(seg, r, footprint);
        if (layer_sample.tau <= 0.0) continue;
        samples[slot] = vec4(layer_sample.linear_color, layer_sample.tau);
        active_mask |= 1u << slot;

        float inner_r = seg.bounds_opacity_scatter.x;
        float outer_r = seg.bounds_opacity_scatter.y;
        float dr = min(footprint * 0.75, (outer_r - inner_r) * 0.05);
        float fade = smoothstep(inner_r - dr, inner_r + dr, r) * (1.0 - smoothstep(outer_r - dr, outer_r + dr, r));
        total_tau += layer_sample.tau;
        total_faded_tau += ring_tau(layer_sample.alpha * fade);
    }

    if (total_tau <= 1e-12) {
        discard;
    }

    vec3 V = -ray_dir;
    float cam_side = dot(pole_n, V);
    float mu_v = max(abs(cam_side), 1e-5);
    float physical_alpha = ring_absorbed(total_tau / mu_v);
    float faded_alpha = ring_absorbed(total_faded_tau / mu_v);

    vec3 total_illum = vec3(0.0);

    for (uint s = 0u; s < (RING_SINGLE_STAR != 0 ? 1u : u_scene.num_stars); ++s) {
        if (!star_lights_body(ring_hosts[f_host_idx].counts_indices.x, s)) continue;
        vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
        float star_r = u_scene.stars_pos_radius[s].w;
        vec3 star_col = u_scene.stars_color_lum[s].rgb;
        float star_lum = u_scene.stars_color_lum[s].w;

        vec3 to_star = star_pos - hit_pos;
        float dist_star = length(to_star);
        if (dist_star <= 1e-5) continue;
        vec3 L = to_star / dist_star;

        vec3 shadow = evaluate_eclipse_shadow(hit_pos, star_pos, star_r, u_scene.num_bodies + f_host_idx, s);
        if (!any(greaterThan(shadow, vec3(0.0)))) continue;

        float star_ang_rad = clamp(star_r / dist_star, 0.0, 0.999);
        float sun_side = dot(pole_n, L);
#if RING_FINITE_SOURCE
        int source_samples = ring_visible_source_count(sun_side, star_ang_rad);
        vec4 polar_offsets = vec4(0.0);
        vec4 polar_weights = vec4(0.0);
        if (source_samples > 1) {
            ring_visible_polar_nodes(sun_side, star_ang_rad, polar_offsets, polar_weights);
        }
#endif

        vec3 direct_illum = vec3(0.0);
        uint remaining_segments = active_mask;
        while (remaining_segments != 0u) {
            uint slot = uint(findLSB(remaining_segments));
            uint i = (u_push.packed_segments >> (slot * 3u)) & 7u;
            remaining_segments &= remaining_segments - 1u;
            RingSegmentGPU seg = ring_hosts[f_host_idx].segments[i];
            vec4 props = vec4(seg.phase_row.xy, seg.bounds_opacity_scatter.w, seg.phase_row.z);
            vec3 seg_rad = vec3(0.0);
#if RING_FINITE_SOURCE
            for (int q = 0; q < source_samples; ++q) {
                if (source_samples > 1 && polar_weights[q] <= 0.0) continue;
                vec4 src = (source_samples == 1)
                    ? vec4(L, 1.0)
                    : ring_visible_polar_sample(L, pole_n, star_ang_rad, polar_offsets[q], polar_weights[q]);
                float sun_q = dot(pole_n, src.xyz);
                float cos_theta = clamp(-dot(src.xyz, V), -1.0, 1.0);
                bool same_side = (cam_side * sun_q >= 0.0);
                seg_rad += src.w * ring_radiance_rgb(total_tau, mu_v, sun_q, cos_theta,
                                                   props, samples[slot].xyz,
                                                   seg.color_flags.w > 0.5, same_side);
            }
#else
            float cos_theta = clamp(-dot(L, V), -1.0, 1.0);
#if RING_FIXED_SIDE >= 0
            bool same_side = RING_FIXED_SIDE != 0;
#else
            bool same_side = (cam_side * sun_side >= 0.0);
#endif
            seg_rad = ring_radiance_rgb(total_tau, mu_v, sun_side, cos_theta,
                                       props, samples[slot].xyz, seg.color_flags.w > 0.5, same_side);
#endif
            direct_illum += (samples[slot].w / total_tau) * seg_rad;
        }

        float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (dist_star * dist_star)) : 1.0;
        total_illum += star_col * direct_illum * (irradiance * shadow);
    }

    if (u_scene.ringshine_params.z != 0u) {
        uint host_body = ring_hosts[f_host_idx].counts_indices.x;
        vec3 moon_same_flux = vec3(0.0);
        vec3 moon_other_flux = vec3(0.0);
        vec3 moon_same_dir = vec3(0.0);
        vec3 moon_other_dir = vec3(0.0);
        float moon_same_sin = 0.0;
        float moon_other_sin = 0.0;
        float moon_same_w = 0.0;
        float moon_other_w = 0.0;
        for (int i = 0; i < 4; ++i) {
            vec3 flux = bodies[host_body].shine_irradiance[i].rgb;
            if (!any(greaterThan(flux, vec3(0.0)))) continue;
            vec3 light = bodies[host_body].shine_direction[i].xyz;
            float ext_sin_rad = sin(clamp(bodies[host_body].shine_direction[i].w, 0.0, 1.5));
            float w = max(dot(flux, vec3(0.2126, 0.7152, 0.0722)), 1e-30);
            if (cam_side * dot(pole_n, light) >= 0.0) {
                moon_same_flux += flux;
                moon_same_dir += light * w;
                moon_same_sin += ext_sin_rad * w;
                moon_same_w += w;
            } else {
                moon_other_flux += flux;
                moon_other_dir += light * w;
                moon_other_sin += ext_sin_rad * w;
                moon_other_w += w;
            }
        }
        bool has_moon_same = any(greaterThan(moon_same_flux, vec3(0.0)));
        bool has_moon_other = any(greaterThan(moon_other_flux, vec3(0.0)));
        if (has_moon_same || has_moon_other) {
            vec3 ms_light = has_moon_same ? normalize(moon_same_dir) : vec3(0.0, 1.0, 0.0);
            vec3 mo_light = has_moon_other ? normalize(moon_other_dir) : vec3(0.0, 1.0, 0.0);
            float ms_sin = has_moon_same ? (moon_same_sin / moon_same_w) : 0.0;
            float mo_sin = has_moon_other ? (moon_other_sin / moon_other_w) : 0.0;
            float ms_elev = dot(pole_n, ms_light);
            float mo_elev = dot(pole_n, mo_light);
            float ms_phase = clamp(-dot(ms_light, V), -1.0, 1.0);
            float mo_phase = clamp(-dot(mo_light, V), -1.0, 1.0);
            vec3 ms_terms = (has_moon_same && ms_sin > 1e-5) ? ring_extended_terms(ms_phase, ms_sin, true) : vec3(0.0);
            vec3 mo_terms = (has_moon_other && mo_sin > 1e-5) ? ring_extended_terms(mo_phase, mo_sin, false) : vec3(0.0);
            uint pending = active_mask;
            while (pending != 0u) {
                uint slot = uint(findLSB(pending));
                pending &= pending - 1u;
                uint seg_idx = (u_push.packed_segments >> (slot * 3u)) & 7u;
                RingSegmentGPU seg = ring_hosts[f_host_idx].segments[seg_idx];
                vec4 props = vec4(seg.phase_row.xy, seg.bounds_opacity_scatter.w, seg.phase_row.z);
                vec3 response = vec3(0.0);
                if (has_moon_same) response += moon_same_flux * ring_radiance_rgb_extended_terms(total_tau, mu_v, ms_elev, ms_phase,
                    props, samples[slot].xyz, seg.color_flags.w > 0.5, true, ms_sin, ms_terms);
                if (has_moon_other) response += moon_other_flux * ring_radiance_rgb_extended_terms(total_tau, mu_v, mo_elev, mo_phase,
                    props, samples[slot].xyz, seg.color_flags.w > 0.5, false, mo_sin, mo_terms);
                total_illum += (samples[slot].w / total_tau) * response;
            }
        }
#if !RING_PLANETSHINE_NUMERICAL
        float host_radius = bodies[host_body].pos_radius.w;
        vec3 host_albedo = clamp(bodies[host_body].color_is_star.rgb, 0.0, 1.0);
        for (uint s = 0u; s < (RING_SINGLE_STAR != 0 ? 1u : u_scene.num_stars); ++s) {
            if (!star_lights_body(host_body, s)) continue;
            vec3 to_star = u_scene.stars_pos_radius[s].xyz - host_center;
            float star_dist = length(to_star);
            if (star_dist <= 1e-12) continue;
            float irradiance = (u_scene.hdr_enabled != 0u) ? (u_scene.stars_color_lum[s].w / (star_dist * star_dist)) : 1.0;
            vec3 flux_scale = host_albedo * u_scene.stars_color_lum[s].rgb * irradiance;
            vec3 same_flux, same_light, other_flux, other_light;
            float same_spread, other_spread;
            ring_planetshine_sample_star(f_host_idx, s, hit_local, pole_n, to_star / star_dist,
                                         flux_scale, host_radius, cam_side,
                                         same_flux, same_light, same_spread,
                                         other_flux, other_light, other_spread);
            bool has_same = any(greaterThan(same_flux, vec3(0.0)));
            bool has_other = any(greaterThan(other_flux, vec3(0.0)));
            if (!has_same && !has_other) continue;
            float same_elevation = dot(pole_n, same_light);
            float other_elevation = dot(pole_n, other_light);
            float same_phase = clamp(-dot(same_light, V), -1.0, 1.0);
            float other_phase = clamp(-dot(other_light, V), -1.0, 1.0);
            vec3 same_terms = (has_same && same_spread > 1e-5) ? ring_extended_terms(same_phase, same_spread, true) : vec3(0.0);
            vec3 other_terms = (has_other && other_spread > 1e-5) ? ring_extended_terms(other_phase, other_spread, false) : vec3(0.0);
            uint remaining = active_mask;
            while (remaining != 0u) {
                uint slot = uint(findLSB(remaining));
                remaining &= remaining - 1u;
                uint seg_idx = (u_push.packed_segments >> (slot * 3u)) & 7u;
                RingSegmentGPU seg = ring_hosts[f_host_idx].segments[seg_idx];
                vec4 props = vec4(seg.phase_row.xy, seg.bounds_opacity_scatter.w, seg.phase_row.z);
                vec3 response = vec3(0.0);
                if (has_same) response += same_flux * ring_radiance_rgb_extended_terms(total_tau, mu_v, same_elevation, same_phase,
                    props, samples[slot].xyz, seg.color_flags.w > 0.5, true, same_spread, same_terms);
                if (has_other) response += other_flux * ring_radiance_rgb_extended_terms(total_tau, mu_v, other_elevation, other_phase,
                    props, samples[slot].xyz, seg.color_flags.w > 0.5, false, other_spread, other_terms);
                total_illum += (samples[slot].w / total_tau) * response;
            }
        }
#else
        for (int q = 0; q < RING_DISK_SAMPLES; ++q) {
            vec3 incident;
            vec4 source = ring_host_disk_sample(f_host_idx, hit_local, pole_n, q, incident);
            if (!any(greaterThan(incident, vec3(0.0)))) continue;
            float sun = dot(pole_n, source.xyz), phase = clamp(-dot(source.xyz, V), -1.0, 1.0);
            uint pending = active_mask;
            while (pending != 0u) {
                uint slot = uint(findLSB(pending)); pending &= pending-1u;
                uint i = (u_push.packed_segments >> (slot*3u)) & 7u;
                RingSegmentGPU seg = ring_hosts[f_host_idx].segments[i];
                vec4 props = vec4(seg.phase_row.xy, seg.bounds_opacity_scatter.w, seg.phase_row.z);
                vec3 response = ring_radiance_rgb(total_tau, mu_v, sun, phase, props,
                    samples[slot].xyz, seg.color_flags.w > 0.5, cam_side*sun >= 0.0);
                total_illum += incident*(samples[slot].w/total_tau)*response;
            }
        }
#endif
    }

    float exposure = (u_scene.hdr_enabled != 0u) ? u_scene.screen_params.w : 1.0;
    vec3 final_color = max(total_illum, vec3(0.0)) * exposure;
    float fade_scale = faded_alpha / max(1e-12, physical_alpha);
    out_color = vec4(final_color * fade_scale, faded_alpha);
#ifdef TRANSPARENCY_CAPTURE
    if (capture_transparency(out_color, fragment_depth)) out_color = vec4(0.0);
#endif
}
