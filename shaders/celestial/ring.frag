#version 460
#ifdef TRANSPARENCY_CAPTURE
#extension GL_KHR_shader_subgroup_ballot : require
#endif
#define RING_REFRACTION_TABLE

#ifndef RING_PLANETSHINE
#define RING_PLANETSHINE 1
#endif
#ifndef RING_REFRACTION
#define RING_REFRACTION 1
#endif

#if defined(TRANSPARENCY_CAPTURE) || !RING_REFRACTION
layout(early_fragment_tests) in;
#endif

layout(location = 0) in vec3 f_local_pos;
layout(location = 1) in vec3 f_normal;
layout(location = 2) flat in uint f_host_idx;

layout(location = 0) out vec4 out_color;
#ifdef TRANSPARENCY_CAPTURE
#define TRANSPARENCY_RING_SLOT
#include "common/transparency.glsl"
#endif

#include "common/celestial_scene.glsl"
#include "common/eclipse_shadow.glsl"
#include "common/surface_ring_shadow.glsl"
#ifndef RING_PLANETSHINE_NUMERICAL
#define RING_PLANETSHINE_NUMERICAL 0
#endif

#include "common/ring_optics.glsl"
#if RING_PLANETSHINE
#if RING_PLANETSHINE_NUMERICAL
#include "common/ring_planetshine_disk.glsl"
#else
#include "common/ring_planetshine_lookup.glsl"
#endif
#endif
#include "common/ring_push.glsl"
#if RING_REFRACTION
#include "common/ring_refraction.glsl"
#endif

layout(set = 0, binding = 8) uniform sampler2DArray u_ring_gradients;

struct RingSample {
    vec3 linear_color;
    float alpha;
    float tau;
    float coverage;
};

float ring_edge_coverage(float r, float inner_r, float outer_r, float filter_w) {
    float t0 = clamp((r - outer_r) / filter_w + 0.5, 0.0, 1.0);
    float t1 = clamp((r - inner_r) / filter_w + 0.5, 0.0, 1.0);
    float d = t1 - t0;
    return d * (3.0 * (t1 + t0) - 2.0 * (t1 * t1 + t1 * t0 + t0 * t0));
}

RingSample sample_ring_segment(RingSegmentGPU seg, float r, float radial_grad, float filter_w) {
    RingSample s;
    s.linear_color = vec3(0.0);
    s.alpha = 0.0;
    s.tau = 0.0;
    s.coverage = 0.0;

    float inner_r = seg.bounds_opacity_scatter.x;
    float outer_r = seg.bounds_opacity_scatter.y;
    float span = max(outer_r - inner_r, 1e-9);
    float dr = 0.5 * filter_w;
    if (r < inner_r - dr || r > outer_r + dr) return s;

    float cov = ring_edge_coverage(r, inner_r, outer_r, filter_w);
    if (cov <= 1e-6) return s;

    float t = clamp((r - inner_r) / span, 0.0, 1.0);
    float texels = radial_grad / span * float(textureSize(u_ring_gradients, 0).x);
    float lod = log2(max(texels, 1.0));
    vec4 tex = textureLod(u_ring_gradients, vec3(t, 0.5, seg.phase_row.w), lod);

    s.linear_color = max(tex.rgb / max(tex.a, 1e-20), vec3(0.0));
    s.alpha = clamp(tex.a * seg.bounds_opacity_scatter.z, 0.0, RING_MAX_ALPHA);
    s.tau = ring_tau(s.alpha);
    s.coverage = cov;
    return s;
}


void main() {
    vec3 host_center = ring_hosts[f_host_idx].center_min_r.xyz;
    float host_outer_r = ring_hosts[f_host_idx].pole_max_r.w;
    uint host_body = ring_hosts[f_host_idx].counts_indices.x;
    vec3 host_ring_u = ring_hosts[f_host_idx].tangent_radius.xyz;
    vec3 host_ring_v = ring_hosts[f_host_idx].bitangent_oblateness.xyz;
    vec3 pole_n = normalize(f_normal);

#if RING_REFRACTION
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
#else
    vec3 hit_local = f_local_pos;
    vec3 hit_pos = host_center + hit_local;
    vec3 ray_dir = normalize(hit_pos);
    float d_n = dot(ray_dir, pole_n);
    float fragment_depth = gl_FragCoord.z;
#endif

    float r = length(hit_local);
    float radial_grad = max(length(vec2(dFdx(r), dFdy(r))), 1e-9);
    float max_aa_dr = host_outer_r * 0.019;
    float filter_w = min(radial_grad * 1.5, 2.0 * max_aa_dr);

    if (abs(d_n) < 1e-7) {
        discard;
    }

    uint seg_count = min(u_push.segment_count, uint(RING_DRAW_CAPACITY));
    vec4 samples[RING_DRAW_CAPACITY];
    float coverages[RING_DRAW_CAPACITY];
    vec4 seg_props[RING_DRAW_CAPACITY];
    uint active_mask = 0u;
    uint textured_mask = 0u;
    float max_cov = 0.0;

    for (uint slot = 0u; slot < seg_count; ++slot) {
        uint i = (u_push.packed_segments >> (slot * 3u)) & 7u;
        RingSegmentGPU seg = ring_hosts[f_host_idx].segments[i];
        RingSample layer_sample = sample_ring_segment(seg, r, radial_grad, filter_w);
        if (layer_sample.tau <= 0.0 || layer_sample.coverage <= 0.0) continue;
        samples[slot] = vec4(layer_sample.linear_color, layer_sample.tau);
        coverages[slot] = layer_sample.coverage;
        seg_props[slot] = vec4(seg.phase_row.xy, seg.bounds_opacity_scatter.w, seg.phase_row.z);
        if (seg.color_flags.w > 0.5) textured_mask |= (1u << slot);
        active_mask |= 1u << slot;
        max_cov = max(max_cov, layer_sample.coverage);
    }

    if (active_mask == 0u || max_cov <= 1e-6) {
        discard;
    }

#if RING_DRAW_CAPACITY == 1
    bool single_active = true;
    uint single_slot = 0u;
#else
    bool single_active = (active_mask & (active_mask - 1u)) == 0u;
    uint single_slot = uint(findLSB(active_mask));
#endif
    float total_tau = 0.0;
    if (single_active) {
        total_tau = samples[single_slot].w;
        samples[single_slot].w = 1.0;
        if (total_tau <= 1e-12) {
            discard;
        }
    } else {
        uint pending_cov = active_mask;
        while (pending_cov != 0u) {
            uint slot = uint(findLSB(pending_cov));
            pending_cov &= pending_cov - 1u;
            samples[slot].w *= coverages[slot] / max_cov;
            total_tau += samples[slot].w;
        }
        if (total_tau <= 1e-12) {
            discard;
        }
        float inv_total_tau = 1.0 / total_tau;
        uint pending_w = active_mask;
        while (pending_w != 0u) {
            uint slot = uint(findLSB(pending_w));
            pending_w &= pending_w - 1u;
            samples[slot].w *= inv_total_tau;
        }
    }

    vec3 V = -ray_dir;
    float cam_side = dot(pole_n, V);
    float mu_v = max(abs(cam_side), 1e-5);
    float physical_alpha = ring_absorbed(total_tau / mu_v);

#if RING_PLANETSHINE || (RING_SINGLE_STAR == 0)
    RingMediumOptics single_medium;
    bool single_medium_ready = false;
#endif
    vec3 total_illum = vec3(0.0);

    for (uint s = 0u; s < (RING_SINGLE_STAR != 0 ? 1u : u_scene.num_stars); ++s) {
        if (!star_lights_body(host_body, s)) continue;
        vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
        float star_r = u_scene.stars_pos_radius[s].w;
        vec3 star_col = u_scene.stars_color_lum[s].rgb;
        float star_lum = u_scene.stars_color_lum[s].w;

        vec3 to_star = star_pos - hit_pos;
        float dist_star = length(to_star);
        if (dist_star <= 1e-5) continue;
        vec3 L = to_star / dist_star;

        vec3 shadow = evaluate_ring_eclipse_shadow(
            hit_pos, hit_local, host_ring_u, host_ring_v, host_outer_r,
            L, dist_star, star_r, u_scene.num_bodies + f_host_idx, s);
        if (!any(greaterThan(shadow, vec3(0.0)))) continue;

        float star_ang_rad = clamp(star_r / dist_star, 0.0, 0.999);
        float sun_side = dot(pole_n, L);
#if RING_PLANETSHINE || (RING_SINGLE_STAR == 0)
        if (single_active && !single_medium_ready) {
            single_medium = ring_medium_setup(
                total_tau, mu_v, seg_props[single_slot], samples[single_slot].xyz,
                (textured_mask & (1u << single_slot)) != 0u);
            single_medium_ready = true;
        }
#endif
#if RING_FINITE_SOURCE
        int source_samples = ring_visible_source_count(sun_side, star_ang_rad);
        vec4 polar_offsets = vec4(0.0);
        vec4 polar_weights = vec4(0.0);
        if (source_samples > 1) {
            ring_visible_polar_nodes(sun_side, star_ang_rad, polar_offsets, polar_weights);
        }
        vec4 q_tf[4];
        float q_w[4];
        float q_sun[4];
        float q_cos[4];
        float q_opp[4];
        bool q_same[4];
        for (int q = 0; q < source_samples; ++q) {
            if (source_samples > 1 && polar_weights[q] <= 0.0) {
                q_w[q] = 0.0;
                continue;
            }
            vec4 src = (source_samples == 1)
                ? vec4(L, 1.0)
                : ring_visible_polar_sample(L, pole_n, star_ang_rad, polar_offsets[q], polar_weights[q]);
            float sun_q = dot(pole_n, src.xyz);
            float cos_q = clamp(-dot(src.xyz, V), -1.0, 1.0);
            bool same_q = (cam_side * sun_q >= 0.0);
            q_w[q] = src.w;
            q_sun[q] = sun_q;
            q_cos[q] = cos_q;
            q_same[q] = same_q;
            q_opp[q] = same_q ? ring_opposition_factor(-cos_q, total_tau) : 1.0;
            q_tf[q] = ring_transfer_face(total_tau, mu_v, sun_q, same_q);
        }
        vec3 direct_illum = vec3(0.0);
#if RING_PLANETSHINE || (RING_SINGLE_STAR == 0)
        if (single_active) {
            bool is_tex = (textured_mask & (1u << single_slot)) != 0u;
            vec4 props = seg_props[single_slot];
            for (int q = 0; q < source_samples; ++q) {
                if (q_w[q] <= 0.0) continue;
                direct_illum += q_w[q] * ring_radiance_face(
                    single_medium, q_tf[q], mu_v, q_sun[q], q_cos[q],
                    props, is_tex, q_same[q], 1.0, q_opp[q]);
            }
        } else {
#endif
            uint remaining_segments = active_mask;
            while (remaining_segments != 0u) {
                uint slot = uint(findLSB(remaining_segments));
                remaining_segments &= remaining_segments - 1u;
                vec4 props = seg_props[slot];
                bool is_tex = (textured_mask & (1u << slot)) != 0u;
                RingMediumOptics seg_m = ring_medium_setup(total_tau, mu_v, props, samples[slot].xyz, is_tex);
                vec3 seg_rad = vec3(0.0);
                for (int q = 0; q < source_samples; ++q) {
                    if (q_w[q] <= 0.0) continue;
                    seg_rad += q_w[q] * ring_radiance_face(
                        seg_m, q_tf[q], mu_v, q_sun[q], q_cos[q],
                        props, is_tex, q_same[q], 1.0, q_opp[q]);
                }
                direct_illum += single_active ? seg_rad : (samples[slot].w * seg_rad);
            }
#if RING_PLANETSHINE || (RING_SINGLE_STAR == 0)
        }
#endif
#else
        float cos_theta = clamp(-dot(L, V), -1.0, 1.0);
#if RING_FIXED_SIDE >= 0
        bool same_side = RING_FIXED_SIDE != 0;
#else
        bool same_side = (cam_side * sun_side >= 0.0);
#endif
        float direct_opp = same_side ? ring_opposition_factor(-cos_theta, total_tau) : 1.0;
        vec4 direct_tf = ring_transfer_face(total_tau, mu_v, sun_side, same_side);

        vec3 direct_illum = vec3(0.0);
#if RING_PLANETSHINE || (RING_SINGLE_STAR == 0)
        if (single_active) {
            direct_illum = ring_radiance_face(
                single_medium, direct_tf, mu_v, sun_side, cos_theta,
                seg_props[single_slot], (textured_mask & (1u << single_slot)) != 0u,
                same_side, 1.0, direct_opp);
        } else {
#endif
            uint remaining_segments = active_mask;
            while (remaining_segments != 0u) {
                uint slot = uint(findLSB(remaining_segments));
                remaining_segments &= remaining_segments - 1u;
                vec4 props = seg_props[slot];
                bool is_tex = (textured_mask & (1u << slot)) != 0u;
                RingMediumOptics seg_m = ring_medium_setup(total_tau, mu_v, props, samples[slot].xyz, is_tex);
                vec3 seg_rad = ring_radiance_face(
                    seg_m, direct_tf, mu_v, sun_side, cos_theta,
                    props, is_tex, same_side, 1.0, direct_opp);
                direct_illum += single_active ? seg_rad : (samples[slot].w * seg_rad);
            }
#if RING_PLANETSHINE || (RING_SINGLE_STAR == 0)
        }
#endif
#endif

        float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (dist_star * dist_star)) : 1.0;
        total_illum += star_col * direct_illum * (irradiance * shadow);
    }

#if RING_PLANETSHINE
    if (u_scene.ringshine_params.z != 0u) {
        if (single_active && !single_medium_ready) {
            single_medium = ring_medium_setup(
                total_tau, mu_v, seg_props[single_slot], samples[single_slot].xyz,
                (textured_mask & (1u << single_slot)) != 0u);
            single_medium_ready = true;
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
            float same_elevation = has_same ? dot(pole_n, same_light) : 0.0;
            float other_elevation = has_other ? dot(pole_n, other_light) : 0.0;
            float same_phase = has_same ? clamp(-dot(same_light, V), -1.0, 1.0) : 0.0;
            float other_phase = has_other ? clamp(-dot(other_light, V), -1.0, 1.0) : 0.0;
            vec3 same_terms = (has_same && same_spread > 1e-5) ? ring_extended_terms(same_phase, same_spread, true) : vec3(0.0);
            vec3 other_terms = (has_other && other_spread > 1e-5) ? ring_extended_terms(other_phase, other_spread, false) : vec3(0.0);
            float same_opp = (has_same && same_spread <= 1e-5) ? ring_opposition_factor(-same_phase, total_tau)
                : (has_same ? ring_opposition_factor_extended_terms(same_terms.z, total_tau, same_terms.y) : 1.0);
            float same_broaden = (same_spread > 1e-5) ? same_terms.x : 1.0;
            float other_broaden = (other_spread > 1e-5) ? other_terms.x : 1.0;
            vec4 same_tf = has_same ? ring_transfer_face(total_tau, mu_v, same_elevation, true) : vec4(0.0);
            vec4 other_tf = has_other ? ring_transfer_face(total_tau, mu_v, other_elevation, false) : vec4(0.0);
            if (single_active) {
                total_illum += ring_radiance_planetshine_pair_medium(
                    single_medium, total_tau, mu_v, seg_props[single_slot],
                    (textured_mask & (1u << single_slot)) != 0u,
                    has_same, same_flux, same_tf, same_elevation, same_phase, same_broaden, same_opp,
                    has_other, other_flux, other_tf, other_elevation, other_phase, other_broaden);
            } else {
                uint remaining = active_mask;
                while (remaining != 0u) {
                    uint slot = uint(findLSB(remaining));
                    remaining &= remaining - 1u;
                    total_illum += samples[slot].w * ring_radiance_planetshine_pair(
                        total_tau, mu_v, seg_props[slot], samples[slot].xyz, (textured_mask & (1u << slot)) != 0u,
                        has_same, same_flux, same_tf, same_elevation, same_phase, same_broaden, same_opp,
                        has_other, other_flux, other_tf, other_elevation, other_phase, other_broaden);
                }
            }
        }
#else
        for (int q = 0; q < RING_DISK_SAMPLES; ++q) {
            vec3 incident;
            vec4 source = ring_host_disk_sample(f_host_idx, hit_local, pole_n, q, incident);
            if (!any(greaterThan(incident, vec3(0.0)))) continue;
            float sun = dot(pole_n, source.xyz), phase = clamp(-dot(source.xyz, V), -1.0, 1.0);
            bool same_q = cam_side * sun >= 0.0;
            vec4 q_tf = ring_transfer_face(total_tau, mu_v, sun, same_q);
            float q_opp = same_q ? ring_opposition_factor(-phase, total_tau) : 1.0;
            if (single_active) {
                vec3 response = ring_radiance_face(
                    single_medium, q_tf, mu_v, sun, phase,
                    seg_props[single_slot], (textured_mask & (1u << single_slot)) != 0u, same_q, 1.0, q_opp);
                total_illum += incident * response;
            } else {
                uint pending = active_mask;
                while (pending != 0u) {
                    uint slot = uint(findLSB(pending)); pending &= pending-1u;
                    bool is_tex = (textured_mask & (1u << slot)) != 0u;
                    RingMediumOptics seg_m = ring_medium_setup(total_tau, mu_v, seg_props[slot], samples[slot].xyz, is_tex);
                    vec3 response = ring_radiance_face(
                        seg_m, q_tf, mu_v, sun, phase, seg_props[slot], is_tex, same_q, 1.0, q_opp);
                    total_illum += incident * samples[slot].w * response;
                }
            }
        }
#endif
    }
#endif

    float exposure = (u_scene.hdr_enabled != 0u) ? u_scene.screen_params.w : 1.0;
    vec3 final_color = max(total_illum, vec3(0.0)) * exposure;
    out_color = vec4(final_color * max_cov, physical_alpha * max_cov);
#ifdef TRANSPARENCY_CAPTURE
    if (capture_ring(out_color, fragment_depth, RING_REFRACTION != 0)) out_color = vec4(0.0);
#endif
}
