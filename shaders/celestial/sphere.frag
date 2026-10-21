#version 460

layout(location = 0) in vec3 f_world_pos;
layout(location = 1) in vec3 f_local_pos;
layout(location = 2) in vec3 f_normal;
layout(location = 3) in float f_subpixel_factor;
layout(location = 4) flat in uint f_body_idx;
layout(location = 7) in vec3 f_base_chord;
layout(location = 8) in vec3 f_apparent_pos;

#ifdef TERRAIN_SURFACE
layout(location = 5) flat in vec4 f_tile_color;
layout(location = 6) in vec2 f_tile_uv;
#endif

#include "common/celestial_scene.glsl"
#include "common/body_surface.glsl"
#include "common/surface_depth.glsl"
#include "common/eclipse_shadow.glsl"
#include "common/surface_textures.glsl"
#include "common/surface_ring_shadow.glsl"
#include "common/ringshine_lookup.glsl"
#include "common/surface_atmosphere.glsl"

layout(location = 0) out vec4 out_color;
layout(depth_greater) out float gl_FragDepth;

void main() {
    BodyInstanceGPU b = bodies[f_body_idx];
    gl_FragDepth = curved_surface_depth(b, f_world_pos, f_apparent_pos, f_base_chord, gl_FragCoord.z);
#ifdef TERRAIN_SURFACE
    if (f_tile_color.w > 0.5) {
        vec2 edge = min(f_tile_uv, 1.0 - f_tile_uv);
        vec2 line = smoothstep(vec2(0.0), max(fwidth(f_tile_uv) * 1.5, vec2(1e-5)), edge);
        out_color = vec4(f_tile_color.rgb * mix(0.2, 1.0, min(line.x, line.y)), 1.0);
        return;
    }
#endif
    if (f_subpixel_factor >= 1.0) {
        discard;
    }

    vec3 N = surface_shading_normal(f_body_idx, f_local_pos, normalize(f_normal),
        body_surface_rotation(body_pole(b), b.spin_flags.x));
    vec3 V = normalize(-f_world_pos);
    float exposure = (u_scene.hdr_enabled != 0u) ? u_scene.screen_params.w : 1.0;

    if (b.spin_flags.z > 0.5) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float mesh_weight = 1.0 - f_subpixel_factor;

    if (b.color_is_star.w > 0.5) {
        vec3 rgb = stellar_radiance(b, N, V) * exposure * mesh_weight;
        out_color = vec4(rgb, 1.0);
        return;
    }

    vec3 base_albedo = b.color_is_star.rgb;
    float lat_band = sin(f_local_pos.y * 18.0) * 0.04 + cos(f_local_pos.x * 9.0 + f_local_pos.z * 9.0) * 0.03;
    vec3 albedo = clamp(base_albedo * (1.0 + lat_band), 0.0, 1.0);
    albedo = surface_albedo(f_body_idx, f_local_pos, albedo);

    vec3 total_light = vec3(0.0);
    for (uint s = 0; s < u_scene.num_stars; ++s) {
        vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
        float star_r = u_scene.stars_pos_radius[s].w;
        vec3 star_col = u_scene.stars_color_lum[s].rgb;
        float star_lum = u_scene.stars_color_lum[s].w;

        vec3 to_star = star_pos - f_world_pos;
        float d_au = max(length(to_star), 1e-5);
        vec3 L = to_star / d_au;
        vec3 transmission, diffuse;
        surface_atmosphere(f_body_idx, f_world_pos-b.pos_radius.xyz, f_base_chord, L, N, transmission, diffuse);
        float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0;

        float ndotl = max(dot(N, L), 0.0);
        float shadow = 1.0;
        if (ndotl > 0.0 || any(greaterThan(diffuse, vec3(0.0)))) {
            shadow = evaluate_eclipse_shadow(f_world_pos, star_pos, star_r, f_body_idx, s);
            if (shadow > 1e-4 && u_scene.num_ring_hosts > 0u) {
                float sun_ang_rad = star_r / d_au;
                uint candidates = u_scene.ring_shadow_params.y != 0u
                    ? (1u << u_scene.num_ring_hosts) - 1u
                    : eclipse_words[u_scene.ring_shadow_params.x + f_body_idx * u_scene.num_stars + s];
                if (candidates != 0u)
                    shadow *= compute_surface_ring_shadow(f_world_pos, L, sun_ang_rad, candidates);
            }
        }

        total_light += albedo * star_col * irradiance * diffuse * shadow;

        if (ndotl > 0.0 && shadow > 1e-4) {
            float brdf = surface_reflectance(N, L, V);
            total_light += albedo * star_col * irradiance * brdf * shadow * transmission;
        }
    }

    if (u_scene.ringshine_params.x != 0u) {
        vec3 ringshine_light = vec3(0.0);
        for (uint r = 0; r < u_scene.ringshine_params.y; ++r) {
            if (ring_hosts[r].counts_indices.x != f_body_idx) continue;
            vec3 host_center = ring_hosts[r].center_min_r.xyz;
            vec3 pole = ring_hosts[r].pole_max_r.xyz;
            float host_r = ring_hosts[r].tangent_radius.w;
            vec3 rel = f_world_pos - host_center;
            float dist2 = dot(rel, rel);
            if (dist2 > 2.25 * host_r * host_r || dist2 < 1e-12) continue;

            vec3 P_eq = rel - pole * dot(rel, pole);
            float len_eq = length(P_eq);
            P_eq = len_eq > 1e-5 ? P_eq / len_eq : vec3(1.0, 0.0, 0.0);
            for (uint s = 0; s < u_scene.num_stars; ++s) {
                vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
                float star_r = u_scene.stars_pos_radius[s].w;
                vec3 star_col = u_scene.stars_color_lum[s].rgb;
                float star_lum = u_scene.stars_color_lum[s].w;
                vec3 to_star = star_pos - f_world_pos;
                float d_au = max(length(to_star), 1e-5);
                vec3 L = to_star / d_au;

                vec3 antiL_eq = (-L) - pole * dot(-L, pole);
                float len_anti = length(antiL_eq);
                antiL_eq = len_anti > 1e-5 ? antiL_eq / len_anti : vec3(1.0, 0.0, 0.0);
                float phi = atan(dot(cross(antiL_eq, P_eq), pole), dot(P_eq, antiL_eq));

                vec3 P_dir = normalize(rel);
                float slat = dot(P_dir, pole);
                vec3 irradiance_map = ringshine_sample(phi, slat, int(r), int(s));
                float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0f;
                ringshine_light += albedo * irradiance_map * star_col * irradiance / 3.14159265358979323846;
            }
        }
        total_light += ringshine_light;
    }

    vec3 rgb = total_light * exposure * mesh_weight;
    out_color = vec4(rgb, 1.0);
}
