#version 460

layout(location = 0) in vec3 f_world_pos;
layout(location = 1) in vec3 f_local_pos;
layout(location = 2) in vec3 f_normal;
layout(location = 3) in float f_subpixel_factor;
layout(location = 4) flat in uint f_body_idx;

#ifdef TERRAIN_SURFACE
layout(location = 5) flat in vec4 f_tile_color;
layout(location = 6) in vec2 f_tile_uv;
#endif

#include "common/celestial_scene.glsl"
#include "common/surface_textures.glsl"

layout(location = 0) out vec4 out_color;

layout(std430, set = 0, binding = 2) readonly buffer EclipseCandidateBuffer {
    uint eclipse_words[];
};

float compute_eclipse_shadow(vec3 frag_pos, vec3 star_pos, float star_radius, uint self_idx, uint star_idx) {
    uint range = (self_idx * u_scene.num_stars + star_idx) * 2u;
    uint start = eclipse_words[range];
    uint count = eclipse_words[range + 1u];
    if (count == 0u) return 1.0;
    vec3 to_star = star_pos - frag_pos;
    float dist_star = length(to_star);
    if (dist_star <= 1e-9) return 1.0;
    vec3 L = to_star / dist_star;

    float vis = 1.0;
    for (uint candidate = 0u; candidate < count; ++candidate) {
        uint i = eclipse_words[start + candidate];
        BodyInstanceGPU caster = bodies[i];

        vec3 oc = caster.pos_radius.xyz - frag_pos;
        float proj = dot(oc, L);
        if (proj <= 0.0 || proj >= dist_star) continue;

        float d_perp = length(oc - L * proj);
        float r_caster = caster.pos_radius.w;
        float r_sun_proj = star_radius * (proj / dist_star);
        float r_umbra = max(0.0, r_caster - r_sun_proj);
        float r_penumbra = r_caster + r_sun_proj;

        if (d_perp < r_penumbra) {
            float t = clamp((d_perp - r_umbra) / max(r_penumbra - r_umbra, 1e-9), 0.0, 1.0);
            vis *= pow(smoothstep(0.0, 1.0, t), 0.625);
            if (vis <= 1e-4) return 0.0;
        }
    }
    return vis;
}

void main() {
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

    BodyInstanceGPU b = bodies[f_body_idx];
    vec3 N = normalize(f_normal);
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

        float ndotl = max(dot(N, L), 0.0);
        if (ndotl > 0.0) {
            float shadow = compute_eclipse_shadow(f_world_pos, star_pos, star_r, f_body_idx, s);
            float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0;
            float brdf = surface_reflectance(N, L, V);
            total_light += albedo * star_col * irradiance * brdf * shadow;
        }
    }

    vec3 rgb = total_light * exposure * mesh_weight;
    out_color = vec4(rgb, 1.0);
}
