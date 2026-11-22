#version 460
#extension GL_EXT_control_flow_attributes : enable
#define SF_ENABLE_DONT_UNROLL 1

#ifndef SURFACE_FULL_LIGHTING
#define SURFACE_FULL_LIGHTING 1
#endif
#ifndef SURFACE_ATMOSPHERE
#define SURFACE_ATMOSPHERE 1
#endif
#ifndef SURFACE_OCEAN
#define SURFACE_OCEAN 1
#endif
#ifndef SURFACE_OCEAN_FFT
#define SURFACE_OCEAN_FFT 1
#endif
#ifndef SURFACE_ALBEDO_MAP
#define SURFACE_ALBEDO_MAP 1
#endif
#ifndef SURFACE_NORMALS
#define SURFACE_NORMALS 1
#endif
#ifndef SURFACE_HORIZONS
#define SURFACE_HORIZONS 1
#endif
#ifndef SURFACE_ECLIPSE_SHADOW
#define SURFACE_ECLIPSE_SHADOW 1
#endif
#ifndef SURFACE_RING_SHADOW
#define SURFACE_RING_SHADOW 1
#endif
#ifndef SURFACE_RINGSHINE
#define SURFACE_RINGSHINE 1
#endif
#ifndef SURFACE_PLANETSHINE
#define SURFACE_PLANETSHINE 1
#endif
#ifndef SURFACE_DEBUG
#define SURFACE_DEBUG 1
#endif
#ifndef SURFACE_ATM_HORIZON_CLEAR
#define SURFACE_ATM_HORIZON_CLEAR 0
#endif

layout(location = 0) in vec3 f_world_pos;
layout(location = 1) in vec3 f_local_pos;
layout(location = 2) in vec3 f_normal;
layout(location = 3) in float f_subpixel_factor;
layout(location = 4) flat in uint f_body_idx;
layout(location = 7) in vec3 f_base_chord;
layout(location = 8) in vec3 f_apparent_pos;
layout(location = 9) in vec3 f_body_offset;

#ifdef TERRAIN_SURFACE
layout(location = 5) flat in vec4 f_tile_color;
layout(location = 6) in vec2 f_tile_uv;
#endif

#include "common/celestial_scene.glsl"
#include "common/body_surface.glsl"
#include "common/surface_depth.glsl"
#include "common/stellar_disk.glsl"
#if SURFACE_ECLIPSE_SHADOW
#include "common/eclipse_shadow.glsl"
#elif SURFACE_RING_SHADOW
#ifndef ECLIPSE_CANDIDATE_BUFFER_DEFINED
#define ECLIPSE_CANDIDATE_BUFFER_DEFINED
layout(std430, set = 0, binding = 2) readonly buffer EclipseCandidateBuffer {
    uint eclipse_words[];
};
#endif
#endif
#if SURFACE_RING_SHADOW || SURFACE_RINGSHINE
#include "common/ring_hosts.glsl"
#endif
#if SURFACE_RING_SHADOW
#include "common/surface_ring_shadow.glsl"
#endif
#if SURFACE_RINGSHINE
#include "common/ringshine_lookup.glsl"
#endif
#if SURFACE_ALBEDO_MAP || SURFACE_NORMALS || SURFACE_HORIZONS
#include "common/surface_textures.glsl"
#endif
#if SURFACE_HORIZONS
#include "common/terrain_horizon.glsl"
#endif
#if SURFACE_PLANETSHINE
#include "common/planetshine_lookup.glsl"
#endif
#if SURFACE_ATMOSPHERE || SURFACE_FULL_LIGHTING
#include "common/surface_atmosphere.glsl"
#endif
#if SURFACE_OCEAN
#define SURFACE_OCEAN_MIP_SUMMARY 1
#define SURFACE_OCEAN_WIND_LUT 1
#include "common/fluid_surface.glsl"
#endif

layout(location = 0) out vec4 out_color;
layout(depth_greater) out float gl_FragDepth;

void main() {
    BodyInstanceGPU b = bodies[f_body_idx];
    float sag_factor = max(0.0, 1.0 - length(f_base_chord));
    gl_FragDepth = curved_surface_depth_sag(b, f_world_pos, f_body_offset, f_apparent_pos, sag_factor, gl_FragCoord.z);
#if defined(TERRAIN_SURFACE) && SURFACE_DEBUG
    if (f_tile_color.w > 0.5) {
        vec2 edge = min(f_tile_uv, 1.0 - f_tile_uv);
        vec2 line = smoothstep(vec2(0.0), max(fwidth(f_tile_uv) * 1.5, vec2(1e-5)), edge);
        out_color = vec4(f_tile_color.rgb * mix(0.2, 1.0, min(line.x, line.y)), 1.0);
        return;
    }
#endif
#if defined(TERRAIN_SURFACE) && SURFACE_DEBUG && !SURFACE_FULL_LIGHTING && !SURFACE_ATMOSPHERE && !SURFACE_OCEAN && !SURFACE_ALBEDO_MAP && !SURFACE_NORMALS && !SURFACE_HORIZONS && !SURFACE_ECLIPSE_SHADOW && !SURFACE_RING_SHADOW && !SURFACE_RINGSHINE && !SURFACE_PLANETSHINE
    out_color = vec4(0.0, 0.0, 0.0, 1.0);
    return;
#else
#ifndef TERRAIN_SURFACE
    if (f_subpixel_factor >= 1.0) {
        discard;
    }
    float mesh_weight = 1.0 - f_subpixel_factor;
#else
    const float mesh_weight = 1.0;
#endif

#if SURFACE_FULL_LIGHTING
    if (b.spin_flags.z > 0.5) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
#endif
    float exposure = (u_scene.hdr_enabled != 0u) ? u_scene.screen_params.w : 1.0;
#if !defined(TERRAIN_SURFACE)
    vec3 V = normalize(-f_world_pos);
    if (b.color_is_star.w > 0.5) {
        vec3 N_star = normalize(f_normal);
        vec3 rgb = stellar_radiance(b, N_star, V) * exposure * mesh_weight;
        out_color = vec4(rgb, 1.0);
        return;
    }
#elif SURFACE_FULL_LIGHTING || SURFACE_OCEAN
    vec3 V = normalize(-f_world_pos);
#endif

#if SURFACE_OCEAN && SURFACE_OCEAN_FFT
    vec3 world_dx = dFdx(f_world_pos), world_dy = dFdy(f_world_pos);
#endif
#if SURFACE_ALBEDO_MAP || SURFACE_NORMALS || SURFACE_HORIZONS
    vec3 surf_dir = normalize(f_local_pos);
    float surf_footprint = max(length(dFdx(surf_dir)), length(dFdy(surf_dir)));
    vec3 surf_face_uv = surface_face_uv(surf_dir);
    vec2 surf_uv01 = clamp(surf_face_uv.xy * 0.5 + 0.5, 0.0, 1.0);
    uint surf_face = uint(surf_face_uv.z);
    float surf_raw_lod = log2(3.0 / max(128.0 * surf_footprint, 1e-9));
#endif
#if SURFACE_FULL_LIGHTING && defined(TERRAIN_SURFACE)
    vec3 geometric = cross(dFdx(f_world_pos), dFdy(f_world_pos));
    float geometric_length = length(geometric);
#endif

    vec3 N;
#if !SURFACE_FULL_LIGHTING
#if SURFACE_HORIZONS
    vec2 terrain_vis = vec2(1.0);
#else
    const vec2 terrain_vis = vec2(1.0);
#endif
    {
#if SURFACE_NORMALS || SURFACE_HORIZONS
        mat3 rotation = body_surface_rotation(body_pole(b), b.spin_flags.x);
        SurfaceNormalLookup normal_lookup = surface_normal_lookup_at(f_body_idx, surf_uv01, surf_face, surf_raw_lod);
#endif
#if SURFACE_NORMALS
        vec3 N_local = surface_local_normal_from_lookup(normal_lookup, transpose(rotation) * normalize(f_normal));
        N = normalize(rotation * N_local);
#else
        N = normalize(f_normal);
#if SURFACE_HORIZONS
        vec3 N_local = transpose(rotation) * N;
#endif
#endif
#if SURFACE_HORIZONS
        if (star_lights_body(f_body_idx, 0u)) {
            vec3 to_star0 = u_scene.stars_pos_radius[0].xyz - f_world_pos;
            float d_au0 = max(length(to_star0), 1e-5);
            vec3 L0 = to_star0 / d_au0;
            float star_ang = asin(clamp(u_scene.stars_pos_radius[0].w / d_au0, 0.0, 1.0));
#if SURFACE_OCEAN
            bool check_horizon = dot(N, L0) > -star_ang || fluid_ocean_active(f_body_idx);
#else
            bool check_horizon = dot(N, L0) > -star_ang;
#endif
            if (check_horizon) {
                terrain_vis = terrain_self_shadow(normal_lookup, surf_dir,
                    max(1.0 - b.pole_oblateness.w, 0.05), N_local, transpose(rotation) * L0,
                    star_ang, SURFACE_OCEAN != 0);
            }
        }
#endif
    }
#else
#if SURFACE_NORMALS || SURFACE_HORIZONS
    mat3 rotation = body_surface_rotation(body_pole(b), b.spin_flags.x);
    SurfaceNormalLookup normal_lookup = surface_normal_lookup_at(f_body_idx, surf_uv01, surf_face, surf_raw_lod);
#endif
#if SURFACE_NORMALS
    vec3 N_local = surface_local_normal_from_lookup(normal_lookup, transpose(rotation) * normalize(f_normal));
    N = normalize(rotation * N_local);
#else
    N = normalize(f_normal);
#if SURFACE_HORIZONS
    vec3 N_local = transpose(rotation) * N;
#endif
#endif
#endif

#if SURFACE_ALBEDO_MAP
    vec4 albedo_sample = surface_albedo_sample_at(f_body_idx, surf_uv01, surf_face, surf_raw_lod, b.color_is_star.rgb);
    vec3 albedo = albedo_sample.rgb;
#else
    vec3 base_albedo = b.color_is_star.rgb;
    float lat_band = sin(f_local_pos.y * 18.0) * 0.04 + cos(f_local_pos.x * 9.0 + f_local_pos.z * 9.0) * 0.03;
    vec3 albedo = clamp(base_albedo * (1.0 + lat_band), 0.0, 1.0);
#endif

#if SURFACE_FULL_LIGHTING
    float emission_cosine = dot(N, V);
#ifdef TERRAIN_SURFACE
    if (geometric_length > 1e-20) emission_cosine = max(emission_cosine, 0.1 * abs(dot(geometric / geometric_length, V)));
#endif
#endif

#if SURFACE_OCEAN
    float ocean = 0.0;
    vec3 water_albedo = vec3(0.0);
    if (fluid_ocean_active(f_body_idx)) {
        SurfaceFluidGPU fluid = fluids[f_body_idx];
        float mode = fluid.ocean_deep.w;
        water_albedo = fluid.ocean_deep.rgb;
        if (mode > 2.5 && mode < 3.5) {
            ocean = 1.0;
        } else {
            float luminance = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
            float blue = (albedo.b - max(albedo.r, albedo.g * 0.85)) / max(albedo.b, 1e-4);
            float albedo_mask = smoothstep(0.05, 0.25, blue) * (1.0 - smoothstep(0.12, 0.3, luminance));
            if (mode > 3.5) {
#if SURFACE_ALBEDO_MAP
                ocean = surface_has_water_mask(f_body_idx) ? albedo_sample.a : albedo_mask;
#else
                ocean = albedo_mask;
#endif
                water_albedo = mix(fluid.ocean_deep.rgb, albedo, 0.5);
            } else {
#ifdef TERRAIN_SURFACE
                if (mode > 1.5) {
                    vec3 pole = body_pole(b);
                    float height_km = length(fluid_spherical(f_body_offset, pole, fluid_flattening(b))) * FLUID_AU_KM
                                    - fluid.ocean_params.z;
                    float tolerance_km = max(0.002, length(f_world_pos) * FLUID_AU_KM * 2e-7);
                    ocean = 1.0 - smoothstep(tolerance_km, 2.0 * tolerance_km, height_km);
                    water_albedo = mix(fluid.ocean_deep.rgb, fluid.ocean_shallow.rgb, albedo_mask * 0.5);
                } else
#endif
                {
                    ocean = albedo_mask;
                    water_albedo = mix(fluid.ocean_deep.rgb, albedo, 0.5);
                }
            }
        }
    }
    bool has_ocean = ocean > 0.0;
#else
    const bool has_ocean = false;
#endif

    vec3 total_light = vec3(0.0);

#if !SURFACE_FULL_LIGHTING
    if (star_lights_body(f_body_idx, 0u)) {
        vec3 star_pos = u_scene.stars_pos_radius[0].xyz;
        float star_r = u_scene.stars_pos_radius[0].w;
        vec3 star_col = u_scene.stars_color_lum[0].rgb;
        float star_lum = u_scene.stars_color_lum[0].w;
        vec3 to_star = star_pos - f_world_pos;
        float d_au = max(length(to_star), 1e-5);
        vec3 L = to_star / d_au;
        float sin_r = star_r / d_au;
        float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0;

#if SURFACE_OCEAN
        OceanSurface water;
        float water_fresnel = 0.0;
        vec3 ocean_up = N;
        if (has_ocean) {
            mat3 rotation = body_surface_rotation(body_pole(b), b.spin_flags.x);
            float q = max(1.0 - b.pole_oblateness.w, 0.05);
#if SURFACE_ALBEDO_MAP || SURFACE_NORMALS || SURFACE_HORIZONS
            vec3 local = surf_dir;
#else
            vec3 local = normalize(f_local_pos);
#endif
            vec3 fixed_normal = normalize(vec3(local.x, local.y / (q * q), local.z));
            ocean_up = normalize(rotation * fixed_normal);
#if SURFACE_OCEAN_FFT
            mat3 fixed_from_world = transpose(rotation) * (FLUID_AU_KM * 1000.0);
            water = ocean_surface(f_body_idx, fixed_normal, fixed_from_world * f_world_pos,
                fixed_from_world * world_dx, fixed_from_world * world_dy, rotation);
#else
            water = ocean_surface(f_body_idx, fixed_normal, vec3(0.0), vec3(0.0), vec3(0.0), rotation);
#endif
            water_fresnel = ocean_fresnel(max(dot(water.normal, V), 0.0), sqrt(water.roughness_squared * 0.5));
            water_albedo = mix(water_albedo * (1.0 - water_fresnel), vec3(0.55), water.foam);
            albedo = mix(albedo, water_albedo, ocean);
            N = normalize(mix(N, ocean_up, ocean));
        }
#endif

#if SURFACE_ATMOSPHERE
        vec3 diffuse;
        vec3 atm_transmission;
        vec3 direct = surface_stellar_light_lambert_small(f_body_idx, f_body_offset, sag_factor,
            N, L, sin_r, b.material.surface.w, has_ocean, diffuse, atm_transmission);
#else
        vec3 diffuse = vec3(0.0);
        vec3 direct = vec3(stellar_lambert_irradiance(clamp(dot(N, L), -1.0, 1.0), clamp(sin_r, 0.0, 1.0)) * b.material.surface.w);
#endif

#if SURFACE_ECLIPSE_SHADOW || SURFACE_RING_SHADOW
        vec3 shadow = vec3(1.0);
        if (any(greaterThan(direct + diffuse, vec3(0.0)))) {
#if SURFACE_ECLIPSE_SHADOW
            shadow = evaluate_eclipse_shadow(f_world_pos, star_pos, star_r, f_body_idx, 0u);
#endif
#if SURFACE_RING_SHADOW
            if (any(greaterThan(shadow, vec3(0.0))) && u_scene.num_ring_hosts > 0u) {
                uint candidates = eclipse_words[u_scene.ring_shadow_params.x + f_body_idx * u_scene.num_stars];
                if (candidates != 0u)
                    shadow *= compute_surface_ring_shadow(f_world_pos, L, sin_r, d_au, candidates, 0u);
            }
#endif
        }
#else
        const vec3 shadow = vec3(1.0);
#endif

#if SURFACE_ATMOSPHERE
        total_light += albedo * star_col * irradiance * diffuse * shadow;
#endif
        if (any(greaterThan(direct, vec3(0.0))) && any(greaterThan(shadow, vec3(0.0)))) {
#if SURFACE_OCEAN
            float direct_vis = mix(terrain_vis.x, terrain_vis.y, ocean);
#else
            float direct_vis = terrain_vis.x;
#endif
            total_light += albedo * star_col * irradiance * direct * shadow * direct_vis;
        }
#if SURFACE_OCEAN
        if (has_ocean) {
#if SURFACE_ATMOSPHERE
            vec3 transmission = atm_transmission;
#else
            vec3 transmission = vec3(smoothstep(-0.01, 0.02, dot(ocean_up, L)));
#endif
            float specular = 3.141592653589793 * ocean_specular(water.normal, V, L,
                water.roughness_squared + 0.5 * sin_r * sin_r) * (1.0 - water.foam);
            vec3 reflected = transmission * specular * terrain_vis.y + water_fresnel * diffuse * (1.0 - water.foam);
            total_light += ocean * star_col * irradiance * shadow * reflected;
        }
#endif
    }
#else
#if SURFACE_OCEAN
    OceanSurface water;
    float water_fresnel = 0.0;
    vec3 ocean_up = N;
    if (has_ocean) {
#if !SURFACE_NORMALS && !SURFACE_HORIZONS
        mat3 rotation = body_surface_rotation(body_pole(b), b.spin_flags.x);
#endif
        float q = max(1.0 - b.pole_oblateness.w, 0.05);
#if SURFACE_ALBEDO_MAP || SURFACE_NORMALS || SURFACE_HORIZONS
        vec3 local = surf_dir;
#else
        vec3 local = normalize(f_local_pos);
#endif
        vec3 fixed_normal = normalize(vec3(local.x, local.y / (q * q), local.z));
        ocean_up = normalize(rotation * fixed_normal);
#if SURFACE_OCEAN_FFT
        mat3 fixed_from_world = transpose(rotation) * (FLUID_AU_KM * 1000.0);
        water = ocean_surface(f_body_idx, fixed_normal, fixed_from_world * f_world_pos,
            fixed_from_world * world_dx, fixed_from_world * world_dy, rotation);
#else
        water = ocean_surface(f_body_idx, fixed_normal, vec3(0.0), vec3(0.0), vec3(0.0), rotation);
#endif
        water_fresnel = ocean_fresnel(max(dot(water.normal, V), 0.0), sqrt(water.roughness_squared * 0.5));
        water_albedo = mix(water_albedo * (1.0 - water_fresnel), vec3(0.55), water.foam);
        albedo = mix(albedo, water_albedo, ocean);
        N = normalize(mix(N, ocean_up, ocean));
    }
#endif

    for (uint s = 0; s < u_scene.num_stars; ++s) {
        if (!star_lights_body(f_body_idx, s)) continue;
        vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
        float star_r = u_scene.stars_pos_radius[s].w;
        vec3 star_col = u_scene.stars_color_lum[s].rgb;
        float star_lum = u_scene.stars_color_lum[s].w;

        vec3 to_star = star_pos - f_world_pos;
        float d_au = max(length(to_star), 1e-5);
        vec3 L = to_star / d_au;
        float sin_r = star_r / d_au;
        vec3 diffuse = vec3(0.0);
#if SURFACE_ATMOSPHERE
        vec3 direct = surface_stellar_light(f_body_idx, f_body_offset, f_base_chord,
            N, V, L, sin_r, emission_cosine, b.material, s, diffuse);
#else
        vec3 direct = (b.material.grain.x < 0.5)
            ? vec3(stellar_lambert_irradiance(clamp(dot(N, L), -1.0, 1.0), clamp(sin_r, 0.0, 1.0)) * b.material.surface.w)
            : vec3(surface_hapke_disk_response(N, V, L, sin_r, b.material, emission_cosine));
#endif
        float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0;

#if SURFACE_ECLIPSE_SHADOW || SURFACE_RING_SHADOW
        vec3 shadow = vec3(1.0);
        if (any(greaterThan(direct + diffuse, vec3(0.0)))) {
#if SURFACE_ECLIPSE_SHADOW
            shadow = evaluate_eclipse_shadow(f_world_pos, star_pos, star_r, f_body_idx, s);
#endif
#if SURFACE_RING_SHADOW
            if (any(greaterThan(shadow, vec3(0.0))) && u_scene.num_ring_hosts > 0u) {
                uint candidates = eclipse_words[u_scene.ring_shadow_params.x + f_body_idx * u_scene.num_stars + s];
                if (candidates != 0u)
                    shadow *= compute_surface_ring_shadow(f_world_pos, L, sin_r, d_au, candidates, uint(s));
            }
#endif
        }
#else
        const vec3 shadow = vec3(1.0);
#endif

#if SURFACE_HORIZONS
        vec2 terrain_vis = vec2(1.0);
        if (any(greaterThan(shadow, vec3(0.0))) && (any(greaterThan(direct, vec3(0.0))) || has_ocean)) {
            terrain_vis = terrain_self_shadow(normal_lookup, surf_dir,
                max(1.0 - b.pole_oblateness.w, 0.05), N_local, transpose(rotation) * L,
                asin(clamp(sin_r, 0.0, 1.0)), has_ocean);
        }
#else
        const vec2 terrain_vis = vec2(1.0);
#endif

#if SURFACE_ATMOSPHERE
        total_light += albedo * star_col * irradiance * diffuse * shadow;
#endif
        if (any(greaterThan(direct, vec3(0.0))) && any(greaterThan(shadow, vec3(0.0)))) {
#if SURFACE_OCEAN
            float direct_vis = mix(terrain_vis.x, terrain_vis.y, ocean);
#else
            float direct_vis = terrain_vis.x;
#endif
            total_light += albedo * star_col * irradiance * direct * shadow * direct_vis;
        }
#if SURFACE_OCEAN
        if (has_ocean) {
            vec3 transmission = vec3(smoothstep(-0.01, 0.02, dot(ocean_up, L)));
#if SURFACE_ATMOSPHERE
            if (surface_atmospheres[f_body_idx].w > 0.5) {
                SurfaceAtmosphereSample lookup = surface_atmosphere_sample(f_body_idx, f_body_offset,
                    f_base_chord, ocean_up);
                transmission = textureLod(surface_atmosphere_lut, surface_atmosphere_coordinates(lookup, L), 0.0).rgb;
            }
#endif
            float specular = 3.141592653589793 * ocean_specular(water.normal, V, L,
                water.roughness_squared + 0.5 * sin_r * sin_r) * (1.0 - water.foam);
            vec3 reflected = transmission * specular * terrain_vis.y + water_fresnel * diffuse * (1.0 - water.foam);
            total_light += ocean * star_col * irradiance * shadow * reflected;
        }
#endif
    }
#endif

#if SURFACE_RINGSHINE
    uint ring_slot = b.surface_flags.z;
    if (u_scene.ringshine_params.x != 0u && ring_slot != 0u) {
        uint r = ring_slot - 1u;
        vec3 pole = ring_hosts[r].pole_max_r.xyz;
        float host_r = ring_hosts[r].tangent_radius.w;
        vec3 rel = f_body_offset;
        float dist2 = dot(rel, rel);
        if (dist2 <= 2.25 * host_r * host_r && dist2 >= 1e-12) {
            vec3 P_eq = rel - pole * dot(rel, pole);
            float len_eq = length(P_eq);
            P_eq = len_eq > 1e-5 ? P_eq / len_eq : vec3(1.0, 0.0, 0.0);
            vec3 P_dir = normalize(rel);
            float slat = dot(P_dir, pole);
#if !SURFACE_FULL_LIGHTING
            if (star_lights_body(f_body_idx, 0u)) {
                vec3 star_pos = u_scene.stars_pos_radius[0].xyz;
                vec3 star_col = u_scene.stars_color_lum[0].rgb;
                float star_lum = u_scene.stars_color_lum[0].w;
                vec3 to_star = star_pos - f_world_pos;
                float d_au = max(length(to_star), 1e-5);
                vec3 L = to_star / d_au;

                vec3 antiL_eq = (-L) - pole * dot(-L, pole);
                float len_anti = length(antiL_eq);
                antiL_eq = len_anti > 1e-5 ? antiL_eq / len_anti : vec3(1.0, 0.0, 0.0);
                float phi = atan(dot(cross(antiL_eq, P_eq), pole), dot(P_eq, antiL_eq));

                vec3 irradiance_map = ringshine_sample(phi, slat, int(r), 0);
                float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0f;
                total_light += albedo * irradiance_map * star_col * irradiance / 3.14159265358979323846;
            }
#else
            vec3 ringshine_light = vec3(0.0);
            for (uint s = 0u; s < u_scene.num_stars; ++s) {
                if (!star_lights_body(f_body_idx, s)) continue;
                vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
                vec3 star_col = u_scene.stars_color_lum[s].rgb;
                float star_lum = u_scene.stars_color_lum[s].w;
                vec3 to_star = star_pos - f_world_pos;
                float d_au = max(length(to_star), 1e-5);
                vec3 L = to_star / d_au;

                vec3 antiL_eq = (-L) - pole * dot(-L, pole);
                float len_anti = length(antiL_eq);
                antiL_eq = len_anti > 1e-5 ? antiL_eq / len_anti : vec3(1.0, 0.0, 0.0);
                float phi = atan(dot(cross(antiL_eq, P_eq), pole), dot(P_eq, antiL_eq));

                vec3 irradiance_map = ringshine_sample(phi, slat, int(r), int(s));
                float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0f;
                ringshine_light += albedo * irradiance_map * star_col * irradiance / 3.14159265358979323846;
            }
            total_light += ringshine_light;
#endif
        }
    }
#endif

#if SURFACE_PLANETSHINE || SURFACE_FULL_LIGHTING
    uint shine_layer = uint(b.shine_irradiance[0].w);
    vec3 surface_up = normalize(f_body_offset);
#if SURFACE_PLANETSHINE
    if (shine_layer > 0u) total_light += albedo * planetshine_map_sample(surface_up, shine_layer - 1u);
#endif
#if SURFACE_FULL_LIGHTING
    float shine_floor = dot(total_light, vec3(0.2126, 0.7152, 0.0722));
    for (int i = 0; i < 4 && shine_layer == 0u; ++i) {
        vec3 light = b.shine_direction[i].xyz;
        if (!any(greaterThan(b.shine_irradiance[i].rgb, vec3(0.0)))
            || (dot(N, light) < 0.0 && dot(surface_up, light) < -0.05)
            || 4000.0 * dot(albedo * b.shine_irradiance[i].rgb, vec3(0.2126, 0.7152, 0.0722)) <= shine_floor) continue;
        vec3 diffuse;
        vec3 direct = surface_planetshine_light(f_body_idx, f_body_offset, f_base_chord,
            N, V, light, sin(b.shine_direction[i].w), emission_cosine, b.material, diffuse);
        total_light += albedo * b.shine_irradiance[i].rgb * (direct + diffuse);
        shine_floor = dot(total_light, vec3(0.2126, 0.7152, 0.0722));
    }
#endif
#endif

    vec3 rgb = total_light * exposure * mesh_weight;
    out_color = vec4(rgb, 1.0);
#endif
}
