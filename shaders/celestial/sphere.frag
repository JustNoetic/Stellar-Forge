#version 460

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
#include "common/eclipse_shadow.glsl"
#include "common/surface_textures.glsl"
#include "common/surface_ring_shadow.glsl"
#include "common/ringshine_lookup.glsl"
#include "common/planetshine_lookup.glsl"
#include "common/surface_atmosphere.glsl"
#include "common/fluid_surface.glsl"

layout(location = 0) out vec4 out_color;
layout(depth_greater) out float gl_FragDepth;

void main() {
    BodyInstanceGPU b = bodies[f_body_idx];
    gl_FragDepth = curved_surface_depth(b, f_world_pos, f_body_offset, f_apparent_pos, f_base_chord, gl_FragCoord.z);
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
    float emission_cosine = dot(N, V);
#ifdef TERRAIN_SURFACE
    vec3 geometric = cross(dFdx(f_world_pos), dFdy(f_world_pos));
    float geometric_length = length(geometric);
    if (geometric_length > 1e-20) emission_cosine = max(emission_cosine, 0.1*abs(dot(geometric/geometric_length, V)));
#endif
    float exposure = (u_scene.hdr_enabled != 0u) ? u_scene.screen_params.w : 1.0;
    float footprint_m = max(length(dFdx(f_world_pos)), length(dFdy(f_world_pos)))*FLUID_AU_KM*1000.0;

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

    float ocean = 0.0;
    OceanSurface water;
    float water_fresnel = 0.0;
    vec3 ocean_up = normalize(f_normal);
    if (fluid_ocean_active(f_body_idx)) {
        SurfaceFluidGPU fluid = fluids[f_body_idx];
        vec3 pole = body_pole(b);
        float mode = fluid.ocean_deep.w;
        float height_km = length(fluid_spherical(f_body_offset, pole, fluid_flattening(b)))*FLUID_AU_KM
                        -fluid.ocean_params.z;
        float luminance = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
        float blue = (albedo.b-max(albedo.r, albedo.g*0.85))/max(albedo.b, 1e-4);
        float albedo_mask = smoothstep(0.05, 0.25, blue)*(1.0-smoothstep(0.12, 0.3, luminance));
        vec3 water_albedo = fluid.ocean_deep.rgb;
        if (mode > 2.5) ocean = 1.0;
#ifdef TERRAIN_SURFACE
        else if (mode > 1.5) {
            float tolerance_km = max(0.002, length(f_world_pos)*FLUID_AU_KM*2e-7);
            ocean = 1.0-smoothstep(tolerance_km, 2.0*tolerance_km, height_km);
            water_albedo = mix(fluid.ocean_deep.rgb, fluid.ocean_shallow.rgb, albedo_mask*0.5);
        }
#endif
        else {
            ocean = albedo_mask;
            water_albedo = mix(fluid.ocean_deep.rgb, albedo, 0.5);
        }
        if (ocean > 0.0) {
            float q = max(1.0-b.pole_oblateness.w, 0.05);
            vec3 local = normalize(f_local_pos);
            vec3 fixed_normal = normalize(vec3(local.x, local.y/(q*q), local.z));
            mat3 rotation = body_surface_rotation(pole, b.spin_flags.x);
            ocean_up = normalize(rotation*fixed_normal);
            vec3 fixed_from_camera_m = transpose(rotation)*f_world_pos*(FLUID_AU_KM*1000.0);
            water = ocean_surface(f_body_idx, fixed_normal, fixed_from_camera_m, footprint_m, rotation);
            water_fresnel = ocean_fresnel(max(dot(water.normal, V), 0.0), sqrt(water.roughness_squared*0.5));
            water_albedo = mix(water_albedo*(1.0-water_fresnel), vec3(0.55), water.foam);
            albedo = mix(albedo, water_albedo, ocean);
        }
    }

    vec3 total_light = vec3(0.0);
    for (uint s = 0; s < u_scene.num_stars; ++s) {
        if (!star_lights_body(f_body_idx, s)) continue;
        vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
        float star_r = u_scene.stars_pos_radius[s].w;
        vec3 star_col = u_scene.stars_color_lum[s].rgb;
        float star_lum = u_scene.stars_color_lum[s].w;

        vec3 to_star = star_pos - f_world_pos;
        float d_au = max(length(to_star), 1e-5);
        vec3 L = to_star / d_au;
        vec3 diffuse;
        vec3 direct = surface_stellar_light(f_body_idx, f_body_offset, f_base_chord,
            N, V, L, star_r/d_au, emission_cosine, b.material, s, diffuse);
        float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0;

        vec3 shadow = vec3(1.0);
        if (any(greaterThan(direct+diffuse, vec3(0.0)))) {
            shadow = evaluate_eclipse_shadow(f_world_pos, star_pos, star_r, f_body_idx, s);
            if (any(greaterThan(shadow, vec3(0.0))) && u_scene.num_ring_hosts > 0u) {
                float sun_ang_rad = star_r / d_au;
                uint candidates = u_scene.ring_shadow_params.y != 0u
                    ? (1u << u_scene.num_ring_hosts) - 1u
                    : eclipse_words[u_scene.ring_shadow_params.x + f_body_idx * u_scene.num_stars + s];
                if (candidates != 0u)
                    shadow *= compute_surface_ring_shadow(f_world_pos, L, sun_ang_rad, d_au, candidates);
            }
        }

        total_light += albedo * star_col * irradiance * diffuse * shadow;

        if (any(greaterThan(direct, vec3(0.0))) && any(greaterThan(shadow, vec3(0.0)))) {
            total_light += albedo * star_col * irradiance * direct * shadow;
        }

        if (ocean > 0.0) {
            vec3 transmission = vec3(smoothstep(-0.01, 0.02, dot(ocean_up, L)));
            if (surface_atmospheres[f_body_idx].w > 0.5) {
                SurfaceAtmosphereSample lookup = surface_atmosphere_sample(f_body_idx, f_body_offset,
                    f_base_chord, ocean_up);
                transmission = textureLod(surface_atmosphere_lut, surface_atmosphere_coordinates(lookup, L), 0.0).rgb;
            }
            float sun_radius = star_r/d_au;
            float specular = 3.141592653589793*ocean_specular(water.normal, V, L,
                water.roughness_squared+0.5*sun_radius*sun_radius)*(1.0-water.foam);
            vec3 reflected = transmission*specular+water_fresnel*diffuse*(1.0-water.foam);
            total_light += ocean*star_col*irradiance*shadow*reflected;
        }
    }

    if (u_scene.ringshine_params.x != 0u) {
        vec3 ringshine_light = vec3(0.0);
        for (uint r = 0; r < u_scene.ringshine_params.y; ++r) {
            if (ring_hosts[r].counts_indices.x != f_body_idx) continue;
            vec3 pole = ring_hosts[r].pole_max_r.xyz;
            float host_r = ring_hosts[r].tangent_radius.w;
            vec3 rel = f_body_offset;
            float dist2 = dot(rel, rel);
            if (dist2 > 2.25 * host_r * host_r || dist2 < 1e-12) continue;

            vec3 P_eq = rel - pole * dot(rel, pole);
            float len_eq = length(P_eq);
            P_eq = len_eq > 1e-5 ? P_eq / len_eq : vec3(1.0, 0.0, 0.0);
            for (uint s = 0; s < u_scene.num_stars; ++s) {
                if (!star_lights_body(f_body_idx, s)) continue;
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

    uint shine_layer = uint(b.shine_irradiance[0].w);
    vec3 surface_up = normalize(f_body_offset);
    if (shine_layer > 0u) total_light += albedo*planetshine_map_sample(surface_up, shine_layer-1u);
    // Reflected sources below both horizons only add their own twilight, and a source
    // whose contribution at four times its unit-response bound stays under 0.1% of the
    // light already here cannot change the pixel.
    float shine_floor = dot(total_light, vec3(0.2126, 0.7152, 0.0722));
    for (int i = 0; i < 4 && shine_layer == 0u; ++i) {
        vec3 light = b.shine_direction[i].xyz;
        if (!any(greaterThan(b.shine_irradiance[i].rgb, vec3(0.0)))
            || (dot(N, light) < 0.0 && dot(surface_up, light) < -0.05)
            || 4000.0*dot(albedo*b.shine_irradiance[i].rgb, vec3(0.2126, 0.7152, 0.0722)) <= shine_floor) continue;
        vec3 diffuse;
        vec3 direct = surface_stellar_light(f_body_idx, f_body_offset, f_base_chord,
            N, V, light, sin(b.shine_direction[i].w), emission_cosine, b.material, 16u, diffuse);
        total_light += albedo*b.shine_irradiance[i].rgb
                     *(direct+diffuse);
    }

    vec3 rgb = total_light * exposure * mesh_weight;
    out_color = vec4(rgb, 1.0);
}
