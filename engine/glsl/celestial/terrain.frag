#version 460 core
in vec3 f_world_pos;
in vec3 f_normal;
in vec3 f_patch_pos_km;
flat in float f_height_slot;
in vec2 f_tile_uv;
in vec2 f_local_uv;
flat in float f_tile_slot;
flat in float f_lod_level;
in float f_clip_z;

flat in vec3 f_atmo_tint;
flat in float f_atmo_h;
flat in float f_scale_height;
flat in vec3 f_o3_tau;
flat in vec2 f_o3_layer;
flat in float f_radius_km;

flat in uvec2 f_caster_mask;
flat in uint f_ring_mask;
flat in vec3 f_planetshine_dir;
flat in vec3 f_planetshine_color;
flat in vec3 f_body_center;
in vec3 f_rel_pos;

#define MAX_STARS 16
#define MAX_CASTERS 64
#define MAX_RING_PLANES 16
#define PI 3.14159265358979323846

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
    vec4 u_caster_ozone[MAX_CASTERS];
    vec4 u_caster_ozone_vert[MAX_CASTERS];
};

uniform sampler2DArray u_tile_array;
uniform bool u_debug_tiles;
uniform bool u_hdr_enabled;
uniform float u_exposure;
uniform vec3 u_camera_pos;
uniform bool u_is_cloud_pass;

uniform int u_num_ring_planes;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec4 u_ring_params[MAX_RING_PLANES];
uniform uint u_ring_coplanar_mask[MAX_RING_PLANES];
uniform float u_caster_max_bend[64];
uniform float u_au_to_km;

uniform sampler2D u_ring_gradients;
uniform sampler2D u_ringshine_map;

uniform bool u_planetshine_enabled;
uniform bool u_ringshine_enabled;

#include "common/surface_ring_shadow.glsl"

out vec4 out_color;

float get_oblate_radius(float r_eq, float r_minor, vec3 pole, vec3 L, vec3 perp_vec) {
    if (r_minor < 1e-6 || r_eq < 1e-6) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / r_minor) * (y / r_minor));
    if (denom < 1e-6) return r_eq;
    return perp_len / denom;
}

vec3 casterShadowTerm(float alpha, float beta, float gamma,
                      float penumbra_outer, float penumbra_inner,
                      float max_bend, vec4 atmo_param, vec4 ozone_param,
                      float scale_height_km, float dist_to_caster) {
    float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
    float occ = clamp(max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma), 0.0, 1.0);
    float geom_sh = pow(clamp(1.0 - occ, 0.0, 1.0), 1.0 / 1.6);
    vec3 sh = vec3(geom_sh);

    if (max_bend > 1e-6 && gamma < penumbra_outer) {
        float H_scale = max(scale_height_km, 0.1);
        float sigma_z = max(0.707 * H_scale, 0.1);
        float z_peak = ozone_param.w;
        float dist_km = max(dist_to_caster * u_au_to_km, 1e-6);

        float caster_r_km = max(beta * dist_km, 100.0);
        float grazing_factor = sqrt(2.0 * PI * caster_r_km / max(H_scale, 1e-3));
        vec3 tau_grazing_0 = atmo_param.xyz * grazing_factor;

        float req_bend = beta - gamma - alpha * 0.8;

        if (req_bend <= max_bend) {
            float atmo_depth = clamp(max(0.0, req_bend) / max_bend, 0.0, 1.0);
            float z_km = -H_scale * log(max(atmo_depth, 1e-5));
            vec3 tau_R = tau_grazing_0 * atmo_depth;
            float z_diff = (z_km - z_peak) / sigma_z;
            vec3 tau_O3 = ozone_param.xyz * exp(-0.5 * z_diff * z_diff);
            vec3 tau_total = tau_R + tau_O3;
            vec3 atmo_transmittance = exp(-tau_total);

            float radial_defocus = 1.0 / (1.0 + (dist_km * max(req_bend, 1e-6)) / H_scale);
            float f_focal_km = caster_r_km / max(max_bend, 1e-6);
            float focal_ratio = clamp(dist_km / f_focal_km, 0.0, 1.0);
            float off_axis_factor = clamp(alpha / max(1e-6, gamma), 0.0, 1.0);
            float annular_factor = (2.0 * H_scale) / max(alpha * dist_km, 1e-9);
            float ring_intensity = clamp(annular_factor * focal_ratio * off_axis_factor, 0.0, 1.0) * radial_defocus;
            float body_surface_fade = smoothstep(1.0, 0.75, atmo_depth);
            float refraction_intensity = ring_intensity * (1.0 - atmo_depth * 0.7) * body_surface_fade;
            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            sh += atmo_transmittance * refraction_intensity * atmo_blend;
        }
    }
    return clamp(sh, vec3(0.0), vec3(1.0));
}

void main() {
    if (f_tile_slot < 0.0) {
        discard;
    }

    // Write logarithmic depth matching sphere and orbit shaders
    gl_FragDepth = log2(max(1e-6, 1.0 + f_clip_z * u_depth_C)) / log2(u_far * u_depth_C + 1.0);

    // Sample from the tiled Texture2DArray
    vec3 tile_coord = vec3(clamp(f_tile_uv, 0.0, 1.0), f_tile_slot);
    vec4 tex_sample = texture(u_tile_array, tile_coord);

    vec3 N = normalize(f_normal);
    if (!u_is_cloud_pass && f_height_slot >= 0.0) {
        // Match the physical triangle before refraction/lensing. Patch-relative
        // km positions keep sub-pixel differences precise near the ground;
        // subtracting the anchor here, AFTER interpolation, would be too late.
        vec3 dx = dFdx(f_patch_pos_km);
        vec3 dy = dFdy(f_patch_pos_km);
        if (dot(dx, dx) > 0.0 && dot(dy, dy) > 0.0) {
            vec3 face = cross(normalize(dx), normalize(dy));
            if (dot(face, face) > 1e-12) {
                face = normalize(face);
                N = dot(face, N) < 0.0 ? -face : face;
            }
        }
    }
    vec3 V = normalize(u_camera_pos - f_world_pos);

    // =========================================================================
    // CLOUD PASS (Transparent Spherified Cube Quadtree Shell)
    // =========================================================================
    if (u_is_cloud_pass) {
        // Cloud alpha / opacity:
        // Use alpha channel if available; if alpha is uniformly 1.0 (raw RGB), fallback to red channel
        float cloud_alpha = tex_sample.a;
        if (cloud_alpha > 0.999 && (tex_sample.r < 0.99 || tex_sample.g < 0.99)) {
            cloud_alpha = tex_sample.r;
        }

        if (cloud_alpha < 0.02) {
            discard;
        }

        // Two-sided normal handling for orbit (outside) vs ground (inside looking up)
        bool is_underside = dot(N, V) < 0.0;
        vec3 cloud_norm = is_underside ? -N : N;

        // Smooth limb falloff to avoid harsh polygon horizon silhouette
        float NdotV = abs(dot(N, V));
        float limb_soften = smoothstep(0.0, 0.015, NdotV);
        cloud_alpha *= limb_soften;

        vec3 cloud_albedo = vec3(0.95);
        vec3 total_cloud_diffuse = vec3(0.0);
        vec3 primary_caster_shadow = vec3(1.0);
        int num_stars = min(u_num_stars, MAX_STARS);

        for (int s = 0; s < num_stars; ++s) {
            vec3 star_pos = u_stars_pos_radius[s].xyz;
            float star_radius = u_stars_pos_radius[s].w;
            vec3 frag_to_star = (star_pos - f_body_center) - f_rel_pos;
            float dist_to_star = length(frag_to_star);
            if (dist_to_star < 1e-5) continue;
            vec3 L = frag_to_star / dist_to_star;

            float star_ang_radius = star_radius / dist_to_star;
            float sin_alpha = clamp(star_ang_radius, 0.0, 1.0);

            // Cloud elevation geometric terminator offset
            float cloud_h_km = max(0.5, f_scale_height * 0.35);
            float R_km = max(f_radius_km, 1e-3);
            float term_offset = sqrt(max(0.0, 2.0 * cloud_h_km / R_km));
            // Solar illumination always strikes the outer (top) surface of the cloud deck
            float sun_cos = dot(N, L);
            float effective_sun_cos = sun_cos + term_offset;
            if (effective_sun_cos <= -sin_alpha) continue;

            float diffuse = 0.0;
            if (is_underside) {
                // Underside of cloud illuminated by transmitted sunlight through cloud slab
                float top_illum = clamp((effective_sun_cos + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);
                float cos_trans = -dot(V, L);
                float forward_trans = pow(max(0.0, cos_trans), 4.0) * 0.35;
                diffuse = top_illum * (0.50 + forward_trans);
            } else {
                float sun_vis = clamp((effective_sun_cos + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);
                diffuse = sun_vis;
                // Forward scattering / silver lining along backlit crescent rim
                float cos_forward = -dot(V, L);
                float silver_lining = pow(max(0.0, cos_forward), 6.0) * 0.35 * sun_vis;
                diffuse += silver_lining;
            }

            // Direct sunlight extinction at cloud altitude
            vec3 incoming_tint = vec3(1.0);
            if (f_atmo_h > 0.0) {
                float mu = max(effective_sun_cos, 0.0);
                float H_scale = max(f_scale_height, 1e-3);
                float am = (sqrt(R_km * R_km * mu * mu + 2.0 * R_km * H_scale + H_scale * H_scale) - R_km * mu) / H_scale;
                vec3 tau_at_cloud = max(f_atmo_tint, vec3(0.0)) * exp(-cloud_h_km / H_scale);
                incoming_tint = exp(-tau_at_cloud * am);
            }

            float star_lum = u_stars_colors[s].w;
            float falloff = star_lum / max(1e-8, dist_to_star * dist_to_star);
            vec3 star_color = u_stars_colors[s].rgb;

            // === Analytical eclipse shadows (casters: moons and planets) ===
            vec3 shadow = vec3(1.0);
            float star_radius_over_dist = star_radius / dist_to_star;
            for (int j = 0; j < u_num_casters; j++) {
                if (dot(shadow, shadow) < 0.001) break;

                if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
                else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }

                vec3 caster_pos = u_casters[j].xyz;
                float caster_r = u_casters[j].w;
                float atmo_h = u_caster_atmos[j].w;

                vec3 frag_to_caster = (caster_pos - f_body_center) - f_rel_pos;
                float t_proj = dot(frag_to_caster, L);
                if (t_proj < 0.0) continue;

                float dist_sq = dot(frag_to_caster, frag_to_caster);
                if (dist_sq < caster_r * caster_r * 1.0404) continue; // Skip self

                float dist_to_caster = sqrt(dist_sq);
                vec3 cross_vec = cross(frag_to_caster, L);
                float perp_sq = dot(cross_vec, cross_vec);

                float atmo_h_au = atmo_h / u_au_to_km;
                float max_effective_r = caster_r + (atmo_h > 0.0 ? atmo_h_au * 4.0 : 0.0);
                float max_r_penumbra = max_effective_r + dist_to_caster * star_radius_over_dist;
                if (perp_sq > max_r_penumbra * max_r_penumbra) continue;

                float caster_r_minor = u_caster_poles_obl[j].w;
                vec3 perp_vec = frag_to_caster - t_proj * L;
                if (caster_r_minor < caster_r - 1e-5) {
                    vec3 pole = u_caster_poles_obl[j].xyz;
                    caster_r = get_oblate_radius(caster_r, caster_r_minor, pole, L, perp_vec);
                }

                float directional_star_r = star_radius;
                float star_r_minor = u_stars_poles_obl[s].w;
                if (star_r_minor < star_radius - 1e-5) {
                    vec3 star_pole = u_stars_poles_obl[s].xyz;
                    directional_star_r = get_oblate_radius(star_radius, star_r_minor, star_pole, L, perp_vec);
                }
                float local_star_radius_over_dist = directional_star_r / dist_to_star;

                float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h_au * 4.0 : 0.0);
                float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
                if (perp_sq > r_penumbra * r_penumbra) continue;

                float inv_dist = 1.0 / dist_to_caster;
                float alpha = local_star_radius_over_dist;
                float beta = caster_r * inv_dist;
                float gamma = sqrt(perp_sq) * inv_dist;
                float penumbra_outer = alpha + beta;
                float penumbra_inner = abs(beta - alpha);
                shadow *= casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, u_caster_max_bend[j], u_caster_atmos[j], u_caster_ozone[j], u_caster_colors[j].w, dist_to_caster);
            }

            if (s == 0) {
                primary_caster_shadow = shadow;
            }

            // === Ring shadow on clouds ===
            if (effective_sun_cos > -sin_alpha) {
                shadow *= surface_ring_shadow(f_body_center, f_rel_pos, L,
                    dist_to_star, star_ang_radius, f_ring_mask);
            }

            total_cloud_diffuse += star_color * incoming_tint * (diffuse * falloff) * shadow;

            if (f_atmo_h > 0.0) {
                float sun_elev = clamp(effective_sun_cos, 0.0, 1.0);
                vec3 atmo_sky_ambient = f_atmo_tint * (0.08 * sun_elev);
                total_cloud_diffuse += star_color * atmo_sky_ambient * shadow * falloff;
            }
        }

        // === Moonshine / Planetshine on Clouds ===
        vec3 bounce_light = vec3(0.0);
        if (dot(f_planetshine_color, f_planetshine_color) > 1e-12) {
            float NdotC = max(0.0, dot(N, f_planetshine_dir));
            float ps_trans = is_underside ? 0.50 : 1.0;
            bounce_light = f_planetshine_color * (NdotC * ps_trans) * primary_caster_shadow;
        }

        // === Ringshine on Clouds ===
        vec3 ring_shine = vec3(0.0);
        if (u_ringshine_enabled) {
            for (int k = 0; k < u_num_ring_planes; k++) {
                vec3 ring_center = u_ring_center[k];
                vec3 ring_normal = u_ring_normal[k];
                float inner_r = u_ring_params[k].x;
                float outer_r = u_ring_params[k].y;

                vec3 frag_to_center = (ring_center - f_body_center) - f_rel_pos;
                float dist_to_center = length(frag_to_center);

                if (dist_to_center > outer_r * 20.0 || dist_to_center < 1e-6) continue;

                int host_caster_idx = -1;
                for (int j = 0; j < u_num_casters; j++) {
                    if (distance(u_casters[j].xyz, ring_center) < 1e-7) {
                        host_caster_idx = j;
                        break;
                    }
                }

                bool is_host_planet = false;
                if (host_caster_idx >= 0) {
                    float host_radius = u_casters[host_caster_idx].w;
                    is_host_planet = (length((u_casters[host_caster_idx].xyz - f_body_center) - f_rel_pos) < host_radius * 1.5);
                } else {
                    is_host_planet = (dist_to_center < inner_r);
                }

                if (!is_host_planet) continue;

                vec3 P_dir = normalize(f_rel_pos);
                float frag_elevation = dot(P_dir, ring_normal);

                vec3 frag_to_star0 = (u_stars_pos_radius[0].xyz - f_body_center) - f_rel_pos;
                float dist_to_star0 = length(frag_to_star0);
                vec3 L0 = dist_to_star0 > 1e-5 ? (frag_to_star0 / dist_to_star0) : vec3(0.0, 1.0, 0.0);
                float sun_elev_0 = dot(L0, ring_normal);

                for (int s = 0; s < num_stars; s++) {
                    vec3 star_pos = u_stars_pos_radius[s].xyz;
                    float star_radius = u_stars_pos_radius[s].w;
                    vec3 frag_to_star = (star_pos - f_body_center) - f_rel_pos;
                    float dist_to_star = length(frag_to_star);
                    if (dist_to_star < 1e-5) continue;
                    vec3 L = frag_to_star / dist_to_star;
                    float sun_elev_s = dot(L, ring_normal);
                    float rel_hemi = (sun_elev_s * sun_elev_0 >= 0.0) ? 1.0 : -1.0;

                    vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
                    float star_sin_lat = abs(dot(L, pole_dir));

                    vec3 eq_color = u_stars_colors[s].rgb;
                    float eq_lum = u_stars_colors[s].a;
                    vec3 pole_color = u_stars_pole_colors[s].rgb;
                    float pole_lum = u_stars_pole_colors[s].a;

                    vec3 star_color = mix(eq_color, pole_color, star_sin_lat);
                    float star_lum = mix(eq_lum, pole_lum, star_sin_lat);

                    vec3 P_eq_raw = P_dir - ring_normal * dot(P_dir, ring_normal);
                    float len_P_eq = length(P_eq_raw);
                    vec3 P_eq = len_P_eq > 1e-5 ? P_eq_raw / len_P_eq : vec3(1.0, 0.0, 0.0);

                    vec3 antiL = -L;
                    vec3 antiL_eq_raw = antiL - ring_normal * dot(antiL, ring_normal);
                    float len_antiL_eq = length(antiL_eq_raw);
                    vec3 antiL_eq = len_antiL_eq > 1e-5 ? antiL_eq_raw / len_antiL_eq : vec3(-1.0, 0.0, 0.0);

                    vec3 cross_rel = cross(antiL_eq, P_eq);
                    float sin_rel = dot(cross_rel, ring_normal);
                    float cos_rel = clamp(dot(P_eq, antiL_eq), -1.0, 1.0);
                    float phi_center = atan(sin_rel, cos_rel);

                    float x_prime = phi_center / PI;
                    float phi_uv = sign(x_prime) * pow(abs(x_prime), 0.666666667) * 0.5 + 0.5;

                    float y_prime = frag_elevation * rel_hemi;
                    float elev_uv = sign(y_prime) * pow(abs(y_prime), 0.666666667) * 0.5 + 0.5;

                    vec2 map_uv = vec2(phi_uv, (float(k) + elev_uv) / 16.0);
                    vec3 total_ring_irradiance = texture(u_ringshine_map, map_uv).rgb;

                    float shine_intensity = 0.318309886; // 1 / PI
                    vec3 ring_tint = total_ring_irradiance;
                    float irradiance = u_hdr_enabled ? (star_lum / max(dist_to_star * dist_to_star, 1e-8)) : 1.0;

                    ring_shine += ring_tint * star_color * shine_intensity * irradiance;
                }
            }
        }

        vec3 cloud_final_color = cloud_albedo * total_cloud_diffuse;
        if (u_planetshine_enabled || u_ringshine_enabled) {
            cloud_final_color += cloud_albedo * bounce_light;
        }
        if (u_ringshine_enabled) {
            cloud_final_color += cloud_albedo * ring_shine;
        }
        if (u_hdr_enabled) {
            cloud_final_color *= u_exposure;
        }

        out_color = vec4(cloud_final_color, cloud_alpha);
        return;
    }

    // =========================================================================
    // TERRAIN SURFACE PASS
    // =========================================================================
    // Decode sRGB to Linear
    vec3 albedo = pow(tex_sample.rgb, vec3(2.2));

    // Accumulate direct illumination from all stars in SceneData
    vec3 total_diffuse = vec3(0.0);
    vec3 primary_caster_shadow = vec3(1.0);
    int num_stars = min(u_num_stars, MAX_STARS);
    for (int s = 0; s < num_stars; ++s) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        vec3 frag_to_star = (star_pos - f_body_center) - f_rel_pos;
        float dist_to_star = length(frag_to_star);
        if (dist_to_star < 1e-5) continue;
        vec3 L = frag_to_star / dist_to_star;

        // Angular radius of star disc for soft terminator penumbra
        float star_ang_radius = star_radius / dist_to_star;
        float sin_alpha = clamp(star_ang_radius, 0.0, 1.0);

        // Physical Lambert cosine law with penumbra transition
        float sun_cos = dot(N, L);
        if (sun_cos <= -sin_alpha) continue;
        float diffuse = clamp((sun_cos + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);

        vec3 incoming_light_tint = vec3(1.0);
        vec3 direct_light_tint = vec3(1.0);
        vec3 diffuse_light_tint = vec3(0.0);

        // Atmosphere light extinction on ground (Rayleigh + Mie + Ozone)
        if (f_atmo_h > 0.0) {
            float R_km = max(f_radius_km, 1e-3);
            float H_scale = max(f_scale_height, 1e-3);
            float mu = max(sun_cos, 0.0);

            // Geometric relative air mass (Rozenberg curved-atmosphere formulation)
            float am = (sqrt(R_km * R_km * mu * mu + 2.0 * R_km * H_scale + H_scale * H_scale) - R_km * mu) / H_scale;

            vec3 tau_vertical = max(f_atmo_tint, vec3(0.0));
            vec3 tau_direct = tau_vertical * am;

            // Ozone absorption (Chappuis band) absorbs orange/red at high air masses
            if (dot(f_o3_tau, f_o3_tau) > 0.0) {
                float o3_w = f_o3_layer.y;
                float R_o3 = R_km + max(f_o3_layer.x, 0.0);
                float am_o3 = (sqrt(R_o3 * R_o3 * mu * mu + 2.0 * R_o3 * o3_w + o3_w * o3_w) - R_o3 * mu) / o3_w;
                tau_direct += f_o3_tau * am_o3;
            }

            direct_light_tint = exp(-tau_direct);

            // Downward diffuse daylight (skylight dome illumination)
            vec3 tau_slant = tau_vertical * am;
            vec3 diffuse_sky = (vec3(1.0) - exp(-tau_slant)) * (max(0.0, mu) / (vec3(1.0) + 0.75 * tau_vertical));
            diffuse_light_tint = diffuse_sky;

            incoming_light_tint = direct_light_tint + diffuse_light_tint;
        }

        float star_lum = u_stars_colors[s].w;
        float falloff = star_lum / max(1e-8, dist_to_star * dist_to_star);
        vec3 star_color = u_stars_colors[s].rgb;

        // === Analytical eclipse shadows (casters: moons and planets) ===
        vec3 shadow = vec3(1.0);
        float star_radius_over_dist = star_radius / dist_to_star;
        for (int j = 0; j < u_num_casters; j++) {
            if (dot(shadow, shadow) < 0.001) break;

            if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
            else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }

            vec3 caster_pos = u_casters[j].xyz;
            float caster_r = u_casters[j].w;
            float atmo_h = u_caster_atmos[j].w;

            vec3 frag_to_caster = (caster_pos - f_body_center) - f_rel_pos;
            float t_proj = dot(frag_to_caster, L);
            if (t_proj < 0.0) continue;

            float dist_sq = dot(frag_to_caster, frag_to_caster);
            if (dist_sq < caster_r * caster_r * 1.0404) continue; // Skip self

            float dist_to_caster = sqrt(dist_sq);
            vec3 cross_vec = cross(frag_to_caster, L);
            float perp_sq = dot(cross_vec, cross_vec);

            float atmo_h_au = atmo_h / u_au_to_km;
            float max_effective_r = caster_r + (atmo_h > 0.0 ? atmo_h_au * 4.0 : 0.0);
            float max_r_penumbra = max_effective_r + dist_to_caster * star_radius_over_dist;
            if (perp_sq > max_r_penumbra * max_r_penumbra) continue;

            float caster_r_minor = u_caster_poles_obl[j].w;
            vec3 perp_vec = frag_to_caster - t_proj * L;
            if (caster_r_minor < caster_r - 1e-5) {
                vec3 pole = u_caster_poles_obl[j].xyz;
                caster_r = get_oblate_radius(caster_r, caster_r_minor, pole, L, perp_vec);
            }

            float directional_star_r = star_radius;
            float star_r_minor = u_stars_poles_obl[s].w;
            if (star_r_minor < star_radius - 1e-5) {
                vec3 star_pole = u_stars_poles_obl[s].xyz;
                directional_star_r = get_oblate_radius(star_radius, star_r_minor, star_pole, L, perp_vec);
            }
            float local_star_radius_over_dist = directional_star_r / dist_to_star;

            float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h_au * 4.0 : 0.0);
            float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
            if (perp_sq > r_penumbra * r_penumbra) continue;

            float inv_dist = 1.0 / dist_to_caster;
            float alpha = local_star_radius_over_dist;
            float beta = caster_r * inv_dist;
            float gamma = sqrt(perp_sq) * inv_dist;
            float penumbra_outer = alpha + beta;
            float penumbra_inner = abs(beta - alpha);
            shadow *= casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, u_caster_max_bend[j], u_caster_atmos[j], u_caster_ozone[j], u_caster_colors[j].w, dist_to_caster);
        }

        if (s == 0) {
            primary_caster_shadow = shadow;
        }

        // === Ring shadow on terrain ===
        if (sun_cos > -sin_alpha) {
            shadow *= surface_ring_shadow(f_body_center, f_rel_pos, L,
                dist_to_star, star_ang_radius, f_ring_mask);
        }

        total_diffuse += star_color * incoming_light_tint * (diffuse * falloff) * shadow;
    }

    // === Moonshine / Planetshine ===
    vec3 bounce_light = vec3(0.0);
    if (dot(f_planetshine_color, f_planetshine_color) > 1e-12) {
        float NdotC = max(0.0, dot(N, f_planetshine_dir));
        bounce_light = f_planetshine_color * NdotC * primary_caster_shadow;
    }

    // === Ringshine ===
    vec3 ring_shine = vec3(0.0);
    if (u_ringshine_enabled) {
        for (int k = 0; k < u_num_ring_planes; k++) {
            vec3 ring_center = u_ring_center[k];
            vec3 ring_normal = u_ring_normal[k];
            float inner_r = u_ring_params[k].x;
            float outer_r = u_ring_params[k].y;

            vec3 frag_to_center = (ring_center - f_body_center) - f_rel_pos;
            float dist_to_center = length(frag_to_center);

            if (dist_to_center > outer_r * 20.0 || dist_to_center < 1e-6) continue;

            int host_caster_idx = -1;
            for (int j = 0; j < u_num_casters; j++) {
                if (distance(u_casters[j].xyz, ring_center) < 1e-7) {
                    host_caster_idx = j;
                    break;
                }
            }

            bool is_host_planet = false;
            if (host_caster_idx >= 0) {
                float host_radius = u_casters[host_caster_idx].w;
                is_host_planet = (length((u_casters[host_caster_idx].xyz - f_body_center) - f_rel_pos) < host_radius * 1.5);
            } else {
                is_host_planet = (dist_to_center < inner_r);
            }

            if (!is_host_planet) continue;

            vec3 P_dir = normalize(f_rel_pos);
            float frag_elevation = dot(P_dir, ring_normal);

            vec3 frag_to_star0 = (u_stars_pos_radius[0].xyz - f_body_center) - f_rel_pos;
            float dist_to_star0 = length(frag_to_star0);
            vec3 L0 = dist_to_star0 > 1e-5 ? (frag_to_star0 / dist_to_star0) : vec3(0.0, 1.0, 0.0);
            float sun_elev_0 = dot(L0, ring_normal);

            for (int s = 0; s < num_stars; s++) {
                vec3 star_pos = u_stars_pos_radius[s].xyz;
                float star_radius = u_stars_pos_radius[s].w;
                vec3 frag_to_star = (star_pos - f_body_center) - f_rel_pos;
                float dist_to_star = length(frag_to_star);
                if (dist_to_star < 1e-5) continue;
                vec3 L = frag_to_star / dist_to_star;
                float sun_elev_s = dot(L, ring_normal);
                float rel_hemi = (sun_elev_s * sun_elev_0 >= 0.0) ? 1.0 : -1.0;

                vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
                float star_sin_lat = abs(dot(L, pole_dir));

                vec3 eq_color = u_stars_colors[s].rgb;
                float eq_lum = u_stars_colors[s].a;
                vec3 pole_color = u_stars_pole_colors[s].rgb;
                float pole_lum = u_stars_pole_colors[s].a;

                vec3 star_color = mix(eq_color, pole_color, star_sin_lat);
                float star_lum = mix(eq_lum, pole_lum, star_sin_lat);

                vec3 P_eq_raw = P_dir - ring_normal * dot(P_dir, ring_normal);
                float len_P_eq = length(P_eq_raw);
                vec3 P_eq = len_P_eq > 1e-5 ? P_eq_raw / len_P_eq : vec3(1.0, 0.0, 0.0);

                vec3 antiL = -L;
                vec3 antiL_eq_raw = antiL - ring_normal * dot(antiL, ring_normal);
                float len_antiL_eq = length(antiL_eq_raw);
                vec3 antiL_eq = len_antiL_eq > 1e-5 ? antiL_eq_raw / len_antiL_eq : vec3(-1.0, 0.0, 0.0);

                vec3 cross_rel = cross(antiL_eq, P_eq);
                float sin_rel = dot(cross_rel, ring_normal);
                float cos_rel = clamp(dot(P_eq, antiL_eq), -1.0, 1.0);
                float phi_center = atan(sin_rel, cos_rel);

                float x_prime = phi_center / PI;
                float phi_uv = sign(x_prime) * pow(abs(x_prime), 0.666666667) * 0.5 + 0.5;

                float y_prime = frag_elevation * rel_hemi;
                float elev_uv = sign(y_prime) * pow(abs(y_prime), 0.666666667) * 0.5 + 0.5;

                vec2 map_uv = vec2(phi_uv, (float(k) + elev_uv) / 16.0);
                vec3 total_ring_irradiance = texture(u_ringshine_map, map_uv).rgb;

                float shine_intensity = 0.318309886; // 1 / PI
                vec3 ring_tint = total_ring_irradiance;
                float irradiance = u_hdr_enabled ? (star_lum / max(dist_to_star * dist_to_star, 1e-8)) : 1.0;

                ring_shine += ring_tint * star_color * shine_intensity * irradiance;
            }
        }
    }

    // Direct and secondary bounce illumination
    vec3 final_color = albedo * total_diffuse;
    if (u_planetshine_enabled || u_ringshine_enabled) {
        final_color += albedo * bounce_light;
    }
    if (u_ringshine_enabled) {
        final_color += albedo * ring_shine;
    }

    // Tile Debug Visualization mode
    if (u_debug_tiles) {
        // 1. LOD Color Palette
        vec3 lod_palette[8] = vec3[8](
            vec3(0.95, 0.25, 0.25), // LOD 0: Red
            vec3(1.00, 0.55, 0.15), // LOD 1: Orange
            vec3(0.95, 0.90, 0.20), // LOD 2: Yellow
            vec3(0.25, 0.90, 0.35), // LOD 3: Green
            vec3(0.15, 0.80, 0.95), // LOD 4: Cyan
            vec3(0.25, 0.45, 1.00), // LOD 5: Blue
            vec3(0.70, 0.25, 0.95), // LOD 6: Purple
            vec3(1.00, 0.25, 0.75)  // LOD 7: Pink
        );
        int lod_i = clamp(int(f_lod_level + 0.5), 0, 7);
        vec3 lod_tint = lod_palette[lod_i];

        // 2. Tile Borders
        vec2 d_edge = min(f_local_uv, 1.0 - f_local_uv);
        float is_border = 1.0 - step(0.018, min(d_edge.x, d_edge.y));

        // False color tint + border overlay
        final_color = mix(final_color * lod_tint * 1.5, vec3(1.0, 1.0, 0.1), is_border);
    }

    if (u_hdr_enabled) {
        final_color *= u_exposure;
    }

    out_color = vec4(final_color, 1.0);
}
