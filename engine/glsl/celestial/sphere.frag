#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define MAX_RING_PLANES 16
#define PI 3.14159265358979323846

in vec3 f_color;
in vec3 f_world_pos;
in vec3 f_normal;
in float f_is_star;
in float f_clip_z;
in float f_brightness_scale;
flat in uvec2 f_caster_mask;
flat in uint f_ring_mask;
flat in vec3 f_planetshine_dir;
flat in vec3 f_planetshine_color;
flat in float f_tex_idx;
flat in float f_rotation_angle;
flat in vec3 f_pole;
in vec3 f_local_pos;

uniform sampler2DArray u_planet_textures;
uniform sampler2DArray u_planet_normal_textures;
uniform sampler2DArray u_planet_specular_textures;

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
};

// Ring shadow planes (sphere-only)
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec4 u_ring_params[MAX_RING_PLANES];
uniform vec3 u_ring_colors[MAX_RING_PLANES];
uniform vec3 u_ring_5colors[MAX_RING_PLANES * 5];
uniform sampler2D u_ring_gradients;
uniform sampler2D u_ringshine_lut;
uniform sampler3D u_ringshine_cdf_lut;
uniform sampler2D u_ringshine_map;
uniform int u_num_ring_planes;
uniform float u_caster_max_bend[64];
uniform uint u_ring_coplanar_mask[16];
uniform int u_atmo_quality;
uniform bool u_planetshine_enabled;
uniform bool u_ringshine_enabled;
uniform int u_ringshine_band_count;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform vec3 u_camera_pos;
uniform float u_au_to_km;

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

// Shared analytical eclipse penumbra + physical atmospheric lens optics & Danjon refraction tinting.
// Inputs are per-caster quantities already resolved by the caller.
vec3 casterShadowTerm(float alpha, float beta, float gamma,
                      float penumbra_outer, float penumbra_inner,
                      float max_bend, vec3 atmo_tint) {
    float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
    float occ = clamp(max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma), 0.0, 1.0);
    // Correct geometric shadow curve: (1 - occ)^GAMMA (Space Engine gamma correction: 1 / 1.6)
    vec3 sh = vec3(pow(clamp(1.0 - occ, 0.0, 1.0), 1.0 / 1.6));

    if (max_bend > 1e-6 && gamma < penumbra_outer) {
        // Required bend angle for light to reach inside the planet's geometric shadow
        float req_bend = beta - gamma;

        // Rays requiring bend > max_bend hit the solid planet body (100% blocked)
        if (req_bend <= max_bend) {
            // Normalized penetration depth in atmosphere (0 = top of atmosphere, 1 = surface level)
            float atmo_depth = clamp(max(0.0, req_bend) / max_bend, 0.0, 1.0);

            // Compute Rayleigh / Mie / absorption spectral transmittance combined with composition tint
            vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (atmo_depth * 2.0));
            vec3 ext_exp = vec3(3.5, 10.0, 22.0) * (1.5 - deep_tint);
            vec3 atmo_transmittance = vec3(
                exp(-ext_exp.r * pow(atmo_depth, 1.5)),
                exp(-ext_exp.g * pow(atmo_depth, 1.5)),
                exp(-ext_exp.b * pow(atmo_depth, 1.5))
            );

            // Physical Spherical Lens Optics Dynamics (Space Engine Model)
            // Focal ratio: f_r = D_caster / D_focal = max_bend / beta
            float focal_ratio = max_bend / max(1e-6, beta);

            // Inverse-square beam divergence beyond focal plane: (D_focal / D_caster)^2 = (1 / f_r)^2
            float distance_divergence = min(1.0, 1.0 / max(1e-6, focal_ratio * focal_ratio));

            // Spherical lens caustic amplification near focal plane (f_r ~ 1.0)
            float lens_amplification = clamp(1.0 / (abs(1.0 - focal_ratio) + 0.3), 0.5, 3.0);

            // Smooth surface grazing fade to zero at solid body boundary (h = 0, atmo_depth = 1)
            float body_surface_fade = smoothstep(1.0, 0.8, atmo_depth);

            // Physical Danjon Lunar Eclipse Scale with Visual Exposure Boost:
            // Base physical value is ~0.00005, boosted to 0.015 for vibrant, clear visual presentation.
            float atmo_base_intensity = 0.015;
            float refraction_intensity = atmo_base_intensity * distance_divergence * lens_amplification * (1.0 - atmo_depth * 0.7) * body_surface_fade;

            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            sh += deep_tint * atmo_transmittance * refraction_intensity * atmo_blend;
        }
    }
    return clamp(sh, vec3(0.0), vec3(1.0));
}

float eval_ringshine_cdf(float angle, float v_tex, float sin_lat) {
    float TWO_PI = 6.28318530717958647692;
    float a_mod = angle - TWO_PI * floor((angle + PI) / TWO_PI);
    float k = floor((angle + PI) / TWO_PI);
    float u = clamp(abs(a_mod) / PI, 0.0, 1.0);

    // Remap coordinates to texel centers to avoid GL_CLAMP_TO_EDGE flat spots (derivative kinks)
    float u_tex = 0.5 / 128.0 + u * (127.0 / 128.0);
    float v_tex_mapped = 0.5 / 128.0 + v_tex * (127.0 / 128.0);
    float sin_lat_mapped = 0.5 / 64.0 + sin_lat * (63.0 / 64.0);

    float base_cdf = texture(u_ringshine_cdf_lut, vec3(u_tex, v_tex_mapped, sin_lat_mapped)).r;
    float signed_cdf = (a_mod < 0.0) ? -base_cdf : base_cdf;
    return 2.0 * k + signed_cdf;
}

void main() {
    if (f_is_star > 0.5) {
        // Star is self-luminous, apply quadratic limb darkening
        vec3 V = normalize(u_camera_pos - f_world_pos);
        vec3 N = normalize(f_normal);
        float mu = max(dot(N, V), 0.0);

        // Sun-like limb darkening coefficients
        float c1 = 0.4;
        float c2 = 0.2;
        float ld = 1.0 - c1 * (1.0 - mu) - c2 * (1.0 - mu) * (1.0 - mu);

        // Shift temperature slightly towards cooler orange/red near the limb
        vec3 edge_tint = vec3(1.0, 0.85, 0.65);
        vec3 color_shift = mix(edge_tint, vec3(1.0), pow(mu, 0.5));

        // Find which star this is to get its luminosity and radius
        float star_lum = 1.0;
        float star_r = 0.0046547454; // 1 solar radius in AU
        vec3 star_base_color = f_color; // fallback
        for (int s = 0; s < u_num_stars; s++) {
            if (distance(f_world_pos, u_stars_pos_radius[s].xyz) < u_stars_pos_radius[s].w * 1.5) {
                star_r = max(u_stars_pos_radius[s].w, 1e-6);

                vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
                float star_sin_lat = abs(dot(N, pole_dir));

                vec3 eq_color = u_stars_colors[s].rgb;
                float eq_lum = u_stars_colors[s].a;
                vec3 pole_color = u_stars_pole_colors[s].rgb;
                float pole_lum = u_stars_pole_colors[s].a;

                star_base_color = mix(eq_color, pole_color, star_sin_lat);
                star_lum = mix(eq_lum, pole_lum, star_sin_lat);
                break;
            }
        }

        float surface_luminance = u_hdr_enabled ? (star_lum / (star_r * star_r)) : 1.0;
        vec3 final_star_color = star_base_color * color_shift * ld * surface_luminance;
        if (u_hdr_enabled) {
            final_star_color *= u_exposure;
        }
        out_color = vec4(final_star_color * f_brightness_scale, 1.0);
    } else {
        vec3 N = normalize(f_normal);
        float spec_intensity = 0.0;
        vec3 local_f_color = f_color;

        if (f_tex_idx > 0.0) {
            vec3 p = normalize(f_local_pos);
            vec3 ref = vec3(0.0, 1.0, 0.0);
            if (abs(dot(f_pole, ref)) > 0.999) {
                ref = vec3(1.0, 0.0, 0.0);
            }
            vec3 tangent = normalize(cross(f_pole, ref));
            vec3 bitangent = normalize(cross(f_pole, tangent));

            vec3 p_local = vec3(dot(p, tangent), dot(p, f_pole), dot(p, bitangent));

            float s = sin(-f_rotation_angle);
            float c = cos(-f_rotation_angle);
            vec3 p_rot = vec3(
                p_local.x * c - p_local.z * s,
                p_local.y,
                p_local.x * s + p_local.z * c
            );

            float u = 0.5 + atan(p_rot.z, p_rot.x) / (2.0 * 3.14159265);
            float v = 0.5 - asin(clamp(p_rot.y, -1.0, 1.0)) / 3.14159265;

            vec2 uv = vec2(u, v);
            vec2 dx = dFdx(uv);
            vec2 dy = dFdy(uv);

            if (dx.x > 0.5) dx.x -= 1.0;
            else if (dx.x < -0.5) dx.x += 1.0;

            if (dy.x > 0.5) dy.x -= 1.0;
            else if (dy.x < -0.5) dy.x += 1.0;

            vec4 tex_color = textureGrad(u_planet_textures, vec3(uv, f_tex_idx - 1.0), dx, dy);
            // Decode sRGB to Linear
            local_f_color = pow(tex_color.rgb, vec3(2.2));

            // SpaceEngine Solstice Winter Model: Blue polar atmosphere/cloud deck appears ONLY on the winter pole during solstice
            vec3 pole_dir_norm = length(f_pole) > 1e-4 ? normalize(f_pole) : vec3(0.0, 1.0, 0.0);
            vec3 frag_local_dir = normalize(f_local_pos);
            float sin_lat = abs(dot(frag_local_dir, pole_dir_norm));
            float lat_factor = smoothstep(0.1, 0.7, sin_lat);

            vec3 primary_star_dir = normalize(u_stars_pos_radius[0].xyz - f_world_pos);
            float sun_pole_dot = dot(primary_star_dir, pole_dir_norm);
            float frag_pole_dot = dot(frag_local_dir, pole_dir_norm);

            // Winter hemisphere condition: Sun and Fragment are on opposite sides of the equatorial plane
            float is_winter = step(sun_pole_dot * frag_pole_dot, 0.0);

            // Solstice progress: cosPhi is sine of solar elevation above equator = |sun_pole_dot|
            float cosPhi = abs(sun_pole_dot);
            float tanPhi = cosPhi / sqrt(max(1.0 - cosPhi * cosPhi, 1e-4));
            float solstice_factor = clamp(tanPhi * 1.8, 0.0, 1.0);

            float winter_solstice_effect = lat_factor * is_winter * solstice_factor;

            vec3 blue_polar_cloud = vec3(0.15, 0.45, 0.95) * dot(local_f_color, vec3(0.299, 0.587, 0.114));
            local_f_color = mix(local_f_color, blue_polar_cloud, winter_solstice_effect * 0.75);

            // Analytical TBN Mapping
            vec3 map_normal = textureGrad(u_planet_normal_textures, vec3(uv, f_tex_idx - 1.0), dx, dy).rgb;
            map_normal = map_normal * 2.0 - 1.0;

            vec3 T_rot = normalize(vec3(-p_rot.z, 0.0, p_rot.x));
            vec3 B_rot = normalize(cross(p_rot, T_rot));

            float s_inv = sin(f_rotation_angle);
            float c_inv = cos(f_rotation_angle);
            vec3 T_local = vec3(T_rot.x * c_inv - T_rot.z * s_inv, T_rot.y, T_rot.x * s_inv + T_rot.z * c_inv);
            vec3 B_local = vec3(B_rot.x * c_inv - B_rot.z * s_inv, B_rot.y, B_rot.x * s_inv + B_rot.z * c_inv);

            vec3 T_world = T_local.x * tangent + T_local.y * f_pole + T_local.z * bitangent;
            vec3 B_world = B_local.x * tangent + B_local.y * f_pole + B_local.z * bitangent;

            mat3 TBN = mat3(T_world, B_world, N);
            N = normalize(TBN * map_normal);

            spec_intensity = textureGrad(u_planet_specular_textures, vec3(uv, f_tex_idx - 1.0), dx, dy).r;
        }

        vec3 total_diffuse_color = vec3(0.0);
        vec3 total_specular_color = vec3(0.0);
        vec3 V = normalize(u_camera_pos - f_world_pos);

        for (int s = 0; s < u_num_stars; s++) {
            vec3 star_pos = u_stars_pos_radius[s].xyz;
            float star_radius = u_stars_pos_radius[s].w;
            vec3 frag_to_star = star_pos - f_world_pos;
            float dist_to_star = length(frag_to_star);
            if (dist_to_star < 1e-5) continue;
            vec3 L = frag_to_star / dist_to_star;

            vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
            float star_sin_lat = abs(dot(L, pole_dir));

            vec3 eq_color = u_stars_colors[s].rgb;
            float eq_lum = u_stars_colors[s].a;
            vec3 pole_color = u_stars_pole_colors[s].rgb;
            float pole_lum = u_stars_pole_colors[s].a;

            vec3 star_color = mix(eq_color, pole_color, star_sin_lat);
            float star_lum = mix(eq_lum, pole_lum, star_sin_lat);

            // Angular radius of the star for soft penumbra at terminator
            float star_ang_radius = star_radius / dist_to_star;
            float sin_alpha = clamp(star_ang_radius, 0.0, 1.0);

            // Lambert cosine law with soft terminator
            float NdotL = dot(N, L);
            float diffuse = clamp((NdotL + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);

            // Blinn-Phong specular highlight
            vec3 H = normalize(L + V);
            float NdotH = max(dot(N, H), 0.0);
            float specular = pow(NdotH, 64.0) * spec_intensity * diffuse;

            // Inverse-square falloff
            float falloff = star_lum / (dist_to_star * dist_to_star);
            diffuse *= falloff;
            specular *= falloff;

            // === Analytical eclipse shadows ===
            vec3 shadow = vec3(1.0);
            float star_radius_over_dist = star_radius / dist_to_star;

            for (int j = 0; j < u_num_casters; j++) {
                if (dot(shadow, shadow) < 0.001) break;

                if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
                else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }

                vec3 caster_pos = u_casters[j].xyz;
                float caster_r = u_casters[j].w;
                float atmo_h = u_caster_atmos[j].w;

                vec3 frag_to_caster = caster_pos - f_world_pos;
                float t_proj = dot(frag_to_caster, L);
                if (t_proj < 0.0) continue;

                float dist_sq = dot(frag_to_caster, frag_to_caster);
                if (dist_sq < caster_r * caster_r * 1.0404) continue; // Skip self

                float dist_to_caster = sqrt(dist_sq);
                vec3 cross_vec = cross(frag_to_caster, L);
                float perp_sq = dot(cross_vec, cross_vec);

                // Bounding cone early out using maximum equatorial radius
                float max_effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
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

                // Perfect Bounding Cone Early Out (Zero Artifacts)
                float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
                float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
                if (perp_sq > r_penumbra * r_penumbra) continue;

                float inv_dist = 1.0 / dist_to_caster;
                float alpha = local_star_radius_over_dist;
                float beta = caster_r * inv_dist;
                float gamma = sqrt(perp_sq) * inv_dist;
                float penumbra_outer = alpha + beta;
                float penumbra_inner = max(0.0, beta - alpha);
                shadow *= casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, u_caster_max_bend[j], u_caster_atmos[j].xyz);
            }

            // === Ring shadow on planet surface ===
            // Only compute ring shadows if the fragment is lit by this star
            if (NdotL > -sin_alpha) {
                uint processed_mask = 0u;
                for (int k = 0; k < u_num_ring_planes; k++) {
                    if ((f_ring_mask & (1u << k)) == 0u) continue;
                    if ((processed_mask & (1u << k)) != 0u) continue;

                    vec3 plane_center = u_ring_center[k];
                    vec3 plane_normal = u_ring_normal[k];

                    uint coplanar_mask = u_ring_coplanar_mask[k];
                    processed_mask |= coplanar_mask;

                    float denom = dot(L, plane_normal);
                    if (abs(denom) < 1e-8) continue;

                    float t = dot(plane_center - f_world_pos, plane_normal) / denom;
                    if (t <= 0.0 || t >= dist_to_star) continue;

                    vec3 hit = f_world_pos + t * L;
                    vec3 vec_radial = hit - plane_center;
                    float d = length(vec_radial);

                    float r_star_proj = star_ang_radius * t;
                    vec3 L_plane = L - denom * plane_normal;
                    float L_plane_len = length(L_plane);

                    float R_eff = r_star_proj;
                    if (L_plane_len > 1e-5 && d > 1e-5) {
                        vec3 L_proj = L_plane / L_plane_len;
                        vec3 T = normalize(cross(plane_normal, L));
                        vec3 dir_radial = vec_radial / d;
                        float cos_theta = dot(dir_radial, L_proj);
                        float sin_theta = dot(dir_radial, T);
                        R_eff = r_star_proj * sqrt( pow(cos_theta / max(1e-6, abs(denom)), 2.0) + pow(sin_theta, 2.0) );
                    }
                    R_eff = max(R_eff, fwidth(d) * 0.75);

                    float plane_occlusion = 0.0;

                    for (int j = k; j < u_num_ring_planes; j++) {
                        if ((coplanar_mask & (1u << j)) == 0u) continue;
                        float inner_r = u_ring_params[j].x;
                        float outer_r = u_ring_params[j].y;

                        float overlap_min = max(inner_r, d - R_eff);
                        float overlap_max = min(outer_r, d + R_eff);

                        if (overlap_min >= overlap_max) continue;

                        float v_min = clamp((overlap_min - d) / max(1e-9, R_eff), -1.0, 1.0);
                        float v_max = clamp((overlap_max - d) / max(1e-9, R_eff), -1.0, 1.0);

                        float f_max = (v_max * sqrt(max(0.0, 1.0 - v_max*v_max)) + asin(v_max)) / 3.14159265358979 + 0.5;
                        float f_min = (v_min * sqrt(max(0.0, 1.0 - v_min*v_min)) + asin(v_min)) / 3.14159265358979 + 0.5;
                        float fraction = max(0.0, f_max - f_min);

                        float sum_alpha = 0.0;
                        float overlap_width = overlap_max - overlap_min;
                        if (overlap_width < 0.002 * (outer_r - inner_r)) {
                            // Penumbra is extremely narrow, 1 sample is sufficient
                            float r = 0.5 * (overlap_min + overlap_max);
                            float p = (r - inner_r) / max(1e-6, outer_r - inner_r);
                            sum_alpha = textureLod(u_ring_gradients, vec2(p, (float(j) + 0.5) / 16.0), 0.0).a;
                        } else {
                            // Penumbra is wide, use 5 samples for filtering
                            int tex_samples = 5;
                            for(int s = 0; s < tex_samples; s++) {
                                float u = (float(s) + 0.5) / float(tex_samples);
                                float r = mix(overlap_min, overlap_max, u);
                                float p = (r - inner_r) / max(1e-6, outer_r - inner_r);
                                sum_alpha += textureLod(u_ring_gradients, vec2(p, (float(j) + 0.5) / 16.0), 0.0).a;
                            }
                            sum_alpha /= float(tex_samples);
                        }
                        float alpha_mult = sum_alpha;
                        float opacity = u_ring_params[j].z;

                        float tau = -log(max(1e-6, 1.0 - opacity * alpha_mult));
                        float light_mu = max(1e-4, abs(dot(normalize(u_ring_normal[j]), L)));
                        plane_occlusion += fraction * (1.0 - exp(-tau / light_mu));
                    }

                    shadow *= vec3(1.0 - min(plane_occlusion, 1.0));
                }
            }

            total_diffuse_color += star_color * diffuse * shadow;
            total_specular_color += star_color * specular * shadow;
        }

        // === Moonshine / Planetshine ===
        vec3 bounce_light = vec3(0.0);
        if (dot(f_planetshine_color, f_planetshine_color) > 1e-12) {
            float NdotC = max(0.0, dot(N, f_planetshine_dir));
            bounce_light = f_planetshine_color * NdotC;
        }

        // === Ringshine ===
        vec3 ring_shine = vec3(0.0);
        for (int k = 0; k < u_num_ring_planes; k++) {
            vec3 ring_center = u_ring_center[k];
            vec3 ring_normal = u_ring_normal[k];
            float inner_r = u_ring_params[k].x;
            float outer_r = u_ring_params[k].y;
            float opacity = u_ring_params[k].z;

            vec3 frag_to_center = ring_center - f_world_pos;
            float dist_to_center = length(frag_to_center);

            // Fast distance cull: ignore rings if we are too far away
            if (dist_to_center > outer_r * 20.0 || dist_to_center < 1e-6) continue;

            int host_caster_idx = -1;
            for (int j = 0; j < u_num_casters; j++) {
                if (distance(u_casters[j].xyz, ring_center) < 1e-4) {
                    host_caster_idx = j;
                    break;
                }
            }

            bool is_host_planet = false;
            if (host_caster_idx >= 0) {
                float host_radius = u_casters[host_caster_idx].w;
                is_host_planet = (distance(f_world_pos, u_casters[host_caster_idx].xyz) < host_radius * 1.5);
            } else {
                is_host_planet = (dist_to_center < inner_r);
            }

            if (!is_host_planet) continue;

            float frag_elevation = dot(N, ring_normal);
            float moon_h = -dot(frag_to_center, ring_normal);

            for (int s = 0; s < u_num_stars; s++) {
                vec3 star_pos = u_stars_pos_radius[s].xyz;
                float star_radius = u_stars_pos_radius[s].w;
                vec3 frag_to_star = star_pos - f_world_pos;
                float dist_to_star = length(frag_to_star);
                if (dist_to_star < 1e-5) continue;
                vec3 L = frag_to_star / dist_to_star;

                vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
                float star_sin_lat = abs(dot(L, pole_dir));

                vec3 eq_color = u_stars_colors[s].rgb;
                float eq_lum = u_stars_colors[s].a;
                vec3 pole_color = u_stars_pole_colors[s].rgb;
                float pole_lum = u_stars_pole_colors[s].a;

                vec3 star_color = mix(eq_color, pole_color, star_sin_lat);
                float star_lum = mix(eq_lum, pole_lum, star_sin_lat);

                float sun_elevation = dot(L, ring_normal);
                float star_ang_radius = star_radius / max(dist_to_star, 1e-6);
                float effective_sun_elev = sqrt(sun_elevation * sun_elevation + 0.180126 * star_ang_radius * star_ang_radius);

                float shine_intensity = 0.0;
                vec3 ring_tint = u_ring_colors[k];

                // --- PRE-INTEGRATED LUT RINGSHINE FOR HOST PLANET ONLY ---
                float sin_lat = clamp(abs(frag_elevation), 0.001, 0.999);

                // Fragment equator normal vector (in ring plane)
                vec3 N_eq_raw = N - ring_normal * dot(N, ring_normal);
                float len_N_eq = length(N_eq_raw);
                vec3 N_eq = len_N_eq > 1e-5 ? N_eq_raw / len_N_eq : vec3(1.0, 0.0, 0.0);

                // Anti-solar vector projection in ring plane
                vec3 antiL = -L;
                vec3 antiL_eq_raw = antiL - ring_normal * dot(antiL, ring_normal);
                float len_antiL_eq = length(antiL_eq_raw);
                vec3 antiL_eq = len_antiL_eq > 1e-5 ? antiL_eq_raw / len_antiL_eq : vec3(-1.0, 0.0, 0.0);

                // Azimuth angle phi_center between anti-solar vector and surface fragment meridian
                vec3 cross_rel = cross(antiL_eq, N_eq);
                float sin_rel = dot(cross_rel, ring_normal);
                float cos_rel = clamp(dot(N_eq, antiL_eq), -1.0, 1.0);
                float phi_center = atan(sin_rel, cos_rel);

                float same_hemisphere = sun_elevation * frag_elevation;
                float same_hemi_t = smoothstep(-0.02, 0.02, same_hemisphere);

                float x_prime = phi_center / PI;
                float phi_uv = sign(x_prime) * pow(abs(x_prime), 0.666666667) * 0.5 + 0.5;

                float y_prime = frag_elevation;
                float elev_uv = sign(y_prime) * pow(abs(y_prime), 0.666666667) * 0.5 + 0.5;

                vec2 map_uv = vec2(phi_uv, (float(k) + elev_uv) / 16.0);
                vec3 total_ring_irradiance = texture(u_ringshine_map, map_uv).rgb;

                float NdotL = dot(N, L);
                float day_face = smoothstep(-0.05, 0.05, NdotL) * max(0.0, NdotL);
                float noon_fade = mix(1.0, 0.4, day_face);

                float face_multiplier = mix(1.0, 1.0 * noon_fade, same_hemi_t);
                // Apply 1/PI (~0.3183) physical BRDF normalization factor
                shine_intensity = effective_sun_elev * face_multiplier * 0.318309886;
                ring_tint = total_ring_irradiance;

                float irradiance = u_hdr_enabled ? (star_lum / max(dist_to_star * dist_to_star, 1e-8)) : 1.0;

                ring_shine += ring_tint * star_color * shine_intensity * irradiance;
            }
        }

        vec3 final_color = local_f_color * total_diffuse_color + total_specular_color;
        if (u_planetshine_enabled) {
            final_color += local_f_color * bounce_light;
        }
        if (u_ringshine_enabled) {
            final_color += local_f_color * ring_shine;
        }
        if (u_hdr_enabled) {
            final_color *= u_exposure;
        }

        // Aerial perspective bypassed for cleaner rendering and matching ray marching
        float trans = 1.0;
        vec3 ap_rgb = vec3(0.0);
        out_color = vec4(final_color * f_brightness_scale * trans + ap_rgb, 1.0);
    }
}
