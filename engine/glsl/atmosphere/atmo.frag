#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define MAX_RING_PLANES 16
#define PI 3.14159265358979

float get_ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

in vec3 f_local_pos;
in float f_clip_z;

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

layout(std430, binding = 2) buffer AllInstances {
    vec4 instances[];
};
uint u_caster_mask_lo;
uint u_caster_mask_hi;
uint u_ring_mask;

uniform vec3 u_camera_pos;

uniform vec3 u_refract_center;
uniform float u_refract_radius;
uniform float u_refract_max_bend;
uniform float u_refract_scale_height;
uniform vec3 u_refract_pole;
uniform float u_refract_oblateness;

layout(std430, binding = 8) buffer AtmoData {
    vec3  u_body_offset;
    float u_atmo_radius_au;
    vec3  u_beta_rayleigh;
    float u_h_rayleigh;
    vec3  u_beta_mie;
    float u_h_mie;
    vec3  u_beta_abs_mixed;
    float u_mie_g;
    vec3  u_beta_abs_layered;
    float u_sun_intensity;
    float u_planet_radius_km;
    float u_atmo_radius_km;
    float u_au_to_km;
    int   u_num_samples;
    vec4  u_pole_obl;
    int   u_num_active_casters;
    bool  u_atmo_adaptive_steps;
    int   u_body_idx;
    float u_frame_counter;
    vec3  u_mie_albedo;          // per-channel single-scattering albedo (omega_0)
    float u_refractivity;        // surface (n_mix - 1), drives eclipse refraction
    vec4  u_active_casters[8];
    vec4  u_active_caster_poles_obl[8];
    float u_active_caster_R_minor[8];
    vec4  u_active_caster_atmos[8];
    float u_active_max_bend[8];
    float u_ozone_peak_km;
    float u_ozone_width_km;
    float u_planet_clip_km;
    float u_max_adaptive_steps;
};

uniform sampler2D u_ring_gradients;
uniform int u_num_ring_planes;
uniform int u_atmo_quality;
uniform bool u_stochastic_noise;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec4 u_ring_params[MAX_RING_PLANES];
uniform int u_atmo_clip_mode;

uniform uint u_ring_coplanar_mask[16];
uniform sampler2D u_eclipse_lut; // Kept to avoid uniform bound errors
uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform bool u_planetshine_enabled;

uniform sampler2D u_ringshine_lut;
uniform sampler3D u_ringshine_cdf_lut;
uniform sampler2D u_ringshine_map;
uniform bool u_ringshine_enabled;
uniform int u_ringshine_band_count;
uniform sampler2D u_depth_texture;
uniform vec2 u_screen_res;

uniform bool u_vrs_highres_pass;
uniform sampler2D u_lowres_trans;

layout(location = 0, index = 0) out vec4 out_scattered;
layout(location = 0, index = 1) out vec4 out_transmittance;

float eval_ringshine_cdf(float angle, float v_tex, float sin_lat) {
    float TWO_PI = 6.28318530717958647692;
    float a_mod = angle - TWO_PI * floor((angle + PI) / TWO_PI);
    float k = floor((angle + PI) / TWO_PI);
    float u = clamp(abs(a_mod) / PI, 0.0, 1.0);

    float u_tex = 0.5 / 128.0 + u * (127.0 / 128.0);
    float v_tex_mapped = 0.5 / 128.0 + v_tex * (127.0 / 128.0);
    float sin_lat_mapped = 0.5 / 64.0 + sin_lat * (63.0 / 64.0);

    float base_cdf = texture(u_ringshine_cdf_lut, vec3(u_tex, v_tex_mapped, sin_lat_mapped)).r;
    float signed_cdf = (a_mod < 0.0) ? -base_cdf : base_cdf;
    return 2.0 * k + signed_cdf;
}
vec3 toSphericalSpace(vec3 p, vec4 pole_scale) {
    float f_scale = pole_scale.w;
    if (f_scale <= 1.00001) return p;
    vec3 pole = pole_scale.xyz;
    float h = dot(p, pole);
    return p + (h * (f_scale - 1.0)) * pole;
}

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

float compute_refraction_angle(vec3 C, vec3 V, float d) {
    if (u_refract_max_bend <= 1e-6) return 0.0;
    float s_min = -dot(C, V);
    vec3 P_min = C + s_min * V;
    float r_min = length(P_min);

    float local_refract_radius = u_refract_radius;
    if (u_refract_oblateness > 0.001 && u_refract_oblateness < 0.99) {
        vec3 P_dir = r_min > 1e-6 ? (P_min / r_min) : vec3(0.0, 1.0, 0.0);
        vec3 pole_dir = length(u_refract_pole) > 1e-4 ? normalize(u_refract_pole) : vec3(0.0, 1.0, 0.0);
        float cos_t = abs(dot(P_dir, pole_dir));
        float k = 1.0 / (1.0 - u_refract_oblateness);
        float denom = sqrt(max(1e-6, 1.0 + (k * k - 1.0) * cos_t * cos_t));
        local_refract_radius = u_refract_radius / denom;
    }

    if (r_min > local_refract_radius + u_refract_scale_height * 15.0) return 0.0;

    float r_min_clamped = max(r_min, local_refract_radius - u_refract_scale_height);
    float delta_rmin = u_refract_max_bend * exp(-(r_min_clamped - local_refract_radius) / max(1e-4, u_refract_scale_height));
    float sigma = sqrt(max(1e-4, r_min_clamped * u_refract_scale_height));

    if (d < 0.1 * sigma) {
        float kappa_0 = (delta_rmin / (sigma * 2.506628)) * exp(-(s_min * s_min) / (2.0 * sigma * sigma));
        return 0.5 * kappa_0 * d;
    }

    float sqrt2_sig = 1.4142135 * sigma;
    float x_d = (d - s_min) / sqrt2_sig;
    float x_0 = s_min / sqrt2_sig;

    float E_d = sign(x_d) * sqrt(max(0.0, 1.0 - exp(-1.239 * x_d * x_d)));
    float E_0 = sign(x_0) * sqrt(max(0.0, 1.0 - exp(-1.239 * x_0 * x_0)));
    float G_d = exp(-clamp(x_d * x_d, 0.0, 50.0));
    float G_0 = exp(-clamp(x_0 * x_0, 0.0, 50.0));

    float alpha = delta_rmin * ( 0.5 * (E_d + E_0) * (1.0 - s_min / d) + (sigma / (d * 2.506628)) * (G_d - G_0) );
    return max(0.0, alpha);
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

vec3 compute_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, vec3 planet_center_render, float star_radius, float star_obl, vec3 star_pole) {
    vec3 shadow = vec3(1.0);
    float base_star_radius_over_dist = star_radius / max(dist_to_star, 1e-6);

    for (int i = 0; i < u_num_active_casters; i++) {
        vec3 caster_pos = u_active_casters[i].xyz;
        float caster_r = u_active_casters[i].w;
        float atmo_h = u_active_caster_atmos[i].w;

        vec3 s_to_c = caster_pos - eval_render_pos;
        float t_proj = dot(s_to_c, L_dir);
        if (t_proj < 0.0) continue;

        float dist_sq = dot(s_to_c, s_to_c);
        if (dist_sq < caster_r * caster_r * 1.0404) continue;

        float dist_to_caster = sqrt(dist_sq);
        vec3 cross_vec = cross(s_to_c, L_dir);
        float perp_sq = dot(cross_vec, cross_vec);

        vec3 perp_vec = s_to_c - t_proj * L_dir;
        float caster_r_minor = u_active_caster_R_minor[i];
        if (caster_r_minor < caster_r - 1e-5) {
            vec3 pole = u_active_caster_poles_obl[i].xyz;
            caster_r = get_oblate_radius(caster_r, caster_r_minor, pole, L_dir, perp_vec);
        }

        float directional_star_r = star_radius;
        if (star_obl < star_radius - 1e-5) {
            directional_star_r = get_oblate_radius(star_radius, star_obl, star_pole, L_dir, perp_vec);
        }
        float local_star_radius_over_dist = directional_star_r / max(dist_to_star, 1e-6);

        float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
        float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
        if (perp_sq > r_penumbra * r_penumbra) continue;

        float inv_dist = 1.0 / dist_to_caster;
        float alpha = local_star_radius_over_dist;
        float beta = caster_r * inv_dist;
        float gamma = sqrt(perp_sq) * inv_dist;

        float penumbra_outer = alpha + beta;
        float penumbra_inner = abs(beta - alpha);

        float max_bend = u_active_max_bend[i];
        vec3 caster_shadow = casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, max_bend, u_active_caster_atmos[i].xyz);
        shadow *= caster_shadow;
    }

    uint processed_mask = 0u;
    for (int k = 0; k < u_num_ring_planes; k++) {
        if ((u_ring_mask & (1u << k)) == 0u) continue;
        if ((processed_mask & (1u << k)) != 0u) continue;

        vec3 plane_center = u_ring_center[k];
        vec3 plane_normal = u_ring_normal[k];

        uint coplanar_mask = u_ring_coplanar_mask[k];
        processed_mask |= coplanar_mask;

        float denom = dot(L_dir, plane_normal);
        if (abs(denom) < 1e-8) continue;

        float t_ring = dot(plane_center - eval_render_pos, plane_normal) / denom;
        if (t_ring <= 0.0 || t_ring >= dist_to_star) continue;

        vec3 hit = eval_render_pos + t_ring * L_dir;
        vec3 vec_radial = hit - plane_center;
        float d = length(vec_radial);

        float r_star_proj = star_radius * t_ring / max(dist_to_star, 1e-6);
        vec3 L_plane = L_dir - denom * plane_normal;
        float L_plane_len = length(L_plane);

        float R_eff = r_star_proj;
        if (L_plane_len > 1e-5 && d > 1e-5) {
            vec3 L_proj = L_plane / L_plane_len;
            vec3 T = normalize(cross(plane_normal, L_dir));
            vec3 dir_radial = vec_radial / d;
            float cos_theta = dot(dir_radial, L_proj);
            float sin_theta = dot(dir_radial, T);
            R_eff = r_star_proj * sqrt( pow(cos_theta / max(1e-6, abs(denom)), 2.0) + pow(sin_theta, 2.0) );
        }
        R_eff = max(R_eff, fwidth(d) * 0.75);

        float plane_occlusion = 0.0;

        for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
            if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
            float inner_r = u_ring_params[ring_idx].x;
            float outer_r = u_ring_params[ring_idx].y;

            float overlap_min = max(inner_r, d - R_eff);
            float overlap_max = min(outer_r, d + R_eff);

            if (overlap_min >= overlap_max) continue;

            float v_min = clamp((overlap_min - d) / max(1e-9, R_eff), -1.0, 1.0);
            float v_max = clamp((overlap_max - d) / max(1e-9, R_eff), -1.0, 1.0);

            float f_max = (v_max * sqrt(max(0.0, 1.0 - v_max*v_max)) + asin(v_max)) / PI + 0.5;
            float f_min = (v_min * sqrt(max(0.0, 1.0 - v_min*v_min)) + asin(v_min)) / PI + 0.5;
            float fraction = max(0.0, f_max - f_min);

            float p_mid = ((overlap_min + overlap_max) * 0.5 - inner_r) / max(1e-6, outer_r - inner_r);
            float alpha_mult = textureLod(u_ring_gradients, vec2(p_mid, (float(ring_idx) + 0.5) / 16.0), 0.0).a;
            float opacity = u_ring_params[ring_idx].z;

            float tau = -log(max(1e-6, 1.0 - opacity * alpha_mult));
            float light_mu = max(1e-4, abs(dot(normalize(u_ring_normal[ring_idx]), L_dir)));
            plane_occlusion += fraction * (1.0 - exp(-tau / light_mu));
        }

        shadow *= vec3(1.0 - min(plane_occlusion, 1.0));
    }
    return shadow;
}

vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float a = dot(dir, dir);
    float b = dot(origin, dir);

    // Use cross product for perpendicular distance — avoids catastrophic cancellation
    // at large distances where origin and (b/a)*dir are both huge but nearly equal.
    // |origin × dir|² / |dir|² = perpendicular distance² from ray to sphere center.
    vec3 cross_vec = cross(origin, dir);
    float p2 = dot(cross_vec, cross_vec) / a;
    float r2 = radius * radius;

    if (p2 > r2) return vec2(1e10, -1e10);

    float d = sqrt((r2 - p2) / a);
    float t_closest = -b / a;

    return vec2(t_closest - d, t_closest + d);
}

vec3 get_transmittance(float r, float cos_theta) {
    float h_norm = clamp((r - u_planet_radius_km) / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float v = sqrt(h_norm);
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

vec3 get_transmittance_precomputed(float v, float cos_theta) {
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

void main() {
    if (f_clip_z < 0.0) discard;
    
    if (u_vrs_highres_pass && u_screen_res.x > 0.0 && u_screen_res.y > 0.0) {
        vec2 depth_uv = gl_FragCoord.xy / u_screen_res;
        
        vec2 low_res_size = vec2(textureSize(u_lowres_trans, 0));
        vec2 texel_pos = depth_uv * low_res_size - 0.5;
        vec2 p_low = floor(texel_pos);
        
        vec2 uv00 = (p_low + vec2(0.5, 0.5)) / low_res_size;
        vec2 uv10 = (p_low + vec2(1.5, 0.5)) / low_res_size;
        vec2 uv01 = (p_low + vec2(0.5, 1.5)) / low_res_size;
        vec2 uv11 = (p_low + vec2(1.5, 1.5)) / low_res_size;
        
        float t00_a = texture(u_lowres_trans, uv00).a;
        float t10_a = texture(u_lowres_trans, uv10).a;
        float t01_a = texture(u_lowres_trans, uv01).a;
        float t11_a = texture(u_lowres_trans, uv11).a;
        
        bool is_atmo_edge = (t00_a > 0.0 || t10_a > 0.0 || t01_a > 0.0 || t11_a > 0.0) &&
                            (t00_a == 0.0 || t10_a == 0.0 || t01_a == 0.0 || t11_a == 0.0);
                            
        float d00 = texture(u_depth_texture, uv00).r;
        float d10 = texture(u_depth_texture, uv10).r;
        float d01 = texture(u_depth_texture, uv01).r;
        float d11 = texture(u_depth_texture, uv11).r;
        
        float max_d = max(max(d00, d10), max(d01, d11));
        float min_d = min(min(d00, d10), min(d01, d11));
        
        float high_depth_raw = texture(u_depth_texture, depth_uv).r;
        
        float log_far_denom = log2(u_depth_C * u_far + 1.0);
        float d00_lin = (exp2(d00 * log_far_denom) - 1.0) / u_depth_C;
        float d10_lin = (exp2(d10 * log_far_denom) - 1.0) / u_depth_C;
        float d01_lin = (exp2(d01 * log_far_denom) - 1.0) / u_depth_C;
        float d11_lin = (exp2(d11 * log_far_denom) - 1.0) / u_depth_C;
        float high_depth = (exp2(high_depth_raw * log_far_denom) - 1.0) / u_depth_C;
        
        float max_d_lin = max(max(d00_lin, d10_lin), max(d01_lin, d11_lin));
        float min_d_lin = min(min(d00_lin, d10_lin), min(d01_lin, d11_lin));
        
        float depth_tol_edge = max(high_depth * 0.05, 0.0001);
        bool is_depth_edge = (max_d_lin - min_d_lin) > depth_tol_edge;
        
        if (!is_atmo_edge && !is_depth_edge) {
            discard; // Inner pixel, rendered efficiently in low-res pass
        }
    }

    u_ring_mask = floatBitsToUint(instances[u_body_idx * 7 + 3].w);
    vec3 body_planetshine_dir = instances[u_body_idx * 7 + 4].xyz;
    vec3 body_planetshine_color = instances[u_body_idx * 7 + 5].xyz;

    vec3 planet_center_render = u_body_offset;
    vec3 f_pos_local_au = f_local_pos * u_atmo_radius_au;

    vec3 ray_origin_au = u_camera_pos;
    vec3 cam_to_body = planet_center_render - u_camera_pos;

    // Compute exact mathematical screen ray to avoid quantization noise
    // and moire patterns caused by interpolating f_pos_local_au across the proxy mesh triangles.
    vec2 ndc = (gl_FragCoord.xy / u_screen_res) * 2.0 - 1.0;
    vec4 clip_ray = vec4(ndc, -1.0, 1.0);
    vec4 eye_ray = inverse(projection) * clip_ray;
    eye_ray = vec4(eye_ray.xy, -1.0, 0.0);
    vec3 view_ray = normalize((inverse(view) * eye_ray).xyz);

    vec3 ray_dir = view_ray;
    bool is_refract_host = length(u_body_offset - u_refract_center) < 1e-4;


    vec3 ring_ray_dir = view_ray;
    vec3 ring_ray_origin_au = u_camera_pos;

    float s_min_au = 0.0;
    float ring_s_min_au = 0.0;

    if (u_refract_max_bend > 1e-6) {
        vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
        float d_km = length(cam_to_body * u_au_to_km);

        vec3 pole_n_approx = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
        float d_n_approx = dot(view_ray, pole_n_approx);
        if (abs(d_n_approx) > 1e-6) {
            float t_approx = dot(cam_to_body, pole_n_approx) / d_n_approx;
            if (t_approx > 0.0) {
                d_km = t_approx * u_au_to_km;
            }
        }

        float alpha = compute_refraction_angle(C_km, view_ray, d_km);
        if (alpha > 1e-7) {
            vec3 u_dir = C_km - view_ray * dot(C_km, view_ray);
            float u_len = length(u_dir);
            if (u_len > 1e-5) {
                u_dir /= u_len;
                ring_ray_dir = normalize(view_ray * cos(alpha) - u_dir * sin(alpha));

                float local_s_min = -dot(u_camera_pos - u_refract_center, view_ray);
                if (local_s_min > 0.0) {
                    ring_s_min_au = local_s_min;
                    ring_ray_origin_au = u_camera_pos + ring_s_min_au * (view_ray - ring_ray_dir);
                }
            }
        }

        if (!is_refract_host) {
            ray_dir = ring_ray_dir;
            ray_origin_au = ring_ray_origin_au;
            s_min_au = ring_s_min_au;
        } else {
            // Terrestrial Refraction for host planet atmosphere ray marching (horizon extension)
            vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
            float r_cam = length(C_km);

            vec3 pole_n = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
            float f_obl = u_pole_obl.w;
            float f_scale = (f_obl > 0.0 && f_obl < 0.99) ? (1.0 / (1.0 - f_obl)) : 1.0;
            vec3 C_scaled = C_km + pole_n * (dot(C_km, pole_n) * (f_scale * f_scale - 1.0));
            vec3 local_up = normalize(C_scaled);

            float local_refract_radius = u_refract_radius;
            if (f_obl > 0.001 && f_obl < 0.99) {
                vec3 P_dir = r_cam > 1e-6 ? (C_km / r_cam) : vec3(0.0, 1.0, 0.0);
                float cos_t = abs(dot(P_dir, pole_n));
                float k = 1.0 / (1.0 - f_obl);
                float denom = sqrt(max(1e-6, 1.0 + (k * k - 1.0) * cos_t * cos_t));
                local_refract_radius = u_refract_radius / denom;
            }

            float h = r_cam - local_refract_radius;
            if (h < u_refract_scale_height * 15.0) {
                float density = exp(-max(h, 0.0) / max(1e-4, u_refract_scale_height));
                float max_terr_alpha = 0.5 * u_refract_max_bend * density;

                if (max_terr_alpha > 1e-7) {
                    float mu = dot(view_ray, local_up);
                    float mu_horiz = -sqrt(max(0.0, 2.0 * max(h, 0.0) / local_refract_radius));
                    float delta_mu = sqrt(2.0 * u_refract_scale_height / local_refract_radius);

                    float x = (mu - mu_horiz) / max(1e-6, delta_mu);
                    float alpha = max_terr_alpha * exp(-x * x);

                    if (alpha > 1e-7) {
                        vec3 u_dir = local_up - view_ray * mu;
                        float u_len = length(u_dir);
                        if (u_len > 1e-5) {
                            u_dir /= u_len;
                            ray_dir = normalize(view_ray * cos(alpha) - u_dir * sin(alpha));
                        }
                    }
                }
            }
        }
    }

    // Precise ray origin in body-local coordinates, avoiding catastrophic cancellation.
    // We use the closest approach on the unrefracted ray to find a stable anchor point.
    float dist_to_center = length(cam_to_body);
    float bounding_radius_au = length(f_pos_local_au);
    float ray_shift_au = 0.0;

    vec3 cam_local_au;
    vec3 ring_cam_local_au;

    if (dist_to_center > bounding_radius_au * 4.0) {
        // Find closest approach of the unrefracted ray relative to the planet center
        float t_ca = -dot(f_pos_local_au, view_ray);
        vec3 closest_approach_au = f_pos_local_au + t_ca * view_ray;

        // Place precise origin 2.0 bounding radii in front of the closest approach
        float dist_from_f_pos = t_ca - 2.0 * bounding_radius_au;
        vec3 precise_origin_au = f_pos_local_au + dist_from_f_pos * view_ray;

        // Compute distance from the camera to this new origin
        float dist_to_f_pos = length(f_pos_local_au + cam_to_body);
        ray_shift_au = dist_to_f_pos + dist_from_f_pos;

        // Apply refraction shift laterally at this distance
        cam_local_au = precise_origin_au + (ray_shift_au - s_min_au) * (ray_dir - view_ray);
        ring_cam_local_au = precise_origin_au + (ray_shift_au - ring_s_min_au) * (ring_ray_dir - view_ray);
    } else {
        cam_local_au = ray_origin_au - planet_center_render;
        ring_cam_local_au = ring_ray_origin_au - planet_center_render;
    }

    vec3 cam_local = cam_local_au * u_au_to_km;
    vec3 ring_cam_local = ring_cam_local_au * u_au_to_km;

    vec3 f_pos_local = f_pos_local_au * u_au_to_km;

    vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
    vec3 cam_local_sph = toSphericalSpace(cam_local, u_pole_obl);

    vec2 s_atmo = raySphereIntersect(cam_local_sph, ray_dir_sph, u_atmo_radius_km);
    if (s_atmo.x > s_atmo.y) discard;

    vec2 s_planet = raySphereIntersect(cam_local_sph, ray_dir_sph, u_planet_clip_km);

    float s_start = max(0.0, s_atmo.x);
    float s_end = s_atmo.y;

    if (s_planet.x > 0.0 && s_planet.x < s_end) {
        s_end = s_planet.x;
    }

    vec3 frag_local = cam_local;

    float closest_s_ring = 1e10;

    for (int k = 0; k < u_num_ring_planes; k++) {
        vec3 ring_center_world_rel = u_ring_center[k];
        // Allow clipping against any ring plane in the system to support moon atmospheres correctly layering with host planet rings
        // if (length(ring_center_world_rel - u_body_offset) > 1e-4) continue;

        vec3 ring_center_local = (ring_center_world_rel - u_body_offset) * u_au_to_km;
        vec3 ring_normal = u_ring_normal[k];

        float denom = dot(ring_ray_dir, ring_normal);
        if (abs(denom) > 1e-8) {
            float s_ring = dot(ring_center_local - ring_cam_local, ring_normal) / denom;
            // ray_shift_au artificially pushes the ray origin forward to avoid precision loss.
            // If it pushes the origin past the ring plane, s_ring becomes negative.
            // We must accept it if it is still physically in front of the TRUE camera.
            if (s_ring + ray_shift_au * u_au_to_km > 0.0) {
                float ring_opacity = u_ring_params[k].z;
                if (ring_opacity > 1e-6) {
                    if (s_ring < closest_s_ring) {
                        closest_s_ring = s_ring;
                    }
                }
            }
        }
    }

    if (u_atmo_clip_mode == 1) {
        s_start = max(s_start, closest_s_ring);
    } else if (u_atmo_clip_mode == 2) {
        s_end = min(s_end, closest_s_ring);
    }

    for (int i = 0; i < u_num_active_casters; i++) {
        vec3 caster_pos = u_active_casters[i].xyz;
        float caster_r = u_active_casters[i].w * u_au_to_km;
        vec3 caster_local = (caster_pos - planet_center_render) * u_au_to_km;

        if (length(caster_local) < 1.0) continue;

        vec4 pole_obl = u_active_caster_poles_obl[i];
        vec3 origin_c = toSphericalSpace(frag_local - caster_local, pole_obl);
        vec3 dir_c = toSphericalSpace(ray_dir, pole_obl);

        vec2 s_c = raySphereIntersect(origin_c, dir_c, caster_r);
        if (s_c.x > 0.0 && s_c.x < s_end) {
            s_end = s_c.x;
        }
    }

    if (u_screen_res.x > 0.0 && u_screen_res.y > 0.0) {
        vec2 depth_uv = gl_FragCoord.xy / u_screen_res;
        
        // Manually bilinear filter the high-res depth texture to prevent perfectly vertical Moiré bands
        // caused by sub-pixel snapping when sampling a GL_NEAREST depth buffer at a lower resolution.
        vec2 high_res_size = vec2(textureSize(u_depth_texture, 0));
        vec2 texel_pos = depth_uv * high_res_size - 0.5;
        vec2 p = floor(texel_pos);
        vec2 f = fract(texel_pos);
        
        float d00 = texture(u_depth_texture, (p + vec2(0.5, 0.5)) / high_res_size).r;
        float d10 = texture(u_depth_texture, (p + vec2(1.5, 0.5)) / high_res_size).r;
        float d01 = texture(u_depth_texture, (p + vec2(0.5, 1.5)) / high_res_size).r;
        float d11 = texture(u_depth_texture, (p + vec2(1.5, 1.5)) / high_res_size).r;
        
        float log_depth = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
        if (log_depth < 0.99999) {
            float log_far_denom = log2(u_depth_C * u_far + 1.0);
            float scene_clip_z = (exp2(log_depth * log_far_denom) - 1.0) / u_depth_C;
            vec3 cam_fw = -vec3(view[0][2], view[1][2], view[2][2]);
            float cos_angle = dot(ray_dir, cam_fw);
            if (cos_angle > 1e-4) {
                float s_depth = (scene_clip_z * u_au_to_km) / cos_angle;
                s_depth -= ray_shift_au * u_au_to_km; // Convert from camera-relative to O_local_km relative

                // Float32 depth buffers and absolute distance calculations are heavily quantized at large distances.
                // Since the planet and rings already provide perfectly smooth analytical intersection bounds (s_end),
                // we ONLY clamp to the depth buffer if it represents a distinct non-analytical object (e.g. spacecraft)
                // clearly in front of the planet. This completely eliminates depth-buffer banding on the planet surface!
                float error_margin = dist_to_center * 2500.0;
                if (s_depth < s_end - error_margin) {
                    s_end = s_depth;
                }
            }
        }
    }

    if (s_start >= s_end) discard;

    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;       // Mie extinction
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;            // Mie scattering = omega_0 * extinction
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M); // Mie absorption
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;

    int steps = u_num_samples;
    if (u_atmo_adaptive_steps) {
        float ray_len = s_end - s_start;
        float atmo_thickness = max(1e-3, u_atmo_radius_km - u_planet_radius_km);

        float s_closest = clamp(-dot(cam_local_sph, ray_dir_sph), s_start, s_end);
        float min_altitude = length(cam_local_sph + s_closest * ray_dir_sph) - u_planet_radius_km;
        float alt_factor = mix(0.25, 1.0, clamp(exp(-max(0.0, min_altitude) / max(1e-3, u_h_rayleigh * 2.0)), 0.0, 1.0));

        float length_boost = clamp(sqrt(ray_len / atmo_thickness), 1.0, 4.0);
        float max_adaptive = max(float(u_num_samples), u_max_adaptive_steps > 0.0 ? u_max_adaptive_steps : 128.0);
        steps = int(clamp(float(u_num_samples) * alt_factor * length_boost, 4.0, max_adaptive));
    }
    float step_size = (s_end - s_start) / float(steps);
    vec3 scattered = vec3(0.0);
    vec3 final_transmittance = vec3(1.0);

    vec3 pole_dir_norm = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    float max_bend = clamp(2.0 * max(u_refractivity, 0.0) * sqrt(3.14159265359 * u_planet_radius_km / max(1e-6, u_h_rayleigh * 2.0)), 0.001, 0.05);

    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;

        vec3 frag_to_star = star_pos - planet_center_render;
        float dist_to_star_au = length(frag_to_star);
        vec3 L = frag_to_star / max(dist_to_star_au, 1e-6);

        vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
        float star_sin_lat = abs(dot(L, pole_dir));

        vec3 eq_color = u_stars_colors[s].rgb;
        float eq_lum = u_stars_colors[s].a;
        vec3 pole_color = u_stars_pole_colors[s].rgb;
        float pole_lum = u_stars_pole_colors[s].a;

        vec3 star_color = mix(eq_color, pole_color, star_sin_lat);
        float star_lum = mix(eq_lum, pole_lum, star_sin_lat);
        float star_radius_au = star_radius;
        float sin_star = star_radius_au / max(dist_to_star_au, star_radius_au + 1e-6);

        vec3 sun_pos_local = (star_pos - planet_center_render) * u_au_to_km;

        vec3 global_eclipse_shadow = vec3(1.0);
        vec3 end_eclipse_shadow = vec3(1.0);
        bool skip_volumetric_shadow = false;

        vec4 ring_s1_out = vec4(1e9);
        vec4 ring_s2_out = vec4(-1e9);
        vec4 ring_inner = vec4(0.0);
        vec4 ring_outer = vec4(1.0);
        vec4 ring_opac = vec4(0.0);
        vec4 ring_v_coord = vec4(0.0);
        vec4 ring_R_eff = vec4(0.0);
        vec3 ring_A_prime = vec3(0.0);
        vec3 ring_B = vec3(0.0);

        int opt_caster_count = 0;
        vec4 c_p0[4];
        vec4 c_p1[4];
        vec4 c_p2[4];
        vec4 c_p3[4];
        vec2 c_p4[4];

        float s_mid = (s_start + s_end) * 0.5;
        vec3 mid_pos = frag_local + s_mid * ray_dir;
        vec3 mid_render = mid_pos / u_au_to_km + planet_center_render;
        vec3 mid_to_star = star_pos - mid_render;
        float dist_mid_star = dist_to_star_au;
        vec3 L_mid = L; // Use fixed planet-relative light vector to prevent camera wobble
        float sr_start = star_radius / max(dist_mid_star, 1e-6);
        vec3 O = frag_local / u_au_to_km + planet_center_render;
        vec3 V = ray_dir; // Fix dimensional error: ray_dir is unit vector in km space

        if (u_atmo_quality > 0) {
            if (u_atmo_quality == 1) {
                global_eclipse_shadow = compute_shadow(mid_render, L_mid, dist_mid_star, planet_center_render, star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz);
                end_eclipse_shadow = global_eclipse_shadow;
            }

            if (u_atmo_quality == 2) {
                int ring_count = 0;
                uint processed_mask = 0u;
                for (int k = 0; k < u_num_ring_planes; k++) {
                    if ((u_ring_mask & (1u << k)) == 0u) continue;
                    if ((processed_mask & (1u << k)) != 0u) continue;

                    vec3 N = u_ring_normal[k];
                    vec3 C_km = u_ring_center[k] * u_au_to_km;
                    vec3 O_km = frag_local + planet_center_render * u_au_to_km;

                    float denom = dot(L_mid, N);
                    if (abs(denom) < 1e-8) continue;

                    float ra_km = dot(C_km - O_km, N) / denom;
                    float rb_km = -dot(ray_dir, N) / denom;

                    vec3 A_km = O_km + ra_km * L_mid;
                    vec3 B_km = ray_dir + rb_km * L_mid;
                    vec3 A_prime_km = A_km - C_km;

                    float qa = dot(B_km, B_km);
                    float qb = 2.0 * dot(A_prime_km, B_km);
                    float qc = dot(A_prime_km, A_prime_km);

                    float r_star_proj_km = abs((star_radius * u_au_to_km) * ra_km / max(dist_mid_star * u_au_to_km, 1e-6));
                    float R_eff_km = r_star_proj_km;
                    vec3 L_plane = L_mid - denom * N;
                    float L_plane_len = length(L_plane);
                    if (L_plane_len > 1e-5 && qa > 1e-8) {
                        vec3 L_proj = L_plane / L_plane_len;
                        vec3 T_vec = normalize(cross(N, L_mid));
                        float s_mid_ring = (-qb) / (2.0 * qa);
                        vec3 dir_radial = normalize(A_prime_km + s_mid_ring * B_km);
                        float cos_theta = dot(dir_radial, L_proj);
                        float sin_theta = dot(dir_radial, T_vec);
                        R_eff_km = r_star_proj_km * sqrt( pow(cos_theta / max(1e-6, abs(denom)), 2.0) + pow(sin_theta, 2.0) );
                    }

                    uint coplanar_mask = u_ring_coplanar_mask[k];
                    processed_mask |= coplanar_mask;
                    for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
                        if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
                        if (ring_count >= 4) continue;

                            float inner_r_km = u_ring_params[ring_idx].x * u_au_to_km;
                            float outer_r_km = u_ring_params[ring_idx].y * u_au_to_km;
                            float opacity = u_ring_params[ring_idx].z;
                            float scaled_tau = 1.0 / max(1e-4, abs(denom));

                            float ring_s_valid_min = -1e9;
                            float ring_s_valid_max = 1e9;
                            if (abs(rb_km) < 1e-8) {
                                if (ra_km <= 0.0) { ring_s_valid_min = 1e9; ring_s_valid_max = -1e9; }
                            } else {
                                float s_cross = -ra_km / rb_km;
                                if (rb_km < 0.0) {
                                    ring_s_valid_max = s_cross;
                                } else {
                                    ring_s_valid_min = s_cross;
                                }
                            }
                            if (ring_s_valid_min > ring_s_valid_max) continue;

                                if (ring_count == 0) {
                                    ring_s1_out.x = ring_s_valid_min; ring_s2_out.x = ring_s_valid_max;
                                    ring_inner.x = inner_r_km; ring_outer.x = outer_r_km; ring_opac.x = scaled_tau; ring_v_coord.x = (float(ring_idx) + 0.5) / 16.0;
                                    ring_A_prime = A_prime_km; ring_B = B_km; ring_R_eff.x = R_eff_km;
                                }
                                else if (ring_count == 1) {
                                    ring_s1_out.y = ring_s_valid_min; ring_s2_out.y = ring_s_valid_max;
                                    ring_inner.y = inner_r_km; ring_outer.y = outer_r_km; ring_opac.y = scaled_tau; ring_v_coord.y = (float(ring_idx) + 0.5) / 16.0;
                                    ring_R_eff.y = R_eff_km;
                                }
                                else if (ring_count == 2) {
                                    ring_s1_out.z = ring_s_valid_min; ring_s2_out.z = ring_s_valid_max;
                                    ring_inner.z = inner_r_km; ring_outer.z = outer_r_km; ring_opac.z = scaled_tau; ring_v_coord.z = (float(ring_idx) + 0.5) / 16.0;
                                    ring_R_eff.z = R_eff_km;
                                }
                                else if (ring_count == 3) {
                                    ring_s1_out.w = ring_s_valid_min; ring_s2_out.w = ring_s_valid_max;
                                    ring_inner.w = inner_r_km; ring_outer.w = outer_r_km; ring_opac.w = scaled_tau; ring_v_coord.w = (float(ring_idx) + 0.5) / 16.0;
                                    ring_R_eff.w = R_eff_km;
                                }

                                ring_count++;
                        }
                    }

                opt_caster_count = 0;
                for (int c = 0; c < u_num_active_casters && opt_caster_count < 4; c++) {
                    vec3 pos = u_active_casters[c].xyz;
                    float base_rad_au = u_active_casters[c].w;
                    float atmo_au = u_active_caster_atmos[c].w;
                    float caster_r_minor_au = u_active_caster_R_minor[c];
                    vec3 pole = u_active_caster_poles_obl[c].xyz;

                    vec3 D = (pos - planet_center_render) * u_au_to_km - frag_local;
                    float t0 = dot(D, L_mid);
                    float t1 = -dot(V, L_mid);

                    vec3 A = D - t0 * L_mid;
                    vec3 B = -V - t1 * L_mid;

                    float qa = dot(B, B);
                    float qb = 2.0 * dot(A, B);
                    float qc = dot(A, A);

                    float s_mid_local = (s_start + s_end) * 0.5;
                    vec3 perp_mid_km = A + s_mid_local * B;

                    float r_au = base_rad_au;
                    if (caster_r_minor_au < base_rad_au - 1e-5) {
                        r_au = get_oblate_radius(base_rad_au, caster_r_minor_au, pole, L_mid, perp_mid_km);
                    }
                    float r_km = r_au * u_au_to_km;
                    float atmo_km = atmo_au * u_au_to_km;
                    float eff_r_km = r_km + (atmo_km > 0.0 ? atmo_km * 4.0 : 0.0);

                    float directional_star_r_au = star_radius;
                    if (u_stars_poles_obl[s].w < star_radius - 1e-5) {
                        directional_star_r_au = get_oblate_radius(star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz, L_mid, perp_mid_km);
                    }
                    float alpha_star = directional_star_r_au / max(dist_mid_star, 1e-6);

                    float dist_approx_min_km = max(0.0, t0 + s_start * t1);
                    float dist_approx_max_km = max(0.0, t0 + s_end * t1);
                    float r_penumbra_max_km = eff_r_km + max(dist_approx_min_km, dist_approx_max_km) * alpha_star;

                    float s_valid_min = s_start;
                    float s_valid_max = s_end;

                    if (qa > 1e-8) {
                        float det = qb * qb - 4.0 * qa * (qc - r_penumbra_max_km * r_penumbra_max_km);
                        if (det < 0.0) continue;

                        float sqrt_det = sqrt(det);
                        float s_c1 = (-qb - sqrt_det) / (2.0 * qa);
                        float s_c2 = (-qb + sqrt_det) / (2.0 * qa);

                        s_valid_min = max(s_start, min(s_c1, s_c2));
                        s_valid_max = min(s_end, max(s_c1, s_c2));

                        if (s_valid_min > s_valid_max) continue;
                    } else {
                        if (qc > r_penumbra_max_km * r_penumbra_max_km) continue;
                    }

                    float s_mid_valid = (s_valid_min + s_valid_max) * 0.5;
                    float t_proj_mid = t0 + s_mid_valid * t1;
                    if (t_proj_mid <= 0.0) continue;

                    float perp_sq_mid = qa * s_mid_valid * s_mid_valid + qb * s_mid_valid + qc;
                    float dist_to_caster_mid = sqrt(max(0.0, perp_sq_mid) + t_proj_mid * t_proj_mid);
                    float inv_dist = 1.0 / max(dist_to_caster_mid, 1e-6);

                    float beta = eff_r_km * inv_dist;
                    float po = alpha_star + beta;
                    float pi = abs(beta - alpha_star);

                    float r_penumbra_sq = po * po * dist_to_caster_mid * dist_to_caster_mid;
                    float occ_mult = min(1.0, (beta * beta) / max(1e-9, alpha_star * alpha_star));

                    c_p0[opt_caster_count] = vec4(qa, qb, qc, r_penumbra_sq);
                    c_p1[opt_caster_count] = vec4(inv_dist, po, pi, occ_mult);
                    c_p2[opt_caster_count] = vec4(alpha_star, beta, u_active_max_bend[c], 1000.0 * min(beta, 0.05) * min(beta, 0.05));
                    c_p3[opt_caster_count] = vec4(u_active_caster_atmos[c].xyz, 0.0);
                    c_p4[opt_caster_count] = vec2(s_valid_min, s_valid_max);

                    opt_caster_count++;
                }

                if (opt_caster_count == 0 && ring_count == 0) skip_volumetric_shadow = true;
            }
        }

        vec3 sun_dir_local = normalize(sun_pos_local - mid_pos);
        vec3 sun_dir_sph_const = normalize(toSphericalSpace(sun_dir_local, u_pole_obl));

        float alpha_sun_local = asin(clamp(sin_star, 0.0, 0.9999));
        float cos_sun = cos(alpha_sun_local);
        float sin_sun = sin_star;

        // --- HOISTED CONSTANTS ---
        vec3 pole_dir_norm = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
        float sun_pole_dot = dot(sun_dir_sph_const, pole_dir_norm);
        float cosPhi = abs(sun_pole_dot);
        float tanPhi = cosPhi / sqrt(max(1.0 - cosPhi * cosPhi, 1e-4));
        float solstice_factor = clamp(tanPhi * 1.8, 0.0, 1.0);

        // Atmospheric refraction limit (max bending angle) for this body.
        // Derived from the surface refractivity (n_mix - 1) instead of a
        // hard-coded Earth-only 0.00029 constant, so Venus/Mars/Titan/gas
        // giants refract eclipses according to their own gas mixtures.
        float max_bend = clamp(2.0 * max(u_refractivity, 0.0) * sqrt(3.14159265359 * u_planet_radius_km / max(1e-6, u_h_rayleigh * 2.0)), 0.001, 0.05);
        float effective_star_rad = sin_star + max_bend;
        float cos_sun_eff = sqrt(max(0.0, 1.0 - effective_star_rad * effective_star_rad));

        float inv_h_rayleigh = 1.0 / max(1e-3, u_h_rayleigh);
        float inv_h_mie = 1.0 / max(1e-3, u_h_mie);
        float inv_ozone_width = 1.0 / max(u_ozone_width_km, 1e-3);

        float jitter = 0.5;
        if (u_stochastic_noise) {
            vec2 frame_offset = vec2(u_frame_counter * 5.588238, u_frame_counter * 13.91849);
            jitter = get_ign(gl_FragCoord.xy + frame_offset);
        }
        vec3 step_dir_sph = ray_dir_sph * step_size;
        vec3 current_pos_sph = cam_local_sph + (s_start + jitter * step_size) * ray_dir_sph;

        vec3 total_rayleigh = vec3(0.0);
        vec3 total_mie = vec3(0.0);
        vec3 total_ms = vec3(0.0);
        vec3 current_transmittance = vec3(1.0);

        vec3 total_rayleigh_ps = vec3(0.0);
        vec3 total_mie_ps = vec3(0.0);

        vec3 total_rayleigh_rs = vec3(0.0);
        vec3 total_mie_rs = vec3(0.0);

        float current_s = s_start + jitter * step_size;
        float t_lerp = jitter / float(steps);
        float t_step = 1.0 / float(steps);

        float inv_atmo_thickness = 1.0 / max(1e-4, u_atmo_radius_km - u_planet_radius_km);

        vec3 initial_pos_start = current_pos_sph - 0.5 * step_dir_sph;
        float h_prev = max(0.0, length(initial_pos_start) - u_planet_radius_km);
        float rho_R_prev = exp(-h_prev / u_h_rayleigh);
        float rho_M_prev = exp(-h_prev / u_h_mie);
        float rho_O_prev = exp(-pow((h_prev - u_ozone_peak_km) / max(u_ozone_width_km, 1e-3), 2.0));

        for (int i = 0; i < steps; i++) {
            float sample_len = length(current_pos_sph);
            float altitude = sample_len - u_planet_radius_km;

            // SpaceEngine Solstice Winter Model: Blue polar atmospheric scattering appears ONLY on the winter pole during solstice
            vec3 sample_dir_norm = current_pos_sph / max(sample_len, 1e-6);
            float sin_lat = abs(dot(sample_dir_norm, pole_dir_norm));
            float lat_factor = smoothstep(0.1, 0.7, sin_lat);

            float frag_pole_dot = dot(sample_dir_norm, pole_dir_norm);

            // Winter hemisphere condition: Sun and Sample point are on opposite sides of the equator
            float is_winter = step(sun_pole_dot * frag_pole_dot, 0.0);

            float winter_solstice_effect = lat_factor * is_winter * solstice_factor;

            float polar_haze_factor = mix(1.0, 0.05, winter_solstice_effect);
            vec3 polar_rayleigh_inscatter_boost = mix(vec3(1.0), vec3(0.65, 0.95, 2.5), winter_solstice_effect);

            float h1 = h_prev;
            float rho_R1 = rho_R_prev;
            float rho_M1 = rho_M_prev;
            float rho_O1 = rho_O_prev;

            float h_mid = max(0.0, altitude);

            vec3 pos_end = current_pos_sph + 0.5 * step_dir_sph;
            float h2 = max(0.0, length(pos_end) - u_planet_radius_km);

            float rho_R_mid = exp(-h_mid / u_h_rayleigh);
            float rho_R2 = exp(-h2 / u_h_rayleigh);
            float rho_R = (rho_R1 + 4.0 * rho_R_mid + rho_R2) * (1.0 / 6.0);

            float rho_M_mid = exp(-h_mid / u_h_mie);
            float rho_M2 = exp(-h2 / u_h_mie);
            float rho_M = (rho_M1 + 4.0 * rho_M_mid + rho_M2) * (1.0 / 6.0) * polar_haze_factor;

            float rho_O_mid = exp(-pow((h_mid - u_ozone_peak_km) / max(u_ozone_width_km, 1e-3), 2.0));
            float rho_O2 = exp(-pow((h2 - u_ozone_peak_km) / max(u_ozone_width_km, 1e-3), 2.0));
            float rho_O = (rho_O1 + 4.0 * rho_O_mid + rho_O2) * (1.0 / 6.0);

            h_prev = h2;
            rho_R_prev = rho_R2;
            rho_M_prev = rho_M2;
            rho_O_prev = rho_O2;

            // Extinction uses physical beta_R to prevent artificial limb color fringing
            vec3 step_extinction = beta_R * rho_R + beta_M * rho_M + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;
            vec3 step_transmittance = exp(-step_extinction * step_size);
            vec3 int_factor = (vec3(1.0) - step_transmittance) / max(step_extinction, 1e-6);

            vec3 sun_dir_sph = sun_dir_sph_const;
            float light_cos_theta = dot(current_pos_sph, sun_dir_sph) / sample_len;
            float h_norm = clamp(altitude * inv_atmo_thickness, 0.0, 1.0);

            float sin_planet = u_planet_radius_km / max(sample_len, u_planet_radius_km + 0.01);
            float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));

            float cos_outer = cos_planet * cos_sun_eff - sin_planet * effective_star_rad;
            float cos_inner = cos_planet * cos_sun_eff + sin_planet * effective_star_rad;

            float neg_light_cos = -light_cos_theta;
            float vis_fraction = 1.0;

            float max_occ = min(1.0, (sin_planet * sin_planet) / max(1e-9, effective_star_rad * effective_star_rad));
            float min_vis = 1.0 - max_occ;

            if (neg_light_cos > cos_inner) {
                vis_fraction = min_vis;
            } else if (neg_light_cos > cos_outer) {
                float x_vis = (cos_planet * cos_sun_eff - neg_light_cos) / max(1e-7, sin_planet * effective_star_rad);
                float s = smoothstep(-1.0, 1.0, x_vis);
                vis_fraction = mix(min_vis, 1.0, s);
            }

            float sin_Z = sqrt(max(0.0, 1.0 - light_cos_theta * light_cos_theta));
            float exact_disc_top_cos = light_cos_theta * cos_sun_eff + sin_Z * effective_star_rad;
            float exact_disc_bot_cos = light_cos_theta * cos_sun_eff - sin_Z * effective_star_rad;

            float disc_top_cos = exact_disc_top_cos;
            if (light_cos_theta > cos_sun_eff) {
                disc_top_cos = 1.0;
            }
            float disc_bot_cos = max(exact_disc_bot_cos, -cos_planet + 1e-5);
            float effective_cos = max(-cos_planet + 1e-5, (disc_top_cos + disc_bot_cos) * 0.5);

            float v_norm = sqrt(h_norm);
            vec3 transmittance_to_sun = get_transmittance_precomputed(v_norm, effective_cos);

            vec3 sample_shadow = global_eclipse_shadow;
            if (u_atmo_quality == 2 && !skip_volumetric_shadow) {
                sample_shadow = vec3(1.0);

                float d = length(ring_A_prime + current_s * ring_B);
                vec4 s_vec = vec4(current_s);
                vec4 mask = step(ring_s1_out, s_vec) * step(s_vec, ring_s2_out);

                if (any(greaterThan(mask, vec4(0.0)))) {
                    vec4 o_min = max(ring_inner, vec4(d) - ring_R_eff);
                    vec4 o_max = min(ring_outer, vec4(d) + ring_R_eff);
                    vec4 valid = step(o_min, o_max) * mask;

                    if (any(greaterThan(valid, vec4(0.0)))) {
                        vec4 r_eff_inv = 1.0 / max(vec4(1e-9), ring_R_eff);
                        vec4 v_min = clamp((o_min - vec4(d)) * r_eff_inv, -1.0, 1.0);
                        vec4 v_max = clamp((o_max - vec4(d)) * r_eff_inv, -1.0, 1.0);
                        vec4 frac = max(vec4(0.0), smoothstep(-1.0, 1.0, v_max) - smoothstep(-1.0, 1.0, v_min));
                        vec4 p_mid = clamp(((o_min + o_max) * 0.5 - ring_inner) / max(vec4(1e-6), ring_outer - ring_inner), 0.0, 1.0);

                        vec4 sh_mult = vec4(1.0);
                        if (valid.x > 0.0) {
                            float a_x = textureLod(u_ring_gradients, vec2(p_mid.x, ring_v_coord.x), 0.0).a;
                            float tau_x = -log(max(1e-6, 1.0 - a_x));
                            sh_mult.x = 1.0 - frac.x * (1.0 - exp(-ring_opac.x * tau_x));
                        }
                        if (valid.y > 0.0) {
                            float a_y = textureLod(u_ring_gradients, vec2(p_mid.y, ring_v_coord.y), 0.0).a;
                            float tau_y = -log(max(1e-6, 1.0 - a_y));
                            sh_mult.y = 1.0 - frac.y * (1.0 - exp(-ring_opac.y * tau_y));
                        }
                        if (valid.z > 0.0) {
                            float a_z = textureLod(u_ring_gradients, vec2(p_mid.z, ring_v_coord.z), 0.0).a;
                            float tau_z = -log(max(1e-6, 1.0 - a_z));
                            sh_mult.z = 1.0 - frac.z * (1.0 - exp(-ring_opac.z * tau_z));
                        }
                        if (valid.w > 0.0) {
                            float a_w = textureLod(u_ring_gradients, vec2(p_mid.w, ring_v_coord.w), 0.0).a;
                            float tau_w = -log(max(1e-6, 1.0 - a_w));
                            sh_mult.w = 1.0 - frac.w * (1.0 - exp(-ring_opac.w * tau_w));
                        }

                        sample_shadow *= sh_mult.x * sh_mult.y * sh_mult.z * sh_mult.w;
                    }
                }

                for (int c = 0; c < 4; c++) {
                    if (c >= opt_caster_count) break;

                    vec2 s_bounds = c_p4[c];
                    if (current_s >= s_bounds.x && current_s <= s_bounds.y) {
                        vec4 p0 = c_p0[c];
                        float perp_sq = p0.x * current_s * current_s + p0.y * current_s + p0.z;
                        if (perp_sq < p0.w) {
                            vec4 p1 = c_p1[c];
                            float gamma = sqrt(max(0.0, perp_sq)) * p1.x;
                            vec4 p2 = c_p2[c];

                            vec3 sh = casterShadowTerm(p2.x, p2.y, gamma, p1.y, p1.z, p2.z, c_p3[c].xyz);
                            sample_shadow *= sh;
                        }
                    }
                }
            }

            vec3 sample_attenuation = current_transmittance * transmittance_to_sun * sample_shadow * vis_fraction;

            total_rayleigh += (rho_R * polar_rayleigh_inscatter_boost) * sample_attenuation * int_factor;
            total_mie      += rho_M * sample_attenuation * int_factor;

            float sun_cos_zenith = light_cos_theta;
            float ms_u = 0.5 + 0.5 * sign(sun_cos_zenith) * sqrt(abs(sun_cos_zenith));
            float ms_v = v_norm;
            vec2 ms_uv = vec2(ms_u, ms_v);
            vec3 psi = textureLod(u_multi_scatter_lut, ms_uv, 0.0).rgb;

            vec3 ms_shadow = sample_shadow;
            total_ms += (beta_R * rho_R + beta_M * rho_M) * psi * current_transmittance * ms_shadow * int_factor;

            if (u_planetshine_enabled) {
                float light_cos_theta_ps = dot(current_pos_sph, body_planetshine_dir) / sample_len;
                float vis_fraction_ps = smoothstep(-cos_planet - 0.05, -cos_planet + 0.05, light_cos_theta_ps);
                vec3 transmittance_to_ps = get_transmittance_precomputed(v_norm, light_cos_theta_ps);
                vec3 ps_attenuation = current_transmittance * transmittance_to_ps * vis_fraction_ps;
                total_rayleigh_ps += rho_R * ps_attenuation * int_factor;
                total_mie_ps      += rho_M * ps_attenuation * int_factor;
            }

            if (u_ringshine_enabled) {
                vec3 rs_attenuation = current_transmittance;
                total_rayleigh_rs += rho_R * rs_attenuation * int_factor;
                total_mie_rs      += rho_M * rs_attenuation * int_factor;
            }

            current_transmittance *= step_transmittance;

            current_pos_sph += step_dir_sph;
            current_s += step_size;
            t_lerp += t_step;
        }

        float cos_theta = dot(ray_dir, L_mid);
        float phase_R = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);

        float g_val = clamp(u_mie_g, 0.0, 0.88);
        float g2_val = g_val * g_val;
        float phase_M_scalar = (3.0 / (8.0 * PI)) * ((1.0 - g2_val) * (1.0 + cos_theta * cos_theta))
                              / ((2.0 + g2_val) * pow(max(1e-4, 1.0 + g2_val - 2.0 * g_val * cos_theta), 1.5));
        vec3 phase_M = vec3(phase_M_scalar);

        float irradiance = star_lum / max(dist_to_star_au * dist_to_star_au, 1e-8);

        float atmo_sun_intensity = u_sun_intensity * PI;
        scattered += star_color * atmo_sun_intensity * irradiance * (
            phase_R * beta_R * total_rayleigh +
            phase_M * beta_M * total_mie +
            total_ms
        );

        if (u_planetshine_enabled && dot(body_planetshine_color, body_planetshine_color) > 1e-12) {
            float cos_theta_ps = dot(ray_dir, body_planetshine_dir);
            float phase_R_ps = (3.0 / (16.0 * PI)) * (1.0 + cos_theta_ps * cos_theta_ps);
            float phase_M_ps_scalar = (3.0 / (8.0 * PI)) * ((1.0 - g2_val) * (1.0 + cos_theta_ps * cos_theta_ps))
                                      / ((2.0 + g2_val) * pow(max(1e-4, 1.0 + g2_val - 2.0 * g_val * cos_theta_ps), 1.5));
            vec3 phase_M_ps = vec3(phase_M_ps_scalar);
            scattered += body_planetshine_color * (
                phase_R_ps * beta_R * total_rayleigh_ps +
                phase_M_ps * beta_M * total_mie_ps
            );
        }

        if (u_ringshine_enabled && u_num_ring_planes > 0) {
            vec3 ringshine_irradiance = vec3(0.0);
            vec3 N = normalize(mid_pos);
            vec3 L_dir = L_mid;

            vec3 ring_normal = u_ring_normal[0];
            float sun_elevation = dot(L_dir, ring_normal);
            float sin_sun_elev = clamp(abs(sun_elevation), 1e-4, 1.0);
            float cos_sun_elev = sqrt(1.0 - sin_sun_elev * sin_sun_elev);

            vec3 antiL = -L_dir;
            vec3 antiL_eq_raw = antiL - ring_normal * dot(antiL, ring_normal);
            float len_antiL_eq = length(antiL_eq_raw);
            vec3 antiL_eq = len_antiL_eq > 1e-5 ? antiL_eq_raw / len_antiL_eq : vec3(-1.0, 0.0, 0.0);

            vec3 N_eq_raw = N - ring_normal * dot(N, ring_normal);
            float len_N_eq = length(N_eq_raw);
            vec3 N_eq = len_N_eq > 1e-5 ? N_eq_raw / len_N_eq : vec3(1.0, 0.0, 0.0);

            vec3 cross_rel = cross(antiL_eq, N_eq);
            float sin_rel = dot(cross_rel, ring_normal);
            float cos_rel = clamp(dot(N_eq, antiL_eq), -1.0, 1.0);
            float phi_center = atan(sin_rel, cos_rel);

            float frag_elevation = dot(N, ring_normal);
            float sin_lat = clamp(abs(frag_elevation), 0.001, 0.999);
            float same_hemisphere = sun_elevation * frag_elevation;
            float same_hemi_t = smoothstep(-0.02, 0.02, same_hemisphere);

            float NdotL = dot(N, L_dir);
            float day_face = smoothstep(0.0, 0.1, NdotL) * max(0.0, NdotL);
            float noon_fade = mix(1.0, 0.4, day_face);
            float face_multiplier = mix(1.0, 1.0 * noon_fade, same_hemi_t);

            float host_radius = u_planet_radius_km / u_au_to_km;
            int band_count = clamp(u_ringshine_band_count, 4, 128);

            for (int j = 0; j < 1; j++) {
                if ((u_ring_mask & (1u << j)) == 0u) continue;
                float x_prime = phi_center / PI;
                float phi_uv = sign(x_prime) * pow(abs(x_prime), 0.666666667) * 0.5 + 0.5;

                float y_prime = frag_elevation;
                float elev_uv = sign(y_prime) * pow(abs(y_prime), 0.666666667) * 0.5 + 0.5;

                vec2 map_uv = vec2(phi_uv, (float(j) + elev_uv) / 16.0);
                ringshine_irradiance += texture(u_ringshine_map, map_uv).rgb;
            }
            float opacity = u_ring_params[0].z;
            ringshine_irradiance *= (sin_sun_elev * face_multiplier * 0.318309886);

            float ambient_phase = 1.0 / (4.0 * PI);
            scattered += star_color * atmo_sun_intensity * irradiance * ringshine_irradiance * (
                ambient_phase * beta_R * total_rayleigh_rs +
                ambient_phase * beta_M * total_mie_rs
            );
        }

        final_transmittance = current_transmittance;
    }

    vec3 transmittance = final_transmittance;

    if (u_hdr_enabled) {
        scattered *= u_exposure;
    }

    out_scattered = vec4(scattered, 1.0);
    out_transmittance = vec4(clamp(transmittance, vec3(0.0), vec3(1.0)), 1.0);
}
