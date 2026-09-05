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
    vec4 u_stars_pos_radius[MAX_STARS];
    vec4 u_stars_colors[MAX_STARS];
    vec4 u_stars_poles_obl[MAX_STARS];
    vec4 u_stars_pole_colors[MAX_STARS];
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
    vec4 u_caster_ozone[MAX_CASTERS];
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
    vec3  u_mie_albedo;
    float u_refractivity;
    vec4  u_active_casters[8];
    vec4  u_active_caster_poles_obl[8];
    float u_active_caster_R_minor[8];
    vec4  u_active_caster_atmos[8];
    float u_active_max_bend[8];
    vec4  u_active_caster_ozone[8];
    float u_ozone_peak_km;
    float u_ozone_width_km;
    float u_planet_clip_km;
    float u_max_adaptive_steps;
    vec4  u_precomp_opt;
    vec4  u_precomp_mie;
    vec4  u_star_color_irrad[4];
    vec4  u_star_dir_sph_eff[4];
    vec4  u_star_pos_local[4];
    vec4  u_star_solstice[4];
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
uniform sampler2D u_transmittance_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform sampler2D u_depth_texture;
uniform vec2 u_screen_res;

uniform bool u_temporal_accum;
uniform bool u_history_valid;
uniform float u_temporal_alpha;
uniform float u_prev_exposure;
uniform mat4 u_prev_view_proj;
uniform vec3 u_prev_body_offset;
uniform sampler2D u_history_godrays;

layout(location = 0) out vec4 out_godrays;

vec3 toSphericalSpace(vec3 v, vec4 pole_obl) {
    if (pole_obl.w < 1e-5) return v;
    vec3 P = length(pole_obl.xyz) > 1e-4 ? normalize(pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    float dotP = dot(v, P);
    vec3 v_perp = v - dotP * P;
    return v_perp + (dotP / (1.0 - pole_obl.w)) * P;
}

vec2 raySphereIntersect(vec3 r0, vec3 rd, float sr) {
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, r0);
    float c = dot(r0, r0) - (sr * sr);
    float d = (b * b) - 4.0 * a * c;
    if (d < 0.0) return vec2(1e5, -1e5);
    float sqrt_d = sqrt(d);
    return vec2((-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a));
}

vec3 get_transmittance_precomputed(float v_norm, float sun_cos_zenith) {
    float u_norm = 0.5 + 0.5 * sign(sun_cos_zenith) * sqrt(abs(sun_cos_zenith));
    vec2 lut_uv = vec2(u_norm, v_norm);
    return textureLod(u_transmittance_lut, lut_uv, 0.0).rgb;
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

        if (gamma >= penumbra_inner) {
            float z_direct_km = (gamma - penumbra_inner) * dist_km;
            if (z_direct_km < 60.0) {
                vec3 tau_R_d = atmo_param.xyz * exp(-z_direct_km / H_scale);
                float z_diff_d = (z_direct_km - z_peak) / sigma_z;
                vec3 tau_O3_d = ozone_param.xyz * exp(-0.5 * z_diff_d * z_diff_d);
                vec3 T_direct = exp(-(tau_R_d + tau_O3_d));
                vec3 pen_filter = clamp(T_direct + vec3(clamp((z_direct_km - 40.0) / 15.0, 0.0, 1.0)), 0.0, 1.0);
                sh = geom_sh * pen_filter;
            }
        }
        float req_bend = beta - gamma - alpha * 0.8;
        if (req_bend <= max_bend) {
            float atmo_depth = clamp(max(0.0, req_bend) / max_bend, 0.0, 1.0);
            float z_km = -H_scale * log(max(atmo_depth, 1e-5));
            vec3 tau_R = atmo_param.xyz * atmo_depth;
            float z_diff = (z_km - z_peak) / sigma_z;
            vec3 tau_O3 = ozone_param.xyz * exp(-0.5 * z_diff * z_diff);
            vec3 tau_total = tau_R + tau_O3;
            vec3 atmo_transmittance = exp(-tau_total);
            float ring_intensity = (2.0 * H_scale) / max(alpha * dist_km, 1e-9);
            float body_surface_fade = smoothstep(1.0, 0.75, atmo_depth);
            float refraction_intensity = ring_intensity * (1.0 - atmo_depth * 0.7) * body_surface_fade;
            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            sh += atmo_transmittance * refraction_intensity * atmo_blend;
        }
    }
    return clamp(sh, vec3(0.0), vec3(1.0));
}

float map_t_to_s(float t, float s_start, float s_end, float s_min, float p) {
    float L = s_end - s_start;
    if (L < 1e-4) return s_start;
    if (abs(p - 1.0) < 0.01) return s_start + t * L;
    float t_min = clamp((s_min - s_start) / L, 0.0, 1.0);
    if (t <= t_min) {
        if (t_min < 1e-5) return s_min;
        float u = t / t_min;
        return s_min - (s_min - s_start) * pow(1.0 - u, p);
    } else {
        if (t_min > 0.99999) return s_min;
        float u = (t - t_min) / (1.0 - t_min);
        return s_min + (s_end - s_min) * pow(u, p);
    }
}

void main() {
    if (f_clip_z < 0.0) discard;
    vec3 planet_center_render = u_body_offset;
    vec3 ray_origin_au = u_camera_pos;
    vec3 cam_to_body = u_camera_pos - planet_center_render;

    vec3 view_ray = normalize(f_local_pos);
    vec3 ray_dir = view_ray;

    // Early exit if body has neither rings nor shadow casters
    u_caster_mask_lo = floatBitsToUint(instances[u_body_idx].z);
    u_caster_mask_hi = floatBitsToUint(instances[u_body_idx].w);
    u_ring_mask = floatBitsToUint(instances[u_body_idx].y);
    if (u_num_active_casters == 0 && (u_ring_mask == 0u || u_num_ring_planes == 0)) discard;

    vec3 f_pos_local_au = f_local_pos * u_atmo_radius_au;
    float dist_to_center = length(cam_to_body);
    float bounding_radius_au = length(f_pos_local_au);
    float ray_shift_au = 0.0;

    vec3 cam_local_au = ray_origin_au - planet_center_render;
    vec3 cam_local = cam_local_au * u_au_to_km;
    vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
    vec3 cam_local_sph = toSphericalSpace(cam_local, u_pole_obl);

    vec2 s_atmo = raySphereIntersect(cam_local_sph, ray_dir_sph, u_atmo_radius_km);
    if (s_atmo.x > s_atmo.y) discard;

    vec2 s_planet = raySphereIntersect(cam_local_sph, ray_dir_sph, u_planet_clip_km);
    float s_start = max(0.0, s_atmo.x);
    float s_end = s_atmo.y;

    bool hits_surface = false;
    if (s_planet.x > 0.0 && s_planet.x < s_end) {
        s_end = s_planet.x;
        hits_surface = true;
    }

    vec3 frag_local = cam_local;
    float closest_s_ring = 1e10;
    if (u_atmo_clip_mode != 0 && u_num_ring_planes > 0) {
        for (int k = 0; k < u_num_ring_planes; k++) {
            vec3 ring_center_world_rel = u_ring_center[k];
            vec3 ring_center_local = (ring_center_world_rel - u_body_offset) * u_au_to_km;
            vec3 ring_normal = u_ring_normal[k];
            float denom = dot(ray_dir, ring_normal);
            if (abs(denom) > 1e-8) {
                float s_ring = dot(ring_center_local - cam_local, ring_normal) / denom;
                if (s_ring > 0.0) {
                    float ring_opacity = u_ring_params[k].z;
                    if (ring_opacity > 1e-6) {
                        vec3 hit_pt = cam_local + s_ring * ray_dir;
                        float hit_r = length(hit_pt - ring_center_local);
                        float inner_r_km = u_ring_params[k].x * u_au_to_km;
                        float outer_r_km = u_ring_params[k].y * u_au_to_km;
                        if (hit_r >= inner_r_km && hit_r <= outer_r_km) {
                            if (s_ring < closest_s_ring) closest_s_ring = s_ring;
                        }
                    }
                }
            }
        }
    }

    if (u_atmo_clip_mode == 1) s_start = max(s_start, closest_s_ring);
    else if (u_atmo_clip_mode == 2) s_end = min(s_end, closest_s_ring);

    if (u_screen_res.x > 0.0 && u_screen_res.y > 0.0) {
        vec2 depth_uv = gl_FragCoord.xy / u_screen_res;
        float log_depth = texture(u_depth_texture, depth_uv).r;
        if (log_depth < 0.99999) {
            float log_far_denom = log2(u_depth_C * u_far + 1.0);
            float scene_clip_z = (exp2(log_depth * log_far_denom) - 1.0) / u_depth_C;
            vec3 cam_fw = -vec3(view[0][2], view[1][2], view[2][2]);
            float cos_angle = dot(ray_dir, cam_fw);
            if (cos_angle > 1e-4) {
                float s_depth = (scene_clip_z * u_au_to_km) / cos_angle;
                float error_margin = dist_to_center * 2500.0;
                if (s_depth < s_end - error_margin) s_end = s_depth;
            }
        }
    }

    if (s_start >= s_end) discard;

    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M);
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;

    float ray_len = s_end - s_start;
    float s_closest = clamp(-dot(cam_local_sph, ray_dir_sph), s_start, s_end);
    float min_altitude = max(0.0, length(cam_local_sph + s_closest * ray_dir_sph) - u_planet_radius_km);

    vec3 pole_dir_norm = length(u_pole_obl.xyz) > 1e-4 ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    float inv_h_rayleigh = u_precomp_opt.x;
    float inv_h_mie = u_precomp_opt.y;
    float inv_ozone_width = u_precomp_opt.z;

    float jitter = 0.5;
    if (u_stochastic_noise) {
        float base_ign = get_ign(gl_FragCoord.xy);
        jitter = fract(base_ign + float(int(u_frame_counter) % 16) * 0.6180339887);
    }

    // Step allocation for dedicated godray pass: 24 to 48 steps is ideal
    int steps = clamp(u_num_samples, 20, 48);

    vec3 total_shadowed_rayleigh = vec3(0.0);
    vec3 total_shadowed_mie = vec3(0.0);
    vec3 current_transmittance = vec3(1.0);

    vec3 weighted_pos = vec3(0.0);
    float total_weight = 0.0;

    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = (s < 4) ? u_star_solstice[s].w : u_stars_pos_radius[s].w;
        float dist_mid_star = (s < 4) ? u_star_solstice[s].z : length(star_pos - planet_center_render);
        float sin_star = (s < 4) ? u_star_color_irrad[s].w : (star_radius / max(dist_mid_star, star_radius + 1e-6));
        vec3 sun_pos_local = (s < 4) ? u_star_pos_local[s].xyz : ((star_pos - planet_center_render) * u_au_to_km);
        float effective_star_rad = (s < 4) ? u_star_pos_local[s].w : (sin_star + u_precomp_opt.w);
        vec3 sun_dir_sph_const = (s < 4) ? u_star_dir_sph_eff[s].xyz : normalize(toSphericalSpace(normalize(sun_pos_local), u_pole_obl));
        float cos_sun_eff = (s < 4) ? u_star_dir_sph_eff[s].w : sqrt(max(0.0, 1.0 - effective_star_rad * effective_star_rad));
        float solstice_factor = (s < 4) ? u_star_solstice[s].x : 0.0;
        float sun_pole_dot = (s < 4) ? u_star_solstice[s].y : 0.0;

        vec3 L = (length(sun_pos_local) > 1e-6) ? normalize(sun_pos_local) : vec3(0.0, 1.0, 0.0);
        vec3 L_mid = L;

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
        int ring_count = 0;
        vec4 c_p0[4], c_p1[4], c_p2[4], c_p3[4], c_p5[4];
        vec2 c_p4[4];

        if (u_num_ring_planes > 0 && u_ring_mask != 0u) {
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
                float r_star_proj_km = abs((star_radius * u_au_to_km) * ra_km / max(dist_mid_star * u_au_to_km, 1e-6));
                float R_eff_km = r_star_proj_km;
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
                        if (rb_km < 0.0) ring_s_valid_max = s_cross;
                        else ring_s_valid_min = s_cross;
                    }
                    if (ring_s_valid_min > ring_s_valid_max) continue;
                    if (ring_count == 0) {
                        ring_s1_out.x = ring_s_valid_min; ring_s2_out.x = ring_s_valid_max;
                        ring_inner.x = inner_r_km; ring_outer.x = outer_r_km; ring_opac.x = scaled_tau; ring_v_coord.x = (float(ring_idx) + 0.5) / 16.0;
                        ring_A_prime = A_prime_km; ring_B = B_km; ring_R_eff.x = R_eff_km;
                    } else if (ring_count == 1) {
                        ring_s1_out.y = ring_s_valid_min; ring_s2_out.y = ring_s_valid_max;
                        ring_inner.y = inner_r_km; ring_outer.y = outer_r_km; ring_opac.y = scaled_tau; ring_v_coord.y = (float(ring_idx) + 0.5) / 16.0;
                        ring_R_eff.y = R_eff_km;
                    }
                    ring_count++;
                }
            }
        }

        if (u_num_active_casters > 0) {
            for (int c = 0; c < 4; c++) {
                if (c >= u_num_active_casters) break;
                vec3 caster_pos_rel = u_active_casters[c].xyz - planet_center_render;
                float caster_r_km = u_active_casters[c].w * u_au_to_km;
                float eff_r_km = caster_r_km;
                vec3 C_caster = caster_pos_rel * u_au_to_km - frag_local;
                float t0 = dot(C_caster, L_mid);
                float t1 = -dot(ray_dir, L_mid);
                vec3 A_caster = C_caster - t0 * L_mid;
                vec3 B_caster = -ray_dir - t1 * L_mid;
                float qa = dot(B_caster, B_caster);
                float qb = -2.0 * dot(A_caster, B_caster);
                float qc = dot(A_caster, A_caster);
                float directional_star_r_au = (s < 4) ? u_star_solstice[s].w : u_stars_pos_radius[s].w;
                float alpha_star = directional_star_r_au / max(dist_mid_star, 1e-6);
                float dist_approx_min_km = max(0.0, t0 + s_start * t1);
                float dist_approx_max_km = max(0.0, t0 + s_end * t1);
                float r_penumbra_max_km = eff_r_km + max(dist_approx_min_km, dist_approx_max_km) * alpha_star;
                float s_valid_min = s_start, s_valid_max = s_end;
                if (qa > 1e-8) {
                    float det = qb * qb - 4.0 * qa * (qc - r_penumbra_max_km * r_penumbra_max_km);
                    if (det < 0.0) continue;
                    float sqrt_det = sqrt(det);
                    float s_c1 = (-qb - sqrt_det) / (2.0 * qa);
                    float s_c2 = (-qb + sqrt_det) / (2.0 * qa);
                    s_valid_min = max(s_start, min(s_c1, s_c2));
                    s_valid_max = min(s_end, max(s_c1, s_c2));
                    if (s_valid_min > s_valid_max) continue;
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
                c_p2[opt_caster_count] = vec4(alpha_star, beta, u_active_max_bend[c], dist_to_caster_mid / u_au_to_km);
                c_p3[opt_caster_count] = u_active_caster_atmos[c];
                c_p4[opt_caster_count] = vec2(s_valid_min, s_valid_max);
                c_p5[opt_caster_count] = u_active_caster_ozone[c];
                opt_caster_count++;
            }
        }

        if (opt_caster_count == 0 && ring_count == 0) discard;

        vec3 mid_pos = cam_local + (s_start + s_end) * 0.5 * ray_dir;
        vec3 mid_pos_sph = toSphericalSpace(mid_pos, u_pole_obl);
        vec3 mid_dir_norm = normalize(mid_pos_sph);
        float mid_sin_lat = abs(dot(mid_dir_norm, pole_dir_norm));
        float lat_factor = smoothstep(0.1, 0.7, mid_sin_lat);
        float frag_pole_dot = dot(mid_dir_norm, pole_dir_norm);
        float is_winter = step(sun_pole_dot * frag_pole_dot, 0.0);
        float has_rings = (u_ring_mask != 0u) ? 1.0 : 0.0;
        float winter_solstice_effect = lat_factor * is_winter * solstice_factor * has_rings;
        vec3 polar_rayleigh_inscatter_boost = mix(vec3(1.0), vec3(0.65, 0.95, 2.5), winter_solstice_effect);

        float inv_atmo_thickness = 1.0 / max(1e-4, u_atmo_radius_km - u_planet_radius_km);

        for (int i = 0; i < steps; i++) {
            float t0 = float(i) / float(steps);
            float t1 = float(i + 1) / float(steps);
            float tj = (float(i) + jitter) / float(steps);

            float s0 = s_start + t0 * ray_len;
            float s1 = s_start + t1 * ray_len;
            float current_s = s_start + tj * ray_len;
            float step_size = max(1e-4, s1 - s0);

            vec3 current_pos_sph = cam_local_sph + current_s * ray_dir_sph;
            float sample_len = length(current_pos_sph);
            float altitude = max(0.0, sample_len - u_planet_radius_km);

            float rho_R = exp(-altitude * inv_h_rayleigh);
            float rho_M = exp(-altitude * inv_h_mie);
            float t_ozone = (altitude - u_ozone_peak_km) * inv_ozone_width;
            float rho_O = exp(-(t_ozone * t_ozone));

            vec3 step_extinction = beta_R * rho_R + beta_M * rho_M + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;
            vec3 step_transmittance = exp(-step_extinction * step_size);
            vec3 int_factor = (vec3(1.0) - step_transmittance) / max(step_extinction, 1e-6);

            float light_cos_theta = dot(current_pos_sph, sun_dir_sph_const) / sample_len;
            float h_norm = clamp(altitude * inv_atmo_thickness, 0.0, 1.0);
            float sin_planet = u_planet_radius_km / max(sample_len, u_planet_radius_km + 0.01);
            float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));
            float neg_light_cos = -light_cos_theta;
            float vis_fraction = 1.0;
            if (neg_light_cos > cos_planet) {
                vis_fraction = 0.0;
            } else {
                float effective_sin = effective_star_rad;
                float x_vis = (cos_planet * cos_sun_eff - neg_light_cos) / max(1e-7, sin_planet * effective_sin);
                vis_fraction = smoothstep(-1.0, 1.0, x_vis);
            }

            float v_norm = sqrt(h_norm);
            vec3 transmittance_to_sun = (vis_fraction > 1e-4) ? get_transmittance_precomputed(v_norm, light_cos_theta) : vec3(0.0);

            vec3 sample_shadow = vec3(1.0);
            if (ring_count > 0) {
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
                        sample_shadow *= sh_mult.x * sh_mult.y;
                    }
                }
            }

            if (opt_caster_count > 0) {
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
                            vec3 sh = casterShadowTerm(p2.x, p2.y, gamma, p1.y, p1.z, p2.z, c_p3[c], c_p5[c], c_p3[c].w, p2.w);
                            sample_shadow *= sh;
                        }
                    }
                }
            }

            vec3 shadow_deficit = clamp(vec3(1.0) - sample_shadow, vec3(0.0), vec3(1.0));
            if (any(greaterThan(shadow_deficit, vec3(1e-4)))) {
                vec3 sample_attenuation = current_transmittance * transmittance_to_sun * vis_fraction * shadow_deficit;
                total_shadowed_rayleigh += (rho_R * polar_rayleigh_inscatter_boost) * sample_attenuation * int_factor;
                total_shadowed_mie += rho_M * sample_attenuation * int_factor;

                float step_w = length(sample_attenuation * (rho_R + rho_M));
                vec3 step_world_pos = u_camera_pos + (current_s / u_au_to_km) * ray_dir;
                weighted_pos += step_w * step_world_pos;
                total_weight += step_w;
            }

            current_transmittance *= step_transmittance;
        }

        vec3 star_color = (s < 4) ? u_star_color_irrad[s].rgb : (u_stars_colors[s].rgb * u_stars_colors[s].w * u_sun_intensity);
        float star_combined_intensity = (s < 4) ? 1.0 : (u_stars_colors[s].w * u_sun_intensity);

        float cos_theta = dot(ray_dir, L);
        float phase_R = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
        float phase_M_scalar = u_precomp_mie.x * (1.0 + cos_theta * cos_theta)
            / pow(max(1e-4, u_precomp_mie.y - u_precomp_mie.z * cos_theta), 1.5);
        vec3 phase_M = vec3(phase_M_scalar);

        vec3 star_deficit = star_color * (
            phase_R * beta_R * total_shadowed_rayleigh +
            phase_M * beta_M * total_shadowed_mie
        );

        if (u_hdr_enabled) {
            star_deficit *= u_exposure;
        }

        // Compute representative centroid for temporal reprojection
        if (total_weight > 1e-6) {
            weighted_pos /= total_weight;
        } else {
            float cam_r = length(cam_local_sph);
            float cos_zenith = dot(cam_local_sph, ray_dir_sph) / max(cam_r, 1e-3);
            float sin_elev = max(0.05, cos_zenith);
            float eff_h = u_h_rayleigh / sin_elev;
            float chord = max(0.0, s_end - s_start);
            float offset = clamp(max(s_closest - s_start, eff_h), chord * 0.05, chord * 0.75);
            weighted_pos = u_camera_pos + ((s_start + offset) / u_au_to_km) * ray_dir;
        }

        float weighted_dist = length(weighted_pos - u_camera_pos);
        float signed_dist = hits_surface ? weighted_dist : -weighted_dist;

        if (u_temporal_accum && u_history_valid) {
            vec3 p_body_local = weighted_pos - u_body_offset;
            vec3 p_prev_world = p_body_local + u_prev_body_offset;
            vec4 clip_prev = u_prev_view_proj * vec4(p_prev_world, 1.0);
            if (clip_prev.w > 1e-13) {
                vec2 uv_prev = (clip_prev.xy / clip_prev.w) * 0.5 + 0.5;
                if (uv_prev.x >= 0.0 && uv_prev.x <= 1.0 && uv_prev.y >= 0.0 && uv_prev.y <= 1.0) {
                    vec4 hist = texture(u_history_godrays, uv_prev);
                    if (abs(hist.a) > 1e-6) {
                        float prev_signed = hist.a * 1000.0;
                        float prev_dist = abs(prev_signed);
                        bool prev_terrain = prev_signed > 0.0;
                        bool curr_terrain = hits_surface;
                        bool crossed_horizon = (prev_terrain != curr_terrain);

                        float d_diff = abs(weighted_dist - prev_dist);
                        float tol = curr_terrain ? 0.15 : 4.0;
                        bool disoccluded = crossed_horizon || (d_diff / max(weighted_dist, 1e-3) > tol);

                        float exp_ratio = (u_prev_exposure > 1e-6) ? (u_exposure / u_prev_exposure) : 1.0;
                        vec3 hist_deficit = hist.rgb * exp_ratio;

                        vec3 d_min = max(vec3(0.0), star_deficit * 0.35 - 0.02);
                        vec3 d_max = star_deficit * 2.5 + 0.05;
                        vec3 clamped_hist = clamp(hist_deficit, d_min, d_max);

                        float alpha = disoccluded ? 1.0 : u_temporal_alpha;
                        star_deficit = mix(clamped_hist, star_deficit, alpha);
                    }
                }
            }
        }

        out_godrays = vec4(star_deficit, signed_dist / 1000.0);
    }
}
