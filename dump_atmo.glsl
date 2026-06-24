
#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define MAX_RING_PLANES 16
#define PI 3.14159265358979

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
uniform int u_body_idx;

uint u_caster_mask_lo;
uint u_caster_mask_hi;
uint u_ring_mask;

uniform vec3  u_body_offset;
uniform float u_atmo_radius_au;
uniform float u_planet_radius_km;
uniform float u_atmo_radius_km;
uniform float u_au_to_km;
uniform vec3  u_beta_rayleigh;
uniform float u_h_rayleigh;
uniform vec3 u_beta_mie;
uniform float u_h_mie;
uniform float u_mie_g;
uniform vec3  u_beta_absorption;
uniform float u_sun_intensity;
uniform vec3  u_camera_pos;
uniform int   u_num_samples;
uniform vec4  u_pole_obl;

uniform sampler2D u_ring_gradients;
uniform int u_num_ring_planes;
uniform int u_atmo_quality;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec3 u_ring_params[MAX_RING_PLANES]; 
uniform int u_atmo_clip_mode; 

uniform int u_num_active_casters;
uniform vec4 u_active_casters[8];
uniform vec4 u_active_caster_poles_obl[8];
uniform vec4 u_active_caster_atmos[8];
uniform sampler2D u_eclipse_lut; // Kept to avoid uniform bound errors
uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;

out vec4 out_color;

// g_local_rings removed to prevent local memory array spilling

vec3 toSphericalSpace(vec3 p, vec4 pole_obl) {
    float f = pole_obl.w;
    if (f == 0.0) return p;
    vec3 pole = pole_obl.xyz;
    float h = dot(p, pole);
    vec3 p_perp = p - h * pole;
    return p_perp + (h / (1.0 - f)) * pole;
}

float get_oblate_radius(float r_eq, float oblateness, vec3 pole, vec3 L, vec3 perp_vec) {
    if (oblateness <= 0.0 || r_eq < 1e-6) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float f_factor = 1.0 - oblateness;
    float R_minor = r_eq * sqrt(PdotL * PdotL + f_factor * f_factor * P_proj_len * P_proj_len);
    if (R_minor < 1e-6) return r_eq;
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / R_minor) * (y / R_minor));
    if (denom < 1e-6) return r_eq;
    return perp_len / denom;
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
        float dist_to_caster = sqrt(dist_sq);
        if (dist_to_caster < caster_r * 1.02) continue;
        
        vec3 cross_vec = cross(s_to_c, L_dir);
        float perp_sq = dot(cross_vec, cross_vec);
        
        vec3 perp_vec = s_to_c - t_proj * L_dir;
        float oblateness = u_active_caster_poles_obl[i].w;
        if (oblateness > 0.0) {
            vec3 pole = u_active_caster_poles_obl[i].xyz;
            caster_r = get_oblate_radius(caster_r, oblateness, pole, L_dir, perp_vec);
        }
        
        float directional_star_r = star_radius;
        if (star_obl > 0.0) {
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
        float penumbra_inner = max(0.0, beta - alpha);
        
        float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
        float occ = max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma);
        
        vec3 caster_shadow = vec3(1.0 - occ);
        
        if (atmo_h > 0.0 && gamma < penumbra_outer) {
            float max_bend = clamp(2.0 * 0.00029 * sqrt(3.14159265359 * caster_r / max(1.0, atmo_h * caster_r * 2.0)), 0.001, 0.05);
            float req_bend = beta - gamma;
            
            float optical_depth = max(0.0, req_bend);
            float atmospheric_transmission = exp(-optical_depth * 150.0);
            float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
            
            float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
            vec3 atmo_tint = u_active_caster_atmos[i].xyz;
            vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
            
            float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
            float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            
            caster_shadow += deep_tint * refraction_intensity * atmo_blend;
        }
        shadow *= clamp(caster_shadow, 0.0, 1.0);
    }
    
    uint processed_mask = 0u;
    for (int k = 0; k < u_num_ring_planes; k++) {
        if ((u_ring_mask & (1u << k)) == 0u) continue;
        if ((processed_mask & (1u << k)) != 0u) continue;
        
        vec3 plane_center = u_ring_center[k];
        vec3 plane_normal = u_ring_normal[k];
        
        uint coplanar_mask = 0u;
        for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
            if ((u_ring_mask & (1u << ring_idx)) == 0u) continue;
            if (distance(plane_center, u_ring_center[ring_idx]) < 1e-6 && 
                dot(plane_normal, u_ring_normal[ring_idx]) > 0.999) {
                coplanar_mask |= (1u << ring_idx);
                processed_mask |= (1u << ring_idx);
            }
        }
        
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
            float alpha_mult = textureLod(u_ring_gradients, vec2(p_mid, (float(ring_idx) + 0.5) / 16.0), 0.0).r;
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
    float c = dot(origin, origin) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant < 0.0) return vec2(1e10, -1e10);
    float d = sqrt(discriminant);
    return vec2((-b - d) / a, (-b + d) / a);
}

vec3 get_transmittance(float r, float cos_theta) {
    float h_norm = clamp((r - u_planet_radius_km) / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float u = cos_theta * 0.5 + 0.5;
    return textureLod(u_transmittance_lut, vec2(u, h_norm), 0.0).rgb;
}

void main() {
    u_ring_mask = floatBitsToUint(instances[u_body_idx * 6 + 3].w);
    
    // g_local_rings removed to prevent local memory array spilling

    vec3 planet_center_render = u_body_offset;
    vec3 cam_local_au = u_camera_pos - planet_center_render;
    vec3 cam_local = cam_local_au * u_au_to_km;

    if (f_clip_z < 0.0) discard;
    vec3 f_pos_local_au = f_local_pos * u_atmo_radius_au;
    vec3 f_pos_local = f_pos_local_au * u_au_to_km;
    vec3 ray_dir = normalize(f_pos_local - cam_local);

    vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
    vec3 cam_local_sph = toSphericalSpace(cam_local, u_pole_obl);

    // AP_VOLUME_HOOK
    // Relative Backwards Parameterization
    vec3 E = f_pos_local;
    vec3 D = ray_dir; 
    float E_dot_D = dot(E, D);
    float E_sq = dot(E, E);
    float delta_atmo = E_dot_D * E_dot_D - (E_sq - u_atmo_radius_km * u_atmo_radius_km);
    if (delta_atmo < 0.0) discard;
    
    float t_atmo_front = E_dot_D + sqrt(delta_atmo);
    float delta_planet = E_dot_D * E_dot_D - (E_sq - u_planet_radius_km * u_planet_radius_km);
    float t_planet_front = -1.0;
    if (delta_planet > 0.0) {
        t_planet_front = E_dot_D + sqrt(delta_planet);
    }
    
    float t_cam = length(E - cam_local);
    float s_start = min(t_atmo_front, t_cam);
    float s_end = 0.0;
    
    if (t_planet_front > 0.0 && t_planet_front < s_start) {
        s_end = t_planet_front;
    }
    
    if (s_start <= s_end) discard;
    cam_local = E;
    ray_dir = -D;
    ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
    cam_local_sph = toSphericalSpace(cam_local, u_pole_obl);
    vec3 frag_local = cam_local;

    float closest_s_ring = 1e10;
    for (int k = 0; k < u_num_ring_planes; k++) {
        vec3 ring_center_world_rel = u_ring_center[k];
        if (length(ring_center_world_rel - u_body_offset) > 1e-4) continue;
        
        vec3 ring_center_local = (ring_center_world_rel - u_body_offset) * u_au_to_km;
        vec3 ring_normal = u_ring_normal[k];
        
        float denom = dot(ray_dir, ring_normal);
        if (abs(denom) > 1e-8) {
            float s_ring = dot(ring_center_local - frag_local, ring_normal) / denom;
            if (s_ring > 0.0) {
                vec3 hit_local = frag_local + s_ring * ray_dir;
                float dist_from_center = length(hit_local - ring_center_local);
                float inner_r_km = u_ring_params[k].x * u_au_to_km;
                float outer_r_km = u_ring_params[k].y * u_au_to_km;
                
                if (dist_from_center >= inner_r_km && dist_from_center <= outer_r_km) {
                    if (s_ring < closest_s_ring) {
                        closest_s_ring = s_ring;
                    }
                }
            }
        }
    }

    if (closest_s_ring < 1e9) {
        if (u_atmo_clip_mode == 1) {
            s_start = min(s_start, closest_s_ring);
        } else if (u_atmo_clip_mode == 2) {
            s_end = max(s_end, closest_s_ring);
        }
    } else {
        if (u_atmo_clip_mode == 2) {
            s_end = s_start;
        }
    }

    for (int i = 0; i < u_num_active_casters; i++) {
        vec3 caster_pos = u_active_casters[i].xyz;
        float caster_r = u_active_casters[i].w;
        vec3 caster_local = (caster_pos - planet_center_render) * u_au_to_km;
        
        if (length(caster_local) < 1.0) continue;
        
        vec4 pole_obl = u_active_caster_poles_obl[i];
        vec3 origin_c = toSphericalSpace(cam_local - caster_local, pole_obl);
        vec3 dir_c = toSphericalSpace(ray_dir, pole_obl);
        vec2 s_c = raySphereIntersect(origin_c, dir_c, caster_r);
        if (s_c.y > 0.0 && s_c.y < s_start) {
            s_end = max(s_end, s_c.y);
        }
    }

    if (s_start <= s_end) discard;

    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M = u_beta_mie * 1000.0;
    vec3 beta_A = u_beta_absorption * 1000.0;

    int steps = u_num_samples;
    float step_size = (s_start - s_end) / float(steps);
    vec3 scattered = vec3(0.0);
    float final_od_rayleigh = 0.0;
    float final_od_mie = 0.0;
    float final_od_ozone = 0.0;

    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        
        vec3 frag_to_star = star_pos - planet_center_render;
        float dist_to_star_au = length(frag_to_star);
        vec3 L = frag_to_star / max(dist_to_star_au, 1e-6);
        
        vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
        float sin_lat = abs(dot(L, pole_dir));
        
        vec3 eq_color = u_stars_colors[s].rgb;
        float eq_lum = u_stars_colors[s].a;
        vec3 pole_color = u_stars_pole_colors[s].rgb;
        float pole_lum = u_stars_pole_colors[s].a;
        
        vec3 star_color = mix(eq_color, pole_color, sin_lat);
        float star_lum = mix(eq_lum, pole_lum, sin_lat);
        float star_radius_au = star_radius / u_au_to_km;
        float sin_star = star_radius_au / max(dist_to_star_au, star_radius_au + 1e-6);

        vec3 sun_pos_local = (star_pos - planet_center_render) * u_au_to_km;
        // sun_dir computed per-sample now

        vec3 global_eclipse_shadow = vec3(1.0);
        vec3 end_eclipse_shadow = vec3(1.0);
        bool skip_volumetric_shadow = false;
        
        vec4 ring_s1_out = vec4(1e9);
        vec4 ring_s2_out = vec4(-1e9);
        vec4 ring_s1_in = vec4(1e9);
        vec4 ring_s2_in = vec4(-1e9);
        vec4 ring_inner = vec4(0.0);
        vec4 ring_outer = vec4(1.0);
        vec4 ring_opac = vec4(0.0);
        vec4 ring_v_coord = vec4(0.0);
        vec4 ring_R_eff = vec4(0.0);
        vec3 ring_A_prime = vec3(0.0);
        vec3 ring_B = vec3(0.0);

        float s_mid = (s_start + s_end) * 0.5;
        vec3 mid_pos = frag_local + s_mid * ray_dir;
        vec3 mid_render = mid_pos / u_au_to_km + planet_center_render;
        vec3 mid_to_star = star_pos - mid_render;
        float dist_mid_star = length(mid_to_star);
        vec3 L_mid = mid_to_star / max(dist_mid_star, 1e-6);
        float sr_start = star_radius / max(dist_mid_star, 1e-6);
        vec3 O = frag_local / u_au_to_km + planet_center_render;
        vec3 V = ray_dir / u_au_to_km;

        if (u_atmo_quality > 0) { 
            if (u_atmo_quality == 1) {
                global_eclipse_shadow = compute_shadow(mid_render, L_mid, dist_mid_star, planet_center_render, star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz);
                end_eclipse_shadow = global_eclipse_shadow;
            }

            // High mode interpolation and early-out removed because they miss thin ring shadows.
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
                    
                    for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
                        if ((u_ring_mask & (1u << ring_idx)) == 0u) continue;
                        if (distance(u_ring_center[k], u_ring_center[ring_idx]) < 1e-5 && dot(N, u_ring_normal[ring_idx]) > 0.999) {
                            processed_mask |= (1u << ring_idx);
                            if (ring_count >= 4) continue;
                            
                            float inner_r_km = u_ring_params[ring_idx].x * u_au_to_km;
                            float outer_r_km = u_ring_params[ring_idx].y * u_au_to_km;
                            float opacity = u_ring_params[ring_idx].z;
                            float base_tau = -log(max(1e-6, 1.0 - opacity));
                            float scaled_tau = base_tau / max(1e-4, abs(denom));
                                
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
                }
                
                vec3 shadow_start = vec3(1.0);
                vec3 shadow_end = vec3(1.0);
                
                vec3 start_render = O + s_start * V;
                vec3 end_render = O + s_end * V;
                vec3 L_start = L_mid;
                float sr_start = star_radius / max(dist_mid_star, 1e-6);
                vec3 L_end = L_mid;
                float sr_end = sr_start;
                
                for (int c = 0; c < u_num_active_casters; c++) {
                    vec3 pos = u_active_casters[c].xyz;
                    float rad = u_active_casters[c].w;
                    float atmo = u_active_caster_atmos[c].w;
                    float obl = u_active_caster_poles_obl[c].w;
                    vec3 pole = u_active_caster_poles_obl[c].xyz;
                    vec3 atmo_tint = u_active_caster_atmos[c].xyz;
                    
                    vec3 s2c = pos - start_render;
                    float t = dot(s2c, L_start);
                    if (t > 0.0) {
                        float dist_sq = dot(s2c, s2c);
                        if (dist_sq > rad * rad * 1.0404) {
                            float dist = sqrt(dist_sq);
                            vec3 cross_vec = cross(s2c, L_start);
                            float p2 = dot(cross_vec, cross_vec);
                            float r = rad;
                            if (obl > 0.0) r = get_oblate_radius(r, obl, pole, L_start, s2c - t * L_start);
                            float eff_r = r + (atmo > 0.0 ? atmo * 4.0 : 0.0);
                            float rp = eff_r + dist * sr_start;
                            if (p2 < rp * rp) {
                                float inv_d = 1.0 / dist;
                                float alpha = sr_start; float beta = r * inv_d; float gamma = sqrt(p2) * inv_d;
                                float po = alpha + beta; float pi = max(0.0, beta - alpha);
                                float occ = min(1.0, (beta*beta)/max(1e-9, alpha*alpha)) * smoothstep(po, pi, gamma);
                                vec3 sh = vec3(1.0 - occ);
                                if (atmo > 0.0 && gamma < po) {
                                    float max_bend = clamp(2.0 * 0.00029 * sqrt(3.14159265359 * r / max(1.0, atmo * r * 2.0)), 0.001, 0.05);
                                    float req_bend = beta - gamma;
                                    
                                    float optical_depth = max(0.0, req_bend);
                                    float atmospheric_transmission = exp(-optical_depth * 150.0);
                                    float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
                                    
                                    float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
                                    vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
                                    
                                    float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
                                    float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
                                    float atmo_blend = smoothstep(po, pi, gamma);
                                    
                                    sh += deep_tint * refraction_intensity * atmo_blend;
                                }
                                shadow_start *= clamp(sh, 0.0, 1.0);
                            }
                        }
                    }
                    
                    s2c = pos - end_render;
                    t = dot(s2c, L_end);
                    if (t > 0.0) {
                        float dist_sq = dot(s2c, s2c);
                        if (dist_sq > rad * rad * 1.0404) {
                            float dist = sqrt(dist_sq);
                            vec3 cross_vec = cross(s2c, L_end);
                            float p2 = dot(cross_vec, cross_vec);
                            float r = rad;
                            if (obl > 0.0) r = get_oblate_radius(r, obl, pole, L_end, s2c - t * L_end);
                            float eff_r = r + (atmo > 0.0 ? atmo * 4.0 : 0.0);
                            float rp = eff_r + dist * sr_end;
                            if (p2 < rp * rp) {
                                float inv_d = 1.0 / dist;
                                float alpha = sr_end; float beta = r * inv_d; float gamma = sqrt(p2) * inv_d;
                                float po = alpha + beta; float pi = max(0.0, beta - alpha);
                                float occ = min(1.0, (beta*beta)/max(1e-9, alpha*alpha)) * smoothstep(po, pi, gamma);
                                vec3 sh = vec3(1.0 - occ);
                                if (atmo > 0.0 && gamma < po) {
                                    float max_bend = clamp(2.0 * 0.00029 * sqrt(3.14159265359 * r / max(1.0, atmo * r * 2.0)), 0.001, 0.05);
                                    float req_bend = beta - gamma;
                                    
                                    float optical_depth = max(0.0, req_bend);
                                    float atmospheric_transmission = exp(-optical_depth * 150.0);
                                    float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
                                    
                                    float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
                                    vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
                                    
                                    float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
                                    float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
                                    float atmo_blend = smoothstep(po, pi, gamma);
                                    
                                    sh += deep_tint * refraction_intensity * atmo_blend;
                                }
                                shadow_end *= clamp(sh, 0.0, 1.0);
                            }
                        }
                    }
                }
                global_eclipse_shadow = shadow_start;
                end_eclipse_shadow = shadow_end;
                
                if (shadow_start.r > 0.999 && shadow_end.r > 0.999 && ring_count == 0) skip_volumetric_shadow = true;
                else if (shadow_start.r < 0.001 && shadow_end.r < 0.001 && ring_count == 0) skip_volumetric_shadow = true;
            }
        }
        
        // OPTIMIZATION 1: Fast limb math invariants pre-calculated
        float alpha_sun_local = asin(clamp(sin_star, 0.0, 0.9999));
        float cos_sun = cos(alpha_sun_local);
        float sin_sun = sin_star;
        
        vec3 step_dir_sph = ray_dir_sph * step_size;
        vec3 current_pos_sph = cam_local_sph + (s_start + 0.5 * step_size) * ray_dir_sph;

        float od_rayleigh = 0.0;
        float od_mie = 0.0;
        float od_ozone = 0.0;
        vec3 total_rayleigh = vec3(0.0);
        vec3 total_mie = vec3(0.0);
        vec3 total_ms = vec3(0.0);

        float current_s = s_start - 0.5 * step_size;
        float t_lerp = 0.5 / float(steps);
        float t_step = 1.0 / float(steps);

        for (int i = 0; i < steps; i++) {
            float sample_len = length(current_pos_sph);
            float altitude = sample_len - u_planet_radius_km;

            float rho_R = exp(-altitude / u_h_rayleigh);
            float rho_M = exp(-altitude / u_h_mie);
            float rho_O = exp(-pow((altitude - 25.0) / 8.0, 2.0));

            od_rayleigh += rho_R * step_size;
            od_mie += rho_M * step_size;
            od_ozone += rho_O * step_size;

            vec3 sun_dir = normalize(sun_pos_local - current_pos_sph);
            vec3 sun_dir_sph = toSphericalSpace(sun_dir, u_pole_obl);
            float light_cos_theta = dot(current_pos_sph / sample_len, sun_dir_sph);
            float h_norm = clamp(altitude / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);

            float sin_planet = u_planet_radius_km / max(sample_len, u_planet_radius_km + 0.01);
            float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));
            
            float cos_outer = cos_planet * cos_sun - sin_planet * sin_sun;
            float cos_inner = cos_planet * cos_sun + sin_planet * sin_sun;

            float neg_light_cos = -light_cos_theta;
            float vis_fraction = 1.0;

            if (neg_light_cos > cos_inner) { 
                vis_fraction = 0.0;
            } else if (neg_light_cos > cos_outer) { 
                float alpha_planet = asin(clamp(sin_planet, 0.0, 0.9999));
                float separation = acos(clamp(neg_light_cos, -1.0, 1.0));
                float x_vis = (separation - alpha_planet) / max(alpha_sun_local, 1e-7);
                vis_fraction = (acos(clamp(-x_vis, -1.0, 1.0)) + x_vis * sqrt(max(0.0, 1.0 - x_vis * x_vis))) / PI;
            }

            float disc_top_cos = min(light_cos_theta + sin_star, 1.0);
            float disc_bot_cos = max(light_cos_theta - sin_star, -cos_planet); // Reused cos_planet
            float effective_cos = (disc_top_cos + disc_bot_cos) * 0.5;

            vec3 tau_to_cam = beta_R * od_rayleigh + beta_M * od_mie + beta_A * od_ozone;
            vec3 transmittance_to_sun = get_transmittance(sample_len, effective_cos);

            vec3 sample_shadow = global_eclipse_shadow;
            if (u_atmo_quality == 2 && !skip_volumetric_shadow) { 
                sample_shadow = mix(global_eclipse_shadow, end_eclipse_shadow, t_lerp);
                
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
                        if (valid.x > 0.0) sh_mult.x = 1.0 - frac.x * (1.0 - exp(-ring_opac.x * textureLod(u_ring_gradients, vec2(p_mid.x, ring_v_coord.x), 0.0).r));
                        if (valid.y > 0.0) sh_mult.y = 1.0 - frac.y * (1.0 - exp(-ring_opac.y * textureLod(u_ring_gradients, vec2(p_mid.y, ring_v_coord.y), 0.0).r));
                        if (valid.z > 0.0) sh_mult.z = 1.0 - frac.z * (1.0 - exp(-ring_opac.z * textureLod(u_ring_gradients, vec2(p_mid.z, ring_v_coord.z), 0.0).r));
                        if (valid.w > 0.0) sh_mult.w = 1.0 - frac.w * (1.0 - exp(-ring_opac.w * textureLod(u_ring_gradients, vec2(p_mid.w, ring_v_coord.w), 0.0).r));
                        
                        sample_shadow *= sh_mult.x * sh_mult.y * sh_mult.z * sh_mult.w;
                    }
                }
            } else if (u_atmo_quality == 3) {
                vec3 sample_pos_local = frag_local + current_s * ray_dir;
                vec3 sample_render = sample_pos_local / u_au_to_km + planet_center_render;
                vec3 sample_to_star = star_pos - sample_render;
                float dist_sample_star = length(sample_to_star);
                sample_shadow = compute_shadow(sample_render, sample_to_star / max(dist_sample_star, 1e-6), dist_sample_star, planet_center_render, star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz);
            }

            vec3 direct_attenuation = exp(-tau_to_cam) * transmittance_to_sun;
            vec3 atten_direct = direct_attenuation * sample_shadow * vis_fraction;

            total_rayleigh += rho_R * atten_direct * step_size;
            total_mie      += rho_M * atten_direct * step_size;
            
            float sun_cos_zenith = dot(normalize(current_pos_sph), normalize(sun_pos_local));
            float h_norm_ms = clamp(altitude / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
            vec2 ms_uv = vec2(sun_cos_zenith * 0.5 + 0.5, h_norm_ms);
            vec3 psi = textureLod(u_multi_scatter_lut, ms_uv, 0.0).rgb;
            
            total_ms += (beta_R * rho_R + beta_M * rho_M) * psi * sample_shadow * vis_fraction * step_size;

            current_pos_sph += step_dir_sph;
            current_s -= step_size;
            t_lerp += t_step;
        }

        vec3 sun_dir = normalize(sun_pos_local - current_pos_sph);
        float cos_theta = dot(ray_dir, sun_dir);
        float phase_R = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);

        float g = u_mie_g;
        float g2 = g * g;
        float phase_M = (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta))
                      / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));

        float irradiance = u_hdr_enabled ? (star_lum / max(dist_to_star_au * dist_to_star_au, 1e-8)) : 1.0;
        
        scattered += star_color * u_sun_intensity * irradiance * (
            phase_R * beta_R * total_rayleigh +
            phase_M * beta_M * total_mie +
            total_ms
        );

        final_od_rayleigh = od_rayleigh;
        final_od_mie = od_mie;
        final_od_ozone = od_ozone;
    }

    vec3 transmittance = exp(-(beta_R * final_od_rayleigh
                             + beta_M * final_od_mie
                             + beta_A * final_od_ozone));

    if (u_hdr_enabled) {
        scattered *= u_exposure;
    }

    float avg_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    out_color = vec4(scattered, (1.0 - avg_transmittance));
}
