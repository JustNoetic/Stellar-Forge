#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16

in vec3 f_world_pos;
in vec3 f_normal;
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
};

uniform vec3 u_host_planet_pos;
uniform float u_host_planet_radius;
uniform vec3 u_host_planet_color;
uniform vec4 u_host_planet_atmo;
uniform float u_host_planet_refractivity;  // surface (n_mix - 1) for eclipse refraction
uniform vec3 u_camera_pos;
uniform vec4 u_host_planet_pole_obl;
uniform float u_host_planet_R_minor;
uniform int u_clip_mode;
uniform uint u_caster_mask_lo;
uniform uint u_caster_mask_hi;
uniform bool u_planetshine_enabled;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform float u_au_to_km;
uniform float u_caster_max_bend[64];

uniform vec3 u_refract_center;
uniform float u_refract_radius;
uniform float u_refract_max_bend;
uniform float u_refract_scale_height;
uniform vec3 u_refract_pole;
uniform float u_refract_oblateness;

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

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - 3.0);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 adjust_hsba(vec3 color, float hue_shift, float saturation, float brightness) {
    vec3 hsv = rgb2hsv(color);
    hsv.x = fract(hsv.x + hue_shift);
    hsv.y = clamp(hsv.y * saturation, 0.0, 2.0);
    hsv.z = hsv.z * brightness;
    return hsv2rgb(hsv);
}

struct RingPlane {
    vec3 color;
    float inner_r;
    float outer_r;
    float opacity;
    float scatter;
    float asymmetry;
    float backscatter;
    int row_idx;
    float is_textured;
    float unlit_factor;
    float saturation;
    float hue_shift;
    float brightness;
    float alpha_boost;
};

#define MAX_RING_PLANES 16
uniform RingPlane u_ring_planes[MAX_RING_PLANES];
uniform int u_num_ring_planes;
uniform sampler2D u_ring_gradients;

out vec4 out_color;

uniform bool u_is_textured;
uniform sampler2D u_ring_texture_front;
uniform sampler2D u_ring_texture_back;

float HenyeyGreensteinPhaseFunction(float eccentricity, float viewDirDotLight) {
    float g = eccentricity;
    return (1.0 - g * g) / (4.0 * 3.14159265359 * pow(max(1e-6, 1.0 + g * g - 2.0 * g * viewDirDotLight), 1.5));
}

vec2 GetRingPhaseFunctionStrengths(float alpha) {
    // Optical depth affects particle size dominance. 
    // Thin rings (dust) are highly forward-scattering.
    // Thick rings (large chunks) are more isotropic / back-scattering.
    float dust_to_chunks = clamp((alpha - 0.1) / 0.5, 0.0, 1.0);
    // Weights must sum to 1.0 for physical energy conservation.
    return mix(vec2(0.95, 0.05), vec2(0.5, 0.5), dust_to_chunks);
}

vec2 GetRingPhaseFunctionsUnweighted(float dotLight, float asym, float back_asym) {
    return vec2(HenyeyGreensteinPhaseFunction(asym, dotLight), HenyeyGreensteinPhaseFunction(back_asym, dotLight));
}

float GetRingPhaseFunctions(float dotLight, float alpha, float asym, float back_asym) {
    vec2 phaseFunctions = GetRingPhaseFunctionsUnweighted(dotLight, asym, back_asym) * GetRingPhaseFunctionStrengths(alpha);
    return phaseFunctions.x + phaseFunctions.y;
}

float CornetteShanksPhaseFunction(float eccentricity, float viewDirDotLight) {
    float g = eccentricity;
    float g2 = g * g;
    float mu = viewDirDotLight;
    float denom = pow(max(1e-6, 1.0 + g2 - 2.0 * g * mu), 1.5);
    // Normalized to integrate to 1.0 over 4PI steradians
    return (1.5 * (1.0 + mu * mu) * (1.0 - g2)) / ((2.0 + g2) * denom * 12.566370614);
}

float OppositionSurge(float cos_phase, float columnDensity) {
    // cos_phase = dot(V, L) where V points to camera, L points to star
    // At opposition: V ~ L -> cos_phase ~ 1 -> phase_angle ~ 0
    float phase_angle = acos(clamp(cos_phase, -1.0, 1.0));

    // Scale surge with optical depth: dense rings have more inter-particle shadowing
    float density_scale = clamp(columnDensity / 1.5, 0.0, 1.0);

    // SHOE: Shadow Hiding Opposition Effect (broad, ~4 degrees)
    float shoe_width = 0.07;
    float shoe_amp = 0.8 * density_scale;
    float shoe = 1.0 + shoe_amp / (1.0 + phase_angle / shoe_width);

    // CBOE: Coherent Backscatter Opposition Effect (narrow, ~0.35 degrees)
    float cboe_width = 0.006;
    float cboe_amp = 0.3 * density_scale;
    float cboe = 1.0 + cboe_amp * exp(-phase_angle / cboe_width);

    return shoe * cboe;
}

float HapkeHFunction(float mu, float gamma) {
    return (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * gamma);
}

float AnalyticMultipleScattering(float mu_v, float mu_0, float tau, bool onLitSide) {
    float w0 = 0.92; // Single-scattering albedo of ring ice particles (water-ice rings)
    float gamma = sqrt(max(1e-4, 1.0 - w0));
    float Hv = HapkeHFunction(mu_v, gamma);
    float H0 = HapkeHFunction(mu_0, gamma);

    float path_term = 1.0 - exp(-tau * (1.0 / max(1e-4, mu_v) + 1.0 / max(1e-4, mu_0)));
    float mu_ratio = mu_0 / max(1e-4, mu_v + mu_0);

    // Normalized by 1 / (4 * PI) to match single scattering phase function scale
    float inv_4pi = 1.0 / (4.0 * 3.14159265358979);

    if (onLitSide) {
        float ms_lit = w0 * mu_ratio * (Hv * H0 - 1.0) * path_term * inv_4pi;
        return max(0.0, ms_lit);
    } else {
        float ms_unlit = w0 * mu_ratio * (Hv * H0) * exp(-gamma * tau) * path_term * inv_4pi;
        return max(0.0, ms_unlit);
    }
}

float AnalyticalSelfShadowing(float sun_side_abs) {
    // Incident solar flux scale: mu_0 = |N . L| = sin(solar elevation)
    // Radiative transfer (scatteredLight & AnalyticMultipleScattering) already accounts for
    // line-of-sight self-absorption (1/mu_v) and solar path extinction (1/mu_0) inside the slab.
    float mu_0 = max(0.001, sun_side_abs);
    return clamp(mu_0 / 0.45, 0.0, 1.0);
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

// Shared analytical eclipse penumbra + physical atmospheric lens optics & Danjon refraction tinting.
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

void main() {
    if (f_clip_z <= 0.0) discard;
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);

    vec3 cam_to_host = u_host_planet_pos - u_camera_pos;
    vec3 view_ray = normalize(f_local_pos + cam_to_host);
    vec3 ray_dir = view_ray;
    vec3 ray_origin = u_camera_pos;

    float s_min_au = 0.0;
    if (u_refract_max_bend > 1e-6) {
        vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
        float d_km = length(cam_to_host * u_au_to_km);
        
        vec3 pole_n_approx = length(u_host_planet_pole_obl.xyz) > 1e-4 ? normalize(u_host_planet_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
        float d_n_approx = dot(view_ray, pole_n_approx);
        if (abs(d_n_approx) > 1e-6) {
            float t_approx = dot(cam_to_host, pole_n_approx) / d_n_approx;
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
                ray_dir = normalize(view_ray * cos(alpha) - u_dir * sin(alpha));
                float local_s_min = -dot(u_camera_pos - u_refract_center, view_ray);
                if (local_s_min > 0.0) {
                    s_min_au = local_s_min;
                    ray_origin = u_camera_pos + s_min_au * (view_ray - ray_dir);
                }
            }
        }
    }

    // Precise ray origin in body-local coordinates, avoiding catastrophic cancellation
    // By starting from the precise bounding mesh vertex and applying the relative refraction shift.
    float d_bounding = length(f_local_pos + cam_to_host);
    vec3 O_local = f_local_pos + (d_bounding - s_min_au) * (ray_dir - view_ray);

    // Body-local ray-plane intersection for precision
    vec3 pole_n = length(u_host_planet_pole_obl.xyz) > 1e-4 ? normalize(u_host_planet_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    float d_n = dot(ray_dir, pole_n);
    if (abs(d_n) < 1e-6) discard;

    // Intersect ray with ring plane
    // Ring plane passes through host planet center, normal = pole_n
    // From O_local along ray_dir: t where dot(O_local + t * ray_dir, pole_n) = 0
    float t_local = -dot(O_local, pole_n) / d_n;

    vec3 hit_local = O_local + ray_dir * t_local;  // body-local hit, ring-radius scale
    vec3 hit_pos = hit_local + u_host_planet_pos;  // world-space, for depth/clip only

    if (u_clip_mode != 0) {
        float d = dot(hit_local, -cam_to_host);
        if (u_clip_mode == 1 && d > 0.0) discard;
        if (u_clip_mode == 2 && d <= 0.0) discard;
    }

    float r = length(hit_local);

    vec3 total_color_front = vec3(0.0);
    vec3 total_color_back = vec3(0.0);
    vec3 total_color_fwd = vec3(0.0);
    float total_tau = 0.0;
    float total_faded_tau = 0.0;
    float total_scatter = 0.0;
    float total_asym = 0.0;
    float total_backscatter = 0.0;
    float total_textured_tau = 0.0;
    float total_unlit_factor = 0.0;

    for (int i=0; i<u_num_ring_planes; i++) {
        float inner_r = u_ring_planes[i].inner_r;
        float outer_r = u_ring_planes[i].outer_r;
        float dr = min(fwidth(r) * 0.75, (outer_r - inner_r) * 0.05);

        if (r >= inner_r - dr && r <= outer_r + dr) {
            float t = (r - inner_r) / max(1e-6, outer_r - inner_r);
            vec4 tex_val_front;
            vec4 tex_val_back;
            bool plane_is_textured = u_is_textured && (u_ring_planes[i].is_textured > 0.5);
            if (plane_is_textured) {
                tex_val_front = texture(u_ring_texture_front, vec2(clamp(t, 0.0, 1.0), 0.5));
                tex_val_back  = texture(u_ring_texture_back,  vec2(clamp(t, 0.0, 1.0), 0.5));
            } else {
                tex_val_front = texture(u_ring_gradients, vec2(clamp(t, 0.0, 1.0), (float(u_ring_planes[i].row_idx) + 0.5)/16.0));
                tex_val_back = tex_val_front;
            }
            tex_val_front.rgb = pow(tex_val_front.rgb, vec3(2.2));
            tex_val_back.rgb  = pow(tex_val_back.rgb,  vec3(2.2));

            float layer_hue = u_ring_planes[i].hue_shift;
            float layer_sat = u_ring_planes[i].saturation;
            float layer_bri = u_ring_planes[i].brightness;
            float layer_boost = u_ring_planes[i].alpha_boost;
            if (plane_is_textured && (abs(layer_hue) > 1e-4 || abs(layer_sat - 1.0) > 1e-4 || abs(layer_bri - 1.0) > 1e-4)) {
                tex_val_front.rgb = adjust_hsba(tex_val_front.rgb, layer_hue, layer_sat, layer_bri);
                tex_val_back.rgb  = adjust_hsba(tex_val_back.rgb,  layer_hue, layer_sat, layer_bri);
            }

            float alpha = tex_val_front.a;
            if (plane_is_textured && alpha > 1e-5 && abs(layer_boost - 1.0) > 1e-4) {
                alpha = clamp(pow(alpha, 1.0 / max(0.01, layer_boost)), 0.0, 1.0);
            }
            vec3 r_color_front = tex_val_front.rgb;
            vec3 r_color_back  = tex_val_back.rgb;
            vec3 r_color_fwd   = r_color_front;

            float edge_alpha = smoothstep(inner_r - dr, inner_r + dr, r) * (1.0 - smoothstep(outer_r - dr, outer_r + dr, r));

            vec3 plane_color = plane_is_textured ? vec3(1.0) : u_ring_planes[i].color;
            float plane_opacity = u_ring_planes[i].opacity;

            float raw_a_physical = alpha * plane_opacity;
            float raw_a_faded = raw_a_physical * edge_alpha;

            float tau = 0.0;
            if (raw_a_physical >= 0.999) {
                tau = 100.0;
            } else if (raw_a_physical > 1e-6) {
                tau = -log(1.0 - raw_a_physical);
            }

            float tau_faded = 0.0;
            if (raw_a_faded >= 0.999) {
                tau_faded = 100.0;
            } else if (raw_a_faded > 1e-6) {
                tau_faded = -log(1.0 - raw_a_faded);
            }

            if (tau > 0.0) {
                total_color_front += plane_color * r_color_front * tau;
                total_color_back  += plane_color * r_color_back  * tau;
                total_color_fwd   += plane_color * r_color_fwd   * tau;
                total_scatter += u_ring_planes[i].scatter * tau;
                total_asym += u_ring_planes[i].asymmetry * tau;
                total_backscatter += u_ring_planes[i].backscatter * tau;
                total_tau += tau;
                total_faded_tau += tau_faded;
                if (plane_is_textured) {
                    total_textured_tau += tau;
                    total_unlit_factor += u_ring_planes[i].unlit_factor * tau;
                }
            }
        }
    }

    if (total_tau <= 1e-6) {
        out_color = vec4(0.0);
        return;
    }

    vec3 V = -ray_dir;
    vec3 N = normalize(f_normal);
    float cam_side = dot(N, V);

    float min_cam_mu = 0.01;
    float cosViewRayVertical = max(abs(cam_side), min_cam_mu);

    float physical_alpha = 1.0 - exp(-total_tau / cosViewRayVertical);
    float faded_alpha = 1.0 - exp(-total_faded_tau / cosViewRayVertical);

    vec3 f_color_front = total_color_front / max(1e-6, total_tau);
    vec3 f_color_back  = total_color_back  / max(1e-6, total_tau);
    vec3 f_color_fwd   = total_color_fwd   / max(1e-6, total_tau);
    vec4 f_color = vec4(f_color_front, physical_alpha);
    float f_scatter = total_scatter / max(1e-6, total_tau);
    float f_asymmetry = total_asym / max(1e-6, total_tau);
    float f_backscatter = total_backscatter / max(1e-6, total_tau);
    float f_unlit_factor = total_textured_tau > 0.0 ? (total_unlit_factor / total_textured_tau) : 1.0;

    bool is_textured_ring = (total_textured_tau > 0.5 * total_tau);

    vec3 total_direct_illum_color = vec3(0.0);

    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        vec3 frag_to_star = (star_pos - u_host_planet_pos) - hit_local;
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

        float cos_theta = -dot(L, V);

        float sun_side = dot(N, L);

        float star_ang_radius = star_radius / dist_to_star;
        float sin_alpha_local = clamp(star_ang_radius, 1e-6, 1.0);
        float alpha = sin_alpha_local;

        float effective_sun_side = sqrt(sun_side * sun_side + 0.180126 * alpha * alpha);
        float solar_elevation = max(0.02, effective_sun_side);

        float v_star = clamp(sun_side / sin_alpha_local, -1.0, 1.0);
        float f_top = (v_star * sqrt(max(0.0, 1.0 - v_star*v_star)) + asin(v_star)) / 3.14159265358979 + 0.5;
        float same_side = (cam_side > 0.0) ? f_top : (1.0 - f_top);

        float direct_illum_s = 0.0;
        float min_cam_mu = 0.01;
        float cosViewRayVertical  = max(abs(cam_side),  min_cam_mu);
        float cosLightRayVertical = max(abs(sun_side), 1e-5);
        float columnDensity = total_tau;
        float viewDensity = columnDensity / cosViewRayVertical;
        float lightDensity = columnDensity / cosLightRayVertical;
        float scatteredLight = 0.0;
        bool onLitSide = (cam_side * sun_side) >= 0.0;

        if (onLitSide) {
            scatteredLight = viewDensity / (viewDensity + lightDensity) * (1.0 - exp(-viewDensity - lightDensity));
        } else {
            float denominator = lightDensity - viewDensity;
            if (abs(denominator) > 1e-6) {
                scatteredLight = (exp(-viewDensity) - exp(-lightDensity)) * viewDensity / (lightDensity - viewDensity);
            } else {
                scatteredLight = viewDensity * exp(-viewDensity);
            }
        }

        float single_scatter_s = 0.0;
        float ms_s = 0.0;

        if (is_textured_ring) {
            float phase_func = GetRingPhaseFunctions(cos_theta, f_color.a, f_asymmetry, f_backscatter);
            float unlit_mult = onLitSide ? 1.0 : f_unlit_factor;
            single_scatter_s = scatteredLight * phase_func * unlit_mult;
            
            // Multiple scattering transmits poorly through macroscopic chunks on the unlit side.
            // Additionally, thin dust rings (low alpha) are highly forward-scattering and do not isotropize light effectively.
            float dust_to_chunks = clamp((f_color.a - 0.1) / 0.5, 0.0, 1.0);
            ms_s = onLitSide ? (AnalyticMultipleScattering(cosViewRayVertical, cosLightRayVertical, columnDensity, onLitSide) * dust_to_chunks) : 0.0;

            if (onLitSide) {
                float cos_phase = -cos_theta;
                single_scatter_s *= OppositionSurge(cos_phase, columnDensity);
            }
        } else {
            // Double Cornette-Shanks phase function using the ring's own parameters
            float pf_forward = CornetteShanksPhaseFunction(f_asymmetry, cos_theta);
            float pf_backward = CornetteShanksPhaseFunction(f_backscatter, cos_theta);
            // f_scatter controls forward/backward balance (0 = backward dominant, 1 = forward dominant)
            float balance = clamp(f_scatter, 0.0, 1.0);
            float phaseFunc = mix(pf_backward, pf_forward, balance);
            single_scatter_s = scatteredLight * phaseFunc;
            
            // Isotropic multiple scattering only applies to backscattering chunks, not pure forward-scattering dust
            ms_s = AnalyticMultipleScattering(cosViewRayVertical, cosLightRayVertical, columnDensity, onLitSide) * (1.0 - balance);

            // Opposition surge: brightening at low phase angles on single scattering (lit side only)
            if (onLitSide) {
                float cos_phase = -cos_theta; // dot(V, L)
                single_scatter_s *= OppositionSurge(cos_phase, columnDensity);
            }
        }

        direct_illum_s = single_scatter_s + ms_s;

        // Inverse-square falloff
        if (u_hdr_enabled) {
            direct_illum_s *= star_lum / (dist_to_star * dist_to_star);
        }

        // Multiply by PI to convert the physically exact radiance to the 
        // engine's Lambertian surface BRDF convention (which drops the 1/PI).
        direct_illum_s *= 3.14159265358979;

        vec3 shadow_s = vec3(1.0);
        float star_radius_over_dist = star_ang_radius;

        for (int j = 0; j < u_num_casters; j++) {
            if (dot(shadow_s, shadow_s) < 0.001) break;

            if (j < 32) { if ((u_caster_mask_lo & (1u << j)) == 0u) continue; }
            else        { if ((u_caster_mask_hi & (1u << (j - 32))) == 0u) continue; }

            vec3 caster_pos = u_casters[j].xyz;
            float caster_r = u_casters[j].w;
            float atmo_h = u_caster_atmos[j].w;

            vec3 frag_to_caster = (caster_pos - u_host_planet_pos) - hit_local;
            float t_proj = dot(frag_to_caster, L);
            if (t_proj < 0.0) continue;

            float dist_sq = dot(frag_to_caster, frag_to_caster);
            if (dist_sq < caster_r * caster_r * 1.0404) continue;

            float dist_to_caster = sqrt(dist_sq);
            float perp_sq = max(0.0, dist_sq - t_proj * t_proj);

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

            float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
            float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
            if (perp_sq > r_penumbra * r_penumbra) continue;

            float inv_dist = 1.0 / dist_to_caster;
            float alpha = local_star_radius_over_dist;
            float beta = caster_r * inv_dist;
            float gamma = sqrt(perp_sq) * inv_dist;
            float penumbra_outer = alpha + beta;
            float penumbra_inner = abs(beta - alpha);
            shadow_s *= casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, u_caster_max_bend[j], u_caster_atmos[j].xyz);
        }

        if (u_host_planet_radius > 0.0) {
            vec3 frag_to_host = -hit_local;
            float t_proj = dot(frag_to_host, L);

            if (t_proj > 0.0 && t_proj < dist_to_star) {
                float dist_sq = dot(frag_to_host, frag_to_host);
                float dist_to_host = sqrt(dist_sq);
                vec3 cross_vec = cross(frag_to_host, L);
                float perp_sq = dot(cross_vec, cross_vec);

                float host_r = u_host_planet_radius;
                float host_atmo_h = u_host_planet_atmo.w;
                float host_r_minor = u_host_planet_R_minor;

                // Bounding cone early out using maximum equatorial radius
                float max_effective_r = host_r + (host_atmo_h > 0.0 ? host_atmo_h * 4.0 : 0.0);
                float max_r_penumbra = max_effective_r + dist_to_host * star_radius_over_dist;

                if (perp_sq <= max_r_penumbra * max_r_penumbra) {
                    vec3 perp_vec = frag_to_host - t_proj * L;
                    if (host_r_minor < host_r - 1e-5) {
                        vec3 host_pole = u_host_planet_pole_obl.xyz;
                        host_r = get_oblate_radius(host_r, host_r_minor, host_pole, L, perp_vec);
                    }

                    float directional_star_r = star_radius;
                    float star_r_minor = u_stars_poles_obl[s].w;
                    if (star_r_minor < star_radius - 1e-5) {
                        vec3 star_pole = u_stars_poles_obl[s].xyz;
                        directional_star_r = get_oblate_radius(star_radius, star_r_minor, star_pole, L, perp_vec);
                    }
                    float local_star_radius_over_dist = directional_star_r / dist_to_star;

                    float effective_r = host_r + (host_atmo_h > 0.0 ? host_atmo_h * 4.0 : 0.0);
                    float r_penumbra = effective_r + dist_to_host * local_star_radius_over_dist;

                    if (perp_sq <= r_penumbra * r_penumbra) {
                        float inv_dist = 1.0 / dist_to_host;
                        float alpha = local_star_radius_over_dist;
                        float beta = host_r * inv_dist;
                        float gamma = sqrt(perp_sq) * inv_dist;
                        float penumbra_outer = alpha + beta;
                        float penumbra_inner = abs(beta - alpha);
                        float max_bend = host_atmo_h > 0.0
                            ? clamp(2.0 * max(u_host_planet_refractivity, 0.0) * sqrt(3.14159265359 * host_r / max(1e-6, host_atmo_h * 2.0)), 0.001, 0.05)
                            : 0.0;
                        shadow_s *= casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, max_bend, u_host_planet_atmo.xyz);
                    }
                }
            }
        }

        float lit_blend = smoothstep(-0.02, 0.02, cam_side * sun_side);
        float fwd_blend = smoothstep(-0.3, 0.7, cos_theta);
        vec3 lit_color_star = (total_textured_tau > 0.5 * total_tau) ? mix(f_color_front, f_color_fwd, fwd_blend) : f_color_front;
        vec3 active_ring_color = (total_textured_tau > 0.5 * total_tau) ? mix(f_color_back, lit_color_star, lit_blend) : f_color_front;
        total_direct_illum_color += star_color * active_ring_color * direct_illum_s * shadow_s;
    }

    vec3 total_planetshine = vec3(0.0);
    if (u_host_planet_radius > 0.0) {
        vec3 frag_to_host = -f_local_pos;
        float dist_host_sq = dot(frag_to_host, frag_to_host);
        float dist_host = sqrt(dist_host_sq);

        // Exact angular radius & spherical cap solid angle of planet as seen from this ring radius
        float sin_alpha_planet = clamp(u_host_planet_radius / max(dist_host, 1e-5), 0.0, 1.0);
        float cos_alpha_planet = sqrt(max(0.0, 1.0 - sin_alpha_planet * sin_alpha_planet));
        float solid_angle = 2.0 * (1.0 - cos_alpha_planet); // Exact solid angle Omega(r) = 2pi(1 - cos alpha)

        // Effective elevation of the 3D spherical planet disk above/below the 2D ring plane
        float planet_elevation = max(0.5 * sin_alpha_planet, 0.05);

        float lit_blend_p = smoothstep(-0.02, 0.02, cam_side);
        vec3 active_shine_color = (is_textured_ring) ? mix(f_color_back, f_color_front, lit_blend_p) : f_color_front;

        for (int s = 0; s < u_num_stars; s++) {
            vec3 star_pos = u_stars_pos_radius[s].xyz;
            vec3 host_to_star = star_pos - u_host_planet_pos;
            float dist_host_star = length(host_to_star);
            vec3 L_host = host_to_star / max(dist_host_star, 1e-6);

            // Crescent-shifted effective direction from ring point to illuminated region of planet
            vec3 frag_to_lit_planet = frag_to_host + L_host * (u_host_planet_radius * 0.5);
            vec3 L_planet = normalize(frag_to_lit_planet);

            float cos_theta_p = -dot(L_planet, V);

            float pf_p = 0.0;
            if (is_textured_ring) {
                float pf_ref_p = GetRingPhaseFunctions(1.0, f_color.a, f_asymmetry, f_backscatter);
                float pf_curr_p = GetRingPhaseFunctions(cos_theta_p, f_color.a, f_asymmetry, f_backscatter);
                pf_p = pf_curr_p / max(1e-4, pf_ref_p);
            } else {
                float pf_forward_p = CornetteShanksPhaseFunction(f_asymmetry, cos_theta_p);
                float pf_backward_p = CornetteShanksPhaseFunction(f_backscatter, cos_theta_p);
                float balance_p = clamp(f_scatter, 0.0, 1.0);
                pf_p = mix(pf_backward_p, pf_forward_p, balance_p);
            }

            float columnDensity_p = -log(max(1.0 - f_color.a, 1e-5));
            float lightDensity_p = columnDensity_p / max(0.04, planet_elevation);
            float scattered_opacity_p = 1.0 - exp(-lightDensity_p);

            float ring_planet_response = pf_p * scattered_opacity_p;

            vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
            float star_sin_lat = abs(dot(L_host, pole_dir));

            vec3 eq_color = u_stars_colors[s].rgb;
            float eq_lum = u_stars_colors[s].a;
            vec3 pole_color = u_stars_pole_colors[s].rgb;
            float pole_lum = u_stars_pole_colors[s].a;

            vec3 star_color = mix(eq_color, pole_color, star_sin_lat);
            float star_lum = mix(eq_lum, pole_lum, star_sin_lat);

            float irradiance = u_hdr_enabled ? (star_lum / max(dist_host_star * dist_host_star, 1e-8)) : 1.0;

            // Illuminated phase of planet as seen from the ring point
            vec3 L_center = frag_to_host / max(dist_host, 1e-6);
            float cos_a_p = clamp(dot(L_host, -L_center), -1.0, 1.0);
            float a_p = acos(cos_a_p);
            float planet_phase = (sin(a_p) + (3.141592653589793 - a_p) * cos_a_p) / 3.141592653589793;

            // Planetshine flux reaching the ring point: Lambertian sphere illumination
            vec3 planetshine_irradiance = u_host_planet_color * star_color * irradiance * planet_phase * (solid_angle * 0.6666667);

            total_planetshine += planetshine_irradiance * active_shine_color * ring_planet_response;
        }
    }

    vec3 raw_color = total_direct_illum_color;
    if (u_planetshine_enabled) {
        raw_color += total_planetshine;
    }
    vec3 final_color = u_hdr_enabled ? (raw_color * u_exposure) : raw_color;

    float fade_scale = faded_alpha / max(1e-6, physical_alpha);
    out_color = vec4(final_color * fade_scale, faded_alpha);
}
