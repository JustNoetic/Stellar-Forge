#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : enable
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
in float f_subpixel_factor;
flat in vec2 f_center_px;
flat in float f_clamped_min_px;
flat in float f_apparent_px;
flat in uvec2 f_caster_mask;
flat in uint f_ring_mask;
flat in vec3 f_planetshine_dir;
flat in vec3 f_planetshine_color;
flat in float f_tex_idx;
flat in float f_rotation_angle;
flat in vec3 f_pole;
in vec3 f_local_pos;
flat in vec3 f_center_pos;
flat in float f_radius;
flat in float f_final_radius;
flat in float f_oblateness;
flat in float f_bounding_radius;
flat in vec3 f_my_atmo_tint;
flat in float f_my_atmo_h;
flat in float f_my_scale_height;
flat in vec3 f_my_atmo_color;
    flat in vec3 f_my_mie_tau;
    flat in float f_my_mie_h;
    flat in vec3 f_my_o3_tau;
    flat in vec2 f_my_o3_layer;

struct BodyTextures {
    uvec2 diffuse;
    uvec2 normal;
    uvec2 specular;
    uvec2 clouds;
};

layout(std430, binding = 10) readonly buffer BodyTextureBlock {
    BodyTextures u_body_textures[];
};

uniform bool u_is_cloud_pass;

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
    // Vertical Chapman-ozone column (rgb) + layer Gaussian width km (w).
    // Appended tail member: older SceneData declarations remain offset-valid.
    vec4 u_caster_ozone_vert[MAX_CASTERS];
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

#include "common/refraction.glsl"

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
                      float max_bend, vec4 atmo_param, vec4 ozone_param,
                      float scale_height_km, float dist_to_caster) {
    float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
    float occ = clamp(max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma), 0.0, 1.0);
    // Correct geometric shadow curve: (1 - occ)^GAMMA (Space Engine gamma correction: 1 / 1.6)
    float geom_sh = pow(clamp(1.0 - occ, 0.0, 1.0), 1.0 / 1.6);
    vec3 sh = vec3(geom_sh);

    if (max_bend > 1e-6 && gamma < penumbra_outer) {
        float H_scale = max(scale_height_km, 0.1);
        float sigma_z = max(0.707 * H_scale, 0.1);
        float z_peak = ozone_param.w;
        float dist_km = max(dist_to_caster * u_au_to_km, 1e-6);

        // Direct sunlight grazing the planetary limb in the inner penumbra passes
        // through the stratosphere and ozone layer before reaching the vacuum.
        float caster_r_km = max(beta * dist_km, 100.0);
        float grazing_factor = sqrt(2.0 * PI * caster_r_km / max(H_scale, 1e-3));
        vec3 tau_grazing_0 = atmo_param.xyz * grazing_factor;

        if (gamma >= penumbra_inner) {
            float z_direct_km = (gamma - penumbra_inner) * dist_km;
            if (z_direct_km < 60.0) {
                vec3 tau_R_d = tau_grazing_0 * exp(-z_direct_km / H_scale);
                float z_diff_d = (z_direct_km - z_peak) / sigma_z;
                vec3 tau_O3_d = ozone_param.xyz * exp(-0.5 * z_diff_d * z_diff_d);
                vec3 T_direct = exp(-(tau_R_d + tau_O3_d));
                vec3 pen_filter = clamp(T_direct + vec3(clamp((z_direct_km - 40.0) / 15.0, 0.0, 1.0)), 0.0, 1.0);
                sh = geom_sh * pen_filter;
            }
        }

        // For an extended light source (like the Sun), the transmitted light is dominated
        // by the rays passing through the highest possible altitude (least required bend).
        // We approximate the integral over the Sun's disk by evaluating the ray from the 
        // upper limb (offset by ~80% of the Sun's radius).
        float req_bend = beta - gamma - alpha * 0.8;

        // Rays requiring bend > max_bend hit the solid planet body (100% blocked)
        if (req_bend <= max_bend) {
            // Normalized penetration depth in atmosphere (0 = top of atmosphere, 1 = surface level)
            float atmo_depth = clamp(max(0.0, req_bend) / max_bend, 0.0, 1.0);

            // Grazing altitude z corresponding to this refraction bend angle:
            // theta(z) = max_bend * exp(-z / H) -> atmo_depth = exp(-z / H)
            float z_km = -H_scale * log(max(atmo_depth, 1e-5));

            // 1. Rayleigh grazing optical depth at altitude z
            vec3 tau_R = tau_grazing_0 * atmo_depth;

            // 2. Stratospheric ozone layer absorption along grazing ray (Chappuis band)
            float z_diff = (z_km - z_peak) / sigma_z;
            vec3 tau_O3 = ozone_param.xyz * exp(-0.5 * z_diff * z_diff);

            // Total optical depth and physical spectral transmittance
            vec3 tau_total = tau_R + tau_O3;
            vec3 atmo_transmittance = exp(-tau_total);

            // Physical Atmospheric Ring Geometric Dilution (1/D falloff)
            // The atmospheric lens is a ring, not a point lens, so light diverges in 1D, not 2D.
            // Geometric intensity factor f = (2 * H_scale) / (alpha * D)
            float ring_intensity = (2.0 * H_scale) / max(alpha * dist_km, 1e-9);
            
            // Smooth surface grazing fade to zero at solid body boundary (h = 0, atmo_depth = 1)
            float body_surface_fade = smoothstep(1.0, 0.75, atmo_depth);
            
            float refraction_intensity = ring_intensity * (1.0 - atmo_depth * 0.7) * body_surface_fade;

            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            sh += atmo_transmittance * refraction_intensity * atmo_blend;
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
    if (f_clip_z <= 0.0) discard;

    // Fully-subpixel bodies are rendered exclusively by the point-light pass
    // (alpha == 0 here). Discard instead of emitting a zero-alpha fragment so
    // the 3 px min-size-clamped geometry does NOT write depth: an invisible
    // subpixel planet would otherwise depth-occlude the subpixel sun sprite
    // behind it as a full 3 px disk (phantom transit / eclipse black-out).
    if (f_subpixel_factor > 0.999) {
        discard;
    }

    // To avoid double-rendering and Z-fighting when CULL_FACE is disabled:
    // If the camera is outside the bounding sphere, we only want the front face.
    // If the camera is inside the bounding sphere, we only want the back face.
    float dist_to_center = length(f_center_pos - u_camera_pos);
    if (dist_to_center > f_bounding_radius) {
        if (!gl_FrontFacing) discard;
    } else {
        if (gl_FrontFacing) discard;
    }

    // Body-local approach: separate small per-pixel mesh offset from large per-instance offset
    vec3 pole_n_pre = length(f_pole) > 1e-4 ? normalize(f_pole) : vec3(0.0, 1.0, 0.0);
    vec3 scaled_local = f_local_pos;
    if (f_oblateness > 0.0) {
        float pp = dot(f_local_pos, pole_n_pre);
        scaled_local -= pole_n_pre * (pp * f_oblateness);
    }
    vec3 f_local_bounding = scaled_local * f_bounding_radius;  // small, per-pixel, smooth
    vec3 cam_to_center = f_center_pos - u_camera_pos;          // flat, per-instance constant
    vec3 view_ray = normalize(f_local_bounding + cam_to_center);
    vec3 ray_dir = view_ray;
    vec3 ray_origin = u_camera_pos;
    bool is_refract_host = length(f_center_pos - u_refract_center) < 1e-7;

    // Black-hole lens host: the sphere mesh renders ONLY the event-horizon
    // shadow silhouette. Any sightline whose impact parameter falls below the
    // spin-aware capture radius b_c(phi) plunges through the horizon — render
    // pure black and KEEP its depth so lensed stars / orbits / subpixel lights
    // behind it are occluded. Outside the shadow the horizon itself is
    // invisible, so the rest of the mesh is discarded.
    bool is_grav_lens_host = u_grav_lens_enabled && u_grav_lens_type == 3 && u_grav_lens_rs > 1e-6
        && length(f_center_pos - u_grav_lens_center) < 1e-6;
    if (is_grav_lens_host) {
        vec3 C_km = (u_camera_pos - u_grav_lens_center) * u_au_to_km;
        // d_km = 0: the fragment ray terminates at the capture sphere, so the
        // capture test must not additionally require the periapsis to lie in
        // front of the fragment (silhouette rays have s_min == d_frag).
        bool is_sh = false;
        compute_gravitational_deflection(C_km, view_ray, 0.0, is_sh);
        // Min-size clamp regime: the whole (inflated) mesh IS the shadow dot —
        // the capture test must not shrink it back below the 3 px floor.
        if (f_apparent_px < f_clamped_min_px) is_sh = true;
        if (is_sh) {
            out_color = vec4(0.0, 0.0, 0.0, 1.0);
            gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
            return;
        }
        discard;
    }

    float s_min_au = 0.0;
    if (u_refract_max_bend > 1e-6) {
        if (!is_refract_host) {
            vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
            float d_km = length(cam_to_center * u_au_to_km);
            // Total (un-parallaxed) bend: the anchored rotation below applies
            // the (1 - s_min/d) parallax geometrically; using the parallaxed
            // alpha here would double-count it.
            float alpha = compute_refraction_total(C_km, view_ray, d_km);
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
        } else {
            // Terrestrial Refraction for host planet ground (horizon extension)
            vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
            float r_cam = length(C_km);

            vec3 pole_n = length(f_pole) > 1e-4 ? normalize(f_pole) : vec3(0.0, 1.0, 0.0);
            float f_scale = (f_oblateness > 0.0 && f_oblateness < 0.99) ? (1.0 / (1.0 - f_oblateness)) : 1.0;
            vec3 C_scaled = C_km + pole_n * (dot(C_km, pole_n) * (f_scale * f_scale - 1.0));
            vec3 local_up = normalize(C_scaled);

            float local_refract_radius = u_refract_radius;
            if (f_oblateness > 0.001 && f_oblateness < 0.99) {
                vec3 P_dir = r_cam > 1e-6 ? (C_km / r_cam) : vec3(0.0, 1.0, 0.0);
                float cos_t = abs(dot(P_dir, pole_n));
                float k = 1.0 / (1.0 - f_oblateness);
                float denom = sqrt(max(1e-6, 1.0 + (k * k - 1.0) * cos_t * cos_t));
                local_refract_radius = u_refract_radius / denom;
            }

            float h = r_cam - local_refract_radius;
            if (h < u_refract_scale_height * 15.0) {
                float density = exp(-max(h, 0.0) / max(1e-4, u_refract_scale_height));
                float max_terr_alpha = min(0.05, 0.5 * u_refract_max_bend * density);
                
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

    // 2. Gravitational Lensing Deflection (background bodies viewed past the
    // lens). Composed after atmospheric refraction: rotates the (possibly
    // already-refracted) ray direction by the gravitational deflection angle
    // and re-anchors the ray origin at the lens periapsis.
    if (u_grav_lens_enabled && u_grav_lens_rs > 1e-6) {
        vec3 C_km = (u_camera_pos - u_grav_lens_center) * u_au_to_km;
        float d_body_km = length(cam_to_center * u_au_to_km);

        bool is_sh = false;
        float alpha_gr = compute_gravitational_deflection(C_km, view_ray, d_body_km, is_sh);
        if (is_sh) {
            discard; // Ray captured by the black hole shadow
        }
        if (alpha_gr > 1e-7) {
            vec3 u_dir = C_km - view_ray * dot(C_km, view_ray);
            float u_len = length(u_dir);
            if (u_len > 1e-5) {
                u_dir /= u_len;
                ray_dir = normalize(ray_dir * cos(alpha_gr) - u_dir * sin(alpha_gr));

                float local_s_min = -dot(u_camera_pos - u_grav_lens_center, view_ray);
                if (local_s_min > 0.0) {
                    s_min_au = max(s_min_au, local_s_min);
                    ray_origin = u_camera_pos + s_min_au * (view_ray - ray_dir);
                }
            }
        }
    }

    vec3 pole_n = pole_n_pre;
    // Precise ray origin in body-local coordinates, avoiding catastrophic cancellation
    // By starting from the precise bounding mesh vertex and applying the relative refraction shift.
    float d_bounding = length(f_local_bounding + cam_to_center);
    vec3 O_local = f_local_bounding + (d_bounding - s_min_au) * (ray_dir - view_ray);
    vec3 D = ray_dir;

    float f_scale = (f_oblateness > 0.0 && f_oblateness < 0.99) ? (1.0 / (1.0 - f_oblateness)) : 1.0;
    vec3 O_sph = O_local + pole_n * (dot(O_local, pole_n) * (f_scale - 1.0));
    vec3 D_sph = D + pole_n * (dot(D, pole_n) * (f_scale - 1.0));

    float a = dot(D_sph, D_sph);
    float inv_a = 1.0 / max(a, 1e-12);
    
    vec3 cross_vec = cross(O_sph, D_sph);
    float d2 = dot(cross_vec, cross_vec) * inv_a;
    float eff_radius = max(f_radius, f_final_radius);
    float R2 = eff_radius * eff_radius;
    
    if (d2 > R2) discard;
    
    float h2 = (R2 - d2) * inv_a;
    float h = sqrt(max(0.0, h2));
    
    float t_closest = -dot(O_sph, D_sph) * inv_a;
    
    if (h2 <= 0.0) discard;
    float t_hit_local = t_closest - h;  // front face (negative, going back toward camera)
    bool is_inside = false;
    vec3 true_cam_local = -cam_to_center;
    vec3 true_cam_sph = true_cam_local + pole_n * (dot(true_cam_local, pole_n) * (f_scale - 1.0));
    if (length(true_cam_sph) < f_final_radius) {
        t_hit_local = t_closest + h;
        is_inside = true;
    }

    vec3 P_rel = O_local + ray_dir * t_hit_local;  // body-local hit position, precise
    vec3 hit_world_pos = P_rel + f_center_pos;  // for clip_pos + depth only

    vec3 P_scaled = P_rel + pole_n * (dot(P_rel, pole_n) * (f_scale * f_scale - 1.0));
    vec3 hit_normal = normalize(P_scaled);
    if (is_inside && !u_is_cloud_pass) {
        hit_normal = -hit_normal;
    }
    vec3 hit_local_pos = P_rel / max(1e-6, f_radius);

    // Compute depth: we need the world-space t from camera for depth buffer
    // cam_to_hit = cam_to_center + P_rel (both body-local-scale except cam_to_center)
    vec4 clip_pos = projection * view * vec4(hit_world_pos, 1.0);
    float hit_clip_z = clip_pos.w;
    if (hit_clip_z <= 0.0) discard;
    gl_FragDepth = log2(max(1e-6, u_depth_C * hit_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);

    vec3 v_world_pos = hit_world_pos;
    vec3 v_normal = hit_normal + f_normal * 1e-12;
    vec3 v_local_pos = hit_local_pos;

    if (f_is_star > 0.5) {
        if (u_is_cloud_pass) discard;
        // Star is self-luminous, apply quadratic limb darkening
        vec3 V = normalize(-(cam_to_center + P_rel));
        vec3 N = normalize(v_normal);
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
            if (length((u_stars_pos_radius[s].xyz - f_center_pos) - P_rel) < u_stars_pos_radius[s].w * 1.5) {
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
        
        float mesh_weight_star = 1.0 - f_subpixel_factor;
        // Flux compensation for the min-size clamp: the mesh is inflated to
        // f_clamped_min_px (3 px) apparent diameter, which mid-fade would
        // overweight its total flux by (3/apparent)^2 relative to the true
        // solid angle. Scaling by (apparent/clamped)^2 keeps the composite
        // mesh+point flux exactly proportional to the true solid angle at
        // every apparent size, so the hand-off to the point light is
        // flux-continuous with no mid-fade bump.
        float flux_comp = (f_apparent_px < f_clamped_min_px)
            ? (f_apparent_px * f_apparent_px) / (f_clamped_min_px * f_clamped_min_px)
            : 1.0;
        final_star_color *= mesh_weight_star * flux_comp;
        float star_alpha = mesh_weight_star;
        out_color = vec4(final_star_color, star_alpha);
    } else {
        vec3 N = normalize(v_normal);
        float spec_intensity = 0.0;
        vec3 local_f_color = f_color;
        float cloud_frag_alpha = 1.0;
        vec4 cloud_tex = vec4(0.0);

        vec3 tangent = vec3(0.0);
        vec3 bitangent = vec3(0.0);
        float rot_sin = 0.0;
        float rot_cos = 1.0;
        bool has_clouds = false;
        sampler2D s_clouds;

        if (f_tex_idx > 0.0) {
            vec3 ref = vec3(0.0, 1.0, 0.0);
            if (abs(dot(pole_n, ref)) > 0.999) {
                ref = vec3(1.0, 0.0, 0.0);
            }
            tangent = normalize(cross(pole_n, ref));
            bitangent = normalize(cross(pole_n, tangent));

            rot_sin = sin(-f_rotation_angle);
            rot_cos = cos(-f_rotation_angle);

            vec3 p = normalize(v_local_pos);
            vec3 p_local = vec3(dot(p, tangent), dot(p, pole_n), dot(p, bitangent));
            vec3 p_rot = vec3(
                p_local.x * rot_cos - p_local.z * rot_sin,
                p_local.y,
                p_local.x * rot_sin + p_local.z * rot_cos
            );

            float u = 0.5 + atan(p_rot.z, p_rot.x) / (2.0 * PI);
            float v = 0.5 - asin(clamp(p_rot.y, -1.0, 1.0)) / PI;

            vec2 uv = vec2(u, v);
            vec2 dx = dFdx(uv);
            vec2 dy = dFdy(uv);

            if (dx.x > 0.5) dx.x -= 1.0;
            else if (dx.x < -0.5) dx.x += 1.0;

            if (dy.x > 0.5) dy.x -= 1.0;
            else if (dy.x < -0.5) dy.x += 1.0;

            uint b_tex_slot = uint(f_tex_idx - 1.0);
            BodyTextures bt = u_body_textures[b_tex_slot];
            if ((bt.clouds.x | bt.clouds.y) != 0u) {
                has_clouds = true;
                s_clouds = sampler2D(bt.clouds);
                cloud_tex = textureGrad(s_clouds, uv, dx, dy);
            }

            if (u_is_cloud_pass) {
                cloud_frag_alpha = cloud_tex.a;
                if (cloud_frag_alpha < 0.005) discard;

                // Cloud albedo: for grayscale cloud maps (where R=G=B=A), render as bright white scatterers;
                // for colored textures, preserve their custom RGB albedo.
                bool is_grayscale = abs(cloud_tex.r - cloud_tex.g) < 0.01 && abs(cloud_tex.g - cloud_tex.b) < 0.01;
                local_f_color = is_grayscale ? vec3(0.95) : cloud_tex.rgb;
                spec_intensity = 0.0;
                N = normalize(v_normal);
            } else {
                sampler2D s_diffuse = sampler2D(bt.diffuse);
                vec4 tex_color = textureGrad(s_diffuse, uv, dx, dy);
                // Decode sRGB to Linear
                local_f_color = pow(tex_color.rgb, vec3(2.2));

                // Analytical TBN Mapping
                sampler2D s_normal = sampler2D(bt.normal);
                vec3 map_normal = textureGrad(s_normal, uv, dx, dy).rgb;
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

                sampler2D s_specular = sampler2D(bt.specular);
                spec_intensity = textureGrad(s_specular, uv, dx, dy).r;

                // Boost dark ocean albedo using specular map as a mask
                vec3 ocean_base = vec3(0.003, 0.015, 0.06); // Deep blue linear albedo
                local_f_color = mix(local_f_color, max(local_f_color, ocean_base), spec_intensity);
            }
        } else {
            if (u_is_cloud_pass) discard;
        }

        vec3 total_diffuse_color = vec3(0.0);
        vec3 total_specular_color = vec3(0.0);
        vec3 V = normalize(-(cam_to_center + P_rel));

        for (int s = 0; s < u_num_stars; s++) {
            vec3 star_pos = u_stars_pos_radius[s].xyz;
            float star_radius = u_stars_pos_radius[s].w;
            vec3 frag_to_star = (star_pos - f_center_pos) - P_rel;
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
            float effective_NdotL = NdotL;
            if (u_is_cloud_pass) {
                // Mean cloud deck ~0.8H (~500 hPa); higher than the old 0.35H haze
                // so the terminator delay sqrt(2*z/R) matches high clouds staying lit.
                float cloud_h_km = f_my_scale_height * 0.8;
                float R_km = max(f_radius * u_au_to_km, 1e-6);
                float term_offset = sqrt(max(0.0, 2.0 * cloud_h_km / R_km));
                effective_NdotL += term_offset;
            }
            float diffuse = clamp((effective_NdotL + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);

            vec3 incoming_light_tint = vec3(1.0);
            vec3 direct_light_tint = vec3(1.0);
            vec3 diffuse_light_tint = vec3(0.0);
            if (f_my_atmo_h > 0.0) {
                float R_km = max(f_radius * u_au_to_km, 1e-6);
                float H_scale = max(f_my_scale_height, 1e-3);
                float mu = u_is_cloud_pass ? max(effective_NdotL, 0.0) : max(NdotL, 0.0);

                // Geometric relative air mass
                float am = (sqrt(R_km*R_km*mu*mu + 2.0*R_km*H_scale + H_scale*H_scale) - R_km*mu) / H_scale;

                // Split total column OD into Rayleigh+mixed (H_R) and Mie (H_Mie).
                // f_my_atmo_tint = total; f_my_mie_tau = Mie part (new uniform).
                vec3 tau_mie_vert = max(f_my_mie_tau, vec3(0.0));
                vec3 tau_rm_vert = max(max(f_my_atmo_tint, vec3(0.0)) - tau_mie_vert, vec3(0.0));

                if (u_is_cloud_pass) {
                    float cloud_h_km = f_my_scale_height * 0.8;
                    float H_mie = max(f_my_mie_h, 0.5);
                    tau_rm_vert *= exp(-cloud_h_km / H_scale);
                    // Mie haze is concentrated near the surface: at cloud altitude
                    // most of it is already below, so attenuate by its own scale height.
                    tau_mie_vert *= exp(-cloud_h_km / H_mie);
                }
                vec3 tau_vertical = tau_rm_vert + tau_mie_vert;

                // Direct sunlight extinction
                vec3 tau_direct = tau_vertical * am;

                // Stratospheric ozone (Chappuis band) absorbs the orange/red
                // direct beam heavily at high air mass — this is what keeps the
                // real terminator only faintly warm instead of saturated red.
                // The layer sits above cloud decks, so the full vertical column
                // applies in both passes; its Chapman air mass uses the layer's
                // own altitude and Gaussian width.
                if (dot(f_my_o3_tau, f_my_o3_tau) > 0.0) {
                    float o3_w = f_my_o3_layer.y;
                    float R_o3 = R_km + max(f_my_o3_layer.x, 0.0);
                    float am_o3 = (sqrt(R_o3*R_o3*mu*mu + 2.0*R_o3*o3_w + o3_w*o3_w) - R_o3*mu) / o3_w;
                    tau_direct += f_my_o3_tau * am_o3;
                }

                direct_light_tint = exp(-tau_direct);

                // Downward diffuse daylight: two-term Eddington, Rayleigh g=0 and
                // Mie g=0.8, so blue sky-shine and white haze keep correct hues.
                // The scattered-out fraction must use the SLANT optical depth
                // (tau_v * am): it saturates toward 1 at low sun instead of
                // decaying ~ mu, which left the ground far too dark at dusk.
                vec3 tau_slant_rm = tau_rm_vert * am;
                vec3 tau_slant_m = tau_mie_vert * am;
                vec3 tau_diff_rm = tau_rm_vert; // (1-0)*tau
                vec3 tau_diff_m = 0.2 * tau_mie_vert; // (1-0.8)*tau
                vec3 diffuse_rm = (vec3(1.0) - exp(-tau_slant_rm)) * (max(0.0, mu) / (vec3(1.0) + 0.75 * tau_diff_rm));
                vec3 diffuse_m = (vec3(1.0) - exp(-tau_slant_m)) * (max(0.0, mu) / (vec3(1.0) + 0.75 * tau_diff_m));
                diffuse_light_tint = diffuse_rm + diffuse_m;

                incoming_light_tint = direct_light_tint + diffuse_light_tint;
            }

            // === Ray-Traced Cloud Shadows on Surface ===
            // Must use the same mean deck height as the cloud shell (0.8H).
            if (!u_is_cloud_pass && f_tex_idx > 0.0 && has_clouds && f_my_scale_height > 0.0) {
                float cloud_h_km = f_my_scale_height * 0.8;
                float cloud_offset_au = cloud_h_km / max(1e-6, u_au_to_km);
                float eff_r = max(f_radius, f_final_radius);
                float R_cloud = eff_r + cloud_offset_au;

                vec3 O_sph_c = P_rel + pole_n * (dot(P_rel, pole_n) * (f_scale - 1.0));
                vec3 D_sph_c = L + pole_n * (dot(L, pole_n) * (f_scale - 1.0));

                float a_c = dot(D_sph_c, D_sph_c);
                float b_c = dot(O_sph_c, D_sph_c);
                float c_c = dot(O_sph_c, O_sph_c) - R_cloud * R_cloud;
                float disc_c = b_c * b_c - a_c * c_c;

                if (disc_c > 0.0) {
                    float t_cloud = (-b_c + sqrt(disc_c)) / max(a_c, 1e-12);
                    if (t_cloud > 0.0) {
                        vec3 hit_cloud_sph = O_sph_c + t_cloud * D_sph_c;
                        vec3 hit_cloud_local = hit_cloud_sph - pole_n * (dot(hit_cloud_sph, pole_n) * (1.0 - 1.0 / f_scale));
                        vec3 p_c = normalize(hit_cloud_local);

                        vec3 p_local_c = vec3(dot(p_c, tangent), dot(p_c, pole_n), dot(p_c, bitangent));
                        vec3 p_rot_c = vec3(
                            p_local_c.x * rot_cos - p_local_c.z * rot_sin,
                            p_local_c.y,
                            p_local_c.x * rot_sin + p_local_c.z * rot_cos
                        );

                        float u_c = 0.5 + atan(p_rot_c.z, p_rot_c.x) / (2.0 * PI);
                        float v_c = 0.5 - asin(clamp(p_rot_c.y, -1.0, 1.0)) / PI;

                        vec4 cloud_shadow_sample = textureLod(s_clouds, vec2(u_c, v_c), 0.0);
                        float shadow_alpha = cloud_shadow_sample.a;
                        // Direct beam is nearly fully blocked by the cloud, but
                        // diffuse skylight arrives from the whole sky dome and
                        // must only be mildly reduced inside the shadow.
                        incoming_light_tint = direct_light_tint * (1.0 - shadow_alpha * 0.85)
                                            + diffuse_light_tint * (1.0 - shadow_alpha * 0.25);
                    }
                }
            }

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

                vec3 frag_to_caster = (caster_pos - f_center_pos) - P_rel;
                float t_proj = dot(frag_to_caster, L);
                if (t_proj < 0.0) continue;

                float dist_sq = dot(frag_to_caster, frag_to_caster);
                if (dist_sq < caster_r * caster_r * 1.0404) continue; // Skip self

                float dist_to_caster = sqrt(dist_sq);
                vec3 cross_vec = cross(frag_to_caster, L);
                float perp_sq = dot(cross_vec, cross_vec);

                // Bounding cone early out using maximum equatorial radius
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

                // Perfect Bounding Cone Early Out (Zero Artifacts)
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

                    float t = dot((plane_center - f_center_pos) - P_rel, plane_normal) / denom;
                    if (t <= 0.0 || t >= dist_to_star) continue;

                    vec3 hit = f_center_pos + P_rel + t * L;
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

            total_diffuse_color += star_color * incoming_light_tint * diffuse * shadow;
            total_specular_color += star_color * direct_light_tint * specular * shadow;
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

            vec3 frag_to_center = (ring_center - f_center_pos) - P_rel;
            float dist_to_center = length(frag_to_center);

            // Fast distance cull: ignore rings if we are too far away
            if (dist_to_center > outer_r * 20.0 || dist_to_center < 1e-6) continue;

            int host_caster_idx = -1;
            for (int j = 0; j < u_num_casters; j++) {
                // Positions are bit-identical copies from the same CPU buffer
                if (distance(u_casters[j].xyz, ring_center) < 1e-7) {
                    host_caster_idx = j;
                    break;
                }
            }

            bool is_host_planet = false;
            if (host_caster_idx >= 0) {
                float host_radius = u_casters[host_caster_idx].w;
                is_host_planet = (length((u_casters[host_caster_idx].xyz - f_center_pos) - P_rel) < host_radius * 1.5);
            } else {
                is_host_planet = (dist_to_center < inner_r);
            }

            if (!is_host_planet) continue;

            float frag_elevation = dot(N, ring_normal);
            float moon_h = -dot(frag_to_center, ring_normal);

            for (int s = 0; s < u_num_stars; s++) {
                vec3 star_pos = u_stars_pos_radius[s].xyz;
                float star_radius = u_stars_pos_radius[s].w;
                vec3 frag_to_star = (star_pos - f_center_pos) - P_rel;
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

        // Mesh transition weight for seamless hand-off with point-light quad (apparent_px in [1.5, 2.5])
        float mesh_weight_planet = 1.0 - f_subpixel_factor;
        // Same flux compensation as the star branch: the 3 px min-size clamp
        // inflates the disk area, so dim the clamped mesh by
        // (apparent/clamped)^2 to keep total flux tied to the true solid angle
        // (the point pass already scales as apparent^2, so the composite is
        // exactly flux-continuous across the whole transition).
        float flux_comp_p = (f_apparent_px < f_clamped_min_px)
            ? (f_apparent_px * f_apparent_px) / (f_clamped_min_px * f_clamped_min_px)
            : 1.0;
        final_color *= mesh_weight_planet * flux_comp_p;
        float planet_alpha = mesh_weight_planet;

        if (u_is_cloud_pass) {
            float trans_view = 1.0;
            float horizon_fade = 1.0;

            if (f_my_atmo_h > 0.0) {
                float R_km = max(f_radius * u_au_to_km, 1e-6);
                float H_scale = max(f_my_scale_height, 1e-3);
                // Same mean deck + per-species split as sun path for consistency.
                float cloud_h_km = f_my_scale_height * 0.8;
                float H_mie = max(f_my_mie_h, 0.5);

                vec3 view_norm = is_inside ? -N : N;
                float NdotV = max(dot(view_norm, V), 0.0);
                float am_view = (sqrt(R_km * R_km * NdotV * NdotV + 2.0 * R_km * H_scale + H_scale * H_scale) - R_km * NdotV) / H_scale;

                float d_cam_km = length(cam_to_center + P_rel) * u_au_to_km;
                float path_fraction = clamp(d_cam_km / max(am_view * H_scale, 1e-3), 0.0, 1.0);

                vec3 tau_mie_vert = max(f_my_mie_tau, vec3(0.0));
                vec3 tau_rm_vert = max(max(f_my_atmo_tint, vec3(0.0)) - tau_mie_vert, vec3(0.0));
                vec3 tau_vert_cloud = tau_rm_vert * exp(-cloud_h_km / H_scale)
                                    + tau_mie_vert * exp(-cloud_h_km / H_mie);
                vec3 tau_view = tau_vert_cloud * am_view * path_fraction;

                // Photopic optical depth along view ray for natural perceptual fading
                float tau_eff = dot(tau_view, vec3(0.2126, 0.7152, 0.0722));
                trans_view = exp(-tau_eff);

                // Smooth horizon falloff to prevent grazing edge aliasing
                horizon_fade = smoothstep(0.0, 0.05, NdotV);
            }

            planet_alpha = cloud_frag_alpha * trans_view * horizon_fade * mesh_weight_planet;
        }

        out_color = vec4(final_color, planet_alpha);
    }
}
