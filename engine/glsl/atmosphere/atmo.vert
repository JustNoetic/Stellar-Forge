#version 460 core
in vec3 in_position;

#define MAX_CASTERS 64
#define MAX_STARS 16
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
    float _atmo_pad1;
};

uniform vec3 u_refract_center;
uniform float u_refract_radius;
uniform float u_refract_max_bend;
uniform float u_refract_scale_height;
uniform vec3 u_camera_pos;

float compute_refraction_angle(vec3 C, vec3 V, float d) {
    if (u_refract_max_bend <= 1e-6) return 0.0;
    float s_min = -dot(C, V);
    vec3 P_min = C + s_min * V;
    float r_min = length(P_min);
    
    if (r_min > u_refract_radius + u_refract_scale_height * 15.0) return 0.0;
    
    float r_min_clamped = max(r_min, u_refract_radius - u_refract_scale_height); 
    float delta_rmin = u_refract_max_bend * exp(-(r_min_clamped - u_refract_radius) / max(1e-4, u_refract_scale_height));
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

vec3 apply_refraction(vec3 world_pos, vec3 cam_pos) {
    if (u_refract_max_bend <= 1e-6) return world_pos;
    vec3 C_km = (cam_pos - u_refract_center) * u_au_to_km;
    vec3 P_km = (world_pos - u_refract_center) * u_au_to_km;
    vec3 true_vec = P_km - C_km;
    float d_km = length(true_vec);
    if (d_km <= 1e-5) return world_pos;
    vec3 V = true_vec / d_km;
    float alpha = compute_refraction_angle(C_km, V, d_km);
    if (alpha <= 1e-7) return world_pos;
    vec3 u_dir = C_km - V * dot(C_km, V);
    float u_len = length(u_dir);
    if (u_len <= 1e-5) return world_pos;
    u_dir /= u_len;
    vec3 V_app = V * cos(alpha) + u_dir * sin(alpha);
    return cam_pos + V_app * (d_km / u_au_to_km);
}

out vec3 f_world_pos;
out vec3 f_local_pos;
out float f_clip_z;

void main() {
    float dist = length(u_body_offset - u_camera_pos);
    float atmo_expand = (u_refract_max_bend > 0.0) ? (dist * tan(u_refract_max_bend) * 1.5 + u_atmo_radius_au * 0.08) : (u_atmo_radius_au * 0.03);
    float bounding_radius = u_atmo_radius_au + atmo_expand;
    float scale_factor = bounding_radius / max(1e-6, u_atmo_radius_au);

    vec3 world_pos = in_position * bounding_radius + u_body_offset;
    f_world_pos = world_pos;
    f_local_pos = in_position * scale_factor;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    f_clip_z = gl_Position.w;
}
