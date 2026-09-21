#version 460 core
#define PI 3.14159265358979323846

in vec2 f_uv;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_transmittance;

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
    vec4  u_precomp_opt;         // x: inv_h_rayleigh, y: inv_h_mie, z: inv_ozone_width, w: max_bend
    vec4  u_precomp_mie;         // x: c1, y: c2, z: c3, w: unused
    vec4  u_star_color_irrad[4]; // rgb = star_color * irradiance * atmo_sun_intensity, w = sin_star
    vec4  u_star_dir_sph_eff[4]; // xyz = sun_dir_sph_const, w = cos_sun_eff
    vec4  u_star_pos_local[4];   // xyz = sun_pos_local_km, w = effective_star_rad
    vec4  u_star_solstice[4];    // x = solstice_factor, y = sun_pole_dot, z = dist_star_au, w = star_radius_au
};

uniform vec3 u_cam_pos; // Camera position in planet local frame (km)
uniform vec3 u_sun_dir; // Sunlight direction in planet local frame (normalized)

uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;

#include "common/sun_terminator.glsl"

vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float delta = b * b - c;
    if (delta < 0.0) return vec2(1e10, -1e10);
    float sq = sqrt(delta);
    return vec2(-b - sq, -b + sq);
}

vec3 get_transmittance(float r, float cos_theta) {
    float h_norm = clamp((r - u_planet_radius_km) / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float v = sqrt(h_norm);
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

void main() {
    float D = length(u_cam_pos);
    vec3 u_zenith = u_cam_pos / max(D, 1e-6);
    float D_clamped = max(D, u_planet_radius_km);
    bool cam_inside = (D <= u_atmo_radius_km);
    vec3 L_sun = normalize(u_sun_dir);

    // Build continuous orthonormal basis aligned with local Zenith and Sun Azimuth
    vec3 L_proj = L_sun - dot(L_sun, u_zenith) * u_zenith;
    vec3 x_basis;
    float len_L_proj = length(L_proj);
    if (len_L_proj > 1e-4) {
        x_basis = L_proj / len_L_proj;
    } else {
        vec3 arb = (abs(u_zenith.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        x_basis = normalize(cross(u_zenith, arb));
    }
    vec3 y_basis = cross(u_zenith, x_basis);

    // Map f_uv.x in [0, 1] to azimuth phi in [0, 2*PI]
    float phi = f_uv.x * 2.0 * PI;

    vec3 ray_origin;
    vec3 V;
    float s_start = 0.0;
    float s_end = 0.0;
    bool hits_ground = false;

    if (cam_inside) {
        // Piecewise non-linear elevation mapping centered at horizon (Hillaire 2020)
        float sin_horizon = clamp(u_planet_radius_km / D_clamped, 0.0, 1.0);
        float theta_horizon = PI - asin(sin_horizon);

        float theta = 0.0;
        if (f_uv.y <= 0.5) {
            // Ground disk: [theta_horizon, PI]
            // Row 0 is Nadir (theta = PI), row 127 is Horizon (theta = theta_horizon)
            float t = (0.5 - f_uv.y) / 0.5;
            theta = theta_horizon + (t * t) * (PI - theta_horizon);
            hits_ground = true;
        } else {
            // Sky dome: [0, theta_horizon]
            // Row 128 is Horizon (theta = theta_horizon), row 255 is Zenith (theta = 0)
            float t = (f_uv.y - 0.5) / 0.5;
            theta = theta_horizon * (1.0 - t * t);
            hits_ground = false;
        }

        V = normalize(sin(theta) * (cos(phi) * x_basis + sin(phi) * y_basis) + cos(theta) * u_zenith);
        ray_origin = u_zenith * D_clamped;

        vec2 t_atmo = raySphereIntersect(ray_origin, V, u_atmo_radius_km);
        if (t_atmo.y < 0.0) {
            out_color = vec4(0.0, 0.0, 0.0, 1.0);
            out_transmittance = vec4(1.0);
            return;
        }

        s_start = max(0.0, t_atmo.x);
        s_end = t_atmo.y;

        if (hits_ground) {
            vec2 t_planet = raySphereIntersect(ray_origin, V, u_planet_radius_km);
            if (t_planet.x > 0.0) {
                s_end = min(s_end, t_planet.x);
            } else {
                s_end = 0.0;
            }
        }
    } else {
        // Space observer: looking towards planet
        // Exact mapping from impact parameter r_ca and azimuth phi
        float r_ca;
        if (f_uv.y <= 0.5) {
            // Ground disk: impact parameter r_ca in [0, R_planet]
            float t = (0.5 - f_uv.y) / 0.5;
            r_ca = u_planet_radius_km * max(0.0, 1.0 - t * t);
            hits_ground = true;
        } else {
            // Atmosphere limb: impact parameter r_ca in [R_planet, R_atmo]
            float t = (f_uv.y - 0.5) / 0.5;
            r_ca = u_planet_radius_km + (t * t) * (u_atmo_radius_km - u_planet_radius_km);
            hits_ground = false;
        }

        vec3 e_phi = cos(phi) * x_basis + sin(phi) * y_basis;
        float sin_alpha = clamp(r_ca / max(D, u_atmo_radius_km), 0.0, 1.0);
        float cos_alpha = sqrt(max(0.0, 1.0 - sin_alpha * sin_alpha));

        V = normalize(sin_alpha * e_phi - cos_alpha * u_zenith);
        ray_origin = r_ca * cos_alpha * e_phi + r_ca * sin_alpha * u_zenith;

        float r_ca_sq = r_ca * r_ca;
        s_start = -sqrt(max(0.0, u_atmo_radius_km * u_atmo_radius_km - r_ca_sq));

        if (hits_ground) {
            s_end = -sqrt(max(0.0, u_planet_radius_km * u_planet_radius_km - r_ca_sq));
        } else {
            s_end = +sqrt(max(0.0, u_atmo_radius_km * u_atmo_radius_km - r_ca_sq));
        }
    }

    if (s_start >= s_end) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        out_transmittance = vec4(1.0);
        return;
    }

    // Physical scattering parameters
    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M);
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;

    float inv_h_rayleigh = (u_precomp_opt.x > 1e-6) ? u_precomp_opt.x : (1.0 / max(u_h_rayleigh, 1e-3));
    float inv_h_mie = (u_precomp_opt.y > 1e-6) ? u_precomp_opt.y : (1.0 / max(u_h_mie, 1e-3));
    float inv_ozone_width = (u_precomp_opt.z > 1e-6) ? u_precomp_opt.z : (1.0 / max(u_ozone_width_km, 1e-3));

    const int NUM_STEPS = 32;
    float ds = (s_end - s_start) / float(NUM_STEPS);

    vec3 total_rayleigh = vec3(0.0);
    vec3 total_mie = vec3(0.0);
    vec3 total_ms = vec3(0.0);
    vec3 current_transmittance = vec3(1.0);

    for (int i = 0; i < NUM_STEPS; i++) {
        float s = s_start + (float(i) + 0.5) * ds;
        vec3 P = ray_origin + s * V;
        float r = length(P);
        float altitude = max(0.0, r - u_planet_radius_km);

        float rho_R = exp(-altitude * inv_h_rayleigh);
        float rho_M = exp(-altitude * inv_h_mie);
        float t_ozone = (altitude - u_ozone_peak_km) * inv_ozone_width;
        float rho_O = exp(-(t_ozone * t_ozone));

        vec3 step_extinction = beta_R * rho_R + beta_M * rho_M + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;
        vec3 step_transmittance = exp(-step_extinction * ds);
        vec3 int_factor = (vec3(1.0) - step_transmittance) / max(step_extinction, vec3(1e-6));

        float light_cos_theta = dot(P, L_sun) / r;
        float sin_planet = u_planet_radius_km / max(r, u_planet_radius_km + 0.01);
        float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));

        // Terminator: stellar-disc rise/set with refraction-extended penumbra
        // (eff_star_rad = sin_star + max_bend), shared with the Mode 1/2 raymarchers.
        vec2 term = sun_terminator(light_cos_theta, sin_planet, cos_planet,
                                   u_star_pos_local[0].w, u_star_dir_sph_eff[0].w);
        float vis_fraction = term.x;
        vec3 trans_to_sun = (vis_fraction > 1e-4) ? get_transmittance(r, term.y) : vec3(0.0);

        vec3 sample_attenuation = current_transmittance * trans_to_sun * vis_fraction * int_factor;
        total_rayleigh += rho_R * sample_attenuation;
        total_mie      += rho_M * sample_attenuation;

        float h_norm = clamp(altitude / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
        float ms_u = 0.5 + 0.5 * sign(light_cos_theta) * sqrt(abs(light_cos_theta));
        float ms_v = sqrt(h_norm);
        vec3 psi = textureLod(u_multi_scatter_lut, vec2(ms_u, ms_v), 0.0).rgb;
        total_ms += (beta_R * rho_R + beta_M * rho_M) * psi * current_transmittance * int_factor;

        current_transmittance *= step_transmittance;
        if (all(lessThan(current_transmittance, vec3(1e-5)))) break;
    }

    float cos_theta_sun = dot(V, L_sun);
    float phase_R = (3.0 / (16.0 * PI)) * (1.0 + cos_theta_sun * cos_theta_sun);

    float g = clamp(u_mie_g, 0.0, 0.88);
    float c1 = (u_precomp_mie.x > 1e-6) ? u_precomp_mie.x : ((3.0 / (8.0 * PI)) * ((1.0 - g * g) / (2.0 + g * g)));
    float c2 = (u_precomp_mie.y > 1e-6) ? u_precomp_mie.y : (1.0 + g * g);
    float c3 = (u_precomp_mie.z > 1e-6) ? u_precomp_mie.z : (2.0 * g);
    float phase_M = c1 * (1.0 + cos_theta_sun * cos_theta_sun) / pow(max(1e-4, c2 - c3 * cos_theta_sun), 1.5);

    vec3 star_combined_intensity = (length(u_star_color_irrad[0].rgb) > 1e-6) ? u_star_color_irrad[0].rgb : vec3(1.0);
    vec3 scattered = star_combined_intensity * (
        phase_R * beta_R * total_rayleigh +
        phase_M * beta_M * total_mie +
        total_ms
    );

    float mean_trans = dot(current_transmittance, vec3(0.333333));
    out_color = vec4(scattered, clamp(mean_trans, 0.0, 1.0));
    out_transmittance = vec4(clamp(current_transmittance, vec3(0.0), vec3(1.0)), 1.0);
}
