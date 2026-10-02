#version 460 core
#define PI 3.14159265358979323846

in vec2 f_uv;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_transmittance;
// Per-star in-scatter slices (locations 2-5). atmo.frag's ring-shadow slicing
// normalizes each star's shadow deficit against that star's OWN baked light,
// so one star's ring shadow can never darken another star's illumination.
layout(location = 2) out vec4 out_star0;
layout(location = 3) out vec4 out_star1;
layout(location = 4) out vec4 out_star2;
layout(location = 5) out vec4 out_star3;

#include "common/scattering_scene.glsl"
// Camera-specific projection of cached fields; no marching or polar winter.
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
            out_star0 = vec4(0.0); out_star1 = vec4(0.0); out_star2 = vec4(0.0); out_star3 = vec4(0.0);
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
        out_star0 = vec4(0.0); out_star1 = vec4(0.0); out_star2 = vec4(0.0); out_star3 = vec4(0.0);
        return;
    }

    vec3 a = ray_origin + s_start * V, b = ray_origin + s_end * V;
    vec3 T = endpoint_transmittance(a, b), star_scatter[4], total = vec3(0.0);
    for (int st = 0; st < 4; ++st) {
        star_scatter[st] = vec3(0.0);
        if (st < u_num_stars && dot(u_star_dir_sph_eff[st].xyz, u_star_dir_sph_eff[st].xyz) > 1e-8) {
            star_scatter[st] = endpoint_radiance_star(a, b, normalize(u_star_dir_sph_eff[st].xyz), T,
                st, u_star_pos_local[st].w) * u_star_color_irrad[st].rgb;
            total += star_scatter[st];
        }
    }
    total += endpoint_secondary_light(a, b, T, instances[u_body_idx*7+4].xyz,
        instances[u_body_idx*7+5].xyz, floatBitsToUint(instances[u_body_idx*7+3].w));
    out_color = vec4(total, dot(T, vec3(1.0/3.0)));
    out_transmittance = vec4(T, 1.0);
    out_star0 = vec4(star_scatter[0], 1.0); out_star1 = vec4(star_scatter[1], 1.0);
    out_star2 = vec4(star_scatter[2], 1.0); out_star3 = vec4(star_scatter[3], 1.0);
}
