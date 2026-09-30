uniform float u_planet_radius_km;
uniform float u_atmo_radius_km;
uniform float u_h_rayleigh;
uniform float u_h_mie;
uniform vec3 u_beta_rayleigh;
uniform vec3 u_beta_mie;
uniform vec3 u_beta_abs_mixed;
uniform vec3 u_beta_abs_layered;
uniform vec3 u_mie_albedo;
uniform float u_ozone_peak_km;
uniform float u_ozone_width_km;
uniform float u_scattering_sun_radius;
uniform int u_scattering_steps;

#include "scattering_coordinates.glsl"

void scattering_medium(float r, out vec3 rayleigh, out vec3 mie, out vec3 extinction) {
    float h = max(0.0, r - u_planet_radius_km);
    float rho_R = exp(-h / max(u_h_rayleigh, 1e-4));
    float rho_M = exp(-h / max(u_h_mie, 1e-4));
    float z = (h - u_ozone_peak_km) / max(u_ozone_width_km, 1e-3);
    rayleigh = u_beta_rayleigh * (1000.0 * rho_R);
    mie = u_beta_mie * clamp(u_mie_albedo, 0.0, 1.0) * (1000.0 * rho_M);
    extinction = (u_beta_rayleigh + u_beta_abs_mixed) * (1000.0 * rho_R)
               + u_beta_mie * (1000.0 * rho_M)
               + u_beta_abs_layered * (1000.0 * exp(-z * z));
}

// Concentrate quadrature at minimum altitude, including both legs of limb rays.
float scattering_station(float t, float distance, float closest) {
    float split = clamp(closest / max(distance, 1e-6), 0.0, 1.0);
    if (t < split) {
        float q = t / max(split, 1e-6);
        return closest * (1.0 - (1.0 - q) * (1.0 - q));
    }
    float q = (t - split) / max(1.0 - split, 1e-6);
    return closest + (distance - closest) * q * q;
}

vec2 scattering_tau_uv(float r, float mu, bool ground, vec2 size) {
    float q = scattering_view_coord(r, mu, ground);
    float half_width = size.x * 0.5;
    float x = (ground ? 0.0 : half_width) + 0.5 + q * (half_width - 1.0);
    float h = sqrt(clamp((r - u_scattering_bottom_km)
        / (u_atmo_radius_km - u_scattering_bottom_km), 0.0, 1.0));
    return vec2(x / size.x, (0.5 + h * (size.y - 1.0)) / size.y);
}
