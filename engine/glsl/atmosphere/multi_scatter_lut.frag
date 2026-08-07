#version 460 core
in vec2 f_uv;
out vec4 out_color;

uniform float u_planet_radius_km;
uniform float u_atmo_radius_km;
uniform float u_h_rayleigh;
uniform float u_h_mie;
uniform vec3 u_beta_rayleigh;
uniform vec3 u_beta_mie;
uniform vec3 u_beta_abs_mixed;
uniform vec3 u_beta_abs_layered;
uniform vec3 u_mie_albedo;
uniform float u_mie_g;
uniform vec3 u_ground_albedo;
uniform float u_ozone_peak_km = 25.0;
uniform float u_ozone_width_km = 8.0;

uniform sampler2D u_transmittance_lut;

// Precomputed direction set: a 6x6 stratified sphere sampling (36 directions),
// baked at compile time so acos/sin/cos are never evaluated per-pixel.
// Generated the same way as the original: theta = acos(1 - 2u), phi = 2*pi*v,
// with u,v the cell-center coordinates of a 6x6 grid.
const int NUM_DIRS = 36;
const vec3 SAMPLE_DIRS[NUM_DIRS] = vec3[NUM_DIRS](
    vec3(0.47871355, 0.83333333, 0.27638540),
    vec3(0.00000000, 0.83333333, 0.55277080),
    vec3(-0.47871355, 0.83333333, 0.27638540),
    vec3(-0.47871355, 0.83333333, -0.27638540),
    vec3(-0.00000000, 0.83333333, -0.55277080),
    vec3(0.47871355, 0.83333333, -0.27638540),
    vec3(0.75000000, 0.50000000, 0.43301270),
    vec3(0.00000000, 0.50000000, 0.86602540),
    vec3(-0.75000000, 0.50000000, 0.43301270),
    vec3(-0.75000000, 0.50000000, -0.43301270),
    vec3(-0.00000000, 0.50000000, -0.86602540),
    vec3(0.75000000, 0.50000000, -0.43301270),
    vec3(0.85391256, 0.16666667, 0.49309414),
    vec3(0.00000000, 0.16666667, 0.98601360),
    vec3(-0.85391256, 0.16666667, 0.49309414),
    vec3(-0.85391256, 0.16666667, -0.49309414),
    vec3(-0.00000000, 0.16666667, -0.98601360),
    vec3(0.85391256, 0.16666667, -0.49309414),
    vec3(0.85391256, -0.16666667, 0.49309414),
    vec3(0.00000000, -0.16666667, 0.98601360),
    vec3(-0.85391256, -0.16666667, 0.49309414),
    vec3(-0.85391256, -0.16666667, -0.49309414),
    vec3(-0.00000000, -0.16666667, -0.98601360),
    vec3(0.85391256, -0.16666667, -0.49309414),
    vec3(0.75000000, -0.50000000, 0.43301270),
    vec3(0.00000000, -0.50000000, 0.86602540),
    vec3(-0.75000000, -0.50000000, 0.43301270),
    vec3(-0.75000000, -0.50000000, -0.43301270),
    vec3(-0.00000000, -0.50000000, -0.86602540),
    vec3(0.75000000, -0.50000000, -0.43301270),
    vec3(0.47871355, -0.83333333, 0.27638540),
    vec3(0.00000000, -0.83333333, 0.55277080),
    vec3(-0.47871355, -0.83333333, 0.27638540),
    vec3(-0.47871355, -0.83333333, -0.27638540),
    vec3(-0.00000000, -0.83333333, -0.55277080),
    vec3(0.47871355, -0.83333333, -0.27638540)
);

// dir is always unit length here, so 'a = dot(dir,dir)' is always 1.0 -> removed
vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float b = dot(origin, dir);
    vec3 p = origin - b * dir;
    float p2 = dot(p, p);
    float r2 = radius * radius;

    if (p2 > r2) return vec2(1e10, -1e10);

    float d = sqrt(r2 - p2);
    float t_closest = -b;

    return vec2(t_closest - d, t_closest + d);
}

vec3 get_transmittance(float r, float cos_theta) {
    float h_norm = clamp((r - u_planet_radius_km) / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float v = sqrt(h_norm);
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

void main() {
    float x = f_uv.x * 2.0 - 1.0;
    float cos_sun_zenith = sign(x) * x * x;
    float sin_sun_zenith = sqrt(max(0.0, 1.0 - cos_sun_zenith * cos_sun_zenith));
    float h = f_uv.y * f_uv.y * max(1e-4, u_atmo_radius_km - u_planet_radius_km);
    float r = u_planet_radius_km + h;

    vec3 origin = vec3(0.0, r, 0.0);
    vec3 sun_dir = vec3(sin_sun_zenith, cos_sun_zenith, 0.0);

    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M);
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;
    float inv_ozone_width = 1.0 / max(u_ozone_width_km, 1e-3);
    float phase = 1.0 / (4.0 * 3.14159265358979);

    // Direction-invariant: computed once per pixel instead of inside the loop.
    vec3 effective_ground_albedo = max(u_ground_albedo, w0_M * clamp((beta_M * u_h_mie - vec3(0.5)) / 2.0, vec3(0.0), vec3(1.0)));

    const int ray_samples = 20;
    vec3 lum_total = vec3(0.0);
    vec3 fms_total = vec3(0.0);

    // Single flat loop over precomputed directions: 36 iterations instead of
    // 8x8 = 64, and no acos/sin/cos evaluated at runtime.
    for (int d = 0; d < NUM_DIRS; d++) {
        vec3 ray_dir = SAMPLE_DIRS[d];

        vec2 t_atmo = raySphereIntersect(origin, ray_dir, u_atmo_radius_km);
        vec2 t_planet = raySphereIntersect(origin, ray_dir, u_planet_radius_km);

        bool hits_ground = (t_planet.x > 0.0 && t_planet.x < t_atmo.y);
        float ray_len = hits_ground ? t_planet.x : t_atmo.y;

        float step_size = ray_len / float(ray_samples);

        vec3 lum = vec3(0.0);
        vec3 fms = vec3(0.0);

        float current_s = 0.5 * step_size;
        vec3 transmittance_accum = vec3(1.0);

        for (int s = 0; s < ray_samples; s++) {
            vec3 p = origin + ray_dir * current_s;
            float p_len = length(p);
            float h_sample = max(0.0, p_len - u_planet_radius_km);

            float rho_R = exp(-h_sample / u_h_rayleigh);
            float rho_M = exp(-h_sample / u_h_mie);
            float t_ozone = (h_sample - u_ozone_peak_km) * inv_ozone_width;
            float rho_O = exp(-(t_ozone * t_ozone));

            vec3 scattering = beta_R * rho_R + beta_M * rho_M;
            vec3 extinction = scattering + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;

            vec3 sample_transmittance = exp(-extinction * step_size);

            float p_cos_sun = dot(p, sun_dir) / p_len;
            vec3 trans_to_sun = get_transmittance(p_len, p_cos_sun);

            vec3 S = scattering * trans_to_sun * phase;

            vec3 Sint = (S - S * sample_transmittance) / max(extinction, 1e-6);
            lum += transmittance_accum * Sint;

            vec3 FMS_Sint = (scattering - scattering * sample_transmittance) / max(extinction, 1e-6);
            fms += transmittance_accum * FMS_Sint;

            transmittance_accum *= sample_transmittance;
            current_s += step_size;
        }

        if (hits_ground) {
            vec3 p = origin + ray_dir * t_planet.x;
            float p_len = length(p);
            float p_cos_sun = dot(p, sun_dir) / max(p_len, 1e-6);
            vec3 trans_to_sun = get_transmittance(p_len, p_cos_sun);

            float NdotL = max(0.0, p_cos_sun);
            vec3 ground_lum = (effective_ground_albedo / 3.14159265358979) * NdotL * trans_to_sun;

            lum += transmittance_accum * ground_lum;
            fms += transmittance_accum * effective_ground_albedo;
        }

        lum_total += lum;
        fms_total += fms;
    }

    float n_samples = float(NUM_DIRS);
    vec3 L2nd = lum_total / n_samples;
    vec3 fms_avg = fms_total / n_samples;

    vec3 psi = L2nd / max(vec3(1.0) - fms_avg, 1e-6);
    out_color = vec4(psi, 1.0);
}
