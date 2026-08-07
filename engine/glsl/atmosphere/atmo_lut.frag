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
uniform float u_ozone_peak_km = 25.0;
uniform float u_ozone_width_km = 8.0;

// dir is always unit length here, so 'a = dot(dir,dir)' is always 1.0 -> removed
vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
<<<<<<< HEAD
    // dir is assumed to be normalized (a = 1.0)
    float b = dot(origin, dir);

    // Improved precision for large distances
=======
    float b = dot(origin, dir);
>>>>>>> another-optimization-test
    vec3 p = origin - b * dir;
    float p2 = dot(p, p);
    float r2 = radius * radius;

    if (p2 > r2) return vec2(1e10, -1e10);

    float d = sqrt(r2 - p2);
    float t_closest = -b;

    return vec2(t_closest - d, t_closest + d);
}

void main() {
    float x = f_uv.x * 2.0 - 1.0;
    float cos_theta = sign(x) * x * x;
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float h = f_uv.y * f_uv.y * max(1e-4, u_atmo_radius_km - u_planet_radius_km);
    float r = u_planet_radius_km + h;

    vec3 origin = vec3(0.0, r, 0.0);
    vec3 dir = vec3(sin_theta, cos_theta, 0.0);

    vec2 t_atmo = raySphereIntersect(origin, dir, u_atmo_radius_km);
    vec2 t_planet = raySphereIntersect(origin, dir, u_planet_radius_km);

    if (t_planet.x > 0.0 && t_planet.x < t_atmo.y) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

<<<<<<< HEAD
    int num_samples = 40;
=======
    float ray_len = t_atmo.y;

    // Hoisted out of the loop: these only depend on uniforms, not on sample index
    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;
    // Note: w0_M, beta_M (Mie scattering), beta_M_abs were computed but never
    // used in the final transmittance expression, so they've been removed.

    const int num_samples = 256;
>>>>>>> another-optimization-test
    float step_size = ray_len / float(num_samples);
    float inv_ozone_width = 1.0 / max(u_ozone_width_km, 1e-3);

    float od_rayleigh = 0.0;
    float od_mie = 0.0;
    float od_ozone = 0.0;

    float inv_h_rayleigh = 1.0 / u_h_rayleigh;
    float inv_h_mie = 1.0 / u_h_mie;
    float inv_ozone_width = 1.0 / max(u_ozone_width_km, 1e-3);

    for (int i = 0; i < num_samples; i++) {
        float t = (float(i) + 0.5) * step_size;
        vec3 p = origin + t * dir;
        float h_sample = max(0.0, length(p) - u_planet_radius_km);

<<<<<<< HEAD
        od_rayleigh += exp(-h_sample * inv_h_rayleigh) * step_size;
        od_mie += exp(-h_sample * inv_h_mie) * step_size;
        od_ozone += exp(-pow((h_sample - u_ozone_peak_km) * inv_ozone_width, 2.0)) * step_size;
=======
        float t_ozone = (h_sample - u_ozone_peak_km) * inv_ozone_width;

        od_rayleigh += exp(-h_sample / u_h_rayleigh) * step_size;
        od_mie += exp(-h_sample / u_h_mie) * step_size;
        od_ozone += exp(-(t_ozone * t_ozone)) * step_size;
>>>>>>> another-optimization-test
    }

    // Moved out of loop: computed once instead of up to 256 times
    vec3 transmittance = exp(-(beta_R * od_rayleigh + beta_M_ext * od_mie +
                                beta_A_mixed * od_rayleigh + beta_A_layered * od_ozone));
    out_color = vec4(transmittance, 1.0);
}
