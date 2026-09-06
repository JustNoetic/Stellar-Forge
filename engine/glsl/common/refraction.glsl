#ifndef REFRACTION_GLSL
#define REFRACTION_GLSL

uniform vec3 u_refract_center;
uniform float u_refract_radius;
uniform float u_refract_max_bend;
uniform float u_refract_scale_height;
uniform vec3 u_refract_pole;
uniform float u_refract_oblateness;
#ifndef ATMO_DATA_HAS_AU_TO_KM
uniform float u_au_to_km;
#endif

// Oblate-aware solid body radius along a given direction from the refract center.
float refract_solid_radius(vec3 P_dir) {
    if (u_refract_oblateness > 0.001 && u_refract_oblateness < 0.99) {
        vec3 pole_dir = length(u_refract_pole) > 1e-4 ? normalize(u_refract_pole) : vec3(0.0, 1.0, 0.0);
        float cos_t = abs(dot(P_dir, pole_dir));
        float k = 1.0 / (1.0 - u_refract_oblateness);
        float denom = sqrt(max(1e-6, 1.0 + (k * k - 1.0) * cos_t * cos_t));
        return u_refract_radius / denom;
    }
    return u_refract_radius;
}

// Shared refraction state setup.
// Returns r_min (closest approach of the ray line to the refract center), or -1.0
// when refraction is inactive / the ray stays above the atmosphere cutoff.
// Outputs: s_min (distance along ray to closest approach), delta_rmin (peak bend
// at closest approach), sigma (Gaussian width of the bend region).
float _refraction_setup(vec3 C, vec3 V, float d, out float s_min, out float delta_rmin, out float sigma) {
    s_min = 0.0;
    delta_rmin = 0.0;
    sigma = 0.0;
    if (u_refract_max_bend <= 1e-6) return -1.0;

    s_min = -dot(C, V);
    vec3 P_min = C + s_min * V;
    float r_min = length(P_min);
    float local_refract_radius = refract_solid_radius(r_min > 1e-6 ? (P_min / r_min) : vec3(0.0, 1.0, 0.0));

    if (r_min > local_refract_radius + u_refract_scale_height * 15.0) return -1.0;

    float r_min_clamped = max(r_min, local_refract_radius);
    delta_rmin = min(0.15, u_refract_max_bend * exp(-(r_min_clamped - local_refract_radius) / max(1e-4, u_refract_scale_height)));
    sigma = sqrt(max(1e-4, r_min_clamped * u_refract_scale_height));
    return r_min;
}

// Apparent angular displacement (parallax-weighted) of an object at distance d.
// This is the quantity needed to displace a POINT object's apparent position.
float compute_refraction_angle(vec3 C, vec3 V, float d) {
    float s_min, delta_rmin, sigma;
    float r_min = _refraction_setup(C, V, d, s_min, delta_rmin, sigma);
    if (r_min < 0.0) return 0.0;

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

// Total accumulated ray turn (NO parallax weighting) between the camera and an
// object at distance d. Use this for anchored ray-bending (mesh / atmosphere /
// ring ray-tracing), where the bend is applied at the closest-approach anchor.
// The anchored rotation by this angle reproduces exactly the apparent
// displacement of compute_refraction_angle; feeding the parallax-weighted alpha
// into an anchored bend would apply the (1 - s_min/d) factor twice.
float compute_refraction_total(vec3 C, vec3 V, float d) {
    float s_min, delta_rmin, sigma;
    float r_min = _refraction_setup(C, V, d, s_min, delta_rmin, sigma);
    if (r_min < 0.0) return 0.0;

    if (d < 0.1 * sigma) {
        float kappa_0 = (delta_rmin / (sigma * 2.506628)) * exp(-(s_min * s_min) / (2.0 * sigma * sigma));
        return kappa_0 * d;
    }

    float sqrt2_sig = 1.4142135 * sigma;
    float x_d = (d - s_min) / sqrt2_sig;
    float x_0 = s_min / sqrt2_sig;

    float E_d = sign(x_d) * sqrt(max(0.0, 1.0 - exp(-1.239 * x_d * x_d)));
    float E_0 = sign(x_0) * sqrt(max(0.0, 1.0 - exp(-1.239 * x_0 * x_0)));

    return max(0.0, delta_rmin * 0.5 * (E_d + E_0));
}

// Inverts the refraction deflection equation theta = alpha(V_app(theta)) for point
// lights, orbits, and celestial markers.
// Given unrefracted ray V, camera C, distance d:
// Solves for apparent vector V_app deflected upward along u_dir via Newton-Raphson.
// If the apparent ray line passes below the solid surface, sets is_occluded = true.
vec3 solve_refraction_apparent(vec3 C, vec3 V, float d, out bool is_occluded) {
    is_occluded = false;
    float s_min = -dot(C, V);
    if (s_min <= 0.0 || s_min >= d) return V;

    vec3 u_dir = C - V * dot(C, V);
    float u_len = length(u_dir);
    if (u_len <= 1e-5) {
        is_occluded = true;
        return V;
    }
    u_dir /= u_len;

    vec3 P_min = C + s_min * V;
    float r_min = length(P_min);
    float local_refract_radius = refract_solid_radius(r_min > 1e-6 ? (P_min / r_min) : vec3(0.0, 1.0, 0.0));

    if (r_min > local_refract_radius + u_refract_scale_height * 15.0) return V;

    float alpha_0 = compute_refraction_angle(C, V, d);
    if (alpha_0 <= 1e-7) return V;

    // Damped initial step: theta_0 = alpha_0 / (1 + (s_min / H) * alpha_0)
    float theta = alpha_0 / (1.0 + (max(s_min, 0.0) / max(1e-4, u_refract_scale_height)) * alpha_0);

    // 3 Newton-Raphson iterations to solve F(theta) = theta - alpha(V_app(theta)) = 0
    for (int i = 0; i < 3; i++) {
        vec3 V_test = V * cos(theta) + u_dir * sin(theta);
        float alpha = compute_refraction_angle(C, V_test, d);
        float s_test = -dot(C, V_test);
        float F = theta - alpha;
        float F_prime = 1.0 + (max(s_test, 0.0) / max(1e-4, u_refract_scale_height)) * alpha;
        theta = max(0.0, theta - F / F_prime);
    }

    vec3 V_app = V * cos(theta) + u_dir * sin(theta);
    float s_app = -dot(C, V_app);
    vec3 P_app = C + s_app * V_app;
    float r_app = length(P_app);
    float solid_r = refract_solid_radius(r_app > 1e-6 ? (P_app / r_app) : vec3(0.0, 1.0, 0.0));

    if (r_app < solid_r) {
        is_occluded = true;
    }

    return V_app;
}

// Backward-compatible occlusion helper
bool refract_chord_blocked(vec3 C, vec3 V, float d) {
    bool is_occ = false;
    solve_refraction_apparent(C, V, d, is_occ);
    return is_occ;
}

vec3 apply_refraction(vec3 world_pos, vec3 cam_pos) {
    if (u_refract_max_bend <= 1e-6) return world_pos;
    if (length(world_pos - u_refract_center) < 1e-7) return world_pos;
    vec3 C_km = (cam_pos - u_refract_center) * u_au_to_km;
    vec3 P_km = (world_pos - u_refract_center) * u_au_to_km;
    vec3 true_vec = P_km - C_km;
    float d_km = length(true_vec);
    if (d_km <= 1e-5) return world_pos;
    vec3 V = true_vec / d_km;

    bool is_occluded = false;
    vec3 V_app = solve_refraction_apparent(C_km, V, d_km, is_occluded);
    if (is_occluded) return world_pos;
    return cam_pos + V_app * (d_km / u_au_to_km);
}

vec3 apply_refraction_eye(vec3 eye_pos, vec3 cam_pos) {
    if (u_refract_max_bend <= 1e-6) return eye_pos;
    vec3 C_km = (cam_pos - u_refract_center) * u_au_to_km;
    vec3 true_vec = eye_pos * u_au_to_km;
    float d_km = length(true_vec);
    if (d_km <= 1e-5) return eye_pos;
    vec3 V = true_vec / d_km;

    bool is_occluded = false;
    vec3 V_app = solve_refraction_apparent(C_km, V, d_km, is_occluded);
    if (is_occluded) return eye_pos;
    return V_app * (d_km / u_au_to_km);
}

#endif // REFRACTION_GLSL
