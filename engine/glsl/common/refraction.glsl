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

// exp(x^2) * erfc(x) for x >= 0. Abramowitz & Stegun 7.1.26, whose polynomial
// is exactly exp(x^2) * (1 - erf(x)), making this form overflow-safe for ALL x
// (the naive exp(+x^2) * erfc(x) overflows/underflows f32 above x ~ 10, which
// happens at high elevations where sqrt(|C|/2H) * sin(e) gets large).
float refract_erfcx(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * x);
    return max(0.0, t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429)))));
}

// Apparent angular displacement (parallax-weighted) of an object at distance d.
// This is the quantity needed to displace a POINT object's apparent position.
float compute_refraction_angle(vec3 C, vec3 V, float d) {
    // Observer-inside regime: the ray LINE's periapsis lies BEHIND the camera
    // (object above the camera's local horizontal plane). The light's closest
    // approach to the planet is then the camera itself, so the bend must be
    // integrated along the ASCENDING path away from the camera — not around
    // the behind-camera periapsis. With the parabolic altitude approximation
    // alt(s) ~ h + s*sin(e) + s^2/(2|C|), the accumulated deflection has a
    // closed form:
    //   R(e) = 0.5 * delta_cam * erfcx(sqrt(|C| / (2H)) * sin(e))
    //   delta_cam = u_refract_max_bend * exp(-h / H)
    // This reproduces standard atmospheric refraction tables (34' at the
    // horizon, 24' @ 1 deg, 18' @ 2 deg, 9.9' @ 5 deg, 5.3' @ 10 deg) and is
    // VALUE-continuous with the limb model at e = 0 (both equal 0.5*delta_cam).
    // Evaluating this regime through the shared E_0 saturation tail instead
    // collapses the deflection to zero by ~7 deg elevation — stars visibly
    // "stop refracting" well before they set behind the horizon.
    float s_min_pre = -dot(C, V);
    if (s_min_pre < 0.0) {
        float Rc = length(C);
        if (Rc < 1e-6) return 0.0;
        vec3 up = C / Rc;
        float sin_e = clamp(dot(V, up), 0.0, 1.0);
        float h = Rc - refract_solid_radius(up);
        if (h > u_refract_scale_height * 15.0) return 0.0;
        float delta_cam = u_refract_max_bend * exp(-max(h, 0.0) / max(1e-4, u_refract_scale_height));
        if (delta_cam <= 1e-9) return 0.0;
        float k = Rc / (2.0 * max(1e-4, u_refract_scale_height));
        float bend = 0.5 * delta_cam * refract_erfcx(sqrt(k) * sin_e);
        // Object-side cutoff: only the bend accumulated between the camera
        // and the object at distance d. E_d ~ 1 for anything beyond a few
        // sigma_cam (every rendered body and all catalog stars).
        float sigma_cam = sqrt(max(1e-4, Rc * u_refract_scale_height));
        float x_d = d / (1.4142135 * sigma_cam);
        float E_d = sign(x_d) * sqrt(max(0.0, 1.0 - exp(-1.239 * x_d * x_d)));
        return max(0.0, bend * E_d);
    }

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
    // NOTE: deliberately NO "s_min <= 0" early-return here. A negative s_min
    // means the true ray LINE's closest approach lies BEHIND the camera — i.e.
    // the object sits above the camera's local horizontal plane (every sky
    // object seen from a landed camera). The 0.5 * (E_d + E_0) factor in the
    // deflection integral already accounts for the camera sitting past the
    // bend's periapsis (E_0 goes negative), so gating here would zero the
    // refraction for all above-horizon objects and let the full ~30 arcmin of
    // horizon bend snap in discontinuously the instant an object sinks below
    // the horizontal plane — the starfield "downward flick" bug. Only exclude
    // rays that end before reaching the periapsis (object in front of the
    // atmosphere; its light never traverses the dense shell).
    if (s_min >= d) return V;

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

    // Solve theta = alpha(V_app(theta)) for the apparent deflection.
    // alpha(V) is monotonically non-increasing as V_app lifts away from the
    // planet (the lifted ray's periapsis rises -> delta_rmin falls), so
    // F(theta) = theta - alpha(V_app(theta)) is monotone increasing with
    // F(0) = -alpha_0 < 0 and F(alpha_0) = alpha_0 - alpha(alpha_0) >= 0:
    // the fixed point is ALWAYS bracketed by [0, alpha_0].
    // The previous 3-step damped Newton-Raphson (F' ~= 1 + (s_min/H)*alpha)
    // collapsed for distant observers: with the camera thousands of km above
    // the bend region, s_min/H reaches the hundreds/thousands, each step
    // closed only ~1/F' of the remaining gap, and 3 iterations left the solve
    // at a small fraction of the true deflection — stars visibly stopped
    // refracting and sank into the limb instead of stacking up compressed
    // against the refracted horizon like the anchored mesh path. Bisection on
    // the guaranteed bracket converges unconditionally at any observer
    // distance; 12 halvings resolve alpha_0/4096 (< 0.02' for max_bend).
    float lo_t = 0.0;
    float hi_t = alpha_0;
    for (int i = 0; i < 12; i++) {
        float mid_t = 0.5 * (lo_t + hi_t);
        vec3 V_test = V * cos(mid_t) + u_dir * sin(mid_t);
        float alpha = compute_refraction_angle(C, V_test, d);
        if (alpha > mid_t) lo_t = mid_t; else hi_t = mid_t;
    }
    float theta = 0.5 * (lo_t + hi_t);

    vec3 V_app = V * cos(theta) + u_dir * sin(theta);
    float s_app = -dot(C, V_app);
    // Occlusion must test the FORWARD ray's periapsis, not the line's. When
    // the apparent direction rises above the local horizontal (s_app < 0), the
    // line periapsis lies behind the camera and is strictly closer to the
    // planet center than the camera itself — using it would spuriously flag
    // refracted objects hanging above the horizon as occluded for any camera
    // within a few hundred meters of the surface. Clamping to the forward
    // extent makes the closest approach the camera position, which is only
    // "blocked" when the camera itself is below the solid surface.
    float s_app_fwd = max(s_app, 0.0);
    vec3 P_app = C + s_app_fwd * V_app;
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
