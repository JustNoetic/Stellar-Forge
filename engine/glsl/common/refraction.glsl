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

// Gravitational Lensing Uniforms
uniform vec3 u_grav_lens_center;      // lens world position (camera-relative frame), AU
uniform float u_grav_lens_rs;         // Schwarzschild radius 2GM/c^2 in km
uniform float u_grav_lens_radius;     // physical (event-horizon / surface) radius in km
uniform int u_grav_lens_type;         // 0=Star, 1=WhiteDwarf, 2=NeutronStar, 3=BlackHole
uniform bool u_grav_lens_enabled;     // master enable flag
uniform float u_grav_lens_strength;   // deflection multiplier (default 1.0)
uniform float u_grav_lens_spin;       // Kerr dimensionless spin a_* in (-1, 1)
uniform vec3 u_grav_lens_pole;        // spin axis unit vector (render frame)

// Gravitational deflection of a sightline past the lens (Einstein weak-field
// + 2PN + strong-field divergence near the photon sphere). C_km is the camera
// position in lens-centric km, V the (unit) ray direction, d_km the distance
// along the ray to the object. Sets is_shadow = true when the ray's impact
// parameter falls below the (spin-aware) critical capture radius b_c — the
// ray plunges through the event horizon and the object must be suppressed.
float compute_gravitational_deflection(vec3 C_km, vec3 V, float d_km, out bool is_shadow) {
    is_shadow = false;
    if (!u_grav_lens_enabled || u_grav_lens_rs <= 1e-6) return 0.0;

    float rs = u_grav_lens_rs;
    float r_cam_km = length(C_km);
    float metric_factor = sqrt(max(1e-4, 1.0 - rs / max(1e-4, r_cam_km)));
    float s_min = -dot(C_km, V);
    vec3 P_min = C_km + s_min * V;
    float b = length(P_min) / metric_factor; // impact parameter in km

    // Safe direction-aware Kerr critical shadow radius b_c(phi, theta_o)
    float b_c_base = 2.5980762 * rs; // 3*sqrt(3)/2 * rs
    float b_c = b_c_base;

    if (abs(u_grav_lens_spin) > 1e-4) {
        vec3 pole_n = length(u_grav_lens_pole) > 1e-4 ? normalize(u_grav_lens_pole) : vec3(0.0, 1.0, 0.0);
        vec3 cam_dir = C_km / max(1e-4, r_cam_km);
        vec3 cross_p = cross(pole_n, cam_dir);
        float len_cp = length(cross_p);
        if (len_cp > 1e-4) {
            vec3 prograde_dir = cross_p / len_cp;
            float len_pmin = length(P_min);
            vec3 ray_perp = len_pmin > 1e-4 ? (P_min / len_pmin) : prograde_dir;
            float cos_phi = clamp(dot(ray_perp, prograde_dir), -1.0, 1.0);
            float spin_factor = u_grav_lens_spin * clamp(len_cp, 0.0, 1.0);
            b_c = b_c_base * (1.0 - 0.35 * spin_factor * cos_phi);
        }
    }

    // Ray capture for black hole shadow
    if (u_grav_lens_type == 3 && b <= b_c && s_min > 0.0 && (d_km <= 0.0 || s_min < d_km)) {
        is_shadow = true;
        return 0.0;
    }

    if (b < 1e-4) return 0.0;

    // Finite-distance geometric factor along ray segment [0, d_km]
    float geom_factor = 1.0;
    if (d_km > 0.0) {
        float d_s = d_km - s_min;
        float denom_s = sqrt(b * b + d_s * d_s);
        float denom_0 = sqrt(b * b + s_min * s_min);
        geom_factor = 0.5 * ( (d_s / max(1e-6, denom_s)) + (s_min / max(1e-6, denom_0)) );
        geom_factor = clamp(geom_factor, 0.0, 1.0);
    }

    // Weak field Einstein deflection: 2 * rs / b
    float alpha_weak = 2.0 * rs / b;

    // Higher-order 2PN and Darwin/Bozza strong-field asymptotic correction
    float b_ratio = rs / b;
    float alpha_gr = alpha_weak + 2.945243 * b_ratio * b_ratio; // (15*pi/16) * (rs/b)^2

    if (u_grav_lens_type == 3) { // Black hole strong field divergence
        float b_over_bc = b / max(1e-5, b_c);
        if (b_over_bc > 1.0 && b_over_bc < 4.0) {
            float delta = b_over_bc - 1.0;
            float strong_div = max(0.0, -log(max(1e-6, delta)) * pow(1.0 / b_over_bc, 3.0));
            alpha_gr += strong_div;
        }
    }

    float strength = u_grav_lens_strength > 0.0 ? u_grav_lens_strength : 1.0;
    float final_alpha = alpha_gr * geom_factor * strength;
    return max(0.0, final_alpha);
}

// Deflection of background sources (stars, orbits, point lights) past the lens.
// Solves the gravitational lens equation for the apparent primary image direction:
//   theta = 0.5 * (beta + sqrt(beta^2 + 4 * theta_E^2))
// Pushes the apparent position radially OUTWARD from the lens center in the sky,
// smoothly forming an Einstein ring at beta = 0 without divergent singularities,
// and matching Einstein weak-field deflection alpha = 2*rs/b at large beta.
// Sets is_shadow = true if the ray passes inside a solid body lens surface.
vec3 apply_gravitational_deflection(vec3 C_km, vec3 V, float d_km, out bool is_shadow) {
    is_shadow = false;
    if (!u_grav_lens_enabled || u_grav_lens_rs <= 1e-6) return V;

    float d_l_km = length(C_km);
    if (d_l_km < 1.0) return V;

    // Optical axis: unit vector from camera towards the lens center
    vec3 L = -C_km / d_l_km;

    // Angle beta between optical axis (lens) and undeflected source direction V.
    // Using cross product avoids catastrophic cancellation of 1.0 - cos(beta) in float32.
    vec3 cross_LV = cross(L, V);
    float sin_beta = length(cross_LV);
    float cos_beta = dot(L, V);
    if (cos_beta <= 0.0) return V; // Source is behind the camera / past 90 degrees
    float beta = atan(sin_beta, cos_beta);

    // If source is strictly in front of the lens plane along the optical axis,
    // its light never traverses past the lens — no deflection.
    if (d_km > 0.0 && d_km * cos_beta <= d_l_km) return V;

    // Finite-distance factor d_ls / d_s (approaches 1.0 for distant catalog stars)
    float geom_factor = 1.0;
    if (d_km > 0.0) {
        float d_ls_km = d_km * cos_beta - d_l_km;
        geom_factor = clamp(d_ls_km / d_km, 0.0, 1.0);
    }

    float strength = u_grav_lens_strength > 0.0 ? u_grav_lens_strength : 1.0;
    float theta_E2 = (2.0 * u_grav_lens_rs * geom_factor * strength) / d_l_km;
    if (theta_E2 <= 1e-18) return V;
    float theta_E = sqrt(theta_E2);

    // Unit vector in the observer's sky plane pointing radially away from lens center towards source
    vec3 u_dir;
    if (sin_beta > 1e-7) {
        u_dir = cross(cross_LV, L) / sin_beta;
    } else {
        vec3 ref = abs(L.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        u_dir = normalize(cross(L, ref));
    }

    // Exact primary image root of the gravitational lens equation:
    // theta^2 - beta*theta - theta_E^2 = 0  ==>  theta = 0.5 * (beta + sqrt(beta^2 + 4*theta_E^2))
    float theta = 0.5 * (beta + sqrt(beta * beta + 4.0 * theta_E2));

    // Solid body occlusion for non-black-hole lenses (e.g. star, white dwarf, neutron star)
    if (u_grav_lens_type != 3 && u_grav_lens_radius > 0.0) {
        float theta_body = u_grav_lens_radius / d_l_km;
        if (theta < theta_body) {
            is_shadow = true;
            return V;
        }
    }

    // Deflected apparent sightline (pushed radially outward along u_dir)
    vec3 V_app = normalize(L * cos(theta) + u_dir * sin(theta));

    // Safe Frame-dragging Lense-Thirring lateral deflection for spinning Kerr lenses
    if (abs(u_grav_lens_spin) > 1e-4) {
        vec3 pole_n = length(u_grav_lens_pole) > 1e-4 ? normalize(u_grav_lens_pole) : vec3(0.0, 1.0, 0.0);
        vec3 drag_dir = cross(pole_n, V_app);
        float len_drag = length(drag_dir);
        if (len_drag > 1e-4) {
            drag_dir /= len_drag;
            float b_km = d_l_km * sin(theta);
            float drag_angle = clamp((u_grav_lens_rs * u_grav_lens_rs * u_grav_lens_spin) / max(1.0, b_km * b_km), -0.5, 0.5);
            V_app = normalize(V_app * cos(drag_angle) + drag_dir * sin(drag_angle));
        }
    }

    return V_app;
}

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
    vec3 result = world_pos;

    // 1. Atmospheric refraction (only when an atmosphere refracts nearby).
    if (u_refract_max_bend > 1e-6 && length(world_pos - u_refract_center) > 1e-7) {
        vec3 C_km = (cam_pos - u_refract_center) * u_au_to_km;
        vec3 true_vec_au = world_pos - cam_pos;
        float d_au = length(true_vec_au);
        if (d_au > 1e-7) {
            vec3 V = true_vec_au / d_au;
            float d_km = d_au * u_au_to_km;
            bool is_occluded = false;
            vec3 V_app = solve_refraction_apparent(C_km, V, d_km, is_occluded);
            if (!is_occluded) result = cam_pos + V_app * d_au;
        }
    }

    // 2. Gravitational lensing (independent of atmospheric state).
    if (u_grav_lens_enabled && u_grav_lens_rs > 1e-6
            && length(world_pos - u_grav_lens_center) > 1e-7) {
        vec3 C_km = (cam_pos - u_grav_lens_center) * u_au_to_km;
        vec3 true_vec_au = result - cam_pos;
        float d_au = length(true_vec_au);
        if (d_au > 1e-7) {
            vec3 V = true_vec_au / d_au;
            float d_km = d_au * u_au_to_km;
            bool is_shadow = false;
            vec3 V_app = apply_gravitational_deflection(C_km, V, d_km, is_shadow);
            if (!is_shadow) {
                result = cam_pos + V_app * d_au;
            }
        }
    }
    return result;
}

vec3 apply_refraction_eye(vec3 eye_pos, vec3 cam_pos) {
    vec3 result = eye_pos;

    // 1. Atmospheric refraction.
    if (u_refract_max_bend > 1e-6) {
        float d_au = length(eye_pos);
        if (d_au > 1e-7) {
            vec3 C_km = (cam_pos - u_refract_center) * u_au_to_km;
            vec3 V = eye_pos / d_au;
            float d_km = d_au * u_au_to_km;
            bool is_occluded = false;
            vec3 V_app = solve_refraction_apparent(C_km, V, d_km, is_occluded);
            if (!is_occluded) result = V_app * d_au;
        }
    }

    // 2. Gravitational lensing. eye_pos is camera-relative.
    if (u_grav_lens_enabled && u_grav_lens_rs > 1e-6) {
        float d_au = length(result);
        if (d_au > 1e-7) {
            vec3 C_km = (cam_pos - u_grav_lens_center) * u_au_to_km;
            vec3 V = result / d_au;
            float d_km = d_au * u_au_to_km;
            bool is_shadow = false;
            vec3 V_app = apply_gravitational_deflection(C_km, V, d_km, is_shadow);
            if (!is_shadow) {
                result = V_app * d_au;
            }
        }
    }
    return result;
}

#endif // REFRACTION_GLSL
