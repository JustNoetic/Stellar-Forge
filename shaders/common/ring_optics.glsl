const float RING_MAX_ALPHA = 0.99999994;
const float RING_MS_MIN_TAU = 1e-3;

float ring_tau(float alpha) {
    float a = clamp(alpha, 0.0, RING_MAX_ALPHA);
    return -log(1.0 - a);
}

float ring_absorbed(float tau_val) {
    return 1.0 - exp(-max(tau_val, 0.0));
}

float ring_transfer(float tau_val, float mu_s, float mu_v, bool same_side) {
    float tau = max(tau_val, 0.0);
    float ms = max(mu_s, 1e-5);
    float mv = max(mu_v, 1e-5);
    float xs = tau / ms;
    float xv = tau / mv;

    if (same_side) {
        float a = min(xs + xv, 80.0);
        float one_minus_exp = (a < 1e-3)
            ? a * (1.0 - 0.5 * a + (1.0 / 6.0) * a * a)
            : (1.0 - exp(-a));
        return (ms / (ms + mv)) * one_minus_exp;
    }

    float d = xs - xv;
    if (abs(d) < 1e-3) {
        float xm = 0.5 * (xs + xv);
        return xv * exp(-xm) * (1.0 + (d * d) / 24.0);
    }
    float ad = abs(d);
    float one_minus_exp = ad < 1e-2
        ? ad * (1.0 - 0.5 * ad + ad * ad / 6.0)
        : 1.0 - exp(-ad);
    return (xv / ad) * exp(-min(xs, xv)) * one_minus_exp;
}

float ring_opposition_factor(float cos_alpha, float tau_val) {
    float ca = clamp(cos_alpha, -1.0, 1.0);
    float alpha_rad = acos(clamp(ca, -1.0, 1.0 - 1e-7));

    float shoe_width = 0.030;
    float shoe_amp = 0.55 * smoothstep(0.02, 0.45, tau_val);
    float shoe = shoe_amp / (1.0 + alpha_rad / shoe_width);

    float cboe_width = 0.0025;
    float cboe_amp = 0.35 * smoothstep(0.005, 0.18, tau_val);
    float x_cb = alpha_rad / cboe_width;
    float cboe = cboe_amp / (1.0 + x_cb * x_cb);

    return 1.0 + shoe + cboe;
}

float ring_phase_hg(float cos_theta, float g) {
    float gg = clamp(g, -0.99, 0.99);
    float mu = clamp(cos_theta, -1.0, 1.0);
    float one_m_g = (gg >= 0.0) ? (1.0 - gg) : (1.0 + gg);
    float delta_mu = (gg >= 0.0) ? (1.0 - mu) : (1.0 + mu);
    float denom = max(one_m_g * one_m_g + 2.0 * abs(gg) * delta_mu, 1e-6);
    float inv_s = inversesqrt(denom);
    return (1.0 - gg * gg) * inv_s * inv_s * inv_s;
}

float ring_phase_dhg(float cos_theta, float scatter, float g_fwd, float g_back, float tau_val, bool is_textured) {
    float dust_factor = 1.0 - smoothstep(0.05, 0.50, tau_val);
    float core_factor = smoothstep(0.30, 1.80, tau_val);

    float g1 = clamp(g_fwd + (is_textured ? 0.14 * dust_factor : 0.0), -0.92, 0.92);
    float g2_mag = clamp(abs(g_back) + (is_textured ? 0.08 * core_factor : 0.0), 0.0, 0.92);
    float g2 = -g2_mag;

    float base_w = (is_textured && scatter > 1.0) ? 0.35 : clamp(scatter, 0.0, 1.0);
    float w_f = clamp(base_w + (is_textured ? (0.12 * dust_factor - 0.08 * core_factor) : 0.0), 0.0, 1.0);

    return mix(ring_phase_hg(cos_theta, g2), ring_phase_hg(cos_theta, g1), w_f);
}

vec3 ring_chandrasekhar_h(float mu, vec3 w0) {
    float m = max(mu, 0.0);
    vec3 s = sqrt(max(vec3(1.0) - w0, vec3(1e-6)));
    return vec3(1.0 + m * 1.7320508) / (vec3(1.0) + m * 1.7320508 * s);
}

vec3 ring_slab_x(float mu, float tau_val, vec3 w0) {
    vec3 h = ring_chandrasekhar_h(mu, w0);
    float m = max(mu, 1e-4);
    vec3 k = sqrt(3.0 * max(vec3(1.0) - w0, vec3(1e-5))) + vec3(0.5);
    vec3 atten = vec3(1.0) - exp(-max(tau_val, 0.0) * (vec3(1.0 / m) + k));
    return vec3(1.0) + (h - vec3(1.0)) * atten;
}

vec3 ring_multiple_scattering(float tau_val, float mu_s, float mu_v, bool same_side, vec3 linear_rgb) {
    float tau = max(tau_val, 0.0);
    if (tau < RING_MS_MIN_TAU) return vec3(0.0);

    float luma = dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 w_transport = clamp(mix(vec3(luma), linear_rgb, 0.20) * 0.78, vec3(0.0), vec3(0.92));
    vec3 w_color = clamp(linear_rgb * 0.78, vec3(0.0), vec3(0.96));
    vec3 x_s = ring_slab_x(mu_s, tau, w_transport);
    vec3 x_v = ring_slab_x(mu_v, tau, w_transport);
    vec3 multi_bounce = max(x_s * x_v - vec3(1.0), vec3(0.0));

    if (same_side) {
        float slab_r = ring_transfer(tau * 1.5, mu_s, mu_v, true);
        return 0.35 * w_color * multi_bounce * slab_r;
    }

    float ms = max(mu_s, 1e-4);
    float mv = max(mu_v, 1e-4);
    float slab_t = ring_transfer(tau, ms, mv, false);
    float slant_tau = min(tau * (0.5 / ms + 0.5 / mv), 80.0);
    vec3 k_diff = sqrt(3.0 * max(vec3(1.0) - w_transport, vec3(1e-4)));
    vec3 diff_core = (ms / (ms + mv)) * slant_tau * exp(-slant_tau * (vec3(0.55) + 0.45 * k_diff));
    float buildup = 1.0 - exp(-slant_tau);
    vec3 y_env = (vec3(slab_t) + 0.35 * diff_core) * buildup;
    return 0.25 * w_color * multi_bounce * y_env;
}

struct RingMediumOptics {
    float tau;
    float dust_factor;
    float core_factor;
    float w_f;
    vec3 eff_rgb;
    vec3 w_color;
    vec3 k_diff;
    vec3 x_v;
};

float ring_fast_asin(float x) {
    float ax = clamp(abs(x), 0.0, 1.0);
    float res;
    if (ax < 0.05) {
        float x2 = ax * ax;
        res = ax * (1.0 + x2 * (1.0 / 6.0 + 0.075 * x2));
    } else {
        res = 1.5707963267948966 - sqrt(max(0.0, 1.0 - ax)) *
            (1.5707288 + ax * (-0.2121144 + ax * (0.0742610 - 0.0187293 * ax)));
    }
    return x < 0.0 ? -res : res;
}

float ring_fast_acos_pos(float x) {
    float ax = clamp(x, 0.0, 1.0);
    return sqrt(max(0.0, 1.0 - ax)) *
        (1.5707288 + ax * (-0.2121144 + ax * (0.0742610 - 0.0187293 * ax)));
}

float ring_fast_acos(float x) {
    float c = clamp(x, -1.0, 1.0);
    float pos = ring_fast_acos_pos(abs(c));
    return c < 0.0 ? (3.1415926535897932 - pos) : pos;
}

// Segment-invariant scalar transfer factors for one face side.
vec4 ring_transfer_face(float total_tau, float mu_v, float sun_elev, bool same_side) {
    float mu_s = max(abs(sun_elev), 1e-5);
    float transfer = ring_transfer(total_tau, mu_s, mu_v, same_side);
    float tau = max(total_tau, 0.0);
    if (tau < RING_MS_MIN_TAU) {
        return vec4(transfer, 0.0, 0.0, 0.0);
    }
    if (same_side) {
        float slab_r = ring_transfer(tau * 1.5, mu_s, mu_v, true);
        return vec4(transfer, slab_r, 0.0, 0.0);
    }
    float ms_s = max(mu_s, 1e-4);
    float ms_v = max(mu_v, 1e-4);
    float slab_t = (mu_s >= 1e-4 && mu_v >= 1e-4) ? transfer : ring_transfer(tau, ms_s, ms_v, false);
    float slant_tau = min(tau * (0.5 / ms_s + 0.5 / ms_v), 80.0);
    float buildup = 1.0 - exp(-slant_tau);
    return vec4(transfer, slab_t, slant_tau, buildup);
}

RingMediumOptics ring_medium_setup(float total_tau, float mu_v, vec4 props, vec3 linear_rgb, bool is_textured) {
    RingMediumOptics m;
    m.tau = max(total_tau, 0.0);
    m.dust_factor = 1.0 - smoothstep(0.05, 0.50, total_tau);
    m.core_factor = smoothstep(0.30, 1.80, total_tau);
    float base_w = (is_textured && props.z > 1.0) ? 0.35 : clamp(props.z, 0.0, 1.0);
    m.w_f = clamp(base_w + (is_textured ? (0.12 * m.dust_factor - 0.08 * m.core_factor) : 0.0), 0.0, 1.0);
    m.eff_rgb = clamp(linear_rgb, vec3(0.0), vec3(1.0));

    if (m.tau >= RING_MS_MIN_TAU) {
        float luma = dot(m.eff_rgb, vec3(0.2126, 0.7152, 0.0722));
        vec3 w_transport = mix(vec3(luma), m.eff_rgb, 0.20) * 0.78;
        m.w_color = m.eff_rgb * 0.78;
        vec3 root = sqrt(vec3(1.0) - w_transport);
        m.k_diff = 1.7320508 * root;
        vec3 k_slab = m.k_diff + vec3(0.5);
        float inv_ms_v = 1.0 / max(mu_v, 1e-4);
        vec3 h_v_m1 = (mu_v * (vec3(1.7320508) - m.k_diff)) / (vec3(1.0) + mu_v * m.k_diff);
        m.x_v = vec3(1.0) + h_v_m1 * (vec3(1.0) - exp(-m.tau * (vec3(inv_ms_v) + k_slab)));
    } else {
        m.w_color = vec3(0.0);
        m.k_diff = vec3(0.0);
        m.x_v = vec3(1.0);
    }
    return m;
}

vec3 ring_radiance_face(RingMediumOptics m, vec4 tf, float mu_v, float sun_elev, float cos_theta,
                        vec4 props, bool is_textured, bool same_side, float broaden, float opp) {
    float g_fwd = clamp(props.x * broaden + (is_textured ? 0.14 * m.dust_factor : 0.0), -0.92, 0.92);
    float g_back_mag = clamp(abs(props.y * broaden) + (is_textured ? 0.08 * m.core_factor : 0.0), 0.0, 0.92);
    float phase = mix(ring_phase_hg(cos_theta, -g_back_mag), ring_phase_hg(cos_theta, g_fwd), m.w_f);

    vec3 ss = m.eff_rgb * (0.25 * phase * opp * tf.x);
    vec3 ms = vec3(0.0);
    if (m.tau >= RING_MS_MIN_TAU) {
        float mu_s = max(abs(sun_elev), 1e-5);
        float ms_s = max(mu_s, 1e-4);
        vec3 k_slab = m.k_diff + vec3(0.5);
        vec3 h_s_m1 = (mu_s * (vec3(1.7320508) - m.k_diff)) / (vec3(1.0) + mu_s * m.k_diff);
        vec3 x_s = vec3(1.0) + h_s_m1 * (vec3(1.0) - exp(-m.tau * (vec3(1.0 / ms_s) + k_slab)));
        vec3 multi_bounce = max(x_s * m.x_v - vec3(1.0), vec3(0.0));

        if (same_side) {
            ms = 0.35 * m.w_color * multi_bounce * tf.y;
        } else {
            float ms_v = max(mu_v, 1e-4);
            vec3 diff_core = (ms_s / (ms_s + ms_v)) * tf.z * exp(-tf.z * (vec3(0.55) + 0.45 * m.k_diff));
            vec3 y_env = (vec3(tf.y) + 0.35 * diff_core) * tf.w;
            ms = 0.25 * m.w_color * multi_bounce * y_env;
        }
    }
    float side_scale = same_side ? 1.0 : props.w;
    return (ss + ms) * side_scale;
}

vec3 ring_radiance_core(float total_tau, float mu_v, float sun_elev, float cos_theta,
                        vec4 props, vec3 linear_rgb, bool is_textured, bool same_side,
                        float broaden, float opp) {
    vec4 tf = ring_transfer_face(total_tau, mu_v, sun_elev, same_side);
    RingMediumOptics m = ring_medium_setup(total_tau, mu_v, props, linear_rgb, is_textured);
    return ring_radiance_face(m, tf, mu_v, sun_elev, cos_theta, props, is_textured, same_side, broaden, opp);
}

vec3 ring_radiance_planetshine_pair_medium(RingMediumOptics m, float total_tau, float mu_v, vec4 props, bool is_textured,
                                           bool has_same, vec3 same_flux, vec4 same_tf, float same_elev,
                                           float same_phase, float same_broaden, float same_opp,
                                           bool has_other, vec3 other_flux, vec4 other_tf, float other_elev,
                                           float other_phase, float other_broaden) {
    bool do_other = has_other && (props.w > 0.0);
    if (!has_same && !do_other) return vec3(0.0);
    vec3 out_rad = vec3(0.0);
    if (has_same) {
        out_rad += same_flux * ring_radiance_face(
            m, same_tf, mu_v, same_elev, same_phase, props, is_textured, true, same_broaden, same_opp);
    }
    if (do_other) {
        out_rad += other_flux * ring_radiance_face(
            m, other_tf, mu_v, other_elev, other_phase, props, is_textured, false, other_broaden, 1.0);
    }
    return out_rad;
}

vec3 ring_radiance_planetshine_pair(float total_tau, float mu_v, vec4 props, vec3 linear_rgb, bool is_textured,
                                    bool has_same, vec3 same_flux, vec4 same_tf, float same_elev,
                                    float same_phase, float same_broaden, float same_opp,
                                    bool has_other, vec3 other_flux, vec4 other_tf, float other_elev,
                                    float other_phase, float other_broaden) {
    bool do_other = has_other && (props.w > 0.0);
    if (!has_same && !do_other) return vec3(0.0);
    RingMediumOptics m = ring_medium_setup(total_tau, mu_v, props, linear_rgb, is_textured);
    return ring_radiance_planetshine_pair_medium(
        m, total_tau, mu_v, props, is_textured,
        has_same, same_flux, same_tf, same_elev, same_phase, same_broaden, same_opp,
        has_other, other_flux, other_tf, other_elev, other_phase, other_broaden);
}

vec3 ring_radiance_rgb(float total_tau, float mu_v, float sun_elev, float cos_theta,
                       vec4 props, vec3 linear_rgb, bool is_textured, bool same_side) {
    float opp = same_side ? ring_opposition_factor(-cos_theta, total_tau) : 1.0;
    return ring_radiance_core(total_tau, mu_v, sun_elev, cos_theta, props, linear_rgb, is_textured, same_side, 1.0, opp);
}

float ring_opposition_factor_extended_terms(float alpha_rad, float tau_val, float spread) {
    float eff_alpha = sqrt(alpha_rad * alpha_rad + 0.36 * spread * spread);

    float shoe_width = 0.030;
    float shoe_amp = 0.55 * smoothstep(0.02, 0.45, tau_val) * (shoe_width / (shoe_width + 0.75 * spread));
    float shoe = shoe_amp / (1.0 + eff_alpha / shoe_width);

    float cboe_width = 0.0025;
    float cboe_amp = 0.35 * smoothstep(0.005, 0.18, tau_val) * (cboe_width / (cboe_width + 0.75 * spread));
    float x_cb = eff_alpha / cboe_width;
    float cboe = cboe_amp / (1.0 + x_cb * x_cb);

    return 1.0 + shoe + cboe;
}

float ring_opposition_factor_extended(float cos_alpha, float tau_val, float source_sin_radius) {
    float ca = clamp(cos_alpha, -1.0, 1.0);
    float alpha_rad = ring_fast_acos(clamp(ca, -1.0, 1.0 - 1e-7));
    float spread = ring_fast_asin(clamp(source_sin_radius, 0.0, 0.999));
    return ring_opposition_factor_extended_terms(alpha_rad, tau_val, spread);
}

vec3 ring_extended_terms(float cos_theta, float source_sin_radius, bool same_side) {
    float broaden = 1.0 / (1.0 + 1.2 * source_sin_radius);
    float spread = ring_fast_asin(clamp(source_sin_radius, 0.0, 0.999));
    float alpha_rad = same_side ? ring_fast_acos(clamp(-cos_theta, -1.0, 1.0 - 1e-7)) : 0.0;
    return vec3(broaden, spread, alpha_rad);
}

vec3 ring_radiance_rgb_extended_terms(float total_tau, float mu_v, float sun_elev, float cos_theta,
                                      vec4 props, vec3 linear_rgb, bool is_textured, bool same_side,
                                      float source_sin_radius, vec3 ext_terms) {
    if (source_sin_radius <= 1e-5) {
        return ring_radiance_rgb(total_tau, mu_v, sun_elev, cos_theta, props, linear_rgb, is_textured, same_side);
    }
    float opp = same_side ? ring_opposition_factor_extended_terms(ext_terms.z, total_tau, ext_terms.y) : 1.0;
    return ring_radiance_core(total_tau, mu_v, sun_elev, cos_theta, props, linear_rgb, is_textured, same_side, ext_terms.x, opp);
}

vec3 ring_radiance_rgb_extended(float total_tau, float mu_v, float sun_elev, float cos_theta,
                                vec4 props, vec3 linear_rgb, bool is_textured, bool same_side,
                                float source_sin_radius) {
    if (source_sin_radius <= 1e-5) {
        return ring_radiance_rgb(total_tau, mu_v, sun_elev, cos_theta, props, linear_rgb, is_textured, same_side);
    }
    vec3 ext_terms = ring_extended_terms(cos_theta, source_sin_radius, same_side);
    return ring_radiance_rgb_extended_terms(total_tau, mu_v, sun_elev, cos_theta, props, linear_rgb,
                                            is_textured, same_side, source_sin_radius, ext_terms);
}

int ring_visible_source_count(float sin_elevation, float sin_radius) {
    return (sin_radius > 1e-6 && abs(sin_elevation) < sin_radius) ? 4 : 1;
}

float ring_strip_area(float a, float b) {
    if (b <= a) return 0.0;
    float ca = clamp(a, -1.0, 1.0);
    float cb = clamp(b, -1.0, 1.0);
    float area_a = ca * sqrt(max(0.0, 1.0 - ca * ca)) + asin(ca);
    float area_b = cb * sqrt(max(0.0, 1.0 - cb * cb)) + asin(cb);
    return (area_b - area_a) / 3.14159265358979323846;
}

float ring_strip_moment(float a, float b) {
    if (b <= a) return 0.0;
    float ca = clamp(a, -1.0, 1.0);
    float cb = clamp(b, -1.0, 1.0);
    float ma = pow(max(0.0, 1.0 - ca * ca), 1.5);
    float mb = pow(max(0.0, 1.0 - cb * cb), 1.5);
    return 2.0 * (ma - mb) / (3.0 * 3.14159265358979323846);
}

void ring_visible_polar_nodes(float sin_elevation, float sin_radius, out vec4 offsets, out vec4 weights) {
    float horizon = clamp(-sin_elevation / max(sin_radius, 1e-7), -1.0, 1.0);
    float bounds[5] = float[5](-1.0, -0.5 * (1.0 - horizon), horizon, 0.5 * (horizon + 1.0), 1.0);
    offsets = vec4(0.0);
    weights = vec4(0.0);
    for (int i = 0; i < 4; ++i) {
        float w = ring_strip_area(bounds[i], bounds[i + 1]);
        weights[i] = w;
        offsets[i] = w > 1e-6 ? ring_strip_moment(bounds[i], bounds[i + 1]) / w : 0.5 * (bounds[i] + bounds[i + 1]);
    }
}

vec4 ring_visible_polar_sample(vec3 center_dir, vec3 ring_normal, float sin_radius, float offset, float weight) {
    vec3 polar = ring_normal - center_dir * dot(ring_normal, center_dir);
    float polar_len = length(polar);
    polar = polar_len > 1e-6 ? polar / polar_len : ring_normal;
    float sin_delta = clamp(offset, -1.0, 1.0) * sin_radius;
    float cos_delta = sqrt(max(0.0, 1.0 - sin_delta * sin_delta));
    return vec4(normalize(center_dir * cos_delta + polar * sin_delta), weight);
}
