const float RING_MAX_ALPHA = 0.99999994;

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
    float xs = min(tau / ms, 80.0);
    float xv = min(tau / mv, 80.0);

    if (same_side) {
        float a = xs + xv;
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
    return xv * (exp(-xv) - exp(-xs)) / d;
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
    return (1.0 - gg * gg) / (denom * sqrt(denom));
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
    if (tau < 1e-5) return vec3(0.0);

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

vec3 ring_radiance_rgb(float total_tau, float mu_v, float sun_elev, float cos_theta,
                       vec4 props, vec3 linear_rgb, bool is_textured, bool same_side) {
    float mu_s = max(abs(sun_elev), 1e-5);
    float g_fwd = props.x;
    float g_back = props.y;
    float scatter = props.z;
    float unlit_factor = props.w;

    vec3 eff_rgb = linear_rgb;
    if (is_textured && !same_side) {
        float core_factor = smoothstep(0.60, 1.80, total_tau);
        float luma = max(dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
        vec3 chroma = linear_rgb / luma;
        vec3 chroma_unlit = mix(chroma, vec3(1.0), 0.65 * core_factor);
        float unlit_luma = pow(luma, 0.35) * 0.60;
        eff_rgb = unlit_luma * chroma_unlit;
    }

    float eff_scatter = same_side ? (scatter * 0.30) : scatter;
    float phase = ring_phase_dhg(cos_theta, eff_scatter, g_fwd, g_back, total_tau, is_textured);
    float opp = same_side ? ring_opposition_factor(-cos_theta, total_tau) : 1.0;
    float transfer = ring_transfer(total_tau, mu_s, mu_v, same_side);

    float ss_scale = same_side ? 0.38 : 0.25;
    vec3 ss = eff_rgb * (ss_scale * phase * opp * transfer);
    vec3 ms = ring_multiple_scattering(total_tau, mu_s, mu_v, same_side, eff_rgb);
    float side_scale = same_side ? 1.0 : unlit_factor;
    return (ss + ms) * side_scale;
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
