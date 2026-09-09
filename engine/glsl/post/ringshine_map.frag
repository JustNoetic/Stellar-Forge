#version 460 core
#define PI 3.14159265358979323846
in vec2 f_uv;
out vec4 out_color;

struct RingPlane {
    vec3 color;
    float inner_r;
    float outer_r;
    float opacity;
    float scatter;
    float asymmetry;
    float backscatter;
    int row_idx;
    float is_textured;
    float unlit_factor;
    float saturation;
    float hue_shift;
    float brightness;
    float alpha_boost;
};

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - 3.0);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 adjust_hsba(vec3 color, float hue_shift, float saturation, float brightness) {
    vec3 hsv = rgb2hsv(color);
    hsv.x = fract(hsv.x + hue_shift);
    hsv.y = clamp(hsv.y * saturation, 0.0, 2.0);
    hsv.z = hsv.z * brightness;
    return hsv2rgb(hsv);
}

uniform sampler2D u_ring_gradients;
uniform sampler2D u_ringshine_lut;
uniform sampler3D u_ringshine_cdf_lut;

uniform vec3 u_sun_dir;
uniform int u_num_ring_planes;
uniform vec3 u_ring_normal[16];
uniform vec4 u_ring_params[16];
uniform int u_ringshine_band_count;
uniform RingPlane u_ring_planes[16];

float HenyeyGreensteinPhaseFunction(float eccentricity, float viewDirDotLight) {
    float g = eccentricity;
    return (1.0 - g * g) / (4.0 * PI * pow(max(1e-6, 1.0 + g * g - 2.0 * g * viewDirDotLight), 1.5));
}

vec2 GetRingPhaseFunctionStrengths(float alpha) {
    float dust_to_chunks = clamp((alpha - 0.1) / 0.5, 0.0, 1.0);
    return mix(vec2(0.95, 0.05), vec2(0.5, 0.5), dust_to_chunks);
}

vec2 GetRingPhaseFunctionsUnweighted(float dotLight, float asym, float back_asym) {
    return vec2(HenyeyGreensteinPhaseFunction(asym, dotLight), HenyeyGreensteinPhaseFunction(back_asym, dotLight));
}

float GetRingPhaseFunctions(float dotLight, float alpha, float asym, float back_asym) {
    vec2 phaseFunctions = GetRingPhaseFunctionsUnweighted(dotLight, asym, back_asym) * GetRingPhaseFunctionStrengths(alpha);
    return phaseFunctions.x + phaseFunctions.y;
}

float CornetteShanksPhaseFunction(float eccentricity, float viewDirDotLight) {
    float g = eccentricity;
    float g2 = g * g;
    float mu = viewDirDotLight;
    float denom = pow(max(1e-6, 1.0 + g2 - 2.0 * g * mu), 1.5);
    return (1.5 * (1.0 + mu * mu) * (1.0 - g2)) / ((2.0 + g2) * denom * 4.0 * PI);
}

float OppositionSurge(float cos_phase, float columnDensity) {
    float phase_angle = acos(clamp(cos_phase, -1.0, 1.0));
    float density_scale = clamp(columnDensity / 1.5, 0.0, 1.0);
    float shoe_width = 0.07;
    float shoe_amp = 0.8 * density_scale;
    float shoe = 1.0 + shoe_amp / (1.0 + phase_angle / shoe_width);

    float cboe_width = 0.006;
    float cboe_amp = 0.3 * density_scale;
    float cboe = 1.0 + cboe_amp * exp(-phase_angle / cboe_width);

    return shoe * cboe;
}

float HapkeHFunction(float mu, float gamma) {
    return (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * gamma);
}

float AnalyticMultipleScattering(float mu_v, float mu_0, float tau, bool onLitSide) {
    float w0 = 0.92;
    float gamma = sqrt(max(1e-4, 1.0 - w0));
    float Hv = HapkeHFunction(mu_v, gamma);
    float H0 = HapkeHFunction(mu_0, gamma);

    if (onLitSide) {
        float path_term = 1.0 - exp(-tau * (1.0 / mu_v + 1.0 / mu_0));
        float mu_ratio = mu_0 / max(1e-4, mu_v + mu_0);
        return max(0.0, w0 * mu_ratio * (Hv * H0 - 1.0) * path_term) / (4.0 * PI);
    } else {
        float exp_trans = exp(-gamma * tau / max(1e-4, mu_0));
        float direct_atten = exp(-tau / max(1e-4, mu_0));
        float diff_trans = max(0.0, exp_trans - direct_atten);
        float boundary_factor = (Hv - 1.0) * (H0 - 1.0);
        return max(0.0, w0 * diff_trans * (1.0 + 0.5 * boundary_factor) * mu_0) / (4.0 * PI);
    }
}

const vec4 QUAD_T = vec4(0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558);
const vec4 QUAD_W = vec4(0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226);

float eval_ringshine_cdf(float angle, float v_tex, float sin_lat) {
    float TWO_PI = 6.28318530717958647692;
    float a_mod = angle - TWO_PI * floor((angle + PI) / TWO_PI);
    float k = floor((angle + PI) / TWO_PI);
    float u = clamp(abs(a_mod) / PI, 0.0, 1.0);

    float u_tex = 0.5 / 128.0 + u * (127.0 / 128.0);
    float v_tex_mapped = 0.5 / 128.0 + v_tex * (127.0 / 128.0);
    float sin_lat_mapped = 0.5 / 64.0 + sin_lat * (63.0 / 64.0);

    float base_cdf = texture(u_ringshine_cdf_lut, vec3(u_tex, v_tex_mapped, sin_lat_mapped)).r;
    float signed_cdf = (a_mod < 0.0) ? -base_cdf : base_cdf;
    return 2.0 * k + signed_cdf;
}

void main() {
    float slot = f_uv.y * 16.0;
    int k = int(floor(slot));
    float local_v = fract(slot);

    if (k >= u_num_ring_planes) {
        out_color = vec4(0.0);
        return;
    }

    vec3 ring_normal = u_ring_normal[k];
    float inner_r = u_ring_params[k].x;
    float outer_r = u_ring_params[k].y;
    float opacity = u_ring_params[k].z;
    float host_radius = u_ring_params[k].w > 1e-5 ? u_ring_params[k].w : inner_r * 0.7;

    if (opacity < 1e-4 || outer_r <= inner_r) {
        out_color = vec4(0.0);
        return;
    }

    float x = f_uv.x * 2.0 - 1.0;
    float phi_center = sign(x) * pow(abs(x), 1.5) * PI;

    float y = local_v * 2.0 - 1.0;
    float frag_elevation = sign(y) * pow(abs(y), 1.5);
    float sin_lat = clamp(abs(frag_elevation), 0.001, 0.999);

    vec3 L = u_sun_dir;
    float sun_elevation = dot(L, ring_normal);
    float sin_sun_elev = clamp(abs(sun_elevation), 1e-4, 1.0);
    float cos_sun_elev = sqrt(max(0.0, 1.0 - sin_sun_elev * sin_sun_elev));

    float same_hemisphere = sun_elevation * frag_elevation;
    float same_hemi_t = smoothstep(-0.02, 0.02, same_hemisphere);

    int band_count = clamp(u_ringshine_band_count, 4, 1024);
    vec3 total_ring_irradiance = vec3(0.0);
    float log_r_ratio = log(max(1.001, outer_r / max(1e-5, inner_r)));

    bool plane_is_textured = (u_ring_planes[k].is_textured > 0.5);
    float layer_hue   = u_ring_planes[k].hue_shift;
    float layer_sat   = u_ring_planes[k].saturation;
    float layer_bri   = u_ring_planes[k].brightness;
    float layer_unlit = u_ring_planes[k].unlit_factor;
    float layer_boost = u_ring_planes[k].alpha_boost;
    float layer_asym  = u_ring_planes[k].asymmetry;
    float layer_backasym = u_ring_planes[k].backscatter;
    float layer_scatter  = u_ring_planes[k].scatter;

    for (int m = 0; m < band_count; m++) {
        float u0 = float(m) / float(band_count);
        float u1 = float(m + 1) / float(band_count);
        float u_mid = 0.5 * (u0 + u1);

        float r_m0 = inner_r * exp(u0 * log_r_ratio);
        float r_m1 = inner_r * exp(u1 * log_r_ratio);
        float r_mid = inner_r * exp(u_mid * log_r_ratio);
        float dr = (r_m1 - r_m0) / max(1e-5, host_radius);

        // --- 4-POINT AREA-WEIGHTED GAUSS-LEGENDRE QUADRATURE ---
        float sum_area = 0.0;
        float sum_alpha = 0.0;
        vec3 sum_color = vec3(0.0);

        for (int s = 0; s < 4; s++) {
            float t_sub = u0 + QUAD_T[s] * (u1 - u0);
            float r_s = inner_r * exp(t_sub * log_r_ratio);
            float frac_s = clamp((r_s - inner_r) / max(1e-5, outer_r - inner_r), 0.0, 1.0);
            vec4 samp = texture(u_ring_gradients, vec2(frac_s, (float(k) + 0.5) / 16.0));
            float w_area = r_s * QUAD_W[s];
            sum_area += w_area;
            sum_alpha += samp.a * w_area;
            sum_color += samp.rgb * (samp.a * w_area);
        }

        if (sum_area < 1e-9 || sum_alpha < 1e-9) continue;

        float band_raw_alpha = sum_alpha / sum_area;
        vec3 band_raw_rgb = sum_color / sum_alpha;

        if (plane_is_textured && (abs(layer_hue) > 1e-4 || abs(layer_sat - 1.0) > 1e-4 || abs(layer_bri - 1.0) > 1e-4)) {
            band_raw_rgb = adjust_hsba(band_raw_rgb, layer_hue, layer_sat, layer_bri);
        }

        if (plane_is_textured && band_raw_alpha > 1e-5 && abs(layer_boost - 1.0) > 1e-4) {
            band_raw_alpha = clamp(pow(band_raw_alpha, 1.0 / max(0.01, layer_boost)), 0.0, 1.0);
        }

        float alpha_phys = clamp(band_raw_alpha * opacity, 0.0, 0.999);
        float tau_phys = -log(max(1e-4, 1.0 - alpha_phys));

        // --- EXACT SLANT ANGLE & DISTANCE ---
        float norm_r = r_mid / max(1e-5, host_radius);
        float cos_lat = sqrt(max(0.0, 1.0 - sin_lat * sin_lat));
        float d2 = max(1e-6, norm_r * norm_r + 1.0 - 2.0 * norm_r * cos_lat);
        float d = sqrt(d2);

        float cosViewRayVertical = clamp(sin_lat / d, 0.001, 1.0);
        float cosLightRayVertical = clamp(sin_sun_elev, 0.001, 1.0);
        float viewDensity = tau_phys / cosViewRayVertical;
        float lightDensity = tau_phys / cosLightRayVertical;

        // --- DOMINANT GEOMETRIC PHASE ANGLE ---
        float cos_theta_phase = ((norm_r - cos_lat) * cos_sun_elev * cos(phi_center) + sin_sun_elev * sin_lat) / d;
        cos_theta_phase = clamp(cos_theta_phase, -1.0, 1.0);

        float pf = 0.0;
        float ms_weight = 0.0;
        if (plane_is_textured) {
            pf = GetRingPhaseFunctions(cos_theta_phase, alpha_phys, layer_asym, layer_backasym);
            ms_weight = clamp((alpha_phys - 0.1) / 0.5, 0.0, 1.0);
        } else {
            float pf_fwd = CornetteShanksPhaseFunction(layer_asym, cos_theta_phase);
            float pf_back = CornetteShanksPhaseFunction(layer_backasym, cos_theta_phase);
            float bal = clamp(layer_scatter, 0.0, 1.0);
            pf = mix(pf_back, pf_fwd, bal);
            ms_weight = 1.0 - bal;
        }

        // Single scattering
        float scatteredLight_sunlit = viewDensity / (viewDensity + lightDensity) * (1.0 - exp(-viewDensity - lightDensity));
        float denom = lightDensity - viewDensity;
        float scatteredLight_unlit = (abs(denom) > 1e-6) ?
            (exp(-viewDensity) - exp(-lightDensity)) * viewDensity / denom :
            viewDensity * exp(-viewDensity);

        // Multiple scattering
        float ms_sunlit = AnalyticMultipleScattering(cosViewRayVertical, cosLightRayVertical, tau_phys, true) * ms_weight;
        float ms_unlit  = AnalyticMultipleScattering(cosViewRayVertical, cosLightRayVertical, tau_phys, false) * ms_weight;

        // Opposition surge
        float surge = OppositionSurge(-cos_theta_phase, tau_phys);

        vec3 band_color_sunlit = band_raw_rgb * (scatteredLight_sunlit * pf * surge + (plane_is_textured ? ms_sunlit : 0.0));
        vec3 band_color_unlit  = band_raw_rgb * (scatteredLight_unlit * pf + (plane_is_textured ? ms_unlit : 0.0)) * layer_unlit;
        vec3 band_color = mix(band_color_unlit, band_color_sunlit, same_hemi_t);

        // --- PLANETARY SHADOW ON RING (CDF & LUT) ---
        float delta_alpha_shadow = 0.0;
        if (norm_r <= 1.0 / sin_sun_elev) {
            float arg = sqrt(max(0.0, 1.0 - 1.0 / (norm_r * norm_r))) / max(1e-4, cos_sun_elev);
            delta_alpha_shadow = acos(clamp(arg, 0.0, 1.0));
        }

        float psi1 = -delta_alpha_shadow - phi_center;
        float psi2 = +delta_alpha_shadow - phi_center;
        float v_tex = clamp((norm_r - 1.0) / 4.0, 0.0, 1.0);

        float cdf1 = eval_ringshine_cdf(psi1, v_tex, sin_lat);
        float cdf2 = eval_ringshine_cdf(psi2, v_tex, sin_lat);

        float shadow_fraction = clamp(0.5 * (cdf2 - cdf1), 0.0, 1.0);
        float band_illum = max(0.0, 1.0 - shadow_fraction);

        float u_tex_lut = 0.5 / 256.0 + sin_lat * (255.0 / 256.0);
        float v_tex_lut = 0.5 / 256.0 + v_tex * (255.0 / 256.0);
        float kernel_val = texture(u_ringshine_lut, vec2(u_tex_lut, v_tex_lut)).r;

        total_ring_irradiance += band_color * kernel_val * dr * band_illum;
    }

    // Multiply by PI to convert physically exact radiance to the Lambertian engine convention
    out_color = vec4(total_ring_irradiance * 3.14159265358979, 1.0);
}
