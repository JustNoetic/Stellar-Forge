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
    float oblateness;
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
uniform sampler2D u_ring_props;
uniform sampler3D u_ringshine_lut;
uniform sampler3D u_ringshine_cdf_lut;

uniform vec3 u_sun_dir[16];
uniform int u_num_ring_planes;
uniform vec3 u_ring_normal[16];
uniform vec4 u_ring_params[16];
uniform int u_ringshine_band_count;
uniform bool u_ringshine_oblate_enabled;
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
    if (abs(frag_elevation) < 1e-5) {
        out_color = vec4(0.0);
        return;
    }
    float sin_lat = clamp(abs(frag_elevation), 0.0, 0.999);
    float obl = (u_ringshine_oblate_enabled) ? clamp(u_ring_planes[k].oblateness, 0.0, 0.3) : 0.0;
    float f_factor = 1.0 - obl;
    float cos_lat = sqrt(max(0.0, 1.0 - sin_lat * sin_lat));
    float denom_rho = sqrt(f_factor * f_factor * cos_lat * cos_lat + sin_lat * sin_lat);
    float rho = (obl > 1e-5) ? (f_factor / max(1e-6, denom_rho)) : 1.0;

    vec3 L = u_sun_dir[k];
    float sun_elevation = dot(L, ring_normal);
    float sin_sun_elev = clamp(abs(sun_elevation), 1e-4, 1.0);
    float cos_sun_elev = sqrt(max(0.0, 1.0 - sin_sun_elev * sin_sun_elev));

    float same_hemisphere = sun_elevation * frag_elevation;
    float same_hemi_t = (same_hemisphere >= 0.0) ? 1.0 : 0.0;

    int band_count = clamp(u_ringshine_band_count, 4, 1024);
    vec3 total_ring_irradiance = vec3(0.0);
    float log_r_ratio = log(max(1.001, outer_r / max(1e-5, inner_r)));

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
            samp.rgb = pow(samp.rgb, vec3(2.2));
            float w_area = r_s * QUAD_W[s];
            sum_area += w_area;
            sum_alpha += samp.a * w_area;
            sum_color += samp.rgb * (samp.a * w_area);
        }

        if (sum_area < 1e-9 || sum_alpha < 1e-9) continue;

        float band_raw_alpha = sum_alpha / sum_area;
        vec3 band_raw_rgb = sum_color / sum_alpha;

        float alpha_phys = clamp(band_raw_alpha * opacity, 0.0, 0.999);
        float tau_phys = -log(max(1e-4, 1.0 - alpha_phys));

        float frac_mid = clamp((r_mid - inner_r) / max(1e-5, outer_r - inner_r), 0.0, 1.0);
        vec4 props = texture(u_ring_props, vec2(frac_mid, (float(k) + 0.5) / 4.0));
        float layer_asym = props.r;
        float layer_backasym = props.g;
        float layer_scatter = props.b;
        bool plane_is_textured = (props.a >= 50.0);
        float layer_unlit = plane_is_textured ? (props.a - 100.0) : props.a;

        float norm_r = r_mid / max(1e-5, host_radius);

        // --- PLANETARY SHADOW ON RING (CDF & LUT) ---
        float delta_alpha_shadow = 0.0;
        if (obl > 1e-5) {
            float a_shadow_sq = (f_factor * f_factor * cos_sun_elev * cos_sun_elev + sin_sun_elev * sin_sun_elev) / max(1e-8, sin_sun_elev * sin_sun_elev);
            float a_shadow = sqrt(a_shadow_sq);
            if (norm_r <= a_shadow) {
                float C_eff = (f_factor * cos_sun_elev) / sqrt(max(1e-8, f_factor * f_factor * cos_sun_elev * cos_sun_elev + sin_sun_elev * sin_sun_elev));
                float arg = sqrt(max(0.0, 1.0 - 1.0 / (norm_r * norm_r))) / max(1e-4, C_eff);
                delta_alpha_shadow = acos(clamp(arg, 0.0, 1.0));
            }
        } else {
            if (norm_r <= 1.0 / sin_sun_elev) {
                float arg = sqrt(max(0.0, 1.0 - 1.0 / (norm_r * norm_r))) / max(1e-4, cos_sun_elev);
                delta_alpha_shadow = acos(clamp(arg, 0.0, 1.0));
            }
        }

        float psi1 = -delta_alpha_shadow - phi_center;
        float psi2 = +delta_alpha_shadow - phi_center;
        float v_tex = clamp((norm_r - 1.0) / 4.0, 0.0, 1.0);

        float cdf1 = eval_ringshine_cdf(psi1, v_tex, sin_lat);
        float cdf2 = eval_ringshine_cdf(psi2, v_tex, sin_lat);

        float shadow_fraction = clamp(0.5 * (cdf2 - cdf1), 0.0, 1.0);
        float band_illum = max(0.0, 1.0 - shadow_fraction);

        if (band_illum < 1e-6) continue;

        // --- EXACT SLANT ANGLE & DISTANCE ---
        float d2 = max(1e-6, norm_r * norm_r + rho * rho - 2.0 * norm_r * rho * cos_lat);
        float d = sqrt(d2);

        float cosViewRayVertical = clamp((rho * sin_lat) / d, 0.001, 1.0);
        float cosLightRayVertical = clamp(sin_sun_elev, 0.001, 1.0);
        float viewDensity = tau_phys / cosViewRayVertical;
        float lightDensity = tau_phys / cosLightRayVertical;

        // --- DOMINANT GEOMETRIC PHASE ANGLE (SMOOTH SHADOW-AWARE) ---
        // When the ring band is partially shadowed, unshadowed light originates from ring elements
        // displaced away from retro-reflection. Modulating the horizontal phase cosine smoothly
        // by (1.0 - 0.45 * shadow_fraction) accounts for this shift with guaranteed C_inf continuity (no vertical seam or boxy edge).
        float cos_phi_eff = cos(phi_center) * (1.0 - 0.45 * shadow_fraction);
        // On the lit side (same_hemi_t > 0.5), light bounces upwards back towards the sun (backscattering: negative vertical term).
        // On the unlit side (same_hemi_t <= 0.5), light penetrates downwards through the ring slab (forward scattering: positive vertical term).
        float vert_phase = (same_hemi_t > 0.5 ? -1.0 : 1.0) * sin_sun_elev * (rho * sin_lat);
        // Dominant scattering phase cosine cos(Theta) = -(L . V_ring_to_surf).
        // Incident light propagates in direction -L; scattered light propagates in direction (P_surf - P_ring)/d.
        // Horizontally, inward vector to planet is -(norm_r - rho*cos_lat), giving -(L . V)_xy = -(norm_r - rho*cos_lat) * cos_sun_elev * cos_phi_eff / d.
        float cos_theta_phase = (-(norm_r - rho * cos_lat) * cos_sun_elev * cos_phi_eff + vert_phase) / d;
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

        // Single scattering (no opposition surge: ring elements at opposition are inside the planet's shadow)
        float scatteredLight_sunlit = viewDensity / (viewDensity + lightDensity) * (1.0 - exp(-viewDensity - lightDensity));
        float denom = lightDensity - viewDensity;
        float scatteredLight_unlit = (abs(denom) > 1e-6) ?
            (exp(-viewDensity) - exp(-lightDensity)) * viewDensity / denom :
            viewDensity * exp(-viewDensity);

        // Multiple scattering
        float ms_sunlit = AnalyticMultipleScattering(cosViewRayVertical, cosLightRayVertical, tau_phys, true) * ms_weight;

        vec3 band_color_sunlit = band_raw_rgb * (scatteredLight_sunlit * pf + ms_sunlit);
        vec3 band_color_unlit  = band_raw_rgb * (scatteredLight_unlit * pf) * layer_unlit;
        vec3 band_color = mix(band_color_unlit, band_color_sunlit, same_hemi_t);

        float u_tex_lut = 0.5 / 256.0 + sin_lat * (255.0 / 256.0);
        float v_tex_lut = 0.5 / 256.0 + v_tex * (255.0 / 256.0);
        float w_tex_lut = 0.5 / 16.0 + clamp(obl / 0.3, 0.0, 1.0) * (15.0 / 16.0);
        float kernel_val = texture(u_ringshine_lut, vec3(u_tex_lut, v_tex_lut, w_tex_lut)).r;

        total_ring_irradiance += band_color * kernel_val * dr * band_illum;
    }

    // Multiply by PI to convert physically exact radiance to the Lambertian engine convention
    out_color = vec4(total_ring_irradiance * 3.14159265358979, 1.0);
}
