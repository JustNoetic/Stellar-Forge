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

    for (int m = 0; m < band_count; m++) {
        float u0 = float(m) / float(band_count);
        float u1 = float(m + 1) / float(band_count);
        float u_mid = 0.5 * (u0 + u1);

        float r_m0 = inner_r * exp(u0 * log_r_ratio);
        float r_m1 = inner_r * exp(u1 * log_r_ratio);
        float r_mid = inner_r * exp(u_mid * log_r_ratio);
        float dr = (r_m1 - r_m0) / max(1e-5, host_radius);
        float frac_mid = clamp((r_mid - inner_r) / max(1e-5, outer_r - inner_r), 0.0, 1.0);

        vec4 ring_texel = texture(u_ring_gradients, vec2(frac_mid, (float(k) + 0.5) / 16.0));
        if (ring_texel.a < 1e-4) continue;

        bool plane_is_textured = (u_ring_planes[k].is_textured > 0.5);
        float layer_hue   = u_ring_planes[k].hue_shift;
        float layer_sat   = u_ring_planes[k].saturation;
        float layer_bri   = u_ring_planes[k].brightness;
        float layer_unlit = u_ring_planes[k].unlit_factor;
        float layer_boost = u_ring_planes[k].alpha_boost;

        if (plane_is_textured && (abs(layer_hue) > 1e-4 || abs(layer_sat - 1.0) > 1e-4 || abs(layer_bri - 1.0) > 1e-4)) {
            ring_texel.rgb = adjust_hsba(ring_texel.rgb, layer_hue, layer_sat, layer_bri);
        }

        if (plane_is_textured && ring_texel.a > 1e-5 && abs(layer_boost - 1.0) > 1e-4) {
            ring_texel.a = clamp(pow(ring_texel.a, 1.0 / max(0.01, layer_boost)), 0.0, 1.0);
        }

        float alpha_phys = clamp(ring_texel.a * opacity, 0.0, 0.999);
        float tau_phys = -log(max(1e-4, 1.0 - alpha_phys));

        float cosViewRayVertical = max(sin_lat, 1e-4);
        float cosLightRayVertical = max(sin_sun_elev, 1e-4);
        float viewDensity = tau_phys / cosViewRayVertical;
        float lightDensity = tau_phys / cosLightRayVertical;

        float layer_asym = u_ring_planes[k].asymmetry;
        float layer_backasym = u_ring_planes[k].backscatter;

        float scatteredLight_sunlit = viewDensity / (viewDensity + lightDensity) * (1.0 - exp(-viewDensity - lightDensity));
        float pf_sunlit = GetRingPhaseFunctions(0.70, alpha_phys, layer_asym, layer_backasym);
        
        // Hapke H-Functions for Multiple Scattering
        float w0 = 0.92;
        float gamma = sqrt(max(1e-4, 1.0 - w0));
        float Hv = (1.0 + 2.0 * cosViewRayVertical) / (1.0 + 2.0 * cosViewRayVertical * gamma);
        float H0 = (1.0 + 2.0 * cosLightRayVertical) / (1.0 + 2.0 * cosLightRayVertical * gamma);
        float path_term = 1.0 - exp(-tau_phys * (1.0 / cosViewRayVertical + 1.0 / cosLightRayVertical));
        float mu_ratio = cosLightRayVertical / max(1e-4, cosViewRayVertical + cosLightRayVertical);
        
        float inv_4pi = 1.0 / (4.0 * 3.14159265358979);
        float dust_to_chunks = clamp((alpha_phys - 0.1) / 0.5, 0.0, 1.0);
        float ms_sunlit = max(0.0, w0 * mu_ratio * (Hv * H0 - 1.0) * path_term * dust_to_chunks) * inv_4pi;

        vec3 band_color_sunlit = ring_texel.rgb * (scatteredLight_sunlit * pf_sunlit + (plane_is_textured ? ms_sunlit : 0.0));

        float denominator = lightDensity - viewDensity;
        float scatteredLight_unlit = 0.0;
        if (abs(denominator) > 1e-6) {
            scatteredLight_unlit = (exp(-viewDensity) - exp(-lightDensity)) * viewDensity / denominator;
        } else {
            scatteredLight_unlit = viewDensity * exp(-viewDensity);
        }
        float pf_unlit = GetRingPhaseFunctions(-0.70, alpha_phys, layer_asym, layer_backasym);
        vec3 band_color_unlit = ring_texel.rgb * (scatteredLight_unlit * pf_unlit * layer_unlit);

        vec3 band_color = mix(band_color_unlit, band_color_sunlit, same_hemi_t);

        float norm_r = r_mid / max(1e-5, host_radius);

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
