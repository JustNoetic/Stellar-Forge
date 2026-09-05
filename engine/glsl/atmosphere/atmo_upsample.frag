#version 460 core

in vec2 v_texcoord;

layout(location = 0, index = 0) out vec4 out_scattered;
layout(location = 0, index = 1) out vec4 out_transmittance;

uniform sampler2D u_lowres_scatter;
uniform sampler2D u_lowres_trans;
uniform sampler2D u_highres_depth;

uniform float u_depth_C;
uniform float u_far;
uniform vec2 u_lowres_size;
uniform bool u_vrs_enabled;

float linearize_depth(float log_depth) {
    if (log_depth >= 0.99999) return 1e12;
    float log_far_denom = log2(u_depth_C * u_far + 1.0);
    return (exp2(log_depth * log_far_denom) - 1.0) / u_depth_C;
}

void main() {
    float high_depth_raw = texture(u_highres_depth, v_texcoord).r;
    float high_depth = linearize_depth(high_depth_raw);

    // Coordinate in low-res texel space
    vec2 texel_pos = v_texcoord * u_lowres_size - 0.5;
    vec2 p00 = floor(texel_pos);
    vec2 f = fract(texel_pos);

    // 4 surrounding sample UVs in low-res space (clamped to texel centers)
    vec2 uv00 = (p00 + 0.5) / u_lowres_size;
    vec2 uv10 = (p00 + vec2(1.0, 0.0) + 0.5) / u_lowres_size;
    vec2 uv01 = (p00 + vec2(0.0, 1.0) + 0.5) / u_lowres_size;
    vec2 uv11 = (p00 + vec2(1.0, 1.0) + 0.5) / u_lowres_size;

    vec4 t00 = texture(u_lowres_trans, uv00);
    vec4 t10 = texture(u_lowres_trans, uv10);
    vec4 t01 = texture(u_lowres_trans, uv01);
    vec4 t11 = texture(u_lowres_trans, uv11);

    // Fetch corresponding depths sampled at the 4 low-res centers from high-res depth texture
    float d00 = linearize_depth(texture(u_highres_depth, uv00).r);
    float d10 = linearize_depth(texture(u_highres_depth, uv10).r);
    float d01 = linearize_depth(texture(u_highres_depth, uv01).r);
    float d11 = linearize_depth(texture(u_highres_depth, uv11).r);

    if (u_vrs_enabled) {
        bool is_atmo_edge = (t00.a > 0.0 || t10.a > 0.0 || t01.a > 0.0 || t11.a > 0.0) &&
                            (t00.a == 0.0 || t10.a == 0.0 || t01.a == 0.0 || t11.a == 0.0);
                            
        float max_d = max(max(d00, d10), max(d01, d11));
        float min_d = min(min(d00, d10), min(d01, d11));
        float depth_tol_edge = max(high_depth * 0.05, 0.0001);
        bool is_depth_edge = (max_d - min_d) > depth_tol_edge;
        
        if (is_atmo_edge || is_depth_edge) {
            out_scattered = vec4(0.0);
            out_transmittance = vec4(1.0);
            return;
        }
    }

    // Bilateral weights: combining standard bilinear weights with depth difference penalty
    float depth_tol = max(high_depth * 0.05, 0.0001);

    float dw00 = exp(-abs(d00 - high_depth) / depth_tol);
    float dw10 = exp(-abs(d10 - high_depth) / depth_tol);
    float dw01 = exp(-abs(d01 - high_depth) / depth_tol);
    float dw11 = exp(-abs(d11 - high_depth) / depth_tol);

    float w00 = (1.0 - f.x) * (1.0 - f.y) * dw00 * step(0.001, t00.a);
    float w10 = f.x * (1.0 - f.y) * dw10 * step(0.001, t10.a);
    float w01 = (1.0 - f.x) * f.y * dw01 * step(0.001, t01.a);
    float w11 = f.x * f.y * dw11 * step(0.001, t11.a);

    float sum_w = w00 + w10 + w01 + w11;

    if (sum_w < 1e-4) {
        // Nearest depth fallback to prevent halos when crossing sharp depth discontinuities
        float m00 = (t00.a > 0.001) ? abs(d00 - high_depth) : 1e15;
        float m10 = (t10.a > 0.001) ? abs(d10 - high_depth) : 1e15;
        float m01 = (t01.a > 0.001) ? abs(d01 - high_depth) : 1e15;
        float m11 = (t11.a > 0.001) ? abs(d11 - high_depth) : 1e15;
        float min_m = min(min(m00, m10), min(m01, m11));
        if (min_m < 1e14) {
            w00 = (m00 == min_m) ? 1.0 : 0.0;
            w10 = (m10 == min_m) ? 1.0 : 0.0;
            w01 = (m01 == min_m) ? 1.0 : 0.0;
            w11 = (m11 == min_m) ? 1.0 : 0.0;
            sum_w = w00 + w10 + w01 + w11;
        } else {
            out_scattered = vec4(0.0);
            out_transmittance = vec4(1.0);
            return;
        }
    }

    w00 /= sum_w;
    w10 /= sum_w;
    w01 /= sum_w;
    w11 /= sum_w;

    vec4 s00 = texture(u_lowres_scatter, uv00);
    vec4 s10 = texture(u_lowres_scatter, uv10);
    vec4 s01 = texture(u_lowres_scatter, uv01);
    vec4 s11 = texture(u_lowres_scatter, uv11);



    // If an unrendered texel was sampled (alpha == 0), its transmittance is 1.0 (identity)
    if (t00.a < 0.001) t00 = vec4(1.0);
    if (t10.a < 0.001) t10 = vec4(1.0);
    if (t01.a < 0.001) t01 = vec4(1.0);
    if (t11.a < 0.001) t11 = vec4(1.0);

    vec4 base_scatter = s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11;

    // Cross-bilateral low-res spatial filter (Solution B):
    // Cleans residual dither checkerboard while strictly preserving silhouettes
    vec4 filt_scatter = base_scatter * 0.4;
    float filt_w = 0.4;

    vec2 step_uv = 1.0 / u_lowres_size;
    vec2 cross_offsets[4] = vec2[](
        vec2(-step_uv.x, 0.0), vec2(step_uv.x, 0.0),
        vec2(0.0, -step_uv.y), vec2(0.0, step_uv.y)
    );

    for (int k = 0; k < 4; k++) {
        vec2 tap_uv = v_texcoord + cross_offsets[k];
        vec4 tap_trans = texture(u_lowres_trans, tap_uv);
        if (tap_trans.a > 0.001) {
            float tap_depth = linearize_depth(texture(u_highres_depth, tap_uv).r);
            float d_diff = abs(tap_depth - high_depth);
            float dw = exp(-d_diff / depth_tol);
            float w = 0.15 * dw;
            filt_scatter += texture(u_lowres_scatter, tap_uv) * w;
            filt_w += w;
        }
    }

    out_scattered = (filt_w > 1e-4) ? (filt_scatter / filt_w) : base_scatter;
    out_transmittance = t00 * w00 + t10 * w10 + t01 * w01 + t11 * w11;
}
