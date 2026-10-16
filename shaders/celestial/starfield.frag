#version 460

layout(location = 0) in vec3 f_color;
layout(location = 1) in float f_intensity;
layout(location = 2) in float f_point_size;

layout(push_constant) uniform StarfieldPush {
    mat4 view_proj;
    dvec4 eye_world_au;
    double sim_time_yr;
    float flux_calib;
    float screen_height;
    float point_base_px;
    float min_point_px;
    float max_point_px;
    float intensity_scale;
    float exposure;
    uint hdr_enabled;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) {
        discard;
    }

    float r = sqrt(r2);
    float falloff = exp(-r2 * 3.5) * (1.0 - smoothstep(0.6, 1.0, r));

    const float FALLOFF_MEAN = 0.2524;
    float flux_norm = 1.0 / (f_point_size * f_point_size * FALLOFF_MEAN);

    vec3 rgb = f_color * (f_intensity * pc.intensity_scale * falloff * flux_norm);
    if (pc.hdr_enabled != 0u) {
        rgb *= pc.exposure;
    }

    float peak_signal = max(rgb.r, max(rgb.g, rgb.b));
    float fade = smoothstep(1.0e-6, 2.5e-6, peak_signal);
    if (fade <= 0.0) {
        discard;
    }

    out_color = vec4(rgb * fade, 0.0);
}
