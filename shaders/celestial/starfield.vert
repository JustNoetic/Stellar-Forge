#version 460

struct StarGpuRecord {
    dvec4 pos0;
    vec4 pm_flux;
    vec4 color;
    vec4 reserved;
};

layout(std430, set = 0, binding = 0) readonly buffer StarCatalogBuffer {
    StarGpuRecord stars[];
};

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

layout(location = 0) out vec3 f_color;
layout(location = 1) out float f_intensity;
layout(location = 2) out float f_point_size;

void main() {
    StarGpuRecord s = stars[gl_VertexIndex];

    vec3 rel_eye = vec3(s.pos0.xyz + dvec3(s.pm_flux.xyz) * pc.sim_time_yr - pc.eye_world_au.xyz);
    float dist_eye = length(rel_eye);
    float d2 = max(dist_eye * dist_eye, 1.0);
    float intensity = s.pm_flux.w * pc.flux_calib / d2;

    vec3 dir_eye = dist_eye > 1e-9 ? (rel_eye / dist_eye) : vec3(0.0, 0.0, -1.0);
    vec4 clip = pc.view_proj * vec4(dir_eye, 1.0);

    if (clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        gl_PointSize = 0.0;
        f_color = vec3(0.0);
        f_intensity = 0.0;
        f_point_size = 0.0;
        return;
    }

    float size = clamp(pc.point_base_px * (pc.screen_height / 1080.0), pc.min_point_px, pc.max_point_px);
    gl_PointSize = size;

    float pad_ndc = pc.max_point_px / max(pc.screen_height, 1.0) * 2.0;
    vec2 ndc = clip.xy / clip.w;
    bool on_screen = all(lessThanEqual(abs(ndc), vec2(1.0 + pad_ndc)));

    float flux_norm = 1.0 / (size * size * 0.2524);
    float eff_exposure = (pc.hdr_enabled != 0u) ? pc.exposure : 1.0;
    float star_peak = max(s.color.r, max(s.color.g, s.color.b));
    float peak_signal = intensity * pc.intensity_scale * star_peak * flux_norm * eff_exposure;

    if (!on_screen || peak_signal < 1.0e-6) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        gl_PointSize = 0.0;
        f_color = vec3(0.0);
        f_intensity = 0.0;
        f_point_size = 0.0;
        return;
    }

    clip.z = 1e-6 * clip.w;
    gl_Position = clip;
    f_color = s.color.rgb;
    f_intensity = intensity;
    f_point_size = size;
}
