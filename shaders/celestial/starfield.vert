#version 460

#include "common/point_source.glsl"
#include "common/refraction.glsl"

struct StarGpuRecord {
    dvec4 pos0;
    vec4 pm_flux;
    vec4 color;
    vec4 reserved;
};

layout(std430, set = 0, binding = 0) readonly buffer StarCatalogBuffer {
    StarGpuRecord stars[];
};

layout(std430, set = 0, binding = 1) readonly buffer VisibleIndexBuffer {
    uint visible_indices[];
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
    float screen_width;
    uint star_count;
} pc;

layout(location = 0) flat out vec3 f_color;
layout(location = 1) flat out float f_intensity;
layout(location = 2) flat out float f_point_radius;
layout(location = 3) out vec2 f_pixel_offset;

void main() {
    uint star_idx = visible_indices[gl_InstanceIndex];
    StarGpuRecord s = stars[star_idx];

    vec3 rel_eye = vec3(s.pos0.xyz + dvec3(s.pm_flux.xyz) * pc.sim_time_yr - pc.eye_world_au.xyz);
    float dist_eye = length(rel_eye);
    float d2 = max(dist_eye * dist_eye, 1.0);
    float intensity = s.pm_flux.w * pc.flux_calib / d2;

    vec3 dir_eye = dist_eye > 1e-9 ? (rel_eye / dist_eye) : vec3(0.0, 0.0, -1.0);
    vec4 apparent = refract_direction(dir_eye);
    dir_eye = apparent.xyz;
    vec4 clip = pc.view_proj * vec4(dir_eye, 1.0);

    if (clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        f_color = vec3(0.0);
        f_intensity = 0.0;
        f_point_radius = 1.0;
        f_pixel_offset = vec2(0.0);
        return;
    }

    float size = clamp(pc.point_base_px, pc.min_point_px, pc.max_point_px);
    float radius = 0.5 * size;
    vec2 pad_ndc = size / vec2(pc.screen_width, pc.screen_height);
    vec2 ndc = clip.xy / clip.w;
    bool on_screen = all(lessThanEqual(abs(ndc), vec2(1.0) + pad_ndc));

    float eff_exposure = (pc.hdr_enabled != 0u) ? pc.exposure : 1.0;
    intensity *= apparent.w;
    vec3 flux = s.color.rgb * (intensity * pc.intensity_scale * eff_exposure);
    float fade = point_signal_fade(flux, radius);

    if (!on_screen || fade <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        f_color = vec3(0.0);
        f_intensity = 0.0;
        f_point_radius = 1.0;
        f_pixel_offset = vec2(0.0);
        return;
    }

    vec2 center_px = (clip.xy / clip.w * 0.5 + 0.5) * vec2(pc.screen_width, pc.screen_height);
    float normalization = point_profile_sum(center_px, radius);
    clip.z = 1e-6 * clip.w;
    vec2 q = POINT_QUAD_VERTS[gl_VertexIndex];
    clip.xy += q * pad_ndc * clip.w;
    gl_Position = clip;
    f_color = s.color.rgb;
    f_intensity = intensity * pc.intensity_scale * eff_exposure * fade / normalization;
    f_point_radius = radius;
    f_pixel_offset = q * radius;
}
