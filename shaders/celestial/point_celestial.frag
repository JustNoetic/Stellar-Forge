#version 460

layout(location = 0) in vec2 f_quad_uv;
layout(location = 1) flat in float f_apparent_px;
layout(location = 2) flat in float f_half_size_px;
layout(location = 3) flat in float f_is_star;
layout(location = 4) flat in vec3 f_surface_color;

layout(std140, set = 0, binding = 0) uniform SceneCelestialUBO {
    mat4 view_proj;
    mat4 view;
    vec4 stars_pos_radius[16];
    vec4 stars_color_lum[16];
    vec4 screen_params;
    uint num_stars;
    uint num_bodies;
    uint hdr_enabled;
    uint pad0;
} u_scene;

layout(location = 0) out vec4 out_color;

void main() {
    float dist_px = length(f_quad_uv);
    if (dist_px > f_half_size_px) {
        discard;
    }

    float r_min = (f_is_star > 0.5) ? 1.0 : 0.5;
    float r_mesh_px = f_apparent_px * 0.5;
    float r_eff = max(r_mesh_px, r_min);

    float aa_half_width = 0.35;
    float disk_coverage = clamp(0.5 - (dist_px - r_eff) / (aa_half_width * 2.0), 0.0, 1.0);
    disk_coverage = smoothstep(0.0, 1.0, disk_coverage);
    if (disk_coverage <= 0.0) {
        discard;
    }

    float r_ratio = clamp(r_mesh_px / r_min, 0.0, 1.0);
    float flux_scale = (r_mesh_px < r_min) ? pow(r_ratio, (f_is_star > 0.5) ? 2.0 : 1.3) : 1.0;
    vec3 rgb = f_surface_color * (disk_coverage * flux_scale);

    if (u_scene.hdr_enabled != 0u) {
        rgb *= u_scene.screen_params.w;
    }

    float peak_signal = max(rgb.r, max(rgb.g, rgb.b));
    float fade = smoothstep(1.0e-6, 2.5e-6, peak_signal);
    if (fade <= 0.0) {
        discard;
    }

    float transition_weight = 1.0 - smoothstep(2.0, 3.0, f_apparent_px);
    rgb *= fade * transition_weight;

    out_color = vec4(rgb, 0.0);
}
