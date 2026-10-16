#version 460

struct BodyInstanceGPU {
    vec4 pos_radius;
    vec4 color_is_star;
    vec4 pole_oblateness;
    vec4 pole_color_lum;
    vec4 spin_flags;
};

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

layout(std430, set = 0, binding = 1) readonly buffer BodyInstanceBuffer {
    BodyInstanceGPU bodies[];
};

layout(location = 0) out vec2 f_quad_uv;
layout(location = 1) flat out float f_apparent_px;
layout(location = 2) flat out float f_half_size_px;
layout(location = 3) flat out float f_is_star;
layout(location = 4) flat out vec3 f_surface_color;

const vec2 QUAD_VERTS[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
);

void main() {
    uint inst_idx = gl_InstanceIndex;
    BodyInstanceGPU b = bodies[inst_idx];

    vec3 center = b.pos_radius.xyz;
    float radius = b.pos_radius.w;
    float is_star = b.color_is_star.w;
    bool is_bh = b.spin_flags.z > 0.5;

    if (is_bh) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        return;
    }

    float dist = length(center);
    float screen_w = u_scene.screen_params.x;
    float screen_h = u_scene.screen_params.y;
    float fov_factor = u_scene.screen_params.z;
    float apparent_px = (radius / max(dist, 1e-12)) * screen_h * fov_factor;

    if (apparent_px >= 3.0) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        return;
    }

    vec4 center_clip = u_scene.view_proj * vec4(center, 1.0);
    if (center_clip.w <= 1e-6) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        return;
    }

    float r_min = (is_star > 0.5) ? 1.0 : 0.5;
    float r_mesh_px = apparent_px * 0.5;
    float r_eff = max(r_mesh_px, r_min);
    float half_size_px = r_eff + 1.0;

    vec2 q_pos = QUAD_VERTS[gl_VertexIndex];
    vec2 ndc_offset = q_pos * ((half_size_px * 2.0) / vec2(screen_w, screen_h));

    vec4 clip = center_clip;
    clip.xy += ndc_offset * center_clip.w;
    gl_Position = clip;

    f_quad_uv = q_pos * half_size_px;
    f_apparent_px = apparent_px;
    f_half_size_px = half_size_px;
    f_is_star = is_star;

    vec3 surf_col = vec3(0.0);
    if (is_star > 0.5) {
        float lum_eq = max(b.pole_color_lum.w, 1e-4);
        float sr = max(radius, 1e-6);
        float surf_lum = (u_scene.hdr_enabled != 0u) ? (0.8333 * lum_eq / (sr * sr)) : 1.0;
        surf_col = b.color_is_star.rgb * surf_lum;
    } else {
        vec3 V_dir = (dist > 1e-6) ? normalize(-center) : vec3(0.0, 0.0, 1.0);
        for (uint s = 0; s < u_scene.num_stars; ++s) {
            vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
            vec3 star_col = u_scene.stars_color_lum[s].rgb;
            float star_lum = u_scene.stars_color_lum[s].w;
            vec3 to_star = star_pos - center;
            float d_au = max(length(to_star), 1e-5);
            vec3 L_dir = to_star / d_au;
            float cos_alpha = clamp(dot(L_dir, V_dir), -1.0, 1.0);
            float alpha = acos(cos_alpha);
            float phase = ((sin(alpha) + (3.14159265 - alpha) * cos_alpha) / 3.14159265);
            float irradiance = (u_scene.hdr_enabled != 0u) ? (star_lum / (d_au * d_au)) : 1.0;
            surf_col += b.color_is_star.rgb * star_col * irradiance * max(phase, 0.08);
        }
    }
    f_surface_color = surf_col;
}
