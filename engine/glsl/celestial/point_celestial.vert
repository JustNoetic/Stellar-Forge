#version 460 core
in vec2 in_position; // Unit quad [-1, 1]

layout(std430, binding = 2) buffer AllInstances {
    vec4 instances[];
};

layout(std430, binding = 3) buffer VisibleIndices {
    uint vis_indices[];
};

#define MAX_CASTERS 64
#define MAX_STARS 16
layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
    vec4 u_caster_ozone[MAX_CASTERS];
};

uniform float screen_height;
uniform float fov_factor;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform vec3 u_camera_pos;

#include "common/refraction.glsl"

out vec3 f_color;
out float f_is_star;
out float f_clip_z;
flat out vec2 f_center_px;
flat out float f_apparent_px;
flat out float f_half_size_px;
flat out vec3 f_surface_color;

#define PI 3.14159265358979323846

void main() {
    uint inst_idx = vis_indices[gl_InstanceID];
    vec4 f0 = instances[inst_idx * 7 + 0];
    vec4 f1 = instances[inst_idx * 7 + 1];
    vec4 f2 = instances[inst_idx * 7 + 2];
    vec4 f3 = instances[inst_idx * 7 + 3];
    vec4 f4 = instances[inst_idx * 7 + 4];
    vec4 f5 = instances[inst_idx * 7 + 5];

    vec3 in_offset = f0.xyz;
    vec3 in_color = vec3(f0.w, f1.x, f1.y);
    float in_radius = f1.z;
    float in_is_star = f2.x;
    float lum_eq = f4.w;

    float aspect = projection[1][1] / max(1e-6, projection[0][0]);
    float screen_width = screen_height * aspect;

    vec3 app_world_pos = apply_refraction(in_offset, u_camera_pos);

    vec4 center_clip = projection * view * vec4(app_world_pos, 1.0);
    if (center_clip.w <= 1e-6) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    vec2 center_ndc = center_clip.xy / center_clip.w;
    vec2 center_px = (center_ndc * 0.5 + 0.5) * vec2(screen_width, screen_height);
    f_center_px = center_px;

    float dist = length((view * vec4(app_world_pos, 1.0)).xyz);
    float apparent_px = (in_radius / max(1e-9, dist)) * screen_height * fov_factor;
    f_apparent_px = apparent_px;

    // Physical mesh radius in screen pixels (apparent_px is diameter)
    float R_mesh_px = apparent_px * 0.5;
    float R_eff = max(R_mesh_px, 0.5);
    float quad_radius_px = R_eff + 1.0;
    float quad_size_px = quad_radius_px * 2.0;
    float half_size_px = quad_radius_px;
    f_half_size_px = half_size_px;

    // Offset quad vertices in NDC
    vec2 ndc_offset = in_position * (quad_size_px / vec2(screen_width, screen_height));
    gl_Position = center_clip;
    gl_Position.xy += ndc_offset * center_clip.w;
    
    // Position depth at the body's front surface so it doesn't fail depth testing against its own 3D mesh
    // when cross-fading in the transition zone. Must match clamped mesh size!
    float clamped_min_px = 3.0;
    float depth_radius = in_radius;
    if (apparent_px > 1e-6 && apparent_px < clamped_min_px) {
        depth_radius = in_radius * (clamped_min_px / apparent_px);
    }
    f_clip_z = max(1e-4, center_clip.w - depth_radius * 1.05);

    f_color = in_color;
    f_is_star = in_is_star;

    // --- Surface Luminance Evaluation (Matches 3D Mesh PBR Radiance) ---
    vec3 surface_color = vec3(0.0);

    if (in_is_star > 0.5) {
        float star_lum = max(lum_eq, 1e-4);
        float star_r = max(in_radius, 1e-6);
        // 0.8333 matches the exact analytical disk-integrated average of the quadratic limb darkening
        // (c1=0.4, c2=0.2) evaluated by the 3D star mesh, ensuring seamless bloom and radiance hand-off
        float surface_luminance = u_hdr_enabled ? (0.8333 * star_lum / (star_r * star_r)) : 1.0;
        surface_color = in_color * surface_luminance;
    } else {
        // Primary star reflection
        vec3 star_pos = u_stars_pos_radius[0].xyz;
        float star_lum = max(u_stars_colors[0].w, 1.0);
        vec3 star_col = u_stars_colors[0].rgb;

        vec3 L = star_pos - in_offset;
        float dist_star = length(L);
        vec3 L_dir = (dist_star > 1e-6) ? (L / dist_star) : vec3(0.0, 1.0, 0.0);
        vec3 V_dir = (dist > 1e-6) ? normalize(u_camera_pos - in_offset) : vec3(0.0, 0.0, 1.0);

        // Lambertian sphere phase integral
        float phase_cos = clamp(dot(V_dir, L_dir), -1.0, 1.0);
        float phase_sin = sqrt(max(0.0, 1.0 - phase_cos * phase_cos));
        float phase_angle = acos(phase_cos);
        float phase_func = max(0.0, (phase_sin + (PI - phase_angle) * phase_cos) / PI);

        float star_irradiance = star_lum / max(dist_star * dist_star, 1e-6);

        // Geometric albedo: use true Top-Of-Atmosphere (TOA) color if body has an atmosphere
        vec3 albedo = in_color;
        for (int j = 0; j < u_num_casters; j++) {
            if (distance(u_casters[j].xyz, in_offset) < 1e-4) {
                if (u_caster_atmos[j].w > 0.0) {
                    albedo = u_caster_colors[j].xyz;
                }
                break;
            }
        }
        surface_color = star_col * albedo * (star_irradiance * phase_func);

        // Secondary planetshine contribution if nearby
        vec3 planetshine_color = f5.xyz;
        if (dot(planetshine_color, planetshine_color) > 1e-12) {
            surface_color += planetshine_color * 0.5;
        }
    }

    // Note: Atmospheric extinction is intentionally NOT applied here because the subsequent
    // atmosphere raymarch pass (render_atmosphere_pass) already attenuates all background buffer
    // pixels using physical transmittance and dual-source blending (GL_SRC1_COLOR).
    // Applying it here would square the optical depth (T -> T^2), artificially crushing sunset bodies.

    f_surface_color = surface_color;
}
