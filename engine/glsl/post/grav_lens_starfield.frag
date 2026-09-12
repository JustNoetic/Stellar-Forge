#version 460 core
// Cubemap-Based Gravitational Lensing for the GAIA Starfield.
// Evaluates backward ray deflection over the full 360-degree (4*pi steradian) celestial sphere.
//
// Features:
// 1. Physically continuous curved Einstein arcs and full closed Einstein rings.
// 2. Full 360-deg sky coverage via cubemap: captures off-screen stars from any angle
//    (including behind the observer) deflected onto the screen.
// 3. Natural secondary mirror images inside theta_E by sampling across the optical axis.
// 4. Kerr spin frame-dragging (Lense-Thirring) without coordinate blowups.
// 5. Black hole shadow silhouette occlusion and depth-buffer integration.
// 6. Uses the unified Gralla-Lupsasca (2020) Kerr deflection kernel from common/refraction.glsl.

in vec2 v_texcoord;

uniform sampler2D u_starfield_tex;       // High-res 1:1 screen-space starfield render
uniform samplerCube u_starfield_cubemap; // Full 360-degree celestial cubemap
uniform bool u_has_starfield_tex;        // Whether 1:1 screen-space texture is bound

uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform vec2 u_tan_half_fov;            // Screen tan half FOV (tan(fov_x/2), tan(fov_y/2))
uniform vec3 u_camera_pos;              // AU (camera position in world render frame)
uniform float u_screen_width;
uniform float u_screen_height;

#include "common/refraction.glsl"

out vec4 out_color;

void main() {
    // Normalized Device Coordinates on the screen [-1, 1]
    vec2 ndc = v_texcoord * 2.0 - 1.0;

    // Reconstruct world-space ray direction V for this screen pixel:
    vec3 V = normalize(u_cam_forward + ndc.x * u_tan_half_fov.x * u_cam_right + ndc.y * u_tan_half_fov.y * u_cam_up);

    // If gravitational lensing is not active or has negligible Schwarzschild radius:
    if (!u_grav_lens_enabled || u_grav_lens_rs <= 1e-6) {
        vec4 base_color = u_has_starfield_tex 
            ? texelFetch(u_starfield_tex, ivec2(gl_FragCoord.xy), 0)
            : texture(u_starfield_cubemap, V);
        if (dot(base_color.rgb, base_color.rgb) <= 1e-12) discard;
        out_color = base_color;
        gl_FragDepth = 0.999999;
        return;
    }

    // Geometry relative to lens center (in AU)
    vec3 cam_to_lens = u_grav_lens_center - u_camera_pos;
    float d_l_au = length(cam_to_lens);
    if (d_l_au < 1e-6) {
        vec4 base_color = u_has_starfield_tex 
            ? texelFetch(u_starfield_tex, ivec2(gl_FragCoord.xy), 0)
            : texture(u_starfield_cubemap, V);
        if (dot(base_color.rgb, base_color.rgb) <= 1e-12) discard;
        out_color = base_color;
        gl_FragDepth = 0.999999;
        return;
    }

    // Lens-centric camera coordinates in km:
    vec3 C_km = (u_camera_pos - u_grav_lens_center) * u_au_to_km;

    // Evaluate backward ray deflection and black hole shadow capture:
    bool is_shadow = false;
    float alpha = compute_gravitational_deflection(C_km, V, 0.0, is_shadow);

    // Solid-body physical radius occlusion for non-BH lenses (stars, white dwarfs, neutron stars)
    if (u_grav_lens_type != 3 && u_grav_lens_radius > 0.0) {
        float s_min = -dot(C_km, V);
        vec3 P_min = C_km + s_min * V;
        float b_km = length(P_min);
        if (s_min > 0.0 && b_km < u_grav_lens_radius) {
            is_shadow = true;
        }
    }

    if (is_shadow) {
        // Ray plunging through the event horizon or into the stellar surface
        discard;
    }

    // If deflection is effectively zero, sample unlensed starfield at 1:1 pixel sharpness:
    if (alpha <= 1e-7) {
        vec4 base_color = u_has_starfield_tex 
            ? texelFetch(u_starfield_tex, ivec2(gl_FragCoord.xy), 0)
            : texture(u_starfield_cubemap, V);
        if (dot(base_color.rgb, base_color.rgb) <= 1e-12) discard;
        out_color = base_color;
        gl_FragDepth = 0.999999;
        return;
    }

    // Unit perpendicular direction from ray line towards the lens center:
    vec3 u_dir = C_km - V * dot(C_km, V);
    float u_len = length(u_dir);
    if (u_len > 1e-5) {
        u_dir /= u_len;
    } else {
        vec3 perp = abs(V.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        u_dir = normalize(cross(V, perp));
    }

    // Deflect the backward sightline TOWARDS the lens by angle alpha:
    vec3 V_deflected = normalize(V * cos(alpha) - u_dir * sin(alpha));

    // Kerr spin frame-dragging (Lense-Thirring):
    // Precesses the deflected sightline around the spin axis pole_n.
    if (abs(u_grav_lens_spin) > 1e-4) {
        vec3 pole_n = length(u_grav_lens_pole) > 1e-4 ? normalize(u_grav_lens_pole) : vec3(0.0, 1.0, 0.0);
        float s_min = -dot(C_km, V);
        float r_c = max(length(C_km), u_grav_lens_rs * 1.5);
        float cos_t = clamp(s_min / r_c, -1.0, 1.0);
        float b_eff;
        if (s_min > 0.0) {
            vec3 P_min = C_km + s_min * V;
            b_eff = max(length(P_min), u_grav_lens_rs * 1.5);
        } else {
            b_eff = r_c;
        }
        float drag_angle = clamp((u_grav_lens_spin * u_grav_lens_rs * u_grav_lens_rs * (1.0 + cos_t)) / (b_eff * b_eff), -0.75, 0.75);
        
        // Rodrigues rotation of V_deflected around pole_n by drag_angle:
        V_deflected = V_deflected * cos(drag_angle) 
                    + cross(pole_n, V_deflected) * sin(drag_angle) 
                    + pole_n * dot(pole_n, V_deflected) * (1.0 - cos(drag_angle));
        V_deflected = normalize(V_deflected);
    }

    // Check if deflected sightline points to an on-screen star:
    float v_z = dot(V_deflected, u_cam_forward);
    float v_x = dot(V_deflected, u_cam_right);
    float v_y = dot(V_deflected, u_cam_up);

    vec4 star_color = vec4(0.0);
    bool sampled_onscreen = false;

    // If deflected sightline points forward into the screen frustum, sample the high-res 1:1 screen texture:
    if (u_has_starfield_tex && v_z > 0.0) {
        vec2 ndc_def = vec2(v_x / (v_z * u_tan_half_fov.x), v_y / (v_z * u_tan_half_fov.y));
        if (abs(ndc_def.x) <= 1.0 && abs(ndc_def.y) <= 1.0) {
            vec2 uv_def = ndc_def * 0.5 + 0.5;
            star_color = texture(u_starfield_tex, uv_def);
            sampled_onscreen = true;
        }
    }

    // Strictly sample the 360-degree celestial cubemap ONLY for offscreen stars
    // (sightline deflected outside the screen frustum or behind the camera), or when no screen texture is provided:
    if (!sampled_onscreen) {
        star_color = texture(u_starfield_cubemap, V_deflected);
    }

    if (dot(star_color.rgb, star_color.rgb) <= 1e-12) {
        discard;
    }

    out_color = star_color;
    gl_FragDepth = 0.999999;
}
