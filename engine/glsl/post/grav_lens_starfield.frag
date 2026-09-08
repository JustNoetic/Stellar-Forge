#version 460 core
// Screen-Space Gravitational Lensing for the GAIA Starfield.
// Replaces the 2-pass forward point-primitive hack with an inverse-space
// backward ray deflection pass.
//
// Features:
// 1. Physically continuous curved Einstein arcs and full closed Einstein rings.
// 2. Dual-texture approach: sharp 1:1 unlensed stars on-screen, falling back
//    to a 360-degree equirectangular map for deflected/off-screen sightlines.
// 3. Natural secondary mirror images inside theta_E by sampling the opposite side.
// 4. Kerr spin frame-dragging (Lense-Thirring) without coordinate blowups.
// 5. Black hole shadow silhouette occlusion and depth-buffer integration.

in vec2 v_texcoord;

uniform sampler2D u_starfield_tex; // Screen-space 1:1 render
uniform sampler2D u_allsky_tex;    // 360-degree equirectangular render

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

    // If gravitational lensing is not active or has negligible Schwarzschild radius:
    if (!u_grav_lens_enabled || u_grav_lens_rs <= 1e-6) {
        vec4 base_color = texture(u_starfield_tex, v_texcoord);
        if (dot(base_color.rgb, base_color.rgb) <= 1e-12) discard;
        out_color = base_color;
        gl_FragDepth = 0.999999;
        return;
    }

    // Reconstruct world-space ray direction V for this screen pixel:
    vec3 V = normalize(u_cam_forward + ndc.x * u_tan_half_fov.x * u_cam_right + ndc.y * u_tan_half_fov.y * u_cam_up);

    // Geometry relative to lens center (in AU)
    vec3 cam_to_lens = u_grav_lens_center - u_camera_pos;
    float d_l_au = length(cam_to_lens);
    if (d_l_au < 1e-6) {
        vec4 base_color = texture(u_starfield_tex, v_texcoord);
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

    // If deflection is effectively zero, sample unlensed high-res starfield directly:
    if (alpha <= 1e-7) {
        vec4 base_color = texture(u_starfield_tex, v_texcoord);
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
    if (abs(u_grav_lens_spin) > 1e-4) {
        vec3 pole_n = length(u_grav_lens_pole) > 1e-4 ? normalize(u_grav_lens_pole) : vec3(0.0, 1.0, 0.0);
        float s_min = -dot(C_km, V);
        vec3 P_min = C_km + s_min * V;
        float b_km = max(length(P_min), u_grav_lens_rs * 1.5);
        
        float drag_angle = (u_grav_lens_spin * u_grav_lens_rs * u_grav_lens_rs * 2.0) / (b_km * b_km);
        drag_angle = clamp(drag_angle, -0.75, 0.75);
        
        V_deflected = V_deflected * cos(drag_angle) 
                    + cross(pole_n, V_deflected) * sin(drag_angle) 
                    + pole_n * dot(pole_n, V_deflected) * (1.0 - cos(drag_angle));
        V_deflected = normalize(V_deflected);
    }

    // Calculate deflected sightline in camera basis
    float v_x = dot(V_deflected, u_cam_right);
    float v_y = dot(V_deflected, u_cam_up);
    float v_z = dot(V_deflected, u_cam_forward);

    vec4 star_color = vec4(0.0);

    // If deflected sightline points forward (z > 0), try to sample the high-res viewport screen texture:
    if (v_z > 0.0) {
        vec2 ndc_deflected = vec2(v_x / (v_z * u_tan_half_fov.x), v_y / (v_z * u_tan_half_fov.y));
        if (abs(ndc_deflected.x) <= 1.0 && abs(ndc_deflected.y) <= 1.0) {
            vec2 uv_deflected = ndc_deflected * 0.5 + 0.5;
            star_color = texture(u_starfield_tex, uv_deflected);
        }
    }

    // If it was completely off-screen or backwards, sample the 360-degree equirectangular allsky map!
    if (dot(star_color.rgb, star_color.rgb) <= 1e-12) {
        float lon = atan(v_x, -v_z); // Note: -v_z because forward is -Z in OpenGL camera space
        float lat = asin(v_y);
        float u = lon / (2.0 * 3.14159265359) + 0.5;
        float v = lat / 3.14159265359 + 0.5;
        star_color = texture(u_allsky_tex, vec2(u, v));
    }

    if (dot(star_color.rgb, star_color.rgb) <= 1e-12) {
        discard;
    }

    out_color = star_color;
    gl_FragDepth = 0.999999;
}
