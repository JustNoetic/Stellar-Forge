#version 460 core
// GAIA starfield point pass — camera-relative star positions packed per frame
// by engine/rendering/star_catalog.py (interleaved f4: pos.xyz, rgb, intensity).

in vec3 in_pos;        // frame-relative position (render frame), AU
in vec3 in_color;      // blackbody RGB from BP-RP (0-1)
in float in_intensity; // HDR radiance: 10^(-0.4*M_G) / d_au^2 * FLUX_SCALE

uniform mat4 projection;
uniform mat4 view;
uniform float screen_height;   // framebuffer height (HiDPI-aware sizing)
uniform float u_point_base;    // base point size in px at reference intensity
uniform float u_max_point_px;  // upper clamp (bloom carries the rest)
uniform float u_far;           // scene far plane (AU) — clamp distance for stars

out vec3 f_color;
out float f_intensity;

void main() {
    vec3 pos = in_pos;

    // Clamp distant stars to just inside the far plane: points have no
    // parallax at these distances, so this is visually exact and prevents
    // far-plane clipping of the genuinely distant catalog entries.
    float dist = length(pos);
    float max_dist = u_far * 0.998;
    if (dist > max_dist) {
        pos *= max_dist / dist;
    }

    vec4 clip = projection * view * vec4(pos, 1.0);
    // Points have no near-plane culling — degenerate any behind-camera vertex.
    if (clip.w <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        f_color = vec3(0.0);
        f_intensity = 0.0;
        return;
    }
    gl_Position = clip;

    // Brightness-driven size with a quarter-power response (intensity spans
    // ~1e-4 .. ~1e1); HiDPI-scaled by framebuffer height.
    float size = u_point_base * pow(max(in_intensity, 1e-9), 0.25)
               * (screen_height / 1080.0);
    gl_PointSize = clamp(size, 1.0, u_max_point_px);

    f_color = in_color;
    f_intensity = in_intensity;
}
