#version 460 core
// GAIA starfield point pass — camera-relative star positions packed per frame
// by engine/rendering/star_catalog.py (interleaved f4: pos.xyz, rgb, flux).

in vec3 in_pos;        // frame-relative position (render frame), AU
in vec3 in_color;      // blackbody RGB from BP-RP (0-1)
in float in_flux;      // pre-scaled intrinsic flux: 10^(-0.4*M_G) * FLUX_SCALE

uniform mat4 projection;
uniform mat4 view;
uniform vec3 u_eye;            // eye position relative to the pack frame origin (AU)
uniform float u_flux_calib;    // HDR calibration: maps catalog flux units onto the
                               // system star's radiance units (see app.py) so a single
                               // exposure treats planets, the sun, and GAIA stars alike
uniform float screen_height;   // framebuffer height (HiDPI-aware sizing)
uniform float u_point_base;    // base point size in px (user setting)
uniform float u_min_point_px;  // lower clamp — keeps sprites large enough to
                               // sample the PSF profile stably (anti-shimmer)
uniform float u_max_point_px;  // upper clamp (bloom carries the rest)
uniform float u_far;           // scene far plane (AU) — clamp distance for stars

out vec3 f_color;
out float f_intensity;
out float f_point_size;

void main() {
    vec3 pos = in_pos;

    // Physical inverse-square flux falloff against the EYE (not the pack
    // frame origin): I = flux / d^2 — the same 1/d^2 law the system star
    // obeys, so GAIA stars brighten on approach and dim on recession.
    // The 1 AU floor saturates at contact instead of diverging (GAIA stars
    // have no mesh LOD to take over at close range).
    vec3 rel_eye = pos - u_eye;
    float dist_eye = length(rel_eye);
    float d2 = max(dist_eye * dist_eye, 1.0);
    float intensity = in_flux * u_flux_calib / d2;

    // Clamp distant stars to just inside the far plane, relative to the EYE:
    // points have no parallax at these distances, so this is visually exact
    // and prevents far-plane clipping of the distant catalog entries. The
    // clamped position preserves the eye->star direction, and the intensity
    // above uses the unclamped distance so brightness stays continuous
    // across the clamp boundary (no popping).
    float max_dist = u_far * 0.998;
    if (dist_eye > max_dist) {
        pos = u_eye + rel_eye * (max_dist / dist_eye);
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

    // Fixed PSF footprint: every star shares the same sprite size, so the
    // peak pixel radiance is directly proportional to flux — preserving the
    // true astronomical magnitude scale under one shared exposure (the old
    // I^0.25 size law compressed dynamic range by sqrt(I) via flux
    // conservation, making m~20 stars visible at daylight exposures).
    // Bloom supplies the apparent size growth for the brightest stars.
    float size = u_point_base * (screen_height / 1080.0);
    gl_PointSize = clamp(size, u_min_point_px, u_max_point_px);

    f_color = in_color;
    f_intensity = intensity;
    f_point_size = gl_PointSize;
}
