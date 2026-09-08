#version 460 core
// GAIA starfield point pass — per-star positions are computed ON THE GPU in
// double precision: p(t) = pos0 + pm*t (heliocentric, static attributes), then
// made frame-relative by subtracting the dvec3 u_origin uniform. No per-frame
// CPU packing / VBO streaming; only two uniform writes per frame.

in dvec3 in_pos0;      // heliocentric position, render frame, AU (f8, static)
in vec3 in_pm;         // proper-motion velocity, AU/yr (f4, static)
in vec3 in_color;      // blackbody RGB from BP-RP (0-1)
in float in_flux;      // pre-scaled intrinsic flux: 10^(-0.4*M_G) * FLUX_SCALE

uniform dvec3 u_origin;        // frame origin: tracked body's barycentric pos (AU)
uniform double u_t;            // simulation time, years (p(t) = pos0 + pm*t)

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
uniform float u_intensity_scale; // user brightness tuning (shared with .frag)
uniform bool u_hdr_enabled;      // shared with .frag — gates u_exposure
uniform float u_exposure;        // shared with .frag — camera exposure

#include "common/refraction.glsl"

out vec3 f_color;
out float f_intensity;
out float f_point_size;

void main() {
    // Double-precision origin subtraction: heliocentric positions reach
    // thousands of parsecs while the eye can sit light-seconds from a star,
    // so the f8 math here is what prevents catastrophic cancellation (the
    // same reason the old CPU packer used f8 before downcasting to f4).
    vec3 pos = vec3(in_pos0 + dvec3(in_pm) * u_t - u_origin);

    // Physical inverse-square flux falloff against the EYE (not the pack
    // frame origin): I = flux / d^2 — the same 1/d^2 law the system star
    // obeys, so GAIA stars brighten on approach and dim on recession.
    // The 1 AU floor saturates at contact instead of diverging (GAIA stars
    // have no mesh LOD to take over at close range). Evaluated on the TRUE
    // position — refraction only bends the apparent direction, it does not
    // change the flux (extinction is handled by the atmosphere pass).
    vec3 rel_eye = pos - u_eye;
    float dist_eye = length(rel_eye);
    float d2 = max(dist_eye * dist_eye, 1.0);
    float intensity = in_flux * u_flux_calib / d2;

    // Atmospheric refraction & gravitational lensing: lift/deflect the apparent
    // position of stars. Clamping distant stars to just inside the far plane
    // relative to the eye BEFORE refraction prevents float32 catastrophic
    // cancellation when catalog stars sit at parsec-scale distances (10^7 AU).
    float max_dist = u_far * 0.998;
    float dist_clamped = min(dist_eye, max_dist);
    vec3 dir_eye = dist_eye > 1e-9 ? (rel_eye / dist_eye) : vec3(0.0, 0.0, -1.0);
    pos = u_eye + dir_eye * dist_clamped;

    pos = apply_refraction(pos, u_eye);

    vec4 clip = projection * view * vec4(pos, 1.0);
    // Points have no near-plane culling — degenerate any behind-camera vertex.
    if (clip.w <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        f_color = vec3(0.0);
        f_intensity = 0.0;
        return;
    }

    // Fixed PSF footprint: every star shares the same sprite size, so the
    // peak pixel radiance is directly proportional to flux — preserving the
    // true astronomical magnitude scale under one shared exposure (the old
    // I^0.25 size law compressed dynamic range by sqrt(I) via flux
    // conservation, making m~20 stars visible at daylight exposures).
    // Bloom supplies the apparent size growth for the brightest stars.
    float size = u_point_base * (screen_height / 1080.0);
    gl_PointSize = clamp(size, u_min_point_px, u_max_point_px);

    // --- Vertex-stage culling (frustum + exposure threshold) -------------
    // Done here rather than in a GPU culling compute pass ON PURPOSE:
    // `pos` above is already the APPARENT (refracted) position, so testing it
    // against the frustum is exact — refraction lifting a star over a
    // planetary limb into view keeps it visible, and a star refracted out of
    // frame is culled because that is precisely where it is rendered. A
    // compute-pass design would have to test TRUE positions and widen the
    // frustum by tan(max_bend) (as culling.comp does for bodies), which is
    // conservative and duplicates the Newton-Raphson refraction solve.
    // The vertex cost for the full catalog is trivial ALU; the real savings
    // is skipping rasterization of off-screen / sub-threshold sprites.

    // 1. Frustum test with sprite-extent padding: the point sprite extends
    //    up to u_max_point_px/2 pixels past the center, so pad the NDC test
    //    by that extent. Padding is computed against screen height (1 px =
    //    2/H NDC), which is conservative in X on wide aspect ratios — it can
    //    only under-cull, never pop a visible star.
    float pad_ndc = u_max_point_px / max(screen_height, 1.0) * 2.0;
    vec2 ndc = clip.xy / clip.w;
    bool on_screen = all(lessThanEqual(abs(ndc), vec2(1.0 + pad_ndc)));

    // 2. Exposure-threshold culling: mirror starfield.frag exactly. The
    //    fragment pass fades to zero below a peak signal of 1.0e-6
    //    (smoothstep(1.0e-6, 2.5e-6, ...)), so any star whose CENTER-pixel
    //    signal (falloff peak = 1) cannot reach that floor contributes
    //    literally nothing — skip it entirely. Includes the star's color
    //    (max channel) since the frag multiplies per channel.
    float flux_norm = 1.0 / (gl_PointSize * gl_PointSize * 0.2524);
    float eff_exposure = u_hdr_enabled ? u_exposure : 1.0;
    float star_peak = max(in_color.r, max(in_color.g, in_color.b));
    float peak_signal = intensity * u_intensity_scale * star_peak
                        * flux_norm * eff_exposure;

    if (!on_screen || peak_signal < 1.0e-6) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        f_color = vec3(0.0);
        f_intensity = 0.0;
        f_point_size = gl_PointSize;
        return;
    }

    gl_Position = clip;

    f_color = in_color;
    f_intensity = intensity;
    f_point_size = gl_PointSize;
}
