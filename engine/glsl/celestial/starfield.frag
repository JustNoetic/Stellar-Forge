#version 460 core
// GAIA starfield point pass — fragment stage.
// Soft circular HDR sprite; additive blending, depth-test on / write-off
// (opaque spheres occlude stars; atmospheres blend over them afterwards).

in vec3 f_color;
in float f_intensity;
in float f_point_size;

uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform float u_intensity_scale;   // user brightness tuning

out vec4 out_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;

    // Gaussian core with a smooth edge so magnitudes roll off cleanly.
    float r = sqrt(r2);
    float falloff = exp(-r2 * 3.5) * (1.0 - smoothstep(0.6, 1.0, r));

    // Flux-conserving normalization (energy conservation for an unresolved
    // source): the sprite's total integrated energy equals f_intensity
    // regardless of point size. FALLOFF_MEAN is the profile's average over
    // the unit disk (= 2 * integral_0^1 falloff(r) * r dr ~ 0.2524), so
    // dividing by size^2 * FALLOFF_MEAN keeps total flux constant while the
    // peak scales with 1/size^2 — bigger sprites spread the same light over
    // more pixels (surface-brightness preservation). This removes the old
    // non-physical size^2 total-flux growth and, together with the >= 2 px
    // sprite floor, stabilizes subpixel sampling to kill shimmer.
    const float FALLOFF_MEAN = 0.2524;
    float flux_norm = 1.0 / (f_point_size * f_point_size * FALLOFF_MEAN);

    vec3 rgb = f_color * (f_intensity * u_intensity_scale * falloff * flux_norm);

    // Camera exposure scaling — mirrors point_celestial.frag behavior
    if (u_hdr_enabled) {
        rgb *= u_exposure;
    }

    // Limiting-magnitude extinction: fade below the sensor detection floor
    float peak_signal = max(rgb.r, max(rgb.g, rgb.b));
    float fade = smoothstep(1.0e-6, 2.5e-6, peak_signal);
    if (fade <= 0.0) discard;

    // Constant depth just inside the far plane. The depth buffer holds
    // logarithmic depths from the sphere pass; planets (log depth << 1)
    // occlude stars, while the cleared far depth (1.0) lets them through.
    // Fixed-function depth would round to exactly 1.0 at these distances
    // and fail the LESS test against the cleared buffer.
    gl_FragDepth = 0.999999;

    out_color = vec4(rgb * fade, 0.0);  // alpha unused (additive blend)
}
