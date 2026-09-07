#version 460 core
// GAIA starfield point pass — fragment stage.
// Soft circular HDR sprite; additive blending, depth-test on / write-off
// (opaque spheres occlude stars; atmospheres blend over them afterwards).

in vec3 f_color;
in float f_intensity;

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

    vec3 rgb = f_color * (f_intensity * u_intensity_scale * falloff);

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
