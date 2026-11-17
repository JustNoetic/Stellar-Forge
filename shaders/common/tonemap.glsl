#include "common/procedural_noise.glsl"

vec3 aces_tonemap(vec3 x) {
    x = min(x, vec3(65504.0));
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 linear_to_srgb(vec3 lin) {
    vec3 lo = lin * 12.92;
    vec3 hi = 1.055 * pow(max(lin, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, greaterThan(lin, vec3(0.0031308)));
}

vec3 dither_srgb(vec3 srgb, vec2 pixel) {
    float u = interleaved_gradient_noise(pixel);
    float t = u * 2.0 - 1.0;
    float tri = sign(t) * (1.0 - sqrt(max(1.0 - abs(t), 0.0)));
    vec3 clamped = clamp(srgb, 0.0, 1.0);
    vec3 offset = mix(vec3(u - 0.5), vec3(tri), clamp(clamped * 255.0 - 0.5, 0.0, 1.0));
    vec3 dithered = clamp(clamped + offset / 255.0, 0.0, 1.0);
    dithered = mix(vec3(0.0), dithered, greaterThan(clamped, vec3(0.0)));
    return mix(vec3(1.0), dithered, lessThan(clamped, vec3(1.0)));
}
