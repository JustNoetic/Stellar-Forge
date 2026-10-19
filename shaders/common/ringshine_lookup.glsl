layout(set = 0, binding = 16) uniform sampler2D u_ringshine_map;

vec3 ringshine_sample(float phi, float slat, int host_index, int star_index) {
    float u = 0.5 + 0.5 * sign(phi) * pow(min(abs(phi) / 3.14159265358979323846, 1.0), 0.666666667);
    float v = 0.5 + 0.5 * sign(slat) * pow(min(abs(slat), 1.0), 0.666666667);
    vec2 local = (0.5 + clamp(vec2(u, v), 0.0, 1.0) * vec2(127.0, 64.0)) / vec2(128.0, 65.0);
    return textureLod(u_ringshine_map, (vec2(float(star_index), float(host_index)) + local)
        / vec2(16.0, 16.0), 0.0).rgb;
}
