#ifndef SF_TERRAIN_HORIZON_MATH
#define SF_TERRAIN_HORIZON_MATH

const float HORIZON_PI = 3.141592653589793;
const float HORIZON_TWO_PI = 6.283185307179586;
const float HORIZON_ZERO_U = 0.12;
const float HORIZON_NEG_MAX = 0.15;
const float HORIZON_SLOPE_BIAS = 0.002;

vec3 horizon_cube_vector(uint face, vec2 uv) {
    switch (face) {
    case 0u: return vec3(1.0, uv.y, -uv.x);
    case 1u: return vec3(-1.0, uv.y, uv.x);
    case 2u: return vec3(uv.x, 1.0, -uv.y);
    case 3u: return vec3(uv.x, -1.0, uv.y);
    case 4u: return vec3(uv.x, uv.y, 1.0);
    default: return vec3(-uv.x, uv.y, -1.0);
    }
}

vec3 horizon_face_uv(vec3 direction) {
    vec3 a = abs(direction);
    if (a.x >= a.y && a.x >= a.z)
        return direction.x >= 0.0 ? vec3(-direction.z / a.x, direction.y / a.x, 0.0) : vec3(direction.z / a.x, direction.y / a.x, 1.0);
    if (a.y >= a.z)
        return direction.y >= 0.0 ? vec3(direction.x / a.y, -direction.z / a.y, 2.0) : vec3(direction.x / a.y, direction.z / a.y, 3.0);
    return direction.z >= 0.0 ? vec3(direction.x / a.z, direction.y / a.z, 4.0) : vec3(-direction.x / a.z, direction.y / a.z, 5.0);
}

vec3 horizon_face_ref_axis(uint face) {
    if (face == 2u) return vec3(0.0, 0.0, -1.0);
    if (face == 3u) return vec3(0.0, 0.0, 1.0);
    return vec3(0.0, 1.0, 0.0);
}

void horizon_tangent_basis(uint face, vec3 U, out vec3 T, out vec3 B) {
    vec3 ref = horizon_face_ref_axis(face);
    T = normalize(cross(ref, U));
    B = cross(U, T);
}

float encode_horizon_sin(float s) {
    float c = clamp(s, -HORIZON_NEG_MAX, 1.0);
    if (c >= 0.0) return HORIZON_ZERO_U + (1.0 - HORIZON_ZERO_U) * sqrt(c);
    return HORIZON_ZERO_U * (1.0 - sqrt(-c / HORIZON_NEG_MAX));
}

float decode_horizon_sin(float u) {
    float c = clamp(u, 0.0, 1.0);
    if (c >= HORIZON_ZERO_U) {
        float t = (c - HORIZON_ZERO_U) / (1.0 - HORIZON_ZERO_U);
        return t * t;
    }
    float t = 1.0 - c / HORIZON_ZERO_U;
    return -HORIZON_NEG_MAX * t * t;
}

vec4 decode_horizon_sin4(vec4 u) {
    vec4 c = clamp(u, vec4(0.0), vec4(1.0));
    vec4 tp = (c - HORIZON_ZERO_U) * (1.0 / (1.0 - HORIZON_ZERO_U));
    vec4 tn = 1.0 - c * (1.0 / HORIZON_ZERO_U);
    return mix(-HORIZON_NEG_MAX * tn * tn, tp * tp, greaterThanEqual(c, vec4(HORIZON_ZERO_U)));
}

float horizon_disk_above(float alpha, float theta_h, float star_ang_rad) {
    float r = max(star_ang_rad, 1e-5);
    float u = clamp((theta_h - alpha) / r, -1.0, 1.0);
    return (acos(u) - u * sqrt(max(1.0 - u * u, 0.0))) / HORIZON_PI;
}

float horizon_catmull_rom(float h_m1, float h_0, float h_1, float h_2, float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    float val = 0.5 * ((2.0 * h_0)
        + (-h_m1 + h_1) * t
        + (2.0 * h_m1 - 5.0 * h_0 + 4.0 * h_1 - h_2) * t2
        + (-h_m1 + 3.0 * h_0 - 3.0 * h_1 + h_2) * t3);
    return clamp(val, min(h_0, h_1), max(h_0, h_1));
}

#endif
