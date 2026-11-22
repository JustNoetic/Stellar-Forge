#ifndef SF_HEIGHT_SAMPLE_GLSL
#define SF_HEIGHT_SAMPLE_GLSL

#include "terrain_horizon_math.glsl"

const uint HEIGHT_POOL_TILE_STRIDE = 17689u;

struct NormalBakeJobGPU {
    uvec4 address;
    uvec4 target;
    vec4 metrics;
};

float height_pool_at(uint slot, uint side, int x, int y) {
    return height_pool[slot * HEIGHT_POOL_TILE_STRIDE + uint(y) * side + uint(x)];
}

float height_pool_sample_linear(uint slot, uint cells, uint gutter, float gx, float gy) {
    uint side = cells + 1u + 2u * gutter;
    float cx = clamp(gx, -1.0, float(cells + 1u));
    float cy = clamp(gy, -1.0, float(cells + 1u));
    int ix = min(int(floor(cx)), int(cells));
    int iy = min(int(floor(cy)), int(cells));
    float u = cx - float(ix);
    float v = cy - float(iy);
    int ox = ix + int(gutter);
    int oy = iy + int(gutter);
    float a00 = height_pool_at(slot, side, ox, oy);
    float a10 = height_pool_at(slot, side, ox + 1, oy);
    float a01 = height_pool_at(slot, side, ox, oy + 1);
    float a11 = height_pool_at(slot, side, ox + 1, oy + 1);
    return u + v <= 1.0
        ? a00 * (1.0 - u - v) + a10 * u + a01 * v
        : a11 * (u + v - 1.0) + a10 * (1.0 - v) + a01 * (1.0 - u);
}

float height_pool_sample_cubic(uint slot, uint cells, uint gutter, float gx, float gy) {
    uint side = cells + 1u + 2u * gutter;
    float cx = clamp(gx, -1.0, float(cells + 1u));
    float cy = clamp(gy, -1.0, float(cells + 1u));
    int ix = clamp(int(floor(cx)), -1, int(cells));
    int iy = clamp(int(floor(cy)), -1, int(cells));
    float u = cx - float(ix);
    float v = cy - float(iy);
    float om_u = 1.0 - u;
    float om_v = 1.0 - v;
    float wx[4] = float[4](
        om_u * om_u * om_u / 6.0,
        (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
        (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0,
        (u * u * u) / 6.0
    );
    float wy[4] = float[4](
        om_v * om_v * om_v / 6.0,
        (3.0 * v * v * v - 6.0 * v * v + 4.0) / 6.0,
        (-3.0 * v * v * v + 3.0 * v * v + 3.0 * v + 1.0) / 6.0,
        (v * v * v) / 6.0
    );
    float sum = 0.0;
    int max_idx = int(side) - 1;
    for (int dy = -1; dy <= 2; ++dy) {
        int row = clamp(iy + dy + int(gutter), 0, max_idx);
        float row_sum = 0.0;
        for (int dx = -1; dx <= 2; ++dx) {
            int col = clamp(ix + dx + int(gutter), 0, max_idx);
            row_sum += height_pool_at(slot, side, col, row) * wx[dx + 1];
        }
        sum += row_sum * wy[dy + 1];
    }
    return sum;
}

float height_pool_sample(uint slot, uint cells, uint gutter, bool cubic, float gx, float gy) {
    if (cubic && gutter >= 2u) return height_pool_sample_cubic(slot, cells, gutter, gx, gy);
    return height_pool_sample_linear(slot, cells, gutter, gx, gy);
}

float height_sample_world(uint page_base, uint level, uint src_face, float u, float v, uint cells, uint gutter, bool cubic) {
    vec3 dir = normalize(horizon_cube_vector(src_face, vec2(u, v)));
    vec3 fuv = horizon_face_uv(dir);
    uint at_face = uint(fuv.z);
    uint cur_level = level;
    while (true) {
        uint n = 1u << cur_level;
        vec2 grid_pos = (fuv.xy + 1.0) * 0.5 * float(n);
        uvec2 tile_xy = uvec2(clamp(floor(grid_pos), vec2(0.0), vec2(float(n - 1u))));
        uint n2 = n * n;
        uint slot_val = height_words[page_base + (2u + at_face) * n2 - 2u + tile_xy.y * n + tile_xy.x];
        if (slot_val != 0u) {
            vec2 local = grid_pos - vec2(tile_xy);
            return height_pool_sample(slot_val - 1u, cells, gutter, cubic, local.x * float(cells), local.y * float(cells));
        }
        if (cur_level == 0u) return 0.0;
        --cur_level;
    }
}

vec3 height_ellipsoid_delta(vec3 raw_p, vec3 raw_m, float h_p_m, float h_m_m, vec3 ell_scale, float radius_m) {
    vec3 raw_0 = 0.5 * (raw_p + raw_m);
    vec3 d_raw = 0.5 * (raw_p - raw_m);
    float len_p = length(raw_p);
    float len_m = length(raw_m);
    float len_sum = len_p + len_m;
    vec3 d_dir = (d_raw * len_sum - raw_0 * (4.0 * dot(raw_0, d_raw) / max(len_sum, 1e-12))) / max(len_p * len_m, 1e-12);

    vec3 s_p = raw_p / ell_scale;
    vec3 s_m = raw_m / ell_scale;
    vec3 s_0 = raw_0 / ell_scale;
    vec3 d_s = d_raw / ell_scale;
    float slen_p = length(s_p);
    float slen_m = length(s_m);
    float slen_sum = slen_p + slen_m;
    vec3 d_n = (d_s * slen_sum - s_0 * (4.0 * dot(s_0, d_s) / max(slen_sum, 1e-12))) / max(slen_p * slen_m, 1e-12);
    vec3 n_avg = 0.5 * (s_p / slen_p + s_m / slen_m);

    return d_dir * ell_scale + n_avg * ((h_p_m - h_m_m) / radius_m) + d_n * (0.5 * (h_p_m + h_m_m) / radius_m);
}

#endif
