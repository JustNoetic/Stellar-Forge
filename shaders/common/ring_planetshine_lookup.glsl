#ifndef RING_PLANETSHINE_LOOKUP_GLSL
#define RING_PLANETSHINE_LOOKUP_GLSL

layout(set = 0, binding = 17) uniform sampler2DArray u_ring_planetshine_map;

const float RP_LOOKUP_PI = 3.14159265358979323846;
const float RP_LOOKUP_MAX_SIN = 0.999;
const float RP_LOOKUP_INV_MAX_ALPHA = 0.655277403854991;

void ring_planetshine_unpack_face(vec4 moment, vec3 flux_scale, float u,
                                  vec3 e_r, vec3 e_phi_folded, vec3 pole_n,
                                  out vec3 flux, out vec3 light, out float spread) {
    flux = vec3(0.0);
    light = e_r;
    spread = 0.0;
    if (moment.r <= 1e-20) return;

    flux = moment.r * (u * u) * flux_scale;
    float o_phi = moment.g / moment.r;
    float o_z = moment.b / moment.r;
    float r2_mean = clamp(moment.a / moment.r, 0.0, 1.0);
    float var_disk = max(0.0, r2_mean - (o_phi * o_phi + o_z * o_z));
    spread = clamp(u * sqrt(2.0 * var_disk), 0.0, 1.0);

    vec2 d_t = u * vec2(o_phi, o_z);
    float d_t2 = dot(d_t, d_t);
    if (d_t2 > 0.999999) {
        d_t *= inversesqrt(d_t2) * 0.9999995;
        d_t2 = dot(d_t, d_t);
    }
    float d_r = sqrt(max(0.0, 1.0 - d_t2));
    light = normalize(d_r * e_r + d_t.x * e_phi_folded + d_t.y * pole_n);
}

float ring_fast_atan_pos(float y_abs, float x) {
    float x_abs = abs(x);
    float a = min(x_abs, y_abs);
    float b = max(x_abs, y_abs);
    float t = (b > 1e-20) ? (a / b) : 0.0;
    float t2 = t * t;
    float r = t * (0.9998660 + t2 * (-0.3302995 + t2 * (0.1801410 + t2 * (-0.0851330 + t2 * 0.0208351))));
    float phi = (y_abs > x_abs) ? (1.5707963267948966 - r) : r;
    return (x < 0.0) ? (RP_LOOKUP_PI - phi) : phi;
}

bool ring_planetshine_sample_moments(uint host_idx, uint star_col, vec3 hit_local, vec3 pole_n,
                                       vec3 star_dir, float host_radius, float cam_side,
                                       out vec4 same_m, out vec4 other_m, out float u,
                                       out vec3 e_r, out vec3 e_phi_folded) {
    same_m = vec4(0.0);
    other_m = vec4(0.0);
    u = 0.0;
    e_r = -hit_local;
    e_phi_folded = vec3(0.0);

    float r = length(hit_local);
    if (r <= max(host_radius, 1e-12)) return false;

    u = clamp(host_radius / r, 0.0, RP_LOOKUP_MAX_SIN);
    float alpha = ring_fast_asin(u);
    float tx = clamp(alpha * RP_LOOKUP_INV_MAX_ALPHA, 0.0, 1.0);

    e_r = -hit_local / r;
    vec3 radial = -e_r;
    vec3 e_t = cross(pole_n, e_r);
    vec3 sun_horiz = star_dir - pole_n * dot(star_dir, pole_n);
    float sun_horiz_sq = dot(sun_horiz, sun_horiz);
    float cos_phi_u;
    float sin_phi_u;
    if (sun_horiz_sq > 1e-14) {
        cos_phi_u = dot(radial, sun_horiz);
        sin_phi_u = dot(e_t, star_dir);
    } else {
        vec3 axis = abs(pole_n.x) < 0.8 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
        vec3 sun_x = normalize(axis - pole_n * dot(axis, pole_n));
        vec3 sun_y = cross(pole_n, sun_x);
        cos_phi_u = dot(radial, sun_x);
        sin_phi_u = dot(radial, sun_y);
    }
    float phi = ring_fast_atan_pos(abs(sin_phi_u), cos_phi_u);
    float ty = clamp(phi / RP_LOOKUP_PI, 0.0, 1.0);

    float phi_sign = sin_phi_u >= 0.0 ? 1.0 : -1.0;
    e_phi_folded = e_t * phi_sign;

    vec2 uv = (vec2(float(star_col), float(host_idx)) * 65.0 + 0.5 + vec2(tx, ty) * 64.0)
            * (1.0 / (16.0 * 65.0));
    vec4 top_m = textureLod(u_ring_planetshine_map, vec3(uv, 0.0), 0.0);
    vec4 bot_m = textureLod(u_ring_planetshine_map, vec3(uv, 1.0), 0.0);

    same_m = cam_side >= 0.0 ? top_m : bot_m;
    other_m = cam_side >= 0.0 ? bot_m : top_m;
    return true;
}

void ring_planetshine_sample_star(uint host_idx, uint star_col, vec3 hit_local, vec3 pole_n,
                                  vec3 star_dir, vec3 flux_scale, float host_radius, float cam_side,
                                  out vec3 same_flux, out vec3 same_light, out float same_spread,
                                  out vec3 other_flux, out vec3 other_light, out float other_spread) {
    same_flux = other_flux = vec3(0.0);
    same_light = other_light = -hit_local;
    same_spread = other_spread = 0.0;

    vec4 same_m, other_m;
    float u;
    vec3 e_r, e_phi_folded;
    if (!ring_planetshine_sample_moments(host_idx, star_col, hit_local, pole_n, star_dir,
                                         host_radius, cam_side, same_m, other_m, u, e_r, e_phi_folded)) {
        return;
    }

    ring_planetshine_unpack_face(same_m, flux_scale, u, e_r, e_phi_folded, pole_n,
                                 same_flux, same_light, same_spread);
    ring_planetshine_unpack_face(other_m, flux_scale, u, e_r, e_phi_folded, pole_n,
                                 other_flux, other_light, other_spread);
}

#endif
