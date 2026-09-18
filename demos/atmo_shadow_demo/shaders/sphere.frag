#version 330 core

in vec3 v_world_pos;
in vec3 v_world_normal;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_albedo;
uniform float u_ambient;

uniform sampler2D u_ring_tex;
uniform float u_ring_inner;
uniform float u_ring_outer;
uniform float u_sun_angular_radius; // in radians (~0.009 for 0.5 deg)
uniform int u_penumbra_samples;     // e.g. 9 or 15

out vec4 out_color;

float compute_ring_shadow(vec3 P, vec3 L) {
    // If the sun direction is parallel to the ring plane (L.y == 0), no planar shadow
    if (abs(L.y) < 1e-6) return 1.0;

    // Distance t along ray P + t*L to intersect plane Y = 0
    float t = -P.y / L.y;

    // If t <= 0, the ring plane is behind the point relative to the sun
    if (t <= 0.0001) return 1.0;

    // Intersection position on ring plane
    vec3 P_ring = P + t * L;
    float r = length(P_ring.xz);

    // Penumbra width: sun angular radius projects onto the ring plane
    float tan_alpha = tan(max(u_sun_angular_radius, 1e-5));
    // Radial projection of penumbra disk onto the ring plane
    float w_r = (t * tan_alpha) / max(abs(L.y), 0.08);

    float ring_span = u_ring_outer - u_ring_inner;
    float u_center = (r - u_ring_inner) / ring_span;
    float du = w_r / ring_span;

    // If penumbra is tiny or samples <= 1, single tap
    if (du < 0.0003 || u_penumbra_samples <= 1) {
        if (u_center >= 0.0 && u_center <= 1.0) {
            float raw_a = texture(u_ring_tex, vec2(u_center, 0.5)).a;
            float tau = -log(max(1e-4, 1.0 - raw_a));
            float opac = 1.0 - exp(-tau / max(abs(L.y), 0.05));
            return 1.0 - opac;
        }
        return 1.0;
    }

    // Gaussian-weighted multi-tap filter across penumbra width
    float total_blocked = 0.0;
    float total_weight = 0.0;
    int samples = clamp(u_penumbra_samples, 3, 21);
    int half_s = samples / 2;

    float inv_sin_sun = 1.0 / max(abs(L.y), 0.05);

    for (int i = -half_s; i <= half_s; i++) {
        float x = float(i) / float(half_s); // -1.0 to 1.0
        float weight = exp(-2.0 * x * x);
        float u = u_center + x * du;

        float sample_opac = 0.0;
        if (u >= 0.0 && u <= 1.0) {
            float raw_a = texture(u_ring_tex, vec2(u, 0.5)).a;
            // Physical optical depth scaled by slant path through ring slab
            float tau = -log(max(1e-4, 1.0 - raw_a));
            sample_opac = 1.0 - exp(-tau * inv_sin_sun);
        }
        total_blocked += sample_opac * weight;
        total_weight += weight;
    }

    float avg_blocked = total_blocked / max(total_weight, 1e-5);
    return clamp(1.0 - avg_blocked, 0.0, 1.0);
}

void main() {
    // Treat the surface as an analytical unit sphere to avoid mesh polygon facet artifacts
    vec3 N = normalize(v_world_normal);
    vec3 P = normalize(v_world_pos);
    vec3 L = normalize(u_sun_dir);

    float NdotL = max(0.0, dot(N, L));
    float shadow = 1.0;

    if (NdotL > 0.0) {
        shadow = compute_ring_shadow(P, L);
    }

    vec3 direct = u_sun_color * u_albedo * (NdotL * shadow);
    vec3 ambient = u_albedo * u_ambient;

    // Simple tone curve
    vec3 final_color = direct + ambient;
    out_color = vec4(final_color, 1.0);
}
