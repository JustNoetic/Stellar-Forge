#version 330 core

in vec3 v_world_pos;
in float v_uv_r;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform sampler2D u_ring_tex;
uniform float u_planet_radius;       // e.g. 1.0
uniform float u_sun_angular_radius; // radians

out vec4 out_color;

float compute_planet_shadow(vec3 Q, vec3 L) {
    // Planet is at origin (0, 0, 0) with radius u_planet_radius.
    // Ray from Q towards sun: Q + t*L, t > 0
    float b = dot(Q, L);

    // If b >= 0, the ray moves away from the origin
    if (b >= 0.0) return 1.0;

    // Closest approach distance squared to planet center
    float dist_sq = dot(Q, Q);
    float d_perp_sq = dist_sq - b * b;
    float d_perp = sqrt(max(0.0, d_perp_sq));

    // Distance to closest approach along ray
    float dist_to_planet = abs(b);

    // Projected sun radius at the planet
    float tan_alpha = tan(max(u_sun_angular_radius, 1e-5));
    float penumbra_radius = dist_to_planet * tan_alpha;

    float r_umbra = max(0.0, u_planet_radius - penumbra_radius);
    float r_penumbra = u_planet_radius + penumbra_radius;

    return smoothstep(r_umbra, r_penumbra, d_perp);
}

void main() {
    float u = clamp(v_uv_r, 0.0, 1.0);
    vec4 tex_val = texture(u_ring_tex, vec2(u, 0.5));
    float opacity = tex_val.a;

    if (opacity < 0.005) discard;

    vec3 L = normalize(u_sun_dir);
    float shadow = compute_planet_shadow(v_world_pos, L);

    // Ring double-sided illumination by sunlight angle relative to the ring plane (Y-axis normal)
    float NdotL = max(0.02, abs(L.y));

    // Warm ring particulate scattering
    vec3 ring_albedo = tex_val.rgb;
    vec3 lit = ring_albedo * u_sun_color * (NdotL * shadow);
    vec3 ambient = ring_albedo * 0.03;

    out_color = vec4(lit + ambient, opacity);
}
