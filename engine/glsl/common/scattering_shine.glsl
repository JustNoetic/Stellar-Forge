// Unit-light transport for secondary directional light and host ringshine.
// The ringshine map remains evaluated at the segment midpoint, matching the
// existing lighting approximation. No spatial quadrature is performed here.
uniform sampler3D u_scattering_shine_rayleigh_lut;
uniform sampler3D u_scattering_shine_mie_lut;
uniform sampler3D u_scattering_shine_multiple_lut;
uniform sampler2D u_scattering_ambient_rayleigh_lut;
uniform sampler2D u_scattering_ambient_mie_lut;
uniform sampler3D u_scattering_ring_multiple_lut;

vec3 endpoint_transport(sampler3D table, vec3 p, vec3 v, vec3 axis, bool ground) {
    float r = max(length(p), 1e-6);
    float mu = clamp(dot(p, v) / r, -1.0, 1.0);
    float mus = clamp(dot(p, axis) / r, -1.0, 1.0);
    float nu = clamp(dot(v, axis), -1.0, 1.0);
    vec3 anchor = scattering_light_anchor(r, mu, ground);
    mus = clamp((r * mus + anchor.z * nu) / anchor.x, -1.0, 1.0);
    float horizontal = sqrt(max(0.0, (1.0 - anchor.y * anchor.y) * (1.0 - mus * mus)));
    float az = horizontal > 1e-5 ? clamp((nu - anchor.y * mus) / horizontal, -1.0, 1.0) : 0.0;
    vec3 size = vec3(textureSize(table, 0));
    float count = size.x / float(u_scattering_azimuth_count);
    float index = scattering_azimuth_coord(az) * float(u_scattering_azimuth_count - 1);
    float lower = min(floor(index), float(u_scattering_azimuth_count - 2));
    float x = 0.5 + scattering_light_coord(anchor.x, mus) * (count - 1.0);
    float half_height = size.y * 0.5;
    float y = (ground ? 0.0 : half_height) + 0.5
        + scattering_view_coord(max(r, u_scattering_bottom_km), mu, ground) * (half_height - 1.0);
    float h = sqrt(clamp((r - u_scattering_bottom_km)
        / (u_atmo_radius_km - u_scattering_bottom_km), 0.0, 1.0));
    vec3 uv = vec3(lower * count + x, y, 0.5 + h * (size.z - 1.0)) / size;
    return mix(textureLod(table, uv, 0.0).rgb,
        textureLod(table, uv + vec3(count / size.x, 0.0, 0.0), 0.0).rgb, index - lower);
}

// Secondary light is smooth over a local scale height. Keep its homogeneous
// source limit longer than the solar path to suppress cancellation in the
// low-amplitude boundary responses without adding raymarch samples.
float endpoint_secondary_long_weight(float d) {
    float scale = max(0.001, min(u_h_rayleigh, u_h_mie));
    return smoothstep(scale * 0.25, scale, d);
}

vec3 endpoint_shine_above(vec3 a, vec3 b, vec3 light, vec3 T) {
    float d = length(b - a);
    if (d < 1e-6) return vec3(0.0);
    vec3 v = (b - a) / d;
    bool ground = dot(a, v) / length(a) < scattering_horizon(max(length(a), u_scattering_bottom_km));
    float nu = dot(v, light), g = clamp(u_mie_g, 0.0, 0.88);
    float phase_R = 3.0 / (16.0 * PI) * (1.0 + nu * nu);
    float phase_M = 3.0 / (8.0 * PI) * (1.0 - g*g) / (2.0 + g*g)
        * (1.0 + nu*nu) / pow(max(1e-4, 1.0 + g*g - 2.0*g*nu), 1.5);
    vec3 R = endpoint_transport(u_scattering_shine_rayleigh_lut, a, v, light, ground)
        - T * endpoint_transport(u_scattering_shine_rayleigh_lut, b, v, light, ground);
    vec3 M = endpoint_transport(u_scattering_shine_mie_lut, a, v, light, ground)
        - T * endpoint_transport(u_scattering_shine_mie_lut, b, v, light, ground);
    vec3 MS = endpoint_transport(u_scattering_shine_multiple_lut, a, v, light, ground)
        - T * endpoint_transport(u_scattering_shine_multiple_lut, b, v, light, ground);
    vec3 result = max(vec3(0.0), R * phase_R + M * phase_M + MS);
    float weight = endpoint_secondary_long_weight(d);
    if (weight < 1.0) {
        vec3 p = 0.5 * (a + b), local_R, local_M, extinction;
        endpoint_medium(p, local_R, local_M, extinction);
        float r = length(p), mus = dot(p, light) / r;
        float horizon = -sqrt(max(0.0, 1.0 - pow(u_planet_radius_km / max(r, u_planet_radius_km + 0.01), 2.0)));
        vec3 light_T = mus < horizon ? vec3(0.0) : exp(-endpoint_tau(p, light, false))
            * smoothstep(horizon - 0.05, horizon + 0.05, mus);
        float h = clamp((r - u_planet_radius_km) / (u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
        vec3 psi = textureLod(u_multi_scatter_lut, vec2(scattering_sun_coord(mus), sqrt(h)), 0.0).rgb;
        vec3 local = ((local_R * phase_R + local_M * phase_M) + (local_R + local_M) * psi)
            * light_T * endpoint_local_factor(extinction, d);
        result = mix(local, result, weight);
    }
    return result;
}

vec3 endpoint_ring_response_above(vec3 a, vec3 b, vec3 normal, vec3 T) {
    float d = length(b - a);
    if (d < 1e-6) return vec3(0.0);
    vec3 v = (b - a) / d;
    bool ground = dot(a, v) / length(a) < scattering_horizon(max(length(a), u_scattering_bottom_km));
    vec2 ua = endpoint_tau_uv(a, v, ground), ub = endpoint_tau_uv(b, v, ground);
    vec3 R = textureLod(u_scattering_ambient_rayleigh_lut, ua, 0.0).rgb
        - T * textureLod(u_scattering_ambient_rayleigh_lut, ub, 0.0).rgb;
    vec3 M = textureLod(u_scattering_ambient_mie_lut, ua, 0.0).rgb
        - T * textureLod(u_scattering_ambient_mie_lut, ub, 0.0).rgb;
    vec3 MS = endpoint_transport(u_scattering_ring_multiple_lut, a, v, normal, ground)
        - T * endpoint_transport(u_scattering_ring_multiple_lut, b, v, normal, ground);
    float g = clamp(u_mie_g, 0.0, 0.88);
    float phase = 1.0 / (4.0 * PI), mie_phase = phase / max(0.15, 1.0 - g);
    vec3 result = max(vec3(0.0), R * phase + M * mie_phase + MS);
    float weight = endpoint_secondary_long_weight(d);
    if (weight < 1.0) {
        vec3 p = 0.5 * (a + b), local_R, local_M, extinction;
        endpoint_medium(p, local_R, local_M, extinction);
        float r = length(p), elevation = dot(p, normal) / r;
        float h = clamp((r - u_planet_radius_km) / (u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
        vec3 psi = textureLod(u_multi_scatter_lut,
            vec2(scattering_sun_coord(sqrt(max(0.0, 1.0 - elevation * elevation))), sqrt(h)), 0.0).rgb;
        vec3 local = (local_R * phase + local_M * mie_phase + (local_R + local_M) * psi)
            * endpoint_local_factor(extinction, d);
        result = mix(local, result, weight);
    }
    return result;
}

// Below-datum transport uses the same homogeneous, midpoint-source extension
// as solar light. Normal above-datum rays only query the cached fields.
vec3 endpoint_subsurface_shine(vec3 a, vec3 b, vec3 light) {
    float d = length(b - a);
    if (d < 1e-6) return vec3(0.0);
    vec3 v = (b - a) / d, p = 0.5 * (a + b);
    float r = length(p), mus = dot(p, light) / r, nu = dot(v, light), g = clamp(u_mie_g, 0.0, 0.88);
    float phase_R = 3.0 / (16.0 * PI) * (1.0 + nu * nu);
    float phase_M = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g) * (1.0 + nu * nu)
        / pow(max(1e-4, 1.0 + g * g - 2.0 * g * nu), 1.5);
    float horizon = -sqrt(max(0.0, 1.0 - pow(u_planet_radius_km / max(r, u_planet_radius_km + 0.01), 2.0)));
    vec3 incoming = mus < horizon ? vec3(0.0) : exp(-endpoint_incoming_tau(p, light))
        * smoothstep(horizon - 0.05, horizon + 0.05, mus);
    vec3 R, M, extinction;
    endpoint_medium(p, R, M, extinction);
    vec3 psi = textureLod(u_multi_scatter_lut, vec2(scattering_sun_coord(mus), 0.0), 0.0).rgb;
    return ((R * phase_R + M * phase_M) + (R + M) * psi) * incoming * endpoint_local_factor(extinction, d);
}

vec3 endpoint_shine(vec3 a, vec3 b, vec3 light, vec3 T) {
    vec2 interval = endpoint_below_interval(a, b);
    if (interval.y - interval.x < 1e-6) return endpoint_shine_above(a, b, light, T);
    vec3 v = normalize(b - a), enter = a + v * interval.x, leave = a + v * interval.y;
    vec3 T0 = endpoint_transmittance_above(a, enter), T1 = endpoint_transmittance_above(leave, b);
    vec3 R, M, extinction;
    endpoint_medium(0.5 * (enter + leave), R, M, extinction);
    vec3 Tb = exp(-extinction * (interval.y - interval.x));
    return endpoint_shine_above(a, enter, light, T0)
        + T0 * (endpoint_subsurface_shine(enter, leave, light) + Tb * endpoint_shine_above(leave, b, light, T1));
}

vec3 endpoint_subsurface_ring(vec3 a, vec3 b, vec3 normal) {
    float d = length(b - a);
    if (d < 1e-6) return vec3(0.0);
    vec3 p = 0.5 * (a + b), R, M, extinction;
    endpoint_medium(p, R, M, extinction);
    float elevation = dot(normalize(p), normal), g = clamp(u_mie_g, 0.0, 0.88);
    vec3 psi = textureLod(u_multi_scatter_lut,
        vec2(scattering_sun_coord(sqrt(max(0.0, 1.0 - elevation * elevation))), 0.0), 0.0).rgb;
    float phase = 1.0 / (4.0 * PI);
    return (R * phase + M * phase / max(0.15, 1.0 - g) + (R + M) * psi) * endpoint_local_factor(extinction, d);
}

vec3 endpoint_ring_response(vec3 a, vec3 b, vec3 normal, vec3 T) {
    vec2 interval = endpoint_below_interval(a, b);
    if (interval.y - interval.x < 1e-6) return endpoint_ring_response_above(a, b, normal, T);
    vec3 v = normalize(b - a), enter = a + v * interval.x, leave = a + v * interval.y;
    vec3 T0 = endpoint_transmittance_above(a, enter), T1 = endpoint_transmittance_above(leave, b);
    vec3 R, M, extinction;
    endpoint_medium(0.5 * (enter + leave), R, M, extinction);
    vec3 Tb = exp(-extinction * (interval.y - interval.x));
    return endpoint_ring_response_above(a, enter, normal, T0)
        + T0 * (endpoint_subsurface_ring(enter, leave, normal) + Tb * endpoint_ring_response_above(leave, b, normal, T1));
}

vec3 endpoint_secondary_light(vec3 a, vec3 b, vec3 T, vec3 ps_dir, vec3 ps_color, uint ring_mask) {
    vec3 result = vec3(0.0);
    if ((u_planetshine_enabled || u_ringshine_enabled) && dot(ps_color, ps_color) > 1e-12
        && dot(ps_dir, ps_dir) > 1e-12) {
        vec3 light = normalize(toSphericalSpace(normalize(ps_dir), u_pole_obl));
        result += endpoint_shine(a, b, light, T) * ps_color * (u_sun_intensity * PI);
    }
    if (!u_ringshine_enabled || ring_mask == 0u) return result;
    vec3 midpoint = fromSphericalSpace(0.5 * (a + b), u_pole_obl);
    vec3 P = normalize(midpoint);
    vec3 L0 = normalize(u_star_pos_local[0].xyz);
    for (int k = 0; k < u_num_ring_planes; ++k) {
        if ((ring_mask & (1u << k)) == 0u) continue;
        vec3 normal = normalize(u_ring_normal[k]);
        vec3 response = endpoint_ring_response(a, b, normalize(toSphericalSpace(normal, u_pole_obl)), T);
        for (int st = 0; st < min(u_num_stars, 4); ++st) {
            vec3 L = normalize(u_star_pos_local[st].xyz);
            float hemi = dot(L, normal) * dot(L0, normal) >= 0.0 ? 1.0 : -1.0;
            vec3 anti = -L + normal * dot(L, normal);
            anti = length(anti) > 1e-5 ? normalize(anti) : vec3(-1.0, 0.0, 0.0);
            vec3 equator = P - normal * dot(P, normal);
            equator = length(equator) > 1e-5 ? normalize(equator) : vec3(1.0, 0.0, 0.0);
            float x = atan(dot(cross(anti, equator), normal), clamp(dot(anti, equator), -1.0, 1.0)) / PI;
            float y = dot(P, normal) * hemi;
            vec2 uv = vec2(0.5 + 0.5 * sign(x) * pow(abs(x), 2.0/3.0),
                (float(k) + 0.5 + 0.5 * sign(y) * pow(abs(y), 2.0/3.0)) / 16.0);
            vec3 irradiance = textureLod(u_ringshine_map, uv, 0.0).rgb / PI;
            result += response * irradiance * u_star_color_irrad[st].rgb;
        }
    }
    return result;
}
