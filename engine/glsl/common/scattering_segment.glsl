// Finite-distance radiative transfer from a position-queryable 4D table.
// The bake omits phase functions and irradiance; each stellar-disc size has a
// cached solar variant. Direct R/M responses use square-root encoding.
uniform bool u_scattering_enabled;
uniform sampler2D u_scattering_tau_lut;
uniform sampler3D u_scattering_rayleigh_lut[4];
uniform sampler3D u_scattering_mie_lut[4];
uniform sampler3D u_scattering_multiple_lut[4];
uniform int u_scattering_azimuth_count;
uniform float u_scattering_sun_radius;
#include "scattering_coordinates.glsl"
#include "polar_winter.glsl"

vec2 endpoint_tau_uv(vec3 p, vec3 v, bool ground) {
    float r = clamp(length(p), u_scattering_bottom_km, u_atmo_radius_km);
    vec2 size = vec2(textureSize(u_scattering_tau_lut, 0));
    float q = scattering_view_coord(r, dot(p, v) / max(length(p), 1e-6), ground);
    float half_width = size.x * 0.5;
    float x = (ground ? 0.0 : half_width) + 0.5 + q * (half_width - 1.0);
    float h = sqrt(clamp((r - u_scattering_bottom_km)
        / (u_atmo_radius_km - u_scattering_bottom_km), 0.0, 1.0));
    return vec2(x / size.x, (0.5 + h * (size.y - 1.0)) / size.y);
}

vec3 endpoint_tau(vec3 p, vec3 v, bool ground) {
    return textureLod(u_scattering_tau_lut, endpoint_tau_uv(p, v, ground), 0.0).rgb;
}

// Full stellar-disc umbra is a convex cone behind the planet. If both ends
// lie inside it, the entire finite segment has no direct sunlight. Preserve
// multiple scattering, which can still illuminate twilight and the night.
bool endpoint_solar_umbra(vec3 p, vec3 sun, float radius) {
    float axial = dot(p, sun);
    float radial = length(p - axial * sun);
    float cosine = sqrt(max(0.0, 1.0 - radius * radius));
    return axial <= -u_planet_radius_km * radius
        && radial * cosine - axial * radius <= u_planet_radius_km;
}

void endpoint_scattering(vec3 p, vec3 v, vec3 sun, bool ground, int slot,
                         out vec3 R, out vec3 M, out vec3 MS) {
    float r = max(length(p), 1e-6);
    float mu = clamp(dot(p, v) / r, -1.0, 1.0);
    float mus = clamp(dot(p, sun) / r, -1.0, 1.0);
    float nu = clamp(dot(v, sun), -1.0, 1.0);
    vec3 anchor = scattering_light_anchor(r, mu, ground);
    mus = clamp((r * mus + anchor.z * nu) / anchor.x, -1.0, 1.0);
    float horizontal = sqrt(max(0.0, (1.0 - anchor.y * anchor.y) * (1.0 - mus * mus)));
    float azimuth = horizontal > 1e-5 ? clamp((nu - anchor.y * mus) / horizontal, -1.0, 1.0) : 0.0;
    vec3 size = vec3(textureSize(u_scattering_rayleigh_lut[slot], 0));
    float sun_count = size.x / float(u_scattering_azimuth_count);
    float index = scattering_azimuth_coord(azimuth) * float(u_scattering_azimuth_count - 1);
    float lower = min(floor(index), float(u_scattering_azimuth_count - 2));
    float fraction = index - lower;
    float x = 0.5 + scattering_light_coord(anchor.x, mus) * (sun_count - 1.0);
    float half_height = size.y * 0.5;
    float y = (ground ? 0.0 : half_height) + 0.5
            + scattering_view_coord(max(r, u_scattering_bottom_km), mu, ground) * (half_height - 1.0);
    float h = sqrt(clamp((r - u_scattering_bottom_km)
        / (u_atmo_radius_km - u_scattering_bottom_km), 0.0, 1.0));
    float z = 0.5 + h * (size.z - 1.0);
    vec3 uv0 = vec3(lower * sun_count + x, y, z) / size;
    vec3 uv1 = uv0 + vec3(sun_count / size.x, 0.0, 0.0);
    R = mix(textureLod(u_scattering_rayleigh_lut[slot], uv0, 0.0).rgb,
            textureLod(u_scattering_rayleigh_lut[slot], uv1, 0.0).rgb, fraction);
    M = mix(textureLod(u_scattering_mie_lut[slot], uv0, 0.0).rgb,
            textureLod(u_scattering_mie_lut[slot], uv1, 0.0).rgb, fraction);
    MS = mix(textureLod(u_scattering_multiple_lut[slot], uv0, 0.0).rgb,
             textureLod(u_scattering_multiple_lut[slot], uv1, 0.0).rgb, fraction);
    // Solar direct responses are stored in square-root space.
    R *= R;
    M *= M;
}

void endpoint_medium(vec3 p, out vec3 R, out vec3 M, out vec3 extinction) {
    float h = max(0.0, length(p) - u_planet_radius_km);
    float rho_R = exp(-h / max(u_h_rayleigh, 1e-4));
    float rho_M = exp(-h / max(u_h_mie, 1e-4));
    float z = (h - u_ozone_peak_km) / max(u_ozone_width_km, 1e-3);
    R = u_beta_rayleigh * (1000.0 * rho_R);
    M = u_beta_mie * clamp(u_mie_albedo, 0.0, 1.0) * (1000.0 * rho_M);
    extinction = (u_beta_rayleigh + u_beta_abs_mixed) * (1000.0 * rho_R)
               + u_beta_mie * (1000.0 * rho_M)
               + u_beta_abs_layered * (1000.0 * exp(-z * z));
}

// Short-segment limit avoids subtracting almost equal, interpolated boundary
// radiances directly under the observer's feet. It uses the local source and
// closed-form cell integral, with a continuous transition to endpoint queries.
float endpoint_long_weight(float distance) {
    float scale = max(0.001, min(u_h_rayleigh, u_h_mie));
    return smoothstep(scale * 0.10, scale * 0.50, distance);
}

vec3 endpoint_transmittance_above(vec3 a, vec3 b) {
    if (length(b - a) < 1e-6) return vec3(1.0);
    vec3 v = normalize(b - a);
    bool ground = dot(a, v) / length(a) < scattering_horizon(max(length(a), u_scattering_bottom_km));
    vec3 tau = max(vec3(0.0), endpoint_tau(a, v, ground) - endpoint_tau(b, v, ground));
    vec3 R, M, extinction;
    endpoint_medium(0.5 * (a + b), R, M, extinction);
    tau = mix(extinction * length(b - a), tau, endpoint_long_weight(length(b - a)));
    return exp(-tau);
}

// Negative terrain elevations do not move the atmospheric table boundary.
// The clamped density profile is homogeneous below the reference radius.
vec2 endpoint_below_interval(vec3 a, vec3 b) {
    float d = length(b - a);
    if (d < 1e-6) return vec2(0.0);
    vec3 v = (b - a) / d;
    float axial = dot(a, v);
    float disc = axial * axial + u_scattering_bottom_km * u_scattering_bottom_km - dot(a, a);
    if (disc <= 0.0) return vec2(0.0);
    float root = sqrt(disc);
    return clamp(vec2(-axial - root, -axial + root), 0.0, d);
}

vec3 endpoint_transmittance(vec3 a, vec3 b) {
    vec2 interval = endpoint_below_interval(a, b);
    if (interval.y - interval.x < 1e-6) return endpoint_transmittance_above(a, b);
    vec3 v = normalize(b - a), enter = a + v * interval.x, leave = a + v * interval.y;
    vec3 R, M, extinction;
    endpoint_medium(0.5 * (enter + leave), R, M, extinction);
    return endpoint_transmittance_above(a, enter) * exp(-extinction * (interval.y - interval.x))
        * endpoint_transmittance_above(leave, b);
}

// The light ray exits the homogeneous extension analytically before querying
// the above-datum optical table.
vec3 endpoint_incoming_tau(vec3 p, vec3 light) {
    float r = length(p);
    if (r >= u_scattering_bottom_km) return endpoint_tau(p, light, false);
    float axial = dot(p, light);
    float d = max(0.0, -axial + sqrt(max(0.0,
        axial * axial + u_scattering_bottom_km * u_scattering_bottom_km - r * r)));
    vec3 R, M, extinction;
    endpoint_medium(p, R, M, extinction);
    return extinction * d + endpoint_tau(p + d * light, light, false);
}

vec3 endpoint_local_factor(vec3 extinction, float d) {
    vec3 tau = extinction * d;
    return mix((1.0 - exp(-tau)) / max(extinction, vec3(1e-20)),
        d * (1.0 - 0.5 * tau + tau * tau / 6.0), lessThan(tau, vec3(1e-3)));
}

// Extinction is exact in this constant-density interval. The source holds
// incoming light and multiple scattering at the interval midpoint, as in
// the short-segment limit; it introduces no spatial integration steps.
vec3 endpoint_subsurface_solar(vec3 a, vec3 b, vec3 sun, float radius) {
    float d = length(b - a);
    if (d < 1e-6) return vec3(0.0);
    vec3 v = (b - a) / d, p = 0.5 * (a + b), radial = normalize(p);
    float r = length(p), nu = dot(v, sun), g = clamp(u_mie_g, 0.0, 0.88);
    float phase_R = 3.0 / (16.0 * PI) * (1.0 + nu * nu);
    float phase_M = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g) * (1.0 + nu * nu)
        / pow(max(1e-4, 1.0 + g * g - 2.0 * g * nu), 1.5);
    vec3 R, M, extinction;
    endpoint_medium(p, R, M, extinction);

    vec2 solstice_params = u_star_solstice[0].xy;
    uint ring_mask = floatBitsToUint(instances[u_body_idx * 7 + 3].w);
    float has_rings = (ring_mask != 0u) ? 1.0 : 0.0;
    float has_methane = u_precomp_mie.w;
    float polar_haze_factor;
    vec3 polar_rayleigh_boost;
    eval_polar_winter(p, solstice_params, has_rings, has_methane, polar_haze_factor, polar_rayleigh_boost);
    R *= polar_rayleigh_boost;
    M *= polar_haze_factor;

    float sin_planet = u_planet_radius_km / max(r, u_planet_radius_km + 0.01);
    vec2 term = sun_terminator(dot(radial, sun), sin_planet, sqrt(max(0.0, 1.0 - sin_planet * sin_planet)),
        radius, sqrt(max(0.0, 1.0 - radius * radius)));
    vec3 tangent = sun - dot(sun, radial) * radial;
    vec3 arbitrary = abs(radial.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    tangent = length(tangent) > 1e-6 ? normalize(tangent) : normalize(cross(radial, arbitrary));
    vec3 effective = radial * term.y + tangent * sqrt(max(0.0, 1.0 - term.y * term.y));
    vec3 incoming = exp(-endpoint_incoming_tau(p, effective)) * term.x;
    vec3 psi = textureLod(u_multi_scatter_lut,
        vec2(scattering_sun_coord(dot(radial, sun)), 0.0), 0.0).rgb;
    return ((R * phase_R + M * phase_M) * incoming + (R + M) * psi) * endpoint_local_factor(extinction, d);
}

vec3 endpoint_radiance_above(vec3 a, vec3 b, vec3 sun, vec3 transmission, int slot, float sun_radius) {
    float distance = length(b - a);
    if (distance < 1e-6) return vec3(0.0);
    vec3 v = (b - a) / distance;
    float nu = clamp(dot(v, sun), -1.0, 1.0);
    float phase_R = (3.0 / (16.0 * PI)) * (1.0 + nu * nu);
    float g = clamp(u_mie_g, 0.0, 0.88);
    float phase_M = (3.0 / (8.0 * PI)) * (1.0 - g * g) / (2.0 + g * g)
                 * (1.0 + nu * nu) / pow(max(1e-4, 1.0 + g * g - 2.0 * g * nu), 1.5);
    bool ground = dot(a, v) / length(a) < scattering_horizon(max(length(a), u_scattering_bottom_km));
    vec3 Ra, Ma, MSa, Rb, Mb, MSb;
    endpoint_scattering(a, v, sun, ground, slot, Ra, Ma, MSa);
    endpoint_scattering(b, v, sun, ground, slot, Rb, Mb, MSb);

    vec3 p = 0.5 * (a + b);
    vec2 solstice_params = (slot < 4) ? u_star_solstice[slot].xy : vec2(0.0);
    uint ring_mask = floatBitsToUint(instances[u_body_idx * 7 + 3].w);
    float has_rings = (ring_mask != 0u) ? 1.0 : 0.0;
    float has_methane = u_precomp_mie.w;
    float polar_haze_factor;
    vec3 polar_rayleigh_boost;
    eval_polar_winter(p, solstice_params, has_rings, has_methane, polar_haze_factor, polar_rayleigh_boost);

    vec3 direct_R = max(vec3(0.0), Ra - transmission * Rb) * phase_R * polar_rayleigh_boost;
    vec3 direct_M = max(vec3(0.0), Ma - transmission * Mb) * phase_M * polar_haze_factor;
    vec3 direct = direct_R + direct_M;
    if (endpoint_solar_umbra(a, sun, sun_radius) && endpoint_solar_umbra(b, sun, sun_radius))
        direct = vec3(0.0);
    vec3 result = max(vec3(0.0), direct + MSa - transmission * MSb);
    if (endpoint_long_weight(distance) < 1.0) {
        float r = length(p);
        vec3 R, M, extinction;
        endpoint_medium(p, R, M, extinction);
        R *= polar_rayleigh_boost;
        M *= polar_haze_factor;
        float sin_planet = u_planet_radius_km / max(r, u_planet_radius_km + 0.01);
        vec2 term = sun_terminator(dot(p, sun) / r, sin_planet, sqrt(max(0.0, 1.0 - sin_planet * sin_planet)),
                                  sun_radius, sqrt(max(0.0, 1.0 - sun_radius * sun_radius)));
        float mus = max(term.y, scattering_horizon(max(r, u_scattering_bottom_km)) + 1e-5);
        vec3 radial = p / r;
        vec3 tangent = sun - dot(sun, radial) * radial;
        vec3 arbitrary = abs(radial.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0);
        tangent = length(tangent) > 1e-6 ? normalize(tangent) : normalize(cross(radial, arbitrary));
        vec3 effective_sun = radial * mus + tangent * sqrt(max(0.0, 1.0 - mus * mus));
        vec3 sun_T = exp(-endpoint_tau(p, effective_sun, false)) * term.x;
        float h = clamp((r - u_planet_radius_km) / (u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
        vec3 psi = textureLod(u_multi_scatter_lut,
            vec2(scattering_sun_coord(dot(p, sun) / r), sqrt(h)), 0.0).rgb;
        vec3 local_tau = extinction * distance;
        vec3 factor = mix((1.0 - exp(-local_tau)) / max(extinction, vec3(1e-20)),
            distance * (1.0 - 0.5 * local_tau + local_tau * local_tau / 6.0),
            lessThan(local_tau, vec3(1e-3)));
        vec3 local = ((R * phase_R + M * phase_M) * sun_T + (R + M) * psi) * factor;
        result = mix(local, result, endpoint_long_weight(distance));
    }
    return result;
}

vec3 endpoint_radiance_star(vec3 a, vec3 b, vec3 sun, vec3 transmission, int slot, float radius) {
    vec2 interval = endpoint_below_interval(a, b);
    if (interval.y - interval.x < 1e-6) return endpoint_radiance_above(a, b, sun, transmission, slot, radius);
    vec3 v = normalize(b - a), enter = a + v * interval.x, leave = a + v * interval.y;
    vec3 T0 = endpoint_transmittance_above(a, enter), T1 = endpoint_transmittance_above(leave, b);
    vec3 R, M, extinction;
    endpoint_medium(0.5 * (enter + leave), R, M, extinction);
    vec3 Tb = exp(-extinction * (interval.y - interval.x));
    return endpoint_radiance_above(a, enter, sun, T0, slot, radius)
        + T0 * (endpoint_subsurface_solar(enter, leave, sun, radius)
            + Tb * endpoint_radiance_above(leave, b, sun, T1, slot, radius));
}

vec3 endpoint_radiance(vec3 a, vec3 b, vec3 sun, vec3 transmission) {
    return endpoint_radiance_star(a, b, sun, transmission, 0, u_scattering_sun_radius);
}
