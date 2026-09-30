// Finite-distance radiative transfer from a position-queryable 4D table.
// The bake omits phase functions and irradiance, so all stars share the tables.
uniform bool u_scattering_enabled;
uniform sampler2D u_scattering_tau_lut;
uniform sampler3D u_scattering_rayleigh_lut;
uniform sampler3D u_scattering_mie_lut;
uniform sampler3D u_scattering_multiple_lut;
uniform int u_scattering_azimuth_count;
uniform float u_scattering_sun_radius;
#include "scattering_coordinates.glsl"

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

void endpoint_scattering(vec3 p, vec3 v, vec3 sun, bool ground,
                         out vec3 R, out vec3 M, out vec3 MS) {
    float r = max(length(p), 1e-6);
    float mu = clamp(dot(p, v) / r, -1.0, 1.0);
    float mus = clamp(dot(p, sun) / r, -1.0, 1.0);
    float nu = clamp(dot(v, sun), -1.0, 1.0);
    float horizontal = sqrt(max(0.0, (1.0 - mu * mu) * (1.0 - mus * mus)));
    float azimuth = horizontal > 1e-5 ? clamp((nu - mu * mus) / horizontal, -1.0, 1.0) : 0.0;
    vec3 size = vec3(textureSize(u_scattering_rayleigh_lut, 0));
    float sun_count = size.x / float(u_scattering_azimuth_count);
    float index = (azimuth * 0.5 + 0.5) * float(u_scattering_azimuth_count - 1);
    float lower = min(floor(index), float(u_scattering_azimuth_count - 2));
    float fraction = index - lower;
    float x = 0.5 + scattering_sun_coord(mus) * (sun_count - 1.0);
    float half_height = size.y * 0.5;
    float y = (ground ? 0.0 : half_height) + 0.5
            + scattering_view_coord(max(r, u_scattering_bottom_km), mu, ground) * (half_height - 1.0);
    float h = sqrt(clamp((r - u_scattering_bottom_km)
        / (u_atmo_radius_km - u_scattering_bottom_km), 0.0, 1.0));
    float z = 0.5 + h * (size.z - 1.0);
    vec3 uv0 = vec3(lower * sun_count + x, y, z) / size;
    vec3 uv1 = uv0 + vec3(sun_count / size.x, 0.0, 0.0);
    R = mix(textureLod(u_scattering_rayleigh_lut, uv0, 0.0).rgb,
            textureLod(u_scattering_rayleigh_lut, uv1, 0.0).rgb, fraction);
    M = mix(textureLod(u_scattering_mie_lut, uv0, 0.0).rgb,
            textureLod(u_scattering_mie_lut, uv1, 0.0).rgb, fraction);
    MS = mix(textureLod(u_scattering_multiple_lut, uv0, 0.0).rgb,
             textureLod(u_scattering_multiple_lut, uv1, 0.0).rgb, fraction);
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

vec3 endpoint_transmittance(vec3 a, vec3 b) {
    if (length(b - a) < 1e-6) return vec3(1.0);
    vec3 v = normalize(b - a);
    bool ground = dot(a, v) / length(a) < scattering_horizon(max(length(a), u_scattering_bottom_km));
    vec3 tau = max(vec3(0.0), endpoint_tau(a, v, ground) - endpoint_tau(b, v, ground));
    vec3 R, M, extinction;
    endpoint_medium(0.5 * (a + b), R, M, extinction);
    tau = mix(extinction * length(b - a), tau, endpoint_long_weight(length(b - a)));
    return exp(-tau);
}

vec3 endpoint_radiance(vec3 a, vec3 b, vec3 sun, vec3 transmission) {
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
    endpoint_scattering(a, v, sun, ground, Ra, Ma, MSa);
    endpoint_scattering(b, v, sun, ground, Rb, Mb, MSb);
    vec3 result = max(vec3(0.0), (Ra - transmission * Rb) * phase_R
                             + (Ma - transmission * Mb) * phase_M + MSa - transmission * MSb);
    if (endpoint_long_weight(distance) < 1.0) {
        vec3 p = 0.5 * (a + b);
        float r = length(p);
        vec3 R, M, extinction;
        endpoint_medium(p, R, M, extinction);
        float sin_planet = u_planet_radius_km / max(r, u_planet_radius_km + 0.01);
        vec2 term = sun_terminator(dot(p, sun) / r, sin_planet, sqrt(max(0.0, 1.0 - sin_planet * sin_planet)),
                                  u_scattering_sun_radius, sqrt(max(0.0, 1.0 - u_scattering_sun_radius * u_scattering_sun_radius)));
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
