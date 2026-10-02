#include "aerial_coordinates.glsl"
uniform bool u_aerial_enabled;
uniform vec3 u_aerial_camera_sph;
uniform sampler3D u_aerial_scatter_lut;
uniform sampler3D u_aerial_tau_lut;

bool aerial_lookup(vec3 a, vec3 b, vec3 world_ray, out vec3 L, out vec3 T, out float weight) {
    if (!u_aerial_enabled) return false;
    // Deflected rays with a displaced origin do not belong to this camera's
    // volume. Ring-split passes are handled by the caller using exact endpoints.
    vec3 direction = normalize(toSphericalSpace(world_ray, u_pole_obl));
    float family;
    vec2 interval = aerial_interval(u_aerial_camera_sph, direction, family);
    float span = interval.y - interval.x;
    if (span <= 1e-5) return false;
    vec3 start = u_aerial_camera_sph + interval.x * direction;
    if (length(a - start) > 0.02) return false;
    float distance = dot(b - start, direction);
    if (distance < 0.0 || distance > span) return false;
    // Even cells wholly inside the ground disk can vary rapidly near its rim.
    // Blend to endpoints at grazing orbital angles instead of drawing a seam
    // where the coarse angular footprint stops resolving the dense atmosphere.
    float thickness = u_atmo_radius_km - u_planet_radius_km;
    float orbital = smoothstep(0.5 * thickness, thickness,
        length(u_aerial_camera_sph) - u_planet_radius_km);
    float incidence = -dot(normalize(b), direction);
    weight = mix(1.0, smoothstep(0.3, 0.55, incidence), orbital * max(family, 0.0));
    weight *= 1.0 - smoothstep(3.0 * u_atmo_radius_km, 4.0 * u_atmo_radius_km,
        length(u_aerial_camera_sph));
    if (weight <= 0.0) return false;
    vec4 projected = projection * view * vec4(world_ray, 0.0);
    if (projected.w <= 0.0) return false;
    vec2 uv = projected.xy / projected.w * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return false;
    vec3 size = vec3(textureSize(u_aerial_scatter_lut, 0));

    // Fade out before a filtering footprint reaches either horizon, at ALL
    // camera altitudes. A hard per-cell family rejection alone exposes the
    // 32x32 grid as bright/dark rectangles when viewing a shallow horizon.
    // Use the analytic ray derivative, not screen derivatives inside this
    // divergent branch, so the margin follows FOV, aspect and oblateness.
    vec4 eye = u_inv_proj * vec4(uv * 2.0 - 1.0, -1.0, 1.0);
    vec3 ray = toSphericalSpace((u_inv_view * vec4(eye.xy, -1.0, 0.0)).xyz, u_pole_obl);
    vec3 dx = toSphericalSpace((u_inv_view * vec4(2.0 * u_inv_proj[0].xy, 0.0, 0.0)).xyz, u_pole_obl);
    vec3 dy = toSphericalSpace((u_inv_view * vec4(2.0 * u_inv_proj[1].xy, 0.0, 0.0)).xyz, u_pole_obl);
    float radius = length(u_aerial_camera_sph);
    vec3 zenith = u_aerial_camera_sph / max(radius, 1e-6);
    float mu = dot(zenith, direction);
    vec3 gradient = (zenith - mu * direction) / max(length(ray), 1e-6);
    float footprint = abs(dot(gradient, dx)) / (size.x - 1.0)
                    + abs(dot(gradient, dy)) / (size.y - 1.0);
    float datum_horizon = -sqrt(max(0.0, 1.0 - pow(u_planet_radius_km / max(radius, u_planet_radius_km), 2.0)));
    float bottom_horizon = -sqrt(max(0.0, 1.0 - pow(u_aerial_bottom_km / max(radius, u_aerial_bottom_km), 2.0)));
    float horizon_distance = min(abs(mu - datum_horizon), abs(mu - bottom_horizon));
    weight *= smoothstep(1.5, 3.0, horizon_distance / max(footprint, 1e-6));
    if (weight <= 0.0) return false;

    vec3 coord = (vec3(uv, aerial_depth_coord(distance / span)) * (size - 1.0) + 0.5) / size;
    vec4 tau = textureLod(u_aerial_tau_lut, coord, 0.0);
    // Never interpolate across ground/sky or atmosphere/vacuum boundaries.
    // Exact endpoints in these narrow strips protect the orbital limb.
    if (abs(tau.a - family) > 0.001) return false;
    vec3 radiance = textureLod(u_aerial_scatter_lut, coord, 0.0).rgb;
    L = radiance * radiance;
    T = exp(-tau.rgb * tau.rgb);
    return true;
}
