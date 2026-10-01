// Position-queryable atmosphere coordinates. Each half of the view axis has
// its own boundary (ground/top), so filtering never mixes the two ray families.
uniform float u_scattering_bottom_km;

float scattering_radius(float q) {
    return mix(u_scattering_bottom_km, u_atmo_radius_km, q * q);
}

float scattering_horizon(float r) {
    return -sqrt(max(0.0, 1.0 - pow(u_scattering_bottom_km / r, 2.0)));
}

// Distance to the outer boundary resolves the thin orbital limb. At the
// atmosphere's top, all upward rays have zero columns, so allocating most
// of the angular axis to them would waste the table's view samples.
vec2 scattering_top_distances(float r) {
    float rho = sqrt(max(0.0, r*r - u_scattering_bottom_km*u_scattering_bottom_km));
    float H = sqrt(max(0.0, u_atmo_radius_km*u_atmo_radius_km
        - u_scattering_bottom_km*u_scattering_bottom_km));
    return vec2(u_atmo_radius_km - r, rho + H);
}

float scattering_view_cos(float r, float q, bool ground) {
    float horizon = scattering_horizon(r);
    if (ground) return mix(-1.0, horizon, 1.0 - q * q);
    vec2 range = scattering_top_distances(r);
    float d = mix(range.y, range.x, q*q);
    return d < 1e-6 ? 1.0 : clamp((u_atmo_radius_km*u_atmo_radius_km
        - r*r - d*d) / (2.0*r*d), horizon, 1.0);
}

float scattering_view_coord(float r, float mu, bool ground) {
    r = clamp(r, u_scattering_bottom_km, u_atmo_radius_km);
    float horizon = scattering_horizon(r);
    if (ground) return sqrt(clamp((horizon - mu) / max(1e-7, horizon + 1.0), 0.0, 1.0));
    vec2 range = scattering_top_distances(r);
    float b = r * mu;
    float d = max(0.0, -b + sqrt(max(0.0, b*b
        + u_atmo_radius_km*u_atmo_radius_km - r*r)));
    return sqrt(clamp((range.y - d) / max(1e-6, range.y - range.x), 0.0, 1.0));
}

float scattering_boundary_distance(float r, float mu, bool ground) {
    float b = r * mu;
    float boundary = ground ? u_scattering_bottom_km : u_atmo_radius_km;
    float disc = max(0.0, boundary * boundary - r * r + b * b);
    return max(0.0, -b + (ground ? -sqrt(disc) : sqrt(disc)));
}

float scattering_sun_cos(float q) {
    float x = q * 2.0 - 1.0;
    return sign(x) * x * x;
}

float scattering_sun_coord(float mu) {
    return 0.5 + 0.5 * sign(mu) * sqrt(abs(mu));
}

// Parameterize light direction at the lowest point of the view ray, where
// most scattering occurs. The 2D multiple-scattering map retains its mapping.
vec3 scattering_light_anchor(float r, float mu, bool ground) {
    float distance = scattering_boundary_distance(r, mu, ground);
    float closest = clamp(-r * mu, 0.0, distance);
    float rc = sqrt(max(1e-8, r * r + closest * (2.0 * r * mu + closest)));
    // Radius, view zenith cosine and station at the reference point.
    return vec3(rc, (r * mu + closest) / rc, closest);
}

float scattering_light_horizon(float r) {
    return -sqrt(max(0.0, 1.0 - pow(u_planet_radius_km / max(r, u_planet_radius_km + 0.01), 2.0)));
}

float scattering_light_cos(float r, float q) {
    float horizon = scattering_light_horizon(r), x = 2.0 * q - 1.0;
    // A sinh warp resolves the narrow sunset interval on either side of the
    // reference horizon while still covering every light direction.
    return horizon + sign(x) * sinh(6.0 * abs(x)) / sinh(6.0)
        * (x < 0.0 ? 1.0 + horizon : 1.0 - horizon);
}

float scattering_light_coord(float r, float mu) {
    float horizon = scattering_light_horizon(r), delta = mu - horizon;
    return 0.5 + 0.5 * sign(delta) * asinh(clamp(abs(delta)
        / (delta < 0.0 ? 1.0 + horizon : 1.0 - horizon), 0.0, 1.0) * sinh(6.0)) / 6.0;
}

float scattering_azimuth_cos(float q) { return 2.0 * q - 1.0; }
float scattering_azimuth_coord(float cosine) { return 0.5 + 0.5 * clamp(cosine, -1.0, 1.0); }
