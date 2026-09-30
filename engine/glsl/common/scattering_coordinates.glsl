// Position-queryable atmosphere coordinates. Each half of the view axis has
// its own boundary (ground/top), so filtering never mixes the two ray families.
uniform float u_scattering_bottom_km;

float scattering_radius(float q) {
    return mix(u_scattering_bottom_km, u_atmo_radius_km, q * q);
}

float scattering_horizon(float r) {
    return -sqrt(max(0.0, 1.0 - pow(u_scattering_bottom_km / r, 2.0)));
}

float scattering_view_cos(float r, float q, bool ground) {
    float horizon = scattering_horizon(r);
    return ground ? mix(-1.0, horizon, 1.0 - q * q)
                  : mix(horizon, 1.0, q * q);
}

float scattering_view_coord(float r, float mu, bool ground) {
    float horizon = scattering_horizon(r);
    return ground ? sqrt(clamp((horizon - mu) / max(1e-7, horizon + 1.0), 0.0, 1.0))
                  : sqrt(clamp((mu - horizon) / max(1e-7, 1.0 - horizon), 0.0, 1.0));
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
