// Depth spans only the atmospheric segment, never the vacuum before it.
// Cosine spacing resolves both nearby haze and the dense end of orbital rays.
uniform float u_aerial_bottom_km;

vec2 aerial_sphere(vec3 origin, vec3 direction, float radius) {
    float axial = dot(origin, direction);
    vec3 perpendicular = cross(origin, direction);
    float disc = radius * radius - dot(perpendicular, perpendicular);
    if (disc < 0.0) return vec2(1e20, -1e20);
    float half_chord = sqrt(disc);
    return vec2(-axial - half_chord, -axial + half_chord);
}

vec2 aerial_interval(vec3 origin, vec3 direction, out float family) {
    vec2 interval = aerial_sphere(origin, direction, u_atmo_radius_km);
    interval.x = max(0.0, interval.x);
    vec2 ground = aerial_sphere(origin, direction, u_aerial_bottom_km);
    family = -1.0;
    if (ground.x >= 0.0 && ground.x < interval.y) {
        interval.y = ground.x;
        family = 1.0;
    }
    return interval;
}

float aerial_distance_fraction(float q) {
    float s = sin(q * (0.5 * PI));
    return s * s;
}

float aerial_depth_coord(float fraction) {
    return (2.0 / PI) * asin(sqrt(clamp(fraction, 0.0, 1.0)));
}


// Orbital volume: separate ground/top ray families, with extra elevation
// samples near the horizon. Bounds cover only the current camera frustum.
uniform bool u_aerial_horizon;
uniform vec3 u_aerial_azimuth_axis;
uniform float u_aerial_azimuth_extent;
uniform vec2 u_aerial_mu_range;

float aerial_horizon_mu(vec3 camera) {
    float r = max(length(camera), u_aerial_bottom_km);
    return -sqrt(max(0.0, 1.0 - pow(u_aerial_bottom_km / r, 2.0)));
}

vec3 aerial_horizon_direction(vec3 camera, vec2 q, bool ground) {
    vec3 zenith = normalize(camera);
    vec3 y = cross(zenith, u_aerial_azimuth_axis);
    float phi = (2.0 * q.x - 1.0) * u_aerial_azimuth_extent;
    float mu = ground
        ? mix(aerial_horizon_mu(camera), u_aerial_mu_range.x, q.y * q.y) - 1e-6
        : mix(aerial_horizon_mu(camera), u_aerial_mu_range.y, q.y * q.y) + 1e-6;
    mu = clamp(mu, -1.0, 1.0);
    return mu * zenith + sqrt(max(0.0, 1.0 - mu * mu))
        * (cos(phi) * u_aerial_azimuth_axis + sin(phi) * y);
}

vec2 aerial_horizon_coord(vec3 camera, vec3 direction, bool ground) {
    vec3 zenith = normalize(camera);
    vec3 y = cross(zenith, u_aerial_azimuth_axis);
    float phi = atan(dot(direction, y), dot(direction, u_aerial_azimuth_axis));
    float mu = dot(zenith, direction), horizon = aerial_horizon_mu(camera);
    float range = ground ? horizon - u_aerial_mu_range.x : u_aerial_mu_range.y - horizon;
    float q = sqrt(clamp((ground ? horizon - mu : mu - horizon) / max(range, 1e-6), 0.0, 1.0));
    return vec2(0.5 + 0.5 * phi / u_aerial_azimuth_extent, q);
}
