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
