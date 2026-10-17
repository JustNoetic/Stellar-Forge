dvec3 terrain_cube_vector(uint face, dvec2 uv) {
    switch (face) {
    case 0u: return dvec3(1.0, uv.y, -uv.x);
    case 1u: return dvec3(-1.0, uv.y, uv.x);
    case 2u: return dvec3(uv.x, 1.0, -uv.y);
    case 3u: return dvec3(uv.x, -1.0, uv.y);
    case 4u: return dvec3(uv.x, uv.y, 1.0);
    default: return dvec3(-uv.x, uv.y, -1.0);
    }
}

dvec3 terrain_surface(uint face, dvec2 uv, double polar_ratio, double amplitude) {
    dvec3 direction = normalize(terrain_cube_vector(face, uv));
    dvec3 base = direction * dvec3(1.0, polar_ratio, 1.0);
    dvec3 normal = normalize(direction / dvec3(1.0, polar_ratio, 1.0));
    vec3 d = vec3(direction);
    float relief = 0.55 * sin(18.0 * d.x + 5.0 * d.y) * cos(14.0 * d.z)
                 + 0.30 * sin(37.0 * d.y - 11.0 * d.z) * cos(23.0 * d.x)
                 + 0.15 * sin(61.0 * d.z + 19.0 * d.x) * cos(47.0 * d.y);
    return base + normal * (amplitude * double(relief));
}
