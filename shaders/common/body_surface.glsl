mat3 body_surface_rotation(vec3 pole, float spin) {
    vec3 reference = abs(pole.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 x = normalize(reference-pole*dot(reference, pole));
    vec3 z = cross(x, pole);
    float c = cos(spin), s = sin(spin);
    return mat3(x*c-z*s, pole, x*s+z*c);
}
