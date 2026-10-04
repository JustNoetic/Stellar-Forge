// Geometric penumbra and atmospheric refraction shared by all body shadows.
// footprint, radius, perpendicular distance and caster distance use the SAME
// units (AU for surfaces, km for atmospheric marches). units_to_km converts them.
// grazing.xyz is CPU-precomputed limb optical depth; grazing.w is inverse
// reference radius in km. Only oblate casters need a projected-radius correction.
// Thickness belongs to conservative bounds; scale height belongs to optics.
vec3 casterShadowTerm(float footprint, float radius, float perpendicular,
                      float max_bend, vec4 grazing, vec4 ozone,
                      float scale_height_km, float caster_distance,
                      float units_to_km) {
    float outer = radius + footprint;

    float max_occ = 1.0;
    if (radius < footprint) {
        float ratio = radius / footprint;
        max_occ = ratio * ratio;
    }
    // Clamp the cubic directly: mixed pixels avoid divergent plateau branches.
    // The tiny distance-unit floor also makes a point-source boundary well-defined.
    float t = clamp((outer - perpendicular) / max(2.0 * min(radius, footprint), 1e-30), 0.0, 1.0);
    float blend = t * t * (3.0 - 2.0 * t);
    float transmission = clamp(1.0 - max_occ * blend, 0.0, 1.0);
    // Preserve the existing artistic 0.625 power.
    vec3 shadow = vec3(pow(transmission, 0.625));
    if (max_bend <= 1e-6 || perpendicular >= outer) return shadow;

    float distance_km = max(caster_distance * units_to_km, 1e-6);
    float req_bend = (radius - perpendicular - footprint * 0.8)
        / max(caster_distance, 1e-30);
    // Rays needing a larger bend hit the solid body; skip all optical work.
    if (req_bend >= max_bend) return shadow;

    float H = max(scale_height_km, 0.1);
    float sigma_z = max(0.707 * H, 0.1);
    float radius_km = max(radius * units_to_km, 100.0);
    vec3 tau_grazing = grazing.xyz;
    float radius_ratio = radius_km * grazing.w;
    if (abs(radius_ratio - 1.0) > 1e-6)
        tau_grazing *= sqrt(max(radius_ratio, 0.0));

    float depth = clamp(max(0.0, req_bend) / max_bend, 0.0, 1.0);
    float altitude_km = -H * log(max(depth, 1e-5));
    float ozone_height = (altitude_km - ozone.w) / sigma_z;
    vec3 tau = tau_grazing * depth
        + ozone.xyz * exp(-0.5 * ozone_height * ozone_height);
    vec3 transmittance = exp(-tau);

    float defocus = 1.0 / (1.0 + distance_km * max(req_bend, 1e-6) / H);
    float focal_distance = radius_km / max_bend;
    float focal_ratio = clamp(distance_km / focal_distance, 0.0, 1.0);
    float off_axis = clamp(footprint / max(1e-6 * caster_distance, perpendicular), 0.0, 1.0);
    float annular = 2.0 * H / max(footprint * units_to_km, 1e-9);
    float intensity = clamp(annular * focal_ratio * off_axis, 0.0, 1.0) * defocus;
    float surface_fade = 1.0 - smoothstep(0.75, 1.0, depth);
    shadow += transmittance * intensity * (1.0 - depth * 0.7) * surface_fade * blend;
    return clamp(shadow, vec3(0.0), vec3(1.0));
}
