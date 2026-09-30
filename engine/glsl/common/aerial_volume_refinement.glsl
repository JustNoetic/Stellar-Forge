// The planet's sunlight horizon varies sharply across coarse frustum columns.
// Resolve the twilight band on the actual pixel ray; fade into the cached volume
// outside that band so refinement never introduces a visible selection edge.
float aerialGroundFootprint(vec2 uv, float distance_km, vec3 O, vec3 V,
    ivec2 size, mat4 inv_proj, mat4 inv_view, vec4 pole_scale)
{
    vec3 endpoint = O + distance_km * V;
    float radius = length(endpoint);
    float footprint = 0.0;
    for (int axis = 0; axis < 2; axis++) {
        vec2 offset = axis == 0 ? vec2(1.0 / float(size.x), 0.0) : vec2(0.0, 1.0 / float(size.y));
        for (int side = -1; side <= 1; side += 2) {
            vec3 neighbor_V = aerialColumnRay(clamp(uv + float(side) * offset, 0.0, 1.0),
                inv_proj, inv_view, pole_scale);
            vec3 neighbor = O + aerialRadiusDistance(O, neighbor_V, radius) * neighbor_V;
            footprint = max(footprint, length(normalize(neighbor) - normalize(endpoint)));
        }
    }
    return footprint;
}

float aerialTerminatorRefinement(
    float distance_km, vec3 O, vec3 V, vec4 light, vec3 radii_height, float footprint)
{
    float R = radii_height.x;
    vec3 endpoint = O + distance_km * V;
    float radius = length(endpoint);
    // Six density scale heights include >99% of the vertical scattering mass.
    float dense_radius = min(radii_height.y, max(radius, R + 6.0 * radii_height.z));
    float dense_start = aerialRadiusDistance(O, V, min(length(O), dense_radius));
    vec3 P0 = O + min(distance_km, dense_start) * V;
    vec3 L = normalize(light.xyz);
    float r0 = max(length(P0), R + 0.01);
    float r1 = max(radius, R + 0.01);
    float margin0 = dot(normalize(P0), L) + sqrt(max(0.0, 1.0 - R * R / (r0 * r0)));
    float margin1 = dot(normalize(endpoint), L) + sqrt(max(0.0, 1.0 - R * R / (r1 * r1)));
    float distance_to_band = max(min(margin0, margin1), -max(margin0, margin1));
    // Include a full coarse-cell footprint and the refracted stellar disc.
    // Near orbit this expands with the projected cell size, not screen pixels.
    float width = max(0.01, footprint + light.w);
    return 1.0 - smoothstep(width, 2.0 * width, distance_to_band);
}

// Retain the standalone sampling API used by volume validation shaders.
float aerialTerminatorRefinement(
    vec2 uv, float distance_km, vec3 O, vec3 V, vec4 light,
    vec3 radii_height, ivec2 size, mat4 inv_proj, mat4 inv_view, vec4 pole_scale)
{
    return aerialTerminatorRefinement(distance_km, O, V, light, radii_height,
        aerialGroundFootprint(uv, distance_km, O, V, size, inv_proj, inv_view, pole_scale));
}
