// Ray geometry and altitude-aware terrain volume sampling.
vec2 aerialSphereIntersect(vec3 O, vec3 V, float radius) {
    float a = dot(V, V);
    float p2 = dot(cross(O, V), cross(O, V)) / a;
    if (p2 > radius * radius) return vec2(1e10, -1e10);
    float tc = -dot(O, V) / a;
    float d = sqrt(max(0.0, (radius * radius - p2) / a));
    return vec2(tc - d, tc + d);
}

vec2 aerialRaySpan(vec3 O, vec3 V, float outer_radius) {
    vec2 span = aerialSphereIntersect(O, V, outer_radius);
    span.x = max(0.0, span.x);
    return span;
}

#include "common/aerial_volume_sampling.glsl"
#include "common/aerial_volume_refinement.glsl"
