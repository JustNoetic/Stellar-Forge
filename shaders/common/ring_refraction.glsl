#ifndef RING_REFRACTION_GLSL
#define RING_REFRACTION_GLSL

#include "refraction.glsl"

bool ring_refracted_sample(vec3 anchor, vec3 pole, vec3 view_ray, float range, vec3 apparent_flat,
                           vec3 away, float bend, out vec3 hit, out vec3 direction, out float distance, out float residual) {
    vec3 true_flat = apparent_flat*cos(bend)+away*sin(bend);
    direction = normalize(refraction_from_atmosphere(true_flat));
    float denominator = dot(direction, pole);
    if (abs(denominator) < 1e-8) return false;
    vec3 origin = anchor+range*(direction-view_ray);
    float offset = -dot(origin, pole)/denominator;
    distance = range+offset;
    if (distance <= 0.0) return false;
    hit = origin+direction*offset;
    vec3 up, tangent;
    vec2 sample_value = refraction_lookup(true_flat, distance*length(refraction_to_atmosphere(direction))*u_refraction.escape.z, up, tangent);
    residual = bend-sample_value.x;
    return true;
}

bool ring_refracted_hit_inverse(vec3 anchor, vec3 center, vec3 pole, float pixel_angle,
                        out vec3 hit, out vec3 direction, out vec3 apparent) {
    vec3 world = center+anchor;
    float range = length(world);
    vec3 view_ray = world/max(range, 1e-20);
    float denominator = dot(view_ray, pole);
    if (abs(denominator) < 1e-8) return false;
    float offset = -dot(anchor, pole)/denominator;
    hit = anchor+view_ray*offset;
    direction = view_ray;
    apparent = world;
    if (!refraction_active()) return true;
    vec3 apparent_flat = normalize(refraction_to_atmosphere(view_ray));
    vec3 up, tangent;
    vec2 initial = refraction_lookup(apparent_flat, range*length(refraction_to_atmosphere(view_ray))*u_refraction.escape.z, up, tangent);
    float tolerance = max(1e-9, pixel_angle*0.03);
    if (abs(initial.x) < tolerance || length(tangent) < 1e-7) return true;
    vec3 away = -tangent;
    float sign_value = initial.x < 0.0 ? -1.0 : 1.0;
    float lo = 0.0, hi = u_refraction.profile.w;
    float flo = -abs(initial.x), fhi = hi;
    float best_error = abs(initial.x), best_distance = range;
    vec3 best_hit = hit, best_direction = direction;
    int side = 0;
    for (int i = 0; i < 32; ++i) {
        float bend = i == 0 ? hi : lo+(hi-lo)*clamp(-flo/max(fhi-flo, 1e-20), 0.05, 0.95);
        if ((i % 4) == 3) bend = 0.5*(lo+hi);
        vec3 candidate_hit, candidate_direction;
        float candidate_distance, residual;
        bool valid = ring_refracted_sample(anchor, pole, view_ray, range, apparent_flat, away, sign_value*bend,
                                           candidate_hit, candidate_direction, candidate_distance, residual);
        if (!valid) { hi = bend; fhi = hi; continue; }
        residual *= sign_value;
        if (abs(residual) < best_error) {
            best_error = abs(residual);
            best_hit = candidate_hit;
            best_direction = candidate_direction;
            best_distance = candidate_distance;
        }
        if (best_error <= tolerance) break;
        if (residual > 0.0) {
            hi = bend; fhi = residual;
            if (side == 1) flo *= 0.5;
            side = 1;
        } else {
            lo = bend; flo = residual;
            if (side == -1) fhi *= 0.5;
            side = -1;
        }
    }
    hit = best_hit;
    direction = best_direction;
    apparent = view_ray*best_distance;
    return best_error <= max(pixel_angle*0.25, 1e-7);
}

layout(set = 0, binding = 22) uniform sampler2D u_refraction_travel;
layout(set = 0, binding = 23) uniform sampler2D u_refraction_rows;
layout(set = 0, binding = 24) uniform sampler2D u_refraction_profile;

bool ring_refracted_hit(vec3 anchor, vec3 center, vec3 pole, float pixel_angle,
                        out vec3 hit, out vec3 direction, out vec3 apparent) {
    vec3 world = center+anchor;
    float range = length(world);
    vec3 view_ray = world/max(range, 1e-20);
    float denominator = dot(view_ray, pole);
    if (abs(denominator) < 1e-8) return false;
    hit = anchor+view_ray*(-dot(anchor, pole)/denominator);
    direction = view_ray;
    apparent = world;
    if (!refraction_active()) return true;
    RefractionGPU g = refraction_parameters();
    vec3 v = normalize(refraction_to_atmosphere(view_ray));
    vec3 up = normalize(refraction_to_atmosphere(-g.host.xyz));
    float mu = clamp(dot(v, up), -1.0, 1.0);
    vec3 side = up-v*mu;
    float sine = length(side);
    if (sine < 1e-7) return mu >= 0.0 || range*g.escape.z <= g.frame.x;
    float radius = g.frame.y, camera_radius = radius+g.frame.x;
    float b = camera_radius*sine;
    float q = g.frame.w > 0.5 ? b-radius : refraction_invariant(g, atan(sine, mu));
    if (g.frame.w > 0.5) {
        if (mu >= 0.0 || q >= g.frame.z) return true;
        float entry = -camera_radius*mu-sqrt(max((g.frame.z-q)*(2.0*radius+g.frame.z+q), 0.0));
        float target = range*length(refraction_to_atmosphere(view_ray))*g.escape.z;
        if (target <= entry) return true;
    }
    bool inward = mu < 0.0;
    bool covered = q >= g.model.x;
    if (q > g.escape.x || (inward && (!covered || refraction_turn(u_refraction_rows, u_refraction_profile, g, q) < 0.0))) return false;
    float top_travel = refraction_table_travel(u_refraction_travel, u_refraction_rows, u_refraction_profile, g, q, g.frame.z);
    float camera_travel = refraction_table_travel(u_refraction_travel, u_refraction_rows, u_refraction_profile, g, q, g.frame.x);
    float polar = covered ? (inward ? top_travel+camera_travel : top_travel-camera_travel)
                          : refraction_direct(u_refraction_profile, g, q, g.frame.x, g.frame.z);
    vec3 azimuth = normalize(v-up*mu);
    vec3 normal_exit = up*cos(polar)+azimuth*sin(polar);
    vec3 tangent_exit = -up*sin(polar)+azimuth*cos(polar);
    float top_radius = radius+g.frame.z;
    float exit_sine = (radius+q)/top_radius;
    float exit_cosine = sqrt(max((g.frame.z-q)*(2.0*radius+g.frame.z+q), 0.0))/top_radius;
    direction = normalize(refraction_from_atmosphere(normal_exit*exit_cosine+tangent_exit*exit_sine));
    float along = -camera_radius*mu;
    float reference = along*length(refraction_from_atmosphere(v))/g.escape.z;
    vec3 correction = refraction_from_atmosphere(top_radius*normal_exit-b*side/sine)/g.escape.z;
    vec3 local_origin = anchor+(range-reference)*(direction-view_ray)+correction;
    denominator = dot(direction, pole);
    if (abs(denominator) < 1e-8) return false;
    float offset = -dot(local_origin, pole)/denominator;
    if (range-reference+offset < 0.0)
        return ring_refracted_hit_inverse(anchor, center, pole, pixel_angle, hit, direction, apparent);
    hit = local_origin+direction*offset;
    apparent = view_ray*length(center+hit);
    return true;
}

#endif
