#ifndef REFRACTION_GLSL
#define REFRACTION_GLSL

#include "refraction_solve.glsl"

layout(std140, set = 0, binding = 20) uniform RefractionParameters {
    vec4 host;
    vec4 pole;
    vec4 frame;
    vec4 mapping;
    vec4 model;
    vec4 profile;
    vec4 escape;
} u_refraction;

layout(set = 0, binding = 21) uniform sampler2D u_refraction_observer;

RefractionGPU refraction_parameters() {
    return RefractionGPU(u_refraction.host, u_refraction.pole, u_refraction.frame, u_refraction.mapping,
                         u_refraction.model, u_refraction.profile, u_refraction.escape);
}

bool refraction_active() { return u_refraction.host.w > 0.5; }

vec3 refraction_to_atmosphere(vec3 v) {
    return v+u_refraction.pole.xyz*(dot(v, u_refraction.pole.xyz)*u_refraction.pole.w);
}

vec3 refraction_from_atmosphere(vec3 v) {
    return v-u_refraction.pole.xyz*(dot(v, u_refraction.pole.xyz)*u_refraction.pole.w/(1.0+u_refraction.pole.w));
}

vec2 refraction_lookup(vec3 direction, float distance_km, out vec3 up, out vec3 tangent) {
    RefractionGPU g = refraction_parameters();
    up = normalize(refraction_to_atmosphere(-u_refraction.host.xyz));
    float mu = clamp(dot(direction, up), -1.0, 1.0);
    vec3 side = up-direction*mu;
    float sine = length(side);
    tangent = sine > 1e-7 ? side/sine : vec3(0.0);
    float theta = atan(sine, mu);
    float camera_radius = g.frame.y+g.frame.x;
    float half_angle = sin(0.5*(0.5*REFRACTION_PI-theta));
    float impact = g.frame.w > 0.5 ? camera_radius*sine-g.frame.y : g.frame.x-2.0*camera_radius*half_angle*half_angle;
    if (sine <= 1e-7 || refraction_unaffected(g, theta, impact)) return vec2(0.0, 1.0);
    return refraction_observer_sample(u_refraction_observer, g, vec3(theta, impact, -camera_radius*mu), distance_km);
}

vec3 refract_position(vec3 position) {
    if (!refraction_active()) return position;
    vec3 a = refraction_to_atmosphere(position);
    float d = length(a);
    if (d <= 0.0) return position;
    vec3 up, tangent;
    vec3 v = a/d;
    vec2 sample_value = refraction_lookup(v, d*u_refraction.escape.z, up, tangent);
    if (sample_value.x == 0.0) return position;
    vec3 apparent = v*cos(sample_value.x)+tangent*sin(sample_value.x);
    return refraction_preserve_range(position, apparent, u_refraction.pole);
}

vec4 refract_direction(vec3 direction) {
    if (!refraction_active()) return vec4(direction, 1.0);
    vec3 v = normalize(refraction_to_atmosphere(direction));
    vec3 up, tangent;
    vec2 sample_value = refraction_lookup(v, 2.0*REFRACTION_FAR, up, tangent);
    vec3 apparent = v*cos(sample_value.x)+tangent*sin(sample_value.x);
    float theta = acos(clamp(dot(v, up), -1.0, 1.0));
    vec3 probe = v*cos(1e-4)-tangent*sin(1e-4);
    vec3 probe_up, probe_tangent;
    vec2 next = refraction_lookup(probe, 2.0*REFRACTION_FAR, probe_up, probe_tangent);
    float compression = clamp(1.0-(next.x-sample_value.x)/1e-4, 0.0, 4.0);
    float spread = sin(max(theta-sample_value.x, 1e-6))/max(sin(theta), 1e-6);
    float flux = length(tangent) > 0.0 ? compression*spread : 1.0;
    return vec4(normalize(refraction_from_atmosphere(apparent)), flux*clamp(sample_value.y, 0.0, 1.0));
}

vec4 refract_point(vec3 position) {
    if (!refraction_active()) return vec4(position, 1.0);
    vec3 a = refraction_to_atmosphere(position);
    float d = length(a);
    if (d <= 0.0) return vec4(position, 1.0);
    vec3 up, tangent;
    vec3 v = a/d;
    float distance_km = d*u_refraction.escape.z;
    vec2 sample_value = refraction_lookup(v, distance_km, up, tangent);
    vec3 apparent = v*cos(sample_value.x)+tangent*sin(sample_value.x);
    float theta = acos(clamp(dot(v, up), -1.0, 1.0));
    vec3 probe = v*cos(1e-4)-tangent*sin(1e-4);
    vec3 probe_up, probe_tangent;
    vec2 next = refraction_lookup(probe, distance_km, probe_up, probe_tangent);
    float compression = clamp(1.0-(next.x-sample_value.x)/1e-4, 0.0, 4.0);
    float spread = sin(max(theta-sample_value.x, 1e-6))/max(sin(theta), 1e-6);
    float flux = length(tangent) > 0.0 ? compression*spread : 1.0;
    vec3 world = sample_value.x == 0.0 ? position : refraction_preserve_range(position, apparent, u_refraction.pole);
    return vec4(world, flux*clamp(sample_value.y, 0.0, 1.0));
}

#endif
