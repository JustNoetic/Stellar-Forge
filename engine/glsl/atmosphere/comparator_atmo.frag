#version 460 core
in vec2 v_xy_km;
layout(location=0,index=0) out vec4 out_scattering;
layout(location=0,index=1) out vec4 out_transmittance;
uniform vec3 u_center;
uniform vec3 u_pole;
uniform float u_oblateness;
uniform float u_depth_scale;
uniform vec2 u_viewport;
uniform int u_steps;
uniform float u_intensity;
uniform float u_mie_g;
uniform sampler2D u_scene_depth;
uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;

#include "common/scattering_bake.glsl"

const vec3 LIGHT = vec3(-0.77459667,0.38729833,0.5);
const vec3 RAY = vec3(0,0,-1);

vec3 stretch(vec3 p) {
    return p + u_pole*dot(p,u_pole)*(1.0/(1.0-u_oblateness)-1.0);
}
vec2 intersect(vec3 o, vec3 d, float radius) {
    float a = dot(d,d), b = dot(o,d);
    vec3 closest = o-d*(b/a);
    float delta = radius*radius-dot(closest,closest);
    if (delta < 0.0) return vec2(1e20,-1e20);
    float t = sqrt(delta/a);
    return vec2(-b/a-t,-b/a+t);
}
vec2 lookup_uv(float r, float mu) {
    float h = sqrt(clamp((r-u_planet_radius_km)/
        max(u_atmo_radius_km-u_planet_radius_km,1e-4),0.0,1.0));
    return vec2(0.5+0.5*sign(mu)*sqrt(abs(mu)),h);
}
vec3 sunlight(vec3 p) {
    vec3 s = stretch(LIGHT);
    vec3 d = normalize(s);
    vec2 ground = intersect(p,d,u_planet_radius_km);
    if (ground.x > 0.01 && ground.y > 0.01) return vec3(0);
    float r = length(p);
    return pow(max(textureLod(u_transmittance_lut,lookup_uv(r,dot(p,d)/r),0).rgb,
                   vec3(1e-30)),vec3(1.0/length(s)));
}
void main() {
    // Parallel rays have a different origin at every screen pixel, but share
    // one direction. No camera-to-fragment perspective normalization.
    vec3 origin = vec3(v_xy_km,u_atmo_radius_km);
    vec3 o = stretch(origin), d = stretch(RAY);
    vec2 interval = intersect(o,d,u_atmo_radius_km);
    if (interval.y <= interval.x) discard;
    vec2 ground = intersect(o,d,u_planet_radius_km);
    float start = max(interval.x,0.0), end = interval.y;
    bool surface = ground.x >= start && ground.x < end;
    if (surface) end = ground.x;
    float depth = texture(u_scene_depth,gl_FragCoord.xy/u_viewport).r;
    if (depth < 0.999999) {
        float world_z = (0.5-depth)*2.0*u_depth_scale;
        float stop = u_atmo_radius_km - (world_z-u_center.z)/1e-5;
        if (surface && stop < ground.x-0.01) surface = false;
        end = min(end,stop);
    } else {
        // Near a tessellated silhouette the analytic sphere can contain a
        // pixel with no opaque surface. Do not apply ground-beam extinction
        // to the sky background in that pixel.
        surface = false;
    }
    if (end <= start) discard;
    float distance = end-start;
    float closest = clamp(-dot(o+d*start,d)/dot(d,d),0.0,distance);
    float cos_theta = dot(RAY,LIGHT);
    float phase_r = 3.0*(1.0+cos_theta*cos_theta)/(16.0*3.14159265359);
    float g = u_mie_g;
    float phase_m = 3.0*(1.0-g*g)*(1.0+cos_theta*cos_theta)/
        (8.0*3.14159265359*(2.0+g*g)*pow(max(1.0+g*g-2.0*g*cos_theta,1e-4),1.5));
    vec3 trans = vec3(1), scatter = vec3(0);
    vec3 scaled_light = normalize(stretch(LIGHT));
    for (int i=0; i<u_steps; i++) {
        float a = scattering_station(float(i)/float(u_steps),distance,closest);
        float b = scattering_station(float(i+1)/float(u_steps),distance,closest);
        float ds = b-a;
        vec3 p = o + d*(start+0.5*(a+b));
        float r = length(p);
        vec3 rayleigh,mie,extinction;
        scattering_medium(r,rayleigh,mie,extinction);
        vec3 sun = sunlight(p);
        vec3 multi = textureLod(u_multi_scatter_lut,
                               lookup_uv(r,dot(p,scaled_light)/r),0).rgb;
        vec3 source = (rayleigh*phase_r+mie*phase_m)*sun + (rayleigh+mie)*multi;
        vec3 cell_trans = exp(-extinction*ds);
        vec3 integral = source*(vec3(1)-cell_trans)/max(extinction,vec3(1e-12));
        scatter += trans*integral;
        trans *= cell_trans;
        if (max(trans.r,max(trans.g,trans.b)) < 1e-5) break;
    }
    if (surface) trans *= sunlight(o+d*ground.x);
    out_scattering = vec4(scatter*u_intensity,0.0);
    out_transmittance = vec4(clamp(trans,0.0,1.0),1.0);
}
