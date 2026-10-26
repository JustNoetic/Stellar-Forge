const ivec2 STELLAR_SOURCE_SIZE = ivec2(128, 64);

float stellar_source_begin(float horizon, float radius) {
    float c = sqrt(max(1.0-horizon*horizon, 0.0));
    return c*cos(radius)+horizon*sin(radius) > 0.0
        ? horizon*cos(radius)-c*sin(radius) : -1.0;
}

vec2 stellar_source_uv(vec3 p, vec3 sun, float radius) {
    float r = length(p), mu = dot(p, sun)/r;
    float first = stellar_source_begin(sun_horizon(p), radius);
    float x = sqrt(clamp((mu-first)/(1.0-first), 0.0, 1.0));
    float y = sqrt(max((r-bottom_radius())*(r+bottom_radius()), 0.0)
                  /((top_radius()-bottom_radius())*(top_radius()+bottom_radius())));
    return (0.5+clamp(vec2(x, y), 0.0, 1.0)*vec2(STELLAR_SOURCE_SIZE-1))/vec2(STELLAR_SOURCE_SIZE);
}

#ifndef ATMOSPHERE_SOURCE_BAKE
layout(set = 0, binding = 10) uniform sampler2DArray stellar_source_lut;

vec3 cached_stellar_source(vec3 p, vec3 view, vec3 sun, vec3 rayleigh, vec3 medium_mie, vec2 uv) {
    vec3 total = textureLod(stellar_source_lut, vec3(uv, 0.0), 0.0).rgb;
    total *= total;
    total *= total;
    float phase = dot(view, sun);
    vec3 source;
#ifdef ATMOSPHERE_SECONDARY_BAKE
    if (stellar_cache_radius() < 0.01) source = rayleigh*rayleigh_phase(phase);
    else
#endif
    {
        vec3 up = normalize(p);
        vec3 tangent = stellar_tangent(up, sun), bitangent = cross(up, tangent);
        vec3 v = vec3(dot(view, up), dot(view, tangent), dot(view, bitangent));
        vec3 uu = textureLod(stellar_source_lut, vec3(uv, 1.0), 0.0).rgb;
        vec3 tt = textureLod(stellar_source_lut, vec3(uv, 2.0), 0.0).rgb;
        vec3 ut = textureLod(stellar_source_lut, vec3(uv, 3.0), 0.0).rgb;
        vec3 second = v.x*v.x*uu+v.y*v.y*tt+v.z*v.z*max(vec3(1.0)-uu-tt, vec3(0.0))+2.0*v.x*v.y*ut;
        source = rayleigh*(3.0/(16.0*ATM_PI))*max(vec3(1.0)+second, vec3(0.0));
    }
#ifdef ATMOSPHERE_RAYLEIGH_ONLY
    return total*source;
#else
    if (atm.aerosol_count.y > 0.5) {
        source += atm.aerosol_count.x < 0.5 ? medium_mie*mie_phase(phase) : aerosol_source(p, phase);
    }
    return total*source;
#endif
}
#endif
