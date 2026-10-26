#include "stellar_disk.glsl"
layout(set = 0, binding = 12) uniform sampler2DArray surface_atmosphere_lut;
layout(std430, set = 0, binding = 13) readonly buffer SurfaceAtmosphereRecords { vec4 surface_atmospheres[]; };

struct SurfaceAtmosphereSample {
    vec3 up;
    vec3 pole;
    vec3 height_layer_scale;
    float flattening;
    vec2 radii;
};

SurfaceAtmosphereSample surface_atmosphere_sample(uint body, vec3 local, vec3 base_chord, vec3 normal) {
    vec4 record = surface_atmospheres[body];
    BodyInstanceGPU b = bodies[body];
    vec3 pole = body_pole(b);
    float flattening = 1.0/max(1.0-b.pole_oblateness.w, 0.05)-1.0;
    vec3 p = local+pole*dot(local, pole)*flattening;
    float r = length(p)*149597870.7;
    float sag = (1.0-length(base_chord))*record.z;
    r += max(sag, 0.0);
    float y = sqrt(clamp((r-record.x)/(record.y-record.x), 0.0, 1.0));
    float scale = 0.5+0.5*max(dot(normal, normalize(local)), 0.0);
    return SurfaceAtmosphereSample(normalize(p), pole, vec3((0.5+y*31.0)/32.0, (record.w-1.0)*3.0, scale), flattening, vec2(record.x, r));
}

vec3 surface_atmosphere_coordinates(SurfaceAtmosphereSample lookup, vec3 sun) {
    vec3 light = normalize(sun+lookup.pole*dot(sun, lookup.pole)*lookup.flattening);
    float mu = clamp(dot(lookup.up, light), -1.0, 1.0);
    float x = sign(mu)*sqrt(abs(mu))*0.5+0.5;
    return vec3((0.5+x*127.0)/128.0, lookup.height_layer_scale.xy);
}

bool surface_stellar_horizon_clear(SurfaceAtmosphereSample lookup, vec3 sun, float radius) {
    float bottom_squared = lookup.radii.x*lookup.radii.x;
    float horizon = -sqrt(max(1.0-bottom_squared/max(lookup.radii.y*lookup.radii.y, bottom_squared), 0.0));
    vec3 light = normalize(sun+lookup.pole*dot(sun, lookup.pole)*lookup.flattening);
    return abs(dot(lookup.up, light)-horizon) > 2.0*radius;
}

vec3 surface_small_stellar_light(SurfaceAtmosphereSample lookup, vec3 N, vec3 sun, float radius, float brightness) {
    float elevation = asin(clamp(dot(N, sun), -1.0, 1.0));
    vec2 range = stellar_elevation_range(elevation, radius, 0.0);
    if (range.y <= range.x) return vec3(0.0);
    vec3 tangent = stellar_tangent(N, sun);
    vec3 direct = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        vec3 slice = stellar_slice(range, elevation, radius, STELLAR_AZIMUTH[i]);
        float cosine = sin(slice.x);
        vec3 light = N*cosine+tangent*cos(slice.x);
        vec3 uv = surface_atmosphere_coordinates(lookup, light);
        direct += textureLod(surface_atmosphere_lut, uv, 0.0).rgb*(cosine*2.0*slice.y*slice.z);
    }
    return direct*brightness;
}

vec3 surface_small_skylight(SurfaceAtmosphereSample lookup, vec3 sun, float radius) {
    vec3 up = normalize(lookup.up+lookup.pole*dot(lookup.up, lookup.pole)*lookup.flattening);
    vec3 tangent = stellar_tangent(up, sun);
    float elevation = asin(clamp(dot(up, sun), -1.0, 1.0));
    vec2 range = stellar_elevation_range(elevation, radius, -0.5*STELLAR_PI);
    vec3 diffuse = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        vec3 slice = stellar_slice(range, elevation, radius, STELLAR_AZIMUTH[i]);
        vec3 light = up*sin(slice.x)+tangent*cos(slice.x);
        vec3 uv = surface_atmosphere_coordinates(lookup, light);
        diffuse += textureLod(surface_atmosphere_lut, uv+vec3(0,0,1), 0.0).rgb*(2.0*slice.y*slice.z);
    }
    return diffuse*lookup.height_layer_scale.z;
}

vec3 surface_stellar_light(uint body, vec3 local, vec3 base_chord, vec3 N, vec3 V, vec3 sun,
                          float sin_radius, float emission_cosine, SurfaceMaterial material, uint source, out vec3 diffuse) {
    bool atmosphere = surface_atmospheres[body].w > 0.5;
    SurfaceAtmosphereSample lookup;
    vec3 transmission = vec3(1.0), uv = vec3(0.0);
    diffuse = vec3(0.0);
    bool indirect = false;
    if (atmosphere) {
        lookup = surface_atmosphere_sample(body, local, base_chord, N);
        uv = surface_atmosphere_coordinates(lookup, sun);
        vec4 sky = textureLod(surface_atmosphere_lut, uv+vec3(0,0,1), 0.0);
        diffuse = sky.rgb*lookup.height_layer_scale.z;
        indirect = sky.a > 0.5;
    }
    float radius = asin(clamp(sin_radius, 0.0, 1.0));
    float mu = clamp(dot(N, sun), -1.0, 1.0);
    if (radius < 1e-5 || (radius < 0.01 && abs(mu) > 4.0*radius
        && abs(dot(normalize(local), sun)) > 4.0*radius))
        return (atmosphere ? textureLod(surface_atmosphere_lut, uv, 0.0).rgb : vec3(1.0))
            *surface_direct_response(N, V, sun, sin_radius, max(mu, 0.0), material, emission_cosine);
    if (!atmosphere && material.grain.x < 0.5)
        return vec3(stellar_lambert_irradiance(mu, sin_radius)*material.surface.w);
    if (source == 0u && atmosphere && !indirect && material.grain.x < 0.5 && radius >= 0.01
        && asin(mu)+radius > 0.15*radius) {
        BodyInstanceGPU b = bodies[body];
        vec3 pole = body_pole(b);
        float polar_ratio = max(1.0-b.pole_oblateness.w, 0.05);
        vec3 geometric = normalize(local+pole*dot(local, pole)*(1.0/(polar_ratio*polar_ratio)-1.0));
        if (b.pole_oblateness.w < 0.01 && dot(N, geometric) > 0.999999) {
            vec4 cached = textureLod(surface_atmosphere_lut, uv+vec3(0,0,2), 0.0);
            if (abs(sin_radius-sin(cached.a)) < 0.01*sin_radius) {
                vec3 light = cached.rgb*cached.rgb;
                return light*light*material.surface.w;
            }
        }
    }
    float elevation = asin(mu);
    float point_weight = radius < 0.01
        ? smoothstep(2.0*radius, 4.0*radius, min(abs(mu), abs(dot(normalize(local), sun)))) : 0.0;
    vec3 point_direct = vec3(0.0);
    if (point_weight > 0.0)
        point_direct = (atmosphere ? textureLod(surface_atmosphere_lut, uv, 0.0).rgb : vec3(1.0))
            *surface_direct_response(N, V, sun, sin_radius, max(mu, 0.0), material, emission_cosine);
    vec3 point_diffuse = diffuse;
    if (atmosphere && radius < 0.01 && lookup.flattening < 0.0101 && material.grain.x < 0.5
        && mu < 0.98
        && surface_stellar_horizon_clear(lookup, sun, radius)) {
        vec3 direct = surface_small_stellar_light(lookup, N, sun, radius, material.surface.w);
        if (indirect && radius > 0.005)
            diffuse = mix(surface_small_skylight(lookup, sun, radius), point_diffuse, point_weight);
        return mix(direct, point_direct, point_weight);
    }
    vec2 range = stellar_elevation_range(elevation, radius, 0.0);
    vec3 tangent = stellar_tangent(N, sun), bitangent = cross(N, tangent);
    vec3 direct = vec3(0.0);
    diffuse = vec3(0.0);
    if (range.y > range.x) for (int i = 0; i < 8; ++i) {
        vec3 slice = stellar_slice(range, elevation, radius, i);
        float s = sin(slice.x), c = cos(slice.x);
        if (!atmosphere && material.grain.x < 0.5) {
            direct += vec3(2.0*slice.y*slice.z*max(s, 0.0)*material.surface.w);
            continue;
        }
        for (int j = 0; j < 4; ++j) {
            float phi = slice.y*STELLAR_AZIMUTH[j].x;
            vec3 light = N*s+c*(tangent*cos(phi)+bitangent*sin(phi));
            transmission = atmosphere ? textureLod(surface_atmosphere_lut, surface_atmosphere_coordinates(lookup, light), 0.0).rgb : vec3(1.0);
            float weight = slice.z*slice.y*STELLAR_AZIMUTH[j].y;
            direct += weight*transmission*surface_point_response(N, V, light, material, emission_cosine);
        }
    }
    if (indirect && radius >= 0.01) {
        range = stellar_elevation_range(elevation, radius, -0.5*STELLAR_PI);
        for (int i = 0; i < 8; ++i) {
            vec3 slice = stellar_slice(range, elevation, radius, i);
            for (int j = 0; j < 4; ++j) {
                float phi = slice.y*STELLAR_AZIMUTH[j].x;
                vec3 light = N*sin(slice.x)+cos(slice.x)*(tangent*cos(phi)+bitangent*sin(phi));
                vec3 sky = textureLod(surface_atmosphere_lut, surface_atmosphere_coordinates(lookup, light)+vec3(0,0,1), 0.0).rgb;
                diffuse += sky*(slice.z*slice.y*STELLAR_AZIMUTH[j].y);
            }
        }
        diffuse *= lookup.height_layer_scale.z;
    } else if (indirect && radius > 0.005) diffuse = surface_small_skylight(lookup, sun, radius);
    else diffuse = point_diffuse;
    diffuse = mix(diffuse, point_diffuse, point_weight);
    return mix(direct, point_direct, point_weight);
}
