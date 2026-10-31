#ifndef SF_FLUID_SURFACE
#define SF_FLUID_SURFACE
#include "procedural_noise.glsl"

const float FLUID_AU_KM = 149597870.7;
const float FLUID_PI = 3.141592653589793;
const uint OCEAN_WAVE_COUNT = 16u;

struct OceanWaveGPU {
    vec4 direction_k;
    vec4 phases;
};

struct SurfaceFluidGPU {
    vec4 ocean_deep;
    vec4 ocean_shallow;
    vec4 ocean_params;
    vec4 cloud_color;
    vec4 cloud_shape;
    vec4 cloud_motion;
    vec4 cloud_style;
    vec4 cloud_noise;
};

layout(set = 0, binding = 29) uniform sampler2DArray cloud_coverage_maps;

layout(std430, set = 0, binding = 28) readonly buffer SurfaceFluidBuffer {
    uvec4 fluid_flags;
    vec4 fluid_header;
    OceanWaveGPU ocean_waves[OCEAN_WAVE_COUNT];
    SurfaceFluidGPU fluids[];
};

bool fluid_ocean_active(uint body) {
    return fluid_flags.x != 0u && fluids[body].ocean_deep.w > 0.5;
}

bool fluid_clouds_active(uint body) {
    return fluid_flags.y != 0u && fluids[body].cloud_color.w > 0.5;
}

float fluid_flattening(BodyInstanceGPU b) {
    return 1.0/max(1.0-b.pole_oblateness.w, 0.05)-1.0;
}

vec3 fluid_spherical(vec3 v, vec3 pole, float flattening) {
    return v+pole*dot(v, pole)*flattening;
}

bool fluid_sphere_roots(vec3 origin, vec3 ray, float radius, out vec2 roots) {
    float a = dot(ray, ray);
    float b = dot(origin, ray);
    vec3 perpendicular = cross(origin, ray);
    float discriminant = a*radius*radius-dot(perpendicular, perpendicular);
    if (discriminant < 0.0) { roots = vec2(-1.0); return false; }
    float s = sqrt(discriminant);
    roots = vec2(-b-s, -b+s)/a;
    return true;
}

vec3 fluid_rotate_pole(vec3 p, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec3(p.x*c-p.z*s, p.y, p.x*s+p.z*c);
}

struct CloudWeather {
    float coverage;
};

CloudWeather cloud_weather(uint body, vec3 fixed_direction, float footprint_km) {
    SurfaceFluidGPU f = fluids[body];
    CloudWeather weather;
    weather.coverage = 0.0;
    if (f.cloud_shape.x <= 0.0) return weather;
    vec3 dir = fluid_rotate_pole(fixed_direction, f.cloud_motion.y);
    float scale = f.cloud_motion.w/max(f.cloud_motion.x, 1e-3);
    vec3 evolution = f.cloud_motion.z*vec3(0.31, 0.17, 0.23);
    float coverage;
    if (f.cloud_style.w > 0.5) {
        float polar = max(1.0-bodies[body].pole_oblateness.w, 0.05);
        vec2 uv = vec2(atan(dir.x, dir.z)/(2.0*FLUID_PI)+0.5,
            0.5-atan(dir.y/(polar*polar), length(dir.xz))/FLUID_PI);
        vec2 dimensions = vec2(textureSize(cloud_coverage_maps, 0).xy);
        float longitude_scale = 1.0/max(length(dir.xz), 0.02);
        float pixels_per_km = max(dimensions.x*longitude_scale/(2.0*FLUID_PI), dimensions.y/FLUID_PI)/f.cloud_motion.w;
        float lod = clamp(log2(max(footprint_km*pixels_per_km, 1.0)), 0.0, float(textureQueryLevels(cloud_coverage_maps)-1));
        float half_texel = 0.5/float(textureSize(cloud_coverage_maps, int(ceil(lod))).y);
        uv.y = clamp(uv.y, half_texel, 1.0-half_texel);
        coverage = textureLod(cloud_coverage_maps, vec3(uv, f.cloud_style.w-1.0), lod).r;
    } else {
        float w = fbm_noise(dir*scale+evolution, 4, 2.07, 0.5);
        w += 0.04*cos(asin(clamp(dir.y, -1.0, 1.0))*6.0);
        coverage = smoothstep(0.35, 0.65, w);
    }
    weather.coverage = clamp(coverage+2.0*(f.cloud_shape.x-0.5), 0.0, 1.0);
    return weather;
}

float cloud_top_softness(float top) {
    return min(top*0.5, clamp(top*0.18, 0.02, 0.12));
}

float cloud_profile(float height, float top) {
    if (height <= 0.0 || height >= top || top <= 0.0) return 0.0;
    float base = smoothstep(0.0, min(0.05, top*0.25), height);
    return base*(1.0-smoothstep(top-cloud_top_softness(top), top, height));
}

float cloud_bulk_density(float height, CloudWeather weather) {
    return cloud_profile(height, weather.coverage)*mix(0.4, 1.0, weather.coverage);
}

float cloud_column_above(float height, float top) {
    if (height >= top || top <= 0.0) return 0.0;
    float base = min(0.05, top*0.25);
    float cap = cloud_top_softness(top), start = top-cap;
    if (height < base) {
        float x = clamp(height/base, 0.0, 1.0);
        return start+0.5*cap-base*(0.5+x*x*x-0.5*x*x*x*x);
    }
    if (height <= start) return start-max(height, 0.0)+0.5*cap;
    float x = clamp((height-start)/cap, 0.0, 1.0);
    return cap*(0.5-x+x*x*x-0.5*x*x*x*x);
}

float cloud_profile_average(float height, float span, float top) {
    span = abs(span);
    if (span < 1e-3) return cloud_profile(height, top);
    return max(cloud_column_above(height-0.5*span, top)-cloud_column_above(height+0.5*span, top), 0.0)/span;
}

float cloud_ramp_integral(float x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return x-0.5;
    return x*x*x*(1.0-0.5*x);
}

float cloud_ramp_average(float value, float span, float width) {
    span = abs(span);
    if (span < 1e-3) return smoothstep(0.0, width, value);
    return width*(cloud_ramp_integral((value+0.5*span)/width)-cloud_ramp_integral((value-0.5*span)/width))/span;
}

float cloud_profile_average(float height, float span, float top, float top_change) {
    if (top <= 0.0) return 0.0;
    if (abs(top_change) < 1e-4) return cloud_profile_average(height, span, top);
    float base = cloud_ramp_average(height, span, min(0.05, top*0.25));
    float cap = cloud_ramp_average(top-height, top_change-span, cloud_top_softness(top));
    return clamp(base+cap-1.0, 0.0, 1.0);
}

float cloud_local_top(uint body, vec3 fixed_position_km, float height_fraction, float height_span, CloudWeather weather, float footprint_km) {
    SurfaceFluidGPU f = fluids[body];
    float detail_scale = max(f.cloud_style.y, 0.05);
    float detail_weight = 1.0-clamp(footprint_km/detail_scale-0.5, 0.0, 1.0);
    float erosion = 0.1*f.cloud_style.z*detail_weight;
    if (erosion <= 0.0 || height_fraction-0.5*height_span >= weather.coverage ||
        weather.coverage-height_fraction-0.5*height_span > cloud_top_softness(weather.coverage)+erosion) return weather.coverage;
    vec3 direction = fluid_rotate_pole(normalize(fixed_position_km), f.cloud_motion.y);
    vec3 p = direction*(f.cloud_motion.w/detail_scale)+f.cloud_motion.z*vec3(0.7, 0.4, 0.5);
    return max(weather.coverage-erosion*(1.0-value_noise(p)), 0.0);
}

float cloud_density(uint body, vec3 fixed_position_km, float height_fraction, CloudWeather weather, float footprint_km) {
    if (cloud_bulk_density(height_fraction, weather) <= 0.0) return 0.0;
    float top = cloud_local_top(body, fixed_position_km, height_fraction, 0.0, weather, footprint_km);
    return cloud_profile(height_fraction, top)*mix(0.4, 1.0, weather.coverage);
}

float cloud_density_average(uint body, vec3 fixed_position_km, float height, float span, CloudWeather weather, float top_change, float footprint_km) {
    if (cloud_profile_average(height, span, weather.coverage, top_change) <= 0.0) return 0.0;
    float top = cloud_local_top(body, fixed_position_km, height, abs(span)+abs(top_change), weather, footprint_km);
    return cloud_profile_average(height, span, top, clamp(top_change, -2.0*top, 2.0*top))*mix(0.4, 1.0, weather.coverage);
}

float ocean_beckmann_lambda(float cosine, float roughness) {
    cosine = clamp(cosine, 1e-3, 1.0);
    float tangent = sqrt(max(1.0-cosine*cosine, 0.0))/cosine;
    float a = 1.0/max(roughness*tangent, 1e-5);
    return a < 1.6 ? (1.0-1.259*a+0.396*a*a)/(3.535*a+2.181*a*a) : 0.0;
}

float ocean_fresnel(float cos_view, float slope_deviation) {
    const float F0 = 0.02;
    float c = clamp(cos_view, 0.0, 1.0);
    return clamp(F0+(1.0-F0)*pow(1.0-c, 5.0*exp(-2.69*slope_deviation))/(1.0+22.7*pow(slope_deviation, 1.5)), 0.0, 1.0);
}

float ocean_specular(vec3 N, vec3 V, vec3 L, float roughness_squared) {
    float cos_view = dot(N, V), cos_light = dot(N, L);
    if (cos_view <= 0.0 || cos_light <= 0.0) return 0.0;
    vec3 H = normalize(V+L);
    float c = dot(N, H);
    if (c <= 0.0) return 0.0;
    float c2 = c*c;
    float tan2 = (1.0-c2)/c2;
    float D = exp(-tan2/roughness_squared)/(FLUID_PI*roughness_squared*c2*c2);
    float roughness = sqrt(roughness_squared);
    float G = 1.0/(1.0+ocean_beckmann_lambda(cos_view, roughness)+ocean_beckmann_lambda(cos_light, roughness));
    float F = ocean_fresnel(dot(V, H), roughness*0.7071);
    return F*D*G/(4.0*max(cos_view, 1e-3));
}

struct OceanSurface {
    vec3 normal;
    float roughness_squared;
    float foam;
};

OceanSurface ocean_surface(uint body, vec3 fixed_normal, vec3 fixed_from_camera_m, float footprint_m, mat3 rotation) {
    SurfaceFluidGPU f = fluids[body];
    bool focus = fluid_flags.w == body;
    vec3 fixed_position_m = fixed_normal*(f.ocean_params.w*1000.0);
    float wind = 0.55+0.9*fbm_noise(fixed_normal*7.0+vec3(3.1, 1.7, 5.3), 3, 2.1, 0.5);
    float steepness_scale = sqrt(max(f.ocean_params.y, 1e-6)/max(fluid_header.x, 1e-6)*wind);
    vec3 gradient = vec3(0.0);
    float unresolved = 0.0, crest = 0.0, total = 0.0;
    for (uint i = 0u; i < OCEAN_WAVE_COUNT; ++i) {
        OceanWaveGPU w = ocean_waves[i];
        float k = w.direction_k.w;
        if (k <= 0.0) continue;
        float s = w.phases.z*steepness_scale;
        float fade = 1.0-clamp((footprint_m*4.0/w.phases.w-1.0)*0.5, 0.0, 1.0);
        float variance = 0.5*s*s;
        total += variance;
        unresolved += (1.0-fade*fade)*variance;
        if (fade <= 0.0) continue;
        float phase = focus ? k*dot(w.direction_k.xyz, fixed_from_camera_m)+w.phases.x
                            : k*dot(w.direction_k.xyz, fixed_position_m)+w.phases.y;
        gradient += w.direction_k.xyz*(s*fade*cos(phase));
        crest += fade*s*sin(phase);
    }
    gradient -= fixed_normal*dot(gradient, fixed_normal);
    OceanSurface o;
    o.normal = normalize(rotation*normalize(fixed_normal-gradient));
    o.roughness_squared = max(unresolved, 0.0)+0.0015;
    float u = f.ocean_params.x*sqrt(wind);
    float whitecaps = 3.84e-6*pow(max(u, 0.0), 3.41)*f.ocean_shallow.w;
    float crest_bias = total > 0.0 ? crest/sqrt(2.0*total) : 0.0;
    float breakup = mix(value_noise(fixed_normal*(f.ocean_params.w*1000.0/30.0)+vec3(fluid_header.y*0.05)), 0.5,
        clamp(footprint_m/30.0-0.5, 0.0, 1.0));
    o.foam = clamp(whitecaps*(1.0+3.0*max(crest_bias, 0.0))*(0.5+breakup)*4.0, 0.0, 0.8);
    return o;
}

#endif
