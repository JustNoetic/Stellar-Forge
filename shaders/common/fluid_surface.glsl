#ifndef SF_FLUID_SURFACE
#define SF_FLUID_SURFACE
#include "procedural_noise.glsl"

const float FLUID_AU_KM = 149597870.7;
const float FLUID_PI = 3.141592653589793;
const uint OCEAN_CASCADE_COUNT = 5u;

#ifndef SURFACE_OCEAN_FFT
#define SURFACE_OCEAN_FFT 1
#endif
#ifndef SF_DONT_UNROLL
#if defined(SF_ENABLE_DONT_UNROLL) && defined(GL_EXT_control_flow_attributes)
#define SF_DONT_UNROLL [[dont_unroll]]
#else
#define SF_DONT_UNROLL
#endif
#endif

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
layout(set = 0, binding = 32) uniform sampler2DArray ocean_moments;
#ifdef SURFACE_OCEAN_WIND_LUT
layout(set = 0, binding = 50) uniform samplerCube ocean_wind_lut;
#endif
#ifdef SURFACE_OCEAN_MIP_SUMMARY
layout(std430, set = 0, binding = 49) readonly buffer OceanMipSummary {
    vec4 ocean_mip_prefix_sv[6];
    vec4 ocean_mip_prefix_j[6];
};
#endif

layout(std430, set = 0, binding = 28) readonly buffer SurfaceFluidBuffer {
    uvec4 fluid_flags;
    vec4 fluid_header;
    vec4 ocean_tangent;
    vec4 ocean_bitangent;
    vec4 ocean_cascades[OCEAN_CASCADE_COUNT];
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

// max_lod caps the map mip; the volumetric bound pyramid is dilated for CLOUD_VOLUME_MAX_COVERAGE_LOD.
CloudWeather cloud_weather_lod(uint body, vec3 fixed_direction, float footprint_km, float max_lod) {
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
        float lod = clamp(log2(max(footprint_km*pixels_per_km, 1.0)), 0.0, min(float(textureQueryLevels(cloud_coverage_maps)-1), max_lod));
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

CloudWeather cloud_weather(uint body, vec3 fixed_direction, float footprint_km) {
    return cloud_weather_lod(body, fixed_direction, footprint_km, 1e6);
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

float ocean_below(float x) {
    return 1.0/(1.0+exp(-1.702*x));
}

OceanSurface ocean_surface(uint body, vec3 fixed_normal, vec3 fixed_from_camera_m, vec3 fixed_dx_m, vec3 fixed_dy_m, mat3 rotation) {
    SurfaceFluidGPU f = fluids[body];
#ifdef SURFACE_OCEAN_WIND_LUT
    float wind = 0.55 + 0.9 * textureLod(ocean_wind_lut, fixed_normal, 0.0).r;
#else
    float wind = 0.55+0.9*fbm_noise(fixed_normal*7.0+vec3(3.1, 1.7, 5.3), 3, 2.1, 0.5);
#endif
    float u = f.ocean_params.x*sqrt(wind);
    float whitecaps = clamp(3.84e-6*pow(max(u, 0.0), 3.41)*f.ocean_shallow.w*2.0, 1e-5, 0.5);
    OceanSurface o;
#if SURFACE_OCEAN_FFT
    if (fluid_flags.w != body) {
#endif
        o.normal = normalize(rotation*fixed_normal);
        o.roughness_squared = f.ocean_params.y*wind+0.001;
        o.foam = min(whitecaps, 0.8);
        return o;
#if SURFACE_OCEAN_FFT
    }
    vec3 tangent = ocean_tangent.xyz, bitangent = ocean_bitangent.xyz;
    vec2 p = vec2(dot(fixed_from_camera_m, tangent), dot(fixed_from_camera_m, bitangent));
    vec2 px = vec2(dot(fixed_dx_m, tangent), dot(fixed_dx_m, bitangent));
    vec2 py = vec2(dot(fixed_dy_m, tangent), dot(fixed_dy_m, bitangent));
#ifdef SURFACE_OCEAN_MIP_SUMMARY
    uint first_c = fluid_flags.z & 7u;
    uint last_c = (fluid_flags.z >> 4u) & 7u;
    float footprint2 = max(dot(px, px), dot(py, py));
    SF_DONT_UNROLL
    while (first_c <= last_c && (ocean_cascades[first_c].w <= 0.0 || footprint2 * (ocean_cascades[first_c].z * ocean_cascades[first_c].z) >= 1.0)) {
        ++first_c;
    }
    vec4 base_sv = ocean_mip_prefix_sv[first_c];
    vec4 base_j = ocean_mip_prefix_j[first_c];
    vec2 slope = base_sv.xy;
    float variance = base_sv.z;
    float jacobian = base_j.x;
    float jacobian_variance = base_j.y;
    SF_DONT_UNROLL
    for (uint i = first_c; i <= last_c; ++i) {
        vec4 c = ocean_cascades[i];
        if (c.w <= 0.0) continue;
        vec2 uv = p*c.z+c.xy;
        vec4 s = textureGrad(ocean_moments, vec3(uv, float(2u*i)), px*c.z, py*c.z);
        vec4 j = textureGrad(ocean_moments, vec3(uv, float(2u*i+1u)), px*c.z, py*c.z);
        slope += s.xy;
        variance += max(s.z-s.x*s.x, 0.0)+max(s.w-s.y*s.y, 0.0);
        jacobian += j.x;
        jacobian_variance += max(j.y-j.x*j.x, 0.0);
    }
#else
    vec2 slope = vec2(0.0);
    float variance = 0.0, jacobian = 0.0, jacobian_variance = 0.0;
    for (uint i = 0u; i < OCEAN_CASCADE_COUNT; ++i) {
        vec4 c = ocean_cascades[i];
        vec2 uv = p*c.z+c.xy;
        vec4 s = textureGrad(ocean_moments, vec3(uv, float(2u*i)), px*c.z, py*c.z);
        vec4 j = textureGrad(ocean_moments, vec3(uv, float(2u*i+1u)), px*c.z, py*c.z);
        slope += s.xy;
        variance += max(s.z-s.x*s.x, 0.0)+max(s.w-s.y*s.y, 0.0);
        jacobian += j.x;
        jacobian_variance += max(j.y-j.x*j.x, 0.0);
    }
#endif
    float scale = sqrt(wind);
    slope *= scale;
    vec3 gradient = slope.x*tangent+slope.y*bitangent;
    gradient -= fixed_normal*dot(gradient, fixed_normal);
    o.normal = normalize(rotation*normalize(fixed_normal-gradient));
    o.roughness_squared = (variance+fluid_header.z)*wind+0.001;
    float spread = sqrt(fluid_header.w*wind);
    float threshold = 1.0-log((1.0-whitecaps)/whitecaps)/1.702*spread;
    float local = max(sqrt(jacobian_variance)*scale, 0.02*spread+1e-4);
    o.foam = min(ocean_below((threshold-1.0-jacobian*scale)/local), 0.8);
    return o;
#endif
}

#endif
