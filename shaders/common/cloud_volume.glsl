#ifndef SF_CLOUD_VOLUME
#define SF_CLOUD_VOLUME
// Volumetric cloud density and conservative empty-space bounds (docs/volumetric_clouds_plan.md §5).
// Needs celestial_scene.glsl and body_surface.glsl before it. Positions passed in are camera-relative km in world
// axes ("view"); the parameter block carries the camera's offset from the focus body and the fract() noise anchors,
// so no large coordinate is formed in FP32.
#include "fluid_surface.glsl"

const uint CLOUD_VOLUME_NONE = 0xffffffffu;
const int CLOUD_PROFILE_TYPES = 4;
const int CLOUD_PROFILE_HEIGHTS = 128;
const int CLOUD_PROFILE_THRESHOLDS = 64;

layout(std140, set = 0, binding = 40) uniform CloudVolumeParameters {
    uvec4 flags;          // x focus body, y bound levels, zw bound level-0 size
    vec4 camera_local;    // xyz camera minus body centre, km
    vec4 shape_anchor;    // xyz noise anchor (tiles), w 1/tile km
    vec4 detail_anchor;
    vec4 erosion;         // x base erosion cap, y detail erosion depth, z max coverage lod
    vec4 bounds[CLOUD_PROFILE_TYPES*CLOUD_PROFILE_THRESHOLDS];
} u_cloud_volume;
layout(set = 0, binding = 41) uniform sampler3D cloud_shape_noise;
layout(set = 0, binding = 42) uniform sampler3D cloud_detail_noise;
layout(set = 0, binding = 43) uniform sampler2D cloud_profile_lut;
layout(set = 0, binding = 44) uniform sampler2DArray cloud_coverage_bounds;

struct CloudShell {
    uint body;
    vec3 pole;
    float flattening;
    mat3 rotation;       // body-fixed to world
    float base_km;       // layer base radius in the pole-stretched (spherical) space
    float thickness_km;
    float type;
};

CloudShell cloud_shell(uint body) {
    BodyInstanceGPU b = bodies[body];
    SurfaceFluidGPU f = fluids[body];
    CloudShell s;
    s.body = body;
    s.pole = body_pole(b);
    s.flattening = fluid_flattening(b);
    s.rotation = body_surface_rotation(s.pole, b.spin_flags.x);
    s.base_km = f.cloud_motion.w+f.cloud_shape.y;
    s.thickness_km = max(f.cloud_shape.z, 1e-3);
    s.type = clamp(floor(f.cloud_style.x+0.5), 0.0, float(CLOUD_PROFILE_TYPES-1));
    return s;
}

vec3 cloud_local(vec3 view_km) {
    return u_cloud_volume.camera_local.xyz+view_km;
}

float cloud_height_fraction(CloudShell s, vec3 local_km) {
    return (length(fluid_spherical(local_km, s.pole, s.flattening))-s.base_km)/s.thickness_km;
}

// x shape (where the type can exist), y density gradient.
vec2 cloud_profile(CloudShell s, float h) {
    float u = (clamp(h, 0.0, 1.0)*float(CLOUD_PROFILE_HEIGHTS-1)+0.5)/float(CLOUD_PROFILE_HEIGHTS);
    return textureLod(cloud_profile_lut, vec2(u, (s.type+0.5)/float(CLOUD_PROFILE_TYPES)), 0.0).rg;
}

float cloud_remap01(float value, float low, float high) {
    return clamp((value-low)/max(high-low, 1e-6), 0.0, 1.0);
}

// Map coverage with the cell field multiplied in: separate cumulus cells at shape_size_km, merging as coverage
// rises. Never exceeds the map coverage, so the map bound still holds.
float cloud_volume_coverage(CloudShell s, vec3 fixed_view, vec3 fixed_dir, float height_km, float footprint_km) {
    float coverage = cloud_weather_lod(s.body, fixed_dir, footprint_km, u_cloud_volume.erosion.z).coverage;
    if (coverage <= 0.0) return 0.0;
    // Columns: project to the base along the radial so a cell keeps its footprint through the layer.
    vec3 column = (fixed_view-fixed_dir*height_km)*u_cloud_volume.shape_anchor.w+u_cloud_volume.shape_anchor.xyz;
    float cell = smoothstep(0.25, 0.75, textureLod(cloud_shape_noise, column, 0.0).g);
    float merge = coverage*coverage;
    return coverage*mix(cell, 1.0, merge*merge);
}

struct CloudSample {
    float sigma;         // extinction, 1/km
    float height;        // height fraction in the shell
    float profile;       // dimensional profile c + shape - 1 (before erosion), for lighting terms
};

// detail_weight in [0, 1] fades the detail erosion with distance; 0 skips its fetch.
CloudSample cloud_volume_sample(CloudShell s, vec3 view_km, float footprint_km, float detail_weight) {
    CloudSample result = CloudSample(0.0, 0.0, 0.0);
    vec3 local = cloud_local(view_km);
    float h = cloud_height_fraction(s, local);
    result.height = h;
    if (!(h > 0.0 && h < 1.0)) return result;
    mat3 to_fixed = transpose(s.rotation);
    vec3 fixed_view = to_fixed*view_km;
    vec3 fixed_dir = normalize(to_fixed*local);
    float c = cloud_volume_coverage(s, fixed_view, fixed_dir, h*s.thickness_km, footprint_km);
    vec2 profile = cloud_profile(s, h);
    float d = c+profile.x-1.0;
    result.profile = d;
    if (d <= 0.0) return result;
    vec4 n = textureLod(cloud_shape_noise, fixed_view*u_cloud_volume.shape_anchor.w+u_cloud_volume.shape_anchor.xyz, 0.0);
    float fbm = dot(n.gba, vec3(0.625, 0.25, 0.125));
    float base = cloud_remap01(n.r, fbm-1.0, 1.0);
    // Capped erosion: noise removes at most erosion.x of the profile, so cores (d > cap) always survive.
    float e = u_cloud_volume.erosion.x*(1.0-base);
    float x = clamp((d-e)/(1.0-e), 0.0, 1.0);
    if (detail_weight > 0.0 && x > 0.0) {
        vec3 detail = textureLod(cloud_detail_noise, fixed_view*u_cloud_volume.detail_anchor.w+u_cloud_volume.detail_anchor.xyz, 0.0).rgb;
        float detail_fbm = dot(detail, vec3(0.625, 0.25, 0.125));
        // Wispy erosion low in the layer, billowy higher up.
        float shaped = mix(detail_fbm, 1.0-detail_fbm, clamp(h*4.0, 0.0, 1.0));
        float e2 = u_cloud_volume.erosion.y*detail_weight*(1.0-shaped);
        x = clamp((x-e2)/(1.0-e2), 0.0, 1.0);
    }
    result.sigma = x*profile.y*fluids[s.body].cloud_shape.w;
    return result;
}

float cloud_volume_density(CloudShell s, vec3 view_km, float footprint_km, float detail_weight) {
    return cloud_volume_sample(s, view_km, footprint_km, detail_weight).sigma;
}

// ---- Conservative bounds -------------------------------------------------------------------------------------

float cloud_geodetic_latitude(float geocentric, float polar) {
    return atan(tan(clamp(geocentric, -1.5707963, 1.5707963))/(polar*polar));
}

// Upper bound of cloud_weather_lod coverage over a cap of body-fixed directions.
float cloud_coverage_bound(CloudShell s, vec3 fixed_center, float angular_radius) {
    SurfaceFluidGPU f = fluids[s.body];
    if (f.cloud_shape.x <= 0.0) return 0.0;
    float bias = 2.0*(f.cloud_shape.x-0.5);
    if (f.cloud_style.w <= 0.5) return clamp(1.0+bias, 0.0, 1.0);
    vec3 dir = fluid_rotate_pole(normalize(fixed_center), f.cloud_motion.y);
    float polar = max(1.0-bodies[s.body].pole_oblateness.w, 0.05);
    const float PI = 3.141592653589793;
    float latitude = asin(clamp(dir.y, -1.0, 1.0));
    float radius = angular_radius+1e-5;
    float v_high = 0.5-cloud_geodetic_latitude(latitude+radius, polar)/PI;
    float v_low = 0.5-cloud_geodetic_latitude(latitude-radius, polar)/PI;
    float u = atan(dir.x, dir.z)/(2.0*PI)+0.5;
    float du = abs(latitude)+radius >= 0.5*PI-1e-4 ? 0.5
        : asin(clamp(sin(radius)/max(cos(latitude), 1e-6), 0.0, 1.0))/(2.0*PI)+1e-6;
    uvec2 size0 = u_cloud_volume.flags.zw;
    int levels = int(u_cloud_volume.flags.y);
    float span = max(2.0*du*float(size0.x), (v_low-v_high)*float(size0.y));
    int level = clamp(int(ceil(log2(max(span, 1.0)))), 0, levels-1);
    ivec2 size = max(ivec2(size0) >> level, ivec2(1));
    int i0 = int(floor((u-du)*float(size.x)));
    int j0 = clamp(int(floor(clamp(v_high, 0.0, 1.0)*float(size.y))), 0, size.y-1);
    int j1 = clamp(int(floor(clamp(v_low, 0.0, 1.0)*float(size.y))), 0, size.y-1);
    int i_count = du >= 0.5 ? size.x : 2;
    float layer = f.cloud_style.w-1.0;
    float best = 0.0;
    for (int j = j0; j <= j1; ++j) for (int k = 0; k < min(i_count, size.x); ++k) {
        int i = ((i0+k) % size.x+size.x) % size.x;
        best = max(best, texelFetch(cloud_coverage_bounds, ivec3(i, j, int(layer)), level).r);
    }
    // One unorm step absorbs filtering round-off in the coverage atlas. The bias applies even to empty texels.
    return clamp(best+1.0/255.0+bias, 0.0, 1.0);
}

// Height-fraction interval that can hold density when coverage is at most coverage_max. Empty when x > y.
vec2 cloud_height_interval(CloudShell s, float coverage_max) {
    if (coverage_max <= 0.0) return vec2(1.0, 0.0);
    float t = 1.0-coverage_max;
    int k = clamp(int(floor(t*float(CLOUD_PROFILE_THRESHOLDS-1)-1e-3)), 0, CLOUD_PROFILE_THRESHOLDS-1);
    return u_cloud_volume.bounds[int(s.type)*CLOUD_PROFILE_THRESHOLDS+k].xy;
}

// False only when density is provably zero everywhere on the segment view0..view1.
bool cloud_span_may_have_density(CloudShell s, vec3 view0, vec3 view1) {
    vec3 local0 = cloud_local(view0), local1 = cloud_local(view1);
    vec3 q0 = fluid_spherical(local0, s.pole, s.flattening), q1 = fluid_spherical(local1, s.pole, s.flattening);
    vec3 dq = q1-q0;
    float t = clamp(-dot(q0, dq)/max(dot(dq, dq), 1e-20), 0.0, 1.0);
    // Radius margin covers FP32 rounding of ~1e4 km lengths.
    float margin = 1e-3+2e-7*s.base_km;
    float r_min = length(q0+dq*t)-margin, r_max = max(length(q0), length(q1))+margin;
    float h_min = (r_min-s.base_km)/s.thickness_km, h_max = (r_max-s.base_km)/s.thickness_km;
    if (h_max <= 0.0 || h_min >= 1.0) return false;
    mat3 to_fixed = transpose(s.rotation);
    vec3 d0 = normalize(to_fixed*local0), d1 = normalize(to_fixed*local1);
    vec3 middle = d0+d1;
    float radius = 0.5*atan(length(cross(d0, d1)), dot(d0, d1));
    float coverage = length(middle) > 1e-6 ? cloud_coverage_bound(s, middle, radius)
        : cloud_coverage_bound(s, d0, 3.2);
    vec2 interval = cloud_height_interval(s, coverage);
    return interval.x <= interval.y && h_max >= interval.x && h_min <= interval.y;
}

#endif
