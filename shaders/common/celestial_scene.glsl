#include "point_source.glsl"
#include "surface_material.glsl"

struct BodyInstanceGPU {
    vec4 pos_radius;
    vec4 color_is_star;
    vec4 pole_oblateness;
    vec4 pole_color_lum;
    vec4 spin_flags;
    uvec4 surface_flags;
    vec4 shine_direction[4];
    vec4 shine_irradiance[4];
    SurfaceMaterial material;
};

layout(std140, set = 0, binding = 0) uniform SceneCelestialUBO {
    mat4 view_proj;
    mat4 view;
    vec4 stars_pos_radius[16];
    vec4 stars_color_lum[16];
    vec4 screen_params;
    vec4 point_params;
    uint num_stars;
    uint num_bodies;
    uint hdr_enabled;
    uint num_ring_hosts;
    uvec4 ring_shadow_params;
    uvec4 eclipse_params;
    uvec4 ringshine_params;
    mat4 projection;
} u_scene;

layout(std430, set = 0, binding = 1) readonly buffer BodyInstanceBuffer {
    BodyInstanceGPU bodies[];
};

float apparent_radius_px(vec4 pos_radius) {
    return pos_radius.w / max(length(pos_radius.xyz), 1e-12)
           * u_scene.screen_params.y * u_scene.screen_params.z;
}

vec3 body_pole(BodyInstanceGPU b) {
    return length(b.pole_oblateness.xyz) > 1e-4
           ? normalize(b.pole_oblateness.xyz) : vec3(0.0, 1.0, 0.0);
}

vec3 stellar_radiance(BodyInstanceGPU b, vec3 N, vec3 V) {
    float lat2 = pow(dot(N, body_pole(b)), 2.0);
    vec3 color = mix(b.color_is_star.rgb, b.pole_color_lum.rgb, lat2);
    float lum = max(mix(b.pole_color_lum.w, b.spin_flags.y, lat2), 0.0);
    float radius = max(b.pos_radius.w, 1e-12);
    float radiance = u_scene.hdr_enabled != 0u ? lum / (radius * radius) : 1.0;
    float limb = 1.0 - clamp(dot(N, V), 0.0, 1.0);
    return color * radiance * (1.0 - 0.4 * limb - 0.2 * limb * limb);
}
