#version 460
#extension GL_ARB_gpu_shader_fp64 : require

#include "common/orbit_curve.glsl"
#include "common/refraction.glsl"

layout(std430, set = 0, binding = 0) readonly buffer Orbits { OrbitData orbits[]; };
layout(push_constant) uniform Push {
    mat4 view_projection;
    dvec4 eye;
    vec4 parameters;
} pc;

layout(location = 0) out vec4 color;
layout(location = 1) out float phase;
layout(location = 2) flat out float patched;

void main() {
    OrbitData orbit = orbits[gl_InstanceIndex];
    float t = float(gl_VertexIndex) / pc.parameters.x;
    dvec3 position = orbit_curve_point(orbit, t);
    gl_Position = pc.view_projection * vec4(refract_position(vec3(position - pc.eye.xyz)), 1.0);
    float alpha;
    if (orbit.d4.z > 0.5) alpha = mix(pc.parameters.z, 1.0, t);
    else if (t > float(orbit.d4.x)) alpha = pc.parameters.z;
    else alpha = mix(pc.parameters.z, 1.0, t / max(float(orbit.d4.x), 0.001));
    if (orbit.d4.y > 0.5) alpha = mix(1.0, pc.parameters.z, t);
    if (orbit.d1.w >= 1.0) {
        float distance = float(length(position - orbit.d2.xyz));
        float extent = max(300.0, float(orbit.d0.w / (1.0lf + orbit.d1.w)) * 10.0);
        alpha *= 1.0 - smoothstep(max(150.0, extent * 0.5), extent, distance);
    }
    vec3 rgb = vec3(orbit.d3.xyz) * 0.4;
    float maximum = max(rgb.r, max(rgb.g, rgb.b));
    if (maximum > 0.0001 && maximum < 0.25) rgb *= 0.25 / maximum;
    else if (maximum <= 0.0001) rgb = vec3(0.25);
    color = vec4(rgb, alpha * pc.parameters.y);
    phase = t * pc.parameters.x / 10.0;
    patched = float(orbit.d4.y);
}
