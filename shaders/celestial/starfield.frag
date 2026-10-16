#version 460

#include "common/point_source.glsl"

layout(location = 0) flat in vec3 f_color;
layout(location = 1) flat in float f_intensity;
layout(location = 2) flat in float f_point_radius;
layout(location = 3) in vec2 f_pixel_offset;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(f_color * f_intensity * point_profile(f_pixel_offset, f_point_radius), 0.0);
}
