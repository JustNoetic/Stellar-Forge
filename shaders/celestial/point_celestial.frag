#version 460

#include "common/point_source.glsl"

layout(location = 0) in vec2 f_pixel_offset;
layout(location = 1) flat in float f_point_radius;
layout(location = 2) flat in vec3 f_flux;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(f_flux * point_profile(f_pixel_offset, f_point_radius), 0.0);
}
