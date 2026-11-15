#version 460
#ifdef TRANSPARENCY_CAPTURE
#extension GL_KHR_shader_subgroup_ballot : require
#endif
layout(early_fragment_tests) in;
layout(push_constant) uniform Push { vec4 color; vec4 parameters; } pc;
layout(location = 0) out vec4 out_color;
#ifdef TRANSPARENCY_CAPTURE
#include "common/transparency.glsl"
#endif
void main() {
    out_color = vec4(pc.color.rgb*pc.color.a, pc.color.a);
#ifdef TRANSPARENCY_CAPTURE
    if (capture_transparency(out_color)) out_color = vec4(0.0);
#endif
}
