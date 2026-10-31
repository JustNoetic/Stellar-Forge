#version 460
layout(early_fragment_tests) in;
#ifdef TRANSPARENCY_CAPTURE
#include "common/transparency.glsl"
#endif

layout(location = 0) in vec4 color;
layout(location = 1) in float phase;
layout(location = 2) flat in float patched;
layout(location = 0) out vec4 out_color;

void main() {
    if (patched > 0.5 && fract(phase) > 0.5) discard;
    if (color.a < 0.01) discard;
    out_color = vec4(pow(color.rgb, vec3(2.2)) * 4.0, color.a);
#ifdef TRANSPARENCY_CAPTURE
    if (capture_transparency(vec4(out_color.rgb*out_color.a, out_color.a), gl_FragCoord.z, true)) out_color = vec4(0.0);
#endif
}
