#version 460
#include "common/size_comparison.glsl"
layout(location = 0) out vec2 local_xy;
void main() {
    const vec2 corners[6] = vec2[6](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,-1), vec2(1,1), vec2(-1,1));
    vec2 p = corners[gl_VertexIndex];
    local_xy = p*cmp.shape.w;
    gl_Position = vec4(cmp.rect.xy+p*cmp.rect.zw, 0.5, 1.0);
}
