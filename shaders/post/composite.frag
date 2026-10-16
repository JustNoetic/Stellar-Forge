#version 460

#include "common/tonemap.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D u_hdr_scene;

void main() {
    vec3 final_color = texture(u_hdr_scene, v_uv).rgb;
    float pre_luma = dot(final_color, vec3(0.2126, 0.7152, 0.0722));
    final_color = aces_tonemap(final_color);
    float eye_hurt_blend = smoothstep(10.0, 150.0, pre_luma);
    vec3 violet_tint = vec3(0.65, 0.45, 1.0);
    final_color = mix(final_color, violet_tint, eye_hurt_blend * 0.03);
    out_color = vec4(pow(max(final_color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}
