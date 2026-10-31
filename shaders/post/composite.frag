#version 460

#include "common/tonemap.glsl"
#ifdef BLOOM
#include "common/bloom.glsl"
#endif

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D u_hdr_scene;
#ifdef BLOOM
layout(set = 0, binding = 1) uniform sampler2D u_physical_scene;
layout(set = 0, binding = 2) uniform sampler2D u_bloom_halo;
layout(push_constant) uniform Bloom { float strength; float highlight_limit; } u_bloom;
#endif

void main() {
    vec3 final_color = texelFetch(u_hdr_scene, ivec2(gl_FragCoord.xy), 0).rgb;
#ifdef BLOOM
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    final_color = bloom_resolve(final_color, texelFetch(u_physical_scene, pixel, 0).rgb, u_bloom_halo, pixel,
                                u_bloom.strength, u_bloom.highlight_limit);
#endif
#ifdef SIZE_COMPARISON
    out_color = vec4(pow(max(final_color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
#else
    float pre_luma = dot(final_color, vec3(0.2126, 0.7152, 0.0722));
    final_color = aces_tonemap(final_color);
    float eye_hurt_blend = smoothstep(10.0, 150.0, pre_luma);
    vec3 violet_tint = vec3(0.65, 0.45, 1.0);
    final_color = mix(final_color, violet_tint, eye_hurt_blend * 0.03);
    out_color = vec4(pow(max(final_color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
#endif
}
