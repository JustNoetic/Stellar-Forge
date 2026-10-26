#version 460
#include "common/size_comparison.glsl"
#include "common/surface_textures.glsl"
#include "common/body_surface.glsl"
layout(location = 0) in vec2 local_xy;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 origin = vec3(local_xy, cmp.shape.w);
    vec2 ground = comparison_roots(origin, vec3(0,0,-1), cmp.display.x);
    if (ground.y < ground.x) discard;
    vec3 p = origin+vec3(0,0,-ground.x);
    vec3 N = normalize(comparison_stretch(comparison_stretch(p)));
    vec3 albedo = pow(cmp.color.rgb, vec3(2.2));
    if (cmp.color.w > 1.5) albedo = vec3(0);
    else if (cmp.color.w > 0.5) {
        float mu = max(N.z, 0.0);
        albedo *= 1.0-0.45*(1.0-mu)-0.15*(1.0-mu)*(1.0-mu);
    } else {
        mat3 rotation = body_surface_rotation(COMPARISON_POLE,cmp.shape.z);
        vec3 local = transpose(rotation)*normalize(comparison_stretch(p));
        if (cmp.resources.x >= 0.0) albedo = surface_albedo(uint(cmp.resources.x), local, albedo);
        if (cmp.resources.y >= 0.0) N = surface_shading_normal(uint(cmp.resources.y), local, N, rotation);
        albedo *= max(dot(N, COMPARISON_LIGHT), 0.0);
    }
    gl_FragDepth = clamp(0.5+0.5*p.z/cmp.display.y, 0.0, 1.0);
    out_color = vec4(albedo, 1.0);
}
