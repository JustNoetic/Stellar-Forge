#version 460 core
in vec3 in_pos;
uniform mat4 projection;
uniform mat4 view_rot;
uniform vec4 u_cam_pos_double;
uniform vec3 u_bary_pos;
uniform vec3 u_color;
uniform float u_depth_C;
uniform float u_far;

#include "common/refraction.glsl"

out vec4 f_color;
out float f_clip_z;
void main() {
    vec3 world_pos = in_pos + u_bary_pos;
    vec3 cam_pos = u_cam_pos_double.xyz;
    vec3 app_world_pos = apply_refraction(world_pos, cam_pos);
    vec3 eye_pos = app_world_pos - cam_pos;

    vec3 final_rgb = u_color * 0.4;
    float max_c = max(final_rgb.r, max(final_rgb.g, final_rgb.b));
    if (max_c < 0.25 && max_c > 0.0001) {
        final_rgb = final_rgb * (0.25 / max_c);
    } else if (max_c <= 0.0001) {
        final_rgb = vec3(0.25);
    }

    f_color = vec4(final_rgb, 1.0);
    gl_Position = projection * view_rot * vec4(eye_pos, 1.0);
    f_clip_z = gl_Position.w;
}
