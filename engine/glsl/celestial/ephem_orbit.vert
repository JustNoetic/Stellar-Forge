#version 460 core
in vec4 in_pos;
uniform mat4 projection;
uniform mat4 view_rot;
uniform dvec4 u_cam_pos_double;
uniform dvec3 u_bary_pos_double;
uniform vec3 u_color;
uniform float u_alpha;
uniform float u_depth_C;
uniform float u_far;

#include "common/refraction.glsl"

out vec4 f_color;
out float f_clip_z;
void main() {
    dvec3 eye_pos_d = (dvec3(in_pos.xyz) + u_bary_pos_double) - u_cam_pos_double.xyz;
    vec3 eye_pos = vec3(eye_pos_d);
    vec3 cam_pos = vec3(u_cam_pos_double.xyz);
    vec3 app_eye_pos = apply_refraction_eye(eye_pos, cam_pos);

    vec3 final_rgb = u_color * 0.4;
    float max_c = max(final_rgb.r, max(final_rgb.g, final_rgb.b));
    if (max_c < 0.25 && max_c > 0.0001) {
        final_rgb = final_rgb * (0.25 / max_c);
    } else if (max_c <= 0.0001) {
        final_rgb = vec3(0.25);
    }

    float alpha = in_pos.w;
    if (u_alpha > 0.0) {
        alpha *= u_alpha;
    }

    f_color = vec4(final_rgb, alpha);
    gl_Position = projection * view_rot * vec4(app_eye_pos, 1.0);
    f_clip_z = gl_Position.w;
}
