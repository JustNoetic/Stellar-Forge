#version 460 core
in vec3 in_position;
uniform mat4 projection;
uniform mat4 view;
uniform vec3 u_body_offset;
uniform float u_inner_r;
uniform float u_outer_r;
uniform float u_depth_C;
uniform float u_far;

out float f_clip_z;
out float f_radius_pct;

void main() {
    float r = mix(u_inner_r, u_outer_r, in_position.y);
    vec3 world_pos = vec3(in_position.x * r, 0.0, in_position.z * r) + u_body_offset;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    f_clip_z = gl_Position.w;
    f_radius_pct = in_position.y;
}
