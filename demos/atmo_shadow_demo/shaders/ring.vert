#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_uv_r;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;

out vec3 v_world_pos;
out float v_uv_r;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_uv_r = in_uv_r;
    gl_Position = u_proj * u_view * world_pos;
}
