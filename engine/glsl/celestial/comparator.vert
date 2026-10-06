#version 460 core
in vec3 in_position;
in vec3 in_normal;
uniform mat4 u_projection;
uniform vec3 u_center;
uniform vec3 u_pole;
uniform float u_radius;
uniform float u_oblateness;
out vec3 v_local;
out vec3 v_normal;
void main() {
    v_local = in_position;
    vec3 position = in_position - u_pole * dot(in_position, u_pole) * u_oblateness;
    v_normal = normalize(in_normal + u_pole * dot(in_normal, u_pole) *
                         (1.0 / (1.0 - u_oblateness) - 1.0));
    gl_Position = u_projection * vec4(u_center + position * u_radius, 1.0);
}
