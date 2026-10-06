#version 460 core
in vec2 in_position;
uniform mat4 u_projection;
uniform vec3 u_center;
uniform float u_atmo_radius_km;
uniform float u_pixel_km;
out vec2 v_xy_km;
void main() {
    v_xy_km = in_position * (u_atmo_radius_km + u_pixel_km);
    gl_Position = u_projection * vec4(u_center + vec3(v_xy_km*1e-5,0.0),1.0);
}
