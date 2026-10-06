#version 460 core
in vec3 in_position;
uniform mat4 u_projection;
uniform vec3 u_center;
uniform vec3 u_pole;
uniform float u_inner;
uniform float u_outer;
uniform float u_pixel_size;
out float v_radial;
out vec3 v_ring_position;
void main() {
    vec3 tangent = normalize(cross(u_pole, vec3(0,0,1)));
    vec3 bitangent = cross(u_pole,tangent);
    // Expand the mesh past its optical edges so fractional pixel coverage is
    // also drawn just outside the original annulus at all zoom levels.
    float fringe = u_pixel_size/max(abs(u_pole.z),0.05);
    float r = mix(max(0.0,u_inner-fringe),u_outer+fringe,in_position.y);
    v_radial = (r-u_inner)/max(u_outer-u_inner,1e-12);
    v_ring_position = r*(tangent*in_position.x + bitangent*in_position.z);
    gl_Position = u_projection * vec4(u_center + v_ring_position,1.0);
}
