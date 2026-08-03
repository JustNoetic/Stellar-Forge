#version 460 core
in vec4 f_color;
in float f_clip_z;
uniform float u_far;
uniform float u_depth_C;
out vec4 out_color;
void main() {
    if (f_clip_z <= 0.0) discard;
    if (f_color.a < 0.01) discard; // discard dashes
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    // Decode sRGB to Linear and boost intensity to survive ACES tonemapping
    vec3 linear_color = pow(f_color.rgb, vec3(2.2)) * 4.0;
    out_color = vec4(linear_color, f_color.a);
}
