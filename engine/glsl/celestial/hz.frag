#version 460 core
in float f_clip_z;
in float f_radius_pct;
uniform vec4 u_color;
uniform float u_far;
uniform float u_depth_C;
out vec4 out_color;

void main() {
    if (f_clip_z <= 0.0) discard;
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    float alpha = sin(f_radius_pct * 3.14159265);
    vec3 linear_color = pow(u_color.rgb, vec3(2.2)) * 4.0;
    out_color = vec4(linear_color, u_color.a * alpha);
}
