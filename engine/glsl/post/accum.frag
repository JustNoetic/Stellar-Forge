#version 460 core
in vec2 v_texcoord;
out vec4 out_color;

uniform sampler2D u_history;
uniform sampler2D u_current;
uniform float u_blend_weight;

void main() {
    vec4 hist_val = texture(u_history, v_texcoord);
    vec4 curr_val = texture(u_current, v_texcoord);
    out_color = mix(hist_val, curr_val, u_blend_weight);
}
