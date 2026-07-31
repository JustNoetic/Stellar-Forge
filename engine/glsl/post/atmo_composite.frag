#version 460 core
in vec2 v_texcoord;
out vec4 out_color;

uniform sampler2D u_atmo_texture;

void main() {
    out_color = texture(u_atmo_texture, v_texcoord);
}
