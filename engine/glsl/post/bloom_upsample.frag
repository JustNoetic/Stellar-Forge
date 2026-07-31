#version 460 core
in vec2 v_texcoord;
out vec4 out_color;
uniform sampler2D u_texture;
uniform vec2 u_texel_size;
uniform float u_radius;

void main() {
    // 9-tap upsample (tent filter)
    vec2 texel = u_texel_size * u_radius;
    
    vec3 a = texture(u_texture, v_texcoord + vec2(-1.0,  1.0) * texel).rgb;
    vec3 b = texture(u_texture, v_texcoord + vec2( 0.0,  1.0) * texel).rgb;
    vec3 c = texture(u_texture, v_texcoord + vec2( 1.0,  1.0) * texel).rgb;
    
    vec3 d = texture(u_texture, v_texcoord + vec2(-1.0,  0.0) * texel).rgb;
    vec3 e = texture(u_texture, v_texcoord + vec2( 0.0,  0.0) * texel).rgb;
    vec3 f = texture(u_texture, v_texcoord + vec2( 1.0,  0.0) * texel).rgb;
    
    vec3 g = texture(u_texture, v_texcoord + vec2(-1.0, -1.0) * texel).rgb;
    vec3 h = texture(u_texture, v_texcoord + vec2( 0.0, -1.0) * texel).rgb;
    vec3 i = texture(u_texture, v_texcoord + vec2( 1.0, -1.0) * texel).rgb;
    
    vec3 color = e * 4.0;
    color += (b+d+f+h) * 2.0;
    color += (a+c+g+i) * 1.0;
    color *= 1.0 / 16.0;
    
    out_color = vec4(color, 1.0);
}
