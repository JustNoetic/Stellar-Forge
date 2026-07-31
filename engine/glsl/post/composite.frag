#version 460 core
in vec2 v_texcoord;
out vec4 out_color;
uniform sampler2D u_main_texture;
uniform sampler2D u_bloom_texture;
uniform float u_bloom_intensity;

// ACES Tone Mapping
vec3 ACESFilm(vec3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    vec3 hdr_color = texture(u_main_texture, v_texcoord).rgb;
    vec3 bloom_color = texture(u_bloom_texture, v_texcoord).rgb;
    
    vec3 final_color = hdr_color + bloom_color * u_bloom_intensity;
    
    // Measure luminance before tone mapping to detect extreme glare
    float pre_luma = dot(final_color, vec3(0.2126, 0.7152, 0.0722));
    
    // Apply Tone Mapping
    final_color = ACESFilm(final_color);
    
    // Violet glare for extreme brightness (simulating hurt eyes)
    float eye_hurt_blend = smoothstep(10.0, 150.0, pre_luma);
    vec3 violet_tint = vec3(0.65, 0.45, 1.0); // Violet
    final_color = mix(final_color, violet_tint, eye_hurt_blend * 0.03);
    
    // Gamma correction for sRGB monitors
    final_color = pow(final_color, vec3(1.0 / 2.2));
    
    out_color = vec4(final_color, 1.0);
}
