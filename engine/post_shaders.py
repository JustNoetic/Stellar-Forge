bloom_downsample_shader_vs = """
#version 460 core
in vec2 in_position;
out vec2 v_texcoord;
void main() {
    v_texcoord = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

bloom_downsample_shader_fs = """
#version 460 core
in vec2 v_texcoord;
out vec4 out_color;
uniform sampler2D u_texture;
uniform vec2 u_texel_size;
uniform float u_threshold;

void main() {
    // 13-tap downsample for smooth, anti-aliased bloom extraction
    vec2 texel = u_texel_size;
    
    vec3 a = texture(u_texture, v_texcoord + vec2(-2.0,  2.0) * texel).rgb;
    vec3 b = texture(u_texture, v_texcoord + vec2( 0.0,  2.0) * texel).rgb;
    vec3 c = texture(u_texture, v_texcoord + vec2( 2.0,  2.0) * texel).rgb;
    
    vec3 d = texture(u_texture, v_texcoord + vec2(-2.0,  0.0) * texel).rgb;
    vec3 e = texture(u_texture, v_texcoord + vec2( 0.0,  0.0) * texel).rgb;
    vec3 f = texture(u_texture, v_texcoord + vec2( 2.0,  0.0) * texel).rgb;
    
    vec3 g = texture(u_texture, v_texcoord + vec2(-2.0, -2.0) * texel).rgb;
    vec3 h = texture(u_texture, v_texcoord + vec2( 0.0, -2.0) * texel).rgb;
    vec3 i = texture(u_texture, v_texcoord + vec2( 2.0, -2.0) * texel).rgb;
    
    vec3 j = texture(u_texture, v_texcoord + vec2(-1.0,  1.0) * texel).rgb;
    vec3 k = texture(u_texture, v_texcoord + vec2( 1.0,  1.0) * texel).rgb;
    vec3 l = texture(u_texture, v_texcoord + vec2(-1.0, -1.0) * texel).rgb;
    vec3 m = texture(u_texture, v_texcoord + vec2( 1.0, -1.0) * texel).rgb;
    
    vec3 color = e * 0.125;
    color += (a+c+g+i)*0.03125;
    color += (b+d+f+h)*0.0625;
    color += (j+k+l+m)*0.125;
    
    // Apply threshold (only on first pass, u_threshold > 0.0)
    if (u_threshold > 0.0) {
        float brightness = max(color.r, max(color.g, color.b));
        if (brightness < u_threshold) {
            color = vec3(0.0);
        } else {
            // Soft thresholding
            color *= smoothstep(u_threshold, u_threshold + 0.5, brightness);
            
            // Make 10x less sensitive
            color *= 0.2;
            
            // Softly compress extreme brightness so it doesn't spread infinitely
            color = color / (1.0 + color * 0.05);
        }
    }
    
    out_color = vec4(color, 1.0);
}
"""

bloom_upsample_shader_vs = bloom_downsample_shader_vs

bloom_upsample_shader_fs = """
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
"""

composite_shader_vs = bloom_downsample_shader_vs

composite_shader_fs = """
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
    
    out_color = vec4(final_color, 1.0);
}
"""
