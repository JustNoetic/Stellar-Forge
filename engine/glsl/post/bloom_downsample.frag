#version 460 core
in vec2 v_texcoord;
out vec4 out_color;
uniform sampler2D u_texture;
uniform vec2 u_texel_size;
uniform float u_threshold;

vec3 apply_thresh(vec3 tap, float thresh) {
    if (thresh <= 0.0) return tap;
    float brightness = max(tap.r, max(tap.g, tap.b));
    if (brightness >= thresh) {
        tap *= smoothstep(thresh, thresh + 0.5, brightness);
        tap *= 0.2;
        return tap / (1.0 + tap * 0.05);
    }
    return vec3(0.0);
}

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
    
    // Apply threshold (only on first pass, u_threshold > 0.0)
    if (u_threshold > 0.0) {
        a = apply_thresh(a, u_threshold);
        b = apply_thresh(b, u_threshold);
        c = apply_thresh(c, u_threshold);
        d = apply_thresh(d, u_threshold);
        e = apply_thresh(e, u_threshold);
        f = apply_thresh(f, u_threshold);
        g = apply_thresh(g, u_threshold);
        h = apply_thresh(h, u_threshold);
        i = apply_thresh(i, u_threshold);
        j = apply_thresh(j, u_threshold);
        k = apply_thresh(k, u_threshold);
        l = apply_thresh(l, u_threshold);
        m = apply_thresh(m, u_threshold);
    }
    
    vec3 color = e * 0.125;
    color += (a + c + g + i) * 0.03125;
    color += (b + d + f + h) * 0.0625;
    color += (j + k + l + m) * 0.125;
    
    out_color = vec4(color, 1.0);
}
