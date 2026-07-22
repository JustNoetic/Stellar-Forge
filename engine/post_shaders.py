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
    
    // Apply threshold (only on first pass, u_threshold > 0.0)
    // We apply it per-tap rather than post-accumulation to prevent small bright sources
    // from becoming pixelated/blocky after the 13-tap filter smears them.
    vec3 color = vec3(0.0);
    
    vec3 taps[13] = vec3[](a, b, c, d, e, f, g, h, i, j, k, l, m);
    
    if (u_threshold > 0.0) {
        for (int idx=0; idx<13; idx++) {
            float brightness = max(taps[idx].r, max(taps[idx].g, taps[idx].b));
            if (brightness >= u_threshold) {
                taps[idx] *= smoothstep(u_threshold, u_threshold + 0.5, brightness);
                taps[idx] *= 0.2;
                taps[idx] = taps[idx] / (1.0 + taps[idx] * 0.05);
            } else {
                taps[idx] = vec3(0.0);
            }
        }
    }
    
    color = taps[4] * 0.125;
    color += (taps[0]+taps[2]+taps[6]+taps[8])*0.03125;
    color += (taps[1]+taps[3]+taps[5]+taps[7])*0.0625;
    color += (taps[9]+taps[10]+taps[11]+taps[12])*0.125;
    
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
    
    // Gamma correction for sRGB monitors
    final_color = pow(final_color, vec3(1.0 / 2.2));
    
    out_color = vec4(final_color, 1.0);
}
"""

taa_resolve_shader_vs = bloom_downsample_shader_vs

taa_resolve_shader_fs = """
#version 460 core
in vec2 v_texcoord;
out vec4 out_color;

uniform sampler2D u_current_color;
uniform sampler2D u_history_color;
uniform sampler2D u_depth_texture;

uniform mat4 u_proj;
uniform mat4 u_inv_view;
uniform mat4 u_prev_view_proj;

uniform vec2 u_texel_size;
uniform float u_depth_C;
uniform float u_far;

void main() {
    float d = texture(u_depth_texture, v_texcoord).r;
    vec4 current_sample = texture(u_current_color, v_texcoord);
    vec3 current_color = current_sample.rgb;

    // Figure out where to sample the history buffer for this pixel.
    bool is_background = (d >= 0.99999);
    vec2 prev_uv;

    if (is_background) {
        // No depth was written for this pixel (empty space or a depthless overlay
        // such as the atmosphere limb/halo, which is rendered with depth writes
        // disabled). We cannot reconstruct a world position to reproject, so fall
        // back to a screen-space identity mapping. This still lets TAA temporally
        // average per-frame stochastic noise (e.g. the atmosphere IGN jitter) in
        // those pixels instead of letting it shimmer. The neighborhood variance
        // clamp below rejects stale history on camera movement, so no ghosting.
        prev_uv = v_texcoord;
    } else {
        // Reconstruct view space depth w from log depth d
        float w = (pow(2.0, d * log2(u_depth_C * u_far + 1.0)) - 1.0) / u_depth_C;

        // Reconstruct view space position V using current projection matrix
        vec2 ndc = v_texcoord * 2.0 - 1.0;
        vec3 V;
        V.z = -w;
        V.x = w * (ndc.x - u_proj[2][0]) / u_proj[0][0];
        V.y = w * (ndc.y - u_proj[2][1]) / u_proj[1][1];

        // View space to current world space
        vec4 world_pos = u_inv_view * vec4(V, 1.0);

        // Current world space to previous clip space
        vec4 prev_clip = u_prev_view_proj * world_pos;
        vec3 prev_ndc = prev_clip.xyz / prev_clip.w;
        prev_uv = prev_ndc.xy * 0.5 + 0.5;

        // If reprojected coordinates are out of screen bounds, discard history
        if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
            out_color = vec4(current_color, current_sample.a);
            return;
        }
    }

    // Variance-based neighborhood clamping to resolve ghosting
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec3 c = texture(u_current_color, v_texcoord + vec2(x, y) * u_texel_size).rgb;
            m1 += c;
            m2 += c * c;
        }
    }

    vec3 mean = m1 / 9.0;
    vec3 stddev = sqrt(max(vec3(0.0), (m2 / 9.0) - (mean * mean)));

    // Scale standard deviation for clipping box
    float gamma = 1.0;
    vec3 box_min = mean - gamma * stddev;
    vec3 box_max = mean + gamma * stddev;

    vec4 history_sample = texture(u_history_color, prev_uv);
    vec3 history_color = history_sample.rgb;
    history_color = clamp(history_color, box_min, box_max);

    float blend = 0.1; // 10% current, 90% history
    float blended_alpha = mix(history_sample.a, current_sample.a, blend);
    out_color = vec4(mix(history_color, current_color, blend), blended_alpha);
}
"""

atmo_composite_shader_vs = bloom_downsample_shader_vs

atmo_composite_shader_fs = """
#version 460 core
in vec2 v_texcoord;
out vec4 out_color;

uniform sampler2D u_atmo_texture;

void main() {
    out_color = texture(u_atmo_texture, v_texcoord);
}
"""

