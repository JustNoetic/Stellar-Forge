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
