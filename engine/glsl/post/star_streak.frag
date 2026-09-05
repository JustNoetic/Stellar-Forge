#version 460 core

in vec2 v_texcoord;
out vec4 out_color;

uniform sampler2D u_bright_texture; // MIP 0 (core needle)
uniform sampler2D u_bloom_mip1;     // MIP 1 (inner body)
uniform sampler2D u_bloom_mip2;     // MIP 2 (mid tail)
uniform sampler2D u_bloom_mip3;     // MIP 3 (outer rays)

uniform vec2 u_texel_size;
uniform int u_spike_count;          // 4, 6, or 8
uniform float u_spike_angle;        // Base rotation angle in radians
uniform float u_spike_length;       // Length/spread scalar
uniform float u_dispersion;         // Chromatic dispersion factor
uniform float u_intensity;          // Spike intensity multiplier

const float PI = 3.14159265358979323846;

// Precomputed geometric tap strides and Fraunhofer power-law falloff weights
const float D0[2] = float[2](2.0, 6.5);
const float W0[2] = float[2](0.85, 0.45);

const float D1[2] = float[2](16.0, 38.0);
const float W1[2] = float[2](0.24, 0.13);

const float D2[2] = float[2](80.0, 160.0);
const float W2[2] = float[2](0.07, 0.035);

const float D3[2] = float[2](300.0, 520.0);
const float W3[2] = float[2](0.018, 0.008);

void main() {
    int num_axes = max(2, u_spike_count / 2);
    float angle_step = PI / float(num_axes);
    
    vec3 total_spikes = vec3(0.0);
    float len_scale = max(0.2, u_spike_length);
    float disp = clamp(u_dispersion, 0.0, 0.08);
    bool use_disp = disp > 0.002;

    for (int axis = 0; axis < num_axes; ++axis) {
        float theta = u_spike_angle + float(axis) * angle_step;
        vec2 dir = vec2(cos(theta), sin(theta));

        for (int side = 0; side < 2; ++side) {
            vec2 arm_dir = (side == 0) ? dir : -dir;
            vec2 step_dir = arm_dir * u_texel_size * len_scale;

            // Phase 1 (MIP 0) - Razor-sharp core
            total_spikes += texture(u_bright_texture, v_texcoord + step_dir * D0[0]).rgb * W0[0];
            total_spikes += texture(u_bright_texture, v_texcoord + step_dir * D0[1]).rgb * W0[1];

            // Phase 2 (MIP 1) - Smooth inner continuous body
            total_spikes += texture(u_bloom_mip1, v_texcoord + step_dir * D1[0]).rgb * W1[0];
            total_spikes += texture(u_bloom_mip1, v_texcoord + step_dir * D1[1]).rgb * W1[1];

            // Phase 3 (MIP 2) - Mid tail with spectral dispersion
            if (use_disp) {
                float r0 = texture(u_bloom_mip2, v_texcoord + step_dir * (D2[0] * (1.0 + disp))).r;
                float g0 = texture(u_bloom_mip2, v_texcoord + step_dir * D2[0]).g;
                float b0 = texture(u_bloom_mip2, v_texcoord + step_dir * (D2[0] * (1.0 - disp))).b;
                total_spikes += vec3(r0, g0, b0) * W2[0];

                float r1 = texture(u_bloom_mip2, v_texcoord + step_dir * (D2[1] * (1.0 + disp))).r;
                float g1 = texture(u_bloom_mip2, v_texcoord + step_dir * D2[1]).g;
                float b1 = texture(u_bloom_mip2, v_texcoord + step_dir * (D2[1] * (1.0 - disp))).b;
                total_spikes += vec3(r1, g1, b1) * W2[1];
            } else {
                total_spikes += texture(u_bloom_mip2, v_texcoord + step_dir * D2[0]).rgb * W2[0];
                total_spikes += texture(u_bloom_mip2, v_texcoord + step_dir * D2[1]).rgb * W2[1];
            }

            // Phase 4 (MIP 3) - Long radiant outer rays
            if (use_disp) {
                float r2 = texture(u_bloom_mip3, v_texcoord + step_dir * (D3[0] * (1.0 + disp * 1.5))).r;
                float g2 = texture(u_bloom_mip3, v_texcoord + step_dir * D3[0]).g;
                float b2 = texture(u_bloom_mip3, v_texcoord + step_dir * (D3[0] * (1.0 - disp * 1.5))).b;
                total_spikes += vec3(r2, g2, b2) * W3[0];

                float r3 = texture(u_bloom_mip3, v_texcoord + step_dir * (D3[1] * (1.0 + disp * 1.5))).r;
                float g3 = texture(u_bloom_mip3, v_texcoord + step_dir * D3[1]).g;
                float b3 = texture(u_bloom_mip3, v_texcoord + step_dir * (D3[1] * (1.0 - disp * 1.5))).b;
                total_spikes += vec3(r3, g3, b3) * W3[1];
            } else {
                total_spikes += texture(u_bloom_mip3, v_texcoord + step_dir * D3[0]).rgb * W3[0];
                total_spikes += texture(u_bloom_mip3, v_texcoord + step_dir * D3[1]).rgb * W3[1];
            }
        }
    }

    total_spikes *= (u_intensity / float(num_axes * 2));
    out_color = vec4(total_spikes, 1.0);
}
