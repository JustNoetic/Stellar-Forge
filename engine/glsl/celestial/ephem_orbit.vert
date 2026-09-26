#version 460 core
in vec3 in_pos;
in float in_time;
uniform mat4 projection;
uniform mat4 view_rot;
uniform dvec4 u_cam_pos_double;
uniform vec3 u_bary_rel_cam;
uniform vec3 u_color;
uniform float u_depth_C;
uniform float u_far;

uniform float u_current_et;
uniform float u_trail_sec;
uniform float u_lead_sec;
uniform float u_mission_start_et;
uniform float u_mission_end_et;
uniform int u_fade_mode;     // 0 = Sliding window, 1 = Full mission
uniform int u_hide_outside;  // 1 = Hide if current_et outside [start, end], 0 = don't

#include "common/refraction.glsl"

out vec4 f_color;
out float f_clip_z;

void main() {
    float alpha = 1.0;
    bool is_in_mission = true;
    if (u_mission_start_et < u_mission_end_et) {
        if (u_current_et < u_mission_start_et || u_current_et > u_mission_end_et) {
            is_in_mission = false;
        }
    }

    if (!is_in_mission && u_hide_outside != 0) {
        float fade_margin = 86400.0; // 1 day smooth fade at start / end
        if (u_current_et < u_mission_start_et) {
            float diff = u_mission_start_et - u_current_et;
            if (diff >= fade_margin) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                f_clip_z = -1.0;
                f_color = vec4(0.0);
                return;
            }
            alpha *= (1.0 - diff / fade_margin);
        } else if (u_current_et > u_mission_end_et) {
            float diff = u_current_et - u_mission_end_et;
            if (diff >= fade_margin) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                f_clip_z = -1.0;
                f_color = vec4(0.0);
                return;
            }
            alpha *= (1.0 - diff / fade_margin);
        }
    }

    if (u_fade_mode == 0) {
        // Sliding window fade
        if (is_in_mission) {
            float dt = in_time - u_current_et; // negative = past, positive = future
            if (dt <= 0.0) {
                // Past trail
                if (dt < -u_trail_sec) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    f_clip_z = -1.0;
                    f_color = vec4(0.0);
                    return;
                }
                float frac = clamp((dt + u_trail_sec) / max(u_trail_sec, 1.0), 0.0, 1.0);
                alpha *= smoothstep(0.0, 1.0, frac);
            } else {
                // Future forecast
                if (dt > u_lead_sec) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    f_clip_z = -1.0;
                    f_color = vec4(0.0);
                    return;
                }
                float frac = clamp(1.0 - (dt / max(u_lead_sec, 1.0)), 0.0, 1.0);
                alpha *= smoothstep(0.0, 1.0, frac) * 0.7; // Lead path is slightly softer
            }
        } else {
            // Outside mission timeline and u_hide_outside == 0:
            // Display full trajectory in a subtle dimmed preview mode
            alpha *= 0.35;
        }
    }

    // High precision camera-relative positioning (eliminates float32 quantization jitter)
    vec3 eye_pos = in_pos + u_bary_rel_cam;
    vec3 cam_pos = vec3(u_cam_pos_double.xyz);
    vec3 app_eye_pos = apply_refraction_eye(eye_pos, cam_pos);

    vec3 final_rgb = u_color * 0.4;
    float max_c = max(final_rgb.r, max(final_rgb.g, final_rgb.b));
    if (max_c < 0.25 && max_c > 0.0001) {
        final_rgb = final_rgb * (0.25 / max_c);
    } else if (max_c <= 0.0001) {
        final_rgb = vec3(0.25);
    }

    f_color = vec4(final_rgb, alpha);
    gl_Position = projection * view_rot * vec4(app_eye_pos, 1.0);
    f_clip_z = gl_Position.w;
}

