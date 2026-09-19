#version 330 core

in vec2 v_uv;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform sampler2D u_sky_view_lut;

uniform mat4 u_inv_proj;
uniform mat4 u_inv_view;
uniform vec3 u_cam_pos;
uniform float u_planet_radius;
uniform float u_atmo_radius;

uniform sampler2D u_ring_tex;
uniform float u_ring_inner;
uniform float u_ring_outer;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_atmo_tint;
uniform float u_scale_height;
uniform float u_density;
uniform bool u_volumetric_shadow;
uniform bool u_analytical_slicing;
uniform float u_ring_tex_res;

out vec4 out_color;

const float PI = 3.14159265358979323846;

// Ray-sphere intersection returning (near, far) distances
vec2 ray_sphere_intersect(vec3 origin, vec3 dir, float radius) {
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float delta = b * b - c;
    if (delta < 0.0) return vec2(-1.0, -1.0);
    float sq = sqrt(delta);
    return vec2(-b - sq, -b + sq);
}

// Atmospheric density at point along ray
float atmo_density_at(vec3 origin, vec3 dir, float s, float planet_r, float H) {
    vec3 P = origin + s * dir;
    float h = max(0.0, length(P) - planet_r);
    return exp(-h / max(H, 1e-4));
}

void main() {
    vec3 scene_color = texture(u_scene_color, v_uv).rgb;

    // Reconstruct view ray V in world space from near-plane NDC point
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 p_near_clip = vec4(ndc, -1.0, 1.0);
    vec4 p_near_view = u_inv_proj * p_near_clip;
    vec3 ray_dir_view = normalize(p_near_view.xyz / p_near_view.w);
    vec3 V = normalize((u_inv_view * vec4(ray_dir_view, 0.0)).xyz);

    float D = length(u_cam_pos);
    vec3 u_hat = u_cam_pos / max(D, 1e-6);
    vec3 C_dir = -u_hat; // Direction towards planet center

    // Build orthonormal frame matching sky_view_lut.frag
    vec3 ref = (abs(C_dir.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(C_dir, ref));
    vec3 up = cross(C_dir, right);

    float sin_horizon = clamp(u_planet_radius / D, 0.0, 1.0);
    float theta_horizon = asin(sin_horizon);

    bool cam_inside_atmo = (D < u_atmo_radius);
    float max_theta = cam_inside_atmo ? PI : asin(clamp(u_atmo_radius / D, 0.0, 1.0));

    float cos_theta = clamp(dot(V, C_dir), -1.0, 1.0);
    float theta = acos(cos_theta);

    // If ray misses the atmosphere shell completely (only possible when outside)
    if (!cam_inside_atmo && theta > max_theta) {
        out_color = vec4(scene_color, 1.0);
        return;
    }

    // Piecewise mapping for elevation v:
    // [0, 0.5] for surface rays (theta in [0, theta_horizon])
    // [0.5, 1.0] for atmosphere limb/sky dome rays (theta in [theta_horizon, max_theta])
    float v = 0.0;
    if (theta <= theta_horizon) {
        v = (theta / max(theta_horizon, 1e-5)) * 0.5;
    } else {
        v = 0.5 + 0.5 * (theta - theta_horizon) / max(max_theta - theta_horizon, 1e-5);
    }

    // Azimuth u around C_dir
    float x_proj = dot(V, right);
    float y_proj = dot(V, up);
    float phi = atan(y_proj, x_proj); // in [-PI, PI]
    float u = fract(phi / (2.0 * PI));

    // Distance to atmosphere entry and exit
    vec2 t_atmo = ray_sphere_intersect(u_cam_pos, V, u_atmo_radius);
    if (t_atmo.y < 0.0) {
        out_color = vec4(scene_color, 1.0);
        return;
    }
    float s_start = max(0.0, t_atmo.x);
    float s_end = t_atmo.y;

    // Check if planet ground is hit
    vec2 t_planet = ray_sphere_intersect(u_cam_pos, V, u_planet_radius);
    if (t_planet.x > 0.0) {
        s_end = min(s_end, t_planet.x);
    }

    // Check depth buffer to see if foreground geometry is closer
    float raw_depth = texture(u_scene_depth, v_uv).r;
    float geom_dist = 1e9;
    if (raw_depth < 0.99999) {
        vec4 p_clip = vec4(ndc, raw_depth * 2.0 - 1.0, 1.0);
        vec4 p_view = u_inv_proj * p_clip;
        geom_dist = length(p_view.xyz / p_view.w);
        s_end = min(s_end, geom_dist);
    }

    // Sample O(1) from the precomputed Sky-View LUT
    vec4 atmo = texture(u_sky_view_lut, vec2(u, clamp(v, 0.0, 1.0)));
    vec3 L_scatter = atmo.rgb;
    float T = atmo.a;

    if (!cam_inside_atmo && theta > theta_horizon) {
        float t_limb = (theta - theta_horizon) / max(max_theta - theta_horizon, 1e-5);
        float limb_fade = clamp((1.0 - t_limb) * 128.0, 0.0, 1.0);
        L_scatter *= limb_fade;
    }

    // =========================================================================
    // Phase 3: Analytical Depth Slicing for Volumetric Ring Shadow in Atmosphere
    // =========================================================================
    vec3 delta_L = vec3(0.0);
    vec3 slice_total = vec3(0.0);
    bool has_shadow_interval = false;
    float total_shadow_len = 0.0;
    float total_ring_blocked_accum = 0.0;
    float shadow_step_count = 0.0;

    if (u_volumetric_shadow && abs(u_sun_dir.y) > 1e-5 && s_start < s_end) {
        vec3 L = normalize(u_sun_dir);

        // Project ray P(s) = C + s*V onto the ring plane Y = 0 along sunlight vector L:
        // P_ring(s) = A + s * B, with A_y = 0 and B_y = 0
        vec3 A = u_cam_pos - (u_cam_pos.y / L.y) * L;
        vec3 B = V - (V.y / L.y) * L;

        // Radial distance squared on ring plane: r^2(s) = qa * s^2 + qb * s + qc
        float qa = B.x * B.x + B.z * B.z;
        float qb = 2.0 * (A.x * B.x + A.z * B.z);
        float qc = A.x * A.x + A.z * A.z;

        // Shadow validity window: lambda = -(C_y + s*V_y)/L_y > 0
        float w_min = s_start;
        float w_max = s_end;

        if (abs(V.y) > 1e-6) {
            float s_crit = -u_cam_pos.y / V.y;
            if ((V.y / L.y) > 0.0) {
                w_max = min(w_max, s_crit);
            } else {
                w_min = max(w_min, s_crit);
            }
        } else {
            if (-u_cam_pos.y / L.y <= 0.0) {
                w_min = 1e9; // Entire ray is unshadowed
            }
        }

        if (w_min < w_max) {
            vec2 intervals[2];
            int num_intervals = 0;

            if (qa > 1e-6) {
                float disc_out = qb * qb - 4.0 * qa * (qc - u_ring_outer * u_ring_outer);
                if (disc_out >= 0.0) {
                    float sq_out = sqrt(disc_out);
                    float s_out_min = (-qb - sq_out) / (2.0 * qa);
                    float s_out_max = (-qb + sq_out) / (2.0 * qa);

                    float disc_in = qb * qb - 4.0 * qa * (qc - u_ring_inner * u_ring_inner);

                    if (disc_in <= 0.0) {
                        intervals[0] = vec2(s_out_min, s_out_max);
                        num_intervals = 1;
                    } else {
                        float sq_in = sqrt(disc_in);
                        float s_in_min = (-qb - sq_in) / (2.0 * qa);
                        float s_in_max = (-qb + sq_in) / (2.0 * qa);
                        intervals[0] = vec2(s_out_min, s_in_min);
                        intervals[1] = vec2(s_in_max, s_out_max);
                        num_intervals = 2;
                    }
                }
            } else {
                // Degenerate case: view ray is parallel or anti-parallel to sunlight (V || L).
                // All points along the chord project onto the equatorial ring plane at point A.
                float r_A = length(A.xz);
                if (r_A >= u_ring_inner && r_A <= u_ring_outer) {
                    intervals[0] = vec2(w_min, w_max);
                    num_intervals = 1;
                }
            }

            if (num_intervals > 0) {
                    float sigma_s = u_density;
                    float sigma_t = u_density * 1.1;
                    float inv_sin_sun = 1.0 / max(abs(L.y), 0.05);
                    float ring_span = u_ring_outer - u_ring_inner;
                    float ring_res = (u_ring_tex_res > 1.0) ? u_ring_tex_res : 2048.0;

                    float rho_start = atmo_density_at(u_cam_pos, V, s_start, u_planet_radius, u_scale_height);

                    for (int inv = 0; inv < 2; inv++) {
                        if (inv >= num_intervals) break;

                        float cs = max(intervals[inv].x, w_min);
                        float ce = min(intervals[inv].y, w_max);

                        if (ce > cs + 1e-5) {
                            has_shadow_interval = true;
                            total_shadow_len += (ce - cs);
                            if (u_analytical_slicing) {
                                // Continuous footprint-filtered quadrature (Physically Exact, No Aliasing)
                                const int QUAD_STEPS = 8;
                                float ds = (ce - cs) / float(QUAD_STEPS);

                                float T_run = 1.0;
                                if (cs > s_start + 1e-4) {
                                    float s_front_mid = 0.5 * (s_start + cs);
                                    float rho_front_mid = atmo_density_at(u_cam_pos, V, s_front_mid, u_planet_radius, u_scale_height);
                                    float rho_cs = atmo_density_at(u_cam_pos, V, cs, u_planet_radius, u_scale_height);
                                    float tau_front = ((cs - s_start) / 6.0) * (rho_start + 4.0 * rho_front_mid + rho_cs);
                                    T_run = exp(-sigma_t * tau_front);
                                }

                                for (int q = 0; q < QUAD_STEPS; q++) {
                                    float s_prev = cs + float(q) * ds;
                                    float s_next = cs + float(q + 1) * ds;
                                    float sq = 0.5 * (s_prev + s_next);

                                    vec3 Pq = u_cam_pos + sq * V;
                                    float rq = length(Pq);
                                    float hq = max(0.0, rq - u_planet_radius);
                                    float rhoq = exp(-hq / max(u_scale_height, 1e-4));

                                    // If air parcel is occluded by the planet body, LUT already zeroed inscatter
                                    vec2 tp_sun = ray_sphere_intersect(Pq, L, u_planet_radius);
                                    if (tp_sun.x > 0.0 && tp_sun.y > tp_sun.x) {
                                        float step_trans = exp(-rhoq * sigma_t * ds);
                                        T_run *= step_trans;
                                        continue;
                                    }

                                    // Footprint-filtered ring opacity across the step
                                    float r_prev = length(A.xz + s_prev * B.xz);
                                    float r_next = length(A.xz + s_next * B.xz);
                                    float r_mid  = length(A.xz + sq * B.xz);

                                    float u_prev = (r_prev - u_ring_inner) / ring_span;
                                    float u_next = (r_next - u_ring_inner) / ring_span;
                                    float u_mid  = (r_mid  - u_ring_inner) / ring_span;

                                    float du = max(1e-5, abs(u_next - u_prev));
                                    float lod = log2(max(1.0, du * ring_res));

                                    float ring_blocked = 0.0;
                                    if (u_mid >= -0.05 && u_mid <= 1.05) {
                                        float raw_a = textureLod(u_ring_tex, vec2(clamp(u_mid, 0.0, 1.0), 0.5), lod).a;
                                        float tau_ring = -log(max(1e-4, 1.0 - raw_a));
                                        ring_blocked = 1.0 - exp(-tau_ring * inv_sin_sun);
                                    }

                                    vec2 tatmo_sun = ray_sphere_intersect(Pq, L, u_atmo_radius);
                                    float dt_sun = max(0.0, tatmo_sun.y) / 4.0;
                                    float tau_sun = 0.0;
                                    for (int k = 0; k < 4; k++) {
                                        vec3 Ps = Pq + (float(k) + 0.5) * dt_sun * L;
                                        float hs = max(0.0, length(Ps) - u_planet_radius);
                                        tau_sun += exp(-hs / max(u_scale_height, 1e-4)) * dt_sun;
                                    }
                                    float trans_to_sun = exp(-sigma_t * tau_sun);

                                    vec3 step_inscatter = u_sun_color * u_atmo_tint * (rhoq * sigma_s) * trans_to_sun * ds;
                                    delta_L += T_run * step_inscatter * ring_blocked;
                                    slice_total += T_run * step_inscatter;
                                    total_ring_blocked_accum += ring_blocked;
                                    shadow_step_count += 1.0;

                                    float step_trans = exp(-rhoq * sigma_t * ds);
                                    T_run *= step_trans;
                                }
                            } else {
                                // Legacy 6-point analytical quadrature across the exact shadow slice
                                float T_run = 1.0;
                                if (cs > s_start + 1e-4) {
                                    vec3 P0 = u_cam_pos + s_start * V;
                                    vec3 Pmid = u_cam_pos + 0.5 * (s_start + cs) * V;
                                    vec3 P1 = u_cam_pos + cs * V;
                                    float h0 = max(0.0, length(P0) - u_planet_radius);
                                    float hmid = max(0.0, length(Pmid) - u_planet_radius);
                                    float h1 = max(0.0, length(P1) - u_planet_radius);
                                    float tau_front = ((cs - s_start) / 6.0) * (
                                        exp(-h0 / max(u_scale_height, 1e-4)) +
                                        4.0 * exp(-hmid / max(u_scale_height, 1e-4)) +
                                        exp(-h1 / max(u_scale_height, 1e-4))
                                    );
                                    T_run = exp(-sigma_t * tau_front);
                                }

                                const int QUAD_STEPS = 6;
                                float ds = (ce - cs) / float(QUAD_STEPS);

                                for (int q = 0; q < QUAD_STEPS; q++) {
                                    float sq = cs + (float(q) + 0.5) * ds;
                                    vec3 Pq = u_cam_pos + sq * V;
                                    float rq = length(Pq);
                                    float hq = max(0.0, rq - u_planet_radius);
                                    float rhoq = exp(-hq / max(u_scale_height, 1e-4));

                                    // If air parcel is occluded by the planet body, LUT already zeroed inscatter
                                    vec2 tp_sun = ray_sphere_intersect(Pq, L, u_planet_radius);
                                    if (tp_sun.x > 0.0 && tp_sun.y > tp_sun.x) {
                                        continue;
                                    }

                                    // Physical ring optical depth at projected coordinates
                                    float t_to_ring = -Pq.y / L.y;
                                    vec3 P_ring = Pq + t_to_ring * L;
                                    float r_ring = length(P_ring.xz);
                                    float u_ring = (r_ring - u_ring_inner) / ring_span;

                                    float ring_blocked = 0.0;
                                    if (u_ring >= 0.0 && u_ring <= 1.0) {
                                        float raw_a = texture(u_ring_tex, vec2(u_ring, 0.5)).a;
                                        float tau_ring = -log(max(1e-4, 1.0 - raw_a));
                                        ring_blocked = 1.0 - exp(-tau_ring * inv_sin_sun);
                                    }

                                    // Sunlight optical depth from parcel to top of atmosphere
                                    vec2 tatmo_sun = ray_sphere_intersect(Pq, L, u_atmo_radius);
                                    float dt_sun = max(0.0, tatmo_sun.y) / 4.0;
                                    float tau_sun = 0.0;
                                    for (int k = 0; k < 4; k++) {
                                        vec3 Ps = Pq + (float(k) + 0.5) * dt_sun * L;
                                        float hs = max(0.0, length(Ps) - u_planet_radius);
                                        tau_sun += exp(-hs / max(u_scale_height, 1e-4)) * dt_sun;
                                    }
                                    float trans_to_sun = exp(-sigma_t * tau_sun);

                                    // In-scattered light erroneously included by the unshadowed LUT
                                    vec3 step_inscatter = u_sun_color * u_atmo_tint * (rhoq * sigma_s) * trans_to_sun * ds;
                                    delta_L += T_run * step_inscatter * ring_blocked;
                                    slice_total += T_run * step_inscatter;
                                    total_ring_blocked_accum += ring_blocked;
                                    shadow_step_count += 1.0;

                                    float step_trans = exp(-rhoq * sigma_t * ds);
                                    T_run *= step_trans;
                                }
                            }
                        }
                    }
                }
            }
        }

    // Subtract volumetric shadow light deficit
    vec3 L_atmo = L_scatter;
    if (has_shadow_interval) {
        float chord_len = max(s_end - s_start, 1e-4);
        float chord_coverage = clamp(total_shadow_len / chord_len, 0.0, 1.0);

        vec3 shadow_factor;
        if (dot(slice_total, vec3(1.0)) > 1e-6) {
            shadow_factor = clamp(delta_L / max(slice_total, vec3(1e-6)), vec3(0.0), vec3(1.0));
        } else {
            float avg_blocked = (shadow_step_count > 0.0) ? (total_ring_blocked_accum / shadow_step_count) : 1.0;
            shadow_factor = vec3(clamp(avg_blocked, 0.0, 1.0));
        }

        vec3 alpha = clamp(slice_total / max(L_scatter, vec3(1e-6)), vec3(0.0), vec3(1.0));
        vec3 blend_alpha = smoothstep(vec3(0.3), vec3(0.7), alpha);
        vec3 blend_cov = vec3(smoothstep(0.7, 0.98, chord_coverage));
        vec3 blend = max(blend_alpha, blend_cov);

        vec3 eff_slice = mix(slice_total, L_scatter, blend);
        L_atmo = max(vec3(0.0), L_scatter - eff_slice * shadow_factor);
    }

    // Foreground ring occlusion: if the view ray pierces a foreground ring before entering the atmosphere
    if (abs(V.y) > 1e-6) {
        float t_ring = -u_cam_pos.y / V.y;
        if (t_ring > 0.001 && t_ring < s_start) {
            vec3 P_ring = u_cam_pos + t_ring * V;
            float r_ring = length(P_ring.xz);
            if (r_ring >= u_ring_inner && r_ring <= u_ring_outer) {
                float u_ring = (r_ring - u_ring_inner) / (u_ring_outer - u_ring_inner);
                float ring_alpha = texture(u_ring_tex, vec2(u_ring, 0.5)).a;
                L_atmo *= (1.0 - ring_alpha);
            }
        }
    }

    // Composite: attenuated background scene radiance + true in-scattered airlight
    vec3 final_color = scene_color * T + L_atmo;
    out_color = vec4(final_color, 1.0);
}

