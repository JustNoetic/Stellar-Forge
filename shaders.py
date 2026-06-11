sphere_vertex_shader = """
#version 460 core
in vec3 in_position;
in vec3 in_normal;
in vec3 in_offset;
in vec3 in_color;
in float in_radius;
in float in_min_size;
in float in_is_star;
in vec3 in_pole;
in float in_oblateness;
in uvec2 in_caster_mask;
in uint in_ring_mask;

#define MAX_CASTERS 64
layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    vec4 u_star_pos_radius;
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad0;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};
uniform float screen_height;
uniform float fov_factor;

out vec3 f_color;
out vec3 f_world_pos;
out vec3 f_normal;
out float f_is_star;
out float f_clip_z;
flat out uvec2 f_caster_mask;
flat out uint f_ring_mask;
void main() {
    f_color = in_color;
    f_caster_mask = in_caster_mask;
    f_ring_mask = in_ring_mask;
    f_is_star = in_is_star;
    float dist = length((view * vec4(in_offset, 1.0)).xyz);
    float apparent_px = (in_radius / dist) * screen_height * fov_factor;
    float final_radius = in_radius;
    
    if (apparent_px < in_min_size) {
        final_radius = (in_min_size * dist) / (screen_height * fov_factor);
    }
    
    vec3 scaled_pos = in_position;
    vec3 adj_normal = in_normal;
    if (in_oblateness > 0.0) {
        float pole_proj = dot(in_position, in_pole);
        scaled_pos -= in_pole * (pole_proj * in_oblateness);
        // Transform normal by inverse-transpose of the squash Jacobian:
        // J = I - f*(p⊗p),  J^(-T) = I + f/(1-f)*(p⊗p)
        float f_inv = in_oblateness / (1.0 - in_oblateness);
        adj_normal = normalize(in_normal + in_pole * (dot(in_normal, in_pole) * f_inv));
    }
    
    vec3 world_pos = (scaled_pos * final_radius) + in_offset;
    f_world_pos = world_pos;
    f_normal = adj_normal;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    f_clip_z = gl_Position.w;
}
"""

sphere_fragment_shader = """
#version 460 core
#define MAX_CASTERS 64
#define MAX_RING_PLANES 16

in vec3 f_color;
in vec3 f_world_pos;
in vec3 f_normal;
in float f_is_star;
in float f_clip_z;
flat in uvec2 f_caster_mask;
flat in uint f_ring_mask;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    vec4 u_star_pos_radius;
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad0;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};

// Ring shadow planes (sphere-only)
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec3 u_ring_params[MAX_RING_PLANES];
uniform vec3 u_ring_colors[MAX_RING_PLANES];
uniform sampler2D u_ring_gradients;
uniform int u_num_ring_planes;
uniform int u_atmo_quality;

out vec4 out_color;

float get_oblate_radius(float r_eq, float oblateness, vec3 pole, vec3 L, vec3 perp_vec) {
    if (oblateness <= 0.0) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float f_factor = 1.0 - oblateness;
    float R_minor = r_eq * sqrt(PdotL * PdotL + f_factor * f_factor * P_proj_len * P_proj_len);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / R_minor) * (y / R_minor));
    return perp_len / denom;
}

void main() {
    if (f_is_star > 0.5) {
        // Star is self-luminous — no shading
        out_color = vec4(f_color, 1.0);
    } else {
        vec3 frag_to_star = u_star_pos_radius.xyz - f_world_pos;
        float dist_to_star = length(frag_to_star);
        vec3 L = frag_to_star / dist_to_star;
        
        // Angular radius of the star for soft penumbra at terminator
        float sin_alpha = clamp(u_star_pos_radius.w / dist_to_star, 0.0, 1.0);
        
        // Lambert cosine law: brightness decreases with angle
        // Combined with soft terminator from finite star angular size
        float NdotL = dot(normalize(f_normal), L);
        float diffuse = clamp((NdotL + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);
        
        // === Analytical eclipse shadows ===
        vec3 shadow = vec3(1.0);
        for (int j = 0; j < u_num_casters; j++) {
            if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
            else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }
            
            vec3 caster_pos = u_casters[j].xyz;
            float caster_r = u_casters[j].w;
            
            vec3 frag_to_caster = caster_pos - f_world_pos;
            float dist_to_caster = length(frag_to_caster);
            
            // Skip self — fragment is on this body's surface (tight threshold to prevent self-shadow acne without skipping close moons)
            if (dist_to_caster < caster_r * 1.02) continue;
            
            // Project caster onto the fragment-to-star ray
            float t = dot(frag_to_caster, L);
            
            // Caster must be between fragment and star
            if (t <= 0.0 || t >= dist_to_star) continue;
            
            // Perpendicular distance from caster center to the ray
            vec3 perp_vec = frag_to_caster - t * L;
            float perp = length(perp_vec);
            
            // Account for oblateness
            float oblateness = u_caster_poles_obl[j].w;
            if (oblateness > 0.0) {
                vec3 pole = u_caster_poles_obl[j].xyz;
                caster_r = get_oblate_radius(caster_r, oblateness, pole, L, perp_vec);
            }
            
            // Project star disc to the caster's distance along the ray
            float r_star_proj = u_star_pos_radius.w * t / dist_to_star;
            
            // Shadow boundaries
            float r_penumbra = caster_r + r_star_proj;  // outer edge
            float r_umbra = abs(caster_r - r_star_proj); // inner edge
            
            // Max occlusion factor (for antumbra/annular eclipses)
            float max_occlusion = (caster_r >= r_star_proj) ? 1.0 : (caster_r * caster_r) / (r_star_proj * r_star_proj);
            
            // Smooth transition from max_occlusion (at umbra) to 0 (at penumbra)
            float occlusion = max_occlusion * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
            
            vec3 caster_shadow = vec3(1.0 - occlusion);
            
            float atmo_h = u_caster_atmos[j].w;
            if (atmo_h > 0.0 && occlusion > 0.0) {
                float depth_into_umbra = clamp(1.0 - (perp / caster_r), 0.0, 1.0);
                vec3 atmo_tint = u_caster_atmos[j].xyz;
                // Much gentler tint deepening
                vec3 deep_tint = pow(atmo_tint, vec3(1.0 + depth_into_umbra * 0.5));
                // Gentler exponential falloff so light reaches deep into the umbra without turning black
                float refraction_intensity = exp(-depth_into_umbra * 1.2) * 1.5;
                caster_shadow += deep_tint * refraction_intensity * occlusion;
            }
            
            shadow *= caster_shadow;
        }
        
        // === Ring shadow on planet surface ===
        // For each ring plane, trace a ray from fragment toward star
        // and check if it intersects within the ring annulus
        for (int k = 0; k < u_num_ring_planes; k++) {
            if ((f_ring_mask & (1u << k)) == 0u) continue;
            
            vec3 ring_center = u_ring_center[k];
            vec3 ring_normal = u_ring_normal[k];
            float inner_r = u_ring_params[k].x;
            float outer_r = u_ring_params[k].y;
            float opacity = u_ring_params[k].z;
            
            // Ray-plane intersection: find t where (frag + t*L) lies on the ring plane
            // Plane equation: dot(P - ring_center, ring_normal) = 0
            float denom = dot(L, ring_normal);
            if (abs(denom) < 1e-8) continue;  // ray parallel to ring plane
            
            float t = dot(ring_center - f_world_pos, ring_normal) / denom;
            
            // Intersection must be between fragment and star (t > 0 and t < dist_to_star)
            if (t <= 0.0 || t >= dist_to_star) continue;
            
            // Compute intersection point and distance from ring center
            vec3 hit = f_world_pos + t * L;
            float dist_from_center = length(hit - ring_center);
            
            // Check if intersection is within the ring annulus
            if (dist_from_center >= inner_r && dist_from_center <= outer_r) {
                // Soft edge falloff at inner and outer boundaries
                float edge_width = (outer_r - inner_r) * 0.05;
                float inner_fade = smoothstep(inner_r, inner_r + edge_width, dist_from_center);
                float outer_fade = smoothstep(outer_r, outer_r - edge_width, dist_from_center);
                float p = (dist_from_center - inner_r) / max(1e-6, outer_r - inner_r);
                float alpha_mult = texture(u_ring_gradients, vec2(p, (float(k) + 0.5) / 16.0)).r;
                float ring_shadow = inner_fade * outer_fade * opacity * alpha_mult;
                shadow *= vec3(1.0 - ring_shadow);
            }
        }
        
        vec3 diffuse_color = vec3(diffuse) * shadow;
        
        // === Moonshine / Planetshine ===
        vec3 bounce_light = vec3(0.0);
        for (int j = 0; j < u_num_casters; j++) {
            if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
            else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }
            
            vec3 caster_pos = u_casters[j].xyz;
            float caster_r = u_casters[j].w;
            vec3 frag_to_caster = caster_pos - f_world_pos;
            float dist = length(frag_to_caster);
            
            if (dist < caster_r * 1.05) continue; // Skip self/very close
            
            vec3 dir_to_caster = frag_to_caster / dist;
            vec3 caster_to_star = normalize(u_star_pos_radius.xyz - caster_pos);
            
            float solid_angle = (caster_r * caster_r) / max(dist * dist, caster_r * caster_r);
            float phase = max(0.0, dot(caster_to_star, -dir_to_caster));
            float NdotC = max(0.0, dot(normalize(f_normal), dir_to_caster));
            
            // Simple form factor & intensity multiplier
            // Boosted moonshine/planetshine intensity to make dark sides glow beautifully
            float intensity = phase * solid_angle * NdotC * 4.0;
            bounce_light += u_caster_colors[j].rgb * intensity;
        }
        
        // === Ringshine ===
        vec3 ring_shine = vec3(0.0);
        for (int k = 0; k < u_num_ring_planes; k++) {
            if ((f_ring_mask & (1u << k)) == 0u) continue;
            
            vec3 ring_center = u_ring_center[k];
            vec3 ring_normal = u_ring_normal[k];
            float inner_r = u_ring_params[k].x;
            float outer_r = u_ring_params[k].y;
            float opacity = u_ring_params[k].z;
            
            vec3 frag_to_ring = ring_center - f_world_pos;
            float dist_to_ring = length(frag_to_ring);
            if (dist_to_ring < 1e-6) continue;
            
            // Determine sun elevation over ring
            float sun_elevation = dot(L, ring_normal);
            
            // Determine fragment hemisphere relative to ring
            float frag_elevation = dot(normalize(f_normal), ring_normal);
            
            // Calculate geometric form factor based on fragment elevation
            // Rings take up max solid angle at mid-latitudes, zero at poles/equator
            float form_factor = abs(frag_elevation) * (1.0 - abs(frag_elevation)) * 4.0; 
            
            // Scale by ring area / distance
            float ring_area = (outer_r * outer_r - inner_r * inner_r);
            float solid_angle = ring_area / max(dist_to_ring * dist_to_ring, ring_area) * 0.1;
            
            float same_hemisphere = sun_elevation * frag_elevation;
            
            // Planet's shadow occlusion on the ring
            // If fragment is on the night side (dot(N, L) < 0), ring overhead might be in shadow
            vec3 N = normalize(f_normal);
            float shadow_occlusion = 1.0;
            if (dot(N, L) < 0.0) {
                // How close is fragment to looking directly at the anti-solar point?
                float anti_solar_alignment = max(0.0, dot(N, -L));
                // Darkest part of the night side sees the shadowed portion of the ring
                shadow_occlusion = 1.0 - (anti_solar_alignment * 0.95); // allow a tiny bit of bleed
            }
            
            float shine_intensity = 0.0;
            if (same_hemisphere > 0.0) {
                // Reflected ringshine (same side as sun)
                shine_intensity = abs(sun_elevation) * form_factor * solid_angle * opacity * 2.5;
            } else {
                // Transmitted ringshine (bleeds through to the other side)
                // Boosted to make the bleeding effect more prominent on the night side
                shine_intensity = abs(sun_elevation) * form_factor * solid_angle * opacity * 1.2;
            }
            
            // Apply shadow occlusion
            shine_intensity *= shadow_occlusion;
            
            vec3 ring_tint = u_ring_colors[k] * 1.2;
            ring_shine += ring_tint * shine_intensity;
        }
        
        vec3 final_color = f_color * diffuse_color + f_color * bounce_light + f_color * ring_shine;
        
        out_color = vec4(final_color, 1.0);
    }
    // Logarithmic depth — C constant improves near-field precision
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
}
"""

orbit_vertex_shader = """
#version 460 core
// Orbital element data stored in SSBO — doubles for precision
struct OrbitData {
    dvec4 d0;  // e_hat.xyz, sl_p
    dvec4 d1;  // q_hat.xyz, e_mag
    dvec4 d2;  // bary_rel.xyz, cos_theta0
    dvec4 d3;  // color.rgb, sin_theta0
    dvec4 d4;  // empty, theta_max, is_patched, padding
};

layout(std430, binding = 0) buffer OrbitBuffer {
    OrbitData orbits[];
};

uniform mat4 projection;
uniform mat4 view_rot;
uniform dvec4 u_cam_pos_double;
uniform float u_fade_dir;
uniform float u_min_alpha;
uniform int u_orbit_res;
uniform int u_base_instance;

out vec4 f_color;
out float f_clip_z;

void main() {
    OrbitData od = orbits[gl_InstanceID + u_base_instance];
    
    dvec3 e_hat = od.d0.xyz;
    double sl_p = od.d0.w;
    dvec3 q_hat = od.d1.xyz;
    double e_mag = od.d1.w;
    dvec3 bary_rel = od.d2.xyz;
    double cos_A = od.d2.w;
    vec3 color = vec3(od.d3.xyz);
    double sin_A = od.d3.w;
    
    float delta_angle = float(od.d4.y);
    float is_patched = float(od.d4.z);
    
    float t = float(gl_VertexID) / float(u_orbit_res - 1);
    float angle_B;
    if (delta_angle > 9.0) {
        angle_B = t * 2.0 * 3.14159265358979;
    } else {
        angle_B = t * delta_angle;
    }
    
    dvec3 pos;
    
    if (e_mag < 0.999) {
        // Elliptical orbit: use 64-bit pre-computed E0 to avoid 32-bit jitter
        double cos_E0 = cos_A;
        double sin_E0 = sin_A;
        
        double cos_B;
        double sin_B;
        
        if (gl_VertexID == u_orbit_res - 1 && delta_angle > 9.0) {
            // GLSL 4.30 lacks 64-bit transcendental functions (cos/sin). 
            // The 32-bit float evaluation of cos(2.0 * PI) introduces a tiny gap.
            // We force the last vertex to perfectly close the loop!
            cos_B = 1.0lf;
            sin_B = 0.0lf;
        } else {
            cos_B = double(cos(angle_B));
            sin_B = double(sin(angle_B));
        }
        
        // E = E0 + angle_B
        double cos_E = cos_E0 * cos_B - sin_E0 * sin_B;
        double sin_E = sin_E0 * cos_B + cos_E0 * sin_B;
        
        double a = sl_p / (1.0 - e_mag * e_mag);
        double x = a * (cos_E - e_mag);
        double y = a * sqrt(1.0 - e_mag * e_mag) * sin_E;
        
        pos = bary_rel + x * e_hat + y * q_hat;
    } else {
        // Hyperbolic/Parabolic fallback using True Anomaly
        double cos_B = double(cos(angle_B));
        double sin_B = double(sin(angle_B));
        
        double cos_a = cos_A * cos_B - sin_A * sin_B;
        double sin_a = sin_A * cos_B + cos_A * sin_B;
        
        double denom = 1.0lf + e_mag * cos_a;
        double r = sl_p / max(denom, 1e-5lf);
        
        pos = bary_rel + (cos_a * e_hat + sin_a * q_hat) * r;
    }
    
    dvec3 eye_pos = pos - u_cam_pos_double.xyz;
    
    float t_val = float(gl_VertexID) / float(u_orbit_res - 1);
    float t_curr = float(od.d4.x);
    
    float alpha = 1.0;
    if (delta_angle > 9.0) {
        if (u_fade_dir > 0.0) alpha = mix(u_min_alpha, 1.0, t_val);
        else alpha = mix(1.0, u_min_alpha, t_val);
    } else {
        if (u_fade_dir > 0.0) {
            if (t_val > t_curr) alpha = u_min_alpha;
            else {
                float fraction = t_val / max(t_curr, 0.001);
                alpha = mix(u_min_alpha, 1.0, fraction);
            }
        } else {
            if (t_val < t_curr) alpha = u_min_alpha;
            else {
                float fraction = (t_val - t_curr) / max(1.0 - t_curr, 0.001);
                alpha = mix(1.0, u_min_alpha, fraction);
            }
        }
    }
    
    // Dashed lines for patched conics
    if (is_patched > 0.5) {
        float dash = fract(float(gl_VertexID) / 10.0);
        if (dash > 0.5) alpha = 0.0;
        else alpha = mix(1.0, u_min_alpha, t_val);
    }
    
    if (e_mag >= 0.999) {
        float dist_from_bary = float(length(pos - bary_rel));
        float q = float(sl_p / (1.0lf + e_mag));
        float r_max_draw = max(300.0, q * 10.0);
        float fade_start = max(150.0, r_max_draw * 0.5);
        float fade_out = 1.0 - smoothstep(fade_start, r_max_draw, dist_from_bary);
        alpha *= fade_out;
    }
    
    f_color = vec4(color * 0.4, alpha);
    gl_Position = projection * view_rot * vec4(float(eye_pos.x), float(eye_pos.y), float(eye_pos.z), 1.0);
    f_clip_z = gl_Position.w;
}
"""

orbit_fragment_shader = """
#version 460 core
in vec4 f_color;
in float f_clip_z;
uniform float u_far;
uniform float u_depth_C;
out vec4 out_color;
void main() {
    if (f_color.a < 0.01) discard; // discard dashes
    out_color = f_color;
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
}
"""

ephem_orbit_vertex_shader = """
#version 460 core
in vec3 in_pos;
uniform mat4 projection;
uniform mat4 view_rot;
uniform vec4 u_cam_pos_double;
uniform vec3 u_bary_pos;
uniform vec3 u_color;
out vec4 f_color;
out float f_clip_z;
void main() {
    vec3 pos = in_pos + u_bary_pos;
    vec3 eye_pos = pos - u_cam_pos_double.xyz;
    f_color = vec4(u_color * 0.4, 1.0);
    gl_Position = projection * view_rot * vec4(eye_pos, 1.0);
    f_clip_z = gl_Position.w;
}
"""

ephem_orbit_fragment_shader = """
#version 460 core
in vec4 f_color;
in float f_clip_z;
uniform float u_far;
uniform float u_depth_C;
out vec4 out_color;
void main() {
    out_color = f_color;
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
}
"""

ring_vertex_shader = """
#version 460 core
in vec3 in_position;
in vec3 in_normal;
in vec4 in_color;
in float in_scatter;
in float in_asymmetry;

#define MAX_CASTERS 64
layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    vec4 u_star_pos_radius;
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad0;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};
uniform vec3 u_body_offset;

out vec3 f_world_pos;
out vec3 f_normal;
out vec4 f_color;
out float f_clip_z;
out float f_scatter;
out float f_asymmetry;

void main() {
    vec3 world_pos = in_position + u_body_offset;
    f_world_pos = world_pos;
    f_normal = in_normal;
    f_color = in_color;
    f_scatter = in_scatter;
    f_asymmetry = in_asymmetry;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    f_clip_z = gl_Position.w;
}
"""

ring_fragment_shader = """
#version 460 core
#define MAX_CASTERS 64

in vec3 f_world_pos;
in vec3 f_normal;
in vec4 f_color;
in float f_clip_z;
in float f_scatter;
in float f_asymmetry;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    vec4 u_star_pos_radius;
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad0;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};

// Per-ring-body uniforms
uniform vec3 u_host_planet_pos;
uniform float u_host_planet_radius;
uniform vec3 u_camera_pos;
uniform vec4 u_host_planet_pole_obl;
uniform int u_clip_mode;
uniform uint u_caster_mask_lo;
uniform uint u_caster_mask_hi;

out vec4 out_color;

float get_oblate_radius(float r_eq, float oblateness, vec3 pole, vec3 L, vec3 perp_vec) {
    if (oblateness <= 0.0) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float f_factor = 1.0 - oblateness;
    float R_minor = r_eq * sqrt(PdotL * PdotL + f_factor * f_factor * P_proj_len * P_proj_len);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / R_minor) * (y / R_minor));
    return perp_len / denom;
}

void main() {
    if (u_clip_mode != 0) {
        vec3 to_cam = u_camera_pos - u_host_planet_pos;
        vec3 to_frag = f_world_pos - u_host_planet_pos;
        float d = dot(to_frag, to_cam);
        if (u_clip_mode == 1 && d > 0.0) discard;
        if (u_clip_mode == 2 && d <= 0.0) discard;
    }
    
    vec3 frag_to_star = u_star_pos_radius.xyz - f_world_pos;
    float dist_to_star = length(frag_to_star);
    vec3 L = frag_to_star / dist_to_star;
    
    vec3 V = normalize(u_camera_pos - f_world_pos);
    
    // Scattering angle: cosine of angle between propagating light direction (-L) and view direction (V)
    float cos_theta = -dot(L, V);
    
    // Henyey-Greenstein phase function for backscattering (rocky/main rings) and forward-scattering (dusty/E-ring)
    float g_rock = -0.3;
    float denom_rock = 1.0 + g_rock * g_rock - 2.0 * g_rock * cos_theta;
    float rocky_phase = (1.0 - g_rock * g_rock) / (denom_rock * sqrt(denom_rock));
    rocky_phase = min(rocky_phase, 2.5); // prevent excessive brightness
    
    float g_dust = f_asymmetry; // Controlled by JSON (e.g. 0.7)
    float denom_dust = 1.0 + g_dust * g_dust - 2.0 * g_dust * cos_theta;
    float dusty_phase = (1.0 - g_dust * g_dust) / (denom_dust * sqrt(denom_dust));
    dusty_phase *= 0.15; // Peak is now ~1.2, preventing extreme blowout
    
    // Determine if camera and sun are on the same side of the ring plane using the ring normal N
    vec3 N = normalize(f_normal);
    float sun_side = dot(N, L);
    float cam_side = dot(N, V);
    
    // Smooth transition over a tiny angle to avoid aliasing / harsh floating point flips
    float same_side = smoothstep(-0.02, 0.02, sun_side * cam_side);
    float is_backlit = 1.0 - same_side;
    
    // Solar elevation attenuation (main rings get dimmer as sun gets edge-on, but not pitch black)
    float solar_elevation = mix(0.15, 1.0, abs(sun_side));
    
    // --- Multiple Scattering and B-Ring Inversion ---
    float rock_albedo = 1.0 - f_scatter;
    
    // Reflected pathway: Brightens with opacity. Adds multiple scattering boost.
    float rock_reflect = rocky_phase * rock_albedo * (f_color.a + f_color.a * f_color.a * 0.5);
    
    // Transmitted pathway (B-ring inversion):
    // Light must scatter through the particles. Peaks at medium opacity, dark at high opacity.
    float rock_transmit = rocky_phase * rock_albedo * f_color.a * pow(1.0 - f_color.a, 1.5) * 4.0;
    
    // Dusty rings (E-ring) behave differently. They scatter efficiently in transmission and reflection.
    // We MUST multiply by f_color.a to preserve the radial density gradient (fade out at edges).
    // Since dust rings usually have a very low base f_color.a, we boost the scattering significantly.
    // Option 1: Reduced multiplier from 10.0 to 2.5 to keep the peak tighter
    float dust_reflect = dusty_phase * f_scatter * f_color.a * 5;
    float dust_transmit = dusty_phase * f_scatter * f_color.a * 5; 
    
    // Apply solar elevation ONLY to the rock (main) rings. Dust rings scatter isotropically 
    // through their volume, so edge-on illumination doesn't dim them (it can actually brighten them).
    rock_reflect *= solar_elevation;
    rock_transmit *= solar_elevation;
    
    float reflected = rock_reflect + dust_reflect;
    float transmitted = rock_transmit + dust_transmit;
    
    // Combine based on which side we are viewing from
    float direct_illum = mix(transmitted, reflected, same_side);
    
    // Eclipse shadows from other bodies
    vec3 shadow = vec3(1.0);
    for (int j = 0; j < u_num_casters; j++) {
        if (j < 32) { if ((u_caster_mask_lo & (1u << j)) == 0u) continue; }
        else        { if ((u_caster_mask_hi & (1u << (j - 32))) == 0u) continue; }
        vec3 caster_pos = u_casters[j].xyz;
        float caster_r = u_casters[j].w;
        vec3 frag_to_caster = caster_pos - f_world_pos;
        float dist_to_caster = length(frag_to_caster);
        
        // Rings don't self-shadow from u_casters (casters are spherical bodies).
        float t = dot(frag_to_caster, L);
        if (t <= 0.0 || t >= dist_to_star) continue;
        vec3 perp_vec = frag_to_caster - t * L;
        float perp = length(perp_vec);
        
        float oblateness = u_caster_poles_obl[j].w;
        if (oblateness > 0.0) {
            vec3 pole = u_caster_poles_obl[j].xyz;
            caster_r = get_oblate_radius(caster_r, oblateness, pole, L, perp_vec);
        }
        
        float r_star_proj = u_star_pos_radius.w * t / dist_to_star;
        float r_penumbra = caster_r + r_star_proj;
        float r_umbra = abs(caster_r - r_star_proj);
        
        float max_occlusion = (caster_r >= r_star_proj) ? 1.0 : (caster_r * caster_r) / (r_star_proj * r_star_proj);
        float occlusion = max_occlusion * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
        
        vec3 caster_shadow = vec3(1.0 - occlusion);
        float atmo_h = u_caster_atmos[j].w;
        if (atmo_h > 0.0 && occlusion > 0.0) {
            float depth_into_umbra = clamp(1.0 - (perp / caster_r), 0.0, 1.0);
            vec3 atmo_tint = u_caster_atmos[j].xyz;
            // Much gentler tint deepening
            vec3 deep_tint = pow(atmo_tint, vec3(1.0 + depth_into_umbra * 0.5));
            // Gentler exponential falloff so light reaches deep into the umbra without turning black
            float refraction_intensity = exp(-depth_into_umbra * 1.2) * 1.5;
            caster_shadow += deep_tint * refraction_intensity * occlusion;
        }
        shadow *= caster_shadow;
    }
    
    // === Host planet shadow on its own ring ===
    if (u_host_planet_radius > 0.0) {
        vec3 frag_to_host = u_host_planet_pos - f_world_pos;
        float dist_to_host = length(frag_to_host);
        
        // Project host onto fragment-to-star ray
        float t = dot(frag_to_host, L);
        
        // Host must be between ring fragment and star
        if (t > 0.0 && t < dist_to_star) {
            vec3 perp_vec = frag_to_host - t * L;
            float perp = length(perp_vec);
            
            float host_r = u_host_planet_radius;
            float host_oblateness = u_host_planet_pole_obl.w;
            if (host_oblateness > 0.0) {
                vec3 host_pole = u_host_planet_pole_obl.xyz;
                host_r = get_oblate_radius(host_r, host_oblateness, host_pole, L, perp_vec);
            }
            
            float r_star_proj = u_star_pos_radius.w * t / dist_to_star;
            float r_penumbra = host_r + r_star_proj;
            float r_umbra = abs(host_r - r_star_proj);
            
            float max_occlusion = (host_r >= r_star_proj) ? 1.0 : (host_r * host_r) / (r_star_proj * r_star_proj);
            float occlusion = max_occlusion * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
            
            vec3 caster_shadow = vec3(1.0 - occlusion);
            // Host planet atmosphere on ring shadow
            float atmo_h = u_caster_atmos[0].w; // Assuming host is near the start, but actually we don't have j for host.
            // We'll just leave it uncolored for ring shadows or use generic. 
            // Actually, let's keep it simple for host shadows on rings and use black for now.
            shadow *= caster_shadow;
        }
    }
    
    vec3 direct_illum_color = vec3(direct_illum) * shadow; // solar_elevation was applied specifically to rock earlier
    
    // === Planetshine on Rings ===
    float planetshine = 0.0;
    if (u_host_planet_radius > 0.0) {
        vec3 frag_to_host = u_host_planet_pos - f_world_pos;
        float dist_host = length(frag_to_host);
        vec3 dir_to_host = frag_to_host / dist_host;
        vec3 host_to_star = normalize(u_star_pos_radius.xyz - u_host_planet_pos);
        
        // Planet phase and solid angle
        float planet_phase = max(0.0, dot(host_to_star, -dir_to_host));
        float R_sq = u_host_planet_radius * u_host_planet_radius;
        float solid_angle = R_sq / max(dist_host * dist_host, R_sq);
        float planet_elevation = mix(0.5, 1.0, abs(dot(N, dir_to_host)));
        
        float planet_side = dot(N, dir_to_host);
        float cam_planet_same_side = smoothstep(-0.02, 0.02, planet_side * cam_side);
        
        float shine_intensity = planet_phase * solid_angle * planet_elevation;
        
        // For planetshine, reflection and transmission are boosted so the dark side is visible
        float p_reflect = rock_reflect + 0.3 * f_color.a; 
        float p_transmit = rock_transmit + 0.2 * f_color.a;
        float shine_response = mix(p_transmit, p_reflect, cam_planet_same_side);
        
        planetshine = shine_intensity * shine_response * 0.8;
    }
    
    // No artificial ambient glow - shadow and unlit sides should be physically black 
    // unless illuminated by planetshine (calculated above).
    vec3 illumination = direct_illum_color + vec3(planetshine);
    
    // Boost alpha when backlit - dust_transmit already factors in f_color.a to preserve gradients.
    float scattered_alpha_boost = dust_transmit * shadow.r * is_backlit * 1.0;
    // With premultiplied alpha, the alpha channel strictly represents OCCLUSION of the background.
    // Dust forward-scattering (boost) does NOT increase occlusion, it only increases brightness!
    // So we strictly use the base opacity (f_color.a) as the occlusion value, allowing dust rings to glow 
    // brighter than their opacity without physically blocking the stars behind them.
    float final_alpha = clamp(f_color.a, 0.0, 1.0);
    
    // Use the actual ring color for backlit scattering instead of a hardcoded blue tint
    vec3 tinted_color = f_color.rgb;
    
    // Options 2 & 3: Soft curve and allowing color clipping (blowing out to white)
    // Instead of a hard clamp that creates a massive flat plateau, we let the raw color 
    // scale with illumination, then apply a soft exponential curve. This preserves 
    // the sharp center of the highlight as it naturally rolls off into pure white.
    vec3 raw_color = tinted_color * illumination;
    vec3 final_color = 1.0 - exp(-raw_color);
    
    out_color = vec4(final_color, final_alpha);
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
}
"""

atmo_vertex_shader = """
#version 460 core
in vec3 in_position;

#define MAX_CASTERS 64
layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    vec4 u_star_pos_radius;
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad0;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};

uniform vec3 u_body_offset;
uniform float u_atmo_radius_au;

out vec3 f_world_pos;
out vec3 f_local_pos;
out float f_clip_z;

void main() {
    vec3 world_pos = in_position * u_atmo_radius_au + u_body_offset;
    f_world_pos = world_pos;
    f_local_pos = in_position;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    f_clip_z = gl_Position.w;
}
"""

atmo_fragment_shader = """
#version 460 core
#define MAX_CASTERS 64
#define MAX_RING_PLANES 16
#define PI 3.14159265358979

in vec3 f_world_pos;
in vec3 f_local_pos;
in float f_clip_z;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    vec4 u_star_pos_radius;
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad0;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};

uniform vec3  u_body_offset;
uniform float u_atmo_radius_au;
uniform float u_planet_radius_km;
uniform float u_atmo_radius_km;
uniform float u_au_to_km;
uniform vec3  u_beta_rayleigh;
uniform float u_h_rayleigh;
uniform float u_beta_mie;
uniform float u_h_mie;
uniform float u_mie_g;
uniform vec3  u_beta_absorption;
uniform float u_sun_intensity;
uniform vec3  u_camera_pos;
uniform int   u_num_samples;
uniform int   u_num_light_samples;
uniform vec4  u_pole_obl;
uniform uint  u_caster_mask_lo;
uniform uint  u_caster_mask_hi;
uniform uint  u_ring_mask;

// Ring shadow planes (sphere-only)
uniform sampler2D u_ring_gradients;
uniform int u_num_ring_planes;
uniform int u_atmo_quality;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec3 u_ring_params[MAX_RING_PLANES]; // x=inner_r, y=outer_r, z=opacity

out vec4 out_color;

int g_num_local_casters;
vec4 g_local_casters[64];

int g_num_local_rings;
int g_local_rings[16];

vec3 toSphericalSpace(vec3 p, vec4 pole_obl) {
    float f = pole_obl.w;
    if (f == 0.0) return p;
    vec3 pole = pole_obl.xyz;
    float h = dot(p, pole);
    vec3 p_perp = p - h * pole;
    return p_perp + (h / (1.0 - f)) * pole;
}

float compute_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, vec3 planet_center_render) {
    float shadow = 1.0;
    for (int i = 0; i < g_num_local_casters; i++) {
        vec3 caster_pos = g_local_casters[i].xyz;
        float caster_r = g_local_casters[i].w;
        vec3 s_to_c = caster_pos - eval_render_pos;
        float proj = dot(s_to_c, L_dir);
        if (proj <= 0.0 || proj >= dist_to_star) continue;
        vec3 perp_vec = s_to_c - proj * L_dir;
        float perp = length(perp_vec);
        float r_star_proj = u_star_pos_radius.w * proj / dist_to_star;
        float r_penumbra = caster_r + r_star_proj;
        float r_umbra = abs(caster_r - r_star_proj);
        float max_occ = (caster_r >= r_star_proj) ? 1.0 : (caster_r * caster_r) / (r_star_proj * r_star_proj);
        float occ = max_occ * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
        shadow *= (1.0 - occ);
    }
    for (int i = 0; i < g_num_local_rings; i++) {
        int k = g_local_rings[i];
        vec3 ring_center = u_ring_center[k];
        vec3 ring_normal = u_ring_normal[k];
        float inner_r = u_ring_params[k].x;
        float outer_r = u_ring_params[k].y;
        float opacity = u_ring_params[k].z;
        float denom = dot(L_dir, ring_normal);
        if (abs(denom) < 1e-8) continue;
        float t_ring = dot(ring_center - eval_render_pos, ring_normal) / denom;
        if (t_ring <= 0.0 || t_ring >= dist_to_star) continue;
        vec3 hit = eval_render_pos + t_ring * L_dir;
        float dist_from_center = length(hit - ring_center);
        if (dist_from_center >= inner_r && dist_from_center <= outer_r) {
            float edge_width = (outer_r - inner_r) * 0.05;
            float inner_fade = smoothstep(inner_r, inner_r + edge_width, dist_from_center);
            float outer_fade = smoothstep(outer_r, outer_r - edge_width, dist_from_center);
            float p = (dist_from_center - inner_r) / max(1e-6, outer_r - inner_r);
            float alpha_mult = texture(u_ring_gradients, vec2(p, (float(k) + 0.5) / 16.0)).r;
            float ring_shadow = inner_fade * outer_fade * opacity * alpha_mult;
            shadow *= (1.0 - ring_shadow);
        }
    }
    return shadow;
}

vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float a = dot(dir, dir);
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant < 0.0) return vec2(1e10, -1e10);
    float d = sqrt(discriminant);
    return vec2((-b - d) / a, (-b + d) / a);
}

void main() {
    g_num_local_casters = 0;
    for (int k = 0; k < u_num_casters; k++) {
        if (k < 32) { if ((u_caster_mask_lo & (1u << k)) == 0u) continue; }
        else        { if ((u_caster_mask_hi & (1u << (k - 32))) == 0u) continue; }
        if (length(u_casters[k].xyz - u_body_offset) < 1e-6) continue;
        if (g_num_local_casters < 64) {
            g_local_casters[g_num_local_casters++] = u_casters[k];
        }
    }
    
    g_num_local_rings = 0;
    for (int k = 0; k < u_num_ring_planes; k++) {
        if ((u_ring_mask & (1u << k)) == 0u) continue;
        if (g_num_local_rings < 16) {
            g_local_rings[g_num_local_rings++] = k;
        }
    }

    // 1. Setup: convert to planet-local km coordinates
    vec3 planet_center_render = u_body_offset;
    vec3 cam_local_au = u_camera_pos - planet_center_render;
    
    // Fix catastrophic precision loss by computing fragment pos purely from local data
    vec3 frag_local_au = f_local_pos * u_atmo_radius_au;

    vec3 cam_local = cam_local_au * u_au_to_km;
    vec3 frag_local = frag_local_au * u_au_to_km;

    vec3 ray_dir = normalize(frag_local - cam_local);

    vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
    vec3 frag_local_sph = toSphericalSpace(frag_local, u_pole_obl);

    // 2. Ray-sphere intersections (planet-local km)
    // Shifting intersection origin to the fragment itself to preserve float32 precision
    // when the camera is extremely far away. 's' is the parameter along ray_dir starting from frag_local.
    vec2 s_atmo = raySphereIntersect(frag_local_sph, ray_dir_sph, u_atmo_radius_km);
    if (s_atmo.x > s_atmo.y) discard;

    vec2 s_planet = raySphereIntersect(frag_local_sph, ray_dir_sph, u_planet_radius_km);

    float dist_to_frag = length(frag_local - cam_local);
    float s_cam = -dist_to_frag; // Camera position relative to frag_local

    float s_start = max(s_atmo.x, s_cam);
    float s_end = s_atmo.y;
    
    if (s_planet.x > s_cam && s_planet.x < s_end) {
        s_end = s_planet.x;
    }

    // Clip against all other opaque bodies (moons, planets) to fix depth sorting
    // since atmosphere is rendered with DEPTH_TEST disabled.
    for (int k = 0; k < u_num_casters; k++) {
        vec3 caster_pos = u_casters[k].xyz;
        float caster_r = u_casters[k].w * u_au_to_km;
        vec3 caster_local = (caster_pos - planet_center_render) * u_au_to_km;
        
        // Skip the host planet itself (already clipped precisely above)
        if (length(caster_local) < 1.0) continue;
        
        vec4 pole_obl = u_caster_poles_obl[k];
        vec3 origin_c = toSphericalSpace(frag_local - caster_local, pole_obl);
        vec3 dir_c = toSphericalSpace(ray_dir, pole_obl);
        
        vec2 s_c = raySphereIntersect(origin_c, dir_c, caster_r);
        if (s_c.x > s_cam && s_c.x < s_end) {
            s_end = s_c.x;
        }
    }

    if (s_start >= s_end) discard;

    // 3. Sun direction (in km-space)
    vec3 sun_pos_local = (u_star_pos_radius.xyz - planet_center_render) * u_au_to_km;
    vec3 sun_dir = normalize(sun_pos_local);
    vec3 sun_dir_sph = toSphericalSpace(sun_dir, u_pole_obl);

    // Scattering coefficients: convert m^-1 to km^-1
    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M = vec3(u_beta_mie) * 1000.0;
    vec3 beta_A = u_beta_absorption * 1000.0;
    float mie_ext_factor = 1.0 / 0.9;

    // 3.5 Global Eclipse & Ring Shadow
    float global_eclipse_shadow = 1.0;
    if (u_atmo_quality == 1) { // Low Quality (2D shadow at midpoint)
        float s_mid = (s_start + s_end) * 0.5;
        vec3 mid_pos = frag_local + s_mid * ray_dir;
        vec3 mid_render = mid_pos / u_au_to_km + planet_center_render;
        vec3 mid_to_star = u_star_pos_radius.xyz - mid_render;
        float dist_mid_star = length(mid_to_star);
        vec3 L_mid = mid_to_star / dist_mid_star;
        global_eclipse_shadow = compute_shadow(mid_render, L_mid, dist_mid_star, planet_center_render);
    }
    // 4. Primary ray march
    float step_size = (s_end - s_start) / float(u_num_samples);
    vec3 total_rayleigh = vec3(0.0);
    vec3 total_mie = vec3(0.0);
    float od_rayleigh = 0.0;
    float od_mie = 0.0;

    for (int i = 0; i < u_num_samples; i++) {
        float s = s_start + (float(i) + 0.5) * step_size;
        vec3 sample_pos = frag_local + s * ray_dir;
        vec3 sample_pos_sph = frag_local_sph + s * ray_dir_sph;
        float altitude = length(sample_pos_sph) - u_planet_radius_km;

        float rho_R = exp(-altitude / u_h_rayleigh);
        float rho_M = exp(-altitude / u_h_mie);

        od_rayleigh += rho_R * step_size;
        od_mie += rho_M * step_size;

        // 5. Light ray: optical depth from sample to sun
        vec2 t_light_atmo = raySphereIntersect(sample_pos_sph, sun_dir_sph, u_atmo_radius_km);
        float light_ray_len = t_light_atmo.y;
        float light_step = light_ray_len / float(u_num_light_samples);

        float od_light_R = 0.0;
        float od_light_M = 0.0;
        bool in_planet_shadow = false;

        for (int j = 0; j < u_num_light_samples; j++) {
            float t_l = (float(j) + 0.5) * light_step;
vec3 light_pos_sph = sample_pos_sph + t_l * sun_dir_sph;
            float light_alt = length(light_pos_sph) - u_planet_radius_km;

            if (light_alt < 0.0) {
                in_planet_shadow = true;
                break;
            }

            od_light_R += exp(-light_alt / u_h_rayleigh) * light_step;
            od_light_M += exp(-light_alt / u_h_mie) * light_step;
        }

        if (in_planet_shadow) continue;

        // 7. Accumulate in-scattered light
        vec3 tau = beta_R * (od_rayleigh + od_light_R)
                 + beta_M * mie_ext_factor * (od_mie + od_light_M)
                 + beta_A * (od_rayleigh + od_light_R);
        float sample_shadow = global_eclipse_shadow;
        if (u_atmo_quality == 2) { // High Quality (Volumetric)
            vec3 sample_render = sample_pos / u_au_to_km + planet_center_render;
            vec3 sample_to_star = u_star_pos_radius.xyz - sample_render;
            float dist_sample_star = length(sample_to_star);
            vec3 L_sample = sample_to_star / dist_sample_star;
            sample_shadow = compute_shadow(sample_render, L_sample, dist_sample_star, planet_center_render);
        }
        // Approximate multiple scattering with a low-extinction term
        vec3 direct_attenuation = exp(-tau);
        vec3 ms_attenuation = max(vec3(0.0), (exp(-tau * 0.2) - direct_attenuation) * 0.4);
        vec3 attenuation = (direct_attenuation + ms_attenuation) * sample_shadow;

        total_rayleigh += rho_R * attenuation * step_size;
        total_mie      += rho_M * attenuation * step_size;
    }

    // 8. Phase functions
    float cos_theta = dot(ray_dir, sun_dir);

    float phase_R = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);

    float g = u_mie_g;
    float g2 = g * g;
    float phase_M = (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta))
                  / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));

    // 9. Final scattered light
    vec3 scattered = u_sun_intensity * (
        phase_R * beta_R * total_rayleigh +
        phase_M * beta_M * total_mie
    );

    // 10. Primary ray transmittance
    vec3 transmittance = exp(-(beta_R * od_rayleigh
                             + beta_M * mie_ext_factor * od_mie
                             + beta_A * od_rayleigh));

    // Tone mapping to prevent blowout while preserving hue
    float max_scatter = max(scattered.r, max(scattered.g, scattered.b));
    if (max_scatter > 1e-6) {
        float mapped_max = 1.0 - exp(-max_scatter);
        scattered = scattered * (mapped_max / max_scatter);
    }

    // 11. Premultiplied alpha output
    float avg_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    out_color = vec4(scattered, 1.0 - avg_transmittance);

    // Logarithmic depth
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0))
                 / log2(u_depth_C * u_far + 1.0);
}
"""

