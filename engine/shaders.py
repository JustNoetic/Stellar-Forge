culling_compute_shader = """
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 2) buffer AllInstances {
    vec4 instances[]; 
};

layout(std430, binding = 3) buffer VisLo { uint vis_lo[]; };
layout(std430, binding = 4) buffer VisHi { uint vis_hi[]; };
layout(std430, binding = 5) buffer VisUltra { uint vis_ultra[]; };

struct DrawCmd {
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};

layout(std430, binding = 6) buffer DrawCmds {
    DrawCmd cmds[3]; // lo=0, hi=1, ultra=2
};

layout(std430, binding = 7) buffer FocusMask {
    uint focused_mask[];
};

uniform int u_num_bodies;
uniform int u_star_idx;
uniform int u_n_casters;
uniform int u_caster_indices[64];
uniform vec4 u_frustum_planes[6];

uniform int u_n_rings;
uniform vec3 u_ring_centers[16];
uniform vec3 u_ring_normals[16];
uniform float u_ring_outer_radii[16];

uniform int u_tracking_idx;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_num_bodies)) return;
    
    vec4 f0 = instances[idx * 6 + 0];
    vec4 f1 = instances[idx * 6 + 1];
    vec4 f2 = instances[idx * 6 + 2];
    
    vec3 pos = f0.xyz;
    float r = f1.z;
    
    // 1. Frustum Culling
    bool visible = true;
    for (int p = 0; p < 6; p++) {
        float dist = dot(pos, u_frustum_planes[p].xyz) + u_frustum_planes[p].w;
        if (dist < -r) {
            visible = false;
            break;
        }
    }
    
    // 2. Caster & Ring Masks
    uint mask_lo = 0;
    uint mask_hi = 0;
    uint r_mask = 0;
    
    vec3 star_pos = instances[u_star_idx * 6 + 0].xyz;
    float star_r = instances[u_star_idx * 6 + 1].z;
    
    if (idx != uint(u_star_idx)) {
        vec3 L = star_pos - pos;
        float dist_star = length(L);
        if (dist_star >= 1e-6) {
            vec3 L_dir = L / dist_star;
            
            for (int c = 0; c < u_n_casters; c++) {
                int j = u_caster_indices[c];
                if (j == int(idx) || j == u_star_idx) continue;
                
                vec3 p_j = instances[j * 6 + 0].xyz;
                float r_j = instances[j * 6 + 1].z;
                
                vec3 vec = p_j - pos;
                float t = dot(vec, L_dir);
                if (t > -(r + r_j) && t < dist_star + r_j) {
                    vec3 perp = vec - t * L_dir;
                    float d = length(perp);
                    float r_cone = r_j + star_r * max(0.0, t) / dist_star + r;
                    if (d < r_cone) {
                        if (c < 32) mask_lo |= (1u << c);
                        else mask_hi |= (1u << (c - 32));
                    }
                }
            }
            
            for (int k = 0; k < u_n_rings; k++) {
                vec3 C_k = u_ring_centers[k];
                vec3 N_k = u_ring_normals[k];
                float r_out = u_ring_outer_radii[k];
                
                float denom = dot(L_dir, N_k);
                if (abs(denom) > 1e-8) {
                    vec3 vec_c = C_k - pos;
                    float dist_centers = length(vec_c);
                    if (dist_centers < 1e-6) {
                        r_mask |= (1u << k);
                    } else {
                        float s = dot(vec_c, N_k) / denom;
                        if (s > -r && s < dist_star + r) {
                            vec3 hit = pos + s * L_dir;
                            float d = length(hit - C_k);
                            float r_cone = star_r * max(0.0, s) / dist_star + r;
                            if (d < r_out + r_cone) {
                                r_mask |= (1u << k);
                            }
                        }
                    }
                }
            }
        }
    }
    
    instances[idx * 6 + 3].y = uintBitsToFloat(mask_lo);
    instances[idx * 6 + 3].z = uintBitsToFloat(mask_hi);
    instances[idx * 6 + 3].w = uintBitsToFloat(r_mask);
    
    // 3. Lodge into visible buffers
    if (visible) {
        bool is_ultra = (int(idx) == u_tracking_idx);
        bool is_focused = (focused_mask[idx] > 0u) || (f2.x > 0.5);
        
        if (is_ultra) {
            uint slot = atomicAdd(cmds[2].instanceCount, 1u);
            vis_ultra[slot] = idx;
        } else if (is_focused) {
            uint slot = atomicAdd(cmds[1].instanceCount, 1u);
            vis_hi[slot] = idx;
        } else {
            uint slot = atomicAdd(cmds[0].instanceCount, 1u);
            vis_lo[slot] = idx;
        }
    }
}
"""

sphere_vertex_shader = """
#version 460 core
in vec3 in_position;
in vec3 in_normal;

layout(std430, binding = 2) buffer AllInstances {
    vec4 instances[];
};

layout(std430, binding = 3) buffer VisibleIndices {
    uint vis_indices[];
};

#define MAX_CASTERS 64
#define MAX_STARS 16
layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
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
flat out vec3 f_planetshine_dir;
flat out vec3 f_planetshine_color;
void main() {
    uint inst_idx = vis_indices[gl_InstanceID];
    vec4 f0 = instances[inst_idx * 6 + 0];
    vec4 f1 = instances[inst_idx * 6 + 1];
    vec4 f2 = instances[inst_idx * 6 + 2];
    vec4 f3 = instances[inst_idx * 6 + 3];
    vec4 f4 = instances[inst_idx * 6 + 4];
    vec4 f5 = instances[inst_idx * 6 + 5];
    
    vec3 in_offset = f0.xyz;
    vec3 in_color = vec3(f0.w, f1.x, f1.y);
    float in_radius = f1.z;
    float in_min_size = f1.w;
    float in_is_star = f2.x;
    vec3 in_pole = f2.yzw;
    float in_oblateness = f3.x;
    uvec2 in_caster_mask = uvec2(floatBitsToUint(f3.y), floatBitsToUint(f3.z));
    uint in_ring_mask = floatBitsToUint(f3.w);
    
    f_planetshine_dir = f4.xyz;
    f_planetshine_color = f5.xyz;
    
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
        float f_inv = in_oblateness / (1.0 - in_oblateness);
        adj_normal = normalize(in_normal + in_pole * (dot(in_normal, in_pole) * f_inv));
    }
    
    vec3 world_pos = (scaled_pos * final_radius) + in_offset;
    f_world_pos = world_pos;
    f_normal = adj_normal;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    gl_Position.z = (log2(max(1e-6, u_depth_C * gl_Position.w + 1.0)) / log2(u_depth_C * u_far + 1.0) * 2.0 - 1.0) * gl_Position.w;
    f_clip_z = gl_Position.w;
}
"""

sphere_fragment_shader = """
#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define MAX_RING_PLANES 16

in vec3 f_color;
in vec3 f_world_pos;
in vec3 f_normal;
in float f_is_star;
in float f_clip_z;
flat in uvec2 f_caster_mask;
flat in uint f_ring_mask;
flat in vec3 f_planetshine_dir;
flat in vec3 f_planetshine_color;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
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
uniform bool u_planetshine_enabled;
uniform bool u_ringshine_enabled;
uniform sampler2D u_eclipse_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform vec3 u_camera_pos;

out vec4 out_color;

float get_oblate_radius(float r_eq, float oblateness, vec3 pole, vec3 L, vec3 perp_vec) {
    if (oblateness <= 0.0 || r_eq < 1e-6) return r_eq;
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
    if (R_minor < 1e-6) return r_eq;
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / R_minor) * (y / R_minor));
    if (denom < 1e-6) return r_eq;
    return perp_len / denom;
}

void main() {
    if (f_is_star > 0.5) {
        // Star is self-luminous, apply quadratic limb darkening
        vec3 V = normalize(u_camera_pos - f_world_pos);
        vec3 N = normalize(f_normal);
        float mu = max(dot(N, V), 0.0);
        
        // Sun-like limb darkening coefficients
        float c1 = 0.4;
        float c2 = 0.2;
        float ld = 1.0 - c1 * (1.0 - mu) - c2 * (1.0 - mu) * (1.0 - mu);
        
        // Shift temperature slightly towards cooler orange/red near the limb
        vec3 edge_tint = vec3(1.0, 0.85, 0.65);
        vec3 color_shift = mix(edge_tint, vec3(1.0), pow(mu, 0.5));
        
        // Find which star this is to get its luminosity and radius
        float star_lum = 1.0;
        float star_r = 0.00465;
        for (int s = 0; s < u_num_stars; s++) {
            if (distance(f_world_pos, u_stars_pos_radius[s].xyz) < u_stars_pos_radius[s].w * 1.5) {
                star_lum = u_stars_colors[s].a;
                star_r = max(u_stars_pos_radius[s].w, 1e-6);
                break;
            }
        }
        
        float surface_luminance = u_hdr_enabled ? (star_lum / (star_r * star_r)) : 1.0;
        vec3 final_star_color = f_color * color_shift * ld * surface_luminance;
        if (u_hdr_enabled) {
            final_star_color *= u_exposure;
        }
        out_color = vec4(final_star_color, 1.0);
    } else {
        vec3 N = normalize(f_normal);
        vec3 total_diffuse_color = vec3(0.0);
        
        for (int s = 0; s < u_num_stars; s++) {
            vec3 star_pos = u_stars_pos_radius[s].xyz;
            float star_radius = u_stars_pos_radius[s].w;
            vec3 star_color = u_stars_colors[s].rgb;
            
            vec3 frag_to_star = star_pos - f_world_pos;
            float dist_to_star = length(frag_to_star);
            if (dist_to_star < 1e-5) continue;
            vec3 L = frag_to_star / dist_to_star;
            
            float star_lum = u_stars_colors[s].a;
            
            // Angular radius of the star for soft penumbra at terminator
            float star_ang_radius = star_radius / dist_to_star;
            float sin_alpha = clamp(star_ang_radius, 0.0, 1.0);
            float alpha = asin(sin_alpha);
            
            // Lambert cosine law with soft terminator
            float NdotL = dot(N, L);
            float diffuse = clamp((NdotL + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);
            
            // Inverse-square falloff
            if (u_hdr_enabled) {
                diffuse *= star_lum / (dist_to_star * dist_to_star);
            }
            
            // === Analytical eclipse shadows ===
            vec3 shadow = vec3(1.0);
            for (int j = 0; j < u_num_casters; j++) {
                // Early-exit: if shadow is already near-zero, remaining casters can't darken further
                if (dot(shadow, shadow) < 0.001) break;
                
                if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
                else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }
                
                vec3 caster_pos = u_casters[j].xyz;
                float caster_r = u_casters[j].w;
                
                vec3 frag_to_caster = caster_pos - f_world_pos;
                float dist_to_caster = length(frag_to_caster);
                
                // Skip self
                if (dist_to_caster < caster_r * 1.02) continue;
                
                vec3 V_c = frag_to_caster / dist_to_caster;
                float t_proj = dot(frag_to_caster, L);
                
                // Account for oblateness
                float oblateness = u_caster_poles_obl[j].w;
                if (oblateness > 0.0) {
                    vec3 pole = u_caster_poles_obl[j].xyz;
                    vec3 perp_vec = frag_to_caster - t_proj * L;
                    caster_r = get_oblate_radius(caster_r, oblateness, pole, L, perp_vec);
                }
                
                float sin_beta = clamp(caster_r / dist_to_caster, 0.0, 1.0);
                float beta = asin(sin_beta);
                
                float cos_gamma = dot(L, V_c);
                float gamma = acos(clamp(cos_gamma, -1.0, 1.0));
                float R_max = max(alpha, beta);
                float R_min = min(alpha, beta);
                float x_norm = gamma / max(1e-9, R_max);
                float y_norm = R_min / max(1e-9, R_max);
                float area_fraction = texture(u_eclipse_lut, vec2(x_norm * 0.5, y_norm)).r;
                float occlusion = clamp(area_fraction * (R_max * R_max) / max(1e-9, alpha * alpha), 0.0, 1.0);
                
                vec3 caster_shadow = vec3(1.0 - occlusion);
                
                float atmo_h = u_caster_atmos[j].w;
                if (atmo_h > 0.0 && occlusion > 0.0) {
                    float r_umbra = abs(alpha - beta);
                    float depth_into_umbra = clamp((r_umbra - gamma) / max(1e-6, r_umbra), 0.0, 1.0);
                    vec3 atmo_tint = u_caster_atmos[j].xyz;
                    vec3 deep_tint = pow(atmo_tint, vec3(1.0 + depth_into_umbra * 0.5));
                    float refraction_intensity = exp(-depth_into_umbra * 1.2) * 1.5;
                    float atmo_blend = smoothstep(0.5, 1.0, occlusion);
                    caster_shadow += deep_tint * refraction_intensity * atmo_blend;
                }
                
                shadow *= caster_shadow;
            }
            
            // === Ring shadow on planet surface ===
            // Only compute ring shadows if the fragment is lit by this star
            if (NdotL > -sin_alpha) {
                uint processed_mask = 0u;
                for (int k = 0; k < u_num_ring_planes; k++) {
                    if ((f_ring_mask & (1u << k)) == 0u) continue;
                    if ((processed_mask & (1u << k)) != 0u) continue;
                    
                    vec3 plane_center = u_ring_center[k];
                    vec3 plane_normal = u_ring_normal[k];
                    
                    uint coplanar_mask = 0u;
                    for (int j = k; j < u_num_ring_planes; j++) {
                        if ((f_ring_mask & (1u << j)) == 0u) continue;
                        if (distance(plane_center, u_ring_center[j]) < 1e-6 && 
                            dot(plane_normal, u_ring_normal[j]) > 0.999) {
                            coplanar_mask |= (1u << j);
                            processed_mask |= (1u << j);
                        }
                    }
                    
                    float denom = dot(L, plane_normal);
                    if (abs(denom) < 1e-8) continue;
                    
                    float t = dot(plane_center - f_world_pos, plane_normal) / denom;
                    if (t <= 0.0 || t >= dist_to_star) continue;
                    
                    vec3 hit = f_world_pos + t * L;
                    vec3 vec_radial = hit - plane_center;
                    float d = length(vec_radial);
                    
                    float r_star_proj = star_ang_radius * t;
                    vec3 L_plane = L - denom * plane_normal;
                    float L_plane_len = length(L_plane);
                    
                    float R_eff = r_star_proj;
                    if (L_plane_len > 1e-5 && d > 1e-5) {
                        vec3 L_proj = L_plane / L_plane_len;
                        vec3 T = normalize(cross(plane_normal, L));
                        vec3 dir_radial = vec_radial / d;
                        float cos_theta = dot(dir_radial, L_proj);
                        float sin_theta = dot(dir_radial, T);
                        R_eff = r_star_proj * sqrt( pow(cos_theta / max(1e-6, abs(denom)), 2.0) + pow(sin_theta, 2.0) );
                    }
                    
                    float plane_occlusion = 0.0;
                    
                    for (int j = k; j < u_num_ring_planes; j++) {
                        if ((coplanar_mask & (1u << j)) == 0u) continue;
                        float inner_r = u_ring_params[j].x;
                        float outer_r = u_ring_params[j].y;
                        
                        float overlap_min = max(inner_r, d - R_eff);
                        float overlap_max = min(outer_r, d + R_eff);
                        
                        if (overlap_min >= overlap_max) continue;
                        
                        float v_min = clamp((overlap_min - d) / max(1e-9, R_eff), -1.0, 1.0);
                        float v_max = clamp((overlap_max - d) / max(1e-9, R_eff), -1.0, 1.0);
                        
                        float f_max = (v_max * sqrt(max(0.0, 1.0 - v_max*v_max)) + asin(v_max)) / 3.14159265358979 + 0.5;
                        float f_min = (v_min * sqrt(max(0.0, 1.0 - v_min*v_min)) + asin(v_min)) / 3.14159265358979 + 0.5;
                        float fraction = max(0.0, f_max - f_min);
                        
                        float sum_alpha = 0.0;
                        int tex_samples = 5;
                        for(int s = 0; s < tex_samples; s++) {
                            float u = (float(s) + 0.5) / float(tex_samples);
                            float r = mix(overlap_min, overlap_max, u);
                            float p = (r - inner_r) / max(1e-6, outer_r - inner_r);
                            sum_alpha += texture(u_ring_gradients, vec2(p, (float(j) + 0.5) / 16.0)).r;
                        }
                        float alpha_mult = sum_alpha / float(tex_samples);
                        float opacity = u_ring_params[j].z;
                        
                        plane_occlusion += fraction * opacity * alpha_mult;
                    }
                    
                    shadow *= vec3(1.0 - min(plane_occlusion, 1.0));
                }
            }
            
            total_diffuse_color += star_color * diffuse * shadow;
        }
        
        // === Moonshine / Planetshine ===
        vec3 bounce_light = vec3(0.0);
        if (dot(f_planetshine_color, f_planetshine_color) > 1e-12) {
            float NdotC = max(0.0, dot(N, f_planetshine_dir));
            bounce_light = f_planetshine_color * NdotC;
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
            float dist_to_ring_sq = dot(frag_to_ring, frag_to_ring);
            if (dist_to_ring_sq < 1e-12) continue;
            
            float frag_elevation = dot(N, ring_normal);
            float abs_elev = abs(frag_elevation);
            float form_factor = abs_elev * (1.0 - abs_elev) * (1.0 - abs_elev) * 6.75; 
            float ring_area = (outer_r * outer_r - inner_r * inner_r);
            float solid_angle = ring_area / max(dist_to_ring_sq, ring_area) * 0.1;
            
            for (int s = 0; s < u_num_stars; s++) {
                vec3 star_pos = u_stars_pos_radius[s].xyz;
                vec3 star_color = u_stars_colors[s].rgb;
                vec3 L = normalize(star_pos - f_world_pos);
                
                float sun_elevation = dot(L, ring_normal);
                float same_hemisphere = sun_elevation * frag_elevation;
                
                float shadow_occlusion = 1.0;
                if (dot(N, L) < 0.0) {
                    float anti_solar_alignment = max(0.0, dot(N, -L));
                    shadow_occlusion = 1.0 - (anti_solar_alignment * 0.85);
                }
                
                float noon_fade = 1.0 - max(0.0, dot(N, L));
                
                float shine_intensity = 0.0;
                if (same_hemisphere > 0.0) {
                    shine_intensity = abs(sun_elevation) * form_factor * solid_angle * opacity * 3.5 * noon_fade;
                } else {
                    shine_intensity = abs(sun_elevation) * form_factor * solid_angle * opacity * 1.25;
                }
                shine_intensity *= shadow_occlusion;
                
                float star_lum = u_stars_colors[s].a;
                float dist_to_star = distance(star_pos, f_world_pos);
                float irradiance = u_hdr_enabled ? (star_lum / max(dist_to_star * dist_to_star, 1e-8)) : 1.0;
                
                vec3 ring_tint = u_ring_colors[k] * 1.2;
                ring_shine += ring_tint * star_color * shine_intensity * irradiance;
            }
        }
        
        vec3 final_color = f_color * total_diffuse_color;
        if (u_planetshine_enabled) {
            final_color += f_color * bounce_light;
        }
        if (u_ringshine_enabled) {
            final_color += f_color * ring_shine;
        }
        if (u_hdr_enabled) {
            final_color *= u_exposure;
        }
        out_color = vec4(final_color, 1.0);
    }
}
"""

orbit_compute_shader = """
#version 460 core
layout(local_size_x = 256) in;

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

layout(std430, binding = 1) buffer OrbitVertexBuffer {
    dvec4 vertices[]; // xyz = pos, w = alpha
};

uniform float u_fade_dir;
uniform float u_min_alpha;
uniform int u_orbit_res;
uniform int u_base_instance;
uniform int u_max_instances;
uniform int u_vertex_base_offset;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_max_instances * u_orbit_res)) return;
    
    uint instance_id = idx / uint(u_orbit_res);
    uint vertex_id = idx % uint(u_orbit_res);
    
    OrbitData od = orbits[instance_id + u_base_instance];
    
    dvec3 e_hat = od.d0.xyz;
    double sl_p = od.d0.w;
    dvec3 q_hat = od.d1.xyz;
    double e_mag = od.d1.w;
    dvec3 bary_rel = od.d2.xyz;
    double cos_A = od.d2.w;
    double sin_A = od.d3.w;
    
    float delta_angle = float(od.d4.y);
    float is_patched = float(od.d4.z);
    
    float t = float(vertex_id) / float(u_orbit_res - 1);
    float angle_B;
    if (delta_angle > 9.0) {
        angle_B = t * 2.0 * 3.14159265358979;
    } else {
        angle_B = t * delta_angle;
    }
    
    dvec3 pos;
    
    if (e_mag < 0.999) {
        double cos_E0 = cos_A;
        double sin_E0 = sin_A;
        
        double cos_B;
        double sin_B;
        
        if (vertex_id == uint(u_orbit_res - 1) && delta_angle > 9.0) {
            cos_B = 1.0lf;
            sin_B = 0.0lf;
        } else {
            cos_B = double(cos(angle_B));
            sin_B = double(sin(angle_B));
        }
        
        double cos_E = cos_E0 * cos_B - sin_E0 * sin_B;
        double sin_E = sin_E0 * cos_B + cos_E0 * sin_B;
        
        double a = sl_p / (1.0 - e_mag * e_mag);
        double x = a * (cos_E - e_mag);
        double y = a * sqrt(1.0 - e_mag * e_mag) * sin_E;
        
        pos = bary_rel + x * e_hat + y * q_hat;
    } else {
        double cos_B = double(cos(angle_B));
        double sin_B = double(sin(angle_B));
        
        double cos_a = cos_A * cos_B - sin_A * sin_B;
        double sin_a = sin_A * cos_B + cos_A * sin_B;
        
        double denom = 1.0lf + e_mag * cos_a;
        double r = sl_p / max(denom, 1e-5lf);
        
        pos = bary_rel + (cos_a * e_hat + sin_a * q_hat) * r;
    }
    
    float t_curr = float(od.d4.x);
    float alpha = 1.0;
    if (delta_angle > 9.0) {
        if (u_fade_dir > 0.0) alpha = mix(u_min_alpha, 1.0, t);
        else alpha = mix(1.0, u_min_alpha, t);
    } else {
        if (u_fade_dir > 0.0) {
            if (t > t_curr) alpha = u_min_alpha;
            else {
                float fraction = t / max(t_curr, 0.001);
                alpha = mix(u_min_alpha, 1.0, fraction);
            }
        } else {
            if (t < t_curr) alpha = u_min_alpha;
            else {
                float fraction = (t - t_curr) / max(1.0 - t_curr, 0.001);
                alpha = mix(1.0, u_min_alpha, fraction);
            }
        }
    }
    
    if (is_patched > 0.5) {
        float dash = fract(float(vertex_id) / 10.0);
        if (dash > 0.5) alpha = 0.0;
        else alpha = mix(1.0, u_min_alpha, t);
    }
    
    if (e_mag >= 0.999) {
        float dist_from_bary = float(length(pos - bary_rel));
        float q = float(sl_p / (1.0lf + e_mag));
        float r_max_draw = max(300.0, q * 10.0);
        float fade_start = max(150.0, r_max_draw * 0.5);
        float fade_out = 1.0 - smoothstep(fade_start, r_max_draw, dist_from_bary);
        alpha *= fade_out;
    }
    
    vertices[u_vertex_base_offset + idx] = dvec4(pos, double(alpha));
}
"""

orbit_vertex_shader = """
#version 460 core
struct OrbitData {
    dvec4 d0;
    dvec4 d1;
    dvec4 d2;
    dvec4 d3;
    dvec4 d4;
};
layout(std430, binding = 0) buffer OrbitBuffer {
    OrbitData orbits[];
};
layout(std430, binding = 1) buffer OrbitVertexBuffer {
    dvec4 vertices[];
};

uniform mat4 projection;
uniform mat4 view_rot;
uniform dvec4 u_cam_pos_double;
uniform int u_orbit_res;
uniform int u_base_instance;
uniform int u_vertex_base_offset;
uniform float u_depth_C;
uniform float u_far;

out vec4 f_color;
out float f_clip_z;

void main() {
    uint idx = u_vertex_base_offset + gl_InstanceID * u_orbit_res + gl_VertexID;
    dvec4 v = vertices[idx];
    
    OrbitData od = orbits[gl_InstanceID + u_base_instance];
    vec3 color = vec3(od.d3.xyz);
    
    dvec3 eye_pos = v.xyz - u_cam_pos_double.xyz;
    
    f_color = vec4(color * 0.4, float(v.w));
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
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    out_color = f_color;
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
uniform float u_depth_C;
uniform float u_far;
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
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    out_color = f_color;
}
"""

ring_vertex_shader = """
#version 460 core
in vec3 in_position;
in vec3 in_normal;

#define MAX_CASTERS 64
#define MAX_STARS 16
layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};
uniform vec3 u_body_offset;

out vec3 f_world_pos;
out vec3 f_normal;
out vec3 f_local_pos;
out float f_clip_z;

void main() {
    vec3 world_pos = in_position + u_body_offset;
    f_world_pos = world_pos;
    f_normal = in_normal;
    f_local_pos = in_position;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    gl_Position.z = (log2(max(1e-6, u_depth_C * gl_Position.w + 1.0)) / log2(u_depth_C * u_far + 1.0) * 2.0 - 1.0) * gl_Position.w;
    f_clip_z = gl_Position.w;
}
"""

ring_fragment_shader = """
#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16

in vec3 f_world_pos;
in vec3 f_normal;
in vec3 f_local_pos;
in float f_clip_z;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS];
    vec4 u_stars_colors[MAX_STARS];
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};

uniform vec3 u_host_planet_pos;
uniform float u_host_planet_radius;
uniform vec3 u_host_planet_color;
uniform vec3 u_camera_pos;
uniform vec4 u_host_planet_pole_obl;
uniform int u_clip_mode;
uniform uint u_caster_mask_lo;
uniform uint u_caster_mask_hi;
uniform bool u_planetshine_enabled;
uniform sampler2D u_eclipse_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;

struct RingPlane {
    vec3 color;
    float inner_r;
    float outer_r;
    float opacity;
    float scatter;
    float asymmetry;
    int row_idx;
    float _pad1;
    float _pad2;
    float _pad3;
};

#define MAX_RING_PLANES 16
uniform RingPlane u_ring_planes[MAX_RING_PLANES];
uniform int u_num_ring_planes;
uniform sampler2D u_ring_gradients;

out vec4 out_color;

float get_oblate_radius(float r_eq, float oblateness, vec3 pole, vec3 L, vec3 perp_vec) {
    if (oblateness <= 0.0 || r_eq < 1e-6) return r_eq;
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
    if (R_minor < 1e-6) return r_eq;
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / R_minor) * (y / R_minor));
    if (denom < 1e-6) return r_eq;
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
    
    float r = length(f_local_pos);
    
    vec3 total_color = vec3(0.0);
    float total_tau = 0.0;
    float total_scatter = 0.0;
    float total_asym = 0.0;
    
    for (int i=0; i<u_num_ring_planes; i++) {
        if (r >= u_ring_planes[i].inner_r && r <= u_ring_planes[i].outer_r) {
            float t = (r - u_ring_planes[i].inner_r) / max(1e-6, u_ring_planes[i].outer_r - u_ring_planes[i].inner_r);
            float alpha = texture(u_ring_gradients, vec2(t, (float(u_ring_planes[i].row_idx) + 0.5)/16.0)).r;
            float raw_a = alpha * u_ring_planes[i].opacity;
            float tau = 0.0;
            if (raw_a >= 0.999) {
                tau = 6.907 + (raw_a - 0.999) * 10.0;
            } else {
                tau = -log(1.0 - raw_a);
            }
            
            total_color += u_ring_planes[i].color * tau;
            total_scatter += u_ring_planes[i].scatter * tau;
            total_asym += u_ring_planes[i].asymmetry * tau;
            total_tau += tau;
        }
    }
    
    if (total_tau <= 1e-6) discard;
    
    float physical_alpha = 1.0 - exp(-total_tau);
    
    vec4 f_color = vec4(total_color / max(1e-6, total_tau), physical_alpha);
    float f_scatter = total_scatter / max(1e-6, total_tau);
    float f_asymmetry = total_asym / max(1e-6, total_tau);

    
    vec3 V = normalize(u_camera_pos - f_world_pos);
    vec3 N = normalize(f_normal);
    float cam_side = dot(N, V);
    
    float g_rock = -0.3;
    float g_dust = f_asymmetry;
    
    float rock_albedo = 1.0 - f_scatter;
    float rock_reflect = rock_albedo * (f_color.a + f_color.a * f_color.a * 0.5);
    float rock_transmit = rock_albedo * f_color.a * pow(1.0 - f_color.a, 1.5) * 4.0;
    
    float dust_reflect = f_scatter * f_color.a * 5.0;
    float dust_transmit = f_scatter * f_color.a * 5.0; 
    
    vec3 total_direct_illum_color = vec3(0.0);
    
    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        vec3 star_color = u_stars_colors[s].rgb;
        
        vec3 frag_to_star = star_pos - f_world_pos;
        float dist_to_star = length(frag_to_star);
        if (dist_to_star < 1e-5) continue;
        vec3 L = frag_to_star / dist_to_star;
        
        float star_lum = u_stars_colors[s].a;
        
        float cos_theta = -dot(L, V);
        
        float denom_rock = 1.0 + g_rock * g_rock - 2.0 * g_rock * cos_theta;
        float rocky_phase = (1.0 - g_rock * g_rock) / (denom_rock * sqrt(denom_rock));
        rocky_phase = min(rocky_phase, 2.5);
        
        float denom_dust = 1.0 + g_dust * g_dust - 2.0 * g_dust * cos_theta;
        float dusty_phase = (1.0 - g_dust * g_dust) / (denom_dust * sqrt(denom_dust));
        dusty_phase *= 0.15;
        
        float sun_side = dot(N, L);
        
        float star_ang_radius = star_radius / dist_to_star;
        float sin_alpha_local = clamp(star_ang_radius, 1e-6, 1.0);
        float alpha = asin(sin_alpha_local);
        
        float effective_sun_side = sqrt(sun_side * sun_side + 0.180126 * alpha * alpha);
        float solar_elevation = max(0.02, effective_sun_side);
        
        float v_star = clamp(sun_side / sin_alpha_local, -1.0, 1.0);
        float f_top = (v_star * sqrt(max(0.0, 1.0 - v_star*v_star)) + asin(v_star)) / 3.14159265358979 + 0.5;
        float same_side = (cam_side > 0.0) ? f_top : (1.0 - f_top);
        
        float rock_reflect_s = rock_reflect * solar_elevation * rocky_phase;
        float rock_transmit_s = rock_transmit * solar_elevation * rocky_phase;
        
        float reflected_s = rock_reflect_s + dust_reflect * dusty_phase;
        float transmitted_s = rock_transmit_s + dust_transmit * dusty_phase;
        
        float direct_illum_s = mix(transmitted_s, reflected_s, same_side);
        
        // Inverse-square falloff
        if (u_hdr_enabled) {
            direct_illum_s *= star_lum / (dist_to_star * dist_to_star);
        }
        vec3 shadow_s = vec3(1.0);
        for (int j = 0; j < u_num_casters; j++) {
            if (dot(shadow_s, shadow_s) < 0.001) break;
            
            if (j < 32) { if ((u_caster_mask_lo & (1u << j)) == 0u) continue; }
            else        { if ((u_caster_mask_hi & (1u << (j - 32))) == 0u) continue; }
            
            vec3 caster_pos = u_casters[j].xyz;
            float caster_r = u_casters[j].w;
            vec3 frag_to_caster = caster_pos - f_world_pos;
            float dist_to_caster = length(frag_to_caster);
            
            if (dist_to_caster < caster_r * 1.02) continue;
            
            vec3 V_c = frag_to_caster / dist_to_caster;
            float t_proj = dot(frag_to_caster, L);
            
            float oblateness = u_caster_poles_obl[j].w;
            if (oblateness > 0.0) {
                vec3 pole = u_caster_poles_obl[j].xyz;
                vec3 perp_vec = frag_to_caster - t_proj * L;
                caster_r = get_oblate_radius(caster_r, oblateness, pole, L, perp_vec);
            }
            
            float sin_beta = clamp(caster_r / dist_to_caster, 0.0, 1.0);
            float beta = asin(sin_beta);
            
            float cos_gamma = dot(L, V_c);
            float gamma = acos(clamp(cos_gamma, -1.0, 1.0));
            float R_max = max(alpha, beta);
            float R_min = min(alpha, beta);
            float x_norm = gamma / max(1e-9, R_max);
            float y_norm = R_min / max(1e-9, R_max);
            float area_fraction = texture(u_eclipse_lut, vec2(x_norm * 0.5, y_norm)).r;
            float occlusion = clamp(area_fraction * (R_max * R_max) / max(1e-9, alpha * alpha), 0.0, 1.0);
            
            vec3 caster_shadow = vec3(1.0 - occlusion);
            
            float atmo_h = u_caster_atmos[j].w;
            if (atmo_h > 0.0 && occlusion > 0.0) {
                float r_umbra = abs(alpha - beta);
                float depth_into_umbra = clamp((r_umbra - gamma) / max(1e-6, r_umbra), 0.0, 1.0);
                vec3 atmo_tint = u_caster_atmos[j].xyz;
                vec3 deep_tint = pow(atmo_tint, vec3(1.0 + depth_into_umbra * 0.5));
                float refraction_intensity = exp(-depth_into_umbra * 1.2) * 1.5;
                float atmo_blend = smoothstep(0.5, 1.0, occlusion);
                caster_shadow += deep_tint * refraction_intensity * atmo_blend;
            }
            
            shadow_s *= caster_shadow;
        }   
        
        if (u_host_planet_radius > 0.0) {
            vec3 frag_to_host = u_host_planet_pos - f_world_pos;
            float t = dot(frag_to_host, L);
            if (t > 0.0 && t < dist_to_star) {
                vec3 perp_vec = frag_to_host - t * L;
                float perp_sq = dot(perp_vec, perp_vec);
                float host_r = u_host_planet_radius;
                float host_oblateness = u_host_planet_pole_obl.w;
                float r_star_proj = star_ang_radius * t;
                float r_penumbra = host_r + r_star_proj;
                
                if (host_oblateness > 0.0 || perp_sq < r_penumbra * r_penumbra) {
                    float perp;
                    if (host_oblateness > 0.0) {
                        vec3 host_pole = u_host_planet_pole_obl.xyz;
                        perp = sqrt(perp_sq);
                        host_r = get_oblate_radius(host_r, host_oblateness, host_pole, L, perp_vec);
                        r_penumbra = host_r + r_star_proj;
                    } else {
                        perp = sqrt(perp_sq);
                    }
                    
                    if (perp < r_penumbra) {
                        float r_umbra = abs(host_r - r_star_proj);
                        float occlusion = 0.0;
                        if (perp <= r_umbra) {
                            float div_denom = max(1e-9, r_star_proj * r_star_proj);
                            occlusion = (host_r >= r_star_proj) ? 1.0 : (host_r * host_r) / div_denom;
                        } else {
                            float rc2 = host_r * host_r;
                            float rs2 = r_star_proj * r_star_proj;
                            float h = perp - host_r;
                            float denom_c = 2.0 * perp * host_r;
                            float denom_s = 2.0 * perp * r_star_proj;
                            if (denom_c > 1e-7 && denom_s > 1e-7 && rs2 > 1e-9) {
                                float eps_c = clamp((rs2 - h * h) / denom_c, 0.0, 2.0);
                                float angle_c = 2.0 * asin(sqrt(clamp(eps_c * 0.5, 0.0, 1.0)));
                                float sin_c = sqrt(clamp(eps_c * (2.0 - eps_c), 0.0, 1.0));
                                float x_s = clamp((h * (perp + host_r) + rs2) / denom_s, -1.0, 1.0);
                                float angle_s = acos(x_s);
                                float area = rc2 * angle_c + rs2 * angle_s - perp * host_r * sin_c;
                                occlusion = clamp(area / (3.14159265358979 * rs2), 0.0, 1.0);
                            } else {
                                occlusion = clamp((r_penumbra - perp) / max(1e-7, r_penumbra - r_umbra), 0.0, 1.0);
                                float max_occ = (host_r >= r_star_proj) ? 1.0 : (rc2 / max(1e-9, rs2));
                                occlusion = clamp(occlusion * max_occ, 0.0, 1.0);
                            }
                        }
                        shadow_s *= vec3(1.0 - occlusion);
                    }
                }
            }
        }
        
        total_direct_illum_color += star_color * direct_illum_s * shadow_s;
    }
    
    vec3 total_planetshine = vec3(0.0);
    if (u_host_planet_radius > 0.0) {
        vec3 frag_to_host = u_host_planet_pos - f_world_pos;
        float dist_host_sq = dot(frag_to_host, frag_to_host);
        float dist_host = sqrt(dist_host_sq);
        vec3 dir_to_host = frag_to_host / dist_host;
        float R_sq = u_host_planet_radius * u_host_planet_radius;
        float solid_angle = R_sq / max(dist_host_sq, R_sq);
        float planet_elevation = mix(0.5, 1.0, abs(dot(N, dir_to_host)));
        float planet_side = dot(N, dir_to_host);
        float cam_planet_same_side = smoothstep(-0.02, 0.02, planet_side * cam_side);
        for (int s = 0; s < u_num_stars; s++) {
            vec3 star_pos = u_stars_pos_radius[s].xyz;
            vec3 star_color = u_stars_colors[s].rgb;
            float star_lum = u_stars_colors[s].a;
            vec3 host_to_star = star_pos - u_host_planet_pos;
            float dist_host_star = length(host_to_star);
            float irradiance = u_hdr_enabled ? (star_lum / max(dist_host_star * dist_host_star, 1e-8)) : 1.0;
            
            host_to_star /= max(dist_host_star, 1e-6);
            float planet_phase = max(0.0, dot(host_to_star, -dir_to_host));
            float shine_intensity = planet_phase * solid_angle * planet_elevation;
            float p_reflect = rock_reflect + 0.3 * f_color.a; 
            float p_transmit = rock_transmit + 0.2 * f_color.a;
            float shine_response = mix(p_transmit, p_reflect, cam_planet_same_side);
            total_planetshine += u_host_planet_color * star_color * irradiance * (shine_intensity * shine_response * 0.2);
        }
    }
    
    vec3 illumination = total_direct_illum_color;
    if (u_planetshine_enabled) {
        illumination += total_planetshine;
    }
    vec3 raw_color = f_color.rgb * illumination;
    vec3 final_color = u_hdr_enabled ? (raw_color * u_exposure) : raw_color;
    out_color = vec4(final_color, physical_alpha);
}
"""

atmo_vertex_shader = """
#version 460 core
in vec3 in_position;

#define MAX_CASTERS 64
#define MAX_STARS 16
layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
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
    gl_Position.z = (log2(max(1e-6, u_depth_C * gl_Position.w + 1.0)) / log2(u_depth_C * u_far + 1.0) * 2.0 - 1.0) * gl_Position.w;
    f_clip_z = gl_Position.w;
}
"""

atmo_fragment_shader = """
#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define MAX_RING_PLANES 16
#define PI 3.14159265358979

in vec3 f_world_pos;
in vec3 f_local_pos;
in float f_clip_z;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};

layout(std430, binding = 2) buffer AllInstances {
    vec4 instances[];
};
uniform int u_body_idx;

uint u_caster_mask_lo;
uint u_caster_mask_hi;
uint u_ring_mask;

uniform vec3  u_body_offset;
uniform float u_atmo_radius_au;
uniform float u_planet_radius_km;
uniform float u_atmo_radius_km;
uniform float u_au_to_km;
uniform float u_star_angular_radius;
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
uniform sampler2D u_optical_depth_lut;

// Ring shadow planes (sphere-only)
uniform sampler2D u_ring_gradients;
uniform int u_num_ring_planes;
uniform int u_atmo_quality;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec3 u_ring_params[MAX_RING_PLANES]; // x=inner_r, y=outer_r, z=opacity
uniform int u_atmo_clip_mode; // 1 = behind rings, 2 = in front of rings

uniform int u_num_active_casters;
uniform vec4 u_active_casters[8];
uniform vec4 u_active_caster_poles_obl[8];
uniform vec4 u_active_caster_atmos[8];
uniform sampler2D u_eclipse_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;

out vec4 out_color;

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

float get_oblate_radius(float r_eq, float oblateness, vec3 pole, vec3 L, vec3 perp_vec) {
    if (oblateness <= 0.0 || r_eq < 1e-6) return r_eq;
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
    float angle = atan(y, x);
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    return (r_eq * R_minor) / sqrt(R_minor * R_minor * cos_a * cos_a + r_eq * r_eq * sin_a * sin_a);
}

vec3 compute_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, vec3 planet_center_render, float star_radius) {
    vec3 shadow = vec3(1.0);
    for (int i = 0; i < u_num_active_casters; i++) {
        vec3 caster_pos = u_active_casters[i].xyz;
        float caster_r = u_active_casters[i].w;
        vec3 s_to_c = caster_pos - eval_render_pos;
        float dist_to_caster = length(s_to_c);
        
        if (dist_to_caster < caster_r * 1.02) continue;
        
        vec3 V_c = s_to_c / dist_to_caster;
        float proj = dot(s_to_c, L_dir);
        
        float oblateness = u_active_caster_poles_obl[i].w;
        if (oblateness > 0.0) {
            vec3 pole = u_active_caster_poles_obl[i].xyz;
            vec3 perp_vec = s_to_c - proj * L_dir;
            caster_r = get_oblate_radius(caster_r, oblateness, pole, L_dir, perp_vec);
        }
        
        float sin_alpha_local = clamp(star_radius / dist_to_star, 0.0, 1.0);
        float alpha = asin(sin_alpha_local);
        
        float sin_beta = clamp(caster_r / dist_to_caster, 0.0, 1.0);
        float beta = asin(sin_beta);
        
        float cos_gamma = dot(L_dir, V_c);
        float gamma = acos(clamp(cos_gamma, -1.0, 1.0));
        float R_max = max(alpha, beta);
        float R_min = min(alpha, beta);
        float x_norm = gamma / max(1e-9, R_max);
        float y_norm = R_min / max(1e-9, R_max);
        float area_fraction = texture(u_eclipse_lut, vec2(x_norm * 0.5, y_norm)).r;
        float occ = clamp(area_fraction * (R_max * R_max) / max(1e-9, alpha * alpha), 0.0, 1.0);
        
        vec3 caster_shadow = vec3(1.0 - occ);
        float atmo_h = u_active_caster_atmos[i].w;
        if (atmo_h > 0.0 && occ > 0.0) {
            float r_umbra = abs(alpha - beta);
            float depth_into_umbra = clamp((r_umbra - gamma) / max(1e-6, r_umbra), 0.0, 1.0);
            vec3 atmo_tint = u_active_caster_atmos[i].xyz;
            vec3 deep_tint = pow(atmo_tint, vec3(1.0 + depth_into_umbra * 0.5));
            float refraction_intensity = exp(-depth_into_umbra * 1.2) * 1.5;
            float atmo_blend = smoothstep(0.5, 1.0, occ);
            caster_shadow += deep_tint * refraction_intensity * atmo_blend;
        }
        shadow *= caster_shadow;
    }
    uint processed_mask = 0u;
    for (int i = 0; i < g_num_local_rings; i++) {
        int k = g_local_rings[i];
        if ((processed_mask & (1u << k)) != 0u) continue;
        
        vec3 plane_center = u_ring_center[k];
        vec3 plane_normal = u_ring_normal[k];
        
        uint coplanar_mask = 0u;
        for (int j = i; j < g_num_local_rings; j++) {
            int ring_idx = g_local_rings[j];
            if (distance(plane_center, u_ring_center[ring_idx]) < 1e-6 && 
                dot(plane_normal, u_ring_normal[ring_idx]) > 0.999) {
                coplanar_mask |= (1u << ring_idx);
                processed_mask |= (1u << ring_idx);
            }
        }
        
        float denom = dot(L_dir, plane_normal);
        if (abs(denom) < 1e-8) continue;
        
        float t_ring = dot(plane_center - eval_render_pos, plane_normal) / denom;
        if (t_ring <= 0.0 || t_ring >= dist_to_star) continue;
        
        vec3 hit = eval_render_pos + t_ring * L_dir;
        vec3 vec_radial = hit - plane_center;
        float d = length(vec_radial);
        
        float r_star_proj = star_radius * t_ring / dist_to_star;
        vec3 L_plane = L_dir - denom * plane_normal;
        float L_plane_len = length(L_plane);
        
        float R_eff = r_star_proj;
        if (L_plane_len > 1e-5 && d > 1e-5) {
            vec3 L_proj = L_plane / L_plane_len;
            vec3 T = normalize(cross(plane_normal, L_dir));
            vec3 dir_radial = vec_radial / d;
            float cos_theta = dot(dir_radial, L_proj);
            float sin_theta = dot(dir_radial, T);
            R_eff = r_star_proj * sqrt( pow(cos_theta / max(1e-6, abs(denom)), 2.0) + pow(sin_theta, 2.0) );
        }
        
        float plane_occlusion = 0.0;
        
        for (int j = i; j < g_num_local_rings; j++) {
            int ring_idx = g_local_rings[j];
            if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
            float inner_r = u_ring_params[ring_idx].x;
            float outer_r = u_ring_params[ring_idx].y;
            
            float overlap_min = max(inner_r, d - R_eff);
            float overlap_max = min(outer_r, d + R_eff);
            
            if (overlap_min >= overlap_max) continue;
            
            float v_min = clamp((overlap_min - d) / max(1e-9, R_eff), -1.0, 1.0);
            float v_max = clamp((overlap_max - d) / max(1e-9, R_eff), -1.0, 1.0);
            
            float f_max = (v_max * sqrt(max(0.0, 1.0 - v_max*v_max)) + asin(v_max)) / 3.14159265358979 + 0.5;
            float f_min = (v_min * sqrt(max(0.0, 1.0 - v_min*v_min)) + asin(v_min)) / 3.14159265358979 + 0.5;
            float fraction = max(0.0, f_max - f_min);
            
            float sum_alpha = 0.0;
            int tex_samples = 5;
            for(int s = 0; s < tex_samples; s++) {
                float u = (float(s) + 0.5) / float(tex_samples);
                float r = mix(overlap_min, overlap_max, u);
                float p = (r - inner_r) / max(1e-6, outer_r - inner_r);
                sum_alpha += texture(u_ring_gradients, vec2(p, (float(ring_idx) + 0.5) / 16.0)).r;
            }
            float alpha_mult = sum_alpha / float(tex_samples);
            float opacity = u_ring_params[ring_idx].z;
            
            plane_occlusion += fraction * opacity * alpha_mult;
        }
        
        shadow *= vec3(1.0 - min(plane_occlusion, 1.0));
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
    u_ring_mask = floatBitsToUint(instances[u_body_idx * 6 + 3].w);
    
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

    // Clip against closest active ring plane to avoid drawing atmosphere over rings that are in front
    // and correctly obscuring rings that are inside the atmosphere.
    float closest_s_ring = 1e10;
    for (int k = 0; k < u_num_ring_planes; k++) {
        vec3 ring_center_world_rel = u_ring_center[k];
        if (length(ring_center_world_rel - u_body_offset) > 1e-4) continue;
        
        vec3 ring_center_local = (ring_center_world_rel - u_body_offset) * u_au_to_km;
        vec3 ring_normal = u_ring_normal[k];
        
        float denom = dot(ray_dir, ring_normal);
        if (abs(denom) > 1e-8) {
            float s_ring = dot(ring_center_local - frag_local, ring_normal) / denom;
            if (s_ring > 0.0) {
                vec3 hit_local = frag_local + s_ring * ray_dir;
                float dist_from_center = length(hit_local - ring_center_local);
                float inner_r_km = u_ring_params[k].x * u_au_to_km;
                float outer_r_km = u_ring_params[k].y * u_au_to_km;
                
                if (dist_from_center >= inner_r_km && dist_from_center <= outer_r_km) {
                    if (s_ring < closest_s_ring) {
                        closest_s_ring = s_ring;
                    }
                }
            }
        }
    }

    if (closest_s_ring < 1e9) {
        if (u_atmo_clip_mode == 1) {
            s_start = max(s_start, closest_s_ring);
        } else if (u_atmo_clip_mode == 2) {
            s_end = min(s_end, closest_s_ring);
        }
    } else {
        if (u_atmo_clip_mode == 2) {
            s_end = s_start;
        }
    }



    // Clip against all neighboring opaque bodies (moons, planet) to fix depth sorting
    // since atmosphere is rendered with DEPTH_TEST disabled.
    for (int i = 0; i < u_num_active_casters; i++) {
        vec3 caster_pos = u_active_casters[i].xyz;
        float caster_r = u_active_casters[i].w * u_au_to_km;
        vec3 caster_local = (caster_pos - planet_center_render) * u_au_to_km;
        
        // Skip the host planet itself (already clipped precisely above)
        if (length(caster_local) < 1.0) continue;
        
        vec4 pole_obl = u_active_caster_poles_obl[i];
        vec3 origin_c = toSphericalSpace(frag_local - caster_local, pole_obl);
        vec3 dir_c = toSphericalSpace(ray_dir, pole_obl);
        
        vec2 s_c = raySphereIntersect(origin_c, dir_c, caster_r);
        if (s_c.x > s_cam && s_c.x < s_end) {
            s_end = s_c.x;
        }
    }

    if (s_start >= s_end) discard;

    // Scattering coefficients: convert m^-1 to km^-1
    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M = vec3(u_beta_mie) * 1000.0;
    vec3 beta_A = u_beta_absorption * 1000.0;
    float mie_ext_factor = 1.0 / 0.9;

    // 4. Primary ray march
    float step_size = (s_end - s_start) / float(u_num_samples);
    vec3 scattered = vec3(0.0);
    float final_od_rayleigh = 0.0;
    float final_od_mie = 0.0;

    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        vec3 star_color = u_stars_colors[s].rgb;
        float star_lum = u_stars_colors[s].a;

        vec3 sun_pos_local = (star_pos - planet_center_render) * u_au_to_km;
        vec3 sun_dir = normalize(sun_pos_local);
        vec3 sun_dir_sph = toSphericalSpace(sun_dir, u_pole_obl);

        // 3.5 Global Eclipse & Ring Shadow
        vec3 global_eclipse_shadow = vec3(1.0);
        vec3 end_eclipse_shadow = vec3(1.0);
        bool skip_volumetric_shadow = false;

        if (u_atmo_quality > 0) { // Low or High Quality
            float s_mid = (s_start + s_end) * 0.5;
            vec3 mid_pos = frag_local + s_mid * ray_dir;
            vec3 mid_render = mid_pos / u_au_to_km + planet_center_render;
            vec3 mid_to_star = star_pos - mid_render;
            float dist_mid_star = length(mid_to_star);
            vec3 L_mid = mid_to_star / dist_mid_star;
            global_eclipse_shadow = compute_shadow(mid_render, L_mid, dist_mid_star, planet_center_render, star_radius);
            end_eclipse_shadow = global_eclipse_shadow;

            // --- High Quality Early-Out ---
            if (u_atmo_quality == 2) {
                vec3 start_pos = frag_local + s_start * ray_dir;
                vec3 start_render = start_pos / u_au_to_km + planet_center_render;
                vec3 start_to_star = star_pos - start_render;
                vec3 shadow_start = compute_shadow(start_render, start_to_star / length(start_to_star), length(start_to_star), planet_center_render, star_radius);
                
                vec3 end_pos = frag_local + s_end * ray_dir;
                vec3 end_render = end_pos / u_au_to_km + planet_center_render;
                vec3 end_to_star = star_pos - end_render;
                vec3 shadow_end = compute_shadow(end_render, end_to_star / length(end_to_star), length(end_to_star), planet_center_render, star_radius);
                
                global_eclipse_shadow = shadow_start;
                end_eclipse_shadow = shadow_end;

                if (shadow_start.r > 0.999 && shadow_end.r > 0.999) {
                    skip_volumetric_shadow = true;
                } else if (shadow_start.r < 0.001 && shadow_end.r < 0.001) {
                    skip_volumetric_shadow = true;
                }
            }
        }

        float od_rayleigh = 0.0;
        float od_mie = 0.0;
        vec3 total_rayleigh = vec3(0.0);
        vec3 total_mie = vec3(0.0);

        for (int i = 0; i < u_num_samples; i++) {
            float s_val = s_start + (float(i) + 0.5) * step_size;
            vec3 sample_pos = frag_local + s_val * ray_dir;
            vec3 sample_pos_sph = frag_local_sph + s_val * ray_dir_sph;
            float sample_len = length(sample_pos_sph);
            float altitude = sample_len - u_planet_radius_km;

            float rho_R = exp(-altitude / u_h_rayleigh);
            float rho_M = exp(-altitude / u_h_mie);

            od_rayleigh += rho_R * step_size;
            od_mie += rho_M * step_size;

            // 5. Light ray: optical depth from sample to sun using LUT
            float light_cos_theta = dot(sample_pos_sph / sample_len, sun_dir_sph);
            float h_norm = clamp(altitude / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
            float uv_x = light_cos_theta * 0.5 + 0.5;
            vec2 lut_uv = vec2(uv_x, h_norm);
            
            vec4 od_light = texture(u_optical_depth_lut, lut_uv);
            
            // --- Soft terminator correction ---
            float sin_planet_angle = u_planet_radius_km / max(sample_len, u_planet_radius_km + 0.01);
            float cos_horizon = -sqrt(max(0.0, 1.0 - sin_planet_angle * sin_planet_angle));

            float od_light_R = od_light.r;
            float od_light_M = od_light.g;
            float penumbra_factor = 1.0;

            float sin_star = u_star_angular_radius;
            if (sin_star > 1e-5) {
                float angular_dist = light_cos_theta - cos_horizon;
                float penumbra_width = sin_star * 2.0;
                penumbra_factor = clamp(angular_dist / max(1e-6, penumbra_width) + 0.5, 0.0, 1.0);
                
                if (penumbra_factor < 0.999) {
                    vec2 horizon_uv = vec2((-cos_horizon) * 0.5 + 0.5, h_norm);
                    vec4 od_horizon = texture(u_optical_depth_lut, horizon_uv);
                    
                    if (penumbra_factor < 0.5) {
                        od_light_R = od_horizon.r;
                        od_light_M = od_horizon.g;
                    } else {
                        float blend_t = (penumbra_factor - 0.5) / 0.5;
                        od_light_R = mix(od_horizon.r, od_light.r, blend_t);
                        od_light_M = mix(od_horizon.g, od_light.g, blend_t);
                    }
                }
            } else {
                if (light_cos_theta < cos_horizon) {
                    penumbra_factor = 0.0;
                }
            }

            // 7. Accumulate in-scattered light
            vec3 tau = beta_R * (od_rayleigh + od_light_R)
                     + beta_M * mie_ext_factor * (od_mie + od_light_M)
                     + beta_A * (od_rayleigh + od_light_R);

            vec3 sample_shadow = global_eclipse_shadow;
            if (u_atmo_quality == 2 && !skip_volumetric_shadow) { // High Quality (Volumetric)
                float t_lerp = (float(i) + 0.5) / float(u_num_samples);
                sample_shadow = mix(global_eclipse_shadow, end_eclipse_shadow, t_lerp);
            }
            // Approximate multiple scattering with a low-extinction term
            vec3 direct_attenuation = exp(-tau);
            vec3 ms_attenuation = max(vec3(0.0), (exp(-tau * 0.2) - direct_attenuation) * 0.4);
            vec3 attenuation = (direct_attenuation + ms_attenuation) * sample_shadow * penumbra_factor;

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

        // Inverse-square falloff for irradiance
        float dist_to_star = length(star_pos - planet_center_render);
        float irradiance = u_hdr_enabled ? (star_lum / max(dist_to_star * dist_to_star, 1e-8)) : 1.0;
        
        // 9. Final scattered light
        scattered += star_color * u_sun_intensity * irradiance * (
            phase_R * beta_R * total_rayleigh +
            phase_M * beta_M * total_mie
        );

        final_od_rayleigh = od_rayleigh;
        final_od_mie = od_mie;
    }

    // 10. Primary ray transmittance
    vec3 transmittance = exp(-(beta_R * final_od_rayleigh
                             + beta_M * mie_ext_factor * final_od_mie
                             + beta_A * final_od_rayleigh));

    // Apply Exposure
    if (u_hdr_enabled) {
        scattered *= u_exposure;
    }

    // 11. Premultiplied alpha output
    float avg_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    out_color = vec4(scattered, (1.0 - avg_transmittance));

    // Logarithmic depth done in vertex shader
}
"""

atmo_lut_vertex_shader = """
#version 460 core
in vec2 in_position;
out vec2 f_uv;
void main() {
    f_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

atmo_lut_fragment_shader = """
#version 460 core
in vec2 f_uv;
out vec4 out_color;

uniform float u_planet_radius_km;
uniform float u_atmo_radius_km;
uniform float u_h_rayleigh;
uniform float u_h_mie;

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
    float cos_theta = f_uv.x * 2.0 - 1.0;
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float h = f_uv.y * max(1e-4, u_atmo_radius_km - u_planet_radius_km);
    float r = u_planet_radius_km + h;
    
    vec3 origin = vec3(0.0, r, 0.0);
    vec3 dir = vec3(sin_theta, cos_theta, 0.0);
    
    vec2 t_atmo = raySphereIntersect(origin, dir, u_atmo_radius_km);
    vec2 t_planet = raySphereIntersect(origin, dir, u_planet_radius_km);
    
    float ray_len = t_atmo.y;
    if (t_planet.x > 0.0 && t_planet.x < t_atmo.y) {
        // Ray hits the planet — integrate to the surface
        ray_len = t_planet.x;
    }
    
    int num_samples = 256;
    float step_size = ray_len / float(num_samples);
    
    float od_rayleigh = 0.0;
    float od_mie = 0.0;
    
    for (int i = 0; i < num_samples; i++) {
        float t = (float(i) + 0.5) * step_size;
        vec3 p = origin + t * dir;
        float h_sample = max(0.0, length(p) - u_planet_radius_km);
        
        od_rayleigh += exp(-h_sample / u_h_rayleigh) * step_size;
        od_mie += exp(-h_sample / u_h_mie) * step_size;
    }
    
    out_color = vec4(od_rayleigh, od_mie, 0.0, 1.0);
}
"""
