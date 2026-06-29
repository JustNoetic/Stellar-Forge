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
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
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
out float f_brightness_scale;
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
    float brightness_scale = 1.0;
    
    if (apparent_px < in_min_size) {
        final_radius = (in_min_size * dist) / (screen_height * fov_factor);
        float ratio = apparent_px / in_min_size;
        brightness_scale = ratio;
    }
    f_brightness_scale = brightness_scale;
    
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
in float f_brightness_scale;
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
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
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
uniform float u_caster_max_bend[64];
uniform uint u_ring_coplanar_mask[16];
uniform int u_atmo_quality;
uniform bool u_planetshine_enabled;
uniform bool u_ringshine_enabled;
uniform sampler2D u_eclipse_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform vec3 u_camera_pos;
uniform sampler3D u_aerial_perspective_volume;
uniform float u_au_to_km;

out vec4 out_color;

float get_oblate_radius(float r_eq, float r_minor, vec3 pole, vec3 L, vec3 perp_vec) {
    if (r_minor < 1e-6 || r_eq < 1e-6) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / r_minor) * (y / r_minor));
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
        float star_r = 0.0046547454; // 1 solar radius in AU
        vec3 star_base_color = f_color; // fallback
        for (int s = 0; s < u_num_stars; s++) {
            if (distance(f_world_pos, u_stars_pos_radius[s].xyz) < u_stars_pos_radius[s].w * 1.5) {
                star_r = max(u_stars_pos_radius[s].w, 1e-6);
                
                vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
                float sin_lat = abs(dot(N, pole_dir));
                
                vec3 eq_color = u_stars_colors[s].rgb;
                float eq_lum = u_stars_colors[s].a;
                vec3 pole_color = u_stars_pole_colors[s].rgb;
                float pole_lum = u_stars_pole_colors[s].a;
                
                star_base_color = mix(eq_color, pole_color, sin_lat);
                star_lum = mix(eq_lum, pole_lum, sin_lat);
                break;
            }
        }
        
        float surface_luminance = u_hdr_enabled ? (star_lum / (star_r * star_r)) : 1.0;
        vec3 final_star_color = star_base_color * color_shift * ld * surface_luminance;
        if (u_hdr_enabled) {
            final_star_color *= u_exposure;
        }
        out_color = vec4(final_star_color * f_brightness_scale, 1.0);
    } else {
        vec3 N = normalize(f_normal);
        vec3 total_diffuse_color = vec3(0.0);
        
        for (int s = 0; s < u_num_stars; s++) {
            vec3 star_pos = u_stars_pos_radius[s].xyz;
            float star_radius = u_stars_pos_radius[s].w;
            vec3 frag_to_star = star_pos - f_world_pos;
            float dist_to_star = length(frag_to_star);
            if (dist_to_star < 1e-5) continue;
            vec3 L = frag_to_star / dist_to_star;
            
            vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
            float sin_lat = abs(dot(L, pole_dir));
            
            vec3 eq_color = u_stars_colors[s].rgb;
            float eq_lum = u_stars_colors[s].a;
            vec3 pole_color = u_stars_pole_colors[s].rgb;
            float pole_lum = u_stars_pole_colors[s].a;
            
            vec3 star_color = mix(eq_color, pole_color, sin_lat);
            float star_lum = mix(eq_lum, pole_lum, sin_lat);
            
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
            float star_radius_over_dist = star_radius / dist_to_star;
            
            for (int j = 0; j < u_num_casters; j++) {
                if (dot(shadow, shadow) < 0.001) break;
                
                if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
                else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }
                
                vec3 caster_pos = u_casters[j].xyz;
                float caster_r = u_casters[j].w;
                float atmo_h = u_caster_atmos[j].w;
                
                vec3 frag_to_caster = caster_pos - f_world_pos;
                float t_proj = dot(frag_to_caster, L);
                if (t_proj < 0.0) continue; 
                
                float dist_sq = dot(frag_to_caster, frag_to_caster);
                if (dist_sq < caster_r * caster_r * 1.0404) continue; // Skip self
                
                float dist_to_caster = sqrt(dist_sq);
                float perp_sq = max(0.0, dist_sq - t_proj * t_proj);
                
                // Bounding cone early out using maximum equatorial radius
                float max_effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
                float max_r_penumbra = max_effective_r + dist_to_caster * star_radius_over_dist;
                if (perp_sq > max_r_penumbra * max_r_penumbra) continue;
                
                float caster_r_minor = u_caster_poles_obl[j].w;
                vec3 perp_vec = frag_to_caster - t_proj * L;
                if (caster_r_minor < caster_r - 1e-5) {
                    vec3 pole = u_caster_poles_obl[j].xyz;
                    caster_r = get_oblate_radius(caster_r, caster_r_minor, pole, L, perp_vec);
                }
                
                float directional_star_r = star_radius;
                float star_r_minor = u_stars_poles_obl[s].w;
                if (star_r_minor < star_radius - 1e-5) {
                    vec3 star_pole = u_stars_poles_obl[s].xyz;
                    directional_star_r = get_oblate_radius(star_radius, star_r_minor, star_pole, L, perp_vec);
                }
                float local_star_radius_over_dist = directional_star_r / dist_to_star;
                
                // Perfect Bounding Cone Early Out (Zero Artifacts)
                float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
                float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
                if (perp_sq > r_penumbra * r_penumbra) continue;
                
                float inv_dist = 1.0 / dist_to_caster;
                float alpha = local_star_radius_over_dist;
                float beta = caster_r * inv_dist;
                float gamma = sqrt(perp_sq) * inv_dist;
                
                float penumbra_outer = alpha + beta;
                float penumbra_inner = max(0.0, beta - alpha);
                
                float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
                float occ = max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma);
                
                vec3 caster_shadow = vec3(1.0 - occ);
                
                float max_bend = u_caster_max_bend[j];
                if (max_bend > 0.0 && gamma < penumbra_outer) {
                    float req_bend = beta - gamma;
                    
                    float optical_depth = max(0.0, req_bend);
                    float atmospheric_transmission = exp(-optical_depth * 150.0);
                    float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
                    
                    float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
                    vec3 atmo_tint = u_caster_atmos[j].xyz;
                    vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
                    
                    float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
                    float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
                    float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
                    
                    caster_shadow += deep_tint * refraction_intensity * atmo_blend;
                }
                
                shadow *= clamp(caster_shadow, 0.0, 1.0);
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
                    
                    uint coplanar_mask = u_ring_coplanar_mask[k];
                    processed_mask |= coplanar_mask;
                    
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
                        float overlap_width = overlap_max - overlap_min;
                        if (overlap_width < 0.002 * (outer_r - inner_r)) {
                            // Penumbra is extremely narrow, 1 sample is sufficient
                            float r = 0.5 * (overlap_min + overlap_max);
                            float p = (r - inner_r) / max(1e-6, outer_r - inner_r);
                            sum_alpha = textureLod(u_ring_gradients, vec2(p, (float(j) + 0.5) / 16.0), 0.0).r;
                        } else {
                            // Penumbra is wide, use 5 samples for filtering
                            int tex_samples = 5;
                            for(int s = 0; s < tex_samples; s++) {
                                float u = (float(s) + 0.5) / float(tex_samples);
                                float r = mix(overlap_min, overlap_max, u);
                                float p = (r - inner_r) / max(1e-6, outer_r - inner_r);
                                sum_alpha += textureLod(u_ring_gradients, vec2(p, (float(j) + 0.5) / 16.0), 0.0).r;
                            }
                            sum_alpha /= float(tex_samples);
                        }
                        float alpha_mult = sum_alpha;
                        float opacity = u_ring_params[j].z;
                        
                        float tau = -log(max(1e-6, 1.0 - opacity * alpha_mult));
                        float light_mu = max(1e-4, abs(dot(normalize(u_ring_normal[j]), L)));
                        plane_occlusion += fraction * (1.0 - exp(-tau / light_mu));
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
            vec3 ring_center = u_ring_center[k];
            vec3 ring_normal = u_ring_normal[k];
            float inner_r = u_ring_params[k].x;
            float outer_r = u_ring_params[k].y;
            float opacity = u_ring_params[k].z;
            
            vec3 frag_to_center = ring_center - f_world_pos;
            float dist_to_center = length(frag_to_center);
            
            // Fast distance cull: ignore rings if we are too far away
            if (dist_to_center > outer_r * 20.0 || dist_to_center < 1e-6) continue;
            
            // If the fragment is closer to the ring center than the inner edge of the rings, 
            // it physically must be the Host Planet. Otherwise, it is a Moon.
            bool is_host_planet = (dist_to_center < inner_r);
            
            float frag_elevation = dot(N, ring_normal);
            float moon_h = -dot(frag_to_center, ring_normal);
            
            int host_caster_idx = -1;
            if (!is_host_planet) {
                for (int j = 0; j < u_num_casters; j++) {
                    if (distance(u_casters[j].xyz, ring_center) < 1e-4) {
                        host_caster_idx = j;
                        break;
                    }
                }
            }
            
            for (int s = 0; s < u_num_stars; s++) {
                vec3 star_pos = u_stars_pos_radius[s].xyz;
                float star_radius = u_stars_pos_radius[s].w;
                vec3 frag_to_star = star_pos - f_world_pos;
                float dist_to_star = length(frag_to_star);
                if (dist_to_star < 1e-5) continue;
                vec3 L = frag_to_star / dist_to_star;
                
                vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
                float sin_lat = abs(dot(L, pole_dir));
                
                vec3 eq_color = u_stars_colors[s].rgb;
                float eq_lum = u_stars_colors[s].a;
                vec3 pole_color = u_stars_pole_colors[s].rgb;
                float pole_lum = u_stars_pole_colors[s].a;
                
                vec3 star_color = mix(eq_color, pole_color, sin_lat);
                float star_lum = mix(eq_lum, pole_lum, sin_lat);
                
                float sun_elevation = dot(L, ring_normal);
                float star_ang_radius = star_radius / max(dist_to_star, 1e-6);
                float effective_sun_elev = sqrt(sun_elevation * sun_elevation + 0.180126 * star_ang_radius * star_ang_radius);
                
                float shine_intensity = 0.0;
                
                if (is_host_planet) {
                    // --- HOST PLANET MACRO APPROXIMATION ---
                    float abs_elev = abs(frag_elevation);
                    float form_factor = abs_elev * (1.0 - abs_elev) * (1.0 - abs_elev) * 6.75; 
                    float ring_area = (outer_r * outer_r - inner_r * inner_r);
                    float solid_angle = ring_area / max(dist_to_center * dist_to_center, ring_area) * 0.1;
                    
                    float same_hemisphere = sun_elevation * frag_elevation;
                    float shadow_occlusion = 1.0;
                    if (dot(N, L) < 0.0) {
                        float anti_solar = max(0.0, dot(N, -L));
                        shadow_occlusion = 1.0 - (anti_solar * 0.85);
                    }
                    float noon_fade = 1.0 - max(0.0, dot(N, L));
                    
                    if (same_hemisphere > 0.0) {
                        shine_intensity = effective_sun_elev * form_factor * solid_angle * opacity * 3.5 * noon_fade;
                    } else {
                        shine_intensity = effective_sun_elev * form_factor * solid_angle * opacity * 1.25;
                    }
                    shine_intensity *= shadow_occlusion;
                    
                } else {
                    // --- DEDICATED MOON RINGSHINE ---
                    vec3 closest_plane_pt = f_world_pos - moon_h * ring_normal;
                    vec3 center_to_plane_pt = closest_plane_pt - ring_center;
                    float rho = length(center_to_plane_pt);
                    
                    // Find the closest point of actual ring matter to the moon
                    vec3 closest_ring_pt;
                    if (rho < 1e-5) {
                        closest_ring_pt = ring_center + inner_r * vec3(1.0, 0.0, 0.0); 
                    } else if (rho < inner_r) {
                        closest_ring_pt = ring_center + (center_to_plane_pt / rho) * inner_r;
                    } else if (rho > outer_r) {
                        closest_ring_pt = ring_center + (center_to_plane_pt / rho) * outer_r;
                    } else {
                        closest_ring_pt = closest_plane_pt; 
                    }
                    
                    // Calculate light direction from the closest ring point
                    vec3 L_ring_unnorm = closest_ring_pt - f_world_pos;
                    float d_ring = max(length(L_ring_unnorm), 1e-6);
                    vec3 L_ring = L_ring_unnorm / d_ring;
                    
                    // --- UMBRA SHADOW SOFTENING ---
                    // Calculate how deep the fragment is inside the host planet's shadow
                    vec3 V = closest_ring_pt - ring_center;
                    float t_proj = dot(V, -L); // -L is the anti-solar direction
                    vec3 V_perp = V - t_proj * (-L);
                    float d_axis = length(V_perp);
                    
                    // Find the host planet radius dynamically to determine shadow width
                    float host_radius = inner_r * 0.7; // sensible fallback
                    if (host_caster_idx >= 0) {
                        int j = host_caster_idx;
                        host_radius = u_casters[j].w;
                        float host_r_minor = u_caster_poles_obl[j].w;
                        if (host_r_minor < host_radius - 1e-5) {
                            host_radius = get_oblate_radius(host_radius, host_r_minor, u_caster_poles_obl[j].xyz, -L, V_perp);
                        }
                    }
                    
                    float ring_shadow_factor = 1.0;
                    float shadow_wrap = 0.2; // Base wrap for rings (they are broad area lights)
                    
                    // Check if inside the cylindrical shadow (behind planet & within radius)
                    if (t_proj > 0.0) {
                        // User requested 0.9 for the inner bound. 
                        // We stretch the outer bound to 1.15 for a very gradual, soft darkening.
                        float shadow_gradient = smoothstep(host_radius * 0.9, host_radius * 1.15, d_axis);
                        ring_shadow_factor = mix(0.05, 1.0, shadow_gradient);
                        
                        // As the moon goes deeper into the shadow, the lit ring becomes a 180-degree halo.
                        // We significantly increase the lighting wrap to simulate this massive area light,
                        // which completely eliminates the sharp terminator line and softly wraps the sphere.
                        shadow_wrap = mix(0.8, 0.2, shadow_gradient); 
                    }
                    
                    // Wrapped Lambertian: Light smoothly wraps around the sphere based on area light size
                    float NdotL_raw = dot(N, L_ring);
                    float NdotL_ring = max(0.0, (NdotL_raw + shadow_wrap) / (1.0 + shadow_wrap));
                    
                    if (NdotL_ring > 0.0) {
                        // Prevent pure black ring-plane edge cases by adding a tiny effective thickness
                        float sin_elev = clamp((abs(moon_h) + inner_r * 0.001) / d_ring, 0.0, 1.0);
                        float ring_area = (outer_r * outer_r - inner_r * inner_r);
                        
                        float d_eff = max(d_ring, inner_r * 0.05); 
                        float solid_angle = (ring_area * sin_elev) / max(d_eff * d_eff, ring_area * 0.1) * 0.15;
                        
                        // Check if the moon is above or below the sunlit side of the rings
                        float same_hemisphere = sun_elevation * moon_h;
                        
                        // Halved brightness per user request
                        float ring_brightness = (same_hemisphere > 0.0) ? (effective_sun_elev * opacity * 1.25) : (effective_sun_elev * opacity * 0.3);
                        
                        shine_intensity = NdotL_ring * ring_brightness * solid_angle * ring_shadow_factor;
                    }
                }
                
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
        
        // Aerial perspective bypassed for cleaner rendering and matching ray marching
        float trans = 1.0;
        vec3 ap_rgb = vec3(0.0);
        out_color = vec4(final_color * f_brightness_scale * trans + ap_rgb, 1.0);
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
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
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
    vec4 u_stars_poles_obl[MAX_STARS];
    vec4 u_stars_pole_colors[MAX_STARS];
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
uniform vec4 u_host_planet_atmo;
uniform vec3 u_camera_pos;
uniform vec4 u_host_planet_pole_obl;
uniform float u_host_planet_R_minor;
uniform int u_clip_mode;
uniform uint u_caster_mask_lo;
uniform uint u_caster_mask_hi;
uniform bool u_planetshine_enabled;
uniform sampler2D u_eclipse_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform sampler3D u_aerial_perspective_volume;
uniform float u_au_to_km;
uniform float u_caster_max_bend[64];

struct RingPlane {
    vec3 color;
    float inner_r;
    float outer_r;
    float opacity;
    float scatter;
    float asymmetry;
    float backscatter;
    int row_idx;
    float _pad1;
    float _pad2;
};

#define MAX_RING_PLANES 16
uniform RingPlane u_ring_planes[MAX_RING_PLANES];
uniform int u_num_ring_planes;
uniform sampler2D u_ring_gradients;

out vec4 out_color;

float get_oblate_radius(float r_eq, float r_minor, vec3 pole, vec3 L, vec3 perp_vec) {
    if (r_minor < 1e-6 || r_eq < 1e-6) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / r_minor) * (y / r_minor));
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
    float total_faded_tau = 0.0;
    float total_scatter = 0.0;
    float total_asym = 0.0;
    float total_backscatter = 0.0;
    
    for (int i=0; i<u_num_ring_planes; i++) {
        float inner_r = u_ring_planes[i].inner_r;
        float outer_r = u_ring_planes[i].outer_r;
        float dr = fwidth(r) * 0.75;
        
        if (r >= inner_r - dr && r <= outer_r + dr) {
            float t = (r - inner_r) / max(1e-6, outer_r - inner_r);
            float alpha = texture(u_ring_gradients, vec2(clamp(t, 0.0, 1.0), (float(u_ring_planes[i].row_idx) + 0.5)/16.0)).r;
            
            float edge_alpha = smoothstep(inner_r - dr, inner_r + dr, r) * (1.0 - smoothstep(outer_r - dr, outer_r + dr, r));
            
            float raw_a_physical = alpha * u_ring_planes[i].opacity;
            float raw_a_faded = raw_a_physical * edge_alpha;
            
            float tau = 0.0;
            if (raw_a_physical >= 0.999) {
                tau = 100.0;
            } else if (raw_a_physical > 1e-6) {
                tau = -log(1.0 - raw_a_physical);
            }
            
            float tau_faded = 0.0;
            if (raw_a_faded >= 0.999) {
                tau_faded = 100.0;
            } else if (raw_a_faded > 1e-6) {
                tau_faded = -log(1.0 - raw_a_faded);
            }
            
            if (tau > 0.0) {
                vec3 V_dir = normalize(u_camera_pos - f_world_pos);
                vec3 N_dir = normalize(f_normal);
                float view_mu = max(1e-4, abs(dot(N_dir, V_dir)));
                
                tau /= view_mu;
                tau_faded /= view_mu;
                
                total_color += u_ring_planes[i].color * tau;
                total_scatter += u_ring_planes[i].scatter * tau;
                total_asym += u_ring_planes[i].asymmetry * tau;
                total_backscatter += u_ring_planes[i].backscatter * tau;
                total_tau += tau;
                total_faded_tau += tau_faded;
            }
        }
    }
    
    if (total_tau <= 1e-6) discard;
    
    float physical_alpha = 1.0 - exp(-total_tau);
    float faded_alpha = 1.0 - exp(-total_faded_tau);
    
    vec4 f_color = vec4(total_color / max(1e-6, total_tau), physical_alpha);
    float f_scatter = total_scatter / max(1e-6, total_tau);
    float f_asymmetry = total_asym / max(1e-6, total_tau);
    float f_backscatter = total_backscatter / max(1e-6, total_tau);

    
    vec3 V = normalize(u_camera_pos - f_world_pos);
    vec3 N = normalize(f_normal);
    float cam_side = dot(N, V);
    
    float g_rock = f_backscatter;
    float g_dust = f_asymmetry;
    
    float rock_albedo = 1.0 - f_scatter;
    float rock_reflect = rock_albedo * (f_color.a + f_color.a * f_color.a * 0.5);
    float rock_transmit = rock_albedo * f_color.a * pow(max(0.0, 1.0 - f_color.a), 1.5) * 0.5;
    
    float dust_reflect = f_scatter * f_color.a * 5.0;
    float dust_transmit = f_scatter * f_color.a * 5.0; 
    
    vec3 total_direct_illum_color = vec3(0.0);
    
    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        vec3 frag_to_star = star_pos - f_world_pos;
        float dist_to_star = length(frag_to_star);
        if (dist_to_star < 1e-5) continue;
        vec3 L = frag_to_star / dist_to_star;
        
        vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
        float sin_lat = abs(dot(L, pole_dir));
        
        vec3 eq_color = u_stars_colors[s].rgb;
        float eq_lum = u_stars_colors[s].a;
        vec3 pole_color = u_stars_pole_colors[s].rgb;
        float pole_lum = u_stars_pole_colors[s].a;
        
        vec3 star_color = mix(eq_color, pole_color, sin_lat);
        float star_lum = mix(eq_lum, pole_lum, sin_lat);
        
        float cos_theta = -dot(L, V);
        
        float denom_rock = 1.0 + g_rock * g_rock - 2.0 * g_rock * cos_theta;
        float rocky_phase = (1.0 - g_rock * g_rock) / (denom_rock * sqrt(denom_rock));
        rocky_phase = min(rocky_phase, 2.5);
        
        float denom_rock_back = 1.0 + g_rock * g_rock + 2.0 * g_rock * cos_theta;
        float rocky_phase_back = (1.0 - g_rock * g_rock) / (denom_rock_back * sqrt(denom_rock_back));
        rocky_phase_back = min(rocky_phase_back, 2.5);
        
        float denom_dust = 1.0 + g_dust * g_dust - 2.0 * g_dust * cos_theta;
        float dusty_phase = (1.0 - g_dust * g_dust) / (denom_dust * sqrt(denom_dust));
        dusty_phase *= 0.15;
        
        float sun_side = dot(N, L);
        
        float star_ang_radius = star_radius / dist_to_star;
        float sin_alpha_local = clamp(star_ang_radius, 1e-6, 1.0);
        float alpha = sin_alpha_local;
        
        float effective_sun_side = sqrt(sun_side * sun_side + 0.180126 * alpha * alpha);
        float solar_elevation = max(0.02, effective_sun_side);
        
        float v_star = clamp(sun_side / sin_alpha_local, -1.0, 1.0);
        float f_top = (v_star * sqrt(max(0.0, 1.0 - v_star*v_star)) + asin(v_star)) / 3.14159265358979 + 0.5;
        float same_side = (cam_side > 0.0) ? f_top : (1.0 - f_top);
        
        float rock_reflect_s = rock_reflect * solar_elevation * rocky_phase;
        float rock_transmit_s = rock_transmit * solar_elevation * rocky_phase;
        float backscatter_leak = rock_reflect * solar_elevation * rocky_phase_back * (1.0 - f_color.a) * 1.5;
        rock_transmit_s += backscatter_leak;
        
        float reflected_s = rock_reflect_s + dust_reflect * dusty_phase;
        float transmitted_s = rock_transmit_s + dust_transmit * dusty_phase;
        
        float direct_illum_s = mix(transmitted_s, reflected_s, same_side);
        
        // Inverse-square falloff
        if (u_hdr_enabled) {
            direct_illum_s *= star_lum / (dist_to_star * dist_to_star);
        }
        vec3 shadow_s = vec3(1.0);
        float star_radius_over_dist = star_ang_radius;
        
        for (int j = 0; j < u_num_casters; j++) {
            if (dot(shadow_s, shadow_s) < 0.001) break;
            
            if (j < 32) { if ((u_caster_mask_lo & (1u << j)) == 0u) continue; }
            else        { if ((u_caster_mask_hi & (1u << (j - 32))) == 0u) continue; }
            
            vec3 caster_pos = u_casters[j].xyz;
            float caster_r = u_casters[j].w;
            float atmo_h = u_caster_atmos[j].w;
            
            vec3 frag_to_caster = caster_pos - f_world_pos;
            float t_proj = dot(frag_to_caster, L);
            if (t_proj < 0.0) continue;
            
            float dist_sq = dot(frag_to_caster, frag_to_caster);
            if (dist_sq < caster_r * caster_r * 1.0404) continue;
            
            float dist_to_caster = sqrt(dist_sq);
            float perp_sq = max(0.0, dist_sq - t_proj * t_proj);
            
            // Bounding cone early out using maximum equatorial radius
            float max_effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
            float max_r_penumbra = max_effective_r + dist_to_caster * star_radius_over_dist;
            if (perp_sq > max_r_penumbra * max_r_penumbra) continue;
            
            float caster_r_minor = u_caster_poles_obl[j].w;
            vec3 perp_vec = frag_to_caster - t_proj * L;
            if (caster_r_minor < caster_r - 1e-5) {
                vec3 pole = u_caster_poles_obl[j].xyz;
                caster_r = get_oblate_radius(caster_r, caster_r_minor, pole, L, perp_vec);
            }
            
            float directional_star_r = star_radius;
            float star_r_minor = u_stars_poles_obl[s].w;
            if (star_r_minor < star_radius - 1e-5) {
                vec3 star_pole = u_stars_poles_obl[s].xyz;
                directional_star_r = get_oblate_radius(star_radius, star_r_minor, star_pole, L, perp_vec);
            }
            float local_star_radius_over_dist = directional_star_r / dist_to_star;
            
            float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
            float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
            if (perp_sq > r_penumbra * r_penumbra) continue;
            
            float inv_dist = 1.0 / dist_to_caster;
            float alpha = local_star_radius_over_dist;
            float beta = caster_r * inv_dist;
            float gamma = sqrt(perp_sq) * inv_dist;
            
            float penumbra_outer = alpha + beta;
            float penumbra_inner = max(0.0, beta - alpha);
            
            float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
            float occ = max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma);
            
            vec3 caster_shadow = vec3(1.0 - occ);
            
            float max_bend = u_caster_max_bend[j];
            if (max_bend > 0.0 && gamma < penumbra_outer) {
                float req_bend = beta - gamma;
                
                float optical_depth = max(0.0, req_bend);
                float atmospheric_transmission = exp(-optical_depth * 150.0);
                float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
                
                float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
                vec3 atmo_tint = u_caster_atmos[j].xyz;
                vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
                
                float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
                float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
                float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
                
                caster_shadow += deep_tint * refraction_intensity * atmo_blend;
            }
            shadow_s *= clamp(caster_shadow, 0.0, 1.0);
        }   
        
        if (u_host_planet_radius > 0.0) {
            vec3 frag_to_host = u_host_planet_pos - f_world_pos;
            float t_proj = dot(frag_to_host, L);
            
            if (t_proj > 0.0 && t_proj < dist_to_star) {
                float dist_sq = dot(frag_to_host, frag_to_host);
                float dist_to_host = sqrt(dist_sq);
                vec3 cross_vec = cross(frag_to_host, L);
                float perp_sq = dot(cross_vec, cross_vec);
                
                float host_r = u_host_planet_radius;
                float host_atmo_h = u_host_planet_atmo.w;
                float host_r_minor = u_host_planet_R_minor;
                
                // Bounding cone early out using maximum equatorial radius
                float max_effective_r = host_r + (host_atmo_h > 0.0 ? host_atmo_h * 4.0 : 0.0);
                float max_r_penumbra = max_effective_r + dist_to_host * star_radius_over_dist;
                
                if (perp_sq <= max_r_penumbra * max_r_penumbra) {
                    vec3 perp_vec = frag_to_host - t_proj * L;
                    if (host_r_minor < host_r - 1e-5) {
                        vec3 host_pole = u_host_planet_pole_obl.xyz;
                        host_r = get_oblate_radius(host_r, host_r_minor, host_pole, L, perp_vec);
                    }
                    
                    float directional_star_r = star_radius;
                    float star_r_minor = u_stars_poles_obl[s].w;
                    if (star_r_minor < star_radius - 1e-5) {
                        vec3 star_pole = u_stars_poles_obl[s].xyz;
                        directional_star_r = get_oblate_radius(star_radius, star_r_minor, star_pole, L, perp_vec);
                    }
                    float local_star_radius_over_dist = directional_star_r / dist_to_star;
                    
                    float effective_r = host_r + (host_atmo_h > 0.0 ? host_atmo_h * 4.0 : 0.0);
                    float r_penumbra = effective_r + dist_to_host * local_star_radius_over_dist;
                    
                    if (perp_sq <= r_penumbra * r_penumbra) {
                        float inv_dist = 1.0 / dist_to_host;
                        float alpha = local_star_radius_over_dist;
                        float beta = host_r * inv_dist;
                        float gamma = sqrt(perp_sq) * inv_dist;
                        
                        float penumbra_outer = alpha + beta;
                        float penumbra_inner = max(0.0, beta - alpha);
                        
                        float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
                        float occ = max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma);
                        
                        vec3 host_shadow = vec3(1.0 - occ);
                        
                        if (host_atmo_h > 0.0 && gamma < penumbra_outer) {
                            float max_bend = clamp(2.0 * 0.00029 * sqrt(3.14159265359 * host_r / max(1e-6, host_atmo_h * 2.0)), 0.001, 0.05);
                            float req_bend = beta - gamma;
                            
                            float optical_depth = max(0.0, req_bend);
                            float atmospheric_transmission = exp(-optical_depth * 150.0);
                            float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
                            
                            float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
                            vec3 atmo_tint = u_host_planet_atmo.xyz;
                            vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
                            
                            float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
                            float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
                            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
                            
                            host_shadow += deep_tint * refraction_intensity * atmo_blend;
                        }
                        shadow_s *= clamp(host_shadow, 0.0, 1.0);
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
            vec3 host_to_star = star_pos - u_host_planet_pos;
            float dist_host_star = length(host_to_star);
            vec3 L_host = host_to_star / max(dist_host_star, 1e-6);
            
            vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
            float sin_lat = abs(dot(L_host, pole_dir));
            
            vec3 eq_color = u_stars_colors[s].rgb;
            float eq_lum = u_stars_colors[s].a;
            vec3 pole_color = u_stars_pole_colors[s].rgb;
            float pole_lum = u_stars_pole_colors[s].a;
            
            vec3 star_color = mix(eq_color, pole_color, sin_lat);
            float star_lum = mix(eq_lum, pole_lum, sin_lat);
            
            float irradiance = u_hdr_enabled ? (star_lum / max(dist_host_star * dist_host_star, 1e-8)) : 1.0;
            
            float planet_phase = max(0.0, dot(L_host, -dir_to_host));
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
    
    // Aerial perspective bypassed for cleaner rendering
    float trans = 1.0;
    vec3 ap_rgb = vec3(0.0);
    
    float fade_scale = faded_alpha / max(1e-6, physical_alpha);
    out_color = vec4(final_color * trans * fade_scale + ap_rgb * faded_alpha, faded_alpha);
}
"""

hz_vertex_shader = """
#version 460 core
in vec3 in_position;
uniform mat4 projection;
uniform mat4 view;
uniform vec3 u_body_offset;
uniform float u_inner_r;
uniform float u_outer_r;
uniform float u_depth_C;
uniform float u_far;

out float f_clip_z;
out float f_radius_pct;

void main() {
    float r = mix(u_inner_r, u_outer_r, in_position.y);
    vec3 world_pos = vec3(in_position.x * r, 0.0, in_position.z * r) + u_body_offset;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    gl_Position.z = (log2(max(1e-6, u_depth_C * gl_Position.w + 1.0)) / log2(u_depth_C * u_far + 1.0) * 2.0 - 1.0) * gl_Position.w;
    f_clip_z = gl_Position.w;
    f_radius_pct = in_position.y;
}
"""

hz_fragment_shader = """
#version 460 core
in float f_clip_z;
in float f_radius_pct;
uniform vec4 u_color;
uniform float u_far;
uniform float u_depth_C;
out vec4 out_color;

void main() {
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    float alpha = sin(f_radius_pct * 3.14159265);
    out_color = vec4(u_color.rgb, u_color.a * alpha);
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
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
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
    vec3 world_pos = in_position * (u_atmo_radius_au * 1.03) + u_body_offset;
    f_world_pos = world_pos;
    f_local_pos = in_position * 1.03;
    gl_Position = projection * view * vec4(world_pos, 1.0);
    f_clip_z = gl_Position.w;
}
"""

sky_view_lut_fragment_shader = """
#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define MAX_RING_PLANES 16
#define PI 3.14159265358979

in vec2 f_uv;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
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
uint u_caster_mask_lo;
uint u_caster_mask_hi;
uint u_ring_mask;

uniform vec3 u_camera_pos;

layout(std430, binding = 8) buffer AtmoData {
    vec3  u_body_offset;
    float u_atmo_radius_au;
    vec3  u_beta_rayleigh;
    float u_h_rayleigh;
    vec3  u_beta_mie;
    float u_h_mie;
    vec3  u_beta_abs_mixed;
    float u_mie_g;
    vec3  u_beta_abs_layered;
    float u_sun_intensity;
    float u_planet_radius_km;
    float u_atmo_radius_km;
    float u_au_to_km;
    int   u_num_samples;
    vec4  u_pole_obl;
    int   u_num_active_casters;
    bool  u_atmo_adaptive_steps;
    int   u_body_idx;
    float _pad_meta;
    vec4  u_active_casters[8];
    vec4  u_active_caster_poles_obl[8];
    float u_active_caster_R_minor[8];
    vec4  u_active_caster_atmos[8];
    float u_active_max_bend[8];
};

uniform sampler2D u_ring_gradients;
uniform int u_num_ring_planes;
uniform int u_atmo_quality;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec3 u_ring_params[MAX_RING_PLANES]; 
uniform int u_atmo_clip_mode; 

uniform uint u_ring_coplanar_mask[16];
uniform sampler2D u_eclipse_lut; // Kept to avoid uniform bound errors
uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;

layout(location = 0, index = 0) out vec4 out_color;
layout(location = 0, index = 1) out vec4 out_transmittance;

// g_local_rings removed to prevent local memory array spilling

vec3 toSphericalSpace(vec3 p, vec4 pole_scale) {
    float f_scale = pole_scale.w;
    if (f_scale <= 1.00001) return p;
    vec3 pole = pole_scale.xyz;
    float h = dot(p, pole);
    return p + (h * (f_scale - 1.0)) * pole;
}

float get_oblate_radius(float r_eq, float r_minor, vec3 pole, vec3 L, vec3 perp_vec) {
    if (r_minor < 1e-6 || r_eq < 1e-6) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / r_minor) * (y / r_minor));
    if (denom < 1e-6) return r_eq;
    return perp_len / denom;
}

vec3 compute_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, vec3 planet_center_render, float star_radius, float star_obl, vec3 star_pole) {
    vec3 shadow = vec3(1.0);
    float base_star_radius_over_dist = star_radius / max(dist_to_star, 1e-6);
    
    for (int i = 0; i < u_num_active_casters; i++) {
        vec3 caster_pos = u_active_casters[i].xyz;
        float caster_r = u_active_casters[i].w;
        float atmo_h = u_active_caster_atmos[i].w;
        
        vec3 s_to_c = caster_pos - eval_render_pos;
        float t_proj = dot(s_to_c, L_dir);
        if (t_proj < 0.0) continue;
        
        float dist_sq = dot(s_to_c, s_to_c);
        if (dist_sq < caster_r * caster_r * 1.0404) continue;
        
        float dist_to_caster = sqrt(dist_sq);
        float perp_sq = max(0.0, dist_sq - t_proj * t_proj);
        
        vec3 perp_vec = s_to_c - t_proj * L_dir;
        float caster_r_minor = u_active_caster_R_minor[i];
        if (caster_r_minor < caster_r - 1e-5) {
            vec3 pole = u_active_caster_poles_obl[i].xyz;
            caster_r = get_oblate_radius(caster_r, caster_r_minor, pole, L_dir, perp_vec);
        }
        
        float directional_star_r = star_radius;
        if (star_obl < star_radius - 1e-5) {
            directional_star_r = get_oblate_radius(star_radius, star_obl, star_pole, L_dir, perp_vec);
        }
        float local_star_radius_over_dist = directional_star_r / max(dist_to_star, 1e-6);
        
        float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * 4.0 : 0.0);
        float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
        if (perp_sq > r_penumbra * r_penumbra) continue;
        
        float inv_dist = 1.0 / dist_to_caster;
        float alpha = local_star_radius_over_dist;
        float beta = caster_r * inv_dist;
        float gamma = sqrt(perp_sq) * inv_dist;
        
        float penumbra_outer = alpha + beta;
        float penumbra_inner = max(0.0, beta - alpha);
        
        float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
        float occ = max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma);
        
        vec3 caster_shadow = vec3(1.0 - occ);
        
        float max_bend = u_active_max_bend[i];
        if (max_bend > 0.0 && gamma < penumbra_outer) {
            float req_bend = beta - gamma;
            
            float optical_depth = max(0.0, req_bend);
            float atmospheric_transmission = exp(-optical_depth * 150.0);
            float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
            
            float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
            vec3 atmo_tint = u_active_caster_atmos[i].xyz;
            vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
            
            float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
            float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            
            caster_shadow += deep_tint * refraction_intensity * atmo_blend;
        }
        shadow *= clamp(caster_shadow, 0.0, 1.0);
    }
    
    uint processed_mask = 0u;
    for (int k = 0; k < u_num_ring_planes; k++) {
        if ((u_ring_mask & (1u << k)) == 0u) continue;
        if ((processed_mask & (1u << k)) != 0u) continue;
        
        vec3 plane_center = u_ring_center[k];
        vec3 plane_normal = u_ring_normal[k];
        
        uint coplanar_mask = u_ring_coplanar_mask[k];
        processed_mask |= coplanar_mask;
        
        float denom = dot(L_dir, plane_normal);
        if (abs(denom) < 1e-8) continue;
        
        float t_ring = dot(plane_center - eval_render_pos, plane_normal) / denom;
        if (t_ring <= 0.0 || t_ring >= dist_to_star) continue;
        
        vec3 hit = eval_render_pos + t_ring * L_dir;
        vec3 vec_radial = hit - plane_center;
        float d = length(vec_radial);
        
        float r_star_proj = star_radius * t_ring / max(dist_to_star, 1e-6);
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
        
        for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
            if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
            float inner_r = u_ring_params[ring_idx].x;
            float outer_r = u_ring_params[ring_idx].y;
            
            float overlap_min = max(inner_r, d - R_eff);
            float overlap_max = min(outer_r, d + R_eff);
            
            if (overlap_min >= overlap_max) continue;
            
            float v_min = clamp((overlap_min - d) / max(1e-9, R_eff), -1.0, 1.0);
            float v_max = clamp((overlap_max - d) / max(1e-9, R_eff), -1.0, 1.0);
            
            float f_max = (v_max * sqrt(max(0.0, 1.0 - v_max*v_max)) + asin(v_max)) / PI + 0.5;
            float f_min = (v_min * sqrt(max(0.0, 1.0 - v_min*v_min)) + asin(v_min)) / PI + 0.5;
            float fraction = max(0.0, f_max - f_min);
            
            float p_mid = ((overlap_min + overlap_max) * 0.5 - inner_r) / max(1e-6, outer_r - inner_r);
            float alpha_mult = textureLod(u_ring_gradients, vec2(p_mid, (float(ring_idx) + 0.5) / 16.0), 0.0).r;
            float opacity = u_ring_params[ring_idx].z;
            
            float tau = -log(max(1e-6, 1.0 - opacity * alpha_mult));
            float light_mu = max(1e-4, abs(dot(normalize(u_ring_normal[ring_idx]), L_dir)));
            plane_occlusion += fraction * (1.0 - exp(-tau / light_mu));
        }
        
        shadow *= vec3(1.0 - min(plane_occlusion, 1.0));
    }
    return shadow;
}

vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float a = dot(dir, dir);
    float b = dot(origin, dir);
    
    // Improved precision for large distances
    vec3 p = origin - (b / a) * dir;
    float p2 = dot(p, p);
    float r2 = radius * radius;
    
    if (p2 > r2) return vec2(1e10, -1e10);
    
    float d = sqrt((r2 - p2) / a);
    float t_closest = -b / a;
    
    return vec2(t_closest - d, t_closest + d);
}

vec3 get_transmittance(float r, float cos_theta) {
    float h_norm = clamp((r - u_planet_radius_km) / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float v = sqrt(h_norm);
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

vec3 get_transmittance_precomputed(float v, float cos_theta) {
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

void main() {
    u_ring_mask = floatBitsToUint(instances[u_body_idx * 6 + 3].w);
    
    // g_local_rings removed to prevent local memory array spilling

    vec3 planet_center_render = u_body_offset;
    vec3 cam_local_au = u_camera_pos - planet_center_render;
    vec3 cam_local = cam_local_au * u_au_to_km;

    // Build local coordinate frame relative to the local vertical (zenith) at camera position
    vec3 up = length(cam_local) > 1e-5 ? normalize(cam_local) : vec3(0.0, 1.0, 0.0);
    vec3 east = cross(vec3(0.0, 1.0, 0.0), up);
    if (length(east) < 1e-4) {
        east = cross(up, vec3(0.0, 0.0, 1.0));
    }
    east = normalize(east);
    vec3 north = cross(up, east);

    float azimuth = f_uv.x * 2.0 * PI;
    float y = f_uv.y * 2.0 - 1.0;
    float elevation = sign(y) * y * y * (PI / 2.0);
    
    vec3 ray_dir_local = vec3(cos(elevation)*sin(azimuth), sin(elevation), cos(elevation)*cos(azimuth));
    vec3 ray_dir = ray_dir_local.x * east + ray_dir_local.y * up + ray_dir_local.z * north;

    vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
    vec3 cam_local_sph = toSphericalSpace(cam_local, u_pole_obl);

    vec2 s_atmo = raySphereIntersect(cam_local_sph, ray_dir_sph, u_atmo_radius_km);
    if (s_atmo.x > s_atmo.y) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        out_transmittance = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    vec2 s_planet = raySphereIntersect(cam_local_sph, ray_dir_sph, u_planet_radius_km);

    float s_start = max(0.0, s_atmo.x);
    float s_end = s_atmo.y;
    // AP_VOLUME_HOOK
    
    if (s_planet.x > 0.0 && s_planet.x < s_end) {
        s_end = s_planet.x;
    }
    
    // We also need frag_local for shadow calculations later
    // Dummy intersection point far away to retain ring logic cleanly
    vec3 frag_local = cam_local;

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

    for (int i = 0; i < u_num_active_casters; i++) {
        vec3 caster_pos = u_active_casters[i].xyz;
        float caster_r = u_active_casters[i].w * u_au_to_km;
        vec3 caster_local = (caster_pos - planet_center_render) * u_au_to_km;
        
        if (length(caster_local) < 1.0) continue;
        
        vec4 pole_obl = u_active_caster_poles_obl[i];
        vec3 origin_c = toSphericalSpace(frag_local - caster_local, pole_obl);
        vec3 dir_c = toSphericalSpace(ray_dir, pole_obl);
        
        vec2 s_c = raySphereIntersect(origin_c, dir_c, caster_r);
        if (s_c.x > 0.0 && s_c.x < s_end) {
            s_end = s_c.x;
        }
    }

    if (s_start >= s_end) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        out_transmittance = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M = u_beta_mie * 1000.0;
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;

    int steps = u_num_samples;
    if (u_atmo_adaptive_steps) {
        float s_closest = clamp(-dot(cam_local_sph, ray_dir_sph), s_start, s_end);
        float min_altitude = length(cam_local_sph + s_closest * ray_dir_sph) - u_planet_radius_km;
        float step_factor = mix(0.25, 1.0, clamp(exp(-max(0.0, min_altitude) / max(1e-3, u_h_rayleigh * 2.0)), 0.0, 1.0));
        steps = int(clamp(float(u_num_samples) * step_factor, min(4.0, float(u_num_samples)), float(u_num_samples)));
    }
    float step_size = (s_end - s_start) / float(steps);
    vec3 scattered = vec3(0.0);
    vec3 final_transmittance = vec3(1.0);

    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        
        vec3 frag_to_star = star_pos - planet_center_render;
        float dist_to_star_au = length(frag_to_star);
        vec3 L = frag_to_star / max(dist_to_star_au, 1e-6);
        
        vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
        float sin_lat = abs(dot(L, pole_dir));
        
        vec3 eq_color = u_stars_colors[s].rgb;
        float eq_lum = u_stars_colors[s].a;
        vec3 pole_color = u_stars_pole_colors[s].rgb;
        float pole_lum = u_stars_pole_colors[s].a;
        
        vec3 star_color = mix(eq_color, pole_color, sin_lat);
        float star_lum = mix(eq_lum, pole_lum, sin_lat);
        float star_radius_au = star_radius / u_au_to_km;
        float sin_star = star_radius_au / max(dist_to_star_au, star_radius_au + 1e-6);

        vec3 sun_pos_local = (star_pos - planet_center_render) * u_au_to_km;
        // sun_dir computed per-sample now

        vec3 global_eclipse_shadow = vec3(1.0);
        vec3 end_eclipse_shadow = vec3(1.0);
        bool skip_volumetric_shadow = false;
        
        vec4 ring_s1_out = vec4(1e9);
        vec4 ring_s2_out = vec4(-1e9);
        vec4 ring_s1_in = vec4(1e9);
        vec4 ring_s2_in = vec4(-1e9);
        vec4 ring_inner = vec4(0.0);
        vec4 ring_outer = vec4(1.0);
        vec4 ring_opac = vec4(0.0);
        vec4 ring_v_coord = vec4(0.0);
        vec4 ring_R_eff = vec4(0.0);
        vec3 ring_A_prime = vec3(0.0);
        vec3 ring_B = vec3(0.0);

        float s_mid = (s_start + s_end) * 0.5;
        vec3 mid_pos = frag_local + s_mid * ray_dir;
        vec3 mid_render = mid_pos / u_au_to_km + planet_center_render;
        vec3 mid_to_star = star_pos - mid_render;
        float dist_mid_star = length(mid_to_star);
        vec3 L_mid = mid_to_star / max(dist_mid_star, 1e-6);
        float sr_start = star_radius / max(dist_mid_star, 1e-6);
        vec3 O = frag_local / u_au_to_km + planet_center_render;
        vec3 V = ray_dir / u_au_to_km;

        if (u_atmo_quality > 0) { 
            if (u_atmo_quality == 1) {
                global_eclipse_shadow = compute_shadow(mid_render, L_mid, dist_mid_star, planet_center_render, star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz);
                end_eclipse_shadow = global_eclipse_shadow;
            }

            // High mode interpolation and early-out removed because they miss thin ring shadows.
            if (u_atmo_quality == 2) {
                int ring_count = 0;
                uint processed_mask = 0u;
                for (int k = 0; k < u_num_ring_planes; k++) {
                    if ((u_ring_mask & (1u << k)) == 0u) continue;
                    if ((processed_mask & (1u << k)) != 0u) continue;
                    
                    vec3 N = u_ring_normal[k];
                    vec3 C_km = u_ring_center[k] * u_au_to_km;
                    vec3 O_km = frag_local + planet_center_render * u_au_to_km;
                    
                    float denom = dot(L_mid, N);
                    if (abs(denom) < 1e-8) continue;
                    
                    float ra_km = dot(C_km - O_km, N) / denom;
                    float rb_km = -dot(ray_dir, N) / denom;
                    
                    vec3 A_km = O_km + ra_km * L_mid;
                    vec3 B_km = ray_dir + rb_km * L_mid;
                    vec3 A_prime_km = A_km - C_km;
                    
                    float qa = dot(B_km, B_km);
                    float qb = 2.0 * dot(A_prime_km, B_km);
                    float qc = dot(A_prime_km, A_prime_km);
                    
                    float r_star_proj_km = abs((star_radius * u_au_to_km) * ra_km / max(dist_mid_star * u_au_to_km, 1e-6));
                    float R_eff_km = r_star_proj_km;
                    vec3 L_plane = L_mid - denom * N;
                    float L_plane_len = length(L_plane);
                    if (L_plane_len > 1e-5 && qa > 1e-8) {
                        vec3 L_proj = L_plane / L_plane_len;
                        vec3 T_vec = normalize(cross(N, L_mid));
                        float s_mid_ring = (-qb) / (2.0 * qa);
                        vec3 dir_radial = normalize(A_prime_km + s_mid_ring * B_km);
                        float cos_theta = dot(dir_radial, L_proj);
                        float sin_theta = dot(dir_radial, T_vec);
                        R_eff_km = r_star_proj_km * sqrt( pow(cos_theta / max(1e-6, abs(denom)), 2.0) + pow(sin_theta, 2.0) );
                    }
                    
                    uint coplanar_mask = u_ring_coplanar_mask[k];
                    processed_mask |= coplanar_mask;
                    for (int ring_idx = k; ring_idx < u_num_ring_planes; ring_idx++) {
                        if ((coplanar_mask & (1u << ring_idx)) == 0u) continue;
                        if (ring_count >= 4) continue;
                            
                            float inner_r_km = u_ring_params[ring_idx].x * u_au_to_km;
                            float outer_r_km = u_ring_params[ring_idx].y * u_au_to_km;
                            float opacity = u_ring_params[ring_idx].z;
                            float base_tau = -log(max(1e-6, 1.0 - opacity));
                            float scaled_tau = base_tau / max(1e-4, abs(denom));
                                
                            float ring_s_valid_min = -1e9;
                            float ring_s_valid_max = 1e9;
                            if (abs(rb_km) < 1e-8) {
                                if (ra_km <= 0.0) { ring_s_valid_min = 1e9; ring_s_valid_max = -1e9; }
                            } else {
                                float s_cross = -ra_km / rb_km;
                                if (rb_km < 0.0) {
                                    ring_s_valid_max = s_cross;
                                } else {
                                    ring_s_valid_min = s_cross;
                                }
                            }
                            if (ring_s_valid_min > ring_s_valid_max) continue;
                            
                                if (ring_count == 0) { 
                                    ring_s1_out.x = ring_s_valid_min; ring_s2_out.x = ring_s_valid_max;
                                    ring_inner.x = inner_r_km; ring_outer.x = outer_r_km; ring_opac.x = scaled_tau; ring_v_coord.x = (float(ring_idx) + 0.5) / 16.0;
                                    ring_A_prime = A_prime_km; ring_B = B_km; ring_R_eff.x = R_eff_km;
                                }
                                else if (ring_count == 1) { 
                                    ring_s1_out.y = ring_s_valid_min; ring_s2_out.y = ring_s_valid_max;
                                    ring_inner.y = inner_r_km; ring_outer.y = outer_r_km; ring_opac.y = scaled_tau; ring_v_coord.y = (float(ring_idx) + 0.5) / 16.0;
                                    ring_R_eff.y = R_eff_km;
                                }
                                else if (ring_count == 2) { 
                                    ring_s1_out.z = ring_s_valid_min; ring_s2_out.z = ring_s_valid_max;
                                    ring_inner.z = inner_r_km; ring_outer.z = outer_r_km; ring_opac.z = scaled_tau; ring_v_coord.z = (float(ring_idx) + 0.5) / 16.0;
                                    ring_R_eff.z = R_eff_km;
                                }
                                else if (ring_count == 3) { 
                                    ring_s1_out.w = ring_s_valid_min; ring_s2_out.w = ring_s_valid_max;
                                    ring_inner.w = inner_r_km; ring_outer.w = outer_r_km; ring_opac.w = scaled_tau; ring_v_coord.w = (float(ring_idx) + 0.5) / 16.0;
                                    ring_R_eff.w = R_eff_km;
                                }
                                
                                ring_count++;
                        }
                    }
                
                vec3 shadow_start = vec3(1.0);
                vec3 shadow_end = vec3(1.0);
                
                vec3 start_render = planet_center_render + (O + s_start * V) / u_au_to_km;
                vec3 end_render = planet_center_render + (O + s_end * V) / u_au_to_km;
                vec3 L_start = L_mid;
                float sr_start = star_radius / max(dist_mid_star, 1e-6);
                vec3 L_end = L_mid;
                float sr_end = sr_start;
                
                for (int c = 0; c < u_num_active_casters; c++) {
                    vec3 pos = u_active_casters[c].xyz;
                    float rad = u_active_casters[c].w;
                    float atmo = u_active_caster_atmos[c].w;
                    float caster_r_minor = u_active_caster_R_minor[c];
                    vec3 pole = u_active_caster_poles_obl[c].xyz;
                    vec3 atmo_tint = u_active_caster_atmos[c].xyz;
                    
                    vec3 s2c = pos - start_render;
                    float t = dot(s2c, L_start);
                    if (t > 0.0) {
                        float dist_sq = dot(s2c, s2c);
                        if (dist_sq > rad * rad * 1.0404) {
                            float dist = sqrt(dist_sq);
                            vec3 cross_vec = cross(s2c, L_start);
                            float p2 = dot(cross_vec, cross_vec);
                            float r = rad;
                            if (caster_r_minor < rad - 1e-5) r = get_oblate_radius(r, caster_r_minor, pole, L_start, s2c - t * L_start);
                            float eff_r = r + (atmo > 0.0 ? atmo * 4.0 : 0.0);
                            float rp = eff_r + dist * sr_start;
                            if (p2 < rp * rp) {
                                float inv_d = 1.0 / dist;
                                float alpha = sr_start; float beta = r * inv_d; float gamma = sqrt(p2) * inv_d;
                                float po = alpha + beta; float pi = max(0.0, beta - alpha);
                                float occ = min(1.0, (beta*beta)/max(1e-9, alpha*alpha)) * smoothstep(po, pi, gamma);
                                vec3 sh = vec3(1.0 - occ);
                                float max_bend = u_active_max_bend[c];
                                if (max_bend > 0.0 && gamma < po) {
                                    float req_bend = beta - gamma;
                                    
                                    float optical_depth = max(0.0, req_bend);
                                    float atmospheric_transmission = exp(-optical_depth * 150.0);
                                    float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
                                    
                                    float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
                                    vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
                                    
                                    float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
                                    float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
                                    float atmo_blend = smoothstep(po, pi, gamma);
                                    
                                    sh += deep_tint * refraction_intensity * atmo_blend;
                                }
                                shadow_start *= clamp(sh, 0.0, 1.0);
                            }
                        }
                    }
                    
                    s2c = pos - end_render;
                    t = dot(s2c, L_end);
                    if (t > 0.0) {
                        float dist_sq = dot(s2c, s2c);
                        if (dist_sq > rad * rad * 1.0404) {
                            float dist = sqrt(dist_sq);
                            vec3 cross_vec = cross(s2c, L_end);
                            float p2 = dot(cross_vec, cross_vec);
                            float r = rad;
                            if (caster_r_minor < rad - 1e-5) r = get_oblate_radius(r, caster_r_minor, pole, L_end, s2c - t * L_end);
                            float eff_r = r + (atmo > 0.0 ? atmo * 4.0 : 0.0);
                            float rp = eff_r + dist * sr_end;
                            if (p2 < rp * rp) {
                                float inv_d = 1.0 / dist;
                                float alpha = sr_end; float beta = r * inv_d; float gamma = sqrt(p2) * inv_d;
                                float po = alpha + beta; float pi = max(0.0, beta - alpha);
                                float occ = min(1.0, (beta*beta)/max(1e-9, alpha*alpha)) * smoothstep(po, pi, gamma);
                                vec3 sh = vec3(1.0 - occ);
                                float max_bend = u_active_max_bend[c];
                                if (max_bend > 0.0 && gamma < po) {
                                    float req_bend = beta - gamma;
                                    
                                    float optical_depth = max(0.0, req_bend);
                                    float atmospheric_transmission = exp(-optical_depth * 150.0);
                                    float transmission_mask = 1.0 - smoothstep(max_bend - alpha, max_bend + alpha, req_bend);
                                    
                                    float rayleigh_depth = clamp(req_bend / max(1e-6, max_bend), 0.0, 1.0);
                                    vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (rayleigh_depth * 2.0));
                                    
                                    float distance_falloff = min(beta, 0.05) * min(beta, 0.05);
                                    float refraction_intensity = atmospheric_transmission * transmission_mask * 1000.0 * distance_falloff;
                                    float atmo_blend = smoothstep(po, pi, gamma);
                                    
                                    sh += deep_tint * refraction_intensity * atmo_blend;
                                }
                                shadow_end *= clamp(sh, 0.0, 1.0);
                            }
                        }
                    }
                }
                global_eclipse_shadow = shadow_start;
                end_eclipse_shadow = shadow_end;
                
                if (shadow_start.r > 0.999 && shadow_end.r > 0.999 && ring_count == 0) skip_volumetric_shadow = true;
                else if (shadow_start.r < 0.001 && shadow_end.r < 0.001 && ring_count == 0) skip_volumetric_shadow = true;
            }
        }
        
        // Precompute star direction invariants
        vec3 sun_dir_local = normalize(sun_pos_local - mid_pos);
        vec3 sun_dir_sph_const = normalize(toSphericalSpace(sun_dir_local, u_pole_obl));

        // OPTIMIZATION 1: Fast limb math invariants pre-calculated
        float alpha_sun_local = asin(clamp(sin_star, 0.0, 0.9999));
        float cos_sun = cos(alpha_sun_local);
        float sin_sun = sin_star;
        
        vec3 step_dir_sph = ray_dir_sph * step_size;
        vec3 current_pos_sph = cam_local_sph + (s_start + 0.5 * step_size) * ray_dir_sph;

        vec3 total_rayleigh = vec3(0.0);
        vec3 total_mie = vec3(0.0);
        vec3 total_ms = vec3(0.0);
        vec3 current_transmittance = vec3(1.0);

        float current_s = s_start + 0.5 * step_size;
        float t_lerp = 0.5 / float(steps);
        float t_step = 1.0 / float(steps);

        float inv_atmo_thickness = 1.0 / max(1e-4, u_atmo_radius_km - u_planet_radius_km);

        for (int i = 0; i < steps; i++) {
            float sample_len = length(current_pos_sph);
            float altitude = sample_len - u_planet_radius_km;

            float rho_R = exp(-altitude / u_h_rayleigh);
            float rho_M = exp(-altitude / u_h_mie);
            float rho_O = exp(-pow((altitude - 25.0) / 8.0, 2.0));

            vec3 step_extinction = beta_R * rho_R + beta_M * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;
            vec3 step_transmittance = exp(-step_extinction * step_size);
            vec3 int_factor = (vec3(1.0) - step_transmittance) / max(step_extinction, 1e-6);

            vec3 sun_dir_sph = sun_dir_sph_const;
            if (u_atmo_quality == 3) {
                // In ultra mode, compute the exact star direction per-sample for maximum precision in close ranges
                vec3 current_pos_local = cam_local + current_s * ray_dir;
                vec3 s_dir_loc = normalize(sun_pos_local - current_pos_local);
                sun_dir_sph = normalize(toSphericalSpace(s_dir_loc, u_pole_obl));
            }

            float light_cos_theta = dot(current_pos_sph, sun_dir_sph) / sample_len;
            float h_norm = clamp(altitude * inv_atmo_thickness, 0.0, 1.0);

            float sin_planet = u_planet_radius_km / max(sample_len, u_planet_radius_km + 0.01);
            float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));
            
            float cos_outer = cos_planet * cos_sun - sin_planet * sin_sun;
            float cos_inner = cos_planet * cos_sun + sin_planet * sin_sun;

            float neg_light_cos = -light_cos_theta;
            float vis_fraction = 1.0;

            if (neg_light_cos > cos_inner) { 
                vis_fraction = 0.0;
            } else if (neg_light_cos > cos_outer) { 
                if (u_atmo_quality == 3) {
                    // Exact mathematical formulation for Ultra quality
                    float alpha_planet = asin(clamp(sin_planet, 0.0, 0.9999));
                    float separation = acos(clamp(neg_light_cos, -1.0, 1.0));
                    float x_vis = (separation - alpha_planet) / max(alpha_sun_local, 1e-7);
                    vis_fraction = (acos(clamp(-x_vis, -1.0, 1.0)) + x_vis * sqrt(max(0.0, 1.0 - x_vis * x_vis))) / PI;
                } else {
                    // Optimized linear cosine-space approximation with smoothstep for High and below
                    float x_vis = (cos_planet * cos_sun - neg_light_cos) / max(1e-7, sin_planet * sin_sun);
                    vis_fraction = smoothstep(-1.0, 1.0, x_vis);
                }
            }

            float disc_top_cos = min(light_cos_theta + sin_star, 1.0);
            float disc_bot_cos = max(light_cos_theta - sin_star, -cos_planet);
            float effective_cos = (disc_top_cos + disc_bot_cos) * 0.5;

            float v_norm = sqrt(h_norm);
            vec3 transmittance_to_sun = get_transmittance_precomputed(v_norm, effective_cos);

            vec3 sample_shadow = global_eclipse_shadow;
            if (u_atmo_quality == 2 && !skip_volumetric_shadow) { 
                sample_shadow = mix(global_eclipse_shadow, end_eclipse_shadow, t_lerp);
                
                float d = length(ring_A_prime + current_s * ring_B);
                vec4 s_vec = vec4(current_s);
                vec4 mask = step(ring_s1_out, s_vec) * step(s_vec, ring_s2_out);
                
                if (any(greaterThan(mask, vec4(0.0)))) {
                    vec4 o_min = max(ring_inner, vec4(d) - ring_R_eff);
                    vec4 o_max = min(ring_outer, vec4(d) + ring_R_eff);
                    vec4 valid = step(o_min, o_max) * mask;
                    
                    if (any(greaterThan(valid, vec4(0.0)))) {
                        vec4 r_eff_inv = 1.0 / max(vec4(1e-9), ring_R_eff);
                        vec4 v_min = clamp((o_min - vec4(d)) * r_eff_inv, -1.0, 1.0);
                        vec4 v_max = clamp((o_max - vec4(d)) * r_eff_inv, -1.0, 1.0);
                        vec4 frac = max(vec4(0.0), smoothstep(-1.0, 1.0, v_max) - smoothstep(-1.0, 1.0, v_min));
                        vec4 p_mid = clamp(((o_min + o_max) * 0.5 - ring_inner) / max(vec4(1e-6), ring_outer - ring_inner), 0.0, 1.0);
                        
                        vec4 sh_mult = vec4(1.0);
                        if (valid.x > 0.0) sh_mult.x = 1.0 - frac.x * (1.0 - exp(-ring_opac.x * textureLod(u_ring_gradients, vec2(p_mid.x, ring_v_coord.x), 0.0).r));
                        if (valid.y > 0.0) sh_mult.y = 1.0 - frac.y * (1.0 - exp(-ring_opac.y * textureLod(u_ring_gradients, vec2(p_mid.y, ring_v_coord.y), 0.0).r));
                        if (valid.z > 0.0) sh_mult.z = 1.0 - frac.z * (1.0 - exp(-ring_opac.z * textureLod(u_ring_gradients, vec2(p_mid.z, ring_v_coord.z), 0.0).r));
                        if (valid.w > 0.0) sh_mult.w = 1.0 - frac.w * (1.0 - exp(-ring_opac.w * textureLod(u_ring_gradients, vec2(p_mid.w, ring_v_coord.w), 0.0).r));
                        
                        sample_shadow *= sh_mult.x * sh_mult.y * sh_mult.z * sh_mult.w;
                    }
                }
            } else if (u_atmo_quality == 3) {
                if (u_num_active_casters > 0 || u_ring_mask != 0u) {
                    vec3 sample_pos_local = frag_local + current_s * ray_dir;
                    vec3 sample_render = sample_pos_local / u_au_to_km + planet_center_render;
                    vec3 sample_to_star = star_pos - sample_render;
                    float dist_sample_star = length(sample_to_star);
                    sample_shadow = compute_shadow(sample_render, sample_to_star / max(dist_sample_star, 1e-6), dist_sample_star, planet_center_render, star_radius, u_stars_poles_obl[s].w, u_stars_poles_obl[s].xyz);
                } else {
                    sample_shadow = global_eclipse_shadow;
                }
            }

            vec3 sample_attenuation = current_transmittance * transmittance_to_sun * sample_shadow * vis_fraction;

            total_rayleigh += rho_R * sample_attenuation * int_factor;
            total_mie      += rho_M * sample_attenuation * int_factor;
            
            float sun_cos_zenith = light_cos_theta;
            float ms_u = 0.5 + 0.5 * sign(sun_cos_zenith) * sqrt(abs(sun_cos_zenith));
            float ms_v = v_norm;
            vec2 ms_uv = vec2(ms_u, ms_v);
            vec3 psi = textureLod(u_multi_scatter_lut, ms_uv, 0.0).rgb;
            
            vec3 ms_shadow = sample_shadow;
            total_ms += (beta_R * rho_R + beta_M * rho_M) * psi * current_transmittance * ms_shadow * int_factor;

            current_transmittance *= step_transmittance;

            current_pos_sph += step_dir_sph;
            current_s += step_size;
            t_lerp += t_step;
        }

        float cos_theta = dot(ray_dir, L_mid);
        float phase_R = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);

        float g = u_mie_g;
        float g2 = g * g;
        float phase_M = (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta))
                      / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));

        float irradiance = u_hdr_enabled ? (star_lum / max(dist_to_star_au * dist_to_star_au, 1e-8)) : 1.0;
        
        scattered += star_color * u_sun_intensity * irradiance * (
            phase_R * beta_R * total_rayleigh +
            phase_M * beta_M * total_mie +
            total_ms
        );

        final_transmittance = current_transmittance;
    }

    vec3 transmittance = final_transmittance;

    if (u_hdr_enabled) {
        scattered *= u_exposure;
    }

    out_color = vec4(scattered, 1.0);
    out_transmittance = vec4(transmittance, 1.0);
}
"""

sky_view_lut_pass_fragment_shader = sky_view_lut_fragment_shader.replace(
    "layout(location = 0, index = 0) out vec4 out_color;",
    "layout(location = 0) out vec4 out_color;"
).replace(
    "layout(location = 0, index = 1) out vec4 out_transmittance;",
    "layout(location = 1) out vec4 out_transmittance;"
).replace(
    "    if (u_hdr_enabled) {\n        scattered *= u_exposure;\n    }",
    ""
)

atmo_fragment_shader = sky_view_lut_fragment_shader.replace(
    "in vec2 f_uv;",
    "in vec3 f_local_pos;\nin float f_clip_z;\nuniform bool u_use_sky_view_lut;\nuniform sampler2D u_sky_view_lut_color;\nuniform sampler2D u_sky_view_lut_transmittance;"
).replace(
    "    // Build local coordinate frame relative to the local vertical (zenith) at camera position\n    vec3 up = length(cam_local) > 1e-5 ? normalize(cam_local) : vec3(0.0, 1.0, 0.0);\n    vec3 east = cross(vec3(0.0, 1.0, 0.0), up);\n    if (length(east) < 1e-4) {\n        east = cross(up, vec3(0.0, 0.0, 1.0));\n    }\n    east = normalize(east);\n    vec3 north = cross(up, east);\n\n    float azimuth = f_uv.x * 2.0 * PI;\n    float y = f_uv.y * 2.0 - 1.0;\n    float elevation = sign(y) * y * y * (PI / 2.0);\n    \n    vec3 ray_dir_local = vec3(cos(elevation)*sin(azimuth), sin(elevation), cos(elevation)*cos(azimuth));\n    vec3 ray_dir = ray_dir_local.x * east + ray_dir_local.y * up + ray_dir_local.z * north;",
    "    if (f_clip_z < 0.0) discard;\n    vec3 f_pos_local_au = f_local_pos * u_atmo_radius_au;\n    vec3 f_pos_local = f_pos_local_au * u_au_to_km;\n    vec3 ray_dir = normalize(f_pos_local - cam_local);"
).replace(
    "    vec3 beta_R = u_beta_rayleigh * 1000.0;",
    "    if (u_use_sky_view_lut && abs(s_start - max(0.0, s_atmo.x)) < 1e-4 && abs(s_end - s_atmo.y) < 1e-4) {\n        vec3 up = length(cam_local) > 1e-5 ? normalize(cam_local) : vec3(0.0, 1.0, 0.0);\n        vec3 east = cross(vec3(0.0, 1.0, 0.0), up);\n        if (length(east) < 1e-4) {\n            east = cross(up, vec3(0.0, 0.0, 1.0));\n        }\n        east = normalize(east);\n        vec3 north = cross(up, east);\n\n        float dir_up = dot(ray_dir, up);\n        float dir_east = dot(ray_dir, east);\n        float dir_north = dot(ray_dir, north);\n\n        float elevation = asin(clamp(dir_up, -1.0, 1.0));\n        float az = atan(dir_east, dir_north);\n        float u = az / (2.0 * 3.14159265358979);\n        if (u < 0.0) u += 1.0;\n        float v = 0.5 + 0.5 * sign(elevation) * sqrt(abs(elevation) / (3.14159265358979 / 2.0));\n        vec3 scattered = textureLod(u_sky_view_lut_color, vec2(u, v), 0.0).rgb;\n        vec3 transmittance = textureLod(u_sky_view_lut_transmittance, vec2(u, v), 0.0).rgb;\n        if (u_hdr_enabled) scattered *= u_exposure;\n        out_color = vec4(scattered, 1.0);\n        out_transmittance = vec4(transmittance, 1.0);\n        return;\n    }\n    vec3 beta_R = u_beta_rayleigh * 1000.0;"
).replace(
    "    if (s_atmo.x > s_atmo.y) {\n        out_color = vec4(0.0, 0.0, 0.0, 1.0);\n        out_transmittance = vec4(1.0, 1.0, 1.0, 1.0);\n        return;\n    }",
    "    if (s_atmo.x > s_atmo.y) discard;"
).replace(
    "    if (s_start >= s_end) {\n        out_color = vec4(0.0, 0.0, 0.0, 1.0);\n        out_transmittance = vec4(1.0, 1.0, 1.0, 1.0);\n        return;\n    }",
    "    if (s_start >= s_end) discard;"
)



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
uniform vec3 u_beta_rayleigh;
uniform vec3 u_beta_mie;
uniform vec3 u_beta_abs_mixed;
uniform vec3 u_beta_abs_layered;

vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float a = dot(dir, dir);
    float b = dot(origin, dir);
    
    // Improved precision for large distances
    vec3 p = origin - (b / a) * dir;
    float p2 = dot(p, p);
    float r2 = radius * radius;
    
    if (p2 > r2) return vec2(1e10, -1e10);
    
    float d = sqrt((r2 - p2) / a);
    float t_closest = -b / a;
    
    return vec2(t_closest - d, t_closest + d);
}

void main() {
    float x = f_uv.x * 2.0 - 1.0;
    float cos_theta = sign(x) * x * x;
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float h = f_uv.y * f_uv.y * max(1e-4, u_atmo_radius_km - u_planet_radius_km);
    float r = u_planet_radius_km + h;
    
    vec3 origin = vec3(0.0, r, 0.0);
    vec3 dir = vec3(sin_theta, cos_theta, 0.0);
    
    vec2 t_atmo = raySphereIntersect(origin, dir, u_atmo_radius_km);
    vec2 t_planet = raySphereIntersect(origin, dir, u_planet_radius_km);
    
    float ray_len = t_atmo.y;
    if (t_planet.x > 0.0 && t_planet.x < t_atmo.y) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    int num_samples = 256;
    float step_size = ray_len / float(num_samples);
    
    float od_rayleigh = 0.0;
    float od_mie = 0.0;
    float od_ozone = 0.0;
    
    for (int i = 0; i < num_samples; i++) {
        float t = (float(i) + 0.5) * step_size;
        vec3 p = origin + t * dir;
        float h_sample = max(0.0, length(p) - u_planet_radius_km);
        
        od_rayleigh += exp(-h_sample / u_h_rayleigh) * step_size;
        od_mie += exp(-h_sample / u_h_mie) * step_size;
        od_ozone += exp(-pow((h_sample - 25.0) / 8.0, 2.0)) * step_size;
    }
    
    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M = u_beta_mie * 1000.0;
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;
    
    vec3 transmittance = exp(-(beta_R * od_rayleigh + beta_M * od_mie + beta_A_mixed * od_rayleigh + beta_A_layered * od_ozone));
    out_color = vec4(transmittance, 1.0);
}
"""

multi_scatter_lut_fragment_shader = """
#version 460 core
in vec2 f_uv;
out vec4 out_color;

uniform float u_planet_radius_km;
uniform float u_atmo_radius_km;
uniform float u_h_rayleigh;
uniform float u_h_mie;
uniform vec3 u_beta_rayleigh;
uniform vec3 u_beta_mie;
uniform vec3 u_beta_abs_mixed;
uniform vec3 u_beta_abs_layered;
uniform float u_mie_g;
uniform vec3 u_ground_albedo;

uniform sampler2D u_transmittance_lut;

vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
    float a = dot(dir, dir);
    float b = dot(origin, dir);
    
    // Improved precision for large distances
    vec3 p = origin - (b / a) * dir;
    float p2 = dot(p, p);
    float r2 = radius * radius;
    
    if (p2 > r2) return vec2(1e10, -1e10);
    
    float d = sqrt((r2 - p2) / a);
    float t_closest = -b / a;
    
    return vec2(t_closest - d, t_closest + d);
}

vec3 get_transmittance(float r, float cos_theta) {
    float h_norm = clamp((r - u_planet_radius_km) / max(1e-4, u_atmo_radius_km - u_planet_radius_km), 0.0, 1.0);
    float v = sqrt(h_norm);
    float u = 0.5 + 0.5 * sign(cos_theta) * sqrt(abs(cos_theta));
    return textureLod(u_transmittance_lut, vec2(u, v), 0.0).rgb;
}

void main() {
    float x = f_uv.x * 2.0 - 1.0;
    float cos_sun_zenith = sign(x) * x * x;
    float sin_sun_zenith = sqrt(max(0.0, 1.0 - cos_sun_zenith * cos_sun_zenith));
    float h = f_uv.y * f_uv.y * max(1e-4, u_atmo_radius_km - u_planet_radius_km);
    float r = u_planet_radius_km + h;
    
    vec3 origin = vec3(0.0, r, 0.0);
    vec3 sun_dir = vec3(sin_sun_zenith, cos_sun_zenith, 0.0);
    
    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M = u_beta_mie * 1000.0;
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;
    
    const int sqrt_samples = 8;
    vec3 lum_total = vec3(0.0);
    vec3 fms_total = vec3(0.0);
    
    for (int i = 0; i < sqrt_samples; i++) {
        for (int j = 0; j < sqrt_samples; j++) {
            float u = (float(i) + 0.5) / float(sqrt_samples);
            float v = (float(j) + 0.5) / float(sqrt_samples);
            
            float theta = acos(1.0 - 2.0 * u);
            float phi = 2.0 * 3.14159265358979 * v;
            
            vec3 ray_dir = vec3(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
            
            vec2 t_atmo = raySphereIntersect(origin, ray_dir, u_atmo_radius_km);
            vec2 t_planet = raySphereIntersect(origin, ray_dir, u_planet_radius_km);
            
            float ray_len = t_atmo.y;
            if (t_planet.x > 0.0 && t_planet.x < t_atmo.y) {
                ray_len = t_planet.x;
            }
            
            int ray_samples = 20;
            float step_size = ray_len / float(ray_samples);
            
            vec3 lum = vec3(0.0);
            vec3 fms = vec3(0.0);
            
            float current_s = 0.5 * step_size;
            vec3 transmittance_accum = vec3(1.0);
            
            for (int s = 0; s < ray_samples; s++) {
                vec3 p = origin + ray_dir * current_s;
                float p_len = length(p);
                float h_sample = max(0.0, p_len - u_planet_radius_km);
                
                float rho_R = exp(-h_sample / u_h_rayleigh);
                float rho_M = exp(-h_sample / u_h_mie);
                float rho_O = exp(-pow((h_sample - 25.0) / 8.0, 2.0));
                
                vec3 scattering = beta_R * rho_R + beta_M * rho_M;
                vec3 extinction = scattering + beta_A_mixed * rho_R + beta_A_layered * rho_O;
                
                vec3 sample_transmittance = exp(-extinction * step_size);
                
                float p_cos_sun = dot(p, sun_dir) / p_len;
                vec3 trans_to_sun = get_transmittance(p_len, p_cos_sun);
                
                float phase = 1.0 / (4.0 * 3.14159265358979);
                
                vec3 S = scattering * trans_to_sun * phase;
                
                vec3 Sint = (S - S * sample_transmittance) / max(extinction, 1e-6);
                lum += transmittance_accum * Sint;
                
                vec3 FMS_Sint = (scattering - scattering * sample_transmittance) / max(extinction, 1e-6);
                fms += transmittance_accum * FMS_Sint;
                
                transmittance_accum *= sample_transmittance;
                current_s += step_size;
            }
            
            if (t_planet.x > 0.0 && t_planet.x < t_atmo.y) {
                vec3 p = origin + ray_dir * t_planet.x;
                float p_len = length(p);
                float p_cos_sun = dot(p, sun_dir) / max(p_len, 1e-6);
                vec3 trans_to_sun = get_transmittance(p_len, p_cos_sun);
                
                float NdotL = max(0.0, p_cos_sun);
                vec3 ground_albedo = u_ground_albedo;
                vec3 ground_lum = (ground_albedo / 3.14159265358979) * NdotL * trans_to_sun;
                
                lum += transmittance_accum * ground_lum;
            }
            
            lum_total += lum;
            fms_total += fms;
        }
    }
    
    float n_samples = float(sqrt_samples * sqrt_samples);
    vec3 L2nd = lum_total / n_samples;
    vec3 fms_avg = fms_total / n_samples;
    
    vec3 psi = L2nd / max(vec3(1.0) - fms_avg, 1e-6);
    out_color = vec4(psi, 1.0);
}
"""

aerial_perspective_compute_shader = sky_view_lut_fragment_shader.replace(
    "in vec2 f_uv;",
    "layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;\nlayout(rgba16f, binding = 0) writeonly uniform image3D destTex;"
).replace(
    "void main() {",
    """void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(gid, ivec3(32, 32, 32)))) return;
    
    vec2 f_uv = vec2(float(gid.x) + 0.5, float(gid.y) + 0.5) / 32.0;
    float depth_slice = float(gid.z) / 31.0;
    float slice_dist = depth_slice * depth_slice * 100000.0;
"""
).replace(
    "// AP_VOLUME_HOOK",
    "s_end = min(s_end, slice_dist);"
).replace(
    "layout(location = 0, index = 0) out vec4 out_color;\nlayout(location = 0, index = 1) out vec4 out_transmittance;",
    "vec4 out_color;\nvec4 out_transmittance;"
).replace(
    "    out_color = vec4(scattered, 1.0);\n    out_transmittance = vec4(transmittance, 1.0);",
    "    float avg_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;\n    imageStore(destTex, gid, vec4(scattered, (1.0 - avg_transmittance)));"
).replace(
    "    if (s_atmo.x > s_atmo.y) {\n        out_color = vec4(0.0, 0.0, 0.0, 1.0);\n        out_transmittance = vec4(1.0, 1.0, 1.0, 1.0);\n        return;\n    }",
    "    if (s_atmo.x > s_atmo.y) {\n        imageStore(destTex, gid, vec4(0.0));\n        return;\n    }"
).replace(
    "    if (s_start >= s_end) {\n        out_color = vec4(0.0, 0.0, 0.0, 1.0);\n        out_transmittance = vec4(1.0, 1.0, 1.0, 1.0);\n        return;\n    }",
    "    if (s_start >= s_end) {\n        imageStore(destTex, gid, vec4(0.0));\n        return;\n    }"
)
