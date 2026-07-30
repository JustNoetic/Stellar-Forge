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
uniform vec3 u_camera_pos;
uniform float u_screen_height;
uniform float u_fov_factor;
uniform float u_lod_thresh_ultra;
uniform float u_lod_thresh_hi;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_num_bodies)) return;

    vec4 f0 = instances[idx * 7 + 0];
    vec4 f1 = instances[idx * 7 + 1];
    vec4 f2 = instances[idx * 7 + 2];

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

    vec3 star_pos = instances[u_star_idx * 7 + 0].xyz;
    float star_r = instances[u_star_idx * 7 + 1].z;

    if (idx != uint(u_star_idx)) {
        vec3 L = star_pos - pos;
        float dist_star = length(L);
        if (dist_star >= 1e-6) {
            vec3 L_dir = L / dist_star;

            for (int c = 0; c < u_n_casters; c++) {
                int j = u_caster_indices[c];
                if (j == int(idx) || j == u_star_idx) continue;

                vec3 p_j = instances[j * 7 + 0].xyz;
                float r_j = instances[j * 7 + 1].z;

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

    instances[idx * 7 + 3].y = uintBitsToFloat(mask_lo);
    instances[idx * 7 + 3].z = uintBitsToFloat(mask_hi);
    instances[idx * 7 + 3].w = uintBitsToFloat(r_mask);

    // 3. Lodge into visible buffers based on screen-space size & tracking state
    if (visible) {
        float cam_dist = length(pos - u_camera_pos);
        float apparent_px = (cam_dist > 1e-6) ? ((r / cam_dist) * u_screen_height * u_fov_factor) : 10000.0;

        bool is_ultra = (int(idx) == u_tracking_idx) || (apparent_px >= u_lod_thresh_ultra);
        bool is_hi = (apparent_px >= u_lod_thresh_hi) || (f2.x > 0.5);

        if (is_ultra) {
            uint slot = atomicAdd(cmds[2].instanceCount, 1u);
            vis_ultra[slot] = idx;
        } else if (is_hi) {
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
flat out float f_tex_idx;
flat out float f_rotation_angle;
out vec3 f_local_pos;
flat out vec3 f_pole;
void main() {
    uint inst_idx = vis_indices[gl_InstanceID];
    vec4 f0 = instances[inst_idx * 7 + 0];
    vec4 f1 = instances[inst_idx * 7 + 1];
    vec4 f2 = instances[inst_idx * 7 + 2];
    vec4 f3 = instances[inst_idx * 7 + 3];
    vec4 f4 = instances[inst_idx * 7 + 4];
    vec4 f5 = instances[inst_idx * 7 + 5];
    vec4 f6 = instances[inst_idx * 7 + 6];

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
    f_tex_idx = f6.x;
    f_rotation_angle = f6.y;
    f_pole = in_pole;

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

    f_local_pos = in_position;
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
#define PI 3.14159265358979323846

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
flat in float f_tex_idx;
flat in float f_rotation_angle;
flat in vec3 f_pole;
in vec3 f_local_pos;

uniform sampler2DArray u_planet_textures;
uniform sampler2DArray u_planet_normal_textures;
uniform sampler2DArray u_planet_specular_textures;

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
uniform vec4 u_ring_params[MAX_RING_PLANES];
uniform vec3 u_ring_colors[MAX_RING_PLANES];
uniform vec3 u_ring_5colors[MAX_RING_PLANES * 5];
uniform sampler2D u_ring_gradients;
uniform sampler2D u_ringshine_lut;
uniform sampler3D u_ringshine_cdf_lut;
uniform sampler2D u_ringshine_map;
uniform int u_num_ring_planes;
uniform float u_caster_max_bend[64];
uniform uint u_ring_coplanar_mask[16];
uniform int u_atmo_quality;
uniform bool u_planetshine_enabled;
uniform bool u_ringshine_enabled;
uniform int u_ringshine_band_count;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform vec3 u_camera_pos;
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

// Shared analytical eclipse penumbra + physical atmospheric lens optics & Danjon refraction tinting.
// Inputs are per-caster quantities already resolved by the caller.
vec3 casterShadowTerm(float alpha, float beta, float gamma,
                      float penumbra_outer, float penumbra_inner,
                      float max_bend, vec3 atmo_tint) {
    float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
    float occ = clamp(max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma), 0.0, 1.0);
    // Correct geometric shadow curve: (1 - occ)^GAMMA (Space Engine gamma correction: 1 / 1.6)
    vec3 sh = vec3(pow(clamp(1.0 - occ, 0.0, 1.0), 1.0 / 1.6));

    if (max_bend > 1e-6 && gamma < penumbra_outer) {
        // Required bend angle for light to reach inside the planet's geometric shadow
        float req_bend = beta - gamma;

        // Rays requiring bend > max_bend hit the solid planet body (100% blocked)
        if (req_bend <= max_bend) {
            // Normalized penetration depth in atmosphere (0 = top of atmosphere, 1 = surface level)
            float atmo_depth = clamp(max(0.0, req_bend) / max_bend, 0.0, 1.0);

            // Compute Rayleigh / Mie / absorption spectral transmittance combined with composition tint
            vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (atmo_depth * 2.0));
            vec3 ext_exp = vec3(3.5, 10.0, 22.0) * (1.5 - deep_tint);
            vec3 atmo_transmittance = vec3(
                exp(-ext_exp.r * pow(atmo_depth, 1.5)),
                exp(-ext_exp.g * pow(atmo_depth, 1.5)),
                exp(-ext_exp.b * pow(atmo_depth, 1.5))
            );

            // Physical Spherical Lens Optics Dynamics (Space Engine Model)
            // Focal ratio: f_r = D_caster / D_focal = max_bend / beta
            float focal_ratio = max_bend / max(1e-6, beta);

            // Inverse-square beam divergence beyond focal plane: (D_focal / D_caster)^2 = (1 / f_r)^2
            float distance_divergence = min(1.0, 1.0 / max(1e-6, focal_ratio * focal_ratio));

            // Spherical lens caustic amplification near focal plane (f_r ~ 1.0)
            float lens_amplification = clamp(1.0 / (abs(1.0 - focal_ratio) + 0.3), 0.5, 3.0);

            // Smooth surface grazing fade to zero at solid body boundary (h = 0, atmo_depth = 1)
            float body_surface_fade = smoothstep(1.0, 0.8, atmo_depth);

            // Physical Danjon Lunar Eclipse Scale with Visual Exposure Boost:
            // Base physical value is ~0.00005, boosted to 0.015 for vibrant, clear visual presentation.
            float atmo_base_intensity = 0.015;
            float refraction_intensity = atmo_base_intensity * distance_divergence * lens_amplification * (1.0 - atmo_depth * 0.7) * body_surface_fade;

            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            sh += deep_tint * atmo_transmittance * refraction_intensity * atmo_blend;
        }
    }
    return clamp(sh, vec3(0.0), vec3(1.0));
}

float eval_ringshine_cdf(float angle, float v_tex, float sin_lat) {
    float TWO_PI = 6.28318530717958647692;
    float a_mod = angle - TWO_PI * floor((angle + PI) / TWO_PI);
    float k = floor((angle + PI) / TWO_PI);
    float u = clamp(abs(a_mod) / PI, 0.0, 1.0);

    // Remap coordinates to texel centers to avoid GL_CLAMP_TO_EDGE flat spots (derivative kinks)
    float u_tex = 0.5 / 128.0 + u * (127.0 / 128.0);
    float v_tex_mapped = 0.5 / 128.0 + v_tex * (127.0 / 128.0);
    float sin_lat_mapped = 0.5 / 64.0 + sin_lat * (63.0 / 64.0);

    float base_cdf = texture(u_ringshine_cdf_lut, vec3(u_tex, v_tex_mapped, sin_lat_mapped)).r;
    float signed_cdf = (a_mod < 0.0) ? -base_cdf : base_cdf;
    return 2.0 * k + signed_cdf;
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
        float spec_intensity = 0.0;
        vec3 local_f_color = f_color;

        if (f_tex_idx > 0.0) {
            vec3 p = normalize(f_local_pos);
            vec3 ref = vec3(0.0, 1.0, 0.0);
            if (abs(dot(f_pole, ref)) > 0.999) {
                ref = vec3(1.0, 0.0, 0.0);
            }
            vec3 tangent = normalize(cross(f_pole, ref));
            vec3 bitangent = normalize(cross(f_pole, tangent));

            vec3 p_local = vec3(dot(p, tangent), dot(p, f_pole), dot(p, bitangent));

            float s = sin(-f_rotation_angle);
            float c = cos(-f_rotation_angle);
            vec3 p_rot = vec3(
                p_local.x * c - p_local.z * s,
                p_local.y,
                p_local.x * s + p_local.z * c
            );

            float u = 0.5 + atan(p_rot.z, p_rot.x) / (2.0 * 3.14159265);
            float v = 0.5 - asin(clamp(p_rot.y, -1.0, 1.0)) / 3.14159265;

            vec2 uv = vec2(u, v);
            vec2 dx = dFdx(uv);
            vec2 dy = dFdy(uv);

            if (dx.x > 0.5) dx.x -= 1.0;
            else if (dx.x < -0.5) dx.x += 1.0;

            if (dy.x > 0.5) dy.x -= 1.0;
            else if (dy.x < -0.5) dy.x += 1.0;

            vec4 tex_color = textureGrad(u_planet_textures, vec3(uv, f_tex_idx - 1.0), dx, dy);
            // Decode sRGB to Linear
            local_f_color = pow(tex_color.rgb, vec3(2.2));

            // Analytical TBN Mapping
            vec3 map_normal = textureGrad(u_planet_normal_textures, vec3(uv, f_tex_idx - 1.0), dx, dy).rgb;
            map_normal = map_normal * 2.0 - 1.0;

            vec3 T_rot = normalize(vec3(-p_rot.z, 0.0, p_rot.x));
            vec3 B_rot = normalize(cross(p_rot, T_rot));

            float s_inv = sin(f_rotation_angle);
            float c_inv = cos(f_rotation_angle);
            vec3 T_local = vec3(T_rot.x * c_inv - T_rot.z * s_inv, T_rot.y, T_rot.x * s_inv + T_rot.z * c_inv);
            vec3 B_local = vec3(B_rot.x * c_inv - B_rot.z * s_inv, B_rot.y, B_rot.x * s_inv + B_rot.z * c_inv);

            vec3 T_world = T_local.x * tangent + T_local.y * f_pole + T_local.z * bitangent;
            vec3 B_world = B_local.x * tangent + B_local.y * f_pole + B_local.z * bitangent;

            mat3 TBN = mat3(T_world, B_world, N);
            N = normalize(TBN * map_normal);

            spec_intensity = textureGrad(u_planet_specular_textures, vec3(uv, f_tex_idx - 1.0), dx, dy).r;
        }

        vec3 total_diffuse_color = vec3(0.0);
        vec3 total_specular_color = vec3(0.0);
        vec3 V = normalize(u_camera_pos - f_world_pos);

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

            // Lambert cosine law with soft terminator
            float NdotL = dot(N, L);
            float diffuse = clamp((NdotL + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);

            // Blinn-Phong specular highlight
            vec3 H = normalize(L + V);
            float NdotH = max(dot(N, H), 0.0);
            float specular = pow(NdotH, 64.0) * spec_intensity * diffuse;

            // Inverse-square falloff
            float falloff = star_lum / (dist_to_star * dist_to_star);
            diffuse *= falloff;
            specular *= falloff;

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
                vec3 cross_vec = cross(frag_to_caster, L);
                float perp_sq = dot(cross_vec, cross_vec);

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
                shadow *= casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, u_caster_max_bend[j], u_caster_atmos[j].xyz);
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
                    R_eff = max(R_eff, fwidth(d) * 0.75);

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
                            sum_alpha = textureLod(u_ring_gradients, vec2(p, (float(j) + 0.5) / 16.0), 0.0).a;
                        } else {
                            // Penumbra is wide, use 5 samples for filtering
                            int tex_samples = 5;
                            for(int s = 0; s < tex_samples; s++) {
                                float u = (float(s) + 0.5) / float(tex_samples);
                                float r = mix(overlap_min, overlap_max, u);
                                float p = (r - inner_r) / max(1e-6, outer_r - inner_r);
                                sum_alpha += textureLod(u_ring_gradients, vec2(p, (float(j) + 0.5) / 16.0), 0.0).a;
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
            total_specular_color += star_color * specular * shadow;
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

            int host_caster_idx = -1;
            for (int j = 0; j < u_num_casters; j++) {
                if (distance(u_casters[j].xyz, ring_center) < 1e-4) {
                    host_caster_idx = j;
                    break;
                }
            }

            bool is_host_planet = false;
            if (host_caster_idx >= 0) {
                float host_radius = u_casters[host_caster_idx].w;
                is_host_planet = (distance(f_world_pos, u_casters[host_caster_idx].xyz) < host_radius * 1.5);
            } else {
                is_host_planet = (dist_to_center < inner_r);
            }

            float frag_elevation = dot(N, ring_normal);
            float moon_h = -dot(frag_to_center, ring_normal);

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

                vec3 ring_tint = u_ring_colors[k];
                if (is_host_planet) {
                    // --- PRE-INTEGRATED LUT RINGSHINE WITH EXACT 3D CYLINDER SHADOW & CDF MIDNIGHT PROPAGATION ---
                    float host_radius = host_caster_idx >= 0 ? u_casters[host_caster_idx].w : inner_r * 0.7;
                    float sin_lat = clamp(abs(frag_elevation), 0.001, 0.999);

                    // Fragment equator normal vector (in ring plane)
                    vec3 N_eq_raw = N - ring_normal * dot(N, ring_normal);
                    float len_N_eq = length(N_eq_raw);
                    vec3 N_eq = len_N_eq > 1e-5 ? N_eq_raw / len_N_eq : vec3(1.0, 0.0, 0.0);

                    // Anti-solar vector projection in ring plane
                    vec3 antiL = -L;
                    vec3 antiL_eq_raw = antiL - ring_normal * dot(antiL, ring_normal);
                    float len_antiL_eq = length(antiL_eq_raw);
                    vec3 antiL_eq = len_antiL_eq > 1e-5 ? antiL_eq_raw / len_antiL_eq : vec3(-1.0, 0.0, 0.0);

                    // Azimuth angle phi_center between anti-solar vector and surface fragment meridian
                    vec3 cross_rel = cross(antiL_eq, N_eq);
                    float sin_rel = dot(cross_rel, ring_normal);
                    float cos_rel = clamp(dot(N_eq, antiL_eq), -1.0, 1.0);
                    float phi_center = atan(sin_rel, cos_rel);

                    float same_hemisphere = sun_elevation * frag_elevation;
                    float same_hemi_t = smoothstep(-0.02, 0.02, same_hemisphere);

                    float x_prime = phi_center / PI;
                    float phi_uv = sign(x_prime) * pow(abs(x_prime), 0.666666667) * 0.5 + 0.5;

                    float y_prime = frag_elevation;
                    float elev_uv = sign(y_prime) * pow(abs(y_prime), 0.666666667) * 0.5 + 0.5;

                    vec2 map_uv = vec2(phi_uv, (float(k) + elev_uv) / 16.0);
                    vec3 total_ring_irradiance = texture(u_ringshine_map, map_uv).rgb;

                    float NdotL = dot(N, L);
                    float day_face = smoothstep(-0.05, 0.05, NdotL) * max(0.0, NdotL);
                    float noon_fade = mix(1.0, 0.4, day_face);

                    float face_multiplier = mix(1.0, 1.0 * noon_fade, same_hemi_t);
                    // Apply 1/PI (~0.3183) physical BRDF normalization factor
                    shine_intensity = effective_sun_elev * face_multiplier * 0.318309886;
                    ring_tint = total_ring_irradiance;

                } else {
                    // --- DEDICATED MOON RINGSHINE ---
                    vec3 closest_plane_pt = f_world_pos - moon_h * ring_normal;
                    vec3 center_to_plane_pt = closest_plane_pt - ring_center;
                    float rho = length(center_to_plane_pt);

                    // Interpolate ring color from the 5 radial texture bands at position rho
                    float u_ring = clamp((rho - inner_r) / max(1e-5, outer_r - inner_r), 0.0, 1.0);
                    float pos_c = u_ring * 4.0;
                    int idx0 = int(floor(pos_c));
                    int idx1 = min(idx0 + 1, 4);
                    float frac_c = pos_c - float(idx0);
                    vec3 c0 = u_ring_5colors[k * 5 + idx0];
                    vec3 c1 = u_ring_5colors[k * 5 + idx1];
                    ring_tint = mix(c0, c1, frac_c);

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
                        float alpha_avg = clamp(opacity, 0.0, 0.999);
                        float tau_moon = -log(max(1e-4, 1.0 - alpha_avg));
                        float moon_trans_factor = exp(-0.35 * tau_moon / max(effective_sun_elev, 0.087));

                        float ring_brightness = (same_hemisphere > 0.0)
                            ? (effective_sun_elev * opacity * 2.0)
                            : (effective_sun_elev * opacity * moon_trans_factor * 0.4);

                        shine_intensity = NdotL_ring * ring_brightness * solid_angle * ring_shadow_factor;
                    }
                }

                float irradiance = u_hdr_enabled ? (star_lum / max(dist_to_star * dist_to_star, 1e-8)) : 1.0;

                ring_shine += ring_tint * star_color * shine_intensity * irradiance;
            }
        }

        vec3 final_color = local_f_color * total_diffuse_color + total_specular_color;
        if (u_planetshine_enabled) {
            final_color += local_f_color * bounce_light;
        }
        if (u_ringshine_enabled) {
            final_color += local_f_color * ring_shine;
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

    vec3 final_rgb = color * 0.4;
    float max_c = max(final_rgb.r, max(final_rgb.g, final_rgb.b));
    if (max_c < 0.25 && max_c > 0.0001) {
        final_rgb = final_rgb * (0.25 / max_c);
    } else if (max_c <= 0.0001) {
        final_rgb = vec3(0.25);
    }

    f_color = vec4(final_rgb, float(v.w));
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
    // Decode sRGB to Linear and boost intensity to survive ACES tonemapping
    vec3 linear_color = pow(f_color.rgb, vec3(2.2)) * 4.0;
    out_color = vec4(linear_color, f_color.a);
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

    vec3 final_rgb = u_color * 0.4;
    float max_c = max(final_rgb.r, max(final_rgb.g, final_rgb.b));
    if (max_c < 0.25 && max_c > 0.0001) {
        final_rgb = final_rgb * (0.25 / max_c);
    } else if (max_c <= 0.0001) {
        final_rgb = vec3(0.25);
    }

    f_color = vec4(final_rgb, 1.0);
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
    // Decode sRGB to Linear and boost intensity to survive ACES tonemapping
    vec3 linear_color = pow(f_color.rgb, vec3(2.2)) * 4.0;
    out_color = vec4(linear_color, f_color.a);
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
uniform float u_host_planet_refractivity;  // surface (n_mix - 1) for eclipse refraction
uniform vec3 u_camera_pos;
uniform vec4 u_host_planet_pole_obl;
uniform float u_host_planet_R_minor;
uniform int u_clip_mode;
uniform uint u_caster_mask_lo;
uniform uint u_caster_mask_hi;
uniform bool u_planetshine_enabled;
uniform float u_exposure;
uniform bool u_hdr_enabled;
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
    float is_textured;
    float _pad2;
};

#define MAX_RING_PLANES 16
uniform RingPlane u_ring_planes[MAX_RING_PLANES];
uniform int u_num_ring_planes;
uniform sampler2D u_ring_gradients;

out vec4 out_color;

uniform bool u_is_textured;
uniform sampler2D u_ring_texture_front;
uniform sampler2D u_ring_texture_back;

float CornetteShanksPhaseFunction(float eccentricity, float viewDirDotLight) {
    float g = eccentricity;
    float g2 = g * g;
    float mu = viewDirDotLight;
    float denom = pow(max(1e-6, 1.0 + g2 - 2.0 * g * mu), 1.5);
    return (1.5 * (1.0 + mu * mu) * (1.0 - g2)) / ((2.0 + g2) * denom);
}

float GetRingPhaseFunctions(float dotLight, float alpha) {
    float pf_forward  = CornetteShanksPhaseFunction(0.70, dotLight);
    float pf_backward = CornetteShanksPhaseFunction(-0.35, dotLight);
    float blend = smoothstep(-0.5, 0.5, dotLight);
    return mix(pf_backward, pf_forward, blend);
}

float OppositionSurge(float cos_phase, float columnDensity) {
    // cos_phase = dot(V, L) where V points to camera, L points to star
    // At opposition: V ~ L -> cos_phase ~ 1 -> phase_angle ~ 0
    float phase_angle = acos(clamp(cos_phase, -1.0, 1.0));

    // Scale surge with optical depth: dense rings have more inter-particle shadowing
    float density_scale = clamp(columnDensity / 1.5, 0.0, 1.0);

    // SHOE: Shadow Hiding Opposition Effect (broad, ~4 degrees)
    float shoe_width = 0.07;
    float shoe_amp = 0.8 * density_scale;
    float shoe = 1.0 + shoe_amp / (1.0 + phase_angle / shoe_width);

    // CBOE: Coherent Backscatter Opposition Effect (narrow, ~0.35 degrees)
    float cboe_width = 0.006;
    float cboe_amp = 0.3 * density_scale;
    float cboe = 1.0 + cboe_amp * exp(-phase_angle / cboe_width);

    return shoe * cboe;
}

float HapkeHFunction(float mu, float gamma) {
    return (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * gamma);
}

float AnalyticMultipleScattering(float mu_v, float mu_0, float tau, bool onLitSide) {
    float w0 = 0.92; // Single-scattering albedo of ring ice particles (water-ice rings)
    float gamma = sqrt(max(1e-4, 1.0 - w0));
    float Hv = HapkeHFunction(mu_v, gamma);
    float H0 = HapkeHFunction(mu_0, gamma);

    float path_term = 1.0 - exp(-tau * (1.0 / max(1e-4, mu_v) + 1.0 / max(1e-4, mu_0)));
    float mu_ratio = mu_0 / max(1e-4, mu_v + mu_0);

    if (onLitSide) {
        float ms_lit = w0 * mu_ratio * (Hv * H0 - 1.0) * path_term;
        return max(0.0, ms_lit);
    } else {
        float ms_unlit = w0 * mu_ratio * (Hv * H0) * exp(-gamma * tau) * path_term;
        return max(0.0, ms_unlit);
    }
}

float AnalyticalSelfShadowing(float sun_side_abs) {
    // Incident solar flux scale: mu_0 = |N . L| = sin(solar elevation)
    // Radiative transfer (scatteredLight & AnalyticMultipleScattering) already accounts for
    // line-of-sight self-absorption (1/mu_v) and solar path extinction (1/mu_0) inside the slab.
    float mu_0 = max(0.001, sun_side_abs);
    return clamp(mu_0 / 0.45, 0.0, 1.0);
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

// Shared analytical eclipse penumbra + physical atmospheric lens optics & Danjon refraction tinting.
vec3 casterShadowTerm(float alpha, float beta, float gamma,
                      float penumbra_outer, float penumbra_inner,
                      float max_bend, vec3 atmo_tint) {
    float max_occ = min(1.0, (beta * beta) / max(1e-9, alpha * alpha));
    float occ = clamp(max_occ * smoothstep(penumbra_outer, penumbra_inner, gamma), 0.0, 1.0);
    // Correct geometric shadow curve: (1 - occ)^GAMMA (Space Engine gamma correction: 1 / 1.6)
    vec3 sh = vec3(pow(clamp(1.0 - occ, 0.0, 1.0), 1.0 / 1.6));

    if (max_bend > 1e-6 && gamma < penumbra_outer) {
        // Required bend angle for light to reach inside the planet's geometric shadow
        float req_bend = beta - gamma;

        // Rays requiring bend > max_bend hit the solid planet body (100% blocked)
        if (req_bend <= max_bend) {
            // Normalized penetration depth in atmosphere (0 = top of atmosphere, 1 = surface level)
            float atmo_depth = clamp(max(0.0, req_bend) / max_bend, 0.0, 1.0);

            // Compute Rayleigh / Mie / absorption spectral transmittance combined with composition tint
            vec3 deep_tint = atmo_tint * exp2(log2(max(atmo_tint, 1e-6)) * (atmo_depth * 2.0));
            vec3 ext_exp = vec3(3.5, 10.0, 22.0) * (1.5 - deep_tint);
            vec3 atmo_transmittance = vec3(
                exp(-ext_exp.r * pow(atmo_depth, 1.5)),
                exp(-ext_exp.g * pow(atmo_depth, 1.5)),
                exp(-ext_exp.b * pow(atmo_depth, 1.5))
            );

            // Physical Spherical Lens Optics Dynamics (Space Engine Model)
            // Focal ratio: f_r = D_caster / D_focal = max_bend / beta
            float focal_ratio = max_bend / max(1e-6, beta);

            // Inverse-square beam divergence beyond focal plane: (D_focal / D_caster)^2 = (1 / f_r)^2
            float distance_divergence = min(1.0, 1.0 / max(1e-6, focal_ratio * focal_ratio));

            // Spherical lens caustic amplification near focal plane (f_r ~ 1.0)
            float lens_amplification = clamp(1.0 / (abs(1.0 - focal_ratio) + 0.3), 0.5, 3.0);

            // Smooth surface grazing fade to zero at solid body boundary (h = 0, atmo_depth = 1)
            float body_surface_fade = smoothstep(1.0, 0.8, atmo_depth);

            // Physical Danjon Lunar Eclipse Scale with Visual Exposure Boost:
            // Base physical value is ~0.00005, boosted to 0.015 for vibrant, clear visual presentation.
            float atmo_base_intensity = 0.015;
            float refraction_intensity = atmo_base_intensity * distance_divergence * lens_amplification * (1.0 - atmo_depth * 0.7) * body_surface_fade;

            float atmo_blend = smoothstep(penumbra_outer, penumbra_inner, gamma);
            sh += deep_tint * atmo_transmittance * refraction_intensity * atmo_blend;
        }
    }
    return clamp(sh, vec3(0.0), vec3(1.0));
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

    vec3 total_color_front = vec3(0.0);
    vec3 total_color_back = vec3(0.0);
    vec3 total_color_fwd = vec3(0.0);
    float total_tau = 0.0;
    float total_faded_tau = 0.0;
    float total_scatter = 0.0;
    float total_asym = 0.0;
    float total_backscatter = 0.0;
    float total_textured_tau = 0.0;

    for (int i=0; i<u_num_ring_planes; i++) {
        float inner_r = u_ring_planes[i].inner_r;
        float outer_r = u_ring_planes[i].outer_r;
        float dr = fwidth(r) * 0.75;

        if (r >= inner_r - dr && r <= outer_r + dr) {
            float t = (r - inner_r) / max(1e-6, outer_r - inner_r);
            vec4 tex_val_front;
            vec4 tex_val_back;
            bool plane_is_textured = u_is_textured && (u_ring_planes[i].is_textured > 0.5);
            if (plane_is_textured) {
                tex_val_front = texture(u_ring_texture_front, vec2(clamp(t, 0.0, 1.0), 0.5));
                tex_val_back  = texture(u_ring_texture_back,  vec2(clamp(t, 0.0, 1.0), 0.5));
            } else {
                tex_val_front = texture(u_ring_gradients, vec2(clamp(t, 0.0, 1.0), (float(u_ring_planes[i].row_idx) + 0.5)/16.0));
                tex_val_back = tex_val_front;
            }
            tex_val_front.rgb = pow(tex_val_front.rgb, vec3(2.2));
            tex_val_back.rgb  = pow(tex_val_back.rgb,  vec3(2.2));

            float alpha = tex_val_front.a;
            vec3 r_color_front = tex_val_front.rgb;
            vec3 r_color_back  = tex_val_back.rgb;
            float fwd_lum = pow(tex_val_back.a, 2.2);
            float back_lum = max(1e-4, dot(r_color_front, vec3(0.2126, 0.7152, 0.0722)));
            vec3 r_color_fwd = r_color_front * (fwd_lum / back_lum);

            float edge_alpha = smoothstep(inner_r - dr, inner_r + dr, r) * (1.0 - smoothstep(outer_r - dr, outer_r + dr, r));

            vec3 plane_color = plane_is_textured ? vec3(1.0) : u_ring_planes[i].color;
            float plane_opacity = plane_is_textured ? 1.0 : u_ring_planes[i].opacity;

            float raw_a_physical = alpha * plane_opacity;
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
                total_color_front += plane_color * r_color_front * tau;
                total_color_back  += plane_color * r_color_back  * tau;
                total_color_fwd   += plane_color * r_color_fwd   * tau;
                total_scatter += u_ring_planes[i].scatter * tau;
                total_asym += u_ring_planes[i].asymmetry * tau;
                total_backscatter += u_ring_planes[i].backscatter * tau;
                total_tau += tau;
                total_faded_tau += tau_faded;
                if (plane_is_textured) total_textured_tau += tau;
            }
        }
    }

    if (total_tau <= 1e-6) {
        out_color = vec4(0.0);
        return;
    }

    vec3 V = normalize(u_camera_pos - f_world_pos);
    vec3 N = normalize(f_normal);
    float cam_side = dot(N, V);

    float min_cam_mu = mix(0.18, 0.04, clamp(1.0 - exp(-total_tau), 0.0, 1.0));
    float cosViewRayVertical = max(abs(cam_side), min_cam_mu);

    float physical_alpha = 1.0 - exp(-total_tau / cosViewRayVertical);
    float faded_alpha = 1.0 - exp(-total_faded_tau / cosViewRayVertical);

    vec3 f_color_front = total_color_front / max(1e-6, total_tau);
    vec3 f_color_back  = total_color_back  / max(1e-6, total_tau);
    vec3 f_color_fwd   = total_color_fwd   / max(1e-6, total_tau);
    vec4 f_color = vec4(f_color_front, physical_alpha);
    float f_scatter = total_scatter / max(1e-6, total_tau);
    float f_asymmetry = total_asym / max(1e-6, total_tau);
    float f_backscatter = total_backscatter / max(1e-6, total_tau);


    // Phase angle for opposition surge: dot(V, L) computed per-star below
    // cos_theta = -dot(L, V) is the scattering angle, but for opposition we need dot(V, L)

    bool is_textured_ring = (total_textured_tau > 0.5 * total_tau);

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

        float sun_side = dot(N, L);

        float star_ang_radius = star_radius / dist_to_star;
        float sin_alpha_local = clamp(star_ang_radius, 1e-6, 1.0);
        float alpha = sin_alpha_local;

        float effective_sun_side = sqrt(sun_side * sun_side + 0.180126 * alpha * alpha);
        float solar_elevation = max(0.02, effective_sun_side);

        float v_star = clamp(sun_side / sin_alpha_local, -1.0, 1.0);
        float f_top = (v_star * sqrt(max(0.0, 1.0 - v_star*v_star)) + asin(v_star)) / 3.14159265358979 + 0.5;
        float same_side = (cam_side > 0.0) ? f_top : (1.0 - f_top);

        float direct_illum_s = 0.0;
        float min_cam_mu = mix(0.18, 0.04, clamp(f_color.a, 0.0, 1.0));
        float min_light_mu = mix(0.18, 0.04, clamp(f_color.a, 0.0, 1.0));
        float cosViewRayVertical  = max(abs(cam_side),  min_cam_mu);
        float cosLightRayVertical = max(abs(sun_side), min_light_mu);
        float columnDensity = total_tau;
        float viewDensity = columnDensity / cosViewRayVertical;
        float lightDensity = columnDensity / cosLightRayVertical;
        float scatteredLight = 0.0;
        bool onLitSide = (cam_side * sun_side) >= 0.0;

        if (onLitSide) {
            scatteredLight = viewDensity / (viewDensity + lightDensity) * (1.0 - exp(-viewDensity - lightDensity));
        } else {
            float denominator = lightDensity - viewDensity;
            if (abs(denominator) > 1e-6) {
                scatteredLight = (exp(-viewDensity) - exp(-lightDensity)) * viewDensity / (lightDensity - viewDensity);
            } else {
                scatteredLight = viewDensity * exp(-viewDensity);
            }
        }

        float single_scatter_s = 0.0;
        float ms_s = 0.0;

        if (is_textured_ring) {
            float pf_ref = GetRingPhaseFunctions(1.0, f_color.a);
            float pf_current = GetRingPhaseFunctions(cos_theta, f_color.a);
            float pf_norm = pf_current / max(1e-4, pf_ref);
            if (onLitSide) {
                // Exposure compensation for baked texture photograph
                single_scatter_s = scatteredLight * pf_norm * 2.0;
                ms_s = 0.05 * (1.0 - exp(-columnDensity));
            } else {
                // Dimmer unlit/transmission side to simulate rock/dense ice absorption
                single_scatter_s = scatteredLight * pf_norm * 0.2;
                ms_s = 0.0;
            }
        } else {
            // Double Cornette-Shanks phase function using the ring's own parameters
            float pf_forward = CornetteShanksPhaseFunction(f_asymmetry, cos_theta);
            float pf_backward = CornetteShanksPhaseFunction(f_backscatter, cos_theta);
            // f_scatter controls forward/backward balance (0 = backward dominant, 1 = forward dominant)
            float balance = clamp(f_scatter, 0.0, 1.0);
            float phaseFunc = mix(pf_backward, pf_forward, balance);
            single_scatter_s = scatteredLight * phaseFunc;
            ms_s = AnalyticMultipleScattering(cosViewRayVertical, cosLightRayVertical, columnDensity, onLitSide);
        }

        // Opposition surge: brightening at low phase angles on single scattering (lit side only)
        if (onLitSide) {
            float cos_phase = -cos_theta; // dot(V, L)
            single_scatter_s *= OppositionSurge(cos_phase, columnDensity);
        }

        direct_illum_s = single_scatter_s + ms_s;

        // Geometric incident flux scaling (solar elevation)
        // Self-shadowing & line-of-sight self-absorption are unified in scatteredLight / AnalyticMultipleScattering
        float self_shadow_factor = AnalyticalSelfShadowing(abs(sun_side));
        direct_illum_s *= self_shadow_factor;

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
            shadow_s *= casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, u_caster_max_bend[j], u_caster_atmos[j].xyz);
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
                        float max_bend = host_atmo_h > 0.0
                            ? clamp(2.0 * max(u_host_planet_refractivity, 0.0) * sqrt(3.14159265359 * host_r / max(1e-6, host_atmo_h * 2.0)), 0.001, 0.05)
                            : 0.0;
                        shadow_s *= casterShadowTerm(alpha, beta, gamma, penumbra_outer, penumbra_inner, max_bend, u_host_planet_atmo.xyz);
                    }
                }
            }
        }

        float lit_blend = smoothstep(-0.02, 0.02, cam_side * sun_side);
        float fwd_blend = smoothstep(-0.3, 0.7, cos_theta);
        vec3 lit_color_star = (total_textured_tau > 0.5 * total_tau) ? mix(f_color_front, f_color_fwd, fwd_blend) : f_color_front;
        vec3 active_ring_color = (total_textured_tau > 0.5 * total_tau) ? mix(f_color_back, lit_color_star, lit_blend) : f_color_front;
        total_direct_illum_color += star_color * active_ring_color * direct_illum_s * shadow_s;
    }

    vec3 total_planetshine = vec3(0.0);
    if (u_host_planet_radius > 0.0) {
        vec3 frag_to_host = u_host_planet_pos - f_world_pos;
        float dist_host_sq = dot(frag_to_host, frag_to_host);
        float dist_host = sqrt(dist_host_sq);
        vec3 L_planet = frag_to_host / dist_host; // Direction towards host planet

        // Exact angular radius & spherical cap solid angle of planet as seen from this ring radius
        float sin_alpha_planet = clamp(u_host_planet_radius / dist_host, 0.0, 1.0);
        float cos_alpha_planet = sqrt(max(0.0, 1.0 - sin_alpha_planet * sin_alpha_planet));
        float solid_angle = 2.0 * (1.0 - cos_alpha_planet); // Exact solid angle Omega(r) = 2pi(1 - cos alpha)

        float cos_theta_p = -dot(L_planet, V);
        // The planet sphere extends above/below the 2D ring plane by its angular radius alpha_planet
        float planet_elevation_scale = clamp(sin_alpha_planet, 0.05, 1.0);
        float lit_blend_p = smoothstep(-0.02, 0.02, cam_side);

        vec3 active_shine_color = (is_textured_ring) ? mix(f_color_back, f_color_front, lit_blend_p) : f_color_front;

        float pf_p = 0.0;
        if (is_textured_ring) {
            float pf_ref_p = GetRingPhaseFunctions(1.0, f_color.a);
            float pf_curr_p = GetRingPhaseFunctions(cos_theta_p, f_color.a);
            pf_p = pf_curr_p / max(1e-4, pf_ref_p);
        } else {
            float pf_forward_p = CornetteShanksPhaseFunction(f_asymmetry, cos_theta_p);
            float pf_backward_p = CornetteShanksPhaseFunction(f_backscatter, cos_theta_p);
            float balance_p = clamp(f_scatter, 0.0, 1.0);
            pf_p = mix(pf_backward_p, pf_forward_p, balance_p);
        }

        float columnDensity_p = -log(max(1.0 - f_color.a, 1e-5));
        float lightDensity_p = columnDensity_p / max(0.04, planet_elevation_scale);
        float scattered_opacity_p = 1.0 - exp(-lightDensity_p);

        float ring_planet_response = pf_p * scattered_opacity_p * planet_elevation_scale;

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

            // Illuminated phase of planet as seen from the ring point: exact Lambertian spherical phase function
            float cos_a_p = clamp(dot(L_host, -L_planet), -1.0, 1.0);
            float a_p = acos(cos_a_p);
            float planet_phase = (sin(a_p) + (3.141592653589793 - a_p) * cos_a_p) / 3.141592653589793;

            // Planetshine flux reaching the ring point
            vec3 planetshine_irradiance = u_host_planet_color * star_color * irradiance * planet_phase * solid_angle * 0.8;

            total_planetshine += planetshine_irradiance * active_shine_color * ring_planet_response;
        }
    }

    vec3 raw_color = total_direct_illum_color;
    if (u_planetshine_enabled) {
        raw_color += total_planetshine;
    }
    vec3 final_color = u_hdr_enabled ? (raw_color * u_exposure) : raw_color;

    float fade_scale = faded_alpha / max(1e-6, physical_alpha);
    out_color = vec4(final_color * fade_scale, faded_alpha);
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
    vec3 linear_color = pow(u_color.rgb, vec3(2.2)) * 4.0;
    out_color = vec4(linear_color, u_color.a * alpha);
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
    float u_frame_counter;
    vec3  u_mie_albedo;          // per-channel single-scattering albedo (omega_0)
    float u_refractivity;        // surface (n_mix - 1), drives eclipse refraction
    vec4  u_active_casters[8];
    vec4  u_active_caster_poles_obl[8];
    float u_active_caster_R_minor[8];
    vec4  u_active_caster_atmos[8];
    float u_active_max_bend[8];
    float u_ozone_peak_km;
    float u_ozone_width_km;
    float _atmo_pad0;
    float _atmo_pad1;
};

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

atmo_fragment_shader = """
#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define MAX_RING_PLANES 16
#define PI 3.14159265358979

in vec3 f_local_pos;
in float f_clip_z;

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
    float u_frame_counter;
    vec3  u_mie_albedo;          // per-channel single-scattering albedo (omega_0)
    float u_refractivity;        // surface (n_mix - 1), drives eclipse refraction
    vec4  u_active_casters[8];
    vec4  u_active_caster_poles_obl[8];
    float u_active_caster_R_minor[8];
    vec4  u_active_caster_atmos[8];
    float u_active_max_bend[8];
    float u_ozone_peak_km;
    float u_ozone_width_km;
    float _atmo_pad0;
    float _atmo_pad1;
};

uniform sampler2D u_ring_gradients;
uniform int u_num_ring_planes;
uniform int u_atmo_quality;
uniform vec3 u_ring_center[MAX_RING_PLANES];
uniform vec3 u_ring_normal[MAX_RING_PLANES];
uniform vec4 u_ring_params[MAX_RING_PLANES];
uniform int u_atmo_clip_mode;

uniform uint u_ring_coplanar_mask[16];
uniform sampler2D u_eclipse_lut; // Kept to avoid uniform bound errors
uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multi_scatter_lut;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform bool u_planetshine_enabled;
uniform int  u_atmo_jitter;

uniform sampler2D u_ringshine_lut;
uniform sampler3D u_ringshine_cdf_lut;
uniform sampler2D u_ringshine_map;
uniform bool u_ringshine_enabled;
uniform int u_ringshine_band_count;

float eval_ringshine_cdf(float angle, float v_tex, float sin_lat) {
    float TWO_PI = 6.28318530717958647692;
    float a_mod = angle - TWO_PI * floor((angle + PI) / TWO_PI);
    float k = floor((angle + PI) / TWO_PI);
    float u = clamp(abs(a_mod) / PI, 0.0, 1.0);

    float u_tex = 0.5 / 128.0 + u * (127.0 / 128.0);
    float v_tex_mapped = 0.5 / 128.0 + v_tex * (127.0 / 128.0);
    float sin_lat_mapped = 0.5 / 64.0 + sin_lat * (63.0 / 64.0);

    float base_cdf = texture(u_ringshine_cdf_lut, vec3(u_tex, v_tex_mapped, sin_lat_mapped)).r;
    float signed_cdf = (a_mod < 0.0) ? -base_cdf : base_cdf;
    return 2.0 * k + signed_cdf;
}

layout(location = 0) out vec4 out_color;

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
        vec3 cross_vec = cross(s_to_c, L_dir);
        float perp_sq = dot(cross_vec, cross_vec);

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
        R_eff = max(R_eff, fwidth(d) * 0.75);

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
            float alpha_mult = textureLod(u_ring_gradients, vec2(p_mid, (float(ring_idx) + 0.5) / 16.0), 0.0).a;
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

float interleaved_gradient_noise(vec2 position, float frame) {
    position += fract(frame * 0.6180339887498949) * vec2(1000.0, 1000.0);
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(position, magic.xy)));
}

void main() {
    if (f_clip_z < 0.0) discard;
    u_ring_mask = floatBitsToUint(instances[u_body_idx * 7 + 3].w);
    vec3 body_planetshine_dir = instances[u_body_idx * 7 + 4].xyz;
    vec3 body_planetshine_color = instances[u_body_idx * 7 + 5].xyz;

    vec3 planet_center_render = u_body_offset;
    vec3 cam_local_au = u_camera_pos - planet_center_render;
    vec3 cam_local = cam_local_au * u_au_to_km;

    vec3 f_pos_local_au = f_local_pos * u_atmo_radius_au;
    vec3 f_pos_local = f_pos_local_au * u_au_to_km;
    vec3 ray_dir = normalize(f_pos_local - cam_local);

    vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
    vec3 cam_local_sph = toSphericalSpace(cam_local, u_pole_obl);

    vec2 s_atmo = raySphereIntersect(cam_local_sph, ray_dir_sph, u_atmo_radius_km);
    if (s_atmo.x > s_atmo.y) discard;

    vec2 s_planet = raySphereIntersect(cam_local_sph, ray_dir_sph, u_planet_radius_km);

    float s_start = max(0.0, s_atmo.x);
    float s_end = s_atmo.y;

    if (s_planet.x > 0.0 && s_planet.x < s_end) {
        s_end = s_planet.x;
    }

    vec3 frag_local = cam_local;

    float closest_s_ring = 1e10;
    for (int k = 0; k < u_num_ring_planes; k++) {
        vec3 ring_center_world_rel = u_ring_center[k];
        // Allow clipping against any ring plane in the system to support moon atmospheres with host planet rings behind them
        // if (length(ring_center_world_rel - u_body_offset) > 1e-4) continue;

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

                float dr = fwidth(dist_from_center) * 1.5;
                if (dist_from_center >= inner_r_km - dr && dist_from_center <= outer_r_km + dr) {
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

    if (s_start >= s_end) discard;

    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;       // Mie extinction
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;            // Mie scattering = omega_0 * extinction
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M); // Mie absorption
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

        vec3 global_eclipse_shadow = vec3(1.0);
        vec3 end_eclipse_shadow = vec3(1.0);
        bool skip_volumetric_shadow = false;

        vec4 ring_s1_out = vec4(1e9);
        vec4 ring_s2_out = vec4(-1e9);
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

                vec3 start_render = O + s_start * V;
                vec3 end_render = O + s_end * V;
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

        vec3 sun_dir_local = normalize(sun_pos_local - mid_pos);
        vec3 sun_dir_sph_const = normalize(toSphericalSpace(sun_dir_local, u_pole_obl));

        float alpha_sun_local = asin(clamp(sin_star, 0.0, 0.9999));
        float cos_sun = cos(alpha_sun_local);
        float sin_sun = sin_star;

        float jitter = (u_atmo_jitter != 0) ? interleaved_gradient_noise(gl_FragCoord.xy, u_frame_counter) : 0.5;
        vec3 step_dir_sph = ray_dir_sph * step_size;
        vec3 current_pos_sph = cam_local_sph + (s_start + jitter * step_size) * ray_dir_sph;

        vec3 total_rayleigh = vec3(0.0);
        vec3 total_mie = vec3(0.0);
        vec3 total_ms = vec3(0.0);
        vec3 current_transmittance = vec3(1.0);

        vec3 total_rayleigh_ps = vec3(0.0);
        vec3 total_mie_ps = vec3(0.0);

        vec3 total_rayleigh_rs = vec3(0.0);
        vec3 total_mie_rs = vec3(0.0);

        float current_s = s_start + jitter * step_size;
        float t_lerp = jitter / float(steps);
        float t_step = 1.0 / float(steps);

        float inv_atmo_thickness = 1.0 / max(1e-4, u_atmo_radius_km - u_planet_radius_km);

        for (int i = 0; i < steps; i++) {
            float sample_len = length(current_pos_sph);
            float altitude = sample_len - u_planet_radius_km;

            float rho_R = exp(-altitude / u_h_rayleigh);
            float rho_M = exp(-altitude / u_h_mie);
            float rho_O = exp(-pow((altitude - u_ozone_peak_km) / max(u_ozone_width_km, 1e-3), 2.0));

            vec3 step_extinction = beta_R * rho_R + beta_M * rho_M + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;
            vec3 step_transmittance = exp(-step_extinction * step_size);
            vec3 int_factor = (vec3(1.0) - step_transmittance) / max(step_extinction, 1e-6);

            vec3 sun_dir_sph = sun_dir_sph_const;
            float light_cos_theta = dot(current_pos_sph, sun_dir_sph) / sample_len;
            float h_norm = clamp(altitude * inv_atmo_thickness, 0.0, 1.0);

            float sin_planet = u_planet_radius_km / max(sample_len, u_planet_radius_km + 0.01);
            float cos_planet = sqrt(max(0.0, 1.0 - sin_planet * sin_planet));

            // Atmospheric refraction limit (max bending angle) for this body.
            // Derived from the surface refractivity (n_mix - 1) instead of a
            // hard-coded Earth-only 0.00029 constant, so Venus/Mars/Titan/gas
            // giants refract eclipses according to their own gas mixtures.
            float max_bend = clamp(2.0 * max(u_refractivity, 0.0) * sqrt(3.14159265359 * u_planet_radius_km / max(1e-6, u_h_rayleigh * 2.0)), 0.001, 0.05);
            float effective_star_rad = sin_star + max_bend;
            float cos_sun_eff = sqrt(max(0.0, 1.0 - effective_star_rad * effective_star_rad));

            float cos_outer = cos_planet * cos_sun_eff - sin_planet * effective_star_rad;
            float cos_inner = cos_planet * cos_sun_eff + sin_planet * effective_star_rad;

            float neg_light_cos = -light_cos_theta;
            float vis_fraction = 1.0;

            if (neg_light_cos > cos_inner) {
                vis_fraction = 0.0;
            } else if (neg_light_cos > cos_outer) {
                float x_vis = (cos_planet * cos_sun_eff - neg_light_cos) / max(1e-7, sin_planet * effective_star_rad);
                vis_fraction = smoothstep(-1.0, 1.0, x_vis);
            }

            float disc_top_cos = min(light_cos_theta + effective_star_rad, 1.0);
            float disc_bot_cos = max(light_cos_theta - effective_star_rad, -cos_planet + 1e-5);
            float effective_cos = max(-cos_planet + 1e-5, (disc_top_cos + disc_bot_cos) * 0.5);

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
                        if (valid.x > 0.0) sh_mult.x = 1.0 - frac.x * (1.0 - exp(-ring_opac.x * textureLod(u_ring_gradients, vec2(p_mid.x, ring_v_coord.x), 0.0).a));
                        if (valid.y > 0.0) sh_mult.y = 1.0 - frac.y * (1.0 - exp(-ring_opac.y * textureLod(u_ring_gradients, vec2(p_mid.y, ring_v_coord.y), 0.0).a));
                        if (valid.z > 0.0) sh_mult.z = 1.0 - frac.z * (1.0 - exp(-ring_opac.z * textureLod(u_ring_gradients, vec2(p_mid.z, ring_v_coord.z), 0.0).a));
                        if (valid.w > 0.0) sh_mult.w = 1.0 - frac.w * (1.0 - exp(-ring_opac.w * textureLod(u_ring_gradients, vec2(p_mid.w, ring_v_coord.w), 0.0).a));

                        sample_shadow *= sh_mult.x * sh_mult.y * sh_mult.z * sh_mult.w;
                    }
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

            if (u_planetshine_enabled) {
                float light_cos_theta_ps = dot(current_pos_sph, body_planetshine_dir) / sample_len;
                float vis_fraction_ps = smoothstep(-cos_planet - 0.05, -cos_planet + 0.05, light_cos_theta_ps);
                vec3 transmittance_to_ps = get_transmittance_precomputed(v_norm, light_cos_theta_ps);
                vec3 ps_attenuation = current_transmittance * transmittance_to_ps * vis_fraction_ps;
                total_rayleigh_ps += rho_R * ps_attenuation * int_factor;
                total_mie_ps      += rho_M * ps_attenuation * int_factor;
            }

            if (u_ringshine_enabled) {
                vec3 rs_attenuation = current_transmittance;
                total_rayleigh_rs += rho_R * rs_attenuation * int_factor;
                total_mie_rs      += rho_M * rs_attenuation * int_factor;
            }

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

        float irradiance = star_lum / max(dist_to_star_au * dist_to_star_au, 1e-8);

        scattered += star_color * u_sun_intensity * irradiance * (
            phase_R * beta_R * total_rayleigh +
            phase_M * beta_M * total_mie +
            total_ms
        );

        if (u_planetshine_enabled && dot(body_planetshine_color, body_planetshine_color) > 1e-12) {
            float cos_theta_ps = dot(ray_dir, body_planetshine_dir);
            float phase_R_ps = (3.0 / (16.0 * PI)) * (1.0 + cos_theta_ps * cos_theta_ps);
            float phase_M_ps = (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta_ps * cos_theta_ps))
                          / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_theta_ps, 1.5));
            scattered += body_planetshine_color * (
                phase_R_ps * beta_R * total_rayleigh_ps +
                phase_M_ps * beta_M * total_mie_ps
            );
        }

        if (u_ringshine_enabled && u_num_ring_planes > 0) {
            vec3 ringshine_irradiance = vec3(0.0);
            vec3 N = normalize(mid_pos);
            vec3 L_dir = L_mid;

            vec3 ring_normal = u_ring_normal[0];
            float sun_elevation = dot(L_dir, ring_normal);
            float sin_sun_elev = clamp(abs(sun_elevation), 1e-4, 1.0);
            float cos_sun_elev = sqrt(1.0 - sin_sun_elev * sin_sun_elev);

            vec3 antiL = -L_dir;
            vec3 antiL_eq_raw = antiL - ring_normal * dot(antiL, ring_normal);
            float len_antiL_eq = length(antiL_eq_raw);
            vec3 antiL_eq = len_antiL_eq > 1e-5 ? antiL_eq_raw / len_antiL_eq : vec3(-1.0, 0.0, 0.0);

            vec3 N_eq_raw = N - ring_normal * dot(N, ring_normal);
            float len_N_eq = length(N_eq_raw);
            vec3 N_eq = len_N_eq > 1e-5 ? N_eq_raw / len_N_eq : vec3(1.0, 0.0, 0.0);

            vec3 cross_rel = cross(antiL_eq, N_eq);
            float sin_rel = dot(cross_rel, ring_normal);
            float cos_rel = clamp(dot(N_eq, antiL_eq), -1.0, 1.0);
            float phi_center = atan(sin_rel, cos_rel);

            float frag_elevation = dot(N, ring_normal);
            float sin_lat = clamp(abs(frag_elevation), 0.001, 0.999);
            float same_hemisphere = sun_elevation * frag_elevation;
            float same_hemi_t = smoothstep(-0.02, 0.02, same_hemisphere);

            float NdotL = dot(N, L_dir);
            float day_face = smoothstep(0.0, 0.1, NdotL) * max(0.0, NdotL);
            float noon_fade = mix(1.0, 0.4, day_face);
            float face_multiplier = mix(1.0, 1.0 * noon_fade, same_hemi_t);

            float host_radius = u_planet_radius_km / u_au_to_km;
            int band_count = clamp(u_ringshine_band_count, 4, 128);

            for (int j = 0; j < 1; j++) {
                if ((u_ring_mask & (1u << j)) == 0u) continue;
                float x_prime = phi_center / PI;
                float phi_uv = sign(x_prime) * pow(abs(x_prime), 0.666666667) * 0.5 + 0.5;

                float y_prime = frag_elevation;
                float elev_uv = sign(y_prime) * pow(abs(y_prime), 0.666666667) * 0.5 + 0.5;

                vec2 map_uv = vec2(phi_uv, (float(j) + elev_uv) / 16.0);
                ringshine_irradiance += texture(u_ringshine_map, map_uv).rgb;
            }
            float opacity = u_ring_params[0].z;
            ringshine_irradiance *= (sin_sun_elev * face_multiplier * 0.318309886);

            float ambient_phase = 1.0 / (4.0 * PI);
            scattered += star_color * u_sun_intensity * irradiance * ringshine_irradiance * (
                ambient_phase * beta_R * total_rayleigh_rs +
                ambient_phase * beta_M * total_mie_rs
            );
        }

        final_transmittance = current_transmittance;
    }

    vec3 transmittance = final_transmittance;

    if (u_hdr_enabled) {
        scattered *= u_exposure;
    }

    out_color = vec4(scattered, 1.0 - clamp((transmittance.r + transmittance.g + transmittance.b) / 3.0, 0.0, 1.0));
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
uniform vec3 u_beta_rayleigh;
uniform vec3 u_beta_mie;
uniform vec3 u_beta_abs_mixed;
uniform vec3 u_beta_abs_layered;
uniform vec3 u_mie_albedo;
uniform float u_ozone_peak_km = 25.0;
uniform float u_ozone_width_km = 8.0;

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
        od_ozone += exp(-pow((h_sample - u_ozone_peak_km) / max(u_ozone_width_km, 1e-3), 2.0)) * step_size;
    }

    vec3 beta_R = u_beta_rayleigh * 1000.0;
    vec3 beta_M_ext = u_beta_mie * 1000.0;
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;              // Mie scattering
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M); // Mie absorption
    vec3 beta_A_mixed = u_beta_abs_mixed * 1000.0;
    vec3 beta_A_layered = u_beta_abs_layered * 1000.0;

    vec3 transmittance = exp(-(beta_R * od_rayleigh + beta_M_ext * od_mie + beta_A_mixed * od_rayleigh + beta_A_layered * od_ozone));
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
uniform vec3 u_mie_albedo;
uniform float u_mie_g;
uniform vec3 u_ground_albedo;
uniform float u_ozone_peak_km = 25.0;
uniform float u_ozone_width_km = 8.0;

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
    vec3 beta_M_ext = u_beta_mie * 1000.0;
    vec3 w0_M = clamp(u_mie_albedo, vec3(0.0), vec3(1.0));
    vec3 beta_M = beta_M_ext * w0_M;              // Mie scattering
    vec3 beta_M_abs = beta_M_ext * (vec3(1.0) - w0_M); // Mie absorption
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
                float rho_O = exp(-pow((h_sample - u_ozone_peak_km) / max(u_ozone_width_km, 1e-3), 2.0));

                vec3 scattering = beta_R * rho_R + beta_M * rho_M;
                vec3 extinction = scattering + beta_M_abs * rho_M + beta_A_mixed * rho_R + beta_A_layered * rho_O;

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
                vec3 effective_ground_albedo = max(u_ground_albedo, w0_M * clamp((beta_M * u_h_mie - vec3(0.5)) / 2.0, vec3(0.0), vec3(1.0)));
                vec3 ground_lum = (effective_ground_albedo / 3.14159265358979) * NdotL * trans_to_sun;

                lum += transmittance_accum * ground_lum;
                fms += transmittance_accum * effective_ground_albedo;
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


ringshine_map_vertex_shader = """
#version 460 core
in vec2 in_position;
out vec2 f_uv;
void main() {
    f_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

ringshine_map_fragment_shader = """
#version 460 core
#define PI 3.14159265358979323846
in vec2 f_uv;
out vec4 out_color;

uniform sampler2D u_ring_gradients;
uniform sampler2D u_ringshine_lut;
uniform sampler3D u_ringshine_cdf_lut;

uniform vec3 u_sun_dir;
uniform int u_num_ring_planes;
uniform vec3 u_ring_normal[16];
uniform vec4 u_ring_params[16];
uniform int u_ringshine_band_count;

float eval_ringshine_cdf(float angle, float v_tex, float sin_lat) {
    float TWO_PI = 6.28318530717958647692;
    float a_mod = angle - TWO_PI * floor((angle + PI) / TWO_PI);
    float k = floor((angle + PI) / TWO_PI);
    float u = clamp(abs(a_mod) / PI, 0.0, 1.0);

    float u_tex = 0.5 / 128.0 + u * (127.0 / 128.0);
    float v_tex_mapped = 0.5 / 128.0 + v_tex * (127.0 / 128.0);
    float sin_lat_mapped = 0.5 / 64.0 + sin_lat * (63.0 / 64.0);

    float base_cdf = texture(u_ringshine_cdf_lut, vec3(u_tex, v_tex_mapped, sin_lat_mapped)).r;
    float signed_cdf = (a_mod < 0.0) ? -base_cdf : base_cdf;
    return 2.0 * k + signed_cdf;
}

void main() {
    float slot = f_uv.y * 16.0;
    int k = int(floor(slot));
    float local_v = fract(slot);

    if (k >= u_num_ring_planes) {
        out_color = vec4(0.0);
        return;
    }

    vec3 ring_normal = u_ring_normal[k];
    float inner_r = u_ring_params[k].x;
    float outer_r = u_ring_params[k].y;
    float opacity = u_ring_params[k].z;
    float host_radius = u_ring_params[k].w > 1e-5 ? u_ring_params[k].w : inner_r * 0.7;

    if (opacity < 1e-4 || outer_r <= inner_r) {
        out_color = vec4(0.0);
        return;
    }

    float x = f_uv.x * 2.0 - 1.0;
    float phi_center = sign(x) * pow(abs(x), 1.5) * PI;

    float y = local_v * 2.0 - 1.0;
    float frag_elevation = sign(y) * pow(abs(y), 1.5);
    float sin_lat = clamp(abs(frag_elevation), 0.001, 0.999);

    vec3 L = u_sun_dir;
    float sun_elevation = dot(L, ring_normal);
    float sin_sun_elev = clamp(abs(sun_elevation), 1e-4, 1.0);
    float cos_sun_elev = sqrt(max(0.0, 1.0 - sin_sun_elev * sin_sun_elev));

    float same_hemisphere = sun_elevation * frag_elevation;
    float same_hemi_t = smoothstep(-0.02, 0.02, same_hemisphere);

    int band_count = clamp(u_ringshine_band_count, 4, 1024);
    vec3 total_ring_irradiance = vec3(0.0);
    float log_r_ratio = log(max(1.001, outer_r / max(1e-5, inner_r)));

    for (int m = 0; m < band_count; m++) {
        float u0 = float(m) / float(band_count);
        float u1 = float(m + 1) / float(band_count);
        float u_mid = 0.5 * (u0 + u1);

        float r_m0 = inner_r * exp(u0 * log_r_ratio);
        float r_m1 = inner_r * exp(u1 * log_r_ratio);
        float r_mid = inner_r * exp(u_mid * log_r_ratio);
        float dr = (r_m1 - r_m0) / max(1e-5, host_radius);
        float frac_mid = clamp((r_mid - inner_r) / max(1e-5, outer_r - inner_r), 0.0, 1.0);

        vec4 ring_texel = texture(u_ring_gradients, vec2(frac_mid, (float(k) + 0.5) / 16.0));
        if (ring_texel.a < 1e-4) continue;

        float alpha_phys = clamp(ring_texel.a * opacity, 0.0, 0.999);
        float tau_phys = -log(max(1e-4, 1.0 - alpha_phys));
        float trans_factor = exp(-0.35 * tau_phys / max(sin_sun_elev, 0.087));

        vec3 band_color_sunlit = ring_texel.rgb * alpha_phys * 2.0;
        vec3 band_color_unlit = ring_texel.rgb * trans_factor * alpha_phys * 0.4;
        vec3 band_color = mix(band_color_unlit, band_color_sunlit, same_hemi_t);

        float norm_r = r_mid / max(1e-5, host_radius);

        float delta_alpha_shadow = 0.0;
        if (norm_r <= 1.0 / sin_sun_elev) {
            float arg = sqrt(max(0.0, 1.0 - 1.0 / (norm_r * norm_r))) / max(1e-4, cos_sun_elev);
            delta_alpha_shadow = acos(clamp(arg, 0.0, 1.0));
        }

        float psi1 = -delta_alpha_shadow - phi_center;
        float psi2 = +delta_alpha_shadow - phi_center;
        float v_tex = clamp((norm_r - 1.0) / 4.0, 0.0, 1.0);

        float cdf1 = eval_ringshine_cdf(psi1, v_tex, sin_lat);
        float cdf2 = eval_ringshine_cdf(psi2, v_tex, sin_lat);

        float shadow_fraction = clamp(0.5 * (cdf2 - cdf1), 0.0, 1.0);
        float band_illum = max(0.0, 1.0 - shadow_fraction);

        float u_tex_lut = 0.5 / 256.0 + sin_lat * (255.0 / 256.0);
        float v_tex_lut = 0.5 / 256.0 + v_tex * (255.0 / 256.0);
        float kernel_val = texture(u_ringshine_lut, vec2(u_tex_lut, v_tex_lut)).r;

        total_ring_irradiance += band_color * kernel_val * dr * band_illum;
    }

    out_color = vec4(total_ring_irradiance, 1.0);
}
"""
