#version 460 core
in vec3 in_position; // xy in [0, 1]x[0, 1] grid patch coords; z is skirt_flag (0.0=surface, 1.0=skirt)

#define MAX_STARS 16
#define MAX_CASTERS 64

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
    vec4 u_caster_ozone[MAX_CASTERS];
    vec4 u_caster_ozone_vert[MAX_CASTERS];
};

layout(std430, binding = 2) readonly buffer AllInstances {
    vec4 instances[];
};

struct TerrainPatchInstance {
    vec4 u_range;      // x=min_u, y=min_v, z=max_u, w=max_v in [-1, 1]
    vec4 u_uv_trans;   // x=uv_scale, y=uv_offset_x, z=uv_offset_y, w=skirt_depth_km
    vec4 u_meta;       // x=face_idx, y=lod_level, z=tile_slot, w=body_idx
    vec4 u_height;     // slot (-1 disables), independent UV scale and offset
    vec4 u_hrange;     // minimum elevation (km), elevation span (km), padding
};

layout(std430, binding = 4) readonly buffer TerrainPatchBuffer {
    TerrainPatchInstance u_patches[];
};

uniform float u_km_to_au; // 1.0 / 149597870.7
uniform bool u_is_cloud_pass;
uniform sampler2DArray u_tile_array;
uniform float u_cloud_altitude_km;
uniform vec3 u_camera_pos;

#include "common/refraction.glsl"

out vec3 f_world_pos;
out vec3 f_normal;
out vec3 f_tan_u;
out vec3 f_tan_v;
out vec2 f_height_uv;
flat out vec3 f_height_meta; // slot, patch-to-tile UV scale, elevation span (km)
out vec2 f_tile_uv;
out vec2 f_local_uv;
flat out float f_tile_slot;
flat out float f_lod_level;
out float f_clip_z;

flat out vec3 f_atmo_tint;
flat out float f_atmo_h;
flat out float f_scale_height;
flat out vec3 f_o3_tau;
flat out vec2 f_o3_layer;
flat out float f_radius_km;

flat out uvec2 f_caster_mask;
flat out uint f_ring_mask;
flat out vec3 f_planetshine_dir;
flat out vec3 f_planetshine_color;
flat out vec3 f_body_center;
out vec3 f_rel_pos;

#define PI 3.14159265358979323846

vec3 base_surface(int face, vec2 uv, float obl) {
    vec2 p = tan(uv * (PI * 0.25));
    vec3 v;
    if (face == 0)      v = vec3(1.0, -p.y, -p.x);
    else if (face == 1) v = vec3(-1.0, -p.y, p.x);
    else if (face == 2) v = vec3(p.x, 1.0, p.y);
    else if (face == 3) v = vec3(p.x, -1.0, -p.y);
    else if (face == 4) v = vec3(p.x, -p.y, 1.0);
    else               v = vec3(-p.x, -p.y, -1.0);
    return normalize(v) * vec3(1.0, 1.0 - obl, 1.0);
}

void main() {
    TerrainPatchInstance t_inst = u_patches[gl_InstanceID];

    // Unpack instance information for this body from AllInstances
    uint inst_idx = uint(max(0.0, t_inst.u_meta.w + 0.5));
    vec4 f0 = instances[inst_idx * 7 + 0];
    vec4 f1 = instances[inst_idx * 7 + 1];
    vec4 f2 = instances[inst_idx * 7 + 2];
    vec4 f3 = instances[inst_idx * 7 + 3];
    vec4 f4 = instances[inst_idx * 7 + 4];
    vec4 f5 = instances[inst_idx * 7 + 5];
    vec4 f6 = instances[inst_idx * 7 + 6];

    vec3 body_pos = f0.xyz;
    float r_au = f1.z;
    float r_km = r_au * (1.0 / max(1e-12, u_km_to_au));
    vec3 pole = normalize(f2.yzw);
    float obl = clamp(f3.x, 0.0, 0.8);
    float rot_angle = f6.y;

    // Local patch cube coordinate in [-1, 1]
    float u_local = mix(t_inst.u_range.x, t_inst.u_range.z, in_position.x);
    float v_local = mix(t_inst.u_range.y, t_inst.u_range.w, in_position.y);

    // Tangent warping to minimize area distortion
    float x_p = tan(u_local * (PI * 0.25));
    float y_p = tan(v_local * (PI * 0.25));

    // Cube face vector
    vec3 v;
    int face = int(t_inst.u_meta.x + 0.5);
    if (face == 0)      v = vec3(1.0, -y_p, -x_p);
    else if (face == 1) v = vec3(-1.0, -y_p, x_p);
    else if (face == 2) v = vec3(x_p, 1.0, y_p);
    else if (face == 3) v = vec3(x_p, -1.0, -y_p);
    else if (face == 4) v = vec3(x_p, -y_p, 1.0);
    else                v = vec3(-x_p, -y_p, -1.0);

    vec3 n_sphere = normalize(v);

    // Oblate spheroid shape in body local frame (Y is polar axis)
    vec3 p_ellip = vec3(n_sphere.x, n_sphere.y * (1.0 - obl), n_sphere.z);
    if (u_is_cloud_pass) {
        r_km += max(0.1, u_cloud_altitude_km);
    }
    vec3 p_local_km = p_ellip * r_km;
    bool height_active = !u_is_cloud_pass && t_inst.u_height.x >= 0.0;
    f_height_uv = in_position.xy * t_inst.u_height.y + t_inst.u_height.zw;
    f_height_meta = vec3(height_active ? t_inst.u_height.x : -1.0,
                         t_inst.u_height.y, t_inst.u_hrange.y);
    if (height_active) {
        float e = textureLod(u_tile_array, vec3(f_height_uv, t_inst.u_height.x), 0.0).r;
        p_local_km += normalize(p_ellip) * (t_inst.u_hrange.x + e * t_inst.u_hrange.y);
    }

    // Boundary skirt extrusion (only for ground terrain, never for clouds)
    if (!u_is_cloud_pass && in_position.z > 0.5) {
        float skirt_depth = t_inst.u_uv_trans.w;
        if (height_active) skirt_depth += t_inst.u_hrange.y;
        p_local_km -= normalize(p_ellip) * skirt_depth;
    }

    // Apply planet spin rotation from texture frame back to body local frame
    float s_rot = sin(rot_angle);
    float c_rot = cos(rot_angle);
    vec3 p_local_body = vec3(
        p_local_km.x * c_rot - p_local_km.z * s_rot,
        p_local_km.y,
        p_local_km.x * s_rot + p_local_km.z * c_rot
    );

    // Normal in rotated local body frame
    vec3 n_local_body = normalize(vec3(
        n_sphere.x * c_rot - n_sphere.z * s_rot,
        n_sphere.y / max(1e-4, 1.0 - obl),
        n_sphere.x * s_rot + n_sphere.z * c_rot
    ));

    // Rotate from planet frame to world frame using pole (matching sphere basis)
    vec3 ref = vec3(0.0, 1.0, 0.0);
    if (abs(dot(pole, ref)) > 0.999) {
        ref = vec3(1.0, 0.0, 0.0);
    }
    vec3 tangent = normalize(cross(pole, ref));
    vec3 bitangent = normalize(cross(pole, tangent));

    vec3 p_world_km = p_local_body.x * tangent + p_local_body.y * pole + p_local_body.z * bitangent;
    vec3 n_world = normalize(n_local_body.x * tangent + n_local_body.y * pole + n_local_body.z * bitangent);

    f_tan_u = vec3(0.0);
    f_tan_v = vec3(0.0);
    if (height_active) {
        // Derivatives in km per PATCH UV unit, from the undisplaced surface.
        // Keep their lengths: normalizing them would destroy physical slope scale.
        const float step_uv = 1e-3;
        vec2 uv = vec2(u_local, v_local);
        vec2 extent = t_inst.u_range.zw - t_inst.u_range.xy;
        vec3 pu = (base_surface(face, uv + vec2(step_uv, 0.0), obl)
                 - base_surface(face, uv - vec2(step_uv, 0.0), obl)) * (r_km * extent.x / (2.0 * step_uv));
        vec3 pv = (base_surface(face, uv + vec2(0.0, step_uv), obl)
                 - base_surface(face, uv - vec2(0.0, step_uv), obl)) * (r_km * extent.y / (2.0 * step_uv));
        vec3 base_n = normalize(vec3(n_sphere.x, n_sphere.y / max(1e-4, 1.0 - obl), n_sphere.z));
        pu -= base_n * dot(pu, base_n);
        pv -= base_n * dot(pv, base_n);
        // Preserve +u/+v derivative directions; orient the final cross in frag.
        mat3 spin = mat3(c_rot, 0.0, s_rot, 0.0, 1.0, 0.0, -s_rot, 0.0, c_rot);
        mat3 frame = mat3(tangent, pole, bitangent);
        f_tan_u = frame * spin * pu;
        f_tan_v = frame * spin * pv;
    }

    // Convert km to world AU and add body center
    vec3 p_world = body_pos + p_world_km * u_km_to_au;

    // Apply atmospheric refraction and gravitational lensing
    bool is_refract_host = (length(body_pos - u_refract_center) < 1e-7);
    if (!is_refract_host) {
        // Background celestial body viewed through a foreground atmosphere or black hole
        p_world = apply_refraction(p_world, u_camera_pos);
    } else {
        if (u_refract_max_bend > 1e-6) {
            // Terrestrial Refraction for host planet ground/clouds (horizon extension)
            vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
            float r_cam = length(C_km);

            vec3 pole_n_refr = pole;
            float f_scale_refr = (obl > 0.0 && obl < 0.99) ? (1.0 / (1.0 - obl)) : 1.0;
            vec3 C_scaled = C_km + pole_n_refr * (dot(C_km, pole_n_refr) * (f_scale_refr * f_scale_refr - 1.0));
            vec3 local_up = normalize(C_scaled);

            float local_refract_radius = u_refract_radius;
            if (obl > 0.001 && obl < 0.99) {
                vec3 P_dir = r_cam > 1e-6 ? (C_km / r_cam) : vec3(0.0, 1.0, 0.0);
                float cos_t = abs(dot(P_dir, pole_n_refr));
                float k = 1.0 / (1.0 - obl);
                float denom = sqrt(max(1e-6, 1.0 + (k * k - 1.0) * cos_t * cos_t));
                local_refract_radius = u_refract_radius / denom;
            }

            float h = r_cam - local_refract_radius;
            if (h < u_refract_scale_height * 15.0) {
                float density = exp(-max(h, 0.0) / max(1e-4, u_refract_scale_height));
                float k_refr = u_refract_max_bend * 0.5 * sqrt((2.0 * local_refract_radius) / max(1e-4, 3.141592653589793 * u_refract_scale_height));
                float h_eff = max(h, 0.002);
                float theta_dip_eff = sqrt(2.0 * h_eff / local_refract_radius);
                float max_terr_alpha = clamp(0.5 * k_refr * density * theta_dip_eff, 0.0, 0.05);

                if (max_terr_alpha > 1e-7) {
                    vec3 view_vec = p_world - u_camera_pos;
                    float d_v = length(view_vec);
                    if (d_v > 1e-7) {
                        float d_v_km = d_v * u_au_to_km;
                        vec3 view_ray = view_vec / d_v;
                        float mu = dot(view_ray, local_up);

                        // Physical geodetic terrestrial refraction: deflection scales with line-of-sight distance d_v
                        // across terrain, smoothly saturating at the horizon dip angle max_terr_alpha:
                        float alpha_dist = 0.5 * k_refr * density * (d_v_km / max(1e-4, local_refract_radius));
                        float alpha = min(alpha_dist, max_terr_alpha);

                        // For sightlines above horizontal (mu > 0), fade smoothly towards zenith:
                        if (mu > 0.0) {
                            float cos_e = sqrt(max(0.0, 1.0 - mu * mu));
                            alpha *= cos_e;
                        }

                        if (alpha > 1e-7) {
                            vec3 u_dir = local_up - view_ray * mu;
                            float u_len = length(u_dir);
                            if (u_len > 1e-5) {
                                u_dir /= u_len;
                                // Rotate view_ray towards local_up (lifting apparent position of terrain upward):
                                vec3 app_ray = normalize(view_ray * cos(alpha) + u_dir * sin(alpha));
                                p_world = u_camera_pos + app_ray * d_v;
                            }
                        }
                    }
                }
            }
        }

        // Gravitational lensing for host planet if orbiting near a black hole
        if (u_grav_lens_enabled && u_grav_lens_rs > 1e-6 && length(body_pos - u_grav_lens_center) > 1e-7) {
            vec3 C_km = (u_camera_pos - u_grav_lens_center) * u_au_to_km;
            vec3 P_km = (p_world - u_grav_lens_center) * u_au_to_km;
            vec3 true_vec = P_km - C_km;
            float d_km = length(true_vec);
            if (d_km > 1e-5) {
                vec3 V = true_vec / d_km;
                bool is_shadow = false;
                vec3 V_app = apply_gravitational_deflection(C_km, V, d_km, is_shadow);
                if (!is_shadow) {
                    p_world = u_camera_pos + V_app * (d_km / u_au_to_km);
                }
            }
        }
    }

    f_world_pos = p_world;
    f_normal = n_world;

    // UV coordinates:
    // Hierarchical UV remapped into ancestor tile if child not yet loaded
    f_tile_uv = in_position.xy * t_inst.u_uv_trans.x + t_inst.u_uv_trans.yz;
    f_local_uv = in_position.xy;
    f_tile_slot = t_inst.u_meta.z;
    f_lod_level = t_inst.u_meta.y;

    // Match atmospheric parameters from SceneData for this body
    vec3 atmo_tint = vec3(0.0);
    float atmo_h = 0.0;
    float scale_height = 0.0;
    vec3 o3_tau = vec3(0.0);
    vec2 o3_layer = vec2(0.0, 6.0);
    for (int j = 0; j < u_num_casters; j++) {
        if (distance(u_casters[j].xyz, body_pos) < 1e-6) {
            atmo_tint = u_caster_atmos[j].xyz;
            atmo_h = u_caster_atmos[j].w;
            scale_height = u_caster_colors[j].w;
            o3_tau = u_caster_ozone_vert[j].xyz;
            o3_layer = vec2(u_caster_ozone[j].w, max(u_caster_ozone_vert[j].w, 0.1));
            break;
        }
    }
    f_atmo_tint = atmo_tint;
    f_atmo_h = atmo_h;
    f_scale_height = scale_height;
    f_o3_tau = o3_tau;
    f_o3_layer = o3_layer;
    f_radius_km = r_km;

    f_caster_mask = uvec2(floatBitsToUint(f3.y), floatBitsToUint(f3.z));
    f_ring_mask = floatBitsToUint(f3.w);
    f_planetshine_dir = f4.xyz;
    f_planetshine_color = f5.xyz;

    f_body_center = body_pos;
    f_rel_pos = p_world_km * u_km_to_au;

    gl_Position = projection * view * vec4(p_world, 1.0);
    f_clip_z = gl_Position.w;

    // Logarithmic depth buffer
    gl_Position.z = log2(max(1e-6, 1.0 + gl_Position.w * u_depth_C)) * (2.0 / log2(u_far * u_depth_C + 1.0)) - 1.0;
    gl_Position.z *= gl_Position.w;
}
