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
    vec4 u_meta;       // x=face_idx, y=lod_level, z=tile_slot, w=radius_km
    vec4 u_body_pos;   // xyz=body_center_au, w=oblateness
    vec4 u_pole;       // xyz=pole_dir, w=rotation_angle
    vec4 u_extra;      // x=body_idx, yzw=unused
};

layout(std430, binding = 4) readonly buffer TerrainPatchBuffer {
    TerrainPatchInstance u_patches[];
};

uniform float u_km_to_au; // 1.0 / 149597870.7
uniform bool u_is_cloud_pass;
uniform float u_cloud_altitude_km;

out vec3 f_world_pos;
out vec3 f_normal;
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

void main() {
    TerrainPatchInstance t_inst = u_patches[gl_InstanceID];

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
    float obl = clamp(t_inst.u_body_pos.w, 0.0, 0.8);
    vec3 p_ellip = vec3(n_sphere.x, n_sphere.y * (1.0 - obl), n_sphere.z);
    float r_km = t_inst.u_meta.w;
    if (u_is_cloud_pass) {
        r_km += max(0.1, u_cloud_altitude_km);
    }
    vec3 p_local_km = p_ellip * r_km;

    // Boundary skirt extrusion (only for ground terrain, never for clouds)
    if (!u_is_cloud_pass && in_position.z > 0.5) {
        float skirt_depth = t_inst.u_uv_trans.w;
        p_local_km -= normalize(p_ellip) * skirt_depth;
    }

    // Apply planet spin rotation from texture frame back to body local frame
    float rot_angle = t_inst.u_pole.w;
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
    vec3 pole = normalize(t_inst.u_pole.xyz);
    vec3 ref = vec3(0.0, 1.0, 0.0);
    if (abs(dot(pole, ref)) > 0.999) {
        ref = vec3(1.0, 0.0, 0.0);
    }
    vec3 tangent = normalize(cross(pole, ref));
    vec3 bitangent = normalize(cross(pole, tangent));

    vec3 p_world_km = p_local_body.x * tangent + p_local_body.y * pole + p_local_body.z * bitangent;
    vec3 n_world = normalize(n_local_body.x * tangent + n_local_body.y * pole + n_local_body.z * bitangent);

    // Convert km to world AU and add body center
    vec3 p_world = t_inst.u_body_pos.xyz + p_world_km * u_km_to_au;

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
        if (distance(u_casters[j].xyz, t_inst.u_body_pos.xyz) < 1e-6) {
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
    f_radius_km = t_inst.u_meta.w;

    // Unpack instance information for this body from AllInstances
    uint inst_idx = uint(max(0.0, t_inst.u_extra.x + 0.5));
    vec4 f3 = instances[inst_idx * 7 + 3];
    vec4 f4 = instances[inst_idx * 7 + 4];
    vec4 f5 = instances[inst_idx * 7 + 5];

    f_caster_mask = uvec2(floatBitsToUint(f3.y), floatBitsToUint(f3.z));
    f_ring_mask = floatBitsToUint(f3.w);
    f_planetshine_dir = f4.xyz;
    f_planetshine_color = vec3(f4.w, f5.x, f5.y);

    f_body_center = t_inst.u_body_pos.xyz;
    f_rel_pos = p_world_km * u_km_to_au;

    gl_Position = projection * view * vec4(p_world, 1.0);
    f_clip_z = gl_Position.w;

    // Logarithmic depth buffer
    gl_Position.z = log2(max(1e-6, 1.0 + gl_Position.w * u_depth_C)) * (2.0 / log2(u_far * u_depth_C + 1.0)) - 1.0;
    gl_Position.z *= gl_Position.w;
}
