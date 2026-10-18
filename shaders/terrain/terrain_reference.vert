#version 460

#include "common/celestial_scene.glsl"
#include "common/terrain_geometry.glsl"

struct PatchGPU {
    dvec4 anchor_radius;
    dvec4 anchor_polar_ratio;
    dvec4 rotation_height;
    dvec4 rotation_error;
    dvec4 rotation_reserved;
    dvec4 eye;
    uvec4 address;
    uvec4 metadata;
    dvec4 eye_anchor;
    dvec4 eye_rotation0;
    dvec4 eye_rotation1;
    dvec4 eye_rotation2;
};

layout(std430, set = 0, binding = 3) readonly buffer TerrainPatches {
    PatchGPU patches[];
};

layout(location = 0) out vec3 f_world_pos;
layout(location = 1) out vec3 f_local_pos;
layout(location = 2) out vec3 f_normal;
layout(location = 3) out float f_subpixel_factor;
layout(location = 4) flat out uint f_body_idx;
layout(location = 5) flat out vec4 f_tile_color;
layout(location = 6) out vec2 f_tile_uv;
layout(location = 7) out vec3 f_base_chord;

dvec3 cube_vector(uint face, dvec2 uv) {
    return terrain_cube_vector(face, uv);
}

dvec3 surface(PatchGPU tile, dvec2 uv) {
    return terrain_surface(tile.address.x, uv, tile.anchor_polar_ratio.w, tile.rotation_height.w);
}

dvec3 parent_surface(PatchGPU tile, dvec2 uv, uint level) {
    double step = 2.0 / double((1u << (level - 1u)) * 16u);
    dvec2 grid = (uv + 1.0) / step;
    dvec2 cell = floor(grid);
    dvec2 f = grid - cell;
    dvec2 origin = cell * step - 1.0;
    dvec3 b = surface(tile, origin + dvec2(step, 0.0));
    dvec3 c = surface(tile, origin + dvec2(0.0, step));
    if (f.x + f.y <= 1.0) {
        dvec3 a = surface(tile, origin);
        return a * (1.0 - f.x - f.y) + b * f.x + c * f.y;
    }
    dvec3 d = surface(tile, origin + step);
    return d * (f.x + f.y - 1.0) + b * (1.0 - f.y) + c * (1.0 - f.x);
}

void main() {
    PatchGPU tile = patches[gl_InstanceIndex];
    uvec2 grid = uvec2(uint(gl_VertexIndex) % 17u, uint(gl_VertexIndex) / 17u);
    uint level = tile.address.y;
    double width = 2.0 / double(1u << level);
    dvec2 uv = -1.0 + (dvec2(tile.address.zw) + dvec2(grid) / 16.0) * width;
    dvec3 point = surface(tile, uv);
    dvec3 base_chord = normalize(cube_vector(tile.address.x, uv));
    uint mask = tile.metadata.y;
    bool coarse_edge = (grid.y == 0u && (mask & 1u) != 0u)
                    || (grid.x == 16u && (mask & 2u) != 0u)
                    || (grid.y == 16u && (mask & 4u) != 0u)
                    || (grid.x == 0u && (mask & 8u) != 0u);
    uint effective_level = coarse_edge && level > 0u ? level - 1u : level;
    if (effective_level > 0u) {
        double parent_width = 4.0 / double(1u << effective_level);
        double error = parent_width * parent_width * tile.rotation_error.w / max(length(point - tile.eye.xyz), 1e-12);
        double morph = clamp(error - 1.0, 0.0, 1.0);
        point = mix(parent_surface(tile, uv, effective_level), point, morph);
        PatchGPU base_tile = tile;
        base_tile.rotation_height.w = 0.0;
        dvec3 parent_base = parent_surface(base_tile, uv, effective_level)/dvec3(1.0, tile.anchor_polar_ratio.w, 1.0);
        base_chord = mix(parent_base, base_chord, morph);
    }
    dmat3 rotation = dmat3(tile.rotation_height.xyz, tile.rotation_error.xyz, tile.rotation_reserved.xyz);
    dvec3 relative = tile.anchor_radius.xyz + rotation * ((point - tile.anchor_polar_ratio.xyz) * tile.anchor_radius.w);
    f_world_pos = vec3(relative);
    f_local_pos = vec3(normalize(cube_vector(tile.address.x, uv)));
    f_base_chord = vec3(base_chord);
    double e = 0.0001;
    dvec3 du = surface(tile, uv + dvec2(e, 0.0)) - surface(tile, uv - dvec2(e, 0.0));
    dvec3 dv = surface(tile, uv + dvec2(0.0, e)) - surface(tile, uv - dvec2(0.0, e));
    dvec3 direction = normalize(cube_vector(tile.address.x, uv));
    dvec3 normal = tile.rotation_height.w == 0.0 ? normalize(direction / dvec3(1.0, tile.anchor_polar_ratio.w, 1.0))
                                                : normalize(cross(du, dv));
    f_normal = vec3(rotation * normal);
    f_body_idx = tile.metadata.x;
    f_subpixel_factor = 0.0;
    uint hash = tile.address.x * 73856093u ^ tile.address.y * 19349663u
              ^ tile.address.z * 83492791u ^ tile.address.w * 2654435761u;
    vec3 color = 0.25 + 0.65 * vec3(hash & 255u, (hash >> 8u) & 255u, (hash >> 16u) & 255u) / 255.0;
    f_tile_color = vec4(color, float(tile.metadata.z));
    f_tile_uv = vec2(grid) / 16.0;
    gl_Position = u_scene.view_proj * vec4(f_world_pos, 1.0);
    if (tile.eye_anchor.w > 0.5) {
        dmat3 eye_rotation = dmat3(tile.eye_rotation0.xyz, tile.eye_rotation1.xyz, tile.eye_rotation2.xyz);
        dvec3 eye = tile.eye_anchor.xyz + eye_rotation * ((point - tile.anchor_polar_ratio.xyz) * tile.anchor_radius.w);
        gl_Position = vec4(dmat4(u_scene.projection) * dvec4(eye, 1.0));
    }
}
