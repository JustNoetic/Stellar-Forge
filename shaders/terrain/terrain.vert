#version 460

#include "common/celestial_scene.glsl"

#ifdef TERRAIN_CACHED_FP64
#define TerrainScalar double
#define TerrainVector dvec3
#define TerrainUV dvec2
#define TerrainMatrix dmat3
#else
#define TerrainScalar float
#define TerrainVector vec3
#define TerrainUV vec2
#define TerrainMatrix mat3
#endif

struct CachedPatchGPU {
    vec4 anchor_high;
    vec4 anchor_low;
    vec4 rotation0;
    vec4 rotation1;
    vec4 rotation2;
    vec4 eye;
    uvec4 address;
    uvec4 metadata;
};

struct GeometryVertex {
    vec4 position;
    vec4 normal;
    vec4 direction;
};

layout(std430, set = 0, binding = 3) readonly buffer TerrainPatches {
    CachedPatchGPU patches[];
};
layout(std430, set = 0, binding = 4) readonly buffer TerrainGeometry {
    GeometryVertex vertices[];
};

layout(location = 0) out vec3 f_world_pos;
layout(location = 1) out vec3 f_local_pos;
layout(location = 2) out vec3 f_normal;
layout(location = 3) out float f_subpixel_factor;
layout(location = 4) flat out uint f_body_idx;
layout(location = 5) flat out vec4 f_tile_color;
layout(location = 6) out vec2 f_tile_uv;

TerrainVector parent_position(uint first, uvec2 grid, uint stride) {
    uvec2 origin = min(grid / stride * stride, uvec2(16u - stride));
    TerrainUV f = TerrainUV(grid - origin) / TerrainScalar(stride);
    TerrainVector b = TerrainVector(vertices[first + origin.y * 17u + origin.x + stride].position.xyz);
    TerrainVector c = TerrainVector(vertices[first + (origin.y + stride) * 17u + origin.x].position.xyz);
    if (f.x + f.y <= 1.0) {
        TerrainVector a = TerrainVector(vertices[first + origin.y * 17u + origin.x].position.xyz);
        return a * (1.0 - f.x - f.y) + b * f.x + c * f.y;
    }
    TerrainVector d = TerrainVector(vertices[first + (origin.y + stride) * 17u + origin.x + stride].position.xyz);
    return d * (f.x + f.y - 1.0) + b * (1.0 - f.y) + c * (1.0 - f.x);
}

void main() {
    CachedPatchGPU tile = patches[gl_InstanceIndex];
    uint first = tile.metadata.y * 289u;
    uvec2 grid = uvec2(uint(gl_VertexIndex) % 17u, uint(gl_VertexIndex) / 17u);
    GeometryVertex vertex = vertices[first + uint(gl_VertexIndex)];
    TerrainVector point = TerrainVector(vertex.position.xyz);
    uint level = tile.address.y;
    uint mask = tile.metadata.z;
    bool coarse_edge = (grid.y == 0u && (mask & 1u) != 0u)
                    || (grid.x == 16u && (mask & 2u) != 0u)
                    || (grid.y == 16u && (mask & 4u) != 0u)
                    || (grid.x == 0u && (mask & 8u) != 0u);
    uint effective_level = coarse_edge && level > 0u ? level - 1u : level;
    if (effective_level > 0u) {
        TerrainScalar parent_width = 4.0 / TerrainScalar(1u << effective_level);
        TerrainScalar error = parent_width * parent_width * TerrainScalar(tile.eye.w) / max(length(point - TerrainVector(tile.eye.xyz)), 1e-12);
        TerrainScalar morph = clamp(error - 1.0, 0.0, 1.0);
        point = mix(parent_position(first, grid, coarse_edge ? 4u : 2u), point, morph);
    }
    TerrainMatrix rotation = TerrainMatrix(TerrainVector(tile.rotation0.xyz), TerrainVector(tile.rotation1.xyz), TerrainVector(tile.rotation2.xyz));
    TerrainVector eye = (TerrainVector(tile.anchor_high.xyz) + rotation * point * TerrainScalar(tile.anchor_high.w)) + TerrainVector(tile.anchor_low.xyz);
    f_world_pos = transpose(mat3(u_scene.view)) * vec3(eye);
    f_local_pos = vertex.direction.xyz;
    f_normal = vec3(transpose(TerrainMatrix(u_scene.view)) * rotation * TerrainVector(vertex.normal.xyz));
    f_body_idx = tile.metadata.x;
    f_subpixel_factor = 0.0;
    uint hash = tile.address.x * 73856093u ^ tile.address.y * 19349663u
              ^ tile.address.z * 83492791u ^ tile.address.w * 2654435761u;
    vec3 color = 0.25 + 0.65 * vec3(hash & 255u, (hash >> 8u) & 255u, (hash >> 16u) & 255u) / 255.0;
    f_tile_color = vec4(color, float(tile.metadata.w));
    f_tile_uv = vec2(grid) / 16.0;
    gl_Position = u_scene.projection * vec4(vec3(eye), 1.0);
#if defined(TERRAIN_PRECISE_PROJECTION) || defined(TERRAIN_CACHED_FP64)
    dvec3 precise_eye = dvec3(tile.anchor_high.xyz) + dvec3(tile.anchor_low.xyz)
        + dmat3(rotation) * dvec3(point) * double(tile.anchor_high.w);
    gl_Position = vec4(dmat4(u_scene.projection) * dvec4(precise_eye, 1.0));
#endif
}
