#ifndef ATMOSPHERE_TILES_GLSL
#define ATMOSPHERE_TILES_GLSL
#if defined(ATMOSPHERE_TILE_LEAN) || defined(ATMOSPHERE_TILE_DEFERRED)
layout(std430, set = 0, binding = 37) buffer AtmosphereTiles {
    uint tile_count;
    uint tile_groups_y;
    uint tile_groups_z;
    uint tile_padding;
    uint tile_ids[];
};
#endif

#ifdef ATMOSPHERE_TILE_DEFERRED
ivec2 atmosphere_tile_pixel() {
    uint id = tile_ids[gl_WorkGroupID.x];
    return ivec2(id & 0xffffu, id >> 16u)*8+ivec2(gl_LocalInvocationID.xy);
}
#endif

#ifdef ATMOSPHERE_TILE_LEAN
shared bool atmosphere_tile_deferred;

void atmosphere_tile_begin() {
    if (gl_LocalInvocationIndex == 0u) atmosphere_tile_deferred = false;
    memoryBarrierShared();
    barrier();
}

void atmosphere_tile_end(bool deferred) {
    if (deferred) atmosphere_tile_deferred = true;
    memoryBarrierShared();
    barrier();
    if (gl_LocalInvocationIndex == 0u && atmosphere_tile_deferred)
        tile_ids[atomicAdd(tile_count, 1u)] = gl_WorkGroupID.x | (gl_WorkGroupID.y << 16u);
}
#endif
#endif
