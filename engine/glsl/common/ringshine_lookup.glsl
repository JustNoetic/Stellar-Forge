// Independent star columns and host rows, with filtering confined to a tile.
vec3 ringshine_sample(vec2 host_uv, int ring_index, int star_index) {
    vec2 tile_size = vec2(textureSize(u_ringshine_map, 0)) / 16.0;
    float row = float(ring_index);
    vec2 local_uv = vec2(host_uv.x, host_uv.y * 16.0 - row);
    local_uv = (0.5 + clamp(local_uv, 0.0, 1.0) * (tile_size - 1.0)) / tile_size;
    return textureLod(u_ringshine_map,
        (vec2(float(star_index), row) + local_uv) / 16.0, 0.0).rgb;
}
