// Baked geometry is body-local and independent of camera, lighting and spin.
struct TerrainBakeJob {
    vec4 range;
    vec4 uv_trans;
    vec4 meta;
    vec4 height;
    vec4 hrange;
    vec4 detail; // stitched edges, water enabled, water level, reserved
    vec4 shape; // radius in AU, oblateness, destination slot, unused
};

vec3 terrain_cache_direction(int face, vec2 uv) {
    vec2 p = tan(uv * 0.7853981633974483);
    vec3 q;
    if (face == 0) q = vec3(1.0, -p.y, -p.x);
    else if (face == 1) q = vec3(-1.0, -p.y, p.x);
    else if (face == 2) q = vec3(p.x, 1.0, p.y);
    else if (face == 3) q = vec3(p.x, -1.0, -p.y);
    else if (face == 4) q = vec3(p.x, -p.y, 1.0);
    else q = vec3(-p.x, -p.y, -1.0);
    return normalize(q);
}
