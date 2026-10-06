#ifndef TERRAIN_SURFACE_GLSL
#define TERRAIN_SURFACE_GLSL

layout(std430, binding = 15) readonly buffer TerrainHeightPages { uvec4 height_pages[]; };
uniform sampler2DArray u_height_array;
uniform uint u_height_table_mask;
uniform int u_height_tile_size;

uint terrain_hash(uint v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15;
    v *= 0x846ca68bu; return v ^ (v >> 16);
}

// GLSL's trigonometric overloads use floats. Evaluate sin/cos in double
// precision on [-pi/4,pi/4] before division; this retains small ground patches.
dvec2 terrain_tan(dvec2 a) {
    dvec2 q = a*a;
    dvec2 s = ((((((((1.0lf/355687428096000.0lf*q-1.0lf/1307674368000.0lf)*q
        +1.0lf/6227020800.0lf)*q-1.0lf/39916800.0lf)*q+1.0lf/362880.0lf)*q
        -1.0lf/5040.0lf)*q+1.0lf/120.0lf)*q-1.0lf/6.0lf)*q+1.0lf)*a;
    dvec2 c = ((((((((1.0lf/20922789888000.0lf*q-1.0lf/87178291200.0lf)*q
        +1.0lf/479001600.0lf)*q-1.0lf/3628800.0lf)*q+1.0lf/40320.0lf)*q
        -1.0lf/720.0lf)*q+1.0lf/24.0lf)*q-1.0lf/2.0lf)*q+1.0lf);
    return s/c;
}

dvec3 terrain_direction_precise(int face, dvec2 uv) {
    dvec2 p = terrain_tan(uv * 0.78539816339744830962lf);
    dvec3 q;
    if (face == 0) q = dvec3(1.0lf, -p.y, -p.x);
    else if (face == 1) q = dvec3(-1.0lf, -p.y, p.x);
    else if (face == 2) q = dvec3(p.x, 1.0lf, p.y);
    else if (face == 3) q = dvec3(p.x, -1.0lf, -p.y);
    else if (face == 4) q = dvec3(p.x, -p.y, 1.0lf);
    else q = dvec3(-p.x, -p.y, -1.0lf);
    return normalize(q);
}

void terrain_face_uv(vec3 p, out int face, out vec2 uv) {
    vec2 q;
    if (abs(p.x) >= abs(p.y) && abs(p.x) >= abs(p.z)) {
        face = p.x >= 0.0 ? 0 : 1;
        q = vec2(p.x >= 0.0 ? -p.z : p.z, -p.y) / abs(p.x);
    } else if (abs(p.y) >= abs(p.z)) {
        face = p.y >= 0.0 ? 2 : 3;
        q = vec2(p.x, p.y >= 0.0 ? p.z : -p.z) / abs(p.y);
    } else {
        face = p.z >= 0.0 ? 4 : 5;
        q = vec2(p.z >= 0.0 ? p.x : -p.x, -p.y) / abs(p.z);
    }
    uv = clamp(atan(q) * 0.636619772367581343 + 0.5, 0.0, 1.0);
}

int terrain_height_page(uint body, uint code) {
    uint at = terrain_hash(code ^ (body * 0x9e3779b9u)) & u_height_table_mask;
    for (int probe = 0; probe < 32; ++probe) {
        uvec4 p = height_pages[at];
        if (p.x == 0xffffffffu) return -1;
        if (p.x == code && p.y == body) return int(p.z);
        at = (at + 1u) & u_height_table_mask;
    }
    return -1;
}

float terrain_height_cubic(vec2 uv, int layer) {
    float size = float(u_height_tile_size);
    float width = size + 4.0;
    vec2 p = uv * size + 1.5;
    vec2 f = fract(p), index = floor(p);
    vec2 w0 = pow(1.0-f, vec2(3.0))/6.0;
    vec2 w1 = (3.0*f*f*f-6.0*f*f+4.0)/6.0;
    vec2 w2 = (-3.0*f*f*f+3.0*f*f+3.0*f+1.0)/6.0;
    vec2 w3 = f*f*f/6.0;
    vec2 a = w0+w1, b = w2+w3;
    vec2 t0 = (index-0.5+w1/a)/width;
    vec2 t1 = (index+1.5+w3/b)/width;
    return textureLod(u_height_array, vec3(t0, layer), 0.0).r*a.x*a.y
         + textureLod(u_height_array, vec3(t1.x,t0.y,layer), 0.0).r*b.x*a.y
         + textureLod(u_height_array, vec3(t0.x,t1.y,layer), 0.0).r*a.x*b.y
         + textureLod(u_height_array, vec3(t1, layer), 0.0).r*b.x*b.y;
}

// Source resolution depends on residency, never on the mesh patch's LOD.
// Fade children into their common ancestor along source-tile boundaries.
float terrain_height(uint body, vec3 direction) {
    int face; vec2 uv; terrain_face_uv(direction, face, uv);
    float value = 0.0;
    bool found = false;
    for (uint lod = 0u; lod <= 10u; ++lod) {
        float count = float(1u << lod);
        vec2 q = clamp(uv*count,vec2(0.0),vec2(count));
        uvec2 xy = min(uvec2(floor(q)),uvec2(count-1.0)); vec2 local = q-vec2(xy);
        uint code = uint(face) | (lod << 3u) | (xy.x << 7u) | (xy.y << 18u);
        int layer = terrain_height_page(body, code);
        if (layer < 0) continue;
        float e = terrain_height_cubic(local, layer);
        float edge = min(min(local.x,1.0-local.x),min(local.y,1.0-local.y));
        float weight = found ? smoothstep(0.0,2.0/float(u_height_tile_size),edge) : 1.0;
        value = mix(value,e,weight); found = true;
    }
    return value;
}

#include "common/procedural_terrain.glsl"

float terrain_elevation(uint body, vec3 direction, vec4 hrange, vec4 detail) {
    float e = terrain_height(body,direction);
    float h = hrange.x + e*hrange.y;
    if (detail.y > 0.5) h = max(detail.z,h);
    return h + terrain_micro_elevation(direction,e,hrange.x,hrange.y,detail.y>0.5,detail.z);
}

vec3 terrain_surface_normal(uint body, vec3 n, float radius, float obl, vec4 hrange, vec4 detail, float elevation) {
    vec3 ref = abs(n.y)<0.9 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 u = normalize(cross(ref,n)), v = cross(n,u);
    const float delta = 0.00001;
    float du = (terrain_elevation(body,normalize(n+u*delta),hrange,detail)
               -terrain_elevation(body,normalize(n-u*delta),hrange,detail))/(2.0*delta);
    float dv = (terrain_elevation(body,normalize(n+v*delta),hrange,detail)
               -terrain_elevation(body,normalize(n-v*delta),hrange,detail))/(2.0*delta);
    vec3 shape = vec3(1.0,1.0-obl,1.0), ell = n*shape;
    float len = length(ell); vec3 radial = ell/len;
    vec3 au = u*shape, av = v*shape;
    vec3 tu = radius*au + radial*du + elevation*(au-radial*dot(radial,au))/len;
    vec3 tv = radius*av + radial*dv + elevation*(av-radial*dot(radial,av))/len;
    return normalize(cross(tu,tv));
}
#endif
