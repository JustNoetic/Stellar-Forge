layout(set = 0, binding = 5) uniform sampler2DArray u_surface_tiles;
layout(std430, set = 0, binding = 6) readonly buffer SurfacePages {
    uint surface_words[];
};

vec3 surface_face_uv(vec3 direction) {
    vec3 a = abs(direction);
    if (a.x >= a.y && a.x >= a.z)
        return direction.x >= 0.0 ? vec3(-direction.z/a.x, direction.y/a.x, 0.0) : vec3(direction.z/a.x, direction.y/a.x, 1.0);
    if (a.y >= a.z)
        return direction.y >= 0.0 ? vec3(direction.x/a.y, -direction.z/a.y, 2.0) : vec3(direction.x/a.y, direction.z/a.y, 3.0);
    return direction.z >= 0.0 ? vec3(direction.x/a.z, direction.y/a.z, 4.0) : vec3(-direction.x/a.z, direction.y/a.z, 5.0);
}

vec3 surface_level(uint base, vec3 face_uv, uint level, vec3 fallback) {
    vec2 uv = clamp(face_uv.xy*.5+.5, 0.0, 1.0);
    for (;;) {
        uint n = 1u << level;
        uvec2 tile = min(uvec2(uv*float(n)), uvec2(n-1u));
        uint address = 2u*(n*n-1u)+uint(face_uv.z)*n*n+tile.y*n+tile.x;
        uint slot = surface_words[base+address];
        if (slot != 0u) {
            vec2 local = uv*float(n)-vec2(tile);
            return textureLod(u_surface_tiles, vec3((2.0+local*128.0)/132.0, float(slot-1u)), 0.0).rgb;
        }
        if (level == 0u) return fallback;
        --level;
    }
}

vec3 surface_albedo(uint body, vec3 local_direction, vec3 fallback) {
    vec3 direction = normalize(local_direction);
    float footprint = max(length(dFdx(direction)), length(dFdy(direction)));
    uint base = surface_words[body*4u];
    if (base == 0u) return fallback;
    uint maximum = surface_words[body*4u+1u];
    float level = clamp(log2(3.0/max(128.0*footprint, 1e-9)), 0.0, float(maximum));
    uint coarse = uint(floor(level)), fine = uint(ceil(level));
    vec3 face_uv = surface_face_uv(direction);
    vec3 a = surface_level(base, face_uv, coarse, fallback);
    vec3 b = surface_level(base, face_uv, fine, fallback);
    return mix(a, b, fract(level));
}
