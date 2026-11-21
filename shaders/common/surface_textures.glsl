layout(set = 0, binding = 5) uniform sampler2DArray u_surface_tiles;
layout(std430, set = 0, binding = 6) readonly buffer SurfacePages {
    uint surface_words[];
};
layout(set = 0, binding = 10) uniform sampler2DArray u_normal_tiles;
layout(std430, set = 0, binding = 11) readonly buffer NormalPages {
    uint normal_words[];
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
    vec3 b = fine == coarse ? a : surface_level(base, face_uv, fine, fallback);
    return mix(a, b, fract(level));
}

vec3 surface_normal_level(uint base, vec3 face_uv, uint level, vec3 fallback) {
    vec2 uv = clamp(face_uv.xy*.5+.5, 0.0, 1.0);
    for (;;) {
        uint n = 1u << level;
        uvec2 tile = min(uvec2(uv*float(n)), uvec2(n-1u));
        uint address = 2u*(n*n-1u)+uint(face_uv.z)*n*n+tile.y*n+tile.x;
        uint slot = normal_words[base+address];
        if (slot != 0u) {
            vec2 local = uv*float(n)-vec2(tile);
            return textureLod(u_normal_tiles, vec3((2.0+local*128.0)/132.0, float(slot-1u)), 0.0).xyz;
        }
        if (level == 0u) return fallback;
        --level;
    }
}

struct SurfaceNormalLookup {
    uint base;
    uint flags;
    vec3 face_uv;
    uint coarse;
    uint fine;
    float blend;
    uint coarse_slot;
    vec2 coarse_uv;
    uint fine_slot;
    vec2 fine_uv;
};

uint surface_normal_slot(uint base, vec3 face_uv, uint level, out vec2 uv_out) {
    vec2 uv = clamp(face_uv.xy*.5+.5, 0.0, 1.0);
    for (;;) {
        uint n = 1u << level;
        uvec2 tile = min(uvec2(uv*float(n)), uvec2(n-1u));
        uint address = 2u*(n*n-1u)+uint(face_uv.z)*n*n+tile.y*n+tile.x;
        uint slot = normal_words[base+address];
        if (slot != 0u) {
            vec2 local = uv*float(n)-vec2(tile);
            uv_out = (2.0+local*128.0)/132.0;
            return slot;
        }
        if (level == 0u) {
            uv_out = vec2(0.5);
            return 0u;
        }
        --level;
    }
}

SurfaceNormalLookup surface_normal_lookup(uint body, vec3 local_direction) {
    vec3 direction = normalize(local_direction);
    float footprint = max(length(dFdx(direction)), length(dFdy(direction)));
    uint base = normal_words[body*4u];
    uint flags = normal_words[body*4u+2u];
    SurfaceNormalLookup lookup;
    lookup.base = base;
    lookup.flags = flags;
    lookup.face_uv = surface_face_uv(direction);
    lookup.coarse = 0u;
    lookup.fine = 0u;
    lookup.blend = 0.0;
    lookup.coarse_slot = 0u;
    lookup.coarse_uv = vec2(0.5);
    lookup.fine_slot = 0u;
    lookup.fine_uv = vec2(0.5);
    if (base == 0u || flags == 0u) return lookup;
    uint maximum = normal_words[body*4u+1u];
    float level = clamp(log2(3.0/max(128.0*footprint, 1e-9)), 0.0, float(maximum));
    lookup.coarse = uint(floor(level));
    lookup.fine = uint(ceil(level));
    lookup.blend = fract(level);
    lookup.coarse_slot = surface_normal_slot(base, lookup.face_uv, lookup.coarse, lookup.coarse_uv);
    if (lookup.fine == lookup.coarse) {
        lookup.fine_slot = lookup.coarse_slot;
        lookup.fine_uv = lookup.coarse_uv;
    } else {
        lookup.fine_slot = surface_normal_slot(base, lookup.face_uv, lookup.fine, lookup.fine_uv);
    }
    return lookup;
}

vec3 surface_shading_normal_from_lookup(SurfaceNormalLookup lookup, vec3 fallback, mat3 rotation) {
    if ((lookup.flags & 1u) == 0u || lookup.coarse_slot == 0u) return fallback;
    vec3 local_fallback = transpose(rotation)*fallback;
    vec3 a = textureLod(u_normal_tiles, vec3(lookup.coarse_uv, float(lookup.coarse_slot-1u)), 0.0).xyz;
    vec3 b = (lookup.fine_slot == 0u) ? local_fallback
           : (lookup.fine_slot == lookup.coarse_slot ? a
           : textureLod(u_normal_tiles, vec3(lookup.fine_uv, float(lookup.fine_slot-1u)), 0.0).xyz);
    return normalize(rotation*mix(a, b, lookup.blend));
}

vec3 surface_shading_normal(uint body, vec3 local_direction, vec3 fallback, mat3 rotation) {
    return surface_shading_normal_from_lookup(surface_normal_lookup(body, local_direction), fallback, rotation);
}
