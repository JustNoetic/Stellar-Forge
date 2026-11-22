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

uint surface_slot_uv(uint base, vec2 uv, uint face, uint level, out vec2 uv_out) {
    uint n = 1u << level;
    uvec2 tile = min(uvec2(uv*float(n)), uvec2(n-1u));
    uint n2 = n*n;
    uint slot = surface_words[base+(2u+face)*n2-2u+tile.y*n+tile.x];
    if (slot != 0u) {
        vec2 local = uv*float(n)-vec2(tile);
        uv_out = (2.0+local*128.0)/132.0;
        return slot;
    }
    while (level > 0u) {
        --level;
        n >>= 1u;
        tile = min(uvec2(uv*float(n)), uvec2(n-1u));
        n2 = n*n;
        slot = surface_words[base+(2u+face)*n2-2u+tile.y*n+tile.x];
        if (slot != 0u) {
            vec2 local = uv*float(n)-vec2(tile);
            uv_out = (2.0+local*128.0)/132.0;
            return slot;
        }
    }
    uv_out = vec2(0.5);
    return 0u;
}

vec3 surface_level_uv(uint base, vec2 uv, uint face, uint level, vec3 fallback) {
    vec2 tile_uv;
    uint slot = surface_slot_uv(base, uv, face, level, tile_uv);
    if (slot == 0u) return fallback;
    return textureLod(u_surface_tiles, vec3(tile_uv, float(slot-1u)), 0.0).rgb;
}

vec3 surface_level(uint base, vec3 face_uv, uint level, vec3 fallback) {
    return surface_level_uv(base, clamp(face_uv.xy*.5+.5, 0.0, 1.0), uint(face_uv.z), level, fallback);
}

vec3 surface_albedo_at(uint body, vec2 uv01, uint face, float raw_lod, vec3 fallback) {
    uint base = surface_words[body*4u];
    if (base == 0u) return fallback;
    uint maximum = surface_words[body*4u+1u];
    float level = clamp(raw_lod, 0.0, float(maximum));
    uint coarse = uint(floor(level));
    float frac = fract(level);
    vec2 coarse_uv;
    uint coarse_slot = surface_slot_uv(base, uv01, face, coarse, coarse_uv);
    if (coarse_slot == 0u) return fallback;
    uint fine_slot = coarse_slot;
    vec2 fine_uv = coarse_uv;
    if (frac > 0.0 && coarse < maximum) {
        uint fn = 1u << (coarse+1u);
        uvec2 ftile = min(uvec2(uv01*float(fn)), uvec2(fn-1u));
        uint fn2 = fn*fn;
        uint slot = surface_words[base+(2u+face)*fn2-2u+ftile.y*fn+ftile.x];
        if (slot != 0u) {
            fine_slot = slot;
            vec2 flocal = uv01*float(fn)-vec2(ftile);
            fine_uv = (2.0+flocal*128.0)/132.0;
        }
    }
    vec3 a = textureLod(u_surface_tiles, vec3(coarse_uv, float(coarse_slot-1u)), 0.0).rgb;
    vec3 b = (fine_slot == coarse_slot) ? a : textureLod(u_surface_tiles, vec3(fine_uv, float(fine_slot-1u)), 0.0).rgb;
    return mix(a, b, frac);
}

vec3 surface_albedo(uint body, vec3 local_direction, vec3 fallback) {
    vec3 direction = normalize(local_direction);
    float footprint = max(length(dFdx(direction)), length(dFdy(direction)));
    uint base = surface_words[body*4u];
    if (base == 0u) return fallback;
    vec3 face_uv = surface_face_uv(direction);
    float raw_lod = log2(3.0/max(128.0*footprint, 1e-9));
    return surface_albedo_at(body, clamp(face_uv.xy*.5+.5, 0.0, 1.0), uint(face_uv.z), raw_lod, fallback);
}

vec3 surface_normal_level(uint base, vec3 face_uv, uint level, vec3 fallback) {
    vec2 uv = clamp(face_uv.xy*.5+.5, 0.0, 1.0);
    uint face = uint(face_uv.z);
    for (;;) {
        uint n = 1u << level;
        uvec2 tile = min(uvec2(uv*float(n)), uvec2(n-1u));
        uint n2 = n*n;
        uint address = (2u+face)*n2-2u+tile.y*n+tile.x;
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
    uint packed;
    float blend;
    vec2 coarse_uv;
    vec2 fine_uv;
};

uint normal_lookup_flags(SurfaceNormalLookup lookup) { return lookup.packed & 3u; }
uint normal_lookup_face(SurfaceNormalLookup lookup) { return (lookup.packed >> 2u) & 7u; }
uint normal_lookup_coarse_slot(SurfaceNormalLookup lookup) { return (lookup.packed >> 5u) & 8191u; }
uint normal_lookup_fine_slot(SurfaceNormalLookup lookup) { return (lookup.packed >> 18u) & 8191u; }

uint surface_normal_slot_uv(uint base, vec2 uv, uint face, uint level, out vec2 uv_out) {
    uint n = 1u << level;
    uvec2 tile = min(uvec2(uv*float(n)), uvec2(n-1u));
    uint n2 = n*n;
    uint slot = normal_words[base+(2u+face)*n2-2u+tile.y*n+tile.x];
    if (slot != 0u) {
        vec2 local = uv*float(n)-vec2(tile);
        uv_out = (2.0+local*128.0)/132.0;
        return slot;
    }
    while (level > 0u) {
        --level;
        n >>= 1u;
        tile = min(uvec2(uv*float(n)), uvec2(n-1u));
        n2 = n*n;
        slot = normal_words[base+(2u+face)*n2-2u+tile.y*n+tile.x];
        if (slot != 0u) {
            vec2 local = uv*float(n)-vec2(tile);
            uv_out = (2.0+local*128.0)/132.0;
            return slot;
        }
    }
    uv_out = vec2(0.5);
    return 0u;
}

uint surface_normal_slot(uint base, vec3 face_uv, uint level, out vec2 uv_out) {
    return surface_normal_slot_uv(base, clamp(face_uv.xy*.5+.5, 0.0, 1.0), uint(face_uv.z), level, uv_out);
}

SurfaceNormalLookup surface_normal_lookup_at(uint body, vec2 uv01, uint face, float raw_lod) {
    uint base = normal_words[body*4u];
    uint flags = normal_words[body*4u+2u] & 3u;
    SurfaceNormalLookup lookup;
    lookup.packed = flags | (face << 2u);
    lookup.blend = 0.0;
    lookup.coarse_uv = vec2(0.5);
    lookup.fine_uv = vec2(0.5);
    if (base == 0u || flags == 0u) return lookup;
    uint maximum = normal_words[body*4u+1u];
    float level = clamp(raw_lod, 0.0, float(maximum));
    uint coarse = uint(floor(level));
    float frac = fract(level);
    lookup.blend = frac;
    uint coarse_slot = surface_normal_slot_uv(base, uv01, face, coarse, lookup.coarse_uv);
    uint fine_slot = coarse_slot;
    lookup.fine_uv = lookup.coarse_uv;
    if (frac > 0.0 && coarse < maximum) {
        uint fn = 1u << (coarse+1u);
        uvec2 ftile = min(uvec2(uv01*float(fn)), uvec2(fn-1u));
        uint fn2 = fn*fn;
        uint slot = normal_words[base+(2u+face)*fn2-2u+ftile.y*fn+ftile.x];
        if (slot != 0u) {
            fine_slot = slot;
            vec2 flocal = uv01*float(fn)-vec2(ftile);
            lookup.fine_uv = (2.0+flocal*128.0)/132.0;
        }
    }
    lookup.packed |= (coarse_slot << 5u) | (fine_slot << 18u);
    return lookup;
}

SurfaceNormalLookup surface_normal_lookup(uint body, vec3 local_direction) {
    vec3 direction = normalize(local_direction);
    float footprint = max(length(dFdx(direction)), length(dFdy(direction)));
    vec3 face_uv = surface_face_uv(direction);
    float raw_lod = log2(3.0/max(128.0*footprint, 1e-9));
    return surface_normal_lookup_at(body, clamp(face_uv.xy*.5+.5, 0.0, 1.0), uint(face_uv.z), raw_lod);
}

vec3 surface_local_normal_from_lookup(SurfaceNormalLookup lookup, vec3 local_fallback) {
    uint flags = normal_lookup_flags(lookup);
    uint coarse_slot = normal_lookup_coarse_slot(lookup);
    if ((flags & 1u) == 0u || coarse_slot == 0u) return normalize(local_fallback);
    uint fine_slot = normal_lookup_fine_slot(lookup);
    vec3 a = textureLod(u_normal_tiles, vec3(lookup.coarse_uv, float(coarse_slot-1u)), 0.0).xyz;
    vec3 b = (fine_slot == 0u) ? local_fallback
           : (fine_slot == coarse_slot ? a
           : textureLod(u_normal_tiles, vec3(lookup.fine_uv, float(fine_slot-1u)), 0.0).xyz);
    return normalize(mix(a, b, lookup.blend));
}

vec3 surface_shading_normal_from_lookup(SurfaceNormalLookup lookup, vec3 fallback, mat3 rotation) {
    uint flags = normal_lookup_flags(lookup);
    uint coarse_slot = normal_lookup_coarse_slot(lookup);
    if ((flags & 1u) == 0u || coarse_slot == 0u) return fallback;
    vec3 local_fallback = transpose(rotation)*fallback;
    uint fine_slot = normal_lookup_fine_slot(lookup);
    vec3 a = textureLod(u_normal_tiles, vec3(lookup.coarse_uv, float(coarse_slot-1u)), 0.0).xyz;
    vec3 b = (fine_slot == 0u) ? local_fallback
           : (fine_slot == coarse_slot ? a
           : textureLod(u_normal_tiles, vec3(lookup.fine_uv, float(fine_slot-1u)), 0.0).xyz);
    return normalize(rotation*mix(a, b, lookup.blend));
}

vec3 surface_shading_normal(uint body, vec3 local_direction, vec3 fallback, mat3 rotation) {
    return surface_shading_normal_from_lookup(surface_normal_lookup(body, local_direction), fallback, rotation);
}
