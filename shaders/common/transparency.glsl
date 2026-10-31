layout(r32ui, set = 0, binding = 14) uniform uimage2D transparency_heads;
layout(std430, set = 0, binding = 15) buffer TransparentFragments {
    uint fragment_count;
    uint overflow_count;
    uint fragment_capacity;
    uint reserved;
    uvec4 fragments[];
};

const uint TRANSPARENCY_END = 0xffffffffu;

vec4 transparent_color(uvec4 fragment) {
    vec4 color = vec4(unpackHalf2x16(fragment.x), unpackHalf2x16(fragment.y));
    color.a = abs(color.a);
    return color;
}

bool transparent_excludes_bloom(uvec4 fragment) { return (fragment.y & 0x80000000u) != 0u; }

uvec2 pack_transparent_color(vec4 color, bool exclude_bloom) {
    color = clamp(color, vec4(0.0), vec4(65000.0, 65000.0, 65000.0, 1.0));
    uvec2 packed = uvec2(packHalf2x16(color.rg), packHalf2x16(color.ba));
    if (exclude_bloom) packed.y |= 0x80000000u;
    return packed;
}

uvec2 pack_transparent_color(vec4 color) { return pack_transparent_color(color, false); }

#ifdef TRANSPARENCY_CAPTURE
bool capture_transparency(vec4 color, float depth, bool exclude_bloom) {
    if (color.a <= 0.0) return true;
    uint index = atomicAdd(fragment_count, 1u);
    if (index >= fragment_capacity) {
        atomicAdd(overflow_count, 1u);
        return false;
    }
    uint previous = imageAtomicExchange(transparency_heads, ivec2(gl_FragCoord.xy), index);
    fragments[index] = uvec4(pack_transparent_color(color, exclude_bloom), floatBitsToUint(depth), previous);
    return true;
}

bool capture_transparency(vec4 color, float depth) { return capture_transparency(color, depth, false); }
bool capture_transparency(vec4 color) { return capture_transparency(color, gl_FragCoord.z, false); }
#endif
