layout(r32ui, set = 0, binding = 14) uniform uimage2D transparency_heads;
layout(std430, set = 0, binding = 15) buffer TransparentFragments {
    uint fragment_count;
    uint overflow_count;
    uint fragment_capacity;
    uint fragment_stride;
    uvec2 fragment_data[];
};

#ifdef TRANSPARENCY_RING_SLOT
layout(rgba16f, set = 0, binding = 45) uniform image2D ring_slot_color;
layout(r32ui, set = 0, binding = 46) uniform uimage2D ring_slot_depth;
// Slot depth bits: reverse-Z depth below 2.0 leaves the top two bits free. A ring fragment that loses the claim sets
// CONTENDED, after which the slot is treated as one more list fragment; DEFERRED marks it for the deferred transport.
const uint RING_SLOT_CONTENDED = 0x80000000u;
const uint RING_SLOT_DEFERRED = 0x40000000u;
const uint RING_SLOT_DEPTH_MASK = 0x3fffffffu;

uint ring_slot_bits(ivec2 pixel) {
    return all(lessThan(pixel, imageSize(ring_slot_depth))) ? imageLoad(ring_slot_depth, pixel).x : 0u;
}

// The slot stores transmittance 1-a, which half floats resolve finely near 0 where dense rings need it.
vec4 load_ring_slot(ivec2 pixel) {
    vec4 color = imageLoad(ring_slot_color, pixel);
    return vec4(color.rgb, 1.0-color.a);
}

void store_ring_slot(ivec2 pixel, vec4 color) {
    imageStore(ring_slot_color, pixel, vec4(clamp(color.rgb, vec3(0.0), vec3(65000.0)), clamp(1.0-color.a, 0.0, 1.0)));
}
#endif

const uint TRANSPARENCY_END = 0xffffffffu;
const uint TRANSPARENCY_MULTI_BIT = 0x80000000u;
const uint TRANSPARENCY_DEPTH_BIT = 0x40000000u;
const uint TRANSPARENCY_INDEX_MASK = 0x3fffffffu;

uvec2 transparent_color_packed(uint idx) { return fragment_data[idx]; }

void set_transparent_color_packed(uint idx, uvec2 packed) { fragment_data[idx] = packed; }

uint transparent_depth_bits(uint idx) {
    uvec2 pair = fragment_data[fragment_stride + (idx >> 1u)];
    return (idx & 1u) == 0u ? pair.x : pair.y;
}

void set_transparent_depth_bits(uint idx, uint bits) {
    uint slot = fragment_stride + (idx >> 1u);
    if ((idx & 1u) == 0u) fragment_data[slot].x = bits;
    else fragment_data[slot].y = bits;
}

uint transparent_next(uint idx) {
    uvec2 pair = fragment_data[fragment_stride + (fragment_stride >> 1u) + (idx >> 1u)];
    return (idx & 1u) == 0u ? pair.x : pair.y;
}

void set_transparent_next(uint idx, uint next) {
    uint slot = fragment_stride + (fragment_stride >> 1u) + (idx >> 1u);
    if ((idx & 1u) == 0u) fragment_data[slot].x = next;
    else fragment_data[slot].y = next;
}

vec4 transparent_color(uvec2 packed) {
    vec4 color = vec4(unpackHalf2x16(packed.x), unpackHalf2x16(packed.y));
    color.a = abs(color.a);
    return color;
}

vec4 transparent_color(uvec4 fragment) { return transparent_color(fragment.xy); }

bool transparent_excludes_bloom(uvec2 packed) { return (packed.y & 0x80000000u) != 0u; }
bool transparent_excludes_bloom(uvec4 fragment) { return transparent_excludes_bloom(fragment.xy); }

uvec2 pack_transparent_color(vec4 color, bool exclude_bloom) {
    color = clamp(color, vec4(0.0), vec4(65000.0, 65000.0, 65000.0, 1.0));
    uvec2 packed = uvec2(packHalf2x16(color.rg), packHalf2x16(color.ba));
    if (exclude_bloom) packed.y |= 0x80000000u;
    return packed;
}

uvec2 pack_transparent_color(vec4 color) { return pack_transparent_color(color, false); }

#ifdef TRANSPARENCY_CAPTURE
bool capture_transparency(vec4 color, float depth, bool exclude_bloom, bool needs_depth_test) {
#ifdef GL_KHR_shader_subgroup_ballot
    bool active_lane = !gl_HelperInvocation && color.a > 0.0;
    uvec4 ballot = subgroupBallot(active_lane);
    if (!active_lane) return true;
    uint count = subgroupBallotBitCount(ballot);
    uint local_offset = subgroupBallotExclusiveBitCount(ballot);
    uint first_lane = subgroupBallotFindLSB(ballot);
    uint base = 0u;
    if (gl_SubgroupInvocationID == first_lane) {
        base = atomicAdd(fragment_count, count);
        uint overflow = base >= fragment_capacity ? count :
                        (count > fragment_capacity - base ? count - (fragment_capacity - base) : 0u);
        if (overflow > 0u) atomicAdd(overflow_count, overflow);
    }
    base = subgroupBroadcast(base, first_lane);
    uint index = base + local_offset;
    if (index >= fragment_capacity) return false;
#else
    if (color.a <= 0.0) return true;
    uint index = atomicAdd(fragment_count, 1u);
    if (index >= fragment_capacity) {
        atomicAdd(overflow_count, 1u);
        return false;
    }
#endif
    uint entry = index | (needs_depth_test ? TRANSPARENCY_DEPTH_BIT : 0u);
    uint previous = imageAtomicExchange(transparency_heads, ivec2(gl_FragCoord.xy), entry);
    uint prev_next = TRANSPARENCY_END;
    if (previous != TRANSPARENCY_END) {
        imageAtomicOr(transparency_heads, ivec2(gl_FragCoord.xy), TRANSPARENCY_MULTI_BIT);
        prev_next = previous & TRANSPARENCY_INDEX_MASK;
    }
    set_transparent_color_packed(index, pack_transparent_color(color, exclude_bloom));
    set_transparent_depth_bits(index, floatBitsToUint(depth));
    set_transparent_next(index, prev_next);
    return true;
}

#ifdef TRANSPARENCY_RING_SLOT
// The first ring fragment at a pixel claims the fixed slot; later ones fall back to the list.
bool capture_ring(vec4 color, float depth, bool needs_depth_test) {
    if (!gl_HelperInvocation && color.a > 0.0) {
        ivec2 pixel = ivec2(gl_FragCoord.xy);
        uint previous = imageAtomicCompSwap(ring_slot_depth, pixel, 0u, floatBitsToUint(depth));
        if (previous == 0u) {
            store_ring_slot(pixel, color);
            return true;
        }
        if ((previous & RING_SLOT_CONTENDED) == 0u) imageAtomicOr(ring_slot_depth, pixel, RING_SLOT_CONTENDED);
    }
    return capture_transparency(color, depth, false, needs_depth_test);
}
#endif

bool capture_transparency(vec4 color, float depth, bool exclude_bloom) {
    return capture_transparency(color, depth, exclude_bloom, true);
}
bool capture_transparency(vec4 color, float depth) { return capture_transparency(color, depth, false, true); }
bool capture_transparency(vec4 color) { return capture_transparency(color, gl_FragCoord.z, false, true); }
#endif
