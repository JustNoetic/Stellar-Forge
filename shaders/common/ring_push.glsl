layout(push_constant) uniform RingPushConstants {
    uint host_index;
    uint packed_segments;
    uint segment_count;
} u_push;
