layout(push_constant) uniform RingPushConstants {
    uint host_index;
    uint packed_segments;
    uint segment_count;
    uint filter_profiles;
} u_push;
