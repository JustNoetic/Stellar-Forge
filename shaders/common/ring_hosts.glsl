struct RingSegmentGPU {
    vec4 bounds_opacity_scatter;
    vec4 phase_row;
    vec4 color_flags;
};

struct RingHostGPU {
    vec4 center_min_r;
    vec4 pole_max_r;
    vec4 tangent_radius;
    vec4 bitangent_oblateness;
    uvec4 counts_indices;
    RingSegmentGPU segments[8];
};

layout(std430, set = 0, binding = 7) readonly buffer RingHostBuffer {
    RingHostGPU ring_hosts[];
};
