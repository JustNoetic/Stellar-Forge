#version 460
#include "common/bloom.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Spikes { uvec4 command; BloomSpike spikes[]; };
layout(push_constant) uniform Spike {
    vec2 extent;
    vec2 direction;
    float half_length;
    float width;
    float scale;
    uint capacity;
} spike;
layout(location = 0) out vec2 local;
layout(location = 1) flat out vec3 amplitude;
layout(location = 2) flat out float center_bin;
layout(location = 3) flat out uvec2 profile_index;

const vec2 CORNERS[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    if (uint(gl_InstanceIndex) >= spike.capacity) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    uint arm = gl_VertexIndex < 6 ? 0u : 1u;
    vec2 center = spikes[gl_InstanceIndex].center_variance.xy;
    vec3 energy = spikes[gl_InstanceIndex].energy_covariance.rgb;
    vec2 axis = arm == 0u ? spike.direction : vec2(-spike.direction.y, spike.direction.x);
    vec2 normal = bloom_spike_normal(spike.direction, arm);
    int first = BLOOM_SPIKE_BINS, last = -1;
    for (int bin = 0; bin < BLOOM_SPIKE_BINS; ++bin)
        if (spikes[gl_InstanceIndex].profile[arm*uint(BLOOM_SPIKE_BINS)+uint(bin)].x > 0.0) {
            first = min(first, bin);
            last = bin;
        }
    center_bin = dot(center, normal)-spikes[gl_InstanceIndex].profile_origin[arm];
    vec2 corner = CORNERS[gl_VertexIndex%6];
    local = vec2(corner.x*(spike.half_length+13.0), corner.y < 0.0 ? float(first)-1.5-center_bin : float(last)+1.5-center_bin);
    vec2 position = center+axis*local.x+normal*local.y;
    gl_Position = vec4(position/spike.extent*2.0-1.0, 0.0, 1.0);
    amplitude = energy*spike.scale;
    profile_index = uvec2(gl_InstanceIndex, arm*uint(BLOOM_SPIKE_BINS));
}
