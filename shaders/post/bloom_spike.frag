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
layout(location = 0) in vec2 local;
layout(location = 1) flat in vec3 amplitude;
layout(location = 2) flat in float center_bin;
layout(location = 3) flat in uvec2 profile_index;
layout(location = 0) out vec4 color;

float along_profile(float x, float half_width) {
    float t = clamp((abs(x)/spike.half_length-0.55)/0.45, 0.0, 1.0);
    float taper = 1.0-t*t*(3.0-2.0*t);
    float w = spike.width;
    return taper*taper*w/(2.0*half_width)*atan(2.0*half_width*w, w*w+x*x-half_width*half_width);
}

void main() {
    float u = center_bin+local.y;
    int nearest = int(floor(u+0.5));
    float value = 0.0;
    for (int bin = max(nearest-1, 0); bin <= min(nearest+1, BLOOM_SPIKE_BINS-1); ++bin) {
        vec4 slice = spikes[profile_index.x].profile[profile_index.y+uint(bin)];
        if (slice.x <= 0.0) continue;
        float half_width = sqrt(3.0*max(slice.z-slice.y*slice.y, 0.0)+0.25);
        value += bloom_spline(u-float(bin))*slice.x*along_profile(local.x-slice.y, half_width);
    }
    color = vec4(amplitude*value, 0.0);
}
