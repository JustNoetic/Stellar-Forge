vec3 bloom_finite(vec3 value) {
    return vec3(isnan(value.x) ? 0.0 : clamp(value.x, 0.0, 65000.0),
                isnan(value.y) ? 0.0 : clamp(value.y, 0.0, 65000.0),
                isnan(value.z) ? 0.0 : clamp(value.z, 0.0, 65000.0));
}

vec3 bloom_extract(vec3 value, float limit) {
    value = bloom_finite(value);
    float peak = max(value.r, max(value.g, value.b));
    float knee = limit*0.5;
    if (peak <= knee) return value;
    return value*((limit-knee*knee/peak)/peak);
}

vec3 bloom_sample(sampler2D source, vec2 uv) {
    vec2 size = vec2(textureSize(source, 0));
    vec2 weight = clamp(uv*size+0.5, 0.0, 1.0)*clamp((1.0-uv)*size+0.5, 0.0, 1.0);
    return textureLod(source, uv, 0.0).rgb*weight.x*weight.y;
}

const int BLOOM_SPIKE_BINS = 26;
const vec3 BLOOM_LUMINANCE = vec3(0.2126, 0.7152, 0.0722);

struct BloomSpike {
    vec4 center_variance;
    vec4 energy_covariance;
    vec4 profile_origin;
    vec4 profile[2*BLOOM_SPIKE_BINS];
};

vec2 bloom_spike_normal(vec2 direction, uint arm) {
    return arm == 0u ? vec2(-direction.y, direction.x) : -direction;
}

float bloom_spline(float distance) {
    distance = abs(distance);
    return distance < 0.5 ? 0.75-distance*distance : distance < 1.5 ? 0.5*(1.5-distance)*(1.5-distance) : 0.0;
}
