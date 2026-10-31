#ifndef SF_PROCEDURAL_NOISE
#define SF_PROCEDURAL_NOISE

uvec3 noise_pcg3d(uvec3 v) {
    v = v*1664525u+1013904223u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    v ^= v >> 16u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    return v;
}

float noise_hash(ivec3 cell) {
    return float(noise_pcg3d(uvec3(cell)).x)*(1.0/4294967296.0);
}

float value_noise(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = p-cell;
    vec3 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    ivec3 c = ivec3(cell);
    float n000 = noise_hash(c), n100 = noise_hash(c+ivec3(1,0,0));
    float n010 = noise_hash(c+ivec3(0,1,0)), n110 = noise_hash(c+ivec3(1,1,0));
    float n001 = noise_hash(c+ivec3(0,0,1)), n101 = noise_hash(c+ivec3(1,0,1));
    float n011 = noise_hash(c+ivec3(0,1,1)), n111 = noise_hash(c+ivec3(1,1,1));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

const mat3 NOISE_OCTAVE_ROTATION = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);

float fbm_noise(vec3 p, int octaves, float lacunarity, float gain) {
    float sum = 0.0, amplitude = 0.5, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amplitude*value_noise(p);
        norm += amplitude;
        p = NOISE_OCTAVE_ROTATION*p*lacunarity;
        amplitude *= gain;
    }
    return sum/max(norm, 1e-6);
}

float billow_noise(vec3 p, int octaves) {
    float sum = 0.0, amplitude = 0.5, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amplitude*(1.0-abs(2.0*value_noise(p)-1.0));
        norm += amplitude;
        p = NOISE_OCTAVE_ROTATION*p*2.03;
        amplitude *= 0.5;
    }
    return sum/max(norm, 1e-6);
}

float interleaved_gradient_noise(vec2 pixel) {
    return fract(52.9829189*fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

#endif
