// Uniform-radiance circular source: four Gauss nodes in projected area,
// sixteen symmetric azimuth nodes. Weight integrates dOmega/(pi*sin(alpha)^2).
const vec4 RING_DISK_R2 = vec4(0.0694318442,0.3300094782,0.6699905218,0.9305681558);
const vec4 RING_DISK_W = vec4(0.1739274226,0.3260725774,0.3260725774,0.1739274226);
const int RING_DISK_SAMPLES = 64;
vec4 ring_disk_sample(vec3 center, vec3 pole, float sin_radius, int sample_index) {
    vec3 vertical = pole-center*dot(pole,center);
    if (dot(vertical,vertical)<1e-8) {
        vec3 axis = abs(center.x)<0.8 ? vec3(1,0,0) : vec3(0,0,1);
        vertical = axis-center*dot(axis,center);
    }
    vertical = normalize(vertical);
    vec3 horizontal = cross(center,vertical);
    int row = sample_index/16;
    float azimuth = (float(sample_index%16)+0.5)*(2.0*RING_PI/16.0);
    float r = min(sin_radius,0.999)*sqrt(RING_DISK_R2[row]);
    float cosine = sqrt(1.0-r*r);
    vec3 direction = center*cosine+r*(vertical*sin(azimuth)+horizontal*cos(azimuth));
    return vec4(direction,RING_DISK_W[row]/(16.0*cosine));
}
bool ring_resolve_disk(float sun, float sin_radius) {
    // Resolve a source crossing the slab and sources large enough to broaden
    // the narrowest supported phase lobe. Otherwise use its point limit.
    return sin_radius>0.0 && (abs(sun)<2.0*sin_radius || sin_radius>0.001);
}
