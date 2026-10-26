#ifndef SF_STELLAR_DISK
#define SF_STELLAR_DISK
const float STELLAR_PI = 3.141592653589793;
const vec2 STELLAR_QUADRATURE[8] = vec2[8](
    vec2(-0.9602898565, 0.1012285363), vec2(-0.7966664774, 0.2223810345),
    vec2(-0.5255324099, 0.3137066459), vec2(-0.1834346425, 0.3626837834),
    vec2( 0.1834346425, 0.3626837834), vec2( 0.5255324099, 0.3137066459),
    vec2( 0.7966664774, 0.2223810345), vec2( 0.9602898565, 0.1012285363));
const vec2 STELLAR_AZIMUTH[4] = vec2[4](
    vec2(-0.8611363116, 0.3478548451), vec2(-0.3399810436, 0.6521451549),
    vec2( 0.3399810436, 0.6521451549), vec2( 0.8611363116, 0.3478548451));

float stellar_lambert_irradiance(float mu, float sin_radius) {
    float s = clamp(sin_radius, 1e-8, 1.0);
    if (mu >= s) return mu;
    if (mu <= -s) return 0.0;
    float c = sqrt(max(1.0-s*s, 0.0));
    float delta = sqrt(max(s*s-mu*mu, 0.0));
    float v = clamp(c*mu/(s*sqrt(max(1.0-mu*mu, 1e-16))), -1.0, 1.0);
    float arc = acos(-v);
    if (delta < 0.1*c) {
        float z = delta/c, z2 = z*z;
        float difference = z*z2*(-1.0/3.0+z2*(1.0/5.0-z2/7.0));
        return max((z+mu*arc+difference/(s*s))/STELLAR_PI, 0.0);
    }
    return max((atan(delta, c)-c*delta+s*s*mu*arc)/(STELLAR_PI*s*s), 0.0);
}

vec3 stellar_tangent(vec3 up, vec3 sun) {
    vec3 tangent = sun-up*dot(up, sun);
    if (dot(tangent, tangent) < 1e-10)
        tangent = cross(up, abs(up.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0));
    return normalize(tangent);
}

vec2 stellar_elevation_range(float elevation, float radius, float horizon) {
    return vec2(max(horizon, elevation-radius), min(0.5*STELLAR_PI, elevation+radius));
}

vec3 stellar_slice(vec2 range, float elevation, float radius, vec2 node) {
    float half_span = max(range.y-range.x, 0.0)*0.5;
    float h = (range.x+range.y)*0.5+half_span*node.x;
    float delta = h-elevation;
    float denominator = cos(h)*cos(elevation);
    float q = sin(0.5*(radius+delta))*sin(0.5*(radius-delta));
    float phi = 2.0*asin(sqrt(clamp(q/max(denominator, 1e-12), 0.0, 1.0)));
    float weight = half_span*node.y*cos(h)
                 /(STELLAR_PI*max(pow(sin(radius), 2.0), 1e-16));
    return vec3(h, phi, weight);
}

vec3 stellar_slice(vec2 range, float elevation, float radius, int index) {
    return stellar_slice(range, elevation, radius, STELLAR_QUADRATURE[index]);
}
#endif
