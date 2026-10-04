// Shared disk-integrated surface response, not a normalized phase function.
// 721 samples/body, 0..pi; I/F averaged across the projected disk.
layout(std430, binding = 12) readonly buffer SurfacePhaseBlock {
    float u_surface_phase[];
};
float surface_phase_lookup(float angle, SurfaceMaterial m) {
    float pos = clamp(angle/PI*720.0, 0.0, 720.0);
    int lo = min(719, int(pos));
    int offset = int(m.opposition.w+0.5);
    return mix(u_surface_phase[offset+lo], u_surface_phase[offset+lo+1], pos-float(lo));
}
float surface_disk_response(float cos_phase, float angular_radius, SurfaceMaterial m) {
    float c = clamp(cos_phase,-1.0,1.0), alpha = acos(c);
    if (angular_radius < 1e-5) return surface_phase_lookup(alpha,m);
    float beta = min(1.0,angular_radius)*sqrt(0.625);
    float s = sqrt(max(0.0,1.0-c*c)), cb = cos(beta), sb = sin(beta);
    float value = surface_phase_lookup(alpha,m);
    value += surface_phase_lookup(acos(clamp(c*cb+s*sb,-1.0,1.0)),m);
    value += surface_phase_lookup(acos(clamp(c*cb-s*sb,-1.0,1.0)),m);
    value += 2.0*surface_phase_lookup(acos(clamp(c*cb,-1.0,1.0)),m);
    return value*0.2;
}
