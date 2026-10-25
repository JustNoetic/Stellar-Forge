#ifndef SF_SURFACE_PHASE
#define SF_SURFACE_PHASE
layout(std430, set = 0, binding = 25) readonly buffer SurfacePhases { float material_phases[]; };

float material_phase_lookup(SurfaceMaterial m, float angle) {
    if (m.grain.x < 0.5) {
        float beta = MATERIAL_PI-angle;
        float b2 = beta*beta;
        float phase = beta < 0.1 ? beta*b2*(1.0/3.0-b2/30.0+b2*b2/840.0)
            : sin(beta)-beta*cos(beta);
        return (2.0/3.0)*m.surface.w*max(phase, 0.0)/MATERIAL_PI;
    }
    float pos = clamp(angle/MATERIAL_PI, 0.0, 1.0)*720.0;
    uint lo = min(uint(pos), 719u), offset = uint(m.opposition.w);
    return mix(material_phases[offset+lo], material_phases[offset+lo+1u], pos-float(lo));
}

float material_disk_response(SurfaceMaterial m, float cosine, float angular_radius) {
    float c = clamp(cosine, -1.0, 1.0), angle = acos(c);
    float value = material_phase_lookup(m, angle);
    if (angular_radius < 1e-5) return value;
    float beta = min(1.0, angular_radius)*sqrt(0.625), cb = cos(beta), sb = sin(beta);
    float s = sqrt(max(0.0, 1.0-c*c));
    return 0.2*(value+material_phase_lookup(m, acos(clamp(c*cb+s*sb, -1.0, 1.0)))
        +material_phase_lookup(m, acos(clamp(c*cb-s*sb, -1.0, 1.0)))
        +2.0*material_phase_lookup(m, acos(clamp(c*cb, -1.0, 1.0))));
}
#endif
