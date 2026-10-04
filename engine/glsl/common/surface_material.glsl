// Hapke 1984/1986 with HG2, isotropic H approximation and coherent opposition.
// CPU counterpart: rendering/surface_materials.py. Roughness equations follow
// the public-domain USGS ISIS Hapke implementation (see counterpart attribution).
// Textures are reference reflectance at i=30, e=0, phase=30 degrees.
struct SurfaceMaterial {
    vec4 grain;       // model (0=Lambert), w, HG width b, backward fraction
    vec4 surface;     // SH strength, SH width, roughness degrees, brightness
    vec4 opposition;  // CB strength, CB width, reference normalization, LUT offset
};
layout(std430, binding = 11) readonly buffer SurfaceMaterialBlock {
    SurfaceMaterial u_surface_materials[];
};

float hapke_raw(float mu0, float mu, float cos_phase, SurfaceMaterial m) {
    if (mu0 <= 0.0 || mu <= 0.0) return 0.0;
    mu0 = min(1.0, mu0); mu = min(1.0, mu);
    float c = clamp(cos_phase, -1.0, 1.0);
    float b = m.grain.z, back = m.grain.w;
    float pf = (1.0-b*b) * ((1.0-back)/pow(max(1e-9, 1.0+b*b+2.0*b*c), 1.5)
                              + back/pow(max(1e-9, 1.0+b*b-2.0*b*c), 1.5));
    float tan_half = sqrt(max(0.0, (1.0-c)/max(1e-12, 1.0+c)));
    float sh = m.surface.x/(1.0+tan_half/m.surface.y);
    float x = tan_half/m.opposition.y;
    float ratio = x > 0.001 ? (1.0-exp(-x))/x : 1.0-0.5*x+x*x/6.0;
    float cb = m.opposition.x*(1.0+ratio)/(2.0*(1.0+x)*(1.0+x));
    float u0 = mu0, u = mu, shadow = 1.0;
    if (m.surface.z > 0.001) {
        float t = tan(radians(m.surface.z));
        float chi = inversesqrt(1.0+PI*t*t);
        float si = sqrt(max(0.0, 1.0-mu0*mu0)), se = sqrt(max(0.0, 1.0-mu*mu));
        float ci = mu0/max(1e-10, si)/t, ce = mu/max(1e-10, se)/t;
        float e1i = exp(-2.0*ci/PI), e1e = exp(-2.0*ce/PI);
        float e2i = exp(-ci*ci/PI), e2e = exp(-ce*ce/PI);
        float u00 = chi*(mu0+si*t*e2i/(2.0-e1i));
        float u10 = chi*(mu+se*t*e2e/(2.0-e1e));
        float cp = si*se > 1e-10 ? clamp((c-mu0*mu)/max(1e-10, si*se), -1.0, 1.0) : 1.0;
        float psi = acos(cp), hs = 0.5*(1.0-cp);
        float f = exp(-2.0*sqrt(max(0.0, (1.0-cp)/max(1e-12, 1.0+cp))));
        float q;
        if (mu0 >= mu) {
            float den = max(1e-10, 2.0-e1e-psi/PI*e1i);
            u0 = chi*(mu0+si*t*(cp*e2e+hs*e2i)/den);
            u = chi*(mu+se*t*(e2e-hs*e2i)/den);
            q = chi*mu0/max(1e-10, u00);
        } else {
            float den = max(1e-10, 2.0-e1i-psi/PI*e1e);
            u0 = chi*(mu0+si*t*(e2i-hs*e2e)/den);
            u = chi*(mu+se*t*(cp*e2i+hs*e2e)/den);
            q = chi*mu/max(1e-10, u10);
        }
        shadow = u*mu0*chi/max(1e-10, u10*u00*(1.0-f+f*q));
    }
    float gamma = sqrt(1.0-m.grain.y);
    float h0 = (1.0+2.0*u0)/(1.0+2.0*u0*gamma);
    float h = (1.0+2.0*u)/(1.0+2.0*u*gamma);
    return max(0.0, 0.25*m.grain.y*u0/max(1e-10, u0+u) * ((1.0+sh)*pf+h0*h-1.0) * shadow * (1.0+cb));
}

float surface_point_response(vec3 N, vec3 V, vec3 L, SurfaceMaterial m) {
    return hapke_raw(dot(N,L), dot(N,V), dot(V,L), m)*m.opposition.z*m.surface.w;
}

float surface_direct_response(vec3 N, vec3 V, vec3 L, float sin_star_radius,
                              float legacy_cosine, SurfaceMaterial m) {
    if (m.grain.x < 0.5) return legacy_cosine*m.surface.w;
    float response = surface_point_response(N,V,L,m);
    float radius = asin(clamp(sin_star_radius,0.0,0.84));
    // Resolve a finite solar disk near the terminator and the opposition peak.
    if (radius < 1e-5 || (abs(dot(N,L)) > 2.0*radius
        && dot(V,L) < cos(4.0*radius) && radius < 0.05)) return response;
    vec3 axis = abs(L.z) < 0.9 ? vec3(0,0,1) : vec3(0,1,0);
    vec3 T = normalize(cross(L,axis)), B = cross(L,T);
    float a = radius*sqrt(0.625), c = cos(a), s = sin(a);
    response += surface_point_response(N,V,L*c+T*s,m);
    response += surface_point_response(N,V,L*c-T*s,m);
    response += surface_point_response(N,V,L*c+B*s,m);
    response += surface_point_response(N,V,L*c-B*s,m);
    return response*0.2;
}
