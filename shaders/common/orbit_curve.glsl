#ifndef ORBIT_CURVE_GLSL
#define ORBIT_CURVE_GLSL

struct OrbitData {
    dvec4 d0;
    dvec4 d1;
    dvec4 d2;
    dvec4 d3;
    dvec4 d4;
};

dvec2 orbit_sincos(double angle) {
    const double pi = 3.14159265358979323846lf;
    angle = mod(angle + pi, 2.0lf * pi) - pi;
    double sign_c = 1.0lf;
    if (angle > 0.5lf * pi) {
        angle = pi - angle;
        sign_c = -1.0lf;
    } else if (angle < -0.5lf * pi) {
        angle = -pi - angle;
        sign_c = -1.0lf;
    }
    double c = 1.0lf, s = angle;
    double tc = c, ts = s;
    for (int k = 1; k <= 12; ++k) {
        tc *= -angle * angle / double((2*k - 1) * (2*k));
        ts *= -angle * angle / double((2*k) * (2*k + 1));
        c += tc;
        s += ts;
    }
    return dvec2(sign_c * c, s);
}

dvec3 orbit_curve_point(OrbitData orbit, float t) {
    double angle = orbit.d2.w + double(t) * orbit.d3.w;
    if (orbit.d4.z > 0.5 && t == 1.0) angle = orbit.d2.w;
    dvec2 cs = orbit_sincos(angle);
    double e = orbit.d1.w;
    if (e < 0.999) {
        double a = orbit.d0.w / (1.0lf - e * e);
        return orbit.d2.xyz + a * (cs.x - e) * orbit.d0.xyz
             + a * sqrt(1.0lf - e * e) * cs.y * orbit.d1.xyz;
    }
    double r = orbit.d0.w / max(1.0lf + e * cs.x, 1e-15lf);
    return orbit.d2.xyz + (cs.x * orbit.d0.xyz + cs.y * orbit.d1.xyz) * r;
}

#endif
