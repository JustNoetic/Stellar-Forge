float refract_precise_sin(float x) {
    return abs(x) < 0.01 ? x*(1.0-x*x*(1.0/6.0)) : sin(x);
}

float refract_precise_cos(float x) {
    return abs(x) < 0.01 ? 1.0-0.5*x*x : cos(x);
}

float refract_precise_atan(float y, float x) {
    float t = y/max(abs(x), 1e-30);
    return x > 0.0 && abs(y) < 0.01*x ? t*(1.0-t*t*(1.0/3.0)) : atan(y, x);
}

float refract_bend_at(int i) {
    vec4 v = atm.refract_bend[i >> 2];
    int c = i & 3;
    return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

vec3 refracted_view_direction(vec2 ndc, out float radial_slope) {
    radial_slope = 0.0;
    vec3 n = vec3(ndc, 1.0);
    vec3 r = vec3(dot(atm.refract_view[0].xyz, n), dot(atm.refract_view[1].xyz, n), dot(atm.refract_view[2].xyz, n));
    if (r.z <= 1e-4) return vec3(0.0, 0.0, 1.0);
    float a1 = r.x/r.z;
    float a2 = r.y/r.z;
    float s0 = atm.refract_trig.x, c0 = atm.refract_trig.y;
    float st0 = atm.refract_trig.z, ct0 = atm.refract_trig.w;
    float u = s0-a2*c0;
    float rho = sqrt(max(u*u+a1*a1, 1e-30));
    float rho_minus_u = u >= 0.0 ? (a1*a1)/max(rho+u, 1e-30) : rho-u;
    float sin_d = rho_minus_u*c0-a2;
    float cos_d = 1.0+s0*rho_minus_u;
    float dtheta = refract_precise_atan(sin_d, cos_d);
    float sin_phi = a1/rho;
    float one_minus_cos_phi = rho_minus_u/rho;
    float ratio = clamp((atm.refract_domain.x-dtheta)*atm.refract_domain.y, 0.0, 1.0);
    float t = 1.0-sqrt(ratio);
    float f = t*63.0;
    int i0 = min(int(f), 62);
    float w = f-float(i0);
    float d0 = refract_bend_at(i0);
    float d1 = refract_bend_at(i0+1);
    float ddelta = mix(d0, d1, w);
    if (ratio > 0.0 && ratio < 1.0)
        radial_slope = max((d1-d0)*63.0*0.5*atm.refract_domain.y/max(sqrt(ratio), 1e-4), 0.0);
    float dtheta_t = dtheta+ddelta;
    float sin_dt = refract_precise_sin(dtheta_t);
    float cos_dt = refract_precise_cos(dtheta_t);
    float sin_tt = st0*cos_dt+ct0*sin_dt;
    vec3 v = vec3(sin_tt*sin_phi, -sin_dt+ct0*sin_tt*one_minus_cos_phi, cos_dt-st0*sin_tt*one_minus_cos_phi);
    vec3 raw = atm.refract_to_atm[0].xyz*v.x+atm.refract_to_atm[1].xyz*v.y+atm.refract_to_atm[2].xyz*v.z;
    return normalize(raw);
}

float depth_atmosphere_distance(vec2 ndc, float depth, vec3 d) {
    vec4 surface = atm.inverse_view_projection*vec4(ndc, depth, 1.0);
    vec3 world = surface.xyz/surface.w;
    if (atm.refract_domain.w <= 0.5) return length(world_to_atmosphere(world))*149597870.7;
    vec3 pole = vec3(dot(atm.world_pole.xyz, atm.axis_x.xyz),
                     dot(atm.world_pole.xyz, atm.axis_y.xyz),
                     dot(atm.world_pole.xyz, atm.axis_z.xyz));
    float dp = dot(d, pole);
    float q = atm.reference_shape.y;
    float scale = inversesqrt(max(1.0-(1.0-q*q)*dp*dp, 1e-6));
    return length(world)*scale*149597870.7;
}

vec2 refract_limb_steps(vec2 ndc, ivec2 size, vec3 radial, float radial_slope, vec2 base_step, float q_len) {
    vec2 grad = base_step/q_len;
    if (atm.refract_domain.w > 0.5 && radial_slope > 0.0) {
        vec3 n = vec3(ndc, 1.0);
        float ry = dot(atm.refract_view[1].xyz, n);
        float rz = dot(atm.refract_view[2].xyz, n);
        if (rz > 1e-4) {
            float a2 = ry/rz;
            vec2 da2 = (vec2(atm.refract_view[1].x, atm.refract_view[1].y)-a2*vec2(atm.refract_view[2].x, atm.refract_view[2].y))/rz;
            float proj = dot(radial, atm.refract_to_atm[1].xyz)/max(length(atm.refract_to_atm[2].xyz), 1e-6);
            grad += proj*(radial_slope*da2);
        }
    }
    return abs(grad)*2.0/vec2(size);
}
