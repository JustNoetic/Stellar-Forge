// All cosines use photon travel: +1 is forward scattering. Alpha is vertical.
const float RING_PI = 3.14159265358979323846;
const float RING_MAX_G = 0.99;
const float RING_MAX_ALPHA = 0.9999999;

float ring_absorbed(float depth) {
    return depth < 0.001 ? depth * (1.0 - depth * (0.5 - depth / 6.0))
                         : 1.0 - exp(-depth);
}
float ring_tau(float alpha) {
    alpha = clamp(alpha, 0.0, RING_MAX_ALPHA);
    // log(1-alpha) loses the thinnest dust profiles in float arithmetic.
    return alpha < 0.001 ? alpha * (1.0 + alpha * (0.5 + alpha / 3.0))
                         : -log(1.0 - alpha);
}
float ring_hg(float g, float mu) {
    g = clamp(g, -RING_MAX_G, RING_MAX_G);
    mu = clamp(mu, -1.0, 1.0);
    float ag = abs(g);
    float d = (1.0-ag)*(1.0-ag) + 2.0*ag*(1.0-sign(g)*mu);
    return (1.0-g*g)/(4.0*RING_PI*d*sqrt(d));
}
float ring_cs(float g, float mu) {
    g = clamp(g, -RING_MAX_G, RING_MAX_G);
    mu = clamp(mu, -1.0, 1.0);
    return ring_hg(g,mu)*1.5*(1.0+mu*mu)/(2.0+g*g);
}
// x: normalized phase, y: authored multiple-scattering weight.
vec2 ring_phase(float mu, float material_alpha, vec4 props, bool textured) {
    float forward, multiple;
    if (textured) {
        multiple = clamp((material_alpha-0.1)/0.5, 0.0, 1.0);
        forward = mix(0.95, 0.5, multiple);
        return vec2(forward*ring_hg(props.r,mu)+(1.0-forward)*ring_hg(props.g,mu),multiple);
    }
    float balance = clamp(props.b,0.0,1.0);
    forward = mix(0.1,0.9,balance);
    multiple = clamp(1.0-balance*0.7,0.1,1.0);
    return vec2(forward*ring_cs(props.r,mu)+(1.0-forward)*ring_cs(props.g,mu),multiple);
}
// x: single-scattering slab, y: isotropic multiple-scattering approximation.
vec2 ring_transfer(float tau, float mu_v, float mu_0, bool lit) {
    if (tau <= 0.0 || mu_0 <= 0.0 || mu_v <= 0.0) return vec2(0.0);
    mu_v = max(mu_v,1e-7);
    mu_0 = max(mu_0,1e-7);
    float vd = tau/mu_v, ld = tau/mu_0;
    if (!lit) {
        float delta = abs(ld-vd);
        float ratio = delta < 0.001 ? 1.0-delta*(0.5-delta/6.0)
                                   : ring_absorbed(delta)/delta;
        return vec2(vd*exp(-min(vd,ld))*ratio,0.0);
    }
    float single = mu_0/(mu_v+mu_0)*ring_absorbed(vd+ld);
    float gamma = sqrt(0.08);
    float hv = (1.0+2.0*mu_v)/(1.0+2.0*mu_v*gamma);
    float h0 = (1.0+2.0*mu_0)/(1.0+2.0*mu_0*gamma);
    return vec2(single,0.92*single*max(0.0,hv*h0-1.0)/(4.0*RING_PI));
}
float ring_opposition(float cos_phase, float tau) {
    float angle = acos(clamp(cos_phase,-1.0,1.0));
    float density = clamp(tau/1.5,0.0,1.0);
    return (1.0+0.8*density/(1.0+angle/0.07))
           *(1.0+0.3*density*exp(-angle/0.006));
}
float ring_radiance(float tau, float material_alpha, float mu_v, float sun,
                    float mu, vec4 props, bool textured, bool lit) {
    vec2 transfer = ring_transfer(tau,mu_v,abs(sun),lit);
    vec2 phase = ring_phase(mu,material_alpha,props,textured);
    float single = transfer.x*phase.x;
    if (lit) single *= ring_opposition(-mu,tau);
    else single *= max(0.0,props.a);
    return single+transfer.y*phase.y;
}
