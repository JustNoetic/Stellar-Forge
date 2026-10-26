vec3 sunlight(vec3 p) {
    vec3 light = comparison_stretch(COMPARISON_LIGHT);
    vec3 direction = normalize(light);
    float mu = dot(p,direction)/length(p);
    if (mu <= sun_horizon(p)) return vec3(0);
    return exp(-textureLod(transmittance_lut,transmittance_uv(length(p),mu),0).rgb/length(light));
}

void integrate_air(vec3 origin, float start, float end, inout vec3 radiance, inout vec3 throughput) {
    if (end <= start || cmp.resources.w <= 0.0) return;
    vec3 o = comparison_stretch(origin), d = comparison_stretch(vec3(0,0,-1));
    vec3 light = normalize(comparison_stretch(COMPARISON_LIGHT));
    int steps = int(cmp.resources.w);
    float closest = clamp(-dot(o+d*start,d)/dot(d,d),0.0,end-start);
    float phase_cosine = -COMPARISON_LIGHT.z;
    for (int i = 0; i < steps; ++i) {
        float fa = float(i)/float(steps), fb = float(i+1)/float(steps);
        float distance = end-start;
        float split = closest/distance;
        float a = fa <= split && split > 0.0 ? closest*(1.0-pow(1.0-fa/split,2.0)) :
            closest+(distance-closest)*pow((fa-split)/max(1.0-split,1e-6),2.0);
        float b = fb <= split && split > 0.0 ? closest*(1.0-pow(1.0-fb/split,2.0)) :
            closest+(distance-closest)*pow((fb-split)/max(1.0-split,1e-6),2.0);
        float ds = b-a;
        vec3 p = o+d*(start+0.5*(a+b));
        vec3 rayleigh,mie,extinction;
        medium(p,rayleigh,mie,extinction);
        vec3 source = (rayleigh*rayleigh_phase(phase_cosine)+aerosol_source(p,phase_cosine))*sunlight(p);
        source += multiple_scattering_source(multiscatter_lut,p,light,rayleigh+mie);
        source *= ATM_PI;
        radiance += throughput*source*segment_integral(extinction,ds);
        throughput *= exp(-extinction*ds);
    }
}

