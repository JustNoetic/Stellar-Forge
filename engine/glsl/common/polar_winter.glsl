#ifndef POLAR_WINTER_GLSL
#define POLAR_WINTER_GLSL

// SpaceEngine Solstice Winter Model: suppressed aerosol Mie haze and azure Rayleigh in-scatter boost.
// Restricted to planetary atmospheres containing methane (e.g. Saturn, Uranus) with ring shadowing.
void eval_polar_winter(vec3 pos_sph, vec2 solstice_params, float has_rings, float has_methane,
                       out float polar_haze_factor, out vec3 polar_rayleigh_boost) {
    if (solstice_params.x < 1e-4 || has_rings < 0.5 || has_methane < 0.5) {
        polar_haze_factor = 1.0;
        polar_rayleigh_boost = vec3(1.0);
        return;
    }
    vec3 pole_dir_norm = (length(u_pole_obl.xyz) > 1e-4) ? normalize(u_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    vec3 dir_norm = normalize(pos_sph);
    float frag_pole_dot = dot(dir_norm, pole_dir_norm);
    float mid_sin_lat = abs(frag_pole_dot);
    float lat_factor = smoothstep(0.1, 0.7, mid_sin_lat);
    float is_winter = step(solstice_params.y * frag_pole_dot, 0.0);
    float winter_solstice_effect = lat_factor * is_winter * solstice_params.x * has_rings * has_methane;
    polar_haze_factor = mix(1.0, 0.05, winter_solstice_effect);
    polar_rayleigh_boost = mix(vec3(1.0), vec3(0.65, 0.95, 2.5), winter_solstice_effect);
}

#endif // POLAR_WINTER_GLSL
