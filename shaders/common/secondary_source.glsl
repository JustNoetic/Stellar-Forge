vec2 secondary_extent[ATMOSPHERE_SECONDARY_LIGHTS];
vec3 secondary_rayleigh[ATMOSPHERE_SECONDARY_LIGHTS];
#ifdef ATMOSPHERE_SECONDARY_RING
#include "ringshine_lookup.glsl"
vec3 secondary_ring_pole, secondary_ring_scale, secondary_ring_stars;
int secondary_ring_host;
#endif
#ifndef ATMOSPHERE_RAYLEIGH_ONLY
vec3 secondary_mie[ATMOSPHERE_SECONDARY_LIGHTS];
#ifndef ATMOSPHERE_NO_CUSTOM_AEROSOLS
vec3 secondary_particles[4*ATMOSPHERE_SECONDARY_LIGHTS];
#endif
#endif

void prepare_secondary(vec3 view) {
    for (int s = 0; s < ATMOSPHERE_SECONDARY_LIGHTS; ++s) {
#ifdef ATMOSPHERE_SECONDARY_RING
        if (secondary.direction[s].w < 0.0) {
            vec3 pole = atm.world_pole.xyz;
            secondary_ring_pole = normalize(vec3(dot(pole, atm.axis_x.xyz), dot(pole, atm.axis_y.xyz), dot(pole, atm.axis_z.xyz)));
            secondary_ring_scale = secondary.irradiance[s].rgb*secondary.irradiance[s].w;
            secondary_ring_host = int(-secondary.direction[s].w)-1;
            vec3 stars = vec3(0.0);
            for (int star = 0; star < int(atm.camera.w); ++star) stars += atm.sun_irradiance[star].rgb;
            secondary_ring_stars = mix(vec3(0.0), 1.0/max(stars, vec3(1e-30)), greaterThan(stars, vec3(0.0)));
            continue;
        }
#endif
        float phase = dot(view, secondary.direction[s].xyz);
        secondary_extent[s] = vec2(sin(secondary.direction[s].w), cos(secondary.direction[s].w));
        secondary_rayleigh[s] = secondary.irradiance[s].rgb*rayleigh_phase(phase);
#ifndef ATMOSPHERE_RAYLEIGH_ONLY
        secondary_mie[s] = secondary.irradiance[s].rgb*mie_phase(phase);
#ifndef ATMOSPHERE_NO_CUSTOM_AEROSOLS
        for (int layer = 0; layer < int(atm.aerosol_count.x); ++layer)
            secondary_particles[s*4+layer] = secondary.irradiance[s].rgb*particle_phase(layer, phase);
#endif
#endif
    }
}

#ifdef ATMOSPHERE_SECONDARY_RING
vec3 diffuse_ring_irradiance(int s, vec3 p, out float mu) {
    mu = 0.0;
    vec3 pole = secondary_ring_pole;
    vec3 physical = p-pole*dot(p, pole)*(1.0-atm.reference_shape.y);
    float slat = dot(physical, pole)/max(length(physical), 1e-20);
    vec3 planar = p-pole*dot(p, pole);
    float planar_length = length(planar);
    planar = planar_length > 1e-20 ? planar/planar_length : normalize(cross(pole, abs(pole.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
    vec3 light = vec3(0.0);
    float cosine = 0.0, scalar = 0.0;
    for (int star = 0; star < int(atm.camera.w); ++star) {
        vec3 weight = atm.sun_irradiance[star].rgb*secondary_ring_stars;
        if (!any(greaterThan(weight, vec3(0.0)))) continue;
        vec3 anti = -atm.sun_direction[star].xyz;
        anti -= pole*dot(anti, pole);
        float anti_length = length(anti);
        anti = anti_length > 1e-6 ? anti/anti_length : planar;
        float phi = atan(dot(cross(anti, planar), pole), dot(planar, anti));
        vec4 tile = ringshine_sample_rgba(phi, slat, secondary_ring_host, star);
        float projected = dot(tile.rgb, vec3(0.2126, 0.7152, 0.0722)), share = dot(weight, vec3(0.2126, 0.7152, 0.0722));
        if (projected > 0.0) light += weight*tile.rgb*(tile.a/projected);
        cosine += share*projected;
        scalar += share*tile.a;
    }
    if (!(scalar > 0.0)) return vec3(0.0);
    mu = clamp(cosine/scalar, 0.0, 1.0);
    return secondary_ring_scale*light;
}
#endif

vec3 secondary_source(sampler2D transmission, sampler2D multiscatter, vec3 p, vec3 rayleigh, vec3 medium_mie) {
    float radius = length(p), horizon = sun_horizon(p);
    float horizon_width = sqrt(max(1.0-horizon*horizon, 0.0));
    vec3 up = p/radius;
    vec2 multiple_uv = multiscatter_uv(p, vec3(0.0));
    vec3 direct_rayleigh = vec3(0.0), multiple = vec3(0.0);
#ifdef ATMOSPHERE_SECONDARY_RING
    vec3 diffuse = vec3(0.0);
#endif
#ifndef ATMOSPHERE_RAYLEIGH_ONLY
    vec3 direct_mie = vec3(0.0);
#ifndef ATMOSPHERE_NO_CUSTOM_AEROSOLS
    vec3 particles[4];
    for (int layer = 0; layer < int(atm.aerosol_count.x); ++layer) particles[layer] = vec3(0.0);
#endif
#endif
    for (int s = 0; s < ATMOSPHERE_SECONDARY_LIGHTS; ++s) {
#ifdef ATMOSPHERE_SECONDARY_RING
        if (secondary.direction[s].w < 0.0) {
            float mu;
            vec3 light = diffuse_ring_irradiance(s,p,mu);
            if (any(greaterThan(light,vec3(0.0)))) {
                diffuse += light*exp(-textureLod(transmission,transmittance_uv(radius,max(mu,horizon+1e-7)),0.0).rgb)/(4.0*ATM_PI);
                if (atm.depth_params.z < 0.5) {
                    multiple_uv.x = (0.5+(mu*0.5+0.5)*float(MULTISCATTER_SIZE.x-1))/float(MULTISCATTER_SIZE.x);
                    multiple += light*textureLod(multiscatter,multiple_uv,0.0).rgb;
                }
            }
            continue;
        }
#endif
        float mu = dot(up, secondary.direction[s].xyz);
        float width = max(horizon_width*secondary_extent[s].x, 1e-7);
        float center = horizon*secondary_extent[s].y;
        float visible = smoothstep(-width, width, mu-center);
        if (visible > 0.0) {
            float sample_mu = clamp(max(mu, 0.5*(mu+center+width)), horizon+1e-7, 1.0);
            vec3 light = visible*(2.0/(1.0+secondary_extent[s].y))
                *exp(-textureLod(transmission, transmittance_uv(radius, sample_mu), 0.0).rgb);
            direct_rayleigh += light*secondary_rayleigh[s];
#ifndef ATMOSPHERE_RAYLEIGH_ONLY
            direct_mie += light*secondary_mie[s];
#ifndef ATMOSPHERE_NO_CUSTOM_AEROSOLS
            for (int layer = 0; layer < int(atm.aerosol_count.x); ++layer)
                particles[layer] += light*secondary_particles[s*4+layer];
#endif
#endif
        }
        if (atm.depth_params.z < 0.5) {
            multiple_uv.x = (0.5+clamp(mu*0.5+0.5, 0.0, 1.0)*float(MULTISCATTER_SIZE.x-1))/float(MULTISCATTER_SIZE.x);
            multiple += secondary.irradiance[s].rgb*textureLod(multiscatter, multiple_uv, 0.0).rgb;
        }
    }
    vec3 source = rayleigh*direct_rayleigh+(rayleigh+medium_mie)*multiple;
#ifdef ATMOSPHERE_SECONDARY_RING
    source += (rayleigh+medium_mie)*diffuse;
#endif
#ifndef ATMOSPHERE_RAYLEIGH_ONLY
#ifdef ATMOSPHERE_NO_CUSTOM_AEROSOLS
    source += medium_mie*direct_mie;
#else
    float height = max(radius, bottom_radius())-atm.reference_shape.x;
    source += atm.mie_scattering.rgb*exp(-max(height, 0.0)/atm.radii_heights.w)*direct_mie;
    for (int layer = 0; layer < int(atm.aerosol_count.x); ++layer)
        source += atm.aerosol_scattering[layer].rgb*particle_density(layer, height)*particles[layer];
#endif
#endif
    return source;
}
