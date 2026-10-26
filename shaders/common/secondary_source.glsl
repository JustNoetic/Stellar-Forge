vec2 secondary_extent[ATMOSPHERE_SECONDARY_LIGHTS];
vec3 secondary_rayleigh[ATMOSPHERE_SECONDARY_LIGHTS];
#ifdef ATMOSPHERE_SECONDARY_RING
vec4 secondary_ring_shape[ATMOSPHERE_SECONDARY_LIGHTS];
vec3 secondary_ring_pole[ATMOSPHERE_SECONDARY_LIGHTS];
vec3 secondary_ring_contrast[ATMOSPHERE_SECONDARY_LIGHTS];
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
            float outer = length(secondary.direction[s].xyz), inner = -secondary.direction[s].w;
            secondary_ring_shape[s] = vec4(inner*inner,outer*outer,sqrt(0.5*(inner*inner+outer*outer)),outer*outer-inner*inner);
            secondary_ring_pole[s] = secondary.direction[s].xyz/outer;
            uint packed = uint(secondary.irradiance[s].w);
            secondary_ring_contrast[s] = vec3(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u)/127.0-1.0;
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
    vec4 shape = secondary_ring_shape[s];
    vec3 pole = secondary_ring_pole[s];
    vec3 point = p/atm.reference_shape.x;
    float height = dot(point,pole), radius2 = dot(point,point);
    float planar = sqrt(max(radius2-height*height,0.0));
    float floor_radius = min(atm.reference_shape.z/atm.reference_shape.x,sqrt(radius2));
    float altitude = sqrt(max(radius2-floor_radius*floor_radius,0.0));
    float first = max(sqrt(shape.x),floor_radius*radius2/max(floor_radius*planar+abs(height)*altitude,1e-7));
    float inner2 = min(first*first,shape.y);
    float ring = sqrt(0.5*(inner2+shape.y));
    float threshold = floor_radius*floor_radius-altitude*sqrt(max(ring*ring-floor_radius*floor_radius,0.0));
    float contact = planar > 1e-7 ? clamp(threshold/(ring*planar),-1.0,1.0) : (threshold < 0.0 ? -1.0 : 1.0);
    float limit = acos(contact);
    float cosine = limit > 1e-4 ? sqrt(max(1.0-contact*contact,0.0))/limit : 1.0;
    float distance2 = max(ring*ring+radius2-2.0*ring*planar*cosine,1e-6);
    mu = clamp((ring*planar*cosine-radius2)*inversesqrt(radius2*distance2),-1.0,1.0);
    float physical_height = height*atm.reference_shape.y;
    float physical_distance2 = max(distance2-height*height+physical_height*physical_height,1e-6);
    float omega = min(2.0*ATM_PI,(shape.y-inner2)*limit*abs(physical_height)
        /(physical_distance2*sqrt(physical_distance2)));
    return secondary.irradiance[s].rgb*omega*max(vec3(1.0)+sign(height)*secondary_ring_contrast[s],vec3(0.0));
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
