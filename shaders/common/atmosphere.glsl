const float ATM_PI = 3.141592653589793;
const ivec2 TRANSMITTANCE_SIZE = ivec2(256, 64);
const ivec2 MULTISCATTER_SIZE = ivec2(32);
const ivec2 SKY_SIZE = ivec2(192, 108);
const ivec3 AERIAL_SIZE = ivec3(192, 108, 64);

layout(std140, set = 0, binding = 0) uniform AtmosphereParameters {
    vec4 radii_heights;
    vec4 reference_shape;
    vec4 world_pole;
    vec4 rayleigh;
    vec4 mie_scattering;
    vec4 mie_extinction;
    vec4 ground_albedo;
    vec4 camera;
    vec4 axis_x;
    vec4 axis_y;
    vec4 axis_z;
    vec4 sun_direction[16];
    vec4 sun_irradiance[16];
    mat4 inverse_view_projection;
    vec4 depth_params;
    vec4 mixed_absorption;
    vec4 ozone_absorption;
    vec4 ozone_params;
    vec4 profile_domain;
    vec4 log_density[256];
    vec4 aerosol_count;
    vec4 aerosol_shape[4];
    vec4 aerosol_scattering[4];
    vec4 aerosol_extinction[4];
    vec4 aerosol_phase[512];
} atm;

float bottom_radius() { return atm.radii_heights.x; }
float top_radius() { return atm.radii_heights.y; }

vec2 sphere_roots(vec3 p, vec3 d, float radius) {
    float b = dot(p, d);
    vec3 perpendicular = cross(p, d);
    float discriminant = radius * radius - dot(perpendicular, perpendicular);
    if (discriminant < 0.0) return vec2(-1.0);
    float root = sqrt(max(discriminant, 0.0));
    return vec2(-b - root, -b + root);
}

vec2 atmosphere_segment(vec3 p, vec3 d) {
    vec2 outer = sphere_roots(p, d, top_radius());
    if (outer.y <= 0.0) return vec2(0.0);
    float start = max(outer.x, 0.0);
    float end = outer.y;
    vec2 ground = sphere_roots(p, d, bottom_radius());
    if (ground.x >= 0.0) end = min(end, ground.x);
    return vec2(start, max(start, end));
}

float particle_density(int layer, float height) {
    vec4 shape = atm.aerosol_shape[layer];
    return shape.z > 0.0 ? exp(-0.5*pow((height-shape.y)/shape.z, 2.0)) : exp(-max(height, 0.0)/shape.x);
}

void medium(vec3 p, out vec3 rayleigh, out vec3 mie, out vec3 extinction) {
    float h = max(length(p), bottom_radius()) - atm.reference_shape.x;
    float r;
    if (atm.profile_domain.w > 0.5) {
        float index = clamp((asinh(h/atm.profile_domain.z)-atm.profile_domain.x)*atm.profile_domain.y, 0.0, 1.0)*255.0;
        int lower = min(int(index), 254);
        r = exp(clamp(mix(atm.log_density[lower].x, atm.log_density[lower+1].x, index-float(lower)), -80.0, 40.0));
    } else r = exp(clamp(-h / atm.radii_heights.z, -80.0, 40.0));
    float m = exp(-max(h, 0.0) / atm.radii_heights.w);
    rayleigh = atm.rayleigh.rgb * r;
    mie = atm.mie_scattering.rgb * m;
    extinction = rayleigh + atm.mie_extinction.rgb * m;
    for (int i = 0; i < int(atm.aerosol_count.x); ++i) {
        float density = particle_density(i, h);
        mie += atm.aerosol_scattering[i].rgb*density;
        extinction += atm.aerosol_extinction[i].rgb*density;
    }
    if (atm.profile_domain.w > 0.5) {
        float ozone = exp(-0.5*pow((h-atm.ozone_params.x)/max(atm.ozone_params.y, 0.001), 2.0));
        extinction += atm.mixed_absorption.rgb*r+atm.ozone_absorption.rgb*ozone;
    }
}

vec3 world_to_atmosphere(vec3 v) {
    return v+atm.world_pole.xyz*dot(v, atm.world_pole.xyz)*atm.world_pole.w;
}

vec3 atmosphere_direction(vec3 world) {
    vec3 d = normalize(world_to_atmosphere(world));
    return vec3(dot(d, atm.axis_x.xyz), dot(d, atm.axis_y.xyz), dot(d, atm.axis_z.xyz));
}

vec3 view_direction(vec2 ndc) {
    vec4 world = atm.inverse_view_projection*vec4(ndc, 1.0, 1.0);
    return atmosphere_direction(world.xyz/world.w);
}

float aerial_distance(float fraction, float range) {
    return range*(atm.camera.z < top_radius() ? fraction*fraction : fraction);
}

vec3 segment_integral(vec3 extinction, float dt) {
    vec3 tau = extinction * dt;
    vec3 exact = (1.0 - exp(-tau)) / max(extinction, vec3(1e-20));
    return mix(exact, vec3(dt), lessThan(tau, vec3(1e-4)));
}

vec2 transmittance_uv(float r, float mu) {
    r = clamp(r, bottom_radius(), top_radius());
    float H = sqrt((top_radius() - bottom_radius()) * (top_radius() + bottom_radius()));
    float rho = sqrt(max((r - bottom_radius()) * (r + bottom_radius()), 0.0));
    float d = max(-r * mu + sqrt(max(r*r*(mu*mu-1.0) + top_radius()*top_radius(), 0.0)), 0.0);
    float x = (d - (top_radius()-r)) / max(rho + H - (top_radius()-r), 1e-6);
    vec2 unit_uv = clamp(vec2(x, rho / H), 0.0, 1.0);
    return (0.5 + unit_uv * vec2(TRANSMITTANCE_SIZE - 1)) / vec2(TRANSMITTANCE_SIZE);
}

void transmittance_ray(vec2 uv, out vec3 p, out vec3 d) {
    float H = sqrt((top_radius()-bottom_radius())*(top_radius()+bottom_radius()));
    float rho = H * uv.y;
    float r = sqrt(rho*rho + bottom_radius()*bottom_radius());
    float distance = mix(top_radius()-r, rho+H, uv.x);
    float mu = distance < 1e-5 ? 1.0 : (H*H-rho*rho-distance*distance)/(2.0*r*distance);
    mu = clamp(mu, -1.0, 1.0);
    p = vec3(0.0, 0.0, r);
    d = vec3(sqrt(max(1.0-mu*mu, 0.0)), 0.0, mu);
}

float sun_visibility(vec3 p, vec3 sun, float angular_radius) {
    float r = length(p);
    float shadow_radius = min(atm.reference_shape.z > 0.0 ? atm.reference_shape.z : atm.reference_shape.x, r);
    float sin_horiz_sq = max((r-shadow_radius)*(r+shadow_radius)/(r*r), 0.0);
    float mu_horiz = -sqrt(sin_horiz_sq);
    float mu = dot(p, sun) / r;
    float width = max(shadow_radius/r*angular_radius, 1e-7);
    return smoothstep(-width, width, mu-mu_horiz);
}

vec3 sun_transmittance(sampler2D lut, vec3 p, vec3 sun, float angular_radius) {
    float vis = sun_visibility(p, sun, angular_radius);
    if (vis <= 0.0) return vec3(0.0);
    float r = length(p);
    return vis * exp(-textureLod(lut, transmittance_uv(r, dot(p, sun)/r), 0.0).rgb);
}

vec3 sun_transmittance(sampler2D lut, vec3 p, vec3 sun) {
    return sun_transmittance(lut, p, sun, 0.01);
}

vec2 multiscatter_uv(vec3 p, vec3 sun) {
    float r = length(p);
    vec2 uv = vec2(dot(p, sun)/r*0.5+0.5, (r-bottom_radius())/(top_radius()-bottom_radius()));
    if (atm.profile_domain.w > 0.5) {
        float start = asinh((bottom_radius()-atm.reference_shape.x)/atm.profile_domain.z);
        float end = asinh((top_radius()-atm.reference_shape.x)/atm.profile_domain.z);
        uv.y = (asinh((r-atm.reference_shape.x)/atm.profile_domain.z)-start)/(end-start);
    }
    return (0.5 + clamp(uv, 0.0, 1.0)*vec2(MULTISCATTER_SIZE-1))/vec2(MULTISCATTER_SIZE);
}

float multiscatter_radius(float fraction) {
    if (atm.profile_domain.w <= 0.5) return mix(bottom_radius()+0.001, top_radius()-0.001, fraction);
    float start = asinh((bottom_radius()-atm.reference_shape.x)/atm.profile_domain.z);
    float end = asinh((top_radius()-atm.reference_shape.x)/atm.profile_domain.z);
    return clamp(atm.reference_shape.x+atm.profile_domain.z*sinh(mix(start, end, fraction)), bottom_radius()+0.001, top_radius()-0.001);
}

int transport_subdivisions(vec3 p, vec3 d, float start, float distance) {
    if (atm.aerosol_count.x < 0.5) return 1;
    vec3 a = p+d*start, b = a+d*distance;
    float closest = clamp(-dot(a, d), 0.0, distance);
    float minimum = length(a+d*closest)-atm.reference_shape.x;
    float maximum = max(length(a), length(b))-atm.reference_shape.x;
    float step = distance;
    for (int i = 0; i < int(atm.aerosol_count.x); ++i) {
        vec4 shape = atm.aerosol_shape[i];
        if (shape.z > 0.0 && maximum > shape.y-4.0*shape.z && minimum < shape.y+4.0*shape.z)
            step = min(step, 0.5*shape.z);
    }
    return clamp(int(ceil((maximum-minimum)/max(step, 0.001))), 1, 32);
}

float rayleigh_phase(float mu) { return 3.0*(1.0+mu*mu)/(16.0*ATM_PI); }
float mie_phase(float mu) {
    float g = atm.mie_scattering.w;
    return 3.0*(1.0-g*g)*(1.0+mu*mu) /
        (8.0*ATM_PI*(2.0+g*g)*pow(max(1.0+g*g-2.0*g*mu, 1e-4), 1.5));
}

vec3 particle_phase(int layer, float mu) {
    float t = sqrt(acos(clamp(mu, -1.0, 1.0))/ATM_PI)*127.0;
    int i = min(int(t), 126);
    float a = float(i)/127.0, b = float(i+1)/127.0;
    float mu_a = cos(ATM_PI*a*a), mu_b = cos(ATM_PI*b*b);
    float blend = clamp((mu_a-mu)/max(mu_a-mu_b, 1e-8), 0.0, 1.0);
    return mix(atm.aerosol_phase[layer*128+i].rgb, atm.aerosol_phase[layer*128+i+1].rgb, blend);
}

vec3 aerosol_source(vec3 p, float mu) {
    float h = max(length(p), bottom_radius())-atm.reference_shape.x;
    vec3 source = atm.mie_scattering.rgb*exp(-max(h, 0.0)/atm.radii_heights.w)*mie_phase(mu);
    for (int i = 0; i < int(atm.aerosol_count.x); ++i)
        source += atm.aerosol_scattering[i].rgb*particle_density(i, h)*particle_phase(i, mu);
    return source;
}

float sky_angle(float t, bool lower) {
    float r = atm.camera.z;
    if (r >= top_radius()) {
        float limb = asin(clamp(bottom_radius()/r, 0.0, 1.0));
        float edge = asin(clamp(top_radius()/r, 0.0, 1.0));
        return ATM_PI - (lower ? mix(limb, edge, t*t) : limb*(1.0-(1.0-t)*(1.0-t)));
    }
    float horizon = acos(-sqrt(max(1.0-bottom_radius()*bottom_radius()/(r*r), 0.0)));
    return lower ? horizon+(ATM_PI-horizon)*t*t : horizon*(1.0-(1.0-t)*(1.0-t));
}

vec3 sky_direction(ivec2 pixel) {
    int half_height = SKY_SIZE.y/2;
    bool lower = pixel.y >= half_height;
    float t = float(lower ? pixel.y-half_height : pixel.y)/float(half_height-1);
    float theta = sky_angle(t, lower);
    float phi = float(pixel.x)/float(SKY_SIZE.x-1)*2.0*ATM_PI-ATM_PI;
    return vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
}

vec2 sky_uv(vec3 direction) {
    float theta = acos(clamp(direction.z, -1.0, 1.0));
    float r = atm.camera.z;
    bool lower;
    float t;
    if (r >= top_radius()) {
        float angle = atan(length(direction.xy), -direction.z);
        float limb = asin(clamp(bottom_radius()/r, 0.0, 1.0));
        float edge = asin(clamp(top_radius()/r, 0.0, 1.0));
        lower = angle > limb;
        t = lower ? sqrt(clamp((angle-limb)/max(edge-limb, 1e-8), 0.0, 1.0))
                  : 1.0-sqrt(clamp(1.0-angle/max(limb, 1e-8), 0.0, 1.0));
    } else {
        float horizon = acos(-sqrt(max(1.0-bottom_radius()*bottom_radius()/(r*r), 0.0)));
        lower = theta > horizon;
        t = lower ? sqrt(clamp((theta-horizon)/(ATM_PI-horizon), 0.0, 1.0))
                  : 1.0-sqrt(clamp(1.0-theta/horizon, 0.0, 1.0));
    }
    float phi = atan(direction.y, direction.x);
    float x = (phi+ATM_PI)/(2.0*ATM_PI);
    float y = (lower ? float(SKY_SIZE.y/2)+0.5 : 0.5) + t*float(SKY_SIZE.y/2-1);
    return vec2((0.5+x*float(SKY_SIZE.x-1))/float(SKY_SIZE.x), y/float(SKY_SIZE.y));
}

vec3 segment_transmittance(sampler2D lut, float rx, float mux, float rp, float mup, bool hits_ground) {
    if (hits_ground) {
        vec3 tp = textureLod(lut, transmittance_uv(rp, -mup), 0.0).rgb;
        vec3 tx = textureLod(lut, transmittance_uv(rx, -mux), 0.0).rgb;
        return exp(-max(tp - tx, vec3(0.0)));
    } else {
        vec3 tx = textureLod(lut, transmittance_uv(rx, mux), 0.0).rgb;
        vec3 tp = textureLod(lut, transmittance_uv(rp, mup), 0.0).rgb;
        return exp(-max(tx - tp, vec3(0.0)));
    }
}

vec2 angular_segment(ivec2 pixel, vec3 d) {
    vec2 outer = sphere_roots(atm.camera.xyz, d, top_radius());
    if (outer.y <= 0.0) return vec2(0.0);
    float start = max(outer.x, 0.0), end = outer.y;
    bool ground = atm.camera.z >= top_radius() ? pixel.y < SKY_SIZE.y/2 : pixel.y >= SKY_SIZE.y/2;
    if (ground) {
        vec2 roots = sphere_roots(atm.camera.xyz, d, bottom_radius());
        end = min(end, roots.x >= 0.0 ? roots.x : -dot(atm.camera.xyz, d));
    }
    return vec2(start, max(start, end));
}

void integrate_atmosphere(sampler2D transmittance_lut, sampler2D multiscatter_lut,
                          vec3 d, float start, float span, inout vec3 L, inout vec3 tau) {
    if (span <= 0.0) return;
    for (int step = 0; step < 2; ++step) {
        float offset = start+float(step)*span*0.5;
        int subdivisions = transport_subdivisions(atm.camera.xyz, d, offset, span*0.5);
        float dt = span*0.5/float(subdivisions);
        for (int j = 0; j < subdivisions; ++j) {
            vec3 q = atm.camera.xyz+d*(offset+(float(j)+0.5)*dt);
            vec3 r, m, e;
            medium(q, r, m, e);
            vec3 source = vec3(0.0);
            for (uint s = 0u; s < uint(atm.camera.w); ++s) {
                vec3 sun = atm.sun_direction[s].xyz;
                float mu = dot(d, sun);
                vec3 direct = sun_transmittance(transmittance_lut, q, sun, atm.sun_direction[s].w)
                            *(r*rayleigh_phase(mu)+aerosol_source(q, mu));
                vec3 multiple = textureLod(multiscatter_lut, multiscatter_uv(q, sun), 0.0).rgb*(r+m);
                source += atm.sun_irradiance[s].rgb*(direct+multiple);
            }
            L += exp(-tau)*source*segment_integral(e, dt);
            tau += e*dt;
        }
    }
}
