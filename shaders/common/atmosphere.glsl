const float ATM_PI = 3.141592653589793;
const ivec2 TRANSMITTANCE_SIZE = ivec2(256, 64);
const ivec2 MULTISCATTER_SIZE = ivec2(32);
const ivec2 SKY_SIZE = ivec2(192, 108);

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

void medium(vec3 p, out vec3 rayleigh, out vec3 mie, out vec3 extinction) {
    float h = max(length(p), bottom_radius()) - atm.reference_shape.x;
    float r = exp(clamp(-h / atm.radii_heights.z, -80.0, 40.0));
    float m = exp(-max(h, 0.0) / atm.radii_heights.w);
    rayleigh = atm.rayleigh.rgb * r;
    mie = atm.mie_scattering.rgb * m;
    extinction = rayleigh + atm.mie_extinction.rgb * m;
}

vec3 world_to_atmosphere(vec3 v) {
    return v+atm.world_pole.xyz*dot(v, atm.world_pole.xyz)*atm.world_pole.w;
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

vec3 sun_transmittance(sampler2D lut, vec3 p, vec3 sun) {
    vec2 ground = sphere_roots(p, sun, bottom_radius());
    if (ground.x > 0.0) return vec3(0.0);
    float r = length(p);
    return textureLod(lut, transmittance_uv(r, dot(p, sun)/r), 0.0).rgb;
}

vec2 multiscatter_uv(vec3 p, vec3 sun) {
    float r = length(p);
    vec2 uv = vec2(dot(p, sun)/r*0.5+0.5, (r-bottom_radius())/(top_radius()-bottom_radius()));
    return (0.5 + clamp(uv, 0.0, 1.0)*vec2(MULTISCATTER_SIZE-1))/vec2(MULTISCATTER_SIZE);
}

float rayleigh_phase(float mu) { return 3.0*(1.0+mu*mu)/(16.0*ATM_PI); }
float mie_phase(float mu) {
    float g = atm.mie_scattering.w;
    return 3.0*(1.0-g*g)*(1.0+mu*mu) /
        (8.0*ATM_PI*(2.0+g*g)*pow(max(1.0+g*g-2.0*g*mu, 1e-4), 1.5));
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
