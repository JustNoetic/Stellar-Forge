layout(set = 0, binding = 12) uniform sampler2DArray surface_atmosphere_lut;
layout(std430, set = 0, binding = 13) readonly buffer SurfaceAtmosphereRecords { vec4 surface_atmospheres[]; };

void surface_atmosphere(uint body, vec3 local, vec3 base_chord, vec3 sun, vec3 normal, out vec3 transmission, out vec3 diffuse) {
    transmission = vec3(1.0);
    diffuse = vec3(0.0);
    vec4 record = surface_atmospheres[body];
    if (record.w < 0.5) return;
    BodyInstanceGPU b = bodies[body];
    vec3 pole = body_pole(b);
    float flattening = 1.0/max(1.0-b.pole_oblateness.w, 0.05)-1.0;
    vec3 p = local+pole*dot(local, pole)*flattening;
    float r = length(p)*149597870.7;
    float sag = (1.0-length(base_chord))*record.z;
    r += max(sag, 0.0);
    vec3 up = normalize(p);
    vec3 light = normalize(sun+pole*dot(sun, pole)*flattening);
    float mu = clamp(dot(up, light), -1.0, 1.0);
    float x = sign(mu)*sqrt(abs(mu))*0.5+0.5;
    float y = sqrt(clamp((r-record.x)/(record.y-record.x), 0.0, 1.0));
    vec2 uv = (vec2(0.5)+vec2(x, y)*vec2(127.0, 31.0))/vec2(128.0, 32.0);
    float layer = (record.w-1.0)*2.0;
    transmission = textureLod(surface_atmosphere_lut, vec3(uv, layer), 0.0).rgb;
    diffuse = textureLod(surface_atmosphere_lut, vec3(uv, layer+1.0), 0.0).rgb;
    diffuse *= 0.5+0.5*max(dot(normal, normalize(local)), 0.0);
}
