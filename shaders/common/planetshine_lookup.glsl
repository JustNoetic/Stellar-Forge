#ifndef SF_PLANETSHINE_LOOKUP
#define SF_PLANETSHINE_LOOKUP
layout(set = 0, binding = 31) uniform sampler2DArray u_planetshine_map;

const vec2 PLANETSHINE_MAP_SIZE = vec2(128.0, 64.0);
const float PLANETSHINE_MAP_PI = 3.141592653589793;

vec3 planetshine_map_direction(vec2 uv) {
    float phi = (uv.x-0.5)*2.0*PLANETSHINE_MAP_PI, theta = uv.y*PLANETSHINE_MAP_PI;
    return vec3(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
}

vec3 planetshine_map_sample(vec3 direction, uint layer) {
    vec2 uv = vec2(atan(direction.z, direction.x)/(2.0*PLANETSHINE_MAP_PI)+0.5,
                   clamp(acos(clamp(direction.y, -1.0, 1.0))/PLANETSHINE_MAP_PI,
                         0.5/PLANETSHINE_MAP_SIZE.y, 1.0-0.5/PLANETSHINE_MAP_SIZE.y));
    return textureLod(u_planetshine_map, vec3(uv, float(layer)), 0.0).rgb;
}
#endif
