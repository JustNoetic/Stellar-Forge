layout(push_constant) uniform SizePush {
    vec4 rect;
    vec4 shape;
    vec4 color;
    vec4 resources;
    vec4 display;
} cmp;

const vec3 COMPARISON_POLE = normalize(vec3(-0.48989795, 0.84852814, 0.2));
const vec3 COMPARISON_LIGHT = vec3(-0.77459667, 0.38729833, 0.5);

vec3 comparison_stretch(vec3 p) {
    return p+COMPARISON_POLE*dot(p, COMPARISON_POLE)*(1.0/cmp.shape.y-1.0);
}

vec2 comparison_roots(vec3 origin, vec3 ray, float radius) {
    vec3 o = comparison_stretch(origin), d = comparison_stretch(ray);
    float a = dot(d, d), b = dot(o, d);
    vec3 closest = o-d*(b/a);
    float delta = radius*radius-dot(closest, closest);
    if (delta < 0.0) return vec2(1e30, -1e30);
    float half_span = sqrt(delta/a);
    return vec2(-b/a-half_span, -b/a+half_span);
}
