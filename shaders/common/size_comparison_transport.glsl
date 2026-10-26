const ivec2 COMPARISON_TRANSPORT_SIZE = ivec2(512,257);

float comparison_projected_polar() {
    return sqrt(COMPARISON_POLE.z*COMPARISON_POLE.z+cmp.shape.y*cmp.shape.y*(1.0-COMPARISON_POLE.z*COMPARISON_POLE.z));
}

vec2 comparison_projected_point(float radius, float angle) {
    vec2 minor_axis = normalize(COMPARISON_POLE.xy);
    vec2 major_axis = vec2(minor_axis.y,-minor_axis.x);
    return radius*(major_axis*cos(angle)+minor_axis*sin(angle)*comparison_projected_polar());
}

vec2 comparison_transport_point(ivec2 pixel) {
    float u = (float(pixel.x)+0.5)/float(COMPARISON_TRANSPORT_SIZE.x);
    float outer = max(top_radius()/cmp.display.x-1.0,1e-6);
    float radius = u < 0.5 ? 1.0-pow(1.0-2.0*u,2.0) : 1.0+outer*pow(2.0*u-1.0,2.0);
    float angle = -ATM_PI+2.0*ATM_PI*float(pixel.y)/float(COMPARISON_TRANSPORT_SIZE.y-1);
    return comparison_projected_point(radius*cmp.display.x,angle);
}

vec2 comparison_transport_uv(vec2 xy) {
    vec2 minor_axis = normalize(COMPARISON_POLE.xy);
    vec2 q = vec2(dot(xy,vec2(minor_axis.y,-minor_axis.x)),dot(xy,minor_axis)/comparison_projected_polar())/cmp.display.x;
    float radius = length(q);
    float outer = max(top_radius()/cmp.display.x-1.0,1e-6);
    float u = radius <= 1.0 ? 0.5*(1.0-sqrt(max(1.0-radius,0.0))) : 0.5+0.5*sqrt(max((radius-1.0)/outer,0.0));
    float half_texel = 0.5/float(COMPARISON_TRANSPORT_SIZE.x);
    u = radius <= 1.0 ? min(u,0.5-half_texel) : max(u,0.5+half_texel);
    float v = (atan(q.y,q.x)+ATM_PI)/(2.0*ATM_PI);
    return vec2(clamp(u,half_texel,1.0-half_texel),(0.5+v*float(COMPARISON_TRANSPORT_SIZE.y-1))/float(COMPARISON_TRANSPORT_SIZE.y));
}
