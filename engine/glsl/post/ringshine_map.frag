#version 460 core
in vec2 f_uv;
out vec4 out_color;
uniform sampler2D u_ring_gradients;
uniform sampler2D u_ring_props;
uniform int u_ring_index;
// Inner/outer radius in host equatorial radii, enabled opacity, flattening.
uniform vec4 u_host_params;
uniform float u_sun_elevation;
uniform int u_ringshine_band_count;
#include "common/ringshine_integral.glsl"
const vec4 QUAD_T=vec4(0.0694318442,0.3300094782,0.6699905218,0.9305681558);
const vec4 QUAD_W=vec4(0.1739274226,0.3260725774,0.3260725774,0.1739274226);

void main() {
    float inner_r=u_host_params.x,outer_r=u_host_params.y;
    if(u_host_params.z<=0.0 || outer_r<=inner_r || abs(u_sun_elevation)<1e-7) {
        out_color=vec4(0.0); return;
    }
    // Texel centres represent the closed angular domain, including the poles.
    vec2 coord=(f_uv*vec2(128.0,65.0)-0.5)/vec2(127.0,64.0);
    float x=2.0*coord.x-1.0,y=2.0*coord.y-1.0;
    float phi=sign(x)*pow(abs(x),1.5)*RS_PI;
    float slat=sign(y)*pow(abs(y),1.5);
    float flattening=clamp(u_host_params.w,0.0,0.95);
    // Ignore portions inside the host without moving the material domain.
    float first_r=max(inner_r,1.000001);
    if(outer_r<=first_r) { out_color=vec4(0.0); return; }
    int bands=clamp(u_ringshine_band_count,4,1024);
    // Resolve near-contact radial peaks as well as angular ones: logarithmic
    // distance from the host surface tends to log-radius spacing far away.
    float first_gap=first_r-1.0;
    float log_span=log((outer_r-1.0)/first_gap);
    float grad_y=(float(u_ring_index)+0.5)/float(textureSize(u_ring_gradients,0).y);
    float props_y=(float(2*u_ring_index)+0.5)/float(textureSize(u_ring_props,0).y);
    float extra_y=(float(2*u_ring_index+1)+0.5)/float(textureSize(u_ring_props,0).y);
    vec3 total=vec3(0.0);
    for(int m=0;m<bands;++m) {
        for(int s=0;s<4;++s) {
            float t=(float(m)+QUAD_T[s])/float(bands);
            float gap=first_gap*exp(t*log_span),r=1.0+gap;
            float u=(r-inner_r)/(outer_r-inner_r);
            vec4 material=textureLod(u_ring_gradients,vec2(u,grad_y),0.0);
            float alpha=clamp(material.a*u_host_params.z,0.0,0.9999999);
            if(alpha<=0.0) continue;
            vec4 props=textureLod(u_ring_props,vec2(u,props_y),0.0);
            bool textured=textureLod(u_ring_props,vec2(u,extra_y),0.0).r>0.5;
            float response=ringshine_band(r,slat,flattening,u_sun_elevation,
                                          phi,alpha,props,textured);
            // K includes r d(azimuth); dr = gap d(log gap). Evaluate nonlinear
            // transfer at each radial node, before integrating the response.
            float dr_weight=gap*log_span*QUAD_W[s]/float(bands);
            total+=pow(max(material.rgb,vec3(0.0)),vec3(2.2))*response*dr_weight;
        }
    }
    out_color=vec4(total*RS_PI,1.0);
}
