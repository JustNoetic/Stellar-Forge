#version 460 core
in vec2 f_uv;
out vec4 out_color;
uniform sampler2D u_ring_gradients;
uniform vec4 u_host_params;
uniform float u_sun_elevation;
uniform float u_sun_radius;
uniform int u_ringshine_band_count;
uniform int u_segment_count;
uniform vec4 u_segment_geometry[16]; // inner/outer host radii, opacity, atlas row
uniform vec4 u_segment_props[16]; // gf, gb, balance, opposite-face multiplier
uniform float u_segment_textured[16];
#include "common/ring_optics.glsl"
#include "common/ring_source.glsl"
float material_tau;
float layer_tau[16];
float layer_alpha[16];
vec3 layer_color[16];

vec3 ringshine_material_radiance(float mu_v,float sun,float mu,bool lit) {
    vec3 response=vec3(0.0);
    vec2 transfer=ring_transfer(material_tau,mu_v,abs(sun),lit);
    float opposition=lit ? ring_opposition(-mu,material_tau) : 1.0;
    for(int i=0;i<u_segment_count;++i) {
        if(layer_tau[i]<=0.0) continue;
        vec2 phase=ring_phase(mu,layer_alpha[i],u_segment_props[i],u_segment_textured[i]>0.5);
        float single=transfer.x*phase.x*(lit ? opposition : max(0.0,u_segment_props[i].a));
        response+=layer_color[i]*(layer_tau[i]/material_tau)
            *(single+transfer.y*phase.y);
    }
    return response;
}
#include "common/ringshine_integral.glsl"
const vec4 QUAD_T=vec4(0.0694318442,0.3300094782,0.6699905218,0.9305681558);
const vec4 QUAD_W=vec4(0.1739274226,0.3260725774,0.3260725774,0.1739274226);

void main() {
    float inner_r=u_host_params.x,outer_r=u_host_params.y;
    if(u_host_params.z<=0.0 || outer_r<=inner_r || u_segment_count<=0) {
        out_color=vec4(0.0); return;
    }
    vec2 coord=(f_uv*vec2(128.0,65.0)-0.5)/vec2(127.0,64.0);
    float x=2.0*coord.x-1.0,y=2.0*coord.y-1.0;
    float phi=sign(x)*pow(abs(x),1.5)*RS_PI;
    float slat=sign(y)*pow(abs(y),1.5);
    float flattening=clamp(u_host_params.w,0.0,0.95);
    float first_r=max(inner_r,1.000001);
    if(outer_r<=first_r) { out_color=vec4(0.0); return; }
    int bands=clamp(u_ringshine_band_count,4,1024);
    float first_gap=first_r-1.0;
    float log_span=log((outer_r-1.0)/first_gap);
    float sun=clamp(u_sun_elevation,-1.0,1.0);
    vec3 center=vec3(-sqrt(max(0.0,1.0-sun*sun)),sun,0.0);
    int sources=ring_resolve_disk(sun,u_sun_radius) ? RING_DISK_SAMPLES : 1;
    vec3 total=vec3(0.0);
    for(int m=0;m<bands;++m) {
        for(int s=0;s<4;++s) {
            float t=(float(m)+QUAD_T[s])/float(bands);
            float gap=first_gap*exp(t*log_span),r=1.0+gap;
            material_tau=0.0;
            for(int i=0;i<u_segment_count;++i) {
                vec4 geometry=u_segment_geometry[i];
                layer_tau[i]=0.0;
                if(r<geometry.x || r>geometry.y) continue;
                float u=(r-geometry.x)/(geometry.y-geometry.x);
                float row=(geometry.w+0.5)/float(textureSize(u_ring_gradients,0).y);
                vec4 material=textureLod(u_ring_gradients,vec2(u,row),0.0);
                layer_alpha[i]=clamp(material.a*geometry.z,0.0,RING_MAX_ALPHA);
                layer_tau[i]=ring_tau(layer_alpha[i]);
                layer_color[i]=pow(max(material.rgb,vec3(0.0)),vec3(2.2));
                material_tau+=layer_tau[i];
            }
            if(material_tau<=0.0) continue;
            vec3 response=vec3(0.0);
            for(int q=0;q<sources;++q) {
                vec4 source=sources==1 ? vec4(center,1.0)
                    : ring_disk_sample(center,vec3(0,1,0),u_sun_radius,q);
                float sample_phi=phi+atan(-source.z,-source.x);
                response+=source.w*ringshine_band(r,slat,flattening,source.y,sample_phi);
            }
            float dr_weight=gap*log_span*QUAD_W[s]/float(bands);
            total+=response*dr_weight;
        }
    }
    out_color=vec4(total*RS_PI,1.0);
}
