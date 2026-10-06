// Shared with native/terrain/src/surface.rs. No patch-LOD-dependent amplitude.
#ifndef PROCEDURAL_TERRAIN_GLSL
#define PROCEDURAL_TERRAIN_GLSL
uint terrain_noise_hash(uint v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15;
    v *= 0x846ca68bu; return v ^ (v >> 16);
}
float terrain_noise(vec3 p) {
    ivec3 base = ivec3(floor(p)); vec3 f = fract(p);
    f = f*f*f*(f*(f*6.0-15.0)+10.0);
    float result = 0.0;
    for (int z=0;z<2;++z) for (int y=0;y<2;++y) for (int x=0;x<2;++x) {
        ivec3 i=base+ivec3(x,y,z);
        uint h=terrain_noise_hash(uint(i.x) ^ (uint(i.y)*0x9e3779b9u) ^ (uint(i.z)*0x85ebca6bu));
        float n=float(h & 0x00ffffffu)*(2.0/16777215.0)-1.0;
        vec3 w=mix(1.0-f,f,vec3(x,y,z)); result += n*w.x*w.y*w.z;
    }
    return result;
}
float terrain_micro_elevation(vec3 n,float e,float lo,float span,bool water,float level) {
    float h=lo+e*span;
    float land=water ? smoothstep(level,level+0.04,h) : 1.0;
    if (land<=0.001) return 0.0;
    float mountain=water ? smoothstep(level+0.8,level+2.8,h) : smoothstep(0.25,0.65,e);
    vec3 p=n*450.0; float rolling=0.0,ridges=0.0,amp=0.5;
    for (int i=0;i<4;++i) {
        float v=terrain_noise(p); rolling+=v*amp*0.45; ridges+=(1.0-abs(v))*(1.0-abs(v))*amp;
        p=vec3(-0.8*p.y-0.6*p.z,0.8*p.x+0.36*p.y-0.48*p.z,0.6*p.x-0.48*p.y+0.64*p.z)*2.02;
        amp*=0.49;
    }
    return mix(rolling,ridges-0.45,mountain)*min(0.35,span*0.045)*land;
}
#endif
