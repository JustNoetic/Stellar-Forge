#version 460 core
in float v_radial;
in vec3 v_ring_position;
uniform sampler2DArray u_profile;
uniform int u_layer;
uniform vec3 u_pole;
uniform int u_atmo_pass; // 0: no atmosphere, 1: behind shell, 2: foreground
uniform float u_atmo_radius;
uniform float u_atmo_oblateness;
out vec4 out_color;
void main() {
    float footprint = max(fwidth(v_radial),1e-6);
    float coverage = clamp(v_radial/footprint+0.5,0.0,1.0)*
                     clamp((1.0-v_radial)/footprint+0.5,0.0,1.0);
    vec4 profile = texture(u_profile,vec3(v_radial,0.5,float(u_layer)))*coverage;
    if (u_atmo_pass != 0) {
        // Split the transparent ring at the front of the oblate atmosphere.
        // Foreground ring radiance must not be attenuated by air behind it.
        float stretch = 1.0/(1.0-u_atmo_oblateness)-1.0;
        vec3 q = vec3(v_ring_position.xy,0.0);
        q += u_pole*dot(q,u_pole)*stretch;
        vec3 d = vec3(0,0,1) + u_pole*u_pole.z*stretch;
        float a=dot(d,d), b=dot(q,d);
        float disc=b*b-a*(dot(q,q)-u_atmo_radius*u_atmo_radius);
        bool front = disc < 0.0 || v_ring_position.z >= (-b+sqrt(max(disc,0.0)))/a;
        if ((u_atmo_pass == 1 && front) || (u_atmo_pass == 2 && !front)) discard;
    }
    float light = abs(dot(u_pole,vec3(-0.77459667,0.38729833,0.5)));
    out_color = vec4(profile.rgb * light,profile.a);
}
