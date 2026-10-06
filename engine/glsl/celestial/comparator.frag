#version 460 core
in vec3 v_local;
in vec3 v_normal;
uniform vec3 u_color;
uniform vec3 u_pole;
uniform float u_spin;
uniform bool u_is_star;
uniform bool u_black_hole;
uniform bool u_textured;
uniform bool u_has_clouds;
uniform bool u_has_normal;
uniform sampler2D u_diffuse;
uniform sampler2D u_clouds;
uniform sampler2D u_normal_map;
out vec4 out_color;
// cos(phase angle)=0.5 => (1+cos(phase))/2 = 75% illuminated disk.
const vec3 LIGHT = vec3(-0.77459667, 0.38729833, 0.5);
void main() {
    vec3 N = normalize(v_normal);
    vec3 albedo = pow(u_color, vec3(2.2));
    if (u_black_hole) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    if (u_is_star) {
        float mu = max(N.z, 0.0);
        float limb = 1.0 - 0.45 * (1.0-mu) - 0.15 * (1.0-mu) * (1.0-mu);
        out_color = vec4(albedo * limb, 1.0);
        return;
    }
    if (u_textured) {
        vec3 ref = abs(u_pole.y) > 0.999 ? vec3(1,0,0) : vec3(0,1,0);
        vec3 tangent = normalize(cross(u_pole, ref));
        vec3 bitangent = normalize(cross(u_pole, tangent));
        vec3 p = normalize(v_local);
        vec3 q = vec3(dot(p, tangent), dot(p, u_pole), dot(p, bitangent));
        float s = sin(-u_spin), c = cos(-u_spin);
        q.xz = mat2(c,s,-s,c) * q.xz;
        vec2 uv = vec2(0.5 + atan(q.z,q.x)/6.28318530718,
                       0.5 - asin(clamp(q.y,-1.0,1.0))/3.14159265359);
        vec2 dx = dFdx(uv), dy = dFdy(uv);
        dx.x -= round(dx.x); dy.x -= round(dy.x);
        albedo = pow(textureGrad(u_diffuse, uv, dx, dy).rgb, vec3(2.2));
        if (u_has_normal) {
            vec3 T = normalize(vec3(-q.z, 0, q.x));
            vec3 B = normalize(cross(q,T));
            T.xz = mat2(c,-s,s,c) * T.xz;
            B.xz = mat2(c,-s,s,c) * B.xz;
            mat3 basis = mat3(tangent, u_pole, bitangent);
            N = normalize(mat3(basis*T, basis*B, N) *
                          (textureGrad(u_normal_map,uv,dx,dy).xyz*2.0-1.0));
        }
        if (u_has_clouds) {
            vec4 cloud = textureGrad(u_clouds, uv, dx, dy);
            bool gray = abs(cloud.r-cloud.g)<0.01 && abs(cloud.g-cloud.b)<0.01;
            albedo = mix(albedo, gray ? vec3(0.95) : pow(cloud.rgb,vec3(2.2)),cloud.a);
        }
    }
    out_color = vec4(albedo * max(dot(N, LIGHT), 0.0), 1.0);
}
