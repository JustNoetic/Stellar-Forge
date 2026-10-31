#version 460

#include "common/ring_hosts.glsl"
#include "common/ring_optics.glsl"
#include "common/ringshine_integral.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 8) uniform sampler2DArray u_ring_gradients;
layout(std430, set = 0, binding = 10) readonly buffer RingshineRadialSamples {
    vec2 ringshine_nodes[];
};

layout(push_constant) uniform RingshineBakePush {
    uvec4 tile;
    vec4 host_params;
    vec4 solar;
    vec4 disk_offset;
    vec4 disk_weight;
} u_push;

float material_tau;
float layer_tau[8];
vec3 layer_color[8];
vec4 layer_props[8];
bool layer_textured[8];

vec3 ringshine_material_radiance(float mu_v, float sun, float mu, bool lit) {
    vec3 response = vec3(0.0);
    for (int i = 0; i < 8; ++i) {
        if (float(i) >= u_push.solar.z || layer_tau[i] <= 0.0) continue;
        vec3 radiance = ring_radiance_rgb(material_tau, mu_v, sun, mu,
                                          layer_props[i], layer_color[i], layer_textured[i], lit);
        response += (layer_tau[i] / material_tau) * radiance;
    }
    return response;
}

vec4 ringshine_disk_sample(vec3 center, int i, uint radial_index) {
    float sr = min(u_push.solar.y, 0.999);
    float edge_cosine = sqrt(max(1e-6, 1.0 - sr * sr));
    vec3 vertical = vec3(0.0, 1.0, 0.0) - center * center.y;
    vertical = dot(vertical, vertical) > 1e-12 ? normalize(vertical) : vec3(1.0, 0.0, 0.0);
    vec3 horizontal = cross(center, vertical);
    float u = (radial_index & 1u) == 0u ? 0.211324865405 : 0.788675134595;
    float angle = 6.28318530718 * fract((float(radial_index) + 0.5) * 0.569840290998);
    // Stable for tiny apparent disks, where 1 - edge_cosine rounds to zero.
    float one_minus_c = u * (sr * sr / (1.0 + edge_cosine));
    float c = 1.0 - one_minus_c;
    float s = sqrt(max(0.0, one_minus_c * (2.0 - one_minus_c)));
    float v = cos(angle), h = sin(angle);
    if ((i & 1) != 0) v = -v;
    if ((i & 2) != 0) h = -h;
    vec3 tangent = vertical * v + horizontal * h;
    // dOmega / (pi sin(alpha)^2): unit perpendicular incident irradiance.
    return vec4(center * c + tangent * s, 0.5 / (1.0 + edge_cosine));
}

void main() {
    out_color = vec4(0.0);
    float flattening = clamp(u_push.host_params.w, 0.0, 0.95);
    float inner_r = u_push.host_params.x, outer_r = u_push.host_params.y;
    if (u_push.host_params.z <= 0.0 || outer_r <= inner_r || u_push.solar.z <= 0.0) return;
    float sun = clamp(u_push.solar.x, -1.0, 1.0);
    vec3 center = vec3(-sqrt(max(0.0, 1.0 - sun * sun)), sun, 0.0);
    int segments = int(u_push.solar.z);
    float host_radius = ring_hosts[u_push.tile.x].tangent_radius.w;

    vec2 coord = clamp((v_uv * vec2(65.0) - 0.5) / vec2(64.0), 0.0, 1.0);
    float y = 2.0 * coord.y - 1.0;
    float phi = pow(coord.x, 1.5) * 3.14159265358979323846;
    float slat = sign(y) * pow(abs(y), 1.5);

    vec4 total = vec4(0.0);
    for (uint node = 0u; node < u_push.tile.w; ++node) {
        vec2 radial = ringshine_nodes[u_push.tile.z + node];
        float r = radial.x;
        material_tau = 0.0;
        for (int i = 0; i < segments; ++i) {
            RingSegmentGPU seg = ring_hosts[u_push.tile.x].segments[i];
            float seg_inner = seg.bounds_opacity_scatter.x / host_radius;
            float seg_outer = seg.bounds_opacity_scatter.y / host_radius;
            layer_tau[i] = 0.0;
            if (r < seg_inner || r > seg_outer) continue;
            float u = (r - seg_inner) / (seg_outer - seg_inner);
            float texels = radial.y / (seg_outer - seg_inner) * float(textureSize(u_ring_gradients, 0).x);
            vec4 material = textureLod(u_ring_gradients, vec3(u, 0.5, seg.phase_row.w), log2(max(texels, 1.0)));
            float alpha = clamp(material.a * seg.bounds_opacity_scatter.z, 0.0, RING_MAX_ALPHA);
            layer_tau[i] = ring_tau(alpha);
            layer_color[i] = max(material.rgb / max(material.a, 1e-20), vec3(0.0));
            layer_props[i] = vec4(seg.phase_row.xy, seg.bounds_opacity_scatter.w, seg.phase_row.z);
            layer_textured[i] = seg.color_flags.w > 0.5;
            material_tau += layer_tau[i];
        }
        if (material_tau <= 0.0) continue;

        vec4 response = vec4(0.0);
        int sources = (u_push.solar.y > 1e-6 && (u_push.solar.y >= 0.01 || abs(sun) < 2.0*u_push.solar.y)) ? 4 : 1;
        for (int q = 0; q < sources; ++q) {

            vec4 source = sources == 1 ? vec4(center, 1.0) : ringshine_disk_sample(center, q, node);
            float sample_phi = phi + atan(-source.z, -source.x);
            response += source.w * ringshine_band(r, slat, flattening, source.y, sample_phi);
        }
        total += response * radial.y;
    }
    out_color = total;
}
