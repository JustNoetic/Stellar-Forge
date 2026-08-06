#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16
#define PI 3.14159265358979323846

in vec3 f_world_pos;
in vec3 f_normal;
in vec3 f_local_pos;
in float f_clip_z;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
};

uniform sampler2DArray u_planet_textures;
uniform float u_cloud_tex_idx;
uniform vec3 u_cloud_color;
uniform float u_cloud_opacity;
uniform float u_cloud_coverage;
uniform vec3 u_camera_pos;
uniform float u_exposure;
uniform bool u_hdr_enabled;

out vec4 out_color;

void main() {
    if (f_clip_z <= 0.0) discard;
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);

    vec3 p = normalize(f_local_pos);
    float u = 0.5 + atan(p.z, p.x) / (2.0 * PI);
    float v = 0.5 - asin(clamp(p.y, -1.0, 1.0)) / PI;
    vec2 uv = vec2(u, v);

    vec3 cloud_albedo = u_cloud_color;
    float cloud_alpha = u_cloud_opacity;

    if (u_cloud_tex_idx >= 0.0) {
        vec4 tex_val = texture(u_planet_textures, vec3(uv, u_cloud_tex_idx));
        cloud_albedo = pow(tex_val.rgb, vec3(2.2)) * u_cloud_color;
        float sample_a = (tex_val.a > 0.01 && tex_val.a < 0.99) ? tex_val.a : max(tex_val.r, max(tex_val.g, tex_val.b));
        cloud_alpha = sample_a * u_cloud_opacity;
    }

    cloud_alpha = clamp(cloud_alpha * u_cloud_coverage, 0.0, 1.0);
    if (cloud_alpha < 0.01) discard;

    vec3 N = normalize(f_normal);
    vec3 V = normalize(u_camera_pos - f_world_pos);
    vec3 total_diffuse = vec3(0.0);

    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        vec3 frag_to_star = star_pos - f_world_pos;
        float dist_to_star = length(frag_to_star);
        if (dist_to_star < 1e-5) continue;

        vec3 L = frag_to_star / dist_to_star;
        float NdotL = dot(N, L);
        float diffuse = clamp((NdotL + 0.15) / 1.15, 0.0, 1.0);

        float star_lum = u_stars_colors[s].a;
        float falloff = star_lum / (dist_to_star * dist_to_star);
        vec3 star_color = u_stars_colors[s].rgb;

        total_diffuse += star_color * (diffuse * falloff);
    }

    vec3 final_color = cloud_albedo * total_diffuse;
    if (u_hdr_enabled) {
        final_color *= u_exposure;
    }

    out_color = vec4(final_color, cloud_alpha);
}
