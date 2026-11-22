#include "ring_hosts.glsl"
#include "ringshine_lookup.glsl"

vec3 cloud_ringshine(uint body, vec3 local, vec3 world_pos, bool fluence) {
    vec3 light = vec3(0.0);
    float dist2 = dot(local, local);
    uint ring_slot = bodies[body].surface_flags.z;
    if (u_scene.ringshine_params.x == 0u || ring_slot == 0u || dist2 < 1e-24) return light;
    uint r = ring_slot - 1u;
    vec3 pole = ring_hosts[r].pole_max_r.xyz;
    float host_r = ring_hosts[r].tangent_radius.w;
    if (dist2 > 2.25*host_r*host_r) return light;
    vec3 up = local*inversesqrt(dist2);
    vec3 planar = up-pole*dot(up, pole);
    float planar_length = length(planar);
    planar = planar_length > 1e-5 ? planar/planar_length : vec3(1.0, 0.0, 0.0);
    float slat = dot(up, pole);
    for (uint s = 0u; s < u_scene.num_stars; ++s) {
        if (!star_lights_body(body, s)) continue;
        vec3 to_star = u_scene.stars_pos_radius[s].xyz-world_pos;
        float d_au = max(length(to_star), 1e-5);
        vec3 anti = -to_star/d_au;
        anti -= pole*dot(anti, pole);
        float anti_length = length(anti);
        anti = anti_length > 1e-5 ? anti/anti_length : vec3(1.0, 0.0, 0.0);
        float phi = atan(dot(cross(anti, planar), pole), dot(planar, anti));
        vec4 tile = ringshine_sample_rgba(phi, slat, int(r), int(s));
        vec3 irradiance = tile.rgb;
        if (fluence) {
            float projected = dot(tile.rgb, vec3(0.2126, 0.7152, 0.0722));
            irradiance = projected > 0.0 ? tile.rgb*(tile.a/projected) : vec3(0.0);
        }
        float scale = u_scene.hdr_enabled != 0u ? u_scene.stars_color_lum[s].w/(d_au*d_au) : 1.0;
        light += irradiance*u_scene.stars_color_lum[s].rgb*scale;
    }
    return light/3.14159265358979323846;
}
