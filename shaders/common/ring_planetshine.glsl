#include "surface_phase.glsl"

void ring_planetshine_add(vec3 light, vec3 flux, vec3 pole,
                             inout vec3 same_flux, inout vec3 other_flux,
                             inout vec3 same_direction, inout vec3 other_direction) {
    float weight = dot(flux, vec3(0.2126, 0.7152, 0.0722));
    float top_weight = smoothstep(-1e-4, 1e-4, dot(pole, light));
    same_flux += flux*top_weight;
    same_direction += light*(weight*top_weight);
    other_flux += flux*(1.0-top_weight);
    other_direction += light*(weight*(1.0-top_weight));
}

float ring_planetshine_phase(float cosine) {
    float beta = acos(clamp(-cosine, -1.0, 1.0));
    float beta2 = beta*beta;
    return beta < 0.1
        ? beta*beta2*(1.0/3.0-beta2/30.0+beta2*beta2/840.0)/3.141592653589793
        : (sin(beta)-beta*cos(beta))/3.141592653589793;
}

void ring_planetshine_incident(uint host_index, vec3 hit_local, vec3 pole_n,
                              out vec3 same_flux, out vec3 other_flux,
                              out vec3 same_direction, out vec3 other_direction) {
    same_flux = other_flux = same_direction = other_direction = vec3(0.0);
    if (u_scene.ringshine_params.z == 0u) return;
    vec3 host_center = ring_hosts[host_index].center_min_r.xyz;
    uint host_body = ring_hosts[host_index].counts_indices.x;
    for (int i = 0; i < 4; ++i) {
        vec3 flux = bodies[host_body].shine_irradiance[i].rgb;
        if (!any(greaterThan(flux, vec3(0.0)))) continue;
        ring_planetshine_add(bodies[host_body].shine_direction[i].xyz, flux, pole_n,
                                same_flux, other_flux, same_direction, other_direction);
    }
    if (u_scene.ringshine_params.w == 1u) return;
    vec3 to_host = -hit_local;
    float distance = length(to_host), radius = bodies[host_body].pos_radius.w;
    if (distance <= max(radius, 1e-12)) return;
    float ratio = clamp(radius/distance, 0.0, 0.9999);
    vec3 center_direction = to_host/max(distance, 1e-12);
    vec3 disk_pole = pole_n-center_direction*dot(pole_n, center_direction);
    disk_pole /= max(length(disk_pole), 1e-12);
    float cutoff = -sqrt(max(0.0, 1.0-ratio*ratio));
    for (uint s = 0u; s < u_scene.num_stars; ++s) {
        if (!star_lights_body(host_body, s)) continue;
        vec3 to_star = u_scene.stars_pos_radius[s].xyz-host_center;
        float star_distance = length(to_star);
        if (star_distance <= 1e-12) continue;
        float cosine = dot(to_star/star_distance, -center_direction);
        if (cosine <= cutoff) continue;
        float mu = clamp(2.0*(cosine-cutoff)/(1.0-cutoff)-1.0, -1.0, 1.0);
        SurfaceMaterial material = bodies[host_body].material;
        float phase = material.grain.x < 0.5 ? (2.0/3.0)*material.surface.w*ring_planetshine_phase(mu)
            : material_disk_response(material, mu, asin(clamp(u_scene.stars_pos_radius[s].w/star_distance, 0.0, 0.84)));
        float scale = u_scene.hdr_enabled != 0u ? u_scene.stars_color_lum[s].w/(star_distance*star_distance) : 1.0;
        vec3 flux = clamp(bodies[host_body].color_is_star.rgb, 0.0, 1.0)*u_scene.stars_color_lum[s].rgb
                  *(scale*phase*ratio*ratio);
        float flattening = clamp(bodies[host_body].pole_oblateness.w, 0.0, 0.95);
        float pole_view = dot(pole_n, center_direction);
        float polar = 1.0-flattening;
        flux *= sqrt(polar*polar+(1.0-polar*polar)*pole_view*pole_view);
        if (!any(greaterThan(flux, vec3(0.0)))) continue;
        vec3 sun = to_star/star_distance;
        vec3 toward_lit = sun-center_direction*dot(sun, center_direction);
        toward_lit /= max(length(toward_lit), 1e-12);
        float shift = 0.45*(1.0-mu);
        vec3 lit_offset = toward_lit*shift;
        float spread = 0.42441318158*sqrt(max(0.0, 1.0-shift*shift));
        for (int side = 0; side < 2; ++side) {
            vec3 offset = lit_offset+disk_pole*(side == 0 ? spread : -spread);
            offset *= min(1.0, 0.9999/max(length(offset), 1e-12));
            vec3 light = normalize(center_direction*sqrt(max(0.0, 1.0-ratio*ratio*dot(offset, offset)))+ratio*offset);
            ring_planetshine_add(light, flux*0.5, pole_n,
                                same_flux, other_flux, same_direction, other_direction);
        }
    }
}
