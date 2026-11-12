// Eclipse and ring-shadow visibility of star s at a cloud point (camera-relative world position), matching the
// surface shader's per-star shadow so clouds darken with the ground and haze beneath them.
#include "eclipse_shadow.glsl"
#include "surface_ring_shadow.glsl"

vec3 cloud_star_shadow(uint body, vec3 world_pos, uint s) {
    vec3 star_pos = u_scene.stars_pos_radius[s].xyz;
    float star_radius = u_scene.stars_pos_radius[s].w;
    vec3 shadow = evaluate_eclipse_shadow(world_pos, star_pos, star_radius, body, s);
    if (any(greaterThan(shadow, vec3(0.0))) && u_scene.num_ring_hosts > 0u) {
        vec3 to_star = star_pos-world_pos;
        float distance = max(length(to_star), 1e-5);
        uint candidates = u_scene.ring_shadow_params.y != 0u
            ? (1u << u_scene.num_ring_hosts)-1u
            : eclipse_words[u_scene.ring_shadow_params.x+body*u_scene.num_stars+s];
        if (candidates != 0u)
            shadow *= compute_surface_ring_shadow(world_pos, to_star/distance, star_radius/distance, distance, candidates);
    }
    return shadow;
}
