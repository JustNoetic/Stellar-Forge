#version 460

#include "common/celestial_scene.glsl"
#include "common/surface_phase.glsl"
#include "common/eclipse_shadow.glsl"
#include "common/refraction.glsl"

layout(std430, set = 0, binding = 27) readonly buffer AtmospherePointResponses { vec4 atmosphere_points[]; };
layout(push_constant) uniform PointPass { uint atmosphere; } point_pass;

layout(location = 0) out vec2 f_pixel_offset;
layout(location = 1) flat out float f_point_radius;
layout(location = 2) flat out vec3 f_flux;

float point_visibility(vec3 center, uint self_idx, float stellar_radius) {
    float distance = length(center);
    vec3 ray = center / max(distance, 1e-12);
    float visibility = 1.0;
    for (uint i = 0; i < u_scene.num_bodies; ++i) {
        if (i == self_idx) continue;
        BodyInstanceGPU caster = bodies[i];
        if (refraction_active() && all(equal(caster.pos_radius.xyz, u_refraction.host.xyz))) continue;
        if (stellar_radius > 0.0) {
            float along = dot(caster.pos_radius.xyz, ray);
            if (along <= 0.0 || along >= distance) continue;
            vec3 perpendicular = caster.pos_radius.xyz - ray * along;
            float source_radius = stellar_radius * (along / distance);
            float bound = caster.pos_radius.w + source_radius;
            if (dot(perpendicular, perpendicular) > bound * bound * 1.00002) continue;
            float oblateness = clamp(caster.pole_oblateness.w, 0.0, 0.95);
            visibility *= eclipse_caster_visibility(caster.pos_radius.w, caster.pos_radius.w * (1.0 - oblateness),
                                                   body_pole(caster), ray, perpendicular, source_radius, u_scene.eclipse_params.x != 0u);
            if (visibility <= 1e-4) return 0.0;
            continue;
        }
        float weight = caster.spin_flags.z > 0.5 ? 1.0
                       : mesh_weight(apparent_radius_px(caster.pos_radius));
        if (weight <= 0.0) continue;

        vec3 pole = body_pole(caster);
        float stretch = 1.0 / max(1.0 - caster.pole_oblateness.w, 1e-4) - 1.0;
        vec3 oc = caster.pos_radius.xyz;
        vec3 scaled_ray = ray + pole * (dot(ray, pole) * stretch);
        vec3 scaled_oc = oc + pole * (dot(oc, pole) * stretch);
        float ray_scale = length(scaled_ray);
        vec3 direction = scaled_ray / ray_scale;
        float along = dot(scaled_oc, direction);
        float perpendicular = length(cross(scaled_oc, direction));
        float radius = caster.pos_radius.w;
        if (along <= 0.0 || perpendicular >= radius) continue;
        float hit = (along - sqrt(max(radius * radius - perpendicular * perpendicular, 0.0))) / ray_scale;
        if (hit > 0.0 && hit < distance) visibility *= 1.0 - weight;
    }
    return visibility;
}

vec3 disc_radiance(BodyInstanceGPU b, vec3 V) {
    if (b.color_is_star.w < 0.5) {
        vec3 sum = vec3(0.0);
        for (uint s = 0; s < u_scene.num_stars; ++s) {
            if (!star_lights_body(uint(b.spin_flags.w), s)) continue;
            vec3 delta = u_scene.stars_pos_radius[s].xyz-b.pos_radius.xyz;
            float distance = max(length(delta), 1e-12);
            float flux = u_scene.hdr_enabled != 0u ? u_scene.stars_color_lum[s].w/(distance*distance) : 1.0;
            float radius = asin(clamp(u_scene.stars_pos_radius[s].w/distance, 0.0, 0.84));
            vec3 eclipse = u_scene.eclipse_params.y == 0u ? vec3(1.0)
                : evaluate_eclipse_shadow(b.pos_radius.xyz, u_scene.stars_pos_radius[s].xyz, u_scene.stars_pos_radius[s].w, uint(b.spin_flags.w), s);
            vec4 air = atmosphere_points[uint(b.spin_flags.w)*16u+s];
            vec3 response = air.w > 0.5 ? air.rgb
                : b.color_is_star.rgb*material_disk_response(b.material, dot(V, delta/distance), radius);
            sum += u_scene.stars_color_lum[s].rgb*flux*response*eclipse;
        }
        for (int k = 0; k < 4; ++k)
            sum += b.color_is_star.rgb*b.shine_irradiance[k].rgb*material_disk_response(b.material, dot(V, b.shine_direction[k].xyz), b.shine_direction[k].w);
        return sum;
    }
    vec3 tangent = normalize(cross(V, abs(V.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(V, tangent);
    vec3 total = vec3(0.0);
    const int SAMPLES = 64;
    for (int i = 0; i < SAMPLES; ++i) {
        float r2 = (float(i) + 0.5) / float(SAMPLES);
        float angle = float(i) * 2.39996323;
        vec3 N = sqrt(1.0 - r2) * V
                 + sqrt(r2) * (cos(angle) * tangent + sin(angle) * bitangent);
        total += stellar_radiance(b, N, V);
    }
    return total / float(SAMPLES);
}

void main() {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    f_pixel_offset = vec2(0.0);
    f_point_radius = 1.0;
    f_flux = vec3(0.0);

    uint idx = gl_InstanceIndex;
    BodyInstanceGPU b = bodies[idx];
    if (b.surface_flags.x != 0u) return;
    bool atmosphere = atmosphere_points[idx*16u].w > 0.5;
    if (atmosphere != (point_pass.atmosphere != 0u)) return;
    float radius_px = apparent_radius_px(b.pos_radius);
    float weight = 1.0 - mesh_weight(radius_px);
    if (b.spin_flags.z > 0.5 || weight <= 0.0) return;

    vec3 center = b.pos_radius.xyz;
    vec4 apparent = refract_point(center);
    if (apparent.w <= 0.0) return;
    vec4 clip = u_scene.view_proj * vec4(apparent.xyz, 1.0);
    if (clip.w <= 0.0) return;

    float half_size = max(radius_px, 0.5 * u_scene.point_params.x);
    vec2 padding = 2.0 * half_size / u_scene.screen_params.xy;
    if (any(greaterThan(abs(clip.xy / clip.w), vec2(1.0) + padding))) return;

    float visibility = point_visibility(center, idx, b.color_is_star.w > 0.5 ? b.pos_radius.w : 0.0);
    if (visibility <= 0.0) return;

    vec3 V = normalize(-center);
    float polar_radius = max(1.0 - b.pole_oblateness.w, 1e-4);
    float pole_view = dot(body_pole(b), V);
    float area_scale = sqrt(polar_radius * polar_radius
                            + (1.0 - polar_radius * polar_radius) * pole_view * pole_view);
    vec3 flux = disc_radiance(b, V) * (POINT_PI * radius_px * radius_px * area_scale);
    if (u_scene.hdr_enabled != 0u) flux *= u_scene.screen_params.w;
    flux *= weight * visibility * apparent.w;
    if (!point_flux_visible(flux)) return;

    vec2 center_px = (clip.xy / clip.w * 0.5 + 0.5) * u_scene.screen_params.xy;
    flux /= point_profile_sum(center_px, half_size);

    vec2 q = POINT_QUAD_VERTS[gl_VertexIndex];
    clip.xy += q * padding * clip.w;
    gl_Position = clip;
    f_pixel_offset = q * half_size;
    f_point_radius = half_size;
    f_flux = flux;
}
