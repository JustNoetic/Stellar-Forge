#include "eclipse_optics.glsl"
#include "ring_shadow_filter.glsl"

struct AtmosphereShadowCaster {
    vec4 position_radius;
    vec4 pole_inner;
    vec4 support;
};
layout(std430, set = 0, binding = 11) readonly buffer AtmosphereShadows {
    vec4 shadow_eye_focal;
    uvec4 shadow_flags;
    vec4 shadow_stars[16];
    uvec4 shadow_ranges[16];
    mat4 shadow_previous_projection;
    uvec4 shadow_temporal;
    vec4 shadow_history_params;
    AtmosphereShadowCaster shadow_casters[];
};
layout(set = 0, binding = 12) uniform sampler2DArray atmosphere_ring_shadow_lut;
layout(set = 0, binding = 20) uniform sampler3D u_stbn_tex;

vec3 atmosphere_to_physical(vec3 p) {
    vec3 world = atm.axis_x.xyz*p.x+atm.axis_y.xyz*p.y+atm.axis_z.xyz*p.z;
    return world-atm.world_pole.xyz*dot(world, atm.world_pole.xyz)*(1.0-atm.reference_shape.y);
}

vec2 shadow_quadratic_interval(float a, float b, float c, vec2 domain) {
    if (abs(a) < 1e-12) {
        if (abs(b) < 1e-12) return c <= 0.0 ? domain : vec2(0.0);
        float root = -c/(2.0*b);
        return b > 0.0 ? vec2(domain.x, min(domain.y, root)) : vec2(max(domain.x, root), domain.y);
    }
    float discriminant = b*b-a*c;
    if (discriminant < 0.0) return a < 0.0 ? domain : vec2(0.0);
    float q = -b-(b >= 0.0 ? 1.0 : -1.0)*sqrt(discriminant);
    vec2 roots = abs(q) > 1e-20 ? vec2(q/a, c/q) : vec2(-b/a);
    roots = vec2(min(roots.x, roots.y), max(roots.x, roots.y));
    if (a > 0.0) return vec2(max(domain.x, roots.x), min(domain.y, roots.y));
    if (domain.x < roots.x && domain.y > roots.y) return domain;
    if (domain.x < roots.x) return vec2(domain.x, min(domain.y, roots.x));
    return vec2(max(domain.x, roots.y), domain.y);
}

vec2 atmosphere_shadow_interval(uint index, bool ring, vec3 origin, vec3 direction, vec4 star, vec2 domain) {
    if (shadow_flags.w != 0u) return domain;
    AtmosphereShadowCaster caster = shadow_casters[index];
    vec3 center = caster.position_radius.xyz;
    float distance = max(length(star.xyz), 1e-6);
    if (!ring) {
        vec3 n0 = cross(center-origin, (star.xyz-origin)/distance);
        vec3 n1 = cross((star.xyz-center)/distance, direction);
        float radius = caster.position_radius.w*(length(star.xyz)+top_radius())/distance
                     + star.w*(length(center)+top_radius())/distance;
        radius = radius*1.0001+0.001;
        float speed2 = dot(n1, n1);
        if (speed2 < 1e-12) return length(n0) <= radius ? domain : vec2(0.0);
        float closest = -dot(n0, n1)/speed2;
        vec3 perpendicular = cross(n0, n1);
        float discriminant = radius*radius-dot(perpendicular, perpendicular)/speed2;
        if (discriminant < 0.0) return vec2(0.0);
        float half_span = sqrt(discriminant/speed2);
        return vec2(max(domain.x, closest-half_span), min(domain.y, closest+half_span));
    }
    if (caster.support.x < 0.0) return domain;
    vec3 pole = caster.pole_inner.xyz;
    vec3 rel = origin-center, to_star = star.xyz-origin;
    float a0 = dot(to_star, pole)/distance, a1 = -dot(direction, pole)/distance;
    vec3 n0 = rel*a0-to_star*(dot(rel, pole)/distance);
    vec3 n1 = direction*(dot(star.xyz-center, pole)/distance)
            -(star.xyz-center)*(dot(direction, pole)/distance);
    float radius = caster.support.x;
    return shadow_quadratic_interval(dot(n1, n1)-radius*radius*a1*a1,
        dot(n0, n1)-radius*radius*a0*a1, dot(n0, n0)-radius*radius*a0*a0, domain);
}

float atmosphere_shadow_visibility(vec3 p, uint star_index) {
    vec4 star = shadow_stars[star_index];
    vec3 to_star = star.xyz-p;
    float distance = length(to_star);
    if (distance <= 1e-6) return 1.0;
    vec3 light = to_star/distance;
    uvec4 range = shadow_ranges[star_index];
    float visibility = 1.0;
    for (uint j = 0u; j < range.y; ++j) {
        AtmosphereShadowCaster caster = shadow_casters[range.x+j];
        vec3 delta = caster.position_radius.xyz-p;
        float along = dot(delta, light);
        if (along <= 0.0 || along >= distance) continue;
        vec3 perpendicular = delta-light*along;
        float source_radius = star.w*(along/distance);
        float bound = caster.position_radius.w+source_radius;
        if (dot(perpendicular, perpendicular) > bound*bound*1.00002) continue;
        visibility *= eclipse_caster_visibility(caster.position_radius.w, caster.pole_inner.w,
            caster.pole_inner.xyz, light, perpendicular, source_radius, shadow_flags.z != 0u);
        if (visibility <= 1e-4) return 0.0;
    }
    float footprint = length(p-shadow_eye_focal.xyz)/shadow_eye_focal.w;
    for (uint j = 0u; j < range.w; ++j) {
        AtmosphereShadowCaster caster = shadow_casters[range.z+j];
        float denominator = dot(light, caster.pole_inner.xyz);
        if (abs(denominator) <= 1e-6) continue;
        float hit = dot(caster.position_radius.xyz-p, caster.pole_inner.xyz)/denominator;
        if (hit <= 0.0) continue;
        vec3 hit_rel = (p-caster.position_radius.xyz)+light*hit;
        visibility *= 1.0-ring_shadow_extinction(atmosphere_ring_shadow_lut, int(caster.support.y),
            hit_rel, caster.pole_inner.xyz, light, abs(denominator), hit,
            vec2(caster.pole_inner.w, caster.position_radius.w), star.w/distance, footprint, shadow_flags.y != 0u);
        if (visibility <= 1e-4) return 0.0;
    }
    return visibility;
}

float atmosphere_shadow_step(vec3 p, vec3 direction, vec4 star, uvec4 range, float step) {
    vec3 to_star = star.xyz-p;
    float distance = max(length(to_star), 1e-6);
    vec3 light = to_star/distance;
    for (uint j = 0u; j < range.y; ++j) {
        AtmosphereShadowCaster caster = shadow_casters[range.x+j];
        float source_radius = star.w*length(caster.position_radius.xyz-p)/distance;
        float width = max(source_radius*0.5, caster.position_radius.w*0.05);
        float speed = length(cross(direction, light));
        step = min(step, width/max(speed, 1e-6));
    }
    for (uint j = 0u; j < range.w; ++j) {
        AtmosphereShadowCaster caster = shadow_casters[range.z+j];
        vec3 pole = caster.pole_inner.xyz;
        float denominator = dot(to_star, pole);
        if (abs(denominator) <= distance*1e-6) continue;
        float numerator = dot(caster.position_radius.xyz-p, pole);
        float fraction = numerator/denominator;
        vec3 hit_rel = p-caster.position_radius.xyz+to_star*fraction;
        vec2 bounds = vec2(caster.pole_inner.w, caster.position_radius.w);
        float footprint = length(p-shadow_eye_focal.xyz)/shadow_eye_focal.w;
        vec3 uv = ring_shadow_coordinates(hit_rel, pole, light, abs(denominator)/distance,
            max(fraction, 0.0)*distance, bounds, star.w/distance, footprint);
        float end_denominator = denominator-dot(direction, pole)*step;
        float minimum_denominator = min(abs(denominator), abs(end_denominator));
        if (denominator*end_denominator <= 0.0) minimum_denominator = 1e-6;
        vec3 velocity = direction*(1.0-fraction)
                      +to_star*(dot(direction, pole)*(numerator-denominator)/(denominator*denominator));
        float speed = length(velocity)*pow(abs(denominator)/max(minimum_denominator, 1e-6), 2.0);
        float width = uv.y*(bounds.y-bounds.x);
        step = min(step, 0.125*width/max(speed, 1e-6));
    }
    return max(step, 0.001);
}

vec3 atmosphere_shadow_reference(vec3 direction, vec2 segment, float path) {
    if (shadow_flags.x == 0u) return vec3(0.0);
    vec3 entry = atm.camera.xyz+direction*segment.x;
    vec3 origin = atmosphere_to_physical(entry);
    vec3 physical_direction = atmosphere_to_physical(direction);
    vec2 domain = vec2(0.0, path);
    float entry_radius = clamp(length(entry), bottom_radius(), top_radius());
    float entry_mu = clamp(dot(entry, direction)/entry_radius, -1.0, 1.0);
    bool hits_ground = entry_mu < 0.0 && entry_radius*entry_radius*(entry_mu*entry_mu-1.0)+bottom_radius()*bottom_radius() >= 0.0;
    vec3 total_energy = vec3(0.0);
    vec3 total_loss = vec3(0.0);
    for (uint s = 0u; s < uint(atm.camera.w); ++s) {
        if (max(max(atm.sun_irradiance[s].r, atm.sun_irradiance[s].g), atm.sun_irradiance[s].b) <= 0.0) continue;
        uvec4 range = shadow_ranges[s];
        uint count = range.y+range.w;
        if (count == 0u) {
            if (atm.camera.w <= 1.0) continue;
            float t = domain.x;
            float base_step = max((domain.y-domain.x)/(shadow_flags.w != 0u ? 1024.0 : 48.0), 0.001);
            vec3 sun = atm.sun_direction[s].xyz;
            float phase_mu = dot(direction, sun);
            for (uint sample_index = 0u; sample_index < 8192u && t < domain.y; ++sample_index) {
                float step = min(base_step, domain.y-t);
                vec3 start = entry+direction*t;
                float scale = atm.radii_heights.z;
                if (dot(atm.mie_extinction.rgb, vec3(1.0)) > 0.0) scale = min(scale, atm.radii_heights.w);
                step = min(step, 0.125*scale/max(abs(dot(normalize(start), direction)), 0.05));
                step /= float(transport_subdivisions(entry, direction, t, step));
                step = min(step, domain.y-t);
                if (sample_index == 8191u) step = domain.y-t;
                float midpoint = t+0.5*step;
                vec3 q = entry+direction*midpoint;
                vec3 r, m, e;
                medium(q, r, m, e);
                float radius = clamp(length(q), bottom_radius(), top_radius());
                float mu = clamp(dot(q, direction)/radius, -1.0, 1.0);
                vec3 view_transmission = bruneton_transmittance(transmittance_lut, entry_radius, entry_mu, radius, mu, hits_ground);
                if (midpoint < 2.0) {
                    vec3 sr, sm, se;
                    medium((entry+q)*0.5, sr, sm, se);
                    view_transmission = mix(exp(-se*midpoint), view_transmission, smoothstep(0.5, 2.0, midpoint));
                }
                vec3 direct = sun_transmittance(transmittance_lut, q, sun)*(r*rayleigh_phase(phase_mu)+aerosol_source(q, phase_mu));
                vec3 multiple = textureLod(multiscatter_lut, multiscatter_uv(q, sun), 0.0).rgb*(r+m);
                vec3 weight = view_transmission*atm.sun_irradiance[s].rgb*(direct+multiple)*step;
                total_energy += weight;
                t += step;
            }
            continue;
        }
        vec4 star = shadow_stars[s];
        float cursor = domain.x;
        for (uint component = 0u; component < count && cursor < domain.y; ++component) {
            vec2 interval = vec2(domain.y, domain.x);
            for (uint j = 0u; j < count; ++j) {
                bool ring = j >= range.y;
                uint index = ring ? range.z+j-range.y : range.x+j;
                vec2 candidate = atmosphere_shadow_interval(index, ring, origin, physical_direction, star, vec2(cursor, domain.y));
                if (candidate.y > candidate.x && candidate.x < interval.x) interval = candidate;
            }
            if (interval.y <= interval.x) break;
            for (uint merge = 0u; merge < count; ++merge) {
                float end = interval.y;
                for (uint j = 0u; j < count; ++j) {
                    bool ring = j >= range.y;
                    uint index = ring ? range.z+j-range.y : range.x+j;
                    vec2 candidate = atmosphere_shadow_interval(index, ring, origin, physical_direction, star, vec2(cursor, domain.y));
                    if (candidate.y > candidate.x && candidate.x <= interval.y) interval.y = max(interval.y, candidate.y);
                }
                if (interval.y == end) break;
            }
            float t = interval.x;
            float base_step = max((interval.y-interval.x)/(shadow_flags.w != 0u ? 1024.0 : 48.0), 0.001);
            for (uint sample_index = 0u; sample_index < 8192u && t < interval.y; ++sample_index) {
                float step = min(base_step, interval.y-t);
                vec3 start = entry+direction*t;
                float scale = atm.radii_heights.z;
                if (dot(atm.mie_extinction.rgb, vec3(1.0)) > 0.0) scale = min(scale, atm.radii_heights.w);
                step = min(step, 0.125*scale/max(abs(dot(normalize(start), direction)), 0.05));
                step /= float(transport_subdivisions(entry, direction, t, step));
                step = atmosphere_shadow_step(origin+physical_direction*t, physical_direction, star, range, step);
                step = min(step, interval.y-t);
                if (sample_index == 8191u) step = interval.y-t;
                float midpoint = t+0.5*step;
                float blocked = 1.0-atmosphere_shadow_visibility(origin+physical_direction*midpoint, s);
                vec3 q = entry+direction*midpoint;
                vec3 r, m, e;
                medium(q, r, m, e);
                float radius = clamp(length(q), bottom_radius(), top_radius());
                float mu = clamp(dot(q, direction)/radius, -1.0, 1.0);
                vec3 view_transmission = bruneton_transmittance(transmittance_lut, entry_radius, entry_mu, radius, mu, hits_ground);
                if (midpoint < 2.0) {
                    vec3 sr, sm, se;
                    medium((entry+q)*0.5, sr, sm, se);
                    view_transmission = mix(exp(-se*midpoint), view_transmission, smoothstep(0.5, 2.0, midpoint));
                }
                vec3 sun = atm.sun_direction[s].xyz;
                float phase_mu = dot(direction, sun);
                vec3 direct = sun_transmittance(transmittance_lut, q, sun)*(r*rayleigh_phase(phase_mu)+aerosol_source(q, phase_mu));
                vec3 multiple = textureLod(multiscatter_lut, multiscatter_uv(q, sun), 0.0).rgb*(r+m);
                vec3 weight = view_transmission*atm.sun_irradiance[s].rgb*(direct+multiple)*step;
                total_energy += weight;
                total_loss += weight*blocked;
                t += step;
            }
            cursor = interval.y;
        }
    }
    if (total_energy.r > 1e-20 || total_energy.g > 1e-20 || total_energy.b > 1e-20)
        return clamp(total_loss / max(total_energy, vec3(1e-20)), vec3(0.0), vec3(1.0));
    return vec3(0.0);
}

vec2 atmosphere_shadow_next(uvec4 range, vec3 origin, vec3 direction, vec4 star, vec2 domain) {
    uint count = range.y+range.w;
    vec2 interval = vec2(domain.y, domain.x);
    for (uint j = 0u; j < count; ++j) {
        bool ring = j >= range.y;
        vec2 candidate = atmosphere_shadow_interval(ring ? range.z+j-range.y : range.x+j,
            ring, origin, direction, star, domain);
        if (candidate.y > candidate.x && candidate.x < interval.x) interval = candidate;
    }
    if (interval.y <= interval.x) return interval;
    for (uint merge = 0u; merge < count; ++merge) {
        float previous = interval.y;
        for (uint j = 0u; j < count; ++j) {
            bool ring = j >= range.y;
            vec2 candidate = atmosphere_shadow_interval(ring ? range.z+j-range.y : range.x+j,
                ring, origin, direction, star, domain);
            if (candidate.y > candidate.x && candidate.x <= interval.y) interval.y = max(interval.y, candidate.y);
        }
        if (previous == interval.y) break;
    }
    return interval;
}

float map_t_to_s(float t, float s_start, float s_end, float s_min, float p) {
    float L = s_end - s_start;
    if (L < 1e-4) return s_start;
    if (abs(p - 1.0) < 0.01) return s_start + t * L;
    float t_min = clamp((s_min - s_start) / L, 0.0, 1.0);
    if (t <= t_min) {
        if (t_min < 1e-5) return s_min;
        float u = t / t_min;
        return s_min - (s_min - s_start) * pow(1.0 - u, p);
    } else {
        if (t_min > 0.99999) return s_min;
        float u = (t - t_min) / (1.0 - t_min);
        return s_min + (s_end - s_min) * pow(u, p);
    }
}

float get_ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float shadow_random(ivec2 pixel, uint sample_index, uint star_index, uint sequence) {
    float base = 0.0;
    if (shadow_temporal.y == 1u) {
        ivec3 sz = textureSize(u_stbn_tex, 0);
        if (sz.x >= 64) {
            ivec3 coord = ivec3(pixel.x & 127, pixel.y & 127, int(sequence) & 63);
            float stbn = texelFetch(u_stbn_tex, coord, 0).r;
            float cycle = floor(float(sequence) / 64.0);
            base = fract(stbn + cycle * 0.61803398875);
        } else {
            float base_ign = get_ign(vec2(pixel));
            base = fract(base_ign + float(sequence & 63u) * 0.61803398875);
        }
    } else {
        float base_ign = get_ign(vec2(pixel));
        base = fract(base_ign + float(sequence & 63u) * 0.61803398875);
    }
    if (star_index > 0u) {
        base = fract(base + float(star_index) * 0.38196601125);
    }
    return base;
}

vec3 atmosphere_shadow_deficit(vec3 direction, vec2 segment, float path, ivec2 pixel, uint sequence,
                              float prefix_path, out vec3 prefix_deficit, out bvec2 supported) {
    supported = bvec2(false);
    prefix_deficit = vec3(0.0);
    if (shadow_flags.x == 0u) return vec3(0.0);
    if (shadow_flags.w != 0u) {
        supported = bvec2(true, prefix_path > 0.0);
        if (prefix_path > 0.0) prefix_deficit = atmosphere_shadow_reference(direction, segment, prefix_path);
        return atmosphere_shadow_reference(direction, segment, path);
    }
    vec3 entry = atm.camera.xyz+direction*segment.x;
    vec3 origin = atmosphere_to_physical(entry), physical_direction = atmosphere_to_physical(direction);
    float entry_radius = clamp(length(entry), bottom_radius(), top_radius());
    float entry_mu = clamp(dot(entry, direction)/entry_radius, -1.0, 1.0);
    bool hits_ground = entry_mu < 0.0 && entry_radius*entry_radius*(entry_mu*entry_mu-1.0)+bottom_radius()*bottom_radius() >= 0.0;
    uint samples = shadow_temporal.z <= 4u ? 4u : shadow_temporal.z <= 8u ? 8u : 16u;

    vec3 total_energy = vec3(0.0);
    vec3 total_loss = vec3(0.0);
    vec3 total_prefix_energy = vec3(0.0);
    vec3 total_prefix_loss = vec3(0.0);

    for (uint s = 0u; s < uint(atm.camera.w); ++s) {
        if (max(max(atm.sun_irradiance[s].r, atm.sun_irradiance[s].g), atm.sun_irradiance[s].b) <= 0.0) continue;
        uvec4 range = shadow_ranges[s];
        uint count = range.y+range.w;
        vec4 star = shadow_stars[s];
        vec3 sun = atm.sun_direction[s].xyz;
        float phase_mu = dot(direction, sun);
        float rayleigh_factor = rayleigh_phase(phase_mu);
        vec3 mie_factor = atm.mie_scattering.rgb*mie_phase(phase_mu);
        vec3 particle_factor[4];
        for (int layer = 0; layer < int(atm.aerosol_count.x); ++layer)
            particle_factor[layer] = atm.aerosol_scattering[layer].rgb*particle_phase(layer, phase_mu);

        vec2 interval = count > 0u ? atmosphere_shadow_next(range, origin, physical_direction, star, vec2(0.0, path)) : vec2(0.0);
        bool has_shadow = count > 0u && interval.y > interval.x;
        if (!has_shadow) {
            if (atm.camera.w <= 1.0) continue;
        } else {
            supported.x = true;
            if (prefix_path > 0.0) supported.y = true;
        }

        float first = 0.0;
        float last = path;
        vec3 a = entry;
        vec3 r0, m0, e0;
        medium(a, r0, m0, e0);
        float dist = last - first;
        float concentration = 1.5 * dist * dot(e0, vec3(1.0 / 3.0));
        bool optical_sampling = concentration > 1e-3 && (dot(a, direction) <= 0.0 || concentration > 1.0);
        float closest = clamp(-dot(entry, direction), first, last);

        for (uint i = 0u; i < samples; ++i) {
            float jitter = shadow_random(pixel, i, s, sequence);
            float q = (float(i) + jitter) / float(samples);
            float t;
            float ds;
            if (optical_sampling) {
                float denom = 1.0 + concentration * (1.0 - q);
                t = first + dist * q / denom;
                ds = dist * (1.0 + concentration) / (denom * denom * float(samples));
            } else {
                float q0 = float(i) / float(samples);
                float q1 = float(i + 1u) / float(samples);
                float s0 = map_t_to_s(q0, first, last, closest, 2.0);
                float s1 = map_t_to_s(q1, first, last, closest, 2.0);
                t = mix(s0, s1, jitter);
                ds = max(0.0, s1 - s0);
            }

            vec3 q_pos = entry + direction * t;
            vec3 r, m, e;
            medium(q_pos, r, m, e);
            float radius = clamp(length(q_pos), bottom_radius(), top_radius());
            float mu = clamp(dot(q_pos, direction) / radius, -1.0, 1.0);
            vec3 view_transmittance = bruneton_transmittance(transmittance_lut, entry_radius, entry_mu, radius, mu, hits_ground);
            if (t < 2.0) {
                vec3 sr, sm, se;
                medium((entry + q_pos) * 0.5, sr, sm, se);
                view_transmittance = mix(exp(-se * t), view_transmittance, smoothstep(0.5, 2.0, t));
            }

            float height = max(length(q_pos), bottom_radius()) - atm.reference_shape.x;
            vec3 scattering = r * rayleigh_factor + mie_factor * exp(-max(height, 0.0) / atm.radii_heights.w);
            for (int layer = 0; layer < int(atm.aerosol_count.x); ++layer)
                scattering += particle_factor[layer] * particle_density(layer, height);

            vec3 direct = sun_transmittance(transmittance_lut, q_pos, sun) * scattering;
            vec3 multiple = textureLod(multiscatter_lut, multiscatter_uv(q_pos, sun), 0.0).rgb * (r + m);
            vec3 weight = view_transmittance * atm.sun_irradiance[s].rgb * (direct + multiple) * ds;

            float vis = has_shadow ? atmosphere_shadow_visibility(origin + physical_direction * t, s) : 1.0;

            total_energy += weight;
            total_loss += weight * (1.0 - vis);

            if (t < prefix_path) {
                total_prefix_energy += weight;
                total_prefix_loss += weight * (1.0 - vis);
            }
        }
    }

    if (total_prefix_energy.r > 1e-20 || total_prefix_energy.g > 1e-20 || total_prefix_energy.b > 1e-20)
        prefix_deficit = clamp(total_prefix_loss / max(total_prefix_energy, vec3(1e-20)), vec3(0.0), vec3(1.0));
    if (total_energy.r > 1e-20 || total_energy.g > 1e-20 || total_energy.b > 1e-20)
        return clamp(total_loss / max(total_energy, vec3(1e-20)), vec3(0.0), vec3(1.0));
    return vec3(0.0);
}

