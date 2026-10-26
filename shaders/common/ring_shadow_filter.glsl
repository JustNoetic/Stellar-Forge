float ring_integral_sample(sampler2DArray lut, int ring_idx, float u, float mu_coord) {
    float texel = (clamp(u, 0.0, 1.0) * 4096.0 + 0.5) / 4097.0;
    return textureLod(lut, vec3(texel, mu_coord, float(ring_idx)), 0.0).r;
}

float ring_box_extinction(sampler2DArray lut, int ring_idx, float u, float h, float mu_coord) {
    float lo = ring_integral_sample(lut, ring_idx, u - h, mu_coord);
    float hi = ring_integral_sample(lut, ring_idx, u + h, mu_coord);
    return max(0.0, hi - lo) / max(2.0 * h, 1e-6);
}

vec3 ring_shadow_coordinates(vec3 hit_rel, vec3 pole, vec3 light, float incidence, float distance,
                             vec2 bounds, float source_angle, float pixel_size, float softness, bool reference_linear) {
    float span = max(bounds.y - bounds.x, 1e-9);
    float radius = length(hit_rel);
    vec3 radial = radius > 1e-12 ? hit_rel / radius : vec3(1, 0, 0);
    vec3 planar = light - pole * dot(light, pole);
    float cosine = length(planar) > 1e-6 ? dot(radial, normalize(planar)) : 0.0;
    float cosine2 = clamp(cosine * cosine, 0.0, 1.0);
    float mu = max(incidence, reference_linear ? 1.0 / 128.0 : 1e-6);
    float radial_stretch = sqrt(cosine2 / (mu * mu) + 1.0 - cosine2);
    float base_width = source_angle * distance;
    float width = base_width * radial_stretch;
    float shift = 0.0;
    if (!reference_linear) {
        float tangential_width = base_width * sqrt((1.0 - cosine2) / (mu * mu) + cosine2);
        float curvature = tangential_width * (tangential_width / (length(vec2(radius, tangential_width)) + radius + 1e-20));
        shift = 0.25 * curvature;
        width = softness * (width + 0.5 * curvature);
        shift *= softness;
    }
    float pixel_width = max(0.75 * pixel_size * radial_stretch, 0.75 * span / 4096.0);
    float h = max(width, pixel_width) / span;
    float mu_coord = clamp(reference_linear ? mu : incidence, 0.5 / 64.0, 63.5 / 64.0);
    return vec3((radius + shift - bounds.x) / span, h, mu_coord);
}

float ring_shadow_filter(sampler2DArray lut, int ring_idx, vec3 uv, bool single_box, bool reference_linear) {
    if (uv.x + uv.y <= 0.0 || uv.x - uv.y >= 1.0) return 0.0;
    if (single_box) return clamp(ring_box_extinction(lut, ring_idx, uv.x, uv.y, uv.z), 0.0, 1.0);
    float inner = ring_box_extinction(lut, ring_idx, uv.x, 0.36 * uv.y, uv.z);
    float middle = ring_box_extinction(lut, ring_idx, uv.x, 0.72 * uv.y, uv.z);
    float outer = ring_box_extinction(lut, ring_idx, uv.x, uv.y, uv.z);
    vec3 weights = reference_linear ? vec3(0.24, 0.46, 0.30) : vec3(0.08, 0.42, 0.50);
    return clamp(dot(weights, vec3(inner, middle, outer)), 0.0, 1.0);
}

float ring_swept_filter(sampler2DArray lut, int ring_idx, vec3 uv, float sweep, bool single_box) {
    float s2 = sweep*sweep, h2 = uv.y*uv.y;
    float reach = sqrt(h2+s2);
    if (uv.x+reach <= 0.0 || uv.x-reach >= 1.0) return 0.0;
    if (single_box) return clamp(ring_box_extinction(lut, ring_idx, uv.x, reach, uv.z), 0.0, 1.0);
    if (sweep > 3.0*uv.y)
        return clamp(ring_box_extinction(lut, ring_idx, uv.x, sqrt(0.728096*h2+s2), uv.z), 0.0, 1.0);
    float inner = ring_box_extinction(lut, ring_idx, uv.x, sqrt(0.1296*h2+s2), uv.z);
    float middle = ring_box_extinction(lut, ring_idx, uv.x, sqrt(0.5184*h2+s2), uv.z);
    float outer = ring_box_extinction(lut, ring_idx, uv.x, reach, uv.z);
    return clamp(dot(vec3(0.08, 0.42, 0.50), vec3(inner, middle, outer)), 0.0, 1.0);
}

float ring_shadow_extinction(sampler2DArray lut, int ring_idx, vec3 hit_rel, vec3 pole, vec3 light,
                             float incidence, float distance, vec2 bounds, float source_angle, float pixel_size,
                             bool single_box, float softness, bool reference_linear) {
    vec3 uv = ring_shadow_coordinates(hit_rel, pole, light, incidence, distance, bounds,
        source_angle, pixel_size, softness, reference_linear);
    return ring_shadow_filter(lut, ring_idx, uv, single_box, reference_linear);
}

float ring_receiver_extinction(sampler2DArray lut, int ring_idx, vec3 receiver_rel, vec3 pole, vec3 light,
                               vec2 bounds, float source_angle, float star_distance, float pixel_size,
                               bool single_box, float softness, bool reference_linear) {
    float offset = -dot(receiver_rel, pole);
    float denominator = dot(light, pole);
    float coverage = 1.0;
    if (!reference_linear && source_angle > 1e-8) {
        if (offset == 0.0) return 0.0;
        float side = offset > 0.0 ? 1.0 : -1.0;
        float signed_mu = denominator * side;
        float angular_height = source_angle * sqrt(max(0.0, 1.0 - denominator * denominator));
        if (angular_height > 1e-8) {
            coverage = smoothstep(-angular_height, angular_height, signed_mu);
            if (coverage <= 0.0) return 0.0;
            float effective_mu = max(signed_mu, 0.5 * (signed_mu + angular_height));
            light = normalize(light + pole * (side * effective_mu - denominator));
            denominator = dot(light, pole);
        }
    }
    if (abs(denominator) < 1e-6) return 0.0;
    float distance = offset / denominator;
    if (distance <= 0.0 || (!reference_linear && distance >= star_distance)) return 0.0;
    vec3 hit = receiver_rel + light * distance;
    return coverage * ring_shadow_extinction(lut, ring_idx, hit, pole, light, abs(denominator), distance,
        bounds, source_angle, pixel_size, single_box, softness, reference_linear);
}

float ring_receiver_extinction_fast(sampler2DArray lut, int ring_idx, vec3 receiver_rel, vec3 pole,
                                    vec3 to_star, float star_radius, float star_distance, vec2 bounds,
                                    float pixel_size, bool single_box, float softness, bool reference_linear) {
    float denominator = dot(to_star, pole);
    float absolute_denominator = abs(denominator);
    if (reference_linear || absolute_denominator <= max(star_radius, 1e-6*star_distance))
        return ring_receiver_extinction(lut, ring_idx, receiver_rel, pole, to_star/star_distance,
            bounds, star_radius/star_distance, star_distance, pixel_size, single_box, softness, reference_linear);
    float fraction = -dot(receiver_rel, pole)/denominator;
    if (fraction <= 0.0 || fraction >= 1.0) return 0.0;
    vec3 hit = receiver_rel+to_star*fraction;
    float radius = length(hit);
    if (radius <= 1e-8)
        return ring_receiver_extinction(lut, ring_idx, receiver_rel, pole, to_star/star_distance,
            bounds, star_radius/star_distance, star_distance, pixel_size, single_box, softness, reference_linear);
    float radial_projection = dot(hit, to_star)/radius;
    float radial_stretch = length(vec2(denominator, radial_projection))/absolute_denominator;
    float tangential_stretch = length(cross(hit, to_star))/(radius*absolute_denominator);
    float base_width = star_radius*fraction;
    float tangential_width = base_width*tangential_stretch;
    float curvature = tangential_width*(tangential_width/(length(vec2(radius, tangential_width))+radius+1e-20));
    float width = softness*(base_width*radial_stretch+0.5*curvature);
    float span = max(bounds.y-bounds.x, 1e-9);
    float pixel_width = max(0.75*pixel_size*radial_stretch, 0.75*span/4096.0);
    vec3 uv = vec3((radius+0.25*softness*curvature-bounds.x)/span, max(width, pixel_width)/span,
        clamp(absolute_denominator/star_distance, 0.5/64.0, 63.5/64.0));
    return ring_shadow_filter(lut, ring_idx, uv, single_box, false);
}
