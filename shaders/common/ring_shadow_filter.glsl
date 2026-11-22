float ring_integral_texel(sampler2DArray lut, int ring_idx, float i, float mu_coord) {
    float texel = (clamp(i, 0.0, 4096.0) + 0.5) / 4097.0;
    return textureLod(lut, vec3(texel, mu_coord, float(ring_idx)), 0.0).r;
}

float ring_box_extinction(sampler2DArray lut, int ring_idx, float u, float h, float mu_coord) {
    float u0 = clamp(u - h, 0.0, 1.0);
    float u1 = clamp(u + h, 0.0, 1.0);
    float x0 = u0 * 4096.0;
    float x1 = u1 * 4096.0;
    float i0 = floor(x0);
    float i1 = floor(x1);
    float f0 = x0 - i0;
    float f1 = x1 - i1;
    float s0 = ring_integral_texel(lut, ring_idx, i0, mu_coord);
    float s0_next = ring_integral_texel(lut, ring_idx, i0 + 1.0, mu_coord);
    float s1 = ring_integral_texel(lut, ring_idx, i1, mu_coord);
    float s1_next = ring_integral_texel(lut, ring_idx, i1 + 1.0, mu_coord);
    float d0 = max(0.0, s0_next - s0);
    float d1 = max(0.0, s1_next - s1);
    float multi = step(0.5, i1 - i0);
    float span = mix(1.0, (1.0 - f0) + ((i1 - i0) - 1.0) + f1, multi);
    float diff = mix(d0, ((1.0 - f0) * d0 + f1 * d1) + max(0.0, s1 - s0_next), multi);
    float inside = step(0.0, u - h) * step(u + h, 1.0) * step(1e-8, h);
    float coverage = mix(clamp((u1 - u0) / max(2.0 * h, 1e-6), 0.0, 1.0), 1.0, inside);
    return diff * (4096.0 / max(span, 1e-6)) * coverage;
}

vec3 ring_shadow_coordinates(vec3 hit_rel, vec3 pole, vec3 light, float incidence, float distance,
                             vec2 bounds, float source_angle, float pixel_size, float softness) {
    float span = max(bounds.y - bounds.x, 1e-9);
    float radius = length(hit_rel);
    vec3 radial = radius > 1e-12 ? hit_rel / radius : vec3(1, 0, 0);
    vec3 planar = light - pole * dot(light, pole);
    float cosine = length(planar) > 1e-6 ? dot(radial, normalize(planar)) : 0.0;
    float cosine2 = clamp(cosine * cosine, 0.0, 1.0);
    float mu = max(incidence, 1e-6);
    float radial_stretch = sqrt(cosine2 / (mu * mu) + 1.0 - cosine2);
    float base_width = source_angle * distance;
    float width = base_width * radial_stretch;
    float tangential_width = base_width * sqrt((1.0 - cosine2) / (mu * mu) + cosine2);
    float curvature = tangential_width * (tangential_width / (length(vec2(radius, tangential_width)) + radius + 1e-20));
    float shift = 0.25 * curvature * softness;
    width = softness * (width + 0.5 * curvature);
    float pixel_width = max(0.75 * pixel_size * radial_stretch, 0.75 * span / 4096.0);
    float h = max(width, pixel_width) / span;
    float mu_coord = clamp(incidence, 0.5 / 64.0, 63.5 / 64.0);
    return vec3((radius + shift - bounds.x) / span, h, mu_coord);
}

float ring_shadow_filter(sampler2DArray lut, int ring_idx, vec3 uv) {
    if (uv.x + uv.y <= 0.0 || uv.x - uv.y >= 1.0) return 0.0;
    float inner = ring_box_extinction(lut, ring_idx, uv.x, 0.36 * uv.y, uv.z);
    float middle = ring_box_extinction(lut, ring_idx, uv.x, 0.72 * uv.y, uv.z);
    float outer = ring_box_extinction(lut, ring_idx, uv.x, uv.y, uv.z);
    return clamp(dot(vec3(0.08, 0.42, 0.50), vec3(inner, middle, outer)), 0.0, 1.0);
}

float ring_swept_filter(sampler2DArray lut, int ring_idx, vec3 uv, float sweep) {
    float s2 = sweep*sweep, h2 = uv.y*uv.y;
    float reach = sqrt(h2+s2);
    if (uv.x+reach <= 0.0 || uv.x-reach >= 1.0) return 0.0;
    bool wide_sweep = sweep > 3.0*uv.y;
    float outer = ring_box_extinction(lut, ring_idx, uv.x, wide_sweep ? sqrt(0.728096*h2+s2) : reach, uv.z);
    if (wide_sweep) return clamp(outer, 0.0, 1.0);
    float inner = ring_box_extinction(lut, ring_idx, uv.x, sqrt(0.1296*h2+s2), uv.z);
    float middle = ring_box_extinction(lut, ring_idx, uv.x, sqrt(0.5184*h2+s2), uv.z);
    return clamp(dot(vec3(0.08, 0.42, 0.50), vec3(inner, middle, outer)), 0.0, 1.0);
}

vec4 ring_receiver_coordinates(vec3 receiver_rel, vec3 pole, vec3 light, vec2 bounds, float source_angle,
                               float star_distance, float pixel_size, float softness) {
    float offset = -dot(receiver_rel, pole);
    float denominator = dot(light, pole);
    float coverage = 1.0;
    if (source_angle > 1e-8) {
        if (offset == 0.0) return vec4(0.0);
        float side = offset > 0.0 ? 1.0 : -1.0;
        float signed_mu = denominator * side;
        float angular_height = source_angle * sqrt(max(0.0, 1.0 - denominator * denominator));
        if (angular_height > 1e-8) {
            coverage = smoothstep(-angular_height, angular_height, signed_mu);
            if (coverage <= 0.0) return vec4(0.0);
            float effective_mu = max(signed_mu, 0.5 * (signed_mu + angular_height));
            light = normalize(light + pole * (side * effective_mu - denominator));
            denominator = dot(light, pole);
        }
    }
    if (abs(denominator) < 1e-6) return vec4(0.0);
    float distance = offset / denominator;
    if (distance <= 0.0 || distance >= star_distance) return vec4(0.0);
    vec3 hit = receiver_rel + light * distance;
    vec3 uv = ring_shadow_coordinates(hit, pole, light, abs(denominator), distance,
        bounds, source_angle, pixel_size, softness);
    return vec4(uv, coverage);
}

bool ring_receiver_coordinates_fast(vec3 receiver_rel, vec3 pole, vec3 to_star, float star_radius,
                                    float star_distance, vec2 bounds, float pixel_size, float softness, out vec4 coords) {
    coords = vec4(0.0);
    float denominator = dot(to_star, pole);
    float absolute_denominator = abs(denominator);
    if (absolute_denominator <= max(star_radius, 1e-6*star_distance)) return false;
    float fraction = -dot(receiver_rel, pole)/denominator;
    if (fraction <= 0.0 || fraction >= 1.0) return true;
    vec3 hit = receiver_rel+to_star*fraction;
    float radius = length(hit);
    if (radius <= 1e-8) return false;
    float radial_projection = dot(hit, to_star)/radius;
    float radial_stretch = length(vec2(denominator, radial_projection))/absolute_denominator;
    float tangential_stretch = length(cross(hit, to_star))/(radius*absolute_denominator);
    float base_width = star_radius*fraction;
    float tangential_width = base_width*tangential_stretch;
    float curvature = tangential_width*(tangential_width/(length(vec2(radius, tangential_width))+radius+1e-20));
    float width = softness*(base_width*radial_stretch+0.5*curvature);
    float span = max(bounds.y-bounds.x, 1e-9);
    float pixel_width = max(0.75*pixel_size*radial_stretch, 0.75*span/4096.0);
    coords = vec4((radius+0.25*softness*curvature-bounds.x)/span, max(width, pixel_width)/span,
        clamp(absolute_denominator/star_distance, 0.5/64.0, 63.5/64.0), 1.0);
    return true;
}
