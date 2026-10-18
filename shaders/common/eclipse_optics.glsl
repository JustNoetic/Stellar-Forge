float oblate_caster_radius(float r_eq, float r_polar, vec3 pole, vec3 L, vec3 perp_vec) {
    if (r_polar < 1e-9 || r_eq < 1e-9) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-9) return r_eq;
    vec3 p_proj = pole - L * dot(pole, L);
    float p_proj_len = length(p_proj);
    if (p_proj_len < 1e-6) return r_eq;
    vec3 p_dir = p_proj / p_proj_len;
    float y = dot(perp_vec, p_dir);
    float x = length(perp_vec - y * p_dir);
    float cosine = clamp(dot(pole, L), -1.0, 1.0);
    float minor = sqrt(r_polar * r_polar + (r_eq * r_eq - r_polar * r_polar) * cosine * cosine);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / minor) * (y / minor));
    return denom > 1e-9 ? (perp_len / denom) : r_eq;
}

float eclipse_reference_visibility(float caster_radius, float projected_star_radius, float separation) {
    float inner = abs(caster_radius - projected_star_radius);
    float outer = caster_radius + projected_star_radius;
    if (separation >= outer) return 1.0;
    float t = clamp((separation - inner) / max(outer - inner, 1e-9), 0.0, 1.0);
    float visibility = pow(smoothstep(0.0, 1.0, t), 0.625);
    if (caster_radius < projected_star_radius) {
        float radius_ratio = caster_radius / projected_star_radius;
        visibility = 1.0 - radius_ratio * radius_ratio * (1.0 - visibility);
    }
    return visibility;
}

float eclipse_segment_area(float radius, float chord, float height) {
    float sine = height / radius;
    if (chord > 0.0 && sine < 0.1) {
        float s2 = sine * sine;
        return radius * radius * sine * s2 * (2.0 / 3.0 + s2 * (0.2 + s2 * (3.0 / 28.0 + s2 * (5.0 / 72.0))));
    }
    return radius * radius * atan(height, chord) - chord * height;
}

float eclipse_disc_visibility(float caster_radius, float projected_star_radius, float separation) {
    if (caster_radius <= 0.0) return 1.0;
    if (separation >= caster_radius + projected_star_radius) return 1.0;
    if (projected_star_radius <= 0.0) return 0.0;
    if (u_scene.eclipse_params.x != 0u)
        return eclipse_reference_visibility(caster_radius, projected_star_radius, separation);
    float r = caster_radius / projected_star_radius;
    float d = separation / projected_star_radius;
    if (d <= abs(r - 1.0)) return max(0.0, 1.0 - min(r * r, 1.0));
    float chord = clamp(((d - r) * (d + r) + 1.0) / (2.0 * d), -1.0, 1.0);
    float height = sqrt(max(0.0, (1.0 - chord) * (1.0 + chord)));
    float area = eclipse_segment_area(1.0, chord, height) + eclipse_segment_area(r, d - chord, height);
    return clamp(1.0 - area / 3.141592653589793, 0.0, 1.0);
}

float eclipse_ellipse_slice(float x, vec2 axes, vec2 center) {
    float dx = abs(x - center.x);
    float h = axes.y / axes.x * sqrt(max(0.0, (axes.x - dx) * (axes.x + dx)));
    float source_h = sqrt(max(0.0, (1.0 - x) * (1.0 + x)));
    return max(0.0, min(source_h, center.y + h) - max(-source_h, center.y - h));
}

float eclipse_ellipse_visibility(vec2 axes, vec2 center, float source_radius) {
    float distance = length(center);
    if (distance >= axes.x + source_radius) return 1.0;
    float normalized_distance = length(center / axes);
    float source_bound = source_radius / axes.y;
    if (normalized_distance + source_bound <= 1.0) return 0.0;
    if (normalized_distance - source_bound >= 1.0) return 1.0;
    if (distance + axes.x <= source_radius)
        return clamp(1.0 - (axes.x / source_radius) * (axes.y / source_radius), 0.0, 1.0);
    axes /= source_radius;
    center /= source_radius;
    float lo = max(-1.0, center.x - axes.x);
    float hi = min(1.0, center.x + axes.x);
    if (lo >= hi) return 1.0;
    const vec2 quadrature[16] = vec2[](
        vec2(0.9972638618, 0.0070186100), vec2(0.9856115115, 0.0162743947),
        vec2(0.9647622556, 0.0253920653), vec2(0.9349060759, 0.0342738629),
        vec2(0.8963211558, 0.0428358980), vec2(0.8493676137, 0.0509980593),
        vec2(0.7944837960, 0.0586840935), vec2(0.7321821187, 0.0658222228),
        vec2(0.6630442669, 0.0723457941), vec2(0.5877157572, 0.0781938958),
        vec2(0.5068999089, 0.0833119242), vec2(0.4213512761, 0.0876520930),
        vec2(0.3318686023, 0.0911738787), vec2(0.2392873623, 0.0938443991),
        vec2(0.1444719616, 0.0956387201), vec2(0.0483076657, 0.0965400885));
    float midpoint = 0.5 * (lo + hi);
    float half_width = 0.5 * (hi - lo);
    float area = 0.0;
    for (int i = 0; i < 16; ++i) {
        float offset = half_width * quadrature[i].x;
        area += quadrature[i].y * (eclipse_ellipse_slice(midpoint - offset, axes, center)
                                 + eclipse_ellipse_slice(midpoint + offset, axes, center));
    }
    return clamp(1.0 - half_width * area / 3.141592653589793, 0.0, 1.0);
}

float eclipse_caster_visibility(float equatorial, float polar, vec3 pole, vec3 L,
                                vec3 perpendicular, float source_radius) {
    if (equatorial - polar <= equatorial * 1e-5)
        return eclipse_disc_visibility(equatorial, source_radius, length(perpendicular));
    float cosine = clamp(dot(pole, L), -1.0, 1.0);
    float minor = sqrt(polar * polar + (equatorial * equatorial - polar * polar) * cosine * cosine);
    vec3 projected_pole = pole - L * cosine;
    float pole_length = length(projected_pole);
    if (pole_length < 1e-6 || equatorial - minor <= equatorial * 1e-5)
        return eclipse_disc_visibility(equatorial, source_radius, length(perpendicular));
    if (u_scene.eclipse_params.x != 0u) {
        float radius = oblate_caster_radius(equatorial, polar, pole, L, perpendicular);
        return eclipse_disc_visibility(radius, source_radius, length(perpendicular));
    }
    vec3 minor_direction = projected_pole / pole_length;
    float y = dot(perpendicular, minor_direction);
    float x = length(perpendicular - y * minor_direction);
    return eclipse_ellipse_visibility(vec2(equatorial, minor), vec2(x, y), source_radius);
}
