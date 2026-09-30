// Select high-quality lighting on finite terrain rays that can intersect a
// shadow. Keep sharp eclipse/ring edges out of the coarse unshadowed volume.
// All geometry here is Cartesian world km, including on oblate planets.
uniform sampler2D u_ring_opacity_bounds;
uniform bool u_ring_opacity_bounds_enabled;
uniform int u_aerial_shadow_steps;

bool aerialRingMayOcclude(int ring, float u0, float u1) {
    if (!u_ring_opacity_bounds_enabled) return true;
    int width = textureSize(u_ring_opacity_bounds, 0).x / 2;
    // Include both taps of the production linear alpha sampler. Two aligned
    // max-tree nodes conservatively cover any interval, including narrow gaps.
    int lo = clamp(int(floor(u0 * float(width) - 0.5)), 0, width - 1);
    int hi = clamp(int(ceil(u1 * float(width) - 0.5)), lo, width - 1);
    int level = int(ceil(log2(float(hi - lo + 1))));
    int offset = width >> level;
    return max(texelFetch(u_ring_opacity_bounds, ivec2(offset + (lo >> level), ring), 0).r,
               texelFetch(u_ring_opacity_bounds, ivec2(offset + (hi >> level), ring), 0).r) > 0.0;
}

bool aerialClipCylinder(vec3 A, vec3 B, float radius, inout vec2 span) {
    float qa = dot(B, B);
    float qb = dot(A, B);
    float qc = dot(A, A) - radius * radius;
    if (qa < 1e-12) return qc <= 0.0;
    float det = qb * qb - qa * qc;
    if (det < 0.0) return false;
    vec2 hit = vec2(-qb - sqrt(det), -qb + sqrt(det)) / qa;
    span = vec2(max(span.x, hit.x), min(span.y, hit.y));
    return span.y > span.x;
}

void aerialIncludeShadow(vec2 span, inout vec2 bounds, inout float minimum_width) {
    if (span.y <= span.x) return;
    bounds = vec2(min(bounds.x, span.x), max(bounds.y, span.y));
    minimum_width = min(minimum_width, span.y - span.x);
}

bool aerialClipLightDistance(float a, float b, float limit, inout vec2 span) {
    // Restrict a + b*s to [0, limit]: a caster/ring must be between air and star.
    if (abs(b) < 1e-10) return a >= 0.0 && a <= limit;
    float s0 = -a / b;
    float s1 = (limit - a) / b;
    span = vec2(max(span.x, min(s0, s1)), min(span.y, max(s0, s1)));
    return span.y > span.x;
}

float aerialShadowFeather(float distance_to_bound, float footprint) {
    // Every possibly shadowed ray gets exact per-pixel lighting. The blend is
    // wholly outside the conservative penumbra, where both paths are unshadowed.
    return 1.0 - smoothstep(0.0, max(0.001, footprint), distance_to_bound);
}

float aerialShadowRefinement(vec3 O, vec3 V, float s_start, float s_end, float footprint,
                            out vec2 shadow_span, out float minimum_shadow_width) {
    float refinement = 0.0;
    shadow_span = vec2(s_end, s_start);
    minimum_shadow_width = s_end - s_start;
    for (int st = 0; st < clamp(u_num_stars, 1, 4); st++) {
        vec3 star = u_star_pos_local[st].xyz;
        float star_distance = length(star);
        if (star_distance < 1e-6) continue;
        vec3 L = star / star_distance;
        float star_sin = u_star_solstice[st].w / max(u_star_solstice[st].z, 1e-6);
        float maximum_light_distance = star_distance + length(O) + s_end;

        for (int c = 0; c < u_num_active_casters; c++) {
            vec3 C = (u_active_casters[c].xyz - u_body_offset) * u_au_to_km;
            vec2 span = vec2(s_start, s_end);
            vec3 D = C - O;
            float a = dot(D, L), b = -dot(V, L);
            if (!aerialClipLightDistance(a, b, maximum_light_distance, span)) continue;
            vec3 A = D - a * L;
            vec3 B = -V - b * L;
            float closest = dot(B, B) > 1e-12
                ? clamp(-dot(A, B) / dot(B, B), span.x, span.y) : span.x;
            float radial_distance = length(A + closest * B);
            float max_distance = max(length(D - span.x * V), length(D - span.y * V));
            // Equatorial radius bounds any projected oblate caster. Include
            // atmospheric lens/penumbra support just as the high-quality culler.
            float radius = (u_active_casters[c].w + 4.0 * max(0.0, u_active_caster_atmos[c].w)) * u_au_to_km;
            float penumbra = max_distance * max(0.0, star_sin);
            refinement = max(refinement, aerialShadowFeather(radial_distance - radius - penumbra, footprint));
            if (aerialClipCylinder(A, B, radius + penumbra, span))
                aerialIncludeShadow(span, shadow_span, minimum_shadow_width);
        }

        for (int k = 0; k < u_num_ring_planes; k++) {
            if ((u_ring_mask & (1u << k)) == 0u || u_ring_params[k].z <= 0.0) continue;
            vec3 N = u_ring_normal[k];
            float denominator = dot(L, N);
            if (abs(denominator) < 1e-8) continue;
            vec3 C = (u_ring_center[k] - u_body_offset) * u_au_to_km;
            float a = dot(C - O, N) / denominator;
            float b = -dot(V, N) / denominator;
            vec2 span = vec2(s_start, s_end);
            if (!aerialClipLightDistance(a, b, maximum_light_distance, span)) continue;
            vec3 A = O + a * L - C;
            vec3 B = V + b * L;
            float closest = dot(B, B) > 1e-12
                ? clamp(-dot(A, B) / dot(B, B), span.x, span.y) : span.x;
            float radial_min = length(A + closest * B);
            float radial_max = max(length(A + span.x * B), length(A + span.y * B));
            // Conservative major axis of the stellar disc projected into the
            // ring plane; covers glancing light and finite-source penumbrae.
            float penumbra = max(a + b * span.x, a + b * span.y)
                * max(0.0, star_sin) / abs(denominator);
            float inner = u_ring_params[k].x * u_au_to_km;
            float outer = u_ring_params[k].y * u_au_to_km;
            float distance_to_bound = max(radial_min - outer - penumbra, inner - penumbra - radial_max);
            float feather = footprint / abs(denominator);
            float selection = aerialShadowFeather(distance_to_bound, feather);
            if (selection <= 0.0) continue;
            float inv_width = 1.0 / max(1e-6, outer - inner);
            if (!aerialRingMayOcclude(k, (radial_min - penumbra - feather - inner) * inv_width,
                                        (radial_max + penumbra + feather - inner) * inv_width)) continue;
            refinement = max(refinement, selection);
            // Bound the high-density quadrature by the projected outer annulus.
            // Keep a crossing inner hole inside the interval to avoid changing
            // sampling phase on either side of it. Trim only empty ends.
            if (aerialClipCylinder(A, B, outer + penumbra, span)) {
                vec2 hole = span;
                if (inner > penumbra && aerialClipCylinder(A, B, inner - penumbra, hole)) {
                    if (hole.x <= span.x) span.x = hole.y;
                    else if (hole.y >= span.y) span.y = hole.x;
                }
                aerialIncludeShadow(span, shadow_span, minimum_shadow_width);
            }
        }
    }
    return refinement;
}
