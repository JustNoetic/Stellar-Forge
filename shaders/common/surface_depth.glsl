float curved_surface_depth(BodyInstanceGPU b, vec3 world_pos, vec3 base_chord, float depth) {
    vec3 pole = body_pole(b);
    float flattening = 1.0/max(1.0-b.pole_oblateness.w, 0.05)-1.0;
    vec3 origin = -b.pos_radius.xyz/b.pos_radius.w;
    vec3 ray = normalize(world_pos);
    vec3 point = (world_pos-b.pos_radius.xyz)/b.pos_radius.w;
    origin += pole*dot(origin, pole)*flattening;
    point += pole*dot(point, pole)*flattening;
    vec3 scaled_ray = ray+pole*dot(ray, pole)*flattening;
    float curvature = max(0.0, 1.0-length(base_chord));
    float surface_radius = length(point)+curvature;
    float a = dot(scaled_ray, scaled_ray);
    vec3 perpendicular = cross(origin, scaled_ray);
    float discriminant = a*surface_radius*surface_radius-dot(perpendicular, perpendicular);
    if (curvature > 1e-7 && discriminant >= 0.0 && dot(point, scaled_ray) < 0.0) {
        float distance = (-dot(origin, scaled_ray)-sqrt(discriminant))/a;
        if (distance > 0.0) {
            vec4 projected = u_scene.view_proj*vec4(ray*(distance*b.pos_radius.w), 1.0);
            return max(depth, clamp(projected.z/projected.w, 0.0, 1.0));
        }
    }
    return depth;
}
