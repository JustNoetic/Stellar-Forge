const float CLOUD_DEPTH_BIAS = 1.0000005;

vec4 cloud_sheet_color(uint body, vec3 ray, vec3 local, float distance, vec3 pole, float flattening, mat3 rotation) {
    SurfaceFluidGPU f = fluids[body];
    vec3 normal = normalize(local+pole*dot(local, pole)*(flattening*(2.0+flattening)));
    float cosine = max(abs(dot(normal, ray)), 0.04);
    float pixel_angle = 1.0/max(u_scene.screen_params.y*u_scene.screen_params.z, 1.0);
    float footprint = distance*FLUID_AU_KM*pixel_angle/cosine;
    CloudWeather weather = cloud_weather(body, normalize(transpose(rotation)*local), footprint);
    float opacity = 1.0-pow(max(1.0-weather.coverage, 0.0), 1.0/cosine);
    opacity *= 1.0-exp(-f.cloud_shape.w*f.cloud_shape.z);
    if (opacity < 1e-3) return vec4(0.0);

    bool atmosphere = surface_atmospheres[body].w > 0.5;
    SurfaceAtmosphereSample lookup;
    if (atmosphere) lookup = surface_atmosphere_sample(body, local, vec3(1.0, 0.0, 0.0), normal);
    vec3 radiance = vec3(0.0);
    bool underside = dot(normal, ray) > 0.0;
    for (uint s = 0u; s < min(u_scene.num_stars, CLOUD_MAX_STARS); ++s) {
        if (!star_lights_body(body, s)) continue;
        vec3 to_star = u_scene.stars_pos_radius[s].xyz-ray*distance;
        float d_au = max(length(to_star), 1e-5);
        vec3 light = to_star/d_au;
        float irradiance = u_scene.hdr_enabled != 0u ? u_scene.stars_color_lum[s].w/(d_au*d_au) : 1.0;
        vec3 transmission, sky = vec3(0.0);
        if (atmosphere) {
            vec3 uv = surface_atmosphere_coordinates(lookup, light);
            transmission = textureLod(surface_atmosphere_lut, uv, 0.0).rgb;
            sky = textureLod(surface_atmosphere_lut, uv+vec3(0, 0, 1), 0.0).rgb;
        } else {
            vec3 spherical = fluid_spherical(local, pole, flattening);
            float radius = length(spherical)*FLUID_AU_KM;
            float horizon = -sqrt(max(1.0-pow(f.cloud_motion.w/radius, 2.0), 0.0));
            float mu = dot(normalize(spherical), normalize(fluid_spherical(light, pole, flattening)));
            transmission = vec3(smoothstep(horizon-0.005, horizon+0.005, mu));
        }
        float diffuse = 0.15+0.85*max(dot(normal, light), 0.0);
        if (underside) diffuse *= 0.25;
        radiance += u_scene.stars_color_lum[s].rgb*irradiance*(transmission*diffuse+sky*0.5);
    }
    float exposure = u_scene.hdr_enabled != 0u ? u_scene.screen_params.w : 1.0;
    return vec4(radiance*f.cloud_color.rgb*(opacity*exposure), opacity);
}

float cloud_sheet_depth(BodyInstanceGPU b, vec3 local) {
    return min(body_offset_depth(b, local)*CLOUD_DEPTH_BIAS, 1.0);
}

void render_cloud_sheet() {
    BodyInstanceGPU b = bodies[f_body_idx];
    SurfaceFluidGPU f = fluids[f_body_idx];
    vec3 pole = body_pole(b);
    float flattening = fluid_flattening(b);
    vec3 ray = normalize(f_world_pos);
    float anchor_distance = length(f_world_pos);
    vec3 origin = fluid_spherical(f_body_offset, pole, flattening);
    vec3 direction = fluid_spherical(ray, pole, flattening);
    float radius = (f.cloud_motion.w+f.cloud_shape.y+0.5*f.cloud_shape.z)/FLUID_AU_KM;
    vec2 roots, ground;
    if (!fluid_sphere_roots(origin, direction, radius, roots) || anchor_distance+roots.y <= 1e-12) discard;
    float ground_offset = 1e30;
    if (fluid_sphere_roots(origin, direction, f.cloud_motion.w/FLUID_AU_KM, ground)) {
        vec3 camera = fluid_spherical(-b.pos_radius.xyz, pole, flattening);
        if (anchor_distance+ground.x > 0.0) ground_offset = ground.x;
        else if (anchor_distance+ground.y > 0.0 && dot(camera, camera) < pow(f.cloud_motion.w/FLUID_AU_KM, 2.0)) discard;
    }
    mat3 rotation = body_surface_rotation(pole, b.spin_flags.x);
    vec4 combined = vec4(0.0);
    float nearest_depth = 0.0;
    for (int i = 1; i >= 0; --i) {
        float distance = anchor_distance+roots[i];
        if (distance <= 1e-12 || roots[i] >= ground_offset || (i == 0 && roots.y-roots.x < 1e-12)) continue;
        vec3 local = f_body_offset+ray*roots[i];
        vec4 color = cloud_sheet_color(f_body_idx, ray, local, distance, pole, flattening, rotation);
        if (color.a < 1e-3) continue;
        float depth = cloud_sheet_depth(b, local);
#ifdef TRANSPARENCY_CAPTURE
        if (capture_transparency(color, depth)) color = vec4(0.0);
#endif
        combined = color+(1.0-color.a)*combined;
        nearest_depth = depth;
    }
#ifdef TRANSPARENCY_CAPTURE
    gl_FragDepth = nearest_depth;
    out_color = combined;
#else
    if (combined.a < 1e-3) discard;
    gl_FragDepth = nearest_depth;
#ifdef CLOUD_DEPTH_ONLY
    if (combined.a < 0.999) discard;
    out_color = vec4(0.0);
#else
    out_color = combined;
#endif
#endif
}
