#version 460

layout(location = 0) in vec3 f_world_pos;
layout(location = 1) flat in uint f_body_idx;
layout(location = 2) in vec3 f_body_offset;
layout(location = 3) in vec3 f_refraction_shift;

#include "common/celestial_scene.glsl"
#include "common/body_surface.glsl"
#include "common/surface_atmosphere.glsl"
#include "common/fluid_surface.glsl"
#ifdef TRANSPARENCY_CAPTURE
#include "common/transparency.glsl"
#endif

layout(location = 0) out vec4 out_color;

const uint CLOUD_MAX_STARS = 4u;
const float CLOUD_DEPTH_BIAS = 1.0000005;

float cloud_hg(float g, float c) {
    float g2 = g*g;
    return (1.0-g2)/(4.0*FLUID_PI*pow(max(1.0+g2-2.0*g*c, 1e-4), 1.5));
}

float cloud_multiple_scattering(float optical_depth) {
    return 1.0/(1.0+0.35*optical_depth+0.065*optical_depth*optical_depth);
}

void main() {
    BodyInstanceGPU b = bodies[f_body_idx];
    SurfaceFluidGPU f = fluids[f_body_idx];
    vec3 pole = body_pole(b);
    float flattening = fluid_flattening(b);
    mat3 rotation = body_surface_rotation(pole, b.spin_flags.x);
    vec3 ray = normalize(f_world_pos);
    float anchor_distance = length(f_world_pos);
    vec3 origin = fluid_spherical(f_body_offset, pole, flattening);
    vec3 scaled_ray = fluid_spherical(ray, pole, flattening);
    float thickness_km = f.cloud_shape.z;
    float bottom_km = f.cloud_motion.w+f.cloud_shape.y;
    float r_ground = f.cloud_motion.w/FLUID_AU_KM;
    float r_bottom = bottom_km/FLUID_AU_KM;
    float r_top = (bottom_km+thickness_km)/FLUID_AU_KM;

    vec2 top, bottom, ground;
    if (!fluid_sphere_roots(origin, scaled_ray, r_top, top) || anchor_distance+top.y <= 0.0) discard;
    float t0 = max(top.x, -anchor_distance), t1 = top.y;
    if (fluid_sphere_roots(origin, scaled_ray, r_bottom, bottom) && anchor_distance+bottom.y > 0.0) {
        if (bottom.x > t0) t1 = bottom.x;
        else t0 = max(bottom.y, t0);
    }
    if (fluid_sphere_roots(origin, scaled_ray, r_ground, ground) && anchor_distance+ground.x > 0.0) t1 = min(t1, ground.x);
    if (t1 <= t0) discard;

    float pixel_angle = 1.0/max(u_scene.screen_params.y*u_scene.screen_params.z, 1.0);
    float segment_km = (t1-t0)*FLUID_AU_KM;
    int steps = int(clamp(segment_km/max(thickness_km/8.0, 0.1), 8.0, 48.0));
#ifdef CLOUD_DEPTH_ONLY
    steps = min(steps, 8);
#endif
    float dt = (t1-t0)/float(steps);
    float dt_km = dt*FLUID_AU_KM;
    vec3 mid = f_body_offset+ray*(0.5*(t0+t1));
    vec3 mid_up = normalize(fluid_spherical(mid, pole, flattening));
    float step_footprint = dt_km*sqrt(max(dot(scaled_ray, scaled_ray)-pow(dot(mid_up, scaled_ray), 2.0), 0.0))*0.5;
    CloudWeather previous_weather = cloud_weather(f_body_idx, normalize(transpose(rotation)*(f_body_offset+ray*t0)),
        max((anchor_distance+t0)*FLUID_AU_KM*pixel_angle, step_footprint));

#ifdef CLOUD_DEPTH_ONLY
    float view_transmittance = 1.0, depth_weighted_t = 0.0, depth_total_weight = 0.0;
    for (int i = 0; i < steps; ++i) {
        float t = t0+(float(i)+0.5)*dt;
        vec3 rel = f_body_offset+ray*t;
        vec3 spherical = fluid_spherical(rel, pole, flattening);
        float r_km = length(spherical)*FLUID_AU_KM;
        float height = (r_km-bottom_km)/thickness_km;
        float height_span = dot(normalize(spherical), scaled_ray)*dt_km/thickness_km;
        float next_t = t0+float(i+1)*dt;
        CloudWeather next_weather = cloud_weather(f_body_idx, normalize(transpose(rotation)*(f_body_offset+ray*next_t)),
            max((anchor_distance+next_t)*FLUID_AU_KM*pixel_angle, step_footprint));
        CloudWeather weather;
        weather.coverage = 0.5*(previous_weather.coverage+next_weather.coverage);
        float top_change = next_weather.coverage-previous_weather.coverage;
        previous_weather = next_weather;
        float extinction = cloud_profile_average(height, height_span, weather.coverage, top_change)*mix(0.4, 1.0, weather.coverage)*f.cloud_shape.w;
        float step_transmittance = exp(-extinction*dt_km);
        float weight = view_transmittance*(1.0-step_transmittance);
        depth_weighted_t += weight*t;
        depth_total_weight += weight;
        view_transmittance *= step_transmittance;
        if (view_transmittance < 0.001) break;
    }
    if (view_transmittance >= 0.001) discard;
    gl_FragDepth = min(body_offset_depth(b, f_refraction_shift+f_body_offset+ray*(depth_weighted_t/max(depth_total_weight, 1e-6)))*CLOUD_DEPTH_BIAS, 1.0);
    out_color = vec4(0.0);
    return;
#endif
    gl_FragDepth = min(body_offset_depth(b, f_refraction_shift+f_body_offset+ray*t0)*CLOUD_DEPTH_BIAS, 1.0);

    bool atmosphere = surface_atmospheres[f_body_idx].w > 0.5;
    SurfaceAtmosphereSample lookup;
    if (atmosphere) lookup = surface_atmosphere_sample(f_body_idx, mid, vec3(1.0, 0.0, 0.0), mid_up);

    uint star_count = min(u_scene.num_stars, CLOUD_MAX_STARS);
    vec3 star_dirs[CLOUD_MAX_STARS];
    vec3 star_direct[CLOUD_MAX_STARS];
    vec3 ambient = vec3(0.0);
    for (uint s = 0u; s < CLOUD_MAX_STARS; ++s) {
        star_dirs[s] = vec3(0.0, 1.0, 0.0);
        star_direct[s] = vec3(0.0);
        if (s >= star_count || !star_lights_body(f_body_idx, s)) continue;
        vec3 to_star = u_scene.stars_pos_radius[s].xyz-ray*(anchor_distance+0.5*(t0+t1));
        float d_au = max(length(to_star), 1e-5);
        vec3 L = to_star/d_au;
        float irradiance = (u_scene.hdr_enabled != 0u) ? u_scene.stars_color_lum[s].w/(d_au*d_au) : 1.0;
        vec3 light = u_scene.stars_color_lum[s].rgb*irradiance;
        vec3 transmission, sky = vec3(0.0);
        if (atmosphere) {
            vec3 uv = surface_atmosphere_coordinates(lookup, L);
            transmission = textureLod(surface_atmosphere_lut, uv, 0.0).rgb;
            sky = textureLod(surface_atmosphere_lut, uv+vec3(0, 0, 1), 0.0).rgb;
        } else {
            transmission = vec3(smoothstep(-0.02, 0.03, dot(mid_up, L)));
        }
        float phase = 4.0*FLUID_PI*(0.25*cloud_hg(0.75, dot(ray, L))+0.60*cloud_hg(0.10, dot(ray, L))
                                     +0.15*cloud_hg(-0.30, dot(ray, L)));
        star_dirs[s] = L;
        star_direct[s] = light*transmission*phase;
        ambient += light*(sky+0.015*transmission);
    }

    vec3 radiance = vec3(0.0);
    float transmittance = 1.0;
    float weighted_t = 0.0, total_weight = 0.0;
    for (int i = 0; i < steps; ++i) {
        float t = t0+(float(i)+0.5)*dt;
        vec3 rel = f_body_offset+ray*t;
        vec3 spherical = fluid_spherical(rel, pole, flattening);
        float r_km = length(spherical)*FLUID_AU_KM;
        float height = (r_km-bottom_km)/thickness_km;
        float height_span = dot(normalize(spherical), scaled_ray)*dt_km/thickness_km;
        vec3 fixed_dir = normalize(transpose(rotation)*rel);
        float next_t = t0+float(i+1)*dt;
        float footprint = max((anchor_distance+next_t)*FLUID_AU_KM*pixel_angle, step_footprint);
        CloudWeather next_weather = cloud_weather(f_body_idx, normalize(transpose(rotation)*(f_body_offset+ray*next_t)), footprint);
        CloudWeather weather;
        weather.coverage = 0.5*(previous_weather.coverage+next_weather.coverage);
        float top_change = next_weather.coverage-previous_weather.coverage;
        previous_weather = next_weather;
        if (weather.coverage <= 0.0) continue;
        float extinction = cloud_density_average(f_body_idx, fixed_dir*r_km, height, height_span, weather, top_change, footprint)*f.cloud_shape.w;
        if (extinction <= 0.0) continue;

        float local_height = clamp(height/max(weather.coverage, 0.01), 0.0, 1.0);
        vec3 in_scatter = ambient*(0.45+0.55*local_height);
        for (uint s = 0u; s < star_count; ++s) {
            if (all(equal(star_direct[s], vec3(0.0)))) continue;
            vec3 L = star_dirs[s];
            vec3 up = spherical*(FLUID_AU_KM/max(r_km, 1e-3));
            float path_km = min(cloud_column_above(height, weather.coverage)*thickness_km/max(dot(up, L), 0.15), 3.0*thickness_km);
            float optical_depth = path_km*f.cloud_shape.w*mix(0.4, 1.0, weather.coverage);
            in_scatter += star_direct[s]*cloud_multiple_scattering(optical_depth);
        }
        in_scatter *= f.cloud_color.rgb;
        float step_transmittance = exp(-extinction*dt_km);
        float weight = transmittance*(1.0-step_transmittance);
        radiance += in_scatter*weight;
        weighted_t += weight*t;
        total_weight += weight;
        transmittance *= step_transmittance;
        if (transmittance < 0.01) break;
    }

    float alpha = 1.0-transmittance;
    if (alpha < 1e-3) discard;
    float exposure = (u_scene.hdr_enabled != 0u) ? u_scene.screen_params.w : 1.0;
    out_color = vec4(radiance*exposure, alpha);
#ifdef TRANSPARENCY_CAPTURE
    float cloud_depth = min(body_offset_depth(b, f_refraction_shift+f_body_offset+ray*(weighted_t/max(total_weight, 1e-6)))*CLOUD_DEPTH_BIAS, 1.0);
    if (capture_transparency(out_color, cloud_depth)) out_color = vec4(0.0);
#endif
}
