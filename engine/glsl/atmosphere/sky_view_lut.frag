#version 460 core

in vec2 f_uv;

uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_atmo_tint;
uniform float u_planet_radius;
uniform float u_atmo_radius;
uniform float u_scale_height;
uniform float u_density;

out vec4 out_color;

const float PI = 3.14159265358979323846;

// Ray-sphere intersection returning (near, far) distances
vec2 ray_sphere_intersect(vec3 origin, vec3 dir, float radius) {
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float delta = b * b - c;
    if (delta < 0.0) return vec2(-1.0, -1.0);
    float sq = sqrt(delta);
    return vec2(-b - sq, -b + sq);
}

void main() {
    float D = length(u_cam_pos);
    vec3 u_hat = u_cam_pos / max(D, 1e-6);
    vec3 C_dir = -u_hat; // Vector pointing towards planet center

    // Build orthonormal basis perpendicular to C_dir
    vec3 ref = (abs(C_dir.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(C_dir, ref));
    vec3 up = cross(C_dir, right);

    // Horizon angle (planet ground edge) and atmosphere top angle
    float sin_horizon = clamp(u_planet_radius / D, 0.0, 1.0);
    float theta_horizon = asin(sin_horizon);

    bool cam_inside_atmo = (D < u_atmo_radius);
    float max_theta = cam_inside_atmo ? PI : asin(clamp(u_atmo_radius / D, 0.0, 1.0));

    // Map f_uv to view ray direction V:
    // f_uv.x in [0, 1] -> azimuth phi in [0, 2*PI]
    // f_uv.y in [0, 1] -> piecewise elevation angle theta
    float phi = f_uv.x * 2.0 * PI;
    vec3 perp = cos(phi) * right + sin(phi) * up;

    float theta = 0.0;
    bool hits_ground = false;

    if (f_uv.y <= 0.5) {
        // Ground disk: [0, theta_horizon]
        theta = (f_uv.y / 0.5) * theta_horizon;
        hits_ground = true;
    } else {
        // Atmosphere limb or sky dome: [theta_horizon, max_theta]
        float t = (f_uv.y - 0.5) / 0.5;
        theta = theta_horizon + t * (max_theta - theta_horizon);
        hits_ground = false;
    }

    vec3 V = normalize(cos(theta) * C_dir + sin(theta) * perp);

    // Calculate ray march bounds through atmosphere
    vec2 t_atmo = ray_sphere_intersect(u_cam_pos, V, u_atmo_radius);
    if (t_atmo.y < 0.0) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float s_start = max(0.0, t_atmo.x);
    float s_end = t_atmo.y;

    if (hits_ground) {
        vec2 t_planet = ray_sphere_intersect(u_cam_pos, V, u_planet_radius);
        if (t_planet.x > 0.0) {
            s_end = min(s_end, t_planet.x);
        }
    }

    if (s_start >= s_end) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Numerical integration along the ray (16 steps)
    const int NUM_STEPS = 16;
    float ds = (s_end - s_start) / float(NUM_STEPS);

    vec3 L = normalize(u_sun_dir);
    vec3 total_scatter = vec3(0.0);
    float total_transmittance = 1.0;

    float sigma_s = u_density;
    float sigma_t = u_density * 1.1; // extinction slightly higher for scattering balance

    for (int i = 0; i < NUM_STEPS; i++) {
        float s = s_start + (float(i) + 0.5) * ds;
        vec3 P = u_cam_pos + s * V;
        float r = length(P);
        float h = max(0.0, r - u_planet_radius);

        float density = exp(-h / max(u_scale_height, 1e-4));

        // Sunlight reaching point P: test planet occlusion
        vec2 t_planet_sun = ray_sphere_intersect(P, L, u_planet_radius);
        float sun_vis = 1.0;
        if (t_planet_sun.x > 0.0 && t_planet_sun.y > t_planet_sun.x) {
            sun_vis = 0.0; // In planet shadow
        }

        // Sunlight optical depth to top of atmosphere (4 steps)
        float tau_sun = 0.0;
        if (sun_vis > 0.0) {
            vec2 t_atmo_sun = ray_sphere_intersect(P, L, u_atmo_radius);
            float dt_sun = max(0.0, t_atmo_sun.y) / 4.0;
            for (int k = 0; k < 4; k++) {
                vec3 P_sun = P + (float(k) + 0.5) * dt_sun * L;
                float h_sun = max(0.0, length(P_sun) - u_planet_radius);
                tau_sun += exp(-h_sun / max(u_scale_height, 1e-4)) * dt_sun;
            }
        }
        float trans_to_sun = sun_vis * exp(-sigma_t * tau_sun);

        // Scattered light at this step
        vec3 in_scatter = u_sun_color * u_atmo_tint * (density * sigma_s) * trans_to_sun * ds;
        float step_trans = exp(-density * sigma_t * ds);

        total_scatter += total_transmittance * in_scatter;
        total_transmittance *= step_trans;
    }

    out_color = vec4(total_scatter, clamp(total_transmittance, 0.0, 1.0));
}
