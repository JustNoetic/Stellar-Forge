#version 460 core
#define MAX_CASTERS 64
#define MAX_STARS 16

in vec3 f_world_pos;
in vec3 f_normal;
in vec3 f_local_pos;
in float f_clip_z;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS];
    vec4 u_stars_colors[MAX_STARS];
    vec4 u_stars_poles_obl[MAX_STARS];
    vec4 u_stars_pole_colors[MAX_STARS];
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[MAX_CASTERS];
    vec4 u_caster_poles_obl[MAX_CASTERS];
    vec4 u_caster_colors[MAX_CASTERS];
    vec4 u_caster_atmos[MAX_CASTERS];
    vec4 u_caster_ozone[MAX_CASTERS];
    vec4 u_caster_ozone_vert[MAX_CASTERS];
    vec4 u_caster_grazing[MAX_CASTERS]; // xyz: grazing tau, w: inverse reference radius km
};

uniform vec3 u_host_planet_pos;
uniform float u_host_planet_radius;
uniform vec3 u_host_planet_color;
uniform vec4 u_host_planet_atmo; // xyz: vertical tau, w: thickness km
uniform vec4 u_host_planet_shadow; // xyz: grazing tau, w: scale height km
uniform vec4 u_host_planet_ozone;
uniform float u_host_planet_refractivity;  // surface (n_mix - 1) for eclipse refraction
uniform float u_host_planet_max_bend;      // precomputed host planet max bend
uniform vec3 u_camera_pos;
uniform vec4 u_host_planet_pole_obl;
uniform float u_host_planet_R_minor;
uniform int u_clip_mode;
uniform float u_clip_atmo_radius; // host atmosphere's equatorial radius, AU
uniform uint u_caster_mask_lo;
uniform uint u_caster_mask_hi;
uniform bool u_planetshine_enabled;
uniform float u_exposure;
uniform bool u_hdr_enabled;
uniform float u_caster_max_bend[64];

#include "common/refraction.glsl"

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - 3.0);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 adjust_hsba(vec3 color, float hue_shift, float saturation, float brightness) {
    vec3 hsv = rgb2hsv(color);
    hsv.x = fract(hsv.x + hue_shift);
    hsv.y = clamp(hsv.y * saturation, 0.0, 1.0);
    hsv.z = clamp(hsv.z * brightness, 0.0, 1.0);
    return hsv2rgb(hsv);
}

struct RingPlane {
    vec3 color;
    float inner_r;
    float outer_r;
    float opacity;
    float scatter;
    float asymmetry;
    float backscatter;
    int row_idx;
    float is_textured;
    float unlit_factor;
    float saturation;
    float hue_shift;
    float brightness;
    float alpha_boost;
};

#define MAX_RING_PLANES 16
uniform RingPlane u_ring_planes[MAX_RING_PLANES];
uniform int u_num_ring_planes;
uniform sampler2D u_ring_gradients;

out vec4 out_color;
const float PI = 3.14159265358979323846;

uniform bool u_is_textured;
uniform sampler2D u_ring_texture;

#include "common/ring_optics.glsl"
#include "common/ring_source.glsl"

struct RingComponent {
    vec3 color;
    vec4 props;
    float alpha;
    float tau;
    bool textured;
};
RingComponent ring_material(int i, float r, float footprint) {
    RingComponent c;
    c.color=vec3(0.0); c.props=vec4(0.0); c.alpha=0.0; c.tau=0.0; c.textured=false;
    float inner_r=u_ring_planes[i].inner_r,outer_r=u_ring_planes[i].outer_r;
    float dr=min(footprint*0.75,(outer_r-inner_r)*0.05);
    if(r<inner_r-dr || r>outer_r+dr) return c;
    float t=clamp((r-inner_r)/max(1e-6,outer_r-inner_r),0.0,1.0);
    c.textured=u_is_textured && u_ring_planes[i].is_textured>0.5;
    vec4 material;
    if(c.textured) {
        material=texture(u_ring_texture,vec2(t,0.5));
        material.rgb=adjust_hsba(material.rgb,u_ring_planes[i].hue_shift,
                               u_ring_planes[i].saturation,u_ring_planes[i].brightness);
        if(material.a>1e-5) material.a=pow(material.a,1.0/max(0.01,u_ring_planes[i].alpha_boost));
        material.rgb*=u_ring_planes[i].color;
    } else {
        float row=(float(u_ring_planes[i].row_idx)+0.5)/float(textureSize(u_ring_gradients,0).y);
        material=texture(u_ring_gradients,vec2(t,row));
    }
    c.color=pow(max(material.rgb,vec3(0.0)),vec3(2.2));
    c.props=vec4(u_ring_planes[i].asymmetry,u_ring_planes[i].backscatter,
                 u_ring_planes[i].scatter,u_ring_planes[i].unlit_factor);
    c.alpha=clamp(material.a*u_ring_planes[i].opacity,0.0,RING_MAX_ALPHA);
    c.tau=ring_tau(c.alpha);
    return c;
}

float get_oblate_radius(float r_eq, float r_minor, vec3 pole, vec3 L, vec3 perp_vec) {
    if (r_minor < 1e-6 || r_eq < 1e-6) return r_eq;
    float perp_len = length(perp_vec);
    if (perp_len < 1e-6) return r_eq;
    float PdotL = dot(pole, L);
    vec3 P_proj = pole - L * PdotL;
    float P_proj_len = length(P_proj);
    if (P_proj_len < 1e-6) return r_eq;
    vec3 P_dir = P_proj / P_proj_len;
    float y = dot(perp_vec, P_dir);
    float x = length(perp_vec - y * P_dir);
    float denom = sqrt((x / r_eq) * (x / r_eq) + (y / r_minor) * (y / r_minor));
    if (denom < 1e-6) return r_eq;
    return perp_len / denom;
}

// Shared analytical eclipse penumbra + physical atmospheric lens optics & Danjon refraction tinting.
#include "common/eclipse_shadow.glsl"

void main() {
    if (f_clip_z <= 0.0) discard;

    vec3 cam_to_host = u_host_planet_pos - u_camera_pos;
    vec3 view_ray = normalize(f_local_pos + cam_to_host);
    vec3 ray_dir = view_ray;
    vec3 ray_origin = u_camera_pos;

    float s_min_au = 0.0;
    if (u_refract_max_bend > 1e-6) {
        vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
        float d_km = length(cam_to_host * u_au_to_km);
        
        vec3 pole_n_approx = length(u_host_planet_pole_obl.xyz) > 1e-4 ? normalize(u_host_planet_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
        float d_n_approx = dot(view_ray, pole_n_approx);
        if (abs(d_n_approx) > 1e-6) {
            float t_approx = dot(cam_to_host, pole_n_approx) / d_n_approx;
            if (t_approx > 0.0) {
                d_km = t_approx * u_au_to_km;
            }
        }
        
        // Total (un-parallaxed) bend: the anchored rotation below applies the
        // (1 - s_min/d) parallax geometrically; using the parallaxed alpha here
        // would double-count it.
        float alpha = compute_refraction_total(C_km, view_ray, d_km);
        if (alpha > 1e-7) {
            vec3 u_dir = C_km - view_ray * dot(C_km, view_ray);
            float u_len = length(u_dir);
            if (u_len > 1e-5) {
                u_dir /= u_len;
                ray_dir = normalize(view_ray * cos(alpha) - u_dir * sin(alpha));
                float local_s_min = -dot(u_camera_pos - u_refract_center, view_ray);
                if (local_s_min > 0.0) {
                    s_min_au = local_s_min;
                    ray_origin = u_camera_pos + s_min_au * (view_ray - ray_dir);
                }
            }
        }
    }

    // Precise ray origin in body-local coordinates, avoiding catastrophic cancellation
    // By starting from the precise bounding mesh vertex and applying the relative refraction shift.
    float d_bounding = length(f_local_pos + cam_to_host);
    vec3 O_local = f_local_pos + (d_bounding - s_min_au) * (ray_dir - view_ray);

    // Body-local ray-plane intersection for precision
    vec3 pole_n = length(u_host_planet_pole_obl.xyz) > 1e-4 ? normalize(u_host_planet_pole_obl.xyz) : vec3(0.0, 1.0, 0.0);
    float d_n = dot(ray_dir, pole_n);
    if (abs(d_n) < 1e-6) discard;

    // Intersect ray with ring plane
    // Ring plane passes through host planet center, normal = pole_n
    // From O_local along ray_dir: t where dot(O_local + t * ray_dir, pole_n) = 0
    float t_local = -dot(O_local, pole_n) / d_n;

    vec3 hit_local = O_local + ray_dir * t_local;  // body-local hit, ring-radius scale
    vec3 hit_pos = hit_local + u_host_planet_pos;  // world-space, for depth/clip only

    vec4 clip_pos = projection * view * vec4(hit_pos, 1.0);
    float hit_clip_z = clip_pos.w;
    if (hit_clip_z <= 0.0) discard;
    gl_FragDepth = log2(max(1e-6, u_depth_C * hit_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);

    if (u_clip_mode != 0) {
        // A camera-facing ring point can still lie behind the atmospheric limb.
        // Split by atmosphere crossed BEFORE the hit, rather than a hemisphere.
        // Work at the precise local ray anchor, using the same oblate-to-sphere
        // scale as atmo.frag; do not reconstruct a distant world-space origin.
        float f_scale = max(1.0, u_host_planet_pole_obl.w);
        vec3 origin_sph = O_local + pole_n * (dot(O_local, pole_n) * (f_scale - 1.0));
        vec3 dir_sph = ray_dir + pole_n * (dot(ray_dir, pole_n) * (f_scale - 1.0));
        float a = dot(dir_sph, dir_sph);
        vec3 cross_vec = cross(origin_sph, dir_sph);
        float p2 = dot(cross_vec, cross_vec) / a;
        float r2 = u_clip_atmo_radius * u_clip_atmo_radius;
        bool atmo_in_front = false;
        if (u_clip_atmo_radius > 0.0 && p2 < r2) {
            float half_chord = sqrt((r2 - p2) / a);
            float t_closest = -dot(origin_sph, dir_sph) / a;
            float entry = max(t_closest - half_chord, -d_bounding);
            float exit = min(t_closest + half_chord, t_local);
            atmo_in_front = entry < exit;
        }
        if (u_clip_mode == 1 && !atmo_in_front) discard;
        if (u_clip_mode == 2 && atmo_in_front) discard;
    }

    float r = length(hit_local);

    float total_tau = 0.0;
    float total_faded_tau = 0.0;

    float footprint=fwidth(r);
    for (int i=0; i<u_num_ring_planes; ++i) {
        RingComponent c=ring_material(i,r,footprint);
        if(c.tau<=0.0) continue;
        float inner_r=u_ring_planes[i].inner_r,outer_r=u_ring_planes[i].outer_r;
        float dr=min(footprint*0.75,(outer_r-inner_r)*0.05);
        float fade=smoothstep(inner_r-dr,inner_r+dr,r)*(1.0-smoothstep(outer_r-dr,outer_r+dr,r));
        total_tau+=c.tau;
        total_faded_tau+=ring_tau(c.alpha*fade);
    }

    if (total_tau <= 1e-6) {
        discard;
    }

    vec3 V = -ray_dir;
    vec3 N = normalize(f_normal);
    float cam_side = dot(N, V);

    float mu_v = max(abs(cam_side),1e-7);
    float physical_alpha = ring_absorbed(total_tau/mu_v);
    float faded_alpha = ring_absorbed(total_faded_tau/mu_v);

    vec3 total_direct_illum_color = vec3(0.0);

    for (int s = 0; s < u_num_stars; s++) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        vec3 frag_to_star = (star_pos - u_host_planet_pos) - hit_local;
        float dist_to_star = length(frag_to_star);
        if (dist_to_star < 1e-5) continue;
        vec3 L = frag_to_star / dist_to_star;

        vec3 pole_dir = normalize(u_stars_poles_obl[s].xyz);
        float star_sin_lat = abs(dot(L, pole_dir));

        vec3 eq_color = u_stars_colors[s].rgb;
        float eq_lum = u_stars_colors[s].a;
        vec3 pole_color = u_stars_pole_colors[s].rgb;
        float pole_lum = u_stars_pole_colors[s].a;

        vec3 star_color = mix(eq_color, pole_color, star_sin_lat);
        float star_lum = mix(eq_lum, pole_lum, star_sin_lat);

        float sun_side = dot(N,L);
        float star_ang_radius = clamp(star_radius/dist_to_star,0.0,0.999);
        int source_samples = ring_resolve_disk(sun_side,star_ang_radius) ? RING_DISK_SAMPLES : 1;
        vec3 direct_illum = vec3(0.0);
        for (int i=0; i<u_num_ring_planes; ++i) {
            RingComponent c = ring_material(i,r,footprint);
            if (c.tau<=0.0) continue;
            float material_response = 0.0;
            for (int q=0; q<source_samples; ++q) {
                vec4 source = source_samples==1 ? vec4(L,1.0)
                    : ring_disk_sample(L,N,star_ang_radius,q);
                float sun = dot(N,source.xyz);
                float mu = clamp(-dot(source.xyz,V),-1.0,1.0);
                material_response += source.w*ring_radiance(total_tau,c.alpha,mu_v,sun,mu,
                    c.props,c.textured,cam_side*sun>=0.0);
            }
            direct_illum += c.color*(c.tau/total_tau)*material_response;
        }
        direct_illum *= PI*(u_hdr_enabled ? star_lum/(dist_to_star*dist_to_star) : 1.0);

        vec3 shadow_s = vec3(1.0);
        float star_radius_over_dist = star_ang_radius;

        for (int j = 0; j < u_num_casters; j++) {
            if (dot(shadow_s, shadow_s) < 0.001) break;

            if (j < 32) { if ((u_caster_mask_lo & (1u << j)) == 0u) continue; }
            else        { if ((u_caster_mask_hi & (1u << (j - 32))) == 0u) continue; }

            vec3 caster_pos = u_casters[j].xyz;
            float caster_r = u_casters[j].w;
            float atmo_h = u_caster_atmos[j].w;

            vec3 frag_to_caster = (caster_pos - u_host_planet_pos) - hit_local;
            float t_proj = dot(frag_to_caster, L);
            if (t_proj < 0.0) continue;

            float dist_sq = dot(frag_to_caster, frag_to_caster);
            if (dist_sq < caster_r * caster_r * 1.0404) continue;

            float dist_to_caster = sqrt(dist_sq);
            float perp_sq = max(0.0, dist_sq - t_proj * t_proj);

            // Bounding cone early out using maximum equatorial radius
            float max_effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * (4.0 / u_au_to_km) : 0.0);
            float max_r_penumbra = max_effective_r + dist_to_caster * star_radius_over_dist;
            if (perp_sq > max_r_penumbra * max_r_penumbra) continue;

            float caster_r_minor = u_caster_poles_obl[j].w;
            vec3 perp_vec = frag_to_caster - t_proj * L;
            if (caster_r_minor < caster_r - 1e-5) {
                vec3 pole = u_caster_poles_obl[j].xyz;
                caster_r = get_oblate_radius(caster_r, caster_r_minor, pole, L, perp_vec);
            }

            float directional_star_r = star_radius;
            float star_r_minor = u_stars_poles_obl[s].w;
            if (star_r_minor < star_radius - 1e-5) {
                vec3 star_pole = u_stars_poles_obl[s].xyz;
                directional_star_r = get_oblate_radius(star_radius, star_r_minor, star_pole, L, perp_vec);
            }
            float local_star_radius_over_dist = directional_star_r / dist_to_star;

            float effective_r = caster_r + (atmo_h > 0.0 ? atmo_h * (4.0 / u_au_to_km) : 0.0);
            float r_penumbra = effective_r + dist_to_caster * local_star_radius_over_dist;
            if (perp_sq > r_penumbra * r_penumbra) continue;

            shadow_s *= casterShadowTerm(local_star_radius_over_dist * dist_to_caster,
                caster_r, sqrt(perp_sq), u_caster_max_bend[j], u_caster_grazing[j],
                u_caster_ozone[j], u_caster_colors[j].w, dist_to_caster, u_au_to_km);
        }

        if (u_host_planet_radius > 0.0) {
            vec3 frag_to_host = -hit_local;
            float t_proj = dot(frag_to_host, L);

            if (t_proj > 0.0 && t_proj < dist_to_star) {
                float dist_sq = dot(frag_to_host, frag_to_host);
                float dist_to_host = sqrt(dist_sq);
                vec3 cross_vec = cross(frag_to_host, L);
                float perp_sq = dot(cross_vec, cross_vec);

                float host_r = u_host_planet_radius;
                float host_atmo_h = u_host_planet_atmo.w;
                float host_r_minor = u_host_planet_R_minor;

                // Bounding cone early out using maximum equatorial radius
                float max_effective_r = host_r + (host_atmo_h > 0.0 ? host_atmo_h * (4.0 / u_au_to_km) : 0.0);
                float max_r_penumbra = max_effective_r + dist_to_host * star_radius_over_dist;

                if (perp_sq <= max_r_penumbra * max_r_penumbra) {
                    vec3 perp_vec = frag_to_host - t_proj * L;
                    if (host_r_minor < host_r - 1e-5) {
                        vec3 host_pole = u_host_planet_pole_obl.xyz;
                        host_r = get_oblate_radius(host_r, host_r_minor, host_pole, L, perp_vec);
                    }

                    float directional_star_r = star_radius;
                    float star_r_minor = u_stars_poles_obl[s].w;
                    if (star_r_minor < star_radius - 1e-5) {
                        vec3 star_pole = u_stars_poles_obl[s].xyz;
                        directional_star_r = get_oblate_radius(star_radius, star_r_minor, star_pole, L, perp_vec);
                    }
                    float local_star_radius_over_dist = directional_star_r / dist_to_star;

                    float effective_r = host_r + (host_atmo_h > 0.0 ? host_atmo_h * (4.0 / u_au_to_km) : 0.0);
                    float r_penumbra = effective_r + dist_to_host * local_star_radius_over_dist;

                    if (perp_sq <= r_penumbra * r_penumbra) {
                        float max_bend = u_host_planet_max_bend > 0.0
                            ? u_host_planet_max_bend
                            : (host_atmo_h > 0.0
                                ? clamp(2.0 * max(u_host_planet_refractivity, 0.0) * sqrt(3.14159265359 * host_r * u_au_to_km / max(1e-6, u_host_planet_shadow.w * 2.0)), 0.001, 0.10)
                                : 0.0);
                        shadow_s *= casterShadowTerm(local_star_radius_over_dist * dist_to_host, host_r, sqrt(perp_sq),
                            max_bend, vec4(u_host_planet_shadow.xyz, 1.0 / max(u_host_planet_radius * u_au_to_km, 100.0)),
                            u_host_planet_ozone, u_host_planet_shadow.w, dist_to_host, u_au_to_km);
                    }
                }
            }
        }

        total_direct_illum_color += star_color * direct_illum * shadow_s;
    }

    vec3 total_planetshine = vec3(0.0);
    if (u_planetshine_enabled && u_host_planet_radius>0.0) {
        float dist_host = length(hit_local);
        vec3 L_center = -hit_local/max(dist_host,1e-10);
        float sin_radius = min(u_host_planet_radius/max(dist_host,1e-10),0.999);
        float eta = max(1.0,u_host_planet_pole_obl.w*u_host_planet_pole_obl.w);
        vec3 O = hit_local;
        float oq = dot(O,pole_n);
        float c = dot(O,O)+(eta-1.0)*oq*oq-u_host_planet_radius*u_host_planet_radius;
        for (int i=0; i<u_num_ring_planes; ++i) {
            RingComponent material=ring_material(i,r,footprint);
            if(material.tau<=0.0) continue;
            for (int q=0; q<RING_DISK_SAMPLES; ++q) {
                vec4 source = ring_disk_sample(L_center,N,sin_radius,q);
                vec3 D = source.xyz;
                float dq = dot(D,pole_n);
                float a = dot(D,D)+(eta-1.0)*dq*dq;
                float b = dot(O,D)+(eta-1.0)*oq*dq;
                float discriminant = b*b-a*c;
                if (discriminant<=0.0) continue;
                float distance_to_surface = (-b-sqrt(discriminant))/a;
                if (distance_to_surface<=0.0) continue;
                vec3 Q = O+distance_to_surface*D;
                vec3 normal = normalize(Q+pole_n*((eta-1.0)*dot(Q,pole_n)));
                float sun=dot(N,D);
                float response=ring_radiance(total_tau,material.alpha,mu_v,sun,-dot(D,V),
                    material.props,material.textured,cam_side*sun>=0.0);
                vec3 ring_response=material.color*(material.tau/total_tau)*response;
                // Ring output multiplies radiance by pi; it cancels the host's
                // Lambertian 1/pi. Remaining weight is the exact disk dOmega.
                float solid_angle_weight = PI*sin_radius*sin_radius*source.w;
                for (int s=0; s<u_num_stars; ++s) {
                    vec3 to_star = (u_stars_pos_radius[s].xyz-u_host_planet_pos)-Q;
                    float distance_to_star = max(length(to_star),1e-10);
                    vec3 L_host = to_star/distance_to_star;
                    float incidence = max(0.0,dot(normal,L_host));
                    if (incidence<=0.0) continue;
                    float latitude = abs(dot(L_host,normalize(u_stars_poles_obl[s].xyz)));
                    vec3 star_color = mix(u_stars_colors[s].rgb,u_stars_pole_colors[s].rgb,latitude);
                    float star_lum = mix(u_stars_colors[s].a,u_stars_pole_colors[s].a,latitude);
                    float flux = u_hdr_enabled ? star_lum/(distance_to_star*distance_to_star) : 1.0;
                    total_planetshine += u_host_planet_color*star_color*flux*incidence
                                        *ring_response*solid_angle_weight;
                }
            }
        }
    }

    vec3 raw_color = total_direct_illum_color;
    if (u_planetshine_enabled) {
        raw_color += total_planetshine;
    }
    vec3 final_color = u_hdr_enabled ? (raw_color * u_exposure) : raw_color;

    float fade_scale = faded_alpha / max(1e-6, physical_alpha);
    out_color = vec4(final_color * fade_scale, faded_alpha);
}
