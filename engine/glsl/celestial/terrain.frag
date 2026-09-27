#version 460 core
in vec3 f_world_pos;
in vec3 f_normal;
in vec2 f_tile_uv;
in vec2 f_local_uv;
flat in float f_tile_slot;
flat in float f_lod_level;
in float f_clip_z;

flat in vec3 f_atmo_tint;
flat in float f_atmo_h;
flat in float f_scale_height;
flat in vec3 f_o3_tau;
flat in vec2 f_o3_layer;
flat in float f_radius_km;

#define MAX_STARS 16
#define MAX_CASTERS 64

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[MAX_STARS]; // xyz = pos, w = radius
    vec4 u_stars_colors[MAX_STARS];     // rgb = color, w = intensity
    vec4 u_stars_poles_obl[MAX_STARS];  // xyz = pole, w = oblateness
    vec4 u_stars_pole_colors[MAX_STARS]; // rgb = pole color, w = pole intensity
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
};

uniform sampler2DArray u_tile_array;
uniform bool u_debug_tiles;
uniform bool u_hdr_enabled;
uniform float u_exposure;
uniform vec3 u_camera_pos;
uniform bool u_is_cloud_pass;

out vec4 out_color;

void main() {
    if (f_tile_slot < 0.0) {
        discard;
    }

    // Write logarithmic depth matching sphere and orbit shaders
    gl_FragDepth = log2(max(1e-6, 1.0 + f_clip_z * u_depth_C)) / log2(u_far * u_depth_C + 1.0);

    // Sample from the tiled Texture2DArray
    vec3 tile_coord = vec3(clamp(f_tile_uv, 0.0, 1.0), f_tile_slot);
    vec4 tex_sample = texture(u_tile_array, tile_coord);

    vec3 N = normalize(f_normal);
    vec3 V = normalize(u_camera_pos - f_world_pos);

    // =========================================================================
    // CLOUD PASS (Transparent Spherified Cube Quadtree Shell)
    // =========================================================================
    if (u_is_cloud_pass) {
        // Cloud alpha / opacity:
        // Use alpha channel if available; if alpha is uniformly 1.0 (raw RGB), fallback to red channel
        float cloud_alpha = tex_sample.a;
        if (cloud_alpha > 0.999 && (tex_sample.r < 0.99 || tex_sample.g < 0.99)) {
            cloud_alpha = tex_sample.r;
        }

        if (cloud_alpha < 0.02) {
            discard;
        }

        // Two-sided normal handling for orbit (outside) vs ground (inside looking up)
        bool is_underside = dot(N, V) < 0.0;
        vec3 cloud_norm = is_underside ? -N : N;

        // Smooth limb falloff to avoid harsh polygon horizon silhouette
        float NdotV = abs(dot(N, V));
        float limb_soften = smoothstep(0.0, 0.015, NdotV);
        cloud_alpha *= limb_soften;

        vec3 cloud_albedo = vec3(0.95);
        vec3 total_cloud_diffuse = vec3(0.0);
        int num_stars = min(u_num_stars, MAX_STARS);

        for (int s = 0; s < num_stars; ++s) {
            vec3 star_pos = u_stars_pos_radius[s].xyz;
            float star_radius = u_stars_pos_radius[s].w;
            vec3 frag_to_star = star_pos - f_world_pos;
            float dist_to_star = length(frag_to_star);
            if (dist_to_star < 1e-5) continue;
            vec3 L = frag_to_star / dist_to_star;

            float star_ang_radius = star_radius / dist_to_star;
            float sin_alpha = clamp(star_ang_radius, 0.0, 1.0);

            // Cloud elevation geometric terminator offset
            float cloud_h_km = max(0.5, f_scale_height * 0.35);
            float R_km = max(f_radius_km, 1e-3);
            float term_offset = sqrt(max(0.0, 2.0 * cloud_h_km / R_km));
            float effective_sun_cos = dot(cloud_norm, L) + term_offset;

            float diffuse = 0.0;
            if (is_underside) {
                // Underside of cloud illuminated by transmitted sunlight through cloud slab
                float top_illum = clamp((effective_sun_cos + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);
                float cos_trans = -dot(V, L);
                float forward_trans = pow(max(0.0, cos_trans), 4.0) * 0.35;
                diffuse = top_illum * (0.50 + forward_trans);
            } else {
                float sun_vis = clamp((effective_sun_cos + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);
                diffuse = sun_vis;
                // Forward scattering / silver lining along backlit crescent rim
                float cos_forward = -dot(V, L);
                float silver_lining = pow(max(0.0, cos_forward), 6.0) * 0.35 * sun_vis;
                diffuse += silver_lining;
            }

            // Direct sunlight extinction at cloud altitude
            vec3 incoming_tint = vec3(1.0);
            if (f_atmo_h > 0.0) {
                float mu = max(effective_sun_cos, 0.0);
                float H_scale = max(f_scale_height, 1e-3);
                float am = (sqrt(R_km * R_km * mu * mu + 2.0 * R_km * H_scale + H_scale * H_scale) - R_km * mu) / H_scale;
                vec3 tau_at_cloud = max(f_atmo_tint, vec3(0.0)) * exp(-cloud_h_km / H_scale);
                incoming_tint = exp(-tau_at_cloud * am);
            }

            float star_lum = u_stars_colors[s].w;
            float falloff = star_lum / max(1e-8, dist_to_star * dist_to_star);
            vec3 star_color = u_stars_colors[s].rgb;

            total_cloud_diffuse += star_color * incoming_tint * (diffuse * falloff);
        }

        vec3 cloud_final_color = cloud_albedo * total_cloud_diffuse;
        if (u_hdr_enabled) {
            cloud_final_color *= u_exposure;
        }

        out_color = vec4(cloud_final_color, cloud_alpha);
        return;
    }

    // =========================================================================
    // TERRAIN SURFACE PASS
    // =========================================================================
    // Decode sRGB to Linear
    vec3 albedo = pow(tex_sample.rgb, vec3(2.2));

    // Accumulate direct illumination from all stars in SceneData
    vec3 total_diffuse = vec3(0.0);
    int num_stars = min(u_num_stars, MAX_STARS);
    for (int s = 0; s < num_stars; ++s) {
        vec3 star_pos = u_stars_pos_radius[s].xyz;
        float star_radius = u_stars_pos_radius[s].w;
        vec3 frag_to_star = star_pos - f_world_pos;
        float dist_to_star = length(frag_to_star);
        if (dist_to_star < 1e-5) continue;
        vec3 L = frag_to_star / dist_to_star;

        // Angular radius of star disc for soft terminator penumbra
        float star_ang_radius = star_radius / dist_to_star;
        float sin_alpha = clamp(star_ang_radius, 0.0, 1.0);

        // Physical Lambert cosine law with penumbra transition
        float sun_cos = dot(N, L);
        float diffuse = clamp((sun_cos + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);

        vec3 incoming_light_tint = vec3(1.0);
        vec3 direct_light_tint = vec3(1.0);
        vec3 diffuse_light_tint = vec3(0.0);

        // Atmosphere light extinction on ground (Rayleigh + Mie + Ozone)
        if (f_atmo_h > 0.0) {
            float R_km = max(f_radius_km, 1e-3);
            float H_scale = max(f_scale_height, 1e-3);
            float mu = max(sun_cos, 0.0);

            // Geometric relative air mass (Rozenberg curved-atmosphere formulation)
            float am = (sqrt(R_km * R_km * mu * mu + 2.0 * R_km * H_scale + H_scale * H_scale) - R_km * mu) / H_scale;

            vec3 tau_vertical = max(f_atmo_tint, vec3(0.0));
            vec3 tau_direct = tau_vertical * am;

            // Ozone absorption (Chappuis band) absorbs orange/red at high air masses
            if (dot(f_o3_tau, f_o3_tau) > 0.0) {
                float o3_w = f_o3_layer.y;
                float R_o3 = R_km + max(f_o3_layer.x, 0.0);
                float am_o3 = (sqrt(R_o3 * R_o3 * mu * mu + 2.0 * R_o3 * o3_w + o3_w * o3_w) - R_o3 * mu) / o3_w;
                tau_direct += f_o3_tau * am_o3;
            }

            direct_light_tint = exp(-tau_direct);

            // Downward diffuse daylight (skylight dome illumination)
            vec3 tau_slant = tau_vertical * am;
            vec3 diffuse_sky = (vec3(1.0) - exp(-tau_slant)) * (max(0.0, mu) / (vec3(1.0) + 0.75 * tau_vertical));
            diffuse_light_tint = diffuse_sky;

            incoming_light_tint = direct_light_tint + diffuse_light_tint;
        }

        float star_lum = u_stars_colors[s].w;
        float falloff = star_lum / max(1e-8, dist_to_star * dist_to_star);
        vec3 star_color = u_stars_colors[s].rgb;

        total_diffuse += star_color * incoming_light_tint * (diffuse * falloff);
    }

    // Zero artificial ambient: night side is naturally pitch black
    vec3 final_color = albedo * total_diffuse;

    // Tile Debug Visualization mode
    if (u_debug_tiles) {
        // 1. LOD Color Palette
        vec3 lod_palette[8] = vec3[8](
            vec3(0.95, 0.25, 0.25), // LOD 0: Red
            vec3(1.00, 0.55, 0.15), // LOD 1: Orange
            vec3(0.95, 0.90, 0.20), // LOD 2: Yellow
            vec3(0.25, 0.90, 0.35), // LOD 3: Green
            vec3(0.15, 0.80, 0.95), // LOD 4: Cyan
            vec3(0.25, 0.45, 1.00), // LOD 5: Blue
            vec3(0.70, 0.25, 0.95), // LOD 6: Purple
            vec3(1.00, 0.25, 0.75)  // LOD 7: Pink
        );
        int lod_i = clamp(int(f_lod_level + 0.5), 0, 7);
        vec3 lod_tint = lod_palette[lod_i];

        // 2. Tile Borders
        vec2 d_edge = min(f_local_uv, 1.0 - f_local_uv);
        float is_border = 1.0 - step(0.018, min(d_edge.x, d_edge.y));

        // False color tint + border overlay
        final_color = mix(final_color * lod_tint * 1.5, vec3(1.0, 1.0, 0.1), is_border);
    }

    if (u_hdr_enabled) {
        final_color *= u_exposure;
    }

    out_color = vec4(final_color, 1.0);
}
