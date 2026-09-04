#version 460 core
in vec3 f_color;
in float f_is_star;
in float f_clip_z;
flat in vec2 f_center_px;
flat in float f_apparent_px;
flat in float f_half_size_px;
flat in vec3 f_surface_color;

layout(std140, binding = 1) uniform SceneData {
    mat4 projection;
    mat4 view;
    int u_num_stars;
    float _pad0, _pad1, _pad2;
    vec4 u_stars_pos_radius[16];
    vec4 u_stars_colors[16];
    vec4 u_stars_poles_obl[16];
    vec4 u_stars_pole_colors[16];
    float u_far;
    float u_depth_C;
    int u_num_casters;
    float _pad3;
    vec4 u_casters[64];
    vec4 u_caster_poles_obl[64];
    vec4 u_caster_colors[64];
    vec4 u_caster_atmos[64];
    vec4 u_caster_ozone[64];
};

uniform float u_exposure;
uniform bool u_hdr_enabled;

out vec4 out_color;

#define PI 3.14159265358979323846

void main() {
    float dist_px = length(gl_FragCoord.xy - f_center_px);
    if (dist_px > f_half_size_px) {
        discard;
    }

    // Effective visual radius:
    // f_apparent_px is the physical projected DIAMETER.
    // Physical mesh radius is therefore R_mesh_px = 0.5 * f_apparent_px.
    float R_mesh_px = f_apparent_px * 0.5;
    float R_min = 0.5;
    float R_eff = max(R_mesh_px, R_min);

    // Crisp anti-aliased circular disk boundary centered precisely at R_eff:
    // dist_px < R_eff - 0.35: coverage = 1.0 (fully inside)
    // dist_px == R_eff:       coverage = 0.5 (exact geometric boundary of the 3D mesh)
    // dist_px > R_eff + 0.35: coverage = 0.0 (fully outside)
    float aa_half_width = 0.35;
    float disk_coverage = clamp(0.5 - (dist_px - R_eff) / (aa_half_width * 2.0), 0.0, 1.0);
    disk_coverage = smoothstep(0.0, 1.0, disk_coverage);

    if (disk_coverage <= 0.0) {
        discard;
    }

    // Subpixel Flux Scaling:
    // On a discrete monitor grid, a pixel covers ~40x the area of a human foveal cone.
    // For stars, strict quadratic 1/d^2 energy conservation is maintained (exponent 2.0).
    // For planets, an exponent of 1.3 models the human eye's point-spread contrast response (Ricco's Law),
    // ensuring brilliant planets like Venus (m = -4.5) remain prominently visible against twilight skies,
    // while smoothly and seamlessly matching the physical 1:1 3D mesh at R_mesh_px >= R_min (1.0 at >= 0.5 px).
    float r_ratio = clamp(R_mesh_px / R_min, 0.0, 1.0);
    float flux_scale = (R_mesh_px < R_min) ? pow(r_ratio, (f_is_star > 0.5) ? 2.0 : 1.3) : 1.0;
    vec3 rgb = f_surface_color * (disk_coverage * flux_scale);

    // Camera Exposure scaling
    if (u_hdr_enabled) {
        rgb *= u_exposure;
    }

    // --- Exposure-Dependent Limiting Magnitude Extinction ---
    // Smoothly fade to black when peak signal drops below sensor detection floor
    float peak_signal = max(rgb.r, max(rgb.g, rgb.b));
    float limiting_threshold = 1.0e-5;
    float extinction_fade = smoothstep(limiting_threshold, limiting_threshold * 2.5, peak_signal);
    if (extinction_fade <= 0.0) {
        discard;
    }
    rgb *= extinction_fade;

    // --- Mesh LOD Hand-off Transition (Seamless cross-fade for apparent_px in [2.0, 3.0]) ---
    float transition_weight = 1.0 - smoothstep(2.0, 3.0, f_apparent_px);
    rgb *= transition_weight;

    // Transit silhouette opacity: only blocks background star light during subpixel transits.
    // During the mesh cross-fade transition, out_alpha MUST be 0.0 so that (ONE, ONE_MINUS_SRC_ALPHA)
    // performs a pure additive blend (mesh * w_mesh + point * w_point = 1.0), preventing any 25% dip in brightness
    // or sudden bloom jumps.
    float silhouette_alpha = (transition_weight < 0.999) ? 0.0 : min(1.0, disk_coverage * flux_scale);
    float out_alpha = (f_is_star > 0.5) ? 0.0 : silhouette_alpha;

    // Logarithmic depth matching the main scene
    gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);

    out_color = vec4(rgb, out_alpha);
}
