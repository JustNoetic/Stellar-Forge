#include "ring_shadow_filter.glsl"
#include "ring_penumbra.glsl"

#include "ring_hosts.glsl"
layout(set = 0, binding = 9) uniform sampler2DArray u_ring_shadow_lut;
layout(set = 0, binding = 39) uniform sampler2DArray u_ring_penumbra;

// Ring extinction integral up to u. From the penumbra table when the ring has a slot built at this
// incidence: texture row 0 is the raw prefix, row y > 0 the integral of kernel row y-1. Otherwise the
// raw prefix from the LUT at mu_coord (hardware-filtered across its incidence rows). All are linear
// between their columns.
float surface_ring_integral(bool table, float slot, float y, int layer, float mu_coord, float u) {
    if (table && y > 0.0) {
        float pad = ring_penumbra_row_pad(y-1.0);
        float x = clamp((u+pad)/(1.0+2.0*pad)*float(RING_PENUMBRA_WIDTH-1), 0.0, float(RING_PENUMBRA_WIDTH-1));
        float i0 = min(floor(x), float(RING_PENUMBRA_WIDTH-2));
        return ring_penumbra_gather(u_ring_penumbra, slot, y, i0, x-i0);
    }
    float x = clamp(u, 0.0, 1.0)*4096.0;
    float i0 = min(floor(x), 4095.0);
    if (table) return ring_penumbra_gather(u_ring_penumbra, slot, 0.0, i0, x-i0);
    float a = textureLod(u_ring_shadow_lut, vec3((i0+0.5)/4097.0, mu_coord, float(layer)), 0.0).r;
    float b = textureLod(u_ring_shadow_lut, vec3((i0+1.5)/4097.0, mu_coord, float(layer)), 0.0).r;
    return mix(a, b, x-i0);
}

// Ring shadow of star `star` (scene star index) at a surface point. ring_shadow_filter's boxes (or the
// table's continuous kernel for wide penumbrae), evaluated with one body for both sources so the
// shader carries a single filter. Rings with a penumbra table slot for this star
// (RingHostGPU.counts_indices.w, 4 bits per star) read it: exact incidence and fp32 weights. The slot was built at the star's incidence on the
// ring centre, so receivers whose incidence differs (the star straddling the ring plane) use the LUT.
float compute_surface_ring_shadow(vec3 frag_pos, vec3 L, float sun_ang_rad, float star_distance, uint candidates, uint star) {
    float ring_atten = 1.0;
    float focal_px = max(u_scene.screen_params.y * u_scene.screen_params.z, 1.0);
    float px_world_size = length(frag_pos) / focal_px;
    bool single_box = (u_scene.ring_shadow_params.z & 1u) != 0u;
    bool reference_linear = (u_scene.ring_shadow_params.z & 2u) != 0u;
    float softness = uintBitsToFloat(u_scene.ring_shadow_params.w);
    vec3 weights = single_box ? vec3(0.0, 0.0, 1.0) : reference_linear ? vec3(0.24, 0.46, 0.30) : vec3(0.08, 0.42, 0.50);

    while (candidates != 0u) {
        uint r = uint(findLSB(candidates));
        candidates &= candidates - 1u;
        vec3 r_center = ring_hosts[r].center_min_r.xyz;
        vec3 r_pole = ring_hosts[r].pole_max_r.xyz;
        vec2 r_bounds = vec2(ring_hosts[r].center_min_r.w, ring_hosts[r].pole_max_r.w);
        int layer = int(ring_hosts[r].counts_indices.y);

        vec4 coords = ring_receiver_coordinates(frag_pos - r_center, r_pole, L, r_bounds, sun_ang_rad,
            star_distance, px_world_size, softness, reference_linear);
        // 1.2: the continuous kernel reaches 1.172 h.
        if (coords.w <= 0.0 || coords.x+1.2*coords.y <= 0.0 || coords.x-1.2*coords.y >= 1.0) continue;
        uint slot = star < 8u && !single_box && !reference_linear ? (ring_hosts[r].counts_indices.w >> (4u*star)) & 15u : 0u;
        bool table = false;
        if (slot != 0u) {
            vec3 star_rel = frag_pos + L*star_distance - r_center;
            float built = clamp(abs(dot(star_rel, r_pole))/max(length(star_rel), 1e-30), 0.5/64.0, 63.5/64.0);
            table = abs(coords.z-built) <= 1e-5+2e-3*built;
        }
        // Up to five boxes (texture row, half-width, weight): ring_shadow_filter's three on the raw
        // prefix, and for wide penumbrae the continuous kernel's box on the two kernel rows bracketing
        // its width (ring_penumbra_extinction), blended in over the same range.
        float blend = table && !single_box ? ring_penumbra_smooth_blend(coords.y) : 0.0;
        vec3 sharp = weights*(1.0-blend);
        vec2 smooth_rows = vec2(0.0), smooth_weights = vec2(0.0);
        if (blend > 0.0) {
            float scale;
            float level = ring_penumbra_level(RING_PENUMBRA_SMOOTH_WIDTH*coords.y, scale);
            float row = floor(level), t = level-row;
            smooth_rows = vec2(row, min(row+1.0, float(RING_PENUMBRA_ROWS-1)))+1.0;
            smooth_weights = blend*scale*vec2(1.0-t, t);
        }
        float ext = 0.0;
        for (int k = 0; k < 5; ++k) {
            float w = k < 3 ? sharp[k] : smooth_weights[k-3];
            if (w <= 0.0) continue;
            float y = k < 3 ? 0.0 : smooth_rows[k-3];
            float a = coords.y*(k == 0 ? 0.36 : k == 1 ? 0.72 : k == 2 ? 1.0 : RING_PENUMBRA_SMOOTH_BOX);
            float lo = surface_ring_integral(table, float(slot)-1.0, y, layer, coords.z, coords.x-a);
            float hi = surface_ring_integral(table, float(slot)-1.0, y, layer, coords.z, coords.x+a);
            ext += w*max(hi-lo, 0.0)/max(2.0*a, 1e-20);
        }
        ring_atten *= (1.0 - coords.w*clamp(ext, 0.0, 1.0));
        if (ring_atten <= 1e-4) return 0.0;
    }
    return ring_atten;
}
