// engine/glsl/common/procedural_terrain.glsl
// Hybrid Procedural Amplification for Planetary Terrain (SpaceEngine / Outerra style)
// Evaluates continuous 3D Simplex noise, Rotated FBM, Domain Warping, and Musgrave
// Erosion Ridges on the unit sphere to synthesize natural micro-elevation detail
// seamlessly on top of baked baseline heightmaps without grid/lattice artifacts.

#ifndef PROCEDURAL_TERRAIN_GLSL
#define PROCEDURAL_TERRAIN_GLSL

// ----------------------------------------------------------------------------
// Fast 3D Simplex Noise (Stefan Gustavson / Ashima Arts)
// Continuous across the entire sphere with zero cube-face boundary seams.
// ----------------------------------------------------------------------------
vec4 _pt_mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 _pt_mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 _pt_permute(vec4 x) { return _pt_mod289(((x * 34.0) + 10.0) * x); }
vec4 _pt_taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float simplex_noise3d(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    // Permutations
    i = _pt_mod289(i);
    vec4 p = _pt_permute(_pt_permute(_pt_permute(
               i.z + vec4(0.0, i1.z, i2.z, 1.0))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0))
             + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron
    float n_ = 0.142857142857; // 1.0 / 7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    // Normalize gradients
    vec4 norm = _pt_taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix contributions with radial kernel (C1 continuous at 0.5)
    vec4 m = max(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    // Exactly dot with distance vectors (p0.x0, p1.x1, p2.x2, p3.x3)
    return 105.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// ----------------------------------------------------------------------------
// 3D Orthogonal Rotation Matrix
// Decorrelates coordinate axes between fractal octaves to destroy grid alignment
// ----------------------------------------------------------------------------
const mat3 _PT_ROT = mat3(
     0.00,  0.80,  0.60,
    -0.80,  0.36, -0.48,
    -0.60, -0.48,  0.64
);

// ----------------------------------------------------------------------------
// Domain Warping
// Bends straight grid lines into organic, flowing drainage basins and winding ridges
// ----------------------------------------------------------------------------
vec3 terrain_domain_warp(vec3 p) {
    vec3 q = vec3(
        simplex_noise3d(p * 0.75),
        simplex_noise3d(p * 0.75 + vec3(5.2, 1.3, 2.8)),
        simplex_noise3d(p * 0.75 + vec3(8.5, 4.1, 7.6))
    );
    return p + q * 0.12;
}

// ----------------------------------------------------------------------------
// Fractal Brownian Motion (FBM) - for rolling hills and low-frequency terrain
// ----------------------------------------------------------------------------
float terrain_fbm(vec3 p, int octaves) {
    float val = 0.0;
    float amp = 0.5;
    vec3 cur_p = p;
    for (int i = 0; i < octaves; i++) {
        val += simplex_noise3d(cur_p) * amp;
        cur_p = _PT_ROT * cur_p * 2.02;
        amp *= 0.49;
    }
    return val;
}

// ----------------------------------------------------------------------------
// Natural Erosion Ridges (Musgrave Arêtes + Dendritic Fluvial Carving)
// Synthesizes razor-sharp ridge crests with dendritic gullies carved along edges
// ----------------------------------------------------------------------------
float terrain_erosion_ridges(vec3 p, int octaves) {
    vec3 p_warped = terrain_domain_warp(p);
    float sum = 0.0;
    float amp = 0.5;
    float weight = 1.0;
    vec3 cur_p = p_warped;

    for (int i = 0; i < octaves; i++) {
        float n = simplex_noise3d(cur_p);
        // Absolute value inverted -> sharp mountain arête crests
        n = 1.0 - abs(n);
        n = n * n; // Sharpen ridge spine

        // Fluvial erosion: dendritic water channels carved into ridge flanks
        float gully = 1.0 - abs(simplex_noise3d(cur_p * 1.5 + vec3(11.2, 33.4, 55.6)));
        n -= gully * gully * 0.20 * weight;

        sum += n * amp * weight;
        // Signal-dependent frequency damping: high ridges retain crisp detail,
        // while lower valleys collect sediment and become smoother
        weight = clamp(n * 1.4, 0.0, 1.0);
        cur_p = _PT_ROT * cur_p * 2.07;
        amp *= 0.48;
    }
    return sum;
}

// ----------------------------------------------------------------------------
// Procedural Micro-Elevation Synthesizer
// Combines the baked baseline elevation with procedural micro-fractals.
// Returns procedural displacement in KILOMETERS.
// ----------------------------------------------------------------------------
float compute_procedural_micro_elevation_km(
    vec3 n_sphere,              // normalized unrotated sphere point in [-1, 1]^3
    float lod_level,            // patch LOD level (e.g. 0 to 10)
    float baked_elevation_norm, // raw baseline elevation in [0, 1]
    float elev_min_km,          // minimum elevation in km (e.g. -10.984 km for Earth)
    float elev_span_km          // elevation span in km (e.g. 19.832 km for Earth)
) {
    // Smoothly fade in procedural micro detail as the camera approaches (LOD > 3)
    // From orbit (LOD <= 3), procedural amplitude is 0 to avoid distance shimmer and keep macro shape.
    float lod_weight = smoothstep(3.0, 6.0, lod_level);
    if (lod_weight <= 0.0) {
        return 0.0;
    }

    // Compute physical baseline elevation in km:
    float elev_km = elev_min_km + baked_elevation_norm * elev_span_km;

    // Mask procedural amplitude:
    // For bodies with bathymetry (elev_min_km < -0.1, like Earth), sea level is at 0.0 km.
    // Ocean surfaces stay perfectly smooth equipotential water with zero displacement.
    float land_mask = 1.0;
    float mountain_mask = 0.0;

    if (elev_min_km < -0.1) {
        land_mask = smoothstep(0.00, 0.04, elev_km);
        mountain_mask = smoothstep(0.8, 2.8, elev_km);
    } else {
        float norm_h = clamp(baked_elevation_norm, 0.0, 1.0);
        mountain_mask = smoothstep(0.25, 0.65, norm_h);
    }

    if (land_mask <= 0.001) {
        return 0.0;
    }

    // Base coordinates scaled by planetary feature density:
    // Scale ~450 means feature wavelength is around 14 km at octave 0,
    // refining down to ~850 meters at octave 4.
    vec3 p_seed = n_sphere * 450.0;

    // Combine rolling terrain (FBM) with sharp mountain erosion ridges (Eroded Musgrave)
    float micro_pattern = 0.0;
    if (mountain_mask > 0.01) {
        float rolling = terrain_fbm(p_seed, 4) * 0.45;
        float ridges = terrain_erosion_ridges(p_seed * 1.35 + vec3(12.3, 45.6, 78.9), 4) - 0.45;
        micro_pattern = mix(rolling, ridges, mountain_mask);
    } else {
        micro_pattern = terrain_fbm(p_seed, 4) * 0.45;
    }

    // Max procedural amplitude scaled by elevation span (up to ~350 meters on Earth)
    float max_amplitude_km = min(0.35, elev_span_km * 0.045);

    return micro_pattern * max_amplitude_km * land_mask * lod_weight;
}

#endif // PROCEDURAL_TERRAIN_GLSL
