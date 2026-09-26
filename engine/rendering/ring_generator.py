"""
Procedural Planetary Ring Texture & Structure Generator for Stellar-Forge.
Inspired by the physics and procedural techniques of SpaceEngine.

Generates 1D radial profiles f(r) mapping normalized radial radius u in [0, 1] to:
- Optical depth tau(u) and physical transparency alpha(u) = 1 - exp(-tau(u))
- Multi-scale concentric ringlet striations via 1D Fractal Brownian Motion (fBm)
- Major concentric band partitioning (like Saturn's D, C, B, A, F rings)
- Sharp Lindblad & shepherd moon resonance gaps (Cassini, Encke, Keeler)
- Damped spiral density wave chirps near ring edges
- Physically based spectral albedo & color palettes (Icy, Silicate Dust, Carbonaceous, Tholins)
"""

import numpy as np
from PIL import Image

def _hash_int(n, seed):
    """Deterministic integer hash function for 1D lattice gradient noise."""
    n = (n.astype(np.int64) * 0x45d9f3b + int(seed)) & 0x7fffffff
    n = ((n >> 16) ^ n) * 0x45d9f3b
    n = ((n >> 16) ^ n) * 0x45d9f3b
    n = (n >> 16) ^ n
    return (n & 0x7fffffff) / float(0x7fffffff) * 2.0 - 1.0

def perlin1d(x, seed=42):
    """Vectorized 1D Perlin gradient noise with smooth quintic interpolation."""
    x0 = np.floor(x).astype(np.int64)
    x1 = x0 + 1
    dx0 = (x - x0).astype(np.float32)
    dx1 = dx0 - 1.0
    
    g0 = _hash_int(x0, seed)
    g1 = _hash_int(x1, seed)
    
    v0 = g0 * dx0
    v1 = g1 * dx1
    
    # 5th order Hermite curve: 6t^5 - 15t^4 + 10t^3
    t = dx0
    s = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return (1.0 - s) * v0 + s * v1

def fbm1d(x, octaves=6, lacunarity=2.15, persistence=0.52, seed=42):
    """Multi-octave 1D Fractal Brownian Motion."""
    val = np.zeros_like(x, dtype=np.float32)
    amp = 1.0
    freq = 1.0
    max_amp = 0.0
    for i in range(octaves):
        val += perlin1d(x * freq, seed=seed + i * 1013904223) * amp
        max_amp += amp
        amp *= persistence
        freq *= lacunarity
    return val / max(1e-6, max_amp)

def smoothstep(edge0, edge1, x):
    """Vectorized smoothstep function."""
    t = np.clip((x - edge0) / max(1e-7, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# Preset definitions with physical parameters and aesthetic profiles
RING_PRESETS = {
    "Saturnian Ice": {
        "band_count": 4,
        "gap_count": 3,
        "striation_freq": 160.0,
        "contrast": 1.7,
        "base_density": 2.2,
        "core_boost": 2.5,
        "inner_radius_ratio": 1.20,
        "outer_radius_ratio": 2.30,
        "asymmetry": 0.436,
        "backscatter": -0.657,
        "scatter": 1.0,
        "opacity": 1.0,
        "unlit_factor": 0.45,
        "color_deep": (0.97, 0.94, 0.89),      # Creamy dense snow
        "color_mid": (0.95, 0.95, 0.96),       # Pure ice
        "color_thin": (0.75, 0.86, 0.98),      # Azure Rayleigh haze in tenuous zones
    },
    "Jovian Silicate Dust": {
        "band_count": 2,
        "gap_count": 1,
        "striation_freq": 60.0,
        "contrast": 1.2,
        "base_density": 0.25,
        "core_boost": 1.0,
        "inner_radius_ratio": 1.25,
        "outer_radius_ratio": 1.85,
        "asymmetry": 0.75,
        "backscatter": -0.15,
        "scatter": 0.85,
        "opacity": 0.5,
        "unlit_factor": 0.8,
        "color_deep": (0.42, 0.35, 0.28),
        "color_mid": (0.34, 0.28, 0.23),
        "color_thin": (0.22, 0.18, 0.15),
    },
    "Uranian Charcoal": {
        "band_count": 6,
        "gap_count": 5,
        "striation_freq": 220.0,
        "contrast": 2.4,
        "base_density": 2.8,
        "core_boost": 2.8,
        "inner_radius_ratio": 1.45,
        "outer_radius_ratio": 2.05,
        "asymmetry": 0.65,
        "backscatter": -0.35,
        "scatter": 0.4,
        "opacity": 0.95,
        "unlit_factor": 0.3,
        "color_deep": (0.16, 0.16, 0.17),
        "color_mid": (0.11, 0.11, 0.12),
        "color_thin": (0.05, 0.05, 0.06),
    },
    "Neptunian Arcs": {
        "band_count": 3,
        "gap_count": 2,
        "striation_freq": 140.0,
        "contrast": 2.0,
        "base_density": 1.4,
        "core_boost": 2.2,
        "inner_radius_ratio": 1.60,
        "outer_radius_ratio": 2.55,
        "asymmetry": 0.70,
        "backscatter": -0.25,
        "scatter": 0.7,
        "opacity": 0.7,
        "unlit_factor": 0.6,
        "color_deep": (0.45, 0.32, 0.26),
        "color_mid": (0.32, 0.24, 0.20),
        "color_thin": (0.18, 0.14, 0.12),
    },
    "Chariklo Centaur": {
        "band_count": 2,
        "gap_count": 1,
        "striation_freq": 240.0,
        "contrast": 2.6,
        "base_density": 2.8,
        "core_boost": 2.6,
        "inner_radius_ratio": 1.80,
        "outer_radius_ratio": 2.15,
        "asymmetry": 0.50,
        "backscatter": -0.55,
        "scatter": 0.35,
        "opacity": 0.98,
        "unlit_factor": 0.35,
        "color_deep": (0.78, 0.79, 0.82),
        "color_mid": (0.65, 0.67, 0.70),
        "color_thin": (0.40, 0.42, 0.46),
    },
    "Warm Tholin Ice": {
        "band_count": 4,
        "gap_count": 3,
        "striation_freq": 150.0,
        "contrast": 1.8,
        "base_density": 2.2,
        "core_boost": 2.4,
        "inner_radius_ratio": 1.30,
        "outer_radius_ratio": 2.40,
        "asymmetry": 0.48,
        "backscatter": -0.50,
        "scatter": 0.55,
        "opacity": 1.0,
        "unlit_factor": 0.4,
        "color_deep": (0.95, 0.72, 0.46),      # Golden amber
        "color_mid": (0.82, 0.54, 0.32),       # Burnt sienna / tholins
        "color_thin": (0.55, 0.30, 0.18),      # Dark organic dusk
    },
    "Custom": {
        "band_count": 4,
        "gap_count": 2,
        "striation_freq": 160.0,
        "contrast": 1.8,
        "base_density": 2.2,
        "core_boost": 2.5,
        "inner_radius_ratio": 1.25,
        "outer_radius_ratio": 2.35,
        "asymmetry": 0.55,
        "backscatter": -0.50,
        "scatter": 0.5,
        "opacity": 1.0,
        "unlit_factor": 0.5,
        "color_deep": (0.95, 0.95, 0.95),
        "color_mid": (0.85, 0.85, 0.85),
        "color_thin": (0.60, 0.70, 0.80),
    }
}

def generate_procedural_ring_profile(
    seed=42,
    preset="Saturnian Ice",
    band_count=None,
    gap_count=None,
    striation_freq=None,
    contrast=None,
    base_density=None,
    core_boost=None,
    color_deep=None,
    color_mid=None,
    color_thin=None,
    tint_color=(1.0, 1.0, 1.0),
    res=4096
):
    """
    Synthesize a procedural 1D ring radial profile.
    
    Returns:
        img_rgba (PIL.Image): 4096x1 RGBA image
        arr_rgba (np.ndarray): (4096, 4) float32 array in [0.0, 1.0]
        suggested_props (dict): Suggested physical rendering parameters
    """
    cfg = RING_PRESETS.get(preset, RING_PRESETS["Custom"]).copy()
    
    # Override with explicit arguments if provided
    if band_count is not None: cfg["band_count"] = max(1, int(band_count))
    if gap_count is not None: cfg["gap_count"] = max(0, int(gap_count))
    if striation_freq is not None: cfg["striation_freq"] = max(10.0, float(striation_freq))
    if contrast is not None: cfg["contrast"] = max(0.5, float(contrast))
    if base_density is not None: cfg["base_density"] = max(0.01, float(base_density))
    if core_boost is not None: cfg["core_boost"] = max(0.1, float(core_boost))
    if color_deep is not None: cfg["color_deep"] = color_deep
    if color_mid is not None: cfg["color_mid"] = color_mid
    if color_thin is not None: cfg["color_thin"] = color_thin
    
    rng = np.random.RandomState(int(seed) & 0x7fffffff)
    u = np.linspace(0.0, 1.0, res, dtype=np.float32)
    
    # -------------------------------------------------------------
    # 1. Macro Band Partitioning
    # -------------------------------------------------------------
    num_bands = cfg["band_count"]
    # Randomly partition [0.05, 0.95] into num_bands
    cut_points = np.sort(rng.uniform(0.12, 0.88, max(0, num_bands - 1)))
    band_edges = [0.0] + list(cut_points) + [1.0]
    
    macro_density = np.zeros(res, dtype=np.float32)
    core_boost_val = cfg.get("core_boost", 2.5)
    
    for b in range(num_bands):
        b_start = band_edges[b]
        b_end = band_edges[b + 1]
        b_width = b_end - b_start
        
        # Band density factor relative to base_density with distinct structural roles
        if num_bands == 1:
            band_weight = 1.0 * core_boost_val
        elif num_bands == 2:
            band_weight = (0.40 if b == 0 else 1.30 * core_boost_val) * rng.uniform(0.9, 1.1)
        elif num_bands == 3:
            roles = [0.30, 1.40 * core_boost_val, 0.85]
            band_weight = roles[b] * rng.uniform(0.9, 1.1)
        else: # 4 or more bands
            # E.g. Saturn: C-ring (faint), B-ring (dense core), Cassini (gap), A-ring (medium), F-ring (narrow ribbon)
            if b == 0:
                band_weight = 0.28 * rng.uniform(0.85, 1.15) # faint inner (C-ring style)
            elif b == 1:
                band_weight = 1.45 * core_boost_val * rng.uniform(0.95, 1.05) # massive dense core (B-ring style)
            elif b == num_bands - 1 and num_bands >= 5:
                band_weight = 1.60 * rng.uniform(0.9, 1.1) # narrow high-contrast outer ringlet (F-ring style)
            else:
                band_weight = rng.uniform(0.70, 1.15) # medium bands (A-ring style)
            
        edge_softness = min(0.04, b_width * 0.15)
        band_mask = smoothstep(b_start, b_start + edge_softness, u) * (1.0 - smoothstep(b_end - edge_softness, b_end, u))
        
        # Smooth internal density envelope inside the band
        band_center = 0.5 * (b_start + b_end)
        band_shape = 1.0 - np.abs(u - band_center) / (0.5 * b_width)
        band_shape = np.clip(band_shape, 0.0, 1.0)
        band_shape = 0.7 + 0.3 * np.sin(band_shape * np.pi * 0.5)
        
        macro_density += band_mask * band_weight * band_shape
        
    # Overall smooth falloff at the extreme inner and outer boundaries
    boundary_fade = smoothstep(0.0, 0.025, u) * (1.0 - smoothstep(0.975, 1.0, u))
    macro_density *= boundary_fade
    
    # -------------------------------------------------------------
    # 2. Resonant Gaps (Lindblad / Shepherd Moon Resonances)
    # -------------------------------------------------------------
    gap_transmission = np.ones(res, dtype=np.float32)
    num_gaps = cfg["gap_count"]
    
    gap_centers = []
    if num_gaps > 0:
        # Place gaps preferentially near band boundaries or golden ratios
        candidate_pos = []
        for b_cut in cut_points:
            candidate_pos.append(b_cut + rng.uniform(-0.02, 0.02))
        while len(candidate_pos) < num_gaps:
            candidate_pos.append(rng.uniform(0.15, 0.85))
            
        gap_centers = sorted(candidate_pos[:num_gaps])
        
        for g_idx, g_pos in enumerate(gap_centers):
            # First gap (e.g. Cassini-like) is often wide and prominent
            is_major = (g_idx == 0 and num_gaps >= 2)
            g_width = rng.uniform(0.025, 0.055) if is_major else rng.uniform(0.005, 0.015)
            sharpness = rng.uniform(0.15, 0.35) if is_major else rng.uniform(0.25, 0.5)
            
            # Smoothstep notch filter
            dist = np.abs(u - g_pos)
            notch = smoothstep(g_width * 0.5 * (1.0 - sharpness), g_width * 0.5, dist)
            
            # Residual optical depth inside the gap (some gaps have very faint dust)
            gap_residual = 0.01 if is_major else rng.uniform(0.0, 0.08)
            gap_transmission *= np.maximum(gap_residual, notch)
            
    # -------------------------------------------------------------
    # 3. High-Frequency 1D Fractal Striations (Fine Ringlets)
    # -------------------------------------------------------------
    f_base = cfg["striation_freq"]
    striations_raw = fbm1d(u * f_base, octaves=7, lacunarity=2.15, persistence=0.54, seed=seed)
    # Normalize to [0, 1]
    s_min, s_max = striations_raw.min(), striations_raw.max()
    striations = (striations_raw - s_min) / max(1e-6, s_max - s_min)
    
    # Non-linear contrast shaping (creates crisp razor edges and dark crevices)
    contrast_pow = cfg["contrast"]
    striations = np.power(striations, contrast_pow)
    
    # Additional micro-detail octave (ultra-fine ringlets)
    micro_raw = perlin1d(u * (f_base * 4.2), seed=seed + 99991)
    micro = 0.5 + 0.5 * micro_raw
    striations = 0.85 * striations + 0.15 * micro
    
    # -------------------------------------------------------------
    # 4. Spiral Density Wave Chirps (Near Gap Edges)
    # -------------------------------------------------------------
    density_waves = np.zeros(res, dtype=np.float32)
    for g_pos in gap_centers:
        dist_g = u - g_pos
        # Wave radiates outwards from gap edge
        wave_mask = smoothstep(0.0, 0.01, np.abs(dist_g)) * np.exp(-np.abs(dist_g) / 0.05)
        chirp = np.cos(350.0 * np.sign(dist_g) * np.power(np.abs(dist_g), 1.25))
        density_waves += wave_mask * chirp * 0.22
        
    # -------------------------------------------------------------
    # 5. Composite Optical Depth tau(u) & Alpha
    # -------------------------------------------------------------
    base_tau = cfg["base_density"]
    # Modulate macro density with fine striations (0.10 trough to 1.35 peak) and density waves
    striation_mod = 0.10 + 1.25 * striations
    tau = base_tau * macro_density * gap_transmission * striation_mod + density_waves * macro_density
    tau = np.maximum(0.0, tau)
    
    # Physical alpha transmission: alpha = 1.0 - exp(-tau)
    alpha = 1.0 - np.exp(-tau)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    
    # -------------------------------------------------------------
    # 6. Physically Grounded Color Palette Mapping
    # -------------------------------------------------------------
    c_deep = np.array(cfg["color_deep"], dtype=np.float32)
    c_mid = np.array(cfg["color_mid"], dtype=np.float32)
    c_thin = np.array(cfg["color_thin"], dtype=np.float32)
    tint = np.array(tint_color, dtype=np.float32)
    
    # Blend color based on local optical depth
    # Thin/sparse rings show forward-scattering / fine dust color (c_thin)
    # Medium density shows bulk particle albedo (c_mid)
    # Dense regions show multiple-scattering brilliant core (c_deep)
    rgb = np.zeros((res, 3), dtype=np.float32)
    
    weight_mid = smoothstep(0.05, 0.45, tau)[:, np.newaxis]
    weight_deep = smoothstep(0.45, 1.50, tau)[:, np.newaxis]
    
    color_low = (1.0 - weight_mid) * c_thin + weight_mid * c_mid
    color_full = (1.0 - weight_deep) * color_low + weight_deep * c_deep
    
    # Add subtle color variation via low-frequency noise (e.g. radial iron/tholin staining)
    color_var = 0.5 + 0.5 * perlin1d(u * 12.0, seed=seed + 7771)
    color_var_3 = 0.95 + 0.10 * color_var[:, np.newaxis]
    
    rgb = color_full * color_var_3 * tint
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # Pack into (4096, 4) RGBA array
    arr_rgba = np.zeros((res, 4), dtype=np.float32)
    arr_rgba[:, 0:3] = rgb
    arr_rgba[:, 3] = alpha
    
    # Convert to PIL Image (4096, 1) in 8-bit RGBA
    arr_u8 = np.clip(arr_rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # PIL Image from 1D array needs shape (1, 4096, 4)
    img_rgba = Image.fromarray(arr_u8[np.newaxis, :, :], mode='RGBA')
    
    suggested_props = {
        "inner_radius_ratio": cfg["inner_radius_ratio"],
        "outer_radius_ratio": cfg["outer_radius_ratio"],
        "asymmetry": cfg["asymmetry"],
        "backscatter": cfg["backscatter"],
        "scatter": cfg["scatter"],
        "opacity": cfg["opacity"],
        "unlit_factor": cfg["unlit_factor"],
        "core_boost": cfg.get("core_boost", 2.5),
        "raw_color": tuple(tint)
    }
    
    return img_rgba, arr_rgba, suggested_props
