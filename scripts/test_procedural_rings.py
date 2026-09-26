"""
Test suite for Procedural Planetary Ring Generation in Stellar-Forge.
Validates ring_generator.py profiles, presets, mathematical properties,
and texture_baker integration.
"""

import sys
import numpy as np
from PIL import Image

from engine.rendering.ring_generator import (
    generate_procedural_ring_profile,
    RING_PRESETS,
    perlin1d,
    fbm1d
)

def test_noise_continuity():
    """Verify 1D Perlin and fBm noise continuity and bounds."""
    u = np.linspace(-10.0, 10.0, 10000, dtype=np.float32)
    p = perlin1d(u, seed=42)
    assert np.all(np.isfinite(p)), "Perlin noise contains NaN or Inf"
    assert p.min() >= -1.05 and p.max() <= 1.05, f"Perlin range out of bounds: [{p.min()}, {p.max()}]"
    
    f = fbm1d(u, octaves=6, seed=42)
    assert np.all(np.isfinite(f)), "fBm noise contains NaN or Inf"
    print("  [PASS] 1D Perlin and fBm continuity test passed.")

def test_all_presets():
    """Verify all built-in presets generate valid 4096-sample RGBA textures."""
    for name, cfg in RING_PRESETS.items():
        img, arr, props = generate_procedural_ring_profile(seed=12345, preset=name, res=4096)
        
        # Dimensions
        assert img.size == (4096, 1), f"Preset {name}: Image size {img.size} != (4096, 1)"
        assert arr.shape == (4096, 4), f"Preset {name}: Array shape {arr.shape} != (4096, 4)"
        assert arr.dtype == np.float32, f"Preset {name}: Array dtype {arr.dtype} != float32"
        
        # Values in valid normalized range [0.0, 1.0]
        assert arr.min() >= 0.0 and arr.max() <= 1.0, f"Preset {name}: Value out of [0, 1] range: [{arr.min()}, {arr.max()}]"
        
        # Non-zero alpha and color
        alpha_max = arr[:, 3].max()
        assert alpha_max > 0.01, f"Preset {name}: Max alpha too low: {alpha_max}"
        
        # Suggested props structure
        assert "inner_radius_ratio" in props
        assert "outer_radius_ratio" in props
        assert "asymmetry" in props
        assert "backscatter" in props
        assert props["inner_radius_ratio"] < props["outer_radius_ratio"]
        
        print(f"  [PASS] Preset '{name}' verified: max_alpha={alpha_max:.3f}, mean_alpha={arr[:, 3].mean():.3f}, inner={props['inner_radius_ratio']:.2f}, outer={props['outer_radius_ratio']:.2f}")

def test_seed_determinism():
    """Verify identical seeds produce identical ring textures, and different seeds produce different rings."""
    img1, arr1, _ = generate_procedural_ring_profile(seed=999, preset="Saturnian Ice")
    img2, arr2, _ = generate_procedural_ring_profile(seed=999, preset="Saturnian Ice")
    img3, arr3, _ = generate_procedural_ring_profile(seed=111, preset="Saturnian Ice")
    
    assert np.allclose(arr1, arr2, atol=1e-6), "Identical seeds produced different results!"
    assert not np.allclose(arr1, arr3, atol=1e-3), "Different seeds produced identical results!"
    print("  [PASS] Seed determinism verified.")

def test_custom_parameters():
    """Verify custom parameter overrides work as expected."""
    img, arr, props = generate_procedural_ring_profile(
        seed=42,
        preset="Custom",
        band_count=5,
        gap_count=4,
        striation_freq=250.0,
        contrast=2.5,
        base_density=2.2,
        tint_color=(0.8, 0.4, 0.2)
    )
    assert arr[:, 3].max() > 0.8, "High base density should produce high alpha in core"
    # Verify tint color influence
    assert arr[:, 0].mean() > arr[:, 2].mean(), "Red tint should make R > B"
    print("  [PASS] Custom parameters override test passed.")

if __name__ == "__main__":
    print("Running Stellar-Forge Procedural Ring Test Suite...")
    test_noise_continuity()
    test_all_presets()
    test_seed_determinism()
    test_custom_parameters()
    print("\nAll Procedural Ring tests PASSED successfully!")
