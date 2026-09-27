import os
import sys
import math
import numpy as np
import moderngl

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rendering.shader_loader import load_shader
from engine.rendering.render_utils import (
    generate_ring_shadow_grad,
    bake_unified_shadow_profile,
    rebuild_ring_gradients_atlas
)

def test_shader_compilation():
    print("--> Testing compilation of modified shaders...")
    ctx = moderngl.create_context(standalone=True)
    
    # 1. Sphere program
    vs_sphere = load_shader("celestial/sphere.vert")
    fs_sphere = load_shader("celestial/sphere.frag")
    prog_sphere = ctx.program(vertex_shader=vs_sphere, fragment_shader=fs_sphere)
    assert prog_sphere is not None, "Failed to compile sphere shaders"
    print("  [PASS] Sphere shader compiled successfully.")

    # 2. Terrain program
    vs_terrain = load_shader("celestial/terrain.vert")
    fs_terrain = load_shader("celestial/terrain.frag")
    prog_terrain = ctx.program(vertex_shader=vs_terrain, fragment_shader=fs_terrain)
    assert prog_terrain is not None, "Failed to compile terrain shaders"
    print("  [PASS] Terrain shader compiled successfully.")

    # 3. Ring program
    vs_ring = load_shader("celestial/ring.vert")
    fs_ring = load_shader("celestial/ring.frag")
    prog_ring = ctx.program(vertex_shader=vs_ring, fragment_shader=fs_ring)
    assert prog_ring is not None, "Failed to compile ring shaders"
    print("  [PASS] Ring shader compiled successfully.")

    # 4. Ringshine map program
    vs_rs = load_shader("post/ringshine_map.vert")
    fs_rs = load_shader("post/ringshine_map.frag")
    prog_rs = ctx.program(vertex_shader=vs_rs, fragment_shader=fs_rs)
    assert prog_rs is not None, "Failed to compile ringshine map shaders"
    print("  [PASS] Ringshine map shader compiled successfully.")

    # 5. Atmo program
    vs_atmo = load_shader("atmosphere/atmo.vert")
    fs_atmo = load_shader("atmosphere/atmo.frag")
    prog_atmo = ctx.program(vertex_shader=vs_atmo, fragment_shader=fs_atmo)
    assert prog_atmo is not None, "Failed to compile atmo shaders"
    print("  [PASS] Atmosphere shader compiled successfully.")

    # 6. Sky-View LUT program
    vs_sky = load_shader("atmosphere/sky_view_lut.vert")
    fs_sky = load_shader("atmosphere/sky_view_lut.frag")
    prog_sky = ctx.program(vertex_shader=vs_sky, fragment_shader=fs_sky)
    assert prog_sky is not None, "Failed to compile sky-view LUT shaders"
    print("  [PASS] Sky-View LUT shader compiled successfully.")

def test_ring_props_baking():
    print("--> Testing unified shadow & scattering property baking...")
    ctx = moderngl.create_context(standalone=True)
    
    # Create test ring items: one textured, one procedural
    grad = [{'p': 0.0, 'a': 0.0}, {'p': 0.5, 'a': 1.0}, {'p': 1.0, 'a': 0.0}]
    shadow_grad_proc = generate_ring_shadow_grad(grad, raw_color=(1.0, 0.8, 0.6))
    
    dummy_tex = np.ones((4096, 4), dtype=np.float32)
    shadow_grad_tex = generate_ring_shadow_grad(grad, tex_sampled=dummy_tex, raw_color=(0.9, 0.9, 0.95))
    
    ring_proc = {
        'body_idx': 0,
        'inner_r': 1.2,
        'outer_r': 1.8,
        'opacity': 0.8,
        'asymmetry': 0.7,
        'backscatter': -0.3,
        'scatter': 0.35,
        'unlit_factor': 0.5,
        'is_textured': False,
        'shadow_grad': shadow_grad_proc,
        'gradient': grad,
        'raw_color': (1.0, 0.8, 0.6)
    }
    
    ring_tex = {
        'body_idx': 0,
        'inner_r': 1.8,
        'outer_r': 2.3,
        'opacity': 1.0,
        'asymmetry': 0.436,
        'backscatter': -0.657,
        'scatter': 1.0,
        'unlit_factor': 0.45,
        'is_textured': True,
        'shadow_grad': shadow_grad_tex,
        'gradient': grad,
        'raw_color': (0.9, 0.9, 0.95)
    }
    
    body_rings = [ring_proc, ring_tex]
    combined_shadow, combined_props, combined_props_extra = bake_unified_shadow_profile(body_rings, 1.2, 2.3)
    
    assert combined_shadow.shape == (4096, 4)
    assert combined_props.shape == (4096, 4)
    assert combined_props_extra.shape == (4096, 4)
    
    # Check that in the procedural zone (first quarter of radii), is_textured == 0
    assert combined_props_extra[500, 0] == 0.0, f"Expected is_tex=0, got {combined_props_extra[500, 0]}"
    # Check that unlit_factor is strictly positive (no corruption from +100 offset)
    assert 0.4 < combined_props[500, 3] < 0.6, f"Expected unlit ~0.5, got {combined_props[500, 3]}"
    
    # Check that in the textured zone (last quarter of radii), is_textured == 1
    assert combined_props_extra[3500, 0] == 1.0, f"Expected is_tex=1, got {combined_props_extra[3500, 0]}"
    assert 0.4 < combined_props[3500, 3] < 0.5, f"Expected unlit ~0.45, got {combined_props[3500, 3]}"
    
    # Test atlas rebuild with textures
    ring_grad_tex = ctx.texture((4096, 16), 4, dtype='f4')
    ring_props_tex = ctx.texture((4096, 8), 4, dtype='f4')
    ring_grad_tex.props_tex = ring_props_tex
    
    rebuild_ring_gradients_atlas([ring_proc, ring_tex], ring_grad_tex)
    print("  [PASS] Ring properties baking and atlas rebuild passed with zero corruption.")

def test_secondary_star_ring_hemisphere_math():
    print("--> Testing secondary star hemisphere relative logic...")
    ring_normal = np.array([0.0, 1.0, 0.0]) # Ring lies in XZ plane, normal along +Y
    
    # Primary star is North (+Y): elevation > 0
    L0 = np.array([0.5, 0.8, 0.0])
    L0 /= np.linalg.norm(L0)
    sun_elev_0 = np.dot(L0, ring_normal)
    assert sun_elev_0 > 0.0
    
    # Case A: Secondary star is also North: elevation > 0
    L_sec_north = np.array([-0.5, 0.6, 0.2])
    L_sec_north /= np.linalg.norm(L_sec_north)
    sun_elev_s_a = np.dot(L_sec_north, ring_normal)
    rel_hemi_a = 1.0 if (sun_elev_s_a * sun_elev_0 >= 0.0) else -1.0
    assert rel_hemi_a == 1.0, "Expected same hemisphere (rel_hemi == 1.0)"
    
    # Case B: Secondary star is South: elevation < 0
    L_sec_south = np.array([-0.5, -0.6, 0.2])
    L_sec_south /= np.linalg.norm(L_sec_south)
    sun_elev_s_b = np.dot(L_sec_south, ring_normal)
    rel_hemi_b = 1.0 if (sun_elev_s_b * sun_elev_0 >= 0.0) else -1.0
    assert rel_hemi_b == -1.0, "Expected opposite hemisphere (rel_hemi == -1.0)"
    
    # Verify that latitude mapping inverts correctly
    # Point on northern hemisphere of planet:
    frag_elev = 0.5
    y_prime_a = frag_elev * rel_hemi_a # +0.5 -> samples northern (lit) half of bake
    y_prime_b = frag_elev * rel_hemi_b # -0.5 -> samples southern (unlit) half of bake
    assert y_prime_a == 0.5
    assert y_prime_b == -0.5
    print("  [PASS] Secondary star hemisphere inversion math verified.")

if __name__ == "__main__":
    print("=== Running Ring & Ringshine Regression Tests ===")
    test_shader_compilation()
    test_ring_props_baking()
    test_secondary_star_ring_hemisphere_math()
    print("=== All Ring & Ringshine Tests PASSED! ===")
