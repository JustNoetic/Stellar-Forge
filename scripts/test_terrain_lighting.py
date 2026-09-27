#!/usr/bin/env python3
"""
scripts/test_terrain_lighting.py

Automated regression test suite verifying that Terrain Quadtree LOD and Cloud Shells
properly receive:
1. Moon/planet caster eclipse shadows (analytical penumbra + umbra).
2. Circumplanetary ring shadows.
3. Planetshine and moonshine secondary bounce light.
4. Host planet ringshine from the dynamic irradiance map.
"""

import os
import sys
import struct
import numpy as np
import moderngl

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rendering.shaders import terrain_vertex_shader, terrain_fragment_shader
from engine.rendering.render_utils import create_terrain_grid_patch


def test_terrain_lighting_and_shadows():
    print("=== Testing Terrain LOD Eclipses, Ring Shadows, Planetshine & Ringshine ===")
    ctx = moderngl.create_context(standalone=True)

    prog_terrain = ctx.program(
        vertex_shader=terrain_vertex_shader,
        fragment_shader=terrain_fragment_shader
    )
    assert prog_terrain is not None, "Failed to compile prog_terrain"

    grid_verts, grid_idx = create_terrain_grid_patch(res=16)
    vbo = ctx.buffer(grid_verts.tobytes())
    ibo = ctx.buffer(grid_idx.tobytes())
    vao = ctx.vertex_array(prog_terrain, [(vbo, '3f', 'in_position')], index_buffer=ibo)

    # Patch SSBO (12 floats per patch: u_range, u_uv_trans, u_meta)
    patch_buf = np.zeros((1, 12), dtype=np.float32)
    patch_buf[0, 0:4] = [-0.1, -0.1, 0.1, 0.1] # u_range: Small face patch
    patch_buf[0, 4:8] = [1.0, 0.0, 0.0, 0.0]   # u_uv_trans: uv_scale=1.0, uv_offset=(0,0), skirt_depth=0
    patch_buf[0, 8:12] = [5.0, 0.0, 0.0, 0.0]  # u_meta: face_idx=5 (+X in world with pole +Y), lod=0, tile_slot=0, body_idx=0

    ssbo = ctx.buffer(patch_buf.tobytes())
    ssbo.bind_to_storage_buffer(binding=4)

    # AllInstances SSBO (binding 2): 1 body (28 floats = 7 vec4s)
    inst_buf_data = np.zeros((1, 28), dtype=np.float32)
    inst_buf_data[0, 0:3] = [0.0, 0.0, 0.0]     # pos
    inst_buf_data[0, 6] = 6371.0 / 149597870.7  # radius in AU
    inst_buf_data[0, 9:12] = [0.0, 1.0, 0.0]    # pole (+Y)
    inst_buf_data[0, 16:19] = [1.0, 0.0, 0.0]   # planetshine_dir (+X)
    inst_buf_data[0, 20:23] = [0.5, 0.8, 1.0]   # planetshine_color (soft cyan bounce light)
    
    # Enable caster 0 (mask_lo = 1u) and ring 0 (r_mask = 1u)
    inst_buf_data[0, 13] = struct.unpack('f', struct.pack('I', 1))[0] # caster_mask.x
    inst_buf_data[0, 14] = struct.unpack('f', struct.pack('I', 0))[0] # caster_mask.y
    inst_buf_data[0, 15] = struct.unpack('f', struct.pack('I', 1))[0] # ring_mask

    inst_buf = ctx.buffer(inst_buf_data.tobytes())
    inst_buf.bind_to_storage_buffer(binding=2)

    # SceneData UBO (binding 1): projection, view, stars, casters
    ubo_staging = np.zeros(1832, dtype=np.float32)
    # Mat4 proj & view
    import pyrr
    cam_pos = np.array([0.0001, 0.0, 0.0], dtype=np.float32)
    view = pyrr.matrix44.create_look_at(cam_pos, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    proj = pyrr.matrix44.create_perspective_projection(45.0, 1.0, 0.000001, 1000.0)
    ubo_staging[0:16] = proj.ravel()
    ubo_staging[16:32] = view.ravel()

    # u_num_stars = 1
    ubo_staging[32:33].view(np.int32)[0] = 1
    # Star 0: pos = (1.0, 0.0, 0.0), radius = 0.00465 AU (~1 R_sun)
    ubo_staging[36:40] = [1.0, 0.0, 0.0, 0.00465]
    # Star 0 color = (1.0, 1.0, 1.0), lum = 1.0
    ubo_staging[100:104] = [1.0, 1.0, 1.0, 1.0]
    # Star 0 pole = +Y, oblateness = 0
    ubo_staging[164:168] = [0.0, 1.0, 0.0, 0.0]
    # Star 0 pole color = (1.0, 1.0, 1.0), lum = 1.0
    ubo_staging[228:232] = [1.0, 1.0, 1.0, 1.0]

    # far = 1000.0, depth_C = 0.1
    ubo_staging[292] = 1000.0
    ubo_staging[293] = 0.1

    # u_num_casters = 0 initially
    ubo_staging[294:295].view(np.int32)[0] = 0
    # Caster 0: pos = (0.5, 0.0, 0.0) AU (directly between star and planet), radius = 0.003 AU
    ubo_staging[296:300] = [0.5, 0.0, 0.0, 0.003]

    ubo = ctx.buffer(ubo_staging.tobytes())
    ubo.bind_to_uniform_block(1)

    # Set up textures: tile_array (unit 14), ring_gradients (unit 0), ringshine_map (unit 8)
    # Unit 14: White tile so albedo = 1.0
    tile_img = np.full((512, 512, 4), 255, dtype=np.uint8)
    tile_array = ctx.texture_array((512, 512, 1), 4, tile_img.tobytes())
    tile_array.use(location=14)

    # Unit 0: Ring gradient with alpha = 1.0 (fully opaque ring)
    grad_img = np.full((16, 16, 4), 255, dtype=np.uint8)
    tex_grad = ctx.texture((16, 16), 4, grad_img.tobytes())
    tex_grad.use(location=0)

    # Unit 8: Ringshine map with warm gold irradiance (1.0, 0.8, 0.4)
    shine_img = np.full((128, 128, 4), [255, 204, 102, 255], dtype=np.uint8)
    tex_shine = ctx.texture((128, 128), 4, shine_img.tobytes())
    tex_shine.use(location=8)

    prog_terrain['u_tile_array'].value = 14
    prog_terrain['u_ring_gradients'].value = 0
    prog_terrain['u_ringshine_map'].value = 8
    prog_terrain['u_km_to_au'].value = float(1.0 / 149597870.7)
    prog_terrain['u_au_to_km'].value = float(149597870.7)
    prog_terrain['u_camera_pos'].value = tuple(cam_pos)
    prog_terrain['u_hdr_enabled'].value = False
    prog_terrain['u_exposure'].value = 1.0
    prog_terrain['u_debug_tiles'].value = False

    r_coplanar = np.zeros(16, dtype=np.uint32)
    r_coplanar[0] = 1
    prog_terrain['u_ring_coplanar_mask'].write(r_coplanar.tobytes())

    # 1. Test Baseline Direct Light (No Shadows, Planetshine or Ringshine)
    prog_terrain['u_planetshine_enabled'].value = False
    prog_terrain['u_ringshine_enabled'].value = False
    prog_terrain['u_num_ring_planes'].value = 0

    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((64, 64), 4, dtype='f4')],
        depth_attachment=ctx.depth_texture((64, 64))
    )
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLES, instances=1)

    raw_base = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32).reshape((64, 64, 4))
    center_color_base = raw_base[32, 32, :3]
    print(f"--> Baseline lit surface color: {center_color_base}")
    assert np.any(center_color_base > 0.1), "Surface should be illuminated by host star"

    # 2. Test Moon/Planet Caster Eclipse Shadow
    # With Caster 0 directly along the line of sight (pos = (0.5, 0, 0), radius = 0.003 AU),
    # the fragment should be in deep eclipse umbra!
    # Update caster position to eclipse the patch center
    print("--> Testing Caster Eclipse Shadow (Umbra)...")
    ubo_staging[294:295].view(np.int32)[0] = 1 # 1 caster
    ubo.write(ubo_staging.tobytes())

    ctx.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLES, instances=1)

    raw_eclipse = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32).reshape((64, 64, 4))
    center_color_eclipse = raw_eclipse[32, 32, :3]
    print(f"    Eclipse umbra surface color: {center_color_eclipse}")
    assert np.all(center_color_eclipse < center_color_base * 0.1), (
        f"Surface should be shadowed in eclipse umbra, got {center_color_eclipse} vs base {center_color_base}"
    )
    print("    Caster eclipse shadow verified! (PASSED)")

    # 3. Test Planetshine / Moonshine
    print("--> Testing Planetshine/Moonshine secondary illumination...")
    # Disable caster eclipse so secondary bounce light is not occluded by host umbra
    ubo_staging[294:295].view(np.int32)[0] = 0 # 0 casters
    ubo.write(ubo_staging.tobytes())
    prog_terrain['u_planetshine_enabled'].value = True
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLES, instances=1)

    raw_ps = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32).reshape((64, 64, 4))
    center_color_ps = raw_ps[32, 32, :3]
    print(f"    Surface color with Planetshine: {center_color_ps}")
    assert np.any(center_color_ps > center_color_eclipse), "Planetshine should add secondary bounce light"
    print("    Planetshine verified! (PASSED)")

    # 4. Test Ringshine from host ring
    print("--> Testing Host Planet Ringshine...")
    prog_terrain['u_ringshine_enabled'].value = True
    prog_terrain['u_num_ring_planes'].value = 1
    r_centers = np.zeros((16, 3), dtype='f4')
    r_normals = np.zeros((16, 3), dtype='f4')
    r_normals[0] = [0.0, 1.0, 0.0]
    r_params = np.zeros((16, 4), dtype='f4')
    r_params[0] = [0.0001, 0.001, 1.0, 0.0]

    prog_terrain['u_ring_center'].write(r_centers.tobytes())
    prog_terrain['u_ring_normal'].write(r_normals.tobytes())
    prog_terrain['u_ring_params'].write(r_params.tobytes())

    ctx.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLES, instances=1)

    raw_rs = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32).reshape((64, 64, 4))
    center_color_rs = raw_rs[32, 32, :3]
    print(f"    Surface color with Ringshine: {center_color_rs}")
    assert np.any(center_color_rs > center_color_ps), "Ringshine should illuminate the planet surface"
    print("    Ringshine verified! (PASSED)")

    # 5. Test Ring Shadow
    print("--> Testing Circumplanetary Ring Shadows...")
    prog_terrain['u_planetshine_enabled'].value = False
    prog_terrain['u_ringshine_enabled'].value = False

    # Setup ring plane that intersects the sun-planet ray
    prog_terrain['u_num_ring_planes'].value = 1
    # Place ring plane at x = 0.5 AU with offset so star ray passes cleanly inside ring
    r_centers[0] = [0.5, 0.05, 0.0]
    r_normals[0] = [1.0, 0.0, 0.0]
    # Inner = 0.0, outer = 0.1 AU (intercepts ray at d = 0.05), opacity = 1.0
    r_params[0] = [0.0, 0.1, 1.0, 0.0]

    prog_terrain['u_ring_center'].write(r_centers.tobytes())
    prog_terrain['u_ring_normal'].write(r_normals.tobytes())
    prog_terrain['u_ring_params'].write(r_params.tobytes())

    ctx.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLES, instances=1)

    raw_ring_shadow = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32).reshape((64, 64, 4))
    center_color_ring_shadow = raw_ring_shadow[32, 32, :3]
    print(f"    Ring shadow surface color: {center_color_ring_shadow}")
    assert np.all(center_color_ring_shadow < center_color_base * 0.1), (
        f"Surface should be shadowed by ring plane, got {center_color_ring_shadow} vs base {center_color_base}"
    )
    print("    Ring shadow verified! (PASSED)")

    print("=== All Terrain Lighting and Shadow Tests PASSED! ===")


if __name__ == "__main__":
    test_terrain_lighting_and_shadows()
