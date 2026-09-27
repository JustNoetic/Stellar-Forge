#!/usr/bin/env python3
"""
scripts/test_terrain_lod.py

Automated integration test for the SpaceEngine-style CubeSphereQuadtree
terrain LOD and tiled streaming system.
"""

import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import moderngl
from engine.rendering.terrain_quadtree import PlanetQuadtree, cube_to_sphere_point
from engine.rendering.terrain_streamer import TerrainTileStreamer
from engine.rendering.render_utils import create_terrain_grid_patch
from engine.rendering.shaders import terrain_vertex_shader, terrain_fragment_shader


def test_cube_to_sphere_continuity():
    print("--> Testing cube-to-sphere face continuity...")
    # Check that adjacent face edges match continuously
    # Face 4 (+Z) right edge (u=1, v=0) vs Face 0 (+X) left edge (u=-1, v=0)
    p_4_right = cube_to_sphere_point(4, 1.0, 0.0)
    p_0_left = cube_to_sphere_point(0, -1.0, 0.0)
    diff = np.linalg.norm(p_4_right - p_0_left)
    assert diff < 1e-5, f"Face seam mismatch between +Z and +X: {diff}"
    print(f"    Seam difference: {diff:.8f} (PASSED)")


def test_quadtree_subdivision():
    print("--> Testing quadtree distance-based subdivision...")
    radius_km = 1737.4  # Moon radius
    quadtree = PlanetQuadtree(radius_km=radius_km, max_lod=4, split_factor=1.25)

    # 1. Distant camera (100,000 km) -> should only have 6 root patches (or fewer due to horizon culling)
    cam_far = np.array([0.0, 0.0, 100000.0], dtype=np.float32)
    patches_far = quadtree.traverse(cam_far, fov_deg=45.0, screen_height=1080.0, max_lod=4)
    print(f"    Patches at 100,000 km: {len(patches_far)} (Expected <= 6)")
    assert len(patches_far) <= 6

    # 2. Medium distance (10,000 km) -> should subdivide
    cam_med = np.array([0.0, 0.0, 10000.0], dtype=np.float32)
    patches_med = quadtree.traverse(cam_med, fov_deg=45.0, screen_height=1080.0, max_lod=4)
    print(f"    Patches at 10,000 km: {len(patches_med)} (Subdivided)")
    assert len(patches_med) > len(patches_far)

    # 3. Near surface (radius + 50 km) -> should subdivide to max_lod near the camera
    cam_near = np.array([0.0, 0.0, radius_km + 50.0], dtype=np.float32)
    patches_near = quadtree.traverse(cam_near, fov_deg=45.0, screen_height=1080.0, max_lod=4)
    max_lod_found = max(p.lod for p in patches_near)
    print(f"    Patches at 50 km altitude: {len(patches_near)} (Max LOD reached: {max_lod_found})")
    assert max_lod_found == 4, f"Expected max LOD 4, got {max_lod_found}"


def test_tiled_streamer_and_fallback():
    print("--> Testing TerrainTileStreamer and hierarchical ancestor fallback...")
    ctx = moderngl.create_context(standalone=True)
    streamer = TerrainTileStreamer(ctx, tiles_base_dir="data/tiles", pool_capacity=64, tile_size=512)

    # Check root preloading (Moon face 0, lod 0)
    key_root = ("moon", "diffuse", 0, 0, 0, 0)
    slot, uv_scale, off_x, off_y = streamer.get_tile_slot_or_fallback("Moon", "diffuse", 0, 0, 0, 0)
    print(f"    Root tile slot: {slot}, uv_scale: {uv_scale}")
    assert uv_scale == 1.0

    # Query a high LOD tile that may not be resident yet (e.g. Moon face 0, LOD 4, tile (5, 5))
    # It must fall back to resident ancestor with scaled UVs and non-negative slot!
    slot_anc, uv_s, ox, oy = streamer.get_tile_slot_or_fallback("Moon", "diffuse", 0, 4, 5, 5)
    print(f"    Child (LOD 4) fallback slot: {slot_anc}, scale: {uv_s}, offset: ({ox}, {oy})")
    assert slot_anc >= 0
    assert 0.0 < uv_s <= 1.0

    streamer.shutdown()


def test_gpu_terrain_rendering_pipeline():
    print("--> Testing GPU terrain shader and instanced rendering...")
    ctx = moderngl.create_context(standalone=True)

    prog_terrain = ctx.program(
        vertex_shader=terrain_vertex_shader,
        fragment_shader=terrain_fragment_shader
    )
    assert prog_terrain is not None

    grid_verts, grid_idx = create_terrain_grid_patch(res=32)
    vbo = ctx.buffer(grid_verts.tobytes())
    ibo = ctx.buffer(grid_idx.tobytes())
    vao = ctx.vertex_array(prog_terrain, [(vbo, '3f', 'in_position')], index_buffer=ibo)

    # Patch SSBO (12 floats per patch: range[4], uv_trans[4], meta[4])
    patch_buf = np.zeros((4, 12), dtype=np.float32)
    for i in range(4):
        patch_buf[i, 0:4] = [-1.0, -1.0, 1.0, 1.0]
        patch_buf[i, 4:8] = [1.0, 0.0, 0.0, 10.0]
        patch_buf[i, 8:12] = [float(i), 0.0, 0.0, 0.0]

    ssbo = ctx.buffer(patch_buf.tobytes())
    ssbo.bind_to_storage_buffer(binding=4)

    # AllInstances SSBO (binding 2) - mock 1 body
    inst_data = np.zeros((1, 28), dtype=np.float32)
    inst_data[0, 6] = 1737.4 / 149597870.7  # radius in AU
    inst_data[0, 10] = 1.0  # pole Y
    inst_buf = ctx.buffer(inst_data.tobytes())
    inst_buf.bind_to_storage_buffer(binding=2)

    # SceneData UBO (binding 1)
    ubo = ctx.buffer(reserve=7328)
    ubo.bind_to_uniform_block(1)

    # Textures (0: ring gradient, 8: ringshine map, 14: tile array)
    tex_grad = ctx.texture((16, 16), 4)
    tex_grad.use(location=0)
    tex_shine = ctx.texture((16, 16), 4)
    tex_shine.use(location=8)
    tex_array = ctx.texture_array((64, 64, 4), 4)
    tex_array.use(location=14)

    if 'u_tile_array' in prog_terrain:
        prog_terrain['u_tile_array'].value = 14
    if 'u_ring_gradients' in prog_terrain:
        prog_terrain['u_ring_gradients'].value = 0
    if 'u_ringshine_map' in prog_terrain:
        prog_terrain['u_ringshine_map'].value = 8

    # Test rendering offscreen
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((512, 512), 4)],
        depth_attachment=ctx.depth_texture((512, 512))
    )
    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST)

    # Set debug tiles mode
    if 'u_debug_tiles' in prog_terrain:
        prog_terrain['u_debug_tiles'].value = True
    if 'u_exposure' in prog_terrain:
        prog_terrain['u_exposure'].value = 1.0

    # Draw instanced (direct)
    vao.render(moderngl.TRIANGLES, instances=4)

    # Draw instanced (indirect via DrawElementsIndirectCommand)
    cmd_data = np.array([len(grid_idx), 4, 0, 0, 0], dtype=np.uint32)
    cmd_buf = ctx.buffer(cmd_data.tobytes())
    vao.render_indirect(cmd_buf, moderngl.TRIANGLES)
    print("    Offscreen terrain render pass (direct & indirect) completed without OpenGL errors!")


def test_quadtree_frustum_culling_and_low_fov():
    print("--> Testing quadtree view frustum culling at low FOV...")
    import pyrr
    from engine.rendering.render_utils import extract_frustum_planes

    radius_km = 1737.4
    cam_dist_km = 10000.0
    cam_pos_km = np.array([0.0, 0.0, cam_dist_km], dtype=np.float32)

    # 1.0 degree FOV camera looking at [0, 0, 0]
    view = pyrr.matrix44.create_look_at(cam_pos_km, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    proj = pyrr.matrix44.create_perspective_projection(1.0, 1.0, 1.0, 100000.0)
    vp = view @ proj
    planes = extract_frustum_planes(vp).astype(np.float32)

    q = PlanetQuadtree(radius_km=radius_km, max_lod=6, split_factor=1.25)

    p_no_frustum = q.traverse_raw(cam_pos_km, fov_deg=1.0, screen_height=1080.0, max_lod=6)
    p_with_frustum = q.traverse_raw(cam_pos_km, fov_deg=1.0, screen_height=1080.0, max_lod=6, frustum_planes=planes)

    print(f"    Patches at FOV 1.0 WITHOUT frustum culling: {len(p_no_frustum)}")
    print(f"    Patches at FOV 1.0 WITH frustum culling:    {len(p_with_frustum)}")

    assert len(p_with_frustum) < len(p_no_frustum) // 10, f"Expected >90% reduction, got {len(p_with_frustum)} vs {len(p_no_frustum)}"
    assert len(p_with_frustum) > 0, "Visible patches should not be zero"
    print("    Frustum culling successfully pruned off-screen quadtree nodes! (PASSED)")


def main():
    print("=== Running Terrain Quadtree LOD & Tiled Streaming Tests ===")
    test_cube_to_sphere_continuity()
    test_quadtree_subdivision()
    test_quadtree_frustum_culling_and_low_fov()
    test_tiled_streamer_and_fallback()
    test_gpu_terrain_rendering_pipeline()
    print("=== All Terrain LOD Tests PASSED Successfully! ===")


if __name__ == "__main__":
    main()
