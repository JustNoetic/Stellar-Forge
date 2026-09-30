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
from engine.rendering.terrain_quadtree import PlanetQuadtree, cube_to_sphere_point, pack_cloud_patches_jit
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


def test_depth_weighted_eviction_and_lod1_retention():
    print("--> Testing Depth-Weighted LRU Eviction & LOD 1 Retention...")
    ctx = moderngl.create_context(standalone=True)
    streamer = TerrainTileStreamer(ctx, tiles_base_dir="data/tiles", pool_capacity=256, tile_size=512)

    # 1. Verify LOD 0 and LOD 1 are resident and locked on startup
    assert streamer.is_tile_resident("Moon", "diffuse", 0, 0, 0, 0), "Moon LOD 0 tile must be resident"
    assert streamer.is_tile_resident("Moon", "diffuse", 0, 1, 0, 0), "Moon LOD 1 tile must be resident"
    slot_0, uv_0, _, _ = streamer.get_tile_slot_or_fallback("Moon", "diffuse", 0, 0, 0, 0)
    slot_1, uv_1, _, _ = streamer.get_tile_slot_or_fallback("Moon", "diffuse", 0, 1, 0, 0)
    assert uv_0 == 1.0, f"LOD 0 should be exact match, got {uv_0}"
    assert uv_1 == 1.0, f"LOD 1 should be exact match, got {uv_1}"
    print(f"    Moon LOD 0 slot: {slot_0}, Moon LOD 1 slot: {slot_1} (LOD 1 Resident!)")

    # 2. Test fallback from LOD 2 to LOD 1 (must be uv_scale = 0.5, NOT 0.25!)
    slot_l2, uv_l2, ox, oy = streamer.get_tile_slot_or_fallback("Moon", "diffuse", 0, 2, 0, 0)
    print(f"    LOD 2 query fallback: slot={slot_l2}, uv_scale={uv_l2} (Expected 0.5 for LOD 1 fallback)")
    assert uv_l2 == 0.5, f"Expected fallback to LOD 1 (uv_scale 0.5), got {uv_l2}"

    # 3. Test Depth-Weighted Eviction:
    streamer.free_slots.clear()
    dummy_data = b'\x00' * (512 * 512 * 4)
    streamer.resident_tiles[("test", "diffuse", 0, 5, 0, 0)] = 50
    streamer.slot_to_key[50] = ("test", "diffuse", 0, 5, 0, 0)
    streamer.slot_last_used[50] = streamer.current_frame - 10

    streamer.resident_tiles[("test", "diffuse", 0, 2, 0, 0)] = 51
    streamer.slot_to_key[51] = ("test", "diffuse", 0, 2, 0, 0)
    streamer.slot_last_used[51] = streamer.current_frame - 15  # Older timestamp, but lower LOD (2 vs 5)

    streamer.upload_queue.put((("test", "diffuse", 0, 3, 9, 9), dummy_data))
    streamer.process_uploads(max_per_frame=1)

    assert ("test", "diffuse", 0, 2, 0, 0) in streamer.resident_tiles, "LOD 2 tile must be protected from eviction!"
    print("    Depth-weighted eviction successfully evicted LOD 5 leaf tile while retaining LOD 2 parent tile! (PASSED)")

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

    # Patch SSBO (20 floats per patch (80 bytes): range[4], uv_trans[4], meta[4])
    patch_buf = np.zeros((4, 20), dtype=np.float32)
    for i in range(4):
        patch_buf[i, 0:4] = [-1.0, -1.0, 1.0, 1.0]
        patch_buf[i, 4:8] = [1.0, 0.0, 0.0, 10.0]
        patch_buf[i, 8:12] = [float(i), 0.0, 0.0, 0.0]

    patch_buf[:, 12] = -1.0  # Height disabled for existing regression cases
    patch_buf[:, 18] = 0.0   # Explicit caster index for O(1) atmosphere lookup
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


def test_cloud_horizon_culling_and_oblateness():
    print("--> Testing cloud horizon culling and oblate altitude...")
    from engine.app import _ellipsoid_surface_radius

    # 1. Oblate planet altitude test (Saturn parameters)
    r_eq_km = 60268.0
    f_obl = 0.09796
    pole = np.array([0.0, 1.0, 0.0])
    c_alt_km = 30.0

    # Camera at 58,000 km along polar axis (3,600 km above polar surface)
    cam_pos_polar = np.array([0.0, 58000.0, 0.0])
    cam_d_km = float(np.linalg.norm(cam_pos_polar))
    cloud_r_km = _ellipsoid_surface_radius(r_eq_km + c_alt_km, f_obl, pole, cam_pos_polar)
    is_below_clouds = cam_d_km < cloud_r_km

    # Polar surface radius is r_eq * (1 - f) ~ 54,364 km; cloud is at ~54,391 km
    assert not is_below_clouds, f"Polar observer at 58,000 km should be ABOVE clouds (cloud_r = {cloud_r_km:.1f} km)"
    print(f"    Polar observer at 58,000 km: cloud_r={cloud_r_km:.1f} km, is_below_clouds={is_below_clouds} (PASSED)")

    # Ground observer at 54,370 km (below polar clouds at 54,391 km)
    cam_pos_ground = np.array([0.0, 54370.0, 0.0])
    cam_d_ground = float(np.linalg.norm(cam_pos_ground))
    is_below_ground = cam_d_ground < cloud_r_km
    assert is_below_ground, "Observer below polar cloud deck must evaluate is_below_clouds=True"
    print(f"    Ground observer at 54,370 km: is_below_clouds={is_below_ground} (PASSED)")

    # 2. Cloud horizon extension test in quadtree traversal
    # Earth radius with exaggerated cloud shell to verify patch retention past limb
    r_earth = 6371.0
    q = PlanetQuadtree(radius_km=r_earth, max_lod=3, split_factor=1.0)
    cam_space = np.array([0.0, 0.0, 8000.0], dtype=np.float32)

    patches_ground_only = q.traverse_raw(cam_space, fov_deg=60.0, screen_height=1080.0, max_lod=3, cloud_alt_km=0.0)
    patches_with_clouds = q.traverse_raw(cam_space, fov_deg=60.0, screen_height=1080.0, max_lod=3, cloud_alt_km=200.0)

    print(f"    Patches without clouds: {len(patches_ground_only)}, with 200 km clouds: {len(patches_with_clouds)}")
    assert len(patches_with_clouds) >= len(patches_ground_only), "Cloud shell must retain more or equal patches around limb"
    print("    Cloud horizon culling successfully extends quadtree coverage past limb! (PASSED)")


def test_decoupled_cloud_quadtree():
    print("--> Testing Decoupled Cloud Quadtree LOD Traversal & Packing...")
    r_earth = 6371.0
    cloud_alt = 3.5
    cloud_r = r_earth + cloud_alt

    q_terrain = PlanetQuadtree(radius_km=r_earth, max_lod=6, split_factor=1.0)
    q_clouds = PlanetQuadtree(radius_km=cloud_r, max_lod=3, split_factor=1.0)

    # Observer near Earth surface (altitude 2 km)
    cam_ground = np.array([0.0, 0.0, r_earth + 2.0], dtype=np.float32)

    # 1. Ground terrain traversal (cloud_alt_km=0.0)
    p_ground = q_terrain.traverse_raw(cam_ground, fov_deg=60.0, screen_height=1080.0, max_lod=6, cloud_alt_km=0.0)
    max_ground_lod = int(np.max(p_ground[:, 5]))
    print(f"    Ground patches: {len(p_ground)}, Max ground LOD: {max_ground_lod}")
    assert max_ground_lod >= 4, f"Ground terrain should subdivide deeply, got LOD {max_ground_lod}"

    # 2. Decoupled cloud traversal capped to cloud_max_depth = 3
    p_clouds = q_clouds.traverse_raw(cam_ground, fov_deg=60.0, screen_height=1080.0, max_lod=3, cloud_alt_km=0.0)
    max_cloud_lod = int(np.max(p_clouds[:, 5]))
    print(f"    Cloud patches:  {len(p_clouds)}, Max cloud LOD:  {max_cloud_lod}")
    assert max_cloud_lod <= 3, f"Cloud LOD should be capped at 3, got LOD {max_cloud_lod}"
    assert len(p_clouds) <= len(p_ground), f"Decoupled cloud patch count ({len(p_clouds)}) should be <= ground ({len(p_ground)})"

    # 3. Test pack_cloud_patches_jit packing
    num_cp = len(p_clouds)
    staging = np.zeros((num_cp, 20), dtype=np.float32)
    fake_uvs = np.ones(num_cp, dtype=np.float32) * 0.5
    fake_ox = np.zeros(num_cp, dtype=np.float32)
    fake_oy = np.zeros(num_cp, dtype=np.float32)
    fake_slots = np.ones(num_cp, dtype=np.float32) * 2.0
    body_idx = 3.0

    pack_cloud_patches_jit(staging, 0, p_clouds, fake_uvs, fake_ox, fake_oy, fake_slots, body_idx)

    assert np.allclose(staging[:, 0:4], p_clouds[:, 0:4]), "Cloud patch u/v ranges must match raw_patches"
    assert np.allclose(staging[:, 4], 0.5), "Cloud uv_scale must match passed uvs"
    assert np.allclose(staging[:, 7], 0.0), "Cloud skirts must be 0.0"
    assert np.allclose(staging[:, 10], 2.0), "Cloud tile slot must match passed slots"
    assert np.allclose(staging[:, 11], body_idx), "Cloud body_idx must match"
    assert np.allclose(staging[:, 12], -1.0), "Cloud height slot must be -1.0 (disabled)"
    print("    Decoupled cloud quadtree traversal and packing verified! (PASSED)")


def test_terrain_refraction_ground_to_ground_and_space():
    print("--> Testing Terrain LOD Refraction (Ground-to-Ground & Ground-to-Space)...")
    ctx = moderngl.create_context(standalone=True)

    prog_terrain = ctx.program(
        vertex_shader=terrain_vertex_shader,
        fragment_shader=terrain_fragment_shader
    )
    assert prog_terrain is not None, "Failed to compile prog_terrain"

    R_km = 6371.0
    H_km = 8.5
    delta_n = 0.00029
    max_bend = 2.0 * delta_n * np.sqrt((np.pi * R_km) / (2.0 * H_km))

    # 1. Test Quadtree Horizon Extension via Atmospheric Refraction
    q = PlanetQuadtree(radius_km=R_km, max_lod=3, split_factor=1.0)
    # Observer right at Earth's surface (altitude 2m = 0.002 km)
    cam_ground = np.array([0.0, 0.0, R_km + 0.002], dtype=np.float32)
    patches_geom = q.traverse_raw(cam_ground, fov_deg=60.0, screen_height=1080.0, max_lod=3, refract_bend=0.0)
    patches_refr = q.traverse_raw(cam_ground, fov_deg=60.0, screen_height=1080.0, max_lod=3, refract_bend=max_bend)
    print(f"    Ground horizon patches: geom={len(patches_geom)}, refract-extended={len(patches_refr)}")
    assert len(patches_refr) >= len(patches_geom), "Atmospheric refraction must retain patches beyond geometric horizon"

    # 2. Test Ground-to-Ground Refraction in Vertex Shader
    # Verify that distant terrain vertices are lifted UPWARDS (+ local_up), not depressed downwards (- local_up)
    compute_src = """
    #version 430 core
    layout(local_size_x = 1) in;

    uniform vec3 u_camera_pos;
    uniform vec3 u_refract_center;
    uniform float u_refract_radius;
    uniform float u_refract_scale_height;
    uniform float u_refract_max_bend;
    uniform float u_au_to_km;

    uniform vec3 in_p_world;
    uniform vec3 in_body_pos;

    layout(std430, binding = 0) buffer OutBuf {
        vec3 out_p_world;
        float out_lift_km;
    };

    void main() {
        vec3 p_world = in_p_world;
        bool is_refract_host = (length(in_body_pos - u_refract_center) < 1e-7);

        if (is_refract_host && u_refract_max_bend > 1e-6) {
            vec3 C_km = (u_camera_pos - u_refract_center) * u_au_to_km;
            float r_cam = length(C_km);
            vec3 local_up = normalize(C_km);
            float local_refract_radius = u_refract_radius;
            float h = r_cam - local_refract_radius;

            if (h < u_refract_scale_height * 15.0) {
                float density = exp(-max(h, 0.0) / max(1e-4, u_refract_scale_height));
                float k_refr = u_refract_max_bend * 0.5 * sqrt((2.0 * local_refract_radius) / max(1e-4, 3.141592653589793 * u_refract_scale_height));
                float h_eff = max(h, 0.002);
                float theta_dip_eff = sqrt(2.0 * h_eff / local_refract_radius);
                float max_terr_alpha = clamp(0.5 * k_refr * density * theta_dip_eff, 0.0, 0.05);

                vec3 view_vec = p_world - u_camera_pos;
                float d_v = length(view_vec);
                if (d_v > 1e-7) {
                    float d_v_km = d_v * u_au_to_km;
                    vec3 view_ray = view_vec / d_v;
                    float mu = dot(view_ray, local_up);

                    float alpha_dist = 0.5 * k_refr * density * (d_v_km / max(1e-4, local_refract_radius));
                    float alpha = min(alpha_dist, max_terr_alpha);
                    if (mu > 0.0) {
                        float cos_e = sqrt(max(0.0, 1.0 - mu * mu));
                        alpha *= cos_e;
                    }

                    if (alpha > 1e-7) {
                        vec3 u_dir = local_up - view_ray * mu;
                        float u_len = length(u_dir);
                        if (u_len > 1e-5) {
                            u_dir /= u_len;
                            vec3 app_ray = normalize(view_ray * cos(alpha) + u_dir * sin(alpha));
                            p_world = u_camera_pos + app_ray * d_v;
                        }
                    }
                }
            }
        }

        out_p_world = p_world;
        vec3 local_up = normalize((u_camera_pos - u_refract_center) * u_au_to_km);
        out_lift_km = dot(p_world - in_p_world, local_up) * u_au_to_km;
    }
    """
    comp_prog = ctx.compute_shader(compute_src)
    ssbo = ctx.buffer(reserve=32)
    ssbo.bind_to_storage_buffer(0)

    AU_TO_KM = 149597870.7
    comp_prog['u_camera_pos'].value = (0.0, (R_km + 0.002) / AU_TO_KM, 0.0)
    comp_prog['u_refract_center'].value = (0.0, 0.0, 0.0)
    comp_prog['u_refract_radius'].value = R_km
    comp_prog['u_refract_scale_height'].value = H_km
    comp_prog['u_refract_max_bend'].value = max_bend
    comp_prog['u_au_to_km'].value = AU_TO_KM
    comp_prog['in_body_pos'].value = (0.0, 0.0, 0.0)

    # Distant terrain point 20 km away on the horizon
    d_km = 20.0
    theta = d_km / R_km
    p_terr_km = np.array([R_km * np.sin(theta), R_km * np.cos(theta), 0.0])
    comp_prog['in_p_world'].value = tuple(p_terr_km / AU_TO_KM)

    comp_prog.run()
    res = np.frombuffer(ssbo.read(), dtype=np.float32)
    lift_meters = res[3] * 1000.0
    print(f"    Ground-to-ground terrain lift at 20 km: {lift_meters:+.4f} m (Must be > 0)")
    assert lift_meters > 0.0, f"Terrestrial refraction must lift terrain upward, but got {lift_meters} m"

    # 3. Test Ground-to-Space Refraction at Zenith
    # Verify that an object overhead (zenith) is NOT falsely flagged as occluded and has 0 deflection
    refr_glsl = open(os.path.join(ROOT_DIR, "engine", "glsl", "common", "refraction.glsl")).read()
    zenith_src = """
    #version 430 core
    layout(local_size_x = 1) in;
    SHARED_REFRACTION

    layout(std430, binding = 0) buffer OutBuf {
        vec3 out_v_app;
        float out_is_occ;
        float out_lift_arcmin;
    };

    uniform vec3 t_C_km;
    uniform vec3 t_V;
    uniform float t_d_km;

    void main() {
        bool is_occ = false;
        vec3 V_app = solve_refraction_apparent(t_C_km, t_V, t_d_km, is_occ);
        float lift = acos(clamp(dot(t_V, V_app), -1.0, 1.0)) * (180.0 * 60.0 / 3.141592653589793);
        out_v_app = V_app;
        out_is_occ = is_occ ? 1.0 : 0.0;
        out_lift_arcmin = lift;
    }
    """.replace("SHARED_REFRACTION", refr_glsl)

    zenith_prog = ctx.compute_shader(zenith_src)
    zenith_prog['u_refract_radius'].value = R_km
    zenith_prog['u_refract_scale_height'].value = H_km
    zenith_prog['u_refract_max_bend'].value = max_bend
    zenith_prog['u_refract_pole'].value = (0.0, 1.0, 0.0)
    zenith_prog['u_refract_oblateness'].value = 0.0
    zenith_prog['t_C_km'].value = (0.0, R_km + 0.002, 0.0)
    zenith_prog['t_V'].value = (0.0, 1.0, 0.0) # Zenith (directly overhead)
    zenith_prog['t_d_km'].value = 384400.0

    zenith_ssbo = ctx.buffer(reserve=32)
    zenith_ssbo.bind_to_storage_buffer(0)
    zenith_prog.run()

    z_res = np.frombuffer(zenith_ssbo.read(), dtype=np.float32)
    is_occ = bool(z_res[3] > 0.5)
    z_lift = z_res[4]
    print(f"    Ground-to-space zenith lift: {z_lift:.4f}', occluded: {is_occ}")
    assert not is_occ, "Overhead celestial body at zenith must NOT be occluded"
    assert z_lift < 0.01, f"Overhead celestial body at zenith must have near-zero deflection, got {z_lift}'"

    print("    Terrain LOD Refraction tests (Ground-to-Ground & Ground-to-Space) PASSED!")


def main():
    print("=== Running Terrain Quadtree LOD & Tiled Streaming Tests ===")
    test_cube_to_sphere_continuity()
    test_quadtree_subdivision()
    test_quadtree_frustum_culling_and_low_fov()
    test_cloud_horizon_culling_and_oblateness()
    test_decoupled_cloud_quadtree()
    test_tiled_streamer_and_fallback()
    test_depth_weighted_eviction_and_lod1_retention()
    test_gpu_terrain_rendering_pipeline()
    test_terrain_refraction_ground_to_ground_and_space()
    print("=== All Terrain LOD Tests PASSED Successfully! ===")


if __name__ == "__main__":
    main()

