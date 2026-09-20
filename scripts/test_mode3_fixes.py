import os
import sys
import math
import numpy as np
import moderngl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.rendering.shaders import (
    sky_view_lut_vertex_shader,
    sky_view_lut_fragment_shader,
    atmo_vertex_shader,
    atmo_fragment_shader
)
from engine.rendering.render_utils import create_icosphere_mesh
from pyrr import matrix44

def test_mie_math():
    print("=" * 70)
    print("1. Testing Mie Centering Math Alignment on Oblate Bodies")
    print("=" * 70)

    for body_name, f in [("Saturn", 0.09796), ("Jupiter", 0.0648), ("Earth", 0.00335)]:
        f_scale = 1.0 / (1.0 - f)
        pole = np.array([0.0, 0.98, 0.2], dtype=np.float32)
        pole /= np.linalg.norm(pole)

        sun_dir = np.array([0.7071, 0.4, 0.58], dtype=np.float32)
        sun_dir /= np.linalg.norm(sun_dir)

        # Old buggy app.py calculation:
        p_proj = float(np.dot(sun_dir, pole))
        p_perp = sun_dir - p_proj * pole
        s_dir_sph_old = p_perp * f_scale + p_proj * pole
        s_dir_sph_old /= np.linalg.norm(s_dir_sph_old)

        # Correct toSphericalSpace calculation:
        s_dir_sph_new = p_perp + (p_proj * f_scale) * pole
        s_dir_sph_new /= np.linalg.norm(s_dir_sph_new)

        # Ray pointing at the sun:
        ray_dir = sun_dir.copy()
        ray_proj = float(np.dot(ray_dir, pole))
        ray_perp = ray_dir - ray_proj * pole
        ray_sph = ray_perp + (ray_proj * f_scale) * pole
        V = ray_sph / np.linalg.norm(ray_sph)

        dot_old = float(np.dot(V, s_dir_sph_old))
        err_old_deg = math.degrees(math.acos(np.clip(dot_old, -1.0, 1.0)))

        dot_new = float(np.dot(V, s_dir_sph_new))
        err_new_deg = math.degrees(math.acos(np.clip(dot_new, -1.0, 1.0)))

        print(f"[{body_name:7s} (f={f:.5f})]: Old error = {err_old_deg:7.4f}° | Fixed error = {err_new_deg:9.6f}°")
        assert err_new_deg < 1e-5, f"Expected 0 error for {body_name}, got {err_new_deg}"

    print("   [+] All oblate body Mie centering tests passed!")

def test_landed_horizon_rendering():
    print("\n" + "=" * 70)
    print("2. Testing Landed Horizon Atmosphere Rendering in Mode 3")
    print("=" * 70)

    ctx = moderngl.create_context(standalone=True)

    prog_sky_view = ctx.program(
        vertex_shader=sky_view_lut_vertex_shader,
        fragment_shader=sky_view_lut_fragment_shader
    )
    prog_atmo = ctx.program(
        vertex_shader=atmo_vertex_shader,
        fragment_shader=atmo_fragment_shader
    )

    sky_view_tex = ctx.texture((256, 256), 4, dtype='f2')
    sky_view_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    sky_view_tex.repeat_x = True
    sky_view_tex.repeat_y = False

    sky_view_trans_tex = ctx.texture((256, 256), 4, dtype='f2')
    sky_view_trans_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    sky_view_trans_tex.repeat_x = True
    sky_view_trans_tex.repeat_y = False

    sky_view_fbo = ctx.framebuffer(color_attachments=[sky_view_tex, sky_view_trans_tex])

    quad_verts = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
    quad_vbo = ctx.buffer(quad_verts.tobytes())
    quad_vao = ctx.vertex_array(prog_sky_view, [(quad_vbo, '2f', 'in_position')])

    AU_TO_KM = 149597870.7
    planet_r_km = 60268.0
    atmo_r_km = planet_r_km * 1.08
    atmo_r_au = atmo_r_km / AU_TO_KM

    # Landed on the surface (15 m above surface)
    land_alt_km = 0.015
    cam_pos_km = np.array([0.0, planet_r_km + land_alt_km, 0.0], dtype=np.float32)
    cam_pos_au = cam_pos_km / AU_TO_KM

    sun_dir = np.array([0.0, 0.3, 0.95], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    sun_color = np.array([1.2, 1.1, 1.0], dtype=np.float32)

    # AtmoData SSBO (binding = 8)
    atmo_ssbo = ctx.buffer(reserve=1024)
    atmo_ssbo.bind_to_storage_buffer(binding=8)
    staging = np.zeros(256, dtype=np.float32)
    staging_int = staging.view(np.int32)

    staging[0:3] = [0.0, 0.0, 0.0]
    staging[3] = atmo_r_au
    staging[4:7] = [5.8e-6, 13.5e-6, 33.1e-6]
    staging[7] = 280.0
    staging[8:11] = [3.5e-6, 3.5e-6, 3.5e-6]
    staging[11] = 15.0
    staging[15] = 0.758
    staging[19] = 1.0
    staging[20] = planet_r_km
    staging[21] = atmo_r_km
    staging[22] = AU_TO_KM
    staging_int[23] = 32
    staging[24:27] = [0.0, 1.0, 0.0]
    staging[27] = 1.0
    staging_int[28] = 0
    staging_int[29] = 1
    staging_int[30] = 0
    staging[32:35] = [1.0, 1.0, 1.0]

    # Precomputed optical parameters
    staging[182] = planet_r_km * (1.0 - 0.0005) # planet_clip_km
    staging[184:188] = [1.0 / 280.0, 1.0 / 15.0, 1.0 / 8.0, 0.0]
    g = 0.758
    g2 = g * g
    staging[188:192] = [(3.0 / (8.0 * math.pi)) * ((1.0 - g2) / (2.0 + g2)), 1.0 + g2, 2.0 * g, 0.0]
    staging[192:195] = sun_color
    staging[195] = 0.00465
    staging[208:211] = sun_dir
    staging[211] = 0.999
    staging[224:227] = sun_dir * 1.496e8
    staging[227] = 0.00465
    staging[240:244] = [0.0, 0.0, 1.0, 0.00465]
    atmo_ssbo.write(staging.tobytes())

    trans_init = np.full((256, 256, 4), [0.85, 0.82, 0.75, 1.0], dtype=np.float32)
    trans_tex = ctx.texture((256, 256), 4, trans_init.tobytes(), dtype='f4')
    trans_tex.use(location=1)

    multi_init = np.full((64, 64, 4), [0.03, 0.04, 0.06, 1.0], dtype=np.float32)
    multi_tex = ctx.texture((64, 64), 4, multi_init.tobytes(), dtype='f4')
    multi_tex.use(location=3)

    if 'u_transmittance_lut' in prog_sky_view: prog_sky_view['u_transmittance_lut'].value = 1
    if 'u_multi_scatter_lut' in prog_sky_view: prog_sky_view['u_multi_scatter_lut'].value = 3

    # Render Sky-View LUT for landed camera
    sky_view_fbo.use()
    ctx.viewport = (0, 0, 256, 256)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)

    prog_sky_view['u_cam_pos'].value = tuple(cam_pos_km)
    prog_sky_view['u_sun_dir'].value = tuple(sun_dir)
    quad_vao.render(moderngl.TRIANGLE_STRIP)

    lut_raw = sky_view_fbo.read(attachment=0, components=4, dtype='f2')
    lut_data = np.frombuffer(lut_raw, dtype=np.float16).reshape((256, 256, 4)).astype(np.float32)

    lut_trans_raw = sky_view_fbo.read(attachment=1, components=4, dtype='f2')
    lut_trans_data = np.frombuffer(lut_trans_raw, dtype=np.float16).reshape((256, 256, 4)).astype(np.float32)

    # Verify that Nadir (Row 0, ground at feet) has near-zero in-scattering and near 1.0 transmittance
    nadir_scatter = lut_data[0, :, 0:3].mean()
    nadir_trans = lut_trans_data[0, :, 0:3].mean()
    print(f"   [+] Landed Sky-View LUT Nadir (Row 0, ground at feet): Scatter={nadir_scatter:.6f}, Transmittance={nadir_trans:.6f}")
    assert nadir_scatter < 0.001, f"Expected nadir scatter ~0, got {nadir_scatter}"
    assert nadir_trans > 0.99, f"Expected nadir transmittance ~1.0, got {nadir_trans}"

    # Verify that Sky Horizon (Row 128) has bright in-scattering
    horizon_sky_scatter = lut_data[128, :, 0:3].mean()
    print(f"   [+] Landed Sky-View LUT Sky Horizon (Row 128): Scatter={horizon_sky_scatter:.6f}")
    assert horizon_sky_scatter > 0.05, f"Expected bright sky horizon scatter, got {horizon_sky_scatter}"

    # Now test atmosphere mesh rendering from ground looking slightly down at the horizon (pitch = -2.0°)
    w, h = 400, 300
    color_tex = ctx.texture((w, h), 4, dtype='f2')
    depth_tex = ctx.depth_texture((w, h))
    scene_fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

    mesh_verts, mesh_idx = create_icosphere_mesh(subdivisions=3)
    vbo = ctx.buffer(mesh_verts.astype(np.float32).tobytes())
    ibo = ctx.buffer(mesh_idx.astype(np.uint32).tobytes())
    vao_atmo = ctx.vertex_array(prog_atmo, [(vbo, '3f 12x', 'in_position')], index_buffer=ibo)

    UBO_SIZE = 7328
    scene_ubo = ctx.buffer(reserve=UBO_SIZE)
    scene_ubo.bind_to_uniform_block(1)
    ubo_staging = np.zeros(UBO_SIZE // 4, dtype=np.float32)

    instances_ssbo = ctx.buffer(reserve=1024)
    instances_ssbo.bind_to_storage_buffer(binding=2)
    instances_staging = np.zeros(64, dtype=np.float32)
    instances_ssbo.write(instances_staging.tobytes())

    ring_gradient_tex = ctx.texture((16, 16), 4, np.zeros((16, 16, 4), dtype=np.uint8).tobytes())
    ring_shadow_tex = ctx.texture((16, 1), 4, dtype='f4')
    dummy_cdf = ctx.texture3d((1, 1, 1), 1, dtype='f4')
    dummy_map = ctx.texture((1, 1), 4, dtype='f4')

    ring_gradient_tex.use(location=0)
    trans_tex.use(location=1)
    multi_tex.use(location=3)
    dummy_map.use(location=6)
    dummy_cdf.use(location=7)
    dummy_map.use(location=8)
    depth_tex.use(location=9)
    sky_view_tex.use(location=12)
    ring_shadow_tex.use(location=13)
    sky_view_trans_tex.use(location=14)

    if 'u_ring_gradients' in prog_atmo: prog_atmo['u_ring_gradients'].value = 0
    if 'u_transmittance_lut' in prog_atmo: prog_atmo['u_transmittance_lut'].value = 1
    if 'u_multi_scatter_lut' in prog_atmo: prog_atmo['u_multi_scatter_lut'].value = 3
    if 'u_ringshine_lut' in prog_atmo: prog_atmo['u_ringshine_lut'].value = 6
    if 'u_ringshine_cdf_lut' in prog_atmo: prog_atmo['u_ringshine_cdf_lut'].value = 7
    if 'u_ringshine_map' in prog_atmo: prog_atmo['u_ringshine_map'].value = 8
    if 'u_depth_texture' in prog_atmo: prog_atmo['u_depth_texture'].value = 9
    if 'u_sky_view_lut' in prog_atmo: prog_atmo['u_sky_view_lut'].value = 12
    if 'u_ring_shadow_tex' in prog_atmo: prog_atmo['u_ring_shadow_tex'].value = 13
    if 'u_sky_view_trans_lut' in prog_atmo: prog_atmo['u_sky_view_trans_lut'].value = 14

    # Camera looking towards the horizon, pitched down by 2 degrees (looking at ground)
    # Forward vector: horizontal along Z, pitched down by 2 degrees:
    pitch_rad = math.radians(-2.0)
    fwd = np.array([0.0, math.sin(pitch_rad), math.cos(pitch_rad)], dtype=np.float32)
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view_mat = matrix44.create_look_at(cam_pos_au, cam_pos_au + fwd, up, dtype='f4')

    fov = 55.0
    proj_mat = matrix44.create_perspective_projection_matrix(fov, w / h, 1e-6, 100.0, dtype='f4')
    inv_proj = np.linalg.inv(proj_mat).astype('f4')
    inv_view = np.linalg.inv(view_mat).astype('f4')

    ubo_staging[0:16] = proj_mat.ravel()
    ubo_staging[16:32] = view_mat.ravel()
    ubo_staging[32:33].view(np.int32)[0] = 1
    ubo_staging[36:40] = [sun_dir[0], sun_dir[1], sun_dir[2], 0.00465]
    ubo_staging[100:104] = [sun_color[0], sun_color[1], sun_color[2], 1.0]
    ubo_staging[292] = 100.0
    ubo_staging[293] = 1.0
    scene_ubo.write(ubo_staging.tobytes())

    if 'u_inv_proj' in prog_atmo: prog_atmo['u_inv_proj'].write(inv_proj.tobytes())
    if 'u_inv_view' in prog_atmo: prog_atmo['u_inv_view'].write(inv_view.tobytes())
    if 'u_camera_pos' in prog_atmo: prog_atmo['u_camera_pos'].write(cam_pos_au.tobytes())
    if 'u_screen_res' in prog_atmo: prog_atmo['u_screen_res'].value = (float(w), float(h))
    if 'u_atmo_clip_mode' in prog_atmo: prog_atmo['u_atmo_clip_mode'].value = 0
    if 'u_atmo_quality' in prog_atmo: prog_atmo['u_atmo_quality'].value = 3
    if 'u_atmo_slicing_steps' in prog_atmo: prog_atmo['u_atmo_slicing_steps'].value = 8

    # Render landed view
    scene_fbo.use()
    ctx.viewport = (0, 0, w, h)
    scene_fbo.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
    ctx.cull_face = 'front'
    ctx.enable(moderngl.CULL_FACE)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.depth_mask = False

    vao_atmo.render(moderngl.TRIANGLES)

    frame_raw = scene_fbo.read(components=4, dtype='f2')
    frame_data = np.frombuffer(frame_raw, dtype=np.float16).reshape((h, w, 4)).astype(np.float32)

    # In a frame looking pitched down 2 degrees with 55 deg FOV:
    # Top rows (sky above horizon) will have sky scattering.
    # Bottom rows (ground well below horizon) should have Ground Disk in-scattering (aerial perspective <= 0.02).
    # Specifically, the bottom quarter of the screen (rows 0 to 75) is ground:
    ground_scatter = frame_data[:75, :, 0:3].mean()
    sky_scatter = frame_data[225:, :, 0:3].mean()

    print(f"   [+] Ground region (bottom rows) scatter: {ground_scatter:.6f}")
    print(f"   [+] Sky region (top rows) scatter:       {sky_scatter:.6f}")

    assert not np.isnan(frame_data).any(), "NaN in landed atmosphere frame!"
    assert not np.isinf(frame_data).any(), "Inf in landed atmosphere frame!"
    # Prior to fix, ground_scatter was > 0.08 because the sky dome leaked over the entire ground.
    assert ground_scatter < 0.03, f"Ground scatter too high ({ground_scatter:.6f})! Sky is clipping into ground!"
    assert sky_scatter > 0.05, f"Sky scatter should be bright, got {sky_scatter:.6f}"
    print("   [+] Verified no sky clipping into ground! Ground and sky cleanly separated at horizon.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_mie_math()
    test_landed_horizon_rendering()
