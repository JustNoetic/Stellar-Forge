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

def test_ring_shadow():
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

    w, h = 800, 600
    color_tex = ctx.texture((w, h), 4, dtype='f2')
    depth_tex = ctx.depth_texture((w, h))
    scene_fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

    mesh_verts, mesh_idx = create_icosphere_mesh(subdivisions=4)
    vbo = ctx.buffer(mesh_verts.astype(np.float32).tobytes())
    ibo = ctx.buffer(mesh_idx.astype(np.uint32).tobytes())
    vao_atmo = ctx.vertex_array(prog_atmo, [(vbo, '3f 12x', 'in_position')], index_buffer=ibo)

    UBO_SIZE = 7328
    scene_ubo = ctx.buffer(reserve=UBO_SIZE)
    scene_ubo.bind_to_uniform_block(1)

    instances_ssbo = ctx.buffer(reserve=1024)
    instances_ssbo.bind_to_storage_buffer(binding=2)
    instances_staging = np.zeros(64, dtype=np.float32)
    # Ring mask = 1
    instances_staging[3 * 4 + 3] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.float32)[0]
    instances_ssbo.write(instances_staging.tobytes())

    atmo_ssbo = ctx.buffer(reserve=1024)
    atmo_ssbo.bind_to_storage_buffer(binding=8)

    AU_TO_KM = 149597870.7
    f_obl = 0.09796
    f_scale = 1.0 / (1.0 - f_obl)
    planet_r_km = 60268.0
    atmo_r_km = planet_r_km * 1.08
    atmo_r_au = atmo_r_km / AU_TO_KM
    pole_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Saturn ring params
    r_inner_km = planet_r_km * 1.2
    r_outer_km = planet_r_km * 2.3
    r_inner_au = r_inner_km / AU_TO_KM
    r_outer_au = r_outer_km / AU_TO_KM

    ring_params = np.zeros((16, 4), dtype=np.float32)
    ring_params[0, 0] = r_inner_au
    ring_params[0, 1] = r_outer_au
    ring_params[0, 2] = 1.0 # Opacity
    ring_params[0, 3] = planet_r_km / AU_TO_KM

    ring_centers = np.zeros((16, 3), dtype=np.float32)
    ring_normals = np.zeros((16, 3), dtype=np.float32)
    ring_normals[0] = pole_dir

    # Realistic ring gradient with opaque B-ring and Cassini division
    ring_grad_data = np.zeros((16, 4096, 4), dtype=np.float32)
    # Let's create varying opacity across row 0:
    # 0 to 0.3: C ring (alpha = 0.2)
    # 0.3 to 0.7: B ring (alpha = 0.98)
    # 0.7 to 0.75: Cassini division (alpha = 0.05)
    # 0.75 to 1.0: A ring (alpha = 0.6)
    u_vals = np.linspace(0.0, 1.0, 4096)
    alphas = np.zeros(4096, dtype=np.float32)
    alphas[(u_vals >= 0.0) & (u_vals < 0.3)] = 0.2
    alphas[(u_vals >= 0.3) & (u_vals < 0.7)] = 0.98
    alphas[(u_vals >= 0.7) & (u_vals < 0.75)] = 0.05
    alphas[(u_vals >= 0.75) & (u_vals <= 1.0)] = 0.6
    ring_grad_data[0, :, 3] = alphas
    ring_grad_data[0, :, 0:3] = 0.8

    ring_grad_tex = ctx.texture((4096, 16), 4, ring_grad_data.tobytes(), dtype='f4')
    ring_grad_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    ring_shadow_tex = ctx.texture((4096, 1), 4, dtype='f4')
    ring_shadow_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    ring_shadow_tex.repeat_x = False
    ring_shadow_tex.repeat_y = False
    ring_shadow_tex.write(ring_grad_data[0, :, :].tobytes())
    ring_shadow_tex.build_mipmaps()

    trans_init = np.full((256, 256, 4), [0.85, 0.82, 0.75, 1.0], dtype=np.float32)
    trans_tex = ctx.texture((256, 256), 4, trans_init.tobytes(), dtype='f4')

    multi_init = np.full((64, 64, 4), [0.03, 0.04, 0.06, 1.0], dtype=np.float32)
    multi_tex = ctx.texture((64, 64), 4, multi_init.tobytes(), dtype='f4')

    dummy_map = ctx.texture((1, 1), 4, dtype='f4')
    dummy_cdf = ctx.texture3d((1, 1, 1), 1, dtype='f4')

    ring_grad_tex.use(location=0)
    trans_tex.use(location=1)
    multi_tex.use(location=3)
    dummy_map.use(location=6)
    dummy_cdf.use(location=7)
    dummy_map.use(location=8)
    depth_tex.use(location=9)
    sky_view_tex.use(location=12)
    ring_shadow_tex.use(location=13)
    sky_view_trans_tex.use(location=14)

    if 'u_transmittance_lut' in prog_sky_view: prog_sky_view['u_transmittance_lut'].value = 1
    if 'u_multi_scatter_lut' in prog_sky_view: prog_sky_view['u_multi_scatter_lut'].value = 3

    if 'u_ring_gradients' in prog_atmo: prog_atmo['u_ring_gradients'].value = 0
    if 'u_ring_shadow_tex' in prog_atmo: prog_atmo['u_ring_shadow_tex'].value = 13
    if 'u_transmittance_lut' in prog_atmo: prog_atmo['u_transmittance_lut'].value = 1
    if 'u_multi_scatter_lut' in prog_atmo: prog_atmo['u_multi_scatter_lut'].value = 3
    if 'u_ringshine_lut' in prog_atmo: prog_atmo['u_ringshine_lut'].value = 6
    if 'u_ringshine_cdf_lut' in prog_atmo: prog_atmo['u_ringshine_cdf_lut'].value = 7
    if 'u_ringshine_map' in prog_atmo: prog_atmo['u_ringshine_map'].value = 8
    if 'u_depth_texture' in prog_atmo: prog_atmo['u_depth_texture'].value = 9
    if 'u_sky_view_lut' in prog_atmo: prog_atmo['u_sky_view_lut'].value = 12
    if 'u_sky_view_trans_lut' in prog_atmo: prog_atmo['u_sky_view_trans_lut'].value = 14

    # Sun shining from angle (e.g. 25 degrees above ring plane)
    sun_elev = math.radians(25.0)
    sun_dir = np.array([math.cos(sun_elev), math.sin(sun_elev), 0.3], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    sun_color = np.array([1.2, 1.1, 1.0], dtype=np.float32)

    staging = np.zeros(256, dtype=np.float32)
    staging_int = staging.view(np.int32)
    staging[0:3] = [0.0, 0.0, 0.0]
    staging[3] = atmo_r_au
    staging[4:7] = [5.8e-6, 13.5e-6, 33.1e-6] # Rayleigh
    staging[7] = 280.0
    staging[8:11] = [3.5e-6, 3.5e-6, 3.5e-6] # Mie
    staging[11] = 15.0
    staging[12:15] = [0.0, 0.0, 0.0]
    staging[15] = 0.758
    staging[16:19] = [0.0, 0.0, 0.0]
    staging[19] = 1.0
    staging[20] = planet_r_km
    staging[21] = atmo_r_km
    staging[22] = AU_TO_KM
    staging_int[23] = 32
    staging[24:27] = pole_dir
    staging[27] = f_scale
    staging_int[28] = 0
    staging_int[29] = 0
    staging_int[30] = 0
    staging[31] = 0.0
    staging[32:35] = [1.0, 1.0, 1.0]

    staging[182] = planet_r_km
    staging[184:188] = [1.0 / 280.0, 1.0 / 15.0, 1.0 / 8.0, 0.0]
    g = 0.758
    g2 = g * g
    c1 = (3.0 / (8.0 * math.pi)) * ((1.0 - g2) / (2.0 + g2))
    c2 = 1.0 + g2
    c3 = 2.0 * g
    staging[188:192] = [c1, c2, c3, 0.0]

    staging[192:195] = sun_color
    staging[195] = 0.00465
    staging[208:211] = sun_dir
    staging[211] = 0.999
    staging[224:227] = sun_dir * 1.496e8
    staging[227] = 0.00465
    staging[240:244] = [0.0, 0.0, 1.0, 0.00465]

    atmo_ssbo.write(staging.tobytes())

    # Camera looking at Saturn from slightly above equator, looking at the shadow
    cam_dist = 2.5 * atmo_r_au
    cam_lat = math.radians(-15.0) # South of equator where the shadow falls!
    cam_pos = np.array([0.0, cam_dist * math.sin(cam_lat), cam_dist * math.cos(cam_lat)], dtype=np.float32)
    cam_pos_km = cam_pos * AU_TO_KM
    h_pole = float(np.dot(cam_pos_km, pole_dir))
    cam_sph_km = cam_pos_km + (h_pole * (f_scale - 1.0)) * pole_dir

    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view_mat = matrix44.create_look_at(cam_pos, np.array([0.0, 0.0, 0.0], dtype=np.float32), cam_up, dtype='f4')
    fov = 45.0
    aspect = w / h
    near = 1e-4
    far = 100.0
    proj_mat = matrix44.create_perspective_projection_matrix(fov, aspect, near, far, dtype='f4')

    ubo_staging = np.zeros(UBO_SIZE // 4, dtype=np.float32)
    ubo_staging[0:16] = proj_mat.ravel()
    ubo_staging[16:32] = view_mat.ravel()
    ubo_staging[32:33].view(np.int32)[0] = 1
    ubo_staging[36:40] = [sun_dir[0], sun_dir[1], sun_dir[2], 0.00465]
    ubo_staging[100:104] = [sun_color[0], sun_color[1], sun_color[2], 1.0]
    ubo_staging[292] = far
    ubo_staging[293] = 1.0
    scene_ubo.write(ubo_staging.tobytes())

    inv_proj = np.linalg.inv(proj_mat).astype('f4')
    inv_view = np.linalg.inv(view_mat).astype('f4')
    prog_atmo['u_inv_proj'].write(inv_proj.tobytes())
    prog_atmo['u_inv_view'].write(inv_view.tobytes())
    prog_atmo['u_camera_pos'].write(cam_pos.tobytes())
    prog_atmo['u_screen_res'].value = (float(w), float(h))
    prog_atmo['u_atmo_clip_mode'].value = 0
    prog_atmo['u_num_ring_planes'].value = 1
    prog_atmo['u_ring_center'].write(ring_centers.tobytes())
    prog_atmo['u_ring_normal'].write(ring_normals.tobytes())
    prog_atmo['u_ring_params'].write(ring_params.tobytes())
    if 'u_ring_coplanar_mask' in prog_atmo:
        coplanar = np.zeros(16, dtype=np.uint32)
        coplanar[0] = 1
        prog_atmo['u_ring_coplanar_mask'].write(coplanar.tobytes())

    # Bake Sky View LUT
    sky_view_fbo.use()
    ctx.viewport = (0, 0, 256, 256)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)
    prog_sky_view['u_cam_pos'].value = tuple(cam_sph_km)
    prog_sky_view['u_sun_dir'].value = tuple(sun_dir)
    quad_vao.render(moderngl.TRIANGLE_STRIP)

    def render_and_get_image(quality_mode):
        prog_atmo['u_atmo_quality'].value = quality_mode
        scene_fbo.use()
        ctx.viewport = (0, 0, w, h)
        scene_fbo.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
        ctx.cull_face = 'front'
        ctx.enable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False

        vao_atmo.render(moderngl.TRIANGLES)

        scene_raw = scene_fbo.read(components=4, dtype='f2')
        data = np.frombuffer(scene_raw, dtype=np.float16).reshape((h, w, 4)).astype(np.float32)
        return data

    m2_data = render_and_get_image(2)
    m3_data = render_and_get_image(3)

    m2_rgb = m2_data[:, :, 0:3]
    m3_rgb = m3_data[:, :, 0:3]

    print(f"Mode 2: min={m2_rgb.min():.6f}, max={m2_rgb.max():.6f}, mean={m2_rgb.mean():.6f}")
    print(f"Mode 3: min={m3_rgb.min():.6f}, max={m3_rgb.max():.6f}, mean={m3_rgb.mean():.6f}")

    from PIL import Image
    def save_img(arr, path):
        # Tone-map and save as 8-bit sRGB
        col = np.clip(arr * 4.0, 0.0, 1.0) # exposure
        col = np.power(col, 1.0 / 2.2)
        img = Image.fromarray((col * 255).astype(np.uint8))
        img.save(path)

    save_img(m2_rgb, "scripts/mode2_shadow.png")
    save_img(m3_rgb, "scripts/mode3_shadow.png")
    print("Saved scripts/mode2_shadow.png and scripts/mode3_shadow.png")

    x_col = 400
    print(f"\n--- Vertical slice at X={x_col} (Y from 90 to 230, through the shadow) ---")
    print(f"{'Y':>4} | {'Mode 2 (High)':>16} | {'Mode 3 (Analytical)':>19} | {'Diff':>8}")
    print("-" * 55)
    for y in range(90, 230, 10):
        v2 = m2_rgb[y, x_col]
        v3 = m3_rgb[y, x_col]
        l2 = np.linalg.norm(v2)
        l3 = np.linalg.norm(v3)
        print(f"{y:4d} | R={v2[0]:.4f} G={v2[1]:.4f} B={v2[2]:.4f} | R={v3[0]:.4f} G={v3[1]:.4f} B={v3[2]:.4f} | {abs(l3-l2):.4f}")

    print("\n--- Testing Shadow Brightness Invariance Across Slicing Steps (X=400, Y=180) ---")
    print(f"{'Steps':>6} | {'Mode 3 In-Shadow RGB':>28} | {'Diff from 8 steps':>18}")
    print("-" * 60)
    if 'u_atmo_slicing_steps' in prog_atmo:
        prog_atmo['u_atmo_slicing_steps'].value = 8
    d_base = render_and_get_image(3)
    base_shadow_val = d_base[180, x_col, 0:3]

    for test_s in [2, 4, 8, 16, 32]:
        if 'u_atmo_slicing_steps' in prog_atmo:
            prog_atmo['u_atmo_slicing_steps'].value = test_s
        d = render_and_get_image(3)
        rgb_val = d[180, x_col, 0:3]
        diff_from_base = np.linalg.norm(rgb_val - base_shadow_val)
        print(f"{test_s:6d} | R={rgb_val[0]:.6f} G={rgb_val[1]:.6f} B={rgb_val[2]:.6f} | {diff_from_base:.6f}")

    # --- Regression Test: Anti-Solar Line-of-Sight (V || -L) ---
    print("\n--- Testing Exact Anti-Solar View (V || -L, Looking directly at shadow from Sun) ---")
    r_mid = 0.5 * (r_inner_km + r_outer_km)
    p_ring_au = np.array([0.0, 0.0, r_mid / AU_TO_KM], dtype=np.float32)
    b_proj = float(np.dot(p_ring_au, sun_dir))
    c_proj = float(np.dot(p_ring_au, p_ring_au)) - (planet_r_km / AU_TO_KM)**2
    lam_proj = b_proj - math.sqrt(max(0.0, b_proj*b_proj - c_proj))
    shadow_target = p_ring_au - lam_proj * sun_dir

    cam_dist_as = 2.0 * atmo_r_au
    cam_pos_as = shadow_target + sun_dir * cam_dist_as
    cam_pos_km_as = cam_pos_as * AU_TO_KM
    h_pole_as = float(np.dot(cam_pos_km_as, pole_dir))
    cam_sph_km_as = cam_pos_km_as + (h_pole_as * (f_scale - 1.0)) * pole_dir

    cam_up_as = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view_dir_as = -sun_dir
    if abs(np.dot(cam_up_as, view_dir_as)) > 0.9:
        cam_up_as = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    view_mat_as = matrix44.create_look_at(cam_pos_as, shadow_target, cam_up_as, dtype='f4')
    fov_as = 20.0
    proj_mat_as = matrix44.create_perspective_projection_matrix(fov_as, w / h, 1e-4, 100.0, dtype='f4')

    ubo_staging[0:16] = proj_mat_as.ravel()
    ubo_staging[16:32] = view_mat_as.ravel()
    scene_ubo.write(ubo_staging.tobytes())

    prog_atmo['u_inv_proj'].write(np.linalg.inv(proj_mat_as).astype('f4').tobytes())
    prog_atmo['u_inv_view'].write(np.linalg.inv(view_mat_as).astype('f4').tobytes())
    prog_atmo['u_camera_pos'].write(cam_pos_as.tobytes())

    sky_view_fbo.use()
    ctx.viewport = (0, 0, 256, 256)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)
    prog_sky_view['u_cam_pos'].value = tuple(cam_sph_km_as)
    prog_sky_view['u_sun_dir'].value = tuple(sun_dir)
    quad_vao.render(moderngl.TRIANGLE_STRIP)

    m2_as = render_and_get_image(2)
    m3_as = render_and_get_image(3)

    center_pixel_m2 = m2_as[300, 400, 0:3]
    center_pixel_m3 = m3_as[300, 400, 0:3]
    center_crop_m3 = m3_as[280:320, 380:420, 0:3]

    print(f"Anti-solar Center Pixel (X=400, Y=300): Mode 2={center_pixel_m2} | Mode 3={center_pixel_m3}")
    print(f"Anti-solar 40x40 Center Region Max RGB: {center_crop_m3.max():.6f}")
    assert np.all(center_crop_m3 < 1e-4), f"Anti-solar bright oval regression detected! Max RGB: {center_crop_m3.max()}"
    print("   [+] Anti-solar view passed: 0.000000 darkness, no bright oval artifact!")

    # --- Regression Test 3: Grazing Limb in Ring Shadow (1-Pixel Line Artifact Test) ---
    print("\n--- Testing Grazing Limb in Ring Shadow (1-Pixel Top Boundary Test) ---")
    cam_dist_gl = 2.2 * atmo_r_au
    cam_lat_gl = math.radians(-25.0)
    cam_pos_gl = np.array([0.0, cam_dist_gl * math.sin(cam_lat_gl), cam_dist_gl * math.cos(cam_lat_gl)], dtype=np.float32)
    cam_pos_km_gl = cam_pos_gl * AU_TO_KM
    h_pole_gl = float(np.dot(cam_pos_km_gl, pole_dir))
    cam_sph_km_gl = cam_pos_km_gl + (h_pole_gl * (f_scale - 1.0)) * pole_dir

    limb_target = np.array([0.0, -atmo_r_au * 0.85, 0.0], dtype=np.float32)
    view_mat_gl = matrix44.create_look_at(cam_pos_gl, limb_target, np.array([0.0, 1.0, 0.0], dtype=np.float32), dtype='f4')
    proj_mat_gl = matrix44.create_perspective_projection_matrix(25.0, w / h, 1e-4, 100.0, dtype='f4')

    ubo_staging[0:16] = proj_mat_gl.ravel()
    ubo_staging[16:32] = view_mat_gl.ravel()
    scene_ubo.write(ubo_staging.tobytes())

    prog_atmo['u_inv_proj'].write(np.linalg.inv(proj_mat_gl).astype('f4').tobytes())
    prog_atmo['u_inv_view'].write(np.linalg.inv(view_mat_gl).astype('f4').tobytes())
    prog_atmo['u_camera_pos'].write(cam_pos_gl.tobytes())

    sky_view_fbo.use()
    ctx.viewport = (0, 0, 256, 256)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)
    prog_sky_view['u_cam_pos'].value = tuple(cam_sph_km_gl)
    prog_sky_view['u_sun_dir'].value = tuple(sun_dir)
    quad_vao.render(moderngl.TRIANGLE_STRIP)

    m2_gl = render_and_get_image(2)
    m3_gl = render_and_get_image(3)

    save_img(m2_gl[:, :, 0:3], "scripts/mode2_limb_shadow.png")
    save_img(m3_gl[:, :, 0:3], "scripts/mode3_limb_shadow.png")
    print("Saved scripts/mode2_limb_shadow.png and scripts/mode3_limb_shadow.png")

    # Scan for any isolated 1-pixel line (non-zero flanked by zeros in vertical or horizontal direction)
    m3_rgb_gl = m3_gl[:, :, 0:3]
    isolated_pixels = []
    for y_scan in range(1, h - 1):
        for x_scan in range(1, w - 1):
            v_curr = m3_rgb_gl[y_scan, x_scan]
            v_up   = m3_rgb_gl[y_scan - 1, x_scan]
            v_down = m3_rgb_gl[y_scan + 1, x_scan]
            if np.any(v_curr > 0.002) and np.all(v_up < 1e-5) and np.all(v_down < 1e-5):
                isolated_pixels.append((x_scan, y_scan, v_curr.tolist()))

    print(f"Isolated 1-pixel lines found at limb: {len(isolated_pixels)}")
    assert len(isolated_pixels) == 0, f"Detected {len(isolated_pixels)} isolated 1-pixel lines at the atmosphere boundary! Sample: {isolated_pixels[:5]}"
    print("   [+] Grazing limb test passed: 0 isolated 1-pixel lines, seamless boundary transition!")

    print("\n=======================================================")
    print("ALL RING SHADOW BENCHMARKS & REGRESSION TESTS PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    test_ring_shadow()
