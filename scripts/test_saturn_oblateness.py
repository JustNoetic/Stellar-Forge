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

def run_oblateness_test():
    print("=" * 70)
    print("Testing Saturn Atmosphere Oblateness: Mode 2 vs Mode 3")
    print("=" * 70)

    ctx = moderngl.create_context(standalone=True)

    # 1. Compile Shaders
    prog_sky_view = ctx.program(
        vertex_shader=sky_view_lut_vertex_shader,
        fragment_shader=sky_view_lut_fragment_shader
    )
    prog_atmo = ctx.program(
        vertex_shader=atmo_vertex_shader,
        fragment_shader=atmo_fragment_shader
    )

    # 2. Sky-View LUT texture and FBO
    sky_view_tex = ctx.texture((256, 256), 4, dtype='f2')
    sky_view_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    sky_view_tex.repeat_x = True
    sky_view_tex.repeat_y = False
    sky_view_fbo = ctx.framebuffer(color_attachments=[sky_view_tex])

    quad_verts = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
    quad_vbo = ctx.buffer(quad_verts.tobytes())
    quad_vao = ctx.vertex_array(prog_sky_view, [(quad_vbo, '2f', 'in_position')])

    # 3. Scene FBO & Atmosphere icosphere mesh
    w, h = 800, 600
    color_tex = ctx.texture((w, h), 4, dtype='f2')
    depth_tex = ctx.depth_texture((w, h))
    scene_fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

    mesh_verts, mesh_idx = create_icosphere_mesh(subdivisions=4)
    vbo = ctx.buffer(mesh_verts.astype(np.float32).tobytes())
    ibo = ctx.buffer(mesh_idx.astype(np.uint32).tobytes())
    vao_atmo = ctx.vertex_array(prog_atmo, [(vbo, '3f 12x', 'in_position')], index_buffer=ibo)

    # 4. Buffers: SceneData UBO (1), AllInstances SSBO (2), AtmoData SSBO (8)
    UBO_SIZE = 7328
    scene_ubo = ctx.buffer(reserve=UBO_SIZE)
    scene_ubo.bind_to_uniform_block(1)

    instances_ssbo = ctx.buffer(reserve=1024)
    instances_ssbo.bind_to_storage_buffer(binding=2)
    instances_staging = np.zeros(64, dtype=np.float32)
    instances_ssbo.write(instances_staging.tobytes())

    atmo_ssbo = ctx.buffer(reserve=1024)
    atmo_ssbo.bind_to_storage_buffer(binding=8)

    # Saturn parameters
    AU_TO_KM = 149597870.7
    f_obl = 0.09796
    f_scale = 1.0 / (1.0 - f_obl) # ~1.1086
    planet_r_km = 60268.0
    atmo_r_km = planet_r_km * 1.08
    atmo_r_au = atmo_r_km / AU_TO_KM
    pole_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Transmittance & Multi-scatter LUTs
    trans_init = np.full((256, 256, 4), [0.85, 0.82, 0.75, 1.0], dtype=np.float32)
    trans_tex = ctx.texture((256, 256), 4, trans_init.tobytes(), dtype='f4')
    trans_tex.use(location=1)

    multi_init = np.full((64, 64, 4), [0.03, 0.04, 0.06, 1.0], dtype=np.float32)
    multi_tex = ctx.texture((64, 64), 4, multi_init.tobytes(), dtype='f4')
    multi_tex.use(location=3)

    dummy_map = ctx.texture((1, 1), 4, dtype='f4')
    dummy_cdf = ctx.texture3d((1, 1, 1), 1, dtype='f4')
    dummy_gradient = np.full((1024, 16, 4), 200, dtype=np.uint8)
    ring_gradient_tex = ctx.texture((1024, 16), 4, dummy_gradient.tobytes())

    ring_gradient_tex.use(location=0)
    dummy_map.use(location=6)
    dummy_cdf.use(location=7)
    dummy_map.use(location=8)
    depth_tex.use(location=9)
    sky_view_tex.use(location=12)

    if 'u_transmittance_lut' in prog_sky_view: prog_sky_view['u_transmittance_lut'].value = 1
    if 'u_multi_scatter_lut' in prog_sky_view: prog_sky_view['u_multi_scatter_lut'].value = 3

    if 'u_ring_gradients' in prog_atmo: prog_atmo['u_ring_gradients'].value = 0
    if 'u_transmittance_lut' in prog_atmo: prog_atmo['u_transmittance_lut'].value = 1
    if 'u_multi_scatter_lut' in prog_atmo: prog_atmo['u_multi_scatter_lut'].value = 3
    if 'u_ringshine_lut' in prog_atmo: prog_atmo['u_ringshine_lut'].value = 6
    if 'u_ringshine_cdf_lut' in prog_atmo: prog_atmo['u_ringshine_cdf_lut'].value = 7
    if 'u_ringshine_map' in prog_atmo: prog_atmo['u_ringshine_map'].value = 8
    if 'u_depth_texture' in prog_atmo: prog_atmo['u_depth_texture'].value = 9
    if 'u_sky_view_lut' in prog_atmo: prog_atmo['u_sky_view_lut'].value = 12

    # Fill SSBO
    staging = np.zeros(256, dtype=np.float32)
    staging_int = staging.view(np.int32)
    staging[0:3] = [0.0, 0.0, 0.0]
    staging[3] = atmo_r_au
    staging[4:7] = [5.8e-6, 13.5e-6, 33.1e-6] # Rayleigh
    staging[7] = 280.0 # scale height
    staging[8:11] = [3.5e-6, 3.5e-6, 3.5e-6] # Mie
    staging[11] = 15.0 # Mie scale height
    staging[12:15] = [0.0, 0.0, 0.0]
    staging[15] = 0.758 # g
    staging[16:19] = [0.0, 0.0, 0.0] # Ozone
    staging[19] = 1.0 # intensity
    staging[20] = planet_r_km
    staging[21] = atmo_r_km
    staging[22] = AU_TO_KM
    staging_int[23] = 32 # num_samples
    staging[24:27] = pole_dir
    staging[27] = f_scale # Oblateness factor!
    staging_int[28] = 0
    staging_int[29] = 0 # no adaptive
    staging_int[30] = 0
    staging[31] = 0.0
    staging[32:35] = [1.0, 1.0, 1.0]

    staging[182] = planet_r_km # clip radius
    staging[184:188] = [1.0 / 280.0, 1.0 / 15.0, 1.0 / 8.0, 0.0]
    g = 0.758
    g2 = g * g
    c1 = (3.0 / (8.0 * math.pi)) * ((1.0 - g2) / (2.0 + g2))
    c2 = 1.0 + g2
    c3 = 2.0 * g
    staging[188:192] = [c1, c2, c3, 0.0]

    sun_dir = np.array([0.7071, 0.0, 0.7071], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    sun_color = np.array([1.2, 1.1, 1.0], dtype=np.float32)

    staging[192:195] = sun_color
    staging[195] = 0.00465
    staging[208:211] = sun_dir
    staging[211] = 0.999
    staging[224:227] = sun_dir * 1.496e8
    staging[227] = 0.00465
    staging[240:244] = [0.0, 0.0, 1.0, 0.00465]

    atmo_ssbo.write(staging.tobytes())

    fov = 45.0
    aspect = w / h
    near = 1e-4
    far = 100.0

    prog_atmo['u_screen_res'].value = (float(w), float(h))
    prog_atmo['u_atmo_clip_mode'].value = 0
    prog_atmo['u_num_ring_planes'].value = 0

    def measure_render(quality_mode):
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
        rgb = data[:, :, 0:3]
        mask = (rgb > 1e-4).any(axis=2)

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return 0, 0, 0, 0, 0, 0
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        height_px = max_y - min_y + 1
        width_px = max_x - min_x + 1
        return min_y, max_y, min_x, max_x, height_px, width_px

    def test_viewpoint(name, cam_p, cam_u):
        print(f"\n--- Measuring {name} View ---")
        cam_p_km = cam_p * AU_TO_KM
        h_p = float(np.dot(cam_p_km, pole_dir))
        c_sph_km = cam_p_km + (h_p * (f_scale - 1.0)) * pole_dir

        v_mat = matrix44.create_look_at(cam_p, np.array([0.0, 0.0, 0.0], dtype=np.float32), cam_u, dtype='f4')
        p_mat = matrix44.create_perspective_projection_matrix(fov, aspect, near, far, dtype='f4')

        ubo_s = np.zeros(UBO_SIZE // 4, dtype=np.float32)
        ubo_s[0:16] = p_mat.ravel()
        ubo_s[16:32] = v_mat.ravel()
        ubo_s[32:33].view(np.int32)[0] = 1
        ubo_s[36:40] = [sun_dir[0], sun_dir[1], sun_dir[2], 0.00465]
        ubo_s[100:104] = [sun_color[0], sun_color[1], sun_color[2], 1.0]
        ubo_s[292] = far
        ubo_s[293] = 1.0
        scene_ubo.write(ubo_s.tobytes())

        i_proj = np.linalg.inv(p_mat).astype('f4')
        i_view = np.linalg.inv(v_mat).astype('f4')
        prog_atmo['u_inv_proj'].write(i_proj.tobytes())
        prog_atmo['u_inv_view'].write(i_view.tobytes())
        prog_atmo['u_camera_pos'].write(cam_p.tobytes())

        # Bake Sky View LUT for this viewpoint
        sky_view_fbo.use()
        ctx.viewport = (0, 0, 256, 256)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        prog_sky_view['u_cam_pos'].value = tuple(c_sph_km)
        prog_sky_view['u_sun_dir'].value = tuple(sun_dir)
        quad_vao.render(moderngl.TRIANGLE_STRIP)

        m2_min_y, m2_max_y, m2_min_x, m2_max_x, m2_h, m2_w = measure_render(2)
        print(f"Mode 2 (Volumetric): Height={m2_h} px, Width={m2_w} px (Y: [{m2_min_y}, {m2_max_y}], X: [{m2_min_x}, {m2_max_x}])")

        m3_min_y, m3_max_y, m3_min_x, m3_max_x, m3_h, m3_w = measure_render(3)
        print(f"Mode 3 (Analytical): Height={m3_h} px, Width={m3_w} px (Y: [{m3_min_y}, {m3_max_y}], X: [{m3_min_x}, {m3_max_x}])")

        dh = abs(m3_h - m2_h)
        dw = abs(m3_w - m2_w)
        print(f"Difference: Height={dh} px, Width={dw} px")
        assert dh <= 1, f"Height mismatch > 1 px: {dh}"
        assert dw <= 1, f"Width mismatch > 1 px: {dw}"

    test_viewpoint("Equatorial", np.array([0.0, 0.0, 2.5 * atmo_r_au], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    test_viewpoint("45-deg Inclination", np.array([0.0, 2.5 * atmo_r_au * 0.7071, 2.5 * atmo_r_au * 0.7071], dtype=np.float32), np.array([0.0, 0.7071, -0.7071], dtype=np.float32))
    test_viewpoint("Polar (Looking Down at Pole)", np.array([0.0, 2.5 * atmo_r_au, 0.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32))

    print("\n" + "=" * 70)
    print("ALL OBLATENESS VIEWPOINT TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_oblateness_test()
