import os
import sys
import math
import numpy as np
import moderngl
import OpenGL.GL as gl

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

def test_mode3_pipeline():
    print("=" * 70)
    print("Testing Atmosphere Mode 3: Sky-View LUT + Analytical Depth Slicing")
    print("=" * 70)

    ctx = moderngl.create_context(standalone=True)
    print(f"ModernGL context created. GL Renderer: {ctx.info['GL_RENDERER']}")

    # 1. Compile Sky-View LUT shader
    print("\n1. Compiling Sky-View LUT shader program...")
    prog_sky_view = ctx.program(
        vertex_shader=sky_view_lut_vertex_shader,
        fragment_shader=sky_view_lut_fragment_shader
    )
    assert prog_sky_view is not None, "Failed to compile prog_sky_view"
    print("   [+] prog_sky_view compiled successfully.")

    # 2. Allocate 256x256 RGBA16F Sky-View LUT textures and FBO
    print("\n2. Allocating 256x256 RGBA16F Sky-View LUT textures & dual-target FBO...")
    sky_view_tex = ctx.texture((256, 256), 4, dtype='f2')
    sky_view_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    sky_view_tex.repeat_x = True
    sky_view_tex.repeat_y = False

    sky_view_trans_tex = ctx.texture((256, 256), 4, dtype='f2')
    sky_view_trans_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    sky_view_trans_tex.repeat_x = True
    sky_view_trans_tex.repeat_y = False

    sky_view_fbo = ctx.framebuffer(color_attachments=[sky_view_tex, sky_view_trans_tex])

    # Screen quad VAO
    quad_verts = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')
    quad_vbo = ctx.buffer(quad_verts.tobytes())
    quad_vao = ctx.vertex_array(prog_sky_view, [(quad_vbo, '2f', 'in_position')])
    print("   [+] Sky-View FBO and screen quad VAO ready.")

    # Shared buffers and parameters
    AU_TO_KM = 149597870.7
    planet_r_km = 60268.0
    atmo_r_km = planet_r_km * 1.08
    atmo_r_au = atmo_r_km / AU_TO_KM

    body_pos_rel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam_pos = np.array([0.0, 1.0 * atmo_r_au, 2.5 * atmo_r_au], dtype=np.float32)
    cam_pos_km = cam_pos * AU_TO_KM
    sun_dir = np.array([0.7071, 0.4, 0.58], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    sun_color = np.array([1.2, 1.1, 1.0], dtype=np.float32)

    # 3. AtmoData SSBO (binding = 8)
    atmo_ssbo = ctx.buffer(reserve=1024)
    atmo_ssbo.bind_to_storage_buffer(binding=8)
    staging = np.zeros(256, dtype=np.float32)
    staging_int = staging.view(np.int32)

    staging[0:3] = body_pos_rel
    staging[3] = atmo_r_au
    staging[4:7] = [5.8e-6, 13.5e-6, 33.1e-6]
    staging[7] = 280.0
    staging[8:11] = [3.5e-6, 3.5e-6, 3.5e-6]
    staging[11] = 15.0
    staging[12:15] = [0.0, 0.0, 0.0]
    staging[15] = 0.758
    staging[16:19] = [0.0, 0.0, 0.0]
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
    staging[31] = 0.0
    staging[32:35] = [1.0, 1.0, 1.0]

    staging[184:188] = [1.0 / 280.0, 1.0 / 15.0, 1.0 / 8.0, 0.0]
    # Precomputed Mie constants
    g = 0.758
    g2 = g * g
    c1 = (3.0 / (8.0 * math.pi)) * ((1.0 - g2) / (2.0 + g2))
    c2 = 1.0 + g2
    c3 = 2.0 * g
    staging[188:192] = [c1, c2, c3, 0.0]

    comb_int = sun_color * 1.0
    staging[192:195] = comb_int
    staging[195] = 0.00465
    staging[208:211] = sun_dir
    staging[211] = 0.999
    staging[224:227] = sun_dir * 1.496e8
    staging[227] = 0.00465
    staging[240:244] = [0.0, 0.0, 1.0, 0.00465]

    atmo_ssbo.write(staging.tobytes())

    # Precomputed transmittance and multi-scatter LUT textures
    trans_init = np.full((256, 256, 4), [0.85, 0.82, 0.75, 1.0], dtype=np.float32)
    trans_tex = ctx.texture((256, 256), 4, trans_init.tobytes(), dtype='f4')
    trans_tex.use(location=1)

    multi_init = np.full((64, 64, 4), [0.03, 0.04, 0.06, 1.0], dtype=np.float32)
    multi_tex = ctx.texture((64, 64), 4, multi_init.tobytes(), dtype='f4')
    multi_tex.use(location=3)

    if 'u_transmittance_lut' in prog_sky_view: prog_sky_view['u_transmittance_lut'].value = 1
    if 'u_multi_scatter_lut' in prog_sky_view: prog_sky_view['u_multi_scatter_lut'].value = 3

    # 3. Render Sky-View LUT for Saturn-like parameters
    print("\n3. Rendering Sky-View LUT for Saturn...")
    sky_view_fbo.use()
    ctx.viewport = (0, 0, 256, 256)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)

    prog_sky_view['u_cam_pos'].value = tuple(cam_pos_km)
    prog_sky_view['u_sun_dir'].value = tuple(sun_dir)

    quad_vao.render(moderngl.TRIANGLE_STRIP)

    # Read back LUT pixels from both attachments
    lut_raw = sky_view_fbo.read(attachment=0, components=4, dtype='f2')
    lut_data = np.frombuffer(lut_raw, dtype=np.float16).reshape((256, 256, 4)).astype(np.float32)

    lut_trans_raw = sky_view_fbo.read(attachment=1, components=4, dtype='f2')
    lut_trans_data = np.frombuffer(lut_trans_raw, dtype=np.float16).reshape((256, 256, 4)).astype(np.float32)

    assert not np.isnan(lut_data).any(), "NaN found in Sky-View LUT output!"
    assert not np.isinf(lut_data).any(), "Inf found in Sky-View LUT output!"
    assert not np.isnan(lut_trans_data).any(), "NaN found in Sky-View Spectral Transmittance output!"
    assert not np.isinf(lut_trans_data).any(), "Inf found in Sky-View Spectral Transmittance output!"

    scatter_rgb = lut_data[:, :, 0:3]
    trans_rgb = lut_trans_data[:, :, 0:3]
    print(f"   [+] Sky-View dual-target LUT rendered successfully.")
    print(f"       Scatter RGB min: {scatter_rgb.min():.6f}, max: {scatter_rgb.max():.6f}, mean: {scatter_rgb.mean():.6f}")
    print(f"       Spectral Transmittance RGB min: {trans_rgb.min(axis=(0,1))}, max: {trans_rgb.max(axis=(0,1))}")
    print(f"       Spectral Transmittance RGB mean: R={trans_rgb[:,:,0].mean():.6f}, G={trans_rgb[:,:,1].mean():.6f}, B={trans_rgb[:,:,2].mean():.6f}")

    assert scatter_rgb.max() > 0.0, "Sky-View LUT scatter RGB is entirely zero!"
    assert trans_rgb.min() >= 0.0 and trans_rgb.max() <= 1.05, "Transmittance out of physical [0, 1] range!"
    assert trans_rgb[:,:,0].mean() > trans_rgb[:,:,2].mean(), "Expected Red transmittance > Blue transmittance due to Rayleigh scattering!"
    print(f"   [+] Verified physical spectral extinction ordering: T_red ({trans_rgb[:,:,0].mean():.4f}) > T_blue ({trans_rgb[:,:,2].mean():.4f})!")

    # 4. Compile atmo shader and test Mode 3 evaluation
    print("\n4. Compiling atmo shader program...")
    prog_atmo = ctx.program(
        vertex_shader=atmo_vertex_shader,
        fragment_shader=atmo_fragment_shader
    )
    assert prog_atmo is not None, "Failed to compile prog_atmo"
    print("   [+] prog_atmo compiled successfully.")

    # 5. Set up test atmosphere rendering pass
    print("\n5. Setting up scene FBO & icosphere atmosphere mesh...")
    w, h = 800, 600
    color_tex = ctx.texture((w, h), 4, dtype='f2')
    depth_tex = ctx.depth_texture((w, h))
    scene_fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

    mesh_verts, mesh_idx = create_icosphere_mesh(subdivisions=3)
    vbo = ctx.buffer(mesh_verts.astype(np.float32).tobytes())
    ibo = ctx.buffer(mesh_idx.astype(np.uint32).tobytes())
    vao_atmo = ctx.vertex_array(prog_atmo, [(vbo, '3f 12x', 'in_position')], index_buffer=ibo)

    # SceneData UBO (binding = 1)
    UBO_SIZE = 7328
    scene_ubo = ctx.buffer(reserve=UBO_SIZE)
    scene_ubo.bind_to_uniform_block(1)
    ubo_staging = np.zeros(UBO_SIZE // 4, dtype=np.float32)

    # AllInstances SSBO (binding = 2)
    instances_ssbo = ctx.buffer(reserve=1024)
    instances_ssbo.bind_to_storage_buffer(binding=2)
    instances_staging = np.zeros(64, dtype=np.float32)
    # Ring mask = 1 (active ring plane 0)
    instances_staging[3 * 4 + 3] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.float32)[0]
    instances_ssbo.write(instances_staging.tobytes())

    # Ring parameters: 1 ring plane
    ring_centers = np.zeros((16, 3), dtype=np.float32)
    ring_normals = np.zeros((16, 3), dtype=np.float32)
    ring_params = np.zeros((16, 4), dtype=np.float32)
    ring_normals[0, 1] = 1.0 # normal along Y
    # Saturn ring inner ~1.14489, outer ~2.26624 in AU units
    ring_params[0, 0] = (planet_r_km * 1.14489) / AU_TO_KM
    ring_params[0, 1] = (planet_r_km * 2.26624) / AU_TO_KM
    ring_params[0, 2] = 0.8 # Opacity

    dummy_gradient = np.full((1024, 16, 4), 200, dtype=np.uint8)
    ring_gradient_tex = ctx.texture((1024, 16), 4, dummy_gradient.tobytes())

    dummy_cdf = ctx.texture3d((1, 1, 1), 1, dtype='f4')
    dummy_map = ctx.texture((1, 1), 4, dtype='f4')

    ring_shadow_tex = ctx.texture((4096, 1), 4, dtype='f4')
    ring_shadow_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    ring_shadow_tex.repeat_x = False
    ring_shadow_tex.repeat_y = False
    shadow_data = np.zeros((1, 4096, 4), dtype='f4')
    shadow_data[0, :, 3] = 0.8
    ring_shadow_tex.write(shadow_data.tobytes())
    ring_shadow_tex.build_mipmaps()

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
    if 'u_ring_shadow_tex' in prog_atmo: prog_atmo['u_ring_shadow_tex'].value = 13
    if 'u_transmittance_lut' in prog_atmo: prog_atmo['u_transmittance_lut'].value = 1
    if 'u_multi_scatter_lut' in prog_atmo: prog_atmo['u_multi_scatter_lut'].value = 3
    if 'u_ringshine_lut' in prog_atmo: prog_atmo['u_ringshine_lut'].value = 6
    if 'u_ringshine_cdf_lut' in prog_atmo: prog_atmo['u_ringshine_cdf_lut'].value = 7
    if 'u_ringshine_map' in prog_atmo: prog_atmo['u_ringshine_map'].value = 8
    if 'u_depth_texture' in prog_atmo: prog_atmo['u_depth_texture'].value = 9
    if 'u_sky_view_lut' in prog_atmo: prog_atmo['u_sky_view_lut'].value = 12
    if 'u_sky_view_trans_lut' in prog_atmo: prog_atmo['u_sky_view_trans_lut'].value = 14

    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view_mat = matrix44.create_look_at(cam_pos, np.array([0.0, 0.0, 0.0], dtype=np.float32), cam_up, dtype='f4')

    fov = 55.0
    aspect = w / h
    near = 1e-4
    far = 100.0
    proj_mat = matrix44.create_perspective_projection_matrix(fov, aspect, near, far, dtype='f4')

    inv_proj = np.linalg.inv(proj_mat).astype('f4')
    inv_view = np.linalg.inv(view_mat).astype('f4')

    # Pack scene_ubo
    ubo_staging[0:16] = proj_mat.ravel()
    ubo_staging[16:32] = view_mat.ravel()
    ubo_staging[32:33].view(np.int32)[0] = 1 # 1 star
    ubo_staging[36:40] = [sun_dir[0] * 1.0, sun_dir[1] * 1.0, sun_dir[2] * 1.0, 0.00465]
    ubo_staging[100:104] = [sun_color[0], sun_color[1], sun_color[2], 1.0]
    ubo_staging[292] = far
    ubo_staging[293] = 1.0 # depth_C
    scene_ubo.write(ubo_staging.tobytes())

    if 'u_inv_proj' in prog_atmo: prog_atmo['u_inv_proj'].write(inv_proj.tobytes())
    if 'u_inv_view' in prog_atmo: prog_atmo['u_inv_view'].write(inv_view.tobytes())
    if 'u_camera_pos' in prog_atmo: prog_atmo['u_camera_pos'].write(cam_pos.tobytes())
    if 'u_screen_res' in prog_atmo: prog_atmo['u_screen_res'].value = (float(w), float(h))
    if 'u_atmo_clip_mode' in prog_atmo: prog_atmo['u_atmo_clip_mode'].value = 0
    if 'u_num_ring_planes' in prog_atmo: prog_atmo['u_num_ring_planes'].value = 1
    if 'u_ring_center' in prog_atmo: prog_atmo['u_ring_center'].write(ring_centers.tobytes())
    if 'u_ring_normal' in prog_atmo: prog_atmo['u_ring_normal'].write(ring_normals.tobytes())
    if 'u_ring_params' in prog_atmo: prog_atmo['u_ring_params'].write(ring_params.tobytes())

    prog_atmo['u_atmo_quality'].value = 3
    if 'u_atmo_slicing_steps' in prog_atmo: prog_atmo['u_atmo_slicing_steps'].value = 8
    if 'u_density' in prog_atmo: prog_atmo['u_density'].value = 24.0
    if 'u_atmo_tint' in prog_atmo: prog_atmo['u_atmo_tint'].value = tuple(atmo_tint)

    for test_steps in [2, 8, 32]:
        print(f"\n6. Rendering atmosphere mesh with Mode 3 (Analytical, u_atmo_slicing_steps={test_steps})...")
        if 'u_atmo_slicing_steps' in prog_atmo:
            prog_atmo['u_atmo_slicing_steps'].value = test_steps
        scene_fbo.use()
        ctx.viewport = (0, 0, w, h)
        scene_fbo.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
        ctx.cull_face = 'front'
        ctx.enable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False

        vao_atmo.render(moderngl.TRIANGLES)

        scene_raw = scene_fbo.read(components=4, dtype='f2')
        scene_data = np.frombuffer(scene_raw, dtype=np.float16).reshape((h, w, 4)).astype(np.float32)

        assert not np.isnan(scene_data).any(), f"NaN in scene output for Mode 3 with steps={test_steps}!"
        assert not np.isinf(scene_data).any(), f"Inf in scene output for Mode 3 with steps={test_steps}!"

        scene_rgb = scene_data[:, :, 0:3]
        print(f"   [+] Atmosphere mesh rendered successfully with steps={test_steps}.")
        print(f"       Scene RGB min: {scene_rgb.min():.6f}, max: {scene_rgb.max():.6f}, mean: {scene_rgb.mean():.6f}")

        assert scene_rgb.max() > 0.0, f"Scene atmosphere render is completely black with steps={test_steps}!"
        print(f"   [+] Mode 3 produced valid non-zero HDR in-scattering with steps={test_steps}!")

    # 7. Test Dual-Source Blending and Spectral Light Extinction
    print("\n7. Testing Mode 3 Dual-Source Blending (glBlendFunc(GL_ONE, GL_SRC1_COLOR)) & Background Extinction...")
    # Render unblended (black background) to get pure inscattering
    scene_fbo.use()
    ctx.viewport = (0, 0, w, h)
    scene_fbo.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
    ctx.disable(moderngl.BLEND)
    ctx.cull_face = 'front'
    ctx.enable(moderngl.CULL_FACE)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.depth_mask = False
    vao_atmo.render(moderngl.TRIANGLES)
    inscatter_raw = scene_fbo.read(components=4, dtype='f2')
    inscatter_data = np.frombuffer(inscatter_raw, dtype=np.float16).reshape((h, w, 4)).astype(np.float32)[:, :, 0:3]

    # Render blended over pure white background (1.0, 1.0, 1.0)
    scene_fbo.use()
    ctx.viewport = (0, 0, w, h)
    scene_fbo.clear(color=(1.0, 1.0, 1.0, 1.0), depth=1.0)
    ctx.enable(moderngl.BLEND)
    gl.glBlendFunc(gl.GL_ONE, gl.GL_SRC1_COLOR)
    vao_atmo.render(moderngl.TRIANGLES)
    ctx.disable(moderngl.BLEND)
    blended_raw = scene_fbo.read(components=4, dtype='f2')
    blended_data = np.frombuffer(blended_raw, dtype=np.float16).reshape((h, w, 4)).astype(np.float32)[:, :, 0:3]

    # Transmitted background light is: Blended - Inscatter (since Background was 1.0)
    transmitted_bg = np.clip(blended_data - inscatter_data, 0.0, 1.0)
    atmo_mask = inscatter_data.max(axis=-1) > 0.005
    assert atmo_mask.any(), "No atmosphere pixels detected!"

    mean_trans_r = float(transmitted_bg[atmo_mask, 0].mean())
    mean_trans_g = float(transmitted_bg[atmo_mask, 1].mean())
    mean_trans_b = float(transmitted_bg[atmo_mask, 2].mean())

    print(f"   [+] Dual-source blending executed successfully.")
    print(f"       Extinguished Background RGB mean inside atmosphere: R={mean_trans_r:.6f}, G={mean_trans_g:.6f}, B={mean_trans_b:.6f}")
    assert mean_trans_r > mean_trans_b, f"Expected Red transmittance ({mean_trans_r:.4f}) > Blue transmittance ({mean_trans_b:.4f}) through atmosphere!"
    print(f"   [+] Physical spectral extinction verified: Background red light transmits more than blue light (R > B)!")

    print("\n" + "=" * 70)
    print("ALL MODE 3 PIPELINE TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    test_mode3_pipeline()
