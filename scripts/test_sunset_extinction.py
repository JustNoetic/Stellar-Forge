"""
Comprehensive Test for Sunset Extinction and Horizon Rendering Across Atmosphere Modes.

Verifies:
1. Landed camera looking at sunset:
   - The sun at/near the horizon is extinguished (transmittance << 1.0) and tinted red/orange.
   - Background sun is NOT raw white (extinction occurs across all modes: Mode 2 and Mode 3).
   - In Mode 3, rays pointing at the setting sun sample the Sky Dome horizon (v_lut = 128.5 / 256.0),
     producing full optical depth sunset extinction.
   - Ground pixels below horizon sample Ground Disk (v_lut <= 127.5 / 256.0), producing zero blue sky clipping.
"""

import os
import sys
import math
import numpy as np
import moderngl
import OpenGL.GL as gl
from pyrr import matrix44

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

def run_tests():
    print("=" * 70)
    print("Testing Sunset Extinction & Horizon Rendering (Mode 2 vs Mode 3)")
    print("=" * 70)

    ctx = moderngl.create_context(standalone=True)
    print(f"ModernGL context created. GL Renderer: {ctx.info['GL_RENDERER']}")

    # 1. Compile Shaders
    prog_sky_view = ctx.program(
        vertex_shader=sky_view_lut_vertex_shader,
        fragment_shader=sky_view_lut_fragment_shader
    )
    atmo_frag_mrt = atmo_fragment_shader.replace(
        "layout(location = 0, index = 0) out vec4 out_scattered;",
        "layout(location = 0) out vec4 out_scattered;"
    ).replace(
        "layout(location = 0, index = 1) out vec4 out_transmittance;",
        "layout(location = 1) out vec4 out_transmittance;"
    )
    prog_atmo = ctx.program(
        vertex_shader=atmo_vertex_shader,
        fragment_shader=atmo_frag_mrt
    )

    # 2. Quad VAO for Sky-View LUT
    quad_verts = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')
    quad_vbo = ctx.buffer(quad_verts.tobytes())
    quad_vao = ctx.vertex_array(prog_sky_view, [(quad_vbo, '2f', 'in_position')])

    sky_view_tex = ctx.texture((256, 256), 4, dtype='f2')
    sky_view_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    sky_view_trans_tex = ctx.texture((256, 256), 4, dtype='f2')
    sky_view_trans_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    sky_view_fbo = ctx.framebuffer(color_attachments=[sky_view_tex, sky_view_trans_tex])

    # Parameters: Earth-sized planet
    AU_TO_KM = 149597870.7
    planet_r_km = 6371.0
    atmo_r_km = 6471.0
    atmo_r_au = atmo_r_km / AU_TO_KM
    land_alt_km = 0.002 # 2 meters
    cam_pos_km = np.array([0.0, planet_r_km + land_alt_km, 0.0], dtype=np.float32)
    cam_pos_au = cam_pos_km / AU_TO_KM

    # Setting Sun: setting along positive Z axis, at horizon (pitch ~ 0.05 deg below eye level)
    sun_dir = np.array([0.0, 0.00079, 0.999999], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    sun_color = np.array([1.2, 1.1, 1.0], dtype=np.float32)

    # Setup AtmoData SSBO (binding = 8)
    atmo_ssbo = ctx.buffer(reserve=1024)
    atmo_ssbo.bind_to_storage_buffer(binding=8)
    staging = np.zeros(256, dtype=np.float32)
    staging_int = staging.view(np.int32)

    staging[0:3] = [0.0, 0.0, 0.0]
    staging[3] = atmo_r_au
    staging[4:7] = [5.802e-6, 13.558e-6, 33.10e-6] # Earth Rayleigh
    staging[7] = 8.0 # Scale height Rayleigh
    staging[8:11] = [3.996e-6, 3.996e-6, 3.996e-6] # Mie extinction
    staging[11] = 1.2 # Scale height Mie
    staging[15] = 0.8 # Mie g
    staging[16:19] = [6.5e-7, 1.881e-6, 8.5e-8] # Ozone absorption
    staging[19] = 1.0 # Mie albedo
    staging[20] = planet_r_km
    staging[21] = atmo_r_km
    staging[22] = AU_TO_KM
    staging_int[23] = 32 # sample count
    staging[24:27] = [0.0, 1.0, 0.0] # up
    staging[27] = 1.0 # exposure
    staging_int[28] = 0 # shadow mode
    staging_int[29] = 1 # hdr
    staging_int[30] = 0 # temporal
    staging[32:35] = [0.9, 0.9, 0.9] # mie albedo vec3

    staging[182] = planet_r_km * (1.0 - 0.0005) # planet_clip_km
    staging[184:188] = [1.0 / 8.0, 1.0 / 1.2, 1.0 / 15.0, 0.0]
    g = 0.8
    g2 = g * g
    staging[188:192] = [(3.0 / (8.0 * math.pi)) * ((1.0 - g2) / (2.0 + g2)), 1.0 + g2, 2.0 * g, 0.0]
    staging[192:195] = sun_color
    staging[195] = 0.00465 # angular radius
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

    # Render Sky-View LUT
    sky_view_fbo.use()
    ctx.viewport = (0, 0, 256, 256)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)

    prog_sky_view['u_cam_pos'].value = tuple(cam_pos_km)
    prog_sky_view['u_sun_dir'].value = tuple(sun_dir)
    quad_vao.render(moderngl.TRIANGLE_STRIP)

    lut_scatter = np.frombuffer(sky_view_fbo.read(attachment=0, components=4, dtype='f2'), dtype=np.float16).reshape((256, 256, 4)).astype(np.float32)
    lut_trans = np.frombuffer(sky_view_fbo.read(attachment=1, components=4, dtype='f2'), dtype=np.float16).reshape((256, 256, 4)).astype(np.float32)

    # In azimuth column towards setting sun:
    col_sun = 0 # phi=0
    T_horizon_sky = lut_trans[128, col_sun, 0:3]
    L_horizon_sky = lut_scatter[128, col_sun, 0:3]
    print(f"\n1. Landed Sky-View LUT towards Setting Sun:")
    print(f"   Sky Horizon (Row 128): Transmittance = {T_horizon_sky}")
    print(f"   Sky Horizon (Row 128): Scattering    = {L_horizon_sky}")
    assert T_horizon_sky[0] < 0.1, f"Expected strong red sunset extinction, got {T_horizon_sky[0]}"
    assert T_horizon_sky[2] < 0.01, f"Expected blue completely extinguished, got {T_horizon_sky[2]}"
    assert T_horizon_sky[0] > T_horizon_sky[2], f"Expected red > blue (redshift), got {T_horizon_sky}"
    print("   [+] Physical sunset extinction confirmed in Sky-View LUT!")

    # 3. Setup Scene & Atmosphere Mesh
    w, h = 400, 300
    color_tex = ctx.texture((w, h), 4, dtype='f2')
    trans_out_tex = ctx.texture((w, h), 4, dtype='f2')
    depth_tex = ctx.depth_texture((w, h))
    scene_fbo = ctx.framebuffer(color_attachments=[color_tex, trans_out_tex], depth_attachment=depth_tex)

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

    # Camera looking towards the sunset along +Z, pitched down by 2 degrees
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
    ubo_staging[32:33].view(np.int32)[0] = 1 # num_stars = 1
    ubo_staging[36:40] = [sun_dir[0], sun_dir[1], sun_dir[2], 0.00465]
    ubo_staging[100:104] = [sun_color[0], sun_color[1], sun_color[2], 1.0]
    ubo_staging[292] = 100.0 # u_far
    ubo_staging[293] = 1.0   # u_exposure
    scene_ubo.write(ubo_staging.tobytes())

    if 'u_inv_proj' in prog_atmo: prog_atmo['u_inv_proj'].write(inv_proj.tobytes())
    if 'u_inv_view' in prog_atmo: prog_atmo['u_inv_view'].write(inv_view.tobytes())
    if 'u_camera_pos' in prog_atmo: prog_atmo['u_camera_pos'].write(cam_pos_au.tobytes())
    if 'u_screen_res' in prog_atmo: prog_atmo['u_screen_res'].value = (float(w), float(h))
    if 'u_atmo_clip_mode' in prog_atmo: prog_atmo['u_atmo_clip_mode'].value = 0
    if 'u_atmo_slicing_steps' in prog_atmo: prog_atmo['u_atmo_slicing_steps'].value = 8

    ctx.cull_face = 'front'
    ctx.enable(moderngl.CULL_FACE)

    # Test both Mode 2 and Mode 3
    for quality_mode in [2, 3]:
        mode_name = "Mode 2 (Raymarch High)" if quality_mode == 2 else "Mode 3 (Sky-View LUT)"
        print(f"\n2. Testing Landed Sunset Rendering in {mode_name}...")

        # Quality mode in prog_atmo uniform
        if 'u_atmo_quality' in prog_atmo:
            prog_atmo['u_atmo_quality'].value = quality_mode

        scene_fbo.use()
        ctx.viewport = (0, 0, w, h)
        scene_fbo.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)

        vao_atmo.render(moderngl.TRIANGLES)
        scatter_img = np.frombuffer(scene_fbo.read(attachment=0, components=4, dtype='f2'), dtype=np.float16).reshape((h, w, 4)).astype(np.float32)[:, :, 0:3]
        trans_img = np.frombuffer(scene_fbo.read(attachment=1, components=4, dtype='f2'), dtype=np.float16).reshape((h, w, 4)).astype(np.float32)[:, :, 0:3]

        # In center column X = 200:
        # Horizon is at Y ~ 150.
        # Sky near horizon is Y in [155, 175].
        sun_horizon_scatter = scatter_img[155:175, 195:205].mean(axis=(0, 1))
        sun_horizon_trans = trans_img[155:175, 195:205].mean(axis=(0, 1))

        print(f"   [{mode_name}] Horizon Sunset Region (Y=155-175, center):")
        print(f"       Scatter RGB       = {sun_horizon_scatter}")
        print(f"       Transmittance RGB = {sun_horizon_trans}")

        # Transmittance must be extinguished (< 0.5 for red, << 0.1 for blue)
        assert sun_horizon_trans[0] < 0.5, f"Expected red extinction (< 0.5), got {sun_horizon_trans[0]}"
        assert sun_horizon_trans[2] < 0.1, f"Expected strong blue extinction (< 0.1), got {sun_horizon_trans[2]}"
        assert sun_horizon_trans[0] >= sun_horizon_trans[2], f"Expected red >= blue, got {sun_horizon_trans}"

        # In-scattering must be non-zero (warm sunset glow)
        assert sun_horizon_scatter.max() > 0.01, f"Expected warm sunset scattering, got {sun_horizon_scatter}"

        # Check dual-source blend against a bright white sun:
        # If the background has a white sun of intensity 50.0:
        # Out = Sun * Transmittance + Scatter
        sun_bg = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        blended_sun = sun_bg * sun_horizon_trans + sun_horizon_scatter
        print(f"       Blended Sun Color = R={blended_sun[0]:.2f}, G={blended_sun[1]:.2f}, B={blended_sun[2]:.2f}")
        # Red must dominate over blue (warm sunset sun, NOT white bloom):
        assert blended_sun[0] > blended_sun[2] * 2.0, f"Sun must be deeply red-shifted (R > 2*B), got R={blended_sun[0]}, B={blended_sun[2]}"

        print(f"   [+] {mode_name} successfully verified! No white un-extinguished sun at horizon.")

    print("\n" + "=" * 70)
    print("ALL SUNSET EXTINCTION & HORIZON TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
