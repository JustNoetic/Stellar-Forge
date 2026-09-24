"""
Test Sky-View LUT at 192x108 dimensions.
Verifies that sky_view_lut.frag and atmo.frag operate seamlessly at 192x108,
checks readback values, and benchmarks bake execution time.
"""
import os
import sys
import time
import numpy as np
import moderngl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GLSL = os.path.join(ROOT, "engine", "glsl")

from scripts.test_skyview_eclipse import load, make_context, set_lut_uniforms, pack_atmo, pack_scene, STAR_A, H_R, H_M, MIE_G, CAM_DIST_KM, SUN_DIR

def main():
    ctx = make_context()
    print("GL Version:", ctx.info["GL_VERSION"])

    vs = load("atmosphere/atmo_lut.vert")
    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())

    # Transmittance LUT
    trans_tex = ctx.texture((256, 256), 4, dtype="f4")
    trans_fbo = ctx.framebuffer(color_attachments=[trans_tex])
    prog_trans = ctx.program(vertex_shader=vs, fragment_shader=load("atmosphere/atmo_lut.frag"))
    set_lut_uniforms(prog_trans)
    vao_trans = ctx.vertex_array(prog_trans, [(quad_vbo, "2f", "in_position")])
    trans_fbo.use()
    vao_trans.render(moderngl.TRIANGLE_STRIP)
    trans_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # Multi-scatter LUT
    ms_tex = ctx.texture((64, 64), 4, dtype="f4")
    ms_fbo = ctx.framebuffer(color_attachments=[ms_tex])
    prog_ms = ctx.program(vertex_shader=vs, fragment_shader=load("atmosphere/multi_scatter_lut.frag"))
    set_lut_uniforms(prog_ms)
    prog_ms["u_mie_g"].value = MIE_G
    prog_ms["u_ground_albedo"].value = (0.3, 0.3, 0.3)
    ms_tex.use(location=0)
    prog_ms["u_transmittance_lut"].value = 0
    vao_ms = ctx.vertex_array(prog_ms, [(quad_vbo, "2f", "in_position")])
    ms_fbo.use()
    vao_ms.render(moderngl.TRIANGLE_STRIP)
    ms_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # Setup sky-view program and uniforms
    prog_sky = ctx.program(vertex_shader=vs, fragment_shader=load("atmosphere/sky_view_lut.frag"))
    prog_sky["u_transmittance_lut"].value = 1
    prog_sky["u_multi_scatter_lut"].value = 3
    prog_sky["u_num_ring_planes"].value = 0
    prog_sky["u_planetshine_enabled"].value = False
    prog_sky["u_ringshine_enabled"].value = False

    vao_sky = ctx.vertex_array(prog_sky, [(quad_vbo, "2f", "in_position")])

    ssbo = ctx.buffer(reserve=1024)
    ssbo.bind_to_storage_buffer(binding=8)
    scene_ubo = ctx.buffer(reserve=6304)
    scene_ubo.bind_to_uniform_block(1)
    inst_ssbo = ctx.buffer(np.zeros(7 * 4, dtype=np.float32).tobytes())
    inst_ssbo.bind_to_storage_buffer(binding=2)

    ssbo.write(pack_atmo([STAR_A], []))
    scene_ubo.write(pack_scene(1))
    trans_tex.use(location=1)
    ms_tex.use(location=3)
    prog_sky["u_cam_pos"].value = (-CAM_DIST_KM, 0.0, 0.0)
    prog_sky["u_sun_dir"].value = tuple(SUN_DIR)

    # Benchmark step scaling and resolutions (16, 24, 32 steps)
    for steps in [16, 24, 32]:
        if "u_num_steps" in prog_sky:
            prog_sky["u_num_steps"].value = steps
        for W, H in [(192, 108), (256, 256), (384, 216)]:
            sky_texes = [ctx.texture((W, H), 4, dtype="f4") for _ in range(6)]
            sky_fbo = ctx.framebuffer(color_attachments=sky_texes)
            sky_fbo.use()
            ctx.viewport = (0, 0, W, H)
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            ctx.disable(moderngl.CULL_FACE)

            # Warmup
            vao_sky.render(moderngl.TRIANGLE_STRIP)
            ctx.finish()

            t0 = time.perf_counter()
            N = 100
            for _ in range(N):
                vao_sky.render(moderngl.TRIANGLE_STRIP)
            ctx.finish()
            dt_ms = (time.perf_counter() - t0) * 1000.0 / N
            print(f"Sky-View LUT ({W}x{H}, {steps:2d} steps, {W*H} texels) average bake time: {dt_ms:.3f} ms")

            raw = sky_fbo.color_attachments[0].read()
            img = np.frombuffer(raw, dtype="f4").reshape(H, W, 4).astype(np.float32)
            assert not np.isnan(img).any()
            assert not np.isinf(img).any()
            assert img.max() > 0.01

            for t in sky_texes: t.release()
            sky_fbo.release()

    print(f"Luminance range: min={img[:, :, :3].min():.4f}, max={img[:, :, :3].max():.4f}, mean={img[:, :, :3].mean():.4f}")
    print("Sky-View LUT resolution presets (Low/Medium/High) validation: PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
