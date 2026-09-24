"""
GPU regression test: Mode 3 (Sky-View LUT) Polar Winter Solstice Parity.

Verifies that atmosphere Mode 3 (Analytical Sky-View LUT) exhibits the iconic
SpaceEngine polar winter effect on ringed planets (such as Saturn) just like Mode 2:
1. When solstice_factor > 0, sun_pole_dot != 0, and ring_mask != 0:
   - Winter pole (hemisphere pointing away from the star) suppresses aerosol Mie haze
     (polar_haze_factor drops to 0.05) and boosts pure Rayleigh in-scatter with azure tint
     vec3(0.65, 0.95, 2.5).
   - In the Sky-View LUT, the winter pole shows a dramatically higher Blue-to-Red (B/R)
     ratio and cleared haze compared to the summer pole.
2. Inactive conditions:
   - When rings are absent (ring_mask == 0), the effect is completely inactive.
   - At equinox (solstice_factor == 0), the effect is completely inactive.
3. atmo.frag and sky_view_lut.frag both compile and validate without errors.
"""
import os
import sys
import numpy as np
import moderngl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLSL = os.path.join(ROOT, "engine", "glsl")


def load(rel):
    """Resolve #include directives exactly like engine/rendering/shader_loader.py."""
    import re
    inc_re = re.compile(r'^\s*#include\s+["<]([^">]+)[">]\s*$', re.MULTILINE)
    path = os.path.join(GLSL, rel)

    def repl(m):
        inc = m.group(1)
        c1 = os.path.join(os.path.dirname(path), inc)
        c2 = os.path.join(GLSL, inc)
        target = c1 if os.path.isfile(c1) else c2
        if not os.path.isfile(target):
            raise FileNotFoundError(f"include not found: {inc}")
        return load(os.path.relpath(target, GLSL).replace("\\", "/"))

    return inc_re.sub(repl, open(path, encoding="utf-8").read())


def make_context():
    try:
        return moderngl.create_standalone_context(require=460)
    except Exception as exc:
        print(f"standalone 4.6 context unavailable ({exc}); falling back to hidden GLFW window")
        import glfw
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        win = glfw.create_window(64, 64, "offscreen", None, None)
        glfw.make_context_current(win)
        return moderngl.create_context(require=460)


# --- Saturn-like atmosphere configuration ---
R_PLANET_KM = 58232.0
R_ATMO_KM = 58832.0
AU_TO_KM = 149597870.7
H_R, H_M = 60.0, 30.0
BETA_R = (5.8e-6, 1.35e-5, 3.3e-5)      # Saturn haze/Rayleigh parameters
BETA_M = (2.0e-5, 2.0e-5, 2.0e-5)      # Moderate Mie haze
BETA_ABS_MIXED = (1.0e-7, 1.0e-7, 1.0e-7)
BETA_ABS_LAYERED = (0.0, 0.0, 0.0)
O3_PEAK, O3_WIDTH = 25.0, 8.0
MIE_G = 0.75
CAM_DIST_KM = 200000.0                   # Space observer looking from equatorial plane


def pack_atmo(solstice_factor=1.0, sun_pole_dot=0.4, pole=(0.0, 1.0, 0.0, 1.0)):
    f = np.zeros(256, dtype=np.float32)
    f[0:3] = (0.0, 0.0, 0.0)               # u_body_offset
    f[3] = R_ATMO_KM / AU_TO_KM            # u_atmo_radius_au
    f[4:7] = BETA_R                        # u_beta_rayleigh
    f[7] = H_R                             # u_h_rayleigh
    f[8:11] = BETA_M                       # u_beta_mie
    f[11] = H_M                            # u_h_mie
    f[12:15] = BETA_ABS_MIXED              # u_beta_abs_mixed
    f[15] = MIE_G                          # u_mie_g
    f[16:19] = BETA_ABS_LAYERED            # u_beta_abs_layered
    f[19] = 1.0                            # u_sun_intensity
    f[20] = R_PLANET_KM                    # u_planet_radius_km
    f[21] = R_ATMO_KM                      # u_atmo_radius_km
    f[22] = AU_TO_KM                       # u_au_to_km
    f[23] = 32.0                           # u_num_samples
    f[24:28] = pole                        # u_pole_obl
    f[32:35] = (1.0, 1.0, 1.0)             # u_mie_albedo
    f[35] = 0.00029                        # u_refractivity
    f[180] = O3_PEAK
    f[181] = O3_WIDTH
    f[182] = R_PLANET_KM - 0.1
    f[183] = 64.0
    f[184:188] = (1.0 / H_R, 1.0 / H_M, 1.0 / O3_WIDTH, 0.0)
    g = MIE_G
    f[188:192] = ((3.0 / (8.0 * np.pi)) * ((1.0 - g * g) / (2.0 + g * g)),
                  1.0 + g * g, 2.0 * g, 0.0)

    # Primary star direction: Sun tilted slightly toward North (+Y)
    sun_dir = np.array([0.9165, sun_pole_dot, 0.0], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    sin_s = 0.00465
    eff = sin_s
    cos_eff = float(np.sqrt(max(0.0, 1.0 - eff * eff)))
    f[192:196] = (1.0, 1.0, 1.0, sin_s)
    f[208:212] = (*sun_dir, cos_eff)
    f[224:228] = (*(sun_dir * AU_TO_KM), eff)
    f[240:244] = (solstice_factor, sun_pole_dot, 1.0, 0.00465)
    return f.tobytes()


def pack_scene():
    b = np.zeros(6304 // 4, dtype=np.float32)
    b.view(np.int32)[128 // 4] = 1          # 1 star
    b[656 // 4: 656 // 4 + 4] = (0.0, 1.0, 0.0, 0.0)
    return b.tobytes()


def set_lut_uniforms(prog):
    prog["u_planet_radius_km"].value = R_PLANET_KM
    prog["u_atmo_radius_km"].value = R_ATMO_KM
    prog["u_h_rayleigh"].value = H_R
    prog["u_h_mie"].value = H_M
    prog["u_beta_rayleigh"].value = BETA_R
    prog["u_beta_mie"].value = BETA_M
    prog["u_beta_abs_mixed"].value = BETA_ABS_MIXED
    prog["u_beta_abs_layered"].value = BETA_ABS_LAYERED
    if "u_mie_albedo" in prog:
        prog["u_mie_albedo"].value = (1.0, 1.0, 1.0)
    prog["u_ozone_peak_km"].value = O3_PEAK
    prog["u_ozone_width_km"].value = O3_WIDTH


def main():
    ctx = make_context()
    print("GL:", ctx.info["GL_VERSION"])

    vs = load("atmosphere/atmo_lut.vert")
    quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())

    # Precomputed transmittance LUT
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
    ms_fbo.use()
    vao_ms = ctx.vertex_array(prog_ms, [(quad_vbo, "2f", "in_position")])
    vao_ms.render(moderngl.TRIANGLE_STRIP)
    ms_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # Sky-View LUT setup
    LUT_W, LUT_H = 256, 256
    sky_texes = [ctx.texture((LUT_W, LUT_H), 4, dtype="f4") for _ in range(6)]
    sky_fbo = ctx.framebuffer(color_attachments=sky_texes)
    prog_sky = ctx.program(vertex_shader=vs, fragment_shader=load("atmosphere/sky_view_lut.frag"))

    trans_tex.use(location=1)
    ms_tex.use(location=3)
    prog_sky["u_transmittance_lut"].value = 1
    prog_sky["u_multi_scatter_lut"].value = 3
    prog_sky["u_num_ring_planes"].value = 0
    prog_sky["u_planetshine_enabled"].value = False
    prog_sky["u_ringshine_enabled"].value = False
    prog_sky["u_atmo_shadow_method"].value = 0
    prog_sky["u_num_steps"].value = 32

    # Camera at equator looking along +Z:
    # cam_pos = (0, 0, CAM_DIST_KM). Zenith is (0, 0, 1).
    # Sun direction is along +X with +Y component.
    cam_pos = np.array([0.0, 0.0, CAM_DIST_KM], dtype=np.float32)
    sun_dir = np.array([0.9165, 0.4, 0.0], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    prog_sky["u_cam_pos"].value = tuple(cam_pos)
    prog_sky["u_sun_dir"].value = tuple(sun_dir)

    vao_sky = ctx.vertex_array(prog_sky, [(quad_vbo, "2f", "in_position")])

    ssbo = ctx.buffer(reserve=1024)
    ssbo.bind_to_storage_buffer(binding=8)

    scene_ubo = ctx.buffer(reserve=6304)
    scene_ubo.bind_to_uniform_block(1)
    scene_ubo.write(pack_scene())

    inst_buf = np.zeros(7 * 4, dtype=np.float32)
    inst_ssbo = ctx.buffer(inst_buf.tobytes())
    inst_ssbo.bind_to_storage_buffer(binding=2)

    def bake_and_read(solstice_factor, sun_pole_dot, ring_mask):
        ssbo.write(pack_atmo(solstice_factor, sun_pole_dot))
        inst_buf[:] = 0.0
        inst_buf[15] = np.uint32(ring_mask).view(np.float32)
        inst_ssbo.write(inst_buf.tobytes())

        cur_sun_dir = np.array([0.9165, sun_pole_dot, 0.0], dtype=np.float32)
        cur_sun_dir /= np.linalg.norm(cur_sun_dir)
        prog_sky["u_sun_dir"].value = tuple(cur_sun_dir)

        sky_fbo.use()
        vao_sky.render(moderngl.TRIANGLE_STRIP)
        ctx.finish()

        raw = sky_texes[0].read()
        return np.frombuffer(raw, dtype=np.float32).reshape((LUT_H, LUT_W, 4))

    # Test 1: Saturn at Solstice with rings (ring_mask = 1)
    # Sun is shining from +Y (North), so South Pole (-Y) is Winter Pole!
    lut_solstice_rings = bake_and_read(solstice_factor=1.0, sun_pole_dot=0.4, ring_mask=1)

    # In the space observer branch of sky_view_lut.frag:
    # Zenith = (0, 0, 1). x_basis = L_proj. y_basis = cross(Zenith, x_basis) ~ (0, 1, 0) (North!)
    # Azimuth phi = 2*PI * f_uv.x.
    # At phi = PI/2 (u = 0.25): direction is +y_basis -> North Pole (Summer).
    # At phi = 3*PI/2 (u = 0.75): direction is -y_basis -> South Pole (Winter).
    # Let's inspect the limb rows (f_uv.y > 0.5, e.g. row ~180-220):
    col_north = int(LUT_W * 0.25)
    col_south = int(LUT_W * 0.75)
    row_limb = int(LUT_H * 0.75)  # Limb tangent altitude

    north_sample = lut_solstice_rings[row_limb, col_north, :3]
    south_sample = lut_solstice_rings[row_limb, col_south, :3]

    print(f"Solstice + Rings: North (Summer) RGB = {north_sample}")
    print(f"Solstice + Rings: South (Winter) RGB = {south_sample}")

    ratio_north_br = north_sample[2] / max(1e-6, north_sample[0])
    ratio_south_br = south_sample[2] / max(1e-6, south_sample[0])
    print(f"  North Pole B/R ratio: {ratio_north_br:.3f}")
    print(f"  South Pole B/R ratio: {ratio_south_br:.3f}")

    assert ratio_south_br > ratio_north_br * 1.5, (
        f"Winter pole should have significantly higher B/R azure tint! Got {ratio_south_br:.3f} vs {ratio_north_br:.3f}"
    )

    # Test 2: Unringed planet (ring_mask = 0) at solstice
    # Has_rings = 0 -> polar winter effect should NOT trigger!
    lut_solstice_norings = bake_and_read(solstice_factor=1.0, sun_pole_dot=0.4, ring_mask=0)
    south_sample_norings = lut_solstice_norings[row_limb, col_south, :3]
    ratio_south_norings = south_sample_norings[2] / max(1e-6, south_sample_norings[0])
    print(f"Solstice + No Rings: South RGB = {south_sample_norings}, B/R = {ratio_south_norings:.3f}")

    # Without rings, south pole should NOT receive the azure boost
    assert abs(ratio_south_norings - ratio_north_br) < 0.25, (
        f"Unringed planet should not trigger polar winter! Got B/R {ratio_south_norings:.3f} vs {ratio_north_br:.3f}"
    )

    # Test 3: Ringed planet at Equinox (solstice_factor = 0.0)
    lut_equinox = bake_and_read(solstice_factor=0.0, sun_pole_dot=0.0, ring_mask=1)
    north_eq = lut_equinox[row_limb, col_north, :3]
    south_eq = lut_equinox[row_limb, col_south, :3]
    ratio_north_eq = north_eq[2] / max(1e-6, north_eq[0])
    ratio_south_eq = south_eq[2] / max(1e-6, south_eq[0])
    print(f"Equinox: North B/R = {ratio_north_eq:.3f}, South B/R = {ratio_south_eq:.3f}")
    assert abs(ratio_north_eq - ratio_south_eq) < 0.1, (
        f"Equinox poles should be symmetric! Got {ratio_north_eq:.3f} vs {ratio_south_eq:.3f}"
    )

    # Compile smoke test for atmo.vert + atmo.frag
    prog_atmo = ctx.program(
        vertex_shader=load("atmosphere/atmo.vert"),
        fragment_shader=load("atmosphere/atmo.frag")
    )
    assert prog_atmo is not None
    print("compile-smoke atmo.vert+atmo.frag: OK")

    print("\nPASS: Mode 3 Sky-View LUT polar winter solstice model correctly reproduces Cassini/SpaceEngine azure pole for ringed planets!")


if __name__ == "__main__":
    main()
