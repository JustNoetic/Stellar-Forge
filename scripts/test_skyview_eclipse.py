"""
GPU regression: Mode 3 (Sky-View LUT) must bake (a) moon/planet caster eclipses
and (b) multi-star illumination into the per-frame sky-view texture.

Scenario 1 - eclipse: Earth-like planet at the origin, Sun-like star at +Z
(1 AU, radius 0.00465 AU), and a Moon-like caster at (0, 0, 0.002) AU with
radius 1.5e-5 AU (beta > alpha -> true totality at the sub-solar point).
Space observer at (-20000, 0, 0) km, basis x_axis = +Z (sun azimuth u = 0).
The shadow cylinder around the Z axis must carve a dark spot into the LUT
texels whose rays cross the sub-solar region, while rays 90 deg off the sun
azimuth (u ~ 0.25/0.75) stay untouched.

Scenario 2 - multi-star additivity: with two equal-intensity stars A (+Z) and
B (+Y), the bake is linear in per-star contribution, so L_AB == L_A + L_B up
to fp16 readback rounding (the LUT is baked with a fixed star-0-aligned basis
for all three bakes, so texel correspondence is exact).

Pre-fix behavior: the bake ignored u_active_casters and all star slots beyond
index 0 -> eclipse ratio exactly 1.0 everywhere and L_AB == L_A -> both FAIL.
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


# --- atmosphere configuration (Earth-like, mirrors test_skyview_terminator) ---
R_PLANET_KM = 6371.0
R_ATMO_KM = 6471.0
AU_TO_KM = 149597870.7
H_R, H_M = 8.5, 1.2
BETA_R = (5.8e-6, 1.35e-2, 4.5e-2)
BETA_M = (2.1e-6, 2.1e-6, 2.1e-6)
BETA_ABS_MIXED = (1.0e-7, 1.0e-7, 1.0e-7)
BETA_ABS_LAYERED = (5.0e-7, 5.0e-7, 5.0e-7)
O3_PEAK, O3_WIDTH = 25.0, 8.0
MIE_G = 0.758
CAM_DIST_KM = 20000.0
SUN_DIR = np.array([0.0, 0.0, 1.0])

STAR_A = dict(dir=(0.0, 0.0, 1.0), dist=1.0, radius=0.00465, color=(1.0, 1.0, 1.0))
STAR_B = dict(dir=(0.0, 1.0, 0.0), dist=1.0, radius=0.00465, color=(1.0, 1.0, 1.0))  # +Y: 90 deg off the view axis like A, so both illuminate comparable visible atmosphere
MOON = dict(pos=(0.0, 0.0, 0.002), radius=1.5e-5)  # beta=7.5e-3 > alpha=4.65e-3: totality


def pack_atmo(stars, casters, body_offset=(0.0, 0.0, 0.0)):
    """Pack the AtmoData SSBO exactly like app.py's 256-float atmo_staging layout."""
    f = np.zeros(256, dtype=np.float32)
    f[0:3] = body_offset                   # u_body_offset
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
    f[24:28] = (0.0, 1.0, 0.0, 1.0)        # u_pole_obl (pole +Y, no oblate scaling)
    iv = f.view(np.int32)
    iv[28] = len(casters)                  # u_num_active_casters
    f[32:35] = (1.0, 1.0, 1.0)             # u_mie_albedo
    f[35] = 0.00029                        # u_refractivity
    for i, c in enumerate(casters):
        f[36 + 4 * i: 40 + 4 * i] = (*c["pos"], c["radius"])       # u_active_casters
        f[68 + 4 * i: 72 + 4 * i] = (0.0, 1.0, 0.0, 1.0)           # poles_obl (spherical)
        f[100 + i] = c["radius"]                                   # R_minor = R_eq
        # atmos (108..140), max_bend (140..148), ozone (148..180) stay 0: airless caster
    f[180] = O3_PEAK
    f[181] = O3_WIDTH
    f[182] = R_PLANET_KM - 0.1             # u_planet_clip_km
    f[183] = 128.0                         # u_max_adaptive_steps
    f[184:188] = (1.0 / H_R, 1.0 / H_M, 1.0 / O3_WIDTH, 0.0)  # u_precomp_opt (max_bend = 0)
    g = MIE_G
    f[188:192] = ((3.0 / (8.0 * np.pi)) * ((1.0 - g * g) / (2.0 + g * g)),
                  1.0 + g * g, 2.0 * g, 0.0)                       # u_precomp_mie
    for s, st in enumerate(stars):
        d = np.asarray(st["dir"], dtype=np.float64)
        d = d / np.linalg.norm(d)
        dist, rad = st["dist"], st["radius"]
        sin_s = rad / dist
        eff = sin_s                        # receiver max_bend = 0
        cos_eff = float(np.sqrt(max(0.0, 1.0 - eff * eff)))
        f[192 + 4 * s: 196 + 4 * s] = (*st["color"], sin_s)        # u_star_color_irrad
        f[208 + 4 * s: 212 + 4 * s] = (*d, cos_eff)                # u_star_dir_sph_eff
        f[224 + 4 * s: 228 + 4 * s] = (*(d * dist * AU_TO_KM), eff)  # u_star_pos_local
        f[240 + 4 * s: 244 + 4 * s] = (0.0, d[1], dist, rad)       # u_star_solstice
    return f.tobytes()


def pack_scene(num_stars):
    """Minimal SceneData std140 UBO (6304 bytes, layout matches atmo.frag)."""
    b = np.zeros(6304 // 4, dtype=np.float32)
    b.view(np.int32)[128 // 4] = num_stars
    for s in range(num_stars):
        b[656 // 4 + 4 * s: 656 // 4 + 4 * s + 4] = (0.0, 1.0, 0.0, 0.0)  # u_stars_poles_obl
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

    # --- transmittance LUT (config-independent) ---
    trans_tex = ctx.texture((256, 256), 4, dtype="f4")
    trans_fbo = ctx.framebuffer(color_attachments=[trans_tex])
    prog_trans = ctx.program(vertex_shader=vs, fragment_shader=load("atmosphere/atmo_lut.frag"))
    set_lut_uniforms(prog_trans)
    vao_trans = ctx.vertex_array(prog_trans, [(quad_vbo, "2f", "in_position")])
    trans_fbo.use()
    vao_trans.render(moderngl.TRIANGLE_STRIP)
    trans_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    # --- multi-scatter LUT (config-independent) ---
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

    # --- sky-view LUT program (the shader under test) ---
    prog_sky = ctx.program(vertex_shader=vs, fragment_shader=load("atmosphere/sky_view_lut.frag"))
    prog_sky["u_transmittance_lut"].value = 1
    prog_sky["u_multi_scatter_lut"].value = 3
    prog_sky["u_num_ring_planes"].value = 0
    prog_sky["u_planetshine_enabled"].value = False
    prog_sky["u_ringshine_enabled"].value = False
    if "u_ring_gradients" in prog_sky:
        prog_sky["u_ring_gradients"].value = 0
    sky_texes = [ctx.texture((256, 256), 4, dtype="f4") for _ in range(6)]
    sky_fbo = ctx.framebuffer(color_attachments=sky_texes)
    vao_sky = ctx.vertex_array(prog_sky, [(quad_vbo, "2f", "in_position")])
    ssbo = ctx.buffer(reserve=1024)
    ssbo.bind_to_storage_buffer(binding=8)
    scene_ubo = ctx.buffer(reserve=6304)
    scene_ubo.bind_to_uniform_block(1)
    inst_buf = np.zeros(7 * 4, dtype=np.float32)
    inst_ssbo = ctx.buffer(inst_buf.tobytes())
    inst_ssbo.bind_to_storage_buffer(binding=2)

    ring_grad_tex = ctx.texture((256, 16), 4, dtype="f4")
    ring_grad_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    grad_init = np.ones((16, 256, 4), dtype=np.float32)
    ring_grad_tex.write(grad_init.tobytes())

    def bake_sky_view(stars, casters, planetshine=None, rings=None, body_offset=(0.0, 0.0, 0.0), ring_mask=0):
        ssbo.write(pack_atmo(stars, casters, body_offset))
        scene_ubo.write(pack_scene(len(stars)))
        inst_buf[:] = 0.0
        if planetshine is not None:
            inst_buf[16:19] = planetshine[0]
            inst_buf[20:23] = planetshine[1]
            prog_sky["u_planetshine_enabled"].value = True
        else:
            prog_sky["u_planetshine_enabled"].value = False
        inst_buf[15] = np.uint32(ring_mask).view(np.float32)
        inst_ssbo.write(inst_buf.tobytes())

        if rings:
            n_r = len(rings)
            prog_sky["u_num_ring_planes"].value = n_r
            r_c = np.zeros((16, 3), dtype=np.float32)
            r_n = np.zeros((16, 3), dtype=np.float32)
            r_p = np.zeros((16, 4), dtype=np.float32)
            r_m = np.zeros(16, dtype=np.uint32)
            for i, r in enumerate(rings):
                r_c[i] = r.get("center", (0.0, 0.0, 0.0))
                r_n[i] = r.get("normal", (0.0, 1.0, 0.0))
                r_p[i] = r.get("params", (0.0, 0.0, 1.0, 0.0))
                r_m[i] = r.get("coplanar_mask", 1 << i)
            if "u_ring_center" in prog_sky:
                prog_sky["u_ring_center"].write(r_c.tobytes())
            if "u_ring_normal" in prog_sky:
                prog_sky["u_ring_normal"].write(r_n.tobytes())
            if "u_ring_params" in prog_sky:
                prog_sky["u_ring_params"].write(r_p.tobytes())
            if "u_ring_coplanar_mask" in prog_sky:
                prog_sky["u_ring_coplanar_mask"].write(r_m.tobytes())
            ring_grad_tex.use(location=0)
        else:
            prog_sky["u_num_ring_planes"].value = 0

        trans_tex.use(location=1)
        ms_tex.use(location=3)
        prog_sky["u_cam_pos"].value = (-CAM_DIST_KM, 0.0, 0.0)
        prog_sky["u_sun_dir"].value = tuple(SUN_DIR)  # star-0-aligned basis for all bakes
        sky_fbo.use()
        ctx.viewport = (0, 0, 256, 256)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        vao_sky.render(moderngl.TRIANGLE_STRIP)
        imgs = []
        for tex_i in range(6):
            raw = sky_fbo.color_attachments[tex_i].read()
            imgs.append(np.frombuffer(raw, dtype="f4").reshape(256, 256, 4).astype(np.float32))
        return imgs  # [combined, trans, star0, star1, star2, star3]

    failures = []

    # === Scenario 1: moon caster eclipse baked into the LUT ===
    img_clear = bake_sky_view([STAR_A], [])[0]
    img_ecl = bake_sky_view([STAR_A], [MOON])[0]

    lum_clear = img_clear[:, :, :3].sum(axis=2)
    lum_ecl = img_ecl[:, :, :3].sum(axis=2)
    ratio = lum_ecl / np.maximum(lum_clear, 1e-6)

    min_ratio = float(ratio.min())
    total_drop = 1.0 - float(lum_ecl.sum()) / max(float(lum_clear.sum()), 1e-12)
    frac_unchanged = float((ratio > 0.98).mean())
    # Control: columns 90 deg off the sun azimuth (u ~ 0.25 / 0.75) must be untouched.
    ctrl_cols = list(range(59, 70)) + list(range(187, 198))
    ctrl_diff = float(np.abs(lum_ecl[:, ctrl_cols] - lum_clear[:, ctrl_cols]).max())
    ctrl_ref = float(lum_clear[:, ctrl_cols].max())

    print(f"eclipse: min_ratio={min_ratio:.4f}  total_drop={total_drop * 100:.3f}%  "
          f"unchanged_texels={frac_unchanged * 100:.1f}%  ctrl_maxdiff={ctrl_diff:.5f} (ref {ctrl_ref:.4f})")

    if min_ratio >= 0.5:
        failures.append(f"no umbra detected in baked LUT (min texel ratio {min_ratio:.3f} >= 0.5)")
    if total_drop <= 0.002:
        failures.append(f"eclipse removed only {total_drop * 100:.3f}% of total light (expected > 0.2%)")
    # The penumbra tube (~3.6k km radius here) grazes many day-side rays and its
    # anti-solar extension dims night-side multi-scatter (same convention as the
    # Mode 1/2 ms_shadow), so only ~half of the texels stay within 2%. Locality
    # is asserted strictly via the 90-degree-off control columns below.
    if frac_unchanged <= 0.45:
        failures.append(f"shadow implausibly global: only {frac_unchanged * 100:.1f}% texels unchanged")
    if ctrl_diff > 0.02 * max(ctrl_ref, 1e-6):
        failures.append(f"control azimuths drifted by {ctrl_diff:.5f} (ref {ctrl_ref:.4f})")

    # === Scenario 2: multi-star additivity + per-star LUT slices ===
    out_a = bake_sky_view([STAR_A], [])
    out_b = bake_sky_view([STAR_B], [])
    out_ab = bake_sky_view([STAR_A, STAR_B], [])
    img_a, img_b, img_ab = out_a[0], out_b[0], out_ab[0]

    la = img_a[:, :, :3]
    lb = img_b[:, :, :3]
    lab = img_ab[:, :, :3]
    denom = np.maximum(lab, 1e-3)
    rel_err = float((np.abs(lab - la - lb) / denom).max())
    sum_a, sum_b, sum_ab = float(la.sum()), float(lb.sum()), float(lab.sum())
    # Per-star slices of the 2-star bake must reproduce the single-star bakes
    # (this is the data the Mode 3 ring-shadow slicing normalizes against).
    slice0_err = float((np.abs(out_ab[2][:, :, :3] - la) / np.maximum(la, 1e-3)).max())
    slice1_err = float((np.abs(out_ab[3][:, :, :3] - lb) / np.maximum(lb, 1e-3)).max())
    print(f"multi-star: sumA={sum_a:.4f}  sumB={sum_b:.4f}  sumAB={sum_ab:.4f}  "
          f"max_rel_err={rel_err:.5f}  slice_errs=({slice0_err:.5f}, {slice1_err:.5f})")

    if sum_b <= 0.5 * sum_a:
        failures.append(f"star-B-only bake too dim ({sum_b:.4f} vs A {sum_a:.4f}) - "
                        "symmetric geometry should give comparable totals; slot packing broken?")
    if sum_ab <= 1.7 * sum_a:
        failures.append(f"two-star bake not twice as bright as one-star ({sum_ab:.4f} vs {sum_a:.4f}) - "
                        "second star ignored")
    if rel_err >= 0.02:
        failures.append(f"multi-star bake not additive: max rel err {rel_err:.5f} >= 0.02")
    if slice0_err >= 0.02 or slice1_err >= 0.02:
        failures.append(f"per-star LUT slices disagree with single-star bakes "
                        f"({slice0_err:.5f}, {slice1_err:.5f}) - ring-shadow normalization data broken")

    # === Scenario 3: planetshine/moonshine baked as a second light ===
    # A bounce source on the night side (-Z) must light night-side sky texels
    # that receive essentially zero direct sunlight.
    img_ps = bake_sky_view([STAR_A], [], planetshine=((0.0, 0.0, -1.0), (0.5, 0.5, 0.5)))[0]
    lum_ps = img_ps[:, :, :3].sum(axis=2)
    night_rows = slice(230, 256)          # limb rows (space-observer branch)
    night_cols = slice(115, 141)          # anti-solar azimuth (u ~ 0.5)
    night_base = float(lum_clear[night_rows, night_cols].sum())
    night_ps = float(lum_ps[night_rows, night_cols].sum())
    total_ps_gain = float(lum_ps.sum() - lum_clear.sum())
    print(f"planetshine: night {night_base:.5f} -> {night_ps:.5f}  total_gain={total_ps_gain:.4f}")

    if night_ps <= night_base * 2.0 + 1e-3:
        failures.append(f"planetshine did not light the night-side sky ({night_base:.5f} -> {night_ps:.5f})")
    if total_ps_gain <= 0.01 * float(lum_clear.sum()):
        failures.append(f"planetshine total gain too small ({total_ps_gain:.4f})")

    # === Scenario 4: external ring shadow on moon atmosphere (Mode 3) ===
    # Parent planet at origin (0,0,0) with ring plane normal (0, 0.6, 0.8),
    # inner radius 0.0002 AU, outer radius 0.0008 AU, opacity 1.0.
    # Moon atmosphere at offset (0.0005, 0.0, -0.005) AU: distance > 1e-4 AU (external ring).
    # Ray along +Z hits the ring plane at distance 0.005 AU, at radial distance d = 0.0005 AU
    # (directly inside the ring's radial extent [0.0002, 0.0008]).
    RING_PARENT = dict(center=(0.0, 0.0, 0.0), normal=(0.0, 0.6, 0.8),
                       params=(0.0002, 0.0008, 1.0, 0.0), coplanar_mask=1)
    moon_offset = (0.0005, 0.0, -0.005)

    img_moon_clear = bake_sky_view([STAR_A], [], rings=[RING_PARENT], body_offset=moon_offset, ring_mask=0)[0]
    img_moon_sh = bake_sky_view([STAR_A], [], rings=[RING_PARENT], body_offset=moon_offset, ring_mask=1)[0]

    lum_m_clear = img_moon_clear[:, :, :3].sum(axis=2)
    lum_m_sh = img_moon_sh[:, :, :3].sum(axis=2)
    m_ratio = lum_m_sh / np.maximum(lum_m_clear, 1e-6)
    m_min_ratio = float(m_ratio.min())
    m_drop = 1.0 - float(lum_m_sh.sum()) / max(float(lum_m_clear.sum()), 1e-12)

    # Circumplanetary control: when body_offset is at the ring center (0, 0, 0),
    # compute_external_ring_shadow must SKIP the ring so atmo.frag can slice it.
    img_own_clear = bake_sky_view([STAR_A], [], rings=[RING_PARENT], body_offset=(0.0, 0.0, 0.0), ring_mask=0)[0]
    img_own_ring = bake_sky_view([STAR_A], [], rings=[RING_PARENT], body_offset=(0.0, 0.0, 0.0), ring_mask=1)[0]
    own_diff = float(np.abs(img_own_ring[:, :, :3] - img_own_clear[:, :, :3]).max())

    print(f"external ring: min_ratio={m_min_ratio:.4f}  drop={m_drop * 100:.3f}%  own_ring_diff={own_diff:.6f}")

    if m_min_ratio >= 0.5:
        failures.append(f"external ring shadow not detected in moon Sky-View LUT (min_ratio {m_min_ratio:.3f} >= 0.5)")
    if m_drop <= 0.002:
        failures.append(f"external ring removed only {m_drop * 100:.3f}% of total light (expected > 0.2%)")
    if own_diff > 1e-5:
        failures.append(f"circumplanetary (own) ring was not skipped by sky_view_lut (diff {own_diff:.6f} > 1e-5)")

    # === Scenario 5: FP32 low-brightness precision (no FP16 subnormal quantization) ===
    # Faint secondary bounce light with flux ~ 1e-5 to 1e-7.
    # In FP16 this suffered severe color banding with only 1-4 bits of precision.
    # In FP32 (f4), hundreds of continuous floating-point gradient values must be preserved.
    img_faint_ps = bake_sky_view([STAR_A], [], planetshine=((0.0, 0.0, -1.0), (1e-5, 1e-5, 1e-5)))[0]
    lum_faint = img_faint_ps[night_rows, night_cols, :3].sum(axis=2)
    unique_levels = len(np.unique(np.round(lum_faint, 8)))
    print(f"fp32 precision: unique levels in faint gradient = {unique_levels} (min={lum_faint.min():.2e}, max={lum_faint.max():.2e})")

    if unique_levels < 50:
        failures.append(f"faint planetshine gradient heavily quantized: only {unique_levels} unique levels")

    # --- compile-smoke: full raymarcher links with the shared caster_shadow include ---
    try:
        ctx.program(vertex_shader=load("atmosphere/atmo.vert"),
                    fragment_shader=load("atmosphere/atmo.frag"))
        print("compile-smoke atmo.vert+atmo.frag: OK")
    except Exception as exc:
        print(f"compile-smoke atmo.vert+atmo.frag: FAIL\n{exc}")
        return 2

    if failures:
        print("\nFAIL:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nPASS: Mode 3 sky-view LUT bakes caster eclipses (localized umbra, untouched controls)")
    print("     and multi-star illumination additively.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

