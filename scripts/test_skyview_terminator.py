"""
GPU regression: Mode 3 (Sky-View LUT) planetary terminator must respond to the
star's angular size and to the refraction-extended effective star radius
(sin_star + max_bend), matching the Mode 1/2 twilight model
(common/sun_terminator.glsl).

Geometry: space observer at quarter phase — camera at (-CAM_DIST_KM, 0, 0), sun
direction (0, 0, 1), 90 deg off the view axis. In the bake's space-observer
branch the basis puts x_basis = L_sun, so a limb parcel at LUT azimuth u = phi/2pi
has sun elevation arcsin(cos_alpha * cos(phi)), dropping monotonically from +71 deg
(u = 0) to -71 deg (u = 0.5). The penumbral crossing (elevation = parcel horizon
dip, 5-10 deg over limb rows) sits at u ~ 0.265-0.28; the u-width of the penumbra
is ~ sin_planet * eff_star_rad / (2*pi*sin(phi)) texels-of-u per unit. A solar-size
disc (0.00465 rad) is sub-texel, so the test drives the physics with large discs:
  A: sin_star = 0.00465 (Sun),        max_bend = 0     -> near-zero terminator signal
  B: sin_star = 0.25     (14.3 deg),  max_bend = 0     -> ~13-texel penumbra band
  C: sin_star = 0.00465, max_bend = 0.10 (5.7 deg bend)-> ~6-texel penumbra band
Post-fix, B and C must radiate a bright partial-disc twilight arc at the crossing
u-window while the day (u ~ 0) and night (u ~ 0.5) limb stay unchanged.

Pre-fix behavior: the bake used a hard-coded smoothstep(-cos_planet-0.02,
-cos_planet+0.02) band, identical for every configuration -> all ratios exactly
1.000 and the absolute signals stay at the smear-band floor -> test FAILS.

Also compile-smoke links atmo.vert + atmo.frag (exercises the shared
common/sun_terminator.glsl include inside the full raymarcher).
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


# --- atmosphere configuration (Earth-like) -----------------------------------
R_PLANET_KM = 6371.0
R_ATMO_KM = 6471.0
AU_TO_KM = 149597870.7
H_R, H_M = 8.5, 1.2
BETA_R = (5.8e-6, 1.35e-2, 4.5e-2)   # per-meter, shader scales by 1000 -> per-km
BETA_M = (2.1e-6, 2.1e-6, 2.1e-6)
BETA_ABS_MIXED = (1.0e-7, 1.0e-7, 1.0e-7)
BETA_ABS_LAYERED = (5.0e-7, 5.0e-7, 5.0e-7)
O3_PEAK, O3_WIDTH = 25.0, 8.0
MIE_G = 0.758
CAM_DIST_KM = 20000.0                 # space observer, quarter phase
SUN_DIR = (0.0, 0.0, 1.0)            # 90 deg off the camera view axis


def pack_atmo(sin_star, max_bend):
    """Pack the AtmoData SSBO exactly like app.py's 256-float atmo_staging layout."""
    f = np.zeros(256, dtype=np.float32)
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
    f[28] = 0.0                            # u_num_active_casters
    f[29] = 0.0                            # u_atmo_adaptive_steps
    f[30] = 0.0                            # u_body_idx
    f[31] = 0.0                            # u_frame_counter
    f[32:35] = (1.0, 1.0, 1.0)             # u_mie_albedo
    f[35] = 0.00029                        # u_refractivity
    f[180] = O3_PEAK                       # u_ozone_peak_km
    f[181] = O3_WIDTH                      # u_ozone_width_km
    f[182] = R_PLANET_KM - 0.1             # u_planet_clip_km
    f[183] = 128.0                         # u_max_adaptive_steps
    f[184:188] = (1.0 / H_R, 1.0 / H_M, 1.0 / O3_WIDTH, max_bend)  # u_precomp_opt
    g = MIE_G
    f[188:192] = ((3.0 / (8.0 * np.pi)) * ((1.0 - g * g) / (2.0 + g * g)),
                  1.0 + g * g, 2.0 * g, 0.0)                       # u_precomp_mie
    eff = sin_star + max_bend
    cos_eff = float(np.sqrt(max(0.0, 1.0 - eff * eff)))
    # star 0: color_irrad (rgb=1, w=sin_star), dir_sph_eff (sun dir, w=cos_sun_eff),
    # pos_local (sun pos km, w=effective_star_rad), solstice
    f[192:196] = (1.0, 1.0, 1.0, sin_star)
    f[208:212] = (*SUN_DIR, cos_eff)
    f[224:228] = (SUN_DIR[0] * AU_TO_KM, SUN_DIR[1] * AU_TO_KM, SUN_DIR[2] * AU_TO_KM, eff)
    f[240:244] = (0.0, SUN_DIR[1], 1.0, 0.00465)
    return f.tobytes()


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


def window_sum(img, rows, cols):
    s = 0.0
    for c0, c1 in cols:
        s += float(img[rows[0]:rows[1], c0:c1, :3].sum())
    return s


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

    # Scene UBO (binding 1): the bake now reads u_num_stars / u_stars_poles_obl
    # from SceneData (same std140 layout as atmo.frag / app.py's scene_ubo).
    scene = np.zeros(6304 // 4, dtype=np.float32)
    scene.view(np.int32)[128 // 4] = 1          # u_num_stars
    scene[656 // 4 : 656 // 4 + 4] = (0.0, 1.0, 0.0, 0.0)  # u_stars_poles_obl[0]
    scene_ubo = ctx.buffer(scene.tobytes())
    scene_ubo.bind_to_uniform_block(1)

    # Instance SSBO (binding 2): the bake reads planetshine dir/color + ring
    # mask from instances[u_body_idx * 7 + 3..5]; zeros disable all shines.
    inst_ssbo = ctx.buffer(np.zeros(7 * 4, dtype=np.float32).tobytes())
    inst_ssbo.bind_to_storage_buffer(binding=2)
    sky_tex_a = ctx.texture((256, 256), 4, dtype="f2")
    sky_tex_b = ctx.texture((256, 256), 4, dtype="f2")
    sky_fbo = ctx.framebuffer(color_attachments=[sky_tex_a, sky_tex_b])
    vao_sky = ctx.vertex_array(prog_sky, [(quad_vbo, "2f", "in_position")])
    ssbo = ctx.buffer(reserve=1024)
    ssbo.bind_to_storage_buffer(binding=8)

    def bake_sky_view(sin_star, max_bend):
        ssbo.write(pack_atmo(sin_star, max_bend))
        trans_tex.use(location=1)
        ms_tex.use(location=3)
        prog_sky["u_cam_pos"].value = (-CAM_DIST_KM, 0.0, 0.0)
        prog_sky["u_sun_dir"].value = SUN_DIR
        sky_fbo.use()
        ctx.viewport = (0, 0, 256, 256)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        vao_sky.render(moderngl.TRIANGLE_STRIP)
        raw = sky_fbo.color_attachments[0].read()
        return np.frombuffer(raw, dtype="f2").reshape(256, 256, 4).astype(np.float32)

    configs = {
        "A_sun": dict(sin_star=0.00465, max_bend=0.0),
        "B_bigstar": dict(sin_star=0.25, max_bend=0.0),
        "C_refraction": dict(sin_star=0.00465, max_bend=0.10),
    }
    imgs = {name: bake_sky_view(**cfg) for name, cfg in configs.items()}

    # Measurement windows (space-observer limb branch, v > 0.5): the penumbral
    # crossing sits at u ~ 0.265-0.28 (v-dependent parcel dip), so
    #   twilight window: u in [0.25, 0.30] + mirror [0.70, 0.75] — contains the
    #     13-texel (B) / 6-texel (C) penumbra bands, and (nearly) nothing for A.
    #   day control:   u < 0.05 or u > 0.95 (sun elevation 63-71 deg, disc fully up)
    #   night control: u in 0.45..0.55 (elevation -19..-71 deg, disc fully set)
    LIMB_ROWS = (230, 256)                       # v in [0.898, 1.0)
    TW_COLS = [(64, 77), (179, 192)]
    DAY_COLS = [(0, 13), (243, 256)]
    NIGHT_COLS = [(115, 141)]

    def windows(img):
        return dict(
            tw=window_sum(img, LIMB_ROWS, TW_COLS),
            day=window_sum(img, LIMB_ROWS, DAY_COLS),
            night=window_sum(img, LIMB_ROWS, NIGHT_COLS),
        )

    # Orientation check: GL texture row 0 is v=0 (bottom). The subsolar (day)
    # limb must be the brightest window for config A; otherwise the readback is
    # v-flipped relative to our assumption.
    w = {name: windows(img) for name, img in imgs.items()}
    if w["A_sun"]["day"] < w["A_sun"]["tw"]:
        imgs = {name: img[::-1] for name, img in imgs.items()}
        w = {name: windows(img) for name, img in imgs.items()}
        print("readback orientation: flipped so v=0 is first row")
    else:
        print("readback orientation: v=0 is first row")

    for name in configs:
        print(f"{name:14s} eff_star={configs[name]['sin_star'] + configs[name]['max_bend']:.5f}  "
              f"terminator={w[name]['tw']:.4f}  day={w[name]['day']:.4f}  night={w[name]['night']:.4f}")

    tw_a, day_a, night_a = w["A_sun"]["tw"], w["A_sun"]["day"], w["A_sun"]["night"]
    tw_b, day_b, night_b = w["B_bigstar"]["tw"], w["B_bigstar"]["day"], w["B_bigstar"]["night"]
    tw_c, _, night_c = w["C_refraction"]["tw"], w["C_refraction"]["day"], w["C_refraction"]["night"]

    r_big = tw_b / max(tw_a, 1e-6)
    r_refr = tw_c / max(tw_a, 1e-6)
    r_day = day_b / max(day_a, 1e-12)
    sig_big = tw_b - tw_a          # direct-disc penumbra light added over A's floor
    sig_refr = tw_c - tw_a         # refraction-tail light added over A's floor
    print(f"ratios vs A: big_star={r_big:.3f}  refraction={r_refr:.3f}  day_control={r_day:.3f}  "
          f"sig_big={sig_big:.4f}  sig_refr={sig_refr:.4f}")

    # --- compile-smoke: full raymarcher links with the shared terminator include ---
    try:
        ctx.program(vertex_shader=load("atmosphere/atmo.vert"),
                    fragment_shader=load("atmosphere/atmo.frag"))
        print("compile-smoke atmo.vert+atmo.frag: OK")
    except Exception as exc:
        print(f"compile-smoke atmo.vert+atmo.frag: FAIL\n{exc}")
        return 2

    failures = []
    if abs(r_big - 1.0) < 0.02 and abs(r_refr - 1.0) < 0.02:
        failures.append(
            "terminator does not respond to star angular size / refraction "
            f"(big_star ratio {r_big:.3f}, refraction ratio {r_refr:.3f} both ~1.0) "
            "- fixed-width smoothstep still in effect")
    else:
        # Post-fix: the widened penumbra must radiate absolute light over A's
        # multi-scatter floor. The refraction tail is intrinsically dim (bent light
        # grazes the dense low atmosphere), so its bar is lower than the big-star's.
        if r_big <= 1.5:
            failures.append(f"big-star terminator ratio {r_big:.3f} <= 1.5 (expected wide bright penumbra)")
        if r_refr <= 1.02:
            failures.append(f"refraction-extended terminator ratio {r_refr:.3f} <= 1.02")
        if sig_big < 1.0:
            failures.append(f"big-star penumbra signal {sig_big:.4f} < 1.0 - band not radiating")
        if sig_refr < 0.05:
            failures.append(f"refraction penumbra signal {sig_refr:.4f} < 0.05 - band not radiating")
    if not (0.90 <= r_day <= 1.10):
        failures.append(f"day-limb control ratio {r_day:.3f} outside [0.90, 1.10] - "
                        "terminator change leaked into the fully lit limb")
    if night_b > night_a * 1.10 + 1e-3 or night_c > night_a * 1.10 + 1e-3:
        failures.append(f"night-limb control drifted {night_a:.4f} -> B:{night_b:.4f} C:{night_c:.4f}")


    if failures:
        print("\nFAIL:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nPASS: Mode 3 terminator scales with stellar angular size and refraction max_bend,")
    print("     day/night limb controls unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
