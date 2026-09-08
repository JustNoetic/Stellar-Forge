"""GPU validation of the gravitational-lensing math in engine/glsl/common/refraction.glsl.

Runs compute_gravitational_deflection() (the exact shipped GLSL) in a compute
shader against known ground truths:
  1. Solar limb deflection of a background star  (~1.75 arcsec, Eddington 1919)
  2. Weak-field scaling  alpha ~ 2*rs/b
  3. Black-hole capture boundary at b_c = (3*sqrt(3)/2) * rs
  4. Kerr spin shifting the shadow radius b_c(phi)

Run:  python scripts/test_lensing_math.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import moderngl

from engine.rendering.shader_loader import load_shader

COMPUTE = """
#version 430 core
layout(local_size_x = 1) in;
#include "../engine/glsl/common/refraction.glsl"
layout(std430, binding=0) buffer Out { float outs[]; };
uniform vec3 t_C; uniform vec3 t_V; uniform float t_d;
void main() {
    bool is_sh = false;
    float a = compute_gravitational_deflection(t_C, t_V, t_d, is_sh);
    outs[0] = a;
    outs[1] = is_sh ? 1.0 : 0.0;
}
"""

def run_case(ctx, C, V, d, rs_km=1.0, lens_type=3, spin=0.0, strength=1.0):
    # Build a minimal standalone program: strip the #include path back to glsl base
    src = COMPUTE.replace('#include "../engine/glsl/common/refraction.glsl"',
                          load_shader('common/refraction.glsl').replace('#version 460 core', ''))
    prog = ctx.compute_shader(src)
    buf = ctx.buffer(reserve=16)
    buf.bind_to_storage_buffer(0)
    for name, val in (
        ('u_grav_lens_enabled', True), ('u_grav_lens_rs', float(rs_km)),
        ('u_grav_lens_type', int(lens_type)), ('u_grav_lens_spin', float(spin)),
        ('u_grav_lens_strength', float(strength)),
    ):
        if name in prog:
            prog[name].value = val
    if 'u_grav_lens_pole' in prog:
        prog['u_grav_lens_pole'].value = (0.0, 1.0, 0.0)
    prog['t_C'].value = tuple(float(x) for x in C)
    prog['t_V'].value = tuple(float(x) for x in V)
    prog['t_d'].value = float(d)
    prog.run(group_x=1)
    vals = np.frombuffer(buf.read(), dtype=np.float32)
    prog.release(); buf.release()
    return float(vals[0]), bool(vals[1] > 0.5)

def main():
    ctx = moderngl.create_standalone_context()
    fails = []

    # 1. Solar limb deflection: light grazing the solar limb from infinity,
    #    observer at Earth. Expected ~1.751 arcsec (GR half + Newtonian half).
    rs_sun = 2.95325008
    r_sun = 696340.0
    d_earth = 149597870.7
    V = np.array([0.0, 0.0, -1.0])
    C = np.array([r_sun, 0.0, d_earth])
    a, sh = run_case(ctx, C, V, d_earth * 1000.0, rs_km=rs_sun, lens_type=0)
    arcsec = a * 206264.806
    print(f"1. Solar limb deflection: {arcsec:.3f} arcsec (expected ~1.75)")
    if not (1.5 < arcsec < 2.0): fails.append("solar limb")

    # 2. Weak-field scaling: alpha = 2*rs/b + 2PN term 2.945243*(rs/b)^2.
    for b_factor in (10.0, 100.0, 1000.0):
        b_km = b_factor * rs_sun
        C = np.array([b_km, 0.0, 1e9])
        a, sh = run_case(ctx, C, V, 1e12, rs_km=rs_sun, lens_type=0)
        b_ratio = 1.0 / b_factor
        expect = 2.0 * b_ratio + 2.945243 * b_ratio * b_ratio
        print(f"2. b={b_factor:.0f} rs: alpha={a:.6g} (expect {expect:.6g})")
        if abs(a - expect) / expect > 1e-3: fails.append(f"weak field b={b_factor}")

    # 3. BH capture boundary: captured just inside b_c, free just outside.
    rs_bh = 2.95325008 * 10.0
    b_c = 2.5980762 * rs_bh
    for label, b_km, expect_sh in (("inside", b_c * 0.999, True), ("outside", b_c * 1.001, False)):
        C = np.array([b_km, 0.0, 1e9])
        a, sh = run_case(ctx, C, V, 0.0, rs_km=rs_bh, lens_type=3)
        print(f"3. shadow {label} (b={b_km:.1f} km, b_c={b_c:.1f}): captured={sh} (expect {expect_sh})")
        if sh != expect_sh: fails.append(f"shadow {label}")

    # 4. Kerr spin: prograde rays (with the spin) capture SMALLER b, retrograde
    #    rays LARGER. pole=+Y, cam_dir=+Z -> prograde periapsis dir = -X.
    #    (Physical photons travel source->camera (+Z here); r=+X gives
    #    L=rxv=-Y = retrograde. The old code used the backward-ray L and had
    #    this mirrored.) cos_phi = dot(ray_perp, prograde): -X -> +1, +X -> -1.
    b_c_pro = 2.5980762 * rs_bh * (1.0 - 0.35 * 0.998)   # ~1.69 rs (prograde, -X side)
    b_c_ret = 2.5980762 * rs_bh * (1.0 + 0.35 * 0.998)   # ~3.50 rs (retrograde, +X side)
    for spin, sign, b_factor, expect_sh in (
            (+0.998, +1.0, 2.90, True),   # retrograde side (+X), b < b_c_ret -> captured
            (+0.998, +1.0, 3.70, False),  # retrograde side (+X), b > b_c_ret -> free
            (+0.998, -1.0, 2.30, False),  # prograde side (-X),  b > b_c_pro -> free
            (+0.998, -1.0, 1.20, True),   # prograde side (-X),  b < b_c_pro -> captured
            (-0.998, +1.0, 2.30, False),  # reversed spin mirrors the asymmetry
            (-0.998, -1.0, 2.30, True)):
        b_km = b_factor * rs_bh
        C = np.array([sign * b_km, 0.0, 1e9])
        a, sh = run_case(ctx, C, V, 0.0, rs_km=rs_bh, lens_type=3, spin=spin)
        # NOTE: sign=+1 (+X periapsis) is RETROGRADE for spin>0 (see above).
        side = 'retrograde' if (sign > 0) == (spin > 0) else 'prograde '
        print(f"4. spin={spin:+.3f} {side} b={b_factor:.2f} rs: captured={sh} (expect {expect_sh})")
        if sh != expect_sh: fails.append(f"kerr spin={spin} {side} b={b_factor}")

    # 5. Background source deflection (apply_refraction / apply_gravitational_deflection):
    #    Verifies:
    #      a. Deflection is directed OUTWARD (away from lens center).
    #      b. Directly aligned star (beta ~ 0) deflects to Einstein ring theta_E.
    #      c. Distant parsec stars (100 pc) retain float32 precision and match theory.
    #      d. Large angles match Einstein weak-field 2*rs/b.
    AU_KM = 149597870.7
    STAR_CS = """
    #version 430 core
    layout(local_size_x = 1) in;
    #include "common/refraction.glsl"
    layout(std430, binding=0) buffer InB { vec4 in_pos[]; };
    layout(std430, binding=1) buffer OutB { vec4 out_pos[]; };
    uniform vec3 u_eye_pos;
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        vec3 p = in_pos[idx].xyz;
        vec3 rel = p - u_eye_pos;
        float d = length(rel);
        vec3 dir = d > 1e-9 ? (rel / d) : vec3(0.0, 0.0, -1.0);
        vec3 p_clamped = u_eye_pos + dir * min(d, 100000.0);
        g_grav_magnification = 1.0;
        vec3 lensed = apply_refraction(p_clamped, u_eye_pos);
        out_pos[idx] = vec4(lensed, g_grav_magnification);
    }
    """
    star_src = STAR_CS.replace('#include "common/refraction.glsl"',
                               load_shader('common/refraction.glsl').replace('#version 460 core', ''))
    prog_star = ctx.compute_shader(star_src)
    for name, val in [
        ('u_grav_lens_enabled', True), ('u_grav_lens_center', (0.0, 0.0, 0.0)),
        ('u_grav_lens_rs', float(rs_bh)), ('u_grav_lens_radius', float(rs_bh)),
        ('u_grav_lens_type', 3), ('u_grav_lens_strength', 1.0),
        ('u_grav_lens_spin', 0.0), ('u_grav_lens_pole', (0.0, 1.0, 0.0)),
        ('u_au_to_km', float(AU_KM)), ('u_refract_max_bend', 0.0),
        ('u_eye_pos', (0.0, 0.0, 1.0)),
    ]:
        if name in prog_star: prog_star[name].value = val

    theta_E = math.sqrt(2.0 * rs_bh / AU_KM)
    theta_E_as = math.degrees(theta_E) * 3600
    x_vals = [0.0, 1e-4, 1e-2, 0.1, 1.0, 100.0]
    in_stars = [[x * 206265.0, 0.0, -20626500.0, 1.0] for x in x_vals]
    in_arr = np.array(in_stars, dtype=np.float32)
    in_b = ctx.buffer(in_arr.tobytes()); out_b = ctx.buffer(reserve=in_arr.nbytes)
    in_b.bind_to_storage_buffer(0); out_b.bind_to_storage_buffer(1)
    prog_star.run(group_x=len(in_stars))
    res_stars = np.frombuffer(out_b.read(), dtype=np.float32).reshape((-1, 4))
    prog_star.release(); in_b.release(); out_b.release()

    for i, x in enumerate(x_vals):
        orig_p = in_stars[i][:3]; lens_p = res_stars[i][:3]
        orig_dir = np.array(orig_p) - np.array([0, 0, 1]); orig_dir /= np.linalg.norm(orig_dir)
        lens_dir = np.array(lens_p) - np.array([0, 0, 1]); lens_dir /= np.linalg.norm(lens_dir)
        orig_as = math.degrees(math.atan2(orig_dir[0], -orig_dir[2])) * 3600
        lens_as = math.degrees(math.atan2(lens_dir[0], -lens_dir[2])) * 3600
        beta = math.radians(orig_as / 3600)
        exp_as = math.degrees(0.5 * (beta + math.sqrt(beta * beta + 4.0 * theta_E * theta_E))) * 3600
        print(f"5. starfield deflection x={x:.4f} AU: orig={orig_as:.2f}\" -> lensed={lens_as:.2f}\" (expect {exp_as:.2f}\")")
        if lens_as < orig_as - 0.05:
            fails.append(f"inverted star deflection at x={x}")
        if abs(lens_as - exp_as) > 1.0:
            fails.append(f"star deflection error at x={x}")

    # 6. Secondary (mirror) lensing image: the second root of the lens equation
    #    theta_2 = (sqrt(beta^2 + 4*theta_E^2) - beta)/2 must appear on the
    #    OPPOSITE side of the lens with |theta_2| < theta_E, flux scaled by the
    #    point-source magnification mu_2 = (u^2+2)/(2u*sqrt(u^2+4)) - 0.5.
    #    Sources whose secondary ray is captured (|theta_2| < b_c/d_l) must be
    #    culled (magnification 0, true position returned).
    def run_star_pass(image):
        src = STAR_CS.replace('#include "common/refraction.glsl"',
                              load_shader('common/refraction.glsl').replace('#version 460 core', ''))
        prog = ctx.compute_shader(src)
        for name, val in [
            ('u_grav_lens_enabled', True), ('u_grav_lens_center', (0.0, 0.0, 0.0)),
            ('u_grav_lens_rs', float(rs_bh)), ('u_grav_lens_radius', float(rs_bh)),
            ('u_grav_lens_type', 3), ('u_grav_lens_strength', 1.0),
            ('u_grav_lens_spin', 0.0), ('u_grav_lens_pole', (0.0, 1.0, 0.0)),
            ('u_au_to_km', float(AU_KM)), ('u_refract_max_bend', 0.0),
            ('u_eye_pos', (0.0, 0.0, 1.0)), ('u_grav_image', int(image)),
        ]:
            if name in prog: prog[name].value = val
        arr = np.array(in_stars, dtype=np.float32)
        b_in = ctx.buffer(arr.tobytes()); b_out = ctx.buffer(reserve=arr.nbytes)
        b_in.bind_to_storage_buffer(0); b_out.bind_to_storage_buffer(1)
        prog.run(group_x=len(in_stars))
        out = np.frombuffer(b_out.read(), dtype=np.float32).reshape((-1, 4)).copy()
        prog.release(); b_in.release(); b_out.release()
        return out

    sec_res = run_star_pass(1)
    for i, x in enumerate(x_vals):
        orig_p = in_stars[i][:3]
        orig_dir = np.array(orig_p) - np.array([0, 0, 1]); orig_dir /= np.linalg.norm(orig_dir)
        orig_as = math.degrees(math.atan2(orig_dir[0], -orig_dir[2])) * 3600
        beta = math.radians(orig_as / 3600)
        u = max(beta / theta_E, 1e-8)
        A = (u * u + 2.0) / (2.0 * u * math.sqrt(u * u + 4.0))
        exp_mu = max(min(A - 0.5, 10.0), 0.0)
        exp2_as = math.degrees(0.5 * (math.sqrt(beta * beta + 4.0 * theta_E * theta_E) - beta)) * 3600
        sec_dir = np.array(sec_res[i][:3]) - np.array([0, 0, 1]); sec_dir /= np.linalg.norm(sec_dir)
        sec_as = math.degrees(math.atan2(sec_dir[0], -sec_dir[2])) * 3600
        got_mu = float(sec_res[i][3])
        captured = exp2_as < math.degrees(2.5980762 * rs_bh / AU_KM) * 3600  # |theta_2| < b_c/d_l
        if captured:
            ok = got_mu == 0.0 and abs(sec_as - orig_as) < 1.0  # true position, culled
        else:
            ok = sec_as < 0.0 and abs(abs(sec_as) - exp2_as) < max(0.1, exp2_as * 1e-2) \
                 and abs(got_mu - exp_mu) <= max(1e-3, exp_mu * 5e-2)
        tag = 'captured' if captured else 'mirror image'
        print(f"6. secondary image x={x}: lensed={sec_as:.4f}\" mag={got_mu:.5g} "
              f"(expect {exp2_as:.4f}\" mu_2={exp_mu:.5g}, {tag})")
        if not ok:
            fails.append(f"secondary image x={x}")

    ctx.release()
    if fails:
        print("FAILED:", fails); sys_exit = 1
    else:
        print("All lensing math tests PASSED"); sys_exit = 0
    raise SystemExit(sys_exit)

if __name__ == "__main__":
    main()
