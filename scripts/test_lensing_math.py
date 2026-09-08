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
    #    rays LARGER. pole=+Y, cam_dir=+Z -> prograde periapsis dir = +X.
    #    cos_phi = dot(ray_perp, prograde): +X periapsis -> +1, -X -> -1.
    b_c_pos = 2.5980762 * rs_bh * (1.0 - 0.35 * 0.998)   # ~1.69 rs (prograde side)
    b_c_neg = 2.5980762 * rs_bh * (1.0 + 0.35 * 0.998)   # ~3.50 rs (retrograde side)
    for spin, sign, b_factor, expect_sh in (
            (+0.998, -1.0, 2.90, True),   # retrograde side, b < b_c_neg -> captured
            (+0.998, -1.0, 3.70, False),  # retrograde side, b > b_c_neg -> free
            (+0.998, +1.0, 2.30, False),  # prograde side,  b > b_c_pos -> free
            (+0.998, +1.0, 1.20, True),   # prograde side,  b < b_c_pos -> captured
            (-0.998, -1.0, 2.30, False),  # reversed spin mirrors the asymmetry
            (-0.998, +1.0, 2.30, True)):
        b_km = b_factor * rs_bh
        C = np.array([sign * b_km, 0.0, 1e9])
        a, sh = run_case(ctx, C, V, 0.0, rs_km=rs_bh, lens_type=3, spin=spin)
        side = 'retrograde' if sign < 0 else 'prograde '
        print(f"4. spin={spin:+.3f} {side} b={b_factor:.2f} rs: captured={sh} (expect {expect_sh})")
        if sh != expect_sh: fails.append(f"kerr spin={spin} {side} b={b_factor}")

    ctx.release()
    if fails:
        print("FAILED:", fails); sys_exit = 1
    else:
        print("All lensing math tests PASSED"); sys_exit = 0
    raise SystemExit(sys_exit)

if __name__ == "__main__":
    main()
