"""
GPU validation for Method 2: Bounded Subtraction (Blackrack) atmosphere shadow.
Tests:
1. Verification of the analytical shadow cylinder intersection bounds (CPU reference).
2. Shader compilation and linking for atmo.vert + atmo.frag and sky_view_lut.vert + sky_view_lut.frag.
3. Verification of Method 2 uniform support (u_atmo_shadow_method, u_stochastic_noise, u_atmo_noise_type, u_ring_shadow_tex, u_ring_station_count).
4. GPU evaluation of Method 2 with deterministic midpoint quadrature (u_stochastic_noise = False).
5. GPU evaluation of Method 2 with stochastic ray steps (u_stochastic_noise = True) using IGN and STBN.
"""
import os
import sys
import numpy as np
import moderngl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.rendering.shaders import atmo_vertex_shader, atmo_fragment_shader
from engine.rendering.scattering_lut import ScatteringLUTCache
from scripts.test_scattering_segment import create_production_probe, PARAMETERS, PSI


def load_glsl(rel):
    import re
    inc_re = re.compile(r'^\s*#include\s+["<]([^">]+)[">]\s*$', re.MULTILINE)
    path = os.path.join(ROOT, "engine", "glsl", rel)

    def repl(m):
        inc = m.group(1)
        c1 = os.path.join(os.path.dirname(path), inc)
        c2 = os.path.join(ROOT, "engine", "glsl", inc)
        target = c1 if os.path.isfile(c1) else c2
        if not os.path.isfile(target):
            raise FileNotFoundError(f"include not found: {inc}")
        return load_glsl(os.path.relpath(target, os.path.join(ROOT, "engine", "glsl")).replace("\\", "/"))

    return inc_re.sub(repl, open(path, encoding="utf-8").read())


def test_cylinder_bounds_cpu():
    """Verify analytical cylinder intersection math matching get_eclipse_cylinder_bounds in GLSL."""
    def get_cylinder_bounds(ray_origin, ray_dir, caster_pos, cone_axis, cylinder_radius):
        origin_to_base = ray_origin - caster_pos
        axis_dot_dir = np.dot(cone_axis, ray_dir)
        axis_dot_orig = np.dot(cone_axis, origin_to_base)

        d_perp = ray_dir - axis_dot_dir * cone_axis
        o_perp = origin_to_base - axis_dot_orig * cone_axis

        a = np.dot(d_perp, d_perp)
        b = np.dot(o_perp, d_perp)
        c = np.dot(o_perp, o_perp) - cylinder_radius * cylinder_radius

        if a < 1e-8:
            if c <= 0.0:
                return -1e9, 1e9
            return -1.0, -1.0

        disc = b * b - a * c
        if disc < 0.0:
            return -1.0, -1.0

        sq = np.sqrt(disc)
        t1 = (-b - sq) / a
        t2 = (-b + sq) / a
        return min(t1, t2), max(t1, t2)

    # Ray passing through a cylinder of radius 1000 km centered at (0, 0, 50000) with axis -Z
    caster = np.array([0.0, 0.0, 50000.0])
    axis = np.array([0.0, 0.0, -1.0])
    radius = 1000.0

    # Ray going along X at Y=0, Z=0
    t1, t2 = get_cylinder_bounds(np.array([-5000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), caster, axis, radius)
    assert abs(t1 - 4000.0) < 1e-4, f"Expected entry at 4000 km, got {t1}"
    assert abs(t2 - 6000.0) < 1e-4, f"Expected exit at 6000 km, got {t2}"

    # Ray missing the cylinder (at Y = 2000 km)
    t1_miss, t2_miss = get_cylinder_bounds(np.array([-5000.0, 2000.0, 0.0]), np.array([1.0, 0.0, 0.0]), caster, axis, radius)
    assert t1_miss == -1.0 and t2_miss == -1.0, f"Expected miss (-1.0), got ({t1_miss}, {t2_miss})"

    print("Analytical cylinder bounds CPU reference math: OK")


def main():
    test_cylinder_bounds_cpu()

    try:
        ctx = moderngl.create_standalone_context(require=460)
    except Exception as exc:
        print(f"standalone 4.6 context unavailable ({exc}); falling back to hidden GLFW window")
        import glfw
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        win = glfw.create_window(64, 64, "offscreen", None, None)
        if not win:
            raise RuntimeError("glfw.create_window failed")
        glfw.make_context_current(win)
        ctx = moderngl.create_context(require=460)

    print(f"GL: {ctx.info['GL_VERSION']}")

    # 1. Test compilation of sky_view_lut
    try:
        prog_sky = ctx.program(
            vertex_shader=load_glsl("atmosphere/sky_view_lut.vert"),
            fragment_shader=load_glsl("atmosphere/sky_view_lut.frag")
        )
        print("Compile sky_view_lut.vert + sky_view_lut.frag: OK")
        assert 'u_atmo_shadow_method' in prog_sky, "u_atmo_shadow_method missing in sky_view_lut"
        print("  - u_atmo_shadow_method uniform present in sky_view_lut: OK")
    except Exception as exc:
        print(f"Compile sky_view_lut: FAIL\n{exc}")
        return 1

    # 2. Test compilation of atmo.vert + atmo.frag
    try:
        prog_atmo = ctx.program(
            vertex_shader=atmo_vertex_shader,
            fragment_shader=atmo_fragment_shader
        )
        print("Compile atmo.vert + atmo.frag: OK")
        assert 'u_atmo_shadow_method' in prog_atmo, "u_atmo_shadow_method missing in atmo.frag"
        assert 'u_stochastic_noise' in prog_atmo, "u_stochastic_noise missing in atmo.frag"
        assert 'u_atmo_noise_type' in prog_atmo, "u_atmo_noise_type missing in atmo.frag"
        assert 'u_ring_shadow_tex' in prog_atmo, "u_ring_shadow_tex missing in atmo.frag"
        assert 'u_ring_station_count' in prog_atmo, "u_ring_station_count missing in atmo.frag"
        print("  - All Method 2 uniforms present in atmo.frag: OK")
    except Exception as exc:
        print(f"Compile atmo: FAIL\n{exc}")
        return 1

    # 3. GPU execution test: verify Method 2 runs with deterministic and stochastic ray steps
    try:
        multiple = ctx.texture((64, 64), 4, np.tile(np.r_[PSI, 1.].astype('f4'), (64 * 64, 1)).tobytes(), dtype='f4')
        multiple.filter = (moderngl.LINEAR, moderngl.LINEAR)
        cache = ScatteringLUTCache(ctx)
        tables = cache.get(PARAMETERS, multiple)
        cache.bind(tables)
        multiple.use(3)

        probe = create_production_probe(ctx, tables, multiple, size=(16, 16))
        draw = probe['draw']
        program = probe['program']
        data = probe['data']
        au = probe['au']

        program['u_atmo_quality'].value = 3
        program['u_atmo_shadow_method'].value = 2  # Method 2 (Blackrack)
        program['u_ring_station_count'].value = 24

        a = (0, 6372.0, 0)
        b = (30, 6375.0, 0)

        # Baseline unshadowed
        data.view('i4')[28] = 0
        L_base, T_base = draw(a, b)
        assert np.all(np.isfinite(L_base)) and np.all(L_base >= 0)
        print("  - Baseline unshadowed draw: OK")

        # Configure an eclipse caster
        data.view('i4')[28] = 1
        data[36:40] = [15 / au, 6450 / au, 0, 30 / au]
        data[68:72] = [0, 1, 0, 1]
        data[100] = 30 / au

        # 3a. Deterministic midpoint quadrature (u_stochastic_noise = False)
        program['u_stochastic_noise'].value = False
        program['u_atmo_noise_type'].value = 0
        L_det, T_det = draw(a, b)
        assert np.all(np.isfinite(L_det)), "NaN/Inf in deterministic Method 2 output"
        assert np.linalg.norm(L_det) < 0.3 * np.linalg.norm(L_base), "Deterministic Method 2 failed to subtract eclipse deficit"
        print("  - Method 2 deterministic midpoint shadow subtraction: OK")

        # 3b. Stochastic ray steps with IGN (u_stochastic_noise = True, noise_type = 0)
        program['u_stochastic_noise'].value = True
        program['u_atmo_noise_type'].value = 0
        L_ign, T_ign = draw(a, b)
        assert np.all(np.isfinite(L_ign)), "NaN/Inf in stochastic IGN Method 2 output"
        assert np.linalg.norm(L_ign) < 0.3 * np.linalg.norm(L_base), "Stochastic IGN Method 2 failed to subtract eclipse deficit"
        print("  - Method 2 stochastic IGN shadow subtraction: OK")

        # 3c. Stochastic ray steps with STBN (u_stochastic_noise = True, noise_type = 1)
        program['u_stochastic_noise'].value = True
        program['u_atmo_noise_type'].value = 1
        L_stbn, T_stbn = draw(a, b)
        assert np.all(np.isfinite(L_stbn)), "NaN/Inf in stochastic STBN Method 2 output"
        assert np.linalg.norm(L_stbn) < 0.3 * np.linalg.norm(L_base), "Stochastic STBN Method 2 failed to subtract eclipse deficit"
        print("  - Method 2 stochastic STBN shadow subtraction: OK")

        # Clean up
        data.view('i4')[28] = 0
        cache.release()
        ctx.release()

    except Exception as exc:
        print(f"GPU execution test: FAIL\n{exc}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nAll Method 2 (Blackrack) stochastic & deterministic validations PASSED successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
