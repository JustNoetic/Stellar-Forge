"""GPU regressions for annular shadow bounds, budgets, and LUT routing.

Run with venv/Scripts/python.exe scripts/test_atmo_shadow_bounds.py.
Requires an OpenGL 4.6 context; no application window or saved settings change.
"""
import sys
from pathlib import Path

import moderngl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_atmo_shadows import (
    ROOT, CASES, Probe, bind_luts, create_luts, source,
)
from engine.rendering.shader_loader import load_shader
from engine.rendering.atmosphere_programs import AtmospherePrograms, specialize_atmosphere


def check_intervals(ctx, shader):
    helper = shader.split('bool ring_shadow_interval(', 1)[1]
    helper = 'bool ring_shadow_interval(' + helper.split('bool check_ray_intersects_shadow(', 1)[0]
    program = ctx.compute_shader('''#version 460
        layout(local_size_x=64) in;
        layout(std430, binding=0) readonly buffer Input { vec4 inputs[]; };
        layout(std430, binding=1) writeonly buffer Output { vec4 outputs[]; };
        uniform uint count;
        ''' + helper + '''
        void main() {
            uint i=gl_GlobalInvocationID.x;
            if (i>=count) return;
            vec4 a=inputs[i*3], b=inputs[i*3+1], interval=inputs[i*3+2];
            float lo=interval.x, hi=interval.y;
            bool hit=ring_shadow_interval(a.xyz,b.xyz,a.w,b.w,lo,hi);
            outputs[i]=vec4(lo,hi,hit ? 1. : 0.,0.);
        }''')
    # Analytic checks: empty center, one lobe, both lobes, outside, parallel.
    exact = [
        ([0, 0, 0, 5], [1, 0, 0, 10], [0, 4, 0, 0], None),
        ([0, 0, 0, 5], [1, 0, 0, 10], [0, 20, 0, 0], (5, 10)),
        ([-20, 0, 0, 5], [1, 0, 0, 10], [0, 40, 0, 0], (10, 30)),
        ([0, 12, 0, 5], [1, 0, 0, 10], [0, 20, 0, 0], None),
        ([7, 0, 0, 5], [0, 0, 0, 10], [0, 20, 0, 0], (0, 20)),
        ([2, 0, 0, 5], [0, 0, 0, 10], [0, 20, 0, 0], None),
    ]
    rng = np.random.default_rng(7)
    rows = [case[:3] for case in exact]
    for _ in range(1000):
        a = rng.uniform(-15000, 15000, 3)
        b = rng.uniform(-4, 4, 3)
        inner = rng.uniform(100, 10000)
        outer = inner + rng.uniform(1, 5000)
        rows.append(([*a, inner], [*b, outer], [0, 10000, 0, 0]))
    rows = np.asarray(rows, dtype='f4')
    inp = ctx.buffer(rows.tobytes())
    out = ctx.buffer(reserve=len(rows) * 16)
    inp.bind_to_storage_buffer(0)
    out.bind_to_storage_buffer(1)
    program['count'].value = len(rows)
    program.run((len(rows) + 63) // 64)
    ctx.memory_barrier()
    result = np.frombuffer(out.read(), dtype='f4').reshape(-1, 4)
    for value, case in zip(result, exact):
        expected = case[3]
        assert bool(value[2]) == (expected is not None), (case, value)
        if expected is not None:
            np.testing.assert_allclose(value[:2], expected, atol=1e-5)
    for row, value in zip(rows, result):
        distances = np.linspace(row[2, 0], row[2, 1], 2001)
        points = row[0, :3] + distances[:, None] * row[1, :3]
        radii = np.linalg.norm(points, axis=1)
        hits = distances[(radii >= row[0, 3]) & (radii <= row[1, 3])]
        if hits.size:
            assert value[2] == 1, (row, value)
            assert value[0] <= hits.min() + .05 and value[1] >= hits.max() - .05
    inp.release()
    out.release()
    program.release()
    print('Annulus bounds: 6 analytic and 1000 sampled rays passed')


def check_budgets(ctx, shader):
    marker = '    float march_s_closest ='
    instrumented = shader.replace(marker, '''
        out_scattered = vec4(float(steps), has_shadow ? 1.0 : 0.0,
                             use_bounded ? 1.0 : 0.0, 1.0);
        out_transmittance = vec4(1.0);
        return;
    float march_s_closest =''', 1)
    probe = Probe(ctx, instrumented, (8, 8))
    for budget in (4, 9, 32, 128):
        for adaptive in (False, True):
            probe.data.view('i4')[23] = budget
            probe.data.view('i4')[29] = int(adaptive)
            for _, camera, target, sun, inner, outer in CASES:
                probe.configure(camera, target, sun, inner, outer)
                probe.set('u_bounded_shadows', False)
                probe.draw()
                unlimited = probe.read()[0, :, :, 0]
                probe.set('u_bounded_shadows', True)
                probe.draw()
                bounded = probe.read()[0, :, :, 0]
                assert (bounded >= 1).all() and (bounded <= unlimited).all()
                if not adaptive:
                    assert (bounded <= budget).all()
    probe.release()
    print('Sample budgets: 4/9/32/128 steps, adaptive on/off passed')


def check_penumbra(ctx, shader):
    fragment = shader.split('void main() {', 1)[0] + '''
        uniform vec3 test_origin, test_direction;
        void main() {
            u_ring_mask = floatBitsToUint(instances[3].w);
            float lo, hi;
            bool hit = check_ray_intersects_shadow(0., 10000., test_direction,
                test_origin, vec3(0.), lo, hi);
            out_scattered = vec4(lo, hi, hit ? 1. : 0., 1.);
            out_transmittance = vec4(1.);
        }'''
    probe = Probe(ctx, fragment, (1, 1))
    probe.configure((9960, 100, 0), (9900, 160, 0), (1, -1, 0), 7500, 10000)
    direction = (-1 / np.sqrt(2), 1 / np.sqrt(2), 0)
    probe.set('test_origin', (9960, 100, 0))
    probe.set('test_direction', direction)
    probe.draw()
    # Projection stays 60 km beyond the outer edge, but the far stellar
    # footprint grows beyond 70 km. A midpoint-sized bound misses this fringe.
    assert probe.read()[0, 0, 0, 2] == 1
    probe.set('test_origin', (2000, 100, 0))
    probe.draw()
    assert probe.read()[0, 0, 0, 2] == 0  # Remains inside the hole.
    probe.set('test_origin', (9960, -100, 0))
    probe.set('test_direction', tuple(-x for x in direction))
    probe.draw()
    assert probe.read()[0, 0, 0, 2] == 0  # Ring plane is behind the light ray.
    probe.release()
    print('Projected ring bounds: growing penumbra, empty hole, and behind-light rejection passed')



def check_caster_parameters(ctx, shader):
    # A small distant caster cannot obscure these camera-facing parcels.
    # Its thickness is km, not AU, and is never an opaque-body radius.
    probe = Probe(ctx, specialize_atmosphere(shader, 2, False), (32, 32), steps=32)
    camera, target, sun = (0, 0, 8500), (0, 0, 6371), (1, 0, 0)
    probe.configure(camera, target, sun, ring=False)
    probe.draw()
    unshadowed = probe.read()
    assert np.max(unshadowed[0, :, :, :3]) > 0.0
    probe.data.view('i4')[28] = 1
    probe.data[36:40] = np.asarray([300000, 0, 0, 1000]) / 149597870.7
    probe.data[68:72] = [0, 1, 0, 1]
    probe.data[100] = 1000 / 149597870.7
    probe.data[108:112] = [10, 10, 10, 100]  # grazing tau + thickness km
    probe.data[140] = 0.02
    probe.data[256] = 8.5  # independent atmospheric scale height km
    probe.inst.view('u4')[12] = 1
    probe.configure(camera, target, sun, ring=False)
    probe.draw()
    np.testing.assert_allclose(probe.read(), unshadowed, atol=1e-6, rtol=0)
    probe.release()
    print('Mode 2 caster: km thickness does not inflate the opaque shadow radius')


def check_rendering(ctx, shader):
    # Compile with the actual vertex shader for both production output layouts.
    for fragment in (shader, shader.replace('layout(location = 0, index = 1)', 'layout(location = 1)')):
        program = ctx.program(vertex_shader=load_shader('atmosphere/atmo.vert'), fragment_shader=fragment)
        program.release()
    cache, tables, multiple, trans = create_luts(ctx)
    probe = Probe(ctx, shader, (32, 32), steps=9)
    bind_luts(probe, tables, multiple, trans)
    _, camera, target, sun, inner, outer = CASES[0]
    probe.configure(camera, target, sun, inner, outer)
    probe.draw()
    through_hole = probe.read()
    probe.configure(camera, target, sun, inner, outer, ring=False)
    probe.draw()
    np.testing.assert_allclose(through_hole, probe.read(), atol=1e-7)
    _, camera, target, sun, inner, outer = CASES[1]
    probe.configure(camera, target, sun, inner, outer)
    probe.set('u_bounded_shadows', False)
    probe.draw()
    unbounded = probe.read()
    probe.set('u_bounded_shadows', True)
    probe.draw()
    np.testing.assert_allclose(probe.read(), unbounded, atol=1e-6)
    # A real boundary crossing must remain close to a high-budget reference.
    _, camera, target, sun, inner, outer = CASES[2]
    probe.data.view('i4')[23] = 128
    probe.configure(camera, target, sun, inner, outer)
    probe.set('u_bounded_shadows', False)
    probe.draw()
    reference = probe.read()
    probe.set('u_bounded_shadows', True)
    probe.draw()
    bounded = probe.read()
    assert np.isfinite(bounded).all()
    for name, actual, expected in zip(('radiance', 'transmission'), bounded, reference):
        error = np.linalg.norm(actual[:, :, :3] - expected[:, :, :3]) / max(1e-8, np.linalg.norm(expected[:, :, :3]))
        print(f'Partial shadow {name} relative L2 error: {error:.4%}')
        assert error < .05, (name, error)
    assert ctx.error == 'GL_NO_ERROR'
    probe.release()
    cache.release()
    multiple.release()
    trans.release()
    print('Production shader compilation, ring-hole routing, and full-shadow parity passed')


def check_specializations(ctx, shader):
    cache, tables, multiple, trans = create_luts(ctx)
    expected = {}
    generic = Probe(ctx, shader, (16, 16), steps=9)
    bind_luts(generic, tables, multiple, trans)
    for quality, bounded in ((1, False), (2, False), (3, False), (3, True)):
        generic.set('u_atmo_quality', quality)
        generic.set('u_bounded_shadows', bounded)
        for i, (_, camera, target, sun, inner, outer) in enumerate(CASES):
            generic.configure(camera, target, sun, inner, outer)
            generic.draw()
            expected[quality, bounded, i] = generic.read()
    generic.release()
    for quality, bounded in ((1, False), (2, False), (3, False), (3, True)):
        probe = Probe(ctx, specialize_atmosphere(shader, quality, bounded), (16, 16), steps=9)
        bind_luts(probe, tables, multiple, trans)
        assert 'u_atmo_quality' not in probe.p and 'u_bounded_shadows' not in probe.p
        for i, (_, camera, target, sun, inner, outer) in enumerate(CASES):
            probe.configure(camera, target, sun, inner, outer)
            probe.draw()
            np.testing.assert_allclose(probe.read(), expected[quality, bounded, i], rtol=2e-4, atol=2e-6)
        probe.release()
    # Exercise the application's actual cache, sampler setup and VAO creation.
    programs = AtmospherePrograms(ctx, load_shader('atmosphere/atmo.vert'), shader)
    vertices = ctx.buffer(np.zeros((3, 6), 'f4').tobytes())
    indices = ctx.buffer(np.arange(3, dtype='u4').tobytes())
    for quality, bounded in ((1, False), (2, False), (3, False), (3, True), (2, True), (3, False)):
        pair = programs.programs(quality, bounded)
        vaos = programs.vaos(pair, vertices, indices)
        assert programs.programs(quality, bounded) is pair
        assert programs.vaos(pair, vertices, indices) is vaos
        assert 'u_atmo_quality' not in pair[0]
    assert len(programs._programs) == 4
    programs.release()
    vertices.release()
    indices.release()
    cache.release()
    multiple.release()
    trans.release()
    print('All four specialized modes match generic output; cached program/VAO switches passed')


def main():
    ctx = moderngl.create_standalone_context(require=460)
    shader = source(ROOT / 'engine/glsl/atmosphere/atmo.frag')
    print(ctx.info['GL_RENDERER'])
    check_intervals(ctx, shader)
    check_penumbra(ctx, shader)
    check_caster_parameters(ctx, shader)
    check_budgets(ctx, shader)
    check_rendering(ctx, shader)
    check_specializations(ctx, shader)
    ctx.release()


if __name__ == '__main__':
    main()
