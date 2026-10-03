"""Offscreen production-shader benchmark for Mode 3 bounded ring shadows.

Run with --steps 9 to reproduce a low-budget, non-adaptive fallback. Timings
are atmosphere-pass GPU milliseconds, not application FPS. LUT baking,
readback, and driver warmup are excluded. --baseline accepts a saved atmo.frag
snapshot (its includes are resolved against the current shader directory).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import moderngl
from engine.rendering.shader_loader import load_shader, _resolve_includes
from engine.rendering.scattering_lut import ScatteringLUTCache
from engine.rendering.atmosphere_programs import specialize_atmosphere

ROOT = Path(__file__).resolve().parents[1]
AU = 149597870.7
PARAMETERS = dict(
    u_planet_radius_km=6371.0, u_atmo_radius_km=6471.0,
    u_scattering_bottom_km=6371.0, u_scattering_sun_radius=0.005,
    u_h_rayleigh=8.0, u_h_mie=1.2,
    u_beta_rayleigh=(5.8e-6, 13.5e-6, 33.1e-6), u_beta_mie=(2e-6,) * 3,
    u_mie_albedo=(0.9, 0.95, 1.0), u_beta_abs_mixed=(0.0,) * 3,
    u_beta_abs_layered=(.6e-6, 1.4e-6, .1e-6),
    u_ozone_peak_km=25.0, u_ozone_width_km=8.0, u_mie_g=0.76,
)


class Probe:
    """Production fragment shader on a fullscreen triangle with fixed scene data."""

    def __init__(self, ctx, source, size=(512, 512), steps=32):
        self.ctx = ctx
        # Read back the two production blend terms as separate attachments.
        source = source.replace('layout(location = 0, index = 1)', 'layout(location = 1)')
        self.p = ctx.program(vertex_shader='''#version 460
            out vec3 f_local_pos;
            out float f_clip_z;
            void main() {
                vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                gl_Position = vec4(p * 2. - 1., 0., 1.);
                f_local_pos = vec3(0, 0, 1);
                f_clip_z = 1.;
            }''', fragment_shader=source)
        self.vao = ctx.vertex_array(self.p, [])
        self.targets = [ctx.texture(size, 4, dtype='f4') for _ in range(2)]
        self.fbo = ctx.framebuffer(self.targets)
        self.resources = []
        # Complete dummy textures for inactive shader branches.
        for unit, (typ, dims) in enumerate((('texture', (4, 4)), ('texture3d', (4, 4, 4)), ('texture_array', (4, 4, 16)))):
            tex = getattr(ctx, typ)(dims, 4, np.ones((*dims, 4), 'f4').tobytes(), dtype='f4')
            tex.use(unit)
            self.resources.append(tex)
        import re
        for kind, name, count in re.findall('uniform sampler(2DArray|2D|3D) (\\w+)(?:\\[(\\d+)\\])?;', source):
            if name in self.p:
                unit = {'2D': 0, '3D': 1, '2DArray': 2}[kind]
                self.p[name].value = (unit,) * int(count) if count else unit
        self.set('u_screen_res', size)
        self.set('u_atmo_quality', 3)
        self.set('u_scattering_enabled', True)
        self.set('u_bounded_shadows', True)
        self.set('u_scattering_bottom_km', 6371.0)
        self.set('u_scattering_terrain_bottom_km', 6371.0)
        self.set('u_scattering_sun_radius', 0.005)
        self.set('u_scattering_azimuth_count', 12)
        self.set('u_atmo_clip_mode', 2)
        # SceneData std140 offsets must match atmo.frag and app.py.
        self.scene = np.zeros(4096, 'f4')
        self.scene[:16] = np.eye(4, dtype='f4').ravel()
        self.scene[16:32] = np.eye(4, dtype='f4').ravel()
        self.scene.view('i4')[32] = 1
        self.scene[292:294] = [100.0, 1.0]
        self.ubo = ctx.buffer(self.scene.tobytes())
        self.ubo.bind_to_uniform_block(1)
        # AtmoData std430 offsets mirror app.py's atmo_staging upload.
        self.data = np.zeros(256, 'f4')
        d = self.data
        d[3] = 6471 / AU
        d[4:7] = PARAMETERS['u_beta_rayleigh']
        d[7] = 8.0
        d[8:11] = 2e-06
        d[11] = 1.2
        d[15] = 0.76
        d[16:19] = PARAMETERS['u_beta_abs_layered']
        d[19:23] = [1.0, 6371.0, 6471.0, AU]
        d.view('i4')[23] = steps
        d[24:28] = [0, 1, 0, 1]
        d[32:35] = [0.9, 0.95, 1.0]
        d[180:184] = [25, 8, 6371, 128]
        d[184:188] = [1 / 8, 1 / 1.2, 1 / 8, 0]
        g = 0.76
        d[188:192] = [3 / (8 * np.pi) * (1 - g * g) / (2 + g * g), 1 + g * g, 2 * g, 0]
        d[192:196] = [1, 1, 1, 0.005]
        self.ssbo = ctx.buffer(d.tobytes())
        self.ssbo.bind_to_storage_buffer(8)
        self.inst = np.zeros(28, 'f4')
        self.instance = ctx.buffer(self.inst.tobytes())
        self.instance.bind_to_storage_buffer(2)
        self.set('u_inv_proj', np.diag([0.3, 0.3, 1, 1]).astype('f4'))
        from engine.rendering.ring_shadow_filter import RingShadowFilter
        atlas = np.ones((16, 4096, 4), 'f4')
        atlas[:, :, 3] = 0.3
        self.atlas = ctx.texture((4096, 16), 4, atlas.tobytes(), dtype='f4')
        self.atlas.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.atlas.use(4)
        self.set('u_ring_gradients', 4)
        self.ring_filter = RingShadowFilter(ctx)
        self.ring_filter.update(atlas)
        self.ring_filter.configure(self.p)

    def set(self, name, value):
        if name in self.p:
            if isinstance(value, np.ndarray):
                self.p[name].write(value.astype('f4').tobytes())
            else:
                self.p[name].value = value

    def configure(self, camera, target, sun, inner=7500.0, outer=12000.0, ring=True):
        camera = np.asarray(camera, float)
        forward = np.asarray(target, float) - camera
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        inv = np.eye(4)
        inv[:3, :3] = np.column_stack((right, up, -forward))
        self.set('u_inv_view', inv.T)
        self.set('u_camera_pos', tuple(camera / AU))
        sun = np.asarray(sun, float)
        sun /= np.linalg.norm(sun)
        self.scene[36:40] = [*sun, 0.005]
        self.scene[100:104] = [1, 1, 1, 1]
        self.scene[164:168] = [0, 1, 0, 0.005]
        self.ubo.write(self.scene.tobytes())
        self.ubo.bind_to_uniform_block(1)
        self.data[208:212] = [*sun, np.sqrt(1 - 0.005 ** 2)]
        self.data[224:228] = [*sun * AU, 0.005]
        self.data[240:244] = [0, sun[1], 1, 0.005]
        self.ssbo.write(self.data.tobytes())
        self.ssbo.bind_to_storage_buffer(8)
        self.inst.view('u4')[15] = 1 if ring else 0
        self.instance.write(self.inst.tobytes())
        self.instance.bind_to_storage_buffer(2)
        self.set('u_num_ring_planes', 1 if ring else 0)
        for name, arr in [('u_ring_center', np.zeros((16, 3), 'f4')), ('u_ring_normal', np.tile([0, 1, 0], (16, 1)).astype('f4')), ('u_ring_params', np.tile([inner / AU, outer / AU, 1, 0], (16, 1)).astype('f4'))]:
            self.set(name, arr)
        if 'u_ring_coplanar_mask' in self.p:
            self.p['u_ring_coplanar_mask'].write(np.ones(16, 'u4').tobytes())
        self.fbo.use()
        self.ctx.viewport = (0, 0, *self.fbo.size)
        self.fbo.clear()

    def draw(self):
        self.vao.render(vertices=3)

    def read(self):
        return np.stack([np.frombuffer(t.read(), dtype='f4').reshape(*self.fbo.size[::-1], 4) for t in self.targets])

    def release(self):
        self.vao.release()
        self.fbo.release()
        self.ring_filter.release()
        for resource in self.resources + self.targets + [self.atlas, self.ubo, self.ssbo, self.instance, self.p]:
            resource.release()

def source(path):
    return _resolve_includes(Path(path).read_text(), str(ROOT / 'engine/glsl/atmosphere'))


def create_luts(ctx):
    """Bake production transport/transmittance; use a fixed multiple-scatter field."""
    multiple = ctx.texture((64, 64), 4,
        np.tile([.01, .01, .01, 1.], (4096, 1)).astype('f4').tobytes(), dtype='f4')
    cache = ScatteringLUTCache(ctx)
    tables = cache.get(PARAMETERS, multiple)
    program = ctx.program(vertex_shader='''#version 460
        out vec2 f_uv;
        void main() {
            f_uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            gl_Position = vec4(f_uv * 2. - 1., 0, 1);
        }''', fragment_shader=load_shader('atmosphere/atmo_lut.frag'))
    for key, value in PARAMETERS.items():
        if key in program:
            program[key].value = value
    trans = ctx.texture((256, 128), 4, dtype='f4')
    trans.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trans.repeat_x = trans.repeat_y = False
    fbo = ctx.framebuffer([trans])
    fbo.use()
    vao = ctx.vertex_array(program, [])
    vao.render(vertices=3)
    vao.release()
    fbo.release()
    program.release()
    return cache, tables, multiple, trans


def bind_luts(probe, tables, multiple, trans):
    ScatteringLUTCache.configure(probe.p)
    ScatteringLUTCache.bind(tables)
    multiple.use(3)
    probe.set('u_multi_scatter_lut', 3)
    trans.use(5)
    probe.set('u_transmittance_lut', 5)


# Planet radius is 6371 km; the viewed patch starts at about 8 km altitude.
# The hole case projects entirely inside the inner ring edge. The partial
# case crosses the inner edge while still inside the atmospheric envelope.
CASES = [
    ('ring hole', (6300, 1000, 0), (6450, 1100, 0), (.7, -.7, 0), 7500., 12000.),
    ('broad shadow', (6300, 1000, 0), (6450, 1100, 0), (.98, -.2, 0), 7500., 12000.),
    ('partial shadow', (6300, 1000, 0), (6450, 1100, 0), (.98, -.2, 0), 11500., 12000.),
]


def main():
    import argparse
    import statistics
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=9)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--quality', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--generic', action='store_true',
                        help='Use the old shared runtime-branch program for comparison')
    args = parser.parse_args()
    if min(args.width, args.height, args.samples) <= 0 or not 4 <= args.steps <= 128:
        parser.error('Positive dimensions/samples and 4-128 steps are required')
    ctx = moderngl.create_standalone_context(require=460)
    print(ctx.info['GL_RENDERER'], flush=True)
    print(f'{args.width}x{args.height}, mode {args.quality}, {args.steps} steps, adaptive off, median GPU ms')
    cache, tables, multiple, trans = create_luts(ctx)
    paths = [ROOT / 'engine/glsl/atmosphere/atmo.frag']
    if args.baseline:
        paths.insert(0, args.baseline)
    for path in paths:
        print(path, flush=True)
        for bounded in ((False, True) if args.quality == 3 else (False,)):
            fragment = source(path)
            if not args.generic:
                fragment = specialize_atmosphere(fragment, args.quality, bounded)
            probe = Probe(ctx, fragment, (args.width, args.height), args.steps)
            bind_luts(probe, tables, multiple, trans)
            probe.set('u_atmo_quality', args.quality)
            for name, camera, target, sun, inner, outer in CASES:
                probe.configure(camera, target, sun, inner, outer)
                probe.set('u_bounded_shadows', bounded)
                # NVIDIA may specialize a newly compiled shader asynchronously.
                start = time.perf_counter()
                while time.perf_counter() - start < 1.0:
                    probe.draw()
                    ctx.finish()
                times = []
                for _ in range(args.samples):
                    with ctx.query(time=True) as query:
                        probe.draw()
                    times.append(query.elapsed / 1e6)
                image = probe.read()
                assert np.isfinite(image).all(), 'Non-finite scattering/transmission'
                assert ctx.error == 'GL_NO_ERROR'
                print(f'  {name}, bounded={bounded}: {statistics.median(times):.3f}', flush=True)
            probe.release()
    cache.release()
    multiple.release()
    trans.release()
    ctx.release()


if __name__ == '__main__':
    main()
