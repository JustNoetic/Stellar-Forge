"""Cached atmosphere variants that exclude unused integration paths at compile time."""
from engine.rendering.ring_shadow_filter import RingShadowFilter
from engine.rendering.scattering_lut import ScatteringLUTCache


def specialize_atmosphere(source, quality, bounded):
    quality = max(1, min(int(quality), 3))
    bounded = quality == 3 and bool(bounded)
    version, rest = source.split('\n', 1)
    return (f'{version}\n#define ATMO_QUALITY {quality}\n'
            f'#define ATMO_BOUNDED_SHADOWS {int(bounded)}\n{rest}')


class AtmospherePrograms:
    """Compile lazily on mode changes; reuse programs and matching VAOs thereafter.

    A runtime branch alone leaves the large bounded/LUT paths in the same GPU
    program as High mode. Specialization lets the driver allocate resources for
    the selected integration mode. All dynamic scene uniforms are still updated
    by the normal render loop after selecting the active pair.
    """

    def __init__(self, ctx, vertex_source, fragment_source):
        self.ctx = ctx
        self.vertex_source = vertex_source
        self.fragment_source = fragment_source
        self._programs = {}
        self._vaos = {}

    def programs(self, quality, bounded):
        quality = max(1, min(int(quality), 3))
        key = (quality, quality == 3 and bool(bounded))
        if key not in self._programs:
            source = specialize_atmosphere(self.fragment_source, *key)
            lowres = source.replace(
                'layout(location = 0, index = 0)', 'layout(location = 0)'
            ).replace('layout(location = 0, index = 1)', 'layout(location = 1)')
            created = []
            try:
                for fragment in (source, lowres):
                    program = self.ctx.program(vertex_shader=self.vertex_source,
                                               fragment_shader=fragment)
                    created.append(program)
                    self._configure(program)
            except Exception:
                for program in created:
                    program.release()
                raise
            self._programs[key] = tuple(created)
        return self._programs[key]

    @staticmethod
    def _configure(program):
        samplers = {
            'u_ring_gradients': 0, 'u_transmittance_lut': 1,
            'u_multi_scatter_lut': 3, 'u_ringshine_map': 8,
            'u_depth_texture': 9, 'u_history_scatter': 10,
            'u_history_trans': 11, 'u_sky_view_lut': 12,
            'u_ring_shadow_tex': 13, 'u_sky_view_trans_lut': 14,
            'u_sky_view_star_lut': (15, 16, 17, 18), 'u_stbn_tex': 19,
        }
        for name, unit in samplers.items():
            if name in program:
                program[name].value = unit
        ScatteringLUTCache.configure(program)
        RingShadowFilter.configure(program)

    def vaos(self, programs, vertex_buffer, index_buffer):
        if programs not in self._vaos:
            created = []
            try:
                for program in programs:
                    created.append(self.ctx.vertex_array(program,
                        [(vertex_buffer, '3f 12x', 'in_position')],
                        index_buffer=index_buffer))
            except Exception:
                for vao in created:
                    vao.release()
                raise
            self._vaos[programs] = tuple(created)
        return self._vaos[programs]

    def release(self):
        for pair in self._vaos.values():
            for vao in pair:
                vao.release()
        for pair in self._programs.values():
            for program in pair:
                program.release()
        self._vaos.clear()
        self._programs.clear()
