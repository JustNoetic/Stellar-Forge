"""Cached atmospheric transport for analytical sky and finite-distance haze.

Numerical quadrature is confined to compute-shader precomputation. Camera,
light directions, light intensities and ringshine-map changes do not rebake.
"""
from collections import OrderedDict
import moderngl
from engine.rendering.shader_loader import load_shader


class ScatteringLUTCache:
    def __init__(self, ctx, capacity=4, size=(48, 128, 32, 12), steps=96):
        self.ctx, self.capacity, self.size, self.steps = ctx, capacity, size, steps
        self.entries = OrderedDict()
        self.tau_program = ctx.compute_shader(load_shader("atmosphere/scattering_tau.comp"))
        self.scattering_program = ctx.compute_shader(load_shader("atmosphere/scattering_lut.comp"))

    @staticmethod
    def _release(entry):
        for texture in entry['transport']:
            texture.release()
        for tables in entry['solar'].values():
            for texture in tables[1:4]:
                texture.release()

    def get(self, parameters, multiple_scattering):
        radius = parameters['u_scattering_sun_radius']
        key = tuple(sorted((k, v) for k, v in parameters.items()
                           if k != 'u_scattering_sun_radius')) + (("multiple_scattering", multiple_scattering),)
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            if radius in entry['solar']:
                entry['solar'].move_to_end(radius)
                return entry['solar'][radius]
        else:
            while len(self.entries) >= self.capacity:
                _, evicted = self.entries.popitem(last=False)
                self._release(evicted)
        sun, view, altitude, azimuth = self.size
        created = []
        def texture(volume=False):
            result = (self.ctx.texture3d((sun * azimuth, view, altitude), 4, dtype='f4')
                      if volume else self.ctx.texture((256, 128), 4, dtype='f4'))
            created.append(result)
            result.filter = (moderngl.LINEAR, moderngl.LINEAR)
            result.repeat_x = result.repeat_y = False
            if hasattr(result, 'repeat_z'): result.repeat_z = False
            return result
        new_medium = entry is None
        try:
            if new_medium:
                # tau, directional shine R/M/MS, ambient R/M, ring MS.
                transport = (texture(), texture(True), texture(True), texture(True),
                             texture(), texture(), texture(True))
            else:
                transport = entry['transport']
            solar = tuple(texture(True) for _ in range(3))
            for program in (self.tau_program, self.scattering_program):
                for name, value in parameters.items():
                    if name in program: program[name].value = value
                program['u_scattering_steps'].value = self.steps
            if new_medium:
                for index, t in enumerate((transport[0], transport[4], transport[5])):
                    t.bind_to_image(index, read=False, write=True)
                self.tau_program.run(32, 16)
                self.ctx.memory_barrier()
            transport[0].use(20)
            multiple_scattering.use(3)
            program = self.scattering_program
            program['u_scattering_tau_lut'].value = 20
            program['u_multi_scatter_lut'].value = 3
            program['u_scattering_azimuth_count'].value = azimuth
            program['u_bake_transport'].value = new_medium
            for index, t in enumerate(solar + transport[1:4] + (transport[6],)):
                t.bind_to_image(index, read=False, write=True)
            program.run((sun * azimuth + 7) // 8, (view + 3) // 4, altitude)
            self.ctx.memory_barrier()
        except Exception:
            for t in created: t.release()
            raise
        if new_medium:
            entry = {'transport': transport, 'solar': OrderedDict()}
            self.entries[key] = entry
        # Four illuminated star slots are supported by the atmosphere pipeline.
        # Round radii on the CPU to avoid orbital motion growing the cache.
        while len(entry['solar']) >= 4:
            _, old = entry['solar'].popitem(last=False)
            for t in old[1:4]: t.release()
        tables = (transport[0],) + solar + transport[1:]
        entry['solar'][radius] = tables
        return tables

    @staticmethod
    def bind_star(tables, slot):
        first = 21 if slot == 0 else 30 + (slot - 1) * 3
        for i, t in enumerate(tables[1:4]): t.use(first + i)

    @staticmethod
    def bind(tables):
        tables[0].use(20)
        for slot in range(4): ScatteringLUTCache.bind_star(tables, slot)
        for unit, t in enumerate(tables[4:], 24): t.use(unit)

    @staticmethod
    def configure(program):
        units = {'u_scattering_tau_lut': 20,
                 'u_scattering_rayleigh_lut': (21, 30, 33, 36),
                 'u_scattering_mie_lut': (22, 31, 34, 37),
                 'u_scattering_multiple_lut': (23, 32, 35, 38),
                 'u_scattering_shine_rayleigh_lut': 24,
                 'u_scattering_shine_mie_lut': 25,
                 'u_scattering_shine_multiple_lut': 26,
                 'u_scattering_ambient_rayleigh_lut': 27,
                 'u_scattering_ambient_mie_lut': 28,
                 'u_scattering_ring_multiple_lut': 29}
        for name, unit in units.items():
            if name in program: program[name].value = unit

    def release(self):
        for entry in self.entries.values(): self._release(entry)
        self.entries.clear()
        self.tau_program.release()
        self.scattering_program.release()
