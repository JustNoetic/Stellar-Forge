"""Cached, position-queryable scattering for analytical terrain aerial perspective.

Tables store unit-light Rayleigh, Mie and multiple scattering to a common
atmosphere boundary. Finite segments use endpoint subtraction in GLSL. The
four coordinates (altitude, view cosine, sun cosine, relative azimuth) are
packed into a 3D texture; this is not a camera-frustum volume.
"""
from collections import OrderedDict

import moderngl

from engine.rendering.shader_loader import load_shader


class ScatteringLUTCache:
    def __init__(self, ctx, capacity=4, size=(32, 128, 32, 8), steps=96):
        self.ctx = ctx
        self.capacity = capacity
        self.size = size  # sun, view (two boundary families), altitude, azimuth
        self.steps = steps
        self.entries = OrderedDict()
        self.tau_program = ctx.compute_shader(load_shader("atmosphere/scattering_tau.comp"))
        self.scattering_program = ctx.compute_shader(load_shader("atmosphere/scattering_lut.comp"))

    def get(self, parameters, multiple_scattering):
        # The MS table changes whenever atmosphere properties or albedo change.
        # Keep the Python texture identity: a rebuilt MS table invalidates the
        # cache even if the driver recycles the previous OpenGL texture name.
        key = tuple(sorted(parameters.items())) + (("multiple_scattering", multiple_scattering),)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        while len(self.entries) >= self.capacity:
            _, textures = self.entries.popitem(last=False)
            for texture in textures:
                texture.release()
        sun, view, altitude, azimuth = self.size
        textures = []
        try:
            tau = self.ctx.texture((256, 128), 4, dtype="f4")
            textures.append(tau)
            for _ in range(3):
                textures.append(self.ctx.texture3d((sun * azimuth, view, altitude), 4, dtype="f4"))
            for texture in textures:
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                texture.repeat_x = texture.repeat_y = False
                if hasattr(texture, "repeat_z"):
                    texture.repeat_z = False
            for program in (self.tau_program, self.scattering_program):
                for name, value in parameters.items():
                    if name in program:
                        program[name].value = value
                program["u_scattering_steps"].value = self.steps
            tau.bind_to_image(0, read=False, write=True)
            self.tau_program.run(32, 16)
            self.ctx.memory_barrier()
            tau.use(20)
            multiple_scattering.use(3)
            self.scattering_program["u_scattering_tau_lut"].value = 20
            self.scattering_program["u_multi_scatter_lut"].value = 3
            self.scattering_program["u_scattering_azimuth_count"].value = azimuth
            for index, texture in enumerate(textures[1:]):
                texture.bind_to_image(index, read=False, write=True)
            self.scattering_program.run((sun * azimuth + 7) // 8, (view + 3) // 4, altitude)
            self.ctx.memory_barrier()
        except Exception:
            for texture in textures:
                texture.release()
            raise
        self.entries[key] = tuple(textures)
        return self.entries[key]

    @staticmethod
    def bind(textures):
        for index, texture in enumerate(textures):
            texture.use(20 + index)

    def release(self):
        for textures in self.entries.values():
            for texture in textures:
                texture.release()
        self.entries.clear()
        self.tau_program.release()
        self.scattering_program.release()
