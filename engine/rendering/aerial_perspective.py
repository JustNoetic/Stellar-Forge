"""Small camera-frustum volume; endpoint tables supply all light sources."""
import moderngl

from engine.rendering.scattering_lut import ScatteringLUTCache
from engine.rendering.shader_loader import load_shader


class AerialPerspectiveVolume:
    def __init__(self, ctx, size=(32, 32, 32)):
        self.ctx, self.size = ctx, size
        self.key = None
        self.program = ctx.compute_shader(load_shader('atmosphere/aerial_perspective.comp'))
        ScatteringLUTCache.configure(self.program)
        self.program['u_multi_scatter_lut'].value = 3
        self.program['u_ringshine_map'].value = 8
        self.textures = [ctx.texture3d(size, 4, dtype='f4') for _ in range(2)]
        for texture in self.textures:
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = texture.repeat_z = False

    def update(self, key, camera_sph, bottom, parameters, azimuth_count,
               inv_projection, inv_view, ring_count, ring_normals,
               planetshine, ringshine):
        # Tiled captures can change the frustum without advancing the frame.
        key = (key, tuple(camera_sph), bottom, inv_projection, inv_view,
               planetshine, ringshine)
        if self.key == key:
            return
        p = self.program
        for name, value in parameters.items():
            if name in p:
                p[name].value = value
        p['u_scattering_azimuth_count'].value = azimuth_count
        p['u_cam_pos'].value = tuple(camera_sph)
        p['u_aerial_bottom_km'].value = bottom
        p['u_inv_proj'].write(inv_projection)
        p['u_inv_view'].write(inv_view)
        p['u_num_ring_planes'].value = ring_count
        if ring_count:
            p['u_ring_normal'].write(ring_normals)
        p['u_planetshine_enabled'].value = planetshine
        p['u_ringshine_enabled'].value = ringshine
        for index, texture in enumerate(self.textures):
            texture.bind_to_image(index, read=False, write=True)
        p.run(*((n + 3) // 4 for n in self.size))
        self.ctx.memory_barrier()
        self.key = key

    def bind(self, program, camera_sph, bottom):
        for unit, texture in zip((39, 40), self.textures):
            texture.use(unit)
        program['u_aerial_camera_sph'].value = tuple(camera_sph)
        program['u_aerial_bottom_km'].value = bottom
        program['u_aerial_enabled'].value = True

    def release(self):
        for texture in self.textures:
            texture.release()
        self.program.release()
