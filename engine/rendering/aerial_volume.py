"""Mode 3 terrain lighting cache; sharp shadows are refined per pixel."""
import moderngl

from engine.rendering.shaders import aerial_volume_compute_shader


class AerialPerspectiveVolume:
    def __init__(self, ctx, size=(64, 64, 128)):
        self.ctx = ctx
        self.size = size
        self.program = ctx.compute_shader(aerial_volume_compute_shader)
        self.scatter = ctx.texture3d(size, 4, dtype="f4")
        self.transmittance = ctx.texture3d(size, 4, dtype="f4")
        for texture in (self.scatter, self.transmittance):
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = texture.repeat_z = False
        for name, unit in (("u_transmittance_lut", 1), ("u_multi_scatter_lut", 3), ("u_ringshine_map", 8)):
            if name in self.program:
                self.program[name].value = unit
        self.baked_key = None
        self.cam_sph = None

    def bake(self, key, cam_sph, sun_dir, inv_proj, inv_view,
             num_rings, ring_normals, planetshine, ringshine, radius_range):
        if self.baked_key == key:
            return
        p = self.program
        p["u_cam_pos"].value = tuple(float(v) for v in cam_sph)
        p["u_sun_dir"].value = tuple(float(v) for v in sun_dir)
        p["u_inv_proj"].write(inv_proj)
        p["u_inv_view"].write(inv_view)
        p["u_radius_range"].value = tuple(float(v) for v in radius_range)
        for name, value in (("u_num_ring_planes", num_rings),
                            ("u_planetshine_enabled", planetshine),
                            ("u_ringshine_enabled", ringshine)):
            if name in p:
                p[name].value = value
        if "u_ring_normal" in p and num_rings:
            p["u_ring_normal"].write(ring_normals)
        self.scatter.bind_to_image(0, read=False, write=True)
        self.transmittance.bind_to_image(1, read=False, write=True)
        p.run(group_x=(self.size[0] + 7) // 8, group_y=(self.size[1] + 7) // 8)
        self.ctx.memory_barrier(moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT | moderngl.TEXTURE_FETCH_BARRIER_BIT)
        self.cam_sph = tuple(float(v) for v in cam_sph)
        self.radius_range = tuple(float(v) for v in radius_range)
        self.baked_key = key

    def bind(self, program):
        self.scatter.use(location=20)
        self.transmittance.use(location=21)
        program["u_aerial_cam_sph"].value = self.cam_sph
        program["u_aerial_radius_range"].value = self.radius_range
        program["u_aerial_volume_enabled"].value = True

    def release(self):
        self.scatter.release()
        self.transmittance.release()
        self.program.release()
