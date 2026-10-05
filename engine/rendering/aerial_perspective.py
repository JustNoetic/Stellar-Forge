"""Camera volumes with horizon-focused sampling for high-altitude terrain haze."""
import moderngl
import numpy as np

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
        self.horizon = False
        self.transition_weight = 1.0
        self.bounds_key = None
        self.texture_sets = {}
        self.textures = self._textures(size)

    @staticmethod
    def _horizon_bounds(camera, inv_projection, inv_view, pole_obl):
        """Conservative azimuth/elevation bounds, including edge extrema."""
        camera = np.asarray(camera, dtype=float)
        zenith = camera / np.linalg.norm(camera)
        projection_inv = np.frombuffer(inv_projection, 'f4').reshape(4, 4).T
        view_inv = np.frombuffer(inv_view, 'f4').reshape(4, 4).T
        pole = np.asarray(pole_obl[:3], dtype=float)
        transform = np.eye(3) + (pole_obl[3] - 1.0) * np.outer(pole, pole)
        rotation = transform @ view_inv[:3, :3]
        forward = -rotation[:, 2]
        axis = forward - np.dot(forward, zenith) * zenith
        if np.linalg.norm(axis) < 1e-8:
            arbitrary = [0, 1, 0] if abs(zenith[1]) < .99 else [1, 0, 0]
            axis = np.cross(zenith, arbitrary)
        axis /= np.linalg.norm(axis)
        across = np.cross(zenith, axis)
        corners = []
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            eye = projection_inv @ [x, y, -1, 1]
            corners.append(rotation @ [eye[0], eye[1], -1])
        points = list(corners)
        for a, end in zip(corners, corners[1:] + corners[:1]):
            delta = end - a
            A, B = np.dot(zenith, a), np.dot(zenith, delta)
            C, D, E = np.dot(a, a), np.dot(a, delta), np.dot(delta, delta)
            denominator = B * D - A * E
            if abs(denominator) > 1e-12:
                t = (A * D - B * C) / denominator
                if 0 < t < 1:
                    points.append(a + t * delta)
        mus = [np.dot(zenith, p) / np.linalg.norm(p) for p in points]
        extent = max(abs(np.arctan2(np.dot(p, across), np.dot(p, axis))) for p in corners)
        projection = np.linalg.inv(projection_inv)
        physical_zenith = np.linalg.solve(transform, zenith)
        for sign in (-1, 1):
            eye = view_inv[:3, :3].T @ (sign * physical_zenith)
            projected = projection @ [*eye, 0]
            if projected[3] > 0 and np.max(np.abs(projected[:2] / projected[3])) <= 1:
                mus.append(float(sign))
                extent = np.pi
        return tuple(axis), min(np.pi, extent * 1.05 + .001), (max(-1., min(mus) - .005), min(1., max(mus) + .005))

    def _textures(self, size):
        if size in self.texture_sets:
            return self.texture_sets[size]
        textures = [self.ctx.texture3d(size, 4, dtype='f4') for _ in range(2)]
        for texture in textures:
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = texture.repeat_z = False
        self.texture_sets[size] = textures
        return textures

    def update(self, key, camera_sph, bottom, parameters, azimuth_count,
               inv_projection, inv_view, ring_count, ring_normals,
               planetshine, ringshine, pole_obl=(0., 1., 0., 1.), viewport_size=None):
        # Tiled captures can change the frustum without advancing the frame.
        key = (key, tuple(camera_sph), bottom, inv_projection, inv_view,
               planetshine, ringshine, tuple(pole_obl), viewport_size)
        if self.key == key:
            return
        radius = sum(float(x) * float(x) for x in camera_sph) ** .5
        altitude = radius - parameters['u_planet_radius_km']
        thickness = parameters['u_atmo_radius_km'] - parameters['u_planet_radius_km']
        candidate = altitude > .45 * thickness
        if candidate:
            bounds_key = (tuple(camera_sph), inv_projection, inv_view, tuple(pole_obl))
            if self.bounds_key != bounds_key:
                self.bounds = self._horizon_bounds(camera_sph, inv_projection, inv_view, pole_obl)
                self.bounds_key = bounds_key
            axis, extent, mu_range = self.bounds
        else:
            axis, extent, mu_range = (1., 0., 0.), np.pi, (-1., 1.)
        full_weight_mu = -max(0., 1. - (parameters['u_planet_radius_km'] / radius) ** 2
            * (1. - .55 ** 2)) ** .5
        grazing = candidate and mu_range[1] > full_weight_mu
        self.horizon = bool(grazing and altitude > .5 * thickness)
        # Fade through exact endpoints as the projection changes, avoiding a
        # discontinuity between the two independently interpolated volumes.
        self.transition_weight = 1.0
        if grazing:
            t = (altitude / thickness - .5) / .05
            t = min(1., max(0., abs(t)))
            self.transition_weight = t * t * (3. - 2. * t)
        orbital_size = (48, 128, 64) if viewport_size and np.prod(viewport_size) >= 1920 * 1080 else (32, 96, 48)
        size = orbital_size if self.horizon else self.size
        self.textures = self._textures(size)
        p = self.program
        p['u_aerial_horizon'].value = self.horizon
        horizon = -(max(0., 1. - (bottom / max(np.linalg.norm(camera_sph), bottom)) ** 2) ** .5)
        self.axis, self.extent = axis, extent
        self.mu_range = (min(mu_range[0], horizon - 1e-5), max(mu_range[1], horizon + 1e-5))
        p['u_aerial_azimuth_axis'].value = axis
        p['u_aerial_azimuth_extent'].value = extent
        p['u_aerial_mu_range'].value = self.mu_range
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
        p.run(*((n + 3) // 4 for n in size))
        self.ctx.memory_barrier()
        self.key = key

    def bind(self, program, camera_sph, bottom):
        for unit, texture in zip((39, 40), self.textures):
            texture.use(unit)
        program['u_aerial_horizon'].value = self.horizon
        program['u_aerial_transition_weight'].value = self.transition_weight
        program['u_aerial_azimuth_axis'].value = self.axis
        program['u_aerial_azimuth_extent'].value = self.extent
        program['u_aerial_mu_range'].value = self.mu_range
        program['u_aerial_camera_sph'].value = tuple(camera_sph)
        program['u_aerial_bottom_km'].value = bottom
        program['u_aerial_enabled'].value = True

    def release(self):
        for textures in self.texture_sets.values():
            for texture in textures:
                texture.release()
        self.program.release()
