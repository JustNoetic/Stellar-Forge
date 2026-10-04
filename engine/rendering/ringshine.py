"""Cached host ringshine maps, one independent tile per host and star."""
import moderngl
import numpy as np

MAX_RING_HOSTS = 16
MAX_RING_SEGMENTS = MAX_RING_HOSTS * 16
RING_PROFILE_ROWS = MAX_RING_HOSTS + MAX_RING_SEGMENTS
RING_PROPERTY_ROWS = MAX_RING_HOSTS * 2
RINGSHINE_TILE_SIZE = (128, 65)


def build_secondary_ring_properties(rings, gradient, indices, body_count):
    """Area/opacity-weighted linear material for the distant-body approximation."""
    params = np.zeros((body_count, 4), dtype='f4')
    normals = np.zeros((body_count, 3), dtype='f4')
    colors = np.zeros((body_count, 3), dtype='f4')
    scattering = np.zeros((body_count, 4), dtype='f4')
    if indices is None:
        indices = {body: index for index, body in enumerate(dict.fromkeys(r['body_idx'] for r in rings))}
    for body, index in indices.items():
        if body >= body_count:
            continue
        segments = [ring for ring in rings if ring['body_idx'] == body]
        if not segments:
            continue
        inner = min(ring['inner_r'] for ring in segments)
        outer = max(ring['outer_r'] for ring in segments)
        if gradient is None:
            # Comparison rings have their own geometry rather than a primary
            # atlas row. Preserve their secondary lighting with the same bake.
            from engine.rendering.render_utils import bake_unified_shadow_profile
            segments = [dict(ring, is_textured=ring.get('is_textured', ring.get('tex_sampled') is not None))
                        for ring in segments]
            profile, props, extra = bake_unified_shadow_profile(segments, inner, outer)
        else:
            profile = gradient.atlas_data[index]
            props = gradient.property_data[index * 2]
            extra = gradient.property_data[index * 2 + 1]
        weights = np.linspace(inner, outer, len(profile), dtype='f8')
        weights[[0, -1]] *= 0.5
        flux = weights * profile[:, 3]
        total = float(flux.sum())
        if total <= 0.0 or outer <= inner:
            continue
        params[body] = (inner, outer, total / weights.sum(),
                        np.sum(props[:, 3] * flux) / total)
        colors[body] = np.sum(np.maximum(profile[:, :3], 0.0)**2.2 * flux[:, None], axis=0) / total
        scattering[body, :3] = np.sum(props[:, :3] * flux[:, None], axis=0) / total
        scattering[body, 3] = np.sum(extra[:, 0] * flux) / total
        normals[body] = segments[0]['pole']
    return params, normals, colors, scattering


class RingshineMap:
    def __init__(self, ctx, vertex_shader, fragment_shader, quad_vbo):
        self.ctx = ctx
        width, height = RINGSHINE_TILE_SIZE
        self.texture = ctx.texture((width * 16, height * MAX_RING_HOSTS), 4, dtype='f4')
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = self.texture.repeat_y = False
        self.framebuffer = ctx.framebuffer(color_attachments=[self.texture])
        self.program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.program['u_ring_gradients'].value = 0
        self.program['u_ring_props'].value = 5
        self.vao = ctx.vertex_array(self.program, [(quad_vbo, '2f', 'in_position')])
        self.cache = {}
        self.last_update_count = 0

    def update(self, gradient, properties, hosts, star_positions, bands, oblate):
        """Rebake changed tiles. Camera, azimuth, flux and color are lookup inputs.

        Elevation is quantized in sine space to 1e-5 (maximum error 5e-6).
        Atlas revisions invalidate material changes immediately. Exactly
        edge-on sunlight contributes zero in this thin-slab model.
        """
        bands = max(4, min(1024, int(bands)))
        revision = getattr(gradient, 'profile_revision', 0)
        self.last_update_count = 0
        active = set()
        dirty = []
        width, height = RINGSHINE_TILE_SIZE
        for index, center, normal, params, flattening in hosts:
            inner, outer, opacity, radius = map(float, params)
            radius = max(radius, 1e-12)
            flattening = min(0.95, max(0.0, float(flattening))) if oblate else 0.0
            geometry = (inner / radius, outer / radius, opacity, flattening)
            for star, position in enumerate(star_positions[:16]):
                direction = np.asarray(position, dtype='f8') - center
                distance = float(np.linalg.norm(direction))
                elevation = float(np.dot(direction, normal) / distance) if distance > 1e-12 else 0.0
                elevation = min(1.0, max(-1.0, round(elevation, 5)))
                tile = (index, star)
                active.add(tile)
                key = (revision, geometry, bands, elevation)
                if self.cache.get(tile) == key:
                    continue
                dirty.append((tile, key, geometry, elevation))
        if dirty:
            # Bake an overwrite, independent of the caller's blend/depth state.
            with self.ctx.scope(framebuffer=self.framebuffer, enable_only=0):
                self.framebuffer.use()
                gradient.use(0)
                properties.use(5)
                for tile, key, geometry, elevation in dirty:
                    index, star = tile
                    self.framebuffer.viewport = (star * width, index * height, width, height)
                    self.program['u_ring_index'].value = index
                    self.program['u_host_params'].value = geometry
                    self.program['u_sun_elevation'].value = elevation
                    self.program['u_ringshine_band_count'].value = bands
                    self.vao.render(moderngl.TRIANGLE_STRIP)
                    self.cache[tile] = key
                    self.last_update_count += 1
        self.cache = {tile: key for tile, key in self.cache.items() if tile in active}
        return self.last_update_count

    def release(self):
        for resource in (self.vao, self.program, self.framebuffer, self.texture):
            resource.release()
