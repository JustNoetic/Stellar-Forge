"""Cached host ringshine maps, one independent tile per host and star."""
import moderngl
import numpy as np

MAX_RING_HOSTS = 16
MAX_RING_SEGMENTS = MAX_RING_HOSTS * 16
RING_PROFILE_ROWS = MAX_RING_HOSTS + MAX_RING_SEGMENTS
RINGSHINE_TILE_SIZE = (128, 65)


def build_secondary_ring_properties(rings, gradient, indices, body_count):
    """Preserve material lobes at radial nodes in the distant-annulus model.

    params[:,3] is a one-based slot into the compact host sample array.
    Geometry remains a distant annulus; optical response is integrated before
    averaging, with shared extinction at overlapping radial nodes.
    """
    from engine.rendering.ring_optics import RING_MAX_ALPHA
    params = np.zeros((body_count,4),dtype='f4')
    normals = np.zeros((body_count,3),dtype='f4')
    colors = np.ones((body_count,3),dtype='f4')
    if indices is None:
        indices = {body:index for index,body in enumerate(dict.fromkeys(r['body_idx'] for r in rings))}
    scattering = np.zeros((len(indices),64,16,12),dtype='f4')
    nodes,weights = np.polynomial.legendre.leggauss(64)
    for slot,body in enumerate(indices):
        if body>=body_count:
            continue
        segments = [ring for ring in rings if ring['body_idx']==body]
        if len(segments)>16:
            raise ValueError("A ring host supports at most 16 material layers")
        if not segments:
            continue
        inner = min(ring['inner_r'] for ring in segments)
        outer = max(ring['outer_r'] for ring in segments)
        if outer<=inner:
            continue
        # Uniform projected annulus area: r^2 is the quadrature coordinate.
        radii = np.sqrt(inner*inner+(nodes+1)*.5*(outer*outer-inner*inner))
        area_weights = weights*.5
        layer_taus = np.zeros((64,len(segments)),dtype='f8')
        for m,ring in enumerate(segments):
            valid = (radii>=ring['inner_r']) & (radii<=ring['outer_r'])
            profile = ring['shadow_grad']
            u = (radii-ring['inner_r'])/max(1e-12,ring['outer_r']-ring['inner_r'])
            xp = np.linspace(0,1,len(profile))
            alpha = np.where(valid,np.clip(np.interp(u,xp,profile[:,3])*ring['opacity'],0,RING_MAX_ALPHA),0)
            tau = -np.log1p(-alpha)
            layer_taus[:,m] = tau
            scattering[slot,:,m,1] = alpha
            scattering[slot,:,m,2:7] = (ring.get('asymmetry',.7),ring.get('backscatter',-.3),
                ring.get('scatter',1),float(ring.get('is_textured',ring.get('tex_sampled') is not None)),
                ring.get('unlit_factor',1))
            for c in range(3):
                scattering[slot,:,m,7+c] = np.maximum(np.interp(u,xp,profile[:,c]),0)**2.2
        total_tau = layer_taus.sum(axis=1)
        scattering[slot,:,:len(segments),0] = total_tau[:,None]
        scattering[slot,:,:len(segments),11] = area_weights[:,None]*layer_taus/np.maximum(total_tau[:,None],1e-30)
        opacity = float(np.dot(area_weights,-np.expm1(-total_tau)))
        params[body] = (inner,outer,opacity,slot+1)
        normals[body] = segments[0]['pole']
    return params,normals,colors,scattering


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
        self.vao = ctx.vertex_array(self.program, [(quad_vbo, '2f', 'in_position')])
        self.cache = {}
        self.last_update_count = 0

    def update(self, gradient, hosts, star_positions, bands, oblate, star_radii=None):
        """Rebake changed tiles. Camera, azimuth, flux and color are lookup inputs.

        Elevation is quantized in sine space to 1e-5 (maximum error 5e-6).
        Atlas revisions invalidate material changes immediately. Finite stellar
        disks contribute to both faces when their centers cross the ring plane.
        """
        bands = max(4, min(1024, int(bands)))
        revision = getattr(gradient, 'profile_revision', 0)
        self.last_update_count = 0
        active = set()
        dirty = []
        width, height = RINGSHINE_TILE_SIZE
        geometry_radius = {}
        for index, center, normal, params, flattening in hosts:
            inner, outer, opacity, radius = map(float, params)
            radius = max(radius, 1e-12)
            geometry_radius[index] = radius
            flattening = min(0.95, max(0.0, float(flattening))) if oblate else 0.0
            geometry = (inner / radius, outer / radius, opacity, flattening)
            for star, position in enumerate(star_positions[:16]):
                direction = np.asarray(position, dtype='f8') - center
                distance = float(np.linalg.norm(direction))
                elevation = float(np.dot(direction, normal) / distance) if distance > 1e-12 else 0.0
                elevation = min(1.0, max(-1.0, round(elevation, 5)))
                tile = (index, star)
                active.add(tile)
                sin_radius = min(.999,max(0.0,float(star_radii[star])/distance)) if star_radii is not None and distance>1e-12 else 0.0
                # Relative quantization preserves small stellar disks at equinox.
                sin_radius = float(format(sin_radius,'.6g'))
                key = (revision, geometry, bands, elevation, sin_radius)
                if self.cache.get(tile) == key:
                    continue
                dirty.append((tile, key, geometry, elevation, sin_radius))
        if dirty:
            # Bake an overwrite, independent of the caller's blend/depth state.
            with self.ctx.scope(framebuffer=self.framebuffer, enable_only=0):
                self.framebuffer.use()
                gradient.use(0)
                for tile, key, geometry, elevation, sin_radius in dirty:
                    index, star = tile
                    self.framebuffer.viewport = (star * width, index * height, width, height)
                    self.program['u_host_params'].value = geometry
                    self.program['u_sun_elevation'].value = elevation
                    self.program['u_sun_radius'].value = sin_radius
                    segments = gradient.host_segments[index]
                    self.program['u_segment_count'].value = len(segments)
                    segment_geometry = np.zeros((16,4),dtype='f4')
                    segment_props = np.zeros((16,4),dtype='f4')
                    families = np.zeros(16,dtype='f4')
                    for m,segment in enumerate(segments):
                        segment_geometry[m] = (segment['inner_r']/geometry_radius[index],
                            segment['outer_r']/geometry_radius[index],segment['opacity'],segment['row_idx'])
                        segment_props[m] = (segment.get('asymmetry',.7),segment.get('backscatter',-.3),
                            segment.get('scatter',1),segment.get('unlit_factor',1))
                        families[m] = float(segment.get('is_textured',False))
                    self.program['u_segment_geometry'].write(segment_geometry.tobytes())
                    self.program['u_segment_props'].write(segment_props.tobytes())
                    self.program['u_segment_textured'].write(families.tobytes())
                    self.program['u_ringshine_band_count'].value = bands
                    self.vao.render(moderngl.TRIANGLE_STRIP)
                    self.cache[tile] = key
                    self.last_update_count += 1
        self.cache = {tile: key for tile, key in self.cache.items() if tile in active}
        return self.last_update_count

    def release(self):
        for resource in (self.vao, self.program, self.framebuffer, self.texture):
            resource.release()
