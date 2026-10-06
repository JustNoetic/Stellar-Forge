"""Python adapter for the native, budgeted and balanced terrain quadtree."""
import math
import numpy as np
from engine.rendering.terrain_native import native


def cube_to_sphere_point(face, u, v):
    x, y = math.tan(u * math.pi / 4), math.tan(v * math.pi / 4)
    p = np.asarray(((1., -y, -x), (-1., -y, x), (x, 1., y),
                    (x, -1., -y), (x, -y, 1.), (-x, -y, -1.))[face], dtype='f8')
    return p / np.linalg.norm(p)


def terrain_refraction_cull_bend(camera, radius, obl, max_bend, scale_height):
    """Bound the host horizon lift using the same altitude law as terrain.vert."""
    distance = float(np.linalg.norm(camera))
    if max_bend <= 1e-6 or distance <= 1e-6:
        return 0.
    scale_height = max(1e-4, float(scale_height))
    k = 1. / (1. - min(.8, max(0., obl)))
    local_radius = radius / math.sqrt(1. + (k*k - 1.) * (camera[1] / distance)**2)
    height = distance - local_radius
    if height >= scale_height * 15.:
        return 0.
    strength = max_bend * .5 * math.sqrt(2. * local_radius / (math.pi * scale_height))
    density = math.exp(-max(height, 0.) / scale_height)
    dip = math.sqrt(2. * max(height, .002) / local_radius)
    return min(.05, max(0., .5 * strength * density * dip))


class QuadtreePatch:
    __slots__ = ('face', 'lod', 'x', 'y', 'min_u', 'max_u', 'min_v',
                 'max_v', 'center', 'radius', 'center_dir', 'edges')

    def __init__(self, face, lod, x, y, radius_km):
        self.face, self.lod, self.x, self.y = face, lod, x, y
        scale = 2. / (1 << lod)
        self.min_u, self.min_v = -1. + x * scale, -1. + y * scale
        self.max_u, self.max_v = self.min_u + scale, self.min_v + scale
        self.center_dir = cube_to_sphere_point(face, self.min_u + scale / 2, self.min_v + scale / 2)
        self.center = self.center_dir * radius_km
        self.radius = max(np.linalg.norm(cube_to_sphere_point(face, u, v) * radius_km - self.center)
                          for u in (self.min_u, self.max_u) for v in (self.min_v, self.max_v)) * 1.05
        self.edges = 0


class PlanetQuadtree:
    def __init__(self, radius_km, max_lod=4, split_factor=1.25):
        self.radius_km, self.max_lod, self.split_factor = float(radius_km), max_lod, split_factor
        self._history = None
        self._selection_key = None
        self._selection_stable = False

    def update_radius(self, radius_km):
        if self.radius_km != float(radius_km):
            self._history = None
            self._selection_key = None
            self._selection_stable = False
        self.radius_km = float(radius_km)

    def traverse_raw(self, cam_pos_local_km, fov_deg, screen_height, max_lod=None,
                     split_factor=None, obl=0., max_patches=4096, frustum_planes=None,
                     cloud_alt_km=0., refract_bend=0., height_max_km=0., surface_altitude_km=None,
                     frustum_bend=None, height_cull_range=None):
        """Return N×10 float32 patches; final field is the coarse-neighbour edge mask."""
        fov = math.radians(max(.001, min(160., fov_deg)))
        planes = [] if frustum_planes is None else np.asarray(frustum_planes, dtype='f8').tolist()
        camera = tuple(map(float, cam_pos_local_km))
        selection_key = (camera, self.radius_km, float(obl), float(screen_height), math.tan(fov / 2),
                         75. * (self.split_factor if split_factor is None else split_factor),
                         self.max_lod if max_lod is None else int(max_lod), int(max_patches),
                         float(cloud_alt_km), float(refract_bend), float(height_max_km), surface_altitude_km)
        if selection_key != self._selection_key or not self._selection_stable:
            previous = self._history
            data = native.traverse(*selection_key[:8], [], *selection_key[8:11], previous, selection_key[11])
            # Let hysteresis settle before reusing a selection. Frustum changes
            # only recull the retained mesh; camera/zoom/LOD changes rebuild it.
            self._selection_stable = data == previous
            self._selection_key = selection_key
            self._history = data
        data = self._history
        if planes:
            # The app supplies the altitude-dependent host horizon lift. Other
            # callers retain the shader's hard 0.05-radian limit as a safe bound.
            data = native.cull(data, tuple(map(float, cam_pos_local_km)), self.radius_km,
                               float(obl), float(cloud_alt_km + height_max_km), planes,
                               (.05 if refract_bend > 1e-6 else 0.) if frustum_bend is None else float(frustum_bend),
                               height_cull_range)
        return np.frombuffer(data, dtype='<f4').reshape(-1, 10)

    def traverse(self, *args, **kwargs):
        patches = []
        for row in self.traverse_raw(*args, **kwargs):
            p = QuadtreePatch(*(int(v) for v in row[4:8]), self.radius_km)
            p.radius, p.edges = float(row[8]), int(row[9])
            patches.append(p)
        return patches


def pack_terrain_patches(staging, st, raw_patches, uvs, ox, oy, slots, body_idx,
                         h_uvs, h_ox, h_oy, h_slots, elev_min_km, elev_span_km,
                         caster_idx=-1., body_slot=0., water_level=None):
    layers = np.ascontiguousarray(np.column_stack((slots, uvs, ox, oy, h_slots, h_uvs, h_ox, h_oy)), dtype='<f4')
    data = native.pack(np.ascontiguousarray(raw_patches, dtype='<f4').tobytes(), layers.tobytes(),
                       float(body_idx), float(elev_min_km), float(elev_span_km),
                       float(caster_idx), float(body_slot), water_level, False)
    staging[st:st + len(raw_patches)] = np.frombuffer(data, '<f4').reshape(-1, 24)


def pack_cloud_patches(staging, st, raw_patches, uvs, ox, oy, slots, body_idx,
                       caster_idx=-1., body_slot=0.):
    zero = np.zeros(len(raw_patches), dtype='f4')
    layers = np.ascontiguousarray(np.column_stack((slots, uvs, ox, oy, zero - 1., zero, zero, zero)), dtype='<f4')
    data = native.pack(np.ascontiguousarray(raw_patches, dtype='<f4').tobytes(), layers.tobytes(),
                       float(body_idx), 0., 0., float(caster_idx), float(body_slot), None, True)
    staging[st:st + len(raw_patches)] = np.frombuffer(data, '<f4').reshape(-1, 24)
