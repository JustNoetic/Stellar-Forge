"""Camera clearance against the same tiled triangle surface as terrain.vert.

Positions and distances here are body-relative kilometers in float64. Height
samples come from resident CPU copies; collision never reads GPU data or disk.
"""
import math
import numpy as np

GROUND_CLEARANCE_KM = 0.001
MIN_NEAR_KM = 0.0001


def terrain_frame(pole, angle):
    pole = np.asarray(pole, dtype='f8')
    pole = pole / max(np.linalg.norm(pole), 1e-30)
    ref = np.array([1., 0., 0.]) if abs(pole[1]) > .999 else np.array([0., 1., 0.])
    tangent = np.cross(pole, ref)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(pole, tangent)
    c, s = math.cos(angle), math.sin(angle)
    return np.column_stack((tangent, pole, bitangent)) @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def terrain_camera_split(position_km, pole, angle):
    """Split the body-local camera origin before GPU rotation/subtraction.

    The low component preserves ground clearance when the high component is
    rounded at planet-radius scale. Rotation then acts on the small relative
    vector, so spin-dependent rounding cannot shake stationary terrain.
    """
    local = terrain_frame(pole, angle).T @ np.asarray(position_km, dtype='f8')
    high = local.astype('f4')
    return high, (local - high.astype('f8')).astype('f4')


def terrain_face_uv(direction, oblateness):
    """Inverse of the shader's tangent-warped, oblate cube mapping."""
    x, y, z = direction
    y /= max(1e-4, 1. - oblateness)
    axis = int(np.argmax(np.abs([x, y, z])))
    if axis == 0:
        face = 0 if x >= 0 else 1
        xp, yp = (-z if x >= 0 else z) / abs(x), -y / abs(x)
    elif axis == 1:
        face = 2 if y >= 0 else 3
        xp, yp = x / abs(y), (z if y >= 0 else -z) / abs(y)
    else:
        face = 4 if z >= 0 else 5
        xp, yp = (x if z >= 0 else -x) / abs(z), -y / abs(z)
    return face, np.clip((np.arctan([xp, yp]) * (4. / np.pi) + 1.) * .5, 0., 1.)


def terrain_vertices(face, uv, radius, oblateness, elevation):
    xp, yp = np.tan((uv * 2. - 1.) * (np.pi * .25)).T
    one = np.ones(len(uv))
    vectors = ((one, -yp, -xp), (-one, -yp, xp), (xp, one, yp),
               (xp, -one, -yp), (xp, -yp, one), (-xp, -yp, -one))
    sphere = np.column_stack(vectors[face])
    sphere /= np.linalg.norm(sphere, axis=1)[:, None]
    sphere[:, 1] *= 1. - oblateness
    return sphere * radius + sphere / np.linalg.norm(sphere, axis=1)[:, None] * elevation[:, None]


def triangle_radius(direction, triangle):
    """Radial intersection and outward plane normal; ignore skirts."""
    a, b, c = triangle
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal /= max(np.linalg.norm(normal), 1e-30)
    if normal @ a < 0:
        normal = -normal
    cosine = normal @ direction
    if cosine <= 1e-8:
        return None
    radius = (normal @ a) / cosine
    q = radius * direction - a
    aa, bb, cc = ab @ ab, ab @ ac, ac @ ac
    denominator = aa * cc - bb * bb
    if denominator <= 1e-30:
        return None
    u = (cc * (q @ ab) - bb * (q @ ac)) / denominator
    v = (aa * (q @ ac) - bb * (q @ ab)) / denominator
    if min(u, v, 1. - u - v) < -1e-7:
        return None
    return radius, cosine


class CameraSurface:
    def __init__(self, radius, oblateness, pole, angle=0., streamer=None,
                 body_name='', elevation_range=(0., 8.848), max_lod=6,
                 patch_res=32, patches=None, cache_queries=False):
        self.radius = float(radius)
        self.oblateness = np.clip(oblateness, 0., .8)
        self.frame = terrain_frame(pole, angle)
        self.streamer, self.body_name = streamer, body_name
        lo, hi = map(float, elevation_range)
        if not (math.isfinite(lo) and math.isfinite(hi) and hi >= lo):
            lo, hi = 0., 8.848
        self.elevation_range = lo, hi
        self.max_lod, self.patch_res = max(0, int(max_lod)), max(1, int(patch_res))
        self.patches = patches
        # App creates a surface after uploads on each frame. Reuse only exact
        # repeated queries within that frame; sweeps still inspect every point.
        self.cache_queries = cache_queries
        self._query_key = self._query_result = None
        self.outer_radius = self.radius + max(0., hi if streamer is not None else 0.) + .01
        self.cell_size = self.radius / (2**self.max_lod * self.patch_res)

    def _patch_surface(self, direction, face, uv, lod, x, y):
        scale = 2**lod
        local = np.clip(uv * scale - [x, y], 0., 1.)
        ij = np.minimum(np.floor(local * self.patch_res), self.patch_res - 1)
        grid = (ij + np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])) / self.patch_res
        slot, factor, ox, oy = self.streamer.get_tile_slot_or_fallback(
            self.body_name, 'height', face, lod, x, y)
        samples = self.streamer.sample_height(slot, grid * factor + [ox, oy])
        if samples is None:
            elevation = np.zeros(4)
        else:
            lo, hi = self.elevation_range
            elevation = lo + samples * (hi - lo)
            if lo < -0.1:
                elevation = np.maximum(0.0, elevation)
        vertices = terrain_vertices(face, (grid + [x, y]) / scale,
                                    self.radius, self.oblateness, elevation)
        # Match create_terrain_grid_patch: (p0,p2,p1), (p1,p2,p3).
        for indices in ([0, 2, 1], [1, 2, 3]):
            hit = triangle_radius(direction, vertices[indices])
            if hit is not None:
                return hit
        return None

    def surface(self, position):
        if not self.cache_queries:
            return self._surface(position)
        key = np.asarray(position, dtype='f8').tobytes()
        if key != self._query_key:
            self._query_result = self._surface(position)
            self._query_key = key
        return self._query_result

    def _surface(self, position):
        r = np.linalg.norm(position)
        direction = (self.frame.T @ position) / r if r > 1e-30 else np.array([0., 1., 0.])
        q = direction.copy(); q[1] /= 1. - self.oblateness
        smooth_radius = self.radius / np.linalg.norm(q)
        normal = direction.copy(); normal[1] /= (1. - self.oblateness)**2
        cosine = (normal @ direction) / np.linalg.norm(normal)
        if self.streamer is None or r > self.outer_radius + 50.:
            return smooth_radius, cosine
        face, uv = terrain_face_uv(direction, self.oblateness)
        count = 2**self.max_lod
        x, y = np.minimum((uv * count).astype(int), count - 1)
        result = self._patch_surface(direction, face, uv, self.max_lod, x, y)
        # During LOD changes protect both the approaching leaf and the displayed
        # parent/children. Resolve against current residency after uploads.
        if self.patches is not None and len(self.patches):
            p = self.patches
            global_uv = uv * 2. - 1.
            matches = p[(p[:, 4] == face) & (p[:, 0] <= global_uv[0] + 1e-8)
                        & (p[:, 2] >= global_uv[0] - 1e-8) & (p[:, 1] <= global_uv[1] + 1e-8)
                        & (p[:, 3] >= global_uv[1] - 1e-8)]
            for patch in matches:
                hit = self._patch_surface(direction, face, uv, *(int(v) for v in patch[5:8]))
                if hit is not None and (result is None or hit[0] > result[0]):
                    result = hit
        return result if result is not None else (smooth_radius, cosine)

    def clearance(self, position):
        radius, cosine = self.surface(position)
        return (np.linalg.norm(position) - radius) * cosine

    def clamp(self, position):
        position = np.asarray(position, dtype='f8')
        radius, cosine = self.surface(position)
        floor = radius + GROUND_CLEARANCE_KM / max(cosine, 1e-4)
        r = np.linalg.norm(position)
        if r >= floor:
            return position, False
        direction = position / r if r > 1e-30 else self.frame[:, 1]
        return direction * floor, True

    def constrain_motion(self, start, end):
        """Sweep physical translation before clamping its endpoint.

        Bound the search to the body's outer sphere, sample crossed mesh cells,
        and bisect the first contact. Include closest approach so a large step
        through a whole body cannot escape detection on its far side.
        """
        start, end = np.asarray(start, 'f8'), np.asarray(end, 'f8')
        delta = end - start
        distance = np.linalg.norm(delta)
        if distance < 1e-12:
            return self.clamp(end)
        v = delta / distance
        axial = start @ v
        disc = axial**2 + self.outer_radius**2 - start @ start
        if disc < 0:
            return end, False
        root = math.sqrt(disc)
        first, last = max(0., -axial - root), min(distance, -axial + root)
        if first > last:
            return end, False
        # Normal ground flight requires only the endpoints. Long, fast flights
        # inspect the crossed cells while keeping the work bounded.
        n = min(512, max(1, int(math.ceil((last - first) / max(self.cell_size * .25, .001)))))
        stations = np.unique(np.r_[np.linspace(first, last, n + 1), np.clip(-axial, first, last)])
        previous = first
        for station in stations:
            point = start + station * v
            if self.clearance(point) < GROUND_CLEARANCE_KM:
                low, high = previous, station
                for _ in range(24):
                    middle = .5 * (low + high)
                    if self.clearance(start + middle * v) < GROUND_CLEARANCE_KM:
                        high = middle
                    else:
                        low = middle
                corrected, _ = self.clamp(start + low * v)
                return corrected, True
            previous = station
        return self.clamp(end)


def camera_near_plane(clearance_au, au_to_km, fov_deg, aspect):
    """10 cm minimum; bound the whole near-plane rectangle by ground clearance."""
    minimum = MIN_NEAR_KM / au_to_km
    clearance_au = max(GROUND_CLEARANCE_KM / au_to_km, float(clearance_au))
    corner = math.sqrt(1. + math.tan(math.radians(fov_deg) * .5)**2 * (1. + aspect**2))
    return min(1e-4, max(minimum, clearance_au * .02), clearance_au * .9 / corner)
