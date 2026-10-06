"""Render-thread OpenGL adapter for the native planetary tile backend."""
import os
import queue
import time
import numpy as np
from engine.rendering.terrain_native import native


class TerrainTileStreamer:
    def __init__(self, ctx, tiles_base_dir='data/tiles', pool_capacity=1024, tile_size=512):
        self.ctx, self.tiles_base_dir = ctx, tiles_base_dir
        self.pool_capacity, self.tile_size = pool_capacity, tile_size
        workers = min(8, max(1, (os.cpu_count() or 2) - 2))
        self.backend = native.TileBackend(os.path.abspath(tiles_base_dir), pool_capacity, tile_size, workers)
        self.color_capacity = self.backend.color_capacity
        width = tile_size + native.GUTTER * 2
        self.texture_array = ctx.texture_array((width, width, self.color_capacity), 4, dtype='f1')
        self.height_array = ctx.texture_array((width, width, pool_capacity - self.color_capacity), 1, dtype='f4')
        for texture in (self.texture_array, self.height_array):
            texture.filter = (0x2601, 0x2601)
            texture.repeat_x = texture.repeat_y = False
        neutral = np.full((width, width, 4), 128, 'u1')
        neutral[..., 3] = 255
        self.texture_array.write(neutral.tobytes(), viewport=(0, 0, 0, width, width, 1))
        self.reload_queue = queue.Queue()
        self.slot_upload_versions = self.backend.versions
        self.height_table = ctx.buffer(reserve=64 * 16)
        self.height_table_mask = 63
        self._table_key = None

    @property
    def resident_count(self):
        return self.backend.resident_count

    @property
    def height_revision(self):
        return self.backend.revision

    def get_max_lod(self, body_name, map_type='diffuse'):
        return self.backend.max_lod(body_name, map_type)

    def request_tile(self, body_name, map_type, face, lod, x, y, priority=None):
        self.backend.request(body_name, map_type, int(face), int(lod), int(x), int(y), priority)

    def get_tile_slot_or_fallback(self, body_name, map_type, face, lod, x, y):
        return self.backend.resolve(body_name, map_type, int(face), int(lod), int(x), int(y))

    def is_tile_resident(self, body_name, map_type, face, lod, x, y):
        return self.backend.is_resident(body_name, map_type, int(face), int(lod), int(x), int(y))

    def resolve_tiles_batch_multi(self, body_name, map_types, faces, lods, xs, ys):
        coords = np.ascontiguousarray(np.column_stack((faces, lods, xs, ys)), dtype='<f4')
        data = self.backend.resolve_batch(body_name, map_types, coords.tobytes())
        result = np.frombuffer(data, '<f4').reshape(len(map_types), len(coords), 4)
        return tuple([result[i, :, j] for i in range(len(map_types))] for j in range(4))

    def resolve_tiles_batch(self, body_name, map_type, faces, lods, xs, ys):
        return tuple(v[0] for v in self.resolve_tiles_batch_multi(body_name, [map_type], faces, lods, xs, ys))

    def preload_body_lod(self, body_name, map_types=('diffuse', 'clouds', 'height'), target_lod=2):
        for map_type in map_types:
            lod = min(target_lod, self.get_max_lod(body_name, map_type))
            if lod < 0:
                continue
            for face in range(6):
                for x in range(1 << lod):
                    for y in range(1 << lod):
                        self.request_tile(body_name, map_type, face, lod, x, y)

    def sample_height(self, slot, uv):
        coords = np.ascontiguousarray(uv, dtype='<f8')
        data = self.backend.sample_height(int(slot), coords.tobytes())
        return None if data is None else np.frombuffer(data, '<f8').reshape(coords.shape[:-1])

    def surface_vertices(self, body_name, face, lod, x, y, grid, radius, obl, lo, span,
                         water=None, edges=0, resolution=32):
        data = self.backend.surface_vertices(body_name, int(face), int(lod), int(x), int(y),
                                            np.ascontiguousarray(grid, dtype='<f8').tobytes(),
                                            float(radius), float(obl), float(lo), float(span),
                                            water, int(edges), int(resolution))
        return np.frombuffer(data, '<f8').reshape(-1, 3)

    def surface_altitude(self, body_name, camera, radius, obl, lo, span, water=None):
        return self.backend.surface_altitude(body_name, tuple(map(float, camera)),
                                             float(radius), float(obl), float(lo), float(span), water)

    def begin_frame(self):
        self.backend.begin_frame()

    def process_uploads(self, max_per_frame=16, max_time_ms=1.5):
        reloads = set()
        while True:
            try:
                reloads.add(self.reload_queue.get_nowait())
            except queue.Empty:
                break
        for body, layer in reloads:
            self.backend.reload(body, layer)
        started, uploaded = time.perf_counter(), 0
        for _ in range(max_per_frame):
            if uploaded and (time.perf_counter() - started) * 1000 >= max_time_ms:
                break
            item = self.backend.poll()
            if item is None:
                break
            slot, layer, data = item
            width = self.tile_size + native.GUTTER * 2
            array = self.height_array if layer == 'height' else self.texture_array
            physical_slot = slot - self.color_capacity if layer == 'height' else slot
            try:
                array.write(data, viewport=(0, 0, physical_slot, width, width, 1))
            except Exception:
                self.backend.reject(slot)
                raise
            uploaded += 1
        if uploaded or reloads:
            self.slot_upload_versions = self.backend.versions
        for message in self.backend.errors():
            print(f'[Terrain] {message}')
        return uploaded

    def prepare_height_table(self, bodies):
        key = self.height_revision, tuple(sorted(bodies.items()))
        if key != self._table_key:
            data, mask = self.backend.height_table(list(key[1]))
            if len(data) != self.height_table.size:
                self.height_table.orphan(len(data))
            self.height_table.write(data)
            self.height_table_mask = mask
            self._table_key = key
        self.height_table.bind_to_storage_buffer(15)

    def configure_height_program(self, program):
        for name, value in (('u_height_array', 15), ('u_height_table_mask', self.height_table_mask),
                            ('u_height_tile_size', self.tile_size)):
            if name in program:
                program[name].value = value

    def use(self, location=14):
        self.texture_array.use(location=location)
        self.height_array.use(location=15)
        self.height_table.bind_to_storage_buffer(15)

    def reload_layer(self, body_name, map_type=None):
        self.reload_queue.put((body_name.lower(), map_type))

    def shutdown(self):
        self.backend.shutdown()
        self.height_table.release()
        self.height_array.release()
        self.texture_array.release()
