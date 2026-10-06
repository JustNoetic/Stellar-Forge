"""Bounded cache of body-local elevation and finite-difference terrain normals.

CPU LOD still selects and packs every patch. Compute only generates changed
geometry; two batched passes share height samples across adjacent vertices.
"""
from collections import OrderedDict
import numpy as np
import moderngl
from engine.rendering.shader_loader import load_shader


class TerrainGeometryCache:
    def __init__(self, ctx, max_patches=4096, max_bytes=96 * 1024 * 1024,
                 max_bakes_per_frame=128):
        self.ctx = ctx
        self.max_patches = max_patches
        self.max_bytes = max_bytes
        self.max_bakes = max_bakes_per_frame
        self.height_program = ctx.compute_shader(load_shader('compute/terrain_cache_height.comp'))
        self.normal_program = ctx.compute_shader(load_shader('compute/terrain_cache_normal.comp'))
        self.jobs = ctx.buffer(reserve=self.max_bakes * 112)
        self.mapping = ctx.buffer(reserve=max_patches * 4)
        self.geometry = self.heights = None
        self.resolution = None
        self.entries = OrderedDict()
        self.free_slots = []
        self.last_hits = self.last_misses = self.last_bakes = 0
        self._ready_signature = None
        self._ready_slots = None

    def _resize(self, resolution):
        if self.resolution == resolution:
            return
        stride = (resolution + 1) ** 2
        self.capacity = min(self.max_patches * 2, self.max_bytes // (stride * 16))
        if self.capacity < 1:
            raise ValueError('terrain grid exceeds geometry cache budget')
        for resource in (self.geometry, self.heights):
            if resource is not None:
                resource.release()
        self.geometry = self.ctx.buffer(reserve=self.capacity * stride * 16)
        self.heights = self.ctx.buffer(reserve=self.max_bakes * stride * 4)
        self.resolution = resolution
        self._ready_signature = None
        self._ready_slots = None
        self.entries.clear()
        self.free_slots = list(range(self.capacity - 1, -1, -1))

    def prepare(self, patches, body_shapes, body_names, streamer, resolution):
        """Return draw slots; unbaked misses retain the original vertex path.

        Keys exclude camera, spin, diffuse residency and lighting. Per-height-slot
        upload generations invalidate recycled layers and edits in place. Protect
        every currently referenced cache entry before choosing eviction victims.
        """
        self._resize(int(resolution))
        count = len(patches)
        if count > self.max_patches:
            raise ValueError('terrain patch count exceeds cache mapping')
        draw_slots = np.full(count, -1, dtype='i4')
        if not count:
            return draw_slots
        body_ids = patches[:, 11].astype('i4')
        shapes = np.asarray(body_shapes, dtype='f4')[body_ids][:, [6, 12]]
        # One compact geometry-only key per patch; diffuse tile uploads do not
        # invalidate height/normal work. Values are packed at draw precision.
        key_data = np.ascontiguousarray(np.column_stack((
            patches[:, :4], patches[:, 8:10], patches[:, 12:18], patches[:, 20:23], shapes)))
        height_slots = patches[:, 12].astype('i4')
        versions = np.asarray(streamer.slot_upload_versions, dtype='u8')[np.maximum(0, height_slots)]
        signature = (streamer.height_revision, key_data.tobytes(), body_ids.tobytes(),
                     versions.tobytes(), tuple(sorted(body_names.items())))
        if signature == self._ready_signature:
            self.last_hits = int(np.count_nonzero(height_slots >= 0))
            self.last_misses = self.last_bakes = 0
            self.mapping.bind_to_storage_buffer(5)
            self.geometry.bind_to_storage_buffer(6)
            return self._ready_slots
        keys = []
        for i, row in enumerate(key_data):
            height_slot = int(patches[i, 12])
            if height_slot < 0:
                keys.append(None)
            else:
                keys.append((body_names[int(body_ids[i])],
                             streamer.height_revision, row.tobytes()))
        pinned = {key for key in keys if key in self.entries}
        pending = []
        self.last_hits = self.last_misses = 0
        for i, key in enumerate(keys):
            if key is None:
                continue
            slot = self.entries.get(key)
            if slot is not None:
                self.entries.move_to_end(key)
                draw_slots[i] = slot
                self.last_hits += 1
                continue
            self.last_misses += 1
            if len(pending) >= self.max_bakes:
                continue
            if self.free_slots:
                slot = self.free_slots.pop()
            else:
                victim = next((k for k in self.entries if k not in pinned), None)
                if victim is None:
                    continue
                slot = self.entries.pop(victim)
            self.entries[key] = slot
            pinned.add(key)
            draw_slots[i] = slot
            pending.append((i, slot))
        self.last_bakes = len(pending)
        if pending:
            records = np.zeros((len(pending), 28), dtype='f4')
            indices = [item[0] for item in pending]
            records[:, :24] = patches[indices]
            records[:, 24:26] = shapes[indices]
            records[:, 26] = [item[1] for item in pending]
            self.jobs.write(records.tobytes())
            self.jobs.bind_to_storage_buffer(5)
            self.geometry.bind_to_storage_buffer(6)
            self.heights.bind_to_storage_buffer(7)
            streamer.configure_height_program(self.height_program)
            streamer.configure_height_program(self.normal_program)
            self.height_program['u_resolution'].value = self.resolution
            self.normal_program['u_resolution'].value = self.resolution
            self.normal_program['u_km_to_au'].value = 1.0 / 149597870.7
            groups = (self.resolution + 8) // 8
            self.height_program.run(group_x=groups, group_y=groups, group_z=len(pending))
            self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
            self.normal_program.run(group_x=groups, group_y=groups, group_z=len(pending))
            self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        self.mapping.write(draw_slots.tobytes())
        self.mapping.bind_to_storage_buffer(5)
        self.geometry.bind_to_storage_buffer(6)
        if np.all(draw_slots[height_slots >= 0] >= 0):
            self._ready_signature = signature
            self._ready_slots = draw_slots
        else:
            self._ready_signature = None
            self._ready_slots = None
        return draw_slots

    def release(self):
        for resource in (self.height_program, self.normal_program, self.jobs,
                         self.mapping, self.geometry, self.heights):
            if resource is not None:
                resource.release()
