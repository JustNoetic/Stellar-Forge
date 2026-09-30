"""
engine/rendering/terrain_streamer.py

Asynchronous Tiled Texture Streamer and GPU Texture2DArray pool manager.
Maintains a fixed VRAM budget (e.g., 256 slices of 512x512) and provides
hierarchical ancestor fallback so rendering never encounters holes or pops.
"""

import os
import threading
import queue
import time
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class TerrainTileStreamer:
    """
    Manages a ModernGL TextureArray containing cached planet tiles.
    """
    def __init__(self, ctx, tiles_base_dir="data/tiles", pool_capacity=512, tile_size=512):
        self.ctx = ctx
        self.tiles_base_dir = tiles_base_dir
        self.pool_capacity = pool_capacity
        self.tile_size = tile_size

        # Create ModernGL Texture2DArray
        # Shape: (width, height, layers), 4 channels (RGBA8)
        self.texture_array = self.ctx.texture_array(
            (self.tile_size, self.tile_size, self.pool_capacity),
            4,
            dtype='f1'
        )
        self.texture_array.filter = (0x2601, 0x2601) # GL_LINEAR, GL_LINEAR
        self.texture_array.repeat_x = False
        self.texture_array.repeat_y = False

        # LRU & Residency state
        # key: (body_name, map_type, face, lod, x, y) -> slot_idx (int)
        self.resident_tiles = {}
        # slot_idx -> key
        self.slot_to_key = {}
        # slot_idx -> last_used_frame (int)
        self.slot_last_used = [0] * self.pool_capacity
        # Locked slots (cannot be evicted, e.g. root LOD 0 tiles)
        self.locked_slots = set()

        self.free_slots = list(range(self.pool_capacity))
        self.current_frame = 0

        # Background worker thread & queues
        self.request_queue = queue.Queue()
        self.upload_queue = queue.Queue()
        self.reload_queue = queue.Queue()
        self.in_flight_requests = set()
        self._shutdown_event = threading.Event()

        # Missing / unavailable tiles cache & per-body max LOD
        self.missing_tiles = set()
        self.max_available_lods = {}

        # Resolution memo partitioned by (b_name, map_type, face) -> {(lod, x, y): (slot, uv_scale, ox, oy)}
        # Invalidated on a per-face basis on residency changes (upload/evict).
        self._resolve_memo = {}

        # Directory file cache: dir_path -> set of filenames, eliminating thousands of os.stat calls
        self._dir_file_cache = {}
        # Initialize default slot 0 with neutral gray
        default_tile = np.full((self.tile_size, self.tile_size, 4), 128, dtype=np.uint8)
        default_tile[..., 3] = 255
        self.upload_tile_to_slot(0, default_tile.tobytes())
        self.locked_slots.add(0)
        self.free_slots.remove(0)

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # Scan available bodies in data/tiles and preload LOD 0
        self._preload_root_tiles()

    def _preload_root_tiles(self):
        """Pre-load LOD 0 and LOD 1 tiles for all available bodies on startup and record max LODs."""
        if not os.path.exists(self.tiles_base_dir):
            return

        for body_name in os.listdir(self.tiles_base_dir):
            b_dir = os.path.join(self.tiles_base_dir, body_name)
            if not os.path.isdir(b_dir):
                continue
            for map_type in os.listdir(b_dir):
                m_dir = os.path.join(b_dir, map_type)
                if not os.path.isdir(m_dir):
                    continue
                for face_s in os.listdir(m_dir):
                    f_dir = os.path.join(m_dir, face_s)
                    if not os.path.isdir(f_dir) or not face_s.isdigit():
                        continue
                    face = int(face_s)
                    max_lod = 0
                    for lod_s in os.listdir(f_dir):
                        if lod_s.isdigit():
                            max_lod = max(max_lod, int(lod_s))
                    self.max_available_lods[(body_name.lower(), map_type, face)] = max_lod

        max_locked = max(6, self.pool_capacity // 2)

        # Pass 1: Preload and lock LOD 0 root tiles for all available bodies (priority bodies first)
        all_bodies = sorted(os.listdir(self.tiles_base_dir), key=lambda b: (0 if b.lower() in ("earth", "moon", "mars") else 1, b))
        for body_name in all_bodies:
            b_dir = os.path.join(self.tiles_base_dir, body_name)
            if not os.path.isdir(b_dir):
                continue
            for m_type in ("diffuse", "clouds", "height"):
                body_m_dir = os.path.join(b_dir, m_type)
                if os.path.isdir(body_m_dir):
                    for face in range(6):
                        key = (body_name.lower(), m_type, face, 0, 0, 0)
                        if key not in self.resident_tiles:
                            raw_bytes = self._read_tile_from_disk(*key)
                            if raw_bytes and self.free_slots and len(self.locked_slots) < max_locked:
                                slot = self.free_slots.pop(0)
                                self.upload_tile_to_slot(slot, raw_bytes)
                                self.resident_tiles[key] = slot
                                self.slot_to_key[slot] = key
                                self.locked_slots.add(slot)

        # Pass 2: Preload and lock LOD 1 for major priority bodies (Earth, Moon) if locked budget allows
        priority_bodies = ("moon", "earth", "mars")
        for b_name in priority_bodies:
            for m_type in ("diffuse", "clouds"):
                body_m_dir = os.path.join(self.tiles_base_dir, b_name, m_type)
                if not os.path.isdir(body_m_dir):
                    continue
                for face in range(6):
                    for x in range(2):
                        for y in range(2):
                            key = (b_name.lower(), m_type, face, 1, x, y)
                            if key not in self.resident_tiles:
                                raw_bytes = self._read_tile_from_disk(*key)
                                if raw_bytes and self.free_slots and len(self.locked_slots) < max_locked:
                                    slot = self.free_slots.pop(0)
                                    self.upload_tile_to_slot(slot, raw_bytes)
                                    self.resident_tiles[key] = slot
                                    self.slot_to_key[slot] = key
                                    self.locked_slots.add(slot)

    def _find_tile_path(self, body_name, map_type, face, lod, x, y):
        dir_path = os.path.join(self.tiles_base_dir, body_name, map_type, str(face), str(lod))
        base_name = f"{x}_{y}"

        files = self._dir_file_cache.get(dir_path)
        if files is None:
            if os.path.isdir(dir_path):
                files = set(os.listdir(dir_path))
            else:
                files = set()
            self._dir_file_cache[dir_path] = files

        for ext in ((".png",) if map_type == 'height' else (".jpg", ".png", ".webp", ".jpeg")):
            fn = base_name + ext
            if fn in files:
                return os.path.join(dir_path, fn)
        return None

    def _read_tile_from_disk(self, body_name, map_type, face, lod, x, y):
        path = self._find_tile_path(body_name, map_type, face, lod, x, y)
        if not path:
            return None
        try:
            img = Image.open(path)
            if map_type == 'clouds':
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img_rgba = img.convert('RGBA')
                else:
                    img_l = img.convert('L')
                    white = Image.new('L', img_l.size, 255)
                    img_rgba = Image.merge('RGBA', (white, white, white, img_l))
            elif map_type == 'height':
                # Single-channel grayscale elevation: replicate luminance into RGB,
                # opaque alpha. Consumers sample .r as normalized elevation in [0, 1].
                img_l = img.convert('L')
                opaque = Image.new('L', img_l.size, 255)
                img_rgba = Image.merge('RGBA', (img_l, img_l, img_l, opaque))
            else:
                img_rgba = img.convert('RGBA')

            if img_rgba.size != (self.tile_size, self.tile_size):
                img_rgba = img_rgba.resize((self.tile_size, self.tile_size), Image.Resampling.BILINEAR)
            return img_rgba.tobytes()
        except Exception as e:
            print(f"[TerrainStreamer] Error reading tile {path}: {e}")
            return None

    def _worker_loop(self):
        while not self._shutdown_event.is_set():
            try:
                key = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            raw_bytes = self._read_tile_from_disk(*key)
            if raw_bytes:
                self.upload_queue.put((key, raw_bytes))
                # Note: key remains in in_flight_requests until process_uploads()
                # uploads it, preventing redundant requests on subsequent frames.
            else:
                self.in_flight_requests.discard(key)
                # Tile absent on disk: request the nearest on-disk ancestor so
                # a resident parent exists for hierarchical UV fallback instead
                # of churning re-requests for the missing key every frame.
                b_name, map_type, face, lod, x, y = key
                max_d = self.max_available_lods.get((b_name, map_type, face), -1)
                min_lod = min(lod - 1, max_d) if max_d >= 0 else lod - 1
                if min_lod >= 0:
                    shift = lod - min_lod
                    anc_key = (b_name, map_type, face, min_lod, x >> shift, y >> shift)
                    if anc_key not in self.resident_tiles and anc_key not in self.missing_tiles and anc_key not in self.in_flight_requests:
                        self.in_flight_requests.add(anc_key)
                        self.request_queue.put(anc_key)
                self.missing_tiles.add(key)

    def request_tile(self, body_name: str, map_type: str, face: int, lod: int, x: int, y: int):
        b_name = body_name.lower()
        max_d = self.max_available_lods.get((b_name, map_type, face), -1)
        if max_d >= 0 and lod > max_d:
            k = lod - max_d
            lod = max_d
            x = x >> k
            y = y >> k

        key = (b_name, map_type, face, lod, x, y)
        if key in self.resident_tiles or key in self.in_flight_requests or key in self.missing_tiles:
            return
        self.in_flight_requests.add(key)
        self.request_queue.put(key)

    def get_max_lod(self, body_name: str, map_type: str = "diffuse") -> int:
        """Return the maximum LOD available on disk across all 6 faces for a body."""
        b_name = body_name.lower()
        max_lod = -1
        for face in range(6):
            lod = self.max_available_lods.get((b_name, map_type, face), -1)
            if lod > max_lod:
                max_lod = lod
        return max_lod

    def is_tile_resident(self, body_name: str, map_type: str, face: int, lod: int, x: int, y: int) -> bool:
        """Returns True if the exact tile or clamped available tile is already resident in GPU VRAM."""
        b_name = body_name.lower()
        max_d = self.max_available_lods.get((b_name, map_type, face), -1)
        if max_d >= 0 and lod > max_d:
            k = lod - max_d
            lod = max_d
            x = x >> k
            y = y >> k
        return (b_name, map_type, face, lod, x, y) in self.resident_tiles

    def preload_body_lod(self, body_name: str, map_types=("diffuse", "clouds"), target_lod: int = 2):
        """
        Request background streaming of low/intermediate LOD tiles up to target_lod for the active body.
        These tiles are loaded with high depth-weighted retention to avoid zoom-out blur.
        """
        b_name = body_name.lower()
        for m_type in map_types:
            for face in range(6):
                max_d = self.max_available_lods.get((b_name, m_type, face), -1)
                eff_lod = min(target_lod, max_d) if max_d >= 0 else target_lod
                if eff_lod < 0:
                    continue
                num_tiles = 1 << eff_lod
                for x in range(num_tiles):
                    for y in range(num_tiles):
                        self.request_tile(b_name, m_type, face, eff_lod, x, y)

    def get_tile_slot_or_fallback(self, body_name: str, map_type: str, face: int, lod: int, x: int, y: int):
        """
        Returns:
            (slot_idx, uv_scale, uv_offset_x, uv_offset_y)
        If the requested tile is resident, returns (slot_idx, 1.0, 0.0, 0.0).
        If not resident, triggers background streaming and returns the closest resident ancestor.
        """
        slot, uv_scale, off_x, off_y = self._resolve_single(body_name, map_type, face, lod, x, y)
        if slot >= 0:
            self.slot_last_used[slot] = self.current_frame
        return slot, uv_scale, off_x, off_y

    def _resolve_single(self, body_name: str, map_type: str, face: int, lod: int, x: int, y: int):
        """
        Single-tile residency probe with hierarchical ancestor fallback.
        Results are memoized per (body_name, map_type, face) until residency
        changes on that face; a memo hit does not touch LRU timestamps — callers stamp.
        """
        b_name = body_name.lower()
        sub_key = (b_name, map_type, face)
        cache = self._resolve_memo.get(sub_key)
        if cache is None:
            cache = self._resolve_memo[sub_key] = {}
        key = (lod, x, y)
        hit = cache.get(key)
        if hit is not None:
            return hit

        result = self._resolve_uncached(b_name, map_type, face, lod, x, y)
        cache[key] = result
        return result

    def _resolve_uncached(self, b_name: str, map_type: str, face: int, lod: int, x: int, y: int):
        """
        Single-tile residency probe with hierarchical ancestor fallback.
        """
        max_d = self.max_available_lods.get((b_name, map_type, face), -1)

        # 1. Exact match
        key = (b_name, map_type, face, lod, x, y)
        slot = self.resident_tiles.get(key)
        if slot is not None:
            return slot, 1.0, 0.0, 0.0

        # 2. Check if the clamped highest available tile (max_d) is resident
        if max_d >= 0 and lod > max_d:
            k_clamp = lod - max_d
            eff_x = x >> k_clamp
            eff_y = y >> k_clamp
            eff_key = (b_name, map_type, face, max_d, eff_x, eff_y)
            slot = self.resident_tiles.get(eff_key)
            if slot is not None:
                uv_scale = 1.0 / (1 << k_clamp)
                return slot, uv_scale, (x & ((1 << k_clamp) - 1)) * uv_scale, (y & ((1 << k_clamp) - 1)) * uv_scale

            # Trigger streaming of the best available tile if not already requested
            if eff_key not in self.missing_tiles and eff_key not in self.in_flight_requests:
                self.request_tile(b_name, map_type, face, max_d, eff_x, eff_y)
        else:
            # Request missing tile if not known to be missing
            if key not in self.missing_tiles:
                self.request_tile(b_name, map_type, face, lod, x, y)

        # 3. Ancestor fallback (nearest resident parent)
        anc_lod = lod - 1
        anc_x = x
        anc_y = y
        while anc_lod >= 0:
            anc_x >>= 1
            anc_y >>= 1
            anc_key = (b_name, map_type, face, anc_lod, anc_x, anc_y)
            slot = self.resident_tiles.get(anc_key)
            if slot is not None:
                k = lod - anc_lod
                uv_scale = 1.0 / (1 << k)
                return slot, uv_scale, (x & ((1 << k) - 1)) * uv_scale, (y & ((1 << k) - 1)) * uv_scale
            anc_lod -= 1

        # 4. Default fallback (slot 0 for diffuse; -1 for clouds/height to avoid
        #    opaque fallback blocks and wrong-layer displacement)
        if map_type in ('clouds', 'height'):
            return -1, 1.0, 0.0, 0.0
        return 0, 1.0, 0.0, 0.0

    def resolve_tiles_batch(self, body_name: str, map_type: str, faces, lods, xs, ys):
        """
        Batch replacement for per-patch get_tile_slot_or_fallback() calls.

        faces/lods/xs/ys are per-patch integer tile coordinates. Returns float32
        arrays (slots, uv_scales, offset_x, offset_y) with identical residency
        and ancestor-fallback semantics. Duplicate tiles are resolved once via
        the cross-frame resolution memo; LRU stamps are written once per slot.
        """
        if not hasattr(faces, '__len__') or len(faces) == 0:
            return (np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))

        faces_l = faces.astype(np.int32).tolist() if isinstance(faces, np.ndarray) else [int(f) for f in faces]
        lods_l = lods.astype(np.int32).tolist() if isinstance(lods, np.ndarray) else [int(l) for l in lods]
        xs_l = xs.astype(np.int32).tolist() if isinstance(xs, np.ndarray) else [int(x) for x in xs]
        ys_l = ys.astype(np.int32).tolist() if isinstance(ys, np.ndarray) else [int(y) for y in ys]
        n = len(faces_l)

        slots = np.empty(n, dtype=np.float32)
        uv_scales = np.empty(n, dtype=np.float32)
        off_x = np.empty(n, dtype=np.float32)
        off_y = np.empty(n, dtype=np.float32)

        resolve = self._resolve_single
        frame = self.current_frame
        slot_last_used = self.slot_last_used
        stamped = set()

        for i in range(n):
            s, u, ox, oy = resolve(body_name, map_type, faces_l[i], lods_l[i], xs_l[i], ys_l[i])
            slots[i] = s
            uv_scales[i] = u
            off_x[i] = ox
            off_y[i] = oy
            if s >= 0 and s not in stamped:
                stamped.add(s)
                slot_last_used[s] = frame
        return slots, uv_scales, off_x, off_y

    def resolve_tiles_batch_multi(self, body_name, map_types, faces, lods, xs, ys):
        """
        Vectorized batch over patches spanning several map types in one call.

        Coordinates (faces/lods/xs/ys) are shared across map types; returns four
        python lists of float32 arrays (slots, uv_scales, off_x, off_y), one
        entry per map_type in map_types, in the same order. Residency,
        ancestor-fallback and memo semantics match resolve_tiles_batch.
        """
        if not hasattr(faces, '__len__') or len(faces) == 0:
            empty = np.empty(0, dtype=np.float32)
            n_mt = len(map_types)
            return ([empty.copy() for _ in range(n_mt)],
                    [empty.copy() for _ in range(n_mt)],
                    [empty.copy() for _ in range(n_mt)],
                    [empty.copy() for _ in range(n_mt)])

        faces_l = faces.astype(np.int32).tolist() if isinstance(faces, np.ndarray) else [int(f) for f in faces]
        lods_l = lods.astype(np.int32).tolist() if isinstance(lods, np.ndarray) else [int(l) for l in lods]
        xs_l = xs.astype(np.int32).tolist() if isinstance(xs, np.ndarray) else [int(x) for x in xs]
        ys_l = ys.astype(np.int32).tolist() if isinstance(ys, np.ndarray) else [int(y) for y in ys]
        n = len(faces_l)

        slots_list = []
        uv_scales_list = []
        off_x_list = []
        off_y_list = []

        resolve = self._resolve_single
        b_name = body_name.lower()
        frame = self.current_frame
        slot_last_used = self.slot_last_used

        for map_type in map_types:
            slots = np.empty(n, dtype=np.float32)
            uv_scales = np.empty(n, dtype=np.float32)
            off_x = np.empty(n, dtype=np.float32)
            off_y = np.empty(n, dtype=np.float32)
            stamped = set()

            for i in range(n):
                s, u, ox, oy = resolve(b_name, map_type, faces_l[i], lods_l[i], xs_l[i], ys_l[i])
                slots[i] = s
                uv_scales[i] = u
                off_x[i] = ox
                off_y[i] = oy
                if s >= 0 and s not in stamped:
                    stamped.add(s)
                    slot_last_used[s] = frame

            slots_list.append(slots)
            uv_scales_list.append(uv_scales)
            off_x_list.append(off_x)
            off_y_list.append(off_y)
        return slots_list, uv_scales_list, off_x_list, off_y_list

    def upload_tile_to_slot(self, slot_idx: int, raw_bytes: bytes):
        """Upload raw RGBA tile bytes to the specified layer of Texture2DArray."""
        self.texture_array.write(
            raw_bytes,
            viewport=(0, 0, slot_idx, self.tile_size, self.tile_size, 1)
        )

    def begin_frame(self):
        """Advances the frame counter used for LRU timestamps. Call once per render frame."""
        self.current_frame += 1

    def process_uploads(self, max_per_frame: int = 4):
        """Called every frame on the main OpenGL render thread to flush completed decodes."""
        reloads = set()
        while not self.reload_queue.empty():
            reloads.add(self.reload_queue.get_nowait())
        if reloads:
            # Rare edit/bake transition: drain the decoder before invalidating so
            # an old in-flight tile cannot reappear after a replacement/deletion.
            self._shutdown_event.set()
            self.worker_thread.join()
            self.request_queue = queue.Queue()
            self.upload_queue = queue.Queue()
            self.in_flight_requests.clear()
            for key, slot in list(self.resident_tiles.items()):
                if (key[0], key[1]) in reloads or (key[0], None) in reloads:
                    del self.resident_tiles[key]
                    self.slot_to_key.pop(slot, None)
                    self.locked_slots.discard(slot)
                    self.free_slots.append(slot)
            self._dir_file_cache.clear()
            self.missing_tiles.clear()
            self._resolve_memo.clear()
            self.max_available_lods.clear()
            self._preload_root_tiles()
            self._shutdown_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

        uploaded_count = 0
        for _ in range(max_per_frame):
            try:
                key, raw_bytes = self.upload_queue.get_nowait()
            except queue.Empty:
                break

            self.in_flight_requests.discard(key)

            # If already resident (e.g. redundant request), skip
            if key in self.resident_tiles:
                continue

            # Pick slot: free slot if available, else depth-weighted LRU eviction
            if self.free_slots:
                slot = self.free_slots.pop(0)
            else:
                # Find least recently used unlocked slot with depth-weighted retention.
                # Lower-LOD parent tiles (LOD 1, 2, 3) get retention bonuses so leaf
                # tiles are evicted first, preserving parent continuity when zooming out.
                best_slot = -1
                lowest_score = float('inf')
                for s in range(self.pool_capacity):
                    if s not in self.locked_slots:
                        old_k = self.slot_to_key.get(s)
                        lod_level = old_k[3] if old_k is not None else 0
                        # 250 frames (~4.2 seconds) of eviction retention per lower LOD level:
                        depth_bonus = max(0, 8 - lod_level) * 250
                        score = self.slot_last_used[s] + depth_bonus
                        if score < lowest_score:
                            lowest_score = score
                            best_slot = s

                if best_slot == -1:
                    continue  # All slots locked

                # Protect active resident tiles from being evicted in recent frames
                # to prevent VRAM pool thrashing and high/low LOD rapid flickering.
                if self.slot_last_used[best_slot] >= self.current_frame - 2:
                    continue

                slot = best_slot
                # Evict old occupant
                if slot in self.slot_to_key:
                    old_key = self.slot_to_key[slot]
                    if old_key in self.resident_tiles:
                        del self.resident_tiles[old_key]
                    # Targeted invalidation: only drop memo for the evicted tile's face
                    self._resolve_memo.pop((old_key[0], old_key[1], old_key[2]), None)

            # Upload to GPU
            self.upload_tile_to_slot(slot, raw_bytes)
            self.resident_tiles[key] = slot
            self.slot_to_key[slot] = key
            self.slot_last_used[slot] = self.current_frame
            # Targeted invalidation: only drop memo for this tile's face
            self._resolve_memo.pop((key[0], key[1], key[2]), None)
            uploaded_count += 1

        return uploaded_count

    def use(self, location: int = 14):
        """Bind the texture array to a texture unit."""
        self.texture_array.use(location=location)

    def reload_layer(self, body_name, map_type=None):
        """Schedule replacement/deletion refresh; safe to call from bake workers."""
        self.reload_queue.put((body_name.lower(), map_type))

    def shutdown(self):
        self._shutdown_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        try:
            self.texture_array.release()
        except Exception:
            pass
