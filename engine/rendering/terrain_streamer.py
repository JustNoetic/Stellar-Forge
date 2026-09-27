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
    def __init__(self, ctx, tiles_base_dir="data/tiles", pool_capacity=256, tile_size=512):
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
        self.in_flight_requests = set()
        self._shutdown_event = threading.Event()

        # Missing / unavailable tiles cache & per-body max LOD
        self.missing_tiles = set()
        self.max_available_lods = {}

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
        """Pre-load LOD 0 root tiles for all available bodies on startup and record max LODs."""
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

            for m_type in ("diffuse", "clouds"):
                body_m_dir = os.path.join(b_dir, m_type)
                if os.path.isdir(body_m_dir):
                    for face in range(6):
                        key = (body_name.lower(), m_type, face, 0, 0, 0)
                        raw_bytes = self._read_tile_from_disk(*key)
                        if raw_bytes and self.free_slots:
                            slot = self.free_slots.pop(0)
                            self.upload_tile_to_slot(slot, raw_bytes)
                            self.resident_tiles[key] = slot
                            self.slot_to_key[slot] = key
                            self.locked_slots.add(slot)

    def _find_tile_path(self, body_name, map_type, face, lod, x, y):
        dir_path = os.path.join(self.tiles_base_dir, body_name, map_type, str(face), str(lod))
        base_name = f"{x}_{y}"
        for ext in (".jpg", ".png", ".webp", ".jpeg"):
            p = os.path.join(dir_path, base_name + ext)
            if os.path.exists(p):
                return p
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
            else:
                img_rgba = img.convert('RGBA')

            if img_rgba.size != (self.tile_size, self.tile_size):
                img_rgba = img_rgba.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
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
            else:
                self.missing_tiles.add(key)
            self.in_flight_requests.discard(key)

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

    def get_tile_slot_or_fallback(self, body_name: str, map_type: str, face: int, lod: int, x: int, y: int):
        """
        Returns:
            (slot_idx, uv_scale, uv_offset_x, uv_offset_y)
        If the requested tile is resident, returns (slot_idx, 1.0, 0.0, 0.0).
        If not resident, triggers background streaming and returns the closest resident ancestor.
        """
        b_name = body_name.lower()
        max_d = self.max_available_lods.get((b_name, map_type, face), -1)

        # 1. Exact match
        key = (b_name, map_type, face, lod, x, y)
        if key in self.resident_tiles:
            slot = self.resident_tiles[key]
            self.slot_last_used[slot] = self.current_frame
            return slot, 1.0, 0.0, 0.0

        # 2. Check if the clamped highest available tile (max_d) is resident
        if max_d >= 0 and lod > max_d:
            k_clamp = lod - max_d
            eff_x = x >> k_clamp
            eff_y = y >> k_clamp
            eff_key = (b_name, map_type, face, max_d, eff_x, eff_y)
            if eff_key in self.resident_tiles:
                slot = self.resident_tiles[eff_key]
                self.slot_last_used[slot] = self.current_frame
                uv_scale = 1.0 / (1 << k_clamp)
                offset_x = (x & ((1 << k_clamp) - 1)) * uv_scale
                offset_y = (y & ((1 << k_clamp) - 1)) * uv_scale
                return slot, uv_scale, offset_x, offset_y

            # Trigger streaming of the best available tile if not already requested
            if eff_key not in self.missing_tiles and eff_key not in self.in_flight_requests:
                self.request_tile(b_name, map_type, face, max_d, eff_x, eff_y)
        else:
            # Request missing tile if not known to be missing
            if key not in self.missing_tiles:
                self.request_tile(b_name, map_type, face, lod, x, y)

        # 3. Ancestor fallback (nearest resident parent)
        for k in range(1, lod + 1):
            anc_lod = lod - k
            anc_x = x >> k
            anc_y = y >> k
            anc_key = (b_name, map_type, face, anc_lod, anc_x, anc_y)
            if anc_key in self.resident_tiles:
                slot = self.resident_tiles[anc_key]
                self.slot_last_used[slot] = self.current_frame

                uv_scale = 1.0 / (1 << k)
                offset_x = (x & ((1 << k) - 1)) * uv_scale
                offset_y = (y & ((1 << k) - 1)) * uv_scale
                return slot, uv_scale, offset_x, offset_y

        # 4. Default fallback (slot 0 for diffuse, -1 for clouds to avoid opaque fallback blocks)
        if map_type == 'clouds':
            return -1, 1.0, 0.0, 0.0
        return 0, 1.0, 0.0, 0.0

    def upload_tile_to_slot(self, slot_idx: int, raw_bytes: bytes):
        """Upload raw RGBA tile bytes to the specified layer of Texture2DArray."""
        self.texture_array.write(
            raw_bytes,
            viewport=(0, 0, slot_idx, self.tile_size, self.tile_size, 1)
        )

    def process_uploads(self, max_per_frame: int = 4):
        """Called every frame on the main OpenGL render thread to flush completed decodes."""
        self.current_frame += 1

        for _ in range(max_per_frame):
            try:
                key, raw_bytes = self.upload_queue.get_nowait()
            except queue.Empty:
                break

            # If already resident (e.g. redundant request), skip
            if key in self.resident_tiles:
                continue

            # Pick slot: free slot if available, else LRU eviction
            if self.free_slots:
                slot = self.free_slots.pop(0)
            else:
                # Find least recently used unlocked slot
                best_slot = -1
                oldest_frame = float('inf')
                for s in range(self.pool_capacity):
                    if s not in self.locked_slots and self.slot_last_used[s] < oldest_frame:
                        oldest_frame = self.slot_last_used[s]
                        best_slot = s

                if best_slot == -1:
                    continue  # All slots locked

                slot = best_slot
                # Evict old occupant
                if slot in self.slot_to_key:
                    old_key = self.slot_to_key[slot]
                    if old_key in self.resident_tiles:
                        del self.resident_tiles[old_key]

            # Upload to GPU
            self.upload_tile_to_slot(slot, raw_bytes)
            self.resident_tiles[key] = slot
            self.slot_to_key[slot] = key
            self.slot_last_used[slot] = self.current_frame

    def use(self, location: int = 14):
        """Bind the texture array to a texture unit."""
        self.texture_array.use(location=location)

    def shutdown(self):
        self._shutdown_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        try:
            self.texture_array.release()
        except Exception:
            pass
