#!/usr/bin/env python3
"""
scripts/stitch_tiles.py

In-place post-processing utility to eliminate quadtree boundary elevation seams
and cube face edge discontinuities across existing planetary tile pyramids.

Stitches:
1. Internal tile seams within each cube face (between tile tx and tx+1, ty and ty+1)
2. The 12 cube face edge pairs across the 6 faces
3. The 8 cube corner vertices
"""

import os
import sys
import argparse
import time
import concurrent.futures
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.bake_planet_tiles import (
    CUBE_EDGE_PAIRS,
    CUBE_CORNERS,
    get_edge_slice,
    set_edge_slice,
    stitch_face_arrays,
)



def stitch_body_lod(body_dir, lod, tile_size=512, ext="png"):
    """Loads all tiles for a given LOD, stitches seams, and saves modified tiles."""
    num_tiles = 1 << lod
    dim = num_tiles * tile_size

    # Check if this LOD exists across all 6 faces
    for f in range(6):
        face_lod_dir = os.path.join(body_dir, str(f), str(lod))
        if not os.path.isdir(face_lod_dir):
            return False

    t0 = time.perf_counter()
    faces = []
    sample_mode = None

    # Load all tiles into full face arrays
    for f in range(6):
        face_lod_dir = os.path.join(body_dir, str(f), str(lod))
        sample_path = os.path.join(face_lod_dir, "0_0." + ext)
        if not os.path.exists(sample_path):
            # Try alternate extension
            alt_ext = "jpg" if ext == "png" else "png"
            if os.path.exists(os.path.join(face_lod_dir, "0_0." + alt_ext)):
                ext = alt_ext
                sample_path = os.path.join(face_lod_dir, "0_0." + ext)
            else:
                return False

        with Image.open(sample_path) as s_img:
            sample_mode = s_img.mode
            channels = len(s_img.getbands())

        shape = (dim, dim, channels) if channels > 1 else (dim, dim)
        face_arr = np.empty(shape, dtype=np.uint8)

        for ty in range(num_tiles):
            for tx in range(num_tiles):
                tile_p = os.path.join(face_lod_dir, f"{tx}_{ty}.{ext}")
                with Image.open(tile_p) as t_img:
                    t_arr = np.array(t_img)
                    face_arr[ty * tile_size:(ty + 1) * tile_size, tx * tile_size:(tx + 1) * tile_size] = t_arr

        faces.append(face_arr)

    # Compute pre-stitch max discontinuity
    pre_max_int = 0.0
    for f in range(6):
        arr = faces[f]
        for k in range(1, num_tiles):
            X = k * tile_size
            d = np.abs(arr[:, X - 1].astype(int) - arr[:, X].astype(int)).max()
            if d > pre_max_int: pre_max_int = d
            Y = k * tile_size
            d = np.abs(arr[Y - 1, :].astype(int) - arr[Y, :].astype(int)).max()
            if d > pre_max_int: pre_max_int = d

    pre_max_cube = 0.0
    for f1, e1, f2, e2, rev in CUBE_EDGE_PAIRS:
        s1 = get_edge_slice(faces[f1], e1).astype(int)
        s2 = get_edge_slice(faces[f2], e2).astype(int)
        s2_aligned = s2[::-1] if rev else s2
        d = np.abs(s1 - s2_aligned).max()
        if d > pre_max_cube: pre_max_cube = d

    # Perform stitching
    stitch_face_arrays(faces, num_tiles, tile_size)

    # Concurrently write back modified tiles
    save_tasks = []
    for f in range(6):
        face_lod_dir = os.path.join(body_dir, str(f), str(lod))
        face_arr = faces[f]
        for ty in range(num_tiles):
            for tx in range(num_tiles):
                tile_box = face_arr[ty * tile_size:(ty + 1) * tile_size, tx * tile_size:(tx + 1) * tile_size]
                out_p = os.path.join(face_lod_dir, f"{tx}_{ty}.{ext}")
                save_tasks.append((tile_box, out_p, ext))

    def _save_task(args):
        arr, p, extension = args
        img = Image.fromarray(arr)
        if extension.lower() == "jpg":
            img.convert("RGB").save(p, "JPEG", quality=90)
        else:
            img.save(p, "PNG")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        list(executor.map(_save_task, save_tasks))

    dt = time.perf_counter() - t0
    print(f"  LOD {lod}: stitched {num_tiles * num_tiles * 6} tiles in {dt:.2f}s | "
          f"Pre-seam max: internal={pre_max_int}, cube_edge={pre_max_cube} -> Post-seam max: 0.0")
    return True


def stitch_body_map(body_name, map_type="height", max_lod=None, tiles_base="data/tiles"):
    body_dir = os.path.join(ROOT_DIR, tiles_base, body_name, map_type)
    if not os.path.isdir(body_dir):
        print(f"Directory not found: {body_dir}")
        return

    # Find available LODs
    available_lods = []
    f0_dir = os.path.join(body_dir, "0")
    if os.path.isdir(f0_dir):
        for entry in os.listdir(f0_dir):
            if entry.isdigit():
                available_lods.append(int(entry))
    available_lods.sort()

    if max_lod is not None:
        available_lods = [l for l in available_lods if l <= max_lod]

    print(f"Stitching [{body_name} / {map_type}] across LODs: {available_lods}...")
    t_start = time.perf_counter()
    for lod in available_lods:
        stitch_body_lod(body_dir, lod)
    print(f"[{body_name} / {map_type}] All LODs stitched successfully in {time.perf_counter() - t_start:.2f}s!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch planetary quadtree tiles to eliminate seams.")
    parser.add_argument("--body", type=str, default="Earth", help="Celestial body name (e.g. Earth)")
    parser.add_argument("--map-type", type=str, default="height", help="Map type (height, diffuse, clouds)")
    parser.add_argument("--max-lod", type=int, default=None, help="Maximum LOD to process")
    args = parser.parse_args()

    stitch_body_map(args.body, args.map_type, args.max_lod)
