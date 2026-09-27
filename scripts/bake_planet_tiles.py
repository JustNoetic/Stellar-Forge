#!/usr/bin/env python3
"""
scripts/bake_planet_tiles.py

High-performance offline tool to reproject equirectangular planetary maps
into tangent-corrected Spherified Cube Quadtree tile pyramids (SpaceEngine style).

Outputs:
    data/tiles/{body}/{map_type}/{face}/{lod}/{x}_{y}.jpg
where:
    face in [0..5] (+X, -X, +Y, -Y, +Z, -Z)
    lod in [0..max_lod]
    x, y in [0 .. 2^lod - 1]
"""

import os
import sys
import argparse
import time
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Ensure engine path is available if needed
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def compute_cube_face_coords(face, face_size):
    """
    Computes equirectangular (u, v) coordinates in [0, 1] for all pixels of a cube face,
    using tangent-corrected mapping to minimize area distortion.
    """
    half_pixel = 0.5 / face_size
    grid = np.linspace(-1.0 + half_pixel, 1.0 - half_pixel, face_size, dtype=np.float32)
    uc, vc = np.meshgrid(grid, grid)  # shape (face_size, face_size)

    # Tangent warping: x' = tan(uc * pi / 4), y' = tan(vc * pi / 4)
    x_prime = np.tan(uc * (np.pi / 4.0))
    y_prime = np.tan(vc * (np.pi / 4.0))

    ones = np.ones_like(x_prime)
    if face == 0:    # +X
        v = np.stack([ones, -y_prime, -x_prime], axis=-1)
    elif face == 1:  # -X
        v = np.stack([-ones, -y_prime, x_prime], axis=-1)
    elif face == 2:  # +Y (North Pole)
        v = np.stack([x_prime, ones, y_prime], axis=-1)
    elif face == 3:  # -Y (South Pole)
        v = np.stack([x_prime, -ones, -y_prime], axis=-1)
    elif face == 4:  # +Z
        v = np.stack([x_prime, -y_prime, ones], axis=-1)
    elif face == 5:  # -Z
        v = np.stack([-x_prime, -y_prime, -ones], axis=-1)
    else:
        raise ValueError(f"Invalid face index {face}")

    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    p = v / norm

    # Equirectangular coordinates:
    # u = 0.5 + atan2(p.z, p.x) / (2 * pi)
    # v = 0.5 - asin(p.y) / pi
    u_eq = (0.5 + np.arctan2(p[..., 2], p[..., 0]) / (2.0 * np.pi)) % 1.0
    v_eq = 0.5 - np.arcsin(np.clip(p[..., 1], -1.0, 1.0)) / np.pi
    return u_eq, v_eq


def reproject_face(src_img, face, face_size):
    """
    Reproject source equirectangular PIL image to a single cube face.
    """
    src_w, src_h = src_img.size
    u_eq, v_eq = compute_cube_face_coords(face, face_size)

    # Pixel coordinates
    px = np.clip(np.round(u_eq * (src_w - 1)).astype(np.int32), 0, src_w - 1)
    py = np.clip(np.round(v_eq * (src_h - 1)).astype(np.int32), 0, src_h - 1)

    src_arr = np.array(src_img)
    if src_arr.ndim == 2:
        face_arr = src_arr[py, px]
    else:
        face_arr = src_arr[py, px, :]

    return Image.fromarray(face_arr)


def slice_quadtree_pyramid(face_img, face_idx, out_face_dir, max_lod, tile_size=512, fmt="jpg"):
    """
    Given a master face image, generates the quadtree pyramid tiles for lod in 0..max_lod.
    """
    os.makedirs(out_face_dir, exist_ok=True)
    ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"

    # For each LOD level:
    for lod in range(max_lod + 1):
        num_tiles_axis = 1 << lod
        target_face_dim = num_tiles_axis * tile_size
        lod_dir = os.path.join(out_face_dir, str(lod))
        os.makedirs(lod_dir, exist_ok=True)

        if target_face_dim == face_img.width:
            lod_face_img = face_img
        else:
            lod_face_img = face_img.resize((target_face_dim, target_face_dim), Image.Resampling.LANCZOS)

        for ty in range(num_tiles_axis):
            for tx in range(num_tiles_axis):
                box = (tx * tile_size, ty * tile_size, (tx + 1) * tile_size, (ty + 1) * tile_size)
                tile = lod_face_img.crop(box)
                out_path = os.path.join(lod_dir, f"{tx}_{ty}.{ext}")
                if ext == "jpg":
                    tile.convert("RGB").save(out_path, "JPEG", quality=90)
                else:
                    tile.save(out_path, "PNG")


def bake_planet(body_name, src_path, map_type="diffuse", max_lod=3, tile_size=512, out_base="data/tiles"):
    t_start = time.perf_counter()
    print(f"[{body_name}] Loading source image: {src_path}...")
    src_img = Image.open(src_path)
    src_w, src_h = src_img.size
    print(f"[{body_name}] Source resolution: {src_w} x {src_h} ({src_img.mode})")

    # Master face resolution matching max LOD: (2^max_lod) * tile_size
    master_face_size = (1 << max_lod) * tile_size
    print(f"[{body_name}] Max LOD {max_lod} -> Master cube face size: {master_face_size}x{master_face_size}")

    body_out_dir = os.path.join(ROOT_DIR, out_base, body_name, map_type)
    os.makedirs(body_out_dir, exist_ok=True)

    face_names = ["+X (Right)", "-X (Left)", "+Y (North)", "-Y (South)", "+Z (Front)", "-Z (Back)"]
    fmt = "png" if (src_img.mode == "RGBA" or "normal" in map_type.lower()) else "jpg"

    for face_idx in range(6):
        t0 = time.perf_counter()
        print(f"[{body_name}] Baking face {face_idx} ({face_names[face_idx]})...")
        face_img = reproject_face(src_img, face_idx, master_face_size)

        out_face_dir = os.path.join(body_out_dir, str(face_idx))
        slice_quadtree_pyramid(face_img, face_idx, out_face_dir, max_lod, tile_size, fmt=fmt)
        dt = time.perf_counter() - t0
        print(f"[{body_name}] Face {face_idx} completed in {dt:.2f} s")

    total_time = time.perf_counter() - t_start
    print(f"[{body_name}] All 6 faces baked successfully to {body_out_dir} in {total_time:.2f} s!")


def bake_all(max_lod=3, tile_size=512, out_base="data/tiles", solar_system_dir="textures/Solar System", include_clouds=True):
    """
    Scans textures/Solar System/ and batch bakes all available planetary maps (diffuse and clouds).
    """
    ss_dir = os.path.join(ROOT_DIR, solar_system_dir)
    if not os.path.exists(ss_dir):
        print(f"Error: {ss_dir} does not exist.")
        return

    bodies = [d for d in os.listdir(ss_dir) if os.path.isdir(os.path.join(ss_dir, d))]
    print(f"Found {len(bodies)} celestial bodies in {solar_system_dir}: {', '.join(sorted(bodies))}")

    for body in sorted(bodies):
        b_dir = os.path.join(ss_dir, body)

        # 1. Diffuse surface map
        diffuse_candidates = [
            os.path.join(b_dir, f"{body}.jpg"),
            os.path.join(b_dir, f"{body}.png"),
        ]
        diffuse_path = None
        for cand in diffuse_candidates:
            if os.path.exists(cand):
                diffuse_path = cand
                break

        if diffuse_path:
            print(f"\n==================================================")
            print(f"[{body}] Batch baking diffuse map ({os.path.basename(diffuse_path)})...")
            print(f"==================================================")
            bake_planet(body, diffuse_path, map_type="diffuse", max_lod=max_lod, tile_size=tile_size, out_base=out_base)

        # 2. Cloud layer map
        if include_clouds:
            cloud_candidates = [
                os.path.join(b_dir, f"{body}_clouds.jpg"),
                os.path.join(b_dir, f"{body}_clouds.png"),
            ]
            cloud_path = None
            for cand in cloud_candidates:
                if os.path.exists(cand):
                    cloud_path = cand
                    break

            if cloud_path:
                print(f"\n==================================================")
                print(f"[{body}] Batch baking cloud map ({os.path.basename(cloud_path)})...")
                print(f"==================================================")
                bake_planet(body, cloud_path, map_type="clouds", max_lod=max_lod, tile_size=tile_size, out_base=out_base)


def main():
    parser = argparse.ArgumentParser(description="Bake equirectangular planet maps to cubemap quadtree tiles.")
    parser.add_argument("--all", action="store_true", help="Batch bake all bodies found in textures/Solar System/")
    parser.add_argument("--no-clouds", action="store_true", help="Skip cloud maps when running with --all")
    parser.add_argument("--body", type=str, default="Moon", help="Body name (e.g. Moon, Earth)")
    parser.add_argument("--input", type=str, default=None, help="Path to input equirectangular texture")
    parser.add_argument("--map-type", type=str, default="diffuse", help="Map type: diffuse, normal, specular, clouds")
    parser.add_argument("--max-lod", type=int, default=3, help="Max LOD level (e.g. 2, 3, 4)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile resolution in pixels")
    parser.add_argument("--out-dir", type=str, default="data/tiles", help="Base output directory")

    args = parser.parse_args()

    if args.all:
        bake_all(max_lod=args.max_lod, tile_size=args.tile_size, out_base=args.out_dir, include_clouds=not args.no_clouds)
        return

    input_path = args.input
    if not input_path:
        # Default lookup in textures/Solar System/{body}/
        cand1 = os.path.join(ROOT_DIR, "textures", "Solar System", args.body, f"{args.body}.jpg")
        cand2 = os.path.join(ROOT_DIR, "textures", "Solar System", args.body, f"{args.body}.png")
        if os.path.exists(cand1):
            input_path = cand1
        elif os.path.exists(cand2):
            input_path = cand2
        else:
            raise FileNotFoundError(f"Could not find texture for body {args.body} in textures/Solar System/")

    bake_planet(args.body, input_path, map_type=args.map_type, max_lod=args.max_lod, tile_size=args.tile_size, out_base=args.out_dir)


if __name__ == "__main__":
    main()
