#!/usr/bin/env python3
"""
scripts/bake_planet_tiles.py

High-performance offline tool to reproject equirectangular planetary maps
into tangent-corrected Spherified Cube Quadtree tile pyramids (SpaceEngine style).

Supports NVIDIA CUDA GPU acceleration (via CuPy / CUDA raw kernels) with seamless
automatic fallback to CPU NumPy.

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
import concurrent.futures
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Ensure engine path is available if needed
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------------------------------------------------------------
# CUDA Initialization & GPU Kernel
# -----------------------------------------------------------------------------
_CUDA_AVAILABLE = False
_CUDA_DEVICE_NAME = None
_CUDA_KERNEL = None
_cp = None

def init_cuda():
    """Attempt to initialize CUDA via CuPy and compile the reprojection kernel."""
    global _CUDA_AVAILABLE, _CUDA_DEVICE_NAME, _CUDA_KERNEL, _cp

    if _CUDA_KERNEL is not None:
        return _CUDA_AVAILABLE

    cuda_runtime_venv = os.path.join(ROOT_DIR, "venv", "Lib", "site-packages", "nvidia", "cuda_runtime")
    if os.path.exists(cuda_runtime_venv) and "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = cuda_runtime_venv
    import warnings
    warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")

    try:
        import cupy as cp
        if cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0:
            dev_props = cp.cuda.runtime.getDeviceProperties(0)
            dev_name = dev_props["name"]
            if isinstance(dev_name, bytes):
                dev_name = dev_name.decode("utf-8", errors="ignore")

            cuda_kernel_source = r'''
            extern "C" __global__
            void reproject_face_kernel(
                const unsigned char* __restrict__ src,
                int src_w, int src_h, int channels,
                int face, int face_size,
                int bilinear,
                unsigned char* __restrict__ dst
            ) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= face_size || y >= face_size) return;

                const float PI = 3.14159265358979323846f;
                float half_pixel = 0.5f / (float)face_size;
                float step = (2.0f - 2.0f * half_pixel) / (float)(face_size > 1 ? face_size - 1 : 1);
                float uc = -1.0f + half_pixel + (float)x * step;
                float vc = -1.0f + half_pixel + (float)y * step;

                float x_prime = tanf(uc * (PI * 0.25f));
                float y_prime = tanf(vc * (PI * 0.25f));

                float vx, vy, vz;
                if (face == 0) { vx = 1.0f; vy = -y_prime; vz = -x_prime; }
                else if (face == 1) { vx = -1.0f; vy = -y_prime; vz = x_prime; }
                else if (face == 2) { vx = x_prime; vy = 1.0f; vz = y_prime; }
                else if (face == 3) { vx = x_prime; vy = -1.0f; vz = -y_prime; }
                else if (face == 4) { vx = x_prime; vy = -y_prime; vz = 1.0f; }
                else { vx = -x_prime; vy = -y_prime; vz = -1.0f; }

                float inv_norm = rsqrtf(vx * vx + vy * vy + vz * vz);
                float px_dir = vx * inv_norm;
                float py_dir = vy * inv_norm;
                float pz_dir = vz * inv_norm;

                float u_eq = 0.5f + atan2f(pz_dir, px_dir) / (2.0f * PI);
                u_eq = u_eq - floorf(u_eq);
                float clamped_py = fminf(1.0f, fmaxf(-1.0f, py_dir));
                float v_eq = 0.5f - asinf(clamped_py) / PI;

                int dst_idx = (y * face_size + x) * channels;

                if (!bilinear) {
                    int src_x = (int)roundf(u_eq * (float)(src_w - 1));
                    int src_y = (int)roundf(v_eq * (float)(src_h - 1));
                    src_x = max(0, min(src_w - 1, src_x));
                    src_y = max(0, min(src_h - 1, src_y));
                    int src_idx = (src_y * src_w + src_x) * channels;
                    for (int c = 0; c < channels; ++c) {
                        dst[dst_idx + c] = src[src_idx + c];
                    }
                } else {
                    float fx = u_eq * (float)(src_w - 1);
                    float fy = v_eq * (float)(src_h - 1);
                    int x0 = (int)floorf(fx);
                    int y0 = (int)floorf(fy);
                    int x1 = min(src_w - 1, x0 + 1);
                    int y1 = min(src_h - 1, y0 + 1);
                    x0 = max(0, min(src_w - 1, x0));
                    y0 = max(0, min(src_h - 1, y0));

                    float wx1 = fx - (float)x0;
                    float wx0 = 1.0f - wx1;
                    float wy1 = fy - (float)y0;
                    float wy0 = 1.0f - wy1;

                    int idx00 = (y0 * src_w + x0) * channels;
                    int idx10 = (y0 * src_w + x1) * channels;
                    int idx01 = (y1 * src_w + x0) * channels;
                    int idx11 = (y1 * src_w + x1) * channels;

                    for (int c = 0; c < channels; ++c) {
                        float val = wx0 * wy0 * (float)src[idx00 + c] +
                                    wx1 * wy0 * (float)src[idx10 + c] +
                                    wx0 * wy1 * (float)src[idx01 + c] +
                                    wx1 * wy1 * (float)src[idx11 + c];
                        dst[dst_idx + c] = (unsigned char)fminf(255.0f, fmaxf(0.0f, roundf(val)));
                    }
                }
            }
            '''
            kernel = cp.RawKernel(cuda_kernel_source, 'reproject_face_kernel')
            _CUDA_KERNEL = kernel
            _CUDA_DEVICE_NAME = dev_name
            _CUDA_AVAILABLE = True
            _cp = cp
    except Exception as e:
        _CUDA_AVAILABLE = False
        _CUDA_DEVICE_NAME = None

    return _CUDA_AVAILABLE

# Try initializing on module load
init_cuda()


# -----------------------------------------------------------------------------
# CPU Fallback Routines
# -----------------------------------------------------------------------------
def compute_cube_face_coords_cpu(face, face_size):
    """
    Computes equirectangular (u, v) coordinates in [0, 1] for all pixels of a cube face,
    using tangent-corrected mapping to minimize area distortion (CPU NumPy fallback).
    """
    half_pixel = 0.5 / face_size
    grid = np.linspace(-1.0 + half_pixel, 1.0 - half_pixel, face_size, dtype=np.float32)
    uc, vc = np.meshgrid(grid, grid)

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

    u_eq = (0.5 + np.arctan2(p[..., 2], p[..., 0]) / (2.0 * np.pi)) % 1.0
    v_eq = 0.5 - np.arcsin(np.clip(p[..., 1], -1.0, 1.0)) / np.pi
    return u_eq, v_eq


def reproject_face_cpu(src_arr, face, face_size):
    """Reproject source equirectangular array to a single cube face using CPU NumPy."""
    src_h, src_w = src_arr.shape[:2]
    u_eq, v_eq = compute_cube_face_coords_cpu(face, face_size)

    px = np.clip(np.round(u_eq * (src_w - 1)).astype(np.int32), 0, src_w - 1)
    py = np.clip(np.round(v_eq * (src_h - 1)).astype(np.int32), 0, src_h - 1)

    if src_arr.ndim == 2:
        return src_arr[py, px]
    return src_arr[py, px, :]


# -----------------------------------------------------------------------------
# Unified Reprojection Dispatcher
# -----------------------------------------------------------------------------
def reproject_face(src_input, face, face_size, src_gpu=None, dst_gpu=None, use_cuda=True, bilinear=True):
    """
    Reprojects source equirectangular texture to a single cube face.
    If use_cuda is True and CUDA is available, computes entirely on the GPU.
    """
    if use_cuda and _CUDA_AVAILABLE and _CUDA_KERNEL is not None:
        cp = _cp
        if src_gpu is None:
            src_arr = np.array(src_input) if isinstance(src_input, Image.Image) else src_input
            src_gpu = cp.asarray(src_arr)

        src_h, src_w = src_gpu.shape[:2]
        channels = src_gpu.shape[2] if src_gpu.ndim == 3 else 1

        if dst_gpu is None:
            dst_shape = (face_size, face_size, channels) if channels > 1 else (face_size, face_size)
            dst_gpu = cp.empty(dst_shape, dtype=cp.uint8)

        block = (16, 16)
        grid = (int((face_size + block[0] - 1) // block[0]), int((face_size + block[1] - 1) // block[1]))
        _CUDA_KERNEL(grid, block, (src_gpu, src_w, src_h, channels, face, face_size, 1 if bilinear else 0, dst_gpu))
        cp.cuda.Stream.null.synchronize()

        face_arr = cp.asnumpy(dst_gpu)
        if face_arr.ndim == 3 and face_arr.shape[2] == 1:
            face_arr = face_arr.squeeze(-1)
        return Image.fromarray(face_arr)

    # CPU Fallback
    src_arr = np.array(src_input) if isinstance(src_input, Image.Image) else src_input
    face_arr = reproject_face_cpu(src_arr, face, face_size)
    return Image.fromarray(face_arr)


# -----------------------------------------------------------------------------
# Quadtree Pyramid Slicing
# -----------------------------------------------------------------------------
def _save_tile_task(tile_img, out_path, ext):
    """Helper for concurrent tile disk writes."""
    if ext == "jpg":
        tile_img.convert("RGB").save(out_path, "JPEG", quality=90)
    else:
        tile_img.save(out_path, "PNG")


def slice_quadtree_pyramid(face_img, face_idx, out_face_dir, max_lod, tile_size=512, fmt="jpg", resample=Image.Resampling.LANCZOS):
    """
    Given a master face image, generates the quadtree pyramid tiles for lod in 0..max_lod.
    Uses multi-threaded I/O to avoid disk bottlenecks.
    """
    os.makedirs(out_face_dir, exist_ok=True)
    ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"

    save_tasks = []
    # For each LOD level:
    for lod in range(max_lod + 1):
        num_tiles_axis = 1 << lod
        target_face_dim = num_tiles_axis * tile_size
        lod_dir = os.path.join(out_face_dir, str(lod))
        os.makedirs(lod_dir, exist_ok=True)

        if target_face_dim == face_img.width:
            lod_face_img = face_img
        else:
            lod_face_img = face_img.resize((target_face_dim, target_face_dim), resample)

        for ty in range(num_tiles_axis):
            for tx in range(num_tiles_axis):
                box = (tx * tile_size, ty * tile_size, (tx + 1) * tile_size, (ty + 1) * tile_size)
                tile = lod_face_img.crop(box)
                out_path = os.path.join(lod_dir, f"{tx}_{ty}.{ext}")
                save_tasks.append((tile, out_path, ext))

    # Concurrently write tiles to disk
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        futures = [executor.submit(_save_tile_task, t, p, e) for t, p, e in save_tasks]
        concurrent.futures.wait(futures)


# -----------------------------------------------------------------------------
# Planet Baking Pipeline
# -----------------------------------------------------------------------------
def bake_planet(body_name, src_path, map_type="diffuse", max_lod=3, tile_size=512, out_base="data/tiles", use_cuda=True, bilinear=True):
    t_start = time.perf_counter()
    print(f"[{body_name}] Loading source image: {src_path}...")
    src_img = Image.open(src_path)
    # Elevation uses 8-bit luminance and lossless PNG (no color-space conversion).
    if map_type.lower() == "height":
        src_img = src_img.convert("L")
    src_w, src_h = src_img.size
    print(f"[{body_name}] Source resolution: {src_w} x {src_h} ({src_img.mode})")

    # Master face resolution matching max LOD: (2^max_lod) * tile_size
    master_face_size = (1 << max_lod) * tile_size
    print(f"[{body_name}] Max LOD {max_lod} -> Master cube face size: {master_face_size}x{master_face_size}")

    body_out_dir = os.path.join(ROOT_DIR, out_base, body_name, map_type)
    os.makedirs(body_out_dir, exist_ok=True)

    face_names = ["+X (Right)", "-X (Left)", "+Y (North)", "-Y (South)", "+Z (Front)", "-Z (Back)"]
    fmt = "png" if (src_img.mode == "RGBA" or "normal" in map_type.lower() or "height" in map_type.lower()) else "jpg"

    # Pre-upload texture to GPU if CUDA is active
    src_gpu = None
    dst_gpu = None
    cuda_active = use_cuda and _CUDA_AVAILABLE

    if cuda_active:
        print(f"[{body_name}] CUDA backend active: using GPU [{_CUDA_DEVICE_NAME}]")
        t_upload = time.perf_counter()
        src_arr = np.array(src_img)
        src_gpu = _cp.asarray(src_arr)
        channels = src_gpu.shape[2] if src_gpu.ndim == 3 else 1
        dst_shape = (master_face_size, master_face_size, channels) if channels > 1 else (master_face_size, master_face_size)
        dst_gpu = _cp.empty(dst_shape, dtype=_cp.uint8)
        _cp.cuda.Stream.null.synchronize()
        print(f"[{body_name}] Transferred source to GPU VRAM in {(time.perf_counter() - t_upload)*1000:.1f} ms")
    else:
        print(f"[{body_name}] Running on CPU (NumPy)")

    for face_idx in range(6):
        t0 = time.perf_counter()
        print(f"[{body_name}] Baking face {face_idx} ({face_names[face_idx]})...")
        face_img = reproject_face(
            src_img, face_idx, master_face_size,
            src_gpu=src_gpu, dst_gpu=dst_gpu,
            use_cuda=cuda_active, bilinear=bilinear
        )

        out_face_dir = os.path.join(body_out_dir, str(face_idx))
        # Grayscale DEMs overshoot with Lanczos (halo ringing along slope breaks);
        # bilinear keeps the downsampled elevation field clean and monotonic.
        resample = Image.Resampling.BILINEAR if "height" in map_type.lower() else Image.Resampling.LANCZOS
        slice_quadtree_pyramid(face_img, face_idx, out_face_dir, max_lod, tile_size, fmt=fmt, resample=resample)
        dt = time.perf_counter() - t0
        print(f"[{body_name}] Face {face_idx} completed in {dt:.2f} s")

    total_time = time.perf_counter() - t_start
    backend_str = f"GPU ({_CUDA_DEVICE_NAME})" if cuda_active else "CPU"
    print(f"[{body_name}] All 6 faces baked successfully using {backend_str} to {body_out_dir} in {total_time:.2f} s!")


def bake_all(max_lod=3, tile_size=512, out_base="data/tiles", solar_system_dir="textures/Solar System", include_clouds=True, include_heights=True, use_cuda=True, bilinear=True):
    """
    Scans textures/Solar System/ and batch bakes all available planetary maps (diffuse, clouds, heightmaps).
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
            bake_planet(body, diffuse_path, map_type="diffuse", max_lod=max_lod, tile_size=tile_size, out_base=out_base, use_cuda=use_cuda, bilinear=bilinear)

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
                bake_planet(body, cloud_path, map_type="clouds", max_lod=max_lod, tile_size=tile_size, out_base=out_base, use_cuda=use_cuda, bilinear=bilinear)

        # 3. Heightmap (elevation) layer
        if include_heights:
            height_candidates = [
                os.path.join(b_dir, f"{body}_heightmap.png"),
                os.path.join(b_dir, f"{body}_heightmap.jpg"),
            ]
            height_path = None
            for cand in height_candidates:
                if os.path.exists(cand):
                    height_path = cand
                    break

            if height_path:
                print(f"\n==================================================")
                print(f"[{body}] Batch baking heightmap ({os.path.basename(height_path)})...")
                print(f"==================================================")
                bake_planet(body, height_path, map_type="height", max_lod=max_lod, tile_size=tile_size, out_base=out_base, use_cuda=use_cuda, bilinear=bilinear)


def main():
    parser = argparse.ArgumentParser(description="Bake equirectangular planet maps to cubemap quadtree tiles.")
    parser.add_argument("--all", action="store_true", help="Batch bake all bodies found in textures/Solar System/")
    parser.add_argument("--no-clouds", action="store_true", help="Skip cloud maps when running with --all")
    parser.add_argument("--no-height", action="store_true", help="Skip heightmap layers when running with --all")
    parser.add_argument("--body", type=str, default="Moon", help="Body name (e.g. Moon, Earth)")
    parser.add_argument("--input", type=str, default=None, help="Path to input equirectangular texture")
    parser.add_argument("--map-type", type=str, default="diffuse", help="Map type: diffuse, normal, specular, clouds, height")
    parser.add_argument("--max-lod", type=int, default=3, help="Max LOD level (e.g. 2, 3, 4)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile resolution in pixels")
    parser.add_argument("--out-dir", type=str, default="data/tiles", help="Base output directory")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU baking even if CUDA is available")
    parser.add_argument("--nearest", action="store_true", help="Use nearest-neighbor sampling instead of bilinear on CUDA")

    args = parser.parse_args()
    use_cuda = not args.no_cuda
    bilinear = not args.nearest

    if args.all:
        bake_all(
            max_lod=args.max_lod,
            tile_size=args.tile_size,
            out_base=args.out_dir,
            include_clouds=not args.no_clouds,
            include_heights=not args.no_height,
            use_cuda=use_cuda,
            bilinear=bilinear
        )
        return

    input_path = args.input
    if not input_path:
        # Default lookup in textures/Solar System/{body}/
        if args.map_type.lower() == "height":
            cand_list = [
                os.path.join(ROOT_DIR, "textures", "Solar System", args.body, f"{args.body}_heightmap.png"),
                os.path.join(ROOT_DIR, "textures", "Solar System", args.body, f"{args.body}_heightmap.jpg"),
            ]
        else:
            cand_list = [
                os.path.join(ROOT_DIR, "textures", "Solar System", args.body, f"{args.body}.jpg"),
                os.path.join(ROOT_DIR, "textures", "Solar System", args.body, f"{args.body}.png"),
            ]
        for cand in cand_list:
            if os.path.exists(cand):
                input_path = cand
                break
        if not input_path:
            raise FileNotFoundError(f"Could not find texture for body {args.body} in textures/Solar System/")

    bake_planet(
        args.body,
        input_path,
        map_type=args.map_type,
        max_lod=args.max_lod,
        tile_size=args.tile_size,
        out_base=args.out_dir,
        use_cuda=use_cuda,
        bilinear=bilinear
    )


if __name__ == "__main__":
    main()
