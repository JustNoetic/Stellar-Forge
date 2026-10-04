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
# Quadtree Pyramid Slicing & Boundary Stitching
# -----------------------------------------------------------------------------
# 12 Edge pairs connecting the 6 cube faces:
# (face1, edge1, face2, edge2, reverse_direction)
CUBE_EDGE_PAIRS = [
    (0, 'right', 5, 'left', False),
    (0, 'left', 4, 'right', False),
    (0, 'bottom', 3, 'right', False),
    (0, 'top', 2, 'right', True),
    (1, 'right', 4, 'left', False),
    (1, 'left', 5, 'right', False),
    (1, 'bottom', 3, 'left', True),
    (1, 'top', 2, 'left', False),
    (4, 'bottom', 3, 'top', False),
    (4, 'top', 2, 'bottom', False),
    (5, 'bottom', 3, 'bottom', True),
    (5, 'top', 2, 'top', True),
]

# 8 Cube corner vertices each meeting at 3 faces:
# (face, x_idx, y_idx) where 0 is min and -1 is max
CUBE_CORNERS = [
    [(0, 0, 0), (2, -1, -1), (4, -1, 0)],
    [(0, -1, 0), (2, -1, 0), (5, 0, 0)],
    [(0, 0, -1), (3, -1, 0), (4, -1, -1)],
    [(0, -1, -1), (3, -1, -1), (5, 0, -1)],
    [(1, -1, 0), (2, 0, -1), (4, 0, 0)],
    [(1, 0, 0), (2, 0, 0), (5, -1, 0)],
    [(1, -1, -1), (3, 0, 0), (4, 0, -1)],
    [(1, 0, -1), (3, 0, -1), (5, -1, -1)],
]


def get_edge_slice(arr, edge):
    if edge == 'right':
        return arr[:, -1]
    elif edge == 'left':
        return arr[:, 0]
    elif edge == 'top':
        return arr[0, :]
    elif edge == 'bottom':
        return arr[-1, :]
    raise ValueError(f"Unknown edge: {edge}")


def set_edge_slice(arr, edge, val):
    if edge == 'right':
        arr[:, -1] = val
    elif edge == 'left':
        arr[:, 0] = val
    elif edge == 'top':
        arr[0, :] = val
    elif edge == 'bottom':
        arr[-1, :] = val
    else:
        raise ValueError(f"Unknown edge: {edge}")


def stitch_face_arrays(faces, num_tiles_axis, tile_size=512):
    """
    Applies seamless C0 stitching across all 6 faces:
    1. Internal quadtree seams inside each face
    2. The 12 cube face boundaries across adjacent faces
    3. The 8 cube corners
    Supports 2D (grayscale) and 3D (RGB, RGBA) arrays.
    """
    # 1. Internal seams within each face
    for f in range(6):
        arr = faces[f]
        # Vertical internal seams between tile tx and tx+1
        for k in range(1, num_tiles_axis):
            X = k * tile_size
            avg = 0.5 * (arr[:, X - 1].astype(np.float32) + arr[:, X].astype(np.float32))
            arr[:, X - 1] = np.round(avg)
            arr[:, X] = np.round(avg)

        # Horizontal internal seams between tile ty and ty+1
        for k in range(1, num_tiles_axis):
            Y = k * tile_size
            avg = 0.5 * (arr[Y - 1, :].astype(np.float32) + arr[Y, :].astype(np.float32))
            arr[Y - 1, :] = np.round(avg)
            arr[Y, :] = np.round(avg)

        # 4-way internal corner intersections
        for kx in range(1, num_tiles_axis):
            X = kx * tile_size
            for ky in range(1, num_tiles_axis):
                Y = ky * tile_size
                avg_c = 0.25 * (
                    arr[Y - 1, X - 1].astype(np.float32) +
                    arr[Y - 1, X].astype(np.float32) +
                    arr[Y, X - 1].astype(np.float32) +
                    arr[Y, X].astype(np.float32)
                )
                c_val = np.round(avg_c)
                arr[Y - 1, X - 1] = c_val
                arr[Y - 1, X] = c_val
                arr[Y, X - 1] = c_val
                arr[Y, X] = c_val

    # 2. Cube face edges across 6 faces
    for f1, e1, f2, e2, rev in CUBE_EDGE_PAIRS:
        s1 = get_edge_slice(faces[f1], e1).astype(np.float32)
        s2 = get_edge_slice(faces[f2], e2).astype(np.float32)
        s2_aligned = s2[::-1] if rev else s2
        avg = np.round(0.5 * (s1 + s2_aligned))
        set_edge_slice(faces[f1], e1, avg)
        set_edge_slice(faces[f2], e2, avg[::-1] if rev else avg)

    # 3. 8 Cube corners
    for c_list in CUBE_CORNERS:
        vals = [faces[f][y, x].astype(np.float32) for f, x, y in c_list]
        avg_corner = np.round(np.mean(vals, axis=0))
        for f, x, y in c_list:
            faces[f][y, x] = avg_corner


def _save_tile_array_task(arr, out_path, ext):
    """Helper for concurrent tile disk writes from numpy array."""
    img = Image.fromarray(arr)
    if ext == "jpg":
        img.convert("RGB").save(out_path, "JPEG", quality=90)
    else:
        img.save(out_path, "PNG")


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


def slice_quadtree_pyramid_multi(face_imgs, body_out_dir, max_lod, tile_size=512, fmt="jpg", resample=Image.Resampling.LANCZOS, stitch=True):
    """
    Given 6 master face images, generates the quadtree pyramid tiles for lod in 0..max_lod
    with seamless C0 stitching across all internal tile seams and cube face borders.
    Uses multi-threaded I/O to avoid disk bottlenecks.
    """
    ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"

    save_tasks = []
    for lod in range(max_lod + 1):
        t_lod = time.perf_counter()
        num_tiles_axis = 1 << lod
        target_face_dim = num_tiles_axis * tile_size

        lod_faces = []
        for face_idx in range(6):
            f_img = face_imgs[face_idx]
            if target_face_dim == f_img.width:
                lod_face_img = f_img
            else:
                lod_face_img = f_img.resize((target_face_dim, target_face_dim), resample)
            lod_faces.append(np.array(lod_face_img))

        if stitch:
            stitch_face_arrays(lod_faces, num_tiles_axis, tile_size)

        for face_idx in range(6):
            lod_dir = os.path.join(body_out_dir, str(face_idx), str(lod))
            os.makedirs(lod_dir, exist_ok=True)
            face_arr = lod_faces[face_idx]

            for ty in range(num_tiles_axis):
                for tx in range(num_tiles_axis):
                    tile_box = face_arr[ty * tile_size:(ty + 1) * tile_size, tx * tile_size:(tx + 1) * tile_size]
                    out_path = os.path.join(lod_dir, f"{tx}_{ty}.{ext}")
                    save_tasks.append((tile_box, out_path, ext))

        dt = time.perf_counter() - t_lod
        print(f"  LOD {lod} prepared ({num_tiles_axis * num_tiles_axis * 6} tiles, dim={target_face_dim}) in {dt:.2f} s")

    # Concurrently write tiles to disk
    t_save = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        futures = [executor.submit(_save_tile_array_task, t, p, e) for t, p, e in save_tasks]
        concurrent.futures.wait(futures)
    print(f"  Wrote {len(save_tasks)} tiles to disk in {time.perf_counter() - t_save:.2f} s")


# -----------------------------------------------------------------------------
# Planet Baking Pipeline
# -----------------------------------------------------------------------------
def bake_planet(body_name, src_path, map_type="diffuse", max_lod=3, tile_size=512, out_base="data/tiles", use_cuda=True, bilinear=True, stitch=True):
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

    master_faces = []
    for face_idx in range(6):
        t0 = time.perf_counter()
        print(f"[{body_name}] Reprojecting face {face_idx} ({face_names[face_idx]})...")
        face_img = reproject_face(
            src_img, face_idx, master_face_size,
            src_gpu=src_gpu, dst_gpu=dst_gpu,
            use_cuda=cuda_active, bilinear=bilinear
        )
        master_faces.append(face_img)
        dt = time.perf_counter() - t0
        print(f"[{body_name}] Face {face_idx} reprojected in {dt:.2f} s")

    print(f"[{body_name}] Slicing and stitching quadtree pyramid across all 6 faces (stitch={stitch})...")
    resample = Image.Resampling.BILINEAR if "height" in map_type.lower() else Image.Resampling.LANCZOS
    slice_quadtree_pyramid_multi(
        master_faces, body_out_dir, max_lod, tile_size, fmt=fmt,
        resample=resample, stitch=stitch
    )

    total_time = time.perf_counter() - t_start
    backend_str = f"GPU ({_CUDA_DEVICE_NAME})" if cuda_active else "CPU"
    print(f"[{body_name}] All 6 faces baked and seamlessly stitched using {backend_str} to {body_out_dir} in {total_time:.2f} s!")


def bake_all(max_lod=3, tile_size=512, out_base="data/tiles", solar_system_dir="textures/Solar System", include_clouds=True, include_heights=True, use_cuda=True, bilinear=True, stitch=True):
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
            bake_planet(body, diffuse_path, map_type="diffuse", max_lod=max_lod, tile_size=tile_size, out_base=out_base, use_cuda=use_cuda, bilinear=bilinear, stitch=stitch)

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
                bake_planet(body, cloud_path, map_type="clouds", max_lod=max_lod, tile_size=tile_size, out_base=out_base, use_cuda=use_cuda, bilinear=bilinear, stitch=stitch)

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
                bake_planet(body, height_path, map_type="height", max_lod=max_lod, tile_size=tile_size, out_base=out_base, use_cuda=use_cuda, bilinear=bilinear, stitch=stitch)


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
    parser.add_argument("--no-stitch", action="store_true", help="Disable seamless quadtree boundary stitching")

    args = parser.parse_args()
    use_cuda = not args.no_cuda
    bilinear = not args.nearest
    stitch = not args.no_stitch

    if args.all:
        bake_all(
            max_lod=args.max_lod,
            tile_size=args.tile_size,
            out_base=args.out_dir,
            include_clouds=not args.no_clouds,
            include_heights=not args.no_height,
            use_cuda=use_cuda,
            bilinear=bilinear,
            stitch=stitch
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
        bilinear=bilinear,
        stitch=stitch
    )


if __name__ == "__main__":
    main()
