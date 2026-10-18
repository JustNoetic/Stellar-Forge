"""CUDA-accelerated tile sampling for engine-oblate-cube-v1 texture pyramid using CuPy."""
import math
import warnings
import numpy as np

CUDA_KERNEL_CODE = r'''
extern "C" __global__ void sample_level_kernel(
    const unsigned char* __restrict__ source,
    int width, int height,
    int level, int tile_start, int num_tiles,
    double polar, double prime, int latitude_mode,
    const float* __restrict__ decode_lut,
    const unsigned char* __restrict__ encode_lut,
    unsigned char* __restrict__ output
) {
    const int CORE = 128;
    const int GUTTER = 2;
    const int SIDE = 132;
    const double PI = 3.14159265358979323846;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int local_tile = blockIdx.z;

    if (col >= SIDE || row >= SIDE || local_tile >= num_tiles) return;

    int tile_idx = tile_start + local_tile;
    int tiles_per_face = (1 << level) * (1 << level);
    int face = tile_idx / tiles_per_face;
    int face_tile = tile_idx % tiles_per_face;
    int ty = face_tile / (1 << level);
    int tx = face_tile % (1 << level);

    double axis_x = ((double)col - (double)GUTTER + 0.5) / (double)CORE;
    double axis_y = ((double)row - (double)GUTTER + 0.5) / (double)CORE;

    double count = (double)(1 << level);
    double u = -1.0 + 2.0 * ((double)tx + axis_x) / count;
    double v = -1.0 + 2.0 * ((double)ty + axis_y) / count;

    double x, y, z;
    switch (face) {
        case 0: x =  1.0; y =  v;   z = -u;   break;
        case 1: x = -1.0; y =  v;   z =  u;   break;
        case 2: x =  u;   y =  1.0; z = -v;   break;
        case 3: x =  u;   y = -1.0; z =  v;   break;
        case 4: x =  u;   y =  v;   z =  1.0; break;
        default: x = -u;  y =  v;   z = -1.0; break;
    }

    if (latitude_mode == 0) { // geodetic
        y = y / polar;
    } else if (latitude_mode == 1) { // planetocentric
        y = y * polar;
    }

    double u_tex = (atan2(x, z) + prime) / (2.0 * PI) + 0.5;
    double v_tex = 0.5 - atan2(y, hypot(x, z)) / PI;

    double u_mod = fmod(u_tex, 1.0);
    if (u_mod < 0.0) u_mod += 1.0;

    double sx = u_mod * (double)width - 0.5;
    double sy = v_tex * (double)height - 0.5;
    if (sy < 0.0) sy = 0.0;
    if (sy > (double)(height - 1)) sy = (double)(height - 1);

    int ix = (int)floor(sx);
    int iy = (int)floor(sy);
    double fx = sx - (double)ix;
    double fy = sy - (double)iy;

    int nx = (ix + 1) % width;
    if (nx < 0) nx += width;
    int ny = iy + 1;
    if (ny > height - 1) ny = height - 1;

    ix = ix % width;
    if (ix < 0) ix += width;

    size_t tile_offset = (size_t)local_tile * (SIDE * SIDE * 4);
    size_t out_idx = tile_offset + (row * SIDE + col) * 4;

    for (int c = 0; c < 3; ++c) {
        double a = (double)decode_lut[source[(iy * width + ix) * 3 + c]];
        double b = (double)decode_lut[source[(iy * width + nx) * 3 + c]];
        double d = (double)decode_lut[source[(ny * width + ix) * 3 + c]];
        double e = (double)decode_lut[source[(ny * width + nx) * 3 + c]];

        double linear_val = (a * (1.0 - fx) + b * fx) * (1.0 - fy) +
                            (d * (1.0 - fx) + e * fx) * fy;

        if (linear_val < 0.0) linear_val = 0.0;
        if (linear_val > 1.0) linear_val = 1.0;

        int enc_idx = (int)nearbyint(linear_val * 65535.0);
        if (enc_idx < 0) enc_idx = 0;
        if (enc_idx > 65535) enc_idx = 65535;

        output[out_idx + c] = encode_lut[enc_idx];
    }
    output[out_idx + 3] = 255;
}
'''


def is_cuda_available():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False, "No CUDA devices detected"
        name = cp.cuda.runtime.getDeviceProperties(0)['name']
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='replace')
        return True, name
    except Exception as exc:
        return False, str(exc)


class CudaTileSampler:
    CHUNK_SIZE = 1024  # Process at most 1024 tiles per launch (~71 MB)

    def __init__(self, decode_lut, encode_lut):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import cupy as cp
        self.cp = cp
        self.module = cp.RawModule(code=CUDA_KERNEL_CODE, options=('--fmad=false',))
        self.kernel = self.module.get_function('sample_level_kernel')
        self.d_decode = cp.asarray(decode_lut.astype(np.float32))
        self.d_encode = cp.asarray(encode_lut.astype(np.uint8))
        self._current_source = None
        self._d_source = None

    def set_source(self, source_array):
        """Upload or update source mip on GPU."""
        contiguous = np.ascontiguousarray(source_array, dtype=np.uint8)
        self._d_source = self.cp.asarray(contiguous)

    def sample_level(self, level, polar, prime, latitude="geodetic"):
        """Sample all 6 * 4^level tiles for this level using GPU."""
        latitude_mode = 0 if latitude == "geodetic" else 1 if latitude == "planetocentric" else 2
        total_tiles = 6 * (1 << level) * (1 << level)
        height, width = self._d_source.shape[:2]

        tiles_out = []
        for start in range(0, total_tiles, self.CHUNK_SIZE):
            chunk_count = min(self.CHUNK_SIZE, total_tiles - start)
            d_chunk_out = self.cp.empty((chunk_count, 132, 132, 4), dtype=self.cp.uint8)

            grid = ((132 + 15) // 16, (132 + 15) // 16, chunk_count)
            self.kernel(grid, (16, 16, 1), (
                self._d_source, np.int32(width), np.int32(height),
                np.int32(level), np.int32(start), np.int32(chunk_count),
                np.float64(polar), np.float64(prime), np.int32(latitude_mode),
                self.d_decode, self.d_encode, d_chunk_out
            ))
            tiles_out.append(self.cp.asnumpy(d_chunk_out))

        return np.concatenate(tiles_out, axis=0) if len(tiles_out) > 1 else tiles_out[0]
