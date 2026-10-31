import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile


CELLS = 16
SIDE = 19
MAX_TILES = 16384


def cube(face, u, v):
    return ((1, v, -u), (-1, v, u), (u, 1, -v), (u, -1, v), (u, v, 1), (-u, v, -1))[face]


def face_uv(p):
    x, y, z = p
    a = max(abs(x), abs(y), abs(z))
    if abs(x) == a:
        return (0, -z / a, y / a) if x >= 0 else (1, z / a, y / a)
    if abs(y) == a:
        return (2, x / a, -z / a) if y >= 0 else (3, x / a, z / a)
    return (4, x / a, y / a) if z >= 0 else (5, -x / a, y / a)


def triangle(values, x, y, side=SIDE, gutter=1):
    ix, iy = min(math.floor(x), CELLS), min(math.floor(y), CELLS)
    u, v = x - ix, y - iy
    def at(dx, dy):
        return values[(iy + dy + gutter) * side + ix + dx + gutter]
    b, c = at(1, 0), at(0, 1)
    return (at(0, 0) * (1-u-v) + b*u + c*v) if u+v <= 1 else (at(1, 1)*(u+v-1) + b*(1-v) + c*(1-u))


def checksum(payload):
    value = 2166136261
    for byte in payload:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


class Raster:
    def __init__(self, document):
        for key, value in {"latitude": "geodetic", "longitude": "east", "orientation": "north-to-south",
                           "registration": "point", "vertical_datum": "ellipsoid"}.items():
            if document.get(key) != value:
                raise ValueError(f"Input requires {key}={value}; convert other conventions before importing")
        self.document = document
        self.rows = document["values"]
        self.width, self.height = len(self.rows[0]), len(self.rows)
        if self.width < 2 or self.height < 2 or any(len(row) != self.width for row in self.rows):
            raise ValueError("Raster must be a rectangular point lattice of at least 2 by 2")
        self.west, self.east, self.south, self.north = document["coverage_deg"]
        if not (-90 <= self.south < self.north <= 90 and 0 < self.east-self.west <= 360):
            raise ValueError("Invalid geographic coverage")
        self.scale, self.offset = float(document["scale"]), float(document["offset"])
        self.fill = float(document["fill_m"])
        self.nodata = document.get("nodata")
        if not all(map(math.isfinite, (self.scale, self.offset, self.fill))):
            raise ValueError("Nonfinite height normalization")

    def sample(self, longitude, latitude):
        longitude = (longitude - self.west) % 360 + self.west
        if longitude > self.east or not self.south <= latitude <= self.north:
            return self.fill
        x = (longitude-self.west) / (self.east-self.west) * (self.width-1)
        y = (self.north-latitude) / (self.north-self.south) * (self.height-1)
        ix, iy = min(int(x), self.width-2), min(int(y), self.height-2)
        u, v = x-ix, y-iy
        result = 0.0
        for dx, dy, weight in ((0, 0, (1-u)*(1-v)), (1, 0, u*(1-v)), (0, 1, (1-u)*v), (1, 1, u*v)):
            raw = self.rows[iy+dy][ix+dx]
            missing = raw is None or raw == self.nodata or not math.isfinite(float(raw))
            value = self.fill if missing else float(raw)*self.scale+self.offset
            result += value*weight
        return result


class ImageRaster:
    def __init__(self, path, scale=None, offset=0.0, height_range_m=None):
        import numpy as np
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 268435456
        with Image.open(path) as image:
            raw = np.asarray(image if image.mode in ("I;16", "I", "F", "L") else image.convert("L"))
        if raw.ndim > 2:
            raw = raw[:, :, 0]
        if raw.dtype in (np.int32, np.uint32, np.int64, np.uint64):
            raw = np.clip(raw, 0, 65535).astype(np.uint16 if raw.max() > 255 else np.uint8)
        elif raw.dtype in (np.float32, np.float64):
            rmin, rmax = float(np.nanmin(raw)), float(np.nanmax(raw))
            norm = (raw - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(raw)
            raw = np.rint(np.clip(norm, 0.0, 1.0) * 65535.0).astype(np.uint16)
        self.values = raw
        self.height, self.width = self.values.shape
        if height_range_m is not None:
            low, high = float(height_range_m[0]), float(height_range_m[1])
            denom = 65535.0 if self.values.dtype == np.uint16 else 255.0
            scale = (high - low) / denom
            offset = low
        self.scale, self.offset = float(scale), float(offset)
        if not math.isfinite(self.scale) or not math.isfinite(self.offset):
            raise ValueError("Image elevation requires finite explicit scale and offset")

    def sample(self, longitude, latitude):
        x = ((longitude+180) % 360)/360*self.width-.5
        y = min(max((90-latitude)/180*self.height-.5, 0), self.height-1)
        ix, iy = math.floor(x), math.floor(y)
        u, v = x-ix, y-iy
        a, b = float(self.values[iy, ix % self.width]), float(self.values[iy, (ix+1) % self.width])
        c, d = float(self.values[min(iy+1, self.height-1), ix % self.width]), float(self.values[min(iy+1, self.height-1), (ix+1) % self.width])
        return ((a*(1-u)+b*u)*(1-v)+(c*(1-u)+d*u)*v)*self.scale+self.offset


class GdalRaster:
    def __init__(self, path, radius, polar, unit, fill):
        try:
            from osgeo import gdal, osr
        except ImportError as error:
            raise ValueError("GDAL Python bindings are unavailable; use geographic JSON input or a Python environment with GDAL") from error
        gdal.UseExceptions()
        self.dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        if self.dataset is None or self.dataset.RasterCount != 1:
            raise ValueError("GDAL input must contain one elevation band")
        self.gdal = gdal
        self.band = self.dataset.GetRasterBand(1)
        crs = self.dataset.GetSpatialRef()
        if crs is None or crs.IsCompound():
            raise ValueError("Convert the input to an explicit horizontal CRS with ellipsoidal heights first")
        geographic = crs.CloneGeogCS()
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        if not geographic.IsSameGeogCS(wgs84):
            raise ValueError("The GDAL adapter currently requires WGS84 horizontal coordinates; convert other horizontal datums first")
        if (abs(geographic.GetSemiMajor()-radius) > radius*1e-8 or
                abs(geographic.GetSemiMinor()/geographic.GetSemiMajor()-polar) > 1e-10):
            raise ValueError("GDAL input must use the target reference ellipsoid and Greenwich longitudes; reproject it first")
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.transform = osr.CoordinateTransformation(geographic, crs)
        self.inverse = gdal.InvGeoTransform(self.dataset.GetGeoTransform())
        if self.inverse is None:
            raise ValueError("GDAL input requires an invertible affine geotransform")
        self.width, self.height = self.dataset.RasterXSize, self.dataset.RasterYSize
        units = {"m": 1, "metre": 1, "meter": 1, "ft": .3048, "foot": .3048, "us survey foot": 1200/3937}
        unit = (unit or self.band.GetUnitType()).lower()
        if unit not in units:
            raise ValueError("Specify --height-unit m or ft when the elevation band has no recognized unit")
        self.unit, self.fill = units[unit], fill
        self.scale, self.offset = self.band.GetScale(), self.band.GetOffset()
        self.scale = 1 if self.scale is None else self.scale
        self.offset = 0 if self.offset is None else self.offset
        self.nodata = self.band.GetNoDataValue()
        if not all(map(math.isfinite, (self.scale, self.offset, fill))):
            raise ValueError("Nonfinite GDAL normalization or fill value")
        self.mask = None if self.band.GetMaskFlags() & gdal.GMF_ALL_VALID else self.band.GetMaskBand()
        self.provenance = dict(crs=crs.ExportToWkt(), geotransform=self.dataset.GetGeoTransform(),
                               scale=self.scale, offset=self.offset, unit=unit, nodata=self.nodata,
                               registration=self.dataset.GetMetadataItem("AREA_OR_POINT"), fill_m=fill,
                               vertical_datum="ellipsoid-explicit-input", gdal=gdal.VersionInfo())

    @functools.lru_cache(maxsize=64)
    def block(self, bx, by):
        x, y = bx*64, by*64
        w, h = min(64, self.width-x), min(64, self.height-y)
        raw = self.band.ReadRaster(x, y, w, h, buf_type=self.gdal.GDT_Float64)
        values = struct.unpack(f"={w*h}d", raw)
        masks = self.mask.ReadRaster(x, y, w, h, buf_type=self.gdal.GDT_Byte) if self.mask else None
        return w, values, masks

    def pixel(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return self.fill
        width, values, mask = self.block(x//64, y//64)
        index = (y % 64)*width+x % 64
        raw = values[index]
        if not math.isfinite(raw) or raw == self.nodata or (mask is not None and mask[index] == 0):
            return self.fill
        return (raw*self.scale+self.offset)*self.unit

    def sample(self, longitude, latitude):
        longitude = (longitude+180) % 360-180
        try:
            x, y, _ = self.transform.TransformPoint(longitude, latitude)
        except RuntimeError:
            return self.fill
        if not math.isfinite(x) or not math.isfinite(y):
            return self.fill
        gt = self.inverse
        x, y = gt[0]+gt[1]*x+gt[2]*y-.5, gt[3]+gt[4]*x+gt[5]*y-.5
        if not (-.5 <= x <= self.width-.5 and -.5 <= y <= self.height-.5):
            return self.fill
        x, y = min(max(x, 0), self.width-1), min(max(y, 0), self.height-1)
        ix, iy = math.floor(x), math.floor(y)
        u, v = x-ix, y-iy
        return sum(self.pixel(min(ix+dx, self.width-1), min(iy+dy, self.height-1))*weight
                   for dx, dy, weight in ((0, 0, (1-u)*(1-v)), (1, 0, u*(1-v)), (0, 1, (1-u)*v), (1, 1, u*v)))


def _cube_vec(face, u, v):
    import numpy as np
    one = np.ones_like(u)
    return ((one, v, -u), (-one, v, u), (u, one, -v), (u, -one, v), (u, v, one), (-u, v, -one))[face]


def _face_uv_vec(px, py, pz):
    import numpy as np
    ax, ay, az = np.abs(px), np.abs(py), np.abs(pz)
    a = np.maximum(np.maximum(ax, ay), az)
    cond_x = ax == a
    cond_y = (~cond_x) & (ay == a)
    c_face = np.where(cond_x, np.where(px >= 0, 0, 1),
             np.where(cond_y, np.where(py >= 0, 2, 3),
                              np.where(pz >= 0, 4, 5)))
    cu = np.where(cond_x, np.where(px >= 0, -pz / a, pz / a),
         np.where(cond_y, px / a,
                          np.where(pz >= 0, px / a, -px / a)))
    cv = np.where(cond_x, py / a,
         np.where(cond_y, np.where(py >= 0, -pz / a, pz / a),
                          py / a))
    return c_face, cu, cv


def _midpoint_residual_2x(g):
    import numpy as np
    a, b, c, d = g[0:-1:2, 0:-1:2], g[0:-1:2, 2::2], g[2::2, 0:-1:2], g[2::2, 2::2]
    r1 = np.abs(g[0:-1:2, 1::2] - 0.5 * (a + b))
    r2 = np.abs(g[1::2, 0:-1:2] - 0.5 * (a + c))
    r3 = np.abs(g[1::2, 2::2] - 0.5 * (b + d))
    r4 = np.abs(g[2::2, 1::2] - 0.5 * (c + d))
    r5 = np.abs(g[1::2, 1::2] - 0.5 * (b + c))
    return np.maximum(np.maximum(np.maximum(r1, r2), np.maximum(r3, r4)), r5)


def import_dense_tiles(output, level, radius, polar, prime, sampler, provenance, name,
                       cells=128, gutter=2, reconstruction="cubic-bspline", raster=None, cuda_sampler=None,
                       progress_cb=None):
    import numpy as np
    count = 2 * (4**(level+1)-1)
    if not 0 <= level <= 5 or count > MAX_TILES:
        raise ValueError("Version 1 supports data levels 0..5 and at most 16,384 metadata records")
    if not (math.isfinite(radius) and radius > 0 and .05 <= polar <= 1 and math.isfinite(prime)):
        raise ValueError("Invalid reference ellipsoid or prime meridian")
    output = Path(output)
    if output.exists():
        raise ValueError("Output must be a new directory")

    side = cells + 1 + 2 * gutter
    finest = cells * (1 << level)
    fside = finest + 1

    if cuda_sampler is None:
        try:
            from scripts.cuda_tile_sampler import is_cuda_available, CudaTileSampler
            if is_cuda_available()[0]:
                cuda_sampler = CudaTileSampler()
        except Exception:
            cuda_sampler = None

    if progress_cb is not None:
        progress_cb(0, count, "Sampling elevation cube faces...")
    if raster is not None and cuda_sampler is not None:
        face_grids, encoding_error = cuda_sampler.sample_height_faces(
            raster.values, finest, polar, prime, raster.scale, raster.offset
        )
    else:
        face_grids = np.empty((6, fside, fside), dtype=np.float32)
        encoding_error = 0.0
        axis = -1.0 + 2.0 * np.arange(fside, dtype=np.float64) / finest
        for face in range(6):
            for y0 in range(0, fside, 256):
                y1 = min(fside, y0 + 256)
                u, v = np.meshgrid(axis, axis[y0:y1])
                px, py, pz = _cube_vec(face, u, v)
                length = np.sqrt(px*px + py*py + pz*pz)
                dx, dy, dz = px / length, py / length, pz / length
                lat = np.degrees(np.arctan2(dy / polar, np.hypot(dx, dz)))
                lon = np.degrees(np.arctan2(dx, dz) + prime)
                if raster is not None:
                    rx = ((lon + 180.0) % 360.0) / 360.0 * raster.width - 0.5
                    ry = np.clip((90.0 - lat) / 180.0 * raster.height - 0.5, 0.0, raster.height - 1.0)
                    ix0 = np.floor(rx).astype(np.int64)
                    iy0 = np.floor(ry).astype(np.int64)
                    fx, fy = rx - ix0, ry - iy0
                    ix1 = (ix0 + 1) % raster.width
                    iy1 = np.minimum(iy0 + 1, raster.height - 1)
                    ix0 = ix0 % raster.width
                    s00 = raster.values[iy0, ix0].astype(np.float64)
                    s10 = raster.values[iy0, ix1].astype(np.float64)
                    s01 = raster.values[iy1, ix0].astype(np.float64)
                    s11 = raster.values[iy1, ix1].astype(np.float64)
                    orig = ((s00*(1.0-fx) + s10*fx)*(1.0-fy) + (s01*(1.0-fx) + s11*fx)*fy)*raster.scale + raster.offset
                else:
                    try:
                        orig = np.asarray(sampler(lon, lat, (dx, dy, dz)), dtype=np.float64)
                        if orig.shape != lon.shape:
                            raise TypeError()
                    except Exception:
                        orig = np.empty_like(lon, dtype=np.float64)
                        for r in range(orig.shape[0]):
                            for c in range(orig.shape[1]):
                                orig[r, c] = sampler(float(lon[r, c]), float(lat[r, c]), (float(dx[r, c]), float(dy[r, c]), float(dz[r, c])))
                enc = orig.astype(np.float32)
                encoding_error = max(encoding_error, float(np.max(np.abs(enc.astype(np.float64) - orig))))
                face_grids[face, y0:y1, :] = enc

    # Enforce exact canonical seam and corner identity across all 6 faces
    axis_idx = np.arange(fside, dtype=np.int64)
    axis_u = -1.0 + 2.0 * axis_idx / finest
    for _ in range(2):
        for face in range(6):
            for edge_y, edge_x in (
                (np.zeros(fside, dtype=np.int64), axis_idx),
                (np.full(fside, finest, dtype=np.int64), axis_idx),
                (axis_idx, np.zeros(fside, dtype=np.int64)),
                (axis_idx, np.full(fside, finest, dtype=np.int64)),
            ):
                u = axis_u[edge_x]
                v = axis_u[edge_y]
                px, py, pz = _cube_vec(face, u, v)
                c_face, cu, cv = _face_uv_vec(px, py, pz)
                cx = np.rint((cu + 1.0) * (finest * 0.5)).astype(np.int64)
                cy = np.rint((cv + 1.0) * (finest * 0.5)).astype(np.int64)
                face_grids[face, edge_y, edge_x] = face_grids[c_face, cy, cx]

    if not np.all(np.isfinite(face_grids)) or float(np.max(np.abs(face_grids))) >= radius * 0.01:
        raise ValueError("Height is nonfinite or exceeds the supported one-percent relief envelope")

    def continuous_vec(face, u, v):
        px, py, pz = _cube_vec(face, u, v)
        c_face, cu, cv = _face_uv_vec(px, py, pz)
        x = (cu + 1.0) * (finest * 0.5)
        y = (cv + 1.0) * (finest * 0.5)
        ix = np.minimum(np.floor(x).astype(np.int64), finest - 1)
        iy = np.minimum(np.floor(y).astype(np.int64), finest - 1)
        fx, fy = x - ix, y - iy
        a = face_grids[c_face, iy, ix].astype(np.float64)
        b = face_grids[c_face, iy, ix + 1].astype(np.float64)
        c = face_grids[c_face, iy + 1, ix].astype(np.float64)
        d = face_grids[c_face, iy + 1, ix + 1].astype(np.float64)
        res = np.where(fx + fy <= 1.0, a * (1.0 - fx - fy) + b * fx + c * fy, d * (fx + fy - 1.0) + b * (1.0 - fy) + c * (1.0 - fx))
        return res.astype(np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terrain-import-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        records = {}
        offsets = {}
        offset = 0
        tile_bytes = side * side * 4
        for depth in range(level + 1):
            for face in range(6):
                for y in range(1 << depth):
                    for x in range(1 << depth):
                        offsets[face, depth, x, y] = offset
                        offset += tile_bytes

        gx_arr = np.arange(-gutter, cells + gutter + 1, dtype=np.int64)
        gy_arr = np.arange(-gutter, cells + gutter + 1, dtype=np.int64)
        step = cells // 16

        payload_path = temporary / "heights.bin"
        done_tiles = 0
        with payload_path.open("w+b") as payload:
            payload.truncate(offset)
            for depth in range(level, -1, -1):
                n = 1 << depth
                width = 2.0 / n
                stride = 1 << (level - depth)
                for face in range(6):
                    face_tiles = np.empty((n * n, side, side), dtype=np.float32)
                    face_meta = []
                    for ty in range(n):
                        fy_arr = (ty * cells + gy_arr) * stride
                        iy_valid = (fy_arr >= 0) & (fy_arr <= finest)
                        vy0, vy1 = int(np.argmax(iy_valid)), int(len(iy_valid) - 1 - np.argmax(iy_valid[::-1]))
                        for tx in range(n):
                            fx_arr = (tx * cells + gx_arr) * stride
                            idx = ty * n + tx
                            if fx_arr[0] >= 0 and fx_arr[-1] <= finest and fy_arr[0] >= 0 and fy_arr[-1] <= finest:
                                tile_vals = face_grids[face, fy_arr[0]:fy_arr[-1]+1:stride, fx_arr[0]:fx_arr[-1]+1:stride]
                                face_tiles[idx] = tile_vals
                            else:
                                ix_valid = (fx_arr >= 0) & (fx_arr <= finest)
                                vx0, vx1 = int(np.argmax(ix_valid)), int(len(ix_valid) - 1 - np.argmax(ix_valid[::-1]))
                                tile_vals = face_tiles[idx]
                                tile_vals[vy0:vy1+1, vx0:vx1+1] = face_grids[
                                    face, fy_arr[vy0]:fy_arr[vy1]+1:stride, fx_arr[vx0]:fx_arr[vx1]+1:stride
                                ]
                                oob_gy, oob_gx = np.where(~(iy_valid[:, None] & ix_valid[None, :]))
                                u = -1.0 + (tx + gx_arr[oob_gx] / cells) * width
                                v = -1.0 + (ty + gy_arr[oob_gy] / cells) * width
                                tile_vals[oob_gy, oob_gx] = continuous_vec(face, u, v)

                            core64 = face_tiles[idx, gutter:gutter+cells+1, gutter:gutter+cells+1].astype(np.float64)
                            low, high = float(core64.min()), float(core64.max())
                            a, b, c, d = core64[:-1, :-1], core64[:-1, 1:], core64[1:, :-1], core64[1:, 1:]
                            gradient = float(max(np.hypot(b - a, c - a).max(), np.hypot(d - c, d - b).max())) * (cells / width)

                            if depth == level:
                                if step == 8:
                                    err_l2 = _midpoint_residual_2x(core64).reshape(4, 16, 4, 16).max(axis=(1, 3))
                                    err_l1 = _midpoint_residual_2x(core64[::2, ::2]).reshape(2, 16, 2, 16).max(axis=(1, 3)) + err_l2.reshape(2, 2, 2, 2).max(axis=(1, 3))
                                    error = float(_midpoint_residual_2x(core64[::4, ::4]).max() + err_l1.max())
                                else:
                                    error = 0.0
                            else:
                                error = 0.0
                                child_res = _midpoint_residual_2x(core64[::step//2, ::step//2]).reshape(2, 8, 2, 8).max(axis=(1, 3))
                                for cy in range(2):
                                    for cx in range(2):
                                        child = records[face, depth + 1, tx * 2 + cx, ty * 2 + cy]
                                        low = min(low, child[6])
                                        high = max(high, child[7])
                                        error = max(error, float(child_res[cy, cx]) + child[8])
                                        gradient = max(gradient, child[9])
                            face_meta.append((tx, ty, low, high, error, gradient))

                    bytes_2d = np.ascontiguousarray(face_tiles).view(np.uint8).reshape(n * n, tile_bytes)
                    if cuda_sampler is not None:
                        csums = cuda_sampler.checksum_tiles(bytes_2d)
                    else:
                        csums = [checksum(bytes_2d[i].tobytes()) for i in range(n * n)]

                    payload.seek(offsets[face, depth, 0, 0])
                    payload.write(bytes_2d.tobytes())
                    for i, (tx, ty, low, high, error, gradient) in enumerate(face_meta):
                        key = face, depth, tx, ty
                        records[key] = [*key, offsets[key], int(csums[i]), low, high, error, gradient]
                    done_tiles += n * n
                    if progress_cb is not None:
                        progress_cb(done_tiles, count, f"Baking Elevation LOD {depth}/{level} (face {face+1}/6)")

        manifest = dict(version=1, encoding="float32-le", reconstruction=reconstruction, grid_cells=cells,
                        gutter=gutter, height_units="m", vertical_datum="ellipsoid", projection="engine-oblate-cube-v1",
                        scale=1, offset=0, coverage="global-filled",
                        orientation="x-east-y-north-z-prime", latitude="geodetic", missing_policy="fill",
                        name=name, equatorial_radius_m=radius, polar_ratio=polar, prime_meridian_rad=prime,
                        encoding_error_m=encoding_error, data_level=level, provenance=provenance,
                        tiles=[records[key] for key in offsets])
        if isinstance(manifest["provenance"].get("nodata"), float) and not math.isfinite(manifest["provenance"]["nodata"]):
            manifest["provenance"]["nodata"] = str(manifest["provenance"]["nodata"])
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        with payload_path.open("rb") as payload:
            for chunk in iter(lambda: payload.read(1024*1024), b""):
                digest.update(chunk)
        manifest["identity"] = int.from_bytes(digest.digest()[:8], "little") or 1
        manifest["revision"] = 1
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        os.rename(temporary, output)
    return manifest


def import_tiles(output, level, radius, polar, prime, sampler, provenance, name,
                 cells=CELLS, gutter=1, reconstruction="nested-triangles", raster=None, cuda_sampler=None,
                 progress_cb=None):
    if cells != CELLS or gutter != 1 or reconstruction != "nested-triangles" or raster is not None:
        return import_dense_tiles(output, level, radius, polar, prime, sampler, provenance, name,
                                  cells=cells, gutter=gutter, reconstruction=reconstruction,
                                  raster=raster, cuda_sampler=cuda_sampler, progress_cb=progress_cb)
    count = 2 * (4**(level+1)-1)
    if not 0 <= level <= 5 or count > MAX_TILES:
        raise ValueError("Version 1 supports data levels 0..5 and at most 16,384 metadata records")
    if not (math.isfinite(radius) and radius > 0 and .05 <= polar <= 1 and math.isfinite(prime)):
        raise ValueError("Invalid reference ellipsoid or prime meridian")
    output = Path(output)
    if output.exists():
        raise ValueError("Output must be a new directory")
    finest = CELLS * (1 << level)
    encoding_error = 0.0

    @functools.lru_cache(maxsize=65536)
    def vertex(face, x, y):
        nonlocal encoding_error
        p = cube(face, -1+2*x/finest, -1+2*y/finest)
        canonical, u, v = face_uv(p)
        if canonical != face:
            return vertex(canonical, round((u+1)*finest/2), round((v+1)*finest/2))
        length = math.sqrt(sum(a*a for a in p))
        direction = tuple(a/length for a in p)
        latitude = math.degrees(math.atan2(direction[1]/polar, math.hypot(direction[0], direction[2])))
        longitude = math.degrees(math.atan2(direction[0], direction[2]) + prime)
        original = sampler(longitude, latitude, direction)
        if not math.isfinite(original) or abs(original) >= radius * .01:
            raise ValueError("Height is nonfinite or exceeds the supported one-percent relief envelope")
        encoded = struct.unpack("<f", struct.pack("<f", original))[0]
        encoding_error = max(encoding_error, abs(encoded-original))
        return encoded

    def continuous(face, u, v):
        canonical, u, v = face_uv(cube(face, u, v))
        x, y = (u+1)*finest/2, (v+1)*finest/2
        ix, iy = min(int(x), finest-1), min(int(y), finest-1)
        a, b, c, d = (vertex(canonical, ix+dx, iy+dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)))
        fx, fy = x-ix, y-iy
        return a*(1-fx-fy)+b*fx+c*fy if fx+fy <= 1 else d*(fx+fy-1)+b*(1-fy)+c*(1-fx)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terrain-import-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        records = {}
        offsets = {}
        offset = 0
        for depth in range(level+1):
            for face in range(6):
                for y in range(1 << depth):
                    for x in range(1 << depth):
                        offsets[face, depth, x, y] = offset
                        offset += SIDE*SIDE*4
        payload_path = temporary / "heights.bin"
        with payload_path.open("w+b") as payload:
            payload.truncate(offset)
            for depth in range(level, -1, -1):
                width = 2/(1 << depth)
                for face in range(6):
                    for y in range(1 << depth):
                        for x in range(1 << depth):
                            values = [continuous(face, -1+(x+gx/CELLS)*width, -1+(y+gy/CELLS)*width)
                                      for gy in range(-1, CELLS+2) for gx in range(-1, CELLS+2)]
                            core = [values[(gy+1)*SIDE+gx+1] for gy in range(CELLS+1) for gx in range(CELLS+1)]
                            low, high, error, gradient = min(core), max(core), 0.0, 0.0
                            for gy in range(CELLS):
                                for gx in range(CELLS):
                                    a, b, c, d = (values[(gy+dy+1)*SIDE+gx+dx+1] for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)))
                                    gradient = max(gradient, math.hypot(b-a, c-a), math.hypot(d-c, d-b))
                            gradient *= CELLS/width
                            if depth < level:
                                for cy in range(2):
                                    for cx in range(2):
                                        child_key = face, depth+1, x*2+cx, y*2+cy
                                        child = records[child_key]
                                        payload.seek(child[4])
                                        heights = struct.unpack("<361f", payload.read(SIDE*SIDE*4))
                                        residual = max(abs(heights[(gy+1)*SIDE+gx+1] - triangle(values, (cx*CELLS+gx)/2, (cy*CELLS+gy)/2))
                                                       for gy in range(CELLS+1) for gx in range(CELLS+1))
                                        low, high = min(low, child[6]), max(high, child[7])
                                        error = max(error, residual+child[8])
                                        gradient = max(gradient, child[9])
                            encoded = struct.pack("<361f", *values)
                            key = face, depth, x, y
                            payload.seek(offsets[key])
                            payload.write(encoded)
                            records[key] = [*key, offsets[key], checksum(encoded), low, high, error, gradient]
        manifest = dict(version=1, encoding="float32-le", reconstruction="nested-triangles", grid_cells=CELLS,
                        gutter=1, height_units="m", vertical_datum="ellipsoid", projection="engine-oblate-cube-v1",
                        scale=1, offset=0, coverage="global-filled",
                        orientation="x-east-y-north-z-prime", latitude="geodetic", missing_policy="fill",
                        name=name, equatorial_radius_m=radius, polar_ratio=polar, prime_meridian_rad=prime,
                        encoding_error_m=encoding_error, data_level=level, provenance=provenance,
                        tiles=[records[key] for key in offsets])
        if isinstance(manifest["provenance"].get("nodata"), float) and not math.isfinite(manifest["provenance"]["nodata"]):
            manifest["provenance"]["nodata"] = str(manifest["provenance"]["nodata"])
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        with payload_path.open("rb") as payload:
            for chunk in iter(lambda: payload.read(1024*1024), b""):
                digest.update(chunk)
        manifest["identity"] = int.from_bytes(digest.digest()[:8], "little") or 1
        manifest["revision"] = 1
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        os.rename(temporary, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build a bounded, nested FP32 terrain pyramid without runtime geospatial dependencies")
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path, help="Geographic JSON point lattice; see docs/terrain.md")
    parser.add_argument("--gdal", type=Path, help="Single-band GDAL raster with already converted ellipsoidal heights")
    parser.add_argument("--vertical-datum", choices=["ellipsoid"], help="Required explicit declaration for GDAL input")
    parser.add_argument("--height-unit", choices=["m", "ft"], help="Override missing GDAL band unit metadata")
    parser.add_argument("--fill-m", type=float, default=0)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--radius-m", type=float, default=6378137)
    parser.add_argument("--polar-ratio", type=float, default=1-1/298.257223563)
    parser.add_argument("--prime-meridian-deg", type=float, default=0)
    args = parser.parse_args()
    if args.input and args.gdal:
        parser.error("Choose either --input or --gdal")
    if args.gdal:
        if args.vertical_datum != "ellipsoid":
            parser.error("GDAL input requires --vertical-datum ellipsoid; convert orthometric heights with geoid grids first")
        raster = GdalRaster(args.gdal, args.radius_m, args.polar_ratio, args.height_unit, args.fill_m)
        sampler = lambda lon, lat, direction: raster.sample(lon, lat)
        provenance = raster.provenance
        name = args.gdal.stem
    elif args.input:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        raster = Raster(document)
        sampler = lambda lon, lat, direction: raster.sample(lon, lat)
        provenance = {key: value for key, value in document.items() if key != "values"}
        provenance["source_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        name = args.input.stem
    else:
        sampler = lambda lon, lat, d: -600 + 1800*d[0]*d[2] + 350*math.sin(7*d[1])*math.cos(5*d[0])
        provenance = {"fixture": "signed-height-v1", "coverage": "global"}
        name = "Signed height fixture"
    manifest = import_tiles(args.output, args.level, args.radius_m, args.polar_ratio,
                            math.radians(args.prime_meridian_deg), sampler, provenance, name)
    print(f"Imported {len(manifest['tiles'])} tiles, identity {manifest['identity']}, encoding error {manifest['encoding_error_m']:.6g} m")


if __name__ == "__main__":
    main()
