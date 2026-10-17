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
    def __init__(self, path, scale, offset):
        import numpy as np
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 268435456
        with Image.open(path) as image:
            self.values = np.asarray(image if image.mode in ("I;16", "I", "F", "L") else image.convert("L"))
        self.height, self.width = self.values.shape
        self.scale, self.offset = scale, offset
        if not math.isfinite(scale) or not math.isfinite(offset):
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


def import_tiles(output, level, radius, polar, prime, sampler, provenance, name):
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
