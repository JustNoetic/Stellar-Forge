import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import tempfile
import zlib

import numpy as np
from PIL import Image


CORE = 128
GUTTER = 2
SIDE = CORE + 2*GUTTER
Image.MAX_IMAGE_PIXELS = 268435456
DECODE = np.where(np.arange(256)/255 <= .04045, np.arange(256)/255/12.92,
                  ((np.arange(256)/255+.055)/1.055)**2.4).astype(np.float32)
LINEAR = np.arange(65536)/65535
ENCODE = np.rint(np.where(LINEAR <= .0031308, LINEAR*12.92, 1.055*LINEAR**(1/2.4)-.055)*255).astype(np.uint8)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(values):
    return ENCODE[np.rint(np.clip(values, 0, 1)*65535).astype(np.uint16)]


def reduce_linear(source):
    height, width = source.shape[:2]
    result = np.empty(((height+1)//2, (width+1)//2, 3), dtype=np.uint8)
    x = np.arange(result.shape[1])*2
    for start in range(0, result.shape[0], 64):
        y = np.arange(start, min(start+64, result.shape[0]))*2
        a = DECODE[source[y[:, None], x]]
        b = DECODE[source[y[:, None], np.minimum(x+1, width-1)]]
        c = DECODE[source[np.minimum(y+1, height-1)[:, None], x]]
        d = DECODE[source[np.minimum(y+1, height-1)[:, None], np.minimum(x+1, width-1)]]
        result[start:start+len(y)] = encode((a+b+c+d)*.25)
    return result


def coordinates(face, level, tx, ty, polar, prime, latitude):
    axis = (np.arange(SIDE, dtype=np.float64)-GUTTER+.5)/CORE
    u, v = np.meshgrid(-1+2*(tx+axis)/(1 << level), -1+2*(ty+axis)/(1 << level))
    one = np.ones_like(u)
    x, y, z = ((one, v, -u), (-one, v, u), (u, one, -v), (u, -one, v), (u, v, one), (-u, v, -one))[face]
    y = y/polar if latitude == "geodetic" else y*polar if latitude == "planetocentric" else y
    return (np.arctan2(x, z)+prime)/(2*math.pi)+.5, .5-np.arctan2(y, np.hypot(x, z))/math.pi


def sample(source, u, v):
    height, width = source.shape[:2]
    x, y = (u % 1)*width-.5, np.clip(v*height-.5, 0, height-1)
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x-ix)[..., None], (y-iy)[..., None]
    nx, ny = (ix+1) % width, np.minimum(iy+1, height-1)
    ix %= width
    return ((DECODE[source[iy, ix]]*(1-fx)+DECODE[source[iy, nx]]*fx)*(1-fy)
            +(DECODE[source[ny, ix]]*(1-fx)+DECODE[source[ny, nx]]*fx)*fy)


def bake_color(source_path, output, level=3, polar=1, prime=0, latitude="geodetic"):
    output = Path(output)
    if not 0 <= level <= 5 or not .05 <= polar <= 1 or not math.isfinite(prime):
        raise ValueError("Albedo requires data LOD 0..5 and a valid oblate reference shape")
    if latitude not in ("geodetic", "planetocentric", "parametric"):
        raise ValueError("Unknown source latitude convention")
    provenance = dict(source_sha256=file_hash(source_path), polar_ratio=polar, prime_meridian_rad=prime,
                      latitude=latitude, data_level=level, core=CORE, gutter=GUTTER, version=1)
    if output.exists():
        existing = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        if existing.get("provenance") != provenance:
            raise ValueError("Existing albedo dataset differs; choose a new output directory")
        return existing
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"))
    pyramid = [source]
    while pyramid[-1].shape[1] > CORE*4 and min(pyramid[-1].shape[:2]) > 1:
        pyramid.append(reduce_linear(pyramid[-1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="albedo-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        checksums = []
        with (temporary/"tiles.rgba").open("wb") as payload:
            for depth in range(level+1):
                mip = min(len(pyramid)-1, max(0, int(math.log2(max(1, source.shape[1]/(CORE*4*(1 << depth)))))))
                for face in range(6):
                    for ty in range(1 << depth):
                        for tx in range(1 << depth):
                            u, v = coordinates(face, depth, tx, ty, polar, prime, latitude)
                            rgba = np.full((SIDE, SIDE, 4), 255, dtype=np.uint8)
                            rgba[:, :, :3] = encode(sample(pyramid[mip], u, v))
                            encoded = rgba.tobytes()
                            payload.write(encoded)
                            checksums.append(zlib.crc32(encoded))
        manifest = dict(version=1, encoding="rgba8-srgb", projection="engine-oblate-cube-v1",
                        orientation="x-east-y-north-z-prime", data_level=level, core=CORE, gutter=GUTTER,
                        polar_ratio=polar, prime_meridian_rad=prime, latitude=latitude,
                        provenance=provenance, checksums=checksums)
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).digest()
        manifest.update(identity=int.from_bytes(digest[:8], "little") or 1, revision=1)
        (temporary/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        os.rename(temporary, output)
    return manifest


def bake_library(root, output_root, catalog_path, level, selected):
    from import_terrain import ImageRaster, import_tiles
    root, output_root = Path(root), Path(output_root)
    project = Path(__file__).resolve().parents[1]
    definitions = {body["name"]: body for body in json.loads((project/"data/system.json").read_text())}
    constants = (project/"src/core/constants.hpp").read_text()
    solar_radius_m = float(re.search(r"SOLAR_RADIUS_KM\s*=\s*([\d.]+)", constants).group(1))*1000
    bindings = []
    existing = json.loads(catalog_path.read_text()) if catalog_path.exists() else {"version": 1, "bodies": []}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or (selected and folder.name not in selected):
            continue
        source = next((path for path in sorted(folder.iterdir()) if path.stem.casefold() == folder.name.casefold()
                       and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")), None)
        if source is None or folder.name not in definitions:
            continue
        body = definitions[folder.name]
        polar = 1-float(body.get("oblateness", 0))
        fingerprint = hashlib.sha256(f"{file_hash(source)}:{polar}:{level}:geodetic-v1".encode()).hexdigest()[:12]
        dataset = output_root/folder.name.lower()/fingerprint
        bake_color(source, dataset/"albedo", level, polar)
        binding = dict(body=folder.name, albedo=(dataset/"albedo").resolve().relative_to(project).as_posix())
        height = folder/(folder.name+"_heightmap.png")
        if height.exists() and "height_range_km" in body:
            low, high = (float(value)*1000 for value in body["height_range_km"])
            radius_m = float(body["r"])*solar_radius_m
            height_tag = hashlib.sha256(f"{file_hash(height)}:{polar}:{radius_m}:{low}:{high}:5".encode()).hexdigest()[:12]
            height_output = output_root/folder.name.lower()/height_tag/"height"
            if not height_output.exists():
                raster = ImageRaster(height, (high-low)/255, low)
                import_tiles(height_output, 5, radius_m, polar, 0,
                    lambda lon, lat, direction: raster.sample(lon, lat),
                    dict(source_sha256=file_hash(height), scale=(high-low)/255, offset=low,
                         interpretation="legacy-visual-range-no-geoid-correction", source_height_range_m=[low, high]),
                    folder.name+" legacy visual elevation")
            binding["height"] = height_output.resolve().relative_to(project).as_posix()
        bindings.append(binding)
        print(f"{folder.name}: baked albedo LOD0..{level}"+(" and elevation" if "height" in binding else ""), flush=True)
    if selected:
        bindings.extend(item for item in existing["bodies"] if item["body"] not in selected)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(dict(version=1, bodies=bindings), indent=2)+"\n", encoding="utf-8")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Bake source color maps into linear-filtered sRGB cube tile packs")
    parser.add_argument("--source-root", type=Path, default=project/"textures/Solar System")
    parser.add_argument("--output-root", type=Path, default=project/"data/tiles")
    parser.add_argument("--catalog", type=Path, default=project/"data/surfaces.json")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--body", action="append")
    args = parser.parse_args()
    bake_library(args.source_root, args.output_root, args.catalog, args.level, set(args.body or []))


if __name__ == "__main__":
    main()
