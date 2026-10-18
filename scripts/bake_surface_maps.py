import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def recommended_lod(image_path):
    with Image.open(image_path) as image:
        return min(5, max(0, math.ceil(math.log2(max(1, image.width / (CORE * 4))))))


def bake_color(source_path, output, level=3, polar=1, prime=0, latitude="geodetic",
               backend="auto", sampler=None, rebake=False):
    output = Path(output)
    if not 0 <= level <= 5 or not .05 <= polar <= 1 or not math.isfinite(prime):
        raise ValueError("Albedo requires data LOD 0..5 and a valid oblate reference shape")
    if latitude not in ("geodetic", "planetocentric", "parametric"):
        raise ValueError("Unknown source latitude convention")
    provenance = dict(source_sha256=file_hash(source_path), polar_ratio=polar, prime_meridian_rad=prime,
                      latitude=latitude, data_level=level, core=CORE, gutter=GUTTER, version=1)
    if output.exists() and not rebake:
        existing = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        if existing.get("provenance") != provenance:
            raise ValueError("Existing albedo dataset differs; choose a new output directory or pass --rebake")
        return existing
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"))
    pyramid = [source]
    while pyramid[-1].shape[1] > CORE*4 and min(pyramid[-1].shape[:2]) > 1:
        pyramid.append(reduce_linear(pyramid[-1]))

    use_cuda = False
    if backend in ("auto", "cuda"):
        if sampler is not None:
            use_cuda = True
        else:
            try:
                from scripts.cuda_tile_sampler import is_cuda_available, CudaTileSampler
                cuda_ok, _ = is_cuda_available()
                if cuda_ok:
                    sampler = CudaTileSampler(DECODE, ENCODE)
                    use_cuda = True
                elif backend == "cuda":
                    raise RuntimeError("CUDA backend requested, but CUDA is unavailable.")
            except Exception as exc:
                if backend == "cuda":
                    raise RuntimeError(f"CUDA backend requested, but failed: {exc}") from exc
                use_cuda = False

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="albedo-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        checksums = []
        with (temporary/"tiles.rgba").open("wb") as payload:
            for depth in range(level+1):
                mip = min(len(pyramid)-1, max(0, int(math.log2(max(1, source.shape[1]/(CORE*4*(1 << depth)))))))
                source_mip = pyramid[mip]
                if use_cuda and sampler is not None:
                    sampler.set_source(source_mip)
                    tiles = sampler.sample_level(depth, polar, prime, latitude)
                    for tile in tiles:
                        encoded = tile.tobytes()
                        payload.write(encoded)
                        checksums.append(zlib.crc32(encoded))
                else:
                    for face in range(6):
                        for ty in range(1 << depth):
                            for tx in range(1 << depth):
                                u, v = coordinates(face, depth, tx, ty, polar, prime, latitude)
                                rgba = np.full((SIDE, SIDE, 4), 255, dtype=np.uint8)
                                rgba[:, :, :3] = encode(sample(source_mip, u, v))
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
        if output.exists():
            shutil.rmtree(output)
        os.rename(temporary, output)
    return manifest


def discover_available_bodies(root, definitions):
    items = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name not in definitions:
            continue
        source = next((path for path in sorted(folder.iterdir()) if path.stem.casefold() == folder.name.casefold()
                       and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")), None)
        if source is not None:
            height = folder/(folder.name+"_heightmap.png")
            has_height = height.exists() and "height_range_km" in definitions[folder.name]
            items.append({
                "body": folder.name,
                "folder": folder,
                "source": source,
                "has_height": has_height,
                "height_path": height if has_height else None,
                "definition": definitions[folder.name]
            })
    return items


def choose_indices(value, count):
    selected = set()
    for token in value.replace(',', ' ').split():
        parts = token.split('-')
        if len(parts) > 2:
            raise ValueError('Use numbers or ranges such as 1,3-5.')
        start, end = int(parts[0]), int(parts[-1])
        if not 1 <= start <= end <= count:
            raise ValueError(f'Choose numbers from 1 to {count}.')
        selected.update(range(start - 1, end))
    if not selected:
        raise ValueError('Choose at least one item.')
    return sorted(selected)


def ask_int(label, default, minimum, maximum):
    while True:
        try:
            val = input(f'{label} [{default}]: ').strip()
            value = int(val or default)
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f'Enter a whole number from {minimum} to {maximum}.')


def interactive_wizard(items, backend_str):
    print("\nStellar Forge Surface Texture Baker (v1 oblate-cube-sphere tiles)")
    print(f"Backend detected: {backend_str}")
    print("-" * 72)
    for idx, item in enumerate(items, 1):
        with Image.open(item["source"]) as img:
            w, h = img.size
        rec = recommended_lod(item["source"])
        h_str = " [+Elevation]" if item["has_height"] else ""
        print(f"  {idx:2}. {item['body']:12} {w:5}x{h:<5} (Recommended LOD {rec}){h_str}")
    print("-" * 72)

    while True:
        choice = input("\n[A]ll bodies, body numbers/ranges (e.g. 1,4-6), or [Q]uit: ").strip().lower()
        if choice in ('q', 'quit'):
            return None, None, False
        if choice == 'a':
            selected = items
            break
        try:
            indices = choose_indices(choice, len(items))
            selected = [items[i] for i in indices]
            break
        except ValueError as exc:
            print(exc)

    lod_choice = input("LOD: [A]utomatic per source (recommended), or [C]ustom level (0..5)? [A]: ").strip().lower()
    custom_lod = ask_int("Custom LOD", 3, 0, 5) if lod_choice == 'c' else None

    rebake_choice = input("Existing packs: [S]kip or [R]ebake? [S]: ").strip().lower()
    rebake = (rebake_choice == 'r')

    jobs = [(item, custom_lod if custom_lod is not None else recommended_lod(item["source"])) for item in selected]
    total_tiles = sum(2 * (4**(lod + 1) - 1) for _, lod in jobs)
    uncomp_mb = total_tiles * (SIDE * SIDE * 4) / (1024 * 1024)

    print("\nBake Plan:")
    for item, lod in jobs:
        print(f"  - {item['body']:12} -> LOD 0..{lod} ({2*(4**(lod+1)-1)} tiles)")
    print(f"Total: {len(jobs)} bodies, {total_tiles:,} tiles (~{uncomp_mb:.1f} MB uncompressed).")

    confirm = input("Start baking? [Y/n]: ").strip().lower()
    if confirm in ('n', 'no'):
        print("Cancelled.")
        return None, None, False

    return jobs, rebake, True


def bake_library(root, output_root, catalog_path, level=None, selected_names=None,
                 backend="auto", rebake=False, is_interactive=False):
    from import_terrain import ImageRaster, import_tiles
    root, output_root = Path(root), Path(output_root)
    project = Path(__file__).resolve().parents[1]
    definitions = {body["name"]: body for body in json.loads((project/"data/system.json").read_text())}
    constants = (project/"src/core/constants.hpp").read_text()
    solar_radius_m = float(re.search(r"SOLAR_RADIUS_KM\s*=\s*([\d.]+)", constants).group(1))*1000

    items = discover_available_bodies(root, definitions)
    if not items:
        print(f"No source textures found in {root}.")
        return 1

    # Device detection
    sampler = None
    cuda_name = None
    if backend in ("auto", "cuda"):
        try:
            from scripts.cuda_tile_sampler import is_cuda_available, CudaTileSampler
            ok, name = is_cuda_available()
            if ok:
                cuda_name = name
                sampler = CudaTileSampler(DECODE, ENCODE)
            elif backend == "cuda":
                raise RuntimeError(f"CUDA required but unavailable: {name}")
        except Exception as exc:
            if backend == "cuda":
                raise RuntimeError(f"CUDA initialization failed: {exc}") from exc
            sampler = None

    backend_str = f"CUDA ({cuda_name})" if sampler is not None else "CPU (NumPy)"

    if is_interactive:
        jobs, rebake_flag, ok = interactive_wizard(items, backend_str)
        if not ok or not jobs:
            return 0
        rebake = rebake_flag
    else:
        if selected_names:
            items = [item for item in items if item["body"] in selected_names]
        target_lod = level if level is not None else 3
        jobs = [(item, target_lod) for item in items]

    print(f"\nProcessing {len(jobs)} bodies on {backend_str}...")
    existing = json.loads(catalog_path.read_text()) if catalog_path.exists() else {"version": 1, "bodies": []}
    existing_map = {item["body"]: item for item in existing.get("bodies", [])}

    for item, lod in jobs:
        body = item["definition"]
        polar = 1 - float(body.get("oblateness", 0))
        fingerprint = hashlib.sha256(f"{file_hash(item['source'])}:{polar}:{lod}:geodetic-v1".encode()).hexdigest()[:12]
        dataset = output_root/item["body"].lower()/fingerprint
        bake_color(item["source"], dataset/"albedo", lod, polar, backend=backend, sampler=sampler, rebake=rebake)

        binding = dict(body=item["body"], albedo=(dataset/"albedo").resolve().relative_to(project).as_posix())

        if item["has_height"]:
            low, high = (float(v)*1000 for v in body["height_range_km"])
            radius_m = float(body["r"])*solar_radius_m
            height_tag = hashlib.sha256(f"{file_hash(item['height_path'])}:{polar}:{radius_m}:{low}:{high}:5".encode()).hexdigest()[:12]
            height_output = output_root/item["body"].lower()/height_tag/"height"
            if not height_output.exists() or rebake:
                if height_output.exists():
                    shutil.rmtree(height_output)
                raster = ImageRaster(item["height_path"], (high-low)/255, low)
                import_tiles(height_output, 5, radius_m, polar, 0,
                    lambda lon, lat, direction: raster.sample(lon, lat),
                    dict(source_sha256=file_hash(item["height_path"]), scale=(high-low)/255, offset=low,
                         interpretation="legacy-visual-range-no-geoid-correction", source_height_range_m=[low, high]),
                    item["body"]+" legacy visual elevation")
            binding["height"] = height_output.resolve().relative_to(project).as_posix()

        existing_map[item["body"]] = binding
        print(f"  [OK] {item['body']}: albedo LOD 0..{lod}" + (" + elevation" if "height" in binding else ""), flush=True)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(dict(version=1, bodies=list(existing_map.values())), indent=2)+"\n", encoding="utf-8")
    print(f"\nCatalog updated: {catalog_path}")
    return 0


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Bake source color maps into linear-filtered sRGB cube tile packs")
    parser.add_argument("--source-root", type=Path, default=project/"textures/Solar System")
    parser.add_argument("--output-root", type=Path, default=project/"data/tiles")
    parser.add_argument("--catalog", type=Path, default=project/"data/surfaces.json")
    parser.add_argument("--level", type=int, default=None, help="Target data LOD (0..5). Defaults to 3 in batch mode.")
    parser.add_argument("--body", action="append", help="Specific body name(s) to process. Can be repeated.")
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto", help="Compute backend (default: auto)")
    parser.add_argument("--rebake", action="store_true", help="Force rebaking even if tile pack exists")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive wizard")
    parser.add_argument("--list", action="store_true", help="List detected maps and exit")

    args = parser.parse_args()

    # Determine if we should run interactively
    no_explicit_action = (len(sys.argv) <= 1)
    is_interactive = args.interactive or (no_explicit_action and sys.stdin.isatty())

    if args.list:
        definitions = {body["name"]: body for body in json.loads((project/"data/system.json").read_text())}
        items = discover_available_bodies(args.source_root, definitions)
        print(f"Found {len(items)} bodies in {args.source_root}:")
        for idx, item in enumerate(items, 1):
            rec = recommended_lod(item["source"])
            print(f"  {idx:2}. {item['body']:12} (Recommended LOD {rec})" + (" [+Height]" if item["has_height"] else ""))
        return 0

    return bake_library(
        args.source_root,
        args.output_root,
        args.catalog,
        level=args.level,
        selected_names=set(args.body or []) if args.body else None,
        backend=args.backend,
        rebake=args.rebake,
        is_interactive=is_interactive
    )


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        sys.exit(1)
