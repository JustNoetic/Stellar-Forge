import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ESSENTIAL_DATA_FILES = [
    "clouds.json",
    "ephemeris.json",
    "ephemeris_settings.json",
    "surfaces.json",
    "system.json",
    "gaia/README.md",
    "gaia/fetch_log.txt",
    "gaia/stars.bin",
    "optics/ch4_karkoschka_1994.txt",
    "kernels/cspice.dll",
    "kernels/SpiceyPy-LICENSE.txt",
    "kernels/additional/artemis2.bsp",
]

ESSENTIAL_TEXTURE_FILES = [
    "noise/stbn_scalar.bin",
    "Solar System/source_manifest.json",
    "Solar System/Saturn/Rings.png",
]


def detect_version() -> str:
    cmake_path = ROOT / "CMakeLists.txt"
    if cmake_path.is_file():
        m = re.search(r"project\s*\(\s*StellarForge\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
                      cmake_path.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return "2.0.0"


def build_release(build_dir: Path) -> Path:
    print("[1/3] Configuring & building StellarForge (Release)...")
    subprocess.run(
        ["cmake", "-B", str(build_dir), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "StellarForge", "--parallel"],
        cwd=ROOT,
        check=True,
    )
    exe = build_dir / "bin" / "StellarForge.exe"
    if not exe.is_file():
        exe = build_dir / "StellarForge.exe"
    if not exe.is_file():
        raise FileNotFoundError(f"Built executable not found in {build_dir}")
    return exe


def copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def stage_essentials(exe_path: Path, stage_dir: Path) -> tuple[int, int]:
    print(f"[2/3] Staging essential release files to {stage_dir.relative_to(ROOT)}...")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    total_bytes = 0

    def add(src: Path, rel_dest: str | Path):
        nonlocal file_count, total_bytes
        if src.is_file():
            total_bytes += copy_file(src, stage_dir / rel_dest)
            file_count += 1

    # 1. Core binary & root scripts/docs
    add(exe_path, "StellarForge.exe")
    add(ROOT / "bake_textures.bat", "bake_textures.bat")
    add(ROOT / "README.md", "README.md")

    # 2. Shaders (excluding test shaders)
    shaders_root = ROOT / "shaders"
    for p in sorted(shaders_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(shaders_root)
        if rel.parts[0] == "tests":
            continue
        add(p, Path("shaders") / rel)

    # 3. Python helper scripts (excluding __pycache__)
    scripts_root = ROOT / "scripts"
    for p in sorted(scripts_root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        add(p, Path("scripts") / p.relative_to(scripts_root))

    # 4. Essential data files & saved system presets (omitting kernels/default and tiles)
    for rel_str in ESSENTIAL_DATA_FILES:
        add(ROOT / "data" / rel_str, Path("data") / rel_str)

    systems_root = ROOT / "data" / "systems"
    if systems_root.is_dir():
        for p in sorted(systems_root.rglob("*")):
            if p.is_file():
                add(p, Path("data/systems") / p.relative_to(systems_root))

    # Create empty runtime directories so structure is ready out of the box
    (stage_dir / "data" / "kernels" / "default").mkdir(parents=True, exist_ok=True)
    (stage_dir / "data" / "tiles").mkdir(parents=True, exist_ok=True)

    # 5. Essential small textures (omitting heavy planetary source maps & baked cloud packs)
    for rel_str in ESSENTIAL_TEXTURE_FILES:
        add(ROOT / "textures" / rel_str, Path("textures") / rel_str)

    return file_count, total_bytes


def create_zip(source_dir: Path, zip_path: Path, arc_prefix: str = "StellarForge") -> int:
    print(f"[3/3] Compressing release archive {zip_path.relative_to(ROOT)}...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    all_files = [p for p in sorted(source_dir.rglob("*")) if p.is_file()]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in all_files:
            arcname = Path(arc_prefix) / p.relative_to(source_dir)
            zf.write(p, arcname.as_posix())

    return zip_path.stat().st_size


def create_source_textures_zip(zip_path: Path) -> tuple[int, int]:
    tex_root = ROOT / "textures" / "Solar System"
    if not tex_root.is_dir():
        raise FileNotFoundError(f"Source texture directory not found: {tex_root}")

    print(f"[Optional] Packaging Solar System source textures into {zip_path.relative_to(ROOT)}...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(tex_root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(tex_root)
            # Exclude pre-baked cloud coverage.r8 (regenerated via bake_textures.bat)
            if "clouds" in rel.parts:
                continue
            arcname = Path("textures/Solar System") / rel
            zf.write(p, arcname.as_posix())
            count += 1

    return count, zip_path.stat().st_size


def main() -> int:
    version = detect_version()
    parser = argparse.ArgumentParser(
        description="Build and package Stellar Forge essentials into a clean release .zip (omitting NAIF kernels and large textures)."
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build", help="CMake build directory (default: build)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist", help="Output directory for staged folder and zip (default: dist)")
    parser.add_argument("--name", type=str, default=f"StellarForge-v{version}-win64", help="Release archive base name")
    parser.add_argument("--skip-build", action="store_true", help="Package existing build/bin/StellarForge.exe without running CMake build")
    parser.add_argument("--include-textures-zip", action="store_true", help="Also create a separate zip of raw source textures for GitHub Releases")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    out_dir = args.out_dir.resolve()

    if args.skip_build:
        exe_path = build_dir / "bin" / "StellarForge.exe"
        if not exe_path.is_file():
            exe_path = build_dir / "StellarForge.exe"
        if not exe_path.is_file():
            raise FileNotFoundError(f"Executable not found at {exe_path}. Run without --skip-build first.")
        print(f"[1/3] Using existing executable: {exe_path.relative_to(ROOT)}")
    else:
        exe_path = build_release(build_dir)

    stage_dir = out_dir / "StellarForge"
    file_count, raw_bytes = stage_essentials(exe_path, stage_dir)

    zip_path = out_dir / f"{args.name}.zip"
    zip_bytes = create_zip(stage_dir, zip_path, arc_prefix="StellarForge")

    raw_mb = raw_bytes / (1024 * 1024)
    zip_mb = zip_bytes / (1024 * 1024)
    print("-" * 68)
    print(f"Release Package Ready:")
    print(f"  Staged Folder : {stage_dir} ({file_count} files, {raw_mb:.2f} MB uncompressed)")
    print(f"  Release Zip   : {zip_path} ({zip_mb:.2f} MB compressed)")
    print("  Excluded      : data/kernels/default/* (~2.1 GB), data/tiles/* (~6.9 GB), large source textures (~702 MB)")

    if args.include_textures_zip:
        tex_zip = out_dir / f"StellarForge-v{version}-SolarSystem-SourceTextures.zip"
        tex_count, tex_bytes = create_source_textures_zip(tex_zip)
        print(f"  Textures Zip  : {tex_zip} ({tex_count} files, {tex_bytes / (1024 * 1024):.2f} MB)")
    print("-" * 68)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
