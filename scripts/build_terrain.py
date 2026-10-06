#!/usr/bin/env python3
"""Build/install the terrain extension into the launching Python environment."""
import argparse
import importlib.util
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'native' / 'terrain'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--if-needed', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.find_spec('stellar_terrain')
    if args.if_needed and spec and spec.origin:
        module = pathlib.Path(spec.origin)
        sources = [NATIVE / 'Cargo.toml', NATIVE / 'Cargo.lock', NATIVE / 'pyproject.toml', *NATIVE.glob('src/*.rs')]
        if all(not p.exists() or p.stat().st_mtime <= module.stat().st_mtime for p in sources):
            return
    if not shutil.which('cargo'):
        raise SystemExit('Rust is required to build terrain. Install a stable Rust toolchain from https://rustup.rs, then rerun this launcher.')
    if not importlib.util.find_spec('maturin'):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'maturin>=1.9,<2'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '--force-reinstall', str(NATIVE)])


if __name__ == '__main__':
    main()
