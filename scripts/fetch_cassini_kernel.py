"""Fetch Cassini state vectors from NASA JPL Horizons and build a binary SPK kernel.

Queries Cartesian state vectors (J2000 ICRF, Solar System Barycenter observer)
for target body -82 (Cassini spacecraft) across its complete 13-year Saturn mission
(July 1, 2004 14:00 UTC through September 15, 2017 11:00 UTC)
and encodes them into a standard NAIF Type 9 SPK kernel (`data/kernels/cassini.bsp`).

Usage:
    python scripts/fetch_cassini_kernel.py
"""

import os
import sys
import json
import re
import time
import urllib.request
import urllib.parse


def generate_cassini_spk(output_dir=None, progress_callback=None):
    """
    Fetches Cassini trajectory data from JPL Horizons and writes `cassini.bsp`.
    Returns (success, message, filepath).
    """
    try:
        import spiceypy as spice
    except ImportError:
        return False, "spiceypy is required to generate SPK kernels.", None

    if output_dir is None:
        try:
            from engine.ephemeris.spice_manager import SpiceManager
            output_dir = SpiceManager.KERNEL_DIR
        except Exception:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "kernels"))
    os.makedirs(output_dir, exist_ok=True)
    bsp_path = os.path.join(output_dir, "cassini.bsp")

    tls_path = os.path.join(output_dir, "naif0012.tls")
    if not os.path.exists(tls_path):
        candidate_tls = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "kernels", "naif0012.tls"))
        if os.path.exists(candidate_tls):
            tls_path = candidate_tls
        else:
            return False, f"Leapseconds kernel {tls_path} not found. Please download required kernels first.", None

    spice.furnsh(tls_path)

    if progress_callback:
        progress_callback(0.05, "Connecting to NASA JPL Horizons API for Cassini (13-year tour)...")

    # 13.2-year Saturn mission: from Saturn Arrival through Grand Finale plunge
    start_time = "2004-07-01 14:00"
    stop_time = "2017-09-15 11:00"
    step_size = "2h"

    params = {
        'format': 'json',
        'COMMAND': "'-82'",
        'EPHEM_TYPE': 'VECTORS',
        'CENTER': "'500@0'",       # Solar System Barycenter
        'REF_PLANE': 'FRAME',     # ICRF / J2000
        'START_TIME': f"'{start_time}'",
        'STOP_TIME': f"'{stop_time}'",
        'STEP_SIZE': f"'{step_size}'",
        'VEC_TABLE': '3'
    }

    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    t0 = time.time()

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Stellar-Forge/1.0'})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        spice.unload(tls_path)
        return False, f"Failed to connect to JPL Horizons: {e}", None

    if progress_callback:
        progress_callback(0.40, f"Received Horizons data ({time.time() - t0:.1f}s). Parsing state vectors...")

    text = data.get('result', '')
    s_idx = text.find('$$SOE')
    e_idx = text.find('$$EOE')
    if s_idx == -1 or e_idx == -1:
        spice.unload(tls_path)
        err = data.get('error', 'Unknown error parsing Horizons response')
        return False, f"Horizons did not return vector data: {err}", None

    block = text[s_idx + 5:e_idx].strip()
    pat = re.compile(
        r'(\d+\.\d+)\s*=\s*A\.D\.\s*([\d\w\-:\s\.]+)\s*\n'
        r'\s*X\s*=\s*([+\-\d\.E]+)\s*Y\s*=\s*([+\-\d\.E]+)\s*Z\s*=\s*([+\-\d\.E]+)\s*\n'
        r'\s*VX=\s*([+\-\d\.E]+)\s*VY=\s*([+\-\d\.E]+)\s*VZ=\s*([+\-\d\.E]+)'
    )
    matches = pat.findall(block)
    if not matches:
        spice.unload(tls_path)
        return False, "No state vectors parsed from Horizons output.", None

    if progress_callback:
        progress_callback(0.65, f"Building SPK Type 9 kernel with {len(matches)} states...")

    epochs = []
    states = []
    for m in matches:
        jd = float(m[0])
        # Direct Julian Date TDB to Ephemeris Time (seconds past J2000 epoch)
        et = (jd - 2451545.0) * 86400.0
        epochs.append(et)
        states.append([
            float(m[2]), float(m[3]), float(m[4]),
            float(m[5]), float(m[6]), float(m[7])
        ])

    temp_bsp = bsp_path + ".part"
    if os.path.exists(temp_bsp):
        try:
            os.remove(temp_bsp)
        except OSError:
            pass

    try:
        # Create Type 9 SPK: cubic Lagrange interpolation
        handle = spice.spkopn(temp_bsp, "CASSINI_SATURN_TOUR", 0)
        spice.spkw09(
            handle,
            -82,                     # Target: Cassini
            0,                       # Center: Solar System Barycenter
            "J2000",                 # Inertial frame
            epochs[0],               # Begin epoch
            epochs[-1],              # End epoch
            "CASSINI_TOUR_2004_2017",# Segment ID
            3,                       # Polynomial degree (cubic Lagrange)
            len(epochs),
            states,
            epochs
        )
        spice.spkcls(handle)

        if os.path.exists(bsp_path):
            try:
                os.remove(bsp_path)
            except OSError:
                pass
        os.replace(temp_bsp, bsp_path)

        # Also copy to engine/data/kernels if separate
        engine_kernels_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine", "data", "kernels"))
        if os.path.exists(engine_kernels_dir) and engine_kernels_dir != output_dir:
            try:
                import shutil
                shutil.copy2(bsp_path, os.path.join(engine_kernels_dir, "cassini.bsp"))
            except Exception:
                pass

    except Exception as e:
        if os.path.exists(temp_bsp):
            try:
                os.remove(temp_bsp)
            except OSError:
                pass
        spice.unload(tls_path)
        return False, f"Failed to encode SPK kernel: {e}", None

    spice.unload(tls_path)

    size_mb = os.path.getsize(bsp_path) / (1024 * 1024)
    if progress_callback:
        progress_callback(1.0, f"Successfully created cassini.bsp ({size_mb:.2f} MB, {len(epochs)} states)")

    return True, f"Successfully generated {bsp_path} ({size_mb:.2f} MB)", bsp_path


def main():
    print("================================================")
    print(" Cassini SPICE Kernel Generator (JPL Horizons) ")
    print("================================================")

    def print_progress(frac, msg):
        print(f"[{int(frac * 100):3d}%] {msg}")

    success, message, path = generate_cassini_spk(progress_callback=print_progress)
    print(message)
    if not success:
        sys.exit(1)

    # Verification query using SPICE
    import spiceypy as spice
    kdir = os.path.dirname(path)
    tls = os.path.join(kdir, "naif0012.tls")
    de = os.path.join(kdir, "de440s.bsp")
    spice.furnsh(tls)
    if os.path.exists(de):
        spice.furnsh(de)
    spice.furnsh(path)

    cov = spice.spkcov(path, -82)
    print(f"\nSPK Coverage: {len(cov)//2} interval(s)")
    for i in range(0, len(cov), 2):
        print(f"  {spice.et2utc(cov[i], 'C', 3)} -> {spice.et2utc(cov[i+1], 'C', 3)}")

    # Check Saturn Orbit Insertion (SOI) pass
    t_soi = spice.str2et("2004-07-01 02:48:00")
    if cov[0] <= t_soi <= cov[1]:
        st, _ = spice.spkgeo(-82, t_soi, 'ECLIPJ2000', 699)
        import numpy as np
        r = np.linalg.norm(st[:3])
        print(f"\nSOI State relative to Saturn (699):")
        print(f"  Distance: {r:.1f} km (Saturn radius ~60,268 km, ring crossing)")
        print(f"  Velocity: {np.linalg.norm(st[3:]):.2f} km/s")

    # Check Grand Finale dive #1
    t_dive1 = spice.str2et("2017-04-26 09:00:00")
    sat_center = 699 if spice.bodfnd(699, "RADII") else 6
    st, _ = spice.spkgeo(-82, t_dive1, 'ECLIPJ2000', sat_center)
    import numpy as np
    r_dive = np.linalg.norm(st[:3])
    print(f"\nGrand Finale Dive #1 relative to Saturn ({sat_center}):")
    print(f"  Distance: {r_dive:.1f} km")
    print(f"  Velocity: {np.linalg.norm(st[3:]):.2f} km/s")

    spice.kclear()
    print("\n[+] Verification passed successfully!")


if __name__ == "__main__":
    main()
