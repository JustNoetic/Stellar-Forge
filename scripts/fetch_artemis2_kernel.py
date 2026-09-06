"""Fetch Artemis II state vectors from NASA JPL Horizons and build a binary SPK kernel.

Queries Cartesian state vectors (J2000 ICRF, Solar System Barycenter observer)
for target body -1024 (Artemis II Orion spacecraft) across its complete mission duration
and encodes them into a standard NAIF Type 9 SPK kernel (`data/kernels/artemis2.bsp`).

Usage:
    python scripts/fetch_artemis2_kernel.py
"""

import os
import sys
import json
import re
import time
import urllib.request
import urllib.parse

def generate_artemis2_spk(output_dir=None, progress_callback=None):
    """
    Fetches Artemis II trajectory data from JPL Horizons and writes `artemis2.bsp`.
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
    bsp_path = os.path.join(output_dir, "artemis2.bsp")

    tls_path = os.path.join(output_dir, "naif0012.tls")
    if not os.path.exists(tls_path):
        # Attempt to load leapseconds from known location
        candidate_tls = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "kernels", "naif0012.tls"))
        if os.path.exists(candidate_tls):
            tls_path = candidate_tls
        else:
            return False, f"Leapseconds kernel {tls_path} not found. Please download required kernels first.", None

    spice.furnsh(tls_path)

    if progress_callback:
        progress_callback(0.1, "Connecting to NASA JPL Horizons API for Artemis II...")

    # Mission duration: from ICPS separation through translunar loop to Pacific entry
    start_time = "2026-04-02 02:00"
    stop_time = "2026-04-10 23:50"
    step_size = "5m"

    params = {
        'format': 'json',
        'COMMAND': "'-1024'",
        'EPHEM_TYPE': 'VECTORS',
        'CENTER': "'500@0'",      # Solar System Barycenter
        'REF_PLANE': 'FRAME',      # ICRF / J2000
        'START_TIME': f"'{start_time}'",
        'STOP_TIME': f"'{stop_time}'",
        'STEP_SIZE': f"'{step_size}'",
        'VEC_TABLE': '3'
    }

    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)
    t0 = time.time()
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Stellar-Forge/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        spice.unload(tls_path)
        return False, f"Failed to connect to JPL Horizons: {e}", None

    if progress_callback:
        progress_callback(0.5, f"Parsing Horizons response ({time.time() - t0:.1f}s)...")

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
        progress_callback(0.7, f"Building SPK Type 9 kernel with {len(matches)} states...")

    epochs = []
    states = []
    for m in matches:
        date_str = m[1].replace("TDB", "").strip()
        et = spice.str2et(date_str)
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
        # Create Type 9 SPK: Lagrange interpolation degree 3
        handle = spice.spkopn(temp_bsp, "ARTEMIS_II_ORION", 0)
        spice.spkw09(
            handle,
            -1024,                  # Target: Artemis II Orion
            0,                      # Center: Solar System Barycenter
            "J2000",                # Inertial frame
            epochs[0],              # Begin epoch
            epochs[-1],             # End epoch
            "ARTEMIS_II_ORION_TRAJ",# Segment ID
            3,                      # Polynomial degree (cubic Lagrange)
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
    except Exception as e:
        if os.path.exists(temp_bsp):
            try:
                os.remove(temp_bsp)
            except OSError:
                pass
        spice.unload(tls_path)
        return False, f"Failed to encode SPK kernel: {e}", None

    spice.unload(tls_path)

    if progress_callback:
        progress_callback(1.0, f"Successfully created artemis2.bsp ({os.path.getsize(bsp_path) / 1024:.1f} KB)")

    return True, f"Successfully generated {bsp_path}", bsp_path


def main():
    print("==================================================")
    print(" Artemis II SPICE Kernel Generator (JPL Horizons)")
    print("==================================================")

    def print_progress(frac, msg):
        print(f"[{int(frac * 100):3d}%] {msg}")

    success, message, path = generate_artemis2_spk(progress_callback=print_progress)
    print(message)
    if not success:
        sys.exit(1)

    # Verification query
    import spiceypy as spice
    kdir = os.path.dirname(path)
    tls = os.path.join(kdir, "naif0012.tls")
    if os.path.exists(tls):
        spice.furnsh(tls)
        spice.furnsh(path)
        et = spice.str2et("2026-04-06 23:01:00")  # Closest approach to Moon
        state, _ = spice.spkgeo(-1024, et, "ECLIPJ2000", 0)
        utc_s = spice.et2utc(et, "C", 0)
        print(f"\nVerification query at {utc_s} (Moon flyby):")
        print(f"  SSB Position (km): {state[0:3]}")
        print(f"  Velocity (km/s):   {state[3:6]}")
        spice.unload(path)
        spice.unload(tls)
    print("\nDone!")

if __name__ == "__main__":
    main()
