"""Fetch state vectors from JPL Horizons for all bodies in Solar System system.json.

Queries ecliptic J2000 state vectors at epoch 2026-Oct-17 12:00:00 TDB.
Planets are relative to Sun center, moons relative to parent planet center.
Velocities are converted from AU/day to AU/yr to match REBOUND / IAS15 units.

Usage:  python scripts/fetch_horizons.py [--epoch "2026-10-17 12:00"]
Output: Updates data/systems/Solar System/system.json and data/system.json in-place,
        adding "sv" field to each body and setting motion_model="n_body".
"""
import argparse
import copy
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
DAYS_PER_YEAR = 365.25

# Mapping: body name -> Horizons NAIF ID
HORIZONS_IDS = {
    # Star
    "Sun":        "10",
    # Planets
    "Mercury":    "199",
    "Venus":      "299",
    "Earth":      "399",
    "Mars":       "499",
    "Jupiter":    "599",
    "Saturn":     "699",
    "Uranus":     "799",
    "Neptune":    "899",
    "Pluto":      "999",
    # Dwarf Planets / Major Asteroids
    "Ceres":      "1;",
    "Pallas":     "2;",
    "Vesta":      "4;",
    "Haumea":     "136108",
    "Eris":       "136199",
    "Makemake":   "136472",
    # Earth's moon
    "Moon":       "301",
    # Mars moons
    "Phobos":     "401",
    "Deimos":     "402",
    # Jupiter moons
    "Io":         "501",
    "Europa":     "502",
    "Ganymede":   "503",
    "Callisto":   "504",
    "Amalthea":   "505",
    "Thebe":      "514",
    "Adrastea":   "515",
    "Metis":      "516",
    # Saturn moons
    "Mimas":      "601",
    "Enceladus":  "602",
    "Tethys":     "603",
    "Dione":      "604",
    "Rhea":       "605",
    "Titan":      "606",
    "Hyperion":   "607",
    "Iapetus":    "608",
    "Phoebe":     "609",
    "Helene":     "612",
    "Telesto":    "613",
    "Calypso":    "614",
    "Methone":    "632",
    "Polydeuces": "634",
    # Uranus moons
    "Puck":       "715",
    "Miranda":    "705",
    "Ariel":      "701",
    "Umbriel":    "702",
    "Titania":    "703",
    "Oberon":     "704",
    # Neptune moons
    "Triton":     "801",
    "Naiad":      "803",
    "Thalassa":   "804",
    "Despina":    "805",
    "Galatea":    "806",
    "Larissa":    "807",
    "Hippocamp":  "814",
    "Proteus":    "808",
    # Pluto
    "Charon":     "901",
}

# Parent body NAIF IDs (for CENTER parameter)
PARENT_NAIF = {
    "Sun":     "0",    # Solar system barycenter (doesn't matter, we zero it)
    "Mercury": "10",   # Sun
    "Venus":   "10",
    "Earth":   "10",
    "Mars":    "10",
    "Jupiter": "10",
    "Saturn":  "10",
    "Uranus":  "10",
    "Neptune": "10",
    "Pluto":   "10",
}

# Outer planet system barycenter NAIF IDs
BARY_MAP = {
    "Jupiter": "5",
    "Saturn":  "6",
    "Uranus":  "7",
    "Neptune": "8",
    "Pluto":   "9",
}

def get_parent_center(body_name, parent_name):
    """Get the Horizons CENTER string for querying relative to parent."""
    if parent_name is None:
        return "500@0"  # SSB
    parent_naif = HORIZONS_IDS.get(parent_name, PARENT_NAIF.get(parent_name))
    if parent_naif:
        return f"500@{parent_naif}"
    return "500@10"  # fallback to Sun

_cache = None

def get_cache_file():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, 'horizons_cache.json')

def load_cache():
    global _cache
    if _cache is None:
        cpath = get_cache_file()
        if os.path.exists(cpath):
            with open(cpath, "r", encoding="utf-8") as f:
                try:
                    _cache = json.load(f)
                except Exception:
                    _cache = {}
        else:
            _cache = {}
    return _cache

def save_cache():
    if _cache is not None:
        cpath = get_cache_file()
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(_cache, f, indent=2)

def query_horizons(body_id, center, start_time="2026-10-17 12:00", stop_time="2026-10-17 12:01", max_retries=3):
    """Query JPL Horizons API for state vector at requested epoch."""
    cache = load_cache()
    cache_key = f"{body_id}_{center}_{start_time}_{stop_time}"
    if cache_key in cache:
        return cache[cache_key]

    params = {
        "format":      "text",
        "COMMAND":     f"'{body_id}'",
        "OBJ_DATA":    "'NO'",
        "MAKE_EPHEM":  "'YES'",
        "EPHEM_TYPE":  "'VECTORS'",
        "CENTER":      f"'{center}'",
        "REF_PLANE":   "'ECLIPTIC'",
        "REF_SYSTEM":  "'J2000'",
        "START_TIME":  f"'{start_time}'",
        "STOP_TIME":   f"'{stop_time}'",
        "STEP_SIZE":   "'1 d'",
        "VEC_TABLE":   "'2'",
        "OUT_UNITS":   "'AU-D'",
        "VEC_LABELS":  "'YES'",
        "CSV_FORMAT":  "'NO'",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params, safe="'@")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "StellarForge/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode('utf-8')
                cache[cache_key] = text
                save_cache()
            return text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2)
            else:
                raise

def parse_state_vector(response_text):
    """Parse state vector from Horizons response text."""
    soe = response_text.find("$$SOE")
    eoe = response_text.find("$$EOE")
    if soe < 0 or eoe < 0:
        return None

    block = response_text[soe:eoe]

    x_match = re.search(r'X\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    y_match = re.search(r'Y\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    z_match = re.search(r'Z\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    vx_match = re.search(r'VX\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    vy_match = re.search(r'VY\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    vz_match = re.search(r'VZ\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)

    if not all([x_match, y_match, z_match, vx_match, vy_match, vz_match]):
        return None

    return {
        "x":  float(x_match.group(1)),
        "y":  float(y_match.group(1)),
        "z":  float(z_match.group(1)),
        "vx": float(vx_match.group(1)) * DAYS_PER_YEAR,
        "vy": float(vy_match.group(1)) * DAYS_PER_YEAR,
        "vz": float(vz_match.group(1)) * DAYS_PER_YEAR,
    }

def apply_barycentric_velocity_alignment(bodies, start_time, stop_time, state_vectors=None):
    """Align planet initial velocities so subsystem barycenter velocity matches JPL system barycenter velocity."""
    print("\n--- Applying Subsystem Barycentric Velocity Alignment ---")
    for p_name, bary_id in BARY_MAP.items():
        p_body = next((b for b in bodies if b["name"] == p_name), None)
        if not p_body:
            continue

        sv_dict = state_vectors.get(p_name) if state_vectors is not None else p_body.get("sv")
        if not sv_dict:
            continue

        p_mass = p_body.get("m", 0.0)
        moons = [b for b in bodies if b.get("parentId") == p_name and ((state_vectors is not None and b["name"] in state_vectors) or ("sv" in b))]
        if not moons:
            continue

        tot_mass = p_mass + sum(m.get("m", 0.0) for m in moons)

        try:
            resp_bary = query_horizons(bary_id, "500@10", start_time=start_time, stop_time=stop_time)
            sv_bary = parse_state_vector(resp_bary)
            if not sv_bary:
                continue

            v_jpl_bary = np.array([sv_bary['vx'], sv_bary['vy'], sv_bary['vz']])
            v_center = np.array([sv_dict['vx'], sv_dict['vy'], sv_dict['vz']])

            sim_moon_momentum = np.zeros(3)
            for m in moons:
                msv = state_vectors[m["name"]] if state_vectors is not None else m["sv"]
                sim_moon_momentum += m.get("m", 0.0) * np.array([msv['vx'], msv['vy'], msv['vz']])

            v_sim_subsys = v_center + sim_moon_momentum / tot_mass
            delta_v = v_jpl_bary - v_sim_subsys

            sv_dict['vx'] += float(delta_v[0])
            sv_dict['vy'] += float(delta_v[1])
            sv_dict['vz'] += float(delta_v[2])
            print(f"  [{p_name:12s}] Applied barycentric velocity alignment (|delta_v| = {np.linalg.norm(delta_v)*149597870.7:.2f} km/yr)")
        except Exception as e:
            print(f"  [{p_name:12s}] Barycentric alignment skipped ({e})")

def fetch_all(epoch_str="2026-10-17 12:00"):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_files = [
        os.path.join(root, 'data', 'systems', 'Solar System', 'system.json'),
        os.path.join(root, 'data', 'system.json')
    ]

    base_path = target_files[0] if os.path.exists(target_files[0]) else target_files[1]
    with open(base_path, "r", encoding="utf-8") as f:
        bodies = json.load(f)

    start_time = epoch_str
    # 1 minute stop time
    # e.g. "2026-10-17 12:00" -> "2026-10-17 12:01"
    parts = start_time.split(":")
    if len(parts) == 2:
        stop_time = f"{parts[0]}:{int(parts[1])+1:02d}"
    else:
        stop_time = "2026-10-17 12:01"

    iso_epoch = epoch_str.replace(" ", "T")
    if len(iso_epoch) == 16:
        iso_epoch += ":00"

    print("=" * 70)
    print("JPL Horizons State Vector Fetcher")
    print(f"Epoch: {epoch_str}:00 TDB")
    print("Frame: Ecliptic J2000")
    print("Units: AU (positions), AU/yr (velocities)")
    print("=" * 70)

    success = 0
    failed = 0
    skipped = 0

    for body in bodies:
        name = body["name"]
        parent_name = body.get("parentId")

        if name not in HORIZONS_IDS:
            print(f"  [{name:12s}] SKIP - No Horizons ID mapped")
            skipped += 1
            continue

        body_id = HORIZONS_IDS[name]

        if parent_name:
            center = get_parent_center(name, parent_name)
        elif name == "Sun":
            body["sv"] = {"x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
            body["epoch"] = iso_epoch
            body["motion_model"] = "n_body"
            print(f"  [{name:12s}] OK    - Set to origin (Sun), epoch={iso_epoch}, motion_model=n_body")
            success += 1
            continue
        else:
            center = "500@10"

        print(f"  [{name:12s}] Querying Horizons (ID={body_id}, CENTER={center})...", end="", flush=True)

        try:
            response = query_horizons(body_id, center, start_time=start_time, stop_time=stop_time)
            sv = parse_state_vector(response)

            if sv:
                body["sv"] = sv
                dist = (sv['x']**2 + sv['y']**2 + sv['z']**2)**0.5
                print(f" OK  |r|={dist:.6f} AU")
                success += 1
            else:
                print(" FAIL - Could not parse response")
                failed += 1
        except Exception as e:
            print(f" ERROR - {e}")
            failed += 1

        time.sleep(0.3)

    print(f"\n{'='*70}")
    print(f"Results: {success} OK, {failed} failed, {skipped} skipped")
    print(f"{'='*70}")

    if success > 0:
        apply_barycentric_velocity_alignment(bodies, start_time=start_time, stop_time=stop_time)

        for target_path in target_files:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(bodies, f, indent=2)
            print(f"Updated {target_path} with {success} state vectors.")

        print(f"\nSuccessfully updated Solar System initial state vectors to epoch {epoch_str} (N-Body default).")
    else:
        print("\nNo state vectors fetched. Files not modified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch JPL Horizons state vectors for Solar System")
    parser.add_argument("--epoch", default="2026-10-17 12:00", help="TDB epoch string (YYYY-MM-DD HH:MM)")
    args = parser.parse_args()
    fetch_all(args.epoch)
