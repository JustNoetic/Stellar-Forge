"""Fetch state vectors from JPL Horizons for all bodies in system.json.

Queries ecliptic J2000 state vectors at epoch J2000 (2000-Jan-01 12:00:00 TDB).
Planets are relative to Sun center, moons relative to parent planet center.
Velocities are converted from AU/day to AU/yr to match REBOUND units.

Usage:  python fetch_horizons.py
Output: Updates system.json in-place, adding "sv" field to each body.
"""
import json, time, re, sys
import urllib.request
import urllib.parse

# JPL Horizons REST API
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


def get_parent_center(body_name, parent_name):
    """Get the Horizons CENTER string for querying relative to parent."""
    if parent_name is None:
        return "500@0"  # SSB
    parent_naif = HORIZONS_IDS.get(parent_name, PARENT_NAIF.get(parent_name))
    if parent_naif:
        return f"500@{parent_naif}"
    return "500@10"  # fallback to Sun


def query_horizons(body_id, center, max_retries=3):
    """Query JPL Horizons API for state vector at J2000 epoch."""
    params = {
        "format":      "text",
        "COMMAND":     f"'{body_id}'",
        "OBJ_DATA":    "'NO'",
        "MAKE_EPHEM":  "'YES'",
        "EPHEM_TYPE":  "'VECTORS'",
        "CENTER":      f"'{center}'",
        "REF_PLANE":   "'ECLIPTIC'",
        "REF_SYSTEM":  "'J2000'",
        "START_TIME":  "'2026-06-03 12:00'",
        "STOP_TIME":   "'2026-06-04'",
        "STEP_SIZE":   "'1 d'",
        "VEC_TABLE":   "'2'",
        "OUT_UNITS":   "'AU-D'",
        "VEC_LABELS":  "'YES'",
        "CSV_FORMAT":  "'NO'",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params, safe="'@")
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode('utf-8')
            return text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2)
            else:
                raise


def parse_state_vector(response_text):
    """Parse state vector from Horizons response text."""
    # Find the $$SOE ... $$EOE block
    soe = response_text.find("$$SOE")
    eoe = response_text.find("$$EOE")
    if soe < 0 or eoe < 0:
        return None
    
    block = response_text[soe:eoe]
    
    # Parse X, Y, Z, VX, VY, VZ using regex
    # Format: " X = 1.234E+00 Y = 2.345E+00 Z = 3.456E-01"
    #         " VX= 1.234E-02 VY= 2.345E-02 VZ= 3.456E-03"
    x_match = re.search(r'X\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    y_match = re.search(r'Y\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    z_match = re.search(r'Z\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    vx_match = re.search(r'VX\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    vy_match = re.search(r'VY\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    vz_match = re.search(r'VZ\s*=\s*([+\-]?\d+\.\d+E[+\-]?\d+)', block)
    
    if not all([x_match, y_match, z_match, vx_match, vy_match, vz_match]):
        return None
    
    # Positions in AU, velocities in AU/day -> convert vel to AU/yr
    sv = {
        "x":  float(x_match.group(1)),
        "y":  float(y_match.group(1)),
        "z":  float(z_match.group(1)),
        "vx": float(vx_match.group(1)) * DAYS_PER_YEAR,
        "vy": float(vy_match.group(1)) * DAYS_PER_YEAR,
        "vz": float(vz_match.group(1)) * DAYS_PER_YEAR,
    }
    return sv


def main():
    # Load system.json
    with open("system.json", "r") as f:
        bodies = json.load(f)
    
    print("=" * 70)
    print("JPL Horizons State Vector Fetcher")
    print("Epoch: 2026-06-03 12:00:00 TDB (Current Time)")
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
        
        # Determine center body
        if parent_name:
            center = get_parent_center(name, parent_name)
        elif name == "Sun":
            # Sun at origin (we'll move_to_com anyway)
            body["sv"] = {"x": 0, "y": 0, "z": 0, "vx": 0, "vy": 0, "vz": 0}
            print(f"  [{name:12s}] OK    - Set to origin (Sun)")
            success += 1
            continue
        else:
            center = "500@10"  # relative to Sun
        
        print(f"  [{name:12s}] Querying Horizons (ID={body_id}, CENTER={center})...", end="", flush=True)
        
        try:
            response = query_horizons(body_id, center)
            sv = parse_state_vector(response)
            
            if sv:
                body["sv"] = sv
                dist = (sv['x']**2 + sv['y']**2 + sv['z']**2)**0.5
                print(f" OK  |r|={dist:.6f} AU")
                success += 1
            else:
                print(f" FAIL - Could not parse response")
                # Debug: print relevant portion of response
                if "$$SOE" in response:
                    soe = response.find("$$SOE")
                    eoe = response.find("$$EOE")
                    print(f"    Block: {response[soe:eoe+5][:200]}")
                else:
                    # Look for error messages
                    for line in response.split('\n'):
                        if 'No ephemeris' in line or 'cannot' in line.lower() or 'error' in line.lower():
                            print(f"    {line.strip()}")
                failed += 1
        except Exception as e:
            print(f" ERROR - {e}")
            failed += 1
        
        time.sleep(0.5)  # Rate limiting (be nice to JPL servers)
    
    print(f"\n{'='*70}")
    print(f"Results: {success} OK, {failed} failed, {skipped} skipped")
    print(f"{'='*70}")
    
    if success > 0:
        # Write updated system.json
        with open("system.json", "w") as f:
            json.dump(bodies, f, indent=2)
        print(f"\nUpdated system.json with {success} state vectors.")
        
        if failed > 0:
            print(f"NOTE: {failed} bodies failed - they will use Keplerian element fallback.")
    else:
        print("\nNo state vectors fetched. system.json not modified.")


if __name__ == "__main__":
    main()
