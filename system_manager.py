"""
Multi-system management for Stellar Forge.

Handles creation, saving, and switching between independent star systems.
Includes basic stellar evolution models for deriving star properties from
the "big three": mass, metallicity, and age.
"""

import json
import os
import math
import datetime


# ═══════════════════════════════════════════════════════════════════════
#  Stellar Evolution — derive star properties from mass, metallicity, age
# ═══════════════════════════════════════════════════════════════════════

def derive_star_properties(mass, metallicity, age):
    """Derive stellar properties from the big three parameters.

    Uses piecewise mass-luminosity/radius relations for the ZAMS,
    age-dependent evolution along the main sequence, and simplified
    post-MS tracks.  Metallicity shifts effective temperature.

    Args:
        mass: stellar mass in solar masses (M☉)
        metallicity: [Fe/H] in dex (0.0 = solar)
        age: stellar age in gigayears (Gyr)

    Returns:
        dict with keys: luminosity (L☉), radius (R☉), temperature (K),
        spectral_class (str), stage (str), luminosity_class (str),
        ms_lifetime (Gyr), progress (0-100%)
    """
    # ── Zero-Age Main Sequence luminosity (mass-luminosity relation) ──
    if mass < 0.08:
        # Sub-stellar / brown dwarf regime
        L_zams = 0.23 * mass ** 2.3
    elif mass < 0.43:
        L_zams = 0.23 * mass ** 2.3
    elif mass < 2.0:
        L_zams = mass ** 4.0
    elif mass < 55.0:
        L_zams = 1.4 * mass ** 3.5
    else:
        L_zams = 32000.0 * mass

    # ── Zero-Age Main Sequence radius ──
    if mass < 1.0:
        R_zams = mass ** 0.8
    else:
        R_zams = mass ** 0.57

    # ── Main sequence lifetime ──
    if L_zams > 0:
        t_ms = (mass / L_zams) * 10.0   # Gyr
    else:
        t_ms = 1e6  # effectively infinite

    # ── Progress through main sequence ──
    if t_ms > 0:
        progress = min(age / t_ms, 1.0)
    else:
        progress = 1.0

    # ── Determine evolutionary stage and current L, R ──
    if mass < 0.08:
        # Brown dwarf — doesn't fuse hydrogen
        stage = "Brown Dwarf"
        lum_class = ""
        L = L_zams * max(0.01, (1.0 - 0.3 * min(age / 10.0, 1.0)))
        R = R_zams
    elif age < t_ms * 0.9:
        # Main sequence — gradual brightening and expansion
        stage = "Main Sequence"
        lum_class = "V"
        L = L_zams * (1.0 + 0.4 * progress)
        R = R_zams * (1.0 + 0.15 * progress)
    elif age < t_ms:
        # Terminal main sequence / subgiant transition
        stage = "Subgiant"
        lum_class = "IV"
        sub_prog = (age - t_ms * 0.9) / (t_ms * 0.1)
        L = L_zams * 1.4 * (1.0 + 2.0 * sub_prog)
        R = R_zams * 1.15 * (1.0 + 3.0 * sub_prog)
    elif mass < 0.5:
        # Very low-mass stars outlive the universe
        stage = "Main Sequence"
        lum_class = "V"
        L = L_zams
        R = R_zams
    elif mass < 8.0:
        # Red Giant Branch
        stage = "Red Giant"
        lum_class = "III"
        post_ms_frac = min((age - t_ms) / max(t_ms * 0.3, 0.01), 1.0)
        L = L_zams * (50.0 + 950.0 * post_ms_frac)
        R = R_zams * (10.0 + 140.0 * post_ms_frac)
    elif mass < 25.0:
        # Red/Yellow Supergiant
        stage = "Supergiant"
        lum_class = "Ib"
        L = L_zams * 500.0
        R = R_zams * 200.0
    else:
        # Blue/Luminous Supergiant
        stage = "Supergiant"
        lum_class = "Ia"
        L = L_zams * 1000.0
        R = R_zams * 300.0

    # ── Effective temperature from Stefan-Boltzmann ──
    #   L = 4π R² σ T⁴  →  T = T☉ × (L/L☉)^0.25 / (R/R☉)^0.5
    if R > 0:
        T = 5778.0 * (L ** 0.25) / (R ** 0.5)
    else:
        T = 5778.0

    # ── Metallicity correction to temperature ──
    #   Higher metallicity → increased opacity → cooler photosphere
    T *= (1.0 - 0.05 * metallicity)
    T = max(T, 500.0)

    # ── Spectral classification ──
    spectral = get_spectral_class(T, lum_class)

    return {
        "luminosity": round(L, 6),
        "radius": round(R, 6),
        "temperature": round(T, 1),
        "spectral_class": spectral,
        "stage": stage,
        "luminosity_class": lum_class,
        "ms_lifetime": round(t_ms, 4),
        "progress": round(progress * 100.0, 2),
    }


def get_spectral_class(temp, lum_class="V"):
    """Determine MK spectral classification from effective temperature.

    Args:
        temp: effective temperature in Kelvin
        lum_class: luminosity class string (V, IV, III, II, Ib, Ia, etc.)

    Returns:
        str like "G2V", "M4III", "O7Ia"
    """
    # Temperature boundaries for spectral types (upper boundary, letter)
    spectral_types = [
        (50000, "O", 30000),   # O: 30000-50000+
        (30000, "B", 10000),   # B: 10000-30000
        (10000, "A", 7500),    # A: 7500-10000
        (7500,  "F", 6000),    # F: 6000-7500
        (6000,  "G", 5200),    # G: 5200-6000
        (5200,  "K", 3700),    # K: 3700-5200
        (3700,  "M", 2400),    # M: 2400-3700
        (2400,  "L", 1300),    # L: 1300-2400
        (1300,  "T", 500),     # T: 500-1300
        (500,   "Y", 200),     # Y: 200-500
    ]

    for i, (t_upper, letter, t_lower) in enumerate(spectral_types):
        if temp >= t_lower:
            # Subdivision 0-9 within the temperature range
            t_range = t_upper - t_lower
            if t_range > 0:
                frac = (t_upper - temp) / t_range
                subtype = int(frac * 10.0)
                subtype = max(0, min(9, subtype))
            else:
                subtype = 0
            return f"{letter}{subtype}{lum_class}"

    return f"Y9{lum_class}"


def temperature_to_rgb(temp_k):
    """Convert blackbody temperature to RGB color (0-1 range).

    Uses the Tanner Helland algorithm for perceptually accurate
    blackbody radiation colors.

    Args:
        temp_k: temperature in Kelvin

    Returns:
        tuple (r, g, b) with values in [0, 1]
    """
    temp = max(1000, min(40000, temp_k)) / 100.0

    # Red channel
    if temp <= 66:
        r = 1.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592) / 255.0

    # Green channel
    if temp <= 66:
        g = (99.4708025861 * math.log(temp) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492) / 255.0

    # Blue channel
    if temp >= 66:
        b = 1.0
    elif temp <= 19:
        b = 0.0
    else:
        b = (138.5177312231 * math.log(temp - 10) - 305.0447927307) / 255.0

    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
    )


def rgb_to_hex(r, g, b):
    """Convert RGB (0-1 range) to hex color string."""
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ═══════════════════════════════════════════════════════════════════════
#  System Data Helpers
# ═══════════════════════════════════════════════════════════════════════

def create_star_body(star_name, mass, metallicity, age):
    """Create a bodies_data entry for a new star.

    Derives all physical properties from mass, metallicity, and age,
    then formats them into the system.json body schema.

    Args:
        star_name: display name for the star (e.g., "Alpha")
        mass: mass in solar masses
        metallicity: [Fe/H] in dex
        age: age in Gyr

    Returns:
        dict in the same format as entries in system.json
    """
    props = derive_star_properties(mass, metallicity, age)
    r, g, b = temperature_to_rgb(props["temperature"])
    color_hex = rgb_to_hex(r, g, b)

    return {
        "name": star_name,
        "type": "Star",
        "m": mass,
        "r": props["radius"],
        "color": color_hex,
        "is_root": True,
        "pole_ra": 270.0,
        "pole_dec": 66.5607089,
        "star_props": {
            "lum": props["luminosity"],
            "temp": props["temperature"],
            "age": age,
            "metallicity": metallicity,
            "prog": props["progress"],
            "class": props["spectral_class"],
            "stage": props["stage"],
        },
        "sv": {"x": 0, "y": 0, "z": 0, "vx": 0, "vy": 0, "vz": 0},
    }


# ═══════════════════════════════════════════════════════════════════════
#  System Manager — multi-system save / load / switch
# ═══════════════════════════════════════════════════════════════════════

class SystemSnapshot:
    """In-memory snapshot of a running system for fast switching.

    Stores a copy of the REBOUND simulation and all associated render/UI
    state so we can switch away and later restore without disk I/O.
    """
    __slots__ = (
        'sim_copy', 'num_bodies', 'bodies_data', 'visual_data',
        'atmo_bodies', 'ring_bodies', 'oblate_physics_list',
        'has_j2', 'has_gr', 'phys_star_idx', 'star_idx',
        'parent_indices', 'sim_time', 'time_multiplier',
        'was_paused',
    )

    def __init__(self):
        self.sim_copy = None
        self.num_bodies = 0
        self.bodies_data = []
        self.visual_data = []
        self.atmo_bodies = []
        self.ring_bodies = []
        self.oblate_physics_list = []
        self.has_j2 = False
        self.has_gr = False
        self.phys_star_idx = -1
        self.star_idx = 0
        self.parent_indices = None
        self.sim_time = 0.0
        self.time_multiplier = 1.0
        self.was_paused = False


class SystemManager:
    """Manages multiple star systems with disk persistence and in-memory snapshots.

    Directory layout::

        data/systems/
        ├── My Star System/
        │   ├── system.json   # body definitions
        │   └── meta.json     # metadata (creation date, star info, body count)
        └── Binary System/
            ├── system.json
            └── meta.json

    The default Solar System is NEVER stored here — it always loads from
    ``data/system.json`` (the "holy grail").
    """

    SOLAR_SYSTEM_NAME = "Solar System"

    def __init__(self, systems_dir="data/systems", default_json="data/system.json"):
        self.systems_dir = systems_dir
        self.default_json = default_json
        self.active_system = self.SOLAR_SYSTEM_NAME
        self.snapshots = {}  # name -> SystemSnapshot (in-memory, for session switching)
        self._ensure_dirs()

    # ── File-system helpers ──────────────────────────────────────────

    def _ensure_dirs(self):
        os.makedirs(self.systems_dir, exist_ok=True)

    def _system_dir(self, name):
        return os.path.join(self.systems_dir, name)

    def _system_json_path(self, name):
        return os.path.join(self._system_dir(name), "system.json")

    def _meta_json_path(self, name):
        return os.path.join(self._system_dir(name), "meta.json")

    # ── Listing ──────────────────────────────────────────────────────

    def list_systems(self):
        """Return list of system names, Solar System always first."""
        names = [self.SOLAR_SYSTEM_NAME]
        if os.path.isdir(self.systems_dir):
            for entry in sorted(os.listdir(self.systems_dir)):
                full = os.path.join(self.systems_dir, entry)
                if os.path.isdir(full) and os.path.isfile(os.path.join(full, "system.json")):
                    if entry != self.SOLAR_SYSTEM_NAME:
                        names.append(entry)
        return names

    def system_exists(self, name):
        if name == self.SOLAR_SYSTEM_NAME:
            return True
        return os.path.isfile(self._system_json_path(name))

    # ── Loading ──────────────────────────────────────────────────────

    def load_default_system(self):
        """Load the sacred Solar System from data/system.json."""
        with open(self.default_json, 'r') as f:
            return json.load(f)

    def load_system_data(self, name):
        """Load bodies_data for a named system.

        For Solar System, if a working copy exists in data/systems/Solar System/,
        it loads that. If not, it loads the default data/system.json template,
        saves a working copy, and returns it.
        """
        if name == self.SOLAR_SYSTEM_NAME:
            if os.path.isfile(self._system_json_path(name)):
                path = self._system_json_path(name)
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                bodies_data = self.load_default_system()
                self.save_system_data(name, bodies_data)
                self.save_meta(name, bodies_data)
                return bodies_data
                
        path = self._system_json_path(name)
        with open(path, 'r') as f:
            return json.load(f)

    # ── Saving ───────────────────────────────────────────────────────

    def save_system_data(self, name, bodies_data):
        """Save bodies_data for a custom system to disk."""
        sdir = self._system_dir(name)
        os.makedirs(sdir, exist_ok=True)
        path = self._system_json_path(name)
        with open(path, 'w') as f:
            json.dump(bodies_data, f, indent=2)

    def save_meta(self, name, bodies_data):
        """Write metadata for a system."""
        star_info = {}
        body_count = len(bodies_data)
        for b in bodies_data:
            if b.get('type') == 'Star':
                star_info = {
                    "star_name": b.get("name", "Unknown"),
                    "mass": b.get("m", 0.0),
                    "spectral_class": b.get("star_props", {}).get("class", ""),
                }
                break

        meta = {
            "created": datetime.datetime.now().isoformat(),
            "modified": datetime.datetime.now().isoformat(),
            "body_count": body_count,
            "star": star_info,
        }
        path = self._meta_json_path(name)
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)

    # ── Creation ─────────────────────────────────────────────────────

    def create_new_system(self, system_name, star_name, mass, metallicity, age):
        """Create a brand-new star system with a single star.

        Generates the star body from mass/metallicity/age, writes files to
        disk, and returns the bodies_data list.

        Args:
            system_name: display name for the system
            star_name: name of the central star
            mass: stellar mass in solar masses
            metallicity: [Fe/H] in dex
            age: stellar age in Gyr

        Returns:
            list containing one star body dict
        """
        star = create_star_body(star_name, mass, metallicity, age)
        bodies_data = [star]

        self.save_system_data(system_name, bodies_data)
        self.save_meta(system_name, bodies_data)

        return bodies_data

    # ── Deletion ─────────────────────────────────────────────────────

    def delete_system(self, name):
        """Delete a custom system from disk. Cannot delete Solar System."""
        if name == self.SOLAR_SYSTEM_NAME:
            return
        sdir = self._system_dir(name)
        if os.path.isdir(sdir):
            import shutil
            shutil.rmtree(sdir)
        if name in self.snapshots:
            del self.snapshots[name]

    # ── Snapshot management (in-memory, for session switching) ────────

    def store_snapshot(self, name, snapshot):
        """Store an in-memory snapshot for later restoration."""
        self.snapshots[name] = snapshot

    def get_snapshot(self, name):
        """Retrieve a stored snapshot, or None if not available."""
        return self.snapshots.get(name)

    def clear_snapshot(self, name):
        """Remove a stored snapshot."""
        self.snapshots.pop(name, None)
