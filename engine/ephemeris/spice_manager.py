import os
import json
import urllib.request
import spiceypy as spice
import numpy as np
import threading

from engine.path_utils import get_external_path

SPICE_LOCK = threading.RLock()

def get_spice_manager():
    """Returns the active SpiceManager instance if one exists."""
    return getattr(SpiceManager, "_instance", None)

class SpiceManager:
    """
    Manages SPICE kernels and Ephemeris Mode operations.
    Handles downloading basic kernels from NAIF and converting state vectors.
    """
    
    KERNEL_DIR = get_external_path("data", "kernels")
    DEFAULT_KERNEL_DIR = get_external_path("data", "kernels", "default")
    ADDITIONAL_KERNEL_DIR = get_external_path("data", "kernels", "additional")
    
    # Essential NAIF kernels for a basic Solar System ephemeris
    DEFAULT_KERNELS = {
        "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
        "pck00010.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
        "de440s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
        "jup347.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup347.bsp",
        "jup348.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup348.bsp",
        "jup349.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup349.bsp",
        "jup365.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup365.bsp",
        "sat455.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat455.bsp",
        "sat456.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat456.bsp",
        "sat457.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat457.bsp",
        "sat459.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat459.bsp",
        "sat441.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat441.bsp",
        "ura111.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/ura111.bsp",
        "nep095.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep095.bsp",
        "nep104.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep104.bsp",
        "nep105.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep105.bsp",
        "plu060.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/plu060.bsp",
        "mar099s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/mar099s.bsp"
    }
    
    KERNEL_DESCRIPTIONS = {
        "naif0012.tls": "Leapseconds (Required)",
        "pck00010.tpc": "Planetary Constants (Required)",
        "de440s.bsp": "Planetary Ephemeris (Required)",
        "jup347.bsp": "Irregular Jupiter Moons (879MB)",
        "jup348.bsp": "Irregular Jupiter Moons (57MB)",
        "jup349.bsp": "Irregular Jupiter Moons (93MB)",
        "jup365.bsp": "Major Jupiter Moons (1.1GB)",
        "sat455.bsp": "Irregular Saturn Moons (278MB)",
        "sat456.bsp": "Irregular Saturn Moons (115MB)",
        "sat457.bsp": "Irregular Saturn Moons (190MB)",
        "sat459.bsp": "Irregular Saturn Moons (80MB)",
        "sat441.bsp": "Major Saturn Moons (631MB)",
        "ura111.bsp": "Major Uranus Moons (Miranda, Ariel, etc.)",
        "nep095.bsp": "Major & Inner Neptune Moons (Triton, Proteus, etc.)",
        "nep104.bsp": "Irregular Neptune Moons (318MB)",
        "nep105.bsp": "Nereid Ephemeris (201MB)",
        "plu060.bsp": "Pluto Moons",
        "mar099s.bsp": "Mars Moons"
    }

    ESSENTIAL_KERNELS = {"naif0012.tls", "pck00010.tpc", "de440s.bsp", "jup365.bsp", "sat441.bsp", "ura111.bsp", "nep095.bsp", "mar099s.bsp", "plu060.bsp"}

    # Standard body mapping from SPICE IDs to recognizable names for rendering
    # SPICE IDs: 10=Sun, 1=Mercury Barycenter, 2=Venus Barycenter, 3=Earth Barycenter, etc.
    # 399=Earth, 301=Moon. Using barycenters (1-9) for outer planets is standard.
    SPICE_BODIES = {
        10: {"name": "Sun", "type": "Star", "color": "#fff5e6", "radius_km": 696340.0, "mass_kg": 1.98847e30},
        199: {"name": "Mercury", "type": "Planet", "color": "#a8a8a8", "radius_km": 2439.7, "mass_kg": 3.3011e23},
        299: {"name": "Venus", "type": "Planet", "color": "#e0c8a0", "radius_km": 6051.8, "mass_kg": 4.8675e24},
        399: {"name": "Earth", "type": "Planet", "color": "#3b5d9c", "radius_km": 6371.0, "mass_kg": 5.972e24},
        301: {"name": "Moon", "type": "Moon", "color": "#d0d0d0", "radius_km": 1737.4, "mass_kg": 7.342e22},
        -1024: {
            "name": "Artemis II (Orion)",
            "type": "Spacecraft",
            "color": "#00ffff",
            "radius_km": 0.010,
            "mass_kg": 26000.0,
            "parentId": "Earth",
            "is_spacecraft": True,
            "mission_start_utc": "2026-04-02 02:00:00",
            "mission_end_utc": "2026-04-10 23:50:00"
        },
        -82: {
            "name": "Cassini",
            "type": "Spacecraft",
            "color": "#e0a040",
            "radius_km": 0.010,
            "mass_kg": 5712.0,
            "parentId": "Saturn",
            "is_spacecraft": True,
            "mission_start_utc": "2004-07-01 14:00:00",
            "mission_end_utc": "2017-09-15 09:58:00"
        },
        499: {"name": "Mars", "type": "Planet", "color": "#c1440e", "radius_km": 3389.5, "mass_kg": 6.4171e23},
        401: {"name": "Phobos", "type": "Moon", "color": "#a0a0a0", "radius_km": 11.26, "mass_kg": 1.0659e16},
        402: {"name": "Deimos", "type": "Moon", "color": "#a0a0a0", "radius_km": 6.2, "mass_kg": 1.4762e15},
        5: {"name": "Jupiter", "type": "Planet", "color": "#c88b3a", "radius_km": 69911.0, "mass_kg": 1.8982e27},
        501: {"name": "Io", "type": "Moon", "color": "#c1b446", "radius_km": 1821.6, "mass_kg": 8.9319e22},
        502: {"name": "Europa", "type": "Moon", "color": "#9b7f59", "radius_km": 1560.8, "mass_kg": 4.7998e22},
        503: {"name": "Ganymede", "type": "Moon", "color": "#8b8173", "radius_km": 2631.2, "mass_kg": 1.4819e23},
        504: {"name": "Callisto", "type": "Moon", "color": "#5e5850", "radius_km": 2410.3, "mass_kg": 1.0759e23},
        505: {"name": "Amalthea", "type": "Moon", "color": "#e65f5c", "radius_km": 83.56, "mass_kg": 2.068e18},
        514: {"name": "Thebe", "type": "Moon", "color": "#5c544e", "radius_km": 48.74, "mass_kg": 4.195e17},
        515: {"name": "Adrastea", "type": "Moon", "color": "#78716c", "radius_km": 8.36, "mass_kg": 1.948e15},
        516: {"name": "Metis", "type": "Moon", "color": "#57534e", "radius_km": 21.59, "mass_kg": 3.596e16},
        6: {"name": "Saturn", "type": "Planet", "color": "#e3d599", "radius_km": 58232.0, "mass_kg": 5.6834e26},
        601: {"name": "Mimas", "type": "Moon", "color": "#a0a0a0", "radius_km": 198.2, "mass_kg": 3.7493e19},
        602: {"name": "Enceladus", "type": "Moon", "color": "#ffffff", "radius_km": 252.1, "mass_kg": 1.0802e20},
        603: {"name": "Tethys", "type": "Moon", "color": "#a0a0a0", "radius_km": 531.1, "mass_kg": 6.1744e20},
        604: {"name": "Dione", "type": "Moon", "color": "#a0a0a0", "radius_km": 561.4, "mass_kg": 1.0954e21},
        605: {"name": "Rhea", "type": "Moon", "color": "#a0a0a0", "radius_km": 763.8, "mass_kg": 2.3065e21},
        606: {"name": "Titan", "type": "Moon", "color": "#e0a040", "radius_km": 2574.7, "mass_kg": 1.3452e23},
        607: {"name": "Hyperion", "type": "Moon", "color": "#a8a29e", "radius_km": 135.1, "mass_kg": 5.584e18},
        608: {"name": "Iapetus", "type": "Moon", "color": "#505050", "radius_km": 734.5, "mass_kg": 1.8056e21},
        609: {"name": "Phoebe", "type": "Moon", "color": "#374151", "radius_km": 106.5, "mass_kg": 8.289e18},
        612: {"name": "Helene", "type": "Moon", "color": "#e2e8f0", "radius_km": 17.4, "mass_kg": 2.547e16},
        613: {"name": "Telesto", "type": "Moon", "color": "#f1f5f9", "radius_km": 11.8, "mass_kg": 7.192e15},
        614: {"name": "Calypso", "type": "Moon", "color": "#f8fafc", "radius_km": 10.4, "mass_kg": 3.596e15},
        632: {"name": "Methone", "type": "Moon", "color": "#cbd5e1", "radius_km": 1.39, "mass_kg": 1.498e13},
        634: {"name": "Polydeuces", "type": "Moon", "color": "#94a3b8", "radius_km": 0.70, "mass_kg": 4.495e12},
        7: {"name": "Uranus", "type": "Planet", "color": "#4b70dd", "radius_km": 25362.0, "mass_kg": 8.6810e25},
        701: {"name": "Ariel", "type": "Moon", "color": "#a0a0a0", "radius_km": 578.9, "mass_kg": 1.353e21},
        702: {"name": "Umbriel", "type": "Moon", "color": "#808080", "radius_km": 584.7, "mass_kg": 1.275e21},
        703: {"name": "Titania", "type": "Moon", "color": "#a0a0a0", "radius_km": 788.4, "mass_kg": 3.400e21},
        704: {"name": "Oberon", "type": "Moon", "color": "#808080", "radius_km": 761.4, "mass_kg": 3.076e21},
        705: {"name": "Miranda", "type": "Moon", "color": "#a0a0a0", "radius_km": 235.8, "mass_kg": 6.59e19},
        715: {"name": "Puck", "type": "Moon", "color": "#3a3a3a", "radius_km": 80.8, "mass_kg": 2.892e18},
        8: {"name": "Neptune", "type": "Planet", "color": "#274687", "radius_km": 24622.0, "mass_kg": 1.0241e26},
        801: {"name": "Triton", "type": "Moon", "color": "#a0a0a0", "radius_km": 1353.4, "mass_kg": 2.14e22},
        803: {"name": "Naiad", "type": "Moon", "color": "#888888", "radius_km": 29.9, "mass_kg": 1.888e17},
        804: {"name": "Thalassa", "type": "Moon", "color": "#808080", "radius_km": 39.7, "mass_kg": 3.506e17},
        805: {"name": "Despina", "type": "Moon", "color": "#7d7d7d", "radius_km": 73.8, "mass_kg": 2.128e18},
        806: {"name": "Galatea", "type": "Moon", "color": "#7a7a7a", "radius_km": 79.4, "mass_kg": 2.113e18},
        807: {"name": "Larissa", "type": "Moon", "color": "#6e6e6e", "radius_km": 96.8, "mass_kg": 4.900e18},
        808: {"name": "Proteus", "type": "Moon", "color": "#505050", "radius_km": 200.5, "mass_kg": 4.390e19},
        814: {"name": "Hippocamp", "type": "Moon", "color": "#555555", "radius_km": 17.4, "mass_kg": 4.944e16},
        9: {"name": "Pluto", "type": "Dwarf Planet", "color": "#ddc8b8", "radius_km": 1188.3, "mass_kg": 1.303e22},
        901: {"name": "Charon", "type": "Moon", "color": "#a0a0a0", "radius_km": 606.0, "mass_kg": 1.586e21}
    }

    # Conversion factors
    KM_TO_AU = 1.0 / 149597870.7
    SEC_TO_YR = 86400.0 * 365.25

    def __init__(self):
        SpiceManager._instance = self
        self._ensure_dirs()
        self.kernels_loaded = False
        self.active_epoch_et = None
        self.active_ephem_et_range = None
        self.active_ephem_bsp = None
        self.download_progress = 0.0
        self.download_status = ""
        self.is_downloading = False
        self.cancel_requested = False
        self.current_download_file = None
        self.download_bytes_current = 0
        self.download_bytes_total = 0
        self.download_speed_str = ""
        self.download_speed_bytes_per_sec = 0.0
        self.download_error = None
        self.settings_path = get_external_path("data", "ephemeris_settings.json")
        self.enabled_kernels = {k: (k in self.ESSENTIAL_KERNELS) for k in self.DEFAULT_KERNELS.keys()}
        self.settings_initialized = False
        self._load_settings()

    def cancel_download(self):
        """Cancel any active downloading process and clean up temporary partial files."""
        self.cancel_requested = True
        self.is_downloading = False
        self.download_status = "Download cancelled by user."
        self.download_speed_str = ""
        self._cleanup_partial_downloads()

    def _cleanup_partial_downloads(self):
        """Removes all .part files in kernel directories."""
        for d in [self.DEFAULT_KERNEL_DIR, self.ADDITIONAL_KERNEL_DIR, self.KERNEL_DIR]:
            if os.path.exists(d):
                try:
                    for f in os.listdir(d):
                        if f.endswith('.part'):
                            try:
                                os.remove(os.path.join(d, f))
                            except Exception as e:
                                print(f"[SPICE] Failed to delete temporary file {f}: {e}")
                except Exception:
                    pass

    def _load_settings(self):
        if os.path.exists(self.settings_path):
            try:
                import json
                with open(self.settings_path, 'r') as f:
                    saved = json.load(f)
                    for k, v in saved.items():
                        self.enabled_kernels[k] = v
                self.settings_initialized = True
            except:
                pass

    def save_settings(self):
        import json
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(self.enabled_kernels, f, indent=4)
            self.settings_initialized = True
            self.kernels_loaded = False
        except:
            pass

    def _ensure_dirs(self):
        os.makedirs(self.KERNEL_DIR, exist_ok=True)
        os.makedirs(self.DEFAULT_KERNEL_DIR, exist_ok=True)
        os.makedirs(self.ADDITIONAL_KERNEL_DIR, exist_ok=True)

    def find_kernel_path(self, filename):
        """
        Locates a kernel file by checking:
        1. data/kernels/default/<filename>
        2. data/kernels/additional/<filename>
        3. data/kernels/<filename>
        Returns the absolute path if found, or None.
        """
        candidates = [
            os.path.join(self.DEFAULT_KERNEL_DIR, filename),
            os.path.join(self.ADDITIONAL_KERNEL_DIR, filename),
            os.path.join(self.KERNEL_DIR, filename)
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def list_additional_kernels(self):
        """
        Scans data/kernels/additional/ and data/kernels/ for any .bsp files
        not in DEFAULT_KERNELS.
        Returns a list of dicts: [{'name': fname, 'path': full_p, 'size_kb': size_kb, 'enabled': bool}].
        """
        results = []
        seen = set()

        search_dirs = [self.ADDITIONAL_KERNEL_DIR, self.KERNEL_DIR]
        for d in search_dirs:
            if not os.path.exists(d):
                continue
            for item in sorted(os.listdir(d)):
                full_p = os.path.join(d, item)
                if not os.path.isfile(full_p):
                    continue
                if not item.lower().endswith(".bsp"):
                    continue
                if item in self.DEFAULT_KERNELS:
                    continue
                if item.endswith(".part"):
                    continue
                if item in seen:
                    continue
                seen.add(item)
                size_kb = os.path.getsize(full_p) / 1024.0
                results.append({
                    "name": item,
                    "path": os.path.abspath(full_p),
                    "size_kb": size_kb,
                    "enabled": self.enabled_kernels.get(item, False)
                })
        return results

    def add_additional_kernel(self, source_path, copy_file=True):
        """
        Adds an external .bsp kernel file into data/kernels/additional/ and marks it as enabled.
        Returns the destination path.
        """
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Kernel file not found: {source_path}")
        
        fname = os.path.basename(source_path)
        dest_path = os.path.join(self.ADDITIONAL_KERNEL_DIR, fname)
        
        if copy_file and os.path.abspath(source_path) != os.path.abspath(dest_path):
            import shutil
            shutil.copy2(source_path, dest_path)
            
        self.enabled_kernels[fname] = True
        self.save_settings()
        self.kernels_loaded = False
        return dest_path

    def check_missing_kernels(self):
        """Returns a list of kernel filenames that need to be downloaded."""
        missing = []
        for filename in self.DEFAULT_KERNELS.keys():
            if not self.enabled_kernels.get(filename, False):
                continue
            path = self.find_kernel_path(filename)
            if not path or not os.path.exists(path):
                missing.append(filename)
        return missing

    def download_kernels_async(self, on_complete=None):
        """Start a background thread to download missing default kernels."""
        if self.is_downloading:
            return
            
        missing = self.check_missing_kernels()
        if not missing:
            self.download_progress = 1.0
            self.download_status = "All kernels present."
            self.load_kernels()
            if on_complete:
                on_complete()
            return

        self.is_downloading = True
        self.cancel_requested = False
        self.download_error = None
        self.download_progress = 0.0
        self.download_bytes_current = 0
        self.download_bytes_total = 0
        self.download_speed_str = "Calculating speed..."
        self.download_speed_bytes_per_sec = 0.0
        
        def _download_thread():
            import time
            total_files = len(missing)
            self._cleanup_partial_downloads()
            
            for i, filename in enumerate(missing):
                if self.cancel_requested:
                    break
                    
                url = self.DEFAULT_KERNELS[filename]
                path = os.path.join(self.DEFAULT_KERNEL_DIR, filename)
                part_path = path + ".part"
                self.current_download_file = filename
                self.download_status = f"Downloading {filename} ({i+1}/{total_files})..."
                
                if filename == "artemis2.bsp":
                    self.current_download_file = filename
                    self.download_status = f"Generating {filename} from JPL Horizons ({i+1}/{total_files})..."
                    try:
                        from scripts.fetch_artemis2_kernel import generate_artemis2_spk
                        def _artemis_hook(frac, msg):
                            if self.cancel_requested:
                                raise InterruptedError("Download cancelled by user.")
                            self.download_progress = (i + frac) / total_files
                            self.download_status = f"Artemis II: {msg}"
                        ok, msg, _ = generate_artemis2_spk(output_dir=self.ADDITIONAL_KERNEL_DIR, progress_callback=_artemis_hook)
                        if not ok:
                            raise RuntimeError(msg)
                    except Exception as e:
                        if self.cancel_requested:
                            self.download_status = "Download cancelled."
                        else:
                            print(f"[SPICE] Failed to generate {filename}: {e}")
                            self.download_status = f"Error generating {filename}"
                            self.download_error = str(e)
                        self.is_downloading = False
                        return
                    continue

                start_time = time.time()
                last_sample_time = [start_time]
                last_sample_bytes = [0]
                
                try:
                    def _hook(count, block_size, total_size):
                        if self.cancel_requested:
                            raise InterruptedError("Download cancelled by user.")
                        cur_bytes = count * block_size
                        self.download_bytes_current = cur_bytes
                        self.download_bytes_total = total_size
                        
                        now = time.time()
                        dt = now - last_sample_time[0]
                        if dt >= 0.4:
                            speed_bps = (cur_bytes - last_sample_bytes[0]) / dt
                            self.download_speed_bytes_per_sec = speed_bps
                            if speed_bps >= 1024 * 1024:
                                self.download_speed_str = f"{speed_bps / (1024 * 1024):.2f} MB/s"
                            else:
                                self.download_speed_str = f"{max(0.0, speed_bps / 1024):.1f} KB/s"
                            last_sample_time[0] = now
                            last_sample_bytes[0] = cur_bytes

                        if total_size > 0:
                            file_prog = min(1.0, cur_bytes / total_size)
                            self.download_progress = (i + file_prog) / total_files
                            
                    urllib.request.urlretrieve(url, part_path, reporthook=_hook)
                    
                    if self.cancel_requested:
                        if os.path.exists(part_path):
                            try: os.remove(part_path)
                            except Exception: pass
                        break
                        
                    if os.path.exists(path):
                        try: os.remove(path)
                        except Exception: pass
                    os.rename(part_path, path)
                    
                except Exception as e:
                    if os.path.exists(part_path):
                        try: os.remove(part_path)
                        except Exception: pass
                    self._cleanup_partial_downloads()
                    self.is_downloading = False
                    self.download_speed_str = ""
                    if self.cancel_requested:
                        self.download_status = "Download cancelled."
                    else:
                        print(f"[SPICE] Failed to download {filename}: {e}")
                        self.download_status = f"Error downloading {filename}"
                        self.download_error = str(e)
                    return

            self._cleanup_partial_downloads()
            self.is_downloading = False
            self.current_download_file = None
            self.download_speed_str = ""
            
            if self.cancel_requested:
                self.download_status = "Download cancelled."
                return

            self.download_progress = 1.0
            self.download_status = "Downloads complete. Loading kernels..."
            self.load_kernels()
            self.download_status = "Ready"
            if on_complete:
                on_complete()

        threading.Thread(target=_download_thread, daemon=True).start()

    def load_kernels(self, force=False):
        """Furnsh all available enabled kernels across default, additional, and root kernel dirs."""
        if self.kernels_loaded and not force:
            return
            
        with SPICE_LOCK:
            try:
                # Reset CSPICE error status and traceback stack
                try:
                    spice.reset()
                except Exception:
                    pass
                # Clear any previously loaded kernels
                spice.kclear()
                
                loaded_paths = set()

                # 1. Load enabled default kernels
                for k_name in self.DEFAULT_KERNELS.keys():
                    if not self.enabled_kernels.get(k_name, False):
                        continue
                    k_path = self.find_kernel_path(k_name)
                    if k_path and os.path.exists(k_path) and k_path not in loaded_paths:
                        spice.furnsh(k_path)
                        loaded_paths.add(k_path)
                
                # 2. Load enabled additional kernels
                for k_name, is_enabled in self.enabled_kernels.items():
                    if not is_enabled or k_name in self.DEFAULT_KERNELS:
                        continue
                    k_path = self.find_kernel_path(k_name)
                    if k_path and os.path.exists(k_path) and k_path not in loaded_paths:
                        spice.furnsh(k_path)
                        loaded_paths.add(k_path)

                self.kernels_loaded = True
                
                # Discover bodies that aren't hardcoded
                self._discover_bodies()
                    
            except Exception as e:
                print(f"[SPICE] Error loading kernels: {e}")
                self.kernels_loaded = False

    def _discover_bodies(self):
        """Scans all loaded SPK kernels and adds any missing bodies to SPICE_BODIES."""
        if not self.kernels_loaded:
            return
            
        scanned_paths = set()
        kernels_to_scan = []
        for k_name, is_enabled in self.enabled_kernels.items():
            if not is_enabled:
                continue
            if not k_name.endswith('.bsp'):
                continue
            k_path = self.find_kernel_path(k_name)
            if k_path and os.path.exists(k_path) and k_path not in scanned_paths:
                kernels_to_scan.append((k_name, k_path))
                scanned_paths.add(k_path)
                
        for k_name, k_path in kernels_to_scan:
            try:
                cell = spice.spkobj(k_path)
                for spid in cell:
                    # Skip inner barycenters and outer CoMs from being added as separate bodies
                    if spid in [1, 2, 3, 4, 599, 699, 799, 899, 999]:
                        continue
                        
                    # If already in SPICE_BODIES, ensure mission range is populated if spacecraft
                    if spid in self.SPICE_BODIES:
                        if self.SPICE_BODIES[spid].get("is_spacecraft") and "mission_start_utc" not in self.SPICE_BODIES[spid]:
                            try:
                                cov = spice.spkcov(k_path, spid)
                                if len(cov) >= 2:
                                    self.SPICE_BODIES[spid]["mission_start_utc"] = spice.et2utc(cov[0], 'ISOC', 0)
                                    self.SPICE_BODIES[spid]["mission_end_utc"] = spice.et2utc(cov[1], 'ISOC', 0)
                            except Exception:
                                pass
                        continue
                        
                    try:
                        name = spice.bodc2n(spid)
                        name = name.title() if name.isupper() else name
                    except Exception:
                        if spid < 0:
                            name = f"Spacecraft ({spid})"
                        else:
                            prefix = str(spid)[0]
                            if prefix == '5': name = f"Jovian Moon ({spid})"
                            elif prefix == '6': name = f"Saturnian Moon ({spid})"
                            elif prefix == '7': name = f"Uranian Moon ({spid})"
                            elif prefix == '8': name = f"Neptunian Moon ({spid})"
                            else: name = f"Asteroid ({spid})"

                    mission_start = None
                    mission_end = None
                    try:
                        cov = spice.spkcov(k_path, spid)
                        if len(cov) >= 2:
                            mission_start = spice.et2utc(cov[0], 'ISOC', 0)
                            mission_end = spice.et2utc(cov[1], 'ISOC', 0)
                    except Exception:
                        pass

                    parent_name = "Sun"
                    if spid < 0 and mission_start:
                        # Auto-detect parent body from proximity at mission start
                        try:
                            start_et = spice.str2et(mission_start)
                            st_body, _ = spice.spkgeo(spid, start_et, 'ECLIPJ2000', 0)
                            pos_b = np.array(st_body[:3])
                            best_p = "Sun"
                            min_d = float("inf")
                            for pid, pname in [(399, "Earth"), (6, "Saturn"), (5, "Jupiter"), (499, "Mars"), (7, "Uranus"), (8, "Neptune")]:
                                try:
                                    st_p, _ = spice.spkgeo(pid, start_et, 'ECLIPJ2000', 0)
                                    d = np.linalg.norm(pos_b - np.array(st_p[:3]))
                                    if d < 5e7 and d < min_d:  # within ~0.33 AU
                                        min_d = d
                                        best_p = pname
                                except Exception:
                                    pass
                            parent_name = best_p
                        except Exception:
                            pass
                        
                    body_entry = {
                        "name": name,
                        "type": "Spacecraft" if spid < 0 else "Moon",
                        "color": "#79c0ff" if spid < 0 else "#a0a0a0",
                        "radius_km": 0.010 if spid < 0 else 10.0,
                        "mass_kg": 26000.0 if spid < 0 else 1e15,
                        "is_spacecraft": (spid < 0),
                        "parentId": parent_name
                    }
                    if mission_start:
                        body_entry["mission_start_utc"] = mission_start
                        body_entry["mission_end_utc"] = mission_end
                        
                    self.SPICE_BODIES[spid] = body_entry
            except Exception as e:
                print(f"[SPICE] Error discovering bodies in {k_name}: {e}")

    def datetime_to_et(self, dt):
        """Convert a Python datetime or ISO string to SPICE Ephemeris Time (ET)."""
        if not self.kernels_loaded:
            return 0.0
        with SPICE_LOCK:
            if isinstance(dt, str):
                try:
                    return float(spice.str2et(dt))
                except Exception as e:
                    print(f"[SPICE] Error parsing time string {dt}: {e}")
                    return 0.0
            # Convert datetime to ISO string compatible with SPICE
            iso_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
            try:
                return float(spice.str2et(iso_str))
            except Exception as e:
                print(f"[SPICE] Error parsing time {iso_str}: {e}")
                return 0.0

    def get_body_state(self, body_id, et):
        """
        Get state vector for a specific body at Ephemeris Time (ET).
        Returns state relative to Solar System Barycenter (0) in AU and AU/day.
        """
        if not self.kernels_loaded:
            return None
            
        with SPICE_LOCK:
            try:
                # SPICE spkgeo returns (state, light_time). State is [x, y, z, vx, vy, vz] in km and km/s.
                # Reference frame 'ECLIPJ2000', observer is Solar System Barycenter (0).
                state, _ = spice.spkgeo(body_id, et, 'ECLIPJ2000', 0)
            except Exception as e:
                # Fallback to barycenter for planets if specific body fails (e.g. Mars 499 kernel expires)
                if body_id > 100 and body_id < 1000 and body_id % 100 == 99:
                    try:
                        state, _ = spice.spkgeo(body_id // 100, et, 'ECLIPJ2000', 0)
                    except:
                        return None
                else:
                    return None
        
        # SPICE standard frame is J2000 (equatorial), but we requested ECLIPJ2000. 
        # If the application uses ecliptic or a different mapping (like y-up),
        # we need to rotate it. Assuming standard +Z up equatorial for now, or match existing app.
        # Let's map it raw and apply scaling.
        
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        
        return {
            "pos": np.array([x, y, z]) * self.KM_TO_AU,
            # Convert km/s to AU/yr
            "vel": np.array([vx, vy, vz]) * self.KM_TO_AU * self.SEC_TO_YR
        }

    def get_all_states(self, et):
        """Get states for all known SPICE bodies at a specific ET."""
        states = {}
        for body_id, body_info in self.SPICE_BODIES.items():
            actual_id = body_id
            state = None
            
            # For outer planet barycenters (5..9), try fetching the physical center of mass (599..999) first
            if body_id in [5, 6, 7, 8, 9]:
                com_id = body_id * 100 + 99
                state = self.get_body_state(com_id, et)
                if state:
                    actual_id = com_id
                    
            if not state:
                state = self.get_body_state(body_id, et)
                
            sample_et = et
            # If body is a spacecraft with a finite mission window and et is outside that window,
            # query its state at mission start so build_ephemeris_system has an initial state.
            if not state and body_info.get("is_spacecraft") and "mission_start_utc" in body_info:
                try:
                    start_et = spice.str2et(body_info["mission_start_utc"]) + 600.0
                    state = self.get_body_state(body_id, start_et)
                    if state:
                        sample_et = start_et
                except Exception:
                    pass

            if state:
                states[actual_id] = {
                    "info": body_info,
                    "state": state,
                    "sample_et": sample_et
                }
        return states

    def get_body_mapping(self, bodies_data, test_et):
        """Precomputes a list of SPICE IDs for a list of bodies_data dicts."""
        mapping = []
        for body in bodies_data:
            if "spice_id" in body and body["spice_id"] is not None:
                mapping.append(int(body["spice_id"]))
                continue
            b_name = body["name"]
            found_id = None
            for sp_id, info in self.SPICE_BODIES.items():
                if info["name"].upper() == b_name.upper():
                    if sp_id in [5, 6, 7, 8, 9]:
                        com_id = sp_id * 100 + 99
                        try:
                            # Test if com_id is available
                            spice.spkgeo(com_id, test_et, 'ECLIPJ2000', 0)
                            found_id = com_id
                            break
                        except:
                            pass
                    found_id = sp_id
                    break
            if found_id is None and self.kernels_loaded:
                try:
                    found_id = spice.bodn2c(b_name)
                except Exception:
                    pass
            mapping.append(found_id)
        return mapping

    def populate_states_fast(self, et, mapping, pos_out, vel_out, valid_out=None):
        """Fast method to populate pos/vel arrays for mapped SPICE bodies."""
        if not self.kernels_loaded:
            return
            
        km_to_au = self.KM_TO_AU
        sec_to_yr = self.SEC_TO_YR
        
        max_idx = min(len(mapping), pos_out.shape[0])
        if valid_out is not None:
            max_idx = min(max_idx, valid_out.shape[0])
            
        with SPICE_LOCK:
            for idx in range(max_idx):
                sp_id = mapping[idx]
                if sp_id is not None:
                    try:
                        state, _ = spice.spkgeo(sp_id, et, 'ECLIPJ2000', 0)
                    except:
                        # Fallback to barycenter if needed (e.g. Mars 499 fails past 2050)
                        if sp_id > 100 and sp_id < 1000 and sp_id % 100 == 99:
                            try:
                                state, _ = spice.spkgeo(sp_id // 100, et, 'ECLIPJ2000', 0)
                            except:
                                if valid_out is not None: valid_out[idx] = False
                                continue
                        else:
                            if valid_out is not None: valid_out[idx] = False
                            continue
                    
                    if valid_out is not None:
                        valid_out[idx] = True
                    
                    # Swap coordinates: X=x, Y=z, Z=-y
                    pos_out[idx, 0] = state[0] * km_to_au
                    pos_out[idx, 1] = state[2] * km_to_au
                    pos_out[idx, 2] = -state[1] * km_to_au
                    vel_out[idx, 0] = state[3] * km_to_au * sec_to_yr
                    vel_out[idx, 1] = state[5] * km_to_au * sec_to_yr
                    vel_out[idx, 2] = -state[4] * km_to_au * sec_to_yr

    def _get_body_properties(self, body_id, fallback_mass_kg, fallback_radius_km):
        """Extracts exact GM and radius from SPICE PCK kernels if available."""
        mass_sun = fallback_mass_kg / 1.98847e30
        radius_sun = fallback_radius_km / 696340.0
        
        try:
            _, gm_sun_vals = spice.bodvrd("10", "GM", 1)
            gm_sun = gm_sun_vals[0]
        except:
            gm_sun = 1.3271244e11
            
        gm_body = None
        try:
            _, gm = spice.bodvrd(str(body_id), "GM", 1)
            gm_body = gm[0]
        except:
            # Fallback to barycenter if CoM fails or vice versa
            if body_id > 100 and body_id % 100 == 99:
                try:
                    _, gm = spice.bodvrd(str(body_id // 100), "GM", 1)
                    gm_body = gm[0]
                except:
                    pass
            elif body_id < 100:
                try:
                    _, gm = spice.bodvrd(str(body_id * 100 + 99), "GM", 1)
                    gm_body = gm[0]
                except:
                    pass
                    
        if gm_body is not None:
            mass_sun = gm_body / gm_sun
            
        try:
            _, radii = spice.bodvrd(str(body_id), "RADII", 3)
            radius_sun = radii[0] / 696340.0
        except:
            if body_id > 100 and body_id % 100 == 99:
                try:
                    _, radii = spice.bodvrd(str(body_id // 100), "RADII", 3)
                    radius_sun = radii[0] / 696340.0
                except:
                    pass
                    
        return mass_sun, radius_sun

    def build_ephemeris_system(self, et, template_bodies=None):
        """Builds a system.json compatible list of bodies from current SPICE states."""
        states = self.get_all_states(et)
        bodies_data = []
        
        # If template bodies are provided, use them to preserve cosmetic data
        # and only update the 'sv' property for matching bodies
        if template_bodies:
            import copy
            for body in template_bodies:
                b_name = body["name"]
                
                # Find matching SPICE state
                matching_state = None
                matching_id = None
                for sp_id, data in states.items():
                    if data["info"]["name"].upper() == b_name.upper():
                        matching_state = data
                        matching_id = sp_id
                        break
                
                if matching_state:
                    body_copy = copy.deepcopy(body)
                    parent_name = body_copy.get("parentId")
                    
                    if parent_name == "Sun":
                        sun_state = states.get(10)
                        if sun_state and matching_id != 10:
                            pos_rel = matching_state["state"]["pos"] - sun_state["state"]["pos"]
                            vel_rel = matching_state["state"]["vel"] - sun_state["state"]["vel"]
                        else:
                            pos_rel = matching_state["state"]["pos"]
                            vel_rel = matching_state["state"]["vel"]
                    elif parent_name:
                        # Try to find parent state
                        parent_state = None
                        for sp_id, data in states.items():
                            if data["info"]["name"] == parent_name:
                                parent_state = data
                                break
                        if parent_state and matching_id != 10:
                            pos_rel = matching_state["state"]["pos"] - parent_state["state"]["pos"]
                            vel_rel = matching_state["state"]["vel"] - parent_state["state"]["vel"]
                        else:
                            # Fallback to Sun if parent is missing
                            sun_state = states.get(10)
                            if sun_state and matching_id != 10:
                                pos_rel = matching_state["state"]["pos"] - sun_state["state"]["pos"]
                                vel_rel = matching_state["state"]["vel"] - sun_state["state"]["vel"]
                            else:
                                pos_rel = matching_state["state"]["pos"]
                                vel_rel = matching_state["state"]["vel"]
                    else:
                        pos_rel = matching_state["state"]["pos"]
                        vel_rel = matching_state["state"]["vel"]
                                
                    body_copy["spice_id"] = matching_id
                    body_copy["sv"] = {
                        "x": pos_rel[0], "y": pos_rel[1], "z": pos_rel[2],
                        "vx": vel_rel[0], "vy": vel_rel[1], "vz": vel_rel[2]
                    }
                    bodies_data.append(body_copy)
                    
            # Add any SPICE bodies that weren't in the template
            template_names = {b["name"].upper() for b in bodies_data}
            
            if "SUN" not in template_names and 10 in states:
                sun_data = states[10]
                m_sun, r_sun = self._get_body_properties(10, sun_data["info"]["mass_kg"], sun_data["info"]["radius_km"])
                bodies_data.append({
                    "name": sun_data["info"]["name"],
                    "type": sun_data["info"]["type"],
                    "spice_id": 10,
                    "m": m_sun,
                    "r": r_sun,
                    "color": sun_data["info"]["color"],
                    "is_root": True,
                    "sv": {"x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
                })
                template_names.add("SUN")
                
            for body_id, data in states.items():
                if body_id == 10 or data["info"]["name"].upper() in template_names:
                    continue
                    
                is_planet = (body_id in [1, 2, 3, 4, 5, 6, 7, 8, 9] or (100 < body_id < 1000 and body_id % 100 == 99))
                parent_id = 10
                parent_name = "Sun"
                if data["info"].get("parentId"):
                    p_name = data["info"]["parentId"]
                    parent_name = p_name
                    for sp_id, pdata in states.items():
                        if pdata["info"]["name"].upper() == p_name.upper():
                            parent_id = sp_id
                            break
                elif not is_planet and 100 < body_id < 1000:
                    planet_id = body_id // 100
                    if planet_id == 3: planet_id = 399
                    com_id = planet_id * 100 + 99
                    if planet_id in states:
                        parent_id = planet_id
                        parent_name = states[planet_id]["info"]["name"]
                    elif com_id in states:
                        parent_id = com_id
                        parent_name = states[com_id]["info"]["name"]
                
                parent_state = states.get(parent_id)
                sample_t = data.get("sample_et", et)
                if parent_state and body_id != 10:
                    if abs(sample_t - et) > 1.0 and parent_id != 10:
                        # Spacecraft sampled at mission start: evaluate parent at that same epoch
                        p_st = self.get_body_state(parent_id, sample_t)
                        if p_st:
                            pos_rel = data["state"]["pos"] - p_st["pos"]
                            vel_rel = data["state"]["vel"] - p_st["vel"]
                        else:
                            pos_rel = data["state"]["pos"] - parent_state["state"]["pos"]
                            vel_rel = data["state"]["vel"] - parent_state["state"]["vel"]
                    else:
                        pos_rel = data["state"]["pos"] - parent_state["state"]["pos"]
                        vel_rel = data["state"]["vel"] - parent_state["state"]["vel"]
                else:
                    pos_rel = data["state"]["pos"]
                    vel_rel = data["state"]["vel"]
                    
                m_sun, r_sun = self._get_body_properties(body_id, data["info"]["mass_kg"], data["info"]["radius_km"])
                body_dict = {
                    "name": data["info"]["name"],
                    "type": data["info"]["type"],
                    "spice_id": body_id,
                    "m": m_sun,
                    "r": r_sun,
                    "color": data["info"]["color"],
                    "parentId": parent_name,
                    "sv": {
                        "x": pos_rel[0], "y": pos_rel[1], "z": pos_rel[2],
                        "vx": vel_rel[0], "vy": vel_rel[1], "vz": vel_rel[2]
                    }
                }
                if data["info"].get("is_spacecraft"):
                    body_dict["is_spacecraft"] = True
                if "mission_start_utc" in data["info"]:
                    body_dict["mission_start_utc"] = data["info"]["mission_start_utc"]
                if "mission_end_utc" in data["info"]:
                    body_dict["mission_end_utc"] = data["info"]["mission_end_utc"]
                bodies_data.append(body_dict)
            return bodies_data

        # Fallback: Build entirely from SPICE states if no template provided
        if 10 in states:
            sun_data = states[10]
            m_sun, r_sun = self._get_body_properties(10, sun_data["info"]["mass_kg"], sun_data["info"]["radius_km"])
            bodies_data.append({
                "name": sun_data["info"]["name"],
                "type": sun_data["info"]["type"],
                "spice_id": 10,
                "m": m_sun,
                "r": r_sun,
                "color": sun_data["info"]["color"],
                "is_root": True,
                "sv": {"x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
            })
            
        for body_id, data in states.items():
            if body_id == 10:
                continue
                
            is_planet = (body_id in [1, 2, 3, 4, 5, 6, 7, 8, 9] or (100 < body_id < 1000 and body_id % 100 == 99))
            parent_id = 10
            parent_name = "Sun"
            if data["info"].get("parentId"):
                p_name = data["info"]["parentId"]
                parent_name = p_name
                for sp_id, pdata in states.items():
                    if pdata["info"]["name"].upper() == p_name.upper():
                        parent_id = sp_id
                        break
            elif not is_planet and 100 < body_id < 1000:
                planet_id = body_id // 100
                if planet_id == 3: planet_id = 399 # Earth
                com_id = planet_id * 100 + 99
                if planet_id in states:
                    parent_id = planet_id
                    parent_name = states[planet_id]["info"]["name"]
                elif com_id in states:
                    parent_id = com_id
                    parent_name = states[com_id]["info"]["name"]
            
            parent_state = states.get(parent_id)
            sample_t = data.get("sample_et", et)
            if parent_state and body_id != 10:
                if abs(sample_t - et) > 1.0 and parent_id != 10:
                    p_st = self.get_body_state(parent_id, sample_t)
                    if p_st:
                        pos_rel = data["state"]["pos"] - p_st["pos"]
                        vel_rel = data["state"]["vel"] - p_st["vel"]
                    else:
                        pos_rel = data["state"]["pos"] - parent_state["state"]["pos"]
                        vel_rel = data["state"]["vel"] - parent_state["state"]["vel"]
                else:
                    pos_rel = data["state"]["pos"] - parent_state["state"]["pos"]
                    vel_rel = data["state"]["vel"] - parent_state["state"]["vel"]
            else:
                pos_rel = data["state"]["pos"]
                vel_rel = data["state"]["vel"]
                
            m_sun, r_sun = self._get_body_properties(body_id, data["info"]["mass_kg"], data["info"]["radius_km"])
            body_dict = {
                "name": data["info"]["name"],
                "type": data["info"]["type"],
                "spice_id": body_id,
                "m": m_sun,
                "r": r_sun,
                "color": data["info"]["color"],
                "parentId": parent_name,
                "sv": {
                    "x": pos_rel[0], "y": pos_rel[1], "z": pos_rel[2],
                    "vx": vel_rel[0], "vy": vel_rel[1], "vz": vel_rel[2]
                }
            }
            if data["info"].get("is_spacecraft"):
                body_dict["is_spacecraft"] = True
            if "mission_start_utc" in data["info"]:
                body_dict["mission_start_utc"] = data["info"]["mission_start_utc"]
            if "mission_end_utc" in data["info"]:
                body_dict["mission_end_utc"] = data["info"]["mission_end_utc"]
            bodies_data.append(body_dict)
            
        return bodies_data

    def _sample_adaptive_et(self, body_id, observer_id, et_start, et_end, num_samples):
        """
        Computes sampling timestamps spaced adaptively by arc length and orbital curvature.
        Prevents over-clustering at apoapsis and eliminates coarse polygon chords at periapsis.
        """
        if et_start >= et_end or num_samples <= 2:
            return np.linspace(et_start, et_end, max(2, num_samples))

        pilot_n = min(80, max(24, num_samples // 30))
        pilot_et = np.linspace(et_start, et_end, pilot_n)
        pilot_pts = np.empty((pilot_n, 3), dtype=np.float64)

        with SPICE_LOCK:
            for i, et in enumerate(pilot_et):
                try:
                    st, _ = spice.spkgeo(body_id, et, 'ECLIPJ2000', observer_id)
                except Exception:
                    try:
                        st, _ = spice.spkgeo(body_id, et, 'ECLIPJ2000', 0)
                    except Exception:
                        st = np.zeros(6)
                pilot_pts[i, 0] = st[0]
                pilot_pts[i, 1] = st[1]
                pilot_pts[i, 2] = st[2]

        diffs = np.diff(pilot_pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        r_mags = np.linalg.norm(pilot_pts[:-1], axis=1)
        median_r = float(np.median(r_mags))
        if median_r > 1.0:
            dtheta = dists / np.maximum(r_mags, 1e-3)
            weights = dists + dtheta * (median_r * 0.5)
        else:
            weights = dists

        min_w = float(weights.mean()) * 0.05 if len(weights) > 0 else 1.0
        weights = np.maximum(weights, max(1e-9, min_w))

        cum_w = np.insert(np.cumsum(weights), 0, 0.0)
        total_w = cum_w[-1]

        if total_w > 1e-6:
            target_w = np.linspace(0.0, total_w, num_samples)
            et_samples = np.interp(target_w, cum_w, pilot_et)
        else:
            et_samples = np.linspace(et_start, et_end, num_samples)

        return et_samples

    def get_trajectory_polyline(self, body_id=-1024, observer_id=None, num_samples=5000, return_times=False, et_start=None, et_end=None):
        """
        Samples the 3D trajectory of a body relative to an observer.
        If et_start and et_end are specified, samples across that specific window (clamped to mission coverage).
        Otherwise samples across the body's entire mission coverage.
        Uses arc-length and curvature adaptive spacing to ensure smooth curves.
        Returns (N, 3) float32 numpy array in engine render frame:
        X = x * KM_TO_AU, Y = z * KM_TO_AU, Z = -y * KM_TO_AU.
        If return_times is True, returns (points, et_samples, (cov_start, cov_end)).
        """
        if not self.kernels_loaded:
            return None
        info = self.SPICE_BODIES.get(body_id)
        with SPICE_LOCK:
            try:
                cov_start = None
                cov_end = None
                if info and "mission_start_utc" in info and "mission_end_utc" in info:
                    cov_start = spice.str2et(info["mission_start_utc"])
                    cov_end = spice.str2et(info["mission_end_utc"])
                else:
                    found_cov = None
                    for k_name, is_en in self.enabled_kernels.items():
                        if is_en and k_name.endswith('.bsp'):
                            kp = self.find_kernel_path(k_name)
                            if kp and os.path.exists(kp):
                                try:
                                    cov = spice.spkcov(kp, body_id)
                                    if len(cov) >= 2:
                                        found_cov = (float(cov[0]), float(cov[-1]))
                                        break
                                except Exception:
                                    pass
                    if not found_cov and getattr(self, "active_ephem_bsp", None):
                        absp = self.active_ephem_bsp
                        if os.path.isfile(absp):
                            try:
                                cov = spice.spkcov(absp, body_id)
                                if len(cov) >= 2:
                                    found_cov = (float(cov[0]), float(cov[-1]))
                            except Exception:
                                pass
                    if found_cov:
                        cov_start, cov_end = found_cov

                if cov_start is None or cov_end is None:
                    return None

                sample_start = cov_start if et_start is None else max(cov_start, float(et_start))
                sample_end = cov_end if et_end is None else min(cov_end, float(et_end))
                if sample_start >= sample_end:
                    return None

                if observer_id is None:
                    parent_name = info.get("parentId", "Sun") if info else "Sun"
                    name_to_id = {"Earth": 399, "Moon": 301, "Mars": 499, "Jupiter": 5, "Saturn": 6, "Uranus": 7, "Neptune": 8, "Sun": 0}
                    observer_id = name_to_id.get(parent_name, 0)

                et_samples = self._sample_adaptive_et(body_id, observer_id, sample_start, sample_end, num_samples)
                points = np.empty((num_samples, 3), dtype=np.float32)
                km_to_au = float(self.KM_TO_AU)
                for i, et in enumerate(et_samples):
                    try:
                        st, _ = spice.spkgeo(body_id, et, 'ECLIPJ2000', observer_id)
                    except Exception:
                        try:
                            st, _ = spice.spkgeo(body_id, et, 'ECLIPJ2000', 0)
                        except Exception:
                            st = np.zeros(6)
                    # Engine render frame swap: X=x, Y=z, Z=-y
                    points[i, 0] = float(st[0] * km_to_au)
                    points[i, 1] = float(st[2] * km_to_au)
                    points[i, 2] = float(-st[1] * km_to_au)

                if return_times:
                    return points, et_samples, (cov_start, cov_end)
                return points
            except Exception as e:
                print(f"[SPICE] Failed to sample trajectory polyline for {body_id}: {e}")
                return None

    def list_available_bsp_files(self):
        """
        Discovers all available .bsp files in exports/ephemeris/ and data/kernels/.
        Returns a list of dicts: {"path": abs_path, "name": display_name, "category": category, "size_kb": size_kb}.
        """
        results = []
        seen = set()

        # 1. Check exports/ephemeris directory
        exp_dir = get_external_path("exports", "ephemeris")
        if os.path.exists(exp_dir):
            for root, _, files in os.walk(exp_dir):
                for f in files:
                    if f.lower().endswith(".bsp"):
                        full_p = os.path.abspath(os.path.join(root, f))
                        if full_p not in seen:
                            seen.add(full_p)
                            rel = os.path.relpath(full_p, exp_dir)
                            size_kb = os.path.getsize(full_p) / 1024.0
                            results.append({
                                "path": full_p,
                                "name": rel,
                                "category": "Exported Ephemeris",
                                "size_kb": size_kb
                            })

        # 2. Check data/kernels directories (additional, default, and root)
        search_dirs = [
            (self.ADDITIONAL_KERNEL_DIR, "Additional Kernels"),
            (self.DEFAULT_KERNEL_DIR, "Default Kernels"),
            (self.KERNEL_DIR, "Kernel Library")
        ]
        for k_dir, cat_name in search_dirs:
            if os.path.exists(k_dir):
                for f in sorted(os.listdir(k_dir)):
                    full_p = os.path.abspath(os.path.join(k_dir, f))
                    if os.path.isfile(full_p) and f.lower().endswith(".bsp"):
                        if full_p not in seen:
                            seen.add(full_p)
                            size_kb = os.path.getsize(full_p) / 1024.0
                            desc = self.KERNEL_DESCRIPTIONS.get(f, f)
                            results.append({
                                "path": full_p,
                                "name": f"{f} ({desc})" if desc != f else f,
                                "category": cat_name,
                                "size_kb": size_kb
                            })

        return results

    def inspect_bsp(self, bsp_path):
        """
        Inspects an arbitrary SPK kernel (.bsp) without permanently altering engine state.
        Returns a dictionary with detected bodies, time coverage, center IDs, and sidecar info.
        """
        if not bsp_path or not os.path.isfile(bsp_path):
            return None

        PALETTE = [
            "#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
            "#39c5cf", "#f0883e", "#56d364", "#79c0ff", "#e3b341",
            "#ff7b72", "#d2a8ff", "#2ea043", "#a5d6ff", "#ffab70"
        ]

        tls_path = os.path.join(self.KERNEL_DIR, "naif0012.tls")
        with SPICE_LOCK:
            if os.path.isfile(tls_path):
                try:
                    spice.furnsh(tls_path)
                except Exception:
                    pass
            try:
                spice.furnsh(bsp_path)
            except Exception as e:
                print(f"[SPICE] Failed to furnsh {bsp_path}: {e}")
                return None

            try:
                obj_ids_cell = spice.spkobj(bsp_path)
                body_ids = [int(x) for x in obj_ids_cell]

                center_map = {}
                cov_map = {}

                # Inspect segments using DAF routines
                try:
                    handle = spice.dafopr(bsp_path)
                    spice.dafbfs(handle)
                    while spice.daffna():
                        res = spice.dafgs()
                        body, center, frame, type_id, b_epoch, e_epoch, first_addr, last_addr = spice.spkuds(res[:5])
                        body = int(body)
                        center = int(center)
                        if body not in center_map:
                            center_map[body] = center
                        if body not in cov_map:
                            cov_map[body] = [float(b_epoch), float(e_epoch)]
                        else:
                            cov_map[body][0] = min(cov_map[body][0], float(b_epoch))
                            cov_map[body][1] = max(cov_map[body][1], float(e_epoch))
                    spice.dafcls(handle)
                except Exception as e:
                    print(f"[SPICE] DAF segment inspection fallback: {e}")

                for bid in body_ids:
                    if bid not in cov_map:
                        try:
                            cov = spice.spkcov(bsp_path, bid)
                            if len(cov) >= 2:
                                cov_map[bid] = [float(cov[0]), float(cov[-1])]
                        except Exception:
                            cov_map[bid] = [0.0, 0.0]
                    if bid not in center_map:
                        center_map[bid] = 0

                # Check sidecar JSON
                sidecar_path = bsp_path + ".json"
                sidecar = None
                if os.path.isfile(sidecar_path):
                    try:
                        with open(sidecar_path, "r", encoding="utf-8") as f:
                            sidecar = json.load(f)
                    except Exception:
                        pass

                # Preserve logical body ordering
                if sidecar and "bodies" in sidecar:
                    ordered_ids = []
                    for sb in sidecar["bodies"]:
                        sbid = sb.get("spice_id")
                        if sbid in body_ids and sbid not in ordered_ids:
                            ordered_ids.append(sbid)
                    for bid in body_ids:
                        if bid not in ordered_ids:
                            ordered_ids.append(bid)
                    body_ids = ordered_ids
                else:
                    body_ids = sorted(body_ids, key=lambda x: (1 if x < 0 else 0, x))

                bodies = []
                min_et = float("inf")
                max_et = float("-inf")

                for idx, bid in enumerate(body_ids):
                    b_cov = cov_map.get(bid, [0.0, 0.0])
                    min_et = min(min_et, b_cov[0])
                    max_et = max(max_et, b_cov[1])

                    b_name = None
                    b_color = None
                    if sidecar and "bodies" in sidecar:
                        for sb in sidecar["bodies"]:
                            if sb.get("spice_id") == bid:
                                b_name = sb.get("name")
                                if "color" in sb:
                                    b_color = sb["color"]
                                break

                    if not b_name:
                        if bid in self.SPICE_BODIES:
                            b_name = self.SPICE_BODIES[bid]["name"]
                            b_color = self.SPICE_BODIES[bid].get("color")

                    if not b_name:
                        try:
                            raw_n = spice.bodc2n(bid)
                            if raw_n:
                                b_name = raw_n.title().replace("_", " ")
                        except Exception:
                            pass

                    if not b_name:
                        b_name = f"Spacecraft ({bid})" if bid < 0 else f"Object {bid}"

                    if not b_color:
                        b_color = PALETTE[idx % len(PALETTE)]

                    cid = center_map.get(bid, 0)
                    c_name = "Solar System Barycenter" if cid == 0 else f"Object {cid}"
                    if cid in self.SPICE_BODIES:
                        c_name = self.SPICE_BODIES[cid]["name"]
                    else:
                        try:
                            c_name_raw = spice.bodc2n(cid)
                            if c_name_raw:
                                c_name = c_name_raw.title().replace("_", " ")
                        except Exception:
                            pass

                    try:
                        b_start_utc = spice.et2utc(b_cov[0], 'C', 0)
                        b_end_utc = spice.et2utc(b_cov[1], 'C', 0)
                    except Exception:
                        b_start_utc = f"{b_cov[0]:.0f}s ET"
                        b_end_utc = f"{b_cov[1]:.0f}s ET"

                    bodies.append({
                        "id": bid,
                        "name": b_name,
                        "center_id": cid,
                        "center_name": c_name,
                        "color": b_color,
                        "et_range": b_cov,
                        "utc_start": b_start_utc,
                        "utc_end": b_end_utc
                    })

                if min_et == float("inf"):
                    min_et = 0.0
                    max_et = 1.0

                try:
                    utc_min = spice.et2utc(min_et, 'C', 0)
                    utc_max = spice.et2utc(max_et, 'C', 0)
                except Exception:
                    utc_min = f"{min_et:.0f}s ET"
                    utc_max = f"{max_et:.0f}s ET"

                duration_days = (max_et - min_et) / 86400.0

                return {
                    "filepath": os.path.abspath(bsp_path),
                    "filename": os.path.basename(bsp_path),
                    "body_ids": body_ids,
                    "bodies": bodies,
                    "et_range": [min_et, max_et],
                    "utc_range": [utc_min, utc_max],
                    "duration_days": duration_days,
                    "sidecar": sidecar
                }
            except Exception as e:
                print(f"[SPICE] Error inspecting {bsp_path}: {e}")
                return None

    def build_generic_bsp_system(self, bsp_path, num_samples=5000):
        """
        Loads an arbitrary SPK kernel (.bsp), samples trajectories for all bodies using adaptive spacing,
        and constructs a bodies_data bundle for playback in the engine.
        """
        info = self.inspect_bsp(bsp_path)
        if not info or not info["bodies"]:
            return None

        sidecar = info.get("sidecar")
        if sidecar and "epoch_et_seconds_past_j2000" in sidecar:
            epoch_et = float(sidecar["epoch_et_seconds_past_j2000"])
        else:
            epoch_et = float(info["et_range"][0])

        if sidecar and "time_span_years" in sidecar:
            time_span_years = [float(sidecar["time_span_years"][0]), float(sidecar["time_span_years"][1])]
            start_sim_t = time_span_years[0]
        else:
            start_sim_t = 0.0
            span_sec = max(1.0, info["et_range"][1] - info["et_range"][0])
            time_span_years = [0.0, span_sec / (86400.0 * 365.25)]

        km_to_au = float(self.KM_TO_AU)
        sec_to_yr = float(self.SEC_TO_YR)

        bodies_data = []
        trajectories = []

        def _hex_to_rgb(h):
            if not isinstance(h, str):
                return [0.7, 0.8, 1.0]
            h = h.lstrip('#')
            if len(h) == 6:
                try:
                    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
                except ValueError:
                    pass
            return [0.7, 0.8, 1.0]

        with SPICE_LOCK:
            try:
                spice.furnsh(bsp_path)
                self.kernels_loaded = True
            except Exception as e:
                print(f"[SPICE] Notice furnshing {bsp_path}: {e}")

            for body in info["bodies"]:
                bid = body["id"]
                cid = body.get("center_id", 0)
                b_cov = body["et_range"]

                # Sample trajectory polyline
                b_start = b_cov[0]
                b_end = b_cov[1]
                sample_observer = cid
                if b_start == b_end:
                    et_samples = np.array([b_start, b_start + 1.0])
                else:
                    et_samples = self._sample_adaptive_et(bid, sample_observer, b_start, b_end, max(2, num_samples))

                pts = np.empty((len(et_samples), 3), dtype=np.float32)

                # Check if sample_observer can be queried, fallback to 0 if needed
                for i, et in enumerate(et_samples):
                    try:
                        st, _ = spice.spkgeo(bid, et, 'ECLIPJ2000', sample_observer)
                    except Exception:
                        try:
                            st, _ = spice.spkgeo(bid, et, 'ECLIPJ2000', 0)
                            sample_observer = 0
                        except Exception:
                            st = np.zeros(6)
                    # Render frame swap: X=x, Y=z, Z=-y
                    pts[i, 0] = float(st[0] * km_to_au)
                    pts[i, 1] = float(st[2] * km_to_au)
                    pts[i, 2] = float(-st[1] * km_to_au)

                # Initial state at start_sim_t
                init_et = epoch_et + start_sim_t * (86400.0 * 365.25)
                init_et = max(b_start, min(b_end, init_et))

                try:
                    st0, _ = spice.spkgeo(bid, init_et, 'ECLIPJ2000', 0)
                except Exception:
                    try:
                        st0, _ = spice.spkgeo(bid, init_et, 'ECLIPJ2000', sample_observer)
                    except Exception:
                        st0 = np.zeros(6)

                # Convert to engine physics frame (AU and AU/year)
                pos0 = [float(st0[0] * km_to_au), float(st0[1] * km_to_au), float(st0[2] * km_to_au)]
                vel0 = [float(st0[3] * km_to_au * sec_to_yr), float(st0[4] * km_to_au * sec_to_yr), float(st0[5] * km_to_au * sec_to_yr)]

                b_type = "Spacecraft" if bid < 0 else ("Star" if bid == 10 else "Planet")
                radius = 0.001 if bid < 0 else (0.02 if bid == 10 else 0.005)

                b_dict = {
                    "name": body["name"],
                    "type": b_type,
                    "spice_id": bid,
                    "center_id": sample_observer,
                    "m": 0.0,
                    "r": radius,
                    "color": body["color"],
                    "is_spacecraft": (bid < 0),
                    "sv": {
                        "x": pos0[0], "y": pos0[1], "z": pos0[2],
                        "vx": vel0[0], "vy": vel0[1], "vz": vel0[2]
                    }
                }
                bodies_data.append(b_dict)

                trajectories.append({
                    "name": body["name"],
                    "spice_id": bid,
                    "center_id": sample_observer,
                    "color": _hex_to_rgb(body["color"]),
                    "points": pts,
                    "times": et_samples,
                    "et_range": b_cov
                })

        # Build simple hierarchy links
        id_to_name = {b["spice_id"]: b["name"] for b in bodies_data}
        for b in bodies_data:
            cid = b.get("center_id", 0)
            if "parentId" not in b:
                if cid != 0 and cid in id_to_name and id_to_name[cid] != b["name"]:
                    b["parentId"] = id_to_name[cid]
                elif b.get("spice_id") in self.SPICE_BODIES and self.SPICE_BODIES[b["spice_id"]].get("parentId"):
                    b["parentId"] = self.SPICE_BODIES[b["spice_id"]]["parentId"]
                elif cid != 0:
                    try:
                        c_name = spice.bodc2n(cid)
                        b["parentId"] = c_name.title() if c_name.isupper() else c_name
                    except Exception:
                        pass

        base_stem = os.path.splitext(info["filename"])[0]
        display_name = f"Ephemeris: {base_stem}"

        return {
            "display_name": display_name,
            "bsp_path": os.path.abspath(bsp_path),
            "bodies_data": bodies_data,
            "trajectories": trajectories,
            "epoch_et": epoch_et,
            "et_range": info["et_range"],
            "time_span_years": time_span_years,
            "start_sim_t": start_sim_t
        }
