import os
import urllib.request
import spiceypy as spice
import numpy as np
import threading

from engine.path_utils import get_external_path

class SpiceManager:
    """
    Manages SPICE kernels and Ephemeris Mode operations.
    Handles downloading basic kernels from NAIF and converting state vectors.
    """
    
    KERNEL_DIR = get_external_path("data", "kernels")
    
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
        "mar099s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/mar099s.bsp",
        "artemis2.bsp": "HORIZONS_API"
    }
    
    KERNEL_DESCRIPTIONS = {
        "naif0012.tls": "Leapseconds (Required)",
        "pck00010.tpc": "Planetary Constants (Required)",
        "de440s.bsp": "Planetary Ephemeris (Required)",
        "artemis2.bsp": "Artemis II Lunar Flyby (Orion) [144KB]",
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
        self._ensure_dirs()
        self.kernels_loaded = False
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
        """Removes all .part files in kernel directory."""
        if os.path.exists(self.KERNEL_DIR):
            try:
                for f in os.listdir(self.KERNEL_DIR):
                    if f.endswith('.part'):
                        try:
                            os.remove(os.path.join(self.KERNEL_DIR, f))
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
                        if k in self.enabled_kernels:
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

    def check_missing_kernels(self):
        """Returns a list of kernel filenames that need to be downloaded."""
        missing = []
        for filename in self.DEFAULT_KERNELS.keys():
            if not self.enabled_kernels.get(filename, False):
                continue
            path = os.path.join(self.KERNEL_DIR, filename)
            if not os.path.exists(path):
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
                path = os.path.join(self.KERNEL_DIR, filename)
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
                        ok, msg, _ = generate_artemis2_spk(output_dir=self.KERNEL_DIR, progress_callback=_artemis_hook)
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
        """Furnsh all available kernels in the KERNEL_DIR."""
        if self.kernels_loaded and not force:
            return
            
        try:
            # Clear any previously loaded kernels
            spice.kclear()
            
            for k_name in self.DEFAULT_KERNELS.keys():
                if not self.enabled_kernels.get(k_name, False):
                    continue
                k_path = os.path.join(self.KERNEL_DIR, k_name)
                if os.path.exists(k_path):
                    spice.furnsh(k_path)
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
            
        for k_name in self.DEFAULT_KERNELS.keys():
            if not self.enabled_kernels.get(k_name, False):
                continue
            if not k_name.endswith('.bsp'):
                continue
                
            k_path = os.path.join(self.KERNEL_DIR, k_name)
            if not os.path.exists(k_path):
                continue
                
            try:
                cell = spice.spkobj(k_path)
                for spid in cell:
                    # Skip inner barycenters and outer CoMs from being added as separate bodies
                    if spid in self.SPICE_BODIES or spid in [1, 2, 3, 4, 599, 699, 799, 899, 999]:
                        continue
                        
                    try:
                        name = spice.bodc2n(spid)
                    except:
                        if spid < 0:
                            name = f"Spacecraft ({spid})"
                        else:
                            prefix = str(spid)[0]
                            if prefix == '5': name = f"Jovian Moon ({spid})"
                            elif prefix == '6': name = f"Saturnian Moon ({spid})"
                            elif prefix == '7': name = f"Uranian Moon ({spid})"
                            elif prefix == '8': name = f"Neptunian Moon ({spid})"
                            else: name = f"Asteroid ({spid})"
                        
                    self.SPICE_BODIES[spid] = {
                        "name": name,
                        "type": "Spacecraft" if spid < 0 else "Moon",
                        "color": "#00ffff" if spid < 0 else "#a0a0a0",
                        "radius_km": 0.010 if spid < 0 else 10.0,
                        "mass_kg": 26000.0 if spid < 0 else 1e15,
                        "is_spacecraft": (spid < 0)
                    }
            except Exception as e:
                print(f"Error discovering bodies in {k_name}: {e}")

    def datetime_to_et(self, dt):
        """Convert a Python datetime to SPICE Ephemeris Time (ET)."""
        if not self.kernels_loaded:
            return 0.0
        # Convert datetime to ISO string compatible with SPICE
        iso_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            return spice.str2et(iso_str)
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
                
            # If body is a spacecraft with a finite mission window and et is outside that window,
            # query its state at mission start so build_ephemeris_system has an initial state.
            if not state and body_info.get("is_spacecraft") and "mission_start_utc" in body_info:
                try:
                    start_et = spice.str2et(body_info["mission_start_utc"]) + 600.0
                    state = self.get_body_state(body_id, start_et)
                except Exception:
                    pass

            if state:
                states[actual_id] = {
                    "info": body_info,
                    "state": state
                }
        return states

    def get_body_mapping(self, bodies_data, test_et):
        """Precomputes a list of SPICE IDs for a list of bodies_data dicts."""
        mapping = []
        for body in bodies_data:
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
                    
                parent_id = 10
                parent_name = "Sun"
                if body_id > 100 and body_id < 1000:
                    planet_id = body_id // 100
                    if planet_id == 3: planet_id = 399
                    
                    # Check if CoM is loaded instead of Barycenter
                    com_id = planet_id * 100 + 99
                    if com_id in states:
                        planet_id = com_id
                        
                    if planet_id in states:
                        parent_id = planet_id
                        parent_name = states[planet_id]["info"]["name"]
                elif data["info"].get("parentId"):
                    p_name = data["info"]["parentId"]
                    parent_name = p_name
                    for sp_id, pdata in states.items():
                        if pdata["info"]["name"].upper() == p_name.upper():
                            parent_id = sp_id
                            break
                
                parent_state = states.get(parent_id)
                if parent_state and body_id != 10:
                    pos_rel = data["state"]["pos"] - parent_state["state"]["pos"]
                    vel_rel = data["state"]["vel"] - parent_state["state"]["vel"]
                else:
                    pos_rel = data["state"]["pos"]
                    vel_rel = data["state"]["vel"]
                    
                m_sun, r_sun = self._get_body_properties(body_id, data["info"]["mass_kg"], data["info"]["radius_km"])
                body_dict = {
                    "name": data["info"]["name"],
                    "type": data["info"]["type"],
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
                bodies_data.append(body_dict)
            return bodies_data

        # Fallback: Build entirely from SPICE states if no template provided
        if 10 in states:
            sun_data = states[10]
            m_sun, r_sun = self._get_body_properties(10, sun_data["info"]["mass_kg"], sun_data["info"]["radius_km"])
            bodies_data.append({
                "name": sun_data["info"]["name"],
                "type": sun_data["info"]["type"],
                "m": m_sun,
                "r": r_sun,
                "color": sun_data["info"]["color"],
                "is_root": True,
                "sv": {"x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
            })
            
        for body_id, data in states.items():
            if body_id == 10:
                continue
                
            parent_id = 10
            parent_name = "Sun"
            if body_id > 100 and body_id < 1000:
                planet_id = body_id // 100
                if planet_id == 3: planet_id = 399 # Earth
                
                # Check if CoM is loaded instead of Barycenter
                com_id = planet_id * 100 + 99
                if com_id in states:
                    planet_id = com_id
                    
                if planet_id in states:
                    parent_id = planet_id
                    parent_name = states[planet_id]["info"]["name"]
            elif data["info"].get("parentId"):
                p_name = data["info"]["parentId"]
                parent_name = p_name
                for sp_id, pdata in states.items():
                    if pdata["info"]["name"].upper() == p_name.upper():
                        parent_id = sp_id
                        break
            
            parent_state = states.get(parent_id)
            if parent_state and body_id != 10:
                pos_rel = data["state"]["pos"] - parent_state["state"]["pos"]
                vel_rel = data["state"]["vel"] - parent_state["state"]["vel"]
            else:
                pos_rel = data["state"]["pos"]
                vel_rel = data["state"]["vel"]
                
            m_sun, r_sun = self._get_body_properties(body_id, data["info"]["mass_kg"], data["info"]["radius_km"])
            body_dict = {
                "name": data["info"]["name"],
                "type": data["info"]["type"],
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
            bodies_data.append(body_dict)
            
        return bodies_data

    def get_trajectory_polyline(self, body_id=-1024, observer_id=399, num_samples=1000):
        """
        Samples the 3D trajectory of a body relative to an observer (e.g. Earth 399)
        across its mission duration. Returns an (N, 3) float32 numpy array in engine render frame:
        X = x * KM_TO_AU, Y = z * KM_TO_AU, Z = -y * KM_TO_AU.
        """
        if not self.kernels_loaded:
            return None
        info = self.SPICE_BODIES.get(body_id)
        if not info or "mission_start_utc" not in info:
            return None
        try:
            et_start = spice.str2et(info["mission_start_utc"])
            et_end = spice.str2et(info["mission_end_utc"])
            et_samples = np.linspace(et_start, et_end, num_samples)
            points = np.empty((num_samples, 3), dtype=np.float32)
            km_to_au = float(self.KM_TO_AU)
            for i, et in enumerate(et_samples):
                st, _ = spice.spkgeo(body_id, et, 'ECLIPJ2000', observer_id)
                # Engine render frame swap: X=x, Y=z, Z=-y
                points[i, 0] = float(st[0] * km_to_au)
                points[i, 1] = float(st[2] * km_to_au)
                points[i, 2] = float(-st[1] * km_to_au)
            return points
        except Exception as e:
            print(f"[SPICE] Failed to sample trajectory polyline for {body_id}: {e}")
            return None
