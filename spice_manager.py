import os
import urllib.request
import spiceypy as spice
import numpy as np
import threading

class SpiceManager:
    """
    Manages SPICE kernels and Ephemeris Mode operations.
    Handles downloading basic kernels from NAIF and converting state vectors.
    """
    
    KERNEL_DIR = "data/kernels"
    
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
        "ura116xl.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/ura116xl.bsp",
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
        "ura116xl.bsp": "Uranus Moons (659MB)",
        "nep104.bsp": "Irregular Neptune Moons (318MB)",
        "nep105.bsp": "Major Neptune Moons (201MB)",
        "plu060.bsp": "Pluto Moons",
        "mar099s.bsp": "Mars Moons"
    }

    ESSENTIAL_KERNELS = {"naif0012.tls", "pck00010.tpc", "de440s.bsp", "jup365.bsp", "sat441.bsp", "mar099s.bsp", "plu060.bsp"}

    # Standard body mapping from SPICE IDs to recognizable names for rendering
    # SPICE IDs: 10=Sun, 1=Mercury Barycenter, 2=Venus Barycenter, 3=Earth Barycenter, etc.
    # 399=Earth, 301=Moon. Using barycenters (1-9) for outer planets is standard.
    SPICE_BODIES = {
        10: {"name": "Sun", "type": "Star", "color": "#fff5e6", "radius_km": 696340.0, "mass_kg": 1.98847e30},
        199: {"name": "Mercury", "type": "Planet", "color": "#a8a8a8", "radius_km": 2439.7, "mass_kg": 3.3011e23},
        299: {"name": "Venus", "type": "Planet", "color": "#e0c8a0", "radius_km": 6051.8, "mass_kg": 4.8675e24},
        399: {"name": "Earth", "type": "Planet", "color": "#3b5d9c", "radius_km": 6371.0, "mass_kg": 5.972e24},
        301: {"name": "Moon", "type": "Moon", "color": "#d0d0d0", "radius_km": 1737.4, "mass_kg": 7.342e22},
        499: {"name": "Mars", "type": "Planet", "color": "#c1440e", "radius_km": 3389.5, "mass_kg": 6.4171e23},
        401: {"name": "Phobos", "type": "Moon", "color": "#a0a0a0", "radius_km": 11.26, "mass_kg": 1.0659e16},
        402: {"name": "Deimos", "type": "Moon", "color": "#a0a0a0", "radius_km": 6.2, "mass_kg": 1.4762e15},
        5: {"name": "Jupiter", "type": "Planet", "color": "#c88b3a", "radius_km": 69911.0, "mass_kg": 1.8982e27},
        501: {"name": "Io", "type": "Moon", "color": "#c1b446", "radius_km": 1821.6, "mass_kg": 8.9319e22},
        502: {"name": "Europa", "type": "Moon", "color": "#9b7f59", "radius_km": 1560.8, "mass_kg": 4.7998e22},
        503: {"name": "Ganymede", "type": "Moon", "color": "#8b8173", "radius_km": 2631.2, "mass_kg": 1.4819e23},
        504: {"name": "Callisto", "type": "Moon", "color": "#5e5850", "radius_km": 2410.3, "mass_kg": 1.0759e23},
        6: {"name": "Saturn", "type": "Planet", "color": "#e3d599", "radius_km": 58232.0, "mass_kg": 5.6834e26},
        601: {"name": "Mimas", "type": "Moon", "color": "#a0a0a0", "radius_km": 198.2, "mass_kg": 3.7493e19},
        602: {"name": "Enceladus", "type": "Moon", "color": "#ffffff", "radius_km": 252.1, "mass_kg": 1.0802e20},
        603: {"name": "Tethys", "type": "Moon", "color": "#a0a0a0", "radius_km": 531.1, "mass_kg": 6.1744e20},
        604: {"name": "Dione", "type": "Moon", "color": "#a0a0a0", "radius_km": 561.4, "mass_kg": 1.0954e21},
        605: {"name": "Rhea", "type": "Moon", "color": "#a0a0a0", "radius_km": 763.8, "mass_kg": 2.3065e21},
        606: {"name": "Titan", "type": "Moon", "color": "#e0a040", "radius_km": 2574.7, "mass_kg": 1.3452e23},
        608: {"name": "Iapetus", "type": "Moon", "color": "#505050", "radius_km": 734.5, "mass_kg": 1.8056e21},
        7: {"name": "Uranus", "type": "Planet", "color": "#4b70dd", "radius_km": 25362.0, "mass_kg": 8.6810e25},
        701: {"name": "Ariel", "type": "Moon", "color": "#a0a0a0", "radius_km": 578.9, "mass_kg": 1.353e21},
        702: {"name": "Umbriel", "type": "Moon", "color": "#808080", "radius_km": 584.7, "mass_kg": 1.275e21},
        703: {"name": "Titania", "type": "Moon", "color": "#a0a0a0", "radius_km": 788.4, "mass_kg": 3.400e21},
        704: {"name": "Oberon", "type": "Moon", "color": "#808080", "radius_km": 761.4, "mass_kg": 3.076e21},
        705: {"name": "Miranda", "type": "Moon", "color": "#a0a0a0", "radius_km": 235.8, "mass_kg": 6.59e19},
        8: {"name": "Neptune", "type": "Planet", "color": "#274687", "radius_km": 24622.0, "mass_kg": 1.0241e26},
        801: {"name": "Triton", "type": "Moon", "color": "#a0a0a0", "radius_km": 1353.4, "mass_kg": 2.14e22},
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
        self.settings_path = os.path.join("data", "ephemeris_settings.json")
        self.enabled_kernels = {k: (k in self.ESSENTIAL_KERNELS) for k in self.DEFAULT_KERNELS.keys()}
        self.settings_initialized = False
        self._load_settings()

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
        self.download_progress = 0.0
        
        def _download_thread():
            total_files = len(missing)
            for i, filename in enumerate(missing):
                url = self.DEFAULT_KERNELS[filename]
                path = os.path.join(self.KERNEL_DIR, filename)
                self.download_status = f"Downloading {filename} ({i+1}/{total_files})..."
                
                try:
                    def _hook(count, block_size, total_size):
                        if total_size > 0:
                            file_prog = count * block_size / total_size
                            self.download_progress = (i + file_prog) / total_files
                            
                    urllib.request.urlretrieve(url, path, reporthook=_hook)
                except Exception as e:
                    print(f"[SPICE] Failed to download {filename}: {e}")
                    self.download_status = f"Error downloading {filename}"
                    self.is_downloading = False
                    return

            self.download_progress = 1.0
            self.download_status = "Downloads complete. Loading kernels..."
            self.load_kernels()
            self.is_downloading = False
            self.download_status = "Ready"
            if on_complete:
                on_complete()

        threading.Thread(target=_download_thread, daemon=True).start()

    def load_kernels(self):
        """Furnsh all available kernels in the KERNEL_DIR."""
        if self.kernels_loaded:
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
                        prefix = str(spid)[0]
                        if prefix == '5': name = f"Jovian Moon ({spid})"
                        elif prefix == '6': name = f"Saturnian Moon ({spid})"
                        elif prefix == '7': name = f"Uranian Moon ({spid})"
                        elif prefix == '8': name = f"Neptunian Moon ({spid})"
                        else: name = f"Asteroid ({spid})"
                        
                    self.SPICE_BODIES[spid] = {
                        "name": name,
                        "type": "Moon",
                        "color": "#a0a0a0",
                        "radius_km": 10.0,
                        "mass_kg": 1e15
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
        except Exception as e:
            # Body might not be in the loaded kernels
            return None

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
                
            if state:
                states[actual_id] = {
                    "info": body_info,
                    "state": state
                }
        return states

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
                    if data["info"]["name"] == b_name:
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
            template_names = {b["name"] for b in bodies_data}
            
            if "Sun" not in template_names and 10 in states:
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
                template_names.add("Sun")
                
            for body_id, data in states.items():
                if body_id == 10 or data["info"]["name"] in template_names:
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
                
                parent_state = states.get(parent_id)
                if parent_state and body_id != 10:
                    pos_rel = data["state"]["pos"] - parent_state["state"]["pos"]
                    vel_rel = data["state"]["vel"] - parent_state["state"]["vel"]
                else:
                    pos_rel = data["state"]["pos"]
                    vel_rel = data["state"]["vel"]
                    
                m_sun, r_sun = self._get_body_properties(body_id, data["info"]["mass_kg"], data["info"]["radius_km"])
                bodies_data.append({
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
                })
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
            
            parent_state = states.get(parent_id)
            if parent_state and body_id != 10:
                pos_rel = data["state"]["pos"] - parent_state["state"]["pos"]
                vel_rel = data["state"]["vel"] - parent_state["state"]["vel"]
            else:
                pos_rel = data["state"]["pos"]
                vel_rel = data["state"]["vel"]
                
            m_sun, r_sun = self._get_body_properties(body_id, data["info"]["mass_kg"], data["info"]["radius_km"])
            bodies_data.append({
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
            })
            
        return bodies_data
