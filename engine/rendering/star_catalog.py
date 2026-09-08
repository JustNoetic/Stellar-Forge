"""GAIA star catalog layer for Stellar-Forge.

Loads the binary produced by scripts/fetch_gaia.py (data/gaia/stars.bin) and
provides a render-only layer of real stars at their true 3D positions:

  - Positions are heliocentric, from Gaia parallaxes, expressed in the engine's
    ecliptic render frame (x, z, -y remap, see PROJECT_MAP gotchas).
  - Proper motion is applied ANALYTICALLY (p(t) = p0 + pm*t, linear approx),
    never through the physics system — GAIA stars are not Simulation bodies.
  - Brightness is physical: the packer emits each star's constant pre-scaled
    flux 10^(-0.4*M_G) * FLUX_SCALE, and the inverse-square distance falloff
    is applied per frame against the EYE in starfield.vert — stars brighten
    as you approach and dim as you recede (same 1/d^2 law as system stars).

Per-frame cost is one numpy pass (~1-2 ms @ 460k stars) packing a streamed f4
VBO of interleaved [rel_xyz, rgb, intensity]. Result is cached: repacking only
happens when the camera position or sim time actually changed.
"""

import os
import struct

import numpy as np

from numba import njit, prange

from engine.core.constants import OBLIQUITY

# Must match GAIA_BIN_DTYPE in scripts/fetch_gaia.py
DTYPE = np.dtype([
    ('ra', '<f4'),        # deg, ICRS
    ('dec', '<f4'),       # deg, ICRS
    ('plx', '<f4'),       # mas  (parallax)
    ('pmra', '<f4'),      # mas/yr (includes cos(dec), Gaia convention)
    ('pmdec', '<f4'),     # mas/yr
    ('rv', '<f4'),        # km/s (radial velocity; stored but not used yet)
    ('g', '<f4'),         # mag  (phot_g_mean_mag)
    ('bmrp', '<f4'),      # mag  (BP - RP color index)
])
MAGIC = b"GAIA1"
VERSION = 1

PC_TO_AU = 206264.80624709636   # 1 parsec in AU
MAS_TO_RAD = np.pi / (180.0 * 3600.0 * 1000.0)  # milliarcsec -> radians

# Gaia DR3 reference epoch is J2016.0; the sim's epoch is J2000. Back-propagate.
EPOCH_OFFSET_YEARS = 16.0

# HDR flux normalization: Sirius (M_G=1.43, d=2.64 pc) lands at intensity ~9.
FLUX_SCALE = 1.0e13

# BP-RP -> Teff cubic fit (Mucciarelli et al. 2021 style), clamped outside
# [0, 3]; visual accuracy at the extreme red/blue ends is not critical.
TEFF_A, TEFF_B, TEFF_C, TEFF_D = 8842.0, -4578.875, 1031.17, -96.233


class StarCatalog:
    """Render-only layer of GAIA stars at true 3D positions."""

    def __init__(self):
        self.loaded = False
        self.count = 0
        self.raw = None            # structured array (DTYPE) for tests/tools
        self.pos0 = None           # f8 (N,3) heliocentric, ecliptic render frame, AU
        self.pm_vec = None         # f8 (N,3) tangential motion, render frame, rad/yr
        self.abs_mag = None        # f4 (N,) absolute G magnitude
        self.rgb = None            # f4 (N,3) blackbody color from BP-RP
        self.dist_ly = None        # f8 (N,) distance in light years
        # pack cache
        self._last_cam = None
        self._last_t = None
        self._last_packed = None
        self.fresh = False   # True when the last pack_render produced new data

    # ------------------------------------------------------------------
    # Loading & baking
    # ------------------------------------------------------------------

    def load(self, path):
        """Load + bake the catalog. Quietly stays unloaded if file missing."""
        try:
            if not os.path.exists(path):
                print(f"[StarCatalog] No GAIA catalog at {path} — run "
                      f"'python scripts/fetch_gaia.py' to build it. Starfield disabled.")
                return False
            with open(path, "rb") as f:
                header = f.read(13)  # magic(5) + u32 version + u32 count
                if len(header) < 13 or header[:5] != MAGIC:
                    print(f"[StarCatalog] Bad header in {path} — re-run scripts/fetch_gaia.py")
                    return False
                version, count = struct.unpack("<II", header[5:13])
                if version != VERSION:
                    print(f"[StarCatalog] Unsupported catalog version {version}")
                    return False
                rec = np.fromfile(f, dtype=DTYPE, count=count)

            self.raw = rec
            self.count = count
            self._bake(rec)
            self.loaded = True
            print(f"[StarCatalog] Loaded {count:,} GAIA stars "
                  f"({os.path.getsize(path) / (1024 * 1024):.1f} MB)")
            return True
        except Exception as e:
            print(f"[StarCatalog] Failed to load {path}: {e}")
            self.loaded = False
            return False

    def _bake(self, rec):
        ra = np.radians(rec['ra'].astype('f8'))
        dec = np.radians(rec['dec'].astype('f8'))
        plx = rec['plx'].astype('f8')             # mas
        pmra = rec['pmra'].astype('f8')           # mas/yr
        pmdec = rec['pmdec'].astype('f8')         # mas/yr
        g = rec['g'].astype('f8')
        bmrp = np.clip(rec['bmrp'].astype('f8'), -0.5, 5.0)

        cos_d = np.cos(dec)
        sin_d = np.sin(dec)
        cos_a = np.cos(ra)
        sin_a = np.sin(ra)

        # --- Unit direction vectors, ICRS equatorial -------------------
        eq_x = cos_d * cos_a
        eq_y = cos_d * sin_a
        eq_z = sin_d
        # East / north tangent basis (for proper motion)
        east = np.stack([-sin_a, cos_a, np.zeros_like(sin_a)], axis=1)
        north = np.stack([-sin_d * cos_a, -sin_d * sin_a, cos_d], axis=1)

        # --- Equatorial -> ecliptic (rotation about X by OBLIQUITY) ----
        ce, se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)

        def to_ecliptic(v):
            x, y, z = v[..., 0], v[..., 1], v[..., 2]
            return np.stack([x, ce * y + se * z, -se * y + ce * z], axis=-1)

        dir_ecl = to_ecliptic(np.stack([eq_x, eq_y, eq_z], axis=1))
        pm_ecl = to_ecliptic((pmra[:, None] * east + pmdec[:, None] * north)
                             * MAS_TO_RAD)  # rad/yr

        # --- Render frame remap (x, z, -y) ------------------------------
        def to_render(v):
            return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)

        dir_r = to_render(dir_ecl)
        pm_rate = to_render(pm_ecl)              # rad/yr

        # --- 3D positions -----------------------------------------------
        dist_pc = 1000.0 / plx
        dist_au = dist_pc * PC_TO_AU
        self.dist_ly = dist_pc * 3.2615638
        # Linear proper-motion velocity (AU/yr): p(t) = p0 + pm_vec * t_years
        self.pm_vec = pm_rate * dist_au[:, None]
        self.pos0 = dir_r * dist_au[:, None]
        # Gaia epoch J2016.0 -> J2000 sim epoch
        self.pos0 -= self.pm_vec * EPOCH_OFFSET_YEARS

        # --- Photometry ---------------------------------------------------
        self.abs_mag = (g - 5.0 * (np.log10(dist_pc) - 1.0)).astype('f4')
        teff = np.clip(TEFF_A + TEFF_B * bmrp + TEFF_C * bmrp**2 + TEFF_D * bmrp**3,
                       2800.0, 40000.0)
        self.rgb = _temperature_to_rgb_vec(teff)

        # Pre-scaled intrinsic flux. The per-frame 1/d_au^2 eye-distance
        # falloff lives in glsl/celestial/starfield.vert.
        self.flux_scaled = (np.power(10.0, -0.4 * self.abs_mag.astype('f8'))
                            * FLUX_SCALE).astype('f4')

        # f4 render buffers. Direction error from f4 rounding is a constant
        # ~1.2e-7 rad (relative eps of the distance), far below one pixel —
        # EXCEPT when the camera is near a star (interstellar travel, deferred
        # phase, where pack_render would need an f8 path).
        self.pos0 = self.pos0.astype('f4')
        self.pm_vec = self.pm_vec.astype('f4')
        n = self.count
        self._packed = np.empty((n, 7), 'f4')
        self._packed[:, 3:6] = self.rgb         # static across frames

    # ------------------------------------------------------------------
    # Per-frame packing
    # ------------------------------------------------------------------

    def pack_render(self, t_years, frame_origin):
        """Pack interleaved f4 (N,7) [rel_xyz, rgb, flux], frame-relative.

        Returns a cached array when origin/time are unchanged. The eye
        position is deliberately NOT an input: the inverse-square flux
        falloff is evaluated against the eye in the vertex shader, so this
        pack (and its cache) survives free camera motion. frame_origin is
        the render-frame origin (tracked body's barycentric position, i.e. the
        same vector subtracted from sim bodies to get pos_rel_all), NOT the eye
        position — the eye offset is applied by the view matrix in the shader.
        """
        if not self.loaded:
            return None
        cam_key = (float(frame_origin[0]), float(frame_origin[1]),
                   float(frame_origin[2]))
        t_key = float(t_years)
        if cam_key == self._last_cam and t_key == self._last_t \
                and self._last_packed is not None:
            self.fresh = False
            return self._last_packed

        # Fused single-pass Numba kernel (matches the project's @njit style)
        self.fresh = True
        packed = self._packed
        _pack_kernel(self.pos0, self.pm_vec, t_key,
                     float(frame_origin[0]), float(frame_origin[1]),
                     float(frame_origin[2]), self.flux_scaled, self.rgb,
                     packed)

        self._last_cam = cam_key
        self._last_t = t_key
        self._last_packed = packed
        return packed


@njit(parallel=True, cache=True, nogil=True)
def _pack_kernel(pos0, pm_vec, t_years, ox, oy, oz, flux_scaled, rgb, packed):
    """Single fused pass: rel = pos0 + pm*t - origin, flux = 10^(-0.4*M_G)*1e13.

    The eye-distance inverse-square falloff is applied in the vertex shader
    (starfield.vert), NOT here — packing must stay independent of the camera
    eye so the per-frame cache survives free flight. All inputs f4;
    per-element math in f8, output stored f4."""
    n = pos0.shape[0]
    for i in prange(n):
        x = float(pos0[i, 0]) + float(pm_vec[i, 0]) * t_years - ox
        y = float(pos0[i, 1]) + float(pm_vec[i, 1]) * t_years - oy
        z = float(pos0[i, 2]) + float(pm_vec[i, 2]) * t_years - oz
        packed[i, 0] = x
        packed[i, 1] = y
        packed[i, 2] = z
        packed[i, 3] = rgb[i, 0]
        packed[i, 4] = rgb[i, 1]
        packed[i, 5] = rgb[i, 2]
        packed[i, 6] = flux_scaled[i]


def _temperature_to_rgb_vec(temp_k):
    """Vectorized Tanner Helland blackbody RGB (matches
    system_manager.temperature_to_rgb exactly, over numpy arrays)."""
    temp = np.clip(temp_k, 1000.0, 40000.0) / 100.0

    # Red
    r = np.where(temp <= 66.0, 1.0,
                 329.698727446 * np.power(np.maximum(temp - 60.0, 1e-6), -0.1332047592) / 255.0)
    # Green
    g = np.where(temp <= 66.0,
                 (99.4708025861 * np.log(np.maximum(temp, 1e-6)) - 161.1195681661) / 255.0,
                 288.1221695283 * np.power(np.maximum(temp - 60.0, 1e-6), -0.0755148492) / 255.0)
    # Blue
    b = np.where(temp >= 66.0, 1.0,
                 np.where(temp <= 19.0, 0.0,
                          (138.5177312231 * np.log(np.maximum(temp - 10.0, 1e-6)) - 305.0447927307) / 255.0))

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype('f4')
