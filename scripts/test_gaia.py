#!/usr/bin/env python
"""Sanity checks for the GAIA star catalog layer (engine/rendering/star_catalog.py).

Two suites:
  1. Synthetic bake tests (no network, always run):
     equatorial->ecliptic transform, render-frame remap, proper-motion packing,
     and vectorized blackbody RGB vs system_manager.temperature_to_rgb.
  2. Real-catalog tests (run if data/gaia/stars.bin exists, i.e. after
     scripts/fetch_gaia.py): Proxima Centauri position/distance, Sirius
     brightness/color, Barnard's star proper motion, HDR flux packing.

Usage:
    python scripts/test_gaia.py
"""
import os
import struct
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.rendering.star_catalog import StarCatalog, DTYPE, MAGIC, VERSION, PC_TO_AU
from engine.core.constants import OBLIQUITY

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def write_temp_catalog(records):
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(np.uint32(VERSION).tobytes())
        f.write(np.uint32(records.size).tobytes())
        f.write(records.astype(DTYPE).tobytes())
    return path


def icrs_to_render(ra_deg, dec_deg):
    """Independent ICRS -> ecliptic -> render-frame transform for validation."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    v = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    ce, se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    ecl = np.array([v[0], ce * v[1] + se * v[2], -se * v[1] + ce * v[2]])
    return np.array([ecl[0], ecl[2], -ecl[1]])


def test_synthetic_bake():
    print("\n=== Suite 1: synthetic bake tests ===")
    # Star A: vernal equinox direction (RA=0, Dec=0), 10 pc away
    # Star B: north celestial pole (Dec=+90), 10 pc
    # Star C: same as A but with large proper motion in Dec (+1000 mas/yr)
    rec = np.zeros(3, dtype=DTYPE)
    rec[0] = (0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 5.0, 0.6)
    rec[1] = (30.0, 90.0, 100.0, 0.0, 0.0, 0.0, 5.0, 0.6)
    rec[2] = (0.0, 0.0, 100.0, 0.0, 1000.0, 0.0, 5.0, 0.6)
    path = write_temp_catalog(rec)
    try:
        cat = StarCatalog()
        check("catalog loads", cat.load(path))
        check("count", cat.count == 3)

        exp_a = icrs_to_render(0.0, 0.0)
        got_a = cat.pos0[0] / np.linalg.norm(cat.pos0[0])
        check("equinox direction (ecliptic+render remap)",
              np.allclose(got_a, exp_a, atol=1e-6), f"{got_a}")

        exp_b = icrs_to_render(30.0, 90.0)
        got_b = cat.pos0[1] / np.linalg.norm(cat.pos0[1])
        check("NCP direction (ecliptic lat 66.56deg)",
              np.allclose(got_b, exp_b, atol=1e-6), f"{got_b}")

        d10pc_au = 10.0 * PC_TO_AU
        check("distance bake (10 pc)", abs(np.linalg.norm(cat.pos0[0]) - d10pc_au) < 1.0)

        # Proper motion: 1000 mas/yr in Dec = 4.848e-6 rad/yr.
        # Back-propagated 16 yr to J2000 -> pos0 shifted by 16yr * rate * dist.
        pm_rate = 1000.0 * np.pi / (180.0 * 3600.0 * 1000.0)  # rad/yr
        pm_shift = pm_rate * 16.0 * d10pc_au
        check("J2000 back-propagation (pm * 16 yr)",
              abs(np.linalg.norm(cat.pos0[2] - cat.pos0[0]) - pm_shift) < 1.0)

        # pack_render: at t=+1000 yr, star C has drifted 984 pm-years from its
        # J2000 position (1000 sim years minus the 16 yr J2000 back-prop),
        # i.e. 984 arcsec = 0.2733 deg away from star A
        packed = cat.pack_render(1000.0, np.zeros(3))
        c_dir = packed[2, 0:3] / np.linalg.norm(packed[2, 0:3])
        a_dir = packed[0, 0:3] / np.linalg.norm(packed[0, 0:3])
        ang = np.degrees(np.arccos(np.clip(np.dot(c_dir, a_dir), -1, 1)))
        check("1000 yr proper-motion displacement ~0.273deg",
              abs(ang - 0.2733) < 0.005, f"{ang:.4f} deg")

        # pack cache: same inputs -> fresh == False
        cat.pack_render(1000.0, np.zeros(3))
        check("pack cache (fresh=False on repeat)", not cat.fresh)

        # Flux: both A and C have M_G = 5 - 5*(log10(10)-1) = 5.
        # Packed flux = 10^(-0.4*5) * 1e13 = 1e11. (The eye-distance
        # inverse-square falloff lives in starfield.vert, not in the pack.)
        check("HDR flux formula", abs(packed[0, 6] - 1.0e11) < 1.0e7,
              f"flux={packed[0, 6]:.6g}")
    finally:
        os.unlink(path)

    # Vectorized Helland RGB vs scalar reference
    print("\n=== Suite 1b: vectorized blackbody RGB ===")
    from engine.ephemeris.system_manager import temperature_to_rgb
    from engine.rendering.star_catalog import _temperature_to_rgb_vec
    temps = np.array([3000.0, 5772.0, 9600.0, 25000.0, 40000.0], dtype='f8')
    ref = np.array([temperature_to_rgb(t) for t in temps])
    got = _temperature_to_rgb_vec(temps)
    check("matches temperature_to_rgb (1e-4)", np.allclose(ref, got, atol=1e-4))


def test_real_catalog(path):
    print("\n=== Suite 2: real catalog tests ===")
    cat = StarCatalog()
    if not cat.load(path):
        print(f"  [SKIP] no catalog at {path}")
        return

    rec = cat.raw
    plx = rec['plx'].astype('f8')

    # --- Nearest star: Alpha Centauri A (~4.37 ly, plx ~747 mas) when the
    # Hipparcos supplement is present; Proxima (~4.246 ly, plx ~768 mas)
    # only appears in deep (G < 11) catalogs.
    i_near = int(np.argmax(plx))
    d_ly = 1000.0 / plx[i_near] * 3.2615638
    check("nearest star is ~4.0-4.5 ly (Alpha Cen / Proxima)",
          4.0 < d_ly < 4.5, f"{d_ly:.3f} ly")
    # Baked direction = stored J2016.0 direction back-propagated 16 yr of pm
    exp_dir = icrs_to_render(float(rec['ra'][i_near]), float(rec['dec'][i_near]))
    dist_au = float(np.linalg.norm(cat.pos0[i_near]))
    pm_rate = cat.pm_vec[i_near] / dist_au          # AU/yr -> rad/yr
    exp_dir = exp_dir - pm_rate * 16.0               # rad, render frame
    exp_dir /= np.linalg.norm(exp_dir)
    got_dir = cat.pos0[i_near] / np.linalg.norm(cat.pos0[i_near])
    check("nearest star bake direction matches (J2016 dir + 16yr pm back-prop)",
          np.allclose(got_dir, exp_dir, atol=1e-6))

    # --- Sirius: brightest star (V ~ -1.46, only via Hipparcos supplement) ---
    if rec['g'].min() < 0.0:
        i_sir = int(np.argmin(rec['g']))
        check("brightest star V ~ -1.46 (Sirius)",
              abs(rec['g'][i_sir] - (-1.46)) < 0.2, f"G={rec['g'][i_sir]:.2f}")
        check("Sirius absolute magnitude ~1.43",
              abs(cat.abs_mag[i_sir] - 1.43) < 0.15, f"M_G={cat.abs_mag[i_sir]:.2f}")
        check("Sirius is blue-white (b >= r)", cat.rgb[i_sir, 2] >= cat.rgb[i_sir, 0])
    else:
        print("  [SKIP] Sirius tests (no supplement / catalog too shallow)")

    # --- Barnard's star: plx ~547 mas, pm ~10360 mas/yr (deep catalogs only) ---
    cand = np.where(np.abs(plx - 547.0) < 5.0)[0]
    if cand.size:
        i_bar = int(cand[np.argmax(np.hypot(rec['pmra'][cand], rec['pmdec'][cand]))])
        pm_tot = np.hypot(float(rec['pmra'][i_bar]), float(rec['pmdec'][i_bar]))
        check("Barnard's star pm ~10.36 arcsec/yr",
              abs(pm_tot - 10360.0) / 10360.0 < 0.1, f"{pm_tot:.0f} mas/yr")
        pm_rate = pm_tot * np.pi / (180.0 * 3600.0 * 1000.0)
        check("Barnard's 1000 yr drift ~2.88 deg",
              abs(np.degrees(pm_rate * 1000.0) - 2.88) < 0.4,
              f"{np.degrees(pm_rate * 1000.0):.2f} deg")
    else:
        print("  [SKIP] Barnard's star tests (not in catalog depth)")

    # --- Packed flux is the constant pre-scaled intrinsic flux
    # 10^(-0.4*M_G) * 1e13; the eye-distance inverse-square falloff is
    # applied per-frame in starfield.vert, so the pack itself is eye-free.
    packed = cat.pack_render(0.0, np.zeros(3))
    fluxes = packed[:, 6]
    expected_flux = np.power(10.0, -0.4 * cat.abs_mag.astype('f8')) * 1e13
    check("packed flux = 10^(-0.4*M_G)*1e13 (all stars)",
          np.allclose(fluxes, expected_flux, rtol=1e-4),
          f"max rel err {np.max(np.abs(fluxes - expected_flux) / expected_flux):.2e}")
    check("brightest intrinsic flux is smallest M_G",
          int(np.argmax(fluxes)) == int(np.argmin(cat.abs_mag)))

    # --- pack_render timing ---
    cat.pack_render(1.0, np.array([1.0, 2.0, 3.0]))  # force fresh
    t0 = time.perf_counter()
    for k in range(10):
        cat.pack_render(1.0 + 0.001 * k, np.array([1.0, 2.0, 3.0 + 0.1 * k]))
    ms = (time.perf_counter() - t0) / 10 * 1000.0
    check("pack_render < 5 ms/frame", ms < 5.0, f"{ms:.2f} ms @ {cat.count:,} stars")

    # Intrinsic luminosity ratio: flux ratio == 10^(-0.4*(M1 - M2)) from
    # absolute magnitudes (the pack is distance-independent by design).
    ratio = fluxes[i_near] / fluxes.max()
    expected = 10.0 ** (-0.4 * (cat.abs_mag[i_near] - cat.abs_mag.min()))
    check("flux ratio matches 10^(-0.4*dM_G) (within 20%)",
          abs(ratio - expected) / expected < 0.2,
          f"{ratio:.4g} vs expected {expected:.4g}")
    if plx.max() > 760.0:
        i_prox = int(np.argmax(plx))
        check("Proxima intrinsically faint (flux < 1% of brightest)",
              fluxes[i_prox] < 0.01 * fluxes.max(),
              f"flux_Prox={fluxes[i_prox]:.4g}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_bin = os.path.normpath(os.path.join(here, "..", "data", "gaia", "stars.bin"))
    path = sys.argv[1] if len(sys.argv) > 1 else default_bin

    test_synthetic_bake()
    if os.path.exists(path):
        test_real_catalog(path)
    else:
        print(f"\n=== Suite 2 skipped: no catalog at {path} ===")
        print("Run 'python scripts/fetch_gaia.py' to build it.")

    print(f"\n{'=' * 40}\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
