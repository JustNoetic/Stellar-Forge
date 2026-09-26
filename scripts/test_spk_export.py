"""
Validation: timeline -> SPK (.bsp) export round-trip against analytic ground truth.

Builds a synthetic two-body system (star fixed at origin + planet on a circular
2 AU orbit), encodes it via engine.ephemeris.spk_exporter.export_timeline_spk,
then reads the kernel back with spiceypy (no leapseconds kernel needed — queries
are ET-based) and compares against the analytic solution at off-node epochs.

Also covers:
- NAIF ID assignment (real, fictional, and duplicate-name protection)
- Full-density & decimated Type 13 Hermite export
- High-precision Type 2 Chebyshev export with adaptive granules
- Gaps and sparse interval Hermite anchor interpolation
- Auto-fallback from Type 2 to Type 13 on tiny timelines
- Analytical 2026 display epoch fallback
- Under-sampling / aliasing probe detection

Usage:
    python scripts/test_spk_export.py
"""

import os
import sys
import json
import math
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import spiceypy as spice

from engine.core.constants import AU_TO_KM, SECONDS_PER_YEAR, G
from engine.ephemeris.spk_exporter import (
    export_timeline_spk,
    assign_spice_ids,
    resolve_display_epoch_et,
    ANALYTICAL_2026_EPOCH_ET,
    SPK_FRAME,
    SPK_TYPE_HERMITE,
    SPK_TYPE_CHEBYSHEV,
    FICTIONAL_ID_START,
)
from engine.physics.physics_core import estimate_min_orbital_period

A_AU = 2.0
PERIOD_YR = A_AU ** 1.5          # G = 39.478 AU^3/Msun/yr^2, M = 1 Msun -> P = a^1.5
OMEGA = 2.0 * np.pi / PERIOD_YR  # rad / yr


def build_synthetic(steps=4000, periods=3.0):
    """Star at origin + planet on circular orbit in the physics x-y (ecliptic) plane."""
    t = np.linspace(0.0, periods * PERIOD_YR, steps)
    raw = np.zeros((steps, 2, 6), dtype=np.float64)
    raw[:, 1, 0] = A_AU * np.cos(OMEGA * t)
    raw[:, 1, 1] = A_AU * np.sin(OMEGA * t)
    raw[:, 1, 3] = -A_AU * OMEGA * np.sin(OMEGA * t)
    raw[:, 1, 4] = A_AU * OMEGA * np.cos(OMEGA * t)
    return t, raw


def analytic_state_km(t_yr):
    return np.array([
        A_AU * np.cos(OMEGA * t_yr) * AU_TO_KM,
        A_AU * np.sin(OMEGA * t_yr) * AU_TO_KM,
        0.0,
        -A_AU * OMEGA * np.sin(OMEGA * t_yr) * AU_TO_KM / SECONDS_PER_YEAR,
        A_AU * OMEGA * np.cos(OMEGA * t_yr) * AU_TO_KM / SECONDS_PER_YEAR,
        0.0,
    ])


def kepler_state_km(t_yr, a_au, e):
    """Analytic planar Kepler orbit state (km, km/s). Works for scalars or arrays."""
    P = a_au ** 1.5
    n = 2.0 * np.pi / P
    M = n * np.asarray(t_yr, dtype=float)
    E = M + e * np.sin(M)
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    fac = a_au * n / (1.0 - e * np.cos(E))
    x = a_au * (np.cos(E) - e)
    y = a_au * np.sqrt(1.0 - e * e) * np.sin(E)
    vx = -np.sin(E) * fac
    vy = np.sqrt(1.0 - e * e) * np.cos(E) * fac
    return np.stack([x * AU_TO_KM, y * AU_TO_KM, np.zeros_like(x),
                     vx * AU_TO_KM / SECONDS_PER_YEAR, vy * AU_TO_KM / SECONDS_PER_YEAR,
                     np.zeros_like(x)], axis=-1)


def build_eccentric(steps=8000, periods=2.0, e=0.7, a_au=1.0):
    """Star at origin + planet on an eccentric (e=0.7) planar Kepler orbit."""
    P = a_au ** 1.5
    t = np.linspace(0.0, periods * P, steps)
    st = kepler_state_km(t, a_au, e)          # (steps, 6) in km, km/s
    raw = np.zeros((steps, 2, 6), dtype=np.float64)
    raw[:, 1, 0:3] = st[:, 0:3] / AU_TO_KM
    raw[:, 1, 3:6] = st[:, 3:6] * SECONDS_PER_YEAR / AU_TO_KM
    return t, raw, e, a_au


def roundtrip_check(bsp_path, planet_id, t_span, label, pos_tol_km, vel_tol_kms,
                    analytic_fn=analytic_state_km):
    spice.furnsh(bsp_path)
    try:
        ids = set(spice.spkobj(bsp_path))
        assert planet_id in ids and 10 in ids, f"[{label}] spkobj missing IDs: {ids}"

        rng = np.random.default_rng(42)
        max_pos_err = 0.0
        max_vel_err = 0.0
        for frac in rng.uniform(0.001, 0.999, 37):
            t_yr = frac * t_span
            et = t_yr * SECONDS_PER_YEAR  # epoch_et = 0 for this test
            state, _ = spice.spkgeo(planet_id, et, SPK_FRAME, 0)
            truth = analytic_fn(t_yr)
            max_pos_err = max(max_pos_err, np.linalg.norm(state[0:3] - truth[0:3]))
            max_vel_err = max(max_vel_err, np.linalg.norm(state[3:6] - truth[3:6]))

        # Star segment must sit at the origin
        star_state, _ = spice.spkgeo(10, 0.5 * t_span * SECONDS_PER_YEAR, SPK_FRAME, 0)
        star_err = np.linalg.norm(star_state[0:3])

        print(f"[{label}] max pos err: {max_pos_err:.4f} km | max vel err: {max_vel_err:.6f} km/s | star err: {star_err:.2e} km")
        assert star_err < 1e-6, f"[{label}] star drifted from origin: {star_err} km"
        assert max_pos_err < pos_tol_km, f"[{label}] position error {max_pos_err} km > {pos_tol_km} km"
        assert max_vel_err < vel_tol_kms, f"[{label}] velocity error {max_vel_err} km/s > {vel_tol_kms} km/s"
    finally:
        spice.unload(bsp_path)


def main():
    print("=" * 70)
    print(" Stellar-Forge SPK Timeline Export — Round-Trip Validation")
    print("=" * 70)

    # 1. NAIF ID assignment (including duplicate name disambiguation)
    names = ["Sun", "Test Planet", "moonlet-9", "Sun", "Test Planet"]
    ids = assign_spice_ids(names)
    assert ids[0] == 10, f"Sun should map to NAIF 10, got {ids[0]}"
    assert ids[1] <= FICTIONAL_ID_START and ids[2] <= FICTIONAL_ID_START, f"Fictional IDs wrong: {ids}"
    assert len(ids) == len(set(ids)), f"Assigned IDs must all be unique to prevent SPICE shadowing: {ids}"
    print(f"1. ID assignment OK (unique disambiguation): {dict(zip(names, ids))}")

    # 1b. Min orbital period estimation (drives auto timeline recording density)
    pos1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel1 = np.array([[0.0, 0.0, 0.0], [0.0, math.sqrt(G), 0.0]])  # circular at a=1 AU
    mass1 = np.array([1.0, 0.0])
    parents1 = np.array([-1, 0], dtype=np.int32)
    p_min = estimate_min_orbital_period(pos1, vel1, mass1, parents1)
    assert abs(p_min - 1.0) < 1e-3, f"circular 1 AU period should be ~1 yr, got {p_min}"

    r_p = 1.0 * (1.0 - 0.7)                       # eccentric e=0.7 at periapsis, a=1 AU
    v_p = math.sqrt(G * (2.0 / r_p - 1.0))
    pos1[1] = [r_p, 0.0, 0.0]
    vel1[1] = [0.0, v_p, 0.0]
    p_min = estimate_min_orbital_period(pos1, vel1, mass1, parents1)
    assert abs(p_min - 1.0) < 1e-3, f"eccentric a=1 period should be ~1 yr, got {p_min}"

    vel1[1] = [0.0, 100.0, 0.0]                   # hyperbolic -> no bound orbit
    assert math.isinf(estimate_min_orbital_period(pos1, vel1, mass1, parents1))
    print("1b. Min orbital period estimation OK (circular, eccentric, hyperbolic)")

    # 1c. Epoch resolution fallback
    et_fallback = resolve_display_epoch_et(spice_manager=None)
    assert abs(et_fallback - ANALYTICAL_2026_EPOCH_ET) < 1.0, f"Fallback ET should be ~2026 epoch, got {et_fallback}"
    print(f"1c. Analytical 2026 display epoch fallback OK: {et_fallback:.3f} s")

    t, raw = build_synthetic()
    names = ["Sun", "Test Planet"]   # synthetic system has 2 bodies
    ids = ids[:2]
    tmpdir = tempfile.mkdtemp(prefix="sf_spk_test_")

    # 2. Full-density export round-trip (Type 13 Hermite)
    bsp_full = os.path.join(tmpdir, "synthetic_full.bsp")
    ok, msg, path = export_timeline_spk(bsp_full, raw, t, names, body_ids=ids, epoch_et=0.0,
                                        spk_type=SPK_TYPE_HERMITE)
    assert ok and path, f"Export failed: {msg}"
    print(f"2. Full export OK: {msg}")
    assert os.path.exists(bsp_full + ".json"), "Sidecar JSON missing"
    with open(bsp_full + ".json") as f:
        meta = json.load(f)
    assert meta["spk_type"] == 13 and meta["frame"] == SPK_FRAME and len(meta["bodies"]) == 2
    assert "time_span_et" in meta and "time_span_utc" in meta, "Sidecar missing time span metadata"
    print(f"   Sidecar OK: {meta['states_per_body']} states/body, span {meta['time_span_years'][1]:.2f} yr")
    roundtrip_check(bsp_full, ids[1], t[-1], "full density", pos_tol_km=100.0, vel_tol_kms=1e-3)

    # 3. Decimated export round-trip (10x fewer states, Type 13 Hermite)
    bsp_dec = os.path.join(tmpdir, "synthetic_decimated.bsp")
    ok, msg, path = export_timeline_spk(bsp_dec, raw, t, names, body_ids=ids, epoch_et=0.0,
                                        max_states=400, spk_type=SPK_TYPE_HERMITE)
    assert ok and path, f"Decimated export failed: {msg}"
    print(f"3. Decimated export OK: {msg}")
    roundtrip_check(bsp_dec, ids[1], t[-1], "decimated x10", pos_tol_km=500.0, vel_tol_kms=1e-2)

    # 4. Duplicate-epoch filtering (non-monotonic input must not crash the writer)
    t_dup = np.concatenate((t, [t[-1]]))
    raw_dup = np.concatenate((raw, raw[-1:]), axis=0)
    bsp_dup = os.path.join(tmpdir, "synthetic_dup.bsp")
    ok, msg, _ = export_timeline_spk(bsp_dup, raw_dup, t_dup, names, body_ids=ids, epoch_et=0.0)
    assert ok, f"Duplicate-epoch export failed: {msg}"
    print(f"4. Duplicate-epoch filtering OK: {msg}")

    # 5. Input validation
    ok, msg, _ = export_timeline_spk(bsp_dup, raw[:, :, :5], t, names, body_ids=ids)
    assert not ok, "Bad-shaped raw array should be rejected"
    ok, msg, _ = export_timeline_spk(bsp_dup, raw[:1], t[:1], names, body_ids=ids)
    assert not ok, "Single-state timeline should be rejected"
    print("5. Input validation OK")

    # 6. Chebyshev (Type 2, JPL DE-style) round-trip — circular orbit
    bsp_ch = os.path.join(tmpdir, "synthetic_cheb.bsp")
    ok, msg, path = export_timeline_spk(bsp_ch, raw, t, names, body_ids=ids, epoch_et=0.0,
                                        spk_type=SPK_TYPE_CHEBYSHEV)
    assert ok and path, f"Chebyshev export failed: {msg}"
    print(f"6. Chebyshev export OK: {msg}")
    with open(bsp_ch + ".json") as f:
        meta = json.load(f)
    assert meta["spk_type"] == 2 and "chebyshev" in meta
    roundtrip_check(bsp_ch, ids[1], t[-1], "chebyshev circular", pos_tol_km=1e-2, vel_tol_kms=1e-8)

    # 7. Chebyshev eccentric orbit (e = 0.7) — stresses periapsis curvature
    t_e, raw_e, ecc, a_e = build_eccentric()
    bsp_ecc = os.path.join(tmpdir, "synthetic_cheb_ecc.bsp")
    ok, msg, path = export_timeline_spk(bsp_ecc, raw_e, t_e, names, body_ids=ids, epoch_et=0.0,
                                        spk_type=SPK_TYPE_CHEBYSHEV, cheb_tol_km=0.1)
    assert ok and path, f"Eccentric Chebyshev export failed: {msg}"
    print(f"7. Eccentric Chebyshev export OK: {msg}")
    with open(bsp_ecc + ".json") as f:
        meta = json.load(f)
    n_int = meta["chebyshev"]["per_body"][1]["intervals"]
    print(f"   eccentric body encoded with {n_int} Chebyshev intervals (tol 0.1 km)")
    roundtrip_check(bsp_ecc, ids[1], t_e[-1], "chebyshev e=0.7", pos_tol_km=1.0, vel_tol_kms=5e-2,
                    analytic_fn=lambda tt: kepler_state_km(tt, a_e, ecc))

    # 8. Sparse-but-adequate sampling (~200 samples/orbit, e=0.7) — probe metric present
    t_s, raw_s, ecc_s, a_s = build_eccentric(steps=1200, periods=6.0)
    bsp_sparse = os.path.join(tmpdir, "synthetic_sparse.bsp")
    ok, msg, path = export_timeline_spk(bsp_sparse, raw_s, t_s, names, body_ids=ids, epoch_et=0.0,
                                        spk_type=SPK_TYPE_CHEBYSHEV, cheb_tol_km=0.1)
    assert ok and path, f"Sparse export failed: {msg}"
    with open(bsp_sparse + ".json") as f:
        meta = json.load(f)
    probe_err = meta["chebyshev"]["per_body"][1]["probe_max_error_km"]
    print(f"8. Sparse (~200 samples/orbit) export OK: probe {probe_err:.3f} km | {msg}")
    roundtrip_check(bsp_sparse, ids[1], t_s[-1], "sparse 200/orbit", pos_tol_km=20000.0,
                    vel_tol_kms=1.0, analytic_fn=lambda tt: kepler_state_km(tt, a_s, ecc_s))

    # 9. Aliased input (~4.3 samples/orbit) — the probe MUST flag it
    t_al = np.linspace(0.0, 3.0 * PERIOD_YR, 13)
    raw_al = np.zeros((13, 2, 6), dtype=np.float64)
    raw_al[:, 1, 0] = A_AU * np.cos(OMEGA * t_al)
    raw_al[:, 1, 1] = A_AU * np.sin(OMEGA * t_al)
    raw_al[:, 1, 3] = -A_AU * OMEGA * np.sin(OMEGA * t_al)
    raw_al[:, 1, 4] = A_AU * OMEGA * np.cos(OMEGA * t_al)
    bsp_alias = os.path.join(tmpdir, "synthetic_aliased.bsp")
    ok, msg, path = export_timeline_spk(bsp_alias, raw_al, t_al, names, body_ids=ids, epoch_et=0.0,
                                        spk_type=SPK_TYPE_CHEBYSHEV)
    assert ok, f"Aliased export failed: {msg}"
    with open(bsp_alias + ".json") as f:
        meta = json.load(f)
    probe_alias = meta["chebyshev"]["per_body"][1]["probe_max_error_km"]
    print(f"9. Aliased input flagged: probe {probe_alias:.1f} km")
    assert probe_alias > 100.0, f"Probe should flag aliased input, got {probe_alias} km"
    assert "WARNING" in msg, "Export message should warn about under-sampling"

    # 10. Tiny timeline auto-fallback from Type 2 to Type 13
    t_tiny = np.linspace(0.0, 0.01, 4)
    raw_tiny = np.zeros((4, 2, 6), dtype=np.float64)
    raw_tiny[:, 1, 0] = A_AU
    bsp_tiny = os.path.join(tmpdir, "synthetic_tiny.bsp")
    ok, msg, path = export_timeline_spk(bsp_tiny, raw_tiny, t_tiny, names, body_ids=ids,
                                        spk_type=SPK_TYPE_CHEBYSHEV)
    assert ok and path, f"Tiny timeline should auto-fallback to Type 13: {msg}"
    with open(bsp_tiny + ".json") as f:
        meta = json.load(f)
    assert meta["spk_type"] == 13, f"Expected fallback to Type 13, got {meta['spk_type']}"
    print(f"10. Tiny timeline auto-fallback to Type 13 OK: {msg}")

    # 11. Timeline with gap — verify Hermite anchor interpolation prevents zero records
    t_gap = np.concatenate([np.linspace(0.0, 0.2, 50), np.linspace(1.8, 2.0, 50)])
    raw_gap = np.zeros((len(t_gap), 2, 6), dtype=np.float64)
    raw_gap[:, 1, 0] = A_AU * np.cos(OMEGA * t_gap)
    raw_gap[:, 1, 1] = A_AU * np.sin(OMEGA * t_gap)
    raw_gap[:, 1, 3] = -A_AU * OMEGA * np.sin(OMEGA * t_gap)
    raw_gap[:, 1, 4] = A_AU * OMEGA * np.cos(OMEGA * t_gap)
    bsp_gap = os.path.join(tmpdir, "synthetic_gap.bsp")
    ok, msg, path = export_timeline_spk(bsp_gap, raw_gap, t_gap, names, body_ids=ids,
                                        spk_type=SPK_TYPE_CHEBYSHEV)
    assert ok and path, f"Gap timeline export failed: {msg}"
    spice.furnsh(bsp_gap)
    try:
        # Query in the middle of the gap (t = 1.0 yr)
        st_gap, _ = spice.spkgeo(ids[1], 1.0 * SECONDS_PER_YEAR, SPK_FRAME, 0)
        norm_gap = np.linalg.norm(st_gap[0:3])
        assert norm_gap > 0.1 * AU_TO_KM, f"Body should not collapse to zero in data gap: {norm_gap} km"
    finally:
        spice.unload(bsp_gap)
    print("11. Timeline with gap Hermite interpolation OK (no zero-granules)")

    spice.kclear()
    print("=" * 70)
    print(" ALL SPK EXPORT TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
