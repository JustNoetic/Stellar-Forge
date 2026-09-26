"""
spk_exporter.py — Export recorded N-body timeline buffers to a NASA NAIF SPK (.bsp) kernel.

Takes the full-precision ``timeline_raw`` (float64, physics frame) and ``timeline_times``
buffers produced by the physics thread's timeline render pass and encodes every body as an
SPK segment relative to the system barycenter (center = 0, flat barycenter-absolute layout).

Two segment types are supported:
  - Type 2 (default): per-interval Chebyshev polynomial fits of position, tolerance-driven
    granule refinement — the same representation JPL uses for the DE planetary ephemerides.
  - Type 13: Hermite interpolation of discrete states (uses pos + vel, unequal spacing).

Units & frames:
  - The engine physics frame is ECLIPJ2000 Cartesian axes, so no rotation is applied.
    (Verified against SpiceManager.populate_states_fast: spice (x,y,z) -> physics (x,y,z)
    with only an AU<->km scale; the render frame's (x, z, -y) swap cancels out.)
  - Positions: AU -> km (x AU_TO_KM). Velocities: AU/yr -> km/s (x AU_TO_KM / SECONDS_PER_YEAR).
  - Epochs: sim years -> ET seconds past J2000: ``et = epoch_et + t * SECONDS_PER_YEAR``,
    where ``epoch_et`` anchors sim t=0 to the app's display epoch (2026-01-01 12:00 UTC,
    matching render_utils.format_sim_time). If no leapseconds kernel is available the
    anchor falls back to 0.0 (sim t=0 == J2000).

A small JSON sidecar (<file>.bsp.json) is written next to the kernel recording the epoch
anchor, frame, and body name <-> NAIF ID mapping for later re-import / replay.
"""

import os
import re
import json
import time
import threading
import datetime

import numpy as np
import spiceypy as spice

from engine.core.constants import AU_TO_KM, SECONDS_PER_YEAR
from engine.path_utils import get_external_path
from engine.ephemeris.spice_manager import SpiceManager, SPICE_LOCK, get_spice_manager

# App-wide display epoch: sim t=0 is shown as 2026-01-01 12:00 (see render_utils.format_sim_time)
DISPLAY_EPOCH_DT = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
# Analytical ET (seconds past J2000 TDB) for 2026-01-01 12:00:00 UTC (9497 days + 69.184s)
ANALYTICAL_2026_EPOCH_ET = 820540869.184

SPK_FRAME = "ECLIPJ2000"
SPK_CENTER_ID = 0                 # Solar System Barycenter (flat, barycenter-absolute)
SPK_TYPE_CHEBYSHEV = 2            # Chebyshev polynomials, position only (JPL DE-series style)
SPK_TYPE_HERMITE = 13             # Hermite interpolation, unequal time steps (uses pos + vel)
SPK_TYPE = SPK_TYPE_CHEBYSHEV     # Default export segment type
DEFAULT_MAX_STATES = 20000        # Uniform decimation cap per body
DEFAULT_DEGREE = 5                # Hermite polynomial degree (must be odd, 1..27)
CHEB_DEGREE = 13                  # Chebyshev polynomial degree (JPL DE series uses ~13)
CHEB_TOL_KM = 1.0                 # Position fit tolerance; interval count doubles until met
CHEB_MAX_INTERVALS = 4096         # Hard cap on Chebyshev granules per body segment
FICTIONAL_ID_START = -100000      # First NAIF ID assigned to bodies with no known SPICE ID


def _sanitize_segid(name, max_len=40):
    """Build a SPICE-safe segment identifier (<= max_len chars) from a body name."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_").upper() or "BODY"
    segid = f"SF_TIMELINE_{cleaned}"
    return segid[:max_len]


def assign_spice_ids(body_names):
    """
    Map body names to NAIF IDs. Bodies matching a known SpiceManager.SPICE_BODIES name
    reuse the real NAIF ID (e.g. 'Earth' -> 399, 'Sun' -> 10). Unknown/fictional bodies
    or duplicate occurrences get unique negative IDs starting at FICTIONAL_ID_START,
    skipping any ID already in use.

    Returns a list[int] parallel to body_names.
    """
    name_to_id = {}
    for sp_id, info in SpiceManager.SPICE_BODIES.items():
        nm = info.get("name")
        if nm:
            name_to_id.setdefault(nm.strip().upper(), sp_id)

    used = set(SpiceManager.SPICE_BODIES.keys())
    ids = []
    next_fictional = FICTIONAL_ID_START
    for name in body_names:
        sp_id = name_to_id.get(str(name).strip().upper())
        if sp_id is None or sp_id in ids:
            while next_fictional in used or next_fictional in ids:
                next_fictional -= 1
            sp_id = next_fictional
            next_fictional -= 1
        ids.append(sp_id)
    return ids


def resolve_display_epoch_et(spice_manager=None):
    """
    Resolve the ET (seconds past J2000 TDB) of the app's display epoch (2026-01-01 12:00 UTC).
    Uses the already-loaded kernel pool when possible; otherwise furnishes just the
    leapseconds kernel if it exists on disk. Falls back to ANALYTICAL_2026_EPOCH_ET
    (820540869.184 s) to maintain the exact 2026 epoch anchor even without downloaded kernels.
    """
    try:
        if spice_manager is not None and getattr(spice_manager, "kernels_loaded", False):
            et = spice_manager.datetime_to_et(DISPLAY_EPOCH_DT)
            if et is not None and et != 0.0:
                return float(et)
    except Exception:
        pass

    try:
        mgr = get_spice_manager() or SpiceManager()
        tls_path = mgr.find_kernel_path("naif0012.tls")
        if tls_path and os.path.exists(tls_path):
            with SPICE_LOCK:
                spice.furnsh(tls_path)
                return float(spice.str2et(DISPLAY_EPOCH_DT.strftime("%Y-%m-%dT%H:%M:%S")))
    except Exception as e:
        print(f"[SPK] Could not resolve display epoch ET ({e}); anchoring sim t=0 to analytical 2026 epoch.")
    return ANALYTICAL_2026_EPOCH_ET


def _chebyshev_fit_records(epochs_s, pos_km, vel_kms, degree, tol_km):
    """
    Fit per-interval Chebyshev polynomials (position records) to sampled states —
    the same representation JPL uses for the DE planetary ephemerides (SPK Type 2).

    Each recorded state contributes BOTH position and velocity constraints (6 equations
    per sample against 3*(degree+1) coefficients), so a granule stays well-determined
    with as few as ~5 samples — this roughly halves the recording density a
    position-only fit would need. Velocity rows are expressed in the Chebyshev time
    domain (dpos/dtau = v * intlen/2, in km) so both equation sets share units.

    Tiles [epochs_s[0], epochs_s[-1]] with equal-length intervals; the interval count
    doubles until the maximum position residual at the sample points is <= tol_km or
    the overdetermination limit / CHEB_MAX_INTERVALS is reached.

    Returns (cdata, intlen_s, n_intervals, max_residual_km), where cdata has shape
    (n_intervals, 3*(degree+1)) — exactly the record layout SPK Type 2 expects
    (X coeffs, then Y, then Z; record midpoints/radii are derived by SPICE from
    btime + intlen).
    """
    t0 = float(epochs_s[0])
    t1 = float(epochs_s[-1])
    n_samp = len(epochs_s)
    recsize = 3 * (degree + 1)
    duration_s = max(t1 - t0, 1e-6)

    # Granule sizing: per-axis stacked LS has 2*n_samples equations against (degree+1)
    # coefficients. Start with a physical baseline (~32 days per granule, matching JPL DE-series
    # conventions) rather than scaling with sample count, and double adaptively until tol_km
    # is met. n_cap maintains >= ~2.3x overdetermination on average.
    n_cap = max(2, min(int(n_samp / 16.0), CHEB_MAX_INTERVALS))
    base_int = max(2, int(np.ceil(duration_s / (32.0 * 86400.0))))
    n_int = max(2, min(base_int, n_cap))

    while True:
        intlen = (t1 - t0) / n_int
        half = 0.5 * intlen
        cdata = np.zeros((n_int, recsize))
        max_resid = 0.0
        has_sparse_interval = False

        for j in range(n_int):
            a = t0 + j * intlen
            b = a + intlen
            if j == n_int - 1:
                m = (epochs_s >= a - 1e-6) & (epochs_s <= b + 1e-6)
            else:
                m = (epochs_s >= a - 1e-6) & (epochs_s < b)
            t_ = epochs_s[m]
            p_ = pos_km[m]
            v_ = vel_kms[m]

            min_pts = max(2, (degree + 1) // 2)
            if len(t_) < min_pts:
                has_sparse_interval = True
                # Interpolate anchor points via cubic Hermite from surrounding samples
                # to prevent all-zero records or underdetermined fits in data gaps
                idx_before = np.where(epochs_s <= a)[0]
                idx_after = np.where(epochs_s >= b)[0]
                i_a = idx_before[-1] if len(idx_before) > 0 else 0
                i_b = idx_after[0] if len(idx_after) > 0 else len(epochs_s) - 1

                t_a_s, t_b_s = epochs_s[i_a], epochs_s[i_b]
                p_a, p_b = pos_km[i_a], pos_km[i_b]
                v_a, v_b = vel_kms[i_a], vel_kms[i_b]
                dt_span = max(t_b_s - t_a_s, 1e-6)

                k_nodes = np.arange(1, degree + 2)
                tau_synth = np.cos((2 * k_nodes - 1) * np.pi / (2 * (degree + 1)))
                t_synth = 0.5 * (b - a) * tau_synth + 0.5 * (a + b)

                s = np.clip((t_synth - t_a_s) / dt_span, 0.0, 1.0)
                s2 = s * s
                s3 = s2 * s
                h00 = 2.0 * s3 - 3.0 * s2 + 1.0
                h10 = s3 - 2.0 * s2 + s
                h01 = -2.0 * s3 + 3.0 * s2
                h11 = s3 - s2

                p_ = (h00[:, None] * p_a + h10[:, None] * dt_span * v_a +
                      h01[:, None] * p_b + h11[:, None] * dt_span * v_b)
                dh00 = (6.0 * s2 - 6.0 * s) / dt_span
                dh10 = 3.0 * s2 - 4.0 * s + 1.0
                dh01 = (-6.0 * s2 + 6.0 * s) / dt_span
                dh11 = 3.0 * s2 - 2.0 * s
                v_ = (dh00[:, None] * p_a + dh10[:, None] * v_a +
                      dh01[:, None] * p_b + dh11[:, None] * v_b)
                tau = tau_synth
            else:
                tau = (2.0 * t_ - (a + b)) / (b - a)

            # Chebyshev basis T_k(tau) and its tau-derivative via three-term recurrences
            A = np.empty((len(tau), degree + 1))
            D = np.zeros((len(tau), degree + 1))
            A[:, 0] = 1.0
            if degree >= 1:
                A[:, 1] = tau
                D[:, 1] = 1.0
            for k in range(2, degree + 1):
                A[:, k] = 2.0 * tau * A[:, k - 1] - A[:, k - 2]
                D[:, k] = 2.0 * A[:, k - 1] + 2.0 * tau * D[:, k - 1] - D[:, k - 2]

            # Stacked LS: [A; D] coef = [pos; vel * intlen/2]  (both sides in km)
            M = np.vstack((A, D))
            rhs = np.vstack((p_, v_ * half))
            coef, *_ = np.linalg.lstsq(M, rhs, rcond=None)    # (degree+1, 3)
            resid = float(np.abs(A @ coef - p_).max())
            max_resid = max(max_resid, resid)

            rec = cdata[j]
            rec[0:degree + 1] = coef[:, 0]
            rec[degree + 1:2 * (degree + 1)] = coef[:, 1]
            rec[2 * (degree + 1):] = coef[:, 2]

        best = (cdata, intlen, n_int, max_resid)
        if (max_resid <= tol_km and not has_sparse_interval) or n_int >= n_cap:
            return best
        if n_int * 2 > n_cap:
            return best
        n_int = n_int * 2


def _probe_chebyshev_vs_hermite(epochs_s, pos_km, vel_kms, cdata, intlen_s, degree,
                                max_probes=20000):
    """
    Aliasing detector for the Chebyshev export path.

    Both the Chebyshev fit and a cubic Hermite interpolation of the raw (pos, vel)
    states are near-exact AT the recorded samples, so comparing them at the
    *midpoints between samples* exposes dynamics the sampling failed to capture
    (under-sampling / aliasing). Two independent wrong interpolants disagreeing is
    the signal; both agreeing on a wrong answer is a known limitation (reported
    fit residual is still valid at the samples).

    Returns the max absolute position disagreement in km.
    """
    n_samp = len(epochs_s)
    if n_samp < 3:
        return 0.0

    n_probe = min(n_samp - 1, max_probes)
    idx = np.unique(np.linspace(0, n_samp - 2, n_probe).astype(np.int64))
    t_a = epochs_s[idx]
    t_b = epochs_s[idx + 1]
    t_p = 0.5 * (t_a + t_b)
    h = t_b - t_a
    valid = h > 0.0
    if not np.any(valid):
        return 0.0
    t_a, t_p, h, idx = t_a[valid], t_p[valid], h[valid], idx[valid]

    # Cubic Hermite interpolation of the raw states at probe times
    s = (t_p - t_a) / h
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    hv = h[:, None]
    p_herm = (h00[:, None] * pos_km[idx] + h10[:, None] * hv * vel_kms[idx]
              + h01[:, None] * pos_km[idx + 1] + h11[:, None] * hv * vel_kms[idx + 1])

    # Evaluate the Chebyshev records at probe times (Clenshaw-free direct recurrence)
    t0 = float(epochs_s[0])
    j = np.clip(((t_p - t0) / intlen_s).astype(np.int64), 0, len(cdata) - 1)
    a = t0 + j * intlen_s
    tau = (2.0 * t_p - (2.0 * a + intlen_s)) / intlen_s
    coefs = cdata[j].reshape(len(t_p), 3, degree + 1)
    p_cheb = coefs[:, :, 0].copy()
    if degree >= 1:
        t_prev = np.ones(len(t_p))
        t_cur = tau
        p_cheb += coefs[:, :, 1] * t_cur[:, None]
        for k in range(2, degree + 1):
            t_next = 2.0 * tau * t_cur - t_prev
            p_cheb += coefs[:, :, k] * t_next[:, None]
            t_prev, t_cur = t_cur, t_next

    return float(np.abs(p_cheb - p_herm).max())



def export_timeline_spk(filepath, timeline_raw, timeline_times, body_names, body_ids=None,
                        epoch_et=0.0, frame=SPK_FRAME, max_states=DEFAULT_MAX_STATES,
                        spk_type=SPK_TYPE_CHEBYSHEV, degree=DEFAULT_DEGREE,
                        cheb_degree=CHEB_DEGREE, cheb_tol_km=CHEB_TOL_KM,
                        metadata=None, progress_callback=None):
    """
    Write recorded timeline buffers to a binary SPK kernel.

    Args:
        filepath:       Destination .bsp path.
        timeline_raw:   (M, N, 6) float64 array — physics-frame [x,y,z (AU), vx,vy,vz (AU/yr)].
        timeline_times: (M,) float64 sim times in years (must be increasing).
        body_names:     list of N body names.
        body_ids:       list of N NAIF IDs (auto-assigned via assign_spice_ids when None).
        epoch_et:       ET (s past J2000) corresponding to sim t=0.
        frame:          SPICE reference frame label for the segments.
        max_states:     Uniform decimation cap on states written per body.
        spk_type:       SPK_TYPE_CHEBYSHEV (2, default — JPL DE-style per-interval Chebyshev
                        fit, tolerance-driven) or SPK_TYPE_HERMITE (13 — discrete states).
        degree:         Hermite polynomial degree (Type 13 only; odd, 1..27; auto-clamped).
        cheb_degree:    Chebyshev polynomial degree (Type 2 only; ~13, JPL-like).
        cheb_tol_km:    Type 2 fit tolerance in km — granule count doubles until met.
        metadata:       Optional dict merged into the JSON sidecar file.
        progress_callback: Optional fn(frac: float, msg: str).

    Returns (success: bool, message: str, filepath: str | None).
    """
    raw = np.asarray(timeline_raw, dtype=np.float64)
    times = np.asarray(timeline_times, dtype=np.float64)

    if raw.ndim != 3 or raw.shape[2] != 6:
        return False, f"timeline_raw must have shape (steps, bodies, 6); got {raw.shape}", None
    num_steps, num_bodies = raw.shape[0], raw.shape[1]
    if times.ndim != 1 or len(times) != num_steps:
        return False, f"timeline_times length {len(times)} does not match {num_steps} steps", None
    if num_steps < 2:
        return False, "Need at least 2 recorded states to build an SPK segment", None
    if len(body_names) != num_bodies:
        return False, f"Got {len(body_names)} names for {num_bodies} bodies", None
    if body_ids is None:
        body_ids = assign_spice_ids(body_names)
    if len(body_ids) != num_bodies:
        return False, f"Got {len(body_ids)} IDs for {num_bodies} bodies", None

    # Enforce strictly increasing epochs (drop duplicate-time rows)
    keep = np.concatenate(([True], np.diff(times) > 0.0))
    if not np.all(keep):
        times = times[keep]
        raw = raw[keep]
        num_steps = len(times)
        if num_steps < 2:
            return False, "Fewer than 2 distinct epochs in timeline buffer", None

    # Decimation strategy:
    # - Type 13 (Hermite): decimate states if num_steps > max_states to avoid giant files.
    # - Type 2 (Chebyshev): keep all states for the least-squares fit (cap at 200,000 only
    #   to avoid lstsq memory blowup), since Chebyshev output file size depends only on
    #   intervals and degree, not on the input sample count.
    if spk_type == SPK_TYPE_HERMITE:
        stride = max(1, int(np.ceil(num_steps / max(2, max_states))))
        sel = np.unique(np.concatenate((np.arange(0, num_steps, stride), [num_steps - 1])))
        times = times[sel]
        raw = raw[sel]
        num_states = len(times)
    else:
        if num_steps > 200000:
            stride = max(1, int(np.ceil(num_steps / 200000)))
            sel = np.unique(np.concatenate((np.arange(0, num_steps, stride), [num_steps - 1])))
            times = times[sel]
            raw = raw[sel]
        num_states = len(times)

    if spk_type == SPK_TYPE_CHEBYSHEV:
        degree = int(cheb_degree)
        while degree > 3 and num_states < 6 * (degree + 1):
            degree -= 1
        if num_states < 2 * (degree + 1):
            # Auto-fallback to Type 13 Hermite for tiny timelines rather than erroring
            spk_type = SPK_TYPE_HERMITE
            degree = 3 if num_states >= 2 else 1

    if spk_type == SPK_TYPE_HERMITE:
        degree = int(degree)
        while degree > 1 and num_states < (degree + 1) // 2:
            degree -= 2
        if degree % 2 == 0:
            degree -= 1
        degree = max(1, degree)
    elif spk_type != SPK_TYPE_CHEBYSHEV:
        return False, f"Unsupported spk_type {spk_type} (use {SPK_TYPE_CHEBYSHEV} or {SPK_TYPE_HERMITE})", None

    # Unit conversion: AU -> km, AU/yr -> km/s
    vel_scale = AU_TO_KM / SECONDS_PER_YEAR
    scale = np.array([AU_TO_KM, AU_TO_KM, AU_TO_KM, vel_scale, vel_scale, vel_scale],
                     dtype=np.float64)
    epochs = epoch_et + times * SECONDS_PER_YEAR

    temp_path = filepath + ".part"
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    handle = None
    with SPICE_LOCK:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            handle = spice.spkopn(temp_path, "STELLAR_FORGE_TIMELINE", 0)
            fit_stats = []
            for i in range(num_bodies):
                segid = _sanitize_segid(body_names[i])
                if spk_type == SPK_TYPE_HERMITE:
                    states = raw[:, i, :] * scale
                    spice.spkw13(
                        handle,
                        int(body_ids[i]),           # Target body
                        SPK_CENTER_ID,              # Center: system barycenter (flat layout)
                        frame,
                        float(epochs[0]),
                        float(epochs[-1]),
                        segid,
                        int(degree),
                        num_states,
                        states,
                        epochs,
                    )
                    fit_stats.append(None)
                else:
                    pos_km = raw[:, i, 0:3] * AU_TO_KM
                    vel_kms = raw[:, i, 3:6] * vel_scale
                    cdata, intlen_s, n_int, resid = _chebyshev_fit_records(
                        epochs, pos_km, vel_kms, int(degree), float(cheb_tol_km))
                    probe_err = _probe_chebyshev_vs_hermite(
                        epochs, pos_km, vel_kms, cdata, intlen_s, int(degree))
                    spice.spkw02(
                        handle,
                        int(body_ids[i]),
                        SPK_CENTER_ID,
                        frame,
                        float(epochs[0]),
                        float(epochs[-1]),
                        segid,
                        float(intlen_s),
                        int(n_int),
                        int(degree),
                        cdata.reshape(-1),
                        float(epochs[0]),           # btime: begin of first logical record
                    )
                    fit_stats.append({"intervals": int(n_int),
                                      "interval_seconds": float(intlen_s),
                                      "max_fit_residual_km": float(resid),
                                      "probe_max_error_km": float(probe_err)})
                if progress_callback:
                    progress_callback((i + 1) / num_bodies, f"Encoded {body_names[i]}")
            spice.spkcls(handle)
            handle = None

            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            os.replace(temp_path, filepath)
        except Exception as e:
            if handle is not None:
                try:
                    spice.spkcls(handle)
                except Exception:
                    pass
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return False, f"Failed to encode SPK kernel: {e}", None

    # JSON sidecar: epoch anchor + ID mapping (for future re-import / replay)
    start_utc_dt = DISPLAY_EPOCH_DT + datetime.timedelta(seconds=float(times[0]) * SECONDS_PER_YEAR)
    end_utc_dt = DISPLAY_EPOCH_DT + datetime.timedelta(seconds=float(times[-1]) * SECONDS_PER_YEAR)
    sidecar = {
        "generator": "Stellar-Forge spk_exporter",
        "spk_type": int(spk_type),
        "frame": frame,
        "center_id": SPK_CENTER_ID,
        "epoch_et_seconds_past_j2000": float(epoch_et),
        "display_epoch_utc": DISPLAY_EPOCH_DT.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sim_time_relation": "et = epoch_et + sim_t_years * SECONDS_PER_YEAR",
        "states_per_body": int(num_states),
        "time_span_years": [float(times[0]), float(times[-1])],
        "time_span_et": [float(epochs[0]), float(epochs[-1])],
        "time_span_utc": [start_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), end_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")],
        "bodies": [{"name": str(n), "spice_id": int(i)} for n, i in zip(body_names, body_ids)],
    }
    if spk_type == SPK_TYPE_HERMITE:
        sidecar["hermite_degree"] = int(degree)
    else:
        sidecar["chebyshev"] = {
            "degree": int(degree),
            "tol_km": float(cheb_tol_km),
            "per_body": [
                {"name": str(n), **(fit_stats[k] or {})} for k, n in enumerate(body_names)
            ],
        }
    if metadata:
        sidecar.update(metadata)
    try:
        with open(filepath + ".json", "w") as f:
            json.dump(sidecar, f, indent=2)
    except Exception as e:
        print(f"[SPK] Warning: failed to write sidecar JSON: {e}")

    size_kb = os.path.getsize(filepath) / 1024.0
    if spk_type == SPK_TYPE_HERMITE:
        detail = f"Type 13 Hermite, deg {degree}"
    else:
        worst = max((s["max_fit_residual_km"] for s in fit_stats if s), default=0.0)
        worst_probe = max((s["probe_max_error_km"] for s in fit_stats if s), default=0.0)
        detail = f"Type 2 Chebyshev, deg {degree}, max fit residual {worst:.2e} km"
        if worst_probe > max(10.0 * cheb_tol_km, 1.0):
            detail += (f" | WARNING: probe error {worst_probe:.1f} km - timeline may be "
                       f"under-sampled; re-render at higher resolution")
    return True, (f"Wrote {num_bodies} bodies from {num_states} states "
                  f"({detail}) - {size_kb:.1f} KB"), filepath


def export_timeline_spk_async(app, bodies_data):
    """
    UI entry point: snapshot the current timeline buffers under the shared-state lock,
    then encode the .bsp on a background thread. Status is reported through the app's
    toast notification (``_screenshot_toast``). Re-entrant calls while an export is
    running are ignored.
    """
    if getattr(app, "_spk_exporting", False):
        return

    ss = app.shared_state
    with ss["lock"]:
        raw = ss.get("timeline_raw")
        times = ss.get("timeline_times")
        if raw is None or times is None or len(times) < 2:
            app._screenshot_toast = ("SPK export: no recorded timeline to export", time.time())
            return
        raw = np.array(raw, dtype=np.float64, copy=True)
        times = np.array(times, dtype=np.float64, copy=True)

    num_bodies = raw.shape[1]
    names = []
    for i in range(num_bodies):
        if bodies_data is not None and i < len(bodies_data):
            names.append(bodies_data[i].get("name", f"Body {i}"))
        else:
            names.append(f"Body {i}")

    body_ids = assign_spice_ids(names)
    epoch_et = resolve_display_epoch_et(getattr(app, "sys_mgr_spice", None))

    sys_name = re.sub(r'[\\/:*?"<>|]+', "_", getattr(app, "active_system_name", "system")).strip() or "system"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ephem_dir = get_external_path("exports", "ephemeris", sys_name)
    os.makedirs(ephem_dir, exist_ok=True)
    filepath = os.path.join(ephem_dir, f"{sys_name}_timeline_{stamp}.bsp")

    app._spk_exporting = True
    app._screenshot_toast = ("Exporting SPK kernel...", time.time())

    def _worker():
        try:
            ok, msg, out = export_timeline_spk(
                filepath, raw, times, names, body_ids=body_ids, epoch_et=epoch_et,
                metadata={"system_name": getattr(app, "active_system_name", "system")},
            )
            if ok:
                display_rel = f"exports/ephemeris/{sys_name}/{os.path.basename(out)}"
                app._screenshot_toast = (f"Saved: {display_rel} ({msg})", time.time())
            else:
                app._screenshot_toast = (f"SPK export failed: {msg}", time.time())
        except Exception as e:
            app._screenshot_toast = (f"SPK export failed: {e}", time.time())
        finally:
            app._spk_exporting = False

    threading.Thread(target=_worker, daemon=True).start()

