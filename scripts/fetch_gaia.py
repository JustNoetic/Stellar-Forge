"""Fetch the Gaia DR3 bright-star catalog as a compact binary for the
Stellar-Forge GAIA starfield layer.

Queries a magnitude-limited, parallax-quality-cut subset of gaiadr3.gaia_source
in 24 RA bands (to stay under TAP sync row limits) and writes a 32-byte-per-star
binary blob to data/gaia/stars.bin. The engine's StarCatalog
(engine/rendering/star_catalog.py) loads this file; the record dtype is defined
identically in both places — keep them in sync.

Sources:
    esa    ESA Gaia Archive TAP (gaiadr3.gaia_source)  [default]
    vizier VizieR TAP ("I/355/gaiadr3")

Bright-star supplement (default on): Gaia DR3 is missing most naked-eye stars
brighter than G ~ 2 (saturated detectors, poor/absent astrometry). Stars with
V < 2.5 are therefore taken from the Hipparcos catalog (VizieR I/239/hip_main),
propagated from its J1991.25 epoch to Gaia's J2016.0 via proper motion; Gaia
entries within 10 arcsec of a Hipparcos star are dropped (Hipparcos wins).

Usage:
    python scripts/fetch_gaia.py                     # G < 10 full sky (~460k stars, ~15 MB)
    python scripts/fetch_gaia.py --test              # tiny G < 7 sample for quick iteration
    python scripts/fetch_gaia.py --source vizier --max-mag 8

Output binary layout (little-endian):
    header:  magic "GAIA1" | u32 version | u32 count
    records: ra f32 | dec f32 | plx_mas f32 | pmra_masyr f32 | pmdec_masyr f32
             | rv_kms f32 | g_mag f32 | bp_rp f32
"""
import argparse
import io
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

# Same workaround as fetch_horizons.py — some Windows Python installs lack the
# CA bundle for cds.unistra.fr
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# Must match StarCatalog.DTYPE in engine/rendering/star_catalog.py
GAIA_BIN_DTYPE = np.dtype([
    ('ra', '<f4'),        # deg, ICRS
    ('dec', '<f4'),       # deg, ICRS
    ('plx', '<f4'),       # mas  (parallax)
    ('pmra', '<f4'),      # mas/yr (* cos(dec) included, Gaia convention)
    ('pmdec', '<f4'),     # mas/yr
    ('rv', '<f4'),        # km/s (radial velocity, may be 0 if unknown)
    ('g', '<f4'),         # mag  (phot_g_mean_mag)
    ('bmrp', '<f4'),      # mag  (BP - RP color index)
])
GAIA_MAGIC = b"GAIA1"
GAIA_VERSION = 1
NUM_RA_BANDS = 24

HIPPARCOS_EPOCH = 1991.25          # Hipparcos catalog reference epoch (Julian years)
GAIA_EPOCH = 2016.0                # Gaia DR3 reference epoch
HIP_DEFAULT_VLIM = 2.5             # supplement stars brighter than this V mag

TAP_ENDPOINTS = {
    "esa": "https://gea.esac.esa.int/tap-server/tap/sync",
    "vizier": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
}

# Unifies ESA gaiadr3.gaia_source and VizieR I/355/gaiadr3 column names
COLUMN_MAP = {
    "esa": {
        "ra": "ra", "dec": "dec", "plx": "parallax", "pmra": "pmra",
        "pmdec": "pmdec", "rv": "radial_velocity",
        "g": "phot_g_mean_mag", "bp": "phot_bp_mean_mag", "rp": "phot_rp_mean_mag",
    },
    "vizier": {
        "ra": "RA_ICRS", "dec": "DE_ICRS", "plx": "Plx", "pmra": "pmRA",
        "pmdec": "pmDE", "rv": "RV",
        "g": "Gmag", "bp": "BPmag", "rp": "RPmag",
    },
}


def build_query(source, ra0, ra1, max_mag, plx_err_cut):
    """ADQL for one RA band [ra0, ra1)."""
    c = COLUMN_MAP[source]
    if source == "esa":
        table = "gaiadr3.gaia_source"
        # ESA exposes parallax_over_error directly
        quality = f"{c['plx']}_over_error > {plx_err_cut}"
    else:
        table = '"I/355/gaiadr3"'
        # VizieR stores the parallax error as e_Plx
        quality = f"{c['plx']} > 0 AND {c['plx']}/e_Plx > {plx_err_cut}"
    return (
        f"SELECT {c['ra']}, {c['dec']}, {c['plx']}, {c['pmra']}, {c['pmdec']}, "
        f"{c['rv']}, {c['g']}, {c['bp']}, {c['rp']} "
        f"FROM {table} "
        f"WHERE {c['g']} < {max_mag} AND {c['plx']} > 0 AND {quality} "
        f"AND {c['ra']} >= {ra0:.6f} AND {c['ra']} < {ra1:.6f}"
    )


def run_tap_query(source, adql, timeout):
    """Execute a TAP sync query, returning CSV text."""
    params = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": adql,
    }).encode()
    req = urllib.request.Request(TAP_ENDPOINTS[source], data=params, headers={
        "User-Agent": "Stellar-Forge/1.0 (GAIA starfield fetcher)",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_csv_to_records(csv_text, source):
    """Parse TAP CSV output into the GAIA_BIN_DTYPE structured array."""
    text = csv_text.strip()
    if not text or text.lstrip().upper().startswith("<"):
        raise ValueError("empty or non-CSV response (rate limit / server error?)")
    rows = np.genfromtxt(io.StringIO(text), delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    if rows.size == 0 or rows.ndim == 0:
        return np.zeros(0, dtype=GAIA_BIN_DTYPE)
    rows = np.atleast_1d(rows)
    c = COLUMN_MAP[source]

    def col(name):
        vals = np.array(rows[c[name]], dtype="f8")
        vals = np.where(np.isfinite(vals), vals, 0.0)  # missing rv/bp/rp -> 0
        return vals

    rec = np.zeros(rows.size, dtype=GAIA_BIN_DTYPE)
    rec["ra"] = col("ra")
    rec["dec"] = col("dec")
    rec["plx"] = col("plx")
    rec["pmra"] = col("pmra")
    rec["pmdec"] = col("pmdec")
    rec["rv"] = col("rv")
    rec["g"] = col("g")
    rec["bmrp"] = col("bp") - col("rp")
    # enforce non-negative parallax (some VizieR rows pass quality with Plx=0)
    return rec[rec["plx"] > 0]


def unit_dirs(ra_deg, dec_deg):
    """Unit direction vectors (N,3) from RA/Dec in degrees (ICRS)."""
    ra = np.radians(np.asarray(ra_deg, dtype='f8'))
    dec = np.radians(np.asarray(dec_deg, dtype='f8'))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


def fetch_hipparcos(v_lim, timeout):
    """Fetch bright stars from Hipparcos, propagated to the Gaia J2016.0 epoch.

    Returns a GAIA_BIN_DTYPE array (V mag stored as G; BP-RP approximated as
    1.0*(B-V) + 0.2), or an empty array on failure.
    """
    adql = ('SELECT RAICRS AS ra, DEICRS AS dec, Plx AS plx, e_Plx AS eplx, '
            'pmRA AS pmra, pmDE AS pmdec, Vmag AS vmag, "B-V" AS bv '
            'FROM "I/239/hip_main" '
            f'WHERE Vmag < {v_lim} AND Plx > 0')
    print(f"Fetching Hipparcos supplement (V < {v_lim:g}) ...")
    try:
        csv_text = run_tap_query("vizier", adql, timeout)
        rows = np.genfromtxt(io.StringIO(csv_text.strip()), delimiter=",",
                             names=True, dtype=None, encoding="utf-8")
        rows = np.atleast_1d(rows)
    except Exception as e:
        print(f"  WARNING: Hipparcos supplement failed ({e}) — continuing without it")
        return np.zeros(0, dtype=GAIA_BIN_DTYPE)

    def col(name):
        v = np.array(rows[name.lower()], dtype='f8')
        return np.where(np.isfinite(v), v, 0.0)

    ra, dec = col("ra"), col("dec")
    plx, pmra, pmdec = col("plx"), col("pmra"), col("pmdec")
    vmag, bv = col("vmag"), col("bv")

    # Propagate J1991.25 -> J2016.0 (24.75 yr) along great circles
    dirs = unit_dirs(ra, dec)
    dec_r, ra_r = np.radians(dec), np.radians(ra)
    east = np.stack([-np.sin(ra_r), np.cos(ra_r), np.zeros_like(ra_r)], axis=1)
    north = np.stack([-np.sin(dec_r) * np.cos(ra_r),
                      -np.sin(dec_r) * np.sin(ra_r), np.cos(dec_r)], axis=1)
    mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    pm_vec = (pmra[:, None] * east + pmdec[:, None] * north) * mas_to_rad  # rad/yr
    dt = GAIA_EPOCH - HIPPARCOS_EPOCH
    dirs = dirs + pm_vec * dt
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Back to RA/Dec (deg) at J2016.0
    ra_new = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 360.0
    dec_new = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0)))

    rec = np.zeros(rows.size, dtype=GAIA_BIN_DTYPE)
    rec["ra"] = ra_new
    rec["dec"] = dec_new
    rec["plx"] = plx
    rec["pmra"] = pmra
    rec["pmdec"] = pmdec
    rec["g"] = vmag                 # V mag as G-mag proxy (bright stars)
    rec["bmrp"] = np.clip(1.0 * bv + 0.2, -0.5, 5.0)  # BP-RP ~ (B-V) approx
    print(f"  Hipparcos supplement: {rec.size} stars (propagated to J{GAIA_EPOCH:.1f})")
    return rec


def dedupe(gaia_rec, hip_rec, sep_arcsec=10.0):
    """Drop Gaia entries that have a Hipparcos bright-star counterpart.

    Hipparcos is authoritative for V < 2.5: Gaia photometry for these stars is
    unreliable due to detector saturation (e.g. Sirius A appears at G=8.5 in
    DR3 instead of -1.46). Hipparcos positions are already propagated to the
    same J2016.0 epoch, so true counterparts land well within sep_arcsec.
    """
    if hip_rec.size == 0 or gaia_rec.size == 0 or cKDTree is None:
        return gaia_rec
    tree = cKDTree(unit_dirs(hip_rec["ra"], hip_rec["dec"]))
    chord = 2.0 * np.sin(np.radians(sep_arcsec / 3600.0) / 2.0)
    dist, _ = tree.query(unit_dirs(gaia_rec["ra"], gaia_rec["dec"]), k=1)
    return gaia_rec[dist > chord]


def write_binary(out_path, records):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(GAIA_MAGIC)
        f.write(np.uint32(GAIA_VERSION).tobytes())
        f.write(np.uint32(records.size).tobytes())
        f.write(records.astype(GAIA_BIN_DTYPE).tobytes())


def print_summary(records):
    n = records.size
    if n == 0:
        print("No stars fetched.")
        return
    dist_pc = 1000.0 / records["plx"]
    nearest = np.argmin(dist_pc)
    print("\n=== GAIA catalog summary ===")
    print(f"Stars:           {n:,}")
    print(f"G mag range:     {records['g'].min():.2f} .. {records['g'].max():.2f}")
    print(f"BP-RP range:     {records['bmrp'].min():.2f} .. {records['bmrp'].max():.2f}")
    print(f"Nearest star:    {dist_pc[nearest]:.3f} pc "
          f"({dist_pc[nearest] * 3.26156:.3f} ly) at "
          f"RA={records['ra'][nearest]:.4f}° Dec={records['dec'][nearest]:.4f}°")
    print(f"                 (expect ~1.34 pc / 4.37 ly = Alpha Centauri A)")
    print(f"Within 50 pc:    {np.count_nonzero(dist_pc < 50):,}")
    print(f"Within 500 pc:   {np.count_nonzero(dist_pc < 500):,}")
    with_pm = np.count_nonzero(np.hypot(records["pmra"], records["pmdec"]) > 100.0)
    print(f"pm > 100 mas/yr: {with_pm:,}")


def main():
    ap = argparse.ArgumentParser(description="Fetch Gaia DR3 starfield binary")
    ap.add_argument("--source", choices=list(TAP_ENDPOINTS), default="esa")
    ap.add_argument("--max-mag", type=float, default=10.0,
                    help="G magnitude limit (default 10 ~ 460k stars)")
    ap.add_argument("--plx-err-cut", type=float, default=5.0,
                    help="minimum parallax_over_error (default 5)")
    ap.add_argument("--out", default=None,
                    help="output path (default data/gaia/stars.bin)")
    ap.add_argument("--timeout", type=int, default=180, help="per-query timeout (s)")
    ap.add_argument("--no-hipparcos", action="store_true",
                    help="skip the bright-star Hipparcos supplement")
    ap.add_argument("--hip-limit", type=float, default=HIP_DEFAULT_VLIM,
                    help="V magnitude limit for the Hipparcos supplement (default 2.5)")
    ap.add_argument("--test", action="store_true",
                    help="quick sample: G < 7 (~14k stars)")
    args = ap.parse_args()

    if args.test:
        args.max_mag = 7.0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(script_dir, "..", "data", "gaia", "stars.bin")
    out_path = os.path.normpath(out_path)

    print(f"Fetching Gaia DR3 (source={args.source}, G < {args.max_mag:g}, "
          f"plx/error > {args.plx_err_cut:g})")
    print(f"Output: {out_path}")

    chunks = []
    t_start = time.time()
    for i in range(NUM_RA_BANDS):
        ra0, ra1 = i * 360.0 / NUM_RA_BANDS, (i + 1) * 360.0 / NUM_RA_BANDS
        adql = build_query(args.source, ra0, ra1, args.max_mag, args.plx_err_cut)
        rec = None
        for attempt in range(1, 4):
            try:
                csv_text = run_tap_query(args.source, adql, args.timeout)
                rec = parse_csv_to_records(csv_text, args.source)
                break
            except Exception as e:
                print(f"  band [{ra0:5.1f}°, {ra1:5.1f}°) attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(3.0 * attempt)
        if rec is None:
            print(f"  WARNING: band [{ra0:.1f}°, {ra1:.1f}°) failed after 3 attempts — skipping")
            continue
        chunks.append(rec)
        total = sum(c.size for c in chunks)
        print(f"  band [{ra0:5.1f}°, {ra1:5.1f}°): {rec.size:>7,} stars  (total {total:,})")

    if not chunks:
        print("ERROR: all bands failed — no output written.")
        sys.exit(1)

    records = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    if not args.no_hipparcos:
        hip = fetch_hipparcos(args.hip_limit, args.timeout)
        if hip.size:
            gaia_kept = dedupe(records, hip)
            print(f"  dedupe: {records.size - gaia_kept.size} Gaia entries replaced "
                  f"by {hip.size} Hipparcos stars")
            records = np.concatenate([gaia_kept, hip])

    order = np.argsort(records["g"])
    records = records[order]

    write_binary(out_path, records)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWrote {records.size:,} stars ({size_mb:.1f} MB) in {time.time() - t_start:.0f}s")
    print_summary(records)


if __name__ == "__main__":
    main()
