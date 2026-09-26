import os
import sys
import numpy as np
import spiceypy as spice

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.ephemeris.spice_manager import SpiceManager, get_spice_manager
from engine.ephemeris.spk_exporter import export_timeline_spk
from engine.physics.physics_core import load_system_from_data
from engine.rendering.render_utils import format_sim_time, format_sim_time_utc, sim_time_from_date

def test_generic_bsp_viewer():
    print("=" * 70)
    print(" Testing Generic SPICE BSP Viewer & Playback Pipeline")
    print("=" * 70)

    mgr = SpiceManager()
    assert get_spice_manager() is mgr, "get_spice_manager() singleton reference failed"
    print("1. Singleton initialization OK")

    # Step A: Discover files
    files = mgr.list_available_bsp_files()
    assert len(files) > 0, "No .bsp files found by list_available_bsp_files"
    print(f"2. File discovery OK: Found {len(files)} kernels")

    # Step B: Test inspection of an existing NASA kernel (cassini.bsp)
    cassini_path = "data/kernels/cassini.bsp"
    assert os.path.exists(cassini_path), f"Missing {cassini_path}"
    c_info = mgr.inspect_bsp(cassini_path)
    assert c_info is not None, "Failed to inspect cassini.bsp"
    assert c_info["filename"] == "cassini.bsp"
    assert -82 in c_info["body_ids"], "Cassini NAIF ID -82 not found in body_ids"
    assert len(c_info["bodies"]) == 1
    assert "2004" in c_info["utc_range"][0] and "2017" in c_info["utc_range"][1]
    print(f"3. Inspection of NASA kernel (Cassini) OK: {c_info['utc_range'][0]} to {c_info['utc_range'][1]}")

    # Step C: Test build_generic_bsp_system for Cassini
    c_bndl = mgr.build_generic_bsp_system(cassini_path, num_samples=500)
    assert c_bndl is not None, "Failed to build bundle for cassini"
    assert len(c_bndl["bodies_data"]) == 1
    assert len(c_bndl["trajectories"]) == 1
    traj = c_bndl["trajectories"][0]
    assert traj["points"].shape == (500, 3)
    assert not np.isnan(traj["points"]).any(), "NaNs found in Cassini polyline points"
    assert not np.isinf(traj["points"]).any(), "Infs found in Cassini polyline points"

    # Verify load_system_from_data compatibility
    sim_bndl = load_system_from_data(c_bndl["bodies_data"])
    assert sim_bndl["num_bodies"] == 1
    print("4. Generic system build for Cassini OK (500-point polyline generated, simulation built)")

    # Step D: Test export and import of a custom multi-body system with sidecar
    test_bsp = "exports/ephemeris/TestSystem/test_orbit_system.bsp"
    os.makedirs(os.path.dirname(test_bsp), exist_ok=True)
    t_steps = 400
    times = np.linspace(0.0, 4.0, t_steps)
    states = np.zeros((t_steps, 2, 6))
    # Star stationary
    # Planet on 1 AU circular orbit
    w = 2.0 * np.pi
    states[:, 1, 0] = np.cos(w * times)
    states[:, 1, 1] = np.sin(w * times)
    states[:, 1, 3] = -w * np.sin(w * times)
    states[:, 1, 4] = w * np.cos(w * times)

    body_names = ["Test Star", "Test Planet"]
    ok, msg, p = export_timeline_spk(test_bsp, states, times, body_names, spk_type=2)
    assert ok, f"Failed to export test SPK: {msg}"

    # Inspect the exported test SPK
    t_info = mgr.inspect_bsp(test_bsp)
    assert t_info is not None, "Failed to inspect exported test SPK"
    assert t_info["sidecar"] is not None, "Sidecar not loaded for exported test SPK"
    assert len(t_info["bodies"]) == 2
    print(f"5. Inspection of exported system OK: {t_info['filename']}, format: Type {t_info['sidecar']['spk_type']} Chebyshev")

    # Build system from exported test SPK
    t_bndl = mgr.build_generic_bsp_system(test_bsp, num_samples=1000)
    assert t_bndl is not None
    assert len(t_bndl["bodies_data"]) == 2
    assert len(t_bndl["trajectories"]) == 2
    assert t_bndl["trajectories"][0]["points"].shape == (1000, 3)
    assert t_bndl["trajectories"][1]["points"].shape == (1000, 3)

    # Test load_system_from_data on the exported system
    loaded_sys = load_system_from_data(t_bndl["bodies_data"])
    assert loaded_sys["num_bodies"] == 2
    print("6. System bundle from exported SPK built and loaded into REBOUND successfully")

    # Step E: Test populate_states_fast playback evaluation
    mgr.active_epoch_et = t_bndl["epoch_et"]
    mgr.active_ephem_bsp = test_bsp
    mgr.active_ephem_et_range = t_bndl["et_range"]

    mapping = mgr.get_body_mapping(t_bndl["bodies_data"], t_bndl["epoch_et"])
    assert mapping == [b["spice_id"] for b in t_bndl["bodies_data"]]

    pos_out = np.zeros((2, 3), dtype=np.float64)
    vel_out = np.zeros((2, 3), dtype=np.float64)
    valid_out = np.zeros(2, dtype=bool)

    # Evaluate at t = 0.25 years (90 degrees along orbit)
    mid_t = 0.25
    et_query = mgr.active_epoch_et + mid_t * (86400.0 * 365.25)
    mgr.populate_states_fast(et_query, mapping, pos_out, vel_out, valid_out)

    assert valid_out.all(), "populate_states_fast failed valid mask"
    # Planet at 0.25 years should be at (cos(pi/2), sin(pi/2)) = (0, 1) in physics frame -> (0, 0, -1) in render frame
    # Render frame swap: X=x, Y=z, Z=-y
    # In physics frame: x ~ 0.0, y ~ 1.0, z ~ 0.0
    # In render frame: X=x ~ 0.0, Y=z ~ 0.0, Z=-y ~ -1.0
    planet_idx = [i for i, b in enumerate(t_bndl["bodies_data"]) if b["name"] == "Test Planet"][0]
    star_idx = [i for i, b in enumerate(t_bndl["bodies_data"]) if b["name"] == "Test Star"][0]
    assert abs(pos_out[planet_idx, 0]) < 1e-3, f"Unexpected X: {pos_out[planet_idx, 0]}"
    assert abs(pos_out[planet_idx, 2] - (-1.0)) < 1e-3, f"Unexpected Z: {pos_out[planet_idx, 2]}"
    assert abs(pos_out[star_idx, 0]) < 1e-3
    assert abs(pos_out[star_idx, 2]) < 1e-3
    print("7. Fast state evaluation (populate_states_fast) playback accuracy verified (< 1e-3 AU)")

    # Step F: Verify date formatting and jumping
    y, m, d, h, mn, s, tz = format_sim_time(0.0)
    assert tz in ["UTC", "WIB", "PST", "EST", "GMT"] or len(tz) > 0
    uy, um, ud, uh, umn, us, _ = format_sim_time_utc(0.0)
    sim_t_rt = sim_time_from_date(uy, um, ud, uh, umn, us, is_utc=True)
    assert abs(sim_t_rt - 0.0) < 1e-4, f"sim_time_from_date round-trip error: {sim_t_rt}"
    print(f"8. Time formatting & date jumping round-trip OK: {uy:04d}-{um:02d}-{ud:02d} {uh:02d}:{umn:02d}:{us:02d} UTC")

    print("=" * 70)
    print(" ALL GENERIC BSP VIEWER TESTS PASSED")
    print("=" * 70)

if __name__ == "__main__":
    test_generic_bsp_viewer()
