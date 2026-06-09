"""Test: Is Pluto's error caused by the Pluto-Charon binary system dynamics?

Hypothesis: When the accuracy test fetches Pluto's START position from Horizons
(ID=999, center=500@10), it gets Pluto's BODY CENTER relative to the Sun.
But in REBOUND, Pluto is a point mass. The test adds Pluto at its body center
and Charon at its body center (relative to Sun).

The issue may be that Horizons' state vector for "Pluto body" (999) includes
the gravitational pull of Charon in its ephemeris computation (using a full
dynamical model of the Pluto system), while our N-body simulation treats
Pluto and Charon as simple point masses without the detailed tidal/rotational
coupling that JPL models.

But more interesting: What if we remove Charon entirely and use the SYSTEM
barycenter (Horizons ID=9) for Pluto? This would test whether the binary
dynamics are the source of the error.
"""
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import rebound
from fetch_horizons import query_horizons, parse_state_vector, HORIZONS_IDS, PARENT_NAIF
from main import setup_reboundx, C_AU_YR, pole_to_ecliptic

START_TIME = "2025-01-01 12:00"
STOP_TIME = "2025-01-02"
END_START = "2026-01-01 12:00"
END_STOP = "2026-01-02"
AU_TO_KM = 149597870.7

def get_parent_center(body_name, parent_name):
    if body_name == "Sun":
        return "500@0"
    if parent_name == "Sun":
        return "500@10"
    parent_naif = HORIZONS_IDS.get(parent_name, PARENT_NAIF.get(parent_name))
    if parent_naif:
        return f"500@{parent_naif}"
    return "500@10"

def run_test(label, use_pluto_bary=False, remove_charon=False):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/system.json')
    with open(data_path, "r") as f:
        bodies = json.load(f)
    
    if remove_charon:
        bodies = [b for b in bodies if b["name"] != "Charon"]
    
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = "ias15"
    
    start_vectors = {}
    for body in bodies:
        name = body["name"]
        parent = body.get("parent")
        body_id = HORIZONS_IDS.get(name)
        if not body_id:
            continue
        
        # For Pluto, optionally use system barycenter
        if name == "Pluto" and use_pluto_bary:
            body_id = "9"  # Pluto system barycenter
            # Also combine masses
            body["m"] = body["m"] + 7.926910e-10  # Add Charon mass
        
        center = get_parent_center(name, parent)
        resp = query_horizons(body_id, center, start_time=START_TIME, stop_time=STOP_TIME)
        sv = parse_state_vector(resp)
        if sv:
            start_vectors[name] = sv
            m = body.get("m", 0.0)
            if name == "Sun":
                sim.add(m=m, x=0, y=0, z=0, vx=0, vy=0, vz=0)
            else:
                sim.add(m=m, x=sv["x"], y=sv["y"], z=sv["z"], vx=sv["vx"], vy=sv["vy"], vz=sv["vz"])
    
    sim.move_to_com()
    
    oblate_physics_list = []
    phys_star_idx = -1
    
    for i, body in enumerate(bodies):
        name = body["name"]
        if name not in start_vectors:
            continue
        if body.get("type") == "Star":
            phys_star_idx = i
        
        j2 = body.get("J2", 0.0)
        if j2 > 0:
            r_mean = body.get("r", 1.0)
            f = body.get("oblateness", 0.0)
            if f > 0:
                req = r_mean / ((1.0 - f)**(1.0/3.0))
            else:
                req = r_mean
            req = req * 0.00465
            if "pole_ra" in body and "pole_dec" in body:
                pole_ecl = pole_to_ecliptic(body["pole_ra"], body["pole_dec"])
                pole = np.array(pole_ecl, dtype=np.float64)
            else:
                pole = np.array([0, 1, 0], dtype=np.float64)
            
            m = body.get("m", 0.0)
            j4 = body.get("j4", 0.0)
            rot_period = body.get("rotation_period", 0.0)
            oblate_physics_list.append((i, j2, j4, req, pole, m, name, rot_period))
    
    has_j2 = bool(oblate_physics_list)
    has_gr = phys_star_idx >= 0
    
    if has_j2 or has_gr:
        setup_reboundx(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list)
        sim.force_is_velocity_dependent = 1
    
    exact_years = 365.0 / 365.25
    t0 = time.time()
    sim.integrate(exact_years)
    dt = time.time() - t0
    
    # Fetch end comparison vector
    # For Pluto with bary mode, compare against system bary
    pluto_end_id = "9" if use_pluto_bary else "999"
    resp = query_horizons(pluto_end_id, "500@10", start_time=END_START, stop_time=END_STOP)
    sv_pluto_end = parse_state_vector(resp)
    
    # Find Pluto in sim
    pluto_idx = next(i for i, b in enumerate(bodies) if b["name"] == "Pluto")
    sun_idx = 0
    
    p = sim.particles[pluto_idx]
    parent_p = sim.particles[sun_idx]
    
    sim_dx = p.x - parent_p.x
    sim_dy = p.y - parent_p.y
    sim_dz = p.z - parent_p.z
    
    hx, hy, hz = sv_pluto_end["x"], sv_pluto_end["y"], sv_pluto_end["z"]
    hvx, hvy, hvz = sv_pluto_end["vx"], sv_pluto_end["vy"], sv_pluto_end["vz"]
    err_x = sim_dx - hx
    err_y = sim_dy - hy
    err_z = sim_dz - hz
    
    dist_km = math.sqrt(err_x**2 + err_y**2 + err_z**2) * AU_TO_KM
    
    r_vec = np.array([hx, hy, hz])
    v_vec = np.array([hvx, hvy, hvz])
    err_vec = np.array([err_x, err_y, err_z])
    
    r_norm = np.linalg.norm(r_vec)
    R_hat = r_vec / r_norm
    C_vec = np.cross(r_vec, v_vec)
    C_hat = C_vec / np.linalg.norm(C_vec)
    A_hat = np.cross(C_hat, R_hat)
    
    return {
        "total": dist_km,
        "along": np.dot(err_vec, A_hat) * AU_TO_KM,
        "radial": np.dot(err_vec, R_hat) * AU_TO_KM,
        "cross": np.dot(err_vec, C_hat) * AU_TO_KM,
    }, dt

print("=" * 90)
print("PLUTO BINARY SYSTEM TEST")  
print("=" * 90)

print("\n[1/3] Pluto body (999) + Charon separately (current behavior)...")
r1, dt1 = run_test("original")
print(f"  Done in {dt1:.1f}s -> {r1['total']:.2f} km")

print("[2/3] Pluto system bary (9), no Charon, combined mass...")
r2, dt2 = run_test("bary", use_pluto_bary=True, remove_charon=True)
print(f"  Done in {dt2:.1f}s -> {r2['total']:.2f} km")

print("[3/3] Pluto body (999), no Charon, Pluto mass only...")
r3, dt3 = run_test("no_charon", remove_charon=True)
print(f"  Done in {dt3:.1f}s -> {r3['total']:.2f} km")

print(f"\n{'Test':>40s} | {'Total':>10s} | {'Along':>10s} | {'Radial':>10s} | {'Cross':>10s}")
print("-" * 90)
print(f"{'Pluto(999) + Charon (original)':>40s} | {r1['total']:10.2f} | {r1['along']:+10.2f} | {r1['radial']:+10.2f} | {r1['cross']:+10.2f}")
print(f"{'System bary(9), no Charon':>40s} | {r2['total']:10.2f} | {r2['along']:+10.2f} | {r2['radial']:+10.2f} | {r2['cross']:+10.2f}")
print(f"{'Pluto(999), no Charon':>40s} | {r3['total']:10.2f} | {r3['along']:+10.2f} | {r3['radial']:+10.2f} | {r3['cross']:+10.2f}")
