"""Test: Does disabling Uranus/Neptune J2 improve their accuracy?

This runs the accuracy test twice:
1. Full physics (current behavior) 
2. With J2/J4 disabled for Uranus and Neptune only

If Uranus's drift improves significantly without J2, it means the J2 force
is somehow affecting its heliocentric orbit (which shouldn't happen, but 
REBOUNDx gravitational_harmonics may have subtle back-reaction effects).
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

def run_test(label, disable_j2_bodies=None):
    """Run integration and return errors for key bodies."""
    if disable_j2_bodies is None:
        disable_j2_bodies = set()
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/system.json')
    with open(data_path, "r") as f:
        bodies = json.load(f)
    
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
        if j2 > 0 and name not in disable_j2_bodies:
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
    
    # Fetch end vectors
    end_vectors = {}
    for body in bodies:
        name = body["name"]
        if name not in ["Uranus", "Neptune", "Pluto"]:
            continue
        parent_name = body.get("parentId")
        body_id = HORIZONS_IDS.get(name)
        if not body_id:
            continue
        center = get_parent_center(name, parent_name)
        resp = query_horizons(body_id, center, start_time=END_START, stop_time=END_STOP)
        sv = parse_state_vector(resp)
        if sv:
            end_vectors[name] = sv
    
    results = {}
    for i, body in enumerate(bodies):
        name = body["name"]
        if name not in end_vectors or name == "Sun":
            continue
        
        p = sim.particles[i]
        parent_name = body.get("parentId")
        parent_idx = next((j for j, b in enumerate(bodies) if b["name"] == parent_name), 0) if parent_name else 0
        parent_p = sim.particles[parent_idx]
        
        sim_dx = p.x - parent_p.x
        sim_dy = p.y - parent_p.y
        sim_dz = p.z - parent_p.z
        
        sv = end_vectors[name]
        err_x = sim_dx - sv["x"]
        err_y = sim_dy - sv["y"]
        err_z = sim_dz - sv["z"]
        
        dist_km = math.sqrt(err_x**2 + err_y**2 + err_z**2) * AU_TO_KM
        
        # RTN decomposition
        r_vec = np.array([sv["x"], sv["y"], sv["z"]])
        v_vec = np.array([sv["vx"], sv["vy"], sv["vz"]])
        err_vec = np.array([err_x, err_y, err_z])
        
        r_norm = np.linalg.norm(r_vec)
        R_hat = r_vec / r_norm
        C_vec = np.cross(r_vec, v_vec)
        C_hat = C_vec / np.linalg.norm(C_vec)
        A_hat = np.cross(C_hat, R_hat)
        
        results[name] = {
            "total": dist_km,
            "along": np.dot(err_vec, A_hat) * AU_TO_KM,
            "radial": np.dot(err_vec, R_hat) * AU_TO_KM,
            "cross": np.dot(err_vec, C_hat) * AU_TO_KM,
        }
    
    return results, dt

print("=" * 90)
print("A/B TEST: Effect of J2 on outer planet accuracy")
print("=" * 90)

print("\n[1/2] Running WITH full J2 physics...")
results_full, dt_full = run_test("full")
print(f"  Done in {dt_full:.1f}s")

print("[2/2] Running WITHOUT Uranus/Neptune J2...")
results_no_j2, dt_no_j2 = run_test("no_j2", disable_j2_bodies={"Uranus", "Neptune"})
print(f"  Done in {dt_no_j2:.1f}s")

print(f"\n{'Body':12s} | {'WITH J2 (km)':>14s} | {'WITHOUT J2 (km)':>16s} | {'Improvement':>14s}")
print("-" * 70)
for name in ["Uranus", "Neptune", "Pluto"]:
    if name in results_full and name in results_no_j2:
        r1 = results_full[name]["total"]
        r2 = results_no_j2[name]["total"]
        improvement = r1 - r2
        pct = improvement / r1 * 100 if r1 > 0 else 0
        print(f"{name:12s} | {r1:14.2f} | {r2:16.2f} | {improvement:+12.2f} km ({pct:+.1f}%)")

print(f"\n{'RTN Decomposition':>20s}")
print("-" * 90)
for name in ["Uranus", "Neptune", "Pluto"]:
    if name in results_full and name in results_no_j2:
        f = results_full[name]
        n = results_no_j2[name]
        print(f"\n  {name}:")
        print(f"    {'Component':12s} | {'WITH J2':>14s} | {'WITHOUT J2':>14s} | {'Delta':>14s}")
        print(f"    {'-'*60}")
        for comp in ["along", "radial", "cross"]:
            d = n[comp] - f[comp]
            print(f"    {comp:12s} | {f[comp]:+14.2f} | {n[comp]:+14.2f} | {d:+14.2f} km")
