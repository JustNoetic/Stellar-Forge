"""Test: Does correcting masses to JPL DE441 values improve accuracy?

We test with corrected GM values for the key bodies:
- Neptune: system.json is -0.0209% off
- Pluto: +0.079%  
- Charon: -0.642% (!)

Also test with Pluto system fetched from Horizons system barycenter (ID=9)
vs body center (ID=999).
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

# JPL DE441 GM values (km^3/s^2) converted to solar masses
GM_SUN = 1.32712440018e11
CORRECTED_MASSES = {
    "Neptune": 6.836527100580397e6 / GM_SUN,   # DE441
    "Pluto":   8.696138e2 / GM_SUN,             # Pluto body only
    "Charon":  1.058799e2 / GM_SUN,             # Charon body
    "Uranus":  5.7939399e6 / GM_SUN,            # DE441
}

def get_parent_center(body_name, parent_name):
    if body_name == "Sun":
        return "500@0"
    if parent_name == "Sun":
        return "500@10"
    parent_naif = HORIZONS_IDS.get(parent_name, PARENT_NAIF.get(parent_name))
    if parent_naif:
        return f"500@{parent_naif}"
    return "500@10"

def run_test(label, mass_overrides=None):
    if mass_overrides is None:
        mass_overrides = {}
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/system.json')
    with open(data_path, "r") as f:
        bodies = json.load(f)
    
    # Apply mass corrections
    for body in bodies:
        if body["name"] in mass_overrides:
            body["m"] = mass_overrides[body["name"]]
    
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
    
    # Fetch end vectors for key bodies
    target_bodies = ["Uranus", "Neptune", "Pluto", "Charon"]
    end_vectors = {}
    for body in bodies:
        name = body["name"]
        if name not in target_bodies:
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
        hx, hy, hz = sv["x"], sv["y"], sv["z"]
        hvx, hvy, hvz = sv["vx"], sv["vy"], sv["vz"]
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
        c_norm = np.linalg.norm(C_vec)
        if c_norm > 0:
            C_hat = C_vec / c_norm
            A_hat = np.cross(C_hat, R_hat)
            results[name] = {
                "total": dist_km,
                "along": np.dot(err_vec, A_hat) * AU_TO_KM,
                "radial": np.dot(err_vec, R_hat) * AU_TO_KM,
                "cross": np.dot(err_vec, C_hat) * AU_TO_KM,
            }
        else:
            results[name] = {"total": dist_km, "along": 0, "radial": 0, "cross": 0}
    
    return results, dt

print("=" * 90)
print("MASS CORRECTION TEST")
print("=" * 90)

print("\n[1/2] Running with ORIGINAL masses...")
results_orig, dt1 = run_test("original")
print(f"  Done in {dt1:.1f}s")

print("[2/2] Running with CORRECTED JPL DE441 masses...")
results_fixed, dt2 = run_test("corrected", mass_overrides=CORRECTED_MASSES)
print(f"  Done in {dt2:.1f}s")

print(f"\n{'Body':12s} | {'ORIGINAL (km)':>14s} | {'CORRECTED (km)':>16s} | {'Improvement':>14s}")
print("-" * 70)
for name in ["Uranus", "Neptune", "Pluto", "Charon"]:
    if name in results_orig and name in results_fixed:
        r1 = results_orig[name]["total"]
        r2 = results_fixed[name]["total"]
        improvement = r1 - r2
        pct = improvement / r1 * 100 if r1 > 0 else 0
        sign = "BETTER" if improvement > 0 else "WORSE"
        print(f"{name:12s} | {r1:14.2f} | {r2:16.2f} | {improvement:+12.2f} km ({sign})")

print(f"\nDetailed RTN breakdown:")
print("-" * 90)
for name in ["Uranus", "Neptune", "Pluto", "Charon"]:
    if name in results_orig and name in results_fixed:
        f = results_orig[name]
        n = results_fixed[name]
        print(f"\n  {name}:")
        print(f"    {'Component':12s} | {'ORIGINAL':>14s} | {'CORRECTED':>14s} | {'Delta':>14s}")
        print(f"    {'-'*60}")
        for comp in ["along", "radial", "cross"]:
            d = n[comp] - f[comp]
            print(f"    {comp:12s} | {f[comp]:+14.2f} | {n[comp]:+14.2f} | {d:+14.2f} km")
