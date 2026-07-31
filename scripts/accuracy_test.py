import sys
import json
import time
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
from engine.physics.physics_core import Simulation, attach_custom_forces
from engine.ephemeris.spice_manager import SpiceManager
from engine.core.constants import C_AU_YR, SOLAR_RADII_TO_AU, AU_TO_KM
from engine.core.math_utils import pole_to_ecliptic
from fetch_horizons import query_horizons, parse_state_vector, HORIZONS_IDS, PARENT_NAIF, apply_barycentric_velocity_alignment

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

START_TIME = "2025-01-01 12:00"
STOP_TIME = "2025-01-02"

END_START = "2026-01-01 12:00"
END_STOP = "2026-01-02"

def get_parent_center(body_name, parent_name):
    if body_name == "Sun":
        return "500@0"
    if parent_name == "Sun" or parent_name is None:
        return "500@10"
    parent_naif = HORIZONS_IDS.get(parent_name, PARENT_NAIF.get(parent_name))
    if parent_naif:
        return f"500@{parent_naif}"
    return "500@10"

def main():
    print(f"Loading system.json...")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/system.json')
    with open(data_path, "r") as f:
        bodies = json.load(f)
    
    sim = Simulation()
    sim.G = 39.476926421373
    
    print(f"\n--- Fetching 2025 Starting State Vectors from JPL Horizons ---")
    start_vectors = {}
    for body in bodies:
        name = body["name"]
        parent_name = body.get("parentId")
        body_id = HORIZONS_IDS.get(name)
        if not body_id:
            print(f"Skipping {name} (No Horizons ID)")
            continue
        center = get_parent_center(name, parent_name)
        
        print(f"Querying {name} at {START_TIME}...")
        resp = query_horizons(body_id, center, start_time=START_TIME, stop_time=STOP_TIME)
        sv = parse_state_vector(resp)
        if sv:
            start_vectors[name] = sv
        else:
            print(f"Failed to parse {name}")

    apply_barycentric_velocity_alignment(bodies, start_time=START_TIME, stop_time=STOP_TIME, state_vectors=start_vectors)

    for body in bodies:
        name = body["name"]
        if name not in start_vectors:
            continue
        sv = start_vectors[name]
        m = body.get("m", 0.0)
        parent_name = body.get("parentId")

        if name == "Sun":
            sim.add(m=m, x=0, y=0, z=0, vx=0, vy=0, vz=0)
        elif parent_name and parent_name in start_vectors and parent_name != "Sun":
            p_sv = start_vectors[parent_name]
            sim.add(m=m, 
                    x=p_sv["x"] + sv["x"], 
                    y=p_sv["y"] + sv["y"], 
                    z=p_sv["z"] + sv["z"], 
                    vx=p_sv["vx"] + sv["vx"], 
                    vy=p_sv["vy"] + sv["vy"], 
                    vz=p_sv["vz"] + sv["vz"])
        else:
            sim.add(m=m, x=sv["x"], y=sv["y"], z=sv["z"], vx=sv["vx"], vy=sv["vy"], vz=sv["vz"])

    sim.move_to_com()

    # Apply J2 and GR logic
    oblate_physics_list = []
    phys_star_idx = -1
    phys_star_mass = 0.0

    for i, body in enumerate(bodies):
        name = body["name"]
        if name not in start_vectors:
            continue
        if body.get("type") == "Star":
            phys_star_idx = i
            phys_star_mass = body.get("m", 0.0)
        
        j2 = body.get("J2", 0.0)
        if j2 > 0:
            req_km = body.get("req_km")
            if req_km:
                req = req_km / AU_TO_KM
            else:
                r_mean = body.get("r", 1.0)
                f = body.get("oblateness", 0.0)
                if f > 0:
                    req = r_mean / ((1.0 - f)**(1.0/3.0))
                else:
                    req = r_mean
                req = req * SOLAR_RADII_TO_AU
            if "pole_ra" in body and "pole_dec" in body:
                pole_ecl = pole_to_ecliptic(body["pole_ra"], body["pole_dec"])
                pole = np.array(pole_ecl, dtype=np.float64)
            else:
                pole = np.array([0, 0, 1], dtype=np.float64)
                
            m = body.get("m", 0.0)
            j4 = body.get("j4", 0.0)
            name = body.get("name", "")
            rot_period = body.get("rotation_period", 0.0)
            oblate_physics_list.append((i, j2, j4, req, pole, m, name, rot_period))
            
    oblate_indices = np.array([x[0] for x in oblate_physics_list], dtype=np.int32)
    oblate_j2 = np.array([x[1] for x in oblate_physics_list], dtype=np.float64)
    oblate_j4 = np.array([x[2] for x in oblate_physics_list], dtype=np.float64)
    oblate_req = np.array([x[3] for x in oblate_physics_list], dtype=np.float64)
    oblate_pole = np.array([x[4] for x in oblate_physics_list], dtype=np.float64)
    oblate_masses = np.array([x[5] for x in oblate_physics_list], dtype=np.float64)

    has_j2 = bool(oblate_physics_list)
    has_gr = phys_star_idx >= 0

    if has_j2 or has_gr:
        attach_custom_forces(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list)

    print(f"\n--- Integrating 1 Calendar Year Forward (365 days) ---")
    start_t = time.time()
    exact_years = 365.0 / 365.25  # Exactly 365 days converted to Rebound's Julian year
    sim.integrate(exact_years)
    print(f"Integration complete in {time.time() - start_t:.2f} seconds.")
    
    print(f"\n--- Fetching 2025 Ground Truth State Vectors from JPL Horizons ---")
    end_vectors = {}
    for body in bodies:
        name = body["name"]
        parent_name = body.get("parentId")
        body_id = HORIZONS_IDS.get(name)
        if not body_id:
            continue
        center = get_parent_center(name, parent_name)
        print(f"Querying {name} at {END_START}...")
        resp = query_horizons(body_id, center, start_time=END_START, stop_time=END_STOP)
        sv = parse_state_vector(resp)
        if sv:
            end_vectors[name] = sv

    print(f"\n{'='*95}")
    print(f"--- 1-YEAR ACCURACY TEST RESULTS (2025 -> 2026) ---")
    print(f"{'='*95}")
    print(f"{'Body Name':<15} | {'Total Drift (km)':>18} | {'Ahead/Behind (km)':>18} | {'Radial (km)':>14} | {'Cross-Track (km)':>15}")
    print(f"{'-'*15}-+-{'-'*18}-+-{'-'*18}-+-{'-'*14}-+-{'-'*15}")
    
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
        
        horizons_sv = end_vectors[name]
        hx, hy, hz = horizons_sv["x"], horizons_sv["y"], horizons_sv["z"]
        hvx, hvy, hvz = horizons_sv["vx"], horizons_sv["vy"], horizons_sv["vz"]
        
        err_x = sim_dx - hx
        err_y = sim_dy - hy
        err_z = sim_dz - hz
        
        err_vec = np.array([err_x, err_y, err_z])
        r_vec = np.array([hx, hy, hz])
        v_vec = np.array([hvx, hvy, hvz])
        
        # Calculate RTN unit vectors (Radial, Transverse/Along-Track, Normal/Cross-Track)
        r_norm = np.linalg.norm(r_vec)
        if r_norm > 0:
            R_hat = r_vec / r_norm
            C_vec = np.cross(r_vec, v_vec)
            c_norm = np.linalg.norm(C_vec)
            if c_norm > 0:
                C_hat = C_vec / c_norm
                A_hat = np.cross(C_hat, R_hat)
                
                radial_err = np.dot(err_vec, R_hat) * AU_TO_KM
                along_err = np.dot(err_vec, A_hat) * AU_TO_KM
                cross_err = np.dot(err_vec, C_hat) * AU_TO_KM
            else:
                radial_err = along_err = cross_err = 0.0
        else:
            radial_err = along_err = cross_err = 0.0
        
        dist_au = math.sqrt(err_x*err_x + err_y*err_y + err_z*err_z)
        dist_km = dist_au * AU_TO_KM
        
        print(f"{name:<15} | {dist_km:>15.2f} km | {along_err:>15.2f} km | {radial_err:>11.2f} km | {cross_err:>12.2f} km")

if __name__ == '__main__':
    main()
