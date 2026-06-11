import reboundx
import numpy as np

def attach_reboundx_forces(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list):
    """
    Attaches reboundx forces for GR and J2/J4.
    """
    rebx = getattr(sim, "_rebx", None)
    if rebx is None:
        rebx = reboundx.Extras(sim)
        sim._rebx = rebx
        
    if has_gr:
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = 63197.79 # C_AU_YR
        
    if has_j2:
        gh = rebx.load_force("gravitational_harmonics")
        rebx.add_force(gh)
        for obl_body in oblate_physics_list:
            idx, j2, j4, req_au, pole_ecl, mass, name, rot_period = obl_body
            p = sim.particles[idx]
            p.params["J2"] = j2
            p.params["J4"] = j4
            p.params["R_eq"] = req_au
            # ReboundX uses Omega vector purely for direction.
            # Passing [0,0,0] causes a division by zero NaN during integration.
            p.params["Omega"] = [float(x) for x in pole_ecl]
