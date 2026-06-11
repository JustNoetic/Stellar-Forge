import math
import numpy as np
import rebound
import ctypes
from numba import njit, prange
import time
from constants import *
from math_utils import *
from system_manager import SystemManager, SystemSnapshot, derive_star_properties
import warnings
from reboundx_physics import attach_reboundx_forces
from render_utils import *

_PARTICLE_STRIDE = None

def _get_particle_array(sim, num_bodies):
    """Get a NumPy view of REBOUND's internal C particle array (zero-copy).

    Returns array of shape (num_bodies, 16) where columns are:
    [0]=x [1]=y [2]=z [3]=vx [4]=vy [5]=vz [6..8]=ax,ay,az [9]=m ...
    """
    global _PARTICLE_STRIDE
    if _PARTICLE_STRIDE is None:
        _PARTICLE_STRIDE = ctypes.sizeof(rebound.Particle) // 8
    n = _PARTICLE_STRIDE
    addr = ctypes.addressof(sim._particles.contents)
    raw = (ctypes.c_double * (num_bodies * n)).from_address(addr)
    return np.frombuffer(raw, dtype=np.float64).reshape(num_bodies, n)

def _extract_render_state(sim, num_bodies, out_pos, out_vel):
    """Extract particle positions/velocities with render coordinate swap (x, z, -y)."""
    arr = _get_particle_array(sim, num_bodies)
    out_pos[:, 0] = arr[:, 0]
    out_pos[:, 1] = arr[:, 2]
    out_pos[:, 2] = -arr[:, 1]
    out_vel[:, 0] = arr[:, 3]
    out_vel[:, 1] = arr[:, 5]
    out_vel[:, 2] = -arr[:, 4]

@njit(cache=True)
def compute_keplerian_elements(rx, ry, rz, vx, vy, vz, mu):
    """Compute classical Keplerian elements from Cartesian state in ecliptic frame.
    
    Args:
        rx, ry, rz: position components (AU, ecliptic)
        vx, vy, vz: velocity components (AU/yr, ecliptic)
        mu: gravitational parameter G*(M_parent + m_body)
    
    Returns tuple of 8 floats:
        (a, e, inc_deg, Omega_deg, omega_deg, nu_deg, M_deg, period_days)
        Returns all zeros if orbit is degenerate.
    """
    r = math.sqrt(rx*rx + ry*ry + rz*rz)
    v2 = vx*vx + vy*vy + vz*vz
    if r < 1e-30 or mu < 1e-30:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    hx = ry*vz - rz*vy
    hy = rz*vx - rx*vz
    hz = rx*vy - ry*vx
    h = math.sqrt(hx*hx + hy*hy + hz*hz)
    if h < 1e-30:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    vhx = vy*hz - vz*hy
    vhy = vz*hx - vx*hz
    vhz = vx*hy - vy*hx
    ex = vhx/mu - rx/r
    ey = vhy/mu - ry/r
    ez = vhz/mu - rz/r
    e = math.sqrt(ex*ex + ey*ey + ez*ez)
    
    energy = v2/2.0 - mu/r
    if abs(energy) < 1e-30:
        a = 1e30
    else:
        a = -mu / (2.0 * energy)
    
    inc = math.acos(max(-1.0, min(1.0, hz / h)))
    
    nx = -hy
    ny = hx
    n_mag = math.sqrt(nx*nx + ny*ny)
    
    if n_mag > 1e-15:
        Omega = math.atan2(ny, nx)
        if Omega < 0:
            Omega += 2.0 * math.pi
    else:
        Omega = 0.0
    
    if n_mag > 1e-15 and e > 1e-10:
        n_dot_e = (nx*ex + ny*ey) / (n_mag * e)
        omega = math.acos(max(-1.0, min(1.0, n_dot_e)))
        if ez < 0:
            omega = 2.0 * math.pi - omega
    elif e > 1e-10:
        omega = math.atan2(ey, ex)
        if omega < 0:
            omega += 2.0 * math.pi
    else:
        omega = 0.0
    
    if e > 1e-10:
        r_dot_e = (rx*ex + ry*ey + rz*ez) / (r * e)
        nu = math.acos(max(-1.0, min(1.0, r_dot_e)))
        rdotv = rx*vx + ry*vy + rz*vz
        if rdotv < 0:
            nu = 2.0 * math.pi - nu
    else:
        if n_mag > 1e-15:
            r_dot_n = (rx*nx + ry*ny) / (r * n_mag)
            nu = math.acos(max(-1.0, min(1.0, r_dot_n)))
            if rz < 0:
                nu = 2.0 * math.pi - nu
        else:
            nu = math.atan2(ry, rx)
            if nu < 0:
                nu += 2.0 * math.pi
    
    if e < 1.0:
        cos_nu = math.cos(nu)
        sin_nu = math.sin(nu)
        sin_E = sin_nu * math.sqrt(max(0.0, 1.0 - e*e)) / (1.0 + e*cos_nu)
        cos_E = (e + cos_nu) / (1.0 + e*cos_nu)
        E = math.atan2(sin_E, cos_E)
        M = E - e * sin_E
        if M < 0:
            M += 2.0 * math.pi
    else:
        M = 0.0
    
    if a > 0 and e < 1.0:
        period_yr = 2.0 * math.pi * math.sqrt(a*a*a / mu)
        period_days = period_yr * 365.25
    else:
        period_days = 0.0
    
    deg = 180.0 / math.pi
    return a, e, inc*deg, Omega*deg, omega*deg, nu*deg, M*deg, period_days

@njit(cache=True)
def compute_orbit_elements(rel_pos, rel_vel, mu, sl_p_scale, bary, cam_origin,
                           flip_ecc, has_theta0_ref, theta0_ref):
    """Compute orbital elements for GPU orbit rendering.
    
    Instead of generating 1001 points on CPU, this returns the lightweight
    orbital parameters that the GPU vertex shader uses to generate points.
    
    Returns tuple (e_hat, q_hat, bary_rel, sl_p, e_mag, theta0, color, is_valid).
    """
    r = fast_norm(rel_pos)
    if mu <= 0 or r <= 0:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0, 0.0, 0.0, False
    
    h_vec = fast_cross(rel_pos, rel_vel)
    e_vec = fast_cross(rel_vel, h_vec) / mu - rel_pos / r
    sl_p = (h_vec[0]*h_vec[0] + h_vec[1]*h_vec[1] + h_vec[2]*h_vec[2]) / mu * sl_p_scale
    
    e_mag = fast_norm(e_vec)
    h_mag = fast_norm(h_vec)
    if h_mag <= 0:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0, 0.0, 0.0, False
    
    h_hat = h_vec / h_mag
    
    if flip_ecc:
        e_vec = -e_vec
        e_mag = fast_norm(e_vec)
    
    if e_mag > 1e-10:
        e_hat = e_vec / e_mag
        q_hat = fast_cross(h_hat, e_hat)
    else:
        ref = np.array([0., 0., 1.], dtype=np.float64) if abs(h_hat[2]) < 0.9 else np.array([1., 0., 0.], dtype=np.float64)
        e_hat = fast_cross(h_hat, ref)
        e_hat /= fast_norm(e_hat)
        q_hat = fast_cross(h_hat, e_hat)
    
    ref_pos = theta0_ref if has_theta0_ref else rel_pos
    theta0 = math.atan2(ref_pos[0]*q_hat[0] + ref_pos[1]*q_hat[1] + ref_pos[2]*q_hat[2],
                        ref_pos[0]*e_hat[0] + ref_pos[1]*e_hat[1] + ref_pos[2]*e_hat[2])
    
    bary_rel = bary - cam_origin
    
    return e_hat, q_hat, bary_rel, float(sl_p), float(e_mag), float(theta0), True

@njit(cache=True)
def compute_barycenters(pos, vel, mass, parent_indices,
                        subsys_pos, subsys_vel, subsys_mass):
    """Compute subsystem barycenters by accumulating children into parents.
    Modifies subsys_pos, subsys_vel, subsys_mass in-place."""
    n = len(mass)
    for i in range(n):
        subsys_mass[i] = mass[i]
        for k in range(3):
            subsys_pos[i, k] = mass[i] * pos[i, k]
            subsys_vel[i, k] = mass[i] * vel[i, k]
    
    for i in range(n):
        pi = parent_indices[i]
        if pi >= 0:
            cm = mass[i]
            subsys_mass[pi] += cm
            for k in range(3):
                subsys_pos[pi, k] += cm * pos[i, k]
                subsys_vel[pi, k] += cm * vel[i, k]
    
    for i in range(n):
        if subsys_mass[i] > 0.0:
            inv_m = 1.0 / subsys_mass[i]
            for k in range(3):
                subsys_pos[i, k] *= inv_m
                subsys_vel[i, k] *= inv_m
        else:
            for k in range(3):
                subsys_pos[i, k] = pos[i, k]
                subsys_vel[i, k] = vel[i, k]

@njit(cache=True)
def compute_all_orbits_batch(pos, vel, mass, parent_indices,
                             subsys_pos, subsys_vel, subsys_mass,
                             visual_colors, cam_origin, G, max_orbits, orbit_buf, lod_levels):
    """Compute all orbital elements (child + reflex) in one Numba-accelerated batch.
    Returns number of valid orbits written to orbit_buf."""
    n_orbits = 0
    num_bodies = len(pos)

    best_child = np.full(num_bodies, -1, dtype=np.int64)
    best_child_mass = np.zeros(num_bodies, dtype=np.float64)

    orb_rel_pos = np.empty(3, dtype=np.float64)
    orb_rel_vel = np.empty(3, dtype=np.float64)
    bary = np.empty(3, dtype=np.float64)
    gp_rel_pos = np.empty(3, dtype=np.float64)
    gp_rel_vel = np.empty(3, dtype=np.float64)
    gp_bary = np.empty(3, dtype=np.float64)
    rel_pos = np.empty(3, dtype=np.float64)
    rel_vel = np.empty(3, dtype=np.float64)
    parent_bary_ref = np.empty(3, dtype=np.float64)
    eff_parent_pos = np.empty(3, dtype=np.float64)
    eff_parent_vel = np.empty(3, dtype=np.float64)

    for i in range(num_bodies):
        p_idx = parent_indices[i]
        if p_idx >= 0:
            if mass[i] > best_child_mass[p_idx]:
                best_child_mass[p_idx] = mass[i]
                best_child[p_idx] = i
                
    for i in range(num_bodies):
        if n_orbits >= max_orbits:
            break
        p_idx = parent_indices[i]
        if p_idx < 0:
            continue
            
        parent_m = mass[p_idx]
        orb_m = subsys_mass[i]
        
        eff_parent_m = parent_m
        for k in range(3):
            eff_parent_pos[k] = parent_m * pos[p_idx, k]
            eff_parent_vel[k] = parent_m * vel[p_idx, k]
            
        dist_i_sq = 0.0
        for k in range(3):
            dx = pos[i, k] - pos[p_idx, k]
            dist_i_sq += dx*dx
            
        for j in range(num_bodies):
            if j != i and parent_indices[j] == p_idx:
                cm = mass[j]
                # Only account for moons that have at least 1/1,000,000th the mass of the primary
                if cm > 1e-6 * parent_m:
                    dist_j_sq = 0.0
                    for k in range(3):
                        dx = pos[j, k] - pos[p_idx, k]
                        dist_j_sq += dx*dx
                    if dist_j_sq < dist_i_sq:
                        eff_parent_m += cm
                        for k in range(3):
                            eff_parent_pos[k] += cm * pos[j, k]
                            eff_parent_vel[k] += cm * vel[j, k]
                        
        for k in range(3):
            eff_parent_pos[k] /= eff_parent_m
            eff_parent_vel[k] /= eff_parent_m

        total_m = eff_parent_m + orb_m
        if total_m <= 0.0:
            continue
            
        q = eff_parent_m / total_m
        mu = G * total_m
        
        for k in range(3):
            orb_rel_pos[k] = subsys_pos[i, k] - eff_parent_pos[k]
            orb_rel_vel[k] = subsys_vel[i, k] - eff_parent_vel[k]
            bary[k] = (eff_parent_m * eff_parent_pos[k] + orb_m * subsys_pos[i, k]) / total_m
            
        e_hat, q_hat, bary_rel, sl_p, e_mag, theta0, is_valid = compute_orbit_elements(
            orb_rel_pos, orb_rel_vel, mu, q, bary, cam_origin,
            False, False, orb_rel_pos)
            
        if is_valid and n_orbits < max_orbits:
            bary_dist = fast_norm(bary_rel)
            if bary_dist > 1e-6 and sl_p / bary_dist < 1e-6:
                pass
            else:
                r_hill = 1e30
                gp_idx = parent_indices[p_idx]
                if gp_idx >= 0:
                    gp_m = mass[gp_idx]
                    dx = pos[p_idx, 0] - pos[gp_idx, 0]
                    dy = pos[p_idx, 1] - pos[gp_idx, 1]
                    dz = pos[p_idx, 2] - pos[gp_idx, 2]
                    dist_p_gp = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if gp_m > 0:
                        r_hill = dist_p_gp * ((parent_m / (3.0 * gp_m))**(1.0/3.0))
                
                is_escape = False
                theta_exit = 0.0
                if e_mag >= 1.0:
                    is_escape = True
                    cos_th = (sl_p / r_hill - 1.0) / e_mag
                    if cos_th < -1.0: cos_th = -1.0
                    if cos_th > 1.0: cos_th = 1.0
                    theta_exit = math.acos(cos_th)
                else:
                    r_apo = sl_p / (1.0 - e_mag)
                    if r_apo > r_hill:
                        is_escape = True
                        cos_th = (sl_p / r_hill - 1.0) / e_mag
                        if cos_th < -1.0: cos_th = -1.0
                        if cos_th > 1.0: cos_th = 1.0
                        theta_exit = math.acos(cos_th)
                
                theta0_curr = theta0
                delta_angle = 10.0
                t_curr = 0.0
                if is_escape:
                    if e_mag >= 1.0:
                        delta_angle = 2.0 * theta_exit
                        theta0 = -theta_exit
                        t_curr = (theta0_curr + theta_exit) / delta_angle if delta_angle > 0 else 0.0
                    else:
                        E_exit = math.atan2(math.sqrt(1.0 - e_mag * e_mag) * math.sin(theta_exit), math.cos(theta_exit) + e_mag)
                        delta_angle = 2.0 * E_exit
                        E0 = -E_exit
                        E0_curr = math.atan2(math.sqrt(1.0 - e_mag * e_mag) * math.sin(theta0_curr), math.cos(theta0_curr) + e_mag)
                        t_curr = (E0_curr + E_exit) / delta_angle if delta_angle > 0 else 0.0
                elif e_mag >= 1.0:
                    r_max = max(300.0, (sl_p / (1.0 + e_mag)) * 10.0)
                    cos_th_max = (sl_p / r_max - 1.0) / e_mag
                    cos_th_max = max(-1.0, min(1.0, cos_th_max))
                    theta_dist = math.acos(cos_th_max)
                    theta_asymp = math.acos(-1.0 / e_mag)
                    theta_max = min(theta_asymp * 0.999, theta_dist)
                    delta_angle = 2.0 * theta_max
                    theta0 = -theta_max
                    t_curr = (theta0_curr + theta_max) / delta_angle if delta_angle > 0 else 0.0
                        
                orbit_buf[n_orbits, 0] = e_hat[0]
                orbit_buf[n_orbits, 1] = e_hat[1]
                orbit_buf[n_orbits, 2] = e_hat[2]
                orbit_buf[n_orbits, 3] = sl_p
                orbit_buf[n_orbits, 4] = q_hat[0]
                orbit_buf[n_orbits, 5] = q_hat[1]
                orbit_buf[n_orbits, 6] = q_hat[2]
                orbit_buf[n_orbits, 7] = e_mag
                orbit_buf[n_orbits, 8] = bary_rel[0]
                orbit_buf[n_orbits, 9] = bary_rel[1]
                orbit_buf[n_orbits, 10] = bary_rel[2]
                
                if e_mag < 0.999:
                    if not is_escape:
                        E0 = math.atan2(math.sqrt(1.0 - e_mag * e_mag) * math.sin(theta0), math.cos(theta0) + e_mag)
                    orbit_buf[n_orbits, 11] = math.cos(E0)
                    orbit_buf[n_orbits, 15] = math.sin(E0)
                else:
                    orbit_buf[n_orbits, 11] = math.cos(theta0)
                    orbit_buf[n_orbits, 15] = math.sin(theta0)
                    
                orbit_buf[n_orbits, 12] = visual_colors[i, 0]
                orbit_buf[n_orbits, 13] = visual_colors[i, 1]
                orbit_buf[n_orbits, 14] = visual_colors[i, 2]
                orbit_buf[n_orbits, 16] = t_curr
                orbit_buf[n_orbits, 17] = delta_angle
                orbit_buf[n_orbits, 18] = 0.0
                orbit_buf[n_orbits, 19] = lod_levels[i]
                n_orbits += 1
                
                if is_escape and gp_idx >= 0 and n_orbits < max_orbits:
                    gp_mu = G * (mass[gp_idx] + orb_m)
                    
                    c_exit_r = sl_p / (1.0 + e_mag * math.cos(theta_exit))
                    c_exit_x = c_exit_r * math.cos(theta_exit)
                    c_exit_y = c_exit_r * math.sin(theta_exit)
                    
                    v_fac = math.sqrt(mu / sl_p)
                    v_exit_x = -v_fac * math.sin(theta_exit)
                    v_exit_y = v_fac * (e_mag + math.cos(theta_exit))
                    
                    dt = 0.0
                    if e_mag >= 1.0:
                        a = sl_p / (e_mag * e_mag - 1.0)
                        if a > 0:
                            n = math.sqrt(mu / (a * a * a))
                            cosh_F0 = (e_mag + math.cos(theta0_curr)) / (1.0 + e_mag * math.cos(theta0_curr))
                            sinh_F0 = math.sqrt(e_mag * e_mag - 1.0) * math.sin(theta0_curr) / (1.0 + e_mag * math.cos(theta0_curr))
                            F0 = math.log(max(1e-10, cosh_F0 + sinh_F0))
                            M0 = e_mag * sinh_F0 - F0
                            
                            cosh_F_exit = (e_mag + math.cos(theta_exit)) / (1.0 + e_mag * math.cos(theta_exit))
                            sinh_F_exit = math.sqrt(e_mag * e_mag - 1.0) * math.sin(theta_exit) / (1.0 + e_mag * math.cos(theta_exit))
                            F_exit = math.log(max(1e-10, cosh_F_exit + sinh_F_exit))
                            M_exit = e_mag * sinh_F_exit - F_exit
                            dt = (M_exit - M0) / n
                    else:
                        a = sl_p / (1.0 - e_mag * e_mag)
                        if a > 0:
                            n = math.sqrt(mu / (a * a * a))
                            sin_E0 = math.sqrt(1.0 - e_mag * e_mag) * math.sin(theta0_curr) / (1.0 + e_mag * math.cos(theta0_curr))
                            cos_E0 = (e_mag + math.cos(theta0_curr)) / (1.0 + e_mag * math.cos(theta0_curr))
                            E0 = math.atan2(sin_E0, cos_E0)
                            M0 = E0 - e_mag * sin_E0
                            
                            sin_E_exit = math.sqrt(1.0 - e_mag * e_mag) * math.sin(theta_exit) / (1.0 + e_mag * math.cos(theta_exit))
                            cos_E_exit = (e_mag + math.cos(theta_exit)) / (1.0 + e_mag * math.cos(theta_exit))
                            E_exit = math.atan2(sin_E_exit, cos_E_exit)
                            M_exit = E_exit - e_mag * sin_E_exit
                            if M_exit < M0:
                                M_exit += 2.0 * math.pi
                            dt = (M_exit - M0) / n
                    
                    if dt < 0: dt = 0.0
                    
                    parent_mu = G * (mass[gp_idx] + eff_parent_m)
                    p_rel_x = eff_parent_pos[0] - pos[gp_idx, 0]
                    p_rel_y = eff_parent_pos[1] - pos[gp_idx, 1]
                    p_rel_z = eff_parent_pos[2] - pos[gp_idx, 2]
                    p_rel_vx = eff_parent_vel[0] - vel[gp_idx, 0]
                    p_rel_vy = eff_parent_vel[1] - vel[gp_idx, 1]
                    p_rel_vz = eff_parent_vel[2] - vel[gp_idx, 2]
                    
                    pa, pe, pinc_d, pOmega_d, pomega_d, pnu_d, pM_d, pperiod = compute_keplerian_elements(
                        p_rel_x, p_rel_y, p_rel_z, p_rel_vx, p_rel_vy, p_rel_vz, parent_mu)
                    
                    if dt > 0 and pa > 0 and pe < 1.0:
                        pn = math.sqrt(parent_mu / max(pa**3, 1e-30))
                        pM_new = (pM_d * math.pi / 180.0) + pn * dt
                        p_pos_new, p_vel_new = orbital_to_cartesian(
                            pa, pe, pinc_d * math.pi / 180.0, pOmega_d * math.pi / 180.0,
                            pomega_d * math.pi / 180.0, pM_new, parent_mu)
                    else:
                        p_pos_new = np.array([p_rel_x, p_rel_y, p_rel_z], dtype=np.float64)
                        p_vel_new = np.array([p_rel_vx, p_rel_vy, p_rel_vz], dtype=np.float64)
                    
                    for k in range(3):
                        c_exit_rel_pos = e_hat[k]*c_exit_x + q_hat[k]*c_exit_y
                        c_exit_rel_vel = e_hat[k]*v_exit_x + q_hat[k]*v_exit_y
                        
                        gp_rel_pos[k] = p_pos_new[k] + c_exit_rel_pos
                        gp_rel_vel[k] = p_vel_new[k] + c_exit_rel_vel
                        
                        abs_pos = p_pos_new[k] + pos[gp_idx, k] + c_exit_rel_pos
                        gp_bary[k] = (mass[gp_idx] * pos[gp_idx, k] + orb_m * abs_pos) / (mass[gp_idx] + orb_m)
                    
                    ge_hat, gq_hat, gbary_rel, gsl_p, ge_mag, gtheta0, gis_valid = compute_orbit_elements(
                        gp_rel_pos, gp_rel_vel, gp_mu, mass[gp_idx] / (mass[gp_idx] + orb_m), gp_bary, cam_origin,
                        False, False, gp_rel_pos)
                        
                    if gis_valid:
                        gtheta_start = gtheta0
                        
                        if ge_mag >= 1.0:
                            r_max = max(300.0, (gsl_p / (1.0 + ge_mag)) * 10.0)
                            cos_th_max = (gsl_p / r_max - 1.0) / ge_mag
                            cos_th_max = max(-1.0, min(1.0, cos_th_max))
                            theta_dist = math.acos(cos_th_max)
                            theta_asymp = math.acos(-1.0 / ge_mag)
                            theta_max = min(theta_asymp * 0.999, theta_dist)
                            if gtheta_start < theta_max:
                                gdelta = theta_max - gtheta_start
                            else:
                                gdelta = 0.0
                        else:
                            gdelta = 10.0
                            gE0 = math.atan2(math.sqrt(1.0 - ge_mag * ge_mag) * math.sin(gtheta_start), math.cos(gtheta_start) + ge_mag)
                            
                        if ge_mag < 0.999:
                            orbit_buf[n_orbits, 11] = math.cos(gE0)
                            orbit_buf[n_orbits, 15] = math.sin(gE0)
                        else:
                            orbit_buf[n_orbits, 11] = math.cos(gtheta_start)
                            orbit_buf[n_orbits, 15] = math.sin(gtheta_start)
                            
                        orbit_buf[n_orbits, 0] = ge_hat[0]
                        orbit_buf[n_orbits, 1] = ge_hat[1]
                        orbit_buf[n_orbits, 2] = ge_hat[2]
                        orbit_buf[n_orbits, 3] = gsl_p
                        orbit_buf[n_orbits, 4] = gq_hat[0]
                        orbit_buf[n_orbits, 5] = gq_hat[1]
                        orbit_buf[n_orbits, 6] = gq_hat[2]
                        orbit_buf[n_orbits, 7] = ge_mag
                        orbit_buf[n_orbits, 8] = gbary_rel[0]
                        orbit_buf[n_orbits, 9] = gbary_rel[1]
                        orbit_buf[n_orbits, 10] = gbary_rel[2]
                            
                        orbit_buf[n_orbits, 12] = visual_colors[i, 0] * 0.5
                        orbit_buf[n_orbits, 13] = visual_colors[i, 1] * 0.5
                        orbit_buf[n_orbits, 14] = visual_colors[i, 2] * 0.5
                        orbit_buf[n_orbits, 16] = 0.0
                        orbit_buf[n_orbits, 17] = gdelta
                        orbit_buf[n_orbits, 18] = 1.0
                        orbit_buf[n_orbits, 19] = lod_levels[i]
                        n_orbits += 1
                        
    for parent_idx in range(num_bodies):
        ci = best_child[parent_idx]
        if ci < 0:
            continue
        
        p_m = mass[parent_idx]
        c_m = mass[ci]
        total_m = p_m + c_m
        if total_m <= 0.0:
            continue
        mu = G * total_m
        
        for k in range(3):
            rel_pos[k] = pos[ci, k] - pos[parent_idx, k]
            rel_vel[k] = vel[ci, k] - vel[parent_idx, k]
            bary[k] = (p_m * pos[parent_idx, k] + c_m * pos[ci, k]) / total_m
            parent_bary_ref[k] = pos[parent_idx, k] - bary[k]
        
        e_hat, q_hat, bary_rel, sl_p, e_mag, theta0, is_valid = compute_orbit_elements(
            rel_pos, rel_vel, mu, c_m / total_m, bary, cam_origin,
            True, True, parent_bary_ref)
        
        theta0_curr = theta0
        if is_valid and n_orbits < max_orbits:
            bary_dist = fast_norm(bary_rel)
            if bary_dist > 1e-6 and sl_p / bary_dist < 1e-6:
                continue
                
            delta_angle = 10.0
            t_curr = 0.0
            if e_mag >= 1.0:
                r_max = max(300.0, (sl_p / (1.0 + e_mag)) * 10.0)
                cos_th_max = (sl_p / r_max - 1.0) / e_mag
                cos_th_max = max(-1.0, min(1.0, cos_th_max))
                theta_dist = math.acos(cos_th_max)
                theta_asymp = math.acos(-1.0 / e_mag)
                theta_max = min(theta_asymp * 0.999, theta_dist)
                delta_angle = 2.0 * theta_max
                theta0 = -theta_max
                t_curr = (theta0_curr + theta_max) / delta_angle if delta_angle > 0 else 0.0
            
            orbit_buf[n_orbits, 0] = e_hat[0]
            orbit_buf[n_orbits, 1] = e_hat[1]
            orbit_buf[n_orbits, 2] = e_hat[2]
            orbit_buf[n_orbits, 3] = sl_p
            orbit_buf[n_orbits, 4] = q_hat[0]
            orbit_buf[n_orbits, 5] = q_hat[1]
            orbit_buf[n_orbits, 6] = q_hat[2]
            orbit_buf[n_orbits, 7] = e_mag
            orbit_buf[n_orbits, 8] = bary_rel[0]
            orbit_buf[n_orbits, 9] = bary_rel[1]
            orbit_buf[n_orbits, 10] = bary_rel[2]
            
            if e_mag < 0.999:
                E0 = math.atan2(math.sqrt(1.0 - e_mag * e_mag) * math.sin(theta0), math.cos(theta0) + e_mag)
                orbit_buf[n_orbits, 11] = math.cos(E0)
                orbit_buf[n_orbits, 15] = math.sin(E0)
            else:
                orbit_buf[n_orbits, 11] = math.cos(theta0)
                orbit_buf[n_orbits, 15] = math.sin(theta0)
                
            orbit_buf[n_orbits, 12] = visual_colors[parent_idx, 0]
            orbit_buf[n_orbits, 13] = visual_colors[parent_idx, 1]
            orbit_buf[n_orbits, 14] = visual_colors[parent_idx, 2]
            orbit_buf[n_orbits, 16] = t_curr
            orbit_buf[n_orbits, 17] = delta_angle
            orbit_buf[n_orbits, 18] = 0.0
            orbit_buf[n_orbits, 19] = lod_levels[parent_idx]
            n_orbits += 1
            
    return n_orbits

@njit(cache=True)
def _update_hierarchy_core(positions, masses, current_parents, num_bodies):
    """Numba-jitted core: recompute parents based on Hill sphere containment.
    No O(N²) memory allocation — distances computed inline as needed."""
    new_parents = current_parents.copy()
    
    root_idx = 0
    max_mass = masses[0]
    for i in range(1, num_bodies):
        if masses[i] > max_mass:
            max_mass = masses[i]
            root_idx = i
    new_parents[root_idx] = -1
    
    r_hill_array = np.zeros(num_bodies, dtype=np.float64)
    for j in range(num_bodies):
        if j == root_idx or masses[j] <= 0.0:
            continue
        j_parent = current_parents[j] if current_parents[j] != -1 else root_idx
        if masses[j_parent] > 0.0:
            dx = positions[j, 0] - positions[j_parent, 0]
            dy = positions[j, 1] - positions[j_parent, 1]
            dz = positions[j, 2] - positions[j_parent, 2]
            dist_j_parent = math.sqrt(dx*dx + dy*dy + dz*dz)
            r_hill_array[j] = dist_j_parent * (masses[j] / (3.0 * masses[j_parent])) ** (1.0 / 3.0)
    
    for i in range(num_bodies):
        if i == root_idx:
            continue
        
        best_parent = root_idx
        best_hill = 1e30
        
        for j in range(num_bodies):
            if j == i or j == root_idx or masses[j] <= masses[i] or r_hill_array[j] == 0.0:
                continue
            
            r_hill = r_hill_array[j]
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            dist_ij = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            threshold = r_hill if j == current_parents[i] else r_hill * 0.9
            
            if dist_ij < threshold and r_hill < best_hill:
                best_parent = j
                best_hill = r_hill
        
        new_parents[i] = best_parent
    
    return new_parents

def update_hierarchy(sim, num_bodies, current_parents):
    """Recompute parent_indices based on Hill sphere containment.
    
    Uses hysteresis: enter at < 0.9 * r_Hill, exit at > 1.0 * r_Hill
    to prevent flickering at boundaries.
    """
    arr = _get_particle_array(sim, num_bodies)
    positions = np.ascontiguousarray(arr[:, 0:3])
    masses = arr[:, 9].copy()
    
    return _update_hierarchy_core(positions, masses, current_parents, num_bodies)

def build_tree_order(parent_indices, num_bodies, positions=None):
    """Return (tree_indices, tree_depths) numpy arrays in depth-first tree-traversal order."""
    children = {i: [] for i in range(num_bodies)}
    roots = []
    for i in range(num_bodies):
        pi = int(parent_indices[i])
        if pi == -1:
            roots.append(i)
        else:
            children[pi].append(i)
            
    if positions is not None:
        for p in children:
            if len(children[p]) > 1:
                children[p].sort(key=lambda c: np.linalg.norm(positions[c] - positions[p]))
    
    result = []
    visited = set()
    def dfs(node, depth):
        if node in visited:
            return
        visited.add(node)
        result.append((node, depth))
        for child in children[node]:
            dfs(child, depth + 1)
    
    for root in roots:
        dfs(root, 0)
    
    tree_indices = np.array([r[0] for r in result], dtype=np.int32)
    tree_depths = np.array([r[1] for r in result], dtype=np.int32)
    
    if len(tree_indices) < num_bodies:
        missing = set(range(num_bodies)) - set(tree_indices)
        for m in missing:
            parent_indices[m] = -1
        tree_indices = np.append(tree_indices, list(missing))
        tree_depths = np.append(tree_depths, [0] * len(missing))
        
    return tree_indices, tree_depths

def physics_loop(sim, num_bodies, shared_state, time_ctrl, running):
    _timer_set = False
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        _timer_set = True
    except (AttributeError, OSError):
        pass

    last_time = time.time()
    
    with shared_state["lock"]:
        current_parents = shared_state["parent_indices"].copy()
    
    current_parents = update_hierarchy(sim, num_bodies, current_parents)
    positions = np.array([[p.x, p.y, p.z] for p in sim.particles[:num_bodies]])
    tree_indices, tree_depths = build_tree_order(current_parents, num_bodies, positions)
    
    with shared_state["lock"]:
        shared_state["parent_indices"][:] = current_parents
        shared_state["tree_indices"][:] = tree_indices
        shared_state["tree_depths"][:] = tree_depths
        shared_state["hierarchy_version"] += 1
        
    frame_count = 0
    while running[0]:
        now = time.time()
        dt_real = min(now - last_time, 0.1)
        last_time = now
        # ── System Switch Handling ──
        switch_req = None
        with shared_state["lock"]:
            switch_req = shared_state.get("system_switch_request")
            if switch_req is not None:
                shared_state["system_switch_request"] = None

        if switch_req is not None:
            # Save current system as snapshot
            snapshot = SystemSnapshot()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                snapshot.sim_copy = sim.copy()
            snapshot.num_bodies = num_bodies
            snapshot.bodies_data = switch_req.get("old_bodies_data", [])
            snapshot.visual_data = switch_req.get("old_visual_data", [])
            snapshot.atmo_bodies = switch_req.get("old_atmo_bodies", [])
            snapshot.ring_bodies = switch_req.get("old_ring_bodies", [])
            snapshot.oblate_physics_list = list(shared_state.get("oblate_physics_list", []))
            snapshot.has_j2 = shared_state.get("has_j2", False)
            snapshot.has_gr = shared_state.get("has_gr", False)
            snapshot.phys_star_idx = shared_state.get("phys_star_idx", -1)
            snapshot.star_idx = switch_req.get("old_star_idx", 0)
            snapshot.parent_indices = shared_state["parent_indices"].copy()
            snapshot.sim_time = sim.t
            snapshot.time_multiplier = time_ctrl["multiplier"]
            snapshot.was_paused = time_ctrl["paused"]

            # Build new system
            new_bundle = switch_req.get("new_bundle")
            if new_bundle is None:
                # Load from snapshot
                restore_snap = switch_req.get("restore_snapshot")
                if restore_snap and restore_snap.sim_copy is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sim = restore_snap.sim_copy.copy()
                    attach_reboundx_forces(sim, restore_snap.has_j2, restore_snap.has_gr, restore_snap.phys_star_idx, restore_snap.oblate_physics_list)
                    if restore_snap.has_j2 and len(restore_snap.oblate_physics_list) > 0:
                        print(f"[System] Restored J2 precession for {len(restore_snap.oblate_physics_list)} oblate bodies")
                    if restore_snap.has_gr and restore_snap.phys_star_idx >= 0:
                        print(f"[System] Restored 1PN GR precession (star index {restore_snap.phys_star_idx})")
                    num_bodies = restore_snap.num_bodies
                    new_bundle_info = {
                        "bodies_data": restore_snap.bodies_data,
                        "visual_data": restore_snap.visual_data,
                        "atmo_bodies": restore_snap.atmo_bodies,
                        "ring_bodies": restore_snap.ring_bodies,
                        "oblate_physics_list": restore_snap.oblate_physics_list,
                        "has_j2": restore_snap.has_j2,
                        "has_gr": restore_snap.has_gr,
                        "phys_star_idx": restore_snap.phys_star_idx,
                        "star_idx": restore_snap.star_idx,
                        "num_bodies": restore_snap.num_bodies,
                    }
                    time_ctrl["multiplier"] = restore_snap.time_multiplier
                    time_ctrl["paused"] = restore_snap.was_paused
                else:
                    # Load from disk via load_system_from_data
                    new_bundle = load_system_from_data(switch_req["bodies_data_raw"])
            
            if new_bundle is not None:
                sim = new_bundle["sim"]
                num_bodies = new_bundle["num_bodies"]
                new_bundle_info = {
                    "bodies_data": new_bundle["bodies_data"],
                    "visual_data": new_bundle["visual_data"],
                    "atmo_bodies": new_bundle["atmo_bodies"],
                    "ring_bodies": new_bundle["ring_bodies"],
                    "oblate_physics_list": new_bundle["oblate_physics_list"],
                    "has_j2": new_bundle["has_j2"],
                    "has_gr": new_bundle["has_gr"],
                    "phys_star_idx": new_bundle["phys_star_idx"],
                    "star_idx": new_bundle["star_idx"],
                    "num_bodies": new_bundle["num_bodies"],
                }
                
                if switch_req.get("preserve_state"):
                    sim.t = switch_req["preserve_t"]
                    time_ctrl["paused"] = switch_req["preserve_paused"]
                    time_ctrl["multiplier"] = switch_req["preserve_speed"]
                else:
                    time_ctrl["paused"] = True
                    time_ctrl["multiplier"] = 1.0

            # Update shared state arrays for new system
            opl = new_bundle_info.get("oblate_physics_list", [])
            with shared_state["lock"]:
                shared_state["has_j2"] = new_bundle_info["has_j2"]
                shared_state["has_gr"] = new_bundle_info["has_gr"]
                shared_state["phys_star_idx"] = new_bundle_info["phys_star_idx"]
                shared_state["oblate_physics_list"] = opl
                shared_state["t"] = sim.t
                shared_state["pos"] = np.zeros((num_bodies, 3), dtype='f8')
                shared_state["vel"] = np.zeros((num_bodies, 3), dtype='f8')
                shared_state["mass"] = np.zeros(num_bodies, dtype='f8')
                shared_state["parent_indices"] = np.full(num_bodies, -1, dtype=np.int32)
                shared_state["tree_indices"] = np.zeros(num_bodies, dtype=np.int32)
                shared_state["tree_depths"] = np.zeros(num_bodies, dtype=np.int32)
                shared_state["timeline_active"] = False
                shared_state["timeline_progress"] = 0.0
                shared_state["timeline_pos"] = np.zeros((0, num_bodies, 3), dtype='f4')
                shared_state["timeline_vel"] = np.zeros((0, num_bodies, 3), dtype='f4')
                shared_state["timeline_times"] = np.zeros(0, dtype='f8')
                shared_state["crud_queue"] = []
                shared_state["crud_completed"] = []
                shared_state["rebuild_flag"] = False

                if new_bundle_info["has_j2"] and opl:
                    shared_state["oblate_indices"] = np.array([x[0] for x in opl], dtype=np.int32)
                    shared_state["oblate_j2"] = np.array([x[1] for x in opl], dtype=np.float64)
                    shared_state["oblate_req"] = np.array([x[3] for x in opl], dtype=np.float64)
                    shared_state["oblate_poles"] = np.array([x[4] for x in opl], dtype=np.float64)
                    shared_state["oblate_masses"] = np.array([x[5] for x in opl], dtype=np.float64)
                else:
                    shared_state["oblate_indices"] = None
                    shared_state["oblate_j2"] = None
                    shared_state["oblate_req"] = None
                    shared_state["oblate_poles"] = None
                    shared_state["oblate_masses"] = None

            # Extract initial state
            local_pos = np.empty((num_bodies, 3), dtype=np.float64)
            local_vel = np.empty((num_bodies, 3), dtype=np.float64)
            _extract_render_state(sim, num_bodies, local_pos, local_vel)
            _init_arr_sw = _get_particle_array(sim, num_bodies)

            current_parents = new_bundle_info.get("parent_indices",
                np.full(num_bodies, -1, dtype=np.int32)).copy() if "parent_indices" in new_bundle_info else np.full(num_bodies, -1, dtype=np.int32)
            # Rebuild parent indices from bodies_data if not from snapshot
            bd = new_bundle_info["bodies_data"]
            nti = {b['name']: i for i, b in enumerate(bd)}
            for i, body in enumerate(bd):
                if 'parentId' in body and body['parentId'] in nti:
                    current_parents[i] = nti[body['parentId']]
            current_parents = update_hierarchy(sim, num_bodies, current_parents)
            positions = np.array([[p.x, p.y, p.z] for p in sim.particles[:num_bodies]])
            tree_indices, tree_depths = build_tree_order(current_parents, num_bodies, positions)

            with shared_state["lock"]:
                np.copyto(shared_state["pos"], local_pos)
                np.copyto(shared_state["vel"], local_vel)
                shared_state["mass"][:] = _init_arr_sw[:, 9]
                shared_state["parent_indices"][:] = current_parents
                shared_state["tree_indices"][:] = tree_indices
                shared_state["tree_depths"][:] = tree_depths
                shared_state["hierarchy_version"] += 1

                # Signal render thread
                shared_state["system_snapshot_out"] = snapshot
                shared_state["system_new_bundle"] = new_bundle_info
                shared_state["system_switch_complete"] = True
                
                if "ephemeris_enter" in switch_req:
                    shared_state["ephemeris_enter"] = switch_req["ephemeris_enter"]
                if "ephemeris_lines" in switch_req:
                    shared_state["ephemeris_orbit_lines"] = switch_req["ephemeris_lines"]
                if "ephemeris_exit" in switch_req:
                    shared_state["ephemeris_exit"] = switch_req["ephemeris_exit"]
                
            frame_count = 0
            last_time = time.time()
            continue

        crud_ops = []
        with shared_state["lock"]:
            if shared_state["crud_queue"]:
                crud_ops = shared_state["crud_queue"][:]
                shared_state["crud_queue"].clear()

        
        if crud_ops:
            for op in crud_ops:
                if op["action"] == "UPDATE":
                    idx = op["idx"]
                    p = sim.particles[idx]
                    if "mass" in op:
                        p.m = op["mass"]
                    if "pos" in op and "vel" in op:
                        dx = op["pos"][0] - p.x
                        dy = op["pos"][1] - p.y
                        dz = op["pos"][2] - p.z
                        
                        dvx = op["vel"][0] - p.vx
                        dvy = op["vel"][1] - p.vy
                        dvz = op["vel"][2] - p.vz
                        
                        p.x, p.y, p.z = op["pos"]
                        p.vx, p.vy, p.vz = op["vel"]
                        
                        parents = shared_state["parent_indices"]
                        children_to_update = []
                        for i in range(num_bodies):
                            curr = parents[i]
                            while curr != -1 and curr != idx:
                                curr = parents[curr]
                            if curr == idx and i != idx:
                                children_to_update.append(i)
                                
                        if children_to_update:
                            with shared_state["lock"]:
                                for i in children_to_update:
                                    dp = sim.particles[i]
                                    dp.x += dx; dp.y += dy; dp.z += dz
                                    dp.vx += dvx; dp.vy += dvy; dp.vz += dvz
                                    shared_state["pos"][i] = (dp.x, dp.y, dp.z)
                                    shared_state["vel"][i] = (dp.vx, dp.vy, dp.vz)
                        
                    with shared_state["lock"]:
                        if "mass" in op:
                            shared_state["mass"][idx] = op["mass"]
                            shared_state["rebuild_flag"] = True
                        if "pos" in op:
                            shared_state["pos"][idx] = [op["pos"][0], op["pos"][2], -op["pos"][1]]
                        if "vel" in op:
                            shared_state["vel"][idx] = [op["vel"][0], op["vel"][2], -op["vel"][1]]
                        if "radius" in op:
                            r_au = op["radius"] / 1.496e8
                            if shared_state.get("oblate_indices") is not None:
                                mask = shared_state["oblate_indices"] == idx
                                if np.any(mask):
                                    shared_state["oblate_req"][mask] = r_au
                                for i, obl in enumerate(shared_state["oblate_physics_list"]):
                                    if obl[0] == idx:
                                        shared_state["oblate_physics_list"][i] = (obl[0], obl[1], obl[2], r_au, obl[4], obl[5], obl[6], obl[7])
                        shared_state["crud_completed"].append(op)
                            
                elif op["action"] == "CREATE":
                    mass = op["mass"]
                    pos = op["pos"]
                    vel = op["vel"]
                    parent_idx = op["parent_idx"]
                    
                    sim.add(m=mass, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])
                    num_bodies += 1
                    
                    with shared_state["lock"]:
                        shared_state["timeline_active"] = False
                        pos_ogl = [pos[0], pos[2], -pos[1]]
                        vel_ogl = [vel[0], vel[2], -vel[1]]
                        shared_state["pos"] = np.vstack([shared_state["pos"], pos_ogl])
                        shared_state["vel"] = np.vstack([shared_state["vel"], vel_ogl])
                        shared_state["mass"] = np.append(shared_state["mass"], mass)
                        shared_state["parent_indices"] = np.append(shared_state["parent_indices"], np.int32(parent_idx))
                        op["idx"] = num_bodies - 1
                        shared_state["crud_completed"].append(op)
                            
                elif op["action"] == "DELETE":
                    idx = op["idx"]
                    sim.remove(index=idx)
                    num_bodies -= 1
                    
                    with shared_state["lock"]:
                        shared_state["timeline_active"] = False
                        shared_state["pos"] = np.delete(shared_state["pos"], idx, axis=0)
                        shared_state["vel"] = np.delete(shared_state["vel"], idx, axis=0)
                        shared_state["mass"] = np.delete(shared_state["mass"], idx)
                        
                        pi = np.delete(shared_state["parent_indices"], idx)
                        pi[pi == idx] = 0
                        pi[pi > idx] -= 1
                        shared_state["parent_indices"] = pi
                        
                        if shared_state.get("oblate_indices") is not None:
                            j2_mask = shared_state["oblate_indices"] != idx
                            shared_state["oblate_indices"] = shared_state["oblate_indices"][j2_mask]
                            shared_state["oblate_j2"] = shared_state["oblate_j2"][j2_mask]
                            shared_state["oblate_req"] = shared_state["oblate_req"][j2_mask]
                            shared_state["oblate_poles"] = shared_state["oblate_poles"][j2_mask]
                            shared_state["oblate_masses"] = shared_state["oblate_masses"][j2_mask]
                            shared_state["oblate_indices"][shared_state["oblate_indices"] > idx] -= 1
                            
                            new_opl = []
                            for obl in shared_state["oblate_physics_list"]:
                                new_idx = obl[0]
                                if new_idx == idx: continue
                                if new_idx > idx: new_idx -= 1
                                new_opl.append((new_idx, obl[1], obl[2], obl[3], obl[4], obl[5], obl[6], obl[7]))
                            shared_state["oblate_physics_list"] = new_opl
                            shared_state["has_j2"] = len(new_opl) > 0
                            
                        if shared_state["phys_star_idx"] == idx:
                            shared_state["phys_star_idx"] = -1
                        elif shared_state["phys_star_idx"] > idx:
                            shared_state["phys_star_idx"] -= 1
                        
                        shared_state["crud_completed"].append(op)
            
            with shared_state["lock"]:
                current_parents = shared_state["parent_indices"].copy()
            current_parents = update_hierarchy(sim, num_bodies, current_parents)
            positions = np.array([[p.x, p.y, p.z] for p in sim.particles[:num_bodies]])
            tree_indices, tree_depths = build_tree_order(current_parents, num_bodies, positions)
            
            with shared_state["lock"]:
                shared_state["parent_indices"][:] = current_parents
                shared_state["tree_indices"] = tree_indices
                shared_state["tree_depths"] = tree_depths
                shared_state["hierarchy_version"] += 1
                shared_state["rebuild_flag"] = True

        if time_ctrl.get("sync_t") is not None:
            sync_time = time_ctrl["sync_t"]
            sync_idx = time_ctrl.get("sync_idx")
            time_ctrl["sync_t"] = None
            time_ctrl["sync_idx"] = None
            
            if sync_idx is not None and "timeline_raw" in shared_state:
                raw_arr = shared_state["timeline_raw"][sync_idx]
                sim.t = sync_time
                for i in range(num_bodies):
                    sim.particles[i].x = raw_arr[i, 0]
                    sim.particles[i].y = raw_arr[i, 1]
                    sim.particles[i].z = raw_arr[i, 2]
                    sim.particles[i].vx = raw_arr[i, 3]
                    sim.particles[i].vy = raw_arr[i, 4]
                    sim.particles[i].vz = raw_arr[i, 5]
            else:
                sim.integrate(sync_time)
                
            local_pos = np.empty((num_bodies, 3), dtype=np.float64)
            local_vel = np.empty((num_bodies, 3), dtype=np.float64)
            _extract_render_state(sim, num_bodies, local_pos, local_vel)
            sim_t = sim.t
            with shared_state["lock"]:
                np.copyto(shared_state["pos"], local_pos)
                np.copyto(shared_state["vel"], local_vel)
                shared_state["t"] = sim_t
                shared_state["timeline_active"] = False
            continue

        if time_ctrl.get("render_timeline", False):
            target_t = time_ctrl["target_t"]
            start_t = sim.t
            
            diff_t = target_t - start_t
            if abs(diff_t) < 0.001:
                time_ctrl["render_timeline"] = False
                continue
                
            num_steps = 1000
            step_dt = diff_t / num_steps
            
            tl_pos = np.zeros((num_steps, num_bodies, 3), dtype='f4')
            tl_vel = np.zeros((num_steps, num_bodies, 3), dtype='f4')
            tl_raw_arr = np.zeros((num_steps, num_bodies, 6), dtype=np.float64)
            tl_times = np.zeros(num_steps, dtype='f8')
            
            with shared_state["lock"]:
                shared_state["timeline_active"] = True
                shared_state["timeline_progress"] = 0.0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim_timeline_base = sim.copy()
                sim_start_copy = sim.copy()
            attach_reboundx_forces(sim_start_copy, shared_state["has_j2"], shared_state["has_gr"], shared_state["phys_star_idx"], shared_state["oblate_physics_list"])
                
            render_start_time = time.time()
            valid_steps = 0
            for step in range(num_steps):
                if not running[0] or time_ctrl.get("cancel_render", False):
                    break
                sim_start_copy.integrate(start_t + (step + 1) * step_dt)
                
                _tl_arr = _get_particle_array(sim_start_copy, num_bodies)
                tl_pos[step, :, 0] = _tl_arr[:, 0]
                tl_pos[step, :, 1] = _tl_arr[:, 2]
                tl_pos[step, :, 2] = -_tl_arr[:, 1]
                tl_vel[step, :, 0] = _tl_arr[:, 3]
                tl_vel[step, :, 1] = _tl_arr[:, 5]
                tl_vel[step, :, 2] = -_tl_arr[:, 4]
                tl_times[step] = sim_start_copy.t
                
                tl_raw_arr[step] = _tl_arr[:, 0:6]
                
                if step % 20 == 0:
                    current_real_time = time.time()
                    elapsed = current_real_time - render_start_time
                    rate = (sim_start_copy.t - start_t) * SECONDS_PER_YEAR / max(elapsed, 0.001)
                    with shared_state["lock"]:
                        shared_state["timeline_progress"] = float(step) / num_steps
                        shared_state["timeline_rate"] = rate
                valid_steps = step + 1
            
            if valid_steps > 0:
                local_pos = np.empty((num_bodies, 3), dtype=np.float64)
                local_vel = np.empty((num_bodies, 3), dtype=np.float64)
                _extract_render_state(sim_start_copy, num_bodies, local_pos, local_vel)
                sim_t = sim_start_copy.t
                with shared_state["lock"]:
                    shared_state["timeline_pos"] = tl_pos[:valid_steps]
                    shared_state["timeline_vel"] = tl_vel[:valid_steps]
                    shared_state["timeline_raw"] = tl_raw_arr[:valid_steps]
                    shared_state["timeline_times"] = tl_times[:valid_steps]
                    shared_state["timeline_progress"] = 1.0
                    np.copyto(shared_state["pos"], local_pos)
                    np.copyto(shared_state["vel"], local_vel)
                    shared_state["t"] = sim_t
                time_ctrl["snap_to_end"] = True
            else:
                with shared_state["lock"]:
                    shared_state["timeline_active"] = False
            
            time_ctrl["render_timeline"] = False
            time_ctrl["cancel_render"] = False
            time_ctrl["paused"] = True
            continue
        if not time_ctrl["paused"]:
            dt_sim = dt_real * time_ctrl["multiplier"] / SECONDS_PER_YEAR
            
            if shared_state.get("ephemeris_mode", False):
                sim.t += dt_sim
            else:
                sim.integrate(sim.t + dt_sim)
            
            frame_count += 1
            if frame_count % 30 == 0:
                new_parents = update_hierarchy(sim, num_bodies, current_parents)
                positions = np.array([[p.x, p.y, p.z] for p in sim.particles[:num_bodies]])
                new_tree_indices, new_tree_depths = build_tree_order(new_parents, num_bodies, positions)
                
                with shared_state["lock"]:
                    changed = False
                    if not np.array_equal(new_parents, shared_state["parent_indices"]):
                        changed = True
                    elif not np.array_equal(new_tree_indices, shared_state["tree_indices"]):
                        changed = True
                        
                    if changed:
                        current_parents = new_parents
                        shared_state["parent_indices"][:] = new_parents
                        shared_state["tree_indices"] = new_tree_indices
                        shared_state["tree_depths"] = new_tree_depths
                        shared_state["hierarchy_version"] += 1
                        shared_state["rebuild_flag"] = True
            
            local_pos = np.empty((num_bodies, 3), dtype=np.float64)
            local_vel = np.empty((num_bodies, 3), dtype=np.float64)
            _extract_render_state(sim, num_bodies, local_pos, local_vel)
            sim_t = sim.t
            with shared_state["lock"]:
                np.copyto(shared_state["pos"], local_pos)
                np.copyto(shared_state["vel"], local_vel)
                shared_state["t"] = sim_t
                shared_state["parent_indices"][:] = current_parents
                shared_state["tree_indices"][:] = tree_indices
                shared_state["tree_depths"][:] = tree_depths
                if frame_count % 30 == 0:
                    shared_state["hierarchy_version"] += 1
                
        elapsed = time.time() - now
        sleep_time = max(0.001, 0.016 - elapsed)
        time.sleep(sleep_time)

    if _timer_set:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except (AttributeError, OSError):
            pass

def load_system_from_data(bodies_data_raw):
    """Build a REBOUND simulation and all associated state from raw bodies data.
    
    This is the core system loading function, called both at startup and when
    switching to a different star system.
    
    Args:
        bodies_data_raw: list of body dicts loaded from a system.json file
    
    Returns:
        dict with sim, num_bodies, bodies_data (sorted), visual_data,
        atmo_bodies, ring_bodies, oblate_physics_list, parent_indices,
        name_to_idx, has_j2, has_gr, phys_star_idx, star_idx, and more.
    """
    G_const = 39.476926421373 # G for AU, Msun, Julian Year (365.25 days)

    # ── Topological sort (parents before children) ──
    name_to_idx = {b['name']: i for i, b in enumerate(bodies_data_raw)}
    sorted_order = []
    _visited = set()
    def _topo(idx):
        if idx in _visited: return
        _visited.add(idx)
        body = bodies_data_raw[idx]
        if 'parentId' in body:
            p = name_to_idx.get(body['parentId'])
            if p is not None: _topo(p)
        sorted_order.append(idx)
    for i in range(len(bodies_data_raw)):
        _topo(i)
    bodies_data = [bodies_data_raw[i] for i in sorted_order]
    name_to_idx = {b['name']: i for i, b in enumerate(bodies_data)}

    # ── Create REBOUND simulation ──
    sim = rebound.Simulation()
    sim.softening = 1e-6
    sim.integrator = "ias15"
    sim.G = G_const

    # ── Process each body ──
    visual_data = []
    pole_dirs = {}
    ring_bodies = []
    oblate_physics_list = []

    for body in bodies_data:
        name = body['name']
        idx = name_to_idx[name]
        mass = body.get('m', 0.0)
        color = hex_to_rgb(body.get('color', '#ffffff'))
        radius_au = body.get('r', 1.0) * 0.00465

        obj_type = body.get('type', 'Unknown')
        if obj_type == "Star": min_px = 3.0
        elif obj_type == "Moon": min_px = 1.0
        else: min_px = 2.0

        pole_ecl = np.array([0.0, 0.0, 1.0], dtype='f8')
        pole_render = np.array([0.0, 1.0, 0.0], dtype='f8')
        if 'pole_ra' in body and 'pole_dec' in body:
            pole_ecl = np.array(pole_to_ecliptic(body['pole_ra'], body['pole_dec']), dtype='f8')
            pole_render = np.array([pole_ecl[0], pole_ecl[2], -pole_ecl[1]], dtype='f8')
            pole_dirs[idx] = pole_render
        obl = body.get('oblateness', 0.0)
        visual_data.append([color[0], color[1], color[2], radius_au, min_px,
                            pole_render[0], pole_render[1], pole_render[2], obl])

        j2 = body.get('J2', 0.0)
        if j2 > 0.0:
            j4 = body.get('j4', 0.0)
            r_mean = body.get('r', 1.0)
            f = body.get('oblateness', 0.0)
            req_solar = r_mean / ((1.0 - f) ** (1.0 / 3.0)) if f > 0 else r_mean
            req_au = req_solar * 0.00465
            rot_period = body.get('rotation_period', 0.0)
            oblate_physics_list.append((idx, j2, j4, req_au, pole_ecl, mass, name, rot_period))

        if 'rings' in body:
            parent_pole_ecl = pole_to_ecliptic(body.get('pole_ra', 0), body.get('pole_dec', 90))
            pole_r = np.array([parent_pole_ecl[0], parent_pole_ecl[2], -parent_pole_ecl[1]])
            ring_bodies.append((idx, body['rings'], pole_r, radius_au))

        # ── Add particle to simulation ──
        if 'parentId' in body:
            parent_name = body['parentId']
            if 'sv' in body:
                sv = body['sv']
                try:
                    primary = sim.particles[parent_name]
                except rebound.ParticleNotFound:
                    available = [p.hash.value for p in sim.particles if hasattr(p, 'hash')]
                    print(f"[Error] Missing parent '{parent_name}' for body '{name}'. Available: {available}")
                    raise
                sim.add(m=mass,
                        x=primary.x + sv['x'], y=primary.y + sv['y'], z=primary.z + sv['z'],
                        vx=primary.vx + sv['vx'], vy=primary.vy + sv['vy'], vz=primary.vz + sv['vz'],
                        hash=name)
            elif body.get('orbitRef') == 'equatorial':
                parent_body_data = next((b for b in bodies_data if b['name'] == parent_name), None)
                if parent_body_data and 'pole_ra' in parent_body_data:
                    pole_ecl_p = pole_to_ecliptic(parent_body_data['pole_ra'], parent_body_data['pole_dec'])
                    R = build_equatorial_frame(pole_ecl_p)
                    primary = sim.particles[parent_name]
                    mu = G_const * (mass + primary.m)
                    pos_eq, vel_eq = orbital_to_cartesian(
                        body.get('a', 0.0), body.get('e', 0.0),
                        math.radians(body.get('inc', 0.0)),
                        math.radians(body.get('Omega', 0.0)),
                        math.radians(body.get('omega', 0.0)),
                        math.radians(body.get('M', 0.0)), mu)
                    pos_ecl_v = R @ pos_eq
                    vel_ecl_v = R @ vel_eq
                    sim.add(m=mass,
                            x=primary.x + pos_ecl_v[0], y=primary.y + pos_ecl_v[1], z=primary.z + pos_ecl_v[2],
                            vx=primary.vx + vel_ecl_v[0], vy=primary.vy + vel_ecl_v[1], vz=primary.vz + vel_ecl_v[2],
                            hash=name)
                else:
                    sim.add(m=mass, primary=sim.particles[parent_name],
                            a=body.get('a', 0.0), e=body.get('e', 0.0),
                            inc=math.radians(body.get('inc', 0.0)), Omega=math.radians(body.get('Omega', 0.0)),
                            omega=math.radians(body.get('omega', 0.0)), M=math.radians(body.get('M', 0.0)),
                            hash=name)
            else:
                sim.add(m=mass, primary=sim.particles[parent_name],
                        a=body.get('a', 0.0), e=body.get('e', 0.0),
                        inc=math.radians(body.get('inc', 0.0)), Omega=math.radians(body.get('Omega', 0.0)),
                        omega=math.radians(body.get('omega', 0.0)), M=math.radians(body.get('M', 0.0)),
                        hash=name)
        elif 'sv' in body:
            sv = body['sv']
            sim.add(m=mass, x=sv['x'], y=sv['y'], z=sv['z'],
                    vx=sv['vx'], vy=sv['vy'], vz=sv['vz'], hash=name)
        else:
            sim.add(m=mass, hash=name)

    sim.move_to_com()

    # ── Atmospheres ──
    atmo_bodies = []
    SOLAR_RADIUS_KM = 696340.0
    AU_TO_KM = 149597870.7
    for i, body in enumerate(bodies_data):
        if 'atmosphere' in body:
            atmo = body['atmosphere']
            r_solar = body.get('r', 1.0)
            planet_radius_km = r_solar * SOLAR_RADIUS_KM
            radius_au = r_solar * 0.00465
            atmo_height_km = atmo['height']
            atmo_radius_km = planet_radius_km + atmo_height_km
            atmo_radius_au = atmo_radius_km / AU_TO_KM
            atmo_bodies.append({
                'body_idx': i,
                'planet_radius_km': planet_radius_km,
                'atmo_radius_km': atmo_radius_km,
                'atmo_radius_au': atmo_radius_au,
                'surface_radius_au': radius_au,
                'beta_rayleigh': np.array(atmo['rayleighCoefficients'], dtype='f4'),
                'h_rayleigh': float(atmo['rayleighScaleHeight']),
                'beta_mie': float(atmo['mieCoefficient']),
                'h_mie': float(atmo['mieScaleHeight']),
                'mie_g': float(atmo['mieAsymmetry']),
                'beta_absorption': np.array(atmo.get('absorptionCoefficients', [0,0,0]), dtype='f4'),
                'intensity': float(atmo['intensity']),
            })

    # ── J2 / GR setup ──
    has_j2 = bool(oblate_physics_list)
    phys_star_idx = -1
    star_idx = 0
    for i, body in enumerate(bodies_data):
        if body.get('type') == 'Star':
            phys_star_idx = i
            star_idx = i
            break
    has_gr = phys_star_idx >= 0

    rebx_obj = None
    if has_j2 or has_gr:
        attach_reboundx_forces(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list)
        if has_j2:
            print(f"[System] Attached J2 precession for {len(oblate_physics_list)} oblate bodies via reboundx")
        if has_gr:
            print(f"[System] Attached 1PN GR precession (star index {phys_star_idx}) via reboundx")
        sim.force_is_velocity_dependent = 1

    num_bodies = len(sim.particles)

    # ── Parent indices ──
    parent_indices = np.full(num_bodies, -1, dtype=np.int32)
    for i, body in enumerate(bodies_data):
        if 'parentId' in body:
            parent_indices[i] = name_to_idx[body['parentId']]

    print(f"[System] Loaded {num_bodies} bodies")

    return {
        "sim": sim,
        "rebx": rebx_obj,
        "num_bodies": num_bodies,
        "bodies_data": bodies_data,
        "name_to_idx": name_to_idx,
        "visual_data": visual_data,
        "parent_indices": parent_indices,
        "pole_dirs": pole_dirs,
        "ring_bodies": ring_bodies,
        "oblate_physics_list": oblate_physics_list,
        "atmo_bodies": atmo_bodies,
        "has_j2": has_j2,
        "has_gr": has_gr,
        "phys_star_idx": phys_star_idx,
        "star_idx": star_idx,
    }

