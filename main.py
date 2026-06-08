import OpenGL
OpenGL.ERROR_CHECKING = False
import glfw
import moderngl
import numpy as np
import ctypes
import rebound
import json
import math
import threading
import time
from pyrr import matrix44
import imgui
from imgui.integrations.glfw import GlfwRenderer
import warnings
from numba import njit
import datetime

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
C_AU_YR = 63197.79
FOV_DEG = 45.0

import os as _os
_PERF_ENABLED = _os.environ.get("STELLAR_FORGE_PERF") == "1"
_PERF_TRACKER = None

class _NullCtx:
    __slots__ = ()
    def __enter__(self): return self
    def __exit__(self, *a): return False

if _PERF_ENABLED:
    class _PerfCtx:
        __slots__ = ('_name',)
        def __init__(self, name):
            self._name = name
            _PERF_TRACKER.begin(name)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            _PERF_TRACKER.end()
            return False

    def _perf(name):
        if _PERF_TRACKER is None:
            return _NullCtx()
        return _PerfCtx(name)
else:
    def _perf(name):
        return _NullCtx()

def _PERF_INSTALL_TRACKER(tracker):
    """Public hook so perf_test.py can attach a tracker after import."""
    global _PERF_TRACKER
    _PERF_TRACKER = tracker

@njit(cache=True)
def fast_cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ], dtype=np.float64)

@njit(cache=True)
def fast_norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

# ---------- Zero-copy REBOUND particle extraction via ctypes ----------
_PARTICLE_STRIDE = None  # lazily initialized

def _get_particle_array(sim, num_bodies):
    """Get a NumPy view of REBOUND's internal C particle array (zero-copy).

    Returns array of shape (num_bodies, 16) where columns are:
    [0]=x [1]=y [2]=z [3]=vx [4]=vy [5]=vz [6..8]=ax,ay,az [9]=m ...
    """
    global _PARTICLE_STRIDE
    if _PARTICLE_STRIDE is None:
        _PARTICLE_STRIDE = ctypes.sizeof(rebound.Particle) // 8  # doubles per particle
    n = _PARTICLE_STRIDE
    addr = ctypes.addressof(sim._particles.contents)
    raw = (ctypes.c_double * (num_bodies * n)).from_address(addr)
    return np.frombuffer(raw, dtype=np.float64).reshape(num_bodies, n)

def _extract_render_state(sim, num_bodies, out_pos, out_vel):
    """Extract particle positions/velocities with render coordinate swap (x, z, -y)."""
    arr = _get_particle_array(sim, num_bodies)
    out_pos[:, 0] = arr[:, 0]    # x
    out_pos[:, 1] = arr[:, 2]    # z
    out_pos[:, 2] = -arr[:, 1]   # -y
    out_vel[:, 0] = arr[:, 3]    # vx
    out_vel[:, 1] = arr[:, 5]    # vz
    out_vel[:, 2] = -arr[:, 4]   # -vy

def setup_reboundx(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list):
    import reboundx
    rebx = reboundx.Extras(sim)
    if has_gr and phys_star_idx >= 0:
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = C_AU_YR
        sim.particles[phys_star_idx].params["gr_source"] = 1
        
    if has_j2 and len(oblate_physics_list) > 0:
        j2_force = rebx.load_force("gravitational_harmonics")
        rebx.add_force(j2_force)
        
        for i, j2, j4, req, pole, _, name in oblate_physics_list:
            p = sim.particles[i]
            p.params["J2"] = j2
            if j4 != 0.0:
                p.params["J4"] = j4
            p.params["R_eq"] = req
                
            omega_mag = 1.0 # default 1 rad/yr
            if name == "Earth":
                omega_mag = 365.25 * 2 * np.pi
            elif name == "Jupiter":
                omega_mag = (365.25 * 24.0 / 9.925) * 2 * np.pi
            elif name == "Saturn":
                omega_mag = (365.25 * 24.0 / 10.656) * 2 * np.pi
                
            p.params["Omega"] = [pole[0]*omega_mag, pole[1]*omega_mag, pole[2]*omega_mag]
            
    return rebx

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def sample_gradient(sorted_grad, p):
    if not sorted_grad:
        return 1.0
    if p <= sorted_grad[0]['p']:
        return sorted_grad[0]['a']
    if p >= sorted_grad[-1]['p']:
        return sorted_grad[-1]['a']
    for i in range(len(sorted_grad) - 1):
        g0 = sorted_grad[i]
        g1 = sorted_grad[i+1]
        if g0['p'] <= p <= g1['p']:
            t = (p - g0['p']) / (g1['p'] - g0['p'])
            return g0['a'] * (1.0 - t) + g1['a'] * t
    return 1.0

OBLIQUITY = math.radians(23.439)

@njit(cache=True)
def pole_to_ecliptic(pole_ra_deg, pole_dec_deg):
    """Convert pole RA/Dec (ICRF J2000) to unit vector in ecliptic coords."""
    ra = math.radians(pole_ra_deg)
    dec = math.radians(pole_dec_deg)
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    cos_e, sin_e = math.cos(OBLIQUITY), math.sin(OBLIQUITY)
    pole = np.array([x, y * cos_e + z * sin_e, -y * sin_e + z * cos_e])
    n = np.linalg.norm(pole)
    return pole / n if n > 0 else np.array([0., 0., 1.])

@njit(cache=True)
def build_equatorial_frame(pole_ecl):
    """Build 3x3 rotation matrix from parent equatorial frame to ecliptic."""
    z = pole_ecl / np.linalg.norm(pole_ecl)
    ref = np.array([0., 0., 1.])
    x = np.cross(ref, z)
    if np.linalg.norm(x) < 1e-10:
        x = np.array([1., 0., 0.])
    else:
        x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))

@njit(cache=True)
def kepler_solve(M, e, tol=1e-12):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
    E = M
    for _ in range(100):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E

@njit(cache=True)
def orbital_to_cartesian(a, e, inc, Omega, omega, M, mu):
    """Convert orbital elements to Cartesian position/velocity."""
    E = kepler_solve(M, e)
    cos_E, sin_E = math.cos(E), math.sin(E)
    r = a * (1.0 - e * cos_E)
    cos_f = (cos_E - e) / (1.0 - e * cos_E)
    sin_f = math.sqrt(max(0, 1.0 - e*e)) * sin_E / (1.0 - e * cos_E)
    x_orb, y_orb = r * cos_f, r * sin_f
    fac = math.sqrt(mu / max(a**3, 1e-30))
    denom = 1.0 - e * cos_E
    vx_orb = -fac * a * sin_E / denom
    vy_orb = fac * a * math.sqrt(max(0, 1.0 - e*e)) * cos_E / denom
    cos_O, sin_O = math.cos(Omega), math.sin(Omega)
    cos_i, sin_i = math.cos(inc), math.sin(inc)
    cos_w, sin_w = math.cos(omega), math.sin(omega)
    Px = cos_O*cos_w - sin_O*sin_w*cos_i
    Py = sin_O*cos_w + cos_O*sin_w*cos_i
    Pz = sin_w*sin_i
    Qx = -cos_O*sin_w - sin_O*cos_w*cos_i
    Qy = -sin_O*sin_w + cos_O*cos_w*cos_i
    Qz = cos_w*sin_i
    pos = np.array([x_orb*Px + y_orb*Qx, x_orb*Py + y_orb*Qy, x_orb*Pz + y_orb*Qz])
    vel = np.array([vx_orb*Px + vy_orb*Qx, vx_orb*Py + vy_orb*Qy, vx_orb*Pz + vy_orb*Qz])
    return pos, vel

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

ORBIT_RESOLUTION = 1001
ORBIT_STRIDE_BYTES = 160

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
                             visual_colors, cam_origin, G, max_orbits, orbit_buf):
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
        total_m = parent_m + orb_m
        if total_m <= 0.0:
            continue
            
        q = parent_m / total_m
        mu = G * total_m
        
        for k in range(3):
            orb_rel_pos[k] = subsys_pos[i, k] - pos[p_idx, k]
            orb_rel_vel[k] = subsys_vel[i, k] - vel[p_idx, k]
            bary[k] = (parent_m * pos[p_idx, k] + orb_m * subsys_pos[i, k]) / total_m
            
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
                    theta_asymp = math.acos(-1.0 / e_mag)
                    theta_max = theta_asymp * 0.95
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
                orbit_buf[n_orbits, 19] = 0.0
                n_orbits += 1
                
                if is_escape and gp_idx >= 0 and n_orbits < max_orbits:
                    gp_mu = G * (mass[gp_idx] + orb_m)
                    for k in range(3):
                        gp_rel_pos[k] = subsys_pos[i, k] - pos[gp_idx, k]
                        gp_rel_vel[k] = subsys_vel[i, k] - vel[gp_idx, k]
                        gp_bary[k] = (mass[gp_idx] * pos[gp_idx, k] + orb_m * subsys_pos[i, k]) / (mass[gp_idx] + orb_m)
                    
                    ge_hat, gq_hat, gbary_rel, gsl_p, ge_mag, gtheta0, gis_valid = compute_orbit_elements(
                        gp_rel_pos, gp_rel_vel, gp_mu, mass[gp_idx] / (mass[gp_idx] + orb_m), gp_bary, cam_origin,
                        False, False, gp_rel_pos)
                        
                    if gis_valid:
                        c_exit_r = sl_p / (1.0 + e_mag * math.cos(theta_exit)) if e_mag < 0.999 else sl_p / (1.0 + e_mag * math.cos(theta_exit))
                        gx = 0.0; gy = 0.0
                        for k in range(3):
                            c_exit_rel = (e_hat[k]*math.cos(theta_exit) + q_hat[k]*math.sin(theta_exit)) * c_exit_r
                            c_exit_abs = pos[p_idx, k] + c_exit_rel
                            g_rel = c_exit_abs - gp_bary[k]
                            gx += g_rel * ge_hat[k]
                            gy += g_rel * gq_hat[k]
                        gtheta_start = math.atan2(gy, gx)
                        
                        if ge_mag >= 1.0:
                            theta_asymp = math.acos(-1.0 / ge_mag)
                            theta_max = theta_asymp * 0.95
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
                        orbit_buf[n_orbits, 19] = 0.0
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
                cos_asym = -1.0 / e_mag
                cos_max = cos_asym + 0.05
                if cos_max > 1.0: cos_max = 1.0
                theta_max = math.acos(cos_max)
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
            orbit_buf[n_orbits, 19] = 0.0
            n_orbits += 1
            
    return n_orbits

def format_time_speed(multiplier):
    """Convert a speed multiplier (in seconds/second) to a human-readable string."""
    if multiplier < 1.5:
        return "Realtime"
    elif multiplier < 60:
        return f"{multiplier:.1f} sec/s"
    elif multiplier < 3600:
        return f"{multiplier / 60:.1f} min/s"
    elif multiplier < 86400:
        return f"{multiplier / 3600:.1f} hr/s"
    elif multiplier < 604800:
        return f"{multiplier / 86400:.1f} days/s"
    elif multiplier < 2629800:
        return f"{multiplier / 604800:.1f} weeks/s"
    elif multiplier < 31557600:
        return f"{multiplier / 2629800:.1f} months/s"
    else:
        return f"{multiplier / 31557600:.1f} years/s"

def format_sim_time(t_years):
    epoch = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    delta_seconds = t_years * 365.25 * 86400
    try:
        dt_utc = epoch + datetime.timedelta(seconds=delta_seconds)
        dt_local = dt_utc.astimezone()
        return dt_local.year, dt_local.month, dt_local.day, dt_local.hour, dt_local.minute
    except OverflowError:
        return 9999, 12, 31, 23, 59

def sim_time_from_date(y, m, d, h=0, mn=0):
    epoch = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    try:
        dt_local = datetime.datetime(int(y), int(m), int(d), int(h), int(mn), 0).astimezone()
        delta = dt_local - epoch
        return delta.total_seconds() / (365.25 * 86400)
    except ValueError:
        return 0.0

@njit(cache=True)
def compute_culling_masks(
    pos_rel_all, body_radii,
    star_idx, 
    n_casters, caster_indices, 
    n_rings, ring_centers, ring_normals, ring_outer_radii,
    num_bodies,
    caster_mask_lo, caster_mask_hi, ring_mask, ring_caster_lo, ring_caster_hi
):
    caster_mask_lo[:] = 0
    caster_mask_hi[:] = 0
    ring_mask[:] = 0
    ring_caster_lo[:] = 0
    ring_caster_hi[:] = 0
    
    star_pos = pos_rel_all[star_idx]
    star_r = body_radii[star_idx]
    
    for i in range(num_bodies):
        if i == star_idx:
            continue
            
        p_i = pos_rel_all[i]
        r_i = body_radii[i]
        
        L = star_pos - p_i
        dist_star = math.sqrt(L[0]*L[0] + L[1]*L[1] + L[2]*L[2])
        if dist_star < 1e-6:
            continue
        L_dir = L / dist_star
        
        mask_lo = np.uint32(0)
        mask_hi = np.uint32(0)
        
        for c_idx in range(n_casters):
            j = caster_indices[c_idx]
            if j == i or j == star_idx:
                continue
                
            p_j = pos_rel_all[j]
            r_j = body_radii[j]
            
            vec = p_j - p_i
            t = vec[0]*L_dir[0] + vec[1]*L_dir[1] + vec[2]*L_dir[2]
            
            if t > 0.0 and t < dist_star:
                perp = vec - t * L_dir
                d = math.sqrt(perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2])
                r_cone = r_j + star_r * (t / dist_star) + r_i
                
                if d < r_cone:
                    if c_idx < 32:
                        mask_lo |= np.uint32(1) << np.uint32(c_idx)
                    else:
                        mask_hi |= np.uint32(1) << np.uint32(c_idx - 32)
                        
        caster_mask_lo[i] = mask_lo
        caster_mask_hi[i] = mask_hi
        
        rmask = np.uint32(0)
        for k in range(n_rings):
            C_k = ring_centers[k]
            N_k = ring_normals[k]
            r_out = ring_outer_radii[k]
            
            denom = L_dir[0]*N_k[0] + L_dir[1]*N_k[1] + L_dir[2]*N_k[2]
            if abs(denom) > 1e-8:
                vec_c = C_k - p_i
                
                dist_centers = math.sqrt(vec_c[0]*vec_c[0] + vec_c[1]*vec_c[1] + vec_c[2]*vec_c[2])
                if dist_centers < 1e-6:
                    # Ring belongs to this planet, it will definitely cast a shadow on the planet
                    rmask |= np.uint32(1) << np.uint32(k)
                else:
                    s = (vec_c[0]*N_k[0] + vec_c[1]*N_k[1] + vec_c[2]*N_k[2]) / denom
                    if s > 0.0 and s < dist_star:
                        hit = p_i + s * L_dir
                        hit_vec = hit - C_k
                        d = math.sqrt(hit_vec[0]*hit_vec[0] + hit_vec[1]*hit_vec[1] + hit_vec[2]*hit_vec[2])
                        r_cone = star_r * (s / dist_star) + r_i
                        if d < r_out + r_cone:
                            rmask |= np.uint32(1) << np.uint32(k)
        ring_mask[i] = rmask

    for k in range(n_rings):
        C_k = ring_centers[k]
        r_out = ring_outer_radii[k]
        
        L = star_pos - C_k
        dist_star = math.sqrt(L[0]*L[0] + L[1]*L[1] + L[2]*L[2])
        if dist_star < 1e-6:
            continue
        L_dir = L / dist_star
        
        mask_lo = np.uint32(0)
        mask_hi = np.uint32(0)
        
        for c_idx in range(n_casters):
            j = caster_indices[c_idx]
            if j == star_idx:
                continue
                
            p_j = pos_rel_all[j]
            r_j = body_radii[j]
            
            vec = p_j - C_k
            t = vec[0]*L_dir[0] + vec[1]*L_dir[1] + vec[2]*L_dir[2]
            
            if t > 0.0 and t < dist_star:
                perp = vec - t * L_dir
                d = math.sqrt(perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2])
                r_cone = r_j + star_r * (t / dist_star) + r_out
                
                if d < r_cone:
                    if c_idx < 32:
                        mask_lo |= np.uint32(1) << np.uint32(c_idx)
                    else:
                        mask_hi |= np.uint32(1) << np.uint32(c_idx - 32)
                        
        ring_caster_lo[k] = mask_lo
        ring_caster_hi[k] = mask_hi

    return

@njit(cache=True)
def compute_frustum_culling(pos_rel_all, body_radii, planes, num_bodies):
    visible = np.ones(num_bodies, dtype=np.bool_)
    for i in range(num_bodies):
        x, y, z = pos_rel_all[i]
        r = body_radii[i]
        for p in range(6):
            dist = x*planes[p, 0] + y*planes[p, 1] + z*planes[p, 2] + planes[p, 3]
            if dist < -r:
                visible[i] = False
                break
    return visible

@njit(cache=True)
def extract_frustum_planes(vp):
    planes = np.empty((6, 4), dtype=np.float32)
    planes[0, :] = vp[:, 3] + vp[:, 0] # Left
    planes[1, :] = vp[:, 3] - vp[:, 0] # Right
    planes[2, :] = vp[:, 3] + vp[:, 1] # Bottom
    planes[3, :] = vp[:, 3] - vp[:, 1] # Top
    planes[4, :] = vp[:, 3] + vp[:, 2] # Near
    planes[5, :] = vp[:, 3] - vp[:, 2] # Far
    for i in range(6):
        x, y, z = planes[i, 0], planes[i, 1], planes[i, 2]
        length = math.sqrt(x*x + y*y + z*z)
        if length > 1e-8:
            planes[i] /= length
    return planes

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
    positions = np.ascontiguousarray(arr[:, 0:3])  # x, y, z (ecliptic)
    masses = arr[:, 9].copy()                      # m
    
    return _update_hierarchy_core(positions, masses, current_parents, num_bodies)

def build_tree_order(parent_indices, num_bodies):
    """Return (tree_indices, tree_depths) numpy arrays in depth-first tree-traversal order."""
    children = {i: [] for i in range(num_bodies)}
    roots = []
    for i in range(num_bodies):
        pi = int(parent_indices[i])
        if pi == -1:
            roots.append(i)
        else:
            children[pi].append(i)
    
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

def create_icosphere_mesh(subdivisions=4):
    """Create a unit icosphere mesh with interleaved positions and normals."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1,  t,  0), ( 1,  t,  0), (-1, -t,  0), ( 1, -t,  0),
        ( 0, -1,  t), ( 0,  1,  t), ( 0, -1, -t), ( 0,  1, -t),
        ( t,  0, -1), ( t,  0,  1), (-t,  0, -1), (-t,  0,  1)
    ]
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    for i in range(len(verts)):
        l = math.sqrt(verts[i][0]**2 + verts[i][1]**2 + verts[i][2]**2)
        verts[i] = (verts[i][0]/l, verts[i][1]/l, verts[i][2]/l)
        
    midpoint_cache = {}
    
    def get_midpoint(v1, v2):
        key = (min(v1, v2), max(v1, v2))
        if key in midpoint_cache:
            return midpoint_cache[key]
        p1, p2 = verts[v1], verts[v2]
        m = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2)
        l = math.sqrt(m[0]**2 + m[1]**2 + m[2]**2)
        m = (m[0]/l, m[1]/l, m[2]/l)
        idx = len(verts)
        verts.append(m)
        midpoint_cache[key] = idx
        return idx
        
    for _ in range(subdivisions):
        new_faces = []
        for tri in faces:
            v1, v2, v3 = tri
            a = get_midpoint(v1, v2)
            b = get_midpoint(v2, v3)
            c = get_midpoint(v3, v1)
            new_faces.extend([(v1, a, c), (v2, b, a), (v3, c, b), (a, b, c)])
        faces = new_faces
        
    v_arr = np.empty((len(verts), 6), dtype='f4')
    for i, v in enumerate(verts):
        v_arr[i, 0:3] = v
        v_arr[i, 3:6] = v
        
    i_arr = np.array(faces, dtype='i4').ravel()
    return v_arr.ravel(), i_arr

camera = {
    "target": np.array([0.0, 0.0, 0.0], dtype='f8'),
    "target_offset": np.array([0.0, 0.0, 0.0], dtype='f8'),
    "pan_offset": np.array([0.0, 0.0, 0.0], dtype='f8'),
    "distance": 45.0,     
    "distance_actual": 45.0,
    "fov": 45.0,
    "yaw": -90.0,         
    "yaw_actual": -90.0,
    "pitch": 25.0,        
    "pitch_actual": 25.0,
    "left_dragging": False,
    "right_dragging": False,
    "last_x": 0.0,
    "last_y": 0.0,
    "tracking_idx": 0,
    "tracking_mode": "body",
    "inspected_idx": None,
    "inspect_bary": False,
    "edit_mode": False,
    "edit_data": {},
}

time_ctrl = {
    "paused": False,
    "multiplier": 1.0,
    "render_timeline": False,
    "cancel_render": False,
    "target_t": 0.0,
    "sync_t": None,
    "timeline_playing": False,
    "timeline_speed": 10.0,
    "timeline_scrub_float": 0.0,
}

window_width, window_height = 1280, 720
fb_width, fb_height = 1280, 720
impl = None

def scroll_callback(window, xoffset, yoffset):
    if impl: impl.scroll_callback(window, xoffset, yoffset)
    if imgui.get_io().want_capture_mouse: return 
    
    is_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    
    if is_shift:
        fov_speed = 2.0
        if yoffset > 0: camera["fov"] -= fov_speed
        elif yoffset < 0: camera["fov"] += fov_speed
        camera["fov"] = max(1.0, min(120.0, camera["fov"]))
    else:
        zoom_speed = camera["distance"] * 0.2 
        if yoffset > 0: camera["distance"] -= zoom_speed
        elif yoffset < 0: camera["distance"] += zoom_speed
        camera["distance"] = max(1e-7, camera["distance"])

def mouse_button_callback(window, button, action, mods):
    if impl: impl.mouse_callback(window, button, action, mods)
    if imgui.get_io().want_capture_mouse: return 
    
    x, y = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            camera["left_dragging"] = True
            camera["last_x"], camera["last_y"] = x, y
        elif action == glfw.RELEASE: camera["left_dragging"] = False
            
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        if action == glfw.PRESS:
            camera["right_dragging"] = True
            camera["last_x"], camera["last_y"] = x, y
        elif action == glfw.RELEASE: camera["right_dragging"] = False

def cursor_pos_callback(window, xpos, ypos):
    dx = xpos - camera["last_x"]
    dy = ypos - camera["last_y"]
    
    if camera["left_dragging"]:
        sensitivity = 0.3
        camera["yaw"] += dx * sensitivity
        camera["pitch"] -= dy * sensitivity
        camera["pitch"] = max(-89.9, min(89.9, camera["pitch"])) 
        
    elif camera["right_dragging"]:
        pan_speed = camera["distance_actual"] * 0.001
        yaw_rad = math.radians(camera["yaw_actual"])
        pitch_rad = math.radians(camera["pitch_actual"])
        
        front = np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad)
        ], dtype='f4')
        front /= np.linalg.norm(front)
        
        right = np.cross(front, np.array([0.0, 1.0, 0.0], dtype='f4'))
        right /= np.linalg.norm(right)
        
        up = np.cross(right, front)
        up /= np.linalg.norm(up)
        
        pan_vec = -right * dx * pan_speed + up * dy * pan_speed
        if camera["tracking_idx"] is not None:
            camera["pan_offset"] += pan_vec
        else:
            camera["target"] += pan_vec

    camera["last_x"], camera["last_y"] = xpos, ypos

def char_callback(window, char):
    if impl: impl.char_callback(window, char)

def key_callback(window, key, scancode, action, mods):
    if impl: impl.keyboard_callback(window, key, scancode, action, mods)
    if imgui.get_io().want_capture_keyboard: return

    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            time_ctrl["paused"] = not time_ctrl["paused"]
        elif key in (glfw.KEY_UP, glfw.KEY_RIGHT):
            time_ctrl["multiplier"] = min(time_ctrl["multiplier"] * 2.0, 1e9)
        elif key in (glfw.KEY_DOWN, glfw.KEY_LEFT):
            time_ctrl["multiplier"] = max(time_ctrl["multiplier"] / 2.0, 1.0)
        elif key == glfw.KEY_R:
            time_ctrl["multiplier"] = 1.0

def resize_callback(window, width, height):
    if impl: impl.resize_callback(window, width, height)
    global window_width, window_height, fb_width, fb_height
    window_width, window_height = max(1, width), max(1, height)
    fb_width, fb_height = glfw.get_framebuffer_size(window)


def physics_loop(sim, num_bodies, shared_state, time_ctrl, running):
    # On Windows, set 1ms timer resolution (default is ~15ms which makes
    # time.sleep() wildly inaccurate for sub-16ms intervals).
    _timer_set = False
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        _timer_set = True
    except (AttributeError, OSError):
        pass  # Not Windows or winmm unavailable

    last_time = time.time()
    
    with shared_state["lock"]:
        current_parents = shared_state["parent_indices"].copy()
    
    current_parents = update_hierarchy(sim, num_bodies, current_parents)
    tree_indices, tree_depths = build_tree_order(current_parents, num_bodies)
    
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
                        for i in range(num_bodies):
                            curr = parents[i]
                            while curr != -1 and curr != idx:
                                curr = parents[curr]
                            if curr == idx and i != idx:
                                dp = sim.particles[i]
                                dp.x += dx; dp.y += dy; dp.z += dz
                                dp.vx += dvx; dp.vy += dvy; dp.vz += dvz
                                with shared_state["lock"]:
                                    shared_state["pos"][i] = (dp.x, dp.y, dp.z)
                                    shared_state["vel"][i] = (dp.vx, dp.vy, dp.vz)
                        
                    with shared_state["lock"]:
                        if "mass" in op:
                            shared_state["mass"][idx] = op["mass"]
                            shared_state["rebuild_flag"] = True
                        if "pos" in op:
                            shared_state["pos"][idx] = op["pos"]
                        if "vel" in op:
                            shared_state["vel"][idx] = op["vel"]
                            
                elif op["action"] == "CREATE":
                    mass = op["mass"]
                    pos = op["pos"]
                    vel = op["vel"]
                    parent_idx = op["parent_idx"]
                    
                    sim.add(m=mass, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])
                    num_bodies += 1
                    
                    with shared_state["lock"]:
                        shared_state["pos"] = np.vstack([shared_state["pos"], pos])
                        shared_state["vel"] = np.vstack([shared_state["vel"], vel])
                        shared_state["mass"] = np.append(shared_state["mass"], mass)
                        shared_state["parent_indices"] = np.append(shared_state["parent_indices"], parent_idx)
                        op["idx"] = num_bodies - 1
                        shared_state["crud_completed"].append(op)
                            
                elif op["action"] == "DELETE":
                    idx = op["idx"]
                    sim.remove(index=idx)
                    num_bodies -= 1
                    
                    with shared_state["lock"]:
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
                            
                        if shared_state["phys_star_idx"] == idx:
                            shared_state["phys_star_idx"] = -1
                        elif shared_state["phys_star_idx"] > idx:
                            shared_state["phys_star_idx"] -= 1
                        
                        shared_state["crud_completed"].append(op)
            
            with shared_state["lock"]:
                current_parents = shared_state["parent_indices"].copy()
            current_parents = update_hierarchy(sim, num_bodies, current_parents)
            tree_indices, tree_depths = build_tree_order(current_parents, num_bodies)
            
            with shared_state["lock"]:
                shared_state["parent_indices"][:] = current_parents
                shared_state["tree_indices"] = tree_indices
                shared_state["tree_depths"] = tree_depths
                shared_state["hierarchy_version"] += 1
                shared_state["rebuild_flag"] = True

        if time_ctrl.get("sync_t") is not None:
            sync_time = time_ctrl["sync_t"]
            time_ctrl["sync_t"] = None
            
            if sync_time < sim.t and 'tl_snapshots' in locals() and len(tl_snapshots) > 0:
                best_s = None
                for s in tl_snapshots:
                    if s.t <= sync_time + 1e-4:
                        best_s = s
                    else:
                        break
                
                if best_s is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sim = best_s.copy()
                    setup_reboundx(sim, shared_state["has_j2"], shared_state["has_gr"], shared_state["phys_star_idx"], shared_state["oblate_physics_list"])
                elif 'sim_timeline_base' in locals() and sync_time >= sim_timeline_base.t:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sim = sim_timeline_base.copy()
                    setup_reboundx(sim, shared_state["has_j2"], shared_state["has_gr"], shared_state["phys_star_idx"], shared_state["oblate_physics_list"])
            elif sync_time < sim.t and 'sim_timeline_base' in locals() and sync_time >= sim_timeline_base.t:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sim = sim_timeline_base.copy()
                setup_reboundx(sim, shared_state["has_j2"], shared_state["has_gr"], shared_state["phys_star_idx"], shared_state["oblate_physics_list"])
            
            diff = sync_time - sim.t
            if diff > 0.001:
                with shared_state["lock"]:
                    shared_state["syncing"] = True
                    shared_state["sync_progress"] = 0.0
                
                steps = max(10, int(diff / 0.1)) # About 100 steps per 10 years
                step_dt = diff / steps
                start_t = sim.t
                
                for i in range(steps):
                    if not running[0]: break
                    sim.integrate(start_t + (i + 1) * step_dt)
                    do_sync = (i % 10 == 0)
                    if do_sync:
                        local_pos = np.empty((num_bodies, 3), dtype=np.float64)
                        local_vel = np.empty((num_bodies, 3), dtype=np.float64)
                        _extract_render_state(sim, num_bodies, local_pos, local_vel)
                        sim_t = sim.t
                    with shared_state["lock"]:
                        shared_state["sync_progress"] = (i + 1.0) / steps
                        # Periodically update rendering so the screen doesn't freeze completely
                        if do_sync:
                            np.copyto(shared_state["pos"], local_pos)
                            np.copyto(shared_state["vel"], local_vel)
                            shared_state["t"] = sim_t
                            
                with shared_state["lock"]:
                    shared_state["syncing"] = False
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
            tl_times = np.zeros(num_steps, dtype='f8')
            tl_snapshots_temp = []
            
            with shared_state["lock"]:
                shared_state["timeline_active"] = True
                shared_state["timeline_progress"] = 0.0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim_timeline_base = sim.copy()
                sim_start_copy = sim.copy()
            setup_reboundx(sim_start_copy, shared_state["has_j2"], shared_state["has_gr"], shared_state["phys_star_idx"], shared_state["oblate_physics_list"])
                
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
                
                if step % 10 == 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tl_snapshots_temp.append(sim_start_copy.copy())
                
                if step % 20 == 0:
                    current_real_time = time.time()
                    elapsed = current_real_time - render_start_time
                    rate = (sim.t - start_t) / max(elapsed, 0.001)
                    with shared_state["lock"]:
                        shared_state["timeline_progress"] = float(step) / num_steps
                        shared_state["timeline_rate"] = rate
                valid_steps = step + 1
            
            if valid_steps > 0:
                tl_snapshots = tl_snapshots_temp
                local_pos = np.empty((num_bodies, 3), dtype=np.float64)
                local_vel = np.empty((num_bodies, 3), dtype=np.float64)
                _extract_render_state(sim_start_copy, num_bodies, local_pos, local_vel)
                sim_t = sim_start_copy.t
                with shared_state["lock"]:
                    shared_state["timeline_pos"] = tl_pos[:valid_steps]
                    shared_state["timeline_vel"] = tl_vel[:valid_steps]
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
            sim.integrate(sim.t + dt_sim)
            
            frame_count += 1
            if frame_count % 30 == 0:
                current_parents = update_hierarchy(sim, num_bodies, current_parents)
                tree_indices, tree_depths = build_tree_order(current_parents, num_bodies)
                
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

    # Restore default timer resolution on exit
    if _timer_set:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except (AttributeError, OSError):
            pass

def get_cartesian_from_keplerian(parent_m, child_m, a, e, inc_deg, Omega_deg, omega_deg, M_deg):
    """Generate Cartesian state from Keplerian elements using a temporary Rebound simulation."""
    import rebound
    sim_tmp = rebound.Simulation()
    sim_tmp.G = 4.0 * math.pi**2
    sim_tmp.add(m=parent_m)
    sim_tmp.add(m=child_m, a=a, e=e, inc=math.radians(inc_deg), Omega=math.radians(Omega_deg), omega=math.radians(omega_deg), M=math.radians(M_deg))
    p0 = sim_tmp.particles[0]
    p1 = sim_tmp.particles[1]
    return np.array([p1.x - p0.x, p1.y - p0.y, p1.z - p0.z]), np.array([p1.vx - p0.vx, p1.vy - p0.vy, p1.vz - p0.vz])

@njit(cache=True)
def rotate_equatorial_to_ecliptic(pos, vel, pole_ecl):
    """Rotate equatorial coordinates to ecliptic coordinates given a pole vector."""
    px, py, pz = pole_ecl
    if abs(px) < 1e-8 and abs(py) < 1e-8 and pz > 0:
        return pos, vel
        
    N = np.array([-py, px, 0.0])
    n_mag = np.linalg.norm(N)
    
    if n_mag < 1e-8:
        X = np.array([1.0, 0.0, 0.0])
        Y = np.array([0.0, -1.0, 0.0])
        Z = np.array([0.0, 0.0, -1.0])
    else:
        X = N / n_mag
        Z = np.array([px, py, pz])
        Z = Z / np.linalg.norm(Z)
        Y = np.cross(Z, X)
        
    R = np.column_stack((X, Y, Z))
    return R @ pos, R @ vel

@njit(cache=True)
def rotate_ecliptic_to_equatorial(pos, vel, pole_ecl):
    """Rotate ecliptic coordinates to equatorial coordinates given a pole vector."""
    px, py, pz = pole_ecl
    if abs(px) < 1e-8 and abs(py) < 1e-8 and pz > 0:
        return pos, vel
        
    N = np.array([-py, px, 0.0])
    n_mag = np.linalg.norm(N)
    
    if n_mag < 1e-8:
        X = np.array([1.0, 0.0, 0.0])
        Y = np.array([0.0, -1.0, 0.0])
        Z = np.array([0.0, 0.0, -1.0])
    else:
        X = N / n_mag
        Z = np.array([px, py, pz])
        Z = Z / np.linalg.norm(Z)
        Y = np.cross(Z, X)
        
    R = np.column_stack((X, Y, Z))
    R_inv = R.T
    return R_inv @ pos, R_inv @ vel

def generate_ring_arrays(pole_render, inner_r, outer_r, r_color, r_opacity, r_scatter, r_asymmetry, sorted_gradient):
    pole_n = pole_render / np.linalg.norm(pole_render)
    ref = np.array([0., 0., 1.])
    tangent = np.cross(pole_n, ref)
    if np.linalg.norm(tangent) < 1e-10:
        ref = np.array([1., 0., 0.])
        tangent = np.cross(pole_n, ref)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(pole_n, tangent)
    R_ring = np.column_stack([tangent, pole_n, bitangent])
    
    RING_SEGMENTS = 128
    RADIAL_SUBDIVISIONS = 32
    
    theta_arr = np.linspace(0, 2.0 * math.pi, RING_SEGMENTS + 1)
    p_arr = np.linspace(0, 1.0, RADIAL_SUBDIVISIONS + 1)
    theta_grid, p_grid = np.meshgrid(theta_arr, p_arr, indexing='ij')
    
    r_grid = inner_r + p_grid * (outer_r - inner_r)
    flat_r = r_grid.ravel()
    flat_ct = np.cos(theta_grid).ravel()
    flat_st = np.sin(theta_grid).ravel()
    n_verts = len(flat_r)
    
    eq_pos = np.zeros((n_verts, 3))
    eq_pos[:, 0] = flat_r * flat_ct
    eq_pos[:, 2] = flat_r * flat_st
    verts = (R_ring @ eq_pos.T).T.astype('f4')
    
    flat_p = p_grid.ravel()
    if sorted_gradient:
        grad_p = np.array([g['p'] for g in sorted_gradient])
        grad_a = np.array([g['a'] for g in sorted_gradient])
        alpha_mults = np.interp(flat_p, grad_p, grad_a).astype('f4')
        tex_p = np.linspace(0.0, 1.0, 256)
        shadow_grad = np.interp(tex_p, grad_p, grad_a).astype('f4')
    else:
        alpha_mults = np.ones(n_verts, dtype='f4')
        shadow_grad = np.ones(256, dtype='f4')
        
    colors = np.zeros((n_verts, 4), dtype='f4')
    colors[:, 0] = r_color[0]
    colors[:, 1] = r_color[1]
    colors[:, 2] = r_color[2]
    colors[:, 3] = r_opacity * alpha_mults
    
    stride = RADIAL_SUBDIVISIONS + 1
    ii, jj = np.meshgrid(np.arange(RING_SEGMENTS), np.arange(RADIAL_SUBDIVISIONS), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    p00 = ii * stride + jj
    p01 = p00 + 1
    p10 = (ii + 1) * stride + jj
    p11 = p10 + 1
    indices = np.column_stack([p00, p01, p10, p10, p01, p11]).ravel().astype('i4')
    
    return verts, indices, pole_n.astype('f4'), colors, shadow_grad

def rebuild_ring_render_group(bi, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex):
    rings = [r for r in ring_precomputed if r['body_idx'] == bi]
    
    existing_group = next((g for g in ring_render_groups if g['body_idx'] == bi), None)
    if existing_group:
        existing_group['vao'].release()
        ring_render_groups.remove(existing_group)
        
    if rings:
        all_ring_verts = []
        all_ring_indices = []
        vert_offset = 0
        for ring in rings:
            n_v = len(ring['verts'])
            ring_packed = np.zeros((n_v, 12), dtype='f4')
            ring_packed[:, 0:3] = ring['verts']
            ring_packed[:, 3:6] = ring['normal']
            ring_packed[:, 6:10] = ring['colors']
            ring_packed[:, 10] = ring['scatter']
            ring_packed[:, 11] = ring['asymmetry']
            all_ring_verts.append(ring_packed)
            all_ring_indices.append(ring['indices'] + vert_offset)
            vert_offset += n_v
        
        all_v = np.concatenate(all_ring_verts, axis=0)
        all_i = np.concatenate(all_ring_indices, axis=0)
        
        ring_vbo = ctx.buffer(all_v.tobytes())
        ring_ibo = ctx.buffer(all_i.tobytes())
        ring_vao = ctx.vertex_array(
            prog_rings,
            [(ring_vbo, '3f 3f 4f 1f 1f', 'in_position', 'in_normal', 'in_color', 'in_scatter', 'in_asymmetry')],
            index_buffer=ring_ibo
        )
        ring_render_groups.append({
            'body_idx': bi,
            'vao': ring_vao,
            'num_indices': len(all_i),
        })
        
    ring_gradient_data = np.zeros((16, 256), dtype='f4')
    for j, ring in enumerate(ring_precomputed):
        if j >= 16: break
        ring_gradient_data[j, :] = ring['shadow_grad']
    ring_gradient_tex.write(ring_gradient_data.tobytes())

def main():
    if _PERF_ENABLED:
        from scripts.perf_test import PerfTracker as _BootTracker
        _BootTracker().__class__
        if _PERF_TRACKER is None:
            _PERF_INSTALL_TRACKER(_BootTracker())

    sim = rebound.Simulation()
    sim.softening = 1e-6
    sim.integrator = "ias15"
    sim.units = ('AU', 'Msun', 'yr')
    
    with open('data/system.json', 'r') as file:
        bodies_data = json.load(file)

    name_to_idx = {b['name']: i for i, b in enumerate(bodies_data)}
    sorted_order = []
    _topo_visited = set()
    def _topo_visit(idx):
        if idx in _topo_visited:
            return
        _topo_visited.add(idx)
        body = bodies_data[idx]
        if 'parentId' in body:
            parent_local_idx = name_to_idx.get(body['parentId'])
            if parent_local_idx is not None:
                _topo_visit(parent_local_idx)
        sorted_order.append(idx)
    
    for i in range(len(bodies_data)):
        _topo_visit(i)
    
    bodies_data = [bodies_data[i] for i in sorted_order]
    name_to_idx = {b['name']: i for i, b in enumerate(bodies_data)}

    visual_data = []
    parent_indices = []
    pole_dirs = {}
    ring_bodies = []
    oblate_physics_list = []
    G_const = 4.0 * math.pi**2

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
        visual_data.append([color[0], color[1], color[2], radius_au, min_px, pole_render[0], pole_render[1], pole_render[2], obl])
        
        j2 = body.get('J2', 0.0)
        if j2 > 0.0:
            j4 = body.get('j4', 0.0)
            oblate_physics_list.append((idx, j2, j4, radius_au, pole_ecl, mass, name))
        
        if 'rings' in body:
            parent_pole_ecl = pole_to_ecliptic(body.get('pole_ra', 0), body.get('pole_dec', 90))
            pole_r = np.array([parent_pole_ecl[0], parent_pole_ecl[2], -parent_pole_ecl[1]])
            ring_bodies.append((idx, body['rings'], pole_r, radius_au))

        if 'parentId' in body:
            parent_name = body['parentId']
            if 'sv' in body:
                sv = body['sv']
                primary = sim.particles[parent_name]
                sim.add(m=mass,
                        x=primary.x + sv['x'], y=primary.y + sv['y'], z=primary.z + sv['z'],
                        vx=primary.vx + sv['vx'], vy=primary.vy + sv['vy'], vz=primary.vz + sv['vz'],
                        hash=name)
            elif body.get('orbitRef') == 'equatorial':
                parent_body_data = next((b for b in bodies_data if b['name'] == parent_name), None)
                if parent_body_data and 'pole_ra' in parent_body_data:
                    pole_ecl = pole_to_ecliptic(parent_body_data['pole_ra'], parent_body_data['pole_dec'])
                    R = build_equatorial_frame(pole_ecl)
                    primary = sim.particles[parent_name]
                    mu = G_const * (mass + primary.m)
                    pos_eq, vel_eq = orbital_to_cartesian(
                        body.get('a', 0.0), body.get('e', 0.0),
                        math.radians(body.get('inc', 0.0)),
                        math.radians(body.get('Omega', 0.0)),
                        math.radians(body.get('omega', 0.0)),
                        math.radians(body.get('M', 0.0)), mu)
                    pos_ecl = R @ pos_eq
                    vel_ecl = R @ vel_eq
                    sim.add(m=mass,
                            x=primary.x + pos_ecl[0], y=primary.y + pos_ecl[1], z=primary.z + pos_ecl[2],
                            vx=primary.vx + vel_ecl[0], vy=primary.vy + vel_ecl[1], vz=primary.vz + vel_ecl[2],
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

    if atmo_bodies:
        print(f"[Atmosphere] Parsed atmosphere data for {len(atmo_bodies)} bodies")

    has_j2 = bool(oblate_physics_list)
    if has_j2:
        oblate_indices = np.array([x[0] for x in oblate_physics_list], dtype=np.int32)
        oblate_j2 = np.array([x[1] for x in oblate_physics_list], dtype=np.float64)
        oblate_req = np.array([x[2] for x in oblate_physics_list], dtype=np.float64)
        oblate_poles = np.array([x[3] for x in oblate_physics_list], dtype=np.float64)
        oblate_masses = np.array([x[4] for x in oblate_physics_list], dtype=np.float64)

    phys_star_idx = -1
    phys_star_mass = 0.0
    for i, body in enumerate(bodies_data):
        if body.get('type') == 'Star':
            phys_star_idx = i
            phys_star_mass = body.get('m', 0.0)
            break
    has_gr = phys_star_idx >= 0

    if has_j2 or has_gr:
        rebx = setup_reboundx(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list)
        if has_j2:
            print(f"[J2] Attached J2 precession for {len(oblate_physics_list)} oblate bodies via REBOUNDx")
        if has_gr:
            print(f"[GR] Attached 1PN relativistic precession (star index {phys_star_idx}) via REBOUNDx")
        sim.force_is_velocity_dependent = 1

    num_bodies = len(sim.particles)
    
    shared_state = {
        "lock": threading.Lock(),
        "has_j2": has_j2,
        "has_gr": has_gr,
        "phys_star_idx": phys_star_idx,
        "oblate_physics_list": oblate_physics_list if has_j2 else [],
        "t": 0.0,
        "pos": np.zeros((num_bodies, 3), dtype='f8'),
        "vel": np.zeros((num_bodies, 3), dtype='f8'),
        "mass": np.zeros(num_bodies, dtype='f8'),
        "parent_indices": np.full(num_bodies, -1, dtype=np.int32),
        "tree_indices": np.zeros(num_bodies, dtype=np.int32),
        "tree_depths": np.zeros(num_bodies, dtype=np.int32),
        "hierarchy_version": 0,
        "timeline_active": False,
        "timeline_progress": 0.0,
        "timeline_pos": np.zeros((0, num_bodies, 3), dtype='f4'),
        "timeline_vel": np.zeros((0, num_bodies, 3), dtype='f4'),
        "timeline_times": np.zeros(0, dtype='f8'),
        "crud_queue": [],
        "crud_completed": [],
        "rebuild_flag": False,
        "oblate_indices": oblate_indices if has_j2 else None,
        "oblate_j2": oblate_j2 if has_j2 else None,
        "oblate_req": oblate_req if has_j2 else None,
        "oblate_poles": oblate_poles if has_j2 else None,
        "oblate_masses": oblate_masses if has_j2 else None,
        "phys_star_idx": phys_star_idx,
    }
    
    _init_arr = _get_particle_array(sim, num_bodies)
    shared_state["pos"][:, 0] = _init_arr[:, 0]    # x
    shared_state["pos"][:, 1] = _init_arr[:, 2]    # z
    shared_state["pos"][:, 2] = -_init_arr[:, 1]   # -y
    shared_state["vel"][:, 0] = _init_arr[:, 3]    # vx
    shared_state["vel"][:, 1] = _init_arr[:, 5]    # vz
    shared_state["vel"][:, 2] = -_init_arr[:, 4]   # -vy
    shared_state["mass"][:] = _init_arr[:, 9]      # m
        
    parent_indices = np.full(num_bodies, -1, dtype=np.int32)
    for i, body in enumerate(bodies_data):
        if 'parentId' in body:
            parent_indices[i] = name_to_idx[body['parentId']]
            
    parent_indices = update_hierarchy(sim, num_bodies, parent_indices)
    tree_indices_init, tree_depths_init = build_tree_order(parent_indices, num_bodies)
    shared_state["parent_indices"][:] = parent_indices
    shared_state["tree_indices"][:] = tree_indices_init
    shared_state["tree_depths"][:] = tree_depths_init
    
    print("[Init] Warming up Numba JIT functions...")
    try:
        _update_hierarchy_core(np.zeros((2, 3)), np.zeros(2), np.array([-1, 0], dtype=np.int32), 2)
        
        # Physics math functions are now offloaded to REBOUNDx C-extension
            
        compute_keplerian_elements(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        compute_barycenters(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), np.array([-1, 0], dtype=np.int32), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2))
        compute_all_orbits_batch(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), np.array([-1, 0], dtype=np.int32), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), np.zeros((2, 3), dtype=np.float64), np.zeros(3), 1.0, 10, np.zeros((10, 20), dtype=np.float64))
    except Exception as e:
        print(f"[Init] Warmup warning: {e}")
        
    running = [True]
    physics_thread = threading.Thread(target=physics_loop, args=(sim, num_bodies, shared_state, time_ctrl, running), daemon=True)
    physics_thread.start()

    def _glfw_error_callback(code, msg):
        print(f"[GLFW Error {code}] {msg}")
    glfw.set_error_callback(_glfw_error_callback)

    if not glfw.init(): raise Exception("GLFW failed to initialize")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(window_width, window_height, "3D Scale Orbit Sim", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW failed to create window. Your GPU may not support OpenGL 4.6.")

    glfw.make_context_current(window)
    glfw.swap_interval(0)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST) 

    global impl
    imgui.create_context()
    impl = GlfwRenderer(window, attach_callbacks=False)
    
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_window_size_callback(window, resize_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_char_callback(window, char_callback)



    sphere_vertex_shader = """
    #version 460 core
    in vec3 in_position;
    in vec3 in_normal;
    in vec3 in_offset;
    in vec3 in_color;
    in float in_radius;
    in float in_min_size;
    in float in_is_star;
    in vec3 in_pole;
    in float in_oblateness;
    in uvec2 in_caster_mask;
    in uint in_ring_mask;
    
    #define MAX_CASTERS 64
    layout(std140, binding = 1) uniform SceneData {
        mat4 projection;
        mat4 view;
        vec4 u_star_pos_radius;
        float u_far;
        float u_depth_C;
        int u_num_casters;
        float _pad0;
        vec4 u_casters[MAX_CASTERS];
        vec4 u_caster_poles_obl[MAX_CASTERS];
        vec4 u_caster_colors[MAX_CASTERS];
    };
    uniform float screen_height;
    uniform float fov_factor;
    
    out vec3 f_color;
    out vec3 f_world_pos;
    out vec3 f_normal;
    out float f_is_star;
    out float f_clip_z;
    flat out uvec2 f_caster_mask;
    flat out uint f_ring_mask;
    void main() {
        f_color = in_color;
        f_caster_mask = in_caster_mask;
        f_ring_mask = in_ring_mask;
        f_is_star = in_is_star;
        float dist = length((view * vec4(in_offset, 1.0)).xyz);
        float apparent_px = (in_radius / dist) * screen_height * fov_factor;
        float final_radius = in_radius;
        
        if (apparent_px < in_min_size) {
            final_radius = (in_min_size * dist) / (screen_height * fov_factor);
        }
        
        vec3 scaled_pos = in_position;
        vec3 adj_normal = in_normal;
        if (in_oblateness > 0.0) {
            float pole_proj = dot(in_position, in_pole);
            scaled_pos -= in_pole * (pole_proj * in_oblateness);
            // Transform normal by inverse-transpose of the squash Jacobian:
            // J = I - f*(p⊗p),  J^(-T) = I + f/(1-f)*(p⊗p)
            float f_inv = in_oblateness / (1.0 - in_oblateness);
            adj_normal = normalize(in_normal + in_pole * (dot(in_normal, in_pole) * f_inv));
        }
        
        vec3 world_pos = (scaled_pos * final_radius) + in_offset;
        f_world_pos = world_pos;
        f_normal = adj_normal;
        gl_Position = projection * view * vec4(world_pos, 1.0);
        f_clip_z = gl_Position.w;
    }
    """
    sphere_fragment_shader = """
    #version 460 core
    #define MAX_CASTERS 64
    #define MAX_RING_PLANES 16
    
    in vec3 f_color;
    in vec3 f_world_pos;
    in vec3 f_normal;
    in float f_is_star;
    in float f_clip_z;
    flat in uvec2 f_caster_mask;
    flat in uint f_ring_mask;
    
    layout(std140, binding = 1) uniform SceneData {
        mat4 projection;
        mat4 view;
        vec4 u_star_pos_radius;
        float u_far;
        float u_depth_C;
        int u_num_casters;
        float _pad0;
        vec4 u_casters[MAX_CASTERS];
        vec4 u_caster_poles_obl[MAX_CASTERS];
        vec4 u_caster_colors[MAX_CASTERS];
    };
    
    // Ring shadow planes (sphere-only)
    uniform vec3 u_ring_center[MAX_RING_PLANES];
    uniform vec3 u_ring_normal[MAX_RING_PLANES];
    uniform vec3 u_ring_params[MAX_RING_PLANES];
    uniform sampler2D u_ring_gradients;
    uniform int u_num_ring_planes;
    uniform int u_atmo_quality;
    
    out vec4 out_color;
    
    float get_oblate_radius(float r_eq, float oblateness, vec3 pole, vec3 L, vec3 perp_vec) {
        if (oblateness <= 0.0) return r_eq;
        float perp_len = length(perp_vec);
        if (perp_len < 1e-6) return r_eq;
        float PdotL = dot(pole, L);
        vec3 P_proj = pole - L * PdotL;
        float P_proj_len = length(P_proj);
        if (P_proj_len < 1e-6) return r_eq;
        vec3 P_dir = P_proj / P_proj_len;
        float y = dot(perp_vec, P_dir);
        float x = length(perp_vec - y * P_dir);
        float f_factor = 1.0 - oblateness;
        float R_minor = r_eq * sqrt(PdotL * PdotL + f_factor * f_factor * P_proj_len * P_proj_len);
        float denom = sqrt((x / r_eq) * (x / r_eq) + (y / R_minor) * (y / R_minor));
        return perp_len / denom;
    }
    
    void main() {
        if (f_is_star > 0.5) {
            // Star is self-luminous — no shading
            out_color = vec4(f_color, 1.0);
        } else {
            vec3 frag_to_star = u_star_pos_radius.xyz - f_world_pos;
            float dist_to_star = length(frag_to_star);
            vec3 L = frag_to_star / dist_to_star;
            
            // Angular radius of the star for soft penumbra at terminator
            float sin_alpha = clamp(u_star_pos_radius.w / dist_to_star, 0.0, 1.0);
            
            // Lambert cosine law: brightness decreases with angle
            // Combined with soft terminator from finite star angular size
            float NdotL = dot(normalize(f_normal), L);
            float diffuse = clamp((NdotL + sin_alpha) / (1.0 + sin_alpha), 0.0, 1.0);
            
            // === Analytical eclipse shadows ===
            float shadow = 1.0;
            for (int j = 0; j < u_num_casters; j++) {
                if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
                else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }
                
                vec3 caster_pos = u_casters[j].xyz;
                float caster_r = u_casters[j].w;
                
                vec3 frag_to_caster = caster_pos - f_world_pos;
                float dist_to_caster = length(frag_to_caster);
                
                // Skip self — fragment is on this body's surface (tight threshold to prevent self-shadow acne without skipping close moons)
                if (dist_to_caster < caster_r * 1.02) continue;
                
                // Project caster onto the fragment-to-star ray
                float t = dot(frag_to_caster, L);
                
                // Caster must be between fragment and star
                if (t <= 0.0 || t >= dist_to_star) continue;
                
                // Perpendicular distance from caster center to the ray
                vec3 perp_vec = frag_to_caster - t * L;
                float perp = length(perp_vec);
                
                // Account for oblateness
                float oblateness = u_caster_poles_obl[j].w;
                if (oblateness > 0.0) {
                    vec3 pole = u_caster_poles_obl[j].xyz;
                    caster_r = get_oblate_radius(caster_r, oblateness, pole, L, perp_vec);
                }
                
                // Project star disc to the caster's distance along the ray
                float r_star_proj = u_star_pos_radius.w * t / dist_to_star;
                
                // Shadow boundaries
                float r_penumbra = caster_r + r_star_proj;  // outer edge
                float r_umbra = abs(caster_r - r_star_proj); // inner edge
                
                // Max occlusion factor (for antumbra/annular eclipses)
                float max_occlusion = (caster_r >= r_star_proj) ? 1.0 : (caster_r * caster_r) / (r_star_proj * r_star_proj);
                
                // Smooth transition from max_occlusion (at umbra) to 0 (at penumbra)
                float occlusion = max_occlusion * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
                shadow *= (1.0 - occlusion);
            }
            
            // === Ring shadow on planet surface ===
            // For each ring plane, trace a ray from fragment toward star
            // and check if it intersects within the ring annulus
            for (int k = 0; k < u_num_ring_planes; k++) {
                if ((f_ring_mask & (1u << k)) == 0u) continue;
                
                vec3 ring_center = u_ring_center[k];
                vec3 ring_normal = u_ring_normal[k];
                float inner_r = u_ring_params[k].x;
                float outer_r = u_ring_params[k].y;
                float opacity = u_ring_params[k].z;
                
                // Ray-plane intersection: find t where (frag + t*L) lies on the ring plane
                // Plane equation: dot(P - ring_center, ring_normal) = 0
                float denom = dot(L, ring_normal);
                if (abs(denom) < 1e-8) continue;  // ray parallel to ring plane
                
                float t = dot(ring_center - f_world_pos, ring_normal) / denom;
                
                // Intersection must be between fragment and star (t > 0 and t < dist_to_star)
                if (t <= 0.0 || t >= dist_to_star) continue;
                
                // Compute intersection point and distance from ring center
                vec3 hit = f_world_pos + t * L;
                float dist_from_center = length(hit - ring_center);
                
                // Check if intersection is within the ring annulus
                if (dist_from_center >= inner_r && dist_from_center <= outer_r) {
                    // Soft edge falloff at inner and outer boundaries
                    float edge_width = (outer_r - inner_r) * 0.05;
                    float inner_fade = smoothstep(inner_r, inner_r + edge_width, dist_from_center);
                    float outer_fade = smoothstep(outer_r, outer_r - edge_width, dist_from_center);
                    float p = (dist_from_center - inner_r) / max(1e-6, outer_r - inner_r);
                    float alpha_mult = texture(u_ring_gradients, vec2(p, (float(k) + 0.5) / 16.0)).r;
                    float ring_shadow = inner_fade * outer_fade * opacity * alpha_mult;
                    shadow *= (1.0 - ring_shadow);
                }
            }
            
            diffuse *= shadow;
            
            // === Moonshine / Planetshine ===
            vec3 bounce_light = vec3(0.0);
            for (int j = 0; j < u_num_casters; j++) {
                if (j < 32) { if ((f_caster_mask.x & (1u << j)) == 0u) continue; }
                else        { if ((f_caster_mask.y & (1u << (j - 32))) == 0u) continue; }
                
                vec3 caster_pos = u_casters[j].xyz;
                float caster_r = u_casters[j].w;
                vec3 frag_to_caster = caster_pos - f_world_pos;
                float dist = length(frag_to_caster);
                
                if (dist < caster_r * 1.05) continue; // Skip self/very close
                
                vec3 dir_to_caster = frag_to_caster / dist;
                vec3 caster_to_star = normalize(u_star_pos_radius.xyz - caster_pos);
                
                float solid_angle = (caster_r * caster_r) / max(dist * dist, caster_r * caster_r);
                float phase = max(0.0, dot(caster_to_star, -dir_to_caster));
                float NdotC = max(0.0, dot(normalize(f_normal), dir_to_caster));
                
                // Simple form factor & intensity multiplier
                float intensity = phase * solid_angle * NdotC * 1.5;
                bounce_light += u_caster_colors[j].rgb * intensity;
            }
            
            // === Ringshine ===
            vec3 ring_shine = vec3(0.0);
            for (int k = 0; k < u_num_ring_planes; k++) {
                if ((f_ring_mask & (1u << k)) == 0u) continue;
                
                vec3 ring_center = u_ring_center[k];
                vec3 ring_normal = u_ring_normal[k];
                float inner_r = u_ring_params[k].x;
                float outer_r = u_ring_params[k].y;
                float opacity = u_ring_params[k].z;
                
                vec3 frag_to_ring = ring_center - f_world_pos;
                float dist_to_ring = length(frag_to_ring);
                if (dist_to_ring < 1e-6) continue;
                
                // Determine sun elevation over ring
                float sun_elevation = dot(L, ring_normal);
                
                // Determine fragment hemisphere relative to ring
                float frag_elevation = dot(normalize(f_normal), ring_normal);
                
                // Calculate geometric form factor based on fragment elevation
                // Rings take up max solid angle at mid-latitudes, zero at poles/equator
                float form_factor = abs(frag_elevation) * (1.0 - abs(frag_elevation)) * 4.0; 
                
                // Scale by ring area / distance
                float ring_area = (outer_r * outer_r - inner_r * inner_r);
                float solid_angle = ring_area / max(dist_to_ring * dist_to_ring, ring_area) * 0.1;
                
                float same_hemisphere = sun_elevation * frag_elevation;
                
                // Planet's shadow occlusion on the ring
                // If fragment is on the night side (dot(N, L) < 0), ring overhead might be in shadow
                vec3 N = normalize(f_normal);
                float shadow_occlusion = 1.0;
                if (dot(N, L) < 0.0) {
                    // How close is fragment to looking directly at the anti-solar point?
                    float anti_solar_alignment = max(0.0, dot(N, -L));
                    // Darkest part of the night side sees the shadowed portion of the ring
                    shadow_occlusion = 1.0 - (anti_solar_alignment * 0.95); // allow a tiny bit of bleed
                }
                
                float shine_intensity = 0.0;
                if (same_hemisphere > 0.0) {
                    // Reflected ringshine (same side as sun)
                    shine_intensity = abs(sun_elevation) * form_factor * solid_angle * opacity * 2.0;
                } else {
                    // Transmitted ringshine (bleeds through to the other side)
                    shine_intensity = abs(sun_elevation) * form_factor * solid_angle * opacity * 0.4;
                }
                
                // Apply shadow occlusion
                shine_intensity *= shadow_occlusion;
                
                // Base ring color is roughly the fragment color or white-ish for icy rings. 
                // We'll use a warm slightly scattered tint.
                vec3 ring_tint = vec3(0.9, 0.85, 0.8);
                ring_shine += ring_tint * shine_intensity;
            }
            
            vec3 final_color = f_color * diffuse + f_color * bounce_light + f_color * ring_shine;
            
            out_color = vec4(final_color, 1.0);
        }
        // Logarithmic depth — C constant improves near-field precision
        gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    }
    """
    prog_spheres = ctx.program(vertex_shader=sphere_vertex_shader, fragment_shader=sphere_fragment_shader)

    orbit_vertex_shader = """
    #version 460 core
    // Orbital element data stored in SSBO — doubles for precision
    struct OrbitData {
        dvec4 d0;  // e_hat.xyz, sl_p
        dvec4 d1;  // q_hat.xyz, e_mag
        dvec4 d2;  // bary_rel.xyz, cos_theta0
        dvec4 d3;  // color.rgb, sin_theta0
        dvec4 d4;  // empty, theta_max, is_patched, padding
    };
    
    layout(std430, binding = 0) buffer OrbitBuffer {
        OrbitData orbits[];
    };
    
    uniform mat4 projection;
    uniform mat4 view_rot;
    uniform dvec4 u_cam_pos_double;
    uniform float u_fade_dir;
    uniform float u_min_alpha;
    
    out vec4 f_color;
    out float f_clip_z;
    
    const int ORBIT_RES = 1001;
    
    void main() {
        OrbitData od = orbits[gl_InstanceID];
        
        dvec3 e_hat = od.d0.xyz;
        double sl_p = od.d0.w;
        dvec3 q_hat = od.d1.xyz;
        double e_mag = od.d1.w;
        dvec3 bary_rel = od.d2.xyz;
        double cos_A = od.d2.w;
        vec3 color = vec3(od.d3.xyz);
        double sin_A = od.d3.w;
        
        float delta_angle = float(od.d4.y);
        float is_patched = float(od.d4.z);
        
        float t = float(gl_VertexID) / float(ORBIT_RES - 1);
        float angle_B;
        if (delta_angle > 9.0) {
            angle_B = t * 2.0 * 3.14159265358979;
        } else {
            angle_B = t * delta_angle;
        }
        
        dvec3 pos;
        
        if (e_mag < 0.999) {
            // Elliptical orbit: use 64-bit pre-computed E0 to avoid 32-bit jitter
            double cos_E0 = cos_A;
            double sin_E0 = sin_A;
            
            double cos_B;
            double sin_B;
            
            if (gl_VertexID == ORBIT_RES - 1 && delta_angle > 9.0) {
                // GLSL 4.30 lacks 64-bit transcendental functions (cos/sin). 
                // The 32-bit float evaluation of cos(2.0 * PI) introduces a tiny gap.
                // We force the last vertex to perfectly close the loop!
                cos_B = 1.0lf;
                sin_B = 0.0lf;
            } else {
                cos_B = double(cos(angle_B));
                sin_B = double(sin(angle_B));
            }
            
            // E = E0 + angle_B
            double cos_E = cos_E0 * cos_B - sin_E0 * sin_B;
            double sin_E = sin_E0 * cos_B + cos_E0 * sin_B;
            
            double a = sl_p / (1.0 - e_mag * e_mag);
            double x = a * (cos_E - e_mag);
            double y = a * sqrt(1.0 - e_mag * e_mag) * sin_E;
            
            pos = bary_rel + x * e_hat + y * q_hat;
        } else {
            // Hyperbolic/Parabolic fallback using True Anomaly
            double cos_B = double(cos(angle_B));
            double sin_B = double(sin(angle_B));
            
            double cos_a = cos_A * cos_B - sin_A * sin_B;
            double sin_a = sin_A * cos_B + cos_A * sin_B;
            
            double denom = 1.0lf + e_mag * cos_a;
            double r = sl_p / max(denom, 1e-5lf);
            
            pos = bary_rel + (cos_a * e_hat + sin_a * q_hat) * r;
        }
        
        dvec3 eye_pos = pos - u_cam_pos_double.xyz;
        
        float t_val = float(gl_VertexID) / float(ORBIT_RES - 1);
        float t_curr = float(od.d4.x);
        
        float alpha = 1.0;
        if (delta_angle > 9.0) {
            if (u_fade_dir > 0.0) alpha = mix(u_min_alpha, 1.0, t_val);
            else alpha = mix(1.0, u_min_alpha, t_val);
        } else {
            if (u_fade_dir > 0.0) {
                if (t_val > t_curr) alpha = u_min_alpha;
                else {
                    float fraction = t_val / max(t_curr, 0.001);
                    alpha = mix(u_min_alpha, 1.0, fraction);
                }
            } else {
                if (t_val < t_curr) alpha = u_min_alpha;
                else {
                    float fraction = (t_val - t_curr) / max(1.0 - t_curr, 0.001);
                    alpha = mix(1.0, u_min_alpha, fraction);
                }
            }
        }
        
        // Dashed lines for patched conics
        if (is_patched > 0.5) {
            float dash = fract(float(gl_VertexID) / 10.0);
            if (dash > 0.5) alpha = 0.0;
            else alpha = mix(1.0, u_min_alpha, t_val);
        }
        
        f_color = vec4(color * 0.4, alpha);
        gl_Position = projection * view_rot * vec4(float(eye_pos.x), float(eye_pos.y), float(eye_pos.z), 1.0);
        f_clip_z = gl_Position.w;
    }
    """
    orbit_fragment_shader = """
    #version 460 core
    in vec4 f_color;
    in float f_clip_z;
    uniform float u_far;
    uniform float u_depth_C;
    out vec4 out_color;
    void main() {
        if (f_color.a < 0.01) discard; // discard dashes
        out_color = f_color;
        gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    }
    """
    prog_gpu_orbits = ctx.program(vertex_shader=orbit_vertex_shader, fragment_shader=orbit_fragment_shader)

    ring_vertex_shader = """
    #version 460 core
    in vec3 in_position;
    in vec3 in_normal;
    in vec4 in_color;
    in float in_scatter;
    in float in_asymmetry;
    
    #define MAX_CASTERS 64
    layout(std140, binding = 1) uniform SceneData {
        mat4 projection;
        mat4 view;
        vec4 u_star_pos_radius;
        float u_far;
        float u_depth_C;
        int u_num_casters;
        float _pad0;
        vec4 u_casters[MAX_CASTERS];
        vec4 u_caster_poles_obl[MAX_CASTERS];
        vec4 u_caster_colors[MAX_CASTERS];
    };
    uniform vec3 u_body_offset;
    
    out vec3 f_world_pos;
    out vec3 f_normal;
    out vec4 f_color;
    out float f_clip_z;
    out float f_scatter;
    out float f_asymmetry;
    
    void main() {
        vec3 world_pos = in_position + u_body_offset;
        f_world_pos = world_pos;
        f_normal = in_normal;
        f_color = in_color;
        f_scatter = in_scatter;
        f_asymmetry = in_asymmetry;
        gl_Position = projection * view * vec4(world_pos, 1.0);
        f_clip_z = gl_Position.w;
    }
    """
    ring_fragment_shader = """
    #version 460 core
    #define MAX_CASTERS 64
    
    in vec3 f_world_pos;
    in vec3 f_normal;
    in vec4 f_color;
    in float f_clip_z;
    in float f_scatter;
    in float f_asymmetry;
    
    layout(std140, binding = 1) uniform SceneData {
        mat4 projection;
        mat4 view;
        vec4 u_star_pos_radius;
        float u_far;
        float u_depth_C;
        int u_num_casters;
        float _pad0;
        vec4 u_casters[MAX_CASTERS];
        vec4 u_caster_poles_obl[MAX_CASTERS];
        vec4 u_caster_colors[MAX_CASTERS];
    };
    
    // Per-ring-body uniforms
    uniform vec3 u_host_planet_pos;
    uniform float u_host_planet_radius;
    uniform vec3 u_camera_pos;
    uniform vec4 u_host_planet_pole_obl;
    uniform int u_clip_mode;
    uniform uint u_caster_mask_lo;
    uniform uint u_caster_mask_hi;
    
    out vec4 out_color;
    
    float get_oblate_radius(float r_eq, float oblateness, vec3 pole, vec3 L, vec3 perp_vec) {
        if (oblateness <= 0.0) return r_eq;
        float perp_len = length(perp_vec);
        if (perp_len < 1e-6) return r_eq;
        float PdotL = dot(pole, L);
        vec3 P_proj = pole - L * PdotL;
        float P_proj_len = length(P_proj);
        if (P_proj_len < 1e-6) return r_eq;
        vec3 P_dir = P_proj / P_proj_len;
        float y = dot(perp_vec, P_dir);
        float x = length(perp_vec - y * P_dir);
        float f_factor = 1.0 - oblateness;
        float R_minor = r_eq * sqrt(PdotL * PdotL + f_factor * f_factor * P_proj_len * P_proj_len);
        float denom = sqrt((x / r_eq) * (x / r_eq) + (y / R_minor) * (y / R_minor));
        return perp_len / denom;
    }
    
    void main() {
        if (u_clip_mode != 0) {
            vec3 to_cam = u_camera_pos - u_host_planet_pos;
            vec3 to_frag = f_world_pos - u_host_planet_pos;
            float d = dot(to_frag, to_cam);
            if (u_clip_mode == 1 && d > 0.0) discard;
            if (u_clip_mode == 2 && d <= 0.0) discard;
        }
        
        vec3 frag_to_star = u_star_pos_radius.xyz - f_world_pos;
        float dist_to_star = length(frag_to_star);
        vec3 L = frag_to_star / dist_to_star;
        
        vec3 V = normalize(u_camera_pos - f_world_pos);
        
        // Scattering angle: cosine of angle between propagating light direction (-L) and view direction (V)
        float cos_theta = -dot(L, V);
        
        // Henyey-Greenstein phase function for backscattering (rocky/main rings) and forward-scattering (dusty/E-ring)
        float g_rock = -0.3;
        float denom_rock = 1.0 + g_rock * g_rock - 2.0 * g_rock * cos_theta;
        float rocky_phase = (1.0 - g_rock * g_rock) / (denom_rock * sqrt(denom_rock));
        rocky_phase = min(rocky_phase, 2.5); // prevent excessive brightness
        
        float g_dust = f_asymmetry; // Controlled by JSON (e.g. 0.7)
        float denom_dust = 1.0 + g_dust * g_dust - 2.0 * g_dust * cos_theta;
        float dusty_phase = (1.0 - g_dust * g_dust) / (denom_dust * sqrt(denom_dust));
        dusty_phase *= 0.15; // Peak is now ~1.2, preventing extreme blowout
        
        // Determine if camera and sun are on the same side of the ring plane using the ring normal N
        vec3 N = normalize(f_normal);
        float sun_side = dot(N, L);
        float cam_side = dot(N, V);
        
        // Smooth transition over a tiny angle to avoid aliasing / harsh floating point flips
        float same_side = smoothstep(-0.02, 0.02, sun_side * cam_side);
        float is_backlit = 1.0 - same_side;
        
        // Solar elevation attenuation (main rings get dimmer as sun gets edge-on, but not pitch black)
        float solar_elevation = mix(0.15, 1.0, abs(sun_side));
        
        // --- Multiple Scattering and B-Ring Inversion ---
        float rock_albedo = 1.0 - f_scatter;
        
        // Reflected pathway: Brightens with opacity. Adds multiple scattering boost.
        float rock_reflect = rocky_phase * rock_albedo * (f_color.a + f_color.a * f_color.a * 0.5);
        
        // Transmitted pathway (B-ring inversion):
        // Light must scatter through the particles. Peaks at medium opacity, dark at high opacity.
        float rock_transmit = rocky_phase * rock_albedo * f_color.a * pow(1.0 - f_color.a, 1.5) * 4.0;
        
        // Dusty rings (E-ring) behave differently. They scatter efficiently in transmission and reflection.
        // We MUST multiply by f_color.a to preserve the radial density gradient (fade out at edges).
        // Since dust rings usually have a very low base f_color.a, we boost the scattering significantly.
        // Option 1: Reduced multiplier from 10.0 to 2.5 to keep the peak tighter
        float dust_reflect = dusty_phase * f_scatter * f_color.a * 5;
        float dust_transmit = dusty_phase * f_scatter * f_color.a * 5; 
        
        // Apply solar elevation ONLY to the rock (main) rings. Dust rings scatter isotropically 
        // through their volume, so edge-on illumination doesn't dim them (it can actually brighten them).
        rock_reflect *= solar_elevation;
        rock_transmit *= solar_elevation;
        
        float reflected = rock_reflect + dust_reflect;
        float transmitted = rock_transmit + dust_transmit;
        
        // Combine based on which side we are viewing from
        float direct_illum = mix(transmitted, reflected, same_side);
        
        // Eclipse shadows from other bodies
        float shadow = 1.0;
        for (int j = 0; j < u_num_casters; j++) {
            if (j < 32) { if ((u_caster_mask_lo & (1u << j)) == 0u) continue; }
            else        { if ((u_caster_mask_hi & (1u << (j - 32))) == 0u) continue; }
            vec3 caster_pos = u_casters[j].xyz;
            float caster_r = u_casters[j].w;
            vec3 frag_to_caster = caster_pos - f_world_pos;
            float dist_to_caster = length(frag_to_caster);
            
            // Rings don't self-shadow from u_casters (casters are spherical bodies).
            float t = dot(frag_to_caster, L);
            if (t <= 0.0 || t >= dist_to_star) continue;
            vec3 perp_vec = frag_to_caster - t * L;
            float perp = length(perp_vec);
            
            float oblateness = u_caster_poles_obl[j].w;
            if (oblateness > 0.0) {
                vec3 pole = u_caster_poles_obl[j].xyz;
                caster_r = get_oblate_radius(caster_r, oblateness, pole, L, perp_vec);
            }
            
            float r_star_proj = u_star_pos_radius.w * t / dist_to_star;
            float r_penumbra = caster_r + r_star_proj;
            float r_umbra = abs(caster_r - r_star_proj);
            
            float max_occlusion = (caster_r >= r_star_proj) ? 1.0 : (caster_r * caster_r) / (r_star_proj * r_star_proj);
            float occlusion = max_occlusion * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
            shadow *= (1.0 - occlusion);
        }
        
        // === Host planet shadow on its own ring ===
        if (u_host_planet_radius > 0.0) {
            vec3 frag_to_host = u_host_planet_pos - f_world_pos;
            float dist_to_host = length(frag_to_host);
            
            // Project host onto fragment-to-star ray
            float t = dot(frag_to_host, L);
            
            // Host must be between ring fragment and star
            if (t > 0.0 && t < dist_to_star) {
                vec3 perp_vec = frag_to_host - t * L;
                float perp = length(perp_vec);
                
                float host_r = u_host_planet_radius;
                float host_oblateness = u_host_planet_pole_obl.w;
                if (host_oblateness > 0.0) {
                    vec3 host_pole = u_host_planet_pole_obl.xyz;
                    host_r = get_oblate_radius(host_r, host_oblateness, host_pole, L, perp_vec);
                }
                
                float r_star_proj = u_star_pos_radius.w * t / dist_to_star;
                float r_penumbra = host_r + r_star_proj;
                float r_umbra = abs(host_r - r_star_proj);
                
                float max_occlusion = (host_r >= r_star_proj) ? 1.0 : (host_r * host_r) / (r_star_proj * r_star_proj);
                float occlusion = max_occlusion * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
                shadow *= (1.0 - occlusion);
            }
        }
        
        direct_illum *= shadow; // solar_elevation was applied specifically to rock earlier
        
        // === Planetshine on Rings ===
        float planetshine = 0.0;
        if (u_host_planet_radius > 0.0) {
            vec3 frag_to_host = u_host_planet_pos - f_world_pos;
            float dist_host = length(frag_to_host);
            vec3 dir_to_host = frag_to_host / dist_host;
            vec3 host_to_star = normalize(u_star_pos_radius.xyz - u_host_planet_pos);
            
            // Planet phase and solid angle
            float planet_phase = max(0.0, dot(host_to_star, -dir_to_host));
            float R_sq = u_host_planet_radius * u_host_planet_radius;
            float solid_angle = R_sq / max(dist_host * dist_host, R_sq);
            float planet_elevation = mix(0.5, 1.0, abs(dot(N, dir_to_host)));
            
            float planet_side = dot(N, dir_to_host);
            float cam_planet_same_side = smoothstep(-0.02, 0.02, planet_side * cam_side);
            
            float shine_intensity = planet_phase * solid_angle * planet_elevation;
            
            // For planetshine, reflection and transmission are boosted so the dark side is visible
            float p_reflect = rock_reflect + 0.3 * f_color.a; 
            float p_transmit = rock_transmit + 0.2 * f_color.a;
            float shine_response = mix(p_transmit, p_reflect, cam_planet_same_side);
            
            planetshine = shine_intensity * shine_response * 0.8;
        }
        
        // No artificial ambient glow - shadow and unlit sides should be physically black 
        // unless illuminated by planetshine (calculated above).
        float illumination = direct_illum + planetshine;
        
        // Boost alpha when backlit - dust_transmit already factors in f_color.a to preserve gradients.
        float scattered_alpha_boost = dust_transmit * shadow * is_backlit * 1.0;
        // With premultiplied alpha, the alpha channel strictly represents OCCLUSION of the background.
        // Dust forward-scattering (boost) does NOT increase occlusion, it only increases brightness!
        // So we strictly use the base opacity (f_color.a) as the occlusion value, allowing dust rings to glow 
        // brighter than their opacity without physically blocking the stars behind them.
        float final_alpha = clamp(f_color.a, 0.0, 1.0);
        
        // Use the actual ring color for backlit scattering instead of a hardcoded blue tint
        vec3 tinted_color = f_color.rgb;
        
        // Options 2 & 3: Soft curve and allowing color clipping (blowing out to white)
        // Instead of a hard clamp that creates a massive flat plateau, we let the raw color 
        // scale with illumination, then apply a soft exponential curve. This preserves 
        // the sharp center of the highlight as it naturally rolls off into pure white.
        vec3 raw_color = tinted_color * illumination;
        vec3 final_color = 1.0 - exp(-raw_color);
        
        out_color = vec4(final_color, final_alpha);
        gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    }
    """
    prog_rings = ctx.program(vertex_shader=ring_vertex_shader, fragment_shader=ring_fragment_shader)

    atmo_vertex_shader = """
    #version 460 core
    in vec3 in_position;

    #define MAX_CASTERS 64
    layout(std140, binding = 1) uniform SceneData {
        mat4 projection;
        mat4 view;
        vec4 u_star_pos_radius;
        float u_far;
        float u_depth_C;
        int u_num_casters;
        float _pad0;
        vec4 u_casters[MAX_CASTERS];
        vec4 u_caster_poles_obl[MAX_CASTERS];
        vec4 u_caster_colors[MAX_CASTERS];
    };

    uniform vec3 u_body_offset;
    uniform float u_atmo_radius_au;

    out vec3 f_world_pos;
    out vec3 f_local_pos;
    out float f_clip_z;

    void main() {
        vec3 world_pos = in_position * u_atmo_radius_au + u_body_offset;
        f_world_pos = world_pos;
        f_local_pos = in_position;
        gl_Position = projection * view * vec4(world_pos, 1.0);
        f_clip_z = gl_Position.w;
    }
    """
    atmo_fragment_shader = """
    #version 460 core
    #define MAX_CASTERS 64
    #define MAX_RING_PLANES 16
    #define PI 3.14159265358979

    in vec3 f_world_pos;
    in vec3 f_local_pos;
    in float f_clip_z;

    layout(std140, binding = 1) uniform SceneData {
        mat4 projection;
        mat4 view;
        vec4 u_star_pos_radius;
        float u_far;
        float u_depth_C;
        int u_num_casters;
        float _pad0;
        vec4 u_casters[MAX_CASTERS];
        vec4 u_caster_poles_obl[MAX_CASTERS];
        vec4 u_caster_colors[MAX_CASTERS];
    };

    uniform vec3  u_body_offset;
    uniform float u_atmo_radius_au;
    uniform float u_planet_radius_km;
    uniform float u_atmo_radius_km;
    uniform float u_au_to_km;
    uniform vec3  u_beta_rayleigh;
    uniform float u_h_rayleigh;
    uniform float u_beta_mie;
    uniform float u_h_mie;
    uniform float u_mie_g;
    uniform vec3  u_beta_absorption;
    uniform float u_sun_intensity;
    uniform vec3  u_camera_pos;
    uniform int   u_num_samples;
    uniform int   u_num_light_samples;
    uniform vec4  u_pole_obl;
    uniform uint  u_caster_mask_lo;
    uniform uint  u_caster_mask_hi;
    uniform uint  u_ring_mask;

    // Ring shadow planes (sphere-only)
    uniform sampler2D u_ring_gradients;
    uniform int u_num_ring_planes;
    uniform int u_atmo_quality;
    uniform vec3 u_ring_center[MAX_RING_PLANES];
    uniform vec3 u_ring_normal[MAX_RING_PLANES];
    uniform vec3 u_ring_params[MAX_RING_PLANES]; // x=inner_r, y=outer_r, z=opacity

    out vec4 out_color;

    int g_num_local_casters;
    vec4 g_local_casters[64];
    
    int g_num_local_rings;
    int g_local_rings[16];

    vec3 toSphericalSpace(vec3 p, vec4 pole_obl) {
        float f = pole_obl.w;
        if (f == 0.0) return p;
        vec3 pole = pole_obl.xyz;
        float h = dot(p, pole);
        vec3 p_perp = p - h * pole;
        return p_perp + (h / (1.0 - f)) * pole;
    }

    float compute_shadow(vec3 eval_render_pos, vec3 L_dir, float dist_to_star, vec3 planet_center_render) {
        float shadow = 1.0;
        for (int i = 0; i < g_num_local_casters; i++) {
            vec3 caster_pos = g_local_casters[i].xyz;
            float caster_r = g_local_casters[i].w;
            vec3 s_to_c = caster_pos - eval_render_pos;
            float proj = dot(s_to_c, L_dir);
            if (proj <= 0.0 || proj >= dist_to_star) continue;
            vec3 perp_vec = s_to_c - proj * L_dir;
            float perp = length(perp_vec);
            float r_star_proj = u_star_pos_radius.w * proj / dist_to_star;
            float r_penumbra = caster_r + r_star_proj;
            float r_umbra = abs(caster_r - r_star_proj);
            float max_occ = (caster_r >= r_star_proj) ? 1.0 : (caster_r * caster_r) / (r_star_proj * r_star_proj);
            float occ = max_occ * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
            shadow *= (1.0 - occ);
        }
        for (int i = 0; i < g_num_local_rings; i++) {
            int k = g_local_rings[i];
            vec3 ring_center = u_ring_center[k];
            vec3 ring_normal = u_ring_normal[k];
            float inner_r = u_ring_params[k].x;
            float outer_r = u_ring_params[k].y;
            float opacity = u_ring_params[k].z;
            float denom = dot(L_dir, ring_normal);
            if (abs(denom) < 1e-8) continue;
            float t_ring = dot(ring_center - eval_render_pos, ring_normal) / denom;
            if (t_ring <= 0.0 || t_ring >= dist_to_star) continue;
            vec3 hit = eval_render_pos + t_ring * L_dir;
            float dist_from_center = length(hit - ring_center);
            if (dist_from_center >= inner_r && dist_from_center <= outer_r) {
                float edge_width = (outer_r - inner_r) * 0.05;
                float inner_fade = smoothstep(inner_r, inner_r + edge_width, dist_from_center);
                float outer_fade = smoothstep(outer_r, outer_r - edge_width, dist_from_center);
                float p = (dist_from_center - inner_r) / max(1e-6, outer_r - inner_r);
                float alpha_mult = texture(u_ring_gradients, vec2(p, (float(k) + 0.5) / 16.0)).r;
                float ring_shadow = inner_fade * outer_fade * opacity * alpha_mult;
                shadow *= (1.0 - ring_shadow);
            }
        }
        return shadow;
    }

    vec2 raySphereIntersect(vec3 origin, vec3 dir, float radius) {
        float a = dot(dir, dir);
        float b = dot(origin, dir);
        float c = dot(origin, origin) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant < 0.0) return vec2(1e10, -1e10);
        float d = sqrt(discriminant);
        return vec2((-b - d) / a, (-b + d) / a);
    }

    void main() {
        g_num_local_casters = 0;
        for (int k = 0; k < u_num_casters; k++) {
            if (k < 32) { if ((u_caster_mask_lo & (1u << k)) == 0u) continue; }
            else        { if ((u_caster_mask_hi & (1u << (k - 32))) == 0u) continue; }
            if (length(u_casters[k].xyz - u_body_offset) < 1e-6) continue;
            if (g_num_local_casters < 64) {
                g_local_casters[g_num_local_casters++] = u_casters[k];
            }
        }
        
        g_num_local_rings = 0;
        for (int k = 0; k < u_num_ring_planes; k++) {
            if ((u_ring_mask & (1u << k)) == 0u) continue;
            if (g_num_local_rings < 16) {
                g_local_rings[g_num_local_rings++] = k;
            }
        }

        // 1. Setup: convert to planet-local km coordinates
        vec3 planet_center_render = u_body_offset;
        vec3 cam_local_au = u_camera_pos - planet_center_render;
        
        // Fix catastrophic precision loss by computing fragment pos purely from local data
        vec3 frag_local_au = f_local_pos * u_atmo_radius_au;

        vec3 cam_local = cam_local_au * u_au_to_km;
        vec3 frag_local = frag_local_au * u_au_to_km;

        vec3 ray_dir = normalize(frag_local - cam_local);

        vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);
        vec3 frag_local_sph = toSphericalSpace(frag_local, u_pole_obl);

        // 2. Ray-sphere intersections (planet-local km)
        // Shifting intersection origin to the fragment itself to preserve float32 precision
        // when the camera is extremely far away. 's' is the parameter along ray_dir starting from frag_local.
        vec2 s_atmo = raySphereIntersect(frag_local_sph, ray_dir_sph, u_atmo_radius_km);
        if (s_atmo.x > s_atmo.y) discard;

        vec2 s_planet = raySphereIntersect(frag_local_sph, ray_dir_sph, u_planet_radius_km);

        float dist_to_frag = length(frag_local - cam_local);
        float s_cam = -dist_to_frag; // Camera position relative to frag_local

        float s_start = max(s_atmo.x, s_cam);
        float s_end = s_atmo.y;
        
        if (s_planet.x > s_cam && s_planet.x < s_end) {
            s_end = s_planet.x;
        }

        // Clip against all other opaque bodies (moons, planets) to fix depth sorting
        // since atmosphere is rendered with DEPTH_TEST disabled.
        for (int k = 0; k < u_num_casters; k++) {
            vec3 caster_pos = u_casters[k].xyz;
            float caster_r = u_casters[k].w * u_au_to_km;
            vec3 caster_local = (caster_pos - planet_center_render) * u_au_to_km;
            
            // Skip the host planet itself (already clipped precisely above)
            if (length(caster_local) < 1.0) continue;
            
            vec4 pole_obl = u_caster_poles_obl[k];
            vec3 origin_c = toSphericalSpace(frag_local - caster_local, pole_obl);
            vec3 dir_c = toSphericalSpace(ray_dir, pole_obl);
            
            vec2 s_c = raySphereIntersect(origin_c, dir_c, caster_r);
            if (s_c.x > s_cam && s_c.x < s_end) {
                s_end = s_c.x;
            }
        }

        if (s_start >= s_end) discard;

        // 3. Sun direction (in km-space)
        vec3 sun_pos_local = (u_star_pos_radius.xyz - planet_center_render) * u_au_to_km;
        vec3 sun_dir = normalize(sun_pos_local);
        vec3 sun_dir_sph = toSphericalSpace(sun_dir, u_pole_obl);

        // Scattering coefficients: convert m^-1 to km^-1
        vec3 beta_R = u_beta_rayleigh * 1000.0;
        vec3 beta_M = vec3(u_beta_mie) * 1000.0;
        vec3 beta_A = u_beta_absorption * 1000.0;
        float mie_ext_factor = 1.0 / 0.9;

        // 3.5 Global Eclipse & Ring Shadow
        float global_eclipse_shadow = 1.0;
        if (u_atmo_quality == 1) { // Low Quality (2D shadow at midpoint)
            float s_mid = (s_start + s_end) * 0.5;
            vec3 mid_pos = frag_local + s_mid * ray_dir;
            vec3 mid_render = mid_pos / u_au_to_km + planet_center_render;
            vec3 mid_to_star = u_star_pos_radius.xyz - mid_render;
            float dist_mid_star = length(mid_to_star);
            vec3 L_mid = mid_to_star / dist_mid_star;
            global_eclipse_shadow = compute_shadow(mid_render, L_mid, dist_mid_star, planet_center_render);
        }
        // 4. Primary ray march
        float step_size = (s_end - s_start) / float(u_num_samples);
        vec3 total_rayleigh = vec3(0.0);
        vec3 total_mie = vec3(0.0);
        float od_rayleigh = 0.0;
        float od_mie = 0.0;

        for (int i = 0; i < u_num_samples; i++) {
            float s = s_start + (float(i) + 0.5) * step_size;
            vec3 sample_pos = frag_local + s * ray_dir;
            vec3 sample_pos_sph = frag_local_sph + s * ray_dir_sph;
            float altitude = length(sample_pos_sph) - u_planet_radius_km;

            float rho_R = exp(-altitude / u_h_rayleigh);
            float rho_M = exp(-altitude / u_h_mie);

            od_rayleigh += rho_R * step_size;
            od_mie += rho_M * step_size;

            // 5. Light ray: optical depth from sample to sun
            vec2 t_light_atmo = raySphereIntersect(sample_pos_sph, sun_dir_sph, u_atmo_radius_km);
            float light_ray_len = t_light_atmo.y;
            float light_step = light_ray_len / float(u_num_light_samples);

            float od_light_R = 0.0;
            float od_light_M = 0.0;
            bool in_planet_shadow = false;

            for (int j = 0; j < u_num_light_samples; j++) {
                float t_l = (float(j) + 0.5) * light_step;
                vec3 light_pos_sph = sample_pos_sph + t_l * sun_dir_sph;
                float light_alt = length(light_pos_sph) - u_planet_radius_km;

                if (light_alt < 0.0) {
                    in_planet_shadow = true;
                    break;
                }

                od_light_R += exp(-light_alt / u_h_rayleigh) * light_step;
                od_light_M += exp(-light_alt / u_h_mie) * light_step;
            }

            if (in_planet_shadow) continue;

            // 7. Accumulate in-scattered light
            vec3 tau = beta_R * (od_rayleigh + od_light_R)
                     + beta_M * mie_ext_factor * (od_mie + od_light_M)
                     + beta_A * (od_rayleigh + od_light_R);
            float sample_shadow = global_eclipse_shadow;
            if (u_atmo_quality == 2) { // High Quality (Volumetric)
                vec3 sample_render = sample_pos / u_au_to_km + planet_center_render;
                vec3 sample_to_star = u_star_pos_radius.xyz - sample_render;
                float dist_sample_star = length(sample_to_star);
                vec3 L_sample = sample_to_star / dist_sample_star;
                sample_shadow = compute_shadow(sample_render, L_sample, dist_sample_star, planet_center_render);
            }
            // Approximate multiple scattering with a low-extinction term
            vec3 direct_attenuation = exp(-tau);
            vec3 ms_attenuation = max(vec3(0.0), (exp(-tau * 0.2) - direct_attenuation) * 0.4);
            vec3 attenuation = (direct_attenuation + ms_attenuation) * sample_shadow;

            total_rayleigh += rho_R * attenuation * step_size;
            total_mie      += rho_M * attenuation * step_size;
        }

        // 8. Phase functions
        float cos_theta = dot(ray_dir, sun_dir);

        float phase_R = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);

        float g = u_mie_g;
        float g2 = g * g;
        float phase_M = (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta))
                      / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));

        // 9. Final scattered light
        vec3 scattered = u_sun_intensity * (
            phase_R * beta_R * total_rayleigh +
            phase_M * beta_M * total_mie
        );

        // 10. Primary ray transmittance
        vec3 transmittance = exp(-(beta_R * od_rayleigh
                                 + beta_M * mie_ext_factor * od_mie
                                 + beta_A * od_rayleigh));

        // Tone mapping to prevent blowout while preserving hue
        float max_scatter = max(scattered.r, max(scattered.g, scattered.b));
        if (max_scatter > 1e-6) {
            float mapped_max = 1.0 - exp(-max_scatter);
            scattered = scattered * (mapped_max / max_scatter);
        }

        // 11. Premultiplied alpha output
        float avg_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
        out_color = vec4(scattered, 1.0 - avg_transmittance);

        // Logarithmic depth
        gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0))
                     / log2(u_depth_C * u_far + 1.0);
    }
    """
    prog_atmo = ctx.program(vertex_shader=atmo_vertex_shader, fragment_shader=atmo_fragment_shader)

    ring_precomputed = []
    RING_SEGMENTS = 128
    for body_idx, rings_data, pole_render, body_radius_au in ring_bodies:
        pole_n = pole_render / np.linalg.norm(pole_render)
        ref = np.array([0., 0., 1.])
        tangent = np.cross(pole_n, ref)
        if np.linalg.norm(tangent) < 1e-10:
            ref = np.array([1., 0., 0.])
            tangent = np.cross(pole_n, ref)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(pole_n, tangent)
        R_ring = np.column_stack([tangent, pole_n, bitangent])
        
        for ring_seg in rings_data:
            inner_r = ring_seg['inner'] * body_radius_au
            outer_r = ring_seg['outer'] * body_radius_au
            r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
            r_opacity = ring_seg.get('opacity', 1.0)
            r_scatter = ring_seg.get('scatter', 0.0)
            r_asymmetry = ring_seg.get('asymmetry', 0.7)
            
            RADIAL_SUBDIVISIONS = 32
            sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
            
            theta_arr = np.linspace(0, 2.0 * math.pi, RING_SEGMENTS + 1)
            p_arr = np.linspace(0, 1.0, RADIAL_SUBDIVISIONS + 1)
            theta_grid, p_grid = np.meshgrid(theta_arr, p_arr, indexing='ij')
            
            r_grid = inner_r + p_grid * (outer_r - inner_r)
            flat_r = r_grid.ravel()
            flat_ct = np.cos(theta_grid).ravel()
            flat_st = np.sin(theta_grid).ravel()
            n_verts = len(flat_r)
            
            eq_pos = np.zeros((n_verts, 3))
            eq_pos[:, 0] = flat_r * flat_ct
            eq_pos[:, 2] = flat_r * flat_st
            verts = (R_ring @ eq_pos.T).T.astype('f4')
            
            flat_p = p_grid.ravel()
            if sorted_gradient:
                grad_p = np.array([g['p'] for g in sorted_gradient])
                grad_a = np.array([g['a'] for g in sorted_gradient])
                alpha_mults = np.interp(flat_p, grad_p, grad_a).astype('f4')
                
                tex_p = np.linspace(0.0, 1.0, 256)
                shadow_grad = np.interp(tex_p, grad_p, grad_a).astype('f4')
            else:
                alpha_mults = np.ones(n_verts, dtype='f4')
                shadow_grad = np.ones(256, dtype='f4')
            
            colors = np.zeros((n_verts, 4), dtype='f4')
            colors[:, 0] = r_color[0]
            colors[:, 1] = r_color[1]
            colors[:, 2] = r_color[2]
            colors[:, 3] = r_opacity * alpha_mults
            
            stride = RADIAL_SUBDIVISIONS + 1
            ii, jj = np.meshgrid(np.arange(RING_SEGMENTS), np.arange(RADIAL_SUBDIVISIONS), indexing='ij')
            ii, jj = ii.ravel(), jj.ravel()
            p00 = ii * stride + jj
            p01 = p00 + 1
            p10 = (ii + 1) * stride + jj
            p11 = p10 + 1
            indices = np.column_stack([p00, p01, p10, p10, p01, p11]).ravel().astype('i4')
            
            ring_precomputed.append({
                'body_idx': body_idx,
                'verts': np.array(verts, dtype='f4'),
                'indices': np.array(indices, dtype='i4'),
                'normal': pole_n.astype('f4'),
                'colors': np.array(colors, dtype='f4'),
                'inner_r': inner_r,
                'outer_r': outer_r,
                'opacity': r_opacity,
                'scatter': r_scatter,
                'asymmetry': r_asymmetry,
                'shadow_grad': shadow_grad,
                'raw_color': r_color,
                'gradient': sorted_gradient,
            })

    ring_gradient_data = np.zeros((16, 256), dtype='f4')
    for j, ring in enumerate(ring_precomputed):
        if j >= 16: break
        ring_gradient_data[j, :] = ring['shadow_grad']
    
    ring_gradient_tex = ctx.texture((256, 16), 1, ring_gradient_data.tobytes(), dtype='f4')
    ring_gradient_tex.filter = (moderngl.LINEAR, moderngl.NEAREST)
    ring_gradient_tex.repeat_x = False
    ring_gradient_tex.repeat_y = False

    ring_render_groups = []
    rings_by_body_init = {}
    for ring in ring_precomputed:
        bi = ring['body_idx']
        if bi not in rings_by_body_init:
            rings_by_body_init[bi] = []
        rings_by_body_init[bi].append(ring)
    
    for bi, rings in rings_by_body_init.items():
        all_ring_verts = []
        all_ring_indices = []
        vert_offset = 0
        for ring in rings:
            n_v = len(ring['verts'])
            ring_packed = np.zeros((n_v, 12), dtype='f4')
            ring_packed[:, 0:3] = ring['verts']
            ring_packed[:, 3:6] = ring['normal']
            ring_packed[:, 6:10] = ring['colors']
            ring_packed[:, 10] = ring['scatter']
            ring_packed[:, 11] = ring['asymmetry']
            all_ring_verts.append(ring_packed)
            all_ring_indices.append(ring['indices'] + vert_offset)
            vert_offset += n_v
        
        all_v = np.concatenate(all_ring_verts, axis=0)
        all_i = np.concatenate(all_ring_indices, axis=0)
        
        ring_vbo = ctx.buffer(all_v.tobytes())
        ring_ibo = ctx.buffer(all_i.tobytes())
        ring_vao = ctx.vertex_array(
            prog_rings,
            [(ring_vbo, '3f 3f 4f 1f 1f', 'in_position', 'in_normal', 'in_color', 'in_scatter', 'in_asymmetry')],
            index_buffer=ring_ibo
        )
        ring_render_groups.append({
            'body_idx': bi,
            'vao': ring_vao,
            'num_indices': len(all_i),
        })

    INSTANCE_FLOATS = 16
    MAX_BODIES = 1000

    mesh_lo_verts, mesh_lo_idx = create_icosphere_mesh(subdivisions=1)
    mesh_hi_verts, mesh_hi_idx = create_icosphere_mesh(subdivisions=4)

    vbo_lo = ctx.buffer(mesh_lo_verts.tobytes())
    ibo_lo = ctx.buffer(mesh_lo_idx.tobytes())
    vbo_hi = ctx.buffer(mesh_hi_verts.tobytes())
    ibo_hi = ctx.buffer(mesh_hi_idx.tobytes())

    vbo_instances_lo = ctx.buffer(reserve=MAX_BODIES * INSTANCE_FLOATS * 4)
    vbo_instances_hi = ctx.buffer(reserve=MAX_BODIES * INSTANCE_FLOATS * 4)

    inst_fmt = '3f 3f 1f 1f 1f 3f 1f 2u 1u/i'
    inst_names = ('in_offset', 'in_color', 'in_radius', 'in_min_size', 'in_is_star', 'in_pole', 'in_oblateness', 'in_caster_mask', 'in_ring_mask')

    vao_lo = ctx.vertex_array(
        prog_spheres,
        [
            (vbo_lo, '3f 3f', 'in_position', 'in_normal'),
            (vbo_instances_lo, inst_fmt, *inst_names)
        ],
        index_buffer=ibo_lo
    )
    vao_hi = ctx.vertex_array(
        prog_spheres,
        [
            (vbo_hi, '3f 3f', 'in_position', 'in_normal'),
            (vbo_instances_hi, inst_fmt, *inst_names)
        ],
        index_buffer=ibo_hi
    )

    vao_atmo = ctx.vertex_array(
        prog_atmo,
        [(vbo_hi, '3f 12x', 'in_position')],
        index_buffer=ibo_hi
    )

    max_orbits = MAX_BODIES * 2
    orbit_ssbo = ctx.buffer(reserve=max_orbits * 160)
    orbit_ssbo.bind_to_storage_buffer(binding=0)
    vao_gpu_orbits = ctx.vertex_array(prog_gpu_orbits, [])

    UBO_SIZE = 3232
    scene_ubo = ctx.buffer(reserve=UBO_SIZE)
    scene_ubo.bind_to_uniform_block(1)
    ubo_staging = np.zeros(UBO_SIZE // 4, dtype=np.float32)
    ubo_casters_int_view = ubo_staging[38:39].view(np.int32)

    uniform_screen_height = prog_spheres['screen_height']
    uniform_fov_factor = prog_spheres['fov_factor']

    uniform_orbit_proj = prog_gpu_orbits['projection']
    uniform_orbit_view_rot = prog_gpu_orbits['view_rot']
    uniform_orbit_cam_pos = prog_gpu_orbits['u_cam_pos_double']
    uniform_orbit_far = prog_gpu_orbits['u_far']
    uniform_orbit_depth_C = prog_gpu_orbits['u_depth_C']

    uniform_ring_centers = prog_spheres['u_ring_center']
    uniform_ring_normals = prog_spheres['u_ring_normal']
    uniform_ring_params = prog_spheres['u_ring_params']
    uniform_num_ring_planes = prog_spheres['u_num_ring_planes']
    if 'u_ring_gradients' in prog_spheres:
        prog_spheres['u_ring_gradients'].value = 0

    star_idx = 0
    for i, b in enumerate(bodies_data):
        if b.get('type') == 'Star':
            star_idx = i
            break
    star_radius_au = visual_data[star_idx][3]

    body_radii = np.array([v[3] for v in visual_data], dtype='f4')
    body_colors = np.array([v[0:3] for v in visual_data], dtype='f4')

    if ring_render_groups:
        u_ring_host_pos = prog_rings['u_host_planet_pos']
        u_ring_host_radius = prog_rings['u_host_planet_radius']
        u_ring_host_pole_obl = prog_rings['u_host_planet_pole_obl']
        u_ring_camera_pos = prog_rings['u_camera_pos']
        u_ring_body_offset = prog_rings['u_body_offset']
        u_ring_clip_mode = prog_rings['u_clip_mode']
        u_ring_caster_mask_lo_uni = prog_rings['u_caster_mask_lo']
        u_ring_caster_mask_hi_uni = prog_rings['u_caster_mask_hi']

    if atmo_bodies:
        u_atmo_body_offset = prog_atmo['u_body_offset']
        if 'u_ring_gradients' in prog_atmo:
            prog_atmo['u_ring_gradients'].value = 0
        u_atmo_planet_radius = prog_atmo['u_planet_radius_km']
        u_atmo_atmo_radius = prog_atmo['u_atmo_radius_km']
        u_atmo_radius_au_uniform = prog_atmo['u_atmo_radius_au']
        u_atmo_au_to_km = prog_atmo['u_au_to_km']
        u_atmo_beta_rayleigh = prog_atmo['u_beta_rayleigh']
        u_atmo_h_rayleigh = prog_atmo['u_h_rayleigh']
        u_atmo_beta_mie = prog_atmo['u_beta_mie']
        u_atmo_h_mie = prog_atmo['u_h_mie']
        u_atmo_mie_g = prog_atmo['u_mie_g']
        u_atmo_beta_absorption = prog_atmo['u_beta_absorption']
        u_atmo_quality_uniform = prog_atmo['u_atmo_quality']
        u_atmo_sun_intensity = prog_atmo['u_sun_intensity']
        u_atmo_camera_pos = prog_atmo['u_camera_pos']
        u_atmo_num_samples = prog_atmo['u_num_samples']
        u_atmo_num_light_samples = prog_atmo['u_num_light_samples']
        u_atmo_pole_obl = prog_atmo['u_pole_obl']
        u_atmo_num_ring_planes = prog_atmo['u_num_ring_planes']
        u_atmo_ring_centers = prog_atmo['u_ring_center']
        u_atmo_ring_normals = prog_atmo['u_ring_normal']
        u_atmo_ring_params = prog_atmo['u_ring_params']
        u_atmo_caster_mask_lo_uni = prog_atmo['u_caster_mask_lo']
        u_atmo_caster_mask_hi_uni = prog_atmo['u_caster_mask_hi']
        u_atmo_ring_mask_uni = prog_atmo['u_ring_mask']

    G = 4.0 * math.pi**2 

    visual_arr = np.array(visual_data, dtype='f4')
    is_star_arr = np.zeros(num_bodies, dtype='f4')
    is_star_arr[star_idx] = 1.0
    non_star_mask = np.ones(num_bodies, dtype=bool)
    non_star_mask[star_idx] = False
    non_star_indices = np.where(non_star_mask)[0][:64]
    n_casters_fixed = len(non_star_indices)

    inst_data_lo = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
    inst_data_hi = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
    orbit_data_buf = np.zeros((max_orbits, 20), dtype='f8')
    subsys_pos_buf = np.zeros((num_bodies, 3), dtype='f8')
    subsys_vel_buf = np.zeros((num_bodies, 3), dtype='f8')
    subsys_mass_buf = np.zeros(num_bodies, dtype='f8')
    caster_data_buf = np.zeros((64, 4), dtype='f4')
    caster_poles_obl_buf = np.zeros((64, 4), dtype='f4')
    caster_colors_buf = np.zeros((64, 4), dtype='f4')
    ring_centers_buf = np.zeros((16, 3), dtype='f4')
    ring_normals_buf = np.zeros((16, 3), dtype='f4')
    ring_params_buf = np.zeros((16, 3), dtype='f4')
    all_instances = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
    cull_mask_lo = np.zeros(num_bodies, dtype=np.uint32)
    cull_mask_hi = np.zeros(num_bodies, dtype=np.uint32)
    cull_ring_mask = np.zeros(num_bodies, dtype=np.uint32)
    cull_ring_caster_lo = np.zeros(16, dtype=np.uint32)
    cull_ring_caster_hi = np.zeros(16, dtype=np.uint32)

    pos_snap = np.zeros((num_bodies, 3), dtype='f8')
    vel_snap = np.zeros((num_bodies, 3), dtype='f8')
    mass_snap = np.zeros(num_bodies, dtype='f8')
    parent_snap = np.zeros(num_bodies, dtype=np.int32)
    tree_indices_snap = np.zeros(num_bodies, dtype=np.int32)
    tree_depths_snap = np.zeros(num_bodies, dtype=np.int32)
    focused_mask = np.zeros(num_bodies, dtype=np.bool_)
    visual_colors_f8 = np.ascontiguousarray(visual_arr[:, 0:3], dtype='f8')
    cached_children_map = {}
    cached_hierarchy_ver = -1

    last_render_time = time.time()
    show_orbits = True
    orbit_fade_dir = 1.0
    orbit_fade_dir_idx = 0
    orbit_min_alpha = 0.0
    atmo_quality = 1
    now_dt = datetime.datetime.now()
    jump_date = [now_dt.year, now_dt.month, now_dt.day, now_dt.hour, now_dt.minute]
    scrub_index = [0]

    last_orbit_pos_snap = None
    last_cam_origin = None

    while not glfw.window_should_close(window):
        with shared_state["lock"]:
            if shared_state.get("rebuild_flag", False):
                for op in shared_state["crud_completed"]:
                    if op["action"] == "CREATE":
                        name = op.get("name", "New Body")
                        radius = op.get("radius", 6000.0)
                        mass = op.get("mass", 3e-6)
                        color = op.get("color", [1.0, 1.0, 1.0])
                        btype = op.get("type", "Moon")
                        parent_idx = op["parent_idx"]
                        pos = op["pos"]
                        vel = op["vel"]
                        
                        bodies_data.append({
                            "name": name,
                            "type": btype,
                            "color": color,
                            "r": radius / 696340.0,
                            "mass": mass
                        })
                        
                        pos_snap = np.vstack([pos_snap, pos])
                        vel_snap = np.vstack([vel_snap, vel])
                        mass_snap = np.append(mass_snap, mass)
                        parent_snap = np.append(parent_snap, parent_idx)
                        
                        tree_indices_snap = np.append(tree_indices_snap, 0)
                        tree_depths_snap = np.append(tree_depths_snap, 0)
                        
                        pos_snap_render = np.vstack([pos_snap_render, pos])
                        vel_snap_render = np.vstack([vel_snap_render, vel])
                        
                        r_au = radius / 1.496e8
                        body_radii = np.append(body_radii, r_au)
                        
                        new_inst = np.zeros(INSTANCE_FLOATS, dtype='f4')
                        num_bodies += 1
                        all_instances = np.vstack([all_instances, new_inst])
                        cull_mask_lo = np.append(cull_mask_lo, 0)
                        cull_mask_hi = np.append(cull_mask_hi, 0)
                        cull_ring_mask = np.append(cull_ring_mask, 0)
                        
                        subsys_pos_buf = np.vstack([subsys_pos_buf, pos])
                        subsys_vel_buf = np.vstack([subsys_vel_buf, vel])
                        subsys_mass_buf = np.append(subsys_mass_buf, mass)
                        
                        new_vis = np.array([color[0], color[1], color[2], r_au, 1.0, 0, 1, 0, 0], dtype='f4')
                        visual_arr = np.vstack([visual_arr, new_vis])
                        visual_colors_f8 = np.vstack([visual_colors_f8, np.array(color, dtype='f8')])
                        is_star_arr = np.append(is_star_arr, 0.0)
                        
                        inst_data_lo = np.vstack([inst_data_lo, np.zeros(INSTANCE_FLOATS, dtype='f4')])
                        inst_data_hi = np.vstack([inst_data_hi, np.zeros(INSTANCE_FLOATS, dtype='f4')])
                        focused_mask = np.append(focused_mask, False)
                        
                        num_bodies += 1
                        
                        non_star_indices = np.array([i for i in range(num_bodies) if i != star_idx][:64], dtype=np.int32)
                        n_casters_fixed = len(non_star_indices)

                    elif op["action"] == "DELETE":
                        idx = op["idx"]
                        bodies_data.pop(idx)
                        
                        pos_snap = np.delete(pos_snap, idx, axis=0)
                        vel_snap = np.delete(vel_snap, idx, axis=0)
                        mass_snap = np.delete(mass_snap, idx)
                        
                        ps = np.delete(parent_snap, idx)
                        ps[ps == idx] = 0
                        ps[ps > idx] -= 1
                        parent_snap = ps
                        
                        tree_indices_snap = np.delete(tree_indices_snap, idx)
                        tree_depths_snap = np.delete(tree_depths_snap, idx)
                        pos_snap_render = np.delete(pos_snap_render, idx, axis=0)
                        vel_snap_render = np.delete(vel_snap_render, idx, axis=0)
                        body_radii = np.delete(body_radii, idx)
                        num_bodies -= 1
                        all_instances = np.delete(all_instances, idx, axis=0)
                        cull_mask_lo = np.delete(cull_mask_lo, idx)
                        cull_mask_hi = np.delete(cull_mask_hi, idx)
                        cull_ring_mask = np.delete(cull_ring_mask, idx)
                        subsys_pos_buf = np.delete(subsys_pos_buf, idx, axis=0)
                        subsys_vel_buf = np.delete(subsys_vel_buf, idx, axis=0)
                        subsys_mass_buf = np.delete(subsys_mass_buf, idx)
                        visual_arr = np.delete(visual_arr, idx, axis=0)
                        visual_colors_f8 = np.delete(visual_colors_f8, idx, axis=0)
                        is_star_arr = np.delete(is_star_arr, idx)
                        inst_data_lo = np.delete(inst_data_lo, idx, axis=0)
                        inst_data_hi = np.delete(inst_data_hi, idx, axis=0)
                        focused_mask = np.delete(focused_mask, idx)
                        
                        num_bodies -= 1
                        
                        if star_idx == idx:
                            star_idx = 0
                        elif star_idx > idx:
                            star_idx -= 1
                            
                        non_star_indices = np.array([i for i in range(num_bodies) if i != star_idx][:64], dtype=np.int32)
                        n_casters_fixed = len(non_star_indices)
                        
                        if camera["tracking_idx"] == idx:
                            camera["tracking_idx"] = None
                        elif camera["tracking_idx"] is not None and camera["tracking_idx"] > idx:
                            camera["tracking_idx"] -= 1
                        
                        if camera["inspected_idx"] == idx:
                            camera["inspected_idx"] = None
                            camera["inspect_bary"] = False
                        elif camera["inspected_idx"] is not None and camera["inspected_idx"] > idx:
                            camera["inspected_idx"] -= 1
                        
                        atmo_bodies = [a for a in atmo_bodies if a['body_idx'] != idx]
                        for a in atmo_bodies:
                            if a['body_idx'] > idx: a['body_idx'] -= 1
                            
                        ring_precomputed = [r for r in ring_precomputed if r['body_idx'] != idx]
                        for r in ring_precomputed:
                            if r['body_idx'] > idx: r['body_idx'] -= 1
                            
                        ring_render_groups = [g for g in ring_render_groups if g['body_idx'] != idx]
                        for g in ring_render_groups:
                            if g['body_idx'] > idx: g['body_idx'] -= 1
                            
                shared_state["crud_completed"].clear()
                shared_state["rebuild_flag"] = False

        now = time.time()
        dt_render = min(now - last_render_time, 0.1)
        last_render_time = now
        
        glfw.poll_events()
        
        impl.process_inputs()
        imgui.new_frame()
        
        if fb_width <= 0 or fb_height <= 0:
            imgui.end_frame()
            continue
        
        ctx.viewport = (0, 0, fb_width, fb_height)
        ctx.clear(0.02, 0.02, 0.03, 1.0) 

        with shared_state["lock"]:
            if len(pos_snap) == len(shared_state["pos"]):
                np.copyto(pos_snap, shared_state["pos"])
                np.copyto(vel_snap, shared_state["vel"])
                np.copyto(mass_snap, shared_state["mass"])
                np.copyto(parent_snap, shared_state["parent_indices"])
            if len(tree_indices_snap) == len(shared_state["tree_indices"]):
                np.copyto(tree_indices_snap, shared_state["tree_indices"])
                np.copyto(tree_depths_snap, shared_state["tree_depths"])
            hierarchy_ver = shared_state["hierarchy_version"]
            
            tl_active = shared_state["timeline_active"]
            tl_prog = shared_state["timeline_progress"]
            current_sim_t = shared_state["t"]
            tl_times = shared_state["timeline_times"]
            tl_pos_buf = shared_state["timeline_pos"]
            tl_vel_buf = shared_state["timeline_vel"]

        is_scrubbing = tl_active and tl_prog >= 1.0

        if is_scrubbing and len(tl_times) > 0:
            scrub_idx = min(scrub_index[0], len(tl_times) - 1)
            display_t = tl_times[scrub_idx]
            pos_snap_render = tl_pos_buf[scrub_idx].astype('f8')
            vel_snap_render = tl_vel_buf[scrub_idx].astype('f8')
        else:
            display_t = current_sim_t
            pos_snap_render = pos_snap
            vel_snap_render = vel_snap

        cur_y, cur_m, cur_d, cur_h, cur_mn = format_sim_time(display_t)

        lerp_factor = 1.0 - math.exp(-15.0 * dt_render)
        camera["distance_actual"] += (camera["distance"] - camera["distance_actual"]) * lerp_factor
        camera["yaw_actual"] += (camera["yaw"] - camera["yaw_actual"]) * lerp_factor
        camera["pitch_actual"] += (camera["pitch"] - camera["pitch_actual"]) * lerp_factor
        camera["target_offset"] *= math.exp(-10.0 * dt_render)

        compute_barycenters(pos_snap_render, vel_snap_render, mass_snap, parent_snap,
                            subsys_pos_buf, subsys_vel_buf, subsys_mass_buf)

        if camera["tracking_idx"] is not None:
            track_idx = camera["tracking_idx"]
            if camera["tracking_mode"] == "barycenter":
                base_pos = subsys_pos_buf[track_idx].copy()
            else:
                base_pos = pos_snap_render[track_idx].copy()
            camera["target"] = base_pos.copy()
        else:
            base_pos = np.array(camera["target"], dtype='f8')
            
        cam_origin = base_pos + camera["target_offset"] + camera["pan_offset"]

        if hierarchy_ver != cached_hierarchy_ver:
            cached_children_map = {}
            for i in range(num_bodies):
                pi = int(parent_snap[i])
                if pi >= 0:
                    if pi not in cached_children_map:
                        cached_children_map[pi] = []
                    cached_children_map[pi].append(i)
            cached_hierarchy_ver = hierarchy_ver
        children_map = cached_children_map

        focused_root = -1
        if camera["tracking_idx"] is not None:
            body_idx = camera["tracking_idx"]
            while True:
                p = int(parent_snap[body_idx])
                if p == -1 or p == star_idx:
                    focused_root = body_idx
                    break
                body_idx = p
        
        focused_mask[:] = False
        if focused_root >= 0:
            stack = [focused_root]
            while stack:
                node = stack.pop()
                focused_mask[node] = True
                if node in children_map:
                    stack.extend(children_map[node])
        
        pos_rel_all = (pos_snap_render - cam_origin).astype('f4')
        
        yaw_rad, pitch_rad = math.radians(camera["yaw_actual"]), math.radians(camera["pitch_actual"])
        cam_pos_f8 = np.array([-math.cos(yaw_rad) * math.cos(pitch_rad) * camera["distance_actual"],
                               -math.sin(pitch_rad) * camera["distance_actual"],
                               -math.sin(yaw_rad) * math.cos(pitch_rad) * camera["distance_actual"]], dtype='f8')
                               
        cam_pos = cam_pos_f8.astype('f4')
        view = matrix44.create_look_at(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], dtype='f4')
        near = max(camera["distance_actual"] * 0.0001, 1e-9)
        max_body_dist_sq = float(np.max(np.sum(pos_rel_all * pos_rel_all, axis=1)))
        far = max(camera["distance_actual"] + math.sqrt(max_body_dist_sq) * 2.0 + 1.0, 1.0)
        depth_C = 1.0 / max(near, 1e-12)
        aspect_ratio = fb_width / max(fb_height, 1)
        projection = matrix44.create_perspective_projection_matrix(camera["fov"], aspect_ratio, near, far, dtype='f4')

        n_ring_planes = 0
        ring_centers_buf[:] = 0
        ring_normals_buf[:] = 0
        ring_params_buf[:] = 0
        
        ring_idx_by_body = {r['body_idx']: i for i, r in enumerate(ring_precomputed)}
        for ring in ring_precomputed:
            if n_ring_planes >= 16:
                break
            bi = ring['body_idx']
            ring_centers_buf[n_ring_planes] = pos_rel_all[bi]
            ring_normals_buf[n_ring_planes] = ring['normal']
            ring_params_buf[n_ring_planes, 0] = ring['inner_r']
            ring_params_buf[n_ring_planes, 1] = ring['outer_r']
            ring_params_buf[n_ring_planes, 2] = ring['opacity']
            n_ring_planes += 1

        caster_indices = non_star_indices
        n_casters_fixed = min(len(caster_indices), 64)
        
        compute_culling_masks(
            pos_rel_all, body_radii, star_idx,
            n_casters_fixed, caster_indices[:n_casters_fixed],
            n_ring_planes, ring_centers_buf, ring_normals_buf, ring_params_buf[:, 1],
            num_bodies,
            cull_mask_lo, cull_mask_hi, cull_ring_mask, cull_ring_caster_lo, cull_ring_caster_hi
        )
        
        vp_matrix = np.asarray(view, dtype=np.float32) @ np.asarray(projection, dtype=np.float32)
        frustum_planes = extract_frustum_planes(vp_matrix)
        visible_mask = compute_frustum_culling(pos_rel_all, body_radii, frustum_planes, num_bodies)

        all_instances[:, 0:3] = pos_rel_all
        all_instances[:, 3:8] = visual_arr[:, 0:5]
        all_instances[:, 8] = is_star_arr
        all_instances[:, 9:12] = visual_arr[:, 5:8]
        all_instances[:, 12] = visual_arr[:, 8]
        all_instances[:, 13] = cull_mask_lo.view(np.float32)
        all_instances[:, 14] = cull_mask_hi.view(np.float32)
        all_instances[:, 15] = cull_ring_mask.view(np.float32)
        
        final_hi_mask = focused_mask & visible_mask
        final_lo_mask = (~focused_mask) & visible_mask
        
        n_hi = int(np.sum(final_hi_mask))
        n_lo = int(np.sum(final_lo_mask))
        if n_hi > 0:
            inst_data_hi[:n_hi] = all_instances[final_hi_mask]
        if n_lo > 0:
            inst_data_lo[:n_lo] = all_instances[final_lo_mask]

        if not show_orbits:
            n_orbits = 0
        else:
            recompute_orbits = True
            if last_orbit_pos_snap is not None and time_ctrl.get("paused"):
                if not camera.get("left_dragging") and not camera.get("right_dragging") and camera.get("tracking_idx") is None:
                    if np.array_equal(pos_snap_render, last_orbit_pos_snap) and np.array_equal(cam_origin, last_cam_origin):
                        recompute_orbits = False
            
            if recompute_orbits:
                n_orbits = compute_all_orbits_batch(
                    pos_snap_render, vel_snap_render, mass_snap, parent_snap,
                    subsys_pos_buf, subsys_vel_buf, subsys_mass_buf,
                    visual_colors_f8, cam_origin, G, max_orbits, orbit_data_buf)
                last_orbit_pos_snap = pos_snap_render.copy()
                last_cam_origin = cam_origin.copy()

        if n_lo > 0:
            vbo_instances_lo.write(inst_data_lo[:n_lo])
        if n_hi > 0:
            vbo_instances_hi.write(inst_data_hi[:n_hi])
        
        if n_orbits > 0:
            orbit_ssbo.write(orbit_data_buf[:n_orbits])
            
        star_cam_rel = (pos_snap[star_idx] - cam_origin).astype('f4')
        
        caster_data_buf[:n_casters_fixed, 0:3] = pos_rel_all[caster_indices[:n_casters_fixed]]
        caster_data_buf[:n_casters_fixed, 3] = body_radii[caster_indices[:n_casters_fixed]]
        if n_casters_fixed < 64:
            caster_data_buf[n_casters_fixed:] = 0
            
        caster_poles_obl_buf[:n_casters_fixed, 0:3] = all_instances[caster_indices[:n_casters_fixed], 9:12]
        caster_poles_obl_buf[:n_casters_fixed, 3] = all_instances[caster_indices[:n_casters_fixed], 12]
        if n_casters_fixed < 64:
            caster_poles_obl_buf[n_casters_fixed:] = 0
            
        caster_colors_buf[:n_casters_fixed, 0:3] = body_colors[caster_indices[:n_casters_fixed]]
        caster_colors_buf[:n_casters_fixed, 3] = 1.0
        if n_casters_fixed < 64:
            caster_colors_buf[n_casters_fixed:] = 0

        ubo_staging[0:16] = projection.ravel()
        ubo_staging[16:32] = view.ravel()
        ubo_staging[32:35] = star_cam_rel
        ubo_staging[35] = star_radius_au
        ubo_staging[36] = far
        ubo_staging[37] = depth_C
        ubo_casters_int_view[0] = n_casters_fixed
        ubo_staging[40:296] = caster_data_buf.ravel()
        ubo_staging[296:552] = caster_poles_obl_buf.ravel()
        ubo_staging[552:808] = caster_colors_buf.ravel()
        scene_ubo.write(ubo_staging)
        
        fov_factor = 1.0 / math.tan(math.radians(camera["fov"] / 2.0))
        uniform_screen_height.value = fb_height
        uniform_fov_factor.value = fov_factor
        uniform_num_ring_planes.value = n_ring_planes
        if n_ring_planes > 0:
            uniform_ring_centers.write(ring_centers_buf)
            uniform_ring_normals.write(ring_normals_buf)
            uniform_ring_params.write(ring_params_buf)
        
        view_rot = view.copy()
        view_rot[3, 0:3] = 0.0

        uniform_orbit_proj.write(projection)
        uniform_orbit_view_rot.write(view_rot)
        
        cam_pos_dvec4 = np.array([cam_pos_f8[0], cam_pos_f8[1], cam_pos_f8[2], 0.0], dtype='f8')
        uniform_orbit_cam_pos.write(cam_pos_dvec4)
        
        uniform_orbit_far.value = far
        uniform_orbit_depth_C.value = depth_C

        ring_gradient_tex.use(location=0)
        ctx.enable(moderngl.CULL_FACE)
        if n_lo > 0:
            vao_lo.render(moderngl.TRIANGLES, instances=n_lo)
        if n_hi > 0:
            vao_hi.render(moderngl.TRIANGLES, instances=n_hi)
        ctx.disable(moderngl.CULL_FACE)


        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        if show_orbits and n_orbits > 0:
            prog_gpu_orbits['u_fade_dir'].value = orbit_fade_dir
            prog_gpu_orbits['u_min_alpha'].value = orbit_min_alpha
            prog_gpu_orbits['u_cam_pos_double'].value = (cam_pos[0], cam_pos[1], cam_pos[2], 1.0)
            vao_gpu_orbits.render(moderngl.LINE_STRIP, vertices=1001, instances=n_orbits)

        if ring_render_groups:
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            u_ring_camera_pos.write(cam_pos)
            ctx.depth_mask = False
            u_ring_clip_mode.value = 1
            

            
            for group in ring_render_groups:
                bi = group['body_idx']
                body_pos_rel = pos_rel_all[bi]
                u_ring_body_offset.write(body_pos_rel)
                u_ring_host_pos.write(body_pos_rel)
                u_ring_host_radius.value = float(body_radii[bi])
                u_ring_host_pole_obl.value = (
                    float(all_instances[bi, 9]),
                    float(all_instances[bi, 10]),
                    float(all_instances[bi, 11]),
                    float(all_instances[bi, 12])
                )
                k = ring_idx_by_body.get(bi)
                if k is not None:
                    u_ring_caster_mask_lo_uni.value = int(cull_ring_caster_lo[k])
                    u_ring_caster_mask_hi_uni.value = int(cull_ring_caster_hi[k])
                group['vao'].render(moderngl.TRIANGLES, vertices=group['num_indices'])
            ctx.depth_mask = True

        ctx.disable(moderngl.BLEND)

        if atmo_quality > 0 and atmo_bodies:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.CULL_FACE)
            ctx.depth_mask = False

            u_atmo_quality_uniform.value = atmo_quality
            u_atmo_camera_pos.write(cam_pos)
            u_atmo_au_to_km.value = AU_TO_KM
        
            u_atmo_num_ring_planes.value = n_ring_planes
            if n_ring_planes > 0:
                u_atmo_ring_centers.write(ring_centers_buf)
                u_atmo_ring_normals.write(ring_normals_buf)
                u_atmo_ring_params.write(ring_params_buf)

            # Precalculate distances for sorting without closure
            atmo_dists = []
            for atmo in atmo_bodies:
                dx = pos_rel_all[atmo['body_idx'], 0] - cam_pos[0]
                dy = pos_rel_all[atmo['body_idx'], 1] - cam_pos[1]
                dz = pos_rel_all[atmo['body_idx'], 2] - cam_pos[2]
                atmo_dists.append((dx*dx + dy*dy + dz*dz, atmo))
                
            sorted_atmos = sorted(atmo_dists, key=lambda x: x[0], reverse=True)

            for sq_dist, atmo in sorted_atmos:
                bi = atmo['body_idx']
                body_pos_rel = pos_rel_all[bi]

                dist_to_body = math.sqrt(sq_dist)
                apparent_px = (atmo['atmo_radius_au'] / max(dist_to_body, 1e-12)) * fb_height * fov_factor
                if apparent_px < 2.0:
                    continue

                if apparent_px < 50:
                    n_samples, n_light = 8, 4
                elif apparent_px < 200:
                    n_samples, n_light = 16, 6
                else:
                    n_samples, n_light = 32, 8

                scaled_intensity = atmo['intensity']

                u_atmo_body_offset.write(body_pos_rel.astype('f4'))
                u_atmo_radius_au_uniform.value = float(atmo['atmo_radius_au'])
                u_atmo_planet_radius.value = float(atmo['planet_radius_km'])
                u_atmo_atmo_radius.value = float(atmo['atmo_radius_km'])
                u_atmo_beta_rayleigh.write(atmo['beta_rayleigh'])
                u_atmo_h_rayleigh.value = atmo['h_rayleigh']
                u_atmo_beta_mie.value = atmo['beta_mie']
                u_atmo_h_mie.value = atmo['h_mie']
                u_atmo_mie_g.value = atmo['mie_g']
                u_atmo_beta_absorption.write(atmo['beta_absorption'])
                u_atmo_sun_intensity.value = scaled_intensity
                u_atmo_num_samples.value = n_samples
                u_atmo_num_light_samples.value = n_light
                u_atmo_pole_obl.value = (
                    float(all_instances[bi, 9]),
                    float(all_instances[bi, 10]),
                    float(all_instances[bi, 11]),
                    float(all_instances[bi, 12])
                )
                u_atmo_caster_mask_lo_uni.value = int(cull_mask_lo[bi])
                u_atmo_caster_mask_hi_uni.value = int(cull_mask_hi[bi])
                u_atmo_ring_mask_uni.value = int(cull_ring_mask[bi])

                vao_atmo.render(moderngl.TRIANGLES)

            ctx.depth_mask = True
            ctx.disable(moderngl.CULL_FACE)
            ctx.enable(moderngl.DEPTH_TEST)
            
        if ring_render_groups:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            u_ring_camera_pos.write(cam_pos)
            ctx.depth_mask = False
            u_ring_clip_mode.value = 2
            

            
            for group in ring_render_groups:
                bi = group['body_idx']
                body_pos_rel = pos_rel_all[bi]
                u_ring_body_offset.write(body_pos_rel)
                u_ring_host_pos.write(body_pos_rel)
                u_ring_host_radius.value = float(body_radii[bi])
                u_ring_host_pole_obl.value = (
                    float(all_instances[bi, 9]),
                    float(all_instances[bi, 10]),
                    float(all_instances[bi, 11]),
                    float(all_instances[bi, 12])
                )
                k = ring_idx_by_body.get(bi)
                if k is not None:
                    u_ring_caster_mask_lo_uni.value = int(cull_ring_caster_lo[k])
                    u_ring_caster_mask_hi_uni.value = int(cull_ring_caster_hi[k])
                group['vao'].render(moderngl.TRIANGLES, vertices=group['num_indices'])
            ctx.depth_mask = True
            
        ctx.disable(moderngl.BLEND)
        imgui.set_next_window_position(fb_width - 250, 20, imgui.ONCE)
        imgui.set_next_window_size(230, min(320, fb_height - 40), imgui.ONCE)
        imgui.begin("Simulation Controls")
        
        imgui.text("Current Date:")
        imgui.text(f"{cur_y:04d}-{cur_m:02d}-{cur_d:02d} {cur_h:02d}:{cur_mn:02d} GMT")
        
        imgui.separator()
        imgui.text("Time Controls")
        _, time_ctrl["paused"] = imgui.checkbox("Paused", time_ctrl["paused"])
        imgui.text(f"Speed: {format_time_speed(time_ctrl['multiplier'])}")
        _, time_ctrl["multiplier"] = imgui.slider_float("##speed", time_ctrl["multiplier"], 1.0, 1e9, "", flags=imgui.SLIDER_FLAGS_LOGARITHMIC)
        if imgui.button("Reset Speed"): time_ctrl["multiplier"] = 1.0
                
        if shared_state.get("syncing", False):
            imgui.text("Synchronizing Physics...")
            imgui.progress_bar(shared_state.get("sync_progress", 0.0), size=(-1, 0.0))
            
        elif tl_active and tl_prog < 1.0:
            imgui.text("Rendering Timeline...")
            
            rate = shared_state.get("timeline_rate", 0.0)
            if rate > 0.0:
                imgui.text(f"Render Pace: {format_time_speed(rate)}")
                
            imgui.progress_bar(tl_prog)
            if imgui.button("Cancel"):
                time_ctrl["cancel_render"] = True
        elif is_scrubbing:
            imgui.separator()
            imgui.text("Timeline Navigation")
            max_idx = max(0, len(tl_times) - 1)
            
            if time_ctrl.get("snap_to_end", False):
                scrub_index[0] = max_idx
                time_ctrl["snap_to_end"] = False
            
            if time_ctrl["timeline_playing"]:
                if imgui.button("Pause Playback"):
                    time_ctrl["timeline_playing"] = False
                    
                time_ctrl["timeline_scrub_float"] += time_ctrl["timeline_speed"] * dt_render
                steps_to_add = int(time_ctrl["timeline_scrub_float"])
                if steps_to_add > 0:
                    scrub_index[0] += steps_to_add
                    time_ctrl["timeline_scrub_float"] -= steps_to_add
                    if scrub_index[0] > max_idx:
                        scrub_index[0] = 0
            else:
                if imgui.button("Play Timeline"):
                    time_ctrl["timeline_playing"] = True
                    time_ctrl["timeline_scrub_float"] = 0.0
                    if scrub_index[0] >= max_idx:
                        scrub_index[0] = 0
                        
            imgui.same_line()
            _, time_ctrl["timeline_speed"] = imgui.slider_float("Speed##tl", time_ctrl["timeline_speed"], 1.0, 100.0, "%.1fx")
            
            changed, scrub_index[0] = imgui.slider_int("##Scrub", scrub_index[0], 0, max_idx, "")
            
            if imgui.button("Resume Here"):
                time_ctrl["sync_t"] = tl_times[scrub_index[0]]
                time_ctrl["paused"] = False
                time_ctrl["timeline_playing"] = False
            imgui.same_line()
            if imgui.button("Cancel"):
                with shared_state["lock"]:
                    shared_state["timeline_active"] = False
                time_ctrl["timeline_playing"] = False
                    
        else:
            imgui.separator()
            imgui.text("Jump in Time")
            _, jump_date[0] = imgui.input_int("Year", jump_date[0])
            _, jump_date[1] = imgui.input_int("Month", jump_date[1])
            _, jump_date[2] = imgui.input_int("Day", jump_date[2])
            
            imgui.text("Time (HH:MM)")
            imgui.push_item_width(40)
            _, jump_date[3] = imgui.input_int("##Hour", jump_date[3], step=0)
            imgui.same_line()
            imgui.text(":")
            imgui.same_line()
            _, jump_date[4] = imgui.input_int("##Minute", jump_date[4], step=0)
            imgui.pop_item_width()
            
            jump_date[1] = max(1, min(12, jump_date[1]))
            jump_date[2] = max(1, min(31, jump_date[2]))
            jump_date[3] = max(0, min(23, jump_date[3]))
            jump_date[4] = max(0, min(59, jump_date[4]))
            
            if imgui.button("Render Timeline"):
                target_t = sim_time_from_date(jump_date[0], jump_date[1], jump_date[2], jump_date[3], jump_date[4])
                if target_t < shared_state["t"]:
                    jump_date[0], jump_date[1], jump_date[2] = cur_y, cur_m, cur_d
                    jump_date[3], jump_date[4] = cur_h, cur_mn
                else:
                    time_ctrl["target_t"] = target_t
                    time_ctrl["render_timeline"] = True
        
        imgui.separator()
        imgui.text("Visual Settings")
        _, show_orbits = imgui.checkbox("Show Orbits", show_orbits)
        _, orbit_fade_dir_idx = imgui.combo("Orbit Fade", orbit_fade_dir_idx, ["Bright Behind", "Bright Ahead"])
        orbit_fade_dir = 1.0 if orbit_fade_dir_idx == 0 else -1.0
        _, orbit_min_alpha = imgui.slider_float("Min Alpha", orbit_min_alpha, 0.0, 1.0, "%.2f")
        
        imgui.spacing()
        _, atmo_quality = imgui.combo("##atmo", atmo_quality, ["Off", "Low (2D Shadows)", "High (Volumetric)"])
        imgui.same_line()
        imgui.text("Atmosphere Quality")
        imgui.text(f"FOV: {camera['fov']:.1f} deg")
        if imgui.button("Reset FOV"): camera["fov"] = 45.0
        imgui.end()

        imgui.set_next_window_position(fb_width - 250, 360, imgui.ONCE)
        imgui.set_next_window_size(230, min(600, fb_height - 380), imgui.ONCE)
        imgui.begin("System Hierarchy")
        
        for k in range(len(tree_indices_snap)):
            idx = int(tree_indices_snap[k])
            depth = int(tree_depths_snap[k])
            indent = depth * 15
            if indent > 0: imgui.indent(indent)
            is_inspected = (camera["inspected_idx"] == idx)
            label = f"{bodies_data[idx]['name']}"
            if camera["tracking_idx"] == idx:
                label += " *"
                
            avail_w = imgui.get_content_region_available()[0]
            clicked = imgui.selectable(label, is_inspected, 0, max(10.0, avail_w - 65.0))[0]
            
            rect_min_y = imgui.get_item_rect_min()[1]
            rect_max_y = imgui.get_item_rect_max()[1]
            rect_min_x = imgui.get_window_position()[0]
            rect_max_x = rect_min_x + imgui.get_window_width()
            mouse_x, mouse_y = imgui.get_io().mouse_pos
            
            # Do not use imgui.is_window_hovered() here because clicking the button makes the button active,
            # which causes is_window_hovered() to become false mid-click, making the button vanish and cancelling the click!
            show_buttons = (rect_min_y <= mouse_y <= rect_max_y) and (rect_min_x <= mouse_x <= rect_max_x)
                
            if clicked:
                if camera["inspected_idx"] == idx:
                    if not camera["inspect_bary"]:
                        camera["inspect_bary"] = True
                    else:
                        camera["inspected_idx"] = None
                        camera["inspect_bary"] = False
                else:
                    camera["inspected_idx"] = idx
                    camera["inspect_bary"] = False
            
            if show_buttons:
                imgui.same_line()
                imgui.set_cursor_pos_x(imgui.get_window_width() - 65)
                # Ensure it perfectly matches line height and text is centered
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                
                current_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos_y(current_y - 2)
                if imgui.button(f"+##add_{idx}", 18, 17):
                    camera["add_mode"] = True
                    camera["add_data"] = {
                        "name": f"Moon of {bodies_data[idx]['name']}",
                        "mass": 3e-6,
                        "radius": 1737.0,
                        "color": [0.7, 0.7, 0.7],
                        "a": 0.0025,
                        "e": 0.0, "inc": 0.0, "Omega": 0.0, "omega": 0.0, "M": 0.0,
                        "frame": 0
                    }
                
                if idx > 0:
                    imgui.same_line()
                    imgui.set_cursor_pos_y(current_y - 2)
                    imgui.push_style_color(imgui.COLOR_BUTTON, 0.6, 0.1, 0.1)
                    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.8, 0.2, 0.2)
                    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.9, 0.3, 0.3)
                    if imgui.button(f"-##del_{idx}", 18, 17):
                        with shared_state["lock"]:
                            shared_state["crud_queue"].append({
                                "action": "DELETE",
                                "idx": idx
                            })
                        camera["inspected_idx"] = None
                        camera["inspect_bary"] = False
                    imgui.pop_style_color(3)
                imgui.pop_style_var(1)
                    
            if indent > 0: imgui.unindent(indent)
        imgui.end()

        insp_idx = camera["inspected_idx"]
        if insp_idx is not None and 0 <= insp_idx < num_bodies:
            inspect_bary = camera["inspect_bary"]
            body_info = bodies_data[insp_idx]
            body_name = body_info['name']
            parent_idx = int(parent_snap[insp_idx])
            
            if inspect_bary:
                win_title = f"{body_name} System Barycenter###inspector"
            else:
                win_title = f"{body_name}###inspector"
            
            imgui.set_next_window_position(20, 20, imgui.ONCE)
            imgui.set_next_window_size(280, min(580, fb_height - 40), imgui.ONCE)
            expanded, opened = imgui.begin(win_title, True)
            if not opened:
                camera["inspected_idx"] = None
                camera["inspect_bary"] = False
            elif expanded:
                if inspect_bary:
                    imgui.text_colored("Barycenter Mode", 0.6, 0.9, 1.0)
                else:
                    obj_type = body_info.get('type', 'Unknown')
                    imgui.text_colored(f"Type: {obj_type}", 0.7, 0.7, 0.7)
                    imgui.same_line(spacing=15)
                    changed, camera["edit_mode"] = imgui.checkbox("Edit Mode", camera["edit_mode"])
                    if changed and camera["edit_mode"]:
                        time_ctrl["multiplier"] = 0.0
                        camera["edit_data"] = {
                            "mass": float(mass_snap[insp_idx]),
                            "radius": float(body_info.get('r', 0.0) * 696340.0),
                            "a": 0.0, "e": 0.0, "inc": 0.0, "Omega": 0.0, "omega": 0.0, "M": 0.0,
                            "init_orbit": True
                        }
                
                imgui.separator()
                imgui.text_colored("Physical Properties", 1.0, 0.85, 0.4)
                
                body_mass = mass_snap[insp_idx]
                body_r_km = body_info.get('r', 0.0) * 696340.0
                
                if camera["edit_mode"] and not inspect_bary:
                    _, camera["edit_data"]["mass"] = imgui.input_double("Mass (M_sun)", camera["edit_data"]["mass"], format="%e")
                    _, camera["edit_data"]["radius"] = imgui.input_double("Radius (km)", camera["edit_data"]["radius"], format="%.1f")
                elif inspect_bary:
                    bary_mass = subsys_mass_buf[insp_idx]
                    if bary_mass > 1e-4:
                        imgui.text(f"  System Mass:  {bary_mass:.6e} M_sun")
                    else:
                        m_earth = bary_mass / 3.003e-6
                        imgui.text(f"  System Mass:  {m_earth:.4f} M_earth")
                else:
                    if body_mass > 1e-4:
                        imgui.text(f"  Mass:    {body_mass:.6e} M_sun")
                    elif body_mass > 1e-10:
                        m_earth = body_mass / 3.003e-6
                        imgui.text(f"  Mass:    {m_earth:.6f} M_earth")
                    else:
                        m_earth = body_mass / 3.003e-6
                        imgui.text(f"  Mass:    {m_earth:.4e} M_earth")
                    if body_r_km > 100:
                        imgui.text(f"  Radius:  {body_r_km:,.0f} km")
                    elif body_r_km > 0.1:
                        imgui.text(f"  Radius:  {body_r_km:.1f} km")
                
                if parent_idx >= 0:
                    imgui.separator()
                    imgui.text_colored("Orbital Elements", 1.0, 0.85, 0.4)
                    
                    parent_name = bodies_data[parent_idx]['name']
                    imgui.text_colored(f"  (around {parent_name})", 0.5, 0.5, 0.5)
                    
                    if inspect_bary:
                        bp = subsys_pos_buf[insp_idx]
                        bv = subsys_vel_buf[insp_idx]
                        pp = pos_snap_render[parent_idx]
                        pv = vel_snap_render[parent_idx]
                        rel_r = bp - pp
                        rel_v = bv - pv
                        orb_mass = subsys_mass_buf[insp_idx]
                    else:
                        rel_r = pos_snap_render[insp_idx] - pos_snap_render[parent_idx]
                        rel_v = vel_snap_render[insp_idx] - vel_snap_render[parent_idx]
                        orb_mass = body_mass
                    
                    ecl_rx, ecl_ry, ecl_rz = rel_r[0], -rel_r[2], rel_r[1]
                    ecl_vx, ecl_vy, ecl_vz = rel_v[0], -rel_v[2], rel_v[1]
                    
                    camera.setdefault("inspector_frame", 0)
                    changed_frame, camera["inspector_frame"] = imgui.combo("Reference Frame", camera["inspector_frame"], ["Ecliptic", "Equatorial"])
                    if changed_frame and camera["edit_mode"]:
                        camera["edit_data"]["init_orbit"] = True
                        
                    if camera["inspector_frame"] == 1:
                        pole_render = visual_arr[parent_idx, 5:8]
                        pole_ecl = np.array([pole_render[0], -pole_render[2], pole_render[1]])
                        c_pos, c_vel = rotate_ecliptic_to_equatorial(
                            np.array([ecl_rx, ecl_ry, ecl_rz]), 
                            np.array([ecl_vx, ecl_vy, ecl_vz]), 
                            pole_ecl
                        )
                        ecl_rx, ecl_ry, ecl_rz = c_pos
                        ecl_vx, ecl_vy, ecl_vz = c_vel
                    
                    mu = G * (mass_snap[parent_idx] + orb_mass)
                    
                    oe_a, oe_e, oe_inc, oe_Omega, oe_omega, oe_nu, oe_M, oe_P = \
                        compute_keplerian_elements(ecl_rx, ecl_ry, ecl_rz,
                                                   ecl_vx, ecl_vy, ecl_vz, mu)
                    
                    dist_au = math.sqrt(rel_r[0]**2 + rel_r[1]**2 + rel_r[2]**2)
                    vel_au_yr = math.sqrt(rel_v[0]**2 + rel_v[1]**2 + rel_v[2]**2)
                    vel_km_s = vel_au_yr * 4.7405
                    
                    if camera["edit_mode"] and camera["edit_data"].get("init_orbit"):
                        camera["edit_data"]["a"] = float(oe_a)
                        camera["edit_data"]["e"] = float(oe_e)
                        camera["edit_data"]["inc"] = float(oe_inc)
                        camera["edit_data"]["Omega"] = float(oe_Omega)
                        camera["edit_data"]["omega"] = float(oe_omega)
                        camera["edit_data"]["M"] = float(oe_M)
                        camera["edit_data"]["init_orbit"] = False
                    
                    if camera["edit_mode"] and not inspect_bary:
                        imgui.text_colored("Edit Orbit", 0.5, 0.8, 1.0)
                        _, camera["edit_data"]["a"] = imgui.input_double("Semi-Major Axis (AU)", camera["edit_data"]["a"], format="%.6f")
                        _, camera["edit_data"]["e"] = imgui.input_double("Eccentricity", camera["edit_data"]["e"], format="%.6f")
                        _, camera["edit_data"]["inc"] = imgui.input_double("Inclination (deg)", camera["edit_data"]["inc"], format="%.3f")
                        _, camera["edit_data"]["Omega"] = imgui.input_double("Long Asc Node (deg)", camera["edit_data"]["Omega"], format="%.3f")
                        _, camera["edit_data"]["omega"] = imgui.input_double("Arg Periapsis (deg)", camera["edit_data"]["omega"], format="%.3f")
                        _, camera["edit_data"]["M"] = imgui.input_double("Mean Anomaly (deg)", camera["edit_data"]["M"], format="%.3f")
                    else:
                        use_km = oe_a < 0.01
                        AU_TO_KM = 1.496e8
                        
                        if use_km:
                            imgui.text(f"  Semi-major:  {oe_a * AU_TO_KM:,.0f} km")
                        else:
                            imgui.text(f"  Semi-major:  {oe_a:.6f} AU")
                        imgui.text(f"  Eccentricity: {oe_e:.6f}")
                        imgui.text(f"  Inclination:  {oe_inc:.4f}\u00b0")
                        imgui.text(f"  Omega (RAAN):   {oe_Omega:.4f}\u00b0")
                        imgui.text(f"  omega (Arg.Per.): {oe_omega:.4f}\u00b0")
                        imgui.text(f"  True Anom.:   {oe_nu:.4f}\u00b0")
                        imgui.text(f"  Mean Anom.:   {oe_M:.4f}\u00b0")
                    
                    imgui.separator()
                    imgui.text_colored("Derived Quantities", 1.0, 0.85, 0.4)
                    
                    if oe_P > 0:
                        if oe_P < 730:
                            imgui.text(f"  Period:      {oe_P:.3f} days")
                        else:
                            imgui.text(f"  Period:      {oe_P/365.25:.4f} years")
                    
                    periapsis = oe_a * (1.0 - oe_e)
                    apoapsis = oe_a * (1.0 + oe_e)
                    if use_km:
                        imgui.text(f"  Periapsis:   {periapsis * AU_TO_KM:,.0f} km")
                        imgui.text(f"  Apoapsis:    {apoapsis * AU_TO_KM:,.0f} km")
                        imgui.text(f"  Distance:    {dist_au * AU_TO_KM:,.0f} km")
                    else:
                        imgui.text(f"  Periapsis:   {periapsis:.6f} AU")
                        imgui.text(f"  Apoapsis:    {apoapsis:.6f} AU")
                        imgui.text(f"  Distance:    {dist_au:.6f} AU")
                    imgui.text(f"  Velocity:    {vel_km_s:.3f} km/s")
                    
                    if oe_a > 0 and oe_e < 1.0 and not inspect_bary:
                        imgui.separator()
                        imgui.text_colored("Precession (arcsec/cy)", 1.0, 0.85, 0.4)
                        
                        c2 = C_AU_YR * C_AU_YR
                        p_param = oe_a * (1.0 - oe_e * oe_e)
                        if p_param > 0:
                            gr_rad_per_orbit = 3.0 * mu / (oe_a * c2 * (1.0 - oe_e*oe_e))
                            orbits_per_century = 36525.0 / oe_P if oe_P > 0 else 0
                            gr_arcsec_cy = gr_rad_per_orbit * orbits_per_century * (180.0/math.pi) * 3600.0
                            imgui.text(f"  GR (apsidal):  {gr_arcsec_cy:.2f}")
                        else:
                            gr_arcsec_cy = 0.0
                            imgui.text(f"  GR (apsidal):  0.00")
                        
                        j2_apsidal = 0.0
                        j2_nodal = 0.0
                        parent_j2 = bodies_data[parent_idx].get('J2', 0.0)
                        if parent_j2 > 0 and oe_P > 0:
                            parent_r_au = bodies_data[parent_idx].get('r', 0.0) * 0.00465
                            n_mean = 2.0 * math.pi / (oe_P / 365.25)
                            if p_param > 0:
                                ratio2 = (parent_r_au / p_param) ** 2
                                j2_apsidal_rad_yr = 1.5 * n_mean * parent_j2 * ratio2
                                j2_nodal_rad_yr = -j2_apsidal_rad_yr * math.cos(math.radians(oe_inc))
                                j2_apsidal = j2_apsidal_rad_yr * 100.0 * (180.0/math.pi) * 3600.0
                                j2_nodal = j2_nodal_rad_yr * 100.0 * (180.0/math.pi) * 3600.0
                        
                        imgui.text(f"  J2 (apsidal):  {j2_apsidal:.2f}")
                        imgui.text(f"  J2 (nodal):    {j2_nodal:.2f}")
                
                if camera["edit_mode"] and not inspect_bary:
                    imgui.separator()
                    if imgui.button("Apply Changes", width=-1):
                        ed = camera["edit_data"]
                        payload = {
                            "action": "UPDATE",
                            "idx": insp_idx,
                            "mass": ed["mass"]
                        }
                        if parent_idx >= 0:
                            p_pos = pos_snap_render[parent_idx]
                            p_vel = vel_snap_render[parent_idx]
                            ppx, ppy, ppz = p_pos[0], -p_pos[2], p_pos[1]
                            pvx, pvy, pvz = p_vel[0], -p_vel[2], p_vel[1]
                            parent_m = mass_snap[parent_idx]
                            
                            c_pos, c_vel = get_cartesian_from_keplerian(
                                parent_m, ed["mass"], ed["a"], ed["e"], 
                                ed["inc"], ed["Omega"], ed["omega"], ed["M"]
                            )
                            
                            if camera["inspector_frame"] == 1:
                                pole_render = visual_arr[parent_idx, 5:8]
                                pole_ecl = np.array([pole_render[0], -pole_render[2], pole_render[1]])
                                c_pos, c_vel = rotate_equatorial_to_ecliptic(c_pos, c_vel, pole_ecl)
                                
                            new_ecl_x = ppx + c_pos[0]
                            new_ecl_y = ppy + c_pos[1]
                            new_ecl_z = ppz + c_pos[2]
                            
                            new_ecl_vx = pvx + c_vel[0]
                            new_ecl_vy = pvy + c_vel[1]
                            new_ecl_vz = pvz + c_vel[2]
                            
                            payload["pos"] = [new_ecl_x, new_ecl_y, new_ecl_z]
                            payload["vel"] = [new_ecl_vx, new_ecl_vy, new_ecl_vz]
                        
                        with shared_state["lock"]:
                            shared_state["crud_queue"].append(payload)
                            
                        body_info['r'] = ed["radius"] / 696340.0
                        visual_arr[insp_idx, 3] = body_info['r'] * 0.00465
                        
                        camera["edit_mode"] = False
                        
                imgui.separator()
                
                is_tracking_this = (camera["tracking_idx"] == insp_idx)
                is_tracking_body = is_tracking_this and camera["tracking_mode"] == "body"
                is_tracking_bary = is_tracking_this and camera["tracking_mode"] == "barycenter"
                
                if is_tracking_body:
                    imgui.text_colored("Tracking Body", 0.3, 1.0, 0.3)
                elif imgui.button("Track Body"):
                    old_pos = (pos_snap_render[camera["tracking_idx"]] + camera["pan_offset"]) if camera["tracking_idx"] is not None else camera["target"]
                    new_pos = pos_snap_render[insp_idx]
                    camera["target_offset"] += (old_pos - new_pos)
                    camera["pan_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                    camera["tracking_idx"] = insp_idx
                    camera["tracking_mode"] = "body"
                
                imgui.same_line()
                
                if is_tracking_bary:
                    imgui.text_colored("Tracking Barycenter", 0.3, 1.0, 0.3)
                elif imgui.button("Track Barycenter"):
                    old_pos = (subsys_pos_buf[camera["tracking_idx"]] + camera["pan_offset"]) if camera["tracking_idx"] is not None else camera["target"]
                    new_pos = subsys_pos_buf[insp_idx]
                    camera["target_offset"] += (old_pos - new_pos)
                    camera["pan_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                    camera["tracking_idx"] = insp_idx
                    camera["tracking_mode"] = "barycenter"
                
                if is_tracking_this:
                    if imgui.button("Untrack"):
                        camera["target"] = camera["target"].copy() + camera["pan_offset"]
                        camera["target_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                        camera["pan_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                        camera["tracking_idx"] = None
                        camera["tracking_mode"] = "body"
                        
                # Editor tools moved to System Hierarchy
                
                if not inspect_bary:
                    imgui.separator()
                    imgui.text_colored("Cosmetics", 1.0, 0.4, 0.4)
                    
                    imgui.same_line(imgui.get_window_width() - 70)
                    if imgui.button("Export"):
                        exp = {}
                        c = visual_arr[insp_idx, 0:3]
                        exp["color"] = '#%02x%02x%02x' % (min(255, max(0, int(c[0]*255))), min(255, max(0, int(c[1]*255))), min(255, max(0, int(c[2]*255))))
                        
                        atmo_item = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)
                        if atmo_item:
                            exp["atmosphere"] = {
                                "height": float(atmo_item["atmo_radius_km"] - atmo_item["planet_radius_km"]),
                                "rayleighCoefficients": [float(x) for x in atmo_item["beta_rayleigh"]],
                                "rayleighScaleHeight": float(atmo_item["h_rayleigh"]),
                                "mieCoefficient": float(atmo_item["beta_mie"]),
                                "mieScaleHeight": float(atmo_item["h_mie"]),
                                "mieAsymmetry": float(atmo_item["mie_g"]),
                                "absorptionCoefficients": [float(x) for x in atmo_item["beta_absorption"]],
                                "intensity": float(atmo_item["intensity"])
                            }
                            
                        ring_segs = [r for r in ring_precomputed if r['body_idx'] == insp_idx]
                        if ring_segs:
                            exp["rings"] = []
                            body_r_au = body_info.get('radius', 1000.0) / 1.496e8
                            for r in ring_segs:
                                rc = r.get('raw_color', [1.0, 1.0, 1.0])
                                hex_col = '#%02x%02x%02x' % (min(255, max(0, int(rc[0]*255))), min(255, max(0, int(rc[1]*255))), min(255, max(0, int(rc[2]*255))))
                                exp["rings"].append({
                                    "inner": float(r['inner_r'] / body_r_au) if body_r_au > 0 else 1.0,
                                    "outer": float(r['outer_r'] / body_r_au) if body_r_au > 0 else 2.0,
                                    "color": hex_col,
                                    "opacity": float(r['opacity']),
                                    "scatter": float(r['scatter']),
                                    "asymmetry": float(r['asymmetry']),
                                    "gradient": [{"p": float(g['p']), "a": float(g['a'])} for g in r.get('gradient', [])]
                                })
                        
                        import os
                        os.makedirs("exports", exist_ok=True)
                        filename = os.path.join("exports", f"{body_info['name'].replace(' ', '_').lower()}_cosmetics.json")
                        with open(filename, 'w') as f:
                            json.dump(exp, f, indent=4)
                        print(f"Exported cosmetics to {filename}")
                        
                    changed_c, new_c = imgui.color_edit3("Base Color", *visual_arr[insp_idx, 0:3])
                    if changed_c:
                        visual_arr[insp_idx, 0:3] = new_c
                    
                    if imgui.collapsing_header("Atmosphere")[0]:
                        atmo_item = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)
                        if atmo_item:
                            changed_r, new_r = imgui.drag_float("Atmosphere Radius (km)", atmo_item['atmo_radius_km'], 10.0, body_r_km, body_r_km * 100.0)
                            if changed_r:
                                atmo_item['atmo_radius_km'] = new_r
                                atmo_item['atmo_radius_au'] = new_r / 1.496e8
                            b_r = atmo_item['beta_rayleigh']
                            changed_b, b_ray = imgui.drag_float3("Rayleigh Beta (x10^-6)", b_r[0]*1e6, b_r[1]*1e6, b_r[2]*1e6, 0.1)
                            if changed_b:
                                atmo_item['beta_rayleigh'] = np.array([b_ray[0]*1e-6, b_ray[1]*1e-6, b_ray[2]*1e-6], dtype='f4')
                            _, atmo_item['h_rayleigh'] = imgui.drag_float("Rayleigh Scale (km)", atmo_item['h_rayleigh'], 0.1, 0.1, 1000.0)
                            
                            changed_m, b_mie = imgui.drag_float("Mie Beta (x10^-6)", atmo_item['beta_mie']*1e6, 0.1)
                            if changed_m:
                                atmo_item['beta_mie'] = b_mie * 1e-6
                            _, atmo_item['h_mie'] = imgui.drag_float("Mie Scale (km)", atmo_item['h_mie'], 0.1, 0.1, 1000.0)
                            _, atmo_item['mie_g'] = imgui.slider_float("Mie Asymmetry", atmo_item['mie_g'], 0.0, 0.999)
                            
                            b_a = atmo_item['beta_absorption']
                            changed_a, b_abs = imgui.drag_float3("Absorption Beta (x10^-6)", b_a[0]*1e6, b_a[1]*1e6, b_a[2]*1e6, 0.1)
                            if changed_a:
                                atmo_item['beta_absorption'] = np.array([b_abs[0]*1e-6, b_abs[1]*1e-6, b_abs[2]*1e-6], dtype='f4')
                                
                            _, atmo_item['intensity'] = imgui.drag_float("Intensity", atmo_item['intensity'], 0.5, 0.0, 1000.0)
                            
                            if imgui.button("Remove Atmosphere"):
                                atmo_bodies.remove(atmo_item)
                        else:
                            if imgui.button("Add Atmosphere"):
                                atmo_bodies.append({
                                    'body_idx': insp_idx,
                                    'planet_radius_km': body_r_km,
                                    'surface_radius_au': body_r_km / 1.496e8,
                                    'atmo_radius_km': body_r_km * 1.025,
                                    'atmo_radius_au': (body_r_km * 1.025) / 1.496e8,
                                    'beta_rayleigh': np.array([5.5e-6, 13.0e-6, 22.4e-6], dtype='f4'),
                                    'h_rayleigh': 8.0,
                                    'beta_mie': 21.0e-6,
                                    'h_mie': 1.2,
                                    'mie_g': 0.758,
                                    'beta_absorption': np.array([0,0,0], dtype='f4'),
                                    'intensity': 1.0
                                })
                                
                    if imgui.collapsing_header("Rings")[0]:
                        ring_group = next((g for g in ring_render_groups if g['body_idx'] == insp_idx), None)
                        rings = [r for r in ring_precomputed if r['body_idx'] == insp_idx]
                        
                        for i, ring_item in enumerate(rings):
                            if imgui.tree_node(f"Ring Layer {i}"):
                                changed_in, new_in = imgui.drag_float(f"Inner Radius (km)##{i}", ring_item['inner_r'] * 1.496e8, 10.0, body_r_km, ring_item['outer_r'] * 1.496e8 - 10)
                                changed_out, new_out = imgui.drag_float(f"Outer Radius (km)##{i}", ring_item['outer_r'] * 1.496e8, 10.0, ring_item['inner_r'] * 1.496e8 + 10, body_r_km * 50.0)
                                changed_col, new_col = imgui.color_edit3(f"Color##{i}", *ring_item['raw_color'])
                                changed_op, new_op = imgui.slider_float(f"Opacity##{i}", ring_item['opacity'], 0.0, 1.0)
                                changed_scat, new_scat = imgui.drag_float(f"Forward Scatter Mult##{i}", ring_item['scatter'], 0.05, 0.0, 100.0)
                                changed_asym, new_asym = imgui.slider_float(f"Forward Scatter Asym##{i}", ring_item['asymmetry'], 0.0, 0.999)
                                
                                grad_changed = False
                                if imgui.tree_node(f"Alpha Gradient##{i}"):
                                    grad = ring_item['gradient']
                                    if imgui.button(f"Add Stop##{i}"):
                                        grad.append({'p': 1.0, 'a': 1.0})
                                    
                                    stops_to_remove = []
                                    for j, stop in enumerate(grad):
                                        imgui.push_item_width(100)
                                        changed_p, n_p = imgui.slider_float(f"Pos##{i}_{j}", stop['p'], 0.0, 1.0)
                                        imgui.same_line()
                                        changed_a, n_a = imgui.slider_float(f"Alpha##{i}_{j}", stop['a'], 0.0, 1.0)
                                        imgui.same_line()
                                        if imgui.button(f"X##{i}_{j}"):
                                            stops_to_remove.append(j)
                                        imgui.pop_item_width()
                                        
                                        if changed_p or changed_a:
                                            stop['p'] = n_p
                                            stop['a'] = n_a
                                            grad_changed = True
                                            
                                    for j in reversed(stops_to_remove):
                                        grad.pop(j)
                                        grad_changed = True
                                        
                                    if grad_changed:
                                        grad.sort(key=lambda x: x['p'])
                                        
                                    imgui.tree_pop()
                                
                                if changed_in or changed_out or changed_col or changed_op or changed_scat or changed_asym or grad_changed:
                                    if changed_in: ring_item['inner_r'] = new_in / 1.496e8
                                    if changed_out: ring_item['outer_r'] = new_out / 1.496e8
                                    if changed_col: ring_item['raw_color'] = new_col
                                    if changed_op: ring_item['opacity'] = new_op
                                    if changed_scat: ring_item['scatter'] = new_scat
                                    if changed_asym: ring_item['asymmetry'] = new_asym
                                    
                                    pole_render = visual_arr[insp_idx, 5:8]
                                    verts, indices, norm, colors, shadow_grad = generate_ring_arrays(
                                        pole_render, ring_item['inner_r'], ring_item['outer_r'],
                                        ring_item['raw_color'], ring_item['opacity'], ring_item['scatter'], ring_item['asymmetry'], ring_item['gradient']
                                    )
                                    ring_item['verts'] = verts
                                    ring_item['indices'] = indices
                                    ring_item['colors'] = colors
                                    ring_item['shadow_grad'] = shadow_grad
                                    rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                                
                                if imgui.button(f"Remove Layer##{i}"):
                                    ring_precomputed.remove(ring_item)
                                    rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                                
                                imgui.tree_pop()
                                
                        if imgui.button("Add Ring Layer"):
                            if len(rings) < 16:
                                r_in = body_r_km * 1.2 / 1.496e8
                                if rings:
                                    r_in = rings[-1]['outer_r'] + (100 / 1.496e8)
                                r_out = r_in + (body_r_km * 0.5 / 1.496e8)
                                pole_render = visual_arr[insp_idx, 5:8]
                                color = visual_arr[insp_idx, 0:3]
                                grad = [{'p': 0.0, 'a': 0.0}, {'p': 0.5, 'a': 1.0}, {'p': 1.0, 'a': 0.0}]
                                verts, indices, norm, colors, shadow_grad = generate_ring_arrays(
                                    pole_render, r_in, r_out, color, 1.0, 2.5, 0.8, grad
                                )
                                ring_precomputed.append({
                                    'body_idx': insp_idx,
                                    'verts': verts, 'indices': indices, 'normal': norm, 'colors': colors,
                                    'inner_r': r_in, 'outer_r': r_out, 'opacity': 1.0,
                                    'scatter': 2.5, 'asymmetry': 0.8, 'shadow_grad': shadow_grad,
                                    'raw_color': color, 'gradient': grad
                                })
                                rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                
            imgui.end()

        if camera.get("add_mode", False):
            imgui.set_next_window_size(350, 400, imgui.FIRST_USE_EVER)
            imgui.set_next_window_position(600, 50, imgui.FIRST_USE_EVER)
            expanded, camera["add_mode"] = imgui.begin("Add Orbiting Body", True)
            if expanded:
                ad = camera["add_data"]
                _, ad["name"] = imgui.input_text("Name", ad["name"], 256)
                _, ad["color"] = imgui.color_edit3("Color", *ad["color"])
                _, ad["mass"] = imgui.input_double("Mass (M_sun)", ad["mass"], format="%e")
                _, ad["radius"] = imgui.input_double("Radius (km)", ad["radius"], format="%.1f")
                
                imgui.separator()
                imgui.text_colored("Orbital Elements", 1.0, 0.85, 0.4)
                _, ad["frame"] = imgui.combo("Reference Frame", ad["frame"], ["Ecliptic", "Equatorial"])
                _, ad["a"] = imgui.input_double("Semi-Major Axis (AU)", ad["a"], format="%.6f")
                _, ad["e"] = imgui.input_double("Eccentricity", ad["e"], format="%.6f")
                _, ad["inc"] = imgui.input_double("Inclination (deg)", ad["inc"], format="%.3f")
                _, ad["Omega"] = imgui.input_double("Long Asc Node (deg)", ad["Omega"], format="%.3f")
                _, ad["omega"] = imgui.input_double("Arg Periapsis (deg)", ad["omega"], format="%.3f")
                _, ad["M"] = imgui.input_double("Mean Anomaly (deg)", ad["M"], format="%.3f")
                
                imgui.separator()
                if imgui.button("Spawn Body", width=-1):
                    insp_idx = camera["inspected_idx"]
                    parent_m = mass_snap[insp_idx]
                    p_pos = pos_snap_render[insp_idx]
                    p_vel = vel_snap_render[insp_idx]
                    ppx, ppy, ppz = p_pos[0], -p_pos[2], p_pos[1]
                    pvx, pvy, pvz = p_vel[0], -p_vel[2], p_vel[1]
                    
                    c_pos, c_vel = get_cartesian_from_keplerian(
                        parent_m, ad["mass"], ad["a"], ad["e"], 
                        ad["inc"], ad["Omega"], ad["omega"], ad["M"]
                    )
                    
                    if ad["frame"] == 1:
                        pole_render = visual_arr[insp_idx, 5:8]
                        pole_ecl = np.array([pole_render[0], -pole_render[2], pole_render[1]])
                        c_pos, c_vel = rotate_equatorial_to_ecliptic(c_pos, c_vel, pole_ecl)
                    
                    new_ecl_x = ppx + c_pos[0]
                    new_ecl_y = ppy + c_pos[1]
                    new_ecl_z = ppz + c_pos[2]
                    
                    new_ecl_vx = pvx + c_vel[0]
                    new_ecl_vy = pvy + c_vel[1]
                    new_ecl_vz = pvz + c_vel[2]
                    
                    payload = {
                        "action": "CREATE",
                        "name": ad["name"],
                        "radius": ad["radius"],
                        "mass": ad["mass"],
                        "color": list(ad["color"]),
                        "type": "Moon" if parent_m < 0.01 else "Planet",
                        "parent_idx": insp_idx,
                        "pos": [new_ecl_x, new_ecl_y, new_ecl_z],
                        "vel": [new_ecl_vx, new_ecl_vy, new_ecl_vz]
                    }
                    with shared_state["lock"]:
                        shared_state["crud_queue"].append(payload)
                    camera["add_mode"] = False
            imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        
    running[0] = False
    physics_thread.join(timeout=1.0)
    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()