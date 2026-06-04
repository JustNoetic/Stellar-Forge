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
from numba import njit

# --- CONSTANTS ---
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0  # ~31,557,600
C_AU_YR = 63197.79  # speed of light in AU/yr (for GR corrections)
FOV_DEG = 45.0

# --- OPTIONAL PERFORMANCE INSTRUMENTATION ---
# Activated only when STELLAR_FORGE_PERF=1 is set in the environment.
# When disabled, _perf("name") compiles to a near-zero-cost no-op (~30 ns).
import os as _os
_PERF_ENABLED = _os.environ.get("STELLAR_FORGE_PERF") == "1"
_PERF_TRACKER = None  # attached in main() if _PERF_ENABLED

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
    def _perf(name):  # noqa: F811
        return _NullCtx()

def _PERF_INSTALL_TRACKER(tracker):
    """Public hook so perf_test.py can attach a tracker after import."""
    global _PERF_TRACKER
    _PERF_TRACKER = tracker

# --- HELPER FUNCTIONS ---
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


@njit(cache=True, fastmath=True)
def compute_j2_numba(p_array, num_particles, oblate_indices, oblate_j2, oblate_req, oblate_poles, oblate_masses):
    G = 4.0 * math.pi**2
    for o_idx in range(len(oblate_indices)):
        body_i = oblate_indices[o_idx]
        J2 = oblate_j2[o_idx]
        Req = oblate_req[o_idx]
        px, py, pz = oblate_poles[o_idx, 0], oblate_poles[o_idx, 1], oblate_poles[o_idx, 2]
        M = oblate_masses[o_idx]
        
        ox, oy, oz = p_array[body_i, 0], p_array[body_i, 1], p_array[body_i, 2]
        C_base = -1.5 * J2 * G * M * Req * Req
        
        for i in range(num_particles):
            if i == body_i: continue
            x, y, z = p_array[i, 0] - ox, p_array[i, 1] - oy, p_array[i, 2] - oz
            r2 = x*x + y*y + z*z
            if r2 == 0.0: continue
            r = math.sqrt(r2)
            C = C_base / (r2 * r2 * r)
            z_prime = x*px + y*py + z*pz
            factor1 = 1.0 - 5.0 * (z_prime*z_prime) / r2
            
            ax = C * (x * factor1 + 2.0 * z_prime * px)
            ay = C * (y * factor1 + 2.0 * z_prime * py)
            az = C * (z * factor1 + 2.0 * z_prime * pz)
            
            p_array[i, 6] += ax
            p_array[i, 7] += ay
            p_array[i, 8] += az
            if M > 0.0:
                m_sat = p_array[i, 9]
                p_array[body_i, 6] -= ax * m_sat / M
                p_array[body_i, 7] -= ay * m_sat / M
                p_array[body_i, 8] -= az * m_sat / M

@njit(cache=True, fastmath=True)
def compute_gr_1pn_numba(p_array, num_particles, star_idx, star_mass, c_au_yr):
    """1PN post-Newtonian GR correction for relativistic perihelion precession.
    
    Applies the velocity-dependent acceleration from the central star:
      a_GR = (GM/(c^2 r^3)) * [(4GM/r - v^2) r + 4(r.v) v]
    
    Produces the correct ~43 arcsec/century precession for Mercury.
    """
    G = 4.0 * math.pi**2
    mu = G * star_mass
    c2 = c_au_yr * c_au_yr
    
    sx = p_array[star_idx, 0]
    sy = p_array[star_idx, 1]
    sz = p_array[star_idx, 2]
    svx = p_array[star_idx, 3]
    svy = p_array[star_idx, 4]
    svz = p_array[star_idx, 5]
    
    for i in range(num_particles):
        if i == star_idx:
            continue
        
        # Relative position and velocity w.r.t. the star
        rx = p_array[i, 0] - sx
        ry = p_array[i, 1] - sy
        rz = p_array[i, 2] - sz
        vx = p_array[i, 3] - svx
        vy = p_array[i, 4] - svy
        vz = p_array[i, 5] - svz
        
        r2 = rx*rx + ry*ry + rz*rz
        if r2 == 0.0:
            continue
        r = math.sqrt(r2)
        r3 = r * r2
        
        v2 = vx*vx + vy*vy + vz*vz
        rdotv = rx*vx + ry*vy + rz*vz
        
        # 1PN acceleration: position-dependent + velocity-dependent terms
        prefactor = mu / (c2 * r3)
        pos_term = 4.0 * mu / r - v2
        vel_term = 4.0 * rdotv
        
        ax = prefactor * (pos_term * rx + vel_term * vx)
        ay = prefactor * (pos_term * ry + vel_term * vy)
        az = prefactor * (pos_term * rz + vel_term * vz)
        
        p_array[i, 6] += ax
        p_array[i, 7] += ay
        p_array[i, 8] += az

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

OBLIQUITY = math.radians(23.439)  # Earth's obliquity for ICRF -> ecliptic

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
    return np.column_stack([x, y, z])

def kepler_solve(M, e, tol=1e-12):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
    E = M
    for _ in range(100):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E

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
    # Position and velocity magnitudes
    r = math.sqrt(rx*rx + ry*ry + rz*rz)
    v2 = vx*vx + vy*vy + vz*vz
    if r < 1e-30 or mu < 1e-30:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Specific angular momentum h = r × v
    hx = ry*vz - rz*vy
    hy = rz*vx - rx*vz
    hz = rx*vy - ry*vx
    h = math.sqrt(hx*hx + hy*hy + hz*hz)
    if h < 1e-30:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Eccentricity vector e = (v × h)/mu - r_hat
    # v × h
    vhx = vy*hz - vz*hy
    vhy = vz*hx - vx*hz
    vhz = vx*hy - vy*hx
    ex = vhx/mu - rx/r
    ey = vhy/mu - ry/r
    ez = vhz/mu - rz/r
    e = math.sqrt(ex*ex + ey*ey + ez*ez)
    
    # Semi-major axis from vis-viva: epsilon = v²/2 - mu/r = -mu/(2a)
    energy = v2/2.0 - mu/r
    if abs(energy) < 1e-30:
        a = 1e30  # parabolic
    else:
        a = -mu / (2.0 * energy)
    
    # Inclination: cos(i) = h_z / |h|
    inc = math.acos(max(-1.0, min(1.0, hz / h)))
    
    # Node vector n = z_hat × h = (-hy, hx, 0)
    nx = -hy
    ny = hx
    n_mag = math.sqrt(nx*nx + ny*ny)
    
    # Longitude of ascending node
    if n_mag > 1e-15:
        Omega = math.atan2(ny, nx)
        if Omega < 0:
            Omega += 2.0 * math.pi
    else:
        Omega = 0.0  # undefined for zero inclination
    
    # Argument of periapsis
    if n_mag > 1e-15 and e > 1e-10:
        # cos(omega) = (n · e) / (|n| * |e|)
        n_dot_e = (nx*ex + ny*ey) / (n_mag * e)
        omega = math.acos(max(-1.0, min(1.0, n_dot_e)))
        if ez < 0:
            omega = 2.0 * math.pi - omega
    elif e > 1e-10:
        # No inclination: omega measured from x-axis
        omega = math.atan2(ey, ex)
        if omega < 0:
            omega += 2.0 * math.pi
    else:
        omega = 0.0
    
    # True anomaly
    if e > 1e-10:
        r_dot_e = (rx*ex + ry*ey + rz*ez) / (r * e)
        nu = math.acos(max(-1.0, min(1.0, r_dot_e)))
        # r · v determines sign
        rdotv = rx*vx + ry*vy + rz*vz
        if rdotv < 0:
            nu = 2.0 * math.pi - nu
    else:
        # Circular: true longitude
        if n_mag > 1e-15:
            r_dot_n = (rx*nx + ry*ny) / (r * n_mag)
            nu = math.acos(max(-1.0, min(1.0, r_dot_n)))
            if rz < 0:
                nu = 2.0 * math.pi - nu
        else:
            nu = math.atan2(ry, rx)
            if nu < 0:
                nu += 2.0 * math.pi
    
    # Mean anomaly from eccentric anomaly
    if e < 1.0:
        # Elliptical
        cos_nu = math.cos(nu)
        sin_nu = math.sin(nu)
        sin_E = sin_nu * math.sqrt(max(0.0, 1.0 - e*e)) / (1.0 + e*cos_nu)
        cos_E = (e + cos_nu) / (1.0 + e*cos_nu)
        E = math.atan2(sin_E, cos_E)
        M = E - e * sin_E
        if M < 0:
            M += 2.0 * math.pi
    else:
        M = 0.0  # hyperbolic — M not standard
    
    # Orbital period (only meaningful for elliptical)
    if a > 0 and e < 1.0:
        period_yr = 2.0 * math.pi * math.sqrt(a*a*a / mu)
        period_days = period_yr * 365.25
    else:
        period_days = 0.0
    
    deg = 180.0 / math.pi
    return a, e, inc*deg, Omega*deg, omega*deg, nu*deg, M*deg, period_days

ORBIT_RESOLUTION = 1001
ORBIT_STRIDE_BYTES = 128  # 4 x dvec4 = 16 doubles per orbit

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
    
    # === Child orbits ===
    for i in range(num_bodies):
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
        
        orb_rel_pos = np.empty(3, dtype=np.float64)
        orb_rel_vel = np.empty(3, dtype=np.float64)
        bary = np.empty(3, dtype=np.float64)
        
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
                continue
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
            orbit_buf[n_orbits, 11] = math.cos(theta0)
            orbit_buf[n_orbits, 12] = visual_colors[i, 0]
            orbit_buf[n_orbits, 13] = visual_colors[i, 1]
            orbit_buf[n_orbits, 14] = visual_colors[i, 2]
            orbit_buf[n_orbits, 15] = math.sin(theta0)
            n_orbits += 1
    
    # === Reflex orbits ===
    # Find most massive child per parent
    best_child = np.full(num_bodies, -1, dtype=np.int64)
    best_child_mass = np.zeros(num_bodies, dtype=np.float64)
    for i in range(num_bodies):
        p_idx = parent_indices[i]
        if p_idx >= 0:
            if mass[i] > best_child_mass[p_idx]:
                best_child_mass[p_idx] = mass[i]
                best_child[p_idx] = i
    
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
        
        rel_pos = np.empty(3, dtype=np.float64)
        rel_vel = np.empty(3, dtype=np.float64)
        bary = np.empty(3, dtype=np.float64)
        parent_bary_ref = np.empty(3, dtype=np.float64)
        
        for k in range(3):
            rel_pos[k] = pos[ci, k] - pos[parent_idx, k]
            rel_vel[k] = vel[ci, k] - vel[parent_idx, k]
            bary[k] = (p_m * pos[parent_idx, k] + c_m * pos[ci, k]) / total_m
            parent_bary_ref[k] = pos[parent_idx, k] - bary[k]
        
        e_hat, q_hat, bary_rel, sl_p, e_mag, theta0, is_valid = compute_orbit_elements(
            rel_pos, rel_vel, mu, c_m / total_m, bary, cam_origin,
            True, True, parent_bary_ref)
        
        if is_valid and n_orbits < max_orbits:
            bary_dist = fast_norm(bary_rel)
            if bary_dist > 1e-6 and sl_p / bary_dist < 1e-6:
                continue
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
            orbit_buf[n_orbits, 11] = math.cos(theta0)
            orbit_buf[n_orbits, 12] = visual_colors[parent_idx, 0]
            orbit_buf[n_orbits, 13] = visual_colors[parent_idx, 1]
            orbit_buf[n_orbits, 14] = visual_colors[parent_idx, 2]
            orbit_buf[n_orbits, 15] = math.sin(theta0)
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
    # Offset from 2026-06-03 12:00 back to 2000-01-01 00:00
    t_years += 26.0 + (153.5 / 365.25)
    
    y = 2000 + int(math.floor(t_years))
    rem_days = (t_years - math.floor(t_years)) * 365.25
    if rem_days < 0:
        rem_days += 365.25
        y -= 1
        
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
    if is_leap: months[1] = 29
    
    m = 1
    d = rem_days + 1
    for days_in_m in months:
        if d <= days_in_m:
            break
        d -= days_in_m
        m += 1
        if m > 12:
            m = 12
            break
            
    hours = (d - int(d)) * 24
    mins = (hours - int(hours)) * 60
    return y, m, int(d), int(hours), int(mins)

def sim_time_from_date(y, m, d):
    is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap: months[1] = 29
    
    days = d - 1
    for i in range(m - 1):
        days += months[i]
        
    t_years = (y - 2000) + (days / 365.25)
    # Offset to our new epoch (2026-06-03 12:00)
    t_years -= (26.0 + (153.5 / 365.25))
    return t_years

@njit(cache=True)
def _update_hierarchy_core(positions, masses, current_parents, num_bodies):
    """Numba-jitted core: recompute parents based on Hill sphere containment.
    No O(N²) memory allocation — distances computed inline as needed."""
    new_parents = current_parents.copy()
    
    # Root = most massive body
    root_idx = 0
    max_mass = masses[0]
    for i in range(1, num_bodies):
        if masses[i] > max_mass:
            max_mass = masses[i]
            root_idx = i
    new_parents[root_idx] = -1
    
    # Compute Hill spheres for all potential parents (inline distance)
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
    
    # Find best parent for each body (inline distance, no matrix)
    for i in range(num_bodies):
        if i == root_idx:
            continue
        
        best_parent = root_idx
        best_hill = 1e30
        
        for j in range(num_bodies):
            if j == i or j == root_idx or masses[j] <= 0.0 or r_hill_array[j] == 0.0:
                continue
            
            r_hill = r_hill_array[j]
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            dist_ij = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Hysteresis: tighter threshold to enter, full sphere to stay
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
    # Extract positions and masses from sim (can't be in Numba)
    positions = np.zeros((num_bodies, 3))
    masses = np.zeros(num_bodies)
    for i in range(num_bodies):
        p = sim.particles[i]
        positions[i] = [p.x, p.y, p.z]
        masses[i] = p.m
    
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
            return  # Guard against cycles from floating-point edge cases
        visited.add(node)
        result.append((node, depth))
        for child in children[node]:
            dfs(child, depth + 1)
    
    for root in roots:
        dfs(root, 0)
    
    tree_indices = np.array([r[0] for r in result], dtype=np.int32)
    tree_depths = np.array([r[1] for r in result], dtype=np.int32)
    return tree_indices, tree_depths

def create_sphere_mesh(segments=32, rings=16):
    """Create a unit sphere mesh with positions and normals (vectorized)."""
    phi = np.linspace(0, math.pi, rings + 1)
    theta = np.linspace(0, 2.0 * math.pi, segments + 1)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Spherical to Cartesian
    x = (np.cos(theta_grid) * np.sin(phi_grid)).ravel()
    y = np.cos(phi_grid).ravel()
    z = (np.sin(theta_grid) * np.sin(phi_grid)).ravel()
    
    # Interleave position + normal (identical for unit sphere): 6 floats per vertex
    n_verts = (rings + 1) * (segments + 1)
    vertices = np.empty((n_verts, 6), dtype='f4')
    vertices[:, 0] = vertices[:, 3] = x
    vertices[:, 1] = vertices[:, 4] = y
    vertices[:, 2] = vertices[:, 5] = z
    
    # Generate triangle indices (vectorized)
    ii, jj = np.meshgrid(np.arange(rings), np.arange(segments), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    p1 = ii * (segments + 1) + jj
    p2 = p1 + segments + 1
    indices = np.column_stack([p1, p1 + 1, p2, p1 + 1, p2 + 1, p2]).ravel().astype('i4')
    
    return vertices.ravel(), indices

# --- STATE MANAGERS ---
camera = {
    "target": np.array([0.0, 0.0, 0.0], dtype='f8'),
    "target_offset": np.array([0.0, 0.0, 0.0], dtype='f8'), # For smooth transitions
    "distance": 45.0,     
    "distance_actual": 45.0,
    "yaw": -90.0,         
    "yaw_actual": -90.0,
    "pitch": 25.0,        
    "pitch_actual": 25.0,
    "left_dragging": False,
    "right_dragging": False,
    "last_x": 0.0,
    "last_y": 0.0,
    "tracking_idx": 0,       # Body index camera follows (None = free)
    "tracking_mode": "body", # "body" or "barycenter"
    "inspected_idx": None,   # Body index for inspector panel (None = closed)
    "inspect_bary": False,   # True = showing barycenter properties
}

time_ctrl = {
    "paused": False,
    "multiplier": 1.0,  # 1.0 = realtime (1 second/second)
    "render_timeline": False,
    "cancel_render": False,
    "target_t": 0.0,
    "sync_t": None,
}

window_width, window_height = 1280, 720
fb_width, fb_height = 1280, 720
impl = None

# --- INPUT CALLBACKS ---
def scroll_callback(window, xoffset, yoffset):
    if impl: impl.scroll_callback(window, xoffset, yoffset) # Forward to ImGui
    if imgui.get_io().want_capture_mouse: return 
    
    zoom_speed = camera["distance"] * 0.2 
    if yoffset > 0: camera["distance"] -= zoom_speed
    elif yoffset < 0: camera["distance"] += zoom_speed
    camera["distance"] = max(1e-7, camera["distance"])

def mouse_button_callback(window, button, action, mods):
    if impl: impl.mouse_callback(window, button, action, mods) # Forward to ImGui
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
            # Panning no longer untracks — tracking is sticky
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
        
        camera["target"] -= right * dx * pan_speed
        camera["target"] += up * dy * pan_speed

    camera["last_x"], camera["last_y"] = xpos, ypos

def key_callback(window, key, scancode, action, mods):
    if impl: impl.keyboard_callback(window, key, scancode, action, mods) # Forward to ImGui
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
    if impl: impl.resize_callback(window, width, height) # Forward to ImGui
    global window_width, window_height, fb_width, fb_height
    window_width, window_height = max(1, width), max(1, height)
    fb_width, fb_height = glfw.get_framebuffer_size(window)


def physics_loop(sim, num_bodies, shared_state, time_ctrl, running):
    last_time = time.time()
    
    # Run an initial hierarchy update
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

        if time_ctrl.get("sync_t") is not None:
            sync_time = time_ctrl["sync_t"]
            time_ctrl["sync_t"] = None
            sim.integrate(sync_time)
            
            with shared_state["lock"]:
                for i in range(num_bodies):
                    p = sim.particles[i]
                    shared_state["pos"][i] = (p.x, p.z, -p.y)
                    shared_state["vel"][i] = (p.vx, p.vz, -p.vy)
                shared_state["t"] = sim.t
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
            
            with shared_state["lock"]:
                shared_state["timeline_active"] = True
                shared_state["timeline_progress"] = 0.0
            
            for step in range(num_steps):
                if not running[0] or time_ctrl.get("cancel_render", False):
                    break
                sim.integrate(start_t + (step + 1) * step_dt)
                
                for i in range(num_bodies):
                    p = sim.particles[i]
                    tl_pos[step, i] = (p.x, p.z, -p.y)
                    tl_vel[step, i] = (p.vx, p.vz, -p.vy)
                tl_times[step] = sim.t
                
                if step % 20 == 0:
                    with shared_state["lock"]:
                        shared_state["timeline_progress"] = float(step) / num_steps
            
            with shared_state["lock"]:
                shared_state["timeline_pos"] = tl_pos
                shared_state["timeline_vel"] = tl_vel
                shared_state["timeline_times"] = tl_times
                shared_state["timeline_progress"] = 1.0
                for i in range(num_bodies):
                    p = sim.particles[i]
                    shared_state["pos"][i] = (p.x, p.z, -p.y)
                    shared_state["vel"][i] = (p.vx, p.vz, -p.vy)
                shared_state["t"] = sim.t
            
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
                
            with shared_state["lock"]:
                for i in range(num_bodies):
                    p = sim.particles[i]
                    shared_state["pos"][i] = (p.x, p.z, -p.y)
                    shared_state["vel"][i] = (p.vx, p.vz, -p.vy)
                shared_state["t"] = sim.t
                shared_state["parent_indices"][:] = current_parents
                shared_state["tree_indices"][:] = tree_indices
                shared_state["tree_depths"][:] = tree_depths
                if frame_count % 30 == 0:
                    shared_state["hierarchy_version"] += 1
                
        # Throttle physics loop to ~60 updates/sec to prevent GIL contention
        elapsed = time.time() - now
        sleep_time = max(0.001, 0.016 - elapsed)
        time.sleep(sleep_time)

# --- MAIN APPLICATION ---
def main():
    # 0. PERF: install tracker (only effective when STELLAR_FORGE_PERF=1)
    if _PERF_ENABLED:
        from perf_test import PerfTracker as _BootTracker
        _BootTracker().__class__  # ensure class import works
        if _PERF_TRACKER is None:
            _PERF_INSTALL_TRACKER(_BootTracker())

    # 1. INITIALIZE REBOUND & DATA
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.units = ('AU', 'Msun', 'yr')
    
    with open('system.json', 'r') as file:
        bodies_data = json.load(file)

    # Topological sort: ensure parents are added to REBOUND before their children
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
    
    # Reorder bodies_data so parents come before children (matches REBOUND particle indices)
    bodies_data = [bodies_data[i] for i in sorted_order]
    name_to_idx = {b['name']: i for i, b in enumerate(bodies_data)}  # Rebuild after reorder

    visual_data = []
    parent_indices = []
    pole_dirs = {}  # body_idx -> pole unit vector in RENDERING coords
    ring_bodies = []  # list of (body_idx, rings_data, pole_render, radius_au)
    oblate_physics_list = []
    G_const = 4.0 * math.pi**2  # gravitational constant in AU, Msun, yr units

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
            oblate_physics_list.append((idx, j2, radius_au, pole_ecl, mass))
        
        # Store ring data for later mesh generation
        if 'rings' in body:
            parent_pole_ecl = pole_to_ecliptic(body.get('pole_ra', 0), body.get('pole_dec', 90))
            pole_r = np.array([parent_pole_ecl[0], parent_pole_ecl[2], -parent_pole_ecl[1]])
            ring_bodies.append((idx, body['rings'], pole_r, radius_au))

        if 'parentId' in body:
            parent_name = body['parentId']
            # Prefer JPL Horizons state vectors when available
            if 'sv' in body:
                sv = body['sv']
                primary = sim.particles[parent_name]
                sim.add(m=mass,
                        x=primary.x + sv['x'], y=primary.y + sv['y'], z=primary.z + sv['z'],
                        vx=primary.vx + sv['vx'], vy=primary.vy + sv['vy'], vz=primary.vz + sv['vz'],
                        hash=name)
            # Check if this body uses equatorial orbit reference
            elif body.get('orbitRef') == 'equatorial':
                # Find parent body data for pole direction
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
                    # Fallback to ecliptic if parent has no pole data
                    sim.add(m=mass, primary=sim.particles[parent_name], 
                            a=body.get('a', 0.0), e=body.get('e', 0.0), 
                            inc=math.radians(body.get('inc', 0.0)), Omega=math.radians(body.get('Omega', 0.0)), 
                            omega=math.radians(body.get('omega', 0.0)), M=math.radians(body.get('M', 0.0)), 
                            hash=name)
            else:
                # Standard ecliptic orbital elements
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

    # === Parse atmosphere data ===
    atmo_bodies = []
    SOLAR_RADIUS_KM = 695700.0  # r field in system.json is in solar radii
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

    # === ADDITIONAL FORCES: J2 oblate precession + GR perihelion precession ===
    has_j2 = bool(oblate_physics_list)
    if has_j2:
        oblate_indices = np.array([x[0] for x in oblate_physics_list], dtype=np.int32)
        oblate_j2 = np.array([x[1] for x in oblate_physics_list], dtype=np.float64)
        oblate_req = np.array([x[2] for x in oblate_physics_list], dtype=np.float64)
        oblate_poles = np.array([x[3] for x in oblate_physics_list], dtype=np.float64)
        oblate_masses = np.array([x[4] for x in oblate_physics_list], dtype=np.float64)

    # Find central star for GR correction
    phys_star_idx = -1
    phys_star_mass = 0.0
    for i, body in enumerate(bodies_data):
        if body.get('type') == 'Star':
            phys_star_idx = i
            phys_star_mass = body.get('m', 0.0)
            break
    has_gr = phys_star_idx >= 0

    if has_j2 or has_gr:
        def additional_forces_callback(sim_pointer):
            n = sim.N
            if n == 0:
                return
            p_address = ctypes.cast(sim._particles, ctypes.c_void_p).value
            ptr = ctypes.cast(p_address, ctypes.POINTER(ctypes.c_double))
            arr = np.ctypeslib.as_array(ptr, shape=(n, 16))
            if has_j2:
                compute_j2_numba(arr, n, oblate_indices, oblate_j2, oblate_req, oblate_poles, oblate_masses)
            if has_gr:
                compute_gr_1pn_numba(arr, n, phys_star_idx, phys_star_mass, C_AU_YR)

        sim.additional_forces = additional_forces_callback
        if has_j2:
            print(f"[J2] Attached J2 precession for {len(oblate_physics_list)} oblate bodies")
        if has_gr:
            print(f"[GR] Attached 1PN relativistic precession (star index {phys_star_idx})")
        sim.force_is_velocity_dependent = 1  # GR correction depends on velocity

    num_bodies = len(sim.particles)
    
    shared_state = {
        "lock": threading.Lock(),
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
    }
    
    for i in range(num_bodies):
        p = sim.particles[i]
        shared_state["pos"][i] = (p.x, p.z, -p.y)
        shared_state["vel"][i] = (p.vx, p.vz, -p.vy)
        shared_state["mass"][i] = p.m
        
    # Build initial parent_indices using name-to-index dict
    parent_indices = np.full(num_bodies, -1, dtype=np.int32)
    for i, body in enumerate(bodies_data):
        if 'parentId' in body:
            parent_indices[i] = name_to_idx[body['parentId']]
            
    # Calculate initial hierarchy dynamically so first frame is fully valid
    parent_indices = update_hierarchy(sim, num_bodies, parent_indices)
    tree_indices_init, tree_depths_init = build_tree_order(parent_indices, num_bodies)
    shared_state["parent_indices"][:] = parent_indices
    shared_state["tree_indices"][:] = tree_indices_init
    shared_state["tree_depths"][:] = tree_depths_init
        
    running = [True]
    physics_thread = threading.Thread(target=physics_loop, args=(sim, num_bodies, shared_state, time_ctrl, running), daemon=True)
    physics_thread.start()

    # 2. INITIALIZE GRAPHICS (GLFW & ModernGL)
    # Register error callback before init for diagnostic messages
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

    # Make the OpenGL context current BEFORE any GL calls (including ImGui init)
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # VSync OFF
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST) 

    # Initialize ImGui (requires an active GL context)
    global impl
    imgui.create_context()
    impl = GlfwRenderer(window, attach_callbacks=False)
    
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_window_size_callback(window, resize_callback)
    glfw.set_key_callback(window, key_callback)

    # 3. SHADERS (SPHERES)
    fov_factor = 1.0 / math.tan(math.radians(FOV_DEG / 2.0))

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
    };
    uniform float screen_height;
    uniform float fov_factor;
    
    out vec3 f_color;
    out vec3 f_world_pos;
    out vec3 f_normal;
    out float f_is_star;
    out float f_clip_z;
    void main() {
        f_color = in_color;
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
    };
    
    // Ring shadow planes (sphere-only)
    uniform vec3 u_ring_center[MAX_RING_PLANES];
    uniform vec3 u_ring_normal[MAX_RING_PLANES];
    uniform vec3 u_ring_params[MAX_RING_PLANES];
    uniform int u_num_ring_planes;
    
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
                    float ring_shadow = inner_fade * outer_fade * opacity;
                    shadow *= (1.0 - ring_shadow);
                }
            }
            
            diffuse *= shadow;
            
            // Pure black dark side (no ambient)
            out_color = vec4(f_color * diffuse, 1.0);
        }
        // Logarithmic depth — C constant improves near-field precision
        gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    }
    """
    prog_spheres = ctx.program(vertex_shader=sphere_vertex_shader, fragment_shader=sphere_fragment_shader)

    # 4. SHADERS (GPU ORBIT LINES — orbital elements computed on GPU via SSBO)
    orbit_vertex_shader = """
    #version 460 core
    // Orbital element data stored in SSBO — doubles for precision
    struct OrbitData {
        dvec4 d0;  // e_hat.xyz, sl_p
        dvec4 d1;  // q_hat.xyz, e_mag
        dvec4 d2;  // bary_rel.xyz, cos_theta0
        dvec4 d3;  // color.rgb, sin_theta0
    };
    
    layout(std430, binding = 0) buffer OrbitBuffer {
        OrbitData orbits[];
    };
    
    uniform mat4 projection;
    uniform mat4 view_rot;
    uniform dvec4 u_cam_pos_double;
    
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
        
        float t = float(gl_VertexID) / float(ORBIT_RES - 1);
        float angle_B = t * 2.0 * 3.14159265358979;
        
        dvec3 pos;
        
        if (e_mag < 0.999) {
            // Elliptical orbit: step Eccentric Anomaly for uniform vertex spacing and linear fade
            float e_f = float(e_mag);
            float cos_A_f = float(cos_A);
            float sin_A_f = float(sin_A);
            
            float E0 = atan(sqrt(1.0 - e_f * e_f) * sin_A_f, cos_A_f + e_f);
            float E = E0 + angle_B;
            
            double cos_E = double(cos(E));
            double sin_E = double(sin(E));
            
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
            double r = (denom > 1e-6lf) ? sl_p / denom : 1e9lf;
            
            pos = bary_rel + (cos_a * e_hat + sin_a * q_hat) * r;
        }
        
        dvec3 eye_pos = pos - u_cam_pos_double.xyz;
        
        f_color = vec4(color * 0.4, t);
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
        out_color = f_color;
        gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    }
    """
    prog_gpu_orbits = ctx.program(vertex_shader=orbit_vertex_shader, fragment_shader=orbit_fragment_shader)

    # 4b. SHADERS (PLANET RINGS) — with u_body_offset for static VAO rendering
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
    };
    
    // Per-ring-body uniforms
    uniform vec3 u_host_planet_pos;
    uniform float u_host_planet_radius;
    uniform vec3 u_camera_pos;
    uniform vec4 u_host_planet_pole_obl;
    
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
        float dust_reflect = dusty_phase * f_scatter * f_color.a * 10.0;
        float dust_transmit = dusty_phase * f_scatter * f_color.a * 10.0; 
        
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
        
        float illumination = direct_illum + planetshine;
        illumination = max(illumination, 0.02); // ensure a tiny isotropic ambient component so unlit side/shadow is not complete black
        
        // Dynamic Alpha for dusty rings (like E-ring)
        float base_alpha = f_color.a;
        // Suppress front-lit alpha based on f_scatter so E-ring is virtually invisible from front
        float front_suppress = 1.0 - f_scatter * 0.85 * same_side;
        base_alpha *= front_suppress;
        
        // Boost alpha when backlit - dust_transmit already factors in f_color.a to preserve gradients.
        float scattered_alpha_boost = dust_transmit * shadow * is_backlit * 1.0;
        float final_alpha = clamp(base_alpha + scattered_alpha_boost, 0.0, 1.0);
        
        // Use the actual ring color for backlit scattering instead of a hardcoded blue tint
        vec3 tinted_color = f_color.rgb;
        
        // Find max channel to see how much we can scale illumination without shifting hue (turning white/cyan)
        float max_c = max(tinted_color.r, max(tinted_color.g, tinted_color.b));
        float max_allowed_illum = 1.0 / max(max_c, 0.001);
        
        // Clamp illumination so it perfectly peaks at max pure blue without clipping channels unevenly
        float safe_illum = min(illumination, max_allowed_illum);
        
        out_color = vec4(tinted_color * safe_illum, final_alpha);
        gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0)) / log2(u_depth_C * u_far + 1.0);
    }
    """
    prog_rings = ctx.program(vertex_shader=ring_vertex_shader, fragment_shader=ring_fragment_shader)

    # 4c. SHADERS (ATMOSPHERE) — per-body raymarching for atmospheric scattering
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
    };

    uniform vec3 u_body_offset;
    uniform float u_atmo_radius_au;

    out vec3 f_world_pos;
    out float f_clip_z;

    void main() {
        vec3 world_pos = in_position * u_atmo_radius_au + u_body_offset;
        f_world_pos = world_pos;
        gl_Position = projection * view * vec4(world_pos, 1.0);
        f_clip_z = gl_Position.w;
    }
    """
    atmo_fragment_shader = """
    #version 460 core
    #define MAX_CASTERS 64
    #define PI 3.14159265358979

    in vec3 f_world_pos;
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
    };

    uniform vec3  u_body_offset;
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

    out vec4 out_color;

    vec3 toSphericalSpace(vec3 p, vec4 pole_obl) {
        float f = pole_obl.w;
        if (f == 0.0) return p;
        vec3 pole = pole_obl.xyz;
        float h = dot(p, pole);
        vec3 p_perp = p - h * pole;
        return p_perp + (h / (1.0 - f)) * pole;
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
        // 1. Setup: convert to planet-local km coordinates
        vec3 planet_center_render = u_body_offset;
        vec3 cam_local_au = u_camera_pos - planet_center_render;
        vec3 frag_local_au = f_world_pos - planet_center_render;

        vec3 cam_local = cam_local_au * u_au_to_km;
        vec3 frag_local = frag_local_au * u_au_to_km;

        vec3 ray_origin = cam_local;
        vec3 ray_dir = normalize(frag_local - cam_local);

        vec3 ray_origin_sph = toSphericalSpace(ray_origin, u_pole_obl);
        vec3 ray_dir_sph = toSphericalSpace(ray_dir, u_pole_obl);

        // 2. Ray-sphere intersections (planet-local km)
        vec2 t_atmo = raySphereIntersect(ray_origin_sph, ray_dir_sph, u_atmo_radius_km);
        if (t_atmo.x > t_atmo.y) discard;

        vec2 t_planet = raySphereIntersect(ray_origin_sph, ray_dir_sph, u_planet_radius_km);

        float t_start = max(t_atmo.x, 0.0);
        float t_end = t_atmo.y;
        if (t_planet.x > 0.0 && t_planet.x < t_end) {
            t_end = t_planet.x;
        }

        if (t_start >= t_end) discard;

        // 3. Sun direction (in km-space)
        vec3 sun_pos_local = (u_star_pos_radius.xyz - planet_center_render) * u_au_to_km;
        vec3 sun_dir = normalize(sun_pos_local);
        vec3 sun_dir_sph = toSphericalSpace(sun_dir, u_pole_obl);

        // Scattering coefficients: convert m^-1 to km^-1
        vec3 beta_R = u_beta_rayleigh * 1000.0;
        vec3 beta_M = vec3(u_beta_mie) * 1000.0;
        vec3 beta_A = u_beta_absorption * 1000.0;
        float mie_ext_factor = 1.0 / 0.9;

        // 4. Primary ray march
        float step_size = (t_end - t_start) / float(u_num_samples);
        vec3 total_rayleigh = vec3(0.0);
        vec3 total_mie = vec3(0.0);
        float od_rayleigh = 0.0;
        float od_mie = 0.0;

        for (int i = 0; i < u_num_samples; i++) {
            float t = t_start + (float(i) + 0.5) * step_size;
            vec3 sample_pos = ray_origin + t * ray_dir;
            vec3 sample_pos_sph = ray_origin_sph + t * ray_dir_sph;
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

            // 6. Eclipse shadow from caster bodies
            vec3 sample_render = sample_pos / u_au_to_km + planet_center_render;
            vec3 sample_to_star = u_star_pos_radius.xyz - sample_render;
            float dist_sample_star = length(sample_to_star);
            vec3 L = sample_to_star / dist_sample_star;

            float eclipse_shadow = 1.0;
            for (int k = 0; k < u_num_casters; k++) {
                vec3 caster_pos = u_casters[k].xyz;
                float caster_r = u_casters[k].w;
                vec3 s_to_c = caster_pos - sample_render;
                float proj = dot(s_to_c, L);
                if (proj <= 0.0 || proj >= dist_sample_star) continue;

                vec3 perp_vec = s_to_c - proj * L;
                float perp = length(perp_vec);

                float r_star_proj = u_star_pos_radius.w * proj / dist_sample_star;
                float r_penumbra = caster_r + r_star_proj;
                float r_umbra = abs(caster_r - r_star_proj);
                float max_occ = (caster_r >= r_star_proj) ? 1.0
                              : (caster_r * caster_r) / (r_star_proj * r_star_proj);
                float occ = max_occ * (1.0 - smoothstep(r_umbra, r_penumbra, perp));
                eclipse_shadow *= (1.0 - occ);
            }

            // 7. Accumulate in-scattered light
            vec3 tau = beta_R * (od_rayleigh + od_light_R)
                     + beta_M * mie_ext_factor * (od_mie + od_light_M)
                     + beta_A * (od_rayleigh + od_light_R);
            vec3 attenuation = exp(-tau) * eclipse_shadow;

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

        // Tone mapping to prevent blowout
        scattered = 1.0 - exp(-scattered);

        // 11. Premultiplied alpha output
        float avg_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
        out_color = vec4(scattered, 1.0 - avg_transmittance);

        // Logarithmic depth
        gl_FragDepth = log2(max(1e-6, u_depth_C * f_clip_z + 1.0))
                     / log2(u_depth_C * u_far + 1.0);
    }
    """
    prog_atmo = ctx.program(vertex_shader=atmo_vertex_shader, fragment_shader=atmo_fragment_shader)

    # Pre-compute ring meshes (in rendering coordinates, centered at origin)
    ring_precomputed = []  # list of dicts with verts, indices, normal, color, body_idx
    RING_SEGMENTS = 128
    for body_idx, rings_data, pole_render, body_radius_au in ring_bodies:
        # Build equatorial-to-ecliptic rotation for ring orientation
        # The pole is already in rendering coords — build a rotation matrix
        # that maps (x, y=up, z) to (rendering frame with y=pole)
        pole_n = pole_render / np.linalg.norm(pole_render)
        # Choose a perpendicular vector
        ref = np.array([0., 0., 1.])
        tangent = np.cross(pole_n, ref)
        if np.linalg.norm(tangent) < 1e-10:
            ref = np.array([1., 0., 0.])
            tangent = np.cross(pole_n, ref)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(pole_n, tangent)
        # R maps: x -> tangent, y -> pole_n, z -> bitangent
        R_ring = np.column_stack([tangent, pole_n, bitangent])
        
        for ring_seg in rings_data:
            inner_r = ring_seg['inner'] * body_radius_au
            outer_r = ring_seg['outer'] * body_radius_au
            r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
            r_opacity = ring_seg.get('opacity', 1.0)
            r_scatter = ring_seg.get('scatter', 0.0)
            r_asymmetry = ring_seg.get('asymmetry', 0.7)
            
            # Generate annulus in equatorial plane (xz plane, y=0)
            RADIAL_SUBDIVISIONS = 32
            # Pre-sort gradient for vectorized sampling
            sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
            
            # Generate vertex grid (vectorized — single matrix multiply for all vertices)
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
            
            # Vectorized gradient sampling with np.interp
            flat_p = p_grid.ravel()
            if sorted_gradient:
                grad_p = np.array([g['p'] for g in sorted_gradient])
                grad_a = np.array([g['a'] for g in sorted_gradient])
                alpha_mults = np.interp(flat_p, grad_p, grad_a).astype('f4')
            else:
                alpha_mults = np.ones(n_verts, dtype='f4')
            
            colors = np.zeros((n_verts, 4), dtype='f4')
            colors[:, 0] = r_color[0]
            colors[:, 1] = r_color[1]
            colors[:, 2] = r_color[2]
            colors[:, 3] = r_opacity * alpha_mults
            
            # Generate triangle indices (vectorized)
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
                'verts': np.array(verts, dtype='f4'),  # (N, 3) in rendering coords, relative to body center
                'indices': np.array(indices, dtype='i4'),
                'normal': pole_n.astype('f4'),
                'colors': np.array(colors, dtype='f4'),  # (N, 4) in rendering coords
                'inner_r': inner_r,   # inner radius in AU
                'outer_r': outer_r,   # outer radius in AU
                'opacity': r_opacity,
                'scatter': r_scatter,
                'asymmetry': r_asymmetry,
            })

    # === Create per-body STATIC ring VAOs (uploaded once, never rebuilt) ===
    ring_render_groups = []  # list of {body_idx, vao, num_indices}
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
            # Pack: position(3) + normal(3) + color(4) + scatter(1) + asymmetry(1) per vertex
            ring_packed = np.zeros((n_v, 12), dtype='f4')
            ring_packed[:, 0:3] = ring['verts']  # relative to body center
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

    # 5. PREPARE BUFFERS & VAOS — LOD system: low-poly and high-poly sphere meshes
    INSTANCE_FLOATS = 13  # offset(3) + color(3) + radius(1) + min_size(1) + is_star(1) + pole(3) + oblateness(1)

    mesh_lo_verts, mesh_lo_idx = create_sphere_mesh(segments=8, rings=4)
    mesh_hi_verts, mesh_hi_idx = create_sphere_mesh(segments=64, rings=32)

    vbo_lo = ctx.buffer(mesh_lo_verts.tobytes())
    ibo_lo = ctx.buffer(mesh_lo_idx.tobytes())
    vbo_hi = ctx.buffer(mesh_hi_verts.tobytes())
    ibo_hi = ctx.buffer(mesh_hi_idx.tobytes())

    vbo_instances_lo = ctx.buffer(reserve=num_bodies * INSTANCE_FLOATS * 4)
    vbo_instances_hi = ctx.buffer(reserve=num_bodies * INSTANCE_FLOATS * 4)

    inst_fmt = '3f 3f 1f 1f 1f 3f 1f/i'
    inst_names = ('in_offset', 'in_color', 'in_radius', 'in_min_size', 'in_is_star', 'in_pole', 'in_oblateness')

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

    # Atmosphere VAO — reuses high-poly sphere mesh, skips normals with padding
    vao_atmo = ctx.vertex_array(
        prog_atmo,
        [(vbo_hi, '3f 12x', 'in_position')],
        index_buffer=ibo_hi
    )

    # GPU orbit rendering: SSBO for orbital elements, empty VAO (positions computed in vertex shader)
    max_orbits = num_bodies * 2  # each body can have child orbit + parent reflex orbit
    orbit_ssbo = ctx.buffer(reserve=max_orbits * ORBIT_STRIDE_BYTES)
    orbit_ssbo.bind_to_storage_buffer(binding=0)
    vao_gpu_orbits = ctx.vertex_array(prog_gpu_orbits, [])

    # Shared scene UBO (binding=1, uploaded once per frame for sphere + ring shaders)
    UBO_SIZE = 2208  # std140: 2×mat4 + vec4 + 3×scalar + pad + 2×vec4[64]
    scene_ubo = ctx.buffer(reserve=UBO_SIZE)
    scene_ubo.bind_to_uniform_block(1)
    ubo_staging = np.zeros(UBO_SIZE // 4, dtype=np.float32)
    ubo_casters_int_view = ubo_staging[38:39].view(np.int32)

    # Sphere shader uniforms (shared uniforms now in SceneData UBO)
    uniform_screen_height = prog_spheres['screen_height']
    uniform_fov_factor = prog_spheres['fov_factor']

    # GPU orbit shader uniforms
    uniform_orbit_proj = prog_gpu_orbits['projection']
    uniform_orbit_view_rot = prog_gpu_orbits['view_rot']
    uniform_orbit_cam_pos = prog_gpu_orbits['u_cam_pos_double']
    uniform_orbit_far = prog_gpu_orbits['u_far']
    uniform_orbit_depth_C = prog_gpu_orbits['u_depth_C']

    # Ring shadow plane uniforms (for ring shadows on planet surfaces)
    uniform_ring_centers = prog_spheres['u_ring_center']
    uniform_ring_normals = prog_spheres['u_ring_normal']
    uniform_ring_params = prog_spheres['u_ring_params']
    uniform_num_ring_planes = prog_spheres['u_num_ring_planes']

    # Find star index and its radius for lighting
    star_idx = 0
    for i, b in enumerate(bodies_data):
        if b.get('type') == 'Star':
            star_idx = i
            break
    star_radius_au = visual_data[star_idx][3]  # radius in AU

    # Pre-extract body radii for eclipse shadow casters
    body_radii = np.array([v[3] for v in visual_data], dtype='f4')  # radii in AU

    # Ring shader uniforms (shared uniforms now in SceneData UBO)
    if ring_render_groups:
        u_ring_host_pos = prog_rings['u_host_planet_pos']
        u_ring_host_radius = prog_rings['u_host_planet_radius']
        u_ring_host_pole_obl = prog_rings['u_host_planet_pole_obl']
        u_ring_camera_pos = prog_rings['u_camera_pos']
        u_ring_body_offset = prog_rings['u_body_offset']

    # Atmosphere shader uniforms
    if atmo_bodies:
        u_atmo_body_offset = prog_atmo['u_body_offset']
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
        u_atmo_sun_intensity = prog_atmo['u_sun_intensity']
        u_atmo_camera_pos = prog_atmo['u_camera_pos']
        u_atmo_num_samples = prog_atmo['u_num_samples']
        u_atmo_num_light_samples = prog_atmo['u_num_light_samples']
        u_atmo_pole_obl = prog_atmo['u_pole_obl']

    G = 4.0 * math.pi**2 

    # === Pre-compute arrays for vectorized per-frame operations ===
    visual_arr = np.array(visual_data, dtype='f4')  # (num_bodies, 5): color(3) + radius(1) + min_px(1)
    is_star_arr = np.zeros(num_bodies, dtype='f4')
    is_star_arr[star_idx] = 1.0
    # Non-star indices for caster packing (computed once, never changes)
    non_star_mask = np.ones(num_bodies, dtype=bool)
    non_star_mask[star_idx] = False
    non_star_indices = np.where(non_star_mask)[0][:64]
    n_casters_fixed = len(non_star_indices)

    # Pre-allocate per-frame buffers to reduce GC pressure
    inst_data_lo = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
    inst_data_hi = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
    # Orbit SSBO buffer: 16 doubles = 128 bytes per orbit
    orbit_data_buf = np.zeros((max_orbits, 16), dtype='f8')
    subsys_pos_buf = np.zeros((num_bodies, 3), dtype='f8')
    subsys_vel_buf = np.zeros((num_bodies, 3), dtype='f8')
    subsys_mass_buf = np.zeros(num_bodies, dtype='f8')
    caster_data_buf = np.zeros((64, 4), dtype='f4')
    caster_poles_obl_buf = np.zeros((64, 4), dtype='f4')
    ring_centers_buf = np.zeros((16, 3), dtype='f4')
    ring_normals_buf = np.zeros((16, 3), dtype='f4')
    ring_params_buf = np.zeros((16, 3), dtype='f4')
    all_instances = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')

    # Pre-allocated snapshot buffers (avoid per-frame allocation under lock)
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
    jump_date = [2026, 6, 3]
    scrub_index = [0]

    # 6. MAIN LOOP
    while not glfw.window_should_close(window):
        now = time.time()
        dt_render = min(now - last_render_time, 0.1)
        last_render_time = now
        
        glfw.poll_events()
        
        # --- IMGUI FRAME START ---
        impl.process_inputs()
        imgui.new_frame()
        
        # Skip rendering if window is minimized (zero-size framebuffer)
        if fb_width <= 0 or fb_height <= 0:
            imgui.end_frame()
            continue
        
        ctx.viewport = (0, 0, fb_width, fb_height)
        ctx.clear(0.02, 0.02, 0.03, 1.0) 

        # Fetch physics state snapshot under lock (pre-allocated, no allocation)
        with _perf("snapshot"), shared_state["lock"]:
            np.copyto(pos_snap, shared_state["pos"])
            np.copyto(vel_snap, shared_state["vel"])
            np.copyto(mass_snap, shared_state["mass"])
            np.copyto(parent_snap, shared_state["parent_indices"])
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

        # Smooth camera variables
        lerp_factor = 1.0 - math.exp(-15.0 * dt_render)
        camera["distance_actual"] += (camera["distance"] - camera["distance_actual"]) * lerp_factor
        camera["yaw_actual"] += (camera["yaw"] - camera["yaw_actual"]) * lerp_factor
        camera["pitch_actual"] += (camera["pitch"] - camera["pitch_actual"]) * lerp_factor
        camera["target_offset"] *= math.exp(-10.0 * dt_render)

        # === BARYCENTER COMPUTATION (Numba-accelerated) ===
        # Done before camera tracking so barycenter mode has zero frame lag
        compute_barycenters(pos_snap_render, vel_snap_render, mass_snap, parent_snap,
                            subsys_pos_buf, subsys_vel_buf, subsys_mass_buf)

        # Camera-relative rendering: compute offset in float64, then cast to float32
        if camera["tracking_idx"] is not None:
            track_idx = camera["tracking_idx"]
            if camera["tracking_mode"] == "barycenter":
                # Use subsystem barycenter position (computed last frame — 1 frame lag is invisible)
                base_pos = subsys_pos_buf[track_idx].copy()
            else:
                base_pos = pos_snap_render[track_idx].copy()
            camera["target"] = base_pos.copy()  # Keep updated for when tracking breaks
        else:
            base_pos = np.array(camera["target"], dtype='f8')
            
        cam_origin = base_pos + camera["target_offset"]

        # Build children_map (cached — only rebuild when hierarchy changes)
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

        # Determine focused system for LOD
        focused_root = -1
        if camera["tracking_idx"] is not None:
            # Walk up the hierarchy to find the root planet (child of star or root itself)
            body_idx = camera["tracking_idx"]
            while True:
                p = int(parent_snap[body_idx])
                if p == -1 or p == star_idx:  # reached root or star
                    focused_root = body_idx
                    break
                body_idx = p
        
        # Build focused mask using iterative DFS (avoids set + list comprehension)
        focused_mask[:] = False
        if focused_root >= 0:
            stack = [focused_root]
            while stack:
                node = stack.pop()
                focused_mask[node] = True
                if node in children_map:
                    stack.extend(children_map[node])
        
        # === VECTORIZED INSTANCE DATA PACKING ===
        with _perf("pack_instances"):
            pos_rel_all = (pos_snap_render - cam_origin).astype('f4')  # (N, 3) camera-relative positions
        all_instances[:, 0:3] = pos_rel_all
        all_instances[:, 3:8] = visual_arr[:, 0:5]  # color(3) + radius(1) + min_px(1)
        all_instances[:, 8] = is_star_arr
        all_instances[:, 9:12] = visual_arr[:, 5:8] # pole(3)
        all_instances[:, 12] = visual_arr[:, 8]     # oblateness(1)
        
        n_hi = int(np.sum(focused_mask))
        n_lo = num_bodies - n_hi
        if n_hi > 0:
            inst_data_hi[:n_hi] = all_instances[focused_mask]
        if n_lo > 0:
            inst_data_lo[:n_lo] = all_instances[~focused_mask]

        # === COMPUTE ORBITAL ELEMENTS FOR GPU (Numba batch — no Python loops) ===
        n_orbits = compute_all_orbits_batch(
            pos_snap_render, vel_snap_render, mass_snap, parent_snap,
            subsys_pos_buf, subsys_vel_buf, subsys_mass_buf,
            visual_colors_f8, cam_origin, G, max_orbits, orbit_data_buf)

        # Write instance buffers
        if n_lo > 0:
            vbo_instances_lo.write(inst_data_lo[:n_lo].tobytes())
        if n_hi > 0:
            vbo_instances_hi.write(inst_data_hi[:n_hi].tobytes())
        
        # Write orbit SSBO
        if n_orbits > 0:
            orbit_ssbo.write(orbit_data_buf[:n_orbits].tobytes())

        yaw_rad, pitch_rad = math.radians(camera["yaw_actual"]), math.radians(camera["pitch_actual"])
        
        # 64-bit precision for orbit shader
        cam_pos_f8 = np.array([-math.cos(yaw_rad) * math.cos(pitch_rad) * camera["distance_actual"],
                               -math.sin(pitch_rad) * camera["distance_actual"],
                               -math.sin(yaw_rad) * math.cos(pitch_rad) * camera["distance_actual"]], dtype='f8')
                               
        # 32-bit for regular view matrix
        cam_pos = cam_pos_f8.astype('f4')
        view = matrix44.create_look_at(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], dtype='f4')
        near = max(camera["distance_actual"] * 0.0001, 1e-9)
        max_body_dist = float(np.max(np.linalg.norm(pos_rel_all, axis=1)))
        far = max(camera["distance_actual"] + max_body_dist * 2.0 + 1.0, 1.0)
        depth_C = 1.0 / max(near, 1e-12)
        projection = matrix44.create_perspective_projection_matrix(FOV_DEG, fb_width / max(fb_height, 1), near, far, dtype='f4')
        
        # Star position for lighting (camera-relative)
        star_cam_rel = (pos_snap[star_idx] - cam_origin).astype('f4')
        
        # === VECTORIZED CASTER PACKING ===
        caster_data_buf[:n_casters_fixed, 0:3] = pos_rel_all[non_star_indices]
        caster_data_buf[:n_casters_fixed, 3] = body_radii[non_star_indices]
        if n_casters_fixed < 64:
            caster_data_buf[n_casters_fixed:] = 0
            
        caster_poles_obl_buf[:n_casters_fixed, 0:3] = all_instances[non_star_indices, 9:12]
        caster_poles_obl_buf[:n_casters_fixed, 3] = all_instances[non_star_indices, 12]
        if n_casters_fixed < 64:
            caster_poles_obl_buf[n_casters_fixed:] = 0
        
        # Pack ring planes for sphere shadows
        n_ring_planes = 0
        ring_centers_buf[:] = 0
        ring_normals_buf[:] = 0
        ring_params_buf[:] = 0
        
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

        # Pack and upload shared scene UBO (single write for sphere + ring shaders)
        ubo_staging[0:16] = projection.ravel()
        ubo_staging[16:32] = view.ravel()
        ubo_staging[32:35] = star_cam_rel
        ubo_staging[35] = star_radius_au
        ubo_staging[36] = far
        ubo_staging[37] = depth_C
        ubo_casters_int_view[0] = n_casters_fixed
        ubo_staging[40:296] = caster_data_buf.ravel()
        ubo_staging[296:552] = caster_poles_obl_buf.ravel()
        scene_ubo.write(ubo_staging)
        
        # Sphere-only uniforms
        uniform_screen_height.value = fb_height
        uniform_fov_factor.value = fov_factor
        uniform_num_ring_planes.value = n_ring_planes
        if n_ring_planes > 0:
            uniform_ring_centers.write(ring_centers_buf.tobytes())
            uniform_ring_normals.write(ring_normals_buf.tobytes())
            uniform_ring_params.write(ring_params_buf.tobytes())
        
        view_rot = view.copy()
        view_rot[3, 0:3] = 0.0

        # Upload orbit shader uniforms
        uniform_orbit_proj.write(projection.tobytes())
        uniform_orbit_view_rot.write(view_rot.tobytes())
        
        cam_pos_dvec4 = np.array([cam_pos_f8[0], cam_pos_f8[1], cam_pos_f8[2], 0.0], dtype='f8')
        uniform_orbit_cam_pos.write(cam_pos_dvec4.tobytes())
        
        uniform_orbit_far.value = far
        uniform_orbit_depth_C.value = depth_C

        # === RENDER: Sphere bodies (opaque — drawn first for early-Z) ===
        with _perf("render_spheres"):
          ctx.enable(moderngl.CULL_FACE)
          if n_lo > 0:
              vao_lo.render(moderngl.TRIANGLES, instances=n_lo)
          if n_hi > 0:
              vao_hi.render(moderngl.TRIANGLES, instances=n_hi)
          ctx.disable(moderngl.CULL_FACE)

        # Enable blending once for all transparent passes (orbits + rings)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # === RENDER: Orbit lines (GPU-computed from orbital elements in SSBO) ===
        with _perf("render_orbits"):
          if show_orbits and n_orbits > 0:
              vao_gpu_orbits.render(moderngl.LINE_LOOP, vertices=ORBIT_RESOLUTION, instances=n_orbits)

        # === RENDER: Planet rings (static VAOs, per-body offset uniform only) ===
        with _perf("render_rings"):
          if ring_render_groups:
              u_ring_camera_pos.write(cam_pos.tobytes())
              ctx.depth_mask = False
              for group in ring_render_groups:
                  bi = group['body_idx']
                  body_pos_rel = pos_rel_all[bi]
                  u_ring_body_offset.write(body_pos_rel.tobytes())
                  u_ring_host_pos.write(body_pos_rel.tobytes())
                  u_ring_host_radius.value = float(body_radii[bi])
                  u_ring_host_pole_obl.value = (
                      float(all_instances[bi, 9]),
                      float(all_instances[bi, 10]),
                      float(all_instances[bi, 11]),
                      float(all_instances[bi, 12])
                  )
                  group['vao'].render(moderngl.TRIANGLES, vertices=group['num_indices'])
              ctx.depth_mask = True

        ctx.disable(moderngl.BLEND)

        # === RENDER: Atmospheres (transparent — premultiplied alpha blending) ===
        with _perf("render_atmo"):
          if atmo_bodies:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.enable(moderngl.CULL_FACE)   # cull back faces (camera outside atmosphere)
            ctx.depth_mask = False

            u_atmo_camera_pos.write(cam_pos.tobytes())
            u_atmo_au_to_km.value = AU_TO_KM

            for atmo in atmo_bodies:
                bi = atmo['body_idx']
                body_pos_rel = pos_rel_all[bi]

                # LOD: skip if atmosphere is subpixel
                dist_to_body = float(np.linalg.norm(body_pos_rel - cam_pos))
                apparent_px = (atmo['atmo_radius_au'] / max(dist_to_body, 1e-12)) * fb_height * fov_factor
                if apparent_px < 2.0:
                    continue

                # Adaptive sample count
                if apparent_px < 50:
                    n_samples, n_light = 8, 4
                elif apparent_px < 200:
                    n_samples, n_light = 16, 6
                else:
                    n_samples, n_light = 32, 8

                # Sun intensity scaled by inverse-square distance
                sun_dist_au = float(np.linalg.norm(
                    pos_snap[bi] - pos_snap[star_idx]
                ))
                scaled_intensity = atmo['intensity'] / max(sun_dist_au * sun_dist_au, 0.01)

                # Upload per-body uniforms
                u_atmo_body_offset.write(body_pos_rel.astype('f4').tobytes())
                u_atmo_radius_au_uniform.value = float(atmo['atmo_radius_au'])
                u_atmo_planet_radius.value = float(atmo['planet_radius_km'])
                u_atmo_atmo_radius.value = float(atmo['atmo_radius_km'])
                u_atmo_beta_rayleigh.write(atmo['beta_rayleigh'].tobytes())
                u_atmo_h_rayleigh.value = atmo['h_rayleigh']
                u_atmo_beta_mie.value = atmo['beta_mie']
                u_atmo_h_mie.value = atmo['h_mie']
                u_atmo_mie_g.value = atmo['mie_g']
                u_atmo_beta_absorption.write(atmo['beta_absorption'].tobytes())
                u_atmo_sun_intensity.value = scaled_intensity
                u_atmo_num_samples.value = n_samples
                u_atmo_num_light_samples.value = n_light
                u_atmo_pole_obl.value = (
                    float(all_instances[bi, 9]),
                    float(all_instances[bi, 10]),
                    float(all_instances[bi, 11]),
                    float(all_instances[bi, 12])
                )

                vao_atmo.render(moderngl.TRIANGLES)

            ctx.depth_mask = True
            ctx.disable(moderngl.CULL_FACE)
            ctx.disable(moderngl.BLEND)
        imgui.set_next_window_position(fb_width - 250, 20, imgui.ONCE)
        imgui.set_next_window_size(230, min(500, fb_height - 40), imgui.ONCE)
        imgui.begin("Simulation Controls")
        
        imgui.text("Current Date:")
        imgui.text(f"{cur_y:04d}-{cur_m:02d}-{cur_d:02d} {cur_h:02d}:{cur_mn:02d}")
        
        if tl_active and tl_prog < 1.0:
            imgui.text("Rendering Timeline...")
            imgui.progress_bar(tl_prog)
            if imgui.button("Cancel"):
                time_ctrl["cancel_render"] = True
        elif is_scrubbing:
            imgui.separator()
            imgui.text("Timeline Navigation")
            max_idx = max(0, len(tl_times) - 1)
            changed, scrub_index[0] = imgui.slider_int("##Scrub", scrub_index[0], 0, max_idx, "")
            
            if imgui.button("Resume Here"):
                time_ctrl["sync_t"] = tl_times[scrub_index[0]]
                time_ctrl["paused"] = False
            imgui.same_line()
            if imgui.button("Cancel"):
                with shared_state["lock"]:
                    shared_state["timeline_active"] = False
                    
        else:
            imgui.separator()
            imgui.text("Jump in Time")
            _, jump_date[0] = imgui.input_int("Year", jump_date[0])
            _, jump_date[1] = imgui.input_int("Month", jump_date[1])
            _, jump_date[2] = imgui.input_int("Day", jump_date[2])
            jump_date[1] = max(1, min(12, jump_date[1]))
            jump_date[2] = max(1, min(31, jump_date[2]))
            
            if imgui.button("Render Timeline"):
                time_ctrl["target_t"] = sim_time_from_date(jump_date[0], jump_date[1], jump_date[2])
                time_ctrl["render_timeline"] = True
        
        imgui.separator()
        imgui.text("Visual Settings")
        _, show_orbits = imgui.checkbox("Show Orbits", show_orbits)
        imgui.separator()
        imgui.text("Time Controls")
        _, time_ctrl["paused"] = imgui.checkbox("Paused", time_ctrl["paused"])
        imgui.text(f"Speed: {format_time_speed(time_ctrl['multiplier'])}")
        _, time_ctrl["multiplier"] = imgui.slider_float("##speed", time_ctrl["multiplier"], 1.0, 1e9, "", flags=imgui.SLIDER_FLAGS_LOGARITHMIC)
        if imgui.button("Reset Speed"): time_ctrl["multiplier"] = 1.0
        imgui.end()

        # === SYSTEM HIERARCHY WINDOW ===
        imgui.set_next_window_position(fb_width - 250, 420, imgui.ONCE)
        imgui.set_next_window_size(230, min(400, fb_height - 440), imgui.ONCE)
        imgui.begin("System Hierarchy")
        
        for k in range(len(tree_indices_snap)):
            idx = int(tree_indices_snap[k])
            depth = int(tree_depths_snap[k])
            indent = depth * 15
            if indent > 0: imgui.indent(indent)
            # Highlight: inspected body gets highlight, tracked body gets different marker
            is_inspected = (camera["inspected_idx"] == idx)
            label = f"{bodies_data[idx]['name']}"
            if camera["tracking_idx"] == idx:
                label += " *"  # asterisk marks tracked body
            if imgui.selectable(label, is_inspected)[0]:
                if camera["inspected_idx"] == idx:
                    # Toggle: body -> barycenter -> close
                    if not camera["inspect_bary"]:
                        camera["inspect_bary"] = True
                    else:
                        camera["inspected_idx"] = None
                        camera["inspect_bary"] = False
                else:
                    camera["inspected_idx"] = idx
                    camera["inspect_bary"] = False
            if indent > 0: imgui.unindent(indent)
        imgui.end()

        # === BODY INSPECTOR WINDOW ===
        insp_idx = camera["inspected_idx"]
        if insp_idx is not None and 0 <= insp_idx < num_bodies:
            inspect_bary = camera["inspect_bary"]
            body_info = bodies_data[insp_idx]
            body_name = body_info['name']
            parent_idx = int(parent_snap[insp_idx])
            
            # Window title
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
                # --- Mode indicator ---
                if inspect_bary:
                    imgui.text_colored("Barycenter Mode", 0.6, 0.9, 1.0)
                else:
                    obj_type = body_info.get('type', 'Unknown')
                    imgui.text_colored(f"Type: {obj_type}", 0.7, 0.7, 0.7)
                
                # --- Physical Properties ---
                imgui.separator()
                imgui.text_colored("Physical Properties", 1.0, 0.85, 0.4)
                
                body_mass = mass_snap[insp_idx]
                body_r_km = body_info.get('r', 0.0) * 6371.0  # Earth radii to km
                
                if inspect_bary:
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
                
                # --- Orbital Elements (only if body has a parent) ---
                if parent_idx >= 0:
                    imgui.separator()
                    imgui.text_colored("Orbital Elements", 1.0, 0.85, 0.4)
                    
                    parent_name = bodies_data[parent_idx]['name']
                    imgui.text_colored(f"  (around {parent_name})", 0.5, 0.5, 0.5)
                    
                    # Convert render coords back to ecliptic for element computation
                    # Render: (sim_x, sim_z, -sim_y) → Ecliptic: (render_x, -render_z, render_y)
                    if inspect_bary:
                        # Barycenter position/velocity relative to parent
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
                    
                    # Convert to ecliptic (REBOUND) frame
                    ecl_rx, ecl_ry, ecl_rz = rel_r[0], -rel_r[2], rel_r[1]
                    ecl_vx, ecl_vy, ecl_vz = rel_v[0], -rel_v[2], rel_v[1]
                    
                    mu = G * (mass_snap[parent_idx] + orb_mass)
                    
                    oe_a, oe_e, oe_inc, oe_Omega, oe_omega, oe_nu, oe_M, oe_P = \
                        compute_keplerian_elements(ecl_rx, ecl_ry, ecl_rz,
                                                   ecl_vx, ecl_vy, ecl_vz, mu)
                    
                    # Distance and velocity
                    dist_au = math.sqrt(rel_r[0]**2 + rel_r[1]**2 + rel_r[2]**2)
                    vel_au_yr = math.sqrt(rel_v[0]**2 + rel_v[1]**2 + rel_v[2]**2)
                    vel_km_s = vel_au_yr * 4.7405  # AU/yr to km/s
                    
                    # Auto-select units: AU for large orbits, km for small
                    use_km = oe_a < 0.01  # moons
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
                    
                    # --- Derived Quantities ---
                    imgui.separator()
                    imgui.text_colored("Derived Quantities", 1.0, 0.85, 0.4)
                    
                    if oe_P > 0:
                        if oe_P < 730:  # less than 2 years
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
                    
                    # --- Precession Rates ---
                    if oe_a > 0 and oe_e < 1.0 and not inspect_bary:
                        imgui.separator()
                        imgui.text_colored("Precession (arcsec/cy)", 1.0, 0.85, 0.4)
                        
                        # GR 1PN apsidal precession: dω/dt = 3μ / (a·c²·(1-e²)) rad/orbit
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
                        
                        # J2 apsidal & nodal precession
                        j2_apsidal = 0.0
                        j2_nodal = 0.0
                        parent_j2 = bodies_data[parent_idx].get('J2', 0.0)
                        if parent_j2 > 0 and oe_P > 0:
                            parent_r_au = bodies_data[parent_idx].get('r', 0.0) * 0.00465
                            n_mean = 2.0 * math.pi / (oe_P / 365.25)  # rad/yr
                            if p_param > 0:
                                ratio2 = (parent_r_au / p_param) ** 2
                                j2_apsidal_rad_yr = 1.5 * n_mean * parent_j2 * ratio2
                                j2_nodal_rad_yr = -j2_apsidal_rad_yr * math.cos(math.radians(oe_inc))
                                j2_apsidal = j2_apsidal_rad_yr * 100.0 * (180.0/math.pi) * 3600.0
                                j2_nodal = j2_nodal_rad_yr * 100.0 * (180.0/math.pi) * 3600.0
                        
                        imgui.text(f"  J2 (apsidal):  {j2_apsidal:.2f}")
                        imgui.text(f"  J2 (nodal):    {j2_nodal:.2f}")
                
                # --- Track Buttons ---
                imgui.separator()
                
                is_tracking_this = (camera["tracking_idx"] == insp_idx)
                is_tracking_body = is_tracking_this and camera["tracking_mode"] == "body"
                is_tracking_bary = is_tracking_this and camera["tracking_mode"] == "barycenter"
                
                if is_tracking_body:
                    imgui.text_colored("Tracking Body", 0.3, 1.0, 0.3)
                elif imgui.button("Track Body"):
                    old_pos = pos_snap_render[camera["tracking_idx"]] if camera["tracking_idx"] is not None else camera["target"]
                    new_pos = pos_snap_render[insp_idx]
                    camera["target_offset"] += (old_pos - new_pos)
                    camera["tracking_idx"] = insp_idx
                    camera["tracking_mode"] = "body"
                
                imgui.same_line()
                
                if is_tracking_bary:
                    imgui.text_colored("Tracking Barycenter", 0.3, 1.0, 0.3)
                elif imgui.button("Track Barycenter"):
                    old_pos = pos_snap_render[camera["tracking_idx"]] if camera["tracking_idx"] is not None else camera["target"]
                    new_pos = subsys_pos_buf[insp_idx]
                    camera["target_offset"] += (old_pos - new_pos)
                    camera["tracking_idx"] = insp_idx
                    camera["tracking_mode"] = "barycenter"
                
                if is_tracking_this:
                    if imgui.button("Untrack"):
                        camera["target"] = camera["target"].copy()
                        camera["target_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                        camera["tracking_idx"] = None
                        camera["tracking_mode"] = "body"
                
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