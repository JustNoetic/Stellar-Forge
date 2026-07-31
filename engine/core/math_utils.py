import math
import numpy as np
from numba import njit
from engine.core.constants import *

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
def rotate_equatorial_to_ecliptic(pos, vel, pole_ecl):
    """Rotate equatorial coordinates to ecliptic coordinates given a pole vector."""
    px, py, pz = pole_ecl
    if abs(px) < 1e-8 and abs(py) < 1e-8 and pz > 0:
        return pos, vel
        
    N = np.array([-py, px, 0.0], dtype=np.float64)
    n_mag = np.linalg.norm(N)
    
    if n_mag < 1e-8:
        X = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        Y = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        Z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        X = N / n_mag
        Z = np.array([px, py, pz], dtype=np.float64)
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
        
    N = np.array([-py, px, 0.0], dtype=np.float64)
    n_mag = np.linalg.norm(N)
    
    if n_mag < 1e-8:
        X = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        Y = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        Z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        X = N / n_mag
        Z = np.array([px, py, pz], dtype=np.float64)
        Z = Z / np.linalg.norm(Z)
        Y = np.cross(Z, X)
        
    R = np.column_stack((X, Y, Z))
    R_inv = R.T
    return R_inv @ pos, R_inv @ vel



@njit(cache=True, nogil=True)
def get_cartesian_from_keplerian(parent_m, body_m, a, e, inc_deg, Omega_deg, omega_deg, M_deg):
    mu = G * (parent_m + body_m)
    inc = math.radians(inc_deg)
    Omega = math.radians(Omega_deg)
    omega = math.radians(omega_deg)
    M = math.radians(M_deg)
    return orbital_to_cartesian(a, e, inc, Omega, omega, M, mu)

@njit(cache=True)
def axis_angle_rotation(v, axis, theta):
    """Rotate vector v around unit vector axis by angle theta."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cross_prod = np.array([
        axis[1]*v[2] - axis[2]*v[1],
        axis[2]*v[0] - axis[0]*v[2],
        axis[0]*v[1] - axis[1]*v[0]
    ], dtype=np.float64)
    dot_prod = v[0]*axis[0] + v[1]*axis[1] + v[2]*axis[2]
    
    return np.array([
        v[0] * cos_t + cross_prod[0] * sin_t + axis[0] * dot_prod * (1.0 - cos_t),
        v[1] * cos_t + cross_prod[1] * sin_t + axis[1] * dot_prod * (1.0 - cos_t),
        v[2] * cos_t + cross_prod[2] * sin_t + axis[2] * dot_prod * (1.0 - cos_t)
    ], dtype=np.float64)

@njit(cache=True)
def vector_orbital_to_cartesian(a, e, n_vec, e_vec, M, mu):
    """Convert orbital elements to Cartesian pos/vel using normal and eccentricity unit vectors.
       n_vec is orbit normal (Z), e_vec is periapsis direction (X)."""
    E = kepler_solve(M, e)
    cos_E, sin_E = math.cos(E), math.sin(E)
    r = a * (1.0 - e * cos_E)
    
    cos_f = (cos_E - e) / (1.0 - e * cos_E)
    sin_f = math.sqrt(max(0.0, 1.0 - e*e)) * sin_E / (1.0 - e * cos_E)
    
    x_orb = r * cos_f
    y_orb = r * sin_f
    
    fac = math.sqrt(mu / max(a**3, 1e-30))
    denom = 1.0 - e * cos_E
    vx_orb = -fac * a * sin_E / denom
    vy_orb = fac * a * math.sqrt(max(0.0, 1.0 - e*e)) * cos_E / denom
    
    # Q vector is Y axis (n_vec x e_vec)
    q_vec = np.array([
        n_vec[1]*e_vec[2] - n_vec[2]*e_vec[1],
        n_vec[2]*e_vec[0] - n_vec[0]*e_vec[2],
        n_vec[0]*e_vec[1] - n_vec[1]*e_vec[0]
    ], dtype=np.float64)
    
    pos = np.array([
        x_orb * e_vec[0] + y_orb * q_vec[0],
        x_orb * e_vec[1] + y_orb * q_vec[1],
        x_orb * e_vec[2] + y_orb * q_vec[2]
    ], dtype=np.float64)
    
    vel = np.array([
        vx_orb * e_vec[0] + vy_orb * q_vec[0],
        vx_orb * e_vec[1] + vy_orb * q_vec[1],
        vx_orb * e_vec[2] + vy_orb * q_vec[2]
    ], dtype=np.float64)
    
    return pos, vel
