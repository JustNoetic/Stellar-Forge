import math
import numpy as np
import ctypes
from numba import njit, prange
import time
from engine.core.constants import *
from engine.core.math_utils import *
from engine.ephemeris.system_manager import SystemManager, SystemSnapshot, derive_star_properties
import warnings
from engine.rendering.render_utils import *
from engine.physics.atmosphere_physics import compute_atmosphere_properties, compute_dynamic_mie_properties
from engine.physics.kepler_analytical import extract_all_kepler_elements, propagate_keplerian_system_numba

def _extract_render_state(sim, num_bodies, out_pos, out_vel):
    """Extract particle positions/velocities with render coordinate swap (x, z, -y)."""
    out_pos[:, 0] = sim.arr[:num_bodies, 0]
    out_pos[:, 1] = sim.arr[:num_bodies, 2]
    out_pos[:, 2] = -sim.arr[:num_bodies, 1]
    out_vel[:, 0] = sim.arr[:num_bodies, 3]
    out_vel[:, 1] = sim.arr[:num_bodies, 5]
    out_vel[:, 2] = -sim.arr[:num_bodies, 4]

@njit(cache=True, nogil=True)
def compute_custom_forces(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles):
    if has_gr and num_bodies > 1:
        # 1. Transform to Jacobi coordinates
        M = np.zeros(num_bodies, dtype=np.float64)
        M[0] = arr[0, 9]
        for i in range(1, num_bodies):
            M[i] = M[i-1] + arr[i, 9]
            
        jx = np.zeros(num_bodies, dtype=np.float64)
        jy = np.zeros(num_bodies, dtype=np.float64)
        jz = np.zeros(num_bodies, dtype=np.float64)
        jvx = np.zeros(num_bodies, dtype=np.float64)
        jvy = np.zeros(num_bodies, dtype=np.float64)
        jvz = np.zeros(num_bodies, dtype=np.float64)
        jax = np.zeros(num_bodies, dtype=np.float64)
        jay = np.zeros(num_bodies, dtype=np.float64)
        jaz = np.zeros(num_bodies, dtype=np.float64)
        
        rx_com = arr[0, 0]
        ry_com = arr[0, 1]
        rz_com = arr[0, 2]
        vx_com = arr[0, 3]
        vy_com = arr[0, 4]
        vz_com = arr[0, 5]
        ax_com = arr[0, 6]
        ay_com = arr[0, 7]
        az_com = arr[0, 8]
        
        for i in range(1, num_bodies):
            jx[i] = arr[i, 0] - rx_com
            jy[i] = arr[i, 1] - ry_com
            jz[i] = arr[i, 2] - rz_com
            jvx[i] = arr[i, 3] - vx_com
            jvy[i] = arr[i, 4] - vy_com
            jvz[i] = arr[i, 5] - vz_com
            jax[i] = arr[i, 6] - ax_com
            jay[i] = arr[i, 7] - ay_com
            jaz[i] = arr[i, 8] - az_com
            
            mi = arr[i, 9]
            if M[i] > 0.0:
                frac = mi / M[i]
                rx_com += frac * jx[i]
                ry_com += frac * jy[i]
                rz_com += frac * jz[i]
                vx_com += frac * jvx[i]
                vy_com += frac * jvy[i]
                vz_com += frac * jvz[i]
                ax_com += frac * jax[i]
                ay_com += frac * jay[i]
                az_com += frac * jaz[i]
                
        # 2. Compute GR forces in Jacobi coordinates
        mu = G_val * arr[0, 9]
        jgr_ax = np.zeros(num_bodies, dtype=np.float64)
        jgr_ay = np.zeros(num_bodies, dtype=np.float64)
        jgr_az = np.zeros(num_bodies, dtype=np.float64)
        
        for i in range(1, num_bodies):
            x = jx[i]
            y = jy[i]
            z = jz[i]
            vx = jvx[i]
            vy = jvy[i]
            vz = jvz[i]
            ax = jax[i]
            ay = jay[i]
            az = jaz[i]
            
            r2 = x*x + y*y + z*z
            if r2 == 0.0: continue
            ri = math.sqrt(r2)
            
            vi_x = vx
            vi_y = vy
            vi_z = vz
            vi2 = vi_x*vi_x + vi_y*vi_y + vi_z*vi_z
            A = (0.5 * vi2 + 3.0 * mu / ri) / c2
            
            for q in range(10):
                old_v_x = vi_x
                old_v_y = vi_y
                old_v_z = vi_z
                
                vi_x = vx / (1.0 - A)
                vi_y = vy / (1.0 - A)
                vi_z = vz / (1.0 - A)
                vi2 = vi_x*vi_x + vi_y*vi_y + vi_z*vi_z
                A = (0.5 * vi2 + 3.0 * mu / ri) / c2
                
                dvx = vi_x - old_v_x
                dvy = vi_y - old_v_y
                dvz = vi_z - old_v_z
                if vi2 > 0.0 and (dvx*dvx + dvy*dvy + dvz*dvz)/vi2 < 4.93038e-32:
                    break
                    
            B = (mu / ri - 1.5 * vi2) * mu / (r2 * ri) / c2
            rdotrdot = x*vx + y*vy + z*vz
            
            vidot_x = ax + B*x
            vidot_y = ay + B*y
            vidot_z = az + B*z
            
            vdotvdot = vi_x*vidot_x + vi_y*vidot_y + vi_z*vidot_z
            D = (vdotvdot - 3.0 * mu / (r2 * ri) * rdotrdot) / c2
            
            jgr_ax[i] = B * (1.0 - A) * x - A * ax - D * vi_x
            jgr_ay[i] = B * (1.0 - A) * y - A * ay - D * vi_y
            jgr_az[i] = B * (1.0 - A) * z - A * az - D * vi_z
            
        # 3. Transform Jacobi GR accelerations back to inertial
        ax_running = 0.0
        ay_running = 0.0
        az_running = 0.0
        
        igr_ax = np.zeros(num_bodies, dtype=np.float64)
        igr_ay = np.zeros(num_bodies, dtype=np.float64)
        igr_az = np.zeros(num_bodies, dtype=np.float64)
        
        for i in range(num_bodies - 1, 0, -1):
            mi = arr[i, 9]
            if M[i] > 0.0:
                frac = mi / M[i]
                ax_running -= frac * jgr_ax[i]
                ay_running -= frac * jgr_ay[i]
                az_running -= frac * jgr_az[i]
                
            igr_ax[i] = jgr_ax[i] + ax_running
            igr_ay[i] = jgr_ay[i] + ay_running
            igr_az[i] = jgr_az[i] + az_running
            
        igr_ax[0] = ax_running
        igr_ay[0] = ay_running
        igr_az[0] = az_running
        
        for i in range(num_bodies):
            arr[i, 6] += igr_ax[i]
            arr[i, 7] += igr_ay[i]
            arr[i, 8] += igr_az[i]

    if has_j2 and oblate_indices is not None:
        for obl_idx in range(len(oblate_indices)):
            body_idx = oblate_indices[obl_idx]
            if body_idx >= num_bodies: continue
            
            j2 = oblate_j2[obl_idx]
            j4 = oblate_j4[obl_idx]
            if j2 == 0.0 and j4 == 0.0: continue
            
            req = oblate_req[obl_idx]
            M = arr[body_idx, 9]
            if M == 0.0: continue
            
            px = oblate_poles[obl_idx, 0]
            py = oblate_poles[obl_idx, 1]
            pz = oblate_poles[obl_idx, 2]
            
            base_j2 = -(1.5) * j2 * G_val * M * (req**2)
            base_j4 = (15.0/8.0) * j4 * G_val * M * (req**4)
            
            bx = arr[body_idx, 0]
            by = arr[body_idx, 1]
            bz = arr[body_idx, 2]
            
            for i in range(num_bodies):
                if i == body_idx: continue
                
                dx = arr[i, 0] - bx
                dy = arr[i, 1] - by
                dz = arr[i, 2] - bz
                
                r2 = dx*dx + dy*dy + dz*dz
                if r2 == 0.0: continue
                r = math.sqrt(r2)
                z_dot = dx*px + dy*py + dz*pz
                z_r = z_dot / r
                z_r_sq = z_r * z_r
                
                a_x = 0.0
                a_y = 0.0
                a_z = 0.0
                
                if j2 != 0.0:
                    prefac_j2 = base_j2 / (r2*r2*r)
                    term1 = 1.0 - 5.0 * z_r_sq
                    term2 = 2.0 * z_dot
                    a_x += prefac_j2 * (term1 * dx + term2 * px)
                    a_y += prefac_j2 * (term1 * dy + term2 * py)
                    a_z += prefac_j2 * (term1 * dz + term2 * pz)
                    
                if j4 != 0.0:
                    prefac_j4 = base_j4 / (r2*r2*r2*r)
                    z_r_4 = z_r_sq * z_r_sq
                    term1 = 1.0 - 14.0 * z_r_sq + 21.0 * z_r_4
                    term2 = 4.0 * z_r - (28.0 / 3.0) * z_r_sq * z_r
                    a_x += prefac_j4 * (term1 * dx + term2 * r * px)
                    a_y += prefac_j4 * (term1 * dy + term2 * r * py)
                    a_z += prefac_j4 * (term1 * dz + term2 * r * pz)
                
                arr[i, 6] += a_x
                arr[i, 7] += a_y
                arr[i, 8] += a_z
                
                mi = arr[i, 9]
                if mi > 0.0:
                    ratio = mi / M
                    arr[body_idx, 6] -= a_x * ratio
                    arr[body_idx, 7] -= a_y * ratio
                    arr[body_idx, 8] -= a_z * ratio

@njit(cache=True, nogil=True)
def compute_all_accelerations(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                              oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles):
    # Zero out accelerations
    for i in range(num_bodies):
        arr[i, 6] = 0.0
        arr[i, 7] = 0.0
        arr[i, 8] = 0.0
        
    # Base N-body Newtonian Gravity
    for i in range(num_bodies):
        xi = arr[i, 0]
        yi = arr[i, 1]
        zi = arr[i, 2]
        for j in range(i + 1, num_bodies):
            dx = arr[j, 0] - xi
            dy = arr[j, 1] - yi
            dz = arr[j, 2] - zi
            r2 = dx*dx + dy*dy + dz*dz
            if r2 == 0.0: continue
            r_inv = 1.0 / math.sqrt(r2)
            r3_inv = r_inv * r_inv * r_inv
            
            f = G_val * r3_inv
            
            mj = arr[j, 9]
            if mj > 0.0:
                arr[i, 6] += f * mj * dx
                arr[i, 7] += f * mj * dy
                arr[i, 8] += f * mj * dz
                
            mi = arr[i, 9]
            if mi > 0.0:
                arr[j, 6] -= f * mi * dx
                arr[j, 7] -= f * mi * dy
                arr[j, 8] -= f * mi * dz

    compute_custom_forces(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                          oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles)

def attach_custom_forces(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list):
    sim.has_j2 = has_j2
    sim.has_gr = has_gr
    sim.phys_star_idx = phys_star_idx
    if has_j2 and oblate_physics_list:
        sim.oblate_indices = np.array([x[0] for x in oblate_physics_list], dtype=np.int32)
        sim.oblate_j2 = np.array([x[1] for x in oblate_physics_list], dtype=np.float64)
        sim.oblate_j4 = np.array([x[2] for x in oblate_physics_list], dtype=np.float64)
        sim.oblate_req = np.array([x[3] for x in oblate_physics_list], dtype=np.float64)
        sim.oblate_poles = np.array([x[4] for x in oblate_physics_list], dtype=np.float64)
    else:
        sim.oblate_indices = np.empty(0, dtype=np.int32)
        sim.oblate_j2 = np.empty(0, dtype=np.float64)
        sim.oblate_j4 = np.empty(0, dtype=np.float64)
        sim.oblate_req = np.empty(0, dtype=np.float64)
        sim.oblate_poles = np.empty((0, 3), dtype=np.float64)


@njit(cache=True, nogil=True)
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

@njit(cache=True, nogil=True)
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

@njit(cache=True, nogil=True)
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

@njit(cache=True, nogil=True)
def compute_all_orbits_batch(pos, vel, mass, parent_indices,
                             subsys_pos, subsys_vel, subsys_mass,
                             visual_colors, cam_origin, G, max_orbits, orbit_buf, lod_levels,
                             bary_offset):
    """Compute all orbital elements (child + reflex) in one Numba-accelerated batch.
    Returns number of valid orbits written to orbit_buf.

    Optimized: sibling lookup uses a precomputed children-of-parent table (CSR)
    instead of scanning all N bodies each iteration (O(N²) → O(N·k)).
    """
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

    # ── Precompute children-of-parent table (CSR format) ──
    children_count = np.zeros(num_bodies, dtype=np.int64)
    for i in range(num_bodies):
        p_idx = parent_indices[i]
        if p_idx >= 0:
            children_count[p_idx] += 1
            if mass[i] > best_child_mass[p_idx]:
                best_child_mass[p_idx] = mass[i]
                best_child[p_idx] = i

    children_offsets = np.zeros(num_bodies + 1, dtype=np.int64)
    for p in range(1, num_bodies + 1):
        children_offsets[p] = children_offsets[p - 1] + children_count[p - 1]
    children_flat = np.zeros(children_offsets[num_bodies], dtype=np.int64)
    fill_pos = children_offsets[:num_bodies].copy()
    for i in range(num_bodies):
        p_idx = parent_indices[i]
        if p_idx >= 0:
            children_flat[fill_pos[p_idx]] = i
            fill_pos[p_idx] += 1

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
            
        # Iterate only over siblings (children of same parent) — O(k) not O(N)
        c_start = children_offsets[p_idx]
        c_end = children_offsets[p_idx + 1]
        for ci in range(c_start, c_end):
            j = children_flat[ci]
            if j != i:
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
                    
        if eff_parent_m > 0.0:
            for k in range(3):
                eff_parent_pos[k] /= eff_parent_m
                eff_parent_vel[k] /= eff_parent_m
        else:
            for k in range(3):
                eff_parent_pos[k] = pos[p_idx, k]
                eff_parent_vel[k] = vel[p_idx, k]

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
                orbit_buf[n_orbits, 8] = bary[0] + bary_offset[0]
                orbit_buf[n_orbits, 9] = bary[1] + bary_offset[1]
                orbit_buf[n_orbits, 10] = bary[2] + bary_offset[2]
                
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
                        orbit_buf[n_orbits, 8] = gp_bary[0] + bary_offset[0]
                        orbit_buf[n_orbits, 9] = gp_bary[1] + bary_offset[1]
                        orbit_buf[n_orbits, 10] = gp_bary[2] + bary_offset[2]
                            
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
            orbit_buf[n_orbits, 8] = bary[0] + bary_offset[0]
            orbit_buf[n_orbits, 9] = bary[1] + bary_offset[1]
            orbit_buf[n_orbits, 10] = bary[2] + bary_offset[2]
            
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

@njit(cache=True, nogil=True)
def _update_hierarchy_core(positions, masses, current_parents, num_bodies, is_star_mask):
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
    
    # Enforce that stars always remain root nodes
    for i in range(num_bodies):
        if is_star_mask[i]:
            new_parents[i] = -1
    
    r_hill_array = np.zeros(num_bodies, dtype=np.float64)
    for j in range(num_bodies):
        if j == root_idx or masses[j] <= 0.0:
            continue
        if is_star_mask[j]:
            continue # Stars cannot be child nodes to other bodies
            
        j_parent = current_parents[j] if current_parents[j] != -1 else root_idx
        if masses[j_parent] > 0.0:
            dx = positions[j, 0] - positions[j_parent, 0]
            dy = positions[j, 1] - positions[j_parent, 1]
            dz = positions[j, 2] - positions[j_parent, 2]
            dist_j_parent = math.sqrt(dx*dx + dy*dy + dz*dz)
            r_hill_array[j] = dist_j_parent * (masses[j] / (3.0 * masses[j_parent])) ** (1.0 / 3.0)
    
    for i in range(num_bodies):
        if i == root_idx or is_star_mask[i]:
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

def update_hierarchy(sim, num_bodies, current_parents, is_star_mask):
    """Recompute parent_indices based on Hill sphere containment.
    
    Uses hysteresis: enter at < 0.9 * r_Hill, exit at > 1.0 * r_Hill
    to prevent flickering at boundaries.
    """
    arr = sim.arr[:num_bodies]
    positions = np.ascontiguousarray(arr[:, 0:3])
    masses = arr[:, 9].copy()
    
    return _update_hierarchy_core(positions, masses, current_parents, num_bodies, is_star_mask)

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

    last_time = time.perf_counter()
    kepler_cached_elements = None
    kepler_init_pos = None
    kepler_init_vel = None
    kepler_subsys_init_pos = None
    kepler_subsys_init_vel = None
    
    was_keplerian = False
    kepler_entry_sim_copy = None
    kepler_entry_parents = None
    kepler_entry_tree_indices = None
    kepler_entry_tree_depths = None
    
    with shared_state["lock"]:
        current_parents = shared_state["parent_indices"].copy()
        is_star_mask = shared_state.get("is_star_mask")
        if is_star_mask is None:
            is_star_mask = np.zeros(num_bodies, dtype=np.bool_)
    
    current_parents = update_hierarchy(sim, num_bodies, current_parents, is_star_mask)
    positions = np.array([[p.x, p.y, p.z] for p in sim.particles[:num_bodies]])
    tree_indices, tree_depths = build_tree_order(current_parents, num_bodies, positions)
    
    with shared_state["lock"]:
        shared_state["parent_indices"][:] = current_parents
        shared_state["tree_indices"][:] = tree_indices
        shared_state["tree_depths"][:] = tree_depths
        shared_state["hierarchy_version"] += 1
        
    frame_count = 0
    while running[0]:
        now = time.perf_counter()
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
                    attach_custom_forces(sim, restore_snap.has_j2, restore_snap.has_gr, restore_snap.phys_star_idx, restore_snap.oblate_physics_list)
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
                is_star_mask = np.array([b.get('type') == 'Star' for b in new_bundle_info["bodies_data"]], dtype=np.bool_)
                shared_state["is_star_mask"] = is_star_mask
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
            _init_arr_sw = sim.arr[:num_bodies]

            current_parents = new_bundle_info.get("parent_indices",
                np.full(num_bodies, -1, dtype=np.int32)).copy() if "parent_indices" in new_bundle_info else np.full(num_bodies, -1, dtype=np.int32)
            # Rebuild parent indices from bodies_data if not from snapshot
            bd = new_bundle_info["bodies_data"]
            nti = {b['name']: i for i, b in enumerate(bd)}
            for i, body in enumerate(bd):
                if 'parentId' in body and body['parentId'] in nti:
                    current_parents[i] = nti[body['parentId']]
            current_parents = update_hierarchy(sim, num_bodies, current_parents, is_star_mask)
            positions = np.array([[p.x, p.y, p.z] for p in sim.particles[:num_bodies]])
            tree_indices, tree_depths = build_tree_order(current_parents, num_bodies, positions)

            with shared_state["lock"]:
                np.copyto(shared_state["pos"], local_pos)
                np.copyto(shared_state["vel"], local_vel)
                shared_state["t"] = sim.t
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
                
            was_keplerian = False
            kepler_entry_sim_copy = None
            kepler_entry_parents = None
            kepler_entry_tree_indices = None
            kepler_entry_tree_depths = None
            kepler_cached_elements = None
            frame_count = 0
            last_time = time.perf_counter()
            continue
        # ── Keplerian Mode Transitions ──
        keplerian_active = shared_state.get("keplerian_mode", False)
        if keplerian_active and not was_keplerian:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kepler_entry_sim_copy = sim.copy()
            kepler_entry_parents = current_parents.copy()
            kepler_entry_tree_indices = tree_indices.copy()
            kepler_entry_tree_depths = tree_depths.copy()
            was_keplerian = True
        elif not keplerian_active and was_keplerian:
            export_requested = False
            with shared_state["lock"]:
                if shared_state.get("keplerian_export", False):
                    export_requested = True
                    shared_state["keplerian_export"] = False
            
            if not export_requested and kepler_entry_sim_copy is not None:
                sim = kepler_entry_sim_copy
                current_parents = kepler_entry_parents
                tree_indices = kepler_entry_tree_indices
                tree_depths = kepler_entry_tree_depths
                with shared_state["lock"]:
                    shared_state["parent_indices"][:] = current_parents
                    shared_state["tree_indices"][:] = tree_indices
                    shared_state["tree_depths"][:] = tree_depths
                    shared_state["hierarchy_version"] += 1
                    shared_state["rebuild_flag"] = True
            
            was_keplerian = False
            kepler_entry_sim_copy = None
            kepler_entry_parents = None
            kepler_entry_tree_indices = None
            kepler_entry_tree_depths = None

        crud_ops = []
        with shared_state["lock"]:
            if shared_state["crud_queue"]:
                crud_ops = shared_state["crud_queue"][:]
                shared_state["crud_queue"].clear()

        if crud_ops:
            kepler_cached_elements = None
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
                                    shared_state["pos"][i] = [dp.x, dp.z, -dp.y]
                                    shared_state["vel"][i] = [dp.vx, dp.vz, -dp.vy]
                        
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
                            if "radii" in shared_state:
                                shared_state["radii"][idx] = r_au
                            if shared_state.get("oblate_indices") is not None:
                                mask = shared_state["oblate_indices"] == idx
                                if np.any(mask):
                                    shared_state["oblate_req"][mask] = r_au
                                for i, obl in enumerate(shared_state["oblate_physics_list"]):
                                    if obl[0] == idx:
                                        shared_state["oblate_physics_list"][i] = (obl[0], obl[1], obl[2], r_au, obl[4], obl[5], obl[6], obl[7])
                        if "oblateness" in op:
                            f = op["oblateness"]
                            j2 = op.get("J2", 0.0)
                            j4 = op.get("j4", 0.0)
                            rot_period = op.get("rotation_period", 0.0)
                            pole_ecl = np.array(op.get("pole_ecl", [0.0, 0.0, 1.0]), dtype=np.float64)
                            r_au = op["radius"] / 1.496e8 if "radius" in op else p.r
                            
                            found = False
                            opl = list(shared_state.get("oblate_physics_list", []))
                            for i, obl in enumerate(opl):
                                if obl[0] == idx:
                                    opl[i] = (obl[0], j2, j4, r_au, pole_ecl, p.m, obl[6], rot_period)
                                    found = True
                                    break
                            if not found and j2 > 0.0:
                                name = op.get("name", "Edited")
                                opl.append((idx, j2, j4, r_au, pole_ecl, p.m, name, rot_period))
                                
                            shared_state["oblate_physics_list"] = opl
                            has_j2 = len(opl) > 0
                            shared_state["has_j2"] = has_j2
                            
                            if has_j2:
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
                                
                            attach_custom_forces(sim, has_j2, sim.has_gr, sim.phys_star_idx, opl)
                        if "type" in op:
                            is_star = op["type"] == "Star"
                            if shared_state.get("is_star_mask") is not None:
                                shared_state["is_star_mask"][idx] = is_star
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
                        if "radii" in shared_state:
                            r_au = op.get("radius", 6000.0) / 1.496e8
                            shared_state["radii"] = np.append(shared_state["radii"], r_au)
                        shared_state["parent_indices"] = np.append(shared_state["parent_indices"], np.int32(parent_idx))
                        is_star = op.get("type") == "Star"
                        if shared_state.get("is_star_mask") is not None:
                            shared_state["is_star_mask"] = np.append(shared_state["is_star_mask"], is_star)
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
                        if "radii" in shared_state:
                            shared_state["radii"] = np.delete(shared_state["radii"], idx)
                        
                        pi = np.delete(shared_state["parent_indices"], idx)
                        pi[pi == idx] = 0
                        pi[pi > idx] -= 1
                        shared_state["parent_indices"] = pi
                        if shared_state.get("is_star_mask") is not None:
                            shared_state["is_star_mask"] = np.delete(shared_state["is_star_mask"], idx)
                        
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
                is_star_mask = shared_state.get("is_star_mask")
                if is_star_mask is None:
                    is_star_mask = np.zeros(num_bodies, dtype=np.bool_)
            current_parents = update_hierarchy(sim, num_bodies, current_parents, is_star_mask)
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
                sim.reset_integrator_state()
            else:
                if shared_state.get("ephemeris_mode", False) or shared_state.get("keplerian_mode", False):
                    sim.t = sync_time
                    if shared_state.get("keplerian_mode", False):
                        sub_pos = np.empty((num_bodies, 3), dtype=np.float64)
                        sub_vel = np.empty((num_bodies, 3), dtype=np.float64)
                        sub_mass = np.empty(num_bodies, dtype=np.float64)
                        sim_pos = sim.arr[:num_bodies, 0:3]
                        sim_vel = sim.arr[:num_bodies, 3:6]
                        sim_mass = sim.arr[:num_bodies, 9]
                        compute_barycenters(sim_pos, sim_vel, sim_mass, current_parents, sub_pos, sub_vel, sub_mass)
                        if kepler_cached_elements is None or shared_state.get("keplerian_reextract", False) or len(kepler_cached_elements) != num_bodies:
                            with shared_state["lock"]:
                                shared_state["keplerian_reextract"] = False
                            kepler_cached_elements = extract_all_kepler_elements(sim_pos, sim_vel, sim_mass, current_parents, sub_pos, sub_vel, sub_mass, sim.t, G,
                                                                                 sim.oblate_indices, sim.oblate_j2, sim.oblate_req, sim.oblate_poles, sim.has_gr, C_AU_YR)
                            kepler_init_pos = sim_pos.copy()
                            kepler_init_vel = sim_vel.copy()
                            kepler_subsys_init_pos = sub_pos.copy()
                            kepler_subsys_init_vel = sub_vel.copy()
                        propagate_keplerian_system_numba(kepler_cached_elements, current_parents, tree_indices, sim.t, kepler_subsys_init_pos, kepler_subsys_init_vel, sim_pos, sim_vel, sim_mass, sub_mass)
                        for i in range(num_bodies):
                            sim.particles[i].x = sim_pos[i, 0]
                            sim.particles[i].y = sim_pos[i, 1]
                            sim.particles[i].z = sim_pos[i, 2]
                            sim.particles[i].vx = sim_vel[i, 0]
                            sim.particles[i].vy = sim_vel[i, 1]
                            sim.particles[i].vz = sim_vel[i, 2]
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
            attach_custom_forces(sim_start_copy, shared_state["has_j2"], shared_state["has_gr"], shared_state["phys_star_idx"], shared_state["oblate_physics_list"])
                
            render_start_time = time.perf_counter()
            valid_steps = 0
            for step in range(num_steps):
                if not running[0] or time_ctrl.get("cancel_render", False):
                    break
                sim_start_copy.integrate(start_t + (step + 1) * step_dt)
                
                _tl_arr = sim_start_copy.arr[:num_bodies]
                tl_pos[step, :, 0] = _tl_arr[:, 0]
                tl_pos[step, :, 1] = _tl_arr[:, 2]
                tl_pos[step, :, 2] = -_tl_arr[:, 1]
                tl_vel[step, :, 0] = _tl_arr[:, 3]
                tl_vel[step, :, 1] = _tl_arr[:, 5]
                tl_vel[step, :, 2] = -_tl_arr[:, 4]
                tl_times[step] = sim_start_copy.t
                
                tl_raw_arr[step] = _tl_arr[:, 0:6]
                
                if step % 20 == 0:
                    current_real_time = time.perf_counter()
                    elapsed = current_real_time - render_start_time
                    rate = (sim_start_copy.t - start_t) * SECONDS_PER_YEAR / max(elapsed, 0.001)
                    with shared_state["lock"]:
                        shared_state["timeline_progress"] = float(step) / num_steps
                        shared_state["timeline_rate"] = rate
                valid_steps = step + 1
            
            if valid_steps > 0 and not time_ctrl.get("cancel_render", False):
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
            time_ctrl["cancel_render"] = False
            time_ctrl["paused"] = True
            continue
        if not time_ctrl["paused"]:
            dt_sim = dt_real * time_ctrl["multiplier"] * time_ctrl.get("time_direction", 1) / SECONDS_PER_YEAR
            
            if shared_state.get("ephemeris_mode", False):
                sim.t += dt_sim
            elif shared_state.get("keplerian_mode", False):
                sim.t += dt_sim
                sub_pos = np.empty((num_bodies, 3), dtype=np.float64)
                sub_vel = np.empty((num_bodies, 3), dtype=np.float64)
                sub_mass = np.empty(num_bodies, dtype=np.float64)
                sim_pos = sim.arr[:num_bodies, 0:3]
                sim_vel = sim.arr[:num_bodies, 3:6]
                sim_mass = sim.arr[:num_bodies, 9]
                compute_barycenters(sim_pos, sim_vel, sim_mass, current_parents, sub_pos, sub_vel, sub_mass)
                if kepler_cached_elements is None or shared_state.get("keplerian_reextract", False) or len(kepler_cached_elements) != num_bodies:
                    with shared_state["lock"]:
                        shared_state["keplerian_reextract"] = False
                    kepler_cached_elements = extract_all_kepler_elements(sim_pos, sim_vel, sim_mass, current_parents, sub_pos, sub_vel, sub_mass, sim.t, G,
                                                                         sim.oblate_indices, sim.oblate_j2, sim.oblate_req, sim.oblate_poles, sim.has_gr, C_AU_YR)
                    kepler_init_pos = sim_pos.copy()
                    kepler_init_vel = sim_vel.copy()
                    kepler_subsys_init_pos = sub_pos.copy()
                    kepler_subsys_init_vel = sub_vel.copy()
                propagate_keplerian_system_numba(kepler_cached_elements, current_parents, tree_indices, sim.t, kepler_subsys_init_pos, kepler_subsys_init_vel, sim_pos, sim_vel, sim_mass, sub_mass)
                for i in range(num_bodies):
                    sim.particles[i].x = sim_pos[i, 0]
                    sim.particles[i].y = sim_pos[i, 1]
                    sim.particles[i].z = sim_pos[i, 2]
                    sim.particles[i].vx = sim_vel[i, 0]
                    sim.particles[i].vy = sim_vel[i, 1]
                    sim.particles[i].vz = sim_vel[i, 2]
            else:
                sim.integrate(sim.t + dt_sim)
                
                # COLLISION DETECTION
                local_pos = np.empty((num_bodies, 3), dtype=np.float64)
                local_vel = np.empty((num_bodies, 3), dtype=np.float64)
                _extract_render_state(sim, num_bodies, local_pos, local_vel)
                
                with shared_state["lock"]:
                    radii = shared_state.get("radii")
                    
                if radii is not None:
                    collisions = []
                    for i in range(num_bodies):
                        for j in range(i + 1, num_bodies):
                            dx = local_pos[i, 0] - local_pos[j, 0]
                            dy = local_pos[i, 1] - local_pos[j, 1]
                            dz = local_pos[i, 2] - local_pos[j, 2]
                            dist_sq = dx*dx + dy*dy + dz*dz
                            r_sum = radii[i] + radii[j]
                            if dist_sq < r_sum * r_sum:
                                collisions.append((i, j))
                                
                    if collisions:
                        # Process only the first collision to avoid overlapping indices complexity this tick
                        idxA, idxB = collisions[0]
                        massA = sim.arr[idxA, 9]
                        massB = sim.arr[idxB, 9]
                        
                        winner = idxA if massA >= massB else idxB
                        loser = idxB if winner == idxA else idxA
                        
                        new_mass = massA + massB
                        if new_mass > 0:
                            new_vx = (massA * sim.arr[idxA, 3] + massB * sim.arr[idxB, 3]) / new_mass
                            new_vy = (massA * sim.arr[idxA, 4] + massB * sim.arr[idxB, 4]) / new_mass
                            new_vz = (massA * sim.arr[idxA, 5] + massB * sim.arr[idxB, 5]) / new_mass
                        else:
                            new_vx = new_vy = new_vz = 0.0
                            
                        # Volume addition for inelastic merge R_new = (R_A^3 + R_B^3)^(1/3)
                        r1 = radii[winner]
                        r2 = radii[loser]
                        new_r_au = (r1**3 + r2**3)**(1/3.0)
                        new_r_km = new_r_au * 1.496e8
                        
                        with shared_state["lock"]:
                            shared_state["crud_queue"].append({
                                "action": "UPDATE",
                                "idx": winner,
                                "mass": new_mass,
                                "radius": new_r_km,
                                "vel": [new_vx, new_vy, new_vz]
                            })
                            shared_state["crud_queue"].append({
                                "action": "DELETE",
                                "idx": loser
                            })
            
            frame_count += 1
            if frame_count % 30 == 0:
                with shared_state["lock"]:
                    is_star_mask = shared_state.get("is_star_mask")
                    if is_star_mask is None:
                        is_star_mask = np.zeros(num_bodies, dtype=np.bool_)
                new_parents = update_hierarchy(sim, num_bodies, current_parents, is_star_mask)
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
                
        elapsed = time.perf_counter() - now
        target_time = 0.00832
        sleep_time = target_time - elapsed
        if sleep_time > 0:
            # Sleep for most of the time to save CPU, but leave 1ms for busy wait
            if sleep_time > 0.001:
                time.sleep(sleep_time - 0.001)
            # Busy wait the remainder for high precision
            while (time.perf_counter() - now) < target_time:
                pass

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
    sim = Simulation()
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

        # Black holes: the event-horizon radius is derived DIRECTLY from the
        # mass (r_s = 2GM/c^2 = 2.95325008 km per solar mass); any stored 'r'
        # property is ignored. The rendered / culled size is the photon-capture
        # (shadow) silhouette b_c = (3*sqrt(3)/2) * r_s — the only visible
        # extent of a black hole.
        _sp_bh = body.get('star_props', {}) or {}
        if ('Black Hole' in (_sp_bh.get('class', '') or '')
                or _sp_bh.get('stage', '') == 'Singularity'
                or body.get('type') == 'Black Hole'):
            _rs_km = 2.95325008 * mass
            body['r'] = _rs_km / 696340.0
            radius_au = (2.5980762 * _rs_km) / 149597870.7
        else:
            radius_au = body.get('r', 1.0) * SOLAR_RADII_TO_AU

        min_px = 1.0

        pole_ecl = np.array([0.0, 0.0, 1.0], dtype='f8')
        pole_render = np.array([0.0, 1.0, 0.0], dtype='f8')
        if 'pole_ra' in body and 'pole_dec' in body:
            pole_ecl = np.array(pole_to_ecliptic(body['pole_ra'], body['pole_dec']), dtype='f8')
            pole_render = np.array([pole_ecl[0], pole_ecl[2], -pole_ecl[1]], dtype='f8')
            pole_dirs[idx] = pole_render
        obl = body.get('oblateness', 0.0)
        pole_color = hex_to_rgb(body.get('color_pole', body.get('color', '#ffffff')))
        lum_eq = body.get('lum_eq', 0.0)
        lum_pole = body.get('lum_pole', 0.0)
        visual_data.append([color[0], color[1], color[2], radius_au, min_px,
                            pole_render[0], pole_render[1], pole_render[2], obl,
                            pole_color[0], pole_color[1], pole_color[2], lum_eq, lum_pole])

        j2 = body.get('J2', 0.0)
        if j2 > 0.0:
            j4 = body.get('j4', 0.0)
            req_km = body.get('req_km')
            if req_km:
                req_au = req_km / 149597870.7
            else:
                r_mean = body.get('r', 1.0)
                f = body.get('oblateness', 0.0)
                req_solar = r_mean / ((1.0 - f) ** (1.0 / 3.0)) if f > 0 else r_mean
                req_au = req_solar * SOLAR_RADII_TO_AU
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
                except KeyError:
                    available = [p.hash for p in sim.particles if hasattr(p, 'hash') and p.hash is not None]
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
            radius_au = r_solar * SOLAR_RADII_TO_AU
            mass = body.get('m', 1.0)
            mass_kg = mass * 1.98847e30
            surface_pressure = float(atmo.get('surface_pressure', 1.0))
            temperature = float(atmo.get('temperature', 288.15))
            composition = atmo.get('composition', {"N2": 0.78, "O2": 0.21, "Ar": 0.01})
            g_m_s2 = (6.67430e-11 * mass_kg) / ((planet_radius_km * 1000.0) ** 2) if planet_radius_km > 0 else 9.81
            props = compute_atmosphere_properties(surface_pressure, temperature, composition, g_m_s2)
            dyn_mie = compute_dynamic_mie_properties(surface_pressure, temperature, composition, g_m_s2)
            atmo_height_km = float(atmo.get('height', props['atmo_height_km']))
            atmo_radius_km = planet_radius_km + atmo_height_km
            atmo_radius_au = atmo_radius_km / AU_TO_KM
            atmo_bodies.append({
                'body_idx': i,
                'planet_radius_km': planet_radius_km,
                'atmo_radius_km': atmo_radius_km,
                'atmo_radius_au': atmo_radius_au,
                'height': atmo_height_km,
                'surface_radius_au': radius_au,
                'surface_pressure': float(atmo.get('surface_pressure', 1.0)),
                'temperature': float(atmo.get('temperature', 288.15)),
                'composition': atmo.get('composition', {"N2": 0.78, "O2": 0.21, "Ar": 0.01}),
                'intensity': float(atmo.get('intensity', 1.0)),
                'beta_mie': float(atmo.get('beta_mie', atmo.get('mieCoefficient', dyn_mie['beta_mie']))),
                'h_mie': float(atmo.get('h_mie', atmo.get('mieScaleHeight', dyn_mie['h_mie']))),
                'mie_g': float(atmo.get('mie_g', atmo.get('mieAsymmetry', dyn_mie['mie_g']))),
                'mie_albedo': np.asarray(atmo.get('mie_albedo', dyn_mie['mie_albedo']), dtype=np.float32),
                'mie_angstrom': atmo.get('mie_angstrom', dyn_mie['mie_angstrom'])
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
        attach_custom_forces(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list)
        if has_j2:
            print(f"[System] Attached J2 precession for {len(oblate_physics_list)} oblate bodies via custom Numba forces")
        if has_gr:
            print(f"[System] Attached 1PN GR precession (star index {phys_star_idx}) via custom Numba forces")
        sim.force_is_velocity_dependent = 1

    num_bodies = len(sim.particles)

    # ── Parent indices ──
    parent_indices = np.full(num_bodies, -1, dtype=np.int32)
    for i, body in enumerate(bodies_data):
        if body.get('type') == 'Star':
            parent_indices[i] = -1
        elif 'parentId' in body:
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


# ==============================================================================
# PURE PYTHON NUMBA IAS15 INTEGRATOR
# ==============================================================================

# IAS15 Constants
_IAS15_H = np.array([0.0, 0.0562625605369221464656521910318, 0.180240691736892364987579942780, 0.352624717113169637373907769648, 0.547153626330555383001448554766, 0.734210177215410531523210605558, 0.885320946839095768090359771030, 0.977520613561287501891174488626], dtype=np.float64)
_IAS15_RR = np.array([0.0562625605369221464656522, 0.1802406917368923649875799, 0.1239781311999702185219278, 0.3526247171131696373739078, 0.2963621565762474909082556, 0.1723840253762772723863278, 0.5471536263305553830014486, 0.4908910657936332365357964, 0.3669129345936630180138686, 0.1945289092173857456275408, 0.7342101772154105315232106, 0.6779476166784883850575584, 0.5539694854785181665356307, 0.3815854601022408941493028, 0.1870565508848551485217621, 0.8853209468390957680903598, 0.8290583863021736216247076, 0.7050802551022034031027798, 0.5326962297259261307164520, 0.3381673205085403850889112, 0.1511107696236852365671492, 0.9775206135612875018911745, 0.9212580530243653554255223, 0.7972799218243951369035945, 0.6248958964481178645172667, 0.4303669872307321188897259, 0.2433104363458769703679639, 0.0921996667221917338008147], dtype=np.float64)
_IAS15_C = np.array([-0.0562625605369221464656522, 0.0101408028300636299864818, -0.2365032522738145114532321, -0.0035758977292516175949345, 0.0935376952594620658957485, -0.5891279693869841488271399, 0.0019565654099472210769006, -0.0547553868890686864408084, 0.4158812000823068616886219, -1.1362815957175395318285885, -0.0014365302363708915424460, 0.0421585277212687077072973, -0.3600995965020568122897665, 1.2501507118406910258505441, -1.8704917729329500633517991, 0.0012717903090268677492943, -0.0387603579159067703699046, 0.3609622434528459832253398, -1.4668842084004269643701553, 2.9061362593084293014237913, -2.7558127197720458314421588], dtype=np.float64)
_IAS15_D = np.array([0.0562625605369221464656522, 0.0031654757181708292499905, 0.2365032522738145114532321, 0.0001780977692217433881125, 0.0457929855060279188954539, 0.5891279693869841488271399, 0.0000100202365223291272096, 0.0084318571535257015445000, 0.2535340690545692665214616, 1.1362815957175395318285885, 0.0000005637641639318207610, 0.0015297840025004658189490, 0.0978342365324440053653648, 0.8752546646840910912297246, 1.8704917729329500633517991, 0.0000000317188154017613665, 0.0002762930909826476593130, 0.0360285539837364596003871, 0.5767330002770787313544596, 2.2485887607691597933926895, 2.7558127197720458314421588], dtype=np.float64)

@njit(cache=True, nogil=True)
def ias15_sqrt7(a):
    scale = 1.0
    while a < 1e-7 and a > 0.0:
        scale *= 0.1
        a *= 1e7
    while a > 1e2:
        scale *= 10.0
        a *= 1e-7
    x = 1.0
    for _ in range(20):
        x6 = x*x*x*x*x*x
        x += (a/x6 - x) / 7.0
    return x * scale

@njit(cache=True, nogil=True)
def _ias15_add_cs(val, cs_val, inp):
    y = inp - cs_val
    t = val + y
    new_cs = (t - val) - y
    return t, new_cs

@njit(cache=True, nogil=True)
def _ias15_predict_next_step(ratio, N3, _e, _b, e, b):
    if ratio > 20.0:
        for k in range(N3):
            for i in range(7):
                e[i, k] = 0.0
                b[i, k] = 0.0
    else:
        q1 = ratio
        q2 = q1 * q1
        q3 = q1 * q2
        q4 = q2 * q2
        q5 = q2 * q3
        q6 = q3 * q3
        q7 = q3 * q4

        for k in range(N3):
            be0 = _b[0, k] - _e[0, k]
            be1 = _b[1, k] - _e[1, k]
            be2 = _b[2, k] - _e[2, k]
            be3 = _b[3, k] - _e[3, k]
            be4 = _b[4, k] - _e[4, k]
            be5 = _b[5, k] - _e[5, k]
            be6 = _b[6, k] - _e[6, k]

            e[0, k] = q1*(_b[6, k]* 7.0 + _b[5, k]* 6.0 + _b[4, k]* 5.0 + _b[3, k]* 4.0 + _b[2, k]* 3.0 + _b[1, k]*2.0 + _b[0, k])
            e[1, k] = q2*(_b[6, k]*21.0 + _b[5, k]*15.0 + _b[4, k]*10.0 + _b[3, k]* 6.0 + _b[2, k]* 3.0 + _b[1, k])
            e[2, k] = q3*(_b[6, k]*35.0 + _b[5, k]*20.0 + _b[4, k]*10.0 + _b[3, k]* 4.0 + _b[2, k])
            e[3, k] = q4*(_b[6, k]*35.0 + _b[5, k]*15.0 + _b[4, k]* 5.0 + _b[3, k])
            e[4, k] = q5*(_b[6, k]*21.0 + _b[5, k]* 6.0 + _b[4, k])
            e[5, k] = q6*(_b[6, k]* 7.0 + _b[5, k])
            e[6, k] = q7* _b[6, k]

            b[0, k] = e[0, k] + be0
            b[1, k] = e[1, k] + be1
            b[2, k] = e[2, k] + be2
            b[3, k] = e[3, k] + be3
            b[4, k] = e[4, k] + be4
            b[5, k] = e[5, k] + be5
            b[6, k] = e[6, k] + be6

@njit(cache=True, nogil=True)
def ias15_step_numba(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                     oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles,
                     dt, min_dt, epsilon, dt_last_done,
                     g, e_arr, b, csb, er, br, at, x0, v0, a0, csx, csv,
                     h_const, rr_const, c_const, d_const):
                     
    N3 = 3 * num_bodies
    safety_factor = 0.25
    
    compute_all_accelerations(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                              oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles)

    for k in range(num_bodies):
        x0[3*k+0] = arr[k, 0]
        x0[3*k+1] = arr[k, 1]
        x0[3*k+2] = arr[k, 2]
        v0[3*k+0] = arr[k, 3]
        v0[3*k+1] = arr[k, 4]
        v0[3*k+2] = arr[k, 5]
        a0[3*k+0] = arr[k, 6]
        a0[3*k+1] = arr[k, 7]
        a0[3*k+2] = arr[k, 8]

    for k in range(N3):
        for i in range(7):
            csb[i, k] = 0.0

    for k in range(N3):
        g[0, k] = b[6, k]*d_const[15] + b[5, k]*d_const[10] + b[4, k]*d_const[6] + b[3, k]*d_const[3]  + b[2, k]*d_const[1]  + b[1, k]*d_const[0]  + b[0, k]
        g[1, k] = b[6, k]*d_const[16] + b[5, k]*d_const[11] + b[4, k]*d_const[7] + b[3, k]*d_const[4]  + b[2, k]*d_const[2]  + b[1, k]
        g[2, k] = b[6, k]*d_const[17] + b[5, k]*d_const[12] + b[4, k]*d_const[8] + b[3, k]*d_const[5]  + b[2, k]
        g[3, k] = b[6, k]*d_const[18] + b[5, k]*d_const[13] + b[4, k]*d_const[9] + b[3, k]
        g[4, k] = b[6, k]*d_const[19] + b[5, k]*d_const[14] + b[4, k]
        g[5, k] = b[6, k]*d_const[20] + b[5, k]
        g[6, k] = b[6, k]

    predictor_corrector_error = 1e300
    predictor_corrector_error_last = 2.0
    iterations = 0
    
    dt_new = dt

    while True:
        if predictor_corrector_error < 1e-16:
            break
        if iterations > 2 and predictor_corrector_error_last <= predictor_corrector_error:
            break
        if iterations >= 12:
            break
            
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0
        iterations += 1

        for n in range(1, 8):
            dt_h = dt * h_const[n]
            
            for i in range(num_bodies):
                k0, k1, k2 = 3*i, 3*i+1, 3*i+2
                
                xk0 = -csx[k0] + ((((((((b[6, k0]*7.*h_const[n]/9. + b[5, k0])*3.*h_const[n]/4. + b[4, k0])*5.*h_const[n]/7. + b[3, k0])*2.*h_const[n]/3. + b[2, k0])*3.*h_const[n]/5. + b[1, k0])*h_const[n]/2. + b[0, k0])*h_const[n]/3. + a0[k0])*dt_h/2. + v0[k0])*dt_h
                xk1 = -csx[k1] + ((((((((b[6, k1]*7.*h_const[n]/9. + b[5, k1])*3.*h_const[n]/4. + b[4, k1])*5.*h_const[n]/7. + b[3, k1])*2.*h_const[n]/3. + b[2, k1])*3.*h_const[n]/5. + b[1, k1])*h_const[n]/2. + b[0, k1])*h_const[n]/3. + a0[k1])*dt_h/2. + v0[k1])*dt_h
                xk2 = -csx[k2] + ((((((((b[6, k2]*7.*h_const[n]/9. + b[5, k2])*3.*h_const[n]/4. + b[4, k2])*5.*h_const[n]/7. + b[3, k2])*2.*h_const[n]/3. + b[2, k2])*3.*h_const[n]/5. + b[1, k2])*h_const[n]/2. + b[0, k2])*h_const[n]/3. + a0[k2])*dt_h/2. + v0[k2])*dt_h
                
                arr[i, 0] = xk0 + x0[k0]
                arr[i, 1] = xk1 + x0[k1]
                arr[i, 2] = xk2 + x0[k2]

                vk0 =  -csv[k0] + (((((((b[6, k0]*7.*h_const[n]/8. + b[5, k0])*6.*h_const[n]/7. + b[4, k0])*5.*h_const[n]/6. + b[3, k0])*4.*h_const[n]/5. + b[2, k0])*3.*h_const[n]/4. + b[1, k0])*2.*h_const[n]/3. + b[0, k0])*h_const[n]/2. + a0[k0])*dt_h
                vk1 =  -csv[k1] + (((((((b[6, k1]*7.*h_const[n]/8. + b[5, k1])*6.*h_const[n]/7. + b[4, k1])*5.*h_const[n]/6. + b[3, k1])*4.*h_const[n]/5. + b[2, k1])*3.*h_const[n]/4. + b[1, k1])*2.*h_const[n]/3. + b[0, k1])*h_const[n]/2. + a0[k1])*dt_h
                vk2 =  -csv[k2] + (((((((b[6, k2]*7.*h_const[n]/8. + b[5, k2])*6.*h_const[n]/7. + b[4, k2])*5.*h_const[n]/6. + b[3, k2])*4.*h_const[n]/5. + b[2, k2])*3.*h_const[n]/4. + b[1, k2])*2.*h_const[n]/3. + b[0, k2])*h_const[n]/2. + a0[k2])*dt_h
                
                arr[i, 3] = vk0 + v0[k0]
                arr[i, 4] = vk1 + v0[k1]
                arr[i, 5] = vk2 + v0[k2]

            compute_all_accelerations(arr, num_bodies, G_val, c2, has_j2, has_gr, phys_star_idx, 
                                      oblate_indices, oblate_j2, oblate_j4, oblate_req, oblate_poles)

            for i in range(num_bodies):
                at[3*i+0] = arr[i, 6]
                at[3*i+1] = arr[i, 7]
                at[3*i+2] = arr[i, 8]

            if n == 1:
                for k in range(N3):
                    tmp = g[0, k]
                    g[0, k] = (at[k] - a0[k]) / rr_const[0]
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], g[0, k] - tmp)
            elif n == 2:
                for k in range(N3):
                    tmp = g[1, k]
                    g[1, k] = ((at[k] - a0[k])/rr_const[1] - g[0, k])/rr_const[2]
                    tmp = g[1, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[0])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp)
            elif n == 3:
                for k in range(N3):
                    tmp = g[2, k]
                    g[2, k] = (((at[k] - a0[k])/rr_const[3] - g[0, k])/rr_const[4] - g[1, k])/rr_const[5]
                    tmp = g[2, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[1])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[2])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp)
            elif n == 4:
                for k in range(N3):
                    tmp = g[3, k]
                    g[3, k] = ((((at[k] - a0[k])/rr_const[6] - g[0, k])/rr_const[7] - g[1, k])/rr_const[8] - g[2, k])/rr_const[9]
                    tmp = g[3, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[3])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[4])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp * c_const[5])
                    b[3, k], csb[3, k] = _ias15_add_cs(b[3, k], csb[3, k], tmp)
            elif n == 5:
                for k in range(N3):
                    tmp = g[4, k]
                    g[4, k] = (((((at[k] - a0[k])/rr_const[10] - g[0, k])/rr_const[11] - g[1, k])/rr_const[12] - g[2, k])/rr_const[13] - g[3, k])/rr_const[14]
                    tmp = g[4, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[6])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[7])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp * c_const[8])
                    b[3, k], csb[3, k] = _ias15_add_cs(b[3, k], csb[3, k], tmp * c_const[9])
                    b[4, k], csb[4, k] = _ias15_add_cs(b[4, k], csb[4, k], tmp)
            elif n == 6:
                for k in range(N3):
                    tmp = g[5, k]
                    g[5, k] = ((((((at[k] - a0[k])/rr_const[15] - g[0, k])/rr_const[16] - g[1, k])/rr_const[17] - g[2, k])/rr_const[18] - g[3, k])/rr_const[19] - g[4, k])/rr_const[20]
                    tmp = g[5, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[10])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[11])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp * c_const[12])
                    b[3, k], csb[3, k] = _ias15_add_cs(b[3, k], csb[3, k], tmp * c_const[13])
                    b[4, k], csb[4, k] = _ias15_add_cs(b[4, k], csb[4, k], tmp * c_const[14])
                    b[5, k], csb[5, k] = _ias15_add_cs(b[5, k], csb[5, k], tmp)
            elif n == 7:
                maxak = 0.0
                maxb6ktmp = 0.0
                for k in range(N3):
                    tmp = g[6, k]
                    g[6, k] = (((((((at[k] - a0[k])/rr_const[21] - g[0, k])/rr_const[22] - g[1, k])/rr_const[23] - g[2, k])/rr_const[24] - g[3, k])/rr_const[25] - g[4, k])/rr_const[26] - g[5, k])/rr_const[27]
                    tmp = g[6, k] - tmp
                    b[0, k], csb[0, k] = _ias15_add_cs(b[0, k], csb[0, k], tmp * c_const[15])
                    b[1, k], csb[1, k] = _ias15_add_cs(b[1, k], csb[1, k], tmp * c_const[16])
                    b[2, k], csb[2, k] = _ias15_add_cs(b[2, k], csb[2, k], tmp * c_const[17])
                    b[3, k], csb[3, k] = _ias15_add_cs(b[3, k], csb[3, k], tmp * c_const[18])
                    b[4, k], csb[4, k] = _ias15_add_cs(b[4, k], csb[4, k], tmp * c_const[19])
                    b[5, k], csb[5, k] = _ias15_add_cs(b[5, k], csb[5, k], tmp * c_const[20])
                    b[6, k], csb[6, k] = _ias15_add_cs(b[6, k], csb[6, k], tmp)

                    ak = abs(at[k])
                    if ak > maxak: maxak = ak
                    b6ktmp = abs(tmp)
                    if b6ktmp > maxb6ktmp: maxb6ktmp = b6ktmp
                    
                if maxak > 0.0:
                    predictor_corrector_error = maxb6ktmp / maxak
                else:
                    predictor_corrector_error = 0.0

        # End for Gauss-Radau nodes
    # End Predictor-corrector loop
    
    # Calculate dt_new
    min_timescale2 = 1e300
    for i in range(num_bodies):
        a0i = 0.0
        y2 = 0.0
        y3 = 0.0
        y4 = 0.0
        for k in range(3*i, 3*i+3):
            a0i += a0[k]*a0[k]
            tmp2 = a0[k] + b[0, k] + b[1, k] + b[2, k] + b[3, k] + b[4, k] + b[5, k] + b[6, k]
            y2 += tmp2*tmp2
            tmp3 = b[0, k] + 2.*b[1, k] + 3.*b[2, k] + 4.*b[3, k] + 5.*b[4, k] + 6.*b[5, k] + 7.*b[6, k]
            y3 += tmp3*tmp3
            tmp4 = 2.*b[1, k] + 6.*b[2, k] + 12.*b[3, k] + 20.*b[4, k] + 30.*b[5, k] + 42.*b[6, k]
            y4 += tmp4*tmp4
            
        if a0i == 0.0: continue
        
        timescale2 = 2.0 * y2 / (y3 + math.sqrt(y4 * y2))
        if timescale2 < min_timescale2:
            min_timescale2 = timescale2
            
    if min_timescale2 < 1e299:
        dt_new = math.sqrt(min_timescale2) * dt * ias15_sqrt7(epsilon * 5040.0)
    else:
        dt_new = dt / safety_factor

    if abs(dt_new) < min_dt:
        dt_new = math.copysign(min_dt, dt_new)
        
    if abs(dt_new / dt) < safety_factor:
        for i in range(num_bodies):
            k0, k1, k2 = 3*i, 3*i+1, 3*i+2
            arr[i, 0] = x0[k0]
            arr[i, 1] = x0[k1]
            arr[i, 2] = x0[k2]
            arr[i, 3] = v0[k0]
            arr[i, 4] = v0[k1]
            arr[i, 5] = v0[k2]
            arr[i, 6] = a0[k0]
            arr[i, 7] = a0[k1]
            arr[i, 8] = a0[k2]
            
        if dt_last_done != 0.0:
            ratio = dt_new / dt_last_done
            _ias15_predict_next_step(ratio, N3, er, br, e_arr, b)
            
        return dt_new, 0.0, False

    if abs(dt_new / dt) > 1.0:
        if dt_new / dt > 1.0 / safety_factor:
            dt_new = dt / safety_factor

    for k in range(N3):
        dt2 = dt * dt
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[6, k]/72. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[5, k]/56. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[4, k]/42. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[3, k]/30. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[2, k]/20. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[1, k]/12. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], b[0, k]/6. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], a0[k]/2. * dt2)
        x0[k], csx[k] = _ias15_add_cs(x0[k], csx[k], v0[k] * dt)

        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[6, k]/8. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[5, k]/7. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[4, k]/6. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[3, k]/5. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[2, k]/4. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[1, k]/3. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], b[0, k]/2. * dt)
        v0[k], csv[k] = _ias15_add_cs(v0[k], csv[k], a0[k] * dt)

    for i in range(num_bodies):
        arr[i, 0] = x0[3*i+0]
        arr[i, 1] = x0[3*i+1]
        arr[i, 2] = x0[3*i+2]
        arr[i, 3] = v0[3*i+0]
        arr[i, 4] = v0[3*i+1]
        arr[i, 5] = v0[3*i+2]
        
    for i in range(7 * N3):
        er.flat[i] = e_arr.flat[i]
        br.flat[i] = b.flat[i]
        
    ratio = dt_new / dt
    _ias15_predict_next_step(ratio, N3, e_arr, b, e_arr, b)
    
    return dt_new, dt, True


class Particle:
    def __init__(self, sim, idx):
        self.sim = sim
        self.idx = idx
        
    @property
    def x(self): return self.sim.arr[self.idx, 0]
    @x.setter
    def x(self, val): self.sim.arr[self.idx, 0] = val
    
    @property
    def y(self): return self.sim.arr[self.idx, 1]
    @y.setter
    def y(self, val): self.sim.arr[self.idx, 1] = val
    
    @property
    def z(self): return self.sim.arr[self.idx, 2]
    @z.setter
    def z(self, val): self.sim.arr[self.idx, 2] = val
    
    @property
    def vx(self): return self.sim.arr[self.idx, 3]
    @vx.setter
    def vx(self, val): self.sim.arr[self.idx, 3] = val
    
    @property
    def vy(self): return self.sim.arr[self.idx, 4]
    @vy.setter
    def vy(self, val): self.sim.arr[self.idx, 4] = val
    
    @property
    def vz(self): return self.sim.arr[self.idx, 5]
    @vz.setter
    def vz(self, val): self.sim.arr[self.idx, 5] = val
    
    @property
    def m(self): return self.sim.arr[self.idx, 9]
    @m.setter
    def m(self, val): self.sim.arr[self.idx, 9] = val
    
    @property
    def hash(self): return self.sim.hashes[self.idx]

class ParticleList:
    def __init__(self, sim):
        self.sim = sim
    def __getitem__(self, key):
        if isinstance(key, str):
            for i, h in enumerate(self.sim.hashes):
                if h == key: return Particle(self.sim, i)
            raise KeyError(f"Particle {key} not found")
        elif isinstance(key, int):
            if key < 0 or key >= self.sim.N: raise IndexError()
            return Particle(self.sim, key)
        elif isinstance(key, slice):
            indices = range(*key.indices(self.sim.N))
            return [Particle(self.sim, i) for i in indices]
        raise TypeError("Invalid key type")
    def __len__(self):
        return self.sim.N

class Simulation:
    def __init__(self):
        self.G = 1.0
        self.t = 0.0
        self.dt = 1e-4
        self.dt_last_done = 0.0
        self.N = 0
        self.arr = np.zeros((0, 10), dtype=np.float64)
        self.hashes = []
        self.particles = ParticleList(self)
        self.has_j2 = False
        self.has_gr = False
        self.phys_star_idx = -1
        self.oblate_indices = np.empty(0, dtype=np.int32)
        self.oblate_j2 = np.empty(0, dtype=np.float64)
        self.oblate_j4 = np.empty(0, dtype=np.float64)
        self.oblate_req = np.empty(0, dtype=np.float64)
        self.oblate_poles = np.empty((0, 3), dtype=np.float64)
        
        self._g = np.zeros((7, 0), dtype=np.float64)
        self._e = np.zeros((7, 0), dtype=np.float64)
        self._b = np.zeros((7, 0), dtype=np.float64)
        self._csb = np.zeros((7, 0), dtype=np.float64)
        self._er = np.zeros((7, 0), dtype=np.float64)
        self._br = np.zeros((7, 0), dtype=np.float64)
        self._at = np.zeros(0, dtype=np.float64)
        self._x0 = np.zeros(0, dtype=np.float64)
        self._v0 = np.zeros(0, dtype=np.float64)
        self._a0 = np.zeros(0, dtype=np.float64)
        self._csx = np.zeros(0, dtype=np.float64)
        self._csv = np.zeros(0, dtype=np.float64)

    def reset_integrator_state(self):
        self._g.fill(0.0)
        self._e.fill(0.0)
        self._b.fill(0.0)
        self._csb.fill(0.0)
        self._er.fill(0.0)
        self._br.fill(0.0)
        self._at.fill(0.0)
        self._x0.fill(0.0)
        self._v0.fill(0.0)
        self._a0.fill(0.0)
        self._csx.fill(0.0)
        self._csv.fill(0.0)
        self.dt_last_done = 0.0

    def _resize_buffers(self, n):
        n3 = n * 3
        if n3 > len(self._at):
            self._g = np.zeros((7, n3), dtype=np.float64)
            self._e = np.zeros((7, n3), dtype=np.float64)
            self._b = np.zeros((7, n3), dtype=np.float64)
            self._csb = np.zeros((7, n3), dtype=np.float64)
            self._er = np.zeros((7, n3), dtype=np.float64)
            self._br = np.zeros((7, n3), dtype=np.float64)
            self._at = np.zeros(n3, dtype=np.float64)
            self._x0 = np.zeros(n3, dtype=np.float64)
            self._v0 = np.zeros(n3, dtype=np.float64)
            self._a0 = np.zeros(n3, dtype=np.float64)
            self._csx = np.zeros(n3, dtype=np.float64)
            self._csv = np.zeros(n3, dtype=np.float64)

    def add(self, m=0.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, hash=None, primary=None, a=0.0, e=0.0, inc=0.0, Omega=0.0, omega=0.0, M=0.0):
        if primary is not None:
            mu = self.G * (primary.m + m)
            pos, vel = orbital_to_cartesian(a, e, inc, Omega, omega, M, mu)
            x = primary.x + pos[0]
            y = primary.y + pos[1]
            z = primary.z + pos[2]
            vx = primary.vx + vel[0]
            vy = primary.vy + vel[1]
            vz = primary.vz + vel[2]
            
        new_row = np.array([[x, y, z, vx, vy, vz, 0.0, 0.0, 0.0, m]], dtype=np.float64)
        self.arr = np.vstack([self.arr, new_row])
        self.hashes.append(hash)
        self.N += 1
        self._resize_buffers(self.N)
        self.reset_integrator_state()

    def remove(self, index=None, hash=None):
        if hash is not None:
            for i, h in enumerate(self.hashes):
                if h == hash:
                    index = i
                    break
        if index is not None and 0 <= index < self.N:
            self.arr = np.delete(self.arr, index, axis=0)
            self.hashes.pop(index)
            self.N -= 1
            
            # Note: We must reset integrator state if a particle is removed or added
            # because the history buffers `b`, `e` will be invalidated.
            self._resize_buffers(self.N)
            self.reset_integrator_state()
            
    def move_to_com(self):
        if self.N == 0: return
        M_tot = np.sum(self.arr[:, 9])
        if M_tot == 0: return
        
        com_x = np.sum(self.arr[:, 0] * self.arr[:, 9]) / M_tot
        com_y = np.sum(self.arr[:, 1] * self.arr[:, 9]) / M_tot
        com_z = np.sum(self.arr[:, 2] * self.arr[:, 9]) / M_tot
        com_vx = np.sum(self.arr[:, 3] * self.arr[:, 9]) / M_tot
        com_vy = np.sum(self.arr[:, 4] * self.arr[:, 9]) / M_tot
        com_vz = np.sum(self.arr[:, 5] * self.arr[:, 9]) / M_tot
        
        self.arr[:, 0] -= com_x
        self.arr[:, 1] -= com_y
        self.arr[:, 2] -= com_z
        self.arr[:, 3] -= com_vx
        self.arr[:, 4] -= com_vy
        self.arr[:, 5] -= com_vz

    def copy(self):
        new_sim = Simulation()
        new_sim.G = self.G
        new_sim.t = self.t
        new_sim.dt = self.dt
        new_sim.dt_last_done = self.dt_last_done
        new_sim.N = self.N
        new_sim.arr = self.arr.copy()
        new_sim.hashes = self.hashes.copy()
        new_sim.has_j2 = self.has_j2
        new_sim.has_gr = self.has_gr
        new_sim.phys_star_idx = self.phys_star_idx
        new_sim.oblate_indices = self.oblate_indices.copy()
        new_sim.oblate_j2 = self.oblate_j2.copy()
        new_sim.oblate_j4 = self.oblate_j4.copy()
        new_sim.oblate_req = self.oblate_req.copy()
        new_sim.oblate_poles = self.oblate_poles.copy()
        
        new_sim._resize_buffers(self.N)
        new_sim._g = self._g.copy()
        new_sim._e = self._e.copy()
        new_sim._b = self._b.copy()
        new_sim._csb = self._csb.copy()
        new_sim._er = self._er.copy()
        new_sim._br = self._br.copy()
        new_sim._at = self._at.copy()
        new_sim._x0 = self._x0.copy()
        new_sim._v0 = self._v0.copy()
        new_sim._a0 = self._a0.copy()
        new_sim._csx = self._csx.copy()
        new_sim._csv = self._csv.copy()
        return new_sim

    def integrate(self, t_target):
        c2 = C_AU_YR * C_AU_YR
        while True:
            t_diff = t_target - self.t
            if abs(t_diff) < 1e-12:
                break
                
            dt_step = self.dt
            
            # Ensure dt_step has the same sign as t_diff
            if (t_diff > 0 and dt_step < 0) or (t_diff < 0 and dt_step > 0):
                dt_step = -dt_step
                
            if t_diff > 0 and dt_step > t_diff:
                dt_step = t_diff
            elif t_diff < 0 and dt_step < t_diff:
                dt_step = t_diff
                
            self.dt, dt_done, success = ias15_step_numba(
                self.arr, self.N, self.G, c2, self.has_j2, self.has_gr, self.phys_star_idx,
                self.oblate_indices, self.oblate_j2, self.oblate_j4, self.oblate_req, self.oblate_poles,
                dt_step, 0.0, 1e-4, self.dt_last_done,
                self._g, self._e, self._b, self._csb, self._er, self._br,
                self._at, self._x0, self._v0, self._a0, self._csx, self._csv,
                _IAS15_H, _IAS15_RR, _IAS15_C, _IAS15_D
            )
            
            if success:
                self.t += dt_done
                self.dt_last_done = dt_done
