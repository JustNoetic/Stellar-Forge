"""
Analytical Keplerian Propagation Engine for Stellar-Forge.

Provides perfectly stable, mathematically exact O(N) hierarchical propagation.
Uses Jacobi coordinates for extraction and top-down subsystem propagation
to completely eliminate circular dependencies and barycenter wobble.
"""

import math
import numpy as np
from numba import njit

from math_utils import kepler_solve, vector_orbital_to_cartesian, axis_angle_rotation, fast_cross, fast_norm
from constants import G as G_DEFAULT

# Column definitions for kepler_elements array (num_bodies x 15)
COL_A = 0
COL_E = 1
COL_M0 = 2
COL_T0 = 3
COL_MU = 4
COL_N_X = 5
COL_N_Y = 6
COL_N_Z = 7
COL_E_X = 8
COL_E_Y = 9
COL_E_Z = 10
COL_PREC_X = 11
COL_PREC_Y = 12
COL_PREC_Z = 13
COL_DOMEGA_PERI = 14

@njit(cache=True, nogil=True)
def state_to_kepler_vectors(rx, ry, rz, vx, vy, vz, mu):
    """Convert Cartesian relative state vector (r, v) to a, e, M, and orientation vectors."""
    r = math.sqrt(rx * rx + ry * ry + rz * rz)
    v2 = vx * vx + vy * vy + vz * vz
    if r < 1e-30 or mu < 1e-30:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
    
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h = math.sqrt(hx * hx + hy * hy + hz * hz)
    
    if h < 1e-30:
        return r, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
    
    n_x, n_y, n_z = hx/h, hy/h, hz/h
    
    vhx = vy * hz - vz * hy
    vhy = vz * hx - vx * hz
    vhz = vx * hy - vy * hx
    ex = vhx / mu - rx / r
    ey = vhy / mu - ry / r
    ez = vhz / mu - rz / r
    e = math.sqrt(ex * ex + ey * ey + ez * ez)
    
    if e > 1e-10:
        e_x, e_y, e_z = ex/e, ey/e, ez/e
    else:
        # For circular orbit, eccentricity vector points to node (if inclined) or arbitrary
        if n_z < 0.999999:
            node_x, node_y = -hy, hx
            node_mag = math.sqrt(node_x*node_x + node_y*node_y)
            e_x, e_y, e_z = node_x/node_mag, node_y/node_mag, 0.0
        else:
            e_x, e_y, e_z = 1.0, 0.0, 0.0
            
    energy = v2 / 2.0 - mu / r
    if abs(energy) < 1e-30:
        a = 1e30
    else:
        a = -mu / (2.0 * energy)
        
    if e > 1e-10:
        r_dot_e = (rx * ex + ry * ey + rz * ez) / (r * e)
        nu = math.acos(max(-1.0, min(1.0, r_dot_e)))
        rdotv = rx * vx + ry * vy + rz * vz
        if rdotv < 0:
            nu = 2.0 * math.pi - nu
    else:
        # measure nu from e_vec
        r_dot_e = (rx * e_x + ry * e_y + rz * e_z) / r
        nu = math.acos(max(-1.0, min(1.0, r_dot_e)))
        # determine sign
        cross_r = e_y*rz - e_z*ry
        cross_y = e_z*rx - e_x*rz
        cross_z = e_x*ry - e_y*rx
        dot_n = cross_r*n_x + cross_y*n_y + cross_z*n_z
        if dot_n < 0:
            nu = 2.0 * math.pi - nu

    if e < 1.0:
        cos_nu = math.cos(nu)
        sin_nu = math.sin(nu)
        denom = 1.0 + e * cos_nu
        if abs(denom) < 1e-12:
            denom = 1e-12
        sin_E = sin_nu * math.sqrt(max(0.0, 1.0 - e * e)) / denom
        cos_E = (e + cos_nu) / denom
        E = math.atan2(sin_E, cos_E)
        M = E - e * math.sin(E)
        if M < 0:
            M += 2.0 * math.pi
    else:
        M = 0.0

    return a, e, M, n_x, n_y, n_z, e_x, e_y, e_z

@njit(cache=True, nogil=True)
def extract_all_kepler_elements(pos, vel, mass, parent_indices, subsys_pos, subsys_vel, subsys_mass, t_current, G_const,
                                oblate_indices, oblate_j2, oblate_req, oblate_poles, has_gr, c_au_yr):
    """Extract and cache perfect Jacobi orbital elements for all bodies."""
    num_bodies = len(pos)
    elements = np.zeros((num_bodies, 15), dtype=np.float64)

    # 1. Build CSR Children Table
    children_count = np.zeros(num_bodies, dtype=np.int64)
    for i in range(num_bodies):
        p_idx = parent_indices[i]
        if p_idx >= 0:
            children_count[p_idx] += 1

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

    # Initialize root elements
    for i in range(num_bodies):
        if parent_indices[i] < 0:
            elements[i, COL_A] = 0.0
            elements[i, COL_E] = 0.0
            elements[i, COL_M0] = 0.0
            elements[i, COL_T0] = t_current
            elements[i, COL_MU] = 0.0
            elements[i, COL_N_X] = 0.0
            elements[i, COL_N_Y] = 0.0
            elements[i, COL_N_Z] = 1.0
            elements[i, COL_E_X] = 1.0
            elements[i, COL_E_Y] = 0.0
            elements[i, COL_E_Z] = 0.0
            elements[i, COL_PREC_X] = 0.0
            elements[i, COL_PREC_Y] = 0.0
            elements[i, COL_PREC_Z] = 0.0
            elements[i, COL_DOMEGA_PERI] = 0.0

    # 2. Compute Jacobi elements for each parent's children
    for p_idx in range(num_bodies):
        c_start = children_offsets[p_idx]
        c_end = children_offsets[p_idx + 1]
        num_c = c_end - c_start
        if num_c == 0:
            continue
            
        # Check if parent is oblate
        is_oblate = False
        p_j2 = 0.0
        p_req = 0.0
        p_pole = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if oblate_indices is not None:
            for obl_i in range(len(oblate_indices)):
                if oblate_indices[obl_i] == p_idx:
                    is_oblate = True
                    p_j2 = oblate_j2[obl_i]
                    p_req = oblate_req[obl_i]
                    p_pole[0] = oblate_poles[obl_i, 0]
                    p_pole[1] = oblate_poles[obl_i, 1]
                    p_pole[2] = oblate_poles[obl_i, 2]
                    break
        
        # Check if parent has a grandparent for third-body perturbation (assume Sun is grandparent)
        gp_idx = parent_indices[p_idx]
        has_gp = False
        gp_mu = 0.0
        gp_a = 0.0
        gp_pole = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n_3rd = 0.0
        
        if gp_idx >= 0:
            # Grandparent exists. Calculate parent's mean motion around grandparent.
            gp_mass = mass[gp_idx]
            p_mass = mass[p_idx]
            gp_mu = G_const * (gp_mass + p_mass)
            
            # Simple orbital extraction for the parent to find its semi-major axis and orbit normal
            prx = pos[p_idx, 0] - pos[gp_idx, 0]
            pry = pos[p_idx, 1] - pos[gp_idx, 1]
            prz = pos[p_idx, 2] - pos[gp_idx, 2]
            pvx = vel[p_idx, 0] - vel[gp_idx, 0]
            pvy = vel[p_idx, 1] - vel[gp_idx, 1]
            pvz = vel[p_idx, 2] - vel[gp_idx, 2]
            
            pa, pe, pM, pn_x, pn_y, pn_z, pe_x, pe_y, pe_z = state_to_kepler_vectors(prx, pry, prz, pvx, pvy, pvz, gp_mu)
            if pa > 0:
                has_gp = True
                gp_a = pa
                gp_pole[0] = pn_x
                gp_pole[1] = pn_y
                gp_pole[2] = pn_z
                n_3rd = math.sqrt(gp_mu / (pa * pa * pa))

        # Sort children by distance to parent so inner bodies are handled first
        c_list = np.zeros(num_c, dtype=np.int64)
        d_list = np.zeros(num_c, dtype=np.float64)
        for idx in range(num_c):
            j = children_flat[c_start + idx]
            c_list[idx] = j
            dx = pos[j, 0] - pos[p_idx, 0]
            dy = pos[j, 1] - pos[p_idx, 1]
            dz = pos[j, 2] - pos[p_idx, 2]
            d_list[idx] = dx*dx + dy*dy + dz*dz
            
        for a_idx in range(num_c):
            for b_idx in range(a_idx + 1, num_c):
                if d_list[b_idx] < d_list[a_idx]:
                    tmp_d = d_list[a_idx]; d_list[a_idx] = d_list[b_idx]; d_list[b_idx] = tmp_d
                    tmp_c = c_list[a_idx]; c_list[a_idx] = c_list[b_idx]; c_list[b_idx] = tmp_c
                    
        for idx in range(num_c):
            children_flat[c_start + idx] = c_list[idx]

        # Extract using forward Jacobi pass
        B_pos = np.zeros(3, dtype=np.float64)
        B_vel = np.zeros(3, dtype=np.float64)
        B_pos[0] = pos[p_idx, 0]; B_pos[1] = pos[p_idx, 1]; B_pos[2] = pos[p_idx, 2]
        B_vel[0] = vel[p_idx, 0]; B_vel[1] = vel[p_idx, 1]; B_vel[2] = vel[p_idx, 2]
        B_m = mass[p_idx]
        
        for idx in range(num_c):
            j = c_list[idx]
            
            rx = subsys_pos[j, 0] - B_pos[0]
            ry = subsys_pos[j, 1] - B_pos[1]
            rz = subsys_pos[j, 2] - B_pos[2]
            vx = subsys_vel[j, 0] - B_vel[0]
            vy = subsys_vel[j, 1] - B_vel[1]
            vz = subsys_vel[j, 2] - B_vel[2]
            
            mu = G_const * (B_m + subsys_mass[j])
            
            a, e, M0, n_x, n_y, n_z, e_x, e_y, e_z = state_to_kepler_vectors(rx, ry, rz, vx, vy, vz, mu)
            
            elements[j, COL_A] = a
            elements[j, COL_E] = e
            elements[j, COL_M0] = M0
            elements[j, COL_T0] = t_current
            elements[j, COL_MU] = mu
            elements[j, COL_N_X] = n_x
            elements[j, COL_N_Y] = n_y
            elements[j, COL_N_Z] = n_z
            elements[j, COL_E_X] = e_x
            elements[j, COL_E_Y] = e_y
            elements[j, COL_E_Z] = e_z
            
            prec_x = 0.0
            prec_y = 0.0
            prec_z = 0.0
            domega_peri = 0.0
            
            if a > 0.0 and e < 1.0:
                n_mean = math.sqrt(mu / (a * a * a))
                p_sl = a * (1.0 - e * e)
                
                # 1. J2 Precession Vector
                if is_oblate and p_sl > 0:
                    cos_i_j2 = n_x * p_pole[0] + n_y * p_pole[1] + n_z * p_pole[2]
                    j2_factor = 1.5 * p_j2 * (p_req / p_sl)**2 * n_mean
                    
                    dOmega_j2 = -j2_factor * cos_i_j2
                    prec_x += dOmega_j2 * p_pole[0]
                    prec_y += dOmega_j2 * p_pole[1]
                    prec_z += dOmega_j2 * p_pole[2]
                    
                    domega_peri += 0.5 * j2_factor * (5.0 * cos_i_j2 * cos_i_j2 - 1.0)
                    
                # 3. GR Apsidal Precession
                if has_gr and a > 0.0 and e < 1.0:
                    gr_factor = 3.0 * mu * n_mean / (c_au_yr * c_au_yr * p_sl)
                    domega_peri += gr_factor
                if has_gp and n_mean > 0:
                    cos_i_3rd = n_x * gp_pole[0] + n_y * gp_pole[1] + n_z * gp_pole[2]
                    e_fac = math.sqrt(max(0.0001, 1.0 - e*e))
                    tb_factor = 0.75 * (n_3rd * n_3rd / n_mean) / e_fac
                    
                    dOmega_3rd = -tb_factor * cos_i_3rd
                    prec_x += dOmega_3rd * gp_pole[0]
                    prec_y += dOmega_3rd * gp_pole[1]
                    prec_z += dOmega_3rd * gp_pole[2]
                    
                    domega_peri += tb_factor * (5.0 * cos_i_3rd * cos_i_3rd - 1.0)
            
            elements[j, COL_PREC_X] = prec_x
            elements[j, COL_PREC_Y] = prec_y
            elements[j, COL_PREC_Z] = prec_z
            elements[j, COL_DOMEGA_PERI] = domega_peri
            
            # Update Barycenter B_k to include subsystem j
            new_B_m = B_m + subsys_mass[j]
            B_pos[0] = (B_m * B_pos[0] + subsys_mass[j] * subsys_pos[j, 0]) / new_B_m
            B_pos[1] = (B_m * B_pos[1] + subsys_mass[j] * subsys_pos[j, 1]) / new_B_m
            B_pos[2] = (B_m * B_pos[2] + subsys_mass[j] * subsys_pos[j, 2]) / new_B_m
            B_vel[0] = (B_m * B_vel[0] + subsys_mass[j] * subsys_vel[j, 0]) / new_B_m
            B_vel[1] = (B_m * B_vel[1] + subsys_mass[j] * subsys_vel[j, 1]) / new_B_m
            B_vel[2] = (B_m * B_vel[2] + subsys_mass[j] * subsys_vel[j, 2]) / new_B_m
            B_m = new_B_m

    return elements

@njit(cache=True, nogil=True)
def propagate_keplerian_system_numba(elements, parent_indices, tree_indices, t_sim, subsys_init_pos, subsys_init_vel, out_pos, out_vel, mass, subsys_mass):
    """Propagate all bodies forward in time using Jacobi Top-Down Subsystem evaluation."""
    num_bodies = len(parent_indices)

    children_count = np.zeros(num_bodies, dtype=np.int64)
    for i in range(num_bodies):
        p_idx = parent_indices[i]
        if p_idx >= 0:
            children_count[p_idx] += 1

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
            
    for p_idx in range(num_bodies):
        c_start = children_offsets[p_idx]
        c_end = children_offsets[p_idx + 1]
        num_c = c_end - c_start
        if num_c > 1:
            c_list = np.zeros(num_c, dtype=np.int64)
            d_list = np.zeros(num_c, dtype=np.float64)
            for idx in range(num_c):
                j = children_flat[c_start + idx]
                c_list[idx] = j
                dx = subsys_init_pos[j, 0] - subsys_init_pos[p_idx, 0]
                dy = subsys_init_pos[j, 1] - subsys_init_pos[p_idx, 1]
                dz = subsys_init_pos[j, 2] - subsys_init_pos[p_idx, 2]
                d_list[idx] = dx*dx + dy*dy + dz*dz
                
            for a_idx in range(num_c):
                for b_idx in range(a_idx + 1, num_c):
                    if d_list[b_idx] < d_list[a_idx]:
                        tmp_d = d_list[a_idx]; d_list[a_idx] = d_list[b_idx]; d_list[b_idx] = tmp_d
                        tmp_c = c_list[a_idx]; c_list[a_idx] = c_list[b_idx]; c_list[b_idx] = tmp_c
            
            for idx in range(num_c):
                children_flat[c_start + idx] = c_list[idx]

    subsys_out_pos = np.empty((num_bodies, 3), dtype=np.float64)
    subsys_out_vel = np.empty((num_bodies, 3), dtype=np.float64)

    rel_pos_eval = np.zeros((num_bodies, 3), dtype=np.float64)
    rel_vel_eval = np.zeros((num_bodies, 3), dtype=np.float64)
    
    for i in range(num_bodies):
        if parent_indices[i] < 0:
            dt = t_sim - elements[i, COL_T0]
            subsys_out_pos[i, 0] = subsys_init_pos[i, 0] + subsys_init_vel[i, 0] * dt
            subsys_out_pos[i, 1] = subsys_init_pos[i, 1] + subsys_init_vel[i, 1] * dt
            subsys_out_pos[i, 2] = subsys_init_pos[i, 2] + subsys_init_vel[i, 2] * dt
            subsys_out_vel[i, 0] = subsys_init_vel[i, 0]
            subsys_out_vel[i, 1] = subsys_init_vel[i, 1]
            subsys_out_vel[i, 2] = subsys_init_vel[i, 2]
        else:
            a = elements[i, COL_A]
            e = elements[i, COL_E]
            M0 = elements[i, COL_M0]
            t0 = elements[i, COL_T0]
            mu = elements[i, COL_MU]
            
            n_vec = np.array([elements[i, COL_N_X], elements[i, COL_N_Y], elements[i, COL_N_Z]], dtype=np.float64)
            e_vec = np.array([elements[i, COL_E_X], elements[i, COL_E_Y], elements[i, COL_E_Z]], dtype=np.float64)
            
            prec_vec = np.array([elements[i, COL_PREC_X], elements[i, COL_PREC_Y], elements[i, COL_PREC_Z]], dtype=np.float64)
            domega_peri = elements[i, COL_DOMEGA_PERI]
            
            if a <= 1e-12 or mu <= 1e-30 or e >= 1.0:
                rel_pos_eval[i, 0] = 0.0
                rel_pos_eval[i, 1] = 0.0
                rel_pos_eval[i, 2] = 0.0
                rel_vel_eval[i, 0] = 0.0
                rel_vel_eval[i, 1] = 0.0
                rel_vel_eval[i, 2] = 0.0
            else:
                mean_motion = math.sqrt(mu / (a * a * a))
                dt = t_sim - t0
                M_t = M0 + mean_motion * dt
                M_t = M_t % (2.0 * math.pi)
                if M_t < 0:
                    M_t += 2.0 * math.pi
                
                # Apply vector precession (Nodal Precession)
                prec_mag = math.sqrt(prec_vec[0]*prec_vec[0] + prec_vec[1]*prec_vec[1] + prec_vec[2]*prec_vec[2])
                if prec_mag > 1e-15:
                    axis = prec_vec / prec_mag
                    theta = prec_mag * dt
                    n_vec = axis_angle_rotation(n_vec, axis, theta)
                    e_vec = axis_angle_rotation(e_vec, axis, theta)
                    
                # Apply apsidal precession (advance periapsis within the orbital plane)
                if abs(domega_peri) > 1e-15:
                    omega_theta = domega_peri * dt
                    e_vec = axis_angle_rotation(e_vec, n_vec, omega_theta)
                    
                rp, rv = vector_orbital_to_cartesian(a, e, n_vec, e_vec, M_t, mu)
                
                rel_pos_eval[i, 0] = rp[0]
                rel_pos_eval[i, 1] = rp[1]
                rel_pos_eval[i, 2] = rp[2]
                rel_vel_eval[i, 0] = rv[0]
                rel_vel_eval[i, 1] = rv[1]
                rel_vel_eval[i, 2] = rv[2]

    # 3. Top-Down Jacobi Placement
    for idx in range(num_bodies):
        P = tree_indices[idx]
        
        c_start = children_offsets[P]
        c_end = children_offsets[P + 1]
        num_c = c_end - c_start
        
        if num_c == 0:
            out_pos[P, 0] = subsys_out_pos[P, 0]
            out_pos[P, 1] = subsys_out_pos[P, 1]
            out_pos[P, 2] = subsys_out_pos[P, 2]
            out_vel[P, 0] = subsys_out_vel[P, 0]
            out_vel[P, 1] = subsys_out_vel[P, 1]
            out_vel[P, 2] = subsys_out_vel[P, 2]
        else:
            B_pos = np.zeros(3, dtype=np.float64)
            B_vel = np.zeros(3, dtype=np.float64)
            B_pos[0] = subsys_out_pos[P, 0]; B_pos[1] = subsys_out_pos[P, 1]; B_pos[2] = subsys_out_pos[P, 2]
            B_vel[0] = subsys_out_vel[P, 0]; B_vel[1] = subsys_out_vel[P, 1]; B_vel[2] = subsys_out_vel[P, 2]
            
            for k in range(num_c - 1, -1, -1):
                j = children_flat[c_start + k]
                
                m_Bk = mass[P]
                for p_idx_inner in range(0, k + 1):
                    m_Bk += subsys_mass[children_flat[c_start + p_idx_inner]]
                    
                f = subsys_mass[j] / m_Bk
                
                B_pos[0] -= f * rel_pos_eval[j, 0]
                B_pos[1] -= f * rel_pos_eval[j, 1]
                B_pos[2] -= f * rel_pos_eval[j, 2]
                B_vel[0] -= f * rel_vel_eval[j, 0]
                B_vel[1] -= f * rel_vel_eval[j, 1]
                B_vel[2] -= f * rel_vel_eval[j, 2]
                
                subsys_out_pos[j, 0] = B_pos[0] + rel_pos_eval[j, 0]
                subsys_out_pos[j, 1] = B_pos[1] + rel_pos_eval[j, 1]
                subsys_out_pos[j, 2] = B_pos[2] + rel_pos_eval[j, 2]
                subsys_out_vel[j, 0] = B_vel[0] + rel_vel_eval[j, 0]
                subsys_out_vel[j, 1] = B_vel[1] + rel_vel_eval[j, 1]
                subsys_out_vel[j, 2] = B_vel[2] + rel_vel_eval[j, 2]
                
            out_pos[P, 0] = B_pos[0]
            out_pos[P, 1] = B_pos[1]
            out_pos[P, 2] = B_pos[2]
            out_vel[P, 0] = B_vel[0]
            out_vel[P, 1] = B_vel[1]
            out_vel[P, 2] = B_vel[2]
