import math
import numpy as np
from numba import njit
from engine.core.constants import SOLAR_RADIUS_KM
from engine.core.math_utils import pole_to_ecliptic
from engine.physics.atmosphere_physics import compute_atmosphere_properties, compute_mie_coefficients

def compute_max_bend(caster_r_au, atmo_h_km, refractivity=0.00029, beta_ext=None):
    """Maximum atmospheric refraction angle (radians) for a spherical shell,
    accounting for transmission optical depth through the planetary limb.
    
    Rays traversing the limb cannot emerge from altitudes where optical depth is
    opaque (tau > tau_cutoff). For thin atmospheres (Earth, Titan, Mars) this
    reaches the surface, giving the true uncapped surface refraction. For hyper-dense
    atmospheres (Venus), this evaluates at the transmission mesosphere (~60-70 km).
    """
    if atmo_h_km <= 0.0 or caster_r_au <= 0.0:
        return 0.0
    caster_r_km = caster_r_au * 149597870.7
    val = (3.141592653589793 * caster_r_km) / max(1e-6, atmo_h_km * 2.0)
    surface_bend = 2.0 * max(refractivity, 0.0) * math.sqrt(val)
    
    # Transmission limit: twilight/limb grazing optical depth threshold (evaluated at the
    # penetrating red wavelength where Rayleigh scattering is minimal and twilight transmission peaks)
    tau_cutoff = 35.0
    if beta_ext is not None:
        try:
            beta_val = float(np.min(beta_ext)) if hasattr(beta_ext, '__iter__') else float(beta_ext)
        except Exception:
            beta_val = (max(refractivity, 0.0) / 0.00029) * 5.4e-6
    else:
        # Physical Gladstone-Dale scaling for red wavelength Rayleigh scattering
        beta_val = (max(refractivity, 0.0) / 0.00029) * 5.4e-6
        
    path_len_m = math.sqrt(2.0 * math.pi * caster_r_km * 1000.0 * atmo_h_km * 1000.0)
    tau_slant_0 = max(0.0, beta_val * path_len_m)
    
    if tau_slant_0 > tau_cutoff:
        trans_factor = tau_cutoff / tau_slant_0
        max_bend = surface_bend * trans_factor
    else:
        max_bend = surface_bend
        
    # Numerical safety envelope: clamp between 0.001 rad (~0.05 deg) and 0.10 rad (~5.7 deg)
    return max(0.001, min(0.10, max_bend))


@njit(cache=True)
def compute_ring_coplanar_masks(n_ring_planes, centers, normals):
    masks = np.zeros(16, dtype=np.uint32)
    for k in range(n_ring_planes):
        mask = 0
        c_k = centers[k]
        n_k = normals[k]
        for j in range(n_ring_planes):
            c_j = centers[j]
            n_j = normals[j]
            dx = c_k[0] - c_j[0]
            dy = c_k[1] - c_j[1]
            dz = c_k[2] - c_j[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            dot_prod = n_k[0] * n_j[0] + n_k[1] * n_j[1] + n_k[2] * n_j[2]
            if dist < 1e-5 and dot_prod > 0.999:
                mask |= (1 << j)
        masks[k] = mask
    return masks

def get_cached_atmosphere_properties(atmo, mass_sm):
    mass_kg = mass_sm * 1.98847e30
    R_km = atmo.get('planet_radius_km', 0.0)
    
    if not atmo.get('_dirty', True) and \
       atmo.get('_cached_mass') == mass_sm and \
       atmo.get('_cached_radius') == R_km:
        return atmo['_cached_props'], atmo['_cached_trans'], atmo['_cached_thick']

    g_m_s2 = (6.67430e-11 * mass_kg) / ((R_km * 1000.0) ** 2) if R_km > 0 else 9.81
    props = compute_atmosphere_properties(
        atmo.get('surface_pressure', 1.0),
        atmo.get('temperature', 288.15),
        atmo.get('composition', {"N2": 0.78, "O2": 0.21, "Ar": 0.01}),
        g_m_s2
    )
    
    beta_r = np.nan_to_num(props['beta_rayleigh'], nan=0.0, posinf=0.0, neginf=0.0)
    beta_m = max(0.0, float(atmo.get('beta_mie', 2.0e-6)))
    h_r = max(0.0, float(props['scale_height_km']))
    h_m = max(0.0, float(atmo.get('h_mie', 1.2)))
    r_km_safe = max(0.0, float(R_km))
    
    od_r = beta_r * 1000.0 * math.sqrt(2.0 * math.pi * r_km_safe * h_r)
    mie_coeffs = np.nan_to_num(compute_mie_coefficients(beta_m, atmo.get('mie_angstrom', None)), nan=0.0, posinf=0.0, neginf=0.0)
    od_m = mie_coeffs * 1000.0 * math.sqrt(2.0 * math.pi * r_km_safe * h_m)

    z_o3_peak_km = max(1.0, float(props.get('ozone_peak_km', 25.0)))
    ozone_w_km = max(0.1, float(props.get('ozone_width_km', 6.0)))
    ozone_slant_km = math.sqrt(2.0 * math.pi * r_km_safe * ozone_w_km)
    beta_layered = np.nan_to_num(props['beta_abs_layered'], nan=0.0, posinf=0.0, neginf=0.0)
    beta_mixed = np.nan_to_num(props['beta_abs_mixed'], nan=0.0, posinf=0.0, neginf=0.0)
    
    od_o3 = beta_layered * 1000.0 * ozone_slant_km
    od_mixed = beta_mixed * 1000.0 * math.sqrt(2.0 * math.pi * r_km_safe * h_r)
    
    props['tau_R0'] = od_r.astype(np.float32)
    props['tau_O3_peak'] = od_o3.astype(np.float32)
    
    tau_vert_r = beta_r * 1000.0 * h_r
    tau_vert_m = mie_coeffs * 1000.0 * h_m
    tau_vert_mixed = beta_mixed * 1000.0 * h_r
    props['tau_vertical'] = (tau_vert_r + tau_vert_m + tau_vert_mixed).astype(np.float32)
    
    tau = od_r + od_m + od_o3 + od_mixed
    direct_trans = np.exp(-tau)
    forward_scatter = tau * np.exp(-tau * 0.8) * 0.3
    multi_scatter = 0.02 * np.exp(-tau * 0.2)
    trans = np.clip(direct_trans + forward_scatter + multi_scatter, 0.0, 1.0)
    trans = np.nan_to_num(trans, nan=1.0)
    thick = max(0.0, float(atmo.get('atmo_radius_au', 0.0) - atmo.get('surface_radius_au', 0.0)))
    
    atmo['_cached_mass'] = mass_sm
    atmo['_cached_radius'] = R_km
    atmo['_cached_props'] = props
    atmo['_cached_trans'] = trans
    atmo['_cached_thick'] = thick
    atmo['_dirty'] = False
    
    return props, trans, thick

def _build_tex_idx_arr(bodies_data, texture_slices):
    """Build the per-body texture-array slice index (1-based; 0 = no texture)."""
    n = len(bodies_data)
    arr = np.zeros(n, dtype='f4')
    for i, b in enumerate(bodies_data):
        name_lower = b['name'].lower()
        if name_lower in texture_slices:
            arr[i] = float(texture_slices[name_lower])
    return arr

def _build_rotation_props(bodies_data):
    """Build per-body rotation properties."""
    n = len(bodies_data)
    name_to_idx = {b.get('name'): i for i, b in enumerate(bodies_data)}
    mass_by_name = {b.get('name'): float(b.get('m', 0.0)) for b in bodies_data}
    
    rot_period_arr = np.zeros(n, dtype='f4')
    w0_arr = np.zeros(n, dtype='f4')
    tidally_locked_arr = np.zeros(n, dtype=bool)
    parent_idx_arr = np.full(n, -1, dtype=int)
    
    pole_n_arr = np.zeros((n, 3), dtype='f8')
    tangent_arr = np.zeros((n, 3), dtype='f8')
    bitangent_arr = np.zeros((n, 3), dtype='f8')
    
    for i, b in enumerate(bodies_data):
        parent_name = b.get('parent', b.get('parentId'))
        if parent_name in name_to_idx:
            parent_idx_arr[i] = name_to_idx[parent_name]
            
        w0_deg = b.get('W0', 0.0)
        w0_arr[i] = math.radians(float(w0_deg))
        
        is_locked = b.get('tidally_locked', False)
        r_hours = b.get('rotation_period', None)
        
        if is_locked or b.get('type') == 'Moon':
            tidally_locked_arr[i] = True
            
        if r_hours is None or r_hours == 0.0:
            body_m = float(b.get('m', 0.0))
            r_rsun = float(b.get('r', 0.0))
            a_val = float(b.get('a', 0.0))
            parent_m = mass_by_name.get(parent_name, 0.0) if parent_name else 0.0
            if parent_m > 0.0 and body_m > 0.0 and r_rsun > 0.0 and a_val > 0.0:
                r_km = r_rsun * SOLAR_RADIUS_KM
                r_tid = 0.00084 * ((parent_m ** 2 / body_m) ** (1.0 / 6.0)) * math.sqrt(r_km)
                if a_val < r_tid:
                    tidally_locked_arr[i] = True
                    total_m = parent_m + body_m
                    p_years = math.sqrt(a_val ** 3 / total_m) if total_m > 0.0 else 0.0
                    r_hours = p_years * 365.25 * 24.0
            if r_hours is None or r_hours == 0.0:
                r_hours = 24.0
                
        rot_period_arr[i] = float(r_hours) * 3600.0
        
        pole_ra = float(b.get('pole_ra', 0.0))
        pole_dec = float(b.get('pole_dec', 90.0))
        pole_ecl = pole_to_ecliptic(pole_ra, pole_dec)
        pole_ren = np.array([pole_ecl[0], pole_ecl[2], -pole_ecl[1]], dtype=np.float64)
        pole_norm = np.linalg.norm(pole_ren)
        pole_n = pole_ren / pole_norm if pole_norm > 0 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(pole_n, ref)) > 0.999:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tangent = np.cross(pole_n, ref)
        t_norm = np.linalg.norm(tangent)
        if t_norm > 0:
            tangent /= t_norm
        bitangent = np.cross(pole_n, tangent)
        b_norm = np.linalg.norm(bitangent)
        if b_norm > 0:
            bitangent /= b_norm
            
        pole_n_arr[i] = pole_n
        tangent_arr[i] = tangent
        bitangent_arr[i] = bitangent
        
    return rot_period_arr, w0_arr, tidally_locked_arr, parent_idx_arr, pole_n_arr, tangent_arr, bitangent_arr

@njit(cache=True)
def compute_body_rotation_angles_jit(sim_t_sec, rot_period_arr, w0_arr, tidally_locked_arr, parent_idx_arr, pos_snap_render, pole_n_arr, tangent_arr, bitangent_arr):
    n = len(rot_period_arr)
    angles = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if tidally_locked_arr[i] and parent_idx_arr[i] >= 0:
            p_idx = parent_idx_arr[i]
            r_moon = pos_snap_render[i]
            r_parent = pos_snap_render[p_idx]
            
            to_parent_x = r_parent[0] - r_moon[0]
            to_parent_y = r_parent[1] - r_moon[1]
            to_parent_z = r_parent[2] - r_moon[2]
            
            norm = math.sqrt(to_parent_x*to_parent_x + to_parent_y*to_parent_y + to_parent_z*to_parent_z)
            if norm > 1e-12:
                to_parent_nx = to_parent_x / norm
                to_parent_ny = to_parent_y / norm
                to_parent_nz = to_parent_z / norm
                
                pole_n = pole_n_arr[i]
                tangent = tangent_arr[i]
                bitangent = bitangent_arr[i]
                
                dot_val = to_parent_nx * pole_n[0] + to_parent_ny * pole_n[1] + to_parent_nz * pole_n[2]
                
                d_eq_x = to_parent_nx - dot_val * pole_n[0]
                d_eq_y = to_parent_ny - dot_val * pole_n[1]
                d_eq_z = to_parent_nz - dot_val * pole_n[2]
                
                d_norm = math.sqrt(d_eq_x*d_eq_x + d_eq_y*d_eq_y + d_eq_z*d_eq_z)
                if d_norm > 1e-12:
                    d_eq_nx = d_eq_x / d_norm
                    d_eq_ny = d_eq_y / d_norm
                    d_eq_nz = d_eq_z / d_norm
                    
                    cos_w = d_eq_nx * tangent[0] + d_eq_ny * tangent[1] + d_eq_nz * tangent[2]
                    sin_w = d_eq_nx * bitangent[0] + d_eq_ny * bitangent[1] + d_eq_nz * bitangent[2]
                    
                    angles[i] = math.atan2(sin_w, cos_w)
                else:
                    angles[i] = (w0_arr[i] + (sim_t_sec / rot_period_arr[i]) * (2.0 * math.pi)) % (2.0 * math.pi) if rot_period_arr[i] != 0.0 else 0.0
            else:
                angles[i] = (w0_arr[i] + (sim_t_sec / rot_period_arr[i]) * (2.0 * math.pi)) % (2.0 * math.pi) if rot_period_arr[i] != 0.0 else 0.0
        else:
            w0 = w0_arr[i]
            p_sec = rot_period_arr[i]
            if p_sec != 0.0:
                angles[i] = (w0 + (sim_t_sec / p_sec) * (2.0 * math.pi)) % (2.0 * math.pi)
            else:
                angles[i] = w0
    return angles

@njit
def compute_planetshine_numba(pos, radii, colors, is_star, star_positions, star_colors, star_lums, star_radii, hdr_enabled):
    N = len(pos)
    num_stars = len(star_positions)
    
    planetshine_dirs = np.zeros((N, 3), dtype=np.float32)
    planetshine_colors = np.zeros((N, 3), dtype=np.float32)
    
    if num_stars == 0:
        return planetshine_dirs, planetshine_colors
        
    j_lit = np.zeros((N, num_stars), dtype=np.float32)
    star_dirs = np.zeros((N, num_stars, 3), dtype=np.float32)
    
    for j in range(N):
        if is_star[j] > 0.5: 
            continue
            
        for s in range(num_stars):
            cx = star_positions[s, 0] - pos[j, 0]
            cy = star_positions[s, 1] - pos[j, 1]
            cz = star_positions[s, 2] - pos[j, 2]
            c_dist = np.sqrt(cx*cx + cy*cy + cz*cz)
            
            if c_dist < 1e-6: 
                continue
                
            cx /= c_dist
            cy /= c_dist
            cz /= c_dist
            
            star_dirs[j, s, 0] = cx
            star_dirs[j, s, 1] = cy
            star_dirs[j, s, 2] = cz
            
            shadow_factor = 1.0
            for k in range(N):
                if k == j or is_star[k] > 0.5: 
                    continue
                
                vk_x = pos[k, 0] - pos[j, 0]
                vk_y = pos[k, 1] - pos[j, 1]
                vk_z = pos[k, 2] - pos[j, 2]
                t = vk_x * cx + vk_y * cy + vk_z * cz
                
                if t > 0.0 and t < c_dist:
                    dist_sq_k = vk_x*vk_x + vk_y*vk_y + vk_z*vk_z
                    perp_sq = max(0.0, dist_sq_k - t*t)
                    
                    r_penumbra = radii[k] + t * (star_radii[s] / c_dist)
                    if perp_sq < r_penumbra * r_penumbra:
                        inv_t = 1.0 / t
                        beta = radii[k] * inv_t
                        gamma = np.sqrt(perp_sq) * inv_t
                        alpha = star_radii[s] / c_dist
                        
                        p_out = alpha + beta
                        p_in = abs(beta - alpha)
                        
                        if gamma < p_in:
                            occ = 1.0
                        else:
                            t_val = max(0.0, min(1.0, (gamma - p_out) / (p_in - p_out + 1e-12)))
                            occ = t_val * t_val * (3.0 - 2.0 * t_val)
                            
                        max_occ = min(1.0, (beta * beta) / max(1e-12, alpha * alpha))
                        scale_area = min(1.0, (radii[k] / max(1e-6, radii[j])) ** 2)
                        max_occ *= scale_area
                        shadow_factor *= (1.0 - max_occ * occ)
                        
            if shadow_factor > 0.001:
                irradiance = (star_lums[s] / max(c_dist * c_dist, 1e-8)) if hdr_enabled else 1.0
                j_lit[j, s] = shadow_factor * irradiance

    for i in range(N):
        if is_star[i] > 0.5:
            continue
            
        pos_i = pos[i]
        
        total_dir_x, total_dir_y, total_dir_z = 0.0, 0.0, 0.0
        total_color_r, total_color_g, total_color_b = 0.0, 0.0, 0.0
        total_weight = 0.0
        
        for j in range(N):
            if i == j or is_star[j] > 0.5:
                continue
                
            pos_j = pos[j]
            r_j = radii[j]
            
            dx = pos_j[0] - pos_i[0]
            dy = pos_j[1] - pos_i[1]
            dz = pos_j[2] - pos_i[2]
            dist_sq = dx*dx + dy*dy + dz*dz
            
            min_dist = r_j * 1.05
            max_dist = r_j * 300.0 
            if dist_sq < min_dist * min_dist or dist_sq > max_dist * max_dist:
                continue
                
            solid_angle = (r_j * r_j) / dist_sq
            if solid_angle < 1e-8:
                continue
            
            for s in range(num_stars):
                if j_lit[j, s] < 1e-6:
                    continue
                
                cx = star_dirs[j, s, 0]
                cy = star_dirs[j, s, 1]
                cz = star_dirs[j, s, 2]
                
                shift_x = pos_j[0] + cx * r_j * 0.7
                shift_y = pos_j[1] + cy * r_j * 0.7
                shift_z = pos_j[2] + cz * r_j * 0.7
                
                dir_to_caster_x = shift_x - pos_i[0]
                dir_to_caster_y = shift_y - pos_i[1]
                dir_to_caster_z = shift_z - pos_i[2]
                
                d_c_sq = dir_to_caster_x**2 + dir_to_caster_y**2 + dir_to_caster_z**2
                d_c = np.sqrt(d_c_sq)
                if d_c > 1e-6:
                    dir_to_caster_x /= d_c
                    dir_to_caster_y /= d_c
                    dir_to_caster_z /= d_c
                else:
                    continue

                cos_a = cx * (-dir_to_caster_x) + cy * (-dir_to_caster_y) + cz * (-dir_to_caster_z)
                cos_a = max(-1.0, min(1.0, cos_a))
                a = np.arccos(cos_a)
                phase = (np.sin(a) + (np.pi - a) * cos_a) / np.pi
                
                light_r = star_colors[s, 0] * phase * j_lit[j, s]
                light_g = star_colors[s, 1] * phase * j_lit[j, s]
                light_b = star_colors[s, 2] * phase * j_lit[j, s]
            
                boost = 1.5 
                bounce_r = colors[j, 0] * light_r * solid_angle * (2.0 / 3.0) * boost
                bounce_g = colors[j, 1] * light_g * solid_angle * (2.0 / 3.0) * boost
                bounce_b = colors[j, 2] * light_b * solid_angle * (2.0 / 3.0) * boost
                
                lum = bounce_r * 0.2126 + bounce_g * 0.7152 + bounce_b * 0.0722
                
                total_dir_x += dir_to_caster_x * lum
                total_dir_y += dir_to_caster_y * lum
                total_dir_z += dir_to_caster_z * lum
                
                total_color_r += bounce_r
                total_color_g += bounce_g
                total_color_b += bounce_b
                total_weight += lum
            
        if total_weight > 1e-12:
            inv_w = 1.0 / total_weight
            planetshine_dirs[i, 0] = total_dir_x * inv_w
            planetshine_dirs[i, 1] = total_dir_y * inv_w
            planetshine_dirs[i, 2] = total_dir_z * inv_w
        
        planetshine_colors[i, 0] = total_color_r
        planetshine_colors[i, 1] = total_color_g
        planetshine_colors[i, 2] = total_color_b
            
    return planetshine_dirs, planetshine_colors

def halton(index, base):
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f = f / base
    return result
