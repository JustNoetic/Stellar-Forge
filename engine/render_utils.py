import math
import numpy as np
from numba import njit
import datetime
from constants import *

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

def format_time_speed(multiplier):
    """Convert a speed multiplier (in seconds/second) to a human-readable string."""
    sign_str = "-" if multiplier < 0 else ""
    multiplier = abs(multiplier)
    if multiplier < 1.5:
        return f"{sign_str}Realtime" if multiplier > 0.5 else "Paused"
    elif multiplier < 60:
        return f"{sign_str}{multiplier:.1f} sec/s"
    elif multiplier < 3600:
        return f"{sign_str}{multiplier / 60:.1f} min/s"
    elif multiplier < 86400:
        return f"{sign_str}{multiplier / 3600:.1f} hr/s"
    elif multiplier < 604800:
        return f"{sign_str}{multiplier / 86400:.1f} days/s"
    elif multiplier < 2629800:
        return f"{sign_str}{multiplier / 604800:.1f} weeks/s"
    elif multiplier < 31557600:
        return f"{sign_str}{multiplier / 2629800:.1f} months/s"
    else:
        return f"{sign_str}{multiplier / 31557600:.1f} years/s"

def format_sim_time(t_years):
    try:
        _mgr = globals().get('sys_mgr_spice')
        if _mgr is not None and getattr(_mgr, 'kernels_loaded', False):
            import spiceypy as spice
            epoch_dt = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
            et_epoch = _mgr.datetime_to_et(epoch_dt)
            et = et_epoch + t_years * 365.25 * 86400.0
            utc_str = spice.et2utc(et, 'C', 0)
            dt_utc = datetime.datetime.strptime(utc_str, "%Y %b %d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
            return dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute
    except:
        pass

    epoch = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    delta_seconds = t_years * 365.25 * 86400
    try:
        dt_utc = epoch + datetime.timedelta(seconds=delta_seconds)
        return dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute
    except (OverflowError, OSError, ValueError):
        return 9999, 12, 31, 23, 59

def sim_time_from_date(y, m, d, h=0, mn=0):
    epoch = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    try:
        dt_utc = datetime.datetime(int(y), int(m), int(d), int(h), int(mn), 0, tzinfo=datetime.timezone.utc)
        delta = dt_utc - epoch
        return delta.total_seconds() / (365.25 * 86400)
    except ValueError:
        return 0.0

@njit(cache=True)
def compute_ring_culling(
    pos_rel_all, body_radii, star_idx, 
    n_casters, caster_indices, 
    n_rings, ring_centers, ring_normals, ring_outer_radii,
    num_bodies, ring_caster_lo, ring_caster_hi
):
    ring_caster_lo[:] = 0
    ring_caster_hi[:] = 0
    
    star_pos = pos_rel_all[star_idx]
    star_r = body_radii[star_idx]
    
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
def extract_frustum_planes(vp):
    planes = np.empty((6, 4), dtype=np.float32)
    planes[0, :] = vp[:, 3] + vp[:, 0]
    planes[1, :] = vp[:, 3] - vp[:, 0]
    planes[2, :] = vp[:, 3] + vp[:, 1]
    planes[3, :] = vp[:, 3] - vp[:, 1]
    planes[4, :] = vp[:, 3] + vp[:, 2]
    planes[5, :] = vp[:, 3] - vp[:, 2]
    for i in range(6):
        x, y, z = planes[i, 0], planes[i, 1], planes[i, 2]
        length = math.sqrt(x*x + y*y + z*z)
        if length > 1e-8:
            planes[i] /= length
    return planes

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

