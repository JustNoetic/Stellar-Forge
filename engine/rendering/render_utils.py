import math
import numpy as np
from numba import njit
import datetime
from engine.core.constants import *

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

def format_distance_au(dist_au, threshold_au=None, precision=5, show_unit=True):
    """Format a distance in AU, dynamically switching to light years ('ly') if crossing threshold_au."""
    if threshold_au is None:
        threshold_au = DEFAULT_LY_THRESHOLD_AU
    abs_dist = abs(dist_au)
    if threshold_au > 0 and abs_dist >= threshold_au:
        dist_ly = dist_au / LY_TO_AU
        unit_str = " ly" if show_unit else ""
        return f"{dist_ly:.{precision}f}{unit_str}"
    else:
        unit_str = " AU" if show_unit else ""
        return f"{dist_au:.{precision}f}{unit_str}"

def format_sim_time(t_years):
    # Try spiceypy first if ephemeris mode is active
    try:
        from app import _mgr
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
        # Fallback to simple math for extreme years beyond 9999
        total_days = t_years * 365.25
        y = 2026 + int(total_days // 365.25)
        rem_days = total_days % 365.25
        
        rem_days += 0.5 # Add the 12-hour offset from Jan 1 12:00
        if rem_days >= 365.25:
            y += 1
            rem_days -= 365.25
            
        m = 1
        days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
        if is_leap: days_in_month[2] = 29
        
        d = int(rem_days) + 1
        rem_hours = (rem_days - int(rem_days)) * 24
        
        for i in range(1, 13):
            if d > days_in_month[i]:
                d -= days_in_month[i]
                m += 1
            else:
                break
                
        if m > 12: m, d = 12, 31
        h = int(rem_hours)
        mn = int((rem_hours - h) * 60)
        return y, m, d, h, mn

def sim_time_from_date(y, m, d, h=0, mn=0):
    if 1 <= y <= 9999:
        try:
            epoch = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
            dt_utc = datetime.datetime(int(y), int(m), int(d), int(h), int(mn), 0, tzinfo=datetime.timezone.utc)
            delta = dt_utc - epoch
            return delta.total_seconds() / (365.25 * 86400)
        except ValueError:
            pass
            
    # For extreme years, use a simplified Julian year progression
    dy = y - 2026
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_ytd = d - 1
    for i in range(1, int(m)):
        days_ytd += days_in_month[i]
        if i == 2 and y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
            days_ytd += 1
            
    total_days = dy * 365.25 + days_ytd + (h - 12) / 24.0 + mn / 1440.0
    return total_days / 365.25

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
    planes = np.empty((6, 4), dtype=vp.dtype)
    planes[0, :] = vp[:, 3] + vp[:, 0]
    planes[1, :] = vp[:, 3] - vp[:, 0]
    planes[2, :] = vp[:, 3] + vp[:, 1]
    planes[3, :] = vp[:, 3] - vp[:, 1]
    planes[4, :] = vp[:, 3] + vp[:, 2]
    planes[5, :] = vp[:, 3] - vp[:, 2]
    for i in range(6):
        x, y, z = planes[i, 0], planes[i, 1], planes[i, 2]
        length = math.sqrt(x*x + y*y + z*z)
        if length > 1e-12:
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

def generate_ring_shadow_grad(sorted_gradient, tex_sampled=None, res=4096):
    if sorted_gradient:
        grad_p = np.array([g['p'] for g in sorted_gradient])
        grad_a = np.array([g['a'] for g in sorted_gradient])
        tex_p = np.linspace(0.0, 1.0, res)
        shadow_grad_a = np.interp(tex_p, grad_p, grad_a).astype('f4')
    else:
        shadow_grad_a = np.ones(res, dtype='f4')
        
    shadow_grad = np.ones((res, 4), dtype='f4')
    if tex_sampled is not None:
        shadow_grad[:, 0:3] = tex_sampled[:, 0:3]
        shadow_grad[:, 3] = tex_sampled[:, 3] * shadow_grad_a
    else:
        shadow_grad[:, 3] = shadow_grad_a
    return shadow_grad

def compute_5_ring_colors(tex_sampled=None, raw_color=(1.0, 1.0, 1.0), gradient=None):
    colors = np.zeros((5, 3), dtype='f4')
    base_color = np.array(raw_color, dtype='f4')
    if tex_sampled is not None:
        n_samples = len(tex_sampled)
        band_size = n_samples // 5
        for k in range(5):
            start = k * band_size
            end = (k + 1) * band_size if k < 4 else n_samples
            slice_data = tex_sampled[start:end]
            rgbs = slice_data[:, 0:3]
            alphas = slice_data[:, 3:4]
            total_alpha = np.sum(alphas)
            if total_alpha > 1e-4:
                avg_rgb = np.sum(rgbs * alphas, axis=0) / total_alpha
            else:
                avg_rgb = np.mean(rgbs, axis=0)
            colors[k] = avg_rgb * base_color
    elif gradient and len(gradient) > 0:
        grad_p = np.array([g['p'] for g in gradient])
        grad_a = np.array([g['a'] for g in gradient])
        u_pts = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype='f4')
        alphas = np.interp(u_pts, grad_p, grad_a)
        for k in range(5):
            colors[k] = base_color * alphas[k]
    else:
        for k in range(5):
            colors[k] = base_color
    return colors

def generate_ring_geometry(pole_render, min_r, max_r):
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
    
    r_grid = min_r + p_grid * (max_r - min_r)
    flat_r = r_grid.ravel()
    flat_ct = np.cos(theta_grid).ravel()
    flat_st = np.sin(theta_grid).ravel()
    n_verts = len(flat_r)
    
    eq_pos = np.zeros((n_verts, 3))
    eq_pos[:, 0] = flat_r * flat_ct
    eq_pos[:, 2] = flat_r * flat_st
    verts = (R_ring @ eq_pos.T).T.astype('f4')
    
    normals = np.tile(pole_n, (n_verts, 1)).astype('f4')
    
    stride = RADIAL_SUBDIVISIONS + 1
    ii, jj = np.meshgrid(np.arange(RING_SEGMENTS), np.arange(RADIAL_SUBDIVISIONS), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    p00 = ii * stride + jj
    p01 = p00 + 1
    p10 = (ii + 1) * stride + jj
    p11 = p10 + 1
    indices = np.column_stack([p00, p01, p10, p10, p01, p11]).ravel().astype('i4')
    
    return verts, normals, indices

@njit(cache=True)
def bake_unified_shadow_profile_jit(tex_radii, segment_inners, segment_outers, segment_opacities, segment_shadow_grads):
    res = len(tex_radii)
    n_segs = len(segment_inners)
    combined = np.zeros((res, 4), dtype=np.float32)
    
    for i in range(res):
        r = tex_radii[i]
        
        # Accumulate RGB and Alpha for this radius
        existing_alpha = 0.0
        existing_r = 0.0
        existing_g = 0.0
        existing_b = 0.0
        
        for j in range(n_segs):
            inner = segment_inners[j]
            outer = segment_outers[j]
            if r >= inner and r <= outer:
                # Interpolate in this segment's shadow_grad
                seg_res = segment_shadow_grads[j].shape[0]
                t = (r - inner) / max(1e-6, outer - inner)
                if t < 0.0: t = 0.0
                if t > 1.0: t = 1.0
                
                # Simple linear interpolation
                idx_float = t * (seg_res - 1)
                idx_low = int(math.floor(idx_float))
                idx_high = int(math.ceil(idx_float))
                weight = idx_float - idx_low
                
                grad = segment_shadow_grads[j]
                
                # Interpolated RGB and Alpha
                r_val = (1.0 - weight) * grad[idx_low, 0] + weight * grad[idx_high, 0]
                g_val = (1.0 - weight) * grad[idx_low, 1] + weight * grad[idx_high, 1]
                b_val = (1.0 - weight) * grad[idx_low, 2] + weight * grad[idx_high, 2]
                a_val = (1.0 - weight) * grad[idx_low, 3] + weight * grad[idx_high, 3]
                
                seg_alpha = a_val * segment_opacities[j]
                
                # Combine alpha (transmittance-based blending)
                new_alpha = existing_alpha + seg_alpha - existing_alpha * seg_alpha
                
                # Blend RGB based on alpha weights
                total_alpha = existing_alpha + seg_alpha
                if total_alpha > 1e-6:
                    existing_r = (existing_r * existing_alpha + r_val * seg_alpha) / total_alpha
                    existing_g = (existing_g * existing_alpha + g_val * seg_alpha) / total_alpha
                    existing_b = (existing_b * existing_alpha + b_val * seg_alpha) / total_alpha
                else:
                    existing_r = r_val
                    existing_g = g_val
                    existing_b = b_val
                    
                existing_alpha = new_alpha
                
        combined[i, 0] = existing_r
        combined[i, 1] = existing_g
        combined[i, 2] = existing_b
        combined[i, 3] = existing_alpha
        
    return combined


def bake_unified_shadow_profile(active_rings, min_r, max_r, res=4096):
    n = len(active_rings)
    tex_radii = np.linspace(min_r, max_r, res, dtype=np.float32)
    segment_inners = np.array([r['inner_r'] for r in active_rings], dtype=np.float32)
    segment_outers = np.array([r['outer_r'] for r in active_rings], dtype=np.float32)
    segment_opacities = np.array([r['opacity'] for r in active_rings], dtype=np.float32)
    
    # Build 3D array of shadow_grads
    segment_shadow_grads = np.zeros((n, res, 4), dtype=np.float32)
    for j, ring in enumerate(active_rings):
        segment_shadow_grads[j] = ring['shadow_grad']
        
    return bake_unified_shadow_profile_jit(tex_radii, segment_inners, segment_outers, segment_opacities, segment_shadow_grads)


def rebuild_ring_gradients_atlas(ring_precomputed, ring_gradient_tex):
    # Group rings by body
    rings_by_body = {}
    for r in ring_precomputed:
        rings_by_body.setdefault(r['body_idx'], []).append(r)
        
    # Assign unified_idx for each body (up to 4)
    body_indices = {bi: i for i, bi in enumerate(rings_by_body.keys())}
    
    ring_gradient_data = np.zeros((16, 4096, 4), dtype='f4')
    
    # 1. Bake unified shadow profile for each body into rows 0 to len(body_indices)-1
    for bi, unified_idx in body_indices.items():
        if unified_idx >= 4:
            break
        body_rings = rings_by_body[bi]
        # Filter active rings (opacity >= 0.005)
        active_rings = [r for r in body_rings if r['opacity'] >= 0.005]
        if active_rings:
            min_r = min(r['inner_r'] for r in active_rings)
            max_r = max(r['outer_r'] for r in active_rings)
            combined_shadow = bake_unified_shadow_profile(active_rings, min_r, max_r)
            ring_gradient_data[unified_idx, :, :] = combined_shadow
            
    # 2. Copy individual segment profiles into rows 4 to 15
    for j, ring in enumerate(ring_precomputed):
        row_idx = 4 + j
        if row_idx >= 16:
            break
        ring_gradient_data[row_idx, :, :] = ring['shadow_grad']
        ring['row_idx'] = row_idx
        
    ring_gradient_tex.write(ring_gradient_data.tobytes())
    return body_indices


def rebuild_ring_render_group(bi, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex):
    rings = [r for r in ring_precomputed if r['body_idx'] == bi]
    
    existing_group = next((g for g in ring_render_groups if g['body_idx'] == bi), None)
    if existing_group:
        existing_group['vao'].release()
        if 'vbo' in existing_group: existing_group['vbo'].release()
        if 'ibo' in existing_group: existing_group['ibo'].release()
        ring_render_groups.remove(existing_group)
        
    if rings:
        min_r = min([r['inner_r'] for r in rings])
        max_r = max([r['outer_r'] for r in rings])
        pole_n = rings[0]['pole']
        
        verts, normals, indices = generate_ring_geometry(pole_n, min_r, max_r)
        
        n_v = len(verts)
        ring_packed = np.zeros((n_v, 6), dtype='f4')
        ring_packed[:, 0:3] = verts
        ring_packed[:, 3:6] = normals
        
        ring_vbo = ctx.buffer(ring_packed.tobytes())
        ring_ibo = ctx.buffer(indices.tobytes())
        ring_vao = ctx.vertex_array(
            prog_rings,
            [(ring_vbo, '3f 3f', 'in_position', 'in_normal')],
            index_buffer=ring_ibo
        )
        ring_render_groups.append({
            'body_idx': bi,
            'vao': ring_vao,
            'vbo': ring_vbo,
            'ibo': ring_ibo,
            'num_indices': len(indices),
        })
        
    body_indices = rebuild_ring_gradients_atlas(ring_precomputed, ring_gradient_tex)
    return body_indices


