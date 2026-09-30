"""
engine/rendering/terrain_quadtree.py

Spherified Cube Quadtree LOD system for planetary terrain (SpaceEngine style).
Recursively subdivides the 6 cube faces based on camera distance and Screen Space Error (SSE).
"""

import math
import numpy as np


def cube_to_sphere_point(face: int, u: float, v: float) -> np.ndarray:
    """
    Map normalized cube face coordinate (u, v) in [-1, 1] to a 3D unit sphere point
    using tangent-corrected mapping to minimize area distortion.
    """
    x_prime = math.tan(u * (math.pi / 4.0))
    y_prime = math.tan(v * (math.pi / 4.0))

    if face == 0:    # +X
        vx, vy, vz = 1.0, -y_prime, -x_prime
    elif face == 1:  # -X
        vx, vy, vz = -1.0, -y_prime, x_prime
    elif face == 2:  # +Y (North Pole)
        vx, vy, vz = x_prime, 1.0, y_prime
    elif face == 3:  # -Y (South Pole)
        vx, vy, vz = x_prime, -1.0, -y_prime
    elif face == 4:  # +Z
        vx, vy, vz = x_prime, -y_prime, 1.0
    elif face == 5:  # -Z
        vx, vy, vz = -x_prime, -y_prime, -1.0
    else:
        vx, vy, vz = 0.0, 0.0, 1.0

    length = math.sqrt(vx * vx + vy * vy + vz * vz)
    return np.array([vx / length, vy / length, vz / length], dtype=np.float32)


class QuadtreePatch:
    """Represents a single terrain patch at a specific face and LOD level."""
    __slots__ = (
        'face', 'lod', 'x', 'y',
        'min_u', 'max_u', 'min_v', 'max_v',
        'center', 'radius', 'center_dir'
    )

    def __init__(self, face: int, lod: int, x: int, y: int, radius_km: float):
        self.face = face
        self.lod = lod
        self.x = x
        self.y = y

        scale = 2.0 / (1 << lod)
        self.min_u = -1.0 + x * scale
        self.max_u = self.min_u + scale
        self.min_v = -1.0 + y * scale
        self.max_v = self.min_v + scale

        mid_u = 0.5 * (self.min_u + self.max_u)
        mid_v = 0.5 * (self.min_v + self.max_v)

        c_dir = cube_to_sphere_point(face, mid_u, mid_v)
        self.center_dir = c_dir
        self.center = c_dir * radius_km

        # Compute bounding sphere radius around the 4 corners
        p0 = cube_to_sphere_point(face, self.min_u, self.min_v) * radius_km
        p1 = cube_to_sphere_point(face, self.max_u, self.min_v) * radius_km
        p2 = cube_to_sphere_point(face, self.max_u, self.max_v) * radius_km
        p3 = cube_to_sphere_point(face, self.min_u, self.max_v) * radius_km

        d0 = np.linalg.norm(p0 - self.center)
        d1 = np.linalg.norm(p1 - self.center)
        d2 = np.linalg.norm(p2 - self.center)
        d3 = np.linalg.norm(p3 - self.center)
        self.radius = float(max(d0, d1, d2, d3) * 1.05)


try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(fastmath=True)
    def _cube_to_sphere_scalar(face, u, v):
        x_p = math.tan(u * (math.pi * 0.25))
        y_p = math.tan(v * (math.pi * 0.25))
        if face == 0:    vx, vy, vz = 1.0, -y_p, -x_p
        elif face == 1:  vx, vy, vz = -1.0, -y_p, x_p
        elif face == 2:  vx, vy, vz = x_p, 1.0, y_p
        elif face == 3:  vx, vy, vz = x_p, -1.0, -y_p
        elif face == 4:  vx, vy, vz = x_p, -y_p, 1.0
        else:            vx, vy, vz = -x_p, -y_p, -1.0
        inv_len = 1.0 / math.sqrt(vx * vx + vy * vy + vz * vz)
        return vx * inv_len, vy * inv_len, vz * inv_len

    @njit(fastmath=True)
    def _traverse_quadtree_jit(cam_pos, radius_km, obl, screen_height, tan_half_fov, threshold_px, max_lod, max_patches, frustum_planes, has_frustum, cloud_alt_km, refract_bend, height_max_km):
        out = np.empty((max_patches, 9), dtype=np.float32)
        count = 0
        
        inv_radius = 1.0 / max(1e-6, radius_km)
        inv_obl = 1.0 / max(1e-4, 1.0 - obl)

        # Transform camera into normalized spheroid-to-sphere space for horizon culling
        cam_x = cam_pos[0]
        cam_y = cam_pos[1]
        cam_z = cam_pos[2]

        cs_x = cam_x
        cs_y = cam_y * inv_obl
        cs_z = cam_z
        cs_dist = math.sqrt(cs_x*cs_x + cs_y*cs_y + cs_z*cs_z)
        inv_cs_dist = 1.0 / max(1e-6, cs_dist)
        cs_dx = cs_x * inv_cs_dist
        cs_dy = cs_y * inv_cs_dist
        cs_dz = cs_z * inv_cs_dist

        # Horizon angle from sphere center in scaled space
        # Elevated cloud shells and atmospheric refraction extend the geometric horizon beyond solid ground:
        eff_dist = max(radius_km + 1e-4, cs_dist)
        surf_horizon_angle = math.acos(min(1.0, radius_km / eff_dist))
        horizon_angle = surf_horizon_angle
        if cloud_alt_km > 0.0:
            cloud_r = radius_km + cloud_alt_km
            cloud_horizon_offset = math.acos(min(1.0, radius_km / max(1e-6, cloud_r)))
            horizon_angle += cloud_horizon_offset
        if refract_bend > 1e-6:
            refract_offset = min(0.05, refract_bend * 0.5)
            horizon_angle += refract_offset
        
        # Sort root faces by alignment with camera direction so that the face pointing
        # most directly at the camera is pushed last and popped FIRST (LIFO).
        # This guarantees front-facing patches always get top priority and never starve.
        face_order = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        face_dots = np.empty(6, dtype=np.float32)
        for f in range(6):
            if f == 0:    fnx, fny, fnz = 1.0, 0.0, 0.0
            elif f == 1:  fnx, fny, fnz = -1.0, 0.0, 0.0
            elif f == 2:  fnx, fny, fnz = 0.0, 1.0, 0.0
            elif f == 3:  fnx, fny, fnz = 0.0, -1.0, 0.0
            elif f == 4:  fnx, fny, fnz = 0.0, 0.0, 1.0
            else:         fnx, fny, fnz = 0.0, 0.0, -1.0
            face_dots[f] = fnx * cs_dx + fny * cs_dy + fnz * cs_dz

        # Insertion sort ascending (smallest dot pushed first, largest pushed last)
        for i in range(1, 6):
            key_f = face_order[i]
            key_dot = face_dots[key_f]
            j = i - 1
            while j >= 0 and face_dots[face_order[j]] > key_dot:
                face_order[j + 1] = face_order[j]
                j -= 1
            face_order[j + 1] = key_f

        stack = np.empty((512, 4), dtype=np.int32)
        s_ptr = 0
        for f_idx in range(6):
            stack[s_ptr, 0] = face_order[f_idx]
            stack[s_ptr, 1] = 0
            stack[s_ptr, 2] = 0
            stack[s_ptr, 3] = 0
            s_ptr += 1
            
        while s_ptr > 0:
            s_ptr -= 1
            face = stack[s_ptr, 0]
            lod = stack[s_ptr, 1]
            x = stack[s_ptr, 2]
            y = stack[s_ptr, 3]
            
            scale = 2.0 / (1 << lod)
            min_u = -1.0 + x * scale
            max_u = min_u + scale
            min_v = -1.0 + y * scale
            max_v = min_v + scale
            
            mid_u = min_u + scale * 0.5
            mid_v = min_v + scale * 0.5
            
            cnx, cny, cnz = _cube_to_sphere_scalar(face, mid_u, mid_v)
            cx = cnx * radius_km
            cy = cny * (1.0 - obl) * radius_km
            cz = cnz * radius_km
            
            p0x, p0y, p0z = _cube_to_sphere_scalar(face, min_u, min_v)
            p1x, p1y, p1z = _cube_to_sphere_scalar(face, max_u, min_v)
            p2x, p2y, p2z = _cube_to_sphere_scalar(face, max_u, max_v)
            p3x, p3y, p3z = _cube_to_sphere_scalar(face, min_u, max_v)
            
            d0 = math.sqrt((p0x*radius_km - cx)**2 + (p0y*(1.0 - obl)*radius_km - cy)**2 + (p0z*radius_km - cz)**2)
            d1 = math.sqrt((p1x*radius_km - cx)**2 + (p1y*(1.0 - obl)*radius_km - cy)**2 + (p1z*radius_km - cz)**2)
            d2 = math.sqrt((p2x*radius_km - cx)**2 + (p2y*(1.0 - obl)*radius_km - cy)**2 + (p2z*radius_km - cz)**2)
            d3 = math.sqrt((p3x*radius_km - cx)**2 + (p3y*(1.0 - obl)*radius_km - cy)**2 + (p3z*radius_km - cz)**2)
            radius = max(max(d0, d1), max(d2, d3)) * 1.05
            
            eff_radius = radius + max(0.0, cloud_alt_km) + max(0.0, height_max_km)
            
            # Precise horizon culling in spheroid-normalized space
            node_dot = cnx * cs_dx + cny * cs_dy + cnz * cs_dz
            angular_radius = min(1.0, eff_radius * inv_radius)
            cull_angle = horizon_angle + math.asin(angular_radius)
            if cull_angle < math.pi and node_dot < math.cos(cull_angle):
                continue

            # Precise view frustum culling in planet local space
            if has_frustum:
                dx_c = cx - cam_x
                dy_c = cy - cam_y
                dz_c = cz - cam_z
                dist_cam_sq = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c
                # Only test frustum planes if camera is outside the patch bounding sphere
                if dist_cam_sq > eff_radius * eff_radius:
                    in_frustum = True
                    for p_i in range(6):
                        d_plane = cx * frustum_planes[p_i, 0] + cy * frustum_planes[p_i, 1] + cz * frustum_planes[p_i, 2] + frustum_planes[p_i, 3]
                        if d_plane < -eff_radius:
                            in_frustum = False
                            break
                    if not in_frustum:
                        continue
                
            dx = cx - cam_x
            dy = cy - cam_y
            dz = cz - cam_z
            dist_to_center = math.sqrt(dx*dx + dy*dy + dz*dz)
            dist_to_surface = max(0.01, dist_to_center - radius - max(0.0, height_max_km))
            screen_size = (radius * screen_height) / (dist_to_surface * 2.0 * tan_half_fov)
            
            if screen_size > threshold_px and lod < max_lod and (s_ptr + 4 < 512):
                nl = lod + 1
                bx = x * 2
                by = y * 2
                stack[s_ptr, 0] = face; stack[s_ptr, 1] = nl; stack[s_ptr, 2] = bx;   stack[s_ptr, 3] = by;   s_ptr += 1
                stack[s_ptr, 0] = face; stack[s_ptr, 1] = nl; stack[s_ptr, 2] = bx+1; stack[s_ptr, 3] = by;   s_ptr += 1
                stack[s_ptr, 0] = face; stack[s_ptr, 1] = nl; stack[s_ptr, 2] = bx;   stack[s_ptr, 3] = by+1; s_ptr += 1
                stack[s_ptr, 0] = face; stack[s_ptr, 1] = nl; stack[s_ptr, 2] = bx+1; stack[s_ptr, 3] = by+1; s_ptr += 1
            else:
                if count < max_patches:
                    out[count, 0] = min_u
                    out[count, 1] = min_v
                    out[count, 2] = max_u
                    out[count, 3] = max_v
                    out[count, 4] = face
                    out[count, 5] = lod
                    out[count, 6] = x
                    out[count, 7] = y
                    out[count, 8] = radius
                    count += 1
                    
        return out[:count]

    @njit(fastmath=True)
    def pack_terrain_patches_jit(staging, st, raw_patches, uvs, ox_arr, oy_arr, slots, body_idx, h_uvs, h_ox, h_oy, h_slots, elev_min_km, elev_span_km):
        n = raw_patches.shape[0]
        for i in range(n):
            idx = st + i
            staging[idx, 0] = raw_patches[i, 0]
            staging[idx, 1] = raw_patches[i, 1]
            staging[idx, 2] = raw_patches[i, 2]
            staging[idx, 3] = raw_patches[i, 3]
            staging[idx, 4] = uvs[i]
            staging[idx, 5] = ox_arr[i]
            staging[idx, 6] = oy_arr[i]
            staging[idx, 7] = raw_patches[i, 8] * 0.05
            staging[idx, 8] = raw_patches[i, 4]
            staging[idx, 9] = raw_patches[i, 5]
            staging[idx, 10] = slots[i]
            staging[idx, 11] = body_idx
            staging[idx, 12] = h_slots[i]
            staging[idx, 13] = h_uvs[i]
            staging[idx, 14] = h_ox[i]
            staging[idx, 15] = h_oy[i]
            staging[idx, 16] = elev_min_km
            staging[idx, 17] = elev_span_km
            staging[idx, 18] = 0.0
            staging[idx, 19] = 0.0

    @njit(fastmath=True)
    def pack_cloud_patches_jit(staging, st, raw_patches, uvs, ox_arr, oy_arr, slots, body_idx):
        n = raw_patches.shape[0]
        for i in range(n):
            idx = st + i
            staging[idx, 0] = raw_patches[i, 0]
            staging[idx, 1] = raw_patches[i, 1]
            staging[idx, 2] = raw_patches[i, 2]
            staging[idx, 3] = raw_patches[i, 3]
            staging[idx, 4] = uvs[i]
            staging[idx, 5] = ox_arr[i]
            staging[idx, 6] = oy_arr[i]
            staging[idx, 7] = 0.0
            staging[idx, 8] = raw_patches[i, 4]
            staging[idx, 9] = raw_patches[i, 5]
            staging[idx, 10] = slots[i]
            staging[idx, 11] = body_idx
            staging[idx, 12] = -1.0
            staging[idx, 13] = 0.0
            staging[idx, 14] = 0.0
            staging[idx, 15] = 0.0
            staging[idx, 16] = 0.0
            staging[idx, 17] = 0.0
            staging[idx, 18] = 0.0
            staging[idx, 19] = 0.0

    # Warmup Numba JIT at import time to prevent first-frame runtime stutter
    try:
        _traverse_quadtree_jit(
            np.array([0.0, 0.0, 1000.0], dtype=np.float32),
            500.0, 0.0, 1080.0, 0.4142, 120.0, 1, 64,
            np.zeros((6, 4), dtype=np.float32), False, 0.0, 0.0, 0.0
        )
        _dummy_raw = np.zeros((1, 9), dtype=np.float32)
        _dummy_f = np.zeros(1, dtype=np.float32)
        _dummy_staging = np.zeros((1, 20), dtype=np.float32)
        pack_terrain_patches_jit(_dummy_staging, 0, _dummy_raw, _dummy_f, _dummy_f, _dummy_f, _dummy_f, 0.0, _dummy_f, _dummy_f, _dummy_f, _dummy_f, 0.0, 0.0)
        pack_cloud_patches_jit(_dummy_staging, 0, _dummy_raw, _dummy_f, _dummy_f, _dummy_f, _dummy_f, 0.0)
    except Exception:
        pass
else:
    def pack_terrain_patches_jit(staging, st, raw_patches, uvs, ox_arr, oy_arr, slots, body_idx, h_uvs, h_ox, h_oy, h_slots, elev_min_km, elev_span_km):
        n = raw_patches.shape[0]
        staging[st:st+n, 0:4] = raw_patches[:, 0:4]
        staging[st:st+n, 4] = uvs
        staging[st:st+n, 5] = ox_arr
        staging[st:st+n, 6] = oy_arr
        staging[st:st+n, 7] = raw_patches[:, 8] * 0.05
        staging[st:st+n, 8] = raw_patches[:, 4]
        staging[st:st+n, 9] = raw_patches[:, 5]
        staging[st:st+n, 10] = slots
        staging[st:st+n, 11] = body_idx
        staging[st:st+n, 12] = h_slots
        staging[st:st+n, 13] = h_uvs
        staging[st:st+n, 14] = h_ox
        staging[st:st+n, 15] = h_oy
        staging[st:st+n, 16] = elev_min_km
        staging[st:st+n, 17] = elev_span_km
        staging[st:st+n, 18] = 0.0
        staging[st:st+n, 19] = 0.0

    def pack_cloud_patches_jit(staging, st, raw_patches, uvs, ox_arr, oy_arr, slots, body_idx):
        n = raw_patches.shape[0]
        staging[st:st+n, 0:4] = raw_patches[:, 0:4]
        staging[st:st+n, 4] = uvs
        staging[st:st+n, 5] = ox_arr
        staging[st:st+n, 6] = oy_arr
        staging[st:st+n, 7] = 0.0
        staging[st:st+n, 8] = raw_patches[:, 4]
        staging[st:st+n, 9] = raw_patches[:, 5]
        staging[st:st+n, 10] = slots
        staging[st:st+n, 11] = body_idx
        staging[st:st+n, 12] = -1.0
        staging[st:st+n, 13:20] = 0.0


class PlanetQuadtree:
    """
    Manages the 6 root cube faces of a planet and performs LOD traversal.
    """
    def __init__(self, radius_km: float, max_lod: int = 4, split_factor: float = 1.25):
        self.radius_km = radius_km
        self.max_lod = max_lod
        self.split_factor = split_factor
        self.root_patches = [
            QuadtreePatch(face=f, lod=0, x=0, y=0, radius_km=radius_km)
            for f in range(6)
        ]

    def update_radius(self, radius_km: float):
        if abs(self.radius_km - radius_km) > 1e-4:
            self.radius_km = radius_km
            self.root_patches = [
                QuadtreePatch(face=f, lod=0, x=0, y=0, radius_km=radius_km)
                for f in range(6)
            ]

    def traverse_raw(self, cam_pos_local_km: np.ndarray, fov_deg: float, screen_height: float, max_lod: int = None, split_factor: float = None, obl: float = 0.0, max_patches: int = 4096, frustum_planes: np.ndarray = None, cloud_alt_km: float = 0.0, refract_bend: float = 0.0, height_max_km: float = 0.0) -> np.ndarray:
        """
        Fast JIT-accelerated traversal returning raw (N, 9) float32 array:
        [min_u, min_v, max_u, max_v, face, lod, x, y, radius].
        Zero Python object allocation overhead.
        """
        if max_lod is None:
            max_lod = self.max_lod
        if split_factor is None:
            split_factor = self.split_factor

        fov_rad = math.radians(max(0.001, min(160.0, fov_deg)))
        tan_half_fov = math.tan(fov_rad * 0.5)
        threshold_px = 75.0 * split_factor

        cam_f = np.ascontiguousarray(cam_pos_local_km, dtype=np.float32)

        has_frustum = False
        if frustum_planes is not None and getattr(frustum_planes, 'shape', None) == (6, 4):
            planes_f = np.ascontiguousarray(frustum_planes, dtype=np.float32)
            has_frustum = True
        else:
            planes_f = np.zeros((6, 4), dtype=np.float32)

        if _HAS_NUMBA:
            return _traverse_quadtree_jit(cam_f, float(self.radius_km), float(obl), float(screen_height), float(tan_half_fov), float(threshold_px), int(max_lod), int(max_patches), planes_f, has_frustum, float(cloud_alt_km), float(refract_bend), float(height_max_km))

        # Fallback pure-python traversal
        patches = self.traverse(cam_pos_local_km, fov_deg, screen_height, max_lod, split_factor, obl, frustum_planes=frustum_planes, cloud_alt_km=cloud_alt_km, refract_bend=refract_bend, height_max_km=height_max_km)
        out = np.empty((len(patches), 9), dtype=np.float32)
        for i, p in enumerate(patches):
            out[i, 0] = p.min_u
            out[i, 1] = p.min_v
            out[i, 2] = p.max_u
            out[i, 3] = p.max_v
            out[i, 4] = p.face
            out[i, 5] = p.lod
            out[i, 6] = p.x
            out[i, 7] = p.y
            out[i, 8] = p.radius
        return out

    def traverse(self, cam_pos_local_km: np.ndarray, fov_deg: float, screen_height: float, max_lod: int = None, split_factor: float = None, obl: float = 0.0, frustum_planes: np.ndarray = None, cloud_alt_km: float = 0.0, refract_bend: float = 0.0, height_max_km: float = 0.0):
        """
        Traverses the quadtree from root patches and returns a list of active leaf QuadtreePatch objects.
        cam_pos_local_km: 3D camera position relative to planet center in the planet's local rotated frame (km).
        """
        if _HAS_NUMBA:
            raw = self.traverse_raw(cam_pos_local_km, fov_deg, screen_height, max_lod, split_factor, obl, frustum_planes=frustum_planes, cloud_alt_km=cloud_alt_km, refract_bend=refract_bend, height_max_km=height_max_km)
            patches = []
            for i in range(len(raw)):
                p = QuadtreePatch.__new__(QuadtreePatch)
                p.min_u = float(raw[i, 0])
                p.min_v = float(raw[i, 1])
                p.max_u = float(raw[i, 2])
                p.max_v = float(raw[i, 3])
                p.face = int(raw[i, 4])
                p.lod = int(raw[i, 5])
                p.x = int(raw[i, 6])
                p.y = int(raw[i, 7])
                p.radius = float(raw[i, 8])
                mid_u = 0.5 * (p.min_u + p.max_u)
                mid_v = 0.5 * (p.min_v + p.max_v)
                p.center_dir = cube_to_sphere_point(p.face, mid_u, mid_v)
                p.center = np.array([p.center_dir[0] * self.radius_km, p.center_dir[1] * (1.0 - obl) * self.radius_km, p.center_dir[2] * self.radius_km], dtype=np.float32)
                patches.append(p)
            return patches

        if max_lod is None:
            max_lod = self.max_lod
        if split_factor is None:
            split_factor = self.split_factor

        fov_rad = math.radians(max(0.001, min(160.0, fov_deg)))
        tan_half_fov = math.tan(fov_rad * 0.5)

        threshold_px = 75.0 * split_factor

        has_frustum = False
        if frustum_planes is not None and getattr(frustum_planes, 'shape', None) == (6, 4):
            planes_f = np.ascontiguousarray(frustum_planes, dtype=np.float32)
            has_frustum = True
        else:
            planes_f = np.zeros((6, 4), dtype=np.float32)

        inv_radius = 1.0 / max(1e-6, self.radius_km)
        inv_obl = 1.0 / max(1e-4, 1.0 - obl)
        cs_pos = np.array([cam_pos_local_km[0], cam_pos_local_km[1] * inv_obl, cam_pos_local_km[2]], dtype=np.float32)
        cs_dist = float(np.linalg.norm(cs_pos))
        cs_dir = cs_pos / max(1e-6, cs_dist)
        eff_dist = max(self.radius_km + 1e-4, cs_dist)
        surf_horizon_angle = math.acos(min(1.0, self.radius_km / eff_dist))
        horizon_angle = surf_horizon_angle
        if cloud_alt_km > 0.0:
            cloud_r = self.radius_km + cloud_alt_km
            cloud_horizon_offset = math.acos(min(1.0, self.radius_km / max(1e-6, cloud_r)))
            horizon_angle += cloud_horizon_offset
        if refract_bend > 1e-6:
            refract_offset = min(0.05, refract_bend * 0.5)
            horizon_angle += refract_offset

        active_leaves = []

        def _evaluate_node(face, lod, x, y):
            scale = 2.0 / (1 << lod)
            min_u = -1.0 + x * scale
            max_u = min_u + scale
            min_v = -1.0 + y * scale
            max_v = min_v + scale

            mid_u = min_u + scale * 0.5
            mid_v = min_v + scale * 0.5

            c_dir = cube_to_sphere_point(face, mid_u, mid_v)
            c_center = np.array([c_dir[0] * self.radius_km, c_dir[1] * (1.0 - obl) * self.radius_km, c_dir[2] * self.radius_km], dtype=np.float32)

            p0 = cube_to_sphere_point(face, min_u, min_v)
            p1 = cube_to_sphere_point(face, max_u, min_v)
            p2 = cube_to_sphere_point(face, max_u, max_v)
            p3 = cube_to_sphere_point(face, min_u, max_v)
            corners = [p0, p1, p2, p3]
            dists = [np.linalg.norm(np.array([c[0]*self.radius_km, c[1]*(1.0-obl)*self.radius_km, c[2]*self.radius_km]) - c_center) for c in corners]
            radius = float(max(dists) * 1.05)

            eff_radius = radius + max(0.0, cloud_alt_km) + max(0.0, height_max_km)
            node_dot = float(np.dot(c_dir, cs_dir))
            angular_radius = min(1.0, eff_radius * inv_radius)
            cull_angle = horizon_angle + math.asin(angular_radius)
            if cull_angle < math.pi and node_dot < math.cos(cull_angle):
                return

            if has_frustum:
                dx_c = c_center[0] - cam_pos_local_km[0]
                dy_c = c_center[1] - cam_pos_local_km[1]
                dz_c = c_center[2] - cam_pos_local_km[2]
                if (dx_c*dx_c + dy_c*dy_c + dz_c*dz_c) > eff_radius * eff_radius:
                    in_f = True
                    for p_i in range(6):
                        d_p = c_center[0]*planes_f[p_i, 0] + c_center[1]*planes_f[p_i, 1] + c_center[2]*planes_f[p_i, 2] + planes_f[p_i, 3]
                        if d_p < -eff_radius:
                            in_f = False
                            break
                    if not in_f:
                        return

            dist_to_center = float(np.linalg.norm(c_center - cam_pos_local_km))
            dist_to_surface = max(0.01, dist_to_center - radius - max(0.0, height_max_km))

            screen_size = (radius * screen_height) / (dist_to_surface * 2.0 * tan_half_fov)

            if screen_size > threshold_px and lod < max_lod:
                nl = lod + 1
                bx = x * 2
                by = y * 2
                _evaluate_node(face, nl, bx, by)
                _evaluate_node(face, nl, bx+1, by)
                _evaluate_node(face, nl, bx, by+1)
                _evaluate_node(face, nl, bx+1, by+1)
            else:
                p = QuadtreePatch.__new__(QuadtreePatch)
                p.face = face
                p.lod = lod
                p.x = x
                p.y = y
                p.min_u = min_u
                p.max_u = max_u
                p.min_v = min_v
                p.max_v = max_v
                p.center_dir = c_dir
                p.center = c_center
                p.radius = radius
                active_leaves.append(p)

        face_order = sorted(range(6), key=lambda f: float(np.dot(cube_to_sphere_point(f, 0.0, 0.0), cs_dir)), reverse=True)
        for f in face_order:
            _evaluate_node(f, 0, 0, 0)

        return active_leaves

