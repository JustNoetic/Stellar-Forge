#!/usr/bin/env python3
"""
scripts/test_terrain_alignment.py

Verifies mathematical alignment between:
1. Quadtree traversal LOD hierarchy and world camera direction (no inverted LODs).
2. Terrain patch vertex world-space positions (terrain.vert).
3. Equirectangular UV coordinates sampled by terrain tiles vs legacy sphere.frag.
"""

import os
import sys
import math
import json
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rendering.terrain_quadtree import PlanetQuadtree, cube_to_sphere_point
from engine.rendering.planetshine import _build_rotation_props, compute_body_rotation_angles_jit


def test_body_alignment(body_name: str):
    print(f"--> Testing alignment for {body_name}...")
    bodies = json.load(open(os.path.join(ROOT_DIR, "data/system.json")))
    b_idx = next(i for i, b in enumerate(bodies) if b["name"].lower() == body_name.lower())
    body = bodies[b_idx]

    radius_au = float(body.get("r", 0.0035)) * 0.00465047  # Solar radius to AU
    radius_km = radius_au * 149597870.7

    # Build rotation basis
    rot_props = _build_rotation_props(bodies)
    rot_period, w0, locked, parent_idx, poles, tangents, bitangents = rot_props

    pos = np.zeros((len(bodies), 3), dtype='f8')
    for i, b in enumerate(bodies):
        sv = b.get('sv', {})
        pos[i] = [sv.get('x', 0.0), sv.get('y', 0.0), sv.get('z', 0.0)]

    angles = compute_body_rotation_angles_jit(123456.0, rot_period, w0, locked, parent_idx, pos, poles, tangents, bitangents)
    rot_angle = angles[b_idx]

    pole_n = poles[b_idx].astype(np.float32)
    tangent = tangents[b_idx].astype(np.float32)
    bitangent = bitangents[b_idx].astype(np.float32)

    # Test camera approaches from multiple directions: +X, -X, +Y, -Y, +Z, -Z, and diagonals
    test_dirs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0.577, 0.577, 0.577], dtype=np.float32),
    ]

    p_quadtree = PlanetQuadtree(radius_km, max_lod=4, split_factor=1.25)

    for cam_dir_w in test_dirs:
        cam_dir_w /= np.linalg.norm(cam_dir_w)
        # Position camera at radius + 200 km
        cam_pos_km = cam_dir_w * (radius_km + 200.0)

        # 1. Coordinate transform as in engine/app.py:
        p_local_x = float(np.dot(cam_pos_km, tangent))
        p_local_y = float(np.dot(cam_pos_km, pole_n))
        p_local_z = float(np.dot(cam_pos_km, bitangent))

        s_r = math.sin(-rot_angle)
        c_r = math.cos(-rot_angle)
        cam_pos_local = np.array([
            p_local_x * c_r - p_local_z * s_r,
            p_local_y,
            p_local_x * s_r + p_local_z * c_r
        ], dtype=np.float32)

        # 2. Quadtree traversal
        active_patches = p_quadtree.traverse(cam_pos_local, fov_deg=45.0, screen_height=1080.0, max_lod=4)
        assert len(active_patches) > 0

        # Find closest patch in world space using terrain.vert transform
        closest_patch = None
        min_world_dist = float('inf')

        for pt in active_patches:
            # Vertex at patch center (terrain.vert formula)
            mid_u = 0.5 * (pt.min_u + pt.max_u)
            mid_v = 0.5 * (pt.min_v + pt.max_v)
            pt_center_sphere = cube_to_sphere_point(pt.face, mid_u, mid_v)
            p_local_km = pt_center_sphere * radius_km

            s_rot = math.sin(rot_angle)
            c_rot = math.cos(rot_angle)
            p_local_body = np.array([
                p_local_km[0] * c_rot - p_local_km[2] * s_rot,
                p_local_km[1],
                p_local_km[0] * s_rot + p_local_km[2] * c_rot
            ])
            p_world_km = p_local_body[0] * tangent + p_local_body[1] * pole_n + p_local_body[2] * bitangent

            w_dist = float(np.linalg.norm(cam_pos_km - p_world_km))
            if w_dist < min_world_dist:
                min_world_dist = w_dist
                closest_patch = pt

        # The closest patch in world space MUST have the maximum LOD level!
        max_lod_in_tree = max(p.lod for p in active_patches)
        assert closest_patch.lod == max_lod_in_tree, (
            f"Front-facing patch LOD inversion! Closest patch has LOD {closest_patch.lod}, max is {max_lod_in_tree}"
        )

        # Verify distance to closest patch is close to camera altitude plus patch radius
        assert min_world_dist < (200.0 + closest_patch.radius * 1.5), (
            f"Closest patch center too far from camera! Expected < {200.0 + closest_patch.radius * 1.5:.1f} km, got {min_world_dist:.1f} km"
        )

        # 3. Check Texture UV Alignment with sphere.frag:
        # For the closest patch, sample its center point in world space
        mid_u = 0.5 * (closest_patch.min_u + closest_patch.max_u)
        mid_v = 0.5 * (closest_patch.min_v + closest_patch.max_v)
        p_sphere = cube_to_sphere_point(closest_patch.face, mid_u, mid_v)

        # Equirectangular UV from bake_planet_tiles:
        u_bake = (0.5 + math.atan2(p_sphere[2], p_sphere[0]) / (2.0 * math.pi)) % 1.0
        v_bake = 0.5 - math.asin(np.clip(p_sphere[1], -1.0, 1.0)) / math.pi

        # Equirectangular UV that sphere.frag computes for this world point:
        p_local_km = p_sphere * radius_km
        s_rot = math.sin(rot_angle)
        c_rot = math.cos(rot_angle)
        p_local_body = np.array([
            p_local_km[0] * c_rot - p_local_km[2] * s_rot,
            p_local_km[1],
            p_local_km[0] * s_rot + p_local_km[2] * c_rot
        ])
        p_world_km = p_local_body[0] * tangent + p_local_body[1] * pole_n + p_local_body[2] * bitangent

        # Feed p_world_km into sphere.frag shader logic:
        p_s = p_world_km / np.linalg.norm(p_world_km)
        p_loc = np.array([np.dot(p_s, tangent), np.dot(p_s, pole_n), np.dot(p_s, bitangent)])
        rot_sin = math.sin(-rot_angle)
        rot_cos = math.cos(-rot_angle)
        p_rot = np.array([
            p_loc[0] * rot_cos - p_loc[2] * rot_sin,
            p_loc[1],
            p_loc[0] * rot_sin + p_loc[2] * rot_cos
        ])
        u_sphere = (0.5 + math.atan2(p_rot[2], p_rot[0]) / (2.0 * math.pi)) % 1.0
        v_sphere = 0.5 - math.asin(np.clip(p_rot[1], -1.0, 1.0)) / math.pi

        u_diff = abs(u_bake - u_sphere)
        if u_diff > 0.5:  # Handle 0.0 vs 1.0 boundary
            u_diff = abs(u_diff - 1.0)
        v_diff = abs(v_bake - v_sphere)

        assert u_diff < 1e-4, f"UV U mismatch between bake and sphere.frag: {u_diff}"
        assert v_diff < 1e-4, f"UV V mismatch between bake and sphere.frag: {v_diff}"

    print(f"    Alignment test for {body_name} PASSED completely (UV diff < 1e-4, closest LOD = {max_lod_in_tree})!")


if __name__ == "__main__":
    test_body_alignment("Moon")
    test_body_alignment("Earth")
    print("=== All Alignment Tests PASSED! ===")
