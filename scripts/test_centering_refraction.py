"""Verification test for camera centering with atmospheric refraction and gravitational lensing.

Tests:
1. Ground observer on Earth looking at Venus near horizon:
   - Geometric vector vs refracted apparent vector
   - Screen-space projection: verify apparent target lands at (0.0, 0.0) in NDC
   - Compare with old unrefracted look direction (which had large NDC offset)
2. Observer in vacuum above atmosphere: verify no spurious deflection
3. Observer viewing the host planet itself: verify no self-refraction
4. Observer near a black hole: verify apparent lensing deflection
"""
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrr import matrix44
from engine.physics.refraction import (
    get_apparent_look_direction,
    solve_refraction_apparent,
    AU_TO_KM
)
from engine.core.input_handler import _camera_yaw_pitch_from, _camera_forward, _camera_get_up

def main():
    print("=== Testing Camera Centering with Atmospheric Refraction ===")

    # Earth parameters
    earth_r_km = 6371.0
    earth_r_au = earth_r_km / AU_TO_KM
    scale_height = 8.5
    max_bend = math.radians(68.0 / 60.0) # ~1.13 deg max space bend
    earth_pos = np.array([1.0, 0.0, 0.0], dtype=np.float64) # 1 AU on X axis
    earth_pole = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Observer on Earth surface (X axis)
    cam_world_pos = earth_pos + np.array([earth_r_au + 0.002 / AU_TO_KM, 0.0, 0.0], dtype=np.float64)

    # Target (Venus) 2 degrees above local horizon
    # Local up at (R, 0, 0) is +X.
    # Local horizon plane is Y-Z. Let's place Venus in X-Z plane, 2 deg elevation above horizontal -Z:
    e_rad = math.radians(2.0)
    # Unit direction from camera: sin(e) along +X (up), -cos(e) along Z (horizon)
    geom_dir = np.array([math.sin(e_rad), 0.0, -math.cos(e_rad)], dtype=np.float64)
    venus_dist_au = 0.5 # ~0.5 AU
    venus_pos = cam_world_pos + geom_dir * venus_dist_au

    refract_params = {
        'body_idx': 3, # Earth
        'is_cmp': False,
        'center_world': earth_pos,
        'radius_km': earth_r_km,
        'max_bend': max_bend,
        'scale_height_km': scale_height,
        'pole': earth_pole,
        'oblateness': 0.0,
    }

    # 1. Evaluate apparent look direction
    fwd_app = get_apparent_look_direction(
        cam_world_pos,
        venus_pos,
        refract_params=refract_params,
        target_body_idx=2, # Venus
        target_is_cmp=False
    )

    # Calculate angular lift between unrefracted and apparent
    dot_val = np.dot(geom_dir, fwd_app)
    cross_val = np.cross(geom_dir, fwd_app)
    lift_rad = math.atan2(np.linalg.norm(cross_val), dot_val)
    lift_arcmin = math.degrees(lift_rad) * 60.0

    print(f"Target geometric elevation: 2.0000 deg")
    print(f"Refraction lift angle:      {lift_arcmin:.4f} arcmin (~{lift_arcmin/60.0:.4f} deg)")
    assert 15.0 < lift_arcmin < 20.0, f"Expected lift ~17 arcmin at 2 deg elevation, got {lift_arcmin}"

    # 2. Verify View Matrix Projection
    # When camera looks along fwd_app:
    yaw, pitch = _camera_yaw_pitch_from(fwd_app)
    fwd_v = _camera_forward(yaw, pitch)
    cam = {"yaw": yaw, "pitch": pitch, "roll": 0.0, "up": [0.0, 1.0, 0.0]}
    cam_up = _camera_get_up(cam, fwd_v)

    view_mat = matrix44.create_look_at(cam_world_pos, cam_world_pos + fwd_v, cam_up, dtype='f8')
    proj_mat = matrix44.create_perspective_projection_matrix(60.0, 16.0 / 9.0, 1e-6, 100.0, dtype='f8')

    # What the shader computes for apparent position of Venus:
    # app_pos = cam_pos + V_app * dist
    app_venus_pos = cam_world_pos + fwd_app * venus_dist_au
    target_vec4 = np.array([app_venus_pos[0], app_venus_pos[1], app_venus_pos[2], 1.0], dtype='f8')
    clip_pos = target_vec4 @ view_mat @ proj_mat
    ndc_x = clip_pos[0] / clip_pos[3]
    ndc_y = clip_pos[1] / clip_pos[3]

    print(f"Apparent target NDC with refraction-corrected centering: ({ndc_x:.8f}, {ndc_y:.8f})")
    assert abs(ndc_x) < 1e-6 and abs(ndc_y) < 1e-6, f"Target must be dead center in NDC, got ({ndc_x}, {ndc_y})"

    # Now compare with what the OLD unrefracted centering would have given:
    old_yaw, old_pitch = _camera_yaw_pitch_from(geom_dir)
    old_fwd = _camera_forward(old_yaw, old_pitch)
    old_cam = {"yaw": old_yaw, "pitch": old_pitch, "roll": 0.0, "up": [0.0, 1.0, 0.0]}
    old_up = _camera_get_up(old_cam, old_fwd)
    old_view = matrix44.create_look_at(cam_world_pos, cam_world_pos + old_fwd, old_up, dtype='f8')
    old_clip = target_vec4 @ old_view @ proj_mat
    old_ndc_x = old_clip[0] / old_clip[3]
    old_ndc_y = old_clip[1] / old_clip[3]
    old_ndc_offset = math.sqrt(old_ndc_x**2 + old_ndc_y**2)
    # On a 1080p display, NDC corresponds to pixel offset:
    px_offset = (old_ndc_offset * 0.5) * 1080.0
    print(f"Old unrefracted centering NDC offset: {old_ndc_offset:.6f} (~{px_offset:.1f} pixels on 1080p)")
    assert old_ndc_offset > 0.004, "Old centering should have shown significant offset"

    # 3. Test Above Atmosphere (in space)
    space_cam_pos = earth_pos + np.array([earth_r_au + 500.0 / AU_TO_KM, 0.0, 0.0], dtype='f8') # 500 km altitude
    fwd_space = get_apparent_look_direction(
        space_cam_pos,
        space_cam_pos + geom_dir * venus_dist_au,
        refract_params=refract_params,
        target_body_idx=2,
        target_is_cmp=False
    )
    diff_space = np.linalg.norm(fwd_space - geom_dir)
    print(f"Space above atmosphere deflection diff: {diff_space:.10f}")
    assert diff_space < 1e-6, "Space observer looking upward should not have atmospheric bend"

    # 4. Test Target is the Refracting Host Planet itself
    fwd_host = get_apparent_look_direction(
        cam_world_pos,
        earth_pos,
        refract_params=refract_params,
        target_body_idx=3, # Earth
        target_is_cmp=False
    )
    geom_to_earth = (earth_pos - cam_world_pos) / np.linalg.norm(earth_pos - cam_world_pos)
    diff_host = np.linalg.norm(fwd_host - geom_to_earth)
    print(f"Host planet target deflection diff: {diff_host:.10f}")
    assert diff_host < 1e-6, "Host planet center must not refract relative to itself"

    # 5. Test Gravitational Lensing
    lens_params = {
        'center_world': np.array([0.0, 0.0, 0.0]),
        'rs_km': 10.0,
        'radius_km': 10.0,
        'lens_type': 3,
        'enabled': True,
        'strength': 1.0,
        'spin': 0.0,
        'pole': (0.0, 1.0, 0.0),
    }
    # Camera at (0, 0, 1000 km), target at (300 km, 0, -5000 km) behind black hole (b ~ 50 km > b_c = 26 km)
    bh_cam = np.array([0.0, 0.0, 1000.0 / AU_TO_KM])
    bh_target = np.array([300.0 / AU_TO_KM, 0.0, -5000.0 / AU_TO_KM])
    fwd_lensed = get_apparent_look_direction(
        bh_cam,
        bh_target,
        grav_lens_params=lens_params,
    )
    geom_bh = (bh_target - bh_cam) / np.linalg.norm(bh_target - bh_cam)
    lens_lift = math.degrees(math.atan2(np.linalg.norm(np.cross(geom_bh, fwd_lensed)), np.dot(geom_bh, fwd_lensed))) * 60.0
    print(f"Gravitational lensing deflection angle: {lens_lift:.4f} arcmin")
    assert lens_lift > 60.0, "Expected significant gravitational lensing deflection"

    print("\nALL CENTERING REFRACTION & LENSING TESTS PASSED!")

if __name__ == "__main__":
    main()
