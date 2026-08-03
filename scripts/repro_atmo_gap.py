"""Repro driver: land the camera on a body's surface, capture screenshots.

Usage: python scripts/repro_atmo_gap.py [BodyName] [--term <sun_dip_deg>]

Captures:
  1. Static surface view (looking slightly down at the horizon).
  2. Surface view while walking (W held) to expose stale-depth artifacts.

--term places the camera at the terminator so the sun is <sun_dip_deg>
degrees BELOW the horizon (reproduces the horizon shadow band).
"""
import os
import sys
import time
import threading
import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine"))

import numpy as np
import glfw
from engine.ephemeris.system_manager import SystemManager
from engine.physics.physics_core import load_system_from_data

_captured_window = {}
_orig_create_window = glfw.create_window


def _create_window_capture(*a, **k):
    w = _orig_create_window(*a, **k)
    _captured_window["w"] = w
    return w


glfw.create_window = _create_window_capture

LAND_ALT_AU = 1e-10


def main():
    args = sys.argv[1:]
    target_name = "Saturn"
    polar = False
    lat_deg = None
    i = 0
    while i < len(args):
        if args[i] == "--polar":
            polar = True
        elif args[i] == "--lat":
            lat_deg = float(args[i + 1])
            i += 1
        elif not args[i].startswith("--"):
            target_name = args[i]
        i += 1

    sys_mgr = SystemManager()
    bodies_data_raw = sys_mgr.load_default_system()
    bundle = load_system_from_data(bodies_data_raw)
    name_to_idx = bundle["name_to_idx"]
    if target_name not in name_to_idx:
        print(f"BODY NOT FOUND: {target_name}")
        return
    idx = name_to_idx[target_name]
    r_au = float(bundle["visual_data"][idx][3])
    atmo_names = [bundle["bodies_data"][a['body_idx']]['name'] for a in bundle.get('atmo_bodies', [])]
    print(f"Landing on {target_name} idx={idx} radius_au={r_au:.12g}")
    print(f"Bodies with atmospheres: {atmo_names}")

    from app import App
    app = App()

    sim = bundle["sim"]
    star_idx = bundle["star_idx"]
    p_sun = np.array([sim.particles[star_idx].x, sim.particles[star_idx].y, sim.particles[star_idx].z], dtype='f8')
    p_body = np.array([sim.particles[idx].x, sim.particles[idx].y, sim.particles[idx].z], dtype='f8')
    sun_dir = p_sun - p_body
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    print(f"sun_dir from {target_name}: {sun_dir}")

    cam = app.camera
    cam["tracking_idx"] = idx
    cam["tracking_is_cmp"] = False
    cam["tracking_mode"] = "body"
    cam["cam_look"] = "free"
    if polar:
        from app import _camera_yaw_pitch_from
        from engine.core.math_utils import pole_to_ecliptic
        bd = bundle["bodies_data"][idx]
        pe = pole_to_ecliptic(bd.get('pole_ra', 0.0), bd.get('pole_dec', 90.0))
        pole_engine = np.array([pe[0], pe[2], -pe[1]], dtype='f8')
        dirv = pole_engine / np.linalg.norm(pole_engine)
        cam["cam_pos_rel"] = dirv * (r_au * (1.0 - float(bd.get('oblateness', 0.0))) + LAND_ALT_AU)
        cam["cam_pos_rel_prev"] = cam["cam_pos_rel"].copy()
        cam["flight_speed"] = 2e-8
        north = np.array([0.0, 1.0, 0.0], dtype='f8')
        fwd_desired = np.cross(dirv, north)
        nf = np.linalg.norm(fwd_desired)
        if nf < 1e-6:
            fwd_desired = np.array([1.0, 0.0, 0.0], dtype='f8')
        else:
            fwd_desired = fwd_desired / nf
        yaw0, pitch0 = _camera_yaw_pitch_from(fwd_desired)
        cam["yaw"] = cam["yaw_actual"] = yaw0
        cam["pitch"] = cam["pitch_actual"] = pitch0 - 2.0
        print(f"[polar] camera placed at pole dir {dirv}, target alt {r_au*(1.0-float(bd.get('oblateness',0.0)))*149597870.7*1000:.0f} m")
        print(f"yaw={yaw0:.2f} pitch={cam['pitch']:.2f}")
    else:
        from app import _camera_yaw_pitch_from
        north = np.array([0.0, 1.0, 0.0], dtype='f8')
        if lat_deg is not None:
            from engine.core.math_utils import pole_to_ecliptic
            from app import _camera_rot_axis
            bd = bundle["bodies_data"][idx]
            pe = pole_to_ecliptic(bd.get('pole_ra', 0.0), bd.get('pole_dec', 90.0))
            pole_engine = np.array([pe[0], pe[2], -pe[1]], dtype='f8')
            pole_engine = pole_engine / np.linalg.norm(pole_engine)
            ref = np.array([0.0, 1.0, 0.0], dtype='f8')
            if abs(np.dot(pole_engine, ref)) > 0.999:
                ref = np.array([1.0, 0.0, 0.0], dtype='f8')
            tangent = np.cross(pole_engine, ref)
            tangent = tangent / np.linalg.norm(tangent)
            R = _camera_rot_axis(tangent, math.radians(90.0 - lat_deg))
            dirv = R @ pole_engine
            dirv = dirv / np.linalg.norm(dirv)
            cam["cam_pos_rel"] = dirv * (r_au + LAND_ALT_AU)
            cam["cam_pos_rel_prev"] = cam["cam_pos_rel"].copy()
            cam["flight_speed"] = 2e-8
            fwd_desired = np.cross(dirv, north)
            nf = np.linalg.norm(fwd_desired)
            if nf < 1e-6:
                fwd_desired = np.array([1.0, 0.0, 0.0], dtype='f8')
            else:
                fwd_desired = fwd_desired / nf
            yaw0, pitch0 = _camera_yaw_pitch_from(fwd_desired)
            cam["yaw"] = cam["yaw_actual"] = yaw0
            cam["pitch"] = cam["pitch_actual"] = pitch0 - 2.0
            print(f"[lat {lat_deg}] camera at sphere radius on lat {lat_deg} dir {dirv}")
            print(f"yaw={yaw0:.2f} pitch={cam['pitch']:.2f}")
        else:
            dirv = sun_dir
            cam["cam_pos_rel"] = dirv * (r_au + LAND_ALT_AU)
            cam["cam_pos_rel_prev"] = cam["cam_pos_rel"].copy()
            cam["flight_speed"] = 2e-8
            fwd_desired = np.cross(sun_dir, north)
            nf = np.linalg.norm(fwd_desired)
            if nf < 1e-6:
                fwd_desired = np.array([1.0, 0.0, 0.0], dtype='f8')
            else:
                fwd_desired = fwd_desired / nf
            yaw0, pitch0 = _camera_yaw_pitch_from(fwd_desired)
            cam["yaw"] = cam["yaw_actual"] = yaw0
            cam["pitch"] = cam["pitch_actual"] = pitch0 - 2.0
            print(f"yaw={yaw0:.2f} pitch={cam['pitch']:.2f}")

    shot_num = [0]
    shot_done = threading.Event()

    def _wait_frames(n):
        deadline = time.time() + 120
        while time.time() < deadline and getattr(app, "frame_counter", 0) < n:
            time.sleep(0.1)

    def _request_shot(tag):
        while app._screenshot_capturing or app._screenshot_saving or app._screenshot_request is not None:
            time.sleep(0.1)
        app._screenshot_request = (1280, 720)
        deadline = time.time() + 60
        while time.time() < deadline and app._screenshot_request is not None:
            time.sleep(0.1)
        while time.time() < deadline and (app._screenshot_capturing or app._screenshot_saving):
            time.sleep(0.1)
        shot_num[0] += 1
        print(f"[shot {shot_num[0]}] {tag} saved")

    def _orchestrator():
        _wait_frames(30)
        time.sleep(2.0)
        r_cam = np.linalg.norm(cam["cam_pos_rel"]) * 149597870.7 * 1000.0
        print(f"[cam] |rel| = {r_cam:.1f} m")
        _request_shot("static surface view")
        time.sleep(2.0)
        cam["keys"]["w"] = True
        time.sleep(6.0)
        _request_shot("walking (W held)")
        cam["keys"]["w"] = False
        shot_done.set()
        w = _captured_window.get("w")
        if w is not None:
            try:
                glfw.set_window_should_close(w, True)
            except Exception:
                pass

    threading.Thread(target=_orchestrator, daemon=True).start()

    try:
        app.run()
    finally:
        print("APP EXITED")


if __name__ == "__main__":
    main()
