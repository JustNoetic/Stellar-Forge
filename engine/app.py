import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import OpenGL
OpenGL.ERROR_CHECKING = False
import glfw
import moderngl
import numpy as np
import ctypes

import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _camera_align_up(up_world):
    """Return the 3x3 rotation R that maps the local +Y axis onto unit `up_world`.

    Used to build the surface-relative view frame when the camera is landed:
    fwd_world = R @ fwd_local, where fwd_local = (cos(yaw)cos(pitch), sin(pitch), sin(yaw)cos(pitch)).
    """
    up_world = np.asarray(up_world, dtype='f8')
    n = np.linalg.norm(up_world)
    if n < 1e-12:
        return np.eye(3, dtype='f8')
    up_world = up_world / n
    from_v = np.array([0.0, 1.0, 0.0], dtype='f8')
    d = max(-1.0, min(1.0, float(np.dot(from_v, up_world))))
    if d > 0.999999:
        return np.eye(3, dtype='f8')
    if d < -0.999999:
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype='f8')
    axis = np.cross(from_v, up_world)
    axis = axis / np.linalg.norm(axis)
    theta = math.acos(d)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]], dtype='f8')
    return np.eye(3, dtype='f8') + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

def _camera_forward(yaw_deg, pitch_deg):
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    return np.array([
        math.cos(yaw_rad) * math.cos(pitch_rad),
        math.sin(pitch_rad),
        math.sin(yaw_rad) * math.cos(pitch_rad)], dtype='f8')

def _camera_yaw_pitch_from(fwd):
    fwd = np.asarray(fwd, dtype='f8')
    n = np.linalg.norm(fwd)
    if n < 1e-300:
        return 0.0, 0.0
    fwd = fwd / n
    yaw = math.degrees(math.atan2(fwd[2], fwd[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[1]))))
    return yaw, pitch

def _ellipsoid_surface_radius(r_eq, f, pole, u):
    """Surface radius of an oblate spheroid (semi-major r_eq, flattening f)
    in the unit direction u (pole = unit spin axis)."""
    r_eq = float(r_eq)
    f = float(f)
    if f <= 0.0:
        return r_eq
    pole = np.asarray(pole, dtype='f8')
    if np.linalg.norm(pole) < 1e-12:
        return r_eq
    pole = pole / np.linalg.norm(pole)
    u = np.asarray(u, dtype='f8')
    n_u = np.linalg.norm(u)
    if n_u < 1e-12:
        return r_eq * (1.0 - f)
    u = u / n_u
    b = r_eq * (1.0 - f)
    cos_t = float(np.dot(u, pole))
    sin_t2 = max(0.0, 1.0 - cos_t * cos_t)
    denom = b * b * sin_t2 + r_eq * r_eq * cos_t * cos_t
    if denom < 1e-30:
        return b
    return math.sqrt((r_eq * r_eq * b * b) / denom)

_ICOSPHERE_SAG_CACHE = {}

def _icosphere_max_sag(subdivisions):
    """Max inward sag of the tessellated icosphere from its circumsphere,
    as a fraction of the radius (1 - inradius/circumradius)."""
    if subdivisions not in _ICOSPHERE_SAG_CACHE:
        mesh_verts, mesh_idx = create_icosphere_mesh(subdivisions=subdivisions)
        v3 = mesh_verts.reshape(-1, 6)[:, 0:3]
        i3 = mesh_idx.reshape(-1, 3)
        a = v3[i3[:, 0]]
        b_v = v3[i3[:, 1]]
        c_v = v3[i3[:, 2]]
        nrm = np.cross(b_v - a, c_v - a)
        denom = np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)
        d = np.abs(np.einsum('ij,ij->i', a, nrm)) / denom
        _ICOSPHERE_SAG_CACHE[subdivisions] = float(1.0 - d.min())
    return _ICOSPHERE_SAG_CACHE[subdivisions]

def _camera_rot_axis(axis, angle):
    """Rodrigues 3x3 rotation matrix about unit `axis` by `angle` radians."""
    axis = np.asarray(axis, dtype='f8')
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3, dtype='f8')
    axis = axis / n
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]], dtype='f8')
    c = math.cos(angle)
    s = math.sin(angle)
    return np.eye(3, dtype='f8') + s * K + (1.0 - c) * (K @ K)

def _camera_get_up(cam, fwd):
    """Retrieve and orthogonalize the continuous camera UP vector."""
    fwd = np.asarray(fwd, dtype='f8')
    n_fwd = np.linalg.norm(fwd)
    if n_fwd < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype='f8')
    fwd = fwd / n_fwd
    
    raw_up = np.array(cam.get("up", [0.0, 1.0, 0.0]), dtype='f8')
    if np.linalg.norm(raw_up) < 1e-6:
        raw_up = np.array([0.0, 1.0, 0.0], dtype='f8')
    
    up_proj = raw_up - float(np.dot(raw_up, fwd)) * fwd
    n_up = np.linalg.norm(up_proj)
    if n_up > 1e-6:
        return up_proj / n_up
    
    alt = np.array([1.0, 0.0, 0.0], dtype='f8')
    alt_proj = alt - float(np.dot(alt, fwd)) * fwd
    n_alt = np.linalg.norm(alt_proj)
    if n_alt > 1e-6:
        return alt_proj / n_alt
    alt2 = np.array([0.0, 0.0, 1.0], dtype='f8')
    alt_proj2 = alt2 - float(np.dot(alt2, fwd)) * fwd
    return alt_proj2 / np.linalg.norm(alt_proj2)

def _camera_screen_up(fwd, roll_rad, cam=None):
    """Screen-space up of the current view."""
    if cam is not None and "up" in cam:
        up = _camera_get_up(cam, fwd)
    else:
        fwd = np.asarray(fwd, dtype='f8')
        n = np.linalg.norm(fwd)
        if n < 1e-12:
            return np.array([0.0, 1.0, 0.0], dtype='f8')
        fwd = fwd / n
        up_proj = np.array([0.0, 1.0, 0.0], dtype='f8') - float(np.dot(fwd, np.array([0.0, 1.0, 0.0], dtype='f8'))) * fwd
        nup = np.linalg.norm(up_proj)
        if nup > 1e-9:
            up = up_proj / nup
        else:
            up = np.array([1.0, 0.0, 0.0], dtype='f8')
            up = up - float(np.dot(fwd, up)) * fwd
            nup = np.linalg.norm(up)
            up = up / nup if nup > 1e-9 else np.array([0.0, 0.0, 1.0], dtype='f8')
    if abs(roll_rad) > 1e-9:
        up = _camera_rot_axis(fwd, -roll_rad) @ up
    return up

def _camera_orient_from_view(fwd, roll_rad, cam=None):
    """Orientation matrix (columns: right, up, back) matching rendered view."""
    up = _camera_screen_up(fwd, roll_rad, cam)
    right = np.cross(fwd, up)
    nr = np.linalg.norm(right)
    if nr > 1e-9:
        right = right / nr
    else:
        right = np.array([1.0, 0.0, 0.0], dtype='f8')
    return np.column_stack([right, up, -fwd])

def _camera_euler_from_orient(R):
    """Extract (yaw, pitch, roll) reproducing the orientation matrix exactly."""
    fwd = -R[:, 2]
    yaw = math.degrees(math.atan2(fwd[2], fwd[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[1]))))
    up = R[:, 1]
    up_ref = _camera_screen_up(fwd, 0.0)
    right_ref = np.cross(fwd, up_ref)
    roll = math.degrees(math.atan2(-float(np.dot(up, right_ref)), float(np.dot(up, up_ref))))
    return yaw, pitch, roll

def _oblate_surface_normal(rel, f=0.0, pole=None):
    """Compute the true outward surface normal vector of an oblate spheroid.
    `rel` is relative position from body center, `f` is oblateness, `pole` is unit spin axis."""
    rel = np.asarray(rel, dtype='f8')
    r_norm = np.linalg.norm(rel)
    if r_norm < 1e-300:
        return np.array([0.0, 1.0, 0.0], dtype='f8')
    
    radial = rel / r_norm
    if f <= 0.0 or pole is None:
        return radial
    
    pole = np.asarray(pole, dtype='f8')
    p_norm = np.linalg.norm(pole)
    if p_norm < 1e-12:
        return radial
    pole = pole / p_norm
    
    f_clamped = min(0.9, max(0.0, float(f)))
    scale_para = 1.0 / ((1.0 - f_clamped) ** 2)
    
    para_len = float(np.dot(rel, pole))
    N = rel + (scale_para - 1.0) * para_len * pole
    n_len = np.linalg.norm(N)
    if n_len > 1e-12:
        return N / n_len
    return radial

def _camera_get_ref_up(cam, rel):
    """Determine the reference vertical axis for horizon-aligned FPS rotation, taking oblate planet geometry into account."""
    f = float(cam.get("track_f", 0.0))
    pole = cam.get("track_pole", None)
    return _oblate_surface_normal(rel, f, pole)

def _camera_pivot_apply(cam, dx, dy, sensitivity):
    """LMB trackball/FPS pivot: rotate view direction and up vector in place (free look)."""
    if cam.get("cam_look", "aim") == "aim":
        rel = np.asarray(cam["cam_pos_rel"], dtype='f8')
        r = np.linalg.norm(rel)
        fwd = -rel / r if r > 1e-300 else _camera_forward(cam.get("yaw_actual", cam["yaw"]), cam.get("pitch_actual", cam["pitch"]))
    else:
        rel = np.asarray(cam.get("cam_pos_rel", [0.0, 0.0, 1.0]), dtype='f8')
        fwd = _camera_forward(cam.get("yaw_actual", cam["yaw"]), cam.get("pitch_actual", cam["pitch"]))
    
    up = _camera_get_up(cam, fwd)
    right = np.cross(fwd, up)
    nr = np.linalg.norm(right)
    if nr > 1e-9:
        right = right / nr
    else:
        right = np.array([1.0, 0.0, 0.0], dtype='f8')
    up = np.cross(right, fwd)
    up = up / np.linalg.norm(up)

    if cam.get("horizon_align", True) and cam.get("is_near_surface", False):
        h_axis = _camera_get_ref_up(cam, rel)
    else:
        h_axis = up

    R_vert = _camera_rot_axis(right, math.radians(-dy * sensitivity))
    fwd_test = R_vert @ fwd
    fwd_test = fwd_test / max(np.linalg.norm(fwd_test), 1e-300)
    
    if cam.get("horizon_align", True) and cam.get("is_near_surface", False):
        dot = np.dot(fwd_test, h_axis)
        if abs(dot) > 0.999:
            old_dot = np.dot(fwd, h_axis)
            if abs(dot) > abs(old_dot):
                R_vert = np.eye(3, dtype='f8')
                
    R = _camera_rot_axis(h_axis, math.radians(-dx * sensitivity)) @ R_vert
    fwd_new = R @ fwd
    fwd_new = fwd_new / max(np.linalg.norm(fwd_new), 1e-300)
    up_new = R @ up
    up_new = up_new / max(np.linalg.norm(up_new), 1e-300)

    yaw_new, pitch_new = _camera_yaw_pitch_from(fwd_new)
    cam["yaw"] = cam["yaw_actual"] = yaw_new
    cam["pitch"] = cam["pitch_actual"] = pitch_new
    cam["up"] = up_new.tolist()

def _camera_orbit_apply(cam, dx, dy, sensitivity):
    """RMB screen-space orbit: rotate camera position about pivot using screen-space axes.
    Guarantees horizontal mouse drag always orbits left/right on screen (including at poles),
    and vertical drag orbits up/down on screen, with continuous cam['up'] to prevent 180° pole flips.
    Preserves free look direction if cam['cam_look'] == 'free' without snapping to 'aim'."""
    rel = np.asarray(cam["cam_pos_rel"], dtype='f8')
    r = np.linalg.norm(rel)
    if r < 1e-300:
        return

    is_free = (cam.get("cam_look", "aim") == "free")
    if is_free:
        fwd = _camera_forward(cam.get("yaw_actual", cam["yaw"]), cam.get("pitch_actual", cam["pitch"]))
    else:
        fwd = -rel / r

    up = _camera_get_up(cam, fwd)
    right = np.cross(fwd, up)
    nr = np.linalg.norm(right)
    if nr > 1e-9:
        right = right / nr
    else:
        right = np.array([1.0, 0.0, 0.0], dtype='f8')
    up = np.cross(right, fwd)
    up = up / np.linalg.norm(up)

    R_h = _camera_rot_axis(up, math.radians(dx * sensitivity))
    R_v = _camera_rot_axis(right, math.radians(dy * sensitivity))
    R = R_h @ R_v

    rel_new = R @ rel
    r_new = np.linalg.norm(rel_new)
    if r_new > 1e-300:
        rel_new = rel_new / r_new * r
    up_new = R @ up
    up_new = up_new / np.linalg.norm(up_new)

    if is_free:
        fwd_new = R @ fwd
        fwd_new = fwd_new / np.linalg.norm(fwd_new)
        yaw_new, pitch_new = _camera_yaw_pitch_from(fwd_new)
        cam["yaw"] = cam["yaw_actual"] = yaw_new
        cam["pitch"] = cam["pitch_actual"] = pitch_new

    cam["cam_pos_rel"] = rel_new.tolist()
    cam["up"] = up_new.tolist()

import threading
import time
from pyrr import matrix44
import imgui
from imgui.integrations.glfw import GlfwRenderer
import warnings
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit
import datetime
import os
from engine.ephemeris.system_manager import SystemManager, SystemSnapshot, derive_star_properties, temperature_to_rgb, rgb_to_hex
from engine.ephemeris.spice_manager import SpiceManager

from engine.core.constants import *
from engine.core.math_utils import *
from engine.physics.physics_core import *
from engine.physics.physics_core import _extract_render_state, _update_hierarchy_core
from engine.rendering.render_utils import *
from engine.rendering.shaders import *
from engine.rendering.post_shaders import *
from engine.physics.atmosphere_physics import compute_atmosphere_properties, GAS_PROPERTIES, compute_mie_coefficients, compute_mie_absorption
from engine.ui import render_ui

_PERF_ENABLED = os.environ.get("STELLAR_FORGE_PERF") == "1"
_PERF_TRACKER = None

class _NullCtx:
    __slots__ = ()
    def __enter__(self): return self
    def __exit__(self, *a): return False

if _PERF_ENABLED:
    class _PerfCtx:
        __slots__ = ('_name',)
        def __init__(self, name):
            self._name = name
            _PERF_TRACKER.begin(name)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            _PERF_TRACKER.end()
            return False

    def _perf(name):
        if _PERF_TRACKER is None:
            return _NullCtx()
        return _PerfCtx(name)
else:
    def _perf(name):
        return _NullCtx()

def _PERF_INSTALL_TRACKER(tracker):
    """Public hook so perf_test.py can attach a tracker after import."""
    global _PERF_TRACKER
    _PERF_TRACKER = tracker

_GPU_PERF_ENABLED = os.environ.get("STELLAR_FORGE_GPU_PERF") == "1"

if _GPU_PERF_ENABLED:
    class _GpuPerfQuery:
        """GL_TIME_ELAPSED query for one render pass, recorded into the PerfTracker.

        The query is ENDED right after its pass (timer queries must never
        overlap — a second BeginQuery while one is active is GL_INVALID_OPERATION),
        but the .elapsed READ is deferred to the next frame boundary so the
        read never stalls the pipeline mid-frame.
        """
        __slots__ = ('_name', '_tracker', '_query', '_done')

        def __init__(self, ctx, name):
            self._name = name
            self._tracker = _PERF_TRACKER
            q = ctx.query(time=True)
            q.__enter__()
            self._query = q
            self._done = False

        def end(self):
            if not self._done:
                self._done = True
                self._query.__exit__(None, None, None)

        def finish(self):
            dt = float(self._query.elapsed) * 1e-9
            if self._tracker is not None:
                self._tracker.record_gpu(self._name, dt)

    _GPU_PENDING = []

    def _perf_gpu_begin(ctx, name):
        q = _GpuPerfQuery(ctx, name)
        _GPU_PENDING.append(q)
        return q

    def _perf_gpu_end(q):
        if q is not None:
            q.end()

    def _perf_gpu_flush():
        global _GPU_PENDING
        pending = _GPU_PENDING
        _GPU_PENDING = []
        for q in pending:
            q.finish()
else:
    def _perf_gpu_begin(ctx, name):
        return None

    def _perf_gpu_end(q):
        pass

    def _perf_gpu_flush():
        pass

import OpenGL.GL as gl
import ctypes

from engine.rendering.imgui_renderer import ModernGLImGuiRenderer, ModernGLGlfwRenderer
from engine.rendering.planetshine import (
    compute_max_bend,
    compute_ring_coplanar_masks,
    get_cached_atmosphere_properties,
    _build_tex_idx_arr,
    _build_rotation_props,
    compute_body_rotation_angles_jit,
    compute_planetshine_numba
)
from engine.rendering.texture_baker import apply_hsba_np, bake_and_export_ring_textures
from engine.core.input_handler import InputHandlerMixin

_ROT_FOLLOW_ALT_AU = 200.0 / AU_TO_KM  # camera rides the body's rotation below 200 km altitude

class App(InputHandlerMixin):
    def __init__(self):
        self.window_width, self.window_height = 1280, 720
        self.fb_width, self.fb_height = 1280, 720
        self.impl = None
        self.pick_request = None
        self.camera = {
            "target": np.array([0.0, 0.0, 0.0], dtype='f8'),
            # Camera offset (AU, f8) from the tracked pivot (body / barycenter / target).
            # Default eye pose reproduces the legacy 45 AU orbit view (yaw -90, pitch 25).
            "cam_pos_rel": np.array([0.0, -19.02, 40.78], dtype='f8'),
            "cam_vel": np.zeros(3, dtype='f8'),
            "flight_speed": 0.1,           # AU/s, adjusted by scroll wheel
            "cam_mode": "flight",          # kept for save-file compat (no landed state anymore)
            "cam_look": "aim",             # "aim" (at pivot) | "free" (yaw/pitch)
            "keys": {},                    # held WASD states
            "approach_delta": 0.0,         # LMB+RMB vertical drag (exp scale factor)
            "exposure": 1.0,
            "hdr_enabled": True,
            "fov": 45.0,
            "yaw": -90.0,         
            "yaw_actual": -90.0,
            "pitch": 25.0,        
            "pitch_actual": 25.0,
            "roll": 0.0,
            "roll_actual": 0.0,
            "horizon_align": True,           # below 200 km, smoothly roll-level the view
            "left_dragging": False,
            "right_dragging": False,
            "last_x": 0.0,
            "last_y": 0.0,
            "tracking_idx": 0,
            "tracking_is_cmp": False,
            "tracking_mode": "body",
            "inspected_idx": None,
            "inspected_is_cmp": False,
            "inspect_bary": False,
            "edit_mode": False,
            "edit_data": {},
            "ly_threshold_au": DEFAULT_LY_THRESHOLD_AU,
            "show_settings_modal": False,
            "atmo_quality": 2,
            "atmo_resolution": 1.0,
            "atmo_vrs_threshold_px": 100.0,
            "atmo_stochastic": True,
            "atmo_steps_max": 32,
            "atmo_adaptive_steps": True,
            "atmo_adaptive_steps_max": 128,
            "atmo_enabled": True,
            "refraction_enabled": True,
            "show_orbits": True,
            "orbit_fade_dir_idx": 0,
            "orbit_min_alpha": 0.3,
            "show_habitable_zone": False,
            "planetshine_enabled": True,
            "ringshine_enabled": True,
            "ringshine_band_count": 10,
            "bloom_intensity": 0.05,
            "bloom_threshold": 1.0,
            "msaa_samples": 4,
            "inspector_frame": 0,
            "shadow_caster_budget": 32,
            "tex_stream_threshold_px": 500.0,
            "screenshot_res_idx": 1,
            "anisotropy": 16.0,
            "photo_accum_enabled": False,
            "screenshot_accum_enabled": False,
            "photo_accum_target": 0,
            "screenshot_accum_target": 16,
            "movement_mode": 0,
        }
        self.time_ctrl = {
            "multiplier": 1.0,
            "paused": False,
        }

        self._screenshot_request = None
        self._screenshot_capturing = False
        self._screenshot_saving = False

        # Post-Processing FBOs
        self.hdr_resolve_fbo = None
        self.hdr_resolve_tex = None
        self.bloom_fbos = []
        self.bloom_texs = []
        self.last_fb_size = (0, 0)
        self.last_msaa_samples = -1
        self.last_atmo_res = -1.0
        
        # Orbit-line MSAA resources (only orbit lines are multisampled)
        self.orbit_msaa_fbo = None
        self.orbit_resolved_tex = None
        self.orbit_resolve_fbo = None
        
        # Atmosphere rendering resources
        self.prog_atmo_composite = None
        self.quad_vao_atmo_comp = None
        self.prog_atmo_lowres = None
        self.prog_atmo_upsample = None
        self.quad_vao_atmo_upsample = None
        self.atmo_lowres_fbo = None
        self.atmo_lowres_scatter_tex = None
        self.atmo_lowres_trans_tex = None
        
        self.depth_texture = None
        self.prev_cam_origin = None
        
        # Screenshot capture state
        self._screenshot_request = None      # (width, height) tuple when capture requested
        self._screenshot_capturing = False   # True during the high-res render frame
        self._screenshot_orig_fb = None      # (orig_w, orig_h) saved during capture
        self._screenshot_toast = None        # (message, timestamp) for status notification
        self._screenshot_saving = False      # True while background save thread is running
        
        self.quad_vao_down = None
        self.quad_vao_up = None
        self.quad_vao_comp = None
        
        self.prog_bloom_down = None
        self.prog_bloom_up = None
        self.prog_composite = None
        self.prog_accum = None
        self.quad_vao_accum = None
        self.time_ctrl = {
            "paused": True,
            "multiplier": 1.0,
            "time_direction": 1,
            "render_timeline": False,
            "cancel_render": False,
            "target_t": 0.0,
            "sync_t": None,
            "timeline_playing": False,
            "timeline_speed": 10.0,
            "timeline_scrub_float": 0.0,
        }
        self.shared_state = None
        self.comparison_enabled = False
        self.comparison_system_name = "Solar System"
        self.comparison_offset_au = 10.0
        self.n_orbits_cmp = 0
        self.n_orbits_hi_cmp = 0
        self.n_orbits_med_cmp = 0
        self.n_orbits_low_cmp = 0
        self.ring_render_groups_cmp = []
        self.ring_precomputed_cmp = []
        self.ui_visible = True
        self.show_outliner = True
        self.show_inspector = True
        self.show_time_hud = True
        self.active_system_name = SystemManager.SOLAR_SYSTEM_NAME
        self.switch_req_name = self.active_system_name
        self._ui_search_query = ""
        self._show_ephem_setup_modal = False
        self._show_ephem_download_modal = False
        self.scrub_index = [0]
        self.jump_date = [2026, 1, 1, 12, 0, 0]
        self.load_settings()

    def load_settings(self):
        try:
            import json
            import os
            settings_path = os.path.join("data", "graphics_settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    self.camera[k] = v
        except Exception as e:
            print(f"Failed to load settings: {e}")

    def save_settings(self):
        try:
            import json
            import os
            os.makedirs("data", exist_ok=True)
            settings_path = os.path.join("data", "graphics_settings.json")
            saved = {
                "atmo_quality": self.camera.get("atmo_quality", 2),
                "atmo_resolution": self.camera.get("atmo_resolution", 1.0),
                "atmo_vrs_threshold_px": self.camera.get("atmo_vrs_threshold_px", 100.0),
                "atmo_stochastic": self.camera.get("atmo_stochastic", True),
                "atmo_steps_max": self.camera.get("atmo_steps_max", 32),
                "atmo_adaptive_steps": self.camera.get("atmo_adaptive_steps", True),
                "atmo_adaptive_steps_max": self.camera.get("atmo_adaptive_steps_max", 128),
                "hdr_enabled": self.camera.get("hdr_enabled", True),
                "exposure": self.camera.get("exposure", 1.0),
                "bloom_intensity": self.camera.get("bloom_intensity", 0.05),
                "bloom_threshold": self.camera.get("bloom_threshold", 1.0),
                "msaa_samples": self.camera.get("msaa_samples", 4),
                "show_orbits": self.camera.get("show_orbits", True),
                "orbit_fade_dir_idx": self.camera.get("orbit_fade_dir_idx", 0),
                "orbit_min_alpha": self.camera.get("orbit_min_alpha", 0.3),
                "show_habitable_zone": self.camera.get("show_habitable_zone", False),
                "planetshine_enabled": self.camera.get("planetshine_enabled", True),
                "ringshine_enabled": self.camera.get("ringshine_enabled", True),
                "ringshine_band_count": self.camera.get("ringshine_band_count", 10),
                "inspector_frame": self.camera.get("inspector_frame", 0),
                "fov": self.camera.get("fov", 45.0),
                "flight_speed": self.camera.get("flight_speed", 0.1),
                "shadow_caster_budget": self.camera.get("shadow_caster_budget", 32),
                "tex_stream_threshold_px": self.camera.get("tex_stream_threshold_px", 500.0),
                "screenshot_res_idx": self.camera.get("screenshot_res_idx", 1),
                "horizon_align": self.camera.get("horizon_align", True),
                "photo_accum_enabled": self.camera.get("photo_accum_enabled", False),
                "screenshot_accum_enabled": self.camera.get("screenshot_accum_enabled", False),
                "photo_accum_target": self.camera.get("photo_accum_target", 0),
                "screenshot_accum_target": self.camera.get("screenshot_accum_target", 16),
                "movement_mode": self.camera.get("movement_mode", 0),
            }
            with open(settings_path, 'w') as f:
                json.dump(saved, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def scroll_callback(self, window, xoffset, yoffset):
        if self.impl: self.impl.scroll_callback(window, xoffset, yoffset)
        if imgui.get_io().want_capture_mouse: return 
        
        is_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        
        if is_shift:
            if yoffset > 0:
                self.camera["fov"] *= (0.85 ** yoffset)
            elif yoffset < 0:
                self.camera["fov"] *= (1.0 / (0.85 ** abs(yoffset)))
            self.camera["fov"] = max(0.001, min(120.0, self.camera["fov"]))
        else:
            movement_mode = self.camera.get("movement_mode", 0)
            if movement_mode == 1:
                # Simple Orbit mode: scroll wheel zooms camera in / out relative to tracked object
                self.camera["approach_delta"] += yoffset * 0.15
            else:
                # Free Flight mode: scroll wheel changes flight velocity (Space Engine style, ~1.15x per notch).
                speed = self.camera.get("flight_speed", 0.1)
                speed *= (1.15 ** yoffset)
                self.camera["flight_speed"] = max(1e-12, min(5.0, speed))
    
    def mouse_button_callback(self, window, button, action, mods):
        if self.impl: self.impl.mouse_callback(window, button, action, mods)
        if imgui.get_io().want_capture_mouse: return 
        
        x, y = glfw.get_cursor_pos(window)
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.camera["left_dragging"] = True
                self.camera["last_x"], self.camera["last_y"] = x, y
                self.camera["click_start_x"], self.camera["click_start_y"] = x, y
            elif action == glfw.RELEASE: 
                self.camera["left_dragging"] = False
                # Don't pick when ending an LMB+RMB approach gesture (RMB still held).
                if not self.camera.get("right_dragging", False):
                    dx = x - self.camera.get("click_start_x", x)
                    dy = y - self.camera.get("click_start_y", y)
                    if dx*dx + dy*dy < 25:
                        self.pick_request = (x, y)
                
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.camera["right_dragging"] = True
                self.camera["last_x"], self.camera["last_y"] = x, y
            elif action == glfw.RELEASE: self.camera["right_dragging"] = False
    
    def cursor_pos_callback(self, window, xpos, ypos):
        dx = xpos - self.camera["last_x"]
        dy = ypos - self.camera["last_y"]
        
        fov_ratio = max(0.0001, min(1.0, self.camera.get("fov", 45.0) / 45.0))
        movement_mode = self.camera.get("movement_mode", 0)
        
        if self.camera["left_dragging"] and self.camera["right_dragging"]:
            # LMB+RMB: radial approach / recede toward the tracked body's surface.
            # Drag down (dy > 0) = move closer; drag up = pull back. Applied per-frame
            # as an exponential scale so the gesture feels uniform at any distance.
            self.camera["approach_delta"] += dy * 0.002
        elif self.camera["left_dragging"]:
            sensitivity = 0.3 * fov_ratio
            if movement_mode == 1:
                # Simple Orbit mode: LMB drag orbits the object
                self.camera["cam_look"] = "aim"
                _camera_orbit_apply(self.camera, dx, dy, sensitivity)
            else:
                # Free Flight mode: LMB trackball pivot in place (free look)
                _camera_pivot_apply(self.camera, dx, dy, sensitivity)
                self.camera["cam_look"] = "free"
        elif self.camera["right_dragging"]:
            sensitivity = 0.3 * fov_ratio
            if movement_mode == 1:
                # Simple Orbit mode: RMB drag pivots in place (free look)
                _camera_pivot_apply(self.camera, dx, dy, sensitivity)
                self.camera["cam_look"] = "free"
            else:
                # Free Flight mode: RMB trackball orbit around the pivot
                _camera_orbit_apply(self.camera, dx, dy, sensitivity)
    
        self.camera["last_x"], self.camera["last_y"] = xpos, ypos
    
    def char_callback(self, window, char):
        if self.impl: self.impl.char_callback(window, char)
    
    def key_callback(self, window, key, scancode, action, mods):
        if self.impl: self.impl.keyboard_callback(window, key, scancode, action, mods)
        if imgui.get_io().want_capture_keyboard: return
        
        # WASD: track held state for free-flight / surface walking
        flight_keys = {glfw.KEY_W: "w", glfw.KEY_A: "a", glfw.KEY_S: "s", glfw.KEY_D: "d"}
        if key in flight_keys:
            self.camera["keys"][flight_keys[key]] = (action != glfw.RELEASE)
            return
    
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_SPACE and action == glfw.PRESS:
                self.time_ctrl["paused"] = not self.time_ctrl["paused"]
            elif key in (glfw.KEY_UP, glfw.KEY_RIGHT):
                self.time_ctrl["multiplier"] = min(self.time_ctrl["multiplier"] * 2.0, 1e12)
            elif key in (glfw.KEY_DOWN, glfw.KEY_LEFT):
                self.time_ctrl["multiplier"] = max(self.time_ctrl["multiplier"] / 2.0, 1.0)
            elif key == glfw.KEY_R:
                self.time_ctrl["multiplier"] = 1.0
            elif key == glfw.KEY_MINUS:
                multiplier = 2.0 if (mods & glfw.MOD_SHIFT) else 1.1
                self.camera["exposure"] /= multiplier
            elif key == glfw.KEY_EQUAL:
                multiplier = 2.0 if (mods & glfw.MOD_SHIFT) else 1.1
                self.camera["exposure"] *= multiplier
            elif key == glfw.KEY_F11 and action == glfw.PRESS:
                monitor = glfw.get_window_monitor(window)
                if monitor:
                    w, h = getattr(self, "windowed_size", (1280, 720))
                    x, y = getattr(self, "windowed_pos", (100, 100))
                    glfw.set_window_monitor(window, None, x, y, w, h, glfw.DONT_CARE)
                else:
                    self.windowed_pos = glfw.get_window_pos(window)
                    self.windowed_size = glfw.get_window_size(window)
                    primary_monitor = glfw.get_primary_monitor()
                    mode = glfw.get_video_mode(primary_monitor)
                    glfw.set_window_monitor(window, primary_monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
            elif key == glfw.KEY_F12 and action == glfw.PRESS:
                if not self._screenshot_capturing and not self._screenshot_saving:
                    _ss_presets = [(3840, 2160), (7680, 4320), (15360, 8640)]
                    _ss_idx = max(0, min(len(_ss_presets) - 1, self.camera.get("screenshot_res_idx", 1)))
                    self._screenshot_request = _ss_presets[_ss_idx]
            elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                if getattr(self, "photo_accum_count", 0) > 1 and not getattr(self, "_screenshot_saving", False):
                    self._accum_save_request = True
    
    def resize_callback(self, window, width, height):
        if self.impl: self.impl.resize_callback(window, width, height)
        self.window_width, self.window_height = max(1, width), max(1, height)
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(window)
    
    
    
    
    
    
    
    
    
    def run(self):
        if _PERF_ENABLED:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from scripts.perf_test import PerfTracker as _BootTracker
            _BootTracker().__class__
            if _PERF_TRACKER is None:
                _PERF_INSTALL_TRACKER(_BootTracker())
    
        # ── Initialize SystemManager ──
        self.sys_mgr = sys_mgr = SystemManager()
        self.sys_mgr_spice = sys_mgr_spice = SpiceManager()
        self.get_cached_atmosphere_properties = get_cached_atmosphere_properties
        ephemeris_mode_active = False
        keplerian_mode_active = False
        active_system_name = SystemManager.SOLAR_SYSTEM_NAME
        bodies_data_raw = sys_mgr.load_default_system()
        bundle = load_system_from_data(bodies_data_raw)
        
        bundle_cmp = load_system_from_data(bodies_data_raw)
        sim_cmp = bundle_cmp["sim"]
        num_bodies_cmp = bundle_cmp["num_bodies"]
        bodies_data_cmp = bundle_cmp["bodies_data"]
        visual_data_cmp = bundle_cmp["visual_data"]
        atmo_bodies_cmp = bundle_cmp["atmo_bodies"]
        ring_bodies_cmp = bundle_cmp["ring_bodies"]
        oblate_physics_list_cmp = bundle_cmp["oblate_physics_list"]
        has_j2_cmp = bundle_cmp["has_j2"]
        has_gr_cmp = bundle_cmp["has_gr"]
        phys_star_idx_cmp = bundle_cmp["phys_star_idx"]
        
        self.bodies_data_cmp = bodies_data_cmp
        self.visual_data_cmp = visual_data_cmp
        self.atmo_bodies_cmp = atmo_bodies_cmp
        self.ring_bodies_cmp = ring_bodies_cmp
        self.num_bodies_cmp = num_bodies_cmp
        self.star_idx_cmp = bundle_cmp["star_idx"]
        
        self.visual_arr_cmp = np.array(visual_data_cmp, dtype='f4')
        self.body_radii_cmp = np.array([v[3] for v in visual_data_cmp], dtype='f4')
        self.body_colors_cmp = np.array([v[0:3] for v in visual_data_cmp], dtype='f4')
        self.is_star_arr_cmp = np.zeros(num_bodies_cmp, dtype='f4')
        for i, b in enumerate(bundle_cmp["bodies_data"]):
            if b.get('type') == 'Star':
                self.is_star_arr_cmp[i] = 1.0
        
        self.pos_snap_cmp = np.zeros((num_bodies_cmp, 3), dtype='f8')
        self.vel_snap_cmp = np.zeros((num_bodies_cmp, 3), dtype='f8')
        self.mass_snap_cmp = np.zeros(num_bodies_cmp, dtype='f8')
        self.parent_snap_cmp = np.zeros(num_bodies_cmp, dtype=np.int32)
        
        # Max orbits is usually 1024, but let's hardcode 1024 or fetch it if defined. Actually wait, let me just initialize subsys ones first.
        self.subsys_pos_buf_cmp = np.zeros((num_bodies_cmp, 3), dtype='f8')
        self.subsys_vel_buf_cmp = np.zeros((num_bodies_cmp, 3), dtype='f8')
        self.subsys_mass_buf_cmp = np.zeros(num_bodies_cmp, dtype='f8')
        self.visual_colors_f8_cmp = np.ascontiguousarray(self.visual_arr_cmp[:, 0:3], dtype='f8')
        self.orbit_data_buf_cmp = np.zeros((2000, 20), dtype='f8')
    
        sim = bundle["sim"]
        num_bodies = bundle["num_bodies"]
        bodies_data = bundle["bodies_data"]
        name_to_idx = bundle["name_to_idx"]
        visual_data = bundle["visual_data"]
        atmo_bodies = bundle["atmo_bodies"]
        ring_bodies = bundle["ring_bodies"]
        oblate_physics_list = bundle["oblate_physics_list"]
        has_j2 = bundle["has_j2"]
        has_gr = bundle["has_gr"]
        phys_star_idx = bundle["phys_star_idx"]

        # Perf harness hook: track a named body at a close orbit so the GPU
        # has real work (atmosphere + rings + ringshine fill the viewport).
        _perf_target = getattr(self, "_perf_target_body", None)
        if _perf_target:
            _found = False
            for _i, _b in enumerate(bodies_data):
                if str(_b.get("name", "")).strip().lower() == str(_perf_target).strip().lower():
                    _rad = float(visual_data[_i][3]) if _i < len(visual_data) else 0.0
                    self.camera["tracking_idx"] = _i
                    self.camera["tracking_mode"] = "body"
                    self.camera["tracking_is_cmp"] = False
                    self.camera["cam_look"] = "aim"
                    if _rad > 0.0:
                        _dist = max(_rad * 3.5, _rad + 1e-9)
                        _off = np.array([0.0, -0.9, 0.6], dtype='f8')
                        _off = _off / np.linalg.norm(_off) * _dist
                        self.camera["cam_pos_rel"] = _off
                    print(f"[Perf] GPU target: tracking '{_b.get('name')}' (idx {_i}, radius {_rad:.6e} AU)")
                    _found = True
                    break
            if not _found:
                print(f"[Perf] Target body '{_perf_target}' not found in system; keeping default camera")
    
        import os
        from PIL import Image
        import glob

        self.planet_textures = []
        self.planet_normal_textures = []
        self.planet_specular_textures = []
        self.planet_cloud_textures = []
        self.texture_slices = {} # name_lower -> 1-based slice_idx
        self.texture_mean_colors = {} # name_lower -> np.ndarray([r,g,b]) in linear space

        def _prepare_cloud_image(img):
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img_rgba = img.convert('RGBA')
            else:
                img_l = img.convert('L')
                white = Image.new('L', img_l.size, 255)
                img_rgba = Image.merge('RGBA', (white, white, white, img_l))
            return img_rgba.resize((2048, 1024), Image.Resampling.LANCZOS)

        self.ring_textures_front = {}  # name_lower -> PIL.Image (4096, 1)
        self.ring_textures_back = {}   # name_lower -> PIL.Image (4096, 1)
        self.ring_gl_textures_front = {} # name_lower -> ModernGL Texture
        self.ring_gl_textures_back = {}  # name_lower -> ModernGL Texture
        self.ring_textures = self.ring_textures_front
        self.ring_gl_textures = self.ring_gl_textures_front
        textures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'textures')

        from engine.rendering.texture_streamer import TextureStreamer

        self.texture_streamer = TextureStreamer(textures_dir, fallback_size=(1024, 512))
        self.texture_slices = self.texture_streamer.name_to_idx
        self.texture_mean_colors = self.texture_streamer.texture_mean_colors

        # Ring texture loading
        if os.path.exists(textures_dir):
            for root, dirs, files in os.walk(textures_dir):
                for dir_name in dirs:
                    folder_path = os.path.join(root, dir_name)
                    r_files = glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.jpg'))
                    name_lower = dir_name.lower()
                    r_front_path, r_back_path, r_single_path = None, None, None
                    for f in r_files:
                        fname_lower = os.path.splitext(os.path.basename(f))[0].lower()
                        if fname_lower in [name_lower + "_ring_front", name_lower + "_rings_front", name_lower + "_front", "ring_front", "rings_front", "front"]:
                            r_front_path = f
                        elif fname_lower in [name_lower + "_ring_back", name_lower + "_rings_back", name_lower + "_back", "ring_back", "rings_back", "back"]:
                            r_back_path = f
                        elif fname_lower in [name_lower + "_ring", name_lower + "_rings", "ring", "rings"]:
                            r_single_path = f
                    if r_front_path:
                        try:
                            img_front = Image.open(r_front_path).convert('RGBA')
                            img_back = Image.open(r_back_path).convert('RGBA') if r_back_path else img_front
                            self.ring_gl_textures_front[name_lower] = img_front
                            self.ring_gl_textures_back[name_lower] = img_back
                            self.ring_textures_front[name_lower] = img_front.resize((4096, 1), Image.Resampling.LANCZOS)
                            self.ring_textures_back[name_lower] = img_back.resize((4096, 1), Image.Resampling.LANCZOS)
                        except Exception as e:
                            print(f"Failed to load front/back ring textures in folder {folder_path}: {e}")
                    elif r_single_path:
                        try:
                            img = Image.open(r_single_path).convert('RGBA')
                            self.ring_gl_textures_front[name_lower] = img
                            self.ring_gl_textures_back[name_lower] = img
                            img_ds = img.resize((4096, 1), Image.Resampling.LANCZOS)
                            self.ring_textures_front[name_lower] = img_ds
                            self.ring_textures_back[name_lower] = img_ds
                        except Exception as e:
                            print(f"Failed to load ring texture in folder {folder_path}: {e}")

            # Root ring textures
            ring_files = glob.glob(os.path.join(textures_dir, '*_ring*.png')) + glob.glob(os.path.join(textures_dir, '*_rings*.png'))
            ring_base_names = set()
            for f in ring_files:
                bname = os.path.splitext(os.path.basename(f))[0]
                clean_name = bname.replace("_rings_front", "").replace("_ring_front", "").replace("_rings_back", "").replace("_ring_back", "").replace("_front", "").replace("_back", "").replace("_rings", "").replace("_ring", "")
                if clean_name:
                    ring_base_names.add(clean_name)
            for name in sorted(list(ring_base_names)):
                name_lower = name.lower()
                if name_lower in self.ring_textures_front:
                    continue
                f_front, f_back, f_single = None, None, None
                for ext in ['.png', '.jpg']:
                    for pattern in [f"{name}_ring_front{ext}", f"{name}_rings_front{ext}", f"{name}_front{ext}"]:
                        p = os.path.join(textures_dir, pattern)
                        if os.path.exists(p): f_front = p; break
                    for pattern in [f"{name}_ring_back{ext}", f"{name}_rings_back{ext}", f"{name}_back{ext}"]:
                        p = os.path.join(textures_dir, pattern)
                        if os.path.exists(p): f_back = p; break
                    for pattern in [f"{name}_ring{ext}", f"{name}_rings{ext}"]:
                        p = os.path.join(textures_dir, pattern)
                        if os.path.exists(p): f_single = p; break
                try:
                    if f_front:
                        img_front = Image.open(f_front).convert('RGBA')
                        img_back = Image.open(f_back).convert('RGBA') if f_back else img_front
                        self.ring_gl_textures_front[name_lower] = img_front
                        self.ring_gl_textures_back[name_lower] = img_back
                        self.ring_textures_front[name_lower] = img_front.resize((4096, 1), Image.Resampling.LANCZOS)
                        self.ring_textures_back[name_lower] = img_back.resize((4096, 1), Image.Resampling.LANCZOS)
                    elif f_single:
                        img = Image.open(f_single).convert('RGBA')
                        self.ring_gl_textures_front[name_lower] = img
                        self.ring_gl_textures_back[name_lower] = img
                        img_ds = img.resize((4096, 1), Image.Resampling.LANCZOS)
                        self.ring_textures_front[name_lower] = img_ds
                        self.ring_textures_back[name_lower] = img_ds
                except Exception as e:
                    print(f"Failed to load root ring texture {name}: {e}")
                            
                    if r_front_path:
                        try:
                            img_front = Image.open(r_front_path).convert('RGBA')
                            img_back = Image.open(r_back_path).convert('RGBA') if r_back_path else img_front
                            self.ring_gl_textures_front[name_lower] = img_front
                            self.ring_gl_textures_back[name_lower] = img_back
                            self.ring_textures_front[name_lower] = img_front.resize((4096, 1), Image.Resampling.LANCZOS)
                            self.ring_textures_back[name_lower] = img_back.resize((4096, 1), Image.Resampling.LANCZOS)
                        except Exception as e:
                            print(f"Failed to load front/back ring textures in folder {path}: {e}")
                    elif r_single_path:
                        try:
                            img = Image.open(r_single_path).convert('RGBA')
                            self.ring_gl_textures_front[name_lower] = img
                            self.ring_gl_textures_back[name_lower] = img
                            img_ds = img.resize((4096, 1), Image.Resampling.LANCZOS)
                            self.ring_textures_front[name_lower] = img_ds
                            self.ring_textures_back[name_lower] = img_ds
                        except Exception as e:
                            print(f"Failed to load ring texture in folder {path}: {e}")
        
        if has_j2:
            oblate_indices = np.array([x[0] for x in oblate_physics_list], dtype=np.int32)
            oblate_j2 = np.array([x[1] for x in oblate_physics_list], dtype=np.float64)
            oblate_req = np.array([x[3] for x in oblate_physics_list], dtype=np.float64)
            oblate_poles = np.array([x[4] for x in oblate_physics_list], dtype=np.float64)
            oblate_masses = np.array([x[5] for x in oblate_physics_list], dtype=np.float64)
    
        if atmo_bodies:
            print(f"[Atmosphere] Parsed atmosphere data for {len(atmo_bodies)} bodies")
    
        self.shared_state = {
            "lock": threading.Lock(),
            "has_j2": has_j2,
            "has_gr": has_gr,
            "phys_star_idx": phys_star_idx,
            "oblate_physics_list": oblate_physics_list if has_j2 else [],
            "t": 0.0,
            "pos": np.zeros((num_bodies, 3), dtype='f8'),
            "vel": np.zeros((num_bodies, 3), dtype='f8'),
            "mass": np.zeros(num_bodies, dtype='f8'),
            "radii": np.array([v[3] for v in visual_data], dtype='f8'),
            "parent_indices": np.full(num_bodies, -1, dtype=np.int32),
            "is_star_mask": np.array([b.get('type') == 'Star' for b in bodies_data], dtype=np.bool_),
            "tree_indices": np.zeros(num_bodies, dtype=np.int32),
            "tree_depths": np.zeros(num_bodies, dtype=np.int32),
            "hierarchy_version": 0,
            "timeline_active": False,
            "timeline_progress": 0.0,
            "timeline_pos": np.zeros((0, num_bodies, 3), dtype='f4'),
            "timeline_vel": np.zeros((0, num_bodies, 3), dtype='f4'),
            "timeline_times": np.zeros(0, dtype='f8'),
            "crud_queue": [],
            "crud_completed": [],
            "rebuild_flag": False,
            "oblate_indices": oblate_indices if has_j2 else None,
            "oblate_j2": oblate_j2 if has_j2 else None,
            "oblate_req": oblate_req if has_j2 else None,
            "oblate_poles": oblate_poles if has_j2 else None,
            "oblate_masses": oblate_masses if has_j2 else None,
            "phys_star_idx": phys_star_idx,
            "ephemeris_mode": False,
            "keplerian_mode": False,
            "keplerian_reextract": False,
            # ── System switching infrastructure ──
            "system_switch_request": None,       # set to dict with system bundle to trigger switch
            "system_switch_complete": False,      # physics thread sets True when switch is done
            "system_new_bundle": None,            # physics thread puts new system info here
            "system_snapshot_out": None,          # physics thread puts saved snapshot here
        }
        
        _init_arr = sim.arr[:num_bodies]
        self.shared_state["pos"][:, 0] = _init_arr[:, 0]
        self.shared_state["pos"][:, 1] = _init_arr[:, 2]
        self.shared_state["pos"][:, 2] = -_init_arr[:, 1]
        self.shared_state["vel"][:, 0] = _init_arr[:, 3]
        self.shared_state["vel"][:, 1] = _init_arr[:, 5]
        self.shared_state["vel"][:, 2] = -_init_arr[:, 4]
        self.shared_state["mass"][:] = _init_arr[:, 9]
            
        parent_indices = bundle["parent_indices"].copy()
        is_star_mask = np.array([b.get('type') == 'Star' for b in bodies_data], dtype=np.bool_)
        parent_indices = update_hierarchy(sim, num_bodies, parent_indices, is_star_mask)
        init_positions = _init_arr[:, 0:3]
        tree_indices_init, tree_depths_init = build_tree_order(parent_indices, num_bodies, init_positions)
        self.shared_state["parent_indices"][:] = parent_indices
        self.shared_state["tree_indices"][:] = tree_indices_init
        self.shared_state["tree_depths"][:] = tree_depths_init

        # Comparison system shared state initialization
        oblate_indices_cmp = None
        oblate_j2_cmp = None
        oblate_req_cmp = None
        oblate_poles_cmp = None
        oblate_masses_cmp = None
        if has_j2_cmp:
            oblate_indices_cmp = np.array([x[0] for x in oblate_physics_list_cmp], dtype=np.int32)
            oblate_j2_cmp = np.array([x[1] for x in oblate_physics_list_cmp], dtype=np.float64)
            oblate_req_cmp = np.array([x[3] for x in oblate_physics_list_cmp], dtype=np.float64)
            oblate_poles_cmp = np.array([x[4] for x in oblate_physics_list_cmp], dtype=np.float64)
            oblate_masses_cmp = np.array([x[5] for x in oblate_physics_list_cmp], dtype=np.float64)

        self.shared_state_cmp = {
            "lock": threading.Lock(),
            "has_j2": has_j2_cmp,
            "has_gr": has_gr_cmp,
            "phys_star_idx": phys_star_idx_cmp,
            "oblate_physics_list": oblate_physics_list_cmp if has_j2_cmp else [],
            "t": 0.0,
            "pos": np.zeros((num_bodies_cmp, 3), dtype='f8'),
            "vel": np.zeros((num_bodies_cmp, 3), dtype='f8'),
            "mass": np.zeros(num_bodies_cmp, dtype='f8'),
            "radii": np.array([v[3] for v in visual_data_cmp], dtype='f8'),
            "parent_indices": np.full(num_bodies_cmp, -1, dtype=np.int32),
            "is_star_mask": np.array([b.get('type') == 'Star' for b in bodies_data_cmp], dtype=np.bool_),
            "tree_indices": np.zeros(num_bodies_cmp, dtype=np.int32),
            "tree_depths": np.zeros(num_bodies_cmp, dtype=np.int32),
            "hierarchy_version": 0,
            "timeline_active": False,
            "timeline_progress": 0.0,
            "timeline_pos": np.zeros((0, num_bodies_cmp, 3), dtype='f4'),
            "timeline_vel": np.zeros((0, num_bodies_cmp, 3), dtype='f4'),
            "timeline_times": np.zeros(0, dtype='f8'),
            "crud_queue": [],
            "crud_completed": [],
            "rebuild_flag": False,
            "oblate_indices": oblate_indices_cmp if has_j2_cmp else None,
            "oblate_j2": oblate_j2_cmp if has_j2_cmp else None,
            "oblate_req": oblate_req_cmp if has_j2_cmp else None,
            "oblate_poles": oblate_poles_cmp if has_j2_cmp else None,
            "oblate_masses": oblate_masses_cmp if has_j2_cmp else None,
            "system_switch_request": None,
            "system_switch_complete": False,
            "system_new_bundle": None,
            "system_snapshot_out": None,
        }
        
        _init_arr_cmp = sim_cmp.arr[:num_bodies_cmp]
        self.shared_state_cmp["pos"][:, 0] = _init_arr_cmp[:, 0]
        self.shared_state_cmp["pos"][:, 1] = _init_arr_cmp[:, 2]
        self.shared_state_cmp["pos"][:, 2] = -_init_arr_cmp[:, 1]
        self.shared_state_cmp["vel"][:, 0] = _init_arr_cmp[:, 3]
        self.shared_state_cmp["vel"][:, 1] = _init_arr_cmp[:, 5]
        self.shared_state_cmp["vel"][:, 2] = -_init_arr_cmp[:, 4]
        self.shared_state_cmp["mass"][:] = _init_arr_cmp[:, 9]
        
        parent_indices_cmp = bundle_cmp["parent_indices"].copy()
        is_star_mask_cmp = np.array([b.get('type') == 'Star' for b in bodies_data_cmp], dtype=np.bool_)
        parent_indices_cmp = update_hierarchy(sim_cmp, num_bodies_cmp, parent_indices_cmp, is_star_mask_cmp)
        init_positions_cmp = _init_arr_cmp[:, 0:3]
        tree_indices_init_cmp, tree_depths_init_cmp = build_tree_order(parent_indices_cmp, num_bodies_cmp, init_positions_cmp)
        self.shared_state_cmp["parent_indices"][:] = parent_indices_cmp
        self.shared_state_cmp["tree_indices"][:] = tree_indices_init_cmp
        self.shared_state_cmp["tree_depths"][:] = tree_depths_init_cmp

        np.copyto(self.pos_snap_cmp, self.shared_state_cmp["pos"])
        np.copyto(self.vel_snap_cmp, self.shared_state_cmp["vel"])
        np.copyto(self.mass_snap_cmp, self.shared_state_cmp["mass"])
        np.copyto(self.parent_snap_cmp, self.shared_state_cmp["parent_indices"])
        self.tree_indices_snap_cmp = np.zeros(num_bodies_cmp, dtype=np.int32)
        self.tree_depths_snap_cmp = np.zeros(num_bodies_cmp, dtype=np.int32)
        np.copyto(self.tree_indices_snap_cmp, self.shared_state_cmp["tree_indices"])
        np.copyto(self.tree_depths_snap_cmp, self.shared_state_cmp["tree_depths"])
        self.visual_colors_f8_cmp = np.ascontiguousarray(self.visual_arr_cmp[:, 0:3], dtype='f8')

        self.time_ctrl_cmp = {
            "paused": True,
            "multiplier": 1.0,
            "time_direction": 1,
            "render_timeline": False,
            "cancel_render": False,
            "target_t": 0.0,
            "sync_t": None,
            "timeline_playing": False,
            "timeline_speed": 10.0,
            "timeline_scrub_float": 0.0,
        }
        
        # Trigger an initial system switch request for comparison system to compile its VBO/VAOs and populate shared states
        req_cmp = {
            "old_bodies_data": bodies_data_cmp,
            "old_visual_data": visual_data_cmp,
            "old_atmo_bodies": atmo_bodies_cmp,
            "old_ring_bodies": ring_bodies_cmp,
            "old_star_idx": self.star_idx_cmp,
            "new_bundle": bundle_cmp,
        }
        self.shared_state_cmp["system_switch_request"] = req_cmp

        running_cmp = [True]
        physics_thread_cmp = threading.Thread(target=physics_loop, args=(sim_cmp, num_bodies_cmp, self.shared_state_cmp, self.time_ctrl_cmp, running_cmp), daemon=True)
        physics_thread_cmp.start()
        
        print("[Init] Warming up Numba JIT functions...")
        try:
            _update_hierarchy_core(np.zeros((2, 3)), np.zeros(2), np.array([-1, 0], dtype=np.int32), 2, np.zeros(2, dtype=np.bool_))
            
                
            compute_keplerian_elements(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
            compute_barycenters(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), np.array([-1, 0], dtype=np.int32), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2))
            compute_all_orbits_batch(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), np.array([-1, 0], dtype=np.int32), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), np.zeros((2, 3), dtype=np.float64), np.zeros(3), 1.0, 10, np.zeros((10, 20), dtype=np.float64), np.zeros(2, dtype=np.float64), np.zeros(3, dtype=np.float64))
            
            # Warm up the main integrator and custom force calculations
            dummy_sim = Simulation()
            dummy_sim.add(m=1.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
            dummy_sim.add(m=1e-6, x=1.0, y=0.0, z=0.0, vx=0.0, vy=1.0, vz=0.0)
            dummy_sim.integrate(1e-6)
        except Exception as e:
            print(f"[Init] Warmup warning: {e}")
            
        running = [True]
        physics_thread = threading.Thread(target=physics_loop, args=(sim, num_bodies, self.shared_state, self.time_ctrl, running), daemon=True)
        physics_thread.start()
    
    
        def _glfw_error_callback(code, msg):
            print(f"[GLFW Error {code}] {msg}")
        glfw.set_error_callback(_glfw_error_callback)
    
        if not glfw.init(): raise Exception("GLFW failed to initialize")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
        window = glfw.create_window(self.window_width, self.window_height, "Stellar Forge", None, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW failed to create window. Your GPU may not support OpenGL 4.6.")
    
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        ctx = moderngl.create_context()
        self.ctx = ctx
        ctx.enable(moderngl.DEPTH_TEST) 
    
        
        imgui.create_context()
        
        io = imgui.get_io()
        io.fonts.add_font_default()
        
        # Merge Segoe UI Symbol for astronomical symbols, arrows, etc.
        font_config = imgui.FontConfig(
            merge_mode=True, 
            pixel_snap_h=True,
            oversample_h=1,
            oversample_v=1,
            rasterizer_multiply=1.5
        )
        # Unicode ranges for Greek, Math Operators, Misc Symbols (Sun, Moon, Planets), etc.
        ranges = imgui.core.GlyphRanges([
            0x0370, 0x03FF, # Greek (omega, etc)
            0x2200, 0x22FF, # Mathematical Operators (earth circle plus)
            0x25A0, 0x25FF, # Geometric Shapes
            0x2600, 0x26FF, # Misc Symbols (Sun, Moon)
            0x2700, 0x27BF, # Dingbats
            0
        ])
        
        import os
        font_path = r"C:\Windows\Fonts\seguisym.ttf"
        if os.path.exists(font_path):
            io.fonts.add_font_from_file_ttf(font_path, 13.0, font_config=font_config, glyph_ranges=ranges)
            
        self.impl = ModernGLGlfwRenderer(window, ctx, attach_callbacks=False)
        
        glfw.set_scroll_callback(window, self.scroll_callback)
        glfw.set_mouse_button_callback(window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(window, self.cursor_pos_callback)
        glfw.set_window_size_callback(window, self.resize_callback)
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_char_callback(window, self.char_callback)
    
    
    
        prog_spheres = ctx.program(vertex_shader=sphere_vertex_shader, fragment_shader=sphere_fragment_shader)
        prog_culling_compute = ctx.compute_shader(culling_compute_shader)
    
        prog_gpu_orbits = ctx.program(vertex_shader=orbit_vertex_shader, fragment_shader=orbit_fragment_shader)
        if 'u_scene_depth' in prog_gpu_orbits:
            prog_gpu_orbits['u_scene_depth'].value = 9
        prog_orbit_compute = ctx.compute_shader(orbit_compute_shader)
    
        prog_ephem_orbits = ctx.program(vertex_shader=ephem_orbit_vertex_shader, fragment_shader=ephem_orbit_fragment_shader)
    
        prog_rings = ctx.program(vertex_shader=ring_vertex_shader, fragment_shader=ring_fragment_shader)
        if 'u_ring_texture_front' in prog_rings:
            prog_rings['u_ring_texture_front'].value = 4
        if 'u_ring_texture_back' in prog_rings:
            prog_rings['u_ring_texture_back'].value = 5
    
        prog_atmo = ctx.program(vertex_shader=atmo_vertex_shader, fragment_shader=atmo_fragment_shader)
        
        atmo_frag_lowres_src = atmo_fragment_shader.replace(
            "layout(location = 0, index = 0) out vec4 out_scattered;",
            "layout(location = 0) out vec4 out_scattered;"
        ).replace(
            "layout(location = 0, index = 1) out vec4 out_transmittance;",
            "layout(location = 1) out vec4 out_transmittance;"
        )
        prog_atmo_lowres = ctx.program(vertex_shader=atmo_vertex_shader, fragment_shader=atmo_frag_lowres_src)
        
        for p in (prog_atmo, prog_atmo_lowres):
            if 'u_ringshine_lut' in p:
                p['u_ringshine_lut'].value = 6
            if 'u_ringshine_cdf_lut' in p:
                p['u_ringshine_cdf_lut'].value = 7
            if 'u_ringshine_map' in p:
                p['u_ringshine_map'].value = 8
            if 'u_depth_texture' in p:
                p['u_depth_texture'].value = 9
            if 'u_transmittance_lut' in p:
                p['u_transmittance_lut'].value = 1
            if 'u_multi_scatter_lut' in p:
                p['u_multi_scatter_lut'].value = 3
            if 'u_ring_gradients' in p:
                p['u_ring_gradients'].value = 0
        
        self.prog_bloom_down = ctx.program(vertex_shader=bloom_downsample_shader_vs, fragment_shader=bloom_downsample_shader_fs)
        self.prog_bloom_up = ctx.program(vertex_shader=bloom_upsample_shader_vs, fragment_shader=bloom_upsample_shader_fs)
        self.prog_composite = ctx.program(vertex_shader=composite_shader_vs, fragment_shader=composite_shader_fs)
        self.prog_accum = ctx.program(vertex_shader=accum_shader_vs, fragment_shader=accum_shader_fs)
        self.prog_atmo_composite = ctx.program(vertex_shader=atmo_composite_shader_vs, fragment_shader=atmo_composite_shader_fs)
        self.prog_atmo_upsample = ctx.program(vertex_shader=bloom_downsample_shader_vs, fragment_shader=atmo_upsample_fragment_shader)
        if 'u_lowres_scatter' in self.prog_atmo_upsample:
            self.prog_atmo_upsample['u_lowres_scatter'].value = 0
        if 'u_lowres_trans' in self.prog_atmo_upsample:
            self.prog_atmo_upsample['u_lowres_trans'].value = 1
        if 'u_highres_depth' in self.prog_atmo_upsample:
            self.prog_atmo_upsample['u_highres_depth'].value = 9
        
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')
        quad_vbo = ctx.buffer(quad_vertices.tobytes())
        self.quad_vao_down = ctx.vertex_array(self.prog_bloom_down, [(quad_vbo, '2f', 'in_position')])
        self.quad_vao_up = ctx.vertex_array(self.prog_bloom_up, [(quad_vbo, '2f', 'in_position')])
        self.quad_vao_comp = ctx.vertex_array(self.prog_composite, [(quad_vbo, '2f', 'in_position')])
        self.quad_vao_accum = ctx.vertex_array(self.prog_accum, [(quad_vbo, '2f', 'in_position')])
        self.quad_vao_atmo_comp = ctx.vertex_array(self.prog_atmo_composite, [(quad_vbo, '2f', 'in_position')])
        self.quad_vao_atmo_upsample = ctx.vertex_array(self.prog_atmo_upsample, [(quad_vbo, '2f', 'in_position')])
        
        # Compile Habitable Zone Shader
        self.prog_hz = ctx.program(vertex_shader=hz_vertex_shader, fragment_shader=hz_fragment_shader)
        
        # Generate HZ unit ring geometry: x=cos, z=sin, y=lerp factor (0 for inner, 1 for outer)
        segments = 64
        theta = np.linspace(0, 2.0 * math.pi, segments + 1)
        vertices = []
        for t in theta:
            cos_t = math.cos(t)
            sin_t = math.sin(t)
            # Inner vertex: y=0.0
            vertices.extend([cos_t, 0.0, sin_t])
            # Outer vertex: y=1.0
            vertices.extend([cos_t, 1.0, sin_t])
        hz_vertices = np.array(vertices, dtype='f4')
        self.hz_vbo = ctx.buffer(hz_vertices.tobytes())
        self.hz_vao = ctx.vertex_array(self.prog_hz, [(self.hz_vbo, '3f', 'in_position')])
        self.hz_num_vertices = len(vertices) // 3

        self.prog_atmo_lut = ctx.program(vertex_shader=atmo_lut_vertex_shader, fragment_shader=atmo_lut_fragment_shader)
        self.prog_multi_scatter_lut = ctx.program(vertex_shader=atmo_lut_vertex_shader, fragment_shader=multi_scatter_lut_fragment_shader)
        lut_vbo = ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'))
        self.lut_vao = ctx.vertex_array(self.prog_atmo_lut, [(lut_vbo, '2f', 'in_position')])
        self.multi_scatter_vao = ctx.vertex_array(self.prog_multi_scatter_lut, [(lut_vbo, '2f', 'in_position')])
        
        if 'u_transmittance_lut' in prog_atmo: prog_atmo['u_transmittance_lut'].value = 1
        if 'u_multi_scatter_lut' in prog_atmo: prog_atmo['u_multi_scatter_lut'].value = 3

        ring_precomputed = []
        RING_SEGMENTS = 128
        

        for body_idx, rings_data, pole_render, body_radius_au in ring_bodies:
            pole_n = pole_render / np.linalg.norm(pole_render)
            ref = np.array([0., 0., 1.])
            tangent = np.cross(pole_n, ref)
            if np.linalg.norm(tangent) < 1e-10:
                ref = np.array([1., 0., 0.])
                tangent = np.cross(pole_n, ref)
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(pole_n, tangent)
            R_ring = np.column_stack([tangent, pole_n, bitangent])
            
            b_data = bodies_data[body_idx]
            b_name = b_data['name']
            name_lower = b_name.lower()
            if name_lower in self.ring_textures_front and len(rings_data) > 0:
                tex_inner_f = b_data.get('ring_texture_inner')
                tex_outer_f = b_data.get('ring_texture_outer')
                
                if tex_inner_f is not None and tex_outer_f is not None:
                    tex_min_inner = float(tex_inner_f)
                    tex_max_outer = float(tex_outer_f)
                    procedural_segments = [seg for seg in rings_data if seg['outer'] <= tex_min_inner + 1e-4 or seg['inner'] >= tex_max_outer - 1e-4]
                else:
                    tex_min_inner = min(seg['inner'] for seg in rings_data)
                    tex_max_outer = max(seg['outer'] for seg in rings_data)
                    procedural_segments = []
                    
                inner_r = tex_min_inner * body_radius_au
                outer_r = tex_max_outer * body_radius_au
                
                # Compute weighted average parameters for textured region
                textured_segs = [seg for seg in rings_data if seg['inner'] >= tex_min_inner - 1e-4 and seg['outer'] <= tex_max_outer + 1e-4]
                if not textured_segs:
                    textured_segs = rings_data
                    
                total_w = 0.0
                sum_color = np.zeros(3)
                sum_opacity = 0.0
                sum_scatter = 0.0
                sum_asymmetry = 0.0
                sum_backscatter = 0.0
                
                for seg in textured_segs:
                    w = seg.get('opacity', 1.0) * (seg['outer'] - seg['inner'])
                    if w < 1e-9:
                        w = 1e-9
                    total_w += w
                    sum_color += np.array(hex_to_rgb(seg.get('color', '#ffffff'))) * w
                    sum_opacity += seg.get('opacity', 1.0) * w
                    sum_scatter += seg.get('scatter', 2.5) * w
                    sum_asymmetry += seg.get('asymmetry', 0.7) * w
                    sum_backscatter += seg.get('backscatter', -0.3) * w
                    
                # For textured rings, color and opacity come directly from the texture map
                r_color = (1.0, 1.0, 1.0)
                r_opacity = 1.0
                if total_w > 0.0:
                    r_scatter = sum_scatter / total_w
                    r_asymmetry = sum_asymmetry / total_w
                    r_backscatter = sum_backscatter / total_w
                else:
                    first = textured_segs[0]
                    r_scatter = first.get('scatter', 2.5)
                    r_asymmetry = first.get('asymmetry', 0.7)
                    r_backscatter = first.get('backscatter', -0.3)
                
                # Sample the texture
                img_data = np.frombuffer(self.ring_textures_front[name_lower].tobytes(), dtype=np.uint8).astype('f4') / 255.0
                img_data = img_data.reshape(4096, 4)
                tex_sampled = img_data # take 4096 samples
                
                # Generate shadow gradient from texture (no procedural gradient)
                shadow_grad = generate_ring_shadow_grad(sorted_gradient=[], tex_sampled=tex_sampled)
                colors5 = compute_5_ring_colors(tex_sampled=tex_sampled, raw_color=r_color)
                
                ring_precomputed.append({
                    'body_idx': body_idx,
                    'pole': pole_n.astype('f4'),
                    'inner_r': inner_r,
                    'outer_r': outer_r,
                    'opacity': r_opacity,
                    'scatter': r_scatter,
                    'asymmetry': r_asymmetry,
                    'backscatter': r_backscatter,
                    'shadow_grad': shadow_grad,
                    'raw_color': r_color,
                    'gradient': [],
                    'tex_sampled': tex_sampled,
                    '5colors': colors5,
                    'is_textured': True,
                    'row_idx': len(ring_precomputed),
                })
                
                # Append remaining outer/inner procedural segments outside texture range
                for ring_seg in procedural_segments:
                    p_inner_r = ring_seg['inner'] * body_radius_au
                    p_outer_r = ring_seg['outer'] * body_radius_au
                    p_r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
                    p_r_opacity = ring_seg.get('opacity', 1.0)
                    p_r_scatter = ring_seg.get('scatter', 2.5)
                    p_r_asymmetry = ring_seg.get('asymmetry', 0.7)
                    p_r_backscatter = ring_seg.get('backscatter', -0.3)
                    
                    sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
                    shadow_grad = generate_ring_shadow_grad(sorted_gradient, tex_sampled=None)
                    colors5 = compute_5_ring_colors(tex_sampled=None, raw_color=p_r_color, gradient=sorted_gradient)
                    
                    ring_precomputed.append({
                        'body_idx': body_idx,
                        'pole': pole_n.astype('f4'),
                        'inner_r': p_inner_r,
                        'outer_r': p_outer_r,
                        'opacity': p_r_opacity,
                        'scatter': p_r_scatter,
                        'asymmetry': p_r_asymmetry,
                        'backscatter': p_r_backscatter,
                        'shadow_grad': shadow_grad,
                        'raw_color': p_r_color,
                        'gradient': sorted_gradient,
                        'tex_sampled': None,
                        '5colors': colors5,
                        'is_textured': False,
                        'row_idx': len(ring_precomputed),
                    })
            else:
                # Procedural or non-textured rings (original logic)
                for ring_seg in rings_data:
                    inner_r = ring_seg['inner'] * body_radius_au
                    outer_r = ring_seg['outer'] * body_radius_au
                    r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
                    r_opacity = ring_seg.get('opacity', 1.0)
                    r_scatter = ring_seg.get('scatter', 2.5)
                    r_asymmetry = ring_seg.get('asymmetry', 0.7)
                    r_backscatter = ring_seg.get('backscatter', -0.3)
                    
                    sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
                    
                    shadow_grad = generate_ring_shadow_grad(sorted_gradient, tex_sampled=None)
                    colors5 = compute_5_ring_colors(tex_sampled=None, raw_color=r_color, gradient=sorted_gradient)
                    
                    ring_precomputed.append({
                        'body_idx': body_idx,
                        'pole': pole_n.astype('f4'),
                        'inner_r': inner_r,
                        'outer_r': outer_r,
                        'opacity': r_opacity,
                        'scatter': r_scatter,
                        'asymmetry': r_asymmetry,
                        'backscatter': r_backscatter,
                        'shadow_grad': shadow_grad,
                        'raw_color': r_color,
                        'gradient': sorted_gradient,
                        'tex_sampled': None,
                        '5colors': colors5,
                        'is_textured': False,
                        'row_idx': len(ring_precomputed),
                    })
    
        ring_gradient_tex = ctx.texture((4096, 16), 4, dtype='f4')
        ring_gradient_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        ring_gradient_tex.repeat_x = False
        ring_gradient_tex.repeat_y = False
        self.body_ring_indices = rebuild_ring_gradients_atlas(ring_precomputed, ring_gradient_tex)


        def build_ringshine_lut(ctx):
            res_x, res_y = 256, 256
            sin_lats = np.linspace(0.001, 0.999, res_x, dtype=np.float32)
            radii = np.linspace(1.001, 5.0, res_y, dtype=np.float32)

            sin_lat_grid = sin_lats[None, :]
            cos_lat_grid = np.sqrt(np.maximum(0.0, 1.0 - sin_lat_grid**2))
            r_grid = radii[:, None]

            num_alpha = 180
            alpha = np.linspace(0.0, 2.0 * np.pi, num_alpha, endpoint=False, dtype=np.float32)[:, None, None]

            cos_alpha = np.cos(alpha)
            d2 = r_grid**2 + 1.0 - 2.0 * r_grid * cos_lat_grid * cos_alpha
            d = np.sqrt(np.maximum(d2, 1e-6))

            ndotl = np.maximum(0.0, (r_grid * cos_lat_grid * cos_alpha - 1.0) / d)
            ring_mu = sin_lat_grid / d

            d_alpha = (2.0 * np.pi) / num_alpha
            diff_irradiance = (ndotl * ring_mu / np.maximum(d2, 1e-6)) * r_grid * d_alpha

            lut_data = np.sum(diff_irradiance, axis=0, dtype=np.float32)

            tex = ctx.texture((res_x, res_y), 1, lut_data.tobytes(), dtype='f4')
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False

            # Precompute 3D Cumulative Distribution Function (CDF) Texture (64x128x128)
            res_cdf_lat, res_cdf_r, res_cdf_theta = 64, 128, 128
            cdf_sin_lats = np.linspace(0.001, 0.999, res_cdf_lat, dtype=np.float32)[:, None, None]
            cdf_cos_lats = np.sqrt(np.maximum(0.0, 1.0 - cdf_sin_lats**2))
            cdf_radii = np.linspace(1.001, 5.0, res_cdf_r, dtype=np.float32)[None, :, None]
            cdf_thetas = np.linspace(0.0, np.pi, res_cdf_theta, dtype=np.float32)[None, None, :]

            cdf_cos_alpha = np.cos(cdf_thetas)
            cdf_d2 = cdf_radii**2 + 1.0 - 2.0 * cdf_radii * cdf_cos_lats * cdf_cos_alpha
            cdf_d = np.sqrt(np.maximum(cdf_d2, 1e-6))

            cdf_ndotl = np.maximum(0.0, (cdf_radii * cdf_cos_lats * cdf_cos_alpha - 1.0) / cdf_d)
            cdf_ring_mu = cdf_sin_lats / cdf_d
            cdf_d_alpha = np.pi / max(1, res_cdf_theta - 1)
            cdf_diff_irrad = (cdf_ndotl * cdf_ring_mu / np.maximum(cdf_d2, 1e-6)) * cdf_radii * cdf_d_alpha

            # Smooth trapezoidal integration for C1 continuous CDF
            trapz_step = 0.5 * (cdf_diff_irrad[:, :, :-1] + cdf_diff_irrad[:, :, 1:])
            cdf_cum_irrad = np.zeros_like(cdf_diff_irrad)
            cdf_cum_irrad[:, :, 1:] = np.cumsum(trapz_step, axis=2)

            cdf_totals = cdf_cum_irrad[:, :, -1:]
            cdf_normalized = np.where(cdf_totals > 1e-12, cdf_cum_irrad / cdf_totals, 1.0).astype(np.float32)

            cdf_tex = ctx.texture3d((res_cdf_theta, res_cdf_r, res_cdf_lat), 1, cdf_normalized.tobytes(), dtype='f4')
            cdf_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            cdf_tex.repeat_x = False
            cdf_tex.repeat_y = False
            cdf_tex.repeat_z = False

            return tex, cdf_tex

        ringshine_lut_tex, ringshine_cdf_tex = build_ringshine_lut(ctx)

        self.ringshine_map_tex = ctx.texture((128, 1040), 4, dtype='f4')
        self.ringshine_map_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ringshine_map_tex.repeat_x = True
        self.ringshine_map_tex.repeat_y = False
        self.ringshine_map_fbo = ctx.framebuffer(color_attachments=[self.ringshine_map_tex])
        self.prog_ringshine_map = ctx.program(vertex_shader=ringshine_map_vertex_shader, fragment_shader=ringshine_map_fragment_shader)
        if 'u_ring_gradients' in self.prog_ringshine_map: self.prog_ringshine_map['u_ring_gradients'].value = 0
        if 'u_ringshine_lut' in self.prog_ringshine_map: self.prog_ringshine_map['u_ringshine_lut'].value = 6
        if 'u_ringshine_cdf_lut' in self.prog_ringshine_map: self.prog_ringshine_map['u_ringshine_cdf_lut'].value = 7
        self.ringshine_map_vao = ctx.vertex_array(self.prog_ringshine_map, [(lut_vbo, '2f', 'in_position')])
    
        ring_render_groups = []
        rings_by_body_init = {}
        for ring in ring_precomputed:
            bi = ring['body_idx']
            if bi not in rings_by_body_init:
                rings_by_body_init[bi] = []
            rings_by_body_init[bi].append(ring)
        
        for bi, rings in rings_by_body_init.items():
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
    
        INSTANCE_FLOATS = 28
        MAX_BODIES = 1000
    
        mesh_lo_verts, mesh_lo_idx = create_icosphere_mesh(subdivisions=1)
        mesh_hi_verts, mesh_hi_idx = create_icosphere_mesh(subdivisions=4)
        mesh_ultra_verts, mesh_ultra_idx = create_icosphere_mesh(subdivisions=6)
    
        vbo_lo = ctx.buffer(mesh_lo_verts.tobytes())
        ibo_lo = ctx.buffer(mesh_lo_idx.tobytes())
        vbo_hi = ctx.buffer(mesh_hi_verts.tobytes())
        ibo_hi = ctx.buffer(mesh_hi_idx.tobytes())
        vbo_ultra = ctx.buffer(mesh_ultra_verts.tobytes())
        ibo_ultra = ctx.buffer(mesh_ultra_idx.tobytes())
    
        # Instead of vbo_instances, use SSBOs for GPU culling
        all_instances_buffer = ctx.buffer(reserve=MAX_BODIES * 112) # 28 floats * 4 bytes
        vis_lo_buffer = ctx.buffer(reserve=MAX_BODIES * 4)
        vis_hi_buffer = ctx.buffer(reserve=MAX_BODIES * 4)
        vis_ultra_buffer = ctx.buffer(reserve=MAX_BODIES * 4)
        draw_cmds_buffer = ctx.buffer(reserve=3 * 20) # 3 structs of 5 uints
        focused_mask_buffer = ctx.buffer(reserve=MAX_BODIES * 4)
        
        # We still need the base geometry vao, but we won't bind instances here
        # The vertex shader uses SSBO bindings directly based on gl_InstanceID
        vao_lo = ctx.vertex_array(
            prog_spheres,
            [(vbo_lo, '3f 3f', 'in_position', 'in_normal')],
            index_buffer=ibo_lo
        )
        vao_hi = ctx.vertex_array(
            prog_spheres,
            [(vbo_hi, '3f 3f', 'in_position', 'in_normal')],
            index_buffer=ibo_hi
        )
        vao_ultra = ctx.vertex_array(
            prog_spheres,
            [(vbo_ultra, '3f 3f', 'in_position', 'in_normal')],
            index_buffer=ibo_ultra
        )
    
        vao_atmo = ctx.vertex_array(
            prog_atmo,
            [(vbo_hi, '3f 12x', 'in_position')],
            index_buffer=ibo_hi
        )
        vao_atmo_lowres = ctx.vertex_array(
            prog_atmo_lowres,
            [(vbo_hi, '3f 12x', 'in_position')],
            index_buffer=ibo_hi
        )
    
        max_orbits = MAX_BODIES * 2
        orbit_ssbo = ctx.buffer(reserve=max_orbits * 160)
        orbit_ssbo.bind_to_storage_buffer(binding=0)
        orbit_ssbo_out = ctx.buffer(reserve=max_orbits * 4000 * 32)
        orbit_ssbo_out.bind_to_storage_buffer(binding=1)
        vao_gpu_orbits = ctx.vertex_array(prog_gpu_orbits, [])
    
        UBO_SIZE = 6304
        scene_ubo = ctx.buffer(reserve=UBO_SIZE)
        scene_ubo.bind_to_uniform_block(1)
        ubo_staging = np.zeros(UBO_SIZE // 4, dtype=np.float32)
        ubo_casters_int_view = ubo_staging[294:295].view(np.int32)
        ubo_num_stars_int_view = ubo_staging[32:33].view(np.int32)
    
        uniform_screen_height = prog_spheres['screen_height']
        uniform_fov_factor = prog_spheres['fov_factor']
    
        uniform_orbit_proj = prog_gpu_orbits['projection']
        uniform_orbit_view_rot = prog_gpu_orbits['view_rot']
        uniform_orbit_cam_pos = prog_gpu_orbits['u_cam_pos_double']
        uniform_orbit_far = prog_gpu_orbits['u_far']
        uniform_orbit_depth_C = prog_gpu_orbits['u_depth_C']
    
        vbo_ephem_orbits = ctx.buffer(reserve=1000 * 12)
        vao_ephem_orbits = ctx.vertex_array(prog_ephem_orbits, [(vbo_ephem_orbits, '3f', 'in_pos')])
    
        uniform_ring_centers = prog_spheres['u_ring_center']
        uniform_ring_normals = prog_spheres['u_ring_normal']
        uniform_ring_params = prog_spheres['u_ring_params']
        uniform_ring_colors = prog_spheres.get('u_ring_colors', None)
        uniform_ring_5colors = prog_spheres.get('u_ring_5colors', None)
        uniform_ring_coplanar_mask = prog_spheres.get('u_ring_coplanar_mask', None)
        uniform_caster_max_bend = prog_spheres.get('u_caster_max_bend', None)
        uniform_num_ring_planes = prog_spheres['u_num_ring_planes']
        if 'u_ring_gradients' in prog_spheres:
            prog_spheres['u_ring_gradients'].value = 0
        if 'u_ringshine_lut' in prog_spheres:
            prog_spheres['u_ringshine_lut'].value = 6
        if 'u_ringshine_cdf_lut' in prog_spheres:
            prog_spheres['u_ringshine_cdf_lut'].value = 7
        if 'u_ringshine_map' in prog_spheres:
            prog_spheres['u_ringshine_map'].value = 8
    
        star_idx = 0
        for i, b in enumerate(bodies_data):
            if b.get('type') == 'Star':
                star_idx = i
                break
        star_radius_au = visual_data[star_idx][3]
    
        body_radii = np.array([v[3] for v in visual_data], dtype='f4')
        body_colors = np.array([v[0:3] for v in visual_data], dtype='f4')
    
        u_ring_host_pos = prog_rings['u_host_planet_pos']
        u_ring_host_radius = prog_rings['u_host_planet_radius']
        u_ring_host_pole_obl = prog_rings['u_host_planet_pole_obl']
        u_ring_host_R_minor = prog_rings.get('u_host_planet_R_minor', None)
        u_ring_host_color = prog_rings.get('u_host_planet_color', None)
        u_ring_host_atmo = prog_rings.get('u_host_planet_atmo', None)
        u_ring_host_refractivity = prog_rings.get('u_host_planet_refractivity', None)
        u_ring_camera_pos = prog_rings['u_camera_pos']
        u_ring_body_offset = prog_rings['u_body_offset']
        u_ring_clip_mode = prog_rings['u_clip_mode']
        u_ring_caster_mask_lo_uni = prog_rings['u_caster_mask_lo']
        u_ring_caster_mask_hi_uni = prog_rings['u_caster_mask_hi']
        u_ring_planetshine_enabled = prog_rings.get('u_planetshine_enabled', None)
        u_ring_caster_max_bend = prog_rings.get('u_caster_max_bend', None)

        if 'u_ring_gradients' in prog_atmo:
            prog_atmo['u_ring_gradients'].value = 0
        u_atmo_quality_uniform = prog_atmo.get('u_atmo_quality', None)
        u_atmo_stochastic_uniform = prog_atmo.get('u_stochastic_noise', None)
        u_atmo_camera_pos = prog_atmo.get('u_camera_pos', None)
        u_atmo_num_ring_planes = prog_atmo.get('u_num_ring_planes', None)
        u_atmo_ring_centers = prog_atmo.get('u_ring_center', None)
        u_atmo_ring_normals = prog_atmo.get('u_ring_normal', None)
        u_atmo_ring_params = prog_atmo.get('u_ring_params', None)
        u_atmo_ring_coplanar_mask = prog_atmo.get('u_ring_coplanar_mask', None)
        u_atmo_clip_mode = prog_atmo.get('u_atmo_clip_mode', None)

        self.atmo_ssbo = ctx.buffer(reserve=1024)
        self.atmo_ssbo.bind_to_storage_buffer(binding=8)
        self.atmo_staging = np.zeros(256, dtype=np.float32)
        self.atmo_staging_int_view = self.atmo_staging.view(np.int32)


    
        G = 4.0 * math.pi**2 
    
        visual_arr = np.array(visual_data, dtype='f4')
        is_star_arr = np.zeros(num_bodies, dtype='f4')
        for i, b in enumerate(bodies_data):
            if b.get('type') == 'Star':
                is_star_arr[i] = 1.0
        self.tex_idx_arr = _build_tex_idx_arr(bodies_data, self.texture_slices)
        self.rot_period_arr, self.w0_arr, self.tidally_locked_arr, self.parent_idx_arr, self.pole_n_arr, self.tangent_arr, self.bitangent_arr = _build_rotation_props(bodies_data)
        self.tex_idx_arr_cmp = _build_tex_idx_arr(bodies_data_cmp, self.texture_slices)
        self.rot_period_arr_cmp, self.w0_arr_cmp, self.tidally_locked_arr_cmp, self.parent_idx_arr_cmp, self.pole_n_arr_cmp, self.tangent_arr_cmp, self.bitangent_arr_cmp = _build_rotation_props(bodies_data_cmp)

        
        non_star_mask = is_star_arr == 0.0
        non_star_indices = np.where(non_star_mask)[0]
        n_casters_fixed = len(non_star_indices)
    
        inst_data_lo = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
        inst_data_hi = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
        inst_data_ultra = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
        orbit_data_buf = np.zeros((max_orbits, 20), dtype='f8')
        subsys_pos_buf = np.zeros((num_bodies, 3), dtype='f8')
        subsys_vel_buf = np.zeros((num_bodies, 3), dtype='f8')
        subsys_mass_buf = np.zeros(num_bodies, dtype='f8')
        caster_data_buf = np.zeros((64, 4), dtype='f4')
        caster_poles_obl_buf = np.zeros((64, 4), dtype='f4')
        caster_colors_buf = np.zeros((64, 4), dtype='f4')
        caster_atmos_buf = np.zeros((64, 4), dtype='f4')
        caster_ozone_buf = np.zeros((64, 4), dtype='f4')
        caster_max_bend_buf = np.zeros(64, dtype='f4')
        ring_centers_buf = np.zeros((16, 3), dtype='f4')
        ring_normals_buf = np.zeros((16, 3), dtype='f4')
        ring_params_buf = np.zeros((16, 4), dtype='f4')
        ring_colors_buf = np.zeros((16, 3), dtype='f4')
        ring_5colors_buf = np.zeros((16, 5, 3), dtype='f4')
        ring_coplanar_mask_buf = np.zeros(16, dtype='u4')
        all_instances = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
        cull_mask_lo = np.zeros(num_bodies, dtype=np.uint32)
        cull_mask_hi = np.zeros(num_bodies, dtype=np.uint32)
        cull_ring_mask = np.zeros(num_bodies, dtype=np.uint32)
        cull_ring_caster_lo = np.zeros(16, dtype=np.uint32)
        cull_ring_caster_hi = np.zeros(16, dtype=np.uint32)
    
        pos_snap = np.zeros((num_bodies, 3), dtype='f8')
        vel_snap = np.zeros((num_bodies, 3), dtype='f8')
        mass_snap = np.zeros(num_bodies, dtype='f8')
        parent_snap = np.zeros(num_bodies, dtype=np.int32)
        tree_indices_snap = np.zeros(num_bodies, dtype=np.int32)
        tree_depths_snap = np.zeros(num_bodies, dtype=np.int32)
        focused_mask = np.zeros(num_bodies, dtype=np.bool_)
        visual_colors_f8 = np.ascontiguousarray(visual_arr[:, 0:3], dtype='f8')
        cached_children_map = {}
        cached_hierarchy_ver = -1
        switch_req_name = active_system_name
    
        last_render_time = time.perf_counter()
        show_orbits = self.camera.get("show_orbits", True)
        show_habitable_zone = self.camera.get("show_habitable_zone", False)
        orbit_fade_dir_idx = self.camera.get("orbit_fade_dir_idx", 0)
        orbit_fade_dir = 1.0 if orbit_fade_dir_idx == 0 else -1.0
        orbit_min_alpha = self.camera.get("orbit_min_alpha", 0.3)
        atmo_quality = min(self.camera.get("atmo_quality", 1), 2)
        now_dt = datetime.datetime.now()
        jump_date = [now_dt.year, now_dt.month, now_dt.day, now_dt.hour, now_dt.minute, now_dt.second]
        scrub_index = [0]
    
        last_orbit_pos_snap = None
        last_orbit_pos_snap_cmp = None
        last_comparison_offset_au = None
        last_cam_origin = None
        
        self.u_planet_textures_obj = None
        self.u_planet_normal_textures_obj = None
        self.u_planet_specular_textures_obj = None
        self.u_planet_cloud_textures_obj = None
        
        # Anisotropic filtering is the key to eliminating spherical-mapping moire:
        # equirectangular UVs compress longitude toward the poles, so the isotropic
        # mip selection from textureGrad under-samples along one axis and produces
        # catastrophic aliasing/moire across the whole planet when zoomed in. Pull
        # the device maximum once and apply it to every mipmapped surface texture.
        try:
            max_aniso = float(ctx.max_anisotropy or 1.0)
        except Exception:
            max_aniso = 1.0
        # Use a high but slightly conservative anisotropy cap (16x is the typical
        # device max; values above the max are clamped by the driver anyway).
        aniso_value = max(1.0, min(max_aniso, 16.0))

        # Dictionary holding resident ModernGL Texture objects:
        # idx -> {'diffuse': tex, 'normal': tex, 'specular': tex, 'clouds': tex}
        self.gpu_body_textures = {}
        self.active_res_level = {} # idx -> 'low' or 'high'

        # SSBO layout: struct BodyTextures { uvec2 diffuse; uvec2 normal; uvec2 specular; uvec2 clouds; };
        # Each uvec2 is 8 bytes (two 32-bit uints). 4 uvec2s = 32 bytes per body texture slot.
        MAX_TEXTURED_BODIES = max(64, len(self.texture_slices) + 16)
        self.body_textures_ssbo_data = np.zeros((MAX_TEXTURED_BODIES, 4, 2), dtype=np.uint32)

        def _upload_texture_obj(size, raw_bytes, components):
            tex = ctx.texture(size, components, raw_bytes, dtype='f1')
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.build_mipmaps()
            tex.anisotropy = aniso_value
            return tex

        # Initialize all fallback textures on the GPU and get their 64-bit bindless handles
        for idx, entry in self.texture_streamer.fallback_data.items():
            slot = idx - 1
            tex_dict = {}
            for map_name, (sz, raw_b, comp) in entry.items():
                tex = _upload_texture_obj(sz, raw_b, comp)
                tex_dict[map_name] = tex
                handle_64 = int(tex.get_handle(resident=True))
                # Split 64-bit uint into two 32-bit uints for GLSL uvec2
                lo = handle_64 & 0xFFFFFFFF
                hi = (handle_64 >> 32) & 0xFFFFFFFF
                map_offset = {'diffuse': 0, 'normal': 1, 'specular': 2, 'clouds': 3}[map_name]
                self.body_textures_ssbo_data[slot, map_offset] = [lo, hi]
            self.gpu_body_textures[idx] = tex_dict
            self.active_res_level[idx] = 'low'

        self.body_textures_ssbo = ctx.buffer(self.body_textures_ssbo_data.tobytes())
        self.body_textures_ssbo.bind_to_storage_buffer(binding=10)

        # Convert ring PIL images into ModernGL textures
        for name_lower, img in list(self.ring_gl_textures_front.items()):
            try:
                tex = ctx.texture(img.size, 4, img.tobytes())
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                tex.repeat_x = False
                tex.repeat_y = False
                tex.build_mipmaps()
                tex.anisotropy = aniso_value
                self.ring_gl_textures_front[name_lower] = tex
            except Exception as e:
                print(f"Failed to compile OpenGL front texture for ring {name_lower}: {e}")

        for name_lower, img in list(self.ring_gl_textures_back.items()):
            try:
                tex = ctx.texture(img.size, 4, img.tobytes())
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                tex.repeat_x = False
                tex.repeat_y = False
                tex.build_mipmaps()
                tex.anisotropy = aniso_value
                self.ring_gl_textures_back[name_lower] = tex
            except Exception as e:
                print(f"Failed to compile OpenGL back texture for ring {name_lower}: {e}")

        n_orbits = 0
        n_orbits_hi = 0
        n_orbits_med = 0
        n_orbits_low = 0
        orbits_hi = np.zeros((0, 20), dtype='f8')
        orbits_med = np.zeros((0, 20), dtype='f8')
        orbits_low = np.zeros((0, 20), dtype='f8')
        orbits_hi_cmp = np.zeros((0, 20), dtype='f8')
        orbits_med_cmp = np.zeros((0, 20), dtype='f8')
        orbits_low_cmp = np.zeros((0, 20), dtype='f8')
    
        def build_atmo_lut(atmo, mass_sm, is_cmp=False):
            if 'lut_tex' in atmo and atmo['lut_tex']:
                atmo['lut_tex'].release()
            if 'lut_multi_scatter' in atmo and atmo['lut_multi_scatter']:
                atmo['lut_multi_scatter'].release()

            prev_fbo = ctx.fbo
            lut_tex = ctx.texture((256, 256), 4, dtype='f4')
            fbo = ctx.framebuffer(color_attachments=[lut_tex])
            
            multi_scatter_tex = ctx.texture((32, 32), 4, dtype='f4')
            ms_fbo = ctx.framebuffer(color_attachments=[multi_scatter_tex])
            
            mass_kg = mass_sm * 1.98847e30
            radius_km = atmo['planet_radius_km']
            props, _, _ = get_cached_atmosphere_properties(atmo, mass_sm)
            
            atmo_h_km = float(atmo.get('height', props['atmo_height_km']))
            atmo['atmo_radius_km'] = atmo['planet_radius_km'] + atmo_h_km
            atmo['atmo_radius_au'] = atmo['atmo_radius_km'] / 149597870.7
            
            beta_rayleigh = props['beta_rayleigh']
            beta_mie = compute_mie_coefficients(atmo.get('beta_mie', 2.0e-6), atmo.get('mie_angstrom', None))
            beta_abs_mixed = props['beta_abs_mixed']
            beta_abs_layered = props['beta_abs_layered']
            mie_albedo = np.asarray(atmo.get('mie_albedo', np.array([1.0, 1.0, 1.0], dtype=np.float32)), dtype=np.float32)
            
            bi = atmo['body_idx']
            if is_cmp and hasattr(self, 'visual_arr_cmp') and self.visual_arr_cmp is not None and len(self.visual_arr_cmp) > bi:
                v_color = self.visual_arr_cmp[bi, 0:3]
                b_list = self.bodies_data_cmp if hasattr(self, 'bodies_data_cmp') else bodies_data
            else:
                v_color = visual_arr[bi, 0:3]
                b_list = bodies_data
            
            b_info = b_list[bi] if (b_list is not None and bi < len(b_list)) else {}
            albedo, _, _ = compute_surface_albedo(b_info, self.texture_mean_colors, v_color)
            
            self.prog_atmo_lut['u_planet_radius_km'].value = float(atmo['planet_radius_km'])
            self.prog_atmo_lut['u_atmo_radius_km'].value = float(atmo['atmo_radius_km'])
            self.prog_atmo_lut['u_h_rayleigh'].value = float(props['scale_height_km'])
            self.prog_atmo_lut['u_h_mie'].value = float(atmo.get('h_mie', 1.2))
            self.prog_atmo_lut['u_beta_rayleigh'].value = tuple(beta_rayleigh)
            self.prog_atmo_lut['u_beta_mie'].value = tuple(beta_mie)
            self.prog_atmo_lut['u_beta_abs_mixed'].value = tuple(beta_abs_mixed)
            self.prog_atmo_lut['u_beta_abs_layered'].value = tuple(beta_abs_layered)
            if 'u_mie_albedo' in self.prog_atmo_lut:
                self.prog_atmo_lut['u_mie_albedo'].value = tuple(mie_albedo)
            if 'u_ozone_peak_km' in self.prog_atmo_lut:
                self.prog_atmo_lut['u_ozone_peak_km'].value = float(props.get('ozone_peak_km', 25.0))
            if 'u_ozone_width_km' in self.prog_atmo_lut:
                self.prog_atmo_lut['u_ozone_width_km'].value = float(props.get('ozone_width_km', 8.0))
            
            fbo.use()
            self.lut_vao.render(moderngl.TRIANGLE_STRIP)
            
            lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            lut_tex.repeat_x = False
            lut_tex.repeat_y = False
            
            lut_tex.use(location=1)
            
            self.prog_multi_scatter_lut['u_planet_radius_km'].value = float(atmo['planet_radius_km'])
            self.prog_multi_scatter_lut['u_atmo_radius_km'].value = float(atmo['atmo_radius_km'])
            self.prog_multi_scatter_lut['u_h_rayleigh'].value = float(props['scale_height_km'])
            self.prog_multi_scatter_lut['u_h_mie'].value = float(atmo.get('h_mie', 1.2))
            self.prog_multi_scatter_lut['u_beta_rayleigh'].value = tuple(beta_rayleigh)
            self.prog_multi_scatter_lut['u_beta_mie'].value = tuple(beta_mie)
            self.prog_multi_scatter_lut['u_beta_abs_mixed'].value = tuple(beta_abs_mixed)
            self.prog_multi_scatter_lut['u_beta_abs_layered'].value = tuple(beta_abs_layered)
            if 'u_mie_albedo' in self.prog_multi_scatter_lut:
                self.prog_multi_scatter_lut['u_mie_albedo'].value = tuple(mie_albedo)
            if 'u_ozone_peak_km' in self.prog_multi_scatter_lut:
                self.prog_multi_scatter_lut['u_ozone_peak_km'].value = float(props.get('ozone_peak_km', 25.0))
            if 'u_ozone_width_km' in self.prog_multi_scatter_lut:
                self.prog_multi_scatter_lut['u_ozone_width_km'].value = float(props.get('ozone_width_km', 8.0))
            if 'u_mie_g' in self.prog_multi_scatter_lut:
                self.prog_multi_scatter_lut['u_mie_g'].value = float(atmo.get('mie_g', 0.8))
            if 'u_ground_albedo' in self.prog_multi_scatter_lut:
                self.prog_multi_scatter_lut['u_ground_albedo'].value = tuple(albedo)
            self.prog_multi_scatter_lut['u_transmittance_lut'].value = 1
            
            ms_fbo.use()
            self.multi_scatter_vao.render(moderngl.TRIANGLE_STRIP)
            
            if prev_fbo:
                prev_fbo.use()
            else:
                ctx.screen.use()
            
            fbo.release()
            ms_fbo.release()
            
            multi_scatter_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            multi_scatter_tex.repeat_x = False
            multi_scatter_tex.repeat_y = False
            
            atmo['lut_tex'] = lut_tex
            atmo['lut_multi_scatter'] = multi_scatter_tex
            atmo['lut_mass'] = mass_sm
            
        print("[Render Loop] Entering main render loop")
        self.frame_counter = 0
        while not glfw.window_should_close(window):
            self.frame_counter += 1
            _perf_gpu_flush()
            # ── System Switch — render-side rebuild ──
            with self.shared_state["lock"]:
                switch_complete = self.shared_state.get("system_switch_complete", False)
            if switch_complete:
                with self.shared_state["lock"]:
                    new_info = self.shared_state["system_new_bundle"]
                    saved_snap = self.shared_state["system_snapshot_out"]
                    is_ephem_enter = self.shared_state.get("ephemeris_enter", False)
                    is_ephem_exit = self.shared_state.get("ephemeris_exit", False)
                    to_keplerian = self.shared_state.get("to_keplerian", False)
                    self.shared_state["system_switch_complete"] = False
                    self.shared_state["system_new_bundle"] = None
                    self.shared_state["system_snapshot_out"] = None
                    self.shared_state["ephemeris_enter"] = False
                    self.shared_state["ephemeris_exit"] = False
                    self.shared_state["to_keplerian"] = False
                    self._ephem_mapping_ver = None
    
                # Store snapshot of old system in SystemManager
                old_name = active_system_name
                if saved_snap is not None:
                    sys_mgr.store_snapshot(old_name, saved_snap)
    
                # Update active system name
                if is_ephem_enter:
                    ephemeris_mode_active = True
                    active_system_name = "Ephemeris Mode"
                    self.shared_state["ephemeris_mode"] = True
                    keplerian_mode_active = False
                    self.shared_state["keplerian_mode"] = False
                elif is_ephem_exit:
                    ephemeris_mode_active = False
                    active_system_name = switch_req_name
                    self.shared_state["ephemeris_mode"] = False
                    if to_keplerian:
                        keplerian_mode_active = True
                        self.shared_state["keplerian_mode"] = True
                        self.shared_state["keplerian_reextract"] = True
                else:
                    active_system_name = switch_req_name
    
                # Rebuild ALL render-side state from new system info
                bodies_data = new_info["bodies_data"]
                visual_data = new_info["visual_data"]
                atmo_bodies = new_info["atmo_bodies"]
                ring_bodies = new_info["ring_bodies"]
                num_bodies = new_info["num_bodies"]
                star_idx = new_info["star_idx"]
    
                visual_arr = np.array(visual_data, dtype='f4')
                body_radii = np.array([v[3] for v in visual_data], dtype='f4')
                body_colors = np.array([v[0:3] for v in visual_data], dtype='f4')
                is_star_arr = np.zeros(num_bodies, dtype='f4')
                for i, b in enumerate(bodies_data):
                    if b.get('type') == 'Star':
                        is_star_arr[i] = 1.0
                # Rebuild texture/rotation arrays for the new system bodies.
                self.tex_idx_arr = _build_tex_idx_arr(bodies_data, self.texture_slices)
                self.rot_period_arr, self.w0_arr, self.tidally_locked_arr, self.parent_idx_arr, self.pole_n_arr, self.tangent_arr, self.bitangent_arr = _build_rotation_props(bodies_data)

                non_star_mask = is_star_arr == 0.0
                non_star_indices = np.where(non_star_mask)[0]
                n_casters_fixed = len(non_star_indices)
                star_radius_au = visual_data[star_idx][3] if star_idx < len(visual_data) else SOLAR_RADII_TO_AU
    
                pos_snap = np.zeros((num_bodies, 3), dtype='f8')
                vel_snap = np.zeros((num_bodies, 3), dtype='f8')
                mass_snap = np.zeros(num_bodies, dtype='f8')
                parent_snap = np.zeros(num_bodies, dtype=np.int32)
                tree_indices_snap = np.zeros(num_bodies, dtype=np.int32)
                tree_depths_snap = np.zeros(num_bodies, dtype=np.int32)
    
                all_instances = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
                inst_data_lo = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
                inst_data_hi = np.zeros((num_bodies, INSTANCE_FLOATS), dtype='f4')
                focused_mask = np.zeros(num_bodies, dtype=np.bool_)
                visual_colors_f8 = np.ascontiguousarray(visual_arr[:, 0:3], dtype='f8')
    
                max_orbits = MAX_BODIES * 2
                orbit_data_buf = np.zeros((max_orbits, 20), dtype='f8')
                subsys_pos_buf = np.zeros((num_bodies, 3), dtype='f8')
                subsys_vel_buf = np.zeros((num_bodies, 3), dtype='f8')
                subsys_mass_buf = np.zeros(num_bodies, dtype='f8')
                caster_data_buf = np.zeros((64, 4), dtype='f4')
                caster_poles_obl_buf = np.zeros((64, 4), dtype='f4')
                caster_colors_buf = np.zeros((64, 4), dtype='f4')
                caster_atmos_buf = np.zeros((64, 4), dtype='f4')
                caster_ozone_buf = np.zeros((64, 4), dtype='f4')
                caster_max_bend_buf = np.zeros(64, dtype='f4')
                ring_centers_buf = np.zeros((16, 3), dtype='f4')
                ring_normals_buf = np.zeros((16, 3), dtype='f4')
                ring_params_buf = np.zeros((16, 4), dtype='f4')
                ring_colors_buf = np.zeros((16, 3), dtype='f4')
                ring_coplanar_mask_buf = np.zeros(16, dtype='u4')
                cull_mask_lo = np.zeros(num_bodies, dtype=np.uint32)
                cull_mask_hi = np.zeros(num_bodies, dtype=np.uint32)
                cull_ring_mask = np.zeros(num_bodies, dtype=np.uint32)
                cull_ring_caster_lo = np.zeros(16, dtype=np.uint32)
                cull_ring_caster_hi = np.zeros(16, dtype=np.uint32)
    
                # Release old ring GL objects and rebuild
                for g in ring_render_groups:
                    g['vao'].release()
                    if 'vbo' in g: g['vbo'].release()
                    if 'ibo' in g: g['ibo'].release()
                ring_render_groups = []
                ring_precomputed = []
    
                # Rebuild ring precomputed data for new system
                RING_SEGMENTS_SW = 128
                for body_idx_r, rings_data_r, pole_render_r, body_radius_au_r in ring_bodies:
                    pole_n_r = pole_render_r / np.linalg.norm(pole_render_r)
                    ref_r = np.array([0., 0., 1.])
                    tangent_r = np.cross(pole_n_r, ref_r)
                    if np.linalg.norm(tangent_r) < 1e-10:
                        ref_r = np.array([1., 0., 0.])
                        tangent_r = np.cross(pole_n_r, ref_r)
                    tangent_r /= np.linalg.norm(tangent_r)
                    bitangent_r = np.cross(pole_n_r, tangent_r)
                    R_ring_sw = np.column_stack([tangent_r, pole_n_r, bitangent_r])
                    b_data_sw = bodies_data[body_idx_r]
                    name_lower_sw = b_data_sw['name'].lower()
                    
                    if name_lower_sw in self.ring_textures_front and len(rings_data_r) > 0:
                        tex_inner_f = b_data_sw.get('ring_texture_inner')
                        tex_outer_f = b_data_sw.get('ring_texture_outer')
                        
                        if tex_inner_f is not None and tex_outer_f is not None:
                            tex_min_inner = float(tex_inner_f)
                            tex_max_outer = float(tex_outer_f)
                            procedural_segments = [seg for seg in rings_data_r if seg['outer'] <= tex_min_inner + 1e-4 or seg['inner'] >= tex_max_outer - 1e-4]
                        else:
                            tex_min_inner = min(seg['inner'] for seg in rings_data_r)
                            tex_max_outer = max(seg['outer'] for seg in rings_data_r)
                            procedural_segments = []
                            
                        inner_r = tex_min_inner * body_radius_au_r
                        outer_r = tex_max_outer * body_radius_au_r
                        
                        textured_segs = [seg for seg in rings_data_r if seg['inner'] >= tex_min_inner - 1e-4 and seg['outer'] <= tex_max_outer + 1e-4]
                        if not textured_segs:
                            textured_segs = rings_data_r
                            
                        total_w = 0.0
                        sum_scatter = 0.0
                        sum_asymmetry = 0.0
                        sum_backscatter = 0.0
                        
                        for seg in textured_segs:
                            w = seg.get('opacity', 1.0) * (seg['outer'] - seg['inner'])
                            if w < 1e-9: w = 1e-9
                            total_w += w
                            sum_scatter += seg.get('scatter', 2.5) * w
                            sum_asymmetry += seg.get('asymmetry', 0.7) * w
                            sum_backscatter += seg.get('backscatter', -0.3) * w
                            
                        r_color = (1.0, 1.0, 1.0)
                        r_opacity = 1.0
                        if total_w > 0.0:
                            r_scatter = sum_scatter / total_w
                            r_asymmetry = sum_asymmetry / total_w
                            r_backscatter = sum_backscatter / total_w
                        else:
                            first = textured_segs[0]
                            r_scatter = first.get('scatter', 2.5)
                            r_asymmetry = first.get('asymmetry', 0.7)
                            r_backscatter = first.get('backscatter', -0.3)
                        
                        img_data = np.frombuffer(self.ring_textures_front[name_lower_sw].tobytes(), dtype=np.uint8).astype('f4') / 255.0
                        img_data = img_data.reshape(4096, 4)
                        tex_sampled_sw = img_data
                        
                        shadow_grad_r = generate_ring_shadow_grad(sorted_gradient=[], tex_sampled=tex_sampled_sw)
                        colors5 = compute_5_ring_colors(tex_sampled=tex_sampled_sw, raw_color=r_color)
                        
                        ring_precomputed.append({
                            'body_idx': body_idx_r, 'pole': pole_n_r.astype('f4'), 'inner_r': inner_r, 'outer_r': outer_r,
                            'opacity': r_opacity, 'scatter': r_scatter, 'asymmetry': r_asymmetry, 'backscatter': r_backscatter,
                            'shadow_grad': shadow_grad_r, 'raw_color': r_color, 'gradient': [],
                            'tex_sampled': tex_sampled_sw, '5colors': colors5, 'is_textured': True, 'row_idx': len(ring_precomputed)
                        })
                        
                        for ring_seg in procedural_segments:
                            p_inner_r = ring_seg['inner'] * body_radius_au_r
                            p_outer_r = ring_seg['outer'] * body_radius_au_r
                            p_r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
                            p_r_opacity = ring_seg.get('opacity', 1.0)
                            p_r_scatter = ring_seg.get('scatter', 2.5)
                            p_r_asymmetry = ring_seg.get('asymmetry', 0.7)
                            p_r_backscatter = ring_seg.get('backscatter', -0.3)
                            
                            sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
                            shadow_grad_r = generate_ring_shadow_grad(sorted_gradient, tex_sampled=None)
                            colors5 = compute_5_ring_colors(tex_sampled=None, raw_color=p_r_color, gradient=sorted_gradient)
                            
                            ring_precomputed.append({
                                'body_idx': body_idx_r, 'pole': pole_n_r.astype('f4'), 'inner_r': p_inner_r, 'outer_r': p_outer_r,
                                'opacity': p_r_opacity, 'scatter': p_r_scatter, 'asymmetry': p_r_asymmetry, 'backscatter': p_r_backscatter,
                                'shadow_grad': shadow_grad_r, 'raw_color': p_r_color, 'gradient': sorted_gradient,
                                'tex_sampled': None, '5colors': colors5, 'is_textured': False, 'row_idx': len(ring_precomputed)
                            })
                    else:
                        for ring_seg in rings_data_r:
                            inner_r = ring_seg['inner'] * body_radius_au_r
                            outer_r = ring_seg['outer'] * body_radius_au_r
                            r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
                            r_opacity = ring_seg.get('opacity', 1.0)
                            r_scatter = ring_seg.get('scatter', 2.5)
                            r_asymmetry = ring_seg.get('asymmetry', 0.7)
                            r_backscatter = ring_seg.get('backscatter', -0.3)
                            sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
                            shadow_grad_r = generate_ring_shadow_grad(sorted_gradient, tex_sampled=None)
                            colors5 = compute_5_ring_colors(tex_sampled=None, raw_color=r_color, gradient=sorted_gradient)
                            ring_precomputed.append({
                                'body_idx': body_idx_r, 'pole': pole_n_r.astype('f4'), 'inner_r': inner_r, 'outer_r': outer_r,
                                'opacity': r_opacity, 'scatter': r_scatter, 'asymmetry': r_asymmetry, 'backscatter': r_backscatter,
                                'shadow_grad': shadow_grad_r, 'raw_color': r_color, 'gradient': sorted_gradient,
                                'tex_sampled': None, '5colors': colors5, 'is_textured': False, 'row_idx': len(ring_precomputed),
                            })
    
                self.body_ring_indices = rebuild_ring_gradients_atlas(ring_precomputed, ring_gradient_tex)

    
                rings_by_body_sw = {}
                for ring in ring_precomputed:
                    bi_r = ring['body_idx']
                    rings_by_body_sw.setdefault(bi_r, []).append(ring)
                for bi_r, rings_r in rings_by_body_sw.items():
                    min_r = min([r['inner_r'] for r in rings_r])
                    max_r = max([r['outer_r'] for r in rings_r])
                    pole_n = rings_r[0]['pole']
                    
                    verts, normals, indices = generate_ring_geometry(pole_n, min_r, max_r)
                    n_v = len(verts)
                    ring_packed = np.zeros((n_v, 6), dtype='f4')
                    ring_packed[:, 0:3] = verts
                    ring_packed[:, 3:6] = normals
                    
                    all_v_sw = ring_packed
                    all_i_sw = indices
                    ring_vbo_sw = ctx.buffer(all_v_sw.tobytes())
                    ring_ibo_sw = ctx.buffer(all_i_sw.tobytes())
                    ring_vao_sw = ctx.vertex_array(
                        prog_rings,
                        [(ring_vbo_sw, '3f 3f', 'in_position', 'in_normal')],
                        index_buffer=ring_ibo_sw)
                    ring_render_groups.append({'body_idx': bi_r, 'vao': ring_vao_sw, 'vbo': ring_vbo_sw, 'ibo': ring_ibo_sw, 'num_indices': len(all_i_sw)})
    
                # Reset self.camera and UI state
                self.camera["tracking_idx"] = 0
                self.camera["tracking_is_cmp"] = False
                self.camera["inspected_idx"] = None
                self.camera["inspected_is_cmp"] = False
                self.camera["inspect_bary"] = False
                self.camera["edit_mode"] = False
                self.camera["target"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                self.camera["cam_pos_rel"] = np.array([0.0, -19.02, 40.78], dtype='f8')
                self.camera["cam_pos_rel_prev"] = self.camera["cam_pos_rel"].copy()
                self.camera["cam_vel"] = np.zeros(3, dtype='f8')
                self.camera["cam_look"] = "aim"
                self.camera["keys"] = {}
                self.camera["approach_delta"] = 0.0
                self.camera["yaw"] = self.camera["yaw_actual"] = -90.0
                self.camera["pitch"] = self.camera["pitch_actual"] = 25.0
                self.camera["roll"] = self.camera["roll_actual"] = 0.0
    
                cached_hierarchy_ver = -1
                last_orbit_pos_snap = None
                last_cam_origin = None
                n_orbits = 0
                scrub_index[0] = 0
    
                print(f"[System] Switched to '{active_system_name}' ({num_bodies} bodies)")
    
            with self.shared_state_cmp["lock"]:
                switch_complete_cmp = self.shared_state_cmp.get("system_switch_complete", False)
            if switch_complete_cmp:
                with self.shared_state_cmp["lock"]:
                    new_info_cmp = self.shared_state_cmp["system_new_bundle"]
                    self.shared_state_cmp["system_switch_complete"] = False
                    self.shared_state_cmp["system_new_bundle"] = None
                    self.shared_state_cmp["system_snapshot_out"] = None
                
                self.bodies_data_cmp = new_info_cmp["bodies_data"]
                self.visual_data_cmp = new_info_cmp["visual_data"]
                self.atmo_bodies_cmp = new_info_cmp["atmo_bodies"]
                self.ring_bodies_cmp = new_info_cmp["ring_bodies"]
                self.num_bodies_cmp = new_info_cmp["num_bodies"]
                self.star_idx_cmp = new_info_cmp["star_idx"]
                
                self.visual_arr_cmp = np.array(self.visual_data_cmp, dtype='f4')
                last_orbit_pos_snap_cmp = None
                self.body_radii_cmp = np.array([v[3] for v in self.visual_data_cmp], dtype='f4')
                self.body_colors_cmp = np.array([v[0:3] for v in self.visual_data_cmp], dtype='f4')
                self.is_star_arr_cmp = np.zeros(self.num_bodies_cmp, dtype='f4')
                for i, b in enumerate(self.bodies_data_cmp):
                    if b.get('type') == 'Star':
                        self.is_star_arr_cmp[i] = 1.0
                # Rebuild comparison texture/rotation arrays for the new system bodies.
                self.tex_idx_arr_cmp = _build_tex_idx_arr(self.bodies_data_cmp, self.texture_slices)
                self.rot_period_arr_cmp, self.w0_arr_cmp, self.tidally_locked_arr_cmp, self.parent_idx_arr_cmp, self.pole_n_arr_cmp, self.tangent_arr_cmp, self.bitangent_arr_cmp = _build_rotation_props(self.bodies_data_cmp)

                
                last_orbit_pos_snap_cmp = None
                last_comparison_offset_au = None
                self.pos_snap_cmp = np.zeros((self.num_bodies_cmp, 3), dtype='f8')
                self.vel_snap_cmp = np.zeros((self.num_bodies_cmp, 3), dtype='f8')
                self.mass_snap_cmp = np.zeros(self.num_bodies_cmp, dtype='f8')
                self.parent_snap_cmp = np.zeros(self.num_bodies_cmp, dtype=np.int32)
                self.tree_indices_snap_cmp = np.zeros(self.num_bodies_cmp, dtype=np.int32)
                self.tree_depths_snap_cmp = np.zeros(self.num_bodies_cmp, dtype=np.int32)
                
                self.visual_colors_f8_cmp = np.ascontiguousarray(self.visual_arr_cmp[:, 0:3], dtype='f8')
                self.orbit_data_buf_cmp = np.zeros((max_orbits, 20), dtype='f8')
                self.subsys_pos_buf_cmp = np.zeros((self.num_bodies_cmp, 3), dtype='f8')
                self.subsys_vel_buf_cmp = np.zeros((self.num_bodies_cmp, 3), dtype='f8')
                self.subsys_mass_buf_cmp = np.zeros(self.num_bodies_cmp, dtype='f8')
                self.n_orbits_cmp = 0
                
                # Release old ring GL objects and rebuild for comparison
                for g in self.ring_render_groups_cmp:
                    g['vao'].release()
                    if 'vbo' in g: g['vbo'].release()
                    if 'ibo' in g: g['ibo'].release()
                self.ring_render_groups_cmp = []
                self.ring_precomputed_cmp = []
                
                for body_idx_r, rings_data_r, pole_render_r, body_radius_au_r in self.ring_bodies_cmp:
                    pole_n_r = pole_render_r / np.linalg.norm(pole_render_r)
                    ref_r = np.array([0., 0., 1.])
                    tangent_r = np.cross(pole_n_r, ref_r)
                    if np.linalg.norm(tangent_r) < 1e-10:
                        ref_r = np.array([1., 0., 0.])
                        tangent_r = np.cross(pole_n_r, ref_r)
                    tangent_r /= np.linalg.norm(tangent_r)
                    bitangent_r = np.cross(pole_n_r, tangent_r)
                    for ring_seg in rings_data_r:
                        inner_r = ring_seg['inner'] * body_radius_au_r
                        outer_r = ring_seg['outer'] * body_radius_au_r
                        r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
                        r_opacity = ring_seg.get('opacity', 1.0)
                        r_scatter = ring_seg.get('scatter', 2.5)
                        r_asymmetry = ring_seg.get('asymmetry', 0.7)
                        r_backscatter = ring_seg.get('backscatter', -0.3)
                        sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
                        name_lower_sw = self.bodies_data_cmp[body_idx_r]['name'].lower()
                        tex_sampled_sw = None
                        if name_lower_sw in self.ring_textures:
                            img_data = np.frombuffer(self.ring_textures[name_lower_sw].tobytes(), dtype=np.uint8).astype('f4') / 255.0
                            tex_sampled_sw = img_data.reshape(4096, 4)
                        shadow_grad_r = generate_ring_shadow_grad(sorted_gradient, tex_sampled=tex_sampled_sw)
                        self.ring_precomputed_cmp.append({
                            'body_idx': body_idx_r, 'pole': pole_n_r.astype('f4'), 'inner_r': inner_r, 'outer_r': outer_r,
                            'opacity': r_opacity, 'scatter': r_scatter, 'asymmetry': r_asymmetry, 'backscatter': r_backscatter,
                            'shadow_grad': shadow_grad_r, 'raw_color': r_color, 'gradient': sorted_gradient,
                            'tex_sampled': tex_sampled_sw,
                            'row_idx': len(self.ring_precomputed_cmp),
                        })
                        
                rings_by_body_cmp = {}
                for ring in self.ring_precomputed_cmp:
                    bi_r = ring['body_idx']
                    rings_by_body_cmp.setdefault(bi_r, []).append(ring)
                for bi_r, rings_r in rings_by_body_cmp.items():
                    all_ring_verts_cmp = []
                    min_r = min([r['inner_r'] for r in rings_r])
                    max_r = max([r['outer_r'] for r in rings_r])
                    pole_n = rings_r[0]['pole']
                    
                    verts, normals, indices = generate_ring_geometry(pole_n, min_r, max_r)
                    n_v = len(verts)
                    ring_packed = np.zeros((n_v, 6), dtype='f4')
                    ring_packed[:, 0:3] = verts
                    ring_packed[:, 3:6] = normals
                    
                    all_v_cmp = ring_packed
                    all_i_cmp = indices
                    ring_vbo_cmp = ctx.buffer(all_v_cmp.tobytes())
                    ring_ibo_cmp = ctx.buffer(all_i_cmp.tobytes())
                    ring_vao_cmp = ctx.vertex_array(
                        prog_rings,
                        [(ring_vbo_cmp, '3f 3f', 'in_position', 'in_normal')],
                        index_buffer=ring_ibo_cmp)
                    self.ring_render_groups_cmp.append({'body_idx': bi_r, 'vao': ring_vao_cmp, 'vbo': ring_vbo_cmp, 'ibo': ring_ibo_cmp, 'num_indices': len(all_i_cmp)})
                    
                print(f"[System] Comparison switched to '{self.comparison_system_name}' ({self.num_bodies_cmp} bodies)")
    
            with self.shared_state["lock"]:
                if self.shared_state.get("rebuild_flag", False):
    
                    for op in self.shared_state["crud_completed"]:
                        if op["action"] == "CREATE":
                            name = op.get("name", "New Body")
                            radius = op.get("radius", 6000.0)
                            mass = op.get("mass", 3e-6)
                            color = op.get("color", [1.0, 1.0, 1.0])
                            btype = op.get("type", "Moon")
                            parent_idx = op["parent_idx"]
                            pos = op["pos"]
                            vel = op["vel"]
                            
                            bodies_data.append({
                                "name": name,
                                "type": btype,
                                "color": f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}",
                                "r": float(radius / 696340.0),
                                "m": float(mass),
                                "parentId": bodies_data[parent_idx]["name"],
                                "sv": {
                                    "x": float(pos[0] - pos_snap[parent_idx][0]),
                                    "y": float(pos[1] - pos_snap[parent_idx][1]),
                                    "z": float(pos[2] - pos_snap[parent_idx][2]),
                                    "vx": float(vel[0] - vel_snap[parent_idx][0]),
                                    "vy": float(vel[1] - vel_snap[parent_idx][1]),
                                    "vz": float(vel[2] - vel_snap[parent_idx][2])
                                }
                            })
                            
                            pos_snap = np.vstack([pos_snap, pos])
                            vel_snap = np.vstack([vel_snap, vel])
                            mass_snap = np.append(mass_snap, mass)
                            parent_snap = np.append(parent_snap, np.int32(parent_idx))
                            
                            tree_indices_snap = np.append(tree_indices_snap, np.int32(0))
                            tree_depths_snap = np.append(tree_depths_snap, np.int32(0))
                            
                            pos_snap_render = np.vstack([pos_snap_render, pos])
                            vel_snap_render = np.vstack([vel_snap_render, vel])
                            
                            r_au = radius / 149597870.7
                            body_radii = np.append(body_radii, r_au)
                            
                            new_inst = np.zeros(INSTANCE_FLOATS, dtype='f4')
                            num_bodies += 1
                            all_instances = np.vstack([all_instances, new_inst])
                            cull_mask_lo = np.append(cull_mask_lo, np.uint32(0))
                            cull_mask_hi = np.append(cull_mask_hi, np.uint32(0))
                            cull_ring_mask = np.append(cull_ring_mask, np.uint32(0))
                            
                            subsys_pos_buf = np.vstack([subsys_pos_buf, pos])
                            subsys_vel_buf = np.vstack([subsys_vel_buf, vel])
                            subsys_mass_buf = np.append(subsys_mass_buf, mass)
                            
                            min_px = 1.0
                            new_vis = np.array([color[0], color[1], color[2], r_au, min_px, 0, 1, 0, 0, color[0], color[1], color[2], 0.0, 0.0], dtype='f4')
                            visual_arr = np.vstack([visual_arr, new_vis])
                            visual_colors_f8 = np.vstack([visual_colors_f8, np.array(color, dtype='f8')])
                            body_colors = np.vstack([body_colors, np.array(color, dtype='f4')])
                            is_star_val = 1.0 if btype == "Star" else 0.0
                            is_star_arr = np.append(is_star_arr, is_star_val)
                            visual_data.append([color[0], color[1], color[2], r_au, min_px, 0, 1, 0, 0, color[0], color[1], color[2], 0.0, 0.0])
                            
                            inst_data_lo = np.vstack([inst_data_lo, np.zeros(INSTANCE_FLOATS, dtype='f4')])
                            inst_data_hi = np.vstack([inst_data_hi, np.zeros(INSTANCE_FLOATS, dtype='f4')])
                            focused_mask = np.append(focused_mask, False)
                            # Grow texture/rotation arrays for the new body.
                            self.tex_idx_arr = _build_tex_idx_arr(bodies_data, self.texture_slices)
                            self.rot_period_arr, self.w0_arr, self.tidally_locked_arr, self.parent_idx_arr, self.pole_n_arr, self.tangent_arr, self.bitangent_arr = _build_rotation_props(bodies_data)

                            
                            non_star_indices = np.where(is_star_arr == 0.0)[0]
                            n_casters_fixed = len(non_star_indices)
    
                        elif op["action"] == "DELETE":
                            idx = op["idx"]
                            bodies_data.pop(idx)
                            
                            pos_snap = np.delete(pos_snap, idx, axis=0)
                            vel_snap = np.delete(vel_snap, idx, axis=0)
                            mass_snap = np.delete(mass_snap, idx)
                            
                            ps = np.delete(parent_snap, idx)
                            ps[ps == idx] = 0
                            ps[ps > idx] -= 1
                            parent_snap = ps
                            
                            tree_indices_snap = np.delete(tree_indices_snap, idx)
                            tree_depths_snap = np.delete(tree_depths_snap, idx)
                            pos_snap_render = np.delete(pos_snap_render, idx, axis=0)
                            vel_snap_render = np.delete(vel_snap_render, idx, axis=0)
                            body_radii = np.delete(body_radii, idx)
                            num_bodies -= 1
                            all_instances = np.delete(all_instances, idx, axis=0)
                            cull_mask_lo = np.delete(cull_mask_lo, idx)
                            cull_mask_hi = np.delete(cull_mask_hi, idx)
                            cull_ring_mask = np.delete(cull_ring_mask, idx)
                            subsys_pos_buf = np.delete(subsys_pos_buf, idx, axis=0)
                            subsys_vel_buf = np.delete(subsys_vel_buf, idx, axis=0)
                            subsys_mass_buf = np.delete(subsys_mass_buf, idx)
                            visual_arr = np.delete(visual_arr, idx, axis=0)
                            visual_colors_f8 = np.delete(visual_colors_f8, idx, axis=0)
                            body_colors = np.delete(body_colors, idx, axis=0)
                            is_star_arr = np.delete(is_star_arr, idx)
                            visual_data.pop(idx)
                            inst_data_lo = np.delete(inst_data_lo, idx, axis=0)
                            inst_data_hi = np.delete(inst_data_hi, idx, axis=0)
                            focused_mask = np.delete(focused_mask, idx)
                            # Keep texture/rotation arrays in sync after deletion.
                            self.tex_idx_arr = _build_tex_idx_arr(bodies_data, self.texture_slices)
                            self.rot_period_arr, self.w0_arr, self.tidally_locked_arr, self.parent_idx_arr, self.pole_n_arr, self.tangent_arr, self.bitangent_arr = _build_rotation_props(bodies_data)

                            
                            if star_idx == idx:
                                star_idx = 0
                            elif star_idx > idx:
                                star_idx -= 1
                                
                            non_star_indices = np.where(is_star_arr == 0.0)[0]
                            n_casters_fixed = len(non_star_indices)
                            
                            if self.camera["tracking_idx"] == idx:
                                self.camera["tracking_idx"] = None
                            elif self.camera["tracking_idx"] is not None and self.camera["tracking_idx"] > idx:
                                self.camera["tracking_idx"] -= 1
                            
                            if self.camera["inspected_idx"] == idx:
                                self.camera["inspected_idx"] = None
                                self.camera["inspect_bary"] = False
                            elif self.camera["inspected_idx"] is not None and self.camera["inspected_idx"] > idx:
                                self.camera["inspected_idx"] -= 1
                            
                            atmo_bodies = [a for a in atmo_bodies if a['body_idx'] != idx]
                            for a in atmo_bodies:
                                if a['body_idx'] > idx: a['body_idx'] -= 1
                                
                            ring_precomputed = [r for r in ring_precomputed if r['body_idx'] != idx]
                            for r in ring_precomputed:
                                if r['body_idx'] > idx: r['body_idx'] -= 1
                                
                            self.body_ring_indices = rebuild_ring_gradients_atlas(ring_precomputed, ring_gradient_tex)

                            ring_render_groups = [g for g in ring_render_groups if g['body_idx'] != idx]
                            for g in ring_render_groups:
                                if g['body_idx'] > idx: g['body_idx'] -= 1
                                
                        elif op["action"] == "UPDATE":
                            idx = op["idx"]
                            if "mass" in op:
                                bodies_data[idx]["m"] = float(op["mass"])
                            if "radius" in op:
                                bodies_data[idx]["r"] = float(op["radius"] / 696340.0)
                                r_au = op["radius"] / 149597870.7
                                body_radii[idx] = r_au
                                visual_arr[idx][3] = r_au
                                visual_data[idx][3] = r_au
                                for a in atmo_bodies:
                                    if a['body_idx'] == idx:
                                        old_radius = a['planet_radius_km']
                                        height = max(1.0, a['atmo_radius_km'] - old_radius)
                                        a['planet_radius_km'] = float(op["radius"])
                                        a['surface_radius_au'] = r_au
                                        a['atmo_radius_km'] = a['planet_radius_km'] + height
                                        a['atmo_radius_au'] = a['atmo_radius_km'] / 149597870.7
                                        if 'lut_tex' in a:
                                            a['lut_tex'].release()
                                            del a['lut_tex']
                            if "oblateness" in op:
                                f = op["oblateness"]
                                bodies_data[idx]["oblateness"] = f
                                bodies_data[idx]["J2"] = op["J2"]
                                bodies_data[idx]["j4"] = op["j4"]
                                bodies_data[idx]["rotation_period"] = op["rotation_period"]
                                
                                visual_arr[idx][8] = f
                                visual_data[idx][8] = f
                            if "color" in op:
                                c = op["color"]
                                bodies_data[idx]["color"] = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
                                visual_arr[idx][0:3] = c
                                visual_colors_f8[idx] = np.array(c, dtype='f8')
                                body_colors[idx] = np.array(c, dtype='f4')
                                visual_data[idx][0:3] = c
                            if "star_props" in op:
                                bodies_data[idx]["star_props"] = op["star_props"]
                            if "pos" in op:
                                pidx = parent_snap[idx]
                                if pidx >= 0 and pidx != idx:
                                    bodies_data[idx]["sv"] = {
                                        "x": float(op["pos"][0] - pos_snap[pidx][0]),
                                        "y": float(op["pos"][1] - pos_snap[pidx][1]),
                                        "z": float(op["pos"][2] - pos_snap[pidx][2]),
                                        "vx": float(op["vel"][0] - vel_snap[pidx][0]),
                                        "vy": float(op["vel"][1] - vel_snap[pidx][1]),
                                        "vz": float(op["vel"][2] - vel_snap[pidx][2])
                                    }
                            if "type" in op:
                                bodies_data[idx]["type"] = op["type"]
                                is_star_arr[idx] = 1.0 if op["type"] == "Star" else 0.0
                                min_px = 1.0
                                visual_arr[idx][4] = min_px
                                visual_data[idx][4] = min_px
                                non_star_indices = np.where(is_star_arr == 0.0)[0]
                                n_casters_fixed = len(non_star_indices)
    
                    sys_mgr.save_system_data(active_system_name, bodies_data)
                    self.shared_state["crud_completed"].clear()
                    self.shared_state["rebuild_flag"] = False
                    cached_hierarchy_ver = -1
    
                if len(pos_snap) == len(self.shared_state["pos"]):
                    np.copyto(pos_snap, self.shared_state["pos"])
                    np.copyto(vel_snap, self.shared_state["vel"])
                    np.copyto(mass_snap, self.shared_state["mass"])
                    np.copyto(parent_snap, self.shared_state["parent_indices"])
                if len(tree_indices_snap) == len(self.shared_state["tree_indices"]):
                    np.copyto(tree_indices_snap, self.shared_state["tree_indices"])
                    np.copyto(tree_depths_snap, self.shared_state["tree_depths"])
                hierarchy_ver = self.shared_state["hierarchy_version"]
                
                tl_active = self.shared_state["timeline_active"]
                tl_prog = self.shared_state["timeline_progress"]
                current_sim_t = self.shared_state["t"]
                tl_times = self.shared_state["timeline_times"]
                tl_pos_buf = self.shared_state["timeline_pos"]
                tl_vel_buf = self.shared_state["timeline_vel"]

            cmp_sim_t = 0.0
            if self.comparison_enabled:
                self.time_ctrl_cmp["paused"] = self.time_ctrl["paused"]
                self.time_ctrl_cmp["multiplier"] = self.time_ctrl["multiplier"]
                self.time_ctrl_cmp["time_direction"] = self.time_ctrl["time_direction"]
                self.time_ctrl_cmp["sync_t"] = self.time_ctrl.get("sync_t")
                self.time_ctrl_cmp["sync_idx"] = self.time_ctrl.get("sync_idx")
                self.time_ctrl_cmp["render_timeline"] = self.time_ctrl.get("render_timeline", False)
                self.time_ctrl_cmp["target_t"] = self.time_ctrl.get("target_t", 0.0)
                self.time_ctrl_cmp["cancel_render"] = self.time_ctrl.get("cancel_render", False)
                
                with self.shared_state_cmp["lock"]:
                    if len(self.pos_snap_cmp) == len(self.shared_state_cmp["pos"]):
                        np.copyto(self.pos_snap_cmp, self.shared_state_cmp["pos"])
                        np.copyto(self.vel_snap_cmp, self.shared_state_cmp["vel"])
                        np.copyto(self.mass_snap_cmp, self.shared_state_cmp["mass"])
                        np.copyto(self.parent_snap_cmp, self.shared_state_cmp["parent_indices"])
                    if len(self.tree_indices_snap_cmp) == len(self.shared_state_cmp["tree_indices"]):
                        np.copyto(self.tree_indices_snap_cmp, self.shared_state_cmp["tree_indices"])
                        np.copyto(self.tree_depths_snap_cmp, self.shared_state_cmp["tree_depths"])
                    cmp_sim_t = self.shared_state_cmp["t"]

            now = time.perf_counter()
            dt_render = min(now - last_render_time, 0.1)
            last_render_time = now
            
            glfw.poll_events()
            
            self.impl.process_inputs()
            imgui.new_frame()
            
            if self.fb_width <= 0 or self.fb_height <= 0:
                imgui.end_frame()
                continue
            
            # Screenshot: temporarily override resolution for high-res capture
            _ss_orig_fb = None
            if self._screenshot_request and not self._screenshot_capturing:
                _ss_orig_fb = (self.fb_width, self.fb_height)
                self._screenshot_orig_fb = _ss_orig_fb
                self.fb_width, self.fb_height = self._screenshot_request
                self._screenshot_capturing = True
                self._screenshot_request = None
                self._screenshot_toast = ("Capturing...", time.time())
            
            # Orbit-line MSAA is disabled for screenshot frames to avoid massive VRAM usage
            if getattr(self, "_screenshot_capturing", False):
                msaa_samples = 0
                atmo_res = 1.0
            else:
                msaa_samples = self.camera.get("msaa_samples", 4)
                atmo_res = float(self.camera.get("atmo_resolution", 1.0))
            if self.last_fb_size != (self.fb_width, self.fb_height) or self.last_msaa_samples != msaa_samples or self.last_atmo_res != atmo_res:
                self.last_fb_size = (self.fb_width, self.fb_height)
                self.last_msaa_samples = msaa_samples
                self.last_atmo_res = atmo_res
                
                # Release old
                if self.hdr_resolve_fbo: self.hdr_resolve_fbo.release(); self.hdr_resolve_fbo = None
                if self.hdr_resolve_tex: self.hdr_resolve_tex.release(); self.hdr_resolve_tex = None
                if self.orbit_msaa_fbo: self.orbit_msaa_fbo.release(); self.orbit_msaa_fbo = None
                if self.orbit_resolve_fbo: self.orbit_resolve_fbo.release(); self.orbit_resolve_fbo = None
                if self.orbit_resolved_tex: self.orbit_resolved_tex.release(); self.orbit_resolved_tex = None
                if self.depth_texture: self.depth_texture.release(); self.depth_texture = None
                if self.atmo_lowres_fbo: self.atmo_lowres_fbo.release(); self.atmo_lowres_fbo = None
                if self.atmo_lowres_scatter_tex: self.atmo_lowres_scatter_tex.release(); self.atmo_lowres_scatter_tex = None
                if self.atmo_lowres_trans_tex: self.atmo_lowres_trans_tex.release(); self.atmo_lowres_trans_tex = None
                if getattr(self, "accum_fbo_a", None): self.accum_fbo_a.release(); self.accum_fbo_a = None
                if getattr(self, "accum_tex_a", None): self.accum_tex_a.release(); self.accum_tex_a = None
                if getattr(self, "accum_fbo_b", None): self.accum_fbo_b.release(); self.accum_fbo_b = None
                if getattr(self, "accum_tex_b", None): self.accum_tex_b.release(); self.accum_tex_b = None
                self.prev_cam_origin = None
                for fbo in self.bloom_fbos: fbo.release()
                for tex in self.bloom_texs: tex.release()
                self.bloom_fbos = []
                self.bloom_texs = []
                
                # Rebuild
                self.hdr_resolve_tex = ctx.texture((self.fb_width, self.fb_height), 4, dtype='f4')
                self.hdr_resolve_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.hdr_resolve_tex.repeat_x = False
                self.hdr_resolve_tex.repeat_y = False
                
                self.depth_texture = ctx.depth_texture((self.fb_width, self.fb_height))
                self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self.depth_texture.repeat_x = False
                self.depth_texture.repeat_y = False
                
                self.hdr_resolve_fbo = ctx.framebuffer(color_attachments=[self.hdr_resolve_tex], depth_attachment=self.depth_texture)
                
                self.accum_tex_a = ctx.texture((self.fb_width, self.fb_height), 4, dtype='f4')
                self.accum_tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.accum_fbo_a = ctx.framebuffer(color_attachments=[self.accum_tex_a])
                self.accum_tex_b = ctx.texture((self.fb_width, self.fb_height), 4, dtype='f4')
                self.accum_tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.accum_fbo_b = ctx.framebuffer(color_attachments=[self.accum_tex_b])
                self.photo_accum_count = 0
                self.photo_accum_last_cam = None
                
                if atmo_res < 0.999:
                    low_w = max(1, int(self.fb_width * atmo_res))
                    low_h = max(1, int(self.fb_height * atmo_res))
                    self.atmo_lowres_scatter_tex = ctx.texture((low_w, low_h), 4, dtype='f4')
                    self.atmo_lowres_scatter_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    self.atmo_lowres_scatter_tex.repeat_x = False
                    self.atmo_lowres_scatter_tex.repeat_y = False
                    
                    self.atmo_lowres_trans_tex = ctx.texture((low_w, low_h), 4, dtype='f4')
                    self.atmo_lowres_trans_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    self.atmo_lowres_trans_tex.repeat_x = False
                    self.atmo_lowres_trans_tex.repeat_y = False
                    
                    self.atmo_lowres_fbo = ctx.framebuffer(color_attachments=[self.atmo_lowres_scatter_tex, self.atmo_lowres_trans_tex])
                else:
                    self.atmo_lowres_fbo = None
                    self.atmo_lowres_scatter_tex = None
                    self.atmo_lowres_trans_tex = None
                
                if msaa_samples > 0:
                    orbit_msaa_color = ctx.renderbuffer((self.fb_width, self.fb_height), components=4, samples=msaa_samples, dtype='f4')
                    orbit_msaa_depth = ctx.depth_renderbuffer((self.fb_width, self.fb_height), samples=msaa_samples)
                    self.orbit_msaa_fbo = ctx.framebuffer(color_attachments=[orbit_msaa_color], depth_attachment=orbit_msaa_depth)
                    self.orbit_resolved_tex = ctx.texture((self.fb_width, self.fb_height), 4, dtype='f4')
                    self.orbit_resolved_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    self.orbit_resolved_tex.repeat_x = False
                    self.orbit_resolved_tex.repeat_y = False
                    self.orbit_resolve_fbo = ctx.framebuffer(color_attachments=[self.orbit_resolved_tex])
                else:
                    self.orbit_msaa_fbo = None
                    self.orbit_resolved_tex = None
                    self.orbit_resolve_fbo = None
                
                # Bloom chain (5 levels)
                bw, bh = self.fb_width // 2, self.fb_height // 2
                for i in range(5):
                    bw = max(1, bw)
                    bh = max(1, bh)
                    btex = ctx.texture((bw, bh), 3, dtype='f4')
                    btex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    btex.repeat_x = False
                    btex.repeat_y = False
                    self.bloom_texs.append(btex)
                    self.bloom_fbos.append(ctx.framebuffer(color_attachments=[btex]))
                    bw //= 2
                    bh //= 2

            ctx.viewport = (0, 0, self.fb_width, self.fb_height)
            self.hdr_resolve_fbo.use()
            ctx.clear(0.0, 0.0, 0.0, 1.0) 
    
            is_scrubbing = tl_active and tl_prog >= 1.0
    
            if is_scrubbing and len(tl_times) > 0:
                scrub_idx = min(scrub_index[0], len(tl_times) - 1)
                display_t = tl_times[scrub_idx]
                pos_snap_render = tl_pos_buf[scrub_idx].astype('f8')
                vel_snap_render = tl_vel_buf[scrub_idx].astype('f8')
            else:
                display_t = current_sim_t
                pos_snap_render = pos_snap.copy()
                vel_snap_render = vel_snap.copy()
    
            spice_valid_mask = np.ones(num_bodies, dtype=bool)
    
            cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s, cur_tz = format_sim_time(display_t)
    
            if ephemeris_mode_active and sys_mgr_spice.kernels_loaded:
                epoch_dt = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
                et_epoch = sys_mgr_spice.datetime_to_et(epoch_dt)
                et = et_epoch + display_t * 365.25 * 86400.0
                
                if (getattr(self, "_ephem_mapping_ver", None) != id(bodies_data) or 
                    getattr(self, "_ephem_mapping_len", 0) != len(bodies_data)):
                    self._ephem_mapping = sys_mgr_spice.get_body_mapping(bodies_data, et_epoch)
                    self._ephem_mapping_ver = id(bodies_data)
                    self._ephem_mapping_len = len(bodies_data)
                    
                sys_mgr_spice.populate_states_fast(et, self._ephem_mapping, pos_snap_render, vel_snap_render, spice_valid_mask)
    
            if not imgui.get_io().want_capture_keyboard:
                d_roll = 0.0
                if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
                    d_roll -= 60.0 * dt_render
                if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
                    d_roll += 60.0 * dt_render
                if d_roll != 0.0:
                    fwd_v = -rel / max(np.linalg.norm(rel), 1e-300) if cam["cam_look"] == "aim" else _camera_forward(cam["yaw_actual"], cam["pitch_actual"])
                    cur_up = _camera_get_up(cam, fwd_v)
                    new_up = _camera_rot_axis(fwd_v, math.radians(d_roll)) @ cur_up
                    cam["up"] = (new_up / np.linalg.norm(new_up)).tolist()
                    cam["roll"] += d_roll

                # Press F to snap camera look to tracked body center
                if glfw.get_key(window, glfw.KEY_F) == glfw.PRESS:
                    if not getattr(self, "_f_key_held", False):
                        self.camera["cam_look"] = "aim"
                        self._f_key_held = True
                else:
                    self._f_key_held = False

            self.body_radii = body_radii

            # ── Camera: Space Engine-style free flight / orbit / landing ──
            cam = self.camera
            LAND_ALT_AU = 1e-10   # ~15 m stick height above the surface
            rel_prev = np.array(cam.get("cam_pos_rel_prev", cam["cam_pos_rel"]), dtype='f8')
            rel = np.array(cam["cam_pos_rel"], dtype='f8')

            lerp_factor = 1.0 - math.exp(-15.0 * dt_render)
            cam["yaw_actual"] += (cam["yaw"] - cam["yaw_actual"]) * lerp_factor
            cam["pitch_actual"] += (cam["pitch"] - cam["pitch_actual"]) * lerp_factor
            cam["roll_actual"] += (cam["roll"] - cam["roll_actual"]) * lerp_factor

            compute_barycenters(pos_snap_render, vel_snap_render, mass_snap, parent_snap,
                                subsys_pos_buf, subsys_vel_buf, subsys_mass_buf)
            if self.comparison_enabled:
                compute_barycenters(self.pos_snap_cmp, self.vel_snap_cmp, self.mass_snap_cmp, self.parent_snap_cmp,
                                    self.subsys_pos_buf_cmp, self.subsys_vel_buf_cmp, self.subsys_mass_buf_cmp)

            # Resolve the camera pivot (tracked body / barycenter, or free target)
            if cam["tracking_idx"] is not None:
                track_idx = cam["tracking_idx"]
                if cam.get("tracking_is_cmp", False):
                    if cam["tracking_mode"] == "barycenter":
                        base_pos = self.subsys_pos_buf_cmp[track_idx].copy() + np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                    else:
                        base_pos = self.pos_snap_cmp[track_idx].copy() + np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                else:
                    if cam["tracking_mode"] == "barycenter":
                        base_pos = subsys_pos_buf[track_idx].copy()
                    else:
                        base_pos = pos_snap_render[track_idx].copy()
            else:
                base_pos = np.array(cam["target"], dtype='f8')

            # Tracked body surface radius (used for approach gesture & landing).
            # Oblate bodies: track_r is the equatorial radius; the surface is an
            # ellipsoid, so the clamp uses the radius along the camera's direction.
            track_r = 0.0
            track_f = 0.0
            track_pole = np.array([0.0, 1.0, 0.0], dtype='f8')
            if cam["tracking_idx"] is not None and cam["tracking_mode"] == "body":
                t_idx = cam["tracking_idx"]
                if cam.get("tracking_is_cmp", False):
                    if hasattr(self, "body_radii_cmp") and self.body_radii_cmp is not None and t_idx < len(self.body_radii_cmp):
                        track_r = float(self.body_radii_cmp[t_idx])
                        track_f = float(self.bodies_data_cmp[t_idx].get('oblateness', 0.0))
                        if hasattr(self, "pole_n_arr_cmp") and t_idx < len(self.pole_n_arr_cmp):
                            track_pole = np.array(self.pole_n_arr_cmp[t_idx], dtype='f8')
                elif hasattr(self, "body_radii") and self.body_radii is not None and t_idx < len(self.body_radii):
                    track_r = float(self.body_radii[t_idx])
                    track_f = float(bodies_data[t_idx].get('oblateness', 0.0))
                    if hasattr(self, "pole_n_arr") and t_idx < len(self.pole_n_arr):
                        track_pole = np.array(self.pole_n_arr[t_idx], dtype='f8')
            
            cam["track_f"] = track_f
            cam["track_pole"] = track_pole.tolist()
            if cam.get("tracking_idx") is not None and track_r > 0.0:
                alt_au = max(0.0, np.linalg.norm(rel) - _ellipsoid_surface_radius(track_r, track_f, track_pole, rel))
                cam["is_near_surface"] = (alt_au < _ROT_FOLLOW_ALT_AU)
            else:
                cam["is_near_surface"] = False

            # 1) Camera approach / zoom gesture: radial move toward the tracked body's surface
            if cam["approach_delta"] != 0.0:
                r_ap = np.linalg.norm(rel)
                if r_ap > 1e-300 and cam["tracking_idx"] is not None and cam["tracking_mode"] == "body" and track_r > 0.0:
                    surf_r = _ellipsoid_surface_radius(track_r, track_f, track_pole, rel)
                    min_r = surf_r + LAND_ALT_AU
                    alt = max(LAND_ALT_AU, r_ap - surf_r)
                    alt_new = max(LAND_ALT_AU, alt * math.exp(-cam["approach_delta"]))
                    r_new = surf_r + alt_new
                    rel = (rel / r_ap) * r_new
                else:
                    # no tracked surface: dolly along the view direction
                    dolly = cam["approach_delta"] * max(1e-4, np.linalg.norm(rel))
                    if cam["cam_look"] == "aim":
                        fwd_a = -rel / max(np.linalg.norm(rel), 1e-300)
                    else:
                        fwd_a = _camera_forward(cam["yaw_actual"], cam["pitch_actual"])
                    rel += fwd_a * dolly
                cam["approach_delta"] = 0.0

            # 2) Rotation follow: below 200 km altitude the camera rides the
            #    tracked body's rotation (full strength at the surface, fading
            #    out smoothly toward the 200 km threshold) so the ground stays
            #    put beneath it without a hard "landed" switch.
            rot_follow = (track_r > 0.0 and cam["tracking_mode"] == "body" and
                          cam["tracking_idx"] is not None and not cam.get("tracking_is_cmp", False))
            if rot_follow:
                alt_au = max(0.0, np.linalg.norm(rel) - _ellipsoid_surface_radius(track_r, track_f, track_pole, rel))
                if alt_au < _ROT_FOLLOW_ALT_AU:
                    alt_km = alt_au * AU_TO_KM
                    if alt_km <= 50.0:
                        rot_weight = 1.0
                    else:
                        t_fade = (alt_km - 50.0) / 150.0
                        rot_weight = 1.0 - (3.0 * t_fade**2 - 2.0 * t_fade**3)
                    spin_angles = compute_body_rotation_angles_jit(
                        float(display_t) * 31557600.0,
                        self.rot_period_arr, self.w0_arr, self.tidally_locked_arr, self.parent_idx_arr,
                        pos_snap_render, self.pole_n_arr, self.tangent_arr, self.bitangent_arr)
                    t_idx = cam["tracking_idx"]
                    if t_idx < len(spin_angles):
                        spin_angle = float(spin_angles[t_idx])
                        prev_spin = getattr(self, "_contact_prev_spin", None)
                        prev_track = getattr(self, "_contact_prev_track", None)
                        if prev_track != t_idx:
                            prev_spin = None
                        self._contact_prev_spin = spin_angle
                        self._contact_prev_track = t_idx
                        if prev_spin is not None:
                            d_spin = spin_angle - prev_spin
                            d_spin = (d_spin + math.pi) % (2.0 * math.pi) - math.pi
                            if abs(d_spin) > 1e-12 and abs(d_spin) < 0.5:
                                pole_axis = self.pole_n_arr[t_idx]
                                R_spin = _camera_rot_axis(pole_axis, d_spin * rot_weight)
                                rel = R_spin @ rel
                                cur_fwd = -rel / max(np.linalg.norm(rel), 1e-300) if cam["cam_look"] == "aim" else _camera_forward(cam["yaw_actual"], cam["pitch_actual"])
                                cur_up = _camera_get_up(cam, cur_fwd)
                                new_up = R_spin @ cur_up
                                cam["up"] = (new_up / np.linalg.norm(new_up)).tolist()
                                if cam["cam_look"] == "free":
                                    fwd_new = R_spin @ cur_fwd
                                    fwd_new = fwd_new / np.linalg.norm(fwd_new)
                                    yaw_new, pitch_new = _camera_yaw_pitch_from(fwd_new)
                                    cam["yaw"] = cam["yaw_actual"] = yaw_new
                                    cam["pitch"] = cam["pitch_actual"] = pitch_new
                    else:
                        self._contact_prev_spin = None
                        self._contact_prev_track = None
                else:
                    self._contact_prev_spin = None
                    self._contact_prev_track = None
            else:
                self._contact_prev_spin = None
                self._contact_prev_track = None

            # 2b) Horizon alignment: below the same 200 km threshold, gently roll
            #     the camera (like holding Q/E) so the view is level to the local
            #     ground horizon — the local vertical (radial from the body's
            #     center through the camera) projects straight up-screen, so the
            #     ground/sky line appears horizontal regardless of the body's
            #     axial tilt. Degenerate (no-op) when looking straight along the
            #     local vertical. Interruptible: manual roll still works, the
            #     view just levels back within a few seconds. Toggleable via
            #     cam["horizon_align"].
            if cam.get("horizon_align", True) and cam.get("tracking_idx") is not None and track_r > 0.0:
                alt_au = max(0.0, np.linalg.norm(rel) - _ellipsoid_surface_radius(track_r, track_f, track_pole, rel))
                if alt_au < _ROT_FOLLOW_ALT_AU:
                    align_strength = 1.0 - alt_au / _ROT_FOLLOW_ALT_AU
                    if cam["cam_look"] == "aim":
                        fwd_al = -rel / max(np.linalg.norm(rel), 1e-300)
                    else:
                        fwd_al = _camera_forward(cam["yaw_actual"], cam["pitch_actual"])
                    local_up = _oblate_surface_normal(rel, track_f, track_pole)
                    target_up = local_up - float(np.dot(local_up, fwd_al)) * fwd_al
                    n_tup = np.linalg.norm(target_up)
                    if n_tup > 1e-6:
                        target_up = target_up / n_tup
                        cur_up = _camera_get_up(cam, fwd_al)
                        c_dot = max(-1.0, min(1.0, float(np.dot(cur_up, target_up))))
                        angle = math.acos(c_dot)
                        if angle > 1e-6:
                            rot_axis = np.cross(cur_up, target_up)
                            if float(np.dot(rot_axis, fwd_al)) < 0.0:
                                angle = -angle
                            d_ang = angle * min(1.0, 1.5 * align_strength * dt_render)
                            new_up = _camera_rot_axis(fwd_al, d_ang) @ cur_up
                            cam["up"] = (new_up / np.linalg.norm(new_up)).tolist()

            # 3) WASD flight (screen-space strafe)
            keys = cam.get("keys", {})
            w_key = keys.get("w", False)
            s_key = keys.get("s", False)
            a_key = keys.get("a", False)
            d_key = keys.get("d", False)

            if cam["cam_look"] == "aim":
                r_f = np.linalg.norm(rel)
                if r_f > 1e-300:
                    front = -rel / r_f
                else:
                    front = np.array([0.0, 0.0, -1.0], dtype='f8')
            else:
                front = _camera_forward(cam["yaw_actual"], cam["pitch_actual"])
            # screen-space strafe: A/D move along the camera's own right
            # (roll-aware), never along the ecliptic horizontal
            cur_up_v = _camera_get_up(cam, front)
            right_v = np.cross(front, cur_up_v)
            nr_r = np.linalg.norm(right_v)
            if nr_r > 1e-9:
                right_v = right_v / nr_r
            else:
                right_v = np.array([1.0, 0.0, 0.0], dtype='f8')
            move = np.zeros(3, dtype='f8')
            if w_key: move += front
            if s_key: move -= front
            if d_key: move += right_v
            if a_key: move -= right_v
            nm = np.linalg.norm(move)
            if nm > 1e-12:
                rel = rel + (move / nm) * cam["flight_speed"] * dt_render

            # 3b) floor: push back out of the ground after movement
            if track_r > 0.0 and cam["tracking_mode"] == "body":
                r_s = np.linalg.norm(rel)
                r_floor = _ellipsoid_surface_radius(track_r, track_f, track_pole, rel) + LAND_ALT_AU
                if r_s <= r_floor:
                    if r_s > 1e-300:
                        rel = rel / r_s * r_floor
                    else:
                        rel = track_pole * r_floor
                    # Cap flight speed to 100 m/s when hitting the ground
                    cam["flight_speed"] = 100.0 / 149597870700.0

            cam["cam_pos_rel"] = rel
            cam["cam_pos_rel_prev"] = rel.copy()
            cam["cam_vel"][:] = (rel - rel_prev) / max(dt_render, 1e-9)

            cam_origin = base_pos
    
            if hierarchy_ver != cached_hierarchy_ver:
                cached_children_map = {}
                for i in range(num_bodies):
                    pi = int(parent_snap[i])
                    if pi >= 0:
                        if pi not in cached_children_map:
                            cached_children_map[pi] = []
                        cached_children_map[pi].append(i)
                cached_hierarchy_ver = hierarchy_ver
            children_map = cached_children_map
    
            focused_root = -1
            focused_root_cmp = -1
            
            if self.camera["tracking_idx"] is not None and not self.camera.get("tracking_is_cmp", False):
                body_idx = self.camera["tracking_idx"]
                if body_idx < num_bodies:
                    while True:
                        p = int(parent_snap[body_idx])
                        if p == -1 or p == star_idx:
                            focused_root = body_idx
                            break
                        body_idx = p
            
            if self.camera["tracking_idx"] is not None and self.camera.get("tracking_is_cmp", False):
                body_idx = self.camera["tracking_idx"]
                if body_idx < self.num_bodies_cmp:
                    while True:
                        p = int(self.parent_snap_cmp[body_idx])
                        if p == -1 or p == self.star_idx_cmp:
                            focused_root_cmp = body_idx
                            break
                        body_idx = p
            
            total_render_bodies = num_bodies
            if self.comparison_enabled:
                total_render_bodies += self.num_bodies_cmp

            focused_mask = np.zeros(total_render_bodies, dtype=np.bool_)
            if focused_root >= 0:
                stack = [focused_root]
                while stack:
                    node = stack.pop()
                    if node < len(focused_mask):
                        focused_mask[node] = True
                    if node in children_map:
                        stack.extend(children_map[node])
            
            if self.comparison_enabled:
                if self.camera.get("tracking_is_cmp", False) and focused_root_cmp >= 0:
                    children_map_cmp = {}
                    for i in range(self.num_bodies_cmp):
                        pi = int(self.parent_snap_cmp[i])
                        if pi >= 0:
                            if pi not in children_map_cmp:
                                children_map_cmp[pi] = []
                            children_map_cmp[pi].append(i)
                            
                    stack = [focused_root_cmp]
                    while stack:
                        node = stack.pop()
                        cmp_node_idx = num_bodies + node
                        if cmp_node_idx < len(focused_mask):
                            focused_mask[cmp_node_idx] = True
                        if node in children_map_cmp:
                            stack.extend(children_map_cmp[node])
                else:
                    focused_mask[num_bodies:] = True
            
            pos_rel_all = (pos_snap_render - cam_origin).astype('f4')
            
            # Hide bodies from rendering if they lost SPICE data
            if ephemeris_mode_active:
                cull_mask_lo = cull_mask_lo & spice_valid_mask
                cull_mask_hi = cull_mask_hi & spice_valid_mask
                cull_ring_mask = cull_ring_mask & spice_valid_mask
            
            # Build the view matrix from the current camera state
            cam_pos_f8 = np.array(rel, dtype='f8')
            yaw_rad_v, pitch_rad_v = math.radians(cam["yaw_actual"]), math.radians(cam["pitch_actual"])
            if cam["cam_look"] == "aim":
                fwd_v = -cam_pos_f8 / max(np.linalg.norm(cam_pos_f8), 1e-300)
                cam_up = _camera_get_up(cam, fwd_v)
                view_f8 = matrix44.create_look_at(cam_pos_f8, [0.0, 0.0, 0.0], cam_up, dtype='f8')
            else:
                fwd_v = _camera_forward(cam["yaw_actual"], cam["pitch_actual"])
                cam_up = _camera_get_up(cam, fwd_v)
                view_f8 = matrix44.create_look_at(cam_pos_f8, cam_pos_f8 + fwd_v, cam_up, dtype='f8')
            
            view = view_f8.astype('f4')
            cam_pos = cam_pos_f8.astype('f4')

            # Adaptive near plane: scale with the nearest surface so landing stays valid
            if num_bodies > 0:
                d_surf = np.linalg.norm(pos_snap_render - cam_origin, axis=1) - body_radii
                near_surf = float(np.min(d_surf))
                if self.comparison_enabled and self.num_bodies_cmp > 0:
                    d_surf_cmp = np.linalg.norm(self.pos_snap_cmp - cam_origin + np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8'), axis=1) - self.body_radii_cmp
                    near_surf = min(near_surf, float(np.min(d_surf_cmp)))
                near = min(max(near_surf * 0.02, 1e-13), 1e-4)
            else:
                near = 1e-4
            
            # Incorporate both primary and comparison system body relative positions and radii
            max_dist_from_target = 0.0
            if len(pos_rel_all) > 0:
                dists_target = np.sqrt(np.sum(pos_rel_all * pos_rel_all, axis=1))
                max_dist_from_target = float(np.max(dists_target + body_radii))
                
            if self.comparison_enabled and self.num_bodies_cmp > 0:
                # Calculate relative positions including the offset for comparison bodies
                cmp_pos_rel = self.pos_snap_cmp - cam_origin + np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                dists_target_cmp = np.sqrt(np.sum(cmp_pos_rel * cmp_pos_rel, axis=1))
                max_dist_from_target_cmp = float(np.max(dists_target_cmp + self.body_radii_cmp))
                max_dist_from_target = max(max_dist_from_target, max_dist_from_target_cmp)
                
            # Set the far plane with a wide margin using logarithmic depth to prevent clipping of wide orbits/bodies
            dist_from_pivot = float(np.linalg.norm(rel))
            far = max(dist_from_pivot * 50.0, (dist_from_pivot + max_dist_from_target) * 20.0, 100000.0)
            depth_C = 1.0 / max(near, 1e-13)
            aspect_ratio = self.fb_width / max(self.fb_height, 1)
            projection_f8 = matrix44.create_perspective_projection_matrix(self.camera["fov"], aspect_ratio, near, far, dtype='f8')
            projection = projection_f8.astype('f4')
    
            n_ring_planes = 0
            ring_centers_buf[:] = 0
            ring_normals_buf[:] = 0
            ring_params_buf[:] = 0
            ring_colors_buf[:] = 0
            ring_5colors_buf[:] = 0
            
            n_ring_planes = len(self.body_ring_indices)
            for bi, unified_idx in self.body_ring_indices.items():
                if unified_idx >= 16:
                    break
                body_rings = [r for r in ring_precomputed if r['body_idx'] == bi]
                active_rings = [r for r in body_rings if r['opacity'] > 0.0]
                
                if active_rings:
                    min_r = min(r['inner_r'] for r in active_rings)
                    max_r = max(r['outer_r'] for r in active_rings)
                    opacity = 1.0
                    pole = active_rings[0]['pole']
                    raw_color = active_rings[0]['raw_color']
                    colors5 = active_rings[0].get('5colors', raw_color)
                else:
                    min_r, max_r, opacity = 0.0, 0.0, 0.0
                    pole = body_rings[0]['pole'] if body_rings else np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    raw_color = (0.0, 0.0, 0.0)
                    colors5 = raw_color
                    
                ring_centers_buf[unified_idx] = pos_rel_all[bi]
                ring_normals_buf[unified_idx] = pole
                ring_params_buf[unified_idx, 0] = min_r
                ring_params_buf[unified_idx, 1] = max_r
                ring_params_buf[unified_idx, 2] = opacity
                ring_params_buf[unified_idx, 3] = body_radii[bi]
                ring_colors_buf[unified_idx, 0:3] = raw_color
                if isinstance(colors5, np.ndarray):
                    ring_5colors_buf[unified_idx, :, :] = colors5
                else:
                    ring_5colors_buf[unified_idx, :, :] = raw_color

            
            ring_coplanar_mask_buf[:] = compute_ring_coplanar_masks(n_ring_planes, ring_centers_buf, ring_normals_buf)
    
            # Dynamic Caster Selection: Sort non-star bodies by camera-relative angular size
            if len(non_star_indices) > 0:
                dists_to_cam = np.sqrt(np.sum(pos_rel_all[non_star_indices] * pos_rel_all[non_star_indices], axis=1))
                visual_scores = body_radii[non_star_indices] / np.maximum(dists_to_cam, 1e-6)
                sorted_sub_indices = np.argsort(visual_scores)[::-1]
                caster_indices = non_star_indices[sorted_sub_indices]
            else:
                caster_indices = non_star_indices
                
            caster_budget = self.camera.get("shadow_caster_budget", 32)
            n_casters_fixed = min(len(caster_indices), caster_budget)
            
            compute_ring_culling(
                pos_rel_all, body_radii, star_idx,
                n_casters_fixed, caster_indices[:n_casters_fixed],
                n_ring_planes, ring_centers_buf, ring_normals_buf, ring_params_buf[:, 1],
                num_bodies, cull_ring_caster_lo, cull_ring_caster_hi
            )
            
            vp_matrix_f8 = view_f8 @ projection_f8
            frustum_planes = extract_frustum_planes(vp_matrix_f8).astype(np.float32)
            vp_matrix = view @ projection
            
            if self.pick_request is not None:
                px, py = self.pick_request
                self.pick_request = None
                
                best_dist_sq = 400.0 # 20 pixels max distance for tiny bodies
                best_idx = -1
                fov_factor_pick = 1.0 / math.tan(math.radians(self.camera["fov"] / 2.0))
                
                for i in range(num_bodies):
                    if not spice_valid_mask[i]: continue
                    
                    b_pos = pos_rel_all[i] - cam_pos
                    r = body_radii[i]
                    
                    b_pos_4 = np.array([b_pos[0], b_pos[1], b_pos[2], 1.0], dtype=np.float32)
                    clip_pos = np.dot(b_pos_4, vp_matrix)
                    
                    if clip_pos[3] > 0.1:
                        sx = (clip_pos[0]/clip_pos[3] + 1.0) * 0.5 * self.window_width
                        sy = (1.0 - clip_pos[1]/clip_pos[3]) * 0.5 * self.window_height
                        dx = sx - px
                        dy = sy - py
                        dist_sq = dx*dx + dy*dy
                        
                        screen_r_sq = (r / clip_pos[3] * self.window_height * fov_factor_pick)**2
                        
                        if dist_sq < best_dist_sq or dist_sq < screen_r_sq:
                            best_dist_sq = dist_sq if dist_sq < screen_r_sq else dist_sq
                            best_idx = i
                            best_is_cmp = False
                            
                if self.comparison_enabled:
                    for i in range(self.num_bodies_cmp):
                        b_pos = cmp_pos_rel[i].astype('f4') - cam_pos
                        r = self.body_radii_cmp[i]
                        
                        b_pos_4 = np.array([b_pos[0], b_pos[1], b_pos[2], 1.0], dtype=np.float32)
                        clip_pos = np.dot(b_pos_4, vp_matrix)
                        
                        if clip_pos[3] > 0.1:
                            sx = (clip_pos[0]/clip_pos[3] + 1.0) * 0.5 * self.window_width
                            sy = (1.0 - clip_pos[1]/clip_pos[3]) * 0.5 * self.window_height
                            dx = sx - px
                            dy = sy - py
                            dist_sq = dx*dx + dy*dy
                            
                            screen_r_sq = (r / clip_pos[3] * self.window_height * fov_factor_pick)**2
                            
                            if dist_sq < best_dist_sq or dist_sq < screen_r_sq:
                                best_dist_sq = dist_sq if dist_sq < screen_r_sq else dist_sq
                                best_idx = i
                                best_is_cmp = True
                                
                if best_idx != -1:
                    self.camera["inspected_idx"] = best_idx
                    self.camera["inspected_is_cmp"] = best_is_cmp

    
            if not hasattr(self, '_all_instances_cache') or self._all_instances_cache.shape[0] != total_render_bodies:
                self._all_instances_cache = np.zeros((total_render_bodies, INSTANCE_FLOATS), dtype='f4')
            else:
                self._all_instances_cache[:] = 0.0
            all_instances = self._all_instances_cache
            all_instances[:num_bodies, 0:3] = pos_rel_all
            all_instances[:num_bodies, 3:8] = visual_arr[:, 0:5]
            all_instances[:num_bodies, 8] = is_star_arr
            all_instances[:num_bodies, 9:12] = visual_arr[:, 5:8]
            all_instances[:num_bodies, 12] = visual_arr[:, 8]
            if visual_arr.shape[1] > 9:
                all_instances[:num_bodies, 13:16] = visual_arr[:, 9:12]
                all_instances[:num_bodies, 19] = visual_arr[:, 12]
                all_instances[:num_bodies, 23] = visual_arr[:, 13]
            
            all_instances[:num_bodies, 24] = self.tex_idx_arr[:num_bodies]
            sim_t_sec = float(display_t) * 31557600.0
            all_instances[:num_bodies, 25] = compute_body_rotation_angles_jit(
                sim_t_sec, self.rot_period_arr, self.w0_arr, self.tidally_locked_arr, self.parent_idx_arr, pos_snap_render, self.pole_n_arr, self.tangent_arr, self.bitangent_arr
            )
            
            if self.comparison_enabled:
                cmp_pos_rel = self.pos_snap_cmp - cam_origin + np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                all_instances[num_bodies:, 0:3] = cmp_pos_rel.astype('f4')
                all_instances[num_bodies:, 3:8] = self.visual_arr_cmp[:, 0:5]
                all_instances[num_bodies:, 8] = self.is_star_arr_cmp
                all_instances[num_bodies:, 9:12] = self.visual_arr_cmp[:, 5:8]
                all_instances[num_bodies:, 24] = self.tex_idx_arr_cmp[:self.num_bodies_cmp]
                cmp_t_sec = float(cmp_sim_t) * 31557600.0
                all_instances[num_bodies:, 25] = compute_body_rotation_angles_jit(
                    cmp_t_sec, self.rot_period_arr_cmp, self.w0_arr_cmp, self.tidally_locked_arr_cmp, self.parent_idx_arr_cmp, self.pos_snap_cmp, self.pole_n_arr_cmp, self.tangent_arr_cmp, self.bitangent_arr_cmp
                )
                
            # --- Dynamic Texture Streaming: evaluate apparent pixel sizes & process completed uploads ---
            stream_thresh_px = float(self.camera.get("tex_stream_threshold_px", 500.0))
            cur_fov_factor = 1.0 / math.tan(math.radians(max(1.0, self.camera["fov"]) / 2.0))
            
            # Check apparent size for each body to trigger high-res streaming
            for b_i in range(num_bodies):
                t_slice = int(self.tex_idx_arr[b_i])
                if t_slice <= 0:
                    continue
                b_name_lower = bodies_data[b_i].get('name', '').lower()
                b_pos = pos_rel_all[b_i]
                b_rad = body_radii[b_i]
                cam_d = math.sqrt(b_pos[0]**2 + b_pos[1]**2 + b_pos[2]**2)
                apparent_px = (b_rad / max(1e-12, cam_d)) * self.fb_height * cur_fov_factor
                is_tracked = (self.camera.get("tracking_idx") == b_i and not self.camera.get("tracking_is_cmp", False))
                
                if (apparent_px >= stream_thresh_px or is_tracked) and self.active_res_level.get(t_slice) != 'high':
                    self.texture_streamer.request_high_res(b_name_lower)

            # Process any completed high-res decoded textures from background worker
            decoded_results = self.texture_streamer.poll_results(max_items=2)
            for res in decoded_results:
                r_idx = res['idx']
                r_slot = r_idx - 1
                maps = res['maps']
                if not maps:
                    continue
                
                tex_dict = self.gpu_body_textures.get(r_idx, {})
                for map_name, (sz, raw_b, comp) in maps.items():
                    # Release previous fallback/old texture if needed
                    if map_name in tex_dict and tex_dict[map_name] is not None:
                        try:
                            tex_dict[map_name].release()
                        except Exception:
                            pass
                    
                    new_tex = _upload_texture_obj(sz, raw_b, comp)
                    tex_dict[map_name] = new_tex
                    h_64 = int(new_tex.get_handle(resident=True))
                    lo = h_64 & 0xFFFFFFFF
                    hi = (h_64 >> 32) & 0xFFFFFFFF
                    m_offset = {'diffuse': 0, 'normal': 1, 'specular': 2, 'clouds': 3}[map_name]
                    self.body_textures_ssbo_data[r_slot, m_offset] = [lo, hi]

                self.gpu_body_textures[r_idx] = tex_dict
                self.active_res_level[r_idx] = 'high'
                # Update the SSBO on GPU
                self.body_textures_ssbo.write(self.body_textures_ssbo_data.tobytes())

            # Bind the body textures SSBO to binding point 10 before rendering
            self.body_textures_ssbo.bind_to_storage_buffer(binding=10)
            is_star_mask = all_instances[:total_render_bodies, 8] > 0.5
            star_positions = all_instances[:total_render_bodies][is_star_mask, 0:3]
            star_colors = all_instances[:total_render_bodies][is_star_mask, 3:6]
            star_radii = all_instances[:total_render_bodies][is_star_mask, 6] # Added star radii
            star_indices_numba = np.where(is_star_mask[:total_render_bodies])[0]
            star_lums = np.ones(len(star_indices_numba), dtype=np.float32)
            
            for _k in range(len(star_indices_numba)):
                _idx = star_indices_numba[_k]
                if _idx < num_bodies:
                    _bd = bodies_data[_idx]
                else:
                    _bd = self.bodies_data_cmp[_idx - num_bodies]
                if "star_props" in _bd and "lum" in _bd["star_props"]:
                    star_lums[_k] = float(_bd["star_props"]["lum"])
                    
            hdr_enabled = self.camera.get("hdr_enabled", True)
            
            planetshine_dirs, planetshine_colors = compute_planetshine_numba(
                all_instances[:total_render_bodies, 0:3],
                all_instances[:total_render_bodies, 6],
                all_instances[:total_render_bodies, 3:6],
                all_instances[:total_render_bodies, 8],
                star_positions,
                star_colors,
                star_lums,
                star_radii, # Pass radii to Numba
                hdr_enabled
            )
            all_instances[:total_render_bodies, 16:19] = planetshine_dirs
            all_instances[:total_render_bodies, 20:23] = planetshine_colors
            
            all_instances_buffer.write(all_instances[:total_render_bodies].tobytes())
            
            cmds_data = np.array([
                len(mesh_lo_idx), 0, 0, 0, 0,
                len(mesh_hi_idx), 0, 0, 0, 0,
                len(mesh_ultra_idx), 0, 0, 0, 0,
            ], dtype=np.uint32)
            draw_cmds_buffer.write(cmds_data.tobytes())
            
            focused_mask_buffer.write(focused_mask[:total_render_bodies].view(np.uint8).astype(np.uint32).tobytes())
            
            prog_culling_compute['u_num_bodies'].value = total_render_bodies
            prog_culling_compute['u_star_idx'].value = star_idx
            prog_culling_compute['u_n_casters'].value = n_casters_fixed
            
            c_idx_arr = np.zeros(64, dtype=np.int32)
            if n_casters_fixed > 0:
                c_idx_arr[:n_casters_fixed] = caster_indices[:n_casters_fixed]
            prog_culling_compute['u_caster_indices'].write(c_idx_arr.tobytes())
            
            prog_culling_compute['u_frustum_planes'].write(frustum_planes.astype(np.float32).tobytes())
                
            prog_culling_compute['u_n_rings'].value = n_ring_planes
            if n_ring_planes > 0:
                prog_culling_compute['u_ring_centers'].write(ring_centers_buf.astype(np.float32).tobytes())
                prog_culling_compute['u_ring_normals'].write(ring_normals_buf.astype(np.float32).tobytes())
                r_out_arr = np.zeros(16, dtype=np.float32)
                r_out_arr[:n_ring_planes] = ring_params_buf[:n_ring_planes, 1]
                prog_culling_compute['u_ring_outer_radii'].write(r_out_arr.tobytes())
                
            tracking_idx_uni = -1
            if self.camera["tracking_idx"] is not None:
                if self.camera.get("tracking_is_cmp", False):
                    tracking_idx_uni = num_bodies + self.camera["tracking_idx"]
                else:
                    tracking_idx_uni = self.camera["tracking_idx"]
            prog_culling_compute['u_tracking_idx'].value = tracking_idx_uni
            if 'u_camera_pos' in prog_culling_compute:
                prog_culling_compute['u_camera_pos'].value = tuple(cam_pos)
            if 'u_screen_height' in prog_culling_compute:
                prog_culling_compute['u_screen_height'].value = float(self.window_height)
            if 'u_fov_factor' in prog_culling_compute:
                prog_culling_compute['u_fov_factor'].value = float(1.0 / math.tan(math.radians(self.camera["fov"] / 2.0)))
            if 'u_lod_thresh_ultra' in prog_culling_compute:
                prog_culling_compute['u_lod_thresh_ultra'].value = 300.0
            if 'u_lod_thresh_hi' in prog_culling_compute:
                prog_culling_compute['u_lod_thresh_hi'].value = 40.0
            
            all_instances_buffer.bind_to_storage_buffer(binding=2)
            vis_lo_buffer.bind_to_storage_buffer(binding=3)
            vis_hi_buffer.bind_to_storage_buffer(binding=4)
            vis_ultra_buffer.bind_to_storage_buffer(binding=5)
            draw_cmds_buffer.bind_to_storage_buffer(binding=6)
            focused_mask_buffer.bind_to_storage_buffer(binding=7)
            
            _gq = _perf_gpu_begin(ctx, "gpu_culling")
            prog_culling_compute.run((total_render_bodies + 255) // 256, 1, 1)
            ctx.memory_barrier()
            _perf_gpu_end(_gq)
    
            if not show_orbits:
                n_orbits = 0
                n_orbits_hi = 0
                n_orbits_med = 0
                n_orbits_low = 0
                self.n_orbits_cmp = 0
                self.n_orbits_hi_cmp = 0
                self.n_orbits_med_cmp = 0
                self.n_orbits_low_cmp = 0
                last_orbit_pos_snap = None
                last_orbit_pos_snap_cmp = None
            else:
                recompute_orbits = True
                if last_orbit_pos_snap is not None and self.time_ctrl.get("paused"):
                    if np.array_equal(pos_snap_render, last_orbit_pos_snap):
                        recompute_orbits = False
                
                if recompute_orbits:
                    lod_levels = np.full(num_bodies, 2.0, dtype=np.float64)
                    for i in range(num_bodies):
                        if parent_snap[i] == star_idx:
                            lod_levels[i] = 1.0
                    
                    t_idx = self.camera.get("tracking_idx")
                    primary_idx = None
                    if not self.camera.get("tracking_is_cmp", False) and t_idx is not None and t_idx != star_idx and t_idx < num_bodies:
                        primary_idx = t_idx
                        if parent_snap[t_idx] >= 0 and parent_snap[parent_snap[t_idx]] == star_idx:
                            primary_idx = parent_snap[t_idx]
                    elif num_bodies > 0:
                        dists_to_cam = np.linalg.norm(pos_snap_render - cam_origin, axis=1)
                        nearest_idx = int(np.argmin(dists_to_cam))
                        if dists_to_cam[nearest_idx] < 0.2:
                            primary_idx = nearest_idx
                            if parent_snap[nearest_idx] >= 0 and parent_snap[parent_snap[nearest_idx]] == star_idx:
                                primary_idx = parent_snap[nearest_idx]

                    if primary_idx is not None:
                        lod_levels[primary_idx] = 0.0
                        if t_idx is not None and t_idx < num_bodies:
                            lod_levels[t_idx] = 0.0
                        for i in range(num_bodies):
                            if parent_snap[i] == primary_idx:
                                lod_levels[i] = 0.0
                                
                    parent_snap_render = parent_snap.copy()
                    if ephemeris_mode_active:
                        parent_snap_render[~spice_valid_mask] = -1

                    n_orbits = compute_all_orbits_batch(
                        pos_snap_render, vel_snap_render, mass_snap, parent_snap_render,
                        subsys_pos_buf, subsys_vel_buf, subsys_mass_buf,
                        visual_colors_f8, cam_origin, G, max_orbits, orbit_data_buf, lod_levels,
                        np.zeros(3, dtype='f8'))
                    last_orbit_pos_snap = pos_snap_render.copy()
                    
                    if n_orbits > 0:
                        valid_orbits = orbit_data_buf[:n_orbits]
                        lod_col = valid_orbits[:, 19]
                        
                        mask_hi = lod_col == 0.0
                        mask_med = lod_col == 1.0
                        mask_low = lod_col == 2.0
                        
                        orbits_hi = valid_orbits[mask_hi]
                        orbits_med = valid_orbits[mask_med]
                        orbits_low = valid_orbits[mask_low]
                        
                        n_orbits_hi = len(orbits_hi)
                        n_orbits_med = len(orbits_med)
                        n_orbits_low = len(orbits_low)
                        
                if self.comparison_enabled:
                    recompute_orbits_cmp = True
                    if last_orbit_pos_snap_cmp is not None and self.time_ctrl_cmp.get("paused"):
                        if np.array_equal(self.pos_snap_cmp, last_orbit_pos_snap_cmp):
                            if last_comparison_offset_au is not None and self.comparison_offset_au == last_comparison_offset_au:
                                recompute_orbits_cmp = False
                                
                    if recompute_orbits_cmp:
                        lod_levels_cmp = np.full(self.num_bodies_cmp, 2.0, dtype=np.float64)
                        for i in range(self.num_bodies_cmp):
                            if self.parent_snap_cmp[i] == self.star_idx_cmp:
                                lod_levels_cmp[i] = 1.0
                                
                        t_idx_cmp = self.camera.get("tracking_idx")
                        primary_idx_cmp = None
                        if self.camera.get("tracking_is_cmp", False) and t_idx_cmp is not None and t_idx_cmp != self.star_idx_cmp and t_idx_cmp < self.num_bodies_cmp:
                            primary_idx_cmp = t_idx_cmp
                            if self.parent_snap_cmp[t_idx_cmp] >= 0 and self.parent_snap_cmp[self.parent_snap_cmp[t_idx_cmp]] == self.star_idx_cmp:
                                primary_idx_cmp = self.parent_snap_cmp[t_idx_cmp]
                        elif self.num_bodies_cmp > 0:
                            cam_origin_cmp_pre = cam_origin - np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                            dists_to_cam_cmp = np.linalg.norm(self.pos_snap_cmp - cam_origin_cmp_pre, axis=1)
                            nearest_idx_cmp = int(np.argmin(dists_to_cam_cmp))
                            if dists_to_cam_cmp[nearest_idx_cmp] < 0.2:
                                primary_idx_cmp = nearest_idx_cmp
                                if self.parent_snap_cmp[nearest_idx_cmp] >= 0 and self.parent_snap_cmp[self.parent_snap_cmp[nearest_idx_cmp]] == self.star_idx_cmp:
                                    primary_idx_cmp = self.parent_snap_cmp[nearest_idx_cmp]

                        if primary_idx_cmp is not None:
                            lod_levels_cmp[primary_idx_cmp] = 0.0
                            if t_idx_cmp is not None and t_idx_cmp < self.num_bodies_cmp:
                                lod_levels_cmp[t_idx_cmp] = 0.0
                            for i in range(self.num_bodies_cmp):
                                if self.parent_snap_cmp[i] == primary_idx_cmp:
                                    lod_levels_cmp[i] = 0.0
                        
                        cam_origin_cmp = cam_origin - np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                        offset_vec = np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                        self.n_orbits_cmp = compute_all_orbits_batch(
                            self.pos_snap_cmp, self.vel_snap_cmp, self.mass_snap_cmp, self.parent_snap_cmp,
                            self.subsys_pos_buf_cmp, self.subsys_vel_buf_cmp, self.subsys_mass_buf_cmp,
                            self.visual_colors_f8_cmp, cam_origin_cmp, G, max_orbits, self.orbit_data_buf_cmp, lod_levels_cmp,
                            offset_vec)
                        last_orbit_pos_snap_cmp = self.pos_snap_cmp.copy()
                        last_comparison_offset_au = self.comparison_offset_au
                        
                        if self.n_orbits_cmp > 0:
                            valid_orbits_cmp = self.orbit_data_buf_cmp[:self.n_orbits_cmp]
                            lod_col_cmp = valid_orbits_cmp[:, 19]
                            
                            mask_hi_cmp = lod_col_cmp == 0.0
                            mask_med_cmp = lod_col_cmp == 1.0
                            mask_low_cmp = lod_col_cmp == 2.0
                            
                            orbits_hi_cmp = valid_orbits_cmp[mask_hi_cmp]
                            orbits_med_cmp = valid_orbits_cmp[mask_med_cmp]
                            orbits_low_cmp = valid_orbits_cmp[mask_low_cmp]
                            
                            self.n_orbits_hi_cmp = len(orbits_hi_cmp)
                            self.n_orbits_med_cmp = len(orbits_med_cmp)
                            self.n_orbits_low_cmp = len(orbits_low_cmp)
                
            # Gather all active stars in the unified scene (both primary and comparison)
            stars_pos_radius = []
            stars_colors = []
            stars_poles_obl = []
            stars_pole_colors = []
            
            for i in range(total_render_bodies):
                if all_instances[i, 8] > 0.5:
                    pos = all_instances[i, 0:3]
                    radius = all_instances[i, 6]
                    color = all_instances[i, 3:6]
                    
                    lum = 1.0
                    if i < num_bodies:
                        b_data = bodies_data[i]
                    else:
                        b_data = self.bodies_data_cmp[i - num_bodies]
                        
                    if "star_props" in b_data and "lum" in b_data["star_props"]:
                        lum = float(b_data["star_props"]["lum"])
                        
                    lum_eq = all_instances[i, 19]
                    lum_pole = all_instances[i, 23]
                    
                    if lum_eq == 0.0:
                        lum_eq = lum
                    if lum_pole == 0.0:
                        lum_pole = lum
                        
                    stars_pos_radius.append([pos[0], pos[1], pos[2], radius])
                    stars_colors.append([color[0], color[1], color[2], lum_eq])
                    
                    pole = all_instances[i, 9:12]
                    obl = all_instances[i, 12]
                    pole_color = all_instances[i, 13:16]
                    if pole_color[0] == 0.0 and pole_color[1] == 0.0 and pole_color[2] == 0.0:
                        pole_color = color # fallback
                        
                    # Calculate star R_minor relative to target_pos
                    track_idx = self.camera.get("tracking_idx")
                    if track_idx is not None and track_idx < total_render_bodies:
                        target_pos = all_instances[track_idx, 0:3]
                    else:
                        target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    
                    to_target = pos - target_pos
                    dist = np.linalg.norm(to_target)
                    if dist > 1e-6:
                        L = to_target / dist
                    else:
                        L = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    
                    p_dot_L = np.dot(pole, L)
                    p_proj_sq = 1.0 - p_dot_L**2
                    f_factor = 1.0 - obl
                    r_minor = radius * np.sqrt(max(0.0, p_dot_L**2 + (f_factor**2) * p_proj_sq))
                    
                    stars_poles_obl.append([pole[0], pole[1], pole[2], r_minor])
                    stars_pole_colors.append([pole_color[0], pole_color[1], pole_color[2], lum_pole])
                    
            num_stars = len(stars_pos_radius)
            num_stars = min(num_stars, 16)
            stars_pos_radius = stars_pos_radius[:num_stars]
            stars_colors = stars_colors[:num_stars]
            stars_poles_obl = stars_poles_obl[:num_stars]
            stars_pole_colors = stars_pole_colors[:num_stars]
            
            if num_stars == 0:
                num_stars = 1
                stars_pos_radius = [[0.0, 0.0, 0.0, SOLAR_RADII_TO_AU]]
                stars_colors = [[1.0, 1.0, 1.0, 1.0]]
                stars_poles_obl = [[0.0, 1.0, 0.0, 0.0]]
                stars_pole_colors = [[1.0, 1.0, 1.0, 1.0]]
            
            caster_data_buf[:n_casters_fixed, 0:3] = pos_rel_all[caster_indices[:n_casters_fixed]]
            caster_data_buf[:n_casters_fixed, 3] = body_radii[caster_indices[:n_casters_fixed]]
            if n_casters_fixed < 64:
                caster_data_buf[n_casters_fixed:] = 0
                
            for i_c in range(n_casters_fixed):
                b_idx = caster_indices[i_c]
                pos_c = pos_rel_all[b_idx]
                r_eq = body_radii[b_idx]
                pole = all_instances[b_idx, 9:12]
                f = all_instances[b_idx, 12]
                
                pos_star = pos_rel_all[star_idx]
                to_star = pos_star - pos_c
                dist_s = math.sqrt(to_star[0]**2 + to_star[1]**2 + to_star[2]**2)
                if dist_s > 1e-6:
                    L = to_star / dist_s
                else:
                    L = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                
                p_dot_L = np.dot(pole, L)
                p_proj_sq = 1.0 - p_dot_L**2
                f_factor = 1.0 - f
                r_minor = r_eq * np.sqrt(max(0.0, p_dot_L**2 + (f_factor**2) * p_proj_sq))
                
                caster_poles_obl_buf[i_c, 0:3] = pole
                caster_poles_obl_buf[i_c, 3] = r_minor
                
            caster_colors_buf[:n_casters_fixed, 0:3] = body_colors[caster_indices[:n_casters_fixed]]
            caster_colors_buf[:n_casters_fixed, 3] = 1.0
            if n_casters_fixed < 64:
                caster_colors_buf[n_casters_fixed:] = 0
                
            # Build atmosphere lookup dict once per frame to avoid expensive generator allocations inside the loop
            atmo_by_body = {a['body_idx']: a for a in atmo_bodies}
            
            # Calibrated Top-Of-Atmosphere appearance colors for planets & moons with atmospheres
            ATMO_TOA_COLORS = {
                'titan': (0.85, 0.55, 0.18),    # Dense orange-amber Tholin photochemical smog
                'venus': (0.96, 0.93, 0.82),    # Thick yellowish-white sulfuric acid cloud tops
                'earth': (0.35, 0.55, 0.90),    # Rayleigh blue + cloud tops + ocean reflection
                'mars': (0.55, 0.35, 0.22),     # Airborne dust haze + rusty surface
                'jupiter': (0.82, 0.74, 0.60),  # Ammonia cloud bands
                'saturn': (0.84, 0.77, 0.58),   # Pale gold ammonia/methane haze
                'uranus': (0.55, 0.84, 0.88),   # Pale cyan methane atmosphere
                'neptune': (0.28, 0.55, 0.95),  # Deep azure methane atmosphere
                'io': (0.95, 0.85, 0.30),       # Sulfur frost & volcanic haze
            }
            
            for i_c in range(n_casters_fixed):
                b_idx = caster_indices[i_c]
                atmo = atmo_by_body.get(b_idx)
                if atmo:
                    props, trans, thickness = get_cached_atmosphere_properties(atmo, mass_snap[b_idx])
                    thick_km = float(atmo.get('atmo_radius_km', 0.0) - atmo.get('planet_radius_km', 0.0))
                    scale_height_km = float(props.get('scale_height_km', 8.5))
                    caster_atmos_buf[i_c, 0:3] = trans
                    caster_atmos_buf[i_c, 3] = thick_km
                    caster_colors_buf[i_c, 3] = scale_height_km
                    
                    b_rad_km = float(bodies_data[b_idx].get('req_km', body_radii[b_idx] * 149597870.7)) if bodies_data is not None and b_idx < len(bodies_data) else 6371.0
                    path_len_m = math.sqrt(2.0 * math.pi * b_rad_km * 1000.0 * scale_height_km * 1000.0)
                    tau_o3_peak = props.get('beta_abs_layered', np.zeros(3)) * path_len_m
                    z_peak_km = float(props.get('ozone_peak_km', 25.0))
                    caster_ozone_buf[i_c, 0:3] = tau_o3_peak
                    caster_ozone_buf[i_c, 3] = z_peak_km

                    # Provide true Top-of-Atmosphere color for subpixel point light appearance
                    b_name = bodies_data[b_idx].get('name', '').lower() if (bodies_data is not None and b_idx < len(bodies_data)) else ''
                    if b_name in ATMO_TOA_COLORS:
                        caster_colors_buf[i_c, 0:3] = ATMO_TOA_COLORS[b_name]
                    elif b_name in self.texture_mean_colors:
                        caster_colors_buf[i_c, 0:3] = self.texture_mean_colors[b_name]
                    
                    caster_max_bend_buf[i_c] = compute_max_bend(
                        body_radii[b_idx], scale_height_km,
                        float(props.get('refractivity', 0.00029)))
                else:
                    caster_atmos_buf[i_c] = 0.0
                    caster_ozone_buf[i_c] = 0.0
                    caster_max_bend_buf[i_c] = 0.0
            if n_casters_fixed < 64:
                caster_atmos_buf[n_casters_fixed:] = 0
                caster_ozone_buf[n_casters_fixed:] = 0
                caster_max_bend_buf[n_casters_fixed:] = 0
    
            ubo_staging[0:16] = projection.ravel()
            ubo_staging[16:32] = view.ravel()
            
            # Write num stars
            ubo_num_stars_int_view[0] = num_stars
            ubo_staging[33:36] = 0.0 # padding
            
            # Write stars arrays
            stars_pos_radius_flat = np.zeros(64, dtype=np.float32)
            stars_colors_flat = np.zeros(64, dtype=np.float32)
            stars_poles_obl_flat = np.zeros(64, dtype=np.float32)
            stars_pole_colors_flat = np.zeros(64, dtype=np.float32)
            for s_idx in range(num_stars):
                stars_pos_radius_flat[s_idx*4 : (s_idx+1)*4] = stars_pos_radius[s_idx]
                stars_colors_flat[s_idx*4 : (s_idx+1)*4] = stars_colors[s_idx]
                stars_poles_obl_flat[s_idx*4 : (s_idx+1)*4] = stars_poles_obl[s_idx]
                stars_pole_colors_flat[s_idx*4 : (s_idx+1)*4] = stars_pole_colors[s_idx]
                
            ubo_staging[36:100] = stars_pos_radius_flat
            ubo_staging[100:164] = stars_colors_flat
            ubo_staging[164:228] = stars_poles_obl_flat
            ubo_staging[228:292] = stars_pole_colors_flat
            
            ubo_staging[292] = far
            ubo_staging[293] = depth_C
            ubo_casters_int_view = ubo_staging[294:295].view(np.int32)
            ubo_casters_int_view[0] = n_casters_fixed
            # Note: ubo_casters_int_view maps to ubo_staging[294:295]
            ubo_staging[295] = 0.0 # padding
            
            ubo_staging[296:552] = caster_data_buf.ravel()
            ubo_staging[552:808] = caster_poles_obl_buf.ravel()
            ubo_staging[808:1064] = caster_colors_buf.ravel()
            ubo_staging[1064:1320] = caster_atmos_buf.ravel()
            ubo_staging[1320:1576] = caster_ozone_buf.ravel()
            
            scene_ubo.write(ubo_staging.tobytes())
            
            fov_factor = 1.0 / math.tan(math.radians(self.camera["fov"] / 2.0))
            uniform_screen_height.value = self.fb_height
            uniform_fov_factor.value = fov_factor
            uniform_num_ring_planes.value = n_ring_planes
            if 'u_camera_pos' in prog_spheres:
                prog_spheres['u_camera_pos'].value = tuple(cam_pos)
            if uniform_caster_max_bend is not None:
                uniform_caster_max_bend.write(caster_max_bend_buf)
            if u_ring_caster_max_bend is not None:
                u_ring_caster_max_bend.write(caster_max_bend_buf)
            if n_ring_planes > 0:
                uniform_ring_centers.write(ring_centers_buf)
                uniform_ring_normals.write(ring_normals_buf)
                uniform_ring_params.write(ring_params_buf)
                if uniform_ring_colors:
                    uniform_ring_colors.write(ring_colors_buf)
                if uniform_ring_5colors:
                    uniform_ring_5colors.write(ring_5colors_buf)
                if uniform_ring_coplanar_mask is not None:
                    uniform_ring_coplanar_mask.write(ring_coplanar_mask_buf)
            
            view_rot = view.copy()
            view_rot[3, 0:3] = 0.0
    
            uniform_orbit_proj.write(projection)
            uniform_orbit_view_rot.write(view_rot)
            
            cam_world_pos_f8 = cam_origin + cam_pos_f8
            cam_pos_dvec4 = np.array([cam_world_pos_f8[0], cam_world_pos_f8[1], cam_world_pos_f8[2], 0.0], dtype='f8')
            uniform_orbit_cam_pos.write(cam_pos_dvec4)
            
            uniform_orbit_far.value = far
            uniform_orbit_depth_C.value = depth_C
    
            ring_gradient_tex.use(location=0)
            ringshine_lut_tex.use(location=6)
            ringshine_cdf_tex.use(location=7)

            if self.camera.get("ringshine_enabled", True) and n_ring_planes > 0:
                self.ringshine_map_fbo.use()
                ctx.viewport = (0, 0, 128, 1040)
                star_pos = pos_rel_all[star_idx]
                host_pos = ring_centers_buf[0]
                l_dir = star_pos - host_pos
                l_norm = np.linalg.norm(l_dir)
                if l_norm > 1e-6: l_dir /= l_norm
                else: l_dir = np.array([0.0, 1.0, 0.0], dtype='f4')
                if 'u_sun_dir' in self.prog_ringshine_map:
                    self.prog_ringshine_map['u_sun_dir'].value = tuple(l_dir.astype('f4'))
                if 'u_num_ring_planes' in self.prog_ringshine_map:
                    self.prog_ringshine_map['u_num_ring_planes'].value = n_ring_planes
                if 'u_ring_normal' in self.prog_ringshine_map:
                    self.prog_ringshine_map['u_ring_normal'].write(ring_normals_buf)
                if 'u_ring_params' in self.prog_ringshine_map:
                    self.prog_ringshine_map['u_ring_params'].write(ring_params_buf)
                for idx, r in enumerate(ring_precomputed[:n_ring_planes]):
                    if idx >= 16: break
                    if f'u_ring_planes[{idx}].unlit_factor' in self.prog_ringshine_map:
                        self.prog_ringshine_map[f'u_ring_planes[{idx}].unlit_factor'].value = float(r.get('unlit_factor', 1.0))
                    if f'u_ring_planes[{idx}].saturation' in self.prog_ringshine_map:
                        self.prog_ringshine_map[f'u_ring_planes[{idx}].saturation'].value = float(r.get('saturation', 1.0))
                    if f'u_ring_planes[{idx}].hue_shift' in self.prog_ringshine_map:
                        self.prog_ringshine_map[f'u_ring_planes[{idx}].hue_shift'].value = float(r.get('hue_shift', 0.0))
                    if f'u_ring_planes[{idx}].brightness' in self.prog_ringshine_map:
                        self.prog_ringshine_map[f'u_ring_planes[{idx}].brightness'].value = float(r.get('brightness', 1.0))
                    if f'u_ring_planes[{idx}].alpha_boost' in self.prog_ringshine_map:
                        self.prog_ringshine_map[f'u_ring_planes[{idx}].alpha_boost'].value = float(r.get('alpha_boost', 1.0))
                    if f'u_ring_planes[{idx}].is_textured' in self.prog_ringshine_map:
                        self.prog_ringshine_map[f'u_ring_planes[{idx}].is_textured'].value = 1.0 if r.get('is_textured', False) else 0.0
                if 'u_ringshine_band_count' in self.prog_ringshine_map:
                    self.prog_ringshine_map['u_ringshine_band_count'].value = int(self.camera.get("ringshine_band_count", 256))
                _gq = _perf_gpu_begin(ctx, "gpu_ringshine_map")
                self.ringshine_map_vao.render(moderngl.TRIANGLE_STRIP)
                _perf_gpu_end(_gq)

                ctx.viewport = (0, 0, self.fb_width, self.fb_height)
                self.hdr_resolve_fbo.use()

            self.ringshine_map_tex.use(location=8)
            
            # Pass Exposure and HDR setting to shaders
            exposure = self.camera.get("exposure", 1.0)
            hdr_enabled = self.camera.get("hdr_enabled", True)
            
            for prog in (prog_spheres, prog_rings, prog_atmo, prog_atmo_lowres):
                if 'u_exposure' in prog:
                    prog['u_exposure'].value = exposure
                if 'u_hdr_enabled' in prog:
                    prog['u_hdr_enabled'].value = hdr_enabled
                if 'u_planetshine_enabled' in prog:
                    prog['u_planetshine_enabled'].value = self.camera.get("planetshine_enabled", True)
                if 'u_ringshine_enabled' in prog:
                    prog['u_ringshine_enabled'].value = self.camera.get("ringshine_enabled", True)
                if 'u_ringshine_band_count' in prog:
                    prog['u_ringshine_band_count'].value = int(self.camera.get("ringshine_band_count", 10))

            inv_proj_bytes = np.linalg.inv(projection).astype('f4').tobytes()
            inv_view_bytes = np.linalg.inv(view).astype('f4').tobytes()
            for prog in (prog_atmo, prog_atmo_lowres):
                if 'u_inv_proj' in prog:
                    prog['u_inv_proj'].write(inv_proj_bytes)
                if 'u_inv_view' in prog:
                    prog['u_inv_view'].write(inv_view_bytes)
            # --- Prepare and sort atmosphere bodies ---
            sorted_atmos = []
            if atmo_quality > 0 and (atmo_bodies or (self.comparison_enabled and self.atmo_bodies_cmp)):
                atmo_dists = []
                if atmo_bodies:
                    for atmo in atmo_bodies:
                        dx = pos_rel_all[atmo['body_idx'], 0] - cam_pos[0]
                        dy = pos_rel_all[atmo['body_idx'], 1] - cam_pos[1]
                        dz = pos_rel_all[atmo['body_idx'], 2] - cam_pos[2]
                        atmo_dists.append((dx*dx + dy*dy + dz*dz, atmo, False))
                        
                if self.comparison_enabled and self.atmo_bodies_cmp:
                    for atmo in self.atmo_bodies_cmp:
                        bi_cmp = atmo['body_idx']
                        dx = cmp_pos_rel[bi_cmp, 0] - cam_pos[0]
                        dy = cmp_pos_rel[bi_cmp, 1] - cam_pos[1]
                        dz = cmp_pos_rel[bi_cmp, 2] - cam_pos[2]
                        atmo_dists.append((dx*dx + dy*dy + dz*dz, atmo, True))
                    
                sorted_atmos = sorted(atmo_dists, key=lambda x: x[0], reverse=True)
                
            if sorted_atmos:
                # Ensure all visible atmospheres have their LUTs built with a clean OpenGL state
                # This prevents issues where 'build_atmo_lut' inherits incorrect culling or blending states.
                ctx.disable(moderngl.BLEND)
                ctx.disable(moderngl.CULL_FACE)
                ctx.disable(moderngl.DEPTH_TEST)
                ctx.depth_mask = False
                
                for sq_dist, atmo, is_cmp in sorted_atmos:
                    dist_to_body = math.sqrt(sq_dist)
                    apparent_px = (atmo['atmo_radius_au'] / max(dist_to_body, 1e-12)) * self.fb_height * fov_factor
                    if apparent_px >= 2.0:
                        bi_curr = atmo['body_idx']
                        mass_sm_curr = self.mass_snap_cmp[bi_curr] if is_cmp else mass_snap[bi_curr]
                        if 'lut_tex' not in atmo or atmo.get('lut_mass') != mass_sm_curr:
                            build_atmo_lut(atmo, mass_sm_curr, is_cmp)

            # --- Active Refraction Uniform Setup ---
            refract_center = (0.0, 0.0, 0.0)
            refract_radius_km = 0.0
            refract_max_bend = 0.0
            refract_scale_height = 1.0
            refract_pole = (0.0, 1.0, 0.0)
            refract_oblateness = 0.0
            au_to_km_val = 149597870.7

            if sorted_atmos:
                closest_sq_dist, closest_atmo, is_cmp = sorted_atmos[-1]
                bi_curr = closest_atmo['body_idx']
                mass_sm_curr = self.mass_snap_cmp[bi_curr] if is_cmp else mass_snap[bi_curr]
                props_c, trans_c, thick_c = get_cached_atmosphere_properties(closest_atmo, mass_sm_curr)
                cam_dist_au = math.sqrt(closest_sq_dist)
                
                pos_rel = cmp_pos_rel if is_cmp else pos_rel_all
                refract_center = (float(pos_rel[bi_curr, 0]), float(pos_rel[bi_curr, 1]), float(pos_rel[bi_curr, 2]))
                refract_radius_km = float(closest_atmo['planet_radius_km'])
                refract_scale_height = float(props_c.get('scale_height_km', 8.5))
                refractivity = float(props_c.get('refractivity', 0.00029))
                planet_radius_au = closest_atmo['planet_radius_km'] / au_to_km_val
                if self.camera.get("refraction_enabled", True):
                    refract_max_bend = compute_max_bend(planet_radius_au, refract_scale_height, refractivity)
                    if cam_dist_au > 0.95:
                        fade = max(0.0, 1.0 - (cam_dist_au - 0.95) / 0.05)
                        refract_max_bend *= fade
                else:
                    refract_max_bend = 0.0

                b_info = self.bodies_data_cmp[bi_curr] if is_cmp else bodies_data[bi_curr]
                pole_ref = self.pole_n_arr_cmp[bi_curr] if is_cmp else self.pole_n_arr[bi_curr]
                refract_pole = (float(pole_ref[0]), float(pole_ref[1]), float(pole_ref[2]))
                refract_oblateness = float(b_info.get('oblateness', 0.0))

            for prog in (prog_spheres, prog_rings, prog_atmo, prog_atmo_lowres, prog_gpu_orbits, prog_ephem_orbits, getattr(self, 'prog_hz', None)):
                if prog is not None:
                    if 'u_refract_center' in prog:
                        prog['u_refract_center'].value = refract_center
                    if 'u_refract_radius' in prog:
                        prog['u_refract_radius'].value = refract_radius_km
                    if 'u_refract_max_bend' in prog:
                        prog['u_refract_max_bend'].value = refract_max_bend
                    if 'u_refract_scale_height' in prog:
                        prog['u_refract_scale_height'].value = refract_scale_height
                    if 'u_refract_pole' in prog:
                        prog['u_refract_pole'].value = refract_pole
                    if 'u_refract_oblateness' in prog:
                        prog['u_refract_oblateness'].value = refract_oblateness
                    if 'u_au_to_km' in prog:
                        prog['u_au_to_km'].value = au_to_km_val

            if 'u_is_cloud_pass' in prog_spheres:
                prog_spheres['u_is_cloud_pass'].value = False

            ctx.enable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.CULL_FACE)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

            # Pass 1: High/Ultra 3D meshes write opaque depth
            _gq = _perf_gpu_begin(ctx, "gpu_spheres")
            ctx.depth_mask = True
            vis_hi_buffer.bind_to_storage_buffer(binding=3)
            vao_hi.render_indirect(draw_cmds_buffer, moderngl.TRIANGLES, first=1)
            
            vis_ultra_buffer.bind_to_storage_buffer(binding=3)
            vao_ultra.render_indirect(draw_cmds_buffer, moderngl.TRIANGLES, first=2)

            # Pass 2: Low LOD / subpixel bodies test depth but do not write depth to allow smooth transit blending
            ctx.depth_mask = False
            vis_lo_buffer.bind_to_storage_buffer(binding=3)
            vao_lo.render_indirect(draw_cmds_buffer, moderngl.TRIANGLES, first=0)
            _perf_gpu_end(_gq)

            ctx.disable(moderngl.BLEND)
            ctx.disable(moderngl.CULL_FACE)


            # --- Pass 2: Orbit Lines ---
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.depth_mask = True
            
            _gq = _perf_gpu_begin(ctx, "gpu_orbits")
            def draw_orbits():
                if n_orbits > 0:
                    orbit_ssbo.bind_to_storage_buffer(binding=0)
                    orbit_ssbo_out.bind_to_storage_buffer(binding=1)
                    
                    if n_orbits_hi > 0:
                        orbit_ssbo.write(orbits_hi.tobytes(), offset=0)
                        prog_orbit_compute['u_fade_dir'].value = orbit_fade_dir
                        prog_orbit_compute['u_min_alpha'].value = orbit_min_alpha
                        prog_orbit_compute['u_orbit_res'].value = 4000
                        prog_orbit_compute['u_base_instance'].value = 0
                        prog_orbit_compute['u_max_instances'].value = n_orbits_hi
                        prog_orbit_compute['u_vertex_base_offset'].value = 0
                        prog_orbit_compute.run((n_orbits_hi * 4000 + 255) // 256, 1, 1)
                        
                    if n_orbits_med > 0:
                        orbit_ssbo.write(orbits_med.tobytes(), offset=n_orbits_hi * 160)
                        prog_orbit_compute['u_fade_dir'].value = orbit_fade_dir
                        prog_orbit_compute['u_min_alpha'].value = orbit_min_alpha
                        prog_orbit_compute['u_orbit_res'].value = 500
                        prog_orbit_compute['u_base_instance'].value = n_orbits_hi
                        prog_orbit_compute['u_max_instances'].value = n_orbits_med
                        prog_orbit_compute['u_vertex_base_offset'].value = n_orbits_hi * 4000
                        prog_orbit_compute.run((n_orbits_med * 500 + 255) // 256, 1, 1)
                        
                    if n_orbits_low > 0:
                        orbit_ssbo.write(orbits_low.tobytes(), offset=(n_orbits_hi + n_orbits_med) * 160)
                        prog_orbit_compute['u_fade_dir'].value = orbit_fade_dir
                        prog_orbit_compute['u_min_alpha'].value = orbit_min_alpha
                        prog_orbit_compute['u_orbit_res'].value = 100
                        prog_orbit_compute['u_base_instance'].value = n_orbits_hi + n_orbits_med
                        prog_orbit_compute['u_max_instances'].value = n_orbits_low
                        prog_orbit_compute['u_vertex_base_offset'].value = n_orbits_hi * 4000 + n_orbits_med * 500
                        prog_orbit_compute.run((n_orbits_low * 100 + 255) // 256, 1, 1)

                    ctx.memory_barrier()

                    prog_gpu_orbits['u_cam_pos_double'].value = (cam_world_pos_f8[0], cam_world_pos_f8[1], cam_world_pos_f8[2], 1.0)
                    
                    if n_orbits_hi > 0:
                        prog_gpu_orbits['u_orbit_res'].value = 4000
                        prog_gpu_orbits['u_base_instance'].value = 0
                        prog_gpu_orbits['u_vertex_base_offset'].value = 0
                        vao_gpu_orbits.render(moderngl.LINE_STRIP, vertices=4000, instances=n_orbits_hi)
                        
                    if n_orbits_med > 0:
                        prog_gpu_orbits['u_orbit_res'].value = 500
                        prog_gpu_orbits['u_base_instance'].value = n_orbits_hi
                        prog_gpu_orbits['u_vertex_base_offset'].value = n_orbits_hi * 4000
                        vao_gpu_orbits.render(moderngl.LINE_STRIP, vertices=500, instances=n_orbits_med)
                        
                    if n_orbits_low > 0:
                        prog_gpu_orbits['u_orbit_res'].value = 100
                        prog_gpu_orbits['u_base_instance'].value = n_orbits_hi + n_orbits_med
                        prog_gpu_orbits['u_vertex_base_offset'].value = n_orbits_hi * 4000 + n_orbits_med * 500
                        vao_gpu_orbits.render(moderngl.LINE_STRIP, vertices=100, instances=n_orbits_low)

                if self.comparison_enabled and self.n_orbits_cmp > 0:
                    orbit_ssbo.bind_to_storage_buffer(binding=0)
                    orbit_ssbo_out.bind_to_storage_buffer(binding=1)

                    if self.n_orbits_hi_cmp > 0:
                        orbit_ssbo.write(orbits_hi_cmp.tobytes(), offset=0)
                        prog_orbit_compute['u_fade_dir'].value = orbit_fade_dir
                        prog_orbit_compute['u_min_alpha'].value = orbit_min_alpha
                        prog_orbit_compute['u_orbit_res'].value = 4000
                        prog_orbit_compute['u_base_instance'].value = 0
                        prog_orbit_compute['u_max_instances'].value = self.n_orbits_hi_cmp
                        prog_orbit_compute['u_vertex_base_offset'].value = 0
                        prog_orbit_compute.run((self.n_orbits_hi_cmp * 4000 + 255) // 256, 1, 1)

                    if self.n_orbits_med_cmp > 0:
                        orbit_ssbo.write(orbits_med_cmp.tobytes(), offset=self.n_orbits_hi_cmp * 160)
                        prog_orbit_compute['u_fade_dir'].value = orbit_fade_dir
                        prog_orbit_compute['u_min_alpha'].value = orbit_min_alpha
                        prog_orbit_compute['u_orbit_res'].value = 500
                        prog_orbit_compute['u_base_instance'].value = self.n_orbits_hi_cmp
                        prog_orbit_compute['u_max_instances'].value = self.n_orbits_med_cmp
                        prog_orbit_compute['u_vertex_base_offset'].value = self.n_orbits_hi_cmp * 4000
                        prog_orbit_compute.run((self.n_orbits_med_cmp * 500 + 255) // 256, 1, 1)

                    if self.n_orbits_low_cmp > 0:
                        orbit_ssbo.write(orbits_low_cmp.tobytes(), offset=(self.n_orbits_hi_cmp + self.n_orbits_med_cmp) * 160)
                        prog_orbit_compute['u_fade_dir'].value = orbit_fade_dir
                        prog_orbit_compute['u_min_alpha'].value = orbit_min_alpha
                        prog_orbit_compute['u_orbit_res'].value = 100
                        prog_orbit_compute['u_base_instance'].value = self.n_orbits_hi_cmp + self.n_orbits_med_cmp
                        prog_orbit_compute['u_max_instances'].value = self.n_orbits_low_cmp
                        prog_orbit_compute['u_vertex_base_offset'].value = self.n_orbits_hi_cmp * 4000 + self.n_orbits_med_cmp * 500
                        prog_orbit_compute.run((self.n_orbits_low_cmp * 100 + 255) // 256, 1, 1)
                        
                    ctx.memory_barrier()

                    prog_gpu_orbits['u_cam_pos_double'].value = (cam_world_pos_f8[0], cam_world_pos_f8[1], cam_world_pos_f8[2], 1.0)
                    
                    if self.n_orbits_hi_cmp > 0:
                        prog_gpu_orbits['u_orbit_res'].value = 4000
                        prog_gpu_orbits['u_base_instance'].value = 0
                        prog_gpu_orbits['u_vertex_base_offset'].value = 0
                        vao_gpu_orbits.render(moderngl.LINE_STRIP, vertices=4000, instances=self.n_orbits_hi_cmp)
                        
                    if self.n_orbits_med_cmp > 0:
                        prog_gpu_orbits['u_orbit_res'].value = 500
                        prog_gpu_orbits['u_base_instance'].value = self.n_orbits_hi_cmp
                        prog_gpu_orbits['u_vertex_base_offset'].value = self.n_orbits_hi_cmp * 4000
                        vao_gpu_orbits.render(moderngl.LINE_STRIP, vertices=500, instances=self.n_orbits_med_cmp)
                        
                    if self.n_orbits_low_cmp > 0:
                        prog_gpu_orbits['u_orbit_res'].value = 100
                        prog_gpu_orbits['u_base_instance'].value = self.n_orbits_hi_cmp + self.n_orbits_med_cmp
                        prog_gpu_orbits['u_vertex_base_offset'].value = self.n_orbits_hi_cmp * 4000 + self.n_orbits_med_cmp * 500
                        vao_gpu_orbits.render(moderngl.LINE_STRIP, vertices=100, instances=self.n_orbits_low_cmp)

            if show_orbits and (n_orbits > 0 or (self.comparison_enabled and self.n_orbits_cmp > 0)):
                if self.orbit_msaa_fbo is not None:
                    # Orbit lines get their own MSAA buffer; the resolved result is
                    # blended back over the (non-MSAA) scene afterwards.
                    self.orbit_msaa_fbo.use()
                    ctx.viewport = (0, 0, self.fb_width, self.fb_height)
                    ctx.clear(0.0, 0.0, 0.0, 0.0)
                    if self.depth_texture:
                        self.depth_texture.use(location=9)
                    prog_gpu_orbits['u_manual_occlusion'].value = 1
                    draw_orbits()
                    ctx.copy_framebuffer(self.orbit_resolve_fbo, self.orbit_msaa_fbo)
                    self.hdr_resolve_fbo.use()
                    ctx.viewport = (0, 0, self.fb_width, self.fb_height)
                    ctx.disable(moderngl.DEPTH_TEST)
                    self.orbit_resolved_tex.use(location=2)
                    self.prog_atmo_composite['u_atmo_texture'].value = 2
                    self.quad_vao_atmo_comp.render(moderngl.TRIANGLE_STRIP)
                    ctx.enable(moderngl.DEPTH_TEST)
                else:
                    prog_gpu_orbits['u_manual_occlusion'].value = 0
                    draw_orbits()
            _perf_gpu_end(_gq)

            def render_atmosphere_pass(clip_mode, atmos_to_render, is_lowres=False):
                if not atmos_to_render:
                    return
                if not is_lowres:
                    ctx.enable(moderngl.BLEND)
                    gl.glBlendFunc(gl.GL_ONE, gl.GL_SRC1_COLOR)
                else:
                    ctx.disable(moderngl.BLEND)
                ctx.depth_func = '<='
                ctx.enable(moderngl.CULL_FACE)
                ctx.cull_face = 'front'
                ctx.disable(moderngl.DEPTH_TEST)
                ctx.depth_mask = False
                
                cur_prog = prog_atmo_lowres if is_lowres else prog_atmo
                cur_vao = vao_atmo_lowres if is_lowres else vao_atmo
    
                if 'u_atmo_quality' in cur_prog: cur_prog['u_atmo_quality'].value = atmo_quality
                if 'u_stochastic_noise' in cur_prog: cur_prog['u_stochastic_noise'].value = self.camera.get("atmo_stochastic", True)
                if 'u_camera_pos' in cur_prog: cur_prog['u_camera_pos'].write(cam_pos)
                AU_TO_KM = 149597870.7
            
                if 'u_num_ring_planes' in cur_prog: cur_prog['u_num_ring_planes'].value = n_ring_planes
                if n_ring_planes > 0:
                    if 'u_ring_center' in cur_prog: cur_prog['u_ring_center'].write(ring_centers_buf)
                    if 'u_ring_normal' in cur_prog: cur_prog['u_ring_normal'].write(ring_normals_buf)
                    if 'u_ring_params' in cur_prog: cur_prog['u_ring_params'].write(ring_params_buf)
                    if 'u_ring_coplanar_mask' in cur_prog: cur_prog['u_ring_coplanar_mask'].write(ring_coplanar_mask_buf.tobytes())
                    ring_gradient_tex.use(location=0)
                
                if 'u_atmo_clip_mode' in cur_prog:
                    cur_prog['u_atmo_clip_mode'].value = clip_mode
                self.atmo_ssbo.bind_to_storage_buffer(binding=8)
    
                # Pre-build lookup tables outside the loop
                atmo_lookup = {a['body_idx']: a for a in atmo_bodies}
                atmo_lookup_cmp = {a['body_idx']: a for a in self.atmo_bodies_cmp} if self.comparison_enabled else {}

                # Pre-allocate scratch arrays outside the loop
                active_casters_buf = np.zeros((8, 4), dtype='f4')
                active_poles_obl_buf = np.zeros((8, 4), dtype='f4')
                active_caster_r_minor_buf = np.zeros(8, dtype='f4')
                active_atmos_buf = np.zeros((8, 4), dtype='f4')
                active_max_bend_buf = np.zeros(8, dtype='f4')
                active_caster_ozone_buf = np.zeros((8, 4), dtype='f4')

                for sq_dist, atmo, is_cmp in atmos_to_render:
                    bi = atmo['body_idx']
                    if is_cmp:
                        body_pos_rel = cmp_pos_rel[bi].astype('f4')
                        body_idx_in_unified = num_bodies + bi
                    else:
                        body_pos_rel = pos_rel_all[bi]
                        body_idx_in_unified = bi
    
                    dist_to_body = math.sqrt(sq_dist)
                    apparent_px = (atmo['atmo_radius_au'] / max(dist_to_body, 1e-12)) * self.fb_height * fov_factor
                    if apparent_px < 2.0:
                        continue
    
                    if atmo_quality == 1:
                        n_samples = max(4, self.camera.get("atmo_steps_max", 32) // 2)
                    else:
                        n_samples = self.camera.get("atmo_steps_max", 32)
                            
                    if is_cmp:
                        mass_sm = self.mass_snap_cmp[atmo['body_idx']]
                    else:
                        mass_sm = mass_snap[atmo['body_idx']]
                    
                    # LUT should already be built cleanly before the pass
                    if 'lut_tex' in atmo and atmo['lut_tex']:
                        atmo['lut_tex'].use(location=1)
                    if 'lut_multi_scatter' in atmo:
                        atmo['lut_multi_scatter'].use(location=3)
                    
                    props, _, _ = get_cached_atmosphere_properties(atmo, mass_sm)
                    scaled_intensity = atmo['intensity']
                    beta_mie_val = atmo.get('beta_mie', 2.0e-6)
                    beta_mie_coeffs = compute_mie_coefficients(beta_mie_val, atmo.get('mie_angstrom', None)).astype('f4')
                    
                    f = float(all_instances[body_idx_in_unified, 12])
                    f_scale = 1.0 / (1.0 - f) if f < 1.0 else 1.0

                    # CPU-side active neighbor culling (parent and children)
                    active_indices = []
                    if is_cmp:
                        p_snap = self.parent_snap_cmp
                        p_idx = int(p_snap[bi]) if bi < len(p_snap) else -1
                        if p_idx >= 0 and p_idx != self.star_idx_cmp:
                            active_indices.append(num_bodies + p_idx)
                        for child_bi in range(self.num_bodies_cmp):
                            if child_bi < len(p_snap) and int(p_snap[child_bi]) == bi:
                                active_indices.append(num_bodies + child_bi)
                    else:
                        p_snap = parent_snap
                        p_idx = int(p_snap[bi]) if bi < len(p_snap) else -1
                        if p_idx >= 0 and p_idx != star_idx:
                            active_indices.append(p_idx)
                        for child_bi in range(num_bodies):
                            if child_bi < len(p_snap) and int(p_snap[child_bi]) == bi:
                                active_indices.append(child_bi)
                                
                    # CPU-side geometric eclipse culling: only keep casters that can geometrically cast an eclipse on this atmosphere
                    if active_indices:
                        pos_star = cmp_pos_rel[self.star_idx_cmp] if is_cmp else pos_rel_all[star_idx]
                        to_star = pos_star - body_pos_rel
                        dist_s_sq = float(to_star[0]**2 + to_star[1]**2 + to_star[2]**2)
                        if dist_s_sq > 1e-12:
                            dist_s = math.sqrt(dist_s_sq)
                            L_star = to_star / dist_s
                            star_rad = float(self.body_radii_cmp[self.star_idx_cmp] if is_cmp else body_radii[star_idx])
                            star_tan = star_rad / max(dist_s, 1e-6)
                            atmo_rad_au = float(atmo['atmo_radius_au'])

                            filtered_active = []
                            for idx_u in active_indices:
                                is_c_cmp = (idx_u >= num_bodies)
                                c_local_idx = idx_u - num_bodies if is_c_cmp else idx_u
                                pos_c = cmp_pos_rel[c_local_idx] if is_c_cmp else pos_rel_all[c_local_idx]
                                rad_c = float(self.body_radii_cmp[c_local_idx] if is_c_cmp else body_radii[c_local_idx])

                                D = pos_c - body_pos_rel
                                t_proj = float(D[0] * L_star[0] + D[1] * L_star[1] + D[2] * L_star[2])
                                if t_proj <= 0.0:
                                    continue # Caster is on the night side of planet

                                d_sq = float(D[0]**2 + D[1]**2 + D[2]**2)
                                perp_sq = max(0.0, d_sq - t_proj * t_proj)
                                r_penumbra_max = rad_c * 1.5 + atmo_rad_au + t_proj * star_tan * 1.5
                                if perp_sq <= r_penumbra_max * r_penumbra_max:
                                    filtered_active.append(idx_u)
                            active_indices = filtered_active

                    n_active = min(len(active_indices), 8)
                    active_indices = active_indices[:n_active]
                    
                    # Clear scratch buffers in-place
                    active_casters_buf[:] = 0.0
                    active_poles_obl_buf[:] = 0.0
                    active_caster_r_minor_buf[:] = 0.0
                    active_atmos_buf[:] = 0.0
                    active_max_bend_buf[:] = 0.0
                    active_caster_ozone_buf[:] = 0.0
                    
                    for i_ac, idx_u in enumerate(active_indices):
                        is_c_cmp = (idx_u >= num_bodies)
                        c_local_idx = idx_u - num_bodies if is_c_cmp else idx_u
                        
                        pos_c = cmp_pos_rel[c_local_idx] if is_c_cmp else pos_rel_all[c_local_idx]
                        rad_c = self.body_radii_cmp[c_local_idx] if is_c_cmp else body_radii[c_local_idx]
                        
                        active_casters_buf[i_ac, 0:3] = pos_c
                        active_casters_buf[i_ac, 3] = rad_c
                        
                        fc = all_instances[idx_u, 12]
                        fc_scale = 1.0 / (1.0 - fc) if fc < 1.0 else 1.0
                        active_poles_obl_buf[i_ac, 0:3] = all_instances[idx_u, 9:12]
                        active_poles_obl_buf[i_ac, 3] = fc_scale
                        
                        pos_star = cmp_pos_rel[self.star_idx_cmp] if is_c_cmp else pos_rel_all[star_idx]
                        to_star = pos_star - pos_c
                        dist_s = math.sqrt(to_star[0]**2 + to_star[1]**2 + to_star[2]**2)
                        if dist_s > 1e-6:
                            L = to_star / dist_s
                        else:
                            L = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                        
                        p_dot_L = np.dot(all_instances[idx_u, 9:12], L)
                        p_proj_sq = 1.0 - p_dot_L**2
                        fc_factor = 1.0 - fc
                        r_minor = rad_c * np.sqrt(max(0.0, p_dot_L**2 + (fc_factor**2) * p_proj_sq))
                        active_caster_r_minor_buf[i_ac] = r_minor
                        
                        lookup = atmo_lookup_cmp if is_c_cmp else atmo_lookup
                        atmo_info = lookup.get(c_local_idx)
                        if atmo_info:
                            mass_sm_c = self.mass_snap_cmp[c_local_idx] if is_c_cmp else mass_snap[c_local_idx]
                            props_c, trans_c, thick_c = get_cached_atmosphere_properties(atmo_info, mass_sm_c)
                            thick_km = float(atmo_info.get('atmo_radius_km', 0.0) - atmo_info.get('planet_radius_km', 0.0))
                            scale_height_km = float(props_c.get('scale_height_km', 8.5))
                            active_atmos_buf[i_ac, 0:3] = trans_c
                            active_atmos_buf[i_ac, 3] = thick_km
                            active_max_bend_buf[i_ac] = compute_max_bend(
                                rad_c, scale_height_km,
                                float(props_c.get('refractivity', 0.00029)))
                            r_c_km = float(atmo_info.get('planet_radius_km', rad_c * 149597870.7))
                            path_len_m_c = math.sqrt(2.0 * math.pi * r_c_km * 1000.0 * scale_height_km * 1000.0)
                            tau_o3_c = props_c.get('beta_abs_layered', np.zeros(3)) * path_len_m_c
                            active_caster_ozone_buf[i_ac, 0:3] = tau_o3_c
                            active_caster_ozone_buf[i_ac, 3] = float(props_c.get('ozone_peak_km', 25.0))
                            
                    # Write all per-body atmosphere parameters into self.atmo_staging (std430 alignment)
                    self.atmo_staging[0:3] = body_pos_rel
                    self.atmo_staging[3] = float(atmo['atmo_radius_au'])
                    self.atmo_staging[4:7] = props['beta_rayleigh']
                    self.atmo_staging[7] = props['scale_height_km']
                    self.atmo_staging[8:11] = beta_mie_coeffs
                    self.atmo_staging[11] = atmo.get('h_mie', 1.2)
                    self.atmo_staging[12:15] = props['beta_abs_mixed']
                    self.atmo_staging[15] = atmo.get('mie_g', 0.758)
                    self.atmo_staging[16:19] = props['beta_abs_layered']
                    self.atmo_staging[19] = scaled_intensity
                    self.atmo_staging[20] = float(atmo['planet_radius_km'])
                    self.atmo_staging[21] = float(atmo['atmo_radius_km'])
                    self.atmo_staging[22] = AU_TO_KM
                    self.atmo_staging_int_view[23] = n_samples
                    self.atmo_staging[24:27] = all_instances[body_idx_in_unified, 9:12]
                    self.atmo_staging[27] = f_scale
                    self.atmo_staging_int_view[28] = n_active
                    self.atmo_staging_int_view[29] = 1 if self.camera.get("atmo_adaptive_steps", True) else 0
                    self.atmo_staging_int_view[30] = body_idx_in_unified
                    self.atmo_staging[31] = float(self.frame_counter % 1024)
                    self.atmo_staging[32:35] = np.asarray(atmo.get('mie_albedo', [1.0, 1.0, 1.0]), dtype=np.float32)
                    # index 35: u_refractivity (surface n_mix - 1, drives eclipse refraction)
                    self.atmo_staging[35] = float(props.get('refractivity', 0.00029))
                    self.atmo_staging[36:68] = active_casters_buf.ravel()
                    self.atmo_staging[68:100] = active_poles_obl_buf.ravel()
                    self.atmo_staging[100:108] = active_caster_r_minor_buf
                    self.atmo_staging[108:140] = active_atmos_buf.ravel()
                    self.atmo_staging[140:148] = active_max_bend_buf
                    self.atmo_staging[148:180] = active_caster_ozone_buf.ravel()
                    self.atmo_staging[180:184] = [float(props.get('ozone_peak_km', 25.0)),
                                                  float(props.get('ozone_width_km', 8.0)),
                                                  0.0, float(self.camera.get('atmo_adaptive_steps_max', 128))]  # pad to 736 bytes

                    # Inner clip radius for the atmosphere march: the planet is an
                    # icosphere whose flat faces sag below the analytic ellipsoid,
                    # leaving an empty band between the surface and the atmosphere
                    # bottom at the limb.  Shrink the clip sphere by the max sag of
                    # the mesh LOD that body actually renders with (same rule as the
                    # culling compute shader) so the atmosphere covers the polyhedron.
                    _atmo_rad_au = self.body_radii_cmp[bi] if is_cmp else body_radii[bi]
                    _apparent_px = (float(_atmo_rad_au) / max(dist_to_body, 1e-12)) * self.window_height * fov_factor
                    _tracking_idx_uni = -1
                    if self.camera["tracking_idx"] is not None:
                        _tracking_idx_uni = self.camera["tracking_idx"] + (num_bodies if self.camera.get("tracking_is_cmp", False) else 0)
                    if (body_idx_in_unified == _tracking_idx_uni) or (_apparent_px >= 300.0):
                        _lod_subdiv = 6
                    elif _apparent_px >= 40.0:
                        _lod_subdiv = 4
                    else:
                        _lod_subdiv = 1
                    self.atmo_staging[182] = float(atmo['planet_radius_km']) * (1.0 - _icosphere_max_sag(_lod_subdiv))

                    # Precomputed optical constants
                    _inv_h_r = 1.0 / max(1e-3, float(props['scale_height_km']))
                    _inv_h_m = 1.0 / max(1e-3, float(atmo.get('h_mie', 1.2)))
                    _inv_oz = 1.0 / max(1e-3, float(props.get('ozone_width_km', 8.0)))
                    _max_b = float(np.clip(2.0 * max(float(props.get('refractivity', 0.00029)), 0.0) * math.sqrt(math.pi * float(atmo['planet_radius_km']) / max(1e-6, float(props['scale_height_km']) * 2.0)), 0.001, 0.05))
                    self.atmo_staging[184:188] = [_inv_h_r, _inv_h_m, _inv_oz, _max_b]

                    # Precomputed Henyey-Greenstein Mie phase constants
                    _g = float(np.clip(float(atmo.get('mie_g', 0.758)), 0.0, 0.88))
                    _g2 = _g * _g
                    _c1 = (3.0 / (8.0 * math.pi)) * ((1.0 - _g2) / (2.0 + _g2))
                    _c2 = 1.0 + _g2
                    _c3 = 2.0 * _g
                    self.atmo_staging[188:192] = [_c1, _c2, _c3, 0.0]

                    # Precomputed star parameters (up to 4 stars)
                    self.atmo_staging[192:256] = 0.0
                    _b_pole_raw = all_instances[body_idx_in_unified, 9:12]
                    _b_pole_len = float(np.linalg.norm(_b_pole_raw))
                    _p_pole_norm = _b_pole_raw / _b_pole_len if _b_pole_len > 1e-4 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

                    _n_stars_to_precomp = min(len(stars_pos_radius), 4)
                    for _s in range(_n_stars_to_precomp):
                        _s_pos = np.array(stars_pos_radius[_s][0:3], dtype=np.float32)
                        _s_rad = float(stars_pos_radius[_s][3])

                        _f_to_s = _s_pos - body_pos_rel
                        _d_star = float(np.linalg.norm(_f_to_s))
                        _L_star = _f_to_s / _d_star if _d_star > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

                        _pole_s = np.array(stars_poles_obl[_s][0:3], dtype=np.float32)
                        _pole_s_len = float(np.linalg.norm(_pole_s))
                        _pole_s_dir = _pole_s / _pole_s_len if _pole_s_len > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
                        _star_sin_lat = abs(float(np.dot(_L_star, _pole_s_dir)))

                        _eq_col = np.array(stars_colors[_s][0:3], dtype=np.float32)
                        _eq_lum = float(stars_colors[_s][3])
                        _pol_col = np.array(stars_pole_colors[_s][0:3], dtype=np.float32)
                        _pol_lum = float(stars_pole_colors[_s][3])

                        _star_col = _eq_col * (1.0 - _star_sin_lat) + _pol_col * _star_sin_lat
                        _star_lum = _eq_lum * (1.0 - _star_sin_lat) + _pol_lum * _star_sin_lat

                        _sin_s = _s_rad / max(_d_star, _s_rad + 1e-6)
                        _eff_s_rad = _sin_s + _max_b
                        _cos_s_eff = math.sqrt(max(0.0, 1.0 - _eff_s_rad * _eff_s_rad))

                        _irrad = _star_lum / max(_d_star * _d_star, 1e-8)
                        _comb_int = _star_col * (_irrad * scaled_intensity * math.pi)

                        _sun_pos_loc = _f_to_s * AU_TO_KM

                        # toSphericalSpace of L
                        _p_proj = float(np.dot(_L_star, _p_pole_norm))
                        _p_perp = _L_star - _p_proj * _p_pole_norm
                        _s_dir_sph = _p_perp * f_scale + _p_proj * _p_pole_norm
                        _s_dir_sph_len = float(np.linalg.norm(_s_dir_sph))
                        _s_dir_sph = _s_dir_sph / _s_dir_sph_len if _s_dir_sph_len > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

                        _s_pole_dot = float(np.dot(_s_dir_sph, _p_pole_norm))
                        _cosPhi = abs(_s_pole_dot)
                        _tanPhi = _cosPhi / math.sqrt(max(1.0 - _cosPhi * _cosPhi, 1e-4))
                        _solstice_f = float(np.clip(_tanPhi * 1.8, 0.0, 1.0))

                        # Pack into staging buffer
                        _b_idx = 192 + _s * 4
                        self.atmo_staging[_b_idx : _b_idx + 3] = _comb_int
                        self.atmo_staging[_b_idx + 3] = _sin_s

                        _b_idx = 208 + _s * 4
                        self.atmo_staging[_b_idx : _b_idx + 3] = _s_dir_sph
                        self.atmo_staging[_b_idx + 3] = _cos_s_eff

                        _b_idx = 224 + _s * 4
                        self.atmo_staging[_b_idx : _b_idx + 3] = _sun_pos_loc
                        self.atmo_staging[_b_idx + 3] = _eff_s_rad

                        _b_idx = 240 + _s * 4
                        self.atmo_staging[_b_idx : _b_idx + 4] = [_solstice_f, _s_pole_dot, _d_star, _s_rad]

                    # Single buffer upload per atmosphere body
                    self.atmo_ssbo.write(self.atmo_staging.tobytes())
                    
                    cur_vao.render(moderngl.TRIANGLES)
    
                ctx.depth_mask = True
                ctx.enable(moderngl.DEPTH_TEST)
                ctx.enable(moderngl.CULL_FACE)
                ctx.cull_face = 'back'
                ctx.depth_func = '<'
                ctx.disable(moderngl.BLEND)

            def execute_atmosphere_pass(clip_mode, target_list=None):
                if not self.camera.get("atmo_enabled", True):
                    return
                atmos_to_march = sorted_atmos if target_list is None else target_list
                if not atmos_to_march:
                    return
                atmo_res = 1.0 if getattr(self, "_screenshot_capturing", False) else float(self.camera.get("atmo_resolution", 1.0))
                use_vrs = atmo_res < 0.999 and getattr(self, "atmo_lowres_fbo", None) is not None
                
                pending_lowres_atmos = []
                
                def flush_lowres():
                    if not pending_lowres_atmos:
                        return
                        
                    # 1. Low-Res Pass (Inner Region)
                    self.atmo_lowres_fbo.use()
                    ctx.viewport = (0, 0, self.atmo_lowres_fbo.width, self.atmo_lowres_fbo.height)
                    self.atmo_lowres_fbo.clear(color=(0.0, 0.0, 0.0, 0.0))
                    
                    low_size = (float(self.atmo_lowres_fbo.width), float(self.atmo_lowres_fbo.height))
                    if 'u_lowres_size' in prog_atmo_lowres:
                        prog_atmo_lowres['u_lowres_size'].value = low_size
                    if 'u_lowres_size' in prog_atmo:
                        prog_atmo['u_lowres_size'].value = low_size
                    if 'u_screen_res' in prog_atmo_lowres:
                        prog_atmo_lowres['u_screen_res'].value = low_size
                    if self.depth_texture:
                        self.depth_texture.use(location=9)
                        
                    render_atmosphere_pass(clip_mode, pending_lowres_atmos, is_lowres=True)
                    
                    # 2. Masked Upsample Pass (Composites inner region to high-res FBO)
                    self.hdr_resolve_fbo.use()
                    ctx.viewport = (0, 0, self.fb_width, self.fb_height)
                    ctx.enable(moderngl.BLEND)
                    gl.glBlendFunc(gl.GL_ONE, gl.GL_SRC1_COLOR)
                    ctx.disable(moderngl.DEPTH_TEST)
                    ctx.depth_mask = False
                    
                    self.atmo_lowres_scatter_tex.use(location=0)
                    self.atmo_lowres_trans_tex.use(location=1)
                    if self.depth_texture:
                        self.depth_texture.use(location=9)
                        
                    if 'u_depth_C' in self.prog_atmo_upsample:
                        self.prog_atmo_upsample['u_depth_C'].value = depth_C
                    if 'u_far' in self.prog_atmo_upsample:
                        self.prog_atmo_upsample['u_far'].value = far
                    if 'u_lowres_size' in self.prog_atmo_upsample:
                        self.prog_atmo_upsample['u_lowres_size'].value = (float(self.atmo_lowres_fbo.width), float(self.atmo_lowres_fbo.height))
                    if 'u_vrs_enabled' in self.prog_atmo_upsample:
                        self.prog_atmo_upsample['u_vrs_enabled'].value = True
                        
                    self.quad_vao_atmo_upsample.render(moderngl.TRIANGLE_STRIP)
                    
                    if 'u_vrs_enabled' in self.prog_atmo_upsample:
                        self.prog_atmo_upsample['u_vrs_enabled'].value = False
                        
                    # 3. High-Res Pass (Edge Region)
                    # Stay bound to hdr_resolve_fbo
                    self.atmo_lowres_trans_tex.use(location=2)
                    if 'u_lowres_trans' in prog_atmo:
                        prog_atmo['u_lowres_trans'].value = 2
                    if 'u_vrs_highres_pass' in prog_atmo:
                        prog_atmo['u_vrs_highres_pass'].value = True
                    if 'u_screen_res' in prog_atmo:
                        prog_atmo['u_screen_res'].value = (float(self.fb_width), float(self.fb_height))
                        
                    render_atmosphere_pass(clip_mode, pending_lowres_atmos, is_lowres=False)
                    
                    if 'u_vrs_highres_pass' in prog_atmo:
                        prog_atmo['u_vrs_highres_pass'].value = False
                        
                    ctx.depth_mask = True
                    ctx.enable(moderngl.DEPTH_TEST)
                    ctx.disable(moderngl.BLEND)
                    
                    pending_lowres_atmos.clear()

                for sq_dist, atmo, is_cmp in atmos_to_march:
                    dist_to_body = math.sqrt(sq_dist)
                    apparent_px = (atmo['atmo_radius_au'] / max(dist_to_body, 1e-12)) * self.fb_height * fov_factor
                    
                    vrs_threshold = float(self.camera.get("atmo_vrs_threshold_px", 100.0))
                    is_this_lowres = use_vrs and apparent_px >= vrs_threshold
                    
                    if is_this_lowres:
                        pending_lowres_atmos.append((sq_dist, atmo, is_cmp))
                    else:
                        flush_lowres()
                        
                        # Render entirely at high-res
                        self.hdr_resolve_fbo.use()
                        ctx.viewport = (0, 0, self.fb_width, self.fb_height)
                        if 'u_screen_res' in prog_atmo:
                            prog_atmo['u_screen_res'].value = (float(self.fb_width), float(self.fb_height))
                        if 'u_vrs_highres_pass' in prog_atmo:
                            prog_atmo['u_vrs_highres_pass'].value = False
                        if self.depth_texture:
                            self.depth_texture.use(location=9)
                            
                        render_atmosphere_pass(clip_mode, [(sq_dist, atmo, is_cmp)], is_lowres=False)
                        
                flush_lowres()

            # Partition atmospheres into those requiring ring clipping vs ringless
            def _body_needs_ring_clip(entry):
                if n_ring_planes == 0:
                    return False
                _sq, _a, _is_c = entry
                _bi = _a['body_idx']
                if _is_c:
                    if any(g.get('body_idx') == _bi for g in getattr(self, 'ring_render_groups_cmp', [])):
                        return True
                else:
                    if _bi in self.body_ring_indices:
                        return True
                _b_pos = cmp_pos_rel[_bi] if _is_c else pos_rel_all[_bi]
                _a_rad = float(_a['atmo_radius_au'])
                for _k in range(n_ring_planes):
                    _c_ring = ring_centers_buf[_k, 0:3]
                    _d_vec = _b_pos - _c_ring
                    _d_sq = float(_d_vec[0]**2 + _d_vec[1]**2 + _d_vec[2]**2)
                    _r_outer = float(ring_params_buf[_k, 1])
                    _reach = _r_outer + _a_rad * 2.0
                    if _d_sq < _reach * _reach:
                        return True
                return False

            ringed_atmos = [a for a in sorted_atmos if _body_needs_ring_clip(a)]
            ringless_atmos = [a for a in sorted_atmos if not _body_needs_ring_clip(a)]

            # --- Pass 1: Atmosphere behind rings ---
            _gq = _perf_gpu_begin(ctx, "gpu_atmo_behind")
            if ringed_atmos:
                execute_atmosphere_pass(1, ringed_atmos)
            _perf_gpu_end(_gq)
    
            # --- Render Rings ---
            _gq = _perf_gpu_begin(ctx, "gpu_rings")
            if ring_render_groups or (self.comparison_enabled and self.ring_render_groups_cmp):
                ctx.disable(moderngl.CULL_FACE)
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
                u_ring_camera_pos.write(cam_pos)
                ctx.depth_mask = False
                u_ring_clip_mode.value = 0
                if u_ring_planetshine_enabled is not None:
                    u_ring_planetshine_enabled.value = self.camera.get("planetshine_enabled", True)
                ring_gradient_tex.use(location=0)
                
                if ring_render_groups:
                    for group in ring_render_groups:
                        bi = group['body_idx']
                        body_pos_rel = pos_rel_all[bi]
                        u_ring_body_offset.write(body_pos_rel)
                        u_ring_host_pos.write(body_pos_rel)
                        u_ring_host_radius.value = float(body_radii[bi])
                        if u_ring_host_color is not None:
                            u_ring_host_color.value = tuple(float(c) for c in body_colors[bi])
                        f = float(all_instances[bi, 12])
                        f_scale = 1.0 / (1.0 - f) if f < 1.0 else 1.0
                        u_ring_host_pole_obl.value = (
                            float(all_instances[bi, 9]),
                            float(all_instances[bi, 10]),
                            float(all_instances[bi, 11]),
                            float(f_scale)
                        )
                        
                        pos_host = pos_rel_all[bi]
                        r_eq = float(body_radii[bi])
                        pole = all_instances[bi, 9:12]
                        pos_star = pos_rel_all[star_idx]
                        to_star = pos_star - pos_host
                        dist_s = math.sqrt(to_star[0]**2 + to_star[1]**2 + to_star[2]**2)
                        if dist_s > 1e-6:
                            L = to_star / dist_s
                        else:
                            L = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                        p_dot_L = np.dot(pole, L)
                        p_proj_sq = 1.0 - p_dot_L**2
                        f_factor = 1.0 - f
                        host_r_minor = r_eq * np.sqrt(max(0.0, p_dot_L**2 + (f_factor**2) * p_proj_sq))
                        if u_ring_host_R_minor is not None:
                            u_ring_host_R_minor.value = float(host_r_minor)
                        if u_ring_host_atmo is not None:
                            atmo = next((a for a in atmo_bodies if a['body_idx'] == bi), None)
                            if atmo is not None:
                                props_c, trans_c, _ = get_cached_atmosphere_properties(atmo, mass_snap[bi])
                                scale_height_km = float(props_c.get('scale_height_km', 8.5))
                                u_ring_host_atmo.value = (float(trans_c[0]), float(trans_c[1]), float(trans_c[2]), scale_height_km)
                                if u_ring_host_refractivity is not None:
                                    u_ring_host_refractivity.value = float(props_c.get('refractivity', 0.00029))
                            else:
                                u_ring_host_atmo.value = (0.0, 0.0, 0.0, 0.0)
                                if u_ring_host_refractivity is not None:
                                    u_ring_host_refractivity.value = 0.0
                        k = self.body_ring_indices.get(bi)
                        if k is not None:
                            u_ring_caster_mask_lo_uni.value = int(cull_ring_caster_lo[k])
                            u_ring_caster_mask_hi_uni.value = int(cull_ring_caster_hi[k])
                        
                        body_rings = [r for r in ring_precomputed if r['body_idx'] == bi]
                        prog_rings['u_num_ring_planes'].value = len(body_rings)
                        b_name = bodies_data[bi]['name']
                        name_lower = b_name.lower()
                        is_textured = name_lower in self.ring_textures_front
                        if 'u_is_textured' in prog_rings:
                            prog_rings['u_is_textured'].value = is_textured
                        if is_textured:
                            if name_lower in self.ring_gl_textures_front:
                                self.ring_gl_textures_front[name_lower].use(location=4)
                            if name_lower in self.ring_gl_textures_back:
                                self.ring_gl_textures_back[name_lower].use(location=5)
                        for idx, r in enumerate(body_rings):
                            if idx >= 16: break
                            prog_rings[f'u_ring_planes[{idx}].color'].value = tuple(float(c) for c in r['raw_color'])
                            prog_rings[f'u_ring_planes[{idx}].inner_r'].value = float(r['inner_r'])
                            prog_rings[f'u_ring_planes[{idx}].outer_r'].value = float(r['outer_r'])
                            prog_rings[f'u_ring_planes[{idx}].opacity'].value = float(r['opacity'])
                            prog_rings[f'u_ring_planes[{idx}].scatter'].value = float(r['scatter'])
                            prog_rings[f'u_ring_planes[{idx}].asymmetry'].value = float(r['asymmetry'])
                            prog_rings[f'u_ring_planes[{idx}].backscatter'].value = float(r['backscatter'])
                            prog_rings[f'u_ring_planes[{idx}].row_idx'].value = int(r['row_idx'])
                            prog_rings[f'u_ring_planes[{idx}].is_textured'].value = 1.0 if r.get('is_textured', False) else 0.0
                            prog_rings[f'u_ring_planes[{idx}].unlit_factor'].value = float(r.get('unlit_factor', 1.0))
                            prog_rings[f'u_ring_planes[{idx}].saturation'].value = float(r.get('saturation', 1.0))
                            prog_rings[f'u_ring_planes[{idx}].hue_shift'].value = float(r.get('hue_shift', 0.0))
                            prog_rings[f'u_ring_planes[{idx}].brightness'].value = float(r.get('brightness', 1.0))
                            prog_rings[f'u_ring_planes[{idx}].alpha_boost'].value = float(r.get('alpha_boost', 1.0))
                        group['vao'].render(moderngl.TRIANGLES, vertices=group['num_indices'])
                
                if self.comparison_enabled and self.ring_render_groups_cmp:
                    for group in self.ring_render_groups_cmp:
                        bi = group['body_idx']
                        body_pos_rel = cmp_pos_rel[bi].astype('f4')
                        u_ring_body_offset.write(body_pos_rel)
                        u_ring_host_pos.write(body_pos_rel)
                        u_ring_host_radius.value = float(self.body_radii_cmp[bi])
                        if u_ring_host_color is not None:
                            u_ring_host_color.value = tuple(float(c) for c in self.body_colors_cmp[bi])
                        
                        body_idx_in_unified = num_bodies + bi
                        f = float(all_instances[body_idx_in_unified, 12])
                        f_scale = 1.0 / (1.0 - f) if f < 1.0 else 1.0
                        u_ring_host_pole_obl.value = (
                            float(all_instances[body_idx_in_unified, 9]),
                            float(all_instances[body_idx_in_unified, 10]),
                            float(all_instances[body_idx_in_unified, 11]),
                            float(f_scale)
                        )
                        
                        pos_host = cmp_pos_rel[bi]
                        r_eq = float(self.body_radii_cmp[bi])
                        pole = all_instances[body_idx_in_unified, 9:12]
                        pos_star = cmp_pos_rel[self.star_idx_cmp]
                        to_star = pos_star - pos_host
                        dist_s = math.sqrt(to_star[0]**2 + to_star[1]**2 + to_star[2]**2)
                        if dist_s > 1e-6:
                            L = to_star / dist_s
                        else:
                            L = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                        p_dot_L = np.dot(pole, L)
                        p_proj_sq = 1.0 - p_dot_L**2
                        f_factor = 1.0 - f
                        host_r_minor = r_eq * np.sqrt(max(0.0, p_dot_L**2 + (f_factor**2) * p_proj_sq))
                        if u_ring_host_R_minor is not None:
                            u_ring_host_R_minor.value = float(host_r_minor)
                        if u_ring_host_atmo is not None:
                            atmo = next((a for a in self.atmo_bodies_cmp if a['body_idx'] == bi), None)
                            if atmo is not None:
                                props_c, trans_c, _ = get_cached_atmosphere_properties(atmo, self.mass_snap_cmp[bi])
                                scale_height_km = float(props_c.get('scale_height_km', 8.5))
                                u_ring_host_atmo.value = (float(trans_c[0]), float(trans_c[1]), float(trans_c[2]), scale_height_km)
                                if u_ring_host_refractivity is not None:
                                    u_ring_host_refractivity.value = float(props_c.get('refractivity', 0.00029))
                            else:
                                u_ring_host_atmo.value = (0.0, 0.0, 0.0, 0.0)
                                if u_ring_host_refractivity is not None:
                                    u_ring_host_refractivity.value = 0.0
                        u_ring_caster_mask_lo_uni.value = 0
                        u_ring_caster_mask_hi_uni.value = 0
                        body_rings = [r for r in ring_precomputed if r['body_idx'] == bi]
                        prog_rings['u_num_ring_planes'].value = len(body_rings)
                        b_name = self.bodies_data_cmp[bi]['name']
                        name_lower = b_name.lower()
                        is_textured = name_lower in self.ring_textures_front
                        if 'u_is_textured' in prog_rings:
                            prog_rings['u_is_textured'].value = is_textured
                        if is_textured:
                            if name_lower in self.ring_gl_textures_front:
                                self.ring_gl_textures_front[name_lower].use(location=4)
                            if name_lower in self.ring_gl_textures_back:
                                self.ring_gl_textures_back[name_lower].use(location=5)
                        for idx, r in enumerate(body_rings):
                            if idx >= 16: break
                            prog_rings[f'u_ring_planes[{idx}].color'].value = tuple(float(c) for c in r['raw_color'])
                            prog_rings[f'u_ring_planes[{idx}].inner_r'].value = float(r['inner_r'])
                            prog_rings[f'u_ring_planes[{idx}].outer_r'].value = float(r['outer_r'])
                            prog_rings[f'u_ring_planes[{idx}].opacity'].value = float(r['opacity'])
                            prog_rings[f'u_ring_planes[{idx}].scatter'].value = float(r['scatter'])
                            prog_rings[f'u_ring_planes[{idx}].asymmetry'].value = float(r['asymmetry'])
                            prog_rings[f'u_ring_planes[{idx}].backscatter'].value = float(r['backscatter'])
                            prog_rings[f'u_ring_planes[{idx}].row_idx'].value = int(r['row_idx'])
                            prog_rings[f'u_ring_planes[{idx}].is_textured'].value = 1.0 if r.get('is_textured', False) else 0.0
                            prog_rings[f'u_ring_planes[{idx}].unlit_factor'].value = float(r.get('unlit_factor', 1.0))
                            prog_rings[f'u_ring_planes[{idx}].saturation'].value = float(r.get('saturation', 1.0))
                            prog_rings[f'u_ring_planes[{idx}].hue_shift'].value = float(r.get('hue_shift', 0.0))
                            prog_rings[f'u_ring_planes[{idx}].brightness'].value = float(r.get('brightness', 1.0))
                            prog_rings[f'u_ring_planes[{idx}].alpha_boost'].value = float(r.get('alpha_boost', 1.0))
                        group['vao'].render(moderngl.TRIANGLES, vertices=group['num_indices'])
            _perf_gpu_end(_gq)
            
            # Render Habitable Zone Visualizer
            _gq = _perf_gpu_begin(ctx, "gpu_hz")
            if self.camera.get("show_habitable_zone", False) and self.hz_vao is not None:
                ctx.disable(moderngl.CULL_FACE)
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                ctx.depth_mask = False
                
                self.prog_hz['view'].write(view.astype('f4').tobytes())
                self.prog_hz['projection'].write(projection.astype('f4').tobytes())
                self.prog_hz['u_depth_C'].value = depth_C
                self.prog_hz['u_far'].value = far
                self.prog_hz['u_color'].value = (0.15, 0.65, 0.25, 0.12)
                
                for i in range(num_bodies):
                    body = bodies_data[i]
                    if body.get('type') == 'Star':
                        sp = body.get('star_props', {})
                        lum = sp.get('lum', 1.0)
                        
                        hz_inner = math.sqrt(lum / 1.78)
                        hz_outer = math.sqrt(lum / 0.32)
                        
                        self.prog_hz['u_inner_r'].value = hz_inner
                        self.prog_hz['u_outer_r'].value = hz_outer
                        self.prog_hz['u_body_offset'].write(pos_rel_all[i].astype('f4'))
                        
                        self.hz_vao.render(moderngl.TRIANGLE_STRIP, vertices=self.hz_num_vertices)

                if self.comparison_enabled:
                    for i in range(num_bodies_cmp):
                        body = self.bodies_data_cmp[i]
                        if body.get('type') == 'Star':
                            sp = body.get('star_props', {})
                            lum = sp.get('lum', 1.0)
                            
                            hz_inner = math.sqrt(lum / 1.78)
                            hz_outer = math.sqrt(lum / 0.32)
                            
                            self.prog_hz['u_inner_r'].value = hz_inner
                            self.prog_hz['u_outer_r'].value = hz_outer
                            self.prog_hz['u_body_offset'].write(cmp_pos_rel[i].astype('f4'))
                            
                            self.hz_vao.render(moderngl.TRIANGLE_STRIP, vertices=self.hz_num_vertices)
                
                ctx.depth_mask = True
                ctx.disable(moderngl.BLEND)
            _perf_gpu_end(_gq)
    
            # --- Pass 2: Atmosphere in front of rings & ringless bodies ---
            _gq = _perf_gpu_begin(ctx, "gpu_atmo_front")
            if ringed_atmos:
                execute_atmosphere_pass(2, ringed_atmos)
            if ringless_atmos:
                execute_atmosphere_pass(0, ringless_atmos)
            _perf_gpu_end(_gq)

            # --- Pass 2.5: Dynamic Cloud Layer Pass (Rendered over atmosphere for physical volumetric ordering) ---
            if 'u_is_cloud_pass' in prog_spheres and getattr(self, 'body_textures_ssbo', None) is not None:
                _gq = _perf_gpu_begin(ctx, "gpu_clouds")
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                ctx.enable(moderngl.DEPTH_TEST)
                ctx.disable(moderngl.CULL_FACE)
                
                prog_spheres['u_is_cloud_pass'].value = True
                ctx.depth_mask = False
                vis_hi_buffer.bind_to_storage_buffer(binding=3)
                vao_hi.render_indirect(draw_cmds_buffer, moderngl.TRIANGLES, first=1)

                vis_ultra_buffer.bind_to_storage_buffer(binding=3)
                vao_ultra.render_indirect(draw_cmds_buffer, moderngl.TRIANGLES, first=2)
                prog_spheres['u_is_cloud_pass'].value = False
                ctx.depth_mask = True
                _perf_gpu_end(_gq)


            # ── Stellar-Forge UI Layer ('Orion UI') ──
            def _trigger_system_switch(target_name):
                nonlocal switch_req_name
                switch_req_name = target_name
                self.switch_req_name = target_name
                existing_snap = sys_mgr.get_snapshot(target_name)
                if existing_snap is not None:
                    req = {
                        "old_bodies_data": bodies_data,
                        "old_visual_data": visual_data,
                        "old_atmo_bodies": atmo_bodies,
                        "old_ring_bodies": ring_bodies,
                        "old_star_idx": star_idx,
                        "restore_snapshot": existing_snap,
                    }
                else:
                    raw_data = sys_mgr.load_system_data(target_name)
                    new_bndl = load_system_from_data(raw_data)
                    req = {
                        "old_bodies_data": bodies_data,
                        "old_visual_data": visual_data,
                        "old_atmo_bodies": atmo_bodies,
                        "old_ring_bodies": ring_bodies,
                        "old_star_idx": star_idx,
                        "new_bundle": new_bndl,
                    }
                with self.shared_state["lock"]:
                    self.shared_state["system_switch_request"] = req

            def _trigger_ephem_switch():
                capture_t = display_t
                capture_paused = self.time_ctrl["paused"]
                capture_speed = self.time_ctrl["multiplier"]
                import copy
                capture_bodies = copy.deepcopy(bodies_data)
                capture_visual = copy.deepcopy(visual_data)
                capture_atmo = list(atmo_bodies)
                capture_ring = list(ring_bodies)
                capture_star_idx = star_idx
                
                def _on_spice_ready():
                    sys_mgr_spice.load_kernels(force=True)
                    self._ephem_mapping_ver = None
                    epoch_dt = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
                    et_epoch = sys_mgr_spice.datetime_to_et(epoch_dt)
                    et = et_epoch + capture_t * 365.25 * 86400.0
                    ephem_bundle = sys_mgr_spice.build_ephemeris_system(et, template_bodies=capture_bodies)
                    new_bndl = load_system_from_data(ephem_bundle)
                    req = {
                        "old_bodies_data": capture_bodies,
                        "old_visual_data": capture_visual,
                        "old_atmo_bodies": capture_atmo,
                        "old_ring_bodies": capture_ring,
                        "old_star_idx": capture_star_idx,
                        "new_bundle": new_bndl,
                        "ephemeris_enter": True,
                        "preserve_state": True,
                        "preserve_t": capture_t,
                        "preserve_paused": capture_paused,
                        "preserve_speed": capture_speed
                    }
                    with self.shared_state["lock"]:
                        self.shared_state["system_switch_request"] = req
                
                self._show_ephem_download_modal = True
                sys_mgr_spice.download_kernels_async(on_complete=_on_spice_ready)

            def _trigger_ephem_exit(to_keplerian=False):
                target_name = SystemManager.SOLAR_SYSTEM_NAME
                nonlocal switch_req_name
                switch_req_name = target_name
                self.switch_req_name = target_name
                existing_snap = sys_mgr.get_snapshot(target_name)
                if existing_snap is not None:
                    req = {
                        "old_bodies_data": bodies_data,
                        "old_visual_data": visual_data,
                        "old_atmo_bodies": atmo_bodies,
                        "old_ring_bodies": ring_bodies,
                        "old_star_idx": star_idx,
                        "restore_snapshot": existing_snap,
                        "ephemeris_exit": True,
                        "to_keplerian": to_keplerian
                    }
                else:
                    raw_data = sys_mgr.load_system_data(target_name)
                    new_bndl = load_system_from_data(raw_data)
                    req = {
                        "old_bodies_data": bodies_data,
                        "old_visual_data": visual_data,
                        "old_atmo_bodies": atmo_bodies,
                        "old_ring_bodies": ring_bodies,
                        "old_star_idx": star_idx,
                        "new_bundle": new_bndl,
                        "ephemeris_exit": True,
                        "to_keplerian": to_keplerian
                    }
                with self.shared_state["lock"]:
                    self.shared_state["system_switch_request"] = req

            def _trigger_ephem_export(cap_t, y, m, d):
                nonlocal switch_req_name
                epoch_dt = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
                et_epoch = sys_mgr_spice.datetime_to_et(epoch_dt)
                et = et_epoch + cap_t * 365.25 * 86400.0
                ephem_bundle = sys_mgr_spice.build_ephemeris_system(et, template_bodies=bodies_data)
                export_name = f"Solar System ({y:04d}-{m:02d}-{d:02d})"
                sys_mgr.save_system_data(export_name, ephem_bundle)
                sys_mgr.save_meta(export_name, ephem_bundle)
                new_bndl = load_system_from_data(ephem_bundle)
                switch_req_name = export_name
                self.switch_req_name = export_name
                req = {
                    "old_bodies_data": bodies_data,
                    "old_visual_data": visual_data,
                    "old_atmo_bodies": atmo_bodies,
                    "old_ring_bodies": ring_bodies,
                    "old_star_idx": star_idx,
                    "new_bundle": new_bndl,
                    "switch_req_name": export_name,
                    "ephemeris_exit": True,
                    "preserve_state": True,
                    "preserve_t": cap_t,
                    "preserve_paused": self.time_ctrl["paused"],
                    "preserve_speed": self.time_ctrl["multiplier"]
                }
                with self.shared_state["lock"]:
                    self.shared_state["system_switch_request"] = req

            def _trigger_cmp_switch(target_name):
                self.comparison_system_name = target_name
                raw_data = sys_mgr.load_system_data(target_name)
                new_bndl = load_system_from_data(raw_data)
                req = {
                    "old_bodies_data": self.bodies_data_cmp,
                    "old_visual_data": self.visual_data_cmp,
                    "old_atmo_bodies": self.atmo_bodies_cmp,
                    "old_ring_bodies": self.ring_bodies_cmp,
                    "old_star_idx": self.star_idx_cmp,
                    "new_bundle": new_bndl,
                }
                with self.shared_state_cmp["lock"]:
                    self.shared_state_cmp["system_switch_request"] = req

            switch_triggers = {
                "switch_system": _trigger_system_switch,
                "ephem_switch": _trigger_ephem_switch,
                "ephem_exit": _trigger_ephem_exit,
                "ephem_export": _trigger_ephem_export,
                "cmp_switch": _trigger_cmp_switch,
                "load_system": load_system_from_data
            }

            self.active_system_name = active_system_name
            render_ui(
                app=self,
                ctx=ctx,
                bodies_data=bodies_data,
                visual_data=visual_data,
                atmo_bodies=atmo_bodies,
                ring_bodies=ring_bodies,
                star_idx=star_idx,
                num_bodies=num_bodies,
                mass_snap=mass_snap,
                parent_snap=parent_snap,
                tree_indices_snap=tree_indices_snap,
                tree_depths_snap=tree_depths_snap,
                subsys_mass_buf=subsys_mass_buf,
                subsys_pos_buf=subsys_pos_buf,
                subsys_vel_buf=subsys_vel_buf,
                pos_snap_render=pos_snap_render,
                vel_snap_render=vel_snap_render,
                visual_arr=visual_arr,
                cam_world_pos_f8=cam_world_pos_f8,
                cur_y=cur_y, cur_m=cur_m, cur_d=cur_d, cur_h=cur_h, cur_mn=cur_mn, cur_s=cur_s, cur_tz=cur_tz,
                display_t=display_t, dt_render=dt_render,
                tl_active=tl_active, tl_prog=tl_prog, tl_times=tl_times, is_scrubbing=is_scrubbing,
                spice_valid_mask=spice_valid_mask,
                prog_rings=prog_rings, ring_precomputed=ring_precomputed,
                ring_render_groups=ring_render_groups, ring_gradient_tex=ring_gradient_tex,
                switch_triggers=switch_triggers
            )
    
            is_pause_accum = self.time_ctrl.get("paused", False) and self.camera.get("photo_accum_enabled", False)
            is_ss_accum = getattr(self, "_screenshot_capturing", False) and self.camera.get("screenshot_accum_enabled", False)
            
            if is_pause_accum or is_ss_accum:
                cam_state_current = (
                    self.fb_width, self.fb_height, self.camera["cam_pos_rel"].tobytes(),
                    self.camera["yaw"], self.camera["pitch"], self.camera["roll"]
                )
                if not hasattr(self, "photo_accum_count"):
                    self.photo_accum_count = 0
                if getattr(self, "photo_accum_last_cam", None) != cam_state_current:
                    self.photo_accum_count = 0
                    self.photo_accum_last_cam = cam_state_current

                target_frames = self.camera.get("screenshot_accum_target", 16) if is_ss_accum else self.camera.get("photo_accum_target", 0)
                
                if target_frames == 0 or self.photo_accum_count < target_frames:
                    if self.photo_accum_count == 0:
                        ctx.copy_framebuffer(self.accum_fbo_a, self.hdr_resolve_fbo)
                        self.photo_accum_count = 1
                    else:
                        self.accum_fbo_b.use()
                        ctx.viewport = (0, 0, self.fb_width, self.fb_height)
                        ctx.disable(moderngl.DEPTH_TEST)
                        ctx.disable(moderngl.BLEND)
                        self.accum_tex_a.use(location=0)
                        self.hdr_resolve_tex.use(location=1)
                        if 'u_history' in self.prog_accum: self.prog_accum['u_history'].value = 0
                        if 'u_current' in self.prog_accum: self.prog_accum['u_current'].value = 1
                        if 'u_blend_weight' in self.prog_accum: self.prog_accum['u_blend_weight'].value = 1.0 / (self.photo_accum_count + 1)
                        self.quad_vao_accum.render(moderngl.TRIANGLE_STRIP)
                        
                        # Swap A and B
                        self.accum_fbo_a, self.accum_fbo_b = self.accum_fbo_b, self.accum_fbo_a
                        self.accum_tex_a, self.accum_tex_b = self.accum_tex_b, self.accum_tex_a
                        self.photo_accum_count += 1
                resolved_tex = self.accum_tex_a
            else:
                if hasattr(self, "photo_accum_count"):
                    self.photo_accum_count = 0
                resolved_tex = self.hdr_resolve_tex

            # --- Post Processing ---
            # Bloom Downsample
            _gq = _perf_gpu_begin(ctx, "gpu_bloom")
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            
            # Pass 1: Extract and Downsample to bloom_fbos[0]
            if len(self.bloom_fbos) == 5:
                self.bloom_fbos[0].use()
                ctx.viewport = (0, 0, self.bloom_texs[0].width, self.bloom_texs[0].height)
                resolved_tex.use(location=0)
                self.prog_bloom_down['u_texture'].value = 0
                self.prog_bloom_down['u_texel_size'].value = (1.0 / resolved_tex.width, 1.0 / resolved_tex.height)
                self.prog_bloom_down['u_threshold'].value = self.camera.get("bloom_threshold", 1.0)
                self.quad_vao_down.render(moderngl.TRIANGLE_STRIP)
                
                # Pass 2..5: Downsample
                self.prog_bloom_down['u_threshold'].value = 0.0 # only threshold on first pass
                for i in range(1, 5):
                    self.bloom_fbos[i].use()
                    ctx.viewport = (0, 0, self.bloom_texs[i].width, self.bloom_texs[i].height)
                    self.bloom_texs[i-1].use(location=0)
                    self.prog_bloom_down['u_texel_size'].value = (1.0 / self.bloom_texs[i-1].width, 1.0 / self.bloom_texs[i-1].height)
                    self.quad_vao_down.render(moderngl.TRIANGLE_STRIP)
                    
                # Pass 6..9: Upsample with Additive Blending
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = moderngl.ONE, moderngl.ONE
                self.prog_bloom_up['u_texture'].value = 0
                self.prog_bloom_up['u_radius'].value = 1.0
                
                for i in range(3, -1, -1):
                    self.bloom_fbos[i].use()
                    ctx.viewport = (0, 0, self.bloom_texs[i].width, self.bloom_texs[i].height)
                    self.bloom_texs[i+1].use(location=0)
                    self.prog_bloom_up['u_texel_size'].value = (1.0 / self.bloom_texs[i+1].width, 1.0 / self.bloom_texs[i+1].height)
                    self.quad_vao_up.render(moderngl.TRIANGLE_STRIP)
                    
                _perf_gpu_end(_gq)
                # Final Composite
                ctx.disable(moderngl.BLEND)
                
                # Screenshot: capture composited result to a temporary high-res FBO
                save_now = False
                if getattr(self, "_accum_save_request", False):
                    save_now = True
                elif getattr(self, "_screenshot_capturing", False):
                    if self.camera.get("screenshot_accum_enabled", False):
                        if getattr(self, "photo_accum_count", 0) >= self.camera.get("screenshot_accum_target", 16):
                            save_now = True
                    else:
                        save_now = True

                if save_now:
                    ss_w, ss_h = self.fb_width, self.fb_height
                    ss_tex = ctx.texture((ss_w, ss_h), 4)  # 8-bit RGBA for final sRGB output
                    ss_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    ss_fbo = ctx.framebuffer(color_attachments=[ss_tex])
                    ss_fbo.use()
                    ctx.viewport = (0, 0, ss_w, ss_h)
                    resolved_tex.use(location=0)
                    self.bloom_texs[0].use(location=1)
                    self.prog_composite['u_main_texture'].value = 0
                    self.prog_composite['u_bloom_texture'].value = 1
                    self.prog_composite['u_bloom_intensity'].value = self.camera.get("bloom_intensity", 0.05)
                    self.quad_vao_comp.render(moderngl.TRIANGLE_STRIP)
                    
                    # Read composited pixels
                    raw_pixels = ss_fbo.read(components=3)
                    ss_fbo.release()
                    ss_tex.release()
                    
                    # Restore original resolution
                    if self._screenshot_orig_fb:
                        self.fb_width, self.fb_height = self._screenshot_orig_fb
                        self.last_fb_size = (0, 0)       # Force FBO rebuild next frame
                        self.last_msaa_samples = -1
                    self._screenshot_capturing = False
                    self._screenshot_orig_fb = None
                    self._accum_save_request = False
                    
                    # Save PNG in a background thread
                    _ss_cap_w, _ss_cap_h = ss_w, ss_h
                    _ss_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    _ss_filename = f"screenshots/StellarForge_{_ss_cap_w}x{_ss_cap_h}_{_ss_timestamp}.png"
                    self._screenshot_saving = True
                    self._screenshot_toast = (f"Saving {_ss_cap_w}\u00d7{_ss_cap_h}...", time.time())
                    _ss_self_ref = self
                    def _save_screenshot_bg(_raw, _w, _h, _fname, _self):
                        try:
                            from PIL import Image
                            import os
                            os.makedirs("screenshots", exist_ok=True)
                            img = Image.frombytes('RGB', (_w, _h), _raw)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            img.save(_fname, 'PNG', optimize=False)
                            _self._screenshot_toast = (f"Saved: {_fname}", time.time())
                        except Exception as e:
                            _self._screenshot_toast = (f"Screenshot failed: {e}", time.time())
                        finally:
                            _self._screenshot_saving = False
                    threading.Thread(
                        target=_save_screenshot_bg,
                        args=(raw_pixels, _ss_cap_w, _ss_cap_h, _ss_filename, _ss_self_ref),
                        daemon=True
                    ).start()
                
                # Normal composite to screen for display
                _gq = _perf_gpu_begin(ctx, "gpu_composite")
                ctx.screen.use()
                ctx.viewport = (0, 0, self.fb_width, self.fb_height)
                resolved_tex.use(location=0)
                self.bloom_texs[0].use(location=1)
                self.prog_composite['u_main_texture'].value = 0
                self.prog_composite['u_bloom_texture'].value = 1
                self.prog_composite['u_bloom_intensity'].value = self.camera.get("bloom_intensity", 0.05)
                self.quad_vao_comp.render(moderngl.TRIANGLE_STRIP)
                _perf_gpu_end(_gq)

            # Re-enable standard settings for ImGui
            ctx.screen.use()
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            # Screenshot toast notification
            if self._screenshot_toast:
                _toast_msg, _toast_time = self._screenshot_toast
                _toast_elapsed = time.time() - _toast_time
                if self._screenshot_saving or _toast_elapsed < 4.0:
                    _toast_alpha = 1.0 if self._screenshot_saving or _toast_elapsed < 3.0 else max(0.0, 1.0 - (_toast_elapsed - 3.0))
                    imgui.set_next_window_position(self.fb_width - 340, self.fb_height - 50, imgui.ALWAYS)
                    imgui.set_next_window_size(330, 40)
                    imgui.set_next_window_bg_alpha(0.75 * _toast_alpha)
                    imgui.push_style_var(imgui.STYLE_ALPHA, _toast_alpha)
                    imgui.begin("##screenshot_toast", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_INPUTS)
                    if self._screenshot_saving:
                        imgui.text_colored(_toast_msg, 1.0, 1.0, 0.4)
                    else:
                        imgui.text_colored(_toast_msg, 0.4, 1.0, 0.4)
                    imgui.end()
                    imgui.pop_style_var()
                elif _toast_elapsed >= 4.0:
                    self._screenshot_toast = None
            
            imgui.render()
     
            _gq = _perf_gpu_begin(ctx, "gpu_imgui")
            self.impl.render(imgui.get_draw_data())
            _perf_gpu_end(_gq)
            glfw.swap_buffers(window)
            
            self._last_fb_width = self.fb_width
            self._last_fb_height = self.fb_height
            
        running[0] = False
        running_cmp[0] = False
        physics_thread.join(timeout=1.0)
        physics_thread_cmp.join(timeout=1.0)
        self.save_settings()
        self.impl.shutdown()
        glfw.terminate()
    
