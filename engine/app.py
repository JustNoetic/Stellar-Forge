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
            # Scroll wheel changes flight velocity (Space Engine style, ~1.15x per notch).
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
        
        if self.camera["left_dragging"] and self.camera["right_dragging"]:
            # LMB+RMB: radial approach / recede toward the tracked body's surface.
            # Drag down (dy > 0) = move closer; drag up = pull back. Applied per-frame
            # as an exponential scale so the gesture feels uniform at any distance.
            self.camera["approach_delta"] += dy * 0.002
        elif self.camera["left_dragging"]:
            # LMB: trackball pivot in place (free look), direct (no smoothing).
            # Rotates about the camera's own screen axes so the view always
            # follows the cursor regardless of ecliptic orientation.
            sensitivity = 0.3 * fov_ratio
            _camera_pivot_apply(self.camera, dx, dy, sensitivity)
            self.camera["cam_look"] = "free"
        elif self.camera["right_dragging"]:
            # RMB: trackball orbit around the pivot, direct (no smoothing).
            # Screen-space: dragging right always orbits the camera right.
            sensitivity = 0.3 * fov_ratio
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
        sys_mgr = SystemManager()
        sys_mgr_spice = SpiceManager()
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
    
        UBO_SIZE = 5280
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

        self.atmo_ssbo = ctx.buffer(reserve=608)
        self.atmo_ssbo.bind_to_storage_buffer(binding=8)
        self.atmo_staging = np.zeros(152, dtype=np.float32)
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

            # 1) LMB+RMB approach gesture: radial move toward the tracked body's surface
            if cam["approach_delta"] != 0.0:
                r_ap = np.linalg.norm(rel)
                if r_ap > 1e-300 and cam["tracking_idx"] is not None and cam["tracking_mode"] == "body" and track_r > 0.0:
                    rel *= math.exp(-cam["approach_delta"])
                    min_r = _ellipsoid_surface_radius(track_r, track_f, track_pole, rel) + LAND_ALT_AU
                    r_ap = np.linalg.norm(rel)
                    if r_ap < min_r:
                        rel = rel / max(r_ap, 1e-300) * min_r
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
                    caster_max_bend_buf[i_c] = 0.0
            if n_casters_fixed < 64:
                caster_atmos_buf[n_casters_fixed:] = 0
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
                                
                    n_active = min(len(active_indices), 8)
                    active_indices = active_indices[:n_active]
                    
                    # Clear scratch buffers in-place
                    active_casters_buf[:] = 0.0
                    active_poles_obl_buf[:] = 0.0
                    active_caster_r_minor_buf[:] = 0.0
                    active_atmos_buf[:] = 0.0
                    active_max_bend_buf[:] = 0.0
                    
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
                    self.atmo_staging[148:152] = [float(props.get('ozone_peak_km', 25.0)),
                                                  float(props.get('ozone_width_km', 8.0)),
                                                  0.0, float(self.camera.get('atmo_adaptive_steps_max', 128))]  # pad to 608 bytes

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
                    self.atmo_staging[150] = float(atmo['planet_radius_km']) * (1.0 - _icosphere_max_sag(_lod_subdiv))

                    # Single buffer upload per atmosphere body
                    self.atmo_ssbo.write(self.atmo_staging.tobytes())
                    
                    cur_vao.render(moderngl.TRIANGLES)
    
                ctx.depth_mask = True
                ctx.enable(moderngl.DEPTH_TEST)
                ctx.enable(moderngl.CULL_FACE)
                ctx.cull_face = 'back'
                ctx.depth_func = '<'
                ctx.disable(moderngl.BLEND)

            def execute_atmosphere_pass(clip_mode):
                if not self.camera.get("atmo_enabled", True):
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
                    
                    if 'u_screen_res' in prog_atmo_lowres:
                        prog_atmo_lowres['u_screen_res'].value = (float(self.atmo_lowres_fbo.width), float(self.atmo_lowres_fbo.height))
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

                for sq_dist, atmo, is_cmp in sorted_atmos:
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

            # --- Pass 1: Atmosphere behind rings ---
            _gq = _perf_gpu_begin(ctx, "gpu_atmo_behind")
            execute_atmosphere_pass(1)
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
    
            # --- Pass 2: Atmosphere in front of rings ---
            _gq = _perf_gpu_begin(ctx, "gpu_atmo_front")
            execute_atmosphere_pass(2)
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


            def compute_body_albedos(body_info, insp_idx, insp_is_cmp):
                """Compute Geometric Albedo (A_g), Bond Albedo (A_b), Phase Integral (q), and Top-of-Atmosphere RGB reflectance."""
                if insp_is_cmp and hasattr(self, 'visual_arr_cmp') and self.visual_arr_cmp is not None and len(self.visual_arr_cmp) > insp_idx:
                    v_color = self.visual_arr_cmp[insp_idx, 0:3]
                elif hasattr(self, 'visual_arr') and visual_arr is not None and len(visual_arr) > insp_idx:
                    v_color = visual_arr[insp_idx, 0:3]
                else:
                    v_color = None

                p_surf, _, _ = compute_surface_albedo(body_info, self.texture_mean_colors, v_color)

                atmo_bodies_list = self.atmo_bodies_cmp if insp_is_cmp else atmo_bodies
                atmo_item = next((a for a in atmo_bodies_list if a.get('body_idx') == insp_idx), None)

                if atmo_item:
                    mass_snap_buf = self.mass_snap_cmp if insp_is_cmp else cur_mass_snap
                    mass_val = mass_snap_buf[insp_idx] if (mass_snap_buf is not None and len(mass_snap_buf) > insp_idx) else 3.0e-6
                    props, _, _ = get_cached_atmosphere_properties(atmo_item, mass_val)

                    H_r_m = max(0.0, float(props['scale_height_km'])) * 1000.0
                    H_m_m = max(0.0, float(atmo_item.get('h_mie', 1.2))) * 1000.0

                    beta_R = np.nan_to_num(props['beta_rayleigh'], nan=0.0, posinf=0.0, neginf=0.0)
                    beta_M = np.nan_to_num(compute_mie_coefficients(atmo_item.get('beta_mie', 2.0e-6), atmo_item.get('mie_angstrom', None)), nan=0.0, posinf=0.0, neginf=0.0)
                    mie_alb = np.nan_to_num(np.asarray(atmo_item.get('mie_albedo', np.array([1.0, 1.0, 1.0], dtype=np.float32)), dtype=np.float64), nan=1.0)
                    beta_O3 = np.nan_to_num(props['beta_abs_layered'], nan=0.0, posinf=0.0, neginf=0.0)
                    beta_mixed = np.nan_to_num(props['beta_abs_mixed'], nan=0.0, posinf=0.0, neginf=0.0)

                    tau_R = beta_R * H_r_m
                    tau_M_ext = beta_M * H_m_m
                    tau_M_sca = tau_M_ext * mie_alb
                    tau_O3 = beta_O3 * 14179.6
                    tau_mixed = beta_mixed * H_r_m

                    tau_total = tau_R + tau_M_ext + tau_O3 + tau_mixed
                    T_surf = np.exp(-tau_total)
                    p_surf_contrib = p_surf * (T_surf ** 2)

                    p_Rayleigh = 0.69 * (1.0 - np.exp(-tau_R)) * np.exp(-2.0 * tau_O3)
                    mie_g = float(atmo_item.get('mie_g', 0.76))
                    p_mie_peak = (1.0 + mie_g) / max(1e-4, 1.0 - mie_g)
                    p_Mie = 0.75 * np.minimum(p_mie_peak, 2.5) * (1.0 - np.exp(-tau_M_sca)) * np.exp(-2.0 * tau_O3)

                    p_rgb = p_surf_contrib + (1.0 - p_surf_contrib) * (p_Rayleigh + p_Mie)
                    p_rgb = np.clip(p_rgb, 0.0, 1.0)
                    A_g = float(0.2126 * p_rgb[0] + 0.7152 * p_rgb[1] + 0.0722 * p_rgb[2])

                    tau_scat = tau_R + tau_M_sca
                    tau_scat_scalar = float(0.2126 * tau_scat[0] + 0.7152 * tau_scat[1] + 0.0722 * tau_scat[2])
                    tau_R_scalar = float(0.2126 * tau_R[0] + 0.7152 * tau_R[1] + 0.0722 * tau_R[2])
                    tau_M_scalar = float(0.2126 * tau_M_sca[0] + 0.7152 * tau_M_sca[1] + 0.0722 * tau_M_sca[2])

                    w_R = tau_R_scalar / max(1e-6, tau_scat_scalar)
                    w_M = tau_M_scalar / max(1e-6, tau_scat_scalar)
                    q_R = 1.35
                    q_M = 1.50 * (1.0 - 0.6 * (mie_g ** 1.2))
                    q_atmo_scat = w_R * q_R + w_M * q_M

                    f_atmo = 1.0 - math.exp(-tau_scat_scalar)
                    b_type = body_info.get('type', 'Terrestrial')
                    # Base q for the surface. Airless rocky bodies have high backscatter (q~0.38).
                    # Bodies with atmospheres have smoothed scattering (q approaches 1.0+).
                    if b_type == 'Gas Giant':
                        base_q_surf = 1.50
                    else:
                        # Interpolate from Moon-like (0.38) to Earth/Venus-like (1.25) based on atmospheric optical depth
                        atmo_weight = min(1.0, tau_scat_scalar * 2.0)
                        base_q_surf = 0.38 * (1.0 - atmo_weight) + 1.25 * atmo_weight
                    q = (1.0 - f_atmo) * base_q_surf + f_atmo * q_atmo_scat
                    A_b = float(np.clip(q * A_g, 0.0, 1.0))
                else:
                    p_rgb = np.clip(p_surf, 0.0, 1.0)
                    A_g = float(0.2126 * p_rgb[0] + 0.7152 * p_rgb[1] + 0.0722 * p_rgb[2])
                    b_type = body_info.get('type', 'Terrestrial')
                    q = 1.50 if b_type == 'Gas Giant' else 0.38
                    A_b = float(np.clip(q * A_g, 0.0, 1.0))

                return A_g, A_b, q, p_rgb



            

            force_layout = getattr(self, "_last_fb_width", 0) != self.fb_width or getattr(self, "_last_fb_height", 0) != self.fb_height
            cond_layout = imgui.ALWAYS if force_layout else imgui.ONCE
            
            imgui.set_next_window_position(10, 10, cond_layout)
            imgui.set_next_window_size(320, self.fb_height - 20, cond_layout)
            imgui.begin("Control Panel")
            
            imgui.text_colored("Simulation & Time Controls", 0.6, 0.9, 1.0)
            imgui.separator()
            
            imgui.text("Current Date:")
            imgui.text(f"{cur_y:04d}-{cur_m:02d}-{cur_d:02d} {cur_h:02d}:{cur_mn:02d}:{cur_s:02d} {cur_tz}")
            
            imgui.separator()
            imgui.text_colored("Camera", 0.6, 0.9, 1.0)
            if self.camera.get("cam_look", "aim") == "aim" and self.camera["tracking_idx"] is not None:
                track_name = "?"
                t_idx = self.camera["tracking_idx"]
                if self.camera.get("tracking_is_cmp", False):
                    if t_idx < len(self.bodies_data_cmp):
                        track_name = self.bodies_data_cmp[t_idx].get('name', '?')
                elif t_idx < len(bodies_data):
                    track_name = bodies_data[t_idx].get('name', '?')
                mode_label = f"Orbit: {track_name}"
            else:
                mode_label = "Flight"
            imgui.text(f"Mode: {mode_label}")
            imgui.text(f"Velocity: {format_flight_speed(self.camera.get('flight_speed', 0.0))}")
            speed_log_cam = math.log10(max(self.camera.get("flight_speed", 0.1), 1e-13))
            _, self.camera["flight_speed"] = imgui.slider_float("##flyspeed", speed_log_cam, -13.0, 0.7, "")
            self.camera["flight_speed"] = 10 ** self.camera["flight_speed"]
            _, self.camera["horizon_align"] = imgui.checkbox("Auto-level horizon", self.camera.get("horizon_align", True))
            
            imgui.separator()
            imgui.text("Time Controls")
            _, self.time_ctrl["paused"] = imgui.checkbox("Paused", self.time_ctrl["paused"])
            
            # Direction toggle button
            td = self.time_ctrl["time_direction"]
            effective_mult = self.time_ctrl["multiplier"] * td
            imgui.text(f"Speed: {format_time_speed(effective_mult)}")
            
            if td >= 0:
                if imgui.button(u"\u25b6 Forward"):
                    self.time_ctrl["time_direction"] = -1
            else:
                if imgui.button(u"\u25c0 Backward"):
                    self.time_ctrl["time_direction"] = 1
            imgui.same_line()
            
            val_log = math.log10(max(1.0, abs(self.time_ctrl["multiplier"])))
            changed, new_log = imgui.slider_float("##speed", val_log, 0.0, 12.0, "")
            if changed:
                self.time_ctrl["multiplier"] = 10 ** new_log
                    
            if imgui.button("Reset Speed"): self.time_ctrl["multiplier"] = 1.0
                    
            if self.shared_state.get("syncing", False):
                imgui.text("Synchronizing Physics...")
                imgui.progress_bar(self.shared_state.get("sync_progress", 0.0), size=(-1, 0.0))
                
            elif tl_active and tl_prog < 1.0:
                imgui.text("Rendering Timeline...")
                
                rate = self.shared_state.get("timeline_rate", 0.0)
                if rate > 0.0:
                    imgui.text(f"Render Pace: {format_time_speed(rate)}")
                    
                imgui.progress_bar(tl_prog)
                if imgui.button("Cancel"):
                    self.time_ctrl["cancel_render"] = True
            elif is_scrubbing:
                imgui.separator()
                imgui.text("Timeline Navigation")
                max_idx = max(0, len(tl_times) - 1)
                
                if self.time_ctrl.get("snap_to_end", False):
                    scrub_index[0] = max_idx
                    self.time_ctrl["snap_to_end"] = False
                
                if self.time_ctrl["timeline_playing"]:
                    if imgui.button("Pause Playback"):
                        self.time_ctrl["timeline_playing"] = False
                        
                    self.time_ctrl["timeline_scrub_float"] += self.time_ctrl["timeline_speed"] * dt_render
                    steps_to_add = int(self.time_ctrl["timeline_scrub_float"])
                    if steps_to_add > 0:
                        scrub_index[0] += steps_to_add
                        self.time_ctrl["timeline_scrub_float"] -= steps_to_add
                        if scrub_index[0] > max_idx:
                            scrub_index[0] = 0
                else:
                    if imgui.button("Play Timeline"):
                        self.time_ctrl["timeline_playing"] = True
                        self.time_ctrl["timeline_scrub_float"] = 0.0
                        if scrub_index[0] >= max_idx:
                            scrub_index[0] = 0
                            
                imgui.same_line()
                _, self.time_ctrl["timeline_speed"] = imgui.slider_float("Speed##tl", self.time_ctrl["timeline_speed"], 1.0, 100.0, "%.1fx")
                
                changed, scrub_index[0] = imgui.slider_int("##Scrub", scrub_index[0], 0, max_idx, "")
                
                if imgui.button("Resume Here"):
                    self.time_ctrl["sync_t"] = tl_times[scrub_index[0]]
                    self.time_ctrl["sync_idx"] = scrub_index[0]
                    self.time_ctrl["paused"] = False
                    self.time_ctrl["timeline_playing"] = False
                    with self.shared_state["lock"]:
                        self.shared_state["timeline_active"] = False
                imgui.same_line()
                if imgui.button("Cancel"):
                    with self.shared_state["lock"]:
                        self.shared_state["timeline_active"] = False
                    self.time_ctrl["timeline_playing"] = False
                        
            else:
                imgui.separator()
                imgui.text("Jump in Time")
                _, jump_date[0] = imgui.input_int("Year", jump_date[0])
                _, jump_date[1] = imgui.input_int("Month", jump_date[1])
                _, jump_date[2] = imgui.input_int("Day", jump_date[2])
                
                imgui.text("Time (HH:MM:SS)")
                imgui.push_item_width(40)
                _, jump_date[3] = imgui.input_int("##Hour", jump_date[3], step=0)
                imgui.same_line()
                imgui.text(":")
                imgui.same_line()
                _, jump_date[4] = imgui.input_int("##Minute", jump_date[4], step=0)
                imgui.same_line()
                imgui.text(":")
                imgui.same_line()
                _, jump_date[5] = imgui.input_int("##Second", jump_date[5], step=0)
                imgui.pop_item_width()
                
                jump_date[1] = max(1, min(12, jump_date[1]))
                jump_date[2] = max(1, min(31, jump_date[2]))
                jump_date[3] = max(0, min(23, jump_date[3]))
                jump_date[4] = max(0, min(59, jump_date[4]))
                jump_date[5] = max(0, min(59, jump_date[5]))
                
                if ephemeris_mode_active or keplerian_mode_active:
                    if imgui.button("Jump to Date"):
                        target_t = sim_time_from_date(jump_date[0], jump_date[1], jump_date[2], jump_date[3], jump_date[4], jump_date[5])
                        self.time_ctrl["sync_t"] = target_t
                else:
                    if imgui.button("Render Timeline"):
                        target_t = sim_time_from_date(jump_date[0], jump_date[1], jump_date[2], jump_date[3], jump_date[4], jump_date[5])
                        self.time_ctrl["target_t"] = target_t
                        self.time_ctrl["cancel_render"] = False
                        self.time_ctrl["render_timeline"] = True
            
            imgui.separator()
            if imgui.button("Graphics & Quality Settings..."):
                self.camera["show_settings_modal"] = True
                
            fov_val = self.camera['fov']
            if fov_val >= 1.0:
                fov_fmt = "%.1f deg"
            elif fov_val >= 0.01:
                fov_fmt = "%.3f deg"
            else:
                fov_fmt = "%.5f deg"
            changed_fov, new_fov = imgui.slider_float("FOV", fov_val, 0.001, 120.0, fov_fmt, imgui.SLIDER_FLAGS_LOGARITHMIC)
            if changed_fov:
                self.camera["fov"] = max(0.001, min(120.0, new_fov))
                self.save_settings()
            if imgui.button("Reset FOV"):
                self.camera["fov"] = 45.0
                self.save_settings()
            
            imgui.separator()
            imgui.text_colored("Screenshot", 0.6, 0.9, 1.0)
            _ss_res_labels = ["4K (3840x2160)", "8K (7680x4320)", "16K (15360x8640)"]
            _ss_res_idx = self.camera.get("screenshot_res_idx", 1)
            _ss_changed, _ss_new_idx = imgui.combo("Resolution##ss", _ss_res_idx, _ss_res_labels)
            if _ss_changed:
                self.camera["screenshot_res_idx"] = _ss_new_idx
                self.save_settings()
            _ss_can_capture = not self._screenshot_capturing and not self._screenshot_saving
            if not _ss_can_capture:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            if imgui.button("Capture Screenshot (F12)"):
                if _ss_can_capture:
                    _ss_presets = [(3840, 2160), (7680, 4320), (15360, 8640)]
                    _ss_idx_clamped = max(0, min(len(_ss_presets) - 1, self.camera.get("screenshot_res_idx", 1)))
                    self._screenshot_request = _ss_presets[_ss_idx_clamped]
            if not _ss_can_capture:
                imgui.pop_style_var()
            if self._screenshot_saving:
                imgui.same_line()
                imgui.text_colored("Saving...", 1.0, 1.0, 0.4)
            
            imgui.separator()
            imgui.text_colored("Accumulation Settings", 0.6, 0.9, 1.0)
            
            p_enabled = self.camera.setdefault("photo_accum_enabled", False)
            s_enabled = self.camera.setdefault("screenshot_accum_enabled", False)
            
            changed_p, new_p = imgui.checkbox("Pause Accumulation", p_enabled)
            if changed_p:
                self.camera["photo_accum_enabled"] = new_p
                self.save_settings()
                
            imgui.same_line()
            changed_s, new_s = imgui.checkbox("Screenshot Accumulation", s_enabled)
            if changed_s:
                self.camera["screenshot_accum_enabled"] = new_s
                self.save_settings()
                
            if self.camera["photo_accum_enabled"]:
                accum_target = self.camera.setdefault("photo_accum_target", 0)
                changed, new_accum = imgui.slider_int("Pause Target Frames", accum_target, 0, 1000, format="%d (0=∞)")
                if changed:
                    self.camera["photo_accum_target"] = new_accum
                    self.save_settings()
                    
            if self.camera["screenshot_accum_enabled"]:
                ss_target = self.camera.setdefault("screenshot_accum_target", 16)
                changed, new_ss = imgui.slider_int("Screenshot Target Frames", ss_target, 1, 128)
                if changed:
                    self.camera["screenshot_accum_target"] = new_ss
                    self.save_settings()
            
            is_pause_accum = self.time_ctrl.get("paused", False) and self.camera.get("photo_accum_enabled", False)
            is_ss_accum = getattr(self, "_screenshot_capturing", False) and self.camera.get("screenshot_accum_enabled", False)
            if is_pause_accum or is_ss_accum:
                count = getattr(self, "photo_accum_count", 0)
                imgui.text(f"Accumulated: {count} frames")
                if is_pause_accum and count > 1:
                    imgui.text_colored("Press ESC to save accumulated frame", 0.4, 1.0, 0.4)
            else:
                imgui.text_colored("Pause to start accumulation", 0.5, 0.5, 0.5)
            
            if self.camera.get("show_settings_modal", False):
                imgui.set_next_window_size(320, 320, imgui.FIRST_USE_EVER)
                imgui.set_next_window_position(self.fb_width // 2 - 160, self.fb_height // 2 - 160, imgui.FIRST_USE_EVER)
                expanded, self.camera["show_settings_modal"] = imgui.begin("Graphics & Quality Settings", True)
                if expanded:
                    settings_changed = False
                    # Atmosphere Quality
                    changed_aq, atmo_quality = imgui.combo("Atmosphere Quality", atmo_quality, ["Off", "Low (2D Shadows)", "High (Volumetric)"])
                    if changed_aq:
                        self.camera["atmo_quality"] = atmo_quality
                        settings_changed = True
                    
                    if atmo_quality > 2:
                        atmo_quality = 2
                        self.camera["atmo_quality"] = 2
                        settings_changed = True
                    if atmo_quality > 0:
                        atmo_res = float(self.camera.get("atmo_resolution", 1.0))
                        changed_res, atmo_res = imgui.slider_float("Atmosphere Render Scale", atmo_res, 0.2, 1.0, "%.2fx")
                        if changed_res:
                            self.camera["atmo_resolution"] = atmo_res
                            settings_changed = True
                            
                        if atmo_res < 0.999:
                            vrs_threshold = float(self.camera.get("atmo_vrs_threshold_px", 100.0))
                            changed_vrs, vrs_threshold = imgui.slider_float("VRS Low-Res Threshold (px)", vrs_threshold, 10.0, 1000.0, "%.1f")
                            if changed_vrs:
                                self.camera["atmo_vrs_threshold_px"] = vrs_threshold
                                settings_changed = True
                            
                        max_steps = self.camera.get("atmo_steps_max", 32)
                        changed_steps, max_steps = imgui.slider_int("Max Ray Steps", max_steps, 4, 128)
                        if changed_steps:
                            self.camera["atmo_steps_max"] = max_steps
                            settings_changed = True
                            
                        adaptive_steps = self.camera.get("atmo_adaptive_steps", True)
                        changed_adapt, adaptive_steps = imgui.checkbox("Adaptive Step Count", adaptive_steps)
                        if changed_adapt:
                            self.camera["atmo_adaptive_steps"] = adaptive_steps
                            settings_changed = True
                            
                        if adaptive_steps:
                            max_adapt_steps = self.camera.get("atmo_adaptive_steps_max", 128)
                            changed_max_adapt, max_adapt_steps = imgui.slider_int("Max Adaptive Steps", max_adapt_steps, 8, 256)
                            if changed_max_adapt:
                                self.camera["atmo_adaptive_steps_max"] = max_adapt_steps
                                settings_changed = True
                            
                        stochastic_steps = self.camera.get("atmo_stochastic", True)
                        changed_stoch, stochastic_steps = imgui.checkbox("Stochastic Raymarching", stochastic_steps)
                        if changed_stoch:
                            self.camera["atmo_stochastic"] = stochastic_steps
                            settings_changed = True
                    
                    # Shadow Caster Budget
                    caster_budget = self.camera.get("shadow_caster_budget", 32)
                    changed_budget, caster_budget = imgui.slider_int("Shadow Caster Budget", caster_budget, 4, 64)
                    if changed_budget:
                        self.camera["shadow_caster_budget"] = caster_budget
                        settings_changed = True
                    
                    imgui.separator()
                    
                    # Exposure & HDR
                    changed_hdr, self.camera["hdr_enabled"] = imgui.checkbox("HDR Mode", self.camera.get("hdr_enabled", True))
                    if changed_hdr:
                        settings_changed = True
                    if self.camera.get("hdr_enabled", True):
                        changed_exp, self.camera["exposure"] = imgui.slider_float("Exposure", self.camera.get("exposure", 1.0), 0.0001, 10000.0, "%.4f", imgui.SLIDER_FLAGS_LOGARITHMIC)
                        if changed_exp:
                            settings_changed = True
                    
                    # Bloom
                    changed_bi, self.camera["bloom_intensity"] = imgui.slider_float("Bloom Intensity", self.camera.get("bloom_intensity", 0.05), 0.0, 1.0, "%.3f")
                    if changed_bi:
                        settings_changed = True
                    changed_bt, self.camera["bloom_threshold"] = imgui.slider_float("Bloom Threshold", self.camera.get("bloom_threshold", 1.0), 0.0, 10.0, "%.2f")
                    if changed_bt:
                        settings_changed = True
                    
                    # MSAA (orbit lines only)
                    msaa_options = [0, 2, 4, 8]
                    msaa_labels = ["Off", "2x", "4x", "8x"]
                    current_msaa = self.camera.get("msaa_samples", 4)
                    current_idx = msaa_options.index(current_msaa) if current_msaa in msaa_options else 2
                    changed_msaa, new_msaa_idx = imgui.combo("Orbit MSAA", current_idx, msaa_labels)
                    if changed_msaa:
                        self.camera["msaa_samples"] = msaa_options[new_msaa_idx]
                        settings_changed = True

                    # Orbit Lines
                    changed_so, show_orbits = imgui.checkbox("Show Orbits", show_orbits)
                    if changed_so:
                        self.camera["show_orbits"] = show_orbits
                        settings_changed = True
                    if show_orbits:
                        changed_ofd, orbit_fade_dir_idx = imgui.combo("Orbit Fade", orbit_fade_dir_idx, ["Bright Behind", "Bright Ahead"])
                        if changed_ofd:
                            orbit_fade_dir = 1.0 if orbit_fade_dir_idx == 0 else -1.0
                            self.camera["orbit_fade_dir_idx"] = orbit_fade_dir_idx
                            settings_changed = True
                        changed_oma, orbit_min_alpha = imgui.slider_float("Min Alpha", orbit_min_alpha, 0.0, 1.0, "%.2f")
                        if changed_oma:
                            self.camera["orbit_min_alpha"] = orbit_min_alpha
                            settings_changed = True
                        
                    # Habitable Zones
                    changed_hz, show_habitable_zone = imgui.checkbox("Show Habitable Zones", show_habitable_zone)
                    if changed_hz:
                        self.camera["show_habitable_zone"] = show_habitable_zone
                        settings_changed = True
                        
                    imgui.separator()
                    # Atmosphere & Refraction toggles
                    changed_atmo, self.camera["atmo_enabled"] = imgui.checkbox("Enable Atmosphere Rendering", self.camera.get("atmo_enabled", True))
                    if changed_atmo:
                        settings_changed = True
                    changed_refr, self.camera["refraction_enabled"] = imgui.checkbox("Enable Atmospheric Refraction", self.camera.get("refraction_enabled", True))
                    if changed_refr:
                        settings_changed = True

                    imgui.separator()
                    # Bounce lighting toggles
                    changed_ps, self.camera["planetshine_enabled"] = imgui.checkbox("Enable Planetshine/Moonshine", self.camera.get("planetshine_enabled", True))
                    if changed_ps:
                        settings_changed = True
                    changed_rs, self.camera["ringshine_enabled"] = imgui.checkbox("Enable Ringshine", self.camera.get("ringshine_enabled", True))
                    if changed_rs:
                        settings_changed = True
                    if self.camera.get("ringshine_enabled", True):
                        imgui.indent()
                        changed_rsb, new_rsb = imgui.slider_int("Ringshine Bands", int(self.camera.get("ringshine_band_count", 256)), 4, 1024)
                        if changed_rsb:
                            self.camera["ringshine_band_count"] = new_rsb
                            settings_changed = True
                        imgui.unindent()

                    
                    imgui.separator()
                    # Dynamic Texture Streaming
                    stream_thresh = float(self.camera.get("tex_stream_threshold_px", 500.0))
                    changed_st, stream_thresh = imgui.slider_float("Min High-Res Size (px)", stream_thresh, 100.0, 2000.0, "%.0f px")
                    if changed_st:
                        self.camera["tex_stream_threshold_px"] = stream_thresh
                        settings_changed = True

                    if settings_changed:
                        self.save_settings()

                    if imgui.button("Close"):
                        self.camera["show_settings_modal"] = False
                imgui.end()
    
            imgui.spacing()
            imgui.spacing()
            imgui.text_colored("System Hierarchy", 0.6, 0.9, 1.0)
            imgui.separator()
            
            # ── System Selector ──
            imgui.text_colored(active_system_name, 0.6, 0.9, 1.0)
            imgui.same_line()
            imgui.set_cursor_pos_x(imgui.get_window_width() - 45)
            if imgui.button("...##sys_menu"):
                imgui.open_popup("SystemMenuPopup")
                
            if imgui.begin_popup("SystemMenuPopup"):
                imgui.text_colored("Switch System:", 0.6, 0.9, 1.0)
                imgui.separator()
                system_list = sys_mgr.list_systems()
                for si, sname in enumerate(system_list):
                    is_selected = (sname == active_system_name)
                    if imgui.selectable(sname, is_selected)[0]:
                        if sname != active_system_name:
                            target_name = sname
                            switch_req_name = target_name
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
    
                imgui.separator()
                if imgui.selectable("+ Create New System")[0]:
                    self.camera["show_create_system"] = True
                    self.camera["create_sys_data"] = {
                        "system_name": "New System",
                        "star_name": "Star",
                        "mass": 1.0,
                        "metallicity": 0.0,
                        "age_pct": 46.0,
                    }
                
                if active_system_name != sys_mgr.SOLAR_SYSTEM_NAME:
                    imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.4, 0.4)
                    if imgui.selectable("- Delete Current System")[0]:
                        sys_mgr.delete_system(active_system_name)
                        target_name = sys_mgr.SOLAR_SYSTEM_NAME
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
                    imgui.pop_style_color()
                    
                imgui.end_popup()
    
            imgui.separator()
            
            # ── Ephemeris Mode UI ──
            def _render_ephem_setup_modal():
                if imgui.begin_popup_modal("Ephemeris Setup", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                    imgui.text("Select which SPICE kernels to download and load:")
                    imgui.text_colored("Warning: High resolution satellite kernels can be large and take time to download.", 1.0, 0.5, 0.2)
                    imgui.separator()
                    
                    import os
                    groups = {
                        "Required Core": ["naif0012.tls", "pck00010.tpc", "de440s.bsp"],
                        "Mars Moons": ["mar099s.bsp"],
                        "Jupiter Moons": ["jup365.bsp", "jup347.bsp", "jup348.bsp", "jup349.bsp"],
                        "Saturn Moons": ["sat441.bsp", "sat455.bsp", "sat456.bsp", "sat457.bsp", "sat459.bsp"],
                        "Uranus Moons": ["ura111.bsp"],
                        "Neptune Moons": ["nep095.bsp", "nep105.bsp", "nep104.bsp"],
                        "Pluto System": ["plu060.bsp"]
                    }
                    
                    for group_name, k_list in groups.items():
                        imgui.text_colored(f"--- {group_name} ---", 0.7, 0.9, 1.0)
                        for k_name in k_list:
                            if k_name not in sys_mgr_spice.DEFAULT_KERNELS:
                                continue
                            desc = sys_mgr_spice.KERNEL_DESCRIPTIONS.get(k_name, k_name)
                            if os.path.exists(os.path.join(sys_mgr_spice.KERNEL_DIR, k_name)):
                                desc += " [Downloaded]"
                            is_enabled = sys_mgr_spice.enabled_kernels.get(k_name, False)
                            changed, new_val = imgui.checkbox(desc, is_enabled)
                            if changed:
                                sys_mgr_spice.enabled_kernels[k_name] = new_val
                                
                    imgui.separator()
                    if ephemeris_mode_active:
                        if imgui.button("Save & Apply Kernel Changes"):
                            sys_mgr_spice.save_settings()
                            self._ephem_mapping_ver = None
                            imgui.close_current_popup()
                            self._show_ephem_download_modal = True
                            sys_mgr_spice.download_kernels_async(on_complete=lambda: sys_mgr_spice.load_kernels(force=True))
                    else:
                        if imgui.button("Save & Switch to Ephemeris Mode"):
                            sys_mgr_spice.save_settings()
                            self._ephem_mapping_ver = None
                            imgui.close_current_popup()
                            _trigger_ephem_switch()
                    imgui.same_line()
                    if imgui.button("Cancel"):
                        imgui.close_current_popup()
                    imgui.end_popup()

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

            # ── Simulation Physics Mode ──
            imgui.text_colored("Simulation Physics Mode", 0.6, 0.9, 1.0)
            cur_mode = 0
            if keplerian_mode_active:
                cur_mode = 1
            elif ephemeris_mode_active:
                cur_mode = 2

            c0 = imgui.radio_button("N-Body", cur_mode == 0)
            imgui.same_line()
            c1 = imgui.radio_button("Keplerian", cur_mode == 1)
            if active_system_name == SystemManager.SOLAR_SYSTEM_NAME or ephemeris_mode_active:
                imgui.same_line()
                c2 = imgui.radio_button("Ephemeris", cur_mode == 2)
            else:
                c2 = False

            if c0 and cur_mode != 0:
                if cur_mode == 2:
                    _trigger_ephem_exit(to_keplerian=False)
                else:
                    ephemeris_mode_active = False
                    keplerian_mode_active = False
                    self.shared_state["ephemeris_mode"] = False
                    self.shared_state["keplerian_mode"] = False
            elif c1 and cur_mode != 1:
                if cur_mode == 2:
                    _trigger_ephem_exit(to_keplerian=True)
                else:
                    ephemeris_mode_active = False
                    keplerian_mode_active = True
                    self.shared_state["ephemeris_mode"] = False
                    self.shared_state["keplerian_mode"] = True
                    with self.shared_state["lock"]:
                        self.shared_state["keplerian_reextract"] = True
            elif c2 and cur_mode != 2:
                if not sys_mgr_spice.settings_initialized:
                    imgui.open_popup("Ephemeris Setup")
                else:
                    _trigger_ephem_switch()

            if keplerian_mode_active:
                imgui.text_colored("ANALYTICAL KEPLERIAN MODE ACTIVE", 0.3, 1.0, 0.3)
                if imgui.button("Export & Switch to N-Body Mode", width=-1):
                    ephemeris_mode_active = False
                    keplerian_mode_active = False
                    self.shared_state["ephemeris_mode"] = False
                    self.shared_state["keplerian_mode"] = False
                    with self.shared_state["lock"]:
                        self.shared_state["keplerian_export"] = True
                imgui.separator()

            if active_system_name == SystemManager.SOLAR_SYSTEM_NAME and not ephemeris_mode_active:
                if imgui.button("Switch to Ephemeris Mode", width=-1):
                    if not sys_mgr_spice.settings_initialized:
                        imgui.open_popup("Ephemeris Setup")
                    else:
                        _trigger_ephem_switch()
                        
                if imgui.button("Ephemeris Kernel Settings", width=-1):
                    imgui.open_popup("Ephemeris Setup")
                    
                _render_ephem_setup_modal()
                
                if getattr(self, "_show_ephem_download_modal", False) or sys_mgr_spice.is_downloading or sys_mgr_spice.download_error:
                    imgui.open_popup("Downloading SPICE Ephemeris Data")

                if imgui.begin_popup_modal("Downloading SPICE Ephemeris Data", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                    if sys_mgr_spice.is_downloading:
                        imgui.text_colored("Downloading NASA JPL SPICE Kernels...", 0.4, 0.8, 1.0)
                        imgui.text(sys_mgr_spice.download_status)
                        
                        if sys_mgr_spice.download_bytes_total > 0:
                            mb_cur = sys_mgr_spice.download_bytes_current / (1024 * 1024)
                            mb_tot = sys_mgr_spice.download_bytes_total / (1024 * 1024)
                            spd = sys_mgr_spice.download_speed_str
                            imgui.text(f"File Progress: {mb_cur:.1f} MB / {mb_tot:.1f} MB  ({spd})")
                        elif sys_mgr_spice.download_speed_str:
                            imgui.text(f"Speed: {sys_mgr_spice.download_speed_str}")
                            
                        imgui.progress_bar(sys_mgr_spice.download_progress, size=(340, 0))
                        imgui.separator()
                        if imgui.button("Cancel Download", width=-1):
                            sys_mgr_spice.cancel_download()
                            self._show_ephem_download_modal = False
                            imgui.close_current_popup()
                    elif sys_mgr_spice.download_error:
                        imgui.text_colored("Download Failed!", 1.0, 0.3, 0.3)
                        imgui.text_wrapped(sys_mgr_spice.download_error)
                        imgui.separator()
                        if imgui.button("Close", width=-1):
                            sys_mgr_spice.download_error = None
                            self._show_ephem_download_modal = False
                            imgui.close_current_popup()
                    else:
                        self._show_ephem_download_modal = False
                        imgui.close_current_popup()
                    imgui.end_popup()
                imgui.separator()
                
            elif ephemeris_mode_active:
                imgui.text_colored("EPHEMERIS MODE ACTIVE", 0.3, 1.0, 0.3)
                if imgui.button("Ephemeris Kernel Settings", width=-1):
                    imgui.open_popup("Ephemeris Setup")
                    
                _render_ephem_setup_modal()
                
                if getattr(self, "_show_ephem_download_modal", False) or sys_mgr_spice.is_downloading or sys_mgr_spice.download_error:
                    imgui.open_popup("Downloading SPICE Ephemeris Data")

                if imgui.begin_popup_modal("Downloading SPICE Ephemeris Data", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                    if sys_mgr_spice.is_downloading:
                        imgui.text_colored("Downloading NASA JPL SPICE Kernels...", 0.4, 0.8, 1.0)
                        imgui.text(sys_mgr_spice.download_status)
                        
                        if sys_mgr_spice.download_bytes_total > 0:
                            mb_cur = sys_mgr_spice.download_bytes_current / (1024 * 1024)
                            mb_tot = sys_mgr_spice.download_bytes_total / (1024 * 1024)
                            spd = sys_mgr_spice.download_speed_str
                            imgui.text(f"File Progress: {mb_cur:.1f} MB / {mb_tot:.1f} MB  ({spd})")
                        elif sys_mgr_spice.download_speed_str:
                            imgui.text(f"Speed: {sys_mgr_spice.download_speed_str}")
                            
                        imgui.progress_bar(sys_mgr_spice.download_progress, size=(340, 0))
                        imgui.separator()
                        if imgui.button("Cancel Download", width=-1):
                            sys_mgr_spice.cancel_download()
                            self._show_ephem_download_modal = False
                            imgui.close_current_popup()
                    elif sys_mgr_spice.download_error:
                        imgui.text_colored("Download Failed!", 1.0, 0.3, 0.3)
                        imgui.text_wrapped(sys_mgr_spice.download_error)
                        imgui.separator()
                        if imgui.button("Close", width=-1):
                            sys_mgr_spice.download_error = None
                            self._show_ephem_download_modal = False
                            imgui.close_current_popup()
                    else:
                        self._show_ephem_download_modal = False
                        imgui.close_current_popup()
                    imgui.end_popup()

                if imgui.button("Export to N-Body System", width=-1):
                    capture_t = display_t
                    capture_paused = self.time_ctrl["paused"]
                    capture_speed = self.time_ctrl["multiplier"]
                    
                    epoch_dt = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
                    et_epoch = sys_mgr_spice.datetime_to_et(epoch_dt)
                    et = et_epoch + capture_t * 365.25 * 86400.0
                    ephem_bundle = sys_mgr_spice.build_ephemeris_system(et, template_bodies=bodies_data)
                    export_name = f"Solar System ({cur_y:04d}-{cur_m:02d}-{cur_d:02d})"
                    sys_mgr.save_system_data(export_name, ephem_bundle)
                    sys_mgr.save_meta(export_name, ephem_bundle)
                    
                    new_bndl = load_system_from_data(ephem_bundle)
                    switch_req_name = export_name
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
                        "preserve_t": capture_t,
                        "preserve_paused": capture_paused,
                        "preserve_speed": capture_speed
                    }
                    with self.shared_state["lock"]:
                        self.shared_state["system_switch_request"] = req
                    
                if imgui.button("Return to N-Body Mode", width=-1):
                    _trigger_ephem_exit(to_keplerian=False)
                imgui.separator()
            
            for k in range(len(tree_indices_snap)):
    
                idx = int(tree_indices_snap[k])
                if ephemeris_mode_active and not spice_valid_mask[idx]:
                    continue
                
                depth = int(tree_depths_snap[k])
                indent = depth * 15
                if indent > 0: imgui.indent(indent)
                is_inspected = (self.camera["inspected_idx"] == idx and not self.camera.get("inspected_is_cmp", False))
                label = f"{bodies_data[idx]['name']}"
                if self.camera["tracking_idx"] == idx and not self.camera.get("tracking_is_cmp", False):
                    label += " *"
                    
                avail_w = imgui.get_content_region_available()[0]
                clicked = imgui.selectable(label + f"##{idx}", is_inspected, 0, max(10.0, avail_w - 65.0))[0]
                
                rect_min_y = imgui.get_item_rect_min()[1]
                rect_max_y = imgui.get_item_rect_max()[1]
                rect_min_x = imgui.get_window_position()[0]
                rect_max_x = rect_min_x + imgui.get_window_width()
                mouse_x, mouse_y = imgui.get_io().mouse_pos
                
                show_buttons = (rect_min_y <= mouse_y <= rect_max_y) and (rect_min_x <= mouse_x <= rect_max_x)
                    
                if clicked:
                    if self.camera["inspected_idx"] == idx and not self.camera.get("inspected_is_cmp", False):
                        if not self.camera["inspect_bary"]:
                            self.camera["inspect_bary"] = True
                        else:
                            self.camera["inspected_idx"] = None
                            self.camera["inspect_bary"] = False
                    else:
                        self.camera["inspected_idx"] = idx
                        self.camera["inspected_is_cmp"] = False
                        self.camera["inspect_bary"] = False
                
                if show_buttons:
                    imgui.same_line()
                    imgui.set_cursor_pos_x(imgui.get_window_width() - 65)
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                    
                    current_y = imgui.get_cursor_pos_y()
                    imgui.set_cursor_pos_y(current_y - 2)
                    if imgui.button(f"+##add_{idx}", 18, 17):
                        is_moon = mass_snap[idx] < 0.01
                        self.camera["add_mode"] = True
                        self.camera["add_data"] = {
                            "name": f"{'Moon' if is_moon else 'Planet'} of {bodies_data[idx]['name']}",
                            "type": "Moon" if is_moon else "Terrestrial",
                            "mass": 1.0,
                            "radius": 1737.0 if is_moon else 6371.0,
                            "color": [0.7, 0.7, 0.7] if is_moon else [0.2, 0.5, 0.8],
                            "a": 384400.0 if is_moon else 1.0,
                            "e": 0.0, "inc": 0.0, "Omega": 0.0, "omega": 0.0, "M": 0.0,
                            "frame": 0,
                            "is_moon": is_moon,
                            "parent_idx": idx
                        }
                    
                    if idx > 0:
                        imgui.same_line()
                        imgui.set_cursor_pos_y(current_y - 2)
                        imgui.push_style_color(imgui.COLOR_BUTTON, 0.6, 0.1, 0.1)
                        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.8, 0.2, 0.2)
                        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.9, 0.3, 0.3)
                        if imgui.button(f"-##del_{idx}", 18, 17):
                            with self.shared_state["lock"]:
                                self.shared_state["crud_queue"].append({
                                    "action": "DELETE",
                                    "idx": idx
                                })
                            self.camera["inspected_idx"] = None
                            self.camera["inspect_bary"] = False
                        imgui.pop_style_color(3)
                    imgui.pop_style_var(1)
                        
                if indent > 0: imgui.unindent(indent)
                
            if self.comparison_enabled and self.num_bodies_cmp > 0:
                imgui.separator()
                imgui.text_colored(f"{self.comparison_system_name} (Comparison)", 0.6, 0.9, 1.0)
                imgui.separator()
                for k in range(len(self.tree_indices_snap_cmp)):
                    idx = int(self.tree_indices_snap_cmp[k])
                    depth = int(self.tree_depths_snap_cmp[k])
                    indent = depth * 15
                    if indent > 0: imgui.indent(indent)
                    
                    is_inspected = (self.camera["inspected_idx"] == idx and self.camera.get("inspected_is_cmp", False))
                    label = f"{self.bodies_data_cmp[idx]['name']}"
                    if self.camera["tracking_idx"] == idx and self.camera.get("tracking_is_cmp", False):
                        label += " *"
                        
                    avail_w = imgui.get_content_region_available()[0]
                    clicked = imgui.selectable(label + f"##cmp_{idx}", is_inspected, 0, max(10.0, avail_w - 10.0))[0]
                    
                    if clicked:
                        if self.camera["inspected_idx"] == idx and self.camera.get("inspected_is_cmp", False):
                            if not self.camera["inspect_bary"]:
                                self.camera["inspect_bary"] = True
                            else:
                                self.camera["inspected_idx"] = None
                                self.camera["inspected_is_cmp"] = False
                                self.camera["inspect_bary"] = False
                        else:
                            self.camera["inspected_idx"] = idx
                            self.camera["inspected_is_cmp"] = True
                            self.camera["inspect_bary"] = False
                            
                    if indent > 0: imgui.unindent(indent)

            imgui.separator()
            if imgui.collapsing_header("System Comparison", flags=imgui.TREE_NODE_DEFAULT_OPEN if self.comparison_enabled else 0)[0]:
                changed_enabled, self.comparison_enabled = imgui.checkbox("Enable Comparison##cmp_enable", self.comparison_enabled)
                
                imgui.text("Compare With:")
                system_list = sys_mgr.list_systems()
                try:
                    current_idx = system_list.index(self.comparison_system_name)
                except ValueError:
                    current_idx = 0
                
                imgui.push_item_width(-1)
                changed_sys, new_idx = imgui.combo("##comp_sys_combo", current_idx, system_list)
                imgui.pop_item_width()
                
                if changed_sys:
                    target_name = system_list[new_idx]
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
                
                changed_offset, self.comparison_offset_au = imgui.drag_float("Offset (AU)##cmp_offset", self.comparison_offset_au, 0.1)

            imgui.separator()
            if imgui.collapsing_header("Distance Unit Settings")[0]:
                cur_thresh_ly = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU) / LY_TO_AU
                changed_thresh, new_thresh_ly = imgui.drag_float("AU -> Light Year (ly)##ly_thresh", cur_thresh_ly, 0.01, 0.001, 1000.0, format="%.3f ly")
                if changed_thresh:
                    self.camera["ly_threshold_au"] = max(0.0001, new_thresh_ly) * LY_TO_AU
                thresh_au_val = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
                imgui.text_colored(f"Switch Threshold: {thresh_au_val:,.1f} AU ({thresh_au_val/LY_TO_AU:.3f} ly)", 0.7, 0.7, 0.7)

            imgui.end()
    
            insp_is_cmp = self.camera.get("inspected_is_cmp", False)
            insp_idx = self.camera["inspected_idx"]
            limit_bodies = self.num_bodies_cmp if insp_is_cmp else num_bodies
            if insp_idx is not None and 0 <= insp_idx < limit_bodies:
                inspect_bary = self.camera["inspect_bary"]
                cur_bodies_data = self.bodies_data_cmp if insp_is_cmp else bodies_data
                cur_parent_snap = self.parent_snap_cmp if insp_is_cmp else parent_snap
                cur_mass_snap = self.mass_snap_cmp if insp_is_cmp else mass_snap
                cur_subsys_mass_buf = self.subsys_mass_buf_cmp if insp_is_cmp else subsys_mass_buf
                cur_subsys_pos_buf = self.subsys_pos_buf_cmp if insp_is_cmp else subsys_pos_buf
                cur_subsys_vel_buf = self.subsys_vel_buf_cmp if insp_is_cmp else subsys_vel_buf
                cur_pos_snap_render = self.pos_snap_cmp if insp_is_cmp else pos_snap_render
                cur_vel_snap_render = self.vel_snap_cmp if insp_is_cmp else vel_snap_render
                cur_visual_arr = self.visual_data_cmp if insp_is_cmp else visual_arr
                
                body_info = cur_bodies_data[insp_idx]
                body_name = body_info['name']
                parent_idx = int(cur_parent_snap[insp_idx])
                
                if inspect_bary:
                    win_title = f"{body_name} System Barycenter###inspector"
                else:
                    win_title = f"{body_name}###inspector"
                
                imgui.set_next_window_position(self.fb_width - 350, 10, cond_layout)
                imgui.set_next_window_size(340, self.fb_height - 20, cond_layout)
                expanded, opened = imgui.begin(win_title, True)
                if not opened:
                    self.camera["inspected_idx"] = None
                    self.camera["inspect_bary"] = False
                elif expanded:
                    if inspect_bary:
                        imgui.text_colored("Barycenter Mode", 0.6, 0.9, 1.0)
                    else:
                        obj_type = body_info.get('type', 'Unknown')
                        imgui.text_colored(f"Type: {obj_type}", 0.7, 0.7, 0.7)
                        if not insp_is_cmp:
                            imgui.same_line(spacing=15)
                            changed, self.camera["edit_mode"] = imgui.checkbox("Edit Mode", self.camera["edit_mode"])
                            if changed and self.camera["edit_mode"]:
                                self.time_ctrl["multiplier"] = 0.0
                                sp = body_info.get("star_props", {})
                                self.camera["edit_data"] = {
                                    "mass": float(mass_snap[insp_idx]),
                                    "radius": float(body_info.get('r', 0.0) * 696340.0),
                                    "rotation_period": float(body_info.get("rotation_period", 0.0)),
                                    "type": body_info.get("type", "Moon"),
                                    "a": 0.0, "e": 0.0, "inc": 0.0, "Omega": 0.0, "omega": 0.0, "M": 0.0,
                                    "init_orbit": True,
                                    "star_mode": sp.get("mode", "evolution"),
                                    "metallicity": sp.get("metallicity", 0.0),
                                    "age_pct": sp.get("age_pct", 0.46) * 100.0,
                                    "s_rad": sp.get("radius", float(body_info.get('r', 1.0))),
                                    "s_temp": sp.get("temp", 5778.0),
                                    "s_lum": sp.get("lum", 1.0),
                                    "locked_rad": sp.get("locked_rad", True),
                                    "locked_temp": sp.get("locked_temp", True),
                                    "locked_lum": sp.get("locked_lum", False)
                                }
                    
                    imgui.separator()
                    imgui.text_colored("Physical Properties", 1.0, 0.85, 0.4)
                    
                    body_mass = cur_mass_snap[insp_idx]
                    body_r_km = body_info.get('r', 0.0) * 696340.0
                    
                    if not insp_is_cmp and self.camera["edit_mode"] and not inspect_bary:
                        ed = self.camera["edit_data"]
                        types_list = ["Star", "Terrestrial", "Gas Giant", "Ice Giant", "Dwarf Planet", "Moon"]
                        if "type" not in ed: ed["type"] = body_info.get("type", "Moon")
                        type_idx = types_list.index(ed["type"]) if ed["type"] in types_list else 1
                        changed_t, type_idx = imgui.combo("Type", type_idx, types_list)
                        if changed_t: ed["type"] = types_list[type_idx]
                        
                        if ed["type"] == "Star":
                            changed_m, mode_idx = imgui.combo("Star Mode", 0 if ed.get("star_mode", "evolution")=="evolution" else 1, ["Evolution Track", "Surface Physics"])
                            if changed_m: ed["star_mode"] = ["evolution", "surface"][mode_idx]
                            
                            if ed["star_mode"] == "evolution":
                                _, ed["mass"] = imgui.drag_float(u"Mass (M\u2609)", ed["mass"], 0.01, 0.01, 300.0, format="%.4f")
                                _, ed["metallicity"] = imgui.drag_float("[Fe/H] (dex)", ed.get("metallicity", 0.0), 0.01, -4.0, 1.0, format="%.3f")
                                _, ed["age_pct"] = imgui.drag_float("Life Cycle (%)", ed.get("age_pct", 46.0), 0.1, -5.0, 120.0, format="%.1f%%")
                                
                                if ed["mass"] >= 45.0:
                                    imgui.text_colored("Humphreys-Davidson Limit Reached", 1.0, 0.5, 0.2)
                                    ed["evo_path"] = "stripping"
                                elif ed["mass"] >= 25.0:
                                    if "evo_path" not in ed: ed["evo_path"] = "standard"
                                    changed_p, p_idx = imgui.combo("Evolution Branch", 0 if ed["evo_path"]=="standard" else 1, ["Red Supergiant", "LBV -> Wolf-Rayet"])
                                    if changed_p: ed["evo_path"] = ["standard", "stripping"][p_idx]
                                else:
                                    ed["evo_path"] = "standard"
                                
                                try:
                                    from engine.physics.star_calc import StarCalculator
                                    age_pct = ed["age_pct"] / 100.0
                                    preview = StarCalculator.forge(mode="evolution", evo_path=ed.get("evo_path", "standard"), mass=ed["mass"], metallicity=ed["metallicity"], age_pct=age_pct)
                                    ed["radius"] = preview["physical"]["radius_rsun"] * 696340.0
                                    ed["_preview"] = preview
                                except Exception as e:
                                    import traceback
                                    traceback.print_exc()
                                    ed["_preview"] = None
                            else:
                                if "locked_rad" not in ed: ed["locked_rad"] = True
                                if "locked_temp" not in ed: ed["locked_temp"] = True
                                if "locked_lum" not in ed: ed["locked_lum"] = False
                                num_locked = ed["locked_rad"] + ed["locked_temp"] + ed["locked_lum"]
                                
                                c1, ed["locked_rad"] = imgui.checkbox("##lr", ed["locked_rad"]); imgui.same_line(); _, ed["s_rad"] = imgui.drag_float(u"Radius (R\u2609)", ed.get("s_rad", 1.0), 0.05, 0.01, 2500.0, format="%.4f")
                                c2, ed["locked_temp"] = imgui.checkbox("##lt", ed["locked_temp"]); imgui.same_line(); _, ed["s_temp"] = imgui.drag_float("Temp (K)", ed.get("s_temp", 5778.0), 10.0, 1000.0, 100000.0, format="%.0f")
                                c3, ed["locked_lum"] = imgui.checkbox("##ll", ed["locked_lum"]); imgui.same_line(); _, ed["s_lum"] = imgui.drag_float(u"Luminosity (L\u2609)", ed.get("s_lum", 1.0), 0.01, 0.0001, 2000000.0, format="%.4f")
                                
                                if c1 and ed["locked_rad"] and num_locked == 2:
                                    if ed["locked_temp"] and not c2: ed["locked_temp"] = False
                                    elif ed["locked_lum"] and not c3: ed["locked_lum"] = False
                                elif c2 and ed["locked_temp"] and num_locked == 2:
                                    if ed["locked_rad"] and not c1: ed["locked_rad"] = False
                                    elif ed["locked_lum"] and not c3: ed["locked_lum"] = False
                                elif c3 and ed["locked_lum"] and num_locked == 2:
                                    if ed["locked_rad"] and not c1: ed["locked_rad"] = False
                                    elif ed["locked_temp"] and not c2: ed["locked_temp"] = False
                                    
                                if ed["locked_rad"] + ed["locked_temp"] + ed["locked_lum"] != 2:
                                    imgui.text_colored("Must lock exactly 2 properties!", 1.0, 0.3, 0.3)
                                    ed["_preview"] = None
                                else:
                                    try:
                                        from engine.physics.star_calc import StarCalculator
                                        preview = StarCalculator.forge(mode="surface", mass=1.0, metallicity=0.0,
                                            radius=ed["s_rad"] if ed["locked_rad"] else None,
                                            temp=ed["s_temp"] if ed["locked_temp"] else None,
                                            lum=ed["s_lum"] if ed["locked_lum"] else None)
                                        ed["radius"] = preview["physical"]["radius_rsun"] * 696340.0
                                        ed["mass"] = preview["physical"]["mass_msun"]
                                        ed["_preview"] = preview
                                    except Exception as e:
                                        import traceback
                                        traceback.print_exc()
                                        ed["_preview"] = None
                            
                            edit_mass = ed["mass"]
                            edit_r_km = ed["radius"]
                        else:
                            _, ed["mass"] = imgui.input_double(u"Mass (M\u2609)", ed["mass"], format="%e")
                            _, ed["radius"] = imgui.input_double("Radius (km)", ed["radius"], format="%.1f")
                            
                            # Check if tidally locked based on edit_data and parent
                            is_locked = False
                            if parent_idx >= 0:
                                parent_m = cur_mass_snap[parent_idx]
                                body_m = ed["mass"]
                                r_km = ed["radius"]
                                if body_m > 0 and r_km > 0:
                                    r_tid = 0.00084 * ((parent_m**2 / body_m)**(1.0/6.0)) * math.sqrt(r_km)
                                    a_val = ed.get("a", body_info.get("a", 1.0))
                                    if a_val < r_tid:
                                        is_locked = True
                                        total_m = parent_m + body_m
                                        p_years = math.sqrt(a_val**3 / total_m) if total_m > 0 else 0.0
                                        ed["rotation_period"] = p_years * 365.25 * 24.0
                            
                            if is_locked:
                                imgui.text("  Rot Period: {:.4f} hours [Tidally Locked]".format(ed["rotation_period"]))
                            else:
                                _, ed["rotation_period"] = imgui.input_double("Rot Period (hours)", ed["rotation_period"], format="%.4f")
                            edit_mass = ed["mass"]
                            edit_r_km = ed["radius"]
                            
                        if edit_r_km > 0.0:
                            g_m_s2 = (1.32712440018e14 * edit_mass) / (edit_r_km ** 2)
                            g_earth = g_m_s2 / 9.80665
                            if g_m_s2 >= 1e-4:
                                imgui.text("  Surface G: {:.3f} m/s² ({:.3f} g)".format(g_m_s2, g_earth))
                            else:
                                imgui.text("  Surface G: {:.3e} m/s² ({:.3e} g)".format(g_m_s2, g_earth))
                    elif inspect_bary:
                        bary_mass = cur_subsys_mass_buf[insp_idx]
                        if bary_mass > 1e-4:
                            imgui.text(u"  System Mass:  {bary_mass:.6e} M\u2609".format(bary_mass=bary_mass))
                        else:
                            m_earth = bary_mass / 3.003e-6
                            if m_earth < 0.1:
                                m_lunar = bary_mass / 3.694e-8
                                imgui.text(u"  System Mass:  {m_lunar:.4f} M\u263E".format(m_lunar=m_lunar))
                            else:
                                imgui.text(u"  System Mass:  {m_earth:.4f} M\u2295".format(m_earth=m_earth))
                    else:
                        if body_mass > 1e-4:
                            imgui.text(u"  Mass:    {body_mass:.6e} M\u2609".format(body_mass=body_mass))
                        elif body_mass > 1e-10:
                            m_earth = body_mass / 3.003e-6
                            if m_earth < 0.1:
                                m_lunar = body_mass / 3.694e-8
                                imgui.text(u"  Mass:    {m_lunar:.4f} M\u263E".format(m_lunar=m_lunar))
                            else:
                                imgui.text(u"  Mass:    {m_earth:.6f} M\u2295".format(m_earth=m_earth))
                        else:
                            m_lunar = body_mass / 3.694e-8
                            imgui.text(u"  Mass:    {m_lunar:.4e} M\u263E".format(m_lunar=m_lunar))
                    if body_r_km > 100:
                        imgui.text(f"  Radius:  {body_r_km:,.0f} km")
                    elif body_r_km > 0.1:
                        imgui.text(f"  Radius:  {body_r_km:.1f} km")
                    if body_r_km > 0.0:
                        g_m_s2 = (1.32712440018e14 * body_mass) / (body_r_km ** 2)
                        g_earth = g_m_s2 / 9.80665
                        if g_m_s2 >= 1e-4:
                            imgui.text("  Surface G: {:.3f} m/s² ({:.3f} g)".format(g_m_s2, g_earth))
                        else:
                            imgui.text("  Surface G: {:.3e} m/s² ({:.3e} g)".format(g_m_s2, g_earth))
                    
                    thresh_au = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
                    target_pos = cur_subsys_pos_buf[insp_idx].copy() if inspect_bary else cur_pos_snap_render[insp_idx].copy()
                    if insp_is_cmp:
                        target_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                    dist_to_center_km = np.linalg.norm(target_pos - cam_world_pos_f8) * 149597870.7
                    dist_to_center_au = dist_to_center_km / 149597870.7
                    if inspect_bary:
                        if dist_to_center_km > 1.49597e7:
                            imgui.text(f"  Cam Dist: {format_distance_au(dist_to_center_au, threshold_au=thresh_au)}")
                        else:
                            imgui.text(f"  Cam Dist: {dist_to_center_km:,.0f} km")
                    else:
                        # Altitude above the oblate ellipsoid surface along the
                        # camera's direction (a plain sphere radius would be wrong
                        # near the poles of a flattened body).
                        to_cam = cam_world_pos_f8 - target_pos
                        n_tc = np.linalg.norm(to_cam)
                        if n_tc > 1e-12:
                            if insp_is_cmp and hasattr(self, "pole_n_arr_cmp") and self.pole_n_arr_cmp is not None and insp_idx < len(self.pole_n_arr_cmp):
                                pole_ref = np.array(self.pole_n_arr_cmp[insp_idx], dtype='f8')
                            elif not insp_is_cmp and hasattr(self, "pole_n_arr") and self.pole_n_arr is not None and insp_idx < len(self.pole_n_arr):
                                pole_ref = np.array(self.pole_n_arr[insp_idx], dtype='f8')
                            else:
                                pole_ref = np.array([0.0, 1.0, 0.0], dtype='f8')
                            surf_r_km = _ellipsoid_surface_radius(body_r_km, float(body_info.get('oblateness', 0.0)), pole_ref, to_cam / n_tc)
                            dist_to_surface_km = n_tc * 149597870.7 - surf_r_km
                        else:
                            dist_to_surface_km = dist_to_center_km - body_r_km
                        dist_to_surface_au = dist_to_surface_km / 149597870.7
                        if dist_to_center_km > 1.49597e7:
                            imgui.text(f"  Cam Dist: {format_distance_au(dist_to_center_au, threshold_au=thresh_au)}")
                        else:
                            imgui.text(f"  Cam Dist: {dist_to_center_km:,.0f} km")
                        if dist_to_surface_km > 1.49597e7:
                            imgui.text(f"  Altitude: {format_distance_au(dist_to_surface_au, threshold_au=thresh_au)}")
                        elif dist_to_surface_km > 0:
                            imgui.text(f"  Altitude: {dist_to_surface_km:,.0f} km")
                        else:
                            imgui.text(f"  Altitude: 0 km (Surface)")
                    
                    if not inspect_bary:
                        if 'star_props' in body_info:
                            sp = body_info['star_props']
                            
                            imgui.separator()
                            imgui.text_colored("Stellar Properties", 1.0, 0.85, 0.4)
                            imgui.text(f"  Temperature: {sp.get('temp', 0.0):,.0f} K")
                            imgui.text(f"  Luminosity:  {sp.get('lum', 0.0):.4f} L\u2609")
                            imgui.text(f"  Spectral Cl: {sp.get('class', 'Unknown')}")
                            imgui.text(f"  Stage:       {sp.get('stage', 'Unknown')}")
                            
                            rot_frac = sp.get('rot_frac', 0.0)
                            if rot_frac > 0:
                                imgui.separator()
                                imgui.text_colored("Stellar Rotation", 1.0, 0.85, 0.4)
                                imgui.text(f"  Rot Period:  {sp.get('rotation_period', 0.0):.2f} hours")
                                imgui.text(f"  Eq Velocity: {sp.get('v_eq', 0.0):.2f} km/s")
                                imgui.text(f"  Eq Radius:   {sp.get('r_eq', sp.get('radius', 0.0)):.4f} R\u2609")
                                imgui.text(f"  Pole Radius: {sp.get('r_pole', sp.get('radius', 0.0)):.4f} R\u2609")
                                imgui.text(f"  Eq Temp:     {sp.get('t_eq', sp.get('temp', 0.0)):,.0f} K")
                                imgui.text(f"  Pole Temp:   {sp.get('t_pole', sp.get('temp', 0.0)):,.0f} K")
                                imgui.text(f"  Eq Lum:      {sp.get('lum_eq', sp.get('lum', 0.0)):.4f} L\u2609")
                                imgui.text(f"  Pole Lum:    {sp.get('lum_pole', sp.get('lum', 0.0)):.4f} L\u2609")
                            
                            if sp.get('mode') == 'evolution':
                                imgui.separator()
                                imgui.text_colored("Stellar Evolution", 1.0, 0.85, 0.4)
                                imgui.text(f"  Metallicity: {sp.get('metallicity', 0.0):.3f}")
                                age_gyr = sp.get('age', 0.0)
                                if age_gyr < 0.1:
                                    imgui.text(f"  Age:         {age_gyr * 1000.0:,.1f} Myr")
                                else:
                                    imgui.text(f"  Age:         {age_gyr:,.3f} Gyr")
                                    
                            imgui.separator()
                            imgui.text_colored("Observational Dynamics", 1.0, 0.85, 0.4)
                            lum = sp.get('lum', 1.0)
                            rot_frac = sp.get('rot_frac', 0.0)
                            
                            # Camera inclination relative to star pole
                            pole_vec = self.visual_arr_cmp[insp_idx, 5:8] if insp_is_cmp else visual_arr[insp_idx, 5:8]
                            to_cam = cam_world_pos_f8 - target_pos
                            dist_au = np.linalg.norm(to_cam)
                            if dist_au > 1e-12:
                                cam_dir = to_cam / dist_au
                            else:
                                cam_dir = np.array([0.0, 1.0, 0.0])
                                
                            cos_i_val = min(1.0, max(0.0, abs(np.dot(cam_dir, pole_vec))))
                            sin_i_val = math.sqrt(1.0 - cos_i_val**2)
                            viewing_inc_deg = math.degrees(math.acos(cos_i_val))
                            
                            if rot_frac > 0.0:
                                r_pole = sp.get('r_pole', sp.get('radius', 1.0))
                                r_eq = sp.get('r_eq', sp.get('radius', 1.0))
                                radius = sp.get('radius', 1.0)
                                t_pole = sp.get('t_pole', sp.get('temp', 5778.0))
                                t_eq = sp.get('t_eq', sp.get('temp', 5778.0))
                                temp = sp.get('temp', 5778.0)
                                
                                r_proj_y = math.sqrt((r_pole * sin_i_val)**2 + (r_eq * cos_i_val)**2)
                                apparent_area_ratio = (r_eq * r_proj_y) / (radius ** 2) if radius > 0 else 1.0
                                apparent_temp = t_pole * (1.0 - sin_i_val) + t_eq * sin_i_val
                                apparent_lum = lum * apparent_area_ratio * ((apparent_temp / temp) ** 4) if temp > 0 else 0
                                
                                imgui.text(f"  View Inclination: {viewing_inc_deg:.1f}°")
                                star_mag_lum = apparent_lum
                            else:
                                star_mag_lum = lum
                                
                            abs_mag = 4.83 - 2.5 * math.log10(max(star_mag_lum, 1e-10))
                            dist_au = dist_to_center_km / 149597870.7
                            dist_pc = dist_au / 206265.0
                            app_mag = abs_mag + 5.0 * math.log10(max(dist_pc, 1e-10)) - 5.0
                            imgui.text(f"  Abs Mag (M): {abs_mag:+.2f}")
                            imgui.text(f"  App Mag (m): {app_mag:+.2f}")
                        else:
                            A_g, A_b, q_val, _ = compute_body_albedos(body_info, insp_idx, insp_is_cmp)
                            p_v = A_g

                            if p_v > 0.0 and body_r_km > 0.0:
                                v_c = self.visual_arr_cmp[insp_idx, 0:3] if insp_is_cmp else visual_arr[insp_idx, 0:3]
                                _, A_surf, surf_src = compute_surface_albedo(body_info, self.texture_mean_colors, v_c)
                                src_label = "texture map" if surf_src == 'texture' else ("albedo scale" if surf_src == 'albedo_scale' else "base color")
                                imgui.text(f"  Surface Albedo: {A_surf:.3f} ({src_label})")
                                imgui.text(f"  Geom Albedo: {A_g:.3f}")
                                imgui.text(f"  Bond Albedo: {A_b:.3f} (q = {q_val:.2f})")

                                
                                is_solar = (self.comparison_system_name if insp_is_cmp else active_system_name) == SystemManager.SOLAR_SYSTEM_NAME
                                if is_solar:
                                    diam_km = body_r_km * 2.0
                                    abs_mag_h = 5.0 * math.log10(1329.0 / (diam_km * math.sqrt(p_v)))
                                    
                                    # Calculate system-specific absolute magnitude and apparent magnitude
                                    star_idx_local = -1
                                    for k, b in enumerate(cur_bodies_data):
                                        if b.get('type') == 'Star':
                                            star_idx_local = k
                                            break
                                            
                                    if star_idx_local != -1:
                                        star_body = cur_bodies_data[star_idx_local]
                                        star_lum = star_body.get('star_props', {}).get('lum', 1.0)
                                        
                                        # Direction from star to planet
                                        star_pos = cur_pos_snap_render[star_idx_local]
                                        planet_pos = cur_pos_snap_render[insp_idx]
                                        to_planet = planet_pos - star_pos
                                        d_sp = np.linalg.norm(to_planet)
                                        if d_sp > 1e-12:
                                            L_dir = to_planet / d_sp
                                        else:
                                            L_dir = np.array([0.0, 1.0, 0.0])
                                            
                                        star_pole = self.visual_arr_cmp[star_idx_local, 5:8] if insp_is_cmp else visual_arr[star_idx_local, 5:8]
                                        sin_lat = abs(np.dot(L_dir, star_pole))
                                        
                                        star_sp = star_body.get('star_props', {})
                                        if star_sp.get('rot_frac', 0.0) > 0.0:
                                            lum_eq = star_sp.get('lum_eq', star_lum)
                                            lum_pole = star_sp.get('lum_pole', star_lum)
                                            star_lum_dir = lum_eq * (1.0 - sin_lat) + lum_pole * sin_lat
                                        else:
                                            star_lum_dir = star_lum
                                            
                                        # System absolute magnitude accounts for local star illumination
                                        abs_mag_sys = abs_mag_h - 2.5 * math.log10(max(star_lum_dir, 1e-10))
                                        
                                        # Apparent magnitude accounts for distance to star, distance to camera, and phase angle
                                        to_cam = cam_world_pos_f8 - planet_pos
                                        d_pc = np.linalg.norm(to_cam)
                                        
                                        if d_sp > 1e-12 and d_pc > 1e-12:
                                            cos_alpha = np.dot(to_planet, -to_cam) / (d_sp * d_pc)
                                            alpha = math.acos(min(1.0, max(-1.0, cos_alpha)))
                                        else:
                                            alpha = 0.0
                                            
                                        # Lambertian phase function
                                        phi = max(1e-10, (1.0 - alpha / math.pi) * math.cos(alpha) + (1.0 / math.pi) * math.sin(alpha))
                                        app_mag = abs_mag_sys + 5.0 * math.log10(max(d_sp * d_pc, 1e-10)) - 2.5 * math.log10(phi)
                                        
                                        imgui.text(f"  Abs Mag (H): {abs_mag_sys:+.2f}")
                                        imgui.text(f"  App Mag (m): {app_mag:+.2f}")
                                
                            rot_period_hours = body_info.get('rotation_period', 0.0)
                            if self.camera["edit_mode"] and not insp_is_cmp:
                                rot_period_hours = self.camera["edit_data"].get("rotation_period", rot_period_hours)
                                current_r_km = self.camera["edit_data"].get("radius", body_r_km)
                            else:
                                current_r_km = body_r_km
                            
                            if rot_period_hours > 0:
                                f = body_info.get('oblateness', 0.0)
                                r_eq_km = current_r_km / ((1.0 - f) ** (1.0 / 3.0)) if f > 0 else current_r_km
                                r_pole_km = r_eq_km * (1.0 - f)
                                v_eq = (2.0 * math.pi * r_eq_km) / (rot_period_hours * 3600.0)
                                
                                imgui.text(f"  Rot Period:  {rot_period_hours:,.2f} hours")
                                imgui.text(f"  Eq Velocity: {v_eq:.4f} km/s")
                                if r_eq_km > 100:
                                    imgui.text(f"  Eq Radius:   {r_eq_km:,.1f} km")
                                    imgui.text(f"  Pole Radius: {r_pole_km:,.1f} km")
                                elif r_eq_km > 0.0:
                                    imgui.text(f"  Eq Radius:   {r_eq_km:.2f} km")
                                    imgui.text(f"  Pole Radius: {r_pole_km:.2f} km")
                                
                            # --- Polish calculations ---
                            star_idx_local = -1
                            for k, b in enumerate(cur_bodies_data):
                                if b.get('type') == 'Star':
                                    star_idx_local = k
                                    break
                                    
                            if star_idx_local != -1:
                                star_body = cur_bodies_data[star_idx_local]
                                star_lum = star_body.get('star_props', {}).get('lum', 1.0)
                                
                                # Direction from star to planet
                                star_pos = cur_pos_snap_render[star_idx_local]
                                planet_pos = cur_pos_snap_render[insp_idx]
                                to_planet = planet_pos - star_pos
                                dist_to_star_au = np.linalg.norm(to_planet)
                                if dist_to_star_au > 1e-12:
                                    L_dir = to_planet / dist_to_star_au
                                else:
                                    L_dir = np.array([0.0, 1.0, 0.0])
                                    
                                star_pole = self.visual_arr_cmp[star_idx_local, 5:8] if insp_is_cmp else visual_arr[star_idx_local, 5:8]
                                sin_lat = abs(np.dot(L_dir, star_pole))
                                
                                star_sp = star_body.get('star_props', {})
                                if star_sp.get('rot_frac', 0.0) > 0.0:
                                    lum_eq = star_sp.get('lum_eq', star_lum)
                                    lum_pole = star_sp.get('lum_pole', star_lum)
                                    star_lum_dir = lum_eq * (1.0 - sin_lat) + lum_pole * sin_lat
                                else:
                                    star_lum_dir = star_lum
                            else:
                                star_lum_dir = 1.0
                            
                            eff_a = 1.0
                            curr = insp_idx
                            while curr >= 0:
                                p_id = cur_parent_snap[curr]
                                if p_id == -1:
                                    break
                                p_body = cur_bodies_data[p_id]
                                if p_body.get('type') == 'Star':
                                    eff_a = cur_bodies_data[curr].get('a', 1.0)
                                    break
                                curr = p_id
                            
                            A_g_calc, A_b_calc, _, _ = compute_body_albedos(body_info, insp_idx, insp_is_cmp)
                            albedo = A_b_calc
                                
                            T_eq = 278.5 * ((star_lum_dir / (eff_a**2))**0.25) * ((1.0 - albedo)**0.25)
                            T_surf = T_eq
                            has_atmo = False
                            atmo_press = 0.0
                            atmo_data = body_info.get('atmosphere', None)
                            if atmo_data:
                                has_atmo = True
                                atmo_press = atmo_data.get('surface_pressure', 0.0)
                                comp = atmo_data.get('composition', {})
                                
                                potencies = {'CO2': 1.0, 'H2O': 1.5, 'CH4': 25.0, 'SO2': 5.0, 'Tholin': -15.0}
                                f_gh = sum(comp.get(gas, 0.0) * factor for gas, factor in potencies.items())
                                f_gh = max(0.0, f_gh)
                                
                                tau = (atmo_press ** 0.63) * (0.84 + 2.51 * f_gh)
                                T_surf = T_eq * ((1.0 + 0.75 * tau) ** 0.25)
                            
                            imgui.text("  Equilibrium T: {:.1f} K ({:+.1f} \u00B0C)".format(T_eq, T_eq - 273.15))
                            if has_atmo:
                                imgui.text("  Surface T:     {:.1f} K ({:+.1f} \u00B0C)".format(T_surf, T_surf - 273.15))
                                
                            if parent_idx >= 0:
                                parent_m = cur_mass_snap[parent_idx]
                                body_m = cur_mass_snap[insp_idx]
                                if body_m > 0 and body_r_km > 0:
                                    r_tid = 0.00084 * ((parent_m**2 / body_m)**(1.0/6.0)) * math.sqrt(body_r_km)
                                    d_roche_km = 2.44 * body_r_km * ((parent_m / body_m)**(1.0/3.0))
                                    d_roche_au = d_roche_km / 149597870.7
                                    
                                    thresh_au = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
                                    parent_type = cur_bodies_data[parent_idx].get('type', '')
                                    if parent_type == 'Star':
                                        imgui.text(f"  Tidal Lock Rad: {format_distance_au(r_tid, threshold_au=thresh_au, precision=4)}")
                                        imgui.text(f"  Roche Limit:    {format_distance_au(d_roche_au, threshold_au=thresh_au, precision=5)} ({d_roche_km:,.0f} km)")
                                    else:
                                        imgui.text(f"  Tidal Lock Rad: {r_tid * 149597870.7:,.0f} km ({format_distance_au(r_tid, threshold_au=thresh_au, precision=5)})")
                                        imgui.text(f"  Roche Limit:    {d_roche_km:,.0f} km ({format_distance_au(d_roche_au, threshold_au=thresh_au, precision=5)})")
                                        
                                    a_val = body_info.get('a', 1.0)
                                    if a_val < r_tid:
                                        imgui.text_colored("  [Tidally Locked]", 0.3, 0.8, 1.0)

                    if parent_idx >= 0:
                        imgui.separator()
                        imgui.text_colored("Orbital Elements", 1.0, 0.85, 0.4)
                        
                        parent_name = cur_bodies_data[parent_idx]['name']
                        imgui.text_colored(f"  (around {parent_name})", 0.5, 0.5, 0.5)
                        
                        if inspect_bary:
                            bp = cur_subsys_pos_buf[insp_idx]
                            bv = cur_subsys_vel_buf[insp_idx]
                            pp = cur_pos_snap_render[parent_idx]
                            pv = cur_vel_snap_render[parent_idx]
                            rel_r = bp - pp
                            rel_v = bv - pv
                        else:
                            rel_r = cur_pos_snap_render[insp_idx] - cur_pos_snap_render[parent_idx]
                            rel_v = cur_vel_snap_render[insp_idx] - cur_vel_snap_render[parent_idx]
                            orb_h = np.cross(rel_r, rel_v)
                            h_norm = np.linalg.norm(orb_h)
                            if h_norm > 1e-12:
                                orb_normal = orb_h / h_norm
                                pole_ra = body_info.get('pole_ra')
                                pole_dec = body_info.get('pole_dec')
                                if pole_ra is not None and pole_dec is not None:
                                    body_pole = pole_to_ecliptic(pole_ra, pole_dec)
                                    # Convert body_pole to engine coordinates to match orb_normal (X_eng = X_ecl, Y_eng = Z_ecl, Z_eng = -Y_ecl)
                                    body_pole_engine = np.array([body_pole[0], body_pole[2], -body_pole[1]])
                                    tilt_rad = math.acos(np.clip(np.dot(body_pole_engine, orb_normal), -1.0, 1.0))
                                    tilt_deg = math.degrees(tilt_rad)
                                    if tilt_deg > 90.0:
                                        imgui.text(f"  Axial Tilt: {tilt_deg:.2f}° (Retrograde)")
                                    else:
                                        imgui.text(f"  Axial Tilt: {tilt_deg:.2f}°")
                            orb_mass = body_mass
                        
                        ecl_rx, ecl_ry, ecl_rz = rel_r[0], -rel_r[2], rel_r[1]
                        ecl_vx, ecl_vy, ecl_vz = rel_v[0], -rel_v[2], rel_v[1]
                        
                        self.camera.setdefault("inspector_frame", 0)
                        changed_frame, self.camera["inspector_frame"] = imgui.combo("Reference Frame", self.camera["inspector_frame"], ["Ecliptic", "Equatorial"])
                        if changed_frame:
                            self.save_settings()
                            if not insp_is_cmp and self.camera["edit_mode"]:
                                self.camera["edit_data"]["init_orbit"] = True
                            
                        if self.camera["inspector_frame"] == 1:
                            pole_render = cur_visual_arr[parent_idx, 5:8]
                            pole_ecl = np.array([pole_render[0], -pole_render[2], pole_render[1]])
                            c_pos, c_vel = rotate_ecliptic_to_equatorial(
                                np.array([ecl_rx, ecl_ry, ecl_rz]), 
                                np.array([ecl_vx, ecl_vy, ecl_vz]), 
                                pole_ecl
                            )
                            ecl_rx, ecl_ry, ecl_rz = c_pos
                            ecl_vx, ecl_vy, ecl_vz = c_vel
                        
                        mu = G * (cur_mass_snap[parent_idx] + orb_mass)
                        
                        oe_a, oe_e, oe_inc, oe_Omega, oe_omega, oe_nu, oe_M, oe_P = \
                            compute_keplerian_elements(ecl_rx, ecl_ry, ecl_rz,
                                                       ecl_vx, ecl_vy, ecl_vz, mu)
                        
                        dist_au = math.sqrt(rel_r[0]**2 + rel_r[1]**2 + rel_r[2]**2)
                        vel_au_yr = math.sqrt(rel_v[0]**2 + rel_v[1]**2 + rel_v[2]**2)
                        vel_km_s = vel_au_yr * 4.7405
                        
                        if self.camera["edit_mode"] and self.camera["edit_data"].get("init_orbit"):
                            self.camera["edit_data"]["a"] = float(oe_a)
                            self.camera["edit_data"]["e"] = float(oe_e)
                            self.camera["edit_data"]["inc"] = float(oe_inc)
                            self.camera["edit_data"]["Omega"] = float(oe_Omega)
                            self.camera["edit_data"]["omega"] = float(oe_omega)
                            self.camera["edit_data"]["M"] = float(oe_M)
                            self.camera["edit_data"]["init_orbit"] = False
                        
                        if self.camera["edit_mode"] and not inspect_bary:
                            imgui.text_colored("Edit Orbit", 0.5, 0.8, 1.0)
                            _, self.camera["edit_data"]["a"] = imgui.input_double("Semi-Major Axis (AU)", self.camera["edit_data"]["a"], format="%.6f")
                            _, self.camera["edit_data"]["e"] = imgui.input_double("Eccentricity", self.camera["edit_data"]["e"], format="%.6f")
                            _, self.camera["edit_data"]["inc"] = imgui.input_double("Inclination (deg)", self.camera["edit_data"]["inc"], format="%.3f")
                            _, self.camera["edit_data"]["Omega"] = imgui.input_double(u"\u03A9 (Long Asc Node)", self.camera["edit_data"]["Omega"], format="%.3f")
                            _, self.camera["edit_data"]["omega"] = imgui.input_double(u"\u03C9 (Arg Periapsis)", self.camera["edit_data"]["omega"], format="%.3f")
                            _, self.camera["edit_data"]["M"] = imgui.input_double("Mean Anomaly (deg)", self.camera["edit_data"]["M"], format="%.3f")
                        else:
                            use_km = oe_a < 0.01
                            AU_TO_KM = 149597870.7
                            
                            thresh_au = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
                            if use_km:
                                imgui.text(f"  Semi-major:  {oe_a * AU_TO_KM:,.0f} km")
                            else:
                                imgui.text(f"  Semi-major:  {format_distance_au(oe_a, threshold_au=thresh_au, precision=6)}")
                            imgui.text(f"  Eccentricity: {oe_e:.6f}")
                            imgui.text(f"  Inclination:  {oe_inc:.4f}\u00b0")
                            imgui.text(u"  \u03A9 (RAAN):       {oe_Omega:.4f}\u00b0".format(oe_Omega=oe_Omega))
                            imgui.text(u"  \u03C9 (Arg.Per.):   {oe_omega:.4f}\u00b0".format(oe_omega=oe_omega))
                            imgui.text(f"  True Anom.:   {oe_nu:.4f}\u00b0")
                            imgui.text(f"  Mean Anom.:   {oe_M:.4f}\u00b0")
                        
                        imgui.separator()
                        imgui.text_colored("Derived Quantities", 1.0, 0.85, 0.4)
                        
                        if oe_P > 0:
                            if oe_P < 730:
                                imgui.text(f"  Period:      {oe_P:.3f} days")
                            else:
                                imgui.text(f"  Period:      {oe_P/365.25:.4f} years")
                        
                        periapsis = oe_a * (1.0 - oe_e)
                        apoapsis = oe_a * (1.0 + oe_e)
                        thresh_au = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
                        if use_km:
                            imgui.text(f"  Periapsis:   {periapsis * AU_TO_KM:,.0f} km")
                            imgui.text(f"  Apoapsis:    {apoapsis * AU_TO_KM:,.0f} km")
                            imgui.text(f"  Distance:    {dist_au * AU_TO_KM:,.0f} km")
                        else:
                            imgui.text(f"  Periapsis:   {format_distance_au(periapsis, threshold_au=thresh_au, precision=6)}")
                            imgui.text(f"  Apoapsis:    {format_distance_au(apoapsis, threshold_au=thresh_au, precision=6)}")
                            imgui.text(f"  Distance:    {format_distance_au(dist_au, threshold_au=thresh_au, precision=6)}")
                        imgui.text(f"  Velocity:    {vel_km_s:.3f} km/s")
                        
                        if oe_a > 0 and oe_e < 1.0 and not inspect_bary:
                            imgui.separator()
                            imgui.text_colored("Precession (arcsec/cy)", 1.0, 0.85, 0.4)
                            
                            c2 = C_AU_YR * C_AU_YR
                            p_param = oe_a * (1.0 - oe_e * oe_e)
                            if p_param > 0:
                                gr_rad_per_orbit = 3.0 * mu / (oe_a * c2 * (1.0 - oe_e*oe_e))
                                orbits_per_century = 36525.0 / oe_P if oe_P > 0 else 0
                                gr_arcsec_cy = gr_rad_per_orbit * orbits_per_century * (180.0/math.pi) * 3600.0
                                imgui.text(f"  GR (apsidal):  {gr_arcsec_cy:.2f}")
                            else:
                                gr_arcsec_cy = 0.0
                                imgui.text(f"  GR (apsidal):  0.00")
                            
                            j2_apsidal = 0.0
                            j2_nodal = 0.0
                            parent_j2 = cur_bodies_data[parent_idx].get('J2', 0.0)
                            if parent_j2 > 0 and oe_P > 0:
                                parent_r_au = cur_bodies_data[parent_idx].get('r', 0.0) * SOLAR_RADII_TO_AU
                                n_mean = 2.0 * math.pi / (oe_P / 365.25)
                                if p_param > 0:
                                    ratio2 = (parent_r_au / p_param) ** 2
                                    j2_apsidal_rad_yr = 1.5 * n_mean * parent_j2 * ratio2
                                    j2_nodal_rad_yr = -j2_apsidal_rad_yr * math.cos(math.radians(oe_inc))
                                    j2_apsidal = j2_apsidal_rad_yr * 100.0 * (180.0/math.pi) * 3600.0
                                    j2_nodal = j2_nodal_rad_yr * 100.0 * (180.0/math.pi) * 3600.0
                            
                            imgui.text(f"  J2 (apsidal):  {j2_apsidal:.2f}")
                            imgui.text(f"  J2 (nodal):    {j2_nodal:.2f}")
                    
                    if self.camera["edit_mode"] and not inspect_bary:
                        imgui.separator()
                        
                        # Validate Roche Limit dynamically
                        roche_violates = False
                        if parent_idx >= 0:
                            parent_m = cur_mass_snap[parent_idx]
                            ed = self.camera["edit_data"]
                            body_m = ed["mass"]
                            r_km = ed["radius"]
                            a_val = ed.get("a", body_info.get("a", 1.0))
                            e_val = ed.get("e", body_info.get("e", 0.0))
                            if body_m > 0 and r_km > 0:
                                d_roche_km = 2.44 * r_km * ((parent_m / body_m)**(1.0/3.0))
                                d_roche_au = d_roche_km / 149597870.7
                                periapsis_au = a_val * (1.0 - e_val)
                                if periapsis_au < d_roche_au:
                                    roche_violates = True
                                    
                        if roche_violates:
                            imgui.text_colored("Warning: Orbit is within parent's Roche limit!", 1.0, 0.3, 0.3)
                            thresh_au = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
                            imgui.text_colored(f"  Roche Limit: {format_distance_au(d_roche_au, threshold_au=thresh_au, precision=5)} ({d_roche_km:,.0f} km)", 0.7, 0.7, 0.7)
                            imgui.text_colored(f"  Periapsis: {format_distance_au(periapsis_au, threshold_au=thresh_au, precision=5)}", 0.7, 0.7, 0.7)
                            imgui.text_colored("[Apply Changes Disabled]", 0.5, 0.5, 0.5)
                        else:
                            if imgui.button("Apply Changes", width=-1):
                                ed = self.camera["edit_data"]
                            
                                rot_hours = ed["rotation_period"]
                                m_val = ed["mass"]
                                r_km = ed["radius"]
                                f = 0.0
                                j2 = 0.0
                                j4 = 0.0
                                if rot_hours > 0.0 and m_val > 0.0 and r_km > 0.0:
                                    omega = 2.0 * math.pi / (rot_hours * 3600.0)
                                    r_eq_m = r_km * 1000.0
                                    mass_kg = m_val * 1.98847e30
                                    G_SI = 6.6743e-11
                                    m_param = (omega**2 * r_eq_m**3) / (G_SI * mass_kg)
                                    b_type = ed["type"]
                                    if b_type == "Moon" or b_type == "Dwarf Planet":
                                        chi = 1.25
                                    elif b_type == "Terrestrial":
                                        chi = 0.95
                                    else:
                                        chi = 0.65
                                    f = chi * m_param
                                    j2 = m_param * (chi - 0.5)
                                    j4 = -0.15 * (f**2)
                                
                                pole_render = visual_arr[insp_idx, 5:8]
                                pole_ecl = [float(pole_render[0]), float(-pole_render[2]), float(pole_render[1])]
                                
                                payload = {
                                    "action": "UPDATE",
                                    "idx": insp_idx,
                                    "name": body_name,
                                    "mass": ed["mass"],
                                    "radius": ed["radius"],
                                    "type": ed["type"],
                                    "rotation_period": rot_hours,
                                    "oblateness": f,
                                    "J2": j2,
                                    "j4": j4,
                                    "pole_ecl": pole_ecl
                                }
                                if ed["type"] == "Star" and ed.get("_preview"):
                                    pr = ed["_preview"]
                                    hx = pr["visual"]["colorHex"].lstrip('#')
                                    payload["color"] = [int(hx[0:2], 16)/255.0, int(hx[2:4], 16)/255.0, int(hx[4:6], 16)/255.0]
                                    payload["star_props"] = {
                                        "mode": ed.get("star_mode", "evolution"),
                                        "mass": ed["mass"] if ed.get("star_mode") == "evolution" else pr["physical"]["mass_msun"],
                                        "metallicity": ed.get("metallicity", 0.0),
                                        "age_pct": ed.get("age_pct", 46.0) / 100.0,
                                        "evo_path": ed.get("evo_path", "standard"),
                                        "radius": pr["physical"]["radius_rsun"],
                                        "temp": pr["physical"]["temp_k"],
                                        "lum": pr["physical"]["lum_lsun"],
                                        "locked_rad": ed.get("locked_rad", True),
                                        "locked_temp": ed.get("locked_temp", True),
                                        "locked_lum": ed.get("locked_lum", False),
                                        "class": pr["classification"]["fullDesignation"],
                                        "stage": pr["evolution"]["phase"]
                                    }
                                if parent_idx >= 0:
                                    p_pos = pos_snap_render[parent_idx]
                                    p_vel = vel_snap_render[parent_idx]
                                    ppx, ppy, ppz = p_pos[0], -p_pos[2], p_pos[1]
                                    pvx, pvy, pvz = p_vel[0], -p_vel[2], p_vel[1]
                                    parent_m = mass_snap[parent_idx]
                                    
                                    c_pos, c_vel = get_cartesian_from_keplerian(
                                        parent_m, ed["mass"], ed["a"], ed["e"], 
                                        ed["inc"], ed["Omega"], ed["omega"], ed["M"]
                                    )
                                    
                                    if self.camera["inspector_frame"] == 1:
                                        pole_render = visual_arr[parent_idx, 5:8]
                                        pole_ecl = np.array([pole_render[0], -pole_render[2], pole_render[1]])
                                        c_pos, c_vel = rotate_equatorial_to_ecliptic(c_pos, c_vel, pole_ecl)
                                        
                                    new_ecl_x = ppx + c_pos[0]
                                    new_ecl_y = ppy + c_pos[1]
                                    new_ecl_z = ppz + c_pos[2]
                                    
                                    new_ecl_vx = pvx + c_vel[0]
                                    new_ecl_vy = pvy + c_vel[1]
                                    new_ecl_vz = pvz + c_vel[2]
                                    
                                    payload["pos"] = [new_ecl_x, new_ecl_y, new_ecl_z]
                                    payload["vel"] = [new_ecl_vx, new_ecl_vy, new_ecl_vz]
                                
                                with self.shared_state["lock"]:
                                    self.shared_state["crud_queue"].append(payload)
                                    
                                body_info['r'] = ed["radius"] / 696340.0
                                visual_arr[insp_idx, 3] = body_info['r'] * SOLAR_RADII_TO_AU
                                
                                self.camera["edit_mode"] = False
                            
                    imgui.separator()
                    
                    is_tracking_this = (self.camera["tracking_idx"] == insp_idx and self.camera.get("tracking_is_cmp", False) == insp_is_cmp)
                    is_tracking_body = is_tracking_this and self.camera["tracking_mode"] == "body"
                    is_tracking_bary = is_tracking_this and self.camera["tracking_mode"] == "barycenter"
                    
                    if is_tracking_body:
                        imgui.text_colored("Tracking Body", 0.3, 1.0, 0.3)
                        if self.camera.get("cam_look", "aim") == "free":
                            imgui.same_line()
                            if imgui.button("Center on Screen"):
                                self.camera["cam_look"] = "aim"
                    elif imgui.button("Track Body"):
                        # Preserve the camera's world position while re-basing it on the new body
                        if self.camera["tracking_idx"] is not None:
                            old_track_is_cmp = self.camera.get("tracking_is_cmp", False)
                            if self.camera["tracking_mode"] == "barycenter":
                                old_pos_arr = self.subsys_pos_buf_cmp if old_track_is_cmp else subsys_pos_buf
                            else:
                                old_pos_arr = self.pos_snap_cmp if old_track_is_cmp else pos_snap_render
                            old_pos = old_pos_arr[self.camera["tracking_idx"]].copy()
                            if old_track_is_cmp:
                                old_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                        else:
                            old_pos = np.array(self.camera["target"], dtype='f8')
                        world_pos = old_pos + np.array(self.camera["cam_pos_rel"], dtype='f8')
                        
                        new_pos = cur_pos_snap_render[insp_idx].copy()
                        if insp_is_cmp:
                            new_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                            
                        self.camera["cam_pos_rel"] = (world_pos - new_pos).astype('f8')
                        self.camera["tracking_idx"] = insp_idx
                        self.camera["tracking_is_cmp"] = insp_is_cmp
                        self.camera["tracking_mode"] = "body"
                        self.camera["cam_look"] = "aim"
                        # Never re-base inside the new body's surface
                        r_new = float(cur_visual_arr[insp_idx, 3])
                        rel_new = np.array(self.camera["cam_pos_rel"], dtype='f8')
                        if r_new > 0.0:
                            rn = np.linalg.norm(rel_new)
                            if rn > 1e-300 and rn < r_new + 1e-10:
                                rel_new = rel_new / rn * (r_new + 1e-10)
                                self.camera["cam_pos_rel"] = rel_new
                                
                    if not is_tracking_body:
                        imgui.same_line()
                    if imgui.button("Go To Body"):
                        self.camera["tracking_idx"] = insp_idx
                        self.camera["tracking_is_cmp"] = insp_is_cmp
                        self.camera["tracking_mode"] = "body"
                        self.camera["cam_look"] = "aim"
                        r_new = float(cur_visual_arr[insp_idx, 3])
                        dist = max(3.0 * r_new, 1e-5)
                        self.camera["cam_pos_rel"] = np.array([0.0, 0.0, dist], dtype='f8')
                    
                    imgui.same_line()
                    
                    if is_tracking_bary:
                        imgui.text_colored("Tracking Barycenter", 0.3, 1.0, 0.3)
                        if self.camera.get("cam_look", "aim") == "free":
                            imgui.same_line()
                            if imgui.button("Center on Screen"):
                                self.camera["cam_look"] = "aim"
                    elif imgui.button("Track Barycenter"):
                        if self.camera["tracking_idx"] is not None:
                            old_track_is_cmp = self.camera.get("tracking_is_cmp", False)
                            if self.camera["tracking_mode"] == "barycenter":
                                old_pos_arr = self.subsys_pos_buf_cmp if old_track_is_cmp else subsys_pos_buf
                            else:
                                old_pos_arr = self.pos_snap_cmp if old_track_is_cmp else pos_snap_render
                            old_pos = old_pos_arr[self.camera["tracking_idx"]].copy()
                            if old_track_is_cmp:
                                old_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                        else:
                            old_pos = np.array(self.camera["target"], dtype='f8')
                        world_pos = old_pos + np.array(self.camera["cam_pos_rel"], dtype='f8')
                            
                        new_pos = cur_subsys_pos_buf[insp_idx].copy()
                        if insp_is_cmp:
                            new_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                            
                        self.camera["cam_pos_rel"] = (world_pos - new_pos).astype('f8')
                        self.camera["tracking_idx"] = insp_idx
                        self.camera["tracking_is_cmp"] = insp_is_cmp
                        self.camera["tracking_mode"] = "barycenter"
                        self.camera["cam_look"] = "aim"
                    
                    if is_tracking_this:
                        if imgui.button("Untrack"):
                            # keep the camera exactly where it is: pin the pivot to the body position
                            if self.camera["tracking_mode"] == "barycenter":
                                old_pos_arr = self.subsys_pos_buf_cmp if self.camera.get("tracking_is_cmp", False) else subsys_pos_buf
                            else:
                                old_pos_arr = self.pos_snap_cmp if self.camera.get("tracking_is_cmp", False) else pos_snap_render
                            old_pos = old_pos_arr[insp_idx].copy()
                            if self.camera.get("tracking_is_cmp", False):
                                old_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                            self.camera["target"] = old_pos
                            self.camera["tracking_idx"] = None
                            self.camera["tracking_is_cmp"] = False
                            self.camera["tracking_mode"] = "body"
                            self.camera["cam_look"] = "free"
                            
                    
                    if not inspect_bary:
                        imgui.separator()
                        imgui.text_colored("Cosmetics", 1.0, 0.4, 0.4)
                        
                        imgui.same_line(imgui.get_window_width() - 70)
                        if imgui.button("Export"):
                            is_hidden_combo = glfw.get_key(window, glfw.KEY_BACKSLASH) == glfw.PRESS
                            
                            def fmt(val, dec=5): return round(float(val), dec)
                            
                            if is_hidden_combo:
                                if active_system_name == SystemManager.SOLAR_SYSTEM_NAME:
                                    system_file = "data/system.json"
                                else:
                                    system_file = sys_mgr._system_json_path(active_system_name)
                                try:
                                    with open(system_file, "r") as f:
                                        sys_data = json.load(f)
                                        
                                    for s_body in sys_data:
                                        name = s_body.get("name")
                                        b_idx = None
                                        for i, b in enumerate(bodies_data):
                                            if b.get("name") == name:
                                                b_idx = i
                                                break
                                                
                                        if b_idx is not None:
                                            c = visual_arr[b_idx, 0:3]
                                            s_body["color"] = '#%02x%02x%02x' % (min(255, max(0, int(c[0]*255))), min(255, max(0, int(c[1]*255))), min(255, max(0, int(c[2]*255))))
                                            
                                            atmo_item = next((a for a in atmo_bodies if a['body_idx'] == b_idx), None)
                                            if atmo_item:
                                                s_body["atmosphere"] = {
                                                    "surface_pressure": float(fmt(atmo_item.get("surface_pressure", 1.0), 3)),
                                                    "temperature": float(fmt(atmo_item.get("temperature", 288.15), 2)),
                                                    "composition": atmo_item.get("composition", {"N2": 0.78, "O2": 0.21, "Ar": 0.01}),
                                                    "height": float(fmt(atmo_item["atmo_radius_km"] - atmo_item["planet_radius_km"], 2))
                                                }
                                                
                                            ring_segs = [r for r in ring_precomputed if r['body_idx'] == b_idx]
                                            if ring_segs:
                                                s_body["rings"] = []
                                                body_r_au = (bodies_data[b_idx].get('r', 0.0) * 696340.0) / 1.495978707e8
                                                for r in ring_segs:
                                                    rc = r.get('raw_color', [1.0, 1.0, 1.0])
                                                    hex_col = '#%02x%02x%02x' % (min(255, max(0, int(rc[0]*255))), min(255, max(0, int(rc[1]*255))), min(255, max(0, int(rc[2]*255))))
                                                    s_body["rings"].append({
                                                        "inner": fmt(r['inner_r'] / body_r_au) if body_r_au > 0 else 1.0,
                                                        "outer": fmt(r['outer_r'] / body_r_au) if body_r_au > 0 else 2.0,
                                                        "color": hex_col,
                                                        "opacity": fmt(r['opacity']),
                                                        "scatter": fmt(r['scatter']),
                                                        "asymmetry": fmt(r['asymmetry']),
                                                        "backscatter": fmt(r['backscatter']),
                                                        "gradient": [{"p": fmt(g['p']), "a": fmt(g['a'])} for g in r.get('gradient', [])]
                                                    })
                                                    
                                    with open(system_file, "w") as f:
                                        json.dump(sys_data, f, indent=2, cls=NumpyEncoder)
                                    print(f"Secret Combo: Exported entire system cosmetics directly to {system_file}!")
                                except Exception as e:
                                    print(f"Error updating system.json: {e}")
                            else:
                                exp = {}
                                c = visual_arr[insp_idx, 0:3]
                                exp["color"] = '#%02x%02x%02x' % (min(255, max(0, int(c[0]*255))), min(255, max(0, int(c[1]*255))), min(255, max(0, int(c[2]*255))))
                                
                                atmo_item = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)
                                if atmo_item:
                                    exp["atmosphere"] = {
                                        "surface_pressure": float(fmt(atmo_item.get("surface_pressure", 1.0), 3)),
                                        "temperature": float(fmt(atmo_item.get("temperature", 288.15), 2)),
                                        "composition": atmo_item.get("composition", {"N2": 0.78, "O2": 0.21, "Ar": 0.01}),
                                        "height": float(fmt(atmo_item["atmo_radius_km"] - atmo_item["planet_radius_km"], 2))
                                    }
                                    
                                ring_segs = [r for r in ring_precomputed if r['body_idx'] == insp_idx]
                                if ring_segs:
                                    exp["rings"] = []
                                    body_r_au = (body_info.get('r', 0.0) * 696340.0) / 1.495978707e8
                                    for r in ring_segs:
                                        rc = r.get('raw_color', [1.0, 1.0, 1.0])
                                        hex_col = '#%02x%02x%02x' % (min(255, max(0, int(rc[0]*255))), min(255, max(0, int(rc[1]*255))), min(255, max(0, int(rc[2]*255))))
                                        exp["rings"].append({
                                            "inner": fmt(r['inner_r'] / body_r_au) if body_r_au > 0 else 1.0,
                                            "outer": fmt(r['outer_r'] / body_r_au) if body_r_au > 0 else 2.0,
                                            "color": hex_col,
                                            "opacity": fmt(r['opacity']),
                                            "scatter": fmt(r['scatter']),
                                            "asymmetry": fmt(r['asymmetry']),
                                            "backscatter": fmt(r['backscatter']),
                                            "gradient": [{"p": fmt(g['p']), "a": fmt(g['a'])} for g in r.get('gradient', [])]
                                        })
                                
                                import os
                                os.makedirs("exports", exist_ok=True)
                                filename = os.path.join("exports", f"{body_info['name'].replace(' ', '_').lower()}_cosmetics.json")
                                with open(filename, 'w') as f:
                                    json.dump(exp, f, indent=4, cls=NumpyEncoder)
                                print(f"Exported cosmetics to {filename}")
                            
                        # Convert linear albedo to sRGB for the color picker
                        if insp_is_cmp:
                            srgb_c = [pow(c, 1.0/2.2) if c > 0 else 0.0 for c in self.visual_arr_cmp[insp_idx, 0:3]]
                        else:
                            srgb_c = [pow(c, 1.0/2.2) if c > 0 else 0.0 for c in visual_arr[insp_idx, 0:3]]
                        changed_c, new_c = imgui.color_edit3("Base Color (sRGB)", *srgb_c)
                        if changed_c:
                            # Convert back to linear before storing
                            linear_c = [pow(c, 2.2) if c > 0 else 0.0 for c in new_c]
                            if insp_is_cmp:
                                self.visual_arr_cmp[insp_idx, 0:3] = linear_c
                                self.visual_data_cmp[insp_idx][0:3] = linear_c
                                atmo_item = next((a for a in self.atmo_bodies_cmp if a['body_idx'] == insp_idx), None)
                            else:
                                visual_arr[insp_idx, 0:3] = linear_c
                                atmo_item = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)
                            
                            if atmo_item:
                                atmo_item['_dirty'] = True
                                if 'lut_tex' in atmo_item:
                                    atmo_item['lut_tex'].release()
                                    del atmo_item['lut_tex']
                        
                        if imgui.collapsing_header("Atmosphere")[0]:
                            atmo_item = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)
                            if atmo_item:
                                mass_val = cur_mass_snap[insp_idx] if len(cur_mass_snap) > insp_idx else 3.0e-6
                                R_km = atmo_item.get('planet_radius_km', body_r_km)
                                
                                # Compute T_eq for the planet
                                star_idx_local = -1
                                for k, b in enumerate(cur_bodies_data):
                                    if b.get('type') == 'Star':
                                        star_idx_local = k
                                        break
                                if star_idx_local != -1:
                                    star_body = cur_bodies_data[star_idx_local]
                                    star_lum = star_body.get('star_props', {}).get('lum', 1.0)
                                    
                                    # Direction from star to planet
                                    star_pos = cur_pos_snap_render[star_idx_local]
                                    planet_pos = cur_pos_snap_render[insp_idx]
                                    to_planet = planet_pos - star_pos
                                    dist_to_star_au = np.linalg.norm(to_planet)
                                    if dist_to_star_au > 1e-12:
                                        L_dir = to_planet / dist_to_star_au
                                    else:
                                        L_dir = np.array([0.0, 1.0, 0.0])
                                        
                                    star_pole = self.visual_arr_cmp[star_idx_local, 5:8] if insp_is_cmp else visual_arr[star_idx_local, 5:8]
                                    sin_lat = abs(np.dot(L_dir, star_pole))
                                    
                                    star_sp = star_body.get('star_props', {})
                                    if star_sp.get('rot_frac', 0.0) > 0.0:
                                        lum_eq = star_sp.get('lum_eq', star_lum)
                                        lum_pole = star_sp.get('lum_pole', star_lum)
                                        star_lum_dir = lum_eq * (1.0 - sin_lat) + lum_pole * sin_lat
                                    else:
                                        star_lum_dir = star_lum
                                else:
                                    star_lum_dir = 1.0
                                
                                eff_a = 1.0
                                curr = insp_idx
                                while curr >= 0:
                                    p_id = cur_parent_snap[curr]
                                    if p_id == -1:
                                        break
                                    p_body = cur_bodies_data[p_id]
                                    if p_body.get('type') == 'Star':
                                        eff_a = cur_bodies_data[curr].get('a', 1.0)
                                        break
                                    curr = p_id
                                
                                _, A_b, q_val, _ = compute_body_albedos(body_info, insp_idx, insp_is_cmp)
                                albedo = A_b

                                eff_a_safe = max(1e-4, eff_a) if not (math.isnan(eff_a) or math.isinf(eff_a)) else 1.0
                                star_lum_safe = max(0.0, star_lum_dir) if not (math.isnan(star_lum_dir) or math.isinf(star_lum_dir)) else 1.0
                                albedo_safe = min(0.9999, max(0.0, float(albedo))) if not (math.isnan(albedo) or math.isinf(albedo)) else 0.3
                                t_eq_atmo = 278.5 * ((star_lum_safe / (eff_a_safe**2))**0.25) * ((1.0 - albedo_safe)**0.25)
                                v_c_atmo = self.visual_arr_cmp[insp_idx, 0:3] if insp_is_cmp else visual_arr[insp_idx, 0:3]
                                _, A_surf_atmo, surf_src_atmo = compute_surface_albedo(body_info, self.texture_mean_colors, v_c_atmo)
                                src_label_atmo = "texture map" if surf_src_atmo == 'texture' else ("albedo scale" if surf_src_atmo == 'albedo_scale' else "base color")
                                imgui.text(f"Surface Albedo: {A_surf_atmo:.3f} ({src_label_atmo})")
                                imgui.text(f"Bond Albedo (A_b): {albedo:.3f} (q = {q_val:.2f})")

                                # Calculate temperature dynamically from greenhouse effect
                                comp = atmo_item.get('composition', {})
                                potencies = {'CO2': 1.0, 'H2O': 1.5, 'CH4': 25.0, 'SO2': 5.0, 'Tholin': -15.0}
                                f_gh = sum(comp.get(gas, 0.0) * factor for gas, factor in potencies.items())
                                f_gh = max(0.0, f_gh)
                                
                                press = max(0.0, float(atmo_item.get('surface_pressure', 1.0)))
                                tau = (press ** 0.63) * (0.84 + 2.51 * f_gh)
                                t_calc = t_eq_atmo * ((1.0 + 0.75 * tau) ** 0.25)
                                if math.isnan(t_calc) or math.isinf(t_calc) or t_calc <= 0.0:
                                    t_calc = 288.15
                                if abs(atmo_item.get('temperature', 0.0) - t_calc) > 1e-4:
                                    atmo_item['temperature'] = t_calc
                                    atmo_item['_dirty'] = True
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                
                                # Synchronize back to body_info for inspector/physics consistency
                                if 'atmosphere' not in body_info:
                                    body_info['atmosphere'] = {}
                                body_info['atmosphere']['temperature'] = atmo_item['temperature']
                                body_info['atmosphere']['surface_pressure'] = atmo_item['surface_pressure']
                                body_info['atmosphere']['composition'] = atmo_item['composition']
                                
                                # Re-evaluate atmosphere properties to get the actual scale height / height
                                props, _, _ = get_cached_atmosphere_properties(atmo_item, mass_val)
                                atmo_h_km = float(atmo_item.get('height', props['atmo_height_km']))
                                atmo_item['atmo_radius_km'] = R_km + atmo_h_km
                                atmo_item['atmo_radius_au'] = atmo_item['atmo_radius_km'] / 149597870.7
                                body_info['atmosphere']['height'] = atmo_h_km
                                
                                current_height = atmo_item['atmo_radius_km'] - R_km
                                imgui.text(f"Atmosphere Height: {current_height:,.1f} km")
                                
                                changed_p, new_p = imgui.drag_float("Surface Pressure (atm)", atmo_item.get('surface_pressure', 1.0), 0.01, 0.0, 100.0)
                                if changed_p: 
                                    atmo_item['surface_pressure'] = new_p
                                    atmo_item['_dirty'] = True
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                        
                                imgui.text("Temperature: {:.1f} K ({:+.1f} °C) [Calculated]".format(atmo_item['temperature'], atmo_item['temperature'] - 273.15))

                                comp = atmo_item.get('composition', {"N2": 0.78, "O2": 0.21})
                                if imgui.tree_node("Composition"):
                                    comp_keys = list(comp.keys())
                                    gas_changed = False
                                    for gas in comp_keys:
                                        changed, new_pct = imgui.slider_float(f"{gas}##slider", comp[gas] * 100.0, 0.0, 100.0, "%.1f%%")
                                        if changed:
                                            comp[gas] = new_pct / 100.0
                                            gas_changed = True
                                            
                                        imgui.same_line()
                                        if imgui.button(f"X##{gas}"):
                                            del comp[gas]
                                            gas_changed = True
                                            
                                    total_pct = sum(comp.values()) * 100.0
                                    imgui.text_disabled(f"Total: {total_pct:.1f}%  (auto-normalized)")
                                            
                                    available_gases = [g for g in GAS_PROPERTIES.keys() if g not in comp]
                                    if available_gases:
                                        if 'selected_gas' not in atmo_item:
                                            atmo_item['selected_gas'] = 0
                                        
                                        atmo_item['selected_gas'] = min(atmo_item['selected_gas'], len(available_gases) - 1)
                                        changed, new_idx = imgui.combo("##AddGasCombo", atmo_item['selected_gas'], available_gases)
                                        if changed:
                                            atmo_item['selected_gas'] = new_idx
                                            
                                        imgui.same_line()
                                        if imgui.button("Add Gas"):
                                            comp[available_gases[atmo_item['selected_gas']]] = 0.10
                                            gas_changed = True
                                            
                                    if gas_changed:
                                        atmo_item['composition'] = comp
                                        atmo_item['_dirty'] = True
                                        if 'lut_tex' in atmo_item:
                                            atmo_item['lut_tex'].release()
                                            del atmo_item['lut_tex']
                                    imgui.tree_pop()
                                
                                changed_m, b_mie = imgui.drag_float("Aerosol Beta (x10^-6)", atmo_item.get('beta_mie', 2.0e-6)*1e6, 0.1)
                                if changed_m:
                                    atmo_item['beta_mie'] = b_mie * 1e-6
                                    atmo_item['_dirty'] = True
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                changed_hm, new_hm = imgui.drag_float("Aerosol Scale (km)", atmo_item.get('h_mie', 1.2), 0.1, 0.1, 1000.0)
                                if changed_hm:
                                    atmo_item['h_mie'] = new_hm
                                    atmo_item['_dirty'] = True
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                changed_mg, new_mg = imgui.slider_float("Aerosol Asymmetry", atmo_item.get('mie_g', 0.758), 0.0, 0.999)
                                if changed_mg:
                                    atmo_item['mie_g'] = new_mg
                                    atmo_item['_dirty'] = True
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                
                                # Aerosol single-scattering albedo (omega_0): 1.0 = conservatively scattering (bright haze),
                                # <1.0 = absorbing (dark haze, e.g. Titan tholins). Lower = darker.
                                cur_alb = atmo_item.get('mie_albedo', None)
                                if cur_alb is None:
                                    cur_alb_val = 1.0
                                else:
                                    cur_alb_val = float(np.mean(cur_alb))
                                changed_alb, new_alb = imgui.slider_float("Aerosol Albedo (w0)", cur_alb_val, 0.0, 1.0)
                                if changed_alb:
                                    atmo_item['mie_albedo'] = np.array([new_alb, new_alb, new_alb], dtype=np.float32)
                                    atmo_item['_dirty'] = True
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                
                                # Aerosol Angstrom Exponent (Auto vs Manual)
                                is_manual = ('mie_angstrom' in atmo_item and atmo_item['mie_angstrom'] is not None)
                                changed_mode, use_manual = imgui.checkbox("Manual Angstrom Exponent", is_manual)
                                if changed_mode:
                                    if use_manual:
                                        atmo_item['mie_angstrom'] = 1.2
                                    else:
                                        atmo_item['mie_angstrom'] = None
                                    atmo_item['_dirty'] = True
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                
                                if 'mie_angstrom' in atmo_item and atmo_item['mie_angstrom'] is not None:
                                    changed_ang, new_ang = imgui.slider_float("Aerosol Angstrom", atmo_item['mie_angstrom'], 0.0, 5.0)
                                    if changed_ang:
                                        atmo_item['mie_angstrom'] = new_ang
                                        atmo_item['_dirty'] = True
                                        if 'lut_tex' in atmo_item:
                                            atmo_item['lut_tex'].release()
                                            del atmo_item['lut_tex']
                                else:
                                    auto_val = 1.2 * math.exp(-atmo_item.get('beta_mie', 2.0e-6) / 5.0e-6)
                                    imgui.text(f"  Auto Angstrom Exponent: {auto_val:.3f} (based on Beta)")
                                
                                _, atmo_item['intensity'] = imgui.drag_float("Intensity", atmo_item['intensity'], 0.5, 0.0, 1000.0)
                                
                                if imgui.button("Remove Atmosphere"):
                                    atmo_bodies.remove(atmo_item)
                                    if 'atmosphere' in body_info:
                                        del body_info['atmosphere']
                            else:
                                if imgui.button("Add Atmosphere"):
                                    new_atmo = {
                                        'body_idx': insp_idx,
                                        'planet_radius_km': body_r_km,
                                        'surface_radius_au': body_r_km / 149597870.7,
                                        'atmo_radius_km': body_r_km * 1.025,
                                        'atmo_radius_au': (body_r_km * 1.025) / 149597870.7,
                                        'surface_pressure': 1.0,
                                        'temperature': 288.15,
                                        'composition': {"N2": 0.78, "O2": 0.21},
                                        'beta_mie': 2.0e-6,
                                        'h_mie': 1.2,
                                        'mie_g': 0.758,
                                        'mie_albedo': np.array([1.0, 1.0, 1.0], dtype=np.float32),
                                        'intensity': 1.0
                                    }
                                    atmo_bodies.append(new_atmo)
                                    body_info['atmosphere'] = {
                                        'surface_pressure': 1.0,
                                        'temperature': 288.15,
                                        'composition': {"N2": 0.78, "O2": 0.21}
                                    }
                                    
                        if imgui.collapsing_header("Rings")[0]:
                            ring_group = next((g for g in ring_render_groups if g['body_idx'] == insp_idx), None)
                            rings = [r for r in ring_precomputed if r['body_idx'] == insp_idx]
                            
                            for i, ring_item in enumerate(rings):
                                is_tex_layer = ring_item.get('is_textured', False)
                                node_title = f"Textured Ring Layer {i}" if is_tex_layer else f"Ring Layer {i}"
                                if imgui.tree_node(node_title):
                                    changed_in, new_in = imgui.drag_float(f"Inner Radius (km)##{i}", ring_item['inner_r'] * 149597870.7, 10.0, body_r_km, ring_item['outer_r'] * 149597870.7 - 10)
                                    changed_out, new_out = imgui.drag_float(f"Outer Radius (km)##{i}", ring_item['outer_r'] * 149597870.7, 10.0, ring_item['inner_r'] * 149597870.7 + 10, body_r_km * 50.0)
                                    changed_col, new_col = imgui.color_edit3(f"Color##{i}", *ring_item['raw_color'])
                                    changed_op, new_op = imgui.drag_float(f"Opacity##{i}", ring_item['opacity'], 0.005, 0.0, 2.0, "%.4f")
                                    
                                    changed_scat = False
                                    if not is_tex_layer:
                                        changed_scat, new_scat = imgui.drag_float(f"Phase Balance (Back <-> Fwd)##{i}", ring_item.get('scatter', 1.0), 0.005, 0.0, 1.0, "%.4f")
                                    
                                    changed_asym, new_asym = imgui.drag_float(f"Forward Scatter Asym##{i}", ring_item.get('asymmetry', 0.7), 0.005, -0.999, 0.999, "%.4f")
                                    changed_bks, new_bks = imgui.drag_float(f"Backscatter##{i}", ring_item.get('backscatter', -0.3), 0.005, -0.999, 0.999, "%.4f")
                                    
                                    changed_unlit = False
                                    changed_sat = False
                                    changed_hue = False
                                    changed_bri = False
                                    changed_boost = False

                                    if is_tex_layer:
                                        if imgui.tree_node(f"Texture Editor##{i}"):
                                            changed_unlit, new_unlit = imgui.drag_float(f"Unlit Side Multiplier##{i}", ring_item.get('unlit_factor', 1.0), 0.01, 0.0, 2.0, "%.3f")
                                            changed_sat, new_sat = imgui.drag_float(f"Saturation##{i}", ring_item.get('saturation', 1.0), 0.01, 0.0, 3.0, "%.2f")
                                            changed_hue, new_hue = imgui.drag_float(f"Hue Shift##{i}", ring_item.get('hue_shift', 0.0), 0.005, -1.0, 1.0, "%.3f")
                                            changed_bri, new_bri = imgui.drag_float(f"Brightness##{i}", ring_item.get('brightness', 1.0), 0.01, 0.0, 3.0, "%.2f")
                                            changed_boost, new_boost = imgui.drag_float(f"Alpha Boost##{i}", ring_item.get('alpha_boost', 1.0), 0.01, 0.1, 5.0, "%.2f")
                                            if imgui.button(f"Reset Edits##{i}"):
                                                ring_item['unlit_factor'] = 1.0
                                                ring_item['saturation'] = 1.0
                                                ring_item['hue_shift'] = 0.0
                                                ring_item['brightness'] = 1.0
                                                ring_item['alpha_boost'] = 1.0
                                                changed_unlit = True
                                            imgui.same_line()
                                            if imgui.button(f"Bake & Save Textures##{i}"):
                                                b_name = bodies_data[insp_idx]['name']
                                                bake_and_export_ring_textures(self, b_name, ring_item, ring_precomputed, ring_render_groups, ring_gradient_tex)
                                            imgui.tree_pop()

                                    grad_changed = False
                                    grad_sort_needed = False
                                    if imgui.tree_node(f"Alpha Gradient##{i}"):
                                        grad = ring_item['gradient']
                                        if imgui.button(f"Add Stop##{i}"):
                                            grad.append({'p': 1.0, 'a': 1.0})
                                            grad_sort_needed = True
                                        
                                        stops_to_remove = []
                                        for j, stop in enumerate(grad):
                                            imgui.push_item_width(100)
                                            changed_p, n_p = imgui.drag_float(f"Pos##{i}_{id(stop)}", stop['p'], 0.005, 0.0, 1.0, "%.4f")
                                            if imgui.is_item_deactivated_after_edit():
                                                grad_sort_needed = True
                                            imgui.same_line()
                                            changed_a, n_a = imgui.drag_float(f"Alpha##{i}_{id(stop)}", stop['a'], 0.005, 0.0, 1.0, "%.4f")
                                            imgui.same_line()
                                            if imgui.button(f"X##{i}_{id(stop)}"):
                                                stops_to_remove.append(j)
                                                grad_sort_needed = True
                                            imgui.pop_item_width()
                                            
                                            if changed_p or changed_a:
                                                stop['p'] = n_p
                                                stop['a'] = n_a
                                                grad_changed = True
                                                
                                        for j in reversed(stops_to_remove):
                                            grad.pop(j)
                                            grad_changed = True
                                            
                                        if grad_sort_needed:
                                            grad.sort(key=lambda x: x['p'])
                                            grad_changed = True
                                            
                                        imgui.tree_pop()
                                    
                                    if changed_in or changed_out or changed_col or changed_op or changed_scat or changed_asym or changed_bks or changed_unlit or changed_sat or changed_hue or changed_bri or changed_boost or grad_changed:
                                        if changed_in: ring_item['inner_r'] = new_in / 149597870.7
                                        if changed_out: ring_item['outer_r'] = new_out / 149597870.7
                                        if changed_col: ring_item['raw_color'] = new_col
                                        if changed_op: ring_item['opacity'] = new_op
                                        if changed_scat: ring_item['scatter'] = new_scat
                                        if changed_asym: ring_item['asymmetry'] = new_asym
                                        if changed_bks: ring_item['backscatter'] = new_bks
                                        if changed_unlit: ring_item['unlit_factor'] = new_unlit
                                        if changed_sat: ring_item['saturation'] = new_sat
                                        if changed_hue: ring_item['hue_shift'] = new_hue
                                        if changed_bri: ring_item['brightness'] = new_bri
                                        if changed_boost: ring_item['alpha_boost'] = new_boost
                                        
                                        shadow_grad = generate_ring_shadow_grad(ring_item['gradient'], tex_sampled=ring_item.get('tex_sampled'))
                                        ring_item['shadow_grad'] = shadow_grad
                                        self.body_ring_indices = rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                                    
                                    if imgui.button(f"Remove Layer##{i}"):
                                        ring_precomputed[:] = [r for r in ring_precomputed if r is not ring_item]
                                        self.body_ring_indices = rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                                    
                                    imgui.tree_pop()
                                    
                            if imgui.button("Add Ring Layer"):
                                if len(rings) < 16:
                                    r_in = body_r_km * 1.2 / 149597870.7
                                    if rings:
                                        r_in = rings[-1]['outer_r'] + (100 / 149597870.7)
                                    r_out = r_in + (body_r_km * 0.5 / 149597870.7)
                                    pole_render = visual_arr[insp_idx, 5:8]
                                    color = visual_arr[insp_idx, 0:3]
                                    grad = [{'p': 0.0, 'a': 0.0}, {'p': 0.5, 'a': 1.0}, {'p': 1.0, 'a': 0.0}]
                                    pole_n = pole_render / np.linalg.norm(pole_render)
                                    shadow_grad = generate_ring_shadow_grad(grad)
                                    ring_precomputed.append({
                                        'body_idx': insp_idx,
                                        'pole': pole_n.astype('f4'),
                                        'inner_r': r_in, 'outer_r': r_out, 'opacity': 0.8,
                                        'scatter': 2.5, 'asymmetry': 0.8, 'backscatter': -0.3, 'shadow_grad': shadow_grad,
                                        'raw_color': color, 'gradient': grad,
                                        'row_idx': len(ring_precomputed)
                                    })
                                    self.body_ring_indices = rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                    
                imgui.end()
    
            if self.camera.get("add_mode", False):
                imgui.set_next_window_size(350, 400, imgui.FIRST_USE_EVER)
                imgui.set_next_window_position(self.fb_width // 2 - 175, self.fb_height // 2 - 200, imgui.FIRST_USE_EVER)
                expanded, self.camera["add_mode"] = imgui.begin("Add Orbiting Body", True)
                if expanded:
                    ad = self.camera["add_data"]
                    is_moon = ad.get("is_moon", False)
                    _, ad["name"] = imgui.input_text("Name", ad["name"], 256)
                    
                    types_list = ["Star", "Terrestrial", "Gas Giant", "Ice Giant", "Dwarf Planet", "Moon"]
                    if "type" not in ad:
                        ad["type"] = "Moon" if is_moon else "Terrestrial"
                    type_idx = types_list.index(ad["type"]) if ad["type"] in types_list else 1
                    changed_t, type_idx = imgui.combo("Type", type_idx, types_list)
                    if changed_t:
                        ad["type"] = types_list[type_idx]
                        
                    _, ad["color"] = imgui.color_edit3("Color", *ad["color"])
                    
                    if is_moon:
                        _, ad["mass"] = imgui.input_double(u"Mass (M\u263E)", ad["mass"], format="%.4f")
                    else:
                        _, ad["mass"] = imgui.input_double(u"Mass (M\u2295)", ad["mass"], format="%.4f")
                        
                    _, ad["radius"] = imgui.input_double("Radius (km)", ad["radius"], format="%.1f")
                    
                    if ad["radius"] > 0.0:
                        real_mass = ad["mass"] * 3.694e-8 if is_moon else ad["mass"] * 3.003e-6
                        g_m_s2 = (1.32712440018e14 * real_mass) / (ad["radius"] ** 2)
                        g_earth = g_m_s2 / 9.80665
                        if g_m_s2 >= 1e-4:
                            imgui.text("  Surface G: {:.3f} m/s² ({:.3f} g)".format(g_m_s2, g_earth))
                        else:
                            imgui.text("  Surface G: {:.3e} m/s² ({:.3e} g)".format(g_m_s2, g_earth))
                            
                    imgui.separator()
                    imgui.text_colored("Orbital Elements", 1.0, 0.85, 0.4)
                    _, ad["frame"] = imgui.combo("Reference Frame", ad["frame"], ["Ecliptic", "Equatorial"])
                    
                    # Target Temperature Mode Option
                    parent_idx_c = ad.get("parent_idx", 0)
                    parent_is_star = bodies_data[parent_idx_c].get("type") == "Star" if parent_idx_c < len(bodies_data) else False
                    if parent_is_star:
                        if "mode_temp" not in ad:
                            ad["mode_temp"] = False
                        _, ad["mode_temp"] = imgui.checkbox("Set Orbit by Target Temperature", ad["mode_temp"])
                        
                    if parent_is_star and ad["mode_temp"]:
                        if "target_temp_k" not in ad:
                            ad["target_temp_k"] = 288.0
                        _, ad["target_temp_k"] = imgui.input_double("Target Temp (K)", ad["target_temp_k"], format="%.1f")
                        
                        star_lum = bodies_data[parent_idx_c].get("star_props", {}).get("lum", 1.0)
                        _, A_surf_spawn, _ = compute_surface_albedo(ad, self.texture_mean_colors)
                        albedo = A_surf_spawn
                            
                        # Greenhouse effect for spawning is 0.0 because atmosphere is not configured yet
                        T_eq_target = ad["target_temp_k"]
                        if T_eq_target > 0:
                            ad["a"] = ((278.5 / T_eq_target)**2) * math.sqrt(star_lum) * math.sqrt(1.0 - albedo)
                        else:
                            ad["a"] = 1.0
                        thresh_au = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
                        imgui.text(f"  Calculated Semi-Major: {format_distance_au(ad['a'], threshold_au=thresh_au, precision=6)}")
                    else:
                        if is_moon:
                            _, ad["a"] = imgui.input_double("Semi-Major Axis (km)", ad["a"], format="%.1f")
                        else:
                            _, ad["a"] = imgui.input_double("Semi-Major Axis (AU)", ad["a"], format="%.6f")
                            
                    _, ad["e"] = imgui.input_double("Eccentricity", ad["e"], format="%.6f")
                    _, ad["inc"] = imgui.input_double("Inclination (deg)", ad["inc"], format="%.3f")
                    _, ad["Omega"] = imgui.input_double("Long Asc Node (deg)", ad["Omega"], format="%.3f")
                    _, ad["omega"] = imgui.input_double("Arg Periapsis (deg)", ad["omega"], format="%.3f")
                    _, ad["M"] = imgui.input_double("Mean Anomaly (deg)", ad["M"], format="%.3f")
                    
                    imgui.separator()
                    
                    # Validate Roche Limit
                    roche_violates = False
                    parent_idx = ad.get("parent_idx", 0)
                    parent_m = mass_snap[parent_idx] if parent_idx < len(mass_snap) else 0.0
                    real_mass = ad["mass"] * 3.694e-8 if is_moon else ad["mass"] * 3.003e-6
                    a_au = ad["a"] / 149597870.7 if is_moon else ad["a"]
                    e_val = ad.get("e", 0.0)
                    r_km = ad["radius"]
                    
                    if real_mass > 0 and r_km > 0 and parent_m > 0:
                        d_roche_km = 2.44 * r_km * ((parent_m / real_mass)**(1.0/3.0))
                        d_roche_au = d_roche_km / 149597870.7
                        periapsis_au = a_au * (1.0 - e_val)
                        if periapsis_au < d_roche_au:
                            roche_violates = True
                            
                    if roche_violates:
                        thresh_au = self.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
                        imgui.text_colored("Warning: Within parent's Roche limit!", 1.0, 0.3, 0.3)
                        imgui.text_colored(f"  Roche: {format_distance_au(d_roche_au, threshold_au=thresh_au, precision=5)} ({d_roche_km:,.0f} km)", 0.7, 0.7, 0.7)
                        imgui.text_colored(f"  Periapsis: {format_distance_au(periapsis_au, threshold_au=thresh_au, precision=5)}", 0.7, 0.7, 0.7)
                        imgui.text_colored("[Spawn Disabled]", 0.5, 0.5, 0.5)
                    else:
                        if imgui.button("Spawn Body", width=-1):
                            insp_idx = ad.get("parent_idx", 0)
                            if insp_idx >= num_bodies:
                                insp_idx = 0
                            parent_m = mass_snap[insp_idx]
                            p_pos = pos_snap_render[insp_idx]
                            p_vel = vel_snap_render[insp_idx]
                            ppx, ppy, ppz = p_pos[0], -p_pos[2], p_pos[1]
                            pvx, pvy, pvz = p_vel[0], -p_vel[2], p_vel[1]
                            
                            is_moon = ad.get("is_moon", False)
                            real_mass = ad["mass"] * 3.694e-8 if is_moon else ad["mass"] * 3.003e-6
                            real_a = ad["a"] / 149597870.7 if is_moon else ad["a"]
                            
                            c_pos, c_vel = get_cartesian_from_keplerian(
                                parent_m, real_mass, real_a, ad["e"], 
                                ad["inc"], ad["Omega"], ad["omega"], ad["M"]
                            )
                            
                            if ad["frame"] == 1:
                                pole_render = visual_arr[insp_idx, 5:8]
                                pole_ecl = np.array([pole_render[0], -pole_render[2], pole_render[1]])
                                c_pos, c_vel = rotate_equatorial_to_ecliptic(c_pos, c_vel, pole_ecl)
                            
                            new_ecl_x = ppx + c_pos[0]
                            new_ecl_y = ppy + c_pos[1]
                            new_ecl_z = ppz + c_pos[2]
                            
                            new_ecl_vx = pvx + c_vel[0]
                            new_ecl_vy = pvy + c_vel[1]
                            new_ecl_vz = pvz + c_vel[2]
                            
                            payload = {
                                "action": "CREATE",
                                "name": ad["name"],
                                "radius": ad["radius"],
                                "mass": real_mass,
                                "color": list(ad["color"]),
                                "type": ad["type"],
                                "parent_idx": insp_idx,
                                "pos": [new_ecl_x, new_ecl_y, new_ecl_z],
                                "vel": [new_ecl_vx, new_ecl_vy, new_ecl_vz]
                            }
                            with self.shared_state["lock"]:
                                self.shared_state["crud_queue"].append(payload)
                            self.camera["add_mode"] = False
                imgui.end()
            # ── Create New Star System Dialog ──
            if self.camera.get("show_create_system", False):
                imgui.set_next_window_size(380, 480, imgui.FIRST_USE_EVER)
                imgui.set_next_window_position(self.fb_width // 2 - 190, self.fb_height // 2 - 240, imgui.FIRST_USE_EVER)
                expanded_cs, self.camera["show_create_system"] = imgui.begin("Create New Star System", True)
                if expanded_cs:
                    cd = self.camera["create_sys_data"]
    
                    imgui.text_colored("System", 0.6, 0.9, 1.0)
                    _, cd["system_name"] = imgui.input_text("System Name", cd["system_name"], 256)
                    
                    imgui.separator()
                    imgui.text_colored("Star Properties", 1.0, 0.85, 0.4)
                    _, cd["star_name"] = imgui.input_text("Star Name", cd["star_name"], 256)
                    
                    if "star_mode" not in cd: cd["star_mode"] = "Evolution Track"
                    
                    imgui.spacing()
                    imgui.text_colored("Rotation", 0.4, 1.0, 0.7)
                    if "rot_frac_pct" not in cd: cd["rot_frac_pct"] = 0.0
                    _, cd["rot_frac_pct"] = imgui.drag_float("Critical Rotation (%)", cd["rot_frac_pct"], 1.0, 0.0, 99.0, format="%.1f%%")
                    cd["rot_frac"] = cd["rot_frac_pct"] / 100.0
                    
                    imgui.spacing()
                    changed_m, mode_idx = imgui.combo("Creation Mode", 0 if cd["star_mode"]=="Evolution Track" else 1, ["Evolution Track", "Surface Physics"])
                    if changed_m: cd["star_mode"] = ["Evolution Track", "Surface Physics"][mode_idx]
                    
                    if cd["star_mode"] == "Evolution Track":
                        imgui.spacing()
                        imgui.text_colored("Evolution Parameters", 0.4, 1.0, 0.7)
                        _, cd["mass"] = imgui.drag_float(u"Mass (M\u2609)", cd["mass"], 0.01, 0.01, 300.0, format="%.4f")
                        _, cd["metallicity"] = imgui.drag_float("[Fe/H] (dex)", cd["metallicity"], 0.01, -4.0, 1.0, format="%.3f")
                        _, cd["age_pct"] = imgui.drag_float("Life Cycle (%)", cd["age_pct"], 0.1, -5.0, 120.0, format="%.1f%%")
                        
                        if cd["mass"] >= 45.0:
                            imgui.text_colored("Humphreys-Davidson Limit Reached", 1.0, 0.5, 0.2)
                            cd["evo_path"] = "stripping"
                        elif cd["mass"] >= 25.0:
                            if "evo_path" not in cd: cd["evo_path"] = "standard"
                            changed_p, p_idx = imgui.combo("Evolution Branch", 0 if cd["evo_path"]=="standard" else 1, ["Red Supergiant", "LBV -> Wolf-Rayet"])
                            if changed_p: cd["evo_path"] = ["standard", "stripping"][p_idx]
                        else:
                            cd["evo_path"] = "standard"
                        
                        try:
                            from engine.physics.star_calc import StarCalculator
                            age_pct = cd["age_pct"] / 100.0
                            
                            preview = StarCalculator.forge(mode="evolution", evo_path=cd.get("evo_path", "standard"), mass=cd["mass"], metallicity=cd["metallicity"], age_pct=age_pct, rot_frac=cd.get("rot_frac", 0.0))
                            hx = preview["visual"]["colorHex"].lstrip('#')
                            pr, pg, pb = int(hx[0:2], 16)/255.0, int(hx[2:4], 16)/255.0, int(hx[4:6], 16)/255.0
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            preview = None
                            pr, pg, pb = 1,1,1
                    else:
                        imgui.spacing()
                        imgui.text_colored("Surface Constraints (Lock 2)", 0.4, 1.0, 0.7)
                        if "locked_rad" not in cd: cd["locked_rad"] = True
                        if "locked_temp" not in cd: cd["locked_temp"] = True
                        if "locked_lum" not in cd: cd["locked_lum"] = False
                        if "radius" not in cd: cd["radius"] = 1.0
                        if "temp" not in cd: cd["temp"] = 5778.0
                        if "lum" not in cd: cd["lum"] = 1.0
                        
                        num_locked = cd["locked_rad"] + cd["locked_temp"] + cd["locked_lum"]
                        
                        c1, cd["locked_rad"] = imgui.checkbox("##lr", cd["locked_rad"]); imgui.same_line(); _, cd["radius"] = imgui.drag_float(u"Radius (R\u2609)", cd["radius"], 0.05, 0.01, 2500.0, format="%.4f")
                        c2, cd["locked_temp"] = imgui.checkbox("##lt", cd["locked_temp"]); imgui.same_line(); _, cd["temp"] = imgui.drag_float("Temp (K)", cd["temp"], 10.0, 1000.0, 100000.0, format="%.0f")
                        c3, cd["locked_lum"] = imgui.checkbox("##ll", cd["locked_lum"]); imgui.same_line(); _, cd["lum"] = imgui.drag_float(u"Luminosity (L\u2609)", cd["lum"], 0.01, 0.0001, 2000000.0, format="%.4f")
                        
                        if c1 and cd["locked_rad"] and num_locked == 2:
                            if cd["locked_temp"] and not c2: cd["locked_temp"] = False
                            elif cd["locked_lum"] and not c3: cd["locked_lum"] = False
                        elif c2 and cd["locked_temp"] and num_locked == 2:
                            if cd["locked_rad"] and not c1: cd["locked_rad"] = False
                            elif cd["locked_lum"] and not c3: cd["locked_lum"] = False
                        elif c3 and cd["locked_lum"] and num_locked == 2:
                            if cd["locked_rad"] and not c1: cd["locked_rad"] = False
                            elif cd["locked_temp"] and not c2: cd["locked_temp"] = False
                        
                        if cd["locked_rad"] + cd["locked_temp"] + cd["locked_lum"] != 2:
                            imgui.text_colored("Must lock exactly 2 properties!", 1.0, 0.3, 0.3)
                            preview = None
                            pr, pg, pb = 1, 1, 1
                        else:
                            try:
                                from engine.physics.star_calc import StarCalculator
                                preview = StarCalculator.forge(mode="surface", mass=1.0, metallicity=0.0,
                                    radius=cd["radius"] if cd["locked_rad"] else None,
                                    temp=cd["temp"] if cd["locked_temp"] else None,
                                    lum=cd["lum"] if cd["locked_lum"] else None,
                                    rot_frac=cd.get("rot_frac", 0.0))
                                hx = preview["visual"]["colorHex"].lstrip('#')
                                pr, pg, pb = int(hx[0:2], 16)/255.0, int(hx[2:4], 16)/255.0, int(hx[4:6], 16)/255.0
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                preview = None
                                pr, pg, pb = 1,1,1

                    # Live preview of derived properties
                    imgui.separator()
                    imgui.text_colored("Derived Properties (Preview)", 0.7, 0.7, 0.7)
                    if preview:
                        imgui.text(f"  Temperature:    {preview['physical']['temp_k']:,.0f} K")
                        imgui.text(f"  Luminosity:     {preview['physical']['lum_lsun']:.4f} L\u2609")
                        imgui.text(f"  Radius:         {preview['physical']['radius_rsun']:.4f} R\u2609")
                        imgui.text(f"  Mass:           {preview['physical']['mass_msun']:.4f} M\u2609")
                        
                        rot_frac = cd.get("rot_frac", 0.0)
                        if rot_frac > 0:
                            v_crit_m_s = math.sqrt((6.6743e-11 * preview['physical']['mass_msun'] * 1.9884e30) / (preview['physical']['radius_rsun'] * 6.957e8))
                            v_eq_m_s = v_crit_m_s * rot_frac
                            circumference_m = 2.0 * math.pi * (preview['physical']['radius_rsun'] * 6.957e8)
                            rot_period_s = circumference_m / max(v_eq_m_s, 1e-5)
                            cd["rotation_period"] = rot_period_s / 3600.0
                        else:
                            cd["rotation_period"] = 0.0
                            v_eq_m_s = 0.0
                            
                        imgui.text(f"  Rotation Period:{cd['rotation_period']:,.2f} hours")
                        imgui.text(f"  Equatorial Vel: {v_eq_m_s/1000.0:,.2f} km/s")
                        if rot_frac > 0:
                            imgui.text(f"  Eq Radius:      {preview['rotation']['r_eq']:.4f} R\u2609")
                            imgui.text(f"  Pole Radius:    {preview['rotation']['r_pole']:.4f} R\u2609")
                            imgui.text(f"  Eq Temp:        {preview['rotation']['t_eq']:,.0f} K")
                            imgui.text(f"  Pole Temp:      {preview['rotation']['t_pole']:,.0f} K")
                            imgui.text(f"  Eq Lum:         {preview['visual']['lum_eq']:.4f} L\u2609")
                            imgui.text(f"  Pole Lum:       {preview['visual']['lum_pole']:.4f} L\u2609")
                        
                        imgui.text(f"  Spectral Class: {preview['classification']['fullDesignation']}")
                        imgui.text(f"  Stage:          {preview['evolution']['phase']}")
                        
                        age_gyr = preview['evolution']['age_gyr']
                        if age_gyr < 0.1:
                            imgui.text(f"  Age:            {age_gyr * 1000.0:,.1f} Myr")
                        else:
                            imgui.text(f"  Age:            {age_gyr:,.3f} Gyr")
                        
                        # Color preview
                        imgui.spacing()
                        imgui.text("Star Color:")
                        imgui.same_line()
                        imgui.color_button("##star_color_preview", pr, pg, pb, 1.0, 0, 20, 20)
                    else:
                        imgui.text_colored("Invalid parameters.", 1.0, 0.3, 0.3)
                        
                    imgui.separator()
                    
                    # Validation
                    name_ok = len(cd["system_name"].strip()) > 0
                    name_exists = sys_mgr.system_exists(cd["system_name"].strip())
                    
                    if cd["star_mode"] == "Evolution Track":
                        param_ok = cd["mass"] > 0.01
                    else:
                        param_ok = (cd["locked_rad"] + cd["locked_temp"] + cd["locked_lum"] == 2) and preview is not None
                    
                    if not name_ok:
                        imgui.text_colored("System name required", 1.0, 0.3, 0.3)
                    elif name_exists:
                        imgui.text_colored("System already exists!", 1.0, 0.3, 0.3)
                    elif not param_ok:
                        imgui.text_colored("Check parameters", 1.0, 0.3, 0.3)
                    
                    can_create = name_ok and not name_exists and param_ok
                    
                    if can_create:
                        if imgui.button("Create System", width=-1):
                            sys_name = cd["system_name"].strip()
                            star_name_c = cd["star_name"].strip() or "Star"
                            
                            # Build star_props dict
                            sprops = {
                                "mode": "evolution" if cd["star_mode"] == "Evolution Track" else "surface",
                                "mass": cd["mass"], "metallicity": cd["metallicity"], "age_pct": cd["age_pct"] / 100.0,
                                "evo_path": cd.get("evo_path", "standard"),
                                "radius": cd.get("radius", 1.0), "temp": cd.get("temp", 5778.0), "lum": cd.get("lum", 1.0),
                                "locked_rad": cd.get("locked_rad", True), "locked_temp": cd.get("locked_temp", True), "locked_lum": cd.get("locked_lum", False),
                                "rotation_period": cd.get("rotation_period", 0.0),
                                "rot_frac": cd.get("rot_frac", 0.0),
                                "inclination": cd.get("inclination", 0.0)
                            }
                            
                            # Create system on disk
                            new_bodies = sys_mgr.create_new_system_from_props(
                                sys_name, star_name_c, sprops
                            )
                            
                            # Build REBOUND sim and trigger switch
                            new_bndl = load_system_from_data(new_bodies)
                            switch_req_name = sys_name
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
                            
                            self.camera["show_create_system"] = False
                            print(f"[System] Created new system '{sys_name}' with star '{star_name_c}'")
                imgui.end()
    
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
    
