import OpenGL
OpenGL.ERROR_CHECKING = False
import glfw
import moderngl
import numpy as np
import ctypes

import json
import math
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
import os as _os
from system_manager import SystemManager, SystemSnapshot, derive_star_properties, temperature_to_rgb, rgb_to_hex
from spice_manager import SpiceManager

from constants import *
from math_utils import *
from physics_core import *
from physics_core import _extract_render_state, _update_hierarchy_core
from render_utils import *
from shaders import *
_PERF_ENABLED = _os.environ.get("STELLAR_FORGE_PERF") == "1"
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

import OpenGL.GL as gl
import ctypes

class ModernGLImGuiRenderer(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self.io = imgui.get_io()
        self.textures = {}
        
        self.prog = ctx.program(
            vertex_shader="""
                #version 330 core
                uniform mat4 ProjMtx;
                in vec2 Position;
                in vec2 UV;
                in vec4 Color;
                out vec2 Frag_UV;
                out vec4 Frag_Color;
                void main() {
                    Frag_UV = UV;
                    Frag_Color = Color;
                    gl_Position = ProjMtx * vec4(Position.xy, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330 core
                uniform sampler2D Texture;
                in vec2 Frag_UV;
                in vec4 Frag_Color;
                out vec4 Out_Color;
                void main() {
                    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);
                }
            """
        )
        self.proj_mtx_uniform = self.prog['ProjMtx']
        self.texture_uniform = self.prog['Texture']
        self.texture_uniform.value = 0
        
        self.vbo = ctx.buffer(reserve=1024 * 1024)
        self.ibo = ctx.buffer(reserve=1024 * 1024)
        
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f 4f1', 'Position', 'UV', 'Color')],
            index_buffer=self.ibo
        )
        
        self.font_texture = None
        self.refresh_font_texture()
        
    def refresh_font_texture(self):
        width, height, pixels = self.io.fonts.get_tex_data_as_rgba32()
        if self.font_texture:
            if self.font_texture.glo in self.textures:
                del self.textures[self.font_texture.glo]
            self.font_texture.release()
        self.font_texture = self.ctx.texture((width, height), 4, data=pixels)
        self.font_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.io.fonts.texture_id = self.font_texture.glo
        self.textures[self.font_texture.glo] = self.font_texture
        self.io.fonts.clear_tex_data()
        
    def render(self, draw_data):
        io = self.io
        display_width, display_height = io.display_size
        fb_width = int(display_width * io.display_fb_scale[0])
        fb_height = int(display_height * io.display_fb_scale[1])
        if fb_width == 0 or fb_height == 0:
            return
            
        draw_data.scale_clip_rects(*io.display_fb_scale)
        
        ortho_projection = np.array([
             [ 2.0/display_width,  0.0,                   0.0, 0.0],
             [ 0.0,                2.0/-display_height,   0.0, 0.0],
             [ 0.0,                0.0,                  -1.0, 0.0],
             [-1.0,                1.0,                   0.0, 1.0]
        ], dtype='f4')
        self.proj_mtx_uniform.write(ortho_projection.tobytes())
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.depth_mask = False
        
        self.ctx.viewport = (0, 0, fb_width, fb_height)
        
        for commands in draw_data.commands_lists:
            vtx_bytes_size = commands.vtx_buffer_size * imgui.VERTEX_SIZE
            idx_bytes_size = commands.idx_buffer_size * imgui.INDEX_SIZE
            
            vtx_data = ctypes.string_at(commands.vtx_buffer_data, vtx_bytes_size)
            idx_data = ctypes.string_at(commands.idx_buffer_data, idx_bytes_size)
            
            if self.vbo.size < vtx_bytes_size:
                self.vbo.orphan(vtx_bytes_size)
            self.vbo.write(vtx_data)
            
            if self.ibo.size < idx_bytes_size:
                self.ibo.orphan(idx_bytes_size)
            self.ibo.write(idx_data)
            
            idx_buffer_offset = 0
            for command in commands.commands:
                texture = self.textures.get(command.texture_id)
                if texture:
                    texture.use(location=0)
                else:
                    gl.glBindTexture(gl.GL_TEXTURE_2D, command.texture_id)
                
                x, y, z, w = command.clip_rect
                self.ctx.scissor = (int(x), int(fb_height - w), int(z - x), int(w - y))
                
                self.vao.render(moderngl.TRIANGLES, vertices=command.elem_count, first=idx_buffer_offset // imgui.INDEX_SIZE)
                idx_buffer_offset += command.elem_count * imgui.INDEX_SIZE
                
        self.ctx.scissor = None
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)
        
    def shutdown(self):
        if self.vao:
            self.vao.release()
        if self.vbo:
            self.vbo.release()
        if self.ibo:
            self.ibo.release()
        if self.prog:
            self.prog.release()
        if self.font_texture:
            self.font_texture.release()

class ModernGLGlfwRenderer(GlfwRenderer):
    def __init__(self, window, ctx, attach_callbacks=True):
        self.modern_renderer = ModernGLImGuiRenderer(ctx)
        super().__init__(window, attach_callbacks)
        # Base class init calls refresh_font_texture which registers our ModernGL font texture.
        # But _invalidate_device_objects resets io.fonts.texture_id to 0. We must restore it.
        font_id = self.modern_renderer.font_texture.glo if self.modern_renderer.font_texture else 0
        self._invalidate_device_objects()
        self.io.fonts.texture_id = font_id
        
    def refresh_font_texture(self):
        self.modern_renderer.refresh_font_texture()
        self._font_texture = 0
        
    def render(self, draw_data):
        self.modern_renderer.render(draw_data)
        
    def shutdown(self):
        self.modern_renderer.shutdown()

@njit
def compute_planetshine_numba(pos, radii, colors, is_star, star_positions, star_colors, hdr_enabled):
    N = len(pos)
    planetshine_dirs = np.zeros((N, 3), dtype=np.float32)
    planetshine_colors = np.zeros((N, 3), dtype=np.float32)
    
    num_stars = len(star_positions)
    if num_stars == 0:
        return planetshine_dirs, planetshine_colors
        
    for i in range(N):
        if is_star[i] > 0.5:
            continue
            
        pos_i = pos[i]
        
        best_j = -1
        best_val = -1.0
        best_dir = np.zeros(3, dtype=np.float32)
        best_color = np.zeros(3, dtype=np.float32)
        
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
                
            dist = np.sqrt(dist_sq)
            dir_to_caster_x = dx / dist
            dir_to_caster_y = dy / dist
            dir_to_caster_z = dz / dist
            
            total_caster_light_r = 0.0
            total_caster_light_g = 0.0
            total_caster_light_b = 0.0
            
            for s in range(num_stars):
                star_pos = star_positions[s]
                star_col = star_colors[s]
                
                cx = star_pos[0] - pos_j[0]
                cy = star_pos[1] - pos_j[1]
                cz = star_pos[2] - pos_j[2]
                c_dist = np.sqrt(cx*cx + cy*cy + cz*cz)
                if c_dist > 1e-6:
                    cx /= c_dist
                    cy /= c_dist
                    cz /= c_dist
                else:
                    cx = 0.0
                    cy = 0.0
                    cz = 0.0
                    
                phase = cx * (-dir_to_caster_x) + cy * (-dir_to_caster_y) + cz * (-dir_to_caster_z)
                if phase < 0.0:
                    phase = 0.0
                    
                if hdr_enabled:
                    irradiance = star_col[3] / max(c_dist * c_dist, 1e-8)
                else:
                    irradiance = 1.0
                    
                total_caster_light_r += star_col[0] * phase * irradiance
                total_caster_light_g += star_col[1] * phase * irradiance
                total_caster_light_b += star_col[2] * phase * irradiance
                
            bounce_r = colors[j, 0] * total_caster_light_r * solid_angle * 2.0
            bounce_g = colors[j, 1] * total_caster_light_g * solid_angle * 2.0
            bounce_b = colors[j, 2] * total_caster_light_b * solid_angle * 2.0
            
            val = bounce_r*bounce_r + bounce_g*bounce_g + bounce_b*bounce_b
            if val > best_val:
                best_val = val
                best_j = j
                best_dir[0] = dir_to_caster_x
                best_dir[1] = dir_to_caster_y
                best_dir[2] = dir_to_caster_z
                best_color[0] = bounce_r
                best_color[1] = bounce_g
                best_color[2] = bounce_b
                
        if best_j != -1 and best_val > 1e-12:
            planetshine_dirs[i] = best_dir
            planetshine_colors[i] = best_color
            
    return planetshine_dirs, planetshine_colors

class App:
    def __init__(self):
        self.window_width, self.window_height = 1280, 720
        self.fb_width, self.fb_height = 1280, 720
        self.impl = None
        self.pick_request = None
        self.camera = {
            "target": np.array([0.0, 0.0, 0.0], dtype='f8'),
            "target_offset": np.array([0.0, 0.0, 0.0], dtype='f8'),
            "pan_offset": np.array([0.0, 0.0, 0.0], dtype='f8'),
            "exposure": 1.0,
            "hdr_enabled": True,
            "distance": 45.0,     
            "distance_actual": 45.0,
            "fov": 45.0,
            "yaw": -90.0,         
            "yaw_actual": -90.0,
            "pitch": 25.0,        
            "pitch_actual": 25.0,
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
            "show_settings_modal": False,
            "planetshine_enabled": True,
            "ringshine_enabled": True,
        }
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

    def scroll_callback(self, window, xoffset, yoffset):
        if self.impl: self.impl.scroll_callback(window, xoffset, yoffset)
        if imgui.get_io().want_capture_mouse: return 
        
        is_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        
        if is_shift:
            fov_speed = 2.0
            if yoffset > 0: self.camera["fov"] -= fov_speed
            elif yoffset < 0: self.camera["fov"] += fov_speed
            self.camera["fov"] = max(1.0, min(120.0, self.camera["fov"]))
        else:
            zoom_speed = self.camera["distance"] * 0.2 
            if yoffset > 0: self.camera["distance"] -= zoom_speed
            elif yoffset < 0: self.camera["distance"] += zoom_speed
            self.camera["distance"] = max(1e-7, self.camera["distance"])
    
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
        
        if self.camera["left_dragging"]:
            sensitivity = 0.3
            self.camera["yaw"] += dx * sensitivity
            self.camera["pitch"] -= dy * sensitivity
            self.camera["pitch"] = max(-89.9, min(89.9, self.camera["pitch"])) 
            
        elif self.camera["right_dragging"]:
            pan_speed = self.camera["distance_actual"] * 0.001
            yaw_rad = math.radians(self.camera["yaw_actual"])
            pitch_rad = math.radians(self.camera["pitch_actual"])
            
            front = np.array([
                math.cos(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                math.sin(yaw_rad) * math.cos(pitch_rad)
            ], dtype='f4')
            front /= np.linalg.norm(front)
            
            right = np.cross(front, np.array([0.0, 1.0, 0.0], dtype='f4'))
            right /= np.linalg.norm(right)
            
            up = np.cross(right, front)
            up /= np.linalg.norm(up)
            
            pan_vec = -right * dx * pan_speed + up * dy * pan_speed
            if self.camera["tracking_idx"] is not None:
                self.camera["pan_offset"] += pan_vec
            else:
                self.camera["target"] += pan_vec
    
        self.camera["last_x"], self.camera["last_y"] = xpos, ypos
    
    def char_callback(self, window, char):
        if self.impl: self.impl.char_callback(window, char)
    
    def key_callback(self, window, key, scancode, action, mods):
        if self.impl: self.impl.keyboard_callback(window, key, scancode, action, mods)
        if imgui.get_io().want_capture_keyboard: return
    
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_SPACE and action == glfw.PRESS:
                self.time_ctrl["paused"] = not self.time_ctrl["paused"]
            elif key in (glfw.KEY_UP, glfw.KEY_RIGHT):
                self.time_ctrl["multiplier"] = min(self.time_ctrl["multiplier"] * 2.0, 1e9)
            elif key in (glfw.KEY_DOWN, glfw.KEY_LEFT):
                self.time_ctrl["multiplier"] = max(self.time_ctrl["multiplier"] / 2.0, 1.0)
            elif key == glfw.KEY_R:
                self.time_ctrl["multiplier"] = 1.0
            elif key == glfw.KEY_MINUS:
                multiplier = 10.0 if (mods & glfw.MOD_SHIFT) else 2.0
                self.camera["exposure"] /= multiplier
            elif key == glfw.KEY_EQUAL:
                multiplier = 10.0 if (mods & glfw.MOD_SHIFT) else 2.0
                self.camera["exposure"] *= multiplier
    
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
        prog_orbit_compute = ctx.compute_shader(orbit_compute_shader)
    
        prog_ephem_orbits = ctx.program(vertex_shader=ephem_orbit_vertex_shader, fragment_shader=ephem_orbit_fragment_shader)
    
        prog_rings = ctx.program(vertex_shader=ring_vertex_shader, fragment_shader=ring_fragment_shader)
    
        prog_atmo = ctx.program(vertex_shader=atmo_vertex_shader, fragment_shader=atmo_fragment_shader)
    

        def build_eclipse_lut(ctx):
            LUT_SIZE = 256
            x_arr = np.linspace(0.0, 2.0, LUT_SIZE, dtype='f4')
            y_arr = np.linspace(0.0, 1.0, LUT_SIZE, dtype='f4')
            x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='xy')
            x_safe = np.maximum(x_grid, 1e-9)
            d1 = (1.0 - y_grid**2 + x_grid**2) / (2.0 * x_safe)
            d2 = x_grid - d1
            theta1 = 2.0 * np.arccos(np.clip(d1, -1.0, 1.0))
            theta2 = 2.0 * np.arccos(np.clip(d2 / np.maximum(y_grid, 1e-9), -1.0, 1.0))
            area = 0.5 * ( (theta1 - np.sin(theta1)) + y_grid**2 * (theta2 - np.sin(theta2)) )
            area = np.where(x_grid >= 1.0 + y_grid, 0.0, area)
            area = np.where(x_grid <= 1.0 - y_grid, math.pi * y_grid**2, area)
            area_frac = area / math.pi
            lut_data = area_frac.astype('f4').tobytes()
            lut_tex = ctx.texture((LUT_SIZE, LUT_SIZE), 1, lut_data, dtype='f4')
            lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            lut_tex.repeat_x = False
            lut_tex.repeat_y = False
            return lut_tex
            
        eclipse_lut_tex = build_eclipse_lut(ctx)
        eclipse_lut_tex.use(location=2)
        
        if 'u_eclipse_lut' in prog_spheres: prog_spheres['u_eclipse_lut'].value = 2
        if 'u_eclipse_lut' in prog_rings: prog_rings['u_eclipse_lut'].value = 2
        if 'u_eclipse_lut' in prog_atmo: prog_atmo['u_eclipse_lut'].value = 2

        prog_atmo_lut = ctx.program(vertex_shader=atmo_lut_vertex_shader, fragment_shader=atmo_lut_fragment_shader)
        lut_vbo = ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'))
        lut_vao = ctx.vertex_array(prog_atmo_lut, [(lut_vbo, '2f', 'in_position')])
        
        def build_atmo_lut(atmo):
            lut_tex = ctx.texture((256, 256), 2, dtype='f4')
            fbo = ctx.framebuffer(color_attachments=[lut_tex])
            prog_atmo_lut['u_planet_radius_km'].value = float(atmo['planet_radius_km'])
            prog_atmo_lut['u_atmo_radius_km'].value = float(atmo['atmo_radius_km'])
            prog_atmo_lut['u_h_rayleigh'].value = float(atmo['h_rayleigh'])
            prog_atmo_lut['u_h_mie'].value = float(atmo['h_mie'])
            fbo.use()
            lut_vao.render(moderngl.TRIANGLE_STRIP)
            ctx.screen.use()
            lut_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            lut_tex.repeat_x = False
            lut_tex.repeat_y = False
            atmo['lut_tex'] = lut_tex
            fbo.release()

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
            
            for ring_seg in rings_data:
                inner_r = ring_seg['inner'] * body_radius_au
                outer_r = ring_seg['outer'] * body_radius_au
                r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
                r_opacity = ring_seg.get('opacity', 1.0)
                r_scatter = ring_seg.get('scatter', 0.0)
                r_asymmetry = ring_seg.get('asymmetry', 0.7)
                
                RADIAL_SUBDIVISIONS = 32
                sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
                
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
                
                ring_precomputed.append({
                    'body_idx': body_idx,
                    'verts': np.array(verts, dtype='f4'),
                    'indices': np.array(indices, dtype='i4'),
                    'normal': pole_n.astype('f4'),
                    'colors': np.array(colors, dtype='f4'),
                    'inner_r': inner_r,
                    'outer_r': outer_r,
                    'opacity': r_opacity,
                    'scatter': r_scatter,
                    'asymmetry': r_asymmetry,
                    'shadow_grad': shadow_grad,
                    'raw_color': r_color,
                    'gradient': sorted_gradient,
                })
    
        ring_idx_by_body = {r['body_idx']: i for i, r in enumerate(ring_precomputed)}
        ring_gradient_data = np.zeros((16, 256), dtype='f4')
        for j, ring in enumerate(ring_precomputed):
            if j >= 16: break
            ring_gradient_data[j, :] = ring['shadow_grad']
        
        ring_gradient_tex = ctx.texture((256, 16), 1, ring_gradient_data.tobytes(), dtype='f4')
        ring_gradient_tex.filter = (moderngl.LINEAR, moderngl.NEAREST)
        ring_gradient_tex.repeat_x = False
        ring_gradient_tex.repeat_y = False
    
        ring_render_groups = []
        rings_by_body_init = {}
        for ring in ring_precomputed:
            bi = ring['body_idx']
            if bi not in rings_by_body_init:
                rings_by_body_init[bi] = []
            rings_by_body_init[bi].append(ring)
        
        for bi, rings in rings_by_body_init.items():
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
    
        INSTANCE_FLOATS = 24
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
        all_instances_buffer = ctx.buffer(reserve=MAX_BODIES * 96) # 24 floats * 4 bytes
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
    
        max_orbits = MAX_BODIES * 2
        orbit_ssbo = ctx.buffer(reserve=max_orbits * 160)
        orbit_ssbo.bind_to_storage_buffer(binding=0)
        orbit_ssbo_out = ctx.buffer(reserve=max_orbits * 4000 * 32)
        orbit_ssbo_out.bind_to_storage_buffer(binding=1)
        vao_gpu_orbits = ctx.vertex_array(prog_gpu_orbits, [])
    
        UBO_SIZE = 4768
        scene_ubo = ctx.buffer(reserve=UBO_SIZE)
        scene_ubo.bind_to_uniform_block(1)
        ubo_staging = np.zeros(UBO_SIZE // 4, dtype=np.float32)
        ubo_casters_int_view = ubo_staging[166:167].view(np.int32)
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
        uniform_num_ring_planes = prog_spheres['u_num_ring_planes']
        if 'u_ring_gradients' in prog_spheres:
            prog_spheres['u_ring_gradients'].value = 0
    
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
        u_ring_host_color = prog_rings.get('u_host_planet_color', None)
        u_ring_camera_pos = prog_rings['u_camera_pos']
        u_ring_body_offset = prog_rings['u_body_offset']
        u_ring_clip_mode = prog_rings['u_clip_mode']
        u_ring_caster_mask_lo_uni = prog_rings['u_caster_mask_lo']
        u_ring_caster_mask_hi_uni = prog_rings['u_caster_mask_hi']
        u_ring_planetshine_enabled = prog_rings.get('u_planetshine_enabled', None)

        u_atmo_body_offset = prog_atmo['u_body_offset']
        if 'u_ring_gradients' in prog_atmo:
            prog_atmo['u_ring_gradients'].value = 0
        if 'u_optical_depth_lut' in prog_atmo:
            prog_atmo['u_optical_depth_lut'].value = 1
        u_atmo_planet_radius = prog_atmo['u_planet_radius_km']
        u_atmo_atmo_radius = prog_atmo['u_atmo_radius_km']
        u_atmo_radius_au_uniform = prog_atmo['u_atmo_radius_au']
        u_atmo_au_to_km = prog_atmo['u_au_to_km']
        u_atmo_star_angular_radius = prog_atmo['u_star_angular_radius']
        u_atmo_beta_rayleigh = prog_atmo['u_beta_rayleigh']
        u_atmo_h_rayleigh = prog_atmo['u_h_rayleigh']
        u_atmo_beta_mie = prog_atmo['u_beta_mie']
        u_atmo_h_mie = prog_atmo['u_h_mie']
        u_atmo_mie_g = prog_atmo['u_mie_g']
        u_atmo_beta_absorption = prog_atmo['u_beta_absorption']
        u_atmo_quality_uniform = prog_atmo['u_atmo_quality']
        u_atmo_sun_intensity = prog_atmo['u_sun_intensity']
        u_atmo_camera_pos = prog_atmo['u_camera_pos']
        u_atmo_num_samples = prog_atmo['u_num_samples']
        u_atmo_num_light_samples = prog_atmo.get('u_num_light_samples', None)
        u_atmo_pole_obl = prog_atmo['u_pole_obl']
        u_atmo_body_idx_uni = prog_atmo['u_body_idx']
        u_atmo_num_ring_planes = prog_atmo['u_num_ring_planes']
        u_atmo_ring_centers = prog_atmo['u_ring_center']
        u_atmo_ring_normals = prog_atmo['u_ring_normal']
        u_atmo_ring_params = prog_atmo['u_ring_params']
        u_atmo_num_active_casters = prog_atmo['u_num_active_casters']
        u_atmo_active_casters = prog_atmo['u_active_casters']
        u_atmo_active_caster_poles_obl = prog_atmo['u_active_caster_poles_obl']
        u_atmo_active_caster_atmos = prog_atmo['u_active_caster_atmos']
        u_atmo_clip_mode = prog_atmo.get('u_atmo_clip_mode', None)


    
        G = 4.0 * math.pi**2 
    
        visual_arr = np.array(visual_data, dtype='f4')
        is_star_arr = np.zeros(num_bodies, dtype='f4')
        for i, b in enumerate(bodies_data):
            if b.get('type') == 'Star':
                is_star_arr[i] = 1.0
        non_star_mask = is_star_arr == 0.0
        non_star_indices = np.where(non_star_mask)[0][:64]
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
        ring_centers_buf = np.zeros((16, 3), dtype='f4')
        ring_normals_buf = np.zeros((16, 3), dtype='f4')
        ring_params_buf = np.zeros((16, 3), dtype='f4')
        ring_colors_buf = np.zeros((16, 3), dtype='f4')
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
    
        last_render_time = time.time()
        show_orbits = True
        orbit_fade_dir = 1.0
        orbit_fade_dir_idx = 0
        orbit_min_alpha = 0.3
        atmo_quality = 1
        now_dt = datetime.datetime.now()
        jump_date = [now_dt.year, now_dt.month, now_dt.day, now_dt.hour, now_dt.minute]
        scrub_index = [0]
    
        last_orbit_pos_snap = None
        last_orbit_pos_snap_cmp = None
        last_cam_origin = None
        
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
    
        print("[Render Loop] Entering main render loop")
        frame_counter = 0
        while not glfw.window_should_close(window):
            frame_counter += 1
            # ── System Switch — render-side rebuild ──
            with self.shared_state["lock"]:
                switch_complete = self.shared_state.get("system_switch_complete", False)
            if switch_complete:
                with self.shared_state["lock"]:
                    new_info = self.shared_state["system_new_bundle"]
                    saved_snap = self.shared_state["system_snapshot_out"]
                    is_ephem_enter = self.shared_state.get("ephemeris_enter", False)
                    is_ephem_exit = self.shared_state.get("ephemeris_exit", False)
                    self.shared_state["system_switch_complete"] = False
                    self.shared_state["system_new_bundle"] = None
                    self.shared_state["system_snapshot_out"] = None
                    self.shared_state["ephemeris_enter"] = False
                    self.shared_state["ephemeris_exit"] = False
    
                # Store snapshot of old system in SystemManager
                old_name = active_system_name
                if saved_snap is not None:
                    sys_mgr.store_snapshot(old_name, saved_snap)
    
                # Update active system name
                if is_ephem_enter:
                    ephemeris_mode_active = True
                    active_system_name = "Ephemeris Mode"
                    self.shared_state["ephemeris_mode"] = True
                elif is_ephem_exit:
                    ephemeris_mode_active = False
                    active_system_name = switch_req_name
                    self.shared_state["ephemeris_mode"] = False
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
                non_star_mask = is_star_arr == 0.0
                non_star_indices = np.where(non_star_mask)[0][:64]
                n_casters_fixed = len(non_star_indices)
                star_radius_au = visual_data[star_idx][3] if star_idx < len(visual_data) else 0.00465
    
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
                ring_centers_buf = np.zeros((16, 3), dtype='f4')
                ring_normals_buf = np.zeros((16, 3), dtype='f4')
                ring_params_buf = np.zeros((16, 3), dtype='f4')
                ring_colors_buf = np.zeros((16, 3), dtype='f4')
                cull_mask_lo = np.zeros(num_bodies, dtype=np.uint32)
                cull_mask_hi = np.zeros(num_bodies, dtype=np.uint32)
                cull_ring_mask = np.zeros(num_bodies, dtype=np.uint32)
                cull_ring_caster_lo = np.zeros(16, dtype=np.uint32)
                cull_ring_caster_hi = np.zeros(16, dtype=np.uint32)
    
                # Release old ring GL objects and rebuild
                for g in ring_render_groups:
                    g['vao'].release()
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
                    for ring_seg in rings_data_r:
                        inner_r = ring_seg['inner'] * body_radius_au_r
                        outer_r = ring_seg['outer'] * body_radius_au_r
                        r_color = hex_to_rgb(ring_seg.get('color', '#ffffff'))
                        r_opacity = ring_seg.get('opacity', 1.0)
                        r_scatter = ring_seg.get('scatter', 0.0)
                        r_asymmetry = ring_seg.get('asymmetry', 0.7)
                        RADIAL_SUBDIVISIONS = 32
                        sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
                        verts_r, indices_r, norm_r, colors_r, shadow_grad_r = generate_ring_arrays(
                            pole_render_r, inner_r, outer_r, r_color, r_opacity, r_scatter, r_asymmetry, sorted_gradient)
                        ring_precomputed.append({
                            'body_idx': body_idx_r, 'verts': verts_r, 'indices': indices_r,
                            'normal': norm_r, 'colors': colors_r, 'inner_r': inner_r, 'outer_r': outer_r,
                            'opacity': r_opacity, 'scatter': r_scatter, 'asymmetry': r_asymmetry,
                            'shadow_grad': shadow_grad_r, 'raw_color': r_color, 'gradient': sorted_gradient,
                        })
    
                ring_idx_by_body = {r['body_idx']: i for i, r in enumerate(ring_precomputed)}
                # Rebuild ring render groups
                ring_gradient_data_sw = np.zeros((16, 256), dtype='f4')
                for j, ring in enumerate(ring_precomputed):
                    if j >= 16: break
                    ring_gradient_data_sw[j, :] = ring['shadow_grad']
                ring_gradient_tex.write(ring_gradient_data_sw.tobytes())
    
                rings_by_body_sw = {}
                for ring in ring_precomputed:
                    bi_r = ring['body_idx']
                    rings_by_body_sw.setdefault(bi_r, []).append(ring)
                for bi_r, rings_r in rings_by_body_sw.items():
                    all_ring_verts_sw = []
                    all_ring_indices_sw = []
                    vert_offset_sw = 0
                    for ring in rings_r:
                        n_v = len(ring['verts'])
                        ring_packed = np.zeros((n_v, 12), dtype='f4')
                        ring_packed[:, 0:3] = ring['verts']
                        ring_packed[:, 3:6] = ring['normal']
                        ring_packed[:, 6:10] = ring['colors']
                        ring_packed[:, 10] = ring['scatter']
                        ring_packed[:, 11] = ring['asymmetry']
                        all_ring_verts_sw.append(ring_packed)
                        all_ring_indices_sw.append(ring['indices'] + vert_offset_sw)
                        vert_offset_sw += n_v
                    all_v_sw = np.concatenate(all_ring_verts_sw, axis=0)
                    all_i_sw = np.concatenate(all_ring_indices_sw, axis=0)
                    ring_vbo_sw = ctx.buffer(all_v_sw.tobytes())
                    ring_ibo_sw = ctx.buffer(all_i_sw.tobytes())
                    ring_vao_sw = ctx.vertex_array(
                        prog_rings,
                        [(ring_vbo_sw, '3f 3f 4f 1f 1f', 'in_position', 'in_normal', 'in_color', 'in_scatter', 'in_asymmetry')],
                        index_buffer=ring_ibo_sw)
                    ring_render_groups.append({'body_idx': bi_r, 'vao': ring_vao_sw, 'num_indices': len(all_i_sw)})
    
                # Reset self.camera and UI state
                self.camera["tracking_idx"] = 0
                self.camera["tracking_is_cmp"] = False
                self.camera["inspected_idx"] = None
                self.camera["inspected_is_cmp"] = False
                self.camera["inspect_bary"] = False
                self.camera["edit_mode"] = False
                self.camera["target"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                self.camera["target_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                self.camera["pan_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                self.camera["distance"] = 45.0
    
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
                        r_scatter = ring_seg.get('scatter', 0.0)
                        r_asymmetry = ring_seg.get('asymmetry', 0.7)
                        sorted_gradient = sorted(ring_seg.get('gradient', []), key=lambda x: x['p'])
                        verts_r, indices_r, norm_r, colors_r, shadow_grad_r = generate_ring_arrays(
                            pole_render_r, inner_r, outer_r, r_color, r_opacity, r_scatter, r_asymmetry, sorted_gradient)
                        self.ring_precomputed_cmp.append({
                            'body_idx': body_idx_r, 'verts': verts_r, 'indices': indices_r,
                            'normal': norm_r, 'colors': colors_r, 'inner_r': inner_r, 'outer_r': outer_r,
                            'opacity': r_opacity, 'scatter': r_scatter, 'asymmetry': r_asymmetry,
                            'shadow_grad': shadow_grad_r, 'raw_color': r_color, 'gradient': sorted_gradient,
                        })
                        
                rings_by_body_cmp = {}
                for ring in self.ring_precomputed_cmp:
                    bi_r = ring['body_idx']
                    rings_by_body_cmp.setdefault(bi_r, []).append(ring)
                for bi_r, rings_r in rings_by_body_cmp.items():
                    all_ring_verts_cmp = []
                    all_ring_indices_cmp = []
                    vert_offset_cmp = 0
                    for ring in rings_r:
                        n_v = len(ring['verts'])
                        ring_packed = np.zeros((n_v, 12), dtype='f4')
                        ring_packed[:, 0:3] = ring['verts']
                        ring_packed[:, 3:6] = ring['normal']
                        ring_packed[:, 6:10] = ring['colors']
                        ring_packed[:, 10] = ring['scatter']
                        ring_packed[:, 11] = ring['asymmetry']
                        all_ring_verts_cmp.append(ring_packed)
                        all_ring_indices_cmp.append(ring['indices'] + vert_offset_cmp)
                        vert_offset_cmp += n_v
                    all_v_cmp = np.concatenate(all_ring_verts_cmp, axis=0)
                    all_i_cmp = np.concatenate(all_ring_indices_cmp, axis=0)
                    ring_vbo_cmp = ctx.buffer(all_v_cmp.tobytes())
                    ring_ibo_cmp = ctx.buffer(all_i_cmp.tobytes())
                    ring_vao_cmp = ctx.vertex_array(
                        prog_rings,
                        [(ring_vbo_cmp, '3f 3f 4f 1f 1f', 'in_position', 'in_normal', 'in_color', 'in_scatter', 'in_asymmetry')],
                        index_buffer=ring_ibo_cmp)
                    self.ring_render_groups_cmp.append({'body_idx': bi_r, 'vao': ring_vao_cmp, 'num_indices': len(all_i_cmp)})
                    
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
                            
                            r_au = radius / 1.496e8
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
                            
                            min_px = 3.0 if btype == "Star" else (1.0 if btype == "Moon" else 2.0)
                            new_vis = np.array([color[0], color[1], color[2], r_au, min_px, 0, 1, 0, 0], dtype='f4')
                            visual_arr = np.vstack([visual_arr, new_vis])
                            visual_colors_f8 = np.vstack([visual_colors_f8, np.array(color, dtype='f8')])
                            body_colors = np.vstack([body_colors, np.array(color, dtype='f4')])
                            is_star_val = 1.0 if btype == "Star" else 0.0
                            is_star_arr = np.append(is_star_arr, is_star_val)
                            visual_data.append([color[0], color[1], color[2], r_au, min_px, 0, 1, 0, 0])
                            
                            inst_data_lo = np.vstack([inst_data_lo, np.zeros(INSTANCE_FLOATS, dtype='f4')])
                            inst_data_hi = np.vstack([inst_data_hi, np.zeros(INSTANCE_FLOATS, dtype='f4')])
                            focused_mask = np.append(focused_mask, False)
                            
                            non_star_indices = np.where(is_star_arr == 0.0)[0][:64]
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
                            
                            if star_idx == idx:
                                star_idx = 0
                            elif star_idx > idx:
                                star_idx -= 1
                                
                            non_star_indices = np.where(is_star_arr == 0.0)[0][:64]
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
                                
                            ring_idx_by_body = {r['body_idx']: i for i, r in enumerate(ring_precomputed)}
                            ring_render_groups = [g for g in ring_render_groups if g['body_idx'] != idx]
                            for g in ring_render_groups:
                                if g['body_idx'] > idx: g['body_idx'] -= 1
                                
                        elif op["action"] == "UPDATE":
                            idx = op["idx"]
                            if "mass" in op:
                                bodies_data[idx]["m"] = float(op["mass"])
                            if "radius" in op:
                                bodies_data[idx]["r"] = float(op["radius"] / 696340.0)
                                r_au = op["radius"] / 1.496e8
                                body_radii[idx] = r_au
                                visual_arr[idx][3] = r_au
                                visual_data[idx][3] = r_au
                                for a in atmo_bodies:
                                    if a['body_idx'] == idx:
                                        a['planet_radius_km'] = float(op["radius"])
                                        a['surface_radius_au'] = r_au
                                        if a['atmo_radius_km'] < a['planet_radius_km']:
                                            a['atmo_radius_km'] = a['planet_radius_km'] * 1.025
                                            a['atmo_radius_au'] = a['atmo_radius_km'] / 1.496e8
                                        if 'lut_tex' in a:
                                            a['lut_tex'].release()
                                            del a['lut_tex']
                            if "color" in op:
                                c = op["color"]
                                bodies_data[idx]["color"] = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
                                visual_arr[idx][0:3] = c
                                visual_colors_f8[idx] = np.array(c, dtype='f8')
                                body_colors[idx] = np.array(c, dtype='f4')
                                visual_data[idx][0:3] = c
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
                                min_px = 3.0 if op["type"] == "Star" else (1.0 if op["type"] == "Moon" else 2.0)
                                visual_arr[idx][4] = min_px
                                visual_data[idx][4] = min_px
                                non_star_indices = np.where(is_star_arr == 0.0)[0][:64]
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

            now = time.time()
            dt_render = min(now - last_render_time, 0.1)
            last_render_time = now
            
            glfw.poll_events()
            
            self.impl.process_inputs()
            imgui.new_frame()
            
            if self.fb_width <= 0 or self.fb_height <= 0:
                imgui.end_frame()
                continue
            
            ctx.viewport = (0, 0, self.fb_width, self.fb_height)
            ctx.screen.use()
            ctx.clear(0.02, 0.02, 0.03, 1.0) 
    
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
    
            cur_y, cur_m, cur_d, cur_h, cur_mn = format_sim_time(display_t)
    
            if ephemeris_mode_active and sys_mgr_spice.kernels_loaded:
                epoch_dt = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
                et_epoch = sys_mgr_spice.datetime_to_et(epoch_dt)
                et = et_epoch + display_t * 365.25 * 86400.0
                
                if getattr(self, "_ephem_mapping_ver", None) != id(bodies_data):
                    self._ephem_mapping = sys_mgr_spice.get_body_mapping(bodies_data, et_epoch)
                    self._ephem_mapping_ver = id(bodies_data)
                    
                sys_mgr_spice.populate_states_fast(et, self._ephem_mapping, pos_snap_render, vel_snap_render, spice_valid_mask)
    
            lerp_factor = 1.0 - math.exp(-15.0 * dt_render)
            self.camera["distance_actual"] += (self.camera["distance"] - self.camera["distance_actual"]) * lerp_factor
            self.camera["yaw_actual"] += (self.camera["yaw"] - self.camera["yaw_actual"]) * lerp_factor
            self.camera["pitch_actual"] += (self.camera["pitch"] - self.camera["pitch_actual"]) * lerp_factor
            self.camera["target_offset"] *= math.exp(-10.0 * dt_render)
    
            compute_barycenters(pos_snap_render, vel_snap_render, mass_snap, parent_snap,
                                subsys_pos_buf, subsys_vel_buf, subsys_mass_buf)
            if self.comparison_enabled:
                compute_barycenters(self.pos_snap_cmp, self.vel_snap_cmp, self.mass_snap_cmp, self.parent_snap_cmp,
                                    self.subsys_pos_buf_cmp, self.subsys_vel_buf_cmp, self.subsys_mass_buf_cmp)
    
            if self.camera["tracking_idx"] is not None:
                track_idx = self.camera["tracking_idx"]
                if self.camera.get("tracking_is_cmp", False):
                    if self.camera["tracking_mode"] == "barycenter":
                        base_pos = self.subsys_pos_buf_cmp[track_idx].copy() + np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                    else:
                        base_pos = self.pos_snap_cmp[track_idx].copy() + np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                else:
                    if self.camera["tracking_mode"] == "barycenter":
                        base_pos = subsys_pos_buf[track_idx].copy()
                    else:
                        base_pos = pos_snap_render[track_idx].copy()
                self.camera["target"] = base_pos.copy()
            else:
                base_pos = np.array(self.camera["target"], dtype='f8')
                
            cam_origin = base_pos + self.camera["target_offset"] + self.camera["pan_offset"]
    
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
            
            yaw_rad, pitch_rad = math.radians(self.camera["yaw_actual"]), math.radians(self.camera["pitch_actual"])
            cam_pos_f8 = np.array([-math.cos(yaw_rad) * math.cos(pitch_rad) * self.camera["distance_actual"],
                                   -math.sin(pitch_rad) * self.camera["distance_actual"],
                                   -math.sin(yaw_rad) * math.cos(pitch_rad) * self.camera["distance_actual"]], dtype='f8')
                                   
            cam_pos = cam_pos_f8.astype('f4')
            view = matrix44.create_look_at(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], dtype='f4')
            near = max(self.camera["distance_actual"] * 0.0001, 1e-9)
            
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
                
            # Set the far plane dynamically to ensure no clipping, with a minimum of 100.0 AU
            far = max(self.camera["distance_actual"] + max_dist_from_target * 2.0 + 10.0, 100.0)
            depth_C = 1.0 / max(near, 1e-12)
            aspect_ratio = self.fb_width / max(self.fb_height, 1)
            projection = matrix44.create_perspective_projection_matrix(self.camera["fov"], aspect_ratio, near, far, dtype='f4')
    
            n_ring_planes = 0
            ring_centers_buf[:] = 0
            ring_normals_buf[:] = 0
            ring_params_buf[:] = 0
            ring_colors_buf[:] = 0
            
            for ring in ring_precomputed:
                if n_ring_planes >= 16:
                    break
                bi = ring['body_idx']
                ring_centers_buf[n_ring_planes] = pos_rel_all[bi]
                ring_normals_buf[n_ring_planes] = ring['normal']
                ring_params_buf[n_ring_planes, 0] = ring['inner_r']
                ring_params_buf[n_ring_planes, 1] = ring['outer_r']
                ring_params_buf[n_ring_planes, 2] = ring['opacity']
                ring_colors_buf[n_ring_planes, 0:3] = ring['raw_color']
                n_ring_planes += 1
    
            caster_indices = non_star_indices
            n_casters_fixed = min(len(caster_indices), 64)
            
            compute_ring_culling(
                pos_rel_all, body_radii, star_idx,
                n_casters_fixed, caster_indices[:n_casters_fixed],
                n_ring_planes, ring_centers_buf, ring_normals_buf, ring_params_buf[:, 1],
                num_bodies, cull_ring_caster_lo, cull_ring_caster_hi
            )
            
            vp_matrix = np.asarray(view, dtype=np.float32) @ np.asarray(projection, dtype=np.float32)
            frustum_planes = extract_frustum_planes(vp_matrix)
            
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
                    self.camera["tracking_idx"] = best_idx
                    self.camera["tracking_is_cmp"] = best_is_cmp

    
            all_instances = np.zeros((total_render_bodies, INSTANCE_FLOATS), dtype='f4')
            all_instances[:num_bodies, 0:3] = pos_rel_all
            all_instances[:num_bodies, 3:8] = visual_arr[:, 0:5]
            all_instances[:num_bodies, 8] = is_star_arr
            all_instances[:num_bodies, 9:12] = visual_arr[:, 5:8]
            all_instances[:num_bodies, 12] = visual_arr[:, 8]
            
            if self.comparison_enabled:
                cmp_pos_rel = self.pos_snap_cmp - cam_origin + np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                all_instances[num_bodies:, 0:3] = cmp_pos_rel.astype('f4')
                all_instances[num_bodies:, 3:8] = self.visual_arr_cmp[:, 0:5]
                all_instances[num_bodies:, 8] = self.is_star_arr_cmp
                all_instances[num_bodies:, 9:12] = self.visual_arr_cmp[:, 5:8]
                all_instances[num_bodies:, 12] = self.visual_arr_cmp[:, 8]
            
            # Precalculate planetshine bounce light direction & color on CPU using Numba
            is_star_mask = all_instances[:total_render_bodies, 8] > 0.5
            star_positions = all_instances[:total_render_bodies][is_star_mask, 0:3]
            star_colors = all_instances[:total_render_bodies][is_star_mask, 3:6]
            hdr_enabled = self.camera.get("hdr_enabled", True)
            planetshine_dirs, planetshine_colors = compute_planetshine_numba(
                all_instances[:total_render_bodies, 0:3],
                all_instances[:total_render_bodies, 6],
                all_instances[:total_render_bodies, 3:6],
                all_instances[:total_render_bodies, 8],
                star_positions,
                star_colors,
                hdr_enabled
            )
            all_instances[:total_render_bodies, 16:19] = planetshine_dirs
            all_instances[:total_render_bodies, 19:22] = planetshine_colors
            
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
            
            all_instances_buffer.bind_to_storage_buffer(binding=2)
            vis_lo_buffer.bind_to_storage_buffer(binding=3)
            vis_hi_buffer.bind_to_storage_buffer(binding=4)
            vis_ultra_buffer.bind_to_storage_buffer(binding=5)
            draw_cmds_buffer.bind_to_storage_buffer(binding=6)
            focused_mask_buffer.bind_to_storage_buffer(binding=7)
            
            prog_culling_compute.run((total_render_bodies + 255) // 256, 1, 1)
            ctx.memory_barrier()
    
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
                    if not self.camera.get("tracking_is_cmp", False) and t_idx is not None and t_idx != star_idx and t_idx < num_bodies:
                        primary_idx = t_idx
                        if parent_snap[t_idx] >= 0 and parent_snap[parent_snap[t_idx]] == star_idx:
                            primary_idx = parent_snap[t_idx]
                        
                        lod_levels[primary_idx] = 0.0
                        for i in range(num_bodies):
                            if parent_snap[i] == primary_idx:
                                lod_levels[i] = 1.0
                                
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
                            recompute_orbits_cmp = False
                                
                    if recompute_orbits_cmp:
                        lod_levels_cmp = np.full(self.num_bodies_cmp, 2.0, dtype=np.float64)
                        for i in range(self.num_bodies_cmp):
                            if self.parent_snap_cmp[i] == self.star_idx_cmp:
                                lod_levels_cmp[i] = 1.0
                                
                        t_idx_cmp = self.camera.get("tracking_idx")
                        if self.camera.get("tracking_is_cmp", False) and t_idx_cmp is not None and t_idx_cmp != self.star_idx_cmp and t_idx_cmp < self.num_bodies_cmp:
                            primary_idx_cmp = t_idx_cmp
                            if self.parent_snap_cmp[t_idx_cmp] >= 0 and self.parent_snap_cmp[self.parent_snap_cmp[t_idx_cmp]] == self.star_idx_cmp:
                                primary_idx_cmp = self.parent_snap_cmp[t_idx_cmp]
                                
                            lod_levels_cmp[primary_idx_cmp] = 0.0
                            for i in range(self.num_bodies_cmp):
                                if self.parent_snap_cmp[i] == primary_idx_cmp:
                                    lod_levels_cmp[i] = 1.0
                        
                        cam_origin_cmp = cam_origin - np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                        offset_vec = np.array([-self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                        self.n_orbits_cmp = compute_all_orbits_batch(
                            self.pos_snap_cmp, self.vel_snap_cmp, self.mass_snap_cmp, self.parent_snap_cmp,
                            self.subsys_pos_buf_cmp, self.subsys_vel_buf_cmp, self.subsys_mass_buf_cmp,
                            self.visual_colors_f8_cmp, cam_origin_cmp, G, max_orbits, self.orbit_data_buf_cmp, lod_levels_cmp,
                            offset_vec)
                        last_orbit_pos_snap_cmp = self.pos_snap_cmp.copy()
                        
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
                        
                    stars_pos_radius.append([pos[0], pos[1], pos[2], radius])
                    stars_colors.append([color[0], color[1], color[2], lum])
                    
            num_stars = len(stars_pos_radius)
            num_stars = min(num_stars, 16)
            stars_pos_radius = stars_pos_radius[:num_stars]
            stars_colors = stars_colors[:num_stars]
            
            if num_stars == 0:
                num_stars = 1
                stars_pos_radius = [[0.0, 0.0, 0.0, 0.00465]]
                stars_colors = [[1.0, 1.0, 1.0, 1.0]]
            
            caster_data_buf[:n_casters_fixed, 0:3] = pos_rel_all[caster_indices[:n_casters_fixed]]
            caster_data_buf[:n_casters_fixed, 3] = body_radii[caster_indices[:n_casters_fixed]]
            if n_casters_fixed < 64:
                caster_data_buf[n_casters_fixed:] = 0
                
            caster_poles_obl_buf[:n_casters_fixed, 0:3] = all_instances[caster_indices[:n_casters_fixed], 9:12]
            caster_poles_obl_buf[:n_casters_fixed, 3] = all_instances[caster_indices[:n_casters_fixed], 12]
            if n_casters_fixed < 64:
                caster_poles_obl_buf[n_casters_fixed:] = 0
                
            caster_colors_buf[:n_casters_fixed, 0:3] = body_colors[caster_indices[:n_casters_fixed]]
            caster_colors_buf[:n_casters_fixed, 3] = 1.0
            if n_casters_fixed < 64:
                caster_colors_buf[n_casters_fixed:] = 0
                
            # Build atmosphere lookup dict once per frame to avoid expensive generator allocations inside the loop
            atmo_by_body = {a['body_idx']: a for a in atmo_bodies}
            for i_c in range(n_casters_fixed):
                b_idx = caster_indices[i_c]
                atmo = atmo_by_body.get(b_idx)
                if atmo:
                    beta_r = atmo['beta_rayleigh']
                    beta_m = atmo['beta_mie']
                    h_r = atmo['h_rayleigh']
                    h_m = atmo['h_mie']
                    R_km = atmo['planet_radius_km']
                    od_r = beta_r * 1000.0 * math.sqrt(2.0 * math.pi * R_km * h_r)
                    od_m = beta_m * 1000.0 * math.sqrt(2.0 * math.pi * R_km * h_m)
                    transmittance = np.exp(-(od_r + od_m) * 0.5)
                    caster_atmos_buf[i_c, 0:3] = transmittance
                    caster_atmos_buf[i_c, 3] = atmo['atmo_radius_au'] - atmo['surface_radius_au']
                else:
                    caster_atmos_buf[i_c] = 0.0
            if n_casters_fixed < 64:
                caster_atmos_buf[n_casters_fixed:] = 0
    
            ubo_staging[0:16] = projection.ravel()
            ubo_staging[16:32] = view.ravel()
            
            # Write num stars
            ubo_num_stars_int_view[0] = num_stars
            ubo_staging[33:36] = 0.0 # padding
            
            # Write stars arrays
            stars_pos_radius_flat = np.zeros(64, dtype=np.float32)
            stars_colors_flat = np.zeros(64, dtype=np.float32)
            for s_idx in range(num_stars):
                stars_pos_radius_flat[s_idx*4 : (s_idx+1)*4] = stars_pos_radius[s_idx]
                stars_colors_flat[s_idx*4 : (s_idx+1)*4] = stars_colors[s_idx]
                
            ubo_staging[36:100] = stars_pos_radius_flat
            ubo_staging[100:164] = stars_colors_flat
            
            ubo_staging[164] = far
            ubo_staging[165] = depth_C
            ubo_casters_int_view[0] = n_casters_fixed
            # Note: ubo_casters_int_view maps to ubo_staging[166:167]
            ubo_staging[167] = 0.0 # padding
            
            ubo_staging[168:424] = caster_data_buf.ravel()
            ubo_staging[424:680] = caster_poles_obl_buf.ravel()
            ubo_staging[680:936] = caster_colors_buf.ravel()
            ubo_staging[936:1192] = caster_atmos_buf.ravel()
            
            scene_ubo.write(ubo_staging.tobytes())
            
            fov_factor = 1.0 / math.tan(math.radians(self.camera["fov"] / 2.0))
            uniform_screen_height.value = self.fb_height
            uniform_fov_factor.value = fov_factor
            uniform_num_ring_planes.value = n_ring_planes
            if 'u_planetshine_enabled' in prog_spheres:
                prog_spheres['u_planetshine_enabled'].value = self.camera.get("planetshine_enabled", True)
            if 'u_camera_pos' in prog_spheres:
                prog_spheres['u_camera_pos'].value = tuple(cam_pos)
            if 'u_ringshine_enabled' in prog_spheres:
                prog_spheres['u_ringshine_enabled'].value = self.camera.get("ringshine_enabled", True)
            if n_ring_planes > 0:
                uniform_ring_centers.write(ring_centers_buf)
                uniform_ring_normals.write(ring_normals_buf)
                uniform_ring_params.write(ring_params_buf)
                if uniform_ring_colors:
                    uniform_ring_colors.write(ring_colors_buf)
            
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
            
            # Pass Exposure and HDR setting to shaders
            exposure = self.camera.get("exposure", 1.0)
            hdr_enabled = self.camera.get("hdr_enabled", True)
            
            for prog in (prog_spheres, prog_rings, prog_atmo):
                if 'u_exposure' in prog:
                    prog['u_exposure'].value = exposure
                if 'u_hdr_enabled' in prog:
                    prog['u_hdr_enabled'].value = hdr_enabled
            
            ctx.enable(moderngl.CULL_FACE)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            vis_lo_buffer.bind_to_storage_buffer(binding=3)
            vao_lo.render_indirect(draw_cmds_buffer, moderngl.TRIANGLES, first=0)
            
            vis_hi_buffer.bind_to_storage_buffer(binding=3)
            vao_hi.render_indirect(draw_cmds_buffer, moderngl.TRIANGLES, first=1)
            
            vis_ultra_buffer.bind_to_storage_buffer(binding=3)
            vao_ultra.render_indirect(draw_cmds_buffer, moderngl.TRIANGLES, first=2)
            ctx.disable(moderngl.BLEND)
            ctx.disable(moderngl.CULL_FACE)
    
    
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    
            if show_orbits:
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

            def render_atmosphere_pass(clip_mode):
                if not sorted_atmos:
                    return
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
                ctx.depth_func = '<='
                ctx.enable(moderngl.CULL_FACE)
                ctx.depth_mask = False
    
                u_atmo_quality_uniform.value = atmo_quality
                u_atmo_camera_pos.write(cam_pos)
                AU_TO_KM = 149597870.7
                u_atmo_au_to_km.value = AU_TO_KM
            
                u_atmo_num_ring_planes.value = n_ring_planes
                if n_ring_planes > 0:
                    u_atmo_ring_centers.write(ring_centers_buf)
                    u_atmo_ring_normals.write(ring_normals_buf)
                    u_atmo_ring_params.write(ring_params_buf)
                
                if u_atmo_clip_mode is not None:
                    u_atmo_clip_mode.value = clip_mode
    
                for sq_dist, atmo, is_cmp in sorted_atmos:
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
                        n_samples = 16
                        n_light = 4
                    else:
                        if apparent_px < 50:
                            n_samples, n_light = 8, 4
                        elif apparent_px < 200:
                            n_samples, n_light = 16, 6
                        else:
                            n_samples, n_light = 32, 8
                            
                    if 'lut_tex' not in atmo:
                        build_atmo_lut(atmo)
                    atmo['lut_tex'].use(location=1)
    
                    scaled_intensity = atmo['intensity']
    
                    u_atmo_body_offset.write(body_pos_rel.astype('f4'))
                    u_atmo_radius_au_uniform.value = float(atmo['atmo_radius_au'])
                    u_atmo_planet_radius.value = float(atmo['planet_radius_km'])
                    u_atmo_atmo_radius.value = float(atmo['atmo_radius_km'])
                    u_atmo_beta_rayleigh.write(atmo['beta_rayleigh'])
                    u_atmo_h_rayleigh.value = atmo['h_rayleigh']
                    u_atmo_beta_mie.value = atmo['beta_mie']
                    u_atmo_h_mie.value = atmo['h_mie']
                    u_atmo_mie_g.value = atmo['mie_g']
                    u_atmo_beta_absorption.write(atmo['beta_absorption'])
                    u_atmo_sun_intensity.value = scaled_intensity
                    
                    if num_stars > 0:
                        star_pos = pos_rel_all[star_idx] if not is_cmp else cmp_pos_rel[self.star_idx_cmp]
                        body_to_star = star_pos - body_pos_rel
                        dist_to_star = np.linalg.norm(body_to_star)
                        star_r_au = body_radii[star_idx] if not is_cmp else self.body_radii_cmp[self.star_idx_cmp]
                        sin_star_angular = star_r_au / max(dist_to_star, 1e-12)
                    else:
                        sin_star_angular = 0.0
                    u_atmo_star_angular_radius.value = float(sin_star_angular)

                    u_atmo_num_samples.value = n_samples
                    if u_atmo_num_light_samples: u_atmo_num_light_samples.value = n_light
                    u_atmo_pole_obl.value = (
                        float(all_instances[body_idx_in_unified, 9]),
                        float(all_instances[body_idx_in_unified, 10]),
                        float(all_instances[body_idx_in_unified, 11]),
                        float(all_instances[body_idx_in_unified, 12])
                    )
                    
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
                    
                    active_casters_buf = np.zeros((8, 4), dtype='f4')
                    active_poles_obl_buf = np.zeros((8, 4), dtype='f4')
                    active_atmos_buf = np.zeros((8, 4), dtype='f4')
                    
                    atmo_lookup = {a['body_idx']: a for a in atmo_bodies}
                    atmo_lookup_cmp = {a['body_idx']: a for a in self.atmo_bodies_cmp} if self.comparison_enabled else {}
                    
                    for i_ac, idx_u in enumerate(active_indices):
                        is_c_cmp = (idx_u >= num_bodies)
                        c_local_idx = idx_u - num_bodies if is_c_cmp else idx_u
                        
                        pos_c = cmp_pos_rel[c_local_idx] if is_c_cmp else pos_rel_all[c_local_idx]
                        rad_c = self.body_radii_cmp[c_local_idx] if is_c_cmp else body_radii[c_local_idx]
                        
                        active_casters_buf[i_ac, 0:3] = pos_c
                        active_casters_buf[i_ac, 3] = rad_c
                        
                        active_poles_obl_buf[i_ac, 0:3] = all_instances[idx_u, 9:12]
                        active_poles_obl_buf[i_ac, 3] = all_instances[idx_u, 12]
                        
                        lookup = atmo_lookup_cmp if is_c_cmp else atmo_lookup
                        atmo_info = lookup.get(c_local_idx)
                        if atmo_info:
                            beta_r_c = atmo_info['beta_rayleigh']
                            beta_m_c = atmo_info['beta_mie']
                            h_r_c = atmo_info['h_rayleigh']
                            h_m_c = atmo_info['h_mie']
                            R_km_c = atmo_info['planet_radius_km']
                            od_r_c = beta_r_c * 1000.0 * math.sqrt(2.0 * math.pi * R_km_c * h_r_c)
                            od_m_c = beta_m_c * 1000.0 * math.sqrt(2.0 * math.pi * R_km_c * h_m_c)
                            transmittance = np.exp(-(od_r_c + od_m_c) * 0.5)
                            thickness = atmo_info['atmo_radius_au'] - atmo_info['surface_radius_au']
                            active_atmos_buf[i_ac, 0:3] = transmittance
                            active_atmos_buf[i_ac, 3] = thickness
                            
                    u_atmo_num_active_casters.value = n_active
                    u_atmo_active_casters.write(active_casters_buf.tobytes())
                    u_atmo_active_caster_poles_obl.write(active_poles_obl_buf.tobytes())
                    u_atmo_active_caster_atmos.write(active_atmos_buf.tobytes())
                    
                    u_atmo_body_idx_uni.value = body_idx_in_unified
                    
                    vao_atmo.render(moderngl.TRIANGLES)
    
                ctx.depth_mask = True
                ctx.disable(moderngl.CULL_FACE)
                ctx.depth_func = '<'
                ctx.disable(moderngl.BLEND)

            # --- Pass 1: Atmosphere behind rings ---
            render_atmosphere_pass(1)
    
            # --- Render Rings ---
            if ring_render_groups or (self.comparison_enabled and self.ring_render_groups_cmp):
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
                u_ring_camera_pos.write(cam_pos)
                ctx.depth_mask = False
                u_ring_clip_mode.value = 0
                if u_ring_planetshine_enabled is not None:
                    u_ring_planetshine_enabled.value = self.camera.get("planetshine_enabled", True)
                
                if ring_render_groups:
                    for group in ring_render_groups:
                        bi = group['body_idx']
                        body_pos_rel = pos_rel_all[bi]
                        u_ring_body_offset.write(body_pos_rel)
                        u_ring_host_pos.write(body_pos_rel)
                        u_ring_host_radius.value = float(body_radii[bi])
                        if u_ring_host_color is not None:
                            u_ring_host_color.value = tuple(float(c) for c in body_colors[bi])
                        u_ring_host_pole_obl.value = (
                            float(all_instances[bi, 9]),
                            float(all_instances[bi, 10]),
                            float(all_instances[bi, 11]),
                            float(all_instances[bi, 12])
                        )
                        k = ring_idx_by_body.get(bi)
                        if k is not None:
                            u_ring_caster_mask_lo_uni.value = int(cull_ring_caster_lo[k])
                            u_ring_caster_mask_hi_uni.value = int(cull_ring_caster_hi[k])
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
                        u_ring_host_pole_obl.value = (
                            float(all_instances[body_idx_in_unified, 9]),
                            float(all_instances[body_idx_in_unified, 10]),
                            float(all_instances[body_idx_in_unified, 11]),
                            float(all_instances[body_idx_in_unified, 12])
                        )
                        u_ring_caster_mask_lo_uni.value = 0
                        u_ring_caster_mask_hi_uni.value = 0
                        group['vao'].render(moderngl.TRIANGLES, vertices=group['num_indices'])
                ctx.depth_mask = True
                ctx.disable(moderngl.BLEND)
    
            # --- Pass 2: Atmosphere in front of rings ---
            render_atmosphere_pass(2)
            force_layout = getattr(self, "_last_fb_width", 0) != self.fb_width or getattr(self, "_last_fb_height", 0) != self.fb_height
            cond_layout = imgui.ALWAYS if force_layout else imgui.ONCE
            
            imgui.set_next_window_position(20, 20, cond_layout)
            imgui.set_next_window_size(230, min(340, self.fb_height - 40), cond_layout)
            imgui.begin("Simulation Controls")
            
            imgui.text("Current Date:")
            imgui.text(f"{cur_y:04d}-{cur_m:02d}-{cur_d:02d} {cur_h:02d}:{cur_mn:02d} UTC")
            
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
            
            # Forward-only logarithmic slider (0 = realtime, 9 = max speed)
            val_log = math.log10(max(1.0, abs(self.time_ctrl["multiplier"])))
            changed, new_log = imgui.slider_float("##speed", val_log, 0.0, 9.0, "")
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
                
                imgui.text("Time (HH:MM)")
                imgui.push_item_width(40)
                _, jump_date[3] = imgui.input_int("##Hour", jump_date[3], step=0)
                imgui.same_line()
                imgui.text(":")
                imgui.same_line()
                _, jump_date[4] = imgui.input_int("##Minute", jump_date[4], step=0)
                imgui.pop_item_width()
                
                jump_date[1] = max(1, min(12, jump_date[1]))
                jump_date[2] = max(1, min(31, jump_date[2]))
                jump_date[3] = max(0, min(23, jump_date[3]))
                jump_date[4] = max(0, min(59, jump_date[4]))
                
                if ephemeris_mode_active:
                    if imgui.button("Jump to Date"):
                        target_t = sim_time_from_date(jump_date[0], jump_date[1], jump_date[2], jump_date[3], jump_date[4])
                        self.time_ctrl["sync_t"] = target_t
                else:
                    if imgui.button("Render Timeline"):
                        target_t = sim_time_from_date(jump_date[0], jump_date[1], jump_date[2], jump_date[3], jump_date[4])
                        self.time_ctrl["target_t"] = target_t
                        self.time_ctrl["render_timeline"] = True
            
            imgui.separator()
            if imgui.button("Graphics & Quality Settings..."):
                self.camera["show_settings_modal"] = True
                
            imgui.text(f"FOV: {self.camera['fov']:.1f} deg")
            if imgui.button("Reset FOV"): self.camera["fov"] = 45.0
            imgui.end()
            
            if self.camera.get("show_settings_modal", False):
                imgui.set_next_window_size(320, 240, imgui.FIRST_USE_EVER)
                expanded, self.camera["show_settings_modal"] = imgui.begin("Graphics & Quality Settings", True)
                if expanded:
                    # Atmosphere Quality
                    _, atmo_quality = imgui.combo("Atmosphere Quality", atmo_quality, ["Off", "Low (2D Shadows)", "High (Volumetric)"])
                    
                    # Exposure & HDR
                    _, self.camera["hdr_enabled"] = imgui.checkbox("HDR Mode", self.camera.get("hdr_enabled", True))
                    if self.camera.get("hdr_enabled", True):
                        _, self.camera["exposure"] = imgui.slider_float("Exposure", self.camera.get("exposure", 1.0), 0.0001, 10000.0, "%.4f", imgui.SLIDER_FLAGS_LOGARITHMIC)
                    
                    # Orbit Lines
                    _, show_orbits = imgui.checkbox("Show Orbits", show_orbits)
                    if show_orbits:
                        _, orbit_fade_dir_idx = imgui.combo("Orbit Fade", orbit_fade_dir_idx, ["Bright Behind", "Bright Ahead"])
                        orbit_fade_dir = 1.0 if orbit_fade_dir_idx == 0 else -1.0
                        _, orbit_min_alpha = imgui.slider_float("Min Alpha", orbit_min_alpha, 0.0, 1.0, "%.2f")
                        
                    imgui.separator()
                    # Bounce lighting toggles
                    _, self.camera["planetshine_enabled"] = imgui.checkbox("Enable Planetshine/Moonshine", self.camera.get("planetshine_enabled", True))
                    _, self.camera["ringshine_enabled"] = imgui.checkbox("Enable Ringshine", self.camera.get("ringshine_enabled", True))
                    

                    if imgui.button("Close"):
                        self.camera["show_settings_modal"] = False
                imgui.end()
    
            imgui.set_next_window_position(20, 370, cond_layout)
            imgui.set_next_window_size(230, min(600, self.fb_height - 380), cond_layout)
            imgui.begin("System Hierarchy")
            
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
                        "age": 4.6,
                    }
                imgui.end_popup()
    
            imgui.separator()
            
            # ── Ephemeris Mode UI ──
            if active_system_name == SystemManager.SOLAR_SYSTEM_NAME and not ephemeris_mode_active:
                
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
                    
                    sys_mgr_spice.download_kernels_async(on_complete=_on_spice_ready)
    
                if imgui.button("Switch to Ephemeris Mode", width=-1):
                    if not sys_mgr_spice.settings_initialized:
                        imgui.open_popup("Ephemeris Setup")
                    else:
                        _trigger_ephem_switch()
                        
                if imgui.button("Ephemeris Kernel Settings", width=-1):
                    imgui.open_popup("Ephemeris Setup")
                    
                if imgui.begin_popup_modal("Ephemeris Setup", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                    imgui.text("Select which SPICE kernels to download and load:")
                    imgui.text_colored("Warning: High resolution satellite kernels are large and take time to download.", 1.0, 0.5, 0.2)
                    imgui.separator()
                    
                    import os
                    for k_name in sys_mgr_spice.DEFAULT_KERNELS.keys():
                        desc = sys_mgr_spice.KERNEL_DESCRIPTIONS.get(k_name, k_name)
                        if os.path.exists(os.path.join(sys_mgr_spice.KERNEL_DIR, k_name)):
                            desc += " [Downloaded]"
                        is_enabled = sys_mgr_spice.enabled_kernels.get(k_name, False)
                        changed, new_val = imgui.checkbox(desc, is_enabled)
                        if changed:
                            sys_mgr_spice.enabled_kernels[k_name] = new_val
                            
                    imgui.separator()
                    if imgui.button("Save & Switch to Ephemeris Mode"):
                        sys_mgr_spice.save_settings()
                        imgui.close_current_popup()
                        _trigger_ephem_switch()
                    imgui.same_line()
                    if imgui.button("Cancel"):
                        imgui.close_current_popup()
                    imgui.end_popup()
                
                if sys_mgr_spice.is_downloading:
                    imgui.text_colored(sys_mgr_spice.download_status, 1.0, 0.8, 0.2)
                    imgui.progress_bar(sys_mgr_spice.download_progress, size=(-1, 0))
                imgui.separator()
                
            elif ephemeris_mode_active:
                imgui.text_colored("EPHEMERIS MODE ACTIVE", 0.3, 1.0, 0.3)
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
                    target_name = SystemManager.SOLAR_SYSTEM_NAME
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
                            "ephemeris_exit": True
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
                            "ephemeris_exit": True
                        }
                    with self.shared_state["lock"]:
                        self.shared_state["system_switch_request"] = req
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
                
                imgui.set_next_window_position(self.fb_width - 300, 20, cond_layout)
                imgui.set_next_window_size(280, min(580, self.fb_height - 40), cond_layout)
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
                                self.camera["edit_data"] = {
                                    "mass": float(mass_snap[insp_idx]),
                                    "radius": float(body_info.get('r', 0.0) * 696340.0),
                                    "type": body_info.get("type", "Moon"),
                                    "a": 0.0, "e": 0.0, "inc": 0.0, "Omega": 0.0, "omega": 0.0, "M": 0.0,
                                    "init_orbit": True
                                }
                    
                    imgui.separator()
                    imgui.text_colored("Physical Properties", 1.0, 0.85, 0.4)
                    
                    body_mass = cur_mass_snap[insp_idx]
                    body_r_km = body_info.get('r', 0.0) * 696340.0
                    
                    if not insp_is_cmp and self.camera["edit_mode"] and not inspect_bary:
                        _, self.camera["edit_data"]["mass"] = imgui.input_double(u"Mass (M\u2609)", self.camera["edit_data"]["mass"], format="%e")
                        _, self.camera["edit_data"]["radius"] = imgui.input_double("Radius (km)", self.camera["edit_data"]["radius"], format="%.1f")
                        edit_mass = self.camera["edit_data"]["mass"]
                        edit_r_km = self.camera["edit_data"]["radius"]
                        if edit_r_km > 0.0:
                            g_m_s2 = (1.32712440018e14 * edit_mass) / (edit_r_km ** 2)
                            g_earth = g_m_s2 / 9.80665
                            if g_m_s2 >= 1e-4:
                                imgui.text("  Surface G: {:.3f} m/s² ({:.3f} g)".format(g_m_s2, g_earth))
                            else:
                                imgui.text("  Surface G: {:.3e} m/s² ({:.3e} g)".format(g_m_s2, g_earth))
                        
                        types_list = ["Star", "Terrestrial", "Gas Giant", "Ice Giant", "Dwarf Planet", "Moon"]
                        if "type" not in self.camera["edit_data"]:
                            self.camera["edit_data"]["type"] = body_info.get("type", "Moon")
                        type_idx = types_list.index(self.camera["edit_data"]["type"]) if self.camera["edit_data"]["type"] in types_list else 1
                        changed_t, type_idx = imgui.combo("Type", type_idx, types_list)
                        if changed_t:
                            self.camera["edit_data"]["type"] = types_list[type_idx]
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
                            orb_mass = cur_subsys_mass_buf[insp_idx]
                        else:
                            rel_r = cur_pos_snap_render[insp_idx] - cur_pos_snap_render[parent_idx]
                            rel_v = cur_vel_snap_render[insp_idx] - cur_vel_snap_render[parent_idx]
                            orb_mass = body_mass
                        
                        ecl_rx, ecl_ry, ecl_rz = rel_r[0], -rel_r[2], rel_r[1]
                        ecl_vx, ecl_vy, ecl_vz = rel_v[0], -rel_v[2], rel_v[1]
                        
                        self.camera.setdefault("inspector_frame", 0)
                        changed_frame, self.camera["inspector_frame"] = imgui.combo("Reference Frame", self.camera["inspector_frame"], ["Ecliptic", "Equatorial"])
                        if changed_frame and not insp_is_cmp and self.camera["edit_mode"]:
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
                            AU_TO_KM = 1.496e8
                            
                            if use_km:
                                imgui.text(f"  Semi-major:  {oe_a * AU_TO_KM:,.0f} km")
                            else:
                                imgui.text(f"  Semi-major:  {oe_a:.6f} AU")
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
                        if use_km:
                            imgui.text(f"  Periapsis:   {periapsis * AU_TO_KM:,.0f} km")
                            imgui.text(f"  Apoapsis:    {apoapsis * AU_TO_KM:,.0f} km")
                            imgui.text(f"  Distance:    {dist_au * AU_TO_KM:,.0f} km")
                        else:
                            imgui.text(f"  Periapsis:   {periapsis:.6f} AU")
                            imgui.text(f"  Apoapsis:    {apoapsis:.6f} AU")
                            imgui.text(f"  Distance:    {dist_au:.6f} AU")
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
                                parent_r_au = cur_bodies_data[parent_idx].get('r', 0.0) * 0.00465
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
                        if imgui.button("Apply Changes", width=-1):
                            ed = self.camera["edit_data"]
                            payload = {
                                "action": "UPDATE",
                                "idx": insp_idx,
                                "mass": ed["mass"],
                                "radius": ed["radius"],
                                "type": ed["type"]
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
                            visual_arr[insp_idx, 3] = body_info['r'] * 0.00465
                            
                            self.camera["edit_mode"] = False
                            
                    imgui.separator()
                    
                    is_tracking_this = (self.camera["tracking_idx"] == insp_idx and self.camera.get("tracking_is_cmp", False) == insp_is_cmp)
                    is_tracking_body = is_tracking_this and self.camera["tracking_mode"] == "body"
                    is_tracking_bary = is_tracking_this and self.camera["tracking_mode"] == "barycenter"
                    
                    if is_tracking_body:
                        imgui.text_colored("Tracking Body", 0.3, 1.0, 0.3)
                    elif imgui.button("Track Body"):
                        if self.camera["tracking_idx"] is not None:
                            old_track_is_cmp = self.camera.get("tracking_is_cmp", False)
                            if self.camera["tracking_mode"] == "barycenter":
                                old_pos_arr = self.subsys_pos_buf_cmp if old_track_is_cmp else subsys_pos_buf
                            else:
                                old_pos_arr = self.pos_snap_cmp if old_track_is_cmp else pos_snap_render
                            old_pos = old_pos_arr[self.camera["tracking_idx"]].copy()
                            if old_track_is_cmp:
                                old_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                            old_pos += self.camera["pan_offset"]
                        else:
                            old_pos = self.camera["target"].copy()
                        
                        new_pos = cur_pos_snap_render[insp_idx].copy()
                        if insp_is_cmp:
                            new_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                            
                        self.camera["target_offset"] += (old_pos - new_pos)
                        self.camera["pan_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                        self.camera["tracking_idx"] = insp_idx
                        self.camera["tracking_is_cmp"] = insp_is_cmp
                        self.camera["tracking_mode"] = "body"
                    
                    imgui.same_line()
                    
                    if is_tracking_bary:
                        imgui.text_colored("Tracking Barycenter", 0.3, 1.0, 0.3)
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
                            old_pos += self.camera["pan_offset"]
                        else:
                            old_pos = self.camera["target"].copy()
                            
                        new_pos = cur_subsys_pos_buf[insp_idx].copy()
                        if insp_is_cmp:
                            new_pos += np.array([self.comparison_offset_au, 0.0, 0.0], dtype='f8')
                            
                        self.camera["target_offset"] += (old_pos - new_pos)
                        self.camera["pan_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                        self.camera["tracking_idx"] = insp_idx
                        self.camera["tracking_is_cmp"] = insp_is_cmp
                        self.camera["tracking_mode"] = "barycenter"
                    
                    if is_tracking_this:
                        if imgui.button("Untrack"):
                            self.camera["target"] = self.camera["target"].copy() + self.camera["pan_offset"]
                            self.camera["target_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                            self.camera["pan_offset"] = np.array([0.0, 0.0, 0.0], dtype='f8')
                            self.camera["tracking_idx"] = None
                            self.camera["tracking_is_cmp"] = False
                            self.camera["tracking_mode"] = "body"
                            
                    
                    if not inspect_bary:
                        imgui.separator()
                        imgui.text_colored("Cosmetics", 1.0, 0.4, 0.4)
                        
                        imgui.same_line(imgui.get_window_width() - 70)
                        if imgui.button("Export"):
                            is_hidden_combo = glfw.get_key(self.window, glfw.KEY_LEFT_BRACKET) == glfw.PRESS and glfw.get_key(self.window, glfw.KEY_RIGHT_BRACKET) == glfw.PRESS
                            
                            def fmt(val, dec=5): return round(float(val), dec)
                            
                            if is_hidden_combo and active_system_name == SystemManager.SOLAR_SYSTEM_NAME:
                                system_file = "data/system.json"
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
                                                    "height": fmt(atmo_item["atmo_radius_km"] - atmo_item["planet_radius_km"], 2),
                                                    "rayleighCoefficients": [fmt(x, 8) for x in atmo_item["beta_rayleigh"]],
                                                    "rayleighScaleHeight": fmt(atmo_item["h_rayleigh"], 2),
                                                    "mieCoefficient": fmt(atmo_item["beta_mie"], 8),
                                                    "mieScaleHeight": fmt(atmo_item["h_mie"], 2),
                                                    "mieAsymmetry": fmt(atmo_item["mie_g"], 4),
                                                    "absorptionCoefficients": [fmt(x, 8) for x in atmo_item["beta_absorption"]],
                                                    "intensity": fmt(atmo_item["intensity"], 3)
                                                }
                                                
                                            ring_segs = [r for r in ring_precomputed if r['body_idx'] == b_idx]
                                            if ring_segs:
                                                s_body["rings"] = []
                                                body_r_au = bodies_data[b_idx].get('radius', 1000.0) / 1.496e8
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
                                                        "gradient": [{"p": fmt(g['p']), "a": fmt(g['a'])} for g in r.get('gradient', [])]
                                                    })
                                                    
                                    with open(system_file, "w") as f:
                                        json.dump(sys_data, f, indent=2)
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
                                        "height": fmt(atmo_item["atmo_radius_km"] - atmo_item["planet_radius_km"], 2),
                                        "rayleighCoefficients": [fmt(x, 8) for x in atmo_item["beta_rayleigh"]],
                                        "rayleighScaleHeight": fmt(atmo_item["h_rayleigh"], 2),
                                        "mieCoefficient": fmt(atmo_item["beta_mie"], 8),
                                        "mieScaleHeight": fmt(atmo_item["h_mie"], 2),
                                        "mieAsymmetry": fmt(atmo_item["mie_g"], 4),
                                        "absorptionCoefficients": [fmt(x, 8) for x in atmo_item["beta_absorption"]],
                                        "intensity": fmt(atmo_item["intensity"], 3)
                                    }
                                    
                                ring_segs = [r for r in ring_precomputed if r['body_idx'] == insp_idx]
                                if ring_segs:
                                    exp["rings"] = []
                                    body_r_au = body_info.get('radius', 1000.0) / 1.496e8
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
                                            "gradient": [{"p": fmt(g['p']), "a": fmt(g['a'])} for g in r.get('gradient', [])]
                                        })
                                
                                import os
                                os.makedirs("exports", exist_ok=True)
                                filename = os.path.join("exports", f"{body_info['name'].replace(' ', '_').lower()}_cosmetics.json")
                                with open(filename, 'w') as f:
                                    json.dump(exp, f, indent=4)
                                print(f"Exported cosmetics to {filename}")
                            
                        changed_c, new_c = imgui.color_edit3("Base Color", *visual_arr[insp_idx, 0:3])
                        if changed_c:
                            visual_arr[insp_idx, 0:3] = new_c
                        
                        if imgui.collapsing_header("Atmosphere")[0]:
                            atmo_item = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)
                            if atmo_item:
                                current_height = atmo_item['atmo_radius_km'] - atmo_item['planet_radius_km']
                                changed_r, new_height = imgui.drag_float("Atmosphere Height (km)", current_height, 1.0, 1.0, body_r_km * 10.0)
                                if changed_r:
                                    atmo_item['atmo_radius_km'] = atmo_item['planet_radius_km'] + new_height
                                    atmo_item['atmo_radius_au'] = atmo_item['atmo_radius_km'] / 1.496e8
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                b_r = atmo_item['beta_rayleigh']
                                changed_b, b_ray = imgui.drag_float3("Rayleigh Beta (x10^-6)", b_r[0]*1e6, b_r[1]*1e6, b_r[2]*1e6, 0.1)
                                if changed_b:
                                    atmo_item['beta_rayleigh'] = np.array([b_ray[0]*1e-6, b_ray[1]*1e-6, b_ray[2]*1e-6], dtype='f4')
                                changed_hr, new_hr = imgui.drag_float("Rayleigh Scale (km)", atmo_item['h_rayleigh'], 0.1, 0.1, 1000.0)
                                if changed_hr:
                                    atmo_item['h_rayleigh'] = new_hr
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                
                                changed_m, b_mie = imgui.drag_float("Mie Beta (x10^-6)", atmo_item['beta_mie']*1e6, 0.1)
                                if changed_m:
                                    atmo_item['beta_mie'] = b_mie * 1e-6
                                changed_hm, new_hm = imgui.drag_float("Mie Scale (km)", atmo_item['h_mie'], 0.1, 0.1, 1000.0)
                                if changed_hm:
                                    atmo_item['h_mie'] = new_hm
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                _, atmo_item['mie_g'] = imgui.slider_float("Mie Asymmetry", atmo_item['mie_g'], 0.0, 0.999)
                                
                                b_a = atmo_item['beta_absorption']
                                changed_a, b_abs = imgui.drag_float3("Absorption Beta (x10^-6)", b_a[0]*1e6, b_a[1]*1e6, b_a[2]*1e6, 0.1)
                                if changed_a:
                                    atmo_item['beta_absorption'] = np.array([b_abs[0]*1e-6, b_abs[1]*1e-6, b_abs[2]*1e-6], dtype='f4')
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                        del atmo_item['lut_tex']
                                    
                                _, atmo_item['intensity'] = imgui.drag_float("Intensity", atmo_item['intensity'], 0.5, 0.0, 1000.0)
                                
                                if imgui.button("Remove Atmosphere"):
                                    if 'lut_tex' in atmo_item:
                                        atmo_item['lut_tex'].release()
                                    atmo_bodies.remove(atmo_item)
                            else:
                                if imgui.button("Add Atmosphere"):
                                    atmo_bodies.append({
                                        'body_idx': insp_idx,
                                        'planet_radius_km': body_r_km,
                                        'surface_radius_au': body_r_km / 1.496e8,
                                        'atmo_radius_km': body_r_km * 1.025,
                                        'atmo_radius_au': (body_r_km * 1.025) / 1.496e8,
                                        'beta_rayleigh': np.array([5.5e-6, 13.0e-6, 22.4e-6], dtype='f4'),
                                        'h_rayleigh': 8.0,
                                        'beta_mie': 21.0e-6,
                                        'h_mie': 1.2,
                                        'mie_g': 0.758,
                                        'beta_absorption': np.array([0,0,0], dtype='f4'),
                                        'intensity': 1.0
                                    })
                                    
                        if imgui.collapsing_header("Rings")[0]:
                            ring_group = next((g for g in ring_render_groups if g['body_idx'] == insp_idx), None)
                            rings = [r for r in ring_precomputed if r['body_idx'] == insp_idx]
                            
                            for i, ring_item in enumerate(rings):
                                if imgui.tree_node(f"Ring Layer {i}"):
                                    changed_in, new_in = imgui.drag_float(f"Inner Radius (km)##{i}", ring_item['inner_r'] * 1.496e8, 10.0, body_r_km, ring_item['outer_r'] * 1.496e8 - 10)
                                    changed_out, new_out = imgui.drag_float(f"Outer Radius (km)##{i}", ring_item['outer_r'] * 1.496e8, 10.0, ring_item['inner_r'] * 1.496e8 + 10, body_r_km * 50.0)
                                    changed_col, new_col = imgui.color_edit3(f"Color##{i}", *ring_item['raw_color'])
                                    changed_op, new_op = imgui.slider_float(f"Opacity##{i}", ring_item['opacity'], 0.0, 1.0)
                                    changed_scat, new_scat = imgui.drag_float(f"Forward Scatter Mult##{i}", ring_item['scatter'], 0.05, 0.0, 100.0)
                                    changed_asym, new_asym = imgui.slider_float(f"Forward Scatter Asym##{i}", ring_item['asymmetry'], 0.0, 0.999)
                                    
                                    grad_changed = False
                                    if imgui.tree_node(f"Alpha Gradient##{i}"):
                                        grad = ring_item['gradient']
                                        if imgui.button(f"Add Stop##{i}"):
                                            grad.append({'p': 1.0, 'a': 1.0})
                                        
                                        stops_to_remove = []
                                        for j, stop in enumerate(grad):
                                            imgui.push_item_width(100)
                                            changed_p, n_p = imgui.slider_float(f"Pos##{i}_{j}", stop['p'], 0.0, 1.0)
                                            imgui.same_line()
                                            changed_a, n_a = imgui.slider_float(f"Alpha##{i}_{j}", stop['a'], 0.0, 1.0)
                                            imgui.same_line()
                                            if imgui.button(f"X##{i}_{j}"):
                                                stops_to_remove.append(j)
                                            imgui.pop_item_width()
                                            
                                            if changed_p or changed_a:
                                                stop['p'] = n_p
                                                stop['a'] = n_a
                                                grad_changed = True
                                                
                                        for j in reversed(stops_to_remove):
                                            grad.pop(j)
                                            grad_changed = True
                                            
                                        if grad_changed:
                                            grad.sort(key=lambda x: x['p'])
                                            
                                        imgui.tree_pop()
                                    
                                    if changed_in or changed_out or changed_col or changed_op or changed_scat or changed_asym or grad_changed:
                                        if changed_in: ring_item['inner_r'] = new_in / 1.496e8
                                        if changed_out: ring_item['outer_r'] = new_out / 1.496e8
                                        if changed_col: ring_item['raw_color'] = new_col
                                        if changed_op: ring_item['opacity'] = new_op
                                        if changed_scat: ring_item['scatter'] = new_scat
                                        if changed_asym: ring_item['asymmetry'] = new_asym
                                        
                                        pole_render = visual_arr[insp_idx, 5:8]
                                        verts, indices, norm, colors, shadow_grad = generate_ring_arrays(
                                            pole_render, ring_item['inner_r'], ring_item['outer_r'],
                                            ring_item['raw_color'], ring_item['opacity'], ring_item['scatter'], ring_item['asymmetry'], ring_item['gradient']
                                        )
                                        ring_item['verts'] = verts
                                        ring_item['indices'] = indices
                                        ring_item['colors'] = colors
                                        ring_item['shadow_grad'] = shadow_grad
                                        rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                                    
                                    if imgui.button(f"Remove Layer##{i}"):
                                        ring_precomputed[:] = [r for r in ring_precomputed if r is not ring_item]
                                        rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                                    
                                    imgui.tree_pop()
                                    
                            if imgui.button("Add Ring Layer"):
                                if len(rings) < 16:
                                    r_in = body_r_km * 1.2 / 1.496e8
                                    if rings:
                                        r_in = rings[-1]['outer_r'] + (100 / 1.496e8)
                                    r_out = r_in + (body_r_km * 0.5 / 1.496e8)
                                    pole_render = visual_arr[insp_idx, 5:8]
                                    color = visual_arr[insp_idx, 0:3]
                                    grad = [{'p': 0.0, 'a': 0.0}, {'p': 0.5, 'a': 1.0}, {'p': 1.0, 'a': 0.0}]
                                    verts, indices, norm, colors, shadow_grad = generate_ring_arrays(
                                        pole_render, r_in, r_out, color, 1.0, 2.5, 0.8, grad
                                    )
                                    ring_precomputed.append({
                                        'body_idx': insp_idx,
                                        'verts': verts, 'indices': indices, 'normal': norm, 'colors': colors,
                                        'inner_r': r_in, 'outer_r': r_out, 'opacity': 1.0,
                                        'scatter': 2.5, 'asymmetry': 0.8, 'shadow_grad': shadow_grad,
                                        'raw_color': color, 'gradient': grad
                                    })
                                    rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
                    
                imgui.end()
    
            if self.camera.get("add_mode", False):
                imgui.set_next_window_size(350, 400, imgui.FIRST_USE_EVER)
                imgui.set_next_window_position(600, 50, imgui.FIRST_USE_EVER)
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
                        real_a = ad["a"] / 1.496e8 if is_moon else ad["a"]
                        
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
                    
                    imgui.spacing()
                    imgui.text_colored("The Big Three", 0.4, 1.0, 0.7)
                    _, cd["mass"] = imgui.input_double(u"Mass (M\u2609)", cd["mass"], format="%.4f")
                    _, cd["metallicity"] = imgui.input_double("[Fe/H] (dex)", cd["metallicity"], format="%.3f")
                    _, cd["age"] = imgui.input_double("Age (Gyr)", cd["age"], format="%.3f")
                    
                    # Live preview of derived properties
                    imgui.separator()
                    imgui.text_colored("Derived Properties (Preview)", 0.7, 0.7, 0.7)
                    preview = derive_star_properties(cd["mass"], cd["metallicity"], cd["age"])
                    pr, pg, pb = temperature_to_rgb(preview["temperature"])
                    
                    imgui.text(f"  Temperature:    {preview['temperature']:,.0f} K")
                    imgui.text(f"  Luminosity:     {preview['luminosity']:.4f} L\u2609")
                    imgui.text(f"  Radius:         {preview['radius']:.4f} R\u2609")
                    imgui.text(f"  Spectral Class: {preview['spectral_class']}")
                    imgui.text(f"  Stage:          {preview['stage']}")
                    imgui.text(f"  MS Lifetime:    {preview['ms_lifetime']:.2f} Gyr")
                    imgui.text(f"  MS Progress:    {preview['progress']:.1f}%")
                    
                    # Color preview
                    imgui.spacing()
                    imgui.text("Star Color:")
                    imgui.same_line()
                    imgui.color_button("##star_color_preview", pr, pg, pb, 1.0, 0, 20, 20)
                    
                    imgui.separator()
                    
                    # Validation
                    name_ok = len(cd["system_name"].strip()) > 0
                    name_exists = sys_mgr.system_exists(cd["system_name"].strip())
                    mass_ok = cd["mass"] > 0.01
                    age_ok = cd["age"] >= 0
                    
                    if not name_ok:
                        imgui.text_colored("System name required", 1.0, 0.3, 0.3)
                    elif name_exists:
                        imgui.text_colored("System already exists!", 1.0, 0.3, 0.3)
                    elif not mass_ok:
                        imgui.text_colored(u"Mass must be > 0.01 M\u2609", 1.0, 0.3, 0.3)
                    elif not age_ok:
                        imgui.text_colored("Age must be >= 0", 1.0, 0.3, 0.3)
                    
                    can_create = name_ok and not name_exists and mass_ok and age_ok
                    
                    if can_create:
                        if imgui.button("Create System", width=-1):
                            sys_name = cd["system_name"].strip()
                            star_name_c = cd["star_name"].strip() or "Star"
                            
                            # Create system on disk
                            new_bodies = sys_mgr.create_new_system(
                                sys_name, star_name_c, cd["mass"], cd["metallicity"], cd["age"]
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
    
            # Re-enable standard settings for ImGui
            ctx.screen.use()
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            imgui.render()
     
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
            
            self._last_fb_width = self.fb_width
            self._last_fb_height = self.fb_height
            
        running[0] = False
        running_cmp[0] = False
        physics_thread.join(timeout=1.0)
        physics_thread_cmp.join(timeout=1.0)
        self.impl.shutdown()
        glfw.terminate()
    
