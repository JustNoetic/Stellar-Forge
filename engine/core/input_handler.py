import math
import os
import json
import numpy as np
import glfw
import imgui

class InputHandlerMixin:
    """Mixin class for GLFW input callbacks and graphics settings persistence."""

    def load_settings(self):
        try:
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
            os.makedirs("data", exist_ok=True)
            settings_path = os.path.join("data", "graphics_settings.json")
            saved = {
                "atmo_quality": self.camera.get("atmo_quality", 1),
                "atmo_steps_max": self.camera.get("atmo_steps_max", 32),
                "atmo_adaptive_steps": self.camera.get("atmo_adaptive_steps", True),
                "atmo_jitter": self.camera.get("atmo_jitter", True),
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
                "shadow_caster_budget": self.camera.get("shadow_caster_budget", 32),
                "taa_enabled": self.camera.get("taa_enabled", True),
                "atmo_render_scale": self.camera.get("atmo_render_scale", 0.5),
                "screenshot_res_idx": self.camera.get("screenshot_res_idx", 1),
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
            # Scroll wheel changes flight velocity (Space Engine style, ~2x per notch).
            speed = self.camera.get("flight_speed", 0.1)
            speed *= (2.0 ** yoffset)
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
            self.camera["approach_delta"] += dy * 0.0125
        elif self.camera["left_dragging"]:
            # LMB: trackball pivot in place (free look), direct (no smoothing).
            sensitivity = 0.3 * fov_ratio
            _camera_pivot_apply(self.camera, dx, dy, sensitivity)
            self.camera["cam_look"] = "free"
        elif self.camera["right_dragging"]:
            # RMB: trackball orbit around the pivot, direct (no smoothing).
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
            elif key == glfw.KEY_F12 and action == glfw.PRESS:
                if not self._screenshot_capturing and not self._screenshot_saving:
                    _ss_presets = [(3840, 2160), (7680, 4320), (15360, 8640)]
                    _ss_idx = max(0, min(len(_ss_presets) - 1, self.camera.get("screenshot_res_idx", 1)))
                    self._screenshot_request = _ss_presets[_ss_idx]
    
    def resize_callback(self, window, width, height):
        if self.impl: self.impl.resize_callback(window, width, height)
        self.window_width, self.window_height = max(1, width), max(1, height)
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(window)

def _camera_align_up(up_world):
    """Return the 3x3 rotation R that maps the local +Y axis onto unit `up_world`."""
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

def _camera_screen_up(fwd, roll_rad):
    """Screen-space up of the current view: world +Y projected onto the plane
    perpendicular to `fwd`, then rotated by roll about the view axis."""
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
        # looking straight up/down: pick a stable perpendicular instead
        up = np.array([1.0, 0.0, 0.0], dtype='f8')
        up = up - float(np.dot(fwd, up)) * fwd
        nup = np.linalg.norm(up)
        up = up / nup if nup > 1e-9 else np.array([0.0, 0.0, 1.0], dtype='f8')
    if abs(roll_rad) > 1e-9:
        up = _camera_rot_axis(fwd, -roll_rad) @ up
    return up

def _camera_orient_from_view(fwd, roll_rad):
    """Orientation matrix (columns: right, up, back) matching the rendered view
    (row-vector convention): look_at(pos, pos+fwd, Y) followed by Rz(roll)."""
    up = _camera_screen_up(fwd, roll_rad)
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

def _camera_pivot_apply(cam, dx, dy, sensitivity):
    """LMB trackball pivot: track a full orientation matrix so the image rotates
    rigidly with the cursor at any pitch/roll (roll-aware, no pole spinning)."""
    if cam.get("cam_look", "aim") == "aim":
        rel = np.asarray(cam["cam_pos_rel"], dtype='f8')
        r = np.linalg.norm(rel)
        fwd = -rel / r if r > 1e-300 else _camera_forward(cam.get("yaw_actual", cam["yaw"]), cam.get("pitch_actual", cam["pitch"]))
    else:
        fwd = _camera_forward(cam.get("yaw_actual", cam["yaw"]), cam.get("pitch_actual", cam["pitch"]))
    roll_rad = math.radians(cam.get("roll_actual", 0.0))
    R = _camera_orient_from_view(fwd, roll_rad)
    R = _camera_rot_axis(R[:, 1], math.radians(-dx * sensitivity)) @ (_camera_rot_axis(R[:, 0], math.radians(-dy * sensitivity)) @ R)
    yaw_new, pitch_new, roll_new = _camera_euler_from_orient(R)
    cam["yaw"] = cam["yaw_actual"] = yaw_new
    cam["pitch"] = cam["pitch_actual"] = pitch_new
    cam["roll"] = cam["roll_actual"] = roll_new

def _camera_orbit_apply(cam, dx, dy, sensitivity):
    """RMB trackball orbit: rotate the camera position about the pivot using the
    camera's own screen axes (radius preserved, no smoothing)."""
    rel = np.asarray(cam["cam_pos_rel"], dtype='f8')
    r = np.linalg.norm(rel)
    if r < 1e-300:
        return
    if cam.get("cam_look", "aim") == "aim":
        fwd = -rel / r
    else:
        fwd = _camera_forward(cam.get("yaw_actual", cam["yaw"]), cam.get("pitch_actual", cam["pitch"]))
    roll_rad = math.radians(cam.get("roll_actual", 0.0))
    up = _camera_screen_up(fwd, roll_rad)
    right = np.cross(fwd, up)
    nr = np.linalg.norm(right)
    if nr > 1e-9:
        right = right / nr
    else:
        right = np.array([1.0, 0.0, 0.0], dtype='f8')
    rel_new = _camera_rot_axis(up, math.radians(-dx * sensitivity)) @ (_camera_rot_axis(right, math.radians(-dy * sensitivity)) @ rel)
    r_new = np.linalg.norm(rel_new)
    if r_new > 1e-300:
        rel_new = rel_new / r_new * r
    cam["cam_pos_rel"] = rel_new.tolist()
    cam["cam_look"] = "aim"
