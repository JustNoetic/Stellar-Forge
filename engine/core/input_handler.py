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
            r_target = 0.0
            if self.camera["tracking_idx"] is not None:
                track_idx = self.camera["tracking_idx"]
                is_cmp = self.camera.get("tracking_is_cmp", False)
                if is_cmp and hasattr(self, "body_radii_cmp") and self.body_radii_cmp is not None:
                    if track_idx < len(self.body_radii_cmp):
                        r_target = float(self.body_radii_cmp[track_idx])
                elif hasattr(self, "body_radii") and self.body_radii is not None:
                    if track_idx < len(self.body_radii):
                        r_target = float(self.body_radii[track_idx])
            
            h = max(1e-11, self.camera["distance"] - r_target)
            zoom_speed = max(1e-10, h * 0.2)
            
            if yoffset > 0:
                self.camera["distance"] -= zoom_speed
            elif yoffset < 0:
                self.camera["distance"] += zoom_speed
                
            min_dist = r_target + 2e-7
            self.camera["distance"] = max(min_dist, self.camera["distance"])
    
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
        
        roll_rad = math.radians(self.camera.get("roll_actual", 0.0))
        cos_r = math.cos(roll_rad)
        sin_r = math.sin(roll_rad)
        
        dx_eff = dx * cos_r + dy * sin_r
        dy_eff = -dx * sin_r + dy * cos_r
        
        fov_ratio = max(0.0001, min(1.0, self.camera.get("fov", 45.0) / 45.0))
        
        if self.camera["left_dragging"]:
            sensitivity = 0.3 * fov_ratio
            self.camera["yaw"] += dx_eff * sensitivity
            self.camera["pitch"] -= dy_eff * sensitivity
            self.camera["pitch"] = max(-89.9, min(89.9, self.camera["pitch"])) 
            
        elif self.camera["right_dragging"]:
            pan_speed = self.camera["distance_actual"] * 0.001 * fov_ratio
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
            
            pan_vec = -right * dx_eff * pan_speed + up * dy_eff * pan_speed
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
