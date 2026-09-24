import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../engine')))

import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ARRAY_SIZE_CHECKING = False

import engine.app as ap_mod
import glfw
import numpy as np
from PIL import Image

def capture_view(view_mode="orbit", out_file="screenshots/earth_view.png"):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    orig_create_window = glfw.create_window
    def patched_create_window(width, height, title, monitor, share):
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        return orig_create_window(width, height, title, monitor, share)
    glfw.create_window = patched_create_window
    
    app_instance = ap_mod.App()
    app_instance._perf_target_body = "Earth"
    app_instance.time_ctrl["paused"] = True
    app_instance.camera["atmo_quality"] = 3
    app_instance.camera["show_orbits"] = False
    app_instance.camera["show_gaia_stars"] = False
    app_instance.ui_visible = False
    
    frame_count = 0
    orig_swap = glfw.swap_buffers
    def patched_swap(window):
        nonlocal frame_count
        frame_count += 1
        
        # After a few frames, adjust camera if ground view requested
        if view_mode == "ground" and frame_count >= 5:
            # Place camera just 20 meters above Earth surface (altitude ~0.02 km)
            # Earth radius ~ 4.263e-5 AU. 1 km ~ 6.684587e-9 AU
            earth_idx = app_instance.camera["tracking_idx"]
            r_earth = app_instance.body_radii[earth_idx]
            # Position camera at altitude 1 km on day side, looking UP at zenith (sky)
            up_dir = np.array([0.0, -1.0, 0.0], dtype='f8') # lit face
            cam_pos = up_dir * (r_earth + 5.0 / 149597870.7) # 5 km altitude, right under the 6.5 km clouds!
            app_instance.camera["cam_pos_rel"] = cam_pos
            app_instance.camera["cam_look"] = "free"
            app_instance.camera["pitch"] = 80.0 # looking almost straight up at zenith
            app_instance.camera["yaw"] = 0.0
            
        if frame_count == 14:
            import OpenGL.GL as gl
            gl.glReadBuffer(gl.GL_BACK)
            raw = gl.glReadPixels(0, 0, app_instance.fb_width, app_instance.fb_height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            img = Image.frombytes('RGB', (app_instance.fb_width, app_instance.fb_height), raw)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(out_file)
            print(f"[Capture] Saved {out_file}")
            glfw.set_window_should_close(window, True)
            
        orig_swap(window)
        
    glfw.swap_buffers = patched_swap
    app_instance.run()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "orbit"
    out = sys.argv[2] if len(sys.argv) > 2 else f"screenshots/earth_{mode}.png"
    capture_view(mode, out)
