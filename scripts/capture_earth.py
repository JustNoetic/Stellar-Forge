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

def main():
    target_body = "Earth"
    out_file = sys.argv[1] if len(sys.argv) > 1 else "screenshots/earth_before.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    # Hidden GLFW window
    orig_create_window = glfw.create_window
    def patched_create_window(width, height, title, monitor, share):
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        return orig_create_window(width, height, title, monitor, share)
    glfw.create_window = patched_create_window
    
    app_instance = ap_mod.App()
    app_instance._perf_target_body = target_body
    app_instance.time_ctrl["paused"] = True
    app_instance.camera["atmo_quality"] = 3 # Analytical Sky-View LUT
    app_instance.camera["show_orbits"] = False
    app_instance.camera["show_gaia_stars"] = False
    app_instance.ui_visible = False
    
    # We want to exit after 15 frames and save image
    frame_count = 0
    orig_swap = glfw.swap_buffers
    def patched_swap(window):
        nonlocal frame_count
        frame_count += 1
        if frame_count == 12:
            # Capture the resolved HDR or composite
            ctx = app_instance.ctx
            fbo = app_instance.hdr_resolve_fbo
            if fbo:
                raw = fbo.read(components=3)
                w, h = fbo.width, fbo.height
                # Note: fbo is float/linear HDR or tone mapped? Wait, composite pass renders to screen / default framebuffer!
                # Let's read from default framebuffer or fbo
            raw_screen = gl_read_screen(app_instance.fb_width, app_instance.fb_height)
            img = Image.frombytes('RGB', (app_instance.fb_width, app_instance.fb_height), raw_screen)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(out_file)
            print(f"[Capture] Saved {out_file} ({app_instance.fb_width}x{app_instance.fb_height})")
            glfw.set_window_should_close(window, True)
        orig_swap(window)
        
    def gl_read_screen(w, h):
        import OpenGL.GL as gl
        gl.glReadBuffer(gl.GL_BACK)
        return gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    glfw.swap_buffers = patched_swap
    
    print(f"[Capture] Launching app to capture {target_body}...")
    app_instance.run()
    print("[Capture] Complete.")

if __name__ == "__main__":
    main()
