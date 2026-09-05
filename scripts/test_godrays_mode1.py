import os, sys, time, threading
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
import glfw
import engine.app as ap_mod

app = ap_mod.App()
app._perf_target_body = 'Saturn'
app.camera['atmo_quality'] = 2
app.camera['atmo_shadow_mode'] = 1
app.camera['atmo_temporal_accum'] = True

def closer():
    for i in range(15):
        time.sleep(0.3)
        print('Frame:', getattr(app, 'frame_counter', 0), flush=True)
    if hasattr(app, 'window') and app.window:
        glfw.set_window_should_close(app.window, True)

threading.Thread(target=closer, daemon=True).start()
try:
    app.run()
    print('App exited cleanly!', flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
