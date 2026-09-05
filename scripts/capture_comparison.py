import os
import sys
import time
import threading
import shutil
import glob
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

import glfw
import engine.app as ap_mod

def main():
    print('=== Starting Saturn Comparison Capture ===', flush=True)
    app = ap_mod.App()
    app._perf_target_body = 'Saturn'
    app.camera['atmo_quality'] = 2
    app.camera['atmo_temporal_accum'] = True
    app.camera['atmo_stochastic'] = True
    app.camera['atmo_shadow_mode'] = 0

    artifact_dir = r'C:\Users\Noetic\.gemini\antigravity\brain\599e529a-9795-4fa5-8788-b660fd0a44df'

    def controller():
        try:
            print('[Controller] Waiting for render loop to begin...', flush=True)
            while getattr(app, 'frame_counter', 0) < 30:
                time.sleep(0.1)
            
            # Position camera over the northern hemisphere where ring shadows fall across the atmosphere
            sat_rad = 3.896022e-4
            cam_dir = np.array([-0.79848832, 0.54205035, -0.26191186], dtype='f8')
            app.camera['cam_pos_rel'] = cam_dir * (sat_rad * 2.4)
            app.camera['cam_look'] = 'aim'
            
            print('[Controller] Positioned camera at ring shadow view.', flush=True)
            time.sleep(1.0)
            
            # Settle Mode 0 (Unified)
            start_f = app.frame_counter
            while app.frame_counter < start_f + 40:
                time.sleep(0.05)
            
            print('[Controller] Settled Mode 0 (Unified)... Frame:', app.frame_counter, flush=True)
            
            # Request shot 1
            before = set(glob.glob('screenshots/*.png'))
            app._screenshot_request = (1280, 720)
            print('[Controller] Shot 1 (Unified) requested', flush=True)
            
            while app._screenshot_request is not None or app._screenshot_capturing or app._screenshot_saving:
                time.sleep(0.05)
            
            time.sleep(0.5)
            after = set(glob.glob('screenshots/*.png'))
            new_files = list(after - before)
            if new_files:
                unified_shot = new_files[0]
                print(f'[Controller] Mode 0 shot captured: {unified_shot}', flush=True)
                if os.path.exists(artifact_dir):
                    shutil.copyfile(unified_shot, os.path.join(artifact_dir, 'saturn_unified.png'))
                    print('[Controller] Copied saturn_unified.png to artifacts', flush=True)
            
            # Switch to Mode 1 (Separated Godrays)
            print('\n[Controller] Switching to Mode 1 (Separated Godrays)...', flush=True)
            app.camera['atmo_shadow_mode'] = 1
            
            start_f = app.frame_counter
            while app.frame_counter < start_f + 40:
                time.sleep(0.05)
            
            print('[Controller] Settled Mode 1. Frame:', app.frame_counter, flush=True)
            time.sleep(1.0)
            
            before = set(glob.glob('screenshots/*.png'))
            app._screenshot_request = (1280, 720)
            print('[Controller] Shot 2 (Godrays) requested', flush=True)
            
            while app._screenshot_request is not None or app._screenshot_capturing or app._screenshot_saving:
                time.sleep(0.05)
                
            time.sleep(0.5)
            after = set(glob.glob('screenshots/*.png'))
            new_files = list(after - before)
            if new_files:
                godrays_shot = new_files[0]
                print(f'[Controller] Mode 1 shot captured: {godrays_shot}', flush=True)
                if os.path.exists(artifact_dir):
                    shutil.copyfile(godrays_shot, os.path.join(artifact_dir, 'saturn_godrays.png'))
                    print('[Controller] Copied saturn_godrays.png to artifacts', flush=True)

            print('\n=== All captures completed successfully! ===', flush=True)
        except Exception as e:
            print('[Controller] Error:', e, flush=True)
            import traceback
            traceback.print_exc()
        finally:
            time.sleep(0.5)
            if hasattr(app, 'window') and app.window:
                glfw.set_window_should_close(app.window, True)

    threading.Thread(target=controller, daemon=True).start()
    app.run()
    print('[Main] Finished cleanly.', flush=True)

if __name__ == '__main__':
    main()
