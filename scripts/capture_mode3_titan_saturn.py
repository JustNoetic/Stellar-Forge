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
    print('=== Starting Mode 3 Titan & Saturn Live Verification Capture ===', flush=True)
    app = ap_mod.App()
    app._perf_target_body = 'Titan'
    app.camera['atmo_quality'] = 3
    app.camera['atmo_temporal_accum'] = True
    app.camera['atmo_stochastic'] = True
    app.camera['atmo_enabled'] = True

    artifact_dir = r'C:\Users\Noetic\.gemini\antigravity\brain\b8c17976-5c49-4e0e-b603-c758c03a7e0e'

    def controller():
        try:
            print('[Controller] Waiting for render loop to begin...', flush=True)
            while getattr(app, 'frame_counter', 0) < 40:
                time.sleep(0.1)

            # Find Titan and Saturn indices
            titan_idx = None
            saturn_idx = None
            for i, b in enumerate(app.bodies_data):
                if b.get('name') == 'Titan':
                    titan_idx = i
                elif b.get('name') == 'Saturn':
                    saturn_idx = i

            print(f'[Controller] Indices: Titan={titan_idx}, Saturn={saturn_idx}', flush=True)
            assert titan_idx is not None, "Titan not found!"
            assert saturn_idx is not None, "Saturn not found!"

            AU_TO_KM = 149597870.7

            # -------------------------------------------------------------
            # Shot 1: Titan at 3.8 Million KM in Mode 3
            # -------------------------------------------------------------
            print('\n[Controller] Setting up Shot 1: Titan at 3.8 Million KM (Mode 3)...', flush=True)
            app.camera['tracking_idx'] = titan_idx
            app.camera['tracking_is_cmp'] = False
            app.camera['tracking_mode'] = 'body'
            app.camera['cam_look'] = 'aim'
            app.camera['fov'] = 0.5  # Narrow FOV to frame Titan at 3.8M km

            dist_3_8M_au = 3800000.0 / AU_TO_KM
            app.camera['cam_pos_rel'] = np.array([0.0, 0.0, dist_3_8M_au], dtype='f8')
            app.camera['cam_pos_rel_prev'] = app.camera['cam_pos_rel'].copy()
            app.camera['atmo_quality'] = 3

            # Settle frames
            start_f = app.frame_counter
            while app.frame_counter < start_f + 50:
                time.sleep(0.05)

            before = set(glob.glob('screenshots/*.png'))
            app._screenshot_request = (1280, 720)
            while app._screenshot_request is not None or app._screenshot_capturing or app._screenshot_saving:
                time.sleep(0.05)
            time.sleep(0.5)
            after = set(glob.glob('screenshots/*.png'))
            new_files = list(after - before)
            if new_files:
                shot1 = new_files[0]
                print(f'[Controller] Shot 1 captured: {shot1}', flush=True)
                dest1 = os.path.join(artifact_dir, 'titan_mode3_3_8M.png')
                shutil.copyfile(shot1, dest1)
                print(f'[Controller] Copied to {dest1}', flush=True)

            # -------------------------------------------------------------
            # Shot 2: Titan BEHIND Saturn's Rings & Atmosphere
            # -------------------------------------------------------------
            print('\n[Controller] Setting up Shot 2: Titan BEHIND Saturn & Rings...', flush=True)
            # Track Saturn, position camera looking towards Saturn with Titan in background
            app.camera['tracking_idx'] = saturn_idx
            app.camera['fov'] = 25.0
            
            # Position camera in front of Saturn, viewing Saturn's ring plane with Titan behind
            # Saturn radius ~ 60,000 km = 0.0004 AU. Rings reach ~ 140,000 km = 0.00093 AU.
            # Titan is at ~ 1.22M km = 0.00816 AU.
            # Get current relative position of Titan to Saturn
            pos_saturn = app.pos_snap_render[saturn_idx]
            pos_titan = app.pos_snap_render[titan_idx]
            titan_to_saturn = pos_titan - pos_saturn
            dist_t_s = np.linalg.norm(titan_to_saturn)
            dir_t_s = titan_to_saturn / max(dist_t_s, 1e-6)
            print(f'[Controller] Vector from Saturn to Titan: dist={dist_t_s * AU_TO_KM:.0f} km', flush=True)

            # To see Titan BEHIND Saturn, place camera on the opposite side of Saturn:
            # cam_pos = -dir_t_s * (saturn_ring_distance)
            sat_cam_dist = 0.003  # ~450,000 km from Saturn
            app.camera['cam_pos_rel'] = -dir_t_s * sat_cam_dist + np.array([0.0, 0.0004, 0.0], dtype='f8')
            app.camera['cam_pos_rel_prev'] = app.camera['cam_pos_rel'].copy()
            app.camera['cam_look'] = 'aim'

            # Settle frames
            start_f = app.frame_counter
            while app.frame_counter < start_f + 50:
                time.sleep(0.05)

            before = set(glob.glob('screenshots/*.png'))
            app._screenshot_request = (1280, 720)
            while app._screenshot_request is not None or app._screenshot_capturing or app._screenshot_saving:
                time.sleep(0.05)
            time.sleep(0.5)
            after = set(glob.glob('screenshots/*.png'))
            new_files = list(after - before)
            if new_files:
                shot2 = new_files[0]
                print(f'[Controller] Shot 2 captured: {shot2}', flush=True)
                dest2 = os.path.join(artifact_dir, 'titan_behind_saturn_mode3.png')
                shutil.copyfile(shot2, dest2)
                print(f'[Controller] Copied to {dest2}', flush=True)

            # -------------------------------------------------------------
            # Shot 3: Titan IN FRONT OF Saturn's Rings & Atmosphere
            # -------------------------------------------------------------
            print('\n[Controller] Setting up Shot 3: Titan IN FRONT OF Saturn & Rings...', flush=True)
            # Place camera behind Titan looking towards Saturn:
            # cam_pos = dir_t_s * (dist_t_s + 0.001)
            app.camera['cam_pos_rel'] = dir_t_s * (dist_t_s + 0.001) + np.array([0.0, 0.0004, 0.0], dtype='f8')
            app.camera['cam_pos_rel_prev'] = app.camera['cam_pos_rel'].copy()
            app.camera['cam_look'] = 'aim'

            start_f = app.frame_counter
            while app.frame_counter < start_f + 50:
                time.sleep(0.05)

            before = set(glob.glob('screenshots/*.png'))
            app._screenshot_request = (1280, 720)
            while app._screenshot_request is not None or app._screenshot_capturing or app._screenshot_saving:
                time.sleep(0.05)
            time.sleep(0.5)
            after = set(glob.glob('screenshots/*.png'))
            new_files = list(after - before)
            if new_files:
                shot3 = new_files[0]
                print(f'[Controller] Shot 3 captured: {shot3}', flush=True)
                dest3 = os.path.join(artifact_dir, 'titan_in_front_saturn_mode3.png')
                shutil.copyfile(shot3, dest3)
                print(f'[Controller] Copied to {dest3}', flush=True)

            print('\n=== All verification captures completed successfully! ===', flush=True)

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
