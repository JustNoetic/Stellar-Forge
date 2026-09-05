import os
import sys
import time
import threading
import shutil
import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'engine'))

import glfw
import engine.app as ap_mod

def main():
    print('=== Stellar-Forge Godrays Verification Script ===')
    app_instance = ap_mod.App()
    app_instance._perf_target_body = 'Saturn'
    app_instance.camera['atmo_quality'] = 2
    app_instance.camera['atmo_temporal_accum'] = True
    app_instance.camera['atmo_stochastic'] = True

    captured_files = {}

    def _wait_frames(target):
        deadline = time.time() + 30
        while time.time() < deadline and getattr(app_instance, 'frame_counter', 0) < target:
            time.sleep(0.05)

    def _take_screenshot(tag):
        while getattr(app_instance, '_screenshot_capturing', False) or getattr(app_instance, '_screenshot_saving', False) or getattr(app_instance, '_screenshot_request', None) is not None:
            time.sleep(0.05)
        
        before_files = set(glob.glob('screenshots/*.png'))
        app_instance._screenshot_request = (1280, 720)
        print(f'[{tag}] Requested screenshot 1280x720...')

        deadline = time.time() + 20
        while time.time() < deadline and (app_instance._screenshot_request is not None or getattr(app_instance, '_screenshot_capturing', False) or getattr(app_instance, '_screenshot_saving', False)):
            time.sleep(0.05)
        
        after_files = set(glob.glob('screenshots/*.png'))
        new_files = list(after_files - before_files)
        if new_files:
            new_file = new_files[0]
            captured_files[tag] = new_file
            print(f'[{tag}] Successfully captured: {new_file}')
        else:
            print(f'[{tag}] Warning: No new screenshot file detected.')

    def orchestrator():
        try:
            print('[Orchestrator] Waiting for app startup...')
            _wait_frames(40)
            print('[Orchestrator] Settled initial scene.')

            # Test 1: Unified Single-Pass
            print('\n--- Testing Mode 0: Unified Single-Pass ---')
            app_instance.camera['atmo_shadow_mode'] = 0
            start_f = getattr(app_instance, 'frame_counter', 0)
            _wait_frames(start_f + 30)
            _take_screenshot('unified')

            # Test 2: Separated Godrays (KSA Style)
            print('\n--- Testing Mode 1: Separated Godrays (KSA Style) ---')
            app_instance.camera['atmo_shadow_mode'] = 1
            start_f = getattr(app_instance, 'frame_counter', 0)
            _wait_frames(start_f + 30)
            _take_screenshot('godrays')

            time.sleep(1.0)
            print('\n[Orchestrator] Both tests completed successfully!')
        except Exception as e:
            print(f'[Orchestrator] Error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            if hasattr(app_instance, 'window') and app_instance.window:
                glfw.set_window_should_close(app_instance.window, True)

    t = threading.Thread(target=orchestrator, daemon=True)
    t.start()

    print('[Main] Starting app loop...')
    try:
        app_instance.run()
    except Exception as e:
        print(f'[Main] App run exception: {e}')

    print('[Main] App finished. Captured files:', captured_files)
    
    artifact_dir = r'C:\Users\Noetic\.gemini\antigravity\brain\599e529a-9795-4fa5-8788-b660fd0a44df'
    if os.path.exists(artifact_dir):
        for tag, src in captured_files.items():
            dst = os.path.join(artifact_dir, f'saturn_{tag}.png')
            shutil.copyfile(src, dst)
            print(f'Copied {src} -> {dst}')

if __name__ == '__main__':
    main()
