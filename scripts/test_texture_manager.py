"""
scripts/test_texture_manager.py

Unit and regression test suite for Texture Management:
- Image import and spherical mean color extraction
- Multi-layer support (diffuse, normal, specular, clouds)
- Heightmap to tangent-space normal map conversion
- Cloud transparency extraction
- Staging and discard lifecycle
- Hot reloading into ModernGL Texture and SSBO handle writing (diffuse=0, normal=1, specular=2, clouds=3)
- Destination folder layout: textures/<system name>/<planet name>/<planet name>[suffix].<ext>
- Quadtree tile pyramid baking integration
- Inspector cosmetic saving directly to master data/system.json for Solar System
"""

import os
import sys
import shutil
import json
import numpy as np
from PIL import Image
import moderngl

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rendering.texture_manager import (
    compute_texture_spherical_mean,
    stage_imported_texture,
    discard_staged_texture,
    hot_reload_body_texture,
    start_bake_async,
    apply_staged_texture,
    bake_and_apply_staged_texture,
    heightmap_to_normal_map,
    prepare_layer_image,
    is_grayscale_image,
    TEXTURE_LAYERS
)
from engine.rendering.planetshine import _build_tex_idx_arr
from engine.ui.inspector import _save_system_cosmetics
from engine.ephemeris.system_manager import SystemManager
from engine.path_utils import get_external_path

class DummySysMgr:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.systems_dir = os.path.join(root_dir, "data", "systems")
        self.default_json = os.path.join(root_dir, "data", "system.json")

    def _system_dir(self, name):
        return os.path.join(self.systems_dir, name)

    def _system_json_path(self, name):
        return os.path.join(self._system_dir(name), "system.json")

    def save_meta(self, name, sys_data):
        pass

class DummyApp:
    def __init__(self, ctx):
        self.ctx = ctx
        self.texture_slices = {"earth": 1, "mars": 2}
        self.texture_mean_colors = {}
        self.gpu_body_textures = {}
        self.active_res_level = {}
        self.body_textures_ssbo_data = np.zeros((32, 4, 2), dtype=np.uint32)
        self.body_textures_ssbo = ctx.buffer(self.body_textures_ssbo_data.tobytes())
        self.bodies_data = [{"name": "Earth"}, {"name": "Mars"}, {"name": "TestPlanet"}]
        self.tex_idx_arr = _build_tex_idx_arr(self.bodies_data, self.texture_slices)
        self._texture_staging = {}
        self._texture_bake_state = {}
        self._screenshot_toast = None
        self.sys_mgr = DummySysMgr(ROOT_DIR)
        self.impl = None

def main():
    print("==================================================")
    print("Running Texture Manager Unit Tests...")
    print("==================================================")

    # 1. Test Spherical Mean Color Extraction
    test_img = Image.new("RGBA", (256, 128), (200, 100, 50, 255))
    mean_col = compute_texture_spherical_mean(test_img)
    print(f"[Test 1] Spherical mean color extracted: {mean_col}")
    assert len(mean_col) == 3, "Mean color must be 3-component linear RGB"
    assert mean_col[0] > mean_col[1] > mean_col[2], "Color hierarchy mismatch"
    print("[PASS] Test 1 Passed: Spherical mean color calculation correct.")

    # 2. ModernGL Standalone Context & Dummy App
    ctx = moderngl.create_context(standalone=True)
    app = DummyApp(ctx)

    # Create temporary test images
    temp_dir = os.path.join(ROOT_DIR, "data", "temp_test_textures")
    os.makedirs(temp_dir, exist_ok=True)
    diffuse_img_path = os.path.join(temp_dir, "test_diffuse.jpg")
    test_img.convert("RGB").save(diffuse_img_path, "JPEG")

    # Grayscale heightmap image
    height_img = Image.new("L", (256, 128), 128)
    # Add a bump feature
    for x in range(100, 150):
        for y in range(40, 80):
            height_img.putpixel((x, y), 220)
    height_img_path = os.path.join(temp_dir, "test_height.png")
    height_img.save(height_img_path, "PNG")

    # Cloud image (RGB white clouds on black background)
    cloud_img = Image.new("RGB", (256, 128), (0, 0, 0))
    for x in range(50, 200):
        for y in range(30, 90):
            cloud_img.putpixel((x, y), (240, 240, 240))
    cloud_img_path = os.path.join(temp_dir, "test_clouds.jpg")
    cloud_img.save(cloud_img_path, "JPEG")

    # 3. Test Heightmap to Normal Conversion & Cloud Transparency
    assert is_grayscale_image(height_img) is True, "Heightmap should be recognized as grayscale"
    normal_conv = heightmap_to_normal_map(height_img, strength=2.0)
    assert normal_conv.mode == "RGB", "Normal map must be RGB"
    assert normal_conv.size == height_img.size, "Normal map dimensions must match"
    print("[PASS] Test 2 Passed: Heightmap successfully converted to tangent-space normal map.")

    cloud_prep = prepare_layer_image(cloud_img, "clouds")
    assert cloud_prep.mode == "RGBA", "Prepared clouds must be RGBA with alpha channel"
    # Check that black background became transparent (alpha ~ 0) and white clouds have high alpha
    assert cloud_prep.getpixel((0, 0))[3] == 0, "Black cloud background should be transparent (alpha=0)"
    assert cloud_prep.getpixel((100, 50))[3] > 200, "White clouds should have high alpha"
    print("[PASS] Test 3 Passed: Cloud transparency correctly extracted from black/white RGB map.")

    # 4. Test Multi-layer Staging & Discard
    stage_imported_texture(app, ctx, "TestPlanet", diffuse_img_path, layer_key='diffuse')
    assert "testplanet_diffuse" in app._texture_staging, "Staged diffuse entry missing"
    staged_diff = app._texture_staging["testplanet_diffuse"]
    assert staged_diff["is_staged"] is True, "Staged flag not set"
    assert staged_diff["preview_tex"] is not None, "Preview texture not created"

    stage_imported_texture(app, ctx, "TestPlanet", height_img_path, layer_key='normal')
    assert "testplanet_normal" in app._texture_staging, "Staged normal entry missing"
    staged_norm = app._texture_staging["testplanet_normal"]
    assert staged_norm["was_heightmap"] is True, "Heightmap auto-conversion flag must be True"

    discard_staged_texture(app, "TestPlanet", layer_key='normal')
    assert "testplanet_normal" not in app._texture_staging, "Discard should remove staged entry"
    print("[PASS] Test 4 Passed: Multi-layer staging, heightmap conversion detection, and discard lifecycle verified.")

    # 5. Test Hot Reload Body Texture across SSBO Offsets
    # Hot reload diffuse (SSBO offset 0)
    hot_reload_body_texture(app, ctx, "TestPlanet", diffuse_img_path, layer_key='diffuse')
    assert "testplanet" in app.texture_slices, "Texture slice not allocated"
    t_idx = app.texture_slices["testplanet"]
    slot = t_idx - 1

    lo, hi = app.body_textures_ssbo_data[slot, 0]
    handle_diffuse = (int(hi) << 32) | int(lo)
    assert handle_diffuse != 0, "SSBO diffuse handle must be non-zero"

    # Hot reload clouds (SSBO offset 3)
    hot_reload_body_texture(app, ctx, "TestPlanet", cloud_img_path, layer_key='clouds')
    lo_c, hi_c = app.body_textures_ssbo_data[slot, 3]
    handle_clouds = (int(hi_c) << 32) | int(lo_c)
    assert handle_clouds != 0, "SSBO clouds handle must be non-zero"

    # Verify tex_idx_arr
    assert app.tex_idx_arr[2] == float(t_idx), f"tex_idx_arr mismatch: expected {t_idx}, got {app.tex_idx_arr[2]}"
    print(f"[Test 5] Hot reloaded handles for slot {slot}: Diffuse=0x{handle_diffuse:016X}, Clouds=0x{handle_clouds:016X}")
    print("[PASS] Test 5 Passed: ModernGL bindless handles correctly written to SSBO offsets.")

    # 6. Test Apply Staged Texture (copy to textures/<system>/<planet>/)
    stage_imported_texture(app, ctx, "TestPlanet", diffuse_img_path, layer_key='diffuse')
    cur_bodies_data = [{"name": "TestPlanet", "color": "#ff8844"}]
    visual_arr = np.zeros((3, 3), dtype=np.float32)

    apply_staged_texture(
        app, ctx, "TestPlanet", diffuse_img_path,
        layer_key='diffuse',
        active_system_name="TestSystem",
        bodies_data=cur_bodies_data,
        insp_idx=0,
        visual_arr=visual_arr,
        atmo_bodies=[],
        ring_precomputed=[]
    )

    dest_dir = os.path.join(ROOT_DIR, "textures", "TestSystem", "TestPlanet")
    dest_file = os.path.join(dest_dir, "TestPlanet.jpg")
    assert os.path.exists(dest_file), f"Target texture file was not copied to {dest_file}"
    assert "testplanet_diffuse" not in app._texture_staging, "Staged state should be cleared after apply"
    assert cur_bodies_data[0]["texture"] == dest_file, "Body texture path should be updated in bodies_data"
    print(f"[Test 6] Applied texture copied to: {dest_file}")
    print("[PASS] Test 6 Passed: Apply copies to textures/<system>/<planet>/ and hot-reloads.")

    # 7. Test Save Cosmetics directly to Master data/system.json for Solar System
    master_sys_path = get_external_path("data", "system.json")
    with open(master_sys_path, "r", encoding="utf-8") as f:
        master_orig_raw = f.read()
    master_orig_data = json.loads(master_orig_raw)

    # Modify Earth color and call _save_system_cosmetics for Solar System
    test_solar_data = json.loads(master_orig_raw)
    earth_idx = next(i for i, b in enumerate(test_solar_data) if b.get("name") == "Earth")
    test_solar_data[earth_idx]["texture"] = "textures/Earth/Earth.png"

    solar_visual_arr = np.zeros((len(test_solar_data), 3), dtype=np.float32)
    for i, b in enumerate(test_solar_data):
        hex_c = b.get("color", "#ffffff").lstrip("#")
        try:
            solar_visual_arr[i] = [int(hex_c[j:j+2], 16) / 255.0 for j in (0, 2, 4)]
        except Exception:
            solar_visual_arr[i] = [1.0, 1.0, 1.0]
    solar_visual_arr[earth_idx] = [0.2, 0.4, 0.8]  # blue

    try:
        _save_system_cosmetics(
            app, test_solar_data, solar_visual_arr,
            atmo_bodies=[], ring_precomputed=[],
            active_system_name=SystemManager.SOLAR_SYSTEM_NAME
        )

        with open(master_sys_path, "r", encoding="utf-8") as f:
            saved_master = json.load(f)

        saved_earth = next(b for b in saved_master if b.get("name") == "Earth")
        assert saved_earth["color"] == "#3366cc", f"Color not saved to master data/system.json: {saved_earth.get('color')}"
        print("[PASS] Test 7 Passed: Saving Solar System cosmetics directly writes to master data/system.json unconditionally.")
    finally:
        # Restore master data/system.json
        with open(master_sys_path, "w", encoding="utf-8") as f:
            f.write(master_orig_raw)
        print("[Cleanup] Restored original data/system.json.")

    # 8. Test Atmosphere Removal from Master JSON (e.g. Io)
    test_solar_data = json.loads(master_orig_raw)
    io_idx = next(i for i, b in enumerate(test_solar_data) if b.get("name") == "Io")
    assert "atmosphere" in test_solar_data[io_idx], "Io must originally have an atmosphere in master json"

    # Simulate deleting atmosphere from Io (cur_atmo_bodies does not contain io_idx)
    atmo_without_io = [
        {"body_idx": 999, "surface_pressure": 1.0, "temperature": 288.15, "composition": {"N2": 0.78, "O2": 0.21}, "atmo_radius_km": 6400, "planet_radius_km": 6371}
    ]
    if "atmosphere" in test_solar_data[io_idx]:
        del test_solar_data[io_idx]["atmosphere"]

    solar_visual_arr = np.zeros((len(test_solar_data), 3), dtype=np.float32)
    for i, b in enumerate(test_solar_data):
        hex_c = b.get("color", "#ffffff").lstrip("#")
        try:
            solar_visual_arr[i] = [int(hex_c[j:j+2], 16) / 255.0 for j in (0, 2, 4)]
        except Exception:
            solar_visual_arr[i] = [1.0, 1.0, 1.0]

    try:
        _save_system_cosmetics(
            app, test_solar_data, solar_visual_arr,
            atmo_bodies=atmo_without_io, ring_precomputed=[],
            active_system_name=SystemManager.SOLAR_SYSTEM_NAME
        )

        with open(master_sys_path, "r", encoding="utf-8") as f:
            saved_master = json.load(f)

        saved_io = next(b for b in saved_master if b.get("name") == "Io")
        assert "atmosphere" not in saved_io, "Io's atmosphere should be deleted from master data/system.json"
        print("[PASS] Test 8 Passed: Atmosphere removal correctly deletes 'atmosphere' from master data/system.json.")
    finally:
        # Restore master data/system.json
        with open(master_sys_path, "w", encoding="utf-8") as f:
            f.write(master_orig_raw)
        print("[Cleanup] Restored original data/system.json.")

    # 9. Test Asynchronous Tile Baking
    import time
    bake_done_flag = [False]
    def on_bake_complete(success, err):
        bake_done_flag[0] = success

    start_bake_async(app, "TestPlanet", dest_file, map_type='diffuse', max_lod=1, on_complete=on_bake_complete)
    t_start = time.time()
    while not bake_done_flag[0] and (time.time() - t_start < 15.0):
        time.sleep(0.05)

    assert bake_done_flag[0] is True, "Bake worker thread did not complete successfully"
    baked_lod0_tile = os.path.join(ROOT_DIR, "data", "tiles", "TestPlanet", "diffuse", "0", "0", "0_0.jpg")
    assert os.path.exists(baked_lod0_tile), f"Baked tile LOD 0 missing at {baked_lod0_tile}"
    print(f"[Test 8] Baked quadtree tiles verified at: {baked_lod0_tile}")
    print("[PASS] Test 8 Passed: Quadtree tile pyramid baking integration verified.")

    # Cleanup test files
    tiles_test_dir = os.path.join(ROOT_DIR, "data", "tiles", "TestPlanet")
    if os.path.exists(tiles_test_dir):
        shutil.rmtree(tiles_test_dir)
    if os.path.exists(dest_dir):
        shutil.rmtree(os.path.join(ROOT_DIR, "textures", "TestSystem"))
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("\n==================================================")
    print("ALL TEXTURE MANAGER & COSMETICS TESTS PASSED! [PASS]")
    print("==================================================")

if __name__ == "__main__":
    main()
