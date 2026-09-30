"""Hidden-app smoke test for Mode 3 render scale and cache invalidation.

Requires the normal Solar System assets and an OpenGL GPU. Does not save the
app's graphics settings or ImGui layout. Run directly from the repository root.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

import glfw
import imgui
import moderngl
import numpy as np
from engine.app import App


def main():
    original_window = glfw.create_window
    original_context = imgui.create_context
    original_swap = glfw.swap_buffers
    original_write = moderngl.Texture.write
    captured = {}

    def hidden_window(width, height, title, monitor, share):
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        return original_window(800, 600, title, None, share)

    def no_ini(*args, **kwargs):
        result = original_context(*args, **kwargs)
        imgui.get_io().ini_file_name = None
        return result

    def capture_atlas(texture, *args, **kwargs):
        if texture.size == (4096, 16) and texture.components == 4:
            captured['atlas'] = texture
        return original_write(texture, *args, **kwargs)

    glfw.create_window = hidden_window
    imgui.create_context = no_ini
    moderngl.Texture.write = capture_atlas
    app = App()
    app.save_settings = lambda: None
    app.camera.update(atmo_quality=3, atmo_resolution=.5, atmo_temporal_accum=False)
    app.time_ctrl['paused'] = True
    app._perf_target_body = 'Saturn'
    frame = 0
    snapshots = []

    def swap(window):
        nonlocal frame
        frame += 1
        if frame % 10 == 0:
            snapshots.append((app.sky_view_baked_key, app.aerial_volume.baked_key, app._ringshine_map_key))
        if frame == 10:
            assert any(np.any(np.frombuffer(texture.read(), dtype='f4')[3::4] > 0)
                       for pair in app.atmo_lowres_trans_tex.values() for texture in pair), 'Half-resolution atmosphere must render'
            app.camera['fov'] += 1.0
        elif frame == 20:
            app.camera['exposure'] *= 1.1
        elif frame == 30:
            app.camera['atmo_sky_view_steps'] = 24
        elif frame == 40:
            # Exercise the same versioned invalidation as a profile hot reload.
            atlas = captured['atlas']
            atlas.opacity_bounds.update(atlas.atlas_data)
            atlas.atlas_version += 1
        elif frame == 50:
            atmo = next(a for a in app.atmo_bodies if a['body_idx'] == app.camera['tracking_idx'])
            atmo['intensity'] *= 1.1
        elif frame == 70:
            glfw.set_window_should_close(window, True)
        return original_swap(window)

    glfw.swap_buffers = swap
    try:
        app.run()
    finally:
        glfw.create_window = original_window
        imgui.create_context = original_context
        glfw.swap_buffers = original_swap
        moderngl.Texture.write = original_write

    initial, fov, exposure, sky_steps, profile, lighting, stationary = snapshots
    assert initial[0] == fov[0] and initial[1] != fov[1], 'FOV changes only the frustum volume'
    assert fov == exposure, 'Exposure must not rebake physical lighting'
    assert exposure[0] != sky_steps[0] and exposure[1:] == sky_steps[1:], 'Sky quality must not rebake the volume'
    assert all(a != b for a, b in zip(sky_steps, profile)), 'Ring profile edits invalidate all lighting caches'
    assert profile[:2] != lighting[:2] and profile[2] == lighting[2], 'Atmosphere edits preserve the ringshine map'
    assert lighting == stationary, 'Stationary frames must reuse lighting'
    print('Mode 3 hidden-app tests passed: half resolution, FOV, exposure, sky steps, ring profiles, lighting, stationary reuse')


if __name__ == '__main__':
    main()
