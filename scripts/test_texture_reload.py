"""
scripts/test_texture_reload.py

Regression test for texture reloading bugs:

Bug 1 — Index collision:
    hot_reload_body_texture previously used ``len(texture_slices) + 1`` to
    allocate a new slice index.  When a body's texture was deleted (popping
    it from ``texture_slices``), ``len()`` shrank, so the next newly-textured
    body reused the freed index and **overwrote another body's SSBO slot**,
    making that other body's texture disappear.

Bug 2 — Streamer overwrites hot-reloaded texture:
    hot_reload_body_texture never set ``active_res_level[idx] = 'high'``, so
    the per-frame streaming loop kept requesting a high-res decode for the
    body.  A stale decode result (queued before the manifest update) would
    land via ``poll_results`` and clobber the just-hot-reloaded texture.
"""

import os
import sys
import numpy as np
from PIL import Image
import moderngl

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rendering.texture_manager import hot_reload_body_texture
from engine.rendering.planetshine import _build_tex_idx_arr


class DummyStreamer:
    """Minimal stand-in for TextureStreamer exposing the fields hot_reload touches."""
    def __init__(self):
        self.name_to_idx = {}
        self.idx_to_name = {}
        self.texture_mean_colors = {}
        self.file_manifest = {}
        self.in_progress = set()
        self._cancelled = []
        self.cancel_calls = []

    def cancel_in_progress(self, name_lower):
        self.cancel_calls.append(name_lower)
        self.in_progress.discard((name_lower, 'high'))


class DummyApp:
    def __init__(self, ctx):
        self.ctx = ctx
        streamer = DummyStreamer()
        streamer.name_to_idx = {"earth": 1, "mars": 2}
        streamer.idx_to_name = {1: "earth", 2: "mars"}
        self.texture_streamer = streamer
        self.texture_slices = streamer.name_to_idx  # same dict (mirrors app.py)
        self.texture_mean_colors = streamer.texture_mean_colors
        self._texture_slice_counter = max(streamer.name_to_idx.values())
        self.gpu_body_textures = {
            1: {"diffuse": ctx.texture((4, 4), 4, b'\x00' * 64, dtype='f1')},
            2: {"diffuse": ctx.texture((4, 4), 4, b'\x00' * 64, dtype='f1')},
        }
        self.active_res_level = {1: 'low', 2: 'low'}
        self.body_textures_ssbo_data = np.zeros((32, 4, 2), dtype=np.uint32)
        self.body_textures_ssbo = ctx.buffer(self.body_textures_ssbo_data.tobytes())
        self.bodies_data = [{"name": "Earth"}, {"name": "Mars"}, {"name": "Venus"}, {"name": "Mercury"}]
        self.tex_idx_arr = _build_tex_idx_arr(self.bodies_data, self.texture_slices)
        self._texture_staging = {}
        self._texture_bake_state = {}
        self._screenshot_toast = None
        self.impl = None


def _make_img(path, color):
    Image.new("RGB", (64, 32), color).save(path, "PNG")
    return path


def main():
    print("=" * 50)
    print("Running Texture Reload Regression Tests...")
    print("=" * 50)

    ctx = moderngl.create_context(standalone=True)
    tmp = os.path.join(ROOT_DIR, "data", "temp_reload_test")
    os.makedirs(tmp, exist_ok=True)

    # ── Bug 1: Index collision after deletion ──────────────────────
    # Simulate: Earth (idx 1) and Mars (idx 2) exist. Delete Mars (frees idx 2).
    # Now hot-reload a brand-new body ("Venus").  With the old len()+1 logic,
    # Venus would get idx 2 — colliding with any body still mapped to slot 2.
    # With the fix, Venus gets idx 3 (monotonic).
    app = DummyApp(ctx)
    earth_img = _make_img(os.path.join(tmp, "earth.png"), (10, 20, 30))
    mars_img = _make_img(os.path.join(tmp, "mars.png"), (40, 50, 60))
    venus_img = _make_img(os.path.join(tmp, "venus.png"), (70, 80, 90))

    # Pre-seed SSBO handles for Earth & Mars so we can detect corruption
    hot_reload_body_texture(app, ctx, "Earth", earth_img, layer_key='diffuse')
    hot_reload_body_texture(app, ctx, "Mars", mars_img, layer_key='diffuse')

    earth_idx = app.texture_slices["earth"]
    mars_idx = app.texture_slices["mars"]
    earth_slot = earth_idx - 1
    mars_slot = mars_idx - 1

    earth_handle_before = (int(app.body_textures_ssbo_data[earth_slot, 0, 1]) << 32) | int(app.body_textures_ssbo_data[earth_slot, 0, 0])
    mars_handle_before = (int(app.body_textures_ssbo_data[mars_slot, 0, 1]) << 32) | int(app.body_textures_ssbo_data[mars_slot, 0, 0])

    # Simulate deletion of Mars: pop from texture_slices (same dict as name_to_idx)
    app.texture_slices.pop("mars")
    app.gpu_body_textures.pop(mars_idx, None)
    app.active_res_level.pop(mars_idx, None)

    # Hot-reload Venus — must NOT reuse idx 2
    hot_reload_body_texture(app, ctx, "Venus", venus_img, layer_key='diffuse')
    venus_idx = app.texture_slices["venus"]
    assert venus_idx != mars_idx, f"Venus got idx {venus_idx} == deleted Mars idx {mars_idx} — index collision!"
    assert venus_idx == mars_idx + 1, f"Expected monotonic idx {mars_idx + 1}, got {venus_idx}"

    # Earth's handle must be untouched
    earth_handle_after = (int(app.body_textures_ssbo_data[earth_slot, 0, 1]) << 32) | int(app.body_textures_ssbo_data[earth_slot, 0, 0])
    assert earth_handle_after == earth_handle_before, "Earth's SSBO handle corrupted by Venus hot-reload!"
    assert earth_handle_after != 0, "Earth's SSBO handle must remain non-zero"

    print(f"[PASS] Bug 1: No index collision — Earth idx={earth_idx}, Mars idx={mars_idx} (deleted), Venus idx={venus_idx}")
    print(f"       Earth handle preserved: 0x{earth_handle_before:016X} == 0x{earth_handle_after:016X}")

    # ── Bug 2: active_res_level set & stale streamer cancelled ─────
    # After hot_reload, active_res_level must be 'high' so the per-frame
    # streaming loop does not re-request / overwrite.  Also cancel_in_progress
    # must have been called to drain any stale in-flight decode.
    app2 = DummyApp(ctx)
    mercury_img = _make_img(os.path.join(tmp, "mercury.png"), (100, 110, 120))
    hot_reload_body_texture(app2, ctx, "Mercury", mercury_img, layer_key='diffuse')
    merc_idx = app2.texture_slices["mercury"]
    assert app2.active_res_level.get(merc_idx) == 'high', \
        f"active_res_level for hot-reloaded Mercury must be 'high', got {app2.active_res_level.get(merc_idx)!r}"
    assert "mercury" in app2.texture_streamer.cancel_calls, \
        "cancel_in_progress was not called for hot-reloaded Mercury — stale decode could clobber it"
    print(f"[PASS] Bug 2: active_res_level=high & cancel_in_progress called for Mercury (idx={merc_idx})")

    # ── Bug 2b: cancel_in_progress actually drains in_progress ────
    app3 = DummyApp(ctx)
    streamer = app3.texture_streamer
    streamer.in_progress.add(("europa", 'high'))
    streamer.cancel_in_progress("europa")
    assert ("europa", 'high') not in streamer.in_progress, "cancel_in_progress did not discard from in_progress"
    print("[PASS] Bug 2b: cancel_in_progress discards from in_progress set")

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    ctx.release()

    print("\n" + "=" * 50)
    print("ALL TEXTURE RELOAD REGRESSION TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
