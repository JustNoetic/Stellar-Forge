"""
engine/rendering/texture_manager.py

Interactive multi-layer texture management for celestial bodies in Stellar-Forge:
- Equirectangular texture map preview fitted to Inspector -> Cosmetics tab
- Layer switcher: Surface (Diffuse), Clouds, Specular / Ocean Mask, Normal / Heightmap
- File explorer dialog and drag & drop support via GLFW drop callback
- Automatic heightmap-to-normal-map conversion with bump strength control
- Automatic cloud luminance-to-alpha transparency extraction
- Dynamic base color preview and hover color picker when surface texture is not loaded
- Layer-specific fallbacks (atmospheric clouds, ocean specular, tangent normal)
- Asynchronous tile pyramid baking (LOD 3) for SpaceEngine-style terrain streaming
- Hot-reloading textures on the fly (updating ModernGL bindless textures & BodyTextures SSBO)
- Auto-copying imported textures to textures/<system_name>/<planet_name>/
"""

import os
import shutil
import time
import threading
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import moderngl
import imgui
import glfw

from engine.path_utils import get_external_path
from engine.rendering.planetshine import _build_tex_idx_arr
from engine.rendering.texture_streamer import compute_texture_spherical_mean, _prepare_cloud_image

# Metadata for each supported texture layer
TEXTURE_LAYERS = {
    'diffuse': {
        'label': 'Surface (Diffuse)',
        'filename_suffix': '',
        'ssbo_offset': 0,
        'bake_map_type': 'diffuse',
        'can_bake': True,
        'fallback_label': 'No Surface Texture Loaded',
    },
    'clouds': {
        'label': 'Clouds',
        'filename_suffix': '_clouds',
        'ssbo_offset': 3,
        'bake_map_type': 'clouds',
        'can_bake': True,
        'fallback_label': 'No Cloud Layer Loaded',
    },
    'specular': {
        'label': 'Specular / Oceans',
        'filename_suffix': '_specular',
        'ssbo_offset': 2,
        'bake_map_type': 'specular',
        'can_bake': False,
        'fallback_label': 'No Specular / Water Mask Loaded',
    },
    'normal': {
        'label': 'Normal / Heightmap',
        'filename_suffix': '_normal',
        'ssbo_offset': 1,
        'bake_map_type': 'normal',
        'can_bake': False,
        'fallback_label': 'No Normal / Heightmap Loaded',
    },
    'height': {
        'label': 'Heightmap',
        'filename_suffix': '_heightmap',
        'ssbo_offset': None,
        'bake_map_type': 'height',
        'can_bake': True,
        'fallback_label': 'No Heightmap Loaded',
    },
}

def is_grayscale_image(img):
    """Check if an image is grayscale (single channel or identical R=G=B)."""
    if img.mode in ('L', '1', 'I', 'F'):
        return True
    if img.mode in ('RGB', 'RGBA'):
        rgb = np.array(img.convert('RGB'), dtype=np.int16)
        # Check difference between color channels on a sampled grid
        sample = rgb[::8, ::8]
        diff1 = np.abs(sample[..., 0] - sample[..., 1]).max()
        diff2 = np.abs(sample[..., 1] - sample[..., 2]).max()
        return bool(diff1 <= 2 and diff2 <= 2)
    return False

def heightmap_to_normal_map(img, strength=2.0):
    """
    Convert a grayscale heightmap into a tangent-space normal map.
    Uses central differences with horizontal coordinate wrapping for equirectangular maps.
    """
    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
    # Central difference with wrapping along X (longitude wrap)
    gx = (np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1)) * 0.5 * float(strength)
    gy = (np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0)) * 0.5 * float(strength)

    nz = np.ones_like(gx)
    length = np.sqrt(gx * gx + gy * gy + 1.0)
    nx = -gx / length
    ny = -gy / length
    nz = nz / length

    rgb = np.stack([
        np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255),
        np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255),
        np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255)
    ], axis=-1).astype(np.uint8)

    return Image.fromarray(rgb, mode='RGB')

def prepare_layer_image(img, layer_key, bump_strength=2.0):
    """Prepare and normalize an image according to its layer role."""
    if layer_key == 'clouds':
        return _prepare_cloud_image(img)
    elif layer_key == 'normal':
        if is_grayscale_image(img):
            return heightmap_to_normal_map(img, strength=bump_strength).convert('RGBA')
        return img.convert('RGBA')
    elif layer_key == 'specular':
        # Single channel luminance or grayscale
        return img.convert('L')
    elif layer_key == 'height':
        # Grayscale elevation, preserved exactly (NOT convertible to a normal map)
        return img.convert('L')
    else:
        # Diffuse surface
        return img.convert('RGBA')

def open_texture_file_dialog(app, layer_key='diffuse'):
    """Open native OS file dialog in a background thread."""
    if getattr(app, '_file_dialog_open', False):
        return
    app._file_dialog_open = True
    app._file_dialog_layer = layer_key

    def _worker():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            layer_title = TEXTURE_LAYERS.get(layer_key, {}).get('label', 'Texture')
            selected = filedialog.askopenfilename(
                title=f"Select {layer_title} Map (Equirectangular)",
                filetypes=[
                    ("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.webp"),
                    ("JPEG files", "*.jpg;*.jpeg"),
                    ("PNG files", "*.png"),
                    ("TIFF files", "*.tif;*.tiff"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            if selected and os.path.isfile(selected):
                app._pending_import_path = (selected, layer_key)
        except Exception as e:
            print(f"[TextureManager] Error opening file dialog: {e}")
        finally:
            app._file_dialog_open = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def stage_imported_texture(app, ctx, body_name, file_path, layer_key='diffuse', bump_strength=2.0):
    """Load and stage an imported texture for preview in the given layer."""
    try:
        raw_img = Image.open(file_path)
        was_heightmap = (layer_key == 'normal' and is_grayscale_image(raw_img))
        img = prepare_layer_image(raw_img, layer_key, bump_strength=bump_strength)

        # Build 1024x512 preview texture for lightweight ImGui rendering
        preview_img = img.resize((1024, 512), Image.Resampling.LANCZOS)
        if preview_img.mode != 'RGBA':
            preview_img = preview_img.convert('RGBA')

        tex = ctx.texture((1024, 512), 4, preview_img.tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = False

        if hasattr(app, 'impl') and hasattr(app.impl, 'renderer') and hasattr(app.impl.renderer, 'textures'):
            app.impl.renderer.textures[tex.glo] = tex

        name_lower = body_name.lower()
        staging_key = f"{name_lower}_{layer_key}"

        if not hasattr(app, '_texture_staging'):
            app._texture_staging = {}

        # Release existing staged texture for this layer
        old_staging = app._texture_staging.get(staging_key)
        if old_staging and old_staging.get('preview_tex') is not None:
            try:
                old_tex = old_staging['preview_tex']
                if hasattr(app, 'impl') and hasattr(app.impl, 'renderer') and hasattr(app.impl.renderer, 'textures'):
                    app.impl.renderer.textures.pop(old_tex.glo, None)
                old_tex.release()
            except Exception:
                pass

        app._texture_staging[staging_key] = {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'img_size': raw_img.size,
            'preview_tex': tex,
            'layer_key': layer_key,
            'bump_strength': bump_strength,
            'was_heightmap': was_heightmap,
            'is_staged': True
        }

        bake_key = f"{name_lower}_{layer_key}"
        if not hasattr(app, '_texture_bake_state'):
            app._texture_bake_state = {}
        app._texture_bake_state[bake_key] = {'status': 'idle', 'msg': ''}

        layer_lbl = TEXTURE_LAYERS.get(layer_key, {}).get('label', layer_key)
        conv_info = " (converted from heightmap)" if was_heightmap else ""
        app._screenshot_toast = (f"Loaded {layer_lbl} '{os.path.basename(file_path)}'{conv_info} for {body_name}! Choose Bake or Apply.", time.time())
        print(f"[TextureManager] Staged {layer_key} texture '{file_path}' ({raw_img.width}x{raw_img.height}) for {body_name}")
    except Exception as e:
        print(f"[TextureManager] Error staging {layer_key} texture for {body_name}: {e}")
        app._screenshot_toast = (f"Failed to load image: {e}", time.time())

def discard_staged_texture(app, body_name, layer_key='diffuse'):
    """Discard a staged texture without applying."""
    name_lower = body_name.lower()
    staging_key = f"{name_lower}_{layer_key}"
    if hasattr(app, '_texture_staging') and staging_key in app._texture_staging:
        staged = app._texture_staging.pop(staging_key)
        if staged.get('preview_tex') is not None:
            try:
                tex = staged['preview_tex']
                if hasattr(app, 'impl') and hasattr(app.impl, 'renderer') and hasattr(app.impl.renderer, 'textures'):
                    app.impl.renderer.textures.pop(tex.glo, None)
                tex.release()
            except Exception:
                pass
    bake_key = f"{name_lower}_{layer_key}"
    if hasattr(app, '_texture_bake_state') and bake_key in app._texture_bake_state:
        app._texture_bake_state[bake_key] = {'status': 'idle', 'msg': ''}

def start_bake_async(app, body_name, src_path, map_type='diffuse', max_lod=3, on_complete=None):
    """Run cubemap tile pyramid baking on a background thread."""
    name_lower = body_name.lower()
    bake_key = f"{name_lower}_{map_type}"
    if not hasattr(app, '_texture_bake_state'):
        app._texture_bake_state = {}

    app._texture_bake_state[bake_key] = {'status': 'baking', 'msg': f'Baking {map_type} tiles (LOD {max_lod})...'}

    def _worker():
        try:
            from scripts.bake_planet_tiles import bake_planet
            out_base = get_external_path("data", "tiles")
            bake_planet(body_name, src_path, map_type=map_type, max_lod=max_lod, out_base=out_base)

            # Invalidate terrain streamer caches so new tiles stream immediately
            if hasattr(app, 'terrain_streamer') and app.terrain_streamer is not None:
                app.terrain_streamer.reload_layer(body_name, map_type)

            if hasattr(app, '_terrain_tiles_dir_cache') and app._terrain_tiles_dir_cache is not None:
                app._terrain_tiles_dir_cache.clear()

            app._texture_bake_state[bake_key] = {'status': 'done', 'msg': f'{map_type.capitalize()} tiles baked.'}
            app._screenshot_toast = (f"Quadtree {map_type} tiles baked for {body_name}!", time.time())
            print(f"[TextureManager] Completed {map_type} tile baking for {body_name}")
            if on_complete:
                on_complete(True, None)
        except Exception as e:
            err_msg = str(e)
            print(f"[TextureManager] Error baking {map_type} tiles for {body_name}: {err_msg}")
            app._texture_bake_state[bake_key] = {'status': 'error', 'msg': err_msg}
            app._screenshot_toast = (f"Bake error: {err_msg}", time.time())
            if on_complete:
                on_complete(False, err_msg)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def hot_reload_body_texture(app, ctx, body_name, file_path, layer_key='diffuse', bump_strength=2.0):
    """
    Hot reload a texture on the fly for any layer (diffuse, normal, specular, clouds):
    - Prepares and normalizes the image
    - Allocates/replaces ModernGL resident texture with anisotropy and mipmaps
    - Writes the 64-bit bindless handle into BodyTextures SSBO (binding 10)
    - Updates instance texture indices for immediate frame update
    """
    if not file_path or not os.path.isfile(file_path):
        print(f"[TextureManager] Cannot hot reload: file '{file_path}' does not exist.")
        return False
    name_lower = body_name.lower()
    try:
        raw_img = Image.open(file_path)
    except Exception as e:
        print(f"[TextureManager] Error opening texture file '{file_path}': {e}")
        return False
    if layer_key == 'height':
        if not hasattr(app, '_height_previews'):
            app._height_previews = {}
        old = app._height_previews.pop(name_lower, None)
        registry = getattr(getattr(getattr(app, 'impl', None), 'renderer', None), 'textures', {})
        if old is not None:
            registry.pop(old.glo, None)
            old.release()
        preview = raw_img.convert('L').resize((1024, 512), Image.Resampling.BILINEAR).convert('RGBA')
        tex = ctx.texture(preview.size, 4, preview.tobytes())
        app._height_previews[name_lower] = tex
        registry[tex.glo] = tex
        return True
    img = prepare_layer_image(raw_img, layer_key, bump_strength=bump_strength)

    # For diffuse, compute spherical latitude-weighted mean linear color
    if layer_key == 'diffuse':
        mean_linear = compute_texture_spherical_mean(img)
        if not hasattr(app, 'texture_mean_colors'):
            app.texture_mean_colors = {}
        app.texture_mean_colors[name_lower] = mean_linear
        if hasattr(app, 'texture_streamer') and app.texture_streamer is not None:
            app.texture_streamer.texture_mean_colors[name_lower] = mean_linear

    # Update texture streamer manifest
    if hasattr(app, 'texture_streamer') and app.texture_streamer is not None:
        manifest_entry = app.texture_streamer.file_manifest.setdefault(name_lower, {
            'name': body_name,
            'diffuse': None,
            'normal': None,
            'specular': None,
            'clouds': None
        })
        manifest_entry[layer_key] = file_path

    # Invalidate clouds exist cache if cloud layer was updated
    if layer_key == 'clouds' and hasattr(app, '_clouds_exist_cache'):
        app._clouds_exist_cache.pop(name_lower, None)

    # Assign / retrieve texture slice index
    if not hasattr(app, 'texture_slices'):
        app.texture_slices = {}

    if name_lower not in app.texture_slices:
        # Use a monotonically-increasing counter to avoid index collisions when
        # bodies have been deleted (len()+1 reuses freed slots, corrupting
        # other bodies' SSBO handles).
        if hasattr(app, '_texture_slice_counter'):
            new_idx = app._texture_slice_counter + 1
        else:
            new_idx = len(app.texture_slices) + 1
            # Initialise counter above any pre-existing streamer indices
            if hasattr(app, 'texture_streamer') and app.texture_streamer is not None:
                app._texture_slice_counter = max(new_idx, max(app.texture_streamer.name_to_idx.values(), default=0))
            else:
                app._texture_slice_counter = new_idx
        app._texture_slice_counter = max(app._texture_slice_counter, new_idx)
        app.texture_slices[name_lower] = new_idx
        if hasattr(app, 'texture_streamer') and app.texture_streamer is not None:
            app.texture_streamer.name_to_idx[name_lower] = new_idx
            app.texture_streamer.idx_to_name[new_idx] = name_lower
        idx = new_idx
    else:
        idx = app.texture_slices[name_lower]

    slot = idx - 1
    m_offset = TEXTURE_LAYERS[layer_key]['ssbo_offset']

    # Cap resolution to 8192x4096 for GPU texture safety
    MAX_HIGH_RES_DIM = 8192
    if img.width > MAX_HIGH_RES_DIM or img.height > (MAX_HIGH_RES_DIM // 2):
        scale = min(MAX_HIGH_RES_DIM / img.width, (MAX_HIGH_RES_DIM // 2) / img.height)
        nw = max(1, int(img.width * scale))
        nh = max(1, int(img.height * scale))
        img_upload = img.resize((nw, nh), Image.Resampling.LANCZOS)
    else:
        img_upload = img

    try:
        max_aniso = float(ctx.max_anisotropy or 1.0)
    except Exception:
        max_aniso = 1.0
    aniso_value = max(1.0, min(max_aniso, 16.0))

    components = 1 if img_upload.mode == 'L' else 4
    if components == 4 and img_upload.mode != 'RGBA':
        img_upload = img_upload.convert('RGBA')

    # Release previous resident texture for this layer
    tex_dict = app.gpu_body_textures.setdefault(idx, {})
    if layer_key in tex_dict and tex_dict[layer_key] is not None:
        try:
            old_tex = tex_dict[layer_key]
            if hasattr(app, 'impl') and hasattr(app.impl, 'renderer') and hasattr(app.impl.renderer, 'textures'):
                app.impl.renderer.textures.pop(old_tex.glo, None)
            old_tex.release()
        except Exception:
            pass

    # Create new texture on GPU
    new_tex = ctx.texture(img_upload.size, components, img_upload.tobytes(), dtype='f1')
    new_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    new_tex.repeat_x = True
    new_tex.build_mipmaps()
    new_tex.anisotropy = aniso_value

    tex_dict[layer_key] = new_tex
    app.gpu_body_textures[idx] = tex_dict

    # Register in ImGui renderer texture map for preview
    if hasattr(app, 'impl') and hasattr(app.impl, 'renderer') and hasattr(app.impl.renderer, 'textures'):
        app.impl.renderer.textures[new_tex.glo] = new_tex

    # Update bindless handle in SSBO
    handle_64 = int(new_tex.get_handle(resident=True))
    lo = handle_64 & 0xFFFFFFFF
    hi = (handle_64 >> 32) & 0xFFFFFFFF

    # Ensure SSBO buffer is large enough
    if slot >= len(app.body_textures_ssbo_data):
        new_capacity = max(slot + 16, len(app.body_textures_ssbo_data) * 2)
        new_data = np.zeros((new_capacity, 4, 2), dtype=np.uint32)
        new_data[:len(app.body_textures_ssbo_data)] = app.body_textures_ssbo_data
        app.body_textures_ssbo_data = new_data
        try:
            app.body_textures_ssbo.release()
        except Exception:
            pass
        app.body_textures_ssbo = ctx.buffer(app.body_textures_ssbo_data.tobytes())
        app.body_textures_ssbo.bind_to_storage_buffer(binding=10)

    app.body_textures_ssbo_data[slot, m_offset] = [lo, hi]
    app.body_textures_ssbo.write(app.body_textures_ssbo_data.tobytes())

    # Mark this body as 'high' so the per-frame streamer doesn't overwrite
    # our hot-reloaded texture with a stale/low-res decode. Also cancel any
    # in-flight decode for this body so a queued stale result can't clobber it.
    if hasattr(app, 'active_res_level'):
        app.active_res_level[idx] = 'high'
    if hasattr(app, 'texture_streamer') and app.texture_streamer is not None:
        if hasattr(app.texture_streamer, 'cancel_in_progress'):
            app.texture_streamer.cancel_in_progress(name_lower)

    # Rebuild tex_idx_arr
    app.tex_idx_arr = _build_tex_idx_arr(app.bodies_data, app.texture_slices)
    if hasattr(app, 'tex_idx_arr_cmp') and getattr(app, 'bodies_data_cmp', None):
        app.tex_idx_arr_cmp = _build_tex_idx_arr(app.bodies_data_cmp, app.texture_slices)

    print(f"[TextureManager] Successfully hot-reloaded {layer_key} for {body_name} (slot {slot}, handle 0x{handle_64:016X})")

def _is_same_path(p1, p2):
    """Robust path equality check accounting for Windows case insensitivity and path format."""
    if not p1 or not p2:
        return False
    try:
        if os.path.exists(p1) and os.path.exists(p2):
            return os.path.samefile(p1, p2)
    except Exception:
        pass
    try:
        return os.path.normcase(os.path.abspath(p1)) == os.path.normcase(os.path.abspath(p2))
    except Exception:
        return False

def _cleanup_layer_disk_files(body_name, layer_key, active_system_name, preserve_file=None):
    """
    Remove existing texture files and baked tiles on disk for a specific layer before replacement or upon deletion.
    Preserves `preserve_file` if specified.
    """
    name_lower = body_name.lower()

    # Check system folder and generic body folder
    dirs_to_check = [
        os.path.join(get_external_path("textures"), active_system_name, body_name),
        os.path.join(get_external_path("textures"), body_name),
    ]

    layer_suffixes = {
        'diffuse': ('',),
        'clouds': ('_clouds', '_cloud'),
        'specular': ('_specular', '_spec'),
        'normal': ('_normal', '_bump', '_nm'),
        'height': ('_heightmap',)
    }
    target_suffixes = layer_suffixes.get(layer_key, ('',))

    generic_names = {
        'diffuse': {'diffuse', 'albedo', 'color', 'map'},
        'clouds': {'clouds', 'cloud'},
        'specular': {'specular', 'spec', 'diffuse_specular', 'specular_map'},
        'normal': {'normal', 'bump', 'diffuse_normal', 'nm'},
        'height': {'height', 'heightmap'}
    }.get(layer_key, set())

    deleted_any = False
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')

    for folder in dirs_to_check:
        if not os.path.isdir(folder):
            continue
        try:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if not os.path.isfile(fpath):
                    continue
                if preserve_file and _is_same_path(fpath, preserve_file):
                    continue

                stem, ext = os.path.splitext(fname)
                stem_lower = stem.lower()
                ext_lower = ext.lower()
                if ext_lower not in valid_exts:
                    continue

                matched = False
                if layer_key == 'diffuse':
                    has_other_suffix = any(
                        stem_lower.endswith(sfx)
                        for sfx in ('_clouds', '_cloud', '_specular', '_spec', '_normal', '_bump', '_nm', '_heightmap', '_ring', '_rings', '_front', '_back')
                    )
                    if not has_other_suffix and (stem_lower == name_lower or stem_lower in generic_names):
                        matched = True
                else:
                    for sfx in target_suffixes:
                        if stem_lower == f"{name_lower}{sfx}" or stem_lower in generic_names:
                            matched = True
                            break

                if matched:
                    try:
                        os.remove(fpath)
                        deleted_any = True
                        print(f"[TextureManager] Cleaned up {layer_key} texture file: {fpath}")
                    except Exception as e:
                        print(f"[TextureManager] Error removing file {fpath}: {e}")
        except Exception as e:
            print(f"[TextureManager] Error scanning folder {folder}: {e}")

    # Check root textures folder for exact body layer matches (e.g. textures/Earth_clouds.png)
    root_tex = get_external_path("textures")
    if os.path.isdir(root_tex):
        try:
            for fname in os.listdir(root_tex):
                fpath = os.path.join(root_tex, fname)
                if not os.path.isfile(fpath):
                    continue
                if preserve_file and _is_same_path(fpath, preserve_file):
                    continue
                stem, ext = os.path.splitext(fname)
                stem_lower = stem.lower()
                if ext.lower() not in valid_exts:
                    continue

                matched = False
                if layer_key == 'diffuse':
                    has_other_suffix = any(
                        stem_lower.endswith(sfx)
                        for sfx in ('_clouds', '_cloud', '_specular', '_spec', '_normal', '_bump', '_nm', '_heightmap', '_ring', '_rings', '_front', '_back')
                    )
                    if not has_other_suffix and stem_lower == name_lower:
                        matched = True
                else:
                    for sfx in target_suffixes:
                        if stem_lower == f"{name_lower}{sfx}":
                            matched = True
                            break
                if matched:
                    try:
                        os.remove(fpath)
                        deleted_any = True
                        print(f"[TextureManager] Cleaned up root texture file: {fpath}")
                    except Exception as e:
                        print(f"[TextureManager] Error removing root file {fpath}: {e}")
        except Exception:
            pass

    # Baked tiles for this layer
    bake_map = TEXTURE_LAYERS.get(layer_key, {}).get('bake_map_type')
    if bake_map:
        for b_cand in (body_name, name_lower):
            tile_dir = os.path.join(get_external_path("data", "tiles"), b_cand, bake_map)
            if os.path.isdir(tile_dir):
                try:
                    shutil.rmtree(tile_dir, ignore_errors=True)
                    deleted_any = True
                    print(f"[TextureManager] Removed baked tile pyramid: {tile_dir}")
                except Exception as e:
                    print(f"[TextureManager] Error removing tile directory {tile_dir}: {e}")

    return deleted_any

def apply_staged_texture(app, ctx, body_name, staged_path, layer_key, active_system_name, bodies_data, insp_idx, visual_arr, atmo_bodies, ring_precomputed, bump_strength=2.0):
    """
    Apply imported texture for the given layer:
    1. Cleans up existing texture files & tiles for this layer (replacement cleanup)
    2. Copies texture to textures/<system_name>/<planet_name>/<planet_name><suffix>.<ext>
    3. Hot reloads texture on the fly
    4. Clears staged preview state for this layer
    """
    if not staged_path or not os.path.isfile(staged_path):
        app._screenshot_toast = (f"Staged file '{staged_path}' does not exist on disk!", time.time())
        print(f"[TextureManager] Error: Staged file '{staged_path}' not found.")
        return False

    dest_dir = os.path.join(get_external_path("textures"), active_system_name, body_name)
    suffix = TEXTURE_LAYERS[layer_key]['filename_suffix']

    ext = os.path.splitext(staged_path)[1].lower()
    if not ext or ext not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'):
        ext = '.jpg'

    name_lower = body_name.lower()
    staging_key = f"{name_lower}_{layer_key}"
    staged = getattr(app, '_texture_staging', {}).get(staging_key, {})

    if layer_key == 'height' or staged.get('was_heightmap'):
        dest_file = os.path.join(dest_dir, f"{body_name}{suffix}.png")
    else:
        dest_file = os.path.join(dest_dir, f"{body_name}{suffix}{ext}")

    # 1. Clean up existing old texture files & tiles (preserving staged_path if it's already in dest_dir)
    _cleanup_layer_disk_files(body_name, layer_key, active_system_name, preserve_file=staged_path)

    # 2. Ensure destination directory exists after cleanup
    os.makedirs(dest_dir, exist_ok=True)

    # 3. Save converted normal map or copy staged file to dest_file
    try:
        if layer_key == 'height':
            with Image.open(staged_path) as source:
                height_img = source.convert('L')
            height_img.save(dest_file, 'PNG')
        elif staged.get('was_heightmap'):
            raw_img = Image.open(staged_path)
            norm_img = heightmap_to_normal_map(raw_img, strength=bump_strength)
            norm_img.save(dest_file, "PNG")
            print(f"[TextureManager] Saved converted normal map to '{dest_file}'")
        elif not _is_same_path(staged_path, dest_file):
            shutil.copy2(staged_path, dest_file)
            print(f"[TextureManager] Copied '{staged_path}' to '{dest_file}'")
    except Exception as e:
        print(f"[TextureManager] Error saving/copying {layer_key} file: {e}")
        app._screenshot_toast = (f"Failed to copy file: {e}", time.time())
        return False

    if not os.path.isfile(dest_file):
        app._screenshot_toast = (f"Texture file '{dest_file}' could not be created!", time.time())
        print(f"[TextureManager] Error: dest_file '{dest_file}' does not exist after copy.")
        return False

    hot_reload_body_texture(app, ctx, body_name, dest_file, layer_key=layer_key, bump_strength=bump_strength)

    if insp_idx < len(bodies_data):
        json_prop = 'texture' if layer_key == 'diffuse' else layer_key
        bodies_data[insp_idx][json_prop] = dest_file

    discard_staged_texture(app, body_name, layer_key=layer_key)
    if layer_key == 'height':
        start_bake_async(app, body_name, dest_file, map_type='height')
    layer_lbl = TEXTURE_LAYERS[layer_key]['label']
    app._screenshot_toast = (f"{layer_lbl} applied & hot-reloaded for {body_name}!", time.time())
    return True

def bake_and_apply_staged_texture(app, ctx, body_name, staged_path, layer_key, active_system_name, bodies_data, insp_idx, visual_arr, atmo_bodies, ring_precomputed, bump_strength=2.0):
    """
    Bake tiles and apply texture for the given layer (diffuse or clouds):
    1. Cleans up existing texture files & tiles for this layer (replacement cleanup)
    2. Copies texture to textures/<system_name>/<planet_name>/<planet_name><suffix>.<ext>
    3. Hot reloads texture on the fly
    4. Bakes tile pyramid asynchronously
    """
    if not staged_path or not os.path.isfile(staged_path):
        app._screenshot_toast = (f"Staged file '{staged_path}' does not exist on disk!", time.time())
        print(f"[TextureManager] Error: Staged file '{staged_path}' not found.")
        return False

    dest_dir = os.path.join(get_external_path("textures"), active_system_name, body_name)
    suffix = TEXTURE_LAYERS[layer_key]['filename_suffix']

    ext = os.path.splitext(staged_path)[1].lower()
    if not ext or ext not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'):
        ext = '.jpg'

    name_lower = body_name.lower()
    staging_key = f"{name_lower}_{layer_key}"
    staged = getattr(app, '_texture_staging', {}).get(staging_key, {})

    if layer_key == 'height' or staged.get('was_heightmap'):
        dest_file = os.path.join(dest_dir, f"{body_name}{suffix}.png")
    else:
        dest_file = os.path.join(dest_dir, f"{body_name}{suffix}{ext}")

    # 1. Clean up existing old texture files & tiles (preserving staged_path if it's already in dest_dir)
    _cleanup_layer_disk_files(body_name, layer_key, active_system_name, preserve_file=staged_path)

    # 2. Ensure destination directory exists after cleanup
    os.makedirs(dest_dir, exist_ok=True)

    # 3. Save converted normal map or copy staged file to dest_file
    try:
        if layer_key == 'height':
            with Image.open(staged_path) as source:
                height_img = source.convert('L')
            height_img.save(dest_file, 'PNG')
        elif staged.get('was_heightmap'):
            raw_img = Image.open(staged_path)
            norm_img = heightmap_to_normal_map(raw_img, strength=bump_strength)
            norm_img.save(dest_file, "PNG")
            print(f"[TextureManager] Saved converted normal map to '{dest_file}'")
        elif not _is_same_path(staged_path, dest_file):
            shutil.copy2(staged_path, dest_file)
            print(f"[TextureManager] Copied '{staged_path}' to '{dest_file}'")
    except Exception as e:
        print(f"[TextureManager] Error copying texture file: {e}")
        app._screenshot_toast = (f"Failed to copy file: {e}", time.time())
        return False

    if not os.path.isfile(dest_file):
        app._screenshot_toast = (f"Texture file '{dest_file}' could not be created!", time.time())
        print(f"[TextureManager] Error: dest_file '{dest_file}' does not exist after copy.")
        return False

    hot_reload_body_texture(app, ctx, body_name, dest_file, layer_key=layer_key, bump_strength=bump_strength)

    if insp_idx < len(bodies_data):
        json_prop = 'texture' if layer_key == 'diffuse' else layer_key
        bodies_data[insp_idx][json_prop] = dest_file

    discard_staged_texture(app, body_name, layer_key=layer_key)

    map_type = TEXTURE_LAYERS[layer_key]['bake_map_type']
    start_bake_async(app, body_name, dest_file, map_type=map_type)
    layer_lbl = TEXTURE_LAYERS[layer_key]['label']
    app._screenshot_toast = (f"{layer_lbl} applied! Baking quadtree tiles for {body_name}...", time.time())
    return True

def delete_body_layer_texture(app, ctx, body_name, layer_key, active_system_name, bodies_data, insp_idx, insp_is_cmp=False):
    """
    Permanently delete the selected texture layer for a body:
    1. Removes layer texture file(s) from textures/
    2. Removes baked cubemap tiles from data/tiles/<body_name>/<map_type>/
    3. Invalidates terrain streamer caches
    4. Unbinds and releases resident GPU texture
    5. Clears SSBO 64-bit handle to uvec2(0, 0)
    6. Removes layer reference from bodies_data and file_manifest
    7. Discards any active staged preview
    """
    name_lower = body_name.lower()

    # 1. Discard any staged texture for this layer
    discard_staged_texture(app, body_name, layer_key=layer_key)

    if layer_key == 'height':
        tex = getattr(app, '_height_previews', {}).pop(name_lower, None)
        if tex is not None:
            registry = getattr(getattr(getattr(app, 'impl', None), 'renderer', None), 'textures', {})
            registry.pop(tex.glo, None)
            tex.release()

    # 2. Delete files and baked tiles on disk
    _cleanup_layer_disk_files(body_name, layer_key, active_system_name, preserve_file=None)

    # 3. Invalidate terrain streamer caches
    if hasattr(app, 'terrain_streamer') and app.terrain_streamer is not None:
        app.terrain_streamer.reload_layer(body_name, layer_key)
    if hasattr(app, '_terrain_tiles_dir_cache') and app._terrain_tiles_dir_cache is not None:
        app._terrain_tiles_dir_cache.clear()

    # 4. Clean up texture streamer manifest & colors
    if hasattr(app, 'texture_streamer') and app.texture_streamer is not None:
        if name_lower in app.texture_streamer.file_manifest:
            app.texture_streamer.file_manifest[name_lower][layer_key] = None
        if layer_key == 'diffuse':
            app.texture_streamer.texture_mean_colors.pop(name_lower, None)

    if layer_key == 'diffuse' and hasattr(app, 'texture_mean_colors'):
        app.texture_mean_colors.pop(name_lower, None)

    if layer_key == 'clouds' and hasattr(app, '_clouds_exist_cache'):
        app._clouds_exist_cache.pop(name_lower, None)

    # 5. GPU textures and SSBO
    idx = getattr(app, 'texture_slices', {}).get(name_lower)
    if idx and layer_key != 'height':
        slot = idx - 1
        m_offset = TEXTURE_LAYERS[layer_key]['ssbo_offset']

        # Zero out SSBO handle
        if hasattr(app, 'body_textures_ssbo_data') and slot < len(app.body_textures_ssbo_data):
            app.body_textures_ssbo_data[slot, m_offset] = [0, 0]
            if hasattr(app, 'body_textures_ssbo') and app.body_textures_ssbo is not None:
                app.body_textures_ssbo.write(app.body_textures_ssbo_data.tobytes())

        # Release resident GPU texture
        tex_dict = getattr(app, 'gpu_body_textures', {}).get(idx, {})
        if layer_key in tex_dict and tex_dict[layer_key] is not None:
            try:
                old_tex = tex_dict.pop(layer_key)
                if hasattr(app, 'impl') and hasattr(app.impl, 'renderer') and hasattr(app.impl.renderer, 'textures'):
                    app.impl.renderer.textures.pop(old_tex.glo, None)
                old_tex.release()
            except Exception:
                pass

        # Check if this body has any remaining active layer textures
        remaining_layers = any(tex is not None for tex in tex_dict.values())
        if not remaining_layers:
            m_entry = getattr(app.texture_streamer, 'file_manifest', {}).get(name_lower, {}) if hasattr(app, 'texture_streamer') and app.texture_streamer else {}
            has_manifest_file = any(m_entry.get(k) is not None for k in TEXTURE_LAYERS)
            if not has_manifest_file:
                app.gpu_body_textures.pop(idx, None)
                app.texture_slices.pop(name_lower, None)
                if hasattr(app, 'texture_streamer') and app.texture_streamer:
                    app.texture_streamer.name_to_idx.pop(name_lower, None)
                    app.texture_streamer.idx_to_name.pop(idx, None)
                app.tex_idx_arr = _build_tex_idx_arr(app.bodies_data, app.texture_slices)
                if hasattr(app, 'tex_idx_arr_cmp') and getattr(app, 'bodies_data_cmp', None):
                    app.tex_idx_arr_cmp = _build_tex_idx_arr(app.bodies_data_cmp, app.texture_slices)

    # 6. Remove property from bodies_data
    json_prop = 'texture' if layer_key == 'diffuse' else layer_key
    if insp_idx < len(bodies_data):
        bodies_data[insp_idx].pop(json_prop, None)
        if layer_key == 'diffuse':
            bodies_data[insp_idx].pop('diffuse', None)

    if hasattr(app, 'bodies_data'):
        for b in app.bodies_data:
            if b.get('name', '').lower() == name_lower:
                b.pop(json_prop, None)
                if layer_key == 'diffuse':
                    b.pop('diffuse', None)

    if hasattr(app, 'bodies_data_cmp') and app.bodies_data_cmp:
        for b in app.bodies_data_cmp:
            if b.get('name', '').lower() == name_lower:
                b.pop(json_prop, None)
                if layer_key == 'diffuse':
                    b.pop('diffuse', None)

    # 7. Reset bake state
    bake_key = f"{name_lower}_{layer_key}"
    if hasattr(app, '_texture_bake_state') and bake_key in app._texture_bake_state:
        app._texture_bake_state[bake_key] = {'status': 'idle', 'msg': ''}

    layer_lbl = TEXTURE_LAYERS[layer_key]['label']
    app._screenshot_toast = (f"Deleted {layer_lbl} texture & tiles for {body_name}!", time.time())
    print(f"[TextureManager] Deleted {layer_key} texture and baked tiles for {body_name}")

def delete_body_all_textures(app, ctx, body_name, active_system_name, bodies_data, insp_idx, insp_is_cmp=False):
    """
    Permanently delete ALL texture layers and baked tiles for a celestial body:
    1. Discards all staged layers
    2. Deletes textures/<system>/<body_name> and fallback textures/<body_name>
    3. Deletes data/tiles/<body_name>/
    4. Invalidates terrain streamer caches
    5. Clears SSBO 64-bit handles for all 4 slots to uvec2(0, 0)
    6. Releases all GPU resident textures
    7. Removes from texture_slices, manifest, and bodies_data
    8. Updates tex_idx_arr
    """
    name_lower = body_name.lower()

    # 1. Discard all staged textures
    for lk in TEXTURE_LAYERS:
        discard_staged_texture(app, body_name, layer_key=lk)
        bake_key = f"{name_lower}_{lk}"
        if hasattr(app, '_texture_bake_state') and bake_key in app._texture_bake_state:
            app._texture_bake_state[bake_key] = {'status': 'idle', 'msg': ''}

    height_preview = getattr(app, '_height_previews', {}).pop(name_lower, None)
    if height_preview is not None:
        registry = getattr(getattr(getattr(app, 'impl', None), 'renderer', None), 'textures', {})
        registry.pop(height_preview.glo, None)
        height_preview.release()

    # 2. Delete all textures on disk for this body
    dirs_to_remove = [
        os.path.join(get_external_path("textures"), active_system_name, body_name),
        os.path.join(get_external_path("textures"), body_name),
    ]
    for d in dirs_to_remove:
        if os.path.isdir(d):
            try:
                shutil.rmtree(d, ignore_errors=True)
                print(f"[TextureManager] Removed texture directory: {d}")
            except Exception as e:
                print(f"[TextureManager] Error removing texture directory {d}: {e}")

    # Also clean up loose files in root textures/
    root_tex = get_external_path("textures")
    if os.path.isdir(root_tex):
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
        all_sfxs = ('', '_clouds', '_cloud', '_specular', '_spec', '_normal', '_bump', '_nm', '_heightmap')
        try:
            for fname in os.listdir(root_tex):
                fpath = os.path.join(root_tex, fname)
                if not os.path.isfile(fpath):
                    continue
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in valid_exts:
                    continue
                stem_lower = stem.lower()
                for sfx in all_sfxs:
                    if stem_lower == f"{name_lower}{sfx}":
                        try:
                            os.remove(fpath)
                            print(f"[TextureManager] Removed loose texture file: {fpath}")
                        except Exception:
                            pass
                        break
        except Exception:
            pass

    # 3. Delete baked tiles
    for b_cand in (body_name, name_lower):
        tile_dir = os.path.join(get_external_path("data", "tiles"), b_cand)
        if os.path.isdir(tile_dir):
            try:
                shutil.rmtree(tile_dir, ignore_errors=True)
                print(f"[TextureManager] Removed tile directory: {tile_dir}")
            except Exception as e:
                print(f"[TextureManager] Error removing tile directory {tile_dir}: {e}")

    # 4. Invalidate terrain streamer caches
    if hasattr(app, 'terrain_streamer') and app.terrain_streamer is not None:
        app.terrain_streamer.reload_layer(body_name)
    if hasattr(app, '_terrain_tiles_dir_cache') and app._terrain_tiles_dir_cache is not None:
        app._terrain_tiles_dir_cache.clear()

    # 5. Clean up texture streamer manifest & colors
    if hasattr(app, 'texture_streamer') and app.texture_streamer is not None:
        app.texture_streamer.file_manifest.pop(name_lower, None)
        app.texture_streamer.texture_mean_colors.pop(name_lower, None)

    if hasattr(app, 'texture_mean_colors'):
        app.texture_mean_colors.pop(name_lower, None)

    if hasattr(app, '_clouds_exist_cache'):
        app._clouds_exist_cache.pop(name_lower, None)

    # 6. GPU textures and SSBO
    idx = getattr(app, 'texture_slices', {}).pop(name_lower, None)
    if idx:
        slot = idx - 1
        # Zero out all 4 offsets in SSBO
        if hasattr(app, 'body_textures_ssbo_data') and slot < len(app.body_textures_ssbo_data):
            for off in range(4):
                app.body_textures_ssbo_data[slot, off] = [0, 0]
            if hasattr(app, 'body_textures_ssbo') and app.body_textures_ssbo is not None:
                app.body_textures_ssbo.write(app.body_textures_ssbo_data.tobytes())

        # Release resident GPU textures
        tex_dict = getattr(app, 'gpu_body_textures', {}).pop(idx, {})
        for layer_k, tex in tex_dict.items():
            if tex is not None:
                if hasattr(app, 'impl') and hasattr(app.impl, 'renderer') and hasattr(app.impl.renderer, 'textures'):
                    app.impl.renderer.textures.pop(tex.glo, None)
                try:
                    tex.release()
                except Exception:
                    pass

        if hasattr(app, 'texture_streamer') and app.texture_streamer:
            app.texture_streamer.name_to_idx.pop(name_lower, None)
            app.texture_streamer.idx_to_name.pop(idx, None)

        app.tex_idx_arr = _build_tex_idx_arr(app.bodies_data, app.texture_slices)
        if hasattr(app, 'tex_idx_arr_cmp') and getattr(app, 'bodies_data_cmp', None):
            app.tex_idx_arr_cmp = _build_tex_idx_arr(app.bodies_data_cmp, app.texture_slices)

    # 7. Remove properties from bodies_data
    for prop in ('texture', 'diffuse', 'clouds', 'specular', 'normal', 'height'):
        if insp_idx < len(bodies_data):
            bodies_data[insp_idx].pop(prop, None)

        if hasattr(app, 'bodies_data'):
            for b in app.bodies_data:
                if b.get('name', '').lower() == name_lower:
                    b.pop(prop, None)

        if hasattr(app, 'bodies_data_cmp') and app.bodies_data_cmp:
            for b in app.bodies_data_cmp:
                if b.get('name', '').lower() == name_lower:
                    b.pop(prop, None)

    app._screenshot_toast = (f"Deleted all textures and tiles for {body_name}!", time.time())
    print(f"[TextureManager] Successfully deleted all textures and tiles for {body_name}")

def _has_layer_texture(app, name_lower, layer_key, body_name=None):
    """Check if a body has a texture loaded, staged, in manifest, or on disk."""
    if layer_key == 'height' and name_lower in getattr(app, '_height_previews', {}):
        return True
    staging_key = f"{name_lower}_{layer_key}"
    staged = getattr(app, '_texture_staging', {}).get(staging_key)
    if staged and staged.get('is_staged'):
        return True

    idx = getattr(app, 'texture_slices', {}).get(name_lower)
    if idx:
        t_dict = getattr(app, 'gpu_body_textures', {}).get(idx, {})
        if t_dict.get(layer_key) is not None:
            return True

    if hasattr(app, 'texture_streamer') and app.texture_streamer:
        p = app.texture_streamer.file_manifest.get(name_lower, {}).get(layer_key)
        if p and os.path.exists(p):
            return True

    # Check tiles
    bake_map = TEXTURE_LAYERS.get(layer_key, {}).get('bake_map_type')
    if bake_map:
        cands = [name_lower]
        if body_name:
            cands.append(body_name)
        for b_cand in cands:
            tile_dir = os.path.join(get_external_path("data", "tiles"), b_cand, bake_map)
            if os.path.isdir(tile_dir):
                return True

    return False

def _has_any_texture(app, name_lower, body_name=None):
    """Check if a body has any texture (surface, clouds, specular, normal) or tiles."""
    for lk in TEXTURE_LAYERS:
        if _has_layer_texture(app, name_lower, lk, body_name=body_name):
            return True
    cands = [name_lower]
    if body_name:
        cands.append(body_name)
    for b_cand in cands:
        tile_dir = os.path.join(get_external_path("data", "tiles"), b_cand)
        if os.path.isdir(tile_dir):
            return True
    return False

def render_texture_management_ui(app, ctx, insp_idx, body_info, visual_arr, active_system_name, cur_bodies_data, atmo_bodies, ring_precomputed, insp_is_cmp):
    """
    Render multi-layer texture management in Inspector -> Cosmetics:
    - Layer selector: Surface (Diffuse), Clouds, Specular / Oceans, Normal / Heightmap
    - Large equirectangular texture preview box (2:1 aspect ratio) fitted to window
    - Base color fill & hover color picker for diffuse surface
    - Layer-specific fallbacks and auto-conversions (clouds transparency, heightmap to normal)
    - Direct map clicking & drag-and-drop support
    - Staging, Bake, Apply, and Bake & Apply actions
    """
    body_name = body_info['name']
    name_lower = body_name.lower()

    # Active layer selection
    cur_layer = getattr(app, '_active_texture_layer', 'diffuse')
    if cur_layer not in TEXTURE_LAYERS:
        cur_layer = 'diffuse'

    # Check for pending file dialog selection
    pending = getattr(app, '_pending_import_path', None)
    if pending:
        pending_path, pending_layer = pending
        stage_imported_texture(app, ctx, body_name, pending_path, layer_key=pending_layer)
        app._pending_import_path = None

    avail_w = imgui.get_content_region_available_width()

    # ── 1. Texture Layer Selector Sub-Tabs ──
    btn_w = int((avail_w - 32) / len(TEXTURE_LAYERS))
    for layer_k, layer_meta in TEXTURE_LAYERS.items():
        is_active = (cur_layer == layer_k)
        has_tex = _has_layer_texture(app, name_lower, layer_k)

        if is_active:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.22, 0.55, 0.85)
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0)
        else:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.18, 0.18, 0.22)
            imgui.push_style_color(imgui.COLOR_TEXT, 0.75, 0.75, 0.75)

        # Short tab label with dot if texture is present
        dot = " *" if has_tex else ""
        short_names = {
            'diffuse': f"Surface{dot}",
            'clouds': f"Clouds{dot}",
            'specular': f"Specular{dot}",
            'normal': f"Normal{dot}",
            'height': f"Height{dot}"
        }
        tab_label = f"{short_names[layer_k]}##tab_{layer_k}"
        if imgui.button(tab_label, width=btn_w):
            app._active_texture_layer = layer_k
            cur_layer = layer_k

        imgui.pop_style_color(2)
        if layer_k != 'height':
            imgui.same_line()

    imgui.spacing()

    # ── 2. Layer Map Preview Box ──
    map_h = max(100, int(avail_w * 0.5)) # 2:1 aspect ratio for equirectangular projection

    cursor_screen = imgui.get_cursor_screen_pos()
    p_min = (cursor_screen[0], cursor_screen[1])
    p_max = (p_min[0] + avail_w, p_min[1] + map_h)

    # Full map invisible button for click and drag detection
    imgui.invisible_button(f"##tex_map_btn_{insp_idx}_{cur_layer}", avail_w, map_h)
    is_box_hovered = imgui.is_item_hovered()
    is_box_clicked = imgui.is_item_clicked(0)
    saved_cursor_pos = imgui.get_cursor_pos()

    # Check GLFW drag & drop
    dropped_info = getattr(app, '_dropped_files', None)
    if dropped_info:
        paths, (mx, my) = dropped_info
        is_drop_in_box = (p_min[0] <= mx <= p_max[0] and p_min[1] <= my <= p_max[1])
        if is_drop_in_box or is_box_hovered:
            valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')
            img_p = next((p for p in paths if p.lower().endswith(valid_exts)), None)
            if img_p:
                stage_imported_texture(app, ctx, body_name, img_p, layer_key=cur_layer)
            app._dropped_files = None

    # Determine texture state for current layer
    staging_key = f"{name_lower}_{cur_layer}"
    staged = getattr(app, '_texture_staging', {}).get(staging_key)
    tex_glo = None
    is_texture_loaded = False
    is_staged = False

    if cur_layer == 'height' and not staged:
        if name_lower not in getattr(app, '_height_previews', {}):
            source = body_info.get('height') or os.path.join(get_external_path('textures'), active_system_name, body_name, body_name + '_heightmap.png')
            if os.path.isfile(source):
                hot_reload_body_texture(app, ctx, body_name, source, layer_key='height')
        height_preview = getattr(app, '_height_previews', {}).get(name_lower)
        if height_preview is not None:
            tex_glo = height_preview.glo
            is_texture_loaded = True

    if staged and staged.get('is_staged'):
        tex_glo = staged['preview_tex'].glo
        is_texture_loaded = True
        is_staged = True
    elif name_lower in getattr(app, 'texture_slices', {}):
        t_idx = app.texture_slices[name_lower]
        t_dict = getattr(app, 'gpu_body_textures', {}).get(t_idx, {})
        layer_tex = t_dict.get(cur_layer)
        if layer_tex is not None:
            tex_glo = layer_tex.glo
            is_texture_loaded = True

    draw_list = imgui.get_window_draw_list()

    # Base color in sRGB (for diffuse surface)
    if insp_is_cmp:
        srgb_c = [pow(c, 1.0/2.2) if c > 0 else 0.0 for c in app.visual_arr_cmp[insp_idx, 0:3]]
    else:
        srgb_c = [pow(c, 1.0/2.2) if c > 0 else 0.0 for c in visual_arr[insp_idx, 0:3]]

    picker_w, picker_h = 24, 22
    picker_x = p_max[0] - picker_w - 6
    picker_y = p_max[1] - picker_h - 6

    trash_w, trash_h = 24, 22
    trash_x = p_min[0] + 6
    trash_y = p_max[1] - trash_h - 6

    io = imgui.get_io()
    mx, my = io.mouse_pos
    is_picker_hovered = (picker_x <= mx <= picker_x + picker_w and picker_y <= my <= picker_y + picker_h)
    is_trash_hovered = (trash_x <= mx <= trash_x + trash_w and trash_y <= my <= trash_y + trash_h)

    has_layer_content = (
        is_texture_loaded
        or is_staged
        or _has_layer_texture(app, name_lower, cur_layer, body_name=body_name)
    )

    if is_texture_loaded and tex_glo is not None:
        # ── Render Loaded Texture Preview ──
        draw_list.add_image(tex_glo, p_min, p_max, (0, 0), (1, 1), 0xFFFFFFFF)
        border_col = imgui.color_convert_float4_to_u32(0.2, 0.6, 1.0, 0.9) if is_box_hovered else imgui.color_convert_float4_to_u32(0.35, 0.35, 0.4, 0.8)
        draw_list.add_rect(p_min[0], p_min[1], p_max[0], p_max[1], border_col, rounding=4.0, thickness=1.5)

        if is_staged:
            # Staged badge
            draw_list.add_rect_filled(p_min[0] + 6, p_min[1] + 6, p_min[0] + 115, p_min[1] + 24, imgui.color_convert_float4_to_u32(0.1, 0.5, 0.8, 0.85), rounding=3.0)
            draw_list.add_text(p_min[0] + 10, p_min[1] + 8, imgui.color_convert_float4_to_u32(1.0, 1.0, 1.0, 1.0), "STAGED (Preview)")

    else:
        # ── Render Fallback Fill when texture not loaded ──
        if cur_layer == 'diffuse':
            bg_col = imgui.color_convert_float4_to_u32(srgb_c[0], srgb_c[1], srgb_c[2], 1.0)
            lum = 0.299 * srgb_c[0] + 0.587 * srgb_c[1] + 0.114 * srgb_c[2]
            txt_col = imgui.color_convert_float4_to_u32(0.1, 0.1, 0.1, 0.85) if lum > 0.5 else imgui.color_convert_float4_to_u32(0.95, 0.95, 0.95, 0.85)
        elif cur_layer == 'clouds':
            bg_col = imgui.color_convert_float4_to_u32(0.08, 0.12, 0.22, 1.0)
            txt_col = imgui.color_convert_float4_to_u32(0.8, 0.88, 1.0, 0.85)
        elif cur_layer == 'specular':
            bg_col = imgui.color_convert_float4_to_u32(0.05, 0.05, 0.07, 1.0)
            txt_col = imgui.color_convert_float4_to_u32(0.75, 0.75, 0.75, 0.85)
        else: # normal
            bg_col = imgui.color_convert_float4_to_u32(0.5, 0.5, 1.0, 1.0) # Tangent space flat normal
            txt_col = imgui.color_convert_float4_to_u32(0.15, 0.15, 0.35, 0.85)

        draw_list.add_rect_filled(p_min[0], p_min[1], p_max[0], p_max[1], bg_col, rounding=4.0)
        border_col = imgui.color_convert_float4_to_u32(0.8, 0.8, 0.8, 0.9) if is_box_hovered else imgui.color_convert_float4_to_u32(0.35, 0.35, 0.4, 0.8)
        draw_list.add_rect(p_min[0], p_min[1], p_max[0], p_max[1], border_col, rounding=4.0, thickness=1.5)

        # Center label
        fallback_msg = TEXTURE_LAYERS[cur_layer]['fallback_label']
        draw_list.add_text(p_min[0] + avail_w * 0.5 - 65, p_min[1] + map_h * 0.5 - 16, txt_col, fallback_msg)
        draw_list.add_text(p_min[0] + avail_w * 0.5 - 72, p_min[1] + map_h * 0.5 + 4, txt_col, "Click or Drag Image Here")

        # Color picker in bottom right when hovered on diffuse surface
        if cur_layer == 'diffuse' and (is_box_hovered or is_picker_hovered):
            draw_list.add_rect_filled(picker_x - 2, picker_y - 2, picker_x + picker_w + 2, picker_y + picker_h + 2, imgui.color_convert_float4_to_u32(0.0, 0.0, 0.0, 0.5), rounding=3.0)
            imgui.set_cursor_screen_pos((picker_x, picker_y))
            changed_c, new_c = imgui.color_edit3(
                f"##hover_map_picker_{insp_idx}",
                *srgb_c,
                imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_INPUTS
            )
            if changed_c:
                linear_c = [pow(c, 2.2) if c > 0 else 0.0 for c in new_c]
                if insp_is_cmp:
                    app.visual_arr_cmp[insp_idx, 0:3] = linear_c
                    app.visual_data_cmp[insp_idx][0:3] = linear_c
                    atmo_item = next((a for a in app.atmo_bodies_cmp if a['body_idx'] == insp_idx), None)
                else:
                    visual_arr[insp_idx, 0:3] = linear_c
                    if hasattr(app, "visual_data") and len(app.visual_data) > insp_idx:
                        app.visual_data[insp_idx][0:3] = linear_c
                    atmo_item = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)

                if atmo_item:
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']

    # Trash can symbol on bottom left of preview box
    if has_layer_content:
        if is_trash_hovered:
            trash_bg = imgui.color_convert_float4_to_u32(0.85, 0.2, 0.2, 0.9)
            trash_border = imgui.color_convert_float4_to_u32(1.0, 0.4, 0.4, 1.0)
            trash_icon = imgui.color_convert_float4_to_u32(1.0, 1.0, 1.0, 1.0)
        else:
            trash_bg = imgui.color_convert_float4_to_u32(0.12, 0.12, 0.15, 0.75)
            trash_border = imgui.color_convert_float4_to_u32(0.55, 0.25, 0.25, 0.8)
            trash_icon = imgui.color_convert_float4_to_u32(0.9, 0.55, 0.55, 0.9)

        draw_list.add_rect_filled(trash_x, trash_y, trash_x + trash_w, trash_y + trash_h, trash_bg, rounding=3.0)
        draw_list.add_rect(trash_x, trash_y, trash_x + trash_w, trash_y + trash_h, trash_border, rounding=3.0, thickness=1.0)

        # Vector trash can symbol centered at (trash_x + 12, trash_y + 11)
        tc_x = trash_x + 12.0
        tc_y = trash_y + 11.0
        # Top lid handle
        draw_list.add_line(tc_x - 2.5, tc_y - 6.0, tc_x + 2.5, tc_y - 6.0, trash_icon, 1.5)
        # Lid horizontal bar
        draw_list.add_line(tc_x - 5.5, tc_y - 4.0, tc_x + 5.5, tc_y - 4.0, trash_icon, 1.5)
        # Can body rectangle
        draw_list.add_rect(tc_x - 4.5, tc_y - 2.0, tc_x + 4.5, tc_y + 6.0, trash_icon, rounding=1.0, thickness=1.2)
        # Vertical slats inside can
        draw_list.add_line(tc_x - 1.5, tc_y - 0.5, tc_x - 1.5, tc_y + 4.5, trash_icon, 1.0)
        draw_list.add_line(tc_x + 1.5, tc_y - 0.5, tc_x + 1.5, tc_y + 4.5, trash_icon, 1.0)

    # Tooltips for preview box
    if is_trash_hovered and has_layer_content:
        imgui.set_tooltip(f"Delete {TEXTURE_LAYERS[cur_layer]['label']} Texture & Tiles\nClick to permanently remove from disk, tiles, and memory.")
    elif cur_layer == 'diffuse' and is_picker_hovered:
        imgui.set_tooltip("Base Color Picker\nClick to customize surface color")
    elif is_box_hovered:
        action_str = "replace" if is_texture_loaded else "load"
        imgui.set_tooltip(f"Click directly to open file explorer or drag & drop an image to {action_str} {TEXTURE_LAYERS[cur_layer]['label']}")

    # Click handling for preview box
    if is_box_clicked:
        if is_trash_hovered and has_layer_content:
            imgui.open_popup(f"Delete Layer Texture?###del_layer_modal_{insp_idx}")
        elif cur_layer == 'diffuse' and is_picker_hovered:
            pass
        else:
            open_texture_file_dialog(app, layer_key=cur_layer)

    # ── 3. Controls Below the Map ──
    imgui.set_cursor_pos(saved_cursor_pos)
    imgui.spacing()

    bake_key = f"{name_lower}_{cur_layer}"
    bake_state = getattr(app, '_texture_bake_state', {}).get(bake_key, {})
    b_status = bake_state.get('status', 'idle')
    is_baking = (b_status == 'baking')
    can_bake = TEXTURE_LAYERS[cur_layer]['can_bake']

    if is_staged:
        # Information banner
        conv_note = " [Heightmap -> Tangent Normal]" if staged.get('was_heightmap') else ""
        imgui.text_colored(f"Staged {TEXTURE_LAYERS[cur_layer]['label']}: {staged['filename']} ({staged['img_size'][0]}x{staged['img_size'][1]}){conv_note}", 0.4, 0.85, 1.0)

        # Bump strength slider for converted heightmaps
        bump_str = staged.get('bump_strength', 2.0)
        if staged.get('was_heightmap'):
            ch_b, n_b = imgui.slider_float("Bump Strength##height_str", bump_str, 0.2, 10.0, "%.1f")
            if ch_b:
                stage_imported_texture(app, ctx, body_name, staged['path'], layer_key='normal', bump_strength=n_b)

        # Action choices
        num_buttons = 4 if can_bake else 2
        btn_w = int((avail_w - 20) / num_buttons)

        if can_bake:
            if is_baking:
                imgui.text_colored("⏳ Baking...", 1.0, 0.85, 0.3)
                imgui.same_line()
            else:
                if imgui.button(f"Bake##stage_{cur_layer}", width=btn_w):
                    start_bake_async(app, body_name, staged['path'], map_type=TEXTURE_LAYERS[cur_layer]['bake_map_type'])
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Bake cubemap quadtree {cur_layer} tiles (LOD 3) for terrain streaming")
                imgui.same_line()

        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.3)
        if imgui.button(f"Apply##stage_{cur_layer}", width=btn_w):
            apply_staged_texture(app, ctx, body_name, staged['path'], cur_layer, active_system_name, cur_bodies_data, insp_idx, visual_arr, atmo_bodies, ring_precomputed, bump_strength=bump_str)
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Save heightmap and bake terrain tiles" if cur_layer == 'height' else
                              f"Copy texture to textures/{active_system_name}/{body_name}/ and reload on the fly without baking")
        imgui.same_line()

        if can_bake:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.15, 0.45, 0.75)
            if imgui.button(f"Bake & Apply##stage_{cur_layer}", width=btn_w + 15):
                bake_and_apply_staged_texture(app, ctx, body_name, staged['path'], cur_layer, active_system_name, cur_bodies_data, insp_idx, visual_arr, atmo_bodies, ring_precomputed, bump_strength=bump_str)
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Bake cubemap tiles AND copy texture to textures/{active_system_name}/{body_name}/ & reload on the fly")
            imgui.same_line()

        if imgui.button(f"Discard##stage_{cur_layer}", width=btn_w - 15 if can_bake else btn_w):
            discard_staged_texture(app, body_name, layer_key=cur_layer)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Discard staged texture")

    else:
        if is_texture_loaded:
            import_btn_w = avail_w - 105 if can_bake else avail_w
            if imgui.button(f"Import {TEXTURE_LAYERS[cur_layer]['label']}...##loaded", width=import_btn_w):
                open_texture_file_dialog(app, layer_key=cur_layer)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Open file explorer to import a new {TEXTURE_LAYERS[cur_layer]['label']} map")

            if can_bake:
                imgui.same_line()
                if is_baking:
                    imgui.text_colored("⏳ Baking...", 1.0, 0.85, 0.3)
                else:
                    if imgui.button(f"Bake Tiles##loaded_{cur_layer}", width=95):
                        d_path = None
                        if hasattr(app, 'texture_streamer') and app.texture_streamer:
                            d_path = app.texture_streamer.file_manifest.get(name_lower, {}).get(cur_layer)
                        if not d_path:
                            json_prop = 'texture' if cur_layer == 'diffuse' else cur_layer
                            d_path = body_info.get(json_prop)
                        if not d_path and cur_layer == 'height':
                            d_path = os.path.join(get_external_path('textures'), active_system_name, body_name, body_name + '_heightmap.png')
                        if d_path and os.path.exists(d_path):
                            start_bake_async(app, body_name, d_path, map_type=TEXTURE_LAYERS[cur_layer]['bake_map_type'])
                        else:
                            app._screenshot_toast = (f"No {cur_layer} file on disk to bake for {body_name}", time.time())
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(f"Bake cubemap quadtree {cur_layer} tiles (LOD 3) for SpaceEngine-style terrain streaming")
        else:
            if imgui.button(f"Import {TEXTURE_LAYERS[cur_layer]['label']}...##empty", width=-1):
                open_texture_file_dialog(app, layer_key=cur_layer)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Open file explorer to select an equirectangular {TEXTURE_LAYERS[cur_layer]['label']} map (.jpg, .png, .tif)")

    # Status readout for tile baking
    if is_baking:
        imgui.text_colored(f"⏳ Baking quadtree {cur_layer} tiles in background (LOD 3)...", 1.0, 0.85, 0.3)
    elif b_status == 'done':
        imgui.text_colored(f"✓ Quadtree {cur_layer} tiles baked to data/tiles/!", 0.3, 1.0, 0.4)
    elif b_status == 'error':
        imgui.text_colored(f"✗ Bake error: {bake_state.get('msg')}", 1.0, 0.3, 0.3)

    # ── 4. Big Delete All Textures Button ──
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    has_any_tex = _has_any_texture(app, name_lower, body_name=body_name)

    if has_any_tex:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.65, 0.18, 0.18)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.85, 0.25, 0.25)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.50, 0.12, 0.12)
        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0)
    else:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.22, 0.22, 0.25)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.22, 0.22, 0.25)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.22, 0.22, 0.25)
        imgui.push_style_color(imgui.COLOR_TEXT, 0.50, 0.50, 0.50)

    if imgui.button(f"Delete All Textures ({body_name})###btn_del_all_{insp_idx}", width=-1):
        if has_any_tex:
            imgui.open_popup(f"Delete All Textures?###del_all_modal_{insp_idx}")

    imgui.pop_style_color(4)

    if imgui.is_item_hovered():
        if has_any_tex:
            imgui.set_tooltip(f"Permanently delete all texture maps (Surface, Clouds, Specular, Normal) and all baked tiles for {body_name}")
        else:
            imgui.set_tooltip(f"No textures or baked tiles to delete for {body_name}")

    # ── 5. Confirmation Modals ──
    modal_single_id = f"Delete Layer Texture?###del_layer_modal_{insp_idx}"
    if imgui.begin_popup_modal(modal_single_id, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
        imgui.text_colored(f"Are you sure you want to delete the {TEXTURE_LAYERS[cur_layer]['label']} texture for {body_name}?", 1.0, 0.8, 0.35)
        imgui.spacing()
        imgui.text("This will permanently remove:")
        imgui.bullet_text(f"Texture files for this layer on disk")
        if TEXTURE_LAYERS[cur_layer]['can_bake']:
            imgui.bullet_text(f"Baked cubemap tiles from data/tiles/{body_name}/{TEXTURE_LAYERS[cur_layer]['bake_map_type']}/")
        imgui.bullet_text("Unload texture from GPU memory and reset shaders")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.2, 0.2)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.95, 0.3, 0.3)
        if imgui.button(f"Yes, Delete {TEXTURE_LAYERS[cur_layer]['label']}", width=210):
            delete_body_layer_texture(app, ctx, body_name, cur_layer, active_system_name, cur_bodies_data, insp_idx, insp_is_cmp)
            imgui.close_current_popup()
        imgui.pop_style_color(2)

        imgui.same_line()
        if imgui.button("Cancel", width=90):
            imgui.close_current_popup()

        imgui.end_popup()

    modal_all_id = f"Delete All Textures?###del_all_modal_{insp_idx}"
    if imgui.begin_popup_modal(modal_all_id, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
        imgui.text_colored(f"⚠️ PERMANENT ACTION: Delete ALL textures for {body_name}?", 1.0, 0.35, 0.35)
        imgui.spacing()
        imgui.text("This will permanently remove:")
        imgui.bullet_text(f"ALL texture maps (Surface, Clouds, Specular, Normal) from textures/")
        imgui.bullet_text(f"ALL baked tiles from data/tiles/{body_name}/")
        imgui.bullet_text("Unload all textures from GPU memory and reset shader slots")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.push_style_color(imgui.COLOR_BUTTON, 0.85, 0.15, 0.15)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1.0, 0.25, 0.25)
        if imgui.button("Yes, Delete ALL Textures", width=210):
            delete_body_all_textures(app, ctx, body_name, active_system_name, cur_bodies_data, insp_idx, insp_is_cmp)
            imgui.close_current_popup()
        imgui.pop_style_color(2)

        imgui.same_line()
        if imgui.button("Cancel", width=90):
            imgui.close_current_popup()

        imgui.end_popup()
