import os
import numpy as np
from PIL import Image
import moderngl
from engine.path_utils import get_external_path
from engine.rendering.render_utils import (
    generate_ring_shadow_grad, 
    rebuild_ring_render_group, 
    apply_hsba_np,
    compute_5_ring_colors
)
from engine.rendering.ring_generator import generate_procedural_ring_profile, RING_PRESETS

def bake_and_export_ring_textures(app, body_name, ring_item, ring_precomputed=None, ring_render_groups=None, ring_gradient_tex=None):
    name_lower = body_name.lower()

    base_textures = get_external_path('textures')

    system_name = getattr(app, 'loaded_system_name', 'Solar System') or 'Solar System'

    cand_sys_body = os.path.join(base_textures, system_name, body_name)
    cand_solar_body = os.path.join(base_textures, 'Solar System', body_name)
    cand_legacy_body = os.path.join(base_textures, body_name)

    if os.path.exists(cand_sys_body):
        textures_dir = cand_sys_body
    elif os.path.exists(cand_solar_body):
        textures_dir = cand_solar_body
    elif os.path.exists(cand_legacy_body):
        textures_dir = cand_legacy_body
    else:
        textures_dir = os.path.join(base_textures, system_name, body_name)

    os.makedirs(textures_dir, exist_ok=True)

    cand_rings_png = os.path.join(textures_dir, "Rings.png")
    cand_body_ring = os.path.join(textures_dir, f"{body_name}_ring.png")
    if os.path.exists(cand_rings_png):
        ring_path = cand_rings_png
    elif os.path.exists(cand_body_ring):
        ring_path = cand_body_ring
    else:
        ring_path = cand_body_ring

    img_ring = app.ring_textures.get(name_lower)
    if img_ring is None:
        print(f"[Texture Editor] No ring texture found for {body_name}")
        return False

    arr_ring = np.array(img_ring, dtype=np.float32) / 255.0

    hue = ring_item.get('hue_shift', 0.0)
    sat = ring_item.get('saturation', 1.0)
    bri = ring_item.get('brightness', 1.0)
    op = ring_item.get('opacity', 1.0)
    boost = ring_item.get('alpha_boost', 1.0)

    baked_ring = apply_hsba_np(arr_ring, hue, sat, bri, opacity_mult=op, alpha_boost=boost)
    baked_ring_u8 = np.clip(baked_ring * 255.0 + 0.5, 0, 255).astype(np.uint8)

    img_ring_new = Image.fromarray(baked_ring_u8, mode='RGBA')

    img_ring_new.save(ring_path)
    print(f"[Texture Editor] Baked and saved ring texture to {ring_path}")

    app.ring_textures[name_lower] = img_ring_new

    ctx = app.ctx
    aniso_value = app.camera.get("anisotropy", 16.0)
    tex = ctx.texture(img_ring_new.size, 4, img_ring_new.tobytes())
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    tex.build_mipmaps()
    tex.anisotropy = aniso_value

    if hasattr(app, 'ring_gl_textures') and name_lower in app.ring_gl_textures:
        try:
            old_tex = app.ring_gl_textures[name_lower]
            if old_tex is not None and old_tex is not tex:
                old_tex.release()
        except Exception:
            pass
    app.ring_gl_textures[name_lower] = tex

    ring_item['saturation'] = 1.0
    ring_item['hue_shift'] = 0.0
    ring_item['brightness'] = 1.0
    ring_item['opacity'] = 1.0
    ring_item['alpha_boost'] = 1.0

    if ring_precomputed is not None and ring_render_groups is not None and ring_gradient_tex is not None:
        sampled_arr = np.array(img_ring_new, dtype='f4') / 255.0
        ring_item['tex_sampled'] = sampled_arr
        shadow_grad = generate_ring_shadow_grad(
            ring_item['gradient'],
            tex_sampled=sampled_arr,
            raw_color=ring_item.get('raw_color', (1.0, 1.0, 1.0))
        )
        ring_item['shadow_grad'] = shadow_grad
        app.body_ring_indices = rebuild_ring_render_group(ring_item['body_idx'], ctx, app.prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
    return True

def apply_procedural_ring_to_body(
    app,
    body_idx,
    preset="Saturnian Ice",
    seed=42,
    custom_params=None,
    bake_to_disk=False,
    bodies_data=None,
    visual_arr=None,
    ring_precomputed=None,
    ring_render_groups=None,
    ring_gradient_tex=None,
    prog_rings=None,
    ctx=None
):
    """
    Generate and apply a procedural 1D ring texture to a body in Stellar-Forge.
    Live updates textures, OpenGL resources, and ring_precomputed items.
    """
    if custom_params is None:
        custom_params = {}

    cur_bodies_data = bodies_data if bodies_data is not None else getattr(app, 'bodies_data', None)
    if cur_bodies_data is None:
        insp_is_cmp = getattr(app, 'camera', {}).get("inspected_is_cmp", False)
        cur_bodies_data = getattr(app, 'bodies_data_cmp', None) if insp_is_cmp else getattr(app, 'bodies_data', None)

    if cur_bodies_data is None or body_idx >= len(cur_bodies_data):
        print(f"[Procedural Rings] Invalid body index {body_idx} (bodies_data length: {len(cur_bodies_data) if cur_bodies_data else 0})")
        return False

    body_info = cur_bodies_data[body_idx]
    body_name = body_info['name']
    name_lower = body_name.lower()

    body_r_km = float(body_info.get('r', 0.0) * 696340.0)
    body_r_au = body_r_km / 1.495978707e8
    if body_r_au <= 1e-9:
        body_r_au = 1e-5

    # Generate profile using ring_generator
    img_ring, arr_rgba, suggested_props = generate_procedural_ring_profile(
        seed=seed,
        preset=preset,
        band_count=custom_params.get('band_count'),
        gap_count=custom_params.get('gap_count'),
        striation_freq=custom_params.get('striation_freq'),
        contrast=custom_params.get('contrast'),
        base_density=custom_params.get('base_density'),
        core_boost=custom_params.get('core_boost'),
        tint_color=custom_params.get('tint_color', (1.0, 1.0, 1.0))
    )

    # Calculate inner/outer radii
    inner_mult = custom_params.get('inner_radius_ratio', suggested_props['inner_radius_ratio'])
    outer_mult = custom_params.get('outer_radius_ratio', suggested_props['outer_radius_ratio'])

    inner_r_au = body_r_au * inner_mult
    outer_r_au = body_r_au * outer_mult

    # Store in app textures dictionary
    app.ring_textures[name_lower] = img_ring

    # Create / update ModernGL OpenGL texture
    ctx = ctx if ctx is not None else getattr(app, 'ctx', None)
    aniso_value = getattr(app, 'camera', {}).get("anisotropy", 16.0) if hasattr(app, 'camera') else 16.0
    if ctx is not None:
        tex = ctx.texture(img_ring.size, 4, img_ring.tobytes())
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.build_mipmaps()
        tex.anisotropy = aniso_value
        if hasattr(app, 'ring_gl_textures') and name_lower in app.ring_gl_textures:
            try:
                old_tex = app.ring_gl_textures[name_lower]
                if old_tex is not None and old_tex is not tex:
                    old_tex.release()
            except Exception:
                pass
        app.ring_gl_textures[name_lower] = tex

    # Resolve pole normal
    cur_visual_arr = visual_arr if visual_arr is not None else getattr(app, 'visual_data', None)
    if cur_visual_arr is not None and body_idx < len(cur_visual_arr):
        pole_render = cur_visual_arr[body_idx, 5:8]
    else:
        pole_render = np.array([0.0, 1.0, 0.0])
    pole_norm = np.linalg.norm(pole_render)
    pole_n = pole_render / pole_norm if pole_norm > 1e-8 else np.array([0.0, 1.0, 0.0])

    ring_precomputed = ring_precomputed if ring_precomputed is not None else getattr(app, 'ring_precomputed', None)
    if ring_precomputed is None:
        ring_precomputed = []
        if hasattr(app, 'ring_precomputed'):
            app.ring_precomputed = ring_precomputed
    ring_render_groups = ring_render_groups if ring_render_groups is not None else getattr(app, 'ring_render_groups', None)
    ring_gradient_tex = ring_gradient_tex if ring_gradient_tex is not None else getattr(app, 'ring_gradient_tex', None)
    prog_rings = prog_rings if prog_rings is not None else getattr(app, 'prog_rings', None)

    sampled_arr = arr_rgba.astype('f4')
    raw_col = tuple(float(c) for c in custom_params.get('tint_color', suggested_props['raw_color']))
    shadow_grad = generate_ring_shadow_grad([], tex_sampled=sampled_arr, raw_color=raw_col)
    colors5 = compute_5_ring_colors(tex_sampled=sampled_arr, raw_color=raw_col)

    # Find existing ring layers for this body
    existing = [r for r in ring_precomputed if r['body_idx'] == body_idx]
    if existing:
        r_item = existing[0]
        r_item['pole'] = pole_n.astype('f4')
        r_item['inner_r'] = inner_r_au
        r_item['outer_r'] = outer_r_au
        r_item['opacity'] = float(custom_params.get('opacity', suggested_props['opacity']))
        r_item['scatter'] = float(suggested_props['scatter'])
        r_item['asymmetry'] = float(suggested_props['asymmetry'])
        r_item['backscatter'] = float(suggested_props['backscatter'])
        r_item['unlit_factor'] = float(suggested_props['unlit_factor'])
        r_item['shadow_grad'] = shadow_grad
        r_item['color'] = raw_col
        r_item['raw_color'] = raw_col
        r_item['gradient'] = []
        r_item['tex_sampled'] = sampled_arr
        r_item['5colors'] = colors5
        r_item['is_textured'] = True
        r_item['saturation'] = 1.0
        r_item['hue_shift'] = 0.0
        r_item['brightness'] = 1.0
        r_item['alpha_boost'] = 1.0

        if len(existing) > 1:
            extras_ids = {id(r) for r in existing[1:]}
            ring_precomputed[:] = [r for r in ring_precomputed if id(r) not in extras_ids]
            for idx, r in enumerate(ring_precomputed):
                r['row_idx'] = idx
    else:
        r_item = {
            'body_idx': body_idx,
            'pole': pole_n.astype('f4'),
            'inner_r': inner_r_au,
            'outer_r': outer_r_au,
            'opacity': float(custom_params.get('opacity', suggested_props['opacity'])),
            'scatter': float(suggested_props['scatter']),
            'asymmetry': float(suggested_props['asymmetry']),
            'backscatter': float(suggested_props['backscatter']),
            'unlit_factor': float(suggested_props['unlit_factor']),
            'shadow_grad': shadow_grad,
            'color': raw_col,
            'raw_color': raw_col,
            'gradient': [],
            'tex_sampled': sampled_arr,
            '5colors': colors5,
            'is_textured': True,
            'row_idx': len(ring_precomputed),
            'saturation': 1.0,
            'hue_shift': 0.0,
            'brightness': 1.0,
            'alpha_boost': 1.0
        }
        ring_precomputed.append(r_item)

    # Rebuild GPU buffers
    if ring_render_groups is not None and ring_gradient_tex is not None and prog_rings is not None and ctx is not None:
        app.body_ring_indices = rebuild_ring_render_group(
            body_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex
        )

    # Update body metadata
    r_hex = '#%02x%02x%02x' % (min(255, max(0, int(raw_col[0]*255))),
                               min(255, max(0, int(raw_col[1]*255))),
                               min(255, max(0, int(raw_col[2]*255))))
    body_info['ring_texture_inner'] = round(float(inner_r_au / body_r_au), 5)
    body_info['ring_texture_outer'] = round(float(outer_r_au / body_r_au), 5)
    body_info['rings'] = [{
        "inner": body_info['ring_texture_inner'],
        "outer": body_info['ring_texture_outer'],
        "color": r_hex,
        "opacity": round(float(r_item['opacity']), 5),
        "scatter": round(float(r_item['scatter']), 5),
        "asymmetry": round(float(r_item['asymmetry']), 5),
        "backscatter": round(float(r_item['backscatter']), 5),
        "gradient": []
    }]

    if bake_to_disk:
        base_textures = get_external_path('textures')
        system_name = getattr(app, 'loaded_system_name', 'Solar System') or 'Solar System'

        cand_sys_body = os.path.join(base_textures, system_name, body_name)
        cand_solar_body = os.path.join(base_textures, 'Solar System', body_name)
        cand_legacy_body = os.path.join(base_textures, body_name)

        if os.path.exists(cand_sys_body):
            textures_dir = cand_sys_body
        elif os.path.exists(cand_solar_body):
            textures_dir = cand_solar_body
        elif os.path.exists(cand_legacy_body):
            textures_dir = cand_legacy_body
        else:
            textures_dir = os.path.join(base_textures, system_name, body_name)

        os.makedirs(textures_dir, exist_ok=True)
        ring_path = os.path.join(textures_dir, "Rings.png")
        img_ring.save(ring_path)
        print(f"[Procedural Rings] Baked and saved procedural ring texture to {ring_path}")

    return True
