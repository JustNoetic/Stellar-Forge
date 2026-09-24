import os
import numpy as np
from PIL import Image
import moderngl
from engine.rendering.render_utils import generate_ring_shadow_grad, rebuild_ring_render_group, apply_hsba_np

def bake_and_export_ring_textures(app, body_name, ring_item, ring_precomputed=None, ring_render_groups=None, ring_gradient_tex=None):
    name_lower = body_name.lower()

    root_dir = getattr(app, 'root_dir', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_textures = os.path.join(root_dir, 'textures')

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
