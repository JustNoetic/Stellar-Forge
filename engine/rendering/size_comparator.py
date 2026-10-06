"""Orthographic size scene. Distances here are km * 1e-5, never orbital AU."""
import math

import imgui
import moderngl
import numpy as np

from engine.core.constants import AU_TO_KM
from engine.ephemeris.system_manager import temperature_to_rgb
from engine.rendering.shader_loader import load_shader
from engine.rendering.comparator_atmosphere import ComparatorAtmospheres

SIZE_SCALE = AU_TO_KM * 1e-5
DISPLAY_POLE = np.array([-0.48989795, 0.84852814, 0.2], dtype='f4')


def comparator_view(camera):
    return camera.setdefault('size_comparator_view', {
        'center': [0.0, 0.0], 'height': 20.0, 'fit': True,
    })


def screen_to_world(view, x, y, width, height):
    scale = view['height'] / max(height, 1)
    return (view['center'][0] + (x - width / 2) * scale,
            view['center'][1] - (y - height / 2) * scale)


def pan_view(view, dx, dy, height):
    scale = view['height'] / max(height, 1)
    view['center'][0] -= dx * scale
    view['center'][1] += dy * scale


def zoom_view(view, steps, x, y, width, height):
    before = screen_to_world(view, x, y, width, height)
    view['height'] = max(1e-8, min(1e12, view['height'] * math.exp(-max(-100, min(100, steps)) * 0.15)))
    after = screen_to_world(view, x, y, width, height)
    view['center'][0] += before[0] - after[0]
    view['center'][1] += before[1] - after[1]


def arrange_bodies(bodies, radii_au, parents, rings):
    """HTML baseline: star + orbital-order planets, satellite columns below.

    Radii all arrive in AU (including moons); one scale applies to every body.
    Traversal also covers starless systems and nested satellites exactly once.
    """
    radii = np.maximum(np.asarray(radii_au, dtype='f8'), 0) * SIZE_SCALE
    stars = {i for i, b in enumerate(bodies) if b.get('type') in ('Star', 'BlackHole')}
    children = {i: [] for i in range(len(bodies))}
    for i, parent in enumerate(parents):
        if 0 <= parent < len(bodies) and parent != i and i not in stars:
            children[int(parent)].append(i)
    def orbital_order(i):
        return (float(bodies[i].get('a', 0.0)), i)
    for items in children.values():
        items.sort(key=orbital_order)
    outer = radii.copy()
    for ring in rings:
        bi = ring['body_idx']
        if 0 <= bi < len(bodies):
            outer[bi] = max(outer[bi], ring['outer_r'] * SIZE_SCALE)
    positions = np.zeros((len(bodies), 2), dtype='f8')
    visited = set()
    bottom = None
    roots = sorted(stars) + [i for i in range(len(bodies)) if i not in stars and
                            not (0 <= parents[i] < len(bodies) and parents[i] != i)]
    for root in roots + list(range(len(bodies))):
        if root in visited:
            continue
        row = [root] + children[root]
        # Keep rings readable at the HTML's 0.2 projected aspect ratio.
        row_top = max((max(radii[i], outer[i] * 0.52) for i in row), default=0)
        y = 0.0 if bottom is None else bottom - 2.0 - row_top
        x = 0.0
        row_bottom = y
        for i in row:
            if i in visited:
                continue
            spacing = radii[i] + (outer[i] - radii[i]) * 0.25
            x += 0.25 + spacing
            positions[i] = (x, y)
            visited.add(i)
            depth = max(radii[i], outer[i] * 0.52)
            moon_y = y - depth - 0.15
            # Depth-first order preserves orbital sorting for nested moons.
            pending = list(children[i]) if i != root else []
            while pending:
                moon = pending.pop(0)
                if moon in visited:
                    continue
                extent = max(radii[moon], outer[moon] * 0.52)
                moon_y -= extent
                positions[moon] = (x, moon_y)
                visited.add(moon)
                moon_y -= extent + 0.125
                pending[0:0] = children[moon]
            row_bottom = min(row_bottom, y - depth, moon_y)
            x += spacing + 0.25
        bottom = row_bottom
    if len(bodies):
        bounds = (float(np.min(positions[:, 0] - outer)),
                  float(np.min(positions[:, 1] - np.maximum(radii, outer * 0.52))),
                  float(np.max(positions[:, 0] + outer)),
                  float(np.max(positions[:, 1] + np.maximum(radii, outer * 0.52))))
    else:
        bounds = (-1.0, -1.0, 1.0, 1.0)
    return positions, radii, bounds


class SizeComparatorRenderer:
    def __init__(self, ctx, sphere_vbo, sphere_ibo):
        self.ctx = ctx
        self.sphere = ctx.program(vertex_shader=load_shader('celestial/comparator.vert'),
                                  fragment_shader=load_shader('celestial/comparator.frag'))
        self.sphere_vao = ctx.vertex_array(self.sphere,
            [(sphere_vbo, '3f 3f', 'in_position', 'in_normal')], sphere_ibo)
        self.ring = ctx.program(vertex_shader=load_shader('celestial/comparator_ring.vert'),
                                fragment_shader=load_shader('celestial/comparator_ring.frag'))
        vertices = [(math.cos(a), edge, math.sin(a)) for a in
                    np.linspace(0, math.tau, 257) for edge in (0, 1)]
        self.ring_vbo = ctx.buffer(np.array(vertices, dtype='f4').tobytes())
        self.ring_vao = ctx.vertex_array(self.ring, [(self.ring_vbo, '3f', 'in_position')])
        self.ring['u_profile'].value = 0
        self.sphere['u_diffuse'].value = 1
        self.sphere['u_clouds'].value = 2
        self.sphere['u_normal_map'].value = 3
        self.layout_key = None
        self.profile_key = None
        self.profile_texture = None
        self.atmospheres = None

    def render(self, app, bodies, visuals, parents, tex_indices, spins, rings, atlas,
               atmospheres=(), masses=(), body_sources=None):
        ctx = self.ctx
        view = comparator_view(app.camera)
        key = tuple((b.get('name'), b.get('type'), int(p)) for b, p in zip(bodies, parents))
        if body_sources is not None:
            key = (tuple(body_sources),key)
        if key != self.layout_key:
            view['fit'] = True
            self.layout_key = key
        positions, radii, bounds = arrange_bodies(bodies, visuals[:, 3], parents, rings)
        w, h = max(app.window_width, 1), max(app.window_height, 1)
        from engine.ui.workspace import comparator_bounds
        left, right_edge, top, bottom = comparator_bounds(app, w, h)
        right = w - right_edge
        # Reserve room for labels, menu, navigation pill, and time transport.
        avail_w, avail_h = max(w * 0.3, right_edge - left), max(h * 0.4, bottom - top)
        if view.pop('fit', False):
            x0, y0, x1, y1 = bounds
            view['height'] = max(y1 - y0, (x1 - x0) * avail_h / avail_w, 0.01) * h / avail_h * 1.15
            scale = view['height'] / h
            view['center'] = [(x0 + x1) / 2 - (left - right) / 2 * scale,
                              (y0 + y1) / 2 + (top - (h - bottom)) / 2 * scale]
        focus = app.camera.pop('size_comparator_focus', None)
        if focus is not None and 0 <= focus < len(bodies):
            view['height'] = max(radii[focus] * 5.0, 1e-6) * h / avail_h
            scale = view['height'] / h
            view['center'] = [positions[focus, 0] - (left - right) / 2 * scale,
                              positions[focus, 1] + (top - (h - bottom)) / 2 * scale]
        if app.pick_request is not None:
            px, py = app.pick_request
            app.pick_request = None
            point = np.array(screen_to_world(view, px, py, w, h))
            distance = np.linalg.norm(positions - point, axis=1)
            candidates = np.flatnonzero(distance <= np.maximum(radii, view['height'] / h * 7))
            if len(candidates):
                bi = int(candidates[np.argmin(distance[candidates])])
                if body_sources is not None:
                    name,local_index,role = body_sources[bi]
                    app.camera['size_comparator_selected'] = [name,local_index]
                    app.camera['inspected_idx'] = local_index if role != 'saved' else None
                    app.camera['inspected_is_cmp'] = role == 'comparison'
                    app.camera['edit_mode'] = False
                    app.camera['inspect_bary'] = False
                else:
                    app.camera['inspected_idx'] = bi
                    app.camera['inspected_is_cmp'] = bi >= app._comparator_primary_count
                    if app.camera['inspected_is_cmp']:
                        app.camera['inspected_idx'] -= app._comparator_primary_count
        ctx.viewport = (0, 0, app.fb_width, app.fb_height)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND | moderngl.CULL_FACE)
        ctx.depth_mask = True
        ctx.depth_func = '<='
        ctx.fbo.clear(0.003, 0.005, 0.012, 1.0, depth=1.0)
        half_h = view['height'] / 2
        half_w = half_h * app.fb_width / max(app.fb_height, 1)
        depth = max(float(np.max(radii)) * 4, 1.0) if len(radii) else 1.0
        # Actual orthographic projection; positive Z faces the observer.
        proj = np.array([[1/half_w, 0, 0, 0], [0, 1/half_h, 0, 0],
                         [0, 0, -1/depth, 0], [0, 0, 0, 1]], dtype='f4')
        for prog in (self.sphere, self.ring):
            prog['u_projection'].write(proj.T.tobytes())
            prog['u_pole'].value = tuple(DISPLAY_POLE)
        pixels = h / view['height']
        visible = []
        for bi, body in enumerate(bodies):
            center = positions[bi] - view['center']
            radius = float(radii[bi])
            extent = max(radius, max((r['outer_r'] * SIZE_SCALE for r in rings if r['body_idx'] == bi), default=0))
            if abs(center[0]) > half_w + extent or abs(center[1]) > half_h + extent or radius * pixels < 0.1:
                continue
            visible.append(bi)
            p = self.sphere
            p['u_center'].value = (float(center[0]), float(center[1]), 0.0)
            p['u_radius'].value = radius
            p['u_oblateness'].value = min(0.95, max(0.0, float(visuals[bi, 8])))
            is_star = body.get('type') == 'Star'
            p['u_is_star'].value = is_star
            p['u_black_hole'].value = body.get('type') == 'BlackHole'
            color = visuals[bi, :3]
            if is_star:
                temp = body.get('star_props', {}).get('temp', body.get('temp'))
                if temp is not None and float(temp) > 0:
                    color = temperature_to_rgb(float(temp))
            p['u_color'].value = tuple(float(c) for c in color)
            p['u_spin'].value = float(spins[bi])
            textures = app.gpu_body_textures.get(int(tex_indices[bi]), {})
            p['u_textured'].value = bool(textures.get('diffuse')) and not is_star
            p['u_has_clouds'].value = bool(textures.get('clouds')) and app.camera.get('clouds_enabled', True)
            p['u_has_normal'].value = bool(textures.get('normal')) and not is_star
            for unit, name in ((1, 'diffuse'), (2, 'clouds'), (3, 'normal')):
                if textures.get(name):
                    textures[name].use(location=unit)
            self.sphere_vao.render(moderngl.TRIANGLES)
        # Own profile atlas also covers an enabled comparison system. Its
        # segment rows need not match the simulation scene's primary atlas.
        profile_key = (atlas.profile_revision, tuple((id(r['shadow_grad']),r['opacity']) for r in rings))
        if rings and profile_key != self.profile_key:
            if self.profile_texture is not None:
                self.profile_texture.release()
            profiles = np.stack([r['shadow_grad'] for r in rings]).astype('f4')
            # Filter premultiplied linear radiance and effective alpha, rather
            # than averaging optical depth and converting after minification.
            # Array layers keep neighboring rings isolated at every mip level.
            opacity = np.array([r['opacity'] for r in rings],dtype='f4')[:,None]
            profiles[:,:,3] = -np.expm1(-np.maximum(0,profiles[:,:,3]*opacity)/abs(DISPLAY_POLE[2]))
            profiles[:,:,:3] = np.maximum(profiles[:,:,:3],0)**2.2*profiles[:,:,3,None]
            self.profile_texture = ctx.texture_array((profiles.shape[1],1,len(rings)), 4, profiles.tobytes(), dtype='f4')
            self.profile_texture.build_mipmaps()
            self.profile_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.profile_texture.anisotropy = min(16.0,float(ctx.max_anisotropy or 1))
            self.profile_texture.repeat_x = self.profile_texture.repeat_y = False
            self.profile_key = profile_key
        if rings:
            self.profile_texture.use(location=0)
            self.ring['u_pixel_size'].value = view['height']/app.fb_height
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        ctx.depth_mask = False
        ring_count = 0
        front_rings = []
        atmo_enabled = app.camera.get('atmo_enabled',True) and app.camera.get('atmo_quality',2) != 0
        atmo_by_body = {a['body_idx']:a for a in atmospheres} if atmo_enabled else {}
        for row, ring in enumerate(rings):
            bi = ring['body_idx']
            if bi not in visible:
                continue
            center = positions[bi] - view['center']
            p = self.ring
            p['u_center'].value = (float(center[0]), float(center[1]), 0.0)
            p['u_inner'].value = float(ring['inner_r'] * SIZE_SCALE)
            p['u_outer'].value = float(ring['outer_r'] * SIZE_SCALE)
            p['u_layer'].value = row
            atmo = atmo_by_body.get(bi)
            split = atmo is not None and float(atmo.get('height',0)) > 0 and float(atmo.get('surface_pressure',1)) > 0
            p['u_atmo_pass'].value = 1 if split else 0
            if split:
                p['u_atmo_radius'].value = float(radii[bi]+atmo['height']*1e-5)
                p['u_atmo_oblateness'].value = min(0.95,max(0.0,float(visuals[bi,8])))
                front_rings.append((row,ring))
            self.ring_vao.render(moderngl.TRIANGLE_STRIP)
            ring_count += 1
        ctx.depth_mask = True
        ctx.depth_func = '<'
        ctx.disable(moderngl.BLEND)
        atmo_triangles = 0
        if atmospheres and app.camera.get('atmo_enabled',True) and app.camera.get('atmo_quality',2) != 0:
            if self.atmospheres is None:
                self.atmospheres = ComparatorAtmospheres(ctx)
            atmo_triangles = self.atmospheres.render(app,bodies,visuals,positions,radii,
                view,proj,DISPLAY_POLE,depth,atmospheres,masses)
        if front_rings:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE,moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.depth_mask = False
            self.profile_texture.use(location=0)
            for row,ring in front_rings:
                bi = ring['body_idx']
                center = positions[bi]-view['center']
                p = self.ring
                p['u_center'].value = (float(center[0]),float(center[1]),0.0)
                p['u_inner'].value = float(ring['inner_r']*SIZE_SCALE)
                p['u_outer'].value = float(ring['outer_r']*SIZE_SCALE)
                p['u_layer'].value = row
                p['u_atmo_radius'].value = float(radii[bi]+atmo_by_body[bi]['height']*1e-5)
                p['u_atmo_oblateness'].value = min(0.95,max(0.0,float(visuals[bi,8])))
                p['u_atmo_pass'].value = 2
                self.ring_vao.render(moderngl.TRIANGLE_STRIP)
                ring_count += 1
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)
        app.terrain_last_triangle_count = 0
        app.terrain_last_patch_count = 0
        app.terrain_last_cloud_patch_count = 0
        app.sphere_last_triangle_count = len(visible) * self.sphere_vao.vertices // 3
        app.total_last_triangle_count = app.sphere_last_triangle_count + ring_count * 512 + atmo_triangles
        ctx.blend_func = (moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
        extents = radii.copy()
        for ring in rings:
            extents[ring['body_idx']] = max(extents[ring['body_idx']], ring['outer_r'] * SIZE_SCALE * 0.52)
        self.labels(app, bodies, positions, radii, extents, parents, pixels,body_sources)

    @staticmethod
    def labels(app, bodies, positions, radii, extents, parents, pixels,body_sources=None):
        if not app.ui_visible:
            return
        view = comparator_view(app.camera)
        draw = imgui.get_background_draw_list()
        white = imgui.get_color_u32_rgba(0.87, 0.91, 1.0, 1.0)
        muted = imgui.get_color_u32_rgba(0.48, 0.56, 0.68, 1.0)
        occupied = []
        from engine.ui.workspace import comparator_bounds
        left_margin, right_edge, top, bottom = comparator_bounds(app, app.window_width, app.window_height)
        if body_sources is not None:
            from collections import Counter
            duplicate_names = Counter(b.get('name','Body') for b in bodies)
        # Largest objects get first choice of label space when zoomed out.
        for bi in sorted(range(len(bodies)), key=lambda i: -radii[i]):
            body = bodies[bi]
            r = float(radii[bi]) * pixels
            if r < 2.0:
                continue
            x = (positions[bi, 0] - view['center'][0]) * pixels + app.window_width / 2
            y = -(positions[bi, 1] - view['center'][1]) * pixels + app.window_height / 2
            if x + r < 0 or x - r > app.window_width or y + r < 0 or y - r > app.window_height:
                continue
            name = body.get('name', 'Body')
            if body_sources is not None and duplicate_names[name] > 1:
                name += f' ({body_sources[bi][0]})'
            star = body.get('type') in ('Star', 'BlackHole')
            parent = int(parents[bi])
            moon = 0 <= parent < len(bodies) and bodies[parent].get('type') not in ('Star', 'BlackHole')
            tx = x - r - 16 - imgui.calc_text_size(name)[0] if star else (
                x + r + 8 if moon else x - imgui.calc_text_size(name)[0] / 2)
            ty = y - 16 if star or moon else y - float(extents[bi]) * pixels - 35
            text = f"{radii[bi] / 1e-5:,.0f} km radius"
            rx = x - r - 16 - imgui.calc_text_size(text)[0] if star else (
                x + r + 8 if moon else x - imgui.calc_text_size(text)[0] / 2)
            if star and min(tx, rx) < left_margin:
                tx = x - imgui.calc_text_size(name)[0] / 2
                rx = x - imgui.calc_text_size(text)[0] / 2
                ty = y - r - 35
            box = (min(tx,rx)-6,ty-5,
                   max(tx+imgui.calc_text_size(name)[0],rx+imgui.calc_text_size(text)[0])+6,ty+34)
            if box[0] < left_margin or box[2] > right_edge or box[1] < top or box[3] > bottom:
                continue
            if any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in occupied):
                continue
            occupied.append(box)
            draw.add_text(tx, ty, white, name)
            draw.add_text(rx, ty + 16, muted, text)

    def release(self):
        if self.atmospheres is not None:
            self.atmospheres.release()
        if self.profile_texture is not None:
            self.profile_texture.release()
        for obj in (self.sphere_vao, self.ring_vao, self.ring_vbo, self.sphere, self.ring):
            obj.release()
