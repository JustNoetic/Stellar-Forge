"""Physical scattering for the comparator's parallel rays and white key light."""
import math

import moderngl
import numpy as np
from OpenGL import GL

from engine.physics.atmosphere_physics import compute_mie_coefficients
from engine.rendering.planetshine import get_cached_atmosphere_properties
from engine.rendering.shader_loader import load_shader


class ComparatorAtmospheres:
    def __init__(self, ctx):
        self.ctx = ctx
        self.quad = ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1], dtype='f4').tobytes())
        self.program = ctx.program(vertex_shader=load_shader('atmosphere/comparator_atmo.vert'),
                                   fragment_shader=load_shader('atmosphere/comparator_atmo.frag'))
        self.vao = ctx.vertex_array(self.program, [(self.quad, '2f', 'in_position')])
        self.lut_program = ctx.program(vertex_shader=load_shader('atmosphere/atmo_lut.vert'),
                                       fragment_shader=load_shader('atmosphere/atmo_lut.frag'))
        self.multi_program = ctx.program(vertex_shader=load_shader('atmosphere/atmo_lut.vert'),
                                         fragment_shader=load_shader('atmosphere/multi_scatter_lut.frag'))
        self.lut_vao = ctx.vertex_array(self.lut_program, [(self.quad, '2f', 'in_position')])
        self.multi_vao = ctx.vertex_array(self.multi_program, [(self.quad, '2f', 'in_position')])
        self.program['u_transmittance_lut'].value = 5
        self.program['u_multi_scatter_lut'].value = 6
        self.program['u_scene_depth'].value = 7
        self.multi_program['u_transmittance_lut'].value = 5
        self.cache = {}
        self.depth_texture = self.depth_fbo = None
        self.bake_count = 0

    @staticmethod
    def parameters(atmo, mass, radius_km, albedo):
        # The visible mesh may use an edited/equatorial radius. Align the shell
        # and gravity to it without mutating the simulation's atmosphere state.
        record = dict(atmo, planet_radius_km=radius_km)
        props, _, _ = get_cached_atmosphere_properties(record, mass)
        height = max(0.0, float(atmo.get('height', props['atmo_height_km'])))
        vector = lambda x: tuple(map(float, np.broadcast_to(x, (3,))))
        return {
            'u_planet_radius_km': radius_km,
            'u_atmo_radius_km': radius_km + height,
            'u_h_rayleigh': max(float(props['scale_height_km']), 1e-4),
            'u_h_mie': max(float(atmo.get('h_mie', 1.2)), 1e-4),
            'u_beta_rayleigh': vector(props['beta_rayleigh']),
            'u_beta_mie': vector(compute_mie_coefficients(atmo.get('beta_mie', 2e-6), atmo.get('mie_angstrom'))),
            'u_beta_abs_mixed': vector(props['beta_abs_mixed']),
            'u_beta_abs_layered': vector(props['beta_abs_layered']),
            'u_mie_albedo': vector(atmo.get('mie_albedo', 1.0)),
            'u_ozone_peak_km': float(props.get('ozone_peak_km', 25.0)),
            'u_ozone_width_km': max(float(props.get('ozone_width_km', 8.0)), 1e-3),
            'u_mie_g': max(-0.99, min(0.99, float(atmo.get('mie_g', 0.8)))),
            'u_ground_albedo': vector(albedo),
        }

    @staticmethod
    def upload(program, values):
        for name, value in values.items():
            uniform = program.get(name, None)
            if uniform is not None:
                uniform.value = value

    def tables(self, bi, values):
        key = tuple(values.items())
        existing = self.cache.get(bi)
        if existing is not None and existing[0] == key:
            return existing[1:]
        if existing is not None:
            existing[1].release()
            existing[2].release()
        ctx = self.ctx
        previous_fbo, previous_viewport = ctx.fbo, ctx.viewport
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        trans = ctx.texture((256,256), 4, dtype='f4')
        multi = ctx.texture((64,64), 4, dtype='f4')
        for tex in (trans,multi):
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
        fbo = ctx.framebuffer([trans])
        fbo.use()
        ctx.viewport = (0,0,256,256)
        self.upload(self.lut_program, values)
        self.lut_vao.render(moderngl.TRIANGLE_STRIP)
        multi_fbo = ctx.framebuffer([multi])
        multi_fbo.use()
        ctx.viewport = (0,0,64,64)
        trans.use(location=5)
        self.upload(self.multi_program, values)
        self.multi_vao.render(moderngl.TRIANGLE_STRIP)
        previous_fbo.use()
        ctx.viewport = previous_viewport
        fbo.release()
        multi_fbo.release()
        self.cache[bi] = (key,trans,multi)
        self.bake_count += 1
        return trans,multi

    def render(self, app, bodies, visuals, positions, radii, view, projection, pole,
               depth_scale, atmospheres, masses):
        quality = int(app.camera.get('atmo_quality', 2))
        if not app.camera.get('atmo_enabled', True) or quality == 0:
            return 0
        ctx = self.ctx
        w,h = app.fb_width,app.fb_height
        half_h = view['height']/2
        half_w = half_h*w/max(h,1)
        candidates = []
        active = set()
        for atmo in atmospheres:
            bi = atmo['body_idx']
            if not 0 <= bi < len(bodies) or bodies[bi].get('type') in ('Star','BlackHole'):
                continue
            active.add(bi)
            radius_km = float(radii[bi]/1e-5)
            extent = radii[bi] + max(float(atmo.get('height',0)),0)*1e-5
            center = positions[bi] - view['center']
            if radius_km <= 0 or abs(center[0]) > half_w+extent or abs(center[1]) > half_h+extent:
                continue
            if extent*h/view['height'] < 0.5 or float(atmo.get('surface_pressure',1)) <= 0:
                continue
            values = self.parameters(atmo, float(masses[bi]), radius_km,
                                     np.clip(visuals[bi,:3],0,1)**2.2)
            if values['u_atmo_radius_km'] <= radius_km:
                continue
            candidates.append((bi,atmo,center,values))
        for bi in set(self.cache)-active:
            _,trans,multi = self.cache.pop(bi)
            trans.release(); multi.release()
        if not candidates:
            return 0
        # Copy depth only; sampling the draw framebuffer's own depth would be
        # an OpenGL feedback loop. DSA blit leaves ModernGL's bindings intact.
        if self.depth_texture is None or self.depth_texture.size != (w,h):
            if self.depth_fbo is not None:
                self.depth_fbo.release(); self.depth_texture.release()
            self.depth_texture = ctx.depth_texture((w,h))
            self.depth_texture.compare_func = ''
            self.depth_texture.filter = (moderngl.NEAREST,moderngl.NEAREST)
            self.depth_fbo = ctx.framebuffer(depth_attachment=self.depth_texture)
        target = ctx.fbo
        GL.glBlitNamedFramebuffer(target.glo,self.depth_fbo.glo,0,0,w,h,0,0,w,h,
                                  GL.GL_DEPTH_BUFFER_BIT,GL.GL_NEAREST)
        self.depth_texture.use(location=7)
        p = self.program
        p['u_projection'].write(projection.T.tobytes())
        p['u_pole'].value = tuple(map(float,pole))
        p['u_depth_scale'].value = float(depth_scale)
        p['u_viewport'].value = (float(w),float(h))
        p['u_pixel_km'].value = view['height']/h/1e-5
        p['u_steps'].value = 16 if quality == 1 else (48 if quality == 2 else 64)
        for bi,atmo,center,values in candidates:
            trans,multi = self.tables(bi,values)
            trans.use(location=5); multi.use(location=6)
            self.upload(p,values)
            p['u_center'].value = (float(center[0]),float(center[1]),0.0)
            p['u_oblateness'].value = min(0.95,max(0.0,float(visuals[bi,8])))
            p['u_intensity'].value = math.pi*max(0.0,float(atmo.get('intensity',1)))
            ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            ctx.enable(moderngl.BLEND)
            ctx.depth_mask = False
            # Dual-source blend: scattering + per-channel extinction * scene.
            ctx.blend_func = (moderngl.ONE,int(GL.GL_SRC1_COLOR),moderngl.ZERO,moderngl.ONE)
            self.vao.render(moderngl.TRIANGLE_STRIP)
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.blend_func = (moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
        return len(candidates)*2

    def release(self):
        for _,trans,multi in self.cache.values():
            trans.release(); multi.release()
        if self.depth_fbo is not None:
            self.depth_fbo.release(); self.depth_texture.release()
        for obj in (self.vao,self.lut_vao,self.multi_vao,self.program,
                    self.lut_program,self.multi_program,self.quad):
            obj.release()
