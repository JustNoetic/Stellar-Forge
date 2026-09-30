"""GPU endpoint scattering versus independent dense numerical integration.

Run with venv/Scripts/python.exe scripts/test_scattering_segment.py.
Includes short ground rays, elevated terrain above the reference horizon,
below-datum terrain, twilight, colored absorption, and camera motion caching.
"""
import os
import sys
import time

import moderngl
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.rendering.scattering_lut import ScatteringLUTCache
from engine.rendering.shader_loader import load_shader
from engine.rendering.shaders import atmo_vertex_shader, atmo_fragment_shader

PARAMETERS = {
    'u_planet_radius_km': 6371., 'u_atmo_radius_km': 6471.,
    'u_scattering_bottom_km': 6368.95, 'u_scattering_sun_radius': 0.005,
    'u_h_rayleigh': 8., 'u_h_mie': 1.2,
    'u_beta_rayleigh': (5.8e-6, 13.5e-6, 33.1e-6),
    'u_beta_mie': (2e-6, 2e-6, 2e-6), 'u_mie_albedo': (0.9, 0.95, 1.),
    'u_beta_abs_mixed': (0., 0., 0.),
    'u_beta_abs_layered': (0.6e-6, 1.4e-6, 0.1e-6),
    'u_ozone_peak_km': 25., 'u_ozone_width_km': 8., 'u_mie_g': 0.76,
}
PSI = np.array([.015, .02, .025])


def medium(radius):
    h = np.maximum(0., radius - PARAMETERS['u_planet_radius_km'])
    rho_R = np.exp(-h / PARAMETERS['u_h_rayleigh'])[..., None]
    rho_M = np.exp(-h / PARAMETERS['u_h_mie'])[..., None]
    rho_O = np.exp(-((h - 25.) / 8.) ** 2)[..., None]
    R = np.array(PARAMETERS['u_beta_rayleigh']) * 1000 * rho_R
    Mext = np.array(PARAMETERS['u_beta_mie']) * 1000 * rho_M
    M = Mext * PARAMETERS['u_mie_albedo']
    ext = R + Mext + np.array(PARAMETERS['u_beta_abs_layered']) * 1000 * rho_O
    return R, M, ext


def sunlight(points, sun):
    r = np.linalg.norm(points, axis=1)
    mus = points @ sun / r
    sin_p = 6371. / np.maximum(r, 6371.01)
    cos_p = np.sqrt(np.maximum(0., 1. - sin_p * sin_p))
    eff = PARAMETERS['u_scattering_sun_radius']
    cos_eff = np.sqrt(1. - eff * eff)
    outer = cos_p * cos_eff - sin_p * eff
    inner = cos_p * cos_eff + sin_p * eff
    x = np.clip(((cos_p * cos_eff + mus) / (sin_p * eff) + 1.) * .5, 0., 1.)
    visible = x * x * (3. - 2. * x)
    top_cos = mus * cos_eff + np.sqrt(np.maximum(0., 1. - mus * mus)) * eff
    bottom_cos = np.maximum(mus * cos_eff - np.sqrt(np.maximum(0., 1. - mus * mus)) * eff,
                            -cos_p + 1e-5)
    effective_mu = np.where(-mus <= outer, mus,
                    np.where(-mus >= inner, np.maximum(-cos_p + 1e-5, mus),
                             np.maximum(-cos_p + 1e-5, .5 * (top_cos + bottom_cos))))
    b = r * effective_mu
    d = -b + np.sqrt(np.maximum(0., b * b + 6471.**2 - r * r))
    closest = np.clip(-b, 0., d)
    # Independent dense midpoint integration of sunlight's curved column.
    t = (np.arange(256) + .5) / 256
    stations = d[:, None] * t
    rq = np.sqrt(r[:, None]**2 + stations * (2 * b[:, None] + stations))
    tau = medium(rq)[2].sum(axis=1) * (d / 256)[:, None]
    return np.exp(-tau) * visible[:, None]


def reference(a, b, sun):
    a, b, sun = np.array(a, float), np.array(b, float), np.array(sun, float)
    sun /= np.linalg.norm(sun)
    distance = np.linalg.norm(b - a)
    v = (b - a) / distance
    n = 4096
    ds = distance / n
    points = a + ((np.arange(n) + .5) * ds)[:, None] * v
    R, M, ext = medium(np.linalg.norm(points, axis=1))
    cell_T = np.exp(-ext * ds)
    factor = -np.expm1(-ext * ds) / np.maximum(ext, 1e-30)
    T_before = np.exp(-(np.cumsum(ext, axis=0) - ext) * ds)
    nu = v @ sun
    phase_R = 3 / (16 * np.pi) * (1 + nu * nu)
    g = PARAMETERS['u_mie_g']
    phase_M = 3 / (8 * np.pi) * (1 - g * g) / (2 + g * g) * (1 + nu * nu) / (1 + g * g - 2 * g * nu)**1.5
    source = (R * phase_R + M * phase_M) * sunlight(points, sun) + (R + M) * PSI
    return np.exp(-np.sum(ext, axis=0) * ds), np.sum(T_before * factor * source, axis=0)


def test_production_depth(ctx, tables, multiple, endpoint_query):
    """Execute the full atmosphere shader with rasterized log-depth inputs."""
    fragment = atmo_fragment_shader.replace(
        'layout(location = 0, index = 0) out vec4 out_scattered;',
        'layout(location = 0) out vec4 out_scattered;').replace(
        'layout(location = 0, index = 1) out vec4 out_transmittance;',
        'layout(location = 1) out vec4 out_transmittance;')
    vertex = '''#version 460
out vec3 f_local_pos; out float f_clip_z;
uniform vec3 u_proxy_local;
void main() { vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
gl_Position=vec4(p*2.-1.,0,1); f_local_pos=u_proxy_local; f_clip_z=1.; }
'''
    program = ctx.program(vertex_shader=vertex, fragment_shader=fragment)
    color = [ctx.texture((1, 1), 4, dtype='f4') for _ in range(2)]
    fbo = ctx.framebuffer(color)
    vao = ctx.vertex_array(program, [])
    white = ctx.texture((1, 1), 4, np.ones(4, 'f4').tobytes(), dtype='f4')
    white3d = ctx.texture3d((1, 1, 1), 4, np.ones(4, 'f4').tobytes(), dtype='f4')
    depth = ctx.texture((1, 1), 1, dtype='f4')
    # Explicit units prevent inactive-branch 2D/3D sampler alias validation.
    for name in program:
        member = program[name]
        if getattr(member, 'gl_type', None) == 35678:  # sampler2D
            member.value = tuple([0] * member.array_length) if member.array_length > 1 else 0
        if getattr(member, 'gl_type', None) == 35679:  # sampler3D
            member.value = 19
    white.use(0); white3d.use(19); depth.use(9); multiple.use(3)
    ScatteringLUTCache.bind(tables)
    for name, unit in [('u_depth_texture',9), ('u_multi_scatter_lut',3),
                       ('u_scattering_tau_lut',20), ('u_scattering_rayleigh_lut',21),
                       ('u_scattering_mie_lut',22), ('u_scattering_multiple_lut',23)]:
        program[name].value = unit
    program['u_scattering_enabled'].value = True
    program['u_scattering_bottom_km'].value = PARAMETERS['u_scattering_bottom_km']
    program['u_scattering_sun_radius'].value = PARAMETERS['u_scattering_sun_radius']
    program['u_scattering_azimuth_count'].value = 8
    program['u_atmo_quality'].value = 3
    program['u_screen_res'].value = (1, 1)
    program['u_inv_proj'].write(np.eye(4, dtype='f4').tobytes())
    au = 149597870.7
    scene = np.zeros(1832, 'f4'); scene[:16] = np.eye(4, dtype='f4').ravel()
    scene.view('i4')[32] = 1; scene[292:294] = [100000 / au, au]
    scene_buffer = ctx.buffer(scene.tobytes()); scene_buffer.bind_to_uniform_block(1)
    instance = ctx.buffer(np.zeros(28, 'f4').tobytes()); instance.bind_to_storage_buffer(2)
    data = np.zeros(256, 'f4')
    data[3] = 6471 / au
    data[4:8] = [*PARAMETERS['u_beta_rayleigh'], 8]
    data[8:12] = [*PARAMETERS['u_beta_mie'], 1.2]
    data[15] = PARAMETERS['u_mie_g']
    data[16:20] = [*PARAMETERS['u_beta_abs_layered'], 1]
    data[20:23] = [6371, 6471, au]; data.view('i4')[23] = 128
    data[24:28] = [0,1,0,1]; data[32:35] = PARAMETERS['u_mie_albedo']
    data[180:184] = [25,8,6371,128]; data[184:188] = [1/8,1/1.2,1/8,0]
    data[192:196] = [1,1,1,.005]
    data[240:244] = [0,0,1e8 / au,5e5 / au]
    atmosphere = ctx.buffer(data.tobytes()); atmosphere.bind_to_storage_buffer(8)
    def draw(a, b, sun=(0,1,0), stars=1, secondary=None, quality=3, scene_depth=True):
        a, b, sun = np.array(a, float), np.array(b, float), np.array(sun, float)
        v = (b-a) / np.linalg.norm(b-a); sun /= np.linalg.norm(sun)
        right = np.cross(v, (0,0,1)); right /= np.linalg.norm(right)
        up = np.cross(right, v)
        rotation = np.eye(4, dtype='f4'); rotation[:3,:3] = np.column_stack((right, up, -v))
        scene[16:32] = rotation.T.T.ravel(); scene.view('i4')[32] = stars
        scene[36:40] = [*(sun * 1e8 / au), 5e5 / au]
        scene_buffer.write(scene.tobytes())
        program['u_inv_view'].write(rotation.T.tobytes())
        program['u_camera_pos'].value = tuple(a / au)
        # A point on the back face of the atmosphere proxy, used by the stable
        # distant-camera anchor path as well as near-ground ray reconstruction.
        bv = a @ v; exit_distance = -bv + np.sqrt(bv*bv + 6471**2 - a@a)
        program['u_proxy_local'].value = tuple((a + v * exit_distance) / 6471)
        data[208:212] = [*sun, np.sqrt(1-.005**2)]
        data[224:228] = [*(sun*1e8), .005]
        if stars == 2:
            data[196:200] = [1,1,1,.005]
            data[212:216] = [*secondary, np.sqrt(1-.005**2)]
            data[228:232] = [*(np.array(secondary)*1e8), .005]
            data[244:248] = [0,0,1e8 / au,5e5 / au]
        atmosphere.write(data.tobytes())
        d = np.linalg.norm(b-a)
        log_depth = np.log2(1+d) / np.log2(1+100000) if scene_depth else 1.
        depth.write(np.array([log_depth], 'f4').tobytes())
        program['u_atmo_quality'].value = quality
        fbo.use(); fbo.clear(); vao.render(moderngl.TRIANGLES, vertices=3)
        return [np.frombuffer(tex.read(), 'f4')[:3].copy() for tex in color]  # L, T
    for name, a, b in [
        ('nearby depth', (0,6371.1,0), (.1,6371.1,0)),
        ('mountain above horizon', (0,6371.1,0), (50,6378,0)),
        ('below-datum depth', (0,6370.,0), (2,6369.5,0)),
        ('space anchor', (0,40000.,0), (30,6375.,0)),
    ]:
        actual_L, actual_T = draw(a,b)
        # Outside-camera queries must start at atmosphere entry.
        qa, qb = np.array(a,float), np.array(b,float)
        if np.linalg.norm(qa) > 6471:
            v=(qb-qa)/np.linalg.norm(qb-qa); bv=qa@v
            qa += (-bv-np.sqrt(bv*bv+6471**2-qa@qa))*v
        expected_T, expected_L = endpoint_query(qa,qb,(0,1,0))
        np.testing.assert_allclose(actual_T, expected_T, atol=2e-4, rtol=.004)
        np.testing.assert_allclose(actual_L, expected_L, atol=2e-5, rtol=.015)
        assert np.max(actual_L) > 0, name
        print('Production shader:', name, 'passed')
    a,b=(0,6372.,0),(30,6375.,0)
    sun2=np.array([.8,1,.8]); sun2/=np.linalg.norm(sun2)
    L1,T1=draw(a,b)
    L2,T2=draw(a,b,sun=sun2)
    L12,T12=draw(a,b,stars=2,secondary=sun2)
    np.testing.assert_allclose(L12,L1+L2,rtol=.003,atol=1e-5)
    np.testing.assert_allclose(T12,T1,rtol=.001)
    # Each star's finite scattering is the cap for its own eclipse deficit.
    # A caster missing the terrain segment cannot remove a distant tail's light.
    data.view('i4')[28]=1
    data[36:40]=[15/au,6450/au,0,30/au]
    data[68:72]=[0,1,0,1]; data[100]=30/au
    program['u_ring_station_count'].value=24
    for method in range(3):
        program['u_atmo_shadow_method'].value=method
        shadow_L,shadow_T=draw(a,b)
        assert np.linalg.norm(shadow_L) < .3*np.linalg.norm(L1), (method,shadow_L,L1)
        np.testing.assert_allclose(shadow_T,T1,atol=1e-5)
        combined_L,_=draw(a,b,stars=2,secondary=sun2)
        assert np.all(combined_L >= L2*.95), (method,combined_L,L2)
    data[36]=200/au
    missed_L,_=draw(a,b)
    np.testing.assert_allclose(missed_L,L1,atol=1e-5,rtol=.003)
    data.view('i4')[28]=0
    print('Finite eclipse bounds and per-star shadow subtraction passed for all methods')
    # Spatially varying secondary light uses the existing integrator, with
    # exactly the same finite terrain endpoint, rather than losing that light.
    inst=np.zeros(28,'f4'); inst[16:19]=[0,1,0]; inst[20:23]=[.1,.2,.3]
    instance.write(inst.tobytes()); program['u_planetshine_enabled'].value=True
    L3,T3=draw(a,b,quality=3)
    Lref,Tref=draw(a,b,quality=2)
    np.testing.assert_allclose(L3,Lref,rtol=.001,atol=1e-5)
    np.testing.assert_allclose(T3,Tref,rtol=.001)
    # Pixels without local geometry retain the existing Sky-View output.
    instance.write(np.zeros(28,'f4').tobytes()); program['u_planetshine_enabled'].value=False
    Lsky,Tsky=draw((0,6372,0),(10,6380,0),scene_depth=False)
    np.testing.assert_allclose(Lsky,np.ones(3),atol=1e-6)
    np.testing.assert_allclose(Tsky,np.ones(3),atol=1e-6)
    print('Production multi-star, secondary-light fallback, and sky isolation passed')


def main():
    ctx = moderngl.create_standalone_context(require=460)
    # Compile the shipped full shader as well as executing its shared solver.
    ctx.program(vertex_shader=atmo_vertex_shader, fragment_shader=atmo_fragment_shader).release()
    multiple = ctx.texture((64, 64), 4, np.tile(np.r_[PSI, 1.].astype('f4'), (64 * 64, 1)).tobytes(), dtype='f4')
    multiple.filter = (moderngl.LINEAR, moderngl.LINEAR)
    cache = ScatteringLUTCache(ctx)
    before = time.perf_counter()
    tables = cache.get(PARAMETERS, multiple)
    ctx.finish()
    print(f'GPU LUT bake: {time.perf_counter() - before:.3f}s; renderer: {ctx.info["GL_RENDERER"]}')
    assert cache.get(PARAMETERS.copy(), multiple) is tables
    cache.bind(tables)
    multiple.use(3)
    vertex = '#version 460\nvoid main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position=vec4(p*2.-1.,0,1); }'
    declarations = '\n'.join('uniform vec3 ' + key + ';' if isinstance(value, tuple)
                              else 'uniform float ' + key + ';'
                              for key, value in PARAMETERS.items() if key not in ('u_scattering_bottom_km', 'u_scattering_sun_radius'))
    fragment = '#version 460\n#define PI 3.14159265358979323846\n' + declarations + '''
uniform sampler2D u_multi_scatter_lut;
uniform vec3 u_a, u_b, u_sun;
layout(location=0) out vec4 out_T;
layout(location=1) out vec4 out_L;
''' + load_shader('common/sun_terminator.glsl') + load_shader('common/scattering_segment.glsl') + '''
void main() { vec3 T=endpoint_transmittance(u_a,u_b);
out_T=vec4(T,1); out_L=vec4(endpoint_radiance(u_a,u_b,normalize(u_sun),T),1); }
'''
    program = ctx.program(vertex_shader=vertex, fragment_shader=fragment)
    for name, value in PARAMETERS.items():
        if name in program:
            program[name].value = value
    for name, unit in [('u_scattering_tau_lut',20), ('u_scattering_rayleigh_lut',21),
                       ('u_scattering_mie_lut',22), ('u_scattering_multiple_lut',23), ('u_multi_scatter_lut',3)]:
        program[name].value = unit
    program['u_scattering_azimuth_count'].value = cache.size[3]
    out = [ctx.texture((1, 1), 4, dtype='f4') for _ in range(2)]
    fbo = ctx.framebuffer(out); fbo.use()
    vao = ctx.vertex_array(program, [])
    def query(a, b, sun):
        fbo.use()
        program['u_a'].value = tuple(a); program['u_b'].value = tuple(b); program['u_sun'].value = tuple(sun)
        vao.render(moderngl.TRIANGLES, vertices=3)
        return [np.frombuffer(tex.read(), 'f4')[:3].copy() for tex in out]
    cases = [
        ('one meter', (0,6371.01,0), (.001,6371.01,0), (0,1,0)),
        ('ten meters', (0,6371.01,0), (.01,6371.01,0), (.5,1,0)),
        ('near ground', (0,6371.1,0), (.1,6371.1,0), (0,1,0)),
        ('one kilometer', (0,6372.,0), (1,6372.,0), (.5,1,0)),
        ('distant ridge', (0,6372.,0), (30,6375.,0), (.5,1,.3)),
        ('peak above horizon', (0,6371.1,0), (50,6378.,0), (.5,1,0)),
        ('below datum', (0,6370.,0), (2,6369.5,0), (0,1,0)),
        ('orbital terrain', (0,6450.,0), (30,6375.,0), (0,1,0)),
        ('twilight ridge', (0,6372.,0), (30,6375.,0), (1,-.01,.3)),
        ('night ridge', (0,6372.,0), (30,6375.,0), (1,-.15,.3)),
        ('near vertical', (0,6371.1,0), (0,6371.4,0), (.5,1,0)),
        ('forward Mie', (0,6372.,0), (30,6375.,0), (1,.1,0)),
    ]
    failures = []
    for name, a, b, sun in cases:
        actual_T, actual_L = query(a, b, sun)
        expected_T, expected_L = reference(a, b, sun)
        error_T = np.max(np.abs(actual_T - expected_T))
        error_L = np.linalg.norm(actual_L - expected_L) / max(np.linalg.norm(expected_L), 1e-8)
        print(f'{name}: transmission error {error_T:.5f}; radiance error {error_L:.2%}; L={actual_L}')
        assert np.all(np.isfinite(actual_L)) and np.all(actual_L >= 0)
        if not np.allclose(actual_T, expected_T, atol=.003, rtol=.02) or error_L >= .06:
            failures.append((name, actual_T, expected_T, actual_L, expected_L))
    # Height changes must affect both fog and extinction; nearby ground has no
    # full-column haze. Cache keys never contain a camera position.
    near_T, near_L = query((0,6371.01,0), (.001,6371.01,0), (0,1,0))
    assert np.min(near_T) > .9999 and np.max(near_L) < 1e-5
    zero_T, zero_L = query((0,6372.,0), (0,6372.,0), (0,1,0))
    np.testing.assert_array_equal(zero_T, np.ones(3))
    np.testing.assert_array_equal(zero_L, np.zeros(3))
    test_production_depth(ctx, tables, multiple, query)
    changed = dict(PARAMETERS, u_scattering_bottom_km=6367.95)
    assert cache.get(changed, multiple) is not tables
    replacement_ms = ctx.texture((64,64),4,np.tile(np.r_[PSI,1.].astype('f4'),(64*64,1)).tobytes(),dtype='f4')
    assert cache.get(PARAMETERS, replacement_ms) is not tables
    cache.release(); ctx.release()
    assert not failures, failures
    print('All endpoint scattering regressions passed')


if __name__ == '__main__':
    main()
