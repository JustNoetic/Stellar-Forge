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
    'u_scattering_bottom_km': 6371., 'u_scattering_sun_radius': 0.005,
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


def secondary_reference(a,b,light=(0,1,0),ring_normal=None):
    a,b=np.array(a,float),np.array(b,float)
    d=np.linalg.norm(b-a); v=(b-a)/d; n=4096; ds=d/n
    points=a+((np.arange(n)+.5)*ds)[:,None]*v
    r=np.linalg.norm(points,axis=1)
    R,M,ext=medium(r)
    factor=-np.expm1(-ext*ds)/np.maximum(ext,1e-30)
    T=np.exp(-(np.cumsum(ext,axis=0)-ext)*ds)
    if ring_normal is not None:
        phase=1/(4*np.pi); phase_M=phase/max(.15,1-PARAMETERS['u_mie_g'])
        source=R*phase+M*phase_M+(R+M)*PSI
    else:
        light=np.array(light,float); light/=np.linalg.norm(light)
        mu=points@light/r
        horizon=-np.sqrt(np.maximum(0.,1-(6371/np.maximum(r,6371.01))**2))
        q=np.clip((mu-horizon+.05)/.1,0,1); visibility=q*q*(3-2*q)
        b_s=r*mu
        distance=-b_s+np.sqrt(np.maximum(0.,b_s*b_s+6471**2-r*r))
        station=distance[:,None]*(np.arange(256)+.5)/256
        rq=np.sqrt(r[:,None]**2+station*(2*b_s[:,None]+station))
        tau=medium(rq)[2].sum(axis=1)*(distance/256)[:,None]
        incoming=np.exp(-tau)*visibility[:,None]*(mu>=horizon)[:,None]
        nu=v@light; g=PARAMETERS['u_mie_g']
        phase=3/(16*np.pi)*(1+nu*nu)
        phase_M=3/(8*np.pi)*(1-g*g)/(2+g*g)*(1+nu*nu)/(1+g*g-2*g*nu)**1.5
        source=(R*phase+M*phase_M+(R+M)*PSI)*incoming
    return np.sum(T*factor*source,axis=0)


def test_production_depth(ctx, tables, multiple, endpoint_query, cache):
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
            member.value = tuple([19] * member.array_length) if member.array_length > 1 else 19
    white.use(0); white3d.use(19); depth.use(9); multiple.use(3)
    ScatteringLUTCache.bind(tables)
    program['u_depth_texture'].value = 9
    program['u_multi_scatter_lut'].value = 3
    ScatteringLUTCache.configure(program)
    program['u_scattering_enabled'].value = True
    program['u_scattering_bottom_km'].value = PARAMETERS['u_scattering_bottom_km']
    program['u_scattering_terrain_bottom_km'].value = 6359.966
    if 'u_scattering_sun_radius' in program:
        program['u_scattering_sun_radius'].value = PARAMETERS['u_scattering_sun_radius']
    program['u_scattering_azimuth_count'].value = cache.size[3]
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
    def draw(a, b, sun=(0,1,0), stars=1, secondary=None, quality=3, scene_depth=True, samples=128, adaptive=False):
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
        data.view('i4')[23] = samples
        data.view('i4')[29] = int(adaptive)
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
    # Earth's ocean-floor bound is for scene clipping only. Above-datum haze
    # at sunset must not change when this unused terrain bound changes.
    sunset = (np.sqrt(1 - .05**2), -.05, 0)
    for mu in (-.085, -.075, -.063, -.056):
        a = np.array([0., 6382., 0.])
        v = np.array([np.sqrt(1-mu*mu), mu, 0.])
        d = -a@v - np.sqrt((a@v)**2 + 6375.**2 - a@a)
        b = a + d*v
        expected_T, expected_L = endpoint_query(a, b, sunset)
        for floor in (6359.966, 6368.95, 6370.95):
            program['u_scattering_terrain_bottom_km'].value = floor
            actual_L, actual_T = draw(a, b, sun=sunset)
            np.testing.assert_allclose(actual_L, expected_L, rtol=.015, atol=2e-5)
            np.testing.assert_allclose(actual_T, expected_T, rtol=.004, atol=2e-4)
    program['u_scattering_terrain_bottom_km'].value = 6359.966
    print('11 km sunset depth and terrain-bound independence passed')
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
    # Analytical secondary transport compared with independent dense integration.
    inst=np.zeros(28,'f4'); inst[16:19]=[0,1,0]; inst[20:23]=[.1,.2,.3]
    baseline_L, baseline_T = draw(a,b)
    instance.write(inst.tobytes()); program['u_planetshine_enabled'].value=True
    L3,T3=draw(a,b)
    expected_ps = secondary_reference(a,b,(0,1,0)) * inst[20:23] * np.pi
    np.testing.assert_allclose(L3-baseline_L,expected_ps,rtol=.06,atol=2e-5)
    np.testing.assert_allclose(T3,baseline_T,atol=1e-6)
    for altitude in (1., -1.):
        for distance in (.001,.1,1.):
            short_a=(0,6371.+altitude,0); short_b=(distance,6371.+altitude,0)
            program['u_planetshine_enabled'].value=False
            direct,_=draw(short_a,short_b)
            program['u_planetshine_enabled'].value=True
            with_shine,_=draw(short_a,short_b)
            expected=secondary_reference(short_a,short_b)*inst[20:23]*np.pi
            np.testing.assert_allclose(with_shine-direct,expected,rtol=.06,atol=2e-7)
    for samples in (4,32,128):
        for adaptive in (False,True):
            L,T=draw(a,b,samples=samples,adaptive=adaptive)
            np.testing.assert_array_equal(L,L3)
            np.testing.assert_array_equal(T,T3)
    # Irrelevant external rings and polar winter cannot select a numerical fallback.
    program['u_num_ring_planes'].value=1
    program['u_ring_center'].write(np.array([10.,0.,0.]+[0.]*45,'f4').tobytes())
    program['u_ring_normal'].write(np.array([0.,1.,0.]+[0.]*45,'f4').tobytes())
    program['u_ring_params'].write(np.array([1e-5,2e-5,1.,1e-6]+[0.]*60,'f4').tobytes())
    data[240]=1.; data[241]=-1.
    np.testing.assert_array_equal(draw(a,b,samples=4,adaptive=True)[0],L3)
    data[240]=0.; data[241]=0.
    # Host ringshine retains midpoint-map irradiance and uses LUT optical response.
    program['u_planetshine_enabled'].value=False
    inst[:]=0.; inst.view('u4')[15]=1
    instance.write(inst.tobytes())
    program['u_ringshine_enabled'].value=True
    program['u_ringshine_map'].value=0  # white unit irradiance map
    program['u_ring_center'].write(np.zeros(48,'f4').tobytes())
    ring_L,ring_T=draw(a,b)
    expected_ring=secondary_reference(a,b,ring_normal=(0,1,0))/np.pi
    np.testing.assert_allclose(ring_L-baseline_L,expected_ring,rtol=.06,atol=2e-5)
    np.testing.assert_allclose(ring_T,baseline_T,atol=1e-6)
    np.testing.assert_array_equal(draw(a,b,samples=4,adaptive=True)[0],ring_L)
    below_a,below_b=(0,6370.,0),(2,6369.5,0)
    program['u_ringshine_enabled'].value=False
    below_direct,_=draw(below_a,below_b)
    program['u_ringshine_enabled'].value=True
    below_ring,_=draw(below_a,below_b)
    expected_below_ring=secondary_reference(below_a,below_b,ring_normal=(0,1,0))/np.pi
    np.testing.assert_allclose(below_ring-below_direct,expected_below_ring,rtol=.06,atol=2e-5)
    # Simultaneous shine contributions must add without applying transmission twice.
    inst[16:19]=[0,1,0]; inst[20:23]=[.1,.2,.3]; instance.write(inst.tobytes())
    both_L,_=draw(a,b)
    np.testing.assert_allclose(both_L,ring_L+(L3-baseline_L),rtol=.002,atol=1e-6)
    program['u_ringshine_enabled'].value=False
    program['u_num_ring_planes'].value=0
    print('Analytical shine, ringshine, additivity, and ray-step independence passed')
    # Pixels without local geometry retain the existing Sky-View output.
    instance.write(np.zeros(28,'f4').tobytes()); program['u_planetshine_enabled'].value=False
    Lsky,Tsky=draw((0,6372,0),(10,6380,0),scene_depth=False)
    np.testing.assert_allclose(Lsky,np.ones(3),atol=1e-6)
    np.testing.assert_allclose(Tsky,np.ones(3),atol=1e-6)
    print('Production multi-star, analytical secondary light, and sky isolation passed')


def test_skyview_lookup(ctx,tables,multiple,endpoint_query,cache):
    program=ctx.program(vertex_shader=load_shader('atmosphere/sky_view_lut.vert'),
                        fragment_shader=load_shader('atmosphere/sky_view_lut.frag'))
    ScatteringLUTCache.configure(program); ScatteringLUTCache.bind(tables)
    multiple.use(3); program['u_multi_scatter_lut'].value=3
    program['u_scattering_bottom_km'].value=PARAMETERS['u_scattering_bottom_km']
    if 'u_scattering_sun_radius' in program:
        program['u_scattering_sun_radius'].value=.005
    program['u_scattering_azimuth_count'].value=cache.size[3]
    program['u_cam_pos'].value=(0,6372,0); program['u_sun_dir'].value=(0,1,0)
    au=149597870.7
    scene=np.zeros(1832,'f4'); scene.view('i4')[32]=1
    scene_buffer=ctx.buffer(scene.tobytes()); scene_buffer.bind_to_uniform_block(1)
    inst=np.zeros(28,'f4'); instance=ctx.buffer(inst.tobytes()); instance.bind_to_storage_buffer(2)
    data=np.zeros(256,'f4')
    data[4:7]=PARAMETERS['u_beta_rayleigh']; data[7]=8
    data[8:11]=PARAMETERS['u_beta_mie']; data[11]=1.2
    data[15]=.76; data[16:19]=PARAMETERS['u_beta_abs_layered']; data[19]=1
    data[20:23]=[6371,6471,au]; data.view('i4')[23]=4
    data[24:28]=[0,1,0,1]; data[32:35]=PARAMETERS['u_mie_albedo']
    data[180:184]=[25,8,6371,128]; data[192:196]=[1,1,1,.005]
    data[208:212]=[0,1,0,np.sqrt(1-.005**2)]; data[224:228]=[0,1e8,0,.005]
    atmosphere=ctx.buffer(data.tobytes()); atmosphere.bind_to_storage_buffer(8)
    output=[ctx.texture((8,8),4,dtype='f4') for _ in range(6)]
    fbo=ctx.framebuffer(output)
    vertices=ctx.buffer(np.array([[-1,-1],[1,-1],[-1,1],[1,1]],'f4').tobytes())
    vao=ctx.vertex_array(program,[(vertices,'2f','in_position')])
    def bake():
        atmosphere.write(data.tobytes()); scene_buffer.write(scene.tobytes())
        fbo.use(); ctx.viewport=(0,0,8,8); vao.render(moderngl.TRIANGLE_STRIP)
        return [np.frombuffer(t.read(),'f4').reshape(8,8,4).copy() for t in output]
    baseline=bake()
    # Verify a sky texel against independently integrated camera-to-boundary light.
    uv=np.array([4.5/8,7.5/8]); horizon=np.pi-np.arcsin(6371/6372)
    theta=horizon*(1-((uv[1]-.5)/.5)**2); phi=uv[0]*2*np.pi
    # Zenith sunlight makes x_basis=cross(zenith,Z)=X and y_basis=-Z.
    v=np.array([np.sin(theta)*np.cos(phi),np.cos(theta),-np.sin(theta)*np.sin(phi)])
    a=np.array([0.,6372.,0.]); bv=a@v
    b=a+v*(-bv+np.sqrt(bv*bv+6471**2-a@a))
    expected_T,expected_L=reference(a,b,(0,1,0))
    np.testing.assert_allclose(baseline[0][7,4,:3],expected_L,rtol=.06,atol=2e-5)
    np.testing.assert_allclose(baseline[1][7,4,:3],expected_T,rtol=.02,atol=.003)
    data.view('i4')[23]=128; data.view('i4')[29]=1; data[240]=1.; data[241]=-1.
    for actual,old in zip(bake(),baseline): np.testing.assert_array_equal(actual,old)
    inst[16:19]=[0,1,0]; inst[20:23]=[.1,.2,.3]; instance.write(inst.tobytes())
    program['u_planetshine_enabled'].value=True
    shine=bake()
    expected_ps=secondary_reference(a,b)*inst[20:23]*np.pi
    np.testing.assert_allclose(shine[0][7,4,:3]-baseline[0][7,4,:3],expected_ps,rtol=.06,atol=2e-5)
    np.testing.assert_array_equal(shine[2],baseline[2])
    np.testing.assert_array_equal(shine[1],baseline[1])
    # Exercise host ringshine in the actual Sky-View producer as well.
    ring_map=ctx.texture((1,16),4,np.ones((16,4),'f4').tobytes(),dtype='f4')
    ring_map.use(8); program['u_ringshine_map'].value=8
    program['u_ringshine_enabled'].value=True
    program['u_num_ring_planes'].value=1
    program['u_ring_normal'].write(np.array([0,1,0]+[0]*45,'f4').tobytes())
    inst.view('u4')[15]=1; instance.write(inst.tobytes())
    rings=bake()
    expected_rs=secondary_reference(a,b,ring_normal=(0,1,0))/np.pi
    np.testing.assert_allclose(rings[0][7,4,:3]-shine[0][7,4,:3],expected_rs,rtol=.06,atol=2e-5)
    np.testing.assert_array_equal(rings[2],shine[2])
    program['u_ringshine_enabled'].value=False
    program['u_num_ring_planes'].value=0
    # Distinct angular-size solar tables reuse the optical/secondary fields.
    second=cache.get(dict(PARAMETERS,u_scattering_sun_radius=.05),multiple)
    assert second[0] is tables[0] and second[4] is tables[4]
    ScatteringLUTCache.bind_star(second,1)
    scene.view('i4')[32]=2
    data[196:200]=[.5,.6,.7,.05]; data[212:216]=[0,1,0,np.sqrt(1-.05**2)]
    data[228:232]=[0,1e8,0,.05]
    two=bake()
    np.testing.assert_allclose(two[0][...,:3],shine[0][...,:3]+two[3][...,:3],rtol=.001,atol=1e-6)
    assert np.max(two[3][...,:3])>0
    # Cache entries remain bounded as changing disc sizes replace old variants.
    for radius in (.01,.02,.03,.04,.06):
        cache.get(dict(PARAMETERS,u_scattering_sun_radius=radius),multiple)
    assert len(cache.entries)==1
    assert len(next(iter(cache.entries.values()))['solar'])==4
    print('Lookup-only Sky-View, dense reference, shine, star discs, and step independence passed')


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
    ScatteringLUTCache.configure(program)
    program['u_scattering_azimuth_count'].value = cache.size[3]
    program['u_multi_scatter_lut'].value = 3
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
        ('crossing datum', (0,6371.5,0), (4,6370.5,0), (0,1,0)),
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
    test_production_depth(ctx, tables, multiple, query, cache)
    test_skyview_lookup(ctx, tables, multiple, query, cache)
    changed = dict(PARAMETERS, u_h_rayleigh=8.1)
    assert cache.get(changed, multiple) is not tables
    replacement_ms = ctx.texture((64,64),4,np.tile(np.r_[PSI,1.].astype('f4'),(64*64,1)).tobytes(),dtype='f4')
    assert cache.get(PARAMETERS, replacement_ms) is not tables
    cache.release(); ctx.release()
    assert not failures, failures
    print('All endpoint scattering regressions passed')


if __name__ == '__main__':
    main()
