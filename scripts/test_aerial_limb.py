"""Orbital ground/limb volume comparison against a 256-step GPU reference.

Pass --terminator for quarter-phase sunlight and --preview to save the
in-scattering comparison PNG (reference left, reconstructed volume right).
"""
from pathlib import Path
import sys

import moderngl
import numpy as np
from pyrr import matrix44

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.rendering.aerial_volume import AerialPerspectiveVolume
from engine.rendering.shader_loader import load_shader

ctx = moderngl.create_context(standalone=True)
scene = np.zeros(1832,dtype='f4')
scene.view('i4')[32] = 1
scene[36:40] = [0,0,1e6,100]
scene[100:104] = [1,1,1,1]
scene[293] = 1
scene_buffer = ctx.buffer(scene.tobytes())
scene_buffer.bind_to_uniform_block(1)
instances = ctx.buffer(np.zeros(28,dtype='f4').tobytes())
instances.bind_to_storage_buffer(2)
atmosphere = np.zeros(256,dtype='f4')
atmosphere[19] = 1
atmosphere[24:28] = [0,1,0,1]
atmosphere[32:35] = 1
atmosphere[192:196] = [1,0.8,0.4,0.0001]
sun_direction = np.array([1,0,0.1] if '--terminator' in sys.argv else [0,0,1], dtype='f4')
sun_direction /= np.linalg.norm(sun_direction)
atmosphere[208:212] = [*sun_direction,1]
atmosphere[224:228] = [0,0,1e6,0.0001]
atmosphere[240:244] = [0,0,1e6,100]
atmo_buffer = ctx.buffer(atmosphere.tobytes())
atmo_buffer.bind_to_storage_buffer(8)
sun_trans = ctx.texture((2,2),4,np.ones((2,2,4),dtype='f4').tobytes(),dtype='f4')
ms = ctx.texture((2,2),4,np.zeros((2,2,4),dtype='f4').tobytes(),dtype='f4')
sun_trans.use(1)
ms.use(3)
volume = AerialPerspectiveVolume(ctx)
R, A, D = 6371.0, 6471.0, 6371.0 * 3.5
projection = matrix44.create_perspective_projection(45, 1, 0.1, 100000, dtype='f4')
view = matrix44.create_look_at([0,0,D], [0,0,0], [0,1,0], dtype='f4')
ip = np.linalg.inv(projection).astype('f4').tobytes()
iv = np.linalg.inv(view).astype('f4').tobytes()
scene[:16] = projection.ravel()
scene[16:32] = view.ravel()
scene[292] = 100000
scene_buffer.write(scene.tobytes())
atmosphere[3] = A
atmosphere[4:7] = [5.8e-6,13.5e-6,33.1e-6]
atmosphere[7] = 8
atmosphere[8:11] = 2e-6
atmosphere[11] = 1.2
atmosphere[15] = 0.758
atmosphere[20:23] = [R,A,1]
atmosphere[181:183] = [8,R]
atmosphere[184:187] = [1/8,1/1.2,1/8]
atmo_buffer.write(atmosphere.tobytes())
if '--terminator' in sys.argv:
    # Use the real grazing sunlight optical depth, rather than a white LUT
    # that hides the steep spectral changes near the physical terminator.
    lut_vertex = '''#version 460 core
out vec2 f_uv;
void main() {
vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
gl_Position=vec4(p*2-1,0,1);f_uv=p;
}'''
    lut_program = ctx.program(vertex_shader=lut_vertex,
        fragment_shader=load_shader('atmosphere/atmo_lut.frag'))
    for name, value in {
        'u_planet_radius_km': R, 'u_atmo_radius_km': A,
        'u_h_rayleigh': 8, 'u_h_mie': 1.2,
        'u_beta_rayleigh': (5.8e-6,13.5e-6,33.1e-6),
        'u_beta_mie': (2e-6,2e-6,2e-6),
        'u_beta_abs_mixed': (0,0,0), 'u_beta_abs_layered': (0,0,0),
        'u_mie_albedo': (1,1,1),
    }.items():
        if name in lut_program: lut_program[name].value = value
    sun_trans.release()
    sun_trans = ctx.texture((512,256),4,dtype='f4')
    sun_trans.filter = (moderngl.LINEAR,moderngl.LINEAR)
    sun_trans.repeat_x = sun_trans.repeat_y = False
    sun_fbo = ctx.framebuffer([sun_trans])
    sun_fbo.use()
    ctx.vertex_array(lut_program,[]).render(vertices=3)
    sun_trans.use(1)
with ctx.query(time=True) as timer:
    volume.bake((0,0), (0,0,D), sun_direction, ip, iv, 0, bytes(192), False, False, (A,R-0.01))
print('Volume bake ms:', timer.elapsed/1e6)
volume.scatter.use(20)
volume.transmittance.use(21)
from engine.rendering.shader_loader import load_shader
vertex = '''#version 460 core
out vec2 f_uv;
void main() {
vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
gl_Position=vec4(p*2-1,0,1);f_uv=p;
}'''
fragment = '''#version 460 core
#define PI 3.14159265358979323846
''' + load_shader('common/atmosphere_lut_lighting.glsl') + load_shader('common/aerial_volume_mapping.glsl') + '''
in vec2 f_uv;
uniform mat4 inv_proj;
uniform mat4 inv_view;
uniform sampler3D volume_S;
uniform sampler3D volume_T;
layout(location=0) out vec4 ref_S;
layout(location=1) out vec4 ref_T;
layout(location=2) out vec4 lut_S;
layout(location=3) out vec4 lut_T;
layout(location=4) out vec4 raw_S;
void main() {
vec4 eye=inv_proj*vec4(f_uv*2-1,-1,1);
vec3 V=normalize((inv_view*vec4(eye.xy,-1,0)).xyz);
vec2 ground=aerialSphereIntersect(u_cam_pos,V,u_planet_radius_km);
if(ground.x<=0 || ground.x>ground.y) discard;
vec2 air=aerialRaySpan(u_cam_pos,V,u_atmo_radius_km);
vec4 s0,s1,s2,s3;
integrateAtmosphereSegment(u_cam_pos,V,max(0,air.x),ground.x,256,false,ref_S,ref_T,s0,s1,s2,s3);
vec3 S,T;
bool valid=sampleAerialGround(volume_S,volume_T,f_uv,ground.x,u_cam_pos,V,
    u_atmo_radius_km,vec2(u_atmo_radius_km,u_planet_radius_km-0.01),inv_proj,inv_view,u_pole_obl,S,T);
raw_S=valid?vec4(S,1):ref_S;
float refinement = valid ? aerialTerminatorRefinement(f_uv,ground.x,u_cam_pos,V,
    vec4(u_star_dir_sph_eff[0].xyz,u_star_pos_local[0].w),
    vec3(u_planet_radius_km,u_atmo_radius_km,max(u_h_rayleigh,u_h_mie)),
    textureSize(volume_S,0).xy,inv_proj,inv_view,u_pole_obl) : 1.0;
if (refinement > 0.0) {
    vec4 refined_S,refined_T;
    integrateAtmosphereSegment(u_cam_pos,V,max(0,air.x),ground.x,128,false,
        refined_S,refined_T,s0,s1,s2,s3);
    if (valid) { S=mix(S,refined_S.rgb,refinement); T=mix(T,refined_T.rgb,refinement); }
    else { S=refined_S.rgb; T=refined_T.rgb; }
}
lut_S=vec4(S,refinement);
lut_T=vec4(T,valid?1:0);
}'''
p=ctx.program(vertex_shader=vertex,fragment_shader=fragment)
p['inv_proj'].write(ip)
p['inv_view'].write(iv)
p['u_cam_pos'].value=(0,0,D)
if 'u_sun_dir' in p: p['u_sun_dir'].value=(0,0,1)
if 'u_transmittance_lut' in p: p['u_transmittance_lut'].value=1
if 'u_multi_scatter_lut' in p: p['u_multi_scatter_lut'].value=3
p['volume_S'].value=20
p['volume_T'].value=21
outputs=[ctx.texture((512,512),4,dtype='f4') for _ in range(5)]
fbo=ctx.framebuffer(outputs)
fbo.use();fbo.clear()
ctx.vertex_array(p,[]).render(vertices=3)
arrays=[np.frombuffer(t.read(),dtype='f4').reshape(512,512,4) for t in outputs]
rs,rt,ls,lt,raw=arrays
mask=rt[:,:,3]>0
ref=rs[:,:,:3]+0.2*rt[:,:,:3]
actual=ls[:,:,:3]+0.2*lt[:,:,:3]
errors=np.abs(ref[mask]-actual[mask])
print('Ground radiance absolute error mean/max/95th:',errors.mean(),errors.max(),np.percentile(errors,95))
print('Transmittance absolute error mean/max:',np.abs(rt[mask,:3]-lt[mask,:3]).mean(),np.abs(rt[mask,:3]-lt[mask,:3]).max())
fallback_fraction = np.mean(lt[mask,3] == 0)
print('Grazing/unsupported fallback fraction:',fallback_fraction)
print('Twilight refinement fraction:',np.mean(ls[mask,3] > 0))
assert np.isfinite(errors).all()
assert errors.mean() < 0.002, 'Orbital grid causes excessive radiance error'
assert np.percentile(errors,95) < 0.005, 'Orbital ground interpolation shows grid artifacts'
assert np.abs(rt[mask,:3]-lt[mask,:3]).mean() < 0.005, 'Ground transmittance leaks across columns'
assert fallback_fraction < 0.1, 'Fallback covers too much of the ground'
if '--terminator' in sys.argv:
    twilight = mask & (ls[:,:,3] > 0)
    twilight_error = np.abs(rs[twilight,:3] - ls[twilight,:3])
    print('Twilight in-scattering error mean/max/99th:',twilight_error.mean(),twilight_error.max(),np.percentile(twilight_error,99))
    raw_error = np.abs(rs[twilight,:3] - raw[twilight,:3])
    print('Unrefined twilight in-scattering error mean/99th:',raw_error.mean(),np.percentile(raw_error,99))
    assert np.percentile(twilight_error,99) < 0.001, 'Twilight has coarse lighting blocks'
    assert np.percentile(twilight_error,99) < np.percentile(raw_error,99) * 0.25, 'Refinement must reduce terminator artifacts'
    assert np.mean(ls[mask,3] > 0) < 0.5, 'Twilight refinement covers most of the globe'
if '--preview' in sys.argv:
    from PIL import Image
    both=np.concatenate([rs[:,:,:3],ls[:,:,:3]],axis=1) * 12
    both=(np.clip(both*2,0,1)**(1/2.2)*255).astype('uint8')[::-1]
    Image.fromarray(both).save('aerial_limb_comparison.png')
volume.release();ctx.release()
print('Orbital aerial-perspective GPU regression passed')
