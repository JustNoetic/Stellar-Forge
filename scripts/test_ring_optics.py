"""Numerical and native OpenGL regressions for shared ring optics."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
import numpy as np
import moderngl

ROOT=Path(os.environ.get('STELLAR_FORGE_ROOT',str(Path(__file__).resolve().parents[1])))
WORK=Path(os.environ.get('RING_TEST_OUTPUT',str(ROOT/'scratch/ring-optics-tests')))
WORK.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
# Avoid app startup; imports below are normal modules in the staged source.
for name,relative in (('engine','engine'),('engine.rendering','engine/rendering'),
                      ('engine.core','engine/core'),('engine.physics','engine/physics')):
    mod=types.ModuleType(name)
    mod.__path__=[str(ROOT/relative)]
    sys.modules[name]=mod
from engine.rendering import ring_optics as cpu

def shader(path):
    import re
    path=ROOT/'engine/glsl'/path
    source=path.read_text()
    def include(match):
        return shader(match.group(1))
    return re.sub(r'^\s*#include "([^\"]+)"\s*$',include,source,flags=re.M)

ctx=moderngl.create_standalone_context(require=460,**({'backend':'egl'} if sys.platform.startswith('linux') else {}))
print(ctx.info['GL_RENDERER'],flush=True)
metrics={'renderer':ctx.info['GL_RENDERER']}
for pair in (('celestial/ring.vert','celestial/ring.frag'),('post/ringshine_map.vert','post/ringshine_map.frag')):
    program=ctx.program(vertex_shader=shader(pair[0]),fragment_shader=shader(pair[1]))
    print('Compiled',pair,flush=True)
    program.release()

# Evaluate production GLSL functions on one pixel per case.
cases=[]
for alpha in (1e-8,.0001,.15,.5,.998999,.999,.999001,1.0):
    for muv,sun in ((1.,1.),(.5,.2),(.01,.0100011),(.1,.1000011),(1.,0.),(.5,-.2)):
        for gf,gb,balance,textured in ((.7,-.3,.35,0),(.999,-.999,1,1),(0.,0.,0.,0)):
            cases.append((alpha,muv,sun,gf,gb,balance,textured))
data=np.array(cases,dtype='f4')
buffer=ctx.buffer(data.tobytes())
buffer.bind_to_storage_buffer(0)
common=shader('common/ring_optics.glsl')
source='''#version 460
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer Input { float inputs[]; };
layout(std430,binding=1) writeonly buffer Output { vec4 outputs[]; };
'''+common+'''
void main() {
    uint i=gl_GlobalInvocationID.x, k=i*7u;
    float alpha=inputs[k],muv=inputs[k+1],sun=inputs[k+2];
    vec4 props=vec4(inputs[k+3],inputs[k+4],inputs[k+5],0.45);
    bool textured=inputs[k+6]>0.5;
    float tau=ring_tau(alpha);
    vec2 phase=ring_phase(-0.4,alpha,props,textured);
    vec2 slab=ring_transfer(tau,muv,abs(sun),sun>=0.0);
    outputs[i]=vec4(tau,phase.x,slab.x,ring_radiance(tau,alpha,muv,sun,-0.4,props,textured,sun>=0.0));
}'''
out=ctx.buffer(reserve=len(cases)*16)
out.bind_to_storage_buffer(1)
program=ctx.compute_shader(source)
program.run(group_x=len(cases))
ctx.memory_barrier()
actual=np.frombuffer(out.read(),dtype='f4').reshape(-1,4)
expected=[]
for alpha,muv,sun,gf,gb,balance,textured in data.astype('f8'):
    tau=cpu.ring_tau(alpha)
    phase=cpu.ring_phase(-.4,alpha,gf,gb,balance,textured)[0]
    single=cpu.ring_transfer(tau,muv,abs(sun),sun>=0)[0]
    radiance=cpu.ring_phase_radiance(muv,sun,alpha,-.4,gf,gb,balance,bool(textured),.45,sun>=0)
    expected.append((tau,phase,single,radiance))
expected=np.array(expected)
np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=2e-7)
assert np.isfinite(actual).all() and np.min(actual)>=0
metrics['gpu_cpu_cases']=len(cases)
print('GPU/CPU optical response:',len(cases),'cases passed',flush=True)

# Phase normalization, including clamped legacy parameters.
x,w=np.polynomial.legendre.leggauss(512)
theta=(x+1)*math.pi/2
mu=np.cos(theta)
solid_weights=w*math.pi/2*np.sin(theta)*2*math.pi
for gf,gb in ((.7,-.3),(.999,-.999),(.1,.9)):
    for textured in (False,True):
        integral=sum(cpu.ring_phase(float(m),.15,gf,gb,.35,textured)[0]*ww
                     for m,ww in zip(mu,solid_weights))
        assert abs(integral-1)<1e-8,integral
print('Both material families normalize to one, including legacy extrema',flush=True)

assert cpu.ring_hg(.999,1)==cpu.ring_hg(.99,1)
below=cpu.ring_slab_response(1,1,.998999,False)
above=cpu.ring_slab_response(1,1,.999001,False)
assert abs(above/below-1)<.003

metrics['dense_threshold_relative_change']=above/below-1
metrics['max_gpu_cpu_absolute_error']=float(np.max(np.abs(actual-expected)))

# Full fragment shader, including atlas sampling and premultiplied alpha.
vertex='''#version 460
out vec3 f_local_pos;
out vec3 f_world_pos;
out vec3 f_normal;
out float f_clip_z;
void main() {
    vec2 uv=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position=vec4(uv*2.-1.,0.,1.);
    f_local_pos=vec3(2.+(uv.x-.5)*.002,0.,0.);
    f_world_pos=f_local_pos;
    f_normal=vec3(0.,1.,0.);
    f_clip_z=1.;
}'''
ring=ctx.program(vertex_shader=vertex,fragment_shader=shader('celestial/ring.frag'))
vao=ctx.vertex_array(ring,[])
target=ctx.texture((8,8),4,dtype='f4')
fbo=ctx.framebuffer([target])
profile=np.ones((272,64,4),dtype='f4')
gradient=ctx.texture((64,272),4,profile.tobytes(),dtype='f4')
gradient.filter=(moderngl.LINEAR,moderngl.LINEAR)
gradient.repeat_x=gradient.repeat_y=False
gradient.use(0)
texture=ctx.texture((64,1),4,np.ones((1,64,4),'f4').tobytes(),dtype='f4')
texture.use(4)
ring['u_ring_gradients'].value=0
ring['u_ring_texture'].value=4
scene=np.zeros(2088,dtype='f4')
scene[:16]=scene[16:32]=np.eye(4,dtype='f4').ravel()
scene.view('i4')[32]=1
scene[292]=1000
scene[293]=1
scene[100:104]=(1,1,1,1)
scene[164:168]=(0,1,0,0)
scene[228:232]=(1,1,1,1)
ubo=ctx.buffer(scene.tobytes())
ubo.bind_to_uniform_block(1)
camera=np.array((2,3,4),dtype='f8')
ring['u_camera_pos'].value=tuple(camera)
ring['u_host_planet_pole_obl'].value=(0,1,0,1)
ring['u_host_planet_pos'].value=(0,0,0)
ring['u_host_planet_radius'].value=0
ring['u_planetshine_enabled'].value=False
ring['u_hdr_enabled'].value=False
ring['u_exposure'].value=1

def render(layers, L=(1.,.3,.4),sin_radius=0.,planet=False):
    L=np.array(L,dtype='f8'); L/=np.linalg.norm(L)
    star=L*100
    scene[36:40]=(*star,sin_radius*100)
    ubo.write(scene.tobytes())
    ring['u_num_ring_planes'].value=len(layers)
    ring['u_is_textured'].value=any(layer.get('textured',False) for layer in layers)
    for i,layer in enumerate(layers):
        prefix=f'u_ring_planes[{i}].'
        values={'color':(1,1,1),'inner_r':1.5,'outer_r':2.5,'opacity':1,
                'scatter':.35,'asymmetry':.7,'backscatter':-.3,'row_idx':16+i,
                'is_textured':float(layer.get('textured',False)),'unlit_factor':.45,
                'saturation':1,'hue_shift':0,'brightness':1,'alpha_boost':1}
        values.update({k:v for k,v in layer.items() if k not in ('alpha','textured')})
        for name,value in values.items():
            if prefix+name in ring: ring[prefix+name].value=value
        profile[16+i,:,:3]=np.array(layer.get('rgb',(1,1,1)))
        profile[16+i,:,3]=layer['alpha']
    gradient.write(profile.tobytes())
    if any(layer.get('textured',False) for layer in layers):
        alpha=next(layer['alpha'] for layer in layers if layer.get('textured',False))
        rgb=next(layer.get('texture_rgb',(1,1,1)) for layer in layers if layer.get('textured',False))
        texture.write(np.tile((*rgb,alpha),(64,1)).astype('f4').tobytes())
    fbo.use(); ctx.viewport=(0,0,8,8); fbo.clear()
    vao.render(vertices=3)
    image=np.frombuffer(target.read(),dtype='f4').reshape(8,8,4).copy()
    assert np.isfinite(image).all() and image.min()>=0
    return image,star

def expected_pixel(layers,star):
    hit=np.array((2.+(.5625-.5)*.002,0,0))
    V=camera-hit; V/=np.linalg.norm(V)
    L=star-hit; L/=np.linalg.norm(L)
    tau=sum(cpu.ring_tau(layer['alpha']) for layer in layers)
    rgb=np.zeros(3)
    for layer in layers:
        textured=layer.get('textured',False)
        color=np.array((1,1,1) if textured else layer.get('rgb',(1,1,1)))**2.2
        response=cpu.ring_component_radiance(tau,layer['alpha'],abs(V[1]),L[1],-np.dot(V,L),
            layer.get('asymmetry',.7),layer.get('backscatter',-.3),layer.get('scatter',.35),
            textured,layer.get('unlit_factor',.45),V[1]*L[1]>=0)
        rgb+=color*cpu.ring_tau(layer['alpha'])/tau*response*math.pi
    return np.r_[rgb,-math.expm1(-tau/abs(V[1]))]

rendered_cases=0
for layers in ([{'alpha':.15,'textured':True}],
               [{'alpha':.5,'rgb':(.6,.8,.9)}],
               [{'alpha':.2,'asymmetry':.1},{'alpha':.2,'asymmetry':.9}],
               [{'alpha':.15,'textured':True},{'alpha':.2,'asymmetry':.9}],
               [{'alpha':.998999}],[{'alpha':.999001}]):
    for L in ((1.,.3,.4),(1.,-.3,.4)):
        image,star=render(layers,L)
        np.testing.assert_allclose(image[4,4],expected_pixel(layers,star),rtol=6e-5,atol=1e-6)
        rendered_cases+=1
metrics['full_ring_render_cases']=rendered_cases
print('Full production ring rendering:',rendered_cases,'cases passed',flush=True)

finite=[]
for sun in (-1e-6,0.,1e-6):
    image,star=render([{'alpha':.5}],(1,sun,0),.005)
    finite.append(image[4,4,:3].mean())
assert finite[1]>0
assert max(finite)/min(finite)<1.002
zero,_=render([{'alpha':.5}],(1,0,0),0.)
assert zero[4,4,:3].max()==0
metrics['finite_star_equinox_rgb']=float(finite[1])
print('Finite stellar disk illuminates equinox continuously; point limit is zero',flush=True)

# Test radial sampling and compiled Numba transport, retaining extreme lobes.
from engine.rendering.ringshine import build_secondary_ring_properties,RingshineMap
def segment(alpha,gf=.7,textured=False):
    prof=np.ones((4096,4),dtype='f4'); prof[:,3]=alpha
    return dict(body_idx=0,inner_r=1.5,outer_r=2.5,opacity=1.,pole=np.array((0,1,0),'f4'),
                shadow_grad=prof,asymmetry=gf,backscatter=-.3,scatter=.35,
                is_textured=textured,unlit_factor=.45,row_idx=16)
segments=[segment(.2,.1),segment(.2,.9)]
params,normals,colors,samples=build_secondary_ring_properties(segments,None,{0:0},1)
rgb=cpu.ring_mixed_radiance(samples[0],.5,.2,-.4,0.)
tau=cpu.ring_tau(.2)*2
expected_rgb=.5*(cpu.ring_component_radiance(tau,.2,.5,.2,-.4,.1,-.3,.35,False,.45,True)
                  +cpu.ring_component_radiance(tau,.2,.5,.2,-.4,.9,-.3,.35,False,.45,True))
np.testing.assert_allclose(rgb,expected_rgb,rtol=2e-6)
np.testing.assert_allclose(params[0,2],.36,rtol=1e-6)
metrics['mixed_annulus_rgb']=rgb.tolist()
print('Compiled CPU radial mixture preserves overlapping lobes',flush=True)

# Production bake and cache lifecycle. Finite stars must affect the bake key.
quad=ctx.buffer(np.array([[-1,-1],[1,-1],[-1,1],[1,1]],'f4').tobytes())
maps=RingshineMap(ctx,shader('post/ringshine_map.vert'),shader('post/ringshine_map.frag'),quad)
gradient.host_segments={0:[segment(.5)]}
gradient.profile_revision=1
props=ctx.texture((64,32),4,dtype='f4')
profile[16,:,3]=.5; gradient.write(profile.tobytes())
hosts=[(0,np.zeros(3),np.array((0,1,0)),(1.5,2.5,1.,1.),0.)]
stars=np.array([[100.,20.,0.]])
assert maps.update(gradient,hosts,stars,4,True,np.array([.5]))==1
assert maps.update(gradient,hosts,stars,4,True,np.array([.5]))==0
assert maps.update(gradient,hosts,stars,4,True,np.array([.6]))==1
gradient.profile_revision+=1
assert maps.update(gradient,hosts,stars,4,True,np.array([.6]))==1
stars=np.array([[100.,0.,0.]])
assert maps.update(gradient,hosts,stars,4,True,np.array([.5]))==1
tile=np.frombuffer(maps.texture.read(),dtype='f4').reshape(1040,2048,4)[:65,:128]
assert np.isfinite(tile).all() and tile[:,:,:3].min()>=0 and tile[:,:,:3].max()>0
np.save(WORK/'ring-finite-star-map.npy',tile)
metrics['host_map_equinox_peak']=float(tile[:,:,:3].max())
print('Production host map: nonzero equinox and material/star-size invalidation passed',flush=True)

# Compile the actual distant-body Numba entry point with its new sample layout.
from engine.rendering.planetshine import compute_planetshine_numba
pos=np.array([[0.,0.,0.],[.03,.01,0.]],dtype='f8')
body_radii=np.array([.001,.0001],dtype='f8')
params2,normals2,colors2,samples2=build_secondary_ring_properties(
    [dict(seg,inner_r=seg['inner_r']*.001,outer_r=seg['outer_r']*.001) for seg in segments],
    None,{0:0},2)
args=(pos,body_radii,np.ones((2,3),'f4'),np.zeros(2,'f4'),np.array([[1.,.2,0.]]),
      np.ones((1,3),'f4'),np.ones(1,'f4'),np.array([.005]),False,
      params2,normals2,colors2,False,True,samples2)
directions,received=compute_planetshine_numba(*args)
assert np.isfinite(received).all() and np.isfinite(directions).all() and received[1].max()>0
metrics['numba_moon_ringshine_rgb']=received[1].tolist()
print('Actual Numba distant-body entry point compiled and received ringshine',flush=True)

# Isolate production planetshine to check finite visible/illuminated geometry.
saved_ring,saved_vao=ring,vao
ring=ctx.program(vertex_shader=vertex,fragment_shader=shader('celestial/ring.frag').replace(
    'vec3 raw_color = total_direct_illum_color;','vec3 raw_color = vec3(0.0);'))
vao=ctx.vertex_array(ring,[])
for name,value in {'u_ring_gradients':0,'u_ring_texture':4,'u_camera_pos':tuple(camera),
                   'u_host_planet_pole_obl':(0,1,0,1),'u_host_planet_pos':(0,0,0),
                   'u_host_planet_radius':1.,'u_host_planet_color':(.8,.8,.8),
                   'u_planetshine_enabled':True,'u_hdr_enabled':False,'u_exposure':1}.items():
    if name in ring: ring[name].value=value
for mode in (0, 1):
    if 'u_planetshine_mode' in ring: ring['u_planetshine_mode'].value = mode
    day,_=render([{'alpha':.5}],(1,0,0))
    night,_=render([{'alpha':.5}],(-1,0,0))
    assert day[4,4,:3].min()>0
    assert night[4,4,:3].max()==0
    ring['u_host_planet_pole_obl'].value=(0,1,0,1.3)
    oblate,_=render([{'alpha':.5}],(1,0,0))
    assert 0<oblate[4,4,0]<day[4,4,0]
    ring['u_host_planet_pole_obl'].value=(0,1,0,1)
    if mode == 0:
        metrics['planetshine_day_rgb']=day[4,4,:3].tolist()
        metrics['planetshine_night_rgb']=night[4,4,:3].tolist()
print('Production planetshine: lit disk, hidden day side, and oblate host passed (both analytical & numerical)',flush=True)
ring,vao=saved_ring,saved_vao

# Source quadrature convergence against a much denser independent disk integral.
image,star=render([{'alpha':.5}],(1,0,0),.005)
hit=np.array((2.+(.5625-.5)*.002,0,0)); V=camera-hit; V/=np.linalg.norm(V)
L=star-hit; L/=np.linalg.norm(L)
radius=.5/np.linalg.norm(star-hit)
ref=0.
nodes,weights=np.polynomial.legendre.leggauss(32)
for r2,wr in zip((nodes+1)*.5,weights*.5):
    r=radius*math.sqrt(r2); cd=math.sqrt(1-r*r)
    for q in range(128):
        az=(q+.5)*2*math.pi/128
        direction=np.array((cd,r*math.sin(az),r*math.cos(az)))
        ref+=cpu.ring_component_radiance(cpu.ring_tau(.5),.5,abs(V[1]),direction[1],
            -np.dot(direction,V),.7,-.3,.35,False,.45,V[1]*direction[1]>=0)*wr/(128*cd)*math.pi
error=abs(image[4,4,0]/ref-1)
assert error<.01,error
metrics['finite_star_64_vs_4096_relative_error']=float(error)
print('Equinox finite source agrees with dense quadrature within',f'{error:.2%}',flush=True)

# Representative full fragment costs; baking is cached and timed separately.
import statistics
large_target=ctx.texture((1024,256),4,dtype='f4')
large_fbo=ctx.framebuffer([large_target])
metrics['gpu_fragment_ms_1024x256']={}
for name,planet,mode,source_radius in (('direct_point',False,0,0.),('direct_finite',False,0,.005),
                                       ('direct_and_planetshine',True,0,0.),
                                       ('direct_and_planetshine_numerical',True,1,0.)):
    ring['u_host_planet_radius'].value=1. if planet else 0.
    ring['u_host_planet_color'].value=(.8,.8,.8)
    ring['u_planetshine_enabled'].value=planet
    if 'u_planetshine_mode' in ring: ring['u_planetshine_mode'].value=mode
    render([{'alpha':.5}],(1,0,0) if source_radius else (1,.3,.4),source_radius)
    large_fbo.use(); ctx.viewport=(0,0,1024,256)
    vao.render(vertices=3); ctx.finish()
    times=[]
    for trial in range(5):
        with ctx.query(time=True) as query:
            for repeat in range(5): vao.render(vertices=3)
        times.append(query.elapsed/5e6)
    metrics['gpu_fragment_ms_1024x256'][name]=statistics.median(times)
print('GPU fragment timings:',metrics['gpu_fragment_ms_1024x256'],flush=True)

# Actual atlas rebuild: optical phase metadata must retain original segments,
# and extinction is order-independent with opacity above one safely bounded.
from engine.rendering.render_utils import rebuild_ring_gradients_atlas,bake_unified_shadow_profile,generate_ring_shadow_grad
ring['u_planetshine_enabled'].value=False
ring['u_host_planet_radius'].value=0.
layer={'alpha':.2,'textured':True,'texture_rgb':(.3,.5,.6),'color':(.8,.9,1.),
       'brightness':2.,'saturation':3.,'hue_shift':.1,'alpha_boost':2.}
edited=generate_ring_shadow_grad([],np.tile((.3,.5,.6,.2),(64,1)).astype('f4'),
    raw_color=(.8,.9,1.),hue_shift=.1,saturation=3.,brightness=2.,alpha_boost=2.,res=64)
image,star=render([layer])
expected=expected_pixel([{'alpha':float(edited[0,3]),'textured':True}],star)
expected[:3]*=edited[0,:3]**2.2
np.testing.assert_allclose(image[4,4],expected,rtol=6e-5,atol=1e-6)
print('Edited/tinted texture matches the CPU secondary-light profile',flush=True)
profile_ab=bake_unified_shadow_profile(segments,1.5,2.5)
profile_ba=bake_unified_shadow_profile(list(reversed(segments)),1.5,2.5)
np.testing.assert_allclose(profile_ab,profile_ba,rtol=1e-6,atol=1e-7)
np.testing.assert_allclose(profile_ab[:,3],.36,rtol=1e-6)
heavy=bake_unified_shadow_profile([dict(segments[0],opacity=10)],1.5,2.5)
assert np.isfinite(heavy).all() and np.min(heavy[:,3])>=0 and np.max(heavy[:,3])<=1
actual_atlas=ctx.texture((4096,272),4,dtype='f4')
indices=rebuild_ring_gradients_atlas(segments,actual_atlas)
assert indices=={0:0}
assert actual_atlas.host_segments[0][0]['asymmetry']==.1
assert actual_atlas.host_segments[0][1]['asymmetry']==.9
assert actual_atlas.host_segments[0][0]['row_idx']==16
assert actual_atlas.host_segments[0][1]['row_idx']==17
assert actual_atlas.profile_revision==1
np.testing.assert_allclose(actual_atlas.atlas_data[0,:,3],.36,rtol=1e-6)
print('Actual atlas rebuild retains segment lobes; extinction union is stable and order-independent',flush=True)

(WORK/'ring-fix-results.json').write_text(json.dumps(metrics,indent=2))
