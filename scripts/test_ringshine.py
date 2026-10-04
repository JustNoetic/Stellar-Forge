"""GPU ringshine regressions. Run with venv/Scripts/python.exe.

Compare the production GLSL integral with independent dense area integration;
exercise cache invalidation, sixteen-host materials and multi-star sampling.
No application window or saved settings are changed.
"""
import math
import sys
from pathlib import Path
import moderngl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.rendering.shader_loader import load_shader
from engine.rendering.ringshine import (RingshineMap, RING_PROFILE_ROWS,
    RING_PROPERTY_ROWS, build_secondary_ring_properties)
from engine.rendering.render_utils import rebuild_ring_gradients_atlas
from engine.rendering.atmosphere_programs import specialize_atmosphere
from engine.rendering.planetshine import compute_planetshine_numba, ring_phase_radiance, ring_slab_response


def reference_band(r, slat, flattening, sun, phi, alpha, count=262144, textured=False):
    """Independent Cartesian emitter/receiver geometry and ray/ellipsoid test."""
    clat = math.sqrt(1-slat*slat)
    ff = 1-flattening
    rho = ff/math.sqrt(ff*ff*clat*clat+slat*slat)
    p = np.array([rho*clat, 0, rho*slat])
    normal = np.array([clat, 0, slat/(ff*ff)])
    normal /= np.linalg.norm(normal)
    angle = (np.arange(count)+.5)*2*math.pi/count
    q = np.column_stack((r*np.cos(angle), r*np.sin(angle), np.zeros(count)))
    ray = p-q
    distance = np.linalg.norm(ray, axis=1)
    v = ray/distance[:,None]
    mu_v = np.abs(v[:,2])
    mu_p = np.maximum(0, -v@normal)
    light = np.array([-math.sqrt(1-sun*sun)*math.cos(phi),
                      -math.sqrt(1-sun*sun)*math.sin(phi), sun])
    theta = -v@light
    b = 2*(q@light)
    a = light[0]**2+light[1]**2+light[2]**2/(ff*ff)
    shadow = (b<0)&(b*b-4*a*(r*r-1)>=0)
    if abs(sun)<1e-7 or alpha==0:
        return 0.
    tau = -math.log1p(-alpha)
    vd,ld = tau/np.maximum(mu_v,1e-7),tau/abs(sun)
    if slat*sun>=0:
        single = abs(sun)/(np.maximum(mu_v,1e-7)+abs(sun))*(-np.expm1(-vd-ld))
        gamma = math.sqrt(.08)
        hv = (1+2*mu_v)/(1+2*mu_v*gamma)
        h0 = (1+2*abs(sun))/(1+2*abs(sun)*gamma)
        ms_weight = min(1.,max(0.,(alpha-.1)/.5)) if textured else .3
        multiple = .92*abs(sun)/(mu_v+abs(sun))*(hv*h0-1)*(-np.expm1(-vd-ld))/(4*math.pi)*ms_weight
    else:
        delta = np.abs(ld-vd)
        ratio = np.divide(-np.expm1(-delta),delta,out=np.ones_like(delta),where=delta>1e-12)
        single = vd*np.exp(-np.minimum(vd,ld))*ratio
        multiple = 0
    def phase(g):
        value=(1-g*g)/((1+g*g-2*g*theta)**1.5*4*math.pi)
        return value if textured else value*1.5*(1+theta*theta)/(2+g*g)
    forward=.95-.45*min(1.,max(0.,(alpha-.1)/.5)) if textured else .9
    radiance = single*(forward*phase(.436)+(1-forward)*phase(-.65711))+multiple
    return float(np.mean(radiance*mu_p*mu_v/distance**2*r*(~shadow))*2*math.pi)


def check_integral(ctx):
    angles = [(10,1.5,0,25,0),(10,1.5,0,25,90),(30,2,0,25,90),
              (50,3,0,10,60),(10,1.5,0,-25,90),(30,1.5,.1,25,0),
              (60,3,.2,10,45),(15,8,.4,15,90),(.2246899,1.001,0,25,90),
              (10,1.5,0,0,90),(90,2,.1,25,0)]
    cases = [(r,math.sin(math.radians(lat)),f,math.sin(math.radians(sun)),math.radians(phi),.5)
             for lat,r,f,sun,phi in angles]
    cases += [(2,-.5,.1,.4,math.pi,.5),(2,.5,.1,-.4,math.pi,.5),
              (2,.5,0,1.,0.,.5),(2,.5,0,-1.,0.,.5),(2,.5,0,.4,0.,0.)]
    cases += [(1.5,.17,.1,.4,math.pi/2,.5,1),
              (1.5,.17,.1,-.4,math.pi/2,.5,1),
              (3,.7,.2,.1,math.pi/3,.01,1),
              (2,.5,0,.4,math.pi/2,.9,1)]
    rows = np.zeros((len(cases),8),dtype='f4')
    for index,case in enumerate(cases):rows[index,:len(case)]=case
    shader = '''#version 460 core
        layout(local_size_x=32) in;
        layout(std430,binding=0) readonly buffer Input { vec4 inputs[]; };
        layout(std430,binding=1) writeonly buffer Output { float results[]; };
        uniform int count;
        ''' + load_shader('common/ringshine_integral.glsl') + '''
        void main() {
            uint i=gl_GlobalInvocationID.x;
            if(i>=count) return;
            vec4 a=inputs[2*i],b=inputs[2*i+1];
            results[i]=ringshine_band(a.x,a.y,a.z,a.w,b.x,b.y,
                vec4(.436,-.65711,1.,1.),b.z>0.5);
        }'''
    program = ctx.compute_shader(shader)
    inputs,out = ctx.buffer(rows.tobytes()),ctx.buffer(reserve=len(cases)*4)
    inputs.bind_to_storage_buffer(0); out.bind_to_storage_buffer(1)
    program['count'].value=len(cases)
    program.run((len(cases)+31)//32); ctx.memory_barrier()
    actual=np.frombuffer(out.read(),dtype='f4')
    errors=[]
    for case,value in zip(rows,actual):
        ref=reference_band(*map(float,case[:6]),textured=bool(case[6]))
        error=abs(float(value)-ref)/max(ref,1e-10)
        print('band',case[:5], 'GPU',float(value),'reference',ref,'error',error,flush=True)
        assert np.isfinite(value) and value>=0
        assert abs(value-ref)<max(2e-7,ref*.015), (case,value,ref,error)
        errors.append(error if ref>1e-8 else 0)
    print('Maximum band error:',max(errors))
    for obj in (inputs,out,program): obj.release()


def check_cache_and_atlas(ctx):
    gradient=ctx.texture((4096,RING_PROFILE_ROWS),4,dtype='f4')
    gradient.filter=(moderngl.LINEAR,moderngl.LINEAR)
    gradient.repeat_x=gradient.repeat_y=False
    props=ctx.texture((4096,RING_PROPERTY_ROWS),4,dtype='f4')
    props.filter=(moderngl.LINEAR,moderngl.LINEAR)
    props.repeat_x=props.repeat_y=False
    gradient.props_tex=props
    rings=[]
    for host in range(16):
        rings.append(dict(body_idx=host,pole=np.array([0.,1.,0.]),inner_r=1.2,outer_r=2.5,opacity=.5,
            shadow_grad=np.tile([.5+host*.02,.8,1.,1.],(4096,1)).astype('f4'),
            asymmetry=.436,backscatter=-.65711,scatter=1.,unlit_factor=1.,is_textured=False))
    indices=rebuild_ring_gradients_atlas(rings,gradient)
    assert len(indices)==16 and len(set(r['row_idx'] for r in rings))==16
    assert min(r['row_idx'] for r in rings)>=16
    assert gradient.atlas_data[15,0,0]>gradient.atlas_data[0,0,0]
    assert gradient.surface_shadow_filter.rows==16
    primary=build_secondary_ring_properties(rings,gradient,indices,16)
    comparison=build_secondary_ring_properties(rings,None,None,16)
    assert all(np.allclose(a,b) for a,b in zip(primary,comparison))
    vbo=ctx.buffer(np.array([[-1,-1],[1,-1],[-1,1],[1,1]],dtype='f4').tobytes())
    maps=RingshineMap(ctx,load_shader('post/ringshine_map.vert'),load_shader('post/ringshine_map.frag'),vbo)
    hosts=[(0,np.zeros(3),np.array([0.,1.,0.]),(1.2,2.5,1.,1.),.1),
           (15,np.zeros(3),np.array([0.,1.,0.]),(1.2,2.5,1.,1.),.1)]
    stars=np.array([[10.,2.,0.],[10.,-8.,0.]])
    # Blending must not contaminate cached maps, even if enabled by a caller.
    ctx.enable(moderngl.BLEND)
    query=ctx.query(time=True)
    with query: assert maps.update(gradient,props,hosts,stars,4,True)==4
    print('Four initial tiles GPU ms:',query.elapsed/1e6,flush=True)
    initial=np.frombuffer(maps.texture.read(),dtype='f4').reshape(1040,2048,4)
    # Full production radial integration, at a real atlas texel, versus a
    # separate 64-node radial and dense-azimuth reference.
    x,y=90,52
    phi=((x/127*2-1)**1.5)*math.pi
    slat=(y/64*2-1)**1.5
    nodes,weights=np.polynomial.legendre.leggauss(64)
    sun=round(2/math.sqrt(104),5)
    ref=sum(w*reference_band(1.85+.65*t,slat,.1,sun,phi,.5,count=8192)
            for t,w in zip(nodes,weights))*.65*math.pi*.5**2.2
    assert abs(initial[y,x,0]-ref)<ref*.01,(initial[y,x,0],ref)
    assert maps.update(gradient,props,hosts,stars,4,True)==0
    moved=[(i,c+12345,n,p,f) for i,c,n,p,f in hosts]
    assert maps.update(gradient,props,moved,stars+12345,4,True)==0
    assert maps.update(gradient,props,hosts,stars,4,False)==4
    assert maps.update(gradient,props,hosts,stars,8,False)==4
    gradient.profile_revision+=1
    assert maps.update(gradient,props,hosts,stars,8,False)==4
    # Independent stars give distinct hemispheric responses; last host is live.
    pixels=np.frombuffer(maps.texture.read(),dtype='f4').reshape(1040,2048,4)
    assert pixels[:65,:128,:3].max()>0
    assert pixels[975:1040,:128,:3].max()>0
    assert not np.allclose(pixels[:65,:128,:3],pixels[:65,128:256,:3],atol=1e-7)
    near=[(0,np.zeros(3),np.array([0.,1.,0.]),(1.001,2.5,1.,1.),0.)]
    assert maps.update(gradient,props,near,stars[:1],10,True)==1
    near_pixels=np.frombuffer(maps.texture.read(),dtype='f4').reshape(1040,2048,4)
    x,y=104,33
    phi=(x/127*2-1)**1.5*math.pi
    slat=(y/64*2-1)**1.5
    nodes,weights=np.polynomial.legendre.leggauss(96)
    span=math.log(1.5/.001)
    ref=0.
    for t,w in zip(nodes,weights):
        gap=.001*math.exp((t+1)*.5*span)
        ref+=w*reference_band(1+gap,slat,0,sun,phi,.5,count=65536)*gap*span*.5
    ref*=math.pi*.5**2.2
    error=abs(near_pixels[y,x,0]/ref-1)
    print('Near-contact radial map error:',error,flush=True)
    assert error<.01,(near_pixels[y,x,0],ref)
    maps.release(); gradient.surface_shadow_filter.release()
    for obj in (gradient,props,vbo): obj.release()
    print('Atlas capacity, separate stars and cache invalidation: passed',flush=True)


def check_moon_response():
    assert ring_slab_response(.4,.5,0.,True)==0
    assert ring_slab_response(.4,0.,.5,True)==0
    assert ring_slab_response(.4,.5,.001,True)<ring_slab_response(.4,.5,.5,True)
    assert ring_slab_response(.4,1e-6,.5,True)<1e-5
    tau=-math.log(.5)
    assert math.isclose(ring_slab_response(.4,-.4,.5,False),tau/.4*math.exp(-tau/.4),rel_tol=1e-12)
    pos=np.array([[0,0,0],[0,4,0],[-20,10,0]],dtype='f4')
    radii=np.array([1,.05,.1],dtype='f4')
    colors=np.ones((3,3),dtype='f4')
    is_star=np.array([0,0,1],dtype='f4')
    params=np.array([[1.2,2.5,.5,1],[0,0,0,0],[0,0,0,0]],dtype='f4')
    normals=np.tile([0,1,0],(3,1)).astype('f4')
    scattering=np.tile([.436,-.65711,1,0],(3,1)).astype('f4')
    def bounce():
        return compute_planetshine_numba(pos,radii,colors,is_star,pos[2:3],
            colors[2:3],np.ones(1,dtype='f4'),radii[2:3],True,
            params,normals,colors,False,True,scattering)[1][1].sum()
    thick=bounce()
    params[0,2]=.001
    thin=bounce()
    assert 0<thin<thick*.02,(thin,thick)
    pos[2,1]=0
    assert bounce()==0
    print('Moon-ring opacity, grazing Sun and transmission limits: passed',flush=True)


def check_tile_boundaries(ctx):
    data=np.zeros((1040,2048,4),dtype='f4')
    for star in range(16):
        for host in range(16):
            data[host*65:(host+1)*65,star*128:(star+1)*128,:3]=(star*16+host+1)/256
    tex=ctx.texture((2048,1040),4,data.tobytes(),dtype='f4')
    tex.filter=(moderngl.LINEAR,moderngl.LINEAR)
    tex.use(8)
    cases=np.array([[x,(h+y)/16,h,s] for s in (0,1,15) for h in (0,4,15)
                    for x in (0,1) for y in (0,1)],dtype='f4')
    shader='''#version 460 core
        layout(local_size_x=64) in;
        uniform sampler2D u_ringshine_map;
        layout(std430,binding=0) readonly buffer Input { vec4 inputs[]; };
        layout(std430,binding=1) writeonly buffer Output { float values[]; };
        uniform int count;
        '''+load_shader('common/ringshine_lookup.glsl')+'''
        void main(){uint i=gl_GlobalInvocationID.x;if(i>=count)return;
            vec4 p=inputs[i]; values[i]=ringshine_sample(p.xy,int(p.z),int(p.w)).r;}
        '''
    program=ctx.compute_shader(shader)
    inp,out=ctx.buffer(cases.tobytes()),ctx.buffer(reserve=len(cases)*4)
    inp.bind_to_storage_buffer(0);out.bind_to_storage_buffer(1)
    program['u_ringshine_map'].value=8;program['count'].value=len(cases)
    program.run((len(cases)+63)//64);ctx.memory_barrier()
    actual=np.frombuffer(out.read(),dtype='f4')
    expected=(cases[:,3]*16+cases[:,2]+1)/256
    assert np.allclose(actual,expected,rtol=0,atol=1e-6),(actual,expected)
    for obj in (program,inp,out,tex):obj.release()
    print('Polar and longitude tile boundaries: passed',flush=True)


def check_programs(ctx):
    for name in ('sphere','terrain','ring'):
        program=ctx.program(vertex_shader=load_shader('celestial/'+name+'.vert'),
                            fragment_shader=load_shader('celestial/'+name+'.frag'))
        program.release()
    for quality,bounded in ((1,False),(2,False),(3,False),(3,True)):
        source=specialize_atmosphere(load_shader('atmosphere/atmo.frag'),quality,bounded)
        program=ctx.program(vertex_shader=load_shader('atmosphere/atmo.vert'),fragment_shader=source)
        program.release()
    program=ctx.program(vertex_shader=load_shader('atmosphere/sky_view_lut.vert'),
                        fragment_shader=load_shader('atmosphere/sky_view_lut.frag'))
    program.release()
    program=ctx.compute_shader(load_shader('atmosphere/aerial_perspective.comp'))
    program.release()
    print('Surface, ring and all atmospheric shader programs: compiled',flush=True)


if __name__=='__main__':
    context=moderngl.create_standalone_context(require=460)
    print(context.info['GL_RENDERER'],flush=True)
    check_integral(context)
    check_cache_and_atlas(context)
    check_moon_response()
    check_tile_boundaries(context)
    check_programs(context)
    context.release()
    print('Ringshine regressions passed')
