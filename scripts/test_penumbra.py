"""Offscreen eclipse regressions: curve, optics, units and production linking.

Run with a Python environment containing numpy and moderngl. No app window,
settings changes, or astronomical data downloads are needed.
"""
import argparse
from pathlib import Path
import re
import numpy as np
import moderngl

AU = 149597870.7
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--overlay', type=Path)
    args = parser.parse_args()
    def load(path):
        candidates = ([args.overlay / 'engine/glsl' / path] if args.overlay else [])
        candidates.append(args.root / 'engine/glsl' / path)
        file = next(p for p in candidates if p.is_file())
        text = file.read_text(encoding='utf-8')
        def include(m):
            name = m[1]
            relative = (Path(path).parent / name).as_posix()
            try:
                return load(relative)
            except StopIteration:
                return load(name)
        return re.sub(r'^\s*#include\s+"([^"]+)"\s*$', include, text, flags=re.M)

    ctx = moderngl.create_standalone_context(require=460)
    print('GPU:', ctx.info['GL_RENDERER'])
    core = load('common/eclipse_shadow.glsl')
    program = ctx.compute_shader('''#version 460
        layout(local_size_x=64) in;
        layout(std430,binding=0) readonly buffer Inputs { vec4 inputs[]; };
        layout(std430,binding=1) writeonly buffer Outputs { vec4 outputs[]; };
        uniform uint count;
    ''' + core + '''
        void main() {
            uint i=gl_GlobalInvocationID.x;
            if(i>=count) return;
            vec4 geometry=inputs[i*4], grazing=inputs[i*4+1];
            vec4 ozone=inputs[i*4+2], params=inputs[i*4+3];
            outputs[i]=vec4(casterShadowTerm(geometry.x,geometry.y,geometry.z,
                params.x,grazing,ozone,params.y,geometry.w,params.z),1.);
        }
    ''')
    def evaluate(rows):
        inp = ctx.buffer(np.asarray(rows, 'f4').tobytes())
        out = ctx.buffer(reserve=len(rows)*16)
        inp.bind_to_storage_buffer(0)
        out.bind_to_storage_buffer(1)
        program['count'].value = len(rows)
        program.run((len(rows)+63)//64)
        ctx.memory_barrier()
        values = np.frombuffer(out.read(), 'f4').reshape(-1,4).copy()
        inp.release(); out.release()
        assert np.isfinite(values).all()
        assert ((values>=0)&(values<=1)).all()
        return values[:,:3]

    rng = np.random.default_rng(27)
    n = 100000
    rows = np.zeros((n,4,4), 'f4')
    distance = 10**rng.uniform(3,7,n)
    alpha = 10**rng.uniform(-4,-1,n)
    radius = distance*alpha*10**rng.uniform(-2,2,n)
    footprint = distance*alpha
    perpendicular = (radius+footprint)*rng.uniform(0,1.2,n)
    rows[:,0] = np.column_stack((footprint,radius,perpendicular,distance))
    rows[:,3,2] = 1
    geometry = rows[:,0].astype(float)
    # Independent angular reference retains the original cubic and 0.625 power.
    a,b,g = geometry[:,0]/geometry[:,3], geometry[:,1]/geometry[:,3], geometry[:,2]/geometry[:,3]
    t = np.clip((a+b-g)/(a+b-np.abs(b-a)),0,1)
    maximum = np.minimum(1,(b/a)**2)
    expected = np.clip(1-maximum*t*t*(3-2*t),0,1)**.625
    actual = evaluate(rows)
    error = np.max(np.abs(actual-expected[:,None]))
    assert error < 8e-5, error
    print(f'100,000 geometric cases: maximum absolute error {error:.3g}')

    # Uniform rescaling must leave a physically identical scene unchanged.
    au_rows = rows.copy()
    au_rows[:,0] /= AU
    au_rows[:,3,2] = AU
    np.testing.assert_allclose(evaluate(au_rows),actual,atol=8e-5,rtol=0)
    print('AU/km equivalence passed')

    # Total eclipse, annular eclipse, tiny star, zero-width source, and zero caster.
    special = np.zeros((7,4,4),'f4')
    special[:,0] = [(1,2,0,1000),(2,1,0,1000),(1e-5,1e-5,0,1),
                    (0,2,0,1000),(0,2,2,1000),(1,0,0,1000),(1,2,3,1000)]
    special[:,3,2] = 1
    result = evaluate(special)
    np.testing.assert_allclose(result[:,0],[0,.75**.625,0,0,1,1,1],atol=1e-6)
    print('Total/annular eclipses, tiny stars and point-source boundaries passed')

    # Reference the pre-change optical equations at a projected oblate limb.
    count=20000
    optical=np.zeros((count,4,4),'f4')
    H=rng.uniform(2,30,count)
    ref_radius=rng.uniform(1000,50000,count)
    projected=ref_radius*rng.uniform(.7,1,count)
    D=rng.uniform(100000,2000000,count)
    F=D*rng.uniform(.0002,.01,count)
    P=(projected+F)*rng.uniform(.3,1.1,count)
    bend=rng.uniform(.001,.08,count)
    tau=rng.uniform(.05,2,(count,3))
    optical[:,0]=np.column_stack((F,projected,P,D))
    optical[:,1,:3]=tau*np.sqrt(2*np.pi*ref_radius/H)[:,None]
    optical[:,1,3]=1/ref_radius
    optical[:,2,:3]=rng.uniform(0,2,(count,3))
    optical[:,2,3]=25
    optical[:,3]=np.column_stack((bend,H,np.ones(count),np.zeros(count)))
    # Evaluate the original optical model in angular coordinates and float64.
    q=optical.astype(float)
    F,R,P,D=q[:,0].T
    bend,H=q[:,3,:2].T
    a,b,g=F/D,R/D,P/D
    t=np.clip((a+b-g)/(a+b-np.abs(b-a)),0,1)
    blend=t*t*(3-2*t)
    maximum=np.minimum(1,(b/a)**2)
    expected=(np.clip(1-maximum*blend,0,1)**.625)[:,None]*np.ones((1,3))
    req=b-g-a*.8
    depth=np.clip(np.maximum(0,req)/bend,0,1)
    z=-H*np.log(np.maximum(depth,1e-5))
    z_diff=(z-q[:,2,3])/np.maximum(.707*H,.1)
    tau_R=q[:,1,:3]*np.sqrt(R*q[:,1,3])[:,None]*depth[:,None]
    tau_O3=q[:,2,:3]*np.exp(-.5*z_diff*z_diff)[:,None]
    defocus=1/(1+D*np.maximum(req,1e-6)/H)
    focal=np.clip(D/(R/bend),0,1)
    off_axis=np.clip(a/np.maximum(1e-6,g),0,1)
    intensity=np.clip(2*H/np.maximum(a*D,1e-9)*focal*off_axis,0,1)*defocus
    fade_t=np.clip((1-depth)/.25,0,1)
    fade=fade_t*fade_t*(3-2*fade_t)
    refract=intensity*(1-depth*.7)*fade*blend
    refract[(req>bend)|(g>=a+b)]=0
    expected=np.clip(expected+np.exp(-tau_R-tau_O3)*refract[:,None],0,1)
    actual=evaluate(optical)
    error=np.max(np.abs(expected-actual))
    assert error<8e-5,error
    print(f'20,000 refractive/oblate cases: maximum absolute error {error:.3g}')
    optical_au=optical.copy()
    optical_au[:,0]/=AU
    optical_au[:,3,2]=AU
    np.testing.assert_allclose(evaluate(optical_au),actual,atol=8e-5,rtol=0)
    print('Refraction AU/km equivalence passed')

    linked=0
    for name in ('sphere','terrain','ring','point_celestial'):
        p=ctx.program(vertex_shader=load('celestial/'+name+'.vert'),fragment_shader=load('celestial/'+name+'.frag'))
        p.release(); linked+=1
    atmo_vertex=load('atmosphere/atmo.vert')
    atmo_fragment=load('atmosphere/atmo.frag')
    for quality,bounded in ((1,0),(2,0),(3,0),(3,1)):
        version,rest=atmo_fragment.split('\n',1)
        fragment=f'{version}\n#define ATMO_QUALITY {quality}\n#define ATMO_BOUNDED_SHADOWS {bounded}\n{rest}'
        for lowres in (False,True):
            f=fragment.replace('layout(location = 0, index = 1)','layout(location = 1)') if lowres else fragment
            p=ctx.program(vertex_shader=atmo_vertex,fragment_shader=f)
            p.release(); linked+=1
    p=ctx.program(vertex_shader=load('atmosphere/sky_view_lut.vert'),fragment_shader=load('atmosphere/sky_view_lut.frag'))
    p.release(); linked+=1
    p=ctx.compute_shader(load('atmosphere/aerial_perspective.comp'))
    p.release(); linked+=1
    p=ctx.program(vertex_shader=atmo_vertex,fragment_shader=load('atmosphere/atmo_godrays.frag'))
    p.release(); linked+=1
    print(f'{linked} production shader programs compiled and linked')

    # Probe actual GLSL buffer layouts with distinct thickness/H/grazing sentinels.
    def block(text,marker):
        start=text.index(marker); end=text.index('};',start)+2
        return text[start:end]
    atmo_block=block(atmo_fragment,'layout(std430, binding = 8) buffer AtmoData')
    scene_block=block(load('celestial/sphere.frag'),'layout(std140, binding = 1) uniform SceneData')
    p=ctx.compute_shader('#version 460\n#define MAX_STARS 16\n#define MAX_CASTERS 64\n'+atmo_block+scene_block+'''
        layout(local_size_x=1) in;
        layout(std430,binding=0) writeonly buffer TestOutput { vec4 output_data; };
        void main() { output_data=vec4(u_active_caster_atmos[3].w,
            u_active_scale_height[3],u_caster_grazing[5].x,u_caster_grazing[5].w); }
    ''')
    data=np.zeros(264,'f4'); data[108+3*4+3]=100; data[256+3]=8.5
    scene=np.zeros(2088,'f4'); scene[1832+5*4]=3.25; scene[1832+5*4+3]=1/6371
    a=ctx.buffer(data.tobytes()); a.bind_to_storage_buffer(8)
    s=ctx.buffer(scene.tobytes()); s.bind_to_uniform_block(1)
    o=ctx.buffer(reserve=16); o.bind_to_storage_buffer(0)
    p.run(); ctx.memory_barrier()
    np.testing.assert_allclose(np.frombuffer(o.read(),'f4'),[100,8.5,3.25,1/6371],rtol=1e-6)
    for resource in (a,s,o,p,program): resource.release()
    assert ctx.error=='GL_NO_ERROR',ctx.error
    print('Production buffer offsets and separate thickness/scale-height/grazing data passed')
    ctx.release()


if __name__=='__main__':
    main()
