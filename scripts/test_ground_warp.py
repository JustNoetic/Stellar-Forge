"""Run ground-warp regressions with the engine Python and an OpenGL 4.6 GPU.

Checks collision merge order, exact frame-local surface-query reuse, and the
production terrain shader's camera-relative transform through one rotation.
"""
from pathlib import Path
import os, sys, json, math, time, statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import moderngl
from engine.rendering.terrain_collision import terrain_frame, terrain_camera_split
from engine.rendering.planetshine import _build_rotation_props
from engine.physics.physics_core import first_body_collision

def reference(positions,radii):
    for i in range(len(positions)):
        for j in range(i+1,len(positions)):
            dx,dy,dz=positions[i]-positions[j]
            r=radii[i]+radii[j]
            if dx*dx+dy*dy+dz*dz<r*r:return i,j
    return -1,-1
rng=np.random.default_rng(20261004)
for dtype in ('f4','f8'):
    for n in (0,1,2,8,56):
        for trial in range(20):
            positions=rng.normal(size=(n,3));radii=rng.uniform(0,.4,n).astype(dtype)
            assert first_body_collision(positions,radii)==reference(positions,radii)
    assert first_body_collision(np.array([[0.,0,0],[2.,0,0],[0.,0,0]]),np.ones(3,dtype=dtype))==(0,2)
    assert first_body_collision(np.array([[0.,0,0],[2.,0,0]]),np.ones(2,dtype=dtype))==(-1,-1)
positions=rng.normal(size=(56,3));radii=np.zeros(56)
def bench(fn,n):
    start=time.perf_counter()
    for _ in range(n):fn(positions,radii)
    return (time.perf_counter()-start)*1000/n
cpu_before=bench(reference,100);cpu_after=bench(first_body_collision,20000)

data=json.loads((ROOT/'data/system.json').read_text())
earth=next(b for b in (data if isinstance(data,list) else data['bodies']) if b['name']=='Earth')
pole=_build_rotation_props([earth])[4][0]
angles=np.linspace(0,2*math.pi,8192,endpoint=False,dtype='f4')
point=np.array([.73,.41,-.546]);point/=np.linalg.norm(point)
ground=(point*6378.137).astype('f4');camera=ground.astype('f8')+point*.001
frames=np.array([terrain_frame(pole,float(a)) for a in angles])
camera_world=np.einsum('nij,j->ni',frames,camera)
inputs=np.zeros((len(angles),12),dtype='f4')
inputs[:,:3]=-camera_world/149597870.7;inputs[:,3]=angles
for i,(position,angle) in enumerate(zip(camera_world,angles)):
    high,low=terrain_camera_split(position,pole,float(angle));inputs[i,4:7]=high;inputs[i,8:11]=low
source=(ROOT/'engine/glsl/celestial/terrain.vert').read_text()
begin=source.index('    vec3 p_cam_rel;')
end=source.index('    // Apply atmospheric refraction',begin)
block=source[begin:end]
ctx=moderngl.create_standalone_context(require=460)
program=ctx.compute_shader('''#version 460
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer In {vec4 input_data[];};
layout(std430,binding=1) writeonly buffer Out {vec4 result[];};
uniform vec3 point_km,pole_raw;
uniform bool u_local_camera_enabled;
const float u_km_to_au=1.0/149597870.7;
void main(){
 uint i=gl_GlobalInvocationID.x;
 vec4 d=input_data[3*i];
 vec3 u_body_camera_high_km[8];u_body_camera_high_km[0]=input_data[3*i+1].xyz;
 vec3 u_body_camera_low_km[8];u_body_camera_low_km[0]=input_data[3*i+2].xyz;
 vec3 pole=normalize(pole_raw),ref=abs(pole.y)>.999?vec3(1,0,0):vec3(0,1,0);
 vec3 tangent=normalize(cross(pole,ref)),bitangent=normalize(cross(pole,tangent));
 float c_rot=cos(d.w),s_rot=sin(d.w);
 vec3 p_local_km=point_km;
 vec3 body=vec3(point_km.x*c_rot-point_km.z*s_rot,point_km.y,point_km.x*s_rot+point_km.z*c_rot);
 vec3 p_world_km=body.x*tangent+body.y*pole+body.z*bitangent;
 vec3 body_cam_rel=d.xyz;int body_slot=0;
'''+block+'''result[i]=vec4(p_cam_rel,0);}
''')
program['point_km'].value=tuple(ground);program['pole_raw'].value=tuple(pole)
source_buf=ctx.buffer(inputs.tobytes());source_buf.bind_to_storage_buffer(0)
out=ctx.buffer(reserve=len(angles)*16);out.bind_to_storage_buffer(1)
measurements={}
for enabled in (False,True):
    program['u_local_camera_enabled'].value=enabled
    program.run(group_x=len(angles)//64);ctx.memory_barrier()
    world=np.frombuffer(out.read(),dtype='f4').reshape(-1,4)[:,:3].astype('f8')*149597870.7
    local=np.einsum('nji,nj->ni',frames,world)
    error=(local-(ground.astype('f8')-camera))*1000
    measurements['fixed' if enabled else 'previous']={'max_error_m':float(np.linalg.norm(error,axis=1).max()),'rms_error_m':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),'max_consecutive_jitter_m':float(np.linalg.norm(np.diff(local,axis=0)*1000,axis=1).max())}
assert measurements['fixed']['max_error_m']<.00001,measurements
result={'collision_regression_cases':204,'collision_scan_before_ms':cpu_before,'collision_scan_after_ms':cpu_after,'gpu':ctx.info['GL_RENDERER'],'camera_gpu_test':measurements}

# Repeated clearance checks share work only when the caller explicitly confines
# the object to a single residency frame. A moved query always reevaluates.
from engine.rendering.terrain_collision import CameraSurface
class CountingSurface(CameraSurface):
    def __init__(self, cache_queries):
        super().__init__(6378., 0., [0, 1, 0], cache_queries=cache_queries)
        self.calls = 0
    def _surface(self, position):
        self.calls += 1
        return super()._surface(position)
for cached in (False, True):
    surface = CountingSurface(cached)
    p = np.array([6378.001, 0., 0.])
    first = surface.clearance(p)
    assert surface.clearance(p.copy()) == first
    assert surface.calls == (1 if cached else 2)
    surface.clearance(p + [.001, 0., 0.])
    assert surface.calls == (2 if cached else 3)

print(json.dumps(result,indent=2))
