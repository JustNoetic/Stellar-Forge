"""Terrain regressions: native streaming, precision, topology and GPU seams.

Run: python -m pytest scripts/test_terrain_native.py -q
GPU checks require an OpenGL 4.6 context (the terrain shaders' existing target).
"""
import importlib
import pathlib
import re
import sys
import time
import types
import numpy as np
from PIL import Image
import pytest
import stellar_terrain as native

ROOT = pathlib.Path(__file__).resolve().parents[1]
GLSL = ROOT / 'engine' / 'glsl'


def shader(name):
    def expand(path):
        def include(match):
            target=GLSL/match[1]
            if not target.exists():
                target=path.parent/match[1]
            return expand(target)
        return re.sub(r'#include\s+"([^"]+)"',include,path.read_text())
    return expand(GLSL/name)


@pytest.fixture
def adapters(monkeypatch):
    # Import rendering adapters without starting the engine's legacy aggregate
    # __init__, which initializes unrelated scientific/GUI modules.
    package = types.ModuleType('engine')
    package.__path__ = [str(ROOT/'engine')]
    monkeypatch.setitem(sys.modules,'engine',package)
    rendering=types.ModuleType('engine.rendering')
    rendering.__path__=[str(ROOT/'engine/rendering')]
    monkeypatch.setitem(sys.modules,'engine.rendering',rendering)
    sys.path.insert(0,str(ROOT))
    try:
        yield importlib.import_module('engine.rendering.terrain_quadtree'), importlib.import_module('engine.rendering.terrain_streamer')
    finally:
        sys.path.remove(str(ROOT))


@pytest.fixture
def ctx():
    import moderngl
    context = moderngl.create_standalone_context(require=460)
    yield context
    context.release()


def make_tiles(base, size=16, lods=1, value=32768):
    for lod in range(lods+1):
        for face in range(6):
            path = base/'Test'/'height'/str(face)/str(lod)
            path.mkdir(parents=True,exist_ok=True)
            for y in range(1 << lod):
                for x in range(1 << lod):
                    arr = np.full((size+4,size+4),value,dtype='u2')
                    Image.fromarray(arr).save(path/f'{x}_{y}.png')


def fill_backend(backend,lod=1):
    for face in range(6):
        for y in range(1 << lod):
            for x in range(1 << lod):
                backend.request('Test','height',face,lod,x,y,None)
    deadline=time.monotonic()+5
    uploads=[]
    target=6*sum(4**i for i in range(lod+1))
    while backend.resident_count<target and time.monotonic()<deadline:
        item=backend.poll()
        if item is not None:
            uploads.append(item)
        else:
            time.sleep(.005)
    assert backend.resident_count==target,backend.errors()
    return uploads


def test_native_view_and_pack(adapters):
    tree_module,_=adapters
    tree=tree_module.PlanetQuadtree(6371.)
    raw=tree.traverse_raw([0.,0.,6371.002],45.,1080.,max_lod=16,max_patches=128,height_max_km=8.)
    assert raw.shape[1]==10 and 0<len(raw)<=128
    staging=np.zeros((128,24),'f4');n=len(raw)
    one=np.ones(n,'f4');zero=np.zeros(n,'f4')
    tree_module.pack_terrain_patches(staging,0,raw,one,zero,zero,zero,2.,one,zero,zero,one*768,-10.,20.,3.,0.,0.)
    assert np.array_equal(staging[:n,20],raw[:,9])
    assert np.all(staging[:n,21]==1.) and np.all(staging[:n,11]==2.)
    with pytest.raises(ValueError):
        native.traverse((0.,0.,0.),float('nan'),0.,1080.,.4,75.,8,128,[],0.,0.,0.)


def test_rotation_preserves_full_selection_and_hidden_history(adapters):
    tree_module, _ = adapters
    tree = tree_module.PlanetQuadtree(6371.)
    args = dict(max_lod=16, max_patches=512, height_max_km=8., surface_altitude_km=.01)
    camera = [0., 0., 6371.01]
    full = tree.traverse_raw(camera, 45., 1080., **args)
    full_history = tree._history
    for plane in [[1.,0.,0.,0.],[-1.,0.,0.,0.],[0.,1.,0.,0.],[0.,-1.,0.,0.]]:
        visible = tree.traverse_raw(camera, 45., 1080., frustum_planes=[plane], **args)
        assert 0 < len(visible) <= len(full)
        assert tree._history == full_history
        expected = {tuple(row[4:8]): row for row in full}
        for row in visible:
            assert np.array_equal(row, expected[tuple(row[4:8])])
    tree.traverse_raw(camera, 45., 1080., frustum_planes=[[0.,0.,1.,-100000.]], **args)
    assert tree._history == full_history # offscreen nodes retain hysteresis
    restored = tree.traverse_raw(camera, 45., 1080., **args)
    assert np.array_equal(restored, full)


def test_settled_selection_reused_but_view_changes_invalidate(adapters, monkeypatch):
    tree_module, _ = adapters
    calls = []
    real_traverse = native.traverse
    def counted(*args):
        calls.append(args)
        return real_traverse(*args)
    monkeypatch.setattr(native, 'traverse', counted)
    tree = tree_module.PlanetQuadtree(6371.)
    camera = [0.,0.,6471.]
    args = dict(max_lod=12, max_patches=512, surface_altitude_km=100.)
    first = tree.traverse_raw(camera, 3., 1080., **args)
    second = tree.traverse_raw(camera, 3., 1080., **args)
    assert np.array_equal(first, second)
    assert len(calls) == 2
    for plane in [[1.,0.,0.,0.], [-1.,0.,0.,0.], [0.,0.,1.,-6471.001]]:
        tree.traverse_raw(camera, 3., 1080., frustum_planes=[plane], **args)
    assert len(calls) == 2 # look-around updates culling, not subdivision
    tree.traverse_raw([.001,0.,6471.], 3., 1080., **args)
    assert len(calls) == 3
    tree.traverse_raw([.001,0.,6471.], .1, 1080., **args)
    assert len(calls) == 4
    tree.traverse_raw([.001,0.,6471.], .1, 1080., **(args | {'max_lod': 8}))
    assert len(calls) == 5
    tree.update_radius(6370.)
    assert tree._history is None
    tree.traverse_raw([.001,0.,6471.], .1, 1080., **args)
    assert len(calls) == 6


def test_lod_altitude_tracks_displaced_oblate_surface(tmp_path):
    make_tiles(tmp_path, lods=0, value=40123)
    backend = native.TileBackend(str(tmp_path), 64, 16, 2)
    try:
        fill_backend(backend, 0)
        for obl in [0., .1]:
            surface = np.frombuffer(backend.surface_vertices(
                'Test', 4, 0, 0, 0, np.array([[.63,.72]],'<f8').tobytes(),
                6371., obl, -10., 20., None, 0, 16), '<f8')
            camera = surface + surface / np.linalg.norm(surface) * .002
            altitude = backend.surface_altitude('Test', tuple(camera), 6371., obl, -10., 20., None)
            assert abs(altitude - .002) < 1e-8
    finally:
        backend.shutdown()


def test_camera_travel_keeps_local_pixel_subdivision(adapters):
    tree_module, _ = adapters
    tree = tree_module.PlanetQuadtree(6371.)
    args = dict(max_lod=16, max_patches=512, height_max_km=8.)
    for altitude in [100., 10., 1., .1, .01]:
        for u in [.173 - 1e-5, .173, .173 + 1e-5]:
            direction = tree_module.cube_to_sphere_point(4, u, -.217)
            camera = direction * (6371. + altitude)
            raw = tree.traverse_raw(camera, 45., 1080., surface_altitude_km=altitude, **args)
            # The patch containing the camera's ground point must remain fine.
            # Empty horizon parents used to force it down to LOD 1 or 2.
            ground = raw[(raw[:, 4] == 4) & (raw[:, 0] <= u) & (raw[:, 2] >= u)
                         & (raw[:, 1] <= -.217) & (raw[:, 3] >= -.217)]
            assert len(ground) == 1
            assert ground[0, 5] >= 6, (altitude, ground)
            assert len(raw) <= 512
            repeated = tree.traverse_raw(camera, 45., 1080., surface_altitude_km=altitude, **args)
            assert np.array_equal(repeated, raw)


@pytest.mark.parametrize('altitude,height', [(100., 9.2), (10., 9.2), (.01, 0.)])
@pytest.mark.parametrize('bend', [0., .02])
def test_zoom_into_empty_sky_submits_no_patches(adapters, altitude, height, bend):
    tree_module, _ = adapters
    tree = tree_module.PlanetQuadtree(6371.)
    camera = [0., 0., 6371. + altitude]
    args = dict(max_lod=12, max_patches=4096, height_max_km=height,
                surface_altitude_km=altitude, refract_bend=bend,
                frustum_bend=tree_module.terrain_refraction_cull_bend(camera, 6371., 0., bend, 8.5))
    for fov in [45., 15., 3., .1]:
        tangent = np.tan(np.radians(fov / 2))
        # Forward is +Z, away from the planet. Same six world/local planes as
        # the app's perspective frustum, with a deliberately tiny near plane.
        planes = [[1.,0.,tangent,-tangent*camera[2]],[-1.,0.,tangent,-tangent*camera[2]],
                  [0.,1.,tangent,-tangent*camera[2]],[0.,-1.,tangent,-tangent*camera[2]],
                  [0.,0.,1.,-camera[2]-.001],[0.,0.,-1.,camera[2]+100000.]]
        planes = np.asarray(planes)
        planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
        raw = tree.traverse_raw(camera, fov, 1080., frustum_planes=planes, **args)
        assert len(raw) == 0, (altitude, fov, len(raw))
        # Retain hidden subdivision history so returning to Earth stays stable.
        assert tree._history
    visible = tree.traverse_raw(camera, 45., 1080.,
                                frustum_planes=[[0.,0.,-1.,camera[2]-.001]], **args)
    assert len(visible) > 0


def test_ocean_depth_does_not_expand_culling_above_eye(adapters):
    tree_module, _ = adapters
    tree = tree_module.PlanetQuadtree(6371.)
    raw = tree.traverse_raw([0., 0., 6381.], 3., 1080., max_lod=12,
                            max_patches=4096, height_max_km=10.35,
                            height_cull_range=(-10.35, 9.2),
                            surface_altitude_km=10.,
                            frustum_planes=[[0.,0.,1.,-6381.001]])
    assert len(raw) == 0


def test_16_bit_streaming_and_reload(tmp_path):
    make_tiles(tmp_path,lods=0,value=32769)
    backend=native.TileBackend(str(tmp_path),64,16,2)
    try:
        uploads=fill_backend(backend,0)
        slot,_,data=uploads[0]
        assert abs(np.frombuffer(data,'<f4')[0]-32769/65535)<1e-7
        sampled=backend.sample_height(slot,np.array([[.5,.5]],'<f8').tobytes())
        assert abs(np.frombuffer(sampled,'<f8')[0]-32769/65535)<1e-7
        old_revision=backend.revision
        make_tiles(tmp_path,lods=0,value=40001)
        backend.reload('Test','height')
        assert backend.resident_count==0 and backend.revision>old_revision
        uploads=fill_backend(backend,0)
        assert all(abs(np.frombuffer(data,'<f4')[0]-40001/65535)<1e-7 for _,_,data in uploads)
    finally:
        backend.shutdown()


def test_missing_tiles_and_case_paths(tmp_path):
    backend=native.TileBackend(str(tmp_path),64,16,1)
    try:
        assert backend.max_lod('absent','height')==-1
        assert backend.resolve('ABSENT','height',4,20,0,0)[0]==-1.
        assert backend.resolve('ABSENT','diffuse',4,20,0,0)[0]==0.
        assert backend.poll() is None
    finally:
        backend.shutdown()


def test_source_boundary_blends_to_common_parent(tmp_path):
    make_tiles(tmp_path,lods=1)
    Image.fromarray(np.full((20,20),50000,'u2')).save(tmp_path/'Test/height/4/1/0_0.png')
    backend=native.TileBackend(str(tmp_path),256,16,2)
    try:
        fill_backend(backend,1)
        grid=np.array([[1.,.5]],'<f8').tobytes()
        args=(6371.,0.,0.,1.,None,0,16)
        a=np.frombuffer(backend.surface_vertices('Test',4,1,0,0,grid,*args),'<f8')
        b=np.frombuffer(backend.surface_vertices('Test',4,1,1,0,np.array([[0.,.5]],'<f8').tobytes(),*args),'<f8')
        assert np.max(abs(a-b))<1e-8
    finally:
        backend.shutdown()


def test_shader_compilation(ctx):
    program=ctx.program(vertex_shader=shader('celestial/terrain.vert'),fragment_shader=shader('celestial/terrain.frag'))
    program.release()
    for name in ('terrain_cache_height.comp','terrain_cache_normal.comp'):
        program=ctx.compute_shader(shader('compute/'+name))
        program.release()


def test_gpu_height_normal_and_cpu_surface_agree(ctx,tmp_path,adapters):
    _,stream_module=adapters
    make_tiles(tmp_path,size=16,lods=1,value=40123)
    streamer=stream_module.TerrainTileStreamer(ctx,str(tmp_path),256,16)
    try:
        for face in range(6):
            for y in range(2):
                for x in range(2):
                    streamer.request_tile('Test','height',face,1,x,y)
        deadline=time.monotonic()+5
        while streamer.resident_count<30 and time.monotonic()<deadline:
            streamer.begin_frame();streamer.process_uploads(64,100.);time.sleep(.005)
        assert streamer.resident_count==30
        streamer.prepare_height_table({0:'Test'});streamer.use()
        source='''#version 460 core
layout(local_size_x=1) in;
'''+shader('common/terrain_surface.glsl')+'''
layout(std430,binding=0) readonly buffer Input { vec4 points[]; };
layout(std430,binding=1) writeonly buffer Output { vec4 results[]; };
void main(){uint i=gl_GlobalInvocationID.x;vec3 n=normalize(points[i].xyz);
vec4 range=vec4(-10.0,20.0,0,0), detail=vec4(0,0,0,0);
float h=terrain_elevation(0u,n,range,detail);
results[i]=vec4(terrain_surface_normal(0u,n,6371.0,0.0,range,detail,h),h);}
'''
        program=ctx.compute_shader(source);streamer.configure_height_program(program)
        points=np.array([[0.,0.,1.,0.],[.1,-.2,1.,0.],[0.,-.5,1.,0.]],'f4')
        input_buffer=ctx.buffer(points.tobytes());output=ctx.buffer(reserve=len(points)*16)
        input_buffer.bind_to_storage_buffer(0);output.bind_to_storage_buffer(1)
        program.run(group_x=len(points));ctx.memory_barrier()
        result=np.frombuffer(output.read(),'<f4').reshape(-1,4)
        for i,p in enumerate(points):
            n=p[:3].astype('f8');n/=np.linalg.norm(n)
            uv=np.arctan([n[0]/n[2],-n[1]/n[2]])/(np.pi/4)*.5+.5
            vertices=streamer.surface_vertices('Test',4,0,0,0,np.array([uv]),6371.,0.,-10.,20.)
            height=np.linalg.norm(vertices[0])-6371.
            assert abs(height-result[i,3])<.00005 # 5 cm, including float shader noise
            assert abs(np.linalg.norm(result[i,:3])-1.)<1e-5
        # Two mesh LODs evaluate precisely the same world position.
        a=streamer.surface_vertices('Test',4,1,0,0,np.array([[1.,.5]]),6371.,0.,-10.,20.)
        b=streamer.surface_vertices('Test',4,2,2,1,np.array([[0.,0.]]),6371.,0.,-10.,20.)
        assert np.max(abs(a-b))<1e-8
        program.release();input_buffer.release();output.release()
    finally:
        streamer.shutdown()


def test_baker_preserves_precision_and_gutters(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    try:
        from terrain_tile_baker import bake_tiles
        source=tmp_path/'source.png'
        arr=(32000+np.arange(64,dtype='u2')[None,:]+np.zeros((32,1),'u2'))
        Image.fromarray(arr).save(source)
        out=tmp_path/'baked'
        bake_tiles('Test',source,'height',1,16,out)
        values=np.asarray(Image.open(out/'4/1/0_0.png'))
        assert values.shape==(20,20)
        assert values.min()>31000 and values.max()<33000
        assert len(np.unique(values))>8
        left=np.asarray(Image.open(out/'4/1/0_0.png'))
        right=np.asarray(Image.open(out/'4/1/1_0.png'))
        # Neighbour border samples are exactly the same source locations.
        assert np.array_equal(left[:,-4:],right[:,:4])
        # The inspector imports this entry point as a module, rather than
        # invoking the CLI from scripts/. Both routes must find the v2 baker.
        sys.path.insert(0,str(ROOT))
        try:
            from scripts.bake_planet_tiles import bake_planet
            bake_planet('Test',str(source),'height',0,16,str(tmp_path/'entry'),False)
            assert (tmp_path/'entry/Test/height/4/0/0_0.png').exists()
        finally:
            sys.path.remove(str(ROOT))
    finally:
        sys.path.remove(str(ROOT/'scripts'))


def test_geometry_cache_uses_shared_boundary_normals(ctx,tmp_path,adapters):
    tree_module,stream_module=adapters
    cache_module=importlib.import_module('engine.rendering.terrain_geometry_cache')
    make_tiles(tmp_path,size=16,lods=0,value=40123)
    streamer=stream_module.TerrainTileStreamer(ctx,str(tmp_path),256,16)
    cache=cache_module.TerrainGeometryCache(ctx,16,max_bytes=1024*1024,max_bakes_per_frame=1)
    try:
        for face in range(6):
            streamer.request_tile('Test','height',face,0,0,0)
        deadline=time.monotonic()+5
        while streamer.resident_count<6 and time.monotonic()<deadline:
            streamer.begin_frame();streamer.process_uploads(64,100.);time.sleep(.005)
        streamer.prepare_height_table({0:'Test'});streamer.use()
        slot=streamer.get_tile_slot_or_fallback('Test','height',4,0,0,0)[0]
        raw=np.array([[0,-1,1,0,4,1,1,0,1000,0],[-.5,-1,0,-.5,4,2,1,0,500,2]],'f4')
        one=np.ones(2,'f4');zero=np.zeros(2,'f4');patches=np.zeros((2,24),'f4')
        tree_module.pack_terrain_patches(patches,0,raw,one,zero,zero,zero,0.,one,zero,zero,one*slot,-10.,20.)
        bodies=np.zeros((1,28),'f4');bodies[0,6]=6371./149597870.7
        mapping=cache.prepare(patches,bodies,{0:'Test'},streamer,4)
        assert mapping[0]>=0 and mapping[1]==-1 # bounded fallback until baked
        mapping=cache.prepare(patches,bodies,{0:'Test'},streamer,4)
        assert np.all(mapping>=0)
        geometry=np.frombuffer(cache.geometry.read(),'<f4').reshape(-1,25,4)
        coarse=geometry[mapping[0],0];fine=geometry[mapping[1],4]
        assert np.max(abs(coarse-fine))<2e-6
        cache.prepare(patches,bodies,{0:'Test'},streamer,4)
        assert cache.last_bakes==0
    finally:
        cache.release();streamer.shutdown()


def test_rendered_fine_edge_matches_coarse_chord(ctx,tmp_path,adapters):
    import moderngl
    _,stream_module=adapters
    make_tiles(tmp_path,size=16,lods=0,value=40123)
    streamer=stream_module.TerrainTileStreamer(ctx,str(tmp_path),256,16)
    objects=[]
    try:
        for face in range(6):
            streamer.request_tile('Test','height',face,0,0,0)
        deadline=time.monotonic()+5
        while streamer.resident_count<6 and time.monotonic()<deadline:
            streamer.begin_frame();streamer.process_uploads(64,100.);time.sleep(.005)
        streamer.prepare_height_table({0:'Test'});streamer.use()
        program=ctx.program(vertex_shader=shader('celestial/terrain.vert'),varyings=['f_world_pos','f_normal'])
        streamer.configure_height_program(program)
        scene=np.zeros(2088,'f4');scene[:16]=np.eye(4,dtype='f4').ravel();scene[16:32]=np.eye(4,dtype='f4').ravel();scene[292]=1000.;scene[293]=1.
        scene_buffer=ctx.buffer(scene.tobytes());scene_buffer.bind_to_uniform_block(1)
        bodies=np.zeros(28,'f4');bodies[6]=6371./149597870.7;bodies[10]=1.
        body_buffer=ctx.buffer(bodies.tobytes());body_buffer.bind_to_storage_buffer(2)
        cache_slots=ctx.buffer(np.array([-1],'<i4').tobytes());cache_slots.bind_to_storage_buffer(5)
        geometry=ctx.buffer(np.zeros(4,'f4').tobytes());geometry.bind_to_storage_buffer(6)
        program['u_km_to_au'].value=1./149597870.7
        program['u_grid_step'].value=.25
        program['u_local_camera_enabled'].value=True
        program['u_is_cloud_pass'].value=False
        program['u_geometry_cache_enabled'].value=False
        high=np.zeros((8,3),'f4');high[0,2]=6371.
        program['u_body_camera_high_km'].write(high.tobytes())
        program['u_body_camera_low_km'].write(np.zeros((8,3),'f4').tobytes())
        objects.extend([program,scene_buffer,body_buffer,cache_slots,geometry])
        def capture(bounds,lod,edges,points):
            patch=np.zeros(24,'f4');patch[:4]=bounds;patch[8:12]=[4,lod,0,0];patch[12]=streamer.get_tile_slot_or_fallback('Test','height',4,0,0,0)[0];patch[16:20]=[-10.,20.,-1.,0.];patch[20]=edges
            instance=ctx.buffer(patch.tobytes());instance.bind_to_storage_buffer(4)
            source=ctx.buffer(np.array(points,'f4').tobytes());output=ctx.buffer(reserve=len(points)*24)
            vao=ctx.vertex_array(program,[(source,'3f','in_position')])
            vao.transform(output,mode=moderngl.POINTS,vertices=len(points))
            result=np.frombuffer(output.read(),'<f4').reshape(-1,6).copy()
            for obj in (instance,source,output,vao):obj.release()
            return result
        coarse=capture([0,-1,1,0],1,0,[[0,0,0],[0,.25,0]])
        fine=capture([-.5,-1,0,-.5],2,2,[[1,.25,0],[1,0,0]])
        assert np.max(abs(fine[0,:3]-coarse[:,:3].mean(axis=0)))<2e-11
        assert np.max(abs(fine[1,3:]-coarse[0,3:]))<2e-5
    finally:
        for obj in objects:obj.release()
        streamer.shutdown()
