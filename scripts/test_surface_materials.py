"""Run with the project Python: python scripts/test_surface_materials.py.

Tests physical response boundaries, CPU/GPU agreement, phase integration,
buffer reordering and production shader linking. Requires OpenGL 4.6.
"""
from pathlib import Path
import math
import sys
import types
import unittest
import ast
import json
import copy
import tempfile
import time
import os

import numpy as np
import moderngl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for name, path in [('engine',ROOT/'engine'),('engine.rendering',ROOT/'engine/rendering')]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module

from engine.rendering.surface_materials import (
    material_key, resolve_material, packed_material, phase_table, surface_response,
    integrate_disk, disk_response, material_albedos, SurfaceMaterialCache,
)
from engine.rendering.shader_loader import load_shader
from engine.rendering.planetshine import compute_planetshine_numba


class SurfaceMaterialsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=460)
        print('GPU:',cls.ctx.info['GL_RENDERER'])

    @classmethod
    def tearDownClass(cls):
        cls.ctx.release()

    def test_defaults_and_bad_data(self):
        self.assertEqual(resolve_material({'name':'Moon'})['model'],'hapke')
        self.assertEqual(resolve_material({'name':'Europa'})['preset'],'icy')
        for body in [{'name':'Saturn','type':'Gas Giant'}, {'name':'Earth'},
                     {'name':'Moon','material':'lambert'}, {'material':{'preset':'unknown'}}]:
            self.assertEqual(resolve_material(body)['model'],'lambert')
        self.assertEqual(resolve_material({'name':'Moon','type':'Star'})['model'],'lambert')
        body = {'name':'Moon','material':{'single_scattering_albedo':float('nan'),
                'roughness_deg':float('inf'),'brightness':-1,'coherent_width':0}}
        row = packed_material(material_key(body))
        self.assertTrue(np.isfinite(row).all())
        self.assertEqual(row[7],0)
        self.assertGreater(row[9],0)

    def test_boundaries_reference_and_opposition(self):
        for preset in ['lunar','icy']:
            row = packed_material(material_key({'material':preset}))
            self.assertEqual(surface_response(-0.1,0.5,0.3,row),0)
            self.assertEqual(surface_response(0.5,0,0.3,row),0)
            ref = math.cos(math.pi/6)
            self.assertAlmostEqual(surface_response(ref,1,ref,row),ref,places=6)
            self.assertGreater(surface_response(1,1,1,row),surface_response(ref,1,ref,row))
            dark = dict(resolve_material({'material':preset}),brightness=0)
            self.assertEqual(surface_response(1,1,1,packed_material(material_key({'material':dark}))),0)
        backward = material_key({'material':dict(resolve_material({'material':'lunar'}),backscatter_fraction=1)})
        forward = material_key({'material':dict(resolve_material({'material':'lunar'}),backscatter_fraction=0)})
        self.assertGreater(phase_table(backward)[0]/phase_table(backward)[360],phase_table(forward)[0]/phase_table(forward)[360])

    def test_disk_integral_and_albedo(self):
        angles = np.radians([0,30,90,150,179])
        nodes, weights = np.polynomial.legendre.leggauss(64)
        nodes, weights = (nodes+1)*0.5, weights*0.5
        for preset in ['lambert','lunar','icy']:
            body = {'material':preset}
            key = material_key(body)
            table = phase_table(key)
            reference = integrate_disk(packed_material(key),angles,nodes,weights,128)
            np.testing.assert_allclose(table[[0,120,360,600,716]],reference,rtol=0.035,atol=2e-5)
            self.assertTrue(np.isfinite(table).all())
            self.assertTrue((table >= 0).all())
            self.assertEqual(table[-1],0)
            g,b,q,rgb = material_albedos(body,np.array([0.1,0.1,0.1]))
            self.assertAlmostEqual(g*q,b,places=6)
            self.assertAlmostEqual(rgb.mean(),g,places=6)
        lambert = phase_table(material_key({'material':'lambert'}))
        self.assertAlmostEqual(lambert[0],2/3,places=6)
        self.assertAlmostEqual(lambert[360],2/(3*math.pi),places=6)
        lunar = phase_table(material_key({'material':'lunar'}))
        self.assertLess(disk_response(lunar,1,0.01),lunar[0])

    def test_cpu_gpu_response_and_phase(self):
        source = load_shader('common/surface_material.glsl')
        phase = load_shader('common/surface_phase.glsl')
        shader = self.ctx.compute_shader('''#version 460
        #define PI 3.14159265358979323846
        layout(local_size_x=64) in;
        layout(std430,binding=0) readonly buffer Inputs { vec4 cases[]; };
        layout(std430,binding=1) writeonly buffer Outputs { vec4 answers[]; };
        uniform int count;
        '''+source+phase+'''
        void main() {
            uint i=gl_GlobalInvocationID.x;
            if (i>=uint(count)) return;
            vec4 v=cases[i]; SurfaceMaterial m=u_surface_materials[0];
            answers[i]=vec4(hapke_raw(v.x,v.y,v.z,m)*m.opposition.z*m.surface.w,
                surface_disk_response(v.z,v.w,m),0,0);
        }''')
        rng = np.random.default_rng(941)
        count = 1024
        normals = rng.normal(size=(count,3)); normals /= np.linalg.norm(normals,axis=1)[:,None]
        lights = rng.normal(size=(count,3)); lights /= np.linalg.norm(lights,axis=1)[:,None]
        # Viewer=(0,0,1); physically consistent incidence, emission, phase.
        cases = np.zeros((count,4),dtype='f4')
        cases[:,0] = np.sum(normals*lights,axis=1)
        cases[:,1] = normals[:,2]
        cases[:,2] = lights[:,2]
        cases[:,3] = rng.uniform(0,0.02,count)
        cases[:4] = [[1,1,1,0],[0.5,1,0.5,0.0047],[1e-5,0.5,0.866,0],[0,0,-1,0.01]]
        inputs = self.ctx.buffer(cases.tobytes()); output = self.ctx.buffer(reserve=cases.nbytes)
        cache = SurfaceMaterialCache(self.ctx)
        try:
            inputs.bind_to_storage_buffer(0); output.bind_to_storage_buffer(1)
            shader['count'].value=count
            for preset in ['lunar','icy']:
                rows,tables = cache.update([{'material':preset}])
                shader.run((count+63)//64); self.ctx.memory_barrier()
                gpu = np.frombuffer(output.read(),dtype='f4').reshape(-1,4)
                cpu = np.array([[surface_response(*c[:3],rows[0]),disk_response(tables[0],float(c[2]),float(c[3]))] for c in cases])
                np.testing.assert_allclose(gpu[:,:2],cpu,rtol=0.002,atol=1e-4)
        finally:
            cache.release(); inputs.release(); output.release(); shader.release()

    def test_buffer_order_and_production_shader_linking(self):
        cache = SurfaceMaterialCache(self.ctx)
        try:
            rows,_ = cache.update([{'name':'Moon'},{'name':'Europa'},{'name':'Saturn'}])
            self.assertEqual(list(rows[:,0]),[1,1,0])
            rows,_ = cache.update([{'name':'Saturn'},{'name':'Europa'},{'name':'Moon'}])
            self.assertEqual(list(rows[:,0]),[0,1,1])
            gpu = np.frombuffer(cache.material_buffer.read(),dtype='f4').reshape(-1,12)
            np.testing.assert_array_equal(rows,gpu)
            cache.update([])
            for name in ['sphere','terrain','point_celestial']:
                program = self.ctx.program(vertex_shader=load_shader('celestial/'+name+'.vert'),fragment_shader=load_shader('celestial/'+name+'.frag'))
                program.release()
        finally:
            cache.release()

    def test_planetshine_legacy_and_material_phase(self):
        # Receiver at x=0.001 AU sees an illuminated small caster at origin.
        pos = np.array([[0.001,0,0],[0,0,0]],dtype='f4')
        radii = np.array([0,1e-5],dtype='f4') # point receiver does not eclipse caster
        colors = np.full((2,3),0.1,dtype='f4')
        stars = np.array([[1,0,0]],dtype='f4')
        args = (pos,radii,colors,np.zeros(2,dtype='f4'),stars,np.ones((1,3),dtype='f4'),
                np.ones(1,dtype='f4'),np.zeros(1,dtype='f4'),True)
        legacy = compute_planetshine_numba(*args)[1]
        keys = [material_key({'material':'lambert'}),material_key({'material':'lambert'})]
        rows = np.array([packed_material(k) for k in keys],dtype='f4')
        tables = np.array([phase_table(k) for k in keys],dtype='f4')
        equivalent = compute_planetshine_numba(*args,surface_materials=rows,surface_phases=tables)[1]
        np.testing.assert_allclose(legacy,equivalent,rtol=1e-6)
        key = material_key({'material':'lunar'})
        rows[1] = packed_material(key); tables[1] = phase_table(key)
        lunar = compute_planetshine_numba(*args,surface_materials=rows,surface_phases=tables)[1]
        self.assertGreater(lunar[0,0],legacy[0,0])
        self.assertAlmostEqual(float(lunar[0,0]/legacy[0,0]),float(tables[1,0]/(2/3)),places=4)
        off = compute_planetshine_numba(*args,planetshine_enabled=False,ringshine_enabled=False,surface_materials=rows,surface_phases=tables)[1]
        self.assertTrue((off==0).all())

    def test_cosmetics_persistence(self):
        # Execute the production save function with isolated system paths.
        tree=ast.parse((ROOT/'engine/ui/inspector.py').read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_save_system_cosmetics')
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory); path=directory/'system.json'
            body={'name':'Moon','r':0.0025,'material':dict(resolve_material({'material':'lunar'}),preset='custom',brightness=0.7)}
            path.write_text(json.dumps([{'name':'Moon','r':0.0025}]))
            app=types.SimpleNamespace(sys_mgr=types.SimpleNamespace(_system_json_path=lambda name:str(path),save_meta=lambda *args:None))
            namespace={'os':os,'json':json,'copy':copy,'time':time,'NumpyEncoder':json.JSONEncoder,
                       'get_external_path':lambda *parts:str(directory.joinpath(*parts)),
                       'SystemManager':types.SimpleNamespace(SOLAR_SYSTEM_NAME='Solar System')}
            exec(compile(ast.Module(body=[function],type_ignores=[]),'save_cosmetics','exec'),namespace)
            save=namespace['_save_system_cosmetics']
            save(app,[body],np.full((1,3),0.1),[],[],'Custom System')
            loaded=json.loads(path.read_text())[0]
            self.assertEqual(loaded['material'],body['material'])
            self.assertEqual(material_key(loaded),material_key(body))
            del body['material']
            save(app,[body],np.full((1,3),0.1),[],[],'Custom System')
            self.assertNotIn('material',json.loads(path.read_text())[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
