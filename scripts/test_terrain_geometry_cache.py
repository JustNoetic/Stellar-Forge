"""Headless regression checks: python scripts/test_terrain_geometry_cache.py.

Requires the normal NumPy/ModernGL/Pillow dependencies and an OpenGL 4.6 context.
Compares cached results against the unchanged live vertex path on the GPU.
"""
from pathlib import Path
import sys
import types
import unittest
import ast
import numpy as np
import moderngl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, path in [('engine', ROOT/'engine'), ('engine.rendering', ROOT/'engine/rendering')]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module

from engine.rendering.shader_loader import load_shader
from engine.rendering.terrain_geometry_cache import TerrainGeometryCache
from engine.rendering.terrain_streamer import TerrainTileStreamer


def grid(resolution):
    # Use the production grid, including the perimeter's duplicated skirt vertices.
    tree = ast.parse((ROOT/'engine/rendering/render_utils.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_terrain_grid_patch')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'terrain_grid', 'exec'), namespace)
    return namespace['create_terrain_grid_patch'](resolution)


class TerrainCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=460)
        print('GPU:', cls.ctx.info['GL_RENDERER'])

    @classmethod
    def tearDownClass(cls):
        cls.ctx.release()

    def setUp(self):
        self.resources = []
        self.cache = TerrainGeometryCache(self.ctx, max_patches=128, max_bakes_per_frame=128)
        au = 149597870.7
        self.shapes = np.zeros((2, 28), dtype='f4')
        self.shapes[:, 6] = [6371/au, 1000/au]
        self.shapes[:, 10] = 1
        self.shapes[:, 12] = [0.00335, 0.3]
        self.shapes[:, 25] = [0.2, -0.7]
        self.bodies = self.keep(self.ctx.buffer(self.shapes.tobytes()))
        self.bodies.bind_to_storage_buffer(2)
        self.scene = np.zeros(1832, dtype='f4')
        self.scene[:16] = np.eye(4, dtype='f4').ravel()
        self.scene[16:32] = np.eye(4, dtype='f4').ravel()
        self.scene[292:294] = [1e5, 1e11]
        self.ubo = self.keep(self.ctx.buffer(self.scene.tobytes()))
        self.ubo.bind_to_uniform_block(1)
        yy, xx = np.mgrid[:512, :512]
        image = np.clip(180 + 30*np.sin(xx/73) + 20*np.cos(yy/81), 0, 255).astype('u1')
        tiles = np.repeat(image[None, :, :, None], 2, axis=0)
        tiles = np.repeat(tiles, 4, axis=3)
        self.tiles = self.keep(self.ctx.texture_array((512, 512, 2), 4, tiles.tobytes()))
        self.tiles.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tiles.repeat_x = self.tiles.repeat_y = False
        self.tiles.use(14)
        self.streamer = types.SimpleNamespace(slot_upload_versions=[1, 1])
        self.names = {0: 'earth', 1: 'oblate'}
        self.fbo = self.keep(self.ctx.simple_framebuffer((32, 32)))
        self.fbo.use()

    def keep(self, resource):
        self.resources.append(resource)
        return resource

    def tearDown(self):
        self.cache.release()
        for resource in reversed(self.resources):
            resource.release()

    def patches(self, levels=(0, 4, 6, 10)):
        rows = []
        for body in range(2):
            for face in range(6):
                for lod in levels:
                    p = np.zeros(20, dtype='f4')
                    scale = 2.0 / 2**lod
                    x = 0 if lod == 0 else 2**(lod-1)
                    lo = -1.0 + x*scale
                    p[:4] = [lo, lo, lo+scale, lo+scale]
                    p[4:8] = [1, 0, 0, 0.5]
                    p[8:12] = [face, lod, 0, body]
                    p[12:16] = [1, 0.5, 0.25, 0.25]
                    p[16:20] = [-10.984 if body == 0 else 0, 19.832 if body == 0 else 8.848, -1, body]
                    rows.append(p)
        return np.asarray(rows)

    def capture(self, patches, resolution, enabled, cloud=False):
        vertices, _ = grid(resolution)
        program = self.keep(self.ctx.program(vertex_shader=load_shader('celestial/terrain.vert'),
                                            varyings=['f_normal', 'f_world_pos']))
        settings = {'u_tile_array': 14, 'u_km_to_au': 1/149597870.7,
                    'u_grid_step': 1/resolution, 'u_geometry_cache_enabled': enabled,
                    'u_cache_grid_resolution': resolution,
                    'u_is_cloud_pass': cloud, 'u_cloud_altitude_km': 3.5}
        for name, value in settings.items():
            if name in program:
                program[name].value = value
        patch_buffer = self.keep(self.ctx.buffer(patches.tobytes()))
        patch_buffer.bind_to_storage_buffer(4)
        self.bodies.bind_to_storage_buffer(2)
        vbo = self.keep(self.ctx.buffer(vertices.tobytes()))
        vao = self.keep(self.ctx.vertex_array(program, [(vbo, '3f', 'in_position')]))
        output = self.keep(self.ctx.buffer(reserve=len(patches)*len(vertices)*24))
        vao.transform(output, mode=moderngl.POINTS, vertices=len(vertices), instances=len(patches))
        return np.frombuffer(output.read(), dtype='f4').reshape(-1, 6)

    def test_matches_live_displacement_and_normals(self):
        patches = self.patches()
        for resolution in (8, 16, 24, 32, 48, 64):
            with self.subTest(resolution=resolution):
                slots = self.cache.prepare(patches, self.shapes, self.names, self.streamer, resolution)
                self.assertTrue(np.all(slots >= 0))
                live = self.capture(patches, resolution, False)
                cached = self.capture(patches, resolution, True)
                # Grid sharing changes floating-point evaluation order slightly.
                # Position stays within centimetres; normals within one degree.
                position_error = np.linalg.norm(live[:, 3:] - cached[:, 3:], axis=1)*149597870.7
                angular_error = np.degrees(np.arccos(np.clip(np.sum(live[:, :3]*cached[:, :3], axis=1), -1, 1)))
                print('resolution', resolution, 'max position km', position_error.max(),
                      'normal degrees p99/max', np.percentile(angular_error, 99), angular_error.max())
                self.assertLess(float(position_error.max()), 0.0001)
                self.assertLess(float(angular_error.max()), 1.0)
        self.assertEqual(self.ctx.error, 'GL_NO_ERROR')

    def test_upload_reuse_and_shape_changes_invalidate(self):
        patches = self.patches(levels=(6,))[:3]
        self.cache.prepare(patches, self.shapes, self.names, self.streamer, 16)
        self.assertEqual(self.cache.last_bakes, 3)
        self.cache.prepare(patches, self.shapes, self.names, self.streamer, 16)
        self.assertEqual(self.cache.last_bakes, 0)
        self.streamer.slot_upload_versions[0] += 1  # diffuse-only upload
        patches[:, 10] = 1
        self.shapes[:, 25] += 0.1  # planet spin does not change local geometry
        self.cache.prepare(patches, self.shapes, self.names, self.streamer, 16)
        self.assertEqual(self.cache.last_bakes, 0)
        self.streamer.slot_upload_versions[1] += 1
        self.cache.prepare(patches, self.shapes, self.names, self.streamer, 16)
        self.assertEqual(self.cache.last_bakes, 3)
        self.shapes[0, 12] += 0.1
        self.cache.prepare(patches, self.shapes, self.names, self.streamer, 16)
        self.assertEqual(self.cache.last_bakes, 3)
        patches[:, 13] = 0.25  # new resident ancestor mapping
        self.cache.prepare(patches, self.shapes, self.names, self.streamer, 16)
        self.assertEqual(self.cache.last_bakes, 3)

    def test_budget_fallback_eviction_and_missing_height(self):
        self.cache.release()
        self.cache = TerrainGeometryCache(self.ctx, max_patches=128,
                                         max_bytes=2*9*9*16, max_bakes_per_frame=1)
        patches = self.patches(levels=(6,))[:3]
        slots = self.cache.prepare(patches, self.shapes, self.names, self.streamer, 8)
        np.testing.assert_array_equal(slots >= 0, [True, False, False])
        live = self.capture(patches, 8, False)
        cached = self.capture(patches, 8, True)
        self.assertLess(float(np.max(np.abs(live - cached))), 0.01)
        slots = self.cache.prepare(patches, self.shapes, self.names, self.streamer, 8)
        np.testing.assert_array_equal(slots >= 0, [True, True, False])
        first_slot = slots[0]
        slots = self.cache.prepare(patches[[0, 2]], self.shapes, self.names, self.streamer, 8)
        self.assertEqual(slots[0], first_slot)
        self.assertTrue(np.all(slots >= 0))
        patches[:, 12] = -1
        slots = self.cache.prepare(patches, self.shapes, self.names, self.streamer, 8)
        self.assertTrue(np.all(slots == -1))
        self.assertEqual(self.cache.last_bakes, 0)

    def test_production_draw_program_compiles(self):
        self.keep(self.ctx.program(vertex_shader=load_shader('celestial/terrain.vert'),
                                   fragment_shader=load_shader('celestial/terrain.frag')))

    def test_clouds_oceans_and_zero_height_span(self):
        patches = self.patches(levels=(6,))[:2]
        patches[0, 16:18] = [-20, 1]  # entirely underwater
        patches[1, 16:18] = [0, 0]
        self.cache.prepare(patches, self.shapes, self.names, self.streamer, 16)
        for cloud in (False, True):
            live = self.capture(patches, 16, False, cloud)
            cached = self.capture(patches, 16, True, cloud)
            np.testing.assert_allclose(cached, live, rtol=1e-5, atol=1e-5)
        self.assertEqual(self.ctx.error, 'GL_NO_ERROR')

    def test_upload_version_changes_only_after_write(self):
        streamer = TerrainTileStreamer.__new__(TerrainTileStreamer)
        streamer.texture_array = self.tiles
        streamer.tile_size = 512
        streamer.slot_upload_versions = [0, 0]
        streamer.height_tiles = {}
        streamer.upload_tile_to_slot(1, bytes(512*512*4), ('earth', 'height', 0, 0, 0, 0))
        self.assertEqual(streamer.slot_upload_versions[1], 1)
        streamer.upload_tile_to_slot(1, bytes(512*512*4), ('earth', 'height', 0, 0, 0, 0))
        self.assertEqual(streamer.slot_upload_versions[1], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
