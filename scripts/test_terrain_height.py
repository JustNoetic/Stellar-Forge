"""Height terrain regression: actual GPU displacement/fragment slopes and tile lifecycle."""
import os
import sys
import tempfile
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import moderngl
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.rendering.shaders import terrain_vertex_shader, terrain_fragment_shader
from engine.rendering.terrain_quadtree import pack_terrain_patches_jit, pack_cloud_patches_jit
from engine.rendering.terrain_streamer import TerrainTileStreamer
from engine.rendering import texture_manager as tm


def test_gpu(ctx):
    # Capture the production vertex shader's relative position in km.
    prog = ctx.program(vertex_shader=terrain_vertex_shader,
                       varyings=['f_rel_pos', 'f_normal', 'f_tan_u', 'f_tan_v'])
    patch = np.zeros((1, 20), 'f4')
    patch[0, :4] = [-0.2, -0.2, 0.2, 0.2]
    patch[0, 4] = 1
    patch[0, 8] = 4
    patch[0, 12:18] = [-1, 1, 0, 0, -2, 10]
    pb = ctx.buffer(patch.tobytes()); pb.bind_to_storage_buffer(4)
    body = np.zeros(28, 'f4'); body[6] = 100; body[10] = 1
    bb = ctx.buffer(body.tobytes()); bb.bind_to_storage_buffer(2)
    scene = np.zeros(1832, 'f4'); scene[292:294] = [1000, 0.1]
    ub = ctx.buffer(scene.tobytes()); ub.bind_to_uniform_block(1)
    tex = ctx.texture_array((32, 32, 1), 4, bytes([128, 128, 128, 255]) * 1024)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = False
    tex.use(14)
    prog['u_tile_array'] = 14; prog['u_km_to_au'] = 1.0
    vertex = np.array([[0.5, 0.5, 0], [0.5, 0.5, 1]], 'f4')
    vb = ctx.buffer(vertex.tobytes())
    vao = ctx.vertex_array(prog, [(vb, '3f', 'in_position')])
    out = ctx.buffer(reserve=2 * 12 * 4)
    fbo = ctx.simple_framebuffer((4, 4)); fbo.use()

    def capture(slot, cloud=False):
        patch[0, 12] = slot; pb.write(patch.tobytes())
        prog['u_is_cloud_pass'] = cloud
        vao.transform(out, mode=moderngl.POINTS, vertices=2)
        return np.frombuffer(out.read(), 'f4').reshape(2, 12).copy()

    elevation = -2 + (128 / 255) * 10
    for face in range(6):
        patch[0, 8] = face
        base = capture(-1)
        raised = capture(0)
        np.testing.assert_allclose(raised[0, :3], base[0, :3] + base[0, 3:6] * elevation, atol=2e-5)
        # Skirts gain the full height span; flat surfaces retain the base normal.
        np.testing.assert_allclose(raised[1, :3], raised[0, :3] - base[0, 3:6] * 10, atol=2e-5)
        np.testing.assert_array_equal(capture(-1, True), capture(0, True))
    print('GPU displacement on six faces, skirt span, and cloud isolation passed')

    # Oblate geometry must displace radially, not along its geodetic normal.
    body[12] = 0.3; body[25] = 0.7; bb.write(body.tobytes())
    vertex[:, :2] = [0.7, 0.8]; vb.write(vertex.tobytes())
    base = capture(-1); raised = capture(0)
    radial = base[0, :3] / np.linalg.norm(base[0, :3])
    np.testing.assert_allclose(raised[0, :3], base[0, :3] + radial * elevation, atol=2e-5)

    # Test actual fragment normal code, with controlled derivatives and a ramp.
    normal_code = terrain_fragment_shader.split('    vec3 N = normalize(f_normal);', 1)[1].split('    vec3 V =', 1)[0]
    frag = '''#version 460
uniform sampler2DArray u_tile_array;
uniform bool u_is_cloud_pass;
uniform vec3 f_height_meta;
uniform vec3 f_tan_u;
uniform vec3 f_tan_v;
uniform vec3 f_normal;
uniform vec2 f_height_uv;
out vec4 color;
void main() { vec3 N = normalize(f_normal);
''' + normal_code + '\ncolor = vec4(N, 1.0); }'
    vsrc = '''#version 460
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }'''
    fp = ctx.program(vertex_shader=vsrc, fragment_shader=frag)
    fv = ctx.vertex_array(fp, [])
    color = ctx.texture((1, 1), 4, dtype='f4'); fb = ctx.framebuffer([color]); fb.use()
    ramp = np.zeros((32, 32, 4), 'u1'); ramp[..., :3] = np.arange(32)[None, :, None] * 8; ramp[..., 3] = 255
    tex.write(ramp.tobytes()); tex.use(14)
    fp['u_tile_array'] = 14
    fp['f_normal'] = (0, 0, 1); fp['f_tan_u'] = (100, 0, 0); fp['f_tan_v'] = (0, 100, 0)
    fp['f_height_uv'] = (0.5, 0.5); fp['f_height_meta'] = (0, 1, 10)

    def normal():
        fv.render(vertices=3)
        return np.frombuffer(color.read(), 'f4')[:3]

    expected = np.array([-10 * 8 * 32 / 255 / 100, 0, 1])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(normal(), expected, atol=2e-6)
    # A child using half an ancestor's UV extent also has half its physical extent.
    fp['f_height_meta'] = (0, 0.5, 10); fp['f_tan_u'] = (50, 0, 0); fp['f_tan_v'] = (0, 50, 0)
    np.testing.assert_allclose(normal(), expected, atol=2e-6)
    fp['f_height_uv'] = (1, 1)
    np.testing.assert_allclose(normal(), expected, atol=2e-6)
    fp['u_is_cloud_pass'] = True
    np.testing.assert_array_equal(normal(), [0, 0, 1])
    fp['u_is_cloud_pass'] = False; fp['f_height_meta'] = (-1, 1, 10)
    np.testing.assert_array_equal(normal(), [0, 0, 1])
    print('Fragment height relief, ancestor scale, edge slopes, and disabled normals passed')


def test_streaming(ctx):
    with tempfile.TemporaryDirectory() as root:
        def save(layer, value):
            path = Path(root) / 'earth' / layer / '0/0/0_0.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('L', (8, 8), value).save(path)
            return path
        save('diffuse', 50); height_path = save('height', 128)
        streamer = TerrainTileStreamer(ctx, root, pool_capacity=16, tile_size=8)
        try:
            a = np.array([0], 'f4'); lod = np.array([2], 'f4'); xy = np.array([1], 'f4')
            slots, scales, ox, oy = streamer.resolve_tiles_batch_multi('Earth', ['diffuse', 'height'], a, lod, xy, xy)
            assert slots[0][0] != slots[1][0]
            assert scales[1][0] == 0.25 and ox[1][0] == oy[1][0] == 0.25
            raw = np.frombuffer(streamer._read_tile_from_disk('earth', 'height', 0, 0, 0, 0), 'u1').reshape(-1, 4)
            np.testing.assert_array_equal(raw[0], [128, 128, 128, 255])
            # Full-detail diffuse may coexist with ancestor height.
            streamer.resident_tiles[('earth', 'diffuse', 0, 2, 1, 1)] = int(slots[0][0])
            streamer._resolve_memo.clear()
            slots, scales, ox, oy = streamer.resolve_tiles_batch_multi('Earth', ['diffuse', 'height'], a, lod, xy, xy)
            assert scales[0][0] == 1 and scales[1][0] == 0.25
            del streamer.resident_tiles[('earth', 'diffuse', 0, 2, 1, 1)]
            raw_patches = np.zeros((1, 9), 'f4'); staging = np.zeros((1, 20), 'f4')
            for pack in [pack_terrain_patches_jit, getattr(pack_terrain_patches_jit, 'py_func', pack_terrain_patches_jit)]:
                pack(staging, 0, raw_patches, scales[0], ox[0], oy[0], slots[0], 3., scales[1], ox[1], oy[1], slots[1], -2., 10.)
                np.testing.assert_array_equal(staging[0, 13:18], [0.25, 0.25, 0.25, -2, 10])
            cloud = staging.copy(); pack_cloud_patches_jit(cloud, staging, 0, 1, scales[0], ox[0], oy[0], slots[0])
            assert cloud[0, 12] == -1 and staging[0, 12] >= 0
            save('height', 200); streamer.reload_layer('Earth', 'height'); streamer.process_uploads()
            slot, *_ = streamer.get_tile_slot_or_fallback('Earth', 'height', 0, 0, 0, 0)
            pool = np.frombuffer(streamer.texture_array.read(), 'u1').reshape(16, 8, 8, 4)
            np.testing.assert_array_equal(pool[slot, 0, 0], [200, 200, 200, 255])
            locked = len(streamer.locked_slots)
            streamer.reload_layer('Earth', 'height'); streamer.process_uploads()
            assert len(streamer.locked_slots) == locked
            height_path.unlink(); streamer.reload_layer('Earth', 'height'); streamer.process_uploads()
            assert streamer.get_tile_slot_or_fallback('Earth', 'height', 0, 0, 0, 0)[0] == -1
        finally:
            streamer.shutdown()
    print('Height channel fidelity, independent fallback, packing, reload and deletion passed')


def test_ui_safety(ctx):
    with tempfile.TemporaryDirectory() as root:
        old_path = tm.get_external_path
        tm.get_external_path = lambda *parts: str(Path(root).joinpath(*parts))
        try:
            folder = Path(root) / 'textures/System/Earth'; folder.mkdir(parents=True)
            diffuse = folder / 'Earth.png'; height = folder / 'Earth_heightmap.png'
            Image.new('L', (8, 4), 12).save(diffuse); Image.new('L', (8, 4), 123).save(height)
            tm._cleanup_layer_disk_files('Earth', 'height', 'System', preserve_file=str(height))
            assert diffuse.exists() and height.exists()
            app = SimpleNamespace(body_textures_ssbo_data=np.zeros((1, 4, 2), 'u4'))
            assert tm.hot_reload_body_texture(app, ctx, 'Earth', str(height), 'height')
            assert not app.body_textures_ssbo_data.any()
            app._height_previews['earth'].release()
            tm._cleanup_layer_disk_files('Earth', 'height', 'System')
            assert diffuse.exists() and not height.exists()
        finally:
            tm.get_external_path = old_path
    print('Height import avoids bindless SSBO and cleanup preserves diffuse source')


def test_cpu_fallback():
    from engine.rendering import terrain_quadtree as jit
    # Import the actual fallback definitions, rather than only testing njit.py_func.
    saved = sys.modules.get('numba')
    try:
        sys.modules['numba'] = None
        spec = importlib.util.spec_from_file_location('terrain_no_numba', Path(jit.__file__))
        fallback = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fallback)
    finally:
        if saved is None:
            sys.modules.pop('numba', None)
        else:
            sys.modules['numba'] = saved
    assert not fallback._HAS_NUMBA
    results = []
    for module in (jit, fallback):
        tree = module.PlanetQuadtree(6371, max_lod=4)
        camera = np.array([0, 0, 6500], 'f4')
        base = tree.traverse_raw(camera, 45, 1080, height_max_km=0)
        raised = tree.traverse_raw(camera, 45, 1080, height_max_km=100)
        assert len(raised) > len(base)
        results.append(sorted(tuple(row[4:8]) for row in raised))
        raw = np.zeros((2, 9), 'f4'); values = np.ones(2, 'f4')
        staging = np.zeros((2, 20), 'f4')
        module.pack_terrain_patches_jit(staging, 0, raw, values, values, values, values, 2.,
                                        values, values, values, values, -2., 10.)
        assert np.all(staging[:, 16:18] == [-2, 10])
        cloud = np.zeros_like(staging)
        module.pack_cloud_patches_jit(cloud, staging, 0, 2, values, values, values, values)
        assert np.all(cloud[:, 12] == -1)
    assert results[0] == results[1]
    print('JIT/no-Numba traversal agree; elevated bounds and both packing paths passed')


if __name__ == '__main__':
    ctx = moderngl.create_context(standalone=True)
    test_gpu(ctx)
    test_streaming(ctx)
    test_ui_safety(ctx)
    test_cpu_fallback()
    ctx.release()
    print('All terrain height tests passed')
