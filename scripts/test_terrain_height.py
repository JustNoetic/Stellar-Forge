"""Height terrain regression: GPU displacement/triangle normals and tile lifecycle."""
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
                       varyings=['f_rel_pos', 'f_normal'])
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
    out = ctx.buffer(reserve=2 * 6 * 4)
    fbo = ctx.simple_framebuffer((4, 4)); fbo.use()

    def capture(slot, cloud=False):
        patch[0, 12] = slot; pb.write(patch.tobytes())
        prog['u_is_cloud_pass'] = cloud
        vao.transform(out, mode=moderngl.POINTS, vertices=2)
        return np.frombuffer(out.read(), 'f4').reshape(2, 6).copy()

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

    # Execute the shipped fragment normal block on two different triangle planes.
    # Their physical slopes must win over the smooth reference normal, including
    # perspective interpolation, reversed winding, and tiny AU-scale derivatives.
    normal_code = terrain_fragment_shader.split('    vec3 N = normalize(f_normal);', 1)[1].split('    vec3 V =', 1)[0]
    frag = '''#version 460
uniform bool u_is_cloud_pass;
uniform float f_height_slot;
uniform vec3 f_normal;
in vec3 f_patch_pos_km;
out vec4 color;
void main() { vec3 N = normalize(f_normal);
''' + normal_code + '\ncolor = vec4(N, 1.0); }'
    vsrc = '''#version 460
in vec3 in_surface;
uniform float position_scale;
uniform bool reverse_surface;
out vec3 f_patch_pos_km;
void main() {
    f_patch_pos_km = in_surface * position_scale;
    vec2 screen = in_surface.xy;
    if (reverse_surface) screen.x = -screen.x;
    float w = 1.0 + in_surface.z * 0.2;
    gl_Position = vec4(screen * w, 0.0, w);
}'''
    fp = ctx.program(vertex_shader=vsrc, fragment_shader=frag)
    surface = np.array([[-1, -1, -0.65], [1, -1, 0.15], [-1, 1, -0.15],
                        [1, -1, 0.15], [1, 1, -0.15], [-1, 1, -0.15]], 'f4')
    surface_buffer = ctx.buffer(surface.tobytes())
    fv = ctx.vertex_array(fp, [(surface_buffer, '3f', 'in_surface')])
    color = ctx.texture((32, 32), 4, dtype='f4'); fb = ctx.framebuffer([color]); fb.use()
    fp['f_normal'] = (0, 0, 1); fp['f_height_slot'] = 0

    def normals():
        fb.clear()
        fv.render(vertices=6)
        return np.frombuffer(color.read(), 'f4').reshape(32, 32, 4)[..., :3]

    expected = []
    for tri in surface.reshape(2, 3, 3):
        face = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        expected.append(face / np.linalg.norm(face))
    yy, xx = np.mgrid[:32, :32]
    # Exclude shared-edge helper lanes from the flatness comparison.
    for scale in (1.0, 1.0 / 149597870.7, 1e-12):
        fp['position_scale'] = scale
        for reverse in (False, True):
            fp['reverse_surface'] = reverse
            actual = normals()
            screen_x = 31 - xx if reverse else xx
            for mask, face in ((screen_x + yy < 25, expected[0]),
                               (screen_x + yy > 38, expected[1])):
                np.testing.assert_allclose(actual[mask], np.broadcast_to(face, actual[mask].shape), atol=2e-5)
    fp['u_is_cloud_pass'] = True
    np.testing.assert_array_equal(normals(), np.broadcast_to([0, 0, 1], (32, 32, 3)))
    fp['u_is_cloud_pass'] = False; fp['f_height_slot'] = -1
    np.testing.assert_array_equal(normals(), np.broadcast_to([0, 0, 1], (32, 32, 3)))
    # Degenerate physical positions must fall back to a finite reference normal.
    fp['f_height_slot'] = 0; fp['position_scale'] = 0
    np.testing.assert_array_equal(normals(), np.broadcast_to([0, 0, 1], (32, 32, 3)))
    print('Triangle normals, perspective, winding, AU scale, clouds and fallback passed')

    # Reproduce the close-up case with the PRODUCTION vertex calculation:
    # a meter-scale triangle on an Earth-sized spheroid filling the screen.
    # The earlier test had no planet-radius offset and missed interpolation noise.
    ground_vertex = terrain_vertex_shader.replace(
        'gl_Position = projection * view * vec4(p_world, 1.0);',
        'float w = 1.0 + in_position.x * 0.5;\n'
        '    gl_Position = vec4((in_position.xy * 4.0 - 1.0) * w, 0.0, w);'
    ).replace('gl_Position.z *= gl_Position.w;', 'gl_Position.z = 0.0;')
    ground_frag = frag.replace('uniform vec3 f_normal;', 'in vec3 f_normal;').replace(
        'uniform float f_height_slot;', 'flat in float f_height_slot;')
    ground = ctx.program(vertex_shader=ground_vertex, fragment_shader=ground_frag,
                         varyings=['f_patch_pos_km', 'f_normal'])
    patch[0, :4] = [0.15, 0.25, 0.1500002, 0.2500002]
    patch[0, 12] = 0
    pb.write(patch.tobytes())
    body[:3] = [10, 20, 30]
    body[6] = 6371 / 149597870.7
    bb.write(body.tobytes())
    ground['u_km_to_au'] = 1 / 149597870.7
    ground['u_tile_array'] = 14
    tex.use(14)
    gv = ctx.buffer(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 'f4').tobytes())
    ga = ctx.vertex_array(ground, [(gv, '3f', 'in_position')])
    captured = ctx.buffer(reserve=3 * 6 * 4)
    ground_color = ctx.texture((128, 128), 4, dtype='f4')
    ground_fb = ctx.framebuffer([ground_color]); ground_fb.use()
    for face in range(6):
        patch[0, 8] = face; pb.write(patch.tobytes())
        ga.transform(captured, mode=moderngl.POINTS, vertices=3)
        positions = np.frombuffer(captured.read(), 'f4').reshape(3, 6).astype('f8')
        expected_face = np.cross(positions[1, :3] - positions[0, :3],
                                 positions[2, :3] - positions[0, :3])
        expected_face /= np.linalg.norm(expected_face)
        if np.dot(expected_face, positions[0, 3:]) < 0:
            expected_face *= -1
        ground_fb.clear(); ga.render(vertices=3)
        actual = np.frombuffer(ground_color.read(), 'f4').reshape(128, 128, 4)
        np.testing.assert_allclose(actual[..., :3],
                                   np.broadcast_to(expected_face, (128, 128, 3)), atol=2e-3)
    print('Meter-scale close-up on six Earth-sized oblate faces stays uniform')


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
            cloud = np.zeros((1, 20), 'f4'); pack_cloud_patches_jit(cloud, 0, raw_patches, scales[0], ox[0], oy[0], slots[0], 3.)
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
        module.pack_cloud_patches_jit(cloud, 0, raw, values, values, values, values, 2.)
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
