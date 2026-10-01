"""Camera landing, streamed terrain geometry and near-plane regressions.

Run with python scripts/test_camera_ground.py; add --gpu to compare the CPU
collision triangles with the production terrain vertex shader on OpenGL 4.6.
"""
from pathlib import Path
import ast
import math
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.rendering.terrain_collision import (CameraSurface, terrain_frame, terrain_face_uv,
    terrain_vertices, GROUND_CLEARANCE_KM, camera_near_plane)
from engine.rendering.terrain_streamer import TerrainTileStreamer


class TexturePool:
    def __init__(self):
        self.data = {}
    def write(self, data, viewport):
        self.data[viewport[2]] = data
    def release(self):
        pass


class CPUContext:
    def texture_array(self, *args, **kwargs):
        return TexturePool()


def test_geometry(streamer):
    pole = np.array([.2, .93, .31]); pole /= np.linalg.norm(pole)
    angle = .7
    frame = terrain_frame(pole, angle)
    for oblateness in (0., .00335, .3):
        surface = CameraSurface(6378.1363, oblateness, pole, angle, streamer, 'earth', (-11., 9.), 6, 32)
        for face in range(6):
            for uv in ([.1, .2], [.4, .65], [.91, .73]):
                local = terrain_vertices(face, np.array([uv]), surface.radius, oblateness, np.array([0.]))[0]
                recovered_face, recovered_uv = terrain_face_uv(local / np.linalg.norm(local), oblateness)
                assert recovered_face == face
                np.testing.assert_allclose(recovered_uv, uv, atol=1e-13)
                world = frame @ local
                landed, contact = surface.clamp(world - world / np.linalg.norm(world) * 20.)
                assert contact
                np.testing.assert_allclose(surface.clearance(landed), GROUND_CLEARANCE_KM, atol=1e-10)
                # Both elevated and below-sea-level tiles are physical terrain.
                tile = streamer.height_tiles[streamer.resident_tiles[('earth', 'height', face, 0, 0, 0)]]
                assert (tile[0, 0] > 200) == (np.linalg.norm(landed) > np.linalg.norm(world))
    print('One-metre landing on all six faces, mountains, depressions, tilt and oblateness passed')


def test_motion(streamer):
    surface = CameraSurface(6378.1363, 0., (0, 1, 0), streamer=streamer,
                            body_name='earth', elevation_range=(-11., 9.))
    direction = surface.frame[:, 0]
    start = direction * 6500.
    # Endpoints on opposite sides are both outside: a final-position clamp alone
    # would let this frame pass through the whole planet.
    end = -direction * 6500.
    corrected, contact = surface.constrain_motion(start, end)
    assert contact and corrected @ direction > 0
    assert surface.clearance(corrected) >= .001 - 1e-9
    outside = direction * 6501.
    untouched, contact = surface.constrain_motion(start, outside)
    assert not contact
    np.testing.assert_array_equal(untouched, outside)
    # A camera inside the body (saved pose / teleport) recovers safely.
    recovered, contact = surface.constrain_motion(np.zeros(3), np.zeros(3))
    assert contact and surface.clearance(recovered) >= .001 - 1e-9
    # Smooth bodies retain collision even with terrain LOD disabled.
    smooth = CameraSurface(100., .3, (0, 1, 0))
    landed, contact = smooth.clamp(np.array([0., 60., 0.]))
    assert contact
    np.testing.assert_allclose(smooth.clearance(landed), .001, atol=1e-12)
    print('Continuous descent, high-speed whole-body crossing and smooth-body fallback passed')


def test_streaming(streamer, root):
    slot = streamer.resident_tiles[('earth', 'height', 0, 0, 0, 0)]
    # Probe bilinear/clamped sampling independently of the collision mapping.
    checker = np.tile(np.arange(8, dtype='u1') * 30, (8, 1))
    rgba = np.repeat(checker[..., None], 4, axis=2); rgba[..., 3] = 255
    streamer.upload_tile_to_slot(slot, rgba.tobytes(), ('earth', 'height', 0, 0, 0, 0))
    uv = np.array([[0., .5], [1., .5], [.25, .5], [.5, .5]])
    np.testing.assert_allclose(streamer.sample_height(slot, uv), np.array([0., 210., 45., 105.]) / 255)
    resolution = streamer.get_tile_slot_or_fallback('earth', 'height', 0, 6, 32, 32)
    assert resolution == (slot, 1/64, .5, .5), resolution
    surface = CameraSurface(6378., 0., (0, 1, 0), streamer=streamer, body_name='earth')
    ground = surface.frame @ np.array([6370., 0., 0.])
    old = surface.clamp(ground)[0]
    # A streamed replacement changes collision on the same frame as GPU data.
    replacement = bytes([255, 255, 255, 255]) * 64
    streamer.upload_tile_to_slot(slot, replacement, ('earth', 'height', 0, 0, 0, 0))
    new = surface.clamp(old)[0]
    assert np.linalg.norm(new) > np.linalg.norm(old) + 1.
    # A reused diffuse slot must not retain its former elevation data.
    streamer.upload_tile_to_slot(slot, replacement, ('earth', 'diffuse', 0, 0, 0, 0))
    assert streamer.sample_height(slot, uv) is None
    # Reload lifecycle discards CPU heights alongside the GPU residency map.
    path = Path(root) / 'earth/height/0/0/0_0.png'
    Image.new('L', (8, 8), 42).save(path)
    streamer.reload_queue.put(('earth', 'height'))
    streamer.process_uploads(0)
    reloaded = streamer.resident_tiles[('earth', 'height', 0, 0, 0, 0)]
    assert np.all(streamer.height_tiles[reloaded] == 42)
    assert len(streamer.height_tiles) <= streamer.pool_capacity
    print('CPU height sampling, ancestor fallback, streaming, slot reuse and reload passed')


def test_near_plane():
    au_to_km = 149597870.7
    one_meter = .001 / au_to_km
    for fov in (1., 45., 120.):
        for aspect in (1., 16/9, 32/9):
            near = camera_near_plane(one_meter, au_to_km, fov, aspect)
            np.testing.assert_allclose(near * au_to_km, .0001)
            corner = np.sqrt(1 + np.tan(np.deg2rad(fov/2))**2 * (1 + aspect**2))
            assert near * corner < one_meter
    assert camera_near_plane(100/au_to_km, au_to_km, 45, 16/9) > one_meter
    assert camera_near_plane(1., au_to_km, 45, 16/9) == 1e-4
    print('Ten-centimetre near plane, wide-FOV clearance and orbital adaptation passed')


def test_camera_update(streamer):
    """Execute the shipped movement block without creating a GLFW/UI app."""
    source = (Path(__file__).resolve().parents[1] / 'engine/app.py').read_text(encoding='utf-8')
    start = source.index('            cam = self.camera\n', source.index('# ── Camera: Space Engine'))
    end = source.index('            cam_origin = base_pos\n', start)
    import textwrap
    block = textwrap.dedent(source[start:end])
    au = 149597870.7
    pole = np.array([0., 1., 0.])
    frame = terrain_frame(pole, 0.)
    radius_au = np.float32(6378.1363 / au)
    visual = np.zeros((1, 14), 'f4'); visual[0, 3] = radius_au; visual[0, 5:8] = pole
    info = {'name': 'Earth', 'oblateness': 0., 'height_range_km': [-11., 9.]}
    for tracking, comparison, face, approach in ((True, False, 0, 300.),
                                                (True, False, 1, 0.),
                                                (False, False, 0, 0.),
                                                (True, True, 0, 300.),
                                                (False, True, 1, 0.)):
        surface = CameraSurface(float(radius_au)*au, 0., pole, streamer=streamer,
                                body_name='earth', elevation_range=(-11, 9))
        local = terrain_vertices(face, np.array([[.5, .5]]), surface.radius, 0., np.array([0.]))[0]
        direction = frame @ (local / np.linalg.norm(local))
        # Begin under a mountain/depression; approach must finish at ground,
        # not at an ellipsoid floor. Free flight uses a distant world pivot.
        body_center = np.array([1., 2., 3.])
        offset = np.array([5., 0., 0.]) if comparison else np.zeros(3)
        pivot = body_center + offset if tracking else np.zeros(3)
        r_start = surface.radius + 10. if approach else surface.radius - 15.
        rel = body_center + offset - pivot + direction * r_start / au
        cam = dict(cam_pos_rel=rel, cam_vel=np.zeros(3), yaw=0., yaw_actual=0., pitch=0.,
                   pitch_actual=0., roll=0., roll_actual=0., tracking_idx=0 if tracking else None,
                   tracking_mode='body', tracking_is_cmp=comparison, target=np.zeros(3),
                   approach_delta=approach, centered_idx=None, flight_speed=.01,
                   horizon_align=False, cam_look='free', terrain_lod_enabled=True, fov=45.)
        sim = SimpleNamespace(camera=cam, terrain_streamer=streamer,
                              comparison_enabled=comparison, comparison_offset_au=5., num_bodies_cmp=1,
                              fb_width=1920, fb_height=1080,
                              pos_snap_cmp=body_center[None, :], body_radii_cmp=np.array([radius_au]),
                              bodies_data_cmp=[info], visual_arr_cmp=visual, pole_n_arr=pole[None, :],
                              pole_n_arr_cmp=pole[None, :], body_radii=np.array([radius_au]),
                              atmo_history_valid={0: False, 1: False, 2: False},
                              subsys_pos_buf_cmp=body_center[None, :], subsys_vel_buf_cmp=np.zeros((1, 3)),
                              subsys_mass_buf_cmp=np.ones(1), vel_snap_cmp=np.zeros((1, 3)),
                              mass_snap_cmp=np.ones(1), parent_snap_cmp=np.array([-1]))
        for suffix in ('', '_cmp'):
            for attr in ('rot_period_arr', 'w0_arr', 'tidally_locked_arr', 'parent_idx_arr', 'tangent_arr', 'bitangent_arr'):
                setattr(sim, attr+suffix, np.zeros(1))
        env = dict(np=np, math=math, self=sim, dt_render=1/60, AU_TO_KM=au,
                   _ROT_FOLLOW_ALT_AU=200/au, display_t=0., cmp_sim_t=0., num_bodies=1,
                   pos_snap_render=body_center[None, :], vel_snap_render=np.zeros((1, 3)),
                   mass_snap=np.ones(1), parent_snap=np.array([-1]),
                   subsys_pos_buf=body_center[None, :], subsys_vel_buf=np.zeros((1, 3)),
                   subsys_mass_buf=np.ones(1), bodies_data=[info], visual_arr=visual,
                   body_radii=np.array([radius_au]), CameraSurface=CameraSurface,
                   GROUND_CLEARANCE_KM=GROUND_CLEARANCE_KM,
                   compute_barycenters=lambda *args: None,
                   compute_body_rotation_angles_jit=lambda *args: np.zeros(1))
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('_camera'):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<camera helper>', 'exec'), env)
        exec(compile(block, '<production camera update>', 'exec'), env)
        final = (cam['cam_pos_rel'] + pivot - body_center - offset) * au
        np.testing.assert_allclose(surface.clearance(final), .001, atol=1e-6)
        if not approach:
            assert cam['flight_speed'] <= .1/au
        near_start = source.index('            # Measure from the actual eye,')
        near_end = source.index('            # Incorporate both primary', near_start)
        env.update(cam_origin=env['base_pos'], camera_near_plane=camera_near_plane)
        near_block = textwrap.dedent(source[near_start:near_end])
        exec(compile(near_block, '<production near plane>', 'exec'), env)
        np.testing.assert_allclose(env['near'] * au, .0001, atol=1e-10)
        # At 10 km above the actual terrain the near plane must grow; using the
        # body-center tracking pivot would still pin it to its minimum.
        env['rel'] += direction * 10/au
        exec(compile(near_block, '<production near plane>', 'exec'), env)
        assert env['near'] * au > .19, env['near'] * au
    print('Production approach/free-flight collision and comparison-system pivot handling passed')


def test_gpu():
    import moderngl
    from engine.rendering.shader_loader import load_shader
    ctx = moderngl.create_standalone_context(require=460)
    program = ctx.program(vertex_shader=load_shader('celestial/terrain.vert'), varyings=['f_rel_pos'])
    patch = np.zeros(20, 'f4'); patch[:4] = np.array([37, 41, 38, 42]) * (2/64) - 1
    patch[4] = 1; patch[12:18] = [0, 1, 0, 0, -11, 20]
    patch_buffer = ctx.buffer(patch.tobytes()); patch_buffer.bind_to_storage_buffer(4)
    pole = np.array([.2, .93, .31], 'f4'); pole /= np.linalg.norm(pole)
    body = np.zeros(28, 'f4'); body[6] = 6378.1363; body[9:12] = pole
    body[12] = .00335; body[25] = .7
    body_buffer = ctx.buffer(body.tobytes()); body_buffer.bind_to_storage_buffer(2)
    scene = np.zeros(1832, 'f4'); scene[292:294] = [1000, .1]
    ctx.buffer(scene.tobytes()).bind_to_uniform_block(1)
    texture = ctx.texture_array((8, 8, 1), 4, bytes([255, 255, 255, 255]) * 64)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = texture.repeat_y = False
    texture.use(14); program['u_tile_array'].value = 14; program['u_km_to_au'].value = 1.
    vertices = np.array([[12, 17, 0], [13, 17, 0], [12, 18, 0], [13, 18, 0]], 'f4') / 32
    vb = ctx.buffer(vertices.tobytes()); out = ctx.buffer(reserve=48)
    vao = ctx.vertex_array(program, [(vb, '3f', 'in_position')])
    ctx.simple_framebuffer((4, 4)).use()
    class ConstantHeight:
        def get_tile_slot_or_fallback(self, *args):
            return 0, 1., 0., 0.
        def sample_height(self, slot, uv):
            return np.ones(len(uv))
    clearances = []
    for pole_value, angle, x, y in [((0,1,0), 0., 37,41), ((.2,.93,.31), .7, 37,41),
                                    ((-.51,.41,.72), 1.9, 8,9), ((.2,.93,.31), 2.5, 55,22)]:
        pole = np.array(pole_value, 'f4'); pole /= np.linalg.norm(pole)
        body[9:12] = pole; body[25] = angle; body_buffer.write(body.tobytes())
        patch[:4] = np.array([x,y,x+1,y+1])*(2/64)-1
        frame = terrain_frame(body[9:12], body[25])
        surface = CameraSurface(float(body[6]), float(body[12]), body[9:12], float(body[25]),
                                ConstantHeight(), 'earth', (-11,9), 6,32)
        for face in range(6):
            patch[8] = face; patch_buffer.write(patch.tobytes())
            vao.transform(out, mode=moderngl.POINTS, vertices=4)
            actual = np.frombuffer(out.read(), 'f4').reshape(4, 3)
            uv = (vertices[:, :2].astype('f8') * (patch[2:4].astype('f8') - patch[:2]) + patch[:2] + 1.) * .5
            local = terrain_vertices(face, uv, float(body[6]), float(body[12]), np.full(4, 9.))
            expected = local @ frame.T
            # The production shader stores planetary-scale positions in float32.
            # Check topology/basis here; the signed camera clearance below is the
            # important near-ground assertion rather than exact world coordinates.
            np.testing.assert_allclose(actual, expected, rtol=0, atol=.003)
            center = expected[[0, 2, 1]].mean(axis=0)
            direction = center / np.linalg.norm(center)
            landed, contact = surface.clamp(direction * (np.linalg.norm(center) - 1.))
            assert contact
            a, b, c = actual[[0, 2, 1]].astype('f8')
            normal = np.cross(b-a, c-a); normal /= np.linalg.norm(normal)
            if normal @ a < 0: normal = -normal
            clearance = normal @ (landed - a)
            clearances.append(clearance*1000)
            assert clearance >= .0001, (face, 'Camera intersects rendered ground', clearance)
    print('GPU rendered-ground clearance range (m):', min(clearances), max(clearances))
    ctx.release()
    print('One-metre CPU landing stays above production GPU terrain on all six faces')


def main():
    with tempfile.TemporaryDirectory() as root:
        for face in range(6):
            path = Path(root) / f'earth/height/{face}/0/0_0.png'
            path.parent.mkdir(parents=True)
            Image.new('L', (8, 8), 255 if face % 2 == 0 else 0).save(path)
        streamer = TerrainTileStreamer(CPUContext(), root, pool_capacity=16, tile_size=8)
        try:
            test_geometry(streamer)
            test_motion(streamer)
            test_camera_update(streamer)
            test_streaming(streamer, root)
            test_near_plane()
        finally:
            streamer.shutdown()
    if '--gpu' in sys.argv:
        test_gpu()
    print('Camera-ground regressions passed')


if __name__ == '__main__':
    main()
