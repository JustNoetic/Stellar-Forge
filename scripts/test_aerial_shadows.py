"""Finite terrain shadow regressions using the complete production atmosphere.

Compare Mode 3's selective refinement with Mode 2, including RGB multi-scatter,
multiple stars, partial/full coverage, overlapping casters, oblate geometry,
ring transparency/gaps, foreground clipping, and shadows beyond opaque depth.
"""
from pathlib import Path
import sys

import moderngl
import numpy as np
from pyrr import matrix44

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.rendering.aerial_volume import AerialPerspectiveVolume
from engine.rendering.ring_opacity import RingOpacityBounds
from engine.rendering.shaders import atmo_fragment_shader


def main():
    ctx = moderngl.create_context(standalone=True)
    volume = AerialPerspectiveVolume(ctx)
    scene = np.zeros(1832, dtype='f4')
    projection = matrix44.create_perspective_projection(45, 1, .1, 1000, dtype='f4')
    view = matrix44.create_look_at([0, 0, 120], [0, 0, 0], [0, 1, 0], dtype='f4')
    ip = np.linalg.inv(projection).astype('f4').tobytes()
    iv = np.linalg.inv(view).astype('f4').tobytes()
    scene[:16], scene[16:32] = projection.ravel(), view.ravel()
    scene.view('i4')[32] = 1
    scene[292:294] = [1000, 1]
    scene_buffer = ctx.buffer(scene.tobytes())
    scene_buffer.bind_to_uniform_block(1)
    instances = np.zeros(28, dtype='f4')
    instance_buffer = ctx.buffer(instances.tobytes())
    instance_buffer.bind_to_storage_buffer(2)
    atmo = np.zeros(256, dtype='f4')
    atmo[3] = 110
    atmo[4:7] = [2e-6, 4e-6, 8e-6]
    atmo[7] = atmo[11] = 1e8
    atmo[19] = 1
    atmo[20:23] = [100, 110, 1]
    atmo.view('i4')[23] = 128
    atmo[24:28] = [0, 1, 0, 1]
    atmo[32:35] = 1
    atmo[181:183] = [1, 100]
    atmo[184:187] = [1e-8, 1e-8, 1]
    atmo_buffer = ctx.buffer(atmo.tobytes())
    atmo_buffer.bind_to_storage_buffer(8)
    sunlight = np.array([.6, 0, .8], dtype='f4')

    def star(index, direction, color):
        scene[36 + index*4:40 + index*4] = [*(direction*1e6), 100]
        scene[100 + index*4:104 + index*4] = [*color, 1]
        atmo[192 + index*4:196 + index*4] = [*color, .0001]
        atmo[208 + index*4:212 + index*4] = [*direction, 1]
        atmo[224 + index*4:228 + index*4] = [*(direction*1e6), .0001]
        atmo[240 + index*4:244 + index*4] = [0, 0, 1e6, 100]
    star(0, sunlight, (1, .8, .4))
    scene_buffer.write(scene.tobytes())
    atmo_buffer.write(atmo.tobytes())
    sun = ctx.texture((2, 2), 4, np.ones((2, 2, 4), dtype='f4').tobytes(), dtype='f4')
    # Nonzero diffuse scattering must also be blocked; it cannot fill an umbra.
    ms = ctx.texture((2, 2), 4, np.full((2, 2, 4), .04, dtype='f4').tobytes(), dtype='f4')
    sun.use(1)
    ms.use(3)
    vertex = '''#version 460 core
out vec3 f_world_pos; out vec3 f_local_pos; out float f_clip_z;
void main() {
vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
gl_Position=vec4(p*2-1,0,1);
f_world_pos=vec3(0,0,110);f_local_pos=vec3(0,0,1);f_clip_z=1;
}'''
    fragment = atmo_fragment_shader.replace(
        'layout(location = 0, index = 0)', 'layout(location = 0)').replace(
        'layout(location = 0, index = 1)', 'layout(location = 1)')
    fragment = fragment.replace('void main() {', 'layout(location=2) out vec4 selection;\nvoid main() {', 1)
    fragment = fragment.replace('vec3 transmittance = final_transmittance;',
        'selection=vec4(aerial_shadow_refinement,aerial_refinement,aerial_sample_valid?1:0,1);\n'
        'vec3 transmittance = final_transmittance;')
    p = ctx.program(vertex_shader=vertex, fragment_shader=fragment)
    vao = ctx.vertex_array(p, [])
    for name, value in {
        'u_camera_pos': (0, 0, 120), 'u_screen_res': (1, 1),
        'u_terrain_depth_enabled': True, 'u_depth_texture': 9,
        'u_transmittance_lut': 1, 'u_multi_scatter_lut': 3,
        'u_aerial_scatter_lut': 20, 'u_aerial_trans_lut': 21,
        'u_ring_gradients': 0,
    }.items():
        p[name].value = value
    p['u_inv_proj'].write(ip)
    p['u_inv_view'].write(iv)
    depth = ctx.texture((1, 1), 1, dtype='f4')
    depth.use(9)
    outputs = [ctx.texture((1, 1), 4, dtype='f4') for _ in range(3)]
    fbo = ctx.framebuffer(outputs)
    key = 0

    def bake(camera=(0, 0, 120)):
        nonlocal key
        key += 1
        scene_buffer.write(scene.tobytes())
        atmo_buffer.write(atmo.tobytes())
        instance_buffer.write(instances.tobytes())
        volume.bake((key, 0), camera, sunlight, ip, iv, 0, bytes(192), False, False,
                    (min(np.linalg.norm(camera), 110), 95))
        volume.bind(p)

    def render(quality=3, distance=20):
        atmo_buffer.write(atmo.tobytes())
        p['u_atmo_quality'].value = quality
        depth.write(np.array([np.log2(distance+1)/np.log2(1001)], dtype='f4').tobytes())
        fbo.use()
        fbo.clear()
        vao.render(vertices=3)
        return [np.frombuffer(t.read(), dtype='f4')[:3] for t in outputs]

    def compare(distance=20):
        actual_S, actual_T, selected = render(3, distance)
        ref_S, ref_T, _ = render(2, distance)
        np.testing.assert_allclose(actual_S, ref_S, atol=3e-6)
        np.testing.assert_allclose(actual_T, ref_T, atol=2e-5)
        assert selected[0] == 1, 'Shadowed ray must fully select physical lighting'
        assert selected[2] == 1, 'Fixture must exercise a valid volume sample'
        return actual_S, actual_T

    def caster(radius, height=105, index=0):
        atmo.view('i4')[28] = index + 1
        atmo[36 + 4*index:40 + 4*index] = [*(np.array([0, 0, height]) + sunlight*30), radius]

    if '--benchmark' in sys.argv:
        import statistics
        size = 512
        print('GPU:', ctx.info['GL_RENDERER'])
        p['u_screen_res'].value = (size, size)
        p['u_atmo_quality'].value = 3
        p['u_aerial_shadow_steps'].value = 32
        atmo.view('i4')[23] = 11
        sunlight = np.array([.6, .35, np.sqrt(1-.6**2-.35**2)], dtype='f4')
        star(0, sunlight, (1, .8, .4))
        coords = ((np.arange(size) + .5) / size * 2 - 1) * np.tan(np.pi / 8)
        x, y = np.meshgrid(coords, coords)
        rays = np.stack([x, y, -np.ones_like(x)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        b = 120 * rays[:, :, 2]
        distance = -b - np.sqrt(b*b - (120**2 - 100**2))
        depth.release()
        depth = ctx.texture((size, size), 1,
            (np.log2(distance * -rays[:, :, 2] + 1) / np.log2(1001)).astype('f4').tobytes(), dtype='f4')
        depth.use(9)
        fbo.release()
        for t in outputs: t.release()
        outputs = [ctx.texture((size, size), 4, dtype='f4') for _ in range(3)]
        fbo = ctx.framebuffer(outputs)
        bake()
        fbo.use()
        ctx.viewport = (0, 0, size, size)

        def measure(label):
            for _ in range(8): vao.render(vertices=3)
            ctx.finish()
            times = []
            for _ in range(15):
                with ctx.query(time=True) as query: vao.render(vertices=3)
                times.append(query.elapsed / 1e6)
            selected = np.frombuffer(outputs[2].read(), dtype='f4').reshape(size, size, 4)
            print(f'{label}: {statistics.median(times):.3f} ms, '
                  f'shadow refinement {np.mean(selected[:, :, 0] > 0) * 100:.1f}%')

        measure('Clear terrain')
        params = np.zeros((16, 4), dtype='f4'); params[0] = [110, 200, 1, 100]
        normals = np.zeros((16, 3), dtype='f4'); normals[0] = [0, 1, 0]
        masks = np.zeros(16, dtype='u4'); masks[0] = 1
        p['u_num_ring_planes'].value = 1
        p['u_ring_params'].write(params.tobytes())
        p['u_ring_normal'].write(normals.tobytes())
        p['u_ring_center'].write(bytes(192))
        p['u_ring_coplanar_mask'].write(masks.tobytes())
        instances.view('u4')[15] = 1; instance_buffer.write(instances.tobytes())
        gradient = np.ones((16, 64, 4), dtype='f4')
        rings = ctx.texture((64, 16), 4, gradient.tobytes(), dtype='f4'); rings.use(0)
        bounds = RingOpacityBounds(ctx, width=64); bounds.update(gradient); bounds.bind(p)
        measure('Opaque ring, bounded 32-step shadows')
        atmo.view('i4')[23] = 128; atmo_buffer.write(atmo.tobytes())
        measure('Opaque ring, uniform 128-step reference')
        atmo.view('i4')[23] = 11; atmo_buffer.write(atmo.tobytes())
        gradient[:, :, 3] = 0; rings.write(gradient.tobytes()); bounds.update(gradient)
        measure('Transparent ring')
        times = []
        for i in range(12):
            with ctx.query(time=True) as query:
                volume.bake((1000+i, 0), (0, 0, 120), sunlight, ip, iv, 0, bytes(192),
                            False, False, (110, 95))
            times.append(query.elapsed / 1e6)
        print(f'Volume bake: {statistics.median(times[2:]):.3f} ms')
        bounds.release(); volume.release(); ctx.release()
        return

    bake()
    clear_S, clear_T, selected = render()
    assert selected[0] == selected[1] == 0, 'Clear daylight should use only the volume'
    assert selected[2] == 1
    for method in (0, 1, 2):
        p['u_atmo_shadow_method'].value = method
        caster(4)
        full_S, full_T = compare()
        assert full_S.max() < 1e-7, 'A fully covering umbra must remove direct and diffuse haze'
        np.testing.assert_allclose(full_T, clear_T, atol=2e-5)
    caster(1)
    partial_S, _ = compare()
    assert np.all((partial_S > clear_S*.4) & (partial_S < clear_S*.9)), 'Only the shadowed air parcel should darken'
    # The performance path concentrates a low budget inside finite penumbrae.
    # A thin shadow must remain visible even when uniform 11-step rays miss it.
    p['u_aerial_shadow_steps'].value = 32
    for radius in (1.0, .1):
        caster(radius)
        atmo.view('i4')[23] = 11
        budget_S, budget_T, selected = render()
        atmo.view('i4')[23] = 512
        reference_S, reference_T, _ = render(2)
        np.testing.assert_allclose(budget_S, reference_S, atol=3e-5, rtol=.01)
        np.testing.assert_allclose(budget_T, reference_T, atol=3e-5)
        assert np.all(budget_S < clear_S), 'Bounded quadrature must retain thin shadows'
    caster(.1, height=103)
    caster(.1, height=108, index=1)
    atmo.view('i4')[23] = 11
    separated_S, separated_T, _ = render()
    atmo.view('i4')[23] = 512
    separated_ref_S, separated_ref_T, _ = render(2)
    np.testing.assert_allclose(separated_S, separated_ref_S, atol=3e-5, rtol=.01)
    np.testing.assert_allclose(separated_T, separated_ref_T, atol=3e-5)
    assert np.all(separated_S < clear_S), 'Separated thin shadows must survive the low budget'
    atmo.view('i4')[23] = 128
    caster(1)
    caster(1, index=1)
    overlap_S, _ = compare()
    np.testing.assert_allclose(overlap_S, partial_S, atol=3e-6)
    # Shadows beyond raised opaque terrain cannot darken the foreground air.
    caster(.5, height=101)
    raised_S, raised_T, selected = render(distance=15)
    assert selected[0] == 0, 'Shadow behind opaque depth must be culled'
    atmo.view('i4')[28] = 0
    clear_raised_S, clear_raised_T, _ = render(distance=15)
    np.testing.assert_array_equal(raised_S, clear_raised_S)
    np.testing.assert_array_equal(raised_T, clear_raised_T)
    caster(1, height=98)
    compare(distance=23)  # below-reference terrain retains a longer finite path
    # The second star is outside the first star's eclipse cone.
    star(1, np.array([-.6, 0, .8], dtype='f4'), (.2, .7, 1.5))
    scene.view('i4')[32] = 2
    atmo.view('i4')[28] = 0
    bake()
    two_clear, _, _ = render()
    caster(4)
    two_shadow, _ = compare()
    assert two_shadow.min() > .0001, 'One eclipse must not darken the other star'
    np.testing.assert_allclose(two_shadow, two_clear - clear_S, atol=4e-6)
    scene.view('i4')[32] = 1
    atmo.view('i4')[28] = 0
    bake()
    # Ring plane is behind the camera ray, but between its air and the star.
    # This tests volumetric lighting rather than ring/scene-depth occlusion.
    params = np.zeros((16, 4), dtype='f4')
    normals = np.zeros((16, 3), dtype='f4')
    centers = np.zeros((16, 3), dtype='f4')
    normals[0] = [0, 0, 1]
    centers[0] = [0, 0, 140]
    params[0] = [0, 100, 1, 0]
    masks = np.zeros(16, dtype='u4')
    masks[0] = 1
    p['u_num_ring_planes'].value = 1
    p['u_ring_params'].write(params.tobytes())
    p['u_ring_normal'].write(normals.tobytes())
    p['u_ring_center'].write(centers.tobytes())
    p['u_ring_coplanar_mask'].write(masks.tobytes())
    instances.view('u4')[15] = 1
    instance_buffer.write(instances.tobytes())
    gradient = np.ones((16, 64, 4), dtype='f4')
    rings = ctx.texture((64, 16), 4, gradient.tobytes(), dtype='f4')
    rings.filter = (moderngl.LINEAR, moderngl.LINEAR)
    rings.repeat_x = rings.repeat_y = False
    rings.use(0)
    opacity_bounds = RingOpacityBounds(ctx, width=64)
    opacity_bounds.update(gradient)
    opacity_bounds.bind(p)
    opaque_S, opaque_T = compare()
    assert opaque_S.max() < 1e-7, 'Opaque ring must remove the whole path light'
    # The production CPU already folds opacity into the unified gradient alpha.
    gradient[:, :, 3] = .5
    rings.write(gradient.tobytes())
    opacity_bounds.update(gradient)
    transparent_S, _ = compare()
    assert np.all((transparent_S > clear_S*.25) & (transparent_S < clear_S*.7))
    # Exponentially varying density must also converge with the reduced
    # shadow budget; constant-density fixtures alone cannot catch this error.
    saved_atmo = atmo.copy()
    atmo[7], atmo[11] = 8, 1.2
    atmo[8:11] = 2e-6
    atmo[184:186] = [1/8, 1/1.2]
    bake()
    atmo.view('i4')[23] = 11
    dense_S, dense_T, selected = render()
    atmo.view('i4')[23] = 512
    dense_reference_S, dense_reference_T, _ = render(2)
    np.testing.assert_allclose(dense_S, dense_reference_S, atol=3e-5, rtol=.01)
    np.testing.assert_allclose(dense_T, dense_reference_T, atol=3e-5)
    atmo[:] = saved_atmo
    bake()
    # Positive plane opacity is insufficient: an empty texture, and a finite
    # interval wholly inside a gap, must bypass the shadow marcher entirely.
    for opaque_elsewhere in (False, True):
        gradient[:, :, 3] = 0
        if opaque_elsewhere:
            gradient[:, 40, 3] = 1
        rings.write(gradient.tobytes())
        opacity_bounds.update(gradient)
        empty_S, empty_T, selected = render()
        assert selected[0] == 0, 'Transparent radial intervals must use the cache'
        np.testing.assert_allclose(empty_S, clear_S, atol=3e-6)
        np.testing.assert_allclose(empty_T, clear_T, atol=2e-5)
    # A resolved transparent gap crossing only part of the path preserves haze.
    gradient[:, :, 3] = 1
    gradient[:, 15:17, 3] = 0
    rings.write(gradient.tobytes())
    opacity_bounds.update(gradient)
    params[0, 2] = 1
    p['u_ring_params'].write(params.tobytes())
    gap_S, _ = compare()
    assert np.all((gap_S > opaque_S) & (gap_S < clear_S))
    # Own rings and external rings use the same per-pixel lighting geometry.
    centers[0] = 0
    normals[0] = [0, 0, -1]
    star(0, np.array([.6, 0, -.8], dtype='f4'), (1, .8, .4))
    scene_buffer.write(scene.tobytes())
    params[0] = [0, 1000, 1, 0]
    gradient[:] = 1
    rings.write(gradient.tobytes())
    opacity_bounds.update(gradient)
    p['u_ring_params'].write(params.tobytes())
    p['u_ring_normal'].write(normals.tobytes())
    p['u_ring_center'].write(centers.tobytes())
    bake()
    own_S, _ = compare()
    assert own_S.max() < 1e-7, 'Own rings must shadow night-side multiple scattering too'
    # Polar scaling must not shift Cartesian eclipse cones.
    p['u_num_ring_planes'].value = 0
    instances.view('u4')[15] = 0
    instance_buffer.write(instances.tobytes())
    star(0, sunlight, (1, .8, .4))
    atmo[24:28] = [0, 0, 1, 1.25]
    caster(2, height=84)
    bake(camera=(0, 0, 150))
    p['u_camera_pos'].value = (0, 0, 120)
    compare(distance=40)
    foreground_S, foreground_T, _ = render(distance=5)
    np.testing.assert_array_equal(foreground_S, 0)
    np.testing.assert_array_equal(foreground_T, 0)
    volume.release()
    opacity_bounds.release()
    ctx.release()
    print('Terrain aerial shadow GPU tests passed: full/partial eclipses, overlap, finite depth, stars, rings, twilight, oblate paths')


if __name__ == '__main__':
    main()
