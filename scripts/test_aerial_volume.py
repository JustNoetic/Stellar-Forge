"""GPU regression checks for terrain aerial perspective and distance clipping."""
import os
import sys

import moderngl
import numpy as np
from pyrr import matrix44

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rendering.aerial_volume import AerialPerspectiveVolume
from engine.rendering.shaders import atmo_fragment_shader, sky_view_lut_vertex_shader, sky_view_lut_fragment_shader


def main():
    ctx = moderngl.create_context(standalone=True)
    # The shared lighting must still compile in the existing Sky-View pipeline.
    ctx.program(vertex_shader=sky_view_lut_vertex_shader, fragment_shader=sky_view_lut_fragment_shader)
    slices = 129
    volume = AerialPerspectiveVolume(ctx, size=(1, 1, slices))

    projection = matrix44.create_perspective_projection(90.0, 1.0, 0.1, 1000.0, dtype="f4")
    view = matrix44.create_look_at([0, 0, 120], [0, 0, 0], [0, 1, 0], dtype="f4")
    inverse_projection = np.linalg.inv(projection).astype("f4").tobytes()
    inverse_view = np.linalg.inv(view).astype("f4").tobytes()
    scene = np.zeros(1832, dtype="f4")
    scene[:16] = projection.ravel()
    scene[16:32] = view.ravel()
    scene.view("i4")[32] = 1
    scene[36:40] = [0, 0, 1e6, 100]
    scene[100:104] = [1, 1, 1, 1]
    scene[292:294] = [1000, 1]
    scene_buffer = ctx.buffer(scene.tobytes())
    scene_buffer.bind_to_uniform_block(1)
    ctx.buffer(np.zeros(28, dtype="f4").tobytes()).bind_to_storage_buffer(2)
    atmosphere = np.zeros(256, dtype="f4")
    atmosphere[3] = 110
    atmosphere[4:7] = [2e-6, 4e-6, 8e-6]
    atmosphere[7] = atmosphere[11] = 1e8  # effectively constant density
    atmosphere[20:23] = [100, 110, 1]
    atmosphere.view("i4")[23] = 128
    atmosphere[24:28] = [0, 1, 0, 1]
    atmosphere[32:35] = 1
    atmosphere[181:183] = [1, 100]
    atmosphere[184:187] = [1e-8, 1e-8, 1]
    atmosphere[192:196] = [1, 0.8, 0.4, 0.0001]
    atmosphere[208:212] = [0, 0, 1, 0.0001]
    atmosphere[224:228] = [0, 0, 1e6, 0.0001]
    atmosphere[240:244] = [0, 0, 1e6, 100]
    atmo_buffer = ctx.buffer(atmosphere.tobytes())
    atmo_buffer.bind_to_storage_buffer(8)
    sun_trans = ctx.texture((2, 2), 4, np.ones((2, 2, 4), dtype="f4").tobytes(), dtype="f4")
    ms = ctx.texture((2, 2), 4, np.zeros((2, 2, 4), dtype="f4").tobytes(), dtype="f4")
    sun_trans.use(1)
    ms.use(3)

    def bake(key, camera=(0, 0, 120)):
        volume.bake(key, camera, (0, 0, 1), inverse_projection, inverse_view,
                    0, bytes(16 * 3 * 4), False, False, (min(np.linalg.norm(camera),110),95))
        scatter = np.frombuffer(volume.scatter.read(), dtype="f4").reshape(slices, 4)[:, :3]
        trans = np.frombuffer(volume.transmittance.read(), dtype="f4").reshape(slices, 4)[:, :3]
        assert np.all(np.isfinite(scatter)) and np.all(np.isfinite(trans))
        return scatter, trans

    S, T = bake((0, 0))
    np.testing.assert_array_equal(S[0], [0, 0, 0])
    np.testing.assert_array_equal(T[0], [1, 1, 1])
    distances = np.linspace(0,15,slices)
    extinction = np.array([0.002, 0.004, 0.008])
    expected_T = np.exp(-distances[:, None] * extinction)
    phase = 3 / (8 * np.pi)
    expected_S = phase * np.array([1, 0.8, 0.4]) * (1 - expected_T)
    # Up to 256 FP32 transmittance products accumulate several ulps.
    np.testing.assert_allclose(T, expected_T, atol=1e-5)
    np.testing.assert_allclose(S, expected_S, atol=1e-6)
    assert np.all(np.diff(T, axis=0) <= 1e-6)
    assert np.all(np.diff(S, axis=0) >= -1e-6)

    # The coarse terrain cache stays unshadowed. Sharp shadow lighting is
    # resolved per pixel, independently of the sky subtraction method.
    atmosphere.view("i4")[28] = 1
    atmosphere[36:40] = [0, 0, 125, 10]
    atmo_buffer.write(atmosphere.tobytes())
    for method in (0, 1, 2):
        if "u_atmo_shadow_method" in volume.program:
            volume.program["u_atmo_shadow_method"].value = method
        shadow_S, shadow_T = bake((2, method))
        np.testing.assert_allclose(shadow_S, S, atol=1e-7)
        np.testing.assert_allclose(shadow_T, T, atol=1e-7)
    atmosphere.view("i4")[28] = 0
    atmo_buffer.write(atmosphere.tobytes())

    # Independent stars contribute additive radiance without altering view
    # transmittance, including distinct spectral irradiances.
    scene.view("i4")[32] = 2
    scene_buffer.write(scene.tobytes())
    atmosphere[196:200] = [0.2, 0.7, 1.5, 0.0001]
    atmosphere[212:216] = atmosphere[208:212]
    atmosphere[228:232] = atmosphere[224:228]
    atmosphere[244:248] = atmosphere[240:244]
    atmo_buffer.write(atmosphere.tobytes())
    two_S, two_T = bake((4, 0))
    np.testing.assert_allclose(two_T, T, atol=1e-7)
    np.testing.assert_allclose(two_S,
                               phase * np.array([1.2, 1.5, 1.9]) * (1 - expected_T), atol=2e-6)
    scene.view("i4")[32] = 1
    scene_buffer.write(scene.tobytes())
    bake((4, 1))

    # Render the complete production fragment shader against actual logarithmic
    # depth. A cheap proxy supplies the same varyings as atmo.vert.
    vertex = """#version 460 core
    out vec3 f_world_pos; out vec3 f_local_pos; out float f_clip_z;
    void main() {
        vec2 p=vec2((gl_VertexID<<1)&2, gl_VertexID&2);
        gl_Position=vec4(p*2.0-1.0,0.0,1.0);
        f_world_pos=vec3(0.0,0.0,110.0);
        f_local_pos=vec3(0.0,0.0,1.0); f_clip_z=1.0;
    }
    """
    fragment = atmo_fragment_shader.replace(
        "layout(location = 0, index = 0)", "layout(location = 0)"
    ).replace("layout(location = 0, index = 1)", "layout(location = 1)")
    p = ctx.program(vertex_shader=vertex, fragment_shader=fragment)
    vao = ctx.vertex_array(p, [])
    p["u_camera_pos"].value = (0, 0, 120)
    p["u_inv_proj"].write(inverse_projection)
    p["u_inv_view"].write(inverse_view)
    p["u_screen_res"].value = (1, 1)
    p["u_atmo_quality"].value = 3
    p["u_terrain_depth_enabled"].value = True
    p["u_depth_texture"].value = 9
    p["u_aerial_scatter_lut"].value = 20
    p["u_aerial_trans_lut"].value = 21
    volume.bind(p)
    depth = ctx.texture((1, 1), 1, dtype="f4")
    depth.use(9)
    outputs = [ctx.texture((1, 1), 4, dtype="f4") for _ in range(2)]
    fbo = ctx.framebuffer(outputs)

    def render(distance):
        depth.write(np.array([np.log2(distance + 1) / np.log2(1001)], dtype="f4").tobytes())
        fbo.use()
        fbo.clear()
        vao.render(vertices=3)
        return [np.frombuffer(t.read(), dtype="f4")[:3] for t in outputs]

    for distance in (15, 20, 23):  # raised mountain, reference ground, negative valley
        scatter, trans = render(distance)
        optical_length = distance - 10
        correct_T = np.exp(-extinction * optical_length)
        correct_S = phase * np.array([1, 0.8, 0.4]) * (1 - correct_T)
        np.testing.assert_allclose(trans, correct_T, atol=2e-5)
        np.testing.assert_allclose(scatter, correct_S, atol=3e-6)

    # A nearer opaque object entirely in front of the atmosphere blocks it.
    foreground_S, foreground_T = render(5)
    np.testing.assert_array_equal(foreground_S, 0)
    np.testing.assert_array_equal(foreground_T, 0)

    # Ring ordering still composes the same unshadowed path exactly once.
    p["u_num_ring_planes"].value = 1
    ring_params = np.zeros((16, 4), dtype="f4")
    ring_params[0] = [0, 1000, 1, 0]
    ring_centers = np.zeros((16, 3), dtype="f4")
    ring_centers[0] = [0, 0, 102]
    ring_normals = np.zeros((16, 3), dtype="f4")
    ring_normals[0] = [0, 0, 1]
    p["u_ring_params"].write(ring_params.tobytes())
    p["u_ring_center"].write(ring_centers.tobytes())
    p["u_ring_normal"].write(ring_normals.tobytes())
    p["u_atmo_clip_mode"].value = 1
    S_back, T_back = render(23)
    p["u_atmo_clip_mode"].value = 2
    S_front, T_front = render(23)
    np.testing.assert_allclose(T_front * T_back, np.exp(-extinction * 13), atol=2e-5)
    np.testing.assert_allclose(S_front + T_front * S_back,
                               phase * np.array([1, 0.8, 0.4]) * (1 - np.exp(-extinction * 13)), atol=3e-6)

    # Force production twilight refinement, then compare the complete output
    # to Mode 2, including an eclipse caster outside the camera path.
    p["u_num_ring_planes"].value = 0
    p["u_atmo_clip_mode"].value = 0
    p["u_transmittance_lut"].value = 1
    p["u_multi_scatter_lut"].value = 3
    sunlight = np.array([1, 0, -0.05], dtype="f4")
    sunlight /= np.linalg.norm(sunlight)
    saved_atmosphere = atmosphere.copy()
    atmosphere[208:212] = [*sunlight, 1]
    atmosphere[224:228] = [*(sunlight * 1e6), 0.0001]
    atmo_buffer.write(atmosphere.tobytes())
    bake((6, 0))
    volume.bind(p)
    twilight_S, twilight_T = render(20)
    assert twilight_S.max() > 0.001, "Twilight reference must contain scattering"
    p["u_atmo_quality"].value = 2
    reference_S, reference_T = render(20)
    np.testing.assert_allclose(twilight_S, reference_S, atol=3e-6)
    np.testing.assert_allclose(twilight_T, reference_T, atol=2e-5)
    p["u_atmo_quality"].value = 3
    atmosphere.view("i4")[28] = 1
    atmosphere[36:40] = [*(np.array([0, 0, 105]) + sunlight * 5), 2]
    atmo_buffer.write(atmosphere.tobytes())
    caster_S, caster_T = render(20)
    p["u_atmo_quality"].value = 2
    reference_S, reference_T = render(20)
    np.testing.assert_allclose(caster_S, reference_S, atol=3e-6)
    np.testing.assert_allclose(caster_T, reference_T, atol=2e-5)
    assert np.all(caster_S < twilight_S * 0.95), "Caster must shadow twilight aerial perspective"
    np.testing.assert_allclose(caster_T, twilight_T, atol=2e-5)
    p["u_atmo_quality"].value = 3
    atmosphere[:] = saved_atmosphere
    atmo_buffer.write(atmosphere.tobytes())

    # Polar scaling changes geometry, but optical integration must still use
    # world kilometers rather than the stretched spherical-space length.
    atmosphere[24:28] = [0, 0, 1, 1.25]
    atmo_buffer.write(atmosphere.tobytes())
    polar_S, polar_T = bake((3, 0), camera=(0, 0, 150))
    np.testing.assert_allclose(polar_T[-1], np.exp(-extinction * 12), atol=1e-5)

    # Vacuum and a camera below the reference surface must remain finite.
    atmosphere[24:28] = [0, 1, 0, 1]
    atmosphere[4:7] = 0
    atmo_buffer.write(atmosphere.tobytes())
    S, T = bake((1, 0), camera=(0, 0, 98))
    np.testing.assert_allclose(S, 0, atol=1e-6)
    np.testing.assert_allclose(T, 1, atol=1e-6)

    # A coarse 2x2 frustum intentionally straddles the ground horizon. All
    # neighboring columns must be sampled at the SAME short distance, so sky
    # rays cannot contribute their distant haze to a near-ground endpoint.
    from engine.rendering.shader_loader import load_shader
    atmosphere[4:7] = [2e-6, 4e-6, 8e-6]
    atmo_buffer.write(atmosphere.tobytes())
    camera = (0, 0, 100.002)  # eye two meters above reference ground
    horizon_view = matrix44.create_look_at(camera, (1, 0, 100.002), (0, 0, 1), dtype="f4")
    coarse = AerialPerspectiveVolume(ctx, size=(2, 2, 128))
    coarse.bake((5, 0), camera, (0, 0, 1), inverse_projection,
                np.linalg.inv(horizon_view).astype("f4").tobytes(),
                0, bytes(16 * 3 * 4), False, False, (100.002,95))
    coarse.scatter.use(20)
    coarse.transmittance.use(21)
    query_fragment = """#version 460 core
    uniform sampler3D scatter_volume;
    uniform sampler3D trans_volume;
    uniform vec2 query_uv;
    uniform float query_distance;
    uniform mat4 inv_proj;
    uniform mat4 inv_view;
    layout(location=0) out vec4 scatter;
    layout(location=1) out vec4 trans;
    """ + load_shader("common/aerial_volume_mapping.glsl") + """
    void main() {
        vec3 V=aerialColumnRay(query_uv,inv_proj,inv_view,vec4(0,1,0,1));
        vec3 S,T;
        bool valid=sampleAerialGround(scatter_volume,trans_volume,query_uv,query_distance,
            vec3(0,0,100.002),V,110.0,vec2(100.002,95),inv_proj,inv_view,vec4(0,1,0,1),S,T);
        scatter=vec4(S,valid?1:0);
        trans=vec4(T,valid?1:0);
    }
    """
    query_program = ctx.program(vertex_shader=vertex, fragment_shader=query_fragment)
    query_program["scatter_volume"].value = 20
    query_program["trans_volume"].value = 21
    query_program["inv_proj"].write(inverse_projection)
    query_program["inv_view"].write(np.linalg.inv(horizon_view).astype("f4").tobytes())
    query_vao = ctx.vertex_array(query_program, [])
    # Each 2x2 direction has |cos(view,sun)|^2 = 1/6. Their source terms
    # are identical near the camera, although their ground intersections differ.
    coarse_phase = 3 / (16 * np.pi) * (1 + 1 / 6)
    for y in (0.35, 0.45, 0.48, 0.49):
        query_program["query_uv"].value = (0.5, y)
        for distance in (0.001, 0.01, 0.05, 0.1):
            query_program["query_distance"].value = distance
            fbo.use()
            query_vao.render(vertices=3)
            assert np.frombuffer(outputs[0].read(), dtype="f4")[3] == 1, 'Ground query must have a compatible column'
            actual_S, actual_T = [np.frombuffer(t.read(), dtype="f4")[:3] for t in outputs]
            expected_T = np.exp(-extinction * distance)
            expected_S = coarse_phase * np.array([1, 0.8, 0.4]) * (1 - expected_T)
            np.testing.assert_allclose(actual_T, expected_T, atol=5e-6)
            np.testing.assert_allclose(actual_S, expected_S, atol=3e-7)
    query_program["query_uv"].value = (0.5, 0.7)
    query_program["query_distance"].value = 0.1
    query_vao.render(vertices=3)
    assert np.frombuffer(outputs[0].read(), dtype="f4")[3] == 0, 'Ascending sky query must use fallback'
    coarse.release()
    volume.release()
    ctx.release()
    print("Aerial volume GPU tests passed: terrain depth, horizon filtering, twilight shadows, foreground occlusion, ring clipping, unshadowed cache, oblate paths")


if __name__ == "__main__":
    main()
