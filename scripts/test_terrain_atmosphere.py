"""Regression tests for terrain-aware atmosphere depth endpoints."""
import os
import sys

import moderngl
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rendering.shaders import atmo_vertex_shader, atmo_fragment_shader


def _endpoint_helper():
    begin = "// TERRAIN_DEPTH_ENDPOINT_BEGIN"
    end = "// TERRAIN_DEPTH_ENDPOINT_END"
    return atmo_fragment_shader.split(begin, 1)[1].split(end, 1)[0]


def main():
    ctx = moderngl.create_context(standalone=True)

    # Compile the complete production atmosphere pair first.
    ctx.program(vertex_shader=atmo_vertex_shader, fragment_shader=atmo_fragment_shader)
    lowres_fragment = atmo_fragment_shader.replace(
        "layout(location = 0, index = 0) out vec4 out_scattered;",
        "layout(location = 0) out vec4 out_scattered;",
    ).replace(
        "layout(location = 0, index = 1) out vec4 out_transmittance;",
        "layout(location = 1) out vec4 out_transmittance;",
    )
    ctx.program(vertex_shader=atmo_vertex_shader, fragment_shader=lowres_fragment)

    vertex = """#version 460 core
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    }
    """
    fragment = """#version 460 core
    uniform float u_depth;
    uniform float u_current_end;
    uniform bool u_terrain;
    out vec4 color;
    """ + _endpoint_helper() + """
    void main() {
        float s_end = u_current_end;
        bool hit = true;
        bool terrain_override = false;
        resolveTerrainDepthEndpoint(
            u_depth, 0.0, vec2(0.0, 120.0),
            vec3(0.0, 0.0, 110.0), vec3(0.0, 0.0, -1.0),
            110.0, 10.0, u_terrain,
            s_end, hit, terrain_override);
        color = vec4(s_end, hit ? 1.0 : 0.0,
                     terrain_override ? 1.0 : 0.0, 1.0);
    }
    """
    prog = ctx.program(vertex_shader=vertex, fragment_shader=fragment)
    vao = ctx.vertex_array(prog, [])
    tex = ctx.texture((1, 1), 4, dtype="f4")
    fbo = ctx.framebuffer([tex])

    def resolve(depth, current_end=10.0, terrain=True):
        prog["u_depth"].value = depth
        prog["u_current_end"].value = current_end
        prog["u_terrain"].value = terrain
        fbo.use()
        vao.render(vertices=3)
        return np.frombuffer(tex.read(), dtype=np.float32)

    # Mountain above the reference ellipsoid shortens aerial perspective.
    np.testing.assert_allclose(resolve(7.0)[:3], [7.0, 1.0, 1.0])
    # A below-reference valley extends it when terrain rendering owns the depth.
    np.testing.assert_allclose(resolve(13.0)[:3], [13.0, 1.0, 1.0])
    # Ordinary non-terrain geometry cannot push the endpoint through the ellipsoid.
    np.testing.assert_allclose(resolve(13.0, terrain=False)[:3], [10.0, 1.0, 0.0])
    # A nearer ring/caster clip remains authoritative over a terrain valley.
    np.testing.assert_allclose(resolve(13.0, current_end=5.0)[:3], [5.0, 1.0, 0.0])
    # Depth outside the local atmosphere is rejected as a background object.
    np.testing.assert_allclose(resolve(250.0)[:3], [10.0, 1.0, 0.0])

    ctx.release()
    print("Terrain-aware atmosphere endpoint tests passed")


if __name__ == "__main__":
    main()
