"""
Test Spatiotemporal Blue Noise (STBN) loading, shader integration, and Mode 3 stochastic raymarching.
"""
import os
import sys
import re
import moderngl
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLSL = os.path.join(ROOT, "engine", "glsl")

def load(rel):
    """Resolve #include directives exactly like engine/rendering/shader_loader.py."""
    inc_re = re.compile(r'^\s*#include\s+["<]([^">]+)[">]\s*$', re.MULTILINE)
    path = os.path.join(GLSL, rel)

    def repl(m):
        inc = m.group(1)
        c1 = os.path.join(os.path.dirname(path), inc)
        c2 = os.path.join(GLSL, inc)
        target = c1 if os.path.isfile(c1) else c2
        if not os.path.isfile(target):
            raise FileNotFoundError(f"include not found: {inc}")
        return load(os.path.relpath(target, GLSL).replace("\\", "/"))

    return inc_re.sub(repl, open(path, encoding="utf-8").read())

def main():
    ctx = moderngl.create_context(standalone=True)
    print(f"GL: {ctx.info['GL_VERSION']}")

    bin_path = os.path.join("textures", "noise", "stbn_scalar.bin")
    if not os.path.exists(bin_path):
        print(f"FAIL: {bin_path} does not exist!")
        sys.exit(1)

    with open(bin_path, "rb") as f:
        data = f.read()

    assert len(data) == 128 * 128 * 64, f"Unexpected STBN size: {len(data)}"
    tex3d = ctx.texture3d((128, 128, 64), 1, data=data, dtype='f1')
    tex3d.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex3d.repeat_x = True
    tex3d.repeat_y = True
    tex3d.repeat_z = True

    # Smoke compile atmo.vert + atmo.frag with u_stbn_tex
    vs_src = load("atmosphere/atmo.vert")
    fs_src = load("atmosphere/atmo.frag")

    prog = ctx.program(vertex_shader=vs_src, fragment_shader=fs_src)
    assert 'u_stbn_tex' in prog, "u_stbn_tex missing from prog_atmo uniforms!"
    assert 'u_atmo_noise_type' in prog, "u_atmo_noise_type missing from prog_atmo uniforms!"
    print("atmo.frag with u_stbn_tex and u_atmo_noise_type: compiled successfully and uniforms found!")

    # Smoke compile atmo_godrays.frag
    fs_godrays = load("atmosphere/atmo_godrays.frag")
    prog_godrays = ctx.program(vertex_shader=vs_src, fragment_shader=fs_godrays)
    assert 'u_stbn_tex' in prog_godrays, "u_stbn_tex missing from atmo_godrays uniforms!"
    assert 'u_atmo_noise_type' in prog_godrays, "u_atmo_noise_type missing from atmo_godrays uniforms!"
    print("atmo_godrays.frag with u_stbn_tex and u_atmo_noise_type: compiled successfully and uniforms found!")

    # Verify STBN sampling distribution
    vs_quad = """
    #version 460 core
    in vec2 in_pos;
    void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
    """
    fs_quad = """
    #version 460 core
    uniform sampler3D u_stbn_tex;
    uniform float u_frame_idx;
    out vec4 fragColor;

    float get_stbn(vec2 screen_pos, float frame_idx) {
        ivec3 sz = textureSize(u_stbn_tex, 0);
        if (sz.x >= 64) {
            ivec3 coord = ivec3(
                int(screen_pos.x) & 127,
                int(screen_pos.y) & 127,
                int(frame_idx) & 63
            );
            float base_noise = texelFetch(u_stbn_tex, coord, 0).r;
            float cycle = floor(frame_idx / 64.0);
            return fract(base_noise + cycle * 0.61803398875);
        }
        return 0.5;
    }

    void main() {
        float n = get_stbn(gl_FragCoord.xy, u_frame_idx);
        fragColor = vec4(n, n, n, 1.0);
    }
    """
    test_prog = ctx.program(vertex_shader=vs_quad, fragment_shader=fs_quad)
    test_prog['u_stbn_tex'].value = 0
    tex3d.use(location=0)

    fbo = ctx.framebuffer(color_attachments=[ctx.texture((256, 256), 4, dtype='f4')])
    fbo.use()
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4'))
    vao = ctx.vertex_array(test_prog, [(vbo, '2f', 'in_pos')])

    # Test frames 0, 1, 63, 64 (wrapping cycle), 100
    for frame in [0, 1, 63, 64, 100]:
        test_prog['u_frame_idx'].value = float(frame)
        vao.render(moderngl.TRIANGLE_STRIP)
        res = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype=np.float32).reshape((256, 256, 4))[:, :, 0]
        mean_v = float(res.mean())
        std_v = float(res.std())
        min_v = float(res.min())
        max_v = float(res.max())
        print(f"Frame {frame:3d}: mean={mean_v:.4f}, std={std_v:.4f}, min={min_v:.4f}, max={max_v:.4f}")
        assert 0.45 < mean_v < 0.55, f"Mean outside expected range: {mean_v}"
        assert 0.25 < std_v < 0.35, f"Std outside expected range: {std_v}"

    print("PASS: Spatiotemporal Blue Noise (STBN) integration fully verified!")

if __name__ == "__main__":
    main()
