"""Finite sunset haze from an 11 km Earth camera to terrain at 0--8 km.

Compare the shipped endpoint solver with 1024-cell direct integration.
Both paths share incoming-light optical depth to isolate endpoint interpolation;
test_scattering_segment.py independently validates the density and sun columns.
The display comparison uses exposure 11. The production-depth test separately
checks that Earth's -11.034 km terrain clipping bound cannot change this haze.
"""
from pathlib import Path
import sys

import moderngl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.rendering.shader_loader import load_shader
from engine.rendering.scattering_lut import ScatteringLUTCache
from test_scattering_segment import PARAMETERS


def display(radiance, transmission):
    # A constant brown surface exposes bands in atmospheric transport alone.
    color = radiance + transmission * np.array([.015, .01, .004])
    x = np.maximum(0, color) * 11
    return np.clip(x * (2.51*x + .03) / (x * (2.43*x + .59) + .14), 0, 1)**(1/2.2)


def main():
    ctx = moderngl.create_standalone_context(require=460)
    multiple = ctx.texture((64, 64), 4, np.zeros((64, 64, 4), 'f4').tobytes(), dtype='f4')
    multiple.filter = (moderngl.LINEAR, moderngl.LINEAR)
    vertex = '''#version 460
    void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
    }'''
    declarations = '\n'.join('uniform vec3 ' + key + ';' if isinstance(value, tuple)
                              else 'uniform float ' + key + ';'
                              for key, value in PARAMETERS.items()
                              if key not in ('u_scattering_bottom_km', 'u_scattering_sun_radius'))
    fragment = '#version 460\n#define PI 3.14159265358979323846\n' + declarations + '''
    uniform sampler2D u_multi_scatter_lut;
    uniform bool u_reference;
    uniform float u_azimuth, u_sun_mu;
    layout(location=0) out vec4 out_L;
    layout(location=1) out vec4 out_T;
    ''' + load_shader('common/sun_terminator.glsl') + load_shader('common/scattering_segment.glsl') + '''
    void main() {
        float r = u_planet_radius_km + 11.0;
        float mu = mix(-0.20, -0.027, gl_FragCoord.y / 512.0);
        vec3 a = vec3(0, r, 0), v = vec3(sqrt(1.0 - mu*mu), mu, 0);
        float horizontal = sqrt(1.0 - u_sun_mu*u_sun_mu);
        vec3 sun = vec3(horizontal*cos(u_azimuth), u_sun_mu, horizontal*sin(u_azimuth));
        float terrain_radius = u_planet_radius_km + mix(0.0, 8.0, gl_FragCoord.x / 128.0);
        float disc = terrain_radius*terrain_radius - r*r + r*r*mu*mu;
        if (disc < 0.0) { out_L = vec4(0); out_T = vec4(0); return; }
        float distance = -r*mu - sqrt(disc);
        vec3 b = a + distance*v;
        vec3 T = endpoint_transmittance(a, b), L = vec3(0);
        if (!u_reference) {
            L = endpoint_radiance(a, b, sun, T);
        } else {
            T = vec3(1);
            float nu = dot(v, sun), g = u_mie_g;
            float phase_R = 3.0/(16.0*PI) * (1.0 + nu*nu);
            float phase_M = 3.0/(8.0*PI) * (1.0 - g*g)/(2.0 + g*g) * (1.0 + nu*nu)
                / pow(1.0 + g*g - 2.0*g*nu, 1.5);
            // These finite terrain rays descend monotonically. Grade the cells
            // toward the endpoint where the density is highest.
            const int N = 1024;
            for (int i = 0; i < N; ++i) {
                float t0 = float(i)/N, t1 = float(i + 1)/N;
                float s0 = distance * (1.0 - (1.0 - t0)*(1.0 - t0));
                float s1 = distance * (1.0 - (1.0 - t1)*(1.0 - t1));
                float ds = s1 - s0;
                vec3 p = a + 0.5*(s0 + s1)*v, R, M, extinction;
                endpoint_medium(p, R, M, extinction);
                float rq = length(p), sin_p = u_planet_radius_km/max(rq, u_planet_radius_km + 0.01);
                vec2 term = sun_terminator(dot(p, sun)/rq, sin_p, sqrt(max(0.0, 1.0 - sin_p*sin_p)),
                    u_scattering_sun_radius, sqrt(1.0 - u_scattering_sun_radius*u_scattering_sun_radius));
                float mus = max(term.y, scattering_horizon(rq) + 1e-5);
                vec3 radial = p/rq, tangent = sun - dot(sun, radial)*radial;
                tangent = length(tangent) > 1e-6 ? normalize(tangent) : normalize(cross(radial, vec3(0, 0, 1)));
                vec3 effective = radial*mus + tangent*sqrt(max(0.0, 1.0 - mus*mus));
                vec3 incoming = exp(-endpoint_tau(p, effective, false)) * term.x;
                vec3 cell_T = exp(-extinction*ds);
                vec3 factor = mix((1.0 - cell_T)/max(extinction, vec3(1e-20)),
                    vec3(ds), lessThan(extinction*ds, vec3(1e-4)));
                L += T * factor * (R*phase_R + M*phase_M) * incoming;
                T *= cell_T;
            }
        }
        out_L = vec4(L, 1); out_T = vec4(T, 1);
    }'''
    program = ctx.program(vertex_shader=vertex, fragment_shader=fragment)
    for name, value in PARAMETERS.items():
        if name in program:
            program[name].value = value
    ScatteringLUTCache.configure(program)
    program['u_multi_scatter_lut'].value = 3
    multiple.use(3)
    cache = ScatteringLUTCache(ctx)
    cache.bind(cache.get(PARAMETERS, multiple))
    program['u_scattering_azimuth_count'].value = cache.size[3]
    targets = [ctx.texture((128, 512), 4, dtype='f4') for _ in range(2)]
    fbo = ctx.framebuffer(targets)
    vao = ctx.vertex_array(program, [])
    fbo.use(); ctx.viewport = (0, 0, 128, 512)
    for sun_mu, azimuth in ((-.03, 0.), (-.05, 0.), (-.06, 0.), (-.05, .7)):
        program['u_sun_mu'].value = sun_mu
        program['u_azimuth'].value = azimuth
        frames = []
        for reference in (False, True):
            program['u_reference'].value = reference
            vao.render(moderngl.TRIANGLES, vertices=3)
            frames.append([np.frombuffer(t.read(), 'f4').reshape(512, 128, 4).copy() for t in targets])
        (actual, ta), (reference, tr) = frames
        valid = reference[..., 3] > .5
        actual, reference, ta, tr = (x[..., :3] for x in (actual, reference, ta, tr))
        assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
        relative = np.linalg.norm(actual - reference, axis=2) / np.maximum(np.linalg.norm(reference, axis=2), .001)
        error99 = np.quantile(relative[valid], .99)
        display_delta = np.max(abs(display(actual, ta) - display(reference, tr)), axis=2)
        print(f'Sunset mu={sun_mu}, az={azimuth}: radiance p99={error99:.4f}, '
              f'exposure 11 display p99={np.quantile(display_delta[valid], .99):.4f}')
        assert error99 < .12, (sun_mu, azimuth, 'horizon endpoint banding', error99)
        assert np.quantile(display_delta[valid], .99) < .025, 'Visible sunset haze bands'
        assert np.max(abs(ta - tr)[valid]) < .006, 'Horizon transmission error'
    cache.release(); multiple.release(); ctx.release()
    print('11 km sunset aerial-perspective horizon regressions passed')


if __name__ == '__main__':
    main()
