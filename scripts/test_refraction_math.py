"""GPU regression for point/mesh atmospheric refraction alignment.

Run: python scripts/test_refraction_math.py
Uses the shipped GLSL and the sphere/ring/atmosphere anchored-ray geometry.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moderngl
import numpy as np

from engine.rendering.shader_loader import load_shader

COMPUTE = """
#version 430 core
layout(local_size_x = 1) in;
SHARED_REFRACTION
layout(std430, binding=0) buffer Out { vec4 result; };
uniform vec3 t_C;
uniform vec3 t_V;
uniform float t_d;
void main() {
    bool occluded;
    vec3 apparent = solve_refraction_apparent(t_C, t_V, t_d, occluded);
    float alpha = compute_refraction_total(t_C, apparent, t_d);
    vec3 up = normalize(t_C - apparent * dot(t_C, apparent));
    vec3 ray = normalize(apparent * cos(alpha) - up * sin(alpha));
    float anchor = max(0.0, -dot(t_C, apparent));
    vec3 origin = anchor * (apparent - ray);
    // Intersect the bent ray with a plane through the true object, normal to
    // its true sightline. The transverse miss is the point/mesh LOD mismatch.
    float t = (t_d - dot(origin, t_V)) / dot(ray, t_V);
    vec3 hit = origin + t * ray;
    float miss = length(hit - t_V * dot(hit, t_V)) / t_d;
    float lift = atan(length(cross(t_V, apparent)), dot(t_V, apparent));
    result = vec4(miss, lift, alpha, occluded ? 1.0 : 0.0);
}
"""


def main():
    ctx = moderngl.create_standalone_context(require=430)
    prog = ctx.compute_shader(COMPUTE.replace("SHARED_REFRACTION", load_shader("common/refraction.glsl")))
    buf = ctx.buffer(reserve=16)
    buf.bind_to_storage_buffer(0)
    for name, value in (
        ("u_refract_radius", 6371.0),
        ("u_refract_max_bend", math.radians(68.0 / 60.0)),
        ("u_refract_scale_height", 8.5),
        ("u_refract_oblateness", 0.0),
        ("u_refract_pole", (0.0, 1.0, 0.0)),
    ):
        if name in prog:
            prog[name].value = value
    cases = []
    for elevation in (0.0, 2.0, 5.0, 10.0, 45.0, 89.0):
        e = math.radians(elevation)
        cases.append((f"ground {elevation:g} deg", (6371.002, 0.0, 0.0), (math.sin(e), 0.0, -math.cos(e))))
    cases.append(("space grazing limb", (6376.0, 0.0, 10000.0), (0.0, 0.0, -1.0)))
    cases.append(("above atmosphere", (6571.0, 0.0, 0.0), (math.sin(0.1), 0.0, -math.cos(0.1))))
    failures = []
    try:
        for label, C, V in cases:
            prog["t_C"].value = C
            prog["t_V"].value = V
            prog["t_d"].value = 1e9
            prog.run(group_x=1)
            miss, lift, alpha, occluded = np.frombuffer(buf.read(), dtype=np.float32)
            arcmin = 180.0 * 60.0 / math.pi
            print(f"{label}: point lift={lift * arcmin:.4f}', mesh turn={alpha * arcmin:.4f}', mismatch={miss * arcmin:.4f}'")
            # Bisection resolves a 68 arcmin bracket to <0.017 arcmin. Allow
            # float32 error and the tiny finite-distance anchor approximation.
            if not math.isfinite(miss) or miss * arcmin > 0.05 or occluded:
                failures.append(label)
            if label == "ground 5 deg" and not 5.0 < lift * arcmin < 20.0:
                failures.append("ground bend unexpectedly absent")
            if label == "above atmosphere" and lift * arcmin > 0.001:
                failures.append("bend above atmospheric cutoff")
    finally:
        buf.release()
        prog.release()
        ctx.release()
    if failures:
        raise AssertionError(f"Refraction alignment failed: {failures}")
    print("All refraction alignment cases PASSED")


if __name__ == "__main__":
    main()
