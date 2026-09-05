import os
import sys
import moderngl

# Ensure repo root is on python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.rendering.shader_loader import load_shader

def main():
    print("Initializing standalone ModernGL context...")
    ctx = moderngl.create_context(standalone=True)

    shaders_to_test = [
        ("prog_spheres", "celestial/sphere.vert", "celestial/sphere.frag"),
        ("prog_point_celestial", "celestial/point_celestial.vert", "celestial/point_celestial.frag"),
        ("prog_rings", "celestial/ring.vert", "celestial/ring.frag"),
        ("prog_ephem_orbits", "celestial/ephem_orbit.vert", "celestial/ephem_orbit.frag"),
        ("prog_gpu_orbits", "celestial/orbit.vert", "celestial/orbit.frag"),
        ("prog_hz", "celestial/hz.vert", "celestial/hz.frag"),
        ("prog_atmo", "atmosphere/atmo.vert", "atmosphere/atmo.frag"),
    ]

    all_passed = True
    for name, vert_path, frag_path in shaders_to_test:
        print(f"Compiling and linking {name} ({vert_path} + {frag_path})...")
        try:
            vert_src = load_shader(vert_path)
            frag_src = load_shader(frag_path)
            prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
            print(f"  -> SUCCESS: {name}")

            if name == "prog_point_celestial":
                refract_uniforms = [
                    'u_refract_center', 'u_refract_radius', 'u_refract_max_bend',
                    'u_refract_scale_height', 'u_refract_pole', 'u_refract_oblateness', 'u_au_to_km'
                ]
                for u in refract_uniforms:
                    if u in prog:
                        print(f"     [+] Found uniform '{u}' in prog_point_celestial")
                    else:
                        print(f"     [-] WARNING: Uniform '{u}' NOT found in prog_point_celestial")
                        all_passed = False
        except Exception as e:
            print(f"  -> FAILED: {name}: {e}")
            all_passed = False

    if all_passed:
        print("\nAll shaders compiled and linked successfully!")
        sys.exit(0)
    else:
        print("\nSome shaders failed validation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
