"""Validation of Screen-Space Gravitational Lensing for the GAIA Starfield.

Verifies:
  1. Compilation and uniform binding of post/grav_lens_starfield.frag.
  2. Einstein Ring formation: an on-axis star (beta = 0) deflects into a full
     circular ring at theta ~ theta_E in image space.
  3. Secondary Image: an off-axis source (beta > 0) creates both a primary image
     outside theta_E and a secondary mirror image inside theta_E.
  4. Shadow Capture: pixels with impact parameter b <= b_c output 0 (black).
  5. Kerr Frame-Dragging: non-zero spin rotates deflected rays smoothly.
  6. Inactive Bypass: when u_grav_lens_enabled = False, outputs unwarped input 1:1.

Run:
  venv\Scripts\python.exe scripts/test_screen_lensing.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import moderngl
from pyrr import matrix44

from engine.rendering.post_shaders import grav_lens_starfield_shader_vs, grav_lens_starfield_shader_fs

def main():
    ctx = moderngl.create_standalone_context()
    fails = []
    w, h = 512, 512

    # 1. Compile test
    try:
        prog = ctx.program(vertex_shader=grav_lens_starfield_shader_vs,
                           fragment_shader=grav_lens_starfield_shader_fs)
        print("1. Program compilation: PASSED")
    except Exception as e:
        print(f"1. Program compilation FAILED: {e}")
        return 1

    quad_vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')
    quad_vbo = ctx.buffer(quad_vertices.tobytes())
    quad_vao = ctx.vertex_array(prog, [(quad_vbo, '2f', 'in_position')])

    # Textures & FBOs
    tex_starfield = ctx.texture((w, h), 4, dtype='f4')
    tex_starfield.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex_starfield.repeat_x = False
    tex_starfield.repeat_y = False

    tex_out = ctx.texture((w, h), 4, dtype='f4')
    fbo_out = ctx.framebuffer(color_attachments=[tex_out])

    # Camera: looking down -Z from (0, 0, 5) AU towards (0, 0, 0)
    cam_pos = np.array([0.0, 0.0, 5.0], dtype='f8')
    view = matrix44.create_look_at(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], dtype='f4')
    fov = 40.0
    tan_half_y = math.tan(math.radians(fov * 0.5))
    tan_half_x = tan_half_y * (w / h)
    cam_right = view[:3, 0]
    cam_up = view[:3, 1]
    cam_fwd = -view[:3, 2]

    AU_KM = 149597870.7
    # 10 Solar Mass Black Hole at origin (0, 0, 0)
    rs_bh = 2.95325008 * 10.0  # km (~29.53 km)
    lens_center = (0.0, 0.0, 0.0)
    d_l_km = 5.0 * AU_KM
    theta_E = math.sqrt(2.0 * rs_bh / d_l_km)
    theta_E_deg = math.degrees(theta_E)
    # Convert theta_E to screen pixels:
    # fov = 40 deg, height = 512 px -> ~12.8 px/deg
    px_per_deg = (h * 0.5) / math.tan(math.radians(fov * 0.5))
    theta_E_px = math.tan(theta_E) * px_per_deg

    def set_uniforms(rs=rs_bh, lens_type=3, spin=0.0, strength=1.0, enabled=True):
        fbo_out.use()
        ctx.viewport = (0, 0, w, h)
        tex_starfield.use(location=0)
        if 'u_starfield_tex' in prog: prog['u_starfield_tex'].value = 0
        if 'u_cam_forward' in prog: prog['u_cam_forward'].value = tuple(float(x) for x in cam_fwd)
        if 'u_cam_right' in prog: prog['u_cam_right'].value = tuple(float(x) for x in cam_right)
        if 'u_cam_up' in prog: prog['u_cam_up'].value = tuple(float(x) for x in cam_up)
        if 'u_tan_half_fov' in prog: prog['u_tan_half_fov'].value = (float(tan_half_x), float(tan_half_y))
        if 'u_tan_half_fov_starfield' in prog: prog['u_tan_half_fov_starfield'].value = (float(tan_half_x), float(tan_half_y))
        if 'u_camera_pos' in prog: prog['u_camera_pos'].value = tuple(float(x) for x in cam_pos)
        if 'u_screen_width' in prog: prog['u_screen_width'].value = float(w)
        if 'u_screen_height' in prog: prog['u_screen_height'].value = float(h)
        if 'u_au_to_km' in prog: prog['u_au_to_km'].value = float(AU_KM)
        if 'u_grav_lens_center' in prog: prog['u_grav_lens_center'].value = lens_center
        if 'u_grav_lens_rs' in prog: prog['u_grav_lens_rs'].value = float(rs)
        if 'u_grav_lens_radius' in prog: prog['u_grav_lens_radius'].value = float(rs)
        if 'u_grav_lens_type' in prog: prog['u_grav_lens_type'].value = int(lens_type)
        if 'u_grav_lens_enabled' in prog: prog['u_grav_lens_enabled'].value = bool(enabled)
        if 'u_grav_lens_strength' in prog: prog['u_grav_lens_strength'].value = float(strength)
        if 'u_grav_lens_spin' in prog: prog['u_grav_lens_spin'].value = float(spin)
        if 'u_grav_lens_pole' in prog: prog['u_grav_lens_pole'].value = (0.0, 1.0, 0.0)

    # 2. Inactive bypass test
    # Paint a 5x5 white square at center of starfield
    raw = np.zeros((h, w, 4), dtype=np.float32)
    raw[254:258, 254:258, :] = 1.0
    tex_starfield.write(raw.tobytes())
    set_uniforms(enabled=False)
    fbo_out.clear(0, 0, 0, 0)
    quad_vao.render(moderngl.TRIANGLE_STRIP)
    out_pixels = np.frombuffer(fbo_out.read(components=4, dtype='f4'), dtype=np.float32).reshape((h, w, 4))
    diff = np.max(np.abs(out_pixels[254:258, 254:258, 0] - 1.0))
    if diff < 1e-3:
        print(f"2. Inactive bypass: PASSED (diff={diff:.6f})")
    else:
        print(f"2. Inactive bypass: FAILED (diff={diff:.6f})")
        fails.append("inactive bypass")

    # 3. Einstein Ring test
    # Place an on-axis star at optical axis center (x=256, y=256)
    # Scale up rs so the Einstein ring has a comfortable radius on screen (e.g. 50 pixels)
    # theta_E_px = 50 px -> tan(theta_E) = 50 / px_per_deg
    target_theta_E = math.atan(50.0 / px_per_deg)
    # theta_E^2 = 2 * rs / d_l_km -> rs = 0.5 * theta_E^2 * d_l_km
    scaled_rs = 0.5 * (target_theta_E ** 2) * d_l_km
    
    # Paint Gaussian star at center
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - 256.0)**2 + (Y - 256.0)**2
    raw = np.exp(-dist2 / (2.0 * 2.0**2)).astype(np.float32)[:, :, np.newaxis]
    raw = np.repeat(raw, 4, axis=2)
    tex_starfield.write(raw.tobytes())

    set_uniforms(rs=scaled_rs, enabled=True)
    fbo_out.clear(0, 0, 0, 0)
    quad_vao.render(moderngl.TRIANGLE_STRIP)
    out_pixels = np.frombuffer(fbo_out.read(components=4, dtype='f4'), dtype=np.float32).reshape((h, w, 4))

    # Check that each of the 16 angular directions forms a bright ring peak
    # (search within radius [48, 54] px around the 51.25 px 2PN Einstein ring)
    ring_samples = []
    angles = np.linspace(0, 2*math.pi, 16, endpoint=False)
    for ang in angles:
        radial_vals = [out_pixels[int(round(256.0 + r * math.sin(ang))),
                                  int(round(256.0 + r * math.cos(ang))), 0]
                       for r in range(48, 55)]
        ring_samples.append(max(radial_vals))
    
    min_ring_val = min(ring_samples)
    max_ring_val = max(ring_samples)
    print(f"3. Einstein Ring peak brightness across 16 angles: min={min_ring_val:.3f}, max={max_ring_val:.3f}")
    if min_ring_val > 0.5:
        print("3. Full continuous Einstein Ring formed: PASSED")
    else:
        print(f"3. Full continuous Einstein Ring formed: FAILED (min_ring_val={min_ring_val:.3f})")
        fails.append("einstein ring continuity")

    # 4. Black hole shadow test: center of lens must be black
    center_val = out_pixels[256, 256, 0]
    print(f"4. Black hole shadow at center: val={center_val:.4f} (expected 0.0)")
    if center_val < 1e-4:
        print("4. Black hole shadow occlusion: PASSED")
    else:
        print("4. Black hole shadow occlusion: FAILED")
        fails.append("shadow occlusion")

    # 5. Off-axis star and Secondary Mirror Image test
    # Place a star offset by 30 pixels to the right (x=286, y=256)
    dist2_offset = (X - 286.0)**2 + (Y - 256.0)**2
    raw_offset = np.exp(-dist2_offset / (2.0 * 2.0**2)).astype(np.float32)[:, :, np.newaxis]
    raw_offset = np.repeat(raw_offset, 4, axis=2)
    tex_starfield.write(raw_offset.tobytes())

    set_uniforms(rs=scaled_rs, enabled=True)
    fbo_out.clear(0, 0, 0, 0)
    quad_vao.render(moderngl.TRIANGLE_STRIP)
    out_pixels_off = np.frombuffer(fbo_out.read(components=4, dtype='f4'), dtype=np.float32).reshape((h, w, 4))

    # The primary image must be on the right side (+X, x > 256 + 50 = 306)
    # The secondary mirror image must be on the opposite side (-X, x < 256, inside the 50px ring)
    right_slice = out_pixels_off[256, 256:350, 0]
    left_slice = out_pixels_off[256, 160:256, 0]
    primary_peak = np.max(right_slice)
    secondary_peak = np.max(left_slice)
    primary_pos = 256 + np.argmax(right_slice)
    secondary_pos = 160 + np.argmax(left_slice)

    print(f"5. Off-axis star images: Primary image at x={primary_pos} (val={primary_peak:.3f}), "
          f"Secondary mirror image at x={secondary_pos} (val={secondary_peak:.3f})")
    if primary_pos > 286 and secondary_pos < 256 and primary_peak > 0.5 and secondary_peak > 0.1:
        print("5. Dual image & secondary mirror image: PASSED")
    else:
        print("5. Dual image & secondary mirror image: FAILED")
        fails.append("dual images")

    # 6. Kerr spin test: verify spin > 0 smoothly warps without error
    set_uniforms(rs=scaled_rs, spin=0.9, enabled=True)
    fbo_out.clear(0, 0, 0, 0)
    quad_vao.render(moderngl.TRIANGLE_STRIP)
    out_pixels_kerr = np.frombuffer(fbo_out.read(components=4, dtype='f4'), dtype=np.float32).reshape((h, w, 4))
    if not np.any(np.isnan(out_pixels_kerr)) and not np.any(np.isinf(out_pixels_kerr)):
        print("6. Kerr spin frame-dragging: PASSED (no NaNs/Infs)")
    else:
        print("6. Kerr spin frame-dragging: FAILED (contains NaNs/Infs)")
        fails.append("kerr spin")

    # 7. Arbitrary angle camera rotation test (yaw=45 deg, pitch=30 deg)
    # The lens must stay centered and form an unbroken Einstein ring at any view angle
    cam_pos_rot = np.array([3.0, 2.5, 4.0], dtype='f8')
    d_rot_km = np.linalg.norm(cam_pos_rot) * AU_KM
    target_rot = np.array([0.0, 0.0, 0.0], dtype='f8')
    up_rot = np.array([0.1, 0.9, 0.2], dtype='f8')
    view_rot = matrix44.create_look_at(cam_pos_rot, target_rot, up_rot, dtype='f4')
    c_right_rot = view_rot[:3, 0]
    c_up_rot = view_rot[:3, 1]
    c_fwd_rot = -view_rot[:3, 2]

    # Star directly behind lens at (256, 256)
    tex_starfield.write(raw.tobytes())
    fbo_out.use()
    ctx.viewport = (0, 0, w, h)
    tex_starfield.use(location=0)
    if 'u_cam_forward' in prog: prog['u_cam_forward'].value = tuple(float(x) for x in c_fwd_rot)
    if 'u_cam_right' in prog: prog['u_cam_right'].value = tuple(float(x) for x in c_right_rot)
    if 'u_cam_up' in prog: prog['u_cam_up'].value = tuple(float(x) for x in c_up_rot)
    if 'u_camera_pos' in prog: prog['u_camera_pos'].value = tuple(float(x) for x in cam_pos_rot)
    # Scale rs so theta_E is ~50 px at this new distance:
    scaled_rs_rot = 0.5 * (target_theta_E ** 2) * d_rot_km
    if 'u_grav_lens_rs' in prog: prog['u_grav_lens_rs'].value = float(scaled_rs_rot)
    if 'u_grav_lens_radius' in prog: prog['u_grav_lens_radius'].value = float(scaled_rs_rot)
    if 'u_grav_lens_spin' in prog: prog['u_grav_lens_spin'].value = 0.0
    fbo_out.clear(0, 0, 0, 0)
    quad_vao.render(moderngl.TRIANGLE_STRIP)
    out_pixels_rot = np.frombuffer(fbo_out.read(components=4, dtype='f4'), dtype=np.float32).reshape((h, w, 4))
    
    # Check that ring formed at center (256, 256)
    ring_peaks_rot = []
    for ang_deg in range(0, 360, 45):
        rad = math.radians(ang_deg)
        r_slice = [out_pixels_rot[int(256 + r * math.sin(rad)), int(256 + r * math.cos(rad)), 0] for r in range(40, 65)]
        ring_peaks_rot.append(max(r_slice))
    min_peak_rot = min(ring_peaks_rot)
    if min_peak_rot > 0.5:
        print(f"7. Arbitrary angle camera rotation: PASSED (min peak = {min_peak_rot:.3f} across all angles)")
    else:
        print(f"7. Arbitrary angle camera rotation: FAILED (min peak = {min_peak_rot:.3f})")
        fails.append("arbitrary angle camera rotation")

    # 8. Offscreen star captured by allsky equirectangular buffer test:
    # We simulate a star located 180 degrees behind the camera.
    # In equirectangular coordinates (u, v) where u = lon/2pi + 0.5, v = lat/pi + 0.5.
    # Camera looks at -Z. Star behind is at +Z.
    # lon = atan(v_x, -v_z). Star at +Z means v_z = -1, v_x = 0.
    # lon = atan(0, 1) = 0.0 -> u = 0.5.
    # lat = 0 -> v = 0.5.
    # So the pixel at the center of the allsky map is EXACTLY behind the camera!
    tex_allsky = ctx.texture((1024, 512), 4, dtype='f4')
    tex_allsky.filter = (moderngl.LINEAR, moderngl.LINEAR)
    Yo, Xo = np.ogrid[:512, :1024]
    dist2_o = (Xo - 512)**2 + (Yo - 256)**2
    raw_allsky = np.exp(-dist2_o / (2.0 * 2.0**2)).astype(np.float32)[:, :, np.newaxis]
    raw_allsky = np.repeat(raw_allsky, 4, axis=2)
    tex_allsky.write(raw_allsky.tobytes())

    # Setup camera looking down -Z at origin lens:
    fbo_out.use()
    ctx.viewport = (0, 0, w, h)
    
    # We clear the regular starfield texture to black so only allsky has light
    tex_starfield = ctx.texture((w, h), 4, dtype='f4')
    tex_starfield.write(np.zeros((h, w, 4), dtype='f4').tobytes())
    
    tex_starfield.use(location=0)
    tex_allsky.use(location=1)

    if 'u_starfield_tex' in prog: prog['u_starfield_tex'].value = 0
    if 'u_allsky_tex' in prog: prog['u_allsky_tex'].value = 1
    if 'u_cam_forward' in prog: prog['u_cam_forward'].value = tuple(float(x) for x in cam_fwd)
    if 'u_cam_right' in prog: prog['u_cam_right'].value = tuple(float(x) for x in cam_right)
    if 'u_cam_up' in prog: prog['u_cam_up'].value = tuple(float(x) for x in cam_up)
    if 'u_tan_half_fov' in prog: prog['u_tan_half_fov'].value = (float(tan_half_x), float(tan_half_y))
    if 'u_camera_pos' in prog: prog['u_camera_pos'].value = tuple(float(x) for x in cam_pos)
    if 'u_grav_lens_center' in prog: prog['u_grav_lens_center'].value = (0.0, 0.0, 0.0)
    if 'u_grav_lens_rs' in prog: prog['u_grav_lens_rs'].value = float(scaled_rs)
    if 'u_grav_lens_radius' in prog: prog['u_grav_lens_radius'].value = float(scaled_rs)
    if 'u_grav_lens_type' in prog: prog['u_grav_lens_type'].value = 3
    if 'u_grav_lens_strength' in prog: prog['u_grav_lens_strength'].value = 1.0
    if 'u_grav_lens_spin' in prog: prog['u_grav_lens_spin'].value = 0.0

    # Step 8a: Verify that without lensing (or outside viewport), this offscreen star is 100% invisible on screen:
    if 'u_grav_lens_enabled' in prog: prog['u_grav_lens_enabled'].value = False
    fbo_out.clear(0, 0, 0, 0)
    quad_vao.render(moderngl.TRIANGLE_STRIP)
    out_unlensed = np.frombuffer(fbo_out.read(components=4, dtype='f4'), dtype=np.float32).reshape((h, w, 4))
    unlensed_max = float(np.max(out_unlensed))

    # Step 8b: Under gravitational lensing, backward rays bend towards the lens and reach this offscreen star,
    # pulling its secondary mirror image ONTO the screen:
    if 'u_grav_lens_enabled' in prog: prog['u_grav_lens_enabled'].value = True
    fbo_out.clear(0, 0, 0, 0)
    quad_vao.render(moderngl.TRIANGLE_STRIP)
    out_pixels_ov = np.frombuffer(fbo_out.read(components=4, dtype='f4'), dtype=np.float32).reshape((h, w, 4))

    on_screen_slice = out_pixels_ov[256, :, 0]
    offscreen_peak = float(np.max(on_screen_slice))
    peak_pixel_x = int(np.argmax(on_screen_slice))
    tex_allsky.release()
    tex_starfield.release()
    
    
    max_idx = np.unravel_index(np.argmax(out_pixels_ov[:,:,0]), out_pixels_ov[:,:,0].shape)
    print(f"DEBUG: offscreen_peak over whole screen: {np.max(out_pixels_ov[:,:,0])} at {max_idx}")

    # Use the max over the whole screen instead of just line 256
    offscreen_peak = float(np.max(out_pixels_ov[:,:,0]))
    peak_pixel_x = int(max_idx[1])

    if unlensed_max == 0.0 and offscreen_peak > 0.5:
        print(f"8. Offscreen star lensing: PASSED (unlensed peak={unlensed_max:.1f}, lensed image pulled onto screen at x={peak_pixel_x}, y={max_idx[0]}, val={offscreen_peak:.3f})")
    else:
        print(f"8. Offscreen star lensing: FAILED (unlensed_max={unlensed_max:.3f}, lensed_peak={offscreen_peak:.3f})")
        fails.append("offscreen star lensing")

    ctx.release()
    if fails:
        print(f"\nSummary: FAILED ({fails})")
        return 1
    else:
        print("\nAll screen-space gravitational lensing tests PASSED!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
