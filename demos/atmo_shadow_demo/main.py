import os
import sys
import math
import numpy as np
from PIL import Image

import glfw
import moderngl
import pyrr
import imgui
from imgui.integrations.glfw import GlfwRenderer


def create_uv_sphere(lat_bands=64, lon_bands=128, radius=1.0):
    """Generates an interleaved (pos3, norm3) vertex array and index buffer for a sphere."""
    vertices = []
    indices = []

    for i in range(lat_bands + 1):
        lat = math.pi * (-0.5 + float(i) / lat_bands)
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)

        for j in range(lon_bands + 1):
            lon = 2.0 * math.pi * float(j) / lon_bands
            sin_lon = math.sin(lon)
            cos_lon = math.cos(lon)

            x = cos_lat * cos_lon
            y = sin_lat
            z = cos_lat * sin_lon

            # Position (3f) and Normal (3f)
            vertices.extend([x * radius, y * radius, z * radius, x, y, z])

    for i in range(lat_bands):
        for j in range(lon_bands):
            first = i * (lon_bands + 1) + j
            second = first + lon_bands + 1

            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_ring_mesh(inner_r=1.144886, outer_r=2.266241, segments=512):
    """Generates a triangle-strip annular mesh with interleaved (pos3, uv_r1)."""
    vertices = []
    theta_step = 2.0 * math.pi / segments

    for i in range(segments + 1):
        theta = i * theta_step
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Inner vertex (uv_r = 0.0)
        vertices.extend([inner_r * cos_t, 0.0, inner_r * sin_t, 0.0])
        # Outer vertex (uv_r = 1.0)
        vertices.extend([outer_r * cos_t, 0.0, outer_r * sin_t, 1.0])

    return np.array(vertices, dtype=np.float32)


class Camera:
    def __init__(self, fov=55.0, aspect=16.0 / 9.0, near=0.001, far=100.0):
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        self.mode = "orbit"  # "orbit" or "freefly"

        # Orbit parameters
        self.orbit_yaw = 40.0       # degrees
        self.orbit_pitch = 22.0     # degrees
        self.orbit_distance = 4.2   # distance from target
        self.orbit_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Free-fly parameters
        self.pos = np.array([0.0, 1.5, 3.5], dtype=np.float32)
        self.fly_yaw = 0.0    # compass azimuth (0 = North, 90 = East)
        self.fly_pitch = 0.0  # elevation from local horizon (-89 to +89)
        self.move_speed = 0.8 # base speed multiplier

    def set_mode(self, new_mode):
        if self.mode == new_mode:
            return
        if new_mode == "freefly":
            # Transition seamlessly from orbit to freefly
            eye = self.get_orbit_eye()
            self.pos = eye.copy()
            # Look towards orbit target
            fwd = self.orbit_target - eye
            fwd /= max(np.linalg.norm(fwd), 1e-6)
            _, _, _, up_loc, north, east = self.get_local_basis(self.pos)
            f_up = np.dot(fwd, up_loc)
            self.fly_pitch = math.degrees(math.asin(max(-1.0, min(1.0, f_up))))
            f_north = np.dot(fwd, north)
            f_east = np.dot(fwd, east)
            self.fly_yaw = math.degrees(math.atan2(f_east, f_north))
            self.mode = "freefly"
        elif new_mode == "orbit":
            # Transition from freefly to orbit
            D = np.linalg.norm(self.pos)
            self.orbit_distance = max(1.002, min(20.0, float(D)))
            norm_pos = self.pos / max(D, 1e-6)
            self.orbit_pitch = math.degrees(math.asin(max(-1.0, min(1.0, norm_pos[1]))))
            self.orbit_yaw = math.degrees(math.atan2(norm_pos[0], norm_pos[2]))
            self.mode = "orbit"

    def get_orbit_eye(self):
        rad_yaw = math.radians(self.orbit_yaw)
        rad_pitch = math.radians(self.orbit_pitch)
        cam_x = self.orbit_target[0] + self.orbit_distance * math.cos(rad_pitch) * math.sin(rad_yaw)
        cam_y = self.orbit_target[1] + self.orbit_distance * math.sin(rad_pitch)
        cam_z = self.orbit_target[2] + self.orbit_distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        return np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    def get_local_basis(self, pos):
        D = np.linalg.norm(pos)
        up_local = pos / max(D, 1e-6)
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(up_local[1]) < 0.99 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        east = np.cross(ref, up_local)
        east /= max(np.linalg.norm(east), 1e-6)
        north = np.cross(up_local, east)
        north /= max(np.linalg.norm(north), 1e-6)

        rad_p = math.radians(self.fly_pitch)
        rad_y = math.radians(self.fly_yaw)
        fwd = math.cos(rad_p) * (math.cos(rad_y) * north + math.sin(rad_y) * east) + math.sin(rad_p) * up_local
        fwd /= max(np.linalg.norm(fwd), 1e-6)

        right = np.cross(fwd, up_local)
        if np.linalg.norm(right) < 1e-5:
            right = east.copy()
        else:
            right /= np.linalg.norm(right)

        cam_up = np.cross(right, fwd)
        cam_up /= max(np.linalg.norm(cam_up), 1e-6)
        return fwd, right, cam_up, up_local, north, east

    def get_eye_position(self):
        if self.mode == "orbit":
            return self.get_orbit_eye()
        else:
            return self.pos

    def get_altitude(self):
        eye = self.get_eye_position()
        return max(0.0, float(np.linalg.norm(eye) - 1.0))

    def get_projection_matrix(self):
        # Dynamically scale near plane with altitude to prevent near clipping
        alt = self.get_altitude()
        near = max(0.0002, min(0.05, alt * 0.2))
        return pyrr.matrix44.create_perspective_projection_matrix(
            self.fov, self.aspect, near, self.far, dtype=np.float32
        )

    def get_view_matrix(self):
        if self.mode == "orbit":
            eye = self.get_orbit_eye()
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return pyrr.matrix44.create_look_at(eye, self.orbit_target, up, dtype=np.float32)
        else:
            fwd, right, cam_up, up_local, _, _ = self.get_local_basis(self.pos)
            target = self.pos + fwd
            return pyrr.matrix44.create_look_at(self.pos, target, cam_up, dtype=np.float32)

    def rotate(self, dx, dy):
        if self.mode == "orbit":
            self.orbit_yaw += dx * 0.3
            self.orbit_pitch = max(-89.0, min(89.0, self.orbit_pitch + dy * 0.3))
        else:
            self.fly_yaw = (self.fly_yaw + dx * 0.2) % 360.0
            self.fly_pitch = max(-89.0, min(89.0, self.fly_pitch - dy * 0.2))

    def move(self, forward, strafe, vertical, speed_mult, dt):
        if self.mode != "freefly":
            return

        alt = self.get_altitude()
        eff_speed = self.move_speed * max(0.005, alt * 1.2) * speed_mult

        fwd, right, cam_up, up_local, _, _ = self.get_local_basis(self.pos)

        move_vec = (fwd * forward + right * strafe + up_local * vertical) * eff_speed * dt
        self.pos += move_vec

        # Prevent clipping inside planet core (minimum altitude 0.0005)
        dist = np.linalg.norm(self.pos)
        min_r = 1.0005
        if dist < min_r:
            self.pos = (self.pos / dist) * min_r

    def preset_orbit(self):
        self.mode = "orbit"
        self.orbit_distance = 4.2
        self.orbit_pitch = 22.0
        self.orbit_yaw = 40.0

    def preset_cloud_surfing(self):
        self.mode = "freefly"
        pos = np.array([0.0, 0.3, 1.075], dtype=np.float32)
        self.pos = (pos / np.linalg.norm(pos)) * 1.075
        self.fly_pitch = -5.0
        self.fly_yaw = 0.0

    def preset_mid_atmo(self):
        self.mode = "freefly"
        pos = np.array([0.5, 0.2, 0.9], dtype=np.float32)
        self.pos = (pos / np.linalg.norm(pos)) * 1.035
        self.fly_pitch = 15.0
        self.fly_yaw = 180.0

    def preset_surface_daylight(self):
        self.mode = "freefly"
        lat = math.radians(22.0)
        lon = math.radians(45.0)
        pos = np.array([math.cos(lat)*math.sin(lon), math.sin(lat), math.cos(lat)*math.cos(lon)], dtype=np.float32)
        self.pos = pos * 1.003
        self.fly_pitch = 30.0
        self.fly_yaw = 180.0

    def preset_surface_eclipse(self):
        self.mode = "freefly"
        lat = math.radians(-22.0)
        lon = math.radians(45.0)
        pos = np.array([math.cos(lat)*math.sin(lon), math.sin(lat), math.cos(lat)*math.cos(lon)], dtype=np.float32)
        self.pos = pos * 1.003
        self.fly_pitch = 30.0
        self.fly_yaw = 0.0

    def preset_surface_sunset(self):
        self.mode = "freefly"
        lat = math.radians(0.0)
        lon = math.radians(-42.0)
        pos = np.array([math.cos(lat)*math.sin(lon), math.sin(lat), math.cos(lat)*math.cos(lon)], dtype=np.float32)
        self.pos = pos * 1.002
        self.fly_pitch = 6.0
        self.fly_yaw = 90.0


class App:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        self.width = 1280
        self.height = 720
        self.window = glfw.create_window(self.width, self.height, "Atmosphere & Shadow Prototype - Phase 3 (Analytical Volumetric Shadow)", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.ctx = moderngl.create_context()
        self.camera = Camera(fov=55.0, aspect=self.width / self.height)

        # ImGui setup
        imgui.create_context()
        self.imgui_renderer = GlfwRenderer(self.window)

        # Mouse & keyboard state
        self.mouse_down_left = False
        self.mouse_down_right = False
        self.last_x = 0.0
        self.last_y = 0.0

        glfw.set_cursor_pos_callback(self.window, self.on_cursor_pos)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_window_size_callback(self.window, self.on_resize)

        # Load Shaders
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "shaders", "sphere.vert"), "r") as f:
            sphere_vs = f.read()
        with open(os.path.join(base_dir, "shaders", "sphere.frag"), "r") as f:
            sphere_fs = f.read()

        with open(os.path.join(base_dir, "shaders", "ring.vert"), "r") as f:
            ring_vs = f.read()
        with open(os.path.join(base_dir, "shaders", "ring.frag"), "r") as f:
            ring_fs = f.read()

        with open(os.path.join(base_dir, "shaders", "sky_view_lut.vert"), "r") as f:
            sky_view_vs = f.read()
        with open(os.path.join(base_dir, "shaders", "sky_view_lut.frag"), "r") as f:
            sky_view_fs = f.read()

        with open(os.path.join(base_dir, "shaders", "atmo_composite.vert"), "r") as f:
            atmo_comp_vs = f.read()
        with open(os.path.join(base_dir, "shaders", "atmo_composite.frag"), "r") as f:
            atmo_comp_fs = f.read()

        self.prog_sphere = self.ctx.program(vertex_shader=sphere_vs, fragment_shader=sphere_fs)
        self.prog_ring = self.ctx.program(vertex_shader=ring_vs, fragment_shader=ring_fs)
        self.prog_sky_view = self.ctx.program(vertex_shader=sky_view_vs, fragment_shader=sky_view_fs)
        self.prog_atmo_comp = self.ctx.program(vertex_shader=atmo_comp_vs, fragment_shader=atmo_comp_fs)

        # Geometry
        sphere_v, sphere_i = create_uv_sphere(lat_bands=64, lon_bands=128, radius=1.0)
        self.sphere_vbo = self.ctx.buffer(sphere_v.tobytes())
        self.sphere_ibo = self.ctx.buffer(sphere_i.tobytes())
        self.sphere_vao = self.ctx.vertex_array(
            self.prog_sphere,
            [(self.sphere_vbo, '3f 3f', 'in_position', 'in_normal')],
            index_buffer=self.sphere_ibo
        )

        self.ring_inner = 1.144886
        self.ring_outer = 2.266241
        ring_v = create_ring_mesh(self.ring_inner, self.ring_outer, segments=512)
        self.ring_vbo = self.ctx.buffer(ring_v.tobytes())
        self.ring_vao = self.ctx.vertex_array(
            self.prog_ring,
            [(self.ring_vbo, '3f 1f', 'in_position', 'in_uv_r')]
        )
        self.ring_vertex_count = len(ring_v) // 4

        # Fullscreen quad for LUT and composite
        quad_v = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad_v.tobytes())
        self.quad_vao_lut = self.ctx.vertex_array(self.prog_sky_view, [(self.quad_vbo, '2f', 'in_position')])
        self.quad_vao_comp = self.ctx.vertex_array(self.prog_atmo_comp, [(self.quad_vbo, '2f', 'in_position')])

        # Load Ring Texture
        repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        tex_path = os.path.join(repo_root, "textures", "Solar System", "Saturn", "Rings.png")
        if not os.path.exists(tex_path):
            raise FileNotFoundError(f"Cannot find ring texture at: {tex_path}")

        ring_img = Image.open(tex_path).convert("RGBA")
        self.ring_tex_res = float(ring_img.size[0])
        self.ring_tex = self.ctx.texture(ring_img.size, 4, ring_img.tobytes())
        self.ring_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.ring_tex.build_mipmaps()
        self.ring_tex.repeat_x = False
        self.ring_tex.repeat_y = False

        # Framebuffers: Sky-View LUT (256x256 RGBA16F)
        self.sky_view_res = (256, 256)
        self.sky_view_tex = self.ctx.texture(self.sky_view_res, 4, dtype='f2')
        self.sky_view_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.sky_view_tex.repeat_x = True # Seamless azimuth
        self.sky_view_tex.repeat_y = False
        self.sky_view_fbo = self.ctx.framebuffer(color_attachments=[self.sky_view_tex])

        # Framebuffers: Scene Color + Depth
        self.scene_color_tex = None
        self.scene_depth_tex = None
        self.scene_fbo = None
        self.recreate_scene_fbo(self.width, self.height)

        # Simulation Parameters
        self.sun_azimuth = 45.0       # degrees
        self.sun_elevation = 26.0     # degrees
        self.sun_angular_deg = 0.53   # degrees
        self.sun_color = [1.0, 1.0, 1.0]
        self.planet_albedo = [0.92, 0.65, 0.35]  # Warm Saturn orange
        self.ambient = 0.04
        self.penumbra_samples = 9

        # Atmosphere Parameters (Phase 2 & Phase 3)
        self.atmo_enabled = True
        self.volumetric_shadow = True
        self.analytical_slicing = True
        self.planet_radius = 1.0
        self.atmo_radius = 1.08
        self.scale_height = 0.02
        self.atmo_density = 24.0
        self.atmo_tint = [0.35, 0.65, 1.0] # Soft sky blue
        self.show_lut_preview = False

    def recreate_scene_fbo(self, w, h):
        if self.scene_fbo:
            self.scene_fbo.release()
            self.scene_color_tex.release()
            self.scene_depth_tex.release()

        self.scene_color_tex = self.ctx.texture((w, h), 4, dtype='f2')
        self.scene_color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_depth_tex = self.ctx.depth_texture((w, h))
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_color_tex],
            depth_attachment=self.scene_depth_tex
        )

    def on_cursor_pos(self, window, xpos, ypos):
        io = imgui.get_io()
        if io.want_capture_mouse:
            self.last_x = xpos
            self.last_y = ypos
            return

        dx = xpos - self.last_x
        dy = ypos - self.last_y

        if self.mouse_down_left or self.mouse_down_right:
            self.camera.rotate(dx, dy)

        self.last_x = xpos
        self.last_y = ypos

    def on_mouse_button(self, window, button, action, mods):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_down_left = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_down_right = (action == glfw.PRESS)

    def on_scroll(self, window, xoffset, yoffset):
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        if self.camera.mode == "orbit":
            self.camera.orbit_distance = max(1.002, min(20.0, self.camera.orbit_distance * (0.9 ** yoffset)))
        else:
            self.camera.move_speed = max(0.05, min(5.0, self.camera.move_speed * (1.15 ** yoffset)))

    def on_key(self, window, key, scancode, action, mods):
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return
        if action == glfw.PRESS:
            if key in (glfw.KEY_TAB, glfw.KEY_F):
                new_mode = "freefly" if self.camera.mode == "orbit" else "orbit"
                self.camera.set_mode(new_mode)

    def handle_movement(self, dt):
        if self.camera.mode != "freefly":
            return
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return

        fwd = 0.0
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            fwd += 1.0
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            fwd -= 1.0

        strafe = 0.0
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            strafe += 1.0
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            strafe -= 1.0

        vertical = 0.0
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS or glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            vertical += 1.0
        if glfw.get_key(self.window, glfw.KEY_C) == glfw.PRESS or glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            vertical -= 1.0

        speed_mult = 1.0
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS:
            speed_mult = 4.0
        elif glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or glfw.get_key(self.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS:
            speed_mult = 0.25

        if fwd != 0.0 or strafe != 0.0 or vertical != 0.0:
            self.camera.move(fwd, strafe, vertical, speed_mult, dt)

    def on_resize(self, window, width, height):
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.camera.aspect = width / height
            self.recreate_scene_fbo(width, height)

    def compute_sun_vector(self):
        az_rad = math.radians(self.sun_azimuth)
        el_rad = math.radians(self.sun_elevation)
        lx = math.cos(el_rad) * math.sin(az_rad)
        ly = math.sin(el_rad)
        lz = math.cos(el_rad) * math.cos(az_rad)
        sun_vec = np.array([lx, ly, lz], dtype=np.float32)
        return sun_vec / np.linalg.norm(sun_vec)

    def render(self, render_gui=True):
        proj = self.camera.get_projection_matrix()
        view = self.camera.get_view_matrix()
        model = pyrr.matrix44.create_identity(dtype=np.float32)

        sun_dir = self.compute_sun_vector()
        sun_ang_rad = math.radians(self.sun_angular_deg * 0.5)
        eye_pos = self.camera.get_eye_position()

        # =========================================================
        # Pass 1: Render 256x256 Sky-View LUT (Precomputed In-scatter)
        # =========================================================
        if self.atmo_enabled:
            self.sky_view_fbo.use()
            self.ctx.viewport = (0, 0, self.sky_view_res[0], self.sky_view_res[1])
            self.ctx.disable(moderngl.DEPTH_TEST)

            self.prog_sky_view['u_cam_pos'].value = tuple(eye_pos)
            self.prog_sky_view['u_sun_dir'].value = tuple(sun_dir)
            self.prog_sky_view['u_sun_color'].value = tuple(self.sun_color)
            self.prog_sky_view['u_atmo_tint'].value = tuple(self.atmo_tint)
            self.prog_sky_view['u_planet_radius'].value = float(self.planet_radius)
            self.prog_sky_view['u_atmo_radius'].value = float(self.atmo_radius)
            self.prog_sky_view['u_scale_height'].value = float(self.scale_height)
            self.prog_sky_view['u_density'].value = float(self.atmo_density)

            self.quad_vao_lut.render(moderngl.TRIANGLE_STRIP)

        # =========================================================
        # Pass 2: Render Planet & Rings to Scene FBO
        # =========================================================
        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.012, 0.012, 0.018, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        proj_bytes = proj.tobytes()
        view_bytes = view.tobytes()
        model_bytes = model.tobytes()

        # 2a. Planet Sphere
        self.ring_tex.use(location=0)
        self.prog_sphere['u_proj'].write(proj_bytes)
        self.prog_sphere['u_view'].write(view_bytes)
        self.prog_sphere['u_model'].write(model_bytes)
        self.prog_sphere['u_sun_dir'].value = tuple(sun_dir)
        self.prog_sphere['u_sun_color'].value = tuple(self.sun_color)
        self.prog_sphere['u_albedo'].value = tuple(self.planet_albedo)
        self.prog_sphere['u_ambient'].value = self.ambient
        self.prog_sphere['u_ring_tex'].value = 0
        self.prog_sphere['u_ring_inner'].value = self.ring_inner
        self.prog_sphere['u_ring_outer'].value = self.ring_outer
        self.prog_sphere['u_sun_angular_radius'].value = sun_ang_rad
        self.prog_sphere['u_penumbra_samples'].value = int(self.penumbra_samples)

        self.sphere_vao.render(moderngl.TRIANGLES)

        # 2b. Rings (Double-sided with alpha blending, depth-write disabled for transparency)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.depth_mask = False

        self.prog_ring['u_proj'].write(proj_bytes)
        self.prog_ring['u_view'].write(view_bytes)
        self.prog_ring['u_model'].write(model_bytes)
        self.prog_ring['u_sun_dir'].value = tuple(sun_dir)
        self.prog_ring['u_sun_color'].value = tuple(self.sun_color)
        self.prog_ring['u_ring_tex'].value = 0
        self.prog_ring['u_planet_radius'].value = float(self.planet_radius)
        self.prog_ring['u_sun_angular_radius'].value = sun_ang_rad

        self.ring_vao.render(moderngl.TRIANGLE_STRIP, vertices=self.ring_vertex_count)
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)

        # =========================================================
        # Pass 3: Composite Atmosphere over Scene onto Screen (O(1))
        # =========================================================
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        if self.atmo_enabled:
            inv_proj = pyrr.matrix44.inverse(proj)
            inv_view = pyrr.matrix44.inverse(view)

            self.scene_color_tex.use(location=0)
            self.scene_depth_tex.use(location=1)
            self.sky_view_tex.use(location=2)
            self.ring_tex.use(location=3)

            if 'u_scene_color' in self.prog_atmo_comp: self.prog_atmo_comp['u_scene_color'].value = 0
            if 'u_scene_depth' in self.prog_atmo_comp: self.prog_atmo_comp['u_scene_depth'].value = 1
            if 'u_sky_view_lut' in self.prog_atmo_comp: self.prog_atmo_comp['u_sky_view_lut'].value = 2
            if 'u_ring_tex' in self.prog_atmo_comp: self.prog_atmo_comp['u_ring_tex'].value = 3
            if 'u_ring_inner' in self.prog_atmo_comp: self.prog_atmo_comp['u_ring_inner'].value = float(self.ring_inner)
            if 'u_ring_outer' in self.prog_atmo_comp: self.prog_atmo_comp['u_ring_outer'].value = float(self.ring_outer)
            if 'u_ring_tex_res' in self.prog_atmo_comp: self.prog_atmo_comp['u_ring_tex_res'].value = float(self.ring_tex_res)
            if 'u_analytical_slicing' in self.prog_atmo_comp: self.prog_atmo_comp['u_analytical_slicing'].value = bool(self.analytical_slicing)

            self.prog_atmo_comp['u_inv_proj'].write(inv_proj.tobytes())
            self.prog_atmo_comp['u_inv_view'].write(inv_view.tobytes())
            self.prog_atmo_comp['u_cam_pos'].value = tuple(eye_pos)
            self.prog_atmo_comp['u_planet_radius'].value = float(self.planet_radius)
            self.prog_atmo_comp['u_atmo_radius'].value = float(self.atmo_radius)
            if 'u_sun_dir' in self.prog_atmo_comp: self.prog_atmo_comp['u_sun_dir'].value = tuple(sun_dir)
            if 'u_sun_color' in self.prog_atmo_comp: self.prog_atmo_comp['u_sun_color'].value = tuple(self.sun_color)
            if 'u_atmo_tint' in self.prog_atmo_comp: self.prog_atmo_comp['u_atmo_tint'].value = tuple(self.atmo_tint)
            if 'u_scale_height' in self.prog_atmo_comp: self.prog_atmo_comp['u_scale_height'].value = float(self.scale_height)
            if 'u_density' in self.prog_atmo_comp: self.prog_atmo_comp['u_density'].value = float(self.atmo_density)
            if 'u_volumetric_shadow' in self.prog_atmo_comp: self.prog_atmo_comp['u_volumetric_shadow'].value = bool(self.volumetric_shadow)

            self.quad_vao_comp.render(moderngl.TRIANGLE_STRIP)
        else:
            # Fallback blit if atmosphere is disabled
            self.ctx.copy_framebuffer(self.ctx.screen, self.scene_fbo)

        # =========================================================
        # Pass 4: ImGui Controls & HUD
        # =========================================================
        if render_gui:
            imgui.new_frame()
            imgui.set_next_window_position(15, 15, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(450, 720, imgui.FIRST_USE_EVER)
            imgui.begin("Shadow & Atmosphere Controls", True)

            imgui.text("Atmosphere Prototype: Free Fly & Inside Atmosphere")
            imgui.separator()

            # Camera & Navigation Controls
            expanded_cam, _ = imgui.collapsing_header("Camera & Free Fly Controls", flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if expanded_cam:
                is_orbit = (self.camera.mode == "orbit")
                is_free = (self.camera.mode == "freefly")

                if imgui.radio_button("Orbit Mode", is_orbit) and not is_orbit:
                    self.camera.set_mode("orbit")
                imgui.same_line()
                if imgui.radio_button("Free Fly Mode [F/Tab]", is_free) and not is_free:
                    self.camera.set_mode("freefly")

                alt = self.camera.get_altitude()
                atmo_depth = self.atmo_radius - self.planet_radius
                if alt < atmo_depth:
                    pct = (alt / atmo_depth) * 100.0
                    imgui.text_colored(f"Altitude: {alt:.4f} ({pct:.1f}% inside atmosphere)", 0.35, 0.8, 1.0)
                else:
                    imgui.text(f"Altitude: {alt:.4f} (Outer Space)")

                if self.camera.mode == "freefly":
                    _, self.camera.move_speed = imgui.slider_float("Flight Speed", self.camera.move_speed, 0.1, 4.0, "%.2f")
                    imgui.text_disabled("WASD: Fly | Space/C: Up/Down | Shift: 4x Boost")
                    imgui.text_disabled("Right-Click / Drag: Look around | Scroll: Speed")

                imgui.text("Teleport Presets:")
                if imgui.button("Space Orbit"):
                    self.camera.preset_orbit()
                imgui.same_line()
                if imgui.button("Cloud Tops"):
                    self.camera.preset_cloud_surfing()
                imgui.same_line()
                if imgui.button("Mid Atmosphere"):
                    self.camera.preset_mid_atmo()

                if imgui.button("Surface: Daylight Rings"):
                    self.camera.preset_surface_daylight()
                imgui.same_line()
                if imgui.button("Surface: Ring Eclipse"):
                    self.camera.preset_surface_eclipse()
                imgui.same_line()
                if imgui.button("Sunset"):
                    self.camera.preset_surface_sunset()

            # Sun & Lighting
            expanded_sun, _ = imgui.collapsing_header("Sun & Shadow Settings", flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if expanded_sun:
                _, self.sun_azimuth = imgui.slider_float("Sun Azimuth", self.sun_azimuth, 0.0, 360.0, "%.1f deg")
                _, self.sun_elevation = imgui.slider_float("Sun Elevation", self.sun_elevation, -85.0, 85.0, "%.1f deg")
                _, self.sun_angular_deg = imgui.slider_float("Sun Angular Size", self.sun_angular_deg, 0.01, 3.0, "%.2f deg")
                _, self.penumbra_samples = imgui.slider_int("Penumbra Taps", self.penumbra_samples, 1, 21)
                _, self.ambient = imgui.slider_float("Ambient Light", self.ambient, 0.0, 0.2, "%.2f")
                _, self.planet_albedo = imgui.color_edit3("Planet Color", *self.planet_albedo)

            # Atmosphere & Volumetric Shadow
            expanded_atmo, _ = imgui.collapsing_header("Atmosphere & Volumetric Shadow", flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if expanded_atmo:
                _, self.atmo_enabled = imgui.checkbox("Enable Atmosphere", self.atmo_enabled)
                _, self.volumetric_shadow = imgui.checkbox("Volumetric Ring Shadow (Phase 3)", self.volumetric_shadow)
                if self.volumetric_shadow:
                    imgui.indent()
                    _, self.analytical_slicing = imgui.checkbox("Analytical Slicing (No Steps, O(1))", self.analytical_slicing)
                    imgui.unindent()
                _, self.atmo_radius = imgui.slider_float("Atmo Radius", self.atmo_radius, 1.01, 1.30, "%.3f")
                _, self.scale_height = imgui.slider_float("Scale Height H", self.scale_height, 0.005, 0.08, "%.3f")
                _, self.atmo_density = imgui.slider_float("Scattering Density", self.atmo_density, 1.0, 80.0, "%.1f")
                _, self.atmo_tint = imgui.color_edit3("Atmosphere Tint", *self.atmo_tint)
                _, self.show_lut_preview = imgui.checkbox("Preview 256x256 LUT", self.show_lut_preview)

            imgui.separator()
            if imgui.button("Reset Orbit"):
                self.camera.preset_orbit()
            imgui.same_line()
            if imgui.button("Edge-on Sun"):
                self.sun_elevation = 0.0

            imgui.end()

            # Picture-in-picture LUT preview window
            if self.show_lut_preview and self.atmo_enabled:
                imgui.set_next_window_position(self.width - 280, 15, imgui.ONCE)
                imgui.set_next_window_size(265, 290, imgui.ONCE)
                imgui.begin("Sky-View LUT (256x256)", True)
                imgui.image(self.sky_view_tex.glo, 240, 240)
                imgui.text("X: Azimuth [0, 2pi], Y: Elevation")
                imgui.end()

            imgui.render()
            self.imgui_renderer.render(imgui.get_draw_data())

    def run(self):
        last_time = glfw.get_time()
        while not glfw.window_should_close(self.window):
            current_time = glfw.get_time()
            dt = min(0.1, max(1e-4, current_time - last_time))
            last_time = current_time

            glfw.poll_events()
            self.imgui_renderer.process_inputs()
            self.handle_movement(dt)
            self.render()
            glfw.swap_buffers(self.window)

        try:
            self.imgui_renderer.shutdown()
        except Exception:
            pass
        glfw.terminate()


if __name__ == "__main__":
    app = App()
    app.run()
