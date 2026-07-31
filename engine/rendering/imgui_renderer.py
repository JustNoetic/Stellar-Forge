import ctypes
import numpy as np
import moderngl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl

class ModernGLImGuiRenderer(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self.io = imgui.get_io()
        self.textures = {}
        
        self.prog = ctx.program(
            vertex_shader="""
                #version 330 core
                uniform mat4 ProjMtx;
                in vec2 Position;
                in vec2 UV;
                in vec4 Color;
                out vec2 Frag_UV;
                out vec4 Frag_Color;
                void main() {
                    Frag_UV = UV;
                    Frag_Color = Color;
                    gl_Position = ProjMtx * vec4(Position.xy, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330 core
                uniform sampler2D Texture;
                in vec2 Frag_UV;
                in vec4 Frag_Color;
                out vec4 Out_Color;
                void main() {
                    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);
                }
            """
        )
        self.proj_mtx_uniform = self.prog['ProjMtx']
        self.texture_uniform = self.prog['Texture']
        self.texture_uniform.value = 0
        
        self.vbo = ctx.buffer(reserve=1024 * 1024)
        self.ibo = ctx.buffer(reserve=1024 * 1024)
        
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f 4f1', 'Position', 'UV', 'Color')],
            index_buffer=self.ibo
        )
        
        self.font_texture = None
        self.refresh_font_texture()
        
    def refresh_font_texture(self):
        width, height, pixels = self.io.fonts.get_tex_data_as_rgba32()
        if self.font_texture:
            if self.font_texture.glo in self.textures:
                del self.textures[self.font_texture.glo]
            self.font_texture.release()
        self.font_texture = self.ctx.texture((width, height), 4, data=pixels)
        self.font_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.io.fonts.texture_id = self.font_texture.glo
        self.textures[self.font_texture.glo] = self.font_texture
        self.io.fonts.clear_tex_data()
        
    def render(self, draw_data):
        io = self.io
        display_width, display_height = io.display_size
        fb_width = int(display_width * io.display_fb_scale[0])
        fb_height = int(display_height * io.display_fb_scale[1])
        if fb_width == 0 or fb_height == 0:
            return
            
        draw_data.scale_clip_rects(*io.display_fb_scale)
        
        ortho_projection = np.array([
             [ 2.0/display_width,  0.0,                   0.0, 0.0],
             [ 0.0,                2.0/-display_height,   0.0, 0.0],
             [ 0.0,                0.0,                  -1.0, 0.0],
             [-1.0,                1.0,                   0.0, 1.0]
        ], dtype='f4')
        self.proj_mtx_uniform.write(ortho_projection.tobytes())
        
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.depth_mask = False
        
        self.ctx.viewport = (0, 0, fb_width, fb_height)
        
        for commands in draw_data.commands_lists:
            vtx_bytes_size = commands.vtx_buffer_size * imgui.VERTEX_SIZE
            idx_bytes_size = commands.idx_buffer_size * imgui.INDEX_SIZE
            
            vtx_data = ctypes.string_at(commands.vtx_buffer_data, vtx_bytes_size)
            idx_data = ctypes.string_at(commands.idx_buffer_data, idx_bytes_size)
            
            if self.vbo.size < vtx_bytes_size:
                self.vbo.orphan(vtx_bytes_size)
            self.vbo.write(vtx_data)
            
            if self.ibo.size < idx_bytes_size:
                self.ibo.orphan(idx_bytes_size)
            self.ibo.write(idx_data)
            
            idx_buffer_offset = 0
            for command in commands.commands:
                texture = self.textures.get(command.texture_id)
                if texture:
                    texture.use(location=0)
                else:
                    gl.glBindTexture(gl.GL_TEXTURE_2D, command.texture_id)
                
                x, y, z, w = command.clip_rect
                self.ctx.scissor = (int(x), int(fb_height - w), int(z - x), int(w - y))
                
                self.vao.render(moderngl.TRIANGLES, vertices=command.elem_count, first=idx_buffer_offset // imgui.INDEX_SIZE)
                idx_buffer_offset += command.elem_count * imgui.INDEX_SIZE
                
        self.ctx.scissor = None
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)
        
    def shutdown(self):
        if self.vao:
            self.vao.release()
        if self.vbo:
            self.vbo.release()
        if self.ibo:
            self.ibo.release()
        if self.prog:
            self.prog.release()
        if self.font_texture:
            self.font_texture.release()

class ModernGLGlfwRenderer(GlfwRenderer):
    def __init__(self, window, ctx, attach_callbacks=True):
        self.modern_renderer = ModernGLImGuiRenderer(ctx)
        super().__init__(window, attach_callbacks)
        # Base class init calls refresh_font_texture which registers our ModernGL font texture.
        # But _invalidate_device_objects resets io.fonts.texture_id to 0. We must restore it.
        font_id = self.modern_renderer.font_texture.glo if self.modern_renderer.font_texture else 0
        self._invalidate_device_objects()
        self.io.fonts.texture_id = font_id
        
    def refresh_font_texture(self):
        self.modern_renderer.refresh_font_texture()
        self._font_texture = 0
        
    def render(self, draw_data):
        self.modern_renderer.render(draw_data)
        
    def shutdown(self):
        self.modern_renderer.shutdown()
