"""Conservative radial opacity bounds; averaging mipmaps cannot reject gaps."""
import moderngl
import numpy as np


class RingOpacityBounds:
    def __init__(self, ctx, width=4096, rows=16):
        if width < 1 or width & (width - 1):
            raise ValueError("Ring opacity width must be a power of two")
        self.width = width
        self.rows = rows
        self.texture = ctx.texture((2 * width, rows), 1, dtype="f4")
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.repeat_x = self.texture.repeat_y = False

    def update(self, atlas):
        alpha = np.asarray(atlas, dtype="f4")[:, :, 3]
        if alpha.shape != (self.rows, self.width):
            raise ValueError("Ring opacity atlas has the wrong dimensions")
        tree = np.zeros((self.rows, 2 * self.width), dtype="f4")
        tree[:, self.width:] = np.maximum(alpha, 0.0)
        start = self.width
        while start > 1:
            end = start * 2
            tree[:, start // 2:start] = np.maximum(tree[:, start:end:2], tree[:, start + 1:end:2])
            start //= 2
        self.texture.write(tree.tobytes())

    def bind(self, program):
        self.texture.use(location=22)
        program["u_ring_opacity_bounds"].value = 22
        program["u_ring_opacity_bounds_enabled"].value = True

    def release(self):
        self.texture.release()
