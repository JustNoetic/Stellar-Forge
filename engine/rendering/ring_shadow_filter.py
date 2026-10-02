"""Cached radial integrals of oblique ring occlusion for surface and Mode 2 atmospheric penumbrae.

The unified atlas already includes segment opacity. Integrating extinction,
rather than alpha, preserves transparent gaps even under shallow illumination.
Only profile edits rebuild this table; camera and light movement just sample it.
"""
import moderngl
import numpy as np


class RingShadowFilter:
    UNIT = 13
    ANGLES = 64
    MIN_MU = 1e-4

    def __init__(self, ctx, width=4096, rows=4):
        self.width = width
        self.rows = rows
        self.profiles = np.full((rows, width), np.nan, dtype=np.float32)
        self.texture = ctx.texture_array((width + 1, self.ANGLES, rows), 1,
                                         dtype='f4')
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.mu = np.geomspace(1., self.MIN_MU, self.ANGLES)[:, None]
        # Four subcell samples integrate the atlas's linear reconstruction,
        # including the narrow transitions between opaque bands and gaps.
        positions = (np.arange(width * 4) + .5) / 4 - .5
        self.left = np.clip(np.floor(positions).astype(int), 0, width - 1)
        self.right = np.clip(np.floor(positions).astype(int) + 1, 0, width - 1)
        self.fraction = positions - np.floor(positions)

    def update(self, atlas):
        """Upload changed unified rows and bind the table. Return row count."""
        changed = 0
        for row in range(self.rows):
            alpha = np.clip(atlas[row, :, 3], 0., 1.)
            if np.array_equal(alpha, self.profiles[row]):
                continue
            integral = np.zeros((self.ANGLES, self.width + 1), dtype='f4')
            if np.any(alpha):
                reconstructed = (alpha[self.left] * (1 - self.fraction)
                                 + alpha[self.right] * self.fraction)
                log_transmission = np.log(np.maximum(1e-6, 1 - reconstructed))
                occlusion = -np.expm1(log_transmission[None, :] / self.mu)
                cells = occlusion.reshape(self.ANGLES, self.width, 4).mean(axis=2)
                integral[:, 1:] = np.cumsum(cells, axis=1) / self.width
            self.texture.write(integral.tobytes(),
                               viewport=(0, 0, row, self.width + 1, self.ANGLES, 1))
            self.profiles[row] = alpha
            changed += 1
        self.bind()
        return changed

    def bind(self):
        # Atmosphere unit 13 uses a 2D texture; this array has a separate binding.
        self.texture.use(location=self.UNIT)

    @classmethod
    def configure(cls, program):
        if 'u_ring_shadow_integral' in program:
            program['u_ring_shadow_integral'].value = cls.UNIT

    def release(self):
        self.texture.release()
