import os
import numpy as np
from PIL import Image

saturn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'textures', 'Saturn')

back_path = os.path.join(saturn_dir, 'backscattered.txt')
fwd_path = os.path.join(saturn_dir, 'forwardscattered.txt')
unlit_path = os.path.join(saturn_dir, 'unlitside.txt')
trans_path = os.path.join(saturn_dir, 'transparency.txt')
color_path = os.path.join(saturn_dir, 'sat_rings_color.txt')

print("Reading text files...")
backscattered = np.loadtxt(back_path, dtype=np.float32)
forwardscattered = np.loadtxt(fwd_path, dtype=np.float32)
unlit = np.loadtxt(unlit_path, dtype=np.float32)
transparency = np.loadtxt(trans_path, dtype=np.float32)
colors = np.loadtxt(color_path, dtype=np.float32)

N = len(backscattered)
print(f"Loaded {N} samples.")

# Opacity (Alpha): 1.0 - transparency
opacity = np.clip(1.0 - transparency, 0.0, 1.0)

# Unlit scale factor: unlitside.txt is normalized independently to 1.0 by Jónsson, 
# but physically unlit transmitted flux is ~22% of opposition lit-side flux.
UNLIT_SCALE_FACTOR = 0.22

# Linear colors * brightness profiles
front_linear = np.clip(colors * backscattered[:, np.newaxis], 0.0, 1.0)
fwd_linear = np.clip(colors * forwardscattered[:, np.newaxis], 0.0, 1.0)
back_linear = np.clip(colors * (unlit * UNLIT_SCALE_FACTOR)[:, np.newaxis], 0.0, 1.0)

# Standard sRGB Gamma Conversion (Linear -> sRGB) so shader's pow(tex, 2.2) decodes to exact Linear
def linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)

front_srgb = linear_to_srgb(front_linear)
fwd_srgb = linear_to_srgb(fwd_linear)
back_srgb = linear_to_srgb(back_linear)

# Forward scatter luminance intensity for phase-dependent lit rendering
fwd_luminance = np.clip(np.dot(fwd_srgb, [0.2126, 0.7152, 0.0722]), 0.0, 1.0)

front_rgba = np.zeros((N, 4), dtype=np.uint8)
front_rgba[:, 0:3] = (front_srgb * 255.0).astype(np.uint8)
front_rgba[:, 3] = (opacity * 255.0).astype(np.uint8)

back_rgba = np.zeros((N, 4), dtype=np.uint8)
back_rgba[:, 0:3] = (back_srgb * 255.0).astype(np.uint8)
back_rgba[:, 3] = (fwd_luminance * 255.0).astype(np.uint8)

# Reshape to 1xN Image (1D texture profile across width)
front_img = Image.fromarray(front_rgba.reshape(1, N, 4), mode='RGBA')
back_img = Image.fromarray(back_rgba.reshape(1, N, 4), mode='RGBA')

out_front = os.path.join(saturn_dir, 'Saturn_ring_front.png')
out_back = os.path.join(saturn_dir, 'Saturn_ring_back.png')

front_img.save(out_front)
back_img.save(out_back)

print(f"Saved {out_front} ({N}x1) with sRGB gamma encoding (Alpha = Opacity)")
print(f"Saved {out_back} ({N}x1) with unlit scale factor {UNLIT_SCALE_FACTOR} and sRGB gamma encoding (Alpha = Forward Scatter Luminance)")

