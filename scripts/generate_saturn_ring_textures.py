import os
import numpy as np
from PIL import Image

def generate_saturn_ring_textures():
    saturn_dir = r"d:\Files\Coding\OpenGL\Stellar-Forge\textures\Saturn"
    
    # Load 5 data files
    backscatter_path = os.path.join(saturn_dir, "backscattered.txt")
    forwardscatter_path = os.path.join(saturn_dir, "forwardscattered.txt")
    transparency_path = os.path.join(saturn_dir, "transparency.txt")
    color_path = os.path.join(saturn_dir, "sat_rings_color.txt")
    unlit_path = os.path.join(saturn_dir, "unlitside.txt")
    
    print("Loading data files...")
    backscatter = np.loadtxt(backscatter_path, dtype=np.float32)
    forwardscatter = np.loadtxt(forwardscatter_path, dtype=np.float32)
    transparency = np.loadtxt(transparency_path, dtype=np.float32)
    color = np.loadtxt(color_path, dtype=np.float32) # shape (13177, 3)
    unlit = np.loadtxt(unlit_path, dtype=np.float32)
    
    n_samples = len(backscatter)
    print(f"Loaded {n_samples} data points.")
    
    # Calculate linear color values
    # Frontlit (backscatter geometry):
    front_rgb_linear = backscatter[:, np.newaxis] * color
    
    # Physical Opacity (Alpha) = 1 - Transparency
    front_alpha_linear = np.clip(1.0 - transparency, 0.0, 1.0)
    
    # Unlit side:
    back_rgb_linear = unlit[:, np.newaxis] * color
    
    # Forward scatter luminance:
    # forwardscatter is relative brightness in forward scattering
    back_alpha_linear = np.clip(forwardscatter, 0.0, 1.0)
    
    # Gamma encode RGB channels (since shader applies pow(rgb, 2.2))
    gamma = 2.2
    front_rgb_srgb = np.power(np.clip(front_rgb_linear, 0.0, 1.0), 1.0 / gamma)
    back_rgb_srgb = np.power(np.clip(back_rgb_linear, 0.0, 1.0), 1.0 / gamma)
    
    # Alpha channels:
    # Front alpha is physical opacity (linear opacity used directly for optical depth tau = -log(1 - alpha))
    front_alpha_srgb = front_alpha_linear
    
    # Back alpha is forward scatter luminance which shader decodes via pow(a, 2.2)
    back_alpha_srgb = np.power(back_alpha_linear, 1.0 / gamma)
    
    # Stack into RGBA arrays (shape 1, 13177, 4)
    front_rgba = np.zeros((1, n_samples, 4), dtype=np.float32)
    front_rgba[0, :, 0:3] = front_rgb_srgb
    front_rgba[0, :, 3] = front_alpha_srgb
    
    back_rgba = np.zeros((1, n_samples, 4), dtype=np.float32)
    back_rgba[0, :, 0:3] = back_rgb_srgb
    back_rgba[0, :, 3] = back_alpha_srgb
    
    # Convert float [0, 1] to uint8 [0, 255]
    front_uint8 = np.clip(front_rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    back_uint8 = np.clip(back_rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    
    # Create PIL Images
    img_front_raw = Image.fromarray(front_uint8, mode='RGBA')
    img_back_raw = Image.fromarray(back_uint8, mode='RGBA')
    
    # Resample to target resolution 8192x1
    target_width = 8192
    img_front_8k = img_front_raw.resize((target_width, 1), Image.Resampling.LANCZOS)
    img_back_8k = img_back_raw.resize((target_width, 1), Image.Resampling.LANCZOS)
    
    # Also create 8192x64 expanded preview images for easy visual inspection
    img_front_preview = img_front_raw.resize((target_width, 64), Image.Resampling.LANCZOS)
    img_back_preview = img_back_raw.resize((target_width, 64), Image.Resampling.LANCZOS)
    
    # Save main ring textures
    out_front_path = os.path.join(saturn_dir, "Saturn_ring_front.png")
    out_back_path = os.path.join(saturn_dir, "Saturn_ring_back.png")
    
    img_front_8k.save(out_front_path)
    img_back_8k.save(out_back_path)
    print(f"Saved {out_front_path} ({target_width}x1)")
    print(f"Saved {out_back_path} ({target_width}x1)")
    
    # Save preview versions
    preview_front_path = os.path.join(saturn_dir, "Saturn_ring_front_preview.png")
    preview_back_path = os.path.join(saturn_dir, "Saturn_ring_back_preview.png")
    img_front_preview.save(preview_front_path)
    img_back_preview.save(preview_back_path)
    print(f"Saved preview images: {preview_front_path}, {preview_back_path}")

if __name__ == "__main__":
    generate_saturn_ring_textures()
