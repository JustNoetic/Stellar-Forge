import os
import shutil
import numpy as np
from PIL import Image

# Target Astronomical Geometric Albedo (p_V) or Surface Reflectance Targets
# Pure uniform exposure scaling (no curve adjustments or contrast/black-level changes)
TARGET_ALBEDOS = {
    'Moon': {'target_pv': 0.120},
    'Saturn': {'target_pv': 0.470},
    'Jupiter': {'target_pv': 0.520},
    'Enceladus': {'target_pv': 0.980}, # Scale up uniformly to max non-clipping white
    'Earth': {'target_pv': 0.130},     # Surface ground (land + ocean)
    'Mercury': {'target_pv': 0.138},
    'Venus': {'target_pv': 0.689},
    'Mars': {'target_pv': 0.170},
    'Titan': {'target_pv': 0.220},
}

def calc_disk_pv(arr_lin):
    """Calculates disk-integrated geometric albedo (Lambertian sphere assumption)."""
    Y_lin = 0.2126 * arr_lin[:,:,0] + 0.7152 * arr_lin[:,:,1] + 0.0722 * arr_lin[:,:,2]
    H, W = Y_lin.shape
    lats = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    lons = ((np.arange(W) + 0.5) / W - 0.5) * 2.0 * np.pi
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    mask = np.abs(lon_grid) <= (np.pi / 2.0)
    mu = np.cos(lat_grid) * np.cos(lon_grid)
    weights = np.where(mask, mu * np.cos(lat_grid), 0.0)
    
    p_V = (np.sum(Y_lin * weights) / np.sum(weights)) * (2.0 / 3.0)
    mean_lum = Y_lin.mean()
    return p_V, mean_lum

def process_texture(name, img_path, target_info):
    if not os.path.exists(img_path):
        # Check if backup exists
        ext = os.path.splitext(img_path)[1]
        backup_path = os.path.splitext(img_path)[0] + f"_orig_backup{ext}"
        if not os.path.exists(backup_path):
            print(f"Skipping {name}: file not found at {img_path}")
            return
        img_path = backup_path

    ext = os.path.splitext(img_path)[1]
    base_name = os.path.splitext(img_path)[0].replace("_orig_backup", "")
    backup_path = base_name + f"_orig_backup{ext}"
    out_target = base_name + ext

    # 1. Ensure backup exists from original
    if not os.path.exists(backup_path):
        shutil.copy2(img_path, backup_path)
        print(f"[{name}] Created backup: {backup_path}")
        source_path = img_path
    else:
        # ALWAYS restore from original backup to avoid compound editing
        source_path = backup_path

    img = Image.open(source_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr_lin = arr ** 2.2

    pv_orig, lum_orig = calc_disk_pv(arr_lin)
    target_pv = target_info['target_pv']

    # Pure uniform linear exposure scale factor (no curves/contrast shifts)
    if name == 'Enceladus':
        # Pure scale up to maximum non-clipping white
        max_lin = arr_lin.max()
        scale = min(target_pv / pv_orig, 0.98 / max_lin)
    else:
        scale = target_pv / pv_orig

    arr_final_lin = np.clip(arr_lin * scale, 0.0, 1.0)
    pv_final, lum_final = calc_disk_pv(arr_final_lin)

    # Convert back to sRGB using pure gamma encoding
    arr_srgb = np.round((arr_final_lin ** (1.0 / 2.2)) * 255.0).astype(np.uint8)
    
    # Save back to main texture file
    result_img = Image.fromarray(arr_srgb, mode='RGB')
    result_img.save(out_target, quality=95)

    ev_change = np.log2(scale)
    print(f"[{name}] Pure Exposure Calibrated:")
    print(f"   Original p_V : {pv_orig:.3f} | Original Mean Lin Lum: {lum_orig:.3f}")
    print(f"   Linear Scale : {scale:.3f} (EV change: {ev_change:+.2f} EV)")
    print(f"   Calibrated p_V: {pv_final:.3f} | Final Mean Lin Lum: {lum_final:.3f}")
    print(f"   Saved to     : {out_target}\n")

def main():
    textures_dir = 'textures'
    for name, target_info in TARGET_ALBEDOS.items():
        for ext in ['.jpg', '.png']:
            path = os.path.join(textures_dir, name, f"{name}{ext}")
            backup = os.path.join(textures_dir, name, f"{name}_orig_backup{ext}")
            if os.path.exists(path) or os.path.exists(backup):
                process_texture(name, path, target_info)
                break

if __name__ == '__main__':
    main()
