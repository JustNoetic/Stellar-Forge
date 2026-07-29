import os
import shutil
import numpy as np
from PIL import Image

# Target Astronomical Geometric Albedo (p_V) or Surface Reflectance Targets
TARGET_ALBEDOS = {
    'Moon': {'target_pv': 0.120, 'mode': 'scale'},
    'Saturn': {'target_pv': 0.470, 'mode': 'saturn_pedestal'},
    'Jupiter': {'target_pv': 0.520, 'mode': 'scale'},
    'Enceladus': {'target_pv': 0.980, 'mode': 'max_bright_hdr_prep'}, # Boost to highest non-clipping bright ice (peak 1.0 linear)
    'Earth': {'target_pv': 0.130, 'mode': 'scale'}, # Surface ground (land + ocean)
    'Mercury': {'target_pv': 0.138, 'mode': 'scale'},
    'Venus': {'target_pv': 0.689, 'mode': 'scale'},
    'Mars': {'target_pv': 0.170, 'mode': 'scale'},
    'Titan': {'target_pv': 0.220, 'mode': 'scale'},
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
        print(f"Skipping {name}: file not found at {img_path}")
        return

    # 1. Create backup if not already present
    ext = os.path.splitext(img_path)[1]
    backup_path = os.path.splitext(img_path)[0] + f"_orig_backup{ext}"
    if not os.path.exists(backup_path):
        shutil.copy2(img_path, backup_path)
        print(f"[{name}] Created backup: {backup_path}")
    else:
        # Load from backup to avoid compounding edits
        img_path = backup_path

    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr_lin = arr ** 2.2

    pv_orig, lum_orig = calc_disk_pv(arr_lin)
    target_pv = target_info['target_pv']
    mode = target_info['mode']

    if mode == 'saturn_pedestal':
        # Saturn has elevated pedestal (min sRGB ~126 / 255 -> min linear ~0.23)
        # 1. Remove pedestal offset to stretch contrast range
        min_val = arr_lin.min()
        arr_stretched = (arr_lin - min_val) / (1.0 - min_val)
        pv_stretched, _ = calc_disk_pv(arr_stretched)
        
        # 2. Scale stretched texture to hit target p_V
        scale = target_pv / pv_stretched
        arr_final_lin = np.clip(arr_stretched * scale, 0.0, 1.0)
        
    elif mode == 'max_bright_hdr_prep':
        # Enceladus: boost texture as bright as possible without clipping highlights (max linear ~0.98)
        max_lin = arr_lin.max()
        scale = 0.98 / max_lin
        arr_final_lin = np.clip(arr_lin * scale, 0.0, 1.0)

    else: # standard 'scale' mode
        scale = target_pv / pv_orig
        arr_final_lin = np.clip(arr_lin * scale, 0.0, 1.0)

    pv_final, lum_final = calc_disk_pv(arr_final_lin)

    # Convert back to sRGB
    arr_srgb = np.round((arr_final_lin ** (1.0 / 2.2)) * 255.0).astype(np.uint8)
    
    # Save back to main texture file
    out_target = os.path.splitext(backup_path)[0].replace("_orig_backup", "") + ext
    result_img = Image.fromarray(arr_srgb, mode='RGB')
    result_img.save(out_target, quality=95)

    print(f"[{name}] Calibrated:")
    print(f"   Original p_V : {pv_orig:.3f} | Mean Lin Lum: {lum_orig:.3f}")
    print(f"   Target p_V   : {target_pv:.3f}")
    print(f"   Calibrated p_V: {pv_final:.3f} | Mean Lin Lum: {lum_final:.3f}")
    print(f"   Saved to     : {out_target}\n")

def main():
    textures_dir = 'textures'
    for name, target_info in TARGET_ALBEDOS.items():
        for ext in ['.jpg', '.png']:
            path = os.path.join(textures_dir, name, f"{name}{ext}")
            if os.path.exists(path) or os.path.exists(os.path.join(textures_dir, name, f"{name}_orig_backup{ext}")):
                process_texture(name, path, target_info)
                break

if __name__ == '__main__':
    main()
