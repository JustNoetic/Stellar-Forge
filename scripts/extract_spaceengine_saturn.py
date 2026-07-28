import os
import zipfile
import numpy as np
from PIL import Image

def extract_saturn_equirectangular(out_path, width=4096, height=2048):
    pak_path = r"E:\SteamLibrary\steamapps\common\SpaceEngine\data\textures\planets\Saturn-Surface-SSS.pak"
    if not os.path.exists(pak_path):
        print(f"Error: PAK file not found at {pak_path}")
        return False

    print(f"Opening Space Engine PAK archive: {pak_path}")
    z = zipfile.ZipFile(pak_path)

    face_names = ['pos_x', 'neg_x', 'pos_y', 'neg_y', 'pos_z', 'neg_z']
    faces = {}

    print("Extracting and stitching 6 cubemap face tiles (1024x1024 each)...")
    for fn in face_names:
        face_img = Image.new('RGB', (1024, 1024))
        for r in range(4):
            for c in range(4):
                tile_path = f'Saturn/Surface-SSS/{fn}/2_{r}_{c}.jpg'
                tile = Image.open(z.open(tile_path))
                face_img.paste(tile, (c * 256, r * 256))
        faces[fn] = np.array(face_img, dtype=np.uint8)

    print(f"Projecting cubemap into {width}x{height} equirectangular texture...")
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    u_grid, v_grid = np.meshgrid(u, v)

    lon = (u_grid - 0.5) * 2.0 * np.pi
    lat = (0.5 - v_grid) * np.pi

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z_dir = np.cos(lat) * np.cos(lon)

    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z_dir)

    out_img = np.zeros((height, width, 3), dtype=np.uint8)
    N = 1024

    # Major axis X
    mask_px = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
    if np.any(mask_px):
        uc = -z_dir[mask_px] / abs_x[mask_px]
        vc = -y[mask_px] / abs_x[mask_px]
        px = np.clip(((uc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        py = np.clip(((vc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        out_img[mask_px] = faces['pos_x'][py, px]

    mask_nx = (abs_x >= abs_y) & (abs_x >= abs_z) & (x <= 0)
    if np.any(mask_nx):
        uc = z_dir[mask_nx] / abs_x[mask_nx]
        vc = -y[mask_nx] / abs_x[mask_nx]
        px = np.clip(((uc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        py = np.clip(((vc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        out_img[mask_nx] = faces['neg_x'][py, px]

    # Major axis Y
    mask_py = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
    if np.any(mask_py):
        uc = x[mask_py] / abs_y[mask_py]
        vc = z_dir[mask_py] / abs_y[mask_py]
        px = np.clip(((uc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        py = np.clip(((vc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        out_img[mask_py] = faces['pos_y'][py, px]

    mask_ny = (abs_y >= abs_x) & (abs_y >= abs_z) & (y <= 0)
    if np.any(mask_ny):
        uc = x[mask_ny] / abs_y[mask_ny]
        vc = -z_dir[mask_ny] / abs_y[mask_ny]
        px = np.clip(((uc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        py = np.clip(((vc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        out_img[mask_ny] = faces['neg_y'][py, px]

    # Major axis Z
    mask_pz = (abs_z >= abs_x) & (abs_z >= abs_y) & (z_dir > 0)
    if np.any(mask_pz):
        uc = x[mask_pz] / abs_z[mask_pz]
        vc = -y[mask_pz] / abs_z[mask_pz]
        px = np.clip(((uc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        py = np.clip(((vc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        out_img[mask_pz] = faces['pos_z'][py, px]

    mask_nz = (abs_z >= abs_x) & (abs_z >= abs_y) & (z_dir <= 0)
    if np.any(mask_nz):
        uc = -x[mask_nz] / abs_z[mask_nz]
        vc = -y[mask_nz] / abs_z[mask_nz]
        px = np.clip(((uc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        py = np.clip(((vc * 0.5 + 0.5) * (N - 1)).astype(np.int32), 0, N - 1)
        out_img[mask_nz] = faces['neg_z'][py, px]

    result_img = Image.fromarray(out_img, mode='RGB')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result_img.save(out_path, quality=95)
    print(f"Successfully saved equirectangular texture ({width}x{height}) to:\n  {out_path}")
    return True

if __name__ == "__main__":
    out = r"d:\Files\Coding\OpenGL\Stellar-Forge\textures\Saturn\Saturn_SE.jpg"
    extract_saturn_equirectangular(out)
