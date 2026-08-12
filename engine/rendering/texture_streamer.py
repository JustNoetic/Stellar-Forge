import os
import glob
import threading
import queue
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

def compute_texture_spherical_mean(img):
    """
    Compute area-weighted spherical average color of an equirectangular texture.
    Linearizes sRGB colors and weights pixels by cos(latitude).
    """
    small = img.resize((128, 64), Image.Resampling.BOX).convert('RGBA')
    arr = np.array(small, dtype=np.float32) / 255.0
    rgb_linear = np.where(arr[..., :3] <= 0.04045, arr[..., :3] / 12.92, ((arr[..., :3] + 0.055) / 1.055) ** 2.4)
    
    H, W = arr.shape[0], arr.shape[1]
    lats = (np.arange(H, dtype=np.float32) + 0.5) / H * np.pi - (np.pi * 0.5)
    weights = np.cos(lats)[:, None, None] # (H, 1, 1)
    
    alpha = arr[..., 3:4] # (H, W, 1)
    weighted_rgb = (rgb_linear * weights * alpha).sum(axis=(0, 1))
    total_w = (weights * alpha).sum()
    if total_w > 1e-6:
        mean_linear = weighted_rgb / total_w
    else:
        mean_linear = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return mean_linear

def _prepare_cloud_image(img, target_size=None):
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img_rgba = img.convert('RGBA')
    else:
        img_l = img.convert('L')
        img_rgba = Image.merge('RGBA', (img_l, img_l, img_l, img_l))
    if target_size:
        return img_rgba.resize(target_size, Image.Resampling.LANCZOS)
    return img_rgba

class TextureStreamer:
    """
    Asynchronous texture streamer for celestial bodies.
    Manages low-resolution fallback textures and dynamically decodes high-resolution
    textures on background worker threads.
    """
    def __init__(self, textures_dir, fallback_size=(1024, 512)):
        self.textures_dir = textures_dir
        self.fallback_size = fallback_size
        
        # Mapping: name_lower -> dict of file paths {diffuse, normal, specular, clouds}
        self.file_manifest = {}
        # Mapping: name_lower -> 1-based body texture index (1..N)
        self.name_to_idx = {}
        self.idx_to_name = {}
        self.texture_mean_colors = {}
        
        # Low-res decoded byte bundles ready for initial GPU upload:
        # idx -> dict with {'diffuse': (size, bytes), 'normal': (size, bytes), ...}
        self.fallback_data = {}
        
        # Threading queues
        self.load_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.in_progress = set() # (name_lower, level)
        
        self._shutdown_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        
        self._scan_manifest()
        self._build_fallbacks()
        self.worker_thread.start()

    def _scan_manifest(self):
        if not os.path.exists(self.textures_dir):
            return

        target_dirs = []
        for root, dirs, files in os.walk(self.textures_dir):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                target_dirs.append((dir_name, folder_path))
        target_dirs.append(("", self.textures_dir))

        for folder_name, path in target_dirs:
            if path == self.textures_dir:
                files = glob.glob(os.path.join(self.textures_dir, '*.png')) + glob.glob(os.path.join(self.textures_dir, '*.jpg'))
                base_names = set()
                for f in files:
                    name = os.path.splitext(os.path.basename(f))[0]
                    if any(name.endswith(s) for s in ["_ring", "_rings", "_normal", "_specular", "_clouds", "_cloud", "_front", "_back"]):
                        continue
                    base_names.add(name)
                
                for name in sorted(list(base_names)):
                    name_lower = name.lower()
                    if name_lower in self.file_manifest:
                        continue
                    
                    d_path = os.path.join(self.textures_dir, name + ".png")
                    if not os.path.exists(d_path): d_path = os.path.join(self.textures_dir, name + ".jpg")
                    if not os.path.exists(d_path): d_path = None
                    
                    n_path = os.path.join(self.textures_dir, name + "_normal.png")
                    if not os.path.exists(n_path): n_path = os.path.join(self.textures_dir, name + "_normal.jpg")
                    if not os.path.exists(n_path): n_path = None
                    
                    s_path = os.path.join(self.textures_dir, name + "_specular.png")
                    if not os.path.exists(s_path): s_path = os.path.join(self.textures_dir, name + "_specular.jpg")
                    if not os.path.exists(s_path): s_path = None
                    
                    c_path = None
                    for c_cand in [name + "_clouds.png", name + "_clouds.jpg", name + "_cloud.png", name + "_cloud.jpg"]:
                        cand_p = os.path.join(self.textures_dir, c_cand)
                        if os.path.exists(cand_p):
                            c_path = cand_p
                            break
                            
                    self.file_manifest[name_lower] = {
                        'name': name,
                        'diffuse': d_path,
                        'normal': n_path,
                        'specular': s_path,
                        'clouds': c_path
                    }
            else:
                obj_name = folder_name
                name_lower = obj_name.lower()
                files = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg'))
                
                d_path, n_path, s_path, c_path = None, None, None, None
                for f in files:
                    fname_lower = os.path.splitext(os.path.basename(f))[0].lower()
                    if fname_lower == name_lower:
                        d_path = f
                    elif fname_lower in [name_lower + "_normal", "normal", "diffuse_normal"]:
                        n_path = f
                    elif fname_lower in [name_lower + "_specular", "specular", "diffuse_specular"]:
                        s_path = f
                    elif fname_lower in [name_lower + "_clouds", name_lower + "_cloud", "clouds", "cloud"]:
                        c_path = f
                
                if not d_path:
                    for f in files:
                        fname_lower = os.path.splitext(os.path.basename(f))[0].lower()
                        if fname_lower in ["diffuse", "albedo", "color", "map"]:
                            d_path = f; break
                if not n_path:
                    for f in files:
                        fname_lower = os.path.splitext(os.path.basename(f))[0].lower()
                        if fname_lower in ["bump", "n", "nm"]:
                            n_path = f; break
                if not s_path:
                    for f in files:
                        fname_lower = os.path.splitext(os.path.basename(f))[0].lower()
                        if fname_lower in ["spec", "s", "specular_map"]:
                            s_path = f; break
                            
                if d_path or n_path or s_path or c_path:
                    self.file_manifest[name_lower] = {
                        'name': obj_name,
                        'diffuse': d_path,
                        'normal': n_path,
                        'specular': s_path,
                        'clouds': c_path
                    }

        for idx, (name_lower, info) in enumerate(sorted(self.file_manifest.items()), start=1):
            self.name_to_idx[name_lower] = idx
            self.idx_to_name[idx] = name_lower

    def _build_fallbacks(self):
        """Pre-decode low-resolution fallback textures at startup."""
        for name_lower, idx in self.name_to_idx.items():
            paths = self.file_manifest[name_lower]
            entry = {}
            
            # Diffuse
            if paths['diffuse'] and os.path.exists(paths['diffuse']):
                try:
                    img_d = Image.open(paths['diffuse']).convert('RGBA')
                    self.texture_mean_colors[name_lower] = compute_texture_spherical_mean(img_d)
                    img_d_low = img_d.resize(self.fallback_size, Image.Resampling.LANCZOS)
                    entry['diffuse'] = (self.fallback_size, img_d_low.tobytes(), 4)
                except Exception as e:
                    print(f"[TextureStreamer] Error loading fallback diffuse for {name_lower}: {e}")
                    img_blank = Image.new('RGBA', self.fallback_size, (255, 255, 255, 255))
                    entry['diffuse'] = (self.fallback_size, img_blank.tobytes(), 4)
            else:
                img_blank = Image.new('RGBA', self.fallback_size, (255, 255, 255, 255))
                entry['diffuse'] = (self.fallback_size, img_blank.tobytes(), 4)

            # Normal
            if paths['normal'] and os.path.exists(paths['normal']):
                try:
                    img_n = Image.open(paths['normal']).convert('RGBA')
                    img_n_low = img_n.resize(self.fallback_size, Image.Resampling.LANCZOS)
                    entry['normal'] = (self.fallback_size, img_n_low.tobytes(), 4)
                except Exception as e:
                    img_flat = Image.new('RGBA', self.fallback_size, (128, 128, 255, 255))
                    entry['normal'] = (self.fallback_size, img_flat.tobytes(), 4)
            else:
                img_flat = Image.new('RGBA', self.fallback_size, (128, 128, 255, 255))
                entry['normal'] = (self.fallback_size, img_flat.tobytes(), 4)

            # Specular
            if paths['specular'] and os.path.exists(paths['specular']):
                try:
                    img_s = Image.open(paths['specular']).convert('L')
                    img_s_low = img_s.resize(self.fallback_size, Image.Resampling.LANCZOS)
                    entry['specular'] = (self.fallback_size, img_s_low.tobytes(), 1)
                except Exception as e:
                    img_black = Image.new('L', self.fallback_size, 0)
                    entry['specular'] = (self.fallback_size, img_black.tobytes(), 1)
            else:
                img_black = Image.new('L', self.fallback_size, 0)
                entry['specular'] = (self.fallback_size, img_black.tobytes(), 1)

            # Clouds
            if paths['clouds'] and os.path.exists(paths['clouds']):
                try:
                    img_c = _prepare_cloud_image(Image.open(paths['clouds']), self.fallback_size)
                    entry['clouds'] = (self.fallback_size, img_c.tobytes(), 4)
                except Exception as e:
                    img_trans = Image.new('RGBA', self.fallback_size, (0, 0, 0, 0))
                    entry['clouds'] = (self.fallback_size, img_trans.tobytes(), 4)
            else:
                img_trans = Image.new('RGBA', self.fallback_size, (0, 0, 0, 0))
                entry['clouds'] = (self.fallback_size, img_trans.tobytes(), 4)

            self.fallback_data[idx] = entry

    def request_high_res(self, name_lower):
        """Request asynchronous decoding of high-resolution textures."""
        if name_lower not in self.file_manifest:
            return
        key = (name_lower, 'high')
        if key in self.in_progress:
            return
        self.in_progress.add(key)
        self.load_queue.put(name_lower)

    def _worker_loop(self):
        while not self._shutdown_event.is_set():
            try:
                name_lower = self.load_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if name_lower not in self.file_manifest:
                self.in_progress.discard((name_lower, 'high'))
                continue

            paths = self.file_manifest[name_lower]
            idx = self.name_to_idx[name_lower]
            result = {'name_lower': name_lower, 'idx': idx, 'maps': {}}

            # High-res Diffuse
            if paths['diffuse'] and os.path.exists(paths['diffuse']):
                try:
                    img_d = Image.open(paths['diffuse']).convert('RGBA')
                    result['maps']['diffuse'] = (img_d.size, img_d.tobytes(), 4)
                except Exception as e:
                    print(f"[TextureStreamer] Error decoding high-res diffuse for {name_lower}: {e}")

            # High-res Normal
            if paths['normal'] and os.path.exists(paths['normal']):
                try:
                    img_n = Image.open(paths['normal']).convert('RGBA')
                    result['maps']['normal'] = (img_n.size, img_n.tobytes(), 4)
                except Exception as e:
                    print(f"[TextureStreamer] Error decoding high-res normal for {name_lower}: {e}")

            # High-res Specular
            if paths['specular'] and os.path.exists(paths['specular']):
                try:
                    img_s = Image.open(paths['specular']).convert('L')
                    result['maps']['specular'] = (img_s.size, img_s.tobytes(), 1)
                except Exception as e:
                    print(f"[TextureStreamer] Error decoding high-res specular for {name_lower}: {e}")

            # High-res Clouds
            if paths['clouds'] and os.path.exists(paths['clouds']):
                try:
                    img_c = _prepare_cloud_image(Image.open(paths['clouds']))
                    result['maps']['clouds'] = (img_c.size, img_c.tobytes(), 4)
                except Exception as e:
                    print(f"[TextureStreamer] Error decoding high-res clouds for {name_lower}: {e}")

            self.result_queue.put(result)
            self.in_progress.discard((name_lower, 'high'))

    def poll_results(self, max_items=2):
        """Retrieve up to max_items completed decode results for main-thread GPU upload."""
        results = []
        for _ in range(max_items):
            try:
                res = self.result_queue.get_nowait()
                results.append(res)
            except queue.Empty:
                break
        return results

    def shutdown(self):
        self._shutdown_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
