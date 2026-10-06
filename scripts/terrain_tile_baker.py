"""Direct, bounded-memory tile baking with two cross-face gutter texels.

Elevation is lossless normalized uint16 PNG. Each tile is evaluated directly
from the source at pixel centres; there is no per-tile edge averaging.
"""
import concurrent.futures
import os
from pathlib import Path
import numpy as np
from PIL import Image

GUTTER = 2


def source_array(image, height=False):
    if height:
        arr = np.asarray(image)
        if image.mode.startswith('I') and arr.ndim == 2:
            if arr.min() < 0 or arr.max() > 65535:
                raise ValueError('Height PNG must contain unsigned values from 0 to 65535.')
            return arr.astype('f4') / 65535.
        return np.asarray(image.convert('L'), dtype='f4') / 255.
    return np.asarray(image.convert('RGBA'), dtype='f4')


def tile_coordinates(face, lod, tx, ty, size):
    count = 1 << lod
    axis = (np.arange(size + 2 * GUTTER, dtype='f8') - GUTTER + .5) / size
    u, v = np.meshgrid((tx + axis) / count * 2. - 1., (ty + axis) / count * 2. - 1.)
    x, y = np.tan(u * np.pi / 4), np.tan(v * np.pi / 4)
    one = np.ones_like(x)
    p = ((one, -y, -x), (-one, -y, x), (x, one, y),
         (x, -one, -y), (x, -y, one), (-x, -y, -one))[face]
    norm = np.sqrt(sum(a*a for a in p))
    return (0.5 + np.arctan2(p[2], p[0]) / (2*np.pi)) % 1., .5 - np.arcsin(np.clip(p[1]/norm, -1., 1.)) / np.pi


def sample_source(source, u, v):
    h, w = source.shape[:2]
    x, y = u*w - .5, np.clip(v*h - .5, 0., h-1.)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-ix, y-iy
    nx, ny = (ix+1) % w, np.minimum(iy+1, h-1)
    ix %= w
    if source.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return ((source[iy, ix]*(1-fx) + source[iy, nx]*fx)*(1-fy)
            + (source[ny, ix]*(1-fx) + source[ny, nx]*fx)*fy)


def bake_tiles(body, src_path, map_type, max_lod, size, output):
    if not 0 <= max_lod <= 10 or not 8 <= size <= 1024:
        raise ValueError('Tile LOD must be 0..10 and content size 8..1024.')
    height = map_type.lower() == 'height'
    with Image.open(src_path) as image:
        if map_type.lower() == 'clouds' and not ('A' in image.getbands() or 'transparency' in image.info):
            alpha = image.convert('L')
            image = Image.new('RGBA', image.size, (255,255,255,255))
            image.putalpha(alpha)
        source = source_array(image, height)
    output = Path(output)
    for face in range(6):
        for lod in range(max_lod+1):
            (output/str(face)/str(lod)).mkdir(parents=True, exist_ok=True)

    def jobs():
        for lod in range(max_lod+1):
            for face in range(6):
                for ty in range(1 << lod):
                    for tx in range(1 << lod):
                        yield face,lod,tx,ty

    def bake(job):
        face,lod,tx,ty = job
        u,v = tile_coordinates(face,lod,tx,ty,size)
        values = sample_source(source,u,v)
        arr = np.round(np.clip(values,0.,1.)*65535).astype('u2') if height else np.round(np.clip(values,0.,255.)).astype('u1')
        path = output/str(face)/str(lod)/f'{tx}_{ty}.png'
        Image.fromarray(arr).save(path,'PNG')

    workers = min(4, max(1, (os.cpu_count() or 2)-1))
    iterator = iter(jobs())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        while True:
            while len(pending) < workers*2:
                try:
                    pending.add(executor.submit(bake,next(iterator)))
                except StopIteration:
                    break
            if not pending:
                break
            done,pending = concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
    (output/'terrain-v2.txt').write_text(f'version=2\ncontent_size={size}\ngutter={GUTTER}\nheight_bits={16 if height else 0}\n',encoding='ascii')
    print(f'[{body}] Baked {map_type}: pixel-centred PNG tiles, {GUTTER}-texel gutters, {"16-bit elevation" if height else "RGBA colour"}.')
