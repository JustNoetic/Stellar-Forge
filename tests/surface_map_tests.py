import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bake_surface_maps", ROOT/"scripts/bake_surface_maps.py")
maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maps)
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"scratch/surface-tests"
SCRATCH.mkdir(parents=True, exist_ok=True)
sys.argv = sys.argv[:1]


class SurfaceMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="fixture-", dir=SCRATCH))
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        image[:, :, 1] = 255
        Image.fromarray(image).save(cls.directory/"green.png")
        cls.manifest = maps.bake_color(cls.directory/"green.png", cls.directory/"green", 1, 1-1/298.257223563)
        y, x = np.meshgrid(np.arange(64), np.arange(128), indexing="ij")
        lon, lat = (x+.5)/128*2*np.pi-np.pi, np.pi*.5-(y+.5)/64*np.pi
        channels = np.stack((.5+.4*np.cos(lon)*np.cos(lat), .5+.4*np.sin(lat), .5+.4*np.sin(lon)*np.cos(lat)), axis=-1)
        Image.fromarray(np.rint(channels*255).astype(np.uint8)).save(cls.directory/"directions.png")
        maps.bake_color(cls.directory/"directions.png", cls.directory/"directions", 2, 1-1/298.257223563)
        (SCRATCH/"fixture-path.txt").write_text(str(cls.directory), encoding="utf-8")

    def test_lossless_tiles_and_index(self):
        payload = (self.directory/"green/tiles.rgba").read_bytes()
        block = maps.SIDE*maps.SIDE*4
        self.assertEqual(len(payload), len(self.manifest["checksums"])*block)
        for i, expected in enumerate(self.manifest["checksums"]):
            tile = payload[i*block:(i+1)*block]
            self.assertEqual(zlib.crc32(tile), expected)
            rgba = np.frombuffer(tile, dtype=np.uint8).reshape(-1, 4)
            self.assertTrue(np.all(rgba == [0, 255, 0, 255]))

    def test_filtering_is_linear_light(self):
        image = np.array([[[0, 0, 0], [255, 255, 255]], [[255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        self.assertTrue(np.all(np.abs(maps.reduce_linear(image).astype(int)-188) <= 1))

    def test_image_height_scale_preserves_negative_and_16bit_values(self):
        spec = importlib.util.spec_from_file_location("import_terrain", ROOT/"scripts/import_terrain.py")
        terrain = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(terrain)
        path = self.directory/"signed-height.png"
        Image.fromarray(np.array([[0, 65535], [0, 65535]], dtype=np.uint16)).save(path)
        raster = terrain.ImageRaster(path, .5, -1000)
        self.assertEqual(raster.sample(-90, 45), -1000)
        self.assertEqual(raster.sample(90, 45), 31767.5)
        self.assertEqual(raster.sample(180, -90), raster.sample(-180, -90))

    def test_cross_face_and_internal_borders(self):
        y, x = np.meshgrid(np.arange(256), np.arange(512), indexing="ij")
        source = np.stack(((x//8+y//8) % 2*255, x % 256, y), axis=-1).astype(np.uint8)
        for face in range(6):
            a = maps.encode(maps.sample(source, *maps.coordinates(face, 1, 0, 0, .99665, 0, "geodetic")))
            b = maps.encode(maps.sample(source, *maps.coordinates(face, 1, 1, 0, .99665, 0, "geodetic")))
            self.assertTrue(np.array_equal(a[:, -4:], b[:, :4]))
        for latitude in ("geodetic", "planetocentric", "parametric"):
            for face in range(6):
                u, v = maps.coordinates(face, 0, 0, 0, .99665, 0, latitude)
                self.assertTrue(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
                self.assertTrue(np.all((v >= 0) & (v <= 1)))

    def test_repeated_bake_preserves_output(self):
        again = maps.bake_color(self.directory/"green.png", self.directory/"green", 1, 1-1/298.257223563)
        self.assertEqual(again, self.manifest)
        with self.assertRaises(ValueError):
            maps.bake_color(self.directory/"green.png", self.directory/"green", 2, 1-1/298.257223563)

    def test_cube_edge_and_corner_filtering(self):
        payload = np.frombuffer((self.directory/"directions/tiles.rgba").read_bytes(), dtype=np.uint8)
        roots = payload[:6*maps.SIDE*maps.SIDE*4].reshape(6, maps.SIDE, maps.SIDE, 4)
        roots = maps.DECODE[roots[:, :, :, :3]]
        def filtered(face, u, v):
            x, y = maps.GUTTER+(u+1)*maps.CORE/2-.5, maps.GUTTER+(v+1)*maps.CORE/2-.5
            ix, iy = int(np.floor(x)), int(np.floor(y))
            fx, fy = x-ix, y-iy
            return (roots[face, iy, ix]*(1-fx)+roots[face, iy, ix+1]*fx)*(1-fy)+(roots[face, iy+1, ix]*(1-fx)+roots[face, iy+1, ix+1]*fx)*fy
        for x in (-1., -.5, 0., .5, 1.):
            for y in (-1., -.5, 0., .5, 1.):
                for z in (-1., -.5, 0., .5, 1.):
                    if [abs(x), abs(y), abs(z)].count(1.) < 2:
                        continue
                    incident = []
                    if abs(x) == 1: incident.append(filtered(0 if x > 0 else 1, -z if x > 0 else z, y))
                    if abs(y) == 1: incident.append(filtered(2 if y > 0 else 3, x, -z if y > 0 else z))
                    if abs(z) == 1: incident.append(filtered(4 if z > 0 else 5, x if z > 0 else -x, y))
                    self.assertLess(np.max(np.ptp(incident, axis=0)), .012)


if __name__ == "__main__":
    unittest.main()
