import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("import_terrain", ROOT / "scripts/import_terrain.py")
terrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terrain)
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scratch/terrain-source-tests"
SCRATCH.mkdir(parents=True, exist_ok=True)
sys.argv = sys.argv[:1]


def fixture(output):
    return terrain.import_tiles(output, 2, 6378137, 1-1/298.257223563, 0,
        lambda lon, lat, d: -600+1800*d[0]*d[2]+350*math.sin(7*d[1])*math.cos(5*d[0]),
        {"fixture": "signed-height-v1"}, "Signed height fixture")


class TerrainImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="fixture-", dir=SCRATCH))
        cls.manifest = fixture(cls.directory / "dataset")
        terrain.import_tiles(cls.directory / "flat", 2, 6378137, 1-1/298.257223563, 0,
            lambda lon, lat, d: -1500, {"fixture": "constant-negative-height"}, "Flat signed height")
        cls.payload = (cls.directory / "dataset/heights.bin").read_bytes()
        cls.records = {tuple(row[:4]): row for row in cls.manifest["tiles"]}
        (SCRATCH / "fixture-path.txt").write_text(str(cls.directory / "dataset"), encoding="utf-8")

    def values(self, key):
        row = self.records[key]
        return struct.unpack_from("<361f", self.payload, row[4])

    def test_determinism_and_checksums(self):
        with tempfile.TemporaryDirectory(prefix="repeat-", dir=SCRATCH) as directory:
            manifest = fixture(Path(directory) / "dataset")
            self.assertEqual(manifest, self.manifest)
            self.assertEqual((Path(directory) / "dataset/heights.bin").read_bytes(), self.payload)
        for record in self.records.values():
            block = self.payload[record[4]:record[4]+1444]
            self.assertEqual(terrain.checksum(block), record[5])

    def test_negative_scale_nodata_and_coverage(self):
        raster = terrain.Raster(dict(latitude="geodetic", longitude="east", orientation="north-to-south",
            registration="point", vertical_datum="ellipsoid", values=[[10, None], [30, 40]],
            coverage_deg=[-10, 10, -5, 5], scale=-2, offset=3, fill_m=-123, nodata=-9999))
        self.assertEqual(raster.sample(-10, 5), -17)
        self.assertEqual(raster.sample(10, 5), -123)
        self.assertEqual(raster.sample(-10, -5), -57)
        self.assertEqual(raster.sample(50, 0), -123)
        self.assertEqual(raster.sample(0, 0), (-17-123-57-77)/4)
        self.assertLess(min(row[6] for row in self.records.values()), 0)
        self.assertGreater(max(row[7] for row in self.records.values()), 0)

    def test_nested_samples_extrema_and_error(self):
        for key, record in self.records.items():
            face, level, x, y = key
            values = self.values(key)
            if level == 2:
                continue
            for cy in range(2):
                for cx in range(2):
                    child_key = face, level+1, x*2+cx, y*2+cy
                    child, heights = self.records[child_key], self.values(child_key)
                    self.assertLessEqual(record[6], child[6])
                    self.assertGreaterEqual(record[7], child[7])
                    for gy in range(17):
                        for gx in range(17):
                            coarse = terrain.triangle(values, (cx*16+gx)/2, (cy*16+gy)/2)
                            fine = heights[(gy+1)*19+gx+1]
                            self.assertLessEqual(abs(coarse-fine)+child[8], record[8]+1e-8)
                            if gx % 2 == 0 and gy % 2 == 0:
                                self.assertEqual(coarse, fine)

    def test_cube_seams_corners_and_gutters(self):
        samples = {}
        for key in self.records:
            face, level, tx, ty = key
            values = self.values(key)
            resolution = 16*(1 << level)
            for gy in range(17):
                for gx in range(17):
                    p = terrain.cube(face, -1+2*(tx*16+gx)/resolution, -1+2*(ty*16+gy)/resolution)
                    canonical = (level, *p)
                    height = values[(gy+1)*19+gx+1]
                    if canonical in samples:
                        self.assertEqual(samples[canonical], height)
                    samples[canonical] = height
            self.assertTrue(all(math.isfinite(v) for v in values))

    def test_invalid_inputs_and_output_preservation(self):
        with self.assertRaises(ValueError):
            fixture(self.directory / "dataset")
        with self.assertRaises(ValueError):
            terrain.import_tiles(self.directory / "bad", 6, 6378137, 1, 0, lambda *args: 0, {}, "bad")
        self.assertFalse((self.directory / "bad").exists())

    def test_geographic_input_fixture(self):
        document = json.loads((ROOT / "tests/terrain_signed_fixture.json").read_text())
        raster = terrain.Raster(document)
        with tempfile.TemporaryDirectory(prefix="raster-", dir=SCRATCH) as temporary:
            manifest = terrain.import_tiles(Path(temporary)/"dataset", 1, 6378137, 1-1/298.257223563, 0,
                lambda lon, lat, d: raster.sample(lon, lat), {"source": "terrain_signed_fixture.json"}, "Geographic fixture")
            self.assertLessEqual(min(row[6] for row in manifest["tiles"]), -1000)
            self.assertGreater(max(row[7] for row in manifest["tiles"]), 0)

    def test_gdal_adapter_when_available(self):
        try:
            from osgeo import gdal, osr
        except ImportError:
            self.skipTest("GDAL Python bindings are not installed")
        with tempfile.TemporaryDirectory(prefix="gdal-", dir=SCRATCH) as temporary:
            path = Path(temporary)/"signed.tif"
            dataset = gdal.GetDriverByName("GTiff").Create(str(path), 4, 2, 1, gdal.GDT_Float32)
            crs = osr.SpatialReference()
            crs.ImportFromEPSG(4326)
            dataset.SetSpatialRef(crs)
            dataset.SetGeoTransform((-180, 90, 0, 90, 0, -90))
            band = dataset.GetRasterBand(1)
            band.SetNoDataValue(-9999)
            band.SetScale(2)
            band.SetOffset(-5)
            band.SetUnitType("m")
            band.WriteRaster(0, 0, 4, 2, struct.pack("=8f", 10, -9999, 30, 40, 50, 60, 70, 80))
            band = None
            dataset = None
            raster = terrain.GdalRaster(path, 6378137, 1-1/298.257223563, None, -123)
            self.assertAlmostEqual(raster.sample(-135, 45), 15)
            self.assertAlmostEqual(raster.sample(-45, 45), -123)


if __name__ == "__main__":
    unittest.main()
