import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("migrate_atmospheres", ROOT / "scripts/migrate_atmospheres.py")
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class AtmosphereMigrationTests(unittest.TestCase):
    def test_manual_custom_optics_survive(self):
        record = dict(surface_pressure=.01, temperature=200, composition={"CO2": 1},
                      aerosol_mode="manual", beta_mie=4e-5, h_mie=12,
                      mie_albedo=[.9, .8, .7], mie_angstrom=.3, legacy_tholin_hint=.05)
        result = migration.migrate(record)
        for key in ("beta_mie", "h_mie", "mie_albedo", "mie_angstrom"):
            self.assertEqual(result[key], record[key])
        self.assertNotIn("legacy_tholin_hint", result)
        self.assertEqual(result["pressure_pa"], 1013.25)
        self.assertEqual(migration.migrate(result), result)

    def test_repair_only_explicitly_selected_solar_presets(self):
        record = dict(surface_pressure=.006, temperature=214, composition={"CO2": 1},
                      aerosol_mode="manual", beta_mie=4e-5)
        self.assertNotIn("aerosols", migration.migrate(record, "Mars"))
        repaired = migration.migrate(record, "Mars", True)
        self.assertEqual(repaired["aerosols"][0]["material"], "mineral_dust")
        self.assertNotIn("beta_mie", repaired)
        self.assertEqual(migration.migrate(repaired, "Mars", True), repaired)

    def test_explicit_clear_gas_and_disabled_inputs(self):
        record = dict(schema_version=2, enabled=False, pressure_pa=140000, temperature_k=94,
                      composition={"N2": .94, "CH4": .06}, aerosols=[])
        result = migration.migrate(record)
        self.assertEqual(result["aerosols"], [])
        self.assertFalse(result["enabled"])
        self.assertEqual(result["temperature_k"], 94)

    def test_duplicate_records_consolidate_without_unrelated_changes(self):
        source = '[\n  {"name": "Earth",\n    "atmosphere": {"surface_pressure": 1, "temperature": 288, "composition": {"N2": 1}},\n    "r": 0.009158,\n    "atmosphere": {"enabled": false},\n    "Omega": 2, "omega": 3}\n]'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system.json"
            path.write_text(source)
            migration.migrate_file(path)
            migrated = path.read_text()
            self.assertEqual(migrated.count('"atmosphere"'), 1)
            self.assertIn('"Omega": 2, "omega": 3', migrated)
            body = json.loads(migrated)[0]
            self.assertEqual(body["r"], .009158)
            self.assertFalse(body["atmosphere"]["enabled"])
            self.assertEqual(body["atmosphere"]["composition"], {"N2": 1})
            migration.migrate_file(path)
            self.assertEqual(migrated, path.read_text())

    def test_bundled_systems_have_no_duplicate_keys_or_legacy_hints(self):
        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                self.assertNotIn(key, result, f"Duplicate JSON key: {key}")
                result[key] = value
            return result
        for path in [ROOT / "data/system.json", *[ROOT / "data/systems" / name / "system.json" for name in ("Solar System", "Ephemeris Mode")]]:
            data = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
            if path.parent.name == "Ephemeris Mode":
                self.assertEqual(data, {"source_preset": "Solar System"})
                continue
            for body in data:
                if "atmosphere" not in body:
                    continue
                atmosphere = body["atmosphere"]
                self.assertEqual(atmosphere["schema_version"], 2)
                for key in ("legacy_tholin_hint", "intensity", "composition_basis", "height_mode", "surface_pressure"):
                    self.assertNotIn(key, atmosphere)
                if body["name"] in ("Mars", "Titan"):
                    self.assertTrue(atmosphere["aerosols"])

    def test_pressure_reference_and_alias_migration_are_lossless(self):
        record = dict(schema_version=2, model="physical", reference_pressure_pa=100000,
                      temperature_k=165, composition={"H2": 1}, bottom_altitude_km=-80, aerosols=[])
        migrated = migration.migrate(record)
        self.assertEqual(migrated["reference_pressure_pa"], 100000)
        self.assertEqual(migrated["bottom_altitude_km"], -80)
        self.assertNotIn("pressure_pa", migrated)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"system.json"
            path.write_text('{"source_preset": "Solar System"}\n')
            self.assertEqual(migration.migrate_file(path), 0)
            self.assertEqual(path.read_text(), '{"source_preset": "Solar System"}\n')

    def test_measured_methane_band_reduction(self):
        spec = importlib.util.spec_from_file_location("methane", ROOT/"scripts/prepare_methane_optics.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        means = module.band_means(ROOT/"data/optics/ch4_karkoschka_1994.txt")
        for actual, expected in zip(means, [3.50986e-30, 6.70107e-31, 8.25387234042553e-32]):
            self.assertAlmostEqual(actual/expected, 1, places=12)


if __name__ == "__main__":
    unittest.main()
