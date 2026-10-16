import contextlib
import copy
import importlib.util
import io
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).resolve().parents[1] / "scripts/compare_benchmarks.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "configuration": {"scenes": ["overview"], "frames": 2},
            "metadata": {"host": {}, "gpu": {}, "validation_enabled": False,
                         "assets": {"data/gaia/stars.bin": "catalog", "shaders/test.vert": "shader"}},
            "runs": [{"scene": "overview", "repetition": 0, "fixture": {"system_fingerprint": "system"},
                      "frames": [{"index": i, "simulation_years": 0, "eye_au": [0, 1, 2],
                                  "timings": {"frame_ms": 1 + i, "gpu_frame_ms": .5}} for i in range(2)]}]
        }
        self.candidate = copy.deepcopy(self.report)

    def test_shader_and_timing_changes_are_comparable(self):
        self.candidate["metadata"]["assets"]["shaders/test.vert"] = "new shader"
        self.candidate["runs"][0]["frames"][0]["timings"]["frame_ms"] = 3
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            comparison.compare(self.report, self.candidate)
        self.assertIn("+66.7%", output.getvalue())

    def test_execution_mode_can_change_with_identical_state_traces(self):
        self.report["configuration"]["simulation_mode"] = "inline"
        self.candidate["configuration"]["simulation_mode"] = "pipeline"
        comparison.compatible(self.report, self.candidate)

    def test_rejects_workload_machine_and_data_changes(self):
        for section, key, value in (("configuration", "frames", 3), ("metadata", "host", {"cpu": "different"}),
                                     ("metadata", "gpu", {"driver": "different"}),
                                     ("metadata", "assets", {"data/gaia/stars.bin": "different"})):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.report)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    comparison.compatible(self.report, changed)

    def test_rejects_missing_gpu_samples(self):
        del self.candidate["runs"][0]["frames"][0]["timings"]["gpu_frame_ms"]
        with self.assertRaises(ValueError):
            comparison.compatible(self.report, self.candidate)

    def test_rejects_camera_or_epoch_changes(self):
        for key, value in (("eye_au", [0, 1, 3]), ("simulation_years", 1)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.report)
                changed["runs"][0]["frames"][0][key] = value
                with self.assertRaises(ValueError):
                    comparison.compatible(self.report, changed)

    def test_compact_and_legacy_reports_compare(self):
        compact = copy.deepcopy(self.report)
        compact["schema_version"] = 2
        compact["scenes"] = list(comparison.scene_results(compact).values())
        del compact["runs"]
        comparison.compatible(self.report, compact)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            comparison.compare(self.report, compact)
        self.assertIn("+0.0%", output.getvalue())

    def test_repetitions_are_weighted_by_sample_count(self):
        extra = copy.deepcopy(self.report["runs"][0])
        extra["frames"] = [extra["frames"][0]]
        extra["frames"][0]["timings"]["frame_ms"] = 9
        self.report["runs"].append(extra)
        scene = comparison.scene_results(self.report)["overview"]
        self.assertEqual(scene["samples"], 3)
        self.assertEqual(scene["averages_ms"]["frame_ms"], 4)


if __name__ == "__main__":
    unittest.main()
