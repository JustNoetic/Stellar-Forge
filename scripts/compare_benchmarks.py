import argparse
import json
import math
from pathlib import Path


def load(path):
    path = Path(path)
    if path.is_dir():
        path /= "report.json"
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema_version") not in (1, 2):
        raise ValueError(f"Unsupported report schema: {path}")
    return report


def scene_results(report):
    if "scenes" in report:
        return {scene["scene"]: scene for scene in report["scenes"]}
    result = {}
    for run in report["runs"]:
        scene = result.setdefault(run["scene"], {"scene": run["scene"], "fixture": run["fixture"],
                                               "samples": 0, "averages_ms": {}, "timing_counts": {}})
        if scene["fixture"] != run["fixture"]:
            raise ValueError("Fixtures differ across repetitions.")
        for frame in run["frames"]:
            scene["samples"] += 1
            times = dict(frame["timings"])
            if all(key in times for key in ("cpu_simulation_ms", "cpu_camera_ms", "cpu_record_ms")):
                times["cpu_work_ms"] = sum(times[key] for key in ("cpu_simulation_ms", "cpu_camera_ms", "cpu_record_ms"))
            if all(key in times for key in ("fence_wait_ms", "acquire_wait_ms", "present_ms")):
                times["wait_ms"] = sum(times[key] for key in ("fence_wait_ms", "acquire_wait_ms", "present_ms"))
            for key, value in times.items():
                scene["averages_ms"][key] = scene["averages_ms"].get(key, 0) + value
                scene["timing_counts"][key] = scene["timing_counts"].get(key, 0) + 1
    for scene in result.values():
        for key in scene["averages_ms"]:
            scene["averages_ms"][key] /= scene["timing_counts"][key]
    return result


def compatible(before, after):
    configuration = lambda report: {key: value for key, value in report["configuration"].items()
                                    if key != "simulation_mode"}
    if configuration(before) != configuration(after):
        raise ValueError("Benchmark configurations differ; use identical capture arguments.")
    for key in ("host", "gpu", "validation_enabled"):
        if before["metadata"][key] != after["metadata"][key]:
            raise ValueError(f"Machine/driver settings differ: {key}")
    data = lambda report: {key: value for key, value in report["metadata"]["assets"].items()
                           if not key.startswith("shaders/")}
    if data(before) != data(after):
        raise ValueError("Catalog, SPICE runtime or kernel fingerprints differ.")
    left_scenes, right_scenes = scene_results(before), scene_results(after)
    if left_scenes.keys() != right_scenes.keys():
        raise ValueError("Scenes differ.")
    for name, left in left_scenes.items():
        right = right_scenes[name]
        for key in ("fixture", "samples", "timing_counts"):
            if left[key] != right[key]:
                raise ValueError(f"Scene {name}: {key} differs.")
        if left["averages_ms"].keys() != right["averages_ms"].keys():
            raise ValueError("Timing coverage differs.")
    if "runs" not in before or "runs" not in after:
        return
    if len(before["runs"]) != len(after["runs"]):
        raise ValueError("Run counts differ.")
    for left, right in zip(before["runs"], after["runs"]):
        for key in ("scene", "repetition", "fixture"):
            if left[key] != right[key]:
                raise ValueError(f"Fixtures differ: {left['scene']} repetition {left['repetition']} ({key})")
        if len(left["frames"]) != len(right["frames"]) or len(left["frames"]) != before["configuration"]["frames"]:
            raise ValueError("Incomplete or mismatched frame counts.")
        for a, b in zip(left["frames"], right["frames"]):
            if a["index"] != b["index"] or not math.isclose(a["simulation_years"], b["simulation_years"], rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Simulation traces differ.")
            if not all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15) for x, y in zip(a["eye_au"], b["eye_au"])):
                raise ValueError("Camera traces differ.")
            if a["timings"].keys() != b["timings"].keys():
                raise ValueError("Timing coverage differs; missing samples cannot be compared as zero.")


def compare(before, after, selected=None):
    compatible(before, after)
    print("Average timings across all repetitions. Lower is faster.")
    print("| Scene | Timing | Before (ms) | After (ms) | Change |")
    print("|---|---|---:|---:|---:|")
    left_scenes, right_scenes = scene_results(before), scene_results(after)
    for scene in before["configuration"]["scenes"]:
        left, right = left_scenes[scene]["averages_ms"], right_scenes[scene]["averages_ms"]
        defaults = [key for key in ("frame_ms", "cpu_work_ms", "gpu_frame_ms") if key in left]
        for metric in selected or defaults:
            if metric not in left or metric not in right:
                raise ValueError(f"Missing metric: {metric}")
            a, b = left[metric], right[metric]
            if not all(math.isfinite(value) and value >= 0 for value in (a, b)):
                raise ValueError("Invalid timing sample.")
            delta = f"{(b / a - 1) * 100:+.1f}%" if a else "n/a"
            print(f"| {scene} | {metric} | {a:.4f} | {b:.4f} | {delta} |")


def main():
    parser = argparse.ArgumentParser(description="Compare matching Stellar Forge benchmark reports; informational, no performance gate.")
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--metric", action="append", help="Show only this timing column; repeat to select multiple.")
    args = parser.parse_args()
    try:
        compare(load(args.before), load(args.after), args.metric)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Comparison rejected: {error}\n")


if __name__ == "__main__":
    main()
