"""Capture and summarize the full application performance audit."""

import argparse
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def capture(args, name, scene, variant="baseline", width=None, height=None, ui=True, frames=None, repeats=None, step=None):
    output = args.output / name
    if (output / "report.json").exists():
        print(f"Retaining completed capture: {name}", flush=True)
        return
    if output.exists():
        raise RuntimeError(f"Incomplete capture exists: {output}; choose a new output directory")
    command = [str(args.exe), "--benchmark", "--benchmark-scene", scene,
               "--benchmark-variant", variant, "--benchmark-width", str(width or args.width),
               "--benchmark-height", str(height or args.height), "--benchmark-warmup", str(args.warmup),
               "--benchmark-frames", str(frames or args.frames), "--benchmark-repeats", str(repeats or args.repeats),
               "--benchmark-simulation", "pipeline", "--benchmark-detailed", "--benchmark-output", str(output)]
    if ui:
        command.append("--benchmark-ui")
    if step is not None:
        command.extend(["--benchmark-step", str(step)])
    print(f"Capturing {name}: {scene}, {variant}", flush=True)
    with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(report):
    config = report["configuration"]
    if not report["metadata"]["gpu"]["timestamps_supported"]:
        raise RuntimeError("This audit requires GPU timestamp support")
    for run in report["runs"]:
        if len(run["frames"]) != config["frames"]:
            raise RuntimeError("Incomplete frame capture")
        for frame in run["frames"]:
            times = frame["timings"]
            if not all(math.isfinite(v) and v >= 0 for v in times.values()):
                raise RuntimeError("Nonfinite or negative timing")
            if "gpu_frame_ms" not in times or "gpu_terrain_draw_ms" not in times:
                raise RuntimeError("Missing required GPU audit scopes")
            if any(frame["counters"].get(key, 0) for key in ("texture_failed", "normal_failed")):
                raise RuntimeError("Streaming failure in measured workload")


def check_trace(baseline, candidate):
    for key in ("host", "gpu", "assets", "build", "validation_enabled"):
        if baseline["metadata"][key] != candidate["metadata"][key]:
            raise RuntimeError(f"Experiment metadata differs: {key}")
    if baseline["configuration"]["step_seconds"] != candidate["configuration"]["step_seconds"]:
        fixtures = {s["scene"]: s["fixture"] for s in baseline["scenes"]}
        for run in candidate["runs"]:
            for key in ("system", "system_fingerprint", "body_count", "motion_model", "fov_deg", "tracking_body", "render_settings"):
                if fixtures[run["scene"]][key] != run["fixture"][key]:
                    raise RuntimeError(f"Cadence fixture differs: {key}")
            expected = candidate["configuration"]["step_seconds"]*candidate["configuration"]["warp"]/(365.25*86400)
            for a, b in zip(run["frames"], run["frames"][1:]):
                if not math.isclose(b["simulation_years"]-a["simulation_years"], expected, rel_tol=1e-8, abs_tol=1e-15):
                    raise RuntimeError("Cadence simulation trace has an unexpected time step")
        return
    original = {(r["scene"], r["repetition"]): r for r in baseline["runs"]}
    for run in candidate["runs"]:
        left = original[(run["scene"], run["repetition"])]["frames"]
        right = run["frames"]
        if len(left) != len(right):
            raise RuntimeError("Experiment frame counts differ")
        for a, b in zip(left, right):
            if a["simulation_years"] != b["simulation_years"] or a["eye_au"] != b["eye_au"]:
                raise RuntimeError("Experiment camera/simulation traces differ")
            for key in ("camera_yaw_deg", "camera_pitch_deg"):
                if a["counters"][key] != b["counters"][key]:
                    raise RuntimeError("Experiment camera orientations differ")


def percentile(values, fraction):
    values = sorted(values)
    offset = (len(values) - 1) * fraction
    low = int(offset)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (offset - low)


def summarize(output):
    baseline = load(output / "baseline" / "report.json")
    validate(baseline)
    scenes = {s["scene"]: s for s in baseline["scenes"]}
    runs = baseline["runs"]
    metadata = baseline["metadata"]
    config = baseline["configuration"]
    started = datetime.strptime(metadata["started_utc"], "%Y%m%dT%H%M%SZ")
    lines = [f"# Performance audit — {started:%d %B %Y}", "",
             f"{metadata['host']['cpu_name'].strip()} · {metadata['gpu']['name']} · "
             f"{config['width']}×{config['height']} · Release · UI on · VSync off · visible window.", "",
             f"{config['warmup']} warmup + {config['frames']} measured frames × {config['repeats']} repetitions per scene. "
             "Deterministic simulation pipeline; views without the -live suffix are paused except analytical/nbody. Timings are milliseconds. "
             "CPU work is main-thread wall time; CPU and GPU overlap. Parent timers include children and must not be summed.", "",
             "## Scene timings", "",
             "| Scene | Frame mean | p95 | CPU work | GPU | Wait | FPS | Repeat mean range |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, scene in scenes.items():
        a = scene["averages_ms"]
        selected = [r for r in runs if r["scene"] == name]
        raw = [f["timings"]["frame_ms"] for r in selected for f in r["frames"]]
        means = [r["summary"]["frame_ms"]["mean"] for r in selected]
        lines.append(f"| {name} | {a['frame_ms']:.3f} | {percentile(raw, .95):.3f} | {a['cpu_work_ms']:.3f} | "
                     f"{a.get('gpu_frame_ms', float('nan')):.3f} | {a['wait_ms']:.3f} | {1000/a['frame_ms']:.0f} | "
                     f"{min(means):.3f}–{max(means):.3f} |")
    lines += ["", "## CPU work breakdown", "",
              "The three largest measured child scopes in each scene; scopes below are disjoint. "
              "Simulation worker integration and snapshot copy overlap these scopes.", "",
              "| Scene | Largest scope | Second | Third | Worker advance / copy |", "|---|---:|---:|---:|---:|"]
    cpu_scopes = {"cpu_ui_ms", "cpu_camera_ms", "cpu_simulation_ms", "cpu_gaia_ms", "cpu_orbits_ms"}
    for name, scene in scenes.items():
        a = scene["averages_ms"]
        children = sorted(((k, v) for k, v in a.items() if k in cpu_scopes or
                           (k.startswith("cpu_") and k.endswith(("_prepare_ms", "_setup_ms", "_draw_ms", "_cull_ms", "_composite_ms")))),
                          key=lambda item: -item[1])[:3]
        labels = [f"{k.removeprefix('cpu_').removesuffix('_ms')}: {v:.3f}" for k, v in children]
        lines.append(f"| {name} | {' | '.join(labels)} | {a['worker_advance_ms']:.3f} / {a['snapshot_copy_ms']:.3f} |")
    lines += ["", "## GPU work breakdown", "",
              "Major timestamp intervals. GPU pipeline overlap can move apparent cost between adjacent passes when a feature changes; "
              "do not sum them into a GPU budget. Prepare includes LUTs, refraction, ringshine, uploads and transparency clear. "
              "Atmosphere composite includes atmospheric shadows. Use whole-frame feature-removal deltas to confirm costs.", "",
              "| Scene | Prepare | Surfaces | Gaia cull + draw | Orbits | Atmosphere | Transparency resolve | Post + UI |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, scene in scenes.items():
        a = scene["averages_ms"]
        def val(key):
            return a.get(key, 0)
        numbers = [val("gpu_terrain_prepare_ms"), val("gpu_celestial_ms"), val("gpu_gaia_cull_ms") + val("gpu_gaia_ms"),
                   val("gpu_orbits_ms"), val("atmosphere_composite_ms"), val("transparency_resolve_ms"),
                   val("gpu_post_composite_ms") + val("gpu_ui_ms")]
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in numbers) + " |")
    lines += ["", "## GPU pass detail", "",
              "Per-frame cost across all measured frames, including frames with no dispatch. "
              "Shadow resolve and reprojection are children of atmospheric shadows; sky/aerial/observer work is inside prepare. "
              "A dash means the pass was not recorded.", "",
              "| Scene | Ring draw | Terrain draw | Atmosphere shadows | Shadow reprojection | Shadow resolve | Sky LUT | Aerial LUT | Observer table |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, scene in scenes.items():
        keys = ["rings_ms", "gpu_terrain_draw_ms", "atmosphere_shadows_ms", "atmosphere_shadow_reproject_ms",
                "atmosphere_shadow_resolve_ms", "atmosphere_sky_ms", "atmosphere_aerial_ms", "refraction_observer_ms"]
        a, counts = scene["averages_ms"], scene["timing_counts"]
        values = [f"{a[k]*counts[k]/scene['samples']:.3f}" if k in a else "—" for k in keys]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "## Feature removal experiments", "",
              "Positive savings mean faster with the feature disabled. These are measured feature costs, "
              "not promised savings from an equivalent-quality optimization. Small deltas near repeat variation are inconclusive.", "",
              "| Scene / change | Frame saving | CPU saving | GPU saving |", "|---|---:|---:|---:|"]
    experiments = []
    cadence = []
    for path in sorted(output.glob("*/report.json")):
        if path.parent.name == "baseline":
            continue
        report = load(path)
        validate(report)
        check_trace(baseline, report)
        if path.parent.name == "baseline-check":
            continue
        if report["configuration"]["step_seconds"] != config["step_seconds"]:
            cadence.append(report)
            continue
        for scene in report["scenes"]:
            name = scene["scene"]
            if name not in scenes:
                continue
            a = scene["averages_ms"]
            b = scenes[name]["averages_ms"]
            delta = {key: b.get(key, 0) - a.get(key, 0) for key in ("frame_ms", "cpu_work_ms", "gpu_frame_ms")}
            experiments.append({"capture": path.parent.name, "scene": name, "savings_ms": delta})
            lines.append(f"| {name} / {path.parent.name} | {delta['frame_ms']:+.3f} | "
                         f"{delta['cpu_work_ms']:+.3f} | {delta['gpu_frame_ms']:+.3f} |")
    if cadence:
        lines += ["", "## Interactive cadence cross-check", "",
                  "The primary suite advances simulation by a fixed 16.667 ms per rendered frame, independent of frame time. "
                  "These additional captures use a step close to measured interactive frame duration at 1×. "
                  "They intentionally change the simulation/camera trace and are not feature-removal comparisons.", "",
                  "| Scene | Simulation step ms | Frame ms | CPU ms | GPU ms |", "|---|---:|---:|---:|---:|"]
        for report in cadence:
            for scene in report["scenes"]:
                a = scene["averages_ms"]
                lines.append(f"| {scene['scene']} | {1000*report['configuration']['step_seconds']:.3f} | "
                             f"{a['frame_ms']:.3f} | {a['cpu_work_ms']:.3f} | {a['gpu_frame_ms']:.3f} |")
    lines += ["", "## Workload and cache checks", "",
              "Per-frame means over measurement only. A nonzero pending count or upload count indicates remaining streaming. "
              "Moving camera and running simulation deliberately invalidate some caches.", "",
              "| Scene | Terrain patches | Texture pending / uploads | Normal pending / uploads | Sky / aerial updates | Shadow resets / rotation reuse |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in scenes:
        frames = [f for r in runs if r["scene"] == name for f in r["frames"]]
        def counter(key):
            return statistics.mean(f["counters"].get(key, 0) for f in frames)
        lines.append(f"| {name} | {counter('terrain_patches'):.0f} | {counter('texture_pending'):.2f} / {counter('texture_uploads'):.2f} | "
                     f"{counter('normal_pending'):.2f} / {counter('normal_uploads'):.2f} | "
                     f"{counter('sky_updates'):.2f} / {counter('aerial_updates'):.2f} | "
                     f"{counter('shadow_resets'):.2f} / {counter('shadow_rotation_updates'):.2f} |")
    lines += ["", "## Scope and raw data", "",
              "- `analytical` matches the normal loaded overview with running propagation; `overview` pauses the same view. `nbody` uses IAS15 at 1× warp.",
              "- Ground views are 2 km above the oblate surface, aimed 3° above the horizon. Turning views yaw ±15° with a fixed eye.",
              "- Earth ring views use the procedural generator's default configuration and seed 42 (1.2–2.3 radii), applied in memory. Saved systems are unchanged.",
              "- `earth-disc` is at 2.5 radii; `earth-full` at 1.1 radii; `earth-limb` aims at the tangent; Saturn is at 5 radii.",
              "- Startup/compilation, asset hashing, warmup, final GPU drain and JSON/CSV writing are excluded. This is steady-state application throughput.",
              "- GPU timestamps measure command intervals, not utilization, occupancy, bandwidth or hardware busy time. Counters and instrumentation add overhead.",
              "- `observer_builds` is a cumulative renderer counter; use adjacent-frame differences, within each repetition, to count rebuilds.",
              "- Raw measurements, hardware/driver/build identities, fixtures and per-run percentiles are in each capture's `report.json` and `frames.csv`.", ""]
    check_path = output / "baseline-check" / "report.json"
    if check_path.exists():
        check = load(check_path)
        lines += ["## Baseline drift check", "", "The full suite was repeated after the feature experiments.", "",
                  "| Scene | Initial frame ms | Final frame ms | Change |", "|---|---:|---:|---:|"]
        for scene in check["scenes"]:
            first = scenes[scene["scene"]]["averages_ms"]["frame_ms"]
            last = scene["averages_ms"]["frame_ms"]
            lines.append(f"| {scene['scene']} | {first:.3f} | {last:.3f} | {(last/first-1)*100:+.1f}% |")
    (output / "timings.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "experiments.json").write_text(json.dumps(experiments, indent=2), encoding="utf-8")
    print(f"Report: {output / 'timings.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exe", type=Path, default=ROOT / "build/bin/StellarForge.exe")
    parser.add_argument("--phase", choices=["baseline", "experiments", "cadence", "summary"], default="baseline")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.exe = args.exe.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "baseline":
        capture(args, "baseline", "audit")
    if args.phase == "experiments":
        experiments = {
            "analytical": ["no-secondary", "no-orbits", "no-gaia", "no-refraction", "no-rings"],
            "earth-ground-rings": ["no-shadows", "no-refraction", "no-atmosphere", "no-secondary", "smooth"],
            "earth-full": ["no-atmosphere", "no-secondary", "smooth", "no-textures", "no-normals"],
            "earth-ground-rings-turn": ["no-shadows", "no-refraction"],
            "earth-ground-rings-live": ["no-shadows"],
        }
        for scene, variants in experiments.items():
            for variant in variants:
                capture(args, f"{scene}-{variant}", scene, variant)
        capture(args, "earth-full-800x450", "earth-full", width=800, height=450)
        capture(args, "earth-ground-rings-800x450", "earth-ground-rings", width=800, height=450)
        capture(args, "earth-ground-rings-turn-800x450", "earth-ground-rings-turn", width=800, height=450)
        capture(args, "analytical-no-ui", "analytical", ui=False)
        capture(args, "baseline-check", "audit")
    if args.phase == "cadence":
        capture(args, "analytical-interactive-cadence", "analytical", step=0.0006)
        capture(args, "earth-ground-live-interactive-cadence", "earth-ground-live", step=0.0012)
        capture(args, "earth-ground-rings-live-interactive-cadence", "earth-ground-rings-live", step=0.004)
    summarize(args.output)


if __name__ == "__main__":
    main()
