import argparse
import json
import statistics
import subprocess
from pathlib import Path

from compare_benchmarks import compatible, load


def main():
    parser = argparse.ArgumentParser(description="Compare inline and worker simulation with identical rendered physics states.")
    parser.add_argument("--exe", type=Path, default=Path("build/StellarForge.exe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heavy-warp", type=float, default=86400)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--step", type=float, default=1 / 60)
    parser.add_argument("--workload", choices=("normal", "heavy", "both"), default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    modes = ["inline", "worker", "pipeline"]
    for block in range(args.blocks):
        order = modes if block % 2 == 0 else list(reversed(modes))
        for workload, warp in (("normal", 1.0), ("heavy", args.heavy_warp)):
            if args.workload != "both" and args.workload != workload:
                continue
            baseline = None
            captures = []
            for mode in order:
                output = args.output / f"{workload}-{block + 1}-{mode}"
                command = [str(args.exe.resolve()), "--benchmark",
                           "--benchmark-detailed", "--benchmark-scene", "nbody", "--benchmark-simulation", mode,
                           "--benchmark-width", "1600", "--benchmark-height", "900", "--benchmark-warp", str(warp),
                           "--benchmark-frames", str(args.frames), "--benchmark-warmup", str(args.warmup),
                           "--benchmark-repeats", str(args.repeats), "--benchmark-output", str(output.resolve())]
                command += ["--benchmark-step", str(args.step)]
                if not args.visible:
                    command.append("--benchmark-hidden")
                if not args.no_ui:
                    command.append("--benchmark-ui")
                print(f"Capturing {workload}, block {block + 1}, {mode}", flush=True)
                completed = subprocess.run(command, capture_output=True, text=True)
                output.mkdir(parents=True, exist_ok=True)
                (output / "console.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(f"Capture failed: {output}/console.txt")
                report = load(output)
                captures.append(report)
                if mode == "inline":
                    baseline = report
                records.append({"workload": workload, "block": block + 1, "mode": mode, "path": str(output),
                                "averages_ms": report["scenes"][0]["averages_ms"]})
            for report in captures:
                compatible(baseline, report)
    summary = {"captures": records, "results": []}
    lines = [f"Simulation thread prototype: matched 56-body IAS15, 1600x900, UI={not args.no_ui}, visible={args.visible}, Gaia and orbit lines.",
             f"{args.blocks} blocks, reversed execution order in alternate blocks; {args.repeats} repeats x {args.frames} measured frames per capture.",
             "All captures passed full camera and simulation trace compatibility checks.", "",
             "| Workload | Mode | Mean frame ms | FPS | Change vs inline | Main sim us | Worker step us | Copy us | p99 frame ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for workload in ("normal", "heavy"):
        if args.workload != "both" and args.workload != workload:
            continue
        baseline_ms = statistics.mean(r["averages_ms"]["frame_ms"] for r in records
                                      if r["workload"] == workload and r["mode"] == "inline")
        for mode in modes:
            matching = [r for r in records if r["workload"] == workload and r["mode"] == mode]
            averages = {key: statistics.mean(r["averages_ms"][key] for r in matching)
                        for key in matching[0]["averages_ms"]}
            frame_values = []
            run_means = []
            for record in matching:
                report = load(record["path"])
                for run in report["runs"]:
                    run_means.append(run["summary"]["frame_ms"]["mean"])
                    frame_values.extend(frame["timings"]["frame_ms"] for frame in run["frames"])
            frame_values.sort()
            p99 = frame_values[int((len(frame_values) - 1) * .99)]
            ms = averages["frame_ms"]
            change = (ms / baseline_ms - 1) * 100
            summary["results"].append({"workload": workload, "mode": mode, "averages_ms": averages,
                                       "fps": 1000 / ms, "frame_change_percent": change, "p99_frame_ms": p99,
                                       "run_frame_means_ms": run_means})
            lines.append(f"| {workload} | {mode} | {ms:.4f} | {1000 / ms:.0f} | {change:+.2f}% | "
                         f"{averages['cpu_simulation_ms'] * 1000:.2f} | {averages['worker_advance_ms'] * 1000:.2f} | "
                         f"{averages['snapshot_copy_ms'] * 1000:.2f} | {p99:.4f} |")
    text = "\n".join(lines) + "\n"
    (args.output / "summary.md").write_text(text, encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
