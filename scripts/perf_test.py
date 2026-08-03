"""
Performance audit harness for Stellar-Forge.

Measures render-loop frame time AND per-sub-system CPU cost so that bottlenecks
can be identified from a single run.

Modes
-----
  --mode quick       3s warmup, 10s measurement (default)
  --mode standard    5s warmup, 20s measurement
  --mode stress      3s warmup, 60s measurement (longest, most stable averages)

Options
-------
  --profile          Enable cProfile (significantly slows the run; only use
                     when you need call-graph attribution of total time)
  --output FILE      Save report to FILE (default: perf_report.txt)
  --disable-rings    Skip ring rendering
  --no-physics       Pause the physics thread (render-loop only benchmark)
  --fps-cap N        Cap framerate at N (default: uncapped)
  --sim-speed X      Set the time multiplier (default: 1.0)

Environment variables
---------------------
  STELLAR_FORGE_PERF=1   Activate the in-process PerfTracker inside main.py
                         (instrumented sub-stages: pack_instances, pack_uniforms,
                         pack_rings, render_orbits, render_spheres, render_rings,
                         imgui_draw, etc.) — only used when --mode is set AND
                         the env var is exported.
"""
import argparse
import cProfile
import pstats
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../engine')))
import time


class PerfTracker:
    """Cumulative per-section timings (seconds, count, min, max, last).

    Used both by perf_test.py (for outer timings) and by main.py
    (for inner-loop sub-stages, when STELLAR_FORGE_PERF=1 is set).
    """

    def __init__(self):
        self.sections = {}  # name -> dict
        self.frame_times = []  # list of seconds (per-frame deltas)
        import threading
        self._local = threading.local()
        self._lock = threading.Lock()

    def begin(self, name):
        # Auto-close any open section so timings are not lost on a missed end().
        open_name = getattr(self._local, 'open_name', None)
        if open_name is not None:
            self.end()
        self._local.open_name = name
        self._local.open_time = time.perf_counter()

    def end(self):
        open_name = getattr(self._local, 'open_name', None)
        if open_name is None:
            return
        dt = time.perf_counter() - self._local.open_time
        with self._lock:
            s = self.sections.get(open_name)
            if s is None:
                s = {'total': 0.0, 'count': 0, 'min': float('inf'), 'max': 0.0, 'last': 0.0}
                self.sections[open_name] = s
            s['total'] += dt
            s['count'] += 1
            s['min'] = min(s['min'], dt)
            s['max'] = max(s['max'], dt)
            s['last'] = dt
        self._local.open_name = None

    def record_frame(self, dt):
        with self._lock:
            self.frame_times.append(dt)

    def report(self):
        with self._lock:
            lines = ["=== Stellar-Forge Performance Report ==="]
            if self.frame_times:
                n = len(self.frame_times)
                avg = sum(self.frame_times) / n
                sorted_ft = sorted(self.frame_times)
                p1 = sorted_ft[max(0, int(n * 0.01))]
                p50 = sorted_ft[min(n - 1, int(n * 0.50))]
                p99 = sorted_ft[min(n - 1, int(n * 0.99))]
                lines.append("")
                lines.append(f"Frames measured : {n}")
                lines.append(f"Avg frame time  : {avg * 1000:7.2f} ms  ({1 / avg:5.1f} FPS)")
                lines.append(f"Min frame time  : {min(self.frame_times) * 1000:7.2f} ms  ({1 / min(self.frame_times):5.1f} FPS)")
                lines.append(f"Max frame time  : {max(self.frame_times) * 1000:7.2f} ms  ({1 / max(self.frame_times):5.1f} FPS)")
                lines.append(f"50%ile          : {p50 * 1000:7.2f} ms  ({1 / p50:5.1f} FPS)")
                lines.append(f" 1% low (99%ile): {p99 * 1000:7.2f} ms  ({1 / p99:5.1f} FPS)")
                lines.append(f" 1% high ( 1%ile): {p1 * 1000:7.2f} ms  ({1 / p1:5.1f} FPS)")
            if self.sections:
                total = sum(s['total'] for s in self.sections.values())
                lines.append("")
                lines.append("--- Per-section CPU cost (sorted by total time) ---")
                header = f"  {'section':<35s}  {'total ms':>10s}  {'count':>7s}  {'avg ms':>9s}  {'min ms':>9s}  {'max ms':>9s}  {'%':>5s}"
                lines.append(header)
                for name, s in sorted(self.sections.items(), key=lambda x: -x[1]['total']):
                    pct = (s['total'] / total * 100) if total > 0 else 0
                    avg_ms = (s['total'] / s['count']) * 1000 if s['count'] > 0 else 0
                    lines.append(
                        f"  {name:<35s}  {s['total'] * 1000:10.2f}  {s['count']:7d}  "
                        f"{avg_ms:9.3f}  {s['min'] * 1000:9.3f}  {s['max'] * 1000:9.3f}  {pct:5.1f}"
                    )
            return "\n".join(lines)


def patch_function(module, name, tracker, label):
    """Wrap module.<name> with a tracker.begin(label)/end() pair."""
    if not hasattr(module, name):
        return
    orig = getattr(module, name)
    def wrapped(*args, **kwargs):
        tracker.begin(label)
        try:
            return orig(*args, **kwargs)
        finally:
            tracker.end()
    wrapped.__wrapped__ = orig
    setattr(module, name, wrapped)


def main():
    ap_mod = None
    ap = argparse.ArgumentParser(description="Stellar-Forge performance audit harness")
    ap.add_argument("--mode", default="standard", choices=["quick", "standard", "stress"])
    ap.add_argument("--profile", action="store_true", help="Enable cProfile (slows run)")
    ap.add_argument("--output", default="perf_report.txt")
    ap.add_argument("--disable-rings", action="store_true", help="Skip ring rendering")
    ap.add_argument("--no-physics", action="store_true", help="Pause physics thread")
    ap.add_argument("--fps-cap", type=float, default=0.0, help="Cap framerate (0=uncapped)")
    ap.add_argument("--sim-speed", type=float, default=1.0, help="Initial time multiplier")
    args = ap.parse_args()

    durations = {"quick": (3, 10), "standard": (5, 20), "stress": (3, 60)}
    warmup, measure = durations[args.mode]
    total_runtime = warmup + measure

    # Activate inner-loop instrumentation if requested.
    if os.environ.get("STELLAR_FORGE_PERF") != "1":
        os.environ["STELLAR_FORGE_PERF"] = "1"

    import engine.app as ap_mod
    import engine.physics.physics_core as pc
    import engine.physics.kepler_analytical as ka
    import engine.rendering.planetshine as ps
    import engine.ephemeris.spice_manager as sm
    import glfw

    tracker = PerfTracker()
    if getattr(ap_mod, "_PERF_TRACKER", None) is not None:
        tracker = ap_mod._PERF_TRACKER
    else:
        ap_mod._PERF_TRACKER = tracker
        if hasattr(ap_mod, "_PERF_INSTALL_TRACKER"):
            ap_mod._PERF_INSTALL_TRACKER(tracker)

    # Patch CPU hot-path functions across physics, math, rendering and app modules.
    modules_to_patch = [pc, ka, ps, sm, ap_mod]
    patch_targets = [
        ("ias15_step_numba", "ias15_step_numba"),
        ("compute_all_accelerations", "compute_all_accelerations"),
        ("compute_custom_forces", "compute_custom_forces"),
        ("compute_barycenters", "compute_barycenters"),
        ("compute_all_orbits_batch", "compute_all_orbits_batch"),
        ("update_hierarchy", "update_hierarchy"),
        ("propagate_keplerian_system_numba", "propagate_keplerian_system_numba"),
        ("extract_all_kepler_elements", "extract_all_kepler_elements"),
        ("compute_planetshine_numba", "compute_planetshine_numba"),
        ("compute_body_rotation_angles_jit", "compute_body_rotation_angles_jit"),
        ("populate_states_fast", "populate_states_fast"),
    ]
    for fn_name, label in patch_targets:
        for m in modules_to_patch:
            if hasattr(m, fn_name):
                patch_function(m, fn_name, tracker, label)

    # Instantiate the application.
    app_instance = ap_mod.App()

    # Apply options to time_ctrl before starting the app.
    if args.sim_speed != 1.0:
        app_instance.time_ctrl["multiplier"] = args.sim_speed
    if args.no_physics:
        app_instance.time_ctrl["paused"] = True
    else:
        app_instance.time_ctrl["paused"] = False

    # Set up frame-time capture by patching swap_buffers.
    warmup_done = [False]
    warmup_start = time.perf_counter()
    last_swap = [time.perf_counter()]
    original_swap = glfw.swap_buffers

    def patched_swap(window):
        now = time.perf_counter()
        if warmup_done[0]:
            tracker.record_frame(now - last_swap[0])
        last_swap[0] = now

        elapsed = now - warmup_start
        if not warmup_done[0] and elapsed >= warmup:
            warmup_done[0] = True
            tracker.sections.clear()  # Clear JIT stalls accumulated during warmup
            print(f"[Perf] Warmup complete at {elapsed:.1f}s")
            print(f"[Perf] Starting measurement window of {measure}s ...")
        if warmup_done[0] and elapsed >= total_runtime:
            print(f"[Perf] Measurement window complete ({measure}s). Stopping ...")
            glfw.set_window_should_close(window, True)
        return original_swap(window)

    glfw.swap_buffers = patched_swap

    # Optional FPS cap (sleep at the start of each frame).
    if args.fps_cap > 0:
        min_dt = 1.0 / args.fps_cap
        original_poll = glfw.poll_events
        def capped_poll():
            now = time.perf_counter()
            wait = min_dt - (now - last_swap[0])
            if wait > 0:
                time.sleep(wait)
            return original_poll()
        glfw.poll_events = capped_poll

    # Run the application.
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
    app_instance.run()
    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')

    report = tracker.report()
    print("\n" + report)
    try:
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\n[Perf] Report saved to {args.output}")
    except OSError as e:
        print(f"[Perf] Failed to save report: {e}")
    if args.profile:
        print("\n--- cProfile top 30 (by tottime) ---")
        stats.print_stats(30)


if __name__ == "__main__":
    main()
