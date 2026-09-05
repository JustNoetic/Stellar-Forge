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
  --gpu              Measure per-pass GPU times with GL timer queries
                     (GL_TIME_ELAPSED). Instrumented passes in app.py:
                     culling, ringshine map, spheres, orbits, atmosphere
                     (behind/front of rings), rings, HZ, TAA, bloom,
                     composite, ImGui.
  --target-body NAME Track NAME at a close 3.5R orbit instead of the default
                     45 AU solar view (default: "Saturn" when --gpu is used,
                     so the GPU gets real work: rings + atmosphere + ringshine).
                     Combine with --no-physics to isolate the render loop and
                     get a pure GPU-bound profile.

Environment variables
----------------------
  STELLAR_FORGE_PERF=1   Activate the in-process PerfTracker inside main.py
                         (instrumented sub-stages: pack_instances, pack_uniforms,
                         pack_rings, render_orbits, render_spheres, render_rings,
                         imgui_draw, etc.) — only used when --mode is set AND
                         the env var is exported.
  STELLAR_FORGE_GPU_PERF=1  Wrap the GPU render passes in GL timer queries and
                         record per-pass GPU milliseconds into the report.

Example
-------
  python scripts/perf_test.py --gpu --target-body Saturn --mode stress
  python scripts/perf_test.py --gpu --no-physics --mode quick   # GPU-only frame cost
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
        self.sections = {}  # name -> dict (CPU wall-clock)
        self.gpu_sections = {}  # name -> dict (GPU GL_TIME_ELAPSED)
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

    def record_gpu(self, name, dt):
        """Record a GPU pass duration (seconds) from a GL timer query."""
        with self._lock:
            s = self.gpu_sections.get(name)
            if s is None:
                s = {'total': 0.0, 'count': 0, 'min': float('inf'), 'max': 0.0, 'last': 0.0}
                self.gpu_sections[name] = s
            s['total'] += dt
            s['count'] += 1
            s['min'] = min(s['min'], dt)
            s['max'] = max(s['max'], dt)
            s['last'] = dt

    def clear_sections(self):
        """Drop CPU + GPU section stats (used after warmup to skip JIT/startup stalls)."""
        with self._lock:
            self.sections.clear()
            self.gpu_sections.clear()
            if hasattr(self, 'swap_times'):
                self.swap_times.clear()

    def _section_lines(self, table, total, title, note=""):
        lines = []
        if not table:
            return lines
        lines.append("")
        lines.append(f"--- {title} ---")
        if note:
            lines.append(f"  {note}")
        header = f"  {'name':<35s}  {'total ms':>10s}  {'count':>7s}  {'avg ms':>9s}  {'min ms':>9s}  {'max ms':>9s}  {'%':>5s}"
        lines.append(header)
        for name, s in sorted(table.items(), key=lambda x: -x[1]['total']):
            pct = (s['total'] / total * 100) if total > 0 else 0
            avg_ms = (s['total'] / s['count']) * 1000 if s['count'] > 0 else 0
            lines.append(
                f"  {name:<35s}  {s['total'] * 1000:10.2f}  {s['count']:7d}  "
                f"{avg_ms:9.3f}  {s['min'] * 1000:9.3f}  {s['max'] * 1000:9.3f}  {pct:5.1f}"
            )
        return lines

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
                if n < 30:
                    lines.append(f"  Note            : only {n} frames measured — "
                                 "use --mode stress or --fps-cap 0 for a larger sample")

                if self.gpu_sections:
                    gpu_total = sum(s['total'] for s in self.gpu_sections.values())
                    gpu_per_frame_ms = gpu_total / n * 1000
                    frame_ms = avg * 1000
                    
                    avg_swap_ms = 0.0
                    if hasattr(self, 'swap_times') and self.swap_times:
                        avg_swap_ms = (sum(self.swap_times) / len(self.swap_times)) * 1000
                    cpu_app_ms = frame_ms - avg_swap_ms
                    
                    if gpu_per_frame_ms >= 0.85 * cpu_app_ms:
                        verdict = f"GPU-bound (GPU {gpu_per_frame_ms:.1f}ms > CPU app {cpu_app_ms:.1f}ms)"
                    else:
                        verdict = f"CPU-bound (CPU app {cpu_app_ms:.1f}ms > GPU {gpu_per_frame_ms:.1f}ms)"
                    
                    lines.append("")
                    lines.append(f"GPU work / frame : {gpu_per_frame_ms:7.2f} ms")
                    lines.append(f"CPU app work     : {cpu_app_ms:7.2f} ms  (Swap block: {avg_swap_ms:7.2f} ms)")
                    lines.append(f"Bottleneck       : {verdict}")

            if self.sections:
                total = sum(s['total'] for s in self.sections.values())
                lines += self._section_lines(self.sections, total,
                    "Per-section CPU cost (sorted by total time)")
            if self.gpu_sections:
                gpu_total = sum(s['total'] for s in self.gpu_sections.values())
                lines.append("")
                lines.append("--- GPU Category Breakdown (GL_TIME_ELAPSED) ---")
                lines.append("  Summed per category; reads are deferred to the next frame boundary.")
                lines.append(f"  {'Category / Pass':<35s}  {'total ms':>10s}  {'count':>7s}  {'avg ms':>9s}  {'% of GPU':>9s}")
                
                categories = [
                    ("Atmosphere Rendering", ["gpu_atmo_behind", "gpu_atmo_front"]),
                    ("Celestial Bodies & Geo", ["gpu_spheres", "gpu_orbits", "gpu_hz"]),
                    ("Rings & Shadows", ["gpu_rings", "gpu_ringshine_map"]),
                    ("Post-Processing", ["gpu_bloom", "gpu_conv_bloom", "gpu_composite", "gpu_taa"]),
                    ("Overhead & UI", ["gpu_culling", "gpu_imgui"])
                ]
                
                categorized_keys = set(k for _, keys in categories for k in keys)
                uncategorized_keys = [k for k in self.gpu_sections.keys() if k not in categorized_keys]
                if uncategorized_keys:
                    categories.append(("Other Passes", uncategorized_keys))

                for cat_name, keys in categories:
                    valid_keys = [k for k in keys if k in self.gpu_sections]
                    if not valid_keys:
                        continue
                    
                    cat_total = sum(self.gpu_sections[k]['total'] for k in valid_keys)
                    cat_count = max(sum(self.gpu_sections[k]['count'] for k in valid_keys), 1)
                    
                    n_frames = len(self.frame_times) if self.frame_times else 1
                    cat_avg_ms = (cat_total / n_frames) * 1000
                    cat_pct = (cat_total / gpu_total * 100) if gpu_total > 0 else 0
                    
                    lines.append(f"  [{cat_name}]".ljust(37) + f"{cat_total * 1000:10.2f}  {cat_count:7d}  {cat_avg_ms:9.3f}  {cat_pct:8.1f}%")
                    
                    valid_keys.sort(key=lambda k: -self.gpu_sections[k]['total'])
                    for idx, k in enumerate(valid_keys):
                        s = self.gpu_sections[k]
                        k_total = s['total']
                        k_count = s['count']
                        k_avg_ms = (k_total / k_count) * 1000 if k_count > 0 else 0
                        k_pct = (k_total / gpu_total * 100) if gpu_total > 0 else 0
                        prefix = "    \\-- " if idx == len(valid_keys) - 1 else "    |-- "
                        name_display = f"{prefix}{k}"
                        lines.append(f"  {name_display:<35s}  {k_total * 1000:10.2f}  {k_count:7d}  {k_avg_ms:9.3f}  {k_pct:8.1f}%")
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
    ap.add_argument("--gpu", action="store_true", help="Measure per-pass GPU times with GL timer queries")
    ap.add_argument("--target-body", default=None, help="Track this body at a close 3.5R orbit (default: 'Saturn' when --gpu is used)")
    args = ap.parse_args()

    durations = {"quick": (3, 10), "standard": (5, 20), "stress": (3, 60)}
    warmup, measure = durations[args.mode]
    total_runtime = warmup + measure

    # Activate inner-loop instrumentation if requested.
    if os.environ.get("STELLAR_FORGE_PERF") != "1":
        os.environ["STELLAR_FORGE_PERF"] = "1"
    if args.gpu:
        os.environ["STELLAR_FORGE_GPU_PERF"] = "1"

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

    # Force fullscreen by monkey-patching glfw.create_window
    original_create_window = glfw.create_window
    def patched_create_window(width, height, title, monitor, share):
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        window = original_create_window(mode.size.width, mode.size.height, title, monitor, share)
        if window:
            app_instance.window_width = mode.size.width
            app_instance.window_height = mode.size.height
            app_instance.fb_width, app_instance.fb_height = glfw.get_framebuffer_size(window)
        return window
    glfw.create_window = patched_create_window

    # Instantiate the application.
    app_instance = ap_mod.App()

    # Move the camera to a heavy scene (tracked body with rings + atmosphere).
    target_body = args.target_body if args.target_body is not None else "Saturn"
    if target_body:
        app_instance._perf_target_body = target_body

    # Apply options to time_ctrl before starting the app.
    if args.sim_speed != 1.0:
        app_instance.time_ctrl["multiplier"] = args.sim_speed
    if args.no_physics:
        app_instance.time_ctrl["paused"] = True
    else:
        app_instance.time_ctrl["paused"] = False

    # Set up frame-time capture by patching swap_buffers.
    # The warmup timer starts at the FIRST swap: App() boot time (textures,
    # Numba JIT, shader compilation) can exceed the warmup window, which would
    # otherwise close the window before any frame was measured.
    warmup_done = [False]
    warmup_start = [None]
    last_swap = [time.perf_counter()]
    original_swap = glfw.swap_buffers

    def patched_swap(window):
        now = time.perf_counter()
        if warmup_start[0] is None:
            warmup_start[0] = now
            last_swap[0] = now
            
            # Hardcoded FOV of 45 degrees and look at Saturn's equator
            app_instance.camera["fov"] = 45.0
            idx = app_instance.camera.get("tracking_idx")
            if idx is not None and hasattr(app_instance, '_bundle'):
                bodies_data = app_instance._bundle['bodies_data']
                if idx < len(bodies_data) and str(bodies_data[idx].get("name", "")).strip().lower() == "saturn":
                    _rad = float(app_instance._bundle['visual_data'][idx][3]) if idx < len(app_instance._bundle['visual_data']) else 0.0
                    _dist = max(_rad * 3.5, _rad + 1e-9)
                    import engine.core.math_utils as mu
                    import numpy as np
                    b = bodies_data[idx]
                    pole_ra = float(b.get("pole_ra", 0.0))
                    pole_dec = float(b.get("pole_dec", 90.0))
                    pole_vec = mu.pole_to_ecliptic(pole_ra, pole_dec)
                    up = np.array([0.0, 0.0, 1.0], dtype='f8')
                    if abs(np.dot(pole_vec, up)) > 0.99:
                        up = np.array([0.0, 1.0, 0.0], dtype='f8')
                    equator_vec = np.cross(pole_vec, up)
                    equator_vec = equator_vec / max(np.linalg.norm(equator_vec), 1e-9)
                    app_instance.camera["cam_pos_rel"] = equator_vec * _dist
                    
            return original_swap(window)
        if warmup_done[0]:
            tracker.record_frame(now - last_swap[0])
        last_swap[0] = now

        elapsed = now - warmup_start[0]
        if not warmup_done[0] and elapsed >= warmup:
            warmup_done[0] = True
            tracker.clear_sections()  # Clear JIT stalls accumulated during warmup
            print(f"[Perf] Warmup complete at {elapsed:.1f}s")
            print(f"[Perf] Starting measurement window of {measure}s ...")
        if warmup_done[0] and elapsed >= total_runtime:
            print(f"[Perf] Measurement window complete ({measure}s). Stopping ...")
            glfw.set_window_should_close(window, True)
            
        swap_t0 = time.perf_counter()
        res = original_swap(window)
        swap_dt = time.perf_counter() - swap_t0
        
        if warmup_done[0]:
            with tracker._lock:
                if not hasattr(tracker, 'swap_times'):
                    tracker.swap_times = []
                tracker.swap_times.append(swap_dt)
                
        return res

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
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n[Perf] Report saved to {args.output}")
    except OSError as e:
        print(f"[Perf] Failed to save report: {e}")
    if args.profile:
        print("\n--- cProfile top 30 (by tottime) ---")
        stats.print_stats(30)


if __name__ == "__main__":
    main()
