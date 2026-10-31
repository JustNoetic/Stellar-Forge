# Performance benchmarks

Build and run the normal application with a deterministic fixture:

```powershell
.\run.bat --benchmark
```

The default suite runs `overview`, `approach`, `analytical`, and `nbody`, with
300 warmup frames followed by 1200 measured frames, repeated three times per
scene. The framebuffer is 1920x1080. Validation and VSync are disabled, and the
actual presentation mode is recorded. A new directory under
`exports/benchmarks/<UTC timestamp>/` contains a short `report.txt` and a compact
`report.json`. The terminal prints the same readable summary: one average per
scene across all measured frames and repetitions, plus GPU pass averages.
Raw frames and CSV are exported only with `--benchmark-detailed`.
Existing output directories are rejected to preserve baselines. Escape or closing
the window cancels the capture with a nonzero exit and no complete report.

For a quick capture:

```powershell
.\build\bin\StellarForge.exe --benchmark --benchmark-warmup 60 --benchmark-frames 300 --benchmark-repeats 2 --benchmark-output exports/benchmarks/baseline
```

Use `--help` for all options. `--benchmark-hidden` hides the window but still
requires Vulkan and a window system; compare hidden captures only with hidden
captures. `--benchmark-ui` adds the application workspace with disabled controls.
Saved ImGui layouts are neither loaded nor written. Camera input is ignored during
capture. Resizing, minimizing a visible window, mismatched framebuffer dimensions,
simulation failures and missing GPU results reject the run.

## Fixtures

For a slow live camera position, open **View > Engine Telemetry**. Its scene
breakdown records preparation, celestial bodies (including terrain and sphere /
point children), Gaia culling/drawing, orbit lines and atmospheric composition.
These scopes are included in Scene HDR; child timings must not be added to their
parents. Sky/aerial update counts distinguish cached frames from rebaking.
The additional scene timestamps are enabled only while telemetry is open or
during a benchmark; opening telemetry does not enable CPU benchmark collection.

| Scene | Simulation | Camera |
|---|---|---|
| `overview` | Paused Solar System | Tracks Sun, render-space offset `(0, 19.02, 40.78)` AU |
| `approach` | Paused Solar System | Tracks Earth along normalized `(0, 19.02, 40.78)`; logarithmic distance from 0.1 AU to four Earth equatorial radii over measured frames |
| `analytical` | Running Jacobi analytical propagation | Same overview camera |
| `nbody` | Running IAS15 | Same overview camera; integrator epsilon recorded |
| `spice` | Running local SPICE Ephemeris Mode | Same overview camera; explicit opt-in |
| `earth-full` | Paused analytical Solar System at time zero | Sunward Earth close-up at 1.1 Earth radii; fills the framebuffer |
| `earth-eclipse` | Paused analytical Solar System at 0.297293 years | Same distance; Moon shadow axis centered on Earth's surface |

All scenes use the default FOV and render settings, including Gaia stars and orbit
lines at 1030 segments. Each repetition reloads its initial system and resets its
camera. The time step is fixed at 1/60 second, independent of render speed;
`--benchmark-step` and `--benchmark-warp` set the workload explicitly. Running
scenes advance during warmup and continue from that epoch during measurement.
Approach warmup holds the starting distance. A one-frame approach capture samples
only its starting pose. These are deterministic workloads, not wall-time playback.
The two Earth close-ups are opt-in. See [eclipse_shadows.md](eclipse_shadows.md)
for their geometry, exact image checks and eclipse filtering measurements.

## Comprehensive application audit

### Imported terrain navigation

`--benchmark-scene terrain-navigation` runs `earth-terrain-still`,
`earth-terrain-walk`, `earth-terrain-fly`, and `earth-terrain-turn`. Individual
scene names are also accepted. These use the normal renderer and default settings
with paused simulation. They require the compatible imported Earth height source.
The fixture selects its highest daylight sample, looks 3° above the horizon, and
precomputes the camera path outside measured frames. Warmup holds the first pose.
Walking follows decoded triangle heights at 5 m/s with 1.7 m clearance; flying
uses 250 m/s with 2 km clearance. Turning holds the eye and rotates 15°/s.
The fixture records the source identity/revision, starting direction/height,
clearance, speed, distance and camera path version.

`--benchmark-variant terrain-detail` lowers the geometry error target from the
default 16 pixels to 2 pixels, preserving the default patch limit and other
settings. Use `--benchmark-detailed` for frame percentiles, streaming counters,
selection reuse/rebuild counts and CPU terrain preparation timing. The
`gpu_terrain_prepare` scope includes all celestial preparation; its name does
not isolate terrain work.

```powershell
.\build\bin\StellarForge.exe --benchmark --benchmark-scene terrain-navigation --benchmark-hidden --benchmark-ui --benchmark-warmup 600 --benchmark-frames 900 --benchmark-repeats 3 --benchmark-detailed --benchmark-output exports/benchmarks/terrain-navigation
```

### Audit suite

```powershell
python scripts/performance_audit.py --output exports/benchmarks/my-audit --phase baseline
python scripts/performance_audit.py --output exports/benchmarks/my-audit --phase experiments
python scripts/performance_audit.py --output exports/benchmarks/my-audit --phase cadence
python scripts/performance_audit.py --output exports/benchmarks/my-audit --phase summary
```

The audit uses the normal Release application at 1600×900 with UI, visible window,
VSync off and deterministic pipelined simulation. Defaults are 600 warmup frames,
1500 measured frames and three repetitions. Completed captures are retained;
incomplete directories are rejected. Run captures serially on an otherwise idle
machine. `timings.md` contains scene mean/p95, CPU and GPU breakdowns, repetition
variation, feature-removal deltas and streaming/cache counters. The raw JSON and
CSV remain alongside it.

`--benchmark-scene audit` includes the four original scenes plus these fixtures:

| Scene | Camera / workload |
|---|---|
| `earth-ground` | 2 km above the oblate surface, 3° above the horizon, fixed eye in a potential ring-shadow region |
| `earth-ground-rings` | Identical pose with the procedural ring generator's default configuration and seed 42 |
| `earth-ground-turn`, `earth-ground-rings-turn` | Fixed eye; yaw ±15°, sinusoidal phase advanced at the fixed timestep |
| `earth-disc` | Sunward at 2.5 equatorial radii, full disc visible |
| `earth-full` | Sunward at 1.1 radii, surface fills the framebuffer |
| `earth-limb` | Same distance, aimed at the tangent |
| `earth-full-rings` | 1.1 radii above the ring-shadow region |
| `earth-eclipse` | Moon-shadow close-up at `ECLIPSE_FIXTURE_YEARS` |
| `saturn` | Sunward at five Saturn radii |
| `earth-ground-live`, `earth-ground-rings-live`, `earth-full-live` | Corresponding views with analytical simulation running at 1× |

Ring fixtures use the same in-memory generator/application path as the UI; source
systems and textures are not saved. All views except `analytical`, `nbody` and
`*-live` are paused. Paused captures expose cached steady rendering; live and
turning captures expose update costs. They must be interpreted separately.

`--benchmark-variant` disables one feature group: `no-atmosphere`, `no-shadows`
(atmospheric shadows), `no-refraction`, `no-secondary` (planetshine and ringshine),
`no-orbits`, `no-gaia`, `smooth` (imported/synthetic terrain and normal maps),
`no-textures`, `no-normals`, `no-rings`, `no-ring-shadows` (surface lighting),
or `no-clouds`. `legacy-clouds` keeps clouds enabled and selects diagnostic legacy
transparency for comparison; it changes atmospheric endpoints and ordering.
`baseline` preserves default settings. Feature removal changes the image; deltas
are measured feature costs, not promises of equivalent-quality optimization.
`no-bloom` bypasses post-process bloom; `fast-bloom` selects the fast halo in
place of the default four-spike sprites. See `bloom.md`.
`baseline` now includes cached atmospheric shadow weights. `native-shadow-weights`
restores the original per-sample lighting; `cached-shadow-weights` explicitly
selects the current default. Cached weights preserve the ring filter and
sample positions but introduce a small LUT approximation; see
[cached weights](atmosphere_shadow_cached_weights.md) for limits and measurements.
`fast-ring-shadows` enables the separate [ring geometry experiment](atmosphere_ring_shadow_geometry.md),
which preserves the existing ring filter and simplifies its projection math.
The script also measures quarter-pixel-count rendering and UI removal.
The optional `cadence` phase cross-checks running views with fixed steps close to
their measured uncapped frame duration, instead of the primary 1/60-second step.
These deliberately change the camera/time trace and are reported separately.

CPU child scopes split refraction, albedo streaming, normal streaming, rings,
ringshine, terrain, planetshine, atmosphere preparation, transparency preparation,
eclipse refraction, surface setup, draw recording, Gaia culling and atmosphere
composition. They remain inside `cpu_record_ms` and must not be added to their
parent. `cpu_celestial_ms` includes prepare and opaque/ring rendering; atmosphere
composition is outside it. `gpu_terrain_prepare_ms` is the entire celestial
preparation interval, despite its historical name. New GPU child scopes separate
sphere, terrain, point and Gaia-cull work. Repeated timestamp names are summed
within a captured frame, so multi-atmosphere pass totals include every instance.

Raw frame `counters` retain terrain patch/bake counts, streaming pending/uploads/
failures, atmosphere LUT updates, shadow resets/rotation reuse, ringshine bakes,
refraction observer builds and camera orientation. Counts are separate from
millisecond metrics. `observer_builds` is cumulative; difference adjacent frames
within a repetition to measure rebuild frequency. Instrumentation is enabled only on benchmark paths for new
CPU/surface-draw timers. Compare captures with matching instrumentation.

```powershell
.\build\bin\StellarForge.exe --benchmark --benchmark-scene nbody --benchmark-warp 3600 --benchmark-output exports/benchmarks/nbody-1h
.\build\bin\StellarForge.exe --benchmark --benchmark-scene spice --benchmark-output exports/benchmarks/spice
```

SPICE requires the local `data/ephemeris.json` runtime/kernel configuration.
The runtime and all configured kernels are fingerprinted before measurement,
which can take time for large kernel sets. Source data is never modified.

## Measurements

`frame_ms` is the wall interval from event polling through submission/presentation
return. Appending the CPU frame sample, report serialization and the final GPU
drain occur outside that interval. GPU readback/capture bookkeeping is inside the
normal frame path and contributes measurement overhead. Startup, asset hashing, pipeline compilation and warmup are
excluded. First-use work that remains after warmup stays in the measured results.
This is application frame throughput, not a display scanout or input-latency metric.

| Timing | Scope |
|---|---|
| `cpu_simulation_ms` | `Simulation::advance`, including prepared render state |
| `cpu_camera_ms` | Scripted camera pose and normal camera update |
| `cpu_record_ms` | Render-target checks, UI, renderer preparation and command recording; excludes camera update, acquire/fence waits and end-frame submission/present |
| `cpu_ui_ms` | ImGui frame setup and workspace construction; child of `cpu_record_ms` |
| `cpu_celestial_ms`, `cpu_gaia_ms`, `cpu_orbits_ms` | Each renderer's CPU preparation and command recording, including its GPU timestamp commands; children of `cpu_record_ms` |
| `fence_wait_ms`, `acquire_wait_ms` | Separate Vulkan frame pacing waits |
| `submit_ms`, `present_ms` | Wall time inside queue submit/present calls |
| `gpu_frame_ms` | Timestamp interval across the frame commands, including barriers |
| `gpu_scene_hdr_ms` | HDR scene interval, including preparation, body/ring draws, Gaia, orbits, atmosphere and transparency |
| `gpu_celestial_ms`, `gpu_gaia_ms`, `gpu_orbits_ms` | Renderer GPU intervals |
| `gpu_terrain_prepare_ms` | Entire celestial preparation interval before dynamic rendering, including terrain, atmosphere LUTs, refraction, ringshine and transparency clear; included in HDR scene time |
| `gpu_post_composite_ms`, `gpu_ui_ms` | Composite and ImGui GPU intervals |

CPU scopes measure elapsed wall time, not scheduled thread/process CPU execution.
GPU timestamps are read only after the frame fence signals, paired using monotonic
submitted-frame IDs, and drained after each repetition. No per-frame device-idle
wait is added. Timestamp support and valid-bit wraparound are handled centrally in
the RHI. Unsupported GPU timings are omitted from JSON/CSV statistics, with blank
CSV cells; they are never fabricated as zero. An otherwise supported capture
requires all seven intervals for every measured frame.

Parent/child GPU intervals can overlap, and CPU/GPU execution also overlaps.
Do not add these timings together. GPU command intervals can include pipeline
overlap or CPU submission gaps; they are not hardware utilization or GPU busy
time. Timestamp instrumentation has overhead, especially for tiny passes. Use
the same instrumentation when comparing versions. See the
[Khronos timestamp guide](https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html).

The default report pools repetitions into one arithmetic mean per timing per
scene. FPS is `1000 / average frame time`. CPU work is simulation + camera + render
recording; wait time is fence + acquire + present. These scopes remain separate
from GPU time because CPU/GPU work overlaps.

The terrain-specific offscreen benchmark is available with render tests enabled:

```powershell
.\build\StellarRenderingTests.exe --terrain-benchmark scratch/terrain-benchmark/capture.json
```

It compares sphere, full procedural FP64, cached FP32 and cached FP64 vertex arithmetic,
including stationary, moving, cold-cache and distant telescope cases. It excludes
the rest of the application frame. See [terrain performance](terrain_performance.md).

With `--benchmark-detailed`, the report additionally includes per-run count, mean,
median, p95, p99, minimum and maximum, raw JSON frames, and `frames.csv`.
Percentiles use linear interpolation at `(count - 1) * percentile`.
Raw frames retain camera position, simulation time and frame ID. Metadata includes
CPU identity, logical processor count, OS family, GPU/driver, build type/compiler,
Git commit and dirty status at CMake configure time, source shader fingerprints,
Gaia identity, fixture settings and simulation accuracy. Fingerprints are FNV-1a
64-bit content checksums for reproducibility, not cryptographic attestations.
Reconfigure after switching commits; dirty builds require retaining their diff.

## Comparing captures

### Simulation worker prototype

Normal launches use the asynchronous simulation worker. Use
`run.bat --no-simulation-thread` for the inline fallback; `--simulation-thread`
remains an explicit enable flag. Deterministic benchmarks retain their inline
baseline default and select worker execution with `--benchmark-simulation`. The worker owns the simulation and publishes
independent snapshots with reused buffers; UI commands are synchronization barriers.
Normal playback accumulates pending elapsed time and renders the latest completed
state, without blocking on physics or interpolating between states.

`--benchmark-simulation inline|worker|pipeline` selects execution for deterministic
captures. `worker` waits for each step, isolating scheduling/publication overhead.
`pipeline` prefetches the next physics step while rendering the current complete
snapshot, then waits at the next frame boundary if necessary. All three complete
the same number of physics steps and render the same camera/time sequence. The
first pipeline step is primed in the first warmup frame. Pipeline benchmarks use
backpressure for matched work; they do not measure unconstrained interactive FPS
or prove input latency. They have no intentional interpolation or physics-rate cap.

`cpu_simulation_ms` measures the main thread's advance/wait/submit scope;
`worker_advance_ms` and `snapshot_copy_ms` separately measure worker integration
and snapshot preparation. Worker scopes overlap rendering and must not be added
to main-thread CPU work. A full-state copy remains O(body count), sharing immutable
definitions/hierarchies. Paused scenes skip worker advances and record zero worker
timings. The comparison script permits execution mode differences while still
rejecting mismatched physics and camera traces in detailed captures.

Run repeated comparisons in forward/reverse mode order, retaining raw frames:

```powershell
python scripts/benchmark_simulation_threads.py --output exports/benchmarks/thread-comparison
python scripts/benchmark_simulation_threads.py --visible --workload normal --step 0.000588235294117647 --output exports/benchmarks/thread-high-fps
```

The script runs 56-body Solar System IAS15 at 1600x900, with Gaia, orbit lines
and UI. It compares normal warp and 86400x warp by default, 3000 measured frames,
600 warmup frames, three repetitions and two blocks with alternate mode order.
Use `--no-ui`, `--frames`, `--repeats`, `--blocks` and `--heavy-warp` to vary the
experiment. Each capture is checked for matching fixtures and full camera/time
traces before `summary.md` and `summary.json` are written. Hidden windows can
introduce driver presentation throttling; visible captures are necessary for
assessing overhead near the application's uncapped high frame rates.

Recorded results and prototype limitations are in
[`simulation_thread_prototype.md`](simulation_thread_prototype.md).

```powershell
python scripts/compare_benchmarks.py exports/benchmarks/baseline exports/benchmarks/candidate
python scripts/compare_benchmarks.py exports/benchmarks/baseline exports/benchmarks/candidate --metric frame_ms --metric gpu_frame_ms
```

The comparison accepts report files or directories. It checks hardware/driver,
configuration, source data, fixtures and timing coverage. When both inputs contain
raw frames, camera/simulation traces are also checked. Build identities and shader
contents may change. It prints average frame, CPU and GPU timings and percentage
changes; use `--metric` to select other stages. Both the original large reports
and the new compact reports are supported. Positive changes mean
more time spent. There are no timing-based test failures or CI thresholds yet.

Use Release builds on the same otherwise idle machine, with matching power settings
and window visibility. Repeat captures and inspect variability before treating a
small change as a regression. A short smoke capture validates the harness; it is
not a performance conclusion. Startup/cold-cache behavior needs a separate test.

## Comparing the original application

The original project has `scripts/profile_cpu.py`, `scripts/profile_ground.py`,
and `docs/performance_profiling.md`. Its default ground/orbit fixtures enable
effects that this rewrite has not ported, so their default FPS is not comparable.
The new comparison script consumes this harness's schema; it does not launch or
convert the original Python profiler automatically.

For a matched original capture, use the same system/epoch/body set, camera trace,
FOV, actual framebuffer, Gaia catalog, orbit segment count, point-source settings,
simulation step/warp and UI visibility. Disable unported terrain, atmosphere,
clouds, bloom and secondary lighting in the original. Check the images for
comparable output before interpreting timings. For IAS15, match physical forces
and verify comparable trajectory accuracy as well as numerical tolerance; a
shared tolerance value alone does not prove equal accuracy. Compare SPICE using
the same kernels, targets and time sequence. Keep feature-matched comparisons
separate from full-feature application captures.

## Verification

With `STELLAR_BUILD_TESTS=ON`, `benchmark_contracts` covers argument validation,
statistics, missing GPU values and report exports. `benchmark_comparison` checks
comparison acceptance and rejection if Python is available. Existing GPU contract
tests and `--self-test` exercise the shared RHI path. A short capture with zero
warmup and one measured frame checks the final in-flight GPU result drain.
