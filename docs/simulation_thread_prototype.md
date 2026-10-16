# Simulation worker prototype — 2026-10-16

The worker is now the default for ordinary launches. Use
`run.bat --no-simulation-thread` for the inline fallback. The measurements below
were collected before making the worker the default. `SimulationRunner` owns a sleeping worker, a coalesced elapsed
time mailbox and three reusable snapshot buffers. Only the worker touches the
live simulation. Rendering reads its own complete snapshot without locking on
each body access. Definitions and immutable hierarchies are shared; changing
arrays are copied on the worker. UI edits, pause/model changes and replacement
systems are synchronization barriers, preserving existing validation results.

## High frame rate measurements

Release / GNU 16.2.0, Core i5-12400F, RTX 4060 Ti / NVIDIA 617.42, Vulkan 1.4.351.
Visible 1600x900 window, immediate presentation, VSync/validation disabled,
56-body Solar System IAS15, Gaia, orbit lines and UI. Simulation delta is
1/1700 second at 1x warp. Each mode has six measured runs of 5000 frames,
600 warmup frames per run, split into two blocks in reversed execution order.
Every rendered camera/time trace matches across modes. No samples were removed.

| Execution | Mean frame time | FPS | Frame-time change | p99 frame time |
|---|---:|---:|---:|---:|
| Inline | 0.5167 ms | 1935 | baseline | 0.8683 ms |
| Worker, waiting every frame | 0.5513 ms | 1814 | +6.70% | 0.9088 ms |
| Worker, overlapping rendering | 0.4991 ms | 2003 | -3.40% | 0.8673 ms |

Waiting every frame exposes scheduling overhead and loses overlap. The pipeline
preserves throughput near 2000 FPS, with essentially unchanged p99 frame time.
The small average improvement should be read as broadly similar throughput on
this machine, rather than a universal speedup. Individual run means are retained
in the summary; the waiting-worker result includes one slower 0.6336 ms run.

The pipeline's main-thread simulation handoff averages 2.96 microseconds;
snapshot preparation averages 0.38 microseconds on the worker. Worker physics
averages 230.60 microseconds, versus 221.86 microseconds inline. These are wall
times, include scheduling/cache effects, and overlap other work. The copy and
handoff measurements are not total process CPU cost. No CPU utilization, power
or end-to-end input latency claim is made.

Raw reports, per-run statistics and all frames:
`exports/benchmarks/simulation-thread-visible-20261016/`.

## High time warp measurements

The same visible scene at 86400x warp uses a fixed 1/60-second delta, advancing
1440 simulated seconds per rendered frame. Each mode has six measured runs of
3000 frames, with 600 warmup frames per run and reversed mode order between
blocks. Camera/time traces again match across all modes.

| Execution | Mean frame time | FPS | Frame-time change | p99 frame time |
|---|---:|---:|---:|---:|
| Inline | 0.8914 ms | 1122 | baseline | 1.2956 ms |
| Worker, waiting every frame | 0.9456 ms | 1057 | +6.09% | 1.3311 ms |
| Worker, overlapping rendering | 0.6385 ms | 1566 | -28.36% | 0.9419 ms |

Overlapping physics and rendering raises throughput by approximately 39.6% in
this workload. Physics itself takes essentially the same time (628.38 microseconds
inline, 628.43 microseconds on the pipelined worker). The main thread still waits
for the next matched step in this benchmark: its simulation scope averages
351.47 microseconds. Snapshot preparation averages 0.40 microseconds.

Raw reports are in `exports/benchmarks/simulation-thread-visible-heavy-20261016/`.
The earlier hidden-window experiment is retained in
`exports/benchmarks/simulation-thread-prototype-20261016/`; its roughly 1 ms
presentation calls limit throughput, so the visible captures above are the primary
evidence for high-FPS overhead. Hidden runs were exploratory, with some development
work concurrent; visible captures ran sequentially without builds or tests.

## Reproduction and correctness

```powershell
cmake --build build --parallel 6
python scripts/benchmark_simulation_threads.py --visible --workload normal --step 0.000588235294117647 --frames 5000 --output exports/benchmarks/thread-high-fps-new
python scripts/benchmark_simulation_threads.py --visible --workload heavy --output exports/benchmarks/thread-heavy-new
```

The benchmark's `pipeline` mode prefetches exactly one next step while rendering
the current published state. It waits for that step at the next frame boundary
if necessary. This preserves equal physics work and rendered state traces across
modes. Normal interactive playback instead polls without waiting and coalesces
pending elapsed seconds, so its adaptive step boundaries may differ when physics
falls behind. There is no interpolation or independent fixed physics cadence.

`simulation_worker_contracts` verifies exact positions, velocities, prepared
render positions and hierarchy revisions against inline integration for both
two-body fixtures and the full Solar System, front-buffer stability while the
worker runs, captured snapshot isolation, coalesced elapsed time, command order,
pause, stale/rejected edits, replacement systems, error recovery, shutdown and
switching back to inline. All eight selected CPU/SPICE/orbit GPU tests,
including the worker contracts, and the threaded `--self-test` passed. The separate existing point-source
`rendering_contracts` test exceeded a 30-second timeout; it uses `Simulation`
directly and does not exercise this runner.

The prototype retains the integrator's existing numerical algorithms and
substep limit. A command can still wait for expensive in-flight physics;
fully asynchronous command acknowledgements and smooth render interpolation
remain follow-up work. Snapshot copying scales with body count, and these
measurements do not predict overhead for much larger systems.
