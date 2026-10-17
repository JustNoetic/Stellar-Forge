# Eclipse shadow candidates

Resolved celestial surfaces and rings evaluate finite-source eclipse shadows in
`shaders/common/eclipse_shadow.glsl`. Each receiver/star pair gets a compact list
of potential blockers from `CelestialRenderer`; pixels iterate that list rather
than every body. The existing umbra/penumbra radii, smooth visibility curve,
blocker order and multiplicative visibility are unchanged.

The CPU builds candidates every rendered frame from the same camera-relative
float positions and AU radii uploaded to the GPU. No hierarchy assumptions,
camera visibility tests or caster screen-size cutoff are used: an offscreen or
unresolved moon can still shadow a resolved planet. Self and stellar blockers
are excluded as before. Receivers with no mesh contribution need no list; the
CPU cutoff is conservatively below the shader's 1.5-pixel transition threshold.
Each scene star has an independent list, including the fallback light in
starless scenes. Lists have no per-pair truncation.

Body records and candidate words are prepared in reused ordinary CPU vectors
and copied to host-mapped upload buffers once. Reading mapped upload memory
during candidate construction was substantially slower on the measured device.
Each in-flight frame owns its candidate buffer; capacity grows after that
frame's fence has completed. Binding 2 contains uint words: first an offset/count
pair for each `body_index * num_stars + star_index`, followed by blocker indices.
Ring receivers follow the body receivers, indexed as `num_bodies + host_index`;
their bounds enclose the full outer ring radius and include the parent planet as
a possible blocker. They use all rendered nonstellar bodies with positive radius,
without hierarchy restrictions or per-ring truncation. Ring GPU shadows retain
oblate-caster projection; surface shadows retain spherical casters. See
[ring measurements and regressions](ring_performance.md).

Offsets are absolute word indices in this buffer. An empty list returns full
visibility before doing per-pixel eclipse geometry.

## Conservative rejection

The bound covers the entire receiving equatorial sphere, which also encloses
its oblate mesh and interpolated triangle interiors. Let `S` be the light center,
`R` the receiver center, `C` the caster center, and `a` the receiving radius:

```
X = R - S                  Y = C - S
D = length(X)              v = length(Y) * a
A = dot(Y, X)              M = D + a
```

For any receiving point `P = R + delta`, with `length(delta) <= a`, the shader's
perpendicular caster distance is `length(cross(Y, P-S)) / length(P-S)`. Its lower
bound is `max(0, (length(cross(Y,X)) - v) / M)`. The shader's projected light-radius
fraction is bounded above by `clamp(1 - (A-v)/M², 0, 1)` whenever the caster lies
on the accepted surface-to-light segment. Reject a caster if the lower distance
bound exceeds its radius plus the light radius times that upper fraction.
Bounds on `A +/- v` also reject casters wholly outside that segment.

Receiver/caster bounds include a float-precision margin based on uploaded
coordinate magnitudes. If the receiving sphere includes the light center, all
nonstellar, non-self casters are retained. This deliberately accepts false
positives near boundaries rather than removing real shadows. CPU contract tests
compare rejection against the original float fragment geometry over 1.92 million
surface/interior samples, including varied scales, translations and alignments.

## Reproducing the close-ups

Two opt-in fixtures extend the existing benchmark harness:

- `earth-full`: paused analytical Solar System at time zero, camera on Earth's
  sunward side at 1.1 equatorial Earth radii from its center.
- `earth-eclipse`: same distance, analytical time exactly 1.565 years. The camera
  direction follows the near intersection of the Sun-to-Moon shadow axis with
  Earth's bounding sphere, centering the solar eclipse. A missing intersection
  rejects the fixture instead of silently measuring a different view.

Both use the normal default FOV, HDR lighting, Gaia and 2048-segment orbit lines.
Earth covers the entire 1920x1080 framebuffer. Both scenes pause before warmup,
so CPU simulation cost is excluded from the rendering comparison.

```powershell
.\build\StellarForge.exe --benchmark --benchmark-scene earth-full --benchmark-warmup 600 --benchmark-frames 5000 --benchmark-repeats 5 --benchmark-detailed --benchmark-output exports/benchmarks/earth-full
.\build\StellarForge.exe --benchmark --benchmark-scene earth-eclipse --benchmark-warmup 600 --benchmark-frames 5000 --benchmark-repeats 5 --benchmark-detailed --benchmark-output exports/benchmarks/earth-eclipse
```

Use distinct output paths for baseline and candidate builds, then use
`scripts/compare_benchmarks.py` with `--metric cpu_celestial_ms`,
`--metric gpu_celestial_ms`, `--metric gpu_frame_ms`, and `--metric frame_ms`.
Detailed reports check that simulation and camera traces match. Keep window
visibility, dimensions and instrumentation identical and alternate build order
when repeating measurements.

With GPU tests enabled, `StellarRenderingTests --earth-closeups <directory>`
exports the two celestial-only 1920x1080 images as raw RGBA16F readbacks for exact
comparison. The ordinary GPU suite also verifies total shadow, rejected casters,
independent lighting from a second star, upload-buffer growth, and changing
body/star counts. `StellarRenderingTests --eclipse-only` runs these eclipse checks
without the longer point-source suite. The close-ups expect zero Earth blockers
in the control view and only the Moon in the eclipse view.

## Candidate filtering measurements: 2026-10-16

Release, NVIDIA GeForce RTX 4060 Ti, visible 1920x1080 window, VSync/validation
off, default rendering settings, UI hidden. Each version/scene has 50,000 measured
frames across two captures, each with five repetitions of 600 warmup + 5,000
measured frames. Repeated captures reversed scene order. All paired reports pass
the benchmark comparison's hardware, fixture, timing-coverage and per-frame
simulation/camera trace checks.

| View | Timing | Before (ms) | Filtered (ms) | Change |
|---|---|---:|---:|---:|
| Full Earth | CPU celestial | 0.004104 | 0.005670 | +0.001566 ms |
| Full Earth | GPU celestial | 0.463928 | 0.221789 | -52.2% |
| Full Earth | GPU frame | 0.721561 | 0.498563 | -30.9% |
| Full Earth | Frame wall time | 0.751434 | 0.529383 | -29.6% |
| Eclipse | CPU celestial | 0.004086 | 0.005644 | +0.001558 ms |
| Eclipse | GPU celestial | 0.457191 | 0.217645 | -52.4% |
| Eclipse | GPU frame | 0.730401 | 0.508956 | -30.3% |
| Eclipse | Frame wall time | 0.761120 | 0.540947 | -28.9% |

CPU celestial includes candidate construction, upload and command recording.
GPU celestial includes sphere and point passes, not a separately isolated eclipse
interval. CPU/GPU execution overlaps; do not sum these values. Individual
captures and other passes vary, so the CPU whole-frame differences are not used
to claim a CPU speedup. The incremental renderer CPU cost is about 1.6 microseconds.

All 2,073,600 pixels in each full RGBA16F celestial-only close-up match the
original renderer exactly: zero differing components and zero maximum absolute
difference. The eclipse center and penumbra are preserved.

Local pooled results and links to the eight detailed captures are retained in
[`exports/benchmarks/eclipse-comparison/report.md`](../exports/benchmarks/eclipse-comparison/report.md)
and its adjacent `report.json`. Original source/shader identity, raw frame traces
and individual repetition statistics are recorded in those captures. An initial
prototype that read mapped upload memory regressed CPU performance and is
excluded from these final results.

## Reverted footprint experiment

Per-pixel footprint rejection was removed after measurement showed no clear
end-to-end FPS improvement and additional CPU work. The retained implementation
is CPU candidate filtering only. Historical measurements remain in
[`exports/benchmarks/footprint-comparison/report.md`](../exports/benchmarks/footprint-comparison/report.md).

## Remaining scope

Filtering is CPU work proportional to resolved receivers, lights and scene
bodies. GPU eclipse work is proportional to shaded fragments and retained
blockers. It does not restrict shading to a projected shadow footprint. The
existing approximate penumbra, spherical casters, lack of correct annular
totality, and omission of incoming eclipses from point-source lighting remain.
