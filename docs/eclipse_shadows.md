# Eclipse shadow candidates

Resolved celestial surfaces and rings evaluate finite-source eclipse shadows in
`shaders/common/eclipse_shadow.glsl`. Each receiver/star pair gets a compact list
of potential blockers from `CelestialRenderer`; pixels iterate that list rather
than every body. Candidate filtering preserves blocker order and multiplicative
visibility. The overlap calculation is shared with stellar point
visibility through `shaders/common/eclipse_optics.glsl`.

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
without hierarchy restrictions or per-ring truncation. Rings, surfaces and local
stellar points all use the projected oblate caster silhouette. See
[ring measurements and regressions](ring_performance.md).

Offsets are absolute word indices in this buffer. An empty list returns full
visibility before doing per-pixel eclipse geometry.

Surface ring-host bit masks are appended after the blocker lists in the same
buffer. `ring_shadow_params.x` gives their absolute word offset in the scene UBO;
each body/star pair reads one mask. These masks filter rings casting shadows on
surfaces, independently of the eclipse lists for bodies casting shadows on rings.
See [surface ring filtering and measurements](ring_performance.md#surface-ring-host-candidates).

## Small casters and annular eclipses

Every nonstellar body is eligible to shadow another body's surface, independent
of its reference parent. This includes moon-to-moon eclipses. Candidate retention
does not imply a dark eclipse: the apparent caster size limits its obscuration.

At a receiving point, the stellar radius projected onto the caster's distance is
`r_source = star_radius * caster_distance_along_ray / star_distance`. The full
and partial phases use an inner radius `abs(caster_radius-r_source)` and outer
radius `caster_radius+r_source`. When the caster is smaller than the source,
maximum blocked light is `(caster_radius/r_source)^2`, the uniform-disc area
fraction. A contained small transit therefore has constant shallow dimming
rather than a dark center. Partial phases now use analytic circular-disc overlap,
replacing the former `smoothstep(t)^0.625` visibility curve. The circular-segment
calculation uses a small-angle series to avoid cancellation for very unequal
apparent sizes. Containment and disjoint cases exit before segment integration.

Previously the inner radius was clamped to zero for small casters, but the
visibility curve still went to zero at the axis. A GPU regression with a moon
covering about 1.2% of the stellar disc returned only 5.46% remaining light.
The corrected calculation returns 98.80%, including when the contained moon is
offset from the source center.

Unresolved modeled stars use the same area calculation in the point
vertex shader along the camera-to-star ray. Stellar transits use physical caster
radii even below the mesh transition threshold, attenuating integrated point
flux rather than inventing an enlarged silhouette. Nonstellar point sources keep
their previous mesh-weighted center-ray visibility. This calculation is separate
from the lists for surface receivers; no camera readback or new CPU pass is used.

`StellarRenderingTests --eclipse-lighting` checks totality, shallow annular
eclipses, contained offset transits, caster rejection, independent stars and
unresolved stellar transit flux at several caster sizes. The historical candidate
filtering measurements below predate this optical correction.

## Oblate parent eclipses

For an oblate spheroid with equatorial radius `a`, polar radius `c`, and pole
inclination `i` to the light ray, the orthographic silhouette is an ellipse with
major radius `a` and minor radius `sqrt(a*a*cos(i)^2+c*c*sin(i)^2)`. Its minor axis
follows the pole projected onto the light's perpendicular plane. A pole-on parent
therefore casts a circular silhouette; an equatorial view exposes its full
flattening. This replaces the previous ring/point approximation that used the
polar radius whenever a projected pole existed. Surface eclipses previously
ignored caster oblateness entirely.

`eclipse_caster_visibility` projects this ellipse for surfaces, rings and local
stellar transit flux. Partial ellipse/circle intersections use 32-point
Gauss-Legendre area integration. Conservative containment and disjoint tests skip
quadrature in fully dark/lit regions; an ellipse wholly inside the source uses
its analytic area. CPU candidate bounds still use the enclosing equatorial
sphere, so no candidate construction or upload work is added for oblateness.

The GPU optics contracts contain 1,512 cases spanning caster/source radius ratios
0.01–1000, flattenings 0–0.5, pole inclinations 0/45/90 degrees, several limb
directions, contacts, totality and annular phases. Circular cases are compared
with a double-precision lens formula; ellipses use independent dense integration
along the other ellipse axis. On the RTX 4060 Ti, maximum brightness errors were
0.00274 percentage points for circles and 0.0350 points for ellipses, versus
14.84 points for the sampled smoothstep reference. These are sampled errors,
not a universal bound. Rendered-surface contracts separately verify a flattened
parent's missing polar shadow, its pole-on circular eclipse and a tilted partial
eclipse on a moon.

```powershell
.\build\StellarRenderingTests.exe --eclipse-optics
.\build\StellarRenderingTests.exe --eclipse-lighting
.\build\StellarRenderingTests.exe --eclipse-benchmark scratch/eclipse-overlap/new-capture
```

`eclipse_reference_smoothstep` is a diagnostic render setting, passed in
`SceneCelestialUBO::eclipse_params.x`; it restores the area-limited smoothstep
curve and directional-radius ellipse approximation for comparisons. It has no
UI control. The benchmark compares that reference with overlap at 1080p in
screen-filling clear, total, spherical-partial and oblate-partial fixtures,
using 40 warmup plus 120 measured frames per mode and three alternating-order
repetitions. Reports contain total offscreen celestial GPU and CPU recording
times, excluding image readback and presentation; these are not application FPS.

RTX 4060 Ti Release results at 1920x1080, 2026-10-18, median GPU total:

| Screen-filling view | Smoothstep reference | Circle/ellipse overlap |
|---|---:|---:|
| Clear receiver | 0.216064 ms | 0.216064 ms |
| Total eclipse | 0.082944 ms | 0.081920 ms |
| Circular penumbra | 0.082944 ms | 0.112640 ms |
| Oblate penumbra | 0.100352 ms | 0.224256 ms |

The partial-eclipse fixtures deliberately put the entire framebuffer inside the
penumbra. Extra GPU cost is about 0.030 ms for circular lenses and 0.124 ms for
ellipse integration in these stress views. CPU recording medians remained
2.5–3.7 microseconds in both modes with no new CPU optical work; the small
differences are timing variation. Original reports and previews are retained in
`scratch/eclipse-overlap/run1/`.

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
uniform-disc illumination, numerical ellipse quadrature and omission of incoming
eclipses from reflected point-source lighting remain. The model projects radii
and spheroids onto the caster plane rather than tracing finite-distance tangent
cones. Stellar limb darkening is not integrated into eclipse visibility, and
overlapping blockers still combine multiplicatively.
