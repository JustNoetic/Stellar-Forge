# Terrain precision and geometry cache performance

Measured on 17 October 2026 with an Intel Core i5-12400F and NVIDIA GeForce RTX 4060 Ti. Release build, 1920 × 1080 offscreen HDR target, 60° vertical FOV except the telescope case, 2-pixel error target and 1024 patches per body. Geometry is synthetic; no disk texture streaming is included.

The original precision/cache tables and the first moving-ground capture below predate balanced refinement. Balanced-refinement results also predate local height bounds. Current selector results are in [local height and error metadata](#local-height-and-error-metadata). Do not treat earlier captures as an equivalent rendering workload.

## Method

Build with `STELLAR_BUILD_RENDER_TESTS=ON`, then use a new output filename:

```powershell
cmake --build build --target StellarForge StellarRenderingTests --parallel
.\build\StellarRenderingTests.exe --terrain-benchmark scratch/terrain-benchmark/capture.json
```

The same executable compares four modes:

- Existing fixed sphere, for context.
- Full procedural FP64 reference: surface and normals are evaluated in the vertex shader on every draw.
- Cached FP32: positions/normals are baked on cache misses, stored as patch-local floats, and reused. Distant views retain selective FP64 final placement/projection.
- Cached FP64 arithmetic: the same cached data, patch selection, index buffers and triangles, but parent interpolation, morphing and vertex placement/projection use double arithmetic. This isolates arithmetic precision from the benefit of caching; it does not increase cache storage precision.

Each mode warms for 40 draws and measures 120 draws per repetition, repeated three times with rotating block order. Stationary modes all use the same selected geometry. The moving trace follows the same 0.002-radian/frame orbital path in each mode. CPU timings include terrain preparation and celestial draw recording, excluding submit/fence/device-idle waits. GPU hardware timestamp intervals include bakes and HDR/depth clears; separate draw intervals are also reported. Initialization, compilation, image readback, simulation, Gaia, orbit lines, UI and presentation are excluded. Each offscreen sample completes before timing readback/reset, so this is not application frame throughput, display latency or FPS. CPU/GPU times overlap and must not be added.

The JSON includes means, medians, p95 values, patch/bake/rebuild counts, build/compiler identity, GPU identity and cold-cache samples. The local capture used below is `scratch/terrain-perf/cached-final.json`.

## GPU results

Mean GPU pass milliseconds, including clears:

| View | Patches | Full FP64 reference | Cached FP64 arithmetic | Default cached path |
|---|---:|---:|---:|---:|
| Orbital planet | 160 | 0.344 | 0.153 | 0.042 |
| 30 km, looking down | 8 | 0.063 | 0.052 | 0.047 |
| 30 km, horizon | 60 | 0.154 | 0.082 | 0.034 |
| 1 m, reference ellipsoid | 4 | 0.051 | 0.045 | 0.043 |
| 1 m, reference horizon | 128 | 0.268 | 0.122 | 0.036 |
| 1 m above synthetic relief | 90 | 0.194 | 0.090 | 0.043 |
| Moving orbital camera | 162 at final sample | 0.341 | 0.153 | 0.040 |
| Off-axis telescope, 200 AU / 0.00003° FOV | 302 | 0.691 | 0.278 | 0.085 |

The orbital case improves approximately 8.1× versus evaluating the whole surface in FP64 every draw. With caching held constant, FP32 arithmetic improves that same GPU pass approximately 3.6× versus cached FP64 arithmetic. These factors apply to this pass and hardware, not the whole application. Small-patch views are dominated more by pixel work and fixed pass overhead, so their gains are smaller. The telescope default includes selective double final projection.

All stationary measured cached draws had zero bakes. The moving FP32 run baked 28 patches across 360 measured frames and had no reference fallback during those frames. Cached-FP64 blocks reused the retained geometry from earlier blocks; use stationary zero-bake cases for the strict arithmetic comparison.

## CPU results

Mean CPU milliseconds for the default cached path:

| View | CPU prepare + record |
|---|---:|
| Orbital planet | 0.040 |
| 30 km, looking down | 0.005 |
| 30 km, horizon | 0.015 |
| 1 m, reference ellipsoid | 0.004 |
| 1 m, reference horizon | 0.034 |
| 1 m above synthetic relief | 0.025 |
| Moving orbital camera | 0.135 |
| Distant telescope | 0.072 |

Stationary selection is reused; the moving camera rebuilt selection on all 360 measured frames. Before this change, local captures measured approximately 0.40 ms for the stationary orbital case and 2.84 ms for the stationary one-meter relief case. Those historical CPU differences include selection reuse and faster neighbor/coarsening operations, not a change in GPU arithmetic. A moving near-relief camera can still incur the conservative height-bound refinement/balancing cost; these stationary improvements do not remove that work from every possible frame.

## Moving near-ground validation

The following capture predates the balanced-refinement fix and records the defect that motivated it. The dedicated moving-ground capture uses the default cached FP32 path on the same hardware and settings:

```powershell
.\build\StellarRenderingTests.exe --terrain-ground-benchmark scratch/terrain-perf/moving-ground-final.json
```

Use a new output filename. Each case runs three independent repetitions, invalidating geometry through a fresh fixture generation before each 60-frame warmup. Each repetition then measures 240 moving frames: four seconds at a fixed simulated 60 Hz, independent of benchmark wall time. Thus each row aggregates 720 measured frames. The camera follows the analytic synthetic surface at the specified radial clearance; it does not follow the coarser rendered triangles or exercise collision detection. Relief amplitude is 2 km. The JSON preserves every measured frame, the first four cold frames per repetition, preparation/draw breakdowns, mean/median/p95/p99/min/max, patch counts, LODs, bakes, fallback and selection rebuild counts. Debug PPM previews are captured after timing.

The offset equatorial path begins 0.15 radians from +Z. The original-location cases begin at +Z, matching the earlier stationary one-meter relief probe. Measured travel per repetition is 20 m at 5 m/s, 400 m at 100 m/s and 4 km at 1000 m/s.

| Path / clearance / view | CPU mean / p95, ms | GPU total mean / p95, ms | Drawn patches | Deepest LOD |
|---|---:|---:|---:|---:|
| Offset, stationary / 1 m / down | 0.007 / 0.014 | 0.043 / 0.044 | 15 | 2 |
| Offset, 5 m/s / 1 m / down | 0.474 / 0.589 | 0.343 / 0.515 | 15 | 2 |
| Offset, 5 m/s / 1 m / horizon | 0.470 / 0.566 | 0.268 / 0.639 | 16 | 4 |
| Offset, 100 m/s / 1 m / down | 0.485 / 0.619 | 0.313 / 0.515 | 15 | 2 |
| Offset, 100 m/s / 1 m / horizon | 0.478 / 0.594 | 0.293 / 0.623 | 16 | 4 |
| Offset, 1000 m/s / 100 m / horizon | 0.474 / 0.587 | 0.326 / 0.599 | 16 | 4 |
| Original, 5 m/s / 1 m / down | 0.989 / 1.232 | 0.537 / 1.545 | 90 | 17 |
| Original, 100 m/s / 1 m / horizon | 1.487 / 1.795 | 0.352 / 0.696 | 289–389 | 17 |

Every moving sample rebuilt selection; the stationary case rebuilt none after warmup. CPU preparation accounts for approximately 99% of the moving CPU interval; celestial draw recording averages 0.005–0.009 ms. Geometry reuse therefore does not remove the moving selector's CPU cost.

All measured frames had zero reference fallback. The original-location horizon case baked 114 patches across 720 measured frames, averaging 0.158/frame with a peak of 13, below the 128/frame allowance. Every other case had zero bakes after warmup. The initial original-location horizon frame needed 418 patches: 128 bakes plus 290 reference fallbacks. Across the recorded cold samples, GPU time peaked at 7.427 ms; measured CPU time peaked at 3.274 ms in that case. The geometry pool remains bounded to 64 MiB.

GPU measurements are sensitive to power management in this serialized offscreen harness. A telemetry sample during moving captures reported P8, 210 MHz graphics, 405 MHz memory and 5% utilization. GPU clocks were not locked or sampled per frame. The original-location walking GPU interval had a 5.573 ms measured maximum despite its 0.537 ms mean. These results include low-load clock behavior and cannot establish a GPU arithmetic speedup over the earlier stationary capture, sustained application throughput or FPS.

### LOD quality limitation exposed by motion

The offset cases are under-refined. Every relief case reaches the selection budget before balancing. A CPU diagnostic using the current selector at 5 m along the offset path found 1023 patches up to LOD17 before balancing and only 15 up to LOD2 afterward; the horizon snapshot went from 1024 up to LOD17 to 16 up to LOD4. The corresponding original-location downward snapshot retained 90 patches up to LOD17 after starting with 1023. The budget cap is applied before balancing, so the final draw count can be far below the cap.

Bounds retain the full global ±2 km relief allowance even on tiny patches. Many nearby bounds consequently overlap the camera, exhaust refinement capacity, and leave large neighbor-level disparities. Balancing repairs those disparities solely through ancestor coarsening, which can cascade through the selected set. The diagnostic also found offset smooth-ellipsoid cases dropping from LOD23 to LOD8 without hitting the budget, so tightening relief bounds alone will not resolve the balancing behavior.

The low GPU load and zero cache churn on the offset path do not validate meter-scale terrain performance. This diagnosis prompted balanced refinement, which reserves transition geometry during selection instead of coarsening away detail afterward. Per-tile height ranges/geometric errors remain useful when source tiles become available. Precision tests establish placement precision for specified geometry; separate selection regressions now require fine geometry below the camera.

## Balanced-refinement validation

This capture records the balanced-refinement fix before tile-local bounds were added. The selector keeps the visible leaf set balanced throughout refinement. A highest-error leaf split recursively plans any necessary coarser-neighbor splits, counts the resulting visible children, and commits the whole group only if it fits the patch budget. Rejected groups leave the previous leaf set intact. There is no final coarsening pass. A live leaf map and inactive-queue-entry checks avoid rebuilding adjacency maps for repeated collapse passes.

Build and reproduce with a new output filename:

```powershell
cmake --build build --parallel
ctest --test-dir build --output-on-failure
.\build\StellarRenderingTests.exe --terrain-ground-benchmark scratch/terrain-perf/moving-ground-balanced.json
```

The same eight cases, fixed paths, fresh generations, 60-frame warmups and 240 measured frames × three repetitions produced the following results. Every one of the 5760 measured frames retained deepest LOD17 and stayed within the 1024-patch cap. These are default cached FP32 draws.

| Path / clearance / view | CPU mean / p95, ms | GPU total mean / p95, ms | Drawn patches | Bakes mean / max per frame |
|---|---:|---:|---:|---:|
| Offset, stationary / 1 m / down | 0.289 / 0.338 | 0.072 / 0.073 | 1023 | 0 / 0 |
| Offset, 5 m/s / 1 m / down | 0.906 / 1.202 | 0.074 / 0.091 | 1023 | 0.088 / 9 |
| Offset, 5 m/s / 1 m / horizon | 0.914 / 1.153 | 0.203 / 0.387 | 1022–1024 | 0.071 / 9 |
| Offset, 100 m/s / 1 m / down | 1.042 / 1.285 | 0.307 / 0.735 | 1015–1024 | 0.754 / 16 |
| Offset, 100 m/s / 1 m / horizon | 0.872 / 1.014 | 0.062 / 0.073 | 1022–1024 | 0.754 / 9 |
| Offset, 1000 m/s / 100 m / horizon | 0.882 / 0.996 | 0.483 / 0.715 | 1013–1024 | 7.588 / 33 |
| Original, 5 m/s / 1 m / down | 0.885 / 1.076 | 0.402 / 0.511 | 1023 | 0.021 / 5 |
| Original, 100 m/s / 1 m / horizon | 0.896 / 1.034 | 0.394 / 0.498 | 1022–1024 | 0.825 / 16 |

The offset ground cases retain fine tiles instead of collapsing to LOD2–4. At the original location, moving-horizon CPU preparation/recording fell from 1.487 to 0.896 ms mean, even while drawing more geometry. The stationary offset CPU interval grew from 0.007 to 0.289 ms because instance/cache preparation now handles 1023 retained patches instead of 15; stationary selection is still reused. All moving frames rebuilt selection. CPU preparation remains the main steady-state cost. GPU intervals retain the power-management variability described above; the lower GPU times in some rows do not demonstrate an arithmetic optimization.

All measured frames had zero reference fallback. The 1000 m/s stress trace baked 7.588 patches/frame on average and at most 33, within the unchanged 128/frame allowance. The geometry pool remains 64 MiB. Conservative global relief bounds still consume the selection budget; finer local height/error metadata is a separate improvement.

Cold geometry remains a hitch risk. An initial 1023-patch view bakes 128 patches and draws 895 uncached patches through the expensive FP64 reference shader. The maximum GPU interval among the recorded first-four-frame samples was 46.748 ms: 4.253 ms preparation and 41.534 ms draw. The identical first-frame walking case in its other repetitions took 2.211 and 1.996 ms, so clock/scheduling behavior materially affects this result. This fix preserves fine selection; it does not add resident-parent fallback or change the cold-cache policy. These cold samples and short steady traces do not establish sustained texture-streaming performance or application FPS.

All 13 CTests passed, including cached/reference image and precision checks. Added CPU regressions cover smooth/relief one-meter views at face interiors, poles, cube seams and a cube corner. They require LOD15 or finer below and around downward-looking cameras at the default budget, check fine horizon selection, and verify coverage, no ancestor overlap, 2:1 edges and stitch masks at 64/128/512/1024-patch budgets. The ground benchmark now rejects any trace frame that loses fine LOD or exceeds its patch budget.

## Local height and error metadata

Tile-local interval bounds replace the global relief envelope for selection. Height extrema and derivative envelopes are computed from the analytic function over each tile, then reused through a bounded 16,384-entry CPU LRU. Bounds account for displacement and oblate normal variation; selection uses a local curvature estimate. Shader morphing keeps its existing global coefficient so adjacent patches use a common edge rule. The geometry pool, bake allowance and cold-miss fallback policy are unchanged.

The relief benchmark reproduces the 2 km altitude comparison while moving at 100 m/s. It includes fixed ellipsoid-altitude views with 0/20 km relief and separate views holding 2 km above the generated surface with 20 km relief. The path starts at 0.15 radians from +Z, over a synthetic valley so both ellipsoid-altitude views remain above the surface. Camera positions, FOV, framebuffer, synthetic heights and all settings other than the bounds implementation match between the before/after captures. Above-surface cases remove the different physical-clearance effect from changing amplitude at a fixed ellipsoid altitude.

Each case uses three fresh geometry generations, 60 warmup frames and 240 measured frames per repetition, for 720 measured frames per row. Metadata can persist across generations because the analytic source is unchanged. There are 4320 measured relief frames and another 5760 in the moving-ground validation. CPU includes preparation/recording and excludes waits; GPU includes bakes and clears. These remain offscreen pass timings, not FPS. GPU power management remains automatic and affects comparisons.

```powershell
.\build\StellarRenderingTests.exe --terrain-relief-benchmark scratch/terrain-benchmark/relief-capture.json
.\build\StellarRenderingTests.exe --terrain-ground-benchmark scratch/terrain-benchmark/ground-local-capture.json
```

Local captures are `scratch/terrain-perf/relief-global-bounds.json` before the change, `relief-local-bounds.json` afterward, and `moving-ground-local-bounds.json` for the repeated ground trace. Use new output filenames when reproducing.

| 2 km altitude reference / relief / view | CPU before mean, ms | CPU after mean / p95, ms | Patches before → after | GPU after mean, ms |
|---|---:|---:|---:|---:|
| Ellipsoid / 0 km / down | 0.041 | 0.040 / 0.063 | 10 → 10 | 0.049 |
| Ellipsoid / 20 km / down | 0.885 | 0.051 / 0.082 | 1024 → 10 | 0.183 |
| Ellipsoid / 0 km / horizon | 0.084 | 0.081 / 0.117 | 80 → 80 | 0.091 |
| Ellipsoid / 20 km / horizon | 0.882 | 0.140 / 0.197 | approximately 1023 → 123 | 0.181 |
| Generated surface / 20 km / down | 0.883 | 0.052 / 0.086 | 1023 → 10 | 0.156 |
| Generated surface / 20 km / horizon | 0.867 | 0.142 / 0.204 | approximately 1023 → 121 | 0.124 |

The fixed-altitude 20 km-relief downward case reduces CPU preparation/recording approximately 17.4×, and the horizon case approximately 6.3×. Flat controls retain their patch counts and similar CPU timings. No relief case hits the patch budget after the change. Warm relief traces build no new metadata: they visit 282–446 selection nodes/frame while reusing up to 530 retained entries. They also have no measured geometry bakes or fallback draws over these short paths. Cold high-relief views now fit their initial geometry within the 128-bake allowance; all recorded relief cold samples have zero reference fallback. Among 20 km-relief cases, the largest recorded GPU cold interval is 2.528 ms, compared with 40.194 ms before. This does not establish a general cold-cache latency guarantee.

The original near-ground traces also retain fine detail with much less unnecessary geometry:

| Path / clearance / view | CPU mean / p95, ms | GPU mean, ms | Patches min–max | Deepest LOD min–max | Maximum bakes/frame |
|---|---:|---:|---:|---:|---:|
| Offset, stationary / 1 m / down | 0.006 / 0.012 | 0.050 | 8 | 23 | 0 |
| Offset, 5 m/s / 1 m / down | 0.088 / 0.131 | 0.185 | 4–10 | 22–23 | 4 |
| Offset, 5 m/s / 1 m / horizon | 0.185 / 0.251 | 0.192 | 156–190 | 22–23 | 46 |
| Offset, 100 m/s / 1 m / down | 0.090 / 0.139 | 0.205 | 4–10 | 22–23 | 8 |
| Offset, 100 m/s / 1 m / horizon | 0.197 / 0.267 | 0.228 | 154–194 | 22–23 | 62 |
| Offset, 1000 m/s / 100 m / horizon | 0.133 / 0.191 | 0.142 | 98–126 | 15–16 | 38 |
| Original, 5 m/s / 1 m / down | 0.083 / 0.124 | 0.205 | 4–10 | 21–23 | 4 |
| Original, 100 m/s / 1 m / horizon | 0.231 / 0.308 | 0.143 | 178–218 | 21–23 | 62 |

Every measured ground frame stays below the patch and bake caps, with no budget pressure or reference fallback. One-meter views reach LOD21–23 rather than the previous LOD17. The metadata LRU retains at most 6514 entries in this capture, below its 16384-entry cap; the GPU geometry pool remains 64 MiB. The largest recorded ground cold GPU interval is 9.916 ms, so the unchanged reference fallback can still hitch on a fresh view. Increasing relief still changes GPU geometry keys; continuous slider changes can require rebaking, even though tighter selection reduces the amount of affected geometry.

All 13 CTests pass, including cached/reference GPU images and distant/near-ground precision. Added tests check analytic and float height containment and displaced sphere containment over tile interiors/edges across six faces, levels 0–24, polar ratios 1/.8/.05 and amplitudes .0032/.01. Triangle interpolation errors are checked against the analytic curvature envelope; the float height padding is a separate allowance. Tests also verify metadata LRU reuse/eviction/capacity, visible ground coverage at 2 km with 20 km relief, and repeated source-metadata reuse. This is metadata for generated terrain; actual Earth elevation tiles are still pending.

## Cold cache, budgets and precision

After a new fixture generation invalidated the orbital cache, the first frame baked 128 patches and drew 32 remaining patches with the reference shader: 0.284 ms CPU and 0.230 ms GPU. The next frame baked the remaining 32: 0.100 ms CPU and 0.081 ms GPU. Subsequent GPU samples returned to approximately 0.042 ms. Pipelines and buffers were already initialized for this capture; these are cold geometry entries, not application startup times.

The geometry pool is bounded to 64 MiB (4837 patch slots); metadata/job/index buffers bring terrain buffer payload to approximately 67.9 MiB. Source dataset size does not grow this pool. Future source textures will need their own explicit budget. Cache misses beyond the 128-bake frame allowance use the existing reference shader. Interactive recording adds no cache-specific GPU fence or device-idle waits.

`--terrain-precision` verifies that the default near-ground shader has no FP64 capability. LOD24 cached position readback agrees with a CPU double ellipsoid oracle within 0.00001 m at a 70,000 km radius; the observed maximum was approximately 1.36e-7 m. This measures the smooth reference geometry and FP32 local-offset conversion, not the accuracy of the synthetic height function or a real heightmap. Rendered comparisons cover oblate geometry, poles, cube seams, one-meter clearance, translated origins and on/off-axis 200 AU telescope views. HDR image comparisons allow subpixel coverage and rounding differences rather than require bit identity.
