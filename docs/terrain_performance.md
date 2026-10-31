# Terrain precision and geometry cache performance

## Full application navigation, 29 October 2026

Release application on an i5-12400F / RTX 4060 Ti (NVIDIA 621.01), hidden
1920×1080 window, UI enabled, VSync off, immediate presentation, default 45° FOV
and rendering settings instead of saved preferences. Imported terrain, albedo,
normals, atmospheres/shadows, refraction, clouds, secondary lighting, bloom, Gaia
and orbit lines remain enabled. The Solar System is paused to isolate camera
movement. Each scene has three repetitions of 600 stationary warmup frames plus
900 measured frames, using a fixed 1/60-second scripted step.

The new `terrain-navigation` suite starts at the highest daylight sample of the
selected imported Earth tile (8795 m in this source), looking 3° above the
horizon. Walking follows decoded triangle heights at 5 m/s with 1.7 m clearance
and travels 74.9 m; flying uses 250 m/s with 2 km clearance and travels 3.746 km.
Turning holds the eye and rotates 224.75°. Camera path construction and asset
initialization are excluded; streaming during the measured path is included.

| Terrain target | Scene | Mean patches | Mean FPS | Frame p95, ms | CPU work mean, ms | GPU frame mean, ms | Terrain CPU mean / p95, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 px | Still | 215 | 257.8 | 4.228 | 0.624 | 2.376 | 0.035 / 0.049 |
| 16 px | Walk | 254 | 209.9 | 5.214 | 0.952 | 2.928 | 0.270 / 0.456 |
| 16 px | Fly | 137 | 232.0 | 4.694 | 0.737 | 2.694 | 0.058 / 0.091 |
| 16 px | Turn | 200 | 245.4 | 4.488 | 0.889 | 2.314 | 0.255 / 0.401 |
| 2 px | Still | 742 | 247.1 | 4.434 | 0.676 | 2.488 | 0.089 / 0.130 |
| 2 px | Walk | 783 | 188.9 | 5.828 | 1.359 | 3.039 | 0.683 / 1.022 |
| 2 px | Fly | 669 | 220.4 | 4.968 | 0.844 | 2.810 | 0.166 / 0.244 |
| 2 px | Turn | 838 | 197.3 | 5.868 | 1.730 | 2.452 | 1.057 / 1.632 |

The 2-pixel variant preserves the 1024-patch limit; turning reaches it in 15.3%
of measured frames, so the target is not always met. All GPU timestamps are
present and all terrain, texture and normal failure counters are zero. The
largest baseline frame is 8.84 ms at 16 pixels and 9.55 ms at 2 pixels. The former
is dominated by a 7.23 ms presentation call; the latter includes a 2.86 ms normal
preparation scope. These are wall-time scopes and do not establish whether a
rare outlier came from computation, a lock or OS scheduling. No measured baseline
frame exceeds 16.67 ms. This is a warmed local path, not a cold-start/teleport test.

At 16 pixels, walking spends about 0.906 ms on GPU terrain drawing, 0.759 ms on
bloom, 0.340 ms on observer refraction rebuilding and 0.209 ms on atmospheric LUTs.
Separate walking feature-removal runs, each with the same frame/repetition counts,
reduce GPU frame time from 2.928 to 2.103 ms without bloom, and to 2.570 ms without
refraction. Frame means change from 4.764 to 3.837 and 4.372 ms respectively.
These change the image and are not equivalent-quality optimization results.
Refraction removal also changes visible terrain coverage (mean 247 versus 254
patches). GPU/CPU clocks are uncontrolled and run order is not randomized.

Default terrain preparation is a small part of total frame time in these views.
Dense turning remains the most useful terrain CPU target: desired-cut reuse can
coincide with a resident-cut rebuild while geometry streams, and close movement
often changes the desired cut itself. Updating changed subtrees and residency
dependencies is the next candidate; the threshold formula remains a minor target
according to the isolated profile below. For total frame time, terrain shading,
bloom and moving-eye refraction/LUT work warrant investigation too. The present
wait is substantial, so CPU/GPU scope comparisons alone do not prove hardware
utilization or identify the source of driver/compositor overhead.

The previous isolated before/after reductions must not be interpreted as
full-application FPS improvements: this capture measures the optimized build
only. CPU work excludes explicit fence/acquire/present waits; GPU and CPU scopes
overlap, and inclusive parents and children must not be added. In particular,
`gpu_terrain_prepare` covers all celestial preparation, whereas
`cpu_terrain_prepare` isolates terrain preparation.

Raw reports, CSV, pooled/per-repetition statistics, worst-frame details, analysis
script and source/binary fingerprints are retained in
`tests/terrain-fullapp-20261029/`. Reproduce with the documented
[`terrain-navigation` scenes](performance_benchmarks.md#imported-terrain-navigation),
adding `--benchmark-variant terrain-detail` for the 2-pixel target.

## Retained selection and dependency plans

Implemented 29 October 2026 after the CPU profile below. `SelectionPlan` validates the previous cut's visibility and refinement decisions against the current camera/projection, retaining patches, stitching, prefetch groups and cache dependencies while those decisions hold. Maximum projected error is updated on every successful validation. A changed cut, bounds/source revision, shape/limit change, live availability constraint or budget pressure uses the full selector. After repeated invalidations a 32-update backoff suppresses reuse attempts while continuing full current-view selection. This is a validated reuse path; it does not yet repair individual changed subtrees.

The resident benchmark uses the same Release high-ground Earth probe, 1080p, 60-degree FOV and 4096-patch cap as the earlier reuse benchmark. Loaded moving views alternate eye positions 10 cm apart. Each process reports the median of nine batches of 256 prepare calls; the table is the median across three before and three after processes, with alternating process order. The baseline already contains the reusable traversal storage and instance-bucket optimization. CPU clocks and scheduling remain uncontrolled.

| Error / selected patches | Stationary CPU prepare before → after, ms | Moving resident CPU prepare before → after, ms | Moving reduction |
| --- | ---: | ---: | ---: |
| 16 px / 130 | 0.0217 → 0.0106 | 0.0968 → 0.0285 | 70.5% |
| 2 px / 632 | 0.0835 → 0.0402 | 0.4207 → 0.1055 | 74.9% |
| 0.5 px / 1603 | 0.2252 → 0.1120 | 1.0987 → 0.2577 | 76.5% |

These are isolated `TerrainRenderer::prepare` measurements with resident geometry and unchanged cuts, not application FPS. Vertex generation, source workers, GPU rendering, albedo/normal preparation, simulation, UI and presentation are excluded. Camera-dependent records still update. Geometry quality, 2:1 balancing, stitching, the 64 MiB geometry cache and admission limits are preserved.

A separate paced 100 m/s source trace exercises five views and loading at a 2-pixel target / 1024-patch cap. Patch counts match the baseline and four stationary near-ground HDR captures are byte-identical. Orbital opaque coverage matches; its shading differences also occur in baseline repeats while normal maps stream, as described below. The one-meter travel cases change cuts frequently and use the full selector much more often. An initial implementation regressed in those cases; adaptive capture backoff and eliminating dependency sorting limit that extra work. Absolute source-trace times vary substantially with pacing, host scheduling and worker activity, so the loaded alternating-view table is the repeatable performance result, not a claimed speedup for arbitrary sustained travel.

CPU oracle traces cover 9120 frames across face interiors/seams/corners, polar ratios 0.05–1, meter approaches, horizon/downward views, camera jumps, projection/error/occluder changes, max-level changes and imported Earth heights. Reused and fresh cuts match exact patch IDs, stitch masks, maximum-error values and required-tile sets. Retained prerequisite ordering can differ from a fresh priority traversal; the renderer's prefetch list is sorted, and committed groups retain all four children. Explicit tests cover source/bounds invalidation, budget pressure, availability callbacks and resuming reuse after churn backoff. The full selector remains the oracle and fallback.

All eight targeted terrain/importer/normals/surface/refraction CTests pass with Vulkan validation requested. The separate cached/reference precision and terrain preview probes also pass, including projection changes while reusing a cut, LOD24 precision, cube seams/poles and meter-scale coverage. Application and rendering/CPU test targets build; the final diff check is clean.

Sources/snapshots, before/after binaries, the standalone oracle test, local build script, raw captures and summaries are retained in `tests/terrain-incremental-20261029/`. `verified-resident-summary.json` contains the final-build table inputs; `manifest.json` records source/binary hashes. The public architecture and invalidation rules are documented in [terrain](terrain.md).

## CPU preparation profile after the reuse optimization

Profiled 29 October 2026 after buffer/instance reuse and before retained selection plans, on Earth pack `10334824041510882139`, at 1920 × 1080, 60-degree FOV and a 4096-patch cap. Standalone diagnostic copies instrument `TerrainRenderer::prepare` and `select`; normal application sources and binaries receive no profiling hooks. Three process repetitions each cover 16/2/0.5-pixel error, stationary resident terrain, alternating loaded eye positions 10 cm apart, and 240 paced frames traveling 100 m/s one meter above reconstructed ground. Resident cases measure 1280 calls each. The location is the same high-ground eastward view used by the resident preparation benchmark below. Clocks, scheduling and CPU power management are uncontrolled.

The table uses the process with the median mean preparation time for each case, at **2-pixel error**. Coarse scopes form disjoint stages: cache/streaming includes the selection wrapper minus `select`, plus frame cache planning; selector time is therefore counted once. Transform/record preparation includes placement checks. The 632-patch resident view is deliberately more detailed than the current default 16-pixel error.

| CPU preparation stage | Stationary / 632 patches | Moving resident / 632 patches | Traveling / about 641 patches |
| --- | ---: | ---: | ---: |
| Tree selection, including visibility/error/refinement/stitching | 0.4% | 45.0% | 47.4% |
| Cache, prefetch, retained dependencies and streaming bookkeeping | 72.7% | 33.4% | 32.2% |
| Instance bucketing, mapped-buffer writes and flush | 23.9% | 16.2% | 14.7% |
| Placement checks and transform/draw-record preparation | 1.0% | 5.0% | 4.8% |
| Other body setup/statistics | 1.9% | 0.4% | 0.9% |
| Mean CPU preparation, ms | 0.0928 | 0.4483 | 1.1220 |

The stationary case includes one possible view transition at the case boundary; its selection time is approximately zero after that transition. Across the three repeated resident captures, the selector contributes 41–53% at 2 pixels. At 0.5 pixels / 1603 patches, the median-run moving breakdown is 44.0% selection, 33.6% cache/streaming, 17.1% instance writes and 5.0% placement/records. At 16 pixels / 130 patches it is 52.6%, 40.3%, 0.8% and 5.4% respectively. Counts, raw stage times and run variability are retained in the JSON reports.

Inside the selector, operations were sampled once per 17 or 31 calls, with a warmed empty-timer-pair correction. In loaded moving terrain at 2 pixels, estimated shares of the **whole preparation** are approximately 11% heap/leaf-map updates, 7% metadata retrieval, 6% visibility tests, **4–6% distance/error projection**, 3% coarser-neighbor queries and 3% output sorting. These estimates are useful for ranking, not exact cycle attribution. The projection scope includes conservative bounds distance, multiplication/division and the orthographic/perspective choice; the final threshold comparison itself remains part of traversal control. New metadata computation is a separately recorded subset of metadata retrieval, not another additive stage.

The warmed distance/error projection accounts for roughly one tenth of selector time. The hypothesis that the threshold formula is the main CPU bottleneck is not supported. The largest measured area during movement is selection together with repeated prefetch/retention/cache planning, approximately 78–80% in the representative detailed views. Incremental selection is therefore a reasonable next target, particularly if it also retains refinement/dependency plans and updates them when topology or residency changes. These shares are not predicted speedups: conservative visibility/error checks, balancing and residency validity still need to be maintained.

### Timing overhead and loading limits

An interleaved control toggles timers off/on in the same compiled binary and alternates order for twelve pairs of 256 resident calls. At 632 patches, coarse scopes add approximately **0.5%**; detailed sampling adds **11.2%** at stride 17 and **7.0%** at stride 31. The compiled-out baseline toggle varies by −0.7%, showing host noise. Coarse overhead ranges from approximately zero at 1603 patches to 2.2% at 130 patches. Detailed resident estimates agree on projection's small share across both strides, but should be read with this instrumentation overhead in mind.

Fine sampled scopes are **not used as a quantitative breakdown for paced travel**: their scaled sum exceeds inclusive selector time. A warmed fixed timer-bias correction does not adequately describe tiny operations after frame pacing. Those raw values remain in the reports with an explicit interpretation warning. Coarse scopes are the basis of the preparation table.

A separate loaded-terrain control uses the same 60 Hz pacing and immediate-submission pattern as travel. At 2 pixels it measures 1.102 ms with coarse scopes, versus the median traveling capture's 1.122 ms; selection is 48.0% in that paced loaded control. Thus the apparent increase from the unpaced loaded benchmark cannot be attributed to streaming alone. This single paced control is descriptive, not a precise measurement of a 2% loading penalty. The median travel run admits/builds 60 geometry patches and spends 10/240 frames loading, with at most 21 pending jobs. It is a local geometry-cache-miss workload, not a broad test of cold elevation-file I/O or teleports across the planet.

Worker CPU, camera height queries, GPU baking/upload recording, submit/fence waits, albedo/normal-map preparation, other scene systems and presentation are outside the preparation timer. Source workers still run concurrently. This profile answers the geometry-preparation question; it does not rank total application CPU consumers or measure application FPS.

The instrumented selector matches 1200 saved deterministic selection digests exactly, including patch IDs, stitch masks, ordered requests/prerequisites, visits, LOD, budget/loading flags and error bits. Resident measurements reject geometry admissions/loading; travel rejects missing imported coverage and worker failures. Production source hashes are preserved and checked. Reproduction sources, build scripts, generated instrumented copies, raw captures, controls, source hashes and `summary.json` are in `tests/terrain-cpu-profile-20261029/`:

```powershell
python tests/terrain-cpu-profile-20261029/build_profile.py
python tests/terrain-cpu-profile-20261029/build_profile.py --controls
.\build\bin\tests\TerrainProfile-coarse.exe tests/terrain-cpu-profile-20261029/new-coarse.json
```

## Reusable CPU preparation and instance buckets

Measured 29 October 2026, Release GCC `-O3`, RTX 4060 Ti, existing Earth elevation identity `10334824041510882139`, 1920 × 1080, 60-degree vertical FOV and a 4096-patch cap. The resident CPU probe finds the highest reconstructed sample in the finest tile with the largest maximum and looks east one meter above it. It waits for geometry and worker requests to settle at two eye positions 10 cm apart. Moving measurements alternate those positions, forcing current-view selection and transforms while avoiding streaming arrivals. Stationary measurements repeat one view. Each process measures nine batches of 256 calls per case; the table is the median of the three process medians. Process order is before/after, after/before, before/after. CPU clocks and scheduling are uncontrolled.

| Error target / patches | Stationary CPU prepare before → after, ms | Moving CPU prepare before → after, ms | Moving reduction |
| --- | ---: | ---: | ---: |
| 16 px / 130 | 0.0365 → 0.0188 | 0.1074 → 0.0812 | 24.4% |
| 2 px / 632 | 0.1922 → 0.0894 | 0.5528 → 0.4555 | 17.6% |
| 0.5 px / 1603 | 0.6852 → 0.2521 | 1.5354 → 1.1471 | 25.3% |

This measures `TerrainRenderer::prepare` only, including its normal cache/protection/record work. Geometry is resident and no bakes or loading occur in measured calls. GPU submissions, drawing, worker work, albedo/normals streaming, simulation, UI and presentation are outside the interval. These are preparation savings, not application FPS or GPU speedups. The small repeated displacement deliberately isolates CPU preparation with loaded terrain; sustained travel into new terrain has additional loading costs.

Preparation now references body-owned draw records and scatters GPU instances into 64 stable mode/stitch buckets. It no longer copies and stable-sorts a second array of full draw records. Geometry-keyed placement reuse avoids recomputing anchors/bake inputs after an unchanged patch selection. Traversal vectors, heap storage, pooled leaf allocations and frame request/retention vectors retain capacity. Albedo and normal selectors also have owner-local traversal scratch. View/error/culling/residency evaluation, balancing, refinement order, geometry quality and cache/admission budgets remain the same. Height-envelope computation also runs outside its metadata mutex, with a duplicate-publication check under the lock.

Two alternating before/after source-benchmark pairs additionally exercise 120 stationary and 120 paced moving frames at 100 m/s in five views, at 2-pixel error and a 1024-patch cap. The new `terrain_prepare_ms` interval separates terrain preparation from the existing whole-celestial `prepare_ms` interval. Results vary with host scheduling and loading; the moving one-meter downward case has essentially no gain in the second pair. All four stationary near-ground HDR captures are byte-identical. Orbital opaque coverage matches, but the first orbital capture has shading differences that also occur between the two original-renderer runs while normal maps stream. That capture is not treated as a deterministic image-equivalence result.

An independent 1200-view comparison against the saved original selector matches exact patch IDs, stitch masks, ordered refinement prerequisites/requests, node visits, deepest levels, budget/loading flags and geometric-error bits. CPU contracts additionally compare fresh/reused scratch at face interiors, seams, corners, delayed residency and LOD24, and concurrent height reads must publish one envelope revision. Eight targeted CTests pass with Vulkan validation requested: terrain importer/CPU/imported GPU/normals, surface map/eviction, smooth surfaces and GPU refraction. The separate cached/reference precision and terrain preview probes also pass with validation requested, including moving-view placement reuse, cube seams, poles and meter-scale coverage. Application and rendering/CPU test targets build successfully.

Sources, original snapshots/binaries, compile scripts, raw timing runs, hashes and image comparisons are retained in `tests/terrain-lod-optimization-20261029/`. `resident-summary.json` contains the table inputs; `resident_prepare.cpp` and `build_resident_probe.py` reproduce the resident measurement. The terrain source benchmark accepts an optional error target following its pacing argument in the local test harness:

```powershell
.\build\bin\tests\StellarRenderingTests.exe --terrain-source-benchmark data/tiles/earth/95cebe9dca5b/height tests/terrain-new/source.json --paced 2
```

## Surface-aware horizon culling

Measured 26 October 2026, Release, 1920 × 1080, 60-degree vertical FOV, 2-pixel error and a 1024-patch cap. The existing Earth elevation pack has identity `10334824041510882139`; heights, error targets, residency budgets and shaders are unchanged. The previous horizon test used only the patch sphere expanded by the inverse polar ratio. These spheres can cross the occlusion cone or tangent-distance boundary even when the entire displaced surface is hidden. A second test uses the thin surface enclosure and a tile-local outer radial bound to reject the fully occluded cap. It applies to both geometry and shared surface-texture selection.

The high-ground CPU probe finds the highest reconstructed sample in the finest tile with the largest declared maximum, approximately 8786 m in this visual elevation pack. It looks horizontally in four orthogonal tangent directions, one meter above the reconstructed ground. Both selectors have the same resident height envelopes and unlimited geometry availability. This measures desired selection rather than streaming arrival or application FPS.

| Tangent direction | Selected patches before → after | Traversal visits before → after |
| --- | ---: | ---: |
| 0 | 785 → 632 | 1502 → 1174 |
| 1 | 532 → 374 | 1174 → 842 |
| 2 | 580 → 459 | 1354 → 882 |
| 3 | 257 → 184 | 790 → 518 |

All four views retain deepest LOD17 without budget pressure. At 100 m clearance, counts change from 784/532/583/262 to 634/374/462/187. Nearby ridges can still hide submitted patches: the selector has no hill-against-hill occlusion or depth-pyramid pass. The global minimum-height allowance remains conservative, so this change reduces planetary-horizon overdraw without claiming to remove all hidden mountain geometry.

The existing GPU source benchmark, on the RTX 4060 Ti with the same pack and settings, reduces stationary orbital / 2 km horizon / 1 m horizon patch counts from 213/149/248 to 165/107/206. Downward views retain 6 and 8 patches. All four stationary near-ground RGBA16F images are bit-identical before and after. Orbital opaque coverage is unchanged; the maximum HDR-channel difference is 0.025635. These captures use the benchmark's original location, separately from the high-ground CPU probe. GPU clocks and host scheduling are uncontrolled; no full-frame timing claim is made.

CPU regressions independently check ray-sphere visibility for signed heights, polar ratios 0.05–1, reduced refraction occluders and LOD0–24, plus radial bounds on displaced samples and morph geometry. Raw GPU reports/readbacks and high-ground CSVs are in `tests/terrain-horizon-20261026/`; probe sources are `tests/terrain_culling_probe.cpp` and `tests/terrain_high_ground_probe.cpp`.

Both application and rendering-test targets build. Nine targeted CTests pass, covering terrain/import fixtures, smooth surfaces, refraction, imported geometry, terrain normals and surface textures. The separate cached/reference precision check passes. The application startup CTest exceeds its 30-second limit on two attempts; running `StellarForge --self-test` separately completes all three frames cleanly. The startup timeout remains unresolved by this terrain change.

## Independent per-pixel terrain normals

Measured 19 October 2026, Release, i5-12400F / RTX 4060 Ti, 1920 × 1080, 60-degree FOV. Earth and Moon use the existing catalog albedo/elevation packs, a camera distance of two equatorial radii, 16-pixel geometric error and a 64-patch cap. Both modes draw exactly the same geometry; only `terrain_normal_maps_enabled` changes. Four repetitions alternate mode order, with 48 submissions per block and the first 16 discarded. Each submission draws the scene 128 times; GPU draw timestamps are divided by 128. Each mode has 128 retained samples per body. No geometry bakes or texture uploads occur in measured samples. CPU preparation is once per submission. Streaming warmup, readback, image export, submission waits and presentation are excluded. Clocks are uncontrolled; these offscreen pass measurements are not application FPS or a full-frame timing guarantee.

| Body | Drawn patches | Vertex-normal GPU draw median, ms | Detailed-normal GPU draw median, ms | Added GPU draw, ms | CPU prepare median before → after, ms |
| --- | ---: | ---: | ---: | ---: | --- |
| Earth | 62 | 0.07542 | 0.11270 | 0.03728 | 0.2063 → 0.2740 |
| Moon | 33 | 0.07145 | 0.10396 | 0.03251 | 0.1977 → 0.2533 |

Normal residency is 168/172 tiles respectively, within a separate 256-layer, 34.03 MiB RGBA16-SNORM array. The completion-managed upload ring is 8 MiB, and two bounded page buffers add approximately 2.03 MiB. Fragment derivatives select normal detail independently of mesh LOD; resident data normally adds two filtered normal reads, page lookup and body rotation/normalization. CPU preparation includes the additional stream's page-table/ownership work and reused selection. Normal generation and source reads run on its worker only on misses. Initial loading retains vertex-normal shading until normal roots are available. This benchmark does not measure cold normal generation latency or sustained moving-camera streaming.

Raw distributions, settings and paired images are in `scratch/terrain-normal-final-128/`. `Moon-comparison.png` shows the same coarse mesh with vertex normals on the left and detailed normals on the right. Capture rendering copies readback bytes to cached CPU memory before validating pixels; CPU image scans and exports are outside the timed intervals. Reproduce with a new output directory:

```powershell
.\build\StellarRenderingTests.exe --terrain-normal-preview data/surfaces.json scratch/terrain-normal-new
```

All 21 CTests passed with Vulkan validation requested, including the new `terrain_normal_contracts`, imported terrain, surface eviction, rendering, eclipse and atmosphere contracts. Normal contracts cover six-face unit/outward normals, matching neighbor gutters, pole/spin orientation, source shape compatibility, unchanged opaque coverage and geometry/cache behavior when toggling quality.

## Direct coarser-neighbor lookup

Measured 19 October 2026, Release GCC -O3, i5-12400F. Both binaries use the same sources and harness except the neighbor lookup. Three alternating before/after runs; each case uses seven batches. Reported numbers are the mean across three Earth regions of each region's median run time (each run reports its median batch). Five views, 32 samples of 100 m/s camera motion at virtual 60 Hz, cubic terrain clearance, 1080p, 60-degree FOV, 2-pixel target, 1024-patch cap. This CPU benchmark deliberately rebuilds desired selection with bounded renderer-style description memoization. A stationary full renderer can reuse selection entirely. Worker CPU, camera height queries, GPU, drawing and presentation are excluded. Host clocks/scheduling were uncontrolled; these are observed timings, not timing thresholds or application FPS.

| View | Original, ms | Optimized, ms | CPU time reduction |
| --- | ---: | ---: | ---: |
| orbit | 0.1247 | 0.0620 | 50.3% |
| 2km-down | 0.0286 | 0.0177 | 38.0% |
| 2km-horizon | 0.6364 | 0.2863 | 55.0% |
| 1m-down | 0.0289 | 0.0181 | 37.2% |
| 1m-horizon | 0.6810 | 0.3191 | 53.1% |

Balancing and stitch generation only need coarser neighbors. Active leaves are constructed from roots by atomic 2:1-balanced splits, so a coarser adjacent leaf can only be one level above. Edges inside a leaf's own parent cannot have a coarser neighbor. Remaining edges query one adjacent parent-level key, with integer addressing inside a face and the existing cube projection across face seams. This removes equal-level lookups and ancestor walks without changing refinement, culling, budgets, residency or source bounds. The existing standard unordered map remains; no persistent lookup cache or additional memory budget was introduced.

All 45 view/region/cache-mode selection trace hashes match across all six final benchmark runs. Digests cover patch keys, stitch masks, ordered refinement prerequisites, visited count, deepest level, budget pressure and exact maximum-error bits. Another 1,200 deterministic varied-view comparisons match the original selector, including ordered availability requests and loading pressure, budgets 6-8192, levels 0-24, polar ratios 0.05-1, culled/full-sphere views and partial residency. The production CPU tests add 180 cases of independent edge sampling through LOD24 at face interiors, seams and corners, including delayed refinement and extreme flattening. CPU imported terrain, GPU terrain preview, cached/reference precision and imported moving terrain tests pass with Vulkan validation requested; no validation errors were emitted. StellarForge, StellarTerrainTests and StellarRenderingTests build successfully.

Raw runs, source/executable hashes, intermediate experiments and the cold/new-tile tables are in `scratch/terrain-neighbor-optimization-20261019/`.

## Surface enclosures and preparation reuse

Measured on 18 October 2026, Release, i5-12400F / RTX 4060 Ti, 1920 × 1080, 60° FOV, 2-pixel target and 1024-patch cap. Both versions use Earth elevation identity `619520292878143246`, the existing geometry-bake changes in the working tree, no color sources, and the same source-benchmark fixtures described below. The baseline was rebuilt in `scratch/terrain-culling` with the previous selector/preparation code; the active checkout was preserved. Each view measures 120 stationary and 120 moving samples at 100 m/s; moving samples are paced to at least 16.667 ms. Worker CPU, camera height queries, waits, readback, simulation, UI and presentation are excluded. GPU clocks and host scheduling were uncontrolled; these are descriptive pass timings, not application FPS or timing thresholds.

| View | CPU preparation mean before → after, ms | Drawn patches before → after |
| --- | --- | --- |
| Orbital, stationary | 0.0318 → 0.0231 | 109 → 100 |
| Orbital, moving | 0.4504 → 0.3762 | 109 → 100 |
| 2 km down, stationary | 0.0314 → 0.0134 | 10 → 6 |
| 2 km down, moving | 0.2924 → 0.2191 | 10 → 6 |
| 2 km horizon, stationary | 0.0640 → 0.0268 | 95 → 83 |
| 2 km horizon, moving | 0.4566 → 0.2934 | 95 → 83 |
| 1 m down, stationary | 0.0337 → 0.0232 | 10 → 8 |
| 1 m horizon, stationary | 0.1311 → 0.0894 | 195 → 185 |
| 1 m horizon, moving | 0.9663 → 0.7422 | 186.8 → 170.8 mean |

Moving traversal visits fell from 340 to 158 in orbit, 655 to 288 at the 2 km horizon, and 1032 to 606 at the 1 m horizon. Thin surface supports improve both frustum rejection and the refinement distance; horizon culling now uses a conservative inner ellipsoid. View-dependent culling constants are computed once per traversal. Imported selection reuses its desired result when all prerequisites are resident. Unchanged selections retain ancestor lists and patch placement data; unchanged transforms/precision/debug flags retain complete draw records. `TerrainStats::patch_updates` reports actual record rebuilding. Color selection separately reuses each source's unchanged `SelectionView`; its node/rebuild counters are available in `TextureStats`, but the timings above exclude color sources.

Stationary HDR readbacks are exactly equal for both horizon views and the 1 m downward view. The 2 km downward mean absolute RGB error is `1.91e-8`; the orbital mean is `3.10e-5`, with 95 of 2,073,600 pixels changing by more than 0.03 in any channel. Before/after orbital and horizon previews were inspected. No stationary capture loses or gains opaque coverage. The 1 m stationary views still reach LOD23. Tighter distance bounds can choose shallower levels at the same conservative error target: 2 km views reach LOD11 rather than LOD12. Paced moving 1 m down/horizon traces remain streaming-limited (116/35 loading frames versus 117/77 before); ready-only minimum deepest levels are 22/21 versus 23/22. These counts include changed selection decisions and do not establish a worker-throughput or streaming-latency improvement.

Captures and HDR/preview comparisons are `scratch/terrain-culling/{before-visual,after}.json`, `comparison.json` and `comparison-*.png`. Reproduce the current measurement using a new output filename:

```powershell
.\build\StellarRenderingTests.exe --terrain-source-benchmark data/tiles/earth/b72a6fa0e156/height scratch/terrain-culling/new-capture.json --paced
```

The source benchmark saves stationary PPM and RGBA16F readbacks outside the timed intervals. CPU contracts additionally check enclosure support and conservative distance against signed displaced samples, fine/parent/grandparent triangles, all six faces, LOD0–24 and polar ratios through 0.05; 8,000 visibility cases cover ground retention. GPU contracts check stationary record reuse, texture selection reuse/FOV invalidation, imported resident coverage and cached/reference precision. The signed imported fixture can now choose LOD20 at the 1 m horizon with a maximum conservative error of 1.89131 pixels. Its former fixed LOD21 assertion was tied to the old selection rather than screen-space quality; it now verifies `TerrainStats::max_error_pixels <= target` for every ready, non-budget-limited frame. Separate meter-scale refinement, seam and coverage requirements remain. Per-frame imported reports retain this error metric. The full run passed the other 16 CTests; after updating this quality assertion, all six targeted fixture, CPU terrain, startup, imported GPU and texture-eviction tests passed.

## Tighter imported bounds and selection reuse

Measured before/after on 17 October 2026, Release, i5-12400F / RTX 4060 Ti, 1920 × 1080, 60° FOV, 2-pixel target and the same 1024-patch cap. The source is the existing Earth legacy visual elevation pack, identity `619520292878143246`, data LOD5. No pack rebake, height change, texture-resolution change or relaxed error setting is involved. This probe uses the pack's reference radius/oblateness, a fixed render pole/spin, a camera starting at 0.15 radians on the equator and no color sources. Orbital clearance is two equatorial radii above the sampled surface; ground clearances are above the decoded surface.

Each view waits for resident coverage, measures 120 stationary frames, then 120 frames moving at 100 m/s and virtual 60 Hz. Pass CPU includes prepare/record; GPU includes preparation and celestial rendering/clears. Worker I/O/baking, camera height queries, submit/fence waits, readback, simulation, UI and presentation are excluded. These are offscreen pass measurements, not application FPS. GPU clocks and OS scheduling were uncontrolled; CPU results below are descriptive captures rather than fixed timing thresholds.

| View | CPU mean before → after, ms | Patches before → after |
| --- | --- | --- |
| Orbital, stationary | 1.754 → 0.040 | 1023 → 109 |
| Orbital, moving | 1.766 → 0.181 | 1023 → 109 |
| 2 km down, stationary | 0.189 → 0.021 | 10 → 10 |
| 2 km down, moving | 0.189 → 0.113 | 10 → 10 |
| 2 km horizon, stationary | 0.824 → 0.041 | 232 → 95 |
| 2 km horizon, moving | 0.839 → 0.207 | 232 → 95 |
| 1 m down, stationary | 0.345 → 0.035 | 10 → 10 |
| 1 m horizon, stationary | 1.164 → 0.107 | 331 → 195 |

The old orbital selection exhausted its patch budget in all measured frames; the new one does not. Moving orbital traversal visits fell from 3212 to 340. Stable views perform zero quadtree visits/rebuilds after residency settles. Orbital CPU draw recording was already about 0.011 ms before the change, versus 1.754 ms preparing a moving frame: the main cost was selection/preparation rather than issuing instanced draws. The 2 km downward comparison isolates CPU improvements with an unchanged patch count. Per-body matrix/per-level morph reuse and writing only the active GPU record layout also reduce preparation work.

The actual textured preview was also repeated at the earlier 768 × 768 setup with all 14 sources bound. Earth changed from 896 to 72 imported geometry patches, with CPU mean 2.133 → 0.237 ms. Its 412 resident color pages are unchanged. The before/after Earth images were inspected side by side; mean absolute display-RGB difference is 0.000771 on a 0–1 scale, with 0.051% of pixels differing by more than 0.1 in any channel. That visual comparison includes shading/silhouette changes and complements the geometric bound tests. New images/report are under `scratch/terrain-bounds/surface-after`, with `earth-comparison.png` beside that directory.

Stationary one-meter views still reach LOD23. Free-running one-meter motion retains parent coverage but is streaming-limited: after optimization, down/horizon traces draw approximately 4–10 / 151–195 patches and reach minimum deepest LOD16/20 while loading; before they reached 17/22. Faster frames provide less worker wall time per simulated meter, so those loading counts/detail minima are not equivalent-time quality comparisons. Ready samples retain fine geometry. A separate `--paced` run gives moving frames at least 16.667 ms: down/horizon minima are LOD17/22, with 117/77 loading frames and ready-sample minima LOD23/22. Streaming latency at one meter and 100 m/s remains a limitation; this change does not claim to eliminate it. Paced CPU/GPU timings are not compared against the unpaced baseline.

Captures are `scratch/terrain-bounds/earth-before.json`, `earth-after.json` and `earth-paced.json`. Reproduce with the actual height path in `data/surfaces.json` and a new output filename:

```powershell
.\build\StellarRenderingTests.exe --terrain-source-benchmark data/tiles/earth/b72a6fa0e156/height scratch/terrain-bounds/capture.json
.\build\StellarRenderingTests.exe --terrain-source-benchmark data/tiles/earth/b72a6fa0e156/height scratch/terrain-bounds/paced.json --paced
```

The geometric bounds are derived from interpolation of normalized cube/oblate-normal vectors, with local gradient, parent height residual, encoding and FP64 rounding allowances. Parent/grandparent interpolation, common dataset-level morph thresholds, atomic cross-face balance and resident coverage remain in place. CPU tests sample fine/parent/grandparent triangles across all faces, LOD0–24, negative constant heights and polar ratios 0.05/0.5/1. GPU regressions verify stable selection reuse, lighting-only reuse, error-setting invalidation, moving imported coverage, procedural cached/reference images and physical normal seams. See [bound derivation and cache contracts](terrain.md#imported-geometric-error-and-selection-cost).

All 17 CTests passed; final CPU/imported-GPU terrain checks passed again after generation-cache cleanup. The rebuilt application also retains its startup smoke test.

The sections below retain earlier captures from the more conservative imported implementation and historical procedural versions.

## Imported surface maps

The copied library contains 19 original images (565,306,762 bytes), verified against the original project with SHA-256. The first packs contain color LOD0–3 for 14 bodies and legacy-calibrated Earth elevation LOD5. Raw sources remain under `textures/Solar System`; derived packs are under `data/tiles`, selected by `data/surfaces.json`. Finest color faces are 1024², so these are starter packs rather than the full detail of the 21,600-pixel Earth original.

The color cache adds at most 512 × 132² RGBA8-sRGB texels = 34.03 MiB, a separate 4 MiB completion-managed staging ring, two approximately 1.02 MiB maximum page buffers, and at most 64 completed CPU tile payloads = 4.25 MiB. Queue/index/container overhead is additional. Hardware array-layer limits can reduce the layer count. Height geometry continues sharing the existing 64 MiB pool and 128-admission/frame budget; color admissions are independently capped at 16/frame.

Release captures on the same i5-12400F / RTX 4060 Ti use 768 × 768 HDR/depth, 60° FOV and a camera three equatorial radii from the center. Each body waits for worker/upload convergence, captures an image, then measures 64 stationary pass samples. All 14 color sources share the same cache; views run sequentially, not as independent cold loads. Earth uses imported elevation; Jupiter uses its ordinary sphere. CPU includes surface selection and render recording; GPU includes preparation, attachment clears and celestial drawing. These intervals exclude source-worker CPU, submit/wait, image readback, tone mapping, simulation, UI and presentation. They are isolated pass measurements, not application FPS.

| Body | CPU mean / p95, ms | GPU mean / p95, ms | Imported geometry patches | Resident color tiles |
| --- | --- | --- | --- | --- |
| Earth | 2.133 / 2.363 | 1.476 / 4.063 | 896 | 412 |
| Jupiter | 0.263 / 0.315 | 0.128 / 0.129 | 0 | 512 |

GPU clocks/power management were uncontrolled; the short Earth block includes substantial timing variance. No comparison or speedup against the earlier procedural fixtures is implied. Moon and Mars captures also exercise the same renderer. Full per-body distributions, warmup-frame counts and previews are written by:

```powershell
.\build\StellarRenderingTests.exe --surface-preview data/surfaces.json scratch/surface-preview-final
```

All 17 CTests passed. The final surface-only rerun passed after adjusting requests to cover the shader's finest blended level. Its deterministic GPU test forced 448 actual texture-slot evictions across 32 bindings, checking color coverage and bounds throughout. Six Python surface contracts cover lossless/indexed payloads, linear-light filtering, interior/cube borders and corners, pole coordinates, repeatable output and signed normalization of a 16-bit height image. Real-image previews verify Earth elevation is actually selected, which also catches engine/importer reference-radius mismatches. The procedural precision/reference and imported near-ground motion suites remain passing.

## Imported-height streaming foundation

Measured on 17 October 2026 on the same i5-12400F / RTX 4060 Ti, Release, 1920 × 1080, 60° FOV, 2-pixel target and 1024-patch cap. `terrain_imported_contracts` generated a deterministic signed-height dataset with global data LOD2, nested FP32 samples and identity `1420171969660023193`. It exercises the disk/import/residency path; it is not measured Earth elevation. GPU geometry remains 64 MiB / 4837 slots, with a separate 4 MiB completion-managed staging ring and a shared 128-admission/frame budget. There are no GPU height textures.

Reproduce with a new dataset directory and inspect the generated JSON:

```powershell
python scripts/import_terrain.py scratch/imported-performance-fixture --level 2
.\build\StellarRenderingTests.exe --terrain-imported scratch/imported-performance-fixture scratch/terrain-benchmark/imported.json
```

The CTest capture is `scratch/terrain-source-tests/imported-gpu.json`; its dataset directory is recorded in `fixture-path.txt` beside it. The four cases run sequentially with one source, system generation and GPU pool; only the first starts empty. Each stationary view waits for a ready balanced selection, then measures 120 moving samples at a fixed virtual 60 Hz and 100 m/s. Clearance is relative to the decoded imported height surface. Camera-height queries, source worker CPU time, submit/fence waits, readback, simulation, UI and presentation are outside the pass intervals. Worker I/O and FP64 geometry baking run concurrently. The moving trace is free-running rather than paced to real 60 Hz, so source jobs can have much less than 16.7 ms of wall time between virtual steps. CPU and GPU intervals must not be added, and these are not application FPS measurements.

| View / surface clearance | CPU mean / p95, ms | GPU mean / p95, ms | Drawn patches | Minimum deepest LOD / ready-only minimum | Loading frames / 120 |
|---|---:|---:|---:|---:|---:|
| Down / 2 km | 0.260 / 0.356 | 0.049 / 0.049 | 10 | 12 / 12 | 0 |
| Horizon / 2 km | 0.688 / 0.802 | 0.043 / 0.045 | 212 | 12 / 12 | 0 |
| Down / 1 m | 0.451 / 0.578 | 0.050 / 0.052 | 4–8 | 17 / 22 | 109 |
| Horizon / 1 m | 1.119 / 1.304 | 0.387 / 0.524 | 276–312 | 22 / 22 | 60 |

All measured frames retained imported parent coverage, with zero procedural fallback and no worker failures. Pending jobs stayed at or below 128, completed blocks at or below 256, and admissions at or below 128/frame. Near-ground motion can temporarily draw a shallower resident ancestor: the downward trace reached LOD17 while loading, while ready samples retained LOD22 or finer. A separate metadata-only one-meter downward regression selects eight patches through LOD23 without budget pressure. Geometry finer than source LOD2 reconstructs its finite height lattice and the oblate surface; it does not add measured detail.

Stationary readiness took approximately 91, 78, 46 and 26 ms in case order; those are sequential-cache observations, not independent cold-source latency guarantees. The largest premeasurement CPU/GPU pass intervals were 1.451/0.627 ms. Worker cumulative geometry builds were 290, 762, 2023 and 2448 after the respective moving traces, with 258 payload reads retained across cases. GPU power management was automatic. Parent fallback bounds frame work and preserves coverage; it does not guarantee requested detail at arbitrary motion speed. The extra desired/resident selections, bounded metadata lookups and CPU worker path make this implementation more expensive than the earlier procedural-only selector on these different surfaces; no procedural/imported speedup is claimed.

All 15 CTests passed, including the original procedural precision/reference images, imported geometry/normal seams, parent coverage under delay and eviction, LOD24 bounds/error containment, corruption/revision checks, and multi-frame staging plus targeted mip/layer readback. The optional GDAL adapter test skipped because Python bindings were unavailable. Khronos validation layers were not installed, so GPU checks used actual submission/readback and CPU synchronization contracts without validation-layer diagnostics. The v1 source format has a bounded flat index and global data levels 0–5; sparse/paged high-resolution Earth datasets remain later work. See [terrain source and streaming contracts](terrain.md#imported-height-sources).

The following sections retain historical procedural-only captures.

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
