# Performance audit — 24 October 2026

Intel i5-12400F · RTX 4060 Ti · Release · 1600×900 · visible workspace · VSync off.
Each capture uses 600 warmup frames, 1500 measured frames and three repetitions.
The suite covers 17 views, with 22 controlled experiments and a final baseline
repeat, plus three interactive-cadence checks: **265,500 measured frames**.
Earth rings use the procedural generator's default configuration and
seed **42**, applied in memory.

Follow-up: the [cached atmospheric shadow weights prototype](atmosphere_shadow_cached_weights.md)
reduced the turning scene's shadow pass by 25% and frame time by 19% in its paired
capture. It is now enabled by default after visual comparison; the original audit
below retains the preceding native-weight baseline.

Tables below use the initial baseline. The full repeat changed frame times by
**−6.3% to +4.7%**. CPU work varied more: the full-Earth case changed from
**0.576 to 0.448 ms**, so small CPU differences between separate launches are
indicative. The large motion-dependent shadow result is well above that variation.

## Where to lean out

| Priority | Area | Measured evidence | Next optimization to investigate |
|---|---|---|---|
| 1 | Atmospheric ring shadows during motion | Turning: **3.890 → 0.985 ms/frame** with atmospheric shadows disabled. Live simulation: **3.682 → 1.398 ms**. | Reduce fresh shadow evaluation during turning and small physical motion; investigate adaptive/VRS work with progressive history. Validate against dense shadows, disocclusions and changing illumination. The removal delta is an upper bound, not an equivalent-quality saving. |
| 2 | CPU terrain and texture selection | Terrain preparation: **0.100 ms** on static ground, **0.282 ms** turning; with rings **0.118 → 0.369 ms**. Live ground costs **0.314–0.363 ms**. | Separate selection invalidation from view-transform updates; reuse unchanged patch/texture selections and avoid rebuilding per-patch data unnecessarily. See [terrain preparation](../src/terrain/terrain_renderer.cpp) and [surface streaming](../src/terrain/surface_textures.cpp). |
| 3 | Fullscreen GPU work | Quarter-pixel rendering changes full Earth **1.265 → 0.691 ms**, ring-shadow turning **3.890 → 1.376 ms**. | Concentrate GPU work on covered/affected pixels, especially shadow resolve and close-up surface shading. Lower resolution changes quality; it is a scaling experiment. |
| 4 | Recurring CPU preparation | Planetshine preparation is **0.083–0.101 ms/frame**, including paused scenes. Workspace construction is typically **0.116–0.187 ms**. | Cache unchanged secondary-lighting results and reuse scratch storage. Cache static UI text/formatting. Preserve generation, geometry, dynamics, lighting and material invalidation. See [planetshine](../src/rendering/planetshine.hpp) and [workspace](../src/app.cpp). |
| 5 | Overview rendering | Removing Gaia saves **0.109 ms/frame**; removing orbits **0.136 ms**; removing ring drawing **0.076 ms**. | Cull invisible draw geometry while retaining hosts for shadows/secondary lighting; investigate ring LOD and unnecessary catalog/orbit work. Adjacent GPU timestamp attribution moves when features change, so use the whole-frame deltas. |

The atmospheric shadow implementation is in
[atmosphere_renderer.cpp](../src/rendering/atmosphere_renderer.cpp) and
[the shadow shaders](../shaders/common/atmosphere_shadow_pass.glsl).
No rendering-quality settings were changed in the baseline, and no renderer
optimization was applied as part of this audit.

## Scene timings

Mean milliseconds. CPU is main-thread work; GPU is the full command interval.
These overlap and must not be added. `live` runs analytical simulation at 1×;
other Earth views are paused. The normal loaded scene is `analytical`.
The ground pair shares an identical pose. The ringed space close-up is recentered
on the shadow; use the controlled experiments to isolate feature cost.

| View | Frame | CPU | GPU |
|---|---:|---:|---:|
| Normal loaded scene | 0.645 | 0.437 | 0.573 |
| Paused overview | 0.609 | 0.395 | 0.563 |
| IAS15 overview, 1× | 0.616 | 0.418 | 0.564 |
| Earth approach / streaming | 1.680 | 0.569 | 1.662 |
| Inside Earth atmosphere | 0.749 | 0.540 | 0.695 |
| Inside atmosphere, procedural ring shadow | 1.281 | 0.633 | 1.261 |
| Inside atmosphere, turning | 0.941 | 0.798 | 0.662 |
| Ring shadow, turning | 3.890 | 1.099 | 3.872 |
| Earth full disc | 1.186 | 0.772 | 1.144 |
| Earth fills screen | 1.265 | 0.576 | 1.245 |
| Earth limb | 1.053 | 0.707 | 1.002 |
| Earth fills screen, ring shadow | 1.753 | 0.548 | 1.735 |
| Earth eclipse | 1.722 | 0.564 | 1.703 |
| Saturn and rings | 1.518 | 0.558 | 1.499 |
| Inside atmosphere, live | 1.175 | 0.910 | 1.109 |
| Inside atmosphere, ring shadow, live | 3.682 | 1.062 | 3.664 |
| Earth fills screen, live | 1.642 | 0.673 | 1.622 |

## What is already inexpensive

- Simulation at 1×: analytical worker integration **0.012–0.014 ms**, IAS15
  **0.248 ms**; snapshot copying approximately **0.001 ms**. These worker scopes
  overlap rendering and do not explain the costly ring-shadow scenes.
- Composite plus UI GPU intervals are approximately **0.03–0.04 ms** in these
  captures. Transparency resolve is approximately **0.03–0.04 ms**.
- Optical atmosphere LUTs remain cached in settled views. Live views rebuild
  sky/aerial LUTs each frame; turning alone keeps those LUTs cached.
- Screen-filling Earth feature-removal savings: normal maps **0.079 ms**, terrain
  **0.143 ms**, secondary lighting **0.098 ms**. These change the image and are
  lower priorities than moving ring shadows.

## Evidence and rerunning

[Full timing tables, p95s, repetition ranges, pass details, feature deltas and
cache counters](../exports/benchmarks/audit-20261024/timings.md) are retained with
raw JSON/CSV and exact fixture/build/driver metadata under
`exports/benchmarks/audit-20261024/`. The final baseline drift check is included
there. Startup, shader compilation, warmup and report writing are excluded.
The main suite uses a fixed 1/60-second simulation step per rendered frame;
at uncapped frame rates this advances simulation faster than wall time. Separate
interactive-cadence captures use steps close to actual frame duration.
Those measured **0.591 ms** for the normal overview (0.6 ms step), **1.146 ms**
inside the atmosphere (1.2 ms step), and **3.852 ms** inside the live ring shadow
(4 ms step), confirming the same bottleneck.

Stationary ground and close-up views finish with no pending texture/normal
requests. The approach intentionally includes streaming; it is a changing path,
not a single steady view. Some limb frames still have normal-map requests pending.
GPU timestamps describe pipeline intervals, not utilization or exclusive pass
cost; feature removal and resolution scaling provide the stronger cost checks.

The benchmark contracts, comparison checks, application GPU self-test and audit
preview/geometry checks passed. The seed-42 ground previews confirm visible ring
shadowing. See [the reproducible audit commands](performance_benchmarks.md#comprehensive-application-audit).
