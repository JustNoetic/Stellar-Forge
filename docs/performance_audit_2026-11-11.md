# Performance audit — 11 November 2026

Intel i5-12400F · RTX 4060 Ti (driver 621.14) · Release `ee2ea03` (clean) · 1600×900 ·
visible workspace · VSync off. Same harness, fixtures and capture settings as the
[24 October audit](performance_audit_2026-10-24.md): 600 warmup + 1500 measured frames ×
3 repetitions per view, deterministic pipelined simulation, procedural Earth ring seed 42.
The suite covers 17 views, 33 feature-removal/resolution/UI experiments and a final
baseline repeat. Raw JSON/CSV, logs and the generated
[timing tables](../exports/benchmarks/audit-20261111/timings.md) are in
`exports/benchmarks/audit-20261111/`.

**Every view is GPU-bound at 1600×900.** GPU time exceeds main-thread CPU work in all 17
views; the closest is `earth-disc` (CPU 1.12 ms, GPU 1.26 ms). At 800×450 `earth-full`
is already CPU-bound, so the CPU items matter on faster GPUs and lower resolutions.

## Where to lean out

Ranked by measured frame-time impact in common views. Each "evidence" figure is a
direct GPU timestamp or a whole-frame feature-removal delta from this capture.

| # | Area | Evidence | Cause | Suggested fix |
|---|---|---|---|---|
| 1 | **Point-sized atmospheres rebake every frame while time runs** | Default loaded scene (`analytical`) 0.645 → **1.193 ms** since October; paused `overview` is 0.718. `atmosphere_point_lut` = **0.47 ms/frame** in `analytical`/`nbody`, 0 when paused; 0.15 ms in `earth-full-live`. | `point_dirty` compares `sun_direction`, `axis_*` and `world_pole` exactly ([atmosphere_luts.cpp:136](../src/rendering/atmosphere_luts.cpp#L136)). Any orbital motion dirties every visible point-sized atmosphere each frame. | Rebake on a phase-angle/axis tolerance (sub-pixel bodies cannot show a 1e-4 rad change), or refresh at most one point table per frame, round-robin. |
| 2 | **Static ring/eclipse shadows are recomputed every frame** | Paused `earth-ground-rings` 1.281 → **2.673 ms** since October, now identical to turning (2.661). Shadows cost **1.29 ms** (ground rings), **1.08** (full rings), **0.88** (eclipse) in paused views. | `atmosphere_shadow_temporal` defaulted off on 27 Oct ([render_settings.hpp:53](../src/rendering/render_settings.hpp#L53)), so the converged-history reuse no longer applies, and `animate_jitter` changes the sequence every frame ([atmosphere_parameters.cpp:49](../src/rendering/atmosphere_parameters.cpp#L49)). | When `shadow_reset_reasons` is zero apart from the jitter sequence, reuse last frame's `shadow_fraction` (freeze jitter on static frames) or fall back to progressive accumulation for static views. Design decision: frozen vs. animated noise. |
| 3 | **Refraction observer table rebuilds on any altitude change** | `refraction_observer` **0.60 ms/frame** in `approach`; whole frame 1.682 → 1.041 with refraction off (**−0.65 ms**). | `same_observer` compares exact floats ([refraction_renderer.cpp:21](../src/rendering/refraction_renderer.cpp#L21)); the camera radius changes every frame while flying, and each of 512×128 cells runs a bisection solve. | Quantize camera radius (rebuild when the table's angular error would exceed a pixel), interpolate between two bracketing tables, or rebuild across several frames. |
| 4 | **Fixed per-frame GPU passes** | Gaia cull **0.14–0.16 ms** in every view (`no-gaia` −0.18). Bloom **0.19 ms** (`no-bloom` −0.188; `fast-bloom` only −0.011, so the cost is the halo/extract, 0.10 ms). Orbit lines **0.125 ms** incl. the OIT clear/resolve they enable. | Gaia: each `StarGpuRecord` is 80 B padded to **96 B** (`alignas(32)`, [star_catalog.hpp:13](../src/rendering/star_catalog.hpp#L13)); 475,724 stars = 45.7 MB read per frame, 0.16 ms at the RTX 4060 Ti's 288 GB/s, so the pass is bandwidth-bound. | Gaia: pack to 32 B (FP32 position is ample at ≥1 pc; half-float colour) for two-thirds less traffic (≈0.1 ms in every view), and skip re-culling when eye, orientation, FOV, time and settings are unchanged. Bloom matches the 0.20 ms in [bloom.md](bloom.md) and is low priority; 32×32 extract tiles would cut the 22²/16² apron overfetch (1.9×). |
| 5 | **Live atmospheres rebake sky + aerial LUTs every frame** | Live views: `atmosphere_luts` **0.32–0.44 ms**, aerial **0.20–0.24 ms**, sky/aerial updates = 1.00 per frame. | `changes.sky` uses exact comparisons of `camera`, `sun_direction`, `sun_irradiance` ([atmosphere_renderer.cpp:375](../src/rendering/atmosphere_renderer.cpp#L375)). At 1× the Sun moves ~1e-6 rad per frame against ~2° sky-view texels (192×108). | Angular/altitude tolerance tied to LUT resolution, with an exact rebake on optics/lighting-mode changes. Validate at sunrise/terminator. |
| 6 | **Ocean FFT runs while paused** | `gpu_ocean_waves` **0.13 ms/frame** in every view within 1000 km of an ocean, paused or not. | `OceanWaves::update` always evolves + FFTs ([ocean_waves.cpp:171](../src/rendering/ocean_waves.cpp#L171)), though the result depends only on `seconds` and the spectrum. | Skip evolve/FFT/moments when `seconds` and the spectrum build are unchanged (`field_` is not per-frame-in-flight). Exact, no image change. |
| 7 | **Surface/terrain fragment shading** | `gpu_terrain_draw` **0.60 ms** (`earth-full`, 7 patches), 0.75 (`earth-eclipse`). `smooth` saves only 0.03: cost is per-pixel `sphere.frag`, not geometry. Textures −0.19, clouds −0.33/−0.18 (full/ground), secondary −0.10. | Per-pixel cost of the shared surface shader ([sphere.frag](../shaders/celestial/sphere.frag)), not overdraw: the early-Z probe below recovered at most 0.05 ms. | Profile `sphere.frag` with `STELLAR_SHADER_STATS=1` and split its terms (albedo/normal streaming lookups, ocean BRDF, eclipse/ring/planetshine lighting, surface atmosphere) with timestamps or per-term variants before choosing a target. |
| 8 | **CPU: refraction solve for atmospheres behind the host** | `earth-full`/`earth-disc` `cpu_atmosphere_prepare` **0.52–0.65 ms** (0.04 in `earth-limb`). With refraction off, `earth-full` CPU 1.148 → **0.563 ms**. | `select_candidates` calls `apparent_rotation` → `PreparedRefraction::solve` (65 bisection steps of `reach_travel`) for every in-frustum atmosphere seen through the host's air ([atmosphere_renderer.cpp:472](../src/rendering/atmosphere_renderer.cpp#L472), [refraction_tables.hpp:46](../src/rendering/refraction_tables.hpp#L46)). Inferred: aimed at Earth's disc, atmospheres behind it pass the frustum test and are solved every frame; the cost disappears with refraction off and in `earth-limb`. | Reject bodies whose sight line is below `lowest_visible_chord_km` (they are hidden by the host) before solving; cache the rotation per body while the host/eye/target are unchanged; stop bisection at double precision (~52 steps). |
| 9 | **CPU: per-frame recomputation in static views** | `planetshine_prepare` 0.083–0.109 ms (`no-secondary` CPU −0.084), texture + normal streaming planning 0.07–0.20 ms, UI 0.12–0.18 ms. | `prepare_planetshine` (O(bodies² × stars) occlusion and phase work) and `SurfaceMaterialCache::update` rerun with unchanged inputs ([celestial_renderer.cpp:320](../src/rendering/celestial_renderer.cpp#L320)); `SurfaceTextures::prepare_views` rebuilds the wanted/protected sets and copies the whole page table every frame ([surface_textures.cpp:425](../src/terrain/surface_textures.cpp#L425)). | Key planetshine/materials on `state_revision` + `definition_revision` + settings; skip streaming planning when views, residency and completions are unchanged and copy the page table only for frame slots whose revision is stale. |

## Change since 24 October

Frame-time and GPU-time means (ms). The October numbers predate bloom, 2D clouds, oceans,
refraction/limb images, planetshine maps and the shadow VRS/analytic defaults.

| View | Frame Oct | Frame Nov | GPU Oct | GPU Nov | CPU Oct | CPU Nov | Main driver of the change |
|---|---:|---:|---:|---:|---:|---:|---|
| overview | 0.609 | 0.718 | 0.563 | 0.644 | 0.395 | 0.517 | Bloom +0.19 |
| analytical (default) | 0.645 | **1.193** | 0.573 | 1.149 | 0.437 | 0.558 | Point LUT +0.47, bloom |
| nbody | 0.616 | **1.204** | 0.564 | 1.159 | 0.418 | 0.565 | Point LUT +0.48, bloom |
| approach | 1.680 | 1.682 | 1.662 | 1.639 | 0.569 | 0.600 | Observer table 0.60 |
| earth-ground | 0.749 | 1.082 | 0.695 | 1.040 | 0.540 | 0.550 | Clouds 0.18, ocean 0.13, bloom |
| earth-ground-rings | 1.281 | **2.673** | 1.261 | 2.636 | 0.633 | 0.710 | No static shadow reuse |
| earth-ground-turn | 0.941 | 1.093 | 0.662 | 1.036 | 0.798 | 0.658 | As earth-ground; CPU terrain improved |
| earth-ground-rings-turn | 3.890 | **2.661** | 3.872 | 2.624 | 1.099 | 0.769 | Improved: shadow VRS/analytic |
| earth-disc | 1.186 | 1.360 | 1.144 | 1.264 | 0.772 | 1.116 | Bloom; CPU +0.34 from refraction solve |
| earth-full | 1.265 | 1.784 | 1.245 | 1.737 | 0.576 | 1.148 | Clouds 0.33, bloom; CPU +0.58 from refraction solve |
| earth-limb | 1.053 | 1.257 | 1.002 | 1.217 | 0.707 | 0.620 | Clouds, bloom, ocean |
| earth-full-rings | 1.753 | **2.935** | 1.735 | 2.900 | 0.548 | 1.038 | Shadows 1.08 every frame |
| earth-eclipse | 1.722 | **2.852** | 1.703 | 2.817 | 0.564 | 0.607 | Shadows 0.88 every frame, terrain shading |
| saturn | 1.518 | 1.530 | 1.499 | 1.493 | 0.558 | 0.671 | Unchanged; rings 0.55 |
| earth-ground-live | 1.175 | 1.421 | 1.109 | 1.380 | 0.910 | 0.731 | Aerial rebake, clouds |
| earth-ground-rings-live | 3.682 | 3.033 | 3.664 | 2.995 | 1.062 | 0.850 | Improved: shadow VRS |
| earth-full-live | 1.642 | 2.221 | 1.622 | 2.178 | 0.673 | 1.300 | LUT rebakes, refraction CPU |

Improvements since October: moving ring-shadow views are 0.6–1.2 ms faster (VRS/analytic
strata), and CPU terrain preparation while turning fell from 0.28–0.37 ms to below 0.05 ms.

## Feature costs

Whole-frame savings with one feature disabled (positive = faster without it). These change
the image; they bound what an equivalent-quality optimization could recover.

| View / change | Frame | CPU | GPU |
|---|---:|---:|---:|
| analytical / no-bloom | +0.169 | +0.021 | +0.188 |
| analytical / fast-bloom | +0.011 | +0.002 | +0.011 |
| analytical / no-gaia | +0.166 | +0.009 | +0.182 |
| analytical / no-orbits | +0.118 | +0.020 | +0.125 |
| analytical / no-rings | +0.031 | +0.014 | +0.029 |
| analytical / no-refraction | +0.032 | +0.020 | +0.030 |
| analytical / no-secondary | +0.005 | +0.084 | +0.001 |
| approach / no-refraction | +0.641 | +0.047 | +0.648 |
| earth-ground / no-clouds | +0.175 | +0.022 | +0.176 |
| earth-ground-rings / no-shadows | +1.284 | +0.149 | +1.286 |
| earth-ground-rings / no-atmosphere | +1.748 | +0.247 | +1.751 |
| earth-ground-rings / no-refraction | +0.262 | +0.101 | +0.265 |
| earth-ground-rings / no-secondary | +0.051 | +0.113 | +0.051 |
| earth-ground-rings / smooth | −0.202 | +0.112 | −0.199 |
| earth-full / no-atmosphere | +0.353 | +0.687 | +0.343 |
| earth-full / no-clouds | +0.333 | +0.040 | +0.345 |
| earth-full / no-textures | +0.182 | +0.038 | +0.191 |
| earth-full / no-secondary | +0.101 | +0.110 | +0.097 |
| earth-full / no-refraction | −0.041 | +0.585 | −0.052 |
| earth-full / smooth | +0.040 | +0.048 | +0.034 |
| earth-full-rings / no-shadows | +1.078 | +0.089 | +1.080 |
| earth-eclipse / no-shadows | +0.879 | +0.057 | +0.880 |
| earth-limb / no-refraction | +0.008 | +0.028 | +0.005 |
| saturn / no-rings | +0.550 | +0.109 | +0.555 |
| saturn / no-shadows | +0.147 | +0.065 | +0.148 |
| earth-full / no-normals | +0.032 | +0.029 | +0.028 |
| earth-ground-rings-turn / no-shadows | +1.294 | +0.153 | +1.295 |
| earth-ground-rings-turn / no-refraction | +0.265 | +0.098 | +0.268 |
| earth-ground-rings-live / no-shadows | +1.292 | +0.192 | +1.295 |
| analytical / no UI | +0.023 | +0.080 | +0.018 |

Quarter pixel count (800×450): `earth-full` GPU 1.737 → 0.793 ms, but the frame only
drops to 1.288 ms because it becomes **CPU-bound** (CPU 1.102 ms, about half of it the refraction
solve in finding 8). `earth-ground-rings` GPU 2.636 → 1.131 ms and `earth-ground-rings-turn`
2.624 → 1.131 ms: the shadow cost scales with pixels.

## Early-Z probe

`sphere.frag` declares `layout(depth_greater) out float gl_FragDepth`. With reverse-Z and
`GREATER_OR_EQUAL`, depth that may only move closer cannot be rejected early, so early-Z is
off for surface and terrain fragments. To bound that cost, the depth write was temporarily
removed (restored afterwards) and paired captures were run (300 warmup + 900 frames × 2,
`exports/benchmarks/audit-20261111-probes/`):

| View | GPU control | GPU no depth write | Terrain draw change |
|---|---:|---:|---:|
| earth-full | 1.733 | 1.680 | −0.010 |
| earth-ground | 1.026 | 1.035 | −0.049 |
| earth-eclipse | 2.832 | 2.797 | −0.015 |
| earth-limb | 1.207 | 1.200 | −0.004 |

Overdraw is not where the surface cost goes; a depth pre-pass would not pay for itself.

## Baseline drift

The full 17-view suite was repeated after all experiments. Frame means changed by
**−3.4% to +1.2%** (October: −6.3% to +4.7%). The GPU findings above (0.13 ms and larger) sit
well outside that band; the smaller CPU items are indicative.

## Method notes

- Captures ran serially with the window visible. GPU timestamps are command intervals,
  not exclusive pass cost; parent and child timers overlap and must not be summed.
  Per-pass figures are per-frame means over all measured frames.
- The fixed 1/60 s simulation step per rendered frame advances time faster than wall
  clock at these frame rates. The live-view rebake findings still hold at real-time
  cadence, because any nonzero motion currently triggers them.
- `smooth` makes `earth-ground-rings` 0.2 ms slower (GPU). Not investigated.
- No renderer code was changed for this audit; the early-Z probe edit was reverted.
