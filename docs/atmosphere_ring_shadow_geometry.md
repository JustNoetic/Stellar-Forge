# Atmospheric ring shadow geometry

Try **Settings → Advanced → Fast ring shadow geometry (experimental)**.
The option and cached atmospheric weights are on by default.
Changing either option invalidates shadow history.

This experiment simplifies the geometry around repeated ring-visibility queries.
Sample count, sample positions, ring profile, three-box penumbra filter, curvature
correction, pixel footprint, finite stellar endpoint and full-shadow cutoff are
preserved. It allocates no additional GPU textures or buffers.

## Calculation

The fast path applies to one star, one candidate ring host and no candidate body
casters. A host can contain multiple ring layers and bands. Other configurations
use the existing visibility function. Grazing stellar horizons, diagnostic linear
filtering and degenerate intersections also use the existing calculation.

With the unnormalized vector to the star `v`, unit ring pole `n`, ring-plane hit
`h`, hit radius `r`, denominator `d = dot(v,n)` and intersection fraction `f`:

- Radial stretch is `sqrt(d*d + pow(dot(h,v)/r, 2))/abs(d)`.
- Tangential stretch is `length(cross(h,v))/(r*abs(d))`.
- The physical penumbra base width is `star_radius*f`.

These identities remove repeated light/planar/radial normalization and cancel
distance factors from the filter footprint. The existing prefix filter evaluates
the resulting coordinates. This is an algebraic simplification, not a lower
sample count or cached ring-visibility field. Floating-point reordering can still
produce small differences when prefix integrals are subtracted.

## Application measurement

RTX 4060 Ti, Release, 1600×900, default seed-42 procedural Earth rings, fixed eye
with changing yaw, cached atmospheric weights enabled, UI and hidden window,
VSync off. Each capture excludes 240 warmup frames and measures 600 frames × two
repetitions. The pre-change shaders were captured before and after the fast path.

| GPU scope | Original before | Fast geometry | Original after |
|---|---:|---:|---:|
| Fresh shadow estimates + accumulation | 2.072 ms | 1.797 ms | 1.922 ms |
| Atmospheric shadows including reprojection | 2.459 ms | 2.271 ms | 2.384 ms |
| Entire GPU frame | 3.518 ms | 3.407 ms | 3.603 ms |

The integration reduction is 6.5–13.3% against the two controls; the shadow pass
reduction is 4.7–7.6%. Unchanged passes also varied. Hidden-window presentation
dominated wall-frame timing, so these measurements establish a GPU-time result,
not a reliable application FPS improvement. Raw captures and a summary are in
`exports/benchmarks/ring-shadow-geometry-hidden-20261024/`.

A separate disabled-mode control measured 2.017 ms for integration and 2.369 ms
for the complete shadow pass, within the pre-change control range.

Exact-resolution atmosphere-only GPU shadow-pass means, three alternating-order
repetitions per view, eight warmup and 32 measured turning frames per mode:

| View | Resolution | Current | Fast geometry | Reduction |
|---|---|---:|---:|---:|
| Ground | 1600×900 | 2.018 ms | 1.918 ms | 4.9% |
| Earth filling screen | 1600×900 | 1.309 ms | 1.267 ms | 3.2% |
| Ground | 3840×2160 | 12.586 ms | 11.777 ms | 6.4% |
| Earth filling screen | 3840×2160 | 7.852 ms | 7.565 ms | 3.7% |

The ground view saves about 0.81 ms at 4K. Six ring-filter reads and two cached
weight reads per eligible sample remain; the experiment reduces surrounding
geometry work. The offscreen report is `tests/ring-shadow-geometry/offscreen.json`.

## Image comparison

257×259 linear HDR, exposure 8, cached weights enabled, 64 progressive frames:

| View | Fast/current RGB RMS | Largest RGB difference |
|---|---:|---:|
| Earth ground | 0.0000240 | 0.001879 |
| Earth filling the screen | 0.0000269 | 0.001465 |

The frozen first frames differ by 0.0000382 RMS for ground and 0.0000408 for the
close-up. The narrow-ring fixture contains visible shadowing and matches exactly;
grazing-star and multiple-star fallbacks also match exactly. Both softness limits
and transparent-prefix transport are included in the characterization. These are
fixture-specific results, not universal error bounds.

At 1600×900, frozen ground and close-up views at yaw offsets −15°, 0° and +15°
had a worst RGB RMS difference of 0.0000480 and largest pixel RGB-vector difference
of 0.002160, again at exposure 8. Comparison captures and quality JSON are retained
in `tests/ring-shadow-geometry/validated-quality/`.

The engine self-test and quality, rotation/history and shared surface-ring filter
checks passed. The combined GPU suite ran with Vulkan validation enabled and
reported no validation errors or warnings. The experiment was off by default during these comparisons; it is enabled in
the current saved-preference baseline.

## Reproduction

```powershell
cmake --build build --target StellarForge StellarRenderingTests --parallel 4
.\build\bin\tests\StellarRenderingTests.exe --ring-shadow-geometry-verify tests/ring-shadow-geometry/verify
.\build\bin\tests\StellarRenderingTests.exe --ring-shadow-geometry-benchmark tests/ring-shadow-geometry/offscreen.json
.\build\bin\StellarForge.exe --benchmark --benchmark-scene earth-ground-rings-turn --benchmark-variant fast-ring-shadows --benchmark-width 1600 --benchmark-height 900 --benchmark-hidden --benchmark-ui --benchmark-output exports/benchmarks/ring-geometry
```

The `baseline` variant keeps this experiment disabled. Both variants use cached
atmospheric weights. Match window mode, resolution and instrumentation between
captures, and choose fresh output directories. The offscreen probe alternates
method order over three repetitions at 1600×900 and 3840×2160, without readback or
presentation during measurement. It isolates atmosphere work and is not application
FPS. Local tests and images follow the repository's git-ignored `tests/` convention.
