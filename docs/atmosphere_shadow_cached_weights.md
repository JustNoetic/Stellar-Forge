# Cached atmospheric shadow weights

Enabled by default for eligible columns. **Settings → Visibility → Cached shadow
weights** can restore native integration for comparison. Toggling the option
invalidates shadow history. Default procedural rings use seed 42.

## Calculation

The existing 192×108×64 RGBA16F aerial-perspective LUT stores cumulative,
unshadowed radiance along each atmospheric column. It already includes density,
observer attenuation, solar attenuation, scattering phase and multiple scattering.
Cached weights estimate the local contribution from the difference between two
adjacent depth slices, divided by their physical separation. It multiplies this
slope by the existing sample's distance weight.

This replaces repeated atmospheric lighting evaluation at each shadow sample with
two bilinear LUT reads. Sample positions, sample budgets, jitter, full-resolution
ring visibility, six-read ring filtering, normalized RGB energy/loss integration,
and progressive/rotation history remain in use. No new LUT, bake pass or
screen-resolution texture is allocated. Existing observer/light changes rebuild
the aerial LUT; rotating at a fixed eye reuses it.

Use requires aerial perspective, one star, a ring-shadow candidate, a path of at
least 2 km and an eye at least 10 m above the atmospheric bottom radius. Transparent
prefixes, multiple stars, short paths, and profiles requiring thin-layer transport
subdivision retain the original calculation. Dense-reference rendering stays native.

The cached derivative is approximate: angular interpolation, 64 depth slices,
half-float cancellation and interval-averaged light differ from the original local
evaluation. It does not remove per-pixel work or change its pixel-count scaling.
The ring intersection/filter and history bandwidth remain substantial costs.

## Timings

RTX 4060 Ti, Release, default procedural rings (seed 42), 24 October 2026.

Original prototype capture, ground view while turning at a fixed eye, 1600×900, UI,
VSync off; 180 warmup frames and 500 measured frames × two repetitions per mode:

| Metric | Native | Cached | Reduction |
|---|---:|---:|---:|
| Frame | 4.346 ms | 3.536 ms | 18.6% |
| GPU frame | 4.325 ms | 3.512 ms | 18.8% |
| Atmospheric shadow pass | 3.180 ms | 2.374 ms | 25.3% |
| Shadow integration/resolve | 2.849 ms | 2.036 ms | 28.5% |
| Shadow reprojection | 0.331 ms | 0.338 ms | — |

Exact-resolution offscreen atmosphere-only rendering, fixed eye with changing yaw;
three alternating-order repetitions, eight warmup and 32 measured frames per mode.
These are GPU shadow-pass means, including history work, **not application FPS**:

| View | Resolution | Native | Cached | Reduction |
|---|---|---:|---:|---:|
| Ground | 1600×900 | 3.450 ms | 2.334 ms | 32.3% |
| Earth filling screen | 1600×900 | 1.855 ms | 1.380 ms | 25.6% |
| Ground | 3840×2160 | 19.789 ms | 14.193 ms | 28.3% |
| Earth filling screen | 3840×2160 | 12.027 ms | 8.716 ms | 27.5% |

Absolute timings vary between captures. An earlier offscreen ground run measured
16.592 → 11.962 ms at 4K (27.9% reduction). A separate disabled-mode control
measured 3.897 ms/frame with the pre-prototype shaders and 3.703 ms with the new
shader's native branch; it showed no regression above run variation. CPU scopes
also drifted between launches; no CPU optimization is claimed.

## Remaining GPU cost

The cached 1600×900 application capture still spends 2.374 ms on atmospheric
shadows: 2.036 ms in integration/resolve (fresh samples and history accumulation)
and 0.338 ms in reprojection. Turning always evaluates fresh current-view samples,
even when it accepts reprojected history.

Each pixel still selects an adaptive 4–16-sample budget. A sample performs caster
visibility, ring-plane intersection, penumbra/curvature/footprint calculations,
up to six ring prefix-filter lookups per relevant ring host, and two aerial-weight
lookups on the cached path. The pass also reconstructs rays/endpoints, checks
support and history, and reads/writes full-resolution energy/loss images.
Transparent prefixes retain native weighting. The capture does not separately
time texture work and arithmetic within that shader.

Outside shadows, the same GPU frame spends approximately 0.493 ms drawing visible
rings, 0.185 ms on remaining atmosphere composition, 0.172 ms on Gaia culling and
0.134 ms drawing terrain. Total GPU frame time is 3.512 ms. These child scopes
must not be added again to their parent intervals. The remaining shadow estimate
is the largest measured target; LUT baking is effectively zero while turning at
a fixed eye.

## Image comparison

RTX 4060 Ti, 257×259 atmosphere-only HDR captures, exposure 8, frozen camera,
64 progressive frames, identical ring profile and rendering settings. RGB RMS
means the RMS length of the RGB difference vector, without tone mapping.

| View | Native vs dense RMS | Cached vs dense RMS | Cached vs native RMS | Largest cached/native RGB difference |
|---|---:|---:|---:|---:|
| Earth, 2 km above ground | 0.000666 | 0.001243 | 0.001127 | 0.006633 |
| Earth filling the screen | 0.000406 | 0.000443 | 0.000295 | 0.002072 |
| Synthetic orbital rings | 0.000026 | 0.000044 | 0.000039 | 0.002986 |

The ground view retains a small bias after convergence. A frozen first frame is
also different: cached/native RMS is 0.00775 for ground and 0.00201 for the close-up.
Those values include sparse-estimator differences; convergence reduces them.
These fixtures support trying the option, not an assertion of universal visual
equivalence. After visual comparison in the engine, cached weights became the
default; the native option and automatic fallbacks remain available.

The engine self-test, shadow, progressive-history and rotation suites passed.
The rotation suite also passed with cached weights enabled. Quality/fallback
contracts passed with Vulkan validation enabled and no validation messages;
multiple stars, disabled aerial perspective and transparent prefixes retained
native pixels. Comparison captures are in `tests/shadow-weights/validated-quality/`;
the final offscreen report is `tests/shadow-weights/offscreen-final.json`.

Default promotion was rebuilt and checked with the engine self-test and the
quality/fallback suite under Vulkan validation. Short application smoke captures
in `exports/benchmarks/shadow-weights-default-20261024/` confirm that `baseline`
records cached weights enabled and `native-shadow-weights` records them disabled.

## Reproduction

```powershell
cmake --build build --target StellarForge StellarRenderingTests --parallel 4
.\build\bin\tests\StellarRenderingTests.exe --cached-shadow-weights tests/shadow-weights/quality
.\build\bin\tests\StellarRenderingTests.exe --cached-shadow-rotation
.\build\bin\tests\StellarRenderingTests.exe --cached-shadow-benchmark tests/shadow-weights/offscreen.json
.\build\bin\StellarForge.exe --benchmark --benchmark-scene earth-ground-rings-turn --benchmark-variant cached-shadow-weights --benchmark-width 1600 --benchmark-height 900 --benchmark-ui --benchmark-detailed --benchmark-output exports/benchmarks/cached-weights
```

Compare application captures with an otherwise identical `native-shadow-weights`
capture and a fresh output directory. `baseline` now uses cached weights. The
historical prototype reports labeled `baseline` used native weights, as recorded
by their `shadow_cached_weights` fixture metadata. Offscreen measurements isolate atmosphere rendering and
exclude readback, surfaces, visible rings, UI and presentation; they are not
application FPS. The window system can clamp a requested 4K framebuffer; the normal
benchmark rejects that mismatch. Use the offscreen probe for exact 3840×2160 work.

Local images, raw half-float captures, quality JSON and test logs are retained in
`tests/shadow-weights/`; application reports are in
`exports/benchmarks/shadow-weights-20261024/`. The local rendering-test harness is
git-ignored under the repository's testing convention.
