# Atmospheric shadow deficit

The default [cached atmospheric weights path](atmosphere_shadow_cached_weights.md)
reuses aerial-perspective lighting for eligible single-star ring-shadow samples.
Ring visibility and filtering retain the existing calculation; unsupported columns
use native weights. The settings checkbox can restore native weights for comparison.

The separate [ring geometry experiment](atmosphere_ring_shadow_geometry.md) simplifies
repeated visibility projection math while retaining the existing filter and samples.

In single-frame mode, [deterministic shadow strata](atmosphere_shadow_strata.md) replace
jittered ring samples with exact stratum energies and swept prefix-filtered ring
visibility. They also sample wide eclipse penumbrae at stratum midpoints.
Progressive refinement keeps the jittered estimator described below.

`RenderSettings::atmosphere_shadows_enabled` (Atmospheric shadows in rendering settings, enabled by default) adds atmospheric shadows with full-resolution endpoints from external bodies and rings. `atmosphere_shadow_samples` selects a 4/8/16 integration-sample ceiling per evaluated ray/star (default 4), shared across primary and nearest-transparent endpoints. It works with the horizon-mapped aerial LUT and both hardware blend paths. No atmosphere shadow volume or shadowed sky LUT is allocated.

Opaque composition consumes the precomputed deficit for its exact sky/opaque depth and compiles out native endpoint integration. Transparent composition retains that fallback for intermediate fragment depths. An empty external-caster list disables shadow work in the per-body header. Source evaluation shares the [normalized incoming-light cache](atmosphere_finite_stars.md) with the clear sky/aerial bake when eligible.

Atmospheric shadows use the original normalized solar loss fraction architecture:
$$\text{fraction} = \text{clamp}\left(\frac{\sum_s \text{loss}_s}{\sum_s \text{energy}_s}, 0.0, 1.0\right)$$
where each star $s$ evaluates its scattered energy and shadow loss along the visible ray column. In composition, the modeled scattering loss $\Delta L = L \times \text{fraction}$ is subtracted from the clear-atmosphere radiance $L$.

## Mathematical formulation and benefits

1. **Exact zero in full umbra**: In full totality/umbra, $\text{loss}_s \equiv \text{energy}_s$, yielding $\text{fraction} \equiv 1.0$. Because the loss is applied as $L - L \times 1.0 = 0.0$, the discretization error between Riemann sum raymarching and cumulative angular LUTs cancels identically, achieving bitwise 0.0 extinction without light leaks or glowing halos.
2. **Bitwise untouched clear sky**: When unshadowed, $\text{loss}_s = 0.0 \implies \text{fraction} = 0.0$, leaving clear sky untouched.
3. **Independent stellar scattering**: In multi-star systems, unshadowed stars contribute their baseline in-scattered energy to the denominator, ensuring an eclipsed star subtracts only its own direct scattering contribution while leaving other stars completely intact.
4. **Rational observer-extinction proposal**: For camera rays entering the atmosphere, sampling uses a closed-form rational coordinate proposal based on the observer's extinction length:
   $$t = s_{\text{first}} + (s_{\text{last}} - s_{\text{first}}) \cdot \frac{q}{1 + c(1 - q)}$$
   $$ds = (s_{\text{last}} - s_{\text{first}}) \cdot \frac{1 + c}{(1 + c(1 - q))^2 \cdot N}$$
   with $c = 1.5 \cdot \text{dist} \cdot (\bar{\beta}_e)_0$. The exact distance Jacobian $ds$ eliminates numerical root-finding loops and binary searches, achieving accurate integration even at 4 and 16 samples.
5. **Immediate scene-motion response**: Eye translation, projection, atmosphere, lighting, caster and ring-profile changes restart from the current native estimate. Optional rotation-only reconstruction reuses compatible sky columns when the eye and scene inputs remain unchanged. Fixed single-frame integration remains available by disabling Progressive atmosphere shadows.
6. **Memory footprint**: The native shadow output uses two 2-layer textures (`shadow_fraction` at RGBA16F and `shadow_depth` at R32F), 24 B/px together (47.5 MiB at 1080p). Progressive mode adds 64 B/px of full-precision integral history (126.6 MiB at 1080p). Rotation adds a lazily allocated, shared 32 B/px scratch pair (63.3 MiB at 1080p, 253.1 MiB at 4K); VRS instead adds the shared scratch storage described below.

## Estimator bias and per-sample work

The native march estimates an RGB energy-weighted blocked fraction, not absolute sky radiance. Each sample evaluates local molecular/aerosol density and scattering, observer transmittance from the optical-depth LUT, solar transmittance and host visibility, the multiple-scattering LUT, and external body/ring visibility. Its weight is observer transmittance times stellar irradiance times the sum of direct and multiple-scattering sources times the distance weight. The same weight enters the energy denominator and, multiplied by the blocked fraction, the loss numerator. The clear radiance being corrected comes from the sky/aerial LUT at composition; the shadow march does not fetch that cumulative radiance at each sample.

This normalized ratio generally has finite-sample bias: the expectation of a ratio need not equal the ratio of the integrals. Shared numerator/denominator samples cancel much of their common energy variation and preserve the sampled clear/umbra identities. Rational or tangent-focused sampling, one shared IGN/STBN jitter across the ray/star's strata, analytic finite-disc body visibility and filtered ring extinction also reduce visible noise. LUT transport is deterministic and approximate; low noise alone does not establish physical accuracy. Progressive history sums energy and loss before dividing, reducing integration error without removing LUT/model approximations. A short transparent prefix can contain no samples on an individual frame, and narrow shadow features can fall between sample positions even when the full ray looks smooth.

External visibility multiplies the combined direct and multiple-scattering weight. Suppressing cached diffuse fill this way is a local approximation, not an eclipse-aware multiple-scattering solution. Likewise, multiple-star weighting uses the marched energy to apportion the combined LUT baseline; exact per-star isolation would require agreement between those transport representations. Samples outside shadow bounds still contribute to the energy denominator; skipping their transport without an equivalent full-column energy integral would change the estimator. The dense diagnostic uses the same transport approximations and tests integration quality rather than independent physical correctness. Its reference flag expands candidate intervals to the whole column, preserving the full-column denominator.

The native path skips interval merging for a single caster, since its sole interval is already complete. IGN/STBN jitter is evaluated once per ray/star outside the sample loop; the removed sample-index argument was unused, so this retains the existing sample sequence. These changes preserve sample budgets, distance proposals, visibility and normalization.

On the RTX 4060 Ti at 1920x1080 with a 16-sample ceiling, a before/after run of the seven-scene shadow benchmark measured 4.0–9.6% lower fresh single-frame shadow-pass time across the six shadowed scenes and fixed/adaptive/VRS modes. Refining progressive mode improved by 3.6–5.9%; completed-history times were effectively unchanged. Each version used three alternating shadow-on/off repetitions, eight warmup and sixteen measured frames (64 warmup for completed history). These synthetic single-caster measurements exclude presentation/readback and do not establish application FPS or multiple-caster gains. Reports are `tests/shadow-audit/baseline-16.json` and `tests/shadow-audit/candidate-16.json`.

The full `--atmosphere-shadows` suite passed before and after the cleanup with identical diagnostic logs and 59 byte-identical saved PPM previews. This includes adaptive 4/8/16-sample comparisons, dense references, VRS, progressive convergence, multiple stars, finite and transparent endpoints, and history invalidation.

## Progressive refinement

`atmosphere_shadow_temporal` (Progressive atmosphere shadows, disabled by default) accumulates the existing jittered integration over 64 stationary frames. It stores RGB energy and loss separately in two RGBA32F two-layer storage images, then divides accumulated loss by accumulated energy. It never averages per-frame normalized fractions. Counts and endpoint depths occupy the alpha channels. Each invocation owns its pixel's two endpoints, allowing in-place history updates without ping-pong images. Explicit compute read/write barriers order successive frames.

CPU validation compares the complete atmosphere uniforms, shadow inputs excluding the jitter sequence, retained body identity and definition/geometry/lighting revisions, ring profile upload revision and image identity, and consecutive preparation/composition. The atmosphere-relative eye position is also compared in double precision, without movement tolerances. A GPU comparison of both endpoint depths resets the pixel when either changes, including transparent fragments appearing or disappearing. Opaque and nearest-transparent integrals have separate storage and share a conservative pixel reset and sample sequence. Additional transparent endpoints retain their native fallback.

The first frame uses sequence zero and is identical to frozen native integration. Subsequent frames use the pixel's accumulated count as the jitter sequence. At 64 samples of the estimator, the pixel retains its output until invalidated. This stops residual flicker and avoids further transport evaluation for completed columns. Clear and full-umbra identities follow from the same accumulated numerator/denominator operations; a single noisy zero or one estimate does not override valid history.

Progressive mode uses native integration and bypasses VRS. Disabling it restores the single-frame VRS path and releases history through completion-based retirement. **Animate shadow jitter** (`atmosphere_shadow_animate_jitter`, enabled by default) advances the IGN/STBN sample positions each prepared frame without accumulating history, including while simulation time is paused. Disable it to freeze the single-frame noise pattern at sequence zero for inspection and deterministic comparisons. Progressive refinement advances its own sample sequence regardless of this control, and dense-reference integration stays deterministic. Disable VRS too for a native-resolution sampling comparison. Integral history is per visible shadowed atmosphere; continuously changing simulation inputs or eye translation can prevent refinement and incur history traffic without a convergence benefit.

### Rotation-only sky reuse

`atmosphere_shadow_rotation` (**Reuse history while turning**, disabled by default) allows camera rotation with exactly unchanged atmosphere-relative eye coordinates, projection, atmosphere/lighting/caster/ring inputs and frame continuity. The strict stationary key remains in use when rotation reuse is disabled. Multiple-star atmospheres retain the native reset path during rotation.

`shadow_reproject.comp` gathers previous energy/loss integrals onto the current grid using the previous-view/current-inverse-view homography. It reads the original history and writes two separate RGBA32F scratch images before the native accumulation pass modifies any history. Explicit compute read/write barriers separate those accesses and sequential scratch reuse by different atmospheres. Scratch is shared across bodies, allocated only after qualifying rotation, and completion-retired on resize or mode disable; stationary frames need no reprojection dispatch.

Reuse is restricted to sky endpoints without transparent fragments, and all contributing bilinear taps must meet these checks:

- Previous-view clip coverage and in-bounds taps, valid history count, matching sky classification, and no previous transparent prefix.
- Matching analytical ground-hit classification, log2 atmospheric path difference at most 0.02, and atmospheric entry difference at most 1% of max(path, entry distance, 1 km). Outside the atmosphere the latter includes the vacuum approach distance; it is not scaled solely by the shorter scattering column.
- The existing conservative penumbra/plane-crossing spatial guards on both current and previous rays; RGB fraction range at most 0.02 across previous taps.

The gather interpolates per-frame energy/loss means and conservatively uses the minimum tap age, capped at eight frames of prior weight. A fresh current-ray estimate rejects history if its RGB fraction differs by more than 0.05 or its energy differs by more than 75% of the larger estimate. The broad energy threshold accommodates sparse-estimator variance; it is not a radiance-error bound. Finite opaque/transparent endpoints, newly visible columns, narrow shadow detail and failed checks use fresh integration. A rejected fresh estimate is retained without old history.

Rotation always evaluates new integration samples. Accepted columns advance the global IGN/STBN sequence rather than repeating a count-capped jitter position. When rotation stops, the native stationary path resumes refinement to 64 frames and freezes. Capped history weight limits repeated interpolation blur; the spatial and fresh-estimate guards remain conservative heuristics, not a guarantee of identical current-ray radiance. Disable rotation reuse for strict stationary-only accumulation. The extra pass and scratch traffic trade GPU time and memory for smoother compatible sky columns.

### History diagnostics and auto horizon

Enable **Show Engine Telemetry Window** in settings to inspect history. **Reset inputs (current)** identifies changes to camera, projection, atmosphere, lighting, casters, ring profile uploads, sampling/transport settings, definitions, viewport or frame continuity. Multiple reasons can apply to one frame. These labels do not relax the strict history key.

The telemetry also shows refining/completed atmospheric columns, the minimum/maximum accumulated frame counts, depth-only pixel resets, and **Rotation reuse / fallback** column counts. Pixel statistics come from the latest completed GPU frame, so they can trail the current CPU reset reasons. Counters are reduced within each 8x8 workgroup, accumulated across visible atmosphere bodies, and read from per-frame mapped buffers only after their completion serial has passed. Closing telemetry disables the GPU counter work; no synchronous readback or GPU wait is added. Transparent endpoint changes contribute to the same depth-reset count.

Auto horizon uses exponential roll correction at 8..12 inverse seconds throughout its 200 km activation range. Once the residual roll falls below 0.00001 radians, the camera adopts the exact target up vector. This removes the altitude-boundary slowdown and the drifting tail that prevented stationary history from converging. Manual Q/E roll suspends alignment until released. **Auto horizon** in telemetry distinguishes inactive alignment, manual roll, settling and settled views. Unpaused motion and deliberate camera movement still invalidate history normally.

## Adaptive sample allocation

`atmosphere_shadow_adaptive` defaults to true and is exposed as Adaptive shadow samples. The selected count becomes Maximum shadow samples; disabling adaptation restores the fixed-count estimator for comparison. The dense diagnostic reference is unchanged.

The allocator estimates a local extinction-weighted scale height near the closest point of the visible column. It combines the finite path length with the radial view cosine and a curvature term to estimate slant complexity. Vertical sky rays, nearby ground and short finite segments can use four samples; long grazing columns retain the selected ceiling. Sample locations keep the existing rational observer-extinction or tangent-focused distance proposal and its distance weights. Allocation changes the number of strata, not the transport equation or the normalized loss/energy ratio.

Sharp-shadow protection uses the existing body/ring transition-width estimate over an approximate contributing span and can raise the budget back to the ceiling. Columns crossing a detached particle layer retain the ceiling. Multiple-star columns and primary rays sharing a nearest-transparent prefix also retain it so adaptation does not further reduce prefix coverage. These are conservative heuristics, not a guaranteed radiance-error bound; complex multi-band rings still require the selected ceiling to resolve their detail.

Fractional budgets are rounded with a stable, spatially decorrelated noise value instead of hard 4/8/16 angular bands. Integration jitter uses the existing IGN/STBN settings; progressive mode advances it using the pixel's history count. Four-sample ceilings produce the exact fixed four-sample result when adaptation is toggled. Empty single-star shadow intervals skip allocation and integration, retaining the existing clear-ray rejection.

## Software VRS

`atmosphere_shadow_vrs` defaults to true and is exposed as Atmospheric shadow VRS. This adapts the original project's analytical loss-fraction reconstruction to the Vulkan compute path. It is software shading reuse and requires no hardware fragment-shading-rate support. It composes with adaptive sample allocation: adaptation reduces integration work along a ray; VRS shares the fraction between compatible neighboring rays. Clear sky, aerial radiance, transmission, scene depth and final composition stay at native resolution.

The first dispatch evaluates shadow fractions on a grid anchored at native even/even pixels, with half the viewport width and height rounded upward. The original native ray, scene depth, sample budget and jitter coordinates are used. A second dispatch produces the existing native-resolution primary and transparent endpoint textures. Anchor pixels reuse their exact coarse result. Other pixels bilinearly reconstruct up to four neighbors only when all participating guides pass:

- Valid atmospheric coverage and supported shadow evaluation.
- Matching sky/opaque classification, log2 path difference at most 0.04 (about 2.8%), and atmospheric entry difference at most 2% of max(path, 1 km).
- RGB fraction range at most 0.02 over the contributing taps.
- A conservative spatial-detail check on both the native ray and coarse rays: the estimated solar penumbra must span at least three native pixel footprints throughout the visible column. Ring-plane/light-direction crossings and very small denominators reject reuse. This guards narrow features that a fraction-only test could miss between coarse samples.

**Shadow VRS rate** (`atmosphere_shadow_vrs_scale`, 2/4/8/16, default 4, log2 passed in `shadow_flags.y` bits 8–11) sets the coarse grid spacing per axis: 25%, 6.25%, 1.6% or 0.39% of native pixels. Coarser grids were tried: at 32 and 64 nearly every pixel fails the agreement and penumbra guards, so the pass costs as much as native integration. The penumbra guard scales with it (1.5 grid spacings of pixel footprint: three pixels at 2, six at the default 4). With deterministic strata at 3840×2160 the four Earth ring/eclipse views measured 2.9–3.6 ms native, 1.9–2.3 ms at 2, 1.3–1.7 ms at 4 and 1.1–1.7 ms at 8 (`--vrs-scale-benchmark`). RGB RMS against native was at most 0.0009 at 4. At 8, the 1600×900 ring limb had 559 pixels above 0.02. Remaining cost is native fallback and per-pixel setup.

Any failed check runs the existing native shadow integration immediately, with no temporal history. Missing border neighbors also fall back rather than extrapolate. The coarse pass skips integration where its spatial check rejects reuse, avoiding duplicate work in sharp regions. Pixels containing transparent fragments keep native primary/prefix integration, including their shared sample budget. Multiple-star atmospheres, dense-reference mode and viewports smaller than 64 pixels on either axis bypass the coarse pass entirely.

One scratch pair is shared sequentially across all atmosphere bodies: RGBA16F fractions and RGBA32F guides, six bytes per native pixel (11.9 MiB at 1080p, 47.5 MiB at 4K, with odd-dimension rounding). The guide stores log2 path, entry distance, sky/opaque class and coverage. Compute write/read barriers separate coarse generation, reconstruction and reuse by the next body. Resizing retires old images after GPU completion. No readback, CPU synchronization, history or per-body coarse allocation is needed in normal frames.

The guards are conservative heuristics, not a universal radiance-error bound. Integration jitter can reduce reuse, sharp shadows can stay mostly native, and the extra dispatch/guide traffic can slightly increase cost when little reuse is possible. The UI toggle provides an immediate native comparison. This optimization does not change sky-view LUT resolution or address its preexisting banding.

## Rotation measurements

RTX 4060 Ti, 1920x1080, 16-sample ceiling, adaptive integration, telemetry counters off. Serialized offscreen runs use 64 stationary frames followed by 8 rotating warmup and 16 measured frames. Median shadow-pass times include reprojection and fresh integration; these are synthetic GPU timings, not application FPS.

| View | Strict fresh integration while turning (ms) | Rotation reuse (ms) | Nested reprojection pass (ms) |
| --- | ---: | ---: | ---: |
| Inside atmosphere, zenith eclipse | 1.043 | 1.530 | 0.435 |
| Orbital rings | 0.779 | 0.898 | 0.098 |

At 257x255 and a four-sample ceiling, the smooth zenith eclipse fixture reused about 85% of columns during a 24-frame combined yaw/roll sequence. Average RGB-vector RMSE against dense current-view integration decreased from approximately 0.00553 to 0.00441 across both blend paths and noise choices. The broad orbital ring fixture remained close to the fresh result: limb and shadow-detail guards rejected much of that view. These are fixture-specific observations, not universal noise/error bounds. The focused rotation command writes comparison previews and `rotation-benchmark.json` when an output directory is supplied.

## Candidate culling and intervals

`CelestialRenderer::prepare` supplies the prepared ring hosts and ring extinction atlas. Each body-owned atmosphere table has a separate per-frame shadow SSBO. Body/star candidates use `eclipse_could_shadow` with the entire outer atmosphere as receiver. Ring candidates use the conservative surface predicate with the same outer bound, pixel footprint and prepared ring hosts. Candidates do not depend on caster visibility, hierarchy or mesh contribution. The receiver body itself is excluded because `sun_transmittance` already includes host shadowing. Body and star coordinates are subtracted in double precision before packing atmosphere-relative physical kilometres; oblate atmosphere integration points are transformed back to physical coordinates for shadow tests.

`eclipse_optics.glsl` and `ring_shadow_filter.glsl` accept explicit diagnostic/filter flags and are shared with the mesh path. Rings retain the approximate single-box penumbra, including stellar angular radius, distance-dependent softness, grazing stretch and pixel filtering. The 4097-by-64 prefix extinction table describes the ring profile and its illumination angles; it does not quantize shadow positions into atmosphere cells.

A full-resolution compute pass produces two prefix corrections: layer 0 for the opaque/sky endpoint and layer 1 for the nearest transparent fragment. Transparent fragments beyond the first use the analytic LUT baseline.

## Verification and cost

```powershell
.\build\bin\tests\StellarRenderingTests.exe --atmosphere-shadows tests/atmosphere-shadows/normalized
.\build\bin\tests\StellarRenderingTests.exe --atmosphere-shadow-benchmark tests/atmosphere-shadows/benchmark-normalized.json 16
.\build\bin\tests\StellarRenderingTests.exe --atmosphere-progressive-only tests/shadow-progressive
.\build\bin\tests\StellarRenderingTests.exe --camera-horizon-only
.\build\bin\tests\StellarRenderingTests.exe --atmosphere-rotation-only tests/shadow-rotation
```

The focused test suite tests:

- Progressive first-frame equivalence, dense-reference convergence, frozen completion, both blend paths/noise choices, and immediate resets for camera, projection, light, endpoints, resize, visibility and ring-profile uploads.
- Diagnostic input categories, GPU age/completion/depth counters, diagnostics toggling without history invalidation, and completed near-surface history after auto horizon settles.
- Camera roll convergence at 30/60/144 FPS across the activation range, bitwise stable settled matrices, manual roll, zero-time and vertical-view behavior.
- Rotation-only current-view comparisons against dense transport in both blend/noise modes, stationary completion after rotation, transparent/narrow-ring/multi-star fallback, double-precision eye/projection/light/caster invalidation, shared scratch across atmospheres and viewport sizes, normal frames in flight, and retained sky history during auto-horizon alignment.
- VRS reconstruction checks for fraction discontinuities, coverage, depth class, path/entry mismatch and borders.
- Native/VRS image comparisons at 257×259 and 1920×1080, frozen jitter, narrow ring detail, independent stars and transparent endpoints.
- Shared VRS scratch across two atmosphere bodies, odd/even resizes and tiny-viewport bypass.
- GPU allocation checks for vertical sky, nearby ground, short horizontal and grazing columns at all sample ceilings.
- Adaptive/fixed/dense comparisons outside and inside the atmosphere, at zenith, ground and horizon, with narrow rings and independent stars.
- Stable frozen noise and exact fixed/adaptive equivalence with a four-sample ceiling.
- Independent 32768-step double-precision single-scattering oracle checks at various caster offsets.
- Full-umbra ambient suppression with zero residual leak.
- Finite opaque and transparent endpoints in the horizon-mapped aerial LUT.
- Overlapping umbrae direct radiance subtraction.
- Multi-star independent stellar scattering preservation.
- Full-resolution own-ring atmosphere shadow contracts at orbital resolution (>32,000 affected pixels).
- Ring deficit RMSE vs. dense unclipped reference on the first frame, including fixed/adaptive comparisons at all sample ceilings.
- Unchanged optical and aerial cache tables across caster motion.

The benchmark reports fixed, adaptive, adaptive-plus-VRS, progressive accumulation, and completed progressive history separately for clear sky, eclipses, orbital rings, just-above/just-below boundary horizon views, and inside-atmosphere zenith/ground views. Progressive rows have `temporal: true`; `history_complete` distinguishes the 8-frame and 64-frame warmups before 16 measured frames. Other rows disable progressive mode explicitly. `atmosphere_composite` includes the nested `atmosphere_shadows` timer; do not add those times together. VRS splits the shadow timer into nested `atmosphere_shadow_coarse` and `atmosphere_shadow_resolve` timers; the benchmark reports these separately too.

Historical timings before the horizon-column migration: RTX 4060 Ti Release benchmark timings (1920x1080, 16 samples):

| Synthetic view | Shadows off composite (ms) | Shadows on composite (ms) | GPU shadow pass (ms) | Net shadow cost (ms) |
| --- | ---: | ---: | ---: | ---: |
| Clear close-up (Bruneton) | 0.194 | 0.217 | 0.000 | 0.023 |
| Eclipse close-up (Bruneton) | 0.189 | 2.681 | 2.355 | 2.492 |
| Orbital ring shadow (Bruneton) | 0.116 | 0.808 | 0.653 | 0.692 |

Performance comparison against legacy stochastic absolute-deficit method:
- **Ring shadows**: GPU shadow cost dropped from **17.58 ms** down to **0.65 ms** (**27x speedup**).
- **Close eclipses**: GPU shadow cost dropped from **6.82 ms** down to **2.35 ms** (**2.8x speedup**).
- **Memory footprint**: The historical single-endpoint comparison was 48 B/px to 12 B/px. The current two-endpoint implementation uses 24 B/px plus optional VRS scratch.
- **Stability**: Zero ghosting, zero history smearing during translation, and instantaneous single-frame convergence.

Adaptive allocation measurements on the RTX 4060 Ti at 1920×1080, 16-sample ceiling, serialized synthetic fixtures (three repetitions, 8 warmup + 16 measured frames):

| View | Fixed shadow pass (ms) | Adaptive shadow pass (ms) | Shadow pass reduction |
| --- | ---: | ---: | ---: |
| Close eclipse | 2.753 | 1.030 | 62.6% |
| Orbital rings | 0.783 | 0.785 | -0.3% |
| Just above atmosphere, horizon | 2.171 | 2.207 | -1.7% |
| Inside atmosphere, horizon | 2.998 | 2.540 | 15.3% |
| Inside atmosphere, zenith | 2.844 | 1.007 | 64.6% |
| Inside atmosphere, ground | 2.623 | 1.220 | 53.5% |

These measure the shadow pass, not application frame time or FPS. The sharp-shadow guard keeps difficult outside/horizon columns near their fixed budget, so allocation overhead can slightly increase their cost. Boundary-horizon and near-ground quality fixtures retained the fixed image; the orbital-ring diagnostic at a 16-sample ceiling had adaptive-to-fixed RGB RMS difference 0.00066 at exposure 8, with dense-reference RMSE increasing from 0.00034 to 0.00092. The narrow-ring diagnostic changed by 0.00016 RMS. These are fixture-specific error bounds, not universal smoothness guarantees. Reports and comparison images are in `scratch/shadow-adaptive/`.

## Software VRS measurements

RTX 4060 Ti, 1920×1080, three repetitions with 8 warmup and 16 measured frames per shadow toggle. These are median GPU shadow-pass times, including both coarse and native resolve dispatches, compared with adaptive sampling already enabled. They exclude presentation/readback and are not application FPS.

16-sample ceiling:

| View | Adaptive native (ms) | Adaptive + VRS (ms) | Shadow pass reduction |
| --- | ---: | ---: | ---: |
| Clear sky | 0.000 | 0.000 | — |
| Close eclipse | 1.033 | 0.698 | 32.4% |
| Orbital rings | 0.782 | 0.723 | 7.6% |
| Just above atmosphere, horizon | 2.203 | 2.279 | -3.5% |
| Inside atmosphere, horizon | 2.529 | 2.483 | 1.8% |
| Inside atmosphere, zenith | 1.006 | 0.839 | 16.6% |
| Inside atmosphere, ground | 1.220 | 0.914 | 25.0% |

4-sample ceiling:

| View | Adaptive native (ms) | Adaptive + VRS (ms) | Shadow pass reduction |
| --- | ---: | ---: | ---: |
| Clear sky | 0.000 | 0.000 | — |
| Close eclipse | 0.980 | 0.669 | 31.8% |
| Orbital rings | 0.311 | 0.323 | -3.9% |
| Just above atmosphere, horizon | 0.782 | 0.849 | -8.5% |
| Inside atmosphere, horizon | 1.024 | 0.985 | 3.8% |
| Inside atmosphere, zenith | 0.932 | 0.795 | 14.7% |
| Inside atmosphere, ground | 0.891 | 0.765 | 14.1% |

At exposure 8, the six 1080p quality fixtures (fixed and adaptive integration) had a maximum native-to-VRS RGB RMS difference of 0.000421 and maximum pixel RGB-vector difference of 0.024896. The narrow-ring and independent-star fixtures stayed pixel-identical. These are fixture-specific observations, not a universal error bound. Vulkan validation, odd/even resizing, shared scratch across two bodies, native transparent endpoints and frozen jitter are covered by the focused suite. Reports and images are in `scratch/shadow-vrs/`.
