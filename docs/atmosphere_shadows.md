# Atmospheric shadow deficit

`RenderSettings::atmosphere_shadows_enabled` (Atmospheric shadows in rendering settings, enabled by default) adds atmospheric shadows with full-resolution endpoints from external bodies and rings. `atmosphere_shadow_samples` selects a 4/8/16 integration-sample ceiling per evaluated ray/star (default 16), shared across primary and nearest-transparent endpoints. It works with the horizon-mapped aerial LUT and both hardware blend paths. No atmosphere shadow volume or shadowed sky LUT is allocated.

Atmospheric shadows use the original normalized solar loss fraction architecture:
$$\text{fraction} = \text{clamp}\left(\frac{\sum_s \text{loss}_s}{\sum_s \text{energy}_s}, 0.0, 1.0\right)$$
where each star $s$ evaluates its scattered energy and shadow loss along the visible ray column. In composition, the exact direct solar loss $\Delta L = L \times \text{fraction}$ is subtracted from the clear-atmosphere radiance $L$.

## Mathematical formulation and benefits

1. **Exact zero in full umbra**: In full totality/umbra, $\text{loss}_s \equiv \text{energy}_s$, yielding $\text{fraction} \equiv 1.0$. Because the loss is applied as $L - L \times 1.0 = 0.0$, the discretization error between Riemann sum raymarching and cumulative angular LUTs cancels identically, achieving bitwise 0.0 extinction without light leaks or glowing halos.
2. **Bitwise untouched clear sky**: When unshadowed, $\text{loss}_s = 0.0 \implies \text{fraction} = 0.0$, leaving clear sky untouched.
3. **Independent stellar scattering**: In multi-star systems, unshadowed stars contribute their baseline in-scattered energy to the denominator, ensuring an eclipsed star subtracts only its own direct scattering contribution while leaving other stars completely intact.
4. **Rational observer-extinction proposal**: For camera rays entering the atmosphere, sampling uses a closed-form rational coordinate proposal based on the observer's extinction length:
   $$t = s_{\text{first}} + (s_{\text{last}} - s_{\text{first}}) \cdot \frac{q}{1 + c(1 - q)}$$
   $$ds = (s_{\text{last}} - s_{\text{first}}) \cdot \frac{1 + c}{(1 + c(1 - q))^2 \cdot N}$$
   with $c = 1.5 \cdot \text{dist} \cdot (\bar{\beta}_e)_0$. The exact distance Jacobian $ds$ eliminates numerical root-finding loops and binary searches, achieving accurate integration even at 4 and 16 samples.
5. **Immediate motion response**: Progressive refinement retains history only for identical rendering inputs. Camera, projection, atmosphere, lighting, caster and ring-profile changes restart from the current native estimate. There is no reprojection or history blend across changed columns. Fixed single-frame integration remains available by disabling Progressive atmosphere shadows.
6. **Memory footprint**: The native shadow output uses two 2-layer textures (`shadow_fraction` at RGBA16F and `shadow_depth` at R32F), 24 B/px together (47.5 MiB at 1080p). Progressive mode adds 64 B/px of full-precision integral history (126.6 MiB at 1080p); VRS instead adds the shared scratch storage described below.

## Progressive refinement

`atmosphere_shadow_temporal` (Progressive atmosphere shadows, enabled by default) accumulates the existing jittered integration over 64 stationary frames. It stores RGB energy and loss separately in two RGBA32F two-layer storage images, then divides accumulated loss by accumulated energy. It never averages per-frame normalized fractions. Counts and endpoint depths occupy the alpha channels. Each invocation owns its pixel's two endpoints, allowing in-place history updates without ping-pong images. Explicit compute read/write barriers order successive frames.

CPU validation compares the complete atmosphere uniforms, shadow inputs excluding the jitter sequence, retained definition identity/generation/revisions, ring profile upload revision and image identity, and consecutive preparation/composition. There are no movement tolerances. A GPU comparison of both endpoint depths resets the pixel when either changes, including transparent fragments appearing or disappearing. Opaque and nearest-transparent integrals have separate storage and share a conservative pixel reset and sample sequence. Additional transparent endpoints retain their native fallback.

The first frame uses sequence zero and is identical to frozen native integration. Subsequent frames use the pixel's accumulated count as the jitter sequence. At 64 samples of the estimator, the pixel retains its output until invalidated. This stops residual flicker and avoids further transport evaluation for completed columns. Clear and full-umbra identities follow from the same accumulated numerator/denominator operations; a single noisy zero or one estimate does not override valid history.

Progressive mode uses native integration and bypasses VRS. Disabling it restores the existing single-frame VRS path and releases history through completion-based retirement. The extra memory is per visible shadowed atmosphere; continuously changing simulation or camera inputs can prevent refinement and incur history traffic without a convergence benefit. This mode guarantees no reuse across changed inputs represented by the renderer, not temporal reconstruction during motion.

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

Any failed check runs the existing native shadow integration immediately, with no temporal history. Missing border neighbors also fall back rather than extrapolate. The coarse pass skips integration where its spatial check rejects reuse, avoiding duplicate work in sharp regions. Pixels containing transparent fragments keep native primary/prefix integration, including their shared sample budget. Multiple-star atmospheres, dense-reference mode and viewports smaller than 64 pixels on either axis bypass the coarse pass entirely.

One scratch pair is shared sequentially across all atmosphere bodies: RGBA16F fractions and RGBA32F guides, six bytes per native pixel (11.9 MiB at 1080p, 47.5 MiB at 4K, with odd-dimension rounding). The guide stores log2 path, entry distance, sky/opaque class and coverage. Compute write/read barriers separate coarse generation, reconstruction and reuse by the next body. Resizing retires old images after GPU completion. No readback, CPU synchronization, history or per-body coarse allocation is needed in normal frames.

The guards are conservative heuristics, not a universal radiance-error bound. Integration jitter can reduce reuse, sharp shadows can stay mostly native, and the extra dispatch/guide traffic can slightly increase cost when little reuse is possible. The UI toggle provides an immediate native comparison. This optimization does not change sky-view LUT resolution or address its preexisting banding.

## Candidate culling and intervals

`CelestialRenderer::prepare` supplies the prepared ring hosts and ring extinction atlas. Each body-owned atmosphere table has a separate per-frame shadow SSBO. Body/star candidates use `eclipse_could_shadow` with the entire outer atmosphere as receiver. Ring candidates use the conservative surface predicate with the same outer bound, pixel footprint and prepared ring hosts. Candidates do not depend on caster visibility, hierarchy or mesh contribution. The receiver body itself is excluded because `sun_transmittance` already includes host shadowing. Body and star coordinates are subtracted in double precision before packing atmosphere-relative physical kilometres; oblate atmosphere integration points are transformed back to physical coordinates for shadow tests.

`eclipse_optics.glsl` and `ring_shadow_filter.glsl` accept explicit diagnostic/filter flags and are shared with the mesh path. Rings retain the approximate single-box penumbra, including stellar angular radius, distance-dependent softness, grazing stretch and pixel filtering. The 4097-by-64 prefix extinction table describes the ring profile and its illumination angles; it does not quantize shadow positions into atmosphere cells.

A full-resolution compute pass produces two prefix corrections: layer 0 for the opaque/sky endpoint and layer 1 for the nearest transparent fragment. Transparent fragments beyond the first use the analytic LUT baseline.

## Verification and cost

```powershell
.\build\StellarRenderingTests.exe --atmosphere-shadows scratch/atmosphere-shadows/normalized
.\build\StellarRenderingTests.exe --atmosphere-shadow-benchmark scratch/atmosphere-shadows/benchmark-normalized.json 16
.\build\StellarRenderingTests.exe --atmosphere-progressive-only scratch/shadow-progressive
```

The focused test suite tests:

- Progressive first-frame equivalence, dense-reference convergence, frozen completion, both blend paths/noise choices, and immediate resets for camera, projection, light, endpoints, resize, visibility and ring-profile uploads.
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
