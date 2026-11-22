# Closed-form atmospheric shadows

**Closed-form shadows** (`RenderSettings::atmosphere_shadow_closed_form`, enabled by
default in single-frame mode via `!atmosphere_shadow_temporal`) evaluate ring and eclipse
shadow deficits without raymarching. Unchecking it falls back to deterministic shadow
strata; **Progressive atmosphere shadows** keep jittered accumulation.

## Method

The [cached weights](atmosphere_shadow_cached_weights.md) already store the cumulative
in-scattered light `W(t)` along each view ray in 64 slices. For each merged shadow
interval the shader splits the ray into 4 strata on the cache's own spacing (square-root
inside the atmosphere, linear outside):

1. The stratum energy is `W(t1) - W(t0)`.
2. One midpoint fetch gives the fraction `a` of that energy in the first half. An
   exponential density with `x = 2 ln(a/(1-a))` matches it, giving the light-weighted
   mean `d(1/x - 1/(e^x - 1))` and variance `d^2(1/x^2 - e^x/(e^x - 1)^2)` (series form
   for `|x| < 0.5`).
3. Eclipses are evaluated at `mean ± σ` with the exact disc-overlap visibility; short
   eclipse transitions use up to four Gauss panels.
4. Rings split the stratum's part on one side of the ring plane into 3 equal
   pieces (`CLOSED_FORM_RING_PIECES`). For parallel light the ring-plane hit moves
   linearly with t, so each piece sweeps an exact radius range, which the ring LUT's
   prefix integral filters as one box. Each piece is weighted by its exact share of the
   fitted light, `F(u) = (1 - e^(-xu)) / (1 - e^(-x))`. Penumbra width, incidence and
   curvature shift come from the light centroid.
5. Loss is `energy × (1 - visibility)`, the product of sphere and ring visibilities.

Each side of the ring plane sees only the part of the star's disc beyond the plane, as
`ring_receiver_coordinates` models it for surfaces: that part's coverage, through light
tilted to its mean elevation. Away from equinox this is the whole, untilted disc on the
far side and nothing on the star's side. While the disc straddles the plane (within about
16 hours of Earth's equinox) both sides are shadowed. A stratum crossing the plane then
takes the side with more covered light and drops the other: evaluating both sides cost
0.13–0.18 ms in every view through register pressure, even with the second side as a
single box.

Uniform-in-t averaging inside a stratum converges only as O(1/N) and was worse than the
sampled path; the light-weighted moments are what make 4 strata sufficient. The first
version gave rings a single variance-matched box per stratum. On grazing views, where
one stratum crosses several shadow bands with its light concentrated at one end, that
smeared band edges (3.1% p99 on `saturn-limb`); 3 pieces halve it.

## Eligibility

The CPU sets header flag `y & 32` whenever progressive mode is off.
Ring geometry no longer clears it. It used to be cleared whenever a ring was within
`star radius + 3 × outer radius` of edge-on to the star, which sent every shadowed pixel
to the sampled march near equinox: `earth-equinox-rings` (Sun 0.15° from the equator,
1600×900) took 2.87 ms in atmosphere shadows, now 1.06 ms. The composite selects the dedicated `ATMOSPHERE_SHADOW_CLOSED_FORM` pipelines
(plain, transparency, VRS, VRS + transparency, coarse, coarse + transparency) only when
cached weights are also active, there is one star, no aerosol layer is shaped, and
reference mode is off. Anything else falls back to the sampled path, which also uses the
closed form per star where the same conditions hold.

The variants matter for speed: the combined shader is limited by register pressure, not
by the integration. Removing all shadow math from it kept 0.99 ms on earth-ground-rings;
compiling the other paths out gave 0.36 ms.

## Measurements

RTX 4060 Ti, 1600x900, high exposure, captures after 90 frames. Error is against the
256-step reference (`reference-shadows`), as a share of the unshadowed pixel's luminance,
clouds off. On `saturn-limb` the reference and converged progressive shadows
(`progressive-shadows`: 16 samples, 64 frames) agree to 0.08% at the 99th percentile.

| Scene | Sampled p99 / worst | Closed form p99 / worst |
|---|---:|---:|
| saturn-limb (VRS off) | 3.5% / 27% | 1.8% / 6.2% |
| earth-ground-rings | 4.5% / 6.5% | 1.5% / 1.6% |
| earth-full-rings | 0.53% / 0.74% | 0.51% / 0.64% |
| earth-eclipse-oblique | 2.0% / 32% | 0.42% / 6.9% |

Four ring pieces reach 1.4% p99 on `saturn-limb` for about 0.07 ms more.

Shadow pass, clouds on (ms):

| Scene | Sampled | Closed form |
|---|---:|---:|
| earth-ground-rings | 1.17 | 0.70 |
| earth-full-rings | 0.88 | 0.86 |
| saturn-limb | 0.54 | 0.76 |
| saturn | 0.15 | 0.18 |

On `saturn-limb` adaptive sampling makes the sampled path cheap, so the closed form
trades about 0.2 ms for accuracy there. Eclipse scenes are bound by a fixed 0.2–0.4 ms
per-pixel cost outside the integration.

## Comparing

Use the `sampled-shadows` benchmark variant for the previous path,
`reference-shadows` or `progressive-shadows` for ground truth, and `no-shadow-vrs` to
remove reconstruction from the comparison, with `--benchmark-capture` PFMs. Capture at
least 90 frames: Saturn and Earth content is still streaming in at 3 frames (2.5–126%
mean change), so short captures compare unsettled images.

- `earth-eclipse-oblique` places the annular-eclipse penumbra across the frame;
  ground-level views cannot show penumbra structure because nearby air dominates sky
  darkening.
- `saturn-limb` is a grazing view through Saturn's ring shadow bands toward the hazy
  limb, where single rays cross several bands.

Remaining error is concentrated in the sky haze just above the limb, in blocks matching
the cached weights' screen resolution; the sampled path, which uses the same cache,
shows it too.

Clouds receive the same eclipse and ring shadows as the surface
(`shaders/common/cloud_shadow.glsl`), applied to their direct and sky light. They previously used unshadowed sunlight, so only the
haze in front of them darkened.
