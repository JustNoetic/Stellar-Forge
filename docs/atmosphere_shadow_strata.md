# Deterministic atmospheric shadow strata

Enabled by default. **Settings → Atmosphere → Shadow sampling & performance →
Deterministic shadow strata** (`RenderSettings::atmosphere_shadow_analytic`,
shader flag `shadow_flags.y & 16`) restores the jittered estimator for comparison.
Toggling it invalidates shadow history.

## Single-ring columns

Eligible columns use [cached weights](atmosphere_shadow_cached_weights.md), one
star, one ring host, no body casters, the default three-box filter and a stellar
elevation above the ring plane of more than the stellar radius plus the column
length. Ring shadows are then integrated without sample jitter or noise.

The column keeps the existing rational observer-extinction or tangent-focused
proposal `t(q)` and its 4/8/16 stratum budget. Each stratum `[q_i, q_{i+1}]` uses:

1. **Exact stratum energy.** `E_i = L(t_{i+1}) - L(t_i)` from the cumulative
   aerial-perspective LUT, one 3D read per boundary (`N+1` per column, versus `2N`
   slope reads for jittered cached weights). Energies telescope to the clear
   column radiance, so the clear-sky and full-umbra identities still hold exactly.
2. **Energy shape within the stratum.** The proposal makes energy roughly uniform
   in `q`. The neighbouring stratum energies give a linear density correction
   `rho(x) = 1 + a*x` on `x in [-1, 1]` with
   `a = (E_{i+1} - E_{i-1}) / (4 E_i)` (one-sided at the ends, clamped to ±1).
3. **Swept ring visibility.** For a distant star the ring-plane hit point is
   affine in `t`, so a stratum sweeps a narrow range of hit radii. Two
   density-weighted Gauss–Legendre nodes (`x = ±1/sqrt(3)`) give the mean radius
   and its variance. Both are exact for a linear density and linear radius. The
   sweep is represented by an equivalent box with half-width `sqrt(3*w0*w1)*|r1-r0|`.
4. **Prefiltered prefix lookup.** `ring_swept_filter` combines the sweep with the
   existing three-box penumbra filter by variance (`h_j' = sqrt(h_j^2 + s^2)`).
   It is exactly `ring_shadow_filter` at zero sweep. When the sweep exceeds three
   penumbra widths, the three boxes collapse into one variance-matched box
   (`0.728096 h^2 + s^2`), two prefix reads instead of six.
5. **Ring-plane crossings.** Strata that cross the ring plane are clipped at the
   exact (linear-in-`t`) crossing. The shadow-side energy portion is
   integrated under the same linear density, so it changes continuously.

The penumbra width, curvature shift, grazing stretch, incidence row and pixel
footprint are evaluated once per stratum at the density-weighted centre.
The full-shadow cutoff (`visibility <= 1e-4`) is applied per stratum.

The estimator is deterministic: no IGN/STBN fetch, no sharp-shadow budget probe
and no fractional sample-count dithering. Progressive history and rotation reuse
still run but have nothing to converge. Its error is a smooth bias from the
residual within-stratum energy shape; it falls about as `N^-2`.

## Body casters (eclipses)

With the option on, eclipse-only columns also become eligible for cached
weights. Body visibility stays a point evaluation at the stratum sample. The
jitter amplitude becomes `smoothstep(0.25, 1, stratum_length / transition)`,
where `transition = 2*min(R_polar, R_star*along/distance) / |direction x light|`
is the penumbra crossing length along the ray. Wide penumbrae (such as the Moon
on Earth) use deterministic midpoint strata. Sharp shadows from small nearby
casters keep full jitter. The amplitude changes continuously, so no seam
appears between the two regimes.

Samples outside the single-caster shadow interval skip visibility evaluation;
the interval is conservative, so their visibility is exactly one.

## Unchanged paths

Multiple stars, multiple ring hosts, rings combined with body casters, grazing
stars, transparent prefixes, thin-layer transport subdivision, disabled aerial
perspective, single-box/linear diagnostics and dense-reference integration keep
the jittered native or cached estimator.

Progressive refinement disables the option (the flag is cleared while
`atmosphere_shadow_temporal` is on): accumulated jitter converges toward the dense
reference, while deterministic strata keep their small bias.

## Measurements

RTX 4060 Ti, exposure 8 linear HDR, mean RGB RMS against dense integration over
eight animated-jitter frames, 4-sample ceiling (`--analytic-shadows`):

| View | Jittered error | Jittered frame-to-frame | Deterministic error | Deterministic frame-to-frame |
|---|---:|---:|---:|---:|
| Earth ground, rings, 1600×900 | 0.229 | 0.413 | 0.064 | 0 |
| Earth filling screen, rings, 1600×900 | 0.0111 | 0.0182 | 0.0060 | 0 |
| Earth ground in lunar eclipse, 1600×900 | 0.0048 | 0.0085 | 0.0022 | 0.0002 |
| Earth lunar eclipse from orbit, 1600×900 | 0.00053 | 0.00075 | 0.00058 | 0 |
| Fixture ring orbit | 0.0037 | 0.0063 | 0.0027 | 0 |

With 8 strata the Earth ground ring view reaches 0.014 (jittered 0.062).
Sharp small-caster eclipse fixtures keep jitter and are unchanged.

GPU shadow pass at 3840×2160, median, default single-frame settings, measured
with an earlier revision of the ring path that used four moment nodes instead of
the two density-weighted nodes now used. The application was running on the
same GPU, so 1600×900 timings were too noisy to report:

| View | Jittered | Deterministic, 4 strata | Deterministic, 8 strata |
|---|---:|---:|---:|
| Earth ground, rings | 4.28 ms | 3.06 ms | 4.39 ms |
| Earth filling screen, rings | 3.73 ms | 3.65 ms | 3.79 ms |
| Lunar eclipse from orbit | 6.42 ms | 3.67 ms | 3.95 ms |
| Ground in lunar eclipse | 5.89 ms | 3.03 ms | 4.02 ms |

Eclipse savings come mainly from cached aerial weights replacing native transport
weights in eclipse-only columns. These are synthetic GPU timings, not application FPS.

## Reproduction

```powershell
.\build\bin\tests\StellarRenderingTests.exe --analytic-shadows tests/analytic-shadows/quality
.\build\bin\tests\StellarRenderingTests.exe --analytic-shadow-benchmark tests/analytic-shadows/benchmark.json
```

`STELLAR_ANALYTIC_FILTER` (comma-separated scene substrings) and
`STELLAR_ANALYTIC_SAMPLES` (fixed sample ceiling) narrow the quality run.
