# Approximate atmospheric eclipse lighting

Implemented 2026-10-23. The target is believable eclipse colour with bounded
rendering cost. The rejected per-fragment atmospheric integration is not used.

Surfaces, both terrain paths, rings and reflected point sources share an RGB
eclipse calculation. An atmospheric caster uses one filtered lookup of positive
RGB visibility; clear rays and airless casters retain the existing circle/ellipse
geometric path. There is
no altitude integration, source quadrature or inverse solve in the draw shaders.

`RenderSettings::eclipse_refraction_enabled`, exposed as **Eclipse refraction**,
controls this lighting independently of observer-position refraction. Hiding
atmospheres also disables it. Turning it off restores the geometric image.

## Optical approximation

`src/physics/eclipse_shadow.hpp` prepares 128 limb samples from an immutable
`PreparedAtmosphere`. Bending uses the local exponential-atmosphere estimate
`delta = (n-1)*sqrt(2*pi*r/H)`, with `H = -(n-1)/(dn/dr)`. One green-channel
bending field is shared by RGB. A 0.1 rad limit bounds this small-angle model.
The opaque ground/cloud radius and exterior critical-refraction branch are
respected; the observer solver's virtual below-ground floor is not used.

RGB transmission uses a 64-sample straight tangent column through the actual
prepared molecular density, mixed absorption, Mie extinction, aerosol layers,
and supplied ozone profile. This approximates extinction along a bent path;
each species keeps its own vertical distribution. Dense and critical atmospheres
are approximations, not a scientific ray-tracing claim.

The normalized finite-source map is `y = p - (q/alpha)*delta(p)`, where `p` is
impact distance in opaque radii, `q` is the stellar radius projected to the caster
plane in the same units, and `alpha` is the star's radius/distance at the caster.
The finite-distance approximation is strongest when the star is far compared
with the caster radius. Both signed branches contribute. A uniform solar disc
regularizes the axis; short fixed quadrature over clipped image intervals occurs
only in the background bake. There is no singular point-source magnification in
the draw path.

The stored visibility is atmospheric annulus light plus visibility outside the
top atmosphere. It includes atmospheric dimming as well as refracted fill.
The bake stores `log2(max(visibility, 1e-12))` and the draw shader decodes the
filtered sample with `exp2`. No signed correction is added to a separately
evaluated geometric shadow. Local enhancement above one is allowed; an internal
correction cap of eight bounds extreme focusing in this visual approximation.

## Texture coordinates and cache

Each table is **256 by 64 RGBA32F**, 256 KiB. The source-size coordinate is
piecewise logarithmic over `q = 0.002..4`, with an exact `q=1` row separating
total and annular configurations. The other coordinate is
`v = separation/(opaque_radius + projected_source_radius)`. Source-dependent
intervals align the inner contact `abs(1-q)/(1+q)` at texture X=0.375 and outer
contact at X=0.75 for every source-size row. The rational curve
`t*t/(t*t+(1-t)*(1-t))` concentrates samples at both ends of each interval;
an outer region covers secondary-image support beyond the geometric penumbra.
The table support depends on atmosphere-top radius and `max(delta)/alpha`.
The optical result blends with geometric visibility during initial activation
and at the source-size domain edges; outside it the model uses geometric
visibility. The log representation preserves deep-umbra colour under filtering.

The original signed-correction representation could interpolate to negative
green/blue values near inner contact. Runtime clamping produced a saturated
red band at high exposure. Positive log visibility and source-aligned contacts
remove this cancellation artifact without adding texture reads or draw-time
integration. Active atmospheric shadows use the projected directional radius;
the geometric ellipse remains the fallback and off-state result.

`EclipseRefractionRenderer` owns a 16-layer atlas: 15 bounded cache entries and
one synthetic performance-test layer. The atlas occupies 4 MiB, with a 2 MiB
staging ring. Cache keys retain the prepared atmosphere's shared ownership and
a source-angle bucket spaced by a factor of 1.025. Body moves and camera moves
do not rebake if they retain the same optical identity and angular-size bucket.
Changing radius, composition, pressure, aerosols or ozone changes the prepared
optical identity. Star radius/distance changes request another bucket.

Only potentially shadowing atmospheric-caster/star pairs request tables.
One background CPU job runs at a time; the render thread polls completion and
never waits for a bake. Finished results upload through `StagingUploads` and
layer-specific shared RHI transitions. Replacement respects completion serials.
A nearby cached source-angle bucket can remain active while a replacement is
built. The first table fades in over 150 ms. Without a ready/cacheable table the
receiver uses geometric shadows; no expensive runtime fallback exists.

Per-frame caster/star records are appended to the existing eclipse-word buffer.
`SceneCelestialUBO::eclipse_params.y` is their absolute word offset, or zero when
inactive; each record stores layer-plus-one, source angle, support and fade.
The sampled atlas uses binding **26**, including the point vertex path.
`BodyInstanceGPU` has no new fields or layout changes. CPU candidate bounds
include the atmosphere top and the baked maximum bend with a 1% margin (the
pre-bake eligibility check uses a conservative 0.12 rad envelope). Casters
can therefore affect pixels outside the original solid penumbra.

## Validation and performance

The final optical-table gate used a Moon at 1.9 radii camera distance at
2560 by 1440, approximately half-screen coverage. On the RTX 4060 Ti, three
alternating-order repetitions with 12 warmup and 40 measured frames per mode
produced these offscreen celestial GPU medians:

| Scene | Geometric only | Cached optical model | Added |
|---|---:|---:|---:|
| Totality | 0.478208 ms | 0.539648 ms | 0.061440 ms |
| Partial eclipse | 0.534528 ms | 0.561152 ms | 0.026624 ms |
| Clear | 0.407552 ms | 0.407552 ms | 0 ms |

All pass the 0.1 ms added-median gate. A first run exposed unnecessary clear-scene
candidate work from a fixed 0.12 rad bound; using the baked maximum bend removed
it. The post-contact-fix report is `tests/eclipse-transition-final-performance/physical-performance.json`. These
are paired rendering-pass timings, excluding readback and presentation, not
application FPS or a guarantee for every scene. Desktop/GPU scheduling outliers
remain in the raw distributions. Earth table preparation takes about 60–64 ms
on the background CPU thread, without a render-thread bake wait.

The analytical CPU tests compare transparent/no-bend limits and a 128-ray,
64-column profile with a denser 1024-ray, 512-column version of the same model.
Central Earth RGB visibility is approximately
`(0.00101538, 0.0000352079, 0.0000000715389)` versus
`(0.00101528, 0.0000351725, 0.0000000710520)` for that denser model. Sampled table
interpolation/coarse-model RGB-vector error reached 0.0078 in normalized full-light
units. A dense contact sweep over `q=0.2..0.35` has under 1% relative RGB error
and no outward brightness reversal; the prior signed table reached negative
visibility of -0.00181 in that interval. This is visual approximation validation, not comparison against a complete
bent-ray/scattering ground truth. Added aerosol extinction and a raised opaque
deck darken the modeled umbra.

Rendered contracts cover a recentered physical Earth/Moon fixture, cached and
reference terrain, rings, reflected point sources, independent stars, optical
identity invalidation, camera reuse and full-image off-state restoration. An
additional high-exposure oblate Earth fixture sweeps the colour ratio through
inner contact to reject an isolated saturated band.
The contact fix passed the CPU model contracts, these GPU contracts with Vulkan
validation enabled (no reported errors), and existing `--eclipse-lighting` cases.
The initial integration also passed `--rings-only` and the application's
three-frame `--self-test`.
Default exposure makes totality very dim; preview exposures are explicitly
increased to show its colour.

Reproduction:

```powershell
cmake --build build --target StellarEclipseShadowModelTests StellarRenderingTests
.\build\bin\tests\StellarEclipseShadowModelTests.exe
.\build\bin\tests\StellarRenderingTests.exe --eclipse-refraction tests/eclipse-lookup-contracts
.\build\bin\tests\StellarRenderingTests.exe --eclipse-lookup-gate tests/eclipse-lookup-gate
.\build\bin\tests\StellarRenderingTests.exe --eclipse-refraction-benchmark tests/eclipse-lookup-final
.\build\bin\tests\StellarRenderingTests.exe --eclipse-refraction-preview tests/eclipse-lookup-preview
.\build\bin\tests\StellarRenderingTests.exe --eclipse-transition tests/eclipse-transition-after
```

## Scope and source

This change colours incident eclipse light. It does not render the glowing
Earth-atmosphere ring viewed from the Moon. It also does not change atmospheric
shadow-deficit transport, resolved stellar images, scattered-light fill, or
secondary-source preparation. Atmospheric shadows use a directional-radius
approximation for oblateness; clear/off-state shadows retain the geometric
ellipse. Multiple casters still multiply approximate visibility.

The precomputed-shadow architecture is informed by
[Schneegans et al., *Physically Based Real-Time Rendering of Eclipses*, 2025](https://diglib.eg.org/items/b2015570-0648-4888-806c-847ba8224cb6).
This implementation uses an analytical optical model and positive log-visibility
tables; it is not a port of the paper's full refracted/scattered atmosphere or
limb-image implementation. The prior review and rejected method are recorded in
[the assessment](eclipse_refraction_assessment.md).
