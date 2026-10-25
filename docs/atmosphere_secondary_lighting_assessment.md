# Secondary atmospheric lighting assessment

24 October 2026. Secondary-lighting design assessment. The primary sky/aerial integration merge identified below has since been implemented; secondary atmospheric lighting remains proposed.

The promising implementation is a separately cached, lower-resolution secondary-radiance LUT using the existing horizon mapping and cumulative transport. Planetshine can reuse the prepared reflected sources. Ringshine needs an incident-light representation suited to scattering. Simply restoring the removed full-resolution combined bake reproduces an already measured moving-light bottleneck.

## Current integration and an immediate opportunity

`shaders/atmosphere/sky_view.comp` and `aerial_perspective.comp` trace the same 192 × 108 angular columns. Both call `integrate_atmosphere` over 63 intervals with two midpoint cells per interval. They share the same segment clipping, distance mapping, medium, stellar source terms and cumulative optical depth. Ignoring empty rays and detached-layer subdivisions, each dispatch evaluates 2,612,736 medium cells; running both doubles that work.

The renderer now uses the aerial pass's final accumulator to write the sky radiance and throughput images when aerial perspective is enabled. The 2D-only bake remains when aerial perspective is disabled. This avoids one complete column march without reducing angular resolution or integration quality. The existing aerial-valid flag preserves reuse on unchanged toggles and forces a combined refresh after changes made while disabled. Both sky outputs receive storage-write/read barriers. See [the atmosphere documentation](atmosphere.md) for parity tests and timer semantics, and the [merge report](atmosphere_column_merge.md) for measured preparation savings.

The [24 October audit](../exports/benchmarks/audit-20261024/timings.md) reports approximately 0.189–0.198 ms for sky and 0.179–0.191 ms for aerial updates in live Earth views on the RTX 4060 Ti. These adjacent pass timings identify substantial duplicate work; they are not a measured prediction of the merged pass's whole-frame saving. The removed shine experiment already implemented this merge for its LUT variant, so this is an identified opportunity in the current tree, not a new algorithm.

The general source/transport separation follows the existing Hillaire implementation. Its [reference integrator](https://github.com/sebh/UnrealEngineSkyAtmosphere/blob/master/Resources/RenderSkyRayMarching.hlsl) evaluates incident illumination, phase-weighted scattering, extinction and accumulated view transmission separately.

## What the earlier experiment established

The retained [comparison report](../tests/shine-approaches/removed-2026-10-22.md) and [raw warmed timings](../tests/shine-approaches/performance-warmed.json) compare analytical per-pixel transport, a shared-node approximation and full-resolution secondary LUTs. Those commands and production implementations were subsequently removed.

Representative total GPU milliseconds, RTX 4060 Ti, 1920 × 1080:

| Case | Analytical reference | Full-resolution LUT | Shine off |
| --- | ---: | ---: | ---: |
| Ground sky, four sources, cached | 0.633 | 0.118 | 0.123 |
| Ground sky, four sources, moving lights | 0.637 | 0.709 | 0.123 |
| Orbit, four sources plus ring, cached | 0.106 | 0.059 | 0.060 |
| Orbit, four sources plus ring, moving lights | 0.105 | 0.717 | 0.060 |

These are historical isolated atmosphere timings, not current application frame times. They exclude geometry, CPU work and ring-source baking; rings use a synthetic surface-irradiance atlas. They demonstrate cheap cached reconstruction and expensive invalidation, not the cost or accuracy of a physically resolved atmospheric ring source.

The old LUT path accumulated secondary light alongside sunlight at every primary integration cell. Its secondary changes dirtied the primary angular tables too. Four reflected sources added four incoming-transmission/multiple-scattering lookups per cell. Separate secondary sky/aerial images added 10.28 MiB per retained atmosphere. Moving a faint secondary source could therefore trigger a complete high-resolution primary and secondary rebuild, even for a small orbital atmosphere.

The tested dense directional single-scattering comparisons support retaining the cumulative integrator: at most 0.063% error per tested channel against 1,024 midpoint steps. That result does not validate ring-source physics, angular interpolation, reduced resolution or arbitrary aerosol profiles.

## Proposed transport path

1. Keep stellar radiance, secondary radiance and RGB transmittance distinct. Reuse the optical-depth and unit-irradiance multiple-scattering tables; secondary light does not change extinction.
2. Give secondary radiance its own angular grid and cache identity. Start with 96 × 54 × 64 aerial radiance plus a 96 × 54 sky image, keeping all 63 intervals and the same subdivision rules. One secondary bake writes its complete column and cumulative finite-distance slices.
3. Rebuild only this secondary product when reflected-source flux/direction/size or ring illumination changes. Optical and primary angular products stay valid unless their own inputs change. Reuse the already prepared `planetshine_` records instead of running source selection again.
4. Composite `background * T + (solar * (1 - solar_deficit) + secondary) * exposure`. The current shadow pass estimates loss of stellar illumination; it must not multiply the added secondary radiance. Reflected-path shadows would need their own model.
5. Reconstruct secondary radiance at each opaque or transparent fragment's finite endpoint. Disabling aerial perspective suppresses finite-endpoint secondary inscattering while retaining sky illumination. A 2D sky texture alone cannot supply consistent nearby haze.

The proposed secondary sky/aerial images total 2.57 MiB per retained atmosphere, before allocation overhead. A matching 96 × 54 R32F range image adds about 20 KiB if required by near-floor/horizon reconstruction. This is one quarter of the former secondary image payload and one quarter of its angular columns, with unchanged per-column quadrature. These ratios are arithmetic, not measured GPU speedups: separate dispatches repeat some medium work and composition still adds filtered texture reads.

Only after that baseline passes quality comparisons should 32 depth slices be tried. Sky/aerial payload then falls to approximately 1.31 MiB, but using 31 rather than 63 integration intervals changes transport accuracy. Storage resolution and integration quality should be evaluated separately.

Parameterize the mapping/reconstruction helpers by the product's dimensions. The current `SKY_SIZE`/`AERIAL_SIZE` constants also control horizon-half texel centres, periodic azimuth, bicubic filtering and depth interpolation; swapping dimensions only at allocation or dispatch is insufficient. Preserve the separate ground/sky horizon rows. Give a reduced-resolution product its own consistent endpoint/range reconstruction.

Cache inputs include optical/shape identity, camera altitude and the actual local source field: reflected RGB, directions, angular sizes, local ring orientation, source/profile revisions and relevant illumination/occlusion changes. Preserve double-precision world-to-local preparation. View rotation, FOV and exposure remain outside transport keys. Use logical ring revisions rather than per-frame atlas image handles. Include all newly retained images in the cache budget.

Initially refresh immediately on physical changes. A later error-bounded update policy may reuse tiny source changes, but an unconditional fixed update frequency is inappropriate for eclipses, ring equinox crossings and time warp. Any significance cutoff must retain night-side illumination when direct sunlight is absent. Avoid adding a fresh per-frame CPU source-selection pass; the current audit already measures about 0.08–0.10 ms in planetshine preparation.

## Planetshine: reuse the source preparation

`prepare_planetshine` already selects four separate material-aware RGB sources per receiver, including phase, approximate emitter eclipses, apparent angular radius and inverse-square flux. `CelestialRenderer::prepare` computes them immediately before atmosphere preparation but does not currently pass them in.

Transform those records into the atmosphere's radial basis and evaluate incoming atmospheric transmission, Rayleigh/particle phase and, for distant directional sources, the existing multiple-scattering closure. The radiance-unit conversion needs explicit attention: stellar atmosphere irradiance is packed as `PI * color * luminosity / distance²`, whereas `PlanetshineGPU::irradiance` uses the surface renderer's relative scale. An identical-light GPU contract should establish normalization rather than copying values blindly.

Keep the existing receiver-centre directional approximation explicit. Apparent disk size should affect horizon visibility and aerosol phase averaging; a large nearby planet must not acquire an artificially narrow point-source forward lobe. Strong near-field parallax across an atmosphere requires source positions/per-sample geometry or a documented restriction. Several forward-peaked particle layers are important validation cases, not just a clear Rayleigh sky.

For directional lights, phase angle is constant along a straight view column. A specialized secondary integrator can prepare source-dependent phase values per column and share per-cell particle densities across sources. Verify shader statistics/timings before claiming savings; the compiler may already hoist some expressions.

## Ringshine: the surface atlas is insufficient

`ringshine_integral.glsl::ringshine_arc` accumulates ring radiance with `mu_p * mu_v / d²`: `mu_p` is the receiver surface-normal cosine and `mu_v / d²` converts ring area to apparent solid angle. `ringshine_band` clips visibility at a point on the reference oblate surface. The resulting `ringshine_sample` is a surface irradiance map, not an angular incident-radiance distribution at an atmospheric point.

Atmospheric scattering instead needs

`source(q, view) = scattering(q) * integral[L_ring(q, direction) * phase(view, direction) dOmega]`.

It has no receiver-surface cosine. Gas above the surface can see different ring arcs, and incoming transmission and aerosol scattering depend on each ray's direction. Multiplying the surface atlas by an isotropic phase or an empirical Mie gain can provide a visual approximation, but cannot recover the missing angular/height information. The removed experiment used that approximation; its performance and directional-reference results do not resolve this limitation.

A plausible next ring representation is a small, camera-independent incident-light field over ring-relative azimuth, latitude and a few heights. Reuse the existing radial planner, ring materials, stellar-disk sampling and host-shadow geometry to accumulate incident radiance/solid angle at gas points, with the proper gas-to-ring visibility and atmospheric attenuation. Do this source quadrature in the field bake rather than inside every sky-column cell.

For Rayleigh scattering, angular moments provide an especially useful target. With RGB moments `J0 = integral[L dOmega]` and symmetric `J2 = integral[L * direction * transpose(direction) dOmega]`, the source is

`rayleigh * 3/(16*PI) * (J0 + transpose(view) * J2 * view)`.

Because `trace(J2) = J0`, six independent RGB coefficients suffice. This exactly represents the angular Rayleigh phase integral for the sampled incoming field; spatial interpolation and ring quadrature remain approximations. General forward-peaked Mie/particle phases need more angular information, such as several finite-width directional lobes or a richer basis. Validate those against dense ring-area integration before selecting a production representation. The existing directional multiple-scattering closure is also not automatically a near-field extended-ring solution.

The light-field dimensions, height nodes, angular representation and refresh cost are unresolved design choices. A field adds a separate source bake and optical dependencies if it includes incoming atmospheric transmission. Its cache must respond to ring material/geometry, star elevation and host shadow changes. Scheduler bounds must include visible atmospheric receivers, including a visible limb with the opaque host offscreen; the current surface-ringshine scheduler only considers opaque/terrain bounds. Preserve lit/unlit face differences and finite-star continuity near equinox.

## Implementation sequence and acceptance

1. Primary sky/aerial integration is merged, with parity and cache tests. Measure movement cases before assigning its savings to the secondary-light budget.
2. Prototype independently cached planetshine at 96 × 54 × 64. Compare against a full-resolution secondary-only reference and a dense directional integrator. Measure one/four sources, ground/orbit, paused, moving lights and moving camera.
3. Develop the ring incident field with a dense ring-area reference. Test Rayleigh moments first; compare aerosol lobe approximations independently. Measure source-field baking as part of total cost.
4. Tune angular/depth resolution and update thresholds only against demonstrated error and measured whole-frame cost. In particular, small orbital atmospheres must not repeat the historical fixed-bake regression.

Required behavior: correct units and linearity; identical incident solar/secondary transport within storage precision; continuous grazing/large-source horizons; finite endpoints approaching identity at zero distance; no cross-horizon or azimuth seams; secondary light survives unrelated stellar shadow deficits; no new black-surface inscattering with aerial perspective disabled; opaque/transparent endpoint agreement; immediate toggles/edits/eclipses; optical-cache reuse on secondary-only motion; rotation/FOV/exposure reuse; translated coordinates; ring equinox and elevated-limb visibility. Update the existing `shine_contracts` assertions that currently require atmospheric secondary scattering to remain absent.

The first prototype should establish the independent planetshine transport cost. Ringshine's central unresolved problem is the source field, and resolving it deserves a separate accuracy/cost comparison rather than hiding it behind the surface atlas.
