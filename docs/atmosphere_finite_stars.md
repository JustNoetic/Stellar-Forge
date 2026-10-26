# Finite stellar illumination

25 October 2026. Direct sunlight now integrates the visible stellar disk before applying ground or atmospheric scattering response. The source is a uniform-radiance spherical disk. The existing straight-ray, sphericalized oblate atmosphere model is unchanged; this does not introduce refracted transport, terrain shadows or a separate host-shadow compositor.

## Why the old penumbra failed

The physical surface transmission atlas included a fixed 0.01-radian horizon fade. The surface BRDF already sampled a larger disk, but its entire result was multiplied by that centre-direction transmission. Once the stellar centre set, this extinguished light from the exposed upper disk.

The sky used the actual stellar angular radius in its horizon fade, but still multiplied the visible fraction by a single centre-ray optical transmission and phase. Centre rays below the horizon map to the grazing edge of the optical LUT. A wide visible disk was therefore assigned the much darker/redder transmission of that edge.

For the project's solar radius, the Sun at 0.01 AU has an angular radius of about 27.7 degrees. Direct illumination should extend well beyond the centre-ray terminator.

## Integration

`common/stellar_disk.glsl` supplies shared disk geometry. Eight Gauss–Legendre elevation slices cover the cap, clipped to the receiver horizon. Each slice has an analytically calculated visible azimuth interval. The half-angle expression avoids subtracting nearly equal cosines for small stars. The integration measure is solid angle divided by `pi * sin(angular_radius)^2`, preserving the engine's luminosity/distance² convention: an unobstructed Lambert receiver facing the stellar centre receives unit irradiance.

`stellar_scattering_source` uses a cached incoming-light field for the first large star. Other large stars retain eight incoming-elevation optical-depth samples, Beer–Lambert transmission and an analytical Rayleigh phase integral over each azimuth interval. Mie and custom particle phases use the stellar centre direction once per local particle component, multiplied by the disk-integrated transmitted light. This keeps finite-disk visibility and RGB attenuation while approximating the angular distribution of particle-scattered light. It applies to cached queries, extra stars and last-contact/unresolved-layer fallback queries alike.

For sources smaller than 0.01 radians, phase is evaluated at the centre direction. The horizon region averages disk transmission with four elevation nodes using the small circular-disk approximation and an analytical visible-area normalization; well above the horizon it takes the point-source shortcut. Separate small-source sky/aerial pipelines compile out the large-disk phase machinery. The native external-shadow integrators use the same source helper; cached weights inherit it from the aerial volume.

The unit multiple-scattering and diffuse-sky reference tables retain their 0.01-radian reference source. Large-source queries angularly average those unit responses, including directions below the ground horizon that can contribute indirect light. Direct-only mode continues to suppress both contributions.

The existing 63-interval sky/aerial integration and full-resolution composite are unchanged. There is no additional spatial raymarch or screen-space shadow pass. The source cache does not eliminate camera-dependent sky/aerial rebakes.

## Incoming-light cache

`atmosphere/stellar_source.comp` builds a unit-irradiance field indexed by height and local stellar elevation. Its 128 × 64 domain concentrates samples near the host's geometric shadow boundary and near the lower atmosphere. It depends on atmospheric optics, lighting mode and apparent stellar radius, rather than camera pose, stellar azimuth, brightness, color or exposure. Stellar intensity and RGB color multiply the sampled response in the existing sky integration. Native external-shadow source evaluation uses the same cache.

The field has five RGBA16F layers (320 KiB) for both molecular and particle media: four RGB angular moments and the disk-averaged unit multiple-scattering response. Contracting the second moments with the view direction evaluates the quadratic Rayleigh phase exactly for the baked angular distribution. Regular cached queries need no incoming-direction loop; the last-contact region retains direct evaluation. Direct-only mode skips the multiple-scattering calculation.

The previous particle field used seventeen layers (1.0625 MiB), including eight transmitted slice weights and four packed geometry layers for 32 phase directions. Those twelve layers and their per-query reads are removed. The particle phase is evaluated once toward the stellar centre and weighted by the same total incoming RGB light used by Rayleigh. The approximation can concentrate the haze glow around a huge star's centre and change brightness when its visible crescent is far from that centre. It preserves finite-source illumination, rather than restoring centre-ray transmission. Close to last contact, lighting blends back to direct angular evaluation over source-grid columns two through four to avoid magnifying interpolation error in a faint, narrow visible crescent; this fallback also uses the single particle phase.

RGB amplitudes use fourth-root encoding, with moments and particle weights normalized by the total incoming light. Two multiplications decode the amplitude. This preserves faint normalized responses before stellar brightness/exposure scaling without logarithmic decoding or larger texel formats. Disk-averaged multiple scattering and cached finite-ground irradiance use the same amplitude encoding.

Large-source sky/aerial pipelines specialize molecular-only and legacy-Mie media to compile out unused custom-layer transport and phase machinery. These variants are created only on their first required bake. Any custom layer, including a purely absorbing one, retains the general transport pipeline and its layer-aware subdivision. Incoming legacy-Mie density is reused from the existing medium evaluation.

Only the first star currently uses this field; additional large stars keep their uncached angular source evaluation. The previous 1.2-radian particle-cache restriction is removed because interpolated per-direction particle geometry is no longer stored. Small stars retain their existing specialized path and allocate no incoming-light image. Atmospheres without scattering skip the source bake. The source-bake pipeline is created on its first required update.

Optically significant Gaussian particle or ozone layers retain direct angular evaluation when their width is less than twice the source grid's local height spacing. The conservative significance threshold is a full vertical optical depth of 0.005 in any RGB channel. This eligibility check runs only when optical parameters change, with no extra GPU query or pass. It avoids interpolating incoming light through an unresolved opaque layer. A 100 m Gaussian layer with vertical optical depth near five produced substantial leakage in the raw cache stress check; the renderer now bypasses that cache. Negligible thin layers remain eligible. This guard is a conservative resolution bound, not an error certificate for arbitrary density profiles.

The existing uniform layout is retained: `aerosol_count.y` marks particle scattering, `.z` marks custom particle emission, and `sun_irradiance[0].w` contains the binned source radius or zero when the cache is ineligible. The actual angular radius remains in `sun_direction[0].w`.

Apparent radius is rounded to 1/512-octave bins, with a maximum relative angular-size error of approximately 0.068 percent. This avoids rebuilding the source and finite-ground layers for tiny radial changes or floating-point noise. Crossing a bin refreshes the source field without rebuilding optical tables. The source query uses that same rounded radius; sky geometry still tracks the actual source. Unit multiple scattering uses a separate full solar-zenith mapping so indirect light is preserved below the direct-light cutoff.

Definition edits retain per-body cache ownership. Optical invalidation compares body identity (`BodyRevisions::identity`) and the actual optical parameters; unrelated edits and changes to stellar color or luminosity do not rebuild optical or incoming-light tables. The existing camera-dependent radiance tables still refresh when the stellar RGB irradiance changes. Incoming-light allocations count toward retained table memory.

## Ground and caching

The surface atlas now stores three layers per atmosphere: hard-visibility point transmission, unit diffuse sky irradiance, and normalized finite-source Lambert irradiance for the first star. It expands from 64 to 96 RGBA16F layers at 128 × 32: 3 MiB total, 1 MiB more than before, shared across 32 surface slots. No per-atmosphere sky/aerial images are added.

Unattenuated Lambert disk irradiance has a closed-form spherical-cap/hemisphere intersection integral. The implementation uses a series for the cancellation-prone small-angle term. Hapke surfaces and atmospheric receivers couple their BRDF and transmission inside the angular quadrature.

In direct-only mode, smooth Lambert surfaces can use the cached first-star response instead. Cache eligibility requires oblateness below 0.01, an aligned reference normal, a matching apparent radius within one percent, and sufficient distance from the faint last-contact edge. Other large sources, reflected sources, displaced normals, indirect mode and the last-contact region use the coupled quadrature. The cache retains the atmosphere's existing sphericalized optical approximation and neglects the small variation in apparent stellar radius across an eligible receiver.

Small-source Lambert ground queries use four Gauss–Legendre elevation samples,
clipped to the shading normal's horizon and weighted by the analytical visible
azimuth width. Transmission is sampled at each slice's central azimuth. This
retains a continuous finite-source terminator while approximating the much
smaller azimuthal transmission variation. It applies below 0.01 radians for
oblateness below 0.01, away from the optical self-shadow boundary and near-zenith
incidence. The existing point-source shortcut remains outside the horizon region.
Hapke, large-source, strongly oblate and optical-shadow contact queries retain
the coupled direct quadrature.

Ground skylight is already a baked hemispheric irradiance response. Sources up
to 0.005 radians, including the normal Sun at Earth, now use one centre-direction
lookup rather than averaging it over another 32 directions per fragment.
Sources between 0.005 and 0.01 radians use four sky-response samples; larger
sources retain the previous angular average. Direct-only mode still suppresses
skylight. All paths hoist the receiver's atmosphere coordinates and normal factor
out of direction loops, sample only the required atlas layer, and skip the unused
point BRDF when its blend weight is zero. No atlas layers, framebuffer passes,
spatial integration or cache invalidations are added.

Surface cache keys belong to atlas slots and contain the lighting mode and binned first-star angular radius, with sub-0.01-radian sizes treated as the same unused finite-source layer. Radius-only updates write just the inexpensive finite-source layer; they do not rebuild point transmission, hemispheric diffuse sky or optical LUTs. Direction, brightness, exposure and camera motion do not invalidate this unit surface layer.

## Verification

The 26 October ground-cost regression checks use
`--finite-star-ground-only`: 156,672 receiver/source/material cases against the
previous ground shader, spanning four angular radii, four heights, seventeen
stellar elevations, six normal tilts, eight normal azimuths, Lambert/Hapke,
spherical/Earth-like/strong oblateness, and direct/indirect modes. Direct response
relative RMS is at most 0.131 percent and the maximum resolved local difference
is 1.52 percent. Skylight RMS is at most 0.100 percent, with maximum resolved
local difference 1.29 percent. The local gate excludes reference RGB norms below
1e-5. These are differences from the previous finite quadrature, rather than an
absolute physical reference.

An RTX 4060 Ti test with imported Himalayan elevation and normal maps, an eye
15 km above the ellipsoid looking down, normal Sun, skylight enabled, and a
2560 × 1440 offscreen target measures the following terrain-draw medians over
48 frames after warmup:

| Solar elevation | Previous ground shader | Four-sample ground shader |
| --- | ---: | ---: |
| -3 degrees | 1.155 ms | 0.645 ms |
| 0 degrees | 3.497 ms | 0.737 ms |
| +3 degrees | 1.751 ms | 0.665 ms |

The zero-elevation case is approximately 79 percent cheaper. These are isolated
surface timings, not a reproduction of the user's exact camera or an application
FPS promise. The fixture has one star and Earth, with refraction and external
atmospheric shadows disabled; its stationary sky/source caches report zero
updates. Preparation remains about 0.055 ms. Reports are retained in
`tests/finite-star/mountains-regression/indirect-before.json` and
`indirect-four-final.json`; the expanded query log is `ground-contracts-final.log`.

`StellarRenderingTests --finite-star-only tests/finite-star` compares a large star against 256 independently shaded small stellar directions in 16-source batches. Flux normalization accounts for the distinction between solid angle and projected disk area; the fixture uses exposure scaling to avoid the surface renderer's minimum per-star luminosity. It samples stellar elevations -35, -25, -10, 0, 10 and 40 degrees for airless ground, attenuated ground, molecular air, Mie air and restored indirect ground lighting. Mie-air differences are now reported as an intentional phase approximation instead of being held to the full-disk accuracy gate; the other profiles retain that gate. Six additional cases check a Sun-sized angular disk near the horizon of a 30 km observer. It also checks stationary cache reuse and sphere/terrain agreement. `--finite-star-cache-only` checks camera/exposure reuse, stellar color/luminosity edits, direction-only motion, sub-bin radius changes, radius-only layer refresh in both lighting modes, and small-source cache reuse.

`--finite-star-source-only` compares cached and direct source evaluation of the same single-phase model over heights, stellar elevations and view directions, including exact zenith/nadir source centres, strongly forward-scattering legacy Mie and tabulated particle phases. A third output preserves the former uncached 32-direction phase evaluation as a comparison, with phase-approximation differences reported separately from cache error. Four additional cases compare the cached and directly averaged multiple-scattering response. Dense exponential and thin Gaussian stress cases exercise height interpolation; the unresolved opaque-layer case deliberately reports the raw cache error as a negative control. The renderer cache contract separately checks that this layer selects direct evaluation, camera movement preserves the fallback, and thinning/removing it restores caching. The source test bypasses the screen compositor and spatial view integration so incoming-light interpolation errors can be measured directly. A clamped lower source coordinate of -1 is not treated as an opaque shadow cutoff: a sufficiently broad star can remain partly visible there.

Opaque composition consumes the already-rendered shadow deficit at its exact sky/opaque depth. Its shader excludes the native shadow-integration fallback; that fallback remains in transparent composition for intermediate fragment depths. An empty external-caster list disables shadow work in the per-body header. This keeps the larger source helper out of the ordinary opaque compositor's register footprint.

The independent reference has discrete disk quadrature and uses the same optical and view-column reconstruction, so it validates source integration rather than absolute atmospheric transport accuracy. The faint last-contact cases are sensitive to discretization, receiver parallax and half-float precision. The source/receiver geometry is synthetic; these are not exact reproductions of the user's screenshot.

`--finite-star-benchmark <report.json>` measures cached and translated ground/orbit views at 1 AU and 0.01 AU, at 1920 × 1080. It records preparation, opaque drawing and atmospheric composition separately. `STELLAR_FINITE_STAR_MOLECULAR=1` removes particle scattering/extinction; `STELLAR_FINITE_STAR_RADIAL=1` changes the source distance by 0.3 percent on each moving frame to measure refresh cost. Incoming-light timing is recorded only when that cache updates, because the timer API otherwise retains its previous measurement. These are serialized offscreen fixture timings, not application FPS. Artifacts and source/reference profiles are retained under ignored `tests/finite-star/`.

Release validation passed the disk and source-cache checks, atmosphere cache ordering, direct/indirect lighting and endpoint regressions, the full body/ring shadow checks, planetshine/ringshine checks, and the application smoke test with Vulkan validation. Merged-column parity still matches all 4,644,864 half-float components across 28 fixtures. The complete `--atmospheres-only` command was not rerun as one invocation; the listed groups were run separately.

RTX 4060 Ti median moving-view preparation milliseconds, 8 warmup and 32 measured frames, physical Earth with legacy Mie scattering 0.002/km, source centre 10 degrees below the observer's ground horizon:

| Source distance | View | Previous centre-ray model | Finite disk |
| --- | --- | ---: | ---: |
| 1 AU | Ground | 0.113 | 0.120 |
| 1 AU | Orbit | 0.137 | 0.222 |
| 0.01 AU | Ground | 0.135 | 1.091 |
| 0.01 AU | Orbit | 0.135 | 1.070 |

Cached preparation medians are zero. At 0.01 AU, the cached ground-response path measured about 0.20 ms for the ground-view opaque draw and 0.13 ms in orbit. These measurements precede a small smoothing change at the small-source/point approximation boundary; they are indicative rather than exact timings of that last shader revision. Driver clock ramping and scheduling affected early composition/draw samples, so those samples are not used to claim an overall frame speedup. The larger-source bake is materially more expensive than the old incorrect model; avoiding movement-dependent sky/aerial rebakes remains separate future work.

### Previous 32-direction incoming-light cache timings

The subsequent cache implementation retains finite-disk phase sampling and the original sky/aerial resolution. The following 1920 × 1080 runs use the same RTX 4060 Ti fixture, eight warmup and thirty-two measured frames. The uncached control evaluates the disk directly with the current finite-star model. Molecular cases remove legacy Mie scattering and extinction; the Mie cases retain 0.002/km legacy scattering/extinction.

| Medium | View | Uncached disk preparation (ms) | Cached preparation (ms) | Cached composition (ms) |
| --- | --- | ---: | ---: | ---: |
| Mie | Inside, 1 km | 1.040 | 0.349 | 0.284 |
| Mie | Orbit | 1.025 | 0.418 | 0.140 |
| Molecular | Inside, 1 km | 0.573 | 0.150 | 0.285 |
| Molecular | Orbit | 0.571 | 0.205 | 0.140 |

The initially delivered Mie bake measured 1.091 ms inside and 1.070 ms in orbit. Preparation plus composition in the inside Mie fixture changes from approximately 1.310 to 0.633 ms. These totals exclude opaque drawing, readback and presentation.

The source field performs zero updates during camera translation, direction changes, exposure changes, or source color/luminosity changes. The radiance bake still refreshes when its geometry or RGB irradiance changes. Repeated 0.3-percent radial source motion refreshed the source field in approximately 0.007 ms, with full preparation medians of 0.352 ms inside and 0.455 ms in orbit. Surface and optical cache counters independently verify that those updates do not regenerate optical tables. Ordinary small stars allocate no source field, and absorption-only media skip its bake.

Composition measured higher than the uncached control in these serialized runs: approximately 0.284 versus 0.220 ms inside, and 0.140 versus 0.112 ms in orbit. No extra framebuffer pass or composition samples were introduced. A diagnostic retaining the native opaque fallback also measured approximately 0.140 ms in the orbital moving case, so restoring that fallback did not remove the difference. Ground diagnostic samples varied together across preparation, drawing and composition. The measurements therefore do not support claiming unchanged composition time or a hardware-independent overall frame improvement; both preparation and composition are retained in the report.

Final reports are `tests/finite-star/verified-timing/`, `verified-timing-molecular/`, and `verified-timing-radial/`; the uncached controls are `native-source/` and `native-molecular/`. Final source-grid checks passed all 28 direct/multiple-scattering cases, with relative RMS below one percent and resolved local errors below five percent where the reference RGB norm exceeds 1e-5. Additional height stress cases and the unresolved-layer fallback are recorded separately. The reference across 36 profiles against 256 stellar directions, cache checks, atmospheric lighting, merged-column parity, body/ring shadows, transparency, planetshine/ringshine and application smoke checks also passed with Vulkan validation. The source-grid check explicitly includes the wide-star antipodal source case and direct fallback regimes.

### Rejected per-column phase hoist, 26 October

An experiment computed the legacy Mie phase before the sky/aerial integration
loop. A sixteen-value per-ray array reused phases for every star, but increased
the normal-Sun ground-view preparation median from 0.113 to 0.143 ms. A smaller
variant retained only the first star's phase as one scalar and left additional
stars on the existing path. It preserved all 10,368,000 half-float values across
five matched HDR images. Merged sky/aerial parity also passed all 4,644,864
components across 28 cases, and the source-grid/cache checks passed.

The scalar experiment's large-star preparation medians were 0.152 and 0.156 ms
inside, against two 0.159 ms controls; orbital results were 0.215 and 0.218 ms,
against 0.222 and 0.216 ms controls. Normal-Sun results and unrelated draw/compose
timings also varied between captures. These serialized fixtures did not show a
dependable gain across views, so the hoist and its helper-interface changes were
reverted. Mie remains part of the sky/aerial bake with the previous single-centre
phase evaluation. No additional phase state, texture or pass is retained.

Reports, the original shader snapshot and matched images are kept under
`tests/finite-star/phase-hoist/`: `before.json`, `array.json`, `scalar.json`,
`final.json`, `before-repeat.json` and `final-repeat.json`. The repeated controls
include scheduling outliers, so these measurements do not justify a frame-rate
claim or a precise estimate of the phase arithmetic's isolated cost.

### Single particle-phase comparison

The current path replaces 32 particle-phase directions per spatial sample with one centre-direction phase, while retaining the disk-integrated transmission and Rayleigh moments. No runtime quality toggle or extra framebuffer pass was added. The previous shaders and standalone comparison executable are retained in ignored `tests/finite-star/phase-comparison/full-disk/`; the current results are in `single-phase/`.

Matched RTX 4060 Ti, 1920 × 1080, physical Earth with legacy Mie scattering/extinction 0.002/km and asymmetry 0.8, source at 0.01 AU, moving camera, eight warmup and thirty-two measured frames:

| View | 32-direction preparation (ms) | Single-phase preparation (ms) | 32-direction composition (ms) | Single-phase composition (ms) |
| --- | ---: | ---: | ---: | ---: |
| Inside, 1 km | 0.340 | 0.160 | 0.276 | 0.221 |
| Orbit | 0.431 | 0.223 | 0.144 | 0.117 |

Preparation falls approximately 53 percent inside and 48 percent in orbit. Preparation plus composition falls from 0.616 to 0.380 ms inside, excluding opaque drawing and presentation. Ordinary small-source preparation remains approximately 0.11 ms inside and 0.23 ms in orbit. Composition also measured lower after the shader change; the benchmark records this separately and does not attribute the difference solely to phase arithmetic. The source field is now always 320 KiB, saving 768 KiB in a particle medium.

`--finite-star-phase-preview <directory>` exports five identical scenes with linear RGBA16F and tone-mapped PPM outputs: an orbital penumbra, twilight toward the star, a view centred on a star 15 degrees above the horizon, a view away from the star, and a star-centred 45-degree view. The sky cases isolate atmospheric scattering without drawing the stellar disk or ground surface. Exposure is identical across the two versions; their source brightness follows the same inverse-square normalization. `compare.py` creates contact sheets and records linear RGB differences independently of tone mapping.

The source-grid comparison distinguishes cache interpolation error from the intentional phase approximation. At angular radius 0.482 radians, same-model cache RMS is about 0.16 percent for legacy Mie g=0.8; centre-phase versus the old full angular evaluation is about 43 percent over the synthetic source-query grid. At g=0.95 that phase difference grows substantially (relative RMS about 7), so this approximation does not preserve strongly concentrated forward scattering from an extended disk. Molecular source evaluation remains equivalent apart from floating-point rounding. These grid errors are not image errors or an absolute physical-reference certificate.

Matched 960 × 540 image comparisons use linear HDR RGB before tone mapping. Relative RMS is the image-wide L2 difference divided by the reference image's L2 norm; the centre measurement averages a 15 × 15 pixel region. The ordinary g=0.8 fixture gives:

| Scene | Relative RGB RMS | Single/full centre luminance |
| --- | ---: | ---: |
| Orbital penumbra | 0.056% | 1.000 |
| Twilight toward star | 12.66% | 1.266 |
| Centred on star at +15 degrees | 7.17% | 1.311 |
| Looking away from star | 0.021% | 1.000 |
| Centred on star at +45 degrees | 1.46% | 1.308 |

These images isolate a visible, intentionally changed forward glow while retaining the orbital penumbra and sky away from the star. The ordinary rendered star disk would cover much of the affected centre region; it is omitted from the sky comparison so the atmospheric approximation remains visible. Tone-mapped contact sheets are `tests/finite-star/phase-comparison/comparison.png` and `orbit-and-haze.png`; exact metrics are in `image-metrics.json` and timings in `timing-comparison.json`. The displayed images use equal exposure with no per-version normalization.

The single-phase build passed the source-grid/cache checks, the finite-star ground and atmospheric profiles with the documented Mie comparison exception, direct/indirect lighting, merged-column parity across 4,644,864 half-float components, body/ring atmospheric shadows, transparent endpoints and both blend paths. The application smoke test completed three frames with Vulkan validation and left the Solar System definition unchanged. No validation errors were reported in these runs.
