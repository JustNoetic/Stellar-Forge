# Shared approximate atmospheric planetshine

26 October 2026. The default atmospheric planetshine path combines up to four existing
prepared reflected sources in one secondary bake. Planetshine enables it; Direct sunlight
only suppresses it. A [diffuse ring-field prototype](atmosphere_ringshine.md) now shares
one slot with up to three reflected bodies; approved GPU validation and timing are
documented there. Aggregation beyond the four prepared sources remains deferred. Direct
stellar transport and reflected surface lighting are unchanged.

## Representation

`PlanetshineGPU` already supplies four receiver-centre apparent disks with reflected RGB
irradiance summed over the illuminating stars. `AtmosphereRenderer::prepare` transforms
their directions into the atmosphere's radial basis and normalizes the source RGB by
the summed RGB irradiance. The 128-byte push block holds four directions/radii and four
relative RGB weights. No extra source-selection dispatch or per-source buffer is needed.

`secondary_source.glsl` evaluates Rayleigh and particle phase once per angular column
and source. Each cell shares medium density, extinction, view transmission, segment
integration and output writes across all sources. Each source uses one incoming
transmission lookup and one point-direction multiple-scattering lookup. A smooth
finite-size horizon visibility and an upward-shifted transmission sample approximate
partially visible disks. Phase uses the source centre; disk quadrature and separate
incoming-light source fields are omitted. Custom aerosol phase responses are prepared
once per column, with the direct-scattering layer-density sum shared across contributing lights.

The output remains one 96 x 54 sky texture, one 96 x 54 x 64 cumulative radiance volume
and one R32F column-range image: 2,716,416 bytes (2.59 MiB) per allocated atmosphere,
independent of source count. The default path uses one midpoint per stored interval,
with detached-layer subdivision retained. The previous single-source reference uses
two midpoints and finite-disk integration with its optional 320 KiB incoming-light field.

Composition still reads the same three secondary textures and multiplies by the summed
RGB irradiance times PI. Source count adds no composition reads or additional volumes.
Opaque compute/dual-source blending and transparent endpoints share this reconstruction.
Primary solar shadow subtraction remains independent of the added reflected radiance.

## Cache and approximation limits

Optics, camera altitude, source count/directions/radii or relative RGB weights invalidate
the combined response. Common per-channel brightness/colour multipliers reuse it;
independent source edits generally rebuild it. Relative-weight comparisons have a 2e-6
absolute tolerance against the last baked weights, so subthreshold edits accumulate.
Zero-light receivers use the original composition shader. Rotation, FOV and exposure
stay outside the transport key; re-enabling aerial perspective rebuilds a missing column.

The source list is the existing brightest-four preparation, with no new weak-source
merging or recursive bounces. The source disk is uniform, without crescent structure,
textures, per-point parallax or reflected-path eclipses. Very large apparent disks can
show appreciable local colour/brightness differences from finite-disk integration,
especially around source rise/set and narrow particle halos. Multiple scattering uses
the centre direction. These are deliberate quality limits of the cheaper prototype.

`RenderSettings::atmosphere_planetshine_reference` retains the previous brightest-source
path for local comparison. It defaults off and is an internal diagnostic, without a UI
or persistent preference. Switching either way invalidates the appropriate response.
Shader variants specialize active counts one through four and the existing medium types;
they compile lazily and then reuse the shader/pipeline caches.

## Local validation and timings

Ignored fixtures:

```powershell
build/bin/tests/StellarRenderingTests.exe --secondary-lights-prototype tests/secondary-lights-prototype/contracts-final
build/bin/tests/StellarRenderingTests.exe --secondary-lights-benchmark tests/secondary-lights-prototype/timing-final.json
```

The fixtures check four-source additivity against independently rendered approximate
sources, normalized common-RGB scaling, relative-weight/source-geometry/altitude cache
invalidation, rotation/FOV/exposure reuse, toggles, finite endpoints and blend agreement.
They compare four-source results with the sum of four original single-disk responses
across molecular air, ordinary Mie, g=0.95 Mie and custom mineral dust, three source sizes,
and sky/orbital/finite views. Reference approximations are shared; this is a transport
comparison, not an independent physical solution. An actual two-star/four-moon fixture
also exercises prepared photometry and matching opaque/transparent endpoints.

With one midpoint, maximum image-wide relative RGB L1 error across those media/views is
0.175 percent for 0.00465-radian disks, 3.77 percent for 0.08-radian disks, and 19.8 percent
for the extreme 0.4-radian disks. Ordinary Earth/Mie cases at 0.4 radians reach about
12.8 percent. Local errors can exceed these image-wide averages. The two-midpoint
experiment had effectively the same errors, so its extra cell evaluations were removed.

The initial two-midpoint experiment measured about 0.35-0.38 ms for four moving sources
versus 0.18-0.32 ms for the previous one-source path on an RTX 4060 Ti at 2560 x 1440.
These are serialized isolated atmosphere GPU timestamps, excluding geometry, readback,
presentation, ring lighting and CPU source preparation. Source colours/directions differ
across the count cohorts; comparisons are representative scaling, not equal-work proofs.
Full distributions remain in `tests/secondary-lights-prototype/timing-64.json`.

The retained one-midpoint version measures 0.183-0.199 ms for four moving sources,
0.131-0.142 ms for two, and 0.112-0.123 ms for one. For normal-sized sources:

| View | Original one source | Approximate one | Approximate two | Approximate four |
| --- | ---: | ---: | ---: | ---: |
| ground sky | 0.190 | 0.118 | 0.138 | 0.193 |
| orbit | 0.226 | 0.121 | 0.139 | 0.183 |
| finite 30 km | 0.183 | 0.112 | 0.131 | 0.184 |

All values are median milliseconds. The two tested source radii are 0.00465 and 0.08
radians, with 24 warmup and 48 measured frames per case in three rotated orders.
Stationary and common-amplitude cases assert zero secondary bakes. One/four-source
cached composition medians are effectively unchanged: roughly 0.28 ms for ground sky,
0.14 ms orbital, and 0.46 ms finite haze, including the base solar atmosphere composite.
The extra secondary composition cost over disabled is about 0.00-0.01 / 0.03 / 0.11-0.13
ms respectively, and is not included in the bake table. Distributions have substantial
scheduling tails (some composition samples exceed 1 ms); these medians are not frame-time
ceilings or application FPS predictions. The original large-source orbital reference is
also variable between runs. Raw results are `timing-63.json`, copied with the sampling
metadata to `tests/secondary-lights-prototype/timing-final.json`.

The final four-source contracts pass, including a deliberately amplified secondary
signal for opaque/transparent endpoint comparisons. Release builds of StellarForge and
StellarRenderingTests pass. Vulkan-validation runs pass `--atmosphere-lighting-only`,
`--atmosphere-cache-only`, `--particles-only`, `--finite-star-cache-only`,
`--finite-star-source-only`, `--shine-only` and `--ringshine-only`; the application
`--self-test` completes 13 frames cleanly. Logs are under ignored `tests/secondary-lights-*`.
The previously reproduced full-suite VRS shadow quality failure remains documented in
the [single-source report](atmosphere_planetshine_prototype.md); this change does not
resolve that separate issue.
