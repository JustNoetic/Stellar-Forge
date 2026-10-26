# Atmospheric planetshine prototype

Historical single-source reference. The default now uses the
[shared approximate four-source bake](atmosphere_secondary_lights.md); set the internal
`atmosphere_planetshine_reference` diagnostic to reproduce this original path.

26 October 2026. One reflected source, independently cached from sunlight. Existing
Planetshine controls the prototype; Direct sunlight only suppresses it. No atmospheric
ringshine is added.

## Transport and cache

`AtmosphereRenderer::prepare` consumes the already prepared `PlanetshineGPU` records.
The greatest luminance among the four source slots selects one disk. Its direction is
transformed through the existing oblate/radial basis; its apparent radius is retained.
`planetshine.comp` integrates a white, unit-irradiance source using the same medium,
finite-disk incoming transmission, single-centre particle phase, multiple-scattering
closure, horizon mapping and detached-layer subdivision as stellar sky/aerial transport.
All 63 intervals and two midpoint cells remain; angular resolution changes to 96 × 54.

The bake produces cumulative aerial RGB, final sky RGB and column range. It adds no
transmittance volume, per-source uniform buffer, source selection pass or screen-space
march. Resolved finite sources optionally allocate the existing 320 KiB incoming-light moment
representation; its source bake reuses `stellar_source.comp` with a pushed radius.
The radiance/range payload is 2,716,416 bytes (2.59 MiB) per allocated atmosphere; a
source field raises this to 2.90 MiB.
The optical tables are shared. Three extra sampled descriptors and a 16-byte irradiance
push constant are present only in the secondary composition variants.

The cache key contains optical validity, camera altitude, local source direction and
apparent radius. RGB/brightness is applied as `PI * reflected irradiance` after sampling,
so amplitude edits need no rebake. A direction/size change refreshes only secondary
radiance. A missing aerial product refreshes when aerial perspective is re-enabled.
Disabling/re-enabling planetshine can reuse a physically unchanged response; edits made
while disabled must still invalidate it. View rotation, FOV and exposure stay outside
transport keys. The incoming field is keyed only by optics and apparent radius binned
at 1/512 octave, and uses the same unresolved-opaque-layer fallback as sunlight. Small disks cache the existing
four-node transmission and centre-phase response; larger disks retain Rayleigh moments. Camera,
source direction and amplitude changes retain it. Allocated image payload enters
retained-cache accounting.

Sunlight's external shadow deficit is applied before adding planetshine. Secondary sky
uses its own horizon texel mapping; secondary finite endpoints use matching range and
bicubic reconstruction. Disabling aerial perspective preserves extinction while adding
no finite-endpoint secondary inscattering. Both opaque composition paths and captured
transparent fragments receive the same secondary transport.

## Scope

This is one uniform directional apparent disk using the existing reflected photometry.
It does not resolve source surface textures, a crescent's angular structure, or parallax
across the receiver atmosphere. Particle phase uses the current stellar approximation.
Other reflected sources, reflected-path occluders, source-switch smoothing and atmospheric
ringshine remain outside this prototype. The source brightness is approximate photometry,
not a recursive radiative-transfer solution.

## Validation and measurements

Local fixtures live under ignored `tests/planetshine-prototype/`:

```powershell
build/bin/tests/StellarRenderingTests.exe --planetshine-prototype tests/planetshine-prototype/contracts
build/bin/tests/StellarRenderingTests.exe --planetshine-prototype-benchmark tests/planetshine-prototype/timing.json
```

The contract fixture checks stationary reuse, RGB/brightness linearity, source direction
and size, altitude, FOV/rotation/exposure, toggles, finite endpoints and both opaque blend
paths. It compares reduced-resolution reflected scattering against the full-resolution
stellar integrator with identical incident illumination. This isolates interpolation and
normalization; the reference shares source/transport approximations.

RTX 4060 Ti, 2560 × 1440, Release build. Serialized GPU timestamps, three
rotated-order repetitions, 12 warmup + 32 measured frames per on/off case, plus 256
movement warmups at each source-size cohort. One physical Earth atmosphere uses
legacy Mie asymmetry 0.8. Ground views start at 1 km; finite haze ends at a 30 km
view-space plane; orbital camera radius is three Earth radii. Geometry, CPU preparation,
readback, presentation, refraction, external shadows and ring-field baking are excluded.

Median milliseconds, moving-source bake and cached composition overhead:

| View | Small disk bake, 0.00465 rad | Larger disk bake, 0.08 rad | Cached composition increase, small disk |
| --- | ---: | ---: | ---: |
| ground sky | 0.205 | 0.173 | 0.001 |
| orbit | 0.237 | 0.285 | 0.028 |
| finite 30km | 0.185 | 0.200 | 0.120 |

Stationary and amplitude-only cases assert zero secondary/source-field updates. Direction
and altitude changes retain the incoming-light field. Forced 0.3-percent radius motion
refreshes it in about 0.010 ms for the small disk and 0.016–0.017 ms for the larger disk,
in addition to the secondary radiance bake. Primary optical and solar radiance counters
remain zero for secondary-only motion.

These are isolated atmosphere timings, not application FPS. The synthetic reference
illumination occupies the second stellar slot to retain native incoming-light evaluation;
large-source moving-camera primary timings therefore do not represent the optimized main
star. Judge secondary updates from their own timer. Clock/scheduling variation affects
other passes too: the larger-source moving-camera finite-haze case has an inflated
composition median of 0.832 ms against 0.311 ms off, while its cached pair is 0.425/0.326 ms.
The cached differences above are not a dynamic-cost ceiling. Raw distributions remain in
`tests/planetshine-prototype/timing-retained-1440.json`.

The first generic large-disk shader cost roughly 1.7–2.6 ms per update. Shader
specializations alone did not remove that bottleneck. Reusing the normalized incoming
field removed repeated finite-disk work. Applying that field to small sources, with their
existing four-node response, also lowered the small orbital update from approximately
0.47 to 0.24 ms in these fixtures. It adds no screen-space integration.

Across 32 reference images (two source radii; ground, orbital day/twilight and finite
endpoints; molecular air, ordinary Mie, g=0.95 Mie and custom mineral dust), maximum RGB
L1 error is 0.415 percent for molecular/ordinary Mie. Strongly forward-scattering stress
cases reach 2.63 percent L1 / 15.64 percent relative RMS for g=0.95, and 3.71 percent L1 /
9.88 percent RMS for mineral dust. Local peak errors reach 34.7 percent. This reduced grid
is a prototype with visible narrow-lobe softening, not a full-resolution replacement.

The fixture also verifies independent RGB/brightness scaling, stationary and toggle reuse,
source direction/size and altitude invalidation, optical edits while inactive, large-world
translation, solar-shadow independence, short-path identity, opaque blend agreement, and
matching opaque/transparent endpoints with aerial perspective enabled/disabled. A Moon
with 1,737.4 km radius at 384,400 km lights a night-side Earth sky through the actual
prepared photometry path; exposure 100,000 makes that faint result visible in the preview.
Captures and `accuracy.json` are under `tests/planetshine-prototype/contracts-all-source/`.


Release validation with Vulkan validation enabled passed `--atmosphere-lighting-only`,
`--atmosphere-cache-only`, `--particles-only`, `--finite-star-cache-only`,
`--finite-star-source-only`, `--shine-only` and `--ringshine-only`. The application
`--self-test` completed 13 frames cleanly. The combined `--atmospheres-only` run also
passed direct-light, merged-column parity (4,644,864 half components) and endpoint checks,
but stopped at the existing software-VRS shadow quality gate for the 1920 × 1080,
large-star ring scene: RMSE 0.000877939, maximum 0.287482 against a 0.05 maximum gate.
An isolated control restores the pre-prototype solar atmosphere/composition/source shaders
and ignores secondary output; it reproduces the same values exactly. That failure is
independent of this prototype and remains unresolved. Logs are
`regression--atmospheres-only.log` and `shadows-solar-baseline.log`; the isolated control
is under `solar-baseline/`. Two old diagnostic fixtures now explicitly enable direct-only
mode instead of assuming its former default-on setting.
