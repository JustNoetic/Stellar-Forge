# Diffuse atmospheric ringshine prototype

26 October 2026. GPU rendering checks and the 1440p benchmark were run after the user's
explicit approval. The diffuse field passes its transport/toggle checks and the targeted
regressions. Its broad illumination is a deliberate approximation, not a full annulus
radiative-transfer solution. Future benchmark sessions should respect the user's request
to ask before starting them.

## Representation

The existing shared secondary bake receives one diffuse ring field, with room for the
three strongest prepared reflected bodies. The field belongs to the receiving body's
own rings. Rings lighting other planets or moons are outside this prototype. Ringshine
and Show rings enable it independently of Planetshine; Direct sunlight only suppresses
it. The original single-source diagnostic excludes atmospheric ringshine.

Ring profile rebuilds reduce occupied material into two radial, area-weighted averages.
Layer boundaries split the reduction so narrow layers survive. Overlapping optical
depths are summed before averaging coverage; premultiplied coverage weights material
colour and phase properties. Fine radial patterns are deliberately blurred.

For each visible atmosphere, a CPU cache integrates those two representatives into
mean RGB radiance on the two ring faces. It reuses `ring_radiance_rgb`, the existing
finite-star elevation sampler and an approximate azimuth-averaged host shadow. Up to
16 illuminating stars combine into the same two colours. Four angular nodes average
the ring phase response; this is a broad approximation, without a resolved angular
image of the ring. Star elevation/extent and relative RGB weights key the cache;
common stellar RGB multipliers apply afterward. Camera motion does not rebuild it.

One existing 32-byte light slot carries ring normal times outer-radius ratio, negative
inner-radius ratio, normalized mean radiance and packed per-channel face contrast.
The packed contrast uses three signed 8-bit coefficients with an exact zero. No extra
GPU buffers, incoming-light fields, descriptors or composition reads are added.

At an atmospheric cell the ring field estimates visible annulus area above the host
horizon. Its contact threshold is

`A = R_floor^2 - sqrt((r^2-R_floor^2)*(R_ring^2-R_floor^2))`.

The visible arc has `cos(limit) = A/(R_ring*r_planar)`. Its mean azimuth cosine supplies
a representative distance and incoming elevation. The inner area is first clipped to
the nearest radius that can appear above the host horizon:
`R_first = R_floor*r^2/(R_floor*r_planar + abs(z)*sqrt(r^2-R_floor^2))`.
This retains the visible outer ring near the polar contact instead of hiding the entire
annulus when its original mean radius is occluded. Projected annulus area divided by
distance cubed estimates solid angle, capped at 2*pi. The field varies with altitude,
latitude, oblateness, ring bounds and ring face; a surface pole does not receive a
uniform ambient fill. Ring-plane points see zero projected area for a zero-thickness
ring. A single incoming-transmission lookup attenuates the broad light.

Direct diffuse scattering uses normalized isotropic phase `1/(4*pi)` for molecular
and particle scattering. It creates a broad glow without separate point-source Mie
halos. The existing multiple-scattering table supplies an approximate additional term.
Medium integration, secondary textures, finite endpoints and composition are shared
with the planetshine prototype. Ordinary planetshine shader variants exclude this
ring-field branch.

## Limits

The estimate discards azimuthal ring illumination, sharp stellar shadows across the
ring, narrow forward lobes, individual radial-band illumination and reflected-path
occluders. It uses representative incoming elevation and distance, not a full annulus
integral. Its angular phase is diffuse even for anisotropic aerosols. Rings embedded
inside an extended atmosphere do not receive exact finite-path incoming transmission.
It is intended for convincing broad illumination, not accurate ring radiative transfer.
An independent dense solid-angle check for a 1.4-2.4-radius annulus viewed at radius
1.001 finds approximately 49/27/6 percent underestimation at latitudes 10/30/45 degrees
and about 8-11 percent overestimation near 60-66 degrees. The check isolates geometry;
material reduction and representative incoming extinction introduce additional errors.
The raw comparison is `tests/ring-atmosphere/solid-angle.json`.

The shared cache includes field geometry and relative RGB/face weights. Common RGB
scaling, view rotation, FOV and exposure reuse the response. Altitude, optical edits,
ring geometry/face-response changes or changes in relative secondary lighting rebuild
it. Disabling ringshine or hiding rings removes only their shared-light contribution.

Ignored test entry points:

```powershell
# CPU material contracts and glslang SPIR-V compilation; no GPU context is created.
build/bin/tests/StellarRenderingTests.exe --ring-atmosphere-static
# Requires explicit user approval before running GPU rendering checks.
build/bin/tests/StellarRenderingTests.exe --ring-atmosphere-prototype tests/ring-atmosphere/contracts
# Requires explicit user approval; isolated bake/composition timing at 1440p.
build/bin/tests/StellarRenderingTests.exe --ring-atmosphere-benchmark tests/ring-atmosphere/timing.json
```

Release builds of the application and test executable pass. CPU-only material contracts
and SPIR-V compilation of all twelve ring-enabled source-count/medium variants pass,
without creating a Vulkan/GPU context. The GPU fixture passes night-side ring lighting,
independent Ringshine/Planetshine/Show-rings/Direct-only behavior, primary/secondary cache
separation, common stellar intensity reuse, face asymmetry, polar occlusion and matching
opaque/transparent endpoints with both blend/aerial modes. A two-star/three-moon scene
also updates one shared secondary response. HDR endpoint tolerance accounts for half-float
precision; an initial absolute-only gate was too strict at values around 10-20.

With Vulkan validation enabled, `--atmosphere-lighting-only`, `--atmosphere-cache-only`,
`--particles-only`, `--finite-star-cache-only`, `--finite-star-source-only`, `--shine-only`
and `--ringshine-only` pass. Application `--self-test` completes 13 frames cleanly.
The existing full-suite VRS shadow quality failure is documented in the
[single-source report](atmosphere_planetshine_prototype.md); it is a separate unresolved issue.

## GPU measurements

RTX 4060 Ti, 2560 x 1440, Release. Three rotated orders, 24 warmup and 48 measured frames
per mode. Fixtures compare disabled, three-body, ring-only and three-body-plus-ring
lighting. Moving-altitude cases force secondary refreshes; stationary cases assert zero.
The orbital camera is aimed at the body, with an assertion that one atmosphere remains
visible. An initial off-screen orbital fixture was rejected; its partial measurements
are not used below.

Median secondary bake milliseconds:

| View | Three bodies | Ring field alone | Three bodies plus ring | Added ring cost |
| --- | ---: | ---: | ---: | ---: |
| ground sky | 0.075 | 0.063 | 0.105 | 0.031 |
| orbit | 0.092 | 0.068 | 0.123 | 0.031 |
| finite 30 km | 0.081 | 0.063 | 0.109 | 0.028 |

These are isolated bake/composition GPU timestamps. The injected ring field excludes CPU
profile/face preparation, surface ringshine, other geometry, readback and presentation.
Camera orientation, source directions and medium determine the work; earlier planetshine
timings are not a matched comparison with this fixture. Static fields retain one 2.59 MiB
secondary payload for all four components, and extra ring composition samples are zero.

Cached orbital composition is about 0.071 ms for any enabled secondary mixture versus
0.057 ms disabled; finite composition is about 0.234 versus 0.176 ms. Adding rings to
existing secondary lighting does not multiply composition work. Ground composition has
large scheduling tails and inconsistent cached medians (0.15-0.99 ms); moving cases have
roughly 0.142 ms for all enabled mixtures versus 0.141 ms disabled. Treat those ground
figures cautiously, not as a frame-time guarantee. These are not application FPS results.
Raw distributions are `tests/ring-atmosphere/timing.json`; contracts and logs remain under
ignored `tests/ring-atmosphere/` and `tests/ring-atmosphere-*.log`.
