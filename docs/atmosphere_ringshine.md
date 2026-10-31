# Diffuse atmospheric ringshine prototype

26 October 2026. GPU rendering checks and the 1440p benchmark were run after the user's
explicit approval. The diffuse field passes its transport/toggle checks and the targeted
regressions. Its broad illumination is a deliberate approximation, not a full annulus
radiative-transfer solution. Future benchmark sessions should respect the user's request
to ask before starting them.

## Representation

The existing shared secondary bake receives one ring field, with room for the three
strongest prepared reflected bodies. The field belongs to the receiving body's own
rings. Rings lighting other planets or moons are outside this path. Ringshine and Show
rings enable it independently of Planetshine; Direct sunlight only suppresses it. The
original single-source diagnostic excludes atmospheric ringshine.

30 October 2026: the haze samples the same per-(host x star) tiles as surface ringshine
([ringshine.md](ringshine.md)). Their quadrature already removes the host's shadow arc
on the ring, uses the oblate horizon, evaluates the ring phase per ring position and
integrates the full radial profile and finite stellar disk. The tile bake now writes
a second quantity into the previously unused alpha channel: the luminance of the
uncosined visible solid-angle integral (scalar fluence for isotropic scattering). RGB
remains the surface's cosine-weighted value, so surface lighting is unchanged.

Per atmospheric cell and illuminating star, the bake shader computes ring-plane
azimuth from the anti-star direction and geocentric latitude of the unstretched cell
position, then reads one tile texel. Scalar RGB is the tile RGB rescaled to the alpha
luminance. The ratio of cosine-weighted to scalar luminance is the mean incoming zenith
cosine, which selects one incoming-transmission and one multiple-scattering lookup.
Stars are weighted by their share of the atmosphere parameters' stellar irradiance.

A CPU cache still reduces the ring profile into about 64 equal-coverage bins and
evaluates their face-averaged response (azimuth-averaged shadow, four phase nodes).
That value only sets the slot's flux for normalization and for competing with
reflected bodies; spatial structure comes from the tiles. The slot carries the host
row in `direction.w` (negative flag) and an RGB scale whose product with the tile
sum gives radiance relative to the shared secondary normalization.

The bake binds the frame's ringshine atlas at binding 16 for ring variants only. Tile
bakes now make writes visible to compute as well as fragment reads, and hosts are not
frustum- or size-culled while the camera is inside their atmosphere, so sky views keep
fresh tiles. The secondary cache adds the local ring pole, local star directions, the
ringshine band count and the ring profile revision to its key.

Direct diffuse scattering uses normalized isotropic phase `1/(4*pi)` for molecular
and particle scattering. It creates a broad glow without separate point-source Mie
halos. The existing multiple-scattering table supplies an approximate additional term.
Medium integration, secondary textures, finite endpoints and composition are shared
with the planetshine path. Ordinary planetshine shader variants exclude this branch.

## Accuracy and limits

Tiles are evaluated on the floor surface, so the field ignores altitude: higher cells
see slightly more ring above the horizon. For Saturn (scale heights of tens of km on a
60,000 km body) this is negligible; a dense check of a uniform 1.14-2.27 annulus found about
+3 percent visible solid angle at 0.003 radii and +15 percent at 0.02 radii at 30 degrees
latitude, so thick atmospheres on small ringed bodies are under-lit aloft. Scalar colour
assumes the cosine and uncosined integrals share chromaticity, and incoming transmission
uses one mean direction. Tile interpolation and the surface quadrature limits apply.
Angular phase in the haze is isotropic even for anisotropic aerosols, and rings embedded
in an extended atmosphere do not receive exact finite-path incoming transmission.

Hosts smaller than 1.5 pixels or outside the frustum are still culled while the camera is
outside their atmosphere; their haze then uses the last tile in that frame slot. Common RGB
scaling, view rotation, FOV and exposure reuse the response. Altitude, camera or star
motion relative to the ring, optical edits, ring profile or band-count changes and changes
in relative secondary lighting rebuild it. Disabling ringshine or hiding rings removes
only their shared-light contribution.

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
On 30 October 2026 the tile-based field re-passed the GPU contracts with Vulkan validation
(no messages), plus `--ring-atmosphere-static`, `--ringshine-only`, `--shine-only` and
application `--self-test`.

With Vulkan validation enabled, `--atmosphere-lighting-only`, `--atmosphere-cache-only`,
`--particles-only`, `--finite-star-cache-only`, `--finite-star-source-only`, `--shine-only`
and `--ringshine-only` pass. Application `--self-test` completes 13 frames cleanly.
The existing full-suite VRS shadow quality failure is documented in the
[single-source report](atmosphere_planetshine_prototype.md); it is a separate unresolved issue.

## GPU measurements

Measured on 26 October 2026 with the previous representative-radius field. The tile
lookup replaced it and has not been re-timed; `--ring-atmosphere-benchmark` now times
reflected bodies only, because the ring field needs RingRenderer tiles.


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
