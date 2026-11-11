# Thin atmospheric limbs and refracted star images

From a distance an atmosphere is thinner than a pixel. At 29 pixels of Earth
radius one pixel is about 220 km, so the whole atmosphere is under half a pixel
and its bright layers are a few percent of one. Sampling one ray per pixel then
turns a smooth ring into dots: a pixel lights up only if its centre happens to
fall in the layer, and the altitude it hits picks its colour.

Two things were doing this. Scattered light came from a single sky-view lookup at
the pixel centre. Star light was worse: refraction squeezes a star's image into
a sliver a tenth of a pixel wide, and bending the star's mesh per vertex left
rasterisation to keep only the fragments where a pixel centre landed inside it.
Neither is an antialiasing problem. Both are solved by integrating the known 1D
radial profile over the pixel, so no temporal history, MSAA or jitter is used.

## Pixel footprint

`limb_footprint` applies to observers outside the atmosphere, for pixels whose
footprint touches the band between the limb and the top of the atmosphere. The
band's bounds are computed on the CPU (`limb_image.zw`, as sines with a pixel
margin), so other pixels leave after one length and two compares. A square pixel projected onto the radial direction is exactly a trapezoid: box
`|ax|` convolved with box `|ay|`, where `ax` and `ay` are the angular steps per
pixel from the `view_ray` rows. `limb_coverage` is its closed-form cumulative
integral, so the weight of any angular interval is one subtraction and does not
depend on how finely the interval is sampled.

The step sizes are analytic. Differencing neighbouring directions would lose
precision at far-field zoom levels.

## Scattered light

`limb_integrate` walks the footprint in the sky-view LUT's own row coordinate,
which already concentrates samples towards the limb. It takes one tap per LUT row
crossed, at most 16 per half, and weights each tap by its exact coverage. Thick
limbs therefore cost a single tap. Ground rows and sky rows are integrated
separately so transmittance can be chosen for what the pixel actually shows: the
ground share for surface pixels, the sky and space share otherwise. This runs
only while the atmosphere spans fewer than `LIMB_RESOLVED_PIXELS` (32); beyond
that the pixel is much smaller than the scale height and the ordinary path,
including finite terrain endpoints, is used.

## Refracted star images

A ray at apparent angle `a` from the planet's centre leaves the atmosphere at
`s = a - bend(a)`, and `s` may be negative. Along a pixel's azimuth a star's disc
is one interval of `s`. `limb_star_image` therefore solves which part of the
footprint lands on the disc, row by row through the `limb_bend` table, and adds

    coverage * transmittance * star radiance * limb darkening

per RGB channel. Radiance is conserved along rays, so no magnification term is
needed. This produces the primary image, the far-side image and the full ring of
a star directly behind the planet from one expression, with per-channel
dispersion, which the single-channel mesh displacement could not do.

`limb_bend` holds 64 rows of `PreparedRefraction::total_deflection`, sampled on
the same `t` mapping as the sky LUT and rebuilt on the CPU per frame; `w` marks
rows above the apparent horizon, and `limb_image.y` is the largest visible bend,
a safe bound for the early rejection. Rows below the horizon are never read for
light. `star_radiance` matches `stellar_radiance` in `celestial_scene.glsl`.
Star images run at any limb thickness and for every atmosphere, host or not,
whenever refraction is enabled.

Stars are therefore no longer bent as meshes or points for an observer outside
the host atmosphere (`refraction_exterior`). Their unrefracted mesh stays behind
the planet's depth, and where the straight ray meets a star inside the band,
transmittance is restricted to the space share so its light is not counted twice.

That test and the image use `star_disc`: the star's direction and angular radius
seen from the camera, unstretched, in the atmosphere axes. `sun_direction` is seen
from the planet's centre, which is right for the lighting LUTs but sits about
`impact / star distance` away from the mesh (1% of the Sun's radius behind Earth's
limb), and the unmasked crescent of mesh showed through the band. The composite
works in the pole-stretched space where the planet is a sphere, so there the disc
is an ellipse (0.34% for Earth, about 10% for Saturn). `limb_star_behind`
unstretches the pixel's ray before the test; `limb_star_image` maps the pixel's
azimuth plane to true space, intersects the circular disc there and maps the chord
ends back, so the image and the mask both match the mesh.
Observers inside an atmosphere keep the per-vertex mapping, which is well resolved
there. Gaia stars keep `refract_direction`: they are points, not slivers.

## Transparent layers

A cloud fragment covers a whole pixel in the list although the cloud occupies only
the ground share of a thin limb pixel. Fragments inside the atmosphere shell are
scaled by the footprint's ground coverage, and each returns what its opacity would
have removed from the star image, which reaches the camera through the sky share
of the pixel and is not behind the cloud. Without this, 30-40% of a ring's flux was
lost. Fragments outside the shell, such as rings, are untouched.

## Limits

- Transmittance along a refracted ray uses the straight-ray LUT at the same
  impact parameter.
- One azimuth per pixel, which assumes the limb is locally straight across a
  pixel. The composite fades out below 2.5 pixels of radius in any case.
- The planet mesh keeps its binary silhouette.
- Star images ignore other bodies blocking the star, such as a moon in front of
  the Sun, and the eclipse shadow deficit applies only to scattered light.
- The ring's blue flux on the star-behind-planet axis is well below the old
  single-channel mesh result. The per-channel bend places blue at a different
  altitude and the rays involved graze within about 1 km of the surface, where
  extinction is extreme. No independent optical reference has confirmed the
  value. See Verification.

## Verification

`StellarRenderingTests --limb-capture <radius px> <size> <sun offset> <shift px>
<out.raw>` writes raw half-float RGBA for a backlit Earth from 60 radii, with the
Sun offset as a fraction of Earth's angular radius. Running it at 16× size and
box-downsampling gives each pixel's expected value. `STELLAR_LIMB_VARIANT` applies
a benchmark variant such as `no-clouds` or `no-refraction`. These are git-ignored
local tools.

At 29 pixels, native output against its own 16× render:

| Sun offset | Native red flux | 16× red flux | Difference |
| :-- | --: | --: | --: |
| 0 | 24224 | 24188 | 0.15% |
| 0.45 | 22230 | 22188 | 0.19% |

Shifting the view by quarter pixels changes the ring's flux by about 0.1%. The
previous code ranged from 2 to 34146 in the same test.

Against the old mesh at 16×, itself aliased, red flux agrees within 8% (0.6% at
offset 0, 6% at 0.2, 8% at 0.45, 2% at 0.68). The bent image sits at the same
radius, within 0.02 pixels.

## Cost

The limb code raises the composite fragment shader from 40 to 69 registers and
from 21 KB to 57 KB, which lowers occupancy for every pixel. It is therefore
compiled only into `ATMOSPHERE_LIMB` variants. `AtmosphereRenderer` binds them for
an atmosphere only when it is exterior and either thinner than
`LIMB_RESOLVED_PIXELS` on screen or has a star within reach behind it; other views
run the original shader. Hardware without dual-source blending uses the compute
composite, which always includes the limb path.

Per-pixel work is bounded: at most 16 taps per half, and at most 63 table rows per
star for the image. Only pixels on the limb band do it, and only in the composite,
unlike the per-fragment annulus integration rejected in
`eclipse_refraction_assessment.md`. That rule asks for a 1440p close-Moon
comparison. The Moon has no atmosphere and takes neither path, so the closest
equivalent measured is `earth-disc` at 1440p.

`atmosphere_composite` on an RTX 4060 Ti, hidden window, 90 warmup and 300 measured
frames, median of three interleaved runs against the same executable with the
previous shaders. Run-to-run noise is about 0.01 ms in this timer.

| Scene | Before | After | Change |
| :-- | --: | --: | --: |
| `earth-backlit` 1080p (29 px radius) | 0.083 | 0.104 | +0.021 |
| `earth-backlit-zoom` 1080p (280 px) | 0.108 | 0.266 | +0.158 |
| `earth-disc` 1080p | 0.233 | 0.284 | +0.051 |
| `earth-disc` 1440p | 0.405 | 0.469 | +0.064 |
| `earth-limb` 1080p | 0.211 | 0.213 | +0.002 |
| `earth-full` 1080p | 0.404 | 0.405 | +0.001 |
| `overview` 1080p | 0.026 | 0.025 | -0.001 |

A brighter limb also feeds bloom more spike tiles: `bloom_spikes` rises by 0.002 ms
in `earth-backlit` and 0.012 ms in `earth-backlit-zoom`.

`earth-backlit` and `earth-backlit-zoom` are the benchmark scenes for this path.
`STELLAR_SHADER_STATS=1` logs the composite's register counts.
