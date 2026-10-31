# Oceans and clouds

Procedural ocean surfaces and switchable texture-mapped / ray-marched cloud layers, designed with the Kitten
Space Agency water/cloud shaders as a reference (moment-filtered Beckmann ocean
BRDF, Bruneton Fresnel fit, whitecap coverage, Henyey-Greenstein cloud
phase, Wrenninge-style multiple-scattering attenuation, erosion of a coverage
field by detail noise). Cloud weather comes from optional grayscale coverage
maps; volume detail and ocean waves use analytic/hash noise. Both render in the
existing scene passes, without temporal reconstruction or FFT cascades.

## Body definition

Optional per-body JSON objects, parsed into `OceanDefinition` / `CloudDefinition`
(`src/ephemeris/system_manager.hpp`). Only bodies that contain the object, or have
it enabled, serialize it back. Earth in the Solar System preset enables both.

```json
"ocean":  { "enabled": true, "mask": "albedo", "sea_level_km": 0.0,
            "deep_color": [0.004, 0.016, 0.035], "shallow_color": [0.02, 0.09, 0.1],
            "wind_speed_ms": 7.0, "wind_direction_deg": 30.0, "foam": 0.5 },
"clouds": { "enabled": true, "coverage_map": "earth", "type": "cumulus", "coverage": 0.5,
            "automatic_altitude": true, "relative_humidity": 0.7,
            "base_altitude_km": 2.0, "thickness_km": 4.0, "density": 1.0,
            "extinction_per_km": 4.0, "detail_size_km": 0.8, "erosion": 0.3,
            "feature_size_km": 900.0, "wind_speed_ms": 12.0,
            "evolution": 0.05, "color": [1.0, 1.0, 1.0] }
```

- `mask`: `albedo` (classify dark blue albedo texels as water), `height`
  (terrain below `sea_level_km`; spheres fall back to albedo) or `global`.
- Colours accept RGB triples (linear) or `#rrggbb`.
- `evolution`: noise-domain units per simulated hour (cloud shape change).
- `coverage_map`: key in `data/clouds.json`; missing maps use procedural weather.
- `coverage`: 0 clears the layer, 0.5 preserves the source weather, 1 is overcast;
  intermediate values bias the mask, rather than promising an exact area percentage.
- Only cumulus is active. Legacy `type` names still parse and serialize for
  compatibility, but no mixed/stratus/cirrus shaping runs in the renderer.
- `base_altitude_km` is a common flat base. `thickness_km` is the maximum height
  above that base: a texel with brightness 0.5 produces a half-height column,
  while white reaches the maximum. Increasing altitude selects progressively
  brighter, smaller contours, producing tapered mounds.
- `automatic_altitude`: derived cloud base instead of `base_altitude_km`.
  Defaults on when no explicit base is supplied; existing explicit bases remain
  manual. `base_altitude_km` is also the fallback if a physical estimate is unavailable.
- `base_pressure_pa`: optional pressure level for automatic placement (0 = unset).
  The engine inverts the existing hydrostatic profile using the body's gravity,
  molecular mixture and temperature layers. Otherwise an atmospheric
  `cloud_deck_pressure_pa` is used when available.
- With neither pressure level supplied, water clouds use an approximate lifting
  condensation level: a dry adiabatic parcel at ambient pressure reaches water
  saturation. Explicit `H2O` mole fraction determines initial vapor pressure;
  otherwise `relative_humidity` (0..1, default 0.7) supplies the moisture assumption.
  The calculation uses mixture-weighted ideal-gas heat capacities and the
  [Murphy–Koop water saturation formula](https://www.eol.ucar.edu/data-software/conventions-and-standards/water-vapor-pressure-formulations).
  It estimates cloud base, not convection, cloud thickness or global weather.
  Zero moisture, unavailable pressure levels, optical-only atmospheres and
  unsupported temperatures/chemistry retain the manual base. Water estimation
  starts at 180..330 K and excludes pressure-reference giant-planet atmospheres;
  those can use an explicit pressure level for any cloud chemistry.
  Simulation prepares the result on definition changes, not every rendered frame.
- `density` multiplies the explicit `extinction_per_km`; there is no hidden ×12.
- `shape_size_km` is retained for serialization compatibility and is inactive.
  The coverage map defines the large cloud shapes.
- `detail_size_km` and `erosion` only perturb the cap of a column. The noise is
  anchored at the surface and is constant through that column, so it cannot
  stretch the bulk sideways or create a cloud above its map-defined ceiling.
  `feature_size_km` only controls weather when no map is available.
- No inspector UI yet; edit `system.json` and reload.

## GPU data

`SurfaceFluidBuffer` (binding 28, `shaders/common/fluid_surface.glsl`), built each
frame by `CelestialRenderer::prepare_fluids`:

- header: `fluid_flags` (oceans, clouds, cloud mode, focus body),
  `fluid_header` (table Cox-Munk variance, wrapped time), 16 `OceanWaveGPU`.
- per body `SurfaceFluidGPU`: ocean colours/mask/foam, wind and Cox-Munk
  variance, sea-level and surface radii (km), cloud colour/shape/motion.
- cloud style: cumulus marker, detail size, erosion and one-based coverage-map
  handle (0 = procedural weather; actual array layer = handle − 1);
  `cloud_noise` is reserved. Binding 29 is the linear R8 coverage array.

Render settings: `oceans_enabled`, `clouds_enabled`, `cloud_mode` (0 = 2D texture,
1 = experimental volumetric; default 0)
(Settings → Lighting & Rings; persisted in `data/settings.json`).
Cloud shadows are disabled, with no UI toggle. The legacy saved shadow setting
is ignored by rendering.

## Ocean

Shaded in `celestial/sphere.frag` (spheres and terrain).

- Waves: 16 directional deep-water waves, wavelengths from 1.5× the
  Pierson-Moskowitz peak (`ω_p = 0.877 g/U`) down by 0.68 per wave, equal
  steepness so the slope variance sums to Cox-Munk `0.003 + 0.00512 U`.
  Dispersion `ω = sqrt(g k)` with the body's surface gravity.
- Precision: the body nearest the camera (relative altitude) is the focus body.
  Its wave directions lie in the tangent frame under the camera and the CPU
  folds `k·(d·camera_fixed) − ωt` into a per-wave phase in double precision; the
  shader only adds `k·d·(Rᵀ·camera-relative position)`. Other bodies use the
  body-fixed position directly (their short waves are filtered anyway).
- Filtering: waves shorter than ~4 pixel footprints fade out of the normal and
  their slope variance moves into the Beckmann roughness (LEADR-style moments).
  A low-frequency wind field modulates the variance (orbital glint variation).
- BRDF: Beckmann D, Smith-Beckmann G, Bruneton/Neyret Fresnel fit; sun disk
  radius widens the lobe. Sky reflection = Fresnel × the existing ground
  skylight term. Water-leaving light uses the water albedo × (1 − F).
- Foam: Monahan whitecap fraction `3.84e-6 U^3.41`, biased toward crests.
- Imported terrain heights are clamped to sea level (`HeightStream::set_floor_m`,
  part of the geometry cache key) so the ocean is a flat surface over bathymetry;
  atmosphere minima use the same floor.

## Clouds

### 2D texture mode (default)

- Settings → Lighting & Rings → Cloud rendering switches modes immediately and
  persists the choice. The volumetric implementation remains available.
- The existing coarse 64×32 sphere is only a conservative raster proxy. Each
  fragment intersects the exact oblate shell at `base + thickness/2`; no terrain
  tessellation, displaced vertices, volume noise or ray marching is needed.
  Manual and automatically prepared bases work in both modes.
- One filtered coverage lookup per visible shell crossing uses the existing R8
  atlas, longitude wrap, polar clamping, wind and projected-footprint mip LOD.
  Coverage supplies opacity; `density × extinction_per_km × thickness_km`
  controls optical thickness. Slant paths increase opacity toward the limb.
- Front and back crossings are captured separately when visible, with the
  opaque planet rejecting the hidden back side. Lighting uses attenuated starlight
  and sky irradiance from the existing surface-atmosphere lookup at cloud height;
  cloud undersides have reduced direct illumination. The sky contribution is
  `0.5 × sky_irradiance` on both sides. It requires atmospheric surface lighting;
  Direct sunlight only suppresses it with the other indirect-light terms.
- Each crossing enters the existing transparency list at its exact depth. The
  atmosphere compositor applies `T * premultiplied_color + alpha * scattering`
  using the same aerial LUT as the surface, then resolves depth-sorted layers.
  There is no additional aerial LUT or atmospheric integration pass.
- `celestial/cloud_sheet.frag` is a dedicated shader; volumetric code is compiled
  separately in `cloud.frag`. Atmospheres with no eligible shadow casters select
  an `ATMOSPHERE_UNSHADOWED` transparent-composition variant, excluding the general
  shadow-integration fallback. Active eclipses/ring shadows and the atmospheric
  shadow reference mode retain the full shader. The LUT sampling and skylight
  calculations are unchanged.
- The legacy transparency diagnostic retains its approximate opaque-depth
  atmosphere behavior. Default fragment capture supplies correct cloud endpoints.
- Cloud textures currently use resident mip pyramids, not surface tile streaming:
  Earth's native 16000×8000 R8 map with mips costs about 162.8 MiB. Mesh vertex
  count is independent of texture resolution. Close views reveal the flat shell;
  this mode does not produce cloud volume or billowing silhouettes.

### Experimental volumetric mode

- `celestial/cloud.{vert,frag}`: the sphere mesh scaled to the cloud top
  (inflated 0.6% to contain the true ellipsoid) drawn after terrain with
  front-face culling. Normal rendering captures the integrated cloud as one
  premultiplied transparent fragment at its opacity-weighted mean distance.
  Back faces provide one exit surface per ray from inside or outside the proxy;
  the fragment computes the exact ellipsoidal slab interval
  (top/bottom/ground roots) and depth itself. Face selection does not switch at
  the true cloud top, which lies inside the inflated proxy mesh.
- 8–48 steps through the slab. Map brightness supplies local height;
  the density profile has a fixed base and a soft cap. One octave of inexpensive
  surface-anchored value noise only erodes caps. Empty columns and interiors
  skip noise. Analytic integration of the vertical profile and interpolated
  map-defined cap between samples preserves thin clouds and reduces holes and
  step bands. Mips filter the coverage/height
  field by pixel footprint and the horizontal width of a march step.
- Lighting per star: atmosphere transmittance and skylight from the surface
  atmosphere LUT at the segment midpoint, three-lobe HG phase and an analytic
  integral of the local column toward the sun. There are no secondary light-ray
  map/noise probes. Multiple-scattering uses the rational
  attenuation `1/(1+0.35τ+0.065τ²)`.
- The existing transparency renderer carries the colour pass's opacity and
  weighted distance through atmosphere composition without another raymarch
  or opaque depth writes. The hardware/legacy fallback uses a conservative
  depth estimate with at most eight map-only samples.
- Cloud shadows on surfaces are disabled. Cloud self-shading remains active.
- Skipped when the body is under 2 px.

`StellarRenderingTests --clouds-only` checks visibility below, inside and above
the cloud layer and across the inflated mesh boundary, including oblate bodies
and oblique views. Outward rays above the layer must remain clear.
The same suite probes coverage endpoints, longitude seams, poles, mip filtering,
brightness-driven heights, nested tapered contours, fixed bases, wind and cap
erosion on the GPU. Visibility runs both capture and legacy paths, including
cloud-only capture activation when rings and orbit lines are disabled.
It also checks the 2D shell in capture and legacy paths: views above and below,
oblate bodies, proxy crossings, planet occlusion, zero coverage and clear outward
rays above the shell. An underside lighting probe disables view-atmosphere
composition and compares normal sky lighting against Direct sunlight only,
verifying that cloud skylight is independent of aerial perspective. Baker tests
verify native non-power-of-two texel preservation and complete mip layouts.
CPU state contracts cover automatic altitude, gravity,
humidity, composition, pressure inversion and serialized cloud fields.

`StellarRenderingTests --cloud-benchmark <report.json>` measures cloud-off/on
orbital views at offscreen 1080p and 4K using native coverage. It excludes UI,
presentation and readback, with three alternating repetitions and warmed GPU
timestamps. `STELLAR_CLOUD_BENCH_REFERENCE=1` selects generic atmosphere transport.
Local reports are in `tests/cloud-orbit-perf/`. On the RTX 4060 Ti, specialization
reduced the extra atmospheric composition cost from about 0.14 to 0.07 ms at
1080p and 0.51 to 0.26 ms at 4K. Total clouded fixture GPU times changed from
1.01 to 0.93 ms and 3.19 to 3.05 ms respectively. These are fixture measurements,
not application FPS guarantees. Separate 2D compilation alone did not materially
change the cloud draw cost. Native texture resolution and all lighting remain
enabled; full-disc transparency and atmospheric transport still cost GPU time.

## Coverage maps

`scripts/bake_cloud_maps.py` converts equirectangular JPEG/PNG images into a
linear R8 mip pyramid. JPEG luminance supplies coverage; images with alpha use
alpha by default. RGB brightness is not multiplied by alpha and masks are not
sRGB-decoded. `--channel luminance/alpha` explicitly selects the interpretation.

```
python scripts/bake_cloud_maps.py "textures/Solar System/Earth/Earth_clouds.jpg" "textures/Solar System/Earth/clouds"
```

The baker preserves the source dimensions by default, including non-power-of-two
sizes such as Earth's 16000×8000 image. Level zero is the decoded native mask
without resizing. `--width` explicitly resamples to an even width up to 16384.

`data/clouds.json` maps names to pack directories. Maps in one atlas must share
dimensions (even-width 2:1, at most 16384 wide and within the GPU's texture limit);
all mips are validated and loaded at renderer initialization. The atlas stores
only actual maps, without a full-resolution unused fallback layer; the shared
image helper supports a one-layer array view. Earth uses the original
16000×8000 JPEG via a 14-level, approximately 162.8 MiB R8 pack. Longitude repeats, latitude clamps at
each mip, and geodetic latitude matches the surface map convention. The cloud
coverage/height field drifts with wind; evolution changes only fine cap breakup.
New packs/catalog entries require a renderer restart. Body cloud values can be
edited in `system.json` and reloaded. Corrupt or missing packs log a warning and
fall back to procedural weather.

## Known limitations

- Cumulus uses one bounded height field. The integrated column is represented as
  one weighted transparent layer; embedded rings/lines and overlapping media
  still require combined volumetric transport for exact ordering.
- Close silhouettes are limited by the coverage map resolution. The height
  field does not represent overhangs or full 3D billows.
- Clouds are not seen by eclipse/ring shadow code, atmospheric multiple
  scattering or planetshine.
- No refraction/underwater pass, no shoreline depth colour after the sea-level
  clamp, no geometric wave displacement.
- Cloud cost scales with covered pixels; no temporal upscaling yet (KSA uses
  checkerboard upscaling — natural next step).
