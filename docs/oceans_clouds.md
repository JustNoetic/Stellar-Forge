# Oceans and clouds

Procedural ocean surfaces and texture-mapped cloud shells, designed with the Kitten
Space Agency water shaders as a reference (moment-filtered Beckmann ocean
BRDF, Bruneton Fresnel fit, whitecap coverage). Cloud weather comes from optional
grayscale coverage maps. Ocean waves come from FFT cascades
of a JONSWAP spectrum (shading only, no geometric displacement). Both render in the
existing scene passes, without temporal reconstruction.

The experimental volumetric cloud mode was removed on 2026-11-13 so it can be
rewritten (see "Volumetric clouds (removed)" below).

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

- `mask`: `albedo` (classify dark blue albedo texels as water), `specular`
  (linear alpha water mask baked from `<Body>_specular.*` into the albedo pack,
  falling back to `albedo` when absent), `height` (terrain below
  `sea_level_km`; spheres fall back to albedo) or `global`.
- Colours accept RGB triples (linear) or `#rrggbb`.
- `evolution`: noise-domain units per simulated hour (cloud shape change).
- `coverage_map`: key in `data/clouds.json`; missing maps use procedural weather.
- `coverage`: 0 clears the layer, 0.5 preserves the source weather, 1 is overcast;
  intermediate values bias the mask, rather than promising an exact area percentage.
- Only cumulus is active. Legacy `type` names still parse and serialize for
  compatibility, but no mixed/stratus/cirrus shaping runs in the renderer.
- `base_altitude_km` is the layer base. The sheet is drawn at
  `base + thickness_km/2`; `density × extinction_per_km × thickness_km` sets its
  optical thickness.
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
- `detail_size_km` and `erosion` are retained for serialization and are
  inactive (they shaped the removed volumetric caps).
  `feature_size_km` only controls weather when no map is available.
- No inspector UI yet; edit `system.json` and reload.

## GPU data

`SurfaceFluidBuffer` (binding 28, `shaders/common/fluid_surface.glsl`), built each
frame by `CelestialRenderer::prepare_fluids`:

- header: `fluid_flags` (oceans, clouds, reserved 0, focus body),
  `fluid_header` (z: residual mean-square slope, w: Jacobian variance),
  `OceanCascadesGPU` (tangent frame, per-cascade UV offset and 1/size).
  Binding 32 is the ocean moment array (`OceanWaves::moments`).
- per body `SurfaceFluidGPU`: ocean colours/mask/foam, wind and Cox-Munk
  variance, sea-level and surface radii (km), cloud colour/shape/motion.
- cloud style: cumulus marker, detail size, erosion and one-based coverage-map
  handle (0 = procedural weather; actual array layer = handle − 1);
  `cloud_noise` is reserved. Binding 29 is the linear R8 coverage array.

Render settings: `oceans_enabled`, `clouds_enabled`, `cloud_sheet_inline`
(Settings → Lighting & Rings; persisted in `data/settings.json`).
Cloud shadows are disabled, with no UI toggle. The legacy saved shadow setting
is ignored by rendering.

## Ocean

Shaded in `celestial/sphere.frag` (spheres and terrain).

- Waves (`src/rendering/ocean_waves.hpp/.cpp`, `shaders/ocean/*.comp`): five 256²
  cascades of 2, 28, 392, 5488 and 76832 m (KSA layout). Each keeps only the
  wavenumbers above the next larger cascade's Nyquist limit. The initial spectrum
  is built on the CPU (fixed seed) on wind/gravity changes: JONSWAP with fetch
  1000 km, clamped to Pierson-Moskowitz (`ω_p ≥ 0.855 g/U`, `α ≥ 0.0081`),
  γ = 3.3, Donelan-Banner spreading blended 10% toward isotropic, small-wave
  suppression `exp(−k²·10⁻⁴)`. It is stored per frame in flight in host-mapped
  storage buffers (5.2 MiB each), refreshed lazily. Deep-water dispersion uses the
  body's surface gravity; the phase `ω t` is reduced in FP64 on the GPU.
- Per frame (focus body below 1000 km altitude only, ~0.11 ms on an RTX 4060 Ti):
  evolve → row/column inverse FFT (shared-memory radix 2) → moments. Each cascade
  writes (sx, sy, sx², sy²) and (J − 1, (J − 1)², sx·sy) with choppiness 1, where
  slopes are divided by the horizontal stretch `1 − λ D_xx`. `VkContext::build_mip_chain`
  blits them into a mipmapped RGBA16F array.
- Mapping: the CPU parallel-transports a tangent frame under the camera (body-fixed)
  and accumulates the camera's planar motion in FP64, wrapped at 76832 m (all
  cascade sizes divide it). The shader projects the camera-relative body-fixed
  position onto that frame and adds per-cascade UV offsets, so the near field
  never subtracts large coordinates.
- Filtering: `textureGrad` with the projected pixel derivatives. Mean slopes
  shade the normal; the mip variance `E[s²] − E[s]²` plus the residual
  `max(Cox-Munk − spectrum, 0)` becomes Beckmann roughness, so the far field and
  the roughness-only path (non-focus bodies, or above 1000 km) both match Cox-Munk
  `0.003 + 0.00512 U`. A low-frequency wind field modulates slope and variance.
- BRDF: Beckmann D, Smith-Beckmann G, Bruneton/Neyret Fresnel fit; sun disk
  radius widens the lobe. Sky reflection = Fresnel × the existing ground
  skylight term. Water-leaving light uses the water albedo × (1 − F).
- Foam: Monahan whitecap fraction `W = 3.84e-6 U^3.41` (× 2 × `foam`) sets a
  Jacobian threshold at the `1 − W` quantile of the spectrum's Jacobian spread. The
  filtered coverage is the probability that the local Jacobian lies below it, so
  the far field averages to W (Dupuy-Bruneton style).
- Culling: imported tiles below sea level describe their bounds with the same
  floor (`HeightSource::describe(..., floor_m)`), so clamped water surfaces are not
  frustum-culled against seabed bounds.
- Imported terrain heights are clamped to sea level (`HeightStream::set_floor_m`,
  part of the geometry cache key) so the ocean is a flat surface over bathymetry;
  atmosphere minima use the same floor.

## Clouds

### 2D texture shell

- The existing coarse 64×32 sphere is only a conservative raster proxy. Each
  fragment intersects the exact oblate shell at `base + thickness/2`; no terrain
  tessellation, displaced vertices, volume noise or ray marching is needed.
  Manual and automatically prepared bases both apply.
- One filtered coverage lookup per fragment uses the existing R8
  atlas, longitude wrap, polar clamping, wind and projected-footprint mip LOD.
  Coverage supplies opacity; `density × extinction_per_km × thickness_km`
  controls optical thickness. Slant paths increase opacity toward the limb.
- One side only: from above the shell the ray's entry crossing (the top
  surface), from inside/below it the exit crossing (the underside); the choice
  switches as the camera passes through the shell altitude. The far side seen
  past the limb from above is not drawn (it lies behind long, dense atmospheric
  paths and is normally invisible). Shading a single crossing cut the fragment
  shader from 80 to 56 registers, removed a local-memory spill and reduced the
  cloud draw by about 30% (0.22 → 0.15 ms full-disc 1080p, RTX 4060 Ti);
  captures are unchanged except at the limb. Lighting uses attenuated starlight
  and sky irradiance from the existing surface-atmosphere lookup at cloud height,
  along with ringshine and per-body planetshine;
  cloud undersides have reduced direct illumination. The sky contribution is
  `0.5 × sky_irradiance` on both sides. It requires atmospheric surface lighting;
  Direct sunlight only suppresses it with the other indirect-light terms.
- Inline composition (default, `cloud_sheet_inline`, Settings → "Composite 2D
  clouds with the atmosphere"; see below) blends the crossing inside the
  atmosphere pass. With it off, or with progressive atmosphere shadows, the
  crossing enters the existing transparency list at its exact depth instead. The
  atmosphere compositor applies `T * premultiplied_color + alpha * scattering`
  using the same aerial LUT as the surface, then resolves depth-sorted layers.
  There is no additional aerial LUT or atmospheric integration pass.
- `celestial/cloud_sheet.frag` (with `common/cloud_sheet.glsl`) is the cloud
  shader; `celestial/cloud.vert` places the proxy mesh. Atmospheres with no eligible shadow casters select
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

### Inline 2D cloud composition

Measured with the transparency-list path, the cloud draw was only about 30% of
what 2D clouds cost (2560×1440, RTX 4060 Ti; Earth full disc with ring shadows
+2.72 ms in total, of which 0.78 ms was the draw). The rest came from treating
every cloud pixel as a general transparent fragment:
- an A-buffer record and per-pixel sort;
- a separate atmospheric transport pass for the fragment endpoint;
- a second full shadowed in-scatter integration up to the cloud.

The inline path keeps the sheet shader but changes where its result goes:

- **Cloud layer.** `CLOUD_SHEET_INLINE` (`cloud_inline_pipeline_`) stores each
  pixel's front crossing in two full-resolution storage images instead of the
  transparency list (no depth test; occlusion is resolved later):
  - an RGBA32F colour: premultiplied, exposure applied; alpha holds
    `2·body + opacity`;
  - an R32F reverse-Z depth.

  Only the depth image is cleared each frame.
- **Composite.** The atmosphere composite (compute and dual-source fragment
  paths) reads the layer for its own body when the cloud lies in front of the
  scene depth. It transports the cloud with one extra `atmosphere_transport`
  call at the cloud depth, then blends with
  `scattered = (1−α)·scattered + C`, `transmitted *= (1−α)`.
- **Shadows.** The shadow pass reads the cloud depth (binding 34) and stores the
  cloud's shadowed prefix as a third layer of the shadow deficit/endpoint
  images; the coarse VRS images also have three layers. The closed-form path
  (`closed_form_shadow_endpoints`) keeps the plain two-integration loop:
  - **Split.** When the cloud lies in the last tenth of the opaque ray (seen from
    above, a few km before the ground), its loss is a partial sum of the opaque
    integration's own strata. Strata before the cloud count fully, and the
    stratum containing it counts with its visibility over the light
    `W(cloud) − W(t0)`. This costs one light-cache lookup, with no visibility
    evaluations.
  - **Own slot.** Otherwise (seen from below, the cloud is near the camera,
    inside a first stratum reaching far past it), the cloud takes the second
    integration slot. A transparent fragment, then usually an orbit line or ring
    behind the cloud, becomes the split.

  Tried and rejected:
  - One shared sweep through all endpoints, with strata in proportion to each
    segment's light share: it cost as much as a separate integration.
  - Splitting from above and below alike: up to 149% error near the deck seen
    from below.
  - A runtime strata count, or more call sites of the strata loop: these slow
    every pixel through register pressure.

  The sampled and reference shadow paths evaluate the cloud endpoint with a
  separate call.
- **Ordering with rings and orbit lines.** These still use the transparency
  list, which resolves after the composite, over a background that already
  contains the cloud. The atmosphere's fragment-transport pass therefore
  rewrites any fragment behind the cloud (premultiplied F, alpha f) as
  `(1−α)F + f·C`. Blending it over the clouded background then gives
  `C + (1−α)(F + (1−f)X)`, the same as sorting it with the cloud.
- **Fallbacks.** Bodies whose atmosphere is not composited this frame keep the
  transparency-list path, as do progressive (temporal) atmosphere shadows,
  whose history has no cloud layer. The same applies on hardware without
  fragment stores. The `layered-clouds` benchmark variant selects the old path.
- **Limitation.** If two clouded bodies overlap on screen, their storage writes
  race; the old path sorted them.

Measured 2026-11-12 (RTX 4060 Ti, 2560×1440 hidden window, 900 frames after 300
warm-up, all defaults), GPU ms added by clouds over the `no-clouds` variant:

| View | Transparency list (`layered-clouds`) | Inline |
|---|---:|---:|
| earth-ground | +0.70 | +0.40 |
| earth-ground-rings | +2.28 | +1.19 |
| earth-full | +0.94 | +0.54 |
| earth-full-rings | +3.19 | +1.11 |
| earth-eclipse | +3.10 | +1.11 |

What changed in earth-full-rings:
- `atmosphere_transparent` dropped from 2.05 to 0.08 ms;
- `transparency_resolve` from 0.48 to 0.16 ms;
- the shadow pass from 1.65 to 1.48 ms (1.20 without clouds);
- the cloud draw from 0.76 to 0.62 ms;
- the composite's opaque pass gained about 0.4 ms for the cloud transport.

Accuracy was checked with 90-frame 1600×900 high-exposure captures against
256-step `reference-shadows`. Error is the same as the transparency-list path:
- earth-ground-rings: mean 1.82% vs 1.79%; max 8.9% vs 74.7%. The old path's
  maximum came from orbit lines drawn through the cloud.
- earth-full-rings: mean 0.32% in both.
- earth-eclipse: mean 0.05% vs 0.03%.

Image differences from the old path concentrate on transparent fragments behind
overcast cloud. Those fragments are now hidden; the old path drew them through
the cloud.

### Volumetric clouds (removed)

The experimental ray-marched mode (`celestial/cloud.frag`, `cloud_mode` 1) was
removed on 2026-11-13; it remains in git history. It was a 2.5D height field
(coverage brightness extruded into columns with one octave of cap erosion),
marched 8–48 uniform steps through the whole slab, approximated self-shadowing
with a horizontally infinite column, and composited the column as one fragment
at its weighted mean depth. Those choices limited it to tapered mounds with no
sub-texel detail, banding at grazing angles and single-depth aerial
perspective, so it is being rewritten rather than tuned. The rewrite's design
is [volumetric_clouds_plan.md](volumetric_clouds_plan.md).

Reusable pieces for the rewrite: `CloudDefinition` and automatic base altitude,
`CloudMaps` coverage and `cloud_weather`, the exact oblate shell roots and
inflated proxy mesh in `cloud.vert`, and the inline cloud layer (bindings 35/36)
that the atmosphere composite already transports and shadows.

`StellarRenderingTests --clouds-only` probes coverage endpoints, longitude seams,
poles, mip filtering and wind on the GPU. It checks the 2D shell in capture and
legacy paths: views above and below, oblate bodies, proxy crossings, planet
occlusion, zero coverage and clear outward rays above the shell. An underside
lighting probe disables view-atmosphere composition and compares normal sky
lighting against Direct sunlight only, verifying that cloud skylight is
independent of aerial perspective. Baker tests verify native non-power-of-two
texel preservation and complete mip layouts. CPU state contracts cover automatic
altitude, gravity, humidity, composition, pressure inversion and serialized
cloud fields.

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
coverage field drifts with wind.
New packs/catalog entries require a renderer restart. Body cloud values can be
edited in `system.json` and reloaded. Corrupt or missing packs log a warning and
fall back to procedural weather.

## Known limitations

- Clouds are a single shell crossing: close views reveal a flat sheet with no
  volume, billows or silhouettes until the volumetric rewrite lands.
- Detail is limited by the coverage map resolution (about 2.5 km per texel on
  Earth).
- Clouds are not seen by eclipse/ring shadow code, atmospheric multiple
  scattering or planetshine.
- No refraction/underwater pass, no shoreline depth colour after the sea-level
  clamp, no geometric wave displacement.
- Cloud cost scales with covered pixels; no temporal upscaling yet (KSA uses
  checkerboard upscaling — natural next step).
