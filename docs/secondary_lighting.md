# Planetshine and surface ringshine

`RenderSettings::planetshine_enabled` (default on) controls reflected planetary light
on ground, unresolved body billboards, rings, and the one-source atmospheric prototype. Settings → Lighting & Rings
exposes Planetshine beside Ringshine. The existing `ringshine_enabled` switch controls
surface ringshine; hiding rings suppresses its illumination.

Settings → Lighting & Rings → **Direct sunlight only** is available for atmospheric artifact diagnosis. It suppresses planetshine and ringshine and disables their controls while retaining the selected values. Turn it off to use the secondary-lighting paths documented below; atmospheric diffuse fill and ground skylight also return. No reflected-source selection or ringshine tile bake runs while this diagnostic is active.

## Planetshine

`src/rendering/planetshine.hpp` prepares up to four separate reflected sources per
receiver from double precision snapshot positions. The source radiance uses
`color_rgb` as its uniform linear reflectance, stellar luminosity and inverse-square
star distance, and the integrated material phase response. Receiver irradiance scales
with `source_radius² / receiver_distance²`. Non-HDR mode replaces only the stellar
inverse-square scale with one, matching direct lighting. The original artistic 1.5
multiplier is absent.

The original Lambert/Hapke photometry lives in `src/physics/surface_material.hpp`
and `shaders/common/surface_material.glsl`. Moon/Luna automatically resolve to lunar
regolith, Europa to icy regolith, and other bodies to Lambert. These presets are
illustrative defaults, not fitted measurements. JSON `material` overrides and the
inspector's Surface material controls support both models and all original grain,
roughness, opposition and brightness parameters. Edits use the existing atomic
physical-property draft and invalidate lighting without changing dynamics. Both
system serialization paths persist material definitions.

`rendering::SurfaceMaterialCache` builds 721-angle phase curves with the original
24-point Gaussian radial and 48-point azimuth lune quadrature. Shared curves are
retained by complete packed material value, including reference normalization;
body rows and phase offsets follow current snapshot order. Old unused material keys
are discarded when inputs change. Per-frame GPU phase buffers upload only when the
cache revision changes, at storage binding 25. Instance records are 272 bytes,
including the 48-byte packed material. Atmospheric resources are described below.

Lambert receivers drawn as meshes (above one pixel) read planetshine from a baked
irradiance map instead of evaluating the four sources per fragment. `CelestialRenderer::
bake_planetshine_maps` assigns each such receiver one layer of a 32-layer 128×64
RGBA32F latitude-longitude array (binding 31, `shaders/common/planetshine_lookup.glsl`)
over the geocentric render-frame direction. `celestial/planetshine_map.comp` stores the
summed `shine_irradiance·(direct + diffuse)` of `surface_stellar_light` at the sea-level
ellipsoid point with its ellipsoid normal, so atmospheric transmission, skylight and
finite-disk terminators are retained. A layer rebakes only when its receiver index,
radius, pole, sources, material, atmosphere record or any surface atmosphere LUT
changes; paused scenes bake once. The layer (plus one) travels in
`shine_irradiance[0].w`. The map ignores shading normals, so moonlight shows no
normal-map relief, like surface ringshine. RGBA16F is insufficient: unexposed moonlight
irradiance (~1e-9) underflows half precision. Hapke receivers, receivers beyond 32 layers
and sub-pixel bodies keep the per-fragment path, which skips sources below both the shading
and geometric horizons and sources under 0.1% of the light already accumulated at four
times their unit response. On the Saturn-midnight benchmark the map removes 0.6 ms
(1440p) and matches the per-fragment result within 0.15% at exposure ×10⁴.

Emitter eclipses use the original smooth angular penumbra and source/caster area
bounds. Finite-distance phase remapping extinguishes the visible illuminated cap at
`cos(phase) <= -sqrt(1 - (radius / distance)²)`. The original 0.7-radius sunward
representative-point shift is retained; multiple stars contribute luminance-weighted
direction moments within each source. Hapke phase curves average five stellar disk
samples. Four selected sources retain separate RGB flux and angular radius.

Ground and terrain evaluate the same material response for direct stars and reflected
sources, including finite-disk sampling near horizons and opposition. Terrain bounds
its emission cosine using the visible geometric triangle, as in the original.
Unresolved body billboards use the same integrated phase curves, rather than a
separate approximate surface model. Existing surface textures remain receiver albedo;
source texture means and cloud reflectance are deferred.

Candidates below `radius² / distance² = 1e-8` are discarded and the four brightest
remaining sources are selected by luminance. Stars, black holes and spacecraft do
not emit reflected light. There is no recursive bounce. The original arbitrary
300-source-radius range cutoff and moon-to-moon exclusion are not reinstated.

## Host planet illumination of rings

Settings → Lighting & Rings → Planetshine on rings selects two modes:

- **Analytical** (default): the existing phase-aware two-lobe apparent disk, evaluated
  at ring vertices, now uses the host material phase curve and oblate projected area.
  A stable Lambert crescent series and continuous face split preserve faint crescents
  and grazing continuity. The 128×32 annulus mesh interpolates flux and direction
  moments. Refracted vertices evaluate their solved physical ring hit.
- **Numerical (64 samples)**: per-fragment projected-disk Gaussian quadrature solves
  each ray against the oblate host ellipsoid. Every surface hit evaluates stellar
  distance, incidence, material response and emitted radiance. The solid-angle
  Jacobian and actual ray direction enter the ring's scattering calculation. This
  restores the original higher-cost mode without its artistic brightness multiplier.

Both modes use the complete existing ring slab response, including each layer's
color, optical depth, DHG phase, multiple scattering, opposition and unlit-side
factor. Analytical sources broaden the ring phase by the host's apparent size;
numerical host rays resolve it explicitly. External planetshine retains the host's
four lights, combined independently on the two physical ring faces. Numerical mode
keeps this external contribution separate from its host-disk integration.

## Atmospheres

Atmospheric planetshine combines up to four prepared reflected sources per receiver in
one approximate secondary bake. It follows the existing Planetshine switch and is
suppressed by Direct sunlight only. A diffuse atmospheric ringshine prototype now
occupies one shared-light slot, leaving up to three reflected bodies. It follows
Ringshine and Show rings independently of Planetshine. Approved GPU checks pass;
measured cost and approximation limits are in [diffuse ringshine](atmosphere_ringshine.md).

A separate 96 × 54 × 64 cumulative radiance volume, 96 × 54 sky texture and matching
range texture retain 63 intervals and detached-layer subdivision, with one midpoint per
interval. The bake reuses optical-depth and unit multiple-scattering tables. Its normalized
combined response is multiplied by `PI * summed reflected irradiance` during composition.
Common per-channel RGB/intensity scaling reuses it; relative source weights, altitude,
local directions, apparent radii and optical changes refresh it. Secondary-only changes
leave primary sky/aerial tables valid. Rotation, FOV and exposure reuse both products.

Composition adds secondary radiance after the solar shadow deficit. Both opaque blend
paths and transparent finite endpoints use the same horizon/depth reconstruction; disabling
aerial perspective suppresses finite-endpoint inscattering. No secondary transmittance
volume is needed. Images allocate on demand, total approximately 2.59 MiB per retained
atmosphere for all four sources. The approximate path needs no additional incoming-light
field. All images enter the retained-image budget. Inactive receivers keep the original
composition shader. Engine Telemetry reports Planetshine bake time and LUT updates.

Sources remain receiver-centre, uniformly bright apparent disks. Their sizes soften
horizon visibility; one representative incoming-transmission sample and centre-direction
Rayleigh/particle phases replace disk quadrature. Close-source parallax, illuminated
crescent structure and weak-source merging beyond the four candidates are not resolved.
Very large disks can visibly differ from finite-disk integration.
The ring component instead uses a spatial diffuse annulus field with two face colours,
approximate visible solid angle, one incoming-transmission sample and isotropic phase.

The older analytical, shared-node and full-resolution combined-LUT experiments were removed
on 2026-10-22. Their captures remain under ignored `tests/shine-approaches/`. The new prototype
and its reference measurements are described in [the original prototype report](atmosphere_planetshine_prototype.md).
The default shared bake is described in [the shared-light report](atmosphere_secondary_lights.md).

## Approximation limits

- Source reflectance is configured color; texture means and cloud reflectance remain
  deferred. Neither spin-dependent source texture integration nor recursive bounces
  are implemented.
- External sources use receiver-center distances and directions. Five-point angular
  sampling accounts for disk extent at the receiving BRDF but does not resolve a
  textured source image or per-point parallax across a body.
- Emitter eclipse coverage remains approximate. Reflected-path occluders and eclipses
  across the host's illuminated ring-source disk are not traced.
- Analytical ring sources collapse angular structure into two face moments and use
  approximate near-field phase/centroid geometry. Numerical mode retains the original
  64-ray sampling limits for thin crescents and oblate limbs.
- Ringshine illuminates host surfaces and approximately the host atmosphere;
  rings lighting other moons remain deferred.
  Atmospheric ringshine approximates a diffuse field for the host's own rings.
  Planetshine uses up to four sources, or three alongside rings.

## Validation

`StellarStateTests` compares Lambert, lunar and icy phase curves and directional
photometry against fixtures generated from the original implementation. It checks
material defaults, JSON roundtrips, atomic/stale edits and cache invalidation.

`StellarRenderingTests --shine-materials` checks CPU/GPU material agreement, finite
source horizons and numerical oblate host flux against independent dense angular
integration. `--shine-only` covers ground, both host-ring modes, luminosity scaling,
atmospheric planetshine, independent solar/secondary cache reuse and the
ring visibility/toggle behavior. `--ring-atmosphere-prototype <directory>` additionally
checks diffuse transport, both ring faces, polar occlusion, stellar intensity reuse,
transparent endpoints and a two-star/three-moon scene.
`--secondary-lights-prototype <directory>` additionally checks four-source additivity,
combined-cache invalidation and a two-star/four-moon scene with transparent endpoints.
`--shine-ring-phase` and `--shine-stability-only` retain analytical crescent and
continuous ring-face checks. `--ringshine-only` retains surface ring radiometry.
The app `--self-test` passes. The additional background-ring refraction check in
`--refraction-rings-only` renders zero signal on both this change and the untouched
`a4f4d73` baseline; it is an existing refraction failure, independent of the shine port.

The Hapke roughness equations are ported from the original engine's shared material
implementation, which follows the public-domain USGS ISIS Hapke algorithm. See
the original project's `docs/surface_materials.md` for its provenance and conventions.

With system comparisons, planetshine, surface ringshine and ring illumination follow the composite light groups (`SystemDefinition::illuminates`): a source only reflects starlight its group may receive, and a receiver only sees sources whose group may light it. See `system_comparator_plan.md`, Phase 3.
