# Planetshine and surface ringshine

`RenderSettings::planetshine_enabled` (default on) controls reflected planetary light
on ground, unresolved body billboards, and rings. Settings → Visibility
exposes Planetshine beside Ringshine. The existing `ringshine_enabled` switch controls
surface ringshine; hiding rings suppresses its illumination.

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
including the 48-byte packed material. No secondary atmosphere resources are added.

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

Settings → Visibility → Planetshine on rings selects two modes:

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

Atmospheric planetshine and ringshine are currently disabled in the implementation.
The analytical reference, shared-node approximation and secondary-radiance LUT experiment
were removed on 2026-10-22 after their performance comparison. Atmospheres integrate stellar
illumination through the existing sky-view and aerial perspective LUTs. Secondary light
adds no atmospheric scattering, descriptors, images, source uploads or shader variants.
Ground shading still uses atmospheric transmission and diffuse lookup for its own lights.
The comparison captures remain under the ignored `tests/shine-approaches/` directory.

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
- Ringshine illuminates host surfaces only; rings lighting other moons remain deferred.
  Atmospheric secondary scattering remains disabled.

## Validation

`StellarStateTests` compares Lambert, lunar and icy phase curves and directional
photometry against fixtures generated from the original implementation. It checks
material defaults, JSON roundtrips, atomic/stale edits and cache invalidation.

`StellarRenderingTests --shine-materials` checks CPU/GPU material agreement, finite
source horizons and numerical oblate host flux against independent dense angular
integration. `--shine-only` covers ground, both host-ring modes, luminosity scaling,
absence of secondary atmospheric scattering and atmosphere cache reuse.
`--shine-ring-phase` and `--shine-stability-only` retain analytical crescent and
continuous ring-face checks. `--ringshine-only` retains surface ring radiometry.
The app `--self-test` passes. The additional background-ring refraction check in
`--refraction-rings-only` renders zero signal on both this change and the untouched
`a4f4d73` baseline; it is an existing refraction failure, independent of the shine port.

The Hapke roughness equations are ported from the original engine's shared material
implementation, which follows the public-domain USGS ISIS Hapke algorithm. See
the original project's `docs/surface_materials.md` for its provenance and conventions.
