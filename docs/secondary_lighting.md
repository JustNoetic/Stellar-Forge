# Planetshine and surface ringshine

`RenderSettings::planetshine_enabled` (default on) controls reflected planetary light
on ground, unresolved body billboards, and rings. Settings → Visibility
exposes Planetshine beside Ringshine. The existing `ringshine_enabled` switch controls
surface ringshine; hiding rings suppresses its illumination.

## Planetshine

`src/rendering/planetshine.hpp` prepares at most four secondary directional lights per
receiver from the system snapshot, using double precision world positions. Each source
is a diffuse sphere with authored `color_rgb` as its mean linear albedo. Its reflected
light scale is

```
star_color * (luminosity / star_distance²) * albedo
    * (2 / 3) * LambertPhase * source_radius² / receiver_distance²
```

Non-HDR mode replaces the stellar inverse square scale with one, matching direct light.
The analytic Lambert phase uses the original implementation's finite distance horizon
remapping: the visible illuminated cap disappears when `cos(phase)` falls below
`-sqrt(1 - (radius / distance)²)`. The original artistic 1.5 brightness multiplier is
omitted. Stellar illumination includes approximate smooth spherical eclipse coverage
evaluated at the emitting body, with the source radius broadening the penumbra.

Candidates below `radius² / distance² = 1e-8` are discarded. The four brightest remaining
sources, ranked by luminance, retain separate directions and RGB flux. Stars, black
holes, and spacecraft do not emit reflected planetshine. There is no recursive bounce.
Preparing illumination and selecting sources costs O(bodies² × stars) CPU work, with at
most sixteen stars. Surface shaders consume a fixed four light slots; no GPU scene-wide
source search or new descriptors are needed. Instance records grow from 96 to 224 bytes.

Ground and terrain reuse the existing surface BRDF and atmospheric transmission/diffuse
lookup. Point billboards integrate the same secondary lights with their existing disk
samples, preserving the mesh transition. External planetshine on rings uses the host's
four secondary lights, combined separately on the two physical ring faces.

The host planet is also a source for its own rings. The vertex shader evaluates its
analytic disk flux and two phase-aware apparent disk lobes. The disk center shifts
toward the illuminated crescent by `0.45 * (1 - effective_phase_cosine)` planetary radii;
the lobe spread contracts as that offset grows. Directions remain inside the apparent
planetary disk. This replaces fixed surface samples that missed thin crescents and
prematurely extinguished shine while the analytic phase was still positive. A stable
small-angle Lambert series preserves the faint tail up to the geometric shadow edge.
The finite stellar penumbra remains approximate.

`shaders/common/ring_planetshine.glsl` combines RGB flux and luminance-weighted direction
moments separately above and below the plane, with a continuous split for grazing light.
The existing 128×32 annulus mesh interpolates these four vectors. Refracted vertices
evaluate lighting at their solved physical ring hit when that solve succeeds.

The fragment shader evaluates at most two secondary single-scattering slab transfers,
regardless of the star/source count. Active layers retain their sampled colors, optical
depth weights, unlit-side factors and a DHG phase broadened by the host's angular size.
Secondary light omits the empirical multiple-scattering and opposition terms; direct
stellar light retains its full response. Face selection remains independent of direct
star pipeline specializations. There are no additional passes, images or buffers.

## Atmospheres

Atmospheric planetshine and ringshine are currently disabled in the implementation.
The analytical reference, shared-node approximation and secondary-radiance LUT experiment
were removed on 2026-10-22 after their performance comparison. Atmospheres integrate stellar
illumination through the existing sky-view and aerial perspective LUTs. Secondary light
adds no atmospheric scattering, descriptors, images, source uploads or shader variants.
Ground shading still uses atmospheric transmission and diffuse lookup for its own lights.
The comparison captures remain under the ignored `tests/shine-approaches/` directory.

## Approximation limits

- Planetshine uses authored mean color, without integrating surface textures or clouds.
- External light directions/flux are evaluated at receiver centers. Close encounters,
  very wide rings, and multiple comparable sources beyond four need a richer model.
- Eclipse coverage at the emitter is approximate. Additional occlusion between a
  secondary source and receiver, and eclipses of the host's own ring illumination, are
  not explicitly traced. The receiver's atmosphere still masks its ground horizon.
- The host disk's phase-aware lobes and finite-distance phase remapping approximate
  the crescent and its center of light; they do not resolve surface detail.
- Ring incident light is interpolated across the existing mesh and collapsed to one
  direction per physical face. This smooths angular structure and thin crescent edges.
  Secondary ring multiple scattering and narrow opposition peaks are omitted for cost;
  different source colors on one face share a representative direction. External sources
  also use the host angular-size phase broadening approximation.

## Validation

`StellarRenderingTests --shine-only` checks Lambert phase, luminosity scaling, ground and
host-ring lighting, and absence of secondary atmospheric scattering when the surface
shine toggles change. It also checks atmosphere cache reuse and compute/dual-source
agreement, including captured ring fragments. `--shine-ring-phase` checks crescents,
centroid shifts, monotone fade and the geometric illuminated-cap horizon.
`--shine-stability-only` checks continuous ring-face splitting. `--ringshine-only` retains
surface ring radiometry and area-integral references. `--shine-ring-benchmark OUTPUT.json`
measures ring planetshine independently of the atmosphere.
