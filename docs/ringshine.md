# Ringshine — Host-Planet Illumination by Scattered Starlight

Ported from the original Python/ModernGL implementation (`D:/Software/Stellar Forge`,
`engine/rendering/ringshine.py` + `engine/glsl/{common/post}/ringshine_*`). Ringshine is
starlight scattered by a planetary ring onto the host planet's surface: the ring's lit
portion acts as a large secondary light source illuminating the planet's night side.

## Physical Model

For each host planet, the irradiance reaching a surface point from its ring is
pre-integrated into a lookup tile, parameterized by:

- **φ** — azimuth between the anti-solar direction and the surface point's meridian,
  measured in the ring plane. Lookup accepts $[-\pi, \pi]$ and folds to
  $[0, \pi]$ by left/right reflection about the star/ring-normal plane.
  The bake uses the $x^{1.5}$ warp so texel density concentrates at the ansae.
- **sin(latitude)** — sine of the surface point's geocentric latitude relative to the
  ring plane, same $^{1.5}$ warp. Sign distinguishes the two ring faces.
- **solar elevation** — star elevation above the host's ring plane, one baked tile per
  (host × star) pair.
- **sin(star angular radius)** — uniformly bright finite stellar disk. Four
  weighted directions form a reflection-symmetric quartet about the star-center
  direction, covering both horizontal and vertical stellar extent. Two Gaussian
  polar nodes alternate across radial integration nodes; tangent orientation also
  varies across nodes. Reflected horizontal offsets share the same elevation,
  preserving left/right symmetry at every radial node. Solid-angle weights use
  the normalization for unit irradiance perpendicular to the star-center direction.
  Tiny disks use the central direction when sufficiently far from the ring plane;
  finite-disk integration is used for `sin_radius >= 0.01` or within twice the
  apparent sine radius of the ring plane. Stable cap construction avoids
  cancellation for tiny equinox sources.

The integral (`shaders/common/ringshine_integral.glsl`) uses 16-node warped azimuth
quadrature per arc segment, with analytic oblate horizon and planetary shadow arcs.
Numerical quadrature and tile interpolation remain approximations. CPU radial planning
splits at all active layer boundaries, excludes empty intervals, and assigns at least
one band to every occupied interval. Remaining bands are distributed by log-radius span.
Each band has four Gauss-Legendre nodes. The default ten bands normally mean forty radial
samples; if more occupied intervals exist than requested bands, coverage increases the
budget to the interval count (at most fifteen intervals for eight layers).
`Ringshine Bands` in Settings → Visibility controls 4–1024 requested bands; the adjacent
rows show requested and actual radial sample counts per host. Quality changes invalidate
the tile cache. CPU plans and per-slot uploads are reused until quality or profiles change.

The material response reuses the rewrite's unified ring optics contract — `ring_radiance_rgb`
(normalized DHG phase, exact single-scattering slab transfer, approximate H-function-based
multiple scattering, empirical SHOE/CBOE opposition, authored unlit-side factor)
weighted per layer by $\tau_i / \tau_{total}$ — instead of the original's older
`ring_phase`/`ring_transfer` scalar pair. Segment materials are read from the shared
gradient atlas (`sampler2DArray`, binding 8), with radial-footprint mip filtering to
reduce aliasing of fine profile structure. Mips store **premultiplied linear color**,
so the bake divides `rgb` by `alpha` before use.

`ring_radiance_rgb` returns the dimensionless radiance factor $I/F$, where
$F=E_\perp/\pi$. Its single-scattering response is $\omega_0 P T/4$ on either face,
with the same material color and phase parameters. The visible ring multiplies this
response by the scene's $L/d^2$ radiance scale, without another factor of $\pi$.
The bake integrates the response over projected ring solid angle, and the receiver
applies its Lambert factor once. The multiple-scattering closure is empirical,
not a solution of Chandrasekhar's coupled finite-slab X/Y equations; opposition,
texture-dependent phase adjustments and `unlit_factor` are also approximate or authored.
Fine-profile filtering averages material inputs rather than solving subpixel transport.

## Tile Layout & Atlas

Tiles are **65×65** texels (half-texel insets encode φ = 0/π and latitude ±1 exactly),
**16 star columns × 16 host rows**, one RGBA32F atlas per frame-in-flight slot:

- Baked values are **total**: the quadrature integrates $d\phi\,dr$ with the
  $1/\pi$ Lambertian factor left to the consumer.
- RGB is the cosine-weighted (surface) integral. Alpha is the luminance of the same
  visible integral without the receiver cosine, read by atmospheric ringshine
  (`docs/atmosphere_ringshine.md`). Compute and fragment shaders both sample the atlas.
- Bake (`shaders/post/ringshine_map.frag`) and lookup (`shaders/common/ringshine_lookup.glsl`)
  use the same φ/ℓ warp and half-texel clamp, so bake orientation (Vulkan top-down
  viewports) cancels out — tile row 0 maps to ℓ = −1 in both passes.
- Lookup: `ringshine_sample(phi, slat, host_index, star_index)` inverts the warp
  ($^{2/3}$ power), folds φ with `abs`, and clamps to tile bounds; consumers
  multiply by $1/\pi$. Latitude stays signed because the two ring faces differ.
- The half-tile preserves approximately the original longitudinal sample spacing
  and includes the mirror plane as an explicit texel column. It uses 4,225 rather
  than 8,320 bake fragments per tile (49.2% fewer), with no copy/publish pass.

## Vulkan Design (single-pass bake, no ping-pong)

The original amortized bakes across frames with a two-texture GPU accumulator and a
publish pass. The rewrite bakes each tile in **one full-pass draw per frame** into
`ringshine_atlases_[frame_index]`:

- **Per-slot atlases** (`FRAMES_IN_FLIGHT` × 1040×1040 RGBA32F) keep every image's
  writer chain fence-ordered — no cross-frame read-after-write hazards on a persistent
  accumulator, and per-slot tile caches mirror that discipline.
- **No persistent GPU state**: each bake renders the complete radial node range
  in a single draw at the tile's viewport/scissor offset inside the atlas;
  `LOAD` op preserves siblings.
- The scheduler (`RingRenderer::update_ringshine`, called from
  `CelestialRenderer::prepare` after hosts upload) rejects host-surface bounding spheres
  outside the camera frustum and conservatively skips receivers below the 1.5-pixel
  mesh-transition radius. Bounds include positive authored terrain heights and preserve
  partly visible receivers. Hidden hosts retain their per-slot keys; stale tiles refresh
  before the receiver is drawn again. All dirty visible tiles bake in full each frame,
  without node-budget slicing. Cost scales with radial quality, materials and finite-star
  samples. GPU timer `ringshine_bake` includes the preparation pass even when no tiles bake;
  the telemetry panel also shows the current frame's baked-tile count.
- Bake ordering inside the frame: after `hosts_gpu_` SSBO upload, before the celestial
  pass samples the atlas — one command buffer, one implicit barrier chain.

### Cache Keys

`RingshineTileKey` per (host, star column): `profile_serial` (bumped by
`rebuild_profiles` — runs when any ring host's index, body identity or definition
revision changes, and on custom-texture edits; edits to other bodies keep profiles), ring geometry (inner/outer ratios), flattening, band count,
quantized star elevation (`round 5 decimals`) and `sin_radius` (6 significant digits).
Keys use **world-space** positions (`render.position.value`) so camera rebasing never
perturbs them. Star column order matches the celestial renderer's star collection order
(`bodies` scan, `definition.is_star`).

## Bindings & Settings

| Item | Location |
|---|---|
| Host/segment SSBO | binding 7 (`shaders/common/ring_hosts.glsl`, shared with ring draw & surface shadows) |
| Gradient atlas | binding 8 (bake only) |
| Radial sample SSBO | binding 10 (bake only), `(radius, dr_weight)` per node in host equatorial radii |
| Ringshine atlas | **binding 16** (`u_ringshine_map`, sphere/terrain pipelines) |
| Push constants (bake) | 80 B: tile (host, star, node offset, node count), host_params (inner/outer ratios, enable, flattening), solar (elevation, sin radius, segments, bands), two reserved vec4 fields retained for layout compatibility |
| Scene UBO | `ringshine_params` = (ringshine_enabled, host_count, planetshine_enabled, 0) |
| Setting | `RenderSettings::ringshine_enabled` (default on), `ringshine_band_count` (default 10, clamp 4–1024, validated in `apply`) |

## Consumer

`shaders/celestial/sphere.frag` (shared by quadtree terrain via `terrain.vert`): after
the direct-light loop, for each ring host whose `counts_indices.x == f_body_idx` and whose
fragment lies within $1.5\,R_{eq}$ of the host center, compute φ from the fragment's
equatorial projection vs the anti-solar direction and ℓ from the geocentric latitude,
then add

$$L_{ringshine} = \text{albedo} \cdot \frac{M(\phi, \ell)}{\pi} \cdot \text{star color} \cdot \frac{L}{d^2}$$

gated on `ringshine_params.x != 0` (independent of the ring-shadow toggle).

The ringshine atlas is consumed by surface shading, and its scheduler uses
opaque-surface/terrain receiver bounds. A separate CPU-cached diffuse field now prototypes
atmospheric ringshine through the shared secondary bake, without extra atlas draws.
Its approved GPU validation and performance measurements are in
[atmospheric ringshine](atmosphere_ringshine.md). See
`docs/secondary_lighting.md` for the remaining planetshine paths and their limits.

## Contracts (`tests/ring_gpu_tests.hpp::ringshine_contracts`)

- Enabled-minus-disabled flux is positive on masked night-side planet pixels, excluding rings/background.
- Disabling ringshine removes all night-side light.
- Identical successive renders reuse the cache (no re-bake).
- Ring material edits invalidate the cache (re-bake occurs).
- Quality edits rebake and expose the actual sample count.
- A narrow opaque embedded layer contributes at minimum quality and agrees with refinement.
- Visible-ring brightness matches an independent quarter-scaled slab reference; ringshine
  matches a separate double-precision ring-area integral with one Lambert division.

Run: `StellarRenderingTests --ringshine-only` or `--rings-only` (also included in the full suite).
CPU eclipse/optics contracts check grazing transfer against independent double precision,
finite-star weight conservation and radial integration of narrow layers.
An isotropic white-material sweep additionally checks hemispherical scattered plus direct
transmitted flux against the incident beam; this is a limiting-case regression, not a
general validation of the empirical anisotropic multiple-scattering closure.

### Mirror validation

The local `StellarRenderingTests --ringshine-mirror` probe checks thirteen fixtures:
point and finite sources, equinox and oblique/polar illumination, flattening up to 0.3,
4/10/128 radial bands, stellar sine radii up to 0.866, and a 0.99 forward lobe.
It compares the compact bake with direct evaluation of both longitude signs, exercises
the real atlas lookup with poisoned neighboring tiles, and checks off-grid bilinear
interpolation and endpoint clamps. Mirrored lookup values match exactly; the half bake
differs from direct evaluation by at most 0.00073% of peak irradiance in these fixtures.

A separate eight-polar by thirty-two-azimuth stellar rule, sharing the existing ring-arc
integrator, gives at most 4.54% relative RMS error over the sampled grid. The symmetric
four-source rule changes finite-star approximation error: the tiny equinox fixture
increases from 0.71% to 2.51% RMS, while the larger-star fixtures generally improve.
These measurements are fixture comparisons, not a universal error bound. The dense
reference runs in bounded radial batches to avoid GPU driver timeouts.

Reports and probe sources are under `tests/ringshine-mirror/` (git-ignored).
The existing ringshine contracts also pass, including an independent full-disk surface
pixel comparison within 3%; the application `--self-test` completes thirteen frames.
The 49.2% reduction above describes texel work and atlas storage; frame timing has not
been benchmarked for this change.

## Differences from the Original

- **Mirror optimization**: restored through a compact half-width tile and folded
  lookup rather than the original `u_mirror_tile` publish pass. Finite-star source
  sampling is reflection-symmetric; asymmetric phase functions still depend only
  on scattering angle and do not invalidate the geometric reflection.
- **Node-budget slicing** (64 nodes/frame): replaced by full single-pass bakes;
  amortization now lives entirely in the cache keys.


## Extended stellar source validation

Finite-disk radiance and shadowing are evaluated together for every sampled
direction. Disk weights integrate solid angle relative to perpendicular incident
irradiance; their sum can exceed one for large disks. No empirical angular-size
brightness multiplier is applied. The four-point source cubature is distributed
across radial integration nodes and remains approximate; increasing radial bands
does not guarantee monotonic source-error reduction.

The source rule was checked with independently ray-traced 64 by 64 surface views
and Monte Carlo annulus/stellar sampling, plus collision-based plane-parallel slab
transport for isotropic and two-term HG materials. The comparison distinguishes
numerical source/integration error from the existing empirical multiple-scattering
and opposition model. The test `ringshine_contracts` includes a 60-degree-star
rendered pixel checked against dense full-disk quadrature; its tolerance includes
the surface's authored albedo bands. Point-source contracts remain in place.

The extended-source fix changes the ringshine bake. Visible-ring shading and its
separate finite-source/shadow approximations are unchanged. `Direct sunlight only`
disables secondary illumination, including ringshine.
