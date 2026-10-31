# Bloom and four-spike diffraction

Display settings provide Off, Fast, and Four-spike modes. The default is
four-spike with 0.08 scatter strength, a 64-unit highlight limit, 0.65
diffraction weight, unit spike length, 45-degree rotation and a spike threshold
of 4 HDR units. Zero degrees gives a +; 45 degrees gives an X. Settings persist
through `RenderSettingsStore` and `user_settings`. Bloom requires HDR rendering.
Size comparison keeps its original presentation.

`BloomRenderer` runs after atmosphere transport and transparency resolution,
before ACES tone mapping and the UI. Exposure is already applied by the scene
shaders and is not applied again. Six half-resolution-and-smaller 13-tap
filtered levels feed a normalized tent reconstruction for the halo. Four-spike
mode adds sparse cross sprites on top of that halo instead of convolving the
whole image.

## First level and spike sources

`bloom_extract.comp` builds the first half-resolution level. Each 8×8 workgroup
covers a 16×16 full-resolution tile and loads its 20×20 apron into shared
memory once, applying highlight extraction per texel. The 13 bilinear taps of
the original downsample all sample at texel corners, so they collapse to a fixed
6×6 stencil read from shared memory. This keeps the original filter while
reading each source texel about 1.6 times instead of 13 times.

The same pass decides spike emission. A pixel's spike fraction is
`diffraction * smoothstep(T, 2T, peak)`, where `peak` is its largest finite HDR
channel before highlight limiting and `T` is the spike threshold. That fraction
of the extracted light is removed from the halo input. The tile sums the removed
light and its luminance-weighted centroid; one invocation appends an 880-byte
`BloomSpike` and increments the instance count of an indirect draw.

Each `BloomSpike` also stores, for each arm, the tile's light projected across
that arm into 26 one-pixel slices. Slices sit on a lattice shared by every
tile (integer values of `dot(pixel centre, arm normal)`), and pixels are splatted
with a quadratic B-spline. Each slice keeps its luminance fraction and the mean
and mean square of its position along the arm. Summed over tiles, the slices
reproduce the image's projected light exactly, so an extended source split
across tiles (a crescent limb, a solar disc) gives continuous bands instead of
one streak per tile. Moment-matched boxes per tile were not enough: limb pieces
clipped by tile boundaries project to ramps, and a curved limb shifts along the
arm from row to row, which produced 40% ripple in a crescent test.

Below the threshold all light stays in the halo, so dim sources and ordinary
lit surfaces cast no spikes and four-spike mode matches Fast for them exactly.
Zero diffraction also skips emission and the spike draw.

## Spike sprites

Each `BloomSpike` is drawn as two instanced quads (12 vertices) with additive
blending directly into the HDR scene. The quad spans the occupied slices and the
arm length plus 13 pixels. Each fragment reconstructs the three nearest slices
with the same quadratic B-spline (together about a 0.7-pixel core). Along the
arm, each slice is a uniform segment of matching mean and variance convolved
with the previous cross PSF: a Cauchy core `1/(1+t²/w²)` with the outer 45%
tapered to zero by a squared smoothstep. The box ⊗ Cauchy term is exact,
`w/(2a) * atan2(2aw, w²+t²-a²)`, so long slices form plateaus rather than one
bright core per tile. Arm length is
`2 * clamp(ceil(48 * height/1080 * length), 4, 192)` full-resolution pixels and
the core width is `2 * max(0.65, 0.8 * height/1080 * length)`; the profile
integral is computed on the CPU when either changes. Each arm carries half the
emitted light, so the sprites return exactly the light removed from the halo.

Capacity is 2048 sprites per frame. When more tiles qualify, for example an
overexposed surface far above the threshold, a tile that fails to claim a slot
keeps its spike light in the halo instead of losing it. Its apron neighbours
assume emission, which leaves a small error at those tile boundaries. Which
tiles overflow depends on workgroup order. Sprite cost scales with the number of
bright tiles and their arm length, not with the viewport area.

## Energy and limits

Highlight extraction preserves values below half the highlight limit. Above
that knee, a continuously differentiable shoulder approaches the limit while
preserving RGB ratios. Extraction happens per source texel before any filtering,
keeping infinities and source motion from changing the extraction footprint.
The 13-tap layout is adapted from the original renderer's
`engine/glsl/post/bloom_downsample.frag`.

The final result is `scene + strength * (halo + spikes - extract(physical))`.
It redistributes the bounded extracted light instead of adding an unnormalized
copy. The actual solar disc and scene radiance retain their original HDR
brightness. Finite nonnegative HDR values are capped at 65000 before filtering
and in the halo correction; NaNs become zero. Sprites add afterwards and are not
clamped again. Screen edges use black extension. Light outside the viewport does
not contribute, and spike arms leaving the viewport are lost, so flux is exact
only away from the boundary.

Orbit capture marks the otherwise unused sign bit of packed half-float alpha.
The shared decoder returns its positive opacity; atmospheric transport preserves
the marker. An optional transparency resolve variant produces both the normal
image and a physical-light image omitting guides. Ring/cloud attenuation and
atmospheric transport remain in both paths. The bloom correction and spike
sources use the physical image, leaving guide-only images unchanged. The
exclusion marker is ignored by the color sort tie break. On fragment-budget
overflow, fallback draw-order fragments cannot be separated and may contribute
to bloom.

When visible orbit guides use legacy transparency, including hardware without
fragment atomics, the application bypasses bloom to preserve that reference
path. Disabling orbit guides allows bloom in the legacy path. The settings page
reports this condition. Opaque-only scenes use the HDR image directly without
the additional physical-light resolve.

Images allocate lazily and retain their dimensions while disabled. Resize retires
old images against GPU completion instead of waiting for the queue. Spike
buffers are per frame in flight. The `bloom` GPU timer is included in
`gpu_post_composite` and shown separately in telemetry, split into
downsample, upsample, apply and spikes.

## Local verification

`StellarBloomTests` / `bloom_gpu_contracts` checks Off/HDR-off/zero-strength
bypass, interior flux, four-arm symmetry and rotation, below-threshold sources
matching Fast exactly, HDR infinities and NaNs, no opposite-edge wrapping, tiny
viewport resize, guide exclusion, subpixel flux and reconstruction phase at odd
viewport dimensions, a saturated full-screen image that overflows sprite
capacity yet stays flat within 1%, and an unsaturated crescent whose spike tails
24 pixels from the limb must vary smoothly (ripple below 3% at 0° and 7% at 45°;
a brute-force per-pixel reference gives 1.6% and 3.1%; the first sprite version
gave 40%). Fully saturated crescents alias at the highlight limit, so their
spikes step wherever the limb gains a pixel, in the reference as well.
`tests/bloom/crescent-*.ppm` previews both angles and Fast. Resolved 6-pixel-radius solar discs at 1000
and 65000 HDR units exercise bounded glare, unmodified disc brightness, halo
circularity and smooth spike-end fading. Saturated discs with 6/12/24-pixel
radii generate `tests/bloom/sun-*.ppm` previews.
`StellarUserSettingsTests` checks nondefault round trips and invalid ranges.
`StellarForge --self-test` cycles the three modes.

Local application timings use the `overview` fixture on the RTX 4060 Ti at
1920x1080, hidden window, 60 warmup / 240 measured frames. The previous
convolution measured 0.644 ms for `bloom` (downsample 0.340, spikes 0.188,
apply 0.072, upsample 0.045). Sprite bloom measured 0.250 ms (downsample 0.150,
of which the shared-memory first level is about 0.106; spikes 0.006, apply
0.047, upsample 0.044), and Fast 0.261 ms. Slice profiles left the overview
timing unchanged (0.232 ms, spikes 0.007). The worst case grew: in the
standalone contracts, a full-screen saturated image at 4K adds roughly 3 ms of
submit/wait time with all 2048 sprites drawn. The previous convolution sources are
kept locally under `tests/bloom-sprites-20261030/previous/`. These are fixture
timings, not hardware-independent guarantees.
