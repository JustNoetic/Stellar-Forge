# Ring rendering performance

## Planetshine cost correction (2026-10-21)

The first planetshine implementation evaluated the full RGB ring response for four
host disk samples per star and up to four external sources, for every active material
at every fragment. This repeated multiple-scattering, opposition, phase and slab
calculations and passed the local material sample array through the secondary helper.
Copying the 224-byte body record into a dynamically indexed shader local added further
compiler pressure. Correctness tests did not expose the resulting timing regression.

The revised path computes smooth incident light in `ring.vert`, using the existing
128×32 annulus mesh and `common/ring_planetshine.glsl`. It accumulates RGB flux and
luminance-weighted direction moments separately on the two physical faces, then
interpolates four vectors. Host directions are evaluated only per vertex; the subsequent
crescent correction replaces the fixed four samples with two phase-aware disk lobes.
Refraction solves the physical lighting position at vertices and exposes its table
samplers to the vertex stage; upload/observer barriers cover both graphics stages.

Fragments evaluate at most two single-scattering slab transfers, shared by all active
layers. Layers retain sampled linear colors, optical-depth weights, unlit-side factors
and the DHG phase, broadened using the host angular size. Secondary light omits the
empirical multiple-scattering and opposition terms. Direct stellar scattering is
unchanged. This trades secondary angular detail and multiple scattering for lower cost;
it is not an equivalent numerical optimization of the original secondary response.

RTX 4060 Ti, 1920×1080 Saturn close-ups, serialized offscreen ring GPU timestamps:

| View | Initial shine off | Initial shine on | Revised shine off | Revised shine on |
|---|---:|---:|---:|---:|
| Lit | 0.230 ms | 0.952 ms | 0.216 ms | 0.286 ms |
| Transmitted | 0.222 ms | 0.779 ms | 0.196 ms | 0.260 ms |

The incremental shine cost drops from 0.56–0.72 ms to 0.065–0.070 ms in these fixtures.
These are medians across three alternating-order on/off repetitions, with 30 warmup
and 70 measured frames per repetition. Atmospheres, ringshine, surface textures and
ring shadows are disabled to isolate this path. This is ring GPU time, not application
FPS or a universal overhead bound. The timing helper uses ordinary premultiplied
rasterization and excludes the OIT resolve; OIT is checked separately for correctness.
Captures include both on and off modes with the
same shader. Intermediate full-scattering face grouping measured 0.474/0.429 ms;
moving that grouping to vertices measured 0.369/0.341 ms before simplifying the
secondary scattering response.

Local reports and initial shader snapshots are in `tests/shine-perf/` and `tests/`.
Reproduce with `StellarRenderingTests --shine-ring-benchmark tests/shine-perf/new.json`;
the output must not exist. `--shine-only` verifies all secondary-light consumers and
both atmosphere composition paths; `--rings-only` checks the 60 direct-light reference
views, material filtering, eclipses, radiometry and ringshine.

Vulkan-validation runs of both suites pass. The broader refraction GPU run passes the
observer, sunrise, terrain, globe, zoomed limb and narrow Saturn ring checks, but fails
`background_ring_pixel_contracts` at zero visible ring flux. Restoring both saved
pre-optimization ring shaders reproduces that same failure; it is not evidence of a
regression introduced by this optimization. That background-ring issue remains open.

The first optimization changed the annulus shaders to read individual SSBO fields
instead of copying a complete `RingHostGPU` into a local variable or a function
argument. The original host was 976 bytes, including eight segments and sixteen
shadow casters. Dynamically indexing those local copies was the dominant cost in
the measured Saturn close-ups. Current hosts are 464 bytes; eclipse candidates
use the shared planetary candidate buffer described below.

The mesh is planar, so its rasterized depth already describes the ring plane.
The fragment shader no longer overwrites `gl_FragDepth` and explicitly requests
early fragment tests. Opaque planet depth can reject hidden ring fragments before
the texture, eclipse and scattering calculations. The pipeline keeps depth writes
disabled for transparent rings.

That first optimization preserved scattering equations, texture filtering, ring
geometry, shadow casters and solar-disc quadrature.

## Measurements

Release, NVIDIA GeForce RTX 4060 Ti, Vulkan 1.4.351, 1920x1080. These are median
ring-pass GPU times from serialized offscreen captures, with surface ring shadows
disabled to isolate visible ring rendering. Each mode has 40 warmup frames and
120 measured frames per repetition, three repetitions in alternating mode order.

| View | Original ring pass | Optimized ring pass |
|---|---:|---:|
| Lit side, 5 planet radii | 2.884 ms | 0.478 ms |
| Transmitted side, 5 planet radii | 2.950 ms | 0.467 ms |
| Planet fills view, 1.1 planet radii | 0.944 ms | 0.002 ms |

The lit close-up's complete celestial pass decreased from 2.916 ms to 0.511 ms;
its rings-disabled baseline was 0.033 ms. Keeping the original fragment-depth
write while using individual SSBO loads measured 0.489 ms for the ring pass,
isolating the major close-up improvement to eliminating host copies.

Original and optimized RGBA16F captures were bit-identical for all four benchmark
views: lit, transmitted, occluded and distant. GPU rendering contracts additionally
check ring radiance, planet shadows on rings, ring shadows on the planet, and exact
occlusion when the planet fills the view. The CPU radiative transfer contracts and
application GPU self-test also pass.

## Remaining cost and second optimization

Diagnostic shader replacements measured the following ring-pass medians for the
same lit close-up, starting with the 0.479 ms shader. These are isolated removals,
not additive timings: removing code also changes compiler optimization and data
lifetimes. The replacements are investigation tools, not production options.

| Diagnostic replacement | Ring GPU time |
|---|---:|
| Full shader | 0.479 ms |
| No multiple scattering | 0.459 ms |
| No eclipse evaluation | 0.427 ms |
| One texture tap instead of three | 0.457 ms |
| No opposition factor | 0.477 ms |
| Replace all radiance calculations with sampled color | 0.370 ms |
| Only sample profiles and accumulate opacity | 0.106 ms |
| Constant fragment output, retaining mesh/depth/blending | 0.040 ms |

The expensive path is the general per-fragment layer processing and lighting loop,
including keeping sampled layers alive across the star/shadow/source loops. The
multiple-scattering equations alone account for only a small removable part of
this workload. These experiments do not establish a hardware-counter diagnosis
such as register spilling or memory-bandwidth saturation.

The second optimization stores just RGB and optical depth in a `vec4[8]`, rather
than complete sample records containing per-layer uniform properties and temporary
alpha. A bit mask records positive-depth layers; lighting visits only those layers,
in the original accumulation order. Compact storage alone measured 0.374 ms, and
skipping inactive layers reduced it to 0.330 ms.

Eclipse evaluation now tests a conservative spherical penumbra bound before taking
square roots or evaluating an oblate projected radius. A small outward margin
keeps boundary candidates for the original exact calculation. The scattering
equations, three texture taps, solar-disc quadrature and angular/radial mesh
resolution remain unchanged.

| View | First optimization | Second optimization |
|---|---:|---:|
| Lit side | 0.478 ms | 0.305 ms |
| Transmitted side | 0.468 ms | 0.289 ms |

All four new HDR captures are byte-identical to the first optimization's captures.
GPU contracts additionally exercise a sole active eighth layer, eight overlapping
layers against combined optical depth, and additive illumination from two stars.
Investigation scripts, diagnostic captures and the final capture are retained in
`scratch/ring-cost/`; `profile.py` through `profile5.py` restore production shader
sources in `finally` blocks and reject existing capture directories.

## Shared planetary eclipse candidates

Rings now use the planetary renderer's `eclipse_could_shadow` CPU filter and the
same binding-2 blocker-index buffer. The table contains body receivers followed by
ring receivers; ring host `h` uses receiver index `num_bodies + h`. Every receiver
has an independent offset/count range for each scene star. The ring bound is its
center and outer radius, conservatively enclosing the entire annulus. The parent
planet remains eligible; all rendered nonstellar bodies with positive radius are
considered, without a moon-parent restriction or sixteen-blocker truncation.

`common/eclipse_shadow.glsl` shares the GPU range lookup, blocker loop, conservative
per-pixel rejection and penumbra calculation between spheres, terrain and rings.
Rings retain their projected oblate-caster radius; planetary surfaces retain their
existing spherical-caster behavior. Candidate construction reads ordinary CPU
vectors and uploads the shared table once per frame. The old host caster records
and moon sorting are removed.

This reduces Saturn's list from sixteen blockers to ten, with one retained for
Uranus, in the benchmark fixture. The benchmark records these counts in host/star
order as `ring_eclipse_candidate_counts`. It does **not** produce a meaningful GPU
speedup in this view: lit-side time measured 0.309 ms versus 0.305 ms previously;
transmitted-side time measured 0.295 ms versus 0.289 ms. Removing a few inexpensive
blocker checks does not remove the remaining layer/lighting workload. All four
HDR captures remain byte-identical; the captures are in `scratch/ring-candidates/`.

GPU regressions verify CPU rejection of 24 irrelevant bodies without image changes,
an unrelated body casting an umbra, nineteen retained blockers without truncation,
and distinct nineteen/one blocker lists for two stars. The CPU oracle additionally
checks 96,000 annulus samples across orientations, scales and large translations.

These measurements exclude presentation, physics, UI, Gaia and orbit lines; they
are not application FPS. The remaining visible-ring cost still includes the full
scattering model. GPU timings vary with camera coverage, ring layers, illumination,
hardware and other workloads. The `rings` timer is nested inside the celestial pass;
do not add it to its parent.

## Investigating the 0.3 ms plateau

The 0.3 ms measurement was not a hardware floor. The general fragment shader
carried all eight layer slots, a variable star loop, reflection and transmission,
and four-sample solar-disc quadrature in one executable. Even where the runtime
selected a single source direction, the other paths affected generated code and
temporary storage. Pipeline executable statistics reported 80 fragment registers
and a 21,248-byte executable for the original indexed-candidate shader. These are
compiler statistics, not measured occupancy or a proof of register spilling.

Fresh ablations of that shader measured 0.310 ms normally, 0.231 ms without eclipse
evaluation, 0.225 ms replacing radiance with sampled color, and 0.040 ms with
constant fragment output. Re-sampling profiles during lighting to eliminate the
sample array was slower (0.386 ms, 96 registers). Thus the array's presence alone
was not the explanation, and simply moving more blocker selection onto the CPU
did not resolve the remaining shader workload.

The renderer now makes two conservative decisions before drawing:

* Static radial strip ranges list only the layers that could contribute to their
  triangles. Their bounds include the inward angular chord, the shader's maximum
  filtering footprint (including its minimum physical span), and a numerical
  margin. Layers retain their original accumulation order. Pipeline capacities
  are one, two, four or eight layers, so most disjoint ring bands use one slot.
  Only the index ordering changes; vertices, triangles and mesh resolution stay
  the same. Ranges are cached with the profiles and merged when adjacent masks
  match. This trades some additional draw recording for less fragment work.
* For a flat ring, `abs(dot(N,L)) < star_radius / distance` reduces to the star's
  absolute height above the ring plane being smaller than its radius. Therefore
  whether quadrature is needed is constant across the ring. Camera and source
  side signs are also constant away from a plane crossing. CPU classification
  uses the actual uploaded FP32 centers, poles and stars, evaluated in double
  with an outward rounding margin. A single star whose disc cannot cross the
  plane selects a single-source reflection or transmission executable. Multiple
  stars keep the general star/side loop. Any potentially crossing disc keeps the
  original quadrature. Nearly grazing camera views also retain the general
  full-annulus path, avoiding unstable radial classification.

No scattering equations, texture taps, eclipse candidates or source quadrature
are removed from scenes that require them. These decisions select equivalent
shader paths for the current geometry and lighting.

Before the final one-star specialization, the lit close-up measured 0.206 ms;
the same shader with a full-annulus draw measured 0.224 ms. Forcing the general
finite-source path measured 0.261 ms with radial ranges and 0.287 ms with the full
annulus. Radial ranges alone initially measured 0.261 ms, and specializing the
single-source choice reduced that to 0.211 ms. The successive shader variants
demonstrate that unused general lighting paths and broad layer processing were
avoidable costs, not an intrinsic 0.3 ms minimum.

Resolution scaling of the 0.206 ms version measured 0.068 ms at 960x540 and
0.692 ms at 3840x2160. Repeating sixteen celestial draws in one submission gave
0.198 ms for the last ring draw, versus 0.277 ms for the forced general reference.
This small batching improvement, together with the strong resolution dependence,
locates most remaining cost in fragment work rather than submission latency.
Batched draws reuse depth and blend over the existing color; they are a diagnostic
for sustained work, not independent rendered frames or application FPS.

Ablations of the specialized 0.206 ms shader measured 0.137 ms without eclipses,
0.144 ms with sampled color replacing radiance, 0.176 ms without multiple
scattering, 0.200 ms with one texture tap, and 0.043 ms with constant output.
Hard-coding the fixture's one-star loop measured 0.190 ms, motivating the final
single-star executable selection. These differences overlap and cannot be added
as independent component timings.

The final single-star path measured **0.191 ms lit** and **0.186 ms transmitted**
at 1920x1080, versus 0.309/0.295 ms before this investigation (38%/37% less ring
GPU time). The full lit celestial pass was 0.224 ms with a 0.033 ms rings-disabled
baseline. Occluded and distant rings each measured approximately 0.001 ms.
These final four captures enabled executable statistics; a separate lit capture
with statistics disabled measured **0.197 ms** (36% below the previous 0.309 ms),
with a 0.229 ms full celestial pass. That repeat is in `final-uninstrumented/`.
The one-layer single-star reflection executable reports 48 registers and 12,032
bytes of code, down from the original general shader's 80 registers and 21,248
bytes. This supports the compiler-complexity diagnosis without claiming measured
occupancy or a specific cache bottleneck.

Compared with the previous four 1080p HDR captures, lit and transmitted views
change only 131 and 105 components respectively, each by one FP16 ULP. Occluded
and distant captures remain byte-identical. The equations and sampling model
are unchanged; separate compiled paths can round differently.

Investigation captures, source checkpoints, ablations, compiler statistics and
resolution/batching reports are under `scratch/ring-bottleneck/`. CPU regressions
sample actual mesh triangles across random layer layouts, overlaps, narrow bands,
zero-opacity layers and physical scales. A separate lighting oracle checks
per-fragment source-count and side decisions. GPU comparisons exercise equinox,
mixed-star illumination, tilted poles and both sides of the grazing fallback
against forced full-annulus finite-source rendering, allowing at most two FP16
ULPs for compiler rounding.
All sixty GPU reference views passed with a maximum difference of one FP16 ULP;
Vulkan validation reported no errors during that sweep.

The broader suite exposed an independent-system cache lifetime bug: a raw
definition pointer plus per-simulation revision counters could match a later
definition allocated at the same address. The ring cache now retains the shared
immutable definition through `SystemSnapshot::share_definition()`, preventing
address reuse until the cache changes. A GPU regression alternates twelve
independent systems with and without rings while their revision counters match.
The final CPU eclipse/optics contracts, application GPU self-test and full rendering
contract suite all pass. The full suite includes the sixty reference views and
the independent-system cache regression.

`STELLAR_SHADER_STATS=1` opts into supported Vulkan pipeline executable statistics
through the shared RHI; ordinary runs do not enable the feature or capture flags.
The tested driver reports an implausible high word for its local-memory statistic,
matching an [NVIDIA developer forum report](https://forums.developer.nvidia.com/t/vkgetpipelineexecutablestatisticskhr-reports-64gb-of-local-memory-size/346960).
Do not interpret that field as a valid absolute allocation or evidence of spilling.

## Radial profile filtering

The visible ring profiles now have thirteen mip levels, down to one texel per
profile. The old 4096x64 two-dimensional atlas forced level zero and averaged
three isolated taps. Its small tap count could miss many alternating bands
inside one pixel, producing moire and camera-dependent shimmer.

Profiles are now separate layers of a 4096x1 texture array. Only the radial axis
shrinks through the mip chain, so neighboring profiles never mix. Mips are baked
on profile rebuild, not every frame. Color is converted to linear light and
premultiplied by opacity before reduction; filtered samples recover color before
the existing optical calculations. Transparent texels therefore cannot introduce
colored fringes. The fragment shader uses `log2(max(fwidth(radius) / layer_span *
4096, 1))` and the shared trilinear sampler in one `textureLod` lookup. Magnification
uses the original-resolution level; minification progressively averages fine bands.

The profile array occupies approximately 8 MiB, versus 4 MiB previously. The
completion-managed staging buffer is sized for complete profile/LUT rebuilds
across the frames in flight. The steady render path does no CPU mip generation.

A synthetic 128-stop alternating-band fixture compares eight subpixel camera
positions with an averaged-density reference. The filtered/unfiltered squared
error ratio is 0.0000237. A separate GPU test keeps minified red and blue layers
isolated. CPU tests check uniform-color preservation, linear-light reduction,
opacity conservation and transparent-color rejection. All sixty optimized versus
general-shader reference views still pass within one FP16 ULP, with no Vulkan
validation errors. Profile filtering addresses band minification; mesh silhouette
coverage is a separate rasterization concern.

The final 1080p captures measured 0.194 ms lit and 0.181 ms transmitted, versus
the previous 0.197 ms lit repeat. Occluded and distant ring passes remain around
0.001 ms. This filtering has no measurable added cost in that fixture. Captures
and a before/after crop are retained in `scratch/ring-filtering/`.

## Surface ring-host candidates

The box-filter timings below describe the earlier implementation. The current
default uses a tunable distance-dependent blur; see
[ring shadow blur](ring_shadows.md) for its algorithm and verification.

Surface ring shadows now build one host bit mask per body/star pair on the CPU.
Masks are appended to the existing binding-2 eclipse buffer; the scene UBO records
their absolute word offset. The surface shader visits set bits in host order and
skips ring evaluation entirely for an empty mask. Up to sixteen ring hosts remain
supported. Disabling rings or surface ring shadows builds no masks.

`ring_could_shadow_surface` projects the entire receiving sphere onto the ring
plane from the star center. For receiver center `R`, radius `a`, star `S`, ring
center `C`, and unit pole `n`, define `D = S-R`, `d = dot(D,n)`, and
`f = dot(C-R,n)/d`. The center ray hits `H = R+f*D`. All receiver hits fit a disc
around `H` with radius bounded by `abs(1-f)*a*length(D)/(abs(d)-a)` when
`abs(d)>a`. The filter expands that disc by upper bounds on the shader's stellar,
pixel and radial-profile blur widths. It also includes coordinate-precision
padding and terrain relief. Parallel/uncertain projections retain the host;
planes wholly behind the receiver are rejected. The shader's existing small
light-distance clamp is included in the illumination-angle bounds.

This preserves shadows from a planet's own rings and ring shadows on moons;
there is no hierarchy or self-host exclusion. Moon-to-ring shadows continue
through the separate planetary eclipse candidate lists, including offscreen and
unrelated blockers. Rings and terrain use the same lighting snapshot.

The default penumbra uses a single box filter over the prefix-integral extinction
LUT, reducing six texture samples to two. It retains finite-star angular size,
distance-dependent softness, grazing stretch and pixel filtering. Setting
`ring_shadow_single_box=false` restores the previous three weighted boxes for
comparisons. `ring_shadow_reference_all_hosts`
bypasses CPU filtering and supplies every host in the surface shader for HDR and
timing comparisons. These are diagnostic render settings without UI controls.

CPU contracts compare the host bound against 768,000 randomized surface/interior
points, including small coordinate scales, grazing light and pixel footprints.
GPU contracts compare filtered HDR pixels exactly against the all-host reference
and cover own rings, ring-to-moon shadows, the sixteenth host, independent stars,
relief and grazing illumination. Existing ring-receiver tests also verify that
assigning a blocker as a moon preserves its eclipse on the rings.

```powershell
.\build\StellarRenderingTests.exe --surface-ring-shadows
.\build\StellarRenderingTests.exe --surface-ring-benchmark scratch/surface-ring-shadows/new-capture
```

The surface benchmark uses controlled 1080p moon close-ups with four ring hosts,
including three irrelevant systems. It compares all hosts/three boxes,
filtered/three boxes and filtered/one box, with rotating mode order, 40 warmup and
240 measured frames per mode across three repetitions. CPU record timing includes
candidate construction, mask uploads and draw recording. GPU total includes
preparation and visible rings; it is not application FPS. The all-host mode skips
mask construction/upload. HDR equality is required for filtered/three-box
captures. Reports and clamped previews are retained in the requested directory.

Release measurements on the RTX 4060 Ti at 1920x1080, 2026-10-18:

| Close-up | All hosts, three boxes | Filtered, three boxes | Filtered, single box (default) |
|---|---:|---:|---:|
| Moon in ring shadow | 0.144384 ms | 0.106496 ms | 0.096256 ms |
| Rings irrelevant to moon | 0.116736 ms | 0.062464 ms | 0.062464 ms |
| Broad stellar penumbra | 0.136192 ms | 0.099328 ms | 0.092160 ms |

These are median total GPU times for the controlled celestial pass. Filtering
alone preserved every HDR component in all three captures. The single box
changes the penumbra profile, with the same distance/angle dependence, and saves
another 0.007–0.010 ms where ring filtering is active. The clamped previews were
visually inspected before enabling it by default.

CPU record medians were 5.2/6.0/5.5 microseconds without masks and
5.5/5.4/5.4 microseconds with the default filtered single-box path, in the table's
scene order. The filtered three-box path measured 6.4/5.5/5.7 microseconds.
Differences are around the timing variability; these measurements establish no
material CPU regression for this small fixture, not a CPU speedup or a universal
overhead bound. CPU filtering scales with shaded receiving bodies, lights and
ring hosts. Reports and previews are retained in `scratch/surface-ring-shadows/run1/`.

### Grazing moon penumbra rejection

The surface shader previously rejected ray-plane hits using a radial margin of
`3 * angular_star_radius * hit_distance + 2 * pixel_size` before evaluating the
actual blurred profile. That margin omitted the ring-plane footprint stretch.
At grazing illumination it could reject a partially shadowed fragment and create
a hard edge even though the blur itself was broad. This affected both box-filter
modes and predates CPU ring-host filtering.

The redundant margin check is removed. `ring_shadow_extinction` now performs the
radial rejection using its actual half-width, including grazing stretch, before
any LUT samples. The default remains two LUT reads per surviving intersection;
CPU host filtering and its conservative blur bounds are unchanged.

The GPU moon regression uses a real-sized star at low ring-plane elevation and
normalizes a scan across the illuminated moon against a shadows-disabled render.
The old path jumped by 0.474 in visibility between adjacent pixels and yielded
only eighteen partial-shadow pixels. The corrected default single-box path has
a largest step of 0.027 and forty-nine partial-shadow pixels; the three-box
reference has a largest step of 0.030 and thirty-seven partial-shadow pixels.
The scan asserts both a broad penumbra and absence of clipped boundaries.

## Reproduction

Build with `STELLAR_BUILD_RENDER_TESTS=ON`, then run:

```powershell
cmake --build build --target StellarForge StellarRenderingTests StellarEclipseTests --parallel 4
.\build\StellarRenderingTests.exe --ring-benchmark scratch/ring-performance/new-capture
.\build\StellarRenderingTests.exe --rings-only
.\build\StellarRenderingTests.exe --ring-filtering-only
ctest --test-dir build --output-on-failure -R '^(eclipse_contracts|rendering_contracts|simulation_tracking_rendering_contracts)$'
```

The benchmark refuses to overwrite an output directory. It writes `report.json`,
raw `.rgba16` HDR images and clamped `.ppm` previews. Optional arguments are
`[scene [width height [draws_per_submit [render_mode]]]]`. Scene selects `lit`,
`transmitted`, `occluded` or `distant`; omission runs all four. The extent defaults
to 1920x1080 and batch size to one. Modes are `optimized` (default), `reference`
(full annulus and general finite-source path), `full` (full annulus with automatic
lighting selection), `finite` (radial ranges with the general lighting path), or
`base` (automatic lighting with the profile sampler forced to mip level zero).
For example, `--ring-benchmark scratch/ring-performance/batch lit 1920 1080 16 reference`
tests a batch of sixteen general draws. Each view compares disabled
rings, visible rings, and visible rings plus surface shadows. The report records
the local Solar System definition, camera direction/distance, GPU, compiler and
build type; compare captures with matching fixtures and shader instrumentation.
Retain shader sources with dirty-tree baselines. Local investigation captures and
original shader copies are in `scratch/ring-performance/`.
