# Eclipse atmosphere refraction assessment

Status, 2026-10-23: the attempted direct per-fragment annulus integration was
rejected and removed. The user measured approximately 1 FPS at 2560x1440 with
the Moon covering roughly half the screen. The implementation placed a
512-sample, three-channel limb loop with nested source/angle quadrature in
surface, terrain, ring and point shaders. Numerical correctness checks did not
establish an acceptable real-time cost, and enabling that path before a
representative performance test was a mistake. Do not revive or tune this
runtime integration method. A subsequent replacement uses cached analytical
shadow textures; see [the implemented design](eclipse_refraction.md). The rejected
integration method has not been restored.

Any future proposal must establish bounded pixel cost and pass a matched
1440p close-Moon performance comparison before production integration. The
optical assessment below remains background research, not approval of the
discarded implementation or a replacement design.

Reviewed 2026-10-22 against both repositories' `project_map.md`, the original
shared eclipse shader and its CPU preparation, and the rewrite's current code
and atmospheric refraction plan. This is an assessment, not a feature implementation.

## Recommendation

Implement eclipse refraction next using the rewrite's physical refraction core.
Reuse the original's shared-consumer architecture and cheap rejection strategy,
but replace its optical calculation. Phase 4 of
[the existing plan](atmospheric_refraction_plan.md#phase-4-eclipse-refraction)
is a useful starting point and needs the corrections below before coding.
Eclipse illumination can proceed independently of the reverted Phase 3 sky/aerial
transport changes. It does not fix the existing observer/sky horizon mismatch.

The first deliverable should be RGB refracted illumination of surfaces and rings,
including Earth casting its shadow on the Moon. Rendering the luminous atmospheric
ring around Earth as seen from the Moon is a separate image-formation feature
(Phase 5); an RGB shadow factor alone cannot draw that ring. An airless Moon
casting a solar eclipse on Earth should retain the current geometric calculation.

## Original implementation

Relevant original files, relative to the original repository:

- `engine/glsl/common/eclipse_shadow.glsl`: `casterShadowTerm`.
- `engine/glsl/common/caster_shadow.glsl`: atmosphere consumers and ray masks.
- `engine/rendering/planetshine.py`: `compute_max_bend`.
- `engine/physics/atmosphere_model.py`: prepared optical columns.
- `engine/app.py`: surface/ring/atmosphere caster uploads.

Good foundations: a single shared eclipse function, explicit AU/km conversion,
separate atmosphere thickness and density scale height, CPU-prepared grazing
columns, Beer-Lambert RGB attenuation, and inexpensive conservative ray masks.
Its weak-refraction estimate `delta ~= (n-1)*sqrt(2*pi*R/H)` is a useful thin,
exponential-atmosphere sanity check.

The final brightness is nevertheless a visual approximation:

| Finding | Consequence | Replacement |
|---|---|---|
| Cubic penumbra followed by a `0.625` power | Incorrect disc overlap even with refraction disabled | Keep the rewrite's circle/ellipse overlap |
| Required bending uses one `0.8 * footprint` offset | No finite stellar-disc integration or consistent near/far image sum | Integrate source radiance over the lens mapping |
| `annular`, `focal_ratio`, `off_axis`, `defocus`, `surface_fade`, and `1-0.7*depth` multiply brightness | No controlled irradiance normalization or flux-conservation guarantee | Use the ray-map area Jacobian or conservative image-area integration |
| Output clamps to one | Removes real local focusing above unobscured irradiance | Bound numerical errors, but permit physical magnification |
| One exponential scale height attenuates the combined grazing column | Mie and molecular columns incorrectly share their altitude dependence | Integrate each prepared extinction profile along the bent ray |
| Ozone uses `sigma_z = 0.707*H` | Ignores the supplied ozone width | Use the actual Gaussian layer and tangent-path integral |
| `compute_max_bend` reduces bending when an optical-depth threshold is crossed and caps at 0.1 rad | Mixes opacity with geometry; reduced bend is still paired with surface-referenced extinction | Keep bending and transmission separate; explicitly reject opaque/trapped paths |
| Refraction is added only inside the geometric solid-body penumbra | Misses atmospheric extinction outside the solid limb and some secondary images | Derive support from the full ray mapping |

The original's eclipse correction also has no explicit opaque cloud-deck radius.
The rewrite already prepares an opaque altitude for pressure-reference atmospheres;
the eclipse path should use it.

## Reproduced numerical checks

`tests/eclipse_refraction_assessment/probe.py` reads the original shader directly,
adapts float literals and two GLSL swizzles for C++/GLM, and compiles its function.
It also exercises the rewrite's double-precision `PreparedRefraction`.
Results are saved locally in that directory's `results.json`.

| Refraction-disabled case | Original remaining light | Uniform-disc area reference |
|---|---:|---:|
| Central transit, caster/source radius = 0.1 | 0.993738234 | 0.990000000 |
| Equal radii, centre separation = one radius | 0.648419797 | 0.608997781 |

These are 0.374 and 3.942 percentage-point errors. In the small-transit example,
the blocked flux is about 37.4% too small. They are isolated shader checks, not
full-image comparisons or a measured error bound over all geometries.

## Corrections to the rewrite plan

1. **Remove the assertion that outside the top-atmosphere penumbra the result
   must equal one.** The plan's own signed mapping `y = p-L*delta(p)` can cross
   the axis. An exterior ray with periapsis 6375 km in a synthetic Earth-sized
   exponential atmosphere (`R=6371 km`, `H=8 km`, surface `n=1.00029`, top 6471 km)
   has `p=6376.121302 km` and `delta=0.013239077 rad` from the existing physical
   core. At `L=2,000,000 km`, its far-side image reaches an axis distance of
   20,102.032 km. A projected source radius of 200 km gives the proposed cutoff
   only 6671 km. This is a counterexample within the plan's paraxial model,
   not a prediction of the brightness of an actual eclipse. Extinction may make
   such paths faint, but cannot justify an unconditional geometric rejection.
   Bound all transmitting branches, expand CPU candidates and GPU interval/VRS
   bounds consistently, and explicitly handle table-domain overflow.
2. **Define finite-source geometry consistently.** `p-L*delta` assumes a distant
   source. For source-to-caster distance `D` and caster-to-receiver distance `L`,
   the paraxial signed coordinate is `(1+L/D)*p-L*delta` for an on-axis source.
   An equivalent caster-plane formulation uses reduced distance `L*D/(L+D)`
   and projected receiver/source coordinates. This can preserve a three-axis
   table; an additional source-distance axis is not automatically necessary.
   Validate the approximation before using it for close or large-angle systems.
3. **Integrate finite-source caustics safely.** The planned
   `(p/rho)*abs(dp/drho)` is singular on the axis for a point source. A finite
   stellar disc regularizes integrated irradiance, but 256 point samples do not
   by themselves establish convergence. Prefer conservative image-area/annular
   integration, or explicit treatment of the singular cells. Test the axis,
   contacts, far-side branch, and resolution convergence against an independent
   dense ray integrator. Include the appropriate projected area for irradiance.
4. **Use one source law for both contributions.** The planned fixed quadratic
   limb-darkening law must apply to the geometric light outside the top radius
   and the atmospheric annulus, with one unobscured-flux normalization. Adding
   limb-darkened annulus light to uniform-disc geometric visibility is
   inconsistent. Keep the current uniform-disc path exactly when the feature is
   off. A zero-bending transparent-atmosphere test should compare against an
   independent solid-occultation calculation using the same active source law.
5. **Use channel-specific physical support.** Red, green and blue have different
   turning/opaque impact limits. `impact_min_km` and stored `turn_radius` are
   green-channel quantities; do not reuse their validity classification for RGB
   eclipse rays. Enforce `ground_radius_km`, not the observer table's virtual
   below-ground floor. Sum exterior reachable branches; handle critical profiles
   without interpolating across invalid/trapped samples.
6. **Document oblateness as an approximation.** An ellipse-aware geometric
   shadow plus a directionally scaled spherical atmospheric table does not
   become an exact oblate refractive lens. Retain the existing projected ellipse
   for geometry and measure any proposed limb approximation against oblate ray
   traces, especially for giant planets.

## Implementation boundaries

Reuse `PreparedAtmosphere::refraction` for physical profiles and deflection;
integrate bent-path extinction with the same species/layer definitions as
`medium` in `shaders/common/atmosphere.glsl`. The current GPU profile packing is
green-only, so it cannot directly supply the RGB eclipse profile.

Cache limb and eclipse tables per prepared optical identity and source-law
version. Retain shared ownership while using pointer identity. Bake lazily for
eligible atmospheric casters, reuse across stars using the table's source-size
axis, and use bounded slots with frame-serial-safe retirement. Camera motion
must not trigger static optical bakes. Use the existing RHI and staging helpers.

Change the shared surface/ring visibility path to RGB, review all scalar early
outs and the multiplication of visibility into BRDF/slab lighting, and preserve
per-star evaluation. Revisit unresolved stellar transit flux explicitly;
unresolved reflected bodies currently omit incoming eclipses and would otherwise
lose lunar-eclipse dimming at the point/mesh transition. Audit the uncommitted
secondary-lighting callers before allocating `surface_flags` fields or bindings.

Atmospheric receivers are additional work: `atmosphere_shadow_visibility` is
scalar and the deficit/history representation clamps losses to [0,1]. RGB
attenuation fits the transport model, but local focusing above one needs an
explicit positive-light contribution or a signed representation. Reuse the same
eclipse evaluator when this integration is added; do not silently clamp focused
irradiance or assume changing a return type completes the transport changes.
Multiple blockers' multiplicative combination remains an approximation.

## Validation and cost

Before enabling the feature, require transparent/no-bend/opaque limits,
flux conservation over a sufficiently large receiving plane, finite-source axis
convergence, RGB reference-ray comparisons, monotonic darkening when raising an
opaque deck, candidate/interval retention of secondary images, independent stars,
cache invalidation and off-state image restoration. Treat a particular clear-Earth
umbra brightness or colour ratio as a diagnostic rather than a universal pass
criterion: aerosol and molecular profiles alter the answer.

The proposed `128*64*8` RGBA16F table costs 512 KiB per atmosphere, as documented.
At 256 stellar samples it requires 16.8 million source evaluations before root
searches or RGB work. Both bake latency and steady-state GPU/CPU costs need
measurement; the existing +0.03 ms runtime target is unverified. Compare disabled,
clear, penumbral, central-totality, annular and strong-lens fixtures at matched
settings. Do not add work to the already expensive per-observer inverse table.

Baseline validation on 2026-10-22: the three relevant targets build with no pending
work; `StellarEclipseTests`, `StellarRefractionTests data/systems`, and
`StellarRenderingTests --eclipse-lighting` pass on the RTX 4060 Ti. GPU geometric
optics cover 1512 samples, with maximum sampled visibility errors of 0.000027315
for circles and 0.000349767 for ellipses. These validate existing foundations,
not the unimplemented eclipse-refraction model.

Physical context: [García Muñoz et al. (2012)](https://arxiv.org/abs/1206.4344)
models refracted atmospheric transmission for lunar eclipses and transits;
[García Muñoz et al. (2011)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2011GL047981)
examines aerosol effects on eclipse illumination. The implementation findings
and numerical counterexample above come from this code review and local probes.
