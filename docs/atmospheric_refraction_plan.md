# Atmospheric refraction: implementation plan

Status: **Phase 1 implemented and verified** (CPU physics core). **Phase 2 implemented with passing CPU and GPU contracts** (per-vertex apparent positions; §4 Phase 2 results). Observer rebuild performance remains above the original budget. Phase 3 bent atmosphere transport was reverted by user request for stability and rendering cost; sky and aerial transport remain straight. Phase 4 incident eclipse lighting now uses the approximate cached model in `eclipse_refraction.md`; its historical design below is superseded. Phases 5–6 remain planned. Source assessment: `refraction_assessment.md` (previous agent). This plan supersedes it where they differ (see §1.2).

## 1. Decisions

### 1.1 Answers to the assessment's open questions

| # | Question | Decision |
|---|---|---|
| 1 | Inverse table or screen-space warp | **Per-vertex inverse table.** No full-screen warp. |
| 2 | Single refracting host for apparent positions | **Yes: only the closest atmosphere.** "Closest" means the smallest distance from the camera to the atmosphere's top sphere. Inside an atmosphere, that distance is zero. A 10% hysteresis stops the host flipping back and forth. |
| 3 | Gravitational lensing | **Out of scope.** Don't port it, and don't leave hooks for it. |
| 4 | Eclipse table fidelity | **Include the apparent-solar-radius axis**, so eccentric orbits and other systems work without re-baking. |
| 5 | Cinematic exaggeration multiplier | **No.** Refraction stays physical. |
| 6 | Fixed solar limb-darkening law in the eclipse table | **Yes.** One shared law, not per-star coefficients. |

### 1.2 Where this plan differs from the assessment, and why

1. **The observer table is indexed by the chord's elevation at the camera, not by impact altitude.** For a camera near the ground, the impact parameter `b ≈ r_c·cos(e)` loses precision near the horizon. A float `b` near Earth's radius has about 0.6 m of error, which becomes about 4e-4 rad of elevation error: more than 1 px at 4K. The table's X axis is therefore `μ = dot(v̂, up)`, computed from unit vectors, with the same horizon-concentrated split mapping as `sky_angle`/`sky_uv`. From orbit, that mapping already concentrates samples at the limb.
2. **The physics core is a ray angular-travel table `Θ(p, r)` built from the Bouguer invariant**, rather than the `G0/G1` moment tables. Every consumer needs one of three things: an inverse solve (vertices), a forward march (bakes) or total deflection plus transmittance (eclipse). `Θ` plus `n(r)` provides all three exactly, including ducting.
3. **Depth correction must use the true ray.** `curved_surface_depth` intersects the *apparent* pixel ray with the ellipsoid. Once vertices are displaced, it has to intersect along the true direction (`normalize(f_world_pos)`) and then project the hit distance along the apparent ray. This is a small change, but without it the host's terrain depth is wrong near the horizon.
4. **Displacement is a range-preserving rotation.** The rotation is in the camera–host-centre–vertex plane. Distance from the camera is unchanged, so reverse-Z depth, the aerial-perspective path length and A-buffer ordering all stay valid without changes. Shading keeps the **true** `f_world_pos`; only `gl_Position` moves.
5. **Stellar limb images (the red ring, the orbital sliver) are not drawn on a proxy mesh.** The per-vertex map draws the *primary* image of every star mesh. The atmosphere composite adds *secondary* (far-side) images per pixel, using the 1D limb profile shared with the eclipse code. Radiance is conserved along rays, so no magnification bookkeeping is needed there.
6. **Manual optical-mode atmospheres don't refract.** Refractivity comes from composition and number density, which only physical mode has.

## 2. Architecture

```
PreparedAtmosphere ──► PreparedRefraction (CPU, double, immutable, per atmosphere revision)
                         n(r) per channel, Θ(p,r), p_lo, δ_total(p), horizon helpers
        │
        ├─► GPU static tables (renderer, keyed by atmosphere revision)
        │     Θ texture · solar-refraction table · limb profile δ/T(p) · eclipse LUT layers
        │
        └─► Per-frame observer table (compute, host only, ~32k cells)
              (μ, d) → apparent rotation, flux factor, occlusion
                 └─► refract_view(): stars · points · spheres · terrain · rings · orbits · cull padding
```

New files:

| File | Contents |
|---|---|
| `src/physics/atmosphere_refraction.hpp/.cpp` | `PreparedRefraction`, `prepare_refraction`, CPU reference ray tracer (tests), envelope and horizon helpers |
| `src/rendering/refraction_renderer.hpp/.cpp` | Host selection, GPU table ownership/cache, observer dispatch, eclipse atlas, per-body refraction records |
| `shaders/common/refraction.glsl` | `refract_view`, `refraction_flux`, `refraction_occluded`, `apparent_sun_mu`, `limb_profile` |
| `shaders/refraction/observer_table.comp` | Per-frame inverse solve |
| `shaders/refraction/limb_profile.comp` | 1D δ_rgb(p), T_rgb(p) via the shared medium evaluator |
| `shaders/refraction/eclipse_bake.comp` | Lens-equation irradiance LUT |
| `docs/atmospheric_refraction.md` | Final architecture doc (replaces this plan when done) |

Ownership follows `docs/state_ownership.md`. `PreparedRefraction` lives inside `PreparedAtmosphere` (prepared by `Simulation`, shared and immutable). GPU tables belong to the renderer, keyed by the optical cache identity. The observer table and per-frame UBO fields exist once per frame in flight.

## 3. Physics core

**Refractivity.** Use Lorentz–Lorenz, which stays accurate for Venus (where `n−1 ≈ 0.015`):
`K = Σ xᵢ·(nᵢ²−1)/(nᵢ²+2)` at the STP reference density `n_STP = 101325/(k_B·273.15)`. Then `(n²−1)/(n²+2) = K·ρ(r)/n_STP`, where `ρ(r)` comes from `log_density`.

**Dispersion.** Three channels (680/550/440 nm). Scale `K` by standard-air dispersion ratios (Ciddor), applied to all gases. This is an approximation, documented as such. Vertices use 550 nm; the eclipse code and limb images use RGB.

**Radial grid.** 1024 samples in double precision, using the same asinh spacing as `log_density`, from `r_lo = R + opaque_altitude_km` (ground, or the cloud deck for giants) to `r_top`. Above `r_top`, `n = 1` and every quantity is analytic.

**Angular travel.**
`Θ(p, r) = ∫_{r_turn(p)}^{r} p dr' / (r'·sqrt(n²r'² − p²))`.
- `r_turn(p)` is the first root of `n(r)·r = p` found by searching downward from `r_top`. That rule handles ducting and super-refraction (Venus).
- Remove the singularity at the turning point with the substitution `r = r_turn + u²`, then use Gauss–Legendre on each grid interval.
- Store `Θ(p, r)` as 512 (p) × 256 (r) entries. The r axis is mapped per row as `sqrt((r − r_turn)/(r_top − r_turn))`, which turns the square-root behaviour at the turn into a linear one.
- The p axis runs from `p_lo = min n·r` over `[r_lo, r_top]` (the trapping or ground limit) to `r_top`. It is asinh-concentrated near `p_lo`.
- **Total deflection:** `δ(p) = 2·[Θ(p, r_top) − acos(p/r_top)]`.
- **Escape test:** a ray launched at radius `r` with invariant `p` escapes upward only if `p < min_{r' ≥ r} n·r'`.
- **Apparent horizon impact** for an observer at `r_c`: `p_h = min n·r` over `[r_lo, r_c]`. Inside the atmosphere this generalizes `bottom_radius()`. From outside, the limb angle is `asin(p_h / r_c)`.

**CPU helpers.** These are used by culling and the UBOs, so no GPU readback is needed:
- `p_h(r_c)`.
- `max_displacement()`: an upper bound equal to `δ(p_lo)`.
- `lowest_visible_chord(r_c)` = `r_c·sin(asin(p_h/r_c) − δ(p_h))`: the smallest straight-chord impact that can still be visible. Use it for horizon culling.

**Reference tracer (CPU, tests only).** An RK4 integration of the ray equation in the (r, φ) plane, using the invariant to fix direction. It also does shooting by bisection. This is the independent oracle for every table.

## 4. Phase plan

Each phase ends green on its contracts and leaves the app shippable. When refraction is off (`refraction_enabled = false`), the output must match the pre-refraction images **pixel-exactly**. That is a standing regression from phase 2 onward.

### Phase 1: `PreparedRefraction` (CPU only)

Tasks:
- Add `PreparedRefraction` to `PreparedAtmosphere` and build it in `prepare_atmosphere` for physical mode only. Manual mode leaves it empty.
- Implement `n(r)` per channel, `Θ`, `δ`, the escape test, the horizon helpers and the reference tracer.
- Cache identity: it rebuilds whenever `PreparedAtmosphere` does. Don't add a new revision counter.

Contracts (`StellarStateTests --refraction`):

| Check | Reference | Tolerance |
|---|---|---|
| Isothermal exponential grazing `δ` | `(n₀−1)·sqrt(2πR/H)` | ≤ 0.5% for `n₀−1 ≤ 1e-3` |
| Earth (`earth_physical_atmosphere()`) ground-observer refraction at h = 0°, 1°, 5°, 10°, 45° | Bennett/Saemundsson, scaled to the profile's P and T | ≤ 0.3′ for h ≥ 5°; ≤ 2′ at the horizon (lapse-rate dependent) |
| `Θ`-table solve compared with the RK4 tracer | Random (r_c, target) pairs: Earth, Venus, Titan, Jupiter (cloud deck) | ≤ 2 µrad, or 0.5% of the local deflection |
| Venus ducting | `p_lo`, trapping radius and escape test compared with the tracer | Exact classification on 10k rays |
| Vacuum and pressure 0 | No refraction, no NaNs | Exact |

Prepare cost: < 5 ms per atmosphere, measured inside the existing `--atmosphere-physics-benchmark` edit case.

**Phase 1 results (implemented 2026-10-20):**

Files:
- `src/physics/atmosphere_refraction.hpp`: header-only, matching `atmosphere_physics.hpp`.
- `src/physics/atmosphere_physics.hpp`: adds `PreparedAtmosphere::refraction` and `prepare_atmosphere_refraction`.
- `tests/refraction_tests.cpp` with the `StellarRefractionTests` / `refraction_contracts` CTest target. This is a separate executable rather than a `StellarStateTests --refraction` flag. It reads `data/systems`.

Implementation notes that differ from the text above:
- **Profile.** The hydrostatic log density is integrated cumulatively on 1024 asinh-spaced samples (0.25 km Simpson steps, split at temperature knots). It is then resampled to an 8192-point uniform-radius Hermite table of n−1 and dn/dr per channel, and every refraction evaluation reads that table.
- **Domain.** Terrestrial bodies use a ground at 0 km and a table bottom at max(profile minimum, −12 km), so virtual turning points below the ground are tabulated. Pressure-referenced giants use ground = bottom = the cloud deck.
- **Impact rows.** Rows are spaced by `p = p_min + s·sinh(span·t²)`. The t² term resolves the square-root singularity at Venus's critical ray. Rows store travel divided by √(r_top − r_turn), and the final row uses the analytic limit `x·√(2/p)`.
- **Table lookups.** These use the exact turning point (node inversion plus two Newton steps) and Catmull–Rom interpolation across rows. The GPU observer pass should do the same: a 1D turning-point lookup and manual cubic interpolation over 4 rows, which is affordable in a compute pass.
- **Steep rays.** Rays with `p < p_min`, which are steeper than any tabulated virtual turn, use direct 64-point quadrature with a quadratic radial map.
- **Ducting.** `critical_radius_km` is non-zero only when a critical-refraction layer exists (Venus in the bundled presets). Monotone profiles use binary search for turning points; ducting profiles use a scan from the top down.

Verified results (gcc 13 / clang 18, `-Wall -Wextra -Wpedantic` clean):

| Contract | Result |
|---|---|
| Exponential atmosphere, weak N (1e-7, 1e-6) | Grazing δ within 0.5% of `N·√(2πR/H)·(1+3H/8R)` |
| Exponential atmosphere, Earth strength, against RK4 tracer | < 2 µrad at 0, 2, 10, 30 km |
| Earth ground observer (bundled US76 profile) compared with Bennett | Within 0.11′ above 0.5°; 0.95′ at 45° (the Pulkovo value at 15 °C is 0.953′); horizon 33.0′ against Bennett's 34.0′ |
| Earth ground observer against RK4 tracer | < 2 µrad at every altitude |
| Horizon dip ratio at 0.1 / 2 / 10 km | 0.911 / 0.918 / 0.940 (k ≈ 0.12–0.17); apparent horizon and orbital limb separate tracer rays exactly |
| Travel table against exact quadrature, all 8 presets | ≤ 1.1e-5 rad, except Venus within 0.1 m of its critical ray (3% relative; log singularity) |
| Inverse solve against RK4 tracer, 310 cases (ground, 3 km, 400 km, lunar distance; stars and finite targets) | Exact path ≤ 2e-8 rad; **tabulated path (the GPU algorithm) ≤ 7.3e-7 rad**, about 0.003 px at 4K |
| Venus critical altitude | 30.6 km; 1500/1500 escape classifications match the tracer |
| Dispersion | δ_R < δ_G < δ_B for every preset; manual optical mode and zero pressure prepare no refraction |
| Existing `StellarStateTests` | Pass unchanged with refraction attached |

Preparation cost on the 2-core verification sandbox is 11–30 ms per atmosphere. Rows run on up to 8 threads, so this should be lower on a desktop. The cost is paid only on load and on atmosphere edits.

### Phase 2: apparent positions (per vertex)

**2a. Host selection and GPU tables (`RefractionRenderer`).**
- Each frame, choose the host: among bodies with a prepared refraction, take the one with the smallest `max(0, |camera − centre| − r_top)`, with 10% hysteresis. If there's no host, `refraction.enabled = 0` and every consumer returns its input unchanged.
- Upload `Θ` as an R32F texture. Float is enough (~1e-7 rad absolute). One texture per prepared atmosphere, cached alongside the existing atmosphere LUT slots under the same LRU budget.

**2b. Observer table (`observer_table.comp`).**
- 256 (μ) × 128 (d) cells, RGBA16F or RG32F+R8. The format is decided by the precision test.
- **X axis:** `μ` of the straight chord at the camera, in two halves split at the geometric horizon or limb, with the `sky_angle` concentration. **Y axis:** distance as `log1p(d/10 m)`, compressed so that `d = ∞` lands on the last row.
- Each cell takes its target, gets `(r_t, φ_t)`, then bisects on the apparent zenith so that the travel angle equals `φ_t`. It uses `Θ` fetches: 24 iterations of the primary-branch solution.
- **Outputs:**
  - R: apparent rotation `z_true − z_app`.
  - G: point-source flux factor `(dz_app/dz_true)·(sin z_app / sin z_true)`, for refractive defocusing at the limb.
  - B: occluded/trapped flag.
- Rebuild only when `r_c` changes. The table depends only on camera radius, so rotation and lateral motion reuse it. Time the pass under a new `refraction_observer` GPU timer, nested in `atmosphere_luts`.
- Multiple images (mirages, ducting) use the primary solution only. Document this.

**2c. `common/refraction.glsl`.**
- `vec3 refract_view(vec3 p_cam)`: takes a camera-relative position (view or render space) and returns the same position rotated about `normalize(cross(C, p))` by the table rotation. The fetch uses `μ` from unit vectors and `|p|`. Cost: 1 bilinear fetch plus about 25 ALU.
- `refraction_flux(...)` and `refraction_occluded(...)` read the same fetch.
- Host data comes from new `SceneCelestialUBO` fields: `refraction_center_radius` (camera-relative centre, `r_top`), `refraction_pole_flattening`, `refraction_params` (enabled, `r_c`, table domain).
- Mapping into flattened atmosphere coordinates: use the same homothetic mapping as `world_to_atmosphere`, so the refracted horizon lines up with the atmosphere LUTs.

**2d. Consumers.** Refract only clip-space positions. Varyings that feed shading stay true.

| Shader | Change |
|---|---|
| `celestial/sphere.vert` | `gl_Position = view_proj * vec4(refract_view(world_pos), 1)`. Add varying `f_apparent_pos`. |
| `terrain/terrain.vert` | Refract `eye` before `projection`, in both the float and the `TERRAIN_CACHED_FP64` / precise paths (rotation in float applied to the double eye; the error stays below 1e-7·\|eye\|). Add `f_apparent_pos`. Do the same in `terrain_reference.vert`. |
| `common/surface_depth.glsl` | Intersect along `normalize(world_pos)` (the true ray). Project `apparent_dir * distance`. Keep `max(depth, …)` so `depth_greater` stays valid. |
| `celestial/ring.vert` | Refract `world_pos`. |
| `celestial/orbit.vert` | Refract `vec3(position − eye)`. |
| `celestial/point_celestial.vert` | Refract the centre before projecting; multiply flux by `refraction_flux`. In `point_visibility`, skip the host's straight-ray mesh test and use `refraction_occluded` instead. Pad the clip test by the envelope. |
| `celestial/starfield_cull.comp` + `starfield.vert` | Refract `dir_eye` using the `d = ∞` row; multiply intensity by the flux factor. Cull with the same function. Add a binding for the observer table. |

**2e. CPU culling.**
- Quadtree horizon culling uses `lowest_visible_chord(r_c)` in place of the occluder radius, in flattened space, combined with the existing relief allowance.
- Pad the frustum planes by `max_displacement()` for host tiles only.
- `TileBounds` and selection stay deterministic: `SelectionView` gets `refraction_occluder_scale` and `refraction_pad_rad` fields, so equality-based caching still works.

**Known limitation (documented):** a triangle or line segment that spans the refracted limb band bends only at its vertices. Orbit lines use `orbit_segments` (default 2048). Rings use their polar grid. Phase 6 adds limb-band subdivision if previews show kinks.

Contracts (`StellarRenderingTests --refraction-only`):
- Vertex mapping compared with the CPU tracer for sampled star directions and finite targets (ground observer, LEO, lunar distance, Venus surface, Titan): ≤ 0.1 px at 3840×2160 and 60° FOV.
- Sunset flattening of the solar disc on Earth: vertical/horizontal ratio at the horizon ≈ 0.80–0.85, matching the tracer.
- Ground/sky boundary: terrain silhouette, sky-LUT split and setting stars agree within 0.5 px at 10 m, 1 km and 12 km altitude. Phase 3 completes this check.
- Off-switch image is pixel-exact; translation and rotation reuse the table when `r_c` is unchanged.

Benchmarks: the `earth-full`, `earth-eclipse` and new `earth-sunset` scenes. Gaia overhead (cull + vertex) should be ≤ 0.05 ms at 1080p, and the observer table ≤ 0.02 ms.

**Phase 2 results (implemented and GPU-tested 2026-10-20):**

New files:

| File | Contents |
|---|---|
| `shaders/common/refraction_solve.glsl` | The observer-table solver and the vertex mapping. Written in a GLSL/C++ common subset (`f` float literals, no swizzles, no arrays), so `tests/glsl_shim.hpp` compiles the shader source unchanged as C++. |
| `shaders/common/refraction.glsl` | The binding 20 UBO, the binding 21 observer table, and `refract_position`, `refract_direction` and `refract_point` (the last two return a flux factor in `w`). |
| `shaders/refraction/observer_table.comp` | 512×128 RG32F table: rotation and visibility flag. |
| `src/rendering/refraction_tables.hpp` | Vulkan-free table packing (`pack_refraction_tables`), per-frame parameters (`refraction_parameters`) and `RefractionHost`. |
| `src/rendering/refraction_renderer.hpp/.cpp` | Host selection (closest atmosphere-top distance, 10% hysteresis) and the table cache: up to 4 atmospheres, staged through the frame command buffer and retired by serial. Per-frame UBOs and observer images; the observer table is rebuilt only when the observer parameters change (GPU timer `refraction_observer`). Also `RefractionFallback`, a disabled UBO and a 2×2 table for renderers used without a host. |
| `tests/glsl_shim.hpp`, `tests/refraction_gpu_tests.hpp` | The shader-as-C++ shim, and the `StellarRenderingTests --refraction-only` GPU contracts (`refraction_gpu_contracts`). |

Changes to existing code:
- **Vertex shaders** (sphere, terrain, terrain reference, orbit, point, starfield draw and cull) refract the clip-space position only. The terrain paths add a view-space shift that is exactly zero when refraction is off, and the FP64 path receives the same shift. Rings use per-fragment ray/plane intersections through a conservative coverage mesh.
- **Depth correction:** `sphere.frag` / `surface_depth.glsl` intersect the true ray and project along the apparent ray (new varying at location 8).
- **Point sprites** skip the host's straight-ray occlusion test and use the table's visibility flag instead.
- **Terrain culling:** the refraction host's frustum is widened by the maximum deflection, and its horizon occluder is scaled to `lowest_visible_chord`. These go through a new `SelectionView::occluder_scale` field.
- **App:** the Gaia cull now runs after `CelestialRenderer::prepare`, which builds the observer table.
- **Settings:** `RenderSettings::refraction_enabled` (default on) with a UI checkbox. The rendering-test probe disables refraction unless a test opts in, so the existing contracts are unaffected.
- **Staging:** `StagingUploads` accepts `R32G32_SFLOAT`.

Design changes made during implementation, with the reason for each:
1. **Short paths use a second-order Born integral.** Targets closer than 2% of the planet radius use `refraction_born`, the ray curvature integrated along a once-corrected chord, blended into the full solve between 0.5% and 2% of the radius. For nearby targets the polar-angle inverse is ill-conditioned: errors reached 1e-3 rad at 120 m from the ground before this change.
2. **Rotation is solved without a ground floor.** The floor is the table bottom (−12 km for terrestrial bodies), so terrain below sea level and cells past the horizon get continuous rotations. The visibility flag still uses the real surface. Occlusion is the rasterizer's job, and blocking at the datum produced arbitrary rotations next to visible cells.
3. **The sampler weights neighbours by visibility,** so invisible cells never pollute visible lookups.
4. **The column mapping below the horizon** is linear over 1.5× the maximum deflection, using three quarters of the lower-half texels, and quadratic beyond. The table is 512 wide: on Titan the visible band just below the horizon was otherwise under-resolved.

Verified here (emulated GPU path, float, compared with double-precision exact or tabulated solves):

| Case | Emulated cells, max error | Vertex lookups p99 / max |
|---|---|---|
| Earth 2 m / 1 km / 50 km | 2.5e-5 / 3.5e-6 / 4.3e-6 rad | 1.5e-5 / 1.1e-5 / 3.2e-6 rad |
| Earth 400 km / lunar distance | 1.9e-6 / 1.3e-5 rad | 9.6e-7 / 1.7e-5 rad |
| Titan 2 m / 1500 km | 2.2e-6 / 3.3e-6 rad | 1.5e-5 / 5.9e-6 rad |
| Jupiter 5 km / 20000 km; Mars 2 m / 300 km | ≤ 8.3e-6 rad | ≤ 8.7e-6 rad |
| Venus 60 km (inside the critical-refraction regime) | 4.5e-6 rad | 4.1e-4 rad (documented 1e-3 bound) |

0.1 px at 4K and 60° FOV is 2.7e-5 rad. Every non-critical case meets it at p99, and none has a visibility-flag mismatch. glslang (Vulkan 1.3) compiles all 14 changed shader variants. The changed C++ translation units, the app, and both GPU test drivers pass `-fsyntax-only` against the pinned Vulkan, volk, VMA, GLFW and ImGui headers. `StellarStateTests`, `StellarTerrainTests` and `StellarRefractionTests` pass.

Review fixes and GPU verification (NVIDIA GeForce RTX 4060 Ti):
- Zero-distance cells explicitly return zero rotation and full visibility. Short paths bypass the polar inverse; nearby lookups classify ground clearance along the second-order bent chord. This fixes visible metre-scale targets disappearing between the first two distance rows.
- Paths ending before atmosphere entry return unchanged. Ordinary profiles use 18 inverse iterations and 16-point direct quadrature; strongly refracting profiles retain 26 iterations and 64-point quadrature. The CPU tracer and emulation tolerances remain unchanged.
- Every body's terrain frustum receives the displacement envelope. Only the host receives the refracted horizon occluder. The world-space envelope includes the inverse oblate transform's angular expansion.
- Position and point mappings normalize after reversing the oblate transform, then restore the original world-space range. Zero-rotation positions return unchanged.
- Observer images permit transfer-source readback, with a compute-to-transfer dependency. The sunrise fixtures aim at the solar centre, keeping both the 1° and 3° discs inside the 4° FOV.
- `StellarRenderingTests --refraction-only` passes observer precision and reuse, zero-distance and nearby visibility, both sunrise centroids, non-host terrain culling, and pixel-exact restoration after switching refraction off. Measured centroid errors are approximately 0.12 px at 1° and 0.01 px at 3°. This restoration check is not a comparison against a separately built pre-feature executable.
- Observer GPU rebuilds measure approximately 0.8–2.0 ms over the Earth ground, 1 km, 400 km and lunar-distance fixtures, down from roughly 5–15 ms before these fixes. **The original 0.02 ms rebuild target is not met.** Gaia overhead and the full benchmark scenes remain to be measured.

Far-terrain follow-up:
- A finite target below the observer can be reached on either side of the ray's turning point. Both CPU and GPU inverse solvers select the branch using the angular travel of a ray tangent to the target radius, then bracket the selected branch. The RK4 oracle can independently check the outward crossing of a lower target radius.
- The observer table's G channel carries physical visibility (`+1` visible, `-1` blocked, `0` for short paths outside the physical domain). Visibility uses only positive values. Vertex displacement samples R with ordinary bilinear interpolation, independently of G; unreachable endpoints receive a bounded continuation rather than a zero-displacement hole.
- For ordinary orbital observers whose limb table excludes steep inward directions, up to a quarter of the columns cover those directions and the remaining columns retain the concentrated limb mapping. This allocation tapers continuously as the limb band expands to the full inward domain, removing an abrupt layout switch with observer distance. Strongly refracting mappings keep their column allocation. Finite ground targets therefore no longer clamp to an unrelated limb ray.
- The inverse retains the best reached candidate when float bisection stagnates. Strongly refracting turning points receive two bounded Newton refinements; failed near-critical outward rays additionally refine the impact offset directly, avoiding absolute-zenith float precision loss.
- New CPU regressions cover Earth observers at 117, 12, 9, 8.01, 8, 7.99, 4.01, 4, 3.99 km and 2 m, with ground and 4/8 km summits at 100–1600 km surface ranges. Maximum sampled displacement error is below 5e-6 rad. Independent RK4 checks cover both radial branches.
- GPU regressions render smooth terrain horizons at 117, 12, 9, 8.01, 7.99 km and 2 m, check for lifted far-side patches against the physical horizon, and sample distant summit displacement from the GPU observer image. Set `STELLAR_REFRACTION_PREVIEW` to an output directory to export these horizon fixtures.
- The follow-up correctness checks pass without relaxing the existing precision or visibility tolerances. Observed Earth GPU rebuild times with the expanded solver are approximately 1–3.6 ms; the original rebuild budget remains outstanding.

Distant-globe and ring follow-up:
- Orbital distance samples previously spanned thousands of kilometres near lunar distance, mixing surface endpoints with rays beyond periapsis. Thin, ordinarily refracting atmospheres now use a two-sided distance mapping concentrated around the front surface intersection, or closest approach for a chord missing the surface. Thick or strongly refracting atmospheres keep the original allocation until the observer is more than roughly eight radii away. The scale follows the projected atmospheric scale height; the exact surface sample is anchored against float cancellation.
- Exterior inverses refine impact distance directly within a float-precision bound, on either radial branch. Geometric endpoints below the virtual table floor continue from the lowest supported endpoint; the physical visibility test still uses the original endpoint.
- Earth ground and 4/8 km targets are checked from 400 km through 384,400 km observer altitude. At 300,000 km the maximum sampled rotation error fell from approximately 3.6e-4 rad to 4.3e-7 rad. GPU disc checks cover 48,000, 54,000, 300,000 and 384,400 km, with no holes or protrusions beyond a two-pixel silhouette tolerance on a 768-pixel smooth globe.
- Rings retain their original 128 by 32 annulus. The temporary dense mesh has been removed; per-fragment intersections now determine ring coordinates, independently of mesh interpolation. Active refraction expands the annulus conservatively and uses a full-annulus draw, since optimized radial ranges no longer bound the apparent bands.
- A GPU fixture renders a narrow Saturn ring at the horizon and compares its central-column centroid with the independent finite-target inverse. The error is approximately 0.08 px. This fixture isolates geometry by disabling the atmosphere composite and using legacy transparency, leaving Phase 3's transport mismatch outside the measurement.

Vertex continuity follow-up:
- Geometry no longer uses a ray-validity mask to select or normalize displacement. The inverse projects unreachable geometric endpoints onto the supported radial/angular boundary, producing a finite continued warp. Physical visibility remains separate for point-source flux; opaque terrain is resolved by the depth buffer.
- Exterior rotation is evaluated from the difference between geometric and apparent impact offsets instead of subtracting two zenith angles close to pi. Exterior impact refinement runs for accepted solutions as well as failed ones, avoiding quantized angle-to-impact conversions when the observer moves. Axis chords are explicitly radial and have zero bending.
- Regressions now include hidden terrain vertices as well as visible ones. At 300,000 km, a 10 m observer movement previously changed displacement by approximately 3.4e-7 rad, while hidden vertices exceeded their endpoint's maximum apparent impact by approximately 26 km. The continued field reduces both errors substantially; the test requires jumps below 5e-8 rad and excess impact below 20 m. Randomizing visibility flags must leave displacement unchanged.
- GPU regressions use an oblate Earth, 0.02 degree FOV, 10 m observer movements at 300,000 km and across the former column-allocation switch, and both cached and reference terrain paths. They check the physical limb with a 1.5-pixel tolerance, substantially tighter in angular extent than the whole-globe fixture. Imported terrain is covered numerically by endpoint heights; the analytic GPU silhouette fixture uses a smooth ellipsoid.
- Current Earth observer rebuild measurements span approximately 2.7–8 ms across the existing GPU fixtures. The original 0.02 ms target remains unmet; filling continued geometry samples and exterior impact refinement need a separate performance pass.

Distant-body consistency follow-up:
- Cloud shells (`cloud.vert`) of non-host bodies apply the same per-vertex `refract_position` shift as surfaces and carry it to the fragment depth, so clouds no longer appear at the true position once a body exceeds 2 px. The host's own clouds stay unrefracted, matching its straight-ray sky: the coarse cloud sphere seen from inside or near the host has triangles spanning large angles, and warping only their vertices tore holes in the cloud deck (`StellarRenderingTests --host-cloud-preview <directory>` renders refraction off/on pairs).
- Atmospheres other than the refraction host are composited along rays rotated by `apparent_rotation` (`refraction_tables.hpp`), the CPU solve of the body centre's refraction. The rotation is folded into that atmosphere's `inverse_view_projection` and `view_ray`, so composite, aerial and shadow passes stay consistent; candidate culling uses the apparent direction. This is a rigid shift: differential flattening across the disc is not applied to the atmosphere, and the CPU solve can differ from the GPU table by its interpolation error (≤ ~3e-5 rad near the apparent horizon). Previously a distant atmosphere was drawn at the unrefracted position, up to several thousand body radii from its surface at high zoom.
- `max_deflection()` interpolates the deflection to a ground-grazing turn instead of taking the lowest sampled row above ground. Earth's clamp rises from 0.019015 to 0.019233 rad, removing a flat rotation plateau for limb rays seen from altitude (errors up to 1.5e-4 rad at 90 km).
- `StellarRenderingTests --distant-atmosphere-only` checks the body, its atmosphere and its clouds against the refracted direction at 2° and 0.5° from the ground.

Known Phase 2 limits:
- **Inconsistent horizons until Phase 3.** The sky and aerial LUTs still use straight rays, so the sky's horizon line and the refracted ground or stars disagree slightly.
- **Mesh limb bending happens only at vertices.** Sphere, terrain and orbit segments can expose interpolation error at extreme zoom. Rings use per-pixel intersections and do not need additional mesh density.
- **Atmospheric shadow deficits** still march straight rays.

### Phase 3: refracted atmosphere transport

**3a. Solar-refraction table**, built once per atmosphere revision: 64 (r) × 256 (`μ_true`), RG16F. It stores `μ_app` and escape visibility for a source at infinity, built from `Θ`/`δ`.

**3b. Bent-ray marching in the bakes.**
- Add `integrate_atmosphere_bent` to `common/atmosphere.glsl`. It marches arc length `s` in the (r, φ) plane, takes direction from the invariant (so it can't drift) and flips at the turning point. It is used by the transmittance, sky-view, aerial and surface-irradiance bakes. Multiple scattering stays straight (isotropic closure).
- The transmittance LUT is reinterpreted as `(r, local apparent μ)`. `sun_transmittance(q, sun)` first converts the true sun `μ` to apparent with the solar-refraction table, then uses the apparent horizon `p_h(r)` for the visibility smoothstep.

**3c. Horizon split.**
- `sky_angle`, `sky_uv` and `angular_segment` replace `bottom_radius()` in the horizon formulas with `p_h(r_c)/n(r_c)` from a new UBO field.
- For an orbital camera, the limb becomes `asin(p_h/r_c)`. The edge stays at `asin(r_top/r_c)`.

**3d. Composite.**
- The bent exit and ground-hit distance come from `aerial_range_lut`, extended to RG32F as (entry, range). The straight `atmosphere_segment` stays only as the entry/miss test, because rays don't bend before entry.
- Depth path length is unchanged: range-preserving vertices give the arc length to a good approximation.
- The finite-transmittance fallback (aerial disabled) keeps the straight approximation. Document that.

**3e. Surface lighting.**
- `surface_atmosphere` bakes run through refracted transmittance, so direct light persists about 2 minutes past geometric sunset on Earth.
- `sphere.frag` uses the apparent `L` for `ndotl` on physical hosts: one solar-table fetch per star.
- External shadows in `atmosphere_shadow.glsl` stay straight for now (documented; phase 6).

Bent-ray bakes apply to **every** prepared atmosphere, not just the host, because each atmosphere bakes for itself. Only vertex displacement is host-only.

Contracts:
- Bent transmittance compared with an 8192-step tracer integration on both sides of the refracted horizon.
- Sun at true elevation −0.5° on Earth: direct visibility > 0 and transmittance matching the tracer.
- Continuity at the shifted LUT boundary (reuse the existing horizon-column scans).
- Existing atmosphere contracts pass with refraction off, and pass within their tolerances with it on, except tests that hard-code the geometric horizon. Update those tests, recording the reason for each one.

Cost: the bake rebuild should be ≤ 1.5× the current `atmosphere_luts` times. Composite and per-pixel costs are unchanged.

### Phase 4: eclipse refraction

Status (2026-10-23): **approximate eclipse lighting implemented** in
[the cached analytical design](eclipse_refraction.md), covering surfaces, terrain,
rings and reflected point sources. The detailed Phase 4 specification below is
historical and is not the implementation contract. Direct per-pixel limb/annulus
integration was rejected after the user measured about 1 FPS at 1440p with the
Moon covering half the screen; that implementation has been removed. Do not
reuse that runtime method. The replacement passed a 1440p lookup-cost gate before
its optical bake was connected.

Review note (2026-10-22): [the original implementation assessment](eclipse_refraction_assessment.md)
identifies required corrections to the provisional design below, especially the
outside-penumbra gate, finite-source geometry, caustic integration and shared
source-law normalization. Resolve these before implementing this phase.

**4a. Limb profile** (`limb_profile.comp`, per atmosphere revision, 512 samples over `p ∈ [p_lo, r_top]`, asinh-concentrated near `p_lo`).
- Stores δ_rgb(p) (dispersion), and T_rgb(p) along the full bent path in → periapsis → out, using the shared medium evaluator: Rayleigh, ozone, mixed absorption, aerosols.
- Rays below the opaque radius or cloud deck are blocked.

**4b. Eclipse LUT** (`eclipse_bake.comp`, only for atmospheres that can shadow something).
- **Axes:**
  - `ρ̂ = ρ/r_top` (128 samples), the distance from the shadow axis.
  - `ℓ = L·δ(p_lo)/r_top` (64 samples, log), the focal parameter; `ℓ ≈ 1` is the focus.
  - `â = α·L/r_top` (8 samples, log), the projected solar radius; `â > 1` is the annular regime. This is the axis from decision 4.
- **Storage:** RGBA16F `sampler2DArray`, one layer per (atmosphere slot, `â` sample). Interpolate `â` manually (two fetches). That is 512 KiB per atmosphere; 16 slots come to 8 MiB.
- **Integrand:**
  - Integrate over the solar disc with 256 spiral samples and a fixed quadratic limb-darkening law. The table is shared across stars; document this.
  - For each source sample, sum over all roots of `p − L·δ(p) = ±ρ_s`. The minus sign is the far-side image, which supplies the on-axis ring.
  - Each root contributes `T(p)·(p/ρ_s)·|dp/dρ_s|`.
- **Coverage:** includes the nearly transparent upper layers, so the runtime value means "everything arriving through the annulus".

**4c. Runtime.**
- Per-body records: `surface_flags.y` gets the eclipse layer (or `~0u`), `surface_flags.z` gets refraction flags. Upload `r_top` in a small per-body SSBO. No change to the `BodyInstanceGPU` layout.
- `evaluate_eclipse_shadow` returns `vec3`. For atmospheric casters it returns `geometric_visibility(r_top ellipse) + refracted(ρ̂, ℓ, â)`, with `ρ` normalized by the existing `oblate_caster_radius`. Other casters are unchanged.
- This is gated: the fetch happens only inside the `r_top` penumbra. Outside it the result is exactly 1, because bent rays always bend inward.
- Convert to `vec3` and wire up:
  - `sphere.frag` (shadow and all `> 1e-4` tests → `max3`).
  - `ring.frag` and `common/ring_shadow_filter.glsl` where they apply eclipse visibility.
  - Stellar point visibility in `point_celestial.vert`.
- Candidate bounds: `eclipse_could_shadow` and the ring/point paths use `r_top` for casters with an eclipse layer.

Contracts (`--eclipse-refraction`):

| Check | Reference / expectation |
|---|---|
| Flux conservation | The integral of refracted irradiance over the shadow plane equals the annulus integral of `T(p)`, within 1%. |
| Point-source caustic | With `â → 0`, the bake matches the CPU tracer and analytic magnification. |
| Earth–Moon totality, clear atmosphere | Umbra centre within roughly 1e-4 to 1e-3 of full light, ordered R > G > B. (A clear sky with no clouds sits at the bright end of the Danjon scale. Record the value rather than fitting to it.) |
| Ozone fringe | At the umbra edge, the B/G ratio exceeds its value at the centre. |
| Cloud deck or opaque aerosol | Raising the cloud deck darkens the umbra monotonically. |
| Off switch, or casters without atmospheres | Pixel-exact match with the current `--eclipse-lighting` and `--eclipse-optics` results. |

Cost: ≤ +0.03 ms in the screen-filling penumbra fixture compared with the current overlap path.

### Phase 5: stellar limb images

- Applies to cameras outside the host atmosphere. The composite adds the star's **secondary** images at limb-annulus pixels where the depth is background or the host's opaque surface beyond the limb.
- For each pixel:
  - compute the straight impact `p` from the pixel direction;
  - read δ_rgb and T_rgb from the limb profile;
  - work out the exit direction on the far side;
  - test it against each star's disc with the same limb law as the existing stellar radiance;
  - add `radiance·T`.
- The primary image is already drawn by the per-vertex refraction of the star mesh.
- **CPU gate:** only enter the per-pixel path when a star lies within `δ(p_lo) + α` of the host disc.

Result: the red ring around Earth seen from the Moon during totality, the thin solar sliver at the limb from orbit, and the green/blue edge from dispersion.

Contracts: ring brightness and colour against a tracer-rendered reference image at 1/4 resolution. No change when the gate is off.

### Phase 6: polish and follow-ups

- Limb-band subdivision for orbit lines, if previews show kinks. Do it with a compute-emitted segment list, not a geometry shader. Rings already use per-fragment ray intersections.
- Refracted RGB fill in the external atmosphere-shadow deficit (`atmosphere_shadow.glsl`), for example Saturn's shadow on Titan's atmosphere.
- `--refraction-preview scratch/refraction` exports: Earth sunset, a Venus horizon (bowl effect), stars occulted at Jupiter's limb (defocusing flicker), a lunar eclipse from the Moon, and an orbital sliver.
- Write `docs/atmospheric_refraction.md`. Update the `project_map.md` routing row and the planned-features list (move both "Atmospheric Refraction" and "Eclipse Refraction" to implemented). Replace the "refraction remains outside the model" line in `docs/atmosphere.md`.

## 5. Render settings

Add to `RenderSettingsStore`:
- `refraction_enabled` (default `true`): positions, bakes and limb images.
- `eclipse_refraction_enabled` (default `true`).

Both are validated whole-settings updates, and the off state must reproduce the pre-feature output exactly.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Observer-table bilinear error where the map is steep (orbital limb compression) | Size the X-axis concentration from the tracer contract; fall back to RG32F if 16F fails the 0.1 px check. |
| Host switch makes refracted stars near the old host's limb pop | 10% hysteresis. The old limb is small on screen when the switch happens. |
| Ducting and multiple images on Venus | Use the primary branch only. The bowl horizon comes out of `p_h`. Document the limits. |
| Sky-LUT horizon split moves every frame at low altitude | `p_h(r_c)` is continuous in `r_c`, and the split already depends on altitude today. |
| Large triangles across the limb | Phase 6 subdivision. Sphere and terrain meshes are dense near the camera already. |

## 7. Open questions

None outstanding. Decisions 5 and 6 in §1.1 resolved the last two.
