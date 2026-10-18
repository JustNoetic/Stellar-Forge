# Physical atmospheres

`AtmosphereDefinition::physical` selects a physical gas column. A complete atmosphere record containing pressure, temperature and composition automatically selects physical mode and enables the atmosphere when those switches are absent. Both modern `pressure_pa`/`temperature_k` and original `surface_pressure` (atm)/`temperature` fields are recognized. Explicit `enabled: false` and an explicit model remain authoritative; absent or incomplete atmosphere records remain disabled.

The inspector exposes pressure, temperature, gas percentages and independent particle layers. Changing a gas percentage balances the remaining gases to keep 100%. Molecular properties and particle optics are calculated from their respective inputs. Gas edits preserve authored particles and manual aerosol ownership. Apply/Revert/Save retain atomic edits. Imported temperature layers, ozone and manual optics remain supported in JSON.

## Inputs and ownership

Pressure is Pa and temperature is K at altitude zero on the reference ellipsoid. Giant records use `reference_pressure_pa` and a negative modeled lower boundary; see [giant atmospheres](giant_atmospheres.md). Composition contains nonnegative relative mole weights for N2, O2, Ar, CO2, CH4, H2, He, H2O, NH3, O3 and SO2. Weights are normalized for calculations; unknown gases, empty mixtures and invalid layers are rejected. Temperature knots use increasing geometric kilometres, include `[0, temperature_k]`, may extend below zero, interpolate linearly, and remain constant outside the supplied range. Empty knots select an isothermal column. Pressure may be zero; an enabled physical atmosphere requires positive body mass.

The present EOS is ideal gas. Reference temperature is supplied, not recomputed from stellar heating. Planet mass and reference radius supply spherical gravity. Variable gravity is `g(h) = GM/(R+h)^2`. Rotation, J2 and the non-spherical geopotential are not included in the hydrostatic solve. Rendering retains the existing homothetically flattened transport coordinates, whose kilometre error is of order oblateness.

`BodyRenderState::atmosphere` owns a shared immutable `PreparedAtmosphere`. Simulation prepares it on system load or relevant definition edits. Unchanged inputs, playback and camera movement reuse its identity; older snapshots retain their original profile. Mass/radius changes rebuild it. The solver runs neither in the frame renderer nor in UI inspection. GPU tables still belong to the renderer.

## Hydrostatic and optical preparation

Version-2 physical records store independent particles: material, 550 nm optical depth, effective radius/variance and exponential or Gaussian geometry. Homogeneous-sphere Mie preparation derives RGB extinction, absorption and angular scattering. Version-1 automatic records migrate once to explicit estimates; physical preparation no longer chooses haze from gas thresholds. Manual optics retain their coefficients and inverse-metre `beta_mie` migration. See [particle schema, migration, validation and limitations](atmosphere_particles.md).

The model solves `d ln P/dh = -M g(h)/(R_gas T(h))`, then evaluates number density `P/(k_B T)` and mass density `P M/(R_gas T)`. Isothermal columns use the exact spherical solution. Layered columns integrate each continuous temperature interval with composite Simpson quadrature; sparse, repeated and unordered point queries do not depend on which layers contain query points.

Rendering consumes 256 samples of log number density, spaced uniformly in `asinh(h / scale)` from a signed lower domain to the optical top. Interpolating log density preserves exponential columns and retains supplied temperature layers. Below-zero molecular gas follows the same hydrostatic column; aerosols retain their bounded constant continuation below the datum. This represents air-filled negative terrain, not water inside ocean basins.

Species Rayleigh cross sections use molecular refractivities, species King factors and a 273.15 K/101325 Pa reference number density. Well-mixed visible absorption follows molecular number density. Fixed coefficients at 680/550/440 nm remain an RGB approximation; CH4 uses archived cold-methane band means; SO2 remains an inherited approximation. This is not a full spectroscopic database. H2O/NH3 visible absorption is currently zero. There is no infrared or climate solver.

An optional ozone layer specifies `column_du`, `peak_km`, and Gaussian standard deviation `width_km`. Its peak density is normalized by the Gaussian integral from altitude zero to infinity, with `1 DU = 2.687e20 molecules/m2`. Well-mixed O3 and the supplied layer are additive. Molecular oxygen does not automatically produce ozone. Ozone cross sections use the 680/550/440 nm bins in Bruneton's documented 233 K table: `1.209e-25`, `3.5e-25`, `1.582e-26 m2`.

Automatic optical height extends past temperature knots, six ozone standard deviations and eight detached-particle standard deviations, then checks molecular/particle tangent-tail estimates against `1e-5` optical depth, capped at 10000 km. This is a bounded rendering cutoff, not an exobase or escape model. Manual height remains available. Very weakly bound or strongly non-isothermal columns need another model. Pressure zero suppresses all gas, ozone and particles.

All transmittance, multiple-scattering, sky, aerial and surface-irradiance bakes use the same GPU medium evaluator, including absorption in extinction but excluding it from scattering sources. Both aerial models retain finite-distance endpoints.

## Surface light and composition

Physical atmospheres prepare one 128x32 solar-angle/altitude table of direct RGB transmission and one of hemispherical diffuse irradiance. They share a 64-layer RGBA16F atlas (2 MiB), covering up to 32 simultaneously prepared atmospheres. Per-body records cost 16 KiB per in-flight frame. Each surface fragment uses two filtered table lookups per star when its host has a prepared physical atmosphere. Bodies beyond the bounded surface cache retain their existing direct illumination. The sky/aerial atmosphere count is unchanged.

Diffuse irradiance integrates 16 cosine-distributed directions with 32 transport cells each, including the existing multiple-scattering closure. The result supplies hemispherical sky light with an approximate slope factor, not directional SH skylight. Sunlight attenuates before view-path aerial composition; diffuse sky light persists through twilight without an artificial night floor. External body/ring shadows affect the existing direct light path but are not baked into diffuse irradiance. Both sphere and terrain materials consume the shared table. Manual optical atmospheres retain their previous surface lighting.

Vulkan dual-source composition writes scattered radiance to color output index 0 and RGB transmission to index 1. Blend factors `ONE, SRC1_COLOR` evaluate `S + T * background`, preserving destination alpha. The device feature and attachment limit are checked, enabled when available, and exposed through the shared RHI pipeline description. A render setting selects it; unsupported devices use compute. Compute remains the default because the measured graphics path is slower on the RTX 4060 Ti. Both paths share `common/atmosphere_composite.glsl`.

Physical preparation, density arrays and surface-table bakes happen on input edits. Sun motion reuses optical and surface tables, while sky-view updates follow the existing lighting key. Camera rotation reuses sky and Bruneton tables; frustum aerial volumes still update when their view changes. Profile sampling adds work to medium evaluation during bakes and the short-segment aerial fallback. Surface lookups and composition execute each frame. Performance is therefore not literally free outside rebuilds.

## Validation and references

The existing frustum mesh-curvature comparison reproduced a maximum absolute HDR error of `0.00219727` with the original medium and surface shaders and forced compute composition. The new paths reproduce the same error. That low-resolution angular reconstruction contract uses a `0.0025` absolute tolerance, including FP16 quantization, instead of its previously failing `0.0015` threshold; its dense transport and exact-depth contracts remain separate.

`StellarRenderingTests --atmosphere-physics-benchmark scratch/atmosphere-physics/costs.json` compares manual/physical media, both compositors, cached frames, translation and input edits at 1920x1080. Two repetitions use 6 warmup plus 16 measured frames each. Serialized timer results exclude readback and presentation and are not application frame times. GPU clock variation is visible across different workloads.

The following timings are the historical pre-particle baseline; they do not measure the new particle populations. On the RTX 4060 Ti, cached physical surface lighting and composition added approximately 0.015-0.041 ms over the corresponding manual/compute cases. Camera translation reused the physics profile and surface irradiance; ground sky preparation increased from approximately 0.045 to 0.054 ms. Physical CPU input edits measured approximately 0.044-0.101 ms, compared with 0.003-0.004 ms for manual edits. Complete physical GPU LUT updates measured approximately 1.96 ms versus 1.28-1.32 ms for manual updates. These cases have different physical media and optical heights, so they compare workload cost rather than isolate individual shader instructions.

Dual-source composition measured approximately 0.167 ms versus compute's 0.154 ms for the physical ground cases. In the physical orbital cached case it measured 0.367 ms versus 0.291 ms; higher-clock orbital input-edit cases measured 0.091 ms versus 0.072 ms. This supports keeping compute as the default on this device while retaining dual-source composition as a tested option.

`StellarStateTests` covers complete/incomplete JSON activation, explicit disabled states, original aerosol-unit migration, percentage balancing, automatic methane haze and vacuum, standard-atmosphere pressures, exact spherical isothermal pressure, signed prepared-density interpolation, ozone-column conservation, vacuum, invalid mixtures/pressure, atomic edits, immutable profile reuse and physical persistence. `StellarRenderingTests --atmospheres-only` adds independent numerical physical transmission, direct surface attenuation, twilight irradiance, vacuum, cache reuse, and pixel comparisons between graphics and compute composition in both aerial models. `STELLAR_VULKAN_VALIDATION=1` requests Vulkan validation; `STELLAR_TEST_ATMOSPHERE_COMPUTE=1` forces the compute compositor in the rendering probe for diagnostics.

- [U.S. Standard Atmosphere 1976](https://ntrs.nasa.gov/citations/19770009539)
- [Bruneton scattering implementation and spectral reference data](https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/demo/demo.cc.html)
- [Bruneton custom density profiles and reference tests](https://ebruneton.github.io/precomputed_atmospheric_scattering/)
- [NASA ozone columns and Dobson units](https://ozonewatch.gsfc.nasa.gov/facts/dobson_NH.html)
- [Vulkan dual-source blending](https://github.khronos.org/Vulkan-Site/spec/latest/chapters/framebuffer.html)
