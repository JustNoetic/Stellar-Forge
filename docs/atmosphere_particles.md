# Particle atmospheres (schema version 2)

Physical records store pressure (Pa), temperature (K), gas mole weights, optional temperature/ozone profiles and independent particle columns. An empty `aerosols` array means clear gas. Gas edits preserve authored particles. Zero pressure disables all atmosphere rendering, retaining the existing vacuum convention.

Bundled Solar System particle populations are retained only for Venus, Titan and Mars. Earth and the four giants use clear molecular gas; Earth's manual Mie haze is removed. Giant cloud textures remain the visible reflective boundary. Ephemeris Mode shares these definitions. Custom particle layers remain available in the inspector.

```json
{
  "schema_version": 2,
  "enabled": true,
  "model": "physical",
  "pressure_pa": 636,
  "temperature_k": 214,
  "composition": {"CO2": 0.957, "N2": 0.027, "Ar": 0.016},
  "aerosols": [{
    "material": "mineral_dust",
    "optical_depth_550": 0.48,
    "effective_radius_um": 1.5,
    "effective_variance": 0.3,
    "scale_height_km": 12
  }]
}
```

Up to four populations have independent materials, sizes, columns, profiles and angular scattering. Exponential profiles use `scale_height_km`. Detached Gaussians replace it with `peak_km` and `width_km` (standard deviation). Optical depth is vertical extinction at 550 nm above the modeled lower boundary (the surface for terrestrial bodies). Each profile integrates to that column. Giant pressure-reference records may contain signed cloud peaks; see [giant atmospheres](giant_atmospheres.md). Exponential particles remain constant below the datum; Gaussians continue their profile.

## Optics and approximations

For a lognormal number distribution, `sigma² = ln(1 + effective_variance)` and `ln(r_m) = ln(effective_radius_um) - 2.5 sigma²`. Preparation integrates 48 log-radius bins over ±4.5 sigma. Homogeneous-sphere Mie coefficients use downward logarithmic-derivative and Riccati–Bessel recurrences. Cross sections determine extinction/scattering; angular amplitudes determine the unpolarized phase function. Cross sections use the sphere approximation. Ice and condensate angular tables use a smooth phase fitted to the Mie first moment; see [giant atmospheres](giant_atmospheres.md).

Representative refractive indices at 680/550/440 nm:

| Material | Real part | Absorptive part, red/green/blue |
| --- | --- | --- |
| Mineral dust | 1.50 | 0.001 / 0.003 / 0.012 |
| Organic haze | 1.65 | 0.004 / 0.020 / 0.060 |
| Water | 1.333 | 1e-8 in each channel |
| Ice | 1.31 | 1e-8 in each channel |
| Sulfate | 1.44 | 1e-8 in each channel |
| Sulfate with UV absorber (effective proxy) | 1.44 | 1e-8 / 5e-4 / 5e-3 |

These are approximate samples, not digitized retrieval tables or exact measurements at these wavelengths. Tholin samples reflect the visible range in [Khare et al. (1984)](https://hitran.org/data/Aerosols/Aerosols%20data%20from%20previous%20HITRAN/Aerosols-2016/papers/khare_tholins_1984.pdf); dust properties vary ([Wolff et al. 2009](https://doi.org/10.1029/2009JE003350)). Martian nonspherical dust, Titan aggregates and ice facets need measured phase functions or other solvers. Plain sulfate excludes Venus's unidentified blue/UV absorber; the separate effective mixture approximates its visible blue absorption without identifying its chemistry. See [giant atmospheres](giant_atmospheres.md). No RGB sky color is prescribed.

Each population has 128 phase samples at `theta = pi * t²`, concentrated near forward scattering. Linear interpolation in cosine is normalized to unit solid-angle integral per channel. All atmosphere passes use these densities, extinction and angular sources. Absorption contributes only to extinction. The uniform block is 13,328 bytes, below Vulkan's guaranteed 16 KiB minimum. Multiple scattering retains the isotropic closure.

The horizon-mapped aerial LUT integrates every population's RGB phase directly at each angular column direction. Bicubic reconstruction filters the resulting cumulative radiance; very narrow forward lobes remain limited by angular resolution. There is no first-population phase factorization or eight-bin residual interpolation.

Bundled Mars has a representative dust column of 0.48 and 1.5 um effective radius. Titan has a main organic column plus small particles at 500 km. These are illustrative static profiles, not observational fits. Ephemeris Mode shares all physical definitions directly from Solar System. Fixed quadrature can underresolve thin layers and grazing paths. Optical height includes particle tails and detached layers, retaining the 10,000 km cap.

## Migration and tests

`python scripts/migrate_atmospheres.py PATH...` edits atmosphere spans, canonicalizes units, removes obsolete hints and consolidates duplicate atmosphere objects. Later duplicate fields win; earlier nonconflicting conditions survive. `--solar-presets` explicitly repairs legacy Mars/Titan columns. Other custom manual coefficients, absorption, slopes and heights survive.

The loader accepts version-1 records. Legacy automatic haze migrates once to explicit estimates; version-2 does not infer particles from gases. Manual `beta_mie` retains inverse-metre conversion. Saves write version 2 and omit derived molecular coefficients and automatic heights. Invalid layers and future versions are rejected. The inspector preserves particle/manual ownership during gas edits; replacement of manual optics is explicit.

- `StellarStateTests`: published Mie benchmarks, Rayleigh limit, energy and column conservation, spectral angular response, particle invariance, persistence, invalid inputs, UI ownership and bundled presets.
- `atmosphere_migration_contracts`: manual optics, repair scope, idempotence, duplicate consolidation and unrelated-data preservation.
- `StellarRenderingTests --particles-only`: analytic exponential/Gaussian extinction at finite endpoints in the horizon-mapped aerial LUT, compositor agreement and cache reuse.
- `StellarRenderingTests --atmospheres-only`: particle plus existing physical/legacy atmosphere contracts.
- `StellarRenderingTests --particle-atmosphere-preview scratch/particle-atmospheres`: matched-irradiance Mars/Titan day, sunset and orbit PPM diagnostics and linear HDR samples. Images use linear clamped color, not calibrated photographic color.

Equations and independent benchmarks: [miepython algorithm notes](https://miepython.readthedocs.io/en/latest/07_algorithm.html). Numerical tests validate sphere optics and transport, not observational accuracy. Multiband radiance, measured material tables, aggregates, anisotropic multiple scattering, cloud microphysics, climate, weather, photochemistry, atmospheric external shadows and refraction remain future work.
