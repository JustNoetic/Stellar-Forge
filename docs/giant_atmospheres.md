# Venus, giant atmospheres and shared ephemeris definitions

The Solar System preset is the canonical physical definition for Ephemeris Mode. `data/systems/Ephemeris Mode/system.json` contains only `{"source_preset": "Solar System"}`. The mode builder always loads the Solar System template and replaces motion with available SPICE states. Kernel discovery and additional spacecraft/moons remain supported. Loading the alias returns the shared template; `SpiceKernelManager::build_ephemeris_system` selects SPICE motion and filters unavailable states. Saving physical properties from Ephemeris Mode targets Solar System definitions while retaining the source preset's orbital elements, parents, epoch and motion model. There is no independent atmosphere, mass, radius, color or ring database for Ephemeris Mode.

## Pressure datum and lower boundary

For giants, `reference_pressure_pa: 100000` denotes pressure at the body's defined radius, altitude zero. `temperature_k` is the temperature at that datum. `bottom_altitude_km` is a modeled opaque lower boundary, not a solid planetary surface. Standard terrestrial records keep `pressure_pa` at their surface datum. Supplying both pressure keys is rejected.

Temperature knots may extend below zero and must include `[0, temperature_k]`. Hydrostatic integration follows all knots in either direction, with spherical variable gravity. Prepared density profiles include the modeled lower boundary. Clouds may have negative-altitude peaks and their 550 nm columns are normalized from that boundary upward. Giant meshes use the lower boundary as an opaque fallback; solid terrain at the one-bar radius is disabled. Molecular density, atmospheric shadow geometry, surface irradiance and camera limits use the corresponding reference or lower-boundary radius. The spherical hydrostatic and homothetically flattened rendering approximation remains.

The giant profiles extend to approximately 10 bar below the reference radius. Their pressure-temperature reference points are:

| Pressure (bar) | Jupiter K | Saturn K | Uranus K | Neptune K |
| --- | --- | --- | --- | --- |
| 10 | 330 | 270 | 150 | 150 |
| 3 | 220 | 180 | 105 | 100 |
| 1 | 165 | 134 | 76 | 72 |
| 0.5 | 135 | 110 | 64 | 62 |
| 0.1 | 110 | 82 | 53 | 55 |
| 0.01 | 150 | 120 | 70 | 90 |
| 0.001 | 165 | 140 | 95 | 130 |
| 0.00001 | 170 | 150 | 110 | 150 |

These are representative static approximations, not a digitized occultation profile or a climate solution. `scripts/update_giant_atmospheres.py` converts them into altitude knots using hydrostatic integration of temperature linear in log pressure, then subdivides the pressure intervals to retain accuracy in the engine's altitude interpolation. Altitudes are rounded to 0.01 km. The source reference radii and gas mole fractions are retained.

## Clouds and temperature

Venus has a surface-to-upper-atmosphere temperature profile: 737 K at the surface, 340 K at 50 km, 260 K at 60 km, and 180 K at 100 km. Three sulfuric-acid proxy populations peak at 53, 60 and 65 km, replacing haze extending from the ground. Their particle radii are 3, 1 and 0.2 um; illustrative 550 nm columns are 10, 15 and 1. These are static reference populations, not a retrieval fit. [NASA's Venus atmospheric environment report](https://ntrs.nasa.gov/api/citations/20120013601/downloads/20120013601.pdf) and the [Venus cloud review](https://doi.org/10.1007/s11214-018-0552-z) constrain the broad temperature and cloud domains.

Jupiter/Saturn use ammonia-ice upper clouds, deeper ammonium-hydrosulfide clouds and a photochemical-haze proxy. Saturn also includes a separate upper-tropospheric chromophore proxy at 0.25 bar (column 1.5), below its thin stratospheric haze at 0.03 bar (column 0.1). The two haze domains are motivated by [Cassini/HST multilayer studies](https://www.nature.com/articles/s41467-020-16110-1); these particular material assignments and columns are illustrative, not measured retrieval values. Uranus/Neptune use methane-ice clouds, deeper hydrogen-sulfide clouds and a haze proxy. Cloud peaks are placed by pressure, then stored as signed-altitude Gaussian layers. Approximate condensate ordering is described in the [giant atmosphere/cloud review](https://doi.org/10.1007/s00159-023-00150-9). Amounts and widths remain illustrative; unknown chromophores are not chemically identified by this engine.

Additional representative real refractive indices are 1.44 for crystalline ammonia ice, 1.30 for methane ice, 1.70 for ammonium hydrosulfide and 1.55 for hydrogen-sulfide ice. The latter two are provisional material proxies rather than measured visible spectral tables. Absorptive parts are 1e-8 for the nearly transparent ices and 1e-5 for ammonium hydrosulfide. Crystalline ammonia's measured visible index is reported by [Hudson et al. (2022), Table 1](https://science.gsfc.nasa.gov/691/cosmicice/reprints/Hudson-2022-2895.pdf).

Cross sections retain the homogeneous-sphere approximation. Ice/condensate phase functions use normalized Henyey–Greenstein tables fitted to the Mie first angular moment independently per channel; this smooth approximation avoids imposing spherical-droplet rainbow structure on irregular ice crystals. Water and sulfate droplets retain the full Mie phase tables. Neither approach is a quantitative nonspherical or aggregate-particle solver.

## Methane and transport

`data/optics/ch4_karkoschka_1994.txt` archives the cold-methane absorption cross sections from the [MPI-Mainz atlas data file](https://www.uv-vis-spectral-atlas-mainz.org/uvvis_data/cross_sections/Alkanes%2Balkyl%20radicals/Alkanes/CH4_Karkoschka%281994%29_125-128K_406-1000nm.txt), attributed to [Karkoschka (1994)](https://doi.org/10.1006/icar.1994.1139). Atlas reference: [Keller-Rudek et al. (2013)](https://doi.org/10.5194/essd-5-365-2013). Original units are cm²/molecule. `scripts/prepare_methane_optics.py` computes equally weighted 1 nm means for 600–699, 500–599 and 406–499 nm and converts to m²/molecule. The RGB means are `3.50986e-30`, `6.70107e-31`, `8.25387234042553e-32`. The 125–128 K data are applied without a temperature/pressure-dependent spectroscopic model. Mean absorption before exponentiation is still a three-band approximation, not spectral or correlated-k transfer.

Gaussian clouds trigger bounded extra quadrature cells where a ray interval crosses a cloud population. This refinement is shared by optical depth, multiple scattering, sky, surface irradiance and both aerial paths. The multiple-scattering table uses a signed asinh altitude coordinate for physical profiles to resolve clouds near the reference radius. Very thin layers can still exceed the bounded quadrature budget. Multiple scattering retains the isotropic closure, and the medium remains horizontally homogeneous. Existing texture maps are an opaque fallback below the modeled atmosphere, not a simulated three-dimensional cloud pattern; opaque uniform clouds may conceal their bands. Weather, latitude-dependent cloud fields, accurate chromophores and anisotropic dense-cloud multiple scattering remain future work.

## Verification

- State tests check shared physical definitions and live SPICE construction/playback when local kernels are enabled, Venus cloud-level pressure, signed giant density interpolation, approximate 10-bar boundaries, column conservation and persistence.
- GPU atmosphere tests compare finite transport below the one-bar datum with independent molecular integration and analytic Gaussian columns in both aerial models, check opaque fallback geometry, and render/cache all five worlds.
- `StellarRenderingTests --giant-atmosphere-preview scratch/giant-atmospheres` writes matched-irradiance orbital diagnostics and HDR samples. These are diagnostic views, not calibrated fits to spacecraft images.
- `ephemeris_contracts` builds the shared preset at the oracle epoch and checks independent SPICE state fixtures.
