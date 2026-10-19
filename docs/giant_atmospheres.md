# Venus, giant atmospheres and shared ephemeris definitions

The Solar System preset is the canonical physical definition for Ephemeris Mode. `data/systems/Ephemeris Mode/system.json` contains only `{"source_preset": "Solar System"}`. The mode builder always loads the Solar System template and replaces motion with available SPICE states. Kernel discovery and additional spacecraft/moons remain supported. Loading the alias returns the shared template; `SpiceKernelManager::build_ephemeris_system` selects SPICE motion and filters unavailable states. Saving physical properties from Ephemeris Mode targets Solar System definitions while retaining the source preset's orbital elements, parents, epoch and motion model. There is no independent atmosphere, mass, radius, color or ring database for Ephemeris Mode.

## Pressure datum and lower boundary

For giants, `reference_pressure_pa: 100000` denotes pressure at the body's defined radius, altitude zero. `temperature_k` is the temperature at that datum. `bottom_altitude_km` is a modeled opaque lower boundary, not a solid planetary surface. Standard terrestrial records keep `pressure_pa` at their surface datum. Supplying both pressure keys is rejected.

Temperature knots may extend below zero and must include `[0, temperature_k]`. Hydrostatic integration follows all knots in either direction, with spherical variable gravity. Prepared density profiles include the modeled lower boundary. Clouds may have negative-altitude peaks and their 550 nm columns are normalized from that boundary upward. Resolved pressure-defined bodies use smooth quadtree geometry at their opaque cloud boundary, with the existing body-fixed texture maps; imported elevation and synthetic relief do not displace that boundary. The spherical hydrostatic and homothetically flattened rendering approximation remains.

`cloud_deck_pressure_pa` optionally places the opaque visible cloud boundary above the modeled lower boundary. Its altitude is derived from the gas/thermal profile, rather than stored independently. The bundled Jupiter/Saturn/Uranus/Neptune boundaries are representative effective cloud tops at 0.25/0.12/0.5/0.5 bar. Their existing maps/base reflectance represent that cloud boundary, with clear molecular gas above it. No extra volumetric cloud or haze populations are present in these presets. Rendering, atmospheric shadows, surface irradiance and camera limits share the derived boundary. These effective pressures are illustrative: real cloud tops vary with wavelength, latitude and weather. Imported maps already contain some atmospheric color and are not retrieved, atmosphere-free cloud reflectances.

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

The uppermost population uses `sulfate_with_uv_absorber`, an effective droplet mixture with real index 1.44 and illustrative absorptive indices 1e-8/5e-4/5e-3 at 680/550/440 nm. Mie scattering and absorption produce the warm cast; this is not an RGB tint. The deeper droplets retain transparent sulfate optics. This is a phenomenological blue-absorption proxy, not a measured optical-constant table or a claim that the absorber's identity is known. [NASA's CUVE description](https://www.nasa.gov/technology/nasa-studies-cubesat-mission-to-solve-venusian-mystery/) describes Venus as yellowish white and the upper-cloud absorber as unidentified. The RGB model does not model ultraviolet radiation.

The bundled presets retain particle populations only for Venus, Titan and Mars. Earth and the four giants use explicit empty `aerosols` arrays; Earth's separate manual Mie haze is also removed. Molecular scattering, methane absorption, thermal profiles and Earth's ozone remain active. This clear-gas baseline avoids imposing unvalidated homogeneous cloud/haze columns over existing giant cloud maps. It does not assert that the real planets lack aerosols. `scripts/update_giant_atmospheres.py` retains empty giant particle arrays when regenerating the reference profiles.

Optional custom particle materials remain supported. Representative real refractive indices are 1.44 for crystalline ammonia ice, 1.30 for methane ice, 1.70 for ammonium hydrosulfide and 1.55 for hydrogen-sulfide ice. The latter two are provisional material proxies rather than measured visible spectral tables. Absorptive parts are 1e-8 for the nearly transparent ices and 1e-5 for ammonium hydrosulfide. Crystalline ammonia's measured visible index is reported by [Hudson et al. (2022), Table 1](https://science.gsfc.nasa.gov/691/cosmicice/reprints/Hudson-2022-2895.pdf).

Cross sections retain the homogeneous-sphere approximation. Ice/condensate phase functions use normalized Henyey–Greenstein tables fitted to the Mie first angular moment independently per channel; this smooth approximation avoids imposing spherical-droplet rainbow structure on irregular ice crystals. Water and sulfate droplets retain the full Mie phase tables. Neither approach is a quantitative nonspherical or aggregate-particle solver.

## Methane and transport

`data/optics/ch4_karkoschka_1994.txt` archives the cold-methane absorption cross sections from the [MPI-Mainz atlas data file](https://www.uv-vis-spectral-atlas-mainz.org/uvvis_data/cross_sections/Alkanes%2Balkyl%20radicals/Alkanes/CH4_Karkoschka%281994%29_125-128K_406-1000nm.txt), attributed to [Karkoschka (1994)](https://doi.org/10.1006/icar.1994.1139). Atlas reference: [Keller-Rudek et al. (2013)](https://doi.org/10.5194/essd-5-365-2013). Original units are cm²/molecule. `scripts/prepare_methane_optics.py` computes equally weighted 1 nm means for 600–699, 500–599 and 406–499 nm and converts to m²/molecule. The RGB means are `3.50986e-30`, `6.70107e-31`, `8.25387234042553e-32`. The 125–128 K data are applied without a temperature/pressure-dependent spectroscopic model. Mean absorption before exponentiation is still a three-band approximation, not spectral or correlated-k transfer.

Gaussian clouds trigger bounded extra quadrature cells where a ray interval crosses a cloud population. This refinement is shared by optical depth, multiple scattering, sky, surface irradiance and both aerial paths. The multiple-scattering table uses a signed asinh altitude coordinate for physical profiles to resolve clouds near the reference radius. Very thin layers can still exceed the bounded quadrature budget. Multiple scattering retains the isotropic closure, and the medium remains horizontally homogeneous. Mapped giant cloud bands are a reflective boundary, not a simulated three-dimensional cloud pattern. Weather, latitude-dependent cloud fields, measured chromophore optics and anisotropic dense-cloud multiple scattering remain future work.

## Verification

- State tests check shared physical definitions and live SPICE construction/playback when local kernels are enabled, clear-gas presets outside Venus/Titan/Mars, Venus cloud-level pressure and selective absorption, signed giant density interpolation, approximate 10-bar boundaries, cloud-top pressure inversion and geometry revisions, and persistence. Cloud tops beyond a fixed atmospheric ceiling are rejected before mutation.
- GPU atmosphere tests compare finite transport below the one-bar datum with independent molecular integration and analytic Gaussian columns in both aerial models, check opaque fallback geometry, render/cache all five worlds, and verify that clear gas above each giant preserves cloud-deck reflectance contrast. The preview command additionally exports streamed cloud-map diagnostics.
- `StellarRenderingTests --giant-atmosphere-preview scratch/giant-atmospheres` writes matched-irradiance orbital diagnostics and HDR samples. These are diagnostic views, not calibrated fits to spacecraft images.
- `ephemeris_contracts` builds the shared preset at the oracle epoch and checks independent SPICE state fixtures.
