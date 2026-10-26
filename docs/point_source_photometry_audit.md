# Point sources and lunar-eclipse brightness audit

2026-10-26. This records the initial investigation and baseline renderer. A subsequent [visibility investigation](point_source_visibility.md) removes the artificial faint-signal fade and evaluates PSF alternatives and Venus photometry. The baseline measurements below precede that change. Saved settings are unchanged.

## Finding

There is no evidence of an orders-of-magnitude error in the inverse-square point-source calculation. At exposure **129.15**, faint stars can nevertheless be practically invisible while the eclipsed Moon appears dark red. The current pixel footprint, projection, tone curve and output precision reproduce this at a wide field of view. A narrow field gives a very different result.

The eclipse model is an approximation for a specified atmosphere, not a calibrated prediction of an observed eclipse. The saved Earth's clear atmosphere can differ substantially from the cloudy, aerosol-bearing limb of the real Earth. The approximation's bending/extinction error alone does not explain an excessively bright Moon: an independent bent-path calculation with the same optical inputs makes central totality moderately brighter.

The reported exposure was 129.150, interpreted as 129.15. The original observation's field of view, framebuffer size, observing position and eclipse geometry were not supplied. The saved navigation preference is approximately 45.26 degrees; this is not proof of the live camera setting.

## Measured point-source behavior

GPU: NVIDIA GeForce RTX 4060 Ti. Current rendering-test target was already up to date. An isolated driver reused its render probe, production shaders and production Gaia loader.

- Doubling observer distance gave a stellar integrated-flux ratio of **0.25**.
- The synthetic Gaia solar analogue and local Sun gave a flux ratio of **1.000**.
- Reflected-body flux near/far was **89.6797 / 22.5681**, consistent with inverse-square observer falloff within rasterization error.
- The reflected resolved-body area sweep was **0.98934–1.00742** of its expected scaling.
- The existing Gaia point-size conservation, culling and compact-footprint checks passed.

The following values are measured after the production composite shader at **1920 × 1080**, exposure **129.15**, intensity **1**, diameter **3 pixels**, no atmospheric attenuation, and a synthetic solar-colored catalog star centered between four pixels. They are the brightest red-channel value multiplied by 255, before integer output rounding. They are not physical luminances or a universal visibility threshold.

| Gaia G magnitude | 45° vertical FOV | 5° vertical FOV | 1° vertical FOV |
|---|---:|---:|---:|
| 0 | 9.00 | 155.51 | 253.88 |
| 3 | 2.38 | 25.34 | 185.65 |
| 6 | 0.67 | 5.37 | 34.71 |
| 8 | 0.29 | 2.26 | 10.88 |

At 45 degrees, a magnitude-6 star is near black on an 8-bit display, despite retaining its expected HDR flux. At 1 degree that same star should be readily visible. Subpixel alignment changes peak intensity; integrated flux is conserved above the deliberate faint-signal fade.

`src/rendering/star_catalog.cpp:215` makes integrated catalog flux proportional to `focal_length_px²`. This agrees with integrating the projected local stellar disc. Going from 45 to 5 degrees multiplies flux by about **90**, and going from 45 to 1 degree multiplies it by about **2253**. Resolved surface brightness does not receive this multiplier. Comparing screenshots or photographs therefore requires matching framing and optical assumptions. The renderer has a fixed pixel PSF and one exposure multiplier; it does not specify a full camera through aperture, sensor, seeing and shutter parameters.

## Code assessment

**Local bodies:** `shaders/celestial/point_celestial.vert` computes mean disc radiance times projected disc area, then divides by the sampled profile's total weight. The emitted-star and reflected-body observer falloffs follow from the projected area. Reflected bodies also include incident inverse-square illumination, material phase and eclipse visibility.

**Gaia:** `src/rendering/star_catalog.cpp:126` converts apparent G magnitude to absolute magnitude, and the cull/draw shaders divide the stored luminosity proxy by distance in AU squared. The parsec/AU and magnitude factors cancel correctly back to apparent flux at the reference observer. `FLUX_SCALE` is a storage scale, not an extra dimming factor. Both paths apply exposure once.

**Deliberate visibility loss:** `shaders/common/point_source.glsl:29` fades sources according to an estimated exposed peak between `1e-6` and `2.5e-6`. `shaders/common/tonemap.glsl` and `shaders/post/composite.frag` then map HDR pixels through the ACES-style curve and a 1/2.2 power. The swapchain uses an 8-bit UNORM format. At 1080p/45 degrees/exposure 129.15, the fade starts affecting approximately G > 8.37 and reaches zero around G = 9.37. The magnitude-6 example is already too dark for useful display without this fade affecting it. Removing the fade alone therefore will not solve that example.

**Remaining calibration issue:** `shaders/common/celestial_scene.glsl:47` applies a limb law whose disc average is **5/6**, without compensating its normalization. Gaia explicitly adopts that same 5/6 factor. Consequently, agreement between Gaia and the local Sun is an internal consistency test, not proof that authored luminosity and illumination obey one physical energy normalization. Holding the existing surface-lighting convention fixed, compensating this particular limb factor would increase stellar flux by **1.2**, about **0.20 magnitude**. It cannot explain a large visibility discrepancy.

**Spectral limits:** Gaia G photometry is combined with an approximate BP−RP-to-temperature-to-RGB color. Local stars use bolometric solar luminosities with RGB colors. These are not a common calibrated spectral measurement for all temperatures. Reflected bodies use illustrative material presets, including an explicitly normalized Hapke response. Further scientific calibration should compare stellar and lunar photometry in the same band or sensor response rather than assume equal RGB scales establish it.

## Eclipse-refraction check

`src/physics/eclipse_shadow.hpp:44` estimates bending from local refractivity/scale height and integrates extinction along a straight tangent chord. The LUT stores positive RGB visibility in logarithmic form. Existing convergence tests compare different resolutions of this same approximation; they do not establish agreement with observations.

An additional CPU probe integrated bending and optical depth along spherical refracted paths using the same prepared atmosphere, then reused the finite solar-disc integration. For conserved ray impact parameter `b = n(r)r`, it used:

```
tau = 2 integral extinction(r) n(r)r / sqrt((n(r)r)^2 - b^2) dr
bend = -2b integral n'(r) / (n(r) sqrt((n(r)r)^2 - b^2)) dr
```

The substitution `r = turning_radius + u²` removes the turning-point singularity. The probe used 1024 radial profile samples and compared 1024 versus 4096 integration steps. Central RGB results converged within 0.03%. This is an independent ray-path check within the project's atmospheric assumptions, not observational validation or a full multiple-scattering reference.

For the **saved Earth atmosphere**, solar angular radius 0.00465 radians, projected solar radius 0.28 Earth radii, and central umbra:

| Method | Red visibility | Green visibility | Blue visibility |
|---|---:|---:|---:|
| Current analytical approximation, dense sampling | 0.00110640 | 0.0000371107 | 0.0000000717284 |
| Bent-path integration, 4096 steps | 0.00132214 | 0.0000457815 | 0.000000123521 |

The reference is **19.5% brighter in red** and **23.4% brighter in green**. Its larger relative blue difference concerns a negligible absolute signal. This approximation error would make the current blood Moon dimmer, not brighter, with those same clear-atmosphere inputs.

A recentered GPU Sun/Earth/Moon fixture used the saved Sun/Moon colors and saved Earth atmosphere, a night-side observer two kilometers above Earth, 45-degree FOV, no surface textures or secondary light, and central totality. At exposure 129.15 its central displayed Moon pixel was approximately **(45.04, 5.52, 0.29) / 255**: visibly dark red. At exposure 1, its integrated red light in eclipse was **0.001028** of the matched clear-Moon red light. That result can coexist with effectively invisible magnitude-6 stars at wide FOV. Very faint blue measurements at exposure 1 underflow the half-float attachment and should not be used to infer physical blue transmission.

The saved Earth prepares with **zero Mie extinction and zero aerosol layers**. The eclipse model also lacks spatial weather/cloud coverage and scattered-light fill. Those omissions matter for comparison with a real event. A sensitivity probe placing a uniform opaque boundary two kilometers higher reduced central direct red visibility from 0.0011064 to 0.0007073. This is a diagnostic, not a proposed global cloud deck or a measured Earth model.

## Interpretation and next implementation target

### Follow-up: PSF and Venus at sunset

Flux conservation alone does not establish realistic visibility. The default point diameter is **3 pixels**, rather than 2.5; 2.5 pixels is the mesh-transition upper **radius**. The Gaussian/tapered profile has a 1.335-pixel continuous FWHM at the default diameter, so the outer footprint should not be interpreted as a uniformly illuminated three-pixel disc.

Direct evaluation of `point_profile` and its sampled normalization gives:

| Diameter | Brightest pixel fraction, source centered on a pixel | Fraction, source centered at a four-pixel corner |
|---|---:|---:|
| 1.5 px | 1.000 | 0.250 |
| 2.5 px | 0.824 | 0.250 |
| 3 px | 0.558 | 0.250 |
| 6 px | 0.140 | 0.116 |

These cases preserve the same total flux before the explicit faint-signal fade. At three pixels, peak pixel intensity is 1.8–4 times lower than putting all the flux into one pixel. Simply shrinking the diameter does not give the same improvement at every subpixel position. A physical sensor records a pixel-area integral of the optical PSF; this implementation samples pixel centers and then renormalizes, making it a discrete approximation rather than a calibrated sensor PSF.

On a bright background the relevant displayed signal is `display(background + point) - display(background)`. A scalar probe using the production ACES/power equations, an exposed source flux of 0.003 and a source centered on a pixel gives a peak increment of **7.33/255 on black**, **2.13/255 on HDR background 0.01**, and **1.06/255 on HDR background 0.1**, all at the default three-pixel diameter. On background 0.1, increasing the diameter to six pixels reduces that increment to **0.27/255**. These are controlled contrast examples, not measured Venus/sunset pixels. Results are saved in `tests/point_psf_contrast_audit.json`.

Real eyes and cameras also have PSFs, and point detection is distinct from resolving a disc. Real visibility depends on collected signal, spatial summation, background, exposure and noise. Spreading light does not automatically imply an incorrect physical model. However, the current fixed pixel PSF, independent faint-signal cutoff and display response have not been validated as either an eye model or a specified camera. The PSF/display path is therefore a credible contributor to the observed dimness and deserves validation independently of aerosol presets.

Venus has another approximation. `src/rendering/atmosphere_renderer.cpp:496` skips an atmosphere whose projected outer radius is below 0.5 pixel. At 1080p/45-degree FOV, Venus's solid radius is approximately **0.030–0.207 pixel** over NASA's listed Earth-distance range; its atmosphere adds only a small fraction to this. Its own atmosphere is consequently omitted at this viewing scale. `point_celestial.vert:55` then uses body RGB and the integrated material phase response; it does not bake the unresolved atmosphere/cloud scattering into the source flux. The preset's light body color may serve as a rough cloud proxy, so simply adding atmospheric light could double-count reflectance. A correct replacement needs a calibrated combined planet/cloud phase curve or integrated radiative-transfer brightness, and a comparison with matched Venus photometry. This code-path limitation does not by itself prove the cause or size of the reported sunset dimness. Earth's atmospheric transport still applies to the resulting source.

Relevant primary sources: [STScI ACS PSFs](https://hst-docs.stsci.edu/acsihb/chapter-5-imaging/5-6-acs-point-spread-functions) documents optical/detector spreading and changes to point-source peak and limiting magnitude; [Strategies for accurate effective point spread function modelling on undersampled images](https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzaf063/8400331) describes pixels as integrals over the instrumental PSF; [NASA's Venus fact sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/venusfact.html) gives its apparent size, Earth-distance range and brightness.

Do not apply a large global point-light multiplier based on this observation alone. First reproduce the exact FOV, pixel dimensions, viewing altitude and eclipse geometry. If the original view was wide, the measured output explains much of the complaint. If it was around one degree, the tested flux predicts visible stars and the next investigation should target the exact scene's atmospheric transport, occlusion and loaded catalog.

For physical calibration, normalize stellar limb darkening consistently with incident luminosity, define a shared spectral/photometric reference, fit the lunar material and atmosphere against measured eclipse data, and specify the intended camera/PSF/display behavior. Cloud and aerosol inputs should represent the desired eclipse conditions. An optional visual star control can remain separate from this calibration.

## Reproduction artifacts

All experimental drivers and reports are under the git-ignored `tests/` directory:

- `point_photometry_audit.cpp`, `point_photometry_audit.json`, `point_photometry_audit.log`
- `eclipse_photometry_cpu_audit.cpp`, `eclipse_photometry_cpu_audit.json`, `eclipse_photometry_cpu_audit.log`
- `build_point_photometry_audit.py`

```
python tests/build_point_photometry_audit.py
build/bin/tests/point_photometry_audit.exe
python tests/build_point_photometry_audit.py eclipse_photometry_cpu_audit
build/bin/tests/eclipse_photometry_cpu_audit.exe
```

## External references

- [NASA lunar photography guide](https://science.nasa.gov/moon/photography-guide/): totality is a thousand or more times dimmer than a full Moon; exposure and optical setup matter.
- [ESO: Eclipsed moon, striking night sky](https://www.eso.org/public/images/potw1119a/): real stars and a red eclipsed Moon coexist in photographs; this does not define a unique camera response or exposure ratio.
- [ESA Gaia science performance](https://www.cosmos.esa.int/web/gaia/science-performance): G is a broad approximately 330–1050 nm passband, rather than a visual or bolometric brightness measurement.
- [Kawauchi et al., 2018](https://arxiv.org/abs/1806.09085): observed lunar-eclipse transmission was consistent with clouds blocking sunlight below approximately 10 km for that observed limb. Real eclipse conditions cannot be identified with a single cloud-free atmosphere.
- [PBRT point lights](https://www.pbr-book.org/4ed/Light_Sources/Point_Lights): radiant intensity propagates with inverse-square distance.
