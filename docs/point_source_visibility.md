# Point-source visibility follow-up

2026-10-26. Production change: remove the artificial faint-source fade from local celestial points and Gaia. Keep the existing PSF, luminosity calibration, tone mapper and authored atmosphere settings.

## What changed

`point_signal_fade` multiplied otherwise valid flux by a smooth threshold based on estimated peak brightness. Because the estimate divided by PSF area, changing point diameter could change total emitted light near the threshold. Sources could also disappear entirely while their HDR signal remained representable.

`shaders/common/point_source.glsl` now provides `point_flux_visible`. It rejects only integrated source flux below `2^-25` in every RGB component. This is half the minimum positive RGBA16F subnormal. Every normalized PSF pixel receives at most the integrated source flux, so a source below this bound cannot change the target attachment under round-to-nearest. Accepted sources retain their full light; fragment blending and attachment precision perform the remaining rounding.

The local point shader applies mesh handoff, geometric visibility and observer refraction before this check. Gaia's compute and vertex stages now both include observer refraction gain before deciding visibility. The earlier compute stage evaluated its threshold before refraction, which could disagree with the vertex stage in magnified views.

This correction restores faint HDR signals. It does not guarantee that an 8-bit display can show every restored source or that a faint source is visible against twilight.

## PSF experiment

Four isolated shader snapshots reused the production Gaia loader and render passes:

1. Baseline radial profile and old faint fade.
2. The same profile without the fade.
3. A pixel-integrated Gaussian with sigma 0.5 pixel at the default diameter.
4. A sharper pixel-integrated Gaussian with sigma 0.3 pixel.

The Gaussian prototypes use analytic pixel-area integrals, bounded support and half-pixel raster padding. They are experiments, not production settings. Their support differs from the existing literal three-pixel footprint.

At 1080p, 45-degree FOV, a solar-colored G=-4 source at exposure 1, centered on a pixel, gave these brightest red-channel values after the production tone mapper:

| Profile | Peak on black, out of 255 |
|---|---:|
| Baseline | 7.42 |
| Same PSF, no fade | 7.42 |
| Pixel-integrated, sigma 0.5 | 6.79 |
| Pixel-integrated, sigma 0.3 | 9.04 |

Pixel integration alone does not brighten the source. Choosing a narrower optical core improves its peak modestly at this alignment. At the intersection of four pixels, the light divides between four central pixels, and narrowing cannot concentrate it into one of them without breaking symmetric pixel integration.

For a G=9 source at exposure 129.15 and the same alignment, removing the fade changes the brightest red pixel from 0.124/255 to 0.274/255. Its measured total red HDR flux rises from 2.38e-7 to 2.32e-6. These signals are still mostly below useful 8-bit display precision. The default PSF is retained because the prototypes do not establish a calibrated optical width or solve the large Venus brightness deficit below.

## Venus photometry

The point shader uses a Lambert phase law for Venus's automatic material. Its own atmosphere is culled below a 0.5-pixel outer radius, so the unresolved source does not include an integrated cloud-scattering phase response. A light body color approximates overall reflectance, but cannot reproduce strong crescent forward scattering.

A controlled Sun/Venus fixture used the saved colors, radius and atmosphere, a circular 0.723332 AU Venus orbit, a geometrically consistent Earth-distance observer, 1080p/45-degree FOV, exposure 1, and no observer atmosphere or secondary illumination. The observer was adjusted for the simulation's barycentric recentering. The source remained unresolved. Its green-channel integrated flux was compared with the modeled solar green disc at 1 AU and assigned the solar V magnitude -26.74. This produces a **visual-magnitude proxy**, not calibrated broadband V photometry.

The reference is the V-band Venus model in [Mallama and Hilton, 2018, equations 3–4](https://arxiv.org/abs/1808.01973). Broadband and RGB differences limit precise numerical agreement, but cannot plausibly account for the large thin-crescent discrepancy.

| Phase angle | Rendered magnitude proxy | Published V reference | Rendered/reference flux proxy |
|---|---:|---:|---:|
| 60° | -4.03 | -4.03 | 1.00 |
| 90° | -4.42 | -4.46 | 0.96 |
| 120° | -4.35 | -4.81 | 0.65 |
| 150° | -2.86 | -4.49 | 0.22 |
| 170° | +0.50 | -4.18 | 0.013 |

The model is roughly 4.5 times too faint at 150 degrees and 75 times too faint at 170 degrees by this proxy. The paper attributes the high-phase brightening to atmospheric forward scattering. A PSF change preserves integrated light and cannot correct this source-flux deficit.

The next Venus correction needs a combined planet/cloud phase response that agrees with the resolved atmosphere through the mesh/point handoff. Adding a brightness multiplier to every point, or adding an empirical Venus term on top of its resolved atmosphere, would not establish that consistency. The empirical curve is useful as a reference; this patch does not introduce a Venus-specific override.

## Actual twilight fixture

A second GPU fixture placed the observer 2 km above Earth with the saved Earth atmosphere, the Sun at -3 degrees altitude and Venus at phase 120 degrees and approximately 35.8 degrees altitude. Refraction, surface textures and secondary illumination were disabled to isolate the shared atmosphere transport and point source. Exposure was 1. The source center lay at a four-pixel intersection.

The central-patch displayed RGB maxima were approximately `(6.45, 5.61, 8.25)/255` without Venus and `(9.02, 8.13, 10.05)/255` with Venus. The sharper Gaussian prototype gave the same maxima at this symmetric alignment. This is one controlled twilight fixture, not the user's original sunset scene, and does not establish a unique visibility threshold.

## Verification

- New faint-source GPU regression fails on the frozen old shader at the intended flux-conservation assertion.
- New catalog and reflected-body tests preserve faint flux across 1.5, 3 and 6 pixel diameters and several subpixel alignments within the attachment rounding bound.
- Existing inverse-square, mesh/point transition, occlusion, Gaia/local-Sun calibration, compact footprint and overexposure contracts pass.
- The targeted GPU suite passed with Vulkan validation enabled and no reported validation errors.
- The application `--self-test` completed 13 frames cleanly, including settings pages.
- The broader `--smooth-surfaces` check fails at `Default surfaces refine smooth ellipsoids without imported or synthetic height`. The same executable and frozen old shaders fail at the identical assertion; this existing terrain-contract failure was not changed by the point-source patch.

A bounded real-catalog microbenchmark used 475,724 Gaia records, a camera looking along render +X, 1920×1080, 45-degree FOV, no atmosphere/occluders, 10 warmup and 30 measured frames per exposure. It measured culling and point drawing with GPU timestamps, excluding image readback. Sequential old/new runs on an RTX 4060 Ti gave:

| Exposure | Old visible instances | New visible instances | Old median cull+draw | New median cull+draw |
|---|---:|---:|---:|---:|
| 1 | 40 | 4,937 | 0.1164 ms | 0.1190 ms |
| 129.15 | 10,857 | 18,900 | 0.1277 ms | 0.1407 ms |

These are sums of the separate pass medians from one fixed-view microbenchmark, not application FPS or a general performance guarantee.

## Reproduction

The experimental drivers and snapshots are under the git-ignored `tests/` directory. `build_point_photometry_audit.py` links a selected driver against the existing rendering-test target; build that target first after C++ changes.

```powershell
cmake --build build --target StellarRenderingTests StellarForge
python tests/build_point_photometry_audit.py point_visibility_contracts
$env:STELLAR_VULKAN_VALIDATION = '1'
.\build\bin\tests\point_visibility_contracts.exe
.\build\bin\StellarForge.exe --self-test
```

Artifacts:

- `tests/point_visibility_contracts.cpp`, `tests/point_visibility_contracts.log`
- `tests/point_visibility_self_test.log`
- `tests/point_psf_experiment.cpp`, `tests/prepare_point_psf_experiment.py`
- `tests/point-psf-experiment/<variant>/measurements.json`, `twilight.json`, PPM previews and logs
- `tests/point_catalog_cost.cpp`, `tests/point_catalog_cost.json`
- `tests/point-psf-experiment/baseline/catalog-cost.json`, `visibility-regression.log`

The frozen baseline and Gaussian snapshots retain the old production shader layout for this comparison. Running the preparation script again replaces those snapshots with the then-current sources; preserve the existing results when comparing this audit.
