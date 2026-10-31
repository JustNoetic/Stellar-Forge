# Mesh and point-source rendering

The mesh/point transition described here applies to modeled stars, spacecraft, and celestial bodies (planets, moons, gas/ice giants) as they shrink. Resolved bodies (projected radius $\ge 2.5$ px) use quadtree geometry. When bodies shrink below $2.5$ px radius, terrain preparation on the CPU is culled completely, and bodies transition through the complementary mesh/point sprite handoff down to $1.5$ px radius, where they become pure point lights. See [terrain](terrain.md).

All apparent sizes in the celestial renderer are **radii in framebuffer pixels**:

`radius_px = radius_au / distance_au * focal_length_px`

The shared transition in `shaders/common/point_source.glsl` is:

`mesh = smoothstep(1.5, 2.5, radius_px)`

`point = 1 - mesh`

Moving away, the handoff begins at a **5 px mesh diameter** and completes at **3 px**, replacing the previous 8-to-4 px band. The mesh keeps its physical radius. At a radius of 1.5 pixels and below it emits no geometry or depth. Mesh brightness and point flux use complementary weights; the point pass does not test against its own fading mesh's depth.

## Flux and appearance

`point_flux = mean_disc_radiance * projected_disc_area_px`

The point shader estimates mean stellar disc radiance with 64 equal-area samples. Stars use the mesh's latitude-dependent color, luminosity and limb darkening. Reflected bodies use the shared integrated Lambert/Hapke material phase curves, incident inverse-square irradiance and receiver-center eclipse visibility, without an artificial night-side floor. Oblateness scales projected area. The point approximation omits mesh albedo textures, detailed ellipsoidal normal integration and spatial eclipse variation across the receiver.

At fixed illumination, shrinking projected area gives inverse-square observer falloff for both emitted and reflected light. A reflected body's brightness also changes with phase and source-to-body distance. Attachment rounding limits precision at extremely faint signals.

Local bodies and Gaia share a compact radial Gaussian profile with a smooth taper to zero at the sprite boundary. The configured point diameter is literal framebuffer pixels and defaults to **3 px**. For local bodies, `sprite_radius_px = max(mesh_radius_px, point_diameter_px / 2)`: the sprite matches the projected mesh throughout the default handoff, shrinking from 5 to 3 px, then stays at the 3 px minimum when fully unresolved. Gaia stays at its configured point diameter. There is no extra halo or antialiasing padding, and brightness and framebuffer height do not enlarge the minimum footprint.

At each source center, the profile is normalized over the pixel centers covered by its billboard. Their weights sum to one, so total point flux remains stable as a source moves between pixels or its configured diameter changes. This is a sampled optical approximation, not an Airy diffraction calculation. At a three-pixel diameter, the visible source occupies at most three columns and three rows.

These footprint contracts apply before post processing. The application can
redistribute light beyond the sprite through Fast bloom or four-spike
sprites, after atmosphere/transparency composition and before tone mapping.
The point diameter and mesh transition remain unchanged. See [bloom](bloom.md).

The tone mapper bounds HDR inputs to the largest finite half-float value before its rational curve. Saturated half-float infinities therefore map to white without an infinity-over-infinity result.

The final composite fetches the corresponding HDR texel directly. Bilinear filtering would spread saturated point pixels beyond their configured footprint and can blend half-float infinities into neighboring pixels.

Both Gaia and local points use instanced billboards with interpolated pixel offsets, avoiding native point-sprite size rounding and keeping the shared profile's width and flux consistent.

The common `point_flux_visible` predicate culls only when every component of the integrated exposed source flux is below `2^-25`, half the smallest positive half-float subnormal. Such a source cannot change an RGBA16F pixel even if all its light falls into that pixel. There is no additional brightness fade or PSF-size-dependent attenuation. Local visibility, mesh weight and observer refraction are applied before the predicate; Gaia compute culling and drawing both include observer refraction gain. Exposure is applied once. Gaia intensity remains a separate artistic control.

Observer refraction returns the original position/direction and unit flux gain
for analytically unaffected rays. Its angular-area ratio uses
`cos(bend) - cos(theta) * sin(bend) / sin(theta)`, with sine obtained from the
direction/up cross product. Reconstructing theta with `acos` and then dividing
two sines lost precision near 180 degrees: a distant Sun could receive zero
gain and disappear as the eye moved. The GPU identity contracts cover both
angular poles and 4050/4095/4096/4097/4101/4150 AU, plus wider distance bounds.

Gaia's magnitude normalization is anchored to a solar absolute G magnitude of 4.67 and the spherical mesh's mean limb factor of 5/6. A solar analogue and a local one-solar-luminosity star consequently agree in integrated brightness at the same distance, exposure and default intensity. This replaces the earlier unrelated catalog coefficient and sprite-size multiplier, so the starfield's brightness changes. G-band catalog flux and bolometric system luminosity still require spectral modeling for exact agreement for other stellar temperatures.

## Occlusion

Celestial point sprites are additive and never write depth. Modeled stellar points evaluate finite-source transits against physical foreground bodies, excluding the source itself and bodies beyond it. Shared eclipse optics use analytic circle overlap and numerical projected-ellipse overlap, including fully unresolved casters. Small-caster dimming follows its area relative to the apparent stellar disc. This attenuates integrated stellar flux without drawing an enlarged subpixel silhouette. Oblate casters use their inclination-dependent projected ellipse. See [eclipse optics](eclipse_shadows.md#oblate-parent-eclipses).

Nonstellar point sources retain mesh-weighted center-ray visibility against foreground ellipsoids; unresolved casters contribute zero to that test. Black holes retain their opaque physical silhouette. Planetary quadtree geometry always writes physical raster depth. Local stellar transit attenuation is whole-source flux, without resolving a silhouette inside a point sprite. Gaia uses raster depth against meshes, including samples covered by subpixel quadtree surfaces. Surface-to-light eclipse shadows use the same finite-source overlap functions with independent receiver/star candidate lists.

## Verification

Enable the optional GPU contracts with `cmake -S . -B build -DSTELLAR_BUILD_RENDER_TESTS=ON`, then build and run `ctest --test-dir build --output-on-failure`. They require Vulkan and a window system, create a hidden window, and inspect rendered HDR pixels for:

- Stellar mesh/point transitions and resolved quadtree reflected-light scaling.
- Inverse-square observer falloff above the visibility cutoff.
- Subpixel stability and flux conservation when changing point size.
- Unresolved, transitioning and resolved foreground occluders.
- Agreement between a synthetic Gaia solar analogue and the local Sun.
- Compact pixel footprints at multiple subpixel offsets, changes in point diameter at 1080p, and a millionfold overexposure through the composite shader.
- Mesh-sized sprites through the 5-to-3 px handoff and a three-pixel unresolved minimum, at 256p, 1080p and 2160p.

`StellarPointRefractionTests` / `point_refraction_identity_contracts` additionally
dispatches the production local-point and Gaia refraction helpers on 256
unaffected near-axis rays. Every ray must retain exactly unit gain and the
local source position must remain unchanged. The original shader returned zero
gain for 13 of 16 near-antipodal rays in the initial 4101-AU reproduction.

Small mesh rasterization errors remain at a few pixels because the mesh pass has no multisample antialiasing. Their reflected-light sweep uses resolved radii of 12–30 pixels and checks area scaling separately from stellar point normalization. `smooth_surface_contracts` additionally verifies resolved quadtree selection down to 2.5 pixels, CPU terrain preparation culling for subpixel bodies and point light transitions, and attachment underflow for imperceptible bodies.

The additional `tests/point_visibility_contracts.cpp` driver checks faint catalog and reflected-body flux across point diameters and subpixel positions against the half-float rounding bound. Its faint-source check fails with the former fade and passes without it. [The visibility investigation](point_source_visibility.md) records PSF experiments, the Venus phase-curve limitation and reproduction commands.

References: [PBRT point lights](https://www.pbr-book.org/4ed/Light_Sources/Point_Lights), [Vulkan fragment depth operations](https://docs.vulkan.org/spec/latest/chapters/fragops.html), and the solar Gaia reference adopted in [A&A 670, A19 (2023)](https://www.aanda.org/articles/aa/pdf/2023/02/aa44250-22.pdf).
