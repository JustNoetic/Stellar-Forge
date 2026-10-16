# Mesh and point-source rendering

All apparent sizes in the celestial renderer are **radii in framebuffer pixels**:

`radius_px = radius_au / distance_au * focal_length_px`

The shared transition in `shaders/common/point_source.glsl` is:

`mesh = smoothstep(1.5, 2.5, radius_px)`

`point = 1 - mesh`

Moving away, the handoff begins at a **5 px mesh diameter** and completes at **3 px**, replacing the previous 8-to-4 px band. The mesh keeps its physical radius. At a radius of 1.5 pixels and below it emits no geometry or depth. Mesh brightness and point flux use complementary weights; the point pass does not test against its own fading mesh's depth.

## Flux and appearance

`point_flux = mean_disc_radiance * projected_disc_area_px`

The point shader estimates mean disc radiance with 64 equal-area samples. Stars use the mesh's latitude-dependent color, luminosity and limb darkening. Reflected bodies use the mesh's Lambert/Lommel-Seeliger mixture and incident inverse-square irradiance, without the former artificial night-side floor. Oblateness scales projected area. The point approximation omits mesh albedo bands, detailed ellipsoidal normal integration and eclipses of the incident light.

At fixed illumination, shrinking projected area gives inverse-square observer falloff for both emitted and reflected light. A reflected body's brightness also changes with phase and source-to-body distance. Distance-dependent behavior near the visibility cutoff intentionally includes the faint-signal fade.

Local bodies and Gaia share a compact radial Gaussian profile with a smooth taper to zero at the sprite boundary. The configured point diameter is literal framebuffer pixels and defaults to **3 px**. For local bodies, `sprite_radius_px = max(mesh_radius_px, point_diameter_px / 2)`: the sprite matches the projected mesh throughout the default handoff, shrinking from 5 to 3 px, then stays at the 3 px minimum when fully unresolved. Gaia stays at its configured point diameter. There is no extra halo or antialiasing padding, and brightness and framebuffer height do not enlarge the minimum footprint.

At each source center, the profile is normalized over the pixel centers covered by its billboard. Their weights sum to one, so total point flux remains stable as a source moves between pixels or its configured diameter changes. This is a sampled optical approximation, not an Airy diffraction calculation. At a three-pixel diameter, the visible source occupies at most three columns and three rows.

The tone mapper bounds HDR inputs to the largest finite half-float value before its rational curve. Saturated half-float infinities therefore map to white without an infinity-over-infinity result.

The final composite fetches the corresponding HDR texel directly. Bilinear filtering would spread saturated point pixels beyond their configured footprint and can blend half-float infinities into neighboring pixels.

Both Gaia and local points use instanced billboards with interpolated pixel offsets, avoiding native point-sprite size rounding and keeping the shared profile's width and flux consistent.

The common faint-signal multiplier is `smoothstep(1e-6, 2.5e-6, peak_exposed_signal)`, evaluated once per source rather than independently on every halo fragment. Exposure is applied once. Gaia intensity remains a separate artistic control.

Gaia's magnitude normalization is anchored to a solar absolute G magnitude of 4.67 and the spherical mesh's mean limb factor of 5/6. A solar analogue and a local one-solar-luminosity star consequently agree in integrated brightness at the same distance, exposure and default intensity. This replaces the earlier unrelated catalog coefficient and sprite-size multiplier, so the starfield's brightness changes. G-band catalog flux and bolometric system luminosity still require spectral modeling for exact agreement for other stellar temperatures.

## Occlusion

Celestial point sprites are additive and never write depth. Their visibility is computed along the source-center ray against physical foreground ellipsoids, excluding the source itself. A caster's opacity follows its mesh weight; fully unresolved bodies contribute zero. Resolved meshes still hide points. This avoids magnified subpixel silhouettes and unstable raster-depth intersections. Black holes retain their opaque physical silhouette.

This is whole-source visibility, without partial eclipses across a point sprite. Gaia still uses raster depth against meshes; fully unresolved celestial bodies no longer write that depth. Surface-to-light eclipse shadows remain a separate physical calculation.

## Verification

Enable the optional GPU contracts with `cmake -S . -B build -DSTELLAR_BUILD_RENDER_TESTS=ON`, then build and run `ctest --test-dir build --output-on-failure`. They require Vulkan and a window system, create a hidden window, and inspect rendered HDR pixels for:

- Stellar and reflected-light mesh/point transitions.
- Inverse-square observer falloff above the visibility cutoff.
- Subpixel stability and flux conservation when changing point size.
- Unresolved, transitioning and resolved foreground occluders.
- Agreement between a synthetic Gaia solar analogue and the local Sun.
- Compact pixel footprints at multiple subpixel offsets, changes in point diameter at 1080p, and a millionfold overexposure through the composite shader.
- Mesh-sized sprites through the 5-to-3 px handoff and a three-pixel unresolved minimum, at 256p, 1080p and 2160p.

Small mesh rasterization errors remain at a few pixels because the mesh pass has no multisample antialiasing. At the smaller five-pixel resolved diameter, the reflected-light sweep permits 26% sampling error (the centered quarter-phase mesh measures about 25% low); inverse-square flux and point normalization retain their strict checks. The contracts distinguish this existing mesh sampling error from a missing or self-occluded point contribution.

References: [PBRT point lights](https://www.pbr-book.org/4ed/Light_Sources/Point_Lights), [Vulkan fragment depth operations](https://docs.vulkan.org/spec/latest/chapters/fragops.html), and the solar Gaia reference adopted in [A&A 670, A19 (2023)](https://www.aanda.org/articles/aa/pdf/2023/02/aa44250-22.pdf).
