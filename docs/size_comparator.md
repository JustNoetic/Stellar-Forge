# Object size comparison

Open **View → Scene: Size Comparator**. The System Outliner becomes a checklist of
available saved systems and the active live system. Check any number of systems;
their bodies share one physical size scale. Switching back to **Scene: Simulation**
restores normal navigation. Comparison selection and navigation are transient.

Drag the viewport to pan, scroll to zoom around the cursor, and press **F** or
**Fit All** to frame the selected systems. Click a body to inspect it; double-click
or use **Focus Selected** to frame it. Repeated names include their source system
in the viewport labels. Labels prioritize larger bodies and hide when they would
overlap or extend into the workspace panels.

## Original behavior

`comparison::arrange_bodies` ports the original `size_comparator.py` layout:
kilometres × 1e-5, stars and planets in orbital order, depth-first satellite
columns, 0.25 body spacing, 0.125 satellite gaps, 2.0 row gaps, and 0.52 projected
ring bounds. Orthographic projection preserves relative sizes at every zoom.
Black holes retain the existing simulation display-radius convention, identified
in the comparison inspector.

The display light is the original white direction (-0.77459667, 0.38729833, 0.5),
giving a 75% illuminated disk. Stars use the original temperature-to-RGB mapping
and quadratic limb darkening (0.45, 0.15). This scene bypasses HDR exposure, ACES,
secondary illumination, starfield and orbital rendering. Surface and normal maps
use the engine's shared right-handed surface rotation and tile streamer. Texture
selection uses orthographic screen bounds and pixels per body radius, including
the viewport's offset when zoomed and panned; it does not approximate this view
with a distant perspective camera. Detail remains limited by the source pack's
maximum resolution. Rings use
the shared texture loader, profile generator and premultiplied mip builder;
filtering includes radial pixel coverage and the original fixed display pole.

Atmospheres use parallel observer rays, the same white display light, and each
body's optical definition. Transmittance and multiple-scattering LUTs reuse the
engine's shaders and `make_atmosphere_parameters`. Rayleigh, Mie, aerosol and
absorption transport is baked once into a 512 × 257 polar lookup for the fixed
display pole and light. The radial mapping concentrates samples around the
silhouette and separates surface and sky filtering. Scattering and RGB
transmission are cached, with a second layer for the path in front of the ring
plane. The two RGBA16F arrays cost about 4 MiB per visible atmospheric body.

Normal frames sample this lookup; pan, zoom and body spin do not rebake it. Rings
retain their pixel-dependent profile filtering and compose between the cached
air segments. Opaque depths that truncate a path before its normal endpoint use
the integrated fallback. A changed definition replaces the cache. Ring visibility
does not invalidate the optical cache. The atmosphere, ring and surface-map
visibility settings apply.

## Ownership and lifetime

`SizeComparison` retains immutable system definitions and prepared atmosphere
profiles. The active system comes from the current `SystemSnapshot`; other
checked systems are loaded as definitions without creating simulations or
requesting ephemeris playback. Saved definitions reload when their file changes;
**Refresh Systems** also clears the definition cache. Load failures are isolated
to the affected system. Selection identifies both source system and body.

Comparison positions never replace physical positions. `SizeView` owns its own
camera, and comparison data never issues simulation commands. There is no fixed
system-count limit. GPU drawing is culled to visible bodies, surface streaming
retains its existing residency budget, and replaced comparison resources retire
after their last possible GPU use. Lazy initialization occurs between frames.

## Validation

```text
cmake --build build --target StellarForge StellarRenderingTests
ctest --test-dir build -R size_comparison --output-on-failure
build/bin/StellarForge.exe --self-test
```

CPU contracts cover physical scale, orbital order, nested satellites, cursor
anchoring, panel-aware focus, three simultaneous systems, duplicate body names,
isolated load failures, immutable simulation state and empty selection. GPU
contracts check the display light, exposure independence, stellar limb darkening,
filtered rings, finite atmospheric limb radiance, asymmetric surface-map
orientation, maximum-detail tile selection after panning, and cached transport
against a 48-step integrated reference. The reference is available to test
harnesses through `SizeComparatorRenderer::init(ctx, true)`. The GPU test also
reports median GPU and rendering/readback times for a 1080p fullscreen body.
The app self-test enters,
focuses, empties and repopulates the comparator, then returns to simulation.

The 26 October 2026 RTX 4060 Ti check used a 1920 × 1080 viewport, a 10,000 km
body with the physical Earth atmosphere model, a 982-pixel projected diameter,
disabled surface/normal maps, and twelve warm rendering samples per path. Median
GPU time for clearing, surface drawing and atmosphere composition fell from
15.78 ms for the integrated reference to 0.24 ms for the cached path. The matching
CPU-submit/GPU-wait/readback medians were 17.73 ms and 2.20 ms. These are controlled
warm-cache measurements, not application FPS; initial LUT baking is excluded.
Reference comparisons also cover pan/zoom reuse, ring visibility changes,
foreground rings and strongly oblate bodies.
