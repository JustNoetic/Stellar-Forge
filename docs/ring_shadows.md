# Ring shadow blur

Surfaces, terrain and atmospheric receivers share the inexpensive prefix filter
in `shaders/common/ring_shadow_filter.glsl`. A ring host costs six texture reads
with the default three weighted boxes, or two with the diagnostic single box.
There is no stellar-disc quadrature in the rendering path.

The base penumbra radius is `star_radius / star_distance * receiver_to_ring_distance`.
The ring-plane projection stretches that radius according to incidence and radial
orientation. Grazing stretch is no longer capped at `1/128`. For a projected
footprint that runs along a curved band, a small curvature correction broadens
and shifts the radial average outward. Three weighted prefix differences produce
a rounded blur, with less emphasis on its narrow dark core than the former kernel.

Prefix coordinates clamp at the physical inner and outer ring boundaries, but
the normalization covers the entire blur footprint, including clear space. Thin
opaque bands consequently become shallower with increasing blur. Pixel size and
the 4096-sample profile retain a minimum antialiasing footprint independently of
physical penumbra width.

Near the ring-plane horizon, a smooth coverage factor approximates the fraction
of the star on the receiving side. A representative direction shifts toward that
visible portion, avoiding a hard center-ray rejection. This remains an artistic
approximation, especially for very broad sources and extreme grazing geometry.
The existing 4097-by-64 extinction LUT and its linear incidence spacing are retained.

`RenderSettings::ring_shadow_softness` scales physical blur from 0.25 to 4,
defaulting to 1.25. Rendering settings expose it as **Shadow Softness**, beneath
**Ring Shadows**. CPU host bounds and atmospheric interval bounds include this
multiplier and curvature support, so tuning cannot clip their broad penumbrae.

`ring_shadow_reference_linear` restores the original footprint and kernel for
diagnostics. `ring_shadow_single_box` selects one box instead of the default
three. `ring_shadow_reference_all_hosts` independently bypasses CPU candidates.
Atmospheric shadow histories invalidate when softness changes.

## Verification

`StellarRenderingTests --surface-ring-shadows` checks narrow-band dilution with
receiver distance, apparent star size, softness, tangential curvature, inner and
outer edge continuity, finite stellar endpoints, and continuity across the
ring-plane horizon. Rendered moon checks normalize against an unshadowed surface
and scan penumbra width and adjacent-pixel changes. Filtered and all-host images
must match exactly, including grazing illumination and both softness limits.
CPU candidate contracts sample both surface/interior points and stellar-rim rays.

```powershell
cmake --build build --target StellarForge StellarRenderingTests StellarEclipseTests --parallel 4
.\build\bin\tests\StellarEclipseTests.exe
.\build\bin\tests\StellarRenderingTests.exe --surface-ring-shadows
.\build\bin\tests\StellarRenderingTests.exe --surface-ring-benchmark tests/ring-penumbra/new-blur
```

The 1080p benchmark compares all-host/original blur, filtered/original blur and
filtered/new blur in moon-shadow, irrelevant-ring and broad-penumbra scenes.
Reported GPU times cover celestial preparation and visible rings, not application
FPS. Results and preview images are retained in the requested directory.

Release measurements on the RTX 4060 Ti, 2026-10-22, median GPU total:

| 1920x1080 view | Original filtered three-box blur | New distance blur |
|---|---:|---:|
| Moon in ring shadow | 0.3021 ms | 0.3174 ms |
| Rings irrelevant to moon | 0.2212 ms | 0.2212 ms |
| Broad stellar penumbra | 0.2908 ms | 0.3144 ms |

Reports are retained in `tests/ring-penumbra/distance-blur/`. The discarded
stellar-disc integration experiment measured 10.02 ms in the broad-penumbra
fixture in a separate capture; clocks and unrelated pass costs varied between
captures, so use the paired table for the new filter's incremental cost.

In the narrow-band probe, extinction at receiver distances 0.1/1/10/100 was
0.9991/0.4717/0.04680/0.004238 for fixed apparent star size. The rendered grazing
moon scan increased from 37 to 55 partial-shadow pixels; its largest adjacent
visibility step fell from 0.03009 to 0.01968. These are controlled fixtures,
not accuracy guarantees for every geometry.
