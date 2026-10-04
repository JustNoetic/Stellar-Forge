# Surface materials

The body inspector's **Cosmetics → Surface Material** controls the solid surface
response. Both sphere meshes and terrain use `common/surface_material.glsl`.
Point rendering and planetshine use disk integrals of the same model.

## Presets and legacy systems

Missing `material` or `preset: "auto"` resolves Moon/Luna to **Lunar regolith**,
Europa to **Icy regolith**, and every other body to **Lambert**. Body type alone
does not identify surface composition. Stars and barycenters retain Lambert.
The lunar and icy presets are illustrative defaults, not measured fits.

Select a preset to initialize its parameters. Editing a Hapke slider creates a
custom material. Existing **Save System Cosmetics / Save to Master Cosmetics**
and single-body cosmetic export include the material. Full system serialization
already preserves the body dictionary. Select Automatic to remove the override.

Example body field:

```json
"material": {
  "preset": "lunar",
  "model": "hapke",
  "single_scattering_albedo": 0.23,
  "phase_width": 0.35,
  "backscatter_fraction": 0.9,
  "opposition_strength": 1.0,
  "opposition_width": 0.06,
  "roughness_deg": 20.0,
  "brightness": 1.0,
  "coherent_strength": 0.15,
  "coherent_width": 0.006
}
```

## Response and units

Hapke uses a two-lobe Henyey–Greenstein particle phase function, isotropic
Chandrasekhar H approximation for multiple scattering, shadow-hiding opposition,
coherent-backscatter opposition, and Hapke's 1984 macroscopic roughness correction.
The backward fraction is 0 for the forward lobe and 1 for the backward lobe.
Higher `phase_width` b means stronger anisotropy/narrower lobes. The two opposition
widths are dimensionless h in tangent-of-half-phase coordinates. Roughness is the
unresolved mean slope angle in degrees, separate from resolved terrain normals.
`single_scattering_albedo` w is a grain property, not geometric or Bond albedo.

Existing textures are display/reflectance maps rather than grain-albedo maps.
To keep their brightness useful, the Hapke I/F response is normalized at
incidence=30°, emission=0°, phase=30° to the existing cosine response. Textures
remain reference reflectance and `brightness` scales the result. This preserves
their color/detail while allowing phase and limb behavior to change. It is a
real-time photometric material, not a spectral composition retrieval model.
Airless Hapke bodies' inspector albedos come from the integrated phase curve,
including the chosen reference reflectance. Brightness can deliberately produce
nonphysical reflectances; the existing Bond-albedo display clamps to [0,1].

The directional model is applied to direct starlight and incident planetshine.
Stars are averaged with a bounded five-point disk quadrature near the terminator
and opposition peak. Existing eclipse and ring shadows still modulate illumination.
Granular Hapke materials suppress the sphere's legacy smooth specular highlight.
Lambert and cloud passes retain their existing shading. Skylight and integrated
ringshine retain their diffuse approximation; the latter is not yet integrated
against the Hapke directional response. Atmosphere albedo estimates retain their
existing approximation. Gas giant cloud transport is a separate future feature.

## Transport and implementation

`SurfaceMaterialCache` packs three vec4s per body in SSBO binding **11**, aligned
with the primary/comparison instance order. A float atlas in binding **12** stores
721 disk samples over 0..180° per unique material. Tables integrate the visible
illuminated lune, including thin crescents, and are rebuilt only when material
parameters or body order change. CPU planetshine uses the same arrays and includes
cached linear texture means for Hapke bodies. The distant point pass shares those
means via otherwise unused nonstellar instance slots 19/23/27.

Point rendering retains its existing primary-star, spherical, spatially averaged
approximation. Planetshine retains its finite-distance cap mapping, eclipse
approximation, distance cutoffs and artistic boost. Surface terrain textures and
sphere textures continue to follow their existing normal/texture sampling paths.

Run `python scripts/test_surface_materials.py` for response boundary checks,
bad-input sanitization, CPU/GPU agreement, higher-resolution phase integration,
material buffer reordering, production shader linking, cosmetics save/reload,
and legacy/material planetshine compatibility. Requires the normal project
dependencies and an OpenGL 4.6 context.

## References

- [USGS ISIS Hapke implementation (public domain)](https://github.com/DOI-USGS/ISIS3/blob/dev/isis/src/base/objs/Hapke/Hapke.cpp): macroscopic roughness and HG2 convention.
- [Hapke (1984), macroscopic roughness](https://doi.org/10.1016/0019-1035(84)90054-X).
- [Regional study of Europa's photometry](https://arxiv.org/abs/2007.11445): icy granular surfaces can also backscatter.
