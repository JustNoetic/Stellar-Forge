# Native terrain backend

The terrain CPU pipeline now runs in the `stellar_terrain` Rust extension. Python
owns the application and OpenGL context; native workers never call OpenGL or
Python. GLSL owns displacement, normals, lighting and drawing.

## Build and distribution

Use a stable Rust toolchain (Rust 1.88 or later) and the normal Python environment:

```sh
python scripts/build_terrain.py
```

Both launchers run the helper with `--if-needed`. An installed extension is reused
until its Rust sources change. `maturin` builds an ABI3 wheel for Python 3.10+.
Windows and Linux build the same Rust sources. End users of a packaged executable
or a matching prebuilt wheel do not need Rust. PyInstaller includes the extension
through the explicit `stellar_terrain` hidden import. Missing/incompatible native
modules produce a build instruction, rather than silently selecting old terrain.

## CPU pipeline

- `native/terrain/src/quadtree.rs`: float64 traversal, horizon/frustum tests,
  priority refinement, complete-parent budget fallback, 2:1 neighbour balancing
  across cube faces, and fine-to-coarse edge masks. Geometry LOD supports 0–20.
  Refinement covers the horizon-visible surface before frustum culling, so
  rotating the camera changes visibility without reallocating a body's LOD
  budget. The Python adapter retains the full native selection for split/merge
  hysteresis: existing branches merge below 80% of the split threshold.
  Drawing uses curved surface bounds with radial elevation limits and the
  altitude-dependent terrestrial refraction allowance. Ocean depth does not
  inflate the positive elevation limit. A fully culled terrain/cloud list stays
  empty; it is never replaced by a traversal with frustum culling disabled.
  Hidden subdivision history is retained for stability but is not submitted to
  the GPU or counted in the rendered-patch HUD.
  Equal/coarser neighbour lookup uses integer face-grid adjacency and ascending
  ancestors; only cube-face boundaries need spherical remapping. Splits known
  to require four children skip geometry evaluation when the budget is full.
  The adapter reuses a settled selection when its camera, projection and LOD
  inputs are unchanged, while still reculling each frame. Camera translation,
  zoom, radius, budget and altitude changes invalidate this selection cache.
- `native/terrain/src/stream.rs`: case-preserving tile catalog, native image
  decoders, bounded priority jobs, ancestor requests/fallback, separate colour
  and elevation residency, last-use protection, and generation-safe reloads.
  Stale queued jobs expire after three untouched frames. Started decodes can
  finish, but results from an old reload generation are discarded.
- `native/terrain/src/surface.rs`: the same position-based hills/ridge field used
  by GLSL, explicit water clipping, and displaced vertices for camera collision.
- PyO3 releases the GIL for traversal, batch residency resolution and surface
  queries. A configurable native worker pool defaults to at most eight threads.
- LOD estimates patch size in pixels, rather than measured terrain geometric
  error. A spherical-cap distance and the camera's sampled surface altitude
  avoid collapsing all patches within the planet's elevation bound to a
  near-zero distance. Elevation bounds remain part of conservative culling.
  Ground LOD Closeness below 1 increases detail as the camera approaches the
  sampled ground. Max LOD Depth remains a hard cap: the user's saved value of 5
  cannot produce fine ground geometry; depths 12–16 allow much finer meshes.
  Stable selection spends some budget on off-screen neighbours rather than
  reallocating it when the view rotates. Multi-body shared-budget changes and
  changes in resident source heights can still change the selected mesh.
- The Python terrain files are adapters: they exchange packed arrays, upload
  completed tiles and bind render resources. No per-object Python entities are
  introduced; future scatter/generation jobs can use this native boundary.

Raw patches contain ten float32 fields: `[u0,v0,u1,v1,face,lod,x,y,radius,edges]`.
The GPU patch record contains six vec4s (96 bytes): range, colour UV transform,
metadata, legacy height fallback, elevation/camera metadata, and edge/water data.
Edges are left/right/down/up bits 1/2/4/8. Supported grid resolutions are even;
odd fine-edge vertices interpolate the actual displaced coarse-edge endpoints.

The colour pool uses three quarters of the layer capacity and height uses one
quarter. Colour is RGBA8; height is R32F. At 512 content texels, a 1024-layer pool
allocates about 1.016 GiB in total, plus the existing bounded geometry cache.
Resident CPU heights are bounded by the height pool. Pending decode results have
a 64 MiB pixel-data budget (8–128 jobs, depending on tile size); workers also hold
bounded temporary source/neighbor buffers. OpenGL uploads retain the per-frame
count/time budget. GL calls remain on the render thread.

## Shared surface and patch continuity

`common/terrain_surface.glsl` maps any planet direction to a canonical cube face
and samples the resident elevation hierarchy via an SSBO hash table at binding
15. Its height texture array uses texture unit 15; colour retains unit 14. Source
LOD is independent of mesh LOD. Child elevation fades into the common parent
within two source texels of a child boundary, including during partial streaming.

Rendering, compute-cache bakes and collision share the elevation convention,
cubic filtering, procedural field and water rule. Finite-difference normal probes
use a fixed angular stencil and resolve across tiles/faces instead of stopping at
patch edges. Analytic displaced tangents avoid subtracting planet-scale float32
positions, and normals already include oblateness before rotation.

The geometry cache invalidates on height residency/reload revisions and physical
parameters. Colour-only uploads do not invalidate it. Unbaked patches use the same
surface functions directly. The vertex path evaluates cube directions and local
positions in float64 before camera-origin subtraction. Camera and frustum inputs
to native traversal retain float64 precision.

Procedural hills/ridges depend on sphere position and elevation, not integer
patch LOD. This removes the old patch-dependent displacement switch. This field
still describes landscape-scale features; dense rocks, grass, collision entities,
screen-space vegetation LOD and sub-metre material shading are future consumers
of this backend, and are not added by this rewrite. Mesh LOD transitions and newly
streamed detail can still change the visible approximation; temporal geomorphing
and upload blending are separate future work.

Water is configured with `height_water_level_km` (default 0 for Earth, otherwise
none). Negative elevation alone does not imply an ocean on a moon or asteroid.

## Tile format and rebaking

```sh
python scripts/bake_planet_tiles.py --body Earth --map-type height --max-lod 5
```

New bakes use direct source sampling at pixel centres, two actual cross-face
gutter texels on every side, and PNG. The content size remains 512 by default;
the stored image is 516×516. Elevation uses normalized uint16 values, with physical
kilometres supplied by the body's `height_range_km`. Colours are lossless RGBA.
`terrain-v2.txt` documents the content size, gutters and height bit depth.

The direct baker processes a bounded number of tiles at once; new-format bakes
currently use CPU NumPy rather than the legacy whole-face CUDA kernel. Save/decode
errors propagate. PNG takes precedence over a stale JPEG of the same tile.

Existing unpadded 8-bit tiles remain readable. Native workers reconstruct border
samples from neighbours where available and convert heights to float32. Neither
this conversion nor rebaking an 8-bit source recovers missing elevation detail.
For a 20 km elevation range, 8-bit steps are about 78 m; 16-bit steps are about
0.305 m. Finer physical features require higher-precision sources or procedural
residuals and separate geometry/material detail. Do not run the legacy
`stitch_tiles.py` edge-averaging utility on new padded tiles.

## Verification

```sh
cargo test --manifest-path native/terrain/Cargo.toml
python -m pytest scripts/test_terrain_native.py -q
```

The Python regressions exercise actual OpenGL 4.6 shaders, CPU/GPU surface
agreement, cross-LOD positions, rendered edge stitching, matching cache normals,
bounded cache fallback, 16-bit preservation, reloads, absent tiles and true baked
gutters. The Rust suite covers coordinate mapping, budgets, balancing and tile
keys. Linux and packaged executable builds need verification on those targets.
LOD regressions also cover rotation-independent selection, retained off-screen
history, nondecreasing local detail during descent, split/merge hysteresis, and
altitude above the shared displaced oblate surface.
