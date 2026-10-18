# Terrain and planetary surface maps

**Settings → Visibility → Planet Surfaces** controls **Surface Textures**, **Imported Elevation** and **Synthetic Terrain Preview**. Planets, moons, gas/ice giants and nonstellar black holes use quadtree geometry when resolved (projected radius $\ge 2.5$ px). Bodies without enabled, compatible elevation use a smooth ellipsoid with zero displacement and analytic normals; their albedo uses the same body-fixed cube projection as imported terrain. Imported color/elevation are enabled by default when their packs exist. Focus Earth and approach its surface to inspect imported relief. **Terrain Tile Colors** shows selected quadtree leaves without lighting. **Terrain Error** sets the requested geometric error in pixels. **Patches per Body** limits refinement; the scene has 8192 patch records, with six reserved for each remaining visible resolved body before a larger body can refine. The panel reports smooth patches, geometry and texture residency/loading. **FP64 Terrain Reference** compares procedural/smooth geometry against the cached path.

When a body's projected radius drops below 2.5 pixels (5 px diameter), it enters the point light handoff band. Terrain preparation on the CPU is culled completely for that body, avoiding quadtree traversal, patch allocation, and cache bakes. The body transitions seamlessly to the sphere mesh and point sprite passes, completing at 1.5 pixels radius as a pure point light. Imperceptible bodies fade to zero flux via faint signal attenuation. Turning off imported elevation or synthetic preview changes displacement, not geometry ownership. Gas/ice giants and black holes remain smooth even when synthetic preview is enabled.

Initial imported-root loading uses smooth quadtree coverage until the imported roots are admitted. Once imported coverage is available, the existing balanced resident-parent mechanism applies. Geometry-cache misses use the same quadtree grid through the reference shader. `TerrainStats::drawn_bodies/smooth_patches` report these paths. Per-body route lookup is indexed, and offscreen or subpixel bodies are culled before allocating refinement budgets. `MAX_PATCHES >= 6*MAX_CELESTIAL_BODIES` is enforced at compile time.

Synthetic relief remains opt-in; imported elevation supports multiple named bodies simultaneously. Camera surface avoidance and picking still use the reference ellipsoid, so the camera can enter raised terrain. **Synthetic Relief** and the procedural FP64 switch do not alter imported heights.

## Source assets and baked packs

The original 19 images were copied without modification to `textures/Solar System/<Body>/`. Their local `source_manifest.json` records SHA-256, dimensions and byte sizes. `data/surfaces.json` binds body names to derived packs under `data/tiles/<body>/<source-fingerprint>/albedo` and optional `height`. Original images and baked packs are git-ignored; the small catalog, importer and deterministic fixtures are versioned. Runtime skips missing packs and retains existing materials. It never loads the original giant images or depends on Pillow, NumPy or GDAL.

Install Python Pillow and NumPy for the color baker, then run:

```powershell
python scripts/bake_surface_maps.py
python scripts/bake_surface_maps.py --body Earth --level 4
.\build\StellarRenderingTests.exe --surface-preview data/surfaces.json scratch/surface-preview
```

The default builds independent color LOD0–3 for 14 bodies. Each face's finest image is 1024 × 1024, split into 128 × 128 texel cores with two-texel gutters; these starter packs retain less detail than the largest originals. Levels through 5 are supported. Increasing `--level` writes a new color pack and updates its catalog entry; old packs remain intact. No GPU cache resize is needed. Each pack uses one indexed lossless `tiles.rgba` file and a bounded manifest, rather than thousands of individual files. CRC32 checks protect tile reads. Filtering and mip reduction occur in linear light, with sRGB storage/decoding. Geographic samples and gutters use the current linear oblate cube mapping, longitude wrapping and explicit source latitude convention. Legacy tangent-cube tiles are incompatible.

Earth's 8-bit PNG height map uses the explicit `[-10984, 8848]` meter visual range, and Moon's 16-bit TIFF height map (`Moon_heightmap.tif`, `5760 × 2880`) uses `[-9120, 10760]` meters from `data/system.json`. Import uses the rewrite's actual radius constant and body oblateness, not an assumed WGS84 shape. This is a visual calibration assigned to the engine ellipsoid, with no geoid correction or survey accuracy claim. Source hashes and that interpretation are stored in provenance. Baked height packs use 128 × 128 cells per tile (`133 × 133` samples with two-cell gutters and `"cubic-bspline"` reconstruction) through LOD5 for Earth (4096 cells per face, `finest_level = 8`) and LOD4 for Moon (2048 cells per face, `finest_level = 7`); finer geometry reconstructs that finite lattice. Clouds, specular and ring images are preserved for later material passes.

## Color streaming

`SurfaceTextures` owns immutable `ColorSource` bindings, one worker, at most 128 pending reads, 64 completed tiles and 128 failed keys. Up to 32 sources and 1024 snapshot body indices are supported. A source identity/revision and source binding index key all GPU pages. Color remains independent of terrain geometry LOD. Spheres and terrain use the same body-fixed pole/spin basis and shader sampler; albedo does not multiply the existing display color. Source shape must match the body's oblateness.

The shared RGBA8-sRGB array has at most 512 layers (34.03 MiB texel payload), reduced to the device's array-layer limit if necessary. Each frame admits at most 16 tiles through a separate 4 MiB `rhi::StagingUploads` ring. Two bounded page-table buffers map tile addresses to layers. Shared RHI image-range barriers order previous reads, layer uploads and subsequent reads on the graphics queue; staging reuse follows submission completion. Interactive streaming performs no immediate submission or wait.

All six roots per source are pinned, and a body switches to mapped color only after its roots are ready. View selection requests finer pages plus ancestors, retaining a resident fallback for each desired region. Each source reuses its geometric selection while its full `SelectionView` is unchanged; residency, uploads and page ownership continue updating independently. Rebinding sources clears these selections. `TextureStats::selection_nodes/selection_rebuilds` report traversal work. Shader derivatives choose and blend adjacent color levels, walking to resident parents whenever a page is absent. Eviction clears the previous owner's page before reusing its layer. Loading can temporarily show blurred imagery; it cannot leave texture holes. Filtering gutters are regenerated from the geographic source across tile/face borders. The current explicit tile pyramid has no anisotropic virtual-texture filtering, temporal arrival fading or paged high-resolution index.

`surface_map_contracts` checks exact lossless reads, linear-light filtering, interior gutter overlap, filtered cube edges/corners, poles and repeatability. `surface_rendering_contracts` checks corruption rejection, sphere/terrain orientation, disabling imported color, bounded asynchronous uploads and actual GPU slot eviction across 32 body bindings. Its fixtures live in `scratch/surface-tests/` and require no external maps.

## Coordinates and selection

`src/terrain/quadtree.hpp/.cpp` provides six outward-wound cube faces mapped onto an oblate ellipsoid. `TileId{face, level, x, y}` addresses a geographic region independently of GPU residency. The 64-bit key supports 24 subdivision levels without allocating a global finest-level grid. Local coordinates use the equatorial radius as one unit; the polar ratio is `1 - oblateness`, clamped to 0.05–1 for the preview. Height displaces along the ellipsoid normal. The synthetic relief amplitude is capped at one percent of the equatorial radius.

The body-fixed north axis follows the snapshot pole. Its reference east/west orientation comes from the projected render X axis (render Z near that singularity), followed by the snapshot spin. This is a rendering convention; geographic import must explicitly establish its prime meridian, latitude convention and height datum rather than assume that this basis matches a source map.

Selection refines the highest projected-error leaf first. The error estimate combines ellipsoid curvature with per-tile derivative bounds for the synthetic height function and a conservative distance to the displaced patch enclosure. It is specific to this generated surface. Real height data must supply minimum/maximum heights and geometric error metadata, including parent approximation error. The pixel setting is a refinement target, not a guarantee when the patch/depth limits are reached.

`tile_metadata.hpp/.cpp` bounds each sine/cosine phase over a tile's normalized-direction region, including interior extrema. Interval products provide normalized minimum/maximum heights; local gradient and curvature envelopes provide a second-derivative bound for selection. Metadata includes a `2e-5` normalized height allowance for float evaluation. The derivative envelopes describe the analytic signal, not the precision of a real height dataset or float trigonometric noise.

Displaced spheres shift their center to the midpoint height. Their radius includes the tile's base extent, half its local height range, and variation in the oblate surface normal. If an unshifted enclosure is smaller, it is retained instead. Fine patch height uncertainty consequently shrinks with extent instead of remaining the full global relief amplitude.

A second, thin surface enclosure tightens frustum tests and refinement distance. For normalized center direction `d` and direction-space radius `r = tile_width / sqrt(2)`, every surface direction has radial component in `[1-r*r/2, 1]` and tangent magnitude at most `min(r, 1)`. With ellipsoid scale `S`, the enclosure is centered at `S*d*(1-r*r/4) + normal*middle_height`. Its support radius along axis `a` is `abs(dot(S*a,d))*r*r/4 + length(S*a-d*dot(S*a,d))*min(r,1)`, plus half the local height span, normal-variation allowance and FP64 rounding padding. The enclosure is convex and contains every fine/parent/grandparent grid vertex and their interpolated triangles. Refinement takes the greater of sphere distance and the supporting-plane distance toward the eye; both are conservative lower bounds. Frustum planes and eye-dependent horizon constants are prepared once per traversal.

Horizon culling scales positions by `(1, 1/polar_ratio, 1)` and uses a sphere of radius `1 - height_amplitude*(1+RELIEF_ROUNDING)/polar_ratio` in that space. This corresponds to a homothetic inner ellipsoid whose support separation from the base ellipsoid is at least the global relief allowance. Patch spheres expand by `1/polar_ratio` under this transform. Rejection requires the entire transformed bound to lie inside the occlusion cone and beyond its tangent distance. Horizon rejection is disabled inside the occluder. The global minimum-height allowance still deliberately retains hidden terrain; there is no hill-against-hill occlusion test.

`TileMetadataCache` is a bounded CPU LRU of 16,384 entries, with 32 bytes of metadata per entry plus container overhead. Entries are shared across body shapes, amplitudes and fixture generations because the current source is one fixed analytic function; shape/amplitude conversion happens during selection. Zero relief bypasses metadata lookup. Real source providers must add dataset identity/revision to this domain. Geometry remains in the separate 64 MiB GPU pool.

Neighbor lookups cross cube-face edges. Every refinement request recursively includes the coarser-neighbor splits needed to retain a maximum one-level difference between adjacent visible leaves. The complete split group is evaluated against the visible patch budget before any parent is removed. If it cannot fit, the current balanced selection remains intact and reports budget pressure. This reserves transition geometry during refinement and avoids a final ancestor-coarsening pass erasing local detail. Sixteen reusable index patterns collapse odd edge vertices against a coarser neighbor. Spatial morphing interpolates fine vertices from the parent grid triangles. Stitched edges use the coarser neighbor's morph level so both sides evaluate the same boundary. Budget-driven changes can still produce visible detail changes; there is no temporal transition history yet.

Selection is reused when generation, geometry revision, source domain, body-local eye, double-precision view rotation, FOV, framebuffer scale, relief, error target and available budget are unchanged. Moving views rebuild selection. Imported selection retains the desired result and directly uses it when all refinement prerequisites are resident, avoiding a second traversal. Resident-parent coverage remains active while loading. Ancestor lists, patch anchors, geometry keys and bake templates are retained with selection; unchanged transforms and precision/debug flags also reuse draw records. Residency protection and scene/lighting uniforms still update every frame. An incrementally maintained leaf map supports neighbor lookup, split planning and final stitch masks. Queue entries for leaves already split to support a neighbor are discarded.

Moving-ground validation exposed excessive balancing coarsening in the previous selector: an offset one-meter relief view shrank to 15–16 patches at LOD2–4. Balanced refinement replaces that behavior. Regressions require fine detail below and around the camera on smooth and raised surfaces, across interiors, poles and cube seams, plus coverage, budget and stitch invariants under small budgets. Tile-local height/error metadata further removes unnecessary refinement caused by the previous global height envelope. See [terrain performance](terrain_performance.md).

## Vulkan integration and memory

CPU physics, selection, patch anchors and camera/body transforms remain FP64. `GeometryCache` uses a bounded compute pass to evaluate changed patch vertices and normals in FP64, subtracts the body-local patch anchor before conversion, and stores FP32 local offsets, normals and surface directions. Native GPU FP64 is used when baking a missing patch, not on every near-ground draw. Synthetic height evaluation itself uses float trigonometry; high-precision placement does not make that signal survey-grade.

The near-ground draw shader contains no FP64 capability. It fetches cached offsets, reconstructs parent/grandparent triangles from the same grid for morphing and stitched edges, and uses CPU-split high/low eye-space anchors with FP32 transforms. The camera is subtracted before casting to floats; no float planet-sized position needs cancellation. Beyond four equatorial radii, only final eye-space placement/projection uses FP64. CPU eye-space anchors use `Camera::view_rotation_double()` so distant, narrow-angle views do not amplify rounded camera rotation into image drift. `terrain_reference.vert` retains the full procedural FP64 path and the same distant projection correction.

Selection uses local error envelopes, while shader morphing deliberately keeps the shared global coefficient. Assigning unrelated local coefficients to adjacent patches would make their shared-edge morphs disagree. The existing parent/grandparent interpolation and stitch masks therefore retain their common morph rule.

One 16 × 16 cell grid is shared by all patches. The 64 MiB geometry budget fits 4837 cached patches at 48 bytes × 289 vertices each. Reference draw records use 352 bytes and cached records 128 bytes, each with two frame buffers. Together with jobs, the sixteen index patterns and the 4 MiB imported staging ring, buffer payload is approximately 75.7 MiB, excluding allocator overhead and existing scene buffers. Height payloads are reconstructed into geometry on the CPU; the independent color atlas and its staging/page buffers are budgeted separately above. Each precision/stitch group uses an instanced draw; at most 48 groups are needed for normal mixed cached/distant/reference rendering. A diagnostic cached-FP64 benchmark mode can add a fourth group family.

Cache keys contain generation, body index, geometry revision, tile identity, polar ratio and normalized height amplitude. Camera motion, spin, exposure and color do not invalidate geometry. The LRU protects every currently requested resident entry before admitting any misses. At most 128 patches bake per frame; unadmitted misses use the full reference shader until ready. GPU vertex-read/compute-write and compute-write/vertex-read barriers order slot recycling across submissions on the existing graphics queue. Host-written job and instance buffers are flushed through the shared RHI. There are no cache-specific fence waits, immediate submissions or device-idle calls in the interactive frame path.

`CelestialRenderer::prepare` selects terrain and records compute bakes before dynamic rendering begins. `render` retains ownership of scene lighting, eclipse candidates and the point pass. Its body GPU record includes `surface_flags`; the terrain flag suppresses the corresponding fixed sphere. The terrain pass shares `sphere.frag`, including the existing surface reflectance and eclipse shading. Debug colors intentionally bypass lighting. Rendering state is derived from the current snapshot; terrain does not mutate simulation definitions or motions. Benchmark CPU celestial time includes both prepare and draw recording; `gpu_terrain_prepare` measures the prepass separately from `gpu_celestial` drawing.

## Verification

Build and run CPU contracts:

```powershell
cmake --build build --target StellarForge StellarTerrainTests StellarRenderingTests --parallel
ctest --test-dir build -R terrain_contracts --output-on-failure
```

With `STELLAR_BUILD_RENDER_TESTS=ON`, capture terrain probes:

```powershell
.\build\StellarRenderingTests.exe --terrain-preview scratch/terrain-preview
.\build\StellarRenderingTests.exe --terrain-precision
.\build\StellarRenderingTests.exe --terrain-benchmark scratch/terrain-benchmark/capture.json
.\build\StellarRenderingTests.exe --terrain-ground-benchmark scratch/terrain-benchmark/moving-ground.json
.\build\StellarRenderingTests.exe --terrain-relief-benchmark scratch/terrain-benchmark/relief.json
```

The GPU probe checks an oblate disc, lit relief, poles, cube-face edges/corners, one-kilometer and one-meter views, translated astronomical origins, budget enforcement and disabling the preview. It writes optional PPM captures. Precision checks validate LOD24 cached positions against a CPU double oracle, distinct neighboring vertices, cache reuse/invalidation, absence of near-draw FP64 capability, and cached/reference HDR image agreement through 200 AU including off-axis telescope views. CPU contracts also exercise LRU protection, duplicate jobs, bake budgets, generation/shape/source invalidation and capacity fallback. The normal rendering-contract suite runs the terrain and precision probes.

The terrain benchmark compares the default smooth quadtree, full procedural FP64 reference, cached FP32 and cached FP64 arithmetic. The latter three share the requested synthetic relief; the smooth mode explicitly reports zero height. It reports CPU preparation/recording, GPU pass/draw timestamps, cache bakes, selection rebuilds, cold frames and a moving camera trace. The ground benchmark exercises stationary, walking and flying views over analytic relief at two starting positions, with fresh geometry generations, per-frame data and post-timing previews. The relief benchmark compares 0/20 km amplitudes at a fixed 2 km ellipsoid altitude and separately holds 2 km above the generated 20 km-relief surface. Metadata builds, retained entries and selection-node visits distinguish source-cache work from rendered patch count. Output paths must be new. These are isolated offscreen timings, not application FPS. Historical fixed-sphere captures in [terrain performance](terrain_performance.md) describe the former implementation.

## Imported height sources

`TerrainSource::describe` supplies displaced bounds and normalized geometric error without disk I/O. `ProceduralSource` wraps the existing analytic metadata LRU. `HeightSource` loads an immutable versioned manifest independently of its payload file. Source identity and revision participate in GPU geometry keys; each source object owns its separate metadata/envelope domain. Rebinding cancels and joins the old worker, clears derived selections and retains reusable GPU entries under their original keys. Rebinding an identical domain is a no-op. Workers receive an immutable source, never live simulation views.

Generate a small deterministic global dataset, or import the checked-in geographic raster fixture:

```powershell
python scripts/import_terrain.py scratch/height-fixture --level 2
python scripts/import_terrain.py scratch/geographic-fixture --input tests/terrain_signed_fixture.json --level 2
```

Output must be a new directory. Import writes a temporary sibling directory and publishes the complete dataset by renaming it. The directory contains `manifest.json` and `heights.bin`; identity derives from the normalized manifest and payload SHA-256. Version 1 stores uncompressed, lossless little-endian FP32 signed meters with scale 1 and offset 0, either 16-cell tiles (`19 × 19` samples with 1-cell gutter and `"bilinear"` reconstruction) or 128-cell tiles (`133 × 133` samples with 2-cell gutter and `"cubic-bspline"` reconstruction). Each payload has an FNV-1a checksum. Other encoding identifiers are reserved for later format versions.

Activate the dataset by creating `data/terrain.json` before starting the app:

```json
{
  "body": "Earth",
  "dataset": "scratch/height-fixture"
}
```

Dataset paths resolve through `sf::paths::external_path`. The binding is a renderer asset configuration; it does not mutate simulation definitions or physical state. The body must match the imported equatorial radius and polar ratio. Incompatible body edits use the body's smooth quadtree ellipsoid rather than stretching the geographic dataset onto another ellipsoid. A malformed configured manifest fails initialization with a diagnostic. Payload errors are reported and retain available parent coverage; initial root failure retains smooth quadtree coverage.

The JSON input is a rectangular geographic point lattice, north-to-south rows, east-positive longitude, geodetic latitude and ellipsoidal heights. `coverage_deg` is `[west,east,south,north]`, with a positive longitude span up to 360 degrees; longitudes wrap relative to west. `scale` and `offset` convert raw valid values into signed meters. Null, the declared `nodata` sentinel, nonfinite values and uncovered regions use the explicit `fill_m`. Missing values are identified before normalization and filled before interpolation. Filling constructs a complete global surface; missing-data regions do not become geometric holes. The source coverage and normalization remain in provenance, while runtime coverage is `global-filled`.

With GDAL Python bindings installed, a single-band affine-georeferenced elevation raster can be converted offline:

```powershell
python scripts/import_terrain.py scratch/imported-dem --gdal path/to/ellipsoidal-dem.tif --vertical-datum ellipsoid --height-unit m --fill-m 0 --level 3
```

The adapter uses bounded 64-block reads, explicit longitude/latitude axis mapping, pixel-center sampling, band scale/offset, units, masks and no-data values. Version 1 accepts WGS84 horizontal coordinates, including projected WGS84 input, on the matching target reference ellipsoid. Inputs must already contain ellipsoidal heights. Orthometric heights, compound CRS and other horizontal datums must be converted first with an explicit GDAL/PROJ pipeline and the required geoid grids. This importer does not silently approximate a datum conversion. GDAL is optional and never linked into the runtime. Its adapter test skips explicitly when bindings are unavailable.

The engine cube parameter is not geodetic latitude. Import converts each cube direction to the normal of the oblate reference ellipsoid, then derives geodetic latitude. Longitude is `atan2(local.x, local.z) + prime_meridian_rad`: local +Z denotes the configured prime-meridian offset, local +X is east, and local +Y is north. The source is subsequently placed by the existing body pole/spin transform. This is an explicit engine orientation contract, not an assertion that existing Earth spin settings implement an authoritative terrestrial reference frame.

Coincident samples are nested across data levels. Canonical cube-face ownership makes boundary values identical at edges and corners. Gutters sample the finest reconstructed surface across face boundaries. Rendering parents are not averaged mipmaps. Parent extrema include descendants; parent scalar error combines child error with the maximum child-to-parent triangle residual. Dyadic triangle subdivision preserves the parent diagonal, so a scalar piecewise-linear residual is bounded by its subdivision vertices. At the finest data level (`finest_level = data_level + log2(tile_cells / 16)`) the reference height is its decoded triangle reconstruction. Error additionally includes a conservative oblate position/normal curvature allowance, height gradients, a coarser edge reconstruction allowance and FP32 encoding error. It bounds reconstruction of the imported representation, not the measurement uncertainty or unresolved features of the original DEM.

Version 1 uses a bounded flat index: at most 16,384 records and a 16 MiB manifest, with global data levels 0–5 (level 5 has 8,190 records). Importer vertex memoization is capped at 65,536 samples. Large/sparse high-resolution datasets require a paged index in a subsequent format. Geometry can still reach LOD24 by reconstructing the finite source lattice; this does not create new measured detail. Stored metadata remains available without payloads. For virtual descendants (`level > data_level`), a 512-entry resident envelope LRU derives tighter local bounds and exact per-`8 × 8`-mip-block geometric residuals (`mip_err8`, `mip_err4`, `mip_err2`) and gradients from already loaded `data_level` payloads; absent envelopes conservatively use stored ancestor ranges, without selection-thread I/O. The worker refreshes envelope residency on payload-cache hits as well as reads.

## Streaming and resident coverage

### Imported geometric error and selection cost

Imported error uses a bound on interpolation of the decoded finite height lattice over the oblate cube, rather than the original blanket curvature coefficient. Write `P = D q/|q| + h normalize(D^-1 q)`, with `D = diag(1,p,1)` and `q` the linear face vector. For a right grid triangle of cell width `s`, barycentric squared distance is at most `s²/2`. The normalized-vector map has directional second derivative at most `2/(sqrt(3)|q|²)`; the normal map adds factors `1/p²` for its second derivative and `1/p` for its first derivative. A coarse height triangle's gradient is at most `sqrt(2)` times the finest lattice's stored gradient envelope.

The implemented normalized position-error allowance is:

```text
height_residual / radius + encoding_error / radius + 32 * double_epsilon
  + s² * (0.5 / q_min²
          + 0.5 * max_abs_height / (p² * normal_q_min²)
          + height_gradient / (p * normal_q_min))
```

Heights and gradient in the last line are divided by equatorial radius. `q_min` and `normal_q_min` are analytic minima over the tile rectangle, computed without payload reads. Constants 0.5 and 1 relax the derived coefficients `1/(2 sqrt(3))` and `1/sqrt(2)`. Height residuals remain conservative metadata bounds, capped by the local stored height range because both the decoded surface and its coarse interpolant lie within it. The selector retains the parent grid's height residual and doubled cell width; it does not simply substitute fine-grid error for morphing geometry. The error helper separately supports strides 1/2/4 and is tested against both triangles of fine/parent/grandparent cells through LOD24, across all faces and polar ratios 0.05, 0.5 and 1. Constant negative height is tested separately.

Shader morph thresholds still use one common dataset-wide allowance per level. Their global curvature uses the same derivation with the most conservative face bounds and source-wide height/gradient maxima. Shared edges therefore keep compatible morph thresholds. Existing height packs remain valid; no rebake or format revision is required.

Imported desired refinement/prefetch is reused while view/source bounds are unchanged. Drawable refinement additionally tracks GPU residency revisions so pending groups become visible after admission and evicted groups return to resident parents. New/evicted CPU payload envelopes advance a source bounds revision. A renderer-owned table memoizes at most 16,384 tile descriptions for the active imported body, shared by desired/drawable traversal and subsequent moving views; bounds, generation or geometry changes invalidate it. System generation changes clear old selection tables. Source workers do not mutate renderer caches or simulation snapshots.

Patch preparation computes body/view matrices once per body and shared morph allowances once per level. Only the active cached or reference draw records are written/flushed. Instanced draw grouping, atomic balanced splitting, patch budgets and the 64 MiB geometry cache remain in place. The source benchmark reports preparation versus draw recording, selection visits/rebuilds and resident loading independently:

```powershell
.\build\StellarRenderingTests.exe --terrain-source-benchmark data/tiles/earth/95cebe9dca5b/height scratch/terrain-bounds/capture.json
.\build\StellarRenderingTests.exe --terrain-source-benchmark data/tiles/earth/95cebe9dca5b/height scratch/terrain-bounds/paced.json --paced
```

Use the actual height path from `data/surfaces.json` when its source fingerprint changes. `--paced` gives moving frames at least 16.667 ms of wall time, outside measured pass intervals; the default remains a free-running throughput trace. See [terrain performance](terrain_performance.md) for before/after results and loading limits.

`HeightStream` owns one background worker, a maximum of 128 pending jobs, 256 completed geometry blocks, a 64-payload LRU and bounded failure bookkeeping. Geometry is baked in CPU FP64 from resident signed heights and stored in the same 48-byte patch-local FP32 vertex layout as procedural geometry. CPU queue payloads total roughly 3.4 MiB at the completed-block limit, plus pending/shared upload references, the height/envelope LRUs and index/container overhead. There are no separate GPU height textures in this milestone: imported geometry shares the fixed 64 MiB GPU cache and the combined 128 geometry-admissions/frame limit with procedural bakes.

Desired selection prefetches accepted refinement groups, including offscreen siblings. Drawable selection independently starts from resident roots and commits a parent split only when all four children and recursively required neighbor splits are ready and the visible patch budget fits. Pending groups leave parents in place. Stitch masks come from the drawable selection; there is no ancestor substitution after selection and no final coarsening pass. All six roots must be resident before switching the body from its ordinary renderer. `Selection::required_tiles` retains the roots and complete child groups of every committed split, including culled siblings. These dependencies and all six roots are protected before cache admission; capacity pressure defers new geometry instead of removing cover or invalidating existing refinement. Protecting only drawn leaves and ancestors allowed an offscreen sibling eviction to collapse loaded ground detail from LOD9 to LOD1 on the next traversal. A CPU regression forces unrelated cache churn and a camera update, then requires the same detailed selection. Camera motion into new regions can temporarily lower drawable detail while requested tiles load. This is visible quality backpressure, not a pixel-error guarantee during loading or budget exhaustion.

Nested samples preserve the draw shader's strided parent/grandparent reconstruction. Imported morph thresholds use common dataset-wide per-level errors, including global shape/gradient allowances, rather than unrelated local coefficients. Normal samples use canonical face coordinates and differences of displaced FP64 positions at the dataset's physical spacing; `"cubic-bspline"` packs evaluate smooth uniform cubic B-spline derivatives from each tile's 2-cell gutter (`height_sample_cubic`) without requiring cross-tile neighbor reads. Duplicate boundary vertices therefore receive matching normals at poles, face seams and cube corners. Normal filtering uses source resolution, independent of geometry subdivision; shading normals retain the existing fine-normal behavior during positional morphing.

`rhi::StagingUploads` owns a fixed 4 MiB mapped staging ring for terrain. It records nonblocking buffer or supported uncompressed image-region copies into the current graphics command buffer. Allocation reuse is tied to monotonic submission completion, advanced by frame-fence or immediate-submit completion. The combined 128-admission budget fits two frames of imported uploads in the ring. GPU geometry-slot reuse remains ordered against previous vertex/compute/transfer uses by shared graphics-queue barriers. No new wait or immediate submission occurs in interactive streaming. Immediate submissions cannot nest inside an active frame recording.

`transition_image_range` takes explicit old/new layouts for selected mips/layers and also records same-layout dependencies. Callers own per-subresource layout state; the existing whole-image `Image::layout` must not be used to represent mixed layouts. `StagingUploads::image` supports the listed uncompressed scalar/color formats and validates extent, layer, row-layout and byte capacity. Destination availability/layout transitions remain explicit. The old blocking `upload_image_2d` continues serving initialization.

Verification includes importer determinism, signed normalization/no-data, nested seams and error containment; LOD24 displaced bounds and geometry errors; bounded completion/staging queues; corruption and source-revision rejection; CPU eviction; delayed/out-of-order balanced refinement; and imported moving GPU views. The GPU streaming probe also exercises three actual frames through fence-based staging reuse and a signed R32 mip/layer upload/readback:

```powershell
ctest --test-dir build --output-on-failure
.\build\StellarRenderingTests.exe --terrain-imported scratch/height-fixture scratch/terrain-benchmark/imported.json
```

The generated CTest fixture is recorded in `scratch/terrain-source-tests/fixture-path.txt`. Set `STELLAR_VULKAN_VALIDATION=1` to request the installed Khronos layer in rendering tests. Benchmark output includes readiness delay, cold maxima, loading frames, drawable LOD, CPU/GPU intervals and worker counts. Timings exclude the trace's height query and source worker CPU work; worker work runs concurrently. These remain serialized offscreen pass measurements, not application FPS. Camera collision/picking, temporal LOD history, automatic datum conversion and paged high-resolution source indexes remain subsequent work.
