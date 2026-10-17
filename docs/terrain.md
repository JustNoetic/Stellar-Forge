# Terrain LOD groundwork

Enable **Settings → Visibility → Terrain LOD**. Focus Earth or another solid body and approach its surface. **Terrain Tile Colors** shows the selected quadtree leaves without lighting; disable it to inspect synthetic relief under the existing celestial lighting and eclipse model. **Terrain Error** sets the requested geometric error in pixels. **Patches per Body** limits refinement; the entire scene also has a hard limit of 4096 patches. The settings panel reports patch count, deepest level, cache draws/bakes and budget pressure. **FP64 Terrain Reference** bypasses the geometry cache for visual/performance comparison.

This is an opt-in geometry preview. It uses generated heights, not geographic imagery or measured elevations. Gas/ice giants, stars, black holes, spacecraft and bodies smaller than eight pixels in projected radius retain the existing renderer. Camera surface avoidance and picking still use the reference ellipsoid, so the camera can enter raised synthetic terrain. Set **Synthetic Relief** to zero when checking the reference surface at very low altitude.

## Coordinates and selection

`src/terrain/quadtree.hpp/.cpp` provides six outward-wound cube faces mapped onto an oblate ellipsoid. `TileId{face, level, x, y}` addresses a geographic region independently of GPU residency. The 64-bit key supports 24 subdivision levels without allocating a global finest-level grid. Local coordinates use the equatorial radius as one unit; the polar ratio is `1 - oblateness`, clamped to 0.05–1 for the preview. Height displaces along the ellipsoid normal. The synthetic relief amplitude is capped at one percent of the equatorial radius.

The body-fixed north axis follows the snapshot pole. Its reference east/west orientation comes from the projected render X axis (render Z near that singularity), followed by the snapshot spin. This is a rendering convention; geographic import must explicitly establish its prime meridian, latitude convention and height datum rather than assume that this basis matches a source map.

Selection refines the highest projected-error leaf first. The error estimate combines ellipsoid curvature with per-tile derivative bounds for the synthetic height function and a conservative distance to the displaced patch bounding sphere. It is specific to this generated surface. Real height data must supply minimum/maximum heights and geometric error metadata, including parent approximation error. The pixel setting is a refinement target, not a guarantee when the patch/depth limits are reached.

`tile_metadata.hpp/.cpp` bounds each sine/cosine phase over a tile's normalized-direction region, including interior extrema. Interval products provide normalized minimum/maximum heights; local gradient and curvature envelopes provide a second-derivative bound for selection. Metadata includes a `2e-5` normalized height allowance for float evaluation. The derivative envelopes describe the analytic signal, not the precision of a real height dataset or float trigonometric noise.

Displaced spheres shift their center to the midpoint height. Their radius includes the tile's base extent, half its local height range, and variation in the oblate surface normal. If an unshifted enclosure is smaller, it is retained instead. Fine patch height uncertainty consequently shrinks with extent instead of remaining the full global relief amplitude. Frustum tests use these spheres. Horizon culling retains a globally inscribed sphere of radius `polar_ratio - height_amplitude * (1 + RELIEF_ROUNDING)`; local mountains do not make a safe global occluder. It only rejects patch spheres entirely inside the occlusion cone and beyond its tangent distance. This intentionally retains extra patches on strongly flattened bodies. Horizon rejection is disabled inside the occluder.

`TileMetadataCache` is a bounded CPU LRU of 16,384 entries, with 32 bytes of metadata per entry plus container overhead. Entries are shared across body shapes, amplitudes and fixture generations because the current source is one fixed analytic function; shape/amplitude conversion happens during selection. Zero relief bypasses metadata lookup. Real source providers must add dataset identity/revision to this domain. Geometry remains in the separate 64 MiB GPU pool.

Neighbor lookups cross cube-face edges. Every refinement request recursively includes the coarser-neighbor splits needed to retain a maximum one-level difference between adjacent visible leaves. The complete split group is evaluated against the visible patch budget before any parent is removed. If it cannot fit, the current balanced selection remains intact and reports budget pressure. This reserves transition geometry during refinement and avoids a final ancestor-coarsening pass erasing local detail. Sixteen reusable index patterns collapse odd edge vertices against a coarser neighbor. Spatial morphing interpolates fine vertices from the parent grid triangles. Stitched edges use the coarser neighbor's morph level so both sides evaluate the same boundary. Budget-driven changes can still produce visible detail changes; there is no temporal transition history yet.

Selection is reused when generation, geometry revision, body-local eye, double-precision view rotation, FOV, framebuffer scale, relief, error target and available budget are unchanged. Moving views rebuild selection. An incrementally maintained leaf map supports neighbor lookup, split planning and final stitch masks. Queue entries for leaves already split to support a neighbor are discarded. Placement and lighting uniforms still update every frame.

Moving-ground validation exposed excessive balancing coarsening in the previous selector: an offset one-meter relief view shrank to 15–16 patches at LOD2–4. Balanced refinement replaces that behavior. Regressions require fine detail below and around the camera on smooth and raised surfaces, across interiors, poles and cube seams, plus coverage, budget and stitch invariants under small budgets. Tile-local height/error metadata further removes unnecessary refinement caused by the previous global height envelope. See [terrain performance](terrain_performance.md).

## Vulkan integration and memory

CPU physics, selection, patch anchors and camera/body transforms remain FP64. `GeometryCache` uses a bounded compute pass to evaluate changed patch vertices and normals in FP64, subtracts the body-local patch anchor before conversion, and stores FP32 local offsets, normals and surface directions. Native GPU FP64 is used when baking a missing patch, not on every near-ground draw. Synthetic height evaluation itself uses float trigonometry; high-precision placement does not make that signal survey-grade.

The near-ground draw shader contains no FP64 capability. It fetches cached offsets, reconstructs parent/grandparent triangles from the same grid for morphing and stitched edges, and uses CPU-split high/low eye-space anchors with FP32 transforms. The camera is subtracted before casting to floats; no float planet-sized position needs cancellation. Beyond four equatorial radii, only final eye-space placement/projection uses FP64. CPU eye-space anchors use `Camera::view_rotation_double()` so distant, narrow-angle views do not amplify rounded camera rotation into image drift. `terrain_reference.vert` retains the full procedural FP64 path and the same distant projection correction.

Selection uses local error envelopes, while shader morphing deliberately keeps the shared global coefficient. Assigning unrelated local coefficients to adjacent patches would make their shared-edge morphs disagree. The existing parent/grandparent interpolation and stitch masks therefore retain their common morph rule.

One 16 × 16 cell grid is shared by all patches. The 64 MiB geometry budget fits 4837 cached patches at 48 bytes × 289 vertices each. Reference draw records use 352 bytes and cached records 128 bytes, each with two frame buffers. Together with jobs and the sixteen index patterns, buffer payload is approximately 67.9 MiB, excluding allocator overhead and existing scene buffers. There are no terrain source textures in this milestone. Each precision/stitch group uses an instanced draw; at most 48 groups are needed for normal mixed cached/distant/fallback rendering. A diagnostic cached-FP64 benchmark mode can add a fourth group family.

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

The terrain benchmark compares the fixed sphere, full FP64 reference, cached FP32 and cached FP64 arithmetic on identical selected geometry. It reports CPU preparation/recording, GPU pass/draw timestamps, cache bakes, selection rebuilds, cold frames and a moving camera trace. The ground benchmark exercises stationary, walking and flying views over analytic relief at two starting positions, with fresh geometry generations, per-frame data and post-timing previews. The relief benchmark compares 0/20 km amplitudes at a fixed 2 km ellipsoid altitude and separately holds 2 km above the generated 20 km-relief surface. Metadata builds, retained entries and selection-node visits distinguish source-cache work from rendered patch count. Output paths must be new. These are isolated offscreen timings, not application FPS. See [terrain performance](terrain_performance.md) for measured results and limits.

## Next milestone

Add a preprocessed height/albedo tile pyramid and a bounded physical tile cache addressed by these tile IDs, with local height ranges and geometric errors for tighter selection. Geometry and material resolution must be selected independently. Resident parents must remain usable while children load; source identity/revision must participate in cache keys. Extend shared RHI support for batched, nonblocking mip/layer uploads and completion-based cache-slot reuse before connecting disk workers. The existing blocking `upload_image_2d` helper is not a streaming path.
