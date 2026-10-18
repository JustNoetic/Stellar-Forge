# Transparent surfaces and atmosphere

Rings and orbit lines use a shared per-pixel linked-list A-buffer. Opaque spheres/terrain retain reverse-Z depth writes. Transparent rasterization performs early opaque depth tests, keeps depth writes disabled, and appends visible fragments instead of blending them immediately. Each 16-byte record contains half-float premultiplied HDR RGBA, reverse-Z depth, and the next record index. Rings retain their existing optical shading, overlapping coplanar radial layers, filtering and eclipse/shadow evaluation; orbit lines retain their rasterization, fade and display colors.

`TransparencyRenderer::prepare` clears an R32_UINT head image and the allocation counters before scene rendering. `CelestialRenderer` supplies the same head image and fragment buffer to rings and the application orbit pass. `composite_atmospheres` transports the opaque/background HDR image and each stored fragment, then resolves transparent surfaces in per-pixel depth order. An in-place linked-list merge sort uses O(n log n) work and no fixed per-pixel layer truncation. Equal-depth records use packed color as a deterministic tie break. Empty pixels skip fragment sorting and HDR read/write in the resolve pass.

## Atmospheric transport

Atmospheric extinction remains RGB, including on GPUs using hardware dual-source blending. For a surface at distance d, let P be premultiplied surface radiance, alpha its opacity, T(d) the RGB transmission from the camera to the surface, and S(d) the atmospheric scattered radiance over that same interval. Before surface blending, transform its stored color to:

`P_camera = T(d) * P + alpha * S(d)`

Keep alpha unchanged. Blend transformed surfaces far to near over the atmosphere-composited opaque/background image. This is algebraically equivalent to transporting each intervening atmospheric interval under the existing medium model: foreground air is added once, while each surface attenuates radiance and atmospheric scattering behind it. Adding S without multiplying by alpha would add fog repeatedly through translucent surfaces.

Transport reuses the same finite-endpoint reconstruction and Bruneton/frustum LUT functions as opaque aerial perspective. Foreground fragments before atmospheric entry remain unchanged; fragments within an atmosphere receive a partial column; fragments beyond it receive the full bounded column. Camera-relative positions, inverse projection, exposure, colored extinction and optical cache invalidation remain shared. No new per-pixel atmospheric ray integration or LUT rebaking is introduced.

The compute atmosphere path transforms stored fragments in the same dispatch as opaque/background composition. With dual-source blending selected and supported, a compute pass transforms only pixels containing stored fragments; the existing hardware dual-source pass still composites the opaque/background image. Both paths then use the same surface resolver. Atmospheric passes preserve their existing far-to-near ordering.

## Bounds and lifetime

Fragment allocation targets two fragments per viewport pixel, capped at 128 MiB of records and the device storage-buffer range. The separate head image uses four bytes per pixel. Steady allocation is approximately 71.2 MiB at 1920x1080 and 159.6 MiB at 3840x2160. Allocation is lazy, retained at its high-water mark, and skipped when neither rings nor a multi-body orbit pass is enabled. Resized images and grown buffers retire after GPU completion; normal frames introduce no queue wait.

When the global record budget is exhausted, excess fragments use ordinary draw-order blending into the background. They remain visible, but ordering and their atmospheric endpoint are approximate in this overload condition. A GPU overflow counter, asynchronously copied to frame-owned host buffers, exposes this condition through `TransparencyRenderer::stats`; the renderer logs when overflow begins. Hardware without fragment storage atomics uses the legacy renderer and logs that fallback. `RenderSettings::transparency_reference_legacy` exists for controlled comparisons. Existing batched terrain/ring raster microbenchmarks explicitly use this legacy raster path; their timings omit the new resolve. The transparency benchmark measures the complete new path.

The current transport still treats overlapping atmospheres as independently composed columns. Physically overlapping participating media require combined extinction/source evaluation. Future volumetric clouds must use that combined-medium model; a single cloud surface or shell would not represent embedded rings/lines correctly. This change addresses camera visibility, not new atmosphere/ring/cloud sunlight shadowing.

## Verification

`StellarRenderingTests --transparency-only` verifies crossing surfaces, bitwise draw-order independence, opaque reverse-Z rejection, 48 layers at one pixel, finite RGB extinction and alpha-weighted inscattering against independent opaque-endpoint renders, foreground vacuum, both atmospheric LUT models, both blend paths, viewport resize and explicit overflow fallback. `StellarOrbitRenderingTests` exercises the real orbit capture shader, compares legacy coverage, and retains its occlusion/culling/large-world checks. Existing ring and atmosphere suites exercise their normal capture/composition paths.

`StellarRenderingTests --transparency-benchmark scratch/transparency/benchmark.json` compares legacy and ordered paths with GPU timestamps on 1080p/4K empty, sparse and fullscreen surface cases, with/without atmosphere and hardware dual-source blending, plus actual lit ring geometry. It uses cached LUTs, three alternating-order repetitions and 8 warmup/24 measured frames per mode. Results exclude presentation/readback and are not application FPS.

On the RTX 4060 Ti, the warmed fixture measured the following added GPU time over the legacy path. These are offscreen fixture costs, not hardware-independent guarantees; clocks and scene coverage affect the result. No benchmark case overflowed.

| Case | 1920x1080 | 3840x2160 |
| --- | ---: | ---: |
| Lit ring geometry | 0.06 ms | 0.22 ms |
| 1% surface coverage inside atmosphere, compute | 0.09 ms | 0.30 ms |
| 1% surface coverage inside atmosphere, dual source | 0.11 ms | 0.37 ms |
| Fullscreen surface inside atmosphere, compute stress case | 0.65 ms | 2.66 ms |
| Fullscreen surface inside atmosphere, dual-source stress case | 0.64 ms | 2.56 ms |
