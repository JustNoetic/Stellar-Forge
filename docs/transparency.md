# Transparent surfaces and atmosphere

Rings and orbit lines use a shared per-pixel linked-list A-buffer. Opaque spheres/terrain retain reverse-Z depth writes. Transparent rasterization performs early opaque depth tests, keeps depth writes disabled, and appends visible fragments instead of blending them immediately. Each 16-byte record contains half-float premultiplied HDR RGBA, reverse-Z depth, and the next record index. Rings retain their existing optical shading, overlapping coplanar radial layers, filtering and eclipse/shadow evaluation; orbit lines retain their rasterization, fade and display colors.

`TransparencyRenderer::prepare` clears an R32_UINT head image and the allocation counters before scene rendering. `CelestialRenderer` supplies the same head image and fragment buffer to rings and the application orbit pass. `composite_atmospheres` transports the opaque/background HDR image and each stored fragment, then resolves transparent surfaces in per-pixel depth order. An in-place linked-list merge sort uses O(n log n) work and no fixed per-pixel layer truncation. Equal-depth records use packed color as a deterministic tie break. Empty pixels skip fragment sorting and HDR read/write in the resolve pass.

## Ring layer slot

Ring views usually hold about one transparent fragment per pixel, and almost all of it is ring. Sending that through the list cost a 16-byte record, a separate transparent-transport dispatch and a resolve read for every ring pixel. So rings first try a fixed per-pixel slot (`transparency.glsl`, `TRANSPARENCY_RING_SLOT`): an RGBA16F colour image at binding 45 and an R32UI depth image at binding 46, cleared to 0 in `prepare`.

- **Claim.** `capture_ring` claims the slot with a compare-and-swap of its reverse-Z depth bits against 0. A ring fragment that loses the claim sets `RING_SLOT_CONTENDED` (bit 31) and is appended to the list as usual. Reverse-Z depth stays below 2.0, so the top two bits are free.
- **Contended slots** are treated as one more list fragment everywhere: transport, shadow prefix and resolve sort. Overlapping ring layers therefore order exactly as before.
- **Transmittance storage.** The slot stores transmittance `1-a`, not alpha. Dense rings sit at alpha near 1, where half floats quantize `1-a` coarsely (one ulp is 4.9e-4); at high exposure that doubled the light seen through them. Readers use `load_ring_slot` / `store_ring_slot`.
- **Fused transport.** The atmosphere composite (both paths, macro `ATMOSPHERE_RING_SLOT`, mode in `CompositePush::ring`) transports an uncontended, visible slot in the same invocation that composites the background. It reuses the pixel's ray setup and shadow lookup and applies the behind-cloud rewrite `(1-c)P + a·C`.
- **Fused blend.** When the pixel has no list fragments and this is the last atmosphere drawn, the dual-source pass also blends the ring straight in: `scattered = P' + (1-a)S` and `transmitted = (1-a)T`. Otherwise it writes the transported slot back.
- **Resolve.** The resolve then takes slot mode 1 (merge the slot only where list fragments exist) or mode 2 (merge it everywhere). The slot enters the sort under the marker key `RING_SLOT_KEY`, so its colour stays fp32 and is never repacked to half.
- **Several atmospheres.** With more than one atmosphere, dual-source draws before the last are separated by a fragment storage barrier, because each draw may rewrite the slot.
- **Lean pass misses.** A contended slot that misses the shadow endpoint cache in the lean pass is marked with `RING_SLOT_DEFERRED` (bit 30) for the deferred pass, the same way a list record is marked with its depth sign bit.

The shadow pass picks the slot as the transparent prefix endpoint when it is uncontended and visible. The nearest list fragment, usually an orbit line, then takes the cloud layer's endpoint when no cloud occupies it. Under the closed form, that endpoint only gets a slot when its path is under `CLOSED_FORM_CLOUD_SPLIT_SHARE` of the primary path. Endpoints past atmosphere exit all see the full column, so their deficits are stored and looked up under the key 0 rather than their own depth. This lets an opaque background and a ring beyond the atmosphere share one cache entry instead of missing to the deferred pass.

At 2560×1440 on the RTX 4060 Ti, mean GPU frame time against the list-only path:

| Scene | Before | After | Transparent transport | Resolve |
| --- | ---: | ---: | ---: | ---: |
| `earth-sky-rings` | 4.50 ms | 3.75 ms | 0.85 → 0.14 ms (deferred 0.27 → 0.00) | 0.28 → 0.11 ms |
| `earth-sky-rings-south` | 3.65 ms | 3.13 ms | 0.61 → 0.06 ms | — |
| `saturn-limb` | 3.14 ms | 2.90 ms | — | — |
| `saturn` | 2.72 ms | 2.56 ms | — | — |

The opaque composite grows by 0.08–0.23 ms because it now carries the ring. `earth-full-rings` and `earth-ground-rings` read 0.05–0.1 ms slower; this was not separated from run-to-run noise. A floor of about 0.06 ms remains for the transparent dispatch on pixels with no list fragments, and about 0.08 ms for the resolve, which includes the bloom physical-light copy.

Images were compared against the list path with `--benchmark-capture`:

- Earth scenes match within 2 half-float ulps.
- Remaining differences come from three sources:
  - the finer transmittance and fp32 slot colour; the old path also flushed denormal ring colours to 0;
  - a shifted shadow prefix where orbit lines cross rings;
  - hidden refracted ring fragments, which the old path could take as the prefix endpoint and the new one does not.

## Atmospheric transport

When bloom is enabled, an optional resolve output excludes orbit guides from
the physical-light image while retaining normal presentation and transport.
Guide records use the sign bit of packed alpha as a marker; decoded opacity
remains positive, transport preserves the marker, and sorting ignores it.
See `bloom.md`.

Atmospheric extinction remains RGB, including on GPUs using hardware dual-source blending. For a surface at distance d, let P be premultiplied surface radiance, alpha its opacity, T(d) the RGB transmission from the camera to the surface, and S(d) the atmospheric scattered radiance over that same interval. Before surface blending, transform its stored color to:

`P_camera = T(d) * P + alpha * S(d)`

Keep alpha unchanged. Blend transformed surfaces far to near over the atmosphere-composited opaque/background image. This is algebraically equivalent to transporting each intervening atmospheric interval under the existing medium model: foreground air is added once, while each surface attenuates radiance and atmospheric scattering behind it. Adding S without multiplying by alpha would add fog repeatedly through translucent surfaces.

Transport reuses the same finite-endpoint reconstruction and horizon-mapped cumulative LUT functions as opaque aerial perspective. Foreground fragments before atmospheric entry remain unchanged; fragments within an atmosphere receive a partial column; fragments beyond it receive the full bounded column. Camera-relative positions, inverse projection, exposure, colored extinction and optical cache invalidation remain shared. No new per-pixel atmospheric ray integration or LUT rebaking is introduced. Atmospheric shadow VRS reconstructs both the primary and the nearest-transparent prefix endpoint from a two-layer coarse grid (see `atmosphere_shadows.md`), falling back to native integration of both when either fails its guards; farther transparent endpoints keep the existing fallback.

With atmospheric shadows active, transparent transport runs in two dispatches. The lean pass (`ATMOSPHERE_TILE_LEAN`) reads shadow deficits only from the shadow pass's endpoint cache: primary, nearest-transparent prefix and cloud layers. It marks fragments whose depth misses that cache by setting the sign bit of the record's reverse-Z depth; fragments behind a 2D cloud whose own endpoint misses are marked too. It leaves those fragments untouched and lists their 8×8 tile. The full shader (`ATMOSPHERE_TILE_DEFERRED`) then runs on the listed tiles by indirect dispatch. It transports only marked fragments and clears the bit before the transparency resolve sorts by depth. The inline fallback integration had cost occupancy even when it never ran: in `earth-sky-rings` (2560×1440, `no-orbits`) the pass went from 1.25 to 0.44 ms with identical images. Orbit lines crossing rings are the usual misses (about 0.16 ms of deferred work in that view with orbits shown). The non-dual-source `composite_transparency` path is unchanged.

Complete columns reuse the sky radiance and RGB transmission instead of reevaluating finite-distance scattering. A surface beyond atmosphere exit therefore cannot introduce a brightness seam against empty sky with the same background radiance. Disabling aerial perspective suppresses the surface inscattering term while retaining finite RGB extinction for rings, orbit lines and opaque surfaces; no aerial scattering volume needs to be baked for this mode.

The compute atmosphere path transforms stored fragments in the same dispatch as opaque/background composition. With dual-source blending selected and supported, a compute pass transforms only pixels containing stored fragments; the existing hardware dual-source pass still composites the opaque/background image. Both paths then use the same surface resolver. Atmospheric passes preserve their existing far-to-near ordering.

## Bounds and lifetime

Fragment allocation targets two fragments per viewport pixel, capped at 128 MiB of records and the device storage-buffer range. The separate head image uses four bytes per pixel, and the ring slot another twelve (RGBA16F colour, R32UI depth); the atmosphere renderer binds 1×1 dummies when no slot exists. Steady allocation is approximately 71.2 MiB at 1920x1080 and 159.6 MiB at 3840x2160. Allocation is lazy, retained at its high-water mark, and skipped when no visible clouds, rings or multi-body orbit pass require it. Resized images and grown buffers retire after GPU completion; normal frames introduce no queue wait.

2D cloud sheets composited inline by the atmosphere pass (the default; see
`oceans_clouds.md`) do not use the fragment list, and do not enable capture. Fragments behind such a
cloud are rewritten during atmospheric transport so they resolve beneath it. 2D sheets on the
fallback path still enable capture when rings and orbit lines are disabled.
They append the premultiplied radiance and opacity of their single shell
crossing at its exact depth. Their enclosing mesh is only a ray proxy, so clouds
use analytically reconstructed crossing depth instead of early testing that
mesh's far-side depth. The existing opaque-depth checks in atmospheric
composition and resolve filter the captured layer. Hardware without fragment
atomics and the legacy reference mode write fully opaque crossings to depth in
a separate depth-only pass.

When the global record budget is exhausted, excess fragments use ordinary draw-order blending into the background. They remain visible, but ordering and their atmospheric endpoint are approximate in this overload condition. A GPU overflow counter, asynchronously copied to frame-owned host buffers, exposes this condition through `TransparencyRenderer::stats`; the renderer logs when overflow begins. Hardware without fragment storage atomics uses the legacy renderer and logs that fallback. `RenderSettings::transparency_reference_legacy` exists for controlled comparisons. Existing batched terrain/ring raster microbenchmarks explicitly use this legacy raster path; their timings omit the new resolve. The transparency benchmark measures the complete new path.

The current transport still treats overlapping atmospheres as independently composed columns. Physically overlapping participating media require combined extinction/source evaluation. Cumulus currently approximates an integrated column as one weighted transparent layer: this improves atmospheric endpoints without a duplicate march, but does not exactly represent rings/lines embedded within the cloud volume or overlapping media. Those cases still require combined-medium transport. Cloud ground shadows remain disabled.

## Verification

`StellarRenderingTests --transparency-only` verifies crossing surfaces, bitwise draw-order independence, opaque reverse-Z rejection, 48 layers at one pixel, finite RGB extinction and alpha-weighted inscattering against independent opaque-endpoint renders, foreground vacuum, the horizon-mapped aerial LUT, both blend paths, viewport resize and explicit overflow fallback. `StellarOrbitRenderingTests` exercises the real orbit capture shader, compares legacy coverage, and retains its occlusion/culling/large-world checks. Existing ring and atmosphere suites exercise their normal capture/composition paths.

`StellarRenderingTests --atmosphere-endpoints-only` also runs in the transparency and atmosphere suites. It checks extinction with aerial perspective disabled against analytic RGB columns and checks complete-column brightness continuity for opaque and translucent surfaces in both blend modes.

`StellarRenderingTests --transparency-benchmark scratch/transparency/benchmark.json` compares legacy and ordered paths with GPU timestamps on 1080p/4K empty, sparse and fullscreen surface cases, with/without atmosphere and hardware dual-source blending, plus actual lit ring geometry. It uses cached LUTs, three alternating-order repetitions and 8 warmup/24 measured frames per mode. Results exclude presentation/readback and are not application FPS.

On the RTX 4060 Ti, the warmed fixture measured the following added GPU time over the legacy path. These are offscreen fixture costs, not hardware-independent guarantees; clocks and scene coverage affect the result. No benchmark case overflowed.

| Case | 1920x1080 | 3840x2160 |
| --- | ---: | ---: |
| Lit ring geometry | 0.06 ms | 0.22 ms |
| 1% surface coverage inside atmosphere, compute | 0.09 ms | 0.30 ms |
| 1% surface coverage inside atmosphere, dual source | 0.11 ms | 0.37 ms |
| Fullscreen surface inside atmosphere, compute stress case | 0.65 ms | 2.66 ms |
| Fullscreen surface inside atmosphere, dual-source stress case | 0.64 ms | 2.56 ms |
