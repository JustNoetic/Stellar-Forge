# PROJECT_MAP.MD (LLM / AGENT FAST-INDEX)

> **Audience**: AI Agents / LLMs & Developers.
> **Purpose**: Instant routing index and shared boilerplate registry. Eliminates whole-codebase scans and duplicate Vulkan/UI/math code.
> **Format**: Dense, zero-prose, symbol-to-file lookup.

---

## 1. CRITICAL CONVENTIONS, CODING STYLE & INVARIANTS

| Domain | Rule / Invariant | References |
| :--- | :--- | :--- |
| **Zero Comment Clutter** | Do NOT litter `.hpp` / `.cpp` / `.glsl` files with docstrings, banner comments, or redundant inline comments. Keep code self-documenting and concise. Put architecture notes here or in `docs/`. | All `src/` and `shaders/` files |
| **Mandatory Shared Boilerplate** | NEVER write raw Vulkan buffer/image/pipeline/barrier/renderpass boilerplate or raw ImGui layout math inside feature modules. Always call the shared helpers in `sf::rhi`, `sf::paths`, `sf::math`, and `sf::ui`. | Section 2 below |
| **Graphics API** | Full **Vulkan 1.3** (`volk` + `VMA`). Dynamic Rendering (`vkCmdBeginRendering`), Synchronization2 (`vkCmdPipelineBarrier2`), Push Descriptors (`vkCmdPushDescriptorSetKHR`, `set = 0`), Buffer Device Address. No `VkRenderPass` or `VkFramebuffer`. | `src/rhi/vk_context.hpp`, `src/rhi/shader_cache.hpp` |
| **Depth & Projection** | **Reverse-Z** (`[1.0 -> 0.0]`, clear to `0.0f`, `VK_COMPARE_OP_GREATER_OR_EQUAL`) with infinite far plane and `GLM_FORCE_DEPTH_ZERO_TO_ONE`. | `sf::math::reverse_z_perspective`, `RenderingAttachment::clear_depth_reverse_z` |
| **State Ownership** | `Simulation` owns physical definitions, motion, and prepared render state. Consumers read `SystemSnapshot`; mutate through validated commands. `RenderSettingsStore`, `Camera`, `WorkspaceSettings`, and `WorkspaceState` have separate responsibilities. | `docs/state_ownership.md`, `src/ephemeris/simulation.hpp` |
| **Typed Boundaries** | Frame time: `units::Seconds`; simulation time: `units::Years`. Positions/velocities use distinct Ecliptic/Render types. Explicit `.value` only at numerical, UI, and GPU boundaries. | `src/core/units.hpp`, `src/core/coordinates.hpp` |
| **Definition Edits & Cache Keys** | Apply complete physical drafts atomically; reject stale generation/body revisions. Domain cache keys include generation + body identity + dynamics/geometry/lighting revision. Retain `SystemSnapshot::share_definition()` when caching definition pointer identity across independent snapshot lifetimes. | `src/ui/inspector_state.hpp`, `src/ephemeris/simulation.hpp` |
| **Coordinate Transform** | Physics is Ecliptic `(x, y, z)` in `double`. Render is `(x, z, -y)` (`Y` up, `-Z` forward). Always convert via `sf::ecliptic_to_render` / `sf::render_to_ecliptic`. | `src/core/constants.hpp` |
| **Units: Distance** | Interplanetary: `AU`. Radii / Altitudes: `km`. Conversion: `AU_TO_KM = 149597870.7`, `KM_TO_AU = 1.0 / AU_TO_KM`. | `src/core/constants.hpp` |
| **Units: Time & Mass** | Orbital physics: `years` (`SECONDS_PER_YEAR = 365.25 * 86400.0`). Frame delta: `seconds`. Mass in Solar Masses ($M_\odot$), `G_CONST = 39.476926421373`. | `src/core/constants.hpp` |
| **Path Resolution** | NEVER hardcode root paths. Use `sf::paths::bundled_path(...)` for read-only engine assets (`shaders/`) and `sf::paths::external_path(...)` for user/runtime data (`data/`, `textures/`, `cache/`). | `src/core/path_utils.hpp` |
| **Testing & Scratchpad** | Put local tests, debug scripts, or experiment artifacts in `tests/` (git-ignored). Use `StellarForge.exe --self-test` for headless GPU/shader verification. | `tests/`, `src/main.cpp` |

---

## 2. SHARED BOILERPLATE REGISTRY (USE THESE INSTEAD OF RAW CALLS)

| Need | Shared Helper / Symbol | Header |
| :--- | :--- | :--- |
| **Simulation State / Commands** | `simulation.snapshot()` (live const reference), `simulation.advance(units::Seconds{...})`, `simulation.apply(SetPaused / SetTimeWarp / SetTimeDirection / ReverseTimeDirection / ApplyBodyPhysicalProperties / ApplyBodyRings / AddBody / DeleteBody)` | `src/ephemeris/simulation.hpp` |
| **Time & Date Formatting** | `sf::time::calculate_sim_date`, `sf::time::format_sim_date`, `sf::time::format_time_speed` (Gregorian/UTC date math & original speed string thresholds) | `src/core/time_utils.hpp` |
| **Rendering Settings** | `RenderSettingsStore::values`, `apply`, `revision`: validated whole-settings update; camera owns no rendering controls | `src/rendering/render_settings.hpp` |
| **Inspector Drafts** | `BodyEditDraft::synchronize`, `reload`, `command`: draft edit -> Apply or Revert, with stale edit rejection; `RingEditState` for live multi-layer ring editing & texture baking | `src/ui/inspector_state.hpp` |
| **Create GPU Buffer (Device-Local)** | `vk.create_device_buffer(size, usage, initial_data)` | `src/rhi/vk_context.hpp` |
| **Create Host-Mapped Buffer (UBO/SSBO/Staging)** | `vk.create_host_buffer(size, usage, initial_data)` (`buf.mapped` persistently mapped, `buf.address` has `VkDeviceAddress`) | `src/rhi/vk_context.hpp` |
| **Flush Host Buffer Writes** | `vk.flush_buffer(buffer, offset, size)` (VMA no-op for coherent memory) | `src/rhi/vk_context.hpp` |
| **Copy GPU Buffer Ranges** | `vk.copy_buffer(cmd, source, destination, size, source_offset, destination_offset)`; caller supplies synchronization | `src/rhi/vk_context.hpp` |
| **Create 2D / Array / 3D Texture** | `vk.create_image_2d(w, h, format, usage, mips, layers)` / `vk.create_image_3d(...)` / `vk.upload_image_2d(img, pixels, bytes)` | `src/rhi/vk_context.hpp` |
| **Image Layout Transition (Sync2)** | `vk.transition_image(cmd, image, new_layout, src_stage, src_access, dst_stage, dst_access)` (auto-tracks `image.layout`) | `src/rhi/vk_context.hpp` |
| **Compute / Memory Barrier (Sync2)** | `vk.memory_barrier(cmd, src_stage, src_access, dst_stage, dst_access)` | `src/rhi/vk_context.hpp` |
| **Begin / End Dynamic Rendering Pass** | `vk.begin_rendering(cmd, extent, color_attachments, &depth_attachment)` / `vk.end_rendering(cmd)` | `src/rhi/vk_context.hpp` |
| **Bind Resources (Push Descriptors)** | `vk.push_descriptors(cmd, pipeline, { PushDescriptorWrite::uniform_buffer(...), storage_buffer(...), sampled_image(...), storage_image(...) })` | `src/rhi/vk_types.hpp`, `src/rhi/vk_context.hpp` |
| **Push Constants** | `vk.push_constants(cmd, pipeline, push_struct)` | `src/rhi/vk_context.hpp` |
| **One-Off GPU Transfer / Compute Submit** | `vk.immediate_submit([&](VkCommandBuffer cmd) { ... })` | `src/rhi/vk_context.hpp` |
| **Bounded Nonblocking Uploads** | `rhi::StagingUploads::buffer/image`, completion-based `UploadRing`, `vk.recording_serial/completed_serial`, `vk.transition_image_range` | `src/rhi/staging_uploads.hpp`, `src/rhi/upload_ring.hpp` |
| **Compile Compute Pipeline (with `#include` cache)** | `ShaderCache::create_compute_pipeline(vk, "path/foo.comp", bindings, sizeof(Push), macros)` | `src/rhi/shader_cache.hpp` |
| **Compile Graphics Pipeline (with `#include` cache)**| `ShaderCache::create_graphics_pipeline(vk, GraphicsPipelineDesc{...})` | `src/rhi/shader_cache.hpp` |
| **GPU Pass Timestamp Profiling** | `uint32_t id = vk.begin_gpu_timer(cmd, "gpu_pass"); ... vk.end_gpu_timer(cmd, id);` -> read via `vk.gpu_time_ms("gpu_pass")` | `src/rhi/vk_context.hpp` |
| **Opt-in Shader Compiler Statistics** | `STELLAR_SHADER_STATS=1`, `GraphicsPipelineDesc::capture_statistics`, `vk.pipeline_statistics_enabled/log_pipeline_statistics` | `src/rhi/vk_context.hpp`, `src/rhi/shader_cache.cpp` |
| **File & JSON Load / Save** | `sf::paths::read_text_file`, `read_binary_file`, `write_binary_file`, `read_json`, `write_json` | `src/core/path_utils.hpp` |
| **Logging & Fatal Assertions** | `sf::log::info(...)`, `sf::log::warn(...)`, `sf::log::error(...)`, `sf::log::fatal(...)`, `VK_CHECK(expr)` | `src/core/log.hpp`, `src/rhi/vk_types.hpp` |
| **ImGui Side Panels & Theme** | `ui.prepare_side_panel(settings, state, PanelSide::Left/Right, layout)` / `ui.compute_layout(...)` | `src/ui/workspace.hpp` |
| **ImGui Inspector Widgets** | `sf::ui::heading(...)`, `sf::ui::section(...)`, `sf::ui::value_row(...)`, `sf::ui::metric_text(...)`, `sf::ui::field(...)` | `src/ui/workspace.hpp` |

---

## 3. FEATURE / TASK ROUTING INDEX

| Feature / Subsystem | Primary Files & Documentation | Key Symbols / APIs | CLI Flags & Tests |
| :--- | :--- | :--- | :--- |
| **App Lifecycle & Main Loop** | `src/main.cpp`<br>`src/app.hpp/.cpp`<br>`run.bat` | `sf::App::init`, `sf::App::run`, `render_frame` | `--self-test` |
| **CPU/GPU Benchmarks** | `src/app_benchmark.cpp`<br>`src/benchmark/benchmark.hpp/.cpp`<br>`src/benchmark/audit_fixture.hpp`<br>`scripts/performance_audit.py`<br>`docs/performance_benchmarks.md` | `benchmark::Options`, `Run`, `FrameSample`, `write_report`, `host_info`; per-subsystem CPU/GPU scopes, streaming/LUT counters, procedural ring seed 42 | `--benchmark`<br>`--benchmark-detailed`<br>`--benchmark-scene audit`<br>`--benchmark-variant`<br>`StellarBenchmarkTests` |
| **Vulkan Context, VMA & Swapchain** | `src/rhi/vk_types.hpp`<br>`src/rhi/vk_context.hpp/.cpp` | `VkContext`, `Buffer`, `Image`, `Pipeline`, `RenderingAttachment`, `set_vsync` | `ctest` |
| **GLSL Preprocessor & SPIR-V Cache** | `src/rhi/shader_cache.hpp/.cpp` | `ShaderCache::load_and_preprocess`, `compile_to_spirv`, `create_compute_pipeline` | Opt-in `STELLAR_SHADER_STATS=1` |
| **Astronomical Constants & Transforms** | `src/core/constants.hpp` | `G_CONST`, `AU_TO_KM`, `SECONDS_PER_YEAR`, `OBLIQUITY`, `ecliptic_to_render`, `render_to_ecliptic` | Used by all math passes |
| **Conic Propagation & Reverse-Z** | `src/core/math_utils.hpp` | `stumpff_cs`, `propagate_conic_state`, `conic_time_at_angle`, `orbital_to_cartesian`, `reverse_z_perspective` | Numerical contracts |
| **Camera & Auto Horizon** | `src/core/camera.hpp/.cpp` | `Camera::update`, FP64 tracking/eye coordinates, near-surface spin follow; 8..12/s exponential horizon roll with exact final snap, `horizon_settling` / `horizon_roll_error_deg` telemetry | `StellarRenderingTests --camera-horizon-only` |
| **Path Resolution & File I/O** | `src/core/path_utils.hpp/.cpp` | `project_root`, `bundled_path`, `external_path`, `read_json`, `write_json` | Engine bootstrap |
| **ImGui Workspace & Widgets** | `src/ui/workspace.hpp/.cpp` | `UiSystem`, `WorkspaceSettings`, `WorkspaceLayout`, `heading`, `section`, `value_row` | Rendered in `App::render_frame` |
| **System Presets & Definitions** | `src/ephemeris/system_manager.hpp/.cpp` | `SystemManager::parse_system`, `serialize_definition`, `SystemDefinition`, `BodyDefinition`, `SystemSnapshot` | `StellarStateTests` |
| **Simulation State & Command Dispatch** | `src/ephemeris/simulation.hpp/.cpp`<br>`docs/state_ownership.md` | `Simulation`, `SimulationClock`, `SetPaused`, `SetTimeWarp`, `SetTimeDirection`, `SetMotionModel` | `StellarStateTests` |
| **IAS15 15th-Order Adaptive Integrator** | `src/ephemeris/ias15.hpp/.cpp`<br>`src/ephemeris/simulation.hpp` | `Ias15Integrator`, `MotionModel::NBody`, 1PN Relativity, $J_2/J_4$ oblateness, Gauss-Radau spacing | `StellarStateTests` |
| **Simulation Worker Thread** | `src/ephemeris/simulation_runner.hpp/.cpp`<br>`docs/performance_benchmarks.md` | `SimulationRunner`, triple snapshot buffers, async playback, command barriers | `--no-simulation-thread`<br>`StellarSimulationRunnerTests` |
| **Jacobi Coordinates & Precession** | `src/ephemeris/jacobi.hpp/.cpp` | `JacobiHierarchy`, `build_jacobi_hierarchy`, `extract_jacobi`, analytical $J_2$/3rd-body precession | `StellarOrbitTests` |
| **Dynamic Reference Hierarchy** | `src/ephemeris/reference_hierarchy.hpp/.cpp`<br>`docs/dynamic_hierarchy.md` | `ReferenceHierarchyTracker`, `SystemSnapshot::reference_hierarchy`, Hill-region AABB search (90/110% hysteresis) | `StellarHierarchyTests` |
| **SPICE Ephemeris Playback** | `src/ephemeris/spice_ephemeris.hpp/.cpp`<br>`data/ephemeris.json`<br>`docs/orbit_lines.md` | `MotionModel::Spice`, `SpiceEphemeris`, `SpiceKernelManager` (NAIF kernels, WinHTTP downloader, DAF/SPK inspector) | `STELLAR_TEST_EPHEMERIS=ON`<br>`StellarOrbitTests --spice` |
| **Orbit Lines & Hill Continuations** | `src/ephemeris/orbit_curves.hpp/.cpp`<br>`src/rendering/orbit_renderer.hpp/.cpp`<br>`docs/orbit_lines.md` | `compute_orbit_curves`, `OrbitRenderer`, FP64 conic evaluation, 1 px Bresenham lines, reverse-Z depth | `StellarOrbitRenderingTests` |
| **Order-Independent Transparency (OIT)** | `src/rendering/transparency_renderer.hpp/.cpp`<br>`shaders/transparency/*.comp`<br>`docs/transparency.md` | `TransparencyRenderer`, per-pixel linked lists (16-byte records), $T \cdot P + \alpha \cdot S$ atmosphere transport | `--transparency-only`<br>`--transparency-benchmark` |
| **Atmospheric Shadows & Deficit VRS** | `src/rendering/atmosphere_renderer.hpp/.cpp`<br>`shaders/atmosphere/shadow*.comp`<br>`shaders/common/atmosphere_shadow_weights.glsl`<br>`docs/atmosphere_shadows.md`<br>`docs/atmosphere_shadow_cached_weights.md`<br>`docs/atmosphere_ring_shadow_geometry.md` | Solar loss fraction deficit, software VRS, multi-scattering suppression, 64-frame progressive history; `atmosphere_shadow_cached_weights` default-on aerial-LUT derivative weights for eligible single-star ring shadows; `atmosphere_shadow_fast_rings` default-off experimental algebraic ring projection; `atmosphere_shadow_animate_jitter` advances per-frame noise without accumulation (default on, off freezes native comparisons); `atmosphere_shadow_rotation` guarded rotation-only sky reuse (single-star, exact unchanged eye/projection/scene, shared scratch, eight-frame prior weight + fresh sample); `ShadowHistoryReset` reasons and asynchronous `shadow_history_diagnostics` pixel/depth/reprojection counters | `--atmosphere-shadows`<br>`--atmosphere-progressive-only`<br>`--atmosphere-rotation-only` |
| **Atmosphere Radiative Transfer & Sky View**| `src/physics/atmosphere_physics.hpp`<br>`src/rendering/atmosphere_renderer.hpp/.cpp`<br>`docs/atmosphere.md` | Bruneton/Hillaire transmittance (256×64), multiple scattering (32×32), sky-view (192×108), aerial LUT (192×108×64) | `--atmospheres-only`<br>`--atmosphere-preview`<br>`--atmosphere-benchmark` |
| **Atmospheric Refraction** | `src/physics/atmosphere_refraction.hpp`<br>`src/rendering/refraction_tables.hpp`<br>`src/rendering/refraction_renderer.hpp/.cpp`<br>`shaders/common/refraction{,_solve}.glsl`<br>`shaders/refraction/observer_table.comp`<br>`docs/atmospheric_refraction_plan.md` | `PreparedRefraction` (on `PreparedAtmosphere::refraction`), `RefractionRenderer` (host selection, per-frame observer table), bindings 20 (UBO) / 21 (observer table), `refract_position` / `refract_direction` / `refract_point`, `RefractionFallback`, `refraction_enabled` | `StellarRefractionTests` (`refraction_contracts`, includes the shader-as-C++ emulation via `tests/glsl_shim.hpp`)<br>`StellarRenderingTests --refraction-only` |
| **Aerosol & Mie Particle Optics** | `src/physics/particle_optics.hpp`<br>`docs/atmosphere_particles.md` | `ParticleLayerDefinition`, Mie cross-sections, normalized RGB phase tables, schema v2 migration | `tests/atmosphere_migration_tests.py` |
| **Planetary Rings & Procedural Generator**| `src/rendering/ring_optics.hpp/.cpp`<br>`src/rendering/ring_generator.hpp/.cpp`<br>`src/rendering/ring_renderer.hpp/.cpp` | 3-param DHG phase, slab multiple scattering, 1D procedural generator (`generate_procedural_ring_profile`), 2D shadow LUTs | `--rings-only`<br>`--ring-benchmark`<br>`StellarRenderingTests` |
| **Ringshine Planet Illumination** | `src/rendering/ring_renderer.hpp/.cpp`<br>`shaders/post/ringshine_map.frag`<br>`docs/ringshine.md` | Single-pass 128×65 tile bake per (host × star), RGBA32F atlas (binding 16), 16-node warped-azimuth quadrature | `ringshine_enabled`<br>`docs/ringshine.md` |
| **Planetshine & Surface Ringshine** | `src/rendering/planetshine.hpp`<br>`src/physics/surface_material.hpp`<br>`src/rendering/surface_materials.hpp`<br>`shaders/common/{surface_material,surface_phase,ring_planetshine,ring_planetshine_disk}.glsl`<br>`shaders/celestial/{sphere,ring}.frag`<br>`docs/secondary_lighting.md` | Four material-aware sources per receiver; cached Lambert/Hapke phase curves; analytical ring lobes or 64-ray oblate host disk; full ring slab response; no atmospheric secondary scattering | `planetshine_enabled`, `ringshine_enabled`<br>`StellarRenderingTests --shine-only`, `--shine-materials`, `--shine-ring-phase`, `--shine-stability-only` |
| **Oblate Terrain LOD & Geometry Cache** | `src/terrain/{quadtree,geometry_cache,terrain_renderer}.hpp/.cpp`<br>`docs/terrain.md` | `TileId`, `SelectionView`, `GeometryCache`, `TerrainRenderer`, cube-face quadtree, FP32 patch LRU (64 MiB) | `--smooth-surfaces`<br>`--terrain-preview`<br>`StellarTerrainTests` |
| **Imported Elevation Pyramids** | `src/terrain/terrain_source.hpp/.cpp`<br>`src/terrain/height_stream.hpp/.cpp`<br>`docs/terrain.md` | `TerrainSource`, `HeightStream`, `TileAvailability`, 128² cubic B-spline DEM tiles with 2-texel gutters, resident virtual envelopes | `--terrain-imported`<br>`--terrain-source-benchmark` |
| **Surface Albedo & Normal Maps** | `src/terrain/surface_textures.hpp/.cpp`<br>`bake_textures.bat`<br>`scripts/bake_surface_maps.py` | `SurfaceTextures`, 14 albedo bodies, 128² cores + gutters, lossless sRGB/linear-light pyramid, RGBA16-SNORM normals | `--surface-preview`<br>`--terrain-normals`<br>`surface_map_contracts` |
| **Surface/Ring Eclipse Shadow Filtering** | `src/rendering/eclipse_candidates.hpp`<br>`shaders/common/eclipse_shadow.glsl`<br>`shaders/common/ring_shadow_filter.glsl`<br>`docs/eclipse_shadows.md`<br>`docs/ring_shadows.md` | `eclipse_could_shadow`, `ring_could_shadow_surface`, analytic circle overlap, 32-point projected ellipse quadrature; six-read ring blur with finite-star/distance scaling, curvature correction and `ring_shadow_softness` (0.25..4, default 1.25) | `--eclipse-optics`<br>`--eclipse-lighting`<br>`--surface-ring-shadows`<br>`StellarEclipseTests` |
| **Approximate Eclipse Refraction Lighting** | `src/physics/eclipse_shadow.hpp`<br>`src/rendering/eclipse_refraction_renderer.hpp/.cpp`<br>`shaders/common/eclipse_lookup.glsl`<br>`docs/eclipse_refraction.md` | Cached 256×64 log-RGB visibility tables with aligned umbra contacts, background analytical optical bake, binding 26, `eclipse_refraction_enabled`; shared surface/terrain/ring/reflected-point lighting; no per-pixel integration | `StellarEclipseShadowModelTests`<br>`StellarRenderingTests --eclipse-refraction <directory>`<br>`--eclipse-refraction-benchmark <directory>`<br>`--eclipse-refraction-preview <directory>`<br>`--eclipse-transition <directory>` |
| **Gaia DR3 Starfield (~460k Stars)** | `src/rendering/star_catalog.hpp/.cpp`<br>`shaders/celestial/starfield*.{comp,vert,frag}` | `StarCatalog`, `StarGpuRecord`, compute culling pre-pass, indirect draw (`vkCmdDrawIndirect`), compact 3 px PSF | Headless & self-test |
| **Celestial Mesh / Subpixel Transitions** | `src/rendering/celestial_renderer.hpp/.cpp`<br>`shaders/common/point_source.glsl` | `mesh_weight` (1.5–2.5 px transition), `point_profile_sum` flux normalization, stellar disc visibility | `docs/point_sources.md` |
| **HDR Exposure & ACES Tone Mapping** | `shaders/common/tonemap.glsl`<br>`shaders/post/composite.frag` | `aces_tonemap`, `linear_to_srgb`, exposure slider, presentation composite pass | Fullscreen post-process |
| **Astrophysical Star Creator** | `src/physics/star_calc.hpp/.cpp`<br>`src/ui/inspector_state.hpp` | `StarCalculator`, `StarParameters`, `StarForgeResult`, Roche rotation, Stefan-Boltzmann, blackbody Planckian RGB | `StellarStarCalcTests` |

---

## 4. DIRECTORY MANIFEST

```text
Stellar Forge Rewrite/
├── CMakeLists.txt                  # Build configuration (Vulkan 1.3, GLFW, GLM, ImGui, glslang; outputs: bin/, lib/)
├── run.bat                         # Ninja parallel build and execution script (build\bin\StellarForge.exe)
├── bake_textures.bat               # Interactive surface/ring tile baker wizard (calls scripts\bake_surface_maps.py)
├── project_map.md                  # LLM / Agent fast-index and shared boilerplate registry
├── src/                            # Application and engine core C++ source
│   ├── main.cpp                    # Application entry point & CLI parser
│   ├── app.hpp / app.cpp           # Window lifecycle, frame loop, dynamic rendering pass coordination
│   ├── app_benchmark.cpp           # Headless/interactive benchmark driver (--benchmark)
│   ├── core/                       # Units, coordinates, camera, math, timing, paths, logging
│   ├── ephemeris/                  # IAS15 N-body, SPICE ephemeris, Jacobi, reference hierarchy, orbit curves
│   ├── physics/                    # Star thermodynamics & astrophysics, atmosphere/particle optics
│   ├── rendering/                  # Celestial, atmosphere, rings, orbit, starfield, transparency renderers
│   ├── rhi/                        # Vulkan 1.3 context, VMA, sync2 barriers, dynamic rendering, shader cache
│   ├── terrain/                    # Quadtree LOD, metadata, residency, geometry cache, surface textures
│   ├── ui/                         # Dear ImGui workspace, layout, and inspector state drafts
│   └── benchmark/                  # Benchmark metrics, host info, and report generation
├── tests/                          # Local contract tests, benchmarks & scratch experiments (git-ignored)
│   ├── CMakeLists.txt              # Test build configuration (isolated output to build/bin/tests/)
│   ├── *.cpp                       # CPU/GPU contract test drivers (state, orbit, terrain, rendering, etc.)
│   ├── *.hpp                       # GPU rendering test headers (atmosphere, rings, shadows, etc.)
│   ├── *.py                        # Python fixture and contract scripts
│   └── fixtures/                   # JSON reference fixtures for orbits, SPICE, and terrain
├── shaders/                        # GLSL compute, vertex, and fragment shaders (compiled to SPIR-V)
│   ├── atmosphere/                 # Transmittance, multiple scattering, sky-view, aerial perspective, shadows
│   ├── celestial/                  # Celestial spheres, point billboards, orbit lines, starfield cull/draw
│   ├── common/                     # Shared GLSL includes (radiance, optics, reverse-Z, noise, tone mapping)
│   ├── post/                       # Post-processing, ringshine tile bakes, HDR composite tonemapping
│   ├── terrain/                    # Terrain geometry cache compute and vertex shaders
│   ├── transparency/               # Per-pixel A-buffer clear and O(n log n) resolve passes
│   └── tests/                      # Dedicated shader probes used by GPU contract test suites
├── scripts/                        # Maintenance, offline processing, texture baking, and benchmarking
│   ├── bake_surface_maps.py        # Tile pyramid baker for planetary albedo and height maps
│   ├── cuda_tile_sampler.py        # CuPy CUDA-accelerated surface map sampler
│   ├── import_terrain.py           # Terrain DEM importer and pyramid generator
│   └── ...                         # Atmosphere migrations, Horizons ephemeris fetchers, benchmark tools
├── data/                           # User & runtime data (systems/, gaia/stars.bin, ephemeris configs)
├── textures/                       # Planetary surface maps and ring textures
├── docs/                           # Deep-dive architecture and design specifications
├── build/                          # CMake / Ninja build tree (bin/ for app, bin/tests/ for tests, lib/)
└── dist/                           # Clean self-contained deployment staging (created via cmake --install)
```

---

## 5. FEATURES & ROADMAP (ORIGINAL REPOSITORY: `D:\Software\Stellar Forge`)

> **Migration Status**: The core migration and architectural rewrite from the original codebase (`D:\Software\Stellar Forge`) is **COMPLETE**.
> Subsequent development proceeds sequentially feature-by-feature rather than through migration phases.

### Implemented Feature Set
- **Vulkan 1.3 Baseline**: Dynamic Rendering (`vkCmdBeginRendering`), Synchronization2, Push Descriptors, runtime GLSL preprocessor & SPIR-V disk cache (`ShaderCache`), Reverse-Z depth.
- **Precision Ephemeris**: IAS15 15th-order adaptive Gauss-Radau integrator (1PN GR + $J_2/J_4$ oblateness), NASA CSPICE playback (18 default NAIF kernels + dynamic download), shared Jacobi coordinate hierarchy, 15 Hz distance-sorted reference hierarchy tracker with Hill entry/exit hysteresis, and threaded simulation runner (`SimulationRunner`).
- **Astrophysics & Stars**: Gaia DR3 catalog (~475k stars) with compute indirect draw, Star Creator (`StarCalculator` evolution tracks, Stefan-Boltzmann, Roche rotation, Planckian RGB).
- **Planetary Rings & Optics**: Multi-layer rings, 3-parameter DHG phase, slab multiple scattering, 1D procedural band generator, 2D prefix shadow LUTs, and ringshine tile bakes (128×65 RGBA32F).
- **Physical Atmospheres**: Bruneton/Hillaire transmittance, multiple scattering, sky-view, and 3D aerial perspective LUTs; schema v2 aerosol Mie scattering; adaptive shadow loss deficit with software VRS and 64-frame progressive accumulation.
- **Secondary Lighting**: Shared Lambert/Hapke surface photometry and cached phase curves on ground, body billboards and planetshine; analytical or 64-ray oblate host illumination of rings; ringshine on host surfaces. Secondary atmospheric scattering is removed.
- **Oblate Terrain**: Native cube-face quadtree LOD with spatial morphs, 64 MiB GPU geometry cache, 128² nested signed FP32 DEMs with cubic B-spline normals, and lossless sRGB albedo / RGBA16-SNORM normal tile streaming.
- **Rendering Optics & Order-Independent Transparency**: Per-pixel linked-list A-buffer (16-byte records, early reverse-Z cull), analytical circle/ellipse eclipse shadow quadrature on bodies and rings.
- **Eclipse Refraction Lighting**: Approximate RGB atmosphere extinction and refracted eclipse fill on surfaces, terrain, rings and reflected points; one filtered log-visibility lookup, background baking and bounded cache. See `docs/eclipse_refraction.md`.

### Planned Features (Next Sequential Additions)
- **Atmospheric Refraction**: Ray-bending in planetary atmospheres during horizon crossings and ground observations.
- **Eclipse Limb Images**: Glowing atmospheric ring and secondary stellar images seen from within an eclipse shadow. Incident-light eclipse colouring is implemented; per-fragment annulus integration remains rejected (`docs/eclipse_refraction_assessment.md`).
- **System Comparator**: Side-by-side comparative inspection and orbital overlay of multiple planetary systems.
- **Object Size Comparison**: Interactive 3D comparative sizing mode for planets, moons, and stars.
- **5-Cascade FFT Oceans**: Dynamic ocean surface displacement and foam compute passes (`water/*.comp`).
- **Convolution Diffraction Bloom**: Physically based optical flare and aperture diffraction.
- **High-Resolution Exporter**: Tiled 4K/8K/16K screenshot capture and presentation output.
- **Spacecraft Ephemeris & Frames**: Sampled polyline trajectories and Lagrange reference frames.
- **Paged Terrain Indexes**: High-resolution DEM indexing and terrain collision queries.
