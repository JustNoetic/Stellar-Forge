# PROJECT_MAP.MD (LLM / AGENT FAST-INDEX)

> **Audience**: AI Agents / LLMs & Developers.
> **Purpose**: Instant routing index and shared boilerplate registry. Eliminate whole-codebase scans and duplicate Vulkan/UI/math code.
> **Format**: Dense, zero-prose, symbol-to-file lookup.

---

## 1. CRITICAL CONVENTIONS, CODING STYLE & INVARIANTS

| Domain | Rule / Invariant | References |
| :--- | :--- | :--- |
| **Zero Comment Clutter** | Do NOT litter `.hpp` / `.cpp` / `.glsl` files with docstrings, banner comments, or redundant inline comments. Keep code self-documenting and concise. Put architecture notes here or in `docs/`. | All `src/` and `shaders/` files |
| **Mandatory Shared Boilerplate** | NEVER write raw Vulkan buffer/image/pipeline/barrier/renderpass boilerplate or raw ImGui layout math inside feature modules. Always call the shared helpers in `sf::rhi`, `sf::paths`, `sf::math`, and `sf::ui`. | Section 2 below |
| **Graphics API** | Full **Vulkan 1.3** (`volk` + `VMA`). Dynamic Rendering (`vkCmdBeginRendering`), Synchronization2 (`vkCmdPipelineBarrier2`), Push Descriptors (`vkCmdPushDescriptorSetKHR`, `set = 0`), Buffer Device Address. No `VkRenderPass` or `VkFramebuffer`. | `src/rhi/vk_context.hpp`, `src/rhi/shader_cache.hpp` |
| **Depth & Projection** | **Reverse-Z** (`[1.0 -> 0.0]`, clear to `0.0f`, `VK_COMPARE_OP_GREATER_OR_EQUAL`) with infinite far plane and `GLM_FORCE_DEPTH_ZERO_TO_ONE`. | `sf::math::reverse_z_perspective`, `RenderingAttachment::clear_depth_reverse_z` |
| **State Ownership** | `Simulation` owns physical definitions, motion and prepared render state. Consumers read `SystemSnapshot`; mutate through validated commands. `RenderSettingsStore`, `Camera`, `WorkspaceSettings`, and `WorkspaceState` have separate responsibilities. | `docs/state_ownership.md`, `src/ephemeris/simulation.hpp` |
| **Typed Boundaries** | Frame time: `units::Seconds`; simulation time: `units::Years`. Positions/velocities use distinct Ecliptic/Render types. Explicit `.value` only at numerical, UI and GPU boundaries. | `src/core/units.hpp`, `src/core/coordinates.hpp` |
| **Definition Edits & Cache Keys** | Apply complete physical drafts atomically; reject stale generation/body revision. Domain cache keys include generation + body identity + dynamics/geometry/lighting revision. No change means no invalidation. | `src/ui/inspector_state.hpp`, `src/ephemeris/simulation.hpp` |
| **Coordinate Transform** | Physics is Ecliptic `(x, y, z)` in `double`. Render is `(x, z, -y)` (`Y` up, `-Z` forward). Always convert via `sf::ecliptic_to_render` / `sf::render_to_ecliptic`. | `src/core/constants.hpp` |
| **Units: Distance** | Interplanetary: `AU`. Radii / Altitudes: `km`. Conversion: `AU_TO_KM = 149597870.7`, `KM_TO_AU = 1.0 / AU_TO_KM`. | `src/core/constants.hpp` |
| **Units: Time & Mass** | Orbital physics: `years` (`SECONDS_PER_YEAR = 365.25 * 86400.0`). Frame delta: `seconds`. Mass in Solar Masses ($M_\odot$), `G_CONST = 39.476926421373`. | `src/core/constants.hpp` |
| **Path Resolution** | NEVER hardcode root paths. Use `sf::paths::bundled_path(...)` for read-only engine assets (`shaders/`) and `sf::paths::external_path(...)` for user/runtime data (`data/`, `textures/`, `cache/`). | `src/core/path_utils.hpp` |
| **Agent Scratchpad** | Put temporary test/debug scripts or binaries in `scratch/` (git-ignored). Use `StellarForge.exe --self-test` for headless GPU/shader verification. | `scratch/`, `src/main.cpp` |

---

## 2. SHARED BOILERPLATE REGISTRY (USE THESE INSTEAD OF RAW CALLS)

| Need | Shared Helper / Symbol | Header |
| **Simulation State / Commands** | `simulation.snapshot()` (live const reference), `simulation.advance(units::Seconds{...})`, `simulation.apply(SetPaused / SetTimeWarp / SetTimeDirection / ReverseTimeDirection / ApplyBodyPhysicalProperties / AddBody / DeleteBody)` | `src/ephemeris/simulation.hpp` |
| **Time & Date Formatting** | `sf::time::calculate_sim_date`, `sf::time::format_sim_date`, `sf::time::format_time_speed` (Gregorian/UTC date math & original speed string thresholds) | `src/core/time_utils.hpp` |
| **Rendering Settings** | `RenderSettingsStore::values`, `apply`, `revision`: validated whole-settings update; camera owns no rendering controls | `src/rendering/render_settings.hpp` |
| **Inspector Drafts** | `BodyEditDraft::synchronize`, `reload`, `command`: draft edit -> Apply or Revert, with stale edit rejection | `src/ui/inspector_state.hpp` |
| **Create GPU Buffer (Device-Local + Staging Upload)** | `vk.create_device_buffer(size, usage, initial_data)` | `src/rhi/vk_context.hpp` |
| **Create Host-Mapped Buffer (UBO / Dynamic SSBO / Staging)** | `vk.create_host_buffer(size, usage, initial_data)` (`buf.mapped` is persistently mapped, `buf.address` has `VkDeviceAddress`) | `src/rhi/vk_context.hpp` |
| **Flush Host Buffer Writes** | `vk.flush_buffer(buffer, offset, size)` (VMA no-op for coherent memory) | `src/rhi/vk_context.hpp` |
| **Create 2D / Array / 3D Texture or Render Target** | `vk.create_image_2d(w, h, format, usage, mips, layers)` / `vk.create_image_3d(...)` / `vk.upload_image_2d(img, pixels, bytes)` | `src/rhi/vk_context.hpp` |
| **Image Layout Transition (Sync2)** | `vk.transition_image(cmd, image, new_layout, src_stage, src_access, dst_stage, dst_access)` (auto-tracks `image.layout`) | `src/rhi/vk_context.hpp` |
| **Compute / Memory Barrier (Sync2)** | `vk.memory_barrier(cmd, src_stage, src_access, dst_stage, dst_access)` | `src/rhi/vk_context.hpp` |
| **Begin / End Dynamic Rendering Pass** | `vk.begin_rendering(cmd, extent, color_attachments, &depth_attachment)` / `vk.end_rendering(cmd)` | `src/rhi/vk_context.hpp` |
| **Bind Resources (Zero Descriptor Pool Boilerplate)** | `vk.push_descriptors(cmd, pipeline, { PushDescriptorWrite::uniform_buffer(...), PushDescriptorWrite::storage_buffer(...), PushDescriptorWrite::sampled_image(...), PushDescriptorWrite::storage_image(...) })` | `src/rhi/vk_types.hpp`, `src/rhi/vk_context.hpp` |
| **Push Constants** | `vk.push_constants(cmd, pipeline, push_struct)` | `src/rhi/vk_context.hpp` |
| **One-Off GPU Transfer / Compute Submit** | `vk.immediate_submit([&](VkCommandBuffer cmd) { ... })` | `src/rhi/vk_context.hpp` |
| **Compile Compute Pipeline (with `#include` + `.spv` Disk Cache)** | `ShaderCache::create_compute_pipeline(vk, "compute/foo.comp", bindings, sizeof(Push), macros)` | `src/rhi/shader_cache.hpp` |
| **Compile Graphics Pipeline (with `#include` + `.spv` Disk Cache)** | `ShaderCache::create_graphics_pipeline(vk, GraphicsPipelineDesc{...})` | `src/rhi/shader_cache.hpp` |
| **GPU Pass Timestamp Profiling** | `uint32_t id = vk.begin_gpu_timer(cmd, "gpu_pass"); ... vk.end_gpu_timer(cmd, id);` -> read via `vk.gpu_time_ms("gpu_pass")` | `src/rhi/vk_context.hpp` |
| **File & JSON Load / Save** | `sf::paths::read_text_file`, `read_binary_file`, `write_binary_file`, `read_json`, `write_json` | `src/core/path_utils.hpp` |
| **Logging & Fatal Assertions** | `sf::log::info(...)`, `sf::log::warn(...)`, `sf::log::error(...)`, `sf::log::fatal(...)`, `VK_CHECK(expr)` | `src/core/log.hpp`, `src/rhi/vk_types.hpp` |
| **ImGui Side Panels & Theme** | `ui.prepare_side_panel(settings, state, PanelSide::Left/Right, layout)` / `ui.compute_layout(...)` | `src/ui/workspace.hpp` |
| **ImGui Inspector Widgets** | `sf::ui::heading(title, subtitle)`, `sf::ui::section(title, note)`, `sf::ui::value_row(label, val)`, `sf::ui::metric_text(str)`, `sf::ui::field(label, [&](const char* id){ ... })` | `src/ui/workspace.hpp` |

---

## 3. FEATURE / TASK ROUTING INDEX

| Task / Feature | Primary Files | Key Symbols |
| :--- | :--- | :--- |
| **Application Lifecycle & Main Loop** | `src/main.cpp`<br>`src/app.hpp`<br>`src/app.cpp`<br>`run.bat` | `sf::App::init`, `sf::App::run`, `sf::App::render_frame`, `--self-test` CLI flag |
| **Repeatable CPU/GPU Benchmarks** | `src/app_benchmark.cpp`<br>`src/benchmark/benchmark.hpp/.cpp`<br>`src/core/timing.hpp`<br>`scripts/compare_benchmarks.py`<br>`docs/performance_benchmarks.md` | `--benchmark`, fixed-step overview/approach/analytical/IAS15 + opt-in SPICE; warmup/repeats, frame-ID GPU pairing/final drain, CPU work/waits; compact averaged `report.txt` + JSON by default, raw JSON/CSV only with `--benchmark-detailed`; fixture-checked comparisons; `StellarBenchmarkTests` |
| **Vulkan 1.3 Context, VMA, Swapchain & Timers** | `src/rhi/vk_types.hpp`<br>`src/rhi/vk_context.hpp`<br>`src/rhi/vk_context.cpp` | `sf::rhi::VkContext`, `Buffer`, `Image`, `Pipeline`, `PushDescriptorWrite`, `RenderingAttachment`, `set_vsync`, `present_mode` |
| **GLSL `#include` Preprocessor & SPIR-V Disk Cache** | `src/rhi/shader_cache.hpp`<br>`src/rhi/shader_cache.cpp` | `sf::rhi::ShaderCache::load_and_preprocess`, `compile_to_spirv`, `create_compute_pipeline`, `create_graphics_pipeline` |
| **Physical Constants & Coordinate Transforms** | `src/core/constants.hpp` | `G_CONST`, `AU_TO_KM`, `SECONDS_PER_YEAR`, `OBLIQUITY`, `ecliptic_to_render`, `render_to_ecliptic` |
| **Keplerian / Universal Conic Math & Reverse-Z** | `src/core/math_utils.hpp` | `stumpff_cs`, `propagate_conic_state`, `conic_time_at_angle`, `orbital_to_cartesian`, `vector_orbital_to_cartesian`, `pole_to_ecliptic`, `reverse_z_perspective` |
| **Path Resolution & JSON/Binary I/O** | `src/core/path_utils.hpp`<br>`src/core/path_utils.cpp` | `project_root`, `bundled_path`, `external_path`, `read_json`, `write_json` |
| **Dear ImGui Workspace & Shared Widgets** | `src/ui/workspace.hpp`<br>`src/ui/workspace.cpp` | `sf::ui::UiSystem`, `WorkspaceSettings`, `WorkspaceLayout`, `heading`, `section`, `value_row`, `field` |
| **System Preset Loader & Analytical Propagator** | `src/ephemeris/system_manager.hpp`<br>`src/ephemeris/system_manager.cpp` | `SystemManager::parse_system`, `serialize_definition`, `LoadedSystem`, `SystemDefinition`, `BodyDefinition`, `BodyMotion`, `SystemSnapshot`, `BodyView` |
| **Simulation Ownership / Clock / Physical Edits** | `src/ephemeris/simulation.hpp/.cpp`<br>`src/core/time_utils.hpp/.cpp`<br>`docs/state_ownership.md` | `Simulation`, `SimulationClock`, `SetPaused`, `SetTimeWarp`, `SetTimeDirection`, `ReverseTimeDirection`, `SetMotionModel`, `calculate_sim_date`, `format_sim_date`, `format_time_speed` |
| **IAS15 15th-Order Adaptive Integrator & GR/Oblateness** | `src/ephemeris/ias15.hpp/.cpp`<br>`src/ephemeris/simulation.hpp` | `Ias15Integrator`, `MotionModel::NBody`, 1PN General Relativity, $J_2 / J_4$ oblateness, Gauss-Radau spacing, compensated summation |
| **Default Simulation Worker** | `src/ephemeris/simulation_runner.hpp/.cpp`<br>`scripts/benchmark_simulation_threads.py`<br>`tests/simulation_runner_tests.cpp`<br>`docs/performance_benchmarks.md` | `SimulationRunner`; asynchronous playback by default, `--no-simulation-thread` inline fallback; triple reused snapshot buffers, coalesced elapsed backlog, control-command barriers; camera update after UI commands, then scene recording from the same snapshot; `--benchmark-simulation inline/worker/pipeline` matched physics/render traces; `StellarSimulationRunnerTests`, `simulation_tracking_rendering_contracts` |
| **Shared Jacobi Coordinates & Analytical Propagation** | `src/ephemeris/jacobi.hpp/.cpp` | `JacobiHierarchy`, `build_jacobi_hierarchy`, `extract_jacobi`, `JacobiPropagator::reset/evaluate`; fixed sibling order, complete descendant masses, cached epoch states, top-down reflex reconstruction, analytical $J_2$ secular precession, third-body (grandparent) secular perturbation, GR apsidal advance; shared by simulation and orbit lines |
| **Dynamic Reference Parents & Distance-Sorted Outliner** | `src/ephemeris/reference_hierarchy.hpp/.cpp`<br>`src/ephemeris/simulation.hpp/.cpp`<br>`docs/dynamic_hierarchy.md`<br>`tests/hierarchy_tests.cpp` | `ReferenceHierarchyTracker`, `SystemSnapshot::reference_hierarchy/reference_parent/orbit_hierarchy/hierarchy_revision`; 15 Hz frame-time cadence + immediate load/edit/model/pause refresh; IAS15 Hill-region AABB search, 90% entry/110% exit, acyclic mass order; stable body indices, cached children in all modes; separate fixed analytical propagation hierarchy; `StellarHierarchyTests` |
| **SPICE Ephemeris Playback & Kernel Manager** | `src/ephemeris/spice_ephemeris.hpp/.cpp`<br>`data/ephemeris.json`<br>`data/ephemeris_settings.json`<br>`docs/orbit_lines.md` | `MotionModel::Spice`, `SpiceEphemeris::open/reload/epoch/state/state_relative/furnish_file/inspect_bsp/body_name/et_to_utc/body_gm_and_radius/default_kernels/known_bodies`, `SpiceKernelManager` (18 NASA NAIF default kernels across 7 groups, `data/kernels/additional/` scanner, async WinHTTP downloader with `.part` staging, dynamic moon/asteroid/spacecraft discovery, standalone `.bsp` playback `build_generic_bsp_system`); atomic coverage failures |
| **Orbit Extraction & Hill Continuations** | `src/ephemeris/orbit_curves.hpp/.cpp`<br>`docs/orbit_lines.md` | `compute_orbit_curves`, `compute_orbit_arc`, `OrbitCurve::point`; complete subsystem barycenters, Jacobi masses, dominant-child reflex, physical Hill exits, dashed outer estimates |
| **Vulkan Orbit Lines** | `src/rendering/orbit_renderer.hpp/.cpp`<br>`shaders/common/orbit_curve.glsl`<br>`shaders/celestial/orbit.vert/.frag` | `OrbitRenderer`; FP64 conic evaluation/camera subtraction; one-pixel Bresenham (`GraphicsPipelineDesc::bresenham_lines`, RHI KHR/EXT feature negotiation); original RGB conditioning/Bright Behind fade; reverse-Z opaque occlusion; `show_orbits`, `orbit_opacity`, `orbit_segments`; paused camera culling |
| **Orbit & Ephemeris Contract Tests** | `tests/orbit_tests.cpp`<br>`tests/fixtures/`<br>`tests/orbit_rendering_tests.cpp` | `StellarOrbitTests`, `StellarOrbitRenderingTests`; original Python conic fixtures; independent SpiceyPy oracles (`STELLAR_TEST_EPHEMERIS` opt-in), `.bsp` DAF segment inspection & generic `.bsp` playback contracts; GPU readback/occlusion/large-world precision |
| **Typed Units & Coordinates** | `src/core/units.hpp`<br>`src/core/coordinates.hpp` | `Seconds`, `Years`, `Kilometres`, `AstronomicalUnits`, `SolarMasses`, `EclipticPosition`, `RenderPosition`, `EclipticVelocity`, `RenderVelocity`, `camera_relative` |
| **Render Controls / UI Selection / Drafts** | `src/rendering/render_settings.hpp`<br>`src/ui/inspector_state.hpp` | `RenderSettingsStore`, `RenderSettings`, `WorkspaceState`, `BodyEditDraft`, `EphemerisSetupDraft`, `BspImportDraft` |
| **CPU State Contract Tests** | `tests/state_tests.cpp`<br>`CMakeLists.txt` | `StellarStateTests`, `ctest --test-dir build --output-on-failure`, `STELLAR_BUILD_TESTS` |
| **GPU Point-Source Contract Tests** | `tests/rendering_tests.cpp`<br>`docs/point_sources.md`<br>`CMakeLists.txt` | `StellarRenderingTests`, `STELLAR_BUILD_RENDER_TESTS` (opt-in; requires Vulkan + window system), HDR readback: transitions, inverse-square flux, occlusion, Gaia calibration |
| **64-Bit Camera Controller & Precision Modes** | `src/core/camera.hpp`<br>`src/core/camera.cpp` | `Camera::update(Seconds, snapshot)`, `focus_body`, `eye_pos_render`, `compute_near_plane_au` |
| **Gaia DR3 Starfield Renderer (~460k Stars)** | `src/rendering/star_catalog.hpp`<br>`src/rendering/star_catalog.cpp`<br>`shaders/celestial/starfield_cull.comp`<br>`shaders/celestial/starfield.vert`<br>`shaders/celestial/starfield.frag` | `StarCatalog`, `StarGpuRecord`, `StarfieldPushConstants`, compute culling pre-pass, indirect draw (`vkCmdDrawIndirect`) |
| **3D Celestial Spheres & Subpixel Point Lights** | `src/rendering/celestial_renderer.hpp`<br>`src/rendering/celestial_renderer.cpp`<br>`shaders/celestial/sphere.vert`<br>`shaders/celestial/sphere.frag`<br>`shaders/celestial/point_celestial.vert`<br>`shaders/celestial/point_celestial.frag` | `CelestialRenderer`, `SceneCelestialUBO`, `BodyInstanceGPU` |
| **Oblate Terrain LOD & Geometry Cache** | `src/terrain/{quadtree,tile_metadata,geometry_residency,geometry_cache,terrain_renderer}.hpp/.cpp`<br>`shaders/terrain/{terrain.vert,terrain_reference.vert,geometry_cache.comp}`<br>`shaders/common/terrain_geometry.glsl`<br>`docs/terrain.md`, `docs/terrain_performance.md`<br>`tests/terrain_tests.cpp`, `tests/terrain_gpu_tests.hpp`, `tests/rendering_tests.cpp` | `TileId`, `SelectionView`, `select`, `grid_indices`, `TileMetadata`, `TileMetadataCache`, `surface_bounds`, `surface_error_coefficient`, `GeometryResidency`, `GeometryCache`, `TerrainRenderer`, `TerrainStats`; opt-in Terrain LOD; six cube faces, selection reuse, 16K-entry CPU LRU of analytic local height/derivative bounds, displaced conservative culling, atomic balanced refinement including neighbor transitions within budget, cross-face stitching/morphs; global shared shader morph rule; FP64 CPU/bakes, 64 MiB LRU of FP32 patch-local geometry, 128 bakes/frame, full FP64 miss/reference fallback, selective distant FP64 projection using double camera rotation; 4096 visible patches; prepare/compute before dynamic rendering; `StellarTerrainTests` includes displaced-bound/error containment, cache budgets and ground-detail/coverage at interiors/poles/seams; GPU `--terrain-preview` / `--terrain-precision` / `--terrain-benchmark` / `--terrain-ground-benchmark` / `--terrain-relief-benchmark` (0/20 km relief at 2 km altitude); real height/texture streaming pending |
| **Surface Eclipse Candidate Filtering** | `src/rendering/eclipse_candidates.hpp`<br>`src/rendering/celestial_renderer.cpp`<br>`shaders/celestial/sphere.frag`<br>`src/benchmark/earth_closeup.hpp`<br>`tests/eclipse_tests.cpp`<br>`docs/eclipse_shadows.md` | `eclipse_could_shadow`; conservative whole-receiver CPU bounds, per-body/star SSBO lists, unchanged GPU penumbra; `earth-full` / `earth-eclipse` benchmark fixtures; `StellarEclipseTests` |
| **Shared Mesh/Point Transition & Photometry** | `shaders/common/point_source.glsl`<br>`shaders/common/celestial_scene.glsl`<br>`docs/point_sources.md` | `mesh_weight` (1.5–2.5 px radius / 3–5 px diameter); local sprite follows mesh size down to 3 px default minimum; `point_profile` (compact radial), `point_profile_sum` (pixel-grid flux normalization), `point_signal_fade`, `stellar_radiance`, `surface_reflectance`; fully unresolved bodies do not occlude points |
| **HDR Post-Processing & Tone Mapping** | `shaders/common/tonemap.glsl`<br>`shaders/post/fullscreen.vert`<br>`shaders/post/composite.frag` | `aces_tonemap`, `linear_to_srgb` |
| **Astrophysical Star Creator & System Management** | `src/physics/star_calc.hpp/.cpp`<br>`src/ephemeris/system_manager.hpp/.cpp`<br>`src/ui/inspector_state.hpp`<br>`src/ui/workspace.hpp/.cpp`<br>`tests/star_calc_tests.cpp` | `StarCalculator`, `StarForgeResult`, `StarParameters`, `StarCreatorDraft`, `create_new_system_from_props`, `delete_system`, `render_star_creator_widgets`, `StellarStarCalcTests` |

---

## 4. DIRECTORY MANIFEST

```text
Stellar Forge Rewrite/
├── CMakeLists.txt                  # CMake 3.28+ build config (FetchContent: Vulkan-Headers, volk, VMA, GLFW, GLM, ImGui, json, glslang)
├── run.bat                         # Configure, Ninja parallel build, and launch script
├── project_map.md                  # LLM / Agent Fast-Index and Shared Boilerplate Registry
├── src/
│   ├── main.cpp                    # Entry point & --self-test runner
│   ├── app.hpp / app.cpp           # Window, input, HDR/Reverse-Z frame loop, Outliner/Inspector/HUD workspace
│   ├── core/
│   │   ├── camera.hpp / .cpp       # FP64 camera (Free Flight, Simple Orbit, Near-Surface), log-depth zoom, ray picking
│   │   ├── units.hpp / coordinates.hpp # Explicit units and framed positions/velocities
│   │   ├── command_result.hpp    # Shared atomic command result
│   │   ├── constants.hpp           # Astronomical constants & Ecliptic <-> Render transforms
│   │   ├── log.hpp                 # Formatted stdout/stderr logging & fatal handler
│   │   ├── math_utils.hpp          # FP64 Kepler/Stumpff conic propagators, frames, Reverse-Z perspective
│   │   └── path_utils.hpp / .cpp   # Bundled/external path resolution, binary & JSON file I/O
│   ├── ephemeris/
│   │   ├── system_manager.hpp/.cpp # JSON definitions, motion seeds, immutable snapshot interfaces
│   │   ├── simulation.hpp/.cpp   # State owner, clock, validated commands, Keplerian/SPICE propagation
│   │   ├── spice_ephemeris.hpp/.cpp # Dynamic CSPICE runtime, local kernels, epoch/state queries
│   │   ├── jacobi.hpp/.cpp        # Shared subsystem/inner coordinates and cached analytical propagation
│   │   └── orbit_curves.hpp/.cpp # Osculating/Jacobi conics, Hill exit patches and primary reflex
│   ├── physics/
│   │   └── star_calc.hpp / .cpp   # Astrophysical star forge (evolution tracks, Stefan-Boltzmann, Roche rotation, Planckian RGB)
│   ├── rendering/
│   │   ├── render_settings.hpp  # Validated render controls, separate from camera pose
│   │   ├── star_catalog.hpp / .cpp # Gaia DR3 stars.bin loader & flux-normalized point-sprite pipeline
│   │   ├── celestial_renderer.hpp/.cpp # Lat/lon UV sphere mesh, 3D star/planet shader, additive subpixel point pass
│   │   └── orbit_renderer.hpp/.cpp # FP64 parameter-driven Vulkan orbit line strips
│   ├── rhi/
│   │   ├── vk_types.hpp            # Buffer, Image, Pipeline, PushDescriptorWrite, RenderingAttachment, GraphicsPipelineDesc
│   │   ├── vk_context.hpp / .cpp   # Vulkan 1.3 device, VMA, swapchain, Sync2 barriers, Dynamic Rendering, GPU timers
│   │   └── shader_cache.hpp / .cpp # Recursive #include resolver, glslang SPIR-V compiler, disk cache, pipeline builders
│   └── ui/
│       ├── inspector_state.hpp  # Selection, transient state, atomic physical drafts
│       └── workspace.hpp / .cpp    # ImGui Vulkan backend, Slate/Cyan/Amber theme, layout & inspector widget helpers
├── shaders/
│   ├── celestial/
│   │   ├── starfield_cull.comp     # Gaia DR3 compute culling pre-pass & dynamic indirect draw generation
│   │   ├── starfield.vert / .frag  # Gaia DR3 starfield with proper motion & shared compact 3 px PSF
│   │   ├── sphere.vert / .frag     # Camera-relative 3D star/planet renderer (limb darkening, Oren-Nayar, rings)
│   │   ├── point_celestial.vert/.frag # Subpixel billboard with disc-integrated flux & resolved-body visibility
│   │   └── orbit.vert/.frag       # Faded/dashed orbital lines with opaque reverse-Z depth testing
│   ├── common/
│   │   ├── point_source.glsl      # Shared complementary transition, normalized PSF & faint-signal fade
│   │   ├── celestial_scene.glsl   # Celestial GPU layout, projected radius & shared surface lighting
│   │   ├── tonemap.glsl            # Shared ACES tone mapping & sRGB conversion
│   │   └── orbit_curve.glsl        # FP64 eccentric/true anomaly curve evaluation
│   └── post/
│       ├── fullscreen.vert         # Shared vertex-pulling fullscreen triangle
│       └── composite.frag          # HDR exposure & tone-mapping composite pass
├── docs/state_ownership.md         # Ownership, mutation, units and invalidation contracts
├── docs/orbit_lines.md             # Conic migration, SPICE runtime setup, model semantics and tests
├── tests/state_tests.cpp           # CPU contract tests and preset checks
├── tests/star_calc_tests.cpp       # Astrophysics & Star Creator contract tests
├── data/                           # Isolated runtime data (systems/, gaia/stars.bin, kernels/, tiles/)
└── textures/                       # Isolated planetary surface & ring textures
```

---

## 5. REWRITE MIGRATION ROADMAP (FROM `D:\Software\Stellar Forge`)

1. **[DONE] Phase 1 — Foundation, Shared RHI/Core/UI Boilerplate & SPIR-V Pipeline**
2. **[DONE] Phase 2 — Data Model (`SystemManager`), FP64 Camera Controller, Gaia DR3 Starfield & Basic Celestial Spheres**
   - Ported `SystemManager` (`data/systems/*.json`), 64-bit `Camera` (Free Flight, Simple Orbit, Near-Surface, raycast picking), `StarCatalog` (`data/gaia/stars.bin`), and `CelestialRenderer` (`sphere.{vert,frag}` + `point_celestial.{vert,frag}`).
   - **[DONE] Ownership foundation**: separated immutable definitions, live motion, prepared render state, render controls and transient UI state; typed time/coordinate boundaries; validated physical drafts; domain revisions and CPU contract tests. See `docs/state_ownership.md`.
3. **[IN PROGRESS] Phase 3 — Physics & Ephemeris Engine (`src/physics/`, `src/ephemeris/`)**
   - **[DONE]** Osculating orbit calculations, subsystem/Jacobi barycenters, dominant-child reflex curves, Hill clipping and dashed outer conic estimates; FP64 Vulkan line renderer and opaque-scene occlusion. See `docs/orbit_lines.md`.
   - **[DONE]** Explicit Keplerian/SPICE motion models; CSPICE kernel playback for Ephemeris Mode via local `data/ephemeris.json`, NAIF IDs and stored UTC epoch; coverage failures roll back and pause.
   - **[DONE]** SPICE Ephemeris Kernel Settings, NAIF Downloader & `.bsp` Preview/Import (`src/ephemeris/spice_ephemeris.hpp/.cpp`, `src/ui/inspector_state.hpp`, `src/app.cpp`): `SpiceKernelManager` with 18 NASA NAIF default kernels across 7 groups, `data/ephemeris_settings.json` persistence synchronized to `data/ephemeris.json`, `data/kernels/additional/` scanner & installer, background WinHTTP downloader with `.part` staging/progress/speed/cancellation (`render_ephem_download_modal`), DAF/SPK segment inspector (`SpiceEphemeris::inspect_bsp` via `dafopr_c`/`dafbfs_c`/`daffna_c`/`dafgs_c`/`spkuds_c`/`bodc2n_c`/`et2utc_c` + sidecar `<bsp>.json` metadata), dynamic moon/asteroid/spacecraft discovery (`build_ephemeris_system`), and standalone `.bsp` playback (`build_generic_bsp_system`, `render_bsp_import_modal`, `render_ephem_setup_modal`).
   - **[DONE]** Shared fixed-order Jacobi propagation/extraction, complete subsystem masses and top-down primary/moon reconstruction; mass-edit reseeding; Pluto/Charon and crossing-sibling time-series regressions. SPICE planets use physical centers (`599/699/899/999`) before accumulating moon barycenters.
   - **[DONE]** IAS15 adaptive 15th-order Gauss-Radau integrator + 1PN relativity + $J_2/J_4$ oblateness harmonics; explicit `MotionModel::NBody` and main menu Physics mode switching between N-Body (IAS15), Analytical Keplerian (Jacobi), and NASA SPICE Ephemeris Playback.
   - **[DONE]** Cached distance-sorted outliner and dynamic IAS15 reference parents at 15 Hz; spatial Hill-region classification with hysteresis, shared orbit/inspector/barycenter references, stable indices and snapshot-safe caches. Analytical parent transitions remain separate future work. See `docs/dynamic_hierarchy.md`.
   - **[DONE]** Star Creator astrophysics and creation feature (`src/physics/star_calc.hpp/.cpp`, `tests/star_calc_tests.cpp`): `StarCalculator` evolution tracks (low/intermediate/massive/WR branches) & surface Stefan-Boltzmann solver; Roche rotation & Von Zeipel gravity darkening; spectral classification & blackbody Planckian RGB; habitable zones & pulsation regimes; atomic physical drafts with generation/revision tracking; system creation & deletion disk persistence; UI integrated into Create System modal, Body Inspector, and Add Body modal.
   - **[NEXT]** Analytical secular precession, sampled spacecraft ephemeris polylines/Lagrange frames, adaptive optical subdivision and `orbit_layer.py` atmosphere transport/MSAA.
4. **[IN PROGRESS] Phase 4 — Native C++ Terrain Quadtree, Async Tile Streamer & GPU Surface Caches (`src/terrain/`)**
   - **[DONE] Geometry preview**: native oblate cube-face quadtree, bounded screen-error selection, conservative culling, cross-face neighbor balance, stitched reusable grids, spatial morphing, FP64 patch-local placement, synthetic relief/debug colors and CPU/GPU contracts. See `docs/terrain.md`.
   - **[DONE] Precision/cache path**: bounded GPU patch-local position/normal cache, FP32 near draws, selective distant FP64 placement, reference/fallback path, stationary selection reuse and controlled CPU/GPU precision benchmarks. See `docs/terrain_performance.md`.
   - **[NEXT]** Preprocessed height/albedo tiles, bounded physical tile cache with resident-parent fallback and shared asynchronous RHI uploads; independent geometry/material resolution.
   - Port `native/terrain/src/{quadtree,stream,surface,erosion}.rs` + GPU height/normal/horizon/local-shadow caches (`terrain_*.comp`, `terrain.{vert,frag}`).
5. **Phase 5 — Volumetric & Analytical Atmospheres, Rings & Eclipse Optics (`src/rendering/`)**
   - Port `atmosphere_physics.py`, `scattering_lut.py`, `sky_view_lut.py`, `atmo.frag`, VRS, surface irradiance SH, refraction, ring optics, planetshine, and ringshine.
6. **Phase 6 — 5-Cascade FFT Oceans, Convolution Diffraction Bloom & 4K/8K/16K Tiled Exporter**
   - Port `ocean_waves.py` (`water/*.comp`), `ocean.py`, `conv_bloom_*.comp`, photo accumulation, Size Comparator scene, and full Inspector/Settings modals.
