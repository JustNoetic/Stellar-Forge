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
| :--- | :--- | :--- |
| **Simulation State / Commands** | `simulation.snapshot()` (live const reference), `simulation.advance(units::Seconds{...})`, `simulation.apply(SetPaused / SetTimeWarp / ApplyBodyPhysicalProperties)` | `src/ephemeris/simulation.hpp` |
| **Rendering Settings** | `RenderSettingsStore::values`, `apply`, `revision`: validated whole-settings update; camera owns no rendering controls | `src/rendering/render_settings.hpp` |
| **Inspector Drafts** | `BodyEditDraft::synchronize`, `reload`, `command`: draft edit -> Apply or Revert, with stale edit rejection | `src/ui/inspector_state.hpp` |
| **Create GPU Buffer (Device-Local + Staging Upload)** | `vk.create_device_buffer(size, usage, initial_data)` | `src/rhi/vk_context.hpp` |
| **Create Host-Mapped Buffer (UBO / Dynamic SSBO / Staging)** | `vk.create_host_buffer(size, usage, initial_data)` (`buf.mapped` is persistently mapped, `buf.address` has `VkDeviceAddress`) | `src/rhi/vk_context.hpp` |
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
| **Vulkan 1.3 Context, VMA, Swapchain & Timers** | `src/rhi/vk_types.hpp`<br>`src/rhi/vk_context.hpp`<br>`src/rhi/vk_context.cpp` | `sf::rhi::VkContext`, `Buffer`, `Image`, `Pipeline`, `PushDescriptorWrite`, `RenderingAttachment` |
| **GLSL `#include` Preprocessor & SPIR-V Disk Cache** | `src/rhi/shader_cache.hpp`<br>`src/rhi/shader_cache.cpp` | `sf::rhi::ShaderCache::load_and_preprocess`, `compile_to_spirv`, `create_compute_pipeline`, `create_graphics_pipeline` |
| **Physical Constants & Coordinate Transforms** | `src/core/constants.hpp` | `G_CONST`, `AU_TO_KM`, `SECONDS_PER_YEAR`, `OBLIQUITY`, `ecliptic_to_render`, `render_to_ecliptic` |
| **Keplerian / Universal Conic Math & Reverse-Z** | `src/core/math_utils.hpp` | `stumpff_cs`, `propagate_conic_state`, `conic_time_at_angle`, `orbital_to_cartesian`, `vector_orbital_to_cartesian`, `pole_to_ecliptic`, `reverse_z_perspective` |
| **Path Resolution & JSON/Binary I/O** | `src/core/path_utils.hpp`<br>`src/core/path_utils.cpp` | `project_root`, `bundled_path`, `external_path`, `read_json`, `write_json` |
| **Dear ImGui Workspace & Shared Widgets** | `src/ui/workspace.hpp`<br>`src/ui/workspace.cpp` | `sf::ui::UiSystem`, `WorkspaceSettings`, `WorkspaceLayout`, `heading`, `section`, `value_row`, `field` |
| **System Preset Loader & Analytical Propagator** | `src/ephemeris/system_manager.hpp`<br>`src/ephemeris/system_manager.cpp` | `SystemManager::parse_system`, `serialize_definition`, `LoadedSystem`, `SystemDefinition`, `BodyDefinition`, `BodyMotion`, `SystemSnapshot`, `BodyView` |
| **Simulation Ownership / Clock / Physical Edits** | `src/ephemeris/simulation.hpp/.cpp`<br>`docs/state_ownership.md` | `Simulation`, `SimulationClock`, `SetPaused`, `SetTimeWarp`, `ApplyBodyPhysicalProperties`, `BodyRevisions` |
| **Typed Units & Coordinates** | `src/core/units.hpp`<br>`src/core/coordinates.hpp` | `Seconds`, `Years`, `Kilometres`, `AstronomicalUnits`, `SolarMasses`, `EclipticPosition`, `RenderPosition`, `EclipticVelocity`, `RenderVelocity`, `camera_relative` |
| **Render Controls / UI Selection / Drafts** | `src/rendering/render_settings.hpp`<br>`src/ui/inspector_state.hpp` | `RenderSettingsStore`, `RenderSettings`, `WorkspaceState`, `BodyEditDraft` |
| **CPU State Contract Tests** | `tests/state_tests.cpp`<br>`CMakeLists.txt` | `StellarStateTests`, `ctest --test-dir build --output-on-failure`, `STELLAR_BUILD_TESTS` |
| **GPU Point-Source Contract Tests** | `tests/rendering_tests.cpp`<br>`docs/point_sources.md`<br>`CMakeLists.txt` | `StellarRenderingTests`, `STELLAR_BUILD_RENDER_TESTS` (opt-in; requires Vulkan + window system), HDR readback: transitions, inverse-square flux, occlusion, Gaia calibration |
| **64-Bit Camera Controller & Precision Modes** | `src/core/camera.hpp`<br>`src/core/camera.cpp` | `Camera::update(Seconds, snapshot)`, `focus_body`, `eye_pos_render`, `compute_near_plane_au` |
| **Gaia DR3 Starfield Renderer (~460k Stars)** | `src/rendering/star_catalog.hpp`<br>`src/rendering/star_catalog.cpp`<br>`shaders/celestial/starfield.vert`<br>`shaders/celestial/starfield.frag` | `StarCatalog`, `StarGpuRecord`, `StarfieldPushConstants` |
| **3D Celestial Spheres & Subpixel Point Lights** | `src/rendering/celestial_renderer.hpp`<br>`src/rendering/celestial_renderer.cpp`<br>`shaders/celestial/sphere.vert`<br>`shaders/celestial/sphere.frag`<br>`shaders/celestial/point_celestial.vert`<br>`shaders/celestial/point_celestial.frag` | `CelestialRenderer`, `SceneCelestialUBO`, `BodyInstanceGPU` |
| **Shared Mesh/Point Transition & Photometry** | `shaders/common/point_source.glsl`<br>`shaders/common/celestial_scene.glsl`<br>`docs/point_sources.md` | `mesh_weight` (1.5–2.5 px radius / 3–5 px diameter); local sprite follows mesh size down to 3 px default minimum; `point_profile` (compact radial), `point_profile_sum` (pixel-grid flux normalization), `point_signal_fade`, `stellar_radiance`, `surface_reflectance`; fully unresolved bodies do not occlude points |
| **HDR Post-Processing & Tone Mapping** | `shaders/common/tonemap.glsl`<br>`shaders/post/fullscreen.vert`<br>`shaders/post/composite.frag` | `aces_tonemap`, `linear_to_srgb` |

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
│   │   └── simulation.hpp/.cpp   # State owner, clock, validated commands, analytical propagation
│   ├── rendering/
│   │   ├── render_settings.hpp  # Validated render controls, separate from camera pose
│   │   ├── star_catalog.hpp / .cpp # Gaia DR3 stars.bin loader & flux-normalized point-sprite pipeline
│   │   └── celestial_renderer.hpp/.cpp # Lat/lon UV sphere mesh, 3D star/planet shader, additive subpixel point pass
│   ├── rhi/
│   │   ├── vk_types.hpp            # Buffer, Image, Pipeline, PushDescriptorWrite, RenderingAttachment, GraphicsPipelineDesc
│   │   ├── vk_context.hpp / .cpp   # Vulkan 1.3 device, VMA, swapchain, Sync2 barriers, Dynamic Rendering, GPU timers
│   │   └── shader_cache.hpp / .cpp # Recursive #include resolver, glslang SPIR-V compiler, disk cache, pipeline builders
│   └── ui/
│       ├── inspector_state.hpp  # Selection, transient state, atomic physical drafts
│       └── workspace.hpp / .cpp    # ImGui Vulkan backend, Slate/Cyan/Amber theme, layout & inspector widget helpers
├── shaders/
│   ├── celestial/
│   │   ├── starfield.vert / .frag  # Gaia DR3 starfield with proper motion & shared compact 3 px PSF
│   │   ├── sphere.vert / .frag     # Camera-relative 3D star/planet renderer (limb darkening, Oren-Nayar, rings)
│   │   └── point_celestial.vert/.frag # Subpixel billboard with disc-integrated flux & resolved-body visibility
│   ├── common/
│   │   ├── point_source.glsl      # Shared complementary transition, normalized PSF & faint-signal fade
│   │   ├── celestial_scene.glsl   # Celestial GPU layout, projected radius & shared surface lighting
│   │   └── tonemap.glsl            # Shared ACES tone mapping & sRGB conversion
│   └── post/
│       ├── fullscreen.vert         # Shared vertex-pulling fullscreen triangle
│       └── composite.frag          # HDR exposure & tone-mapping composite pass
├── docs/state_ownership.md         # Ownership, mutation, units and invalidation contracts
├── tests/state_tests.cpp           # CPU contract tests and preset checks
├── data/                           # Isolated runtime data (systems/, gaia/stars.bin, kernels/, tiles/)
└── textures/                       # Isolated planetary surface & ring textures
```

---

## 5. REWRITE MIGRATION ROADMAP (FROM `D:\Software\Stellar Forge`)

1. **[DONE] Phase 1 — Foundation, Shared RHI/Core/UI Boilerplate & SPIR-V Pipeline**
2. **[DONE] Phase 2 — Data Model (`SystemManager`), FP64 Camera Controller, Gaia DR3 Starfield & Basic Celestial Spheres**
   - Ported `SystemManager` (`data/systems/*.json`), 64-bit `Camera` (Free Flight, Simple Orbit, Near-Surface, raycast picking), `StarCatalog` (`data/gaia/stars.bin`), and `CelestialRenderer` (`sphere.{vert,frag}` + `point_celestial.{vert,frag}`).
   - **[DONE] Ownership foundation**: separated immutable definitions, live motion, prepared render state, render controls and transient UI state; typed time/coordinate boundaries; validated physical drafts; domain revisions and CPU contract tests. See `docs/state_ownership.md`.
3. **[NEXT] Phase 3 — Physics & Ephemeris Engine (`src/physics/`, `src/ephemeris/`)**
   - Port `physics_core.py` (IAS15 adaptive integrator + 1PN relativity + $J_2/J_4$), `kepler_analytical.py`, `orbit_curve.glsl` + `orbit_layer.py`, and NASA SPICE (`CSPICE`) / Lagrange frames.
4. **Phase 4 — Native C++ Terrain Quadtree, Async Tile Streamer & GPU Surface Caches (`src/terrain/`)**
   - Port `native/terrain/src/{quadtree,stream,surface,erosion}.rs` + GPU height/normal/horizon/local-shadow caches (`terrain_*.comp`, `terrain.{vert,frag}`).
5. **Phase 5 — Volumetric & Analytical Atmospheres, Rings & Eclipse Optics (`src/rendering/`)**
   - Port `atmosphere_physics.py`, `scattering_lut.py`, `sky_view_lut.py`, `atmo.frag`, VRS, surface irradiance SH, refraction, ring optics, planetshine, and ringshine.
6. **Phase 6 — 5-Cascade FFT Oceans, Convolution Diffraction Bloom & 4K/8K/16K Tiled Exporter**
   - Port `ocean_waves.py` (`water/*.comp`), `ocean.py`, `conv_bloom_*.comp`, photo accumulation, Size Comparator scene, and full Inspector/Settings modals.
