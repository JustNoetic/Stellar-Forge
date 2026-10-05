# PROJECT_MAP.MD (LLM FAST-INDEX)

> **Audience**: AI Agents / LLMs.
> **Purpose**: Instant routing index. Eliminate whole-codebase scans.
> **Format**: Dense, zero-prose, symbol-to-file lookup.

---

## 1. CRITICAL CONVENTIONS & INVARIANTS

| Domain | Rule / Invariant | References |
| :--- | :--- | :--- |
| **Coordinate Transform** | Physics is Ecliptic `(x, y, z)`. Render is OpenGL `(x, z, -y)`. | `engine/physics/physics_core.py` (`_extract_render_state`), `engine/core/math_utils.py` |
| **Units: Distance** | Interplanetary: `AU`. Radii / Altitudes: `km`. Unit conversion: `AU_TO_KM = 149597870.7`. | `engine/core/constants.py` |
| **Units: Time** | Orbital physics: `years`. Frame delta: `seconds`. Conversion: `SECONDS_PER_YEAR = 365.25 * 86400`. | `engine/core/constants.py` |
| **Units: Mass & G** | Mass in Solar Masses ($M_\odot$). Gravitational constant `G = 39.476926421373` ($\approx 4\pi^2$). | `engine/core/constants.py` (`G`), `engine/physics/physics_core.py` |
| **Path Resolution** | NEVER hardcode root paths. Bundle/frozen paths vs. external user data paths. | `engine/path_utils.py` (`get_bundled_path`, `get_external_path`) |
| **Thread Synchronization** | Sim thread & Render thread communicate through `app.shared_state` under `app.shared_state["lock"]`. | `engine/app.py`, `engine/physics/physics_core.py` |
| **Agent Scratchpad** | Put temporary test/debug Python scripts in `scratch/`. Ignored by git; prevents repo pollution. | `scratch/` |
| **OS & Platform Support** | Fully cross-platform (Windows & Linux). High-res timer gracefully no-ops on Linux. GLFW supports X11/Wayland. Use `run.sh` on Linux, `run.bat` on Windows. | `run.bat`, `run.sh`, `engine/physics/physics_core.py` |

---

## 2. FEATURE / TASK ROUTING INDEX

| Task / Feature | Primary File | Key Functions / Classes / Symbols |
| :--- | :--- | :--- |
| **Application Lifecycle & Launchers** | `engine/app.py`<br>`engine/main.py`<br>`run.bat` (Windows)<br>`run.sh` (Linux) | `App.__init__`, `App.run`, `App.render_frame`<br>Venv bootstrap & launch, crash logging to `main_error.txt` |
| **N-Body Gravitational Physics** | `engine/physics/physics_core.py` | `Simulation`, `compute_custom_forces`, `compute_all_accelerations`, `attach_custom_forces` |
| **Relativity (1PN) & Oblateness (J2/J4)** | `engine/physics/physics_core.py` | `compute_custom_forces` (Jacobi coords, J2, J4 zonal harmonics) |
| **Analytical Keplerian Propagator** | `engine/physics/kepler_analytical.py` | `propagate_keplerian_system_numba`, `extract_all_kepler_elements`, `state_to_kepler_vectors` |
| **SPICE Ephemeris Playback** | `engine/ephemeris/spice_manager.py` | `SpiceManager`, `SpiceManager.populate_states_fast`, `get_spice_manager` |
| **SPK Kernel Export** | `engine/ephemeris/spk_exporter.py` | `export_timeline_spk`, `export_timeline_spk_async`, `_chebyshev_fit_records` |
| **Star System Hierarchy & State** | `engine/ephemeris/system_manager.py` | `SystemManager`, `SystemSnapshot`, `derive_star_properties`, `get_spectral_class` |
| **Atmospheric Physics (Rayleigh/Mie)** | `engine/physics/atmosphere_physics.py` | `compute_atmosphere_properties`, `compute_dynamic_mie_properties`, `compute_mie_coefficients` |
| **Atmospheric Scattering LUTs** | `engine/rendering/scattering_lut.py`<br>`engine/glsl/atmosphere/` | `ScatteringLUTCache`<br>`scattering_lut.comp`, `sky_view_lut.frag`, `multi_scatter_lut.frag`, `atmo_lut.frag` |
| **Atmospheric Raymarching & God Rays** | `engine/glsl/atmosphere/`<br>`engine/rendering/scattering_lut.py` | `atmo.frag`, `atmo.vert`, `atmo_godrays.frag`, `atmo_upsample.frag`. High uses a cached optical-depth table and physical-cell jitter; ray steps sample lighting only. High and Analytical reuse per-ray/per-star caster masks so sample loops visit only possible eclipsers. Analytical keeps its baseline and subtracts per-star shadow deficits, with optional bounded sampling. Secondary-light marching skips planetshine with zero received color and ringshine without eligible rings; planetshine accumulates only for the first star. Surface clipping is independent of mesh LOD. |
| **Atmosphere Program Variants** | `engine/rendering/atmosphere_programs.py` | `AtmospherePrograms`, `specialize_atmosphere`: cached Low/High/Analytical programs; Analytical bounded on/off is a compile-time choice to avoid shared-shader performance regressions. |
| **Light Refraction & Bending** | `engine/physics/refraction.py`<br>`engine/glsl/common/refraction.glsl` | `compute_refraction_angle`, `solve_refraction_apparent`, `get_apparent_look_direction` |
| **Terrain Quadtree & LOD Geometry** | `engine/rendering/terrain_quadtree.py` | `PlanetQuadtree`, `QuadtreePatch`, `cube_to_sphere_point`, `_tan_warp` |
| **Terrain Tile Streaming** | `engine/rendering/terrain_streamer.py` | `TerrainTileStreamer` |
| **Terrain Geometry Cache** | `engine/rendering/terrain_geometry_cache.py`<br>`engine/glsl/compute/terrain_cache_{height,normal}.comp` | `TerrainGeometryCache`: CPU LOD retained; bounded body-local elevation/normal cache, shared grid samples, 128 patch bakes/frame, 96 MiB geometry cap. Height-slot upload versions invalidate edited/recycled layers; misses use original vertex shading. |
| **Surface Landing & Collision** | `engine/rendering/terrain_collision.py` | `CameraSurface`, `terrain_vertices`, `terrain_frame`, `terrain_face_uv`, `terrain_camera_split`: split body-local camera origin; exact frame-local query reuse. Rotation-follow probes use the predicted post-spin pose. |
| **N-body Warp Frame Stalls** | `engine/physics/physics_core.py`<br>`engine/app.py`<br>`scripts/test_ground_warp.py` | `first_body_collision`: compiled pair scan releases the GIL and preserves merge order. Corrected hierarchy/integrator warmup precedes both physics threads. |
| **Planetary Rings & Procedural Profiles** | `engine/rendering/ring_generator.py`<br>`engine/rendering/texture_baker.py` | `generate_procedural_ring_profile`<br>`bake_and_export_ring_textures`, `apply_procedural_ring_to_body` |
| **Ring Radiance & Finite Sources** | `engine/glsl/common/ring_{optics,source}.glsl`<br>`engine/rendering/ring_optics.py`<br>`engine/glsl/celestial/ring.frag`<br>`docs/ring_optics.md` | Shared normalized HG/CS lobes, ±0.99 g, continuous vertical optical depth, stable slab response, and opposition approximation. Each overlapping layer retains its phase; extinction is shared. 64-node finite-source quadrature handles stellar disks at equinox and the visible illuminated oblate host for planetshine. `scripts/test_ring_optics.py`: native GPU/Numba regressions. |
| **Body Eclipse Penumbrae** | `engine/glsl/common/eclipse_shadow.glsl`<br>`engine/glsl/common/caster_shadow.glsl` | Shared distance-space cubic penumbra (0.625 power), CPU grazing optical depth, separate atmosphere thickness/scale height. `caster_shadow_interval`: solid radius plus stellar footprint, clipped behind caster; atmosphere thickness does not inflate the fallback region. `caster_shadow_ray_mask` adds conservative AU float-precision padding; masked evaluation visits selected caster bits in original order. Scene UBO grazing tail: floats 1832–2087; Atmo SSBO scale-height tail: floats 256–263. |
| **Ring Shadows & Filtering** | `engine/rendering/ring_shadow_filter.py`<br>`engine/glsl/common/`<br>`engine/glsl/atmosphere/atmo.frag` | `RingShadowFilter`<br>`ring_shadow_filter.glsl`, `surface_ring_shadow.glsl`, `caster_shadow.glsl`. Analytical ring-shadow selection matches the grazing filter footprint and adds a 3-degree latitude guard (`RING_SHADOW_ANGLE_PADDING`), with a finite plane slab at equinox. Padding changes sampling bounds only. |
| **Surface Materials & Hapke Photometry** | `engine/rendering/surface_materials.py`<br>`engine/glsl/common/surface_{material,phase}.glsl`<br>`engine/ui/surface_material_editor.py`<br>`docs/surface_materials.md` | `SurfaceMaterialCache`, `resolve_material`, `hapke_raw`, `phase_table`: body-indexed SSBOs 11/12; shared sphere/terrain BRDF, point flux and planetshine phase tables; Cosmetics editor and JSON persistence. Missing material resolves Moon/Luna to lunar, Europa to icy, otherwise Lambert. |
| **Planetshine & Moonshine** | `engine/rendering/planetshine.py` | `compute_planetshine_numba`, `compute_body_rotation_angles_jit`, `get_cached_atmosphere_properties` |
| **Ringshine & Oblate Ring Transport** | `engine/rendering/ringshine.py`<br>`engine/glsl/post/ringshine_map.frag`<br>`engine/glsl/common/ringshine_{integral,lookup}.glsl` | `RingshineMap.update`, `build_secondary_ring_properties`, `ring_phase_radiance`, `ring_mixed_radiance` (`ring_optics.py`). Cached 128×65 tiles per host/star in a 16×16 atlas; exact oblate lit arcs with warped 16-node azimuth quadrature and four-node radial transfer. No geometry/CDF LUT. Extinction atlas reserves 16 host rows, followed by up to 256 original segment rows. Optical bake reads each layer, not averaged g or a majority material family. Cache includes stellar angular radius; distant-body lighting preserves lobes at 64 radial area nodes. Readers use texture height. |
| **Starfield & Gaia DR3 Catalog** | `engine/rendering/star_catalog.py`<br>`engine/glsl/celestial/` | `StarCatalog`, `_pack_kernel`<br>`starfield.vert`, `starfield.frag`, `point_celestial.frag` |
| **HDR, Tone Mapping & Bloom** | `engine/rendering/post_shaders.py`<br>`engine/glsl/post/` | Post-processing program compilations<br>`composite.frag`, `accum.frag`, `bloom_downsample.frag`, `bloom_upsample.frag` |
| **FFT / Convolution Diffraction Spikes** | `engine/glsl/post/` | `conv_bloom_scene.comp`, `conv_bloom_kernel.comp`, `conv_bloom_convolve.comp`, `conv_bloom_common.glsl` |
| **Texture Management & Hot-Reloading** | `engine/rendering/texture_manager.py`<br>`engine/rendering/texture_streamer.py` | `render_texture_management_ui`, `stage_imported_texture`, `hot_reload_body_texture`<br>`TextureStreamer` |
| **Input Handling & Keybindings** | `engine/core/input_handler.py` | `InputHandlerMixin.key_callback`, `scroll_callback`, `mouse_button_callback`, `save_settings`, `load_settings` |
| **Camera Math & Coordinate Frames** | `engine/core/input_handler.py`<br>`engine/core/math_utils.py` | `_camera_align_up`, `_camera_forward`, `_camera_yaw_pitch_from`, `_camera_rot_axis`, `_camera_get_up` |
| **UI: Top Menu Bar** | `engine/ui/menu_bar.py` | `render_main_menu_bar` (Systems, Physics modes, View, Render, Tools, Quick Actions) |
| **UI: System Outliner Panel** | `engine/ui/outliner.py` | `render_system_outliner` (Celestial body tree hierarchy, Add Body trigger) |
| **UI: Body Inspector Panel** | `engine/ui/inspector.py` | `render_body_inspector` (Live astrophysics telemetry, orbit/atmosphere/ring/texture editor) |
| **UI: Time Transport HUD** | `engine/ui/time_hud.py` | `render_time_hud` (Warp speed, pause, timeline render progress, scrubbing) |
| **UI: Viewport HUD & Triangle Stats** | `engine/ui/viewport_hud.py` | `render_viewport_hud` (Camera mode pill, flight speed slider, triangle count overlay) |
| **UI: Modals & Popups** | `engine/ui/modals.py` | `render_modals` (Graphics Settings, Add Body, Create System, Ephemeris Setup, Date Jump) |
| **Shader Loading & Preprocessor** | `engine/rendering/shader_loader.py` | `load_shader` (handles `#include` directives via `_resolve_includes`), `clear_shader_cache` |
| **ImGui Backend (ModernGL + GLFW)** | `engine/rendering/imgui_renderer.py` | `ModernGLImGuiRenderer`, `ModernGLGlfwRenderer` |

---

## 3. FILE-BY-FILE DIRECTORY MANIFEST

```
scratch/                        # Temporary scratchpad for agents to run test/debug scripts (git-ignored)
engine/
├── app.py                      # Main App class: window setup, GL context, camera state, render pipeline loop
├── main.py                     # Entry point: stdout/stderr redirection, faulthandler, App launch
├── path_utils.py               # get_bundled_path(), get_external_path() for frozen/runtime paths
├── __init__.py                 # Backward-compatibility module aliases
│
├── core/
│   ├── constants.py            # Physical constants (G, AU_TO_KM, SECONDS_PER_YEAR, C_AU_YR, LY_TO_AU)
│   ├── input_handler.py        # InputHandlerMixin (GLFW callbacks, camera modes, settings persistence)
│   └── math_utils.py           # Kepler/Cartesian transforms, coordinate rotators, fast math helpers
│
├── ephemeris/
│   ├── spice_manager.py        # SpiceManager (NASA NAIF SPICE kernel loading, state evaluation)
│   ├── system_manager.py       # SystemManager, SystemSnapshot (systems persistence, star classification)
│   └── spk_exporter.py         # Chebyshev polynomial trajectory fitting & SPK kernel binary exporter
│
├── physics/
│   ├── atmosphere_physics.py   # Rayleigh & Mie scattering coefficient generators, ozone profiles
│   ├── kepler_analytical.py    # Numba-accelerated analytical Keplerian orbit propagator
│   ├── physics_core.py         # Particle, Simulation, Numba N-body integrator with 1PN relativity + J2/J4
│   ├── refraction.py           # Ray deflection through atmospheric gradient & gravitational lensing
│   └── star_calc.py            # StarCalculator (stellar evolution tracks, mass-luminosity, habitable zones)
│
├── rendering/
│   ├── imgui_renderer.py       # pyimgui ModernGL + GLFW integration backend
│   ├── planetshine.py          # Numba secondary diffuse reflection from planets/moons/rings
│   ├── post_shaders.py         # Post-processing shader programs (HDR, composite, down/up bloom)
│   ├── render_utils.py         # Formatting utilities (speed, distance, time, altitude), color math
│   ├── ring_generator.py       # Procedural 1D FBM noise ring profile generator
│   ├── ring_shadow_filter.py   # Penumbral ring shadow filter kernel generator
│   ├── scattering_lut.py       # ScatteringLUTCache for Sky-View and Multi-Scattering bakes
│   ├── shader_loader.py        # load_shader() with recursive #include resolution and caching
│   ├── shaders.py              # Primary celestial rendering shaders compilation
│   ├── star_catalog.py         # StarCatalog (Gaia DR3 binary parser, GPU buffer streaming)
│   ├── terrain_collision.py    # CameraSurface: terrain vertex elevation, surface walking altitude
│   ├── terrain_quadtree.py     # PlanetQuadtree, QuadtreePatch: Spherified cube quadtree LOD logic
│   ├── terrain_streamer.py     # Asynchronous worker for quadtree tile streaming
│   ├── texture_baker.py        # Offline / runtime baking for procedural rings & surface maps
│   ├── texture_manager.py      # UI and logic for importing, normal map generating, hot-reloading maps
│   └── texture_streamer.py     # Background mipmap generation and texture streaming
│
├── ui/
│   ├── inspector.py            # Body Inspector window (telemetry, physical parameters, orbital elements)
│   ├── menu_bar.py             # Top menu bar (Systems, Physics, View, Render, Tools)
│   ├── modals.py               # Popup dialogs (Graphics Settings, Add Body, Create System, Date Jump)
│   ├── outliner.py             # Left-side celestial hierarchy outliner tree
│   ├── time_hud.py             # Bottom transport controls (play, pause, warp, scrub, date jump)
│   └── viewport_hud.py         # Floating HUD (flight speed indicator, triangle count overlay)
│
└── glsl/
    ├── atmosphere/             # atmo.frag/vert, atmo_godrays.frag, sky_view_lut.frag/vert, scattering_lut.comp
    ├── celestial/              # sphere.frag/vert, terrain.frag/vert, ring.frag/vert, orbit.frag/vert, starfield.*
    ├── common/                 # caster_shadow.glsl, refraction.glsl, ring_shadow_filter.glsl, scattering_bake.glsl
    ├── compute/                # culling.comp (frustum/horizon culling), orbit.comp
    └── post/                   # composite.frag, accum.frag, bloom_*.frag, conv_bloom_*.comp (diffraction spikes)
run.bat                         # Automated setup & launcher batch script (Windows)
run.sh                          # Automated setup & launcher shell script (Linux)
requirements.txt                # Python package dependencies
stellar_forge.spec              # PyInstaller standalone build configuration
```

---

## 4. SCRIPTS & OFFLINE PIPELINES

| Script Path | Purpose |
| :--- | :--- |
| scripts/bake_planet_tiles.py | Reprojects equirectangular planetary maps to tangent-corrected Spherified Cube Quadtree pyramids (data/tiles/{body}/{map}/{face}/{lod}/{x}_{y}.jpg). NVIDIA CUDA-accelerated (via CuPy/RawKernel) with automatic CPU fallback. |
| `scripts/fetch_gaia.py` | Queries ESA Gaia Archive TAP (and VizieR TAP) in RA bands; outputs compact binary `data/gaia/stars.bin`. |
| `scripts/fetch_horizons.py` | Queries JPL Horizons REST API for J2000 state vectors and updates `data/systems/*/system.json`. |
| `scripts/fetch_artemis2_kernel.py` | Downloads Artemis II SPK trajectories into `data/kernels/additional/`. |
| `scripts/accuracy_test.py` | Benchmarks N-body numerical integration accuracy against JPL Horizons real-world orbits. |
| `scripts/perf_test.py` | CPU/GPU profiling suite outputting to `perf_report.txt`. |
| `scripts/benchmark_atmo_shadows.py`| Performance microbenchmark for atmospheric raymarching and ring shadow algorithms. |

---

## 5. DATA STORES & FORMATS

| Path | Format | Notes |
| :--- | :--- | :--- |
| `data/systems/<SystemName>/system.json` | JSON Array | Array of body definitions: `name`, `type`, `m`, `r`, `a`, `e`, `i`, `sv` (state vector), `atmo`, `ring`. |
| `data/systems/<SystemName>/meta.json` | JSON Object | System metadata: `created`, `modified`, `body_count`, `star` properties. |
| `data/kernels/default/` | SPICE (`.bsp`, `.tls`, `.tpc`) | Core planetary ephemeris (`de440s.bsp`), leapseconds (`naif0012.tls`), physical constants (`pck00010.tpc`). |
| `data/kernels/additional/` | SPICE (`.bsp`) | Spacecraft / mission trajectories (e.g., `cassini.bsp`, Artemis II). Auto-detected. |
| `data/gaia/stars.bin` | Packed Little-Endian Binary | `GAIA1` magic (5B) + u32 version + u32 count + array of 32-byte records (`ra, dec, plx, pmra, pmdec, rv, g_mag, bp_rp` all f32). |
| `data/tiles/<Body>/<MapType>/<Face>/<LOD>/<x>_<y>.jpg` | JPEG Quadtree Tiles | Spherified cube face tiles (`Face` 0..5, `LOD` 0..N). |
| `data/graphics_settings.json` | JSON Object | Persisted graphics & camera settings loaded/saved by `InputHandlerMixin`. |

---

## 6. KEY GLOBALS & RUNTIME STATE KEYS

### `app.camera` (Dict)
- `movement_mode`: `0` = Free Flight, `1` = Simple Orbit
- `flight_speed`: Current flight speed in $\text{AU/s}$ (log-scaled with scroll wheel)
- `fov`: Field of view in degrees (default `45.0`)
- `exposure`: Logarithmic HDR exposure multiplier
- `bloom_mode`: `0` = Gaussian Blur, `1` = Diffraction Spikes, `2` = Hybrid
- `spike_count`: Number of diffraction spikes (`4`, `6`, `8`)
- `atmo_quality`: `0` = Off, `1` = Low (2D), `2` = High (Volumetric), `3` = Analytical (Sky-View)
- `atmo_aerial_volume`: Experimental Mode 3 terrain haze acceleration (default `True`); `engine/rendering/aerial_perspective.py`, `engine/glsl/atmosphere/aerial_perspective.comp`, `engine/glsl/common/aerial_{coordinates,lookup}.glsl`. Low-altitude/nadir 32³ frustum volume; grazing high-altitude views use a horizon-focused ground/sky atlas (32×96×48, or 48×128×64 at 1080p+), cosine depth slices, and exact endpoint extinction. Shared solar/planetshine/ringshine transport in `common/scattering_scene.glsl`. Shadow-marched rays bypass the volume and use exact per-star baselines; ring-split passes also use endpoints. Skip unused volume bakes outside the atmosphere. Disable in graphics settings for direct endpoint comparison.
- `tracking_idx`: Integer index of tracked celestial body (or `None`)
- `ringshine_enabled`, `ringshine_oblate_enabled`: Toggle host/moon ringshine and host oblateness. `ringshine_band_count`: 4–1024 radial bands (default 10). Host maps rebake on material revision, normalized ring geometry, flattening, band count, or per-star sine elevation quantized to 1e-5; camera/azimuth/stellar flux changes reuse maps.
- `inspected_idx`: Integer index of currently inspected body in Inspector panel

### `app.shared_state` (Dict)
- `lock`: `threading.Lock()` synchronizing physics simulation and rendering
- `keplerian_mode`: `bool` flag toggling analytical Keplerian propagation
- `ephemeris_mode`: `bool` flag toggling NASA SPICE ephemeris playback
- `timeline_active`: `bool` flag indicating offline timeline rendering
- `timeline_progress`: `float` ($0.0 \to 1.0$)
