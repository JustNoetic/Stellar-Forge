# Stellar-Forge — Project Map & Function Flow

> **Purpose:** A single, token-efficient reference so an LLM (or developer) can locate the
> correct file/function to edit **without** reading every source file. Always consult this
> first before opening source. Keep this file in sync with the codebase when refactoring.

---

## 1. Tech Stack & Entry Point

- **Python 3.10+**, **ModernGL** (OpenGL 4.3 compute), **GLFW**, **PyImGui**, **Numba** (`@njit`,
  `cache=True`, `nogil=True`), **NumPy/SciPy/pyrr**, **Pillow**, **spiceypy** (NASA SPICE).
- **Entry point:** `engine/main.py` → `App()` then `app.run()`. Enables `faulthandler` and writes
  `main_error.txt` on crash. Launch with `python engine/main.py` or `run.bat`.
- **Constants** (AU, year, G=39.477 in AU³/M☉/yr², c, obliquity, solar radius, orbit resolution)
  live in `engine/constants.py` and are star-imported by most modules.

---

## 2. Repository Layout (source only)

```
Stellar-Forge/
├── engine/
│   ├── main.py                 # Entry point. stderr redirect (frozen-safe) + faulthandler + crash logger.
│   ├── app.py                  # ★ MAIN: App class, GLFW window, render loop, ImGui UI
│   ├── path_utils.py           # get_bundled_path() / get_external_path() — frozen/.exe-safe paths
│   ├── __init__.py             # Master package re-exporter with sys.modules aliases
│   ├── core/                   # Base constants, SIMD/Numba math, GLFW input
│   │   ├── constants.py        # Physical & astronomical constants
│   │   ├── math_utils.py      # Numba vector ops, Kepler solvers, orbital<->cartesian
│   │   └── input_handler.py    # GLFW input callbacks & settings persistence (InputHandlerMixin)
│   ├── physics/                # N-body numerical integration & celestial dynamics
│   │   ├── physics_core.py     # ★ IAS15 integrator, GR/J2 forces, physics_loop thread, Simulation class
│   │   ├── kepler_analytical.py# Analytical Jacobi-coordinate Keplerian propagation (Numba)
│   │   ├── atmosphere_physics.py # Rayleigh/Mie/absorption coefficients, scale heights
│   │   └── star_calc.py        # StarCalculator: stellar evolution, HR classification, HZ bounds
│   ├── rendering/              # Graphics pipeline, shader management, Planetshine & baking
│   │   ├── render_utils.py     # Mesh/geometry/frustum helpers + time formatting
│   │   ├── imgui_renderer.py   # ModernGL + ImGui GLFW render bridge
│   │   ├── planetshine.py      # Planetshine & JIT rotation angle calculations
│   │   ├── ring_generator.py   # Procedural ring texture synthesis (SpaceEngine style 1D fBm & resonance gaps)
│   │   ├── texture_baker.py    # Procedural ring baking & texture exporter (apply_procedural_ring_to_body)
│   │   ├── shader_loader.py    # GLSL shader loader with in-memory caching
│   │   ├── shaders.py          # Shader re-exports & uniform bindings
│   │   └── post_shaders.py     # Post-processing shader re-exports (Bloom, HDR)
│   ├── ui/                     # Modular Dear ImGui Interface ('Orion UI')
│   │   ├── __init__.py         # Master render_ui orchestrator
│   │   ├── menu_bar.py         # Top Main Menu Bar (Systems, Physics, View, Render, Tools, Quick HUD)
│   │   ├── time_hud.py         # Bottom Time Transport HUD & Timeline Scrubber
│   │   ├── outliner.py         # Left System Outliner hierarchy panel with search filter
│   │   ├── inspector.py        # Right Tabbed Body Inspector (Physical, Orbit, Atmo, Rings, Cosmetics)
│   │   ├── modals.py           # Centralized dialogs (Graphics settings, Add Body, Create System, Ephemeris setup)
│   │   └── viewport_hud.py     # Viewport floating HUD pill & toasts
│   ├── ephemeris/              # Astronomical data & JPL Horizons/SPICE integration
│   │   ├── system_manager.py   # System I/O, SystemSnapshot, derive_star_properties
│   │   ├── spice_manager.py    # SpiceManager: kernel download/load, SPICE state queries
│   │   └── spk_exporter.py     # Timeline -> Type 2/13 SPK (.bsp) export + async UI glue
│   └── glsl/                   # Dedicated GLSL shader source files
│       ├── compute/            # Frustum culling & orbit compute shaders (.comp)
│       ├── celestial/          # Sphere, orbit, ring, HZ shaders (.vert, .frag)
│       ├── atmosphere/         # Raymarching, Sky-View LUT & analytical slicing shaders (.vert, .frag)
│       ├── common/             # Shared GLSL includes (refraction.glsl, sun_terminator.glsl)
│       └── post/               # Bloom, composite, accumulation & ringshine shaders (.vert, .frag)
├── data/
│   ├── gaia/                   # GAIA star catalog binary (stars.bin) + fetch logs
│   ├── system.json             # Active default system
│   ├── systems/<Name>/{meta.json,system.json}   # System presets (Solar System, Achernar, Ephemeris Mode)
│   ├── kernels/                # SPICE kernels: default/ (core & planetary), additional/ (missions & custom)
│   ├── ephemeris_settings.json # SPICE playback dates & active kernel list
│   ├── graphics_settings.json  # Persistent graphics/quality settings
│   └── horizons_cache.json     # Cache for JPL Horizons REST queries
├── scripts/
│   ├── fetch_horizons.py       # Query JPL Horizons REST → update system JSON state vectors
│   ├── fetch_gaia.py           # Fetch Gaia DR3 (+Hipparcos bright supplement) → data/gaia/stars.bin
│   ├── test_gaia.py            # GAIA starfield sanity checks (synthetic bake + real catalog)
│   ├── accuracy_test.py        # 1-yr integration benchmark vs JPL Horizons ground truth (RTN errors)
│   ├── perf_test.py            # PerfTracker: patches functions to measure startup/frame timing
│   ├── calibrate_moon_albedo.py# Rescale a diffuse map's linear-space mean to a real albedo target (Moon → 0.12)
│   ├── test_skyview_terminator.py # GPU regression: Mode 3 terminator scales with star angular size + refraction (bakes real LUTs, quarter-phase limb geometry)
│   ├── test_skyview_polar_winter.py # GPU regression: Mode 3 & Mode 2 SpaceEngine polar winter solstice parity for ringed planets
│   ├── test_spice.py           # Quick SPICE kernel loader validation
│   ├── test_spk_export.py      # Timeline → .bsp export round-trip vs analytic ground truth
│   ├── ringshine_benchmark.py  # Ground-truth Monte Carlo validator for ringshine radiative transfer
│   └── benchmark_ringshine_oblateness.py # Thesis benchmark & validation of host planet oblateness in ringshine
├── textures/                   # Planet/ring textures (loaded by app.py at startup)
├── exports/                    # Exported cosmetic JSON per body
├── dist/                       # PyInstaller output (generated; not committed to git)
│   └── Stellar-Forge/          # Ready-to-run: Stellar-Forge.exe + _internal/ + data/ + textures/
├── run.bat                     # Windows one-click venv + deps + launch
├── build.bat                   # One-click PyInstaller build + zip for GitHub Releases
├── stellar_forge.spec          # PyInstaller configuration (entry point, DLLs, datas, excludes)
├── imgui.ini                   # Saved ImGui layout
└── README.md / PROJECT_MAP.md  # Docs
```

---

## 3. Module-by-Module Reference

For each file: responsibilities, key symbols (with approximate line numbers), and
"edit here when you want to…".

### 3.1 `engine/main.py` (~32 lines)
- **Frozen-safe stderr redirect** (top of file, before any imports): when running as a
  PyInstaller windowed `.exe`, `sys.stdout`/`sys.stderr` are `None`; they are redirected to
  `os.devnull` / `main_error.txt` before anything else runs.
- `if __name__ == "__main__"`: enables `faulthandler`, wraps `App().run()` in try/except,
  appends to `main_error.txt` on crash. Launch with `python engine/main.py` or `run.bat`.
- **Edit when:** changing startup/error handling/CLI args, or the crash-log path.

### 3.1b `engine/path_utils.py` (~60 lines)
- `_project_root()` — returns project root when running from source, or the `.exe`'s directory
  when frozen (PyInstaller sets `sys.frozen = True` and `sys._MEIPASS`).
- `get_bundled_path(*parts)` — resolves paths for read-only engine assets bundled *inside* the
  `.exe` (e.g. GLSL shaders): uses `sys._MEIPASS` when frozen, project root otherwise.
- `get_external_path(*parts)` — resolves paths for user-editable assets that live *next to* the
  `.exe` (e.g. `data/`, `textures/`, `exports/`): always relative to `_project_root()`.
- **Used by:** `shader_loader.py`, `spice_manager.py`, `input_handler.py`, `app.py`.
- **Edit when:** adding new asset directories, or changing what is bundled vs. external.

### 3.2 `engine/core/constants.py` (16 lines)
- `SECONDS_PER_YEAR`, `C_AU_YR`, `FOV_DEG`, `G` (AU³/M☉/yr²), `OBLIQUITY`,
  `ORBIT_RESOLUTION` (200), `ORBIT_STRIDE_BYTES` (28), `SOLAR_RADIUS_KM`, `AU_TO_KM`,
  `SOLAR_RADII_TO_AU`.
- **Edit when:** tuning physical constants or orbit buffer sizes.

### 3.3 `engine/physics/atmosphere_physics.py` (135 lines)
- `GAS_PROPERTIES` (dict): per-gas Rayleigh coefficients, molar mass, absorption.
- `compute_atmosphere_properties(pressure_atm, temperature_k, composition, gravity_m_s2)` (~L33)
  → returns beta_rayleigh, beta_mie, absorption, scale heights. **Pure Python (cached by app).**
- `compute_mie_coefficients(base_beta, angstrom_exponent)` (~L120).
- **Edit when:** changing scattering physics, gas composition table, atmosphere LUT inputs.

### 3.4 `engine/core/math_utils.py` (197 lines) — all `@njit(cache=True)`
- `fast_cross`, `fast_norm` — vector ops.
- `pole_to_ecliptic(pole_ra_deg, pole_dec_deg)`, `build_equatorial_frame(pole_ecl)` — pole frames.
- `kepler_solve(M, e, tol)` — Newton solver for Kepler's equation.
- `orbital_to_cartesian(a,e,inc,Ω,ω,M,μ)` — classical elements → pos/vel.
- `rotate_equatorial_to_ecliptic` / `rotate_ecliptic_to_equatorial` — frame rotations.
- `get_cartesian_from_keplerian(...)` — parent/body mass + elements → cartesian.
- `axis_angle_rotation(v, axis, theta)` — Rodrigues rotation (used by precession).
- `vector_orbital_to_cartesian(a,e,n_vec,e_vec,M,μ)` — vector-form orbit → cartesian (precession).
- **Edit when:** changing orbital math, frame transforms, Kepler solver.

### 3.5 `engine/physics/kepler_analytical.py` (478 lines) — `@njit(cache=True, nogil=True)`
- `state_to_kepler_vectors(rx,ry,rz,vx,vy,vz,μ)` (~L34) → orbit normal `n` + eccentricity vector `e`.
- `extract_all_kepler_elements(pos, vel, mass, parent_indices, subsys_pos/vel/mass, t_current, G, …)` (~L112)
  → batch osculating elements + J2/GR/third-body precession vectors for whole system.
- `propagate_keplerian_system_numba(elements, parent_indices, tree_indices, t_sim, subsys_init_pos/vel, out_pos, out_vel, mass, subsys_mass)` (~L318)
  → **the analytical propagation kernel**: mean anomaly advance, Kepler solve, precessed vectors →
    Jacobi→Cartesian top-down placement.
- **Edit when:** changing analytical mode, precession model, barycentric placement logic.

### 3.6 `engine/physics/physics_core.py` (2506 lines) — core physics
Numba kernels:
- `compute_custom_forces(arr, …)` (~L24) — **GR 1PN + J2/J4 oblate harmonics** acceleration extras.
- `compute_all_accelerations(arr, …)` (~L235) — pairwise Newtonian + custom forces.
- `compute_keplerian_elements(rx,ry,rz,vx,vy,vz,μ)` (~L293) — single osculating elements.
- `compute_orbit_elements(rel_pos, rel_vel, μ, sl_p_scale, bary, cam_origin, …)` (~L395) — orbit
  geometry for rendering (camera-relative).
- `compute_barycenters(pos, vel, mass, parent_indices, …)` (~L441).
- `compute_all_orbits_batch(pos, vel, mass, parent_indices, …)` (~L473) — batch orbit polylines.
- `_update_hierarchy_core(positions, masses, current_parents, num_bodies, is_star_mask)` (~L873) —
  rebuild parent tree (closest massive parent per body).
- `ias15_sqrt7`, `_ias15_add_cs`, `_ias15_predict_next_step` (~L1949–2010) — IAS15 helpers.
- `ias15_step_numba(arr, …)` (~L2013) — **15th-order Gauss-Radau adaptive step**.

Pure-Python / wrappers:
- `_extract_render_state(sim, num_bodies, out_pos, out_vel)` (~L14).
- `attach_custom_forces(sim, has_j2, has_gr, phys_star_idx, oblate_physics_list)` (~L274).
- `update_hierarchy(sim, num_bodies, current_parents, is_star_mask)` (~L933) → wraps `_update_hierarchy_core`.
- `build_tree_order(parent_indices, num_bodies, positions)` (~L945) → DFS topo order + depths.
- `estimate_min_orbital_period(pos, vel, mass, parent_indices)` (~L992) — shortest osculating
  period via two-body energy; drives auto timeline recording density.
- `physics_loop(sim, num_bodies, shared_state, time_ctrl, running)` (~L1027) — **physics thread main loop**:
    handles system-switch requests, snapshot save/restore, IAS15 vs Keplerian mode switching,
    hierarchy rebuild, step timing, writes `shared_state` under `shared_state["lock"]`.
    Timeline render pass auto-scales `num_steps` to `TIMELINE_SAMPLES_PER_ORBIT` (100) per
    fastest orbit (`time_ctrl["timeline_steps"]` > 0 overrides; capped by
    `TIMELINE_MAX_STEPS` / `TIMELINE_MEM_BUDGET_MB`); publishes `timeline_steps_total`.
- `load_system_from_data(bodies_data_raw)` (~L1710) — builds `Simulation`, attaches custom forces,
  derives visual data / atmo bodies / ring bodies / oblate list → returns **bundle dict** with keys:
    `sim, num_bodies, bodies_data, name_to_idx, visual_data, atmo_bodies, ring_bodies,
     oblate_physics_list, has_j2, has_gr, phys_star_idx, star_idx`.
    Black-hole bodies: `r` property is ignored — event-horizon radius derived from mass (`2.95325008 km × M☉`, written back to `body['r']` for the inspector) and the visual/culling radius is the shadow silhouette `b_c = (3√3/2)·r_s`.

Classes:
- `Particle` (~L2267) — property accessors `x,y,z,vx,vy,vz,m,hash` into `sim.arr`.
- `ParticleList` (~L2310) — `__getitem__`, `__len__`.
- `Simulation` (~L2328) — `__init__`, `reset_integrator_state`, `_resize_buffers`, `add`, `remove`,
  `move_to_com`, `copy`, `integrate(t_target)`. Holds `arr` (Nx10: pos,vel,?,?,mass), `hashes`,
  `has_j2/has_gr/phys_star_idx/oblate_physics_list`, `t`.

**Edit when:** integrator algorithm, forces (GR/J2/J4), hierarchy rebuild, orbit computation,
system loading/bundle shape, Simulation data layout.

### 3.7 `engine/ephemeris/system_manager.py` (510 lines)
Top-level functions:
- `derive_star_properties(mass, metallicity, age)` (~L21) → temp, lum, radius, spectral class.
- `get_spectral_class(temp, lum_class)` (~L142).
- `temperature_to_rgb(temp_k)` (~L181), `rgb_to_hex(r,g,b)` (~L222).

Classes:
- `SystemSnapshot` (~L237) — holds `sim_copy, num_bodies, bodies_data, visual_data, atmo_bodies,
  ring_bodies, oblate_physics_list, has_j2, has_gr, phys_star_idx, star_idx, parent_indices,
  sim_time, time_multiplier, was_paused`.
- `SystemManager` (~L269):
  - Class const `SOLAR_SYSTEM_NAME`.
  - `__init__(systems_dir="data/systems", default_json="data/system.json")`.
  - `_ensure_dirs`, `_system_dir`, `_system_json_path`, `_meta_json_path`.
  - `list_systems()`, `system_exists(name)`.
  - `load_default_system()` → loads `data/system.json`.
  - `load_system_data(name)` → loads `data/systems/<name>/system.json`.
  - `save_system_data(name, bodies_data)`, `save_meta(name, bodies_data)`.
  - `create_new_system_from_props(system_name, star_name, star_props)` — new system from UI.
  - `delete_system(name)`.
  - `store_snapshot(name, snapshot)`, `get_snapshot(name)`, `clear_snapshot(name)` — for system switching.

**Edit when:** system file I/O, presets, snapshot persistence, star property derivation,
spectral classification.

### 3.8 `engine/ephemeris/spice_manager.py` (735 lines)
- `SpiceManager` (~L7):
  - `__init__()` (~L120) — loads `ephemeris_settings.json`, ensures `data/kernels/{default,additional}/`.
  - `cancel_download()`, `_cleanup_partial_downloads()`.
  - `_load_settings()`, `save_settings()`, `_ensure_dirs()`.
  - `find_kernel_path(filename)` — searches `default/`, `additional/`, and root `data/kernels/`.
  - `list_additional_kernels()` — auto-discovers extra BSPs in `additional/` and `kernels/`.
  - `add_additional_kernel(path)` — imports external BSP into `data/kernels/additional/` and marks enabled.
  - `check_missing_kernels()` (~L185) — compares default settings vs on-disk.
  - `download_kernels_async(on_complete)` (~L196) — threaded download with progress hook into `default/`.
  - `load_kernels(force=False)` (~L308) — `spiceypy.furnsh` for all enabled default and additional kernels.
  - `_discover_bodies()` (~L332) — enumerates SPICE IDs, auto-detects spacecraft mission start/end epochs and parents.
  - `datetime_to_et(dt)` (~L374), `get_body_state(body_id, et)` (~L386),
    `get_all_states(et)` (~L422).
  - `get_body_mapping(bodies_data, test_et)` (~L446) — match JSON bodies → SPICE IDs (supports direct `"spice_id"`).
  - `populate_states_fast(et, mapping, pos_out, vel_out, valid_out)` (~L473) — batch state query.
  - `_get_body_properties(body_id, fallback_mass_kg, fallback_radius_km)` (~L513).
  - `build_ephemeris_system(et, template_bodies)` (~L559) — assembles bodies_data for Ephemeris Mode.
  - `get_trajectory_polyline(body_id, observer_id, num_samples, return_times, et_start, et_end)` — samples 3D trajectory polyline with adaptive arc-length & curvature spacing (full mission or sliding window) and optional timestamps.
  - `list_available_bsp_files()` — discovers `.bsp` files across `exports/ephemeris/`, `additional/`, `default/`, and `kernels/`.
  - `inspect_bsp(bsp_path)` — inspects arbitrary `.bsp` kernels (DAF segment coverage, target NAIF IDs, reference centers, sidecar JSON, UTC range).
  - `build_generic_bsp_system(bsp_path, num_samples)` — samples 3D trajectory polylines with adaptive arc-length spacing and timestamps for arbitrary BSP playback.

**Edit when:** SPICE kernel handling, ephemeris playback, body mapping, Ephemeris Mode system build, generic BSP viewer.

### 3.8a `engine/ephemeris/spk_exporter.py`
- `assign_spice_ids(body_names)` — map names → NAIF IDs (real IDs from `SpiceManager.SPICE_BODIES`,
  fictional bodies get unique negatives from `FICTIONAL_ID_START = -100000`).
- `resolve_display_epoch_et(spice_manager)` — ET of the display epoch (2026-01-01 12:00 UTC);
  falls back to analytical 2026 epoch ET (820540869.184 s) if no leapseconds kernel is available.
- `export_timeline_spk(filepath, timeline_raw, timeline_times, body_names, ..., spk_type=)` — encodes
  `shared_state["timeline_raw"]` (f8 physics frame, AU / AU·yr⁻¹) as per-body SPK segments,
  center = 0 (flat barycenter-absolute), frame `ECLIPJ2000` (physics frame == ECLIPJ2000 axes;
  no rotation). **Type 2 (default):** per-interval Chebyshev least-squares fits using
  position AND velocity rows (`_chebyshev_fit_records`, degree 13, tolerance-driven
  interval doubling, `CHEB_TOL_KM`); `_probe_chebyshev_vs_hermite` cross-checks the fit
  against cubic Hermite at sample midpoints to detect under-sampled input (warning surfaced
  in the export message). **Type 13:** Hermite discrete states (pos+vel). Decimation applies
  to Type 13 via `max_states`; auto-fallback to Type 13 on tiny timelines. Writes `<file>.bsp.json`
  sidecar (epoch anchor, UTC/ET time span, fit residuals, probe errors, ID mapping) for re-import.
- `export_timeline_spk_async(app, bodies_data)` — UI entry: lock-guarded snapshot of
  `timeline_raw`/`timeline_times`, background writer thread, outputs to `exports/ephemeris/<System>/`,
  toast via `app._screenshot_toast`.

**Edit when:** changing SPK export format, sampling/decimation, ID scheme, or epoch anchor.

### 3.9 `engine/physics/star_calc.py` (365 lines)
- `StarCalculator` (~L3) — all `@staticmethod`/`@classmethod`:
  - `calc_lum(r,t)`, `calc_rad(l,t)`, `calc_temp(l,r)`.
  - `ms_mass_from_lum(l)`, `ms_lum_from_mass(m)`.
  - `evolve_star(initial_mass, age_val, evo_path, metallicity)` (~L41) — stellar evolution track.
  - `forge(mode, evo_path, mass, metallicity, rot_frac, inclination, age_pct, radius, temp, lum)` (~L133)
    — master stellar property builder (mode: evolution/fixed).
  - inner `temp_to_hex(temp_k)` (~L271).

**Edit when:** stellar evolution models, HR diagram classification, HZ computation, star UI inputs.

### 3.10 `engine/rendering/shaders.py` & `engine/glsl/` — Shader Management & Source Code
- `engine/rendering/shader_loader.py`: In-memory GLSL loader (`load_shader`) loading source files from `engine/glsl/`.
- `engine/rendering/shaders.py`: Dynamically re-exports loaded GLSL shaders:
  - `culling_compute_shader` (`glsl/compute/culling.comp`) — GPU frustum/occlusion culling compute.
  - `sphere_vertex_shader` / `sphere_fragment_shader` (`glsl/celestial/sphere.*`) — PBR planet/star spheres (with ray refraction `compute_refraction_angle` & vertex bounding expansion). Fully-subpixel fragments (f_subpixel_factor > 0.999) are discarded so the 3 px min-size-clamped mesh never writes phantom depth over the subpixel star sprite (transit black-out fix).
  - `orbit_compute_shader` (`glsl/compute/orbit.comp`), `orbit_*` (`glsl/celestial/orbit.*`).
  - `ephem_orbit_*` (`glsl/celestial/ephem_orbit.*`).
  - `ring_*` (`glsl/celestial/ring.*`), `hz_*` (`glsl/celestial/hz.*`).
  - `atmo_*` (`glsl/atmosphere/atmo.*`), `atmo_lut_*`, `multi_scatter_lut_*`, `sky_view_lut_*` (`glsl/atmosphere/sky_view_lut.*`) — Atmosphere shaders supporting Mode 1 (Low 2D shadows), Mode 2 (High Volumetric raymarching), and Mode 3 (Analytical Sky-View LUT with selectable resolution [Low 192×108, Medium 256×256, High 384×216] and switchable shadow integration via `u_atmo_shadow_method`: Method 0 = Station-Locked Slicing partitioned by CPU-baked cell boundaries fixed in ring coordinate u — `bake_station_cells` in `render_utils.py` snaps them to ring profile discontinuities — with per-cell footprint-filtered 1D shadow texture `u_ring_shadow_tex`, multi-scattering deficit subtraction, and normalized slice shading matching Mode 2 pitch-black ground truth; Method 1 = Uniform Stochastic Raymarching with $N$ uniform view-distance steps [4–64, default 24], stochastic jitter across screen pixels via Spatiotemporal Blue Noise (NVIDIA STBN 128x128x64 3D texture with Cranley-Patterson Weyl sequence rotation), evaluating parcels at jittered sample points to eliminate onion banding, and un-blurred linear LOD ring sampling preserving fine gaps like the Cassini division on glancing chords; Method 2 = Bounded Subtraction (Blackrack) porting Kitten Space Agency's approach with deterministic midpoint quadrature, analytical bounding cylinder culling for celestial moon/planet eclipse cones, step and screen-space derivative footprint LOD for ring shadows, and full single and multiple scattering deficit subtraction reaching pitch-black shadows). All modes share the terminator in `common/sun_terminator.glsl`: stellar-disc rise/set per parcel with penumbra from the star's angular radius widened by refraction (`eff_star_rad = sin_star + max_bend` from `u_star_pos_local[s].w`); Mode 1/2 call it in the raymarch loop, Mode 3 in the Sky-View bake and every shadow accumulation step. All modes also share analytical moon/planet eclipse casters via `common/caster_shadow.glsl` (`get_oblate_radius` / `casterShadowTerm` / `compute_caster_shadow` — oblate caster+star projection, penumbra cone, atmospheric refraction ring & Danjon tint): Mode 1/2 call it through `compute_shadow` (which adds ring-plane occlusion), Mode 3 bakes it into the Sky-View LUT per (step, star) together with multi-star illumination (up to 4 precomputed star slots; `u_num_stars`/`u_stars_poles_obl` read from the global SceneData UBO binding 1), properly transforming parcel positions from oblate spherical coordinates (`fromSphericalSpace`) to eliminate pole-scaling offsets on oblate planets like Saturn and Jupiter, with pre-loop ray-cone caster culling and external ring plane culling so inactive casters and ring planes are bypassed during raymarching; step count is UI-configurable via `u_num_steps` ("Sky-View LUT Steps" slider in graphics settings, defaulting to 24, clamped 4–64; defaults to 32 if unprovided); resolution is UI-configurable via "Sky-View LUT Resolution" combo (Low 192×108, Medium 256×256, High 384×216; stored as `atmo_sky_view_res`). The Mode 3 bake also evaluates external parent ring shadows (`compute_external_ring_shadow` in `sky_view_lut.frag`, skipping circumplanetary rings with distance to body center < 1e-4 AU so `atmo.frag` can slice them) and folds in planetshine/moonshine (instance SSBO binding 2: `instances[u_body_idx*7+4]` dir / `+5` color, treated as one extra directional light) and host-planet ringshine (`u_ringshine_map` unit 8 with `textureLod`, star-independent accumulators + per-star map lookups), matching Mode 1/2 exactly. Sky-View textures (`sky_view_tex`, `sky_view_trans_tex`, and `sky_view_star_tex`) are allocated in FP32 (`dtype='f4'`) to eliminate subnormal precision collapse and color banding at low brightness. The bake writes per-star in-scatter slices to LUT attachments 2-5 (`u_sky_view_star_lut`, units 15-18); Mode 3's ring-shadow slicing then solves each star's shadow interval independently and normalizes each deficit against that star's own slice — one star's ring shadow can never darken another star's light, and every star casts its own ring shadow. Both Mode 2 and Mode 3 share the SpaceEngine polar winter solstice model (`eval_polar_winter`) on ringed planets (Saturn), clearing aerosol Mie haze (0.05) and boosting pure Rayleigh in-scatter with azure tint (`vec3(0.65, 0.95, 2.5)`) at the winter pole. Validate with `python scripts/test_skyview_terminator.py`, `python scripts/test_skyview_eclipse.py`, `python scripts/test_skyview_res.py`, `python scripts/test_atmo_blackrack.py`, and `python scripts/test_skyview_polar_winter.py`.
  - `point_celestial.*` (`glsl/celestial/`) — Subpixel point-light quad pass (apparent_px < 3.0); refracts the body's apparent position via `apply_refraction` (parallax-weighted with bracketed-bisection inverse apparent solver `solve_refraction_apparent` and solid-body occlusion guard), cross-fades against the mesh over apparent_px ∈ [2.0, 3.0]. Star sprites apply analytic circle-circle transit/occultation occlusion against all casters (true (Rp/Rs)² transit depth, partial phases for subpixel eclipses).
  - `starfield.*` (`glsl/celestial/`) — GAIA catalog point sprites; per-star positions computed per frame ON THE GPU in double precision (static f8 pos0 + f4 pm/color/flux buffers, `dvec3 u_origin` + `double u_t` uniforms — no CPU packing, no per-frame VBO upload); per-eye inverse-square flux falloff `I = flux·u_flux_calib / d_eye²` in the vertex shader (`u_eye` + `u_flux_calib` uniforms; calib anchors M_G=4.67 @ 1 AU to the system sun's sprite peak `0.8333·(h·fov)²/4`, so one exposure governs planets, sun, and stars alike); FIXED PSF sprite size (peak ∝ flux — true magnitude scale; bloom grows bright stars); atmospheric refraction via `apply_refraction` (same parallax-weighted model + limb occlusion guard as point lights); far-plane distance clamp (eye-relative) + manual behind-camera degeneration; **vertex-stage culling** in `starfield.vert` — frustum test with sprite-extent padding and exposure-threshold culling (peak center-pixel signal < 1e-6, mirroring the frag's limiting-magnitude fade) evaluated on the APPARENT (refracted) position so refraction-lifted stars stay visible; `gl_FragDepth = 0.999999` for log-depth occlusion; additive blend, HDR, feeds bloom/diffraction spikes.
  - `common/refraction.glsl` — Shared refraction math: `compute_refraction_angle` (parallax-weighted apparent displacement) vs `compute_refraction_total` (un-parallaxed total ray turn); `compute_refraction_angle` has a dedicated observer-inside closed form for s_min < 0 (object above the camera's horizontal plane): `R(e) = 0.5·delta_cam·erfcx(√(|C|/2H)·sin e)` via `refract_erfcx` (A&S 7.1.26), matching standard refraction tables and staying continuous with the limb model at e = 0; `solve_refraction_apparent` inverts the deflection for point lights/orbits/stars via 12-step bracketed bisection on [0, alpha_0] (the fixed point θ = α(V_app(θ)) is always bracketed there since α is monotone non-increasing — the old 3-step damped Newton-Raphson under-converged for distant observers, s_min/H ≫ 1, making stars stop refracting and sink into the limb); `refract_chord_blocked` tests apparent-ray periapsis to occlude bodies behind the solid planet. The solve deliberately has NO `s_min <= 0` gate (above-horizon objects from near-surface cameras must refract — the `0.5·(E_d+E_0)` factor handles the camera-past-periapsis geometry) and its occlusion test clamps to the forward-ray periapsis; removing either would resurface the starfield/point-light "downward flick" discontinuity at the camera's horizontal plane.
    For above-horizon rays (`s_min < 0`), `compute_refraction_total` delegates to `compute_refraction_angle`: mesh callers clamp the bend anchor to the camera, so they require the same effective displacement as points, not the behind-camera limb Gaussian. Regression: `python scripts/test_refraction_math.py` executes the shipped GLSL on GPU and checks point/anchored-mesh alignment for ground elevations, a space limb, and the atmospheric cutoff.
    Also hosts the **gravitational lensing** math: `u_grav_lens_*` uniforms (center/rs/radius/type/enabled/strength/spin/pole), `compute_gravitational_deflection` (Einstein weak-field + 2PN + strong-field divergence, Kerr `b_c(φ)`, capture → `is_shadow`), and `apply_gravitational_deflection` (+ Lense-Thirring drag), composed into `apply_refraction` / `apply_refraction_eye` after the atmospheric bend (captured rays converge onto the lens center so the shadow disk depth-occludes them). Validate with `scripts/test_lensing_math.py` (GPU test of the shipped GLSL).

**Edit when:** editing GLSL shader logic inside `engine/glsl/` or uniform bindings in `shaders.py`.

### 3.11 `engine/rendering/post_shaders.py` — Post-Processing GLSL
- Re-exports post-processing shaders loaded from `engine/glsl/post/`:
  - `bloom_downsample.frag`, `bloom_upsample.frag` — HDR bloom pyramid.
  - `composite.frag` — tonemap (ACES) + exposure + bloom composite.
  - `accum.frag` — jitter accumulation resolve for screenshots and scene.
  - `ringshine_map.frag` — dynamic per-frame ringshine irradiance map bake (4-point area-weighted Gauss-Legendre quadrature, shadow CDF integration, and runtime-switchable host planet oblateness via oblate shadow ellipse projection, $\rho(\lambda)$ surface altitude, and 3D form factor LUT sampling).

**Edit when:** bloom, tonemapping, exposure, accumulation, ringshine dynamic bake map.

### 3.12 `engine/rendering/render_utils.py`
- `hex_to_rgb`, `sample_gradient` — color helpers.
- `format_flight_speed(speed_au_s)`, `format_time_speed(multiplier)`, `format_distance_au(dist_au)`,
  `format_altitude(alt_km, alt_au)`, `format_sim_time(t_years)`,
  `sim_time_from_date(y,m,d,h,mn)` — UI formatting helpers.
- `compute_ring_culling(...)` (~L121, `@njit`) — ring visibility.
- `extract_frustum_planes(vp)` (~L174, `@njit`) — frustum planes from VP matrix.
- `create_icosphere_mesh(subdivisions=4)` (~L189) — UV/ico sphere geometry.
- `generate_ring_shadow_grad(...)` (~L242), `generate_ring_geometry(pole_render, min_r, max_r)` (~L259).
- `rebuild_ring_render_group(bi, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)` (~L301).
- `build_ringshine_3d_lut_numba(...)` — Precomputes 3D Form-Factor LUT $(256 \times 256 \times 16)$ covering oblateness $f \in [0.0, 0.3]$ via parallel Numba JIT.

**Edit when:** mesh generation, ring geometry, 3D ringshine form-factor LUT, frustum culling math, time formatting.

### 3.12b `engine/rendering/star_catalog.py` — GAIA starfield layer (render-only)
- `StarCatalog` (~L40) — loads `data/gaia/stars.bin` (from `scripts/fetch_gaia.py`):
  - `load(path)` — reads header + structured records; silently disabled when missing.
  - `_bake(rec)` — ICRS→ecliptic (rotation about X by `OBLIQUITY`)→render frame `(x, z, -y)`;
    3D positions from parallax; proper motion as linear AU/yr velocity; J2016.0→J2000
    back-propagation; absolute G magnitude; BP−RP→Teff→RGB (vectorized Tanner Helland,
    matches `system_manager.temperature_to_rgb`); pre-scaled intrinsic flux `10^(-0.4·M_G)·1e13` (constant per star — the eye-distance 1/d² falloff and HDR calibration live in `starfield.vert` via `u_eye`/`u_flux_calib`, keeping the pack cache independent of camera motion).
  - `pack_render(t_years, frame_origin)` (~L170) — per-frame interleaved f4 (N,7)
    `[rel_xyz, rgb, intensity]` via `_pack_kernel` (`@njit(parallel=True, cache=True)`);
    caches on (origin, t); `fresh` flag gates the VBO upload in `app.py`.
- **Not a physics body** — never enters `Simulation`/`shared_state`; drawn once per frame
  after the sphere pass (depth-test on / write-off, additive blend), before atmosphere pass 1.
- **Runtime packing is GPU-side**: `pos0_f8` (f8, kept for the vertex-shader dvec3 math),
  `pm_vec`, `rgb`, `flux_scaled` are uploaded as static VBOs once at startup (app.py); the
  per-frame position is `pos0 + pm·t − origin` evaluated in `starfield.vert` (dvec3). Only two
  uniform writes per frame. `pack_render()`/`_pack_kernel` (CPU, f8→f4) remain for the offline
  validation in `scripts/test_gaia.py` — note the CPU path packs from the f4-cast `pos0`, so it
  is slightly LESS precise than the GPU path (GPU verified bit-exact vs an f8 numpy reference).
- **Edit when:** changing starfield data flow, brightness model, or adding interstellar
  camera support (needs an f8 path / per-star re-anchoring near stars).

### 3.13 `engine/app.py` (6404 lines) — ★ the big one
**Top-level (module scope):**
- `NumpyEncoder(json.JSONEncoder)` (~L10) — JSON encode ndarrays.
- `_NullCtx`, `_perf`/`_PERF_INSTALL_TRACKER` (~L42–67) — perf instrumentation hooks
  (active when env `STELLAR_FORGE_PERF=1`).
- `ModernGLImGuiRenderer` (~L75) — custom ImGui renderer (font texture, VBO/IBO resize, render, shutdown).
- `ModernGLGlfwRenderer(GlfwRenderer)` (~L205) — GLFW integration wrapper.
- `compute_max_bend(caster_r_au, atmo_h_km)` (~L225) — eclipse shadow bending limit.
- `compute_ring_coplanar_masks(...)` (~L234, `@njit(cache=True)`) — coplanar ring-plane masks.
- `get_cached_atmosphere_properties(atmo, mass_sm)` (~L253) — memoized atmosphere props.
- `compute_planetshine_numba(pos, radii, colors, is_star, star_positions, star_colors, star_lums, star_radii, hdr_enabled, ring_params, ring_normals, ring_colors, planetshine_enabled, ringshine_enabled)` (~L278, `@njit`) — CPU planetshine & moon ringshine precompute.
- **Active Refraction Uniform Setup** (~L4318) — passes distance-agnostic refraction parameters (`u_refract_center`, `u_refract_radius`, `u_refract_max_bend`, `u_refract_scale_height`, `u_refract_pole`, `u_refract_oblateness`) dynamically across active shaders (`prog_spheres`, `prog_rings`, `prog_atmo`, `prog_orbit`, etc.).
- **Active Gravitational Lens Uniform Setup** (immediately after) — scores all bodies (primary + comparison; BH ×1000, NS ×100, inspected ×5000) by `rs_km / cam_dist_au`, picks the active lens, and pushes `u_grav_lens_*` (center, rs = 2.95325008 km × M☉, type 0=Star/1=WD/2=NS/3=BH, enabled, strength, spin, pole) to every refraction-consuming program. Settings: `grav_lensing_enabled` / `grav_lensing_multiplier` (Graphics & Quality modal).

**`class App`** (~L487) — the engine:
- `__init__` (~L488) — camera dict (target, distance, fov, yaw/pitch/roll, exposure, hdr, tracking_idx,
  inspected_idx, add_mode, modals…), loads `graphics_settings.json` via `load_settings()`.
- `load_settings()` (~L590), `save_settings()` (~L603).
- **GLFW callbacks:** `scroll_callback` (~L634, zoom/speed/FOV by movement_mode), `mouse_button_callback` (~L672,
  pick + drag), `cursor_pos_callback` (~L695, orbit/pivot by movement_mode: Free Flight vs Simple Orbit),
  `char_callback` (~L738), `key_callback` (~L741, speed/exposure/roll/pause), `resize_callback` (~L761).
- **`run()`** (~L774) — the whole application lifecycle. Major phases inside:
  1. Perf tracker setup (~L778).
  2. `SystemManager`, `SpiceManager`, load default system, build main + comparison bundles via
     `load_system_from_data` (~L786–840). Comparison system = duplicate for side-by-side mode.
  3. Load `textures/` planet/normal/specular + ring textures (~L850).
  4. Initialize `shared_state` / `shared_state_cmp` dicts (pos, vel, mass, parent_indices,
     tree_indices, tree_depths, lock, hierarchy_version, oblate lists, flags) (~L1060–1190).
  5. Create ModernGL context, window, ImGui, compile all shader programs
     (`prog_spheres`, `prog_rings`, `prog_atmo`, `prog_culling_compute`, `prog_orbit_compute`,
      `prog_orbit`, `prog_ephem_orbit`, `prog_hz`, `prog_atmo_lut`, `prog_multi_scatter_lut`,
      - bloom/composite programs) (~L1209+).
  6. `build_eclipse_lut(ctx)` (~L1318) — eclipse LUT texture (oblate star aware).
  7. Build ring render groups, body instance buffers, orbit buffers (~L1372–2150).
  8. `build_atmo_lut(atmo, mass_sm, is_cmp)` (~L1765) — transmittance + multi-scatter LUTs per body.
  9. Spawn **physics thread** running `physics_loop(...)` (~L1691 region).
  10. **Main render loop** (`while not glfw.window_should_close`): per-frame work incl.
      - System/comparison state snapshot from `shared_state` under lock (~L2359).
      - Camera matrix, tracking, hierarchy refresh (~L2400–2660).
      - `compute_planetshine_numba` → planetshine dirs/colors into instance buffer (~L2819).
      - GPU culling compute dispatch (`prog_culling_compute.run`) (~L2877).
      - Orbit polyline compute + draw (~L2880–3264).
      - Sphere PBR draw (LOD: lo/hi/ultra via indirect draw cmds).
      - Orbit polyline compute + draw.
      - `render_atmosphere_pass(clip_mode)` — two-pass with dual-source blending (behind/in front of rings); Mode 3 bakes dynamic Sky-View LUT (selectable resolution: Low 192×108, Medium 256×256, High 384×216; `prog_sky_view` into `sky_view_fbo`, texture unit 12) aligned with parent planet axial tilt.
      - Rings render, Habitable Zones, second atmosphere pass.
      - Dynamic Cloud Layer pass (~L4994) — rendered over the atmosphere with physical view-transmittance fading so the volumetric atmosphere is preserved behind semi-transparent clouds while distant clouds naturally dissolve into horizon haze.
      - Modular Dear ImGui UI ('Orion UI'): delegated to `engine.ui.render_ui(...)`:
        - Top Menu Bar: Systems, Physics, View, Render, Tools, Quick Actions (screenshot & settings)
        - System Outliner (Left Panel): hierarchy tree, quick search filter, body add/delete buttons
        - Body Inspector (Right Panel): tabbed layout (Overview, Orbit, Atmosphere, Rings, Cosmetics)
        - Time Transport HUD (Bottom Bar): playback transport, speed slider, UTC date/time, Jump in Time popup, timeline scrubber
        - Centralized Modals: Graphics & Quality settings, Add Body, Create System, Ephemeris setup/download
        - Viewport HUD: minimal floating camera info pill
      - Accumulation resolve, post-accum orbits/HZ, bloom + composite.
  11. Teardown: stop physics thread, release GL objects, save settings, GLFW destroy.

**Edit when:** rendering pipeline, camera, input, LUT building, instance buffer layout, post-processing.

### 3.14 `engine/ui/` — Modular Dear ImGui Interface ('Orion UI')
- `__init__.py`: Master `render_ui` orchestrator coordinating all UI passes when `app.ui_visible` is True.
- `menu_bar.py`: `render_main_menu_bar` — Systems switch/create/delete, Physics modes (IAS15, Keplerian, SPICE), View (Free Flight, Simple Orbit, FOV, UI toggles), Render (MSAA, HDR, Atmosphere, Refraction, Planetshine, Ringshine), Tools (Ephemeris, Comparison, Distance Units, Accumulation), Quick HUD (physics badge, F12 Screenshot button, Settings cog).
- `time_hud.py`: `render_time_hud` — Centered floating transport bar with Play/Pause, Forward/Backward, formatted speed readout & logarithmic slider, 1x reset, UTC date display, and Jump in Time popup / timeline playback & scrubbing.
- `inspector.py`: `render_body_inspector` — Right-anchored tabbed inspector with Edit Mode (interactive editing of physical/stellar properties and osculating Keplerian orbital elements with reference frame selection, dynamic Roche limit checks, Darwin/Maclaurin oblateness auto-calculation, and real-time CRUD synchronization), real-time camera Distance & Altitude readouts (accounting for oblate spheroid surface radius $R(\mathbf{u})$, flattening $f$, and axial spin pole), Overview, Orbit, Atmosphere, Rings, and Cosmetics.
  - Orbit gravitational-limit readouts: instantaneous Hill radius from live body/parent mass and separation (same approximation as hierarchy selection); fluid Roche distance from the parent's center for the selected body's mass/radius, with proposed edit values and periapsis warning. No limit readouts for barycenters or parentless bodies; the existing edit Apply restriction remains unchanged.
  - Stellar Overview exposes optimistic HZ bounds matching the overlay (`sqrt(lum/1.78)`, `sqrt(lum/0.32)`); Atmosphere exposes its existing equilibrium temperature and cached gas scale height/molar mass.

- `modals.py`: `render_modals` — Centralized modal dialogs for Graphics & Quality Settings (incl. atmospheric quality mode 0–3, Mode 3 Sky-View LUT Resolution combo [Low 192×108, Medium 256×256, High 384×216], Mode 3 Shadow Method combo [Station-Locked Slicing vs Uniform Stochastic Raymarching], dynamic slider [Shadow Slicing Cells 2–32 vs Shadow Ray Steps 4–64], Stochastic Raymarching noise toggle, Noise Type combo [IGN vs STBN], atmospheric refraction, and gravitational-lensing toggle/strength), Add Orbiting Body, Create New Star System, Ephemeris Kernel Setup, SPICE Downloader, and Import Ephemeris Kernel (.bsp).
- `viewport_hud.py`: `render_viewport_hud` — Viewport floating camera mode & flight speed indicator pill.

### 3.15 `scripts/`
- `accuracy_test.py` — `get_parent_center(body_name, parent_name)`, `main()`. Runs 1-yr forward integration vs JPL Horizons, prints RTN km error table.
- `fetch_horizons.py` — `get_parent_center`, `_load_cache`/`_save_cache`, `query_horizons(body_id, center, start_time, stop_time)`, `parse_state_vector(response_text)`, `main()`. Writes `data/horizons_cache.json` + updates system JSON.
- `fetch_gaia.py` — fetches a magnitude-limited Gaia DR3 subset (24 RA bands, TAP sync) plus a Hipparcos bright-star supplement (V < 2.5; Gaia photometry is saturation-broken for these, e.g. Sirius A), propagated to the J2016.0 epoch. Writes `data/gaia/stars.bin` (32-byte records: ra/dec/plx/pmra/pmdec/rv/G/BP-RP, f4).
- `fetch_artemis2_kernel.py` — fetches Artemis II trajectory state vectors from JPL Horizons and encodes them into a standard NAIF Type 9 SPK kernel (`data/kernels/artemis2.bsp`).
- `test_spk_export.py` — comprehensive unit and round-trip validation tests for timeline SPK kernel export (Type 2 Chebyshev and Type 13 Hermite).
- `test_bsp_viewer.py` — unit and integration tests for generic BSP inspection, trajectory polyline generation, fast state evaluation, and time round-trips.
- `perf_test.py` — `class PerfTracker` (~L40), `patch_function(module, name, tracker, label)` (~L113), `main()`. Activated via `STELLAR_FORGE_PERF=1`; `--gpu` adds per-pass GL timer queries (env `STELLAR_FORGE_GPU_PERF=1`, instrumented passes via `_perf_gpu_begin/_end/_flush` in `app.py`) and `--target-body NAME` parks the camera on a body (default `Saturn` when `--gpu`).

---

## 4. Cross-Module Function Flow

### 4.1 Startup
```
main.py
  └─ App.__init__()            [app.py L488]  loads settings, camera defaults
  └─ App.run()                 [app.py L774]
       ├─ SystemManager().load_default_system()   → bodies_data_raw (dict list)
       ├─ load_system_from_data(bodies_data_raw)  [physics_core L1710]
       │     └─ Simulation() + add() per body + attach_custom_forces()
       │     └─ returns bundle{sim, num_bodies, bodies_data, name_to_idx,
       │                       visual_data, atmo_bodies, ring_bodies,
       │                       oblate_physics_list, has_j2, has_gr, phys_star_idx, star_idx}
       ├─ build shared_state / shared_state_cmp dicts (np buffers + threading.Lock)
       ├─ ModernGL context + compile shaders.py / post_shaders.py programs
       ├─ build_eclipse_lut(ctx)                  [app.py L1318]
       ├─ build_atmo_lut(atmo, mass_sm) per body  [app.py L1765]
       │     └─ get_cached_atmosphere_properties() [app.py L253]
       │           └─ atmosphere_physics.compute_atmosphere_properties()
       ├─ ring geometry via render_utils.generate_ring_geometry / rebuild_ring_render_group
       └─ spawn thread: physics_loop(sim, num_bodies, shared_state, time_ctrl, running)
```

### 4.2 Physics Thread (`physics_core.physics_loop`, L986)
```
loop:
  read time_ctrl["multiplier"], ["paused"]   (under shared_state["lock"])
  handle shared_state["system_switch_request"]:
     save SystemSnapshot (sim.copy()) → SystemManager.store_snapshot()
     restore from snapshot OR load_system_from_data(new raw) → swap sim/num_bodies/bundle
     reattach custom forces (J2/GR), rebuild hierarchy
  rebuild parent tree:
     update_hierarchy(sim, …) → _update_hierarchy_core()   [physics_core L873]
     build_tree_order(parent_indices, …)                   [physics_core L945]
  mode = shared_state["mode"]:
     "ias15":  ias15_step_numba(sim.arr, …)                 [physics_core L2013]
               (internally calls compute_all_accelerations → compute_custom_forces
                for GR 1PN + J2/J4 oblate harmonics)
     "kepler": extract_all_kepler_elements(...)             [kepler_analytical L112]
               propagate_keplerian_system_numba(...)        [kepler_analytical L318]
     "ephemeris": SpiceManager.populate_states_fast(et, mapping, pos_out, vel_out)
  write shared_state["pos"/"vel"/"mass"/"parent_indices"/"tree_indices"/"tree_depths"]
     and increment "hierarchy_version" under lock
  advance sim.t
```

### 4.3 Render Thread (`App.run` main loop, ~app.py L2359+)
```
per frame:
  snapshot shared_state → pos_snap, vel_snap, mass_snap, parent_snap, tree_indices
  (comparison: also shared_state_cmp)
  camera update (tracking, orbit/pan from callbacks, roll, FOV, exposure)
  if hierarchy_version changed: rebuild buffers
  compute_planetshine_numba(...) → instance attrs [app.py L295]
  pack all_instances buffer (pos, color, radius, is_star, flags, planetshine dirs/colors)
  prog_culling_compute.run() → writes indirect draw cmd buffer + visibility
  if show_orbits: prog_orbit_compute.run() / ephemeris orbit draw
  draw spheres (LOD indirect): prog_spheres with eclipse LUT, planetshine, ringshine uniforms
  render_atmosphere_pass(1)  [behind rings]   ─┐
  draw rings (prog_rings)                       ├─ two-pass atmo/ring ordering
  draw habitable zones (prog_hz)                │
  render_atmosphere_pass(2)  [in front]        ─┘
  ImGui: top bar, system menu, mode radio, inspector, modals
  TAA resolve (taa_resolve_shader_fs) when enabled
  bloom pyramid (5 MIP fbos) + composite (tonemap + exposure)
  glfw.swap_buffers / poll_events
```

### 4.4 System Switch (IAS15 ↔ Keplerian ↔ Ephemeris)
- UI mode radio in `App.run` (~L4219) sets `shared_state["mode"]` + may post
  `system_switch_request` (with `preserve_state`, snapshots, or new raw bodies).
- `_trigger_ephem_switch()` (~L4146) → `SpiceManager.download_kernels_async(on_complete=_on_spice_ready)`
  → on completion `build_ephemeris_system(et, template_bodies)` produces new bodies_data_raw →
  posted as switch request → `physics_loop` loads it.
- `_trigger_ephem_exit(to_keplerian)` (~L4184) restores the pre-ephemeris snapshot.
- Snapshots persisted via `SystemManager.store_snapshot/get_snapshot`.

### 4.5 Body Inspector / Export
- Inspector panel (~app.py L4544) reads osculating elements via
  `physics_core.compute_keplerian_elements` / `kepler_analytical` extraction.
- Cosmetics edits update `visual_data` + ring precomputed; ring edits call
  `render_utils.rebuild_ring_render_group`.
- **Export button:** single body → `exports/<name>.json`. **`\`+Export** (Solar System only) →
  dumps all cosmetics into `data/system.json` via `SystemManager.save_system_data`.

---

## 5. Shared State & Data Contracts

### 5.1 `shared_state` dict (built in `App.run`, consumed by `physics_loop`)
Keys: `pos (N,3) f8`, `vel (N,3) f8`, `mass (N,) f8`, `parent_indices (N,) i32`,
`tree_indices (N,) i32`, `tree_depths (N,) i32`, `lock` (`threading.Lock`),
`hierarchy_version` (int), `is_star_mask` (bool), `has_j2`, `has_gr`, `phys_star_idx`,
`oblate_physics_list`, `mode` ("ias15"|"kepler"|"ephemeris"), `system_switch_request` (dict|None),
`syncing` (bool), plus comparison counterparts in `shared_state_cmp`.

### 5.2 `bundle` dict (from `load_system_from_data`)
`sim, num_bodies, bodies_data (list[dict]), name_to_idx (dict), visual_data (list of
[pos3,color3,radius, ...] f4 rows), atmo_bodies, ring_bodies, oblate_physics_list,
has_j2, has_gr, phys_star_idx, star_idx`.

### 5.3 `bodies_data` item shape (JSON / in-memory)
Common keys: `name`, `type` (Star/Planet/Moon/DwarfPlanet/Asteroid…), `mass` (M☉),
`a, e, inc, Omega, omega, M` (initial Kepler elements), `radius` (R☉ or km),
`parent`, `texture`/`texture_slices`, `rings` (list), `atmosphere` (composition/pressure),
`pole_ra`/`pole_dec`, `oblateness`/`J2`/`J4`/`R_eq`, rotation period, cosmetics.

### 5.4 Instance buffer layout (`all_instances`, `INSTANCE_FLOATS`)
Per-body row of floats fed to `prog_spheres` / `prog_culling_compute`. Fields include:
`[0:3] pos, [3:6] color, [6] radius, [7] ?, [8] is_star, … [16:19] planetshine_dir,
[20:23] planetshine_color, …`. (Exact count = `INSTANCE_FLOATS` defined in app.py.)
**When editing instance attributes, update both the CPU pack in `App.run` and the GLSL
`sphere_vertex_shader`/`culling_compute_shader` accordingly.**

---

## 6. "Where do I edit…?" Quick Lookup

| Want to… | File | Symbol / Region |
|---|---|---|
| Change launch/crash handling | `engine/main.py` | top-level |
| Resolve paths in frozen .exe | `engine/path_utils.py` | `get_bundled_path`, `get_external_path` |
| Add a bundled asset (inside .exe) | `stellar_forge.spec` → `datas`, then `get_bundled_path` in code | — |
| Add an external asset (next to .exe) | `build.bat` (xcopy step) + `get_external_path` in code | — |
| Tune physical constants | `engine/core/constants.py` | — |
| Change Newtonian/GR/J2/J4 forces | `engine/physics/physics_core.py` | `compute_custom_forces`, `compute_all_accelerations` |
| Change IAS15 integrator | `engine/physics/physics_core.py` | `ias15_step_numba` + helpers |
| Change analytical Keplerian mode | `engine/physics/kepler_analytical.py` | `propagate_keplerian_system_numba`, `extract_all_kepler_elements` |
| Change hierarchy/parent-tree logic | `engine/physics/physics_core.py` | `_update_hierarchy_core`, `build_tree_order` |
| Change physics thread / system switching | `engine/physics/physics_core.py` | `physics_loop` |
| Tune timeline recording density | `engine/physics/physics_core.py`, `engine/ui/modals.py` | `TIMELINE_SAMPLES_PER_ORBIT`, `estimate_min_orbital_period`, resolution combo |
| Change how systems load from JSON | `engine/physics/physics_core.py` | `load_system_from_data` |
| Change system file I/O / presets | `engine/ephemeris/system_manager.py` | `SystemManager`, `SystemSnapshot` |
| Change star classification/evolution | `engine/physics/star_calc.py`, `engine/ephemeris/system_manager.py` | `StarCalculator.forge`, `derive_star_properties` |
| Change SPICE / Ephemeris Mode | `engine/ephemeris/spice_manager.py` | `SpiceManager` |
| Export recorded timeline → .bsp SPK kernel | `engine/ephemeris/spk_exporter.py`, `engine/ui/time_hud.py` | `export_timeline_spk`, `export_timeline_spk_async`, "Export .bsp" button |
| Validate SPK timeline export round-trip | `scripts/test_spk_export.py` | `main` |
| Change scattering physics / gas table | `engine/physics/atmosphere_physics.py` | `compute_atmosphere_properties`, `GAS_PROPERTIES` |
| Change orbital math / frame rotations | `engine/core/math_utils.py` | — |
| Change sphere/ring/atmo/orbit/HZ shaders | `engine/glsl/` (`celestial/`, `atmosphere/`, `compute/`) | GLSL files loaded via `engine/rendering/shaders.py` |
| Change gravitational lensing / BH shadow | `engine/glsl/common/refraction.glsl`, `engine/glsl/celestial/sphere.*`, `engine/glsl/atmosphere/atmo.*`, `engine/glsl/compute/culling.comp`, `engine/app.py` (lens uniform setup), `engine/physics/physics_core.py` (`load_system_from_data` BH radius) | — |
| Validate lensing math on GPU | `scripts/test_lensing_math.py` | `main` |
| Change bloom/tonemap/Accumulation shaders | `engine/glsl/post/` | GLSL files loaded via `engine/rendering/post_shaders.py` |
| Change mesh/ring geometry, frustum culling | `engine/rendering/render_utils.py` | — |
| Change planetshine & moon ringshine CPU precompute | `engine/rendering/planetshine.py` | `compute_planetshine_numba` |
| Change procedural ring generation / presets | `engine/rendering/ring_generator.py` | `generate_procedural_ring_profile`, `RING_PRESETS` |
| Change ring texture baking & application | `engine/rendering/texture_baker.py` | `apply_procedural_ring_to_body`, `bake_and_export_ring_textures` |
| Change ringshine irradiance map / shadow CDF | `engine/glsl/post/ringshine_map.frag`, `engine/app.py` (`build_ringshine_lut`) | `build_ringshine_lut` |
| Validate ringshine accuracy vs Monte Carlo | `scripts/ringshine_benchmark.py` | `main` |
| Change GLFW input callbacks & settings persistence | `engine/core/input_handler.py`, `engine/app.py` | `InputHandlerMixin`, `App` callbacks & movement mode |
| Change camera, UI, main render loop | `engine/app.py` | `class App`, `run()` |
| Change eclipse LUT build | `engine/app.py` | `build_eclipse_lut` |
| Change atmosphere LUT build | `engine/app.py` | `build_atmo_lut` |
| Change cloud texture generation | `engine/app.py` | `_prepare_cloud_image` |
| Change Body Inspector | `engine/app.py` | inspector block |
| Change system-switch / ephemeris modal | `engine/app.py` | `_trigger_ephem_switch/exit`, `_render_ephem_setup_modal` |
| Fetch real ephemerides offline | `scripts/fetch_horizons.py` | `main` |
| Fetch/rebuild GAIA star catalog | `scripts/fetch_gaia.py` | `main` |
| Change starfield rendering / catalog packing | `engine/rendering/star_catalog.py`, `engine/glsl/celestial/starfield.*` | `StarCatalog`, `_pack_kernel` |
| Validate GAIA starfield | `scripts/test_gaia.py` | `main` |
| Validate Mode 3 terminator (star size + refraction) | `scripts/test_skyview_terminator.py` | `main` |
| Validate Mode 3 baked eclipses + multi-star | `scripts/test_skyview_eclipse.py` | `main` |
| Validate Mode 3 bounded shadow subtraction (Blackrack) | `scripts/test_atmo_blackrack.py` | `main` |
| Validate Mode 3 polar winter for ringed bodies | `scripts/test_skyview_polar_winter.py` | `main` |
| Calibrate a texture's real-world albedo | `scripts/calibrate_moon_albedo.py` | `calibrate` |

---

## 7. Gotchas & Conventions

- **Coordinate frame swap:** physics uses (x,y,z) but `shared_state["pos"]` is remapped to
  render frame `(x, z, -y)` and velocity `(vx, vz, -vy)` — see `App.run` ~L1066. Keep this
  consistent when adding new buffers.
- **Numba caching:** `@njit(cache=True)` writes `.nbc`/`.nbi` files into `engine/__pycache__/`.
  Stale caches after editing numba functions are normal; they auto-rebuild. Deleting the cache
  forces recompilation.
- **Locking:** all `shared_state` reads/writes from the render thread must hold
  `shared_state["lock"]` to avoid tearing with the physics thread.
- **Comparison system** (`*_cmp`) mirrors the main system for side-by-side rendering; many code
  paths are duplicated — edit both branches together.
- **Shader uniforms are guarded** with `if 'u_name' in prog:` checks — add new uniforms there.
- **GLSL edits** must stay in sync with the CPU instance-buffer pack and any compute shader
  struct layout.
- **Export behavior** differs by system: custom systems save cosmetics directly to their
  `system.json`; Solar System single-export writes `exports/`; `\`+Export dumps all to master JSON.
- **Settings persistence:** `graphics_settings.json` via `App.load_settings/save_settings`;
  `ephemeris_settings.json` via `SpiceManager`; `imgui.ini` auto-saved by ImGui.

---

*Last updated against the source tree at the time of writing. Line numbers are approximate and
will drift — search by symbol name for precision.*
