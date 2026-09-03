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
│   ├── main.py                 # Entry point. faulthandler + try/except crash logger.
│   ├── app.py                  # ★ MAIN: App class, GLFW window, render loop, ImGui UI
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
│   │   ├── texture_baker.py    # Procedural ring HSBA baking & texture exporter
│   │   ├── shader_loader.py    # GLSL shader loader with in-memory caching
│   │   ├── shaders.py          # Shader re-exports & uniform bindings
│   │   └── post_shaders.py     # Post-processing shader re-exports (Bloom, HDR)
│   ├── ephemeris/              # Astronomical data & JPL Horizons/SPICE integration
│   │   ├── system_manager.py   # System I/O, SystemSnapshot, derive_star_properties
│   │   └── spice_manager.py    # SpiceManager: kernel download/load, SPICE state queries
│   └── glsl/                   # Dedicated GLSL shader source files
│       ├── compute/            # Frustum culling & orbit compute shaders (.comp)
│       ├── celestial/          # Sphere, orbit, ring, HZ shaders (.vert, .frag)
│       ├── atmosphere/         # Raymarching & LUT generation shaders (.vert, .frag)
│       └── post/               # Bloom, composite, accumulation & ringshine shaders (.vert, .frag)
├── data/
│   ├── system.json             # Active default system
│   ├── systems/<Name>/{meta.json,system.json}   # System presets (Solar System, Achernar, Ephemeris Mode)
│   ├── kernels/                # Downloaded SPICE kernels (.bsp/.tpc/.tls)
│   ├── ephemeris_settings.json # SPICE playback dates & active kernel list
│   ├── graphics_settings.json  # Persistent graphics/quality settings
│   └── horizons_cache.json     # Cache for JPL Horizons REST queries
├── scripts/
│   ├── fetch_horizons.py       # Query JPL Horizons REST → update system JSON state vectors
│   ├── accuracy_test.py        # 1-yr integration benchmark vs JPL Horizons ground truth (RTN errors)
│   ├── perf_test.py            # PerfTracker: patches functions to measure startup/frame timing
│   └── test_spice.py           # Quick SPICE kernel loader validation
├── textures/                   # Planet/ring textures (loaded by app.py at startup)
├── exports/                    # Exported cosmetic JSON per body
├── run.bat                     # Windows one-click venv + deps + launch
├── imgui.ini                   # Saved ImGui layout
└── README.md / PROJECT_MAP.md  # Docs
```

---

## 3. Module-by-Module Reference

For each file: responsibilities, key symbols (with approximate line numbers), and
"edit here when you want to…".

### 3.1 `engine/main.py` (15 lines)
- `if __name__ == "__main__"`: enables `faulthandler`, wraps `App().run()` in try/except,
  dumps `main_error.txt`.
- **Edit when:** changing startup/error handling/CLI args.

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
- `physics_loop(sim, num_bodies, shared_state, time_ctrl, running)` (~L986) — **physics thread main loop**:
    handles system-switch requests, snapshot save/restore, IAS15 vs Keplerian mode switching,
    hierarchy rebuild, step timing, writes `shared_state` under `shared_state["lock"]`.
- `load_system_from_data(bodies_data_raw)` (~L1710) — builds `Simulation`, attaches custom forces,
  derives visual data / atmo bodies / ring bodies / oblate list → returns **bundle dict** with keys:
    `sim, num_bodies, bodies_data, name_to_idx, visual_data, atmo_bodies, ring_bodies,
     oblate_physics_list, has_j2, has_gr, phys_star_idx, star_idx`.

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
  - `__init__()` (~L120) — loads `ephemeris_settings.json`, ensures `data/kernels/`.
  - `cancel_download()`, `_cleanup_partial_downloads()`.
  - `_load_settings()`, `save_settings()`, `_ensure_dirs()`.
  - `check_missing_kernels()` (~L185) — compares settings vs on-disk.
  - `download_kernels_async(on_complete)` (~L196) — threaded download with progress hook.
  - `load_kernels(force=False)` (~L308) — `spiceypy.furnsh`.
  - `_discover_bodies()` (~L332) — enumerate SPICE IDs.
  - `datetime_to_et(dt)` (~L374), `get_body_state(body_id, et)` (~L386),
    `get_all_states(et)` (~L422).
  - `get_body_mapping(bodies_data, test_et)` (~L446) — match JSON bodies → SPICE IDs.
  - `populate_states_fast(et, mapping, pos_out, vel_out, valid_out)` (~L473) — batch state query.
  - `_get_body_properties(body_id, fallback_mass_kg, fallback_radius_km)` (~L513).
  - `build_ephemeris_system(et, template_bodies)` (~L559) — assembles bodies_data for Ephemeris Mode.

**Edit when:** SPICE kernel handling, ephemeris playback, body mapping, Ephemeris Mode system build.

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
  - `sphere_vertex_shader` / `sphere_fragment_shader` (`glsl/celestial/sphere.*`) — PBR planet/star spheres (with ray refraction `compute_refraction_angle` & vertex bounding expansion).
  - `orbit_compute_shader` (`glsl/compute/orbit.comp`), `orbit_*` (`glsl/celestial/orbit.*`).
  - `ephem_orbit_*` (`glsl/celestial/ephem_orbit.*`).
  - `ring_*` (`glsl/celestial/ring.*`), `hz_*` (`glsl/celestial/hz.*`).
  - `atmo_*` (`glsl/atmosphere/atmo.*`), `atmo_lut_*`, `multi_scatter_lut_*` — Volumetric raymarching atmosphere shaders (with primary ray refraction & vertex bounding expansion).

**Edit when:** editing GLSL shader logic inside `engine/glsl/` or uniform bindings in `shaders.py`.

### 3.11 `engine/rendering/post_shaders.py` — Post-Processing GLSL
- Re-exports post-processing shaders loaded from `engine/glsl/post/`:
  - `bloom_downsample.frag`, `bloom_upsample.frag` — HDR bloom pyramid.
  - `composite.frag` — tonemap (ACES) + exposure + bloom composite.
  - `accum.frag` — jitter accumulation resolve for screenshots and scene.

**Edit when:** bloom, tonemapping, exposure, accumulation.

### 3.12 `engine/rendering/render_utils.py` (343 lines)
- `hex_to_rgb`, `sample_gradient` — color helpers.
- `format_time_speed(multiplier)` (~L26), `format_sim_time(t_years)` (~L47),
  `sim_time_from_date(y,m,d,h,mn)` (~L98) — UI time strings.
- `compute_ring_culling(...)` (~L121, `@njit`) — ring visibility.
- `extract_frustum_planes(vp)` (~L174, `@njit`) — frustum planes from VP matrix.
- `create_icosphere_mesh(subdivisions=4)` (~L189) — UV/ico sphere geometry.
- `generate_ring_shadow_grad(...)` (~L242), `generate_ring_geometry(pole_render, min_r, max_r)` (~L259).
- `rebuild_ring_render_group(bi, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)` (~L301).

**Edit when:** mesh generation, ring geometry, frustum culling math, time formatting.

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
- `compute_planetshine_numba(pos, radii, colors, is_star, star_positions, star_colors, star_lums, star_radii, hdr_enabled)` (~L295, `@njit`) — CPU planetshine precompute.
- **Active Refraction Uniform Setup** (~L3837) — passes distance-agnostic refraction parameters (`u_refract_center`, `u_refract_radius`, `u_refract_max_bend`, `u_refract_scale_height`, `u_refract_pole`, `u_refract_oblateness`) dynamically across active shaders (`prog_spheres`, `prog_rings`, `prog_atmo`, `prog_orbit`, etc.).

**`class App`** (~L487) — the engine:
- `__init__` (~L488) — camera dict (target, distance, fov, yaw/pitch/roll, exposure, hdr, tracking_idx,
  inspected_idx, add_mode, modals…), loads `graphics_settings.json` via `load_settings()`.
- `load_settings()` (~L590), `save_settings()` (~L603).
- **GLFW callbacks:** `scroll_callback` (~L634, zoom/FOV/tracking), `mouse_button_callback` (~L672,
  pick + drag), `cursor_pos_callback` (~L695, orbit/pan), `char_callback` (~L738),
  `key_callback` (~L741, speed/exposure/roll/pause), `resize_callback` (~L761).
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
      - `render_atmosphere_pass(clip_mode)` — two-pass with dual-source blending (behind/in front of rings).
      - Rings render, Habitable Zones, second atmosphere pass.
      - Dynamic Cloud Layer pass (~L4733) — rendered over the atmosphere via radiative transfer blending so the volumetric atmosphere renders behind semi-transparent clouds.
      - ImGui UI: top bar (speed/exposure/graphics modal), system menu popup (~L3795–4030),
        mode switch (IAS15 / Keplerian / Ephemeris) radio (~L4219–4320),
        `_render_ephem_setup_modal()` (~L4097), `_trigger_ephem_switch()` (~L4146, spawns spice
        download thread + `_on_spice_ready` ~L4157), `_trigger_ephem_exit()` (~L4184).
        Body Inspector panel (~L4544–5832) — orbital/physical/thermal/atmosphere/stellar tabs,
        cosmetics, ring editing (`rebuild_ring_render_group`), export.
        Add-body mode (~L5832), Create System modal (~L5984).
      - Accumulation resolve (~L6192), post-accum orbits/HZ (~L6243), bloom + composite (~L6330).
  11. Teardown: stop physics thread, release GL objects, save settings, GLFW destroy.

**Edit when:** UI, rendering pipeline, camera, input, modal dialogs, system switching,
inspector, LUT building, instance buffer layout, post-processing.

### 3.14 `scripts/`
- `fetch_horizons.py` — `get_parent_center`, `_load_cache`/`_save_cache`, `query_horizons(body_id, center, start_time, stop_time)`, `parse_state_vector(response_text)`, `main()`. Writes `data/horizons_cache.json` + updates system JSON.
- `accuracy_test.py` — `get_parent_center(body_name, parent_name)`, `main()`. Runs 1-yr forward integration vs JPL Horizons, prints RTN km error table.
- `perf_test.py` — `class PerfTracker` (~L40), `patch_function(module, name, tracker, label)` (~L113), `main()`. Activated via `STELLAR_FORGE_PERF=1`; `--gpu` adds per-pass GL timer queries (env `STELLAR_FORGE_GPU_PERF=1`, instrumented passes via `_perf_gpu_begin/_end/_flush` in `app.py`) and `--target-body NAME` parks the camera on a body (default `Saturn` when `--gpu`).
- `test_spice.py` — minimal SPICE loader sanity check.

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
| Tune physical constants | `engine/core/constants.py` | — |
| Change Newtonian/GR/J2/J4 forces | `engine/physics/physics_core.py` | `compute_custom_forces`, `compute_all_accelerations` |
| Change IAS15 integrator | `engine/physics/physics_core.py` | `ias15_step_numba` + helpers |
| Change analytical Keplerian mode | `engine/physics/kepler_analytical.py` | `propagate_keplerian_system_numba`, `extract_all_kepler_elements` |
| Change hierarchy/parent-tree logic | `engine/physics/physics_core.py` | `_update_hierarchy_core`, `build_tree_order` |
| Change physics thread / system switching | `engine/physics/physics_core.py` | `physics_loop` |
| Change how systems load from JSON | `engine/physics/physics_core.py` | `load_system_from_data` |
| Change system file I/O / presets | `engine/ephemeris/system_manager.py` | `SystemManager`, `SystemSnapshot` |
| Change star classification/evolution | `engine/physics/star_calc.py`, `engine/ephemeris/system_manager.py` | `StarCalculator.forge`, `derive_star_properties` |
| Change SPICE / Ephemeris Mode | `engine/ephemeris/spice_manager.py` | `SpiceManager` |
| Change scattering physics / gas table | `engine/physics/atmosphere_physics.py` | `compute_atmosphere_properties`, `GAS_PROPERTIES` |
| Change orbital math / frame rotations | `engine/core/math_utils.py` | — |
| Change sphere/ring/atmo/orbit/HZ shaders | `engine/glsl/` (`celestial/`, `atmosphere/`, `compute/`) | GLSL files loaded via `engine/rendering/shaders.py` |
| Change bloom/tonemap/Accumulation shaders | `engine/glsl/post/` | GLSL files loaded via `engine/rendering/post_shaders.py` |
| Change mesh/ring geometry, frustum culling | `engine/rendering/render_utils.py` | — |
| Change planetshine CPU precompute | `engine/rendering/planetshine.py` | `compute_planetshine_numba` |
| Change ring texture baking | `engine/rendering/texture_baker.py` | `bake_and_export_ring_textures` |
| Change GLFW input callbacks & settings persistence | `engine/core/input_handler.py` | `InputHandlerMixin` |
| Change camera, UI, main render loop | `engine/app.py` | `class App`, `run()` |
| Change eclipse LUT build | `engine/app.py` | `build_eclipse_lut` |
| Change atmosphere LUT build | `engine/app.py` | `build_atmo_lut` |
| Change cloud texture generation | `engine/app.py` | `_prepare_cloud_image` |
| Change Body Inspector | `engine/app.py` | inspector block |
| Change system-switch / ephemeris modal | `engine/app.py` | `_trigger_ephem_switch/exit`, `_render_ephem_setup_modal` |
| Fetch real ephemerides offline | `scripts/fetch_horizons.py` | `main` |
| Validate physics accuracy | `scripts/accuracy_test.py` | `main` |
| Profile startup/frames | `scripts/perf_test.py` | `PerfTracker` (env `STELLAR_FORGE_PERF=1`) |

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
