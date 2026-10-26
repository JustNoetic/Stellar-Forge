# System comparator: implementation plan

Status: **Phases 0–4 implemented with passing contracts.** The optional size-comparator link to live systems is not implemented. Source behaviour: the original `D:\Software\Stellar Forge` comparison mode (`engine/app.py` `*_cmp` state, `engine/ui/menu_bar.py`, `outliner.py`, `inspector.py`). This is separate from the object size comparator (`size_comparator.md`), which arranges definitions at one physical scale without simulating them.

## 1. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Number of compared systems | **N, soft cap 3.** Each compared system has its own simulation worker; all systems share `MAX_SCENE_STARS = 16`. The original supported exactly one. |
| 2 | Cross-system lighting | **Directional switches.** Per compared system: *primary starlight → this system* and *this system's starlight → primary*. Both default off. |
| 3 | Editing compared bodies | **Read-only.** **Open for Editing** swaps the compared system into the primary slot; edits always go through the primary runner. |
| 4 | Rendering integration | **One composite snapshot.** Renderers and `Camera` keep consuming a single `const SystemSnapshot&`. |
| 5 | Motion model | **Keplerian for compared systems** by default (original "Force Keplerian"); SPICE remains primary-only. |
| 6 | Timeline rendering | **Disabled while comparisons are active** (phase 1). |

## 2. Original behaviour

- A second complete system (`shared_state_cmp`) on its own physics thread.
- Its clock copied the primary's pause, warp, direction, jump and timeline flags every frame.
- It rendered in the same world, offset by `comparison_offset_au` along +X (default 10 AU). Bodies, orbits, atmospheres, rings, clouds, refraction and lensing all used a second instance range `num_bodies + i`.
- Tracking, centering and inspection carried `*_is_cmp` flags. The outliner showed a second section; the inspector was read-only for compared bodies.
- Tools menu: Enable Comparison Mode, Force Keplerian Mode, Compare With, Offset (AU).

## 3. Architecture

```
SimulationRunner (primary) ──┐
SimulationRunner (slot 1) ───┼─► SystemComposer ──► composite SystemSnapshot ──► Camera, picking, all renderers
SimulationRunner (slot N) ───┘        │
                                      └─► CompositeMap: composite index ↔ (slot, local index), light group per body
UI edits, inspector drafts, time HUD ──► primary snapshot only (App::system())
```

- `App::system()` remains the primary snapshot. A new `App::scene()` returns the composite; with no comparisons it is the primary snapshot itself, not a copy.
- Primary bodies come first and keep indices `0..N-1`. Compared bodies follow in slot order.
- The composite applies each slot's offset to ecliptic motion **and** render positions, so barycentres, `extract_jacobi`, orbit curves and camera tracking agree without special cases. Offsets never reach a simulation.
- Merged definition, reference hierarchy and Jacobi hierarchies (index-shifted concatenation) are rebuilt only when a member's generation, definition or hierarchy revision changes. Motion, render state and revisions are copied each frame (O(total bodies)).
- Composite construction needs a builder with `SystemSnapshot` friend access in `ephemeris`.

## 4. Phases

### Phase 0: per-body cache identity

Renderer caches currently key on whole-snapshot identity:

| Site | Key |
|---|---|
| `atmosphere_renderer.cpp` (~526, 540, 659, 685, 1107) | `generation()`, `share_definition()` pointer, body index |
| `ring_renderer.cpp` (~312, 321) | `generation()`, `definition_revision()` |
| `orbit_renderer.cpp` (~59) | `generation()`, `state_revision()`, `definition_revision()` |
| `terrain_renderer.cpp` (~148, 220, 227, 261, 285, 317) | `generation()` in selection and `GeometryKey` |

A composite would change these on every membership change and rebuild primary atmosphere LUTs, eclipse tables and terrain geometry. `generation()` is also numbered per `Simulation`, so two runners collide on the same values.

**Implemented.** `BodyRevisions::identity` is a process-unique, nonzero identity assigned on load and `AddBody`, and it survives edits, steps, motion-model changes, copies and reindexing. The renderers now key on it:

| Renderer | New key |
|---|---|
| Atmosphere tables | Slot match and eviction by identity; optics by identity plus optical parameters; shadow history by identity plus definition/geometry/lighting revisions. Tables survive reindexing and edits to other bodies. |
| Ring profiles | Ordered `(index, identity, definition revision)` list of ring hosts. Edits to other bodies no longer rebuild; reindexed hosts rebuild once. |
| Orbit curves | Ordered identity list plus state, definition and hierarchy revisions. |
| Terrain | `GeometryKey::identity` replaces `generation`; selections whose index no longer holds the same identity are discarded. |

Contracts: `StellarStateTests` (`body_identity_contracts`), `StellarRenderingTests --atmosphere-cache-only` (`atmosphere_identity_contracts`), `--rings-only` (`ring_system_cache_contracts`), `StellarOrbitRenderingTests` (independent simulations with equal revision numbers).

Merged hierarchies with several roots: `extract_jacobi` merges each root's subtree independently and `compute_orbit_curves` builds curves per parent, so concatenated hierarchies are valid by construction. A composite contract in Phase 2 should confirm it.

Name-keyed lookups for composite snapshots (resolved in Phase 2): ring texture paths and their cache key use the body's source system; terrain height streams and surface textures bind by body name, so duplicate names across systems share assets, as they would when each system is primary.

### Phase 1: comparison runners and clock sync

- `comparison::ComparisonSet` owns 0..N `SimulationRunner` slots, each with system name, offset and light switches.
- Clock commands (`SetPaused`, `SetTimeWarp`, `SetTimeDirection`, `ReverseTimeDirection`, `JumpToTime`) go to every runner; each frame's elapsed seconds go to every `advance_async`.
- Independent adaptive steps may publish slightly different times; the HUD warns above a tolerance.
- **Keplerian for compared systems** setting (default on) issues `SetMotionModel` on load.
- Timeline rendering is disabled while any slot exists.

**Implemented.** `comparison::ComparisonSet` (`src/comparison/system_comparison.hpp/.cpp`) owns `ComparisonSlot`s, each with a system name, an offset (default `10 AU × slot` along +X), both light switches (off) and its own `SimulationRunner`, threaded like the primary. App integration is in `src/app_system_comparison.cpp`.

- **Clock flags** are not intercepted at each UI site. Every frame `synchronize` copies the primary's warp, direction and pause to any slot that differs, then each slot receives the same `advance_async` elapsed seconds as the primary. A command waits for in-flight steps, so steps are never coalesced across a warp change and slots match the primary up to rounding.
- **Time** is synchronised as elapsed simulation time (`SimulationClock::time`), not UTC, because systems have different epochs. A slot jumps to the primary's time when added, when the primary jumps (Jump to Date), and when the primary's generation changes (system switch, ephemeris mode, BSP playback, N-body export). The primary's in-flight step is awaited first so the slot does not lag one frame.
- **Keplerian override** (`set_force_keplerian`, default on) issues `SetMotionModel{Keplerian}` on add. With it off, an N-body slot can join only at the primary's current time, since N-body cannot jump; the add is rejected otherwise.
- **Failures:** every error is prefixed with the slot's system name. A failed slot step pauses the primary, and the next synchronisation pauses every slot.
- **Limits:** `App::add_system_comparison` rejects Ephemeris Mode, more than `COMPARISON_SOFT_CAP` (3) slots, and an active timeline render or scrub. The Render Timeline button is disabled while slots exist.
- **Time HUD:** shows "Comparing N systems", or a warning when `max_time_skew_years` exceeds `skew_tolerance_years` (the larger of 1 s and four frames of warped time).
- The slots are not yet rendered; the composite scene is Phase 2.

Contracts: `StellarSystemComparisonTests` (slot alignment, Keplerian override, rejected N-body joins, removal, inline lockstep with zero skew and threaded lockstep within tolerance across warp, direction and pause changes, jumps, error naming); `--self-test` compares the Solar System with a second copy of itself, checks clock flags and skew every frame, and removes the slot.

### Phase 2: composite scene, placement and UI

- `comparison::SystemComposer` and `CompositeMap` (section 3).
- Placement: per-slot offset (AU, ecliptic) with **Auto Spacing** from each system's outermost orbit.
- Outliner: one section per system. Inspector: read-only view from the source snapshot and local index; `BodyEditDraft` never sees composite indices.
- Camera tracking/centering use composite indices; on slot change or swap they remap by (slot, body name).
- **Open for Editing** swaps runners between the primary and the slot (no reload, state and caches preserved), then remaps camera and selection.
- Menu: View → System Comparator (add/remove systems, offsets, light switches). Preferences: Keplerian default and auto spacing in `UserSettings`; slots and offsets stay transient.

**Implemented.**

- **Composite** (`src/comparison/system_composer.hpp/.cpp`): `SystemComposer::compose` takes the primary and each slot with its offset and returns one `SystemSnapshot` (friend access). Primary bodies keep indices `0..N-1`. Offsets are added to ecliptic and render positions, never velocities. Definitions are rebuilt only when a member's definition pointer changes, and hierarchies (Jacobi, orbit Jacobi, reference) only when their pointers change; both concatenate with shifted indices. Motions, render states and revisions are copied every compose. The composite's state revision moves when any member's state revision or an offset changes; its generation (`layout_revision`) moves when the composite identity sequence changes. `CompositeMap` maps composite indices to `(slot, local)`, slot 0 being the primary, and `light_group` returns the slot for Phase 3.
- **Per-body source system:** `SystemDefinition::source_systems` and `body_system(i)` record each composite body's system; ring textures now load from `body_system(i)`. Terrain height streams and surface textures bind by body name across all systems, so a compared body gets the same assets it would get as the primary system.
- **App:** `App::scene()` is the composite (or `system()` with no slots) and feeds the camera, all renderers and the tracking pill. UI edits, drafts and the time HUD keep using `system()`. The scene is composed after advancing, after the UI and after every comparator change.
- **Selection:** `camera_.tracking_idx`, `camera_.centered_idx` and `ui_state_.inspected_body` are composite indices. Each compose remembers the identity at each index; if an index is unchanged but now holds a different body, it moves to the index holding the remembered identity, or falls back (star, or nothing) when that body is gone. This covers primary additions and deletions, slot removal and swaps without per-site remapping, and leaves deliberate index changes alone.
- **Placement:** default `10 AU × slot` along +X. **Auto spacing** (default on, `WorkspaceSettings::comparison_auto_spacing`) places systems along +X separated by `1.25 × (extent A + extent B)`, where `system_extent_au` is the larger of each body's chained apoapsis and its current distance from the star. It reruns on add, remove, swap and primary reloads, not every frame. Manual offsets are editable when it is off.
- **UI:** View → System Comparator adds systems (soft cap 3), toggles Keplerian motion for new comparisons (`comparison_force_keplerian`) and auto spacing, and per slot edits offsets, opens for editing and removes. The outliner lists each compared system below the primary in its own read-only section. Selecting a compared body opens a compact read-only inspector (physical values, orbit, Track/Center/Go To, Open for Editing); edit mode is cleared and edit commands never see composite indices. The starlight switches and overlay mode are described under Phase 3.
- **Open for Editing:** `SimulationRunner::swap_simulation` stops both workers, swaps the `Simulation` objects and restarts them, so state, identities and renderer caches survive. The UI state is reset as on a system switch (drafts carry per-simulation generations), and the inspected and tracked bodies follow by identity. Each system keeps its motion model.
- **Lighting:** Phase 3 adds per-slot light groups; with every switch off each system is lit only by its own stars.

Contracts: `StellarSystemComparisonTests` (`composite_contracts`: layout, unchanged primary indices and identities, offsets in both frames, shifted definition/reference/Jacobi parents, one root per system, `extract_jacobi` barycentres follow the offset, rebuild and revision counters, primary additions shifting compared bodies, removal; `swap_and_spacing`: swaps keep workers, identities, names, motion models and time; auto spacing clears both extents); `StellarRenderingTests --system-comparison-only` (adding a compared system builds only its own optical tables, a stable composite rebuilds nothing, removing it keeps the primary tables, composite render); `--self-test` (track a compared body, Open for Editing keeps tracking and inspection on it, remove).

### Phase 3: directional light groups and overlay

- Each composite body carries a light group (its slot). A group mask `lit_by[receiver]` lists the groups whose light reaches it. The two switches per slot set the primary ↔ slot bits; compared ↔ compared pairs default off.
- The mask governs all light leaving a group: direct starlight, planetshine and ringshine. Occluders of any group can shadow light that is allowed to arrive.
- CPU filters: `celestial_renderer.cpp` (~274 star packing, ~324 occluders), `atmosphere_renderer.cpp` (~76, 84, 624), `planetshine.hpp`, `ring_renderer.cpp` (~553, 627), `eclipse_refraction_renderer.cpp` (~99, 119), `eclipse_candidates.hpp`. GPU: instance data carries the group and shader star loops test the mask.
- **Overlay mode:** offset 0 with each system centred on its barycentre. Requires both switches off to avoid double lighting.

**Implemented.**

- **Rule:** `SystemDefinition::light_groups` (per body) and `lit_by` (per group, a bitmask of groups) define `illuminates(source, receiver)`: true within a group, otherwise only when the receiver's group lists the source's group. Plain snapshots have empty vectors and light everything. `SystemSnapshot::illuminates` forwards to it.
- **Composer:** `ComposedMember::lit_by` is the member's mask; group = slot. A changed mask rebuilds the composite definition and bumps that slot's lighting epoch, which is added to its bodies' `BodyRevisions::lighting` (`epoch << 32`), so per-body lighting caches such as atmosphere shadow history reset for that slot only. A slot whose member changes also bumps its epoch.
- **Switches:** `ComparisonSlot::primary_lights_slot` and `slot_lights_primary` (default off) map to `ComparisonSet::slot_lit_by(i)` (bit 0) and `primary_lit_by()` (bit `i+1`). Compared systems never light each other. Open for Editing exchanges the two switches so each keeps its physical direction.
- **GPU:** `BodyInstanceGPU::surface_flags.y` holds the receiver's blocked-star mask over the scene star list (bit `s` set when star `s` may not light it, so zero-initialised instances keep full lighting); `star_lights_body` in `celestial_scene.glsl` gates the star loops in `sphere.frag` (direct light and surface ringshine, so also terrain), `point_celestial.vert`, `ring.frag` (by ring host) and `ring_planetshine*.glsl`.
- **CPU:** `prepare_planetshine` drops starlight a source may not receive and sources a receiver may not see. The atmosphere renderer keeps each star's slot but zeroes the irradiance of disallowed stars and skips their shadow casters, so per-star indices stay aligned with the scene star list. Ringshine tiles are not baked for disallowed (host, star) pairs, and the ring atmosphere field ignores disallowed stars.
- **Occluders** of any group still cast shadows, eclipse candidates and refraction, matching the rule that the mask governs light, not geometry.
- **Overlay mode** (`ComparisonSet::set_overlay`, transient): every member is placed at minus its own barycentre, offsets and auto spacing are ignored, and both directions are treated as off. The window disables the offset, spacing and switch controls while it is on.
- **UI:** each slot in View → System Comparator has "<primary> starlight reaches this system" and "This system's starlight reaches <primary>", plus a global Overlay systems toggle.

Contracts: `StellarSystemComparisonTests` (`light_switch_contracts`: defaults, one-direction masks, light groups per slot, compared↔compared isolation, plain snapshots, overlay placement and switches, swap keeps directions); `StellarRenderingTests --system-comparison-only` (`system_light_group_contracts`: a half-lit planet in each system; switches off render the primary exactly as alone and exchange no planetshine; each switch lights only the other system's night side; only the changed slot's lighting revisions move. Removing the shader mask fails it); `--self-test` (switch on, swap, overlay on and off).

### Phase 4: comparison panel

Side-by-side table of matching bodies or elements (period, a, e, radius, mass, equilibrium temperature). Optionally let the size comparator use compared live systems as sources.

**Implemented.**

- **Metrics** (`src/comparison/body_metrics.hpp/.cpp`): `measure_body` reads a body from its own system's snapshot. Semi-major axis, eccentricity and period are osculating two-body elements around the body's reference parent (`mu = G (M_parent + m)` in AU, years and solar masses; no period for unbound states). The equilibrium temperature is `278.6 K × (Σ L/d²)^¼` over that system's own stars, for zero albedo and full redistribution; stars report none. Mass and radius come from the definition.
- **Matching** (`match_bodies`): *Orbit order* gives a Star row (each system's main star) followed by Planet 1..N, the bodies orbiting a star ranked by semi-major axis (current distance when unbound); moons are not ranked. *Body name* gives one row per distinct name, in the editing system's order followed by names found only in compared systems.
- **UI:** View → Comparison Table (also from the System Comparator window). It has Match by and Metric selectors and one column per system, the editing system first. Each cell names the body (clicking it selects that body in the scene) and shows the metric with its ratio to the editing system's value (no ratio for eccentricity). Selecting a row opens a detail table of every metric for that match. Values are recomputed each frame while the window is open.
- **Not implemented:** feeding compared live systems into the size comparator. It still reads saved system definitions.

Contracts: `StellarSystemComparisonTests` (`metric_contracts`: 1 AU circular orbit gives a = 1 AU, e = 0, a 1 yr period and 278.6 K; stars have no orbit or temperature; moons orbit their reference parent; orbit-order rows rank planets and skip moons; name rows keep the editing system's order and append the rest); `--self-test` opens both windows and drives both match modes, a metric and a detail row.

## 5. Validation

- CPU (`StellarStateTests` or a new `StellarComparisonTests`): primary indices and cache identities unchanged when slots are added; clock commands reach all runners; offsets affect composite positions only; selection and tracking survive slot changes and swaps; compared bodies reject edit commands.
- GPU (`StellarRenderingTests --system-comparison-only`): adding a slot does not rebuild primary atmosphere tables; a lit-by-nothing overlay matches the single-system reference; each light switch affects only its direction.
- `--self-test`: add a comparison, track a compared body, change warp, swap for editing, remove it.
