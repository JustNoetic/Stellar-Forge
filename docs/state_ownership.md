# State ownership and update contracts

| Data | Owner | Readers | Mutation path |
| --- | --- | --- | --- |
| Physical properties, initial orbit elements, identity, propagation hierarchy | `Simulation` owns a shared immutable `SystemDefinition` | Snapshot consumers | Validated `replace_system`, `ApplyBodyPhysicalProperties`, `AddBody`, or `DeleteBody` |
| Current reference parents, distance-sorted children, orbit reference topology | `Simulation` owns shared immutable snapshot hierarchy data | Outliner, inspector, camera, orbit curves | 15 Hz refresh after successful steps; immediate load/edit/model changes and pause flush |
| Ecliptic positions and velocities | `Simulation` inside `SystemSnapshot` | Camera, inspector, renderer | `advance(Seconds)` |
| Simulation time, pause, warp, direction | `SimulationClock`, private to `Simulation` | App and HUD through `clock()` | `advance`, `SetPaused`, `SetTimeWarp`, `SetTimeDirection`, `ReverseTimeDirection` |
| Render positions, velocities, pole directions, spin | Prepared snapshot data owned by `Simulation` | Camera and renderers | Prepared after a successful load, edit, or simulation step |
| Exposure, HDR, Gaia visibility, star size and brightness | App-owned `RenderSettingsStore` | Renderers receive `const RenderSettings&` | Validated atomic `apply(RenderSettings)` |
| Navigation pose, tracking, input and motion history | `Camera` | App and renderers | Camera methods and navigation input |
| Panel visibility, widths and scale | App-owned `WorkspaceSettings` | `UiSystem` | UI configuration edits |
| Inspector selection, draft values, validation messages | App-owned `WorkspaceState` | UI | Selection events, draft edits, Apply/Revert |
| GPU allocations, pipelines, per-frame upload buffers | Owning renderer/RHI subsystem | Responsible rendering passes | Explicit subsystem lifecycle; independent of simulation definitions |

## Definitions, state and snapshots

`SystemManager` parses JSON into `LoadedSystem`: definitions and initial motion are separate. `Simulation::replace_system` validates and prepares a candidate before replacing the active system. Names must be unique, all parents must exist, the hierarchy must be acyclic and ordered parent first, and physical/motion values must be valid. A rejected replacement preserves the current system.

`SystemSnapshot::body(index)` returns a `BodyView` containing four const references: `definition`, `motion`, `render`, and `revisions`. Renderers, cameras and UI cannot mutate those values. Initial orbital elements belong to the definition; current positions and velocities belong to motion. Editing mass does not reset the trajectory to its initial orbital elements.

Normal interactive playback runs simulation on a worker thread; `--no-simulation-thread` selects the inline fallback. `Simulation::snapshot()` returns a live const reference without copying arrays or metadata. Its views must not be retained across a load, edit, or step. Explicitly copying a snapshot owns independent motion/render/revision arrays and shares the immutable definitions. Such a captured snapshot remains unchanged by later edits or steps. Copying costs O(body count); take the copy on the simulation thread before handing it to another thread. This API does not make concurrent reads of the live snapshot safe.

`SimulationRunner` runs a worker by default (`--simulation-thread` remains an explicit enable flag). Its public API has one caller, the main thread. Once started, only the worker accesses `Simulation`. Three reusable publication buffers hold snapshot arrays, clock, status and timing data; immutable definitions and hierarchy data remain shared. Copying happens outside the mailbox mutex on the worker. Publication and consumption swap buffers under the mutex; the main thread polls with `try_lock` at frame boundaries, keeping its front snapshot stable throughout rendering. Normal playback never waits for a physics step. Pending elapsed seconds are coalesced into a single backlog entry, without dropping elapsed time; this can change adaptive step boundaries compared with per-frame integration. Control commands wait for earlier steps, execute on the worker and publish their result before returning. These barriers preserve existing edit validation and UI behavior, but a command can still wait for an expensive in-flight step. Shutdown drains pending steps and joins the worker. Numerical integration and its existing substep limit are unchanged.

The analytical propagator reuses its previous-motion scratch vector. It does not copy body definitions, strings or JSON on each tick. JSON and definition copying occurs on loading, saving, or explicit physical edits.

`Simulation` also owns the cached `JacobiPropagator`. Snapshots share an immutable
`JacobiHierarchy` containing sibling order and subsystem/inner masses. Sibling
order is established from the loaded state and remains fixed while moving bodies
cross in radius. Both analytical propagation and orbit extraction use that same
hierarchy and complete descendant centers. A mass edit publishes updated hierarchy
masses and reseeds the propagator at the current epoch without moving any body;
captured snapshots retain their previous hierarchy. `motions()` exposes only a
const span with the same lifetime rules as `body()` views.

`reference_hierarchy()` separately publishes cached distance-sorted children and
current reference parents. IAS15 mode classifies parents using Hill regions and
hysteresis; other models retain declared parents. `orbit_hierarchy()` supplies
coordinates for these references without reseeding analytical propagation.
Reference parents can follow their children in array order. Body indices remain
stable, and captured snapshots retain old reference data. Include
`hierarchy_revision()` in hierarchy-dependent cache keys, since pausing can flush
a pending reference update without advancing `state_revision()`. See
`docs/dynamic_hierarchy.md` for scheduling and classification rules.

Motion selection is explicit in `SystemDefinition::motion_model`. SPICE playback queries geometric kernel states at the stored UTC epoch plus elapsed simulation seconds; the Keplerian propagator remains the model for ordinary presets. `Simulation` owns the shared CSPICE source, validates initial availability and commits all queried states atomically. A coverage failure preserves motion, prepared render state, clock and revisions. The app pauses on a rejected tick. See `docs/orbit_lines.md` for runtime setup and orbit-line approximation boundaries.

## Typed boundaries

`units::Seconds`, `Years`, `Kilometres`, `AstronomicalUnits`, and `SolarMasses` are explicit wrappers around a double. Convert with `to_years`, `to_seconds`, `to_au`, and `to_km`. `Simulation::advance` and `Camera::update` accept seconds; snapshots expose years; physical edit commands accept kilometres and solar masses.

`EclipticPosition`, `RenderPosition`, `EclipticVelocity`, and `RenderVelocity` are distinct types. Convert frames through the typed `ecliptic_to_render`/`render_to_ecliptic` overloads. `camera_relative(RenderPosition, RenderPosition)` subtracts in double precision before GPU packing. There are no implicit conversions between units or frames. `.value` is the explicit boundary into numerical kernels, GLM calculations, widgets, or GPU structs.

Serialized definitions and existing numerical kernels retain unit-suffixed scalar names (`mass_msun`, `radius_rsun`, `semi_major_au`, etc.). Packed GPU structures retain their existing GLSL ABI. Do not remove unit suffixes when a raw scalar is necessary.

## Atomic edits and revisions

The inspector edits a `BodyEditDraft`. Changes have no effect until Apply. The command identifies the system generation, body index, and expected body definition revision. A stale draft is rejected; Revert reloads the current committed values. Switching system or selection reloads the draft. Invalid values reject the entire edit. Changes are applied after the UI has finished using its body views, before scene rendering.

`BodyDefinition::radius_au()` derives the rendering radius from the committed physical radius, including the black-hole display-radius convention. It is not an independently editable duplicate. Black-hole physical radius is derived from mass and its radius widget is read-only.

| Change | Definition revision | Dynamics revision | Geometry revision | Lighting revision |
| --- | --- | --- | --- | --- |
| Mass | Increment | Increment | Unchanged, unless derived radius changes | Unchanged, unless derived radius changes |
| Radius | Increment | Increment when zonal harmonics depend on radius | Increment | Increment |
| Oblateness | Increment | Increment | Increment | Increment |
| J2/J4 | Increment | Increment | Unchanged | Unchanged |
| Identical or rejected edit | Unchanged | Unchanged | Unchanged | Unchanged |

`SystemSnapshot::definition_revision()` changes once per accepted definition edit. `state_revision()` changes on a load or successful nonzero simulation step. A new load increments `generation()` and resets each body's revision counters. Cache keys must include generation and body identity as well as the relevant revision. Pause, rejected steps, and unchanged settings do not invalidate prepared state.

These domain revision keys establish contracts for future terrain, atmosphere, and illumination caches; those caches are not implemented by this change. Keys must include every dependency: an orbital result also depends on its parent's dynamics revision, while a shadow result depends on the geometry of relevant occluders. Include `state_revision()` for results that depend on moving bodies and the relevant camera identity for view-dependent results. A new editable property must define its validation, preparation, and dependent revision domains in the owning subsystem before being exposed to the UI. Add coverage for unrelated domains remaining unchanged.

`RenderSettingsStore::apply` validates the complete settings candidate. On success it increments its revision only if values changed. Renderers still upload camera-dependent data per frame; a settings revision is not a substitute for camera or simulation dependencies.

## Persistence and checks

Physical edits affect the current session. The existing `SystemManager::save_system` API serializes the committed definition fields, preserving unknown JSON fields and initial state-vector data. This change does not add automatic disk writes or a Save button.

Build and run the CPU contract tests with:

```text
cmake --build build --target StellarStateTests
ctest --test-dir build --output-on-failure
```

The tests cover typed boundaries, coordinate transforms, pause/warp behavior, snapshot isolation, atomic edits, stale drafts, dependency revisions, persistence, hierarchy validation, and loading/stepping the available presets. `--self-test` remains a separate GPU startup smoke test; the Vulkan validation/synchronization concerns from the foundation review are separate work.
