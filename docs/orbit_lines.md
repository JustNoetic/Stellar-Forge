# Orbit lines and ephemeris playback

`ephemeris/orbit_curves.cpp` ports the original `physics_core.py` orbit extraction:
complete descendant barycenters, inner-sibling Jacobi masses, osculating conics,
dominant-companion reflex curves, Hill clipping and one dashed outer continuation.
The physical relative semilatus rectum controls Hill intersections and exit states;
the mass-scaled value controls drawing about the barycenter. The parent is propagated
to the exit time with the universal conic solver. The outer estimate stays in the
grandparent frame; an apparent gap from the frozen local arc is intentional.

`ephemeris/jacobi.cpp` supplies the shared Jacobi hierarchy for propagation and
curve extraction. At loading, siblings are ordered by distance to their physical
parent. This order is retained when instantaneous radii cross. Complete descendant
masses and centers are accumulated bottom-up. For each sibling in that order,
the relative state is its subsystem center minus the center of the parent plus
all preceding sibling subsystems. Every sibling mass participates, including
small moons and massless tracers; there is no rendering-only mass cutoff.

Keplerian motion propagates those cached relative states from a common epoch.
Placement runs top-down: it starts with the complete parent's subsystem center
and removes the children from outermost to innermost with their mass fractions.
This reconstructs the physical parent and every moon, including the parent's
reflex motion. Advancing each moon independently about the physical parent
would produce incompatible states and visibly changing fitted conics, especially
for Pluto/Charon. Mass edits reseed from current positions and velocities at the
current epoch, preserving sibling order and the instantaneous state. Ordinary
ticks do not re-extract seeds or accumulate propagation error frame by frame.
The drawn primary reflex curve retains the original dominant-companion
approximation; a primary with several massive moons has composite motion.

Curves use eccentric anomaly below `e = 0.999` and true anomaly above it. Circular
states have a stable basis without dividing by eccentricity. Complete ellipses
close exactly. Unbound root pairs share a finite draw domain, including their
reflex curve. A companion outside its parent's Hill sphere has no local or reflex
arc. Degenerate radial states do not produce an orbital curve.

`OrbitRenderer` sends compact double-precision conic parameters to a Vulkan
line-strip pipeline. `common/orbit_curve.glsl` evaluates each vertex and subtracts
the double-precision camera position before conversion to float. It retains the
original fade direction, dashed continuations and open-conic distance fading.
Orbit lines explicitly select one-pixel Bresenham rasterization through
`VK_KHR_line_rasterization` or its EXT equivalent, matching OpenGL's unsmoothed
connected pixel coverage for small, densely sampled curves. Device support is
queried and enabled by the shared RHI; unsupported devices use default rasterization
with a log message. The original RGB conditioning (`0.4` scale and `0.25` brightness
floor), minimum alpha (`0.3`) and Bright Behind fade are restored. Overall orbit
opacity defaults to `1.0` before the along-orbit fade.
Lines alpha-blend into HDR, test against opaque reverse-Z scene depth and do not
write scene depth. View > Orbit Lines and Orbit Opacity control visibility.
`RenderSettings::orbit_segments` defaults to 2048 and accepts 128–8192 segments.

Extraction caches include system generation, state revision and definition
revision. Camera culling and GPU packing run on every draw, so moving the camera
while paused restores previously culled curves. Allocations are per frame in flight.

These lines are instantaneous two-body/Jacobi approximations, including during
SPICE playback. They are not integrated future N-body trajectories or sampled
spacecraft mission polylines. Rotating Lagrange trajectory frames, adaptive optical
subdivision, orbit MSAA and transport through atmospheres remain migration work.

## Motion models

The rewrite previously loaded the Ephemeris Mode preset's stored state vectors
and propagated them through the same Kepler solver as every other preset.
It now has explicit `MotionModel::Keplerian` and `MotionModel::Spice` definitions.
The first hierarchy root's `motion_model` selects the model; the legacy preset
name `Ephemeris Mode` also selects SPICE. Unknown model names reject loading.
Serialization retains the model and epoch when saving with another name.

The ephemeris preset has explicit NAIF `spice_id` values and starts at its stored
UTC epoch, `2026-06-03`. CSPICE converts UTC with the configured leap-second kernel.
Every successful tick requests geometric `ECLIPJ2000` states relative to the solar
system barycenter at `epoch_ET + elapsed_seconds`. There is no light-time correction
or Kepler fallback. Positions convert from km to AU and velocities from km/s to
AU/year, then the simulation recenters the selected system as usual. Relative
states retain the SPICE values. Physical edits affect display and osculating
curve extraction; kernel-prescribed trajectories remain prescribed.

Rendered planets must use physical-center SPICE targets: Jupiter `599`, Saturn
`699`, Neptune `899`, and Pluto `999`. IDs `5`, `6`, `8`, and `9` are their complete
system barycenters. Substituting a system barycenter for its physical planet and
then accumulating its moons counts their reflex contribution twice. The original
Python manager also resolves those physical-center IDs when kernels provide them.
The rewrite's explicit preset IDs now match that behavior without a center switch
during playback.

`data/ephemeris.json` lists the runtime filename and kernel load order relative to
`data/kernels/`, using `paths::external_path`. The current Windows workspace has
the original project's `cspice.dll` and listed kernels copied locally. They remain
git-ignored runtime data, independent of the original project's paths. A fresh
checkout needs those files installed separately. On other platforms, provide a
matching CSPICE shared library and update the runtime filename in the manifest.
Kernel order follows SPICE precedence. The bridge owns one shared CSPICE kernel
pool and is used on the simulation thread.

Missing runtime files, IDs, kernels or required body states reject a system switch
and preserve the previous system. Spacecraft without coverage at the initial epoch
are omitted with a log message; descendants of an omitted parent reject loading.
For this preset, Artemis II has no coverage at the June epoch and is omitted.
A missing state during playback rolls back all bodies and the clock; the app
pauses and displays the error. The outliner and HUD identify the active motion
model, separately from the camera navigation mode.

## Checks

`orbit_contracts` compares against 14 frozen fixtures from the original Python
functions, covering circular, elliptic, nearly parabolic and hyperbolic conics,
bound/escaping/outside-Hill encounters, reflex intervals and nested subsystems.
`tests/fixtures/orbit_reference.json` contains original packed records in the
original coordinate frame before recentering. Tests normalize its full-ellipse
span sentinel (`10`) to `2*pi` and account for the rewrite's center of mass.

Time-series regressions additionally cover a massive nested multi-moon system,
siblings whose radial ordering crosses, frame-partition independence and mass
edits. The real Solar System preset is stepped through 128 days to check both
Charon's conic and Pluto's correctly scaled reflex curve about their common
barycenter. These checks detect propagation/extraction disagreement that isolated
snapshot fixtures cannot catch.

`ephemeris_contracts` uses independent SpiceyPy 8.2.0 outputs with the manifest's
kernel set at 0, 1, 30 and 365 days after the preset epoch. It checks relative
positions/velocities for Earth/Sun, Moon/Earth, Io/Jupiter, Titan/Saturn,
Triton/Neptune, Pluto/Sun and Charon/Pluto against physical-center observers,
replacement of all initial seeds, persistence and atomic failure outside kernel
coverage.

```text
cmake -S . -B build -DSTELLAR_BUILD_TESTS=ON -DSTELLAR_TEST_EPHEMERIS=ON -DSTELLAR_BUILD_RENDER_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
build/StellarForge.exe --self-test
```

Ephemeris tests are opt-in because the runtime/kernel data is not in Git. Default
CPU state tests skip SPICE playback presets unless that option is enabled.
`orbit_rendering_contracts` reads back actual GPU output to verify visibility,
opaque depth occlusion, paused camera culling and geometry at a 100 million AU
world offset. Rendering tests require Vulkan and a window system.
