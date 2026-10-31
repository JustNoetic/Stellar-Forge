# Application settings

Settings uses a sidebar with five graphics pages (Display, Scene & Orbits,
Atmosphere, Lighting & Rings, Planet Surfaces), Navigation and Workspace
preferences, and an Advanced page for experiments and diagnostics. Dependent
controls retain their values while disabled. Changes apply immediately.

Display includes Bloom & diffraction: Off, Fast, or Four-spike (default), with
scatter strength, highlight limit, spike length, strength, rotation and threshold controls.
Only pixels brighter than the spike threshold (default 4 HDR units) cast spikes.
The highlight limit softly caps only the light extracted for glare, preserving
the scene's HDR solar disc. The default is 64; lower values soften overexposed
halos and spike tips. Spike tails also taper smoothly to zero.
Bloom requires HDR and preserves orbit guide exclusion in ordered transparency.
The legacy orbit reference path bypasses bloom. See `bloom.md` for the filter,
HDR saturation limits and local verification.

Press **F11** to toggle fullscreen on the display containing most of the window.
Fullscreen uses that display's current resolution and refresh rate. Pressing F11
again restores the previous window position, size and maximized state. The
shortcut also works while UI controls have keyboard focus. Fullscreen is session
state and is not saved to preferences.

Lighting & Rings includes Cloud rendering: 2D texture (default) or Volumetric
(experimental). `cloud_mode` persists as 0 or 1 alongside `clouds_enabled`.
The 2D mode reuses existing coverage mips and atmosphere transport; see
`oceans_clouds.md` for per-body altitude and moisture controls.

Interactive sessions load `data/settings.json` through `paths::external_path`.
Graphics settings, panel visibility and sizes, interface scale, distance units,
deletion confirmation, and navigation preferences persist. Camera pose, tracked
objects, simulation state, hidden UI, open dialogs and the selected settings page
remain session state. Flight speed is saved after navigation changes, including
automatic speed changes when focusing a body or switching systems.

Changes from the settings window, main menu, keyboard shortcuts and camera
controls share the same save path. Saving waits 650 ms after the last change and
until active controls are released; shutdown flushes pending changes. Failed
writes retry after three seconds and appear in the settings footer. The existing
ImGui layout file continues to own floating window positions and docking.

`core/user_settings` owns the versioned JSON boundary. Each field is checked for
its JSON type, representable numeric range and owning settings domain's accepted
range. Invalid fields fall back individually without discarding valid fields.
Missing files, sections and fields use the current C++ defaults. Corrupt files
produce a warning and are replaced only after a subsequent preference change.
Unsupported schema versions use defaults and disable saving to preserve the file.

Saves write and close a sibling temporary file before replacing the destination
atomically. The user's file is ignored by Git and excluded from installation.
Benchmarks and `--self-test` neither load nor write these preferences or the
ImGui layout file, so local choices cannot affect their results. The smoke test
renders each settings page in its hidden window.

Defaults remain in `RenderSettings`, `WorkspaceSettings` and
`NavigationSettings`. The built-in defaults match the saved preference baseline
chosen on 26 October 2026, with orbit lines enabled. This includes 1,030 orbit
segments, 16-pixel terrain error, four atmospheric shadow samples, fast ring
shadows and dual-source blending enabled, and progressive shadows, rotation
history reuse, VRS and direct-sunlight-only rendering disabled. Default flight
speed is `9.59133562721224e-05` AU/s; workspace defaults are unchanged.
Existing saved values take precedence over later default edits; new fields use
their code defaults. **Restore all defaults** immediately reapplies those three
default objects and saves them through the normal path. **Reset Workspace
Layout** only resets the existing workspace layout preferences.

Local verification:

```text
cmake --build build --target StellarForge StellarUserSettingsTests
ctest --test-dir build -R "user_settings_contracts|simulation_tracking_rendering_contracts" --output-on-failure
```

The CPU contracts exercise first launch, partial files, every persisted field,
invalid types and ranges, corrupt files, future versions, transient state,
atomic replacement and write failures.
