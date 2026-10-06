# Stellar Forge workspace

The interface uses dark slate panels with cyan navigation accents and amber telemetry headings. Side panels share a layout with the transport bar and viewport overlays; coordinates use ImGui display pixels rather than framebuffer pixels.

## Navigation

- **System Outliner:** system switching, body search, selected/hovered body add and delete actions. The body list scrolls independently of its header. Comparison controls remain available in the Size Comparator scene.
- **Body Inspector:** tracking, centering, Go To, body/barycenter selection and edit mode stay above the scrolling detail area. The tab strip and physical-edit action footer stay visible; each tab scrolls independently. Narrow panels scroll their tab strip and wrap navigation actions into two rows. Camera distance and altitude are grouped in Overview.
- **Time transport:** playback and warp on the first row; date/time and Jump to Date or Render Timeline on the second. Timeline playback, scrubbing, Resume Here, cancellation and SPK export remain available.
- **Render:** opens categorized settings; the GAIA starfield toggle remains a quick action.

## Settings

| Category | Controls |
| --- | --- |
| Atmosphere | Quality, sky-view LUT, sampling, experimental aerial perspective, shadow bounds, noise and temporal integration |
| Optics | HDR, exposure, bloom, diffraction spikes and orbit MSAA |
| Visibility | Orbits, trajectory display, mission trails and habitable zones |
| Lighting | Atmosphere/cloud rendering, refraction, gravitational lensing, planetshine and ringshine |
| Terrain | Texture streaming, quadtree LOD, patch resolution and geometry HUD |
| Workspace | Comparison behavior, aligned/floating panels, panel widths, interface scale and layout reset |

Changes save automatically. Long input labels move above their controls when the pane is narrow. Each category scrolls independently and the close action remains outside the scrolling content.

Use **View → Reset Workspace Layout** to restore the standard panels. Use **Settings → Workspace → Align side panels** to enable freely movable/resizable windows. Existing graphics values and system data are retained. The new settings window has its own ImGui ID so obsolete settings-window coordinates do not place it off screen.

## Implementation and verification

`engine/ui/workspace.py` owns the theme and common geometry. The comparator uses these same bounds when fitting bodies and placing labels. Both settings writers persist panel widths, layout mode and scale. Physics, shaders, import/export implementations and body-edit callbacks retain their original behavior.

Run `python scripts/test_ui_workspace.py -v` in the project environment for native ImGui interaction and layout checks. `--render <directory>` additionally produces OpenGL UI previews with an empty viewport. The tests cover selection/search, CRUD requests, tracking/centering/Go To, inspector tabs, timeline actions, category navigation, capture requests, persistence, interface visibility, compact windows and HiDPI scaling.

The existing comparator suite (`python scripts/test_size_comparator.py --gpu`) checks rendering, atmosphere, rings, picking and focus. A bounded full-application smoke run also verified the real 56-body Solar System in simulation and comparator modes.

## Inspector organization

- **Overview:** aligned metric rows for mass, radius and gravity; separate rotation/shape, stellar-model, reflectance and camera-position groups. Proposed physical values and stellar previews are labeled while editing. Astronomical units have readable text labels when rendered in the inspector.
- **Orbit:** parent, live period, current relative distance/speed and orbital range first; expandable orbital elements, approximate gravitational limits and analytical precession estimates. Editing opens the elements and separates proposed range from live telemetry. The reference-frame selector retains its original behavior.
- **Atmosphere:** calculated properties and immediate controls, composition, expandable aerosol/scattering controls, then actions. Input gas weights remain normalized by the existing physics model.
- **Rings:** a selectable layer list and one layer editor, followed by the procedural generator and save actions. Selection survives ordinary edits and safely falls back when a layer is removed or regenerated.
- **Surface:** formerly Cosmetics; texture tools, expandable material controls, color/reflectance, and save/export groups. Texture map selectors wrap to preserve readable labels in narrow panels. The internal Cosmetics tab ID is retained.

Physical edits use the original Apply/Cancel callbacks. The Roche-limit guard remains active on every tab; its warning and Cancel action stay in the footer. Atmosphere, ring and surface edits still apply immediately, and existing save/import/export actions remain available.

While physically editing in a short, scaled panel, a Navigation menu gives the editable fields more room. Tracking, centering, Go To and barycenter actions remain inside it.

`engine/ui/inspector_widgets.py` supplies responsive metric rows, section headings and full-width field labels. The UI regression suite also checks physical-update dispatch, footer visibility after scrolling, the Roche guard, comparison arrays, atmosphere cache invalidation, selected-ring updates and surface material events. Native OpenGL previews and a bounded full-application run exercise Earth telemetry/atmosphere/textures and Saturn's textured and color ring layers.
