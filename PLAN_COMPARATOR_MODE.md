# Comparator Mode — Feature Plan

> **Status:** Draft for review. Edit freely.
> **Goal:** A new rendering mode that lays out bodies side-by-side for visual comparison,
> lit by an artificial directional light (the real star's color) instead of the on-screen star.
> Supports a single-system comparator and a multi-system comparator.

---

## 0. TL;DR of what already exists (do NOT re-implement)

- **Comparison system infrastructure** (`app.py`): `comparison_enabled`, `comparison_system_name`,
  `comparison_offset_au`, `shared_state_cmp`, a second physics thread, `bodies_data_cmp`,
  `visual_arr_cmp`, dual instance packing into `all_instances[num_bodies:]`, dual orbit compute,
  dual atmo pass. The UI is in the "System Comparison" collapsing header (~L5202).
  **This only supports ONE extra system offset on +X. Comparator mode is a different beast and
  will largely bypass / supersede it.**
- **Star lighting** is driven by a single UBO (`SceneData`, binding 1) shared by the sphere,
  ring, and atmosphere fragment shaders. It contains `u_num_stars` + `u_stars_pos_radius[]`,
  `u_stars_colors[]` (rgb=color, a=lum_eq), `u_stars_poles_obl[]`, `u_stars_pole_colors[]`.
  The sphere frag loops `for (s = 0; s < u_num_stars; s++)` doing Lambert + Blinn-Phong +
  inverse-square falloff + analytical eclipse shadows + ring shadows.
- **Camera**: perspective only. `view = look_at(cam_pos, [0,0,0], up)` with cam_pos from
  yaw/pitch/distance (`app.py` ~L3154). Projection built once per frame
  (`matrix44.create_perspective_projection_matrix`, ~L3184). `pyrr.matrix44` also has
  `create_orthogonal_projection_matrix` / `create_orthogonal_projection` (verified available).
- **Left-drag = orbit** (yaw/pitch), **right-drag = pan** (`cursor_pos_callback` ~L888).
- **Body hierarchy**: `parent_indices` / `tree_indices` / `tree_depths` in shared_state; JSON uses
  `parentId` (+ `orbitRef`). Star has `star_props` with `temp`, `lum`, `lum_eq`, `lum_pole`,
  `color` (#hex), etc. `temperature_to_rgb(temp_k)` in `system_manager.py` gives the star RGB.
- **Systems list**: `SystemManager.list_systems()`; `load_system_data(name)` → bodies_data raw;
  `load_system_from_data(raw)` → bundle. `meta.json` has `star.star_name`, `star.spectral_class`.

---

## 1. Feature Overview

### 1.1 Single-System Comparator Mode
- Enter via a UI toggle / button. Replaces the normal orbital scene.
- Uses the **3D scene with an orthographic camera** (front-on, no orbit).
- Layout (all in render-frame coords, +X right, +Y up, +Z toward camera):
  - **Star** at the **left** of the row.
  - **Planets** arranged **left → right in order of semi-major axis `a`** (or current radial
    distance), one per "column", further planets to the right.
  - **Moons** placed **below their parent planet** (−Y), stacked vertically if multiple.
- Planets are **NOT lit by the on-screen star**. Instead an **artificial directional light** lights
  ~3/4 of each body's visible disk, using the **actual star's color** (from `star_props.temp` →
  `temperature_to_rgb`, or the star body's `color` hex).
- **Dragging left/right pans** the whole layout horizontally (not orbit). Vertical drag pans
  vertically. Scroll zooms by changing the ortho view height (world units per pixel).

### 1.2 Multi-System Comparator Mode
- A modal/window lists **all systems** (`list_systems()`) with **checkboxes**.
- Each checked system is laid out as its **own row**; systems are **stacked vertically** (one below
  the next).
- Within each system row: same star-left, planets-right, moons-below-parent layout as 1.1.
- Each system uses **its own star's color** for its artificial light (so a row can be lit blue,
  another orange, etc.).
- Panning scrolls the whole grid; zoom scales ortho height.

### 1.3 Shared behavior
- Still real 3D meshes, textures, rings, atmospheres, oblateness, rotation/spin — just lit by the
  artificial light and laid out in a grid instead of orbits.
- Body Inspector still works (click a body → inspect). Picking uses the same screen-space code.
- Orbits **hidden** in comparator mode (they're meaningless in a grid). Habitable zones hidden.
- TAA / bloom / tonemap still apply.

---

## 2. Architecture Decisions

### 2.1 New module: `engine/comparator.py`
A pure-Python module (no GL) that owns all comparator layout + state logic. Keeps `app.py` from
balloating further.

```
class ComparatorManager:
    - systems: list of {name, bundle-ish metadata, star_color_rgb, star_lum, bodies_layout}
    - enabled: bool
    - multi: bool
    - pan_offset: np.array([x, y])   # world-space pan of the whole grid
    - ortho_height_au: float         # vertical world span shown (zoom)
    - layout() -> dict describing per-body world positions + per-system light color
    - add_system(name) / remove_system(name) / clear()
    - recompute_layout()
```

### 2.2 Layout algorithm (`ComparatorManager.layout`)
For a single system with bodies `b[0..N-1]` and parent tree:
1. Identify the **star** (type == 'Star', or `star_idx`).
2. **Planets** = direct children of the star, sorted by `a` ascending (fallback: current
   `|pos - star_pos|`).
3. Assign each planet an **X column** = `col_x + col_spacing * (i+1)` (star at col 0, x=0).
4. For each planet, its **moons** = children of that planet; place at
   `planet_x + 0`, `planet_y - moon_spacing * (j+1)` (stacked below).
5. **Spacing** is **radius-aware** (not `a`-based — `a` varies by orders of magnitude across
   systems). Each body gets a slot sized to `max(radius, min_slot)`; columns/gaps scaled so bodies
   never overlap. A `body_scale` multiplier (UI slider) lets the user grow/shrink the whole grid.
6. World position = `(col_x, row_y_for_system, 0)` + per-system row offset on Y for multi-system.
7. **Multi-system**: each system row gets a Y band = `-(system_index+1) * system_row_height`,
   with `system_row_height` derived from the tallest column (max moons) in that system.

Output per body: `{world_x, world_y, world_z, radius_render, system_index}`.
Star color per system: from `star_props.temp` via `temperature_to_rgb` (fallback to hex `color`).

### 2.3 Artificial light (shader-level)
**Approach: directional light, no inverse-square, no eclipses.** Cleanest and matches "3/4 lit".

Add to `SceneData` UBO (or as standalone uniforms — see decision below):
- `int u_comparator_mode`     // 0 = off, 1 = on
- `vec3 u_comp_light_dir`     // world-space, normalized; points from body toward light
- `vec3 u_comp_light_color`   // linear rgb, pre-exposed
- `float u_comp_light_intensity`

**Where to put it:** The UBO is `std140` with a fixed binary layout consumed by 3 shaders. Adding
fields in the middle would shift all offsets. **Safest: append new fields at the END of the UBO**
(after `u_caster_atmos`) and update `ubo_staging` size + all three shader `SceneData` blocks
identically. Alternatively use **standalone `uniform` values** (like `u_camera_pos`, `u_exposure`
already do) — **DECISION: use standalone uniforms** to avoid touching the std140 layout and the
`ubo_staging` packing. Standalone uniforms are guarded with `if 'u_xxx' in prog:` like the rest of
the codebase.

**Shader changes** (sphere_fragment_shader, ring_fragment_shader, atmo_fragment_shader):
```
uniform bool u_comparator_mode;     // or int
uniform vec3  u_comp_light_dir;
uniform vec3  u_comp_light_color;
uniform float u_comp_light_intensity;

// In the planet (non-star) branch of sphere frag, replace the `for (s...)` star loop with:
if (u_comparator_mode) {
    vec3 L = normalize(u_comp_light_dir);
    float NdotL = dot(N, L);
    float diffuse = clamp(NdotL, 0.0, 1.0);          // hard terminator (or soft with a small eps)
    // optional soft terminator: clamp((NdotL + 0.05)/1.05, 0, 1)
    vec3 lit = local_f_color * u_comp_light_color * u_comp_light_intensity * diffuse;
    // specular
    vec3 H = normalize(L + V);
    float specular = pow(max(dot(N,H),0.0), 64.0) * spec_intensity * diffuse;
    total_diffuse_color  = lit;
    total_specular_color = u_comp_light_color * u_comp_light_intensity * specular * 0.5;
    // SKIP eclipse caster loop, SKIP ring shadow loop in comparator mode.
} else {
    // existing star loop (unchanged)
}
```
- **3/4 lit**: choose `u_comp_light_dir` so the angle between light dir and view dir (~+Z, since
  camera looks down −Z) is ~30°–45°. E.g. `normalize([0.3, 0.2, 0.93])` gives a gibbous 3/4 phase.
  Make it a UI slider ("Light Angle") if desired.
- **Star sphere itself** (`f_is_star > 0.5` branch): unchanged — stars are self-luminous, so the
  on-screen star renders normally with its own color. The artificial light only affects planets.
- **Rings**: in the ring frag star loop, same `if (u_comparator_mode)` branch using
  `u_comp_light_dir` for the sun direction and `u_comp_light_color` for tint; skip planet-cast
  eclipse on rings (or keep host-planet shadow which still makes sense geometrically).
- **Atmosphere**: atmo frag uses the star for the sun direction. Add the same branch: when
  comparator mode, sun dir = `u_comp_light_dir`, sun color = `u_comp_light_color`, no falloff.
- **Planetshine**: disable in comparator mode (set `u_planetshine_enabled=false` for the pass) —
  it's a bounce-light from the on-screen star and is meaningless here. Or feed the comp light.

### 2.4 Camera: orthographic + pan
In `App.run` per-frame, when `comparator.enabled`:
- **Projection**: replace the perspective matrix with
  `matrix44.create_orthographic_projection_matrix(width, height, near, far, dtype='f4')` where
  `height = comparator.ortho_height_au`, `width = height * aspect`. Keep the log-depth rewrite
  (the shaders do `gl_Position.z = log2(...)` using `u_depth_C`/`u_far`; set `near` small, `far`
  large as today). **Verify**: `create_orthogonal_projection_matrix` signature in this pyrr version
  (params may be `left,right,bottom,top,near,far` or `w,h,near,far`). Use the bounds form if needed.
- **View**: front-on, no yaw/pitch. `view = look_at([0,0,cam_dist], [0,0,0], [0,1,0])`. `cam_dist`
  can be a constant (e.g. 100 AU) since ortho doesn't care about distance for sizing — only for
  depth ordering. Keep `cam_pos` consistent for the shaders' `u_camera_pos`.
- **Pan**: `comparator.pan_offset` (x,y) is applied by offsetting the view matrix translation, or
  more simply by adding it to `cam_origin` so all `pos_rel_all` shifts. Cleanest: translate the
  view: `view[3][0] -= pan_x; view[3][1] -= pan_y` (or build view with target = pan_offset).
- **Zoom**: scroll changes `ortho_height_au` (smaller = zoom in), NOT `camera["distance"]`.
  `scroll_callback` gets a comparator branch.
- **Drag**: `cursor_pos_callback` left-drag branch: when comparator enabled, adjust `pan_offset`
  by `(dx, dy) * world_per_pixel` instead of yaw/pitch. Right-drag can do nothing or also pan.

### 2.5 Instance packing in comparator mode
Instead of packing `pos_snap_render` (orbital positions), pack **layout positions** from
`ComparatorManager.layout`. The bodies still come from the loaded systems (primary + however many
comparison systems), but their world positions are overwritten by the grid layout each frame.

Concretely, in the per-frame instance-packing block (~L3316):
```
if comparator.enabled:
    layout = comparator.layout()           # dict: body_key -> (x,y,z,r)
    for each rendered body (primary + cmp systems):
        all_instances[i, 0:3] = layout_pos  # overrides orbital pos
        # radius: optionally scale by comparator.body_scale
else:
    # existing pos_rel_all packing
```
**Body identity across systems**: use `(system_index, body_index_in_system)` as the key. The
existing code already concatenates primary then `cmp` into `all_instances`; generalize to a list
of "active comparator systems" each with its own shared_state slot.

### 2.6 Multi-system plumbing
The current code hardcodes exactly two systems (primary + one `*_cmp`). Multi-system needs N.
**Plan**: refactor the per-system state into a **list** `self.system_slots` where each slot is:
```
{
  shared_state, time_ctrl, physics_thread, running,
  bodies_data, visual_arr, num_bodies, star_idx, star_color_rgb, star_lum,
  pos_snap, vel_snap, mass_snap, parent_snap, tree_*, ...
  ring_render_groups, ring_precomputed, atmo_bodies, tex_idx_arr, rot_*_arr,
}
```
- Slot 0 = the primary/active system (reuses existing main-system state).
- Slots 1..N = comparison systems (generalization of the current single `*_cmp`).
- `ComparatorManager` holds the list of *active* system names; adding/removing a checkbox
  spawns/tears-down a slot (post a `system_switch_request` to that slot's physics thread, exactly
  like the current comparison combo does at `app.py` ~L5213).
- All slots' bodies are packed into one big `all_instances` buffer (resize as needed — the code
  already does `np.vstack`/`np.delete` for add-body mode, ~L2668; reuse that resizing logic).
- Star gather loop (~L3560) iterates all slots; in comparator mode it does **not** feed the
  real stars into the UBO for lighting — instead it sets the standalone comp-light uniforms per
  the *currently inspected/primary* system's star color (or, for multi-system, we need per-body
  light color — see 2.7).

### 2.7 Per-system light color in multi-system (the one real complication)
The artificial light color should differ per system row. But the sphere frag's comp-light uniform
is global (one `u_comp_light_color`). Options:
- **(A) One global light color** = the primary system's star color. Simplest. Other rows lit with
  a "neutral" or the primary's tint. **Probably fine for v1**; document as a known limitation.
- **(B) Per-body light color via instance buffer.** We have spare instance floats
  (`INSTANCE_FLOATS=28`, 7 vec4s = 28 used; check if any are free — `f6.z`, `f6.w` appear unused
  after `tex_idx`/`rotation_angle`). Pack a per-body `comp_light_color` into 2 spare floats?
  Only 3 floats needed for RGB; we likely have `instances[idx*7+6].z` and `.w` = 2 floats free →
  not quite enough for RGB. **Alternative**: add an 8th vec4 to the instance buffer
  (`INSTANCE_FLOATS=32`) holding `comp_light_color.rgb` + `comp_light_intensity`. This requires
  updating `culling_compute_shader` and `sphere_vertex_shader` instance stride (`idx*7` → `idx*8`)
  and the `all_instances_buffer` reserve size. **Moderate but contained change.**
- **(C) Draw the scene once per system row** with the right comp-light color, clearing depth
  between. N draw calls instead of 1. Heavier but zero shader/instance changes.

**Recommendation: v1 uses (A). v2 upgrades to (B)** by extending the instance buffer to 32 floats
and adding `u_comp_per_body_light` mode in the shader that reads the per-instance color. This keeps
v1 small and leaves a clean upgrade path. **Confirm preference.**

### 2.8 Orbits, HZ, eclipses in comparator mode
- Orbits: force `show_orbits=False` behavior (skip the orbit compute + draw blocks) when
  `comparator.enabled`.
- Habitable zones: skip the HZ draw block.
- Eclipse caster selection / ring coplanar masks: still computed but the shader skips the eclipse
  loops when `u_comparator_mode` is true, so it's wasted CPU — **gate the caster selection with
  `if not comparator.enabled`** to save the per-frame sort.

### 2.9 UI
New collapsing header **"Comparator Mode"** (next to "System Comparison"), containing:
- `imgui.checkbox("Enable Comparator Mode", ...)`
- `imgui.radio_button("Single System"/"Multi-System", ...)`
- **Single**: shows the current system name (read-only or a combo to pick which system to
  comparator-ize — defaults to the loaded primary system).
- **Multi**: a child window listing all `list_systems()` with checkboxes; checking adds a row,
  unchecking removes it. Show star name + spectral class from `meta.json`.
- Sliders: **Body Scale** (grid spacing multiplier), **Light Angle** (rotates comp light dir),
  **Light Intensity**, **Ortho Height / Zoom** (also scroll).
- A "Reset View" button (pan = 0, ortho_height = auto-fit).
- When comparator is enabled, **disable/hide** the "System Comparison" header (mutually exclusive)
  and the orbit/HZ toggles.

---

## 3. Implementation Steps (ordered, each independently testable)

### Step 1 — `engine/comparator.py` (layout only, no GL)
- `ComparatorManager` class with `enabled`, `multi`, `pan_offset`, `ortho_height_au`,
  `body_scale`, `light_angle_deg`, `light_intensity`, `active_systems` (list of names).
- `compute_system_layout(bodies_data, star_idx, origin_x, origin_y) -> (positions np.array
  [N,3], radii np.array [N], star_color_rgb, star_lum)`.
- `compute_grid()` → builds the full per-body layout for all active systems (rows stacked).
- Pure functions, unit-testable with a fake bodies_data list.
- **Test:** print layout for Solar System + Achernar; verify star left, planets right, moons below.

### Step 2 — Shader uniforms + comp-light branch
- Add `uniform bool u_comparator_mode; uniform vec3 u_comp_light_dir; uniform vec3 u_comp_light_color;
  uniform float u_comp_light_intensity;` to **sphere_fragment_shader**, **ring_fragment_shader**,
  **atmo_fragment_shader** (guarded `if 'u_xxx' in prog:` in app.py).
- Add the `if (u_comparator_mode) { ... } else { existing }` branch in the planet lighting path of
  each shader. Skip eclipse + ring-shadow sub-loops in comp mode.
- **No UBO layout change.**
- **Test:** set uniforms from app.py with hardcoded values, toggle on with a fake layout, confirm
  planets lit from an angle with no eclipses.

### Step 3 — Orthographic camera path in `app.py`
- Add `if self.comparator.enabled:` branch in the per-frame projection/view build (~L3184).
  Use `create_orthogonal_projection_matrix`. Keep log-depth uniforms consistent.
- Pan via `pan_offset` applied to view translation.
- Zoom via `ortho_height_au` in `scroll_callback` (new comparator branch).
- Left-drag → pan in `cursor_pos_callback` (new comparator branch).
- **Test:** enable comparator with the *existing* orbital positions still being packed; confirm
  ortho + pan + zoom work and nothing breaks.

### Step 4 — Instance packing from layout
- In the instance-packing block (~L3316), when `comparator.enabled`, overwrite `all_instances[:,
  0:3]` with `comparator.compute_grid()` positions (per body, per slot). Keep radii (optionally
  scaled by `body_scale`).
- Map each slot's bodies to their layout rows. Star sphere still packs normally (self-luminous).
- **Test:** single-system comparator: see star left, planets right, moons below, lit by comp light.

### Step 5 — Multi-system slots refactor
- Generalize the single `*_cmp` state into `self.system_slots` list.
- Slot add/remove driven by `ComparatorManager.active_systems` + checkboxes; each add posts a
  `system_switch_request` to a fresh `shared_state` + spawns a physics thread (reuse the existing
  comparison-thread spawn code, parameterized).
- Resize `all_instances` buffer for total body count across slots (reuse the vstack/resize helpers).
- **Test:** check 2 systems → see 2 rows; check 3 → 3 rows; uncheck → row removed.

### Step 6 — UI panel
- "Comparator Mode" collapsing header with all controls from §2.9.
- Wire checkboxes to `ComparatorManager.add/remove_system`.
- Mutual exclusion with "System Comparison".
- Auto-fit `ortho_height_au` to the grid bounds on enable / on system add.
- **Test:** full UX flow.

### Step 7 — Polish & edge cases
- Picking in ortho (the pick code uses `vp_matrix` + perspective clip; verify it works with ortho —
  the math is projection-agnostic since it uses `clip_pos/clip_pos.w`, but the FOV-based pixel-size
  fallback for tiny bodies uses `fov_factor_pick`; switch to an ortho-aware size test).
- Inspector: orbital elements are meaningless in comparator; show physical/thermal/atmo tabs only,
  or keep elements but label them as the real orbital elements (still valid — they come from the
  physics sim, not the layout). **Decision: keep showing real elements** (they're still computed
  by the physics thread); the layout is purely visual.
- Orbit lines are MSAA-only; confirm they composite correctly in comp mode.
- Bloom/exposure unchanged.
- Performance: skip orbit compute, skip caster sort, skip HZ in comp mode.

---

## 4. Files to edit (summary)

| File | Changes |
|---|---|
| `engine/comparator.py` | **NEW** — ComparatorManager + layout. |
| `engine/shaders.py` | Add 4 comp-light uniforms + `if (u_comparator_mode)` branch in `sphere_fragment_shader`, `ring_fragment_shader`, `atmo_fragment_shader`. No UBO/stride change in v1. |
| `engine/app.py` | Ortho projection/view branch; pan/zoom/scroll branches; instance packing from layout; `system_slots` list refactor ( generalize `*_cmp` ); star-gather → comp-light uniform setup; skip orbits/HZ/casters in comp mode; new "Comparator Mode" UI panel; picking ortho fix. |
| `engine/system_manager.py` | Possibly a helper `get_star_color_rgb(bodies_data)` wrapping `temperature_to_rgb`. (Optional.) |
| `PROJECT_MAP.md` | Document comparator mode + new module. |

No changes expected in `physics_core.py`, `kepler_analytical.py`, `render_utils.py`,
`post_shaders.py`, `system_manager.py` (core), `spice_manager.py`.

---

## 5. Open Questions for you

1. **Multi-system light color (§2.7):** v1 = single global light color (primary system's star
   color), or go straight to per-body light color via an 8th instance vec4 (INSTANCE_FLOATS 28→32)?
2. **Light direction:** fixed 3/4 angle, or expose a "Light Angle" slider? (Recommend slider,
   default ~35°.)
3. **Body scale / spacing:** auto-fit so the widest system fills the view, or fixed AU spacing
   with a slider? (Recommend auto-fit + slider override.)
4. **Moons:** stack straight down (−Y) in order of `a`, or fan out slightly in X too? Straight-down
   is cleaner; fanning avoids overlap for many moons.
5. **Should the on-screen star still be rendered** (self-luminous sphere on the left), or hidden?
   (Plan renders it; it just doesn't light the planets.)
6. **Multi-system rows:** sorted alphabetically, by spectral class, or by user check order?
7. **Should comparator mode pause physics** (bodies don't move/spin) or keep spinning? Layout is
   static positions, but spin/rotation could still animate. (Recommend: keep spin, freeze orbit
   positions — i.e. physics thread runs but we ignore positions for layout.)
8. **Atmosphere in comparator:** render atmo with comp light (needs the atmo shader branch) or
   skip atmo for v1? (Recommend include — it's a small branch.)

---

## 6. Risks / things to watch

- **Ortho + log-depth**: the shaders rewrite `gl_Position.z` with a log formula that assumes
  perspective `w = eye_z`. Under ortho, `w` is constant (1.0), so the log-depth mapping may
  produce wrong/flat depth. **Mitigation:** in comp mode either (a) use a standard linear depth
  in the shaders via a `u_ortho` flag, or (b) disable TAA/log-depth and rely on a large far plane.
  Test early in Step 3.
- **Picking** pixel-size fallback uses `fov_factor_pick = 1/tan(fov/2)` — invalid for ortho. Replace
  with `pixel_size = radius / cam_dist * screen_height` equivalent for ortho
  (`radius / ortho_height * screen_height`).
- **Instance buffer resize** when systems are added/removed mid-frame — guard against races with
  the physics threads (already handled for add-body mode; reuse the pattern).
- **`system_slots` refactor** touches a lot of duplicated `*_cmp` code; do it as a mechanical
  search-replace pass, keeping a working commit before starting Step 5.

---

*End of draft. Awaiting your edits.*
