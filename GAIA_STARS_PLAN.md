# GAIA Star Catalog — Implementation Plan

> Goal: place real stars from **Gaia DR3** at their true 3D heliocentric/barycentric
> positions in the engine's coordinate frame (e.g. Proxima Centauri 4.24 ly in the
> correct direction), animated by **proper motion** over sim time, rendered as HDR
> point sprites that integrate with the existing bloom / diffraction-spike pipeline.
>
> **In scope (now):** fetch script, catalog layer, 3D point pass, settings/UI toggle.
> **Deferred:** near-star sphere-mesh crossfade (zoom-to-surface), background
> at-infinity fallback layer (Milky Way band from faint stars), star-name picking.

---

## 0. Design constraints (why it's built this way)

1. **Never in the physics system.** GAIA stars are *not* `Simulation` bodies —
   460k stars in O(N²) IAS15 is instant death, and their gravitational effect on
   the Solar System is ~10⁻¹⁴ of the Sun's. Positions are **static + analytic**:
   `p(t) = p₀ + (pm_vec · t_years)`, renormalized. Costs nothing, exact enough
   for millennia of sim time (linear proper-motion approximation).
2. **Separate catalog layer, not `bodies_data`.** `shared_state`, the outliner,
   inspector, and every per-body loop in `app.py` are body-indexed. 460k entries
   there would choke ImGui and all CPU loops. Stars live in their own module,
   VBO, and draw call.
3. **Camera-relative packing in float64.** Star distances reach 10⁸–10⁹ AU.
   f8 camera-relative subtraction on the CPU each frame → f4 VBO. (Pattern
   already exists: `compute_orbit_elements(..., cam_origin, ...)`.)
4. **Physical brightness, not painted.** Intensity per frame =
   `10^(-0.4 · M_G) / d_au² · scale` (absolute magnitude + inverse square).
   Proxima is invisible from Earth and blazes when you approach — emergent.

---

## 1. Data pipeline — `scripts/fetch_gaia.py` (new)

Mirrors the `fetch_horizons.py` CLI pattern (`--out`, `--max-mag`, `--test`).

**Source:** VizieR TAP (`I/355/gaiadr3`, sync, easier) with ESA Gaia Archive
async TAP as documented fallback. **Query in RA chunks** (e.g. 24 × 15° bands)
to stay under sync row limits; merge and sort at the end.

ADQL (per chunk):
```sql
SELECT ra, dec, parallax, parallax_over_error, pmra, pmdec, radial_velocity,
       phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
FROM "I/355/gaiadr3"
WHERE phot_g_mean_mag < {max_mag}
  AND parallax > 0 AND parallax_over_error > 5
  AND ra >= {ra0} AND ra < {ra1}
```
- Default `--max-mag 10` → ~460k stars full sky. Covers the nearest-star 3D
  layer now **and** the deferred background fallback (same binary).
- `parallax_over_error > 5` keeps distances trustworthy (≤20% error).
- The Sun is not a Gaia source — irrelevant, system stars render separately.

**Output: `data/gaia/stars.bin`** — custom binary, `.npy`-free for PyInstaller
friendliness:
```
header:  magic "GAIA1" | u32 version | u32 count
records: structured dtype (little-endian):
  ra_deg f32, dec_deg f32, parallax_mas f32, pmra_masyr f32,
  pmdec_masyr f32, rv_kms f32, g_mag f32, bmrp f32
32 bytes/star → ~15 MB at 460k.
```
`--test` mode fetches ~5k bright stars only, for quick iteration.

---

## 2. Catalog layer — `engine/rendering/star_catalog.py` (new)

```
class StarCatalog:
    load(path)               -> None (returns quietly if file missing)
    __len__, loaded flag
    pack_render(t_years, cam_pos_render, vbo_map)  -> None (numpy, per frame)
```

**Load-time bake (once, cached):**
- Unit vectors from RA/Dec (ICRS equatorial) → **ecliptic** via rotation about X
  by `OBLIQUITY` (`engine/core/constants.py`) → **render frame** remap
  `(x, z, -y)` (PROJECT_MAP §7 gotcha). Store `dir_ecl` f8 (N,3).
- `dist_au = 1000·AU_TO... / parallax_mas` (f8). `pos0 = dir_ecl · dist_au`
  (heliocentric; Solar-barycenter offset ≤ 0.005 AU — ignore, note it).
- Proper motion: `pm_tan = (pmra·cos(dec), pmdec)` mas/yr → rad/yr in the
  local tangent frame (east/north basis from RA/Dec) → f8 (N,3) in render
  frame. Radial velocity affects distance at ~km/s — negligible for now
  (store in binary anyway; document as ignored).
- `M_G = g_mag - 5·log10(dist_pc/10)` (absolute magnitude, f4).
- Color: `bmrp` → T_eff via a Gaia BP−RP polynomial fit → RGB via
  **existing** `temperature_to_rgb()` (`engine/ephemeris/system_manager.py`).
  Bake `rgb` f4 (N,3). Add a small clamp for extreme colors (subdwarfs/WDs).

**Per-frame `pack_render` (all numpy, target < 2 ms @ 460k):**
```
pos_t = pos0 + pm_tan * t_years            # (N,3) f8
rel   = pos_t - cam_pos_render             # f8 camera-relative
dist  = norm(rel)                          # f8 (N,)
intensity = flux_scale * 10**(-0.4*M_G) / dist²   # f8 (N,)
pack interleaved f4 (N,7): [rel.xyz, rgb, intensity]
→ vbo.write(packed)                        # streamed
```
Optional fast path: pre-mask to stars with intensity above ~10⁻⁵ of max to
shrink the VBO (frustum culling is pointless for points — they're cheap; the
mask is mainly so distant-fuzz doesn't accumulate shading cost).

---

## 3. GLSL — `engine/glsl/celestial/starfield.vert/.frag` (new)

**vert:**
- In: `in vec3 in_pos; in vec3 in_color; in float in_intensity;`
- Uniforms: `u_mvp` (proj·view), `screen_height`, `fov_factor`,
  `u_exposure`, `u_hdr_enabled`, `u_max_point_px`.
- `gl_PointSize = clamp(base_px · sqrt(intensity·norm), 1.0, u_max_point_px)`
  (mirrors `point_celestial.vert`'s screen-height/fov scaling).
- Discard (`gl_Position = vec4(0,0,2,1)`) when the star is behind the camera
  (`dot(rel, cam_fwd) < 0`) — points have no built-in near-plane culling.

**frag:**
- Circular Gaussian/Gauss-Legendre-ish falloff from `gl_PointCoord`,
  soft edge; output `vec4(color · intensity · falloff, 1.0)`.
- HDR-first: respect `u_hdr_enabled`/`u_exposure` exactly like
  `point_celestial.frag` so bright stars **feed the existing bloom pyramid and
  get diffraction spikes from `star_streak.frag` for free**.

**Blending & state:** additive blending (`ONE, ONE`), **depth test ON,
depth write OFF**, drawn **after the opaque sphere pass, before atmosphere
pass 1** — planets/moons correctly occlude stars, and atmospheres/rings/clouds
blend over star pixels naturally.

---

## 4. Engine integration — `engine/app.py` (edits)

1. **Import & load** (~L850 region, after textures): try `StarCatalog.load()`;
   if `data/gaia/stars.bin` is missing, print a one-line hint
   (`python scripts/fetch_gaia.py`) and disable the feature silently.
2. **GL objects** (~L1415 region, next to `prog_point_celestial`):
   compile `prog_starfield`; create a streamed interleaved VBO +
   VAO (`'3f 3f 1f', 'in_pos', 'in_color', 'in_intensity'`).
   ⚠ Enable `ctx.enable(moderngl.PROGRAM_POINT_SIZE)` or `gl_PointSize` is
   ignored by the core profile driver.
3. **Render loop:**
   - After the state snapshot (~L2359 region): if `camera["show_gaia_stars"]`
     and catalog loaded → `catalog.pack_render(display_t, cam_pos_render, …)`
     and `vbo.write(...)`.
   - Draw after the sphere LOD pass, before `render_atmosphere_pass(1)`
     (~L5000 region): set uniforms, `ctx.enable(DEPTH_TEST)` + write-off +
     additive blend, `vao.render(moderngl.POINTS)`, restore state.
   - **Draw once** — the catalog is world-fixed, shared by the main and
     comparison systems. Do *not* duplicate per system.
4. **Settings** (`load_settings` L650 / `save_settings` L666):
   add `show_gaia_stars` (default True when catalog exists) to the saved dict.
5. **Coordinate note:** `cam_pos` must be the same barycentric render-frame
   position used for orbit geometry (`cam_origin` in `compute_orbit_elements`)
   — reuse that exact variable, don't recompute.

## 5. UI — `engine/ui/menu_bar.py` (edit)

View menu, next to "Show Orbits" (~L158):
`_, app.camera["show_gaia_stars"] = imgui.checkbox("GAIA Starfield", ...)`.
No inspector changes (stars are not inspectable bodies yet).

---

## 6. Verification — `scripts/test_gaia.py` (new)

Console test, no window:
1. Load catalog; assert Proxima Centauri (parallax ≈ 768.5 mas,
   RA 14h29m42.9s Dec −62°40′46″) is at ≈ 4.246 ly and its ecliptic-frame
   unit vector matches an independently computed ICRS→ecliptic transform.
2. Assert Sirius (A1V, M_G ≈ 1.43) has plausibly blue-white RGB;
   Betelgeuse (M_G ≈ −5.85) red and `10^(-0.4·M_G)` consistent.
3. Barnard's star: pm ≈ 10.4″/yr → after 1000 sim-years, angular
   displacement ≈ 2.9° on sky.
4. Simulated `pack_render` timing @ 460k stars (target < 2 ms).
5. From Earth (cam at origin): brightest catalog star should be Sirius
   with apparent G ≈ −1.46, and Proxima should be near-invisible (G ≈ 11).

Manual smoke test in-app: stars visible from Earth, occluded by planets,
bloom + spikes appear on Sirius/Canopus, toggle round-trips through
`graphics_settings.json`, comparison mode unaffected.

---

## 7. Deferred (do NOT build now)

- **Near-star sphere crossfade** — when a star's apparent size exceeds the
  point threshold, crossfade into `prog_spheres` (reuse `is_star` path and
  `point_celestial`'s 2.0–3.0 px fade trick). Requires per-star radius/luminosity
  (derivable from M_G + BC(color)) and camera-relative instance packing.
- **Background at-infinity fallback** — stars with poor parallax or beyond a
  distance cap rendered as directions on the unit sphere; reconstructs the
  Milky Way band from the same binary.
- **Picking & travel** — click a star to track it; "Nearest Stars" searchable
  panel; raised flight-speed cap or jump-to-star (5 AU/s → 14.8 h to Proxima).
- **Full 3D parallax when flying between stars** — falls out of the design
  automatically once distances are real; only depth handling needs care.

## 8. Implementation status

**Implemented (phase 1):**
- `scripts/fetch_gaia.py` — Gaia DR3 G<10 (475,724 stars, 14.5 MB at `data/gaia/stars.bin`)
  + Hipparcos V<2.5 bright supplement (fixes the Gaia saturation gap; Sirius A, Canopus,
  Vega, ...). Test fetch validated against known sky.
- `engine/rendering/star_catalog.py` + `glsl/celestial/starfield.*` + `app.py` draw pass
  (after spheres, before atmosphere pass 1) + View-menu toggle + settings persistence.
- `scripts/test_gaia.py` — 21/21 checks pass; pack_render 0.31 ms @ 475k stars.

**Deferred:** near-star sphere crossfade, background at-infinity fallback (Milky Way band),
picking/travel, interstellar camera (f8 pack path).

## 9. File checklist

| File | Action |
|---|---|
| `scripts/fetch_gaia.py` | new — TAP fetch → `data/gaia/stars.bin` |
| `engine/rendering/star_catalog.py` | new — load, bake, per-frame pack |
| `engine/glsl/celestial/starfield.vert/.frag` | new — HDR point sprites |
| `engine/rendering/shaders.py` | edit — re-export `starfield_*` |
| `engine/app.py` | edit — load, GL objects, pack+draw pass, settings |
| `engine/ui/menu_bar.py` | edit — View menu toggle |
| `scripts/test_gaia.py` | new — catalog sanity checks |
| `PROJECT_MAP.md` | edit — add the new module/passes |
