"""Categorized settings, retaining the engine's original setting callbacks."""
import imgui
from engine.core.constants import SKY_VIEW_RES_LABELS
from engine.ui.workspace import workspace, heading, SETTINGS_SECTIONS, reset_workspace


def _field(widget, label, *args, **kwargs):
    """Place long field labels above controls when the settings pane is narrow."""
    available = imgui.get_content_region_available()[0]
    label_width = imgui.calc_text_size(label.split('##')[0])[0]
    if label_width + imgui.get_style().item_inner_spacing.x <= available * 0.5:
        return widget(label, *args, **kwargs)
    imgui.text_wrapped(label.split('##')[0])
    imgui.push_item_width(-1)
    result = widget('##' + label, *args, **kwargs)
    imgui.pop_item_width()
    return result

def _atmosphere(app):
    settings_changed = False
    # Atmosphere Quality
    atmo_quality = app.camera.get("atmo_quality", 1)
    changed_aq, atmo_quality = _field(imgui.combo, "Atmosphere Quality", atmo_quality, ["Off", "Low (2D Shadows)", "High (Volumetric)", "Analytical (Sky-View + Slicing)"])
    if changed_aq:
        app.camera["atmo_quality"] = min(3, atmo_quality)
        settings_changed = True

    if atmo_quality == 3:
        sky_view_res = int(app.camera.get("atmo_sky_view_res", 0))
        sky_view_res_idx = min(len(SKY_VIEW_RES_LABELS) - 1, max(0, sky_view_res))
        changed_svr, new_svr_idx = _field(imgui.combo, "Sky-View LUT Resolution", sky_view_res_idx, SKY_VIEW_RES_LABELS)
        if changed_svr:
            app.camera["atmo_sky_view_res"] = new_svr_idx
            settings_changed = True

        imgui.text("Analytical sky and terrain lighting")
        changed_ap, aerial_volume = imgui.checkbox(
            "Fast Aerial Perspective (Experimental)",
            app.camera.get("atmo_aerial_volume", True))
        if changed_ap:
            app.camera["atmo_aerial_volume"] = aerial_volume
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Use a small 3D LUT for terrain haze. Disable to compare with direct endpoint lighting.")

        changed_bs, bounded_shadows = imgui.checkbox(
            "Bounded Shadow Raymarching (Experimental)",
            app.camera.get("atmo_bounded_shadows", True))
        if changed_bs:
            app.camera["atmo_bounded_shadows"] = bounded_shadows
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "When ring or body shadows intersect the atmosphere in Mode 3,\n"
                "only raymarch inside the shadow bounds instead of the entire chord.\n"
                "Respects the ray-step budget below. Savings depend on shadow coverage."
            )

        changed_steps, shadow_steps = _field(imgui.slider_int, 
            "Shadow Ray Steps", app.camera.get("atmo_steps_max", 32), 4, 128)
        if changed_steps:
            app.camera["atmo_steps_max"] = shadow_steps
            settings_changed = True
        changed_adapt, adaptive_steps = imgui.checkbox(
            "Adaptive Shadow Steps", app.camera.get("atmo_adaptive_steps", True))
        if changed_adapt:
            app.camera["atmo_adaptive_steps"] = adaptive_steps
            settings_changed = True
        if adaptive_steps:
            changed_max, max_steps = _field(imgui.slider_int, 
                "Max Adaptive Shadow Steps", app.camera.get("atmo_adaptive_steps_max", 128), 8, 256)
            if changed_max:
                app.camera["atmo_adaptive_steps_max"] = max_steps
                settings_changed = True

    elif atmo_quality > 0:
        atmo_res = float(app.camera.get("atmo_resolution", 1.0))
        changed_res, atmo_res = _field(imgui.slider_float, "Atmosphere Render Scale", atmo_res, 0.2, 1.0, "%.2fx")
        if changed_res:
            app.camera["atmo_resolution"] = atmo_res
            settings_changed = True

        if atmo_res < 0.999:
            vrs_threshold = float(app.camera.get("atmo_vrs_threshold_px", 100.0))
            changed_vrs, vrs_threshold = _field(imgui.slider_float, "VRS Low-Res Threshold (px)", vrs_threshold, 10.0, 1000.0, "%.1f")
            if changed_vrs:
                app.camera["atmo_vrs_threshold_px"] = vrs_threshold
                settings_changed = True

        max_steps = app.camera.get("atmo_steps_max", 32)
        changed_steps, max_steps = _field(imgui.slider_int, "Max Ray Steps", max_steps, 4, 128)
        if changed_steps:
            app.camera["atmo_steps_max"] = max_steps
            settings_changed = True

        adaptive_steps = app.camera.get("atmo_adaptive_steps", True)
        changed_adapt, adaptive_steps = imgui.checkbox("Adaptive Step Count", adaptive_steps)
        if changed_adapt:
            app.camera["atmo_adaptive_steps"] = adaptive_steps
            settings_changed = True

        if adaptive_steps:
            max_adapt_steps = app.camera.get("atmo_adaptive_steps_max", 128)
            changed_max_adapt, max_adapt_steps = _field(imgui.slider_int, "Max Adaptive Steps", max_adapt_steps, 8, 256)
            if changed_max_adapt:
                app.camera["atmo_adaptive_steps_max"] = max_adapt_steps
                settings_changed = True

        stochastic_steps = app.camera.get("atmo_stochastic", True)
        changed_stoch, stochastic_steps = imgui.checkbox("Stochastic Raymarching", stochastic_steps)
        if changed_stoch:
            app.camera["atmo_stochastic"] = stochastic_steps
            settings_changed = True

        if stochastic_steps:
            noise_types = ["Interleaved Gradient Noise (IGN)", "Spatiotemporal Blue Noise (STBN)"]
            cur_noise = int(app.camera.get("atmo_noise_type", 0))
            changed_nt, cur_noise = _field(imgui.combo, "Noise Type##mode2", cur_noise, noise_types)
            if changed_nt:
                app.camera["atmo_noise_type"] = cur_noise
                settings_changed = True

        temporal_accum = app.camera.get("atmo_temporal_accum", True)
        changed_ta, temporal_accum = imgui.checkbox("Temporal Integration (TAA)", temporal_accum)
        if changed_ta:
            app.camera["atmo_temporal_accum"] = temporal_accum
            settings_changed = True

        if temporal_accum:
            temporal_blend = float(app.camera.get("atmo_temporal_blend", 0.90))
            changed_tb, temporal_blend = _field(imgui.slider_float, "Temporal History Weight", temporal_blend, 0.50, 0.98, "%.2f")
            if changed_tb:
                app.camera["atmo_temporal_blend"] = temporal_blend
                settings_changed = True

    # Shadow Caster Budget
    caster_budget = app.camera.get("shadow_caster_budget", 32)
    changed_budget, caster_budget = _field(imgui.slider_int, "Shadow Caster Budget", caster_budget, 4, 64)
    if changed_budget:
        app.camera["shadow_caster_budget"] = caster_budget
        settings_changed = True

    imgui.separator()


    return settings_changed


def _optics(app):
    settings_changed = False
    # Exposure & HDR
    changed_hdr, app.camera["hdr_enabled"] = imgui.checkbox("HDR Mode", app.camera.get("hdr_enabled", True))
    if changed_hdr: settings_changed = True

    if app.camera.get("hdr_enabled", True):
        changed_exp, app.camera["exposure"] = _field(imgui.slider_float, "Exposure", app.camera.get("exposure", 1.0), 0.0001, 10000.0, "%.4f", imgui.SLIDER_FLAGS_LOGARITHMIC)
        if changed_exp: settings_changed = True

    # Bloom & Diffraction Spikes
    bloom_modes = ["Gaussian Blur", "Diffraction Spikes", "Hybrid (Spikes + Haze)"]
    current_bm = app.camera.get("bloom_mode", 2)
    current_bm_idx = current_bm if 0 <= current_bm < len(bloom_modes) else 2
    changed_bm, new_bm = _field(imgui.combo, "Bloom Mode", current_bm_idx, bloom_modes)
    if changed_bm:
        app.camera["bloom_mode"] = new_bm
        settings_changed = True

    if app.camera.get("bloom_mode", 2) in (1, 2):
        spike_counts = [4, 6, 8]
        spike_labels = ["4 Spikes (Cross)", "6 Spikes (JWST / Newtonian)", "8 Spikes (Octagram)"]
        curr_sc = app.camera.get("spike_count", 6)
        sc_idx = spike_counts.index(curr_sc) if curr_sc in spike_counts else 1
        changed_sc, new_sc_idx = _field(imgui.combo, "Spike Pattern", sc_idx, spike_labels)
        if changed_sc:
            app.camera["spike_count"] = spike_counts[new_sc_idx]
            settings_changed = True

        changed_cbi, app.camera["conv_bloom_intensity"] = _field(imgui.slider_float, "Diffraction Spikes Intensity", app.camera.get("conv_bloom_intensity", 0.5), 0.0, 5.0, "%.2f")
        if changed_cbi: settings_changed = True
        changed_sl, app.camera["spike_length"] = _field(imgui.slider_float, "Diffraction Spikes Length", app.camera.get("spike_length", 1.0), 0.2, 3.0, "%.2f")
        if changed_sl: settings_changed = True
        changed_sa, app.camera["spike_angle"] = _field(imgui.slider_float, "Diffraction Spikes Angle", app.camera.get("spike_angle", 0.0), 0.0, 180.0, "%.1f deg")
        if changed_sa: settings_changed = True
        changed_srl, app.camera["spike_roll_lock"] = imgui.checkbox("Lock Spikes to Camera Roll", app.camera.get("spike_roll_lock", True))
        if changed_srl: settings_changed = True
        changed_sd, app.camera["spike_dispersion"] = _field(imgui.slider_float, "Dispersion (Rainbow)", app.camera.get("spike_dispersion", 0.015), 0.0, 0.05, "%.3f")
        if changed_sd: settings_changed = True

        spike_res_opts = [0, 1]
        spike_res_labels = ["Quarter Res (Fast - 0.1ms)", "Half Res (Ultra)"]
        curr_sq = app.camera.get("spike_quality", 0)
        sq_idx = curr_sq if 0 <= curr_sq < len(spike_res_opts) else 0
        changed_sq, new_sq_idx = _field(imgui.combo, "Spikes Resolution", sq_idx, spike_res_labels)
        if changed_sq:
            app.camera["spike_quality"] = spike_res_opts[new_sq_idx]
            app.last_fb_size = (0, 0)
            settings_changed = True

        changed_cds, app.camera["conv_bloom_dynamic_scale"] = imgui.checkbox("Dynamic Spike Distance Shrinking", app.camera.get("conv_bloom_dynamic_scale", True))
        if changed_cds: settings_changed = True

    if app.camera.get("bloom_mode", 2) in (0, 2):
        changed_bi, app.camera["bloom_intensity"] = _field(imgui.slider_float, "Gaussian Haze Intensity", app.camera.get("bloom_intensity", 0.05), 0.0, 1.0, "%.3f")
        if changed_bi: settings_changed = True
        changed_bt, app.camera["bloom_threshold"] = _field(imgui.slider_float, "Gaussian Haze Threshold", app.camera.get("bloom_threshold", 1.0), 0.0, 10.0, "%.2f")
        if changed_bt: settings_changed = True

    # MSAA
    msaa_options = [0, 2, 4, 8]
    msaa_labels = ["Off", "2x", "4x", "8x"]
    current_msaa = app.camera.get("msaa_samples", 4)
    current_idx = msaa_options.index(current_msaa) if current_msaa in msaa_options else 2
    changed_msaa, new_msaa_idx = _field(imgui.combo, "Orbit MSAA", current_idx, msaa_labels)
    if changed_msaa:
        app.camera["msaa_samples"] = msaa_options[new_msaa_idx]
        settings_changed = True


    return settings_changed


def _visibility(app):
    settings_changed = False
    # Orbits & Fade
    changed_so, app.camera["show_orbits"] = imgui.checkbox("Show Orbits", app.camera.get("show_orbits", True))
    if changed_so: settings_changed = True

    if app.camera.get("show_orbits", True):
        orbit_fade_dir_idx = app.camera.get("orbit_fade_dir_idx", 0)
        changed_ofd, orbit_fade_dir_idx = _field(imgui.combo, "Orbit Fade", orbit_fade_dir_idx, ["Bright Behind", "Bright Ahead"])
        if changed_ofd:
            app.camera["orbit_fade_dir_idx"] = orbit_fade_dir_idx
            settings_changed = True

        orbit_min_alpha = app.camera.get("orbit_min_alpha", 0.1)
        changed_oma, orbit_min_alpha = _field(imgui.slider_float, "Min Orbit Alpha", orbit_min_alpha, 0.0, 1.0, "%.2f")
        if changed_oma:
            app.camera["orbit_min_alpha"] = orbit_min_alpha
            settings_changed = True

        imgui.spacing()
        imgui.text_colored("Spacecraft / Ephemeris Orbits", 0.4, 0.8, 1.0)

        ephem_pts = int(app.camera.get("ephem_orbit_points", 5000))
        changed_ep, ephem_pts = _field(imgui.slider_int, "Ephemeris Points", ephem_pts, 500, 20000)
        if changed_ep:
            app.camera["ephem_orbit_points"] = ephem_pts
            settings_changed = True
            app._ephem_trajectories_loaded = False
        if imgui.is_item_hovered():
            imgui.set_tooltip("Total sampling points along the trajectory polyline.\nHigher values yield smoother curves around periapsis.")

        traj_mode = int(app.camera.get("ephem_orbit_mode", 0))
        changed_tm, traj_mode = _field(imgui.combo, "Trajectory Display", traj_mode, ["Sliding Window (Fade)", "Full Mission Polyline"])
        if changed_tm:
            app.camera["ephem_orbit_mode"] = traj_mode
            settings_changed = True

        if traj_mode == 0:
            trail_days = float(app.camera.get("ephem_trail_days", 30.0))
            changed_td, trail_days = _field(imgui.slider_float, "Trail Duration", trail_days, 1.0, 365.0, "%.1f days")
            if changed_td:
                app.camera["ephem_trail_days"] = trail_days
                settings_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("How many days into the past the spacecraft trail extends with a smooth fade.")

            lead_days = float(app.camera.get("ephem_lead_days", 10.0))
            changed_ld, lead_days = _field(imgui.slider_float, "Lead Duration", lead_days, 0.0, 90.0, "%.1f days")
            if changed_ld:
                app.camera["ephem_lead_days"] = lead_days
                settings_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("How many days into the future the predicted trajectory extends.")

        changed_ho, app.camera["ephem_hide_outside"] = imgui.checkbox("Hide Outside Mission Timeline", app.camera.get("ephem_hide_outside", True))
        if changed_ho:
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Automatically hides the spacecraft orbit when the current simulation date is before launch or after mission end.")

    # Habitable Zones
    changed_hz, app.camera["show_habitable_zone"] = imgui.checkbox("Show Habitable Zones", app.camera.get("show_habitable_zone", False))
    if changed_hz: settings_changed = True


    return settings_changed


def _lighting(app):
    settings_changed = False
    # Atmosphere & Refraction
    changed_atmo, app.camera["atmo_enabled"] = imgui.checkbox("Enable Atmosphere Rendering", app.camera.get("atmo_enabled", True))
    if changed_atmo: settings_changed = True

    changed_clouds, app.camera["clouds_enabled"] = imgui.checkbox("Enable Dynamic Cloud Layers", app.camera.get("clouds_enabled", True))
    if changed_clouds: settings_changed = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Toggle dynamic planetary cloud layers and quadtree cloud shells on/off across all bodies.")

    if app.camera.get("clouds_enabled", True):
        imgui.indent()
        c_max_d = int(app.camera.get("cloud_max_depth", 3))
        changed_cmd, c_max_d = _field(imgui.slider_int, "Cloud Max LOD Depth", c_max_d, 0, 5)
        if changed_cmd:
            app.camera["cloud_max_depth"] = c_max_d
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Maximum quadtree LOD depth for planetary cloud shells (default 3).\nDecoupled from terrain: prevents high-LOD over-tessellation at ground level.")

        c_split_f = float(app.camera.get("cloud_lod_split_factor", 1.0))
        changed_csf, c_split_f = _field(imgui.slider_float, "Cloud LOD Sensitivity", c_split_f, 0.5, 3.0, "%.2fx")
        if changed_csf:
            app.camera["cloud_lod_split_factor"] = c_split_f
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("LOD subdivision distance threshold multiplier for planetary clouds.")
        imgui.unindent()

    changed_refr, app.camera["refraction_enabled"] = imgui.checkbox("Enable Atmospheric Refraction", app.camera.get("refraction_enabled", True))
    if changed_refr: settings_changed = True

    # Gravitational Lensing
    changed_lens, app.camera["grav_lensing_enabled"] = imgui.checkbox("Enable Gravitational Lensing", app.camera.get("grav_lensing_enabled", True))
    if changed_lens: settings_changed = True
    if app.camera.get("grav_lensing_enabled", True):
        imgui.indent()
        lens_mult = float(app.camera.get("grav_lensing_multiplier", 1.0))
        changed_lensm, lens_mult = _field(imgui.slider_float, "Lensing Strength", lens_mult, 0.0, 5.0, "%.2fx")
        if changed_lensm:
            app.camera["grav_lensing_multiplier"] = lens_mult
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Deflection multiplier for the dominant nearby lens (1.0 = GR).\nBlack-hole shadows, Einstein arcs and star deflection are always physical.")
        imgui.unindent()

    changed_ps, app.camera["planetshine_enabled"] = imgui.checkbox("Enable Planetshine/Moonshine", app.camera.get("planetshine_enabled", True))
    if changed_ps: settings_changed = True

    if app.camera.get("planetshine_enabled", True):
        imgui.indent()
        ring_ps_modes = ["Analytical (Fast)", "Numerical (64 Samples)"]
        curr_rm = app.camera.get("ring_planetshine_mode", 0)
        curr_rm_idx = curr_rm if 0 <= curr_rm < len(ring_ps_modes) else 0
        changed_rm, new_rm = _field(imgui.combo, "Ring Planetshine Mode", curr_rm_idx, ring_ps_modes)
        if changed_rm:
            app.camera["ring_planetshine_mode"] = new_rm
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Analytical: Fast O(1) closed-form Lambertian phase model (high FPS).\nNumerical: 64-sample ray-traced disk quadrature (heavy GPU cost).")
        imgui.unindent()

    changed_rs, app.camera["ringshine_enabled"] = imgui.checkbox("Enable Ringshine", app.camera.get("ringshine_enabled", True))
    if changed_rs: settings_changed = True

    if app.camera.get("ringshine_enabled", True):
        imgui.indent()
        rs_modes = ["Precomputed Map (Cached)", "Per-Pixel Monte Carlo (Ground Truth)"]
        curr_rsm = app.camera.get("ringshine_mode", 0)
        curr_rsm_idx = curr_rsm if 0 <= curr_rsm < len(rs_modes) else 0
        changed_rsm, new_rsm = _field(imgui.combo, "Ringshine Method", curr_rsm_idx, rs_modes)
        if changed_rsm:
            app.camera["ringshine_mode"] = new_rsm
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Precomputed Map: Cached 128x65 Gauss-Legendre transfer map (high performance).\nPer-Pixel Monte Carlo: Real-time per-pixel stochastic ray tracing (ground truth reference).")

        if app.camera.get("ringshine_mode", 0) == 1:
            mc_samples = int(app.camera.get("ringshine_mc_samples", 64))
            changed_mcs, mc_samples = _field(imgui.slider_int, "MC Samples/Pixel", mc_samples, 16, 4096)
            if changed_mcs:
                app.camera["ringshine_mc_samples"] = mc_samples
                settings_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Number of ray samples cast per surface pixel (16 to 4096).\nHigher values reduce grain/variance at the cost of framerate.")

            imgui.text("Presets:")
            imgui.same_line()
            for preset in [64, 128, 256, 512, 1024, 2048, 4096]:
                if imgui.small_button(f"{preset}##preset_mc"):
                    app.camera["ringshine_mc_samples"] = preset
                    settings_changed = True
                imgui.same_line()
            imgui.new_line()

            changed_dith, app.camera["ringshine_mc_dither"] = imgui.checkbox("Stochastic Dither (Noise)", app.camera.get("ringshine_mc_dither", False))
            if changed_dith:
                settings_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Disabled (default): Deterministic Quasi-Monte Carlo produces perfectly smooth, grain-free shading.\nEnabled: Adds per-pixel screen-space dithering, useful for temporal Pause/Screenshot Accumulation.")
        else:
            ringshine_bands = int(app.camera.get("ringshine_band_count", 10))
            changed_rsb, ringshine_bands = _field(imgui.slider_int, "Ringshine Bands", ringshine_bands, 4, 1024)
            if changed_rsb:
                app.camera["ringshine_band_count"] = ringshine_bands
                settings_changed = True
        changed_rso, app.camera["ringshine_oblate_enabled"] = imgui.checkbox("Account for Host Oblateness", app.camera.get("ringshine_oblate_enabled", True))
        if changed_rso:
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Physically models planetary polar flattening on shadow length, surface slant distance, and ring form factors.")
        imgui.unindent()


    return settings_changed


def _terrain(app):
    settings_changed = False
    # Dynamic Texture Streaming
    stream_thresh = float(app.camera.get("tex_stream_threshold_px", 500.0))
    changed_st, stream_thresh = _field(imgui.slider_float, "Min High-Res Size (px)", stream_thresh, 100.0, 2000.0, "%.0f px")
    if changed_st:
        app.camera["tex_stream_threshold_px"] = stream_thresh
        settings_changed = True

    # Planetary Terrain & Surface LOD (SpaceEngine Style)
    imgui.separator()
    imgui.text_colored("Planetary Terrain & Surface LOD (SpaceEngine Style)", 0.4, 0.8, 1.0)

    changed_tlod, app.camera["terrain_lod_enabled"] = imgui.checkbox("Enable Terrain Quadtree LOD", app.camera.get("terrain_lod_enabled", False))
    if changed_tlod: settings_changed = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Toggle between legacy monolithic sphere rendering and SpaceEngine-style\nSpherified Cube Quadtree LOD with tiled virtual texture streaming.")

    if app.camera.get("terrain_lod_enabled", False):
        imgui.indent()
        changed_dt, app.camera["terrain_debug_tiles"] = imgui.checkbox("Debug Terrain Tiles", app.camera.get("terrain_debug_tiles", False))
        if changed_dt: settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Highlight active quadtree patch borders and false-color each patch by its LOD level.")

        lod_dist = float(app.camera.get("terrain_lod_distance", 1.0))
        changed_ld, lod_dist = _field(imgui.slider_float, "LOD Distance Scale", lod_dist, 0.2, 3.0, "%.2fx")
        if changed_ld:
            app.camera["terrain_lod_distance"] = lod_dist
            app.camera["terrain_lod_split_factor"] = 1.0 / max(0.1, lod_dist)
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Overall distance scale for terrain LOD subdivision.\nHigher values extend high-detail LOD patches farther toward the horizon.\nLower values keep high-detail patches closer to the camera for higher FPS.")

        ground_dist = float(app.camera.get("terrain_ground_lod_distance", 1.0))
        changed_gd, ground_dist = _field(imgui.slider_float, "Ground LOD Closeness", ground_dist, 0.2, 3.0, "%.2fx")
        if changed_gd:
            app.camera["terrain_ground_lod_distance"] = ground_dist
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Near-ground detail multiplier, relative to overall LOD distance.\nValues below 1.0x add detail near the sampled surface.\nValues above 1.0x reduce near-ground detail. Camera rotation does not change refinement.")

        max_depth = int(app.camera.get("terrain_max_depth", 10))
        changed_md, max_depth = _field(imgui.slider_int, "Max LOD Depth", max_depth, 0, 20)
        if changed_md:
            app.camera["terrain_max_depth"] = max_depth
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Maximum quadtree subdivision depth level (0 to 20).\nGoverns the finest mesh resolution available near the camera.")

        res_options = [8, 16, 24, 32, 48, 64]
        res_labels = [
            "8x8 (192 tris/patch - Low/Fast)",
            "16x16 (640 tris/patch - Medium)",
            "24x24 (1,344 tris/patch - Balanced)",
            "32x32 (2,304 tris/patch - High/Default)",
            "48x48 (4,992 tris/patch - Very High)",
            "64x64 (8,704 tris/patch - Ultra/Dense)"
        ]
        current_res = int(app.camera.get("terrain_patch_res", 32))
        current_idx = res_options.index(current_res) if current_res in res_options else 3
        changed_res, new_idx = _field(imgui.combo, "Patch Grid Resolution", current_idx, res_labels)
        if changed_res:
            sel_res = res_options[new_idx]
            if hasattr(app, "rebuild_terrain_grid_patch"):
                app.rebuild_terrain_grid_patch(sel_res)
            else:
                app.camera["terrain_patch_res"] = sel_res
            settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Set the vertex grid density and triangle count for each quadtree patch.\nLower resolution dramatically increases framerate when many patches are visible.\nHigher resolution gives smoother planet curvature and elevation detail.")

        changed_stc, app.camera["show_triangle_count"] = imgui.checkbox("Show Triangle Count (HUD Overlay)", app.camera.get("show_triangle_count", False))
        if changed_stc: settings_changed = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Display live on-screen viewport HUD overlay with rendered triangle counts for terrain, celestial bodies, and current FPS.")

        if getattr(app, 'terrain_streamer', None) is not None:
            res_cnt = app.terrain_streamer.resident_count
            cap = app.terrain_streamer.pool_capacity
            p_cnt = getattr(app, 'terrain_last_patch_count', 0)
            cp_cnt = getattr(app, 'terrain_last_cloud_patch_count', 0)
            t_cnt = getattr(app, 'terrain_last_triangle_count', 0)
            if cp_cnt > 0:
                imgui.text_disabled(f"Resident Tiles: {res_cnt}/{cap} ({(res_cnt/cap)*100:.0f}%) | Ground: {p_cnt:,} p | Clouds: {cp_cnt:,} p | Tris: {t_cnt:,}")
            else:
                imgui.text_disabled(f"Resident Tiles: {res_cnt}/{cap} ({(res_cnt/cap)*100:.0f}%) | Patches: {p_cnt:,} | Tris: {t_cnt:,}")
        imgui.unindent()


    return settings_changed


def _workspace(app):
    settings_changed = False
    imgui.separator()
    imgui.text_colored("System Comparison Settings", 0.6, 0.9, 1.0)
    changed_fk, force_kep = imgui.checkbox("Force Keplerian Mode on Comparison", app.camera.get("comparison_force_keplerian", True))
    if changed_fk:
        app.camera["comparison_force_keplerian"] = force_kep
        settings_changed = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Automatically switch simulation to analytical Keplerian mode when enabling system comparison.")


    imgui.separator()
    heading("Workspace layout")
    changed, aligned = imgui.checkbox("Align side panels", app.camera.get("ui_aligned_panels", True))
    if changed:
        app.camera["ui_aligned_panels"] = aligned
        settings_changed = True
    imgui.text_wrapped("Aligned panels reserve space for navigation and time controls. Turn this off to move and resize panels freely.")
    for label, key, default, lo, hi in (
        ("Outliner width", "ui_outliner_width", 260, 220, 360),
        ("Inspector width", "ui_inspector_width", 420, 350, 560),
    ):
        changed, value = _field(imgui.slider_int, label, int(app.camera.get(key, default)), lo, hi)
        if changed:
            app.camera[key] = value
            settings_changed = True
    changed, scale = _field(imgui.slider_float, "Interface scale", app.camera.get("ui_scale", 1.0), 0.9, 1.5, "%.2fx")
    if changed:
        app.camera["ui_scale"] = scale
        settings_changed = True
    if imgui.button("Reset workspace layout"):
        reset_workspace(app)

    return settings_changed


_RENDERERS = (_atmosphere, _optics, _visibility, _lighting, _terrain, _workspace)
_DESCRIPTIONS = (
    "Scattering quality, sampling and temporal integration.",
    "Exposure, bloom, diffraction spikes and anti-aliasing.",
    "Orbital paths, mission trajectories and habitable zones.",
    "Clouds, refraction and light exchanged between bodies.",
    "Surface detail, texture streaming and geometry diagnostics.",
    "Panel layout, interface scale and comparison behavior.",
)


def render_settings(app):
    if not app.camera.get("show_settings_modal", False):
        return
    layout = workspace(app)
    width = min(960 * layout.scale, layout.width - layout.gap * 4)
    height = min(650 * layout.scale, layout.height - layout.top - layout.gap * 3)
    signature = (layout.width, layout.height, layout.scale)
    first = getattr(app, "_ui_settings_geometry", None) != signature
    app._ui_settings_geometry = signature
    cond = imgui.ALWAYS if first else imgui.FIRST_USE_EVER
    imgui.set_next_window_size(width, height, cond)
    imgui.set_next_window_position((layout.width - width) / 2, (layout.height - height) / 2, cond)
    expanded, opened = imgui.begin("Settings / Stellar Forge###forge_settings", True, imgui.WINDOW_NO_COLLAPSE)
    app.camera["show_settings_modal"] = opened
    if expanded:
        section = max(0, min(len(SETTINGS_SECTIONS) - 1, getattr(app, "_ui_settings_section", 0)))
        imgui.begin_child("settings_navigation", min(150 * layout.scale, width * 0.23), -40 * layout.scale, border=True)
        imgui.text_disabled("SETTINGS")
        imgui.spacing()
        for index, label in enumerate(SETTINGS_SECTIONS):
            if imgui.selectable(label, section == index, height=28 * layout.scale)[0]:
                section = index
                app._ui_settings_section = index
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("settings_section_" + str(section), 0, -40 * layout.scale, border=True)
        heading(SETTINGS_SECTIONS[section], _DESCRIPTIONS[section])
        imgui.push_item_width(max(90, imgui.get_content_region_available()[0] * 0.43))
        if _RENDERERS[section](app):
            app.save_settings()
        imgui.pop_item_width()
        imgui.end_child()
        imgui.separator()
        imgui.text_disabled("Changes save automatically")
        imgui.same_line()
        if imgui.button("Close", width=85 * layout.scale):
            app.camera["show_settings_modal"] = False
    imgui.end()
