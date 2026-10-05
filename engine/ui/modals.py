import math
import random
import datetime
import os
import numpy as np
import imgui
from engine.core.constants import DEFAULT_LY_THRESHOLD_AU, SKY_VIEW_RES_LABELS
from engine.core.math_utils import get_cartesian_from_keplerian, rotate_equatorial_to_ecliptic, ecliptic_to_pole
from engine.rendering.render_utils import compute_surface_albedo, format_distance_au, sim_time_from_date, format_sim_time_utc
from engine.ephemeris.system_manager import SystemManager

def render_modals(app, bodies_data, visual_data, atmo_bodies, ring_bodies, star_idx, num_bodies, mass_snap, pos_snap_render, vel_snap_render, visual_arr, cur_y, cur_m, cur_d, display_t, switch_triggers, cur_h=0, cur_mn=0, cur_s=0, cur_tz="UTC"):
    """Render all popup dialogs and modal windows."""
    load_system_from_data = switch_triggers["load_system"]
    trigger_ephem_switch = switch_triggers["ephem_switch"]
    trigger_system_switch = switch_triggers["switch_system"]

    # ── 1. Graphics & Quality Settings Modal ──
    if app.camera.get("show_settings_modal", False):
        imgui.set_next_window_size(380, 520, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(app.fb_width // 2 - 190, app.fb_height // 2 - 260, imgui.FIRST_USE_EVER)
        expanded, app.camera["show_settings_modal"] = imgui.begin("Graphics & Quality Settings", True)
        if expanded:
            settings_changed = False

            # Atmosphere Quality
            atmo_quality = app.camera.get("atmo_quality", 1)
            changed_aq, atmo_quality = imgui.combo("Atmosphere Quality", atmo_quality, ["Off", "Low (2D Shadows)", "High (Volumetric)", "Analytical (Sky-View + Slicing)"])
            if changed_aq:
                app.camera["atmo_quality"] = min(3, atmo_quality)
                settings_changed = True

            if atmo_quality == 3:
                sky_view_res = int(app.camera.get("atmo_sky_view_res", 0))
                sky_view_res_idx = min(len(SKY_VIEW_RES_LABELS) - 1, max(0, sky_view_res))
                changed_svr, new_svr_idx = imgui.combo("Sky-View LUT Resolution", sky_view_res_idx, SKY_VIEW_RES_LABELS)
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

                changed_steps, shadow_steps = imgui.slider_int(
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
                    changed_max, max_steps = imgui.slider_int(
                        "Max Adaptive Shadow Steps", app.camera.get("atmo_adaptive_steps_max", 128), 8, 256)
                    if changed_max:
                        app.camera["atmo_adaptive_steps_max"] = max_steps
                        settings_changed = True

            elif atmo_quality > 0:
                atmo_res = float(app.camera.get("atmo_resolution", 1.0))
                changed_res, atmo_res = imgui.slider_float("Atmosphere Render Scale", atmo_res, 0.2, 1.0, "%.2fx")
                if changed_res:
                    app.camera["atmo_resolution"] = atmo_res
                    settings_changed = True

                if atmo_res < 0.999:
                    vrs_threshold = float(app.camera.get("atmo_vrs_threshold_px", 100.0))
                    changed_vrs, vrs_threshold = imgui.slider_float("VRS Low-Res Threshold (px)", vrs_threshold, 10.0, 1000.0, "%.1f")
                    if changed_vrs:
                        app.camera["atmo_vrs_threshold_px"] = vrs_threshold
                        settings_changed = True

                max_steps = app.camera.get("atmo_steps_max", 32)
                changed_steps, max_steps = imgui.slider_int("Max Ray Steps", max_steps, 4, 128)
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
                    changed_max_adapt, max_adapt_steps = imgui.slider_int("Max Adaptive Steps", max_adapt_steps, 8, 256)
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
                    changed_nt, cur_noise = imgui.combo("Noise Type##mode2", cur_noise, noise_types)
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
                    changed_tb, temporal_blend = imgui.slider_float("Temporal History Weight", temporal_blend, 0.50, 0.98, "%.2f")
                    if changed_tb:
                        app.camera["atmo_temporal_blend"] = temporal_blend
                        settings_changed = True

            # Shadow Caster Budget
            caster_budget = app.camera.get("shadow_caster_budget", 32)
            changed_budget, caster_budget = imgui.slider_int("Shadow Caster Budget", caster_budget, 4, 64)
            if changed_budget:
                app.camera["shadow_caster_budget"] = caster_budget
                settings_changed = True

            imgui.separator()

            # Exposure & HDR
            changed_hdr, app.camera["hdr_enabled"] = imgui.checkbox("HDR Mode", app.camera.get("hdr_enabled", True))
            if changed_hdr: settings_changed = True

            if app.camera.get("hdr_enabled", True):
                changed_exp, app.camera["exposure"] = imgui.slider_float("Exposure", app.camera.get("exposure", 1.0), 0.0001, 10000.0, "%.4f", imgui.SLIDER_FLAGS_LOGARITHMIC)
                if changed_exp: settings_changed = True

            # Bloom & Diffraction Spikes
            bloom_modes = ["Gaussian Blur", "Diffraction Spikes", "Hybrid (Spikes + Haze)"]
            current_bm = app.camera.get("bloom_mode", 2)
            current_bm_idx = current_bm if 0 <= current_bm < len(bloom_modes) else 2
            changed_bm, new_bm = imgui.combo("Bloom Mode", current_bm_idx, bloom_modes)
            if changed_bm:
                app.camera["bloom_mode"] = new_bm
                settings_changed = True

            if app.camera.get("bloom_mode", 2) in (1, 2):
                spike_counts = [4, 6, 8]
                spike_labels = ["4 Spikes (Cross)", "6 Spikes (JWST / Newtonian)", "8 Spikes (Octagram)"]
                curr_sc = app.camera.get("spike_count", 6)
                sc_idx = spike_counts.index(curr_sc) if curr_sc in spike_counts else 1
                changed_sc, new_sc_idx = imgui.combo("Spike Pattern", sc_idx, spike_labels)
                if changed_sc:
                    app.camera["spike_count"] = spike_counts[new_sc_idx]
                    settings_changed = True

                changed_cbi, app.camera["conv_bloom_intensity"] = imgui.slider_float("Diffraction Spikes Intensity", app.camera.get("conv_bloom_intensity", 0.5), 0.0, 5.0, "%.2f")
                if changed_cbi: settings_changed = True
                changed_sl, app.camera["spike_length"] = imgui.slider_float("Diffraction Spikes Length", app.camera.get("spike_length", 1.0), 0.2, 3.0, "%.2f")
                if changed_sl: settings_changed = True
                changed_sa, app.camera["spike_angle"] = imgui.slider_float("Diffraction Spikes Angle", app.camera.get("spike_angle", 0.0), 0.0, 180.0, "%.1f deg")
                if changed_sa: settings_changed = True
                changed_srl, app.camera["spike_roll_lock"] = imgui.checkbox("Lock Spikes to Camera Roll", app.camera.get("spike_roll_lock", True))
                if changed_srl: settings_changed = True
                changed_sd, app.camera["spike_dispersion"] = imgui.slider_float("Dispersion (Rainbow)", app.camera.get("spike_dispersion", 0.015), 0.0, 0.05, "%.3f")
                if changed_sd: settings_changed = True

                spike_res_opts = [0, 1]
                spike_res_labels = ["Quarter Res (Fast - 0.1ms)", "Half Res (Ultra)"]
                curr_sq = app.camera.get("spike_quality", 0)
                sq_idx = curr_sq if 0 <= curr_sq < len(spike_res_opts) else 0
                changed_sq, new_sq_idx = imgui.combo("Spikes Resolution", sq_idx, spike_res_labels)
                if changed_sq:
                    app.camera["spike_quality"] = spike_res_opts[new_sq_idx]
                    app.last_fb_size = (0, 0)
                    settings_changed = True

                changed_cds, app.camera["conv_bloom_dynamic_scale"] = imgui.checkbox("Dynamic Spike Distance Shrinking", app.camera.get("conv_bloom_dynamic_scale", True))
                if changed_cds: settings_changed = True

            if app.camera.get("bloom_mode", 2) in (0, 2):
                changed_bi, app.camera["bloom_intensity"] = imgui.slider_float("Gaussian Haze Intensity", app.camera.get("bloom_intensity", 0.05), 0.0, 1.0, "%.3f")
                if changed_bi: settings_changed = True
                changed_bt, app.camera["bloom_threshold"] = imgui.slider_float("Gaussian Haze Threshold", app.camera.get("bloom_threshold", 1.0), 0.0, 10.0, "%.2f")
                if changed_bt: settings_changed = True

            # MSAA
            msaa_options = [0, 2, 4, 8]
            msaa_labels = ["Off", "2x", "4x", "8x"]
            current_msaa = app.camera.get("msaa_samples", 4)
            current_idx = msaa_options.index(current_msaa) if current_msaa in msaa_options else 2
            changed_msaa, new_msaa_idx = imgui.combo("Orbit MSAA", current_idx, msaa_labels)
            if changed_msaa:
                app.camera["msaa_samples"] = msaa_options[new_msaa_idx]
                settings_changed = True

            # Orbits & Fade
            changed_so, app.camera["show_orbits"] = imgui.checkbox("Show Orbits", app.camera.get("show_orbits", True))
            if changed_so: settings_changed = True

            if app.camera.get("show_orbits", True):
                orbit_fade_dir_idx = app.camera.get("orbit_fade_dir_idx", 0)
                changed_ofd, orbit_fade_dir_idx = imgui.combo("Orbit Fade", orbit_fade_dir_idx, ["Bright Behind", "Bright Ahead"])
                if changed_ofd:
                    app.camera["orbit_fade_dir_idx"] = orbit_fade_dir_idx
                    settings_changed = True

                orbit_min_alpha = app.camera.get("orbit_min_alpha", 0.1)
                changed_oma, orbit_min_alpha = imgui.slider_float("Min Orbit Alpha", orbit_min_alpha, 0.0, 1.0, "%.2f")
                if changed_oma:
                    app.camera["orbit_min_alpha"] = orbit_min_alpha
                    settings_changed = True

                imgui.spacing()
                imgui.text_colored("Spacecraft / Ephemeris Orbits", 0.4, 0.8, 1.0)

                ephem_pts = int(app.camera.get("ephem_orbit_points", 5000))
                changed_ep, ephem_pts = imgui.slider_int("Ephemeris Points", ephem_pts, 500, 20000)
                if changed_ep:
                    app.camera["ephem_orbit_points"] = ephem_pts
                    settings_changed = True
                    app._ephem_trajectories_loaded = False
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Total sampling points along the trajectory polyline.\nHigher values yield smoother curves around periapsis.")

                traj_mode = int(app.camera.get("ephem_orbit_mode", 0))
                changed_tm, traj_mode = imgui.combo("Trajectory Display", traj_mode, ["Sliding Window (Fade)", "Full Mission Polyline"])
                if changed_tm:
                    app.camera["ephem_orbit_mode"] = traj_mode
                    settings_changed = True

                if traj_mode == 0:
                    trail_days = float(app.camera.get("ephem_trail_days", 30.0))
                    changed_td, trail_days = imgui.slider_float("Trail Duration", trail_days, 1.0, 365.0, "%.1f days")
                    if changed_td:
                        app.camera["ephem_trail_days"] = trail_days
                        settings_changed = True
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("How many days into the past the spacecraft trail extends with a smooth fade.")

                    lead_days = float(app.camera.get("ephem_lead_days", 10.0))
                    changed_ld, lead_days = imgui.slider_float("Lead Duration", lead_days, 0.0, 90.0, "%.1f days")
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
                changed_cmd, c_max_d = imgui.slider_int("Cloud Max LOD Depth", c_max_d, 0, 5)
                if changed_cmd:
                    app.camera["cloud_max_depth"] = c_max_d
                    settings_changed = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Maximum quadtree LOD depth for planetary cloud shells (default 3).\nDecoupled from terrain: prevents high-LOD over-tessellation at ground level.")

                c_split_f = float(app.camera.get("cloud_lod_split_factor", 1.0))
                changed_csf, c_split_f = imgui.slider_float("Cloud LOD Sensitivity", c_split_f, 0.5, 3.0, "%.2fx")
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
                changed_lensm, lens_mult = imgui.slider_float("Lensing Strength", lens_mult, 0.0, 5.0, "%.2fx")
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
                changed_rm, new_rm = imgui.combo("Ring Planetshine Mode", curr_rm_idx, ring_ps_modes)
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
                changed_rsm, new_rsm = imgui.combo("Ringshine Method", curr_rsm_idx, rs_modes)
                if changed_rsm:
                    app.camera["ringshine_mode"] = new_rsm
                    settings_changed = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Precomputed Map: Cached 128x65 Gauss-Legendre transfer map (high performance).\nPer-Pixel Monte Carlo: Real-time per-pixel stochastic ray tracing (ground truth reference).")

                if app.camera.get("ringshine_mode", 0) == 1:
                    mc_samples = int(app.camera.get("ringshine_mc_samples", 64))
                    changed_mcs, mc_samples = imgui.slider_int("MC Samples/Pixel", mc_samples, 16, 4096)
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
                    changed_rsb, ringshine_bands = imgui.slider_int("Ringshine Bands", ringshine_bands, 4, 1024)
                    if changed_rsb:
                        app.camera["ringshine_band_count"] = ringshine_bands
                        settings_changed = True
                changed_rso, app.camera["ringshine_oblate_enabled"] = imgui.checkbox("Account for Host Oblateness", app.camera.get("ringshine_oblate_enabled", True))
                if changed_rso:
                    settings_changed = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Physically models planetary polar flattening on shadow length, surface slant distance, and ring form factors.")
                imgui.unindent()

            # Dynamic Texture Streaming
            stream_thresh = float(app.camera.get("tex_stream_threshold_px", 500.0))
            changed_st, stream_thresh = imgui.slider_float("Min High-Res Size (px)", stream_thresh, 100.0, 2000.0, "%.0f px")
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
                changed_ld, lod_dist = imgui.slider_float("LOD Distance Scale", lod_dist, 0.2, 3.0, "%.2fx")
                if changed_ld:
                    app.camera["terrain_lod_distance"] = lod_dist
                    app.camera["terrain_lod_split_factor"] = 1.0 / max(0.1, lod_dist)
                    settings_changed = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Overall distance scale for terrain LOD subdivision.\nHigher values extend high-detail LOD patches farther toward the horizon.\nLower values keep high-detail patches closer to the camera for higher FPS.")

                ground_dist = float(app.camera.get("terrain_ground_lod_distance", 1.0))
                changed_gd, ground_dist = imgui.slider_float("Ground LOD Closeness", ground_dist, 0.2, 3.0, "%.2fx")
                if changed_gd:
                    app.camera["terrain_ground_lod_distance"] = ground_dist
                    settings_changed = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Near-ground LOD transition distance.\nValues below 1.0x bring higher quality LOD patches closer to the camera\nwhen near the surface, concentrating dense geometry nearby while keeping distant terrain lightweight.")

                max_depth = int(app.camera.get("terrain_max_depth", 10))
                changed_md, max_depth = imgui.slider_int("Max LOD Depth", max_depth, 0, 10)
                if changed_md:
                    app.camera["terrain_max_depth"] = max_depth
                    settings_changed = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Maximum quadtree subdivision depth level (0 to 10).\nGoverns the finest mesh resolution available near the camera.")

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
                changed_res, new_idx = imgui.combo("Patch Grid Resolution", current_idx, res_labels)
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
                    res_cnt = len(app.terrain_streamer.resident_tiles)
                    cap = app.terrain_streamer.pool_capacity
                    p_cnt = getattr(app, 'terrain_last_patch_count', 0)
                    cp_cnt = getattr(app, 'terrain_last_cloud_patch_count', 0)
                    t_cnt = getattr(app, 'terrain_last_triangle_count', 0)
                    if cp_cnt > 0:
                        imgui.text_disabled(f"Resident Tiles: {res_cnt}/{cap} ({(res_cnt/cap)*100:.0f}%) | Ground: {p_cnt:,} p | Clouds: {cp_cnt:,} p | Tris: {t_cnt:,}")
                    else:
                        imgui.text_disabled(f"Resident Tiles: {res_cnt}/{cap} ({(res_cnt/cap)*100:.0f}%) | Patches: {p_cnt:,} | Tris: {t_cnt:,}")
                imgui.unindent()

            imgui.separator()
            imgui.text_colored("System Comparison Settings", 0.6, 0.9, 1.0)
            changed_fk, force_kep = imgui.checkbox("Force Keplerian Mode on Comparison", app.camera.get("comparison_force_keplerian", True))
            if changed_fk:
                app.camera["comparison_force_keplerian"] = force_kep
                settings_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Automatically switch simulation to analytical Keplerian mode when enabling system comparison.")

            if settings_changed:
                app.save_settings()

            imgui.separator()
            if imgui.button("Close", width=-1):
                app.camera["show_settings_modal"] = False
        imgui.end()

    # ── 2. Add Orbiting Body Modal ──
    if app.camera.get("add_mode", False):
        imgui.set_next_window_size(380, 580, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(app.fb_width // 2 - 190, app.fb_height // 2 - 290, imgui.FIRST_USE_EVER)
        expanded, app.camera["add_mode"] = imgui.begin("Add Orbiting Body", True)
        if expanded:
            ad = app.camera["add_data"]
            is_moon = ad.get("is_moon", False)
            _, ad["name"] = imgui.input_text("Name", ad["name"], 256)

            types_list = ["Star", "Terrestrial", "Gas Giant", "Ice Giant", "Dwarf Planet", "Moon"]
            if "type" not in ad:
                ad["type"] = "Moon" if is_moon else "Terrestrial"
            type_idx = types_list.index(ad["type"]) if ad["type"] in types_list else 1
            changed_t, type_idx = imgui.combo("Type", type_idx, types_list)
            if changed_t:
                ad["type"] = types_list[type_idx]

            _, ad["color"] = imgui.color_edit3("Color", *ad["color"])

            if is_moon:
                _, ad["mass"] = imgui.input_double("Mass (M☾)", ad["mass"], format="%.4f")
            else:
                _, ad["mass"] = imgui.input_double("Mass (M⊕)", ad["mass"], format="%.4f")

            _, ad["radius"] = imgui.input_double("Radius (km)", ad["radius"], format="%.1f")

            # ── Rotational & Oblateness Properties ──
            imgui.separator()
            imgui.text_colored("Rotational Properties", 1.0, 0.85, 0.4)

            insp_idx = ad.get("parent_idx", 0)
            if insp_idx >= num_bodies:
                insp_idx = 0
            parent_m = mass_snap[insp_idx]

            real_mass = ad["mass"] * 3.694e-8 if is_moon else ad["mass"] * 3.003e-6
            real_a = ad["a"] / 149597870.7 if is_moon else ad["a"]

            is_locked = ad.get("tidally_locked", False)
            chg_lock, is_locked = imgui.checkbox("Tidally Locked", is_locked)
            if chg_lock:
                ad["tidally_locked"] = is_locked

            if is_locked:
                tot_m = parent_m + real_mass
                p_years = math.sqrt(real_a**3 / tot_m) if tot_m > 0 and real_a > 0 else 0.0
                ad["rotation_period"] = p_years * 365.25 * 24.0
                imgui.text(f"  Rot Period:     {ad['rotation_period']:.4f} h [Synchronous]")
            else:
                cur_rot = ad.get("rotation_period", 24.0)
                _, cur_rot = imgui.input_double("Rot Period (hours)", cur_rot, format="%.4f")
                ad["rotation_period"] = max(0.0, cur_rot)

            cur_tilt = ad.get("axial_tilt", 0.0)
            _, cur_tilt = imgui.input_double("Axial Tilt (deg)", cur_tilt, format="%.2f")
            ad["axial_tilt"] = cur_tilt

            # Live calculation of density and oblateness based on mass, radius, and rotation
            r_km = max(0.1, float(ad.get("radius", 6371.0)))
            rot_h = float(ad.get("rotation_period", 24.0))
            b_type = ad.get("type", "Terrestrial")

            r_m = r_km * 1000.0
            mass_kg = real_mass * 1.98847e30
            vol_m3 = (4.0 / 3.0) * math.pi * (r_m ** 3)
            density_g_cm3 = (mass_kg / vol_m3) / 1000.0 if vol_m3 > 0 else 0.0

            f_calc = 0.0
            j2_calc = 0.0
            j4_calc = 0.0
            if rot_h > 0.0 and real_mass > 0.0 and r_km > 0.0:
                G_SI = 6.6743e-11
                omega = 2.0 * math.pi / (rot_h * 3600.0)
                m_param = (omega**2 * r_m**3) / (G_SI * mass_kg)
                if b_type in ("Moon", "Dwarf Planet"):
                    chi = 1.25
                elif b_type == "Terrestrial":
                    chi = 0.95
                else:
                    chi = 0.65
                f_calc = float(np.clip(chi * m_param, 0.0, 0.5))
                j2_calc = float(m_param * (chi - 0.5))
                j4_calc = float(-0.15 * (f_calc**2))

            ad["oblateness"] = f_calc
            ad["J2"] = j2_calc
            ad["j4"] = j4_calc

            imgui.text(f"  Mean Density:   {density_g_cm3:.2f} g/cm³")
            imgui.text(f"  Oblateness (f): {f_calc:.5f}")
            if j2_calc > 0.0:
                imgui.text(f"  J2 Grav Harm:   {j2_calc:.6f}")

            # ── Orbital Elements ──
            imgui.separator()
            imgui.text_colored("Orbital Elements", 1.0, 0.85, 0.4)
            imgui.same_line(spacing=15)
            if imgui.small_button("Randomize Angles##rnd_angles"):
                ad["Omega"] = round(random.uniform(0.0, 360.0), 3)
                ad["omega"] = round(random.uniform(0.0, 360.0), 3)
                ad["M"] = round(random.uniform(0.0, 360.0), 3)

            if "Omega" not in ad or ad["Omega"] is None:
                ad["Omega"] = round(random.uniform(0.0, 360.0), 3)
            if "omega" not in ad or ad["omega"] is None:
                ad["omega"] = round(random.uniform(0.0, 360.0), 3)
            if "M" not in ad or ad["M"] is None:
                ad["M"] = round(random.uniform(0.0, 360.0), 3)

            _, ad["frame"] = imgui.combo("Reference Frame", ad["frame"], ["Ecliptic", "Equatorial"])

            if is_moon:
                _, ad["a"] = imgui.input_double("Semi-Major Axis (km)", ad["a"], format="%.1f")
            else:
                _, ad["a"] = imgui.input_double("Semi-Major Axis (AU)", ad["a"], format="%.6f")

            _, ad["e"] = imgui.input_double("Eccentricity", ad["e"], format="%.6f")
            _, ad["inc"] = imgui.input_double("Inclination (deg)", ad["inc"], format="%.3f")
            _, ad["Omega"] = imgui.input_double("Long Asc Node (deg)", ad["Omega"], format="%.3f")
            _, ad["omega"] = imgui.input_double("Arg Periapsis (deg)", ad["omega"], format="%.3f")
            _, ad["M"] = imgui.input_double("Mean Anomaly (deg)", ad["M"], format="%.3f")

            imgui.separator()
            if imgui.button("Spawn Body", width=-1):
                insp_idx = ad.get("parent_idx", 0)
                if insp_idx >= num_bodies:
                    insp_idx = 0
                parent_m = mass_snap[insp_idx]
                p_pos = pos_snap_render[insp_idx]
                p_vel = vel_snap_render[insp_idx]
                ppx, ppy, ppz = p_pos[0], -p_pos[2], p_pos[1]
                pvx, pvy, pvz = p_vel[0], -p_vel[2], p_vel[1]

                real_mass = ad["mass"] * 3.694e-8 if is_moon else ad["mass"] * 3.003e-6
                real_a = ad["a"] / 149597870.7 if is_moon else ad["a"]

                c_pos, c_vel = get_cartesian_from_keplerian(
                    parent_m, real_mass, real_a, ad["e"],
                    ad["inc"], ad["Omega"], ad["omega"], ad["M"]
                )

                if ad["frame"] == 1:
                    pole_render_p = visual_arr[insp_idx, 5:8]
                    pole_ecl_p = np.array([pole_render_p[0], -pole_render_p[2], pole_render_p[1]])
                    c_pos, c_vel = rotate_equatorial_to_ecliptic(c_pos, c_vel, pole_ecl_p)

                # Compute orbit normal in ecliptic frame
                orb_h = np.cross(c_pos, c_vel)
                h_norm = np.linalg.norm(orb_h)
                if h_norm > 1e-12:
                    orb_n = orb_h / h_norm
                else:
                    orb_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                # Tilt spin axis relative to orbit normal by axial_tilt
                tilt_deg = float(ad.get("axial_tilt", 0.0))
                tilt_rad = math.radians(tilt_deg)
                if abs(tilt_rad) < 1e-6:
                    pole_ecl = orb_n
                else:
                    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    if abs(np.dot(orb_n, ref)) > 0.99:
                        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    perp = np.cross(orb_n, ref)
                    perp_n = np.linalg.norm(perp)
                    if perp_n > 1e-10:
                        perp /= perp_n
                    else:
                        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    pole_ecl = orb_n * math.cos(tilt_rad) + perp * math.sin(tilt_rad)
                    pole_norm = np.linalg.norm(pole_ecl)
                    if pole_norm > 0:
                        pole_ecl /= pole_norm

                pole_render = np.array([pole_ecl[0], pole_ecl[2], -pole_ecl[1]], dtype=np.float32)
                pole_ra, pole_dec = ecliptic_to_pole(pole_ecl)

                new_ecl_x = ppx + c_pos[0]
                new_ecl_y = ppy + c_pos[1]
                new_ecl_z = ppz + c_pos[2]

                new_ecl_vx = pvx + c_vel[0]
                new_ecl_vy = pvy + c_vel[1]
                new_ecl_vz = pvz + c_vel[2]

                payload = {
                    "action": "CREATE",
                    "name": ad["name"],
                    "radius": ad["radius"],
                    "mass": real_mass,
                    "color": list(ad["color"]),
                    "type": ad["type"],
                    "parent_idx": insp_idx,
                    "pos": [new_ecl_x, new_ecl_y, new_ecl_z],
                    "vel": [new_ecl_vx, new_ecl_vy, new_ecl_vz],
                    "rotation_period": float(ad.get("rotation_period", 24.0)),
                    "axial_tilt": tilt_deg,
                    "tidally_locked": bool(ad.get("tidally_locked", False)),
                    "oblateness": float(ad.get("oblateness", 0.0)),
                    "J2": float(ad.get("J2", 0.0)),
                    "j4": float(ad.get("j4", 0.0)),
                    "pole_ra": float(pole_ra),
                    "pole_dec": float(pole_dec),
                    "pole_ecl": [float(pole_ecl[0]), float(pole_ecl[1]), float(pole_ecl[2])],
                    "pole_render": [float(pole_render[0]), float(pole_render[1]), float(pole_render[2])]
                }
                with app.shared_state["lock"]:
                    app.shared_state["crud_queue"].append(payload)
                app.camera["add_mode"] = False
        imgui.end()

    # ── 3. Create New Star System Modal ──
    if app.camera.get("show_create_system", False):
        imgui.set_next_window_size(380, 480, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(app.fb_width // 2 - 190, app.fb_height // 2 - 240, imgui.FIRST_USE_EVER)
        expanded_cs, app.camera["show_create_system"] = imgui.begin("Create New Star System", True)
        if expanded_cs:
            cd = app.camera["create_sys_data"]
            imgui.text_colored("System", 0.6, 0.9, 1.0)
            _, cd["system_name"] = imgui.input_text("System Name", cd["system_name"], 256)
            imgui.separator()
            imgui.text_colored("Star Properties", 1.0, 0.85, 0.4)
            _, cd["star_name"] = imgui.input_text("Star Name", cd["star_name"], 256)
            _, cd["mass"] = imgui.drag_float("Mass (M☉)", cd.get("mass", 1.0), 0.01, 0.01, 300.0, format="%.4f")
            _, cd["metallicity"] = imgui.drag_float("[Fe/H] (dex)", cd.get("metallicity", 0.0), 0.01, -4.0, 1.0, format="%.3f")
            _, cd["age_pct"] = imgui.drag_float("Life Cycle (%)", cd.get("age_pct", 46.0), 0.1, -5.0, 120.0, format="%.1f%%")

            imgui.separator()
            name_ok = len(cd["system_name"].strip()) > 0
            name_exists = app.sys_mgr.system_exists(cd["system_name"].strip())

            if not name_ok:
                imgui.text_colored("System name required", 1.0, 0.3, 0.3)
            elif name_exists:
                imgui.text_colored("System already exists!", 1.0, 0.3, 0.3)

            if name_ok and not name_exists and cd.get("mass", 1.0) > 0.01:
                if imgui.button("Create System", width=-1):
                    sys_name = cd["system_name"].strip()
                    star_name_c = cd["star_name"].strip() or "Star"
                    sprops = {
                        "mode": "evolution",
                        "mass": cd["mass"],
                        "metallicity": cd["metallicity"],
                        "age_pct": cd["age_pct"] / 100.0,
                        "evo_path": "standard",
                        "radius": 1.0,
                        "temp": 5778.0,
                        "lum": 1.0,
                        "rotation_period": 0.0,
                        "rot_frac": 0.0,
                        "inclination": 0.0
                    }
                    app.sys_mgr.create_new_system_from_props(sys_name, star_name_c, sprops)
                    trigger_system_switch(sys_name)
                    app.camera["show_create_system"] = False
        imgui.end()

    # ── 4. Ephemeris Setup Modal ──
    if getattr(app, "_show_ephem_setup_modal", False):
        imgui.open_popup("Ephemeris Setup")

    if imgui.begin_popup_modal("Ephemeris Setup", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
        imgui.text("Select which SPICE kernels to download and load:")
        imgui.text_colored("Warning: High resolution satellite kernels can be large.", 1.0, 0.5, 0.2)
        imgui.separator()

        groups = {
            "Required Core": ["naif0012.tls", "pck00010.tpc", "de440s.bsp"],
            "Mars Moons": ["mar099s.bsp"],
            "Jupiter Moons": ["jup365.bsp", "jup347.bsp", "jup348.bsp", "jup349.bsp"],
            "Saturn Moons": ["sat441.bsp", "sat455.bsp", "sat456.bsp", "sat457.bsp", "sat459.bsp"],
            "Uranus Moons": ["ura111.bsp"],
            "Neptune Moons": ["nep095.bsp", "nep105.bsp", "nep104.bsp"],
            "Pluto System": ["plu060.bsp"]
        }

        for group_name, k_list in groups.items():
            imgui.text_colored(f"--- {group_name} ---", 0.7, 0.9, 1.0)
            for k_name in k_list:
                if k_name not in app.sys_mgr_spice.DEFAULT_KERNELS:
                    continue
                desc = app.sys_mgr_spice.KERNEL_DESCRIPTIONS.get(k_name, k_name)
                if app.sys_mgr_spice.find_kernel_path(k_name):
                    desc += " [Downloaded]"
                is_enabled = app.sys_mgr_spice.enabled_kernels.get(k_name, False)
                changed_k, new_val = imgui.checkbox(desc, is_enabled)
                if changed_k:
                    app.sys_mgr_spice.enabled_kernels[k_name] = new_val

        # ── Additional Kernels (Auto-detected from data/kernels/additional/ and data/kernels/) ──
        imgui.separator()
        imgui.text_colored("--- Additional Kernels (Missions & Custom) ---", 0.7, 0.9, 1.0)
        imgui.text_colored("Placed in data/kernels/additional/", 0.6, 0.8, 0.6)

        additional_kernels = app.sys_mgr_spice.list_additional_kernels()
        if additional_kernels:
            for ak in additional_kernels:
                k_name = ak["name"]
                is_enabled = app.sys_mgr_spice.enabled_kernels.get(k_name, False)
                label = f"{k_name} ({ak['size_kb']:.1f} KB)"
                changed_ak, new_val = imgui.checkbox(label, is_enabled)
                if changed_ak:
                    app.sys_mgr_spice.enabled_kernels[k_name] = new_val
        else:
            imgui.text_disabled("No additional kernels found in data/kernels/additional/.")

        # In-modal Import / Add BSP feature
        if imgui.tree_node("Import / Add External BSP to Ephemeris Mode..."):
            imgui.text("Enter path to an external .bsp kernel file:")
            changed_p, new_p = imgui.input_text("BSP File Path", getattr(app, "_ephem_add_bsp_path", ""), 512)
            if changed_p:
                app._ephem_add_bsp_path = new_p

            target_p = getattr(app, "_ephem_add_bsp_path", "").strip()
            can_add = bool(target_p and os.path.isfile(target_p) and target_p.lower().endswith(".bsp"))
            
            if can_add:
                if getattr(app, "_ephem_add_inspected_path", "") != target_p:
                    app._ephem_add_inspected_data = app.sys_mgr_spice.inspect_bsp(target_p)
                    app._ephem_add_inspected_path = target_p
                
                info = getattr(app, "_ephem_add_inspected_data", None)
                if info and info.get("bodies"):
                    b_names = [f"{b['name']} (ID {b['id']})" for b in info["bodies"][:3]]
                    if len(info["bodies"]) > 3:
                        b_names.append(f"... +{len(info['bodies']) - 3} more")
                    imgui.text_colored(f"Targets: {', '.join(b_names)}", 0.4, 0.85, 1.0)
                    imgui.text(f"Span: {info['utc_range'][0]} to {info['utc_range'][1]} ({info['duration_days']:.1f} days)")

                if imgui.button("  Add to Additional Kernels  "):
                    try:
                        app.sys_mgr_spice.add_additional_kernel(target_p)
                        app._ephem_add_bsp_path = ""
                        app._ephem_add_inspected_path = ""
                        app._ephem_add_inspected_data = None
                    except Exception as e:
                        print(f"[Ephemeris Setup] Failed to add kernel: {e}")
            else:
                imgui.text_disabled("Specify a valid .bsp file path to preview and import.")
            imgui.tree_pop()

        imgui.separator()
        if app.shared_state.get("ephemeris_mode", False):
            if imgui.button("Save & Apply Changes"):
                app.sys_mgr_spice.save_settings()
                app._ephem_mapping_ver = None
                app._show_ephem_setup_modal = False
                app._show_ephem_download_modal = True
                imgui.close_current_popup()
                trigger_ephem_switch()
        else:
            if imgui.button("Save & Switch to Ephemeris Mode"):
                app.sys_mgr_spice.save_settings()
                app._ephem_mapping_ver = None
                app._show_ephem_setup_modal = False
                imgui.close_current_popup()
                trigger_ephem_switch()

        imgui.same_line()
        if imgui.button("Cancel"):
            app._show_ephem_setup_modal = False
            imgui.close_current_popup()

        imgui.end_popup()

    # ── 5. Downloading SPICE Data Modal ──
    if getattr(app, "_show_ephem_download_modal", False) or app.sys_mgr_spice.is_downloading or app.sys_mgr_spice.download_error:
        imgui.open_popup("Downloading SPICE Ephemeris Data")

    if imgui.begin_popup_modal("Downloading SPICE Ephemeris Data", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
        if app.sys_mgr_spice.is_downloading:
            imgui.text_colored("Downloading NASA JPL SPICE Kernels...", 0.4, 0.8, 1.0)
            imgui.text(app.sys_mgr_spice.download_status)

            if app.sys_mgr_spice.download_bytes_total > 0:
                mb_cur = app.sys_mgr_spice.download_bytes_current / (1024 * 1024)
                mb_tot = app.sys_mgr_spice.download_bytes_total / (1024 * 1024)
                spd = app.sys_mgr_spice.download_speed_str
                imgui.text(f"Progress: {mb_cur:.1f} MB / {mb_tot:.1f} MB  ({spd})")
            elif app.sys_mgr_spice.download_speed_str:
                imgui.text(f"Speed: {app.sys_mgr_spice.download_speed_str}")

            imgui.progress_bar(app.sys_mgr_spice.download_progress, size=(340, 0))
            imgui.separator()
            if imgui.button("Cancel Download", width=-1):
                app.sys_mgr_spice.cancel_download()
                app._show_ephem_download_modal = False
                imgui.close_current_popup()
        elif app.sys_mgr_spice.download_error:
            imgui.text_colored("Download Failed!", 1.0, 0.3, 0.3)
            imgui.text_wrapped(app.sys_mgr_spice.download_error)
            imgui.separator()
            if imgui.button("Close", width=-1):
                app.sys_mgr_spice.download_error = None
                app._show_ephem_download_modal = False
                imgui.close_current_popup()
        else:
            app._show_ephem_download_modal = False
            imgui.close_current_popup()
        imgui.end_popup()

    # ── 5b. Import Ephemeris Kernel (.bsp) Modal ──
    if getattr(app, "_show_bsp_import_modal", False):
        imgui.open_popup("Import Ephemeris Kernel (.bsp)")

    if imgui.begin_popup_modal("Import Ephemeris Kernel (.bsp)", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
        imgui.text_colored("SPICE Ephemeris Viewer & Playback", 0.4, 0.85, 1.0)
        imgui.text("Select or specify a NASA SPICE SPK kernel (.bsp) to inspect and play back:")
        imgui.separator()

        # Initialize modal state if needed
        if not hasattr(app, "_bsp_available_files") or getattr(app, "_bsp_refresh_needed", True):
            app._bsp_available_files = app.sys_mgr_spice.list_available_bsp_files()
            app._bsp_refresh_needed = False
            app._bsp_selected_idx = 0 if app._bsp_available_files else -1
            app._bsp_custom_path = ""
            app._bsp_inspected_data = None
            app._bsp_inspected_path = ""

        # Available file options
        file_items = [f"[{f['category']}] {f['name']} ({f['size_kb']:.1f} KB)" for f in app._bsp_available_files]
        file_items.append("Custom File Path...")

        cur_idx = getattr(app, "_bsp_selected_idx", 0)
        if cur_idx < 0 or cur_idx >= len(file_items):
            cur_idx = 0

        changed_f, new_idx = imgui.combo("Select Kernel", cur_idx, file_items)
        if changed_f:
            app._bsp_selected_idx = new_idx

        imgui.same_line()
        if imgui.button("Refresh"):
            app._bsp_available_files = app.sys_mgr_spice.list_available_bsp_files()
            app._bsp_refresh_needed = False
            if app._bsp_selected_idx >= len(app._bsp_available_files):
                app._bsp_selected_idx = 0 if app._bsp_available_files else -1

        # Determine target path
        is_custom = (app._bsp_selected_idx == len(app._bsp_available_files)) or (not app._bsp_available_files)
        if is_custom:
            changed_p, new_p = imgui.input_text("File Path", getattr(app, "_bsp_custom_path", ""), 512)
            if changed_p:
                app._bsp_custom_path = new_p
            target_path = getattr(app, "_bsp_custom_path", "").strip()
        else:
            target_path = app._bsp_available_files[app._bsp_selected_idx]["path"]

        # Run inspect_bsp if path changed or not inspected yet
        if target_path and os.path.isfile(target_path):
            if getattr(app, "_bsp_inspected_path", "") != target_path:
                app._bsp_inspected_data = app.sys_mgr_spice.inspect_bsp(target_path)
                app._bsp_inspected_path = target_path
        else:
            app._bsp_inspected_data = None
            app._bsp_inspected_path = ""

        imgui.separator()

        info = getattr(app, "_bsp_inspected_data", None)
        if info:
            # Metadata display
            sidecar = info.get("sidecar")
            fmt_desc = "Standard NASA SPICE SPK"
            if sidecar:
                spk_t = sidecar.get("spk_type", 2)
                if spk_t == 2:
                    deg = sidecar.get("chebyshev", {}).get("degree", 13)
                    fmt_desc = f"Type 2 Chebyshev (Degree {deg}) [Stellar-Forge Export]"
                elif spk_t == 13:
                    deg = sidecar.get("hermite_degree", 5)
                    fmt_desc = f"Type 13 Hermite (Degree {deg}) [Stellar-Forge Export]"

            imgui.text_colored("Kernel Info:", 0.8, 0.9, 1.0)
            imgui.text(f"File: {info['filename']}  ({fmt_desc})")
            imgui.text(f"Time Span (UTC): {info['utc_range'][0]}  ->  {info['utc_range'][1]}")
            imgui.text(f"Duration: {info['duration_days']:.1f} days ({info['duration_days'] / 365.25:.2f} years)")
            
            imgui.separator()
            imgui.text_colored(f"Detected Bodies ({len(info['bodies'])}):", 0.8, 0.9, 1.0)

            # Child window for body list
            imgui.begin_child("BodiesList###bsp_bodies", 580, 160, border=True)
            for b in info["bodies"]:
                c_hex = b.get("color", "#ffffff")
                try:
                    h = c_hex.lstrip("#")
                    cr, cg, cb = [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
                except Exception:
                    cr, cg, cb = 0.7, 0.85, 1.0

                imgui.text_colored(" * ", cr, cg, cb)
                imgui.same_line()
                imgui.text_colored(f"{b['name']}", 1.0, 1.0, 1.0)
                imgui.same_line()
                imgui.text_colored(f"(NAIF ID: {b['id']})", 0.65, 0.75, 0.85)
                imgui.same_line()
                imgui.text_colored(f"Ref: {b['center_name']} ({b['center_id']})", 0.55, 0.65, 0.75)
                imgui.same_line()
                imgui.text(f"[{b['utc_start']} - {b['utc_end']}]")
            imgui.end_child()

            imgui.separator()
            if imgui.button("  Open Ephemeris Playback  ", width=220):
                app._show_bsp_import_modal = False
                imgui.close_current_popup()
                if "open_generic_ephem" in switch_triggers:
                    switch_triggers["open_generic_ephem"](info["filepath"])
            imgui.same_line()
        else:
            if target_path:
                imgui.text_colored(f"File not found or unreadable: {target_path}", 1.0, 0.4, 0.4)
            else:
                imgui.text_colored("Select or specify a valid .bsp file above.", 0.7, 0.7, 0.7)

        if imgui.button("Cancel"):
            app._show_bsp_import_modal = False
            imgui.close_current_popup()

        imgui.end_popup()

    # ── 6. Jump to Date / Render Timeline Modal ──
    if app.camera.get("show_jump_modal", False):
        imgui.set_next_window_size(410, 480, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(app.fb_width // 2 - 205, app.fb_height // 2 - 240, imgui.FIRST_USE_EVER)
        expanded, app.camera["show_jump_modal"] = imgui.begin("Jump to Date / Render Timeline###jump_modal", True)
        if expanded:
            ephem_active = app.shared_state.get("ephemeris_mode", False)
            kepler_active = app.shared_state.get("keplerian_mode", False)

            # Mode Header
            if ephem_active:
                imgui.text_colored("Mode: NASA SPICE Ephemeris Playback", 0.4, 0.8, 1.0)
            elif kepler_active:
                imgui.text_colored("Mode: Analytical Keplerian Orbiting", 0.4, 0.8, 1.0)
            else:
                imgui.text_colored("Mode: N-Body Simulation (IAS15)", 0.4, 0.8, 1.0)

            utc_y, utc_m, utc_d, utc_h, utc_mn, utc_s, _ = format_sim_time_utc(display_t)
            imgui.text_colored(f"Sim Time (UTC):   {utc_y:04d}-{utc_m:02d}-{utc_d:02d} {utc_h:02d}:{utc_mn:02d}:{utc_s:02d} UTC", 0.6, 0.9, 1.0)
            imgui.text_colored(f"Sim Time (Local): {cur_y:04d}-{cur_m:02d}-{cur_d:02d} {cur_h:02d}:{cur_mn:02d}:{cur_s:02d} ({cur_tz})", 0.7, 0.85, 1.0)
            imgui.separator()

            # Timezone input selection (UTC vs Local)
            use_utc = app.camera.setdefault("jump_use_utc", True)
            imgui.text("Input Time Standard:")
            imgui.same_line()
            r_utc = imgui.radio_button("UTC", use_utc)
            if r_utc:
                app.camera["jump_use_utc"] = True
            imgui.same_line()
            r_loc = imgui.radio_button(f"Local ({cur_tz})", not use_utc)
            if r_loc:
                app.camera["jump_use_utc"] = False
            use_utc = app.camera["jump_use_utc"]

            imgui.separator()
            init_vals = [utc_y, utc_m, utc_d, utc_h, utc_mn, utc_s] if use_utc else [cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s]
            jd = app.camera.setdefault("jump_date", init_vals)

            imgui.text("Target Date:")
            imgui.push_item_width(90)
            _, jd[0] = imgui.input_int("Year", jd[0])
            imgui.same_line()
            _, jd[1] = imgui.input_int("Month", jd[1])
            imgui.same_line()
            _, jd[2] = imgui.input_int("Day", jd[2])
            imgui.pop_item_width()

            time_label = "Target Time (UTC):" if use_utc else f"Target Time (Local - {cur_tz}):"
            imgui.text(time_label)
            imgui.push_item_width(50)
            _, jd[3] = imgui.input_int("##Hour", jd[3], step=0)
            imgui.same_line(); imgui.text(":")
            imgui.same_line()
            _, jd[4] = imgui.input_int("##Min", jd[4], step=0)
            imgui.same_line(); imgui.text(":")
            imgui.same_line()
            _, jd[5] = imgui.input_int("##Sec", jd[5], step=0)
            imgui.pop_item_width()

            # Date clamping
            jd[1] = max(1, min(12, jd[1]))
            jd[2] = max(1, min(31, jd[2]))
            jd[3] = max(0, min(23, jd[3]))
            jd[4] = max(0, min(59, jd[4]))
            jd[5] = max(0, min(59, jd[5]))

            # Quick Presets
            imgui.separator()
            imgui.text_colored("Quick Presets:", 0.6, 0.9, 1.0)
            if imgui.button("Today / Now"):
                if use_utc:
                    now_dt = datetime.datetime.now(datetime.timezone.utc)
                else:
                    now_dt = datetime.datetime.now()
                jd[0], jd[1], jd[2] = now_dt.year, now_dt.month, now_dt.day
                jd[3], jd[4], jd[5] = now_dt.hour, now_dt.minute, now_dt.second

            imgui.same_line()
            if imgui.button("J2000"):
                jd[0], jd[1], jd[2], jd[3], jd[4], jd[5] = 2000, 1, 1, 12, 0, 0

            imgui.same_line()
            if imgui.button("+1 Year"):
                jd[0] += 1

            imgui.same_line()
            if imgui.button("-1 Year"):
                jd[0] -= 1

            imgui.separator()
            # Action button
            if ephem_active or kepler_active:
                imgui.text_wrapped("Directly synchronizes the celestial positions to the specified target epoch.")
                imgui.spacing()
                if imgui.button("🚀 Jump to Date", width=-1):
                    target_t = sim_time_from_date(jd[0], jd[1], jd[2], jd[3], jd[4], jd[5], is_utc=use_utc)
                    app.time_ctrl["sync_t"] = target_t
                    app.camera["show_jump_modal"] = False
            else:
                imgui.text_wrapped("Integrates an accurate N-body trajectory from the current time to the target date, opening an interactive timeline scrubber.")
                imgui.spacing()
                res_options = ["Auto (adaptive)", "1,000 steps", "10,000 steps", "50,000 steps", "200,000 steps"]
                res_values = [0, 1000, 10000, 50000, 200000]
                res_idx = app.camera.setdefault("timeline_res_idx", 0)
                imgui.text("Timeline Resolution:")
                imgui.same_line()
                imgui.push_item_width(170)
                changed_res, res_idx = imgui.combo("##tl_res", res_idx, res_options)
                imgui.pop_item_width()
                if changed_res:
                    app.camera["timeline_res_idx"] = res_idx
                app.time_ctrl["timeline_steps"] = res_values[res_idx]
                if imgui.button("⏳ Render Timeline to Date", width=-1):
                    target_t = sim_time_from_date(jd[0], jd[1], jd[2], jd[3], jd[4], jd[5], is_utc=use_utc)
                    app.time_ctrl["target_t"] = target_t
                    app.time_ctrl["cancel_render"] = False
                    app.time_ctrl["render_timeline"] = True
                    app.camera["show_jump_modal"] = False

            imgui.spacing()
            if imgui.button("Close", width=-1):
                app.camera["show_jump_modal"] = False
        imgui.end()
