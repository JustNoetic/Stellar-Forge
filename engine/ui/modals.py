import math
import datetime
import os
import imgui
from engine.core.constants import DEFAULT_LY_THRESHOLD_AU
from engine.core.math_utils import get_cartesian_from_keplerian, rotate_equatorial_to_ecliptic
from engine.rendering.render_utils import compute_surface_albedo, format_distance_au
from engine.ephemeris.system_manager import SystemManager

def render_modals(app, bodies_data, visual_data, atmo_bodies, ring_bodies, star_idx, num_bodies, mass_snap, pos_snap_render, vel_snap_render, visual_arr, cur_y, cur_m, cur_d, display_t, switch_triggers):
    """Render all popup dialogs and modal windows."""
    load_system_from_data = switch_triggers["load_system"]
    trigger_ephem_switch = switch_triggers["ephem_switch"]

    # ── 1. Graphics & Quality Settings Modal ──
    if app.camera.get("show_settings_modal", False):
        imgui.set_next_window_size(380, 520, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(app.fb_width // 2 - 190, app.fb_height // 2 - 260, imgui.FIRST_USE_EVER)
        expanded, app.camera["show_settings_modal"] = imgui.begin("Graphics & Quality Settings", True)
        if expanded:
            settings_changed = False

            # Atmosphere Quality
            atmo_quality = app.camera.get("atmo_quality", 1)
            changed_aq, atmo_quality = imgui.combo("Atmosphere Quality", atmo_quality, ["Off", "Low (2D Shadows)", "High (Volumetric)"])
            if changed_aq:
                app.camera["atmo_quality"] = min(2, atmo_quality)
                settings_changed = True

            if atmo_quality > 0:
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

            # Bloom
            changed_bi, app.camera["bloom_intensity"] = imgui.slider_float("Bloom Intensity", app.camera.get("bloom_intensity", 0.05), 0.0, 1.0, "%.3f")
            if changed_bi: settings_changed = True
            changed_bt, app.camera["bloom_threshold"] = imgui.slider_float("Bloom Threshold", app.camera.get("bloom_threshold", 1.0), 0.0, 10.0, "%.2f")
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

            # Habitable Zones
            changed_hz, app.camera["show_habitable_zone"] = imgui.checkbox("Show Habitable Zones", app.camera.get("show_habitable_zone", False))
            if changed_hz: settings_changed = True

            # Atmosphere & Refraction
            changed_atmo, app.camera["atmo_enabled"] = imgui.checkbox("Enable Atmosphere Rendering", app.camera.get("atmo_enabled", True))
            if changed_atmo: settings_changed = True

            changed_refr, app.camera["refraction_enabled"] = imgui.checkbox("Enable Atmospheric Refraction", app.camera.get("refraction_enabled", True))
            if changed_refr: settings_changed = True

            # Planetshine & Ringshine
            changed_ps, app.camera["planetshine_enabled"] = imgui.checkbox("Enable Planetshine/Moonshine", app.camera.get("planetshine_enabled", True))
            if changed_ps: settings_changed = True

            changed_rs, app.camera["ringshine_enabled"] = imgui.checkbox("Enable Ringshine", app.camera.get("ringshine_enabled", True))
            if changed_rs: settings_changed = True

            if app.camera.get("ringshine_enabled", True):
                imgui.indent()
                ringshine_bands = int(app.camera.get("ringshine_band_count", 256))
                changed_rsb, ringshine_bands = imgui.slider_int("Ringshine Bands", ringshine_bands, 4, 1024)
                if changed_rsb:
                    app.camera["ringshine_band_count"] = ringshine_bands
                    settings_changed = True
                imgui.unindent()

            # Dynamic Texture Streaming
            stream_thresh = float(app.camera.get("tex_stream_threshold_px", 500.0))
            changed_st, stream_thresh = imgui.slider_float("Min High-Res Size (px)", stream_thresh, 100.0, 2000.0, "%.0f px")
            if changed_st:
                app.camera["tex_stream_threshold_px"] = stream_thresh
                settings_changed = True

            if settings_changed:
                app.save_settings()

            imgui.separator()
            if imgui.button("Close", width=-1):
                app.camera["show_settings_modal"] = False
        imgui.end()

    # ── 2. Add Orbiting Body Modal ──
    if app.camera.get("add_mode", False):
        imgui.set_next_window_size(360, 420, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(app.fb_width // 2 - 180, app.fb_height // 2 - 210, imgui.FIRST_USE_EVER)
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

            imgui.separator()
            imgui.text_colored("Orbital Elements", 1.0, 0.85, 0.4)
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
                if insp_idx >= num_bodies: insp_idx = 0
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
                    pole_render = visual_arr[insp_idx, 5:8]
                    pole_ecl = np.array([pole_render[0], -pole_render[2], pole_render[1]])
                    c_pos, c_vel = rotate_equatorial_to_ecliptic(c_pos, c_vel, pole_ecl)

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
                    "vel": [new_ecl_vx, new_ecl_vy, new_ecl_vz]
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
                    new_bodies = app.sys_mgr.create_new_system_from_props(sys_name, star_name_c, sprops)
                    new_bndl = load_system_from_data(new_bodies)
                    app.switch_req_name = sys_name
                    req = {
                        "old_bodies_data": bodies_data,
                        "old_visual_data": visual_data,
                        "old_atmo_bodies": atmo_bodies,
                        "old_ring_bodies": ring_bodies,
                        "old_star_idx": star_idx,
                        "new_bundle": new_bndl,
                    }
                    with app.shared_state["lock"]:
                        app.shared_state["system_switch_request"] = req
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
                if os.path.exists(os.path.join(app.sys_mgr_spice.KERNEL_DIR, k_name)):
                    desc += " [Downloaded]"
                is_enabled = app.sys_mgr_spice.enabled_kernels.get(k_name, False)
                changed_k, new_val = imgui.checkbox(desc, is_enabled)
                if changed_k:
                    app.sys_mgr_spice.enabled_kernels[k_name] = new_val

        imgui.separator()
        if app.shared_state.get("ephemeris_mode", False):
            if imgui.button("Save & Apply Changes"):
                app.sys_mgr_spice.save_settings()
                app._ephem_mapping_ver = None
                app._show_ephem_setup_modal = False
                app._show_ephem_download_modal = True
                imgui.close_current_popup()
                app.sys_mgr_spice.download_kernels_async(on_complete=lambda: app.sys_mgr_spice.load_kernels(force=True))
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
