import imgui
import math
from engine.ui.workspace import workspace, ACCENT, AMBER, open_settings, reset_workspace, SETTINGS_SECTIONS
from engine.ephemeris.system_manager import SystemManager

def render_main_menu_bar(app, bodies_data, visual_data, atmo_bodies, ring_bodies, star_idx, cur_y, cur_m, cur_d, display_t, switch_triggers, cur_h=0, cur_mn=0, cur_s=0, cur_tz="UTC"):
    """Render the top main menu bar for Stellar-Forge."""
    trigger_system_switch = switch_triggers["switch_system"]
    trigger_ephem_switch = switch_triggers["ephem_switch"]
    trigger_ephem_exit = switch_triggers["ephem_exit"]

    active_system_name = getattr(app, "active_system_name", SystemManager.SOLAR_SYSTEM_NAME)
    keplerian_mode_active = app.shared_state.get("keplerian_mode", False)
    ephemeris_mode_active = app.shared_state.get("ephemeris_mode", False)

    if not imgui.begin_main_menu_bar():
        return

    layout = workspace(app)
    if layout.width > 1050 * layout.scale:
        imgui.text_colored("STELLAR FORGE", *ACCENT)
        imgui.same_line(spacing=18)

    # ── Systems Menu ──
    if imgui.begin_menu("Systems"):
        imgui.text_colored("Switch System:", 0.6, 0.9, 1.0)
        imgui.separator()
        system_list = app.sys_mgr.list_systems()
        for sname in system_list:
            is_active = (sname == active_system_name)
            clicked, _ = imgui.menu_item(sname, None, is_active)
            if clicked and sname != active_system_name:
                trigger_system_switch(sname)

        imgui.separator()
        if imgui.menu_item("+ Create New System...")[0]:
            app.camera["show_create_system"] = True
            app.camera["create_sys_data"] = {
                "system_name": "New System",
                "star_name": "Star",
                "mass": 1.0,
                "metallicity": 0.0,
                "age_pct": 46.0,
            }

        if imgui.menu_item("Import Ephemeris Kernel (.bsp)...")[0]:
            app._show_bsp_import_modal = True
            app._bsp_refresh_needed = True

        if active_system_name != SystemManager.SOLAR_SYSTEM_NAME:
            imgui.separator()
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.4, 0.4)
            if imgui.menu_item("- Delete Current System")[0]:
                app.sys_mgr.delete_system(active_system_name)
                trigger_system_switch(SystemManager.SOLAR_SYSTEM_NAME)
            imgui.pop_style_color()

        imgui.end_menu()

    # ── Physics Mode Menu ──
    if imgui.begin_menu("Physics"):
        cur_mode = 0
        if keplerian_mode_active:
            cur_mode = 1
        elif ephemeris_mode_active:
            cur_mode = 2

        clicked_nbody, _ = imgui.menu_item("N-Body (IAS15 Integrator)", None, cur_mode == 0)
        if clicked_nbody and cur_mode != 0:
            if cur_mode == 2:
                trigger_ephem_exit(to_keplerian=False)
            else:
                app.shared_state["ephemeris_mode"] = False
                app.shared_state["keplerian_mode"] = False
            if hasattr(app, "shared_state_cmp"):
                app.shared_state_cmp["ephemeris_mode"] = False
                app.shared_state_cmp["keplerian_mode"] = False

        clicked_kepler, _ = imgui.menu_item("Analytical Keplerian (Jacobi)", None, cur_mode == 1)
        if clicked_kepler and cur_mode != 1:
            if cur_mode == 2:
                trigger_ephem_exit(to_keplerian=True)
            else:
                app.shared_state["ephemeris_mode"] = False
                app.shared_state["keplerian_mode"] = True
                with app.shared_state["lock"]:
                    app.shared_state["keplerian_reextract"] = True
            if hasattr(app, "shared_state_cmp"):
                app.shared_state_cmp["ephemeris_mode"] = False
                app.shared_state_cmp["keplerian_mode"] = True
                with app.shared_state_cmp["lock"]:
                    app.shared_state_cmp["keplerian_reextract"] = True

        can_ephem = (active_system_name == SystemManager.SOLAR_SYSTEM_NAME or ephemeris_mode_active)
        if can_ephem:
            clicked_ephem, _ = imgui.menu_item("NASA SPICE Ephemeris Playback", None, cur_mode == 2)
            if clicked_ephem and cur_mode != 2:
                if not app.sys_mgr_spice.settings_initialized:
                    app._show_ephem_setup_modal = True
                else:
                    trigger_ephem_switch()

        imgui.separator()
        jump_label = "Render Timeline to Date..." if cur_mode == 0 else "Jump to Date..."
        if imgui.menu_item(jump_label)[0]:
            app.camera["show_jump_modal"] = True
            use_utc = app.camera.get("jump_use_utc", True)
            if use_utc:
                from engine.rendering.render_utils import format_sim_time_utc
                uy, um, ud, uh, umn, us, _ = format_sim_time_utc(display_t)
                jd = app.camera.setdefault("jump_date", [uy, um, ud, uh, umn, us])
                jd[0], jd[1], jd[2] = uy, um, ud
                jd[3], jd[4], jd[5] = uh, umn, us
            else:
                jd = app.camera.setdefault("jump_date", [cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s])
                jd[0], jd[1], jd[2] = cur_y, cur_m, cur_d
                jd[3], jd[4], jd[5] = cur_h, cur_mn, cur_s

        if keplerian_mode_active:
            if imgui.menu_item("Export & Switch to N-Body Mode")[0]:
                app.shared_state["ephemeris_mode"] = False
                app.shared_state["keplerian_mode"] = False
                with app.shared_state["lock"]:
                    app.shared_state["keplerian_export"] = True
                if hasattr(app, "shared_state_cmp"):
                    app.shared_state_cmp["ephemeris_mode"] = False
                    app.shared_state_cmp["keplerian_mode"] = False
                    with app.shared_state_cmp["lock"]:
                        app.shared_state_cmp["keplerian_export"] = True

        if ephemeris_mode_active:
            if imgui.menu_item("Export to N-Body System")[0]:
                switch_triggers["ephem_export"](display_t, cur_y, cur_m, cur_d)

            if imgui.menu_item("Return to N-Body Mode")[0]:
                trigger_ephem_exit(to_keplerian=False)

        imgui.end_menu()

    # ── View & Camera Menu ──
    if imgui.begin_menu("View"):
        comparator = app.camera.get("scene_mode") == "size_comparator"
        if imgui.menu_item("Scene: Simulation", None, not comparator)[0]:
            app.camera["scene_mode"] = "simulation"
            app.camera["left_dragging"] = app.camera["right_dragging"] = False
            app.pick_request = None
        if imgui.menu_item("Scene: Size Comparator", None, comparator)[0]:
            app.camera["scene_mode"] = "size_comparator"
            app.camera["left_dragging"] = app.camera["right_dragging"] = False
            app.camera["keys"] = {}
            app.camera["approach_delta"] = 0.0
            app.pick_request = None
        if comparator:
            if imgui.menu_item("Fit All Bodies")[0]:
                from engine.rendering.size_comparator import comparator_view
                comparator_view(app.camera)["fit"] = True
        imgui.separator()
        move_mode = app.camera.get("movement_mode", 0)
        if imgui.menu_item("Camera: Free Flight", None, move_mode == 0)[0]:
            app.camera["movement_mode"] = 0
            app.save_settings()
        if imgui.menu_item("Camera: Simple Orbit", None, move_mode == 1)[0]:
            app.camera["movement_mode"] = 1
            app.camera["cam_look"] = "aim"
            app.save_settings()

        imgui.separator()
        changed_ha, app.camera["horizon_align"] = imgui.checkbox("Auto-level Horizon", app.camera.get("horizon_align", True))
        if changed_ha:
            app.save_settings()

        fov_val = app.camera.get("fov", 45.0)
        changed_fov, new_fov = imgui.slider_float("FOV", fov_val, 0.001, 120.0, "%.1f deg")
        if changed_fov:
            app.camera["fov"] = max(0.001, min(120.0, new_fov))
            app.save_settings()

        if imgui.menu_item("Reset FOV (45°)")[0]:
            app.camera["fov"] = 45.0
            app.save_settings()

        imgui.separator()
        _, app.show_outliner = imgui.checkbox("System Outliner Panel", getattr(app, "show_outliner", True))
        _, app.show_inspector = imgui.checkbox("Body Inspector Panel", getattr(app, "show_inspector", True))
        _, app.show_time_hud = imgui.checkbox("Time Transport HUD", getattr(app, "show_time_hud", True))
        changed_stc, app.camera["show_triangle_count"] = imgui.checkbox("Geometry Statistics (Triangle Count)", app.camera.get("show_triangle_count", False))
        if changed_stc:
            app.save_settings()

        if imgui.menu_item("Reset Workspace Layout")[0]:
            reset_workspace(app)
        imgui.separator()
        if imgui.menu_item("Toggle Full UI (H)")[0]:
            app.ui_visible = not getattr(app, "ui_visible", True)

        imgui.end_menu()

    # Quality controls live in one categorized workspace.
    if imgui.begin_menu("Render"):
        changed, app.camera["show_gaia_stars"] = imgui.checkbox("GAIA Starfield", app.camera.get("show_gaia_stars", True))
        if changed:
            app.save_settings()
        if app.camera.get("show_gaia_stars", True) and app.star_catalog is not None and not app.star_catalog.loaded:
            imgui.text_wrapped("Run scripts/fetch_gaia.py to build the star catalog.")
        imgui.separator()
        for index, label in enumerate(SETTINGS_SECTIONS):
            if imgui.menu_item(label + " Settings...")[0]:
                open_settings(app, index)
        imgui.end_menu()

    # ── Tools Menu ──
    if imgui.begin_menu("Tools"):
        if imgui.menu_item("Jump to Date / Render Timeline...")[0]:
            app.camera["show_jump_modal"] = True
            use_utc = app.camera.get("jump_use_utc", True)
            if use_utc:
                from engine.rendering.render_utils import format_sim_time_utc
                uy, um, ud, uh, umn, us, _ = format_sim_time_utc(display_t)
                jd = app.camera.setdefault("jump_date", [uy, um, ud, uh, umn, us])
                jd[0], jd[1], jd[2] = uy, um, ud
                jd[3], jd[4], jd[5] = uh, umn, us
            else:
                jd = app.camera.setdefault("jump_date", [cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s])
                jd[0], jd[1], jd[2] = cur_y, cur_m, cur_d
                jd[3], jd[4], jd[5] = cur_h, cur_mn, cur_s

        if imgui.menu_item("Import Ephemeris Kernel (.bsp)...")[0]:
            app._show_bsp_import_modal = True
            app._bsp_refresh_needed = True

        if imgui.menu_item("SPICE Ephemeris Kernel Settings...")[0]:
            app._show_ephem_setup_modal = True

        imgui.separator()
        imgui.text_colored("System Comparison", 0.6, 0.9, 1.0)
        c_cmp, app.comparison_enabled = imgui.checkbox("Enable Comparison Mode", app.comparison_enabled)
        if c_cmp and app.comparison_enabled and app.camera.get("comparison_force_keplerian", True):
            if ephemeris_mode_active:
                trigger_ephem_exit(to_keplerian=True)
            else:
                app.shared_state["ephemeris_mode"] = False
                app.shared_state["keplerian_mode"] = True
                with app.shared_state["lock"]:
                    app.shared_state["keplerian_reextract"] = True
            if hasattr(app, "shared_state_cmp"):
                app.shared_state_cmp["ephemeris_mode"] = False
                app.shared_state_cmp["keplerian_mode"] = True
                with app.shared_state_cmp["lock"]:
                    app.shared_state_cmp["keplerian_reextract"] = True

        c_fk, force_kep = imgui.checkbox("Force Keplerian Mode##cmp_fk", app.camera.get("comparison_force_keplerian", True))
        if c_fk:
            app.camera["comparison_force_keplerian"] = force_kep
            app.save_settings()
            if force_kep and app.comparison_enabled:
                if ephemeris_mode_active:
                    trigger_ephem_exit(to_keplerian=True)
                else:
                    app.shared_state["ephemeris_mode"] = False
                    app.shared_state["keplerian_mode"] = True
                    with app.shared_state["lock"]:
                        app.shared_state["keplerian_reextract"] = True
                if hasattr(app, "shared_state_cmp"):
                    app.shared_state_cmp["ephemeris_mode"] = False
                    app.shared_state_cmp["keplerian_mode"] = True
                    with app.shared_state_cmp["lock"]:
                        app.shared_state_cmp["keplerian_reextract"] = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Automatically switch simulation to analytical Keplerian mode when enabling system comparison")

        if app.comparison_enabled:
            system_list = app.sys_mgr.list_systems()
            try:
                curr_cmp_idx = system_list.index(app.comparison_system_name)
            except ValueError:
                curr_cmp_idx = 0
            c_cmp_sys, new_cmp_idx = imgui.combo("Compare With", curr_cmp_idx, system_list)
            if c_cmp_sys:
                switch_triggers["cmp_switch"](system_list[new_cmp_idx])
            _, app.comparison_offset_au = imgui.drag_float("Offset (AU)##cmp", app.comparison_offset_au, 0.1)

        imgui.separator()
        imgui.text_colored("Distance Units", 0.6, 0.9, 1.0)
        LY_TO_AU = 63241.077
        cur_thresh_ly = app.camera.get("ly_threshold_au", 1000.0) / LY_TO_AU
        changed_thresh, new_thresh_ly = imgui.drag_float("AU -> ly Threshold", cur_thresh_ly, 0.01, 0.001, 1000.0, format="%.3f ly")
        if changed_thresh:
            app.camera["ly_threshold_au"] = max(0.0001, new_thresh_ly) * LY_TO_AU

        imgui.separator()
        imgui.text_colored("Accumulation", 0.6, 0.9, 1.0)
        cp, app.camera["photo_accum_enabled"] = imgui.checkbox("Pause Accumulation", app.camera.get("photo_accum_enabled", False))
        if cp: app.save_settings()
        cs, app.camera["screenshot_accum_enabled"] = imgui.checkbox("Screenshot Accumulation", app.camera.get("screenshot_accum_enabled", False))
        if cs: app.save_settings()

        imgui.end_menu()

    # ── Right-Side Quick Actions HUD ──
    # Display active mode badge and quick screenshot/settings controls
    badge_label = "[N-BODY]"
    badge_color = (0.4, 0.8, 1.0)
    if keplerian_mode_active:
        badge_label = "[KEPLERIAN]"
        badge_color = (0.3, 1.0, 0.3)
    elif ephemeris_mode_active:
        badge_label = "[EPHEMERIS]"
        badge_color = (1.0, 0.8, 0.2)

    # Right align controls
    cursor_x = imgui.get_window_width() - 360 * layout.scale
    if cursor_x > imgui.get_cursor_pos_x():
        imgui.set_cursor_pos_x(cursor_x)

    if layout.width > 900 * layout.scale:
        imgui.text_colored(badge_label, *badge_color)
        imgui.same_line(spacing=15)

    # Screenshot quick button
    _ss_can_capture = not getattr(app, "_screenshot_capturing", False) and not getattr(app, "_screenshot_saving", False)
    if not _ss_can_capture:
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

    if imgui.button("Capture" if layout.width < 1100 * layout.scale else "Capture (F12)"):
        if _ss_can_capture:
            _ss_presets = [(3840, 2160), (7680, 4320), (15360, 8640)]
            _ss_idx = max(0, min(len(_ss_presets) - 1, app.camera.get("screenshot_res_idx", 1)))
            app._screenshot_request = _ss_presets[_ss_idx]

    if not _ss_can_capture:
        imgui.pop_style_var()

    imgui.same_line(spacing=5)
    _ss_res_labels = ["4K", "8K", "16K"]
    _ss_res_idx = app.camera.get("screenshot_res_idx", 1)
    imgui.push_item_width(55)
    _ss_changed, _ss_new_idx = imgui.combo("##res_quick", _ss_res_idx, _ss_res_labels)
    if _ss_changed:
        app.camera["screenshot_res_idx"] = _ss_new_idx
        app.save_settings()
    imgui.pop_item_width()

    imgui.same_line(spacing=10)
    if imgui.button("Settings"):
        open_settings(app)

    imgui.end_main_menu_bar()
