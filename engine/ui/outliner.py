import imgui
from engine.ephemeris.system_manager import SystemManager

def render_system_outliner(app, bodies_data, num_bodies, mass_snap, tree_indices_snap, tree_depths_snap, spice_valid_mask, ephemeris_mode_active, active_system_name, switch_triggers):
    """Render the left-side System Outliner panel (celestial body hierarchy)."""
    if not getattr(app, "show_outliner", True):
        return

    trigger_system_switch = switch_triggers["switch_system"]

    outliner_w = 300
    outliner_x = 10
    outliner_y = 32
    # Leave room for bottom time HUD
    bottom_space = 70 if getattr(app, "show_time_hud", True) else 15
    outliner_h = max(200, app.fb_height - outliner_y - bottom_space)

    force_layout = getattr(app, "_last_fb_width", 0) != app.fb_width or getattr(app, "_last_fb_height", 0) != app.fb_height
    cond = imgui.ALWAYS if force_layout else imgui.ONCE

    imgui.set_next_window_position(outliner_x, outliner_y, cond)
    imgui.set_next_window_size(outliner_w, outliner_h, cond)

    expanded, opened = imgui.begin("System Outliner###outliner", True)
    if not opened:
        app.show_outliner = False
        imgui.end()
        return

    if not expanded:
        imgui.end()
        return

    # ── Header: Active System & Actions ──
    imgui.text_colored(f"System: {active_system_name}", 0.6, 0.9, 1.0)
    
    # System switch dropdown
    system_list = app.sys_mgr.list_systems()
    try:
        cur_sys_idx = system_list.index(active_system_name)
    except ValueError:
        cur_sys_idx = 0

    imgui.push_item_width(170)
    c_sys, new_sys_idx = imgui.combo("##sys_select", cur_sys_idx, system_list)
    if c_sys and system_list[new_sys_idx] != active_system_name:
        trigger_system_switch(system_list[new_sys_idx])
    imgui.pop_item_width()

    imgui.same_line(spacing=6)
    if imgui.button("+ Body"):
        # Open Add Orbiting Body targeting the primary star or currently inspected body
        target_parent = app.camera["inspected_idx"] if app.camera["inspected_idx"] is not None and not app.camera.get("inspected_is_cmp", False) else 0
        is_moon = mass_snap[target_parent] < 0.01 if target_parent < len(mass_snap) else False
        app.camera["add_mode"] = True
        app.camera["add_data"] = {
            "name": f"{'Moon' if is_moon else 'Planet'} of {bodies_data[target_parent]['name']}",
            "type": "Moon" if is_moon else "Terrestrial",
            "mass": 1.0,
            "radius": 1737.0 if is_moon else 6371.0,
            "color": [0.7, 0.7, 0.7] if is_moon else [0.2, 0.5, 0.8],
            "a": 384400.0 if is_moon else 1.0,
            "e": 0.0, "inc": 0.0, "Omega": 0.0, "omega": 0.0, "M": 0.0,
            "frame": 0,
            "is_moon": is_moon,
            "parent_idx": target_parent
        }

    # ── Search / Filter Bar ──
    search_query = getattr(app, "_ui_search_query", "")
    imgui.push_item_width(-35 if search_query else -1)
    changed_search, search_query = imgui.input_text("##body_search", search_query, 64)
    if changed_search:
        app._ui_search_query = search_query
    imgui.pop_item_width()

    if search_query:
        imgui.same_line(spacing=4)
        if imgui.button("X##clear_search"):
            app._ui_search_query = ""
            search_query = ""

    imgui.separator()

    # ── Primary System Hierarchy Tree ──
    filter_text = search_query.strip().lower()

    for k in range(len(tree_indices_snap)):
        idx = int(tree_indices_snap[k])
        if ephemeris_mode_active and not spice_valid_mask[idx]:
            continue

        body_name = bodies_data[idx]['name']
        if filter_text and filter_text not in body_name.lower():
            continue

        depth = int(tree_depths_snap[k])
        indent = depth * 14
        if indent > 0:
            imgui.indent(indent)

        is_inspected = (app.camera["inspected_idx"] == idx and not app.camera.get("inspected_is_cmp", False))
        label = body_name
        if app.camera["tracking_idx"] == idx and not app.camera.get("tracking_is_cmp", False):
            label += " *"

        avail_w = imgui.get_content_region_available()[0]
        clicked = imgui.selectable(f"{label}##{idx}", is_inspected, 0, max(10.0, avail_w - 55.0))[0]

        rect_min_y = imgui.get_item_rect_min()[1]
        rect_max_y = imgui.get_item_rect_max()[1]
        rect_min_x = imgui.get_window_position()[0]
        rect_max_x = rect_min_x + imgui.get_window_width()
        mouse_x, mouse_y = imgui.get_io().mouse_pos

        show_buttons = (rect_min_y <= mouse_y <= rect_max_y) and (rect_min_x <= mouse_x <= rect_max_x)

        if clicked:
            if app.camera["inspected_idx"] == idx and not app.camera.get("inspected_is_cmp", False):
                if not app.camera["inspect_bary"]:
                    app.camera["inspect_bary"] = True
                else:
                    app.camera["inspected_idx"] = None
                    app.camera["inspect_bary"] = False
            else:
                app.camera["inspected_idx"] = idx
                app.camera["inspected_is_cmp"] = False
                app.camera["inspect_bary"] = False
                app.show_inspector = True

        if show_buttons:
            imgui.same_line()
            imgui.set_cursor_pos_x(imgui.get_window_width() - 55)
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
            cur_y_btn = imgui.get_cursor_pos_y()
            imgui.set_cursor_pos_y(cur_y_btn - 2)

            if imgui.button(f"+##add_{idx}", 18, 17):
                is_moon = mass_snap[idx] < 0.01
                app.camera["add_mode"] = True
                app.camera["add_data"] = {
                    "name": f"{'Moon' if is_moon else 'Planet'} of {bodies_data[idx]['name']}",
                    "type": "Moon" if is_moon else "Terrestrial",
                    "mass": 1.0,
                    "radius": 1737.0 if is_moon else 6371.0,
                    "color": [0.7, 0.7, 0.7] if is_moon else [0.2, 0.5, 0.8],
                    "a": 384400.0 if is_moon else 1.0,
                    "e": 0.0, "inc": 0.0, "Omega": 0.0, "omega": 0.0, "M": 0.0,
                    "frame": 0,
                    "is_moon": is_moon,
                    "parent_idx": idx
                }

            if idx > 0:
                imgui.same_line()
                imgui.set_cursor_pos_y(cur_y_btn - 2)
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.6, 0.1, 0.1)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.8, 0.2, 0.2)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.9, 0.3, 0.3)
                if imgui.button(f"-##del_{idx}", 18, 17):
                    with app.shared_state["lock"]:
                        app.shared_state["crud_queue"].append({
                            "action": "DELETE",
                            "idx": idx
                        })
                    app.camera["inspected_idx"] = None
                    app.camera["inspect_bary"] = False
                imgui.pop_style_color(3)

            imgui.pop_style_var(1)

        if indent > 0:
            imgui.unindent(indent)

    # ── Comparison System Hierarchy (if enabled) ──
    if app.comparison_enabled and getattr(app, "num_bodies_cmp", 0) > 0:
        imgui.separator()
        imgui.text_colored(f"{app.comparison_system_name} (Comparison)", 0.6, 0.9, 1.0)
        imgui.separator()
        for k in range(len(app.tree_indices_snap_cmp)):
            idx = int(app.tree_indices_snap_cmp[k])
            b_name = app.bodies_data_cmp[idx]['name']
            if filter_text and filter_text not in b_name.lower():
                continue

            depth = int(app.tree_depths_snap_cmp[k])
            indent = depth * 14
            if indent > 0:
                imgui.indent(indent)

            is_inspected = (app.camera["inspected_idx"] == idx and app.camera.get("inspected_is_cmp", False))
            label = b_name
            if app.camera["tracking_idx"] == idx and app.camera.get("tracking_is_cmp", False):
                label += " *"

            avail_w = imgui.get_content_region_available()[0]
            clicked = imgui.selectable(f"{label}##cmp_{idx}", is_inspected, 0, max(10.0, avail_w - 10.0))[0]

            if clicked:
                if app.camera["inspected_idx"] == idx and app.camera.get("inspected_is_cmp", False):
                    if not app.camera["inspect_bary"]:
                        app.camera["inspect_bary"] = True
                    else:
                        app.camera["inspected_idx"] = None
                        app.camera["inspected_is_cmp"] = False
                        app.camera["inspect_bary"] = False
                else:
                    app.camera["inspected_idx"] = idx
                    app.camera["inspected_is_cmp"] = True
                    app.camera["inspect_bary"] = False
                    app.show_inspector = True

            if indent > 0:
                imgui.unindent(indent)

    imgui.end()
