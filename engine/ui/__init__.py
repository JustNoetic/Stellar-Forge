"""
Stellar-Forge UI Package ('Orion UI')

Organized into modular components:
- menu_bar: Top Main Menu Bar (Systems, Physics, View, Render, Tools, Quick HUD)
- time_hud: Bottom Time Transport & Timeline Scrubber Bar
- outliner: Left Panel System Hierarchy with Search/Filter
- inspector: Right Panel Tabbed Body Inspector
- modals: Centralized Dialogs & Setup Modals
- viewport_hud: Viewport Overlay & Toasts
"""

from engine.ui.menu_bar import render_main_menu_bar
from engine.ui.time_hud import render_time_hud
from engine.ui.outliner import render_system_outliner
from engine.ui.inspector import render_body_inspector
from engine.ui.modals import render_modals
from engine.ui.viewport_hud import render_viewport_hud

def render_ui(
    app,
    ctx,
    bodies_data,
    visual_data,
    atmo_bodies,
    ring_bodies,
    star_idx,
    num_bodies,
    mass_snap,
    parent_snap,
    tree_indices_snap,
    tree_depths_snap,
    subsys_mass_buf,
    subsys_pos_buf,
    subsys_vel_buf,
    pos_snap_render,
    vel_snap_render,
    visual_arr,
    cam_world_pos_f8,
    cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s, cur_tz,
    display_t, dt_render,
    tl_active, tl_prog, tl_times, is_scrubbing,
    spice_valid_mask,
    prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex,
    switch_triggers
):
    """Main orchestrator for rendering all ImGui interface layers in Stellar-Forge."""
    if not getattr(app, "ui_visible", True):
        # Even when UI is hidden, render active screenshot toast notifications
        render_viewport_hud(app, bodies_data)
        return

    active_system_name = getattr(app, "active_system_name", "Solar System")
    keplerian_mode_active = app.shared_state.get("keplerian_mode", False)
    ephemeris_mode_active = app.shared_state.get("ephemeris_mode", False)

    # 1. Top Main Menu Bar
    render_main_menu_bar(
        app,
        bodies_data,
        visual_data,
        atmo_bodies,
        ring_bodies,
        star_idx,
        cur_y, cur_m, cur_d,
        display_t,
        switch_triggers
    )

    # 2. System Outliner (Left Panel)
    render_system_outliner(
        app,
        bodies_data,
        num_bodies,
        mass_snap,
        tree_indices_snap,
        tree_depths_snap,
        spice_valid_mask,
        ephemeris_mode_active,
        active_system_name,
        switch_triggers
    )

    # 3. Body Inspector (Right Panel)
    render_body_inspector(
        app,
        ctx,
        bodies_data,
        num_bodies,
        parent_snap,
        mass_snap,
        subsys_mass_buf,
        subsys_pos_buf,
        subsys_vel_buf,
        pos_snap_render,
        vel_snap_render,
        visual_arr,
        cam_world_pos_f8,
        active_system_name,
        atmo_bodies,
        ring_bodies,
        prog_rings,
        ring_precomputed,
        ring_render_groups,
        ring_gradient_tex
    )

    # 4. Bottom Time Transport & Timeline Bar
    render_time_hud(
        app,
        cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s, cur_tz,
        display_t, dt_render,
        tl_active, tl_prog, tl_times,
        is_scrubbing,
        ephemeris_mode_active,
        keplerian_mode_active
    )

    # 5. Viewport Floating HUD & Toast Notifications
    render_viewport_hud(app, bodies_data)

    # 6. Centralized Modals & Dialogs
    render_modals(
        app,
        bodies_data,
        visual_data,
        atmo_bodies,
        ring_bodies,
        star_idx,
        num_bodies,
        mass_snap,
        pos_snap_render,
        vel_snap_render,
        visual_arr,
        cur_y, cur_m, cur_d,
        display_t,
        switch_triggers
    )
