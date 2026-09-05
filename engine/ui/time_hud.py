import imgui
import math
from engine.rendering.render_utils import format_time_speed, sim_time_from_date

def render_time_hud(app, cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s, cur_tz, display_t, dt_render, tl_active, tl_prog, tl_times, is_scrubbing, ephemeris_mode_active, keplerian_mode_active):
    """Render the bottom time transport bar and timeline controls."""
    if not getattr(app, "show_time_hud", True):
        return

    scrub_index = getattr(app, "scrub_index", [0])
    jump_date = getattr(app, "jump_date", [cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s])

    bar_w = min(940, app.fb_width - 40)
    bar_h = 58 if is_scrubbing or (tl_active and tl_prog < 1.0) else 44
    bar_x = (app.fb_width - bar_w) / 2
    bar_y = app.fb_height - bar_h - 10

    imgui.set_next_window_position(bar_x, bar_y, imgui.ALWAYS)
    imgui.set_next_window_size(bar_w, bar_h, imgui.ALWAYS)
    
    flags = (
        imgui.WINDOW_NO_TITLE_BAR | 
        imgui.WINDOW_NO_RESIZE | 
        imgui.WINDOW_NO_MOVE | 
        imgui.WINDOW_NO_SCROLLBAR | 
        imgui.WINDOW_NO_COLLAPSE
    )

    expanded, _ = imgui.begin("TimeHUD###time_hud", False, flags)
    if not expanded:
        imgui.end()
        return

    if app.shared_state.get("syncing", False):
        imgui.text("Synchronizing Physics...")
        imgui.same_line()
        imgui.progress_bar(app.shared_state.get("sync_progress", 0.0), size=(300, 0.0))
        imgui.end()
        return

    if tl_active and tl_prog < 1.0:
        imgui.text("Rendering Timeline...")
        imgui.same_line()
        rate = app.shared_state.get("timeline_rate", 0.0)
        if rate > 0.0:
            imgui.text(f"Pace: {format_time_speed(rate)}")
            imgui.same_line()
        imgui.progress_bar(tl_prog, size=(260, 0.0))
        imgui.same_line()
        if imgui.button("Cancel"):
            app.time_ctrl["cancel_render"] = True
        imgui.end()
        return

    if is_scrubbing:
        max_idx = max(0, len(tl_times) - 1)
        if app.time_ctrl.get("snap_to_end", False):
            scrub_index[0] = max_idx
            app.time_ctrl["snap_to_end"] = False

        if app.time_ctrl["timeline_playing"]:
            if imgui.button("Pause Playback"):
                app.time_ctrl["timeline_playing"] = False

            app.time_ctrl["timeline_scrub_float"] += app.time_ctrl["timeline_speed"] * dt_render
            steps_to_add = int(app.time_ctrl["timeline_scrub_float"])
            if steps_to_add > 0:
                scrub_index[0] += steps_to_add
                app.time_ctrl["timeline_scrub_float"] -= steps_to_add
                if scrub_index[0] > max_idx:
                    scrub_index[0] = 0
        else:
            if imgui.button("Play Timeline"):
                app.time_ctrl["timeline_playing"] = True
                app.time_ctrl["timeline_scrub_float"] = 0.0
                if scrub_index[0] >= max_idx:
                    scrub_index[0] = 0

        imgui.same_line(spacing=10)
        imgui.push_item_width(80)
        _, app.time_ctrl["timeline_speed"] = imgui.slider_float("Speed##tl", app.time_ctrl["timeline_speed"], 1.0, 100.0, "%.1fx")
        imgui.pop_item_width()

        imgui.same_line(spacing=10)
        imgui.push_item_width(320)
        changed_scrub, scrub_index[0] = imgui.slider_int("##Scrub", scrub_index[0], 0, max_idx, "")
        imgui.pop_item_width()

        imgui.same_line(spacing=10)
        if imgui.button("Resume Here"):
            app.time_ctrl["sync_t"] = tl_times[scrub_index[0]]
            app.time_ctrl["sync_idx"] = scrub_index[0]
            app.time_ctrl["paused"] = False
            app.time_ctrl["timeline_playing"] = False
            with app.shared_state["lock"]:
                app.shared_state["timeline_active"] = False

        imgui.same_line()
        if imgui.button("Cancel"):
            with app.shared_state["lock"]:
                app.shared_state["timeline_active"] = False
            app.time_ctrl["timeline_playing"] = False

        imgui.end()
        return

    # Standard Transport Controls
    # 1. Play / Pause
    is_paused = app.time_ctrl["paused"]
    btn_play_pause = " Play " if is_paused else " Pause "
    if imgui.button(btn_play_pause, width=54):
        app.time_ctrl["paused"] = not is_paused

    # 2. Direction (Forward / Backward)
    imgui.same_line(spacing=6)
    td = app.time_ctrl["time_direction"]
    dir_label = " Forward " if td >= 0 else " Reverse "
    if imgui.button(dir_label, width=68):
        app.time_ctrl["time_direction"] = -td

    # 3. 1x Reset Speed
    imgui.same_line(spacing=6)
    if imgui.button("1x", width=28):
        app.time_ctrl["multiplier"] = 1.0

    # 4. Speed Slider (Fixed position, placed BEFORE speed text so text length never moves the slider)
    imgui.same_line(spacing=8)
    imgui.push_item_width(140)
    val_log = math.log10(max(1.0, abs(app.time_ctrl["multiplier"])))
    changed_speed, new_log = imgui.slider_float("##speed_slider", val_log, 0.0, 12.0, "")
    if changed_speed:
        app.time_ctrl["multiplier"] = 10 ** new_log
    imgui.pop_item_width()

    # 5. Speed readout (Text placed AFTER slider)
    imgui.same_line(spacing=8)
    effective_mult = app.time_ctrl["multiplier"] * td
    imgui.text(format_time_speed(effective_mult))

    # 6. Date / Time Display + Jump Button (Anchored to right side)
    btn_label = "Jump to Date..." if (ephemeris_mode_active or keplerian_mode_active) else "Render Timeline..."
    right_section_w = 330
    right_x = bar_w - right_section_w - 12
    if right_x > imgui.get_cursor_pos_x():
        imgui.same_line()
        imgui.set_cursor_pos_x(right_x)
    else:
        imgui.same_line(spacing=15)

    imgui.text_colored(f"{cur_y:04d}-{cur_m:02d}-{cur_d:02d} {cur_h:02d}:{cur_mn:02d}:{cur_s:02d} {cur_tz}", 0.85, 0.9, 1.0)

    # Button to open Jump in Time & Render Timeline modal
    imgui.same_line(spacing=8)
    if imgui.button(btn_label):
        app.camera["show_jump_modal"] = True
        jd = app.camera.setdefault("jump_date", [cur_y, cur_m, cur_d, cur_h, cur_mn, cur_s])
        jd[0], jd[1], jd[2] = cur_y, cur_m, cur_d
        jd[3], jd[4], jd[5] = cur_h, cur_mn, cur_s

    imgui.end()
