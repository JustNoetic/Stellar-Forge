import math
import imgui
from engine.rendering.render_utils import format_flight_speed

def render_viewport_hud(app, bodies_data):
    """Render minimal floating HUD over the 3D viewport (camera info pill)."""
    # ── Floating Camera / Navigation Pill (Top-Left under Menu Bar when Outliner is closed) ──
    if not getattr(app, "show_outliner", True):
        move_mode = app.camera.get("movement_mode", 0)
        pill_w = 260
        pill_h = 58 if move_mode == 0 else 38

        imgui.set_next_window_position(12, 36, imgui.ALWAYS)
        imgui.set_next_window_size(pill_w, pill_h, imgui.ALWAYS)
        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR
        imgui.begin("##camera_pill", False, flags)

        mode_name = "Free Flight" if move_mode == 0 else "Simple Orbit"
        imgui.text_colored(f"Camera: {mode_name}", 0.6, 0.9, 1.0)

        if move_mode == 0:
            speed_val = app.camera.get("flight_speed", 0.1)
            imgui.text(f"Speed: {format_flight_speed(speed_val)}")
            imgui.same_line()
            imgui.push_item_width(120)
            speed_log = math.log10(max(speed_val, 1e-13))
            changed, new_log = imgui.slider_float("##fly_quick", speed_log, -13.0, 0.7, "")
            if changed:
                app.camera["flight_speed"] = 10 ** new_log
            imgui.pop_item_width()
        else:
            track_name = "None"
            t_idx = app.camera.get("tracking_idx")
            if t_idx is not None and t_idx < len(bodies_data):
                track_name = bodies_data[t_idx].get("name", "Unknown")
            imgui.text(f"Tracking: {track_name}")

        imgui.end()
