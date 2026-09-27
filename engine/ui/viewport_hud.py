import math
import imgui
from engine.rendering.render_utils import format_flight_speed
from engine.core.constants import MAX_FLIGHT_SPEED_AU_S

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
            changed, new_log = imgui.slider_float("##fly_quick", speed_log, -13.0, math.log10(MAX_FLIGHT_SPEED_AU_S), "")
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

    # ── Geometry Statistics / Triangle Count Overlay ──
    if app.camera.get("show_triangle_count", False):
        total_tris = getattr(app, "total_last_triangle_count", 0)
        terrain_tris = getattr(app, "terrain_last_triangle_count", 0)
        sphere_tris = getattr(app, "sphere_last_triangle_count", 0)
        patches = getattr(app, "terrain_last_patch_count", 0)

        x_pos = 12
        y_pos = 100 if not getattr(app, "show_outliner", True) else 36
        if getattr(app, "show_outliner", True):
            x_pos = 285

        imgui.set_next_window_position(x_pos, y_pos, imgui.FIRST_USE_EVER)
        flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE
        expanded, opened = imgui.begin("Geometry Statistics###geom_stats", True, flags)
        if not opened:
            app.camera["show_triangle_count"] = False
            app.save_settings()
        elif expanded:
            imgui.text_colored("Rendered Triangles", 0.4, 0.8, 1.0)
            imgui.separator()
            imgui.text(f"Total:   {total_tris:,}")
            p_res = int(app.camera.get("terrain_patch_res", 32))
            imgui.text_colored(f"Terrain: {terrain_tris:,}", 0.3, 1.0, 0.5)
            imgui.same_line()
            imgui.text_disabled(f"({patches:,} patches @ {p_res}x{p_res})")
            if sphere_tris > 0:
                imgui.text(f"Spheres: {sphere_tris:,}")

            io = imgui.get_io()
            fps = io.framerate
            ms = 1000.0 / max(1.0, fps)
            color_fps = (0.3, 1.0, 0.4) if fps >= 55.0 else ((1.0, 0.8, 0.2) if fps >= 30.0 else (1.0, 0.3, 0.3))
            imgui.separator()
            imgui.text_colored(f"{fps:.1f} FPS", *color_fps)
            imgui.same_line()
            imgui.text_disabled(f"({ms:.1f} ms)")
        imgui.end()
