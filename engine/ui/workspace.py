"""Shared visual language and display-space geometry for the ImGui workspace."""
from dataclasses import dataclass

import imgui


ACCENT = (0.38, 0.78, 0.86)
AMBER = (0.94, 0.70, 0.38)


@dataclass(frozen=True)
class Workspace:
    width: float
    height: float
    gap: float
    top: float
    bottom: float
    outliner_width: float
    inspector_width: float
    left: float
    right: float
    transport_height: float
    scale: float

    @property
    def panel_height(self):
        return max(40, self.bottom - self.top)


def workspace(app):
    # ImGui coordinates are logical pixels; framebuffer pixels differ on HiDPI.
    width, height = imgui.get_io().display_size
    if width <= 0 or height <= 0:
        width, height = app.fb_width, app.fb_height
    menu_height = imgui.get_font_size() + imgui.get_style().frame_padding.y * 2
    return calculate_workspace(app, width, height, menu_height)


def calculate_workspace(app, width, height, menu_height=None):
    """Also usable by scene layout before an ImGui frame has begun."""
    scale = max(0.9, min(1.5, float(app.camera.get("ui_scale", 1.0))))
    gap = 8 * scale
    top = (23 * scale if menu_height is None else menu_height) + gap
    transport_height = 76 * scale
    bottom = height - gap
    if getattr(app, "show_time_hud", True):
        bottom -= transport_height + gap
    outliner_w = min(float(app.camera.get("ui_outliner_width", 260)) * scale, width * 0.25)
    inspector_w = min(float(app.camera.get("ui_inspector_width", 420)) * scale, width * 0.40)
    left = outliner_w + gap * 2 if getattr(app, "show_outliner", True) else gap
    idx = app.camera.get("inspected_idx")
    show_inspector = getattr(app, "show_inspector", True) and idx is not None
    right = width - (inspector_w + gap * 2 if show_inspector else gap)
    return Workspace(width, height, gap, top, bottom, outliner_w, inspector_w,
                     left, right, transport_height, scale)


def comparator_bounds(app, width, height):
    if not getattr(app, "ui_visible", True):
        return 20, width - 20, 20, height - 20
    layout = calculate_workspace(app, width, height)
    return layout.left, layout.right, layout.top + 120 * layout.scale, layout.bottom


def panel(app, side):
    layout = workspace(app)
    width = layout.outliner_width if side == "left" else layout.inspector_width
    x = layout.gap if side == "left" else layout.width - width - layout.gap
    aligned = app.camera.get("ui_aligned_panels", True)
    signature = (layout.width, layout.height, width, layout.top, layout.bottom,
                 aligned, getattr(app, "_ui_layout_revision", 0))
    key = "_ui_layout_" + side
    changed = getattr(app, key, None) != signature
    setattr(app, key, signature)
    condition = imgui.ALWAYS if aligned or changed else imgui.ONCE
    imgui.set_next_window_position(x, layout.top, condition)
    imgui.set_next_window_size(width, layout.panel_height, condition)
    flags = imgui.WINDOW_NO_COLLAPSE
    if aligned:
        flags |= imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE
    return layout, flags


def reset_workspace(app):
    app.show_outliner = app.show_inspector = app.show_time_hud = True
    app.camera.update(ui_aligned_panels=True, ui_outliner_width=260,
                      ui_inspector_width=420, ui_scale=1.0)
    app._ui_layout_revision = getattr(app, "_ui_layout_revision", 0) + 1
    app.save_settings()


def apply_theme(app):
    scale = max(0.9, min(1.5, float(app.camera.get("ui_scale", 1.0))))
    context = imgui.get_current_context()
    if getattr(app, "_ui_theme_signature", None) == (context, scale):
        return
    app._ui_theme_signature = (context, scale)
    imgui.get_io().font_global_scale = scale
    style = imgui.get_style()
    style.window_padding = (12 * scale, 10 * scale)
    style.frame_padding = (7 * scale, 5 * scale)
    style.item_spacing = (8 * scale, 7 * scale)
    style.item_inner_spacing = (6 * scale, 4 * scale)
    style.indent_spacing = 14 * scale
    style.scrollbar_size = 10 * scale
    style.window_rounding = 5 * scale
    style.child_rounding = 4 * scale
    style.frame_rounding = 3 * scale
    style.popup_rounding = 5 * scale
    style.grab_rounding = 3 * scale
    style.tab_rounding = 3 * scale
    style.window_border_size = 1
    style.frame_border_size = 0
    colors = {
        "TEXT": (0.87, 0.91, 0.95, 1), "TEXT_DISABLED": (0.48, 0.56, 0.64, 1),
        "WINDOW_BACKGROUND": (0.055, 0.075, 0.105, 0.97),
        "CHILD_BACKGROUND": (0.045, 0.062, 0.088, 0.65),
        "POPUP_BACKGROUND": (0.075, 0.10, 0.14, 0.99),
        "BORDER": (0.18, 0.24, 0.30, 0.65),
        "FRAME_BACKGROUND": (0.11, 0.15, 0.20, 1),
        "FRAME_BACKGROUND_HOVERED": (0.15, 0.25, 0.31, 1),
        "FRAME_BACKGROUND_ACTIVE": (0.18, 0.31, 0.38, 1),
        "TITLE_BACKGROUND": (0.07, 0.10, 0.14, 1),
        "TITLE_BACKGROUND_ACTIVE": (0.10, 0.17, 0.22, 1),
        "MENUBAR_BACKGROUND": (0.06, 0.085, 0.12, 1),
        "BUTTON": (0.13, 0.20, 0.26, 1),
        "BUTTON_HOVERED": (0.19, 0.34, 0.40, 1),
        "BUTTON_ACTIVE": (0.23, 0.43, 0.50, 1),
        "HEADER": (0.13, 0.27, 0.33, 0.8),
        "HEADER_HOVERED": (0.20, 0.37, 0.44, 0.85),
        "HEADER_ACTIVE": (0.23, 0.43, 0.50, 1),
        "TAB": (0.10, 0.15, 0.20, 1),
        "TAB_HOVERED": (0.20, 0.37, 0.44, 1),
        "TAB_ACTIVE": (0.15, 0.29, 0.35, 1),
        "CHECK_MARK": (*ACCENT, 1), "SLIDER_GRAB": (*ACCENT, 0.85),
        "SLIDER_GRAB_ACTIVE": (*ACCENT, 1),
        "SEPARATOR": (0.18, 0.24, 0.30, 0.7),
        "SCROLLBAR_BACKGROUND": (0.04, 0.06, 0.08, 0.6),
        "SCROLLBAR_GRAB": (0.20, 0.28, 0.34, 1),
        "SCROLLBAR_GRAB_HOVERED": (0.28, 0.40, 0.46, 1),
        "SCROLLBAR_GRAB_ACTIVE": (*ACCENT, 1),
        "PLOT_HISTOGRAM": (*AMBER, 1),
    }
    for name, color in colors.items():
        style.colors[getattr(imgui, "COLOR_" + name)] = color


def heading(title, subtitle=None):
    imgui.text_colored(title, *ACCENT)
    if subtitle:
        imgui.text_wrapped(subtitle)
    imgui.spacing()
    imgui.separator()
    imgui.spacing()


SETTINGS_SECTIONS = ("Atmosphere", "Optics", "Visibility", "Lighting", "Terrain", "Workspace")


def open_settings(app, section=None):
    app.camera["show_settings_modal"] = True
    if section is not None:
        app._ui_settings_section = section
