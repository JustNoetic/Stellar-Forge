"""Small, responsive presentation helpers for the body inspector."""
import imgui
from engine.ui.workspace import AMBER


_LABELS = {
    'Surface G': 'Surface gravity', 'Rot Period': 'Rotation period',
    'Spectral Cl': 'Spectral class', 'Geom Albedo': 'Geometric albedo',
    'Bond Albedo': 'Bond albedo', 'Oblateness (f)': 'Flattening (f)',
    'System Mass': 'System mass', 'Semi-major (a)': 'Semi-major axis (a)',
    'Long Asc Node (Ω)': 'Ascending node', 'Arg Periapsis (ω)': 'Periapsis argument',
    'True Anomaly (ν)': 'True anomaly', 'Mean Anomaly (M)': 'Mean anomaly (M)',
    'Hill Radius (now)': 'Hill radius (live estimate)', 'Roche Limit (fluid)': 'Roche limit (fluid estimate)',
    'Surface Albedo': 'Surface albedo', 'Bond Albedo (A_b)': 'Bond albedo (A_b)',
    'Equilibrium Temperature': 'Equilibrium temperature', 'Atmosphere Height': 'Atmosphere height',
    'Gas Scale Height': 'Gas scale height', 'Mean Molar Mass': 'Mean molar mass',
}


def _visible(text):
    # The default ImGui font lacks astronomical unit symbols and Greek letters.
    for symbol, label in (('M☉', 'solar masses'), ('M⊕', 'Earth masses'),
                          ('M☾', 'lunar masses'), ('R☉', 'solar radii'),
                          ('L☉', 'solar luminosities'), ('Ω (Long Asc Node)', 'Ascending node (deg)'),
                          ('ω (Arg Periapsis)', 'Periapsis argument (deg)')):
        text = text.replace(symbol, label)
    return text


def value_row(label, value):
    """Align labels and values, stacking long content instead of clipping it."""
    label = _visible(_LABELS.get(label, label))
    value = _visible(str(value))
    start = imgui.get_cursor_pos_x()
    available = imgui.get_content_region_available()[0]
    label_width = imgui.calc_text_size(label)[0]
    value_width = imgui.calc_text_size(str(value))[0]
    spacing = imgui.get_style().item_inner_spacing.x
    imgui.text_disabled(label)
    if label_width + value_width + spacing * 2 <= available:
        imgui.same_line()
        imgui.set_cursor_pos_x(max(start + label_width + spacing, start + available - value_width))
        imgui.text(str(value))
    else:
        imgui.text_wrapped(str(value))


def metric_text(text):
    label, separator, value = text.strip().partition(':')
    if separator and value.strip():
        value_row(label.strip(), value.strip())
    else:
        imgui.text_wrapped(text.strip())


def section(title, note=None):
    imgui.spacing()
    imgui.text_colored(title, *AMBER)
    imgui.separator()
    if note:
        imgui.text_wrapped(note)


def field(widget, label, *args, **kwargs):
    """Keep field labels above full-width editors, with unique hidden IDs."""
    visible = _visible(label.split('##')[0])
    imgui.text_wrapped(visible)
    imgui.push_item_width(-1)
    try:
        return widget('##' + label, *args, **kwargs)
    finally:
        imgui.pop_item_width()
