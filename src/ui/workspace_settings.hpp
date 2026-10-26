#pragma once

#include "core/constants.hpp"

namespace sf::ui {

struct WorkspaceSettings {
    bool ui_visible = true;
    bool aligned_panels = true;
    bool show_outliner = true;
    bool show_inspector = true;
    bool show_time_hud = true;
    bool show_settings = false;
    int settings_section = 0;
    bool show_telemetry = false;
    double ly_threshold_au = DEFAULT_LY_THRESHOLD_AU;
    float outliner_width = 260.0f;
    float inspector_width = 420.0f;
    float scale = 1.0f;
    bool confirm_delete = true;
    bool comparison_force_keplerian = true;
    bool comparison_auto_spacing = true;

    void reset_layout() {
        const WorkspaceSettings defaults;
        aligned_panels = defaults.aligned_panels;
        show_outliner = defaults.show_outliner;
        show_inspector = defaults.show_inspector;
        show_time_hud = defaults.show_time_hud;
        outliner_width = defaults.outliner_width;
        inspector_width = defaults.inspector_width;
        scale = defaults.scale;
        confirm_delete = defaults.confirm_delete;
    }

    bool operator==(const WorkspaceSettings&) const = default;
};

}
