#include "app.hpp"
#include "core/log.hpp"
#include <GLFW/glfw3.h>

namespace sf {

UserSettings App::capture_preferences() const {
    UserSettings values;
    values.graphics = render_settings_.values();
    values.workspace = ui_settings_;
    values.workspace.ui_visible = true;
    values.workspace.show_settings = false;
    values.workspace.settings_section = 0;
    values.navigation = {camera_.fov_deg, camera_.flight_speed_au_s, camera_.movement_mode, camera_.horizon_align};
    return values;
}

void App::load_preferences() {
    if (self_test_ || benchmark_.enabled) return;
    const auto loaded = load_user_settings(user_settings_path());
    render_settings_.apply(loaded.values.graphics);
    ui_settings_ = loaded.values.workspace;
    camera_.fov_deg = loaded.values.navigation.fov_deg;
    camera_.flight_speed_au_s = loaded.values.navigation.flight_speed_au_s;
    camera_.movement_mode = loaded.values.navigation.movement_mode;
    camera_.horizon_align = loaded.values.navigation.horizon_align;
    saved_preferences_ = pending_preferences_ = capture_preferences();
    preferences_writable_ = loaded.writable;
    preferences_notice_ = loaded.warning;
    if (!preferences_notice_.empty()) sf::log::warn("{} ({})", preferences_notice_, user_settings_path().string());
}

void App::save_preferences(bool force) {
    if (self_test_ || benchmark_.enabled || !preferences_writable_) return;
    const auto current = capture_preferences();
    const double now = glfwGetTime();
    if (current != pending_preferences_) {
        pending_preferences_ = current;
        preferences_save_at_ = now + 0.65;
    }
    if (current == saved_preferences_ && !preferences_reset_requested_) {
        preferences_save_error_.clear();
        return;
    }
    if (!force && (now < preferences_save_at_ || ImGui::IsAnyItemActive())) return;
    const auto saved = save_user_settings(user_settings_path(), current);
    preferences_save_error_ = saved.error;
    if (saved) {
        saved_preferences_ = current;
        preferences_reset_requested_ = false;
        preferences_notice_.clear();
    } else {
        preferences_save_at_ = now + 3.0;
        sf::log::warn("{} ({})", saved.error, user_settings_path().string());
    }
}

void App::restore_preferences(rendering::RenderSettings& render_draft) {
    const UserSettings defaults;
    render_draft = defaults.graphics;
    const bool open = ui_settings_.show_settings;
    const int section = ui_settings_.settings_section;
    ui_settings_ = defaults.workspace;
    ui_settings_.show_settings = open;
    ui_settings_.settings_section = section;
    camera_.fov_deg = defaults.navigation.fov_deg;
    camera_.flight_speed_au_s = defaults.navigation.flight_speed_au_s;
    camera_.movement_mode = defaults.navigation.movement_mode;
    camera_.horizon_align = defaults.navigation.horizon_align;
    preferences_reset_requested_ = true;
    preferences_save_at_ = glfwGetTime() + 0.65;
}

}
