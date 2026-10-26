#pragma once

#include "rendering/render_settings.hpp"
#include "ui/workspace_settings.hpp"
#include <filesystem>
#include <string>

namespace sf {

struct NavigationSettings {
    double fov_deg = FOV_DEG;
    double flight_speed_au_s = 9.59133562721224e-05;
    int movement_mode = 0;
    bool horizon_align = true;
    bool operator==(const NavigationSettings&) const = default;
};

struct UserSettings {
    rendering::RenderSettings graphics;
    ui::WorkspaceSettings workspace;
    NavigationSettings navigation;
    bool operator==(const UserSettings&) const = default;
};

struct UserSettingsLoad {
    UserSettings values;
    std::string warning;
    bool writable = true;
};

std::filesystem::path user_settings_path();
UserSettingsLoad load_user_settings(const std::filesystem::path& path);
CommandResult save_user_settings(const std::filesystem::path& path, const UserSettings& values);

}
