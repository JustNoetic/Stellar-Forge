#pragma once

#include "core/camera.hpp"
#include "ephemeris/system_manager.hpp"
#include "ephemeris/simulation.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/celestial_renderer.hpp"
#include "rendering/star_catalog.hpp"
#include "rendering/orbit_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include "rhi/vk_context.hpp"
#include "ui/workspace.hpp"
#include <string>
#include <vector>

struct GLFWwindow;

namespace sf {

class App {
public:
    void init(bool self_test = false);
    void run();
    void shutdown();

    void on_framebuffer_resize(int width, int height);
    void on_key(int key, int scancode, int action, int mods);
    void on_mouse_button(int button, int action, int mods);
    void on_cursor_pos(double xpos, double ypos);
    void on_scroll(double xoffset, double yoffset);

private:
    void switch_system(std::string_view system_name);
    void recreate_render_targets();
    void destroy_render_targets();
    void render_frame();
    void render_ui();
    void render_outliner_node(int body_idx);

    GLFWwindow* window_ = nullptr;
    rhi::VkContext vk_;
    ui::UiSystem ui_;
    ui::WorkspaceSettings ui_settings_{};

    ephemeris::SystemManager sys_mgr_{};
    ephemeris::Simulation simulation_{};
    rendering::RenderSettingsStore render_settings_{};
    ui::WorkspaceState ui_state_{};

    const ephemeris::SystemSnapshot& system() const noexcept { return simulation_.snapshot(); }
    std::vector<std::string> system_names_;

    Camera camera_{};
    rendering::StarCatalog star_catalog_{};
    rendering::CelestialRenderer celestial_renderer_{};
    rendering::OrbitRenderer orbit_renderer_{};

    rhi::Image hdr_color_{};
    rhi::Image scene_depth_{};
    rhi::Pipeline composite_pipeline_{};

    bool self_test_ = false;
    std::string simulation_error_;
    units::Seconds frame_dt_{};
};

} // namespace sf
