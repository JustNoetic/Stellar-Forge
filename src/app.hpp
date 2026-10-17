#pragma once

#include "benchmark/benchmark.hpp"
#include "core/camera.hpp"
#include "ephemeris/system_manager.hpp"
#include "ephemeris/simulation_runner.hpp"
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
    void init(bool self_test = false, benchmark::Options benchmark = {}, bool simulation_thread = true);
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
    bool render_frame(benchmark::FrameSample* sample = nullptr);
    void run_benchmark();
    void render_ui();
    void render_main_menu_bar(rendering::RenderSettings& render_draft);
    void render_settings_window(rendering::RenderSettings& render_draft);
    void render_viewport_overlays();
    void render_outliner();
    void render_outliner_node(int body_idx);
    void render_body_inspector();
    void render_time_hud(rendering::RenderSettings& render_draft);
    void render_add_body_modal();
    void render_delete_body_modal();
    void render_create_system_modal();
    void render_delete_system_modal();
    void delete_body_immediate(int body_idx);

    GLFWwindow* window_ = nullptr;
    rhi::VkContext vk_;
    ui::UiSystem ui_;
    ui::WorkspaceSettings ui_settings_{};

    ephemeris::SystemManager sys_mgr_{};
    ephemeris::SimulationRunner simulation_{};
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
    benchmark::Options benchmark_;
    std::string simulation_error_;
    units::Seconds frame_dt_{};
    int pending_delete_body_ = -1;
};

} // namespace sf
