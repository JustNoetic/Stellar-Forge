#pragma once

#include "benchmark/benchmark.hpp"
#include "comparison/system_comparison.hpp"
#include "comparison/system_composer.hpp"
#include "comparison/body_metrics.hpp"
#include "core/camera.hpp"
#include "core/user_settings.hpp"
#include "ephemeris/system_manager.hpp"
#include "ephemeris/simulation_runner.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/bloom_renderer.hpp"
#include "rendering/celestial_renderer.hpp"
#include "rendering/star_catalog.hpp"
#include "rendering/orbit_renderer.hpp"
#include "rendering/size_comparator.hpp"
#include "rhi/shader_cache.hpp"
#include "rhi/vk_context.hpp"
#include "terrain/surface_baker.hpp"
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
    void on_drop(int count, const char** paths);

private:
    void toggle_fullscreen();
    void switch_system(std::string_view system_name);
    void recreate_render_targets();
    void destroy_render_targets();
    bool render_frame(benchmark::FrameSample* sample = nullptr);
    void run_benchmark();
    void save_hdr_capture(const std::filesystem::path& path);
    void render_ui();
    void render_main_menu_bar(rendering::RenderSettings& render_draft);
    void render_settings_window(rendering::RenderSettings& render_draft);
    UserSettings capture_preferences() const;
    void load_preferences();
    void save_preferences(bool force = false);
    void restore_preferences(rendering::RenderSettings& render_draft);
    void render_viewport_overlays();
    void render_outliner();
    void render_outliner_node(int body_idx);
    void render_body_inspector();
    void render_body_surface_tab(int insp_idx);
    void stage_surface_file(const std::filesystem::path& path, int target_tab = -1);
    void poll_surface_baker();
    void sync_system_height_ranges();
    bool save_body_surface_to_disk(const std::string& body_name, glm::dvec2 height_range_km, glm::vec3 color_rgb);
    void render_time_hud(rendering::RenderSettings& render_draft);
    void set_size_comparison(bool active);
    void apply_size_comparison();
    void update_size_comparison();
    void render_comparison_outliner();
    void render_comparison_inspector();
    void render_comparison_labels();
    CommandResult add_system_comparison(std::string_view system_name);
    void remove_system_comparison(size_t slot);
    void advance_system_comparisons();
    void check_system_comparison_sync();
    void compose_scene();
    void respace_system_comparisons();
    CommandResult open_comparison_for_editing(size_t slot);
    void render_system_comparator_window();
    void render_compared_outliner();
    void render_compared_outliner_node(size_t slot, int local);
    void render_compared_inspector();
    void render_comparison_table();
    void render_add_body_modal();
    void render_delete_body_modal();
    void render_create_system_modal();
    void render_delete_system_modal();
    void render_export_nbody_modal();
    void render_ephem_setup_modal();
    void render_ephem_download_modal();
    void render_bsp_import_modal();
    void render_jump_timeline_modal();
    void open_export_nbody_modal();
    void open_jump_timeline_modal();
    void trigger_timeline_bsp_export();
    void trigger_ephemeris_mode();
    void apply_ephemeris_system();
    void open_generic_bsp_playback(const std::filesystem::path& bsp_path);
    void delete_body_immediate(int body_idx);

    GLFWwindow* window_ = nullptr;
    int windowed_x_ = 0;
    int windowed_y_ = 0;
    int windowed_width_ = 1600;
    int windowed_height_ = 900;
    bool windowed_maximized_ = false;
    rhi::VkContext vk_;
    ui::UiSystem ui_;
    ui::WorkspaceSettings ui_settings_{};

    ephemeris::SystemManager sys_mgr_{};
    ephemeris::SpiceKernelManager spice_mgr_{};
    ephemeris::SimulationRunner simulation_{};
    comparison::ComparisonSet comparisons_{};
    comparison::SystemComposer composer_{};
    uint64_t comparison_generation_ = 0;
    struct SceneSelection {
        int index = -1;
        uint64_t identity = 0;
    };
    SceneSelection tracked_memory_{}, centered_memory_{}, inspected_memory_{};
    bool show_system_comparator_ = false;
    bool show_comparison_table_ = false;
    int comparison_metric_ = 0;
    comparison::MatchMode comparison_match_ = comparison::MatchMode::OrbitOrder;
    std::string comparison_table_row_;
    int comparator_candidate_ = 0;
    std::string comparator_error_;
    rendering::RenderSettingsStore render_settings_{};
    UserSettings saved_preferences_{};
    UserSettings pending_preferences_{};
    double preferences_save_at_ = 0.0;
    bool preferences_writable_ = true;
    bool preferences_reset_requested_ = false;
    std::string preferences_notice_;
    std::string preferences_save_error_;
    ui::WorkspaceState ui_state_{};

    const ephemeris::SystemSnapshot& system() const noexcept { return simulation_.snapshot(); }
    const ephemeris::SystemSnapshot& scene() const noexcept { return comparisons_.empty() ? system() : composer_.scene(); }
    std::vector<std::string> system_names_;

    Camera camera_{};
    rendering::StarCatalog star_catalog_{};
    rendering::CelestialRenderer celestial_renderer_{};
    rendering::OrbitRenderer orbit_renderer_{};
    rendering::BloomRenderer bloom_renderer_{};
    rendering::SizeComparatorRenderer size_renderer_{};
    comparison::SizeComparison size_comparison_{};
    comparison::SizeView size_view_{};
    bool size_comparison_active_ = false;
    bool size_comparison_requested_ = false;
    bool size_comparison_initialized_ = false;
    double comparison_refresh_at_ = 0;
    int comparison_focus_ = -1;

    rhi::Image hdr_color_{};
    rhi::Image scene_depth_{};
    rhi::Pipeline composite_pipeline_{};
    rhi::Pipeline bloom_composite_pipeline_{};
    rhi::Pipeline size_composite_pipeline_{};
    terrain::SurfaceBaker surface_baker_{};
    rhi::Image surface_preview_img_{};
    rhi::Buffer surface_preview_staging_{};
    VkDescriptorSet surface_preview_ds_ = VK_NULL_HANDLE;
    bool surface_preview_dirty_ = false;

    bool self_test_ = false;
    benchmark::Options benchmark_;
    std::string simulation_error_;
    units::Seconds frame_dt_{};
    int pending_delete_body_ = -1;
};

} // namespace sf
