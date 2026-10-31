#include "app.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"
#include "core/time_utils.hpp"
#include "rendering/ring_optics.hpp"
#include "rendering/ring_generator.hpp"

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <exception>
#include <format>
#include <stdexcept>

namespace sf {

namespace {

static std::string trim_trailing_zeros(std::string s) {
    auto dot = s.find('.');
    if (dot != std::string::npos) {
        auto end = s.find_first_not_of("0123456789", dot + 1);
        size_t num_end = (end != std::string::npos) ? end : s.size();
        size_t last = num_end - 1;
        while (last > dot && s[last] == '0') {
            --last;
        }
        if (s[last] == '.') {
            --last;
        }
        s.erase(last + 1, num_end - (last + 1));
    }
    return s;
}

std::string format_distance_au(double dist_au, double threshold_au = DEFAULT_LY_THRESHOLD_AU) {
    double dist_km = dist_au * AU_TO_KM;
    if (dist_au >= threshold_au) {
        return trim_trailing_zeros(std::format("{:.3f} ly", dist_au * AU_TO_LY));
    }
    if (dist_au >= 0.05) {
        return trim_trailing_zeros(std::format("{:.4f} AU", dist_au));
    }
    if (dist_km >= 1.0) {
        return trim_trailing_zeros(std::format("{:.2f} km", dist_km));
    }
    return trim_trailing_zeros(std::format("{:.1f} m", dist_km * 1000.0));
}

std::string format_altitude(double alt_km, double alt_au, double threshold_au = DEFAULT_LY_THRESHOLD_AU) {
    if (alt_au >= threshold_au) {
        return trim_trailing_zeros(std::format("{:.3f} ly", alt_au * AU_TO_LY));
    }
    if (alt_km > 1.49597e7) {
        return trim_trailing_zeros(std::format("{:.4f} AU", alt_au));
    }
    if (alt_km >= 1000.0) {
        return std::format("{:.0f} km", alt_km);
    }
    if (alt_km >= 1.0) {
        return std::format("{:.1f} km", alt_km);
    }
    return std::format("{:.0f} m", alt_km * 1000.0);
}

std::string format_speed_au_s(double speed_au_s) {
    double km_s = speed_au_s * AU_TO_KM;
    if (speed_au_s >= LY_TO_AU * 0.01) {
        return trim_trailing_zeros(std::format("{:.2f} ly/s", speed_au_s * AU_TO_LY));
    }
    if (speed_au_s >= 0.01) {
        return trim_trailing_zeros(std::format("{:.3f} AU/s", speed_au_s));
    }
    if (km_s >= 1.0) {
        return trim_trailing_zeros(std::format("{:.2f} km/s", km_s));
    }
    return trim_trailing_zeros(std::format("{:.1f} m/s", km_s * 1000.0));
}

} // namespace

void App::init(bool self_test, benchmark::Options benchmark, bool simulation_thread) {
    self_test_ = self_test;
    benchmark_ = std::move(benchmark);

    if (!glfwInit()) {
        sf::log::fatal("Failed to initialize GLFW.");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (self_test_ || (benchmark_.enabled && benchmark_.hidden)) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    }
    if (benchmark_.enabled) glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window_ = glfwCreateWindow(benchmark_.enabled ? benchmark_.width : 1600,
                               benchmark_.enabled ? benchmark_.height : 900, "Stellar Forge", nullptr, nullptr);
    if (!window_) {
        sf::log::fatal("Failed to create GLFW window.");
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int width, int height) {
        static_cast<App*>(glfwGetWindowUserPointer(w))->on_framebuffer_resize(width, height);
    });
    glfwSetKeyCallback(window_, [](GLFWwindow* w, int key, int scancode, int action, int mods) {
        static_cast<App*>(glfwGetWindowUserPointer(w))->on_key(key, scancode, action, mods);
    });
    glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int button, int action, int mods) {
        static_cast<App*>(glfwGetWindowUserPointer(w))->on_mouse_button(button, action, mods);
    });
    glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double xpos, double ypos) {
        static_cast<App*>(glfwGetWindowUserPointer(w))->on_cursor_pos(xpos, ypos);
    });
    glfwSetScrollCallback(window_, [](GLFWwindow* w, double xoff, double yoff) {
        static_cast<App*>(glfwGetWindowUserPointer(w))->on_scroll(xoff, yoff);
    });
    glfwSetDropCallback(window_, [](GLFWwindow* w, int count, const char** paths) {
        static_cast<App*>(glfwGetWindowUserPointer(w))->on_drop(count, paths);
    });

    rhi::ShaderCache::init();
    vk_.init(window_, false);
    ui_.init(window_, vk_);
    {
        auto init_preview = terrain::SurfaceBaker::build_solid_preview(0.14f, 0.16f, 0.20f);
        surface_preview_img_ = vk_.create_image_2d(
            terrain::SURFACE_PREVIEW_W, terrain::SURFACE_PREVIEW_H,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        vk_.upload_image_2d(surface_preview_img_, init_preview.data(), init_preview.size());
        surface_preview_staging_ = vk_.create_host_buffer(
            init_preview.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, init_preview.data());
        surface_preview_ds_ = ImGui_ImplVulkan_AddTexture(
            vk_.sampler_linear_clamp(), surface_preview_img_.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    if (self_test_ || benchmark_.enabled) {
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::GetIO().LogFilename = nullptr;
    }
    if (benchmark_.enabled) {
        ui_settings_.ui_visible = benchmark_.ui;
    }

    recreate_render_targets();

    system_names_ = sys_mgr_.list_systems();
    switch_system(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
    load_preferences();
    simulation_.set_threaded(benchmark_.enabled ? benchmark_.simulation_mode != "inline" : simulation_thread);
    comparisons_.set_threaded(simulation_.threaded());
    sf::log::info("Simulation execution: {}", simulation_.threaded() ? "worker" : "inline");

    star_catalog_.init(vk_, sf::paths::external_path("data", "gaia", "stars.bin"));
    celestial_renderer_.init(vk_);
    sync_system_height_ranges();
    orbit_renderer_.init(vk_);
    bloom_renderer_.init(vk_);

    rhi::GraphicsPipelineDesc comp_desc{};
    comp_desc.vert_path = "post/fullscreen.vert";
    comp_desc.frag_path = "post/composite.frag";
    comp_desc.bindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };
    comp_desc.color_formats = {vk_.swapchain_format()};
    comp_desc.depth_format = VK_FORMAT_UNDEFINED;
    comp_desc.depth_test = false;
    comp_desc.depth_write = false;
    comp_desc.cull_mode = VK_CULL_MODE_NONE;

    composite_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(vk_, comp_desc);
    comp_desc.macros = {{"SIZE_COMPARISON", "1"}};
    size_composite_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(vk_, comp_desc);
    sf::log::info("Stellar Forge initialized (Root: {})", sf::paths::project_root().string());
}

void App::switch_system(std::string_view system_name) {
    if (system_name == "Ephemeris Mode" && !self_test_ && !benchmark_.enabled) {
        trigger_ephemeris_mode();
        return;
    }
    CommandResult result;
    try {
        result = simulation_.replace_system(sys_mgr_.load_system(system_name));
    } catch (const std::exception& error) {
        result = {false, false, error.what()};
    }
    if (!result) {
        sf::log::error("Cannot switch system: {}", result.error);
        simulation_error_ = result.error;
        return;
    }
    simulation_error_.clear();
    ui_state_ = {};
    camera_.tracking_idx = system().star_index();
    ui_state_.inspected_body = system().find_body("Earth");
    if (ui_state_.inspected_body < 0) {
        ui_state_.inspected_body = system().star_index();
    }
    camera_.cam_pos_rel = {0.0, 19.02, 40.78};
    camera_.up = {0.0, 1.0, 0.0};
    camera_.cam_look_aim = true;
    camera_.flight_speed_au_s = NavigationSettings{}.flight_speed_au_s;
    camera_.has_prev_spin = false;
    camera_.origin_render = system().body(camera_.tracking_idx).render.position.value;
    sync_system_height_ranges();
}

void App::trigger_ephemeris_mode() {
    if (!spice_mgr_.settings_initialized()) {
        ui_state_.ephem_setup.reset();
        return;
    }
    if (!spice_mgr_.check_missing_kernels().empty()) {
        ui_state_.show_ephem_download_modal = true;
        spice_mgr_.start_download_missing();
        return;
    }
    apply_ephemeris_system();
}

void App::apply_ephemeris_system() {
    bool was_spice = (!system().empty() && system().definition().motion_model == ephemeris::MotionModel::Spice &&
                      system().definition().custom_bsp_path.empty());
    std::string prev_tracked_name;
    std::string prev_inspected_name;
    if (was_spice) {
        if (camera_.tracking_idx >= 0 && static_cast<size_t>(camera_.tracking_idx) < system().size()) {
            prev_tracked_name = system().body(camera_.tracking_idx).definition.name;
        }
        if (ui_state_.inspected_body >= 0 && static_cast<size_t>(ui_state_.inspected_body) < system().size()) {
            prev_inspected_name = system().body(ui_state_.inspected_body).definition.name;
        }
    }

    ephemeris::LoadedSystem tmpl{};
    try {
        tmpl = sys_mgr_.load_system(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
    } catch (const std::exception& e) {
        simulation_error_ = e.what();
        return;
    }

    std::string err;
    auto ephem_loaded = spice_mgr_.build_ephemeris_system(&tmpl.definition, tmpl.definition.epoch_utc, err);
    if (ephem_loaded.definition.bodies.empty()) {
        sf::log::error("Cannot build ephemeris system: {}", err);
        simulation_error_ = err;
        return;
    }

    auto result = simulation_.replace_system(std::move(ephem_loaded));
    if (!result) {
        sf::log::error("Cannot switch to ephemeris system: {}", result.error);
        simulation_error_ = result.error;
        return;
    }
    simulation_error_.clear();

    if (was_spice) {
        int trk = prev_tracked_name.empty() ? -1 : system().find_body(prev_tracked_name);
        camera_.tracking_idx = (trk >= 0) ? trk : system().star_index();
        int insp = prev_inspected_name.empty() ? -1 : system().find_body(prev_inspected_name);
        ui_state_.inspected_body = (insp >= 0) ? insp : system().find_body("Earth");
        if (ui_state_.inspected_body < 0) ui_state_.inspected_body = system().star_index();
        camera_.origin_render = system().body(camera_.tracking_idx).render.position.value;
    } else {
        ui_state_ = {};
        camera_.tracking_idx = system().star_index();
        ui_state_.inspected_body = system().find_body("Earth");
        if (ui_state_.inspected_body < 0) ui_state_.inspected_body = system().star_index();
        camera_.cam_pos_rel = {0.0, 19.02, 40.78};
        camera_.up = {0.0, 1.0, 0.0};
        camera_.cam_look_aim = true;
        camera_.flight_speed_au_s = NavigationSettings{}.flight_speed_au_s;
        camera_.has_prev_spin = false;
        camera_.origin_render = system().body(camera_.tracking_idx).render.position.value;
    }
}

void App::open_generic_bsp_playback(const std::filesystem::path& bsp_path) {
    std::string err;
    auto loaded = spice_mgr_.build_generic_bsp_system(bsp_path, err);
    if (loaded.definition.bodies.empty()) {
        simulation_error_ = err;
        ui_state_.bsp_import.error = err;
        return;
    }
    auto result = simulation_.replace_system(std::move(loaded));
    if (!result) {
        simulation_error_ = result.error;
        ui_state_.bsp_import.error = result.error;
        return;
    }
    simulation_error_.clear();
    ui_state_ = {};

    int focus_idx = 0;
    for (size_t i = 0; i < system().size(); ++i) {
        if (system().body(i).definition.is_spacecraft) {
            focus_idx = static_cast<int>(i);
            break;
        }
        if (!system().body(i).definition.is_star && focus_idx == 0) {
            focus_idx = static_cast<int>(i);
        }
    }
    ui_state_.inspected_body = focus_idx;
    camera_.focus_body(focus_idx, system(), true, false);
}

void App::run() {
    if (benchmark_.enabled) {
        run_benchmark();
        return;
    }
    if (self_test_) {
        camera_.focus_body(system().find_body("Earth"), system(), true);
        auto result = simulation_.apply(ephemeris::SetTimeWarp{86400.0});
        if (!result) throw std::runtime_error(result.error);
    }
    double last_time = glfwGetTime();
    uint32_t frames_rendered = 0;
    uint64_t compared_identity = 0;

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        double now = glfwGetTime();
        frame_dt_ = units::Seconds{self_test_ ? 1.0 / 1700.0 : std::clamp(now - last_time, 1e-5, 0.25)};
        last_time = now;
        auto step = simulation_.advance_async(frame_dt_);
        if (!step) {
            simulation_error_ = step.error;
            simulation_.apply(ephemeris::SetPaused{true});
            sf::log::error("Simulation paused: {}", step.error);
        }
        advance_system_comparisons();
        compose_scene();

        if (self_test_) {
            if (!comparisons_.empty()) check_system_comparison_sync();
            if (frames_rendered == 1) {
                auto result = add_system_comparison(system().name());
                if (!result) throw std::runtime_error(result.error);
            } else if (frames_rendered == 2 && !comparisons_.empty()) {
                int earth = composer_.map().composite(1, size_t(std::max(0, comparisons_.slot(0).snapshot().find_body("Earth"))));
                if (earth < int(system().size())) throw std::runtime_error("Compared bodies must follow the primary bodies.");
                camera_.focus_body(earth, scene(), true);
                ui_state_.inspected_body = earth;
                compared_identity = scene().body(size_t(earth)).revisions.identity;
            } else if (frames_rendered == 3 && !comparisons_.empty()) {
                comparisons_.slot(0).slot_lights_primary = true;
                compose_scene();
                size_t compared_star = size_t(composer_.map().begin(1)) + size_t(comparisons_.slot(0).snapshot().star_index());
                if (!scene().illuminates(compared_star, 0) || scene().illuminates(size_t(system().star_index()), compared_star + 1))
                    throw std::runtime_error("Starlight switches do not light the requested direction only.");
            } else if (frames_rendered == 5 && !comparisons_.empty()) {
                comparisons_.set_overlay(true);
                compose_scene();
                if (scene().illuminates(size_t(composer_.map().begin(1)), 0) || comparisons_.slot(0).slot_lights_primary ||
                    !comparisons_.slot(0).primary_lights_slot)
                    throw std::runtime_error("Overlay must isolate lighting and the swap must keep switch directions.");
            } else if (frames_rendered == 6 && !comparisons_.empty()) {
                show_system_comparator_ = show_comparison_table_ = true;
                comparison_match_ = comparison::MatchMode::OrbitOrder;
                comparison_table_row_ = "Planet 3";
            } else if (frames_rendered == 7 && !comparisons_.empty()) {
                comparisons_.set_overlay(false);
                compose_scene();
                comparison_match_ = comparison::MatchMode::Name;
                comparison_metric_ = 5;
                comparison_table_row_ = "Earth";
            } else if (frames_rendered == 9) {
                show_system_comparator_ = show_comparison_table_ = false;
            } else if (frames_rendered == 4 && compared_identity != 0) {
                auto result = open_comparison_for_editing(0);
                if (!result) throw std::runtime_error(result.error);
                if (camera_.tracking_idx < 0 || camera_.tracking_idx >= int(system().size()) ||
                    system().body(size_t(camera_.tracking_idx)).revisions.identity != compared_identity ||
                    ui_state_.inspected_body != camera_.tracking_idx)
                    throw std::runtime_error("Open for Editing lost the tracked and inspected compared body.");
            }
            if (compared_identity != 0 && !comparisons_.empty() &&
                scene().body(size_t(camera_.tracking_idx)).revisions.identity != compared_identity)
                throw std::runtime_error("Camera tracking lost the compared body.");
            uint32_t settings_frames = static_cast<uint32_t>(ui::SettingsSection::Count);
            ui_settings_.show_settings = frames_rendered < settings_frames;
            ui_settings_.settings_section = static_cast<int>(frames_rendered);
            auto settings = render_settings_.values();
            settings.bloom_mode = frames_rendered % 3;
            render_settings_.apply(settings);
            if (frames_rendered == settings_frames) {
                set_size_comparison(true);
                size_comparison_.selected_systems = sys_mgr_.list_systems();
            } else if (frames_rendered == settings_frames+1) {
                size_comparison_.select(3);
                comparison_focus_ = size_comparison_.selection();
            } else if (frames_rendered == settings_frames+2) {
                size_comparison_.selected_systems.clear();
            } else if (frames_rendered == settings_frames+3) {
                size_comparison_.selected_systems = sys_mgr_.list_systems();
            } else if (frames_rendered == settings_frames+4) {
                set_size_comparison(false);
                if (!comparisons_.empty()) remove_system_comparison(0);
            }
        }
        render_frame();
        save_preferences();

        ++frames_rendered;
        if (self_test_ && frames_rendered >= static_cast<uint32_t>(ui::SettingsSection::Count)+5) {
            sf::log::info("Self-test completed {} frames cleanly.", frames_rendered);
            break;
        }
    }
}

void App::shutdown() {
    save_preferences(true);
    comparisons_.clear();
    simulation_.set_threaded(false);
    vk_.wait_idle();
    rhi::ShaderCache::destroy_pipeline(vk_, size_composite_pipeline_);
    rhi::ShaderCache::destroy_pipeline(vk_, composite_pipeline_);
    if (surface_preview_ds_ != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(surface_preview_ds_);
        surface_preview_ds_ = VK_NULL_HANDLE;
    }
    vk_.destroy_buffer(surface_preview_staging_);
    vk_.destroy_image(surface_preview_img_);
    size_renderer_.shutdown(vk_);
    celestial_renderer_.shutdown(vk_);
    orbit_renderer_.shutdown(vk_);
    bloom_renderer_.shutdown(vk_);
    star_catalog_.shutdown(vk_);
    destroy_render_targets();
    ui_.shutdown(vk_);
    vk_.shutdown();
    rhi::ShaderCache::shutdown();

    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

void App::on_framebuffer_resize(int, int) {
    vk_.request_swapchain_resize();
}

void App::toggle_fullscreen() {
    if (glfwGetWindowMonitor(window_)) {
        glfwSetWindowMonitor(window_, nullptr, windowed_x_, windowed_y_,
                             windowed_width_, windowed_height_, GLFW_DONT_CARE);
        if (windowed_maximized_) glfwMaximizeWindow(window_);
        return;
    }

    glfwGetWindowPos(window_, &windowed_x_, &windowed_y_);
    glfwGetWindowSize(window_, &windowed_width_, &windowed_height_);
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    int monitor_count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
    int64_t best_overlap = 0;
    for (int i = 0; i < monitor_count; ++i) {
        const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
        if (!mode) continue;
        int x = 0, y = 0;
        glfwGetMonitorPos(monitors[i], &x, &y);
        int overlap_width = std::max(0, std::min(windowed_x_ + windowed_width_, x + mode->width) - std::max(windowed_x_, x));
        int overlap_height = std::max(0, std::min(windowed_y_ + windowed_height_, y + mode->height) - std::max(windowed_y_, y));
        int64_t overlap = int64_t(overlap_width) * overlap_height;
        if (overlap > best_overlap) {
            best_overlap = overlap;
            monitor = monitors[i];
        }
    }
    const GLFWvidmode* mode = monitor ? glfwGetVideoMode(monitor) : nullptr;
    if (!mode) return;

    windowed_maximized_ = glfwGetWindowAttrib(window_, GLFW_MAXIMIZED) == GLFW_TRUE;
    if (windowed_maximized_) {
        glfwRestoreWindow(window_);
        glfwGetWindowPos(window_, &windowed_x_, &windowed_y_);
        glfwGetWindowSize(window_, &windowed_width_, &windowed_height_);
    }
    glfwSetWindowMonitor(window_, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
}

void App::on_key(int key, int, int action, int mods) {
    if (benchmark_.enabled) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window_, GLFW_TRUE);
        return;
    }
    if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {
        toggle_fullscreen();
        return;
    }
    if (key == GLFW_KEY_H && action == GLFW_PRESS &&
        (!ImGui::GetCurrentContext() || !ImGui::GetIO().WantTextInput)) {
        ui_settings_.ui_visible = !ui_settings_.ui_visible;
        return;
    }

    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard) {
        camera_.key_w = camera_.key_a = camera_.key_s = camera_.key_d = false;
        camera_.key_q = camera_.key_e = false;
        return;
    }

    if (size_comparison_active_) {
        if (key == GLFW_KEY_F && action == GLFW_PRESS) { size_view_.fit = true; return; }
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) { size_comparison_.select(-1); ui_state_.comparison_selection = false; return; }
        if (key == GLFW_KEY_W || key == GLFW_KEY_A || key == GLFW_KEY_S || key == GLFW_KEY_D || key == GLFW_KEY_Q || key == GLFW_KEY_E) return;
    }
    bool pressed = (action != GLFW_RELEASE);
    if (key == GLFW_KEY_W) { camera_.key_w = pressed; return; }
    if (key == GLFW_KEY_A) { camera_.key_a = pressed; return; }
    if (key == GLFW_KEY_S) { camera_.key_s = pressed; return; }
    if (key == GLFW_KEY_D) { camera_.key_d = pressed; return; }
    if (key == GLFW_KEY_Q) { camera_.key_q = pressed; return; }
    if (key == GLFW_KEY_E) { camera_.key_e = pressed; return; }

    if (action != GLFW_PRESS && action != GLFW_REPEAT) {
        return;
    }

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        simulation_.apply(ephemeris::SetPaused{!simulation_.clock().paused});
    } else if (key == GLFW_KEY_F && action == GLFW_PRESS) {
        camera_.snap_to_tracked(scene());
    } else if (key == GLFW_KEY_R) {
        simulation_.apply(ephemeris::SetTimeWarp{1.0});
    } else if (key == GLFW_KEY_BACKSPACE && action == GLFW_PRESS) {
        simulation_.apply(ephemeris::ReverseTimeDirection{});
    } else if (key == GLFW_KEY_UP || key == GLFW_KEY_RIGHT) {
        simulation_.apply(ephemeris::SetTimeWarp{std::min(simulation_.clock().warp * 2.0, 1e16)});
    } else if (key == GLFW_KEY_DOWN || key == GLFW_KEY_LEFT) {
        simulation_.apply(ephemeris::SetTimeWarp{std::max(simulation_.clock().warp * 0.5, 1e-4)});
    } else if (key == GLFW_KEY_EQUAL) {
        float step = (mods & GLFW_MOD_SHIFT) ? 2.0f : 1.15f;
        auto settings = render_settings_.values();
        settings.exposure *= step;
        render_settings_.apply(settings);
    } else if (key == GLFW_KEY_MINUS) {
        float step = (mods & GLFW_MOD_SHIFT) ? 2.0f : 1.15f;
        auto settings = render_settings_.values();
        settings.exposure = std::max(settings.exposure / step, 1e-12f);
        render_settings_.apply(settings);
    } else if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        ui_state_.inspected_body = -1;
    }
}

void App::on_mouse_button(int button, int action, int) {
    if (benchmark_.enabled || size_comparison_active_) return;
    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window_, &x, &y);
    if (action == GLFW_RELEASE) {
        camera_.on_mouse_button(button, action, x, y);
        return;
    }
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    camera_.on_mouse_button(button, action, x, y);
}

void App::on_cursor_pos(double xpos, double ypos) {
    if (benchmark_.enabled || size_comparison_active_ || ui_.is_dragging()) return;
    camera_.on_cursor_pos(xpos, ypos, system());
}

void App::on_scroll(double, double yoffset) {
    if (benchmark_.enabled || size_comparison_active_) return;
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    bool shift = (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ||
                 (glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    camera_.on_scroll(yoffset, shift);
}

void App::on_drop(int count, const char** paths) {
    if (benchmark_.enabled || count <= 0 || !paths || !paths[0]) return;
    if (!ui_state_.has_selection(system())) return;
    int insp_idx = ui_state_.inspected_body;
    const auto& b = system().body(insp_idx);
    if (b.definition.is_star || b.definition.is_black_hole || b.definition.is_spacecraft) return;

    std::filesystem::path dropped(reinterpret_cast<const char8_t*>(paths[0]));
    std::string stem_lower = dropped.stem().string();
    for (char& c : stem_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    int target_tab = ui_state_.surface_edit.selected_tab;
    if (stem_lower.find("height") != std::string::npos ||
        stem_lower.find("dem") != std::string::npos ||
        stem_lower.find("elev") != std::string::npos ||
        stem_lower.find("bump") != std::string::npos) {
        target_tab = 1;
    }
    ui_state_.surface_edit.selected_tab = target_tab;
    ui_state_.surface_edit.focus_surface_tab = true;
    stage_surface_file(dropped, target_tab);
}

void App::stage_surface_file(const std::filesystem::path& path, int target_tab) {
    auto& se = ui_state_.surface_edit;
    int tab = (target_tab >= 0) ? target_tab : se.selected_tab;
    auto info_opt = terrain::SurfaceBaker::inspect_and_load_image(path);
    if (!info_opt) {
        se.error_msg = std::format("Failed to decode image: {}", path.filename().string());
        se.status_msg.clear();
        return;
    }
    auto& stage = (tab == 1) ? se.height_stage : se.albedo_stage;
    stage.staged_path = info_opt->path.string();
    stage.width = info_opt->width;
    stage.height = info_opt->height;
    stage.recommended_lod = info_opt->recommended_lod;
    stage.selected_lod = info_opt->recommended_lod;
    stage.preview_rgba = std::move(info_opt->preview_rgba);
    stage.has_staged = true;
    se.invalidate_preview();
    se.error_msg.clear();
    se.status_msg = std::format("Staged {} ({}x{}, recommended LOD {}).",
                                path.filename().string(), stage.width, stage.height, stage.recommended_lod);
}

void App::sync_system_height_ranges() {
    for (size_t i = 0; i < system().size(); ++i) {
        const auto& b = system().body(i);
        if (b.definition.is_star || b.definition.is_black_hole || b.definition.is_spacecraft) continue;
        if (b.definition.height_range_km.x == 0.0 && b.definition.height_range_km.y == 0.0) {
            if (auto hs = celestial_renderer_.height_source(b.definition.name)) {
                glm::dvec2 range_km{hs->info().minimum_m * 0.001, hs->info().maximum_m * 0.001};
                if (range_km.x < range_km.y) {
                    simulation_.apply(ephemeris::ApplyBodySurface{
                        system().generation(), static_cast<int>(i), 0, range_km, b.definition.color_rgb
                    });
                }
            }
        }
    }
}

bool App::save_body_surface_to_disk(const std::string& body_name, glm::dvec2 height_range_km, glm::vec3 color_rgb) {
    std::string save_sys_name = system().name();
    if (save_sys_name == "Ephemeris Mode") {
        save_sys_name = std::string(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
    }
    if (save_sys_name.starts_with("Ephemeris: ") || !sys_mgr_.system_exists(save_sys_name)) {
        return false;
    }
    try {
        auto loaded = sys_mgr_.load_system(save_sys_name);
        int s_idx = loaded.definition.find_body(body_name);
        if (s_idx < 0) return false;
        loaded.definition.bodies[s_idx].height_range_km = height_range_km;
        loaded.definition.bodies[s_idx].color_rgb = color_rgb;
        loaded.definition.bodies[s_idx].color_pole_rgb = color_rgb;
        return sys_mgr_.save_system(save_sys_name, loaded.definition);
    } catch (...) {
        return false;
    }
}

void App::poll_surface_baker() {
    auto res_opt = surface_baker_.poll();
    if (!res_opt) return;
    const auto& res = *res_opt;
    auto& se = ui_state_.surface_edit;
    if (!res.success) {
        se.error_msg = res.error;
        se.status_msg.clear();
        return;
    }
    try {
        if (res.is_height) {
            auto source = terrain::HeightSource::open(res.pack_path);
            celestial_renderer_.set_height_source(res.body_name, source);
            size_renderer_.set_height_source(res.body_name, source);
            int b_idx = system().find_body(res.body_name);
            if (b_idx >= 0) {
                const auto& b = system().body(b_idx);
                glm::dvec2 range_km{res.height_min_km, res.height_max_km};
                simulation_.apply(ephemeris::ApplyBodySurface{
                    system().generation(), b_idx, 0, range_km, b.definition.color_rgb
                });
                save_body_surface_to_disk(res.body_name, range_km, b.definition.color_rgb);
            }
            se.height_stage.clear();
            se.height_stage.selected_lod = res.level;
            se.height_range_dirty = false;
        } else {
            auto source = terrain::ColorSource::open(res.pack_path);
            celestial_renderer_.set_color_source(res.body_name, source);
            size_renderer_.set_color_source(res.body_name, source);
            se.albedo_stage.clear();
            se.albedo_stage.selected_lod = res.level;
        }
        se.invalidate_preview();
        se.error_msg.clear();
        se.status_msg = std::format("Baked & applied {} {} (LOD 0..{}) on {} in {:.1f}s.",
                                    res.body_name,
                                    res.is_height ? "Elevation" : "Albedo",
                                    res.level,
                                    res.backend_used.empty() ? "CPU" : res.backend_used,
                                    res.elapsed_sec);
    } catch (const std::exception& e) {
        se.error_msg = std::format("Baked pack failed validation: {}", e.what());
        se.status_msg.clear();
    }
}

void App::recreate_render_targets() {
    VkExtent2D ext = vk_.swapchain_extent();
    if (hdr_color_.extent.width == ext.width && hdr_color_.extent.height == ext.height && hdr_color_) {
        return;
    }
    destroy_render_targets();

    hdr_color_ = vk_.create_image_2d(
        ext.width, ext.height, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    scene_depth_ = vk_.create_image_2d(
        ext.width, ext.height, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
}

void App::destroy_render_targets() {
    vk_.destroy_image(hdr_color_);
    vk_.destroy_image(scene_depth_);
}

bool App::render_frame(benchmark::FrameSample* sample) {
    poll_surface_baker();
    apply_size_comparison();
    if (vk_.vsync_enabled() != render_settings_.values().vsync) {
        vk_.set_vsync(render_settings_.values().vsync);
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    uint32_t swapchain_idx = 0;
    if (!vk_.begin_frame(cmd, swapchain_idx, sample ? &sample->host : nullptr)) {
        return false;
    }

    auto record_start = sample ? SteadyClock::now() : SteadyClock::time_point{};
    uint32_t t_frame = sample ? vk_.begin_gpu_timer(cmd, "gpu_frame") : UINT32_MAX;
    recreate_render_targets();

    {
        ScopedTimer timer(sample ? &sample->cpu_ui_ms : nullptr);
        ui_.begin_frame(ui_settings_);
        if (ui_settings_.ui_visible) {
            if (benchmark_.enabled) ImGui::BeginDisabled();
            render_ui();
            if (benchmark_.enabled) ImGui::EndDisabled();
        }
    }

    if (size_comparison_active_ && !ui_settings_.ui_visible) update_size_comparison();

    if (self_test_) {
        auto result = simulation_.apply(ephemeris::SetTimeWarp{simulation_.clock().warp * 1.25});
        if (!result) throw std::runtime_error(result.error);
    }
    compose_scene();
    double camera_update_ms = 0.0;
    {
        ScopedTimer timer(sample ? &camera_update_ms : nullptr);
        camera_.update(frame_dt_, scene());
    }
    if (sample) sample->cpu_camera_ms += camera_update_ms;
    if (self_test_ && camera_.origin_render != scene().body(camera_.tracking_idx).render.position.value)
        throw std::runtime_error("Camera tracking and rendered body use different snapshots after a time-warp change.");

    VkExtent2D ext = vk_.swapchain_extent();

    bool bloom_enabled = !size_comparison_active_ && rendering::BloomRenderer::enabled(render_settings_.values());
    rhi::Image* bloom_scene = nullptr;

    bool profile_scene = sample || ui_settings_.show_telemetry;
    uint32_t t_scene = vk_.begin_gpu_timer(cmd, "gpu_scene_hdr");
    vk_.transition_image(cmd, hdr_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    vk_.transition_image(cmd, scene_depth_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    if (size_comparison_active_) {
        size_renderer_.prepare(cmd, vk_, size_comparison_, size_view_, render_settings_.values(), ext);
        auto color_att = rhi::RenderingAttachment::clear_color(hdr_color_, .003f, .005f, .012f, 1);
        auto depth_att = rhi::RenderingAttachment::clear_depth_reverse_z(scene_depth_, 0);
        vk_.begin_rendering(cmd, ext, std::span{&color_att, 1}, &depth_att);
        size_renderer_.render(cmd, vk_);
        vk_.end_rendering(cmd);
        size_renderer_.composite(cmd, vk_, hdr_color_, scene_depth_);
    } else {
        double celestial_prepare_ms = 0.0;
        {
            ScopedTimer timer(sample ? &celestial_prepare_ms : nullptr);
            uint32_t id = profile_scene ? vk_.begin_gpu_timer(cmd, "gpu_terrain_prepare") : UINT32_MAX;
            celestial_renderer_.prepare(cmd, vk_, camera_, scene(), render_settings_.values(), ext, ui_settings_.show_telemetry, sample ? &sample->cpu : nullptr);
            vk_.end_gpu_timer(cmd, id);
        }
        auto refraction_writes = celestial_renderer_.refraction_writes(vk_);
        if (!celestial_renderer_.transparency().active() && render_settings_.values().show_orbits && scene().size() > 1)
            bloom_enabled = false;
        if (bloom_enabled) {
            bloom_renderer_.prepare(vk_, ext);
            if (celestial_renderer_.transparency().active()) bloom_scene = &bloom_renderer_.physical_scene();
        }
        {
            ScopedTimer timer(sample ? &sample->cpu["cpu_gaia_cull_ms"] : nullptr);
            auto id = profile_scene ? vk_.begin_gpu_timer(cmd, "gpu_gaia_cull") : UINT32_MAX;
            star_catalog_.cull(cmd, vk_, camera_, system().time(), render_settings_.values(), ext, refraction_writes);
            vk_.end_gpu_timer(cmd, id);
        }

        auto color_att = rhi::RenderingAttachment::clear_color(hdr_color_, 0.0f, 0.0f, 0.0f, 1.0f);
        auto depth_att = rhi::RenderingAttachment::clear_depth_reverse_z(scene_depth_, 0.0f);
        vk_.begin_rendering(cmd, ext, std::span{&color_att, 1}, &depth_att);

        {
            ScopedTimer timer(sample ? &sample->cpu_celestial_ms : nullptr);
            uint32_t id = profile_scene ? vk_.begin_gpu_timer(cmd, "gpu_celestial") : UINT32_MAX;
            celestial_renderer_.render(cmd, vk_, camera_, scene(), render_settings_.values(), ext, sample ? &sample->cpu : nullptr, ui_settings_.show_telemetry);
            vk_.end_gpu_timer(cmd, id);
        }
        if (sample) sample->cpu_celestial_ms += celestial_prepare_ms;
        {
            ScopedTimer timer(sample ? &sample->cpu_gaia_ms : nullptr);
            uint32_t id = profile_scene ? vk_.begin_gpu_timer(cmd, "gpu_gaia") : UINT32_MAX;
            star_catalog_.render(cmd, vk_, camera_, system().time(), render_settings_.values(), ext, refraction_writes);
            vk_.end_gpu_timer(cmd, id);
        }
        {
            ScopedTimer timer(sample ? &sample->cpu_orbits_ms : nullptr);
            uint32_t id = profile_scene ? vk_.begin_gpu_timer(cmd, "gpu_orbits") : UINT32_MAX;
            auto transparency_writes = celestial_renderer_.transparency().writes();
            orbit_renderer_.render(cmd, vk_, camera_, scene(), render_settings_.values(), ext,
                celestial_renderer_.transparency().active() ? std::span{transparency_writes} : std::span<const rhi::PushDescriptorWrite>{},
                refraction_writes);
            vk_.end_gpu_timer(cmd, id);
        }

        vk_.end_rendering(cmd);
        {
            ScopedTimer timer(sample ? &sample->cpu["cpu_atmosphere_composite_ms"] : nullptr);
            celestial_renderer_.composite_atmospheres(cmd, vk_, hdr_color_, scene_depth_, bloom_scene);
        }
    }
    vk_.end_gpu_timer(cmd, t_scene);

    uint32_t t_post = vk_.begin_gpu_timer(cmd, "gpu_post_composite");
    if (bloom_enabled) bloom_renderer_.render(cmd, vk_, hdr_color_, render_settings_.values(), bloom_scene);
    uint32_t t_tonemap = vk_.begin_gpu_timer(cmd, "tonemap_composite");
    vk_.transition_image(cmd, hdr_color_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    vk_.transition_raw_image(cmd, vk_.swapchain_image(swapchain_idx), VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    rhi::RenderingAttachment swap_att{};
    swap_att.view = vk_.swapchain_view(swapchain_idx);
    swap_att.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    swap_att.load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
    swap_att.store_op = VK_ATTACHMENT_STORE_OP_STORE;

    vk_.begin_rendering(cmd, ext, std::span{&swap_att, 1}, nullptr);
    const auto& presentation = size_comparison_active_ ? size_composite_pipeline_ : composite_pipeline_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, presentation.handle);

    rhi::PushDescriptorWrite writes[] = {
        rhi::PushDescriptorWrite::sampled_image(0, hdr_color_, vk_.sampler_linear_clamp()),
    };
    vk_.push_descriptors(cmd, presentation, writes);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vk_.end_rendering(cmd);
    vk_.end_gpu_timer(cmd, t_tonemap);
    vk_.end_gpu_timer(cmd, t_post);

    uint32_t t_ui = vk_.begin_gpu_timer(cmd, "gpu_ui");
    if (surface_preview_dirty_ && surface_preview_img_ && surface_preview_staging_) {
        surface_preview_dirty_ = false;
        vk_.transition_image(cmd, surface_preview_img_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = surface_preview_img_.extent;
        vkCmdCopyBufferToImage(cmd, surface_preview_staging_.handle, surface_preview_img_.handle,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        vk_.transition_image(cmd, surface_preview_img_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    ui_.render(cmd, vk_, swapchain_idx);
    vk_.end_gpu_timer(cmd, t_ui);

    vk_.transition_raw_image(cmd, vk_.swapchain_image(swapchain_idx), VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);

    vk_.end_gpu_timer(cmd, t_frame);
    if (sample) sample->cpu_record_ms = elapsed_ms(record_start) - camera_update_ms;
    vk_.end_frame(cmd, swapchain_idx, sample ? &sample->host : nullptr);
    return true;
}

void App::render_outliner_node(int body_idx) {
    const auto& b = system().body(body_idx);
    const auto& children = system().reference_hierarchy().children[body_idx];
    bool has_children = !children.empty();

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth |
                               ImGuiTreeNodeFlags_AllowOverlap | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (!has_children) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }
    if (b.definition.is_star) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
    if (ui_state_.inspected_body == body_idx && ui_state_.has_selection(system())) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    std::string label = (camera_.tracking_idx == body_idx)
        ? std::format("{} [Tracked]", b.definition.name)
        : b.definition.name;

    bool can_delete = (body_idx != system().definition().star_idx && !b.definition.is_star && system().definition().motion_model != ephemeris::MotionModel::Spice);
    bool can_add = (system().definition().motion_model != ephemeris::MotionModel::Spice);

    float btn_w = 18.0f;
    float btn_h = 16.0f;
    float btn_spacing = 3.0f;
    float total_w = can_delete ? (btn_w * 2.0f + btn_spacing) : btn_w;
    float right_screen_x = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x - 4.0f;
    float btn_start_x = right_screen_x - total_w;
    float clip_right = btn_start_x - 4.0f;

    bool clip_row = can_add && (clip_right > ImGui::GetWindowPos().x + 30.0f);
    if (clip_row) {
        ImVec2 win_pos = ImGui::GetWindowPos();
        ImVec2 win_size = ImGui::GetWindowSize();
        ImGui::PushClipRect(ImVec2(win_pos.x, win_pos.y), ImVec2(clip_right, win_pos.y + win_size.y), true);
    }

    bool opened = ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<intptr_t>(body_idx)), flags, "%s", label.c_str());

    if (clip_row) {
        ImGui::PopClipRect();
    }

    ImVec2 node_min = ImGui::GetItemRectMin();
    ImVec2 node_max = ImGui::GetItemRectMax();
    float row_h = node_max.y - node_min.y;
    float btn_y = node_min.y + (row_h - btn_h) * 0.5f;

    bool is_hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenOverlappedByItem);
    bool is_selected = (ui_state_.inspected_body == body_idx);

    ImVec2 mouse_pos = ImGui::GetIO().MousePos;
    bool in_button_rect = (can_add && (is_hovered || is_selected) &&
                           mouse_pos.x >= btn_start_x - 2.0f && mouse_pos.x <= right_screen_x + 4.0f &&
                           mouse_pos.y >= node_min.y && mouse_pos.y <= node_max.y);

    if (ImGui::IsItemClicked() && !in_button_rect) {
        ui_state_.inspected_body = body_idx;
    }
    if (is_hovered && !in_button_rect && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        ui_state_.inspected_body = body_idx;
        camera_.focus_body(body_idx, system(), true);
    }

    if (can_add && (is_hovered || is_selected) && btn_start_x > node_min.x + 30.0f) {
        ImVec2 saved_cursor = ImGui::GetCursorPos();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
        ImGui::PushID(body_idx);

        ImGui::SetCursorScreenPos(ImVec2(btn_start_x, btn_y));
        if (ImGui::Button("+", ImVec2(btn_w, btn_h))) {
            ui_state_.add_body.reset_for_parent(system(), body_idx);
            ui_state_.add_body.active = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Add orbiting body to %s", b.definition.name.c_str());
        }

        if (can_delete) {
            ImGui::SetCursorScreenPos(ImVec2(btn_start_x + btn_w + btn_spacing, btn_y));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("-", ImVec2(btn_w, btn_h))) {
                if (ui_settings_.confirm_delete) {
                    ui_state_.delete_prompt.active = true;
                    ui_state_.delete_prompt.body_index = body_idx;
                    ui_state_.delete_prompt.body_name = b.definition.name;
                    int fallback = system().reference_parent(body_idx);
                    if (fallback < 0 || fallback == body_idx) fallback = system().definition().star_idx;
                    ui_state_.delete_prompt.fallback_parent_name = system().body(fallback).definition.name;
                } else {
                    pending_delete_body_ = body_idx;
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Delete %s", b.definition.name.c_str());
            }
            ImGui::PopStyleColor(3);
        }

        ImGui::PopID();
        ImGui::PopStyleVar();
        ImGui::SetCursorPos(saved_cursor);
    }

    if (has_children && opened) {
        ImGui::Indent();
        for (int child : children) render_outliner_node(child);
        ImGui::Unindent();
    }
}

void App::render_ui() {
    if (spice_mgr_.consume_download_completed()) {
        ui_state_.show_ephem_download_modal = false;
        apply_ephemeris_system();
    }

    auto render_draft = render_settings_.values();

    render_main_menu_bar(render_draft);
    render_outliner();
    update_size_comparison();
    render_body_inspector();
    render_viewport_overlays();
    render_settings_window(render_draft);
    render_time_hud(render_draft);
    render_system_comparator_window();
    render_comparison_table();
    render_add_body_modal();
    render_delete_body_modal();
    render_create_system_modal();
    render_delete_system_modal();
    render_export_nbody_modal();
    render_jump_timeline_modal();
    render_ephem_setup_modal();
    render_ephem_download_modal();
    render_bsp_import_modal();

    if (pending_delete_body_ >= 0) {
        delete_body_immediate(pending_delete_body_);
        pending_delete_body_ = -1;
    }

    auto settings_result = render_settings_.apply(render_draft);
    ui_state_.settings_error = settings_result.error;
}

void App::render_main_menu_bar(rendering::RenderSettings& render_draft) {
    if (!ImGui::BeginMainMenuBar()) return;

    if (ImGui::BeginMenu("Systems")) {
        for (const auto& sname : system_names_) {
            bool active = (system().name() == sname);
            if (ImGui::MenuItem(sname.c_str(), nullptr, active) && !active) {
                switch_system(sname);
            }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("+ Create New System...")) {
            ui_state_.create_system.reset();
        }
        if (system().definition().motion_model != ephemeris::MotionModel::NBody) {
            if (ImGui::MenuItem("Export to N-Body System...")) {
                open_export_nbody_modal();
            }
        }
        if (ImGui::MenuItem("Import Ephemeris Kernel (.bsp)...")) {
            ui_state_.bsp_import.reset();
        }
        bool can_delete_sys = (system().name() != ephemeris::SystemManager::SOLAR_SYSTEM_NAME &&
                               system().name() != "Ephemeris Mode" &&
                               !system().name().starts_with("Ephemeris: "));
        if (can_delete_sys) {
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
            if (ImGui::MenuItem("- Delete Current System")) {
                ui_state_.confirm_delete_system = true;
                ui_state_.delete_system_name = system().name();
            }
            ImGui::PopStyleColor();
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Edit", !size_comparison_active_)) {
        bool can_add = (system().definition().motion_model != ephemeris::MotionModel::Spice);
        if (ImGui::MenuItem("Add Orbiting Body...", nullptr, false, can_add)) {
            int p_idx = (ui_state_.inspected_body >= 0 && ui_state_.inspected_body < static_cast<int>(system().size()))
                ? ui_state_.inspected_body : system().definition().star_idx;
            ui_state_.add_body.reset_for_parent(system(), p_idx);
            ui_state_.add_body.active = true;
        }
        bool can_del = (ui_state_.inspected_body > 0 && ui_state_.inspected_body < static_cast<int>(system().size()) &&
                        ui_state_.inspected_body != system().definition().star_idx &&
                        !system().body(ui_state_.inspected_body).definition.is_star &&
                        system().definition().motion_model != ephemeris::MotionModel::Spice);
        if (ImGui::MenuItem("Delete Selected Body", nullptr, false, can_del)) {
            if (ui_settings_.confirm_delete) {
                ui_state_.delete_prompt.active = true;
                ui_state_.delete_prompt.body_index = ui_state_.inspected_body;
                ui_state_.delete_prompt.body_name = system().body(ui_state_.inspected_body).definition.name;
                int fallback = system().reference_parent(ui_state_.inspected_body);
                if (fallback < 0 || fallback == ui_state_.inspected_body) fallback = system().definition().star_idx;
                ui_state_.delete_prompt.fallback_parent_name = system().body(fallback).definition.name;
            } else {
                pending_delete_body_ = ui_state_.inspected_body;
            }
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Physics")) {
        auto current_mode = system().definition().motion_model;
        if (ImGui::MenuItem("N-Body (IAS15 Integrator)", nullptr, current_mode == ephemeris::MotionModel::NBody)) {
            auto result = simulation_.apply(ephemeris::SetMotionModel{ephemeris::MotionModel::NBody});
            if (!result) simulation_error_ = result.error;
            else simulation_error_.clear();
        }
        if (ImGui::MenuItem("Analytical Keplerian (Jacobi)", nullptr, current_mode == ephemeris::MotionModel::Keplerian)) {
            auto result = simulation_.apply(ephemeris::SetMotionModel{ephemeris::MotionModel::Keplerian});
            if (!result) simulation_error_ = result.error;
            else simulation_error_.clear();
        }
        bool can_spice = (system().name() == ephemeris::SystemManager::SOLAR_SYSTEM_NAME ||
                          system().name() == "Ephemeris Mode" ||
                          current_mode == ephemeris::MotionModel::Spice);
        if (ImGui::MenuItem("NASA SPICE Ephemeris Playback", nullptr, current_mode == ephemeris::MotionModel::Spice, can_spice)) {
            if (current_mode != ephemeris::MotionModel::Spice) {
                trigger_ephemeris_mode();
            }
        }
        if (ImGui::MenuItem("SPICE Ephemeris Kernel Settings...")) {
            ui_state_.ephem_setup.reset();
        }
        if (ImGui::MenuItem("Jump to Date / Render Timeline...")) {
            open_jump_timeline_modal();
        }
        if (current_mode != ephemeris::MotionModel::NBody) {
            ImGui::Separator();
            const char* export_physics_label = (current_mode == ephemeris::MotionModel::Spice)
                ? "Export Ephemeris to N-Body System..."
                : "Export Current State to N-Body System...";
            if (ImGui::MenuItem(export_physics_label)) {
                open_export_nbody_modal();
            }
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Tools")) {
        if (ImGui::MenuItem("Jump to Date / Render Timeline...")) {
            open_jump_timeline_modal();
        }
        if (ImGui::MenuItem("Export Timeline to SPK (.bsp)...")) {
            trigger_timeline_bsp_export();
        }
        if (system().definition().motion_model != ephemeris::MotionModel::NBody) {
            if (ImGui::MenuItem("Export to N-Body System...")) {
                open_export_nbody_modal();
            }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Import Ephemeris Kernel (.bsp)...")) {
            ui_state_.bsp_import.reset();
        }
        if (ImGui::MenuItem("SPICE Ephemeris Kernel Settings...")) {
            ui_state_.ephem_setup.reset();
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Navigation")) {
        if (ImGui::MenuItem("Free Flight Mode", nullptr, camera_.movement_mode == 0)) {
            camera_.movement_mode = 0;
        }
        if (ImGui::MenuItem("Simple Orbit Mode", nullptr, camera_.movement_mode == 1)) {
            camera_.movement_mode = 1;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Snap Look to Tracked (F)")) {
            camera_.snap_to_tracked(scene());
        }
        ImGui::MenuItem("Auto Horizon Level (< 200 km)", nullptr, &camera_.horizon_align);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Scene: Simulation", nullptr, !size_comparison_active_)) set_size_comparison(false);
        if (ImGui::MenuItem("Scene: Size Comparator", nullptr, size_comparison_active_)) set_size_comparison(true);
        ImGui::Separator();
        ImGui::MenuItem("System Outliner", nullptr, &ui_settings_.show_outliner);
        ImGui::MenuItem("Body Inspector", nullptr, &ui_settings_.show_inspector);
        ImGui::MenuItem("Time Transport HUD", nullptr, &ui_settings_.show_time_hud);
        ImGui::MenuItem("System Comparator...", nullptr, &show_system_comparator_);
        ImGui::MenuItem("Comparison Table...", nullptr, &show_comparison_table_);
        ImGui::Separator();
        ImGui::MenuItem("Engine Telemetry", nullptr, &ui_settings_.show_telemetry);
        ImGui::Separator();
        if (ImGui::MenuItem("Settings...")) {
            ui_settings_.show_settings = true;
        }
        ImGui::EndMenu();
    }

    const char* mode_badge = "Keplerian";
    if (system().definition().motion_model == ephemeris::MotionModel::Spice) mode_badge = "SPICE";
    else if (system().definition().motion_model == ephemeris::MotionModel::NBody) mode_badge = "IAS15 N-Body";
    ImGui::SameLine(0.0f, 15.0f);
    ImGui::TextDisabled("[%s]", mode_badge);

    float btn_w = ImGui::CalcTextSize(" Settings ").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float right_pos = ImGui::GetWindowWidth() - btn_w - ImGui::GetStyle().WindowPadding.x;
    if (right_pos > ImGui::GetCursorPosX()) {
        ImGui::SameLine(right_pos);
    }
    if (ImGui::Button(" Settings ")) {
        ui_settings_.show_settings = !ui_settings_.show_settings;
    }

    ImGui::EndMainMenuBar();
}

void App::render_viewport_overlays() {
    if (size_comparison_active_) render_comparison_labels();
    if (ui_settings_.show_telemetry) {
        ImGui::SetNextWindowSize(ImVec2(460.0f, 620.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Engine Telemetry", &ui_settings_.show_telemetry)) {
            ui::heading("GPU Telemetry", vk_.device_name());
            ui::value_row("Frame Rate", std::format("{:.1f} FPS", ImGui::GetIO().Framerate));
            const char* pm_str = "FIFO (VSync)";
            if (vk_.present_mode() == VK_PRESENT_MODE_IMMEDIATE_KHR) pm_str = "Immediate (Uncapped)";
            else if (vk_.present_mode() == VK_PRESENT_MODE_MAILBOX_KHR) pm_str = "Mailbox (Uncapped)";
            ui::value_row("Present Mode", pm_str);
            ui::value_row("Gaia Stars Loaded", std::format("{}", star_catalog_.count()));
            ui::value_row("GPU Scene HDR", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_scene_hdr")));
            ui::value_row("GPU Ringshine Bake", std::format("{:.3f} ms", vk_.gpu_time_ms("ringshine_bake")));
            ui::value_row("Ringshine Tiles Baked", std::format("{}", celestial_renderer_.ringshine_frame_bake_count()));
            bool bloom_active = rendering::BloomRenderer::enabled(render_settings_.values()) &&
                !size_comparison_active_ && (!render_settings_.values().show_orbits || scene().size() <= 1 || celestial_renderer_.transparency().active());
            bool bloom_spikes_active = bloom_active && render_settings_.values().bloom_mode == 2 && render_settings_.values().bloom_diffraction > 0.0f;
            ui::value_row("GPU Composite", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_post_composite")));
            ui::value_row("GPU Bloom", std::format("{:.3f} ms", bloom_active ? vk_.gpu_time_ms("bloom") : 0.0));
            if (bloom_active) {
                ui::value_row("GPU Bloom down / up", std::format("{:.3f} / {:.3f} ms", vk_.gpu_time_ms("bloom_downsample"), vk_.gpu_time_ms("bloom_upsample")));
                ui::value_row("GPU Bloom spikes / apply", std::format("{:.3f} / {:.3f} ms",
                    bloom_spikes_active ? vk_.gpu_time_ms("bloom_spikes") : 0.0, vk_.gpu_time_ms("bloom_apply")));
            }
            ui::value_row("GPU Tonemap", std::format("{:.3f} ms", vk_.gpu_time_ms("tonemap_composite")));
            ui::value_row("GPU Dear ImGui", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_ui")));
            const auto& settings = render_settings_.values();
            const auto& stats = celestial_renderer_.atmosphere_stats();
            bool oit_active = celestial_renderer_.transparency().active();
            ui::section("Scene breakdown", "These timings are included in GPU Scene HDR.");
            ui::value_row("GPU Preparation", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_terrain_prepare")));
            ui::value_row("GPU Celestial bodies", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_celestial")));
            ui::value_row("GPU Terrain", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_terrain_draw")));
            ui::value_row("GPU Spheres / points", std::format("{:.3f} / {:.3f} ms", vk_.gpu_time_ms("gpu_spheres"), vk_.gpu_time_ms("gpu_points")));
            ui::value_row("GPU Planetary rings", std::format("{:.3f} ms", settings.show_rings ? vk_.gpu_time_ms("rings") : 0.0));
            ui::value_row("GPU Gaia cull / draw", std::format("{:.3f} / {:.3f} ms", vk_.gpu_time_ms("gpu_gaia_cull"), vk_.gpu_time_ms("gpu_gaia")));
            ui::value_row("GPU Orbit lines", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_orbits")));
            ui::value_row("GPU Atmosphere composite", std::format("{:.3f} ms", stats.visible ? vk_.gpu_time_ms("atmosphere_composite") : 0.0));
            ui::value_row("GPU Atmo shadows / OIT / opaque", std::format("{:.3f} / {:.3f} / {:.3f} ms",
                stats.shadow_updates ? vk_.gpu_time_ms("atmosphere_shadows") : 0.0,
                stats.visible && oit_active ? vk_.gpu_time_ms("atmosphere_transparent") : 0.0,
                stats.visible ? vk_.gpu_time_ms("atmosphere_opaque") : 0.0));
            ui::value_row("GPU OIT clear / resolve", std::format("{:.3f} / {:.3f} ms",
                oit_active ? vk_.gpu_time_ms("transparency_clear") : 0.0,
                oit_active ? vk_.gpu_time_ms("transparency_resolve") : 0.0));
            ui::value_row("Sky / aerial LUT updates", std::format("{} / {}", stats.sky_lut_updates, stats.aerial_lut_updates));
            ui::value_row("GPU Planetshine bake", std::format("{:.3f} ms", stats.secondary_lut_updates ? vk_.gpu_time_ms("atmosphere_secondary") : 0.0));
            ui::value_row("GPU Planetshine source", std::format("{:.3f} ms", stats.secondary_source_lut_updates ? vk_.gpu_time_ms("atmosphere_secondary_source") : 0.0));
            ui::value_row("Planetshine LUT updates", std::format("{}", stats.secondary_lut_updates));
            ui::section("Atmosphere shadow history", "Pixel counts use the latest completed GPU frame.");
            ui::value_row("Auto horizon", !camera_.horizon_align || !camera_.is_near_surface ? "Inactive" :
                camera_.key_q || camera_.key_e ? "Manual roll" : camera_.horizon_settling ?
                std::format("Settling ({:.4f} deg)", camera_.horizon_roll_error_deg) : "Settled");
            if (!settings.show_atmospheres || !settings.atmosphere_shadows_enabled || !settings.atmosphere_shadow_temporal || settings.atmosphere_shadow_reference) {
                ui::value_row("Progressive history", "Off");
            } else if (stats.shadow_updates == 0) {
                ui::value_row("Progressive history", "No visible shadowed atmosphere");
            } else {
                const std::pair<uint32_t, const char*> labels[] = {
                    {rendering::ShadowHistoryInitial, "initial history"}, {rendering::ShadowHistoryInterrupted, "interrupted frames"},
                    {rendering::ShadowHistoryCamera, "camera"}, {rendering::ShadowHistoryProjection, "projection"},
                    {rendering::ShadowHistoryAtmosphere, "atmosphere"}, {rendering::ShadowHistoryLighting, "lighting"},
                    {rendering::ShadowHistoryCasters, "casters"}, {rendering::ShadowHistoryRings, "ring profile"},
                    {rendering::ShadowHistorySampling, "sampling settings"}, {rendering::ShadowHistoryViewport, "viewport"},
                    {rendering::ShadowHistoryDefinition, "system definition"}, {rendering::ShadowHistoryTransport, "transport settings"}};
                std::string reasons;
                for (auto [mask, label] : labels) if (stats.shadow_history_reset_reasons & mask) {
                    if (!reasons.empty()) reasons += ", ";
                    reasons += label;
                }
                ui::value_row("Reset inputs (current)", reasons.empty() ? "None" : reasons);
                auto history = celestial_renderer_.shadow_history_diagnostics(vk_);
                if (history.valid && history.columns != 0) {
                    ui::value_row("Refining / complete columns", std::format("{} / {}", history.refining, history.complete));
                    ui::value_row("History weight (frames)", std::format("{}..{} / 64", history.minimum_frames, history.maximum_frames));
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Turning reuse caps prior history at eight frames. Stationary views refine to 64 frames.");
                    ui::value_row("Depth resets (completed)", std::format("{}", history.depth_resets));
                    ui::value_row("Rotation reuse / fallback", std::format("{} / {}", history.reprojected, history.reprojection_rejected));
                } else {
                    ui::value_row("Pixel history", history.valid ? "No atmospheric columns" : "Waiting for GPU counters");
                }
            }
        }
        ImGui::End();
    }

    if (!ui_settings_.show_outliner && !size_comparison_active_) {
        ImGui::SetNextWindowPos(ImVec2(10.0f, 38.0f), ImGuiCond_Always);
        ImGuiWindowFlags pill_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove;
        if (ImGui::Begin("##CameraPill", nullptr, pill_flags)) {
            std::string tracked_name = "Free Camera";
            if (camera_.tracking_idx >= 0 && camera_.tracking_idx < static_cast<int>(scene().size())) {
                tracked_name = camera_.tracking_barycenter
                    ? std::format("{} Barycenter", scene().body(camera_.tracking_idx).definition.name)
                    : scene().body(camera_.tracking_idx).definition.name;
            }
            ImGui::TextColored(ui::ACCENT, "Track: %s", tracked_name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("| %s", camera_.movement_mode == 0 ? "Free Flight" : "Simple Orbit");
            ImGui::SameLine();
            if (ImGui::SmallButton("Outliner")) {
                ui_settings_.show_outliner = true;
            }
        }
        ImGui::End();
    }
}

void App::render_outliner() {
    if (size_comparison_active_) { render_comparison_outliner(); return; }
    if (!ui_settings_.show_outliner) return;

    ui::WorkspaceLayout layout{};
    ImGuiWindowFlags flags = ui_.prepare_side_panel(ui_settings_, ui_state_, ui::PanelSide::Left, layout);
    if (ImGui::Begin("System Outliner", &ui_settings_.show_outliner, flags)) {
        ui::heading(system().name(), "Click to inspect, double-click to fly to body.");
        const char* mode_label = "Keplerian";
        if (system().definition().motion_model == ephemeris::MotionModel::Spice) mode_label = "SPICE ephemeris";
        else if (system().definition().motion_model == ephemeris::MotionModel::NBody) mode_label = "N-Body (IAS15)";
        ImGui::TextDisabled("Motion: %s", mode_label);
        if (system().definition().motion_model == ephemeris::MotionModel::Spice) {
            ImGui::TextDisabled("Epoch: %s UTC", system().definition().epoch_utc.c_str());
        }
        if (!simulation_error_.empty()) ImGui::TextWrapped("Last error: %s", simulation_error_.c_str());
        if (system().definition().motion_model != ephemeris::MotionModel::Spice) {
            ImGui::Spacing();
            if (ImGui::Button("+ Add Orbiting Body...", ImVec2(-1.0f, 0.0f))) {
                int p_idx = (ui_state_.inspected_body >= 0 && ui_state_.inspected_body < static_cast<int>(system().size()))
                    ? ui_state_.inspected_body : system().definition().star_idx;
                ui_state_.add_body.reset_for_parent(system(), p_idx);
                ui_state_.add_body.active = true;
            }
            ImGui::Spacing();
        }
        ImGui::PushID(static_cast<int>(system().generation()));
        for (int root : system().reference_hierarchy().roots) render_outliner_node(root);
        ImGui::PopID();
        render_compared_outliner();
    }
    ImGui::End();
}

void App::render_body_inspector() {
    if (size_comparison_active_) { render_comparison_inspector(); return; }
    if (ui_settings_.show_inspector && ui_state_.inspected_body >= static_cast<int>(system().size())) { render_compared_inspector(); return; }
    if (!ui_settings_.show_inspector || !ui_state_.has_selection(system())) return;
    int insp_idx = ui_state_.inspected_body;
    if (insp_idx < 0 || insp_idx >= static_cast<int>(system().size())) return;

    ui::WorkspaceLayout layout{};
    ImGuiWindowFlags flags = ui_.prepare_side_panel(ui_settings_, ui_state_, ui::PanelSide::Right, layout);

    const auto& b = system().body(insp_idx);
    bool inspect_bary = ui_state_.inspect_barycenter;
    std::string win_title = inspect_bary
        ? std::format("{} System Barycenter###inspector", b.definition.name)
        : std::format("{}###inspector", b.definition.name);

    flags |= ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    if (!ImGui::Begin(win_title.c_str(), &ui_settings_.show_inspector, flags)) {
        ImGui::End();
        return;
    }

    std::string obj_type = inspect_bary ? "Mode: Barycenter" : b.definition.type;
    ImGui::TextColored(ui::ACCENT, "%s", obj_type.c_str());

    if (!inspect_bary) {
        float text_w = ImGui::CalcTextSize("Edit Mode").x + ImGui::GetStyle().FramePadding.x * 2.0f + 25.0f;
        float right_pos = ImGui::GetWindowWidth() - text_w - ImGui::GetStyle().WindowPadding.x;
        if (right_pos > ImGui::GetCursorPosX()) {
            ImGui::SameLine(right_pos);
        }
        bool prev_edit = ui_state_.edit_mode;
        if (ImGui::Checkbox("Edit Mode", &ui_state_.edit_mode)) {
            if (ui_state_.edit_mode && !prev_edit) {
                simulation_.apply(ephemeris::SetPaused{true});
                ui_state_.body_draft.reload(system(), insp_idx);
            }
        }
    }

    bool is_tracked = (camera_.tracking_idx == insp_idx && camera_.tracking_barycenter == inspect_bary);
    bool is_centered = (camera_.centered_idx == insp_idx && camera_.centered_barycenter == inspect_bary);

    float avail_w = ImGui::GetContentRegionAvail().x;
    float gap = 6.0f * layout.scale;
    float btn_w = (avail_w - gap * 3.0f) / 4.0f;

    if (is_tracked) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
    if (ImGui::Button(is_tracked ? "Tracking##trk" : "Track##trk", ImVec2(btn_w, 0.0f))) {
        if (is_tracked) {
            camera_.tracking_idx = -1;
            camera_.tracking_barycenter = false;
        } else {
            camera_.focus_body(insp_idx, system(), false, inspect_bary);
        }
    }
    if (is_tracked) ImGui::PopStyleColor();

    ImGui::SameLine(0.0f, gap);

    if (is_centered) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
    if (ImGui::Button(is_centered ? "Centered##cnt" : "Center##cnt", ImVec2(btn_w, 0.0f))) {
        if (is_centered) {
            camera_.centered_idx = -1;
        } else {
            camera_.center_body(insp_idx, system(), inspect_bary);
        }
    }
    if (is_centered) ImGui::PopStyleColor();

    ImGui::SameLine(0.0f, gap);

    if (ImGui::Button("Go To##goto", ImVec2(btn_w, 0.0f))) {
        camera_.focus_body(insp_idx, system(), true, inspect_bary);
    }

    ImGui::SameLine(0.0f, gap);

    if (ImGui::Button(inspect_bary ? "Body##bary" : "Barycenter##bary", ImVec2(btn_w, 0.0f))) {
        ui_state_.inspect_barycenter = !inspect_bary;
        if (camera_.tracking_idx == insp_idx) camera_.tracking_barycenter = ui_state_.inspect_barycenter;
        if (camera_.centered_idx == insp_idx) camera_.centered_barycenter = ui_state_.inspect_barycenter;
    }

    ImGui::Separator();

    if (system().definition().motion_model != ephemeris::MotionModel::Spice && !inspect_bary) {
        if (ImGui::Button("+ Add Satellite to This Body", ImVec2(-1.0f, 0.0f))) {
            ui_state_.add_body.reset_for_parent(system(), insp_idx);
            ui_state_.add_body.active = true;
        }
        ImGui::Spacing();
    }

    int parent_idx = system().reference_parent(insp_idx);
    bool has_parent = (parent_idx >= 0 && parent_idx < static_cast<int>(system().size()));
    math::KeplerianElements oe{};
    glm::dvec3 rel_r{0.0};
    glm::dvec3 rel_v{0.0};
    double mu = 0.0;
    std::optional<double> axial_tilt_deg;

    if (has_parent) {
        const auto& p = system().body(parent_idx);
        if (inspect_bary) {
            auto coords = ephemeris::extract_jacobi(system().definition(), system().motions(), system().orbit_hierarchy());
            rel_r = coords.subsystems[insp_idx].position - coords.subsystems[parent_idx].position;
            rel_v = coords.subsystems[insp_idx].velocity - coords.subsystems[parent_idx].velocity;
        } else {
            rel_r = b.motion.position.value - p.motion.position.value;
            rel_v = b.motion.velocity.value - p.motion.velocity.value;
            glm::dvec3 orb_h = glm::cross(rel_r, rel_v);
            double h_norm = glm::length(orb_h);
            if (h_norm > 1e-12) {
                glm::dvec3 orb_normal = orb_h / h_norm;
                glm::dvec3 body_pole = b.definition.pole_ecl;
                if (glm::length(body_pole) > 1e-12) {
                    body_pole = glm::normalize(body_pole);
                    double tilt_rad = std::acos(std::clamp(glm::dot(body_pole, orb_normal), -1.0, 1.0));
                    axial_tilt_deg = tilt_rad * RAD_TO_DEG;
                }
            }
        }

        glm::dvec3 calc_r = rel_r;
        glm::dvec3 calc_v = rel_v;
        if (ui_state_.inspector_frame == 1) {
            auto rotated = math::rotate_ecliptic_to_equatorial(rel_r, rel_v, p.definition.pole_ecl);
            calc_r = rotated.first;
            calc_v = rotated.second;
        }

        double orb_mass = inspect_bary ? 0.0 : b.definition.mass_msun;
        mu = G_CONST * (p.definition.mass_msun + orb_mass);
        oe = math::compute_keplerian_elements(calc_r, calc_v, mu);
    }

    auto& draft = ui_state_.body_draft;
    bool physical_editing = ui_state_.edit_mode && !inspect_bary;
    if (physical_editing) {
        draft.synchronize(system(), insp_idx);
        if (has_parent && !draft.values.update_orbit) {
            draft.values.a_au = oe.a;
            draft.values.e = oe.e;
            draft.values.inc_deg = oe.inc_deg;
            draft.values.omega_node_deg = oe.omega_node_deg;
            draft.values.arg_peri_deg = oe.arg_peri_deg;
            draft.values.mean_anom_deg = oe.mean_anomaly_deg;
            draft.values.equatorial_frame = (ui_state_.inspector_frame == 1);
        }
    }

    bool roche_violates = false;
    double prop_periapsis = 0.0;
    double prop_roche = 0.0;
    if (physical_editing && has_parent) {
        double p_mass = system().body(parent_idx).definition.mass_msun;
        double m_body = draft.values.mass.value;
        double r_km = draft.values.radius.value;
        double a_val = draft.values.a_au;
        double e_val = draft.values.e;
        prop_periapsis = a_val * (1.0 - e_val);
        if (m_body > 0.0 && r_km > 0.0 && p_mass > 0.0) {
            prop_roche = 2.44 * r_km / AU_TO_KM * std::cbrt(p_mass / m_body);
            if (prop_periapsis < prop_roche) {
                roche_violates = true;
            }
        }
    }

    bool can_del = (insp_idx != system().definition().star_idx && !b.definition.is_star && system().definition().motion_model != ephemeris::MotionModel::Spice);
    std::string edit_notice = roche_violates ? "Periapsis inside Roche limit. Apply disabled." : "";
    if (!draft.error.empty()) {
        if (!edit_notice.empty()) edit_notice += '\n';
        edit_notice += draft.error;
    }
    float footer_h = physical_editing ? ui::panel_footer_height(can_del ? 2 : 1, edit_notice) : 0.0f;
    auto render_tilt_input = [&] {
        double tilt = draft.axial_tilt_deg(system());
        ui::field("Axial tilt (deg)", [&](const char* id) {
            if (ImGui::InputDouble(id, &tilt, 1.0, 5.0, "%.2f")) draft.set_axial_tilt_deg(system(), tilt);
        });
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(has_parent ? "Angle to the orbital normal (0 to 180 degrees)." : "Angle to the ecliptic normal (0 to 180 degrees).");
        }
        if (!b.definition.is_star && !system().reference_hierarchy().children[insp_idx].empty()) {
            bool ephemeris_orbits = system().definition().motion_model == ephemeris::MotionModel::Spice;
            ImGui::BeginDisabled(ephemeris_orbits);
            ImGui::Checkbox("Rotate moon orbits with tilt", &draft.rotate_moon_orbits);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip(ephemeris_orbits ? "Switch to Keplerian or N-body motion to edit moon orbits." :
                    "On Apply, rotate bound moons and their satellites around this body's centre with its pole. Orbit sizes, shapes and phases are preserved.");
            }
        }
    };

    ui::begin_panel_content("inspector_tab_content", footer_h);
    if (ImGui::BeginTabBar("InspectorTabBar", ImGuiTabBarFlags_FittingPolicyScroll)) {
        if (ImGui::BeginTabItem("Overview")) {
            ImGui::BeginChild("tab_overview_content", ImVec2(0.0f, 0.0f), false);
            ui::section("Physical properties", physical_editing ? "Proposed values" : "");

            if (inspect_bary) {
                auto coords = ephemeris::extract_jacobi(system().definition(), system().motions(), system().orbit_hierarchy());
                double b_mass = coords.subsystems[insp_idx].mass;
                if (b_mass > 1e-4) ui::metric_text(std::format("System Mass: {:.6e} M\u2609", b_mass));
                else ui::metric_text(std::format("System Mass: {:.4f} M\u2295", b_mass / 3.003e-6));
            } else if (physical_editing) {
                static const char* s_types[] = {"Star", "Terrestrial", "Gas Giant", "Ice Giant", "Dwarf Planet", "Moon"};
                int type_idx = 1;
                for (int i = 0; i < 6; ++i) {
                    if (draft.values.type == s_types[i]) { type_idx = i; break; }
                }
                if (ImGui::Combo("Type", &type_idx, s_types, 6)) {
                    draft.values.type = s_types[type_idx];
                    draft.recompute_hydrostatic();
                    draft.dirty = true;
                }

                if (draft.values.type == "Star") {
                    if (ui::render_star_creator_widgets(draft.star)) {
                        draft.dirty = true;
                    }
                    const auto& pr = draft.star.preview;
                    draft.values.mass = units::SolarMasses{pr.mass_msun};
                    draft.values.radius = units::Kilometres{pr.radius_rsun * SOLAR_RADIUS_KM};
                    draft.values.oblateness = pr.oblateness;
                    draft.values.temp_k = pr.temp_k;
                    draft.values.lum_eq = pr.lum_eq_lsun;
                    draft.values.lum_pole = pr.lum_pole_lsun;
                    draft.values.color_rgb = pr.color_rgb;
                    draft.values.color_pole_rgb = pr.color_pole_rgb;
                    draft.values.spectral_class = pr.spectral_class;
                    draft.values.stage = pr.stage_name;
                    draft.values.star_props = physics::StarCalculator::to_json(pr, draft.star.params);
                } else {
                    if (ImGui::InputDouble("Mass (M\u2609)", &draft.values.mass.value, 0.0, 0.0, "%.6e")) {
                        draft.recompute_hydrostatic();
                        draft.dirty = true;
                    }
                    if (ImGui::InputDouble("Radius (km)", &draft.values.radius.value, 0.0, 0.0, "%.1f")) {
                        draft.recompute_hydrostatic();
                        draft.dirty = true;
                    }

                }

                ui::section("Rotation & shape");
                if (has_parent && ImGui::Checkbox("Tidally locked", &draft.values.tidally_locked)) {
                    if (draft.values.tidally_locked && oe.period_days > 0.0) {
                        draft.values.rotation_period_hr = oe.period_days * 24.0;
                        draft.recompute_hydrostatic();
                    }
                    draft.dirty = true;
                }
                ImGui::BeginDisabled(has_parent && draft.values.tidally_locked);
                ui::field("Rotation period (hours)", [&](const char* id) {
                    if (ImGui::InputDouble(id, &draft.values.rotation_period_hr, 0.0, 0.0, "%.4f")) {
                        if (draft.values.type != "Star") draft.recompute_hydrostatic();
                        draft.dirty = true;
                    }
                });
                ImGui::EndDisabled();
                if (has_parent && draft.values.tidally_locked) ImGui::TextWrapped("Rotation follows the parent. Unlock to edit the rotation period.");
                render_tilt_input();
                if (draft.values.type != "Star") {
                    ui::field("Oblateness (f)", [&](const char* id) {
                        if (ImGui::InputDouble(id, &draft.values.oblateness, 0.0, 0.0, "%.6f")) draft.dirty = true;
                    });
                }
                ui::field("J2 moment", [&](const char* id) {
                    if (ImGui::InputDouble(id, &draft.values.j2, 0.0, 0.0, "%.6e")) draft.dirty = true;
                });
                ui::field("J4 moment", [&](const char* id) {
                    if (ImGui::InputDouble(id, &draft.values.j4, 0.0, 0.0, "%.6e")) draft.dirty = true;
                });
                if (draft.values.type != "Star" && ImGui::Button("Calculate hydrostatic shape")) {
                    draft.recompute_hydrostatic();
                    draft.dirty = true;
                }

                double edit_m = draft.values.mass.value;
                double edit_r_km = draft.values.radius.value;
                if (edit_r_km > 0.0) {
                    double g_m_s2 = (1.32712440018e14 * edit_m) / (edit_r_km * edit_r_km);
                    double g_earth = g_m_s2 / 9.80665;
                    ui::metric_text(std::format("Surface G: {:.3f} m/s² ({:.3f} g)", g_m_s2, g_earth));
                }
            } else {
                double b_mass = b.definition.mass_msun;
                if (b_mass > 1e-4) ui::metric_text(std::format("Mass: {:.6e} M\u2609", b_mass));
                else if (b_mass > 1e-10) ui::metric_text(std::format("Mass: {:.6f} M\u2295", b_mass / 3.003e-6));
                else ui::metric_text(std::format("Mass: {:.4e} M\u263E", b_mass / 3.694e-8));

                double r_km = b.definition.radius_au() * AU_TO_KM;
                if (r_km > 100.0) ui::metric_text(std::format("Radius: {:.0f} km", r_km));
                else ui::metric_text(std::format("Radius: {:.1f} km", r_km));

                if (r_km > 0.0) {
                    double g_m_s2 = (1.32712440018e14 * b_mass) / (r_km * r_km);
                    double g_earth = g_m_s2 / 9.80665;
                    ui::metric_text(std::format("Surface G: {:.3f} m/s² ({:.3f} g)", g_m_s2, g_earth));
                    double vol_cm3 = (4.0 / 3.0) * sf::PI * std::pow(r_km * 1e5, 3.0);
                    double mass_g = b_mass * 1.98847e33;
                    double density = (vol_cm3 > 0.0) ? (mass_g / vol_cm3) : 0.0;
                    ui::metric_text(std::format("Mean Density: {:.4f} g/cm³", density));
                }

                ui::section("Rotation & shape");
                if (std::abs(b.definition.rotation_period_hr) > 0.0) {
                    ui::metric_text(std::format("Rot Period: {:.2f} hours", b.definition.rotation_period_hr));
                }
                if (axial_tilt_deg.has_value()) {
                    std::string tilt_str = std::format("{:.2f}°", *axial_tilt_deg);
                    if (*axial_tilt_deg > 90.0) tilt_str += " / retrograde";
                    ui::value_row("Axial tilt", tilt_str);
                }
                if (b.definition.oblateness > 0.0) {
                    ui::metric_text(std::format("Oblateness (f): {:.4f}", b.definition.oblateness));
                    if (ImGui::IsItemHovered()) {
                        double r_pole = r_km * (1.0 - b.definition.oblateness);
                        ImGui::SetTooltip("Equatorial Radius: %s km\nPolar Radius: %s km\nPolar Flattening: %s km",
                                          trim_trailing_zeros(std::format("{:.1f}", r_km)).c_str(),
                                          trim_trailing_zeros(std::format("{:.1f}", r_pole)).c_str(),
                                          trim_trailing_zeros(std::format("{:.1f}", r_km - r_pole)).c_str());
                    }
                    ui::metric_text(std::format("J2 Moment: {:.6e}", b.definition.j2));
                    ui::metric_text(std::format("J4 Moment: {:.6e}", b.definition.j4));
                }
            }

            if (!inspect_bary && b.definition.is_star) {
                if (!physical_editing) {
                    ui::section("Stellar properties");
                    ui::metric_text(std::format("Temperature: {:.0f} K", b.definition.temp_k));
                    ui::metric_text(std::format("Luminosity: {:.4f} L\u2609", b.definition.lum_eq));
                    ui::metric_text(std::format("Spectral Cl: {}", b.definition.spectral_class.empty() ? "G2V" : b.definition.spectral_class));
                    ui::metric_text(std::format("Stage: {}", b.definition.stage.empty() ? "Main Sequence" : b.definition.stage));
                    if (b.definition.lum_eq > 0.0) {
                        double hz_in = std::sqrt(b.definition.lum_eq / 1.78);
                        double hz_out = std::sqrt(b.definition.lum_eq / 0.32);
                        ui::metric_text("Habitable Zone (optimistic):");
                        ui::metric_text(std::format("  Inner: {}", format_distance_au(hz_in, ui_settings_.ly_threshold_au)));
                        ui::metric_text(std::format("  Outer: {}", format_distance_au(hz_out, ui_settings_.ly_threshold_au)));
                    }
                }
            } else if (!inspect_bary) {
                ui::section("Reflectance", "Estimated from body surface and atmosphere model.");
                double ag = 0.367, ab = 0.306, q = 0.83;
                if (b.definition.type == "Gas Giant") { ag = 0.52; ab = 0.34; q = 0.65; }
                else if (b.definition.type == "Ice Giant") { ag = 0.41; ab = 0.29; q = 0.71; }
                else if (b.definition.type == "Moon" || b.definition.type == "Dwarf Planet") { ag = 0.12; ab = 0.08; q = 0.67; }
                ui::metric_text(std::format("Geom Albedo: {:.3f}", ag));
                ui::metric_text(std::format("Bond Albedo: {:.3f} (q = {:.2f})", ab, q));
            }

            if (!inspect_bary && !b.definition.is_star && !b.definition.is_black_hole && !b.definition.is_spacecraft) {
                ui::section("Surface material");
                auto& authored = draft.values.material;
                auto material = physics::resolve_material(physical_editing ? authored : b.definition.material, b.definition.name);
                const char* names[] = {"Automatic", "Lambert", "Lunar regolith", "Icy regolith", "Custom"};
                const char* presets[] = {"auto", "lambert", "lunar", "icy", "custom"};
                const auto& selected = physical_editing ? authored.preset : b.definition.material.preset;
                int preset = 0;
                for (int i = 0; i < 5; ++i) if (selected == presets[i]) preset = i;
                ImGui::BeginDisabled(!physical_editing);
                ui::field("Material preset", [&](const char* id) {
                    if (ImGui::Combo(id, &preset, names, 5)) {
                        authored = preset == 4 ? material : physics::material_preset(presets[preset]);
                        authored.preset = presets[preset];
                        material = physics::resolve_material(authored, b.definition.name);
                        draft.dirty = true;
                    }
                });
                bool hapke = material.hapke;
                if (ImGui::Checkbox("Hapke photometry", &hapke)) {
                    authored = material; authored.preset = "custom"; authored.hapke = hapke;
                    material = authored; draft.dirty = true;
                }
                const char* labels[] = {"Grain scattering albedo", "Particle phase width", "Backscatter fraction",
                    "Shadow opposition strength", "Shadow opposition width", "Roughness (degrees)", "Brightness",
                    "Coherent opposition strength", "Coherent opposition width"};
                for (size_t i = 0; i < physics::MATERIAL_FIELDS.size(); ++i) {
                    if (!material.hapke && i != 6) continue;
                    double value = material.*physics::MATERIAL_FIELDS[i];
                    ui::field(labels[i], [&](const char* id) {
                        if (ImGui::SliderScalar(id, ImGuiDataType_Double, &value,
                            &physics::MATERIAL_MIN[i], &physics::MATERIAL_MAX[i], "%.3f")) {
                            authored = material; authored.preset = "custom";
                            authored.*physics::MATERIAL_FIELDS[i] = value;
                            material = authored; draft.dirty = true;
                        }
                    });
                }
                ImGui::EndDisabled();
            }

            ui::section("Camera position", "Distances from camera to inspected body.");
            double dist_eye_au = glm::length(b.render.position.value - camera_.eye_pos_render().value);
            double dist_to_center_km = dist_eye_au * AU_TO_KM;
            if (dist_to_center_km > 1.49597e7) {
                ui::metric_text(std::format("Distance: {}", format_distance_au(dist_eye_au, ui_settings_.ly_threshold_au)));
            } else {
                ui::metric_text(std::format("Distance: {:.0f} km", dist_to_center_km));
            }

            if (!inspect_bary && b.definition.radius_au() > 0.0) {
                double r_km = (physical_editing ? draft.values.radius.value : b.definition.radius_au() * AU_TO_KM);
                double f_oblate = physical_editing ? draft.values.oblateness : b.definition.oblateness;
                glm::dvec3 cam_rel = camera_.eye_pos_render().value - b.render.position.value;
                double surf_r_km = ellipsoid_surface_radius(r_km, f_oblate, b.render.pole, cam_rel);
                double alt_km = dist_to_center_km - surf_r_km;
                double alt_au = alt_km * KM_TO_AU;
                ui::metric_text(std::format("Altitude: {}", format_altitude(alt_km, alt_au, ui_settings_.ly_threshold_au)));
            }

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Orbit")) {
            ImGui::BeginChild("tab_orbit_content", ImVec2(0.0f, 0.0f), false);

            if (!has_parent) {
                ImGui::TextWrapped("Primary central body / no parent orbit.");
            } else {
                const auto& p = system().body(parent_idx);
                ui::value_row("Reference parent", p.definition.name);

                ui::section("Live orbit", "Osculating elements relative to the reference parent.");
                double dist_au = glm::length(rel_r);
                double vel_au_yr = glm::length(rel_v);
                double vel_km_s = vel_au_yr * AU_TO_KM / SECONDS_PER_YEAR;

                if (oe.period_days > 0.0) {
                    if (oe.period_days < 730.0) ui::value_row("Period", std::format("{:.3f} days", oe.period_days));
                    else ui::value_row("Period", std::format("{:.4f} years", oe.period_days / 365.25));
                } else {
                    ui::value_row("Period", "N/A / unbound");
                }
                ui::value_row("Distance to parent", format_distance_au(dist_au, ui_settings_.ly_threshold_au));
                ui::value_row("Speed relative to parent", std::format("{:.3f} km/s", vel_km_s));

                double live_peri = oe.a * (1.0 - oe.e);
                double live_apo = oe.a * (1.0 + oe.e);
                ui::value_row("Periapsis", format_distance_au(live_peri, ui_settings_.ly_threshold_au));
                if (oe.a > 0.0 && oe.e < 1.0) ui::value_row("Apoapsis", format_distance_au(live_apo, ui_settings_.ly_threshold_au));
                else ui::value_row("Apoapsis", "N/A / open orbit");

                if (physical_editing) {
                    render_tilt_input();
                } else if (axial_tilt_deg.has_value()) {
                    std::string tilt_str = std::format("{:.2f}°", *axial_tilt_deg);
                    if (*axial_tilt_deg > 90.0) tilt_str += " / retrograde";
                    ui::value_row("Axial tilt", tilt_str);
                }

                ImGui::Spacing();
                const char* frame_names[] = {"Ecliptic", "Equatorial"};
                int cur_frame = ui_state_.inspector_frame;
                if (ImGui::Combo("Reference Frame", &cur_frame, frame_names, 2)) {
                    ui_state_.inspector_frame = cur_frame;
                    if (physical_editing) {
                        draft.values.equatorial_frame = (cur_frame == 1);
                    }
                }

                if (physical_editing) {
                    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
                }
                if (ImGui::CollapsingHeader("Orbital elements###orbit_elements")) {
                    if (physical_editing) {
                        ImGui::Separator();
                        ImGui::TextColored(ui::ACCENT, "Edit Orbit");
                        if (ImGui::Checkbox("Bring moon system (Hill sphere)", &draft.values.bring_moons)) {
                            draft.dirty = true;
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Shift all bodies inside this object's Hill sphere to its new position and velocity.");
                        }
                        if (ImGui::InputDouble("Semi-Major Axis (AU)", &draft.values.a_au, 0.001, 0.01, "%.6f")) {
                            draft.values.update_orbit = true;
                            draft.dirty = true;
                        }
                        if (ImGui::InputDouble("Eccentricity", &draft.values.e, 0.001, 0.01, "%.6f")) {
                            draft.values.update_orbit = true;
                            draft.dirty = true;
                        }
                        if (ImGui::InputDouble("Inclination (deg)", &draft.values.inc_deg, 0.1, 1.0, "%.3f")) {
                            draft.values.update_orbit = true;
                            draft.dirty = true;
                        }
                        if (ImGui::InputDouble("\u03A9 (Long Asc Node)", &draft.values.omega_node_deg, 0.1, 1.0, "%.3f")) {
                            draft.values.update_orbit = true;
                            draft.dirty = true;
                        }
                        if (ImGui::InputDouble("\u03C9 (Arg Periapsis)", &draft.values.arg_peri_deg, 0.1, 1.0, "%.3f")) {
                            draft.values.update_orbit = true;
                            draft.dirty = true;
                        }
                        if (ImGui::InputDouble("Mean Anomaly (deg)", &draft.values.mean_anom_deg, 0.1, 1.0, "%.3f")) {
                            draft.values.update_orbit = true;
                            draft.dirty = true;
                        }
                        ui::section("Proposed orbital range");
                        double p_peri = draft.values.a_au * (1.0 - draft.values.e);
                        double p_apo = draft.values.a_au * (1.0 + draft.values.e);
                        ui::value_row("Periapsis", format_distance_au(p_peri, ui_settings_.ly_threshold_au));
                        if (draft.values.a_au > 0.0 && draft.values.e < 1.0) ui::value_row("Apoapsis", format_distance_au(p_apo, ui_settings_.ly_threshold_au));
                        else ui::value_row("Apoapsis", "N/A / open orbit");
                    } else {
                        ui::metric_text(std::format("Semi-major (a): {}", format_distance_au(oe.a, ui_settings_.ly_threshold_au)));
                        ui::metric_text(std::format("Eccentricity (e): {:.6f}", oe.e));
                        ui::metric_text(std::format("Inclination (i): {:.3f}°", oe.inc_deg));
                        ui::metric_text(std::format("Long Asc Node (\u03A9): {:.3f}°", oe.omega_node_deg));
                        ui::metric_text(std::format("Arg Periapsis (\u03C9): {:.3f}°", oe.arg_peri_deg));
                        ui::metric_text(std::format("True Anomaly (\u03BD): {:.3f}°", oe.true_anomaly_deg));
                        ui::metric_text(std::format("Mean Anomaly (M): {:.3f}°", oe.mean_anomaly_deg));
                    }
                }

                if (ImGui::CollapsingHeader("Gravitational limits###orbit_limits")) {
                    ImGui::TextWrapped("Hill radius uses live state; Roche limit uses proposed values while editing.");
                    double p_m = p.definition.mass_msun;
                    double b_m = physical_editing ? draft.values.mass.value : b.definition.mass_msun;
                    double r_km = physical_editing ? draft.values.radius.value : b.definition.radius_au() * AU_TO_KM;
                    if (p_m > 0.0 && b_m > 0.0 && dist_au > 0.0) {
                        double hill_au = dist_au * std::cbrt(b_m / (3.0 * p_m));
                        ui::metric_text(std::format("Hill Radius: {}", format_distance_au(hill_au, ui_settings_.ly_threshold_au)));
                    }
                    if (p_m > 0.0 && b_m > 0.0 && r_km > 0.0) {
                        double roche_au = 2.44 * r_km / AU_TO_KM * std::cbrt(p_m / b_m);
                        ui::metric_text(std::format("Roche Limit (fluid): {}", format_distance_au(roche_au, ui_settings_.ly_threshold_au)));
                        double active_peri = physical_editing ? (draft.values.a_au * (1.0 - draft.values.e)) : live_peri;
                        if (active_peri < roche_au) {
                            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Periapsis is inside Roche limit");
                        }
                    }
                }

                if (ImGui::CollapsingHeader("Precession estimates###orbit_precession")) {
                    if (oe.a > 0.0 && oe.e < 1.0) {
                        double p_param = oe.a * (1.0 - oe.e * oe.e);
                        double gr_arcsec_cy = 0.0;
                        if (p_param > 0.0 && oe.period_days > 0.0) {
                            double c2 = C_AU_YR * C_AU_YR;
                            double gr_rad_per_orb = 3.0 * mu / (oe.a * c2 * (1.0 - oe.e * oe.e));
                            double orbs_per_cy = 36525.0 / oe.period_days;
                            gr_arcsec_cy = gr_rad_per_orb * orbs_per_cy * RAD_TO_DEG * 3600.0;
                        }
                        ui::metric_text(std::format("GR (apsidal): {:.2f}\"/cy", gr_arcsec_cy));

                        double j2_aps = 0.0, j2_nod = 0.0;
                        if (p.definition.j2 > 0.0 && oe.period_days > 0.0 && p_param > 0.0) {
                            double p_r_au = p.definition.radius_au();
                            double n_mean = TWO_PI / (oe.period_days / 365.25);
                            double ratio2 = (p_r_au / p_param) * (p_r_au / p_param);
                            double j2_aps_rad_yr = 1.5 * n_mean * p.definition.j2 * ratio2;
                            double j2_nod_rad_yr = -j2_aps_rad_yr * std::cos(oe.inc_deg * DEG_TO_RAD);
                            j2_aps = j2_aps_rad_yr * 100.0 * RAD_TO_DEG * 3600.0;
                            j2_nod = j2_nod_rad_yr * 100.0 * RAD_TO_DEG * 3600.0;
                        }
                        ui::metric_text(std::format("J2 (apsidal): {:.2f}\"/cy", j2_aps));
                        ui::metric_text(std::format("J2 (nodal): {:.2f}\"/cy", j2_nod));
                    }
                }
            }

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        if (!inspect_bary && !b.definition.is_star && !b.definition.is_black_hole && !b.definition.is_spacecraft) {
            render_body_surface_tab(insp_idx);
            if (ImGui::BeginTabItem("Rings")) {
                ImGui::BeginChild("tab_rings_content", ImVec2(0.0f, 0.0f), false);

                auto rings = b.definition.rings;
                auto tex_in = b.definition.ring_texture_inner;
                auto tex_out = b.definition.ring_texture_outer;
                uint64_t def_rev = b.revisions.definition;
                double r_eq_km = b.definition.radius_au() * AU_TO_KM;

                auto tex_path = ephemeris::find_ring_texture_path(system().name(), b.definition.name);
                bool has_ring_tex_file = tex_path.has_value() || celestial_renderer_.has_custom_ring_texture(system().name(), b.definition.name);

                auto& re = ui_state_.ring_edit;
                re.sync_body(insp_idx, static_cast<int>(rings.size()));
                bool ring_changed = false;

                ui::section("Ring Layers", "Configure concentric ring segments and optical properties.");

                if (!rings.empty()) {
                    auto fmt_layer = [&](int idx) {
                        return std::format("Layer {} ({:.2f} - {:.2f} R)", idx + 1, rings[idx].inner, rings[idx].outer);
                    };
                    std::string cur_lbl = fmt_layer(re.selected_layer);
                    if (ImGui::BeginCombo("Selected Layer", cur_lbl.c_str())) {
                        for (int i = 0; i < static_cast<int>(rings.size()); ++i) {
                            bool sel = (i == re.selected_layer);
                            if (ImGui::Selectable(fmt_layer(i).c_str(), sel)) {
                                re.selected_layer = i;
                            }
                            if (sel) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                } else {
                    ImGui::TextDisabled("This body has no planetary rings.");
                }

                float btn_half_w = std::max(80.0f, (ImGui::GetContentRegionAvail().x - 8.0f) * 0.5f);
                ImGui::BeginDisabled(rings.size() >= 8);
                if (ImGui::Button("+ Add Layer", ImVec2(rings.empty() ? -1.0f : btn_half_w, 0.0f))) {
                    ephemeris::RingLayerDefinition nl{};
                    nl.inner = rings.empty() ? 1.5 : std::round((rings.back().outer + 0.05) * 100.0) / 100.0;
                    nl.outer = std::round((nl.inner + 0.5) * 100.0) / 100.0;
                    nl.color_rgb = glm::vec3(0.847f, 0.792f, 0.616f);
                    nl.opacity = 0.7f;
                    nl.scatter = 0.5f;
                    nl.asymmetry = 0.75f;
                    nl.backscatter = -0.45f;
                    nl.unlit_factor = 1.0f;
                    nl.is_textured = has_ring_tex_file;
                    nl.gradient = {{0.0f, 0.0f}, {0.1f, 1.0f}, {0.9f, 1.0f}, {1.0f, 0.0f}};
                    rings.push_back(std::move(nl));
                    re.selected_layer = static_cast<int>(rings.size()) - 1;
                    ring_changed = true;
                }
                ImGui::EndDisabled();

                if (!rings.empty()) {
                    ImGui::SameLine(0.0f, 8.0f);
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.15f, 0.15f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
                    if (ImGui::Button("Remove Layer", ImVec2(btn_half_w, 0.0f))) {
                        rings.erase(rings.begin() + re.selected_layer);
                        re.sync_body(insp_idx, static_cast<int>(rings.size()));
                        ring_changed = true;
                    }
                    ImGui::PopStyleColor(3);
                }

                ui::section("Procedural Generator", "Synthesize SpaceEngine-style multi-band ring profiles with fine striations.");
                if (ImGui::CollapsingHeader("Procedural Generator###proc_ring_hdr", rings.empty() ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
                    ImGui::TextWrapped("Generate a new ring profile from a preset or customize parameters, then apply or bake it.");
                    ImGui::Spacing();

                    auto presets = rendering::get_ring_presets();
                    int cur_p_idx = std::clamp(re.procedural.preset_idx, 0, static_cast<int>(presets.size()) - 1);
                    const char* cur_p_name = presets[cur_p_idx].name.data();
                    if (ImGui::BeginCombo("Ring Preset##proc", cur_p_name)) {
                        for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
                            bool sel = (i == cur_p_idx);
                            if (ImGui::Selectable(presets[i].name.data(), sel)) {
                                re.procedural.preset_idx = i;
                                const auto& p = presets[i];
                                re.procedural.band_count = p.band_count;
                                re.procedural.gap_count = p.gap_count;
                                re.procedural.striation_freq = p.striation_freq;
                                re.procedural.contrast = p.contrast;
                                re.procedural.base_density = p.base_density;
                                re.procedural.core_boost = p.core_boost;
                                re.procedural.inner_mult = p.inner_radius_ratio;
                                re.procedural.outer_mult = p.outer_radius_ratio;
                                re.procedural.opacity = p.opacity;
                            }
                            if (sel) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }

                    ui::metric_text("Seed");
                    float seed_w = std::max(60.0f, ImGui::GetContentRegionAvail().x - 85.0f);
                    ImGui::SetNextItemWidth(seed_w);
                    ImGui::DragInt("##Seed##proc", &re.procedural.seed, 1.0f, 0, 999999);
                    ImGui::SameLine(0.0f, 8.0f);
                    if (ImGui::Button("Random##proc", ImVec2(75.0f, 0.0f))) {
                        re.procedural.seed = std::rand() % 1000000;
                    }

                    ImGui::SliderInt("Bands##proc", &re.procedural.band_count, 1, 8);
                    ImGui::SliderInt("Gaps##proc", &re.procedural.gap_count, 0, 6);
                    ImGui::SliderFloat("Striations##proc", &re.procedural.striation_freq, 20.0f, 300.0f, "%.0f");
                    ImGui::SliderFloat("Sharpness##proc", &re.procedural.contrast, 0.5f, 3.5f, "%.2f");
                    ImGui::SliderFloat("Density (Tau)##proc", &re.procedural.base_density, 0.05f, 8.0f, "%.2f");
                    ImGui::SliderFloat("Core Opacity##proc", &re.procedural.core_boost, 0.5f, 5.0f, "%.2f");

                    ui::drag_float("Inner (R_eq)##proc", &re.procedural.inner_mult, 0.01f, 1.05f, 4.0f, "%.2f R");
                    float min_out = re.procedural.inner_mult + 0.05f;
                    if (re.procedural.outer_mult < min_out) re.procedural.outer_mult = min_out;
                    ui::drag_float("Outer (R_eq)##proc", &re.procedural.outer_mult, 0.01f, min_out, 8.0f, "%.2f R");

                    float tint_col[3] = {re.procedural.tint.r, re.procedural.tint.g, re.procedural.tint.b};
                    if (ImGui::ColorEdit3("Tint Color##proc", tint_col)) {
                        re.procedural.tint = glm::vec3(tint_col[0], tint_col[1], tint_col[2]);
                    }
                    if (ui::drag_float("Opacity##proc", &re.procedural.opacity, 0.005f, 0.0f, 10.0f, "%.8f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat)) {
                        re.procedural.opacity = std::clamp(re.procedural.opacity, 0.0f, 10.0f);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Logarithmic opacity drag (0 to 10).\nHold Alt for 100x fine-tuning down to 0.00000001.\nHold Shift for fast scrub.\nCtrl+Click to type exact value.");
                    }

                    ImGui::Spacing();

                    auto make_config = [&]() {
                        rendering::ProceduralRingConfig cfg;
                        cfg.seed = static_cast<uint32_t>(re.procedural.seed);
                        cfg.band_count = re.procedural.band_count;
                        cfg.gap_count = re.procedural.gap_count;
                        cfg.striation_freq = re.procedural.striation_freq;
                        cfg.contrast = re.procedural.contrast;
                        cfg.base_density = re.procedural.base_density;
                        cfg.core_boost = re.procedural.core_boost;
                        cfg.inner_radius_ratio = re.procedural.inner_mult;
                        cfg.outer_radius_ratio = re.procedural.outer_mult;
                        cfg.tint_color = re.procedural.tint;
                        cfg.opacity = re.procedural.opacity;
                        const auto& p = rendering::get_ring_preset(re.procedural.preset_idx);
                        cfg.asymmetry = p.asymmetry;
                        cfg.backscatter = p.backscatter;
                        cfg.scatter = p.scatter;
                        cfg.unlit_factor = p.unlit_factor;
                        cfg.color_deep = p.color_deep;
                        cfg.color_mid = p.color_mid;
                        cfg.color_thin = p.color_thin;
                        return cfg;
                    };

                    if (ImGui::Button("Generate & Apply Rings##proc", ImVec2(-1.0f, 0.0f))) {
                        auto cfg = make_config();
                        auto result = rendering::generate_procedural_ring_profile(cfg);
                        celestial_renderer_.set_custom_ring_texture(system().name(), b.definition.name, result.profile);
                        ephemeris::RingLayerDefinition nl{};
                        rendering::apply_procedural_ring_layer(result, nl);
                        rings = {std::move(nl)};
                        tex_in = rings[0].inner;
                        tex_out = rings[0].outer;
                        re.selected_layer = 0;
                        ring_changed = true;
                        re.status_msg = "Generated & applied procedural rings.";
                        re.error_msg.clear();
                    }

                    if (ImGui::Button("Bake & Save to System Disk##proc", ImVec2(-1.0f, 0.0f))) {
                        auto cfg = make_config();
                        auto result = rendering::generate_procedural_ring_profile(cfg);
                        ephemeris::RingLayerDefinition nl{};
                        if (rendering::bake_and_save_procedural_ring_texture(system().name(), b.definition.name, result, nl)) {
                            rings = {std::move(nl)};
                            tex_in = rings[0].inner;
                            tex_out = rings[0].outer;
                            re.selected_layer = 0;
                            celestial_renderer_.invalidate_ring_textures();

                            auto res = simulation_.apply(ephemeris::ApplyBodyRings{
                                system().generation(), insp_idx, def_rev, rings, tex_in, tex_out
                            });
                            ring_changed = false;

                            std::string save_sys_name = system().name();
                            if (save_sys_name == "Ephemeris Mode") {
                                save_sys_name = std::string(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
                            }
                            if (!save_sys_name.starts_with("Ephemeris: ") && sys_mgr_.system_exists(save_sys_name)) {
                                try {
                                    auto loaded = sys_mgr_.load_system(save_sys_name);
                                    int s_idx = loaded.definition.find_body(b.definition.name);
                                    if (s_idx >= 0) {
                                        loaded.definition.bodies[s_idx].rings = rings;
                                        loaded.definition.bodies[s_idx].ring_texture_inner = tex_in;
                                        loaded.definition.bodies[s_idx].ring_texture_outer = tex_out;
                                        sys_mgr_.save_system(save_sys_name, loaded.definition);
                                    }
                                } catch (...) {}
                            }
                            if (res) {
                                re.status_msg = "Baked procedural Rings.png & saved system to disk.";
                                re.error_msg.clear();
                            } else {
                                re.error_msg = res.error;
                            }
                        } else {
                            re.error_msg = "Failed to bake and save procedural Rings.png.";
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                }

                if (!rings.empty()) {
                    auto& layer = rings[re.selected_layer];

                    ui::section("Geometry & Appearance", "Radii in equatorial planetary radii (R_eq).");
                    float r_in = static_cast<float>(layer.inner);
                    float r_out = static_cast<float>(layer.outer);
                    if (ui::drag_float("Inner Radius (R_eq)", &r_in, 0.01f, 1.01f, 100.0f, "%.3f R")) {
                        layer.inner = std::clamp(static_cast<double>(r_in), 1.01, 100.0);
                        if (layer.outer < layer.inner + 0.01) layer.outer = layer.inner + 0.01;
                        ring_changed = true;
                    }
                    if (ui::drag_float("Outer Radius (R_eq)", &r_out, 0.01f, static_cast<float>(layer.inner + 0.01), 200.0f, "%.3f R")) {
                        layer.outer = std::clamp(static_cast<double>(r_out), layer.inner + 0.01, 200.0);
                        ring_changed = true;
                    }
                    if (r_eq_km > 0.0) {
                        ui::metric_text(std::format("Span: {:.0f} km - {:.0f} km (Width: {:.0f} km)",
                                                    layer.inner * r_eq_km, layer.outer * r_eq_km,
                                                    (layer.outer - layer.inner) * r_eq_km));
                    }

                    if (has_ring_tex_file) {
                        if (ImGui::Checkbox("Use Radial Texture (Rings.png)", &layer.is_textured)) {
                            ring_changed = true;
                        }
                    }

                    if (!layer.is_textured || !has_ring_tex_file) {
                        float col[3] = {layer.color_rgb.r, layer.color_rgb.g, layer.color_rgb.b};
                        if (ImGui::ColorEdit3("Ring Albedo Color", col)) {
                            layer.color_rgb = glm::vec3(col[0], col[1], col[2]);
                            ring_changed = true;
                        }
                    }

                    if (ui::drag_float("Opacity / Optical Depth", &layer.opacity, 0.005f, 0.0f, 10.0f, "%.8f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat)) {
                        layer.opacity = std::clamp(layer.opacity, 0.0f, 10.0f);
                        ring_changed = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Logarithmic opacity drag (0 to 10).\nHold Alt for 100x fine-tuning down to 0.00000001.\nHold Shift for fast scrub.\nCtrl+Click to type exact value.");
                    }

                    ui::section("Radiative Transfer & Phase", "3-parameter Double Henyey-Greenstein + Chandrasekhar X/Y slab.");
                    if (ImGui::SliderFloat("Phase Balance (Back <-> Fwd)", &layer.scatter, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                        ring_changed = true;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Forward lobe weight w_f in the normalized Double Henyey-Greenstein phase function.\n0.0 = pure backscatter lobe, 1.0 = pure forward diffraction lobe.");
                    }
                    if (ImGui::SliderFloat("Forward Scatter Asym (g_f)", &layer.asymmetry, -0.99f, 0.99f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                        ring_changed = true;
                    }
                    if (ImGui::SliderFloat("Backscatter Lobe (g_b)", &layer.backscatter, -0.99f, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                        ring_changed = true;
                    }
                    if (ImGui::SliderFloat("Unlit Side Multiplier", &layer.unlit_factor, 0.0f, 2.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                        ring_changed = true;
                    }

                    if (layer.is_textured && has_ring_tex_file) {
                        ui::section("Texture Editor", "Adjust HSBA on this layer or bake into Rings.png.");
                        if (ImGui::SliderFloat("Saturation", &layer.saturation, 0.0f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                            ring_changed = true;
                        }
                        if (ImGui::SliderFloat("Hue Shift", &layer.hue_shift, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                            ring_changed = true;
                        }
                        if (ImGui::SliderFloat("Brightness", &layer.brightness, 0.0f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                            ring_changed = true;
                        }
                        if (ImGui::SliderFloat("Alpha Boost", &layer.alpha_boost, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                            ring_changed = true;
                        }

                        if (ImGui::Button("Reset Edits", ImVec2(btn_half_w, 0.0f))) {
                            layer.saturation = 1.0f;
                            layer.hue_shift = 0.0f;
                            layer.brightness = 1.0f;
                            layer.alpha_boost = 1.0f;
                            ring_changed = true;
                        }
                        ImGui::SameLine(0.0f, 8.0f);
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.42f, 0.28f, 1.0f));
                        if (ImGui::Button("Bake & Save Texture", ImVec2(btn_half_w, 0.0f))) {
                            if (rendering::bake_and_save_ring_texture(system().name(), b.definition.name, layer)) {
                                tex_in = layer.inner;
                                tex_out = layer.outer;
                                ring_changed = true;
                                celestial_renderer_.invalidate_ring_textures();
                                re.status_msg = "Baked & saved Rings.png.";
                                re.error_msg.clear();
                            } else {
                                re.error_msg = "Failed to bake and save Rings.png.";
                            }
                        }
                        ImGui::PopStyleColor();
                    }

                    if (!layer.is_textured) {
                        ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
                    }
                    if (ImGui::CollapsingHeader("Alpha Gradient###ring_alpha_gradient")) {
                        if (ImGui::Button("Add Stop")) {
                            layer.gradient.push_back({0.5f, 1.0f});
                            std::sort(layer.gradient.begin(), layer.gradient.end(),
                                      [](const ephemeris::RingGradientStop& a, const ephemeris::RingGradientStop& b_stop) {
                                          return a.p < b_stop.p;
                                      });
                            ring_changed = true;
                        }
                        int remove_stop_idx = -1;
                        bool pos_changed = false;
                        for (int si = 0; si < static_cast<int>(layer.gradient.size()); ++si) {
                            ImGui::PushID(si);
                            float avail = ImGui::GetContentRegionAvail().x;
                            float item_w = std::max(50.0f, (avail - 42.0f) * 0.5f);
                            ImGui::SetNextItemWidth(item_w);
                            if (ImGui::SliderFloat("##pos", &layer.gradient[si].p, 0.0f, 1.0f, "Pos %.2f")) {
                                pos_changed = true;
                                ring_changed = true;
                            }
                            ImGui::SameLine(0.0f, 4.0f);
                            ImGui::SetNextItemWidth(item_w);
                            if (ImGui::SliderFloat("##alpha", &layer.gradient[si].a, 0.0f, 1.0f, "Alpha %.2f")) {
                                ring_changed = true;
                            }
                            if (layer.gradient.size() > 2) {
                                ImGui::SameLine(0.0f, 4.0f);
                                if (ImGui::Button("X", ImVec2(24.0f, 0.0f))) {
                                    remove_stop_idx = si;
                                }
                            }
                            ImGui::PopID();
                        }
                        if (remove_stop_idx >= 0 && layer.gradient.size() > 2) {
                            layer.gradient.erase(layer.gradient.begin() + remove_stop_idx);
                            ring_changed = true;
                        }
                        if (pos_changed) {
                            std::sort(layer.gradient.begin(), layer.gradient.end(),
                                      [](const ephemeris::RingGradientStop& a, const ephemeris::RingGradientStop& b_stop) {
                                          return a.p < b_stop.p;
                                      });
                        }
                    }
                }

                if (ring_changed) {
                    auto res = simulation_.apply(ephemeris::ApplyBodyRings{
                        system().generation(), insp_idx, def_rev, rings, tex_in, tex_out
                    });
                    if (!res) {
                        re.error_msg = res.error;
                    } else {
                        re.error_msg.clear();
                        re.status_msg.clear();
                    }
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                std::string save_sys_name = system().name();
                if (save_sys_name == "Ephemeris Mode") {
                    save_sys_name = std::string(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
                }
                bool can_save_sys = !save_sys_name.starts_with("Ephemeris: ") && sys_mgr_.system_exists(save_sys_name);
                ImGui::BeginDisabled(!can_save_sys);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
                if (ImGui::Button("Save Ring Changes", ImVec2(-1.0f, 0.0f))) {
                    bool ok = false;
                    if (system().name() == "Ephemeris Mode") {
                        try {
                            auto solar = sys_mgr_.load_system(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
                            int s_idx = solar.definition.find_body(b.definition.name);
                            if (s_idx >= 0) {
                                solar.definition.bodies[s_idx].rings = system().body(insp_idx).definition.rings;
                                solar.definition.bodies[s_idx].ring_texture_inner = system().body(insp_idx).definition.ring_texture_inner;
                                solar.definition.bodies[s_idx].ring_texture_outer = system().body(insp_idx).definition.ring_texture_outer;
                                ok = sys_mgr_.save_system(ephemeris::SystemManager::SOLAR_SYSTEM_NAME, solar.definition);
                            }
                        } catch (...) {
                            ok = false;
                        }
                    } else {
                        ok = sys_mgr_.save_system(save_sys_name, system().definition());
                    }
                    if (ok) {
                        re.status_msg = "Saved ring configuration to system.json.";
                        re.error_msg.clear();
                    } else {
                        re.error_msg = "Failed to save system.json.";
                    }
                }
                ImGui::PopStyleColor();
                ImGui::EndDisabled();

                if (!re.status_msg.empty()) {
                    ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), "%s", re.status_msg.c_str());
                }
                if (!re.error_msg.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", re.error_msg.c_str());
                }

                ImGui::EndChild();
                ImGui::EndTabItem();
            }
        }

        if (!b.definition.is_star && !b.definition.is_black_hole && !b.definition.is_spacecraft && ImGui::BeginTabItem("Atmosphere")) {
            ImGui::BeginChild("tab_atmosphere_content", ImVec2(0.0f, 0.0f), false);
            auto& edit = ui_state_.atmosphere_edit;
            edit.synchronize(system(), insp_idx);
            auto& a = edit.values;
            ui::section("Atmosphere", "Set the gas conditions and the dust or haze suspended in it.");
            if (ImGui::Checkbox("Enable atmosphere", &a.enabled)) {
                edit.dirty = true;
                if (a.enabled) edit.use_calculated_properties();
            }
            if (a.enabled) {
                ui::section("Surface conditions");
                if (a.pressure_reference) {
                    ui::value_row("Pressure datum", "Body radius (altitude zero)");
                    ui::value_row("Modeled lower boundary", std::format("{:.1f} km", a.bottom_altitude_km));
                    if (a.cloud_deck_pressure_pa > 0) {
                        double cloud_bar = a.cloud_deck_pressure_pa/100000;
                        ui::field("Visible cloud pressure (bar)", [&](const char* id) {
                            if (ImGui::InputDouble(id, &cloud_bar, 0, 0, "%.4g")) {
                                a.cloud_deck_pressure_pa = cloud_bar*100000;
                                edit.use_calculated_properties();
                            }
                        });
                    }
                }
                double pressure_atm = a.pressure_pa/101325.0;
                ui::field(a.pressure_reference ? "Reference pressure (atm)" : "Surface pressure (atm)", [&](const char* id) {
                    if (ImGui::InputDouble(id, &pressure_atm, 0, 0, "%.6g")) {
                        a.pressure_pa = pressure_atm*101325.0;
                        edit.use_calculated_properties();
                    }
                });
                ui::field(a.pressure_reference ? "Reference temperature (K)" : "Surface temperature (K)", [&](const char* id) {
                    if (ImGui::InputDouble(id, &a.temperature_k, 0, 0, "%.2f")) {
                        for (auto& knot : a.temperature_layers) if (knot.x == 0) knot.y = a.temperature_k;
                        edit.use_calculated_properties();
                    }
                });
                ui::section("Gas composition", "Percentages total 100%. Changing one adjusts the other gases.");
                static constexpr const char* gas_names[] = {"Nitrogen", "Oxygen", "Argon", "Carbon dioxide", "Methane",
                    "Hydrogen", "Helium", "Water vapor", "Ammonia", "Ozone", "Sulfur dioxide"};
                for (size_t i = 0; i < physics::ATMOSPHERIC_GASES.size(); ++i) {
                    const auto& gas = physics::ATMOSPHERIC_GASES[i];
                    auto found = a.composition.find(gas.name);
                    if (found == a.composition.end()) continue;
                    double total = 0;
                    for (const auto& [_, weight] : a.composition) total += weight;
                    double percentage = total > 0 ? found->second/total*100.0 : 0.0;
                    ui::field(std::format("{} ({}) %", gas_names[i], gas.name), [&](const char* id) {
                        if (ImGui::InputDouble(id, &percentage, 0, 0, "%.6g")) edit.set_gas_percentage(gas.name, percentage);
                    });
                }
                if (ImGui::BeginCombo("Add gas", "Choose a gas...")) {
                    for (size_t i = 0; i < physics::ATMOSPHERIC_GASES.size(); ++i) {
                        const auto& gas = physics::ATMOSPHERIC_GASES[i];
                        if (!a.composition.contains(gas.name) && ImGui::Selectable(std::format("{} ({})", gas_names[i], gas.name).c_str())) {
                            a.composition[gas.name] = 0.0;
                            edit.use_calculated_properties();
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::BeginDisabled(a.composition.size() <= 1);
                if (ImGui::BeginCombo("Remove gas", "Choose a gas...")) {
                    for (size_t i = 0; i < physics::ATMOSPHERIC_GASES.size(); ++i) {
                        const auto& gas = physics::ATMOSPHERIC_GASES[i];
                        if (a.composition.contains(gas.name) && ImGui::Selectable(std::format("{} ({})", gas_names[i], gas.name).c_str()))
                            edit.remove_gas(gas.name);
                    }
                    ImGui::EndCombo();
                }
                ImGui::EndDisabled();
                ui::section("Dust and haze", "Particles are independent of the gas mixture. Optical depth describes the column above the modeled lower boundary at 550 nm.");
                if (!a.automatic_aerosols) {
                    ImGui::TextWrapped("Imported custom optical coefficients are preserved.");
                    if (ImGui::Button("Replace with particle layers")) {
                        a.automatic_aerosols = true;
                        a.aerosols.clear();
                        edit.use_calculated_properties();
                    }
                } else {
                    int remove = -1;
                    static constexpr const char* materials[] = {"mineral_dust", "organic_haze", "water", "ice", "sulfate",
                        "ammonia_ice", "ammonium_hydrosulfide", "methane_ice", "hydrogen_sulfide_ice", "sulfate_with_uv_absorber"};
                    static constexpr const char* labels[] = {"Mineral dust", "Organic haze", "Water droplets", "Ice particles", "Sulfate droplets",
                        "Ammonia ice", "Ammonium hydrosulfide", "Methane ice", "Hydrogen sulfide ice", "Sulfate + UV absorber (approx.)"};
                    for (size_t layer_index = 0; layer_index < a.aerosols.size(); ++layer_index) {
                        ImGui::PushID(int(layer_index));
                        auto& layer = a.aerosols[layer_index];
                        int material = 0;
                        for (int k = 0; k < int(std::size(materials)); ++k) if (layer.material == materials[k]) material = k;
                        ui::field("Particle material", [&](const char* id) {
                            if (ImGui::Combo(id, &material, labels, int(std::size(labels)))) { layer.material = materials[material]; edit.use_calculated_properties(); }
                        });
                        auto input = [&](const char* label, double& value) {
                            ui::field(label, [&](const char* id) {
                                if (ImGui::InputDouble(id, &value, 0, 0, "%.5g")) edit.use_calculated_properties();
                            });
                        };
                        input("Optical depth (550 nm)", layer.optical_depth_550);
                        input("Effective particle radius (um)", layer.effective_radius_um);
                        input("Size variance", layer.effective_variance);
                        bool detached = layer.width_km > 0;
                        if (ImGui::Checkbox("Detached layer", &detached)) {
                            layer.width_km = detached ? 10.0 : 0.0;
                            layer.peak_km = detached ? 50.0 : 0.0;
                            edit.use_calculated_properties();
                        }
                        if (detached) { input("Peak altitude (km)", layer.peak_km); input("Layer width (km)", layer.width_km); }
                        else input("Particle scale height (km)", layer.scale_height_km);
                        if (ImGui::Button("Remove layer")) remove = int(layer_index);
                        ImGui::PopID();
                    }
                    if (remove >= 0) { a.aerosols.erase(a.aerosols.begin()+remove); edit.use_calculated_properties(); }
                    ImGui::BeginDisabled(a.aerosols.size() >= ephemeris::MAX_AEROSOL_LAYERS);
                    if (ImGui::Button("Add particle layer")) {
                        a.aerosols.push_back({"water", .05, .1, .2, 1.2});
                        edit.use_calculated_properties();
                    }
                    ImGui::EndDisabled();
                    if (a.aerosols.empty()) ImGui::TextWrapped("Clear gas: no dust or haze layers.");
                }
                ui::section("Calculated properties", edit.dirty ? "Apply your changes to update these values." : "Based on the atmosphere and the planet's gravity.");
                if (auto prepared = b.render.atmosphere) {
                    ui::value_row("Air density", std::format("{:.4g} kg/m3", prepared->surface.density_kg_m3));
                    ui::value_row("Gravity", std::format("{:.3f} m/s2", prepared->surface.gravity_m_s2));
                    ui::value_row("Atmosphere thickness", std::format("{:.2f} km", prepared->height_km));
                    ui::value_row("Density scale height", std::format("{:.2f} km", prepared->surface.scale_height_km));
                } else ImGui::TextWrapped("Apply to calculate the atmosphere from these inputs.");
            }
            ImGui::BeginDisabled(!edit.dirty);
            if (ImGui::Button("Apply atmosphere", ImVec2(-1.0f, 0.0f))) {
                auto result = simulation_.apply(edit.command());
                if (result) edit.reload(system(), insp_idx);
                else edit.error = result.error;
            }
            ImGui::EndDisabled();
            if (ImGui::Button("Revert atmosphere", ImVec2(-1.0f, 0.0f))) edit.reload(system(), insp_idx);
            ImGui::BeginDisabled(edit.dirty || !sys_mgr_.system_exists(system().name()));
            if (ImGui::Button("Save atmosphere to system", ImVec2(-1.0f, 0.0f))) {
                try {
                    auto saved = sys_mgr_.load_system(system().name());
                    int index = saved.definition.find_body(system().body(insp_idx).definition.name);
                    bool ok = index >= 0;
                    if (ok) {
                        saved.definition.bodies[index].atmosphere = system().body(insp_idx).definition.atmosphere;
                        ok = sys_mgr_.save_system(system().name(), saved.definition);
                    }
                    edit.error = ok ? "Atmosphere saved to system.json." : "Unable to save atmosphere.";
                } catch (const std::exception& error) {
                    edit.error = error.what();
                }
            }
            ImGui::EndDisabled();
            if (!edit.error.empty()) ImGui::TextWrapped("%s", edit.error.c_str());
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
    ImGui::EndChild();

    if (physical_editing) {
        ImGui::Separator();
        if (!edit_notice.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::TextWrapped("%s", edit_notice.c_str());
            ImGui::PopStyleColor();
        }
        if (roche_violates) {
            if (ImGui::Button("Cancel##edit_cancel_disabled", ImVec2(-1.0f, 0.0f))) {
                ui_state_.edit_mode = false;
                draft.reload(system(), insp_idx);
            }
        } else {
            float half_w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
            if (ImGui::Button("Apply Changes##apply_edits", ImVec2(half_w, 0.0f))) {
                auto res = simulation_.apply(draft.command());
                if (!res) draft.error = res.error;
                else {
                    draft.reload(system(), insp_idx);
                    ui_state_.edit_mode = false;
                }
            }
            ImGui::PopStyleColor();
            ImGui::SameLine();
            if (ImGui::Button("Cancel##edit_cancel", ImVec2(half_w, 0.0f))) {
                ui_state_.edit_mode = false;
                draft.reload(system(), insp_idx);
            }
        }
        if (can_del) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("Delete This Body##inspector_del", ImVec2(-1.0f, 0.0f))) {
                if (ui_settings_.confirm_delete) {
                    ui_state_.delete_prompt.active = true;
                    ui_state_.delete_prompt.body_index = insp_idx;
                    ui_state_.delete_prompt.body_name = b.definition.name;
                    int fallback = system().reference_parent(insp_idx);
                    if (fallback < 0 || fallback == insp_idx) fallback = system().definition().star_idx;
                    ui_state_.delete_prompt.fallback_parent_name = system().body(fallback).definition.name;
                } else {
                    pending_delete_body_ = insp_idx;
                }
            }
            ImGui::PopStyleColor(3);
        }
    }

    ImGui::End();
}

void App::render_body_surface_tab(int insp_idx) {
    const auto& b = system().body(insp_idx);
    double r_eq_km = b.definition.radius_au() * AU_TO_KM;
    double r_eq_m = r_eq_km * 1000.0;
    double polar_ratio = std::clamp(1.0 - b.definition.oblateness, 0.05, 1.0);
    double max_relief_km = std::max(0.01, r_eq_km * 0.0099);

    auto active_col = celestial_renderer_.color_source(b.definition.name);
    auto active_hgt = celestial_renderer_.height_source(b.definition.name);

    auto& se = ui_state_.surface_edit;
    if (se.body_index != insp_idx || se.system_generation != system().generation()) {
        se.body_index = insp_idx;
        se.system_generation = system().generation();
        se.albedo_stage.clear();
        se.height_stage.clear();
        se.albedo_stage.selected_lod = active_col ? static_cast<int>(active_col->level) : 3;
        se.height_stage.selected_lod = active_hgt ? static_cast<int>(active_hgt->info().data_level) : 3;
        if (b.definition.height_range_km.x != 0.0 || b.definition.height_range_km.y != 0.0) {
            se.height_min_km = b.definition.height_range_km.x;
            se.height_max_km = b.definition.height_range_km.y;
        } else if (active_hgt) {
            se.height_min_km = active_hgt->info().minimum_m * 0.001;
            se.height_max_km = active_hgt->info().maximum_m * 0.001;
        } else {
            double def_ext = std::min(10.0, r_eq_km * 0.005);
            se.height_min_km = -def_ext;
            se.height_max_km = def_ext;
        }
        se.height_range_dirty = false;
        se.status_msg.clear();
        se.error_msg.clear();
        se.invalidate_preview();
    }

    ImGuiTabItemFlags tab_flags = 0;
    if (se.focus_surface_tab) {
        tab_flags |= ImGuiTabItemFlags_SetSelected;
        se.focus_surface_tab = false;
    }

    if (!ImGui::BeginTabItem("Surface", nullptr, tab_flags)) {
        return;
    }
    ImGui::BeginChild("tab_surface_content", ImVec2(0.0f, 0.0f), false);

    ui::section("Surface & Elevation Maps", "Drag & drop an equirectangular image or click the preview box to stage and bake.");

    float avail_w = ImGui::GetContentRegionAvail().x;
    float sub_btn_w = std::max(80.0f, (avail_w - 6.0f) * 0.5f);
    bool has_albedo_ind = (active_col != nullptr) || se.albedo_stage.has_staged;
    bool has_height_ind = (active_hgt != nullptr) || se.height_stage.has_staged;

    for (int t = 0; t < 2; ++t) {
        if (t > 0) ImGui::SameLine(0.0f, 6.0f);
        bool is_sel = (se.selected_tab == t);
        if (is_sel) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.44f, 0.52f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.52f, 0.60f, 1.0f));
        }
        std::string lbl = (t == 0)
            ? std::format("Surface (Albedo){}", has_albedo_ind ? " *" : "")
            : std::format("Elevation (Height){}", has_height_ind ? " *" : "");
        if (ImGui::Button(lbl.c_str(), ImVec2(sub_btn_w, 0.0f))) {
            se.selected_tab = t;
        }
        if (is_sel) ImGui::PopStyleColor(2);
    }

    ImGui::Spacing();

    bool is_height = (se.selected_tab == 1);
    auto& stage = is_height ? se.height_stage : se.albedo_stage;
    uint64_t cur_domain = is_height
        ? (active_hgt ? active_hgt->domain().identity : 0)
        : (active_col ? active_col->domain.identity : 0);

    bool need_preview_upload = (se.preview_tab_key != se.selected_tab) ||
                               (se.preview_is_staged != stage.has_staged) ||
                               (stage.has_staged && se.preview_staged_key != stage.staged_path) ||
                               (!stage.has_staged && se.preview_domain_key != cur_domain) ||
                               (!stage.has_staged && !is_height && cur_domain == 0 && se.preview_color_key != b.definition.color_rgb);

    if (need_preview_upload && surface_preview_staging_.mapped) {
        std::vector<uint8_t> rgba;
        if (stage.has_staged && stage.preview_rgba.size() == terrain::SURFACE_PREVIEW_W * terrain::SURFACE_PREVIEW_H * 4) {
            rgba = stage.preview_rgba;
        } else if (is_height && active_hgt) {
            rgba = terrain::SurfaceBaker::build_height_pack_preview(*active_hgt);
        } else if (!is_height && active_col) {
            rgba = terrain::SurfaceBaker::build_color_pack_preview(*active_col);
        } else if (!is_height) {
            rgba = terrain::SurfaceBaker::build_solid_preview(
                b.definition.color_rgb.r, b.definition.color_rgb.g, b.definition.color_rgb.b);
        } else {
            rgba = terrain::SurfaceBaker::build_solid_preview(0.13f, 0.14f, 0.17f);
        }
        if (rgba.size() == terrain::SURFACE_PREVIEW_W * terrain::SURFACE_PREVIEW_H * 4) {
            std::memcpy(surface_preview_staging_.mapped, rgba.data(), rgba.size());
            surface_preview_dirty_ = true;
        }
        se.preview_tab_key = se.selected_tab;
        se.preview_is_staged = stage.has_staged;
        se.preview_staged_key = stage.staged_path;
        se.preview_domain_key = cur_domain;
        se.preview_color_key = b.definition.color_rgb;
    }

    float box_w = ImGui::GetContentRegionAvail().x;
    float box_h = std::max(90.0f, box_w * 0.5f);
    ImVec2 p_min = ImGui::GetCursorScreenPos();
    ImVec2 p_max(p_min.x + box_w, p_min.y + box_h);

    if (surface_preview_ds_ != VK_NULL_HANDLE) {
        ImGui::Image(reinterpret_cast<ImTextureID>(surface_preview_ds_), ImVec2(box_w, box_h));
    } else {
        ImGui::Dummy(ImVec2(box_w, box_h));
    }

    ImGui::SetCursorScreenPos(p_min);
    ImGui::InvisibleButton("##surface_drop_box", ImVec2(box_w, box_h));
    bool box_hovered = ImGui::IsItemHovered();
    bool box_clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImU32 border_col = box_hovered
        ? IM_COL32(95, 205, 230, 240)
        : (stage.has_staged ? IM_COL32(240, 195, 70, 210) : IM_COL32(90, 100, 115, 180));
    dl->AddRect(p_min, p_max, border_col, 4.0f, 0, box_hovered ? 2.0f : 1.2f);

    std::string top_badge;
    ImU32 badge_col = IM_COL32(230, 235, 245, 255);
    if (stage.has_staged) {
        top_badge = std::format("STAGED PREVIEW ({}x{})", stage.width, stage.height);
        badge_col = IM_COL32(255, 215, 80, 255);
    } else if (is_height && active_hgt) {
        bool compat = active_hgt->compatible(r_eq_m, polar_ratio);
        top_badge = std::format("ACTIVE PACK (LOD 0..{}, {} tiles){}",
                                active_hgt->info().data_level,
                                active_hgt->records().size(),
                                compat ? "" : " [SHAPE MISMATCH]");
        badge_col = compat ? IM_COL32(130, 235, 165, 255) : IM_COL32(255, 120, 100, 255);
    } else if (!is_height && active_col) {
        bool compat = active_col->compatible(r_eq_m, polar_ratio);
        top_badge = std::format("ACTIVE PACK (LOD 0..{}, {} tiles){}",
                                active_col->level,
                                active_col->count(),
                                compat ? "" : " [SHAPE MISMATCH]");
        badge_col = compat ? IM_COL32(130, 235, 165, 255) : IM_COL32(255, 120, 100, 255);
    }

    if (!top_badge.empty()) {
        ImVec2 ts = ImGui::CalcTextSize(top_badge.c_str());
        dl->AddRectFilled(ImVec2(p_min.x + 6.0f, p_min.y + 6.0f),
                          ImVec2(p_min.x + ts.x + 16.0f, p_min.y + ts.y + 12.0f),
                          IM_COL32(10, 12, 16, 200), 3.0f);
        dl->AddText(ImVec2(p_min.x + 11.0f, p_min.y + 9.0f), badge_col, top_badge.c_str());
    }

    const char* center_prompt = stage.has_staged
        ? "Click or Drop Image to Replace Staged Map"
        : ((cur_domain != 0) ? "Click or Drop Image to Stage New Map" : "Drop Equirectangular Image Here or Click to Browse");
    if (box_hovered || (cur_domain == 0 && !stage.has_staged)) {
        ImVec2 cs = ImGui::CalcTextSize(center_prompt);
        ImVec2 cp(p_min.x + (box_w - cs.x) * 0.5f, p_min.y + (box_h - cs.y) * 0.5f);
        dl->AddRectFilled(ImVec2(cp.x - 8.0f, cp.y - 4.0f),
                          ImVec2(cp.x + cs.x + 8.0f, cp.y + cs.y + 4.0f),
                          IM_COL32(12, 15, 20, 195), 4.0f);
        dl->AddText(cp, IM_COL32(240, 244, 250, 255), center_prompt);
    }

    if (box_clicked && !surface_baker_.is_baking()) {
        if (auto picked = terrain::SurfaceBaker::open_image_file_dialog()) {
            stage_surface_file(*picked, se.selected_tab);
        }
    }

    auto existing_disk_src = terrain::SurfaceBaker::find_existing_source_image(
        system().name(), b.definition.name, is_height);

    float action_half_w = std::max(70.0f, (avail_w - 6.0f) * 0.5f);
    ImGui::BeginDisabled(surface_baker_.is_baking());
    if (ImGui::Button("Browse Image...", ImVec2((stage.has_staged || cur_domain != 0) ? action_half_w : -1.0f, 0.0f))) {
        if (auto picked = terrain::SurfaceBaker::open_image_file_dialog()) {
            stage_surface_file(*picked, se.selected_tab);
        }
    }
    if (stage.has_staged) {
        ImGui::SameLine(0.0f, 6.0f);
        if (ImGui::Button("Discard Staged", ImVec2(action_half_w, 0.0f))) {
            stage.clear();
            se.invalidate_preview();
            se.status_msg = "Cleared staged preview.";
            se.error_msg.clear();
        }
    } else if (cur_domain != 0) {
        ImGui::SameLine(0.0f, 6.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.16f, 0.16f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.72f, 0.22f, 0.22f, 1.0f));
        if (ImGui::Button(is_height ? "Unbind Elevation" : "Unbind Albedo", ImVec2(action_half_w, 0.0f))) {
            terrain::SurfaceBaker::remove_from_catalog(b.definition.name, is_height);
            if (is_height) {
                celestial_renderer_.set_height_source(b.definition.name, nullptr);
                size_renderer_.set_height_source(b.definition.name, nullptr);
            } else {
                celestial_renderer_.set_color_source(b.definition.name, nullptr);
                size_renderer_.set_color_source(b.definition.name, nullptr);
            }
            se.invalidate_preview();
            se.status_msg = std::format("Removed {} pack from {}.", is_height ? "elevation" : "albedo", b.definition.name);
            se.error_msg.clear();
        }
        ImGui::PopStyleColor(2);
    }
    ImGui::EndDisabled();

    if (stage.has_staged) {
        std::filesystem::path sp(stage.staged_path);
        ui::metric_text(std::format("Staged: {} ({}x{}, Recommended LOD {})",
                                    sp.filename().string(), stage.width, stage.height, stage.recommended_lod));
    } else if (existing_disk_src) {
        ui::metric_text(std::format("Disk source available: {}", existing_disk_src->filename().string()));
    } else {
        ui::metric_text("No source image staged yet.");
    }

    if (!is_height) {
        ImGui::Spacing();
        float base_col[3] = {b.definition.color_rgb.r, b.definition.color_rgb.g, b.definition.color_rgb.b};
        if (ImGui::ColorEdit3("Fallback Base Color", base_col)) {
            glm::vec3 new_col{base_col[0], base_col[1], base_col[2]};
            simulation_.apply(ephemeris::ApplyBodySurface{
                system().generation(), insp_idx, 0, b.definition.height_range_km, new_col
            });
            save_body_surface_to_disk(b.definition.name, b.definition.height_range_km, new_col);
            se.invalidate_preview();
        }
    }

    if (is_height) {
        ui::section("Elevation Range", "Physical min and max elevation (km) saved to system.json and used when baking.");
        ui::field("Minimum Elevation (km)##surf_h_min", [&](const char* id) {
            if (ImGui::InputDouble(id, &se.height_min_km, 0.1, 1.0, "%.3f")) {
                se.height_min_km = std::clamp(se.height_min_km, -max_relief_km, max_relief_km);
                if (se.height_max_km < se.height_min_km) se.height_max_km = se.height_min_km;
                se.height_range_dirty = true;
            }
        });
        ui::field("Maximum Elevation (km)##surf_h_max", [&](const char* id) {
            if (ImGui::InputDouble(id, &se.height_max_km, 0.1, 1.0, "%.3f")) {
                se.height_max_km = std::clamp(se.height_max_km, -max_relief_km, max_relief_km);
                if (se.height_min_km > se.height_max_km) se.height_min_km = se.height_max_km;
                se.height_range_dirty = true;
            }
        });
        ui::metric_text(std::format("Total Vertical Relief: {:.3f} km (1% Radius Envelope: ±{:.2f} km)",
                                    se.height_max_km - se.height_min_km, max_relief_km));
        if (active_hgt) {
            ui::metric_text(std::format("Active Baked Pack Envelope: [{:.3f}, {:.3f}] km",
                                        active_hgt->info().minimum_m * 0.001,
                                        active_hgt->info().maximum_m * 0.001));
        }

        if (ImGui::Button("Apply & Save Elevation Range to JSON", ImVec2(-1.0f, 0.0f))) {
            glm::dvec2 range_km{se.height_min_km, se.height_max_km};
            auto res = simulation_.apply(ephemeris::ApplyBodySurface{
                system().generation(), insp_idx, 0, range_km, b.definition.color_rgb
            });
            if (!res && !res.error.empty()) {
                se.error_msg = res.error;
                se.status_msg.clear();
            } else {
                bool saved = save_body_surface_to_disk(b.definition.name, range_km, b.definition.color_rgb);
                se.height_range_dirty = false;
                se.error_msg.clear();
                se.status_msg = saved
                    ? std::format("Saved height_range_km [{:.3f}, {:.3f}] km to system.json.", range_km.x, range_km.y)
                    : std::format("Applied height_range_km [{:.3f}, {:.3f}] km in simulation.", range_km.x, range_km.y);
            }
        }
    }

    ui::section("Tile Pack Baker", "Select target LOD pyramid depth (0..5) and compute backend.");

    static constexpr const char* lod_labels[] = {
        "LOD 0  (6 tiles, ~0.4 MB — 0.5K)",
        "LOD 1  (30 tiles, ~2.0 MB — 1K)",
        "LOD 2  (126 tiles, ~8.4 MB — 2K)",
        "LOD 3  (510 tiles, ~34 MB — 4K)",
        "LOD 4  (2,046 tiles, ~136 MB — 8K)",
        "LOD 5  (8,190 tiles, ~545 MB — 16K+)"
    };
    stage.selected_lod = std::clamp(stage.selected_lod, 0, 5);
    ui::field("Target LOD Level##surf_lod", [&](const char* id) {
        ImGui::Combo(id, &stage.selected_lod, lod_labels, static_cast<int>(std::size(lod_labels)));
    });
    if (stage.has_staged && stage.selected_lod != stage.recommended_lod) {
        if (ImGui::SmallButton(std::format("Use Recommended (LOD {})", stage.recommended_lod).c_str())) {
            stage.selected_lod = stage.recommended_lod;
        }
    }

    static constexpr const char* backend_labels[] = {
        "Auto (CUDA if available, else CPU)",
        "CUDA (GPU CuPy RawModule)",
        "CPU (NumPy)"
    };
    static constexpr const char* backend_keys[] = {"auto", "cuda", "cpu"};
    se.backend_choice = std::clamp(se.backend_choice, 0, 2);
    ui::field("Compute Backend##surf_backend", [&](const char* id) {
        ImGui::Combo(id, &se.backend_choice, backend_labels, static_cast<int>(std::size(backend_labels)));
    });

    ImGui::Spacing();

    if (surface_baker_.is_baking()) {
        float prog = surface_baker_.progress();
        std::string overlay = std::format("{:.0f}% ({:.1f}s)", prog * 100.0f, surface_baker_.elapsed_seconds());
        ImGui::ProgressBar(prog, ImVec2(-1.0f, 0.0f), overlay.c_str());
        std::string bname = surface_baker_.backend_name();
        if (!bname.empty()) {
            ui::metric_text(std::format("Backend: {}", bname));
        }
        ImGui::TextWrapped("%s", surface_baker_.status_text().c_str());
    } else {
        std::filesystem::path source_to_bake;
        if (stage.has_staged) {
            source_to_bake = std::filesystem::path(stage.staged_path);
        } else if (existing_disk_src) {
            source_to_bake = *existing_disk_src;
        }
        bool can_bake = !source_to_bake.empty();

        ImGui::BeginDisabled(!can_bake);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.45f, 0.30f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.56f, 0.38f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.26f, 0.64f, 0.44f, 1.0f));
        std::string bake_btn_lbl = stage.has_staged
            ? std::format("Bake & Apply {} (LOD 0..{})", is_height ? "Elevation" : "Albedo", stage.selected_lod)
            : std::format("Re-Bake & Apply {} (LOD 0..{})", is_height ? "Elevation" : "Albedo", stage.selected_lod);
        if (ImGui::Button(bake_btn_lbl.c_str(), ImVec2(-1.0f, 32.0f))) {
            terrain::SurfaceBakeRequest req{};
            req.system_name = system().name();
            req.body_name = b.definition.name;
            req.is_height = is_height;
            req.source_file = source_to_bake;
            req.level = stage.selected_lod;
            req.polar_ratio = polar_ratio;
            req.radius_m = r_eq_m;
            req.height_min_km = se.height_min_km;
            req.height_max_km = se.height_max_km;
            req.backend = backend_keys[se.backend_choice];
            req.copy_to_system_textures = true;
            se.status_msg.clear();
            se.error_msg.clear();
            surface_baker_.start_bake(std::move(req));
        }
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();
    }

    if (!se.status_msg.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), "%s", se.status_msg.c_str());
    }
    if (!se.error_msg.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", se.error_msg.c_str());
    }

    ImGui::EndChild();
    ImGui::EndTabItem();
}

void App::render_time_hud(rendering::RenderSettings& render_draft) {
    if (!ui_settings_.show_time_hud) return;

    if (auto rec = simulation_.take_completed_timeline()) {
        ui_state_.timeline_scrub.active = true;
        ui_state_.timeline_scrub.recording = rec;
        ui_state_.timeline_scrub.scrub_index = static_cast<int>(rec->num_steps() - 1);
        ui_state_.timeline_scrub.is_playing = false;
        ui_state_.timeline_scrub.playback_speed = 1.0;
        ui_state_.timeline_scrub.accumulated_steps = 0.0;
        ui_state_.timeline_scrub.pre_timeline_snapshot = system();
        ui_state_.timeline_scrub.pre_timeline_clock = simulation_.clock();
        simulation_.apply_timeline_frame(*rec, ui_state_.timeline_scrub.scrub_index);
    }

    ui::WorkspaceLayout layout = ui_.compute_layout(ui_settings_, ui_state_,
                                                   static_cast<float>(vk_.swapchain_extent().width),
                                                   static_cast<float>(vk_.swapchain_extent().height));
    ImGui::SetNextWindowPos(ImVec2(layout.gap, layout.height - layout.transport_height - layout.gap), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(layout.width - layout.gap * 2.0f, layout.transport_height), ImGuiCond_Always);
    ImGuiWindowFlags hud_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
    if (!ImGui::Begin("TimeTransportHUD", nullptr, hud_flags)) {
        ImGui::End();
        return;
    }

    if (ui_state_.timeline_scrub.active && ui_state_.timeline_scrub.recording) {
        auto& ts = ui_state_.timeline_scrub;
        int max_step = static_cast<int>(ts.recording->num_steps() - 1);

        const char* play_label = ts.is_playing ? " Pause " : " Play ";
        if (ImGui::Button(play_label, ImVec2(65.0f * layout.scale, 0.0f))) {
            ts.is_playing = !ts.is_playing;
        }

        ImGui::SameLine(0.0f, 6.0f * layout.scale);
        ImGui::PushItemWidth(80.0f * layout.scale);
        float spd = static_cast<float>(ts.playback_speed);
        if (ImGui::SliderFloat("##scrub_speed", &spd, 0.1f, 100.0f, "%.1fx")) {
            ts.playback_speed = std::max(0.1, static_cast<double>(spd));
        }
        ImGui::PopItemWidth();

        if (ts.is_playing) {
            ts.accumulated_steps += frame_dt_.value * ts.playback_speed * 60.0;
            int adv = static_cast<int>(ts.accumulated_steps);
            if (adv > 0) {
                ts.accumulated_steps -= adv;
                ts.scrub_index = (ts.scrub_index + adv) % (max_step + 1);
                simulation_.apply_timeline_frame(*ts.recording, ts.scrub_index);
            }
        }

        ImGui::SameLine(0.0f, 8.0f * layout.scale);
        ImGui::PushItemWidth(std::max(120.0f, (layout.width - layout.gap * 2.0f) * 0.20f));
        if (ImGui::SliderInt("##timeline_scrubber", &ts.scrub_index, 0, max_step, "Step %d")) {
            simulation_.apply_timeline_frame(*ts.recording, ts.scrub_index);
        }
        ImGui::PopItemWidth();

        ImGui::SameLine(0.0f, 8.0f * layout.scale);
        std::string date_str = time::format_sim_date(system().definition().epoch_utc, system().time(), ui_state_.use_local_timezone);
        ImGui::TextColored(ImVec4(0.85f, 0.90f, 1.0f, 1.0f), "%s", date_str.c_str());

        ImGui::SameLine(0.0f, 4.0f * layout.scale);
        if (ImGui::SmallButton(ui_state_.use_local_timezone ? "Local" : "UTC")) {
            ui_state_.use_local_timezone = !ui_state_.use_local_timezone;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Local vs UTC display");

        ImGui::SameLine(0.0f, 6.0f * layout.scale);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.45f, 0.25f, 0.9f));
        if (ImGui::Button("Resume Here")) {
            size_t nb = ts.recording->num_bodies();
            std::vector<ephemeris::BodyMotion> frame_m(ts.recording->motions.begin() + ts.scrub_index * nb,
                                                       ts.recording->motions.begin() + (ts.scrub_index + 1) * nb);
            simulation_.apply(ephemeris::SyncTimelineState{ts.recording->times[ts.scrub_index], std::move(frame_m)});
            ts.clear();
        }
        ImGui::PopStyleColor();

        ImGui::SameLine(0.0f, 4.0f * layout.scale);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.2f, 0.2f, 0.9f));
        if (ImGui::Button("Cancel##scrub_cancel")) {
            auto m_span = ts.pre_timeline_snapshot.motions();
            simulation_.apply(ephemeris::SyncTimelineState{ts.pre_timeline_clock.time,
                std::vector<ephemeris::BodyMotion>(m_span.begin(), m_span.end())});
            ts.clear();
        }
        ImGui::PopStyleColor();

        ImGui::SameLine(0.0f, 4.0f * layout.scale);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.35f, 0.50f, 0.8f));
        if (ImGui::Button("Export .bsp")) {
            trigger_timeline_bsp_export();
        }
        ImGui::PopStyleColor();

        if (!ts.export_status.empty()) {
            ImGui::SameLine(0.0f, 6.0f * layout.scale);
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", ts.export_status.c_str());
        }
    } else {
        bool is_paused = simulation_.clock().paused;
        const char* play_pause_label = is_paused ? " Play " : " Pause ";
        if (ImGui::Button(play_pause_label, ImVec2(65.0f * layout.scale, 0.0f))) {
            simulation_.apply(ephemeris::SetPaused{!is_paused});
        }

        ImGui::SameLine(0.0f, 6.0f * layout.scale);
        bool is_forward = simulation_.clock().direction >= 0.0;
        const char* dir_label = is_forward ? " Forward " : " Reverse ";
        if (ImGui::Button(dir_label, ImVec2(75.0f * layout.scale, 0.0f))) {
            simulation_.apply(ephemeris::ReverseTimeDirection{});
        }

        ImGui::SameLine(0.0f, 6.0f * layout.scale);
        if (ImGui::Button("1x", ImVec2(32.0f * layout.scale, 0.0f))) {
            simulation_.apply(ephemeris::SetTimeWarp{1.0});
        }

        ImGui::SameLine(0.0f, 8.0f * layout.scale);
        float bar_w = layout.width - layout.gap * 2.0f;
        ImGui::PushItemWidth(std::max(110.0f, std::min(200.0f * layout.scale, bar_w * 0.22f)));
        float val_log = static_cast<float>(std::log10(std::max(1e-4, simulation_.clock().warp)));
        double effective_mult = simulation_.clock().effective_warp();
        std::string speed_str = time::format_time_speed(effective_mult);
        if (is_paused) {
            speed_str = "Paused (" + speed_str + ")";
        }
        std::string fmt_str;
        for (char c : speed_str) {
            if (c == '%') fmt_str += "%%";
            else fmt_str += c;
        }
        if (ui::drag_float("##speed_drag", &val_log, 0.003f, -4.0f, 16.0f, fmt_str.c_str())) {
            simulation_.apply(ephemeris::SetTimeWarp{std::pow(10.0, static_cast<double>(val_log))});
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Drag or Ctrl+Click to edit warp (10^x). Range: 10^-4 (slow-mo) to 10^16.");
        }
        ImGui::PopItemWidth();

        ImGui::SameLine(0.0f, 6.0f * layout.scale);
        if (ImGui::Button("Jump / Timeline...", ImVec2(125.0f * layout.scale, 0.0f))) {
            open_jump_timeline_modal();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Jump to specific date/time or render timeline to date.");

        if (simulation_.is_rendering_timeline()) {
            auto prog = simulation_.timeline_progress();
            ImGui::SameLine(0.0f, 6.0f * layout.scale);
            std::string prog_text = std::format("{:.1f}% ({}/{}) | {:.1f} yr/s",
                                                prog.progress * 100.0f, prog.steps_done, prog.steps_total, prog.sim_rate_yr_s);
            ImGui::ProgressBar(prog.progress, ImVec2(160.0f * layout.scale, 0.0f), prog_text.c_str());
            ImGui::SameLine(0.0f, 4.0f * layout.scale);
            if (ImGui::Button("Cancel##cancel_render")) {
                simulation_.cancel_timeline_render();
            }
        }

        std::string date_str = time::format_sim_date(system().definition().epoch_utc, system().time(), ui_state_.use_local_timezone);
        ImGui::TextColored(ImVec4(0.85f, 0.90f, 1.0f, 1.0f), "%s", date_str.c_str());

        ImGui::SameLine(0.0f, 4.0f * layout.scale);
        if (ImGui::SmallButton(ui_state_.use_local_timezone ? "Local" : "UTC")) {
            ui_state_.use_local_timezone = !ui_state_.use_local_timezone;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Local vs UTC display");

        ImGui::SameLine(0.0f, 6.0f * layout.scale);
        const char* hud_mode = "Keplerian";
        if (system().definition().motion_model == ephemeris::MotionModel::Spice) hud_mode = "SPICE";
        else if (system().definition().motion_model == ephemeris::MotionModel::NBody) hud_mode = "N-Body";
        ImGui::TextDisabled("| %s", hud_mode);

        if (!comparisons_.empty()) {
            const auto& clock = simulation_.clock();
            double skew = comparisons_.max_time_skew_years(clock.time);
            ImGui::SameLine(0.0f, 8.0f * layout.scale);
            if (skew > comparison::ComparisonSet::skew_tolerance_years(clock, frame_dt_)) {
                ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.2f, 1.0f), "| Compared clocks differ by %.3g days", skew * 365.25);
            } else {
                ImGui::TextDisabled("| Comparing %zu system%s", comparisons_.size(), comparisons_.size() == 1 ? "" : "s");
            }
        }
    }

    float exp_val = render_draft.exposure;
    float ev = (exp_val > 0.0f) ? std::log2(exp_val) : -10.0f;
    double cam_vel_speed = glm::length(camera_.cam_vel);
    std::string right_telemetry = std::format("Exp: {:.2f} EV ({:.2f}x) | Cam Vel: {}", ev, exp_val, format_speed_au_s(cam_vel_speed));
    float right_w = ImGui::CalcTextSize(right_telemetry.c_str()).x;
    float target_x = ImGui::GetWindowWidth() - right_w - ImGui::GetStyle().WindowPadding.x;
    if (target_x > ImGui::GetCursorPosX()) {
        ImGui::SameLine(target_x);
    } else {
        ImGui::SameLine();
    }
    ImGui::TextColored(ImVec4(0.70f, 0.85f, 0.95f, 1.0f), "%s", right_telemetry.c_str());

    ImGui::End();
}

void App::render_add_body_modal() {
    if (!ui_state_.add_body.active) return;

    ImGui::SetNextWindowSize(ImVec2(440.0f, 620.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Add Orbiting Body", &ui_state_.add_body.active, ImGuiWindowFlags_NoCollapse)) {
        auto& ad = ui_state_.add_body;
        
        ui::heading("Add Celestial Body", "Configure physical properties and initial Keplerian orbit.");

        if (ad.parent_idx < 0 || ad.parent_idx >= static_cast<int>(system().size())) {
            ad.parent_idx = system().definition().star_idx;
        }

        if (ImGui::BeginCombo("Parent Body", system().body(ad.parent_idx).definition.name.c_str())) {
            for (size_t i = 0; i < system().size(); ++i) {
                bool is_selected = (ad.parent_idx == static_cast<int>(i));
                if (ImGui::Selectable(system().body(i).definition.name.c_str(), is_selected)) {
                    ad.reset_for_parent(system(), static_cast<int>(i));
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        char name_buf[128];
        std::strncpy(name_buf, ad.name.c_str(), sizeof(name_buf) - 1);
        name_buf[sizeof(name_buf) - 1] = '\0';
        if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
            ad.name = name_buf;
        }

        const char* types[] = {"Terrestrial", "Moon", "Gas Giant", "Ice Giant", "Dwarf Planet", "Star"};
        int type_idx = 0;
        for (int i = 0; i < 6; ++i) {
            if (ad.type == types[i]) { type_idx = i; break; }
        }
        if (ImGui::Combo("Type", &type_idx, types, 6)) {
            ad.type = types[type_idx];
            ad.recompute_hydrostatic();
        }

        double parent_m = system().body(ad.parent_idx).definition.mass_msun;
        double child_m = ad.get_mass_msun();
        double semi_a = ad.get_semi_major_au();

        if (ad.type == "Star") {
            ui::render_star_creator_widgets(ad.star);
        } else {
            ImGui::ColorEdit3("Color", &ad.color_rgb.x);

            const char* mass_units[] = {"Earth Masses (M\u2295)", "Lunar Masses (M\u263D)", "Solar Masses (M\u2609)"};
            if (ImGui::Combo("Mass Unit", &ad.mass_unit, mass_units, 3)) {
                ad.recompute_hydrostatic();
            }
            if (ImGui::InputDouble("Mass", &ad.mass_input, 0.01, 0.1, "%.6f")) {
                if (ad.mass_input < 0.0) ad.mass_input = 0.0;
                ad.recompute_hydrostatic();
            }

            if (ImGui::InputDouble("Radius (km)", &ad.radius_km, 10.0, 100.0, "%.1f")) {
                if (ad.radius_km < 0.1) ad.radius_km = 0.1;
                ad.recompute_hydrostatic();
            }

            ui::section("Rotation & Shape");

            if (ImGui::Checkbox("Tidally Locked", &ad.tidally_locked)) {
                ad.recompute_hydrostatic();
            }

            if (ad.tidally_locked) {
                double tot_m = parent_m + child_m;
                double p_years = (tot_m > 0.0 && semi_a > 0.0) ? std::sqrt(semi_a * semi_a * semi_a / tot_m) : 0.0;
                ad.rotation_period_hr = p_years * 365.25 * 24.0;
                ImGui::TextDisabled("  Rot Period: %.4f h [Synchronous]", ad.rotation_period_hr);
            } else {
                if (ImGui::InputDouble("Rot Period (hours)", &ad.rotation_period_hr, 1.0, 10.0, "%.4f")) {
                    if (ad.rotation_period_hr < 0.0) ad.rotation_period_hr = 0.0;
                    ad.recompute_hydrostatic();
                }
            }

            ImGui::InputDouble("Axial Tilt (deg)", &ad.axial_tilt_deg, 1.0, 5.0, "%.2f");

            double r_m = ad.radius_km * 1000.0;
            double vol_m3 = (4.0 / 3.0) * PI * (r_m * r_m * r_m);
            double mass_kg = child_m * 1.98847e30;
            double density_g_cm3 = (vol_m3 > 0.0) ? ((mass_kg / vol_m3) / 1000.0) : 0.0;
            ui::metric_text(std::format("Mean Density: {:.2f} g/cm\u00B3", density_g_cm3));
            ui::metric_text(std::format("Oblateness (f): {:.5f}", ad.oblateness));
            if (ad.j2 > 0.0) {
                ui::metric_text(std::format("J2 Moment: {:.6e}", ad.j2));
            }
        }

        ui::section("Orbital Elements");
        if (ImGui::Button("Randomize Angles##rnd_angles")) {
            ad.randomize_angles();
        }

        const char* frames[] = {"Ecliptic", "Equatorial"};
        ImGui::Combo("Reference Frame", &ad.frame, frames, 2);

        ImGui::Checkbox("Distance in km", &ad.distance_in_km);
        if (ad.distance_in_km) {
            ImGui::InputDouble("Semi-Major Axis (km)", &ad.semi_major_input, 1000.0, 10000.0, "%.1f");
        } else {
            ImGui::InputDouble("Semi-Major Axis (AU)", &ad.semi_major_input, 0.01, 0.1, "%.6f");
        }
        if (ad.semi_major_input < 1e-6) ad.semi_major_input = 1e-6;

        if (ImGui::InputDouble("Eccentricity", &ad.eccentricity, 0.01, 0.05, "%.4f")) {
            ad.eccentricity = std::clamp(ad.eccentricity, 0.0, 0.999);
        }
        ImGui::InputDouble("Inclination (deg)", &ad.inc_deg, 0.5, 5.0, "%.3f");
        ImGui::InputDouble("Long Asc Node (\u03A9 deg)", &ad.omega_node_deg, 1.0, 10.0, "%.3f");
        ImGui::InputDouble("Arg Periapsis (\u03C9 deg)", &ad.arg_peri_deg, 1.0, 10.0, "%.3f");
        ImGui::InputDouble("Mean Anomaly (M deg)", &ad.mean_anom_deg, 1.0, 10.0, "%.3f");

        double p_peri = semi_a * (1.0 - ad.eccentricity);
        double p_apo = semi_a * (1.0 + ad.eccentricity);
        ui::value_row("Periapsis", format_distance_au(p_peri, ui_settings_.ly_threshold_au));
        ui::value_row("Apoapsis", format_distance_au(p_apo, ui_settings_.ly_threshold_au));

        double eff_r_km = (ad.type == "Star") ? (ad.star.preview.radius_rsun * SOLAR_RADIUS_KM) : ad.radius_km;
        if (parent_m > 0.0 && child_m > 0.0 && eff_r_km > 0.0) {
            double roche_au = 2.44 * eff_r_km / AU_TO_KM * std::cbrt(parent_m / child_m);
            if (p_peri < roche_au) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Warning: Periapsis is inside Roche limit (%s)", format_distance_au(roche_au, ui_settings_.ly_threshold_au).c_str());
            }
        }

        if (!ad.error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", ad.error.c_str());
        }

        ImGui::Separator();
        float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
        if (ImGui::Button("Spawn Body", ImVec2(half_w, 32.0f))) {
            auto cmd = ad.build_command(system());
            auto res = simulation_.apply(cmd);
            if (!res) {
                ad.error = res.error;
            } else {
                ad.active = false;
                int new_idx = static_cast<int>(system().size() - 1);
                ui_state_.inspected_body = new_idx;
                ui_state_.inspect_barycenter = false;
                camera_.focus_body(new_idx, system(), true, false);
            }
        }
        ImGui::PopStyleColor();
        ImGui::SameLine(0.0f, 10.0f);
        if (ImGui::Button("Cancel", ImVec2(half_w, 32.0f))) {
            ad.active = false;
        }
    }
    ImGui::End();
}

void App::render_delete_body_modal() {
    if (!ui_state_.delete_prompt.active) return;

    ImGui::SetNextWindowSize(ImVec2(380.0f, 200.0f), ImGuiCond_Always);
    ImGui::OpenPopup("Delete Celestial Body?###delete_body_modal");
    if (ImGui::BeginPopupModal("Delete Celestial Body?###delete_body_modal", &ui_state_.delete_prompt.active, ImGuiWindowFlags_NoResize)) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Permanently remove body from simulation:");
        ImGui::Spacing();
        ImGui::Text("Body: %s", ui_state_.delete_prompt.body_name.c_str());
        if (!ui_state_.delete_prompt.fallback_parent_name.empty()) {
            ImGui::TextWrapped("Any orbiting moons or children will be reparented to: %s", ui_state_.delete_prompt.fallback_parent_name.c_str());
        }
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.15f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.25f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.95f, 0.35f, 0.35f, 1.0f));
        if (ImGui::Button("Delete Body", ImVec2(half_w, 28.0f))) {
            int del_idx = ui_state_.delete_prompt.body_index;
            delete_body_immediate(del_idx);
            ui_state_.delete_prompt.active = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleColor(3);

        ImGui::SameLine(0.0f, 10.0f);
        if (ImGui::Button("Cancel", ImVec2(half_w, 28.0f))) {
            ui_state_.delete_prompt.active = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void App::delete_body_immediate(int del_idx) {
    if (del_idx < 0 || del_idx >= static_cast<int>(system().size())) return;
    if (del_idx == system().definition().star_idx || system().body(del_idx).definition.is_star) return;
    if (system().definition().motion_model == ephemeris::MotionModel::Spice) return;

    auto cmd = ephemeris::DeleteBody{system().generation(), del_idx};
    auto res = simulation_.apply(cmd);
    if (!res) {
        simulation_error_ = res.error;
        return;
    }

    simulation_error_.clear();

    if (camera_.tracking_idx == del_idx) {
        camera_.tracking_idx = -1;
        camera_.tracking_barycenter = false;
    } else if (camera_.tracking_idx > del_idx) {
        camera_.tracking_idx -= 1;
    }

    if (camera_.centered_idx == del_idx) {
        camera_.centered_idx = -1;
        camera_.centered_barycenter = false;
    } else if (camera_.centered_idx > del_idx) {
        camera_.centered_idx -= 1;
    }

    if (ui_state_.inspected_body == del_idx) {
        ui_state_.inspected_body = -1;
        ui_state_.edit_mode = false;
        ui_state_.inspect_barycenter = false;
    } else if (ui_state_.inspected_body > del_idx) {
        ui_state_.inspected_body -= 1;
        if (ui_state_.edit_mode && ui_state_.has_selection(system())) {
            ui_state_.body_draft.reload(system(), ui_state_.inspected_body);
        }
    }

    if (ui_state_.add_body.active) {
        if (ui_state_.add_body.parent_idx == del_idx) {
            ui_state_.add_body.active = false;
        } else if (ui_state_.add_body.parent_idx > del_idx) {
            ui_state_.add_body.parent_idx -= 1;
        }
    }

    if (ui_state_.delete_prompt.active) {
        if (ui_state_.delete_prompt.body_index == del_idx) {
            ui_state_.delete_prompt.active = false;
        } else if (ui_state_.delete_prompt.body_index > del_idx) {
            ui_state_.delete_prompt.body_index -= 1;
        }
    }
}

void App::render_create_system_modal() {
    if (!ui_state_.create_system.active) return;

    auto& cs = ui_state_.create_system;
    ImGui::SetNextWindowSize(ImVec2(520.0f, 650.0f), ImGuiCond_FirstUseEver);
    ImGui::OpenPopup("Create New Star System###create_system_modal");
    if (ImGui::BeginPopupModal("Create New Star System###create_system_modal", &cs.active)) {
        char sys_buf[128];
        std::strncpy(sys_buf, cs.system_name.c_str(), sizeof(sys_buf) - 1);
        sys_buf[sizeof(sys_buf) - 1] = '\0';
        if (ImGui::InputText("System Name", sys_buf, sizeof(sys_buf))) {
            cs.system_name = sys_buf;
        }

        char star_buf[128];
        std::strncpy(star_buf, cs.star_name.c_str(), sizeof(star_buf) - 1);
        star_buf[sizeof(star_buf) - 1] = '\0';
        if (ImGui::InputText("Primary Star Name", star_buf, sizeof(star_buf))) {
            cs.star_name = star_buf;
        }

        ImGui::Separator();
        ui::heading("Primary Star Configuration", "Design the central star using astrophysical tracks or surface properties.");

        ui::render_star_creator_widgets(cs.star);

        if (!cs.error.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", cs.error.c_str());
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
        if (ImGui::Button("Create System", ImVec2(half_w, 32.0f))) {
            if (cs.system_name.empty()) {
                cs.error = "System name cannot be empty.";
            } else if (cs.system_name == ephemeris::SystemManager::SOLAR_SYSTEM_NAME ||
                       cs.system_name == "Ephemeris Mode" ||
                       sys_mgr_.system_exists(cs.system_name)) {
                cs.error = "A system with this name already exists.";
            } else {
                bool ok = sys_mgr_.create_new_system_from_props(cs.system_name, cs.star_name, cs.star.preview, cs.star.params);
                if (!ok) {
                    cs.error = "Failed to write system file to disk.";
                } else {
                    system_names_ = sys_mgr_.list_systems();
                    std::string new_sys_name = cs.system_name;
                    cs.active = false;
                    ImGui::CloseCurrentPopup();
                    switch_system(new_sys_name);
                }
            }
        }
        ImGui::PopStyleColor();

        ImGui::SameLine(0.0f, 10.0f);
        if (ImGui::Button("Cancel", ImVec2(half_w, 32.0f))) {
            cs.active = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void App::render_delete_system_modal() {
    if (!ui_state_.confirm_delete_system) return;

    ImGui::SetNextWindowSize(ImVec2(400.0f, 180.0f), ImGuiCond_Always);
    ImGui::OpenPopup("Delete System?###delete_system_modal");
    if (ImGui::BeginPopupModal("Delete System?###delete_system_modal", &ui_state_.confirm_delete_system, ImGuiWindowFlags_NoResize)) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Permanently delete system directory and files?");
        ImGui::Spacing();
        ImGui::Text("System: %s", ui_state_.delete_system_name.c_str());
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.15f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.25f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.95f, 0.35f, 0.35f, 1.0f));
        if (ImGui::Button("Delete System", ImVec2(half_w, 28.0f))) {
            std::string to_delete = ui_state_.delete_system_name;
            ui_state_.confirm_delete_system = false;
            ImGui::CloseCurrentPopup();
            sys_mgr_.delete_system(to_delete);
            system_names_ = sys_mgr_.list_systems();
            switch_system(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
        }
        ImGui::PopStyleColor(3);

        ImGui::SameLine(0.0f, 10.0f);
        if (ImGui::Button("Cancel", ImVec2(half_w, 28.0f))) {
            ui_state_.confirm_delete_system = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void App::open_export_nbody_modal() {
    if (system().definition().motion_model == ephemeris::MotionModel::NBody) return;
    auto dt = time::calculate_sim_date(system().definition().epoch_utc, system().time());
    std::string default_name = std::format("{} ({:04d}-{:02d}-{:02d})", system().name(), dt.year, dt.month, dt.day);
    ui_state_.export_nbody.reset(default_name);
}

void App::render_export_nbody_modal() {
    if (!ui_state_.export_nbody.active) return;

    ImGui::SetNextWindowSize(ImVec2(480.0f, 290.0f), ImGuiCond_Always);
    ImGui::OpenPopup("Export to N-Body System?###export_nbody_modal");
    if (ImGui::BeginPopupModal("Export to N-Body System?###export_nbody_modal", &ui_state_.export_nbody.active, ImGuiWindowFlags_NoResize)) {
        auto& ed = ui_state_.export_nbody;

        ui::heading("Export to N-Body System", "Save current positions & velocities into a standalone N-Body system.");

        const char* current_mode_str = "Analytical Keplerian";
        if (system().definition().motion_model == ephemeris::MotionModel::Spice) {
            current_mode_str = "NASA SPICE Ephemeris Playback";
        } else if (system().definition().motion_model == ephemeris::MotionModel::NBody) {
            current_mode_str = "IAS15 N-Body Integrator";
        }

        std::string date_str = time::format_sim_date(system().definition().epoch_utc, system().time());
        ImGui::Text("Source: %s", current_mode_str);
        ImGui::Text("Sim Epoch: %s", date_str.c_str());
        ImGui::TextDisabled("Target Motion Model: N-Body (IAS15 Integrator)");
        ImGui::Spacing();

        ui::field("System Name", [&](const char* id) {
            char buf[128];
            std::strncpy(buf, ed.system_name.c_str(), sizeof(buf));
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputText(id, buf, sizeof(buf))) {
                ed.system_name = buf;
            }
        });

        ImGui::Checkbox("Switch to new system after export", &ed.switch_after_export);

        if (!ed.error.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", ed.error.c_str());
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
        if (ImGui::Button("Export System", ImVec2(half_w, 32.0f))) {
            if (ed.system_name.empty()) {
                ed.error = "System name cannot be empty.";
            } else if (ed.system_name == ephemeris::SystemManager::SOLAR_SYSTEM_NAME ||
                       ed.system_name == "Ephemeris Mode" ||
                       sys_mgr_.system_exists(ed.system_name)) {
                ed.error = "A system with this name already exists.";
            } else {
                bool ok = sys_mgr_.export_snapshot_to_nbody(ed.system_name, system());
                if (!ok) {
                    ed.error = "Failed to export system to disk.";
                } else {
                    system_names_ = sys_mgr_.list_systems();
                    std::string new_sys_name = ed.system_name;
                    bool do_switch = ed.switch_after_export;
                    ed.active = false;
                    ImGui::CloseCurrentPopup();
                    if (do_switch) {
                        switch_system(new_sys_name);
                    }
                }
            }
        }
        ImGui::PopStyleColor();

        ImGui::SameLine(0.0f, 10.0f);
        if (ImGui::Button("Cancel", ImVec2(half_w, 32.0f))) {
            ed.active = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void App::open_jump_timeline_modal() {
    ui_state_.jump_timeline.reset_to_now(system().definition().epoch_utc, system().time(), !ui_state_.use_local_timezone);
}

void App::render_jump_timeline_modal() {
    if (!ui_state_.jump_timeline.active) return;

    ImGui::SetNextWindowSize(ImVec2(520.0f, 500.0f), ImGuiCond_Always);
    ImGui::OpenPopup("Jump to Date / Render Timeline###jump_timeline_modal");
    if (ImGui::BeginPopupModal("Jump to Date / Render Timeline###jump_timeline_modal", &ui_state_.jump_timeline.active, ImGuiWindowFlags_NoResize)) {
        auto& jd = ui_state_.jump_timeline;

        ui::heading("Jump to Date / Render Timeline", "Jump instantly or render a scrubbable timeline to a target epoch.");

        auto cur_utc = time::calculate_sim_date(system().definition().epoch_utc, system().time(), false);
        auto cur_local = time::calculate_sim_date(system().definition().epoch_utc, system().time(), true);

        ImGui::TextDisabled("Current Sim Time (UTC):   %04lld-%02d-%02d %02d:%02d:%02d UTC",
                            static_cast<long long>(cur_utc.year), cur_utc.month, cur_utc.day,
                            cur_utc.hour, cur_utc.minute, cur_utc.second);
        ImGui::TextDisabled("Current Sim Time (Local): %04lld-%02d-%02d %02d:%02d:%02d %s",
                            static_cast<long long>(cur_local.year), cur_local.month, cur_local.day,
                            cur_local.hour, cur_local.minute, cur_local.second, cur_local.tz.c_str());
        ImGui::Separator();

        ImGui::Text("Input Time Standard:");
        ImGui::SameLine();
        bool prev_utc = jd.is_utc;
        if (ImGui::RadioButton("Local Time", !jd.is_utc)) {
            jd.is_utc = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("UTC", jd.is_utc)) {
            jd.is_utc = true;
        }
        if (prev_utc != jd.is_utc) {
            time::DateTimeUTC current_draft{jd.target_year, jd.target_month, jd.target_day, jd.target_hour, jd.target_minute, jd.target_second, ""};
            auto sim_t = time::sim_time_from_date(system().definition().epoch_utc, current_draft, prev_utc);
            auto converted = time::calculate_sim_date(system().definition().epoch_utc, sim_t, !jd.is_utc);
            jd.target_year = converted.year;
            jd.target_month = converted.month;
            jd.target_day = converted.day;
            jd.target_hour = converted.hour;
            jd.target_minute = converted.minute;
            jd.target_second = converted.second;
        }

        ImGui::PushItemWidth(100.0f);
        ImGui::InputScalar("Year", ImGuiDataType_S64, &jd.target_year);
        ImGui::SameLine();
        ImGui::InputInt("Month", &jd.target_month);
        ImGui::SameLine();
        ImGui::InputInt("Day", &jd.target_day);
        ImGui::PopItemWidth();

        ImGui::PushItemWidth(100.0f);
        ImGui::InputInt("Hour", &jd.target_hour);
        ImGui::SameLine();
        ImGui::InputInt("Min", &jd.target_minute);
        ImGui::SameLine();
        ImGui::InputInt("Sec", &jd.target_second);
        ImGui::PopItemWidth();

        jd.target_month = std::clamp(jd.target_month, 1, 12);
        jd.target_day = std::clamp(jd.target_day, 1, 31);
        jd.target_hour = std::clamp(jd.target_hour, 0, 23);
        jd.target_minute = std::clamp(jd.target_minute, 0, 59);
        jd.target_second = std::clamp(jd.target_second, 0, 59);

        ImGui::Text("Presets:");
        ImGui::SameLine();
        if (ImGui::SmallButton("Today/Now")) {
            auto now = time::calculate_sim_date(system().definition().epoch_utc, units::Years{0.0}, !jd.is_utc);
            jd.target_year = now.year; jd.target_month = now.month; jd.target_day = now.day;
            jd.target_hour = now.hour; jd.target_minute = now.minute; jd.target_second = now.second;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("J2000")) {
            time::DateTimeUTC j2000{2000, 1, 1, 12, 0, 0, "UTC"};
            if (!jd.is_utc) {
                auto st = time::sim_time_from_date(system().definition().epoch_utc, j2000, true);
                j2000 = time::calculate_sim_date(system().definition().epoch_utc, st, true);
            }
            jd.target_year = j2000.year; jd.target_month = j2000.month; jd.target_day = j2000.day;
            jd.target_hour = j2000.hour; jd.target_minute = j2000.minute; jd.target_second = j2000.second;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("+1 Year")) { jd.target_year += 1; }
        ImGui::SameLine();
        if (ImGui::SmallButton("-1 Year")) { jd.target_year -= 1; }
        ImGui::SameLine();
        if (ImGui::SmallButton("+10 Years")) { jd.target_year += 10; }
        ImGui::SameLine();
        if (ImGui::SmallButton("+100 Years")) { jd.target_year += 100; }

        time::DateTimeUTC target_dt{jd.target_year, jd.target_month, jd.target_day, jd.target_hour, jd.target_minute, jd.target_second, ""};
        units::Years target_sim_time = time::sim_time_from_date(system().definition().epoch_utc, target_dt, jd.is_utc);
        double diff_years = target_sim_time.value - system().time().value;

        ImGui::Spacing();
        ImGui::TextDisabled("Target Sim Time: %.5f yr (%+.5f yr from current)", target_sim_time.value, diff_years);
        ImGui::Separator();

        auto motion_model = system().definition().motion_model;
        bool can_jump = (motion_model == ephemeris::MotionModel::Keplerian || motion_model == ephemeris::MotionModel::Spice);

        if (can_jump) {
            if (ImGui::Button("Jump to Date (Instant)", ImVec2(240.0f, 32.0f))) {
                auto res = simulation_.apply(ephemeris::JumpToTime{target_sim_time});
                if (res) res = comparisons_.jump(target_sim_time);
                if (!res) {
                    jd.error = res.error;
                } else {
                    jd.active = false;
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(No timeline recording)");
            ImGui::Spacing();
        }

        const char* res_names[] = {
            "Auto (Adaptive - Min Orbital Period)",
            "1,000 Steps",
            "10,000 Steps",
            "50,000 Steps",
            "200,000 Steps"
        };
        ImGui::PushItemWidth(280.0f);
        ImGui::Combo("Timeline Resolution", &jd.resolution_choice, res_names, IM_ARRAYSIZE(res_names));
        ImGui::PopItemWidth();

        size_t requested_steps = 0;
        if (jd.resolution_choice == 1) requested_steps = 1000;
        else if (jd.resolution_choice == 2) requested_steps = 10000;
        else if (jd.resolution_choice == 3) requested_steps = 50000;
        else if (jd.resolution_choice == 4) requested_steps = 200000;

        ImGui::BeginDisabled(!comparisons_.empty());
        if (ImGui::Button("Render Timeline to Date", ImVec2(240.0f, 32.0f))) {
            if (std::abs(diff_years) < 1e-9) {
                jd.error = "Target time is identical to current time.";
            } else {
                simulation_.start_timeline_render(target_sim_time, requested_steps);
                jd.active = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled(comparisons_.empty() ? "(Background worker integration)" : "(Unavailable while comparing systems)");

        if (motion_model == ephemeris::MotionModel::NBody) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f),
                               "Note: Instant jump without integration is only physical in Keplerian/SPICE modes.\n"
                               "N-Body uses Render Timeline to integrate equations of motion physically.");
        }

        if (!jd.error.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", jd.error.c_str());
        }

        ImGui::Spacing();
        ImGui::Separator();
        if (ImGui::Button("Cancel", ImVec2(120.0f, 28.0f))) {
            jd.active = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void App::trigger_timeline_bsp_export() {
    if (!ui_state_.timeline_scrub.recording || ui_state_.timeline_scrub.recording->empty()) {
        open_jump_timeline_modal();
        return;
    }
    if (ui_state_.timeline_scrub.export_in_progress) return;

    auto slugify = [](std::string_view s) {
        std::string out;
        for (char c : s) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
                out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else if (c == ' ') {
                out += '_';
            }
        }
        return out.empty() ? "system" : out;
    };

    std::string sys_slug = slugify(system().name());
    auto export_dir = paths::external_path("exports", "ephemeris", sys_slug);
    std::error_code ec;
    std::filesystem::create_directories(export_dir, ec);

    auto now_t = std::time(nullptr);
    std::tm tm_now{};
#if defined(_WIN32)
    _gmtime64_s(&tm_now, &now_t);
#else
    gmtime_r(&now_t, &tm_now);
#endif
    std::string filename = std::format("{}_timeline_{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}.bsp",
        sys_slug, tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday,
        tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec);
    auto bsp_path = export_dir / filename;

    ui_state_.timeline_scrub.export_in_progress = true;
    ui_state_.timeline_scrub.export_progress = 0.0f;
    ui_state_.timeline_scrub.export_status = "Exporting SPK...";

    std::thread([this, bsp_path, recording = ui_state_.timeline_scrub.recording,
                 def = system().definition(), hier = system().reference_hierarchy()]() {
        ephemeris::SpkExportOptions opts;
        opts.spk_type = 2;
        opts.cheb_degree = 13;
        opts.cheb_tol_km = 1.0;
        opts.use_hierarchical_center = true;
        opts.progress_callback = [this](float p, std::string_view s) {
            ui_state_.timeline_scrub.export_progress = p;
            ui_state_.timeline_scrub.export_status = std::string(s);
        };
        std::string spice_err;
        auto spice = ephemeris::SpiceEphemeris::open(spice_err);
        auto res = ephemeris::SpkExporter::export_timeline(bsp_path, *recording, def, hier, spice, opts);
        if (res.success) {
            ui_state_.timeline_scrub.export_status = "Exported: " + res.bsp_path.filename().string();
        } else {
            ui_state_.timeline_scrub.export_status = "Error: " + res.error;
        }
        ui_state_.timeline_scrub.export_in_progress = false;
    }).detach();
}

namespace {

std::filesystem::path resolve_user_bsp_path(std::string_view raw_input) {
    std::string trimmed(raw_input);
    while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\t' || trimmed.front() == '"' || trimmed.front() == '\'')) {
        trimmed.erase(trimmed.begin());
    }
    while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\t' || trimmed.back() == '"' || trimmed.back() == '\'')) {
        trimmed.pop_back();
    }
    if (trimmed.empty()) return {};
    std::filesystem::path p(trimmed);
    std::error_code ec;
    if (std::filesystem::is_regular_file(p, ec)) return p;
    std::filesystem::path rel = sf::paths::project_root() / p;
    if (std::filesystem::is_regular_file(rel, ec)) return rel;
    return p;
}

ephemeris::BspInspectResult inspect_bsp_file(const std::filesystem::path& bsp_path) {
    std::string err;
    auto sp = ephemeris::SpiceEphemeris::open(err);
    if (!sp) {
        ephemeris::BspInspectResult res;
        res.error = "SPICE runtime unavailable: " + err;
        return res;
    }
    return sp->inspect_bsp(bsp_path);
}

} // namespace

void App::render_ephem_setup_modal() {
    if (!ui_state_.ephem_setup.active) return;

    ImGui::SetNextWindowSize(ImVec2(680.0f, 650.0f), ImGuiCond_FirstUseEver);
    ImGui::OpenPopup("Ephemeris Setup###ephem_setup_modal");
    if (ImGui::BeginPopupModal("Ephemeris Setup###ephem_setup_modal", &ui_state_.ephem_setup.active)) {
        auto& draft = ui_state_.ephem_setup;
        bool in_spice_mode = (!system().empty() &&
                              system().definition().motion_model == ephemeris::MotionModel::Spice &&
                              system().definition().custom_bsp_path.empty());

        ui::heading("SPICE Ephemeris Kernel Configuration",
                    "Configure NASA NAIF default kernels and custom .bsp trajectories for Ephemeris Mode.");

        const auto& defaults = ephemeris::SpiceEphemeris::default_kernels();

        if (ImGui::BeginChild("ephem_kernels_scroll", ImVec2(0.0f, -195.0f), true)) {
            struct GroupSpec {
                const char* key;
                const char* alt_key;
                const char* title;
            };
            static constexpr GroupSpec kGroups[] = {
                {"Required Core", "Core", "Core Ephemeris (Essential)"},
                {"Mars Moons", "Inner Moons", "Inner Planet Moons (Mars / Phobos / Deimos)"},
                {"Jupiter Moons", "Jupiter", "Jupiter Moons (Optional)"},
                {"Saturn Moons", "Saturn", "Saturn Moons (Optional)"},
                {"Uranus Moons", "Uranus", "Uranus Moons (Optional)"},
                {"Neptune Moons", "Neptune", "Neptune Moons (Optional)"},
                {"Pluto System", "Pluto", "Pluto System (Optional)"},
            };

            std::vector<bool> rendered(defaults.size(), false);

            for (const auto& grp : kGroups) {
                bool has_any = false;
                for (size_t i = 0; i < defaults.size(); ++i) {
                    const auto& k = defaults[i];
                    if (k.group == grp.key || (grp.alt_key && k.group == grp.alt_key)) {
                        has_any = true;
                        break;
                    }
                }
                if (!has_any) continue;

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.78f, 0.95f, 1.0f));
                ImGui::TextUnformatted(grp.title);
                ImGui::PopStyleColor();
                ImGui::Separator();

                for (size_t i = 0; i < defaults.size(); ++i) {
                    const auto& k = defaults[i];
                    if (k.group != grp.key && (!grp.alt_key || k.group != grp.alt_key)) continue;
                    rendered[i] = true;
                    bool enabled = spice_mgr_.is_kernel_enabled(k.filename);
                    bool downloaded = !spice_mgr_.find_kernel_path(k.filename).empty();
                    std::string chk_id = "##def_chk_" + k.filename;
                    if (ImGui::Checkbox(chk_id.c_str(), &enabled)) {
                        spice_mgr_.set_kernel_enabled(k.filename, enabled);
                        if (in_spice_mode && (downloaded || !enabled)) {
                            apply_ephemeris_system();
                        }
                    }
                    ImGui::SameLine();
                    ImGui::Text("%s - %s", k.filename.c_str(), k.description.c_str());
                    ImGui::SameLine();
                    if (downloaded) {
                        ImGui::TextColored(ImVec4(0.4f, 0.88f, 0.45f, 1.0f), "[Downloaded]");
                    } else {
                        ImGui::TextDisabled("[Not Downloaded]");
                    }
                }
                ImGui::Spacing();
            }

            bool has_unrendered = false;
            for (size_t i = 0; i < defaults.size(); ++i) {
                if (!rendered[i]) {
                    has_unrendered = true;
                    break;
                }
            }
            if (has_unrendered) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.78f, 0.95f, 1.0f));
                ImGui::TextUnformatted("Other Ephemeris Kernels");
                ImGui::PopStyleColor();
                ImGui::Separator();

                for (size_t i = 0; i < defaults.size(); ++i) {
                    if (rendered[i]) continue;
                    const auto& k = defaults[i];
                    bool enabled = spice_mgr_.is_kernel_enabled(k.filename);
                    bool downloaded = !spice_mgr_.find_kernel_path(k.filename).empty();
                    std::string chk_id = "##def_chk_" + k.filename;
                    if (ImGui::Checkbox(chk_id.c_str(), &enabled)) {
                        spice_mgr_.set_kernel_enabled(k.filename, enabled);
                        if (in_spice_mode && (downloaded || !enabled)) {
                            apply_ephemeris_system();
                        }
                    }
                    ImGui::SameLine();
                    ImGui::Text("%s - %s", k.filename.c_str(), k.description.c_str());
                    ImGui::SameLine();
                    if (downloaded) {
                        ImGui::TextColored(ImVec4(0.4f, 0.88f, 0.45f, 1.0f), "[Downloaded]");
                    } else {
                        ImGui::TextDisabled("[Not Downloaded]");
                    }
                }
                ImGui::Spacing();
            }

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.78f, 0.95f, 1.0f));
            ImGui::TextUnformatted("Additional Custom Kernels (data/kernels/additional/)");
            ImGui::PopStyleColor();
            ImGui::Separator();

            auto add_kernels = spice_mgr_.list_additional_kernels();
            if (add_kernels.empty()) {
                ImGui::TextDisabled("No custom .bsp files in data/kernels/additional/.");
            } else {
                for (const auto& ak : add_kernels) {
                    bool enabled = ak.enabled;
                    std::string chk_id = "##add_chk_" + ak.name;
                    if (ImGui::Checkbox(chk_id.c_str(), &enabled)) {
                        spice_mgr_.set_kernel_enabled(ak.name, enabled);
                        if (in_spice_mode) {
                            apply_ephemeris_system();
                        }
                    }
                    ImGui::SameLine();
                    ImGui::Text("%s (%.1f KB)", ak.name.c_str(), ak.size_kb);
                    ImGui::SameLine();
                    std::string preview_btn = "Preview##prev_" + ak.name;
                    if (ImGui::SmallButton(preview_btn.c_str())) {
                        draft.external_bsp_path = ak.path.string();
                    }
                }
            }
            ImGui::EndChild();
        }

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.78f, 0.95f, 1.0f));
        ImGui::TextUnformatted("Import / Preview External .bsp Kernel");
        ImGui::PopStyleColor();

        float avail_w = ImGui::GetContentRegionAvail().x;
        float btn_import_w = 135.0f;
        ImGui::SetNextItemWidth(avail_w - btn_import_w - 8.0f);
        char path_buf[512];
        std::strncpy(path_buf, draft.external_bsp_path.c_str(), sizeof(path_buf));
        path_buf[sizeof(path_buf) - 1] = '\0';
        if (ImGui::InputTextWithHint("##ephem_import_path", "Path to .bsp file...", path_buf, sizeof(path_buf))) {
            draft.external_bsp_path = path_buf;
        }
        ImGui::SameLine(0.0f, 8.0f);
        if (ImGui::Button("Import & Enable", ImVec2(btn_import_w, 0.0f))) {
            auto resolved = resolve_user_bsp_path(draft.external_bsp_path);
            std::string err;
            if (spice_mgr_.add_additional_kernel(resolved, err)) {
                draft.error.clear();
                if (in_spice_mode) {
                    apply_ephemeris_system();
                }
            } else {
                draft.error = err;
            }
        }

        if (draft.external_bsp_path != draft.inspected_external_path) {
            draft.inspected_external_path = draft.external_bsp_path;
            draft.external_preview = {};
            draft.error.clear();
            if (!draft.external_bsp_path.empty()) {
                auto resolved = resolve_user_bsp_path(draft.external_bsp_path);
                draft.external_preview = inspect_bsp_file(resolved);
                if (!draft.external_preview.valid) {
                    draft.error = draft.external_preview.error;
                }
            }
        }

        if (ImGui::BeginChild("ephem_setup_preview", ImVec2(0.0f, 68.0f), true)) {
            if (!draft.error.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", draft.error.c_str());
            } else if (draft.external_preview.valid) {
                for (const auto& b : draft.external_preview.bodies) {
                    const char* kind = (b.id < 0) ? "Spacecraft" : ((b.id == 10) ? "Star" : "Body");
                    ImGui::Text("%s (NAIF %d) [%s]:  %s  ->  %s",
                                b.name.c_str(), b.id, kind, b.utc_start.c_str(), b.utc_end.c_str());
                }
            } else {
                ImGui::TextDisabled("Select 'Preview' on an additional kernel or enter a .bsp path above to inspect segments.");
            }
            ImGui::EndChild();
        }

        int enabled_count = 0;
        int missing_count = static_cast<int>(spice_mgr_.check_missing_kernels().size());
        for (const auto& k : defaults) {
            if (spice_mgr_.is_kernel_enabled(k.filename)) {
                ++enabled_count;
            }
        }
        ImGui::TextDisabled("Enabled Default Kernels: %d / %zu   |   Missing to Download: %d",
                            enabled_count, defaults.size(), missing_count);

        ImGui::Separator();
        ImGui::Spacing();

        float third_w = (ImGui::GetContentRegionAvail().x - 16.0f) / 3.0f;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
        if (ImGui::Button("Save & Launch Ephemeris", ImVec2(third_w, 32.0f))) {
            spice_mgr_.save_settings();
            draft.active = false;
            ImGui::CloseCurrentPopup();
            trigger_ephemeris_mode();
        }
        ImGui::PopStyleColor();

        ImGui::SameLine(0.0f, 8.0f);
        if (ImGui::Button("Save & Close", ImVec2(third_w, 32.0f))) {
            spice_mgr_.save_settings();
            draft.active = false;
            ImGui::CloseCurrentPopup();
            if (in_spice_mode) {
                trigger_ephemeris_mode();
            }
        }

        ImGui::SameLine(0.0f, 8.0f);
        if (ImGui::Button("Cancel", ImVec2(third_w, 32.0f))) {
            draft.active = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void App::render_ephem_download_modal() {
    if (!ui_state_.show_ephem_download_modal) return;

    ImGui::SetNextWindowSize(ImVec2(500.0f, 250.0f), ImGuiCond_Always);
    ImGui::OpenPopup("Downloading SPICE Ephemeris Data###ephem_download_modal");
    if (ImGui::BeginPopupModal("Downloading SPICE Ephemeris Data###ephem_download_modal", nullptr, ImGuiWindowFlags_NoResize)) {
        auto dl = spice_mgr_.download_snapshot();

        ui::heading("Downloading NASA NAIF SPICE Kernels",
                    "Fetching missing ephemeris kernels into data/kernels/default/...");

        ImGui::Text("Status: %s", dl.status_text.empty() ? "Preparing..." : dl.status_text.c_str());
        ImGui::ProgressBar(std::clamp(dl.progress, 0.0f, 1.0f), ImVec2(-1.0f, 24.0f));

        double cur_mb = static_cast<double>(dl.bytes_current) / (1024.0 * 1024.0);
        double tot_mb = static_cast<double>(dl.bytes_total) / (1024.0 * 1024.0);
        ImGui::TextDisabled("Current File: %.2f / %.2f MB   |   Speed: %s",
                            cur_mb, tot_mb, dl.speed_str.empty() ? "---" : dl.speed_str.c_str());

        ImGui::Spacing();
        if (!dl.error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", dl.error.c_str());
            ImGui::Spacing();
            float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;
            if (ImGui::Button("Retry Download", ImVec2(half_w, 30.0f))) {
                spice_mgr_.clear_download_error();
                spice_mgr_.start_download_missing();
            }
            ImGui::SameLine(0.0f, 10.0f);
            if (ImGui::Button("Close", ImVec2(half_w, 30.0f))) {
                ui_state_.show_ephem_download_modal = false;
                ImGui::CloseCurrentPopup();
            }
        } else {
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Cancel Download", ImVec2(-1.0f, 30.0f))) {
                spice_mgr_.cancel_download();
                ui_state_.show_ephem_download_modal = false;
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::EndPopup();
    }
}

void App::render_bsp_import_modal() {
    if (!ui_state_.bsp_import.active) return;

    ImGui::SetNextWindowSize(ImVec2(660.0f, 520.0f), ImGuiCond_FirstUseEver);
    ImGui::OpenPopup("Import Ephemeris Kernel (.bsp)###bsp_import_modal");
    if (ImGui::BeginPopupModal("Import Ephemeris Kernel (.bsp)###bsp_import_modal", &ui_state_.bsp_import.active)) {
        auto& bd = ui_state_.bsp_import;

        ui::heading("Import & Preview SPICE Kernel (.bsp)",
                    "Inspect DAF/SPK segments and launch standalone trajectory playback or install to Additional Kernels.");

        auto avail_files = spice_mgr_.list_available_bsp_files();
        if (bd.custom_path.empty() && !avail_files.empty()) {
            bd.custom_path = avail_files.front().path.string();
        }

        ui::field("Quick Select (.bsp)", [&](const char* id) {
            std::string current_label = "Select discovered .bsp kernel...";
            for (const auto& f : avail_files) {
                if (f.path.string() == bd.custom_path) {
                    current_label = std::format("[{}] {} ({:.0f} KB)", f.category, f.name, f.size_kb);
                    break;
                }
            }
            if (ImGui::BeginCombo(id, current_label.c_str())) {
                for (const auto& f : avail_files) {
                    bool selected = (f.path.string() == bd.custom_path);
                    std::string item_label = std::format("[{}] {} ({:.0f} KB)", f.category, f.name, f.size_kb);
                    if (ImGui::Selectable(item_label.c_str(), selected)) {
                        bd.custom_path = f.path.string();
                        bd.error.clear();
                        bd.status_msg.clear();
                    }
                }
                ImGui::EndCombo();
            }
        });

        ui::field("Kernel File Path", [&](const char* id) {
            char buf[512];
            std::strncpy(buf, bd.custom_path.c_str(), sizeof(buf));
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputTextWithHint(id, "Enter or paste path to .bsp file...", buf, sizeof(buf))) {
                bd.custom_path = buf;
                bd.error.clear();
                bd.status_msg.clear();
            }
        });

        if (bd.custom_path != bd.inspected_path) {
            bd.inspected_path = bd.custom_path;
            bd.preview = {};
            if (!bd.custom_path.empty()) {
                auto resolved = resolve_user_bsp_path(bd.custom_path);
                bd.preview = inspect_bsp_file(resolved);
            }
        }

        ImGui::Spacing();
        if (bd.preview.valid) {
            ImGui::Text("Discovered Segments: %zu bodies/spacecraft   |   Window: %s -> %s (%.1f days)",
                        bd.preview.bodies.size(),
                        bd.preview.utc_min.c_str(),
                        bd.preview.utc_max.c_str(),
                        bd.preview.duration_days);
            if (!bd.preview.sidecar.is_null() && !bd.preview.sidecar.empty()) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.45f, 1.0f), "[Sidecar Metadata JSON Detected]");
            }
        } else {
            ImGui::TextDisabled("Kernel Preview");
        }

        if (ImGui::BeginChild("bsp_import_preview_box", ImVec2(0.0f, -85.0f), true)) {
            if (!bd.preview.error.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", bd.preview.error.c_str());
            } else if (bd.preview.valid) {
                if (ImGui::BeginTable("bsp_segments_tbl", 4,
                                      ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn("Object Name", ImGuiTableColumnFlags_WidthStretch, 0.28f);
                    ImGui::TableSetupColumn("NAIF ID", ImGuiTableColumnFlags_WidthFixed, 75.0f);
                    ImGui::TableSetupColumn("Category", ImGuiTableColumnFlags_WidthFixed, 95.0f);
                    ImGui::TableSetupColumn("Coverage Window (UTC)", ImGuiTableColumnFlags_WidthStretch, 0.45f);
                    ImGui::TableHeadersRow();

                    for (const auto& b : bd.preview.bodies) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::TextUnformatted(b.name.c_str());

                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%d", b.id);

                        ImGui::TableSetColumnIndex(2);
                        if (b.id < 0) {
                            ImGui::TextColored(ImVec4(0.35f, 0.85f, 1.0f, 1.0f), "Spacecraft");
                        } else if (b.id == 10) {
                            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.35f, 1.0f), "Star");
                        } else {
                            ImGui::TextUnformatted("Planet/Moon");
                        }

                        ImGui::TableSetColumnIndex(3);
                        ImGui::Text("%s  ->  %s", b.utc_start.c_str(), b.utc_end.c_str());
                    }
                    ImGui::EndTable();
                }
            } else {
                ImGui::TextDisabled("Select or enter a valid .bsp file path to preview its SPK ephemeris segments.");
            }
            ImGui::EndChild();
        }

        if (!bd.error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", bd.error.c_str());
        } else if (!bd.status_msg.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.45f, 1.0f), "%s", bd.status_msg.c_str());
        }

        ImGui::Separator();
        ImGui::Spacing();

        bool can_act = bd.preview.valid && !bd.preview.bodies.empty();
        float third_w = (ImGui::GetContentRegionAvail().x - 16.0f) / 3.0f;

        ImGui::BeginDisabled(!can_act);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
        if (ImGui::Button("Open Ephemeris Playback", ImVec2(third_w, 32.0f))) {
            auto resolved = resolve_user_bsp_path(bd.custom_path);
            open_generic_bsp_playback(resolved);
            if (ui_state_.bsp_import.error.empty()) {
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::PopStyleColor();

        ImGui::SameLine(0.0f, 8.0f);
        if (ImGui::Button("Install to Additional", ImVec2(third_w, 32.0f))) {
            auto resolved = resolve_user_bsp_path(bd.custom_path);
            std::string err;
            if (!spice_mgr_.add_additional_kernel(resolved, err)) {
                bd.error = err;
                bd.status_msg.clear();
            } else {
                bd.error.clear();
                bd.status_msg = "Installed and enabled in data/kernels/additional/.";
                if (!system().empty() &&
                    system().definition().motion_model == ephemeris::MotionModel::Spice &&
                    system().definition().custom_bsp_path.empty()) {
                    apply_ephemeris_system();
                }
            }
        }
        ImGui::EndDisabled();

        ImGui::SameLine(0.0f, 8.0f);
        if (ImGui::Button("Cancel", ImVec2(third_w, 32.0f))) {
            bd.active = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

} // namespace sf

