#include "app.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"
#include "core/time_utils.hpp"

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
    auto bloom_desc = comp_desc;
    bloom_desc.bindings.push_back({1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
    bloom_desc.bindings.push_back({2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1});
    bloom_desc.push_constant_size = 2*sizeof(float);
    bloom_desc.macros = {{"BLOOM", "1"}};
    bloom_composite_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(vk_, bloom_desc);
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
    rhi::ShaderCache::destroy_pipeline(vk_, bloom_composite_pipeline_);
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
    bool bloom_deferred = bloom_enabled &&
        bloom_renderer_.render(cmd, vk_, hdr_color_, render_settings_.values(), bloom_scene, true);
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
    const auto& presentation = size_comparison_active_ ? size_composite_pipeline_ :
        bloom_deferred ? bloom_composite_pipeline_ : composite_pipeline_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, presentation.handle);

    if (bloom_deferred) {
        vk_.push_descriptors(cmd, presentation, {
            rhi::PushDescriptorWrite::sampled_image(0, hdr_color_, vk_.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(1, *bloom_scene, vk_.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::sampled_image(2, bloom_renderer_.halo(), vk_.sampler_linear_clamp())});
        const auto& values = render_settings_.values();
        vk_.push_constants(cmd, presentation, std::array{values.bloom_strength, values.bloom_highlight_limit});
    } else {
        rhi::PushDescriptorWrite writes[] = {
            rhi::PushDescriptorWrite::sampled_image(0, hdr_color_, vk_.sampler_linear_clamp()),
        };
        vk_.push_descriptors(cmd, presentation, writes);
    }
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

} // namespace sf
