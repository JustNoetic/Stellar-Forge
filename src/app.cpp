#include "app.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"
#include "core/time_utils.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
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

    rhi::ShaderCache::init();
    vk_.init(window_, false);
    ui_.init(window_, vk_);
    if (benchmark_.enabled) {
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::GetIO().LogFilename = nullptr;
        ui_settings_.ui_visible = benchmark_.ui;
    }

    recreate_render_targets();

    system_names_ = sys_mgr_.list_systems();
    switch_system(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
    simulation_.set_threaded(benchmark_.enabled ? benchmark_.simulation_mode != "inline" : simulation_thread);
    sf::log::info("Simulation execution: {}", simulation_.threaded() ? "worker" : "inline");

    star_catalog_.init(vk_, sf::paths::external_path("data", "gaia", "stars.bin"));
    celestial_renderer_.init(vk_);
    orbit_renderer_.init(vk_);

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
    camera_.flight_speed_au_s = 0.2;
    camera_.has_prev_spin = false;
    camera_.origin_render = system().body(camera_.tracking_idx).render.position.value;
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
        if (sys_mgr_.system_exists("Ephemeris Mode")) {
            tmpl = sys_mgr_.load_system("Ephemeris Mode");
        } else {
            tmpl = sys_mgr_.load_system(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);
        }
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
        camera_.flight_speed_au_s = 0.2;
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

        render_frame();

        ++frames_rendered;
        if (self_test_ && frames_rendered >= 3) {
            sf::log::info("Self-test completed {} frames cleanly.", frames_rendered);
            break;
        }
    }
}

void App::shutdown() {
    simulation_.set_threaded(false);
    vk_.wait_idle();
    rhi::ShaderCache::destroy_pipeline(vk_, composite_pipeline_);
    celestial_renderer_.shutdown(vk_);
    orbit_renderer_.shutdown(vk_);
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

void App::on_key(int key, int, int action, int mods) {
    if (benchmark_.enabled) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window_, GLFW_TRUE);
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
        camera_.snap_to_tracked(system());
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
    if (benchmark_.enabled) return;
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
    if (benchmark_.enabled || ui_.is_dragging()) return;
    camera_.on_cursor_pos(xpos, ypos, system());
}

void App::on_scroll(double, double yoffset) {
    if (benchmark_.enabled) return;
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    bool shift = (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ||
                 (glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    camera_.on_scroll(yoffset, shift);
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

    if (self_test_) {
        auto result = simulation_.apply(ephemeris::SetTimeWarp{simulation_.clock().warp * 1.25});
        if (!result) throw std::runtime_error(result.error);
    }
    double camera_update_ms = 0.0;
    {
        ScopedTimer timer(sample ? &camera_update_ms : nullptr);
        camera_.update(frame_dt_, system());
    }
    if (sample) sample->cpu_camera_ms += camera_update_ms;
    if (self_test_ && camera_.origin_render != system().body(camera_.tracking_idx).render.position.value)
        throw std::runtime_error("Camera tracking and rendered body use different snapshots after a time-warp change.");

    VkExtent2D ext = vk_.swapchain_extent();

    uint32_t t_scene = vk_.begin_gpu_timer(cmd, "gpu_scene_hdr");
    vk_.transition_image(cmd, hdr_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    vk_.transition_image(cmd, scene_depth_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    star_catalog_.cull(cmd, vk_, camera_, system().time(), render_settings_.values(), ext);

    auto color_att = rhi::RenderingAttachment::clear_color(hdr_color_, 0.0f, 0.0f, 0.0f, 1.0f);
    auto depth_att = rhi::RenderingAttachment::clear_depth_reverse_z(scene_depth_, 0.0f);
    vk_.begin_rendering(cmd, ext, std::span{&color_att, 1}, &depth_att);

    {
        ScopedTimer timer(sample ? &sample->cpu_celestial_ms : nullptr);
        uint32_t id = sample ? vk_.begin_gpu_timer(cmd, "gpu_celestial") : UINT32_MAX;
        celestial_renderer_.render(cmd, vk_, camera_, system(), render_settings_.values(), ext);
        vk_.end_gpu_timer(cmd, id);
    }
    {
        ScopedTimer timer(sample ? &sample->cpu_gaia_ms : nullptr);
        uint32_t id = sample ? vk_.begin_gpu_timer(cmd, "gpu_gaia") : UINT32_MAX;
        star_catalog_.render(cmd, vk_, camera_, system().time(), render_settings_.values(), ext);
        vk_.end_gpu_timer(cmd, id);
    }
    {
        ScopedTimer timer(sample ? &sample->cpu_orbits_ms : nullptr);
        uint32_t id = sample ? vk_.begin_gpu_timer(cmd, "gpu_orbits") : UINT32_MAX;
        orbit_renderer_.render(cmd, vk_, camera_, system(), render_settings_.values(), ext);
        vk_.end_gpu_timer(cmd, id);
    }

    vk_.end_rendering(cmd);
    vk_.end_gpu_timer(cmd, t_scene);

    uint32_t t_post = vk_.begin_gpu_timer(cmd, "gpu_post_composite");
    vk_.transition_image(cmd, hdr_color_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composite_pipeline_.handle);

    rhi::PushDescriptorWrite writes[] = {
        rhi::PushDescriptorWrite::sampled_image(0, hdr_color_, vk_.sampler_linear_clamp()),
    };
    vk_.push_descriptors(cmd, composite_pipeline_, writes);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vk_.end_rendering(cmd);
    vk_.end_gpu_timer(cmd, t_post);

    uint32_t t_ui = vk_.begin_gpu_timer(cmd, "gpu_ui");
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
    render_body_inspector();
    render_viewport_overlays();
    render_settings_window(render_draft);
    render_time_hud(render_draft);
    render_add_body_modal();
    render_delete_body_modal();
    render_create_system_modal();
    render_delete_system_modal();
    render_export_nbody_modal();
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
        if (ImGui::MenuItem("Export to N-Body System...")) {
            open_export_nbody_modal();
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
    if (ImGui::BeginMenu("Edit")) {
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
        ImGui::Separator();
        const char* export_physics_label = (current_mode == ephemeris::MotionModel::Spice)
            ? "Export Ephemeris to N-Body System..."
            : "Export Current State to N-Body System...";
        if (ImGui::MenuItem(export_physics_label)) {
            open_export_nbody_modal();
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Tools")) {
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
            camera_.snap_to_tracked(system());
        }
        ImGui::MenuItem("Auto Horizon Level (< 200 km)", nullptr, &camera_.horizon_align);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("System Outliner", nullptr, &ui_settings_.show_outliner);
        ImGui::MenuItem("Body Inspector", nullptr, &ui_settings_.show_inspector);
        ImGui::MenuItem("Time Transport HUD", nullptr, &ui_settings_.show_time_hud);
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

void App::render_settings_window(rendering::RenderSettings& render_draft) {
    if (!ui_settings_.show_settings) return;

    ui::WorkspaceLayout layout = ui_.compute_layout(ui_settings_, ui_state_,
                                                   static_cast<float>(vk_.swapchain_extent().width),
                                                   static_cast<float>(vk_.swapchain_extent().height));
    float width = std::min(960.0f * layout.scale, layout.width - layout.gap * 4.0f);
    float height = std::min(620.0f * layout.scale, layout.height - layout.top - layout.gap * 3.0f);
    ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2((layout.width - width) * 0.5f, (layout.height - height) * 0.5f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Settings / Stellar Forge###forge_settings", &ui_settings_.show_settings, ImGuiWindowFlags_NoCollapse)) {
        static const char* s_categories[] = {"Optics", "Visibility", "Navigation", "Workspace"};
        static const char* s_descriptions[] = {
            "Exposure, star brightness, HDR rendering and post-processing.",
            "Orbital paths, background starfield and celestial overlays.",
            "Flight speed, field of view, horizon alignment and distance threshold.",
            "Panel layout, interface visibility and window docking."
        };

        int section_idx = std::clamp(ui_settings_.settings_section, 0, 3);
        float nav_w = std::min(160.0f * layout.scale, width * 0.25f);
        float footer_h = 38.0f * layout.scale;

        ImGui::BeginChild("settings_navigation", ImVec2(nav_w, -footer_h), true);
        ImGui::TextDisabled("SETTINGS");
        ImGui::Spacing();
        for (int i = 0; i < 4; ++i) {
            if (ImGui::Selectable(s_categories[i], section_idx == i, 0, ImVec2(0.0f, 26.0f * layout.scale))) {
                section_idx = i;
                ui_settings_.settings_section = i;
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("settings_content", ImVec2(0.0f, -footer_h), true);
        ui::heading(s_categories[section_idx], s_descriptions[section_idx]);

        if (section_idx == 0) {
            ui::field("Exposure##optics_exp", [&](const char* id) {
                float speed = std::max(0.005f, render_draft.exposure * 0.02f);
                const char* fmt = (render_draft.exposure >= 10000.0f || (render_draft.exposure > 0.0f && render_draft.exposure < 0.005f)) ? "%.4g" : "%.3f";
                if (ui::drag_float(id, &render_draft.exposure, speed, 0.0f, 0.0f, fmt)) {
                    render_draft.exposure = std::max(0.0f, render_draft.exposure);
                }
            });
            ImGui::Checkbox("HDR Rendering", &render_draft.hdr_enabled);
            ImGui::Checkbox("VSync", &render_draft.vsync);
            ui::field("Point Diameter##star_pt", [&](const char* id) {
                ImGui::SliderFloat(id, &render_draft.star_point_size, 1.5f, 6.0f, "%.1f px");
            });
            ui::field("Star Brightness##star_br", [&](const char* id) {
                float star_speed = std::max(0.01f, render_draft.star_intensity * 0.02f);
                if (ui::drag_float(id, &render_draft.star_intensity, star_speed, 0.0f, 0.0f, "%.2f")) {
                    render_draft.star_intensity = std::max(0.0f, render_draft.star_intensity);
                }
            });
        } else if (section_idx == 1) {
            ImGui::Checkbox("Gaia DR3 Starfield", &render_draft.show_gaia_stars);
            ImGui::Checkbox("Orbit Lines", &render_draft.show_orbits);
            ui::field("Orbit Opacity##orb_op", [&](const char* id) {
                ImGui::SliderFloat(id, &render_draft.orbit_opacity, 0.0f, 1.0f, "%.2f");
            });
        } else if (section_idx == 2) {
            const char* modes[] = {"Free Flight", "Simple Orbit"};
            int cur_mode = camera_.movement_mode;
            if (ImGui::Combo("Movement Mode", &cur_mode, modes, 2)) {
                camera_.movement_mode = cur_mode;
            }
            float fov_f = static_cast<float>(camera_.fov_deg);
            if (ImGui::SliderFloat("Field of View", &fov_f, 1.0f, 120.0f, "%.1f deg")) {
                camera_.fov_deg = fov_f;
            }
            ui::field("Flight Speed", [&](const char* id) {
                float sp_log = static_cast<float>(std::log10(std::max(1e-12, camera_.flight_speed_au_s)));
                std::string sp_fmt = format_speed_au_s(camera_.flight_speed_au_s);
                if (ui::drag_float(id, &sp_log, 0.02f, -12.0f, 8.0f, sp_fmt.c_str())) {
                    camera_.flight_speed_au_s = std::pow(10.0, static_cast<double>(sp_log));
                }
            });
            ImGui::Checkbox("Auto Horizon Level (< 200 km)", &camera_.horizon_align);
            ui::field("Light-Year Display Threshold (AU)", [&](const char* id) {
                ImGui::InputDouble(id, &ui_settings_.ly_threshold_au, 100.0, 1000.0, "%.1f");
                ui_settings_.ly_threshold_au = std::max(1.0, ui_settings_.ly_threshold_au);
            });
        } else if (section_idx == 3) {
            ImGui::Checkbox("Aligned Panels", &ui_settings_.aligned_panels);
            ImGui::Checkbox("Show System Outliner", &ui_settings_.show_outliner);
            ImGui::Checkbox("Show Body Inspector", &ui_settings_.show_inspector);
            ImGui::Checkbox("Show Time Transport HUD", &ui_settings_.show_time_hud);
            ImGui::Checkbox("Show Engine Telemetry Window", &ui_settings_.show_telemetry);
            ImGui::Separator();
            ImGui::Checkbox("Confirm Object Deletion", &ui_settings_.confirm_delete);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Require confirmation dialog when deleting celestial bodies. Disable to quickly delete objects.");
            }
            ImGui::Separator();
            ImGui::SliderFloat("Outliner Width", &ui_settings_.outliner_width, 180.0f, 500.0f, "%.0f px");
            ImGui::SliderFloat("Inspector Width", &ui_settings_.inspector_width, 300.0f, 700.0f, "%.0f px");
            ImGui::Spacing();
            if (ImGui::Button("Reset Workspace Layout")) {
                ui_settings_.reset_layout();
            }
        }

        ImGui::EndChild();

        ImGui::Separator();
        ImGui::TextDisabled("Changes take effect immediately");
        ImGui::SameLine(ImGui::GetWindowWidth() - 95.0f * layout.scale);
        if (ImGui::Button("Close", ImVec2(85.0f * layout.scale, 0.0f))) {
            ui_settings_.show_settings = false;
        }
    }
    ImGui::End();
}

void App::render_viewport_overlays() {
    if (ui_settings_.show_telemetry) {
        ImGui::SetNextWindowSize(ImVec2(340.0f, 220.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Engine Telemetry", &ui_settings_.show_telemetry)) {
            ui::heading("GPU Telemetry", vk_.device_name());
            ui::value_row("Frame Rate", std::format("{:.1f} FPS", ImGui::GetIO().Framerate));
            const char* pm_str = "FIFO (VSync)";
            if (vk_.present_mode() == VK_PRESENT_MODE_IMMEDIATE_KHR) pm_str = "Immediate (Uncapped)";
            else if (vk_.present_mode() == VK_PRESENT_MODE_MAILBOX_KHR) pm_str = "Mailbox (Uncapped)";
            ui::value_row("Present Mode", pm_str);
            ui::value_row("Gaia Stars Loaded", std::format("{}", star_catalog_.count()));
            ui::value_row("GPU Scene HDR", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_scene_hdr")));
            ui::value_row("GPU Composite", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_post_composite")));
            ui::value_row("GPU Dear ImGui", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_ui")));
        }
        ImGui::End();
    }

    if (!ui_settings_.show_outliner) {
        ImGui::SetNextWindowPos(ImVec2(10.0f, 38.0f), ImGuiCond_Always);
        ImGuiWindowFlags pill_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove;
        if (ImGui::Begin("##CameraPill", nullptr, pill_flags)) {
            std::string tracked_name = "Free Camera";
            if (camera_.tracking_idx >= 0 && camera_.tracking_idx < static_cast<int>(system().size())) {
                tracked_name = camera_.tracking_barycenter
                    ? std::format("{} Barycenter", system().body(camera_.tracking_idx).definition.name)
                    : system().body(camera_.tracking_idx).definition.name;
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
    }
    ImGui::End();
}

void App::render_body_inspector() {
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

    float footer_h = physical_editing ? (roche_violates ? 85.0f * layout.scale : 55.0f * layout.scale) : 0.0f;

    if (ImGui::BeginTabBar("InspectorTabBar", ImGuiTabBarFlags_FittingPolicyScroll)) {
        if (ImGui::BeginTabItem("Overview")) {
            ImGui::BeginChild("tab_overview_content", ImVec2(0.0f, -footer_h), false);
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

                    bool tidally_locked = false;
                    if (has_parent && draft.values.mass.value > 0.0 && draft.values.radius.value > 0.0) {
                        double p_m = system().body(parent_idx).definition.mass_msun;
                        double r_tid = 0.00084 * std::pow(p_m * p_m / draft.values.mass.value, 1.0 / 6.0) * std::sqrt(draft.values.radius.value);
                        if (draft.values.a_au < r_tid) {
                            tidally_locked = true;
                            double tot_m = p_m + draft.values.mass.value;
                            double p_yr = tot_m > 0.0 ? std::sqrt(std::pow(draft.values.a_au, 3.0) / tot_m) : 0.0;
                            draft.values.rotation_period_hr = p_yr * 365.25 * 24.0;
                            draft.values.tidally_locked = true;
                        }
                    }

                    if (tidally_locked) {
                        ui::metric_text(std::format("Rot Period: {:.4f} hours [Tidally Locked]", draft.values.rotation_period_hr));
                    } else {
                        if (ImGui::InputDouble("Rot Period (hours)", &draft.values.rotation_period_hr, 0.0, 0.0, "%.4f")) {
                            draft.recompute_hydrostatic();
                            draft.dirty = true;
                        }
                    }
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
            ImGui::BeginChild("tab_orbit_content", ImVec2(0.0f, -footer_h), false);

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

                if (axial_tilt_deg.has_value()) {
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

        ImGui::EndTabBar();
    }

    if (physical_editing) {
        ImGui::BeginChild("inspector_edit_footer", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_NoScrollbar);
        ImGui::Separator();
        if (roche_violates) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Periapsis inside Roche limit. Apply disabled.");
            if (ImGui::Button("Cancel##edit_cancel_disabled", ImVec2(-1.0f, 0.0f))) {
                ui_state_.edit_mode = false;
                draft.reload(system(), insp_idx);
            }
        } else {
            float half_w = std::max(60.0f, (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f);
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
            ImGui::SameLine(0.0f, 10.0f);
            if (ImGui::Button("Cancel##edit_cancel", ImVec2(half_w, 0.0f))) {
                ui_state_.edit_mode = false;
                draft.reload(system(), insp_idx);
            }
        }
        bool can_del = (insp_idx != system().definition().star_idx && !b.definition.is_star && system().definition().motion_model != ephemeris::MotionModel::Spice);
        if (can_del) {
            ImGui::Spacing();
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
        if (!draft.error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", draft.error.c_str());
        }
        ImGui::EndChild();
    }

    ImGui::End();
}

void App::render_time_hud(rendering::RenderSettings& render_draft) {
    if (!ui_settings_.show_time_hud) return;

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
    ImGui::PushItemWidth(std::max(110.0f, std::min(220.0f * layout.scale, bar_w * 0.26f)));
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

    std::string date_str = time::format_sim_date(system().definition().epoch_utc, system().time());
    ImGui::TextColored(ImVec4(0.85f, 0.90f, 1.0f, 1.0f), "%s", date_str.c_str());

    ImGui::SameLine(0.0f, 10.0f * layout.scale);
    ImGui::TextDisabled("| Sim Time: %.5f yr", system().time().value);

    ImGui::SameLine(0.0f, 10.0f * layout.scale);
    const char* hud_mode = "Keplerian";
    if (system().definition().motion_model == ephemeris::MotionModel::Spice) hud_mode = "SPICE";
    ImGui::TextDisabled("| %s", hud_mode);

    ImGui::SameLine(0.0f, 6.0f * layout.scale);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.35f, 0.50f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.45f, 0.65f, 1.0f));
    if (ImGui::SmallButton("Export to N-Body")) {
        open_export_nbody_modal();
    }
    ImGui::PopStyleColor(2);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Export current positions & velocities as a new N-Body system");
    }

    ImGui::SameLine(0.0f, 10.0f * layout.scale);
    ImGui::TextDisabled("| %s", camera_.movement_mode == 0 ? "Free Flight" : "Simple Orbit");

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
        if (ui_state_.edit_mode) {
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
                const char* title;
            };
            static constexpr GroupSpec kGroups[] = {
                {"Core", "Core Ephemeris (Essential)"},
                {"Planetary", "Planetary Moons (Essential)"},
                {"Inner Moons", "Inner Planet Moons (Optional)"},
                {"Jupiter Moons", "Jupiter Moons (Optional)"},
                {"Saturn Moons", "Saturn Moons (Optional)"},
                {"Uranus Moons", "Uranus Moons (Optional)"},
                {"Neptune Moons", "Neptune Moons (Optional)"},
            };

            for (const auto& grp : kGroups) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.78f, 0.95f, 1.0f));
                ImGui::TextUnformatted(grp.title);
                ImGui::PopStyleColor();
                ImGui::Separator();

                for (const auto& k : defaults) {
                    if (k.group != grp.key) continue;
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

