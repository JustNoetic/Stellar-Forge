#include "app.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <format>

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

std::string format_distance_au(double dist_au) {
    double dist_km = dist_au * AU_TO_KM;
    if (dist_au >= DEFAULT_LY_THRESHOLD_AU) {
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

void App::init(bool self_test, benchmark::Options benchmark) {
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

void App::run() {
    if (benchmark_.enabled) {
        run_benchmark();
        return;
    }
    double last_time = glfwGetTime();
    uint32_t frames_rendered = 0;

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        double now = glfwGetTime();
        frame_dt_ = units::Seconds{std::clamp(now - last_time, 1e-5, 0.25)};
        last_time = now;
        auto step = simulation_.advance(frame_dt_);
        if (!step) {
            simulation_error_ = step.error;
            simulation_.apply(ephemeris::SetPaused{true});
            sf::log::error("Simulation paused: {}", step.error);
        }

        camera_.update(frame_dt_, system());
        render_frame();

        ++frames_rendered;
        if (self_test_ && frames_rendered >= 3) {
            sf::log::info("Self-test completed {} frames cleanly.", frames_rendered);
            break;
        }
    }
}

void App::shutdown() {
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
    } else if (key == GLFW_KEY_UP || key == GLFW_KEY_RIGHT) {
        simulation_.apply(ephemeris::SetTimeWarp{std::min(simulation_.clock().warp * 2.0, 1e12)});
    } else if (key == GLFW_KEY_DOWN || key == GLFW_KEY_LEFT) {
        simulation_.apply(ephemeris::SetTimeWarp{std::max(simulation_.clock().warp * 0.5, 1.0)});
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
    if (benchmark_.enabled) return;
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
    if (sample) sample->cpu_record_ms = elapsed_ms(record_start);
    vk_.end_frame(cmd, swapchain_idx, sample ? &sample->host : nullptr);
    return true;
}

void App::render_outliner_node(int body_idx) {
    const auto& b = system().body(body_idx);
    bool has_children = false;
    for (size_t i = 0; i < system().size(); ++i) {
        if (system().body(i).definition.parent_idx == body_idx) {
            has_children = true;
            break;
        }
    }

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
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

    bool opened = ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<intptr_t>(body_idx)), flags, "%s", label.c_str());
    if (ImGui::IsItemClicked()) {
        ui_state_.inspected_body = body_idx;
    }
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        ui_state_.inspected_body = body_idx;
        camera_.focus_body(body_idx, system(), true);
    }

    if (has_children && opened) {
        for (int i = 0; i < static_cast<int>(system().size()); ++i) {
            if (system().body(i).definition.parent_idx == body_idx) {
                render_outliner_node(i);
            }
        }
        ImGui::TreePop();
    }
}

void App::render_ui() {
    auto render_draft = render_settings_.values();
    bool apply_body_draft = false;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Systems")) {
            for (const auto& sname : system_names_) {
                bool active = (system().name() == sname);
                if (ImGui::MenuItem(sname.c_str(), nullptr, active) && !active) {
                    switch_system(sname);
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
            bool can_spice = (system().name() == ephemeris::SystemManager::SOLAR_SYSTEM_NAME || current_mode == ephemeris::MotionModel::Spice);
            if (ImGui::MenuItem("NASA SPICE Ephemeris Playback", nullptr, current_mode == ephemeris::MotionModel::Spice, can_spice)) {
                auto result = simulation_.apply(ephemeris::SetMotionModel{ephemeris::MotionModel::Spice});
                if (!result) simulation_error_ = result.error;
                else simulation_error_.clear();
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
            ImGui::MenuItem("Gaia DR3 Starfield", nullptr, &render_draft.show_gaia_stars);
            ImGui::MenuItem("Orbit Lines", nullptr, &render_draft.show_orbits);
            ImGui::SliderFloat("Orbit Opacity", &render_draft.orbit_opacity, 0.0f, 1.0f);
            ImGui::MenuItem("HDR Rendering", nullptr, &render_draft.hdr_enabled);
            ImGui::MenuItem("Aligned Panels", nullptr, &ui_settings_.aligned_panels);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ui::WorkspaceLayout layout{};
    if (ui_settings_.show_outliner) {
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
            for (int i = 0; i < static_cast<int>(system().size()); ++i) {
                if (system().body(i).definition.parent_idx < 0) {
                    render_outliner_node(i);
                }
            }
        }
        ImGui::End();
    }

    if (ui_settings_.show_inspector && ui_state_.has_selection(system()) &&
        ui_state_.inspected_body >= 0 && ui_state_.inspected_body < static_cast<int>(system().size())) {
        ImGuiWindowFlags flags = ui_.prepare_side_panel(ui_settings_, ui_state_, ui::PanelSide::Right, layout);
        if (ImGui::Begin("Body Inspector", &ui_settings_.show_inspector, flags)) {
            const auto& b = system().body(ui_state_.inspected_body);
            std::string sub = b.definition.is_star
                ? std::format("{} | {}", b.definition.spectral_class.empty() ? "Star" : b.definition.spectral_class, b.definition.stage)
                : b.definition.type;
            ui::heading(b.definition.name, sub);

            if (ImGui::Button("Track Body")) {
                camera_.focus_body(ui_state_.inspected_body, system(), false);
            }
            ImGui::SameLine();
            if (ImGui::Button("Fly To Orbit")) {
                camera_.focus_body(ui_state_.inspected_body, system(), true);
            }

            ui::section("Physical Telemetry");
            constexpr double MSUN_TO_MEARTH = 332946.0487;
            constexpr double G_SI = 6.67430e-11;
            constexpr double MSUN_KG = 1.98847e30;
            double r_km = b.definition.radius_au() * AU_TO_KM;
            double r_m = r_km * 1000.0;
            double mass_kg = b.definition.mass_msun * MSUN_KG;
            double surf_g = (r_m > 0.0) ? ((G_SI * mass_kg) / (r_m * r_m) / 9.80665) : 0.0;
            double v_esc_kms = (r_m > 0.0) ? (std::sqrt(2.0 * G_SI * mass_kg / r_m) / 1000.0) : 0.0;

            if (b.definition.is_star) {
                ui::value_row("Mass", trim_trailing_zeros(std::format("{:.4f} solar masses", b.definition.mass_msun)));
                ui::value_row("Radius", trim_trailing_zeros(std::format("{:.4f} solar radii", b.definition.radius_rsun)));
                ui::value_row("Luminosity", std::format("{:.4g} solar luminosities", b.definition.lum_eq));
                ui::value_row("Temperature", std::format("{:.0f} K", b.definition.temp_k));
            } else {
                ui::value_row("Mass", std::format("{:.4g} Earth masses", b.definition.mass_msun * MSUN_TO_MEARTH));
                ui::value_row("Mean Radius", trim_trailing_zeros(std::format("{:.1f} km", r_km)));
            }
            ui::value_row("Surface Gravity", std::format("{:.3f} g", surf_g));
            ui::value_row("Escape Velocity", std::format("{:.2f} km/s", v_esc_kms));
            ui::value_row("Oblateness", std::format("{:.5f}", b.definition.oblateness));
            if (std::abs(b.definition.rotation_period_hr) > 0.0) {
                ui::value_row("Rotation Period", std::format("{:.2f} h", b.definition.rotation_period_hr));
            }

            double dist_eye_au = glm::length(b.render.position.value - camera_.eye_pos_render().value);
            double alt_au = std::max(0.0, dist_eye_au - b.definition.radius_au());
            ui::value_row("Camera Distance", format_distance_au(dist_eye_au));
            ui::value_row("Surface Altitude", format_distance_au(alt_au));

            if (b.definition.parent_idx >= 0) {
                const auto& p = system().body(b.definition.parent_idx);
                double rel_v_kms = glm::length(b.motion.velocity.value - p.motion.velocity.value) * AU_TO_KM / SECONDS_PER_YEAR;
                ui::section("Initial Orbital Elements", "Current velocity is shown below.");
                ui::value_row("Primary Body", p.definition.name);
                ui::value_row("Initial Semi-major Axis", format_distance_au(b.definition.semi_major_au));
                ui::value_row("Initial Eccentricity", std::format("{:.5f}", b.definition.eccentricity));
                ui::value_row("Initial Inclination", std::format("{:.3f} deg", b.definition.inc_deg));
                ui::value_row("Orbital Speed", std::format("{:.2f} km/s", rel_v_kms));
            }

            ui::section("Physical Definition", "Edits take effect together when applied.");
            auto& draft = ui_state_.body_draft;
            draft.synchronize(system(), ui_state_.inspected_body);
            ui::field("Mass (solar masses)##body_mass", [&](const char* id) {
                draft.dirty |= ImGui::InputDouble(id, &draft.values.mass.value, 0.0, 0.0, "%.12g");
            });
            ImGui::BeginDisabled(b.definition.is_black_hole);
            ui::field("Physical radius (km)##body_radius", [&](const char* id) {
                draft.dirty |= ImGui::InputDouble(id, &draft.values.radius.value, 0.0, 0.0, "%.12g");
            });
            ImGui::EndDisabled();
            ui::field("Oblateness##body_oblateness", [&](const char* id) {
                draft.dirty |= ImGui::InputDouble(id, &draft.values.oblateness, 0.0, 0.0, "%.8g");
            });
            ui::field("J2##body_j2", [&](const char* id) {
                draft.dirty |= ImGui::InputDouble(id, &draft.values.j2, 0.0, 0.0, "%.12g");
            });
            ui::field("J4##body_j4", [&](const char* id) {
                draft.dirty |= ImGui::InputDouble(id, &draft.values.j4, 0.0, 0.0, "%.12g");
            });
            ImGui::BeginDisabled(!draft.dirty);
            apply_body_draft = ImGui::Button("Apply Physical Changes");
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Revert")) draft.reload(system(), ui_state_.inspected_body);
            if (!draft.error.empty()) ImGui::TextWrapped("%s", draft.error.c_str());

            ui::section("Camera & Starfield");
            if (!ui_state_.settings_error.empty()) ImGui::TextWrapped("%s", ui_state_.settings_error.c_str());
            ui::value_row("Flight Speed", format_speed_au_s(camera_.flight_speed_au_s));
            ui::value_row("FOV", std::format("{:.2f} deg", camera_.fov_deg));
            ui::field("Exposure##exp", [&](const char* id) {
                float speed = std::max(0.005f, render_draft.exposure * 0.02f);
                const char* fmt = (render_draft.exposure >= 10000.0f || (render_draft.exposure > 0.0f && render_draft.exposure < 0.005f)) ? "%.4g" : "%.3f";
                if (ImGui::DragFloat(id, &render_draft.exposure, speed, 0.0f, 0.0f, fmt)) {
                    render_draft.exposure = std::max(0.0f, render_draft.exposure);
                }
            });
            ui::field("Point Diameter##sps", [&](const char* id) {
                ImGui::SliderFloat(id, &render_draft.star_point_size, 1.5f, 6.0f, "%.1f px");
            });
            ui::field("Star Brightness##sbi", [&](const char* id) {
                float star_speed = std::max(0.01f, render_draft.star_intensity * 0.02f);
                if (ImGui::DragFloat(id, &render_draft.star_intensity, star_speed, 0.0f, 0.0f, "%.2f")) {
                    render_draft.star_intensity = std::max(0.0f, render_draft.star_intensity);
                }
            });
            ui::field("VSync##vsync_field", [&](const char* id) {
                ImGui::Checkbox(id, &render_draft.vsync);
            });

            ui::section("Engine Telemetry", vk_.device_name());
            ui::value_row("Frame Rate", std::format("{:.1f} FPS", ImGui::GetIO().Framerate));
            const char* pm_str = "FIFO (VSync)";
            if (vk_.present_mode() == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                pm_str = "Immediate (Uncapped)";
            } else if (vk_.present_mode() == VK_PRESENT_MODE_MAILBOX_KHR) {
                pm_str = "Mailbox (Uncapped)";
            }
            ui::value_row("Present Mode", pm_str);
            ui::value_row("Gaia Stars Loaded", std::format("{}", star_catalog_.count()));
            ui::value_row("GPU Scene HDR", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_scene_hdr")));
            ui::value_row("GPU Composite", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_post_composite")));
            ui::value_row("GPU Dear ImGui", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_ui")));
        }
        ImGui::End();
    }

    if (ui_settings_.show_time_hud) {
        layout = ui_.compute_layout(ui_settings_, ui_state_, static_cast<float>(vk_.swapchain_extent().width),
                                    static_cast<float>(vk_.swapchain_extent().height));
        ImGui::SetNextWindowPos(ImVec2(layout.gap, layout.height - layout.transport_height - layout.gap), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(layout.width - layout.gap * 2.0f, layout.transport_height), ImGuiCond_Always);
        ImGuiWindowFlags hud_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
        if (ImGui::Begin("TimeTransportHUD", nullptr, hud_flags)) {
            if (ImGui::Button(simulation_.clock().paused ? "Resume (Space)" : "Pause (Space)")) {
                simulation_.apply(ephemeris::SetPaused{!simulation_.clock().paused});
            }
            ImGui::SameLine();
            if (ImGui::Button("1x")) simulation_.apply(ephemeris::SetTimeWarp{1.0});
            ImGui::SameLine();
            if (ImGui::Button("100x")) simulation_.apply(ephemeris::SetTimeWarp{100.0});
            ImGui::SameLine();
            if (ImGui::Button("10,000x")) simulation_.apply(ephemeris::SetTimeWarp{10000.0});
            ImGui::SameLine();
            if (ImGui::Button("1e6x")) simulation_.apply(ephemeris::SetTimeWarp{1.0e6});
            ImGui::SameLine();
            if (ImGui::Button("1 yr/s")) simulation_.apply(ephemeris::SetTimeWarp{SECONDS_PER_YEAR});
            ImGui::SameLine();
            const char* hud_mode = "Keplerian";
            if (system().definition().motion_model == ephemeris::MotionModel::Spice) hud_mode = "SPICE";
            else if (system().definition().motion_model == ephemeris::MotionModel::NBody) hud_mode = "N-Body";
            ImGui::Text("| Warp: %.4gx | Sim Time: %.5f yr | %s | %s | Speed: %s",
                        simulation_.clock().warp, system().time().value,
                        hud_mode,
                        camera_.movement_mode == 0 ? "Free Flight" : "Simple Orbit",
                        format_speed_au_s(camera_.flight_speed_au_s).c_str());
        }
        ImGui::End();
    }
    auto settings_result = render_settings_.apply(render_draft);
    ui_state_.settings_error = settings_result.error;
    if (apply_body_draft) {
        auto result = simulation_.apply(ui_state_.body_draft.command());
        if (result) ui_state_.body_draft.reload(system(), ui_state_.inspected_body);
        else ui_state_.body_draft.error = result.error;
    }
}

} // namespace sf
