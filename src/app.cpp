#include "app.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <format>

namespace sf {

namespace {

std::string format_distance_au(double dist_au) {
    double dist_km = dist_au * AU_TO_KM;
    if (dist_au >= DEFAULT_LY_THRESHOLD_AU) {
        return std::format("{:.3f} ly", dist_au * AU_TO_LY);
    }
    if (dist_au >= 0.05) {
        return std::format("{:.4f} AU ({:.0f} km)", dist_au, dist_km);
    }
    if (dist_km >= 1.0) {
        return std::format("{:.2f} km", dist_km);
    }
    return std::format("{:.1f} m", dist_km * 1000.0);
}

std::string format_speed_au_s(double speed_au_s) {
    double km_s = speed_au_s * AU_TO_KM;
    if (speed_au_s >= LY_TO_AU * 0.01) {
        return std::format("{:.2f} ly/s", speed_au_s * AU_TO_LY);
    }
    if (speed_au_s >= 0.01) {
        return std::format("{:.3f} AU/s ({:.2f} c)", speed_au_s, speed_au_s * SECONDS_PER_YEAR / C_AU_YR);
    }
    if (km_s >= 1.0) {
        return std::format("{:.2f} km/s", km_s);
    }
    return std::format("{:.1f} m/s", km_s * 1000.0);
}

} // namespace

void App::init(bool self_test) {
    self_test_ = self_test;

    if (!glfwInit()) {
        sf::log::fatal("Failed to initialize GLFW.");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (self_test_) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }

    window_ = glfwCreateWindow(1600, 900, "Stellar Forge", nullptr, nullptr);
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

    recreate_render_targets();

    system_names_ = sys_mgr_.list_systems();
    switch_system(ephemeris::SystemManager::SOLAR_SYSTEM_NAME);

    star_catalog_.init(vk_, sf::paths::external_path("data", "gaia", "stars.bin"));
    celestial_renderer_.init(vk_);

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
    system_ = sys_mgr_.load_system(system_name);
    sim_time_yr_ = 0.0;
    camera_.tracking_idx = system_.star_idx;
    camera_.inspected_idx = system_.find_body("Earth");
    if (camera_.inspected_idx < 0) {
        camera_.inspected_idx = system_.star_idx;
    }
    camera_.cam_pos_rel = {0.0, 19.02, 40.78};
    camera_.up = {0.0, 1.0, 0.0};
    camera_.cam_look_aim = true;
    camera_.flight_speed_au_s = 0.2;
    camera_.has_prev_spin = false;
    ui_settings_.has_inspected_body = !system_.bodies.empty();
}

void App::run() {
    double last_time = glfwGetTime();
    uint32_t frames_rendered = 0;

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        double now = glfwGetTime();
        frame_dt_ = std::clamp(now - last_time, 1e-5, 0.25);
        last_time = now;
        elapsed_sec_ = now;

        if (!paused_) {
            double dt_yr = (frame_dt_ * sim_warp_) / SECONDS_PER_YEAR;
            sim_time_yr_ += dt_yr;
            ephemeris::SystemManager::step_keplerian(system_, dt_yr, sim_time_yr_);
        }

        camera_.update(frame_dt_, system_);
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
        paused_ = !paused_;
    } else if (key == GLFW_KEY_F && action == GLFW_PRESS) {
        camera_.snap_to_tracked(system_);
    } else if (key == GLFW_KEY_R) {
        sim_warp_ = 1.0;
    } else if (key == GLFW_KEY_UP || key == GLFW_KEY_RIGHT) {
        sim_warp_ = std::min(sim_warp_ * 2.0, 1e12);
    } else if (key == GLFW_KEY_DOWN || key == GLFW_KEY_LEFT) {
        sim_warp_ = std::max(sim_warp_ * 0.5, 1.0);
    } else if (key == GLFW_KEY_EQUAL) {
        float step = (mods & GLFW_MOD_SHIFT) ? 2.0f : 1.15f;
        camera_.exposure = std::min(camera_.exposure * step, 1024.0f);
    } else if (key == GLFW_KEY_MINUS) {
        float step = (mods & GLFW_MOD_SHIFT) ? 2.0f : 1.15f;
        camera_.exposure = std::max(camera_.exposure / step, 1e-4f);
    } else if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        ui_settings_.has_inspected_body = false;
    }
}

void App::on_mouse_button(int button, int action, int) {
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
    camera_.on_cursor_pos(xpos, ypos, system_);
}

void App::on_scroll(double, double yoffset) {
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

void App::render_frame() {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    uint32_t swapchain_idx = 0;
    if (!vk_.begin_frame(cmd, swapchain_idx)) {
        return;
    }

    recreate_render_targets();

    ui_.begin_frame(ui_settings_);
    if (ui_settings_.ui_visible) {
        render_ui();
    }

    VkExtent2D ext = vk_.swapchain_extent();

    uint32_t t_scene = vk_.begin_gpu_timer(cmd, "gpu_scene_hdr");
    vk_.transition_image(cmd, hdr_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    vk_.transition_image(cmd, scene_depth_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    auto color_att = rhi::RenderingAttachment::clear_color(hdr_color_, 0.0f, 0.0f, 0.0f, 1.0f);
    auto depth_att = rhi::RenderingAttachment::clear_depth_reverse_z(scene_depth_, 0.0f);
    vk_.begin_rendering(cmd, ext, std::span{&color_att, 1}, &depth_att);

    celestial_renderer_.render(cmd, vk_, camera_, system_, ext);
    star_catalog_.render(cmd, vk_, camera_, sim_time_yr_, ext);

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

    vk_.end_frame(cmd, swapchain_idx);
}

void App::render_outliner_node(int body_idx) {
    const auto& b = system_.bodies[body_idx];
    bool has_children = false;
    for (size_t i = 0; i < system_.bodies.size(); ++i) {
        if (system_.bodies[i].parent_idx == body_idx) {
            has_children = true;
            break;
        }
    }

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (!has_children) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }
    if (b.is_star) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
    if (camera_.inspected_idx == body_idx && ui_settings_.has_inspected_body) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    std::string label = (camera_.tracking_idx == body_idx)
        ? std::format("{} [Tracked]", b.name)
        : b.name;

    bool opened = ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<intptr_t>(body_idx)), flags, "%s", label.c_str());
    if (ImGui::IsItemClicked()) {
        camera_.inspected_idx = body_idx;
        ui_settings_.has_inspected_body = true;
    }
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        camera_.focus_body(body_idx, system_, true);
        ui_settings_.has_inspected_body = true;
    }

    if (has_children && opened) {
        for (int i = 0; i < static_cast<int>(system_.bodies.size()); ++i) {
            if (system_.bodies[i].parent_idx == body_idx) {
                render_outliner_node(i);
            }
        }
        ImGui::TreePop();
    }
}

void App::render_ui() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Systems")) {
            for (const auto& sname : system_names_) {
                bool active = (system_.name == sname);
                if (ImGui::MenuItem(sname.c_str(), nullptr, active) && !active) {
                    switch_system(sname);
                }
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
                camera_.snap_to_tracked(system_);
            }
            ImGui::MenuItem("Auto Horizon Level (< 200 km)", nullptr, &camera_.horizon_align);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("System Outliner", nullptr, &ui_settings_.show_outliner);
            ImGui::MenuItem("Body Inspector", nullptr, &ui_settings_.show_inspector);
            ImGui::MenuItem("Time Transport HUD", nullptr, &ui_settings_.show_time_hud);
            ImGui::Separator();
            ImGui::MenuItem("Gaia DR3 Starfield", nullptr, &camera_.show_gaia_stars);
            ImGui::MenuItem("HDR Rendering", nullptr, &camera_.hdr_enabled);
            ImGui::MenuItem("Aligned Panels", nullptr, &ui_settings_.aligned_panels);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ui::WorkspaceLayout layout{};
    if (ui_settings_.show_outliner) {
        ImGuiWindowFlags flags = ui_.prepare_side_panel(ui_settings_, ui::PanelSide::Left, layout);
        if (ImGui::Begin("System Outliner", &ui_settings_.show_outliner, flags)) {
            ui::heading(system_.name, "Click to inspect, double-click to fly to body.");
            for (int i = 0; i < static_cast<int>(system_.bodies.size()); ++i) {
                if (system_.bodies[i].parent_idx < 0) {
                    render_outliner_node(i);
                }
            }
        }
        ImGui::End();
    }

    if (ui_settings_.show_inspector && ui_settings_.has_inspected_body &&
        camera_.inspected_idx >= 0 && camera_.inspected_idx < static_cast<int>(system_.bodies.size())) {
        ImGuiWindowFlags flags = ui_.prepare_side_panel(ui_settings_, ui::PanelSide::Right, layout);
        if (ImGui::Begin("Body Inspector", &ui_settings_.show_inspector, flags)) {
            const auto& b = system_.bodies[camera_.inspected_idx];
            std::string sub = b.is_star
                ? std::format("{} | {}", b.spectral_class.empty() ? "Star" : b.spectral_class, b.stage)
                : b.type;
            ui::heading(b.name, sub);

            if (ImGui::Button("Track Body")) {
                camera_.focus_body(camera_.inspected_idx, system_, false);
            }
            ImGui::SameLine();
            if (ImGui::Button("Fly To Orbit")) {
                camera_.focus_body(camera_.inspected_idx, system_, true);
            }

            ui::section("Physical Telemetry");
            constexpr double MSUN_TO_MEARTH = 332946.0487;
            constexpr double G_SI = 6.67430e-11;
            constexpr double MSUN_KG = 1.98847e30;
            double r_km = b.radius_au * AU_TO_KM;
            double r_m = r_km * 1000.0;
            double mass_kg = b.mass_msun * MSUN_KG;
            double surf_g = (r_m > 0.0) ? ((G_SI * mass_kg) / (r_m * r_m) / 9.80665) : 0.0;
            double v_esc_kms = (r_m > 0.0) ? (std::sqrt(2.0 * G_SI * mass_kg / r_m) / 1000.0) : 0.0;

            if (b.is_star) {
                ui::value_row("Mass", std::format("{:.4f} solar masses", b.mass_msun));
                ui::value_row("Radius", std::format("{:.4f} solar radii ({:.0f} km)", b.radius_rsun, r_km));
                ui::value_row("Luminosity", std::format("{:.4g} solar luminosities", b.lum_eq));
                ui::value_row("Temperature", std::format("{:.0f} K", b.temp_k));
            } else {
                ui::value_row("Mass", std::format("{:.4g} Earth masses", b.mass_msun * MSUN_TO_MEARTH));
                ui::value_row("Mean Radius", std::format("{:.1f} km", r_km));
            }
            ui::value_row("Surface Gravity", std::format("{:.3f} g", surf_g));
            ui::value_row("Escape Velocity", std::format("{:.2f} km/s", v_esc_kms));
            ui::value_row("Oblateness (f)", std::format("{:.5f}", b.oblateness));
            if (std::abs(b.rotation_period_hr) > 0.0) {
                ui::value_row("Rotation Period", std::format("{:.2f} h", b.rotation_period_hr));
            }

            double dist_eye_au = glm::length(b.pos_render - camera_.eye_pos_render());
            double alt_au = std::max(0.0, dist_eye_au - b.radius_au);
            ui::value_row("Camera Distance", format_distance_au(dist_eye_au));
            ui::value_row("Surface Altitude", format_distance_au(alt_au));

            if (b.parent_idx >= 0) {
                const auto& p = system_.bodies[b.parent_idx];
                double rel_v_kms = glm::length(b.vel_ecl - p.vel_ecl) * AU_TO_KM / SECONDS_PER_YEAR;
                ui::section("Orbital Elements");
                ui::value_row("Primary Body", p.name);
                ui::value_row("Semi-major Axis (a)", format_distance_au(b.semi_major_au));
                ui::value_row("Eccentricity (e)", std::format("{:.5f}", b.eccentricity));
                ui::value_row("Inclination (i)", std::format("{:.3f} deg", b.inc_deg));
                ui::value_row("Orbital Speed", std::format("{:.2f} km/s", rel_v_kms));
            }

            ui::section("Camera & Starfield");
            ui::value_row("Flight Speed", format_speed_au_s(camera_.flight_speed_au_s));
            ui::value_row("FOV", std::format("{:.2f} deg", camera_.fov_deg));
            ui::field("Exposure##exp", [&](const char* id) {
                ImGui::SliderFloat(id, &camera_.exposure, 0.05f, 32.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            });
            ui::field("Star Point Size (px)##sps", [&](const char* id) {
                ImGui::SliderFloat(id, &camera_.star_point_size, 1.5f, 6.0f, "%.1f");
            });
            ui::field("Star Brightness##sbi", [&](const char* id) {
                ImGui::SliderFloat(id, &camera_.star_intensity, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            });

            ui::section("Engine Telemetry", vk_.device_name());
            ui::value_row("Frame Rate", std::format("{:.1f} FPS ({:.2f} ms)", ImGui::GetIO().Framerate, frame_dt_ * 1000.0));
            ui::value_row("Gaia Stars Loaded", std::format("{}", star_catalog_.count()));
            ui::value_row("GPU Scene HDR", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_scene_hdr")));
            ui::value_row("GPU Composite", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_post_composite")));
            ui::value_row("GPU Dear ImGui", std::format("{:.3f} ms", vk_.gpu_time_ms("gpu_ui")));
        }
        ImGui::End();
    }

    if (ui_settings_.show_time_hud) {
        layout = ui_.compute_layout(ui_settings_, static_cast<float>(vk_.swapchain_extent().width),
                                    static_cast<float>(vk_.swapchain_extent().height));
        ImGui::SetNextWindowPos(ImVec2(layout.gap, layout.height - layout.transport_height - layout.gap), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(layout.width - layout.gap * 2.0f, layout.transport_height), ImGuiCond_Always);
        ImGuiWindowFlags hud_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
        if (ImGui::Begin("TimeTransportHUD", nullptr, hud_flags)) {
            if (ImGui::Button(paused_ ? "Resume (Space)" : "Pause (Space)")) {
                paused_ = !paused_;
            }
            ImGui::SameLine();
            if (ImGui::Button("1x")) sim_warp_ = 1.0;
            ImGui::SameLine();
            if (ImGui::Button("100x")) sim_warp_ = 100.0;
            ImGui::SameLine();
            if (ImGui::Button("10,000x")) sim_warp_ = 10000.0;
            ImGui::SameLine();
            if (ImGui::Button("1e6x")) sim_warp_ = 1.0e6;
            ImGui::SameLine();
            if (ImGui::Button("1 yr/s")) sim_warp_ = SECONDS_PER_YEAR;
            ImGui::SameLine();
            ImGui::Text("| Warp: %.4gx | Sim Time: %.5f yr | Mode: %s | Speed: %s",
                        sim_warp_, sim_time_yr_,
                        camera_.movement_mode == 0 ? "Free Flight" : "Simple Orbit",
                        format_speed_au_s(camera_.flight_speed_au_s).c_str());
        }
        ImGui::End();
    }
}

} // namespace sf
