#include "app.hpp"
#include "app_format.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"
#include "core/time_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <exception>
#include <format>
#include <stdexcept>

namespace sf {

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
    const bool inactive_spice = (system().definition().motion_model == ephemeris::MotionModel::Spice && !b.render.spice_valid);
    if (inactive_spice) {
        label += " [Inactive]";
    }

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

    if (inactive_spice) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.58f, 0.65f, 0.85f));
    }
    bool opened = ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<intptr_t>(body_idx)), flags, "%s", label.c_str());
    if (inactive_spice) {
        ImGui::PopStyleColor();
    }

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
    render_export_hdri_modal();
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
        if (ImGui::MenuItem("Export Skybox HDRI (.exr)...", nullptr, false, !size_comparison_active_)) {
            open_export_hdri_modal();
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
        bool has_selection = ui_state_.inspected_body >= 0 && ui_state_.inspected_body < static_cast<int>(scene().size());
        if (ImGui::MenuItem("Center Selected Object (F)", nullptr, false, has_selection)) {
            bool is_primary = comparisons_.empty() || composer_.map().source(size_t(ui_state_.inspected_body)).slot == 0;
            camera_.center_body(ui_state_.inspected_body, scene(), is_primary && ui_state_.inspect_barycenter);
        }
        if (ImGui::MenuItem("Snap Look to Tracked")) {
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

} // namespace sf
