#include "app.hpp"
#include "ui/size_comparison.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <format>

namespace sf {

void App::set_size_comparison(bool active) {
    size_comparison_requested_ = active;
}

void App::apply_size_comparison() {
    bool active = size_comparison_requested_;
    if (active == size_comparison_active_) return;
    if (active && !size_renderer_.initialized()) size_renderer_.init(vk_);
    size_comparison_active_ = active;
    camera_.left_dragging = camera_.right_dragging = false;
    camera_.key_w = camera_.key_a = camera_.key_s = camera_.key_d = camera_.key_q = camera_.key_e = false;
    ui_state_.comparison_selection = false;
    ui_state_.comparison_mode = active;
    if (active) {
        if (!size_comparison_initialized_) {
            if (size_comparison_.selected_systems.empty()) size_comparison_.selected_systems = {system().name()};
            size_comparison_initialized_ = true;
        }
        size_view_.fit = true;
    }
}

void App::update_size_comparison() {
    if (!size_comparison_active_) return;
    ui_state_.comparison_mode = true;
    if (size_comparison_.synchronize(sys_mgr_,system())) size_view_.fit = true;
    ui_state_.comparison_selection = size_comparison_.selection() >= 0;
    const auto& io = ImGui::GetIO();
    glm::dvec2 viewport{std::max(io.DisplaySize.x,1.0f),std::max(io.DisplaySize.y,1.0f)};
    auto bounds = ui_.comparison_bounds(ui_settings_,ui_state_);
    if (size_view_.fit) size_view_.frame(size_comparison_.bounds,bounds,viewport);
    if (comparison_focus_ >= 0 && size_t(comparison_focus_) < size_comparison_.bodies.size()) {
        size_view_.focus(size_comparison_.bodies[comparison_focus_],bounds,viewport);
        comparison_focus_ = -1;
    }
    if (!io.WantCaptureMouse && !ui_.is_dragging()) {
        if (io.MouseWheel != 0) size_view_.zoom(io.MouseWheel,{io.MousePos.x,io.MousePos.y},viewport);
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left,3) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle,3) ||
            ImGui::IsMouseDragging(ImGuiMouseButton_Right,3)) size_view_.pan({io.MouseDelta.x,io.MouseDelta.y},viewport.y);
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            int index = size_comparison_.pick(size_view_.screen_to_world({io.MousePos.x,io.MousePos.y},viewport),size_view_.height/viewport.y*7);
            size_comparison_.select(index);
            ui_state_.comparison_selection = index >= 0;
            if (index >= 0 && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                size_view_.focus(size_comparison_.bodies[index],ui_.comparison_bounds(ui_settings_,ui_state_),viewport);
        }
    }
}

void App::render_comparison_outliner() {
    if (!ui_settings_.show_outliner) return;
    if (glfwGetTime() >= comparison_refresh_at_) {
        system_names_ = sys_mgr_.list_systems();
        comparison_refresh_at_ = glfwGetTime()+1;
    }
    auto names = system_names_;
    if (std::find(names.begin(),names.end(),system().name()) == names.end()) names.push_back(system().name());
    for (const auto& name : size_comparison_.selected_systems)
        if (std::find(names.begin(),names.end(),name) == names.end()) names.push_back(name);
    ui::WorkspaceLayout layout;
    auto flags = ui_.prepare_side_panel(ui_settings_,ui_state_,ui::PanelSide::Left,layout);
    if (ImGui::Begin("System Outliner",&ui_settings_.show_outliner,flags)) {
        ui::heading("Systems to compare","Check systems to show their bodies at the same scale.");
        for (const auto& name : names) {
            bool checked = std::find(size_comparison_.selected_systems.begin(),size_comparison_.selected_systems.end(),name) !=
                size_comparison_.selected_systems.end();
            if (ImGui::Checkbox(name.c_str(),&checked)) {
                if (checked) size_comparison_.selected_systems.push_back(name);
                else std::erase(size_comparison_.selected_systems,name);
                size_comparison_.select(-1);
                ui_state_.comparison_selection = false;
                comparison_focus_ = -1;
                size_view_.fit = true;
            }
            if (auto error = size_comparison_.errors.find(name); error != size_comparison_.errors.end())
                ImGui::TextWrapped("Could not load: %s",error->second.c_str());
        }
        if (size_comparison_.selected_systems.empty()) ImGui::TextDisabled("Check a system to begin.");
        ImGui::Separator();
        if (ImGui::Button("Fit All",ImVec2(-1,0))) size_view_.fit = true;
        if (ImGui::Button("Refresh Systems",ImVec2(-1,0))) { comparison_refresh_at_ = 0; size_comparison_.refresh(); }
        int selected = size_comparison_.selection();
        if (selected >= 0) {
            const auto& body = size_comparison_.bodies[selected];
            ui::section("Selected object");
            ImGui::TextWrapped("%s",body.definition().name.c_str());
            ImGui::TextDisabled("%s",body.system->name.c_str());
            if (ImGui::Button("Focus Selected",ImVec2(-1,0))) comparison_focus_ = selected;
        }
    }
    ImGui::End();
}

void App::render_comparison_inspector() {
    int index = size_comparison_.selection();
    if (!ui_settings_.show_inspector || index < 0) return;
    const auto& body = size_comparison_.bodies[index];
    const auto& def = body.definition();
    ui::WorkspaceLayout layout;
    auto flags = ui_.prepare_side_panel(ui_settings_,ui_state_,ui::PanelSide::Right,layout);
    if (ImGui::Begin((def.name+"###size_inspector").c_str(),&ui_settings_.show_inspector,flags)) {
        ui::heading(def.name,body.system->name);
        ui::value_row("Type",def.type);
        ui::value_row("Equatorial radius",std::format("{:.3f} km",body.radius/comparison::SIZE_SCALE));
        ui::value_row("Diameter",std::format("{:.3f} km",body.radius/comparison::SIZE_SCALE*2));
        ui::value_row("Polar radius",std::format("{:.3f} km",body.radius/comparison::SIZE_SCALE*(1-def.oblateness)));
        ui::value_row("Earth radii",std::format("{:.4f}",body.radius/comparison::SIZE_SCALE/6378.137));
        ui::value_row("Mass",std::format("{:.6g} solar masses",def.mass_msun));
        if (def.is_star) ui::value_row("Temperature",std::format("{:.0f} K",def.temp_k));
        if (!def.rings.empty()) ui::value_row("Ring outer radius",std::format("{:.3f} km",body.outer/comparison::SIZE_SCALE));
        if (def.is_black_hole) ImGui::TextWrapped("Radius follows the simulation's black-hole display convention.");
        ImGui::Separator();
        if (ImGui::Button("Focus Selected",ImVec2(-1,0))) comparison_focus_ = index;
        if (ImGui::Button("Fit All",ImVec2(-1,0))) size_view_.fit = true;
        ImGui::TextWrapped("Drag to pan. Scroll to zoom at the cursor. Double-click a body to focus.");
    }
    ImGui::End();
}

void App::render_comparison_labels() {
    if (!ui_settings_.ui_visible) return;
    const auto& io = ImGui::GetIO();
    ui::size_comparison_labels(size_comparison_,size_view_,ui_.comparison_bounds(ui_settings_,ui_state_),
        {io.DisplaySize.x,io.DisplaySize.y},ui_settings_.scale);
    if (ui::size_comparison_hud(ui_.compute_layout(ui_settings_,ui_state_,0,0),size_comparison_.selected_systems.size(),size_comparison_.bodies.size()))
        size_view_.fit = true;
}

}
