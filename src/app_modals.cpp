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

void App::open_export_hdri_modal() {
    if (hdri_exporter_.active()) return;
    auto dt = time::calculate_sim_date(system().definition().epoch_utc, system().time());
    std::string slug;
    for (char c : system().name()) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
            slug += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (c == ' ') {
            slug += '_';
        }
    }
    if (slug.empty()) slug = "scene";
    auto default_path = paths::external_path("exports", "hdri",
        std::format("{}_{:04d}-{:02d}-{:02d}_8k.exr", slug, dt.year, dt.month, dt.day));
    ui_state_.export_hdri.reset(default_path.string());
}

void App::render_export_hdri_modal() {
    if (!ui_state_.export_hdri.active) return;

    ImGui::SetNextWindowSize(ImVec2(540.0f, 430.0f), ImGuiCond_Always);
    ImGui::OpenPopup("Export Skybox HDRI (.exr)###export_hdri_modal");
    bool open = ui_state_.export_hdri.active;
    if (ImGui::BeginPopupModal("Export Skybox HDRI (.exr)###export_hdri_modal", &open, ImGuiWindowFlags_NoResize)) {
        auto& ed = ui_state_.export_hdri;
        const bool running = hdri_exporter_.active() && !hdri_exporter_.finished();
        const bool finished = hdri_exporter_.finished();

        if (!open && !running) {
            if (finished) hdri_exporter_.acknowledge();
            ed.active = false;
            ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
            return;
        }

        ui::heading("Export 32-bit Lighting HDRI (.exr)",
                    "Equirectangular 360 skybox without tone mapping, bloom, or orbit guides.");

        std::string date_str = time::format_sim_date(system().definition().epoch_utc, system().time());
        ImGui::Text("System: %s", system().name().c_str());
        ImGui::Text("Epoch: %s", date_str.c_str());
        ImGui::TextDisabled("Format: 32-bit float RGB OpenEXR (uncompressed scanlines, Blender World Z-up)");
        ImGui::Spacing();

        if (running) {
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::Text("%s", hdri_exporter_.status_text().c_str());
            ImGui::ProgressBar(hdri_exporter_.progress(), ImVec2(-1.0f, 24.0f));
            ImGui::Spacing();
            ImGui::TextDisabled("Freezing simulation and capturing 6 orthogonal HDR faces...");
            ImGui::Spacing();
            if (ImGui::Button("Cancel Export", ImVec2(-1.0f, 32.0f))) {
                pending_hdri_export_cancel_ = true;
                ed.active = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
            return;
        }

        if (finished) {
            ImGui::Separator();
            ImGui::Spacing();
            if (hdri_exporter_.succeeded()) {
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.55f, 1.0f), "Export completed successfully.");
                ImGui::TextWrapped("Saved to: %s", hdri_exporter_.output_path().string().c_str());
                ImGui::Spacing();
                ImGui::TextDisabled("In Blender: World Properties -> Color -> Environment Texture -> open .exr (Linear).");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Export failed:");
                ImGui::TextWrapped("%s", hdri_exporter_.error().c_str());
            }
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Done", ImVec2(-1.0f, 32.0f))) {
                hdri_exporter_.acknowledge();
                ed.active = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
            return;
        }

        const uint32_t max_dim = vk_.device_properties().limits.maxImageDimension2D;
        const char* quality_labels[int(rendering::HdriQuality::Count)]{};
        for (int i = 0; i < int(rendering::HdriQuality::Count); ++i) {
            quality_labels[i] = rendering::HdriExporter::preset_info(static_cast<rendering::HdriQuality>(i), max_dim).label;
        }
        ui::field("Quality / Resolution", [&](const char* id) {
            ImGui::Combo(id, &ed.quality_idx, quality_labels, int(rendering::HdriQuality::Count));
        });

        const auto pinfo = rendering::HdriExporter::preset_info(static_cast<rendering::HdriQuality>(ed.quality_idx), max_dim);
        const double approx_mb = (double(pinfo.width) * double(pinfo.height) * 12.0) / (1024.0 * 1024.0);
        if (approx_mb >= 1024.0) {
            ImGui::TextDisabled("  %u x %u equirectangular (%u px faces) | ~%.2f GB .exr",
                                pinfo.width, pinfo.height, pinfo.face_size, approx_mb / 1024.0);
        } else {
            ImGui::TextDisabled("  %u x %u equirectangular (%u px faces) | ~%.0f MB .exr",
                                pinfo.width, pinfo.height, pinfo.face_size, approx_mb);
        }

        const char* orient_labels[] = {
            "Camera Horizon (Blender Z-up, current view centered)",
            "Ecliptic North (Blender Z-up)"
        };
        ui::field("Orientation", [&](const char* id) {
            ImGui::Combo(id, &ed.orientation_idx, orient_labels, 2);
        });

        std::string norm_label = std::format("Current Exposure Normalization ({:.3g}x)", render_settings_.values().exposure);
        const char* exp_labels[] = {norm_label.c_str(), "Raw Physical Radiance (1.0x)"};
        ui::field("Exposure Scaling", [&](const char* id) {
            ImGui::Combo(id, &ed.exposure_idx, exp_labels, 2);
        });

        ui::field("Output File (.exr)", [&](const char* id) {
            float browse_w = 90.0f;
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - browse_w - 6.0f);
            char buf[1024];
            std::strncpy(buf, ed.output_path.c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputText(id, buf, sizeof(buf))) {
                ed.output_path = buf;
            }
            ImGui::SameLine(0.0f, 6.0f);
            if (ImGui::Button("Browse...", ImVec2(browse_w, 0.0f))) {
                if (auto picked = rendering::HdriExporter::save_file_dialog(ed.output_path)) {
                    ed.output_path = picked->string();
                }
            }
        });

        if (!ed.error.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", ed.error.c_str());
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float half_w = (ImGui::GetContentRegionAvail().x - 10.0f) * 0.5f;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.38f, 0.43f, 1.0f));
        if (ImGui::Button("Export (.exr)", ImVec2(half_w, 32.0f))) {
            if (ed.output_path.empty()) {
                ed.error = "Output path cannot be empty.";
            } else {
                ed.error.clear();
                pending_hdri_export_start_ = true;
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

} // namespace sf
