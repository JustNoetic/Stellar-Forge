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
            render_body_rings_tab(insp_idx);
        }

        if (!b.definition.is_star && !b.definition.is_black_hole && !b.definition.is_spacecraft) {
            render_body_atmosphere_tab(insp_idx);
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

void App::render_body_atmosphere_tab(int insp_idx) {
    if (!ImGui::BeginTabItem("Atmosphere")) return;
    const auto& b = system().body(insp_idx);
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

} // namespace sf
