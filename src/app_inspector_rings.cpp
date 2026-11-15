#include "app.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"
#include "core/time_utils.hpp"
#include "rendering/ring_optics.hpp"
#include "rendering/ring_generator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <exception>
#include <format>
#include <random>
#include <stdexcept>

namespace sf {

void App::render_body_rings_tab(int insp_idx) {
    const auto& b = system().body(insp_idx);
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
                        re.procedural.opacity_variety = p.opacity_variety;
                        re.procedural.single_band_color_variety = p.single_band_color_variety;
                        re.procedural.per_band_color_variety = p.per_band_color_variety;
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
            static std::mt19937 ui_rng{std::random_device{}()};
            if (ImGui::Button("Random##proc", ImVec2(75.0f, 0.0f))) {
                re.procedural.seed = static_cast<int>(ui_rng() % 1000000u);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Randomize only the generation seed.");
            }
            if (ImGui::Button("Randomize Parameters##proc", ImVec2(-1.0f, 0.0f))) {
                re.procedural.seed = static_cast<int>(ui_rng() % 1000000u);
                auto rcfg = rendering::ProceduralRingConfig::randomize_from_preset(
                    presets[cur_p_idx], static_cast<uint32_t>(re.procedural.seed));
                re.procedural.band_count = rcfg.band_count;
                re.procedural.gap_count = rcfg.gap_count;
                re.procedural.striation_freq = rcfg.striation_freq;
                re.procedural.contrast = rcfg.contrast;
                re.procedural.base_density = rcfg.base_density;
                re.procedural.core_boost = rcfg.core_boost;
                re.procedural.inner_mult = rcfg.inner_radius_ratio;
                re.procedural.outer_mult = rcfg.outer_radius_ratio;
                re.procedural.opacity = rcfg.opacity;
                re.procedural.opacity_variety = rcfg.opacity_variety;
                re.procedural.single_band_color_variety = rcfg.single_band_color_variety;
                re.procedural.per_band_color_variety = rcfg.per_band_color_variety;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Randomize seed and ring parameter sliders within the selected preset's material limits.");
            }

            ImGui::SliderInt("Bands##proc", &re.procedural.band_count, 1, 8);
            ImGui::SliderInt("Gaps##proc", &re.procedural.gap_count, 0, 6);
            ImGui::SliderFloat("Striations##proc", &re.procedural.striation_freq, 20.0f, 300.0f, "%.0f");
            ImGui::SliderFloat("Sharpness##proc", &re.procedural.contrast, 0.5f, 3.5f, "%.2f");
            ImGui::SliderFloat("Density (Tau)##proc", &re.procedural.base_density, 0.05f, 8.0f, "%.2f");
            ImGui::SliderFloat("Core Opacity##proc", &re.procedural.core_boost, 0.5f, 5.0f, "%.2f");
            ImGui::SliderFloat("Opacity Variety##proc", &re.procedural.opacity_variety, 0.0f, 2.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Controls intra-band opacity shelf, ripple, and striation modulation depth (0 = uniform, 1 = balanced, 2 = extreme).");
            }
            ImGui::SliderFloat("Single-Band Color Variety##proc", &re.procedural.single_band_color_variety, 0.0f, 2.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Controls compositional color variation within each individual band (0 = uniform hue, 1 = subtle, 2 = vivid).");
            }
            ImGui::SliderFloat("Per-Band Color Variety##proc", &re.procedural.per_band_color_variety, 0.0f, 2.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Controls color and albedo tint differences between separate bands (0 = identical palette, 1 = subtle, 2 = distinct).");
            }

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
                const auto& p = rendering::get_ring_preset(re.procedural.preset_idx);
                auto cfg = rendering::ProceduralRingConfig::from_preset(p, static_cast<uint32_t>(re.procedural.seed));
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
                cfg.opacity_variety = re.procedural.opacity_variety;
                cfg.single_band_color_variety = re.procedural.single_band_color_variety;
                cfg.per_band_color_variety = re.procedural.per_band_color_variety;
                return cfg;
            };

            if (ImGui::Button("Generate & Apply Rings##proc", ImVec2(-1.0f, 0.0f))) {
                auto cfg = make_config();
                auto result = rendering::generate_procedural_ring_profile(cfg);
                celestial_renderer_.set_custom_ring_texture(system().name(), b.definition.name, result.profile);
                rings = rendering::build_procedural_ring_layers(result);
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
                    rings = rendering::build_procedural_ring_layers(result);
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

} // namespace sf
