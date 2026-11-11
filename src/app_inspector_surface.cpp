#include "app.hpp"
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

} // namespace sf
