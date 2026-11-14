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

void App::open_jump_timeline_modal() {
    ui_state_.jump_timeline.reset_to_now(system().definition().epoch_utc, system().time(), !ui_state_.use_local_timezone);
}

void App::render_jump_timeline_modal() {
    if (!ui_state_.jump_timeline.active) return;

    ImGui::SetNextWindowSize(ImVec2(540.0f, 540.0f), ImGuiCond_Always);
    ImGui::OpenPopup("Jump to Date / Render Timeline###jump_timeline_modal");
    if (ImGui::BeginPopupModal("Jump to Date / Render Timeline###jump_timeline_modal", &ui_state_.jump_timeline.active, ImGuiWindowFlags_NoResize)) {
        auto& jd = ui_state_.jump_timeline;

        ui::heading("Jump to Date / Render Timeline", "Jump instantly or render a scrubbable timeline to a target epoch.");

        auto motion_model = system().definition().motion_model;
        if (motion_model == ephemeris::MotionModel::Spice) {
            ImGui::TextColored(ImVec4(0.35f, 0.85f, 1.0f, 1.0f), "Mode: NASA SPICE Ephemeris Playback");
        } else if (motion_model == ephemeris::MotionModel::Keplerian) {
            ImGui::TextColored(ImVec4(0.45f, 0.90f, 0.55f, 1.0f), "Mode: Analytical Keplerian (Jacobi)");
        } else {
            ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.30f, 1.0f), "Mode: N-Body Simulation (IAS15)");
        }

        auto cur_utc = time::calculate_sim_date(system().definition().epoch_utc, system().time(), false);
        auto cur_local = time::calculate_sim_date(system().definition().epoch_utc, system().time(), true);

        ImGui::TextDisabled("Current Sim Time (UTC):   %04lld-%02d-%02d %02d:%02d:%02d UTC",
                            static_cast<long long>(cur_utc.year), cur_utc.month, cur_utc.day,
                            cur_utc.hour, cur_utc.minute, cur_utc.second);
        ImGui::TextDisabled("Current Sim Time (Local): %04lld-%02d-%02d %02d:%02d:%02d %s",
                            static_cast<long long>(cur_local.year), cur_local.month, cur_local.day,
                            cur_local.hour, cur_local.minute, cur_local.second, cur_local.tz.c_str());
        ImGui::Separator();

        ImGui::Text("Input Time Standard:");
        ImGui::SameLine();
        bool prev_utc = jd.is_utc;
        std::string local_radio = std::format("Local Time ({})###jd_local_tz", cur_local.tz);
        if (ImGui::RadioButton(local_radio.c_str(), !jd.is_utc)) {
            jd.is_utc = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("UTC###jd_utc_tz", jd.is_utc)) {
            jd.is_utc = true;
        }
        if (prev_utc != jd.is_utc) {
            time::DateTimeUTC current_draft{jd.target_year, jd.target_month, jd.target_day, jd.target_hour, jd.target_minute, jd.target_second, ""};
            auto sim_t = time::sim_time_from_date(system().definition().epoch_utc, current_draft, prev_utc);
            auto converted = time::calculate_sim_date(system().definition().epoch_utc, sim_t, !jd.is_utc);
            jd.target_year = converted.year;
            jd.target_month = converted.month;
            jd.target_day = converted.day;
            jd.target_hour = converted.hour;
            jd.target_minute = converted.minute;
            jd.target_second = converted.second;
        }

        ImGui::PushItemWidth(100.0f);
        ImGui::InputScalar("Year", ImGuiDataType_S64, &jd.target_year);
        ImGui::SameLine();
        ImGui::InputInt("Month", &jd.target_month);
        ImGui::SameLine();
        ImGui::InputInt("Day", &jd.target_day);
        ImGui::PopItemWidth();

        ImGui::PushItemWidth(100.0f);
        ImGui::InputInt("Hour", &jd.target_hour);
        ImGui::SameLine();
        ImGui::InputInt("Min", &jd.target_minute);
        ImGui::SameLine();
        ImGui::InputInt("Sec", &jd.target_second);
        ImGui::PopItemWidth();

        jd.target_month = std::clamp(jd.target_month, 1, 12);
        jd.target_day = std::clamp(jd.target_day, 1, 31);
        jd.target_hour = std::clamp(jd.target_hour, 0, 23);
        jd.target_minute = std::clamp(jd.target_minute, 0, 59);
        jd.target_second = std::clamp(jd.target_second, 0, 59);

        ImGui::Text("Presets:");
        ImGui::SameLine();
        if (ImGui::SmallButton("Today/Now")) {
            auto now = time::current_wall_clock_date(!jd.is_utc);
            jd.target_year = now.year; jd.target_month = now.month; jd.target_day = now.day;
            jd.target_hour = now.hour; jd.target_minute = now.minute; jd.target_second = now.second;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Set target date to the current real-world wall-clock time.");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Current Sim")) {
            auto cur = time::calculate_sim_date(system().definition().epoch_utc, system().time(), !jd.is_utc);
            jd.target_year = cur.year; jd.target_month = cur.month; jd.target_day = cur.day;
            jd.target_hour = cur.hour; jd.target_minute = cur.minute; jd.target_second = cur.second;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Reset target date to the current simulation clock.");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Sim Epoch")) {
            auto ep = time::calculate_sim_date(system().definition().epoch_utc, units::Years{0.0}, !jd.is_utc);
            jd.target_year = ep.year; jd.target_month = ep.month; jd.target_day = ep.day;
            jd.target_hour = ep.hour; jd.target_minute = ep.minute; jd.target_second = ep.second;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Set target date to the initial epoch of the loaded system/kernel.");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("J2000")) {
            time::DateTimeUTC j2000{2000, 1, 1, 12, 0, 0, "UTC"};
            if (!jd.is_utc) {
                auto st = time::sim_time_from_date(system().definition().epoch_utc, j2000, true);
                j2000 = time::calculate_sim_date(system().definition().epoch_utc, st, true);
            }
            jd.target_year = j2000.year; jd.target_month = j2000.month; jd.target_day = j2000.day;
            jd.target_hour = j2000.hour; jd.target_minute = j2000.minute; jd.target_second = j2000.second;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("+1 Year")) { jd.target_year += 1; }
        ImGui::SameLine();
        if (ImGui::SmallButton("-1 Year")) { jd.target_year -= 1; }
        ImGui::SameLine();
        if (ImGui::SmallButton("+10 Years")) { jd.target_year += 10; }

        time::DateTimeUTC target_dt{jd.target_year, jd.target_month, jd.target_day, jd.target_hour, jd.target_minute, jd.target_second, ""};
        units::Years target_sim_time = time::sim_time_from_date(system().definition().epoch_utc, target_dt, jd.is_utc);
        double diff_years = target_sim_time.value - system().time().value;

        ImGui::Spacing();
        ImGui::TextDisabled("Target Sim Time: %.5f yr (%+.5f yr from current)", target_sim_time.value, diff_years);
        ImGui::Separator();

        bool can_jump = (motion_model == ephemeris::MotionModel::Keplerian || motion_model == ephemeris::MotionModel::Spice);

        if (can_jump) {
            if (ImGui::Button("Jump to Date (Instant)", ImVec2(240.0f, 32.0f))) {
                auto res = simulation_.apply(ephemeris::JumpToTime{target_sim_time});
                if (res) res = comparisons_.jump(target_sim_time);
                if (!res) {
                    jd.error = res.error;
                } else {
                    jd.active = false;
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(No timeline recording)");
            ImGui::Spacing();
        }

        const char* res_names[] = {
            "Auto (Adaptive - Min Orbital Period)",
            "1,000 Steps",
            "10,000 Steps",
            "50,000 Steps",
            "200,000 Steps"
        };
        ImGui::PushItemWidth(280.0f);
        ImGui::Combo("Timeline Resolution", &jd.resolution_choice, res_names, IM_ARRAYSIZE(res_names));
        ImGui::PopItemWidth();

        size_t requested_steps = 0;
        if (jd.resolution_choice == 1) requested_steps = 1000;
        else if (jd.resolution_choice == 2) requested_steps = 10000;
        else if (jd.resolution_choice == 3) requested_steps = 50000;
        else if (jd.resolution_choice == 4) requested_steps = 200000;

        ImGui::BeginDisabled(!comparisons_.empty());
        if (ImGui::Button("Render Timeline to Date", ImVec2(240.0f, 32.0f))) {
            if (std::abs(diff_years) < 1e-9) {
                jd.error = "Target time is identical to current time.";
            } else {
                simulation_.start_timeline_render(target_sim_time, requested_steps);
                jd.active = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled(comparisons_.empty() ? "(Background worker integration)" : "(Unavailable while comparing systems)");

        if (motion_model == ephemeris::MotionModel::NBody) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f),
                               "Note: Instant jump without integration is only physical in Keplerian/SPICE modes.\n"
                               "N-Body uses Render Timeline to integrate equations of motion physically.");
        }

        if (!jd.error.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", jd.error.c_str());
        }

        ImGui::Spacing();
        ImGui::Separator();
        if (ImGui::Button("Cancel", ImVec2(120.0f, 28.0f))) {
            jd.active = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void App::trigger_timeline_bsp_export() {
    if (!ui_state_.timeline_scrub.recording || ui_state_.timeline_scrub.recording->empty()) {
        open_jump_timeline_modal();
        return;
    }
    if (ui_state_.timeline_scrub.export_in_progress) return;

    auto slugify = [](std::string_view s) {
        std::string out;
        for (char c : s) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
                out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else if (c == ' ') {
                out += '_';
            }
        }
        return out.empty() ? "system" : out;
    };

    std::string sys_slug = slugify(system().name());
    auto export_dir = paths::external_path("exports", "ephemeris", sys_slug);
    std::error_code ec;
    std::filesystem::create_directories(export_dir, ec);

    auto now_t = std::time(nullptr);
    std::tm tm_now{};
#if defined(_WIN32)
    _gmtime64_s(&tm_now, &now_t);
#else
    gmtime_r(&now_t, &tm_now);
#endif
    std::string filename = std::format("{}_timeline_{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}.bsp",
        sys_slug, tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday,
        tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec);
    auto bsp_path = export_dir / filename;

    ui_state_.timeline_scrub.export_in_progress = true;
    ui_state_.timeline_scrub.export_progress = 0.0f;
    ui_state_.timeline_scrub.export_status = "Exporting SPK...";

    std::thread([this, bsp_path, recording = ui_state_.timeline_scrub.recording,
                 def = system().definition(), hier = system().reference_hierarchy()]() {
        ephemeris::SpkExportOptions opts;
        opts.spk_type = 2;
        opts.cheb_degree = 13;
        opts.cheb_tol_km = 1.0;
        opts.use_hierarchical_center = true;
        opts.progress_callback = [this](float p, std::string_view s) {
            ui_state_.timeline_scrub.export_progress = p;
            ui_state_.timeline_scrub.export_status = std::string(s);
        };
        std::string spice_err;
        auto spice = ephemeris::SpiceEphemeris::open(spice_err);
        auto res = ephemeris::SpkExporter::export_timeline(bsp_path, *recording, def, hier, spice, opts);
        if (res.success) {
            ui_state_.timeline_scrub.export_status = "Exported: " + res.bsp_path.filename().string();
        } else {
            ui_state_.timeline_scrub.export_status = "Error: " + res.error;
        }
        ui_state_.timeline_scrub.export_in_progress = false;
    }).detach();
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
                const char* alt_key;
                const char* title;
            };
            static constexpr GroupSpec kGroups[] = {
                {"Required Core", "Core", "Core Ephemeris (Essential)"},
                {"Mars Moons", "Inner Moons", "Inner Planet Moons (Mars / Phobos / Deimos)"},
                {"Jupiter Moons", "Jupiter", "Jupiter Moons (Optional)"},
                {"Saturn Moons", "Saturn", "Saturn Moons (Optional)"},
                {"Uranus Moons", "Uranus", "Uranus Moons (Optional)"},
                {"Neptune Moons", "Neptune", "Neptune Moons (Optional)"},
                {"Pluto System", "Pluto", "Pluto System (Optional)"},
            };

            std::vector<bool> rendered(defaults.size(), false);

            for (const auto& grp : kGroups) {
                bool has_any = false;
                for (size_t i = 0; i < defaults.size(); ++i) {
                    const auto& k = defaults[i];
                    if (k.group == grp.key || (grp.alt_key && k.group == grp.alt_key)) {
                        has_any = true;
                        break;
                    }
                }
                if (!has_any) continue;

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.78f, 0.95f, 1.0f));
                ImGui::TextUnformatted(grp.title);
                ImGui::PopStyleColor();
                ImGui::Separator();

                for (size_t i = 0; i < defaults.size(); ++i) {
                    const auto& k = defaults[i];
                    if (k.group != grp.key && (!grp.alt_key || k.group != grp.alt_key)) continue;
                    rendered[i] = true;
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

            bool has_unrendered = false;
            for (size_t i = 0; i < defaults.size(); ++i) {
                if (!rendered[i]) {
                    has_unrendered = true;
                    break;
                }
            }
            if (has_unrendered) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.78f, 0.95f, 1.0f));
                ImGui::TextUnformatted("Other Ephemeris Kernels");
                ImGui::PopStyleColor();
                ImGui::Separator();

                for (size_t i = 0; i < defaults.size(); ++i) {
                    if (rendered[i]) continue;
                    const auto& k = defaults[i];
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
