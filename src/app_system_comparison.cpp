#include "app.hpp"
#include "core/log.hpp"
#include <format>
#include <optional>
#include <stdexcept>

namespace sf {

namespace {

constexpr double SOLAR_TO_EARTH_MASSES = 332946.0487;

const char* motion_label(ephemeris::MotionModel model) {
    if (model == ephemeris::MotionModel::Spice) return "SPICE ephemeris";
    if (model == ephemeris::MotionModel::NBody) return "N-Body (IAS15)";
    return "Keplerian";
}

}

CommandResult App::add_system_comparison(std::string_view system_name) {
    if (system_name == "Ephemeris Mode") return {false, false, "Ephemeris Mode cannot be compared."};
    if (comparisons_.size() >= comparison::COMPARISON_SOFT_CAP)
        return {false, false, std::format("At most {} systems can be compared at once.", comparison::COMPARISON_SOFT_CAP)};
    if (simulation_.is_rendering_timeline() || ui_state_.timeline_scrub.active)
        return {false, false, "Close the timeline before comparing systems."};
    ephemeris::LoadedSystem loaded;
    try {
        loaded = sys_mgr_.load_system(system_name);
    } catch (const std::exception& error) {
        return {false, false, error.what()};
    }
    if (auto result = simulation_.wait(); !result) return result;
    if (comparisons_.empty()) comparison_generation_ = system().generation();
    comparisons_.set_force_keplerian(ui_settings_.comparison_force_keplerian);
    auto result = comparisons_.add(std::string(system_name), std::move(loaded), simulation_.clock());
    if (!result) return result;
    sf::log::info("Comparing with {} ({} of {})", system_name, comparisons_.size(), comparison::COMPARISON_SOFT_CAP);
    respace_system_comparisons();
    return result;
}

void App::remove_system_comparison(size_t slot) {
    comparisons_.remove(slot);
    respace_system_comparisons();
}

void App::respace_system_comparisons() {
    if (ui_settings_.comparison_auto_spacing) comparisons_.auto_space(system());
    compose_scene();
}

void App::advance_system_comparisons() {
    if (comparisons_.empty()) return;
    auto result = comparisons_.synchronize(simulation_.clock());
    if (result && system().generation() != comparison_generation_) {
        comparison_generation_ = system().generation();
        result = simulation_.wait();
        if (result) result = comparisons_.jump(system().time());
        if (result && ui_settings_.comparison_auto_spacing) comparisons_.auto_space(system());
    }
    if (result) result = comparisons_.advance_async(frame_dt_);
    if (result) return;
    simulation_error_ = result.error;
    simulation_.apply(ephemeris::SetPaused{true});
    comparisons_.synchronize(simulation_.clock());
    sf::log::error("Comparison paused: {}", result.error);
}

void App::check_system_comparison_sync() {
    if (auto result = simulation_.wait(); !result) throw std::runtime_error(result.error);
    if (auto result = comparisons_.wait(); !result) throw std::runtime_error(result.error);
    const auto& clock = simulation_.clock();
    for (size_t i = 0; i < comparisons_.size(); ++i) {
        const auto& slot = comparisons_.slot(i).clock();
        if (slot.paused != clock.paused || slot.warp != clock.warp || slot.direction != clock.direction)
            throw std::runtime_error("Compared system clock flags diverged from the primary.");
    }
    if (comparisons_.max_time_skew_years(clock.time) > comparison::ComparisonSet::skew_tolerance_years(clock, frame_dt_))
        throw std::runtime_error("Compared system time diverged from the primary.");
}

void App::compose_scene() {
    if (comparisons_.empty()) {
        if (composer_.map().slots() != 0) composer_.reset();
    } else {
        std::vector<comparison::ComposedMember> members;
        members.reserve(comparisons_.size() + 1);
        glm::dvec3 primary_offset = comparisons_.overlay() ? -comparison::barycentre_au(system()) : glm::dvec3{0.0};
        members.push_back({&system(), primary_offset, comparisons_.primary_lit_by()});
        for (size_t i = 0; i < comparisons_.size(); ++i)
            members.push_back({&comparisons_.slot(i).snapshot(), comparisons_.placement_au(i), comparisons_.slot_lit_by(i)});
        composer_.compose(members);
    }
    const auto& world = scene();
    auto identity_at = [&](int index) -> uint64_t {
        return index >= 0 && size_t(index) < world.size() ? world.body(size_t(index)).revisions.identity : 0;
    };
    auto follow = [&](int& index, SceneSelection& memory, int fallback) {
        if (index >= 0 && index == memory.index && memory.identity != 0 && identity_at(index) != memory.identity) {
            int found = -1;
            for (size_t i = 0; i < world.size() && found < 0; ++i)
                if (world.body(i).revisions.identity == memory.identity) found = int(i);
            if (found >= 0) index = found;
        }
        if (index >= 0 && size_t(index) >= world.size()) index = fallback;
        memory = {index, identity_at(index)};
    };
    follow(camera_.tracking_idx, tracked_memory_, world.empty() ? -1 : world.star_index());
    follow(camera_.centered_idx, centered_memory_, -1);
    follow(ui_state_.inspected_body, inspected_memory_, -1);
}

CommandResult App::open_comparison_for_editing(size_t slot) {
    if (slot >= comparisons_.size()) return {false, false, "No compared system in that slot."};
    if (auto result = simulation_.wait(); !result) return result;
    if (auto result = comparisons_.wait(); !result) return result;
    uint64_t inspected = inspected_memory_.identity;
    bool timezone = ui_state_.use_local_timezone;
    auto result = comparisons_.swap_primary(slot, simulation_);
    if (!result) return result;
    comparison_generation_ = system().generation();
    ui_state_ = {};
    ui_state_.use_local_timezone = timezone;
    respace_system_comparisons();
    for (size_t i = 0; i < system().size(); ++i)
        if (system().body(i).revisions.identity == inspected) ui_state_.inspected_body = int(i);
    inspected_memory_ = {};
    compose_scene();
    simulation_error_.clear();
    sf::log::info("Editing {}; {} is now compared", system().name(), comparisons_.slot(slot).system_name);
    return result;
}

void App::render_system_comparator_window() {
    if (!show_system_comparator_) return;
    ImGui::SetNextWindowSize(ImVec2(440.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("System Comparator", &show_system_comparator_)) {
        ImGui::End();
        return;
    }
    ui::heading("System Comparator", "Simulate other systems beside the one you are editing.");
    ImGui::TextDisabled("Editing: %s", system().name().c_str());

    std::vector<const std::string*> candidates;
    for (const auto& name : system_names_) if (name != "Ephemeris Mode") candidates.push_back(&name);
    comparator_candidate_ = std::clamp(comparator_candidate_, 0, std::max(0, int(candidates.size()) - 1));
    bool can_add = !candidates.empty() && comparisons_.size() < comparison::COMPARISON_SOFT_CAP &&
        !simulation_.is_rendering_timeline() && !ui_state_.timeline_scrub.active;
    ImGui::SetNextItemWidth(-90.0f);
    if (ImGui::BeginCombo("##compare_candidate", candidates.empty() ? "No systems" : candidates[comparator_candidate_]->c_str())) {
        for (int i = 0; i < int(candidates.size()); ++i)
            if (ImGui::Selectable(candidates[i]->c_str(), i == comparator_candidate_)) comparator_candidate_ = i;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!can_add);
    if (ImGui::Button("Compare", ImVec2(-1.0f, 0.0f))) {
        auto result = add_system_comparison(*candidates[comparator_candidate_]);
        comparator_error_ = result ? std::string{} : result.error;
    }
    ImGui::EndDisabled();
    if (comparisons_.size() >= comparison::COMPARISON_SOFT_CAP)
        ImGui::TextDisabled("At most %zu systems can be compared at once.", comparison::COMPARISON_SOFT_CAP);
    else if (simulation_.is_rendering_timeline() || ui_state_.timeline_scrub.active)
        ImGui::TextDisabled("Close the timeline before comparing systems.");

    if (ImGui::Checkbox("Keplerian motion for new comparisons", &ui_settings_.comparison_force_keplerian))
        comparisons_.set_force_keplerian(ui_settings_.comparison_force_keplerian);
    if (ImGui::Button("Comparison Table...")) show_comparison_table_ = true;
    bool overlay = comparisons_.overlay();
    if (ImGui::Checkbox("Overlay systems", &overlay)) {
        comparisons_.set_overlay(overlay);
        compose_scene();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Centre every system on its own barycentre. Each system is lit only by its own stars.");
    ImGui::BeginDisabled(overlay);
    if (ImGui::Checkbox("Auto spacing", &ui_settings_.comparison_auto_spacing)) respace_system_comparisons();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Place systems along +X from each system's outermost orbit.");
    ImGui::EndDisabled();

    std::optional<size_t> remove, edit;
    for (size_t i = 0; i < comparisons_.size(); ++i) {
        auto& slot = comparisons_.slot(i);
        ImGui::PushID(int(i));
        ImGui::SeparatorText(slot.system_name.c_str());
        ImGui::TextDisabled("%zu bodies | %s", slot.snapshot().size(), motion_label(slot.snapshot().definition().motion_model));
        ImGui::BeginDisabled(overlay || ui_settings_.comparison_auto_spacing);
        float offset[3] = {float(slot.offset_au.x), float(slot.offset_au.y), float(slot.offset_au.z)};
        if (ImGui::DragFloat3("Offset (AU)", offset, 0.1f, -1e6f, 1e6f, "%.2f")) {
            slot.offset_au = {offset[0], offset[1], offset[2]};
            compose_scene();
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(overlay);
        std::string inbound = std::format("{} starlight reaches this system", system().name());
        if (ImGui::Checkbox(inbound.c_str(), &slot.primary_lights_slot)) compose_scene();
        std::string outbound = std::format("This system's starlight reaches {}", system().name());
        if (ImGui::Checkbox(outbound.c_str(), &slot.slot_lights_primary)) compose_scene();
        ImGui::EndDisabled();
        if (ImGui::Button("Open for Editing")) edit = i;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Swap this system with %s. Both keep their state.", system().name().c_str());
        ImGui::SameLine();
        if (ImGui::Button("Remove")) remove = i;
        ImGui::PopID();
    }
    if (edit) {
        auto result = open_comparison_for_editing(*edit);
        comparator_error_ = result ? std::string{} : result.error;
    } else if (remove) {
        remove_system_comparison(*remove);
    }
    if (!comparator_error_.empty()) ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", comparator_error_.c_str());
    ImGui::End();
}

void App::render_compared_outliner() {
    if (comparisons_.empty()) return;
    for (size_t i = 0; i < comparisons_.size(); ++i) {
        const auto& snapshot = comparisons_.slot(i).snapshot();
        ImGui::PushID(std::format("compared_system_{}", i).c_str());
        ImGui::Spacing();
        ImGui::SeparatorText(comparisons_.slot(i).system_name.c_str());
        ImGui::TextDisabled("Compared (read-only) | %s", motion_label(snapshot.definition().motion_model));
        if (ImGui::SmallButton("Open for Editing")) {
            auto result = open_comparison_for_editing(i);
            comparator_error_ = result ? std::string{} : result.error;
            if (!result) simulation_error_ = result.error;
            ImGui::PopID();
            return;
        }
        for (int root : snapshot.reference_hierarchy().roots) render_compared_outliner_node(i, root);
        ImGui::PopID();
    }
}

void App::render_compared_outliner_node(size_t slot, int local) {
    const auto& snapshot = comparisons_.slot(slot).snapshot();
    int index = composer_.map().composite(slot + 1, size_t(local));
    if (index < 0) return;
    const auto& body = snapshot.body(size_t(local));
    const auto& children = snapshot.reference_hierarchy().children[size_t(local)];
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (children.empty()) flags |= ImGuiTreeNodeFlags_Leaf;
    if (body.definition.is_star) flags |= ImGuiTreeNodeFlags_DefaultOpen;
    if (ui_state_.inspected_body == index) flags |= ImGuiTreeNodeFlags_Selected;
    std::string label = camera_.tracking_idx == index ? std::format("{} [Tracked]", body.definition.name) : body.definition.name;
    bool opened = ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<intptr_t>(local)), flags, "%s", label.c_str());
    if (ImGui::IsItemClicked()) {
        ui_state_.inspected_body = index;
        ui_state_.edit_mode = false;
        ui_state_.inspect_barycenter = false;
    }
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) camera_.focus_body(index, scene(), true);
    if (opened) {
        for (int child : children) render_compared_outliner_node(slot, child);
        ImGui::TreePop();
    }
}

void App::render_compared_inspector() {
    int index = ui_state_.inspected_body;
    if (comparisons_.empty() || index < 0 || size_t(index) >= scene().size()) return;
    auto source = composer_.map().source(size_t(index));
    if (source.slot == 0 || source.slot > comparisons_.size()) return;
    const auto& slot = comparisons_.slot(source.slot - 1);
    const auto& snapshot = slot.snapshot();
    if (source.local >= snapshot.size()) return;
    const auto& body = snapshot.body(source.local);
    const auto& def = body.definition;

    ui::WorkspaceLayout layout{};
    ImGuiWindowFlags flags = ui_.prepare_side_panel(ui_settings_, ui_state_, ui::PanelSide::Right, layout);
    std::string title = std::format("{} ({})###inspector", def.name, slot.system_name);
    if (!ImGui::Begin(title.c_str(), &ui_settings_.show_inspector, flags)) {
        ImGui::End();
        return;
    }
    ImGui::TextColored(ui::ACCENT, "%s", def.type.c_str());
    ImGui::TextDisabled("Compared from %s (read-only)", slot.system_name.c_str());

    bool tracked = camera_.tracking_idx == index;
    float gap = 6.0f * layout.scale;
    float button = (ImGui::GetContentRegionAvail().x - gap * 2.0f) / 3.0f;
    if (ImGui::Button(tracked ? "Tracking##trk" : "Track##trk", ImVec2(button, 0.0f))) {
        if (tracked) camera_.tracking_idx = -1;
        else camera_.focus_body(index, scene(), false);
    }
    ImGui::SameLine(0.0f, gap);
    if (ImGui::Button("Center##cnt", ImVec2(button, 0.0f))) camera_.center_body(index, scene());
    ImGui::SameLine(0.0f, gap);
    if (ImGui::Button("Go To##goto", ImVec2(button, 0.0f))) camera_.focus_body(index, scene(), true);
    if (ImGui::Button("Open for Editing", ImVec2(-1.0f, 0.0f))) {
        auto result = open_comparison_for_editing(source.slot - 1);
        if (!result) simulation_error_ = result.error;
        ImGui::End();
        return;
    }

    ui::section("Physical", "Values from the compared system's definition.");
    ui::value_row("Mass", def.is_star || def.is_black_hole ? std::format("{:.4g} M_sun", def.mass_msun)
                                                           : std::format("{:.4g} M_earth", def.mass_msun * SOLAR_TO_EARTH_MASSES));
    ui::value_row("Radius", std::format("{:.6g} km", def.radius_au() * AU_TO_KM));
    if (def.oblateness > 0.0) ui::value_row("Oblateness", std::format("{:.5f}", def.oblateness));
    if (std::abs(def.rotation_period_hr) > 1e-9) ui::value_row("Rotation period", std::format("{:.4g} h", def.rotation_period_hr));
    if (def.is_star) {
        ui::value_row("Temperature", std::format("{:.0f} K", def.temp_k));
        ui::value_row("Luminosity", std::format("{:.4g} L_sun", def.lum_eq));
        if (!def.spectral_class.empty()) ui::value_row("Spectral class", def.spectral_class);
    }
    if (def.atmosphere.enabled) ui::value_row("Atmosphere", "Yes");
    if (!def.rings.empty()) ui::value_row("Ring layers", std::format("{}", def.rings.size()));

    int parent = snapshot.reference_parent(source.local);
    if (parent >= 0 && size_t(parent) < snapshot.size()) {
        const auto& host = snapshot.body(size_t(parent));
        ui::section("Orbit", std::format("Around {}", host.definition.name));
        double distance = glm::length(body.motion.position.value - host.motion.position.value);
        ui::value_row("Distance", std::format("{:.6g} AU", distance));
        if (def.semi_major_au > 0.0) {
            ui::value_row("Semi-major axis", std::format("{:.6g} AU", def.semi_major_au));
            ui::value_row("Eccentricity", std::format("{:.5f}", def.eccentricity));
            ui::value_row("Inclination", std::format("{:.3f} deg", def.inc_deg));
            double mass = host.definition.mass_msun + def.mass_msun;
            if (mass > 0.0) ui::value_row("Period", std::format("{:.6g} yr", std::sqrt(std::pow(def.semi_major_au, 3.0) / mass)));
        }
    }
    ImGui::End();
}

namespace {

enum ComparisonMetric { MetricPeriod, MetricSemiMajor, MetricEccentricity, MetricRadius, MetricMass, MetricTemperature, MetricCount };
constexpr const char* METRIC_NAMES[] = {"Orbital period", "Semi-major axis", "Eccentricity", "Radius", "Mass", "Equilibrium temperature (A = 0)"};

std::optional<double> metric_value(const comparison::BodyMetrics& metrics, int metric) {
    switch (metric) {
    case MetricPeriod: return metrics.period_yr;
    case MetricSemiMajor: return metrics.semi_major_au;
    case MetricEccentricity: return metrics.eccentricity;
    case MetricRadius: return metrics.radius_km;
    case MetricMass: return metrics.mass_msun;
    case MetricTemperature: return metrics.equilibrium_k;
    default: return std::nullopt;
    }
}

std::string metric_text(const comparison::BodyMetrics& metrics, int metric) {
    auto value = metric_value(metrics, metric);
    if (!value) return "-";
    switch (metric) {
    case MetricPeriod: return *value >= 1.0 ? std::format("{:.4g} yr", *value) : std::format("{:.4g} d", *value * 365.25);
    case MetricSemiMajor: return std::format("{:.4g} AU", *value);
    case MetricEccentricity: return std::format("{:.4f}", *value);
    case MetricRadius: return std::format("{:.5g} km", *value);
    case MetricMass: return metrics.star ? std::format("{:.4g} M_sun", *value) : std::format("{:.4g} M_earth", *value * SOLAR_TO_EARTH_MASSES);
    case MetricTemperature: return std::format("{:.0f} K", *value);
    default: return "-";
    }
}

}

void App::render_comparison_table() {
    if (!show_comparison_table_) return;
    ImGui::SetNextWindowSize(ImVec2(640.0f, 420.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Comparison Table", &show_comparison_table_)) {
        ImGui::End();
        return;
    }
    if (comparisons_.empty()) {
        ImGui::TextDisabled("Add a system in View > System Comparator to compare bodies side by side.");
        ImGui::End();
        return;
    }
    std::vector<const ephemeris::SystemSnapshot*> systems{&system()};
    std::vector<std::string> names{system().name() + " (editing)"};
    for (size_t i = 0; i < comparisons_.size(); ++i) {
        systems.push_back(&comparisons_.slot(i).snapshot());
        names.push_back(comparisons_.slot(i).system_name);
    }

    int match = comparison_match_ == comparison::MatchMode::Name ? 1 : 0;
    const char* match_names[] = {"Orbit order", "Body name"};
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::Combo("Match by", &match, match_names, IM_ARRAYSIZE(match_names)))
        comparison_match_ = match == 1 ? comparison::MatchMode::Name : comparison::MatchMode::OrbitOrder;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(230.0f);
    ImGui::Combo("Metric", &comparison_metric_, METRIC_NAMES, IM_ARRAYSIZE(METRIC_NAMES));
    comparison_metric_ = std::clamp(comparison_metric_, 0, int(MetricCount) - 1);
    ImGui::TextDisabled("Orbits are osculating around each body's reference parent. Ratios compare with the system being edited.");

    auto rows = comparison::match_bodies(systems, comparison_match_);
    const comparison::MatchRow* selected = nullptr;
    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchSame;
    float detail_height = comparison_table_row_.empty() ? 0.0f : 170.0f;
    if (ImGui::BeginTable("comparison_rows", int(systems.size()) + 1, flags, ImVec2(0.0f, -detail_height))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("Match", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        for (const auto& name : names) ImGui::TableSetupColumn(name.c_str());
        ImGui::TableHeadersRow();
        for (const auto& row : rows) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            bool is_selected = row.label == comparison_table_row_;
            if (ImGui::Selectable(row.label.c_str(), is_selected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap))
                comparison_table_row_ = is_selected ? std::string{} : row.label;
            if (is_selected) selected = &row;
            std::optional<double> reference;
            if (row.bodies[0] >= 0) reference = metric_value(comparison::measure_body(system(), size_t(row.bodies[0])), comparison_metric_);
            for (size_t slot = 0; slot < systems.size(); ++slot) {
                ImGui::TableSetColumnIndex(int(slot) + 1);
                int local = row.bodies[slot];
                if (local < 0) {
                    ImGui::TextDisabled("-");
                    continue;
                }
                auto metrics = comparison::measure_body(*systems[slot], size_t(local));
                ImGui::PushID(int(slot) * 100000 + local);
                if (ImGui::SmallButton(metrics.name.c_str())) {
                    int index = comparisons_.empty() ? local : composer_.map().composite(slot, size_t(local));
                    if (index >= 0) {
                        ui_state_.inspected_body = index;
                        ui_state_.edit_mode = false;
                        ui_state_.inspect_barycenter = false;
                    }
                }
                ImGui::PopID();
                ImGui::SameLine();
                auto value = metric_value(metrics, comparison_metric_);
                if (slot > 0 && value && reference && *reference != 0.0 && comparison_metric_ != MetricEccentricity)
                    ImGui::Text("%s  (x%.3g)", metric_text(metrics, comparison_metric_).c_str(), *value / *reference);
                else
                    ImGui::TextUnformatted(metric_text(metrics, comparison_metric_).c_str());
            }
        }
        ImGui::EndTable();
    }
    if (!selected) comparison_table_row_.clear();

    if (selected && ImGui::BeginTable("comparison_detail", int(systems.size()) + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn(selected->label.c_str(), ImGuiTableColumnFlags_WidthFixed, 200.0f);
        for (const auto& name : names) ImGui::TableSetupColumn(name.c_str());
        ImGui::TableHeadersRow();
        std::vector<std::optional<comparison::BodyMetrics>> metrics(systems.size());
        for (size_t slot = 0; slot < systems.size(); ++slot)
            if (selected->bodies[slot] >= 0) metrics[slot] = comparison::measure_body(*systems[slot], size_t(selected->bodies[slot]));
        for (int metric = 0; metric < MetricCount; ++metric) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(METRIC_NAMES[metric]);
            for (size_t slot = 0; slot < systems.size(); ++slot) {
                ImGui::TableSetColumnIndex(int(slot) + 1);
                ImGui::TextUnformatted(metrics[slot] ? metric_text(*metrics[slot], metric).c_str() : "-");
            }
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

}
