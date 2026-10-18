#pragma once

#include "core/math_utils.hpp"
#include "core/time_utils.hpp"
#include "ephemeris/simulation.hpp"
#include "ephemeris/spice_ephemeris.hpp"
#include "ephemeris/spk_exporter.hpp"
#include "physics/star_calc.hpp"
#include <algorithm>
#include <format>
#include <string>

namespace sf::ui {

struct StarCreatorDraft {
    physics::StarParameters params{};
    physics::StarForgeResult preview{};
    bool dirty = false;

    void recompute() {
        preview = physics::StarCalculator::forge(params);
        dirty = false;
    }

    void init_from_body(const ephemeris::BodyDefinition& b) {
        if (b.source_json.contains("star_props") && b.source_json["star_props"].is_object()) {
            params = physics::StarCalculator::from_json(b.source_json["star_props"]);
        } else {
            params = {};
            params.mass_msun = b.mass_msun > 0.0 ? b.mass_msun : 1.0;
            params.radius_rsun = b.radius_rsun > 0.0 ? b.radius_rsun : 1.0;
            params.temp_k = b.temp_k > 0.0 ? b.temp_k : 5778.0;
            params.lum_lsun = b.lum_eq > 0.0 ? b.lum_eq : 1.0;
        }
        recompute();
    }
};

struct CreateSystemDraft {
    bool active = false;
    std::string system_name = "New System";
    std::string star_name = "Star";
    StarCreatorDraft star{};
    std::string error;

    void reset() {
        active = true;
        system_name = "New System";
        star_name = "Star";
        star = {};
        star.recompute();
        error.clear();
    }
};

struct AtmosphereEditDraft {
    int body_index = -1;
    uint64_t generation = 0, revision = 0;
    ephemeris::AtmosphereDefinition values;
    bool dirty = false;
    std::string error;

    void reload(const ephemeris::SystemSnapshot& snapshot, int index) {
        body_index = index;
        generation = snapshot.generation();
        revision = snapshot.body(index).revisions.definition;
        values = snapshot.body(index).definition.atmosphere;
        dirty = false;
        error.clear();
    }
    void synchronize(const ephemeris::SystemSnapshot& snapshot, int index) {
        if (generation != snapshot.generation() || body_index != index ||
            (!dirty && revision != snapshot.body(index).revisions.definition)) reload(snapshot, index);
    }
    ephemeris::ApplyBodyAtmosphere command() const { return {generation, body_index, revision, values}; }
};

struct BodyEditDraft {
    int body_index = -1;
    uint64_t system_generation = 0;
    uint64_t body_revision = 0;
    int reference_parent = -1;
    ephemeris::BodyPhysicalProperties values{};
    StarCreatorDraft star{};
    bool dirty = false;
    std::string error;

    void reload(const ephemeris::SystemSnapshot& snapshot, int index) {
        body_index = index;
        system_generation = snapshot.generation();
        body_revision = snapshot.body(index).revisions.definition;
        reference_parent = snapshot.reference_parent(index);
        values = ephemeris::BodyPhysicalProperties::from(snapshot.body(index).definition);
        if (values.type == "Star" || snapshot.body(index).definition.is_star) {
            star.init_from_body(snapshot.body(index).definition);
        }
        dirty = false;
        error.clear();
    }

    void synchronize(const ephemeris::SystemSnapshot& snapshot, int index) {
        if (system_generation != snapshot.generation() || body_index != index ||
            (!dirty && (body_revision != snapshot.body(index).revisions.definition || reference_parent != snapshot.reference_parent(index)))) {
            reload(snapshot, index);
        }
    }

    void recompute_hydrostatic() {
        if (values.rotation_period_hr > 0.0 && values.mass.value > 0.0 && values.radius.value > 0.0) {
            auto shape = math::compute_hydrostatic_shape(values.mass.value, values.radius.value,
                                                         values.rotation_period_hr, values.type);
            values.oblateness = shape.oblateness;
            values.j2 = shape.j2;
            values.j4 = shape.j4;
        }
    }

    ephemeris::ApplyBodyPhysicalProperties command() const {
        return {system_generation, body_index, body_revision, values, reference_parent};
    }
};

struct AddBodyDraft {
    bool active = false;
    int parent_idx = 0;
    std::string name = "New Body";
    std::string type = "Terrestrial";
    glm::vec3 color_rgb{0.2f, 0.5f, 0.8f};

    double mass_input = 1.0;
    int mass_unit = 0;
    double radius_km = 6371.0;

    bool tidally_locked = false;
    double rotation_period_hr = 24.0;
    double axial_tilt_deg = 0.0;
    double oblateness = 0.0;
    double j2 = 0.0;
    double j4 = 0.0;

    int frame = 0;
    double semi_major_input = 1.0;
    bool distance_in_km = false;
    double eccentricity = 0.0;
    double inc_deg = 0.0;
    double omega_node_deg = 0.0;
    double arg_peri_deg = 0.0;
    double mean_anom_deg = 0.0;

    StarCreatorDraft star{};
    std::string error;

    void reset_for_parent(const ephemeris::SystemSnapshot& snapshot, int p_idx) {
        parent_idx = std::clamp(p_idx, 0, static_cast<int>(snapshot.size()) - 1);
        const auto& p = snapshot.body(parent_idx);
        bool is_moon = (parent_idx != snapshot.star_index() && !p.definition.is_star);
        if (is_moon) {
            name = std::format("{} Moon", p.definition.name);
            type = "Moon";
            mass_input = 1.0;
            mass_unit = 1;
            radius_km = 1737.0;
            color_rgb = glm::vec3(0.7f, 0.7f, 0.7f);
            tidally_locked = true;
            semi_major_input = 384400.0;
            distance_in_km = true;
            eccentricity = 0.05;
            inc_deg = 5.0;
        } else {
            name = std::format("{} Planet", p.definition.name);
            type = "Terrestrial";
            mass_input = 1.0;
            mass_unit = 0;
            radius_km = 6371.0;
            color_rgb = glm::vec3(0.2f, 0.5f, 0.8f);
            tidally_locked = false;
            semi_major_input = 1.0;
            distance_in_km = false;
            eccentricity = 0.02;
            inc_deg = 1.5;
        }
        rotation_period_hr = 24.0;
        axial_tilt_deg = 0.0;
        omega_node_deg = static_cast<double>(std::rand() % 360);
        arg_peri_deg = static_cast<double>(std::rand() % 360);
        mean_anom_deg = static_cast<double>(std::rand() % 360);
        star = {};
        star.params.mass_msun = 0.5;
        star.params.radius_rsun = 0.6;
        star.params.temp_k = 4000.0;
        star.recompute();
        error.clear();
        recompute_hydrostatic();
    }

    void randomize_angles() {
        omega_node_deg = std::round((static_cast<double>(std::rand()) / RAND_MAX * 360.0) * 1000.0) / 1000.0;
        arg_peri_deg = std::round((static_cast<double>(std::rand()) / RAND_MAX * 360.0) * 1000.0) / 1000.0;
        mean_anom_deg = std::round((static_cast<double>(std::rand()) / RAND_MAX * 360.0) * 1000.0) / 1000.0;
    }

    double get_mass_msun() const noexcept {
        if (type == "Star") return star.preview.mass_msun;
        if (mass_unit == 1) return mass_input * 3.694e-8;
        if (mass_unit == 0) return mass_input * 3.003e-6;
        return mass_input;
    }

    double get_semi_major_au() const noexcept {
        return distance_in_km ? (semi_major_input / AU_TO_KM) : semi_major_input;
    }

    void recompute_hydrostatic() {
        double m_msun = get_mass_msun();
        if (rotation_period_hr > 0.0 && m_msun > 0.0 && radius_km > 0.0) {
            auto shape = math::compute_hydrostatic_shape(m_msun, radius_km, rotation_period_hr, type);
            oblateness = shape.oblateness;
            j2 = shape.j2;
            j4 = shape.j4;
        } else {
            oblateness = 0.0;
            j2 = 0.0;
            j4 = 0.0;
        }
    }

    ephemeris::AddBody build_command(const ephemeris::SystemSnapshot& snapshot) const {
        ephemeris::BodyDefinition bdef{};
        bdef.name = name;
        bdef.type = type;
        bdef.parent_idx = parent_idx;
        bdef.mass_msun = get_mass_msun();
        bdef.radius_rsun = (type == "Star") ? star.preview.radius_rsun : (radius_km / SOLAR_RADIUS_KM);
        bdef.oblateness = (type == "Star") ? star.preview.oblateness : oblateness;
        bdef.j2 = j2;
        bdef.j4 = j4;
        bdef.semi_major_au = get_semi_major_au();
        bdef.eccentricity = eccentricity;
        bdef.inc_deg = inc_deg;
        bdef.omega_node_deg = omega_node_deg;
        bdef.arg_peri_deg = arg_peri_deg;
        bdef.mean_anom_deg = mean_anom_deg;
        bdef.rotation_period_hr = rotation_period_hr;
        bdef.tidally_locked = tidally_locked;
        bdef.color_rgb = (type == "Star") ? star.preview.color_rgb : color_rgb;
        bdef.color_pole_rgb = (type == "Star") ? star.preview.color_pole_rgb : color_rgb;
        bdef.is_star = (type == "Star");
        bdef.is_black_hole = (type == "Black Hole") || (type == "Star" && star.preview.is_black_hole);
        if (type == "Star") {
            bdef.temp_k = star.preview.temp_k;
            bdef.lum_eq = star.preview.lum_eq_lsun;
            bdef.lum_pole = star.preview.lum_pole_lsun;
            bdef.spectral_class = star.preview.spectral_class;
            bdef.stage = star.preview.stage_name;
            bdef.source_json["star_props"] = physics::StarCalculator::to_json(star.preview, star.params);
        }
        return {snapshot.generation(), bdef, frame == 1, axial_tilt_deg};
    }
};

struct DeleteBodyPrompt {
    bool active = false;
    int body_index = -1;
    std::string body_name;
    std::string fallback_parent_name;
};

struct ExportNBodyDraft {
    bool active = false;
    std::string system_name;
    bool switch_after_export = true;
    std::string error;

    void reset(std::string_view default_name) {
        active = true;
        system_name = std::string(default_name);
        switch_after_export = true;
        error.clear();
    }
};

struct EphemerisSetupDraft {
    bool active = false;
    std::string external_bsp_path;
    std::string inspected_external_path;
    ephemeris::BspInspectResult external_preview{};
    std::string error;

    void reset() {
        active = true;
        external_bsp_path.clear();
        inspected_external_path.clear();
        external_preview = {};
        error.clear();
    }
};

struct BspImportDraft {
    bool active = false;
    bool refresh_needed = true;
    std::vector<ephemeris::AvailableBspFile> available_files;
    int selected_idx = 0;
    std::string custom_path;
    std::string inspected_path;
    ephemeris::BspInspectResult preview{};
    std::string status_msg;
    std::string error;

    void reset() {
        active = true;
        refresh_needed = true;
        selected_idx = 0;
        custom_path.clear();
        inspected_path.clear();
        preview = {};
        status_msg.clear();
        error.clear();
    }
};

struct JumpTimelineDraft {
    bool active = false;
    bool is_utc = false;
    int64_t target_year = 2026;
    int target_month = 1;
    int target_day = 1;
    int target_hour = 0;
    int target_minute = 0;
    int target_second = 0;
    int resolution_choice = 0;
    std::string error;

    void reset_to_now(std::string_view epoch_utc, units::Years current_sim_time, bool default_utc = false) {
        active = true;
        is_utc = default_utc;
        auto dt = time::calculate_sim_date(epoch_utc, current_sim_time, !is_utc);
        target_year = dt.year;
        target_month = dt.month;
        target_day = dt.day;
        target_hour = dt.hour;
        target_minute = dt.minute;
        target_second = dt.second;
        resolution_choice = 0;
        error.clear();
    }
};

struct TimelineScrubState {
    bool active = false;
    std::shared_ptr<ephemeris::TimelineRecording> recording;
    int scrub_index = 0;
    bool is_playing = false;
    double playback_speed = 1.0;
    double accumulated_steps = 0.0;
    ephemeris::SystemSnapshot pre_timeline_snapshot;
    ephemeris::SimulationClock pre_timeline_clock;
    std::string export_status;
    float export_progress = 0.0f;
    bool export_in_progress = false;

    void clear() {
        active = false;
        recording.reset();
        scrub_index = 0;
        is_playing = false;
        playback_speed = 1.0;
        accumulated_steps = 0.0;
        export_status.clear();
        export_progress = 0.0f;
        export_in_progress = false;
    }
};

struct ProceduralRingDraft {
    int preset_idx = 0;
    int seed = 42;
    int band_count = 4;
    int gap_count = 3;
    float striation_freq = 160.0f;
    float contrast = 1.7f;
    float base_density = 2.2f;
    float core_boost = 2.5f;
    float inner_mult = 1.20f;
    float outer_mult = 2.30f;
    glm::vec3 tint{1.0f, 1.0f, 1.0f};
    float opacity = 1.0f;
};

struct RingEditState {
    int body_index = -1;
    int selected_layer = 0;
    std::string status_msg;
    std::string error_msg;
    ProceduralRingDraft procedural{};

    void sync_body(int idx, int layer_count) {
        if (body_index != idx) {
            body_index = idx;
            selected_layer = 0;
            status_msg.clear();
            error_msg.clear();
        }
        if (layer_count <= 0) {
            selected_layer = 0;
        } else if (selected_layer >= layer_count) {
            selected_layer = layer_count - 1;
        }
    }
};

struct WorkspaceState {
    int inspected_body = -1;
    bool edit_mode = false;
    bool inspect_barycenter = false;
    int inspector_frame = 0;
    BodyEditDraft body_draft{};
    RingEditState ring_edit{};
    AtmosphereEditDraft atmosphere_edit{};
    AddBodyDraft add_body{};
    DeleteBodyPrompt delete_prompt{};
    CreateSystemDraft create_system{};
    bool confirm_delete_system = false;
    std::string delete_system_name;
    ExportNBodyDraft export_nbody{};
    EphemerisSetupDraft ephem_setup{};
    BspImportDraft bsp_import{};
    bool show_ephem_download_modal = false;
    std::string settings_error;

    bool use_local_timezone = true;
    JumpTimelineDraft jump_timeline{};
    TimelineScrubState timeline_scrub{};

    bool has_selection() const noexcept { return inspected_body >= 0; }
    bool has_selection(const ephemeris::SystemSnapshot& snapshot) const noexcept {
        return inspected_body >= 0 && static_cast<size_t>(inspected_body) < snapshot.size();
    }
};

} // namespace sf::ui
