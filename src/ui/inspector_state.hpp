#pragma once

#include "ephemeris/simulation.hpp"
#include "core/math_utils.hpp"
#include <format>

namespace sf::ui {

struct BodyEditDraft {
    int body_index = -1;
    uint64_t system_generation = 0;
    uint64_t body_revision = 0;
    int reference_parent = -1;
    ephemeris::BodyPhysicalProperties values{};
    bool dirty = false;
    std::string error;

    void reload(const ephemeris::SystemSnapshot& snapshot, int index) {
        body_index = index;
        system_generation = snapshot.generation();
        body_revision = snapshot.body(index).revisions.definition;
        reference_parent = snapshot.reference_parent(index);
        values = ephemeris::BodyPhysicalProperties::from(snapshot.body(index).definition);
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
        error.clear();
        recompute_hydrostatic();
    }

    void randomize_angles() {
        omega_node_deg = std::round((static_cast<double>(std::rand()) / RAND_MAX * 360.0) * 1000.0) / 1000.0;
        arg_peri_deg = std::round((static_cast<double>(std::rand()) / RAND_MAX * 360.0) * 1000.0) / 1000.0;
        mean_anom_deg = std::round((static_cast<double>(std::rand()) / RAND_MAX * 360.0) * 1000.0) / 1000.0;
    }

    double get_mass_msun() const noexcept {
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
        bdef.radius_rsun = radius_km / SOLAR_RADIUS_KM;
        bdef.oblateness = oblateness;
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
        bdef.color_rgb = color_rgb;
        bdef.color_pole_rgb = color_rgb;
        bdef.is_star = (type == "Star");
        bdef.is_black_hole = (type == "Black Hole");
        return {snapshot.generation(), bdef, frame == 1, axial_tilt_deg};
    }
};

struct DeleteBodyPrompt {
    bool active = false;
    int body_index = -1;
    std::string body_name;
    std::string fallback_parent_name;
};

struct WorkspaceState {
    int inspected_body = -1;
    bool edit_mode = false;
    bool inspect_barycenter = false;
    int inspector_frame = 0;
    BodyEditDraft body_draft{};
    AddBodyDraft add_body{};
    DeleteBodyPrompt delete_prompt{};
    std::string settings_error;

    bool has_selection() const noexcept { return inspected_body >= 0; }
    bool has_selection(const ephemeris::SystemSnapshot& snapshot) const noexcept {
        return inspected_body >= 0 && static_cast<size_t>(inspected_body) < snapshot.size();
    }
};

}
