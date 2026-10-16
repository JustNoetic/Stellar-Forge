#pragma once

#include "core/coordinates.hpp"
#include "core/units.hpp"
#include <memory>
#include <span>
#include <cstdint>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sf::ephemeris {

enum class MotionModel { Keplerian, Spice, NBody };

struct DerivedStarProperties {
    double luminosity_lsun = 1.0;
    double radius_rsun = 1.0;
    double temperature_k = 5778.0;
    std::string spectral_class = "G2V";
    std::string stage = "Main Sequence";
    std::string luminosity_class = "V";
    double ms_lifetime_gyr = 10.0;
    double progress_pct = 46.0;
};

struct BodyDefinition {
    std::string name;
    std::string type = "Terrestrial";
    std::string parent_id;
    std::string spectral_class;
    std::string stage;

    int parent_idx = -1;
    int spice_id = 0;
    bool is_star = false;
    bool is_black_hole = false;
    bool is_spacecraft = false;
    bool tidally_locked = false;

    double mass_msun = 0.0;
    double radius_rsun = 1.0;
    double oblateness = 0.0;
    double j2 = 0.0;
    double j4 = 0.0;

    double semi_major_au = 0.0;
    double eccentricity = 0.0;
    double inc_deg = 0.0;
    double omega_node_deg = 0.0;
    double arg_peri_deg = 0.0;
    double mean_anom_deg = 0.0;

    double pole_ra_deg = 0.0;
    double pole_dec_deg = 90.0;
    double rotation_period_hr = 0.0;
    double w0_deg = 0.0;

    double lum_eq = 1.0;
    double lum_pole = 1.0;
    double temp_k = 5778.0;

    glm::vec3 color_rgb{1.0f, 1.0f, 1.0f};
    glm::vec3 color_pole_rgb{1.0f, 1.0f, 1.0f};

    glm::dvec3 pole_ecl{0.0, 0.0, 1.0};
    nlohmann::json source_json;

    double radius_au() const noexcept {
        double physical = radius_rsun * SOLAR_RADII_TO_AU;
        return is_black_hole ? 2.5980762 * physical : physical;
    }
};

struct SystemDefinition {
    std::string name = "Solar System";
    MotionModel motion_model = MotionModel::Keplerian;
    std::string epoch_utc;
    std::vector<BodyDefinition> bodies;
    std::unordered_map<std::string, int> name_to_idx;
    int star_idx = 0;

    int find_body(std::string_view body_name) const;
};

struct BodyMotion {
    EclipticPosition position{};
    EclipticVelocity velocity{};
};

struct BodyRenderState {
    RenderPosition position{};
    RenderVelocity velocity{};
    glm::dvec3 pole{0.0, 1.0, 0.0};
    double spin_angle_rad = 0.0;
};

struct BodyRevisions {
    uint64_t definition = 1;
    uint64_t dynamics = 1;
    uint64_t geometry = 1;
    uint64_t lighting = 1;
};

struct BodyView {
    const BodyDefinition& definition;
    const BodyMotion& motion;
    const BodyRenderState& render;
    const BodyRevisions& revisions;
};

struct ReferenceHierarchy {
    std::vector<int> parents;
    std::vector<std::vector<int>> children;
    std::vector<int> roots;
};

struct LoadedSystem {
    SystemDefinition definition;
    std::vector<BodyMotion> motions;
};

class Simulation;
struct JacobiHierarchy;

class SystemSnapshot {
public:
    const SystemDefinition& definition() const noexcept { return *definition_; }
    size_t size() const noexcept { return motions_.size(); }
    bool empty() const noexcept { return motions_.empty(); }
    const std::string& name() const noexcept { return definition_->name; }
    int star_index() const noexcept { return definition_->star_idx; }
    int find_body(std::string_view name) const { return definition_->find_body(name); }
    units::Years time() const noexcept { return time_; }
    uint64_t generation() const noexcept { return generation_; }
    uint64_t state_revision() const noexcept { return state_revision_; }
    uint64_t definition_revision() const noexcept { return definition_revision_; }
    std::span<const BodyMotion> motions() const noexcept { return motions_; }
    const JacobiHierarchy& jacobi_hierarchy() const noexcept { return *jacobi_; }
    const JacobiHierarchy& orbit_hierarchy() const noexcept { return reference_jacobi_ ? *reference_jacobi_ : *jacobi_; }
    const ReferenceHierarchy& reference_hierarchy() const noexcept { return *reference_; }
    int reference_parent(size_t index) const { return reference_->parents.at(index); }
    uint64_t hierarchy_revision() const noexcept { return hierarchy_revision_; }
    BodyView body(size_t index) const {
        return {definition_->bodies.at(index), motions_.at(index), render_.at(index), revisions_.at(index)};
    }

private:
    friend class Simulation;
    std::shared_ptr<const SystemDefinition> definition_ = std::make_shared<const SystemDefinition>();
    std::shared_ptr<const JacobiHierarchy> jacobi_;
    std::shared_ptr<const JacobiHierarchy> reference_jacobi_;
    std::shared_ptr<const ReferenceHierarchy> reference_ = std::make_shared<const ReferenceHierarchy>();
    std::vector<BodyMotion> motions_;
    std::vector<BodyRenderState> render_;
    std::vector<BodyRevisions> revisions_;
    units::Years time_{};
    uint64_t generation_ = 0;
    uint64_t state_revision_ = 0;
    uint64_t definition_revision_ = 0;
    uint64_t hierarchy_revision_ = 0;
};

glm::vec3 hex_to_rgb(std::string_view hex) noexcept;
std::string rgb_to_hex(const glm::vec3& rgb);
glm::vec3 temperature_to_rgb(double temp_k) noexcept;
std::string get_spectral_class(double temp_k, std::string_view lum_class = "V");
DerivedStarProperties derive_star_properties(double mass_msun, double metallicity, double age_gyr);

class SystemManager {
public:
    static constexpr std::string_view SOLAR_SYSTEM_NAME = "Solar System";

    std::vector<std::string> list_systems() const;
    LoadedSystem load_system(std::string_view name) const;
    bool save_system(std::string_view name, const SystemDefinition& definition) const;
    static LoadedSystem parse_system(const nlohmann::json& source, std::string_view name);
    static nlohmann::json serialize_definition(const SystemDefinition& definition);

};

} // namespace sf::ephemeris
