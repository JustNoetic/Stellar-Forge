#pragma once

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sf::ephemeris {

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

struct CelestialBody {
    std::string name;
    std::string type = "Terrestrial";
    std::string parent_id;
    std::string spectral_class;
    std::string stage;

    int parent_idx = -1;
    bool is_star = false;
    bool is_black_hole = false;
    bool is_spacecraft = false;
    bool tidally_locked = false;

    double mass_msun = 0.0;
    double radius_rsun = 1.0;
    double radius_au = 0.0;
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
    glm::dvec3 pole_render{0.0, 1.0, 0.0};

    glm::dvec3 pos_ecl{0.0};
    glm::dvec3 vel_ecl{0.0};
    glm::dvec3 pos_render{0.0};
    glm::dvec3 vel_render{0.0};

    double spin_angle_rad = 0.0;
    nlohmann::json raw_json;
};

struct SystemState {
    std::string name = "Solar System";
    std::vector<CelestialBody> bodies;
    std::unordered_map<std::string, int> name_to_idx;
    int star_idx = 0;

    int find_body(std::string_view body_name) const;
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
    SystemState load_system(std::string_view name) const;
    bool save_system(std::string_view name, const SystemState& state) const;

    static void sync_render_coords(SystemState& state, double sim_time_yr = 0.0) noexcept;
    static void step_keplerian(SystemState& state, double dt_yr, double sim_time_yr) noexcept;
};

} // namespace sf::ephemeris
