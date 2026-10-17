#pragma once

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <tuple>

namespace sf::physics {

enum class StarCreationMode { Evolution, Surface };
enum class EvolutionPath { Standard, Stripping };

struct StarParameters {
    StarCreationMode mode = StarCreationMode::Evolution;
    EvolutionPath evo_path = EvolutionPath::Standard;
    double mass_msun = 1.0;
    double metallicity_dex = 0.0;
    double age_pct = 0.46; // 0.0 = ZAMS, 1.0 = TAMS
    double rot_frac = 0.0; // 0.0 to 1.0 of critical breakup speed
    double inclination_deg = 0.0;

    bool locked_rad = true;
    bool locked_temp = true;
    bool locked_lum = false;
    double radius_rsun = 1.0;
    double temp_k = 5778.0;
    double lum_lsun = 1.0;
};

struct StarForgeResult {
    double mass_msun = 1.0;
    double radius_rsun = 1.0;
    double temp_k = 5778.0;
    double lum_lsun = 1.0;
    double log_g = 4.44;
    double density_g_cm3 = 1.41;
    double escape_vel_km_s = 617.5;

    std::string spectral_class = "G2 V";
    std::string stage_name = "Yellow Dwarf";

    double lifespan_gyr = 10.0;
    double age_gyr = 4.6;
    double age_pct = 46.0;

    double v_eq_km_s = 0.0;
    double r_eq_rsun = 1.0;
    double r_pole_rsun = 1.0;
    double oblateness = 0.0;
    double t_eq_k = 5778.0;
    double t_pole_k = 5778.0;
    double lum_eq_lsun = 1.0;
    double lum_pole_lsun = 1.0;

    glm::vec3 color_rgb{1.0f, 0.95f, 0.90f};
    glm::vec3 color_pole_rgb{1.0f, 0.95f, 0.90f};
    std::string color_hex = "#fff2e6";

    bool is_pulsating = false;
    std::string pulsation_type;
    double pulsation_period_days = 0.0;
    std::string pulsation_period_str;

    double hz_inner_au = 0.75;
    double hz_outer_au = 1.77;

    bool is_black_hole = false;
    bool is_neutron_star = false;
    bool is_white_dwarf = false;
    bool is_wolf_rayet = false;
};

class StarCalculator {
public:
    static constexpr double G_SI = 6.67430e-11;
    static constexpr double MSUN_KG = 1.98847e30;
    static constexpr double RSUN_M = 6.957e8;
    static constexpr double LSUN_W = 3.828e26;
    static constexpr double T_SUN_K = 5778.0;

    static double calc_lum(double r_rsun, double t_k) noexcept;
    static double calc_rad(double l_lsun, double t_k) noexcept;
    static double calc_temp(double l_lsun, double r_rsun) noexcept;

    static double ms_mass_from_lum(double l_lsun) noexcept;
    static double ms_lum_from_mass(double m_msun) noexcept;

    static std::tuple<double, double, double, std::string> evolve_star(
        double initial_mass, double age_val, EvolutionPath evo_path, double metallicity);

    static StarForgeResult forge(const StarParameters& params);

    static glm::vec3 temp_to_rgb(double temp_k) noexcept;
    static std::string temp_to_hex(double temp_k);
    static glm::vec3 hex_to_rgb(std::string_view hex) noexcept;
    static std::string rgb_to_hex(const glm::vec3& rgb);

    static nlohmann::json to_json(const StarForgeResult& res, const StarParameters& params);
    static StarParameters from_json(const nlohmann::json& j);
};

} // namespace sf::physics
