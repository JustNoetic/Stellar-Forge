#include "physics/star_calc.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace sf::physics;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_analytic_formulas() {
    double l = StarCalculator::calc_lum(1.0, 5778.0);
    check(std::abs(l - 1.0) < 1e-4, "Sun luminosity should be ~1.0");

    double r = StarCalculator::calc_rad(1.0, 5778.0);
    check(std::abs(r - 1.0) < 1e-4, "Sun radius should be ~1.0");

    double t = StarCalculator::calc_temp(1.0, 1.0);
    check(std::abs(t - 5778.0) < 1e-2, "Sun temperature should be ~5778.0");

    double l4 = StarCalculator::calc_lum(2.0, 5778.0);
    check(std::abs(l4 - 4.0) < 1e-4, "Doubling radius quadruples luminosity");
}

void test_mass_luminosity_relation() {
    double l_low = StarCalculator::ms_lum_from_mass(0.2);
    check(l_low > 0.0 && l_low < 0.1, "Low mass star has very low luminosity");

    double l_solar = StarCalculator::ms_lum_from_mass(1.0);
    check(std::abs(l_solar - 1.0) < 1e-4, "Solar mass luminosity should be 1.0");

    double l_high = StarCalculator::ms_lum_from_mass(10.0);
    check(l_high > 1000.0, "10 Msun star has high luminosity");

    double m_inv = StarCalculator::ms_mass_from_lum(1.0);
    check(std::abs(m_inv - 1.0) < 1e-2, "1 Lsun yields ~1 Msun");
}

void test_solar_evolution() {
    StarParameters p{};
    p.mode = StarCreationMode::Evolution;
    p.mass_msun = 1.0;
    p.metallicity_dex = 0.0;
    p.age_pct = 0.46; // Present-day Sun
    p.rot_frac = 0.0;

    auto res = StarCalculator::forge(p);
    check(std::abs(res.mass_msun - 1.0) < 1e-3, "Mass preserved");
    check(std::abs(res.radius_rsun - 1.0) < 0.15, "Sun radius within reasonable MS bounds");
    check(std::abs(res.temp_k - 5778.0) < 300.0, "Sun temperature close to 5778 K");
    check(res.spectral_class.starts_with("G"), "Sun spectral class is G");
    check(res.spectral_class.find("V") != std::string::npos, "Sun luminosity class is V (Dwarf)");
    check(res.stage_name == "Yellow Dwarf", "Sun stage name is Yellow Dwarf");
    check(std::abs(res.lifespan_gyr - 10.0) < 0.1, "Solar lifespan is ~10 Gyr");
    check(res.hz_inner_au > 0.6 && res.hz_inner_au < 0.9, "Habitable zone inner boundary in expected range");
    check(res.hz_outer_au > 1.4 && res.hz_outer_au < 2.0, "Habitable zone outer boundary in expected range");
}

void test_pre_main_sequence() {
    StarParameters p{};
    p.mode = StarCreationMode::Evolution;
    p.mass_msun = 1.0;
    p.age_pct = -0.03; // Protostar

    auto res = StarCalculator::forge(p);
    check(res.stage_name.find("T-Tauri") != std::string::npos || res.spectral_class.find("T-Tauri") != std::string::npos,
          "Sub-solar pre-MS star classified as T-Tauri");
    check(res.radius_rsun > 1.5, "Protostar is swollen compared to ZAMS");
}

void test_massive_star_stripping() {
    StarParameters p{};
    p.mode = StarCreationMode::Evolution;
    p.mass_msun = 35.0;
    p.evo_path = EvolutionPath::Stripping;
    p.age_pct = 1.10; // Post-MS Wolf-Rayet bare core phase

    auto res = StarCalculator::forge(p);
    check(res.is_wolf_rayet, "Stripped massive star recognized as Wolf-Rayet");
    check(res.spectral_class.find("Wolf-Rayet") != std::string::npos, "Wolf-Rayet spectral designation");
    check(res.radius_rsun < 10.0, "Stripped Wolf-Rayet envelope shrinks down to bare core");
    check(res.lum_lsun > 10000.0, "Wolf-Rayet retains high luminosity");
}

void test_remnants() {
    StarParameters p_wd{};
    p_wd.mode = StarCreationMode::Evolution;
    p_wd.mass_msun = 1.0;
    p_wd.age_pct = 1.15; // Remnant
    auto res_wd = StarCalculator::forge(p_wd);
    check(res_wd.is_white_dwarf, "Low mass star becomes White Dwarf");
    check(res_wd.radius_rsun < 0.05, "White dwarf radius is very small");

    StarParameters p_bh{};
    p_bh.mode = StarCreationMode::Evolution;
    p_bh.mass_msun = 30.0;
    p_bh.age_pct = 1.15; // Remnant
    auto res_bh = StarCalculator::forge(p_bh);
    check(res_bh.is_black_hole, "Massive star becomes Black Hole");
    check(res_bh.lum_lsun == 0.0, "Black hole has zero luminosity");
    check(res_bh.color_hex == "#000000", "Black hole color is black");
}

void test_surface_physics_mode() {
    StarParameters p{};
    p.mode = StarCreationMode::Surface;
    p.locked_rad = true;
    p.locked_temp = true;
    p.locked_lum = false;
    p.radius_rsun = 2.0;
    p.temp_k = 10000.0; // B-star / A-star territory

    auto res = StarCalculator::forge(p);
    check(std::abs(res.radius_rsun - 2.0) < 1e-4, "Surface radius preserved");
    check(res.lum_lsun > 30.0, "Computed luminosity matches Stefan-Boltzmann");
    check(res.spectral_class.starts_with("A") || res.spectral_class.starts_with("B"), "A/B spectral class");
}

void test_roche_rotation() {
    StarParameters p{};
    p.mode = StarCreationMode::Evolution;
    p.mass_msun = 1.5;
    p.age_pct = 0.2;
    p.rot_frac = 0.8; // Rapid rotator

    auto res = StarCalculator::forge(p);
    check(res.oblateness > 0.05, "Rapid rotator has noticeable oblateness");
    check(res.r_eq_rsun > res.r_pole_rsun, "Equatorial radius exceeds polar radius");
    check(res.t_pole_k > res.t_eq_k, "Von Zeipel gravity darkening makes poles hotter than equator");
    check(res.v_eq_km_s > 100.0, "Equatorial speed computed");
}

void test_color_and_serialization() {
    glm::vec3 c_blue = StarCalculator::temp_to_rgb(25000.0);
    check(c_blue.b > c_blue.r, "Hot star is bluish");

    glm::vec3 c_red = StarCalculator::temp_to_rgb(3000.0);
    check(c_red.r > c_red.b, "Cool star is reddish");

    StarParameters p{};
    p.mass_msun = 1.2;
    p.age_pct = 0.5;
    auto res = StarCalculator::forge(p);
    auto j = StarCalculator::to_json(res, p);

    check(j.contains("mass") && j["mass"].get<double>() > 0.0, "JSON contains mass");
    check(j.contains("temp") && j["temp"].get<double>() > 0.0, "JSON contains temp");
    check(j.contains("class"), "JSON contains class");

    auto p_back = StarCalculator::from_json(j);
    check(std::abs(p_back.mass_msun - p.mass_msun) < 1e-4, "Deserialized mass matches");
    check(p_back.mode == p.mode, "Deserialized mode matches");
}

int main() {
    try {
        test_analytic_formulas();
        test_mass_luminosity_relation();
        test_solar_evolution();
        test_pre_main_sequence();
        test_massive_star_stripping();
        test_remnants();
        test_surface_physics_mode();
        test_roche_rotation();
        test_color_and_serialization();
        std::cout << "All StarCalculator contracts passed cleanly.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}
