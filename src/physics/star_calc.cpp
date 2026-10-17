#include "physics/star_calc.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>

namespace sf::physics {

namespace {

constexpr double PI = std::numbers::pi;

} // namespace

double StarCalculator::calc_lum(double r_rsun, double t_k) noexcept {
    if (r_rsun <= 0.0 || t_k <= 0.0) return 0.0;
    double tr = t_k / T_SUN_K;
    return (r_rsun * r_rsun) * (tr * tr * tr * tr);
}

double StarCalculator::calc_rad(double l_lsun, double t_k) noexcept {
    if (l_lsun <= 0.0 || t_k <= 0.0) return 0.0;
    double tr = t_k / T_SUN_K;
    return std::sqrt(l_lsun / (tr * tr * tr * tr));
}

double StarCalculator::calc_temp(double l_lsun, double r_rsun) noexcept {
    if (l_lsun <= 0.0 || r_rsun <= 0.0) return 0.0;
    return T_SUN_K * std::pow(l_lsun / (r_rsun * r_rsun), 0.25);
}

double StarCalculator::ms_mass_from_lum(double l_lsun) noexcept {
    if (l_lsun <= 0.0) return 0.1;
    if (l_lsun < 0.03) return std::pow(l_lsun / 0.23, 1.0 / 2.3);
    if (l_lsun < 16.0) return std::pow(l_lsun, 0.25);
    if (l_lsun < 1000000.0) return std::pow(l_lsun / 1.4, 1.0 / 3.5);
    return l_lsun / 32000.0;
}

double StarCalculator::ms_lum_from_mass(double m_msun) noexcept {
    if (m_msun < 0.43) return 0.23 * std::pow(m_msun, 2.3);
    if (m_msun < 2.0) return std::pow(m_msun, 4.0);
    if (m_msun < 55.0) return 2.0 * std::pow(m_msun, 3.5);
    return 32000.0 * m_msun;
}

std::tuple<double, double, double, std::string> StarCalculator::evolve_star(
    double initial_mass, double age_val, EvolutionPath evo_path, double metallicity) {
    double rad_mult = std::clamp(1.0 + 0.15 * metallicity, 0.6, 1.2);
    double lum_mult = std::clamp(1.0 - 0.2 * metallicity, 0.5, 2.0);

    double base_r = std::pow(initial_mass, 0.8) * rad_mult;
    double r_tams_mult = (initial_mass < 1.5) ? 1.50 : 2.00;
    double l_tams_mult = (initial_mass < 1.5) ? 2.00 : 2.50;
    double curve_r_ms = (initial_mass < 1.5) ? 2.0 : 1.5;
    double curve_l_ms = (initial_mass < 1.5) ? 1.9 : 1.6;

    double r_zams = base_r * 0.87;
    double r_tams = base_r * r_tams_mult;

    double base_l = ms_lum_from_mass(initial_mass) * lum_mult;
    double l_zams = base_l * 0.70;
    double l_tams = base_l * l_tams_mult;

    // Phase 0: Pre-Main Sequence
    if (age_val < 0.0) {
        double p = std::max(0.0, 1.0 + (age_val / 0.05));
        double r = r_zams * (1.0 + 9.0 * std::pow(1.0 - p, 2.0));
        double l = l_zams * (1.0 + 4.0 * std::pow(1.0 - p, 2.0));
        std::string phase = (initial_mass < 2.0) ? "T-Tauri Star" : "Herbig Ae/Be Star";
        return {initial_mass, r, l, phase};
    }

    // Phase 1: Main Sequence
    if (age_val <= 1.0) {
        double r = r_zams + (r_tams - r_zams) * std::pow(age_val, curve_r_ms);
        double l = l_zams + (l_tams - l_zams) * std::pow(age_val, curve_l_ms);
        return {initial_mass, r, l, ""};
    }

    // Phase 2: Giant / Post-MS
    if (age_val <= 1.1) {
        double progress = (age_val - 1.0) * 10.0;

        if ((initial_mass >= 25.0 && evo_path == EvolutionPath::Stripping) || initial_mass >= 45.0) {
            if (progress < 0.5) {
                double sub_prog = progress * 2.0;
                double r = r_tams + (60.0 * std::sqrt(initial_mass) - r_tams) * sub_prog;
                double l = l_tams * (1.5 + sub_prog * 0.5);
                double current_mass = initial_mass * (1.0 - 0.3 * sub_prog);
                return {current_mass, r, l, "Luminous Blue Variable (LBV)"};
            }
            double sub_prog = (progress - 0.5) * 2.0;
            double r_lbv = 60.0 * std::sqrt(initial_mass);
            double r = r_lbv - (r_lbv - std::max(1.5, 0.15 * initial_mass)) * std::pow(sub_prog, 0.3);
            double l = l_tams * 2.0;
            double current_mass = (initial_mass * 0.7) * (1.0 - 0.4 * sub_prog);
            return {current_mass, r, l, "Wolf-Rayet (WR)"};
        }

        double r_max = 200.0 * std::pow(initial_mass, 0.6) * rad_mult;
        double l_max = 2500.0 * std::pow(initial_mass, 1.5) * lum_mult;
        if (initial_mass >= 8.0) {
            r_max = std::max(r_max, r_tams * (40.0 * std::sqrt(initial_mass)));
            l_max = std::max(l_max, l_tams * (1.2 + (initial_mass * 0.015)));
        }

        double mass_lost = 0.0;
        double curve_r = 0.0;
        double curve_l = 0.0;
        if (initial_mass < 2.0) {
            mass_lost = 0.45 * progress;
            curve_r = std::pow(progress, 3.0);
            curve_l = std::pow(progress, 4.0);
        } else if (initial_mass < 8.0) {
            mass_lost = 0.6 * progress;
            curve_r = std::pow(progress, 2.5);
            curve_l = std::pow(progress, 3.0);
        } else {
            mass_lost = 0.4 * progress;
            curve_r = std::pow(progress, 2.0);
            curve_l = std::pow(progress, 1.5);
        }

        double r = r_tams + (r_max - r_tams) * curve_r;
        double l = l_tams + (l_max - l_tams) * curve_l;
        double current_mass = std::max(0.1, initial_mass * (1.0 - mass_lost));
        return {current_mass, r, l, ""};
    }

    // Phase 3: Remnant
    double progress = (age_val - 1.1) * 10.0;
    if (initial_mass < 8.0) {
        double rem_mass = std::min(1.4, initial_mass * 0.5);
        return {rem_mass, 0.01, 1.0 * std::pow(0.0001, progress), ""};
    }
    if (initial_mass < 25.0) {
        return {1.4, 1.4e-5, 1e-5, ""};
    }
    double rem_mass = initial_mass * 0.3;
    return {rem_mass, (3.0 * rem_mass) / 695700.0, 0.0, ""};
}

StarForgeResult StarCalculator::forge(const StarParameters& params) {
    StarForgeResult out{};
    double mass = params.mass_msun <= 0.0 ? 1.0 : params.mass_msun;
    double age_pct = params.age_pct;
    double metallicity = params.metallicity_dex;
    double rot_frac = std::clamp(params.rot_frac, 0.0, 1.0);
    double inclination = params.inclination_deg;

    double rad_mult = std::clamp(1.0 + 0.15 * metallicity, 0.6, 1.2);
    double lum_mult = std::clamp(1.0 - 0.2 * metallicity, 0.5, 2.0);

    double current_mass = mass;
    double radius = 1.0;
    double lum = 1.0;
    double temp = 5778.0;
    std::string phase_override;

    if (params.mode == StarCreationMode::Evolution) {
        auto evolved = evolve_star(mass, age_pct, params.evo_path, metallicity);
        current_mass = std::get<0>(evolved);
        radius = std::get<1>(evolved);
        lum = std::get<2>(evolved);
        phase_override = std::get<3>(evolved);
        temp = calc_temp(lum, radius);
    } else {
        radius = params.radius_rsun;
        temp = params.temp_k;
        lum = params.lum_lsun;

        if (params.locked_rad && params.locked_temp && !params.locked_lum) {
            lum = calc_lum(radius, temp);
        } else if (params.locked_lum && params.locked_temp && !params.locked_rad) {
            radius = calc_rad(lum, temp);
        } else if (params.locked_lum && params.locked_rad && !params.locked_temp) {
            temp = calc_temp(lum, radius);
        }

        current_mass = ms_mass_from_lum(lum / lum_mult);
        double base_r = std::pow(current_mass, 0.8) * rad_mult;
        double r_tams_mult = (current_mass < 1.5) ? 1.50 : 2.00;
        double r_zams = base_r * 0.87;
        double r_tams = base_r * r_tams_mult;

        if (radius <= r_zams) {
            age_pct = 0.0;
        } else if (radius <= r_tams) {
            age_pct = std::sqrt((radius - r_zams) / std::max(1e-6, r_tams - r_zams));
        } else {
            double r_max = 200.0 * std::pow(current_mass, 0.6) * rad_mult;
            if (current_mass >= 8.0) {
                r_max = std::max(r_max, r_tams * (40.0 * std::sqrt(current_mass)));
            }
            double exp_val = (current_mass < 2.0) ? 3.0 : ((current_mass < 8.0) ? 2.5 : 2.0);
            if (r_max > r_tams) {
                double progress = std::pow(std::max(0.0, (radius - r_tams) / (r_max - r_tams)), 1.0 / exp_val);
                age_pct = 1.0 + std::min(0.1, progress * 0.1);
            } else {
                age_pct = 1.1;
            }
        }
    }

    // Roche model rotation and Von Zeipel gravity darkening
    double f_oblateness = 1.0 + 0.22 * (rot_frac * rot_frac);
    double r_pole = radius / std::pow(f_oblateness, 2.0 / 3.0);
    double r_eq = radius * std::pow(f_oblateness, 1.0 / 3.0);

    constexpr double beta = 0.08;
    double g_ratio = ((r_pole / std::max(1e-6, r_eq)) * (r_pole / std::max(1e-6, r_eq))) *
                     std::max(0.01, 1.0 - rot_frac * rot_frac);
    double t_ratio = std::pow(g_ratio, beta);

    double t_pole = temp * (1.0 + 0.08 * rot_frac);
    double t_eq = t_pole * t_ratio;

    double inc_rad = inclination * (PI / 180.0);
    double cos_i = std::cos(inc_rad);
    double sin_i = std::sin(inc_rad);

    double r_proj_y = std::sqrt((r_pole * sin_i) * (r_pole * sin_i) + (r_eq * cos_i) * (r_eq * cos_i));
    double apparent_area_ratio = (r_eq * r_proj_y) / std::max(1e-12, radius * radius);

    double apparent_temp = t_pole * (1.0 - sin_i) + t_eq * sin_i;
    double apparent_lum = (temp > 0.0) ? (lum * apparent_area_ratio * std::pow(apparent_temp / temp, 4.0)) : 0.0;
    double apparent_radius = std::sqrt(r_eq * r_proj_y);

    double logg = 0.0;
    double density = 0.0;
    double escape_vel = 0.0;
    if (apparent_radius > 0.0 && current_mass > 0.0) {
        double r_m = apparent_radius * RSUN_M;
        double m_kg = current_mass * MSUN_KG;
        double g_cgs = ((G_SI * m_kg) / (r_m * r_m)) * 100.0; // m/s^2 to cm/s^2
        logg = std::log10(std::max(1.0, g_cgs));
        double vol_m3 = (4.0 / 3.0) * PI * (r_m * r_m * r_m);
        density = (m_kg / vol_m3) / 1000.0; // kg/m^3 to g/cm^3
        escape_vel = std::sqrt((2.0 * G_SI * m_kg) / r_m) / 1000.0; // km/s
    }

    bool is_bh = (apparent_lum == 0.0 && apparent_radius > 0.0 && params.mode == StarCreationMode::Evolution && mass >= 25.0);
    bool is_ns = (apparent_radius < 1e-4 && apparent_temp > 0.0);
    bool is_wd = (apparent_radius < 0.05 && current_mass <= 1.4 && !is_ns && !is_bh);

    std::string full_designation;
    std::string phase;

    if (!phase_override.empty()) {
        full_designation = phase_override;
        phase = "Envelope Stripping Phase";
    } else if (is_bh) {
        full_designation = "Black Hole";
        phase = "Singularity";
    } else if (is_ns) {
        full_designation = "Neutron Star";
        phase = "Degenerate Remnant";
    } else if (is_wd) {
        full_designation = "White Dwarf (D)";
        phase = "Degenerate Remnant";
    } else {
        struct SpectralBand { double min_t; char c; };
        static constexpr SpectralBand bands[] = {
            {30000.0, 'O'}, {10000.0, 'B'}, {7500.0, 'A'},
            {6000.0, 'F'}, {5200.0, 'G'}, {3700.0, 'K'},
            {2400.0, 'M'}, {1300.0, 'L'}, {550.0, 'T'}
        };
        char spectral_type = 'Y';
        int subclass = 0;
        for (size_t i = 0; i < std::size(bands); ++i) {
            if (apparent_temp >= bands[i].min_t) {
                spectral_type = bands[i].c;
                double max_t = (i == 0) ? 100000.0 : bands[i - 1].min_t;
                double frac = (max_t - apparent_temp) / std::max(1.0, max_t - bands[i].min_t);
                subclass = std::clamp(static_cast<int>(std::floor(frac * 10.0)), 0, 9);
                break;
            }
        }

        std::string lum_class;
        if (logg < 0.0) lum_class = "Ia-0";
        else if (logg < 1.5) lum_class = "Ia";
        else if (logg < 2.2) lum_class = "II";
        else if (logg < 3.2) lum_class = "III";
        else if (logg < 3.8) lum_class = "IV";
        else if (logg < 5.5) lum_class = "V";
        else lum_class = "VI";

        full_designation = std::format("{}{} {}", spectral_type, subclass, lum_class);

        std::string color_name;
        if (spectral_type == 'O') color_name = "Blue";
        else if (spectral_type == 'B') color_name = "Blue-White";
        else if (spectral_type == 'A') color_name = "White";
        else if (spectral_type == 'F') color_name = "Yellow-White";
        else if (spectral_type == 'G') color_name = "Yellow";
        else if (spectral_type == 'K') color_name = "Orange";
        else if (spectral_type == 'M') color_name = "Red";
        else color_name = "Brown";

        std::string size_name;
        if (lum_class == "Ia-0") size_name = "Hypergiant";
        else if (lum_class == "Ia") size_name = "Supergiant";
        else if (lum_class == "II") size_name = "Bright Giant";
        else if (lum_class == "III") size_name = "Giant";
        else if (lum_class == "IV") size_name = "Subgiant";
        else if (lum_class == "V") size_name = "Dwarf";
        else size_name = "Subdwarf";

        phase = std::format("{} {}", color_name, size_name);
    }

    glm::vec3 col_rgb = is_bh ? glm::vec3(0.0f) : temp_to_rgb(apparent_temp);
    glm::vec3 col_pole = is_bh ? glm::vec3(0.0f) : temp_to_rgb(t_pole);
    std::string col_hex = is_bh ? "#000000" : temp_to_hex(apparent_temp);

    double lum_eq = (apparent_temp > 0.0 && temp > 0.0) ? (lum * ((r_eq * r_pole) / (radius * radius)) * std::pow(t_eq / temp, 4.0)) : 0.0;
    double lum_pole = (apparent_temp > 0.0 && temp > 0.0) ? (lum * ((r_eq * r_eq) / (radius * radius)) * std::pow(t_pole / temp, 4.0)) : 0.0;

    double base_m_life = (params.mode == StarCreationMode::Evolution) ? mass : current_mass;
    double base_l_life = ms_lum_from_mass(base_m_life) * lum_mult;
    double lifespan_gyr = 10.0 * (base_m_life / std::max(1e-10, base_l_life));

    double v_eq_km_s = 0.0;
    if (r_eq > 0.0 && current_mass > 0.0) {
        double v_crit = std::sqrt((G_SI * (current_mass * MSUN_KG)) / (r_eq * RSUN_M)) / 1000.0;
        v_eq_km_s = v_crit * rot_frac;
    }

    double hz_inner = (apparent_lum > 0.0) ? std::sqrt(apparent_lum / 1.78) : 0.0;
    double hz_outer = (apparent_lum > 0.0) ? std::sqrt(apparent_lum / 0.32) : 0.0;

    bool is_pulsating = false;
    std::string pulsation_type;
    double pulsation_period_days = 0.0;
    std::string pulsation_period_str;

    if (apparent_lum > 0.0 && params.mode == StarCreationMode::Evolution && age_pct > 0.0 && apparent_radius > 0.0) {
        double log_l = std::log10(apparent_lum);
        double t_blue = 8500.0 - 500.0 * log_l;
        double t_red = 7000.0 - 500.0 * log_l;
        if (apparent_temp >= t_red && apparent_temp <= t_blue && !is_wd && !is_ns && !is_bh) {
            is_pulsating = true;
            if (apparent_lum < 100.0) pulsation_type = "Delta Scuti";
            else if (apparent_lum < 1000.0) pulsation_type = "RR Lyrae";
            else pulsation_type = "Classical Cepheid";

            pulsation_period_days = 0.04 * std::sqrt(std::pow(apparent_radius, 3.0) / current_mass);
            if (pulsation_period_days < 1.0) {
                pulsation_period_str = std::format("{:.1f} hours", pulsation_period_days * 24.0);
            } else {
                pulsation_period_str = std::format("{:.1f} days", pulsation_period_days);
            }
        }
    }

    out.mass_msun = current_mass;
    out.radius_rsun = apparent_radius;
    out.temp_k = apparent_temp;
    out.lum_lsun = apparent_lum;
    out.log_g = logg;
    out.density_g_cm3 = density;
    out.escape_vel_km_s = escape_vel;
    out.spectral_class = full_designation;
    out.stage_name = phase;
    out.lifespan_gyr = lifespan_gyr;
    out.age_gyr = lifespan_gyr * std::min(1.2, age_pct);
    out.age_pct = age_pct * 100.0;
    out.v_eq_km_s = v_eq_km_s;
    out.r_eq_rsun = r_eq;
    out.r_pole_rsun = r_pole;
    out.oblateness = (r_eq > 0.0) ? std::max(0.0, 1.0 - (r_pole / r_eq)) : 0.0;
    out.t_eq_k = t_eq;
    out.t_pole_k = t_pole;
    out.lum_eq_lsun = lum_eq;
    out.lum_pole_lsun = lum_pole;
    out.color_rgb = col_rgb;
    out.color_pole_rgb = col_pole;
    out.color_hex = col_hex;
    out.is_pulsating = is_pulsating;
    out.pulsation_type = pulsation_type;
    out.pulsation_period_days = pulsation_period_days;
    out.pulsation_period_str = pulsation_period_str;
    out.hz_inner_au = hz_inner;
    out.hz_outer_au = hz_outer;
    out.is_black_hole = is_bh;
    out.is_neutron_star = is_ns;
    out.is_white_dwarf = is_wd;
    out.is_wolf_rayet = (phase_override == "Wolf-Rayet (WR)");

    return out;
}

glm::vec3 StarCalculator::temp_to_rgb(double temp_k) noexcept {
    double t = temp_k / 100.0;
    double r_c = (t <= 66.0) ? 255.0 : std::clamp(329.69 * std::pow(t - 60.0, -0.133), 0.0, 255.0);
    double g_c = (t <= 66.0) ? std::clamp(99.47 * std::log(std::max(1.0, t)) - 161.1, 0.0, 255.0)
                             : std::clamp(288.1 * std::pow(t - 60.0, -0.075), 0.0, 255.0);
    double b_c = (t >= 66.0) ? 255.0 : ((t <= 19.0) ? 0.0 : std::clamp(138.5 * std::log(std::max(1.0, t - 10.0)) - 305.0, 0.0, 255.0));

    return glm::vec3(static_cast<float>(r_c / 255.0), static_cast<float>(g_c / 255.0), static_cast<float>(b_c / 255.0));
}

std::string StarCalculator::temp_to_hex(double temp_k) {
    glm::vec3 rgb = temp_to_rgb(temp_k);
    return rgb_to_hex(rgb);
}

glm::vec3 StarCalculator::hex_to_rgb(std::string_view hex) noexcept {
    if (hex.starts_with('#')) hex.remove_prefix(1);
    if (hex.size() < 6) return glm::vec3(1.0f);
    unsigned int r = 255, g = 255, b = 255;
    std::sscanf(hex.data(), "%02x%02x%02x", &r, &g, &b);
    return glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

std::string StarCalculator::rgb_to_hex(const glm::vec3& rgb) {
    int r = std::clamp(static_cast<int>(std::round(rgb.r * 255.0f)), 0, 255);
    int g = std::clamp(static_cast<int>(std::round(rgb.g * 255.0f)), 0, 255);
    int b = std::clamp(static_cast<int>(std::round(rgb.b * 255.0f)), 0, 255);
    return std::format("#{:02x}{:02x}{:02x}", r, g, b);
}

nlohmann::json StarCalculator::to_json(const StarForgeResult& res, const StarParameters& params) {
    nlohmann::json j;
    j["mode"] = (params.mode == StarCreationMode::Evolution) ? "evolution" : "surface";
    j["evo_path"] = (params.evo_path == EvolutionPath::Stripping) ? "stripping" : "standard";
    j["mass"] = res.mass_msun;
    j["metallicity"] = params.metallicity_dex;
    j["rot_frac"] = params.rot_frac;
    j["inclination"] = params.inclination_deg;
    j["age_pct"] = res.age_pct / 100.0;
    j["age"] = res.age_gyr;
    j["radius"] = res.radius_rsun;
    j["temp"] = res.temp_k;
    j["lum"] = res.lum_lsun;
    j["locked_rad"] = params.locked_rad;
    j["locked_temp"] = params.locked_temp;
    j["locked_lum"] = params.locked_lum;
    j["class"] = res.spectral_class;
    j["stage"] = res.stage_name;
    j["r_eq"] = res.r_eq_rsun;
    j["r_pole"] = res.r_pole_rsun;
    j["t_eq"] = res.t_eq_k;
    j["t_pole"] = res.t_pole_k;
    j["lum_eq"] = res.lum_eq_lsun;
    j["lum_pole"] = res.lum_pole_lsun;
    j["v_eq"] = res.v_eq_km_s;
    return j;
}

StarParameters StarCalculator::from_json(const nlohmann::json& j) {
    StarParameters p{};
    if (j.value("mode", "evolution") == "surface") {
        p.mode = StarCreationMode::Surface;
    } else {
        p.mode = StarCreationMode::Evolution;
    }
    if (j.value("evo_path", "standard") == "stripping") {
        p.evo_path = EvolutionPath::Stripping;
    } else {
        p.evo_path = EvolutionPath::Standard;
    }
    p.mass_msun = j.value("mass", 1.0);
    p.metallicity_dex = j.value("metallicity", 0.0);
    p.age_pct = j.value("age_pct", 0.46);
    p.rot_frac = j.value("rot_frac", 0.0);
    p.inclination_deg = j.value("inclination", 0.0);
    p.locked_rad = j.value("locked_rad", true);
    p.locked_temp = j.value("locked_temp", true);
    p.locked_lum = j.value("locked_lum", false);
    p.radius_rsun = j.value("radius", 1.0);
    p.temp_k = j.value("temp", 5778.0);
    p.lum_lsun = j.value("lum", 1.0);
    return p;
}

} // namespace sf::physics
