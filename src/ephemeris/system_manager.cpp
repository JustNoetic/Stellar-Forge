#include "ephemeris/system_manager.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/math_utils.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <unordered_set>

namespace sf::ephemeris {

namespace {

void topo_visit(int idx, const nlohmann::json& raw_arr,
                const std::unordered_map<std::string, int>& raw_map,
                std::unordered_set<int>& visited,
                std::vector<int>& order) {
    if (visited.contains(idx)) {
        return;
    }
    visited.insert(idx);
    const auto& item = raw_arr[idx];
    if (item.contains("parentId") && item["parentId"].is_string()) {
        auto it = raw_map.find(item["parentId"].get<std::string>());
        if (it != raw_map.end()) {
            topo_visit(it->second, raw_arr, raw_map, visited, order);
        }
    }
    order.push_back(idx);
}

} // namespace

int SystemState::find_body(std::string_view body_name) const {
    auto it = name_to_idx.find(std::string(body_name));
    return it != name_to_idx.end() ? it->second : -1;
}

glm::vec3 hex_to_rgb(std::string_view hex) noexcept {
    if (!hex.empty() && hex.front() == '#') {
        hex.remove_prefix(1);
    }
    if (hex.size() != 6) {
        return {1.0f, 1.0f, 1.0f};
    }
    unsigned int r = 255, g = 255, b = 255;
    std::string s(hex);
    if (std::sscanf(s.c_str(), "%02x%02x%02x", &r, &g, &b) == 3) {
        return {r / 255.0f, g / 255.0f, b / 255.0f};
    }
    return {1.0f, 1.0f, 1.0f};
}

std::string rgb_to_hex(const glm::vec3& rgb) {
    int r = std::clamp(static_cast<int>(rgb.r * 255.0f), 0, 255);
    int g = std::clamp(static_cast<int>(rgb.g * 255.0f), 0, 255);
    int b = std::clamp(static_cast<int>(rgb.b * 255.0f), 0, 255);
    return std::format("#{:02x}{:02x}{:02x}", r, g, b);
}

glm::vec3 temperature_to_rgb(double temp_k) noexcept {
    double temp = std::clamp(temp_k, 1000.0, 40000.0) / 100.0;
    double r = (temp <= 66.0)
        ? 1.0
        : (329.698727446 * std::pow(std::max(temp - 60.0, 1e-6), -0.1332047592) / 255.0);
    double g = (temp <= 66.0)
        ? ((99.4708025861 * std::log(std::max(temp, 1e-6)) - 161.1195681661) / 255.0)
        : (288.1221695283 * std::pow(std::max(temp - 60.0, 1e-6), -0.0755148492) / 255.0);
    double b = 0.0;
    if (temp >= 66.0) {
        b = 1.0;
    } else if (temp > 19.0) {
        b = (138.5177312231 * std::log(std::max(temp - 10.0, 1e-6)) - 305.0447927307) / 255.0;
    }
    return {
        static_cast<float>(std::clamp(r, 0.0, 1.0)),
        static_cast<float>(std::clamp(g, 0.0, 1.0)),
        static_cast<float>(std::clamp(b, 0.0, 1.0)),
    };
}

std::string get_spectral_class(double temp_k, std::string_view lum_class) {
    struct Band { double upper; const char* letter; double lower; };
    static constexpr Band bands[] = {
        {50000.0, "O", 30000.0},
        {30000.0, "B", 10000.0},
        {10000.0, "A", 7500.0},
        {7500.0,  "F", 6000.0},
        {6000.0,  "G", 5200.0},
        {5200.0,  "K", 3700.0},
        {3700.0,  "M", 2400.0},
        {2400.0,  "L", 1300.0},
        {1300.0,  "T", 500.0},
        {500.0,   "Y", 200.0},
    };
    for (const auto& b : bands) {
        if (temp_k >= b.lower) {
            int sub = 0;
            if (b.upper > b.lower) {
                sub = std::clamp(static_cast<int>(((b.upper - temp_k) / (b.upper - b.lower)) * 10.0), 0, 9);
            }
            return std::format("{}{}{}", b.letter, sub, lum_class);
        }
    }
    return std::format("Y9{}", lum_class);
}

DerivedStarProperties derive_star_properties(double mass, double metallicity, double age) {
    double l_zams = 1.0;
    if (mass < 0.43) l_zams = 0.23 * std::pow(mass, 2.3);
    else if (mass < 2.0) l_zams = std::pow(mass, 4.0);
    else if (mass < 55.0) l_zams = 1.4 * std::pow(mass, 3.5);
    else l_zams = 32000.0 * mass;

    double r_zams = (mass < 1.0) ? std::pow(mass, 0.8) : std::pow(mass, 0.57);
    double t_ms = (l_zams > 0.0) ? ((mass / l_zams) * 10.0) : 1e6;
    double progress = (t_ms > 0.0) ? std::min(age / t_ms, 1.0) : 1.0;

    DerivedStarProperties out{};
    out.ms_lifetime_gyr = t_ms;
    out.progress_pct = progress * 100.0;

    if (mass < 0.08) {
        out.stage = "Brown Dwarf";
        out.luminosity_class = "";
        out.luminosity_lsun = l_zams * std::max(0.01, 1.0 - 0.3 * std::min(age / 10.0, 1.0));
        out.radius_rsun = r_zams;
    } else if (age < t_ms * 0.9) {
        out.stage = "Main Sequence";
        out.luminosity_class = "V";
        out.luminosity_lsun = l_zams * (1.0 + 0.4 * progress);
        out.radius_rsun = r_zams * (1.0 + 0.15 * progress);
    } else if (age < t_ms) {
        out.stage = "Subgiant";
        out.luminosity_class = "IV";
        double sub = (age - t_ms * 0.9) / (t_ms * 0.1);
        out.luminosity_lsun = l_zams * 1.4 * (1.0 + 2.0 * sub);
        out.radius_rsun = r_zams * 1.15 * (1.0 + 3.0 * sub);
    } else if (mass < 0.5) {
        out.stage = "Main Sequence";
        out.luminosity_class = "V";
        out.luminosity_lsun = l_zams;
        out.radius_rsun = r_zams;
    } else if (mass < 8.0) {
        out.stage = "Red Giant";
        out.luminosity_class = "III";
        double post = std::min((age - t_ms) / std::max(t_ms * 0.3, 0.01), 1.0);
        out.luminosity_lsun = l_zams * (50.0 + 950.0 * post);
        out.radius_rsun = r_zams * (10.0 + 140.0 * post);
    } else if (mass < 25.0) {
        out.stage = "Supergiant";
        out.luminosity_class = "Ib";
        out.luminosity_lsun = l_zams * 500.0;
        out.radius_rsun = r_zams * 200.0;
    } else {
        out.stage = "Supergiant";
        out.luminosity_class = "Ia";
        out.luminosity_lsun = l_zams * 1000.0;
        out.radius_rsun = r_zams * 300.0;
    }

    double temp = (out.radius_rsun > 0.0)
        ? (5778.0 * std::pow(out.luminosity_lsun, 0.25) / std::sqrt(out.radius_rsun))
        : 5778.0;
    temp = std::max(500.0, temp * (1.0 - 0.05 * metallicity));
    out.temperature_k = temp;
    out.spectral_class = get_spectral_class(temp, out.luminosity_class);
    return out;
}

std::vector<std::string> SystemManager::list_systems() const {
    std::vector<std::string> names = {std::string(SOLAR_SYSTEM_NAME)};
    std::filesystem::path sys_dir = sf::paths::external_path("data", "systems");
    std::error_code ec;
    if (std::filesystem::is_directory(sys_dir, ec)) {
        std::vector<std::string> extras;
        for (const auto& entry : std::filesystem::directory_iterator(sys_dir, ec)) {
            if (entry.is_directory() && std::filesystem::is_regular_file(entry.path() / "system.json", ec)) {
                std::string folder = entry.path().filename().string();
                if (folder != SOLAR_SYSTEM_NAME) {
                    extras.push_back(folder);
                }
            }
        }
        std::sort(extras.begin(), extras.end());
        names.insert(names.end(), extras.begin(), extras.end());
    }
    return names;
}

SystemState SystemManager::load_system(std::string_view name) const {
    std::filesystem::path path = sf::paths::external_path("data", "systems", name, "system.json");
    if (!std::filesystem::exists(path)) {
        path = sf::paths::external_path("data", "system.json");
    }

    auto raw_opt = sf::paths::read_json(path);
    if (!raw_opt || !raw_opt->is_array()) {
        sf::log::fatal("Failed to load system JSON at {}", path.string());
    }
    const auto& raw_arr = *raw_opt;

    std::unordered_map<std::string, int> raw_map;
    for (int i = 0; i < static_cast<int>(raw_arr.size()); ++i) {
        if (raw_arr[i].contains("name") && raw_arr[i]["name"].is_string()) {
            raw_map[raw_arr[i]["name"].get<std::string>()] = i;
        }
    }

    std::vector<int> sorted_order;
    sorted_order.reserve(raw_arr.size());
    std::unordered_set<int> visited;
    for (int i = 0; i < static_cast<int>(raw_arr.size()); ++i) {
        topo_visit(i, raw_arr, raw_map, visited, sorted_order);
    }

    SystemState state{};
    state.name = std::string(name);
    state.bodies.reserve(sorted_order.size());

    for (int old_idx : sorted_order) {
        const auto& j = raw_arr[old_idx];
        CelestialBody b{};
        b.raw_json = j;
        b.name = j.value("name", "Body");
        b.type = j.value("type", "Terrestrial");
        b.parent_id = j.value("parentId", "");
        b.is_star = (b.type == "Star");
        b.is_spacecraft = j.value("is_spacecraft", false) || (b.type == "Spacecraft");
        b.tidally_locked = j.value("tidally_locked", false);

        b.mass_msun = j.value("m", 0.0);
        b.radius_rsun = j.value("r", 1.0);
        b.oblateness = j.value("oblateness", 0.0);
        b.j2 = j.value("J2", 0.0);
        b.j4 = j.value("j4", 0.0);

        b.semi_major_au = j.value("a", 0.0);
        b.eccentricity = j.value("e", 0.0);
        b.inc_deg = j.value("inc", 0.0);
        b.omega_node_deg = j.value("Omega", 0.0);
        b.arg_peri_deg = j.value("omega", 0.0);
        b.mean_anom_deg = j.value("M", 0.0);

        b.pole_ra_deg = j.value("pole_ra", 0.0);
        b.pole_dec_deg = j.value("pole_dec", 90.0);
        b.rotation_period_hr = j.value("rotation_period", 0.0);
        b.w0_deg = j.value("W0", 0.0);

        std::string col_hex = j.value("color", "#ffffff");
        b.color_rgb = hex_to_rgb(col_hex);
        b.color_pole_rgb = hex_to_rgb(j.value("color_pole", col_hex));

        b.lum_eq = j.value("lum_eq", 0.0);
        b.lum_pole = j.value("lum_pole", 0.0);
        if (j.contains("star_props") && j["star_props"].is_object()) {
            const auto& sp = j["star_props"];
            b.spectral_class = sp.value("class", "");
            b.stage = sp.value("stage", "");
            b.temp_k = sp.value("temp", 5778.0);
            if (b.lum_eq <= 0.0) b.lum_eq = sp.value("lum_eq", sp.value("lum", 1.0));
            if (b.lum_pole <= 0.0) b.lum_pole = sp.value("lum_pole", sp.value("lum", b.lum_eq));
        } else if (b.is_star && b.lum_eq <= 0.0) {
            b.lum_eq = 1.0;
            b.lum_pole = 1.0;
        }

        if (b.type == "Black Hole" || b.stage == "Singularity" ||
            b.spectral_class.find("Black Hole") != std::string::npos) {
            b.is_black_hole = true;
            double rs_km = 2.95325008 * b.mass_msun;
            b.radius_rsun = rs_km / SOLAR_RADIUS_KM;
            b.radius_au = (2.5980762 * rs_km) / AU_TO_KM;
        } else {
            b.radius_au = b.radius_rsun * SOLAR_RADII_TO_AU;
        }

        if (j.contains("pole_ra") && j.contains("pole_dec")) {
            b.pole_ecl = sf::math::pole_to_ecliptic(b.pole_ra_deg, b.pole_dec_deg);
            b.pole_render = glm::normalize(sf::ecliptic_to_render(b.pole_ecl));
        }

        int new_idx = static_cast<int>(state.bodies.size());
        state.name_to_idx[b.name] = new_idx;
        if (!b.parent_id.empty()) {
            auto pit = state.name_to_idx.find(b.parent_id);
            if (pit != state.name_to_idx.end()) {
                b.parent_idx = pit->second;
            }
        }

        glm::dvec3 p_pos{0.0};
        glm::dvec3 p_vel{0.0};
        double p_mass = 0.0;
        if (b.parent_idx >= 0) {
            const auto& parent = state.bodies[b.parent_idx];
            p_pos = parent.pos_ecl;
            p_vel = parent.vel_ecl;
            p_mass = parent.mass_msun;
        }

        if (j.contains("sv") && j["sv"].is_object()) {
            const auto& sv = j["sv"];
            glm::dvec3 rel_p{sv.value("x", 0.0), sv.value("y", 0.0), sv.value("z", 0.0)};
            glm::dvec3 rel_v{sv.value("vx", 0.0), sv.value("vy", 0.0), sv.value("vz", 0.0)};
            b.pos_ecl = p_pos + rel_p;
            b.vel_ecl = p_vel + rel_v;
        } else if (b.parent_idx >= 0) {
            auto [rel_p, rel_v] = sf::math::get_cartesian_from_keplerian(
                p_mass, b.mass_msun, b.semi_major_au, b.eccentricity,
                b.inc_deg, b.omega_node_deg, b.arg_peri_deg, b.mean_anom_deg);
            if (j.value("orbitRef", "") == "equatorial") {
                const auto& parent = state.bodies[b.parent_idx];
                glm::dmat3 R = sf::math::build_equatorial_frame(parent.pole_ecl);
                rel_p = R * rel_p;
                rel_v = R * rel_v;
            }
            b.pos_ecl = p_pos + rel_p;
            b.vel_ecl = p_vel + rel_v;
        }

        if (b.is_star && state.star_idx == 0 && new_idx > 0 && !state.bodies[0].is_star) {
            state.star_idx = new_idx;
        }
        state.bodies.push_back(std::move(b));
    }

    double total_mass = 0.0;
    glm::dvec3 com_pos{0.0};
    glm::dvec3 com_vel{0.0};
    for (const auto& b : state.bodies) {
        total_mass += b.mass_msun;
        com_pos += b.mass_msun * b.pos_ecl;
        com_vel += b.mass_msun * b.vel_ecl;
    }
    if (total_mass > 0.0) {
        com_pos /= total_mass;
        com_vel /= total_mass;
        for (auto& b : state.bodies) {
            b.pos_ecl -= com_pos;
            b.vel_ecl -= com_vel;
        }
    }

    sync_render_coords(state, 0.0);
    sf::log::info("Loaded system '{}' with {} bodies", state.name, state.bodies.size());
    return state;
}

bool SystemManager::save_system(std::string_view name, const SystemState& state) const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& b : state.bodies) {
        arr.push_back(b.raw_json);
    }
    auto path = sf::paths::external_path("data", "systems", name, "system.json");
    return sf::paths::write_json(path, arr, 2);
}

void SystemManager::sync_render_coords(SystemState& state, double sim_time_yr) noexcept {
    double t_sec = sim_time_yr * SECONDS_PER_YEAR;
    for (auto& b : state.bodies) {
        b.pos_render = sf::ecliptic_to_render(b.pos_ecl);
        b.vel_render = sf::ecliptic_to_render(b.vel_ecl);

        double w0_rad = b.w0_deg * DEG_TO_RAD;
        if (b.tidally_locked && b.parent_idx >= 0) {
            glm::dvec3 rel = b.pos_render - state.bodies[b.parent_idx].pos_render;
            b.spin_angle_rad = std::atan2(rel.z, rel.x) + PI + w0_rad;
        } else if (std::abs(b.rotation_period_hr) > 1e-9) {
            double omega = TWO_PI / (b.rotation_period_hr * 3600.0);
            b.spin_angle_rad = std::fmod(w0_rad + omega * t_sec, TWO_PI);
        } else {
            b.spin_angle_rad = w0_rad;
        }
    }
}

void SystemManager::step_keplerian(SystemState& state, double dt_yr, double sim_time_yr) noexcept {
    if (std::abs(dt_yr) <= 0.0 || state.bodies.empty()) {
        sync_render_coords(state, sim_time_yr);
        return;
    }

    std::vector<glm::dvec3> old_pos(state.bodies.size());
    std::vector<glm::dvec3> old_vel(state.bodies.size());
    for (size_t i = 0; i < state.bodies.size(); ++i) {
        old_pos[i] = state.bodies[i].pos_ecl;
        old_vel[i] = state.bodies[i].vel_ecl;
    }

    for (size_t i = 0; i < state.bodies.size(); ++i) {
        auto& b = state.bodies[i];
        if (b.parent_idx >= 0) {
            int p = b.parent_idx;
            glm::dvec3 rel_p = old_pos[i] - old_pos[p];
            glm::dvec3 rel_v = old_vel[i] - old_vel[p];
            double mu = G_CONST * (state.bodies[p].mass_msun + b.mass_msun);
            auto [new_rel_p, new_rel_v] = sf::math::propagate_conic_state(rel_p, rel_v, mu, dt_yr);
            b.pos_ecl = state.bodies[p].pos_ecl + new_rel_p;
            b.vel_ecl = state.bodies[p].vel_ecl + new_rel_v;
        } else {
            b.pos_ecl += b.vel_ecl * dt_yr;
        }
    }

    double total_mass = 0.0;
    glm::dvec3 com_pos{0.0};
    glm::dvec3 com_vel{0.0};
    for (const auto& b : state.bodies) {
        total_mass += b.mass_msun;
        com_pos += b.mass_msun * b.pos_ecl;
        com_vel += b.mass_msun * b.vel_ecl;
    }
    if (total_mass > 0.0) {
        com_pos /= total_mass;
        com_vel /= total_mass;
        for (auto& b : state.bodies) {
            b.pos_ecl -= com_pos;
            b.vel_ecl -= com_vel;
        }
    }

    sync_render_coords(state, sim_time_yr);
}

} // namespace sf::ephemeris
