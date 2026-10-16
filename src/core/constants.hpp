#pragma once

#include <array>
#include <glm/glm.hpp>
#include <numbers>
#include <string_view>
#include <utility>

namespace sf {

inline constexpr double PI = std::numbers::pi_v<double>;
inline constexpr double TWO_PI = 2.0 * PI;
inline constexpr double DEG_TO_RAD = PI / 180.0;
inline constexpr double RAD_TO_DEG = 180.0 / PI;

inline constexpr double SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0;
inline constexpr double C_AU_YR = 63197.79;
inline constexpr double FOV_DEG = 45.0;
inline constexpr double G_CONST = 39.476926421373;

inline constexpr double OBLIQUITY = 23.439 * DEG_TO_RAD;

inline constexpr int ORBIT_RESOLUTION = 200;
inline constexpr int ORBIT_STRIDE_BYTES = 28;

inline constexpr double SOLAR_RADIUS_KM = 696340.0;
inline constexpr double AU_TO_KM = 149597870.7;
inline constexpr double KM_TO_AU = 1.0 / AU_TO_KM;
inline constexpr double SOLAR_RADII_TO_AU = SOLAR_RADIUS_KM / AU_TO_KM;

inline constexpr double LY_TO_AU = 63241.07708426963;
inline constexpr double AU_TO_LY = 1.0 / LY_TO_AU;
inline constexpr double LY_TO_KM = LY_TO_AU * AU_TO_KM;
inline constexpr double DEFAULT_LY_THRESHOLD_AU = 0.1 * LY_TO_AU;
inline constexpr double MAX_FLIGHT_SPEED_AU_S = 100.0 * LY_TO_AU;

inline constexpr std::array<std::pair<int, int>, 3> SKY_VIEW_RESOLUTIONS = {{
    {192, 108},
    {256, 256},
    {384, 216},
}};

inline constexpr std::array<std::string_view, 3> SKY_VIEW_RES_LABELS = {
    "Low (192x108)",
    "Medium (256x256)",
    "High (384x216)",
};

constexpr glm::dvec3 ecliptic_to_render(const glm::dvec3& ecl) noexcept {
    return {ecl.x, ecl.z, -ecl.y};
}

constexpr glm::dvec3 render_to_ecliptic(const glm::dvec3& rnd) noexcept {
    return {rnd.x, -rnd.z, rnd.y};
}

} // namespace sf
