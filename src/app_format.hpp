#pragma once

#include "core/constants.hpp"

#include <format>
#include <string>

namespace sf {

inline std::string trim_trailing_zeros(std::string s) {
    auto dot = s.find('.');
    if (dot != std::string::npos) {
        auto end = s.find_first_not_of("0123456789", dot + 1);
        size_t num_end = (end != std::string::npos) ? end : s.size();
        size_t last = num_end - 1;
        while (last > dot && s[last] == '0') {
            --last;
        }
        if (s[last] == '.') {
            --last;
        }
        s.erase(last + 1, num_end - (last + 1));
    }
    return s;
}

inline std::string format_distance_au(double dist_au, double threshold_au = DEFAULT_LY_THRESHOLD_AU) {
    double dist_km = dist_au * AU_TO_KM;
    if (dist_au >= threshold_au) {
        return trim_trailing_zeros(std::format("{:.3f} ly", dist_au * AU_TO_LY));
    }
    if (dist_au >= 0.05) {
        return trim_trailing_zeros(std::format("{:.4f} AU", dist_au));
    }
    if (dist_km >= 1.0) {
        return trim_trailing_zeros(std::format("{:.2f} km", dist_km));
    }
    return trim_trailing_zeros(std::format("{:.1f} m", dist_km * 1000.0));
}

inline std::string format_altitude(double alt_km, double alt_au, double threshold_au = DEFAULT_LY_THRESHOLD_AU) {
    if (alt_au >= threshold_au) {
        return trim_trailing_zeros(std::format("{:.3f} ly", alt_au * AU_TO_LY));
    }
    if (alt_km > 1.49597e7) {
        return trim_trailing_zeros(std::format("{:.4f} AU", alt_au));
    }
    if (alt_km >= 1000.0) {
        return std::format("{:.0f} km", alt_km);
    }
    if (alt_km >= 1.0) {
        return std::format("{:.1f} km", alt_km);
    }
    return std::format("{:.0f} m", alt_km * 1000.0);
}

inline std::string format_speed_au_s(double speed_au_s) {
    double km_s = speed_au_s * AU_TO_KM;
    if (speed_au_s >= LY_TO_AU * 0.01) {
        return trim_trailing_zeros(std::format("{:.2f} ly/s", speed_au_s * AU_TO_LY));
    }
    if (speed_au_s >= 0.01) {
        return trim_trailing_zeros(std::format("{:.3f} AU/s", speed_au_s));
    }
    if (km_s >= 1.0) {
        return trim_trailing_zeros(std::format("{:.2f} km/s", km_s));
    }
    return trim_trailing_zeros(std::format("{:.1f} m/s", km_s * 1000.0));
}

} // namespace sf
