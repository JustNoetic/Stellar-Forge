#include "core/time_utils.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>

namespace sf::time {

DateTimeUTC parse_iso_or_default(std::string_view iso_str, DateTimeUTC fallback) {
    if (iso_str.empty()) return fallback;
    std::string s(iso_str);
    int y = 0, m = 0, d = 0, h = 0, mn = 0, sec = 0;
    if (std::sscanf(s.c_str(), "%d-%d-%d%*c%d:%d:%d", &y, &m, &d, &h, &mn, &sec) == 6) {
        if (m >= 1 && m <= 12 && d >= 1 && d <= 31 && h >= 0 && h <= 23 && mn >= 0 && mn <= 59 && sec >= 0 && sec <= 60) {
            return {y, m, d, h, mn, sec, "UTC"};
        }
    } else if (std::sscanf(s.c_str(), "%d-%d-%d", &y, &m, &d) == 3) {
        if (m >= 1 && m <= 12 && d >= 1 && d <= 31) {
            return {y, m, d, 0, 0, 0, "UTC"};
        }
    }
    return fallback;
}

DateTimeUTC calculate_sim_date(std::string_view epoch_utc, units::Years sim_time) {
    auto base = parse_iso_or_default(epoch_utc);
    std::chrono::year_month_day base_ymd{std::chrono::year{base.year},
                                         std::chrono::month{static_cast<unsigned>(base.month)},
                                         std::chrono::day{static_cast<unsigned>(base.day)}};
    if (!base_ymd.ok()) {
        base = {2026, 1, 1, 12, 0, 0, "UTC"};
        base_ymd = std::chrono::year_month_day{std::chrono::year{2026}, std::chrono::month{1}, std::chrono::day{1}};
    }
    std::chrono::sys_days base_days{base_ymd};
    int64_t base_sec = std::chrono::duration_cast<std::chrono::seconds>(base_days.time_since_epoch()).count() +
                       base.hour * 3600 + base.minute * 60 + base.second;

    double delta_seconds = sim_time.value * SECONDS_PER_YEAR;
    int64_t target_sec = base_sec + static_cast<int64_t>(std::floor(delta_seconds + 0.5));

    std::chrono::sys_seconds target_tp{std::chrono::seconds{target_sec}};
    auto target_days = std::chrono::floor<std::chrono::days>(target_tp);
    std::chrono::year_month_day ymd{target_days};
    auto time_of_day = std::chrono::hh_mm_ss{target_tp - target_days};

    DateTimeUTC result;
    result.year = static_cast<int>(ymd.year());
    result.month = static_cast<unsigned>(ymd.month());
    result.day = static_cast<unsigned>(ymd.day());
    result.hour = static_cast<int>(time_of_day.hours().count());
    result.minute = static_cast<int>(time_of_day.minutes().count());
    result.second = static_cast<int>(time_of_day.seconds().count());
    result.tz = "UTC";
    return result;
}

std::string format_sim_date(std::string_view epoch_utc, units::Years sim_time) {
    auto dt = calculate_sim_date(epoch_utc, sim_time);
    return std::format("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d} {}",
                       dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.tz);
}

std::string format_time_speed(double multiplier) {
    std::string sign = multiplier < 0.0 ? "-" : "";
    double mag = std::abs(multiplier);
    if (mag == 0.0) {
        return "Paused";
    } else if (mag >= 0.75 && mag <= 1.25) {
        return sign + "Realtime";
    } else if (mag < 1.0) {
        return std::format("{}{:.2f}x", sign, mag);
    } else if (mag < 60.0) {
        return std::format("{}{:.1f} sec/s", sign, mag);
    } else if (mag < 3600.0) {
        return std::format("{}{:.1f} min/s", sign, mag / 60.0);
    } else if (mag < 86400.0) {
        return std::format("{}{:.1f} hr/s", sign, mag / 3600.0);
    } else if (mag < 604800.0) {
        return std::format("{}{:.1f} days/s", sign, mag / 86400.0);
    } else if (mag < 2629800.0) {
        return std::format("{}{:.1f} weeks/s", sign, mag / 604800.0);
    } else if (mag < 31557600.0) {
        return std::format("{}{:.1f} months/s", sign, mag / 2629800.0);
    } else {
        return std::format("{}{:.1f} years/s", sign, mag / 31557600.0);
    }
}

} // namespace sf::time
