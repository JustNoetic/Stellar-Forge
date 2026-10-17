#pragma once

#include "core/units.hpp"
#include <cstdint>
#include <string>
#include <string_view>

namespace sf::time {

struct DateTimeUTC {
    int64_t year = 2026;
    int month = 1;
    int day = 1;
    int hour = 0;
    int minute = 0;
    int second = 0;
    std::string tz = "UTC";
};

struct LocalTimezoneInfo {
    int64_t offset_seconds = 0;
    std::string name = "UTC";
};

LocalTimezoneInfo get_local_tz_info(int64_t unix_epoch_seconds = 1767268800);

constexpr int64_t days_from_civil(int64_t y, unsigned m, unsigned d) noexcept {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

struct CivilDate {
    int64_t year = 1970;
    unsigned month = 1;
    unsigned day = 1;
};

constexpr CivilDate civil_from_days(int64_t z) noexcept {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t y = static_cast<int64_t>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;
    const unsigned m = mp < 10 ? mp + 3 : mp - 9;
    return CivilDate{y + (m <= 2), m, d};
}

DateTimeUTC parse_iso_or_default(std::string_view iso_str, DateTimeUTC fallback = {2026, 1, 1, 12, 0, 0, "UTC"});
DateTimeUTC calculate_sim_date(std::string_view epoch_utc, units::Years sim_time, bool use_local_tz = false);
std::string format_sim_date(std::string_view epoch_utc, units::Years sim_time, bool use_local_tz = false);
units::Years sim_time_from_date(std::string_view epoch_utc, const DateTimeUTC& target, bool input_is_utc = true);
std::string format_time_speed(double multiplier);

} // namespace sf::time
