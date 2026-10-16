#pragma once

#include "core/units.hpp"
#include <string>
#include <string_view>

namespace sf::time {

struct DateTimeUTC {
    int year = 2026;
    int month = 1;
    int day = 1;
    int hour = 0;
    int minute = 0;
    int second = 0;
    std::string tz = "UTC";
};

DateTimeUTC parse_iso_or_default(std::string_view iso_str, DateTimeUTC fallback = {2026, 1, 1, 12, 0, 0, "UTC"});
DateTimeUTC calculate_sim_date(std::string_view epoch_utc, units::Years sim_time);
std::string format_sim_date(std::string_view epoch_utc, units::Years sim_time);
std::string format_time_speed(double multiplier);

} // namespace sf::time
