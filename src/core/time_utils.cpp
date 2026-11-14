#include "core/time_utils.hpp"
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <format>

namespace sf::time {

namespace {

LocalTimezoneInfo make_tz_info(const std::tm& tm_local, const std::tm& tm_utc) {
    LocalTimezoneInfo info;
    int64_t d_days = days_from_civil(tm_local.tm_year + 1900, tm_local.tm_mon + 1, tm_local.tm_mday) -
                     days_from_civil(tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday);
    info.offset_seconds = d_days * 86400 +
                          (tm_local.tm_hour - tm_utc.tm_hour) * 3600 +
                          (tm_local.tm_min - tm_utc.tm_min) * 60 +
                          (tm_local.tm_sec - tm_utc.tm_sec);
    char buf[64] = {0};
    std::strftime(buf, sizeof(buf), "%Z", &tm_local);
    std::string raw_name = buf;
    if (raw_name.empty()) {
        int hrs = static_cast<int>(info.offset_seconds / 3600);
        int mins = std::abs(static_cast<int>((info.offset_seconds % 3600) / 60));
        if (mins == 0) info.name = std::format("UTC{:+d}", hrs);
        else info.name = std::format("UTC{:+d}:{:02d}", hrs, mins);
    } else if (raw_name.size() > 6) {
        std::string initials;
        for (char c : raw_name) {
            if (std::isupper(static_cast<unsigned char>(c))) initials += c;
        }
        if (!initials.empty() && initials.size() <= 6) {
            info.name = initials;
        } else {
            int hrs = static_cast<int>(info.offset_seconds / 3600);
            int mins = std::abs(static_cast<int>((info.offset_seconds % 3600) / 60));
            if (mins == 0) info.name = std::format("UTC{:+d}", hrs);
            else info.name = std::format("UTC{:+d}:{:02d}", hrs, mins);
        }
    } else {
        info.name = raw_name;
    }
    return info;
}

double parse_iso_epoch_seconds(std::string_view iso_str) {
    auto base = parse_iso_or_default(iso_str);
    int64_t base_days = days_from_civil(base.year, static_cast<unsigned>(base.month), static_cast<unsigned>(base.day));
    double sec = static_cast<double>(base_days * 86400 + base.hour * 3600 + base.minute * 60 + base.second);
    if (!iso_str.empty()) {
        std::string s(iso_str);
        long long y = 0;
        int m = 0, d = 0, h = 0, mn = 0;
        double fsec = 0.0;
        if (std::sscanf(s.c_str(), "%lld-%d-%d%*c%d:%d:%lf", &y, &m, &d, &h, &mn, &fsec) == 6) {
            if (m >= 1 && m <= 12 && d >= 1 && d <= 31 && h >= 0 && h <= 23 && mn >= 0 && mn <= 59 && fsec >= 0.0 && fsec < 61.0) {
                sec = static_cast<double>(base_days * 86400 + h * 3600 + mn * 60) + fsec;
            }
        }
    }
    return sec;
}

} // namespace

LocalTimezoneInfo get_local_tz_info(int64_t unix_epoch_seconds) {
    static const auto ref_tz = []() {
#if defined(_WIN32)
        _tzset();
#else
        tzset();
#endif
        const time_t ref_time = 1767268800;
        std::tm tm_local{}, tm_utc{};
#if defined(_WIN32)
        _localtime64_s(&tm_local, &ref_time);
        _gmtime64_s(&tm_utc, &ref_time);
#else
        localtime_r(&ref_time, &tm_local);
        gmtime_r(&ref_time, &tm_utc);
#endif
        return make_tz_info(tm_local, tm_utc);
    }();

    constexpr int64_t max_valid_time = sizeof(time_t) >= 8 ? 32503680000LL : 2145916800LL;
    if (unix_epoch_seconds >= 86400 && unix_epoch_seconds <= max_valid_time) {
        time_t t = static_cast<time_t>(unix_epoch_seconds);
        std::tm tm_local{}, tm_utc{};
#if defined(_WIN32)
        if (_localtime64_s(&tm_local, &t) == 0 && _gmtime64_s(&tm_utc, &t) == 0)
#else
        if (localtime_r(&t, &tm_local) && gmtime_r(&t, &tm_utc))
#endif
        {
            return make_tz_info(tm_local, tm_utc);
        }
    }

    return ref_tz;
}

DateTimeUTC parse_iso_or_default(std::string_view iso_str, DateTimeUTC fallback) {
    if (iso_str.empty()) return fallback;
    std::string s(iso_str);
    long long y = 0;
    int m = 0, d = 0, h = 0, mn = 0;
    double fsec = 0.0;
    if (std::sscanf(s.c_str(), "%lld-%d-%d%*c%d:%d:%lf", &y, &m, &d, &h, &mn, &fsec) == 6) {
        int sec = static_cast<int>(std::floor(fsec + 1e-9));
        if (m >= 1 && m <= 12 && d >= 1 && d <= 31 && h >= 0 && h <= 23 && mn >= 0 && mn <= 59 && sec >= 0 && sec <= 60) {
            return {static_cast<int64_t>(y), m, d, h, mn, sec, "UTC"};
        }
    } else if (std::sscanf(s.c_str(), "%lld-%d-%d", &y, &m, &d) == 3) {
        if (m >= 1 && m <= 12 && d >= 1 && d <= 31) {
            return {static_cast<int64_t>(y), m, d, 0, 0, 0, "UTC"};
        }
    }
    return fallback;
}

DateTimeUTC current_wall_clock_date(bool use_local_tz) {
    const auto now = std::chrono::system_clock::now();
    const int64_t unix_sec = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    int64_t target_sec = unix_sec;
    std::string tz_name = "UTC";
    if (use_local_tz) {
        auto tz_info = get_local_tz_info(unix_sec);
        target_sec += tz_info.offset_seconds;
        tz_name = tz_info.name;
    }

    int64_t days = target_sec / 86400;
    int64_t rem = target_sec % 86400;
    if (rem < 0) {
        rem += 86400;
        days -= 1;
    }
    auto civil = civil_from_days(days);
    int hour = static_cast<int>(rem / 3600);
    rem %= 3600;
    int minute = static_cast<int>(rem / 60);
    int second = static_cast<int>(rem % 60);

    DateTimeUTC result;
    result.year = civil.year;
    result.month = static_cast<int>(civil.month);
    result.day = static_cast<int>(civil.day);
    result.hour = hour;
    result.minute = minute;
    result.second = second;
    result.tz = tz_name;
    return result;
}

DateTimeUTC calculate_sim_date(std::string_view epoch_utc, units::Years sim_time, bool use_local_tz) {
    double base_sec = parse_iso_epoch_seconds(epoch_utc);
    double delta_seconds = sim_time.value * SECONDS_PER_YEAR;
    int64_t target_sec = static_cast<int64_t>(std::floor(base_sec + delta_seconds + 0.5));

    std::string tz_name = "UTC";
    if (use_local_tz) {
        auto tz_info = get_local_tz_info(target_sec);
        target_sec += tz_info.offset_seconds;
        tz_name = tz_info.name;
    }

    int64_t days = target_sec / 86400;
    int64_t rem = target_sec % 86400;
    if (rem < 0) {
        rem += 86400;
        days -= 1;
    }
    auto civil = civil_from_days(days);
    int hour = static_cast<int>(rem / 3600);
    rem %= 3600;
    int minute = static_cast<int>(rem / 60);
    int second = static_cast<int>(rem % 60);

    DateTimeUTC result;
    result.year = civil.year;
    result.month = static_cast<int>(civil.month);
    result.day = static_cast<int>(civil.day);
    result.hour = hour;
    result.minute = minute;
    result.second = second;
    result.tz = tz_name;
    return result;
}

std::string format_sim_date(std::string_view epoch_utc, units::Years sim_time, bool use_local_tz) {
    auto dt = calculate_sim_date(epoch_utc, sim_time, use_local_tz);
    if (dt.year >= 0 && dt.year <= 9999) {
        return std::format("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d} {}",
                           dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.tz);
    } else if (dt.year > 9999) {
        return std::format("{:d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d} {}",
                           dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.tz);
    } else {
        return std::format("-{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d} {}",
                           -dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.tz);
    }
}

units::Years sim_time_from_date(std::string_view epoch_utc, const DateTimeUTC& target, bool input_is_utc) {
    double base_sec = parse_iso_epoch_seconds(epoch_utc);

    int64_t target_days = days_from_civil(target.year, static_cast<unsigned>(target.month), static_cast<unsigned>(target.day));
    int64_t target_sec = target_days * 86400 + target.hour * 3600 + target.minute * 60 + target.second;

    if (!input_is_utc) {
        auto tz_info = get_local_tz_info(target_sec);
        target_sec -= tz_info.offset_seconds;
    }

    double delta_sec = static_cast<double>(target_sec) - base_sec;
    return units::Years{delta_sec / SECONDS_PER_YEAR};
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
