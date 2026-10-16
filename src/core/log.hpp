#pragma once

#include <cstdio>
#include <cstdlib>
#include <format>
#include <string_view>

namespace sf::log {

template <typename... Args>
inline void info(std::format_string<Args...> fmt, Args&&... args) {
    auto msg = std::format(fmt, std::forward<Args>(args)...);
    std::fprintf(stdout, "[INFO] %s\n", msg.c_str());
    std::fflush(stdout);
}

template <typename... Args>
inline void warn(std::format_string<Args...> fmt, Args&&... args) {
    auto msg = std::format(fmt, std::forward<Args>(args)...);
    std::fprintf(stderr, "[WARN] %s\n", msg.c_str());
    std::fflush(stderr);
}

template <typename... Args>
inline void error(std::format_string<Args...> fmt, Args&&... args) {
    auto msg = std::format(fmt, std::forward<Args>(args)...);
    std::fprintf(stderr, "[ERROR] %s\n", msg.c_str());
    std::fflush(stderr);
}

template <typename... Args>
[[noreturn]] inline void fatal(std::format_string<Args...> fmt, Args&&... args) {
    auto msg = std::format(fmt, std::forward<Args>(args)...);
    std::fprintf(stderr, "[FATAL] %s\n", msg.c_str());
    std::fflush(stderr);
    std::abort();
}

} // namespace sf::log
