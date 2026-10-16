#pragma once

#include "core/command_result.hpp"
#include "ephemeris/system_manager.hpp"
#include <filesystem>

namespace sf::ephemeris {

class SpiceEphemeris {
public:
    static std::shared_ptr<SpiceEphemeris> open(std::string& error);
    ~SpiceEphemeris();
    SpiceEphemeris(const SpiceEphemeris&) = delete;
    SpiceEphemeris& operator=(const SpiceEphemeris&) = delete;
    CommandResult epoch(std::string_view utc, double& et) const;
    CommandResult state(int target, double et, BodyMotion& motion) const;

private:
    SpiceEphemeris() = default;
    CommandResult initialize();
    CommandResult result() const;
    void* library_ = nullptr;
    void (*furnish_)(const char*) = nullptr;
    void (*clear_)() = nullptr;
    void (*error_action_)(const char*, int, char*) = nullptr;
    void (*error_print_)(const char*, int, char*) = nullptr;
    int (*failed_)() = nullptr;
    void (*message_)(const char*, int, char*) = nullptr;
    void (*reset_)() = nullptr;
    void (*to_et_)(const char*, double*) = nullptr;
    void (*state_)(int, double, const char*, int, double*, double*) = nullptr;
};

}
