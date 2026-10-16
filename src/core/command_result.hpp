#pragma once

#include <string>

namespace sf {

struct CommandResult {
    bool accepted = true;
    bool changed = false;
    std::string error;
    explicit operator bool() const noexcept { return accepted; }
};

}
