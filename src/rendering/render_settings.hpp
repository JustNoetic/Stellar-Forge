#pragma once

#include "core/command_result.hpp"
#include <cmath>
#include <cstdint>

namespace sf::rendering {

struct RenderSettings {
    float exposure = 1.0f;
    bool hdr_enabled = true;
    bool show_gaia_stars = true;
    float star_point_size = 2.0f;
    float star_intensity = 1.0f;
    bool operator==(const RenderSettings&) const = default;
};

class RenderSettingsStore {
public:
    const RenderSettings& values() const noexcept { return values_; }
    uint64_t revision() const noexcept { return revision_; }

    CommandResult apply(const RenderSettings& candidate) {
        if (!std::isfinite(candidate.exposure) || candidate.exposure < 0.0f ||
            !std::isfinite(candidate.star_point_size) || candidate.star_point_size < 1.5f || candidate.star_point_size > 6.0f ||
            !std::isfinite(candidate.star_intensity) || candidate.star_intensity < 0.0f) {
            return {false, false, "Exposure and intensity must be finite and nonnegative; star size must be 1.5 to 6 pixels."};
        }
        if (candidate == values_) return {};
        values_ = candidate;
        ++revision_;
        return {true, true, {}};
    }

private:
    RenderSettings values_{};
    uint64_t revision_ = 1;
};

}
