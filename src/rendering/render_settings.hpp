#pragma once

#include "core/command_result.hpp"
#include <cmath>
#include <cstdint>

namespace sf::rendering {

inline constexpr float DEFAULT_POINT_DIAMETER_PX = 3.0f;

struct RenderSettings {
    float exposure = 1.0f;
    bool hdr_enabled = true;
    bool show_gaia_stars = true;
    bool show_orbits = true;
    float orbit_opacity = 1.0f;
    uint32_t orbit_segments = 2048;
    float star_point_size = DEFAULT_POINT_DIAMETER_PX;
    float star_intensity = 1.0f;
    bool vsync = false;
    bool operator==(const RenderSettings&) const = default;
};

class RenderSettingsStore {
public:
    const RenderSettings& values() const noexcept { return values_; }
    uint64_t revision() const noexcept { return revision_; }

    CommandResult apply(const RenderSettings& candidate) {
        if (!std::isfinite(candidate.orbit_opacity) || candidate.orbit_opacity < 0.0f || candidate.orbit_opacity > 1.0f ||
            candidate.orbit_segments < 128 || candidate.orbit_segments > 8192) {
            return {false, false, "Orbit opacity must be 0 to 1 and segment count 128 to 8192."};
        }
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
