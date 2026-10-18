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
    bool terrain_enabled = false;
    bool terrain_imported_enabled = true;
    bool surface_textures_enabled = true;
    bool terrain_debug_tiles = false;
    bool terrain_reference_fp64 = false;
    bool terrain_cached_fp64 = false;
    float terrain_error_pixels = 2.0f;
    float terrain_height_km = 2.0f;
    uint32_t terrain_patch_limit = 1024;
    bool show_rings = true;
    bool ring_shadows_enabled = true;
    bool eclipse_reference_smoothstep = false;
    bool ring_shadow_reference_all_hosts = false;
    bool ring_shadow_single_box = true;
    bool ring_reference_full_annulus = false;
    bool ring_reference_finite_source = false;
    bool ring_reference_base_level = false;
    bool operator==(const RenderSettings&) const = default;
};

class RenderSettingsStore {
public:
    const RenderSettings& values() const noexcept { return values_; }
    uint64_t revision() const noexcept { return revision_; }

    CommandResult apply(const RenderSettings& candidate) {
        if (!std::isfinite(candidate.terrain_error_pixels) || candidate.terrain_error_pixels < 0.5f || candidate.terrain_error_pixels > 16.0f ||
            !std::isfinite(candidate.terrain_height_km) || candidate.terrain_height_km < 0.0f || candidate.terrain_height_km > 20.0f ||
            candidate.terrain_patch_limit < 64 || candidate.terrain_patch_limit > 4096) {
            return {false, false, "Terrain error must be 0.5 to 16 pixels, relief 0 to 20 km and patch limit 64 to 4096."};
        }
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
