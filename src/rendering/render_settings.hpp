#pragma once

#include "core/command_result.hpp"
#include <cmath>
#include <cstdint>

namespace sf::rendering {

inline constexpr float DEFAULT_POINT_DIAMETER_PX = 3.0f;

struct RenderSettings {
    float exposure = 1.0f;
    bool hdr_enabled = true;
    uint32_t bloom_mode = 2;
    float bloom_strength = 0.08f;
    float bloom_highlight_limit = 64.0f;
    float bloom_radius = 1.0f;
    float bloom_diffraction = 0.65f;
    float bloom_rotation_deg = 45.0f;
    float bloom_spike_threshold = 4.0f;
    bool show_gaia_stars = true;
    bool show_orbits = true;
    float orbit_opacity = 1.0f;
    uint32_t orbit_segments = 1030;
    float star_point_size = DEFAULT_POINT_DIAMETER_PX;
    float star_intensity = 1.0f;
    bool vsync = false;
    bool terrain_enabled = false;
    bool terrain_imported_enabled = true;
    bool terrain_normal_maps_enabled = true;
    bool surface_textures_enabled = true;
    bool terrain_debug_tiles = false;
    bool terrain_reference_fp64 = false;
    bool terrain_cached_fp64 = false;
    float terrain_error_pixels = 16.0f;
    float terrain_height_km = 2.0f;
    uint32_t terrain_patch_limit = 1024;
    bool show_rings = true;
    bool transparency_reference_legacy = false;
    bool show_atmospheres = true;
    bool direct_sunlight_only = false;
    bool aerial_perspective_enabled = true;
    bool refraction_enabled = true;
    bool eclipse_refraction_enabled = true;
    bool eclipse_refraction_probe = false;
    bool atmosphere_dual_source = true;
    bool atmosphere_surface_lighting = true;
    bool atmosphere_shadows_enabled = true;
    bool atmosphere_shadow_reference = false;
    bool atmosphere_shadow_cached_weights = true;
    bool atmosphere_shadow_fast_rings = true;
    bool atmosphere_shadow_analytic = true;
    bool atmosphere_shadow_temporal = false;
    bool atmosphere_shadow_animate_jitter = true;
    bool atmosphere_shadow_rotation = false;
    bool atmosphere_shadow_adaptive = true;
    bool atmosphere_shadow_vrs = true;
    uint32_t atmosphere_shadow_vrs_scale = 4;
    uint32_t atmosphere_shadow_noise_type = 0;
    uint32_t atmosphere_shadow_samples = 4;
    bool ring_shadows_enabled = true;
    bool ringshine_enabled = true;
    bool planetshine_enabled = true;
    bool oceans_enabled = true;
    bool clouds_enabled = true;
    uint32_t cloud_mode = 0;
    bool cloud_shadows_enabled = false;
    bool atmosphere_planetshine_reference = false;
    uint32_t ring_planetshine_mode = 0;
    uint32_t ringshine_band_count = 10;
    bool eclipse_reference_smoothstep = false;
    bool ring_shadow_reference_all_hosts = false;
    float ring_shadow_softness = 1.25f;
    bool ring_shadow_single_box = false;
    bool ring_shadow_reference_linear = false;
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
        if (candidate.bloom_mode > 2 ||
            !std::isfinite(candidate.bloom_strength) || candidate.bloom_strength < 0.0f || candidate.bloom_strength > 1.0f ||
            !std::isfinite(candidate.bloom_highlight_limit) || candidate.bloom_highlight_limit < 1.0f || candidate.bloom_highlight_limit > 4096.0f ||
            !std::isfinite(candidate.bloom_radius) || candidate.bloom_radius < 0.25f || candidate.bloom_radius > 4.0f ||
            !std::isfinite(candidate.bloom_diffraction) || candidate.bloom_diffraction < 0.0f || candidate.bloom_diffraction > 1.0f ||
            !std::isfinite(candidate.bloom_rotation_deg) || candidate.bloom_rotation_deg < 0.0f || candidate.bloom_rotation_deg > 90.0f ||
            !std::isfinite(candidate.bloom_spike_threshold) || candidate.bloom_spike_threshold < 0.25f || candidate.bloom_spike_threshold > 4096.0f)
            return {false, false, "Bloom requires a valid mode, strength and diffraction from 0 to 1, highlight limit from 1 to 4096, radius from 0.25 to 4, rotation from 0 to 90 degrees, and spike threshold from 0.25 to 4096."};
        if (candidate.cloud_mode > 1) return {false, false, "Cloud mode must be 2D texture or volumetric."};
        if (!std::isfinite(candidate.ring_shadow_softness) || candidate.ring_shadow_softness < 0.25f || candidate.ring_shadow_softness > 4.0f)
            return {false, false, "Ring shadow softness must be 0.25 to 4."};
        if (candidate.atmosphere_shadow_samples != 4 && candidate.atmosphere_shadow_samples != 8 && candidate.atmosphere_shadow_samples != 16)
            return {false, false, "Atmospheric shadows use 4, 8 or 16 samples."};
        if (candidate.atmosphere_shadow_vrs_scale < 2 || candidate.atmosphere_shadow_vrs_scale > 16 ||
            (candidate.atmosphere_shadow_vrs_scale & (candidate.atmosphere_shadow_vrs_scale-1)) != 0)
            return {false, false, "Atmospheric shadow VRS uses a 2, 4, 8 or 16 pixel coarse grid."};
        if (candidate.atmosphere_shadow_noise_type > 1)
            return {false, false, "Atmospheric shadow noise type must be 0 (IGN) or 1 (STBN)."};
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
        if (candidate.ring_planetshine_mode > 1) return {false, false, "Ring planetshine mode must be analytical or numerical."};
        if (candidate.ringshine_band_count < 4 || candidate.ringshine_band_count > 1024) {
            return {false, false, "Ringshine bands must be 4 to 1024."};
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
