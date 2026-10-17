#pragma once

#include "ephemeris/system_manager.hpp"
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <string_view>
#include <vector>

namespace sf::rendering {

inline constexpr uint32_t RING_ATLAS_WIDTH = 4096;
inline constexpr uint32_t RING_ATLAS_ROWS = 64;
inline constexpr uint32_t RING_PROFILE_MIP_LEVELS = 13;
inline constexpr uint32_t RING_SHADOW_LUT_WIDTH = RING_ATLAS_WIDTH + 1;
inline constexpr uint32_t RING_SHADOW_LUT_ANGLES = 64;
inline constexpr uint32_t MAX_RING_HOSTS = 16;
inline constexpr uint32_t MAX_RING_SEGMENTS = 8;
inline constexpr uint32_t RING_ANGULAR_SEGMENTS = 128;
inline constexpr uint32_t RING_RADIAL_SUBDIVISIONS = 32;
inline constexpr float RING_MAX_ALPHA = 0.99999994f;

struct RingDrawRange {
    uint32_t first_index = 0;
    uint32_t index_count = 0;
    uint32_t segment_mask = 0;
    uint32_t packed_segments = 0;
    uint32_t segment_count = 0;
};

std::vector<RingDrawRange> build_ring_draw_ranges(std::span<const ephemeris::RingLayerDefinition> layers,
                                                double min_radius, double max_radius, double planet_radius_au,
                                                uint32_t angular_segments = RING_ANGULAR_SEGMENTS,
                                                uint32_t radial_subdivisions = RING_RADIAL_SUBDIVISIONS);

enum class RingLightingMode : uint32_t { Point, FiniteSource, Reflection, Transmission };

struct RingDrawClassification {
    RingLightingMode lighting = RingLightingMode::FiniteSource;
    bool radial_ranges = false;
};

RingDrawClassification classify_ring_draw(glm::vec3 center, glm::vec3 pole, float outer_radius,
                                          std::span<const glm::vec4> stars) noexcept;

float ring_tau(float alpha) noexcept;
float ring_absorbed(float tau_val) noexcept;
float ring_transfer(float tau_val, float mu_s, float mu_v, bool same_side) noexcept;
float ring_opposition_factor(float cos_alpha, float tau_val) noexcept;
float ring_phase_hg(float cos_theta, float g) noexcept;
float ring_phase_dhg(float cos_theta, float scatter, float g_fwd, float g_back, float tau_val, bool is_textured) noexcept;
glm::vec3 ring_chandrasekhar_h(float mu, glm::vec3 w0) noexcept;
glm::vec3 ring_slab_x(float mu, float tau_val, glm::vec3 w0) noexcept;
glm::vec3 ring_multiple_scattering(float tau_val, float mu_s, float mu_v, bool same_side, glm::vec3 linear_rgb) noexcept;
glm::vec3 ring_radiance_rgb(float total_tau, float mu_v, float sun_elev, float cos_theta,
                            glm::vec4 props, glm::vec3 linear_rgb, bool is_textured, bool same_side) noexcept;

float sample_ring_gradient(std::span<const ephemeris::RingGradientStop> stops, float p) noexcept;
glm::vec4 apply_hsba_pixel(glm::vec4 rgba, float hue_shift, float saturation, float brightness,
                           float opacity_mult = 1.0f, float alpha_boost = 1.0f) noexcept;

std::vector<glm::vec4> generate_ring_segment_profile(
    const ephemeris::RingLayerDefinition& layer,
    std::span<const glm::vec4> raw_texture_samples = {},
    uint32_t res = RING_ATLAS_WIDTH);

std::vector<std::vector<glm::vec4>> build_ring_profile_mips(std::span<const glm::vec4> profile);

struct RingSegmentProfileInput {
    double inner_r = 1.5;
    double outer_r = 2.5;
    float opacity = 0.7f;
    std::span<const glm::vec4> profile;
};

std::vector<glm::vec4> bake_unified_shadow_profile(
    std::span<const RingSegmentProfileInput> segments,
    double min_r, double max_r,
    uint32_t res = RING_ATLAS_WIDTH);

std::vector<float> bake_ring_shadow_lut_layer(
    std::span<const glm::vec4> unified_profile,
    uint32_t width = RING_ATLAS_WIDTH,
    uint32_t angles = RING_SHADOW_LUT_ANGLES);

std::optional<std::vector<glm::vec4>> load_ring_png_1d(
    const std::filesystem::path& path,
    uint32_t target_samples = RING_ATLAS_WIDTH);

bool save_ring_png_2d(
    const std::filesystem::path& path,
    std::span<const glm::vec4> radial_rgba,
    uint32_t height = 64);

bool bake_and_save_ring_texture(
    std::string_view system_name,
    std::string_view body_name,
    ephemeris::RingLayerDefinition& layer);

} // namespace sf::rendering
