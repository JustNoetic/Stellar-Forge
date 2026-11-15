#pragma once

#include "ephemeris/system_manager.hpp"
#include "rendering/ring_optics.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace sf::rendering {

struct RingMaterialLimits {
    float scatter_min = 0.25f;
    float scatter_max = 0.80f;
    float asymmetry_min = 0.20f;
    float asymmetry_max = 0.70f;
    float backscatter_min = -0.80f;
    float backscatter_max = -0.35f;
    float unlit_min = 0.45f;
    float unlit_max = 1.10f;
    float inner_min = 1.14f;
    float inner_max = 1.45f;
    float outer_min = 2.00f;
    float outer_max = 2.75f;
    int band_min = 3;
    int band_max = 7;
    int gap_min = 2;
    int gap_max = 6;
    float striation_min = 100.0f;
    float striation_max = 290.0f;
    float contrast_min = 1.2f;
    float contrast_max = 2.8f;
    float density_min = 0.65f;
    float density_max = 2.10f;
    float core_boost_min = 1.1f;
    float core_boost_max = 2.4f;
    float opacity_min = 0.85f;
    float opacity_max = 1.15f;
    float hue_jitter = 0.06f;
    float sat_jitter = 0.18f;
    float val_jitter = 0.12f;
    float halo_prob = 0.65f;
    float halo_opacity_max = 0.015f;
    float opacity_variety_min = 0.55f;
    float opacity_variety_max = 1.45f;
    float single_band_color_variety_min = 0.45f;
    float single_band_color_variety_max = 1.40f;
    float per_band_color_variety_min = 0.45f;
    float per_band_color_variety_max = 1.45f;
};

struct RingPreset {
    std::string_view name;
    int band_count = 4;
    int gap_count = 3;
    float striation_freq = 160.0f;
    float contrast = 1.7f;
    float base_density = 1.15f;
    float core_boost = 1.65f;
    float inner_radius_ratio = 1.20f;
    float outer_radius_ratio = 2.30f;
    float asymmetry = 0.436f;
    float backscatter = -0.657f;
    float scatter = 0.68f;
    float opacity = 1.0f;
    float unlit_factor = 0.75f;
    glm::vec3 color_deep{0.97f, 0.93f, 0.86f};
    glm::vec3 color_mid{0.92f, 0.92f, 0.94f};
    glm::vec3 color_thin{0.72f, 0.84f, 0.98f};
    RingMaterialLimits limits{};
    float opacity_variety = 1.0f;
    float single_band_color_variety = 1.0f;
    float per_band_color_variety = 1.0f;
};

std::span<const RingPreset> get_ring_presets() noexcept;
const RingPreset& get_ring_preset(size_t index) noexcept;
const RingPreset& get_ring_preset(std::string_view name) noexcept;

struct ProceduralRingConfig {
    uint32_t seed = 42;
    int band_count = 4;
    int gap_count = 3;
    float striation_freq = 160.0f;
    float contrast = 1.7f;
    float base_density = 1.15f;
    float core_boost = 1.65f;
    float inner_radius_ratio = 1.20f;
    float outer_radius_ratio = 2.30f;
    glm::vec3 tint_color{1.0f, 1.0f, 1.0f};
    float opacity = 1.0f;
    float opacity_variety = 1.0f;
    float single_band_color_variety = 1.0f;
    float per_band_color_variety = 1.0f;

    float asymmetry = 0.436f;
    float backscatter = -0.657f;
    float scatter = 0.68f;
    float unlit_factor = 0.75f;

    glm::vec3 color_deep{0.97f, 0.93f, 0.86f};
    glm::vec3 color_mid{0.92f, 0.92f, 0.94f};
    glm::vec3 color_thin{0.72f, 0.84f, 0.98f};

    RingMaterialLimits limits{};
    bool randomize_optics = true;
    int morphology = -1;

    static ProceduralRingConfig from_preset(const RingPreset& preset, uint32_t seed = 42) noexcept;
    static ProceduralRingConfig randomize_from_preset(const RingPreset& preset, uint32_t seed) noexcept;
};

struct ProceduralRingResult {
    std::vector<glm::vec4> profile;
    float inner_radius_ratio = 1.20f;
    float outer_radius_ratio = 2.30f;
    float opacity = 1.0f;
    float scatter = 0.68f;
    float asymmetry = 0.436f;
    float backscatter = -0.657f;
    float unlit_factor = 0.75f;
    glm::vec3 raw_color{1.0f, 1.0f, 1.0f};
    std::vector<ephemeris::RingLayerDefinition> extra_layers;
};

ProceduralRingResult generate_procedural_ring_profile(
    const ProceduralRingConfig& cfg,
    uint32_t res = RING_ATLAS_WIDTH);

void apply_procedural_ring_layer(
    const ProceduralRingResult& result,
    ephemeris::RingLayerDefinition& out_layer) noexcept;

std::vector<ephemeris::RingLayerDefinition> build_procedural_ring_layers(
    const ProceduralRingResult& result);

bool bake_and_save_procedural_ring_texture(
    std::string_view system_name,
    std::string_view body_name,
    const ProceduralRingResult& result,
    ephemeris::RingLayerDefinition& out_layer,
    uint32_t texture_height = 64);

} // namespace sf::rendering
