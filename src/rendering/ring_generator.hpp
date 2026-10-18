#pragma once

#include "ephemeris/system_manager.hpp"
#include "rendering/ring_optics.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace sf::rendering {

struct RingPreset {
    std::string_view name;
    int band_count = 4;
    int gap_count = 3;
    float striation_freq = 160.0f;
    float contrast = 1.7f;
    float base_density = 2.2f;
    float core_boost = 2.5f;
    float inner_radius_ratio = 1.20f;
    float outer_radius_ratio = 2.30f;
    float asymmetry = 0.436f;
    float backscatter = -0.657f;
    float scatter = 1.0f;
    float opacity = 1.0f;
    float unlit_factor = 0.45f;
    glm::vec3 color_deep{0.97f, 0.94f, 0.89f};
    glm::vec3 color_mid{0.95f, 0.95f, 0.96f};
    glm::vec3 color_thin{0.75f, 0.86f, 0.98f};
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
    float base_density = 2.2f;
    float core_boost = 2.5f;
    float inner_radius_ratio = 1.20f;
    float outer_radius_ratio = 2.30f;
    glm::vec3 tint_color{1.0f, 1.0f, 1.0f};
    float opacity = 1.0f;

    float asymmetry = 0.436f;
    float backscatter = -0.657f;
    float scatter = 1.0f;
    float unlit_factor = 0.45f;

    glm::vec3 color_deep{0.97f, 0.94f, 0.89f};
    glm::vec3 color_mid{0.95f, 0.95f, 0.96f};
    glm::vec3 color_thin{0.75f, 0.86f, 0.98f};

    static ProceduralRingConfig from_preset(const RingPreset& preset, uint32_t seed = 42) noexcept;
};

struct ProceduralRingResult {
    std::vector<glm::vec4> profile;
    float inner_radius_ratio = 1.20f;
    float outer_radius_ratio = 2.30f;
    float opacity = 1.0f;
    float scatter = 1.0f;
    float asymmetry = 0.436f;
    float backscatter = -0.657f;
    float unlit_factor = 0.45f;
    glm::vec3 raw_color{1.0f, 1.0f, 1.0f};
};

ProceduralRingResult generate_procedural_ring_profile(
    const ProceduralRingConfig& cfg,
    uint32_t res = RING_ATLAS_WIDTH);

void apply_procedural_ring_layer(
    const ProceduralRingResult& result,
    ephemeris::RingLayerDefinition& out_layer) noexcept;

bool bake_and_save_procedural_ring_texture(
    std::string_view system_name,
    std::string_view body_name,
    const ProceduralRingResult& result,
    ephemeris::RingLayerDefinition& out_layer,
    uint32_t texture_height = 64);

} // namespace sf::rendering
