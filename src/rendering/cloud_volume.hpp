#pragma once

#include "rhi/vk_context.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>
#include <glm/glm.hpp>

namespace sf {
struct Camera;
namespace ephemeris { class SystemSnapshot; }
}

namespace sf::rendering {

class CloudMaps;
struct RenderSettings;

// Volumetric cloud data (docs/volumetric_clouds_plan.md, phase 1): tileable noise, the per-type vertical profile
// and its conservative height bounds, and the focus body's camera-relative noise anchors.
inline constexpr uint32_t CLOUD_SHAPE_NOISE_SIZE = 128;
inline constexpr uint32_t CLOUD_DETAIL_NOISE_SIZE = 32;
// Base Worley cells per noise tile, so a tile spans this many `shape_size_km` (or `detail_size_km`) cells.
inline constexpr double CLOUD_NOISE_TILE_CELLS = 4.0;
inline constexpr uint32_t CLOUD_PROFILE_TYPES = 4;
inline constexpr uint32_t CLOUD_PROFILE_HEIGHTS = 128;
inline constexpr uint32_t CLOUD_PROFILE_THRESHOLDS = 64;
inline constexpr uint32_t CLOUD_VOLUME_NONE = UINT32_MAX;
// Highest coverage-map mip the density reads. The max pyramid is dilated for this level's bilinear footprint.
inline constexpr uint32_t CLOUD_VOLUME_MAX_COVERAGE_LOD = 1;
// Above the layer top the volume stays prepared up to this altitude (the hand-off ends here, plan §8).
inline constexpr double CLOUD_VOLUME_FADE_END_KM = 80.0;

enum class CloudType : uint32_t { Stratus = 0, Mixed = 1, Cumulus = 2, Cirrus = 3 };
uint32_t cloud_type_index(std::string_view type) noexcept;

// Analytic vertical profile the LUT is built from. `shape` is where the type can exist (0 at both shell
// boundaries), `density` scales extinction with height.
struct CloudProfileSample {
    double shape = 0.0;
    double density = 0.0;
};
CloudProfileSample cloud_profile(uint32_t type, double h) noexcept;

struct CloudProfileTable {
    // RG16F texels, row-major: type rows of CLOUD_PROFILE_HEIGHTS (shape, density) pairs.
    std::vector<uint16_t> texels;
    // Per type and threshold t_k = k/(THRESHOLDS-1): the height interval outside which the filtered shape is
    // provably <= t_k. Empty when x > y.
    std::array<glm::vec2, CLOUD_PROFILE_TYPES * CLOUD_PROFILE_THRESHOLDS> bounds{};
};
CloudProfileTable build_cloud_profile_table();

struct alignas(16) CloudVolumeGPU {
    glm::uvec4 flags{CLOUD_VOLUME_NONE, 0u, 0u, 0u};  // x focus body, y bound pyramid levels, zw bound level-0 size
    glm::vec4 camera_local{0.0f};    // xyz camera minus focus-body centre, world axes, km
    glm::vec4 shape_anchor{0.0f};    // xyz fract((camera_fixed - wind)/tile), w 1/shape tile km
    glm::vec4 detail_anchor{0.0f};   // same for the detail tile
    glm::vec4 erosion{0.0f};         // x base erosion depth cap, y detail erosion depth, z max coverage lod
    glm::vec4 bounds[CLOUD_PROFILE_TYPES * CLOUD_PROFILE_THRESHOLDS]{};
};

// FP64 wind accumulation. The GPU only receives fract((camera_fixed - wind)/tile) and camera-relative offsets,
// so no large coordinate reaches FP32 and the wrap at a tile period is exact.
class CloudWindOffset {
public:
    void reset() noexcept { tracking_ = false; wind_shape_ = wind_detail_ = glm::dvec3(0.0); }
    void advance(uint64_t identity, glm::dvec3 camera_fixed_km, double seconds, double wind_ms,
                 double shape_tile_km, double detail_tile_km);
    glm::dvec3 shape_anchor(glm::dvec3 camera_fixed_km) const noexcept;
    glm::dvec3 detail_anchor(glm::dvec3 camera_fixed_km) const noexcept;
    glm::dvec3 shape_wind_km() const noexcept { return wind_shape_; }

private:
    bool tracking_ = false;
    uint64_t identity_ = 0;
    double seconds_ = 0.0;
    double shape_tile_km_ = 1.0, detail_tile_km_ = 1.0;
    glm::dvec3 wind_shape_{0.0}, wind_detail_{0.0};
};

class CloudVolumeRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void prepare(rhi::VkContext& ctx, const Camera& cam, const ephemeris::SystemSnapshot& sys,
                 const RenderSettings& settings, const CloudMaps& maps, uint32_t count);
    // Parameters with the profile bounds and coverage-bound dimensions filled and no focus body; probes adjust them.
    CloudVolumeGPU base_parameters(const CloudMaps& maps) const;
    const CloudVolumeGPU& parameters() const noexcept { return params_; }
    // Bindings 40 (parameters) to 44 (max-coverage pyramid), for the frame's parameter buffer.
    std::array<rhi::PushDescriptorWrite, 5> writes(rhi::VkContext& ctx, const CloudMaps& maps) const;
    std::array<rhi::PushDescriptorWrite, 4> resource_writes(rhi::VkContext& ctx, const CloudMaps& maps) const;
    const rhi::Image& shape_noise() const noexcept { return shape_noise_; }
    const rhi::Image& detail_noise() const noexcept { return detail_noise_; }
    const rhi::Image& profile_lut() const noexcept { return profile_lut_; }
    const CloudProfileTable& profile_table() const noexcept { return profile_; }
    double noise_generation_ms() const noexcept { return noise_ms_; }

private:
    void generate_noise(rhi::VkContext& ctx);

    rhi::Image shape_noise_{}, detail_noise_{}, profile_lut_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> param_buffers_{};
    CloudProfileTable profile_{};
    CloudVolumeGPU params_{};
    CloudWindOffset wind_{};
    double noise_ms_ = 0.0;
};

}
