#pragma once

#include "rhi/vk_context.hpp"
#include <array>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sf::ephemeris { struct CloudDefinition; }

namespace sf::rendering {

// Max reduction of a coverage map for conservative empty-space bounds (volumetric_clouds_plan.md §4/§5). The mean
// mips in the atlas cannot bound anything. Level 0 has at most `max_width` cells; each cell holds the maximum over
// its source texels dilated by `dilation_texels`, so a bilinear read of the atlas at any mip up to the dilation's
// footprint never exceeds the cell covering the read point.
struct CloudBoundPyramid {
    uint32_t width = 1, height = 1;
    std::vector<std::vector<uint8_t>> levels;
};
CloudBoundPyramid build_cloud_bound_pyramid(std::span<const uint8_t> level0, uint32_t width, uint32_t height,
                                            uint32_t max_width, double dilation_texels);
inline constexpr uint32_t CLOUD_BOUND_MAX_WIDTH = 1024;

class CloudMaps {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    const rhi::Image& image() const noexcept { return atlas_; }
    // R8 max-coverage pyramid with the atlas's layers (texelFetch only).
    const rhi::Image& bound_image() const noexcept { return bounds_; }
    uint32_t layer(const std::string& name) const noexcept;
    double mean_disk_opacity(const ephemeris::CloudDefinition& clouds) const noexcept;

private:
    rhi::Image atlas_;
    rhi::Image bounds_;
    std::unordered_map<std::string, uint32_t> layers_;
    std::unordered_map<std::string, std::array<float, 256>> histograms_;
};

}
