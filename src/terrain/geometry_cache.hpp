#pragma once

#include "rhi/vk_context.hpp"
#include "terrain/quadtree.hpp"
#include "terrain/geometry_residency.hpp"
#include "terrain/height_stream.hpp"
#include "rhi/staging_uploads.hpp"

namespace sf::terrain {

inline constexpr uint32_t PATCH_VERTICES = (GRID_CELLS + 1) * (GRID_CELLS + 1);
inline constexpr uint32_t GEOMETRY_VERTEX_BYTES = 48;
inline constexpr uint32_t GEOMETRY_CACHE_BYTES = 64 * 1024 * 1024;
inline constexpr uint32_t GEOMETRY_CACHE_CAPACITY = GEOMETRY_CACHE_BYTES / (PATCH_VERTICES * GEOMETRY_VERTEX_BYTES);
inline constexpr uint32_t MAX_GEOMETRY_BAKES = 128;

struct alignas(16) GeometryBakeGPU {
    glm::vec4 anchor_polar_ratio{0.0f};
    glm::vec4 shape{0.0f};
    glm::uvec4 address{0};
    glm::uvec4 destination{0};
};

class GeometryCache {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    CachePlan prepare(rhi::VkContext& ctx, std::span<const GeometryKey> keys, std::span<const GeometryBakeGPU> requests,
                      std::span<const std::shared_ptr<const HeightGeometry>> heights = {});
    int32_t find(const GeometryKey& key) const { return residency_.find(key); }
    uint64_t revision() const noexcept { return residency_.revision(); }
    void bake(VkCommandBuffer cmd, rhi::VkContext& ctx);
    const rhi::Buffer& buffer() const noexcept { return geometry_; }

private:
    GeometryResidency residency_{GEOMETRY_CACHE_CAPACITY};
    rhi::Pipeline bake_pipeline_{};
    rhi::Buffer geometry_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> jobs_{};
    uint32_t bake_count_ = 0;
    rhi::StagingUploads uploads_;
    std::vector<std::pair<uint32_t, std::shared_ptr<const HeightGeometry>>> pending_uploads_;
};

}
