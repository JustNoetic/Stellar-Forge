#pragma once

#include "rhi/vk_context.hpp"
#include "terrain/quadtree.hpp"
#include "terrain/geometry_residency.hpp"

namespace sf::terrain {

inline constexpr uint32_t PATCH_VERTICES = (GRID_CELLS + 1) * (GRID_CELLS + 1);
inline constexpr uint32_t GEOMETRY_VERTEX_BYTES = 48;
inline constexpr uint32_t GEOMETRY_CACHE_BYTES = 64 * 1024 * 1024;
inline constexpr uint32_t GEOMETRY_CACHE_CAPACITY = GEOMETRY_CACHE_BYTES / (PATCH_VERTICES * GEOMETRY_VERTEX_BYTES);
inline constexpr uint32_t MAX_GEOMETRY_BAKES = 128;

struct alignas(32) GeometryBakeGPU {
    glm::dvec4 anchor_polar_ratio{0.0};
    glm::dvec4 shape{0.0};
    glm::uvec4 address{0};
    glm::uvec4 destination{0};
};

class GeometryCache {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    CachePlan prepare(rhi::VkContext& ctx, std::span<const GeometryKey> keys, std::span<const GeometryBakeGPU> requests);
    void bake(VkCommandBuffer cmd, rhi::VkContext& ctx);
    const rhi::Buffer& buffer() const noexcept { return geometry_; }

private:
    GeometryResidency residency_{GEOMETRY_CACHE_CAPACITY};
    rhi::Pipeline bake_pipeline_{};
    rhi::Buffer geometry_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> jobs_{};
    uint32_t bake_count_ = 0;
};

}
