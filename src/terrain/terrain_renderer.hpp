#pragma once

#include "core/camera.hpp"
#include "rendering/render_settings.hpp"
#include "rhi/vk_context.hpp"
#include "terrain/quadtree.hpp"
#include "terrain/geometry_cache.hpp"
#include "terrain/tile_metadata.hpp"
#include <array>
#include <span>

namespace sf::terrain {

struct TerrainStats {
    uint32_t patches = 0;
    uint32_t deepest_level = 0;
    bool budget_limited = false;
    uint32_t selection_rebuilds = 0;
    uint32_t selection_nodes = 0;
    uint32_t metadata_builds = 0;
    uint32_t metadata_entries = 0;
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;
    uint32_t cache_bakes = 0;
    uint32_t cached_patches = 0;
    uint32_t reference_patches = 0;
    uint32_t imported_patches = 0;
    uint32_t loading_bodies = 0;
    StreamStats stream;
};

struct alignas(32) PatchGPU {
    glm::dvec4 anchor_radius{0.0};
    glm::dvec4 anchor_polar_ratio{0.0};
    glm::dvec4 rotation_height{0.0};
    glm::dvec4 rotation_error{0.0};
    glm::dvec4 rotation_reserved{0.0};
    glm::dvec4 eye{0.0};
    glm::uvec4 address{0};
    glm::uvec4 metadata{0};
    glm::dvec4 eye_anchor{0.0};
    glm::dvec4 eye_rotation0{0.0};
    glm::dvec4 eye_rotation1{0.0};
    glm::dvec4 eye_rotation2{0.0};
};

struct alignas(16) CachedPatchGPU {
    glm::vec4 anchor_high{0.0f};
    glm::vec4 anchor_low{0.0f};
    glm::vec4 rotation0{0.0f};
    glm::vec4 rotation1{0.0f};
    glm::vec4 rotation2{0.0f};
    glm::vec4 eye{0.0f};
    glm::uvec4 address{0};
    glm::uvec4 metadata{0};
};

class TerrainRenderer {
public:
    void init(rhi::VkContext& ctx, bool load_config = true);
    void shutdown(rhi::VkContext& ctx);
    void prepare(rhi::VkContext& ctx, const Camera& camera, const ephemeris::SystemSnapshot& system,
                 const rendering::RenderSettings& settings, VkExtent2D extent, uint32_t body_count);
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, std::span<const rhi::PushDescriptorWrite> scene);
    void bake(VkCommandBuffer cmd, rhi::VkContext& ctx) { cache_.bake(cmd, ctx); }
    bool covers(uint32_t body) const noexcept;
    const TerrainStats& stats() const noexcept { return stats_; }
    void set_height_source(std::string body, std::shared_ptr<const HeightSource> source);
    float relief_extent_au(uint32_t body) const noexcept;

private:
    struct SelectionEntry {
        uint64_t generation = 0;
        uint64_t geometry_revision = 0;
        uint64_t bounds_revision = 0;
        uint64_t residency_revision = 0;
        bool desired_valid = false;
        std::vector<TileId> prefetch;
        std::unordered_map<uint64_t, SurfaceTile> descriptions;
        SelectionView view;
        Selection selection;
    };
    struct DrawPatch {
        PatchGPU reference;
        CachedPatchGPU cached;
        GeometryKey key;
        GeometryBakeGPU bake;
        uint32_t mode = 0;
    };
    rhi::Pipeline pipeline_{};
    rhi::Pipeline precise_pipeline_{};
    rhi::Pipeline reference_pipeline_{};
    rhi::Pipeline cached_fp64_pipeline_{};
    rhi::Buffer indices_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> patches_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> cached_patches_{};
    std::vector<DrawPatch> prepared_;
    std::vector<uint32_t> bodies_;
    std::vector<std::pair<uint32_t, float>> relief_extents_;
    std::string source_body_;
    std::unique_ptr<HeightStream> stream_;
    std::string reported_error_;
    std::unordered_map<uint32_t, SelectionEntry> selections_;
    uint64_t selection_generation_ = 0;
    GeometryCache cache_;
    TileMetadataCache metadata_;
    TerrainStats stats_{};
};

}
