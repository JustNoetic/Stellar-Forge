#pragma once

#include "core/camera.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/refraction_tables.hpp"
#include "rhi/vk_context.hpp"
#include "terrain/quadtree.hpp"
#include "terrain/geometry_cache.hpp"
#include "terrain/tile_metadata.hpp"
#include <array>
#include <span>

namespace sf::terrain {

inline constexpr double TERRAIN_MIN_PROJECTED_RADIUS_PX = 2.5;

struct TerrainStats {
    double max_error_pixels = 0.0;
    uint32_t patches = 0;
    uint32_t drawn_bodies = 0;
    uint32_t deepest_level = 0;
    bool budget_limited = false;
    uint32_t selection_rebuilds = 0;
    uint32_t selection_nodes = 0;
    uint32_t selection_reuses = 0;
    uint32_t selection_checks = 0;
    uint32_t dependency_rebuilds = 0;
    uint32_t patch_updates = 0;
    uint32_t placement_updates = 0;
    uint32_t metadata_builds = 0;
    uint32_t metadata_entries = 0;
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;
    uint32_t cache_bakes = 0;
    uint32_t cached_patches = 0;
    uint32_t reference_patches = 0;
    uint32_t imported_patches = 0;
    uint32_t smooth_patches = 0;
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
    glm::vec4 anchor_offset{0.0f};
};

class TerrainRenderer {
public:
    void init(rhi::VkContext& ctx, bool load_config = true);
    void shutdown(rhi::VkContext& ctx);
    void prepare(rhi::VkContext& ctx, const Camera& camera, const ephemeris::SystemSnapshot& system,
                 const rendering::RenderSettings& settings, VkExtent2D extent, uint32_t body_count,
                 const rendering::RefractionHost* refraction = nullptr);
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, std::span<const rhi::PushDescriptorWrite> scene);
    void bake(VkCommandBuffer cmd, rhi::VkContext& ctx) { cache_.bake(cmd, ctx); }
    bool covers(uint32_t body) const noexcept;
    const TerrainStats& stats() const noexcept { return stats_; }
    void set_height_source(std::string body, std::shared_ptr<const HeightSource> source);
    std::shared_ptr<const HeightSource> height_source(std::string_view body) const noexcept;
    std::vector<std::pair<std::string, std::shared_ptr<const HeightSource>>> height_sources() const;
    float relief_extent_au(uint32_t body) const noexcept;
    double minimum_height_km(const ephemeris::BodyDefinition& body, const rendering::RenderSettings& settings) const noexcept;

private:
    struct DrawPatch {
        PatchGPU reference;
        CachedPatchGPU cached;
        GeometryKey key;
        GeometryBakeGPU bake;
        uint32_t mode = 0;
    };
    struct PatchPlacement {
        glm::dvec3 anchor;
        GeometryKey key;
        GeometryBakeGPU bake;
    };
    struct PreparedPatch {
        const DrawPatch* draw;
        uint32_t mode;
        uint32_t slot = 0;
    };
    struct DrawBatch {
        uint32_t first;
        uint32_t count;
        uint32_t mode;
        uint32_t mask;
    };
    struct SelectionEntry {
        uint64_t identity = 0;
        uint64_t geometry_revision = 0;
        uint64_t bounds_revision = 0;
        uint64_t residency_revision = 0;
        SourceDomain source;
        bool desired_valid = false;
        bool selection_from_desired = false;
        SelectionPlan plan;
        std::vector<TileId> prefetch;
        std::vector<GeometryKey> protected_keys;
        uint64_t protected_revision = UINT64_MAX;
        std::unordered_map<uint64_t, SurfaceTile> descriptions;
        SelectionView view;
        Selection selection;
        Selection desired;
        std::vector<PatchPlacement> placements;
        std::vector<DrawPatch> draws;
        glm::dvec3 draw_center{0};
        glm::dmat3 draw_rotation{1};
        glm::dmat3 draw_view{1};
        uint32_t draw_flags = UINT32_MAX;
    };
    struct StreamBinding {
        std::string body;
        std::unique_ptr<HeightStream> stream;
        std::string reported_error;
    };
    rhi::Pipeline pipeline_{};
    rhi::Pipeline precise_pipeline_{};
    rhi::Pipeline reference_pipeline_{};
    rhi::Pipeline cached_fp64_pipeline_{};
    rhi::Buffer indices_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> patches_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> cached_patches_{};
    std::vector<PreparedPatch> prepared_;
    std::array<DrawBatch, 64> batches_{};
    uint32_t batch_count_ = 0;
    std::vector<GeometryKey> retained_keys_;
    std::vector<std::shared_ptr<const HeightGeometry>> retained_heights_;
    std::vector<HeightStream*> retained_streams_;
    std::vector<GeometryKey> request_keys_;
    std::vector<GeometryBakeGPU> request_jobs_;
    std::vector<std::shared_ptr<const HeightGeometry>> request_heights_;
    std::vector<std::pair<double, uint32_t>> candidates_;
    std::vector<TileId> prefetch_scratch_;
    SelectionScratch selection_scratch_;
    std::vector<uint8_t> covered_;
    std::vector<std::pair<uint32_t, float>> relief_extents_;
    std::vector<StreamBinding> streams_;
    std::unordered_map<uint32_t, SelectionEntry> selections_;
    GeometryCache cache_;
    TileMetadataCache metadata_;
    TerrainStats stats_{};
};

}
