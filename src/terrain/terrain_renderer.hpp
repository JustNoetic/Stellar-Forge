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
#include <unordered_map>

namespace sf::terrain {

inline constexpr double TERRAIN_MIN_PROJECTED_RADIUS_PX = 2.5;

enum SurfaceShaderBits : uint32_t {
    SURFACE_BIT_FULL_LIGHTING  = 1u << 0,
    SURFACE_BIT_ATMOSPHERE     = 1u << 1,
    SURFACE_BIT_OCEAN          = 1u << 2,
    SURFACE_BIT_OCEAN_FFT      = 1u << 3,
    SURFACE_BIT_ALBEDO_MAP     = 1u << 4,
    SURFACE_BIT_NORMALS        = 1u << 5,
    SURFACE_BIT_HORIZONS       = 1u << 6,
    SURFACE_BIT_SHADOW_CASTERS = 1u << 7,
    SURFACE_BIT_PLANETSHINE        = 1u << 8,
    SURFACE_BIT_DEBUG              = 1u << 9,
    SURFACE_BIT_ATM_HORIZON_CLEAR  = 1u << 10,
    SURFACE_BIT_ALL                = 0x7FFu,
};

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
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, std::span<const rhi::PushDescriptorWrite> scene,
                std::span<const uint32_t> body_shader_keys = {},
                VkFormat color_format = VK_FORMAT_R16G16B16A16_SFLOAT);
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
    struct GroupBatch {
        uint32_t mode = 0;
        uint32_t shader_key = SURFACE_BIT_ALL;
        std::array<std::pair<uint32_t, uint32_t>, 16> draws{};
        uint16_t mask_bits = 0;
    };
    struct KeyGroup {
        uint32_t mode = 0;
        uint32_t shader_key = SURFACE_BIT_ALL;
        std::array<uint32_t, 16> counts{};
        std::array<uint32_t, 16> offsets{};
    };
    struct FrameUploadState {
        uint64_t generation = 0;
        std::vector<uint32_t> body_keys;
        std::vector<GroupBatch> batches;
        uint32_t ref_count = 0;
        uint32_t cached_count = 0;
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
    rhi::Pipeline& pipeline_for(rhi::VkContext& ctx, uint32_t mode, uint32_t shader_key, VkFormat color_format);
    std::unordered_map<uint64_t, rhi::Pipeline> pipelines_;
    rhi::Buffer indices_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> patches_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> cached_patches_{};
    std::vector<PreparedPatch> prepared_;
    std::vector<KeyGroup> groups_;
    std::vector<uint16_t> patch_group_;
    std::array<FrameUploadState, rhi::FRAMES_IN_FLIGHT> frame_uploads_{};
    uint64_t prepared_generation_ = 0;
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
