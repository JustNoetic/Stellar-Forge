#pragma once

#include "core/camera.hpp"
#include "ephemeris/system_manager.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/ring_optics.hpp"
#include "rhi/staging_uploads.hpp"
#include "rhi/vk_context.hpp"
#include <array>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sf::rendering {

struct alignas(16) RingSegmentGPU {
    glm::vec4 bounds_opacity_scatter{0.0f};
    glm::vec4 phase_row{0.0f};
    glm::vec4 color_flags{1.0f, 1.0f, 1.0f, 0.0f};
};

struct alignas(16) RingHostGPU {
    glm::vec4 center_min_r{0.0f};
    glm::vec4 pole_max_r{0.0f, 1.0f, 0.0f, 0.0f};
    glm::vec4 tangent_radius{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 bitangent_oblateness{0.0f, 0.0f, 1.0f, 0.0f};
    glm::uvec4 counts_indices{0};
    RingSegmentGPU segments[MAX_RING_SEGMENTS]{};
};

static_assert(sizeof(RingSegmentGPU) == 48);
static_assert(sizeof(RingHostGPU) == 464);

class RingRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);

    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                 const ephemeris::SystemSnapshot& sys, uint32_t body_count);
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, const RenderSettings& settings,
                std::span<const rhi::PushDescriptorWrite> scene_writes, std::span<const glm::vec4> stars,
                std::span<const rhi::PushDescriptorWrite> transparency_writes = {});
    void update_ringshine(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                          const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, uint32_t body_count,
                          VkExtent2D extent);

    void invalidate_texture_cache() noexcept {
        force_rebuild_ = true;
        cached_raw_textures_.clear();
    }

    void set_custom_ring_texture(std::string_view system_name, std::string_view body_name, std::vector<glm::vec4> texture_samples) {
        std::string sys = system_name.empty() ? "Solar System" : std::string(system_name);
        std::string cache_key = sys + "/" + std::string(body_name);
        cached_raw_textures_[cache_key] = std::move(texture_samples);
        force_rebuild_ = true;
    }

    bool has_custom_ring_texture(std::string_view system_name, std::string_view body_name) const noexcept {
        std::string sys = system_name.empty() ? "Solar System" : std::string(system_name);
        std::string cache_key = sys + "/" + std::string(body_name);
        auto it = cached_raw_textures_.find(cache_key);
        return it != cached_raw_textures_.end() && !it->second.empty();
    }

    uint32_t host_count() const noexcept { return active_host_count_; }
    std::span<const RingHostGPU> hosts() const noexcept { return hosts_gpu_; }
    const rhi::Buffer& host_buffer(uint32_t frame) const noexcept { return host_ssbos_[frame]; }
    const rhi::Image& gradient_atlas() const noexcept { return gradient_atlas_; }
    const rhi::Image& shadow_lut() const noexcept { return shadow_lut_; }
    uint64_t shadow_revision() const noexcept { return ringshine_profile_serial_; }
    const rhi::Image& ringshine_atlas(uint32_t frame) const noexcept { return ringshine_atlases_[frame]; }
    uint32_t ringshine_bake_count() const noexcept { return ringshine_bakes_; }
    uint32_t ringshine_frame_bake_count() const noexcept { return ringshine_frame_bakes_; }
    std::span<const uint32_t> ringshine_sample_counts() const noexcept { return ringshine_sample_counts_; }

private:
    struct CachedHostMapping {
        uint32_t body_index = 0;
        uint32_t unified_row = 0;
        double min_r_ratio = 1.5;
        double max_r_ratio = 2.5;
        std::vector<uint32_t> segment_rows;
        std::vector<RingDrawRange> draw_ranges;
    };

    struct RingshineTileKey {
        uint64_t profile_serial = 0;
        double inner = 0.0;
        double outer = 0.0;
        double flattening = 0.0;
        uint32_t bands = 0;
        double elevation = 0.0;
        double sin_radius = 0.0;
        bool operator==(const RingshineTileKey&) const = default;
    };
    struct RingshineJob {
        uint32_t host_idx = 0;
        uint32_t star_col = 0;
        RingshineTileKey key{};
    };

    void build_annulus_mesh(rhi::VkContext& ctx, uint32_t angular_segments, uint32_t radial_subdivisions);
    void rebuild_profiles(VkCommandBuffer cmd, rhi::VkContext& ctx, const ephemeris::SystemSnapshot& sys, uint32_t body_count);
    void bake_ringshine_tile(VkCommandBuffer cmd, rhi::VkContext& ctx, const RingshineJob& job);

    rhi::Buffer vbo_{};
    rhi::Buffer ibo_{};
    uint32_t index_count_ = 0;

    std::array<rhi::Pipeline, 16> pipelines_{};
    std::array<rhi::Pipeline, 16> transparency_pipelines_{};
    rhi::Image gradient_atlas_{};
    rhi::Image shadow_lut_{};
    std::array<rhi::Image, rhi::FRAMES_IN_FLIGHT> ringshine_atlases_{};
    rhi::Pipeline ringshine_pipeline_{};
    std::array<std::map<std::pair<uint32_t, uint32_t>, RingshineTileKey>, rhi::FRAMES_IN_FLIGHT> ringshine_cache_;
    uint32_t ringshine_bakes_ = 0;
    uint32_t ringshine_frame_bakes_ = 0;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> ringshine_radial_buffers_{};
    std::array<uint64_t, rhi::FRAMES_IN_FLIGHT> ringshine_radial_upload_serials_{};
    std::vector<glm::vec2> ringshine_radial_samples_;
    std::vector<glm::uvec2> ringshine_radial_ranges_;
    std::vector<uint32_t> ringshine_sample_counts_;
    uint64_t ringshine_radial_serial_ = 0;
    uint64_t ringshine_radial_profile_serial_ = 0;
    uint32_t ringshine_radial_bands_ = 0;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> host_ssbos_{};
    rhi::StagingUploads staging_;

    std::vector<RingHostGPU> hosts_gpu_;
    std::vector<std::span<const RingDrawRange>> host_draw_ranges_;
    std::vector<CachedHostMapping> host_mappings_;
    std::unordered_map<std::string, std::vector<glm::vec4>> cached_raw_textures_;
    uint32_t active_host_count_ = 0;
    std::shared_ptr<const ephemeris::SystemDefinition> cached_definition_;
    uint64_t cached_generation_ = 0;
    uint64_t cached_definition_revision_ = 0;
    uint64_t ringshine_profile_serial_ = 0;
    bool force_rebuild_ = true;
};

} // namespace sf::rendering
