#pragma once

#include "core/camera.hpp"
#include "ephemeris/system_manager.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/ring_optics.hpp"
#include "rendering/planetshine.hpp"
#include "rhi/staging_uploads.hpp"
#include "rhi/vk_context.hpp"
#include <algorithm>
#include <array>
#include <cmath>
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

// Ring penumbra table (shaders/common/ring_penumbra.glsl): layers of star-disk-filtered ring
// extinction, one per (ring LUT layer, star incidence). RingHostGPU::counts_indices.w packs, per
// scene star s < RING_PENUMBRA_STARS, the slot + 1 in bits [4s, 4s+4) (0 = none). Layers persist
// across frames and are rebuilt only when re-keyed or when the ring LUT changes.
inline constexpr uint32_t RING_PENUMBRA_WIDTH = 6144;
inline constexpr uint32_t RING_PENUMBRA_ROWS = 56;
inline constexpr uint32_t RING_PENUMBRA_SLOTS = 8;
inline constexpr uint32_t RING_PENUMBRA_STARS = 8;
struct RingPenumbraSlots {
    struct Slot {
        glm::vec2 key{-1.0f, 0.0f};  // x = ring LUT layer, y = mu coordinate built at
        uint64_t used = 0;
        uint64_t built_revision = 0;
        bool built = false;
    };
    std::array<Slot, RING_PENUMBRA_SLOTS> slots{};
    uint64_t serial = 0;

    // Incidence snapped to a fixed grid (steps of ~1.7e-4 relative) and clamped to the prefix
    // LUT's rows, so a slot depends only on the current frame, never on which incidence an
    // earlier frame built; the shaders accept 2e-3.
    static float key_mu(double mu) {
        float snapped = std::exp2(std::round(std::log2(float(std::max(mu, 1e-6)))*4096.0f)/4096.0f);
        return std::clamp(snapped, float(0.5/64.0), float(63.5/64.0));
    }

    // Slot for (layer, key mu); -1 when all are taken this frame.
    int acquire(float layer, float mu) {
        for (size_t i = 0; i < slots.size(); ++i)
            if (slots[i].key.x == layer && slots[i].key.y == mu) {
                slots[i].used = serial;
                return int(i);
            }
        int victim = -1;
        for (size_t i = 0; i < slots.size(); ++i)
            if (slots[i].used != serial && (victim < 0 || slots[i].used < slots[size_t(victim)].used)) victim = int(i);
        if (victim >= 0) slots[size_t(victim)] = {{layer, mu}, serial, 0, false};
        return victim;
    }
};

class RingRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);

    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                 const ephemeris::SystemSnapshot& sys, uint32_t body_count, int refraction_body = -1);
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, const RenderSettings& settings,
                std::span<const rhi::PushDescriptorWrite> scene_writes, std::span<const glm::vec4> stars,
                std::span<const rhi::PushDescriptorWrite> transparency_writes = {});
    void update_ringshine(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                          const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, uint32_t body_count,
                          VkExtent2D extent);
    void update_ring_planetshine(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                                 const ephemeris::SystemSnapshot& sys, const RenderSettings& settings,
                                 uint32_t body_count, VkExtent2D extent, const SurfaceMaterialCache& materials);

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
    bool drawn(const RenderSettings& settings) const noexcept { return settings.show_rings && active_host_count_ != 0; }
    std::span<const RingHostGPU> hosts() const noexcept { return hosts_gpu_; }
    const rhi::Buffer& host_buffer(uint32_t frame) const noexcept { return host_ssbos_[frame]; }
    const rhi::Image& gradient_atlas() const noexcept { return gradient_atlas_; }
    const rhi::Image& shadow_lut() const noexcept { return shadow_lut_; }
    uint64_t shadow_revision() const noexcept { return ringshine_profile_serial_; }
    const rhi::Image& penumbra_table() const noexcept { return penumbra_table_; }
    const RingPenumbraSlots& penumbra_slots() const noexcept { return penumbra_slots_; }
    const rhi::Image& ringshine_atlas(uint32_t frame) const noexcept { return ringshine_atlases_[frame]; }
    uint32_t ringshine_bake_count() const noexcept { return ringshine_bakes_; }
    uint32_t ringshine_frame_bake_count() const noexcept { return ringshine_frame_bakes_; }
    uint32_t ring_planetshine_bake_count() const noexcept { return ring_planetshine_bakes_; }
    uint32_t ring_planetshine_frame_bake_count() const noexcept { return ring_planetshine_frame_bakes_; }
    std::span<const uint32_t> ringshine_sample_counts() const noexcept { return ringshine_sample_counts_; }
    PlanetshineGPU atmosphere_sources(const ephemeris::SystemSnapshot& system,
                                     const RenderSettings& settings, uint32_t body_index) const;
    uint64_t atmosphere_source_bake_count() const noexcept { return atmosphere_source_bakes_; }

private:
    struct CachedHostMapping {
        uint32_t body_index = 0;
        uint32_t unified_row = 0;
        double min_r_ratio = 1.5;
        double max_r_ratio = 2.5;
        std::vector<uint32_t> segment_rows;
        std::vector<RingDrawRange> draw_ranges;
        std::vector<AtmosphereRingMaterial> atmosphere_materials;
        mutable PlanetshineGPU atmosphere_sources{};
        mutable std::array<glm::vec4, 16> atmosphere_stars{}, atmosphere_weights{};
        mutable bool atmosphere_valid = false;
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
    struct RingPlanetshineTileKey {
        double flattening = 0.0;
        double elevation = 0.0;
        double sin_radius = 0.0;
        physics::SurfaceMaterialGPU material{};
        bool operator==(const RingPlanetshineTileKey& o) const noexcept {
            return flattening == o.flattening && elevation == o.elevation && sin_radius == o.sin_radius
                && material.grain == o.material.grain && material.surface == o.material.surface
                && glm::vec3(material.opposition) == glm::vec3(o.material.opposition);
        }
    };
    struct RingPlanetshineJob {
        uint32_t host_idx = 0;
        uint32_t star_col = 0;
        RingPlanetshineTileKey key{};
    };

    struct AnnulusMesh {
        rhi::Buffer vertices{}, indices{};
        uint32_t index_count = 0;
    };
    AnnulusMesh build_annulus_mesh(rhi::VkContext& ctx, uint32_t angular_segments, uint32_t radial_subdivisions);
    void update_penumbra(VkCommandBuffer cmd, rhi::VkContext& ctx, const ephemeris::SystemSnapshot& sys, uint32_t body_count);
    void rebuild_profiles(VkCommandBuffer cmd, rhi::VkContext& ctx, const ephemeris::SystemSnapshot& sys, uint32_t body_count);
    void bake_ringshine_tile(VkCommandBuffer cmd, rhi::VkContext& ctx, const RingshineJob& job);
    void ensure_ring_planetshine_atlas(VkCommandBuffer cmd, rhi::VkContext& ctx, uint32_t frame);
    void bake_ring_planetshine_tile(VkCommandBuffer cmd, rhi::VkContext& ctx, const RingPlanetshineJob& job);
    static constexpr uint32_t PIPELINE_TABLE_COUNT = 12;
    rhi::Pipeline& select_pipeline(rhi::VkContext& ctx, uint32_t variant_idx, bool transparency,
                                   bool planetshine, bool numerical, bool refracted);

    AnnulusMesh mesh_{};
    int refraction_body_ = -1;

    std::array<std::array<rhi::Pipeline, 16>, PIPELINE_TABLE_COUNT> pipeline_tables_{};
    rhi::Image gradient_atlas_{};
    rhi::Image shadow_lut_{};
    rhi::Pipeline penumbra_pipeline_{};
    rhi::Image penumbra_table_{};
    RingPenumbraSlots penumbra_slots_;
    std::array<rhi::Image, rhi::FRAMES_IN_FLIGHT> ringshine_atlases_{};
    rhi::Pipeline ringshine_pipeline_{};
    std::array<std::map<std::pair<uint32_t, uint32_t>, RingshineTileKey>, rhi::FRAMES_IN_FLIGHT> ringshine_cache_;
    uint32_t ringshine_bakes_ = 0;
    uint32_t ringshine_frame_bakes_ = 0;
    rhi::Pipeline ring_planetshine_pipeline_{};
    rhi::Image ring_planetshine_dummy_{};
    std::array<rhi::Image, rhi::FRAMES_IN_FLIGHT> ring_planetshine_atlases_{};
    std::array<std::map<std::pair<uint32_t, uint32_t>, RingPlanetshineTileKey>, rhi::FRAMES_IN_FLIGHT> ring_planetshine_cache_;
    std::vector<RingPlanetshineJob> ring_planetshine_jobs_;
    std::vector<rhi::PushDescriptorWrite> render_writes_;
    uint32_t ring_planetshine_bakes_ = 0;
    uint32_t ring_planetshine_frame_bakes_ = 0;
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
    struct HostKey {
        uint32_t body_index = 0;
        uint64_t identity = 0;
        uint64_t definition_revision = 0;
        bool operator==(const HostKey&) const = default;
    };
    std::vector<HostKey> cached_hosts_, host_keys_;
    uint64_t ringshine_profile_serial_ = 0;
    bool force_rebuild_ = true;
    mutable uint64_t atmosphere_source_bakes_ = 0;
};

} // namespace sf::rendering
