#pragma once

#include "core/camera.hpp"
#include "rendering/render_settings.hpp"
#include "rhi/vk_context.hpp"
#include <array>
#include <memory>
#include <vector>

namespace sf::rendering {

struct RingHostGPU;

struct alignas(16) AtmosphereShadowHeaderGPU {
    glm::vec4 eye_focal{};
    glm::uvec4 flags{};
    std::array<glm::vec4, 16> stars{};
    std::array<glm::uvec4, 16> ranges{};
    glm::uvec4 temporal{};
};

struct alignas(16) AtmosphereShadowCasterGPU {
    glm::vec4 position_radius{};
    glm::vec4 pole_inner{};
    glm::vec4 support{};
};

struct alignas(16) AtmosphereParametersGPU {
    glm::vec4 radii_heights{};
    glm::vec4 reference_shape{};
    glm::vec4 world_pole{};
    glm::vec4 rayleigh{};
    glm::vec4 mie_scattering{};
    glm::vec4 mie_extinction{};
    glm::vec4 ground_albedo{};
    glm::vec4 camera{};
    glm::vec4 axis_x{}, axis_y{}, axis_z{};
    std::array<glm::vec4, 16> sun_direction{};
    std::array<glm::vec4, 16> sun_irradiance{};
    glm::mat4 inverse_view_projection{1.0f};
    glm::vec4 depth_params{};
    glm::vec4 mixed_absorption{}, ozone_absorption{}, ozone_params{}, profile_domain{};
    std::array<glm::vec4, physics::ATMOSPHERE_PROFILE_SAMPLES> log_density{};
    glm::vec4 aerosol_count{};
    std::array<glm::vec4, ephemeris::MAX_AEROSOL_LAYERS> aerosol_shape{}, aerosol_scattering{}, aerosol_extinction{};
    std::array<glm::vec4, ephemeris::MAX_AEROSOL_LAYERS*physics::AEROSOL_PHASE_SAMPLES> aerosol_phase{};
};

struct AtmosphereStats {
    uint32_t visible = 0;
    uint32_t optical_lut_updates = 0;
    uint32_t sky_lut_updates = 0;
    uint32_t aerial_lut_updates = 0;
    uint32_t surface_lut_updates = 0;
    uint32_t shadow_updates = 0;
    uint32_t shadow_vrs_updates = 0;
    uint32_t shadow_history_resets = 0;
};

class AtmosphereRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                 const ephemeris::SystemSnapshot& system, const RenderSettings& settings, VkExtent2D extent,
                 std::span<const float> minimum_heights_km = {}, std::span<const RingHostGPU> rings = {},
                 const rhi::Image* ring_shadow_lut = nullptr, uint64_t ring_shadow_revision = 0);
    void composite(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                   std::span<const rhi::PushDescriptorWrite> transparency_writes = {});
    const AtmosphereStats& stats() const noexcept { return stats_; }
    const rhi::Image& surface_lut() const noexcept { return surface_lut_; }
    const rhi::Buffer& surface_records(uint32_t frame) const noexcept { return surface_records_[frame]; }

private:
    struct Tables {
        rhi::Image transmittance, multiscatter, sky, throughput;
        rhi::Image aerial_radiance, aerial_optical_depth, aerial_range;
        std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> uniforms;
        std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> shadows;
        rhi::Image shadow_fraction, shadow_depth;
        rhi::Image shadow_energy, shadow_loss;
        AtmosphereShadowHeaderGPU shadow_header{};
        std::vector<AtmosphereShadowCasterGPU> shadow_casters;
        bool shadow_active = false;
        bool shadow_history_valid = false, shadow_history_reusable = false;
        uint64_t shadow_recorded_prepare = 0, shadow_ring_revision = 0;
        VkImage shadow_ring_image = VK_NULL_HANDLE;
        AtmosphereParametersGPU cached{};
        std::shared_ptr<const ephemeris::SystemDefinition> definition;
        uint64_t generation = 0, geometry_revision = 0, lighting_revision = 0;
        uint64_t last_used = 0;
        uint32_t body_index = 0;
        bool valid = false;
        bool aerial_valid = false;
    };
    std::vector<std::unique_ptr<Tables>> tables_;
    std::vector<Tables*> active_;
    std::array<Tables*, 32> surface_layers_{};
    uint64_t prepare_serial_ = 0;
    rhi::Pipeline transmittance_pipeline_, multiscatter_pipeline_, sky_pipeline_, composite_pipeline_;
    rhi::Pipeline aerial_pipeline_;
    rhi::Pipeline dual_source_pipeline_, surface_pipeline_;
    rhi::Pipeline transparent_pipeline_, composite_transparency_pipeline_;
    rhi::Pipeline shadow_pipeline_, shadow_transparency_pipeline_;
    rhi::Pipeline shadow_accumulate_pipeline_, shadow_accumulate_transparency_pipeline_;
    rhi::Pipeline shadow_coarse_pipeline_, shadow_vrs_pipeline_, shadow_vrs_transparency_pipeline_;
    struct RetiredShadow { rhi::Image image; uint64_t serial; };
    std::vector<RetiredShadow> retired_shadows_;
    rhi::Image shadow_coarse_fraction_, shadow_coarse_guide_;
    rhi::Image surface_lut_;
    rhi::Image stbn_tex_;
    rhi::Image dummy_stbn_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> surface_records_;
    bool dual_source_ = false;
    bool shadow_vrs_enabled_ = true;
    bool shadow_temporal_enabled_ = false;
    const rhi::Image* ring_shadow_lut_ = nullptr;
    AtmosphereStats stats_;
};

}
