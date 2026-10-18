#pragma once

#include "core/camera.hpp"
#include "rendering/render_settings.hpp"
#include "rhi/vk_context.hpp"
#include <array>
#include <memory>
#include <vector>

namespace sf::rendering {

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
    glm::vec4 aerial_params{};
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
    uint32_t bruneton_lut_updates = 0;
    uint32_t surface_lut_updates = 0;
};

class AtmosphereRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                 const ephemeris::SystemSnapshot& system, const RenderSettings& settings, VkExtent2D extent,
                 std::span<const float> minimum_heights_km = {});
    void composite(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                   std::span<const rhi::PushDescriptorWrite> transparency_writes = {});
    const AtmosphereStats& stats() const noexcept { return stats_; }
    const rhi::Image& surface_lut() const noexcept { return surface_lut_; }
    const rhi::Buffer& surface_records(uint32_t frame) const noexcept { return surface_records_[frame]; }

private:
    struct Tables {
        rhi::Image transmittance, multiscatter, sky, throughput;
        rhi::Image aerial_radiance, aerial_optical_depth;
        rhi::Image bruneton_rayleigh, bruneton_mie;
        std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> uniforms;
        AtmosphereParametersGPU cached{};
        std::shared_ptr<const ephemeris::SystemDefinition> definition;
        uint64_t generation = 0, geometry_revision = 0, lighting_revision = 0;
        uint32_t body_index = 0;
        bool valid = false;
        bool aerial_valid = false;
        bool bruneton_valid = false;
    };
    std::vector<std::unique_ptr<Tables>> tables_;
    std::vector<Tables*> active_;
    rhi::Pipeline transmittance_pipeline_, multiscatter_pipeline_, sky_pipeline_, composite_pipeline_;
    rhi::Pipeline aerial_pipeline_, bruneton_pipeline_;
    rhi::Pipeline dual_source_pipeline_, surface_pipeline_;
    rhi::Pipeline transparent_pipeline_, composite_transparency_pipeline_;
    rhi::Image surface_lut_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> surface_records_;
    bool dual_source_ = false;
    AtmosphereStats stats_;
};

}
