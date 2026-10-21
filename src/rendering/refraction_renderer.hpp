#pragma once

#include "core/camera.hpp"
#include "ephemeris/system_manager.hpp"
#include "rendering/refraction_tables.hpp"
#include "rendering/render_settings.hpp"
#include "rhi/staging_uploads.hpp"
#include "rhi/vk_context.hpp"
#include <array>
#include <memory>
#include <vector>

namespace sf::rendering {

class RefractionFallback {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    std::array<rhi::PushDescriptorWrite, 2> writes(rhi::VkContext& ctx) const;

private:
    rhi::Buffer ubo_{};
    rhi::Image observer_{};
};

class RefractionRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                 const ephemeris::SystemSnapshot& system, const RenderSettings& settings);
    std::array<rhi::PushDescriptorWrite, 2> writes(rhi::VkContext& ctx) const;
    const RefractionHost& host() const noexcept { return host_; }
    std::array<rhi::PushDescriptorWrite, 3> ray_writes(rhi::VkContext& ctx) const;
    uint32_t observer_builds() const noexcept { return observer_builds_; }
    const rhi::Image& observer(uint32_t frame) const noexcept { return observers_[frame]; }
    const RefractionGPU& parameters(uint32_t frame) const noexcept { return built_[frame]; }

private:
    struct Tables {
        std::shared_ptr<const physics::PreparedAtmosphere> atmosphere;
        rhi::Image travel{}, rows{}, profile{};
        uint64_t last_serial = 0;
        bool ready = false;
    };

    Tables* tables_for(VkCommandBuffer cmd, rhi::VkContext& ctx, const std::shared_ptr<const physics::PreparedAtmosphere>& atmosphere);
    void retire_tables(rhi::VkContext& ctx, const physics::PreparedAtmosphere* keep);

    rhi::StagingUploads staging_;
    std::vector<Tables> tables_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> ubos_{};
    std::array<rhi::Image, rhi::FRAMES_IN_FLIGHT> observers_{};
    std::array<RefractionGPU, rhi::FRAMES_IN_FLIGHT> built_{};
    std::array<const physics::PreparedRefraction*, rhi::FRAMES_IN_FLIGHT> built_source_{};
    rhi::Pipeline pipeline_{};
    RefractionHost host_{};
    uint32_t observer_builds_ = 0;
};

}
