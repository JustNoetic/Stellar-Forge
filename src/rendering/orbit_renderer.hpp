#pragma once

#include "core/camera.hpp"
#include "ephemeris/orbit_curves.hpp"
#include "rendering/render_settings.hpp"
#include "rhi/vk_context.hpp"
#include <array>

namespace sf::rendering {

class OrbitRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                const ephemeris::SystemSnapshot& snapshot, const RenderSettings& settings, VkExtent2D extent,
                std::span<const rhi::PushDescriptorWrite> transparency_writes = {});

private:
    rhi::Pipeline pipeline_{};
    rhi::Pipeline transparency_pipeline_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> buffers_{};
    std::vector<ephemeris::OrbitCurve> curves_;
    uint64_t generation_ = 0, state_revision_ = 0, definition_revision_ = 0, hierarchy_revision_ = 0;
};

}
