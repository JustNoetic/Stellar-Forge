#pragma once

#include "ephemeris/system_manager.hpp"
#include "rendering/render_settings.hpp"
#include "rhi/vk_context.hpp"
#include <memory>

namespace sf::rendering {

class EclipseRefractionRenderer {
public:
    EclipseRefractionRenderer();
    ~EclipseRefractionRenderer();
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const ephemeris::SystemSnapshot& system,
                 const RenderSettings& settings);
    rhi::PushDescriptorWrite write(rhi::VkContext& ctx) const;
    std::span<const glm::vec4> records() const;
    std::span<const glm::vec2> bounds() const;
    uint32_t builds() const;
    double last_bake_ms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
