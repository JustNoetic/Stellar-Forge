#pragma once

#include "rendering/render_settings.hpp"
#include "rhi/vk_context.hpp"
#include <array>

namespace sf::rendering {

class BloomRenderer {
public:
    static constexpr uint32_t SPIKE_CAPACITY = 2048;

    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void prepare(rhi::VkContext& ctx, VkExtent2D extent);
    // With defer_resolve and a physical-light image, the halo correction is left
    // for the presentation pass (bloom_resolve in composite.frag) and true is
    // returned. Otherwise it is applied to color before the spikes are drawn.
    bool render(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color,
                const RenderSettings& settings, rhi::Image* physical_scene = nullptr, bool defer_resolve = false);
    // Standalone form of the deferred correction, for verification.
    void resolve(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color,
                 const RenderSettings& settings, rhi::Image& physical_scene);
    rhi::Image& physical_scene() { return physical_scene_; }
    rhi::Image& halo() { return up_[0]; }
    static bool enabled(const RenderSettings& settings) {
        return settings.hdr_enabled && settings.bloom_mode != 0 && settings.bloom_strength > 0.0f;
    }

private:
    void apply(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& source,
               const RenderSettings& settings);
    // Quarter resolution down to 1/64.
    static constexpr size_t LEVELS = 5;
    rhi::Pipeline extract_, downsample_, upsample_, apply_, spike_;
    rhi::Image physical_scene_;
    std::array<rhi::Image, LEVELS> down_;
    std::array<rhi::Image, LEVELS-1> up_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> spikes_;
    float spike_length_ = 0, spike_width_ = 0, spike_normalization_ = 1;
    struct Retired { rhi::Image image; uint64_t serial; };
    std::vector<Retired> retired_;
};

}
