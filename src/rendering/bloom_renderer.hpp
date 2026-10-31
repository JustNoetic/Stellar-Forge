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
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color,
                const RenderSettings& settings, rhi::Image* physical_scene = nullptr);
    rhi::Image& physical_scene() { return physical_scene_; }
    static bool enabled(const RenderSettings& settings) {
        return settings.hdr_enabled && settings.bloom_mode != 0 && settings.bloom_strength > 0.0f;
    }

private:
    static constexpr size_t LEVELS = 6;
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
