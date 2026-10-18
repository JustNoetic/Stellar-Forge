#pragma once

#include "rhi/vk_context.hpp"
#include <array>

namespace sf::rendering {

struct TransparencyStats {
    uint32_t fragments = 0;
    uint32_t overflow = 0;
    uint32_t capacity = 0;
    VkDeviceSize bytes = 0;
};

class TransparencyRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, VkExtent2D extent, bool enabled);
    void resolve(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color);
    bool active() const noexcept { return active_; }
    std::array<rhi::PushDescriptorWrite, 2> writes() const;
    const rhi::Buffer& fragments() const noexcept { return fragments_; }
    uint32_t capacity() const noexcept { return capacity_; }
    TransparencyStats stats(rhi::VkContext& ctx) const;

private:
    rhi::Image heads_;
    rhi::Buffer fragments_;
    rhi::Pipeline clear_, resolve_;
    uint32_t capacity_ = 0;
    bool active_ = false;
    struct Retired {
        rhi::Image image;
        rhi::Buffer buffer;
        uint64_t serial;
    };
    std::vector<Retired> retired_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> counters_;
    std::array<uint64_t, rhi::FRAMES_IN_FLIGHT> counter_serials_{};
    uint32_t last_overflow_ = 0;
};

}
