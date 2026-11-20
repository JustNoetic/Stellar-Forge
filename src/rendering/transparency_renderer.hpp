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
    // slot_mode 0 ignores the ring slot, 1 merges it only where list fragments exist (the atmosphere composite
    // blended the rest), 2 merges it everywhere.
    void resolve(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                 rhi::Image* physical_color = nullptr, uint32_t slot_mode = 0);
    bool active() const noexcept { return active_; }
    std::array<rhi::PushDescriptorWrite, 2> writes() const;
    std::array<rhi::PushDescriptorWrite, 4> slot_writes() const;
    const rhi::Buffer& fragments() const noexcept { return fragments_; }
    uint32_t capacity() const noexcept { return capacity_; }
    TransparencyStats stats(rhi::VkContext& ctx) const;
    const TransparencyStats& last_stats() const noexcept { return last_stats_; }

private:
    rhi::Image heads_, slot_color_, slot_depth_;
    rhi::Buffer fragments_;
    rhi::Pipeline resolve_, bloom_resolve_, resolve_fp32_;
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
    TransparencyStats last_stats_{};
};

}
