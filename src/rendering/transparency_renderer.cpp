#include "rendering/transparency_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <cstring>

namespace sf::rendering {

void TransparencyRenderer::init(rhi::VkContext& ctx) {
    if (!ctx.fragment_stores_enabled())
        sf::log::warn("Fragment storage atomics are unavailable; transparent objects use legacy draw-order blending.");
    std::vector<rhi::DescriptorLayoutBinding> bindings{
        {14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {45, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {46, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}};
    resolve_ = rhi::ShaderCache::create_compute_pipeline(ctx, "transparency/resolve.comp", bindings, sizeof(uint32_t));
    bindings.push_back({5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT});
    const std::array macros{std::pair<std::string, std::string>{"BLOOM_OUTPUT", "1"}};
    bloom_resolve_ = rhi::ShaderCache::create_compute_pipeline(ctx, "transparency/resolve.comp", bindings, sizeof(uint32_t), macros);
    for (auto& buffer : counters_) buffer = ctx.create_host_buffer(16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

void TransparencyRenderer::shutdown(rhi::VkContext& ctx) {
    ctx.destroy_image(heads_);
    ctx.destroy_image(slot_color_);
    ctx.destroy_image(slot_depth_);
    ctx.destroy_buffer(fragments_);
    for (auto& entry : retired_) {
        ctx.destroy_image(entry.image);
        ctx.destroy_buffer(entry.buffer);
    }
    retired_.clear();
    for (auto& buffer : counters_) ctx.destroy_buffer(buffer);
    rhi::ShaderCache::destroy_pipeline(ctx, resolve_);
    rhi::ShaderCache::destroy_pipeline(ctx, bloom_resolve_);
}

void TransparencyRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, VkExtent2D extent, bool enabled) {
    std::erase_if(retired_, [&](auto& entry) {
        if (entry.serial > ctx.completed_serial()) return false;
        ctx.destroy_image(entry.image);
        ctx.destroy_buffer(entry.buffer);
        return true;
    });
    auto previous = stats(ctx);
    if (previous.overflow > 0 && last_overflow_ == 0)
        sf::log::warn("Transparency fragment budget exceeded by {} fragments; excess fragments use draw-order blending.", previous.overflow);
    last_overflow_ = previous.overflow;
    active_ = enabled && ctx.fragment_stores_enabled();
    last_stats_ = active_ ? previous : TransparencyStats{};
    if (!active_) return;
    auto timer = ctx.begin_gpu_timer(cmd, "transparency_clear");
    if (heads_.extent.width != extent.width || heads_.extent.height != extent.height) {
        for (auto* image : {&heads_, &slot_color_, &slot_depth_})
            if (*image) retired_.push_back({*image, {}, ctx.recording_serial()});
        heads_ = ctx.create_image_2d(extent.width, extent.height, VK_FORMAT_R32_UINT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        slot_color_ = ctx.create_image_2d(extent.width, extent.height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
        slot_depth_ = ctx.create_image_2d(extent.width, extent.height, VK_FORMAT_R32_UINT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }
    uint32_t capacity = uint32_t(std::min<uint64_t>(uint64_t(extent.width)*extent.height*2, 8*1024*1024));
    uint32_t max_cap = uint32_t(((ctx.device_properties().limits.maxStorageBufferRange - 16) / 16) & ~3u);
    capacity = std::min(capacity, max_cap);
    uint32_t stride = (capacity + 3u) & ~3u;
    if (capacity > capacity_) {
        if (fragments_) retired_.push_back({{}, fragments_, ctx.recording_serial()});
        capacity_ = capacity;
        fragments_ = ctx.create_device_buffer(16 + VkDeviceSize(stride) * 16,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    for (auto* image : {&heads_, &slot_depth_})
        ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    ctx.transition_image(cmd, slot_color_, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    ctx.memory_barrier(cmd,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue clear_val{};
    clear_val.uint32[0] = 0xffffffffu;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, heads_.handle, VK_IMAGE_LAYOUT_GENERAL, &clear_val, 1, &range);
    // Only the slot depth is cleared: colour is read only where a ring fragment claimed the slot.
    VkClearColorValue zero{};
    vkCmdClearColorImage(cmd, slot_depth_.handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    const uint32_t header[4]{0u, 0u, capacity, stride};
    vkCmdUpdateBuffer(cmd, fragments_.handle, 0, sizeof(header), header);
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    ctx.end_gpu_timer(cmd, timer);
}

std::array<rhi::PushDescriptorWrite, 2> TransparencyRenderer::writes() const {
    return {rhi::PushDescriptorWrite::storage_image(14, heads_),
            rhi::PushDescriptorWrite::storage_buffer(15, fragments_)};
}

std::array<rhi::PushDescriptorWrite, 4> TransparencyRenderer::slot_writes() const {
    return {rhi::PushDescriptorWrite::storage_image(14, heads_),
            rhi::PushDescriptorWrite::storage_buffer(15, fragments_),
            rhi::PushDescriptorWrite::storage_image(45, slot_color_),
            rhi::PushDescriptorWrite::storage_image(46, slot_depth_)};
}

void TransparencyRenderer::resolve(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                                   rhi::Image* physical_color, uint32_t slot_mode) {
    if (!active_) return;
    auto timer = ctx.begin_gpu_timer(cmd, "transparency_resolve");
    ctx.transition_image(cmd, color, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    ctx.transition_image(cmd, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    if (physical_color) {
        ctx.transition_image(cmd, *physical_color, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    auto& pipeline = physical_color ? bloom_resolve_ : resolve_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    auto resources = slot_writes();
    std::vector<rhi::PushDescriptorWrite> descriptors(resources.begin(), resources.end());
    descriptors.push_back(rhi::PushDescriptorWrite::storage_image(4, color));
    descriptors.push_back(rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()));
    if (physical_color) descriptors.push_back(rhi::PushDescriptorWrite::storage_image(5, *physical_color));
    ctx.push_descriptors(cmd, pipeline, descriptors);
    ctx.push_constants(cmd, pipeline, slot_mode);
    vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
    ctx.memory_barrier(cmd);
    auto frame = ctx.frame_index();
    ctx.copy_buffer(cmd, fragments_, counters_[frame], 16);
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    counter_serials_[frame] = ctx.recording_serial();
    ctx.end_gpu_timer(cmd, timer);
}

TransparencyStats TransparencyRenderer::stats(rhi::VkContext& ctx) const {
    uint32_t frame = 0;
    uint64_t latest = 0;
    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        if (counter_serials_[i] > latest && counter_serials_[i] <= ctx.completed_serial()) {
            latest = counter_serials_[i];
            frame = i;
        }
    }
    TransparencyStats result;
    result.capacity = capacity_;
    result.bytes = fragments_.size+VkDeviceSize(heads_.extent.width)*heads_.extent.height*16;
    if (latest != 0) {
        VK_CHECK(vmaInvalidateAllocation(ctx.allocator(), counters_[frame].allocation, 0, VK_WHOLE_SIZE));
        uint32_t values[4];
        std::memcpy(values, counters_[frame].mapped, sizeof(values));
        result.fragments = values[0];
        result.overflow = values[1];
        result.capacity = values[2];
    }
    return result;
}

}
