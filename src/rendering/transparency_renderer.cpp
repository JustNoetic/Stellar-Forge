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
        {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    clear_ = rhi::ShaderCache::create_compute_pipeline(ctx, "transparency/clear.comp", bindings, sizeof(uint32_t));
    bindings.push_back({4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT});
    bindings.push_back({3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT});
    resolve_ = rhi::ShaderCache::create_compute_pipeline(ctx, "transparency/resolve.comp", bindings);
    for (auto& buffer : counters_) buffer = ctx.create_host_buffer(16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

void TransparencyRenderer::shutdown(rhi::VkContext& ctx) {
    ctx.destroy_image(heads_);
    ctx.destroy_buffer(fragments_);
    for (auto& entry : retired_) {
        ctx.destroy_image(entry.image);
        ctx.destroy_buffer(entry.buffer);
    }
    retired_.clear();
    for (auto& buffer : counters_) ctx.destroy_buffer(buffer);
    rhi::ShaderCache::destroy_pipeline(ctx, clear_);
    rhi::ShaderCache::destroy_pipeline(ctx, resolve_);
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
    if (!active_) return;
    auto timer = ctx.begin_gpu_timer(cmd, "transparency_clear");
    if (heads_.extent.width != extent.width || heads_.extent.height != extent.height) {
        if (heads_) retired_.push_back({heads_, {}, ctx.recording_serial()});
        heads_ = ctx.create_image_2d(extent.width, extent.height, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT);
    }
    uint32_t capacity = uint32_t(std::min<uint64_t>(uint64_t(extent.width)*extent.height*2, 8*1024*1024));
    capacity = std::min(capacity, uint32_t((ctx.device_properties().limits.maxStorageBufferRange-16)/16));
    if (capacity > capacity_) {
        if (fragments_) retired_.push_back({{}, fragments_, ctx.recording_serial()});
        capacity_ = capacity;
        fragments_ = ctx.create_device_buffer(16+VkDeviceSize(capacity_)*16,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    }
    ctx.transition_image(cmd, heads_, VK_IMAGE_LAYOUT_GENERAL);
    ctx.memory_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clear_.handle);
    ctx.push_descriptors(cmd, clear_, writes());
    ctx.push_constants(cmd, clear_, capacity);
    vkCmdDispatch(cmd, (extent.width+7)/8, (extent.height+7)/8, 1);
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    ctx.end_gpu_timer(cmd, timer);
}

std::array<rhi::PushDescriptorWrite, 2> TransparencyRenderer::writes() const {
    return {rhi::PushDescriptorWrite::storage_image(14, heads_),
            rhi::PushDescriptorWrite::storage_buffer(15, fragments_)};
}

void TransparencyRenderer::resolve(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth) {
    if (!active_) return;
    auto timer = ctx.begin_gpu_timer(cmd, "transparency_resolve");
    ctx.transition_image(cmd, color, VK_IMAGE_LAYOUT_GENERAL);
    ctx.transition_image(cmd, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ctx.memory_barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resolve_.handle);
    auto resources = writes();
    rhi::PushDescriptorWrite descriptors[] = {resources[0], resources[1], rhi::PushDescriptorWrite::storage_image(4, color),
        rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp())};
    ctx.push_descriptors(cmd, resolve_, descriptors);
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
    result.bytes = fragments_.size+VkDeviceSize(heads_.extent.width)*heads_.extent.height*4;
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
