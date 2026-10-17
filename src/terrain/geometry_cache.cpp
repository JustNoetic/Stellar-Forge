#include "terrain/geometry_cache.hpp"
#include "rhi/shader_cache.hpp"
#include <cstring>
#include <stdexcept>

namespace sf::terrain {

static_assert(sizeof(GeometryBakeGPU) == 96);

void GeometryCache::init(rhi::VkContext& ctx) {
    residency_ = GeometryResidency{GEOMETRY_CACHE_CAPACITY};
    bake_count_ = 0;
    pending_uploads_.clear();
    uploads_.init(ctx, 4 * 1024 * 1024);
    geometry_ = ctx.create_device_buffer(VkDeviceSize(GEOMETRY_CACHE_CAPACITY) * PATCH_VERTICES * GEOMETRY_VERTEX_BYTES,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    for (auto& buffer : jobs_)
        buffer = ctx.create_host_buffer(MAX_GEOMETRY_BAKES * sizeof(GeometryBakeGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
    };
    bake_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "terrain/geometry_cache.comp", bindings);
}

void GeometryCache::shutdown(rhi::VkContext& ctx) {
    uploads_.shutdown(ctx);
    rhi::ShaderCache::destroy_pipeline(ctx, bake_pipeline_);
    for (auto& buffer : jobs_) ctx.destroy_buffer(buffer);
    ctx.destroy_buffer(geometry_);
}

CachePlan GeometryCache::prepare(rhi::VkContext& ctx, std::span<const GeometryKey> keys, std::span<const GeometryBakeGPU> requests,
                                 std::span<const std::shared_ptr<const HeightGeometry>> heights) {
    auto plan = residency_.request(keys, MAX_GEOMETRY_BAKES);
    bake_count_ = 0;
    pending_uploads_.clear();
    auto* mapped = static_cast<GeometryBakeGPU*>(jobs_[ctx.frame_index()].mapped);
    for (const auto& bake : plan.bakes) {
        if (!heights.empty() && heights[bake.request]) {
            pending_uploads_.emplace_back(bake.slot, heights[bake.request]);
        } else {
            if (keys[bake.request].source.identity != 0) throw std::logic_error("Imported geometry admitted without a payload");
            auto job = requests[bake.request];
            job.destination.x = bake.slot;
            std::memcpy(mapped + bake_count_++, &job, sizeof(job));
        }
    }
    if (bake_count_ > 0) ctx.flush_buffer(jobs_[ctx.frame_index()], 0, bake_count_ * sizeof(GeometryBakeGPU));
    return plan;
}

void GeometryCache::bake(VkCommandBuffer cmd, rhi::VkContext& ctx) {
    if (bake_count_ == 0 && pending_uploads_.empty()) return;
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    for (const auto& [slot, data] : pending_uploads_) {
        if (!uploads_.buffer(cmd, ctx, geometry_, VkDeviceSize(slot) * sizeof(HeightGeometry), std::as_bytes(std::span{*data})))
            throw std::logic_error("Terrain staging budget exhausted");
    }
    pending_uploads_.clear();
    if (bake_count_ > 0) {
        const rhi::PushDescriptorWrite writes[] = {
            rhi::PushDescriptorWrite::storage_buffer(0, jobs_[ctx.frame_index()], 0, bake_count_ * sizeof(GeometryBakeGPU)),
            rhi::PushDescriptorWrite::storage_buffer(1, geometry_),
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bake_pipeline_.handle);
        ctx.push_descriptors(cmd, bake_pipeline_, writes);
        vkCmdDispatch(cmd, (GRID_CELLS + 8) / 8, (GRID_CELLS + 8) / 8, bake_count_);
    }
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    bake_count_ = 0;
}

}
