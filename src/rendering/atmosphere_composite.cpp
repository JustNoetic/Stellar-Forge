#include "rendering/atmosphere_internal.hpp"
#include <algorithm>
#include <vector>

namespace sf::rendering {

void AtmosphereRenderer::composite(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                                   std::span<const rhi::PushDescriptorWrite> transparency_writes,
                                   const std::function<void(uint32_t)>& draw_point,
                                   std::span<const rhi::PushDescriptorWrite> cloud_writes) {
    if (active_.empty()) return;
    auto clouds = cloud_layer_writes(ctx, cloud_writes);
    ctx.transition_image(cmd, color, dual_source_ ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL);
    ctx.transition_image(cmd, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    uint32_t timer = ctx.begin_gpu_timer(cmd, "atmosphere_composite");
    if (!transparency_writes.empty()) ctx.memory_barrier(cmd);
    for (auto* t : active_)
        if (t->shadow_active) resolve_shadows(cmd, ctx, *t, color, depth, transparency_writes, clouds);
    if (shadow_diagnostics_enabled_ && shadow_temporal_enabled_ && stats_.shadow_updates != 0) {
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        auto& diagnostics = shadow_diagnostics_[ctx.frame_index()];
        diagnostics.serial = ctx.recording_serial();
        diagnostics.reset_reasons = stats_.shadow_history_reset_reasons;
    }
    if (dual_source_ && !transparency_writes.empty()) {
        uint32_t transparent_timer = ctx.begin_gpu_timer(cmd, "atmosphere_transparent");
        for (auto* t : active_) {
            if (t->cached.depth_params.w <= 0.0f) continue;
            auto& stbn = stbn_tex_ ? stbn_tex_ : dummy_stbn_;
            std::vector<rhi::PushDescriptorWrite> writes{
                rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
                rhi::PushDescriptorWrite::sampled_image(1, t->sky, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(2, t->throughput, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(5, t->aerial_radiance, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(6, t->aerial_optical_depth, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(7, t->transmittance, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(8, t->multiscatter, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(10, t->stellar_source ? t->stellar_source : surface_lut_, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(9, t->aerial_range, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::storage_buffer(11, t->shadows[ctx.frame_index()]),
                rhi::PushDescriptorWrite::sampled_image(12, *ring_shadow_lut_, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(38, *ring_penumbra_, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(16, t->shadow_active ? t->shadow_fraction : surface_lut_, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(17, t->shadow_active ? t->shadow_depth : surface_lut_, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(20, stbn, ctx.sampler_nearest_clamp())};
            writes.insert(writes.end(), transparency_writes.begin(), transparency_writes.end());
            writes.insert(writes.end(), clouds.begin(), clouds.end());
            bool shadow_shader = t->shadow_active || t->shadow_header.flags.w != 0u;
            auto& pipeline = t->limb_active
                ? (t->secondary_active
                    ? (shadow_shader ? limb_secondary_transparent_pipeline_ : limb_secondary_unshadowed_transparent_pipeline_)
                    : (shadow_shader ? limb_transparent_pipeline_ : limb_unshadowed_transparent_pipeline_))
                : t->secondary_active
                ? (shadow_shader ? secondary_transparent_pipeline_ : secondary_unshadowed_transparent_pipeline_)
                : (shadow_shader ? transparent_pipeline_ : unshadowed_transparent_pipeline_);
            if (t->secondary_active) {
                writes.push_back(rhi::PushDescriptorWrite::sampled_image(28, t->secondary_sky, ctx.sampler_linear_clamp()));
                writes.push_back(rhi::PushDescriptorWrite::sampled_image(29, t->secondary_aerial, ctx.sampler_linear_clamp()));
                writes.push_back(rhi::PushDescriptorWrite::sampled_image(30, t->secondary_range, ctx.sampler_linear_clamp()));
                ctx.push_constants(cmd, pipeline, t->secondary_irradiance);
            }
            if (shadow_shader) {
                auto& tiles = t->transparent_tiles[ctx.frame_index()];
                begin_tiles(cmd, ctx, tiles, {color.extent.width, color.extent.height});
                writes.push_back(rhi::PushDescriptorWrite::storage_buffer(37, tiles));
                auto& lean = transparent_lean_pipelines_[(t->limb_active ? 1u : 0u)+(t->secondary_active ? 2u : 0u)];
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lean.handle);
                ctx.push_descriptors(cmd, lean, writes);
                if (t->secondary_active) ctx.push_constants(cmd, lean, t->secondary_irradiance);
                vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
                uint32_t deferred_timer = ctx.begin_gpu_timer(cmd, "atmosphere_transparent_deferred");
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
                ctx.push_descriptors(cmd, pipeline, writes);
                if (t->secondary_active) ctx.push_constants(cmd, pipeline, t->secondary_irradiance);
                dispatch_tiles(cmd, ctx, tiles);
                ctx.end_gpu_timer(cmd, deferred_timer);
            } else {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
                ctx.push_descriptors(cmd, pipeline, writes);
                vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
            }
            ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        ctx.end_gpu_timer(cmd, transparent_timer);
    }
    uint32_t opaque_timer = ctx.begin_gpu_timer(cmd, "atmosphere_opaque");
    if (dual_source_) {
        auto attachment = rhi::RenderingAttachment::load_color(color);
        ctx.begin_rendering(cmd, {color.extent.width, color.extent.height}, {&attachment, 1});
        ctx.set_viewport_and_scissor(cmd, {color.extent.width, color.extent.height});
    }
    for (auto* t : active_) {
        auto point = [&] {
            if (!draw_point || t->cached.depth_params.w >= 1.0f) return;
            if (!dual_source_) {
                ctx.transition_image(cmd, color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                auto attachment = rhi::RenderingAttachment::load_color(color);
                ctx.begin_rendering(cmd, {color.extent.width, color.extent.height}, {&attachment, 1});
                ctx.set_viewport_and_scissor(cmd, {color.extent.width, color.extent.height});
            }
            draw_point(t->body_index);
            if (!dual_source_) {
                ctx.end_rendering(cmd);
                ctx.transition_image(cmd, color, VK_IMAGE_LAYOUT_GENERAL);
            }
        };
        if (t->cached.depth_params.w <= 0.0f) { point(); continue; }
        auto& stbn = stbn_tex_ ? stbn_tex_ : dummy_stbn_;
        rhi::PushDescriptorWrite writes[] = {
            rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
            rhi::PushDescriptorWrite::sampled_image(1, t->sky, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(2, t->throughput, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::storage_image(4, color),
            rhi::PushDescriptorWrite::sampled_image(5, t->aerial_radiance, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(6, t->aerial_optical_depth, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(7, t->transmittance, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(8, t->multiscatter, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(10, t->stellar_source ? t->stellar_source : surface_lut_, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(9, t->aerial_range, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::storage_buffer(11, t->shadows[ctx.frame_index()]),
            rhi::PushDescriptorWrite::sampled_image(12, *ring_shadow_lut_, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(38, *ring_penumbra_, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(16, t->shadow_active ? t->shadow_fraction : surface_lut_, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::sampled_image(17, t->shadow_active ? t->shadow_depth : surface_lut_, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::sampled_image(20, stbn, ctx.sampler_nearest_clamp())};
        if (dual_source_) {
            auto& pipeline = t->limb_active
                ? (t->secondary_active ? limb_secondary_dual_source_pipeline_ : limb_dual_source_pipeline_)
                : (t->secondary_active ? secondary_dual_source_pipeline_ : dual_source_pipeline_);
            std::vector<rhi::PushDescriptorWrite> graphics;
            for (const auto& write : writes) if (write.binding != 4) graphics.push_back(write);
            graphics.insert(graphics.end(), clouds.begin(), clouds.end());
            if (t->secondary_active) {
                graphics.push_back(rhi::PushDescriptorWrite::sampled_image(28, t->secondary_sky, ctx.sampler_linear_clamp()));
                graphics.push_back(rhi::PushDescriptorWrite::sampled_image(29, t->secondary_aerial, ctx.sampler_linear_clamp()));
                graphics.push_back(rhi::PushDescriptorWrite::sampled_image(30, t->secondary_range, ctx.sampler_linear_clamp()));
                ctx.push_constants(cmd, pipeline, t->secondary_irradiance);
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
            ctx.push_descriptors(cmd, pipeline, graphics);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        } else {
            bool shadow_shader = t->shadow_active || t->shadow_header.flags.w != 0u;
            auto& pipeline = t->secondary_active
                ? (transparency_writes.empty() ? secondary_composite_pipeline_ :
                    shadow_shader ? secondary_composite_transparency_pipeline_ : secondary_unshadowed_composite_transparency_pipeline_)
                : (transparency_writes.empty() ? composite_pipeline_ :
                    shadow_shader ? composite_transparency_pipeline_ : unshadowed_composite_transparency_pipeline_);
            std::vector<rhi::PushDescriptorWrite> compute(std::begin(writes), std::end(writes));
            compute.insert(compute.end(), transparency_writes.begin(), transparency_writes.end());
            compute.insert(compute.end(), clouds.begin(), clouds.end());
            if (t->secondary_active) {
                compute.push_back(rhi::PushDescriptorWrite::sampled_image(28, t->secondary_sky, ctx.sampler_linear_clamp()));
                compute.push_back(rhi::PushDescriptorWrite::sampled_image(29, t->secondary_aerial, ctx.sampler_linear_clamp()));
                compute.push_back(rhi::PushDescriptorWrite::sampled_image(30, t->secondary_range, ctx.sampler_linear_clamp()));
                ctx.push_constants(cmd, pipeline, t->secondary_irradiance);
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
            ctx.push_descriptors(cmd, pipeline, compute);
            vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
            ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        point();
    }
    if (dual_source_) ctx.end_rendering(cmd);
    ctx.end_gpu_timer(cmd, opaque_timer);
    ctx.end_gpu_timer(cmd, timer);
}

void AtmosphereRenderer::begin_tiles(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Buffer& tiles, VkExtent2D extent) {
    VkDeviceSize bytes = 4*sizeof(uint32_t)+VkDeviceSize((extent.width+7)/8)*((extent.height+7)/8)*sizeof(uint32_t);
    if (tiles.size < bytes) {
        ctx.destroy_buffer(tiles);
        tiles = ctx.create_device_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    const uint32_t header[4] = {0, 1, 1, 0};
    vkCmdUpdateBuffer(cmd, tiles.handle, 0, sizeof(header), header);
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

void AtmosphereRenderer::dispatch_tiles(VkCommandBuffer cmd, rhi::VkContext& ctx, const rhi::Buffer& tiles) {
    ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    vkCmdDispatchIndirect(cmd, tiles.handle, 0);
}

std::array<rhi::PushDescriptorWrite, 2> AtmosphereRenderer::cloud_layer_writes(rhi::VkContext& ctx,
                                                                              std::span<const rhi::PushDescriptorWrite> cloud_writes) const {
    if (cloud_writes.size() == 2) return {cloud_writes[0], cloud_writes[1]};
    return {rhi::PushDescriptorWrite::sampled_image(33, dummy_cloud_color_, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::sampled_image(34, dummy_cloud_depth_, ctx.sampler_nearest_clamp())};
}

void AtmosphereRenderer::resolve_shadows(VkCommandBuffer cmd, rhi::VkContext& ctx, Tables& table, rhi::Image& color, rhi::Image& depth,
                                         std::span<const rhi::PushDescriptorWrite> transparency_writes,
                                         std::span<const rhi::PushDescriptorWrite> cloud_writes) {
    auto* t = &table;
    if (t->shadow_fraction.extent.width != color.extent.width || t->shadow_fraction.extent.height != color.extent.height) {
        for (auto* image : {&t->shadow_fraction, &t->shadow_depth}) {
            if (*image) retired_images_.push_back({*image, ctx.recording_serial()});
            *image = ctx.create_image_2d(color.extent.width, color.extent.height,
                image == &t->shadow_fraction ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 3);
            ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        t->shadow_history_reusable = false;
        t->shadow_history_reproject = false;
        t->shadow_reset_reasons |= ShadowHistoryViewport;
    }
    const auto& header = t->shadow_header;
    bool temporal = shadow_temporal_enabled_;
    for (auto* image : {&t->shadow_energy, &t->shadow_loss}) {
        bool resize = image->extent.width != color.extent.width || image->extent.height != color.extent.height;
        if (*image && (!temporal || resize)) {
            retired_images_.push_back({*image, ctx.recording_serial()});
            *image = {};
        }
        if (temporal && !*image) {
            *image = ctx.create_image_2d(color.extent.width, color.extent.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT, 1, 2);
            t->shadow_history_reusable = false;
            t->shadow_history_reproject = false;
            t->shadow_reset_reasons |= ShadowHistoryInitial;
            ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_GENERAL);
        }
    }
    bool vrs = !temporal && shadow_vrs_enabled_ && header.flags.w == 0u && t->cached.camera.w == 1.0f &&
               color.extent.width >= 64 && color.extent.height >= 64;
    ctx.transition_image(cmd, t->shadow_fraction, VK_IMAGE_LAYOUT_GENERAL);
    ctx.transition_image(cmd, t->shadow_depth, VK_IMAGE_LAYOUT_GENERAL);
    bool reproject = temporal && t->shadow_history_reproject;
    // The closed-form variants need the single-star cumulative in-scatter cache, which layered aerosols (transport
    // subdivisions) invalidate; prepare_shadows clears flag 32 when a ring is too close to edge-on.
    bool layered_aerosols = false;
    for (int i = 0; i < int(t->cached.aerosol_count.x); ++i) layered_aerosols = layered_aerosols || t->cached.aerosol_shape[i].z > 0.0f;
    bool closed = !temporal && header.flags.w == 0u && (header.flags.y & 36u) == 36u && t->cached.camera.w == 1.0f && !layered_aerosols;
    bool transparent = !transparency_writes.empty();
    // The VRS resolve only upsamples the coarse pass, which is where the closed form applies.
    auto& pipeline = vrs ? (transparent ? shadow_vrs_transparency_pipeline_ : shadow_vrs_pipeline_)
                   : closed ? shadow_closed_pipelines_[transparent ? 1 : 0]
                   : reproject ? (transparent ? shadow_rotate_transparency_pipeline_ : shadow_rotate_pipeline_)
                   : temporal ? (transparent ? shadow_accumulate_transparency_pipeline_ : shadow_accumulate_pipeline_)
                   : (transparent ? shadow_transparency_pipeline_ : shadow_pipeline_);
    auto& stbn = stbn_tex_ ? stbn_tex_ : dummy_stbn_;
    std::vector<rhi::PushDescriptorWrite> writes{
        rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
        rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
        rhi::PushDescriptorWrite::sampled_image(5, t->aerial_radiance, ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(9, t->aerial_range, ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(7, t->transmittance, ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(8, t->multiscatter, ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(10, t->stellar_source ? t->stellar_source : surface_lut_, ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(11, t->shadows[ctx.frame_index()]),
        rhi::PushDescriptorWrite::sampled_image(12, *ring_shadow_lut_, ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(38, *ring_penumbra_, ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_image(18, t->shadow_fraction),
        rhi::PushDescriptorWrite::storage_image(19, t->shadow_depth),
        rhi::PushDescriptorWrite::sampled_image(20, stbn, ctx.sampler_nearest_clamp()),
        cloud_writes[1]};
    uint32_t shadow_timer = ctx.begin_gpu_timer(cmd, "atmosphere_shadows");
    if (temporal) {
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        writes.push_back(rhi::PushDescriptorWrite::storage_image(23, t->shadow_energy));
        writes.push_back(rhi::PushDescriptorWrite::storage_image(24, t->shadow_loss));
        writes.push_back(rhi::PushDescriptorWrite::storage_buffer(25, shadow_diagnostics_[ctx.frame_index()].counters));
        if (reproject) {
            for (auto* image : {&shadow_reproject_energy_, &shadow_reproject_loss_}) {
                if (image->extent.width != color.extent.width || image->extent.height != color.extent.height) {
                    if (*image) retired_images_.push_back({*image, ctx.recording_serial()});
                    *image = ctx.create_image_2d(color.extent.width, color.extent.height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
                }
                ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_GENERAL);
            }
            rhi::PushDescriptorWrite reprojection_writes[] = {
                rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
                rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::storage_buffer(11, t->shadows[ctx.frame_index()]),
                rhi::PushDescriptorWrite::storage_image(23, t->shadow_energy),
                rhi::PushDescriptorWrite::storage_image(24, t->shadow_loss),
                rhi::PushDescriptorWrite::storage_image(26, shadow_reproject_energy_),
                rhi::PushDescriptorWrite::storage_image(27, shadow_reproject_loss_)};
            const std::array<glm::mat4, 2> transforms{t->shadow_reprojection, t->shadow_previous_inverse};
            auto reproject_timer = ctx.begin_gpu_timer(cmd, "atmosphere_shadow_reproject");
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shadow_reproject_pipeline_.handle);
            ctx.push_descriptors(cmd, shadow_reproject_pipeline_, reprojection_writes);
            ctx.push_constants(cmd, shadow_reproject_pipeline_, transforms);
            vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
            ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            ctx.end_gpu_timer(cmd, reproject_timer);
            writes.push_back(rhi::PushDescriptorWrite::storage_image(26, shadow_reproject_energy_));
            writes.push_back(rhi::PushDescriptorWrite::storage_image(27, shadow_reproject_loss_));
            ++stats_.shadow_rotation_updates;
        }
        ctx.push_constants(cmd, pipeline, glm::uvec2(t->shadow_history_reusable, shadow_diagnostics_enabled_));
        if (!t->shadow_history_reusable && !reproject) {
            ++stats_.shadow_history_resets;
            stats_.shadow_history_reset_reasons |= t->shadow_reset_reasons;
        }
    }
    if (vrs) {
        uint32_t scale = 1u << std::clamp(header.flags.y >> 8u & 15u, 1u, 4u);
        uint32_t width = (color.extent.width+scale-1)/scale, height = (color.extent.height+scale-1)/scale;
        if (shadow_coarse_fraction_.extent.width != width || shadow_coarse_fraction_.extent.height != height) {
            for (auto* image : {&shadow_coarse_fraction_, &shadow_coarse_guide_}) {
                if (*image) retired_images_.push_back({*image, ctx.recording_serial()});
                *image = ctx.create_image_2d(width, height,
                    image == &shadow_coarse_fraction_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, 3);
            }
        }
        ctx.transition_image(cmd, shadow_coarse_fraction_, VK_IMAGE_LAYOUT_GENERAL);
        ctx.transition_image(cmd, shadow_coarse_guide_, VK_IMAGE_LAYOUT_GENERAL);
        auto coarse_writes = writes;
        std::erase_if(coarse_writes, [](auto write) { return write.binding == 18 || write.binding == 19; });
        coarse_writes.push_back(rhi::PushDescriptorWrite::storage_image(18, shadow_coarse_fraction_));
        coarse_writes.push_back(rhi::PushDescriptorWrite::storage_image(19, shadow_coarse_guide_));
        coarse_writes.insert(coarse_writes.end(), transparency_writes.begin(), transparency_writes.end());
        auto& coarse = closed ? shadow_closed_pipelines_[transparent ? 3 : 2]
            : transparent ? shadow_coarse_transparency_pipeline_ : shadow_coarse_pipeline_;
        uint32_t cw = uint32_t(std::max(0, header.shadow_coarse_bounds.z-header.shadow_coarse_bounds.x));
        uint32_t ch = uint32_t(std::max(0, header.shadow_coarse_bounds.w-header.shadow_coarse_bounds.y));
        uint32_t coarse_timer = ctx.begin_gpu_timer(cmd, "atmosphere_shadow_coarse");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, coarse.handle);
        ctx.push_descriptors(cmd, coarse, coarse_writes);
        vkCmdDispatch(cmd, (cw+7)/8, (ch+7)/8, 1);
        ctx.end_gpu_timer(cmd, coarse_timer);
        for (auto* image : {&shadow_coarse_fraction_, &shadow_coarse_guide_})
            ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        writes.push_back(rhi::PushDescriptorWrite::sampled_image(21, shadow_coarse_fraction_, ctx.sampler_nearest_clamp()));
        writes.push_back(rhi::PushDescriptorWrite::sampled_image(22, shadow_coarse_guide_, ctx.sampler_nearest_clamp()));
        ++stats_.shadow_vrs_updates;
    }
    writes.insert(writes.end(), transparency_writes.begin(), transparency_writes.end());
    uint32_t sw = uint32_t(std::max(0, header.shadow_bounds.z-header.shadow_bounds.x));
    uint32_t sh = uint32_t(std::max(0, header.shadow_bounds.w-header.shadow_bounds.y));
    uint32_t resolve_timer = ctx.begin_gpu_timer(cmd, "atmosphere_shadow_resolve");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    ctx.push_descriptors(cmd, pipeline, writes);
    vkCmdDispatch(cmd, (sw+7)/8, (sh+7)/8, 1);
    ctx.end_gpu_timer(cmd, resolve_timer);
    for (auto* image : {&t->shadow_fraction, &t->shadow_depth})
        ctx.transition_image(cmd, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    ctx.end_gpu_timer(cmd, shadow_timer);
    t->shadow_history_valid = temporal;
    t->shadow_recorded_prepare = prepare_serial_;
    ++stats_.shadow_updates;
}

}
