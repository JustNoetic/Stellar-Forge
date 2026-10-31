#include "rendering/bloom_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <cmath>

namespace sf::rendering {
namespace {

struct ExtractPush { float dx, dy, highlight_limit, diffraction, threshold; uint32_t capacity; };
struct ApplyPush { float strength, highlight_limit; };
struct UpsamplePush { float coarse_weight; };
struct SpikePush { float width, height, dx, dy, half_length, core_width, scale; uint32_t capacity; };
constexpr VkDeviceSize SPIKE_BYTES = 880;

void readable(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& image) {
    ctx.transition_image(cmd, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void dispatch(VkCommandBuffer cmd, const rhi::Image& image) {
    vkCmdDispatch(cmd, (image.extent.width+7)/8, (image.extent.height+7)/8, 1);
}

double spike_profile_integral(float half_length, float width) {
    constexpr int STEPS_PER_PIXEL = 16;
    int steps = int(std::ceil(half_length))*STEPS_PER_PIXEL;
    double sum = 0;
    for (int i = -steps; i <= steps; ++i) {
        double x = double(i)/STEPS_PER_PIXEL;
        double t = std::clamp((std::abs(x)/half_length-0.55)/0.45, 0.0, 1.0);
        double taper = 1-t*t*(3-2*t);
        sum += taper*taper/(1+x*x/(double(width)*width));
    }
    return sum/STEPS_PER_PIXEL;
}

}

void BloomRenderer::init(rhi::VkContext& ctx) {
    const std::array bindings{
        rhi::DescriptorLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        rhi::DescriptorLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}};
    auto extract_bindings = std::vector(bindings.begin(), bindings.end());
    extract_bindings.push_back({2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT});
    extract_ = rhi::ShaderCache::create_compute_pipeline(ctx, "post/bloom_extract.comp", extract_bindings, sizeof(ExtractPush));
    downsample_ = rhi::ShaderCache::create_compute_pipeline(ctx, "post/bloom_downsample.comp", bindings);
    auto up_bindings = std::vector(bindings.begin(), bindings.end());
    up_bindings.push_back({2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT});
    upsample_ = rhi::ShaderCache::create_compute_pipeline(ctx, "post/bloom_upsample.comp", up_bindings, sizeof(UpsamplePush));
    std::vector<rhi::DescriptorLayoutBinding> apply_bindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}};
    apply_ = rhi::ShaderCache::create_compute_pipeline(ctx, "post/bloom_apply.comp", apply_bindings, sizeof(ApplyPush));
    rhi::GraphicsPipelineDesc desc;
    desc.vert_path = "post/bloom_spike.vert";
    desc.frag_path = "post/bloom_spike.frag";
    desc.bindings = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}};
    desc.push_constant_size = sizeof(SpikePush);
    desc.depth_format = VK_FORMAT_UNDEFINED;
    desc.depth_test = false;
    desc.depth_write = false;
    desc.additive_blend = true;
    desc.cull_mode = VK_CULL_MODE_NONE;
    spike_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    for (auto& buffer : spikes_)
        buffer = ctx.create_device_buffer(sizeof(VkDrawIndirectCommand)+SPIKE_CAPACITY*SPIKE_BYTES,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

void BloomRenderer::shutdown(rhi::VkContext& ctx) {
    for (auto& image : down_) ctx.destroy_image(image);
    for (auto& image : up_) ctx.destroy_image(image);
    for (auto& buffer : spikes_) ctx.destroy_buffer(buffer);
    ctx.destroy_image(physical_scene_);
    for (auto& entry : retired_) ctx.destroy_image(entry.image);
    retired_.clear();
    for (auto* pipeline : {&extract_, &downsample_, &upsample_, &apply_, &spike_})
        rhi::ShaderCache::destroy_pipeline(ctx, *pipeline);
}

void BloomRenderer::prepare(rhi::VkContext& ctx, VkExtent2D extent) {
    std::erase_if(retired_, [&](auto& entry) {
        if (entry.serial > ctx.completed_serial()) return false;
        ctx.destroy_image(entry.image);
        return true;
    });
    if (physical_scene_.extent.width == extent.width && physical_scene_.extent.height == extent.height) return;
    auto resize = [&](rhi::Image& image, uint32_t width, uint32_t height) {
        if (image) retired_.push_back({image, ctx.recording_serial()});
        image = ctx.create_image_2d(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    };
    resize(physical_scene_, extent.width, extent.height);
    uint32_t width = (extent.width+3)/4, height = (extent.height+3)/4;
    for (size_t i = 0; i < LEVELS; ++i) {
        if (i > 0) {
            width = std::max(1u, (width+1)/2);
            height = std::max(1u, (height+1)/2);
        }
        resize(down_[i], width, height);
        if (i < up_.size()) resize(up_[i], width, height);
    }
}

bool BloomRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color,
                           const RenderSettings& settings, rhi::Image* physical_scene, bool defer_resolve) {
    if (!enabled(settings)) return false;
    prepare(ctx, {color.extent.width, color.extent.height});
    bool spikes = settings.bloom_mode == 2 && settings.bloom_diffraction > 0.0f;
    auto& spike_buffer = spikes_[ctx.frame_index()];
    float angle = settings.bloom_rotation_deg*0.017453292519943295f;
    float dx = std::cos(angle), dy = std::sin(angle);
    if (spikes) {
        VkDrawIndirectCommand command{12, 0, 0, 0};
        vkCmdUpdateBuffer(cmd, spike_buffer.handle, 0, sizeof(command), &command);
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    auto timer = ctx.begin_gpu_timer(cmd, "bloom");
    auto downsample_timer = ctx.begin_gpu_timer(cmd, "bloom_downsample");
    auto& source = physical_scene ? *physical_scene : color;
    readable(cmd, ctx, source);
    for (size_t i = 0; i < LEVELS; ++i) {
        auto& input = i == 0 ? source : down_[i-1];
        auto& output = down_[i];
        auto& pipeline = i == 0 ? extract_ : downsample_;
        ctx.transition_image(cmd, output, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        if (i == 0) {
            ctx.push_descriptors(cmd, pipeline, {
                rhi::PushDescriptorWrite::sampled_image(0, input, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::storage_image(1, output),
                rhi::PushDescriptorWrite::storage_buffer(2, spike_buffer)});
            ctx.push_constants(cmd, pipeline, ExtractPush{dx, dy, settings.bloom_highlight_limit,
                spikes ? settings.bloom_diffraction : 0.0f, settings.bloom_spike_threshold, SPIKE_CAPACITY});
        } else {
            ctx.push_descriptors(cmd, pipeline, {
                rhi::PushDescriptorWrite::sampled_image(0, input, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::storage_image(1, output)});
        }
        if (i == 0) vkCmdDispatch(cmd, (output.extent.width+3)/4, (output.extent.height+3)/4, 1);
        else dispatch(cmd, output);
        readable(cmd, ctx, output);
    }
    ctx.end_gpu_timer(cmd, downsample_timer);
    auto upsample_timer = ctx.begin_gpu_timer(cmd, "bloom_upsample");
    for (int i = int(LEVELS)-2; i >= 0; --i) {
        auto& coarse = i == int(LEVELS)-2 ? down_[i+1] : up_[i+1];
        ctx.transition_image(cmd, up_[i], VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, upsample_.handle);
        ctx.push_descriptors(cmd, upsample_, {
            rhi::PushDescriptorWrite::sampled_image(0, coarse, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::storage_image(1, up_[i]),
            rhi::PushDescriptorWrite::sampled_image(2, down_[i], ctx.sampler_linear_clamp())});
        // The quarter level also stands in for the dropped half-resolution level,
        // which took 0.35 of the halo, so it keeps 0.35 + 0.65*0.35 of its own.
        ctx.push_constants(cmd, upsample_, UpsamplePush{i == 0 ? 0.65f*0.65f : 0.65f});
        dispatch(cmd, up_[i]);
        readable(cmd, ctx, up_[i]);
    }
    ctx.end_gpu_timer(cmd, upsample_timer);
    // The deferred correction reads the physical image, which the spikes leave
    // untouched. Without one, the scene itself is the source and must be
    // corrected before the spikes are added to it.
    bool deferred = defer_resolve && physical_scene;
    if (!deferred) {
        auto apply_timer = ctx.begin_gpu_timer(cmd, "bloom_apply");
        apply(cmd, ctx, color, source, settings);
        ctx.end_gpu_timer(cmd, apply_timer);
    }
    if (spikes) {
        auto spikes_timer = ctx.begin_gpu_timer(cmd, "bloom_spikes");
        float scale = float(color.extent.height)/1080.0f*settings.bloom_radius;
        float half_length = 2.0f*float(std::clamp(int(std::ceil(48.0f*scale)), 4, 192));
        float width = 2.0f*std::max(0.65f, 0.8f*scale);
        if (spike_length_ != half_length || spike_width_ != width) {
            spike_length_ = half_length;
            spike_width_ = width;
            spike_normalization_ = float(1.0/spike_profile_integral(half_length, width));
        }
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        ctx.transition_image(cmd, color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        auto attachment = rhi::RenderingAttachment::load_color(color);
        VkExtent2D extent{color.extent.width, color.extent.height};
        ctx.begin_rendering(cmd, extent, {&attachment, 1});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, spike_.handle);
        ctx.push_descriptors(cmd, spike_, {rhi::PushDescriptorWrite::storage_buffer(0, spike_buffer)});
        ctx.push_constants(cmd, spike_, SpikePush{float(extent.width), float(extent.height), dx, dy,
            half_length, width, 0.5f*settings.bloom_strength*spike_normalization_, SPIKE_CAPACITY});
        vkCmdDrawIndirect(cmd, spike_buffer.handle, 0, 1, sizeof(VkDrawIndirectCommand));
        ctx.end_rendering(cmd);
        ctx.end_gpu_timer(cmd, spikes_timer);
    }
    ctx.memory_barrier(cmd);
    ctx.end_gpu_timer(cmd, timer);
    return deferred;
}

void BloomRenderer::apply(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& source,
                          const RenderSettings& settings) {
    ctx.transition_image(cmd, color, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, apply_.handle);
    ctx.push_descriptors(cmd, apply_, {
        rhi::PushDescriptorWrite::storage_image(0, color),
        rhi::PushDescriptorWrite::sampled_image(1, source, ctx.sampler_nearest_clamp(), source.layout),
        rhi::PushDescriptorWrite::sampled_image(2, up_[0], ctx.sampler_linear_clamp())});
    ctx.push_constants(cmd, apply_, ApplyPush{settings.bloom_strength, settings.bloom_highlight_limit});
    dispatch(cmd, color);
}

void BloomRenderer::resolve(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color,
                            const RenderSettings& settings, rhi::Image& physical_scene) {
    readable(cmd, ctx, physical_scene);
    apply(cmd, ctx, color, physical_scene, settings);
    ctx.memory_barrier(cmd);
}

}
