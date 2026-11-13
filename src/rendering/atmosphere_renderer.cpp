#include "core/path_utils.hpp"
#include "rendering/atmosphere_internal.hpp"
#include "rendering/refraction_tables.hpp"
#include "rendering/ring_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>
#include <glm/gtc/matrix_inverse.hpp>

namespace sf::rendering {

void AtmosphereRenderer::init(rhi::VkContext& ctx) {
    constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
    const rhi::DescriptorLayoutBinding trans[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    const rhi::DescriptorLayoutBinding multi[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    const rhi::DescriptorLayoutBinding sky[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}};

    transmittance_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/transmittance.comp", trans);
    multiscatter_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/multiscatter.comp", multi);
    sky_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/sky_view.comp", sky);
    const std::pair<std::string, std::string> small_stars[] = {{"ATMOSPHERE_SMALL_STARS", "1"}};
    small_sky_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/sky_view.comp", sky, 0, small_stars);
    std::vector<rhi::DescriptorLayoutBinding> aerial(std::begin(sky), std::end(sky));
    aerial.push_back({5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    aerial.push_back({6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    aerial.push_back({7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    aerial_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/aerial_perspective.comp", aerial);
    small_aerial_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/aerial_perspective.comp", aerial, 0, small_stars);
    composite_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/composite.comp", composite_bindings);
    const rhi::DescriptorLayoutBinding shadow[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage}, {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {34, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {38, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}};
    shadow_pipeline_ =rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow.comp", shadow);
    std::vector<rhi::DescriptorLayoutBinding> shadow_transparency(std::begin(shadow), std::end(shadow));
    shadow_transparency.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    shadow_transparency.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_transparency.comp", shadow_transparency);
    auto accumulate = std::vector<rhi::DescriptorLayoutBinding>(std::begin(shadow), std::end(shadow));
    accumulate.push_back({23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    accumulate.push_back({24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    accumulate.push_back({25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_accumulate_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_accumulate.comp", accumulate, sizeof(glm::uvec2));
    auto rotate = accumulate;
    rotate.push_back({26, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    rotate.push_back({27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    shadow_rotate_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_rotate.comp", rotate, sizeof(glm::uvec2));
    rotate.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    rotate.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_rotate_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_rotate_transparency.comp", rotate, sizeof(glm::uvec2));
    const rhi::DescriptorLayoutBinding reproject[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage}, {23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {26, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    shadow_reproject_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_reproject.comp", reproject, 2*sizeof(glm::mat4));
    accumulate.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    accumulate.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_accumulate_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_accumulate_transparency.comp", accumulate, sizeof(glm::uvec2));
    shadow_coarse_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_coarse.comp", shadow);
    std::vector<rhi::DescriptorLayoutBinding> coarse_transparency(std::begin(shadow), std::end(shadow));
    coarse_transparency.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    coarse_transparency.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_coarse_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/shadow_coarse_transparency.comp", coarse_transparency);
    const std::pair<std::string, std::string> tile_deferred[] = {{"ATMOSPHERE_TILE_DEFERRED", "1"}};
    const std::pair<std::string, std::string> tile_lean[] = {{"ATMOSPHERE_TILE_LEAN", "1"}};
    std::vector<rhi::DescriptorLayoutBinding> shadow_vrs(std::begin(shadow), std::end(shadow));
    shadow_vrs.push_back({21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage});
    shadow_vrs.push_back({22, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage});
    shadow_vrs_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_vrs.comp", shadow_vrs);
    shadow_vrs.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    shadow_vrs.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    shadow_vrs_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_vrs_transparency.comp", shadow_vrs);
    {
        const std::pair<std::string, std::string> closed[] = {{"ATMOSPHERE_SHADOW_CLOSED_FORM", "1"}};
        shadow_closed_pipelines_[0] = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow.comp", shadow, 0, closed);
        shadow_closed_pipelines_[1] = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_transparency.comp", shadow_transparency, 0, closed);
        shadow_closed_pipelines_[2] = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_coarse.comp", shadow, 0, closed);
        shadow_closed_pipelines_[3] = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/shadow_coarse_transparency.comp", coarse_transparency, 0, closed);
    }
    std::vector<rhi::DescriptorLayoutBinding> transparent_bindings(std::begin(composite_bindings), std::end(composite_bindings));
    transparent_bindings.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    transparent_bindings.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    composite_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/composite_transparency.comp", transparent_bindings);
    const std::pair<std::string, std::string> unshadowed[] = {{"ATMOSPHERE_UNSHADOWED", "1"}};
    unshadowed_composite_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/composite_transparency.comp", transparent_bindings, 0, unshadowed);
    std::erase_if(transparent_bindings, [](auto binding) { return binding.binding == 4; });
    auto tiled_bindings = transparent_bindings;
    tiled_bindings.push_back({37, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", tiled_bindings, 0, tile_deferred);
    transparent_lean_pipelines_[0] = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", tiled_bindings, 0, tile_lean);
    unshadowed_transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", transparent_bindings, 0, unshadowed);
    const std::pair<std::string, std::string> limb_deferred[] = {{"ATMOSPHERE_LIMB", "1"}, {"ATMOSPHERE_TILE_DEFERRED", "1"}};
    const std::pair<std::string, std::string> limb_lean[] = {{"ATMOSPHERE_LIMB", "1"}, {"ATMOSPHERE_TILE_LEAN", "1"}};
    const std::pair<std::string, std::string> unshadowed_limb[] = {{"ATMOSPHERE_UNSHADOWED", "1"}, {"ATMOSPHERE_LIMB", "1"}};
    limb_transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", tiled_bindings, 0, limb_deferred);
    transparent_lean_pipelines_[1] = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", tiled_bindings, 0, limb_lean);
    limb_unshadowed_transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", transparent_bindings, 0, unshadowed_limb);
    surface_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/surface_irradiance.comp", source_bindings, sizeof(glm::uvec2));
    {
        const float zero[4] = {};
        dummy_cloud_color_ = ctx.create_image_2d(1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ctx.upload_image_2d(dummy_cloud_color_, zero, sizeof(zero));
        dummy_cloud_depth_ = ctx.create_image_2d(1, 1, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ctx.upload_image_2d(dummy_cloud_depth_, zero, sizeof(float));
    }
    surface_lut_ = ctx.create_image_2d(128, 32, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 1, 96);
    {
        const float zero = 0.0f;
        ring_penumbra_dummy_ = ctx.create_image_2d(1, 1, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, 1, true);
        ctx.upload_image_2d(ring_penumbra_dummy_, &zero, sizeof(zero));
        ring_penumbra_ = &ring_penumbra_dummy_;
    }
    ctx.immediate_submit([&](VkCommandBuffer cmd) { ctx.transition_image(cmd, surface_lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); });
    auto stbn_data = paths::read_binary_file(paths::external_path("textures", "noise", "stbn_scalar.bin"));
    if (stbn_data && stbn_data->size() == 128 * 128 * 64) {
        stbn_tex_ = ctx.create_image_3d(128, 128, 64, VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ctx.upload_image_2d(stbn_tex_, stbn_data->data(), stbn_data->size(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    if (!stbn_tex_) {
        uint8_t dummy_val = 128;
        dummy_stbn_ = ctx.create_image_3d(1, 1, 1, VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ctx.upload_image_2d(dummy_stbn_, &dummy_val, 1,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    for (auto& buffer : point_responses_) buffer = ctx.create_host_buffer(1024*16*sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    for (auto& buffer : surface_records_) buffer = ctx.create_host_buffer(1024*sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    for (auto& frame : shadow_diagnostics_) frame.counters = ctx.create_host_buffer(8*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (ctx.dual_source_blend_enabled()) {
        rhi::GraphicsPipelineDesc desc;
        desc.vert_path = "post/fullscreen.vert";
        desc.frag_path = "atmosphere/composite.frag";
        desc.capture_statistics = ctx.pipeline_statistics_enabled();
        for (auto binding : composite_bindings) {
            if (binding.binding == 4) continue;
            binding.stages = VK_SHADER_STAGE_FRAGMENT_BIT;
            desc.bindings.push_back(binding);
        }
        desc.depth_format = VK_FORMAT_UNDEFINED;
        desc.depth_test = desc.depth_write = false;
        desc.cull_mode = VK_CULL_MODE_NONE;
        desc.dual_source_blend = true;
        dual_source_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
        desc.macros = {{"ATMOSPHERE_LIMB", "1"}};
        limb_dual_source_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    }
    init_secondary(ctx);
}

void AtmosphereRenderer::init_secondary(rhi::VkContext& ctx) {
    if (secondary_composite_pipeline_) return;
    constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
    auto composite = std::vector<rhi::DescriptorLayoutBinding>(std::begin(composite_bindings), std::end(composite_bindings));
    for (uint32_t binding : {28u, 29u, 30u}) composite.push_back({binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage});
    const std::pair<std::string, std::string> macros[] = {{"ATMOSPHERE_SECONDARY", "1"}};
    const std::pair<std::string, std::string> unshadowed[] = {{"ATMOSPHERE_SECONDARY", "1"}, {"ATMOSPHERE_UNSHADOWED", "1"}};
    secondary_composite_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/composite.comp", composite, sizeof(glm::vec4), macros);
    composite.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
    composite.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    secondary_composite_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/composite_transparency.comp", composite, sizeof(glm::vec4), macros);
    secondary_unshadowed_composite_transparency_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/composite_transparency.comp", composite, sizeof(glm::vec4), unshadowed);
    std::erase_if(composite, [](auto binding) { return binding.binding == 4; });
    auto tiled = composite;
    tiled.push_back({37, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage});
    const std::pair<std::string, std::string> deferred[] = {{"ATMOSPHERE_SECONDARY", "1"}, {"ATMOSPHERE_TILE_DEFERRED", "1"}};
    const std::pair<std::string, std::string> lean[] = {{"ATMOSPHERE_SECONDARY", "1"}, {"ATMOSPHERE_TILE_LEAN", "1"}};
    secondary_transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", tiled, sizeof(glm::vec4), deferred);
    transparent_lean_pipelines_[2] = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", tiled, sizeof(glm::vec4), lean);
    secondary_unshadowed_transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", composite, sizeof(glm::vec4), unshadowed);
    const std::pair<std::string, std::string> limb[] = {{"ATMOSPHERE_SECONDARY", "1"}, {"ATMOSPHERE_LIMB", "1"}};
    const std::pair<std::string, std::string> limb_deferred[] = {{"ATMOSPHERE_SECONDARY", "1"}, {"ATMOSPHERE_LIMB", "1"}, {"ATMOSPHERE_TILE_DEFERRED", "1"}};
    const std::pair<std::string, std::string> limb_lean[] = {{"ATMOSPHERE_SECONDARY", "1"}, {"ATMOSPHERE_LIMB", "1"}, {"ATMOSPHERE_TILE_LEAN", "1"}};
    const std::pair<std::string, std::string> unshadowed_limb[] = {{"ATMOSPHERE_SECONDARY", "1"}, {"ATMOSPHERE_UNSHADOWED", "1"}, {"ATMOSPHERE_LIMB", "1"}};
    limb_secondary_transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", tiled, sizeof(glm::vec4), limb_deferred);
    transparent_lean_pipelines_[3] = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", tiled, sizeof(glm::vec4), limb_lean);
    limb_secondary_unshadowed_transparent_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
        "atmosphere/transparent.comp", composite, sizeof(glm::vec4), unshadowed_limb);
    if (ctx.dual_source_blend_enabled()) {
        rhi::GraphicsPipelineDesc desc;
        desc.vert_path = "post/fullscreen.vert";
        desc.frag_path = "atmosphere/composite.frag";
        desc.capture_statistics = ctx.pipeline_statistics_enabled();
        desc.macros.assign(std::begin(macros), std::end(macros));
        desc.push_constant_size = sizeof(glm::vec4);
        for (auto binding : composite) {
            if (binding.binding == 14 || binding.binding == 15) continue;
            binding.stages = VK_SHADER_STAGE_FRAGMENT_BIT;
            desc.bindings.push_back(binding);
        }
        desc.depth_format = VK_FORMAT_UNDEFINED;
        desc.depth_test = desc.depth_write = false;
        desc.cull_mode = VK_CULL_MODE_NONE;
        desc.dual_source_blend = true;
        secondary_dual_source_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
        desc.macros.assign(std::begin(limb), std::end(limb));
        limb_secondary_dual_source_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    }
}

void AtmosphereRenderer::shutdown(rhi::VkContext& ctx) {
    for (auto& t : tables_) {
        ctx.destroy_buffer(t->point_partials);
        ctx.destroy_buffer(t->point_response);
        ctx.destroy_image(t->transmittance);
        ctx.destroy_image(t->multiscatter);
        ctx.destroy_image(t->stellar_source);
        ctx.destroy_image(t->sky);
        ctx.destroy_image(t->throughput);
        ctx.destroy_image(t->aerial_radiance);
        ctx.destroy_image(t->aerial_optical_depth);
        ctx.destroy_image(t->aerial_range);
        ctx.destroy_image(t->secondary_sky);
        ctx.destroy_image(t->secondary_aerial);
        ctx.destroy_image(t->secondary_range);
        ctx.destroy_image(t->secondary_source);
        for (auto& u : t->uniforms) ctx.destroy_buffer(u);
        for (auto& shadow : t->shadows) ctx.destroy_buffer(shadow);
        for (auto& tiles : t->transparent_tiles) ctx.destroy_buffer(tiles);
        ctx.destroy_image(t->shadow_fraction);
        ctx.destroy_image(t->shadow_depth);
        ctx.destroy_image(t->shadow_energy);
        ctx.destroy_image(t->shadow_loss);
    }
    tables_.clear();
    active_.clear();
    surface_layers_.fill(nullptr);
    surface_keys_.fill(glm::vec2(0.0f));
    rhi::ShaderCache::destroy_pipeline(ctx, transmittance_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, multiscatter_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, sky_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, aerial_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, point_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, point_reduce_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, composite_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, transparent_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, composite_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, unshadowed_transparent_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, unshadowed_composite_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, dual_source_pipeline_);
    for (auto* pipeline : {&limb_dual_source_pipeline_, &limb_transparent_pipeline_, &limb_unshadowed_transparent_pipeline_,
                           &limb_secondary_dual_source_pipeline_, &limb_secondary_transparent_pipeline_, &limb_secondary_unshadowed_transparent_pipeline_})
        rhi::ShaderCache::destroy_pipeline(ctx, *pipeline);
    rhi::ShaderCache::destroy_pipeline(ctx, surface_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, stellar_source_pipeline_);
    for (auto& pipeline : secondary_bake_pipelines_) rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    for (auto& pipeline : secondary_approximate_pipelines_) rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    rhi::ShaderCache::destroy_pipeline(ctx, secondary_source_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, secondary_composite_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, secondary_composite_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, secondary_transparent_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, secondary_unshadowed_transparent_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, secondary_unshadowed_composite_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, secondary_dual_source_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, small_sky_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, small_aerial_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, molecular_sky_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, molecular_aerial_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, legacy_sky_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, legacy_aerial_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_accumulate_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_accumulate_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_reproject_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_rotate_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_rotate_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_coarse_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_coarse_transparency_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_vrs_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, shadow_vrs_transparency_pipeline_);
    for (auto& pipeline : shadow_closed_pipelines_) rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    for (auto& pipeline : transparent_lean_pipelines_) rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    ctx.destroy_image(ring_penumbra_dummy_);
    ring_penumbra_ = nullptr;
    ctx.destroy_image(shadow_coarse_fraction_);
    ctx.destroy_image(shadow_coarse_guide_);
    ctx.destroy_image(shadow_reproject_energy_);
    ctx.destroy_image(shadow_reproject_loss_);
    for (auto& retired : retired_images_) ctx.destroy_image(retired.image);
    retired_images_.clear();
    ctx.destroy_image(surface_lut_);
    ctx.destroy_image(dummy_cloud_color_);
    ctx.destroy_image(dummy_cloud_depth_);
    if (stbn_tex_) ctx.destroy_image(stbn_tex_);
    if (dummy_stbn_) ctx.destroy_image(dummy_stbn_);
    for (auto& buffer : surface_records_) ctx.destroy_buffer(buffer);
    for (auto& buffer : point_responses_) ctx.destroy_buffer(buffer);
    for (auto& frame : shadow_diagnostics_) ctx.destroy_buffer(frame.counters);
}

ShadowHistoryDiagnostics AtmosphereRenderer::shadow_history_diagnostics(rhi::VkContext& ctx) const {
    if (!shadow_diagnostics_enabled_ || !shadow_temporal_enabled_ || stats_.shadow_updates == 0) return {};
    auto result = shadow_last_diagnostics_;
    const ShadowDiagnosticFrame* latest = nullptr;
    for (const auto& frame : shadow_diagnostics_)
        if (frame.serial > result.serial && frame.serial <= ctx.completed_serial() && (!latest || frame.serial > latest->serial))
            latest = &frame;
    if (!latest) return result;
    VK_CHECK(vmaInvalidateAllocation(ctx.allocator(), latest->counters.allocation, 0, VK_WHOLE_SIZE));
    std::array<uint32_t, 8> values;
    std::memcpy(values.data(), latest->counters.mapped, sizeof(values));
    result.columns = values[0];
    result.serial = latest->serial;
    result.depth_resets = values[1];
    result.refining = values[2];
    result.complete = values[3];
    result.minimum_frames = values[0] ? values[4] : 0;
    result.maximum_frames = values[5];
    result.reprojected = values[6];
    result.reprojection_rejected = values[7];
    result.reset_reasons = latest->reset_reasons;
    result.valid = true;
    return result;
}

void AtmosphereRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                                 const ephemeris::SystemSnapshot& system, const RenderSettings& settings, VkExtent2D extent,
                                 std::span<const float> minimum_heights_km, std::span<const RingHostGPU> rings,
                                 const rhi::Image* ring_shadow_lut, uint64_t ring_shadow_revision, bool shadow_diagnostics,
                                 std::span<const PlanetshineGPU> planetshine, const RingRenderer* ring_lighting,
                                 const RefractionHost* refraction) {
    if (!begin_prepare(ctx, system, settings, extent, ring_shadow_lut, shadow_diagnostics)) return;
    auto candidates = select_candidates(camera, system, extent, refraction);
    std::vector<size_t> slots, surface_slots;
    assign_tables(candidates, system, slots, surface_slots);
    double focal = double(extent.height)/(2.0*std::tan(camera.fov_deg*DEG_TO_RAD*0.5));
    glm::dvec3 eye = camera.eye_pos_render().value;
    const LutRecorder lut{cmd, ctx};
    ring_penumbra_ = ring_lighting ? &ring_lighting->penumbra_table() : &ring_penumbra_dummy_;
    if (!*ring_penumbra_) ring_penumbra_ = &ring_penumbra_dummy_;
    uint32_t timer = candidates.empty() ? UINT32_MAX : ctx.begin_gpu_timer(cmd, "atmosphere_luts");
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto candidate = candidates[i];
        size_t slot = slots[i];
        if (slot == tables_.size()) {
            auto t = std::make_unique<Tables>();
            t->transmittance = lut.image(256, 64, VK_FORMAT_R32G32B32A32_SFLOAT);
            t->multiscatter = lut.image(32, 32);
            t->sky = lut.image(192, 108);
            t->throughput = lut.image(192, 108);
            for (auto& u : t->uniforms) u = ctx.create_host_buffer(sizeof(AtmosphereParametersGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            for (auto& shadow : t->shadows) shadow = ctx.create_host_buffer(sizeof(AtmosphereShadowHeaderGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            tables_.push_back(std::move(t));
        }
        auto& t = *tables_[slot];
        const auto b = system.body(candidate.index);
        auto frame = make_atmosphere_frame(system, candidate.index, camera, settings, extent, candidate.apparent,
                                           candidate.distance, focal, eye, minimum_heights_km);
        auto& p = frame.p;
        // The 2D sheet is blended here from the cloud layer; progressive shadows keep it as a transparent fragment
        // because their history has no cloud endpoint layer.
        bool inline_clouds = settings.clouds_enabled && settings.cloud_mode == 0 && settings.cloud_sheet_inline &&
            !settings.atmosphere_shadow_temporal && ctx.fragment_stores_enabled() && b.definition.clouds.enabled;
        p.cloud_layer = glm::vec4(float(candidate.index), inline_clouds ? 1.0f : 0.0f, 0.0f, 0.0f);
        t.limb_active = frame.limb_active;
        if (frame.point_radius >= 2.5) t.point_valid = false;
        if (p.depth_params.w > 0.0f && !t.aerial_radiance) {
            t.aerial_radiance = ctx.create_image_3d(192, 108, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
            t.aerial_optical_depth = ctx.create_image_3d(192, 108, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
            t.aerial_range = lut.image(192, 108, VK_FORMAT_R32_SFLOAT);
            lut.readable(t.aerial_radiance);
            lut.readable(t.aerial_optical_depth);
            lut.readable(t.aerial_range);
        }
        AtmosphereChanges changes;
        changes.optics = !t.valid || t.identity != b.revisions.identity ||
            p.reference_shape != t.cached.reference_shape || p.radii_heights != t.cached.radii_heights || p.rayleigh != t.cached.rayleigh ||
            p.mie_scattering != t.cached.mie_scattering || p.mie_extinction != t.cached.mie_extinction || p.ground_albedo != t.cached.ground_albedo ||
            p.mixed_absorption != t.cached.mixed_absorption || p.ozone_absorption != t.cached.ozone_absorption ||
            p.ozone_params != t.cached.ozone_params || p.profile_domain != t.cached.profile_domain || p.log_density != t.cached.log_density ||
            p.aerosol_count != t.cached.aerosol_count || p.aerosol_shape != t.cached.aerosol_shape ||
            p.aerosol_scattering != t.cached.aerosol_scattering || p.aerosol_extinction != t.cached.aerosol_extinction ||
            p.aerosol_phase != t.cached.aerosol_phase;
        if (changes.optics) {
            t.stellar_source_resolved = stellar_source_resolved(p);
            t.point_valid = false;
        }
        float stellar_radius = p.sun_direction[0].w;
        changes.finite_radius = stellar_radius >= 0.01f
            ? std::clamp(std::exp2(std::round(std::log2(stellar_radius)*512.0f)/512.0f), 0.01f, float(PI*0.5)) : 0.0f;
        changes.scattering = p.aerosol_count.y > 0.5f || glm::any(glm::greaterThan(glm::vec3(p.rayleigh), glm::vec3(0.0f)));
        p.sun_irradiance[0].w = changes.scattering && t.stellar_source_resolved &&
            glm::any(glm::greaterThan(glm::vec3(p.sun_irradiance[0]), glm::vec3(0.0f))) ? changes.finite_radius : 0.0f;
        uint32_t shadow_changes = 0;
        double refraction_pad = refraction && refraction->refraction && int(candidate.index) != refraction->body
            ? refraction->max_displacement : 0.0;
        t.shadow_active = prepare_shadows(ctx, t.shadows[ctx.frame_index()], system, candidate.index, eye, frame.radius+frame.height,
                        float(frame.bottom*frame.polar_ratio), float(focal), camera, candidate.apparent, extent, refraction_pad,
                        settings, ring_shadow_lut ? rings : std::span<const RingHostGPU>{},
                        t.shadow_header, t.shadow_casters, uint32_t(prepare_serial_), shadow_changes,
                        ring_shadow_lut && ring_lighting ? &ring_lighting->penumbra_slots() : nullptr);
        t.shadow_active = t.shadow_active && p.depth_params.w > 0.0f;
        auto& uniform = t.uniforms[ctx.frame_index()];
        std::memcpy(uniform.mapped, &p, sizeof(p));
        ctx.flush_buffer(uniform);
        changes.identity = t.identity != b.revisions.identity || t.definition_revision != b.revisions.definition ||
            t.geometry_revision != b.revisions.geometry || t.lighting_revision != b.revisions.lighting;
        update_shadow_history(t, frame, eye-b.render.position.value, changes.identity, shadow_changes, ring_shadow_revision, settings);
        changes.lighting_mode = p.depth_params.z != t.cached.depth_params.z;
        changes.sky = t.cached.depth_params.w == 0.0f || changes.optics || changes.lighting_mode || p.camera != t.cached.camera || p.sun_direction != t.cached.sun_direction || p.sun_irradiance != t.cached.sun_irradiance;
        changes.aerial = !t.aerial_valid || changes.sky;
        changes.small_stars = std::all_of(p.sun_direction.begin(), p.sun_direction.end(),
            [](const glm::vec4& sun) { return sun.w < 0.01f; });
        if (changes.aerial) t.aerial_valid = false;
        bake_luts(lut, t, frame, changes, system, candidate.index, surface_slots[i], settings);
        bake_secondary(lut, t, frame, changes, system, candidate.index, settings, planetshine, ring_lighting);
        t.cached = p;
        t.identity = b.revisions.identity;
        t.definition_revision = b.revisions.definition;
        t.body_index = candidate.index;
        t.geometry_revision = b.revisions.geometry;
        t.lighting_revision = b.revisions.lighting;
        t.valid = true;
        t.last_used = prepare_serial_;
        active_.push_back(&t);
    }
    ctx.end_gpu_timer(cmd, timer);
    for (auto& table : tables_)
        if (table->last_used != prepare_serial_ || !table->shadow_active) retire_shadow_history(ctx, *table);
    ctx.flush_buffer(surface_records_[ctx.frame_index()]);
    stats_.visible = static_cast<uint32_t>(active_.size());
}

bool AtmosphereRenderer::composites_clouds(uint32_t body) const noexcept {
    return std::any_of(active_.begin(), active_.end(), [&](const Tables* t) {
        return t->body_index == body && t->cached.depth_params.w > 0.0f && t->cached.cloud_layer.y > 0.5f;
    });
}

bool AtmosphereRenderer::begin_prepare(rhi::VkContext& ctx, const ephemeris::SystemSnapshot& system, const RenderSettings& settings,
                                       VkExtent2D extent, const rhi::Image* ring_shadow_lut, bool shadow_diagnostics) {
    active_.clear();
    ++prepare_serial_;
    std::erase_if(retired_images_, [&](auto& entry) {
        if (entry.serial > ctx.completed_serial()) return false;
        ctx.destroy_image(entry.image);
        return true;
    });
    ring_shadow_lut_ = ring_shadow_lut ? ring_shadow_lut : &surface_lut_;
    shadow_last_diagnostics_ = shadow_diagnostics ? shadow_history_diagnostics(ctx) : ShadowHistoryDiagnostics{};
    stats_ = {};
    shadow_diagnostics_enabled_ = shadow_diagnostics;
    auto& diagnostics = shadow_diagnostics_[ctx.frame_index()];
    diagnostics.serial = 0;
    diagnostics.reset_reasons = 0;
    if (shadow_diagnostics) {
        const std::array<uint32_t, 8> values{0, 0, 0, 0, 64, 0, 0, 0};
        std::memcpy(diagnostics.counters.mapped, values.data(), sizeof(values));
        ctx.flush_buffer(diagnostics.counters);
    }
    shadow_vrs_enabled_ = settings.atmosphere_shadow_vrs;
    shadow_temporal_enabled_ = settings.atmosphere_shadow_temporal && !settings.atmosphere_shadow_reference;
    bool rotation_extent_changed = shadow_reproject_energy_ &&
        (shadow_reproject_energy_.extent.width != extent.width || shadow_reproject_energy_.extent.height != extent.height);
    if (!shadow_temporal_enabled_ || !settings.atmosphere_shadow_rotation || !settings.show_atmospheres || system.empty() || rotation_extent_changed) {
        for (auto* image : {&shadow_reproject_energy_, &shadow_reproject_loss_}) {
            if (*image) retired_images_.push_back({*image, ctx.recording_serial()});
            *image = {};
        }
    }
    dual_source_ = settings.atmosphere_dual_source && bool(dual_source_pipeline_);
    auto& point_records = point_responses_[ctx.frame_index()];
    std::memset(point_records.mapped, 0, point_records.size);
    ctx.flush_buffer(point_records);
    auto& records = surface_records_[ctx.frame_index()];
    std::memset(records.mapped, 0, 1024*sizeof(glm::vec4));
    ctx.flush_buffer(records);
    if (!settings.show_atmospheres || system.empty()) {
        for (auto& table : tables_) retire_shadow_history(ctx, *table);
        return false;
    }
    return true;
}

std::vector<AtmosphereRenderer::Candidate> AtmosphereRenderer::select_candidates(const Camera& camera, const ephemeris::SystemSnapshot& system,
                                                                             VkExtent2D extent, const RefractionHost* refraction) const {
    std::vector<Candidate> candidates;
    double tan_y = std::tan(camera.fov_deg*DEG_TO_RAD*0.5);
    double tan_x = tan_y*double(extent.width)/std::max(1u, extent.height);
    glm::dvec3 eye = camera.eye_pos_render().value;
    auto outside = [&](glm::dvec3 to_body, double outer, double tx, double ty) {
        double depth = glm::dot(to_body, camera.forward_dir());
        return depth+outer <= 0.0 ||
            std::abs(glm::dot(to_body, camera.right_dir())) > depth*tx+outer*std::sqrt(1.0+tx*tx) ||
            std::abs(glm::dot(to_body, camera.up_dir())) > depth*ty+outer*std::sqrt(1.0+ty*ty);
    };
    double pad = refraction && refraction->refraction ? refraction->max_displacement : 0.0;
    auto padded = [&](double tangent) { return std::tan(std::min(std::atan(tangent)+pad, 1.5)); };
    for (uint32_t i = 0; i < system.size(); ++i) {
        const auto b = system.body(i);
        const auto& a = b.definition.atmosphere;
        if (!a.enabled || b.definition.is_star || b.definition.is_black_hole || b.definition.is_spacecraft) continue;
        double radius = b.definition.radius_au()*AU_TO_KM;
        glm::dvec3 to_body = (b.render.position.value-eye)*AU_TO_KM;
        double distance = glm::length(to_body);
        double height = b.render.atmosphere ? b.render.atmosphere->height_km : a.height_km;
        double outer = radius+height;
        glm::dmat3 apparent(1.0);
        if (distance > outer) {
            bool refracted = pad > 0.0 && int(i) != refraction->body;
            if (outside(to_body, outer, refracted ? padded(tan_x) : tan_x, refracted ? padded(tan_y) : tan_y)) continue;
            if (refracted) {
                apparent = apparent_rotation(*refraction, b.render.position.value-eye);
                if (outside(apparent*to_body, outer, tan_x, tan_y)) continue;
            }
        }
        if (i >= 1024) continue;
        candidates.push_back({i, distance, apparent});
    }
    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.distance > b.distance; });
    return candidates;
}

void AtmosphereRenderer::assign_tables(std::span<const Candidate> candidates, const ephemeris::SystemSnapshot& system,
                                       std::vector<size_t>& slots, std::vector<size_t>& surface_slots) const {
    constexpr size_t retained_budget = 256 * 1024 * 1024;
    constexpr size_t table_bytes = 256*64*16 + 32*32*8 + 2*192*108*8 +
        2*192*108*64*8 + 192*108*4 + sizeof(AtmosphereParametersGPU)*rhi::FRAMES_IN_FLIGHT;
    size_t source_bytes = 0;
    for (const auto& table : tables_) {
        if (table->stellar_source) source_bytes += 128*64*table->stellar_source.layers*8;
        if (table->secondary_sky) source_bytes += 96*54*(65*8+4);
        if (table->secondary_source) source_bytes += 128*64*5*8;
    }
    const size_t capacity = std::max(candidates.size(), (retained_budget-std::min(source_bytes, retained_budget))/table_bytes);
    std::unordered_set<uint64_t> live;
    live.reserve(system.size());
    for (uint32_t i = 0; i < system.size(); ++i) live.insert(system.body(i).revisions.identity);
    slots.assign(candidates.size(), SIZE_MAX);
    std::vector<bool> reserved(tables_.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = 0; j < tables_.size(); ++j) {
            const auto& t = *tables_[j];
            if (t.valid && t.identity == system.body(candidates[i].index).revisions.identity) {
                slots[i] = j;
                reserved[j] = true;
                break;
            }
        }
    }
    for (auto& slot : slots) {
        if (slot != SIZE_MAX) continue;
        size_t available = SIZE_MAX;
        for (size_t j = 0; j < tables_.size(); ++j) {
            if (reserved[j]) continue;
            const auto& t = *tables_[j];
            bool stale = !t.valid || !live.contains(t.identity);
            if (stale) { available = j; break; }
            if (available == SIZE_MAX || t.last_used < tables_[available]->last_used) available = j;
        }
        bool stale = available != SIZE_MAX && (!tables_[available]->valid || !live.contains(tables_[available]->identity));
        if (!stale && reserved.size() < capacity) {
            slot = reserved.size();
            reserved.push_back(true);
        } else {
            slot = available;
            reserved[slot] = true;
        }
    }
    surface_slots.assign(candidates.size(), SIZE_MAX);
    std::array<bool, 32> surface_reserved{};
    for (size_t i = 0; i < candidates.size() && i < surface_layers_.size(); ++i) {
        if (slots[i] >= tables_.size()) continue;
        for (size_t j = 0; j < surface_layers_.size(); ++j) {
            if (surface_layers_[j] == tables_[slots[i]].get()) {
                surface_slots[i] = j;
                surface_reserved[j] = true;
                break;
            }
        }
    }
    for (size_t i = 0; i < candidates.size() && i < surface_layers_.size(); ++i) {
        if (surface_slots[i] != SIZE_MAX) continue;
        auto layer = std::find(surface_reserved.begin(), surface_reserved.end(), false);
        surface_slots[i] = size_t(layer-surface_reserved.begin());
        *layer = true;
    }
}

void AtmosphereRenderer::update_shadow_history(Tables& t, const AtmosphereFrame& frame, glm::dvec3 relative_eye, bool identity_dirty,
                                               uint32_t shadow_changes, uint64_t ring_shadow_revision, const RenderSettings& settings) {
    const auto& p = frame.p;
    const auto& view = frame.view;
    const auto& projection = frame.projection;
    t.shadow_reset_reasons = shadow_changes;
    if (!t.shadow_history_valid) t.shadow_reset_reasons |= ShadowHistoryInitial;
    if (identity_dirty) t.shadow_reset_reasons |= ShadowHistoryDefinition;
    if (t.shadow_recorded_prepare+1 != prepare_serial_) t.shadow_reset_reasons |= ShadowHistoryInterrupted;
    if (view != t.shadow_view || glm::vec3(p.camera) != glm::vec3(t.cached.camera) ||
        p.axis_x != t.cached.axis_x || p.axis_y != t.cached.axis_y || p.axis_z != t.cached.axis_z)
        t.shadow_reset_reasons |= ShadowHistoryCamera;
    if (projection != t.shadow_projection) t.shadow_reset_reasons |= ShadowHistoryProjection;
    if (p.camera.w != t.cached.camera.w || p.sun_direction != t.cached.sun_direction || p.sun_irradiance != t.cached.sun_irradiance)
        t.shadow_reset_reasons |= ShadowHistoryLighting;
    if (p.depth_params != t.cached.depth_params) t.shadow_reset_reasons |= ShadowHistoryTransport;
    auto atmosphere_key = t.cached;
    atmosphere_key.camera = p.camera;
    atmosphere_key.axis_x = p.axis_x; atmosphere_key.axis_y = p.axis_y; atmosphere_key.axis_z = p.axis_z;
    atmosphere_key.sun_direction = p.sun_direction;
    atmosphere_key.star_disc = p.star_disc;
    atmosphere_key.sun_irradiance = p.sun_irradiance;
    atmosphere_key.inverse_view_projection = p.inverse_view_projection;
    atmosphere_key.view_ray = p.view_ray;
    atmosphere_key.depth_params = p.depth_params;
    if (std::memcmp(&p, &atmosphere_key, sizeof(p)) != 0) t.shadow_reset_reasons |= ShadowHistoryAtmosphere;
    if (t.shadow_ring_revision != ring_shadow_revision || t.shadow_ring_image != ring_shadow_lut_->handle)
        t.shadow_reset_reasons |= ShadowHistoryRings;
    if (t.shadow_reset_reasons == 0 && std::memcmp(&p, &t.cached, sizeof(p)) != 0)
        t.shadow_reset_reasons |= ShadowHistoryTransport;
    bool rotation_mode_unchanged = t.shadow_rotation_enabled == settings.atmosphere_shadow_rotation;
    if (!rotation_mode_unchanged) t.shadow_reset_reasons |= ShadowHistorySampling;
    if (relative_eye != t.shadow_eye_relative) t.shadow_reset_reasons |= ShadowHistoryCamera;
    t.shadow_history_reusable = shadow_temporal_enabled_ && t.shadow_history_valid && !identity_dirty &&
        t.shadow_recorded_prepare+1 == prepare_serial_ && shadow_changes == 0 &&
        std::memcmp(&p, &t.cached, sizeof(p)) == 0 &&
        t.shadow_ring_revision == ring_shadow_revision && t.shadow_ring_image == ring_shadow_lut_->handle && rotation_mode_unchanged &&
        relative_eye == t.shadow_eye_relative;
    auto rotation_key = t.cached;
    rotation_key.inverse_view_projection = p.inverse_view_projection;
    rotation_key.view_ray = p.view_ray;
    t.shadow_history_reproject = shadow_temporal_enabled_ && settings.atmosphere_shadow_rotation &&
        p.camera.w == 1.0f && !t.shadow_history_reusable && t.shadow_history_valid && !identity_dirty && rotation_mode_unchanged &&
        t.shadow_recorded_prepare+1 == prepare_serial_ && shadow_changes == 0 &&
        relative_eye == t.shadow_eye_relative && view != t.shadow_view && projection == t.shadow_projection &&
        std::memcmp(&p, &rotation_key, sizeof(p)) == 0 &&
        t.shadow_ring_revision == ring_shadow_revision && t.shadow_ring_image == ring_shadow_lut_->handle;
    t.shadow_reprojection = t.shadow_projection*t.shadow_view*p.inverse_view_projection;
    t.shadow_previous_inverse = t.cached.inverse_view_projection;
    t.shadow_eye_relative = relative_eye;
    t.shadow_rotation_enabled = settings.atmosphere_shadow_rotation;
    t.shadow_ring_revision = ring_shadow_revision;
    t.shadow_ring_image = ring_shadow_lut_->handle;
    t.shadow_view = view;
    t.shadow_projection = projection;
}

void AtmosphereRenderer::retire_shadow_history(rhi::VkContext& ctx, Tables& target) {
    auto* table = &target;
    for (auto* image : {&table->shadow_fraction, &table->shadow_depth, &table->shadow_energy, &table->shadow_loss}) {
        if (*image) retired_images_.push_back({*image, ctx.recording_serial()});
        *image = {};
    }
    table->shadow_history_valid = false;
}

}
