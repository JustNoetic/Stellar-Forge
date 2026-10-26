#include "rendering/size_comparator.hpp"
#include "rendering/ring_optics.hpp"
#include "core/path_utils.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace sf::rendering {

void SizeComparatorRenderer::init(rhi::VkContext& ctx, bool reference_atmosphere) {
    reference_atmosphere_ = reference_atmosphere;
    textures_.init(ctx);
    normals_.init(ctx, true);
    for (const auto& [body, path] : textures_.load_catalog(paths::external_path("data", "surfaces.json")))
        normals_.set_height_source(body, terrain::HeightSource::open(path));
    dummy_image_ = ctx.create_image_2d(1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    glm::vec4 empty{};
    ctx.upload_image_2d(dummy_image_, &empty, sizeof(empty));
    rhi::GraphicsPipelineDesc desc;
    desc.vert_path = "celestial/size_comparison.vert";
    desc.frag_path = "celestial/size_comparison.frag";
    desc.push_constant_size = sizeof(Push);
    desc.cull_mode = VK_CULL_MODE_NONE;
    desc.bindings = {
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT},
        {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}};
    surface_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    constexpr auto compute = VK_SHADER_STAGE_COMPUTE_BIT;
    const rhi::DescriptorLayoutBinding overlay_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, compute},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, compute},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, compute},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, compute},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, compute},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, compute},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, compute},
        {13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, compute}};
    std::vector<std::pair<std::string,std::string>> macros;
    if (reference_atmosphere_) macros.emplace_back("COMPARISON_REFERENCE_AIR","1");
    overlay_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/size_comparison.comp", overlay_bindings, sizeof(Push),macros);
    const rhi::DescriptorLayoutBinding trans_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, compute}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, compute}};
    const rhi::DescriptorLayoutBinding multi_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, compute}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, compute},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, compute}};
    transmittance_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/transmittance.comp", trans_bindings);
    multiscatter_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/multiscatter.comp", multi_bindings);
    const rhi::DescriptorLayoutBinding transport_bindings[] = {
        {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,compute}, {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,compute},
        {2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,compute}, {3,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,compute},
        {4,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,compute}};
    transport_ = rhi::ShaderCache::create_compute_pipeline(ctx,"atmosphere/size_comparison_transport.comp",transport_bindings,sizeof(Push));
    initialized_ = true;
}

void SizeComparatorRenderer::destroy(rhi::VkContext& ctx, Resources& r) {
    for (auto& uniform : r.uniforms) ctx.destroy_buffer(uniform);
    ctx.destroy_buffer(r.profiles);
    ctx.destroy_image(r.transmittance);
    ctx.destroy_image(r.multiscatter);
    ctx.destroy_image(r.transport_radiance);
    ctx.destroy_image(r.transport_throughput);
}

void SizeComparatorRenderer::shutdown(rhi::VkContext& ctx) {
    if (!initialized_) return;
    for (auto& [name, r] : resources_) destroy(ctx, *r);
    for (auto& r : retired_) destroy(ctx, *r.resources);
    resources_.clear();
    retired_.clear();
    draws_.clear();
    textures_.shutdown(ctx);
    normals_.shutdown(ctx);
    ctx.destroy_image(dummy_image_);
    for (auto* pipeline : {&surface_, &overlay_, &transmittance_, &multiscatter_, &transport_})
        rhi::ShaderCache::destroy_pipeline(ctx, *pipeline);
    initialized_ = false;
}

void SizeComparatorRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const comparison::SizeComparison& scene,
    const comparison::SizeView& view, const RenderSettings& settings, VkExtent2D extent) {
    if (!initialized_) init(ctx);
    ++serial_;
    draws_.clear();
    std::erase_if(retired_, [&](auto& retired) {
        if (retired.serial > ctx.completed_serial()) return false;
        destroy(ctx, *retired.resources);
        return true;
    });
    double pixels = extent.height/view.height;
    double depth_km = 1;
    for (const auto& b : scene.bodies) depth_km = std::max(depth_km,b.outer/comparison::SIZE_SCALE*4);
    std::map<int, terrain::SurfaceTextureView> colors, normals;
    auto texture_view = [&](const comparison::SizeBody& b, terrain::SurfaceTextures& textures, auto& inputs) {
        int source = textures.source_index(b.definition());
        if (source < 0) return source;
        double radius_pixels = b.radius*pixels;
        auto existing = inputs.find(source);
        if (existing != inputs.end() && existing->second.radius_pixels >= radius_pixels) return source;
        auto rotation = terrain::body_rotation(comparison::DISPLAY_POLE,b.spin);
        terrain::SurfaceTextureView input;
        input.definition = &b.definition();
        input.index = uint32_t(source);
        input.radius_pixels = radius_pixels;
        input.view.eye = glm::transpose(rotation)*glm::dvec3(0,0,10000);
        input.view.view_from_local = rotation;
        input.view.orthographic_pixels = radius_pixels;
        auto center = view.world_to_screen(b.position,{extent.width,extent.height});
        input.view.orthographic_bounds = {-center.x/radius_pixels,(center.y-extent.height)/radius_pixels,
            (extent.width-center.x)/radius_pixels,center.y/radius_pixels};
        inputs[source] = input;
        return source;
    };
    for (const auto& b : scene.bodies) {
        const auto& def = b.definition();
        double radius_km = b.radius/comparison::SIZE_SCALE;
        double opaque_km = radius_km;
        if (def.atmosphere.pressure_reference)
            opaque_km += b.atmosphere ? b.atmosphere->opaque_altitude_km : def.atmosphere.bottom_altitude_km;
        bool air = settings.show_atmospheres && def.atmosphere.enabled && !def.is_star && !def.is_black_hole &&
            (!def.atmosphere.physical || def.atmosphere.pressure_pa > 0);
        double top_km = radius_km+(air ? b.atmosphere ? b.atmosphere->height_km : def.atmosphere.height_km : 0);
        double outer_km = settings.show_rings ? b.outer/comparison::SIZE_SCALE : radius_km;
        double half_extent = std::max({opaque_km,top_km,outer_km});
        auto screen = view.world_to_screen(b.position, {extent.width,extent.height});
        double extent_px = half_extent*comparison::SIZE_SCALE*pixels;
        if (b.radius*pixels < .1 || screen.x+extent_px < 0 || screen.y+extent_px < 0 ||
            screen.x-extent_px > extent.width || screen.y-extent_px > extent.height || radius_km <= 0) continue;
        int color_index = settings.surface_textures_enabled && !def.is_star ? texture_view(b,textures_,colors) : -1;
        int normal_index = settings.terrain_normal_maps_enabled && settings.terrain_imported_enabled && !def.is_star ?
            texture_view(b,normals_,normals) : -1;
        auto key = b.system->name+"\x1f"+def.name;
        auto& slot = resources_[key];
        if (slot && slot->definition != b.system) retired_.push_back({std::move(slot),ctx.recording_serial()});
        if (!slot) {
            slot = std::make_unique<Resources>();
            slot->definition = b.system;
            slot->body_index = b.index;
            std::vector<glm::vec4> profiles(def.rings.size());
            std::vector<glm::vec4> raw;
            if (auto path = ephemeris::find_ring_texture_path(b.system->name,def.name))
                if (auto decoded = load_ring_png_1d(*path)) raw = std::move(*decoded);
            for (size_t i = 0; i < def.rings.size(); ++i) {
                const auto& layer = def.rings[i];
                auto profile = generate_ring_segment_profile(layer,raw);
                for (auto& sample : profile) sample.a = -std::expm1(-std::max(0.0f,sample.a*layer.opacity)/.2f);
                profiles[i] = {float(layer.inner*radius_km),float(layer.outer*radius_km),float(profiles.size()),0};
                auto mips = build_ring_profile_mips(profile);
                for (const auto& mip : mips) profiles.insert(profiles.end(),mip.begin(),mip.end());
            }
            if (profiles.empty()) profiles.emplace_back(0);
            slot->profiles = ctx.create_host_buffer(profiles.size()*sizeof(glm::vec4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,profiles.data());
            ctx.flush_buffer(slot->profiles);
            for (auto& uniform : slot->uniforms)
                uniform = ctx.create_host_buffer(sizeof(AtmosphereParametersGPU),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        }
        auto& r = *slot;
        r.used = serial_;
        auto parameters = make_atmosphere_parameters(def,b.atmosphere,glm::normalize(comparison::DISPLAY_POLE));
        auto& uniform = r.uniforms[ctx.frame_index()];
        std::memcpy(uniform.mapped,&parameters,sizeof(parameters));
        ctx.flush_buffer(uniform);
        if (air && !r.optics) {
            auto image = [&](uint32_t w,uint32_t h,VkFormat format) {
                return ctx.create_image_2d(w,h,format,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);
            };
            r.transmittance = image(256,64,VK_FORMAT_R32G32B32A32_SFLOAT);
            r.multiscatter = image(32,32,VK_FORMAT_R16G16B16A16_SFLOAT);
            ctx.transition_image(cmd,r.transmittance,VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,transmittance_.handle);
            ctx.push_descriptors(cmd,transmittance_,{
                rhi::PushDescriptorWrite::uniform_buffer(0,uniform),rhi::PushDescriptorWrite::storage_image(1,r.transmittance)});
            vkCmdDispatch(cmd,32,8,1);
            ctx.transition_image(cmd,r.transmittance,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            ctx.transition_image(cmd,r.multiscatter,VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,multiscatter_.handle);
            ctx.push_descriptors(cmd,multiscatter_,{
                rhi::PushDescriptorWrite::uniform_buffer(0,uniform),
                rhi::PushDescriptorWrite::sampled_image(1,r.transmittance,ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::storage_image(2,r.multiscatter)});
            vkCmdDispatch(cmd,32,32,1);
            ctx.transition_image(cmd,r.multiscatter,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            r.optics = true;
        }
        Push surface;
        surface.rect = {float(2*screen.x/extent.width-1),float(2*screen.y/extent.height-1),
            float(2*extent_px/extent.width),float(-2*extent_px/extent.height)};
        surface.shape = {float(radius_km),float(std::clamp(1-def.oblateness,.05,1.0)),float(b.spin),float(half_extent)};
        auto color = def.is_star && def.temp_k > 0 ? ephemeris::temperature_to_rgb(def.temp_k) : def.color_rgb;
        surface.color = {color,def.is_black_hole ? 2.0f : def.is_star ? 1.0f : 0.0f};
        surface.resources = {float(color_index),float(normal_index),settings.show_rings ? float(def.rings.size()) : 0.0f,air ? 48.0f : 0.0f};
        surface.display = {float(opaque_km),float(depth_km),float(1/(pixels*comparison::SIZE_SCALE)),0};
        if (air && !reference_atmosphere_ && !r.transport_radiance) {
            auto image = [&] {
                return ctx.create_image_2d(512,257,VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT,1,2);
            };
            r.transport_radiance = image();
            r.transport_throughput = image();
            ctx.transition_image(cmd,r.transport_radiance,VK_IMAGE_LAYOUT_GENERAL);
            ctx.transition_image(cmd,r.transport_throughput,VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,transport_.handle);
            ctx.push_descriptors(cmd,transport_,{
                rhi::PushDescriptorWrite::uniform_buffer(0,uniform),
                rhi::PushDescriptorWrite::sampled_image(1,r.transmittance,ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::sampled_image(2,r.multiscatter,ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::storage_image(3,r.transport_radiance),
                rhi::PushDescriptorWrite::storage_image(4,r.transport_throughput)});
            auto transport_push = surface;
            transport_push.resources.z = float(def.rings.size());
            ctx.push_constants(cmd,transport_,transport_push);
            vkCmdDispatch(cmd,64,33,1);
            for (auto* lut : {&r.transport_radiance,&r.transport_throughput})
                ctx.transition_image(cmd,*lut,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            ++transport_bakes_;
        }
        Push overlay = surface;
        overlay.rect = {float(std::max(0.0,std::floor(screen.x-extent_px-1))),float(std::max(0.0,std::floor(screen.y-extent_px-1))),
            float(std::min(double(extent.width),std::ceil(screen.x+extent_px+1))),float(std::min(double(extent.height),std::ceil(screen.y+extent_px+1)))};
        overlay.shape.x = float(half_extent);
        overlay.shape.z = float(screen.x);
        overlay.shape.w = float(screen.y);
        draws_.push_back({&r,surface,overlay});
    }
    auto prepare_textures = [&](terrain::SurfaceTextures& textures,const auto& input,bool enabled) {
        std::vector<terrain::SurfaceTextureView> views;
        for (const auto& [index,entry] : input) views.push_back(entry);
        textures.prepare_views(cmd,ctx,views,enabled);
    };
    prepare_textures(textures_,colors,settings.surface_textures_enabled);
    prepare_textures(normals_,normals,settings.terrain_normal_maps_enabled && settings.terrain_imported_enabled);
    std::erase_if(resources_,[&](auto& item) {
        if (item.second->used == serial_) return false;
        retired_.push_back({std::move(item.second),ctx.recording_serial()});
        return true;
    });
}

void SizeComparatorRenderer::render(VkCommandBuffer cmd,rhi::VkContext& ctx) {
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,surface_.handle);
    ctx.push_descriptors(cmd,surface_,{
        rhi::PushDescriptorWrite::sampled_image(5,textures_.image(),ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(6,textures_.pages(ctx.frame_index())),
        rhi::PushDescriptorWrite::sampled_image(10,normals_.image(),ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(11,normals_.pages(ctx.frame_index()))});
    for (const auto& draw : draws_) {
        ctx.push_constants(cmd,surface_,draw.surface);
        vkCmdDraw(cmd,6,1,0,0);
    }
}

void SizeComparatorRenderer::composite(VkCommandBuffer cmd,rhi::VkContext& ctx,rhi::Image& color,rhi::Image& depth) {
    ctx.transition_image(cmd,color,VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_READ_BIT|VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    ctx.transition_image(cmd,depth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,overlay_.handle);
    for (const auto& draw : draws_) {
        if (draw.overlay.resources.z == 0 && draw.overlay.resources.w == 0) continue;
        const auto& r = *draw.resources;
        ctx.push_descriptors(cmd,overlay_,{
            rhi::PushDescriptorWrite::uniform_buffer(0,r.uniforms[ctx.frame_index()]),
            rhi::PushDescriptorWrite::sampled_image(1,r.optics ? r.transmittance : dummy_image_,ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(2,r.optics ? r.multiscatter : dummy_image_,ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::storage_image(4,color),rhi::PushDescriptorWrite::storage_buffer(8,r.profiles),
            rhi::PushDescriptorWrite::sampled_image(9,depth,ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(12,r.transport_radiance ? r.transport_radiance : textures_.image(),ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(13,r.transport_throughput ? r.transport_throughput : textures_.image(),ctx.sampler_linear_clamp())});
        ctx.push_constants(cmd,overlay_,draw.overlay);
        vkCmdDispatch(cmd,uint32_t(draw.overlay.rect.z-draw.overlay.rect.x+7)/8,
            uint32_t(draw.overlay.rect.w-draw.overlay.rect.y+7)/8,1);
        ctx.memory_barrier(cmd,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_READ_BIT|VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
}

}
